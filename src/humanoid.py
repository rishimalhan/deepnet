#!/usr/bin/env python3
# PPO for MuJoCo Humanoid with MPS + checkpointing every X steps.
# Optimized for maximum GPU utilization with vectorized environments.

import os, math, time, random, argparse, imageio, glob
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv


# ---------------------------
# Device
# ---------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = get_device()


# ---------------------------
# Actor-Critic
# ---------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(512, 512), log_std_init=-0.5):
        super().__init__()
        self.pi_body = self._mlp(obs_dim, hidden, last_act=nn.Tanh())
        self.mu_head = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)
        self.v_body = self._mlp(obs_dim, hidden, last_act=nn.Tanh())
        self.v_head = nn.Linear(hidden[-1], 1)

    def _mlp(self, in_dim, hidden, last_act=None):
        layers = []
        dims = (in_dim, *hidden)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        if last_act is not None:
            layers[-1] = last_act
        return nn.Sequential(*layers)

    def pi(self, obs):
        x = self.pi_body(obs)
        mu = self.mu_head(x)
        # Clamp mu to prevent extreme actions
        mu = torch.clamp(mu, -2.0, 2.0)
        std = torch.exp(self.log_std)
        # Clamp std to prevent too small or too large values
        std = torch.clamp(std, 1e-4, 2.0)
        return mu, std

    def v(self, obs):
        x = self.v_body(obs)
        v = self.v_head(x).squeeze(-1)
        # Clamp value predictions to reasonable range
        v = torch.clamp(v, -100.0, 100.0)
        return v

    def act(self, obs):
        with torch.no_grad():
            mu, std = self.pi(obs)
            dist = torch.distributions.Normal(mu, std)
            a = dist.sample()
            # Clamp actions to valid range [-1, 1] for Humanoid
            a = torch.clamp(a, -1.0, 1.0)
            logp = dist.log_prob(a).sum(-1)
            return a, logp

    def log_prob(self, obs, act):
        mu, std = self.pi(obs)
        # DO NOT clamp actions here - it breaks PPO log-prob computation
        # Actions are already in valid range from environment
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(act).sum(-1)
        ent = dist.entropy().sum(-1)
        # Clamp outputs to prevent overflow
        logp = torch.clamp(logp, -50.0, 50.0)
        ent = torch.clamp(ent, 0.0, 100.0)
        return logp, ent


# ---------------------------
# Rollout Buffer with GAE
# ---------------------------
@dataclass
class PPOConfig:
    env_id: str = "Humanoid-v4"
    total_steps: int = 2_000_000
    rollout_steps: int = 8192
    minibatch_size: int = 4096
    update_epochs: int = 10
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01  # CRITICAL: Enable exploration!
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    seed: int = 42
    render_mode: str = "human"  # "human" or "rgb_array"
    eval_episodes: int = 3
    video_out: str = "humanoid_walk.mp4"
    ckpt_dir: str = "checkpoints"
    save_every: int = 100_000  # save every N env steps
    num_envs: int = 8  # number of parallel environments for vectorization
    normalize_obs: bool = True  # Normalize observations (critical for MuJoCo)
    normalize_returns: bool = True  # Normalize returns for stability


class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, device, normalize_obs=False, num_envs=1):
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.acts = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.logp = torch.zeros(size, dtype=torch.float32, device=device)
        self.rew = torch.zeros(size, dtype=torch.float32, device=device)
        self.val = torch.zeros(size, dtype=torch.float32, device=device)
        self.done = torch.zeros(size, dtype=torch.float32, device=device)
        self.adv = torch.zeros(size, dtype=torch.float32, device=device)
        self.ret = torch.zeros(size, dtype=torch.float32, device=device)
        self.env_id = torch.zeros(
            size, dtype=torch.long, device=device
        )  # Track which env each step belongs to
        self.ptr = 0
        self.max_size = size
        self.device = device
        self.normalize_obs = normalize_obs
        self.num_envs = num_envs
        # Running stats for observation normalization
        if normalize_obs:
            self.obs_mean = torch.zeros(obs_dim, dtype=torch.float32, device=device)
            self.obs_var = torch.ones(obs_dim, dtype=torch.float32, device=device)
            self.obs_count = 1e-4  # Start with small count to avoid division issues

    def add(self, o, a, logp, r, v, d, env_ids=None):
        # Handle both single and batched inputs
        batch_size = o.shape[0] if o.ndim > 1 else 1
        if batch_size == 1:
            idx = self.ptr
            self.obs[idx] = o.squeeze(0) if o.ndim > 1 else o
            self.acts[idx] = a.squeeze(0) if a.ndim > 1 else a
            self.logp[idx] = logp.squeeze(0) if logp.ndim > 0 else logp
            self.rew[idx] = (
                r.squeeze(0) if isinstance(r, torch.Tensor) and r.ndim > 0 else r
            )
            self.val[idx] = v.squeeze(0) if v.ndim > 0 else v
            self.done[idx] = (
                d.squeeze(0) if isinstance(d, torch.Tensor) and d.ndim > 0 else d
            )
            self.env_id[idx] = env_ids[0] if env_ids is not None else 0
            self.ptr += 1
        else:
            # Batch add
            end = min(self.ptr + batch_size, self.max_size)
            actual_batch = end - self.ptr
            self.obs[self.ptr : end] = o[:actual_batch]
            self.acts[self.ptr : end] = a[:actual_batch]
            self.logp[self.ptr : end] = logp[:actual_batch]
            self.rew[self.ptr : end] = (
                r[:actual_batch]
                if isinstance(r, torch.Tensor)
                else torch.tensor(r[:actual_batch], device=self.device)
            )
            self.val[self.ptr : end] = v[:actual_batch]
            self.done[self.ptr : end] = (
                d[:actual_batch]
                if isinstance(d, torch.Tensor)
                else torch.tensor(d[:actual_batch], device=self.device)
            )
            if env_ids is not None:
                self.env_id[self.ptr : end] = torch.tensor(
                    env_ids[:actual_batch], dtype=torch.long, device=self.device
                )
            else:
                # Default: assign sequentially (for single env)
                self.env_id[self.ptr : end] = (
                    torch.arange(self.ptr, end, dtype=torch.long, device=self.device)
                    % self.num_envs
                )
            self.ptr = end

    def update_obs_stats(self, obs_batch):
        """Update running statistics for observation normalization using Welford's algorithm."""
        if not self.normalize_obs:
            return

        # Welford's online algorithm for numerical stability
        batch_size = obs_batch.shape[0]
        for i in range(batch_size):
            obs = obs_batch[i]
            self.obs_count += 1
            delta = obs - self.obs_mean
            self.obs_mean += delta / self.obs_count
            delta2 = obs - self.obs_mean
            self.obs_var += delta * delta2

    def normalize_observations(self, obs):
        """Normalize observations using running statistics."""
        if not self.normalize_obs or self.obs_count < 2:
            return obs
        obs_std = torch.sqrt(self.obs_var / self.obs_count + 1e-8)
        normalized = (obs - self.obs_mean) / obs_std
        # Clip to prevent extreme values
        normalized = torch.clamp(normalized, -10.0, 10.0)
        return normalized

    def compute_gae(self, last_vals, gamma, lam, normalize_returns=True):
        """
        Compute GAE with proper per-environment trajectory handling.
        last_vals: tensor of shape (num_envs,) with bootstrap values for each env
        """
        # Initialize advantages per environment
        adv_per_env = {env_id: 0.0 for env_id in range(self.num_envs)}

        # Process backwards, maintaining separate advantage for each env
        for t in reversed(range(self.ptr)):
            env_id = self.env_id[t].item()
            nonterminal = 1.0 - self.done[t]

            # Get next value: either from next step (same env) or bootstrap value
            if t == self.ptr - 1:
                # Last step: use bootstrap value for this env
                next_val = (
                    last_vals[env_id]
                    if isinstance(last_vals, torch.Tensor)
                    else last_vals
                )
            else:
                # Check if next step belongs to same env
                next_env_id = self.env_id[t + 1].item()
                if next_env_id == env_id:
                    next_val = self.val[t + 1]
                else:
                    # Different env: use bootstrap value
                    next_val = (
                        last_vals[env_id]
                        if isinstance(last_vals, torch.Tensor)
                        else last_vals
                    )

            # Compute TD error and advantage
            delta = self.rew[t] + gamma * next_val * nonterminal - self.val[t]
            adv_per_env[env_id] = (
                delta + gamma * lam * nonterminal * adv_per_env[env_id]
            )
            self.adv[t] = adv_per_env[env_id]

        # Compute returns
        self.ret[: self.ptr] = self.adv[: self.ptr] + self.val[: self.ptr]

        # Normalize advantages
        if self.ptr > 0:
            am = self.adv[: self.ptr].mean()
            asd = self.adv[: self.ptr].std(unbiased=False) + 1e-8
            self.adv[: self.ptr] = (self.adv[: self.ptr] - am) / asd

        # Normalize returns if requested
        if normalize_returns and self.ptr > 0:
            ret_mean = self.ret[: self.ptr].mean()
            ret_std = self.ret[: self.ptr].std(unbiased=False) + 1e-8
            self.ret[: self.ptr] = (self.ret[: self.ptr] - ret_mean) / ret_std

    def get(self, batch_size):
        idxs = torch.randperm(self.ptr, device=self.device)
        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            mb_idx = idxs[start:end]
            yield (
                self.obs[mb_idx],
                self.acts[mb_idx],
                self.logp[mb_idx],
                self.adv[mb_idx],
                self.ret[mb_idx],
                self.val[mb_idx],  # Old value predictions for value clipping
            )

    def reset(self):
        self.ptr = 0


# ---------------------------
# Utils: seed, env, save
# ---------------------------
def set_seed(env, seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(env, "reset"):
        env.reset(seed=seed)
    elif hasattr(env, "envs"):  # vectorized env
        for i, e in enumerate(env.envs):
            e.reset(seed=seed + i)


def make_env(env_id, render_mode=None, num_envs=1):
    """Create environment(s). If num_envs > 1, returns vectorized env."""
    if num_envs == 1:
        return gym.make(env_id, render_mode=render_mode)
    else:

        def make_single_env(rank):
            def _make():
                env = gym.make(env_id, render_mode=None)
                env.reset(seed=42 + rank)
                return env

            return _make

        return AsyncVectorEnv([make_single_env(i) for i in range(num_envs)])


def save_checkpoint(path, net, opt, step, cfg, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Handle compiled models - get underlying module if compiled
    model_to_save = net._orig_mod if hasattr(net, "_orig_mod") else net
    payload = {
        "step": step,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": opt.state_dict(),
        "cfg": cfg.__dict__,
        "extra": extra or {},
    }
    torch.save(payload, path)
    print(f"💾 Saved checkpoint: {path}")


def find_latest_checkpoint(ckpt_dir: str, env_id: str) -> Optional[str]:
    """Find the latest checkpoint file in the checkpoint directory."""
    if not os.path.exists(ckpt_dir):
        return None

    # Look for checkpoint files matching the pattern
    pattern = os.path.join(ckpt_dir, f"{env_id}_*.pt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    # Sort by modification time, return the most recent
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_checkpoint(path: str, device) -> Tuple[ActorCritic, PPOConfig, dict]:
    """Load a checkpoint and return the model, config, and metadata."""
    print(f"📂 Loading checkpoint: {path}")
    payload = torch.load(path, map_location=device)

    cfg_dict = payload.get("cfg", {})
    cfg = PPOConfig(**cfg_dict)

    # Get observation and action dimensions from config or infer
    # We'll need to create the env temporarily to get dimensions
    temp_env = make_env(cfg.env_id, render_mode=None, num_envs=1)
    obs_dim = temp_env.observation_space.shape[0]
    act_dim = temp_env.action_space.shape[0]
    temp_env.close()

    net = ActorCritic(obs_dim, act_dim).to(device)

    # Load state dict - handle compiled models
    model_state = payload["model_state"]
    net.load_state_dict(model_state)
    net.eval()

    metadata = {
        "step": payload.get("step", 0),
        "extra": payload.get("extra", {}),
    }

    print(f"✅ Loaded checkpoint from step {metadata['step']}")
    return net, cfg, metadata


def load_checkpoint_for_training(
    path: str, device, net: ActorCritic, opt: optim.Optimizer
) -> Tuple[int, dict]:
    """
    Load a checkpoint for training (includes optimizer state).
    Returns the step count and metadata.
    """
    print(f"📂 Resuming training from checkpoint: {path}")
    payload = torch.load(path, map_location=device)

    # Load model state
    model_state = payload["model_state"]
    # Handle compiled models
    model_to_load = net._orig_mod if hasattr(net, "_orig_mod") else net
    model_to_load.load_state_dict(model_state)

    # Load optimizer state
    if "optimizer_state" in payload:
        opt.load_state_dict(payload["optimizer_state"])
        print("✅ Loaded optimizer state")

    step = payload.get("step", 0)
    metadata = payload.get("extra", {})

    print(f"✅ Resuming from step {step}")
    if metadata:
        if "sps" in metadata:
            print(f"   Previous SPS: {metadata['sps']}")

    return step, metadata


# ---------------------------
# Training / Evaluation
# ---------------------------
def ppo_train(cfg: PPOConfig, resume: bool = True):
    env = make_env(cfg.env_id, render_mode=None, num_envs=cfg.num_envs)
    set_seed(env, cfg.seed)

    # Handle both single and vectorized envs
    is_vectorized = hasattr(env, "num_envs")
    num_envs = env.num_envs if is_vectorized else 1
    obs_dim = (
        env.observation_space.shape[0]
        if not is_vectorized
        else env.single_observation_space.shape[0]
    )
    act_dim = (
        env.action_space.shape[0]
        if not is_vectorized
        else env.single_action_space.shape[0]
    )

    net = ActorCritic(obs_dim, act_dim).to(DEVICE)

    # Compile model for faster inference (PyTorch 2.0+)
    # Note: We compile after moving to device but before optimizer creation
    compiled = False
    try:
        if (
            hasattr(torch, "compile") and DEVICE.type != "mps"
        ):  # MPS may not support compile yet
            net = torch.compile(net, mode="reduce-overhead")
            compiled = True
            print("✅ Model compiled with torch.compile")
    except Exception as e:
        print(f"⚠️  torch.compile not available: {e}")

    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    # Learning rate annealing - will be updated manually based on total_collected
    def get_lr_factor(total_steps):
        # Linear decay from 1.0 to 0.1 over training
        progress = min(total_steps / cfg.total_steps, 1.0)
        return 1.0 - 0.9 * progress

    # Check for existing checkpoint to resume from
    total_collected = 0
    start_time = time.time()

    if resume:
        ckpt_path = find_latest_checkpoint(cfg.ckpt_dir, cfg.env_id)
    else:
        ckpt_path = None
        print("🔄 Starting fresh training (--no-resume flag set)")

    if ckpt_path:
        # Resume from checkpoint
        resume_step, metadata = load_checkpoint_for_training(
            ckpt_path, DEVICE, net, opt
        )
        total_collected = resume_step

        # Adjust start_time for accurate SPS calculation
        if "sps" in metadata and metadata["sps"] > 0:
            # Estimate elapsed time based on previous SPS
            estimated_elapsed = total_collected / metadata["sps"]
            start_time = time.time() - estimated_elapsed
            print(f"   Adjusted start time for SPS calculation")

        # Calculate next save checkpoint
        next_save_at = ((total_collected // cfg.save_every) + 1) * cfg.save_every
        print(f"   Resuming from step {total_collected}, next save at {next_save_at}")
    else:
        print("🆕 Starting fresh training (no checkpoint found)")
        next_save_at = cfg.save_every

    buf = RolloutBuffer(
        obs_dim,
        act_dim,
        cfg.rollout_steps,
        DEVICE,
        normalize_obs=cfg.normalize_obs,
        num_envs=num_envs,
    )

    # Initialize observations
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    if not is_vectorized:
        obs = obs.unsqueeze(0)

    # Normalize initial observations
    if cfg.normalize_obs:
        buf.update_obs_stats(obs)
        obs = buf.normalize_observations(obs)

    ep_rets = np.zeros(num_envs)
    ep_lens = np.zeros(num_envs, dtype=int)
    steps_per_rollout = cfg.rollout_steps // num_envs

    # Track episode returns for logging
    recent_ep_rets = []
    print_every = 10  # Print stats every N rollouts

    while total_collected < cfg.total_steps:
        buf.reset()
        # Collect rollout with vectorized envs
        for step in range(steps_per_rollout):
            # Normalize observations before network forward pass
            obs_normalized = (
                buf.normalize_observations(obs) if cfg.normalize_obs else obs
            )

            # Batch inference on GPU
            with torch.no_grad():
                a, logp = net.act(obs_normalized)  # Already batched
                v = net.v(obs_normalized)
                a_np = a.cpu().numpy()

            next_obs, r, terminated, truncated, infos = env.step(a_np)
            d = (terminated | truncated).astype(float)

            # Add batch to buffer
            r_tensor = (
                torch.tensor(r, dtype=torch.float32, device=DEVICE)
                if not isinstance(r, torch.Tensor)
                else r
            )
            d_tensor = (
                torch.tensor(d, dtype=torch.float32, device=DEVICE)
                if not isinstance(d, torch.Tensor)
                else d
            )
            # Store normalized observations in buffer with environment IDs
            env_ids = np.arange(num_envs) if is_vectorized else [0]
            buf.add(obs_normalized, a, logp, r_tensor, v, d_tensor, env_ids=env_ids)

            # Track episode stats
            ep_rets += r
            ep_lens += 1
            total_collected += num_envs

            # Reset episode stats for terminated envs (AsyncVectorEnv handles actual resets)
            reset_mask = terminated | truncated
            if reset_mask.any():
                reset_indices = np.where(reset_mask)[0]
                for idx in reset_indices:
                    if ep_lens[idx] > 0:  # Only log if episode had steps
                        recent_ep_rets.append(ep_rets[idx])
                    ep_rets[idx] = 0.0
                    ep_lens[idx] = 0

            obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)
            if not is_vectorized:
                obs = obs.unsqueeze(0)

            # Update observation statistics (use raw observations)
            if cfg.normalize_obs:
                buf.update_obs_stats(obs)

            # periodic checkpoint by steps
            if total_collected >= next_save_at:
                ckpt_path = os.path.join(
                    cfg.ckpt_dir, f"{cfg.env_id}_steps{total_collected}.pt"
                )
                save_checkpoint(
                    ckpt_path,
                    net,
                    opt,
                    total_collected,
                    cfg,
                    extra={
                        "sps": int(total_collected / (time.time() - start_time + 1e-9))
                    },
                )
                next_save_at += cfg.save_every

        # Bootstrap and compute GAE
        with torch.no_grad():
            obs_normalized = (
                buf.normalize_observations(obs) if cfg.normalize_obs else obs
            )
            last_vals = net.v(obs_normalized)  # Shape: (num_envs,)
        buf.compute_gae(
            last_vals, cfg.gamma, cfg.lam, normalize_returns=cfg.normalize_returns
        )

        # PPO update
        for _ in range(cfg.update_epochs):
            for obs_b, act_b, logp_old_b, adv_b, ret_b, v_old_b in buf.get(
                cfg.minibatch_size
            ):
                # Check for NaN in inputs
                if torch.isnan(obs_b).any() or torch.isnan(act_b).any():
                    print("⚠️  NaN detected in batch inputs, skipping update")
                    continue

                logp_b, ent_b = net.log_prob(obs_b, act_b)

                # Clamp log probabilities to prevent overflow
                logp_b = torch.clamp(logp_b, -20.0, 20.0)
                logp_old_b = torch.clamp(logp_old_b, -20.0, 20.0)

                ratio = (logp_b - logp_old_b).exp()
                ratio = torch.clamp(ratio, 0.0, 10.0)  # Prevent extreme ratios

                unclipped = ratio * adv_b
                clipped = (
                    torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                    * adv_b
                )
                pi_loss = -torch.min(unclipped, clipped).mean()

                v_pred = net.v(obs_b)
                # Value clipping: clip prediction around old value predictions
                v_pred_clipped = v_old_b + torch.clamp(
                    v_pred - v_old_b, -cfg.clip_ratio, cfg.clip_ratio
                )
                v_loss_unclipped = 0.5 * (ret_b - v_pred).pow(2)
                v_loss_clipped = 0.5 * (ret_b - v_pred_clipped).pow(2)
                v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

                ent = ent_b.mean()
                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️  NaN/Inf detected in loss, skipping update")
                    continue

                opt.zero_grad(set_to_none=True)
                loss.backward()

                # Check for NaN gradients
                has_nan_grad = False
                for param in net.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    print("⚠️  NaN detected in gradients, skipping update")
                    opt.zero_grad(set_to_none=True)
                    continue

                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

        # Update learning rate based on total steps
        lr_factor = get_lr_factor(total_collected)
        for param_group in opt.param_groups:
            param_group["lr"] = cfg.lr * lr_factor

        sps = int(total_collected / (time.time() - start_time + 1e-9))

        # Calculate episode statistics
        avg_ep_ret = np.mean(recent_ep_rets[-100:]) if recent_ep_rets else 0.0
        max_ep_ret = np.max(recent_ep_rets[-100:]) if recent_ep_rets else 0.0
        num_episodes = len(recent_ep_rets)

        print(
            f"[{total_collected}/{cfg.total_steps}] sps={sps} | "
            f"loss={loss.item():.4f} pi={pi_loss.item():.4f} v={v_loss.item():.4f} ent={ent.item():.4f} | "
            f"ep_ret_avg={avg_ep_ret:.2f} ep_ret_max={max_ep_ret:.2f} episodes={num_episodes}"
        )

        # Keep only recent episodes for memory efficiency
        if len(recent_ep_rets) > 1000:
            recent_ep_rets = recent_ep_rets[-1000:]

    # final checkpoint
    final_path = os.path.join(
        cfg.ckpt_dir, f"{cfg.env_id}_final_steps{total_collected}.pt"
    )
    save_checkpoint(final_path, net, opt, total_collected, cfg)

    env.close()
    return net


def visualize_policy(
    net: ActorCritic,
    cfg: PPOConfig,
    step_delay: float = 0.0,
    max_steps: Optional[int] = None,
):
    """
    Visualize the policy by running episodes.

    Args:
        net: The trained ActorCritic network
        cfg: PPOConfig with environment settings
        step_delay: Delay in seconds between environment steps (for visualization)
        max_steps: Maximum steps per episode (None for no limit)
    """
    env = make_env(cfg.env_id, render_mode=cfg.render_mode, num_envs=1)
    o, _ = env.reset(seed=cfg.seed + 123)
    o_t = torch.tensor(o, dtype=torch.float32, device=DEVICE)

    frames = []
    ep_rets = []
    for ep in range(cfg.eval_episodes):
        ep_ret = 0.0
        ep_steps = 0
        terminated = truncated = False
        o_t = torch.tensor(o, dtype=torch.float32, device=DEVICE)

        print(f"Episode {ep + 1}/{cfg.eval_episodes} starting...")

        while not (terminated or truncated):
            if max_steps is not None and ep_steps >= max_steps:
                print(f"Episode {ep + 1} reached max steps ({max_steps})")
                break

            with torch.no_grad():
                mu, std = net.pi(o_t.unsqueeze(0))
                a = mu.squeeze(0).cpu().numpy()  # deterministic eval

            o, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            ep_steps += 1

            if cfg.render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            elif cfg.render_mode == "human":
                # Render is called automatically, but we add delay for visualization
                if step_delay > 0:
                    time.sleep(step_delay)

            o_t = torch.tensor(o, dtype=torch.float32, device=DEVICE)

        ep_rets.append(ep_ret)
        print(f"Episode {ep + 1} finished: Return={ep_ret:.2f}, Steps={ep_steps}")
        o, _ = env.reset()

    print(
        f"\nEval returns over {cfg.eval_episodes} episodes: "
        f"mean={np.mean(ep_rets):.1f} ± {np.std(ep_rets):.1f}"
    )

    if cfg.render_mode == "rgb_array" and frames:
        print(f"Saving video to {cfg.video_out} ({len(frames)} frames)")
        imageio.mimsave(cfg.video_out, frames, fps=30)

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="PPO training and evaluation for MuJoCo Humanoid"
    )
    parser.add_argument("--steps", type=int, default=300_000, help="total env steps")
    parser.add_argument("--rollout", type=int, default=8192, help="steps per rollout")
    parser.add_argument("--mb", type=int, default=4096, help="minibatch size")
    parser.add_argument("--epochs", type=int, default=10, help="epochs per update")
    parser.add_argument(
        "--render", type=str, default="human", choices=["human", "rgb_array"]
    )
    parser.add_argument("--eval_eps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--save_every", type=int, default=100_000, help="save every N env steps"
    )
    parser.add_argument(
        "--num_envs", type=int, default=8, help="number of parallel environments"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Load latest checkpoint and play simulation (no training)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.01,
        help="Delay in seconds between environment steps when playing (default: 0.01)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Specific checkpoint path to load (if not provided, loads latest)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum steps per episode when playing (None for no limit)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch even if checkpoint exists",
    )
    args = parser.parse_args()

    cfg = PPOConfig(
        total_steps=args.steps,
        rollout_steps=args.rollout,
        minibatch_size=args.mb,
        update_epochs=args.epochs,
        render_mode=args.render,
        eval_episodes=args.eval_eps,
        seed=args.seed,
        ckpt_dir=args.ckpt_dir,
        save_every=args.save_every,
        num_envs=args.num_envs,
    )

    print(f"Using device: {DEVICE}")

    if args.play:
        # Play mode: load checkpoint and visualize
        if args.ckpt_path:
            ckpt_path = args.ckpt_path
            if not os.path.exists(ckpt_path):
                print(f"❌ Checkpoint not found: {ckpt_path}")
                return
        else:
            ckpt_path = find_latest_checkpoint(cfg.ckpt_dir, cfg.env_id)
            if ckpt_path is None:
                print(f"❌ No checkpoint found in {cfg.ckpt_dir} for {cfg.env_id}")
                print("   Run training first or specify --ckpt_path")
                return

        net, loaded_cfg, metadata = load_checkpoint(ckpt_path, DEVICE)

        # Override config with command line args for visualization
        if args.render:
            loaded_cfg.render_mode = args.render
        if args.eval_eps:
            loaded_cfg.eval_episodes = args.eval_eps

        print(f"🎮 Playing simulation with step delay: {args.delay}s")
        visualize_policy(
            net, loaded_cfg, step_delay=args.delay, max_steps=args.max_steps
        )
    else:
        # Training mode
        print(f"Running {cfg.num_envs} parallel environments")
        net = ppo_train(cfg, resume=not args.no_resume)
        visualize_policy(net, cfg)


if __name__ == "__main__":
    main()
