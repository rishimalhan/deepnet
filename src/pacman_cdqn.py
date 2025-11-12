#! /usr/bin/env python3

import imageio
from PIL import Image
from torchvision import transforms
import os
import signal
import gc
import traceback
import numpy as np
import random
import dataclasses
import gymnasium as gym
import torch
from torch.nn import Module, Conv2d, BatchNorm2d, Linear
from torch.optim import Adam
from torch.nn.functional import relu, mse_loss
from collections import deque

SEED = 42
SIM = True

if not SIM:
    torch.manual_seed(SEED)


def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocess = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    return preprocess(frame).unsqueeze(0)


@dataclasses.dataclass
class CDQNconfig:
    epsilon = 1.0
    decay_factor = 0.995
    epsilon_stop = 0.01
    max_episodes = 2000
    steps_per_episode = 10000
    device = "mps" if torch.mps.is_available() else "cpu"
    action_dim = None
    minibatch_size = 64
    memory_size = int(1e4)
    discount_factor = 0.99
    environment_name = "MsPacmanDeterministic-v0"  # correct Atari ID
    learning_rate = 5e-4
    interpolation_factor = 1e-2
    learn_every = 4
    save_every = 20
    model_path = None


config = CDQNconfig()


class CDQN(Module):
    def __init__(self, action_dim):
        super(CDQN, self).__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = BatchNorm2d(32)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = BatchNorm2d(64)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = BatchNorm2d(64)
        self.conv4 = Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = BatchNorm2d(128)
        self.fc1 = Linear(10 * 10 * 128, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, action_dim)

    def forward(self, x):
        # x: (N,C,128,128)
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        x = relu(self.bn3(self.conv3(x)))
        x = relu(self.bn4(self.conv4(x)))
        # Make contiguous before reshaping to ensure backward pass compatibility
        x = x.contiguous().reshape(x.size(0), -1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        return self.fc3(x)  # linear Q-head (do NOT clamp with ReLU)


class Agent:
    def __init__(self, config):
        self.local_network = CDQN(action_dim=config.action_dim).to(config.device)
        self.target_network = CDQN(action_dim=config.action_dim).to(config.device)

        self.load_model(filepath=config.model_path)

        # Hard-sync target at start (very important)
        self.target_network.load_state_dict(self.local_network.state_dict())
        # Target net never needs grads
        self.target_network.requires_grad_(False)
        self.replay = deque(maxlen=config.memory_size)

        self.optimizer = Adam(
            params=self.local_network.parameters(), lr=config.learning_rate
        )
        self.step_count = 0

    def act(self, state, epsilon):
        state = preprocess_frame(state)
        if random.random() <= epsilon:
            return random.randrange(config.action_dim)
        self.local_network.eval()
        with torch.no_grad():
            q = self.local_network(state.to(config.device))
            return int(q.argmax(dim=1).item())

    def step(self, state, action, reward, next_state, done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.replay.append((state, action, reward, next_state, done))
        if len(self.replay) < config.minibatch_size:
            return None
        self.local_network.train()
        experiences = random.sample(self.replay, k=config.minibatch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(config.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(config.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(config.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(config.device)
        dones = (
            torch.from_numpy(np.vstack(dones).astype(np.uint8))
            .float()
            .to(config.device)
        )
        q_current = self.local_network(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + config.discount_factor * q_next * (1.0 - dones)
        loss = mse_loss(q_current, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()
        return loss.item()

    def soft_update(self):
        for target_param, local_param in zip(
            self.target_network.parameters(), self.local_network.parameters()
        ):
            target_param.data.copy_(
                config.interpolation_factor * local_param.data
                + (1.0 - config.interpolation_factor) * target_param.data
            )

    def save_model(self, filepath):
        torch.save(self.local_network.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.local_network.load_state_dict(
                torch.load(filepath, map_location=config.device, weights_only=True)
            )
            self.target_network.load_state_dict(
                torch.load(filepath, map_location=config.device, weights_only=True)
            )
            print(f"Model loaded from {filepath}")
            return True
        return False


def cleanup_resources(env):
    print("\nCleaning up resources...")
    if env is not None:
        try:
            env.close()
        except Exception:
            pass

    # Clear GPU cache
    if config.device == "mps":
        torch.mps.empty_cache()
    elif config.device == "cuda":
        torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()

    print("Cleanup complete.")


def signal_handler(sig, frame):
    print("\n\nInterrupted by user (Ctrl+C). Shutting down gracefully...")
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    model_path = "./model/pacman_cdqn.pth"
    config.model_path = model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    env = gym.make(
        config.environment_name,
        full_action_space=False,
    )
    config.action_dim = env.action_space.n

    if not SIM:
        try:
            agent = Agent(config=config)
            episode = 1
            scores_on_100_episodes = deque(maxlen=100)
            while episode <= config.max_episodes:
                state, _ = env.reset()
                losses = deque(maxlen=100)
                scores = 0
                for _ in range(config.steps_per_episode):
                    action = agent.act(
                        state,
                        epsilon=config.epsilon,
                    )
                    next_state, reward, done, _, _ = env.step(action)
                    loss = agent.step(state, action, reward, next_state, done)
                    if loss is not None:
                        losses.append(loss)
                    scores += reward
                    state = next_state
                    if done:
                        break

                scores_on_100_episodes.append(scores)
                config.epsilon = max(
                    config.epsilon_stop, config.decay_factor * config.epsilon
                )
                avg_loss = float(np.mean(losses)) if len(losses) else float("nan")
                avg_score_on_100_episodes = (
                    float(np.mean(scores_on_100_episodes).round(2))
                    if len(scores_on_100_episodes) >= 100
                    else "N/A"
                )
                print(
                    f"Episode {episode}\tTotal Score: {scores:.2f}\tScore on 100 episodes mean: {avg_score_on_100_episodes}"
                    f"\tEpsilon: {config.epsilon:.3f}"
                )
                if episode % config.save_every == 0:
                    agent.save_model(model_path)
                episode += 1
                if (
                    avg_score_on_100_episodes != "N/A"
                    and avg_score_on_100_episodes >= 700.0
                ):
                    print(f"Environment solved in {episode} episodes!")
                    agent.save_model(model_path)
                    break
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt received. Cleaning up...")
        except Exception as e:
            print(f"\n\nError occurred: {e}")
            traceback.print_exc()
        finally:
            cleanup_resources(env)
    else:
        agent = Agent(config=config)

        def show_video_of_model(agent):
            env = gym.make(config.environment_name, render_mode="rgb_array")
            state, _ = env.reset(seed=SEED)
            done = False
            frames = []
            total_reward = 0
            config.epsilon = -1.0
            while not done:
                frame = env.render()
                frames.append(frame)
                action = agent.act(state, config.epsilon)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                if done:
                    print(f"Total reward: {total_reward}")
                    break
            env.close()
            imageio.mimsave("./videos/pacman_cdqn.mp4", frames, fps=30)

        show_video_of_model(agent)
