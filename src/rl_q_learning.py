#! /usr/bin/python3

import os
import random
import numpy as np
import gym
from collections import defaultdict
import time
from torch.utils.tensorboard import SummaryWriter

render_mode = "rgb_array"
bins = np.array([10, 10, 10, 10])

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
SIM = True
EVAL = True

lower_bound = np.array([-1.91444771, -3.51599329, -0.26745233, -3.52754485])
upper_bound = np.array([2.12142583, 3.42483365, 0.269554, 3.50725647])
diff_bound = np.subtract(upper_bound, lower_bound)
q_values = defaultdict(float)


def get_state(env):
    x = env.state
    normalized_state = np.array(
        np.divide(np.subtract(x, lower_bound), diff_bound) * bins, dtype=np.int64
    )
    return tuple(normalized_state.tolist())


# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=os.path.join(ROOT, "logs", "rl_q_learning"))

cartpole_env = gym.make("CartPole-v1", render_mode=render_mode)
cartpole_env.reset()
prev_key = None
epsilon = 0.1
alpha = 0.1
gamma = 0.9

# Training metrics
episode = 0
episode_reward = 0
episode_rewards = []
step = 0

print("Starting Q-Learning training...")

for training_step in range(100000):
    state = get_state(cartpole_env)

    # ε-greedy action selection
    if random.random() < epsilon:
        action = random.choice([0, 1])
    else:
        q0 = q_values[state + (0,)]
        q1 = q_values[state + (1,)]
        action = 0 if q0 >= q1 else 1

    obs, reward, done, _, _ = cartpole_env.step(action)
    episode_reward += reward
    next_state = get_state(cartpole_env)

    # Q-learning update
    if prev_key is not None:
        max_q_next = max(
            q_values[next_state + (0,)],
            q_values[next_state + (1,)],
        )
        q_values[prev_key] += alpha * (reward + gamma * max_q_next - q_values[prev_key])

    prev_key = state + (action,)

    # Log Q-values for current state periodically
    writer.add_scalar("Q-value[0]", q_values[state + (0,)], training_step)
    writer.add_scalar("Q-value[1]", q_values[state + (1,)], training_step)

    step += 1
    if done:
        # Log episode reward
        writer.add_scalar("Episode Reward", episode_reward, episode)
        episode_rewards.append(episode_reward)

        # Log episode length
        writer.add_scalar("Episode Length", step, episode)

        if episode % 100 == 0:
            avg_reward = (
                np.mean(episode_rewards[-100:])
                if len(episode_rewards) >= 100
                else np.mean(episode_rewards)
            )
            writer.add_scalar("Average Reward (last 100)", avg_reward, episode)
            print(
                f"Episode {episode}: Reward = {episode_reward}, Avg Reward = {avg_reward:.2f}"
            )

        cartpole_env.reset()
        prev_key = None
        episode += 1
        episode_reward = 0
        step = 0

print(f"Training completed! Total episodes: {episode}")

# EVAL Section
if EVAL:
    print("\n=== EVALUATION PHASE ===")

    # Evaluation metrics
    eval_episodes = 100
    eval_rewards = []
    eval_episode_lengths = []

    # Create evaluation environment
    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")

    for eval_ep in range(eval_episodes):
        eval_env.reset()
        eval_reward = 0
        eval_steps = 0
        done = False

        while not done:
            state = get_state(eval_env)

            # Use learned policy (greedy action selection)
            q0 = q_values[state + (0,)]
            q1 = q_values[state + (1,)]
            action = 0 if q0 >= q1 else 1

            _, reward, done, _, _ = eval_env.step(action)
            eval_reward += reward
            eval_steps += 1

        eval_rewards.append(eval_reward)
        eval_episode_lengths.append(eval_steps)

        # Log evaluation metrics
        writer.add_scalar("Eval/Episode Reward", eval_reward, eval_ep)
        writer.add_scalar("Eval/Episode Length", eval_steps, eval_ep)

    # Calculate and log evaluation statistics
    mean_eval_reward = np.mean(eval_rewards)
    std_eval_reward = np.std(eval_rewards)
    mean_eval_length = np.mean(eval_episode_lengths)
    std_eval_length = np.std(eval_episode_lengths)

    writer.add_scalar("Eval/Mean Reward", mean_eval_reward, 0)
    writer.add_scalar("Eval/Std Reward", std_eval_reward, 0)
    writer.add_scalar("Eval/Mean Length", mean_eval_length, 0)
    writer.add_scalar("Eval/Std Length", std_eval_length, 0)

    print(f"Evaluation Results ({eval_episodes} episodes):")
    print(f"  Mean Reward: {mean_eval_reward:.2f} ± {std_eval_reward:.2f}")
    print(f"  Mean Episode Length: {mean_eval_length:.2f} ± {std_eval_length:.2f}")
    print(f"  Max Reward: {max(eval_rewards)}")
    print(f"  Min Reward: {min(eval_rewards)}")

    # Log Q-value statistics
    non_zero_q_values = [v for v in q_values.values() if v != 0]
    if non_zero_q_values:
        writer.add_scalar("Q-values/Mean", np.mean(non_zero_q_values), 0)
        writer.add_scalar("Q-values/Max", np.max(non_zero_q_values), 0)
        writer.add_scalar("Q-values/Min", np.min(non_zero_q_values), 0)
        print(f"Q-value Statistics:")
        print(f"  Total Q-values learned: {len(q_values)}")
        print(f"  Non-zero Q-values: {len(non_zero_q_values)}")
        print(f"  Mean Q-value: {np.mean(non_zero_q_values):.4f}")

    eval_env.close()

# Simulate
if SIM:
    print("\n=== SIMULATION PHASE ===")
    render_mode = "human"
    cartpole_env = gym.make("CartPole-v1", render_mode=render_mode)
    cartpole_env.reset()
    time.sleep(1.0)

    done = False
    sim_steps = 0
    while not done and sim_steps < 500:  # Limit simulation steps
        state = get_state(cartpole_env)

        q_value = [
            q_values.get(state + (0,)),
            q_values.get(state + (1,)),
        ]
        if q_value[0] is None or q_value[1] is None:
            action = random.choice([0, 1])
        else:
            action = int(np.argmax(q_value))

        print(f"Step {sim_steps}: q: {q_value}, action: {action}")
        _, _, done, _, _ = cartpole_env.step(action)
        sim_steps += 1
        time.sleep(0.1)

    print(f"Simulation ended after {sim_steps} steps")

# Close TensorBoard writer
writer.close()
print("\nTensorBoard logs saved. Run 'tensorboard --logdir=runs' to view them.")
