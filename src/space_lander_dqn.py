#! /usr/bin/env python3

"""
Architecture:

Network
- Brain of the agent that learns the action-distribution or policy (probability of choosing an action in a particular state).
- Input is state dimension of observation space and output is action dimension

Memory
- Replay buffer for experiences agent has
- Maintain a max length buffer and sample it randomly for a batch
- Needs a push and sample command

Agent itself
- Uses network and memory alongside optimizer from torch
- Act: Using some epsilon greedy algorithm choose an action from online network
- Step: Called after every environment step and based on some time step factor- learn from experience buffer
- Learn: Actual training part where collection from experiences are used to get q next state from target and q from rewards and online network.

Training
Choose start epsilon and decay it until a certain threshold
For max episodes
- For timesteps per episode
- - get action from online network
- - act on environment
- - collect rewards and take step
"""

import os
import time
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import dataclasses

# Set SIM=True to visualize the trained model
SIM = False


@dataclasses.dataclass
class DQNconfig:
    epsilon = 1.0
    decay_factor = 0.995
    epsilon_stop = 0.01
    max_episodes = 2000
    steps_per_episode = 1000
    device = "mps" if torch.mps.is_available() else "cpu"
    datatype = torch.float32
    action_dim = None
    obs_dim = None
    training_timestep = 4
    minibatch_size = 100
    memory_size = int(1e5)
    discount_factor = 0.99
    environment_name = "LunarLander-v3"
    learning_rate = 5e-4
    interpolation_factor = 1e-3


config = DQNconfig()


class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(in_features=state_size, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=64)
        self.layer3 = nn.Linear(in_features=64, out_features=action_size)

    def forward(self, state):
        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x


class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=config.memory_size)

    def push(self, state, action, reward, new_state, done):
        # Convert to tensors and store on GPU directly
        state_tensor = torch.from_numpy(state).float().to(config.device)
        action_tensor = torch.tensor(
            action, dtype=torch.long, device=config.device
        ).unsqueeze(0)
        reward_tensor = torch.tensor(
            reward, dtype=torch.float32, device=config.device
        ).unsqueeze(0)
        next_state_tensor = torch.from_numpy(new_state).float().to(config.device)
        done_tensor = torch.tensor(
            done, dtype=torch.float32, device=config.device
        ).unsqueeze(0)

        experience = (
            state_tensor,
            action_tensor,
            reward_tensor,
            next_state_tensor,
            done_tensor,
        )
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, k=config.minibatch_size)
        # Stack tensors directly (already on GPU)
        states = torch.stack([e[0] for e in experiences])
        actions = torch.stack([e[1] for e in experiences])
        rewards = torch.stack([e[2] for e in experiences])
        next_states = torch.stack([e[3] for e in experiences])
        dones = torch.stack([e[4] for e in experiences])
        return (states, actions, rewards, next_states, dones)


class Agent:
    def __init__(self):
        self.local_network = Network(
            state_size=config.obs_dim, action_size=config.action_dim
        ).to(config.device)
        self.target_network = Network(
            state_size=config.obs_dim, action_size=config.action_dim
        ).to(config.device)
        self.optimizer = optim.Adam(
            params=self.local_network.parameters(), lr=config.learning_rate
        )
        self.replay = ReplayMemory()
        self.stepper = 0

    def act(self, state):
        state_tensor = (
            torch.from_numpy(state)
            .type(dtype=config.datatype)
            .unsqueeze(0)
            .to(config.device)
        )
        self.local_network.eval()
        with torch.no_grad():
            model_output = self.local_network.forward(state_tensor)
        self.local_network.train()
        if config.epsilon > random.random():
            return random.choice(np.arange(config.action_dim))
        else:
            return model_output.argmax().item()

    def step(self, state, action, reward, new_state, done):
        self.replay.push(state, action, reward, new_state, done)
        self.stepper = (self.stepper + 1) % config.training_timestep
        if self.stepper == 0 and len(self.replay.memory) > config.minibatch_size:
            experiences = self.replay.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        self.optimizer.zero_grad()

        # Get max Q values for next states from target network
        with torch.no_grad():
            Q_targets_next = self.target_network(next_states).max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (config.discount_factor * Q_targets_next * (1 - dones))

        # Get expected Q values from local network
        Q_expected = self.local_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(
            self.target_network.parameters(), self.local_network.parameters()
        ):
            target_param.data.copy_(
                config.interpolation_factor * local_param.data
                + (1.0 - config.interpolation_factor) * target_param.data
            )

    def save_model(self, filepath):
        """Simple model save - just save the local network"""
        torch.save(self.local_network.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Simple model load - just load the local network"""
        if os.path.exists(filepath):
            self.local_network.load_state_dict(
                torch.load(filepath, map_location=config.device, weights_only=True)
            )
            # Copy weights to target network
            self.target_network.load_state_dict(
                torch.load(filepath, map_location=config.device, weights_only=True)
            )
            print(f"Model loaded from {filepath}")
            return True
        return False


if __name__ == "__main__":
    # Simple model path setup - model is sibling to src directory
    model_path = "../model/lunar_lander_dqn.pth"

    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Create environment with rendering if SIM mode
    if SIM:
        env = gym.make(config.environment_name, render_mode="human")
    else:
        env = gym.make(config.environment_name)

    config.obs_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.n
    agent = Agent()

    if SIM:
        config.epsilon = 0.0
        # Visualization mode - load model and run episodes
        if agent.load_model(model_path):
            print("Model loaded! Starting visualization...")
            print("Close the window to stop.")

            for episode in range(5):  # Run 5 episodes for visualization
                state, _ = env.reset()
                score = 0

                for step in range(config.steps_per_episode):
                    action = agent.act(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    state = next_state
                    score += reward

                    if done:
                        break

                    time.sleep(0.001)

                print(f"Episode {episode + 1} Score: {score:.2f}")
    else:
        # Training mode
        # Try to load existing model
        if agent.load_model(model_path):
            print("Loaded existing model, continuing training...")
        else:
            print("No existing model found, starting fresh...")

        scores = []
        scores_window = deque(maxlen=100)

        for episode in range(config.max_episodes):
            state, _ = env.reset()
            score = 0

            for step in range(config.steps_per_episode):
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward

                if done:
                    break

            scores_window.append(score)
            scores.append(score)

            # Decay epsilon
            if config.epsilon > config.epsilon_stop:
                config.epsilon *= config.decay_factor

            if episode % 100 == 0:
                print(
                    f"Episode {episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {config.epsilon:.3f}"
                )

            if np.mean(scores_window) >= 200.0:
                print(
                    f"\nEnvironment solved in {episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}"
                )
                break

        # Save model at the end
        agent.save_model(model_path)

    env.close()
