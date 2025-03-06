#! /usr/bin/python3

"""
Reinforcement learning is a system where agents act on environment following a learnt way to maximize cummulative reward.

Supervised:
- Given labeled data, model learns to predict the labels
i.e. learning mappings

Unsupervised:
- Model learns structure to unorganized data
i.e. learning patterns

RL:
- Learns from experience while interacting with the environment
- Reward communicates what you want to achieve and not how you want to achieve
- Rewards are discounted to bias towards immediate action
i.e. present value of future rewards

Agent, reward, state, value function is part of RL design

Q-learning
- At the heart is a Q-table. Rows are all states and columns are all actions
- Fields is the value for state-action pair
- Q(st, at) = Q(st, at) + alpha * [Rt+1 + gamma * max(Q(st+1, a)) - Q(st, at)]
- alpha is learning rate and gamma is discount factor
- learning rate is the amount by which to override the existing value

Exploration vs Exploitation
- Exploration is to try new things
- Exploitation is to use what you know to get rewards
- epsilon-greedy strategy is to choose random action with epsilon probability

Deep-Q Networks and Experience Replay
- State space may not be discrete
- State space can be huge

DQN solves the above two problems
- Learn the state to action mapping or table using a neural network
- Can be ANN or CNN e.g. in Atari game researchers used a CNN over snapshots of the game

The network is learning to approximate the expected sum of future rewards from a given state-action pair, discounted over time.
Input is state (feature, image, etc) and output is cummulative future reward for N-dim action space.

Training:
- If we pass state, action, reward, and next state to network one by one, it is not efficient since variety is missing
- Instead we maintain experience queue and sample from it to train the network
- Experience queue gets new experiences and dequeues old ones


Cartpole problem:
State: angle, angular speed, position, horizontal velocity
Episode ends if angle is beyond X degrees and reward +1 received when pole upright
- All states may not be discrete
"""

import time
import os
import gym
from torch.nn import Module
from torch.nn.functional import relu
from torch.nn import Linear
from torch.autograd import Variable
from torch.optim import Adam
import torch
import numpy as np
from collections import deque
import random
from torch.nn import MSELoss
from torch import Tensor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEVICE = "mps" if torch.mps.is_available() else "cpu"
EPOCHS = 500
MEMORY_SIZE = 2000


class DQNModel(Module):
    """
    Model predicts the rewards in action space
    hence argmax is the action index
    we train the max reward with target reward to compute loss
    """

    def __init__(self, observation_dim, action_dim):
        super(DQNModel, self).__init__()
        self._layer1 = Linear(in_features=observation_dim, out_features=120)
        self._relu1 = relu
        self._layer2 = Linear(in_features=120, out_features=action_dim)

    def forward(self, x):
        x = self._layer1(x)
        x = self._relu1(x)
        x = self._layer2(x)
        return x


class DQNAgent:
    def __init__(self, render_mode="rgb_array"):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.env.reset()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.model = DQNModel(
            observation_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
        ).to(DEVICE)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = MSELoss()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.iterations = 10
        self.batch_size = int(MEMORY_SIZE / self.iterations)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        act_values = self.model(Variable(data=Tensor(state)).to(DEVICE))
        return torch.argmax(act_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        average_loss = 0.0

        for i in range(self.iterations):
            mini_batch = random.sample(self.memory, self.batch_size)
            # Convert batch to tensors
            states = Tensor(np.array([value[0] for value in mini_batch])).to(DEVICE)
            actions = (
                Tensor(np.array([value[1] for value in mini_batch])).to(DEVICE).long()
            )  # Assuming discrete actions
            rewards = Tensor(np.array([value[2] for value in mini_batch])).to(DEVICE)
            new_states = Tensor(np.array([value[3] for value in mini_batch])).to(DEVICE)
            dones = Tensor(np.array([value[-1] for value in mini_batch])).to(
                DEVICE
            )  # 1 if terminal, 0 otherwise
            # Compute current Q-values
            current_values = self.model(states)
            # Get the Q-values of the next states
            with torch.no_grad():  # Don't compute gradients for next state values
                next_q_values = self.model(new_states).max(dim=1)[
                    0
                ]  # max Q-value across actions. 0 because 1 represents indices.
            # Compute target values: If done, use reward; otherwise, use reward + gamma * max(Q(s', a'))
            target_values = rewards + self.gamma * next_q_values * (1 - dones)
            # Get Q-values corresponding to taken actions since some are random actions and not necessarily max Q value actions
            current_q_values = current_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            # Compute loss
            loss = self.criterion(current_q_values, target_values)
            average_loss += loss.item()
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        average_loss /= self.iterations
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return average_loss

    def save_model(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(ROOT, "model", "dqn.model.pth"),
        )

    def load_model(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(ROOT, "model", "dqn.model.pth"),
                map_location=DEVICE,
            )
        )
        self.model.eval()

    def train(self):
        self.model.train()
        state = self.env.reset()[0]
        for epoch in range(EPOCHS):
            for time in range(MEMORY_SIZE):
                action = self.act(state)
                new_state, reward, done, _, _ = self.env.step(action=action)
                if done:
                    state = self.env.reset()[0]
                    done = False
                self.remember(state, action, reward, new_state, done)
                state = new_state
            average_loss = self.replay()
            logger.info(f"Epoch: {epoch}. Loss: {average_loss}")
            self.save_model()


if __name__ == "__main__":
    render_mode = "human"
    # render_mode = "rgb_array"
    agent = DQNAgent(render_mode=render_mode)

    # Train
    if render_mode == "rgb_array":
        agent.train()

    # Load and use
    if render_mode == "human":
        agent.load_model()
        agent.epsilon = 0.0  # Important piece here
        for i in range(10):
            state = agent.env.reset()[0]
            done = False
            while not done:
                action = agent.act(state)
                state, _, done, _, _ = agent.env.step(action=action)
                agent.env.render()
                time.sleep(0.01)
        agent.env.close()
