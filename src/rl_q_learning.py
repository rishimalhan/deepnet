#! /usr/bin /python3

import random
import numpy as np
import gym
from collections import defaultdict
import time

render_mode = "rgb_array"
bins = np.array([10, 10, 10, 10])

SIM = True

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


cartpole_env = gym.make("CartPole-v1", render_mode=render_mode)
cartpole_env.reset()
prev_key = None
epsilon = 0.1
alpha = 0.1
gamma = 0.9

for _ in range(100000):
    state = get_state(cartpole_env)

    # ε-greedy action selection
    if random.random() < epsilon:
        action = random.choice([0, 1])
    else:
        q0 = q_values[state + (0,)]
        q1 = q_values[state + (1,)]
        action = 0 if q0 >= q1 else 1

    obs, reward, done, _, _ = cartpole_env.step(action)
    next_state = get_state(cartpole_env)

    # Q-learning update
    if prev_key is not None:
        max_q_next = max(
            q_values[next_state + (0,)],
            q_values[next_state + (1,)],
        )
        q_values[prev_key] += alpha * (reward + gamma * max_q_next - q_values[prev_key])

    prev_key = state + (action,)

    if done:
        cartpole_env.reset()
        prev_key = None

# Simulate
if SIM:
    render_mode = "human"
    cartpole_env = gym.make("CartPole-v1", render_mode=render_mode)
    cartpole_env.reset()
    time.sleep(1.0)

    done = False
    while not done:
        state = get_state(cartpole_env)

        q_value = [
            q_values.get(state + (0,)),
            q_values.get(state + (1,)),
        ]
        if q_value[0] is None or q_value[1] is None:
            action = random.choice([0, 1])
        else:
            action = int(np.argmax(q_value))

        print(f"q: {q_value}, action: {action}")
        _, _, done, _, _ = cartpole_env.step(action)
        time.sleep(0.1)
