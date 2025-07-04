#! /usr/bin/python3

import numpy as np
import random

NUM_DEST = 12
GAMMA = 0.9
ALPHA = 0.1

REWARDS = np.zeros(shape=(NUM_DEST, NUM_DEST))
state_to_index = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
}
index_to_state = {v: k for k, v in state_to_index.items()}

# Set all valid rewards or transitions to 1.
REWARDS[0, 1] = 1  # A -> B

REWARDS[1, 0] = 1  # B -> A
REWARDS[1, 2] = 1  # B -> C
REWARDS[1, 5] = 1  # B -> F

REWARDS[2, 1] = 1  # C -> B
REWARDS[2, 6] = 1  # C -> G

REWARDS[3, 7] = 1  # D -> H

REWARDS[4, 8] = 1  # E -> I

REWARDS[5, 1] = 1  # F -> B
REWARDS[5, 9] = 1  # F -> J

REWARDS[6, 2] = 1  # G -> C
REWARDS[6, 7] = 1  # G -> H

REWARDS[7, 3] = 1  # H -> D
REWARDS[7, 6] = 1  # H -> G
REWARDS[7, 11] = 1  # H -> L

REWARDS[8, 4] = 1  # I -> E
REWARDS[8, 9] = 1  # I -> J

REWARDS[9, 5] = 1  # J -> F
REWARDS[9, 8] = 1  # J -> I
REWARDS[9, 10] = 1  # J -> K

REWARDS[10, 9] = 1  # K -> J
REWARDS[10, 11] = 1  # K -> L

REWARDS[11, 7] = 1  # L -> H
REWARDS[11, 10] = 1  # L -> K


def learner(goal, rewards, q_matrix, iterations=1000):
    """
    Generic policy for any start given the priority list and goal.
    """

    rewards_copy = rewards.copy()
    goal = state_to_index[goal]
    rewards_copy[:, goal] *= 1000.0
    all_nodes = set(range(NUM_DEST)) - {goal}
    node = random.choice(list(all_nodes))
    for _ in range(iterations):
        new_node = random.choice(np.where(rewards_copy[node] > 0)[0])
        q_current = q_matrix[node, new_node]
        q_future = rewards_copy[node, new_node] + GAMMA * np.max(q_matrix[new_node])
        td = q_future - q_current
        # learning
        q_matrix[node, new_node] += ALPHA * td
        if new_node == goal:  # Episode ends
            node = random.choice(list(all_nodes))
        else:
            node = new_node


def router(start, goal, max_steps=100):
    """
    Given a start, end, and q values, return the path taken by the robot.
    """
    q_matrix = np.zeros(shape=(NUM_DEST, NUM_DEST))
    learner(goal=goal, rewards=REWARDS, q_matrix=q_matrix)

    node = state_to_index[start]
    goal = state_to_index[goal]
    path = [node]

    for _ in range(max_steps):
        next_node = np.argmax(q_matrix[node])
        if next_node == node:  # Stuck in place (no improvement)
            break
        path.append(next_node)
        if next_node == goal:
            break
        node = next_node
    else:
        print("Max steps exceeded. Possible loop or poor training.")
        return [index_to_state[i] for i in path]

    return [index_to_state[i] for i in path]


if __name__ == "__main__":
    print("Stop at K")
    print(router(start="E", goal="K")[:-1] + router(start="K", goal="G"))
    print("Stop at F")
    print(router(start="E", goal="F")[:-1] + router(start="F", goal="G"))
