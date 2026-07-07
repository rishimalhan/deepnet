#! /usr/bin/python3

from pathlib import Path
import pandas
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class RobotBCDataset(Dataset):

    def __init__(self, csv_path: str, seq_len: int = 10):

        self._csv_path = Path(csv_path)
        self._seq_len = seq_len

        df = pandas.read_csv(self._csv_path)

        self.state_cols = [
            "q0",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "dq0",
            "dq1",
            "dq2",
            "dq3",
            "dq4",
            "dq5",
            "dq6",
            "ee_x",
            "ee_y",
            "ee_z",
            "ee_roll",
            "ee_pitch",
            "ee_yaw",
            "object_x",
            "object_y",
            "object_z",
            "target_x",
            "target_y",
            "target_z",
            "gripper_width",
        ]

        self.action_cols = [
            "action_dx",
            "action_dy",
            "action_dz",
            "action_droll",
            "action_dpitch",
            "action_dyaw",
            "action_gripper",
        ]

        self.episodes = []
        self.windows = []

        for episode_idx, ep_df in df.groupby("episode"):
            ep_df = ep_df.sort_values("t").reset_index(drop=True)
            states = ep_df[self.state_cols].to_numpy(dtype=np.float32)
            actions = ep_df[self.action_cols].to_numpy(dtype=np.float32)
            ep_idx = len(self.episodes)
            self.episodes.append((states, actions))

            for end in range(seq_len - 1, len(ep_df)):
                start = end - seq_len + 1
                self.windows.append((ep_idx, start, end))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ep_idx, start, end = self.windows[idx]
        states, actions = self.episodes[ep_idx]
        x = states[start : end + 1]  # T, state_dim
        y = actions[end]  # action_dim

        return torch.from_numpy(x), torch.from_numpy(y)


def make_dataloader(
    csv_path="data/robot_bc_grasps.csv",
    seq_len=10,
    batch_size=64,
    num_workers=4,
):
    dataset = RobotBCDataset(csv_path=csv_path, seq_len=seq_len)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return dataset, loader
