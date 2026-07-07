#! /usr/bin/python3

"""
Given q, dq, gripper, object pose history
predict robot action

i.e. states: B x T x state_dim
actions: B x action_dim
"""

import os
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_block import TransformerBlock
from robot_bc_dataset import make_dataloader

curr_dir = os.getcwd()
data_root = os.path.join(curr_dir, "data")
model_root = os.path.join(curr_dir, "model")
device = "cuda" if torch.cuda.is_available() else "cpu"


class RobotTransformerPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        super().__init__()

        self.state_proj = nn.Linear(in_features=state_dim, out_features=d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        self._blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=4 * d_model,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self._action_head = nn.Linear(in_features=d_model, out_features=action_dim)

    def forward(
        self,
        states: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        causal: bool = False,
    ):
        _, T, _ = states.shape
        x = self.state_proj(states)  # [B, T, D]
        x = x + self.pos_embedding[:, :T]  # [B, T, D]

        causal_mask = None
        if causal:
            causal_mask = torch.triu(
                input=torch.ones(T, T, device=states.device), diagonal=1
            ).bool()
        for block in self._blocks:
            x = block(x, key_padding_mask=key_padding_mask, causal_mask=causal_mask)

        final_token = x[:, -1, :]  # [B, D]

        action_pred = self._action_head(final_token)

        return action_pred


if __name__ == "__main__":
    dataset, data_loader = make_dataloader(
        csv_path=os.path.join(data_root, "robot_bc_grasp.csv"),
        seq_len=30,
        batch_size=2000,
        num_workers=4,
    )

    state_dim = len(dataset.state_cols)
    action_dim = len(dataset.action_cols)

    model = RobotTransformerPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=256,
        num_heads=8,
        num_layers=4,
        max_seq_len=50,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(1000):
        for states, actions in data_loader:
            states = states.to(device)
            actions = actions.to(device)
            pred_actions = model(states, causal=False)
            loss = F.mse_loss(pred_actions, actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if epoch % 100 == 0:
            print(f"EPOCH={epoch}, loss={loss.item():.8f}")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss.item(),
        },
        os.path.join(model_root, "robot_bc_checkpoint.pt"),
    )
