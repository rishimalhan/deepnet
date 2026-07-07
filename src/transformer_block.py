#! /usr/bin/python3

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._attn_out = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self._ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
        )

        self._dropout = nn.Dropout(p=dropout)

        self._norm1 = nn.LayerNorm(d_model)
        self._norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        causal_mask: torch.Tensor = None,
    ):
        """
        x: [B, T, D]
        key_padding_mask: [B, T]
            True means ignore this token.
        """

        attn_out, _ = self._attn_out(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + self._dropout(attn_out)
        x = self._norm1(x)
        ffn_out = self._ffn(x)
        x = x + self._dropout(ffn_out)
        x = self._norm2(x)
        return x
