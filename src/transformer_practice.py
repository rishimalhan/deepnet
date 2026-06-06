import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import math

x_dataset = torch.randn(1000, 4)
y_dataset = torch.randn(1000, 1, dtype=torch.int64)
dataset = TensorDataset(x_dataset, y_dataset)
loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)


def single_head_attention(x, W_q, W_k, W_v):
    """
    x: (B, T, embed_dim)
    W_q: nn.Linear(embed_dim, d_q)
    W_k: nn.Linear(embed_dim, d_k)
    W_v: nn.Linear(embed_dim, d_v)
    nn.Linear always operates on the last dimension and treats all leading dimensions as batch.
    It doesn’t “ignore” B,T – it just vectorizes over them.
    usually d_v = d_k

    In multi-head attention (MHA):
        •	We run several heads in parallel:
        •	each head produces (B, T, d_head)
        •	Then we concatenate heads on the last dim:
        •	(B, T, num_heads * d_head) = (B, T, d_model)
    Assume:
        •	d_model = 256
        •	num_heads = 4
        •	d_head = d_model // num_heads = 64
        •	x: (B, T, 256)
    """

    Q = W_q(x)
    K = W_k(x)
    V = W_v(x)

    d_k = K.size(-1)

    scores = (
        Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    )  # (B, T, d_q) @ (d_k, T) = (B, T, T)
    weights = F.softmax(scores, dim=-1)  # (B, T, T)
    out = weights @ V  # (B, T, T) @ (B, T, d_v) = (B, T, d_v)
    return out


def map_back(out):
    d_v = out.size(-1)
    d_model = 128
    W_o = nn.Linear(d_v, d_model)
    attn_out = W_o(out)  # (B, T, d_model)
    return attn_out
