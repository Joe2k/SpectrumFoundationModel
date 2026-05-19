"""Transformer building blocks."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, causal: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        ctx = x if context is None else context
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(ctx).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(ctx).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        if self.causal and context is None:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(y)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, causal=False)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = MultiHeadAttention(dim, n_heads, causal=True)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, n_heads, causal=False)
        self.norm3 = RMSNorm(dim)
        self.ff = FeedForward(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.norm1(x)))
        x = x + self.drop(self.cross_attn(self.norm2(x), context=memory))
        x = x + self.drop(self.ff(self.norm3(x)))
        return x
