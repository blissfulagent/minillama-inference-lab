import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import apply_rope


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads  # GQA repeat factor
        self.scale = math.sqrt(self.head_dim)
        self.dropout = config.dropout

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Project and reshape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # [B, n_heads, T, head_dim]
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, n_kv_heads, T, head_dim]
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, n_kv_heads, T, head_dim]

        # Apply RoPE to Q and K only
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: repeat K and V to match n_heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # [B, n_heads, T, head_dim]
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, n_heads, T, T]

        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)

        out = torch.matmul(scores, v)                      # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, -1)        # [B, T, n_heads * head_dim]
        return self.wo(out)                                 # [B, T, dim]
