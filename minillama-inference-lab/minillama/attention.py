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
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        # Project and reshape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # [B, n_heads, T, head_dim]
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, n_kv_heads, T, head_dim]
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, n_kv_heads, T, head_dim]

        # Apply RoPE to newly computed Q and K only
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Concatenate past K/V (compact, pre-GQA-repeat) before expanding
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # [B, n_kv_heads, past+T, head_dim]
            v = torch.cat([past_v, v], dim=2)

        # GQA: repeat K and V to match n_heads (only for computation, never stored)
        if self.n_rep > 1:
            k_full = k.repeat_interleave(self.n_rep, dim=1)  # [B, n_heads, total_T, head_dim]
            v_full = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_full = k
            v_full = v

        # Scaled dot-product attention
        scores = torch.matmul(q, k_full.transpose(-2, -1)) / self.scale  # [B, n_heads, T, total_T]

        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)

        out = torch.matmul(scores, v_full)                 # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, -1)        # [B, T, n_heads * head_dim]
        result = self.wo(out)                               # [B, T, dim]

        if return_kv:
            return result, (k, v)  # k, v are compact [B, n_kv_heads, total_T, head_dim]
        return result
