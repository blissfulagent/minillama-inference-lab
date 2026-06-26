import torch
import torch.nn as nn

from .config import ModelConfig
from .rmsnorm import RMSNorm
from .attention import Attention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if return_kv:
            attn_out, kv = self.attention(
                self.attention_norm(x), cos, sin, mask, past_kv=past_kv, return_kv=True
            )
            h = x + attn_out
            return h + self.feed_forward(self.ffn_norm(h)), kv
        h = x + self.attention(self.attention_norm(x), cos, sin, mask, past_kv=past_kv)
        return h + self.feed_forward(self.ffn_norm(h))
