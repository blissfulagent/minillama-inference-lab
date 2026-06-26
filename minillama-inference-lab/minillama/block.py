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
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), cos, sin, mask)
        return h + self.feed_forward(self.ffn_norm(h))
