import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)  # down
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) * up -> down
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
