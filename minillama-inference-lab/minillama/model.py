import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rmsnorm import RMSNorm
from .rope import precompute_rope_cache
from .block import TransformerBlock


class MiniLLaMA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        head_dim = config.dim // config.n_heads
        cos, sin = precompute_rope_cache(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos)  # [max_seq_len, head_dim//2]
        self.register_buffer("rope_sin", sin)

        self._weights_loaded = False

    def mark_weights_loaded(self):
        self._weights_loaded = True

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict:
        if not self._weights_loaded:
            warnings.warn(
                "No checkpoint loaded — model uses random weights. Generated text is not meaningful.",
                stacklevel=2,
            )

        B, T = tokens.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        x = self.tok_embeddings(tokens)  # [B, T, dim]

        cos = self.rope_cos[:T]  # [T, head_dim//2]
        sin = self.rope_sin[:T]

        # Causal mask: 0 on/below diagonal, -inf above diagonal
        mask = torch.full((1, 1, T, T), float("-inf"), device=x.device, dtype=x.dtype)
        mask = torch.tril(torch.zeros_like(mask)) + torch.triu(mask, diagonal=1)

        for layer in self.layers:
            x = layer(x, cos, sin, mask)

        x = self.norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1).long(),
            )

        return {"logits": logits, "loss": loss, "kv_caches": None}
