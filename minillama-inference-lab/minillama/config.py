from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 256
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    hidden_dim: int = 384
    max_seq_len: int = 128
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {self.n_heads}")
        if self.n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be > 0, got {self.n_kv_heads}")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        head_dim = self.dim // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim (dim // n_heads = {head_dim}) must be even for RoPE")
        if self.hidden_dim <= self.dim:
            raise ValueError("hidden_dim must be greater than dim")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
