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
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.hidden_dim > self.dim, "hidden_dim must be greater than dim"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
