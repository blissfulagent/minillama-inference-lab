import torch


def precompute_rope_cache(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    # freq_i = 1 / theta^(2i / head_dim), i in [0, head_dim//2)
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [max_seq_len, half]
    return angles.cos(), angles.sin()       # each [max_seq_len, half]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: [..., head_dim] — rotate pairs (x0, x1) -> (-x1, x0)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # x: [B, n_heads, T, head_dim]
    # cos/sin: [T, head_dim//2]  -> broadcast to [1, 1, T, head_dim] via repeat
    head_dim = x.shape[-1]
    # Expand cos/sin to full head_dim by repeating: [T, head_dim//2] -> [T, head_dim]
    cos_full = cos.repeat(1, 2).unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
    sin_full = sin.repeat(1, 2).unsqueeze(0).unsqueeze(0)
    return x * cos_full + _rotate_half(x) * sin_full
