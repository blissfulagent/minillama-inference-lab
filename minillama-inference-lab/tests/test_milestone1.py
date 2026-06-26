import pytest
import torch

from minillama.config import ModelConfig
from minillama.tokenizer import ByteTokenizer
from minillama.rope import precompute_rope_cache, apply_rope
from minillama.attention import Attention
from minillama.model import MiniLLaMA


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def test_tokenizer_encode_decode_roundtrip():
    tok = ByteTokenizer()
    text = "Hello, world! 123"
    ids = tok.encode(text)
    assert all(0 <= i < 256 for i in ids), "All token IDs must be in [0, 255]"
    assert tok.decode(ids) == text

def test_tokenizer_encode_produces_ints():
    tok = ByteTokenizer()
    ids = tok.encode("abc")
    assert ids == [97, 98, 99]

def test_tokenizer_vocab_size():
    assert ByteTokenizer.vocab_size == 256


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def test_rope_cache_shape():
    head_dim = 32
    max_seq_len = 128
    cos, sin = precompute_rope_cache(head_dim, max_seq_len, theta=10000.0)
    assert cos.shape == (max_seq_len, head_dim // 2), f"cos shape: {cos.shape}"
    assert sin.shape == (max_seq_len, head_dim // 2), f"sin shape: {sin.shape}"

def test_rope_apply_preserves_shape():
    head_dim = 32
    cos, sin = precompute_rope_cache(head_dim, max_seq_len=128, theta=10000.0)
    q = torch.randn(1, 4, 8, head_dim)
    q_rot = apply_rope(q, cos[:8], sin[:8])
    assert q_rot.shape == q.shape, f"RoPE output shape mismatch: {q_rot.shape} vs {q.shape}"

def test_rope_does_not_change_norm():
    # RoPE is a rotation so ||q_rot|| ≈ ||q||
    head_dim = 32
    cos, sin = precompute_rope_cache(head_dim, max_seq_len=128, theta=10000.0)
    q = torch.randn(2, 4, 16, head_dim)
    q_rot = apply_rope(q, cos[:16], sin[:16])
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5), "RoPE should preserve L2 norm"


# ---------------------------------------------------------------------------
# Attention — MHA (n_kv_heads == n_heads)
# ---------------------------------------------------------------------------

def test_attention_mha_output_shape():
    config = ModelConfig(n_heads=4, n_kv_heads=4)
    attn = Attention(config)
    head_dim = config.dim // config.n_heads
    cos, sin = precompute_rope_cache(head_dim, config.max_seq_len, config.rope_theta)
    x = torch.randn(2, 6, config.dim)
    out = attn(x, cos[:6], sin[:6])
    assert out.shape == (2, 6, config.dim), f"MHA output shape: {out.shape}"


# ---------------------------------------------------------------------------
# Attention — GQA (n_kv_heads < n_heads)
# ---------------------------------------------------------------------------

def test_attention_gqa_output_shape():
    config = ModelConfig(n_heads=4, n_kv_heads=2)
    attn = Attention(config)
    head_dim = config.dim // config.n_heads
    cos, sin = precompute_rope_cache(head_dim, config.max_seq_len, config.rope_theta)
    x = torch.randn(2, 6, config.dim)
    out = attn(x, cos[:6], sin[:6])
    assert out.shape == (2, 6, config.dim), f"GQA output shape: {out.shape}"


# ---------------------------------------------------------------------------
# Model forward — logits shape, no loss
# ---------------------------------------------------------------------------

def test_model_forward_logits_shape():
    config = ModelConfig()
    model = MiniLLaMA(config)
    tokens = torch.randint(0, config.vocab_size, (1, 8))
    with pytest.warns(UserWarning, match="random weights"):
        result = model(tokens)
    assert "logits" in result
    assert "loss" in result
    assert "kv_caches" in result
    assert result["logits"].shape == (1, 8, config.vocab_size), f"logits shape: {result['logits'].shape}"
    assert result["loss"] is None
    assert result["kv_caches"] is None


# ---------------------------------------------------------------------------
# Model forward — loss when targets provided
# ---------------------------------------------------------------------------

def test_model_forward_with_loss():
    config = ModelConfig()
    model = MiniLLaMA(config)
    tokens = torch.randint(0, config.vocab_size, (1, 8))
    targets = torch.randint(0, config.vocab_size, (1, 8))
    with pytest.warns(UserWarning, match="random weights"):
        result = model(tokens, targets=targets)
    assert result["loss"] is not None, "Loss should not be None when targets provided"
    assert result["loss"].item() > 0, "Loss should be positive"
    assert result["logits"].shape == (1, 8, config.vocab_size)


# ---------------------------------------------------------------------------
# Model — causal mask sanity
# ---------------------------------------------------------------------------

def test_model_forward_batch():
    config = ModelConfig()
    model = MiniLLaMA(config)
    tokens = torch.randint(0, config.vocab_size, (3, 16))
    with pytest.warns(UserWarning, match="random weights"):
        result = model(tokens)
    assert result["logits"].shape == (3, 16, config.vocab_size)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_validation_dim_divisible_by_heads():
    with pytest.raises(AssertionError):
        ModelConfig(dim=128, n_heads=3)

def test_config_validation_heads_divisible_by_kv_heads():
    with pytest.raises(AssertionError):
        ModelConfig(n_heads=4, n_kv_heads=3)

def test_config_validation_hidden_dim_greater_than_dim():
    with pytest.raises(AssertionError):
        ModelConfig(dim=128, hidden_dim=64)
