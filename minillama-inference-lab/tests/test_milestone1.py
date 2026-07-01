import warnings

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
# Model — shifted next-token loss
# ---------------------------------------------------------------------------

def test_loss_ignores_first_target_token():
    # logits[:, :-1] predicts targets[:, 1:], so the first target position
    # (nothing precedes it to predict it) is never used by the loss.
    config = ModelConfig()
    torch.manual_seed(0)
    model = MiniLLaMA(config)
    tokens = torch.randint(0, config.vocab_size, (1, 8))
    targets_a = torch.randint(0, config.vocab_size, (1, 8))
    targets_b = targets_a.clone()
    targets_b[0, 0] = (targets_b[0, 0] + 1) % config.vocab_size

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss_a = model(tokens, targets=targets_a)["loss"]
        loss_b = model(tokens, targets=targets_b)["loss"]

    assert torch.allclose(loss_a, loss_b), "Changing only the first target must not change shifted loss"


def test_loss_seq_len_one_with_targets_raises():
    config = ModelConfig()
    model = MiniLLaMA(config)
    tokens = torch.randint(0, config.vocab_size, (1, 1))
    targets = torch.randint(0, config.vocab_size, (1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="sequence length"):
            model(tokens, targets=targets)


def test_loss_uses_shifted_target_tokens():
    config = ModelConfig()
    torch.manual_seed(0)
    model = MiniLLaMA(config)
    tokens = torch.randint(0, config.vocab_size, (1, 8))
    targets_a = torch.randint(0, config.vocab_size, (1, 8))
    targets_c = targets_a.clone()
    targets_c[0, 1] = (targets_c[0, 1] + 1) % config.vocab_size

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss_a = model(tokens, targets=targets_a)["loss"]
        loss_c = model(tokens, targets=targets_c)["loss"]

    assert not torch.allclose(loss_a, loss_c), "Changing a used target position should change the loss"


# ---------------------------------------------------------------------------
# Model — sequence-limit validation
# ---------------------------------------------------------------------------

def test_start_pos_plus_seq_len_exceeds_max_seq_len_raises():
    config = ModelConfig(max_seq_len=16)
    model = MiniLLaMA(config)
    tokens = torch.randint(0, config.vocab_size, (1, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model(tokens, start_pos=14)  # 14 + 4 > 16


def test_negative_start_pos_raises():
    config = ModelConfig(max_seq_len=16)
    model = MiniLLaMA(config)
    tokens = torch.randint(0, config.vocab_size, (1, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="start_pos must be >= 0"):
            model(tokens, start_pos=-1)


# ---------------------------------------------------------------------------
# Model — malformed kv_caches length
# ---------------------------------------------------------------------------

def test_kv_caches_wrong_length_raises():
    config = ModelConfig(n_layers=2)
    model = MiniLLaMA(config)
    tokens = torch.randint(0, config.vocab_size, (1, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        prefill = model(tokens, kv_caches=[], start_pos=0)
        bad_kv_caches = prefill["kv_caches"][:1]  # drop one layer's cache
        assert len(bad_kv_caches) != config.n_layers
        with pytest.raises(ValueError, match="one entry per layer"):
            model(torch.randint(0, config.vocab_size, (1, 1)), kv_caches=bad_kv_caches, start_pos=4)


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
    with pytest.raises(ValueError):
        ModelConfig(dim=128, n_heads=3)

def test_config_validation_heads_divisible_by_kv_heads():
    with pytest.raises(ValueError):
        ModelConfig(n_heads=4, n_kv_heads=3)

def test_config_validation_hidden_dim_greater_than_dim():
    with pytest.raises(ValueError):
        ModelConfig(dim=128, hidden_dim=64)

def test_config_validation_n_layers_positive():
    with pytest.raises(ValueError):
        ModelConfig(n_layers=0)

def test_config_validation_n_heads_positive():
    with pytest.raises(ValueError):
        ModelConfig(n_heads=0)

def test_config_validation_n_kv_heads_positive():
    with pytest.raises(ValueError):
        ModelConfig(n_kv_heads=0)

def test_config_validation_head_dim_even():
    # dim=96, n_heads=32 -> head_dim=3 (odd), also fails divisibility of dim/n_heads? 96/32=3 ok divisible
    with pytest.raises(ValueError):
        ModelConfig(dim=96, n_heads=32, n_kv_heads=32, hidden_dim=192)

def test_config_validation_dropout_range():
    with pytest.raises(ValueError):
        ModelConfig(dropout=1.0)
    with pytest.raises(ValueError):
        ModelConfig(dropout=-0.1)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def test_load_checkpoint_marks_weights_loaded(tmp_path):
    config = ModelConfig()
    model = MiniLLaMA(config)
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), ckpt_path)

    fresh_model = MiniLLaMA(config)
    assert fresh_model._weights_loaded is False
    fresh_model.load_checkpoint(str(ckpt_path))
    assert fresh_model._weights_loaded is True

    tokens = torch.randint(0, config.vocab_size, (1, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fresh_model(tokens)  # should not warn about random weights


@pytest.mark.parametrize("wrapper_key", ["model_state_dict", "state_dict", "model"])
def test_load_checkpoint_supports_wrapped_formats(tmp_path, wrapper_key):
    config = ModelConfig()
    model = MiniLLaMA(config)
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({wrapper_key: model.state_dict()}, ckpt_path)

    fresh_model = MiniLLaMA(config)
    assert fresh_model._weights_loaded is False
    fresh_model.load_checkpoint(str(ckpt_path))
    assert fresh_model._weights_loaded is True
