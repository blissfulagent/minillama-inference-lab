import warnings

import pytest
import torch

from minillama.config import ModelConfig
from minillama.model import MiniLLaMA
from minillama.tokenizer import ByteTokenizer
from minillama.generation import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_repetition_penalty,
    generate,
)


def tiny_config():
    return ModelConfig(dim=64, n_heads=4, n_kv_heads=2, n_layers=2,
                       hidden_dim=128, max_seq_len=64)


def make_model():
    model = MiniLLaMA(tiny_config())
    return model


def suppress_random_warning():
    return warnings.catch_warnings()


# ── Sampling helper unit tests ──────────────────────────────────────────────

def test_temperature_scales_logits():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = apply_temperature(logits, 2.0)
    assert torch.allclose(result, logits / 2.0)


def test_top_k_keeps_k_tokens():
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    result = apply_top_k(logits, k=3)
    finite_count = (result != float("-inf")).sum().item()
    assert finite_count == 3


def test_top_k_zero_is_disabled():
    logits = torch.tensor([1.0, 2.0, 3.0])
    result = apply_top_k(logits, k=0)
    assert torch.equal(result, logits)


def test_top_p_keeps_nucleus():
    # Highly skewed distribution so that a single token exceeds 0.9 mass
    logits = torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0])
    result = apply_top_p(logits, p=0.9)
    # The dominant token must not be masked
    assert result[0] != float("-inf")


def test_top_p_one_is_disabled():
    logits = torch.tensor([1.0, 2.0, 3.0])
    result = apply_top_p(logits, p=1.0)
    assert torch.equal(result, logits)


def test_repetition_penalty_reduces_positive_logit():
    logits = torch.tensor([0.0, 2.0, -1.0])
    penalized = apply_repetition_penalty(logits.clone(), generated_ids=[1], penalty=2.0)
    assert penalized[1] < logits[1]


def test_repetition_penalty_one_is_no_op():
    logits = torch.tensor([1.0, 2.0, 3.0])
    result = apply_repetition_penalty(logits.clone(), generated_ids=[0, 1], penalty=1.0)
    assert torch.equal(result, logits)


# ── generate() integration tests ─────────────────────────────────────────────

@pytest.fixture
def model_and_tok():
    model = make_model()
    tokenizer = ByteTokenizer()
    return model, tokenizer


def test_random_weights_warning(model_and_tok):
    model, tokenizer = model_and_tok
    with pytest.warns(UserWarning, match="random weights"):
        generate(model, tokenizer, prompt="hi", max_new_tokens=2, greedy=True)


def test_metadata_keys(model_and_tok):
    model, tokenizer = model_and_tok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt="hi", max_new_tokens=2, greedy=True)
    expected_keys = {"text", "token_ids", "tokens_generated", "latency_ms", "tokens_per_second", "used_kv_cache"}
    assert set(result.keys()) == expected_keys


def test_greedy_returns_string(model_and_tok):
    model, tokenizer = model_and_tok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt="hello", max_new_tokens=4, greedy=True)
    assert isinstance(result["text"], str)
    assert result["tokens_generated"] == 4
    assert result["used_kv_cache"] is False


def test_sampling_returns_string(model_and_tok):
    model, tokenizer = model_and_tok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt="hello", max_new_tokens=4,
                          greedy=False, temperature=1.5)
    assert isinstance(result["text"], str)
    assert result["tokens_generated"] == 4


def test_generate_no_cache(model_and_tok):
    model, tokenizer = model_and_tok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt="abc", max_new_tokens=4,
                          greedy=True, use_kv_cache=False)
    assert result["used_kv_cache"] is False
    assert result["tokens_generated"] == 4
    assert result["latency_ms"] >= 0.0


def test_generate_with_kv_cache(model_and_tok):
    model, tokenizer = model_and_tok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt="abc", max_new_tokens=4,
                          greedy=True, use_kv_cache=True)
    assert result["used_kv_cache"] is True
    assert result["tokens_generated"] == 4
    assert result["latency_ms"] >= 0.0


def test_kv_cache_matches_no_cache(model_and_tok):
    model, tokenizer = model_and_tok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result_no_cache = generate(model, tokenizer, prompt="hello", max_new_tokens=4,
                                   greedy=True, use_kv_cache=False)
        result_kv_cache = generate(model, tokenizer, prompt="hello", max_new_tokens=4,
                                   greedy=True, use_kv_cache=True)
    assert result_no_cache["token_ids"] == result_kv_cache["token_ids"], (
        f"KV cache and no-cache disagree:\n  no_cache={result_no_cache['token_ids']}\n  kv_cache={result_kv_cache['token_ids']}"
    )


def test_token_ids_length(model_and_tok):
    model, tokenizer = model_and_tok
    prompt = "test"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt=prompt, max_new_tokens=4, greedy=True)
    prompt_len = len(tokenizer.encode(prompt))
    assert len(result["token_ids"]) == prompt_len + result["tokens_generated"]


def test_temperature_zero_raises(model_and_tok):
    model, tokenizer = model_and_tok
    with pytest.raises(ValueError, match="temperature"):
        generate(model, tokenizer, prompt="hi", max_new_tokens=2, temperature=0.0)


def test_max_new_tokens_zero(model_and_tok):
    model, tokenizer = model_and_tok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt="hi", max_new_tokens=0, greedy=True)
    assert result["tokens_generated"] == 0
    assert result["latency_ms"] == 0.0


def test_top_k_generation(model_and_tok):
    model, tokenizer = model_and_tok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt="hi", max_new_tokens=4,
                          greedy=False, temperature=1.0, top_k=5)
    assert result["tokens_generated"] == 4


def test_top_p_generation(model_and_tok):
    model, tokenizer = model_and_tok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt="hi", max_new_tokens=4,
                          greedy=False, temperature=1.0, top_p=0.9)
    assert result["tokens_generated"] == 4


def test_repetition_penalty_generation(model_and_tok):
    model, tokenizer = model_and_tok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt="hi", max_new_tokens=4,
                          greedy=False, temperature=1.0, repetition_penalty=1.3)
    assert result["tokens_generated"] == 4


# ── Input validation ────────────────────────────────────────────────────────

def test_empty_prompt_raises(model_and_tok):
    model, tokenizer = model_and_tok
    with pytest.raises(ValueError, match="prompt"):
        generate(model, tokenizer, prompt="", max_new_tokens=2, greedy=True)


def test_negative_max_new_tokens_raises(model_and_tok):
    model, tokenizer = model_and_tok
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(model, tokenizer, prompt="hi", max_new_tokens=-1, greedy=True)


def test_invalid_top_p_too_high_raises(model_and_tok):
    model, tokenizer = model_and_tok
    with pytest.raises(ValueError, match="top_p"):
        generate(model, tokenizer, prompt="hi", max_new_tokens=2, top_p=1.5)


def test_invalid_top_p_zero_raises(model_and_tok):
    model, tokenizer = model_and_tok
    with pytest.raises(ValueError, match="top_p"):
        generate(model, tokenizer, prompt="hi", max_new_tokens=2, top_p=0.0)


def test_invalid_top_k_raises(model_and_tok):
    model, tokenizer = model_and_tok
    with pytest.raises(ValueError, match="top_k"):
        generate(model, tokenizer, prompt="hi", max_new_tokens=2, top_k=-1)


def test_invalid_repetition_penalty_raises(model_and_tok):
    model, tokenizer = model_and_tok
    with pytest.raises(ValueError, match="repetition_penalty"):
        generate(model, tokenizer, prompt="hi", max_new_tokens=2, repetition_penalty=0.5)


# ── Device awareness ─────────────────────────────────────────────────────────

def test_generate_respects_model_device(model_and_tok):
    model, tokenizer = model_and_tok
    model.to("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt="hi", max_new_tokens=2, greedy=True)
    assert isinstance(result["text"], str)
    assert result["tokens_generated"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_generate_on_cuda_device(model_and_tok):
    model, tokenizer = model_and_tok
    model.to("cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = generate(model, tokenizer, prompt="hi", max_new_tokens=2,
                          greedy=True, use_kv_cache=True)
    assert result["tokens_generated"] == 2


# ── Sequence-limit validation ────────────────────────────────────────────────

def test_start_pos_plus_seq_len_exceeds_max_seq_len(model_and_tok):
    model, tokenizer = model_and_tok
    max_seq_len = model.config.max_seq_len
    tokens = torch.randint(0, model.config.vocab_size, (1, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(AssertionError, match="exceeds max_seq_len"):
            model(tokens, start_pos=max_seq_len - 1)
