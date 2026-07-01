import time
import warnings

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    top_k_values, _ = torch.topk(logits, min(k, logits.size(-1)))
    threshold = top_k_values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens beyond the nucleus (shift right so the pivot token itself is kept)
    sorted_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits = sorted_logits.masked_fill(sorted_remove, float("-inf"))
    # Restore original order
    result = torch.zeros_like(logits)
    result.scatter_(-1, sorted_indices, sorted_logits)
    return result


def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: list[int], penalty: float
) -> torch.Tensor:
    if penalty == 1.0 or not generated_ids:
        return logits
    unique_ids = list(set(generated_ids))
    for token_id in unique_ids:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def sample_token(
    logits: torch.Tensor,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    # logits shape: [vocab_size]
    if greedy:
        return int(logits.argmax().item())

    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    use_kv_cache: bool = False,
) -> dict:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")
    if repetition_penalty < 1.0:
        raise ValueError(f"repetition_penalty must be >= 1, got {repetition_penalty}")

    if not model._weights_loaded:
        warnings.warn(
            "No checkpoint loaded — model uses random weights. Generated text is not meaningful.",
            UserWarning,
            stacklevel=2,
        )
        model._warned_random = True  # prevent model.forward() from repeating the warning

    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    max_seq_len = model.config.max_seq_len

    available = max_seq_len - len(prompt_ids)
    if available <= 0:
        return {
            "text": prompt,
            "token_ids": prompt_ids,
            "tokens_generated": 0,
            "latency_ms": 0.0,
            "tokens_per_second": 0.0,
            "used_kv_cache": use_kv_cache,
        }

    if max_new_tokens > available:
        warnings.warn(
            f"max_new_tokens ({max_new_tokens}) clamped to {available} due to max_seq_len={max_seq_len}.",
            UserWarning,
            stacklevel=2,
        )
        max_new_tokens = available

    if max_new_tokens == 0:
        return {
            "text": tokenizer.decode(prompt_ids),
            "token_ids": prompt_ids,
            "tokens_generated": 0,
            "latency_ms": 0.0,
            "tokens_per_second": 0.0,
            "used_kv_cache": use_kv_cache,
        }

    generated_ids: list[int] = []

    t_start = time.perf_counter()

    with torch.no_grad():
        if use_kv_cache:
            _generate_kv_cache(
                model, tokenizer, prompt_ids, max_new_tokens,
                greedy, temperature, top_k, top_p, repetition_penalty,
                generated_ids,
            )
        else:
            _generate_no_cache(
                model, tokenizer, prompt_ids, max_new_tokens,
                greedy, temperature, top_k, top_p, repetition_penalty,
                generated_ids,
            )

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0
    tokens_generated = len(generated_ids)
    tokens_per_second = tokens_generated / (t_end - t_start) if (t_end - t_start) > 0 else 0.0

    all_ids = prompt_ids + generated_ids
    generated_text = tokenizer.decode(all_ids)

    return {
        "text": generated_text,
        "token_ids": all_ids,
        "tokens_generated": tokens_generated,
        "latency_ms": latency_ms,
        "tokens_per_second": tokens_per_second,
        "used_kv_cache": use_kv_cache,
    }


def _generate_no_cache(
    model, tokenizer, prompt_ids, max_new_tokens,
    greedy, temperature, top_k, top_p, repetition_penalty,
    generated_ids,
):
    device = next(model.parameters()).device
    current_ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        tokens = torch.tensor([current_ids], dtype=torch.long, device=device)
        out = model(tokens)
        logits = out["logits"][0, -1, :].clone()  # [vocab_size]

        all_so_far = current_ids + generated_ids
        logits = apply_repetition_penalty(logits, all_so_far, repetition_penalty)

        next_id = sample_token(logits, greedy, temperature, top_k, top_p)
        generated_ids.append(next_id)
        current_ids.append(next_id)


def _generate_kv_cache(
    model, tokenizer, prompt_ids, max_new_tokens,
    greedy, temperature, top_k, top_p, repetition_penalty,
    generated_ids,
):
    device = next(model.parameters()).device

    # Prefill: run full prompt, collect KV caches
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model(prompt_tensor, kv_caches=[], start_pos=0)
    kv_caches = out["kv_caches"]

    logits = out["logits"][0, -1, :].clone()  # [vocab_size]
    logits = apply_repetition_penalty(logits, prompt_ids, repetition_penalty)
    next_id = sample_token(logits, greedy, temperature, top_k, top_p)
    generated_ids.append(next_id)

    prompt_len = len(prompt_ids)

    # Decode: one token at a time
    for step in range(1, max_new_tokens):
        token_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        start_pos = prompt_len + step - 1
        out = model(token_tensor, kv_caches=kv_caches, start_pos=start_pos)
        kv_caches = out["kv_caches"]

        logits = out["logits"][0, 0, :].clone()  # [vocab_size]
        all_so_far = prompt_ids + generated_ids
        logits = apply_repetition_penalty(logits, all_so_far, repetition_penalty)

        next_id = sample_token(logits, greedy, temperature, top_k, top_p)
        generated_ids.append(next_id)
