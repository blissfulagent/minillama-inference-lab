import json
import os
import statistics
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minillama.config import ModelConfig
from minillama.model import MiniLLaMA
from minillama.tokenizer import ByteTokenizer
from minillama.generation import generate

PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 32
N_RUNS = 3
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def kv_cache_elements(config: ModelConfig, prompt_token_count: int, max_new_tokens: int) -> int:
    head_dim = config.dim // config.n_heads
    total_seq_len = prompt_token_count + max_new_tokens
    return config.n_layers * 2 * 1 * total_seq_len * config.n_kv_heads * head_dim


def run_config(config: ModelConfig, tokenizer: ByteTokenizer, label: str) -> dict:
    model = MiniLLaMA(config)
    model.eval()

    param_count = count_parameters(model)
    prompt_token_count = len(tokenizer.encode(PROMPT))
    kv_elements = kv_cache_elements(config, prompt_token_count, MAX_NEW_TOKENS)

    latencies = []
    tps_list = []

    for _ in range(N_RUNS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = generate(
                model,
                tokenizer,
                PROMPT,
                max_new_tokens=MAX_NEW_TOKENS,
                greedy=True,
                use_kv_cache=True,
            )
        latencies.append(result["latency_ms"])
        tps_list.append(result["tokens_per_second"])

    return {
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "param_count": param_count,
        "kv_cache_elements": kv_elements,
        "latency_ms": round(statistics.median(latencies), 3),
        "tokens_per_second": round(statistics.median(tps_list), 3),
        "runs_latency_ms": [round(v, 3) for v in latencies],
    }


def main():
    tokenizer = ByteTokenizer()

    mha_config = ModelConfig(n_heads=4, n_kv_heads=4)
    gqa_config = ModelConfig(n_heads=4, n_kv_heads=2)

    print("Running GQA vs MHA benchmark (3 runs each, greedy decoding, KV cache on)...")
    print(f"  Prompt: {PROMPT!r}  |  max_new_tokens={MAX_NEW_TOKENS}")
    print()

    print("  [1/2] MHA (n_kv_heads=4)...")
    mha = run_config(mha_config, tokenizer, "MHA")

    print("  [2/2] GQA (n_kv_heads=2)...")
    gqa = run_config(gqa_config, tokenizer, "GQA")

    output = {
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "mha": mha,
        "gqa": gqa,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "gqa_vs_mha_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"{'Config':<8} {'n_kv_heads':>12} {'Params':>12} {'KV Elements':>14} {'Latency (ms)':>14} {'Tokens/s':>10}")
    print("-" * 76)
    print(f"{'MHA':<8} {mha['n_kv_heads']:>12} {mha['param_count']:>12,} {mha['kv_cache_elements']:>14,} {mha['latency_ms']:>14.1f} {mha['tokens_per_second']:>10.1f}")
    print(f"{'GQA':<8} {gqa['n_kv_heads']:>12} {gqa['param_count']:>12,} {gqa['kv_cache_elements']:>14,} {gqa['latency_ms']:>14.1f} {gqa['tokens_per_second']:>10.1f}")
    print()
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
