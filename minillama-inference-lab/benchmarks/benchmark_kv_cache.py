import json
import os
import statistics
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from minillama.config import ModelConfig
from minillama.model import MiniLLaMA
from minillama.tokenizer import ByteTokenizer
from minillama.generation import generate

PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 32
N_RUNS = 3
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_mode(model, tokenizer, use_kv_cache: bool) -> dict:
    latencies = []
    tps_list = []
    tokens_generated = None

    for _ in range(N_RUNS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = generate(
                model,
                tokenizer,
                PROMPT,
                max_new_tokens=MAX_NEW_TOKENS,
                greedy=True,
                use_kv_cache=use_kv_cache,
            )
        latencies.append(result["latency_ms"])
        tps_list.append(result["tokens_per_second"])
        tokens_generated = result["tokens_generated"]

    return {
        "latency_ms": round(statistics.median(latencies), 3),
        "tokens_generated": tokens_generated,
        "tokens_per_second": round(statistics.median(tps_list), 3),
        "used_kv_cache": use_kv_cache,
        "runs_latency_ms": [round(v, 3) for v in latencies],
    }


def main():
    config = ModelConfig()
    tokenizer = ByteTokenizer()
    model = MiniLLaMA(config)
    model.eval()

    print("Running KV-cache benchmark (3 runs each, greedy decoding)...")
    print(f"  Prompt: {PROMPT!r}  |  max_new_tokens={MAX_NEW_TOKENS}")
    print()

    print("  [1/2] No KV cache...")
    no_cache = run_mode(model, tokenizer, use_kv_cache=False)

    print("  [2/2] With KV cache...")
    kv_cache = run_mode(model, tokenizer, use_kv_cache=True)

    speedup = round(no_cache["latency_ms"] / kv_cache["latency_ms"], 4) if kv_cache["latency_ms"] > 0 else None

    output = {
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "no_cache": no_cache,
        "kv_cache": kv_cache,
        "speedup": speedup,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "kv_cache_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"{'Mode':<14} {'Latency (ms)':>14} {'Tokens/s':>12} {'KV Cache':>10}")
    print("-" * 54)
    print(f"{'no_cache':<14} {no_cache['latency_ms']:>14.1f} {no_cache['tokens_per_second']:>12.1f} {'No':>10}")
    print(f"{'kv_cache':<14} {kv_cache['latency_ms']:>14.1f} {kv_cache['tokens_per_second']:>12.1f} {'Yes':>10}")
    print()
    print(f"Speedup (no_cache / kv_cache latency): {speedup}")
    print(f"Note: speedup < 1 is expected on tiny CPU configs — KV cache overhead dominates at small scale.")
    print()
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
