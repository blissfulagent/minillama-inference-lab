import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minillama.config import ModelConfig
from minillama.model import MiniLLaMA
from minillama.tokenizer import ByteTokenizer
from minillama.generation import generate


def main():
    parser = argparse.ArgumentParser(description="MiniLLaMA text generation")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0, help="Top-k filtering (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus filtering (1.0 = disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty (1.0 = none)")
    parser.add_argument("--use_kv_cache", action="store_true", help="Enable KV cache for faster inference")
    args = parser.parse_args()

    config = ModelConfig()
    model = MiniLLaMA(config)
    tokenizer = ByteTokenizer()

    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        use_kv_cache=args.use_kv_cache,
    )

    print(f"\n--- Generated Text ---")
    print(result["text"])
    print(f"\n--- Stats ---")
    print(f"tokens_generated : {result['tokens_generated']}")
    print(f"latency_ms       : {result['latency_ms']:.2f} ms")
    print(f"tokens_per_second: {result['tokens_per_second']:.2f}")
    print(f"used_kv_cache    : {result['used_kv_cache']}")


if __name__ == "__main__":
    main()
