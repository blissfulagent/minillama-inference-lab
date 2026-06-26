# MiniLLaMA Inference Lab — Compact Specification

## Goal

Build a CPU-compatible MiniLLaMA-style inference lab in PyTorch.

This project is focused on architecture and inference benchmarking, not real LLaMA 3 training.

## MVP Features

Implement:

* ByteTokenizer with vocab size 256
* ModelConfig dataclass
* RMSNorm
* RoPE
* causal self-attention
* MHA and GQA
* SwiGLU FFN
* TransformerBlock
* MiniLLaMA model
* greedy generation
* temperature sampling
* top-k sampling
* top-p sampling
* repetition penalty
* KV-cache generation
* benchmark for KV cache vs no cache
* benchmark for MHA vs GQA
* FastAPI `/generate` endpoint

## Default CPU-Safe Config

```python
vocab_size = 256
dim = 128
n_layers = 2
n_heads = 4
n_kv_heads = 2
hidden_dim = 384
max_seq_len = 128
rope_theta = 10000.0
dropout = 0.0
tie_embeddings = True
```

Validation:

* `dim % n_heads == 0`
* `n_heads % n_kv_heads == 0`
* `hidden_dim > dim`
* `vocab_size > 0`
* `max_seq_len > 0`

## Repository Structure

```text
minillama-inference-lab/
├── minillama/
├── tests/
├── scripts/
├── benchmarks/
├── api/
├── checkpoints/
├── SPEC.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Milestone 1 — Core Model

Files:

* `minillama/config.py`
* `minillama/tokenizer.py`
* `minillama/rmsnorm.py`
* `minillama/rope.py`
* `minillama/attention.py`
* `minillama/feedforward.py`
* `minillama/block.py`
* `minillama/model.py`

Acceptance:

* model forward pass works
* logits shape is `[batch, seq_len, vocab_size]`
* loss computes when targets are provided
* MHA and GQA shape tests pass
* RoPE shape tests pass

## Milestone 2 — Generation

Files:

* `minillama/generation.py`
* `scripts/run_generate.py`

Acceptance:

* greedy generation runs
* sampling generation runs
* temperature divides logits before softmax
* top-k works
* top-p works
* repetition penalty works
* KV-cache generation runs
* no-cache generation runs
* if no checkpoint is loaded, print random-weights warning

## Milestone 3 — Benchmarks

Files:

* `benchmarks/benchmark_kv_cache.py`
* `benchmarks/benchmark_gqa_vs_mha.py`

Acceptance:

* benchmark scripts produce JSON files in `benchmarks/results/`
* KV-cache benchmark compares `use_kv_cache=True` vs `False`
* GQA/MHA benchmark compares `n_kv_heads=2` vs `n_kv_heads=4`
* benchmark numbers are measured, not invented

## Milestone 4 — FastAPI

Files:

* `api/main.py`
* `api/schemas.py`
* `api/inference_service.py`

Routes:

* `GET /health`
* `GET /model/info`
* `POST /generate`

Acceptance:

* model loads once at startup
* `/generate` returns generated text, latency, tokens/sec, and cache mode
* CPU works by default

## Milestone 5 — README

README must include:

* what this project is
* what this project is not
* setup commands
* generation command
* API command
* benchmark commands
* limitations
* honest random-weights warning

## Non-Goals

Do not implement unless explicitly requested:

* real LLaMA 3 weights
* Hugging Face model loading
* large training
* LoRA
* quantization
* FlashAttention
* CUDA kernels
* database
* frontend
* Docker

## Done Definition

The MVP is done when:

* tests pass
* generation works with and without KV cache
* MHA and GQA both work
* benchmark scripts produce JSON
* FastAPI `/generate` works
* README honestly explains limitations
