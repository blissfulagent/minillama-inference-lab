LAUDE.md

This file gives Claude Code stable instructions for this repository.

Repository Overview

This repository contains one primary project:

minillama-inference-lab/

The goal is to build a CPU-compatible MiniLLaMA-style inference lab in PyTorch.

This is not real LLaMA 3 training. This project focuses on implementing and benchmarking a compact LLaMA-style decoder-only Transformer.

Project Scope

Core features:

byte-level tokenizer
token embeddings
RMSNorm
RoPE
causal self-attention
MHA and GQA
SwiGLU feed-forward network
autoregressive generation
greedy decoding
temperature sampling
top-k sampling
top-p sampling
repetition penalty
KV-cache inference
KV-cache vs non-cache benchmark
MHA vs GQA benchmark
FastAPI inference endpoint
Non-Goals

Do not implement these unless explicitly requested later:

real LLaMA 3 weight loading
Hugging Face model loading
large-scale training
distributed training
RLHF
instruction tuning
LoRA
quantization
FlashAttention
custom CUDA kernels
database
authentication
Docker
frontend before backend works
Development Rules
Work inside minillama-inference-lab/.
Read SPEC.md before implementation.
Implement one milestone at a time.
Do not implement future milestones early.
Keep everything CPU-compatible.
Do not assume CUDA exists.
Keep model defaults tiny.
Do not create large datasets, checkpoints, or generated artifacts.
Do not add heavy dependencies.
Do not invent benchmark results.
Do not claim random-weight generation is meaningful.
Prefer clear PyTorch code over clever abstractions.
Run relevant tests after each milestone.
Do not rewrite working files unnecessarily.
Important Technical Rules
Token IDs are integers.
Embeddings are vectors selected by token IDs.
RoPE applies only to Q and K, not V.
Temperature must divide logits before softmax.
KV cache stores compact K/V, not repeated GQA K/V.
KV cache improves speed; it should not change mathematical output except tiny floating-point differences.
If no checkpoint is loaded, print a warning that random generated text is not meaningful.
Commands

Install dependencies:

pip install -r requirements.txt

Run tests:

pytest

Run generation:

python scripts/run_generate.py --prompt "Once upon a time" --max_new_tokens 32 --use_kv_cache

Run API:

uvicorn api.main:app --reload

Run benchmarks:

python benchmarks/benchmark_kv_cache.py
python benchmarks/benchmark_gqa_vs_mha.py
Recommended Claude Code Effort

Use medium effort for normal implementation.

Use high effort only for:

RoPE bugs
attention shape bugs
GQA bugs
KV-cache bugs
final review before commit

Avoid max unless explicitly requested.