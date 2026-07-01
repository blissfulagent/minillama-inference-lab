# MiniLLaMA Inference Lab

A CPU-first, LLaMA-style decoder-only Transformer built from scratch in PyTorch — implemented as an
architecture and inference-mechanics lab rather than a model you'd deploy.

---

## Overview

Most people who want to understand how LLaMA-style models actually work end up either reading a
paper or importing a Hugging Face model that hides every implementation detail behind
`from_pretrained()`. This project takes the middle path: it implements every core piece of a
LLaMA-style decoder — RMSNorm, RoPE, grouped-query attention, SwiGLU, KV-cache decoding — by hand,
in plain PyTorch, small enough to run and debug on a laptop CPU.

It's not trying to be a good language model. The default weights are random and untrained, so the
text it generates is architecturally correct but semantically meaningless. What it's actually for is
answering questions like: *does my KV cache produce the same output as no cache? how much does GQA
actually save over MHA? does RoPE only touch Q and K like it's supposed to?* — with real code and
real (if tiny) benchmark numbers instead of taking it on faith.

## What This Project Is Not

- **Not real LLaMA 3.** No Meta weights, no Hugging Face model loading.
- **Not meaningful output.** No trained checkpoint ships with the repo — generated text from the
  default model is random-weight noise, and the code warns about this explicitly.
- **Not a chatbot or a training framework.** No LoRA, no RLHF, no instruction tuning, no
  distributed training.

---

## Key Features

- Byte-level tokenizer (`vocab_size=256`, encodes/decodes raw UTF-8 bytes — no BPE, no vocab file)
- `RMSNorm` and rotary position embeddings (`RoPE`), applied only to Q and K, never V
- Causal self-attention supporting both Multi-Head Attention (MHA) and Grouped-Query Attention (GQA)
  via a configurable `n_kv_heads`
- SwiGLU feed-forward block (`silu(gate) * up -> down`)
- Autoregressive generation with greedy decoding, temperature, top-k, top-p (nucleus), and
  repetition penalty — composable in any combination
- KV-cache decoding that stores compact (non-repeated) K/V per layer and expands them for GQA only
  at attention time, plus a no-cache path for comparison
- Checkpoint loading (`MiniLLaMA.load_checkpoint`) if you bring your own trained `state_dict`
- FastAPI backend with `/health`, `/model/info`, and `/generate`
- Two benchmark scripts that measure (not assume) KV-cache speedup and MHA-vs-GQA cost, writing
  results to JSON

---

## Architecture / Workflow

**Model forward pass** (`minillama/model.py`):

1. Token IDs → `nn.Embedding` lookup → `[B, T, dim]`.
2. RoPE cos/sin tables are precomputed once at model construction (`precompute_rope_cache`) and
   sliced per forward call starting at `start_pos`, so cached decode steps get the correct absolute
   position.
3. Each of `n_layers` `TransformerBlock`s does: `RMSNorm -> Attention -> residual -> RMSNorm ->
   SwiGLU FFN -> residual`.
4. Inside `Attention`, Q/K/V are projected (K/V into fewer heads than Q when GQA is enabled), RoPE
   is applied to Q and K, past K/V are concatenated in if a KV cache was passed in, K/V are
   repeat-interleaved up to `n_heads` only for the score computation, then standard scaled
   dot-product attention with a causal mask runs.
5. Final `RMSNorm` + tied/untied `lm_head` linear produces `[B, T, vocab_size]` logits. If `targets`
   are passed, loss is a shifted next-token cross-entropy (`logits[:, :-1]` vs `targets[:, 1:]`).

**Generation** (`minillama/generation.py`) wraps the model in a decode loop:

- No-cache path re-runs the full sequence through the model every step.
- KV-cache path does one prefill pass over the prompt (collecting per-layer K/V), then feeds one
  token at a time with `start_pos` advancing and the running KV cache passed back in.
- Sampling (temperature → top-k → top-p → repetition penalty → multinomial, or plain argmax for
  greedy) happens on the logits of the last position, independent of which decode path was used.

**API layer** (`api/`): `inference_service.py` builds the model and tokenizer once at FastAPI
startup (loading a checkpoint from `MINILLAMA_CHECKPOINT_PATH` if set) and hands out singletons;
`main.py` wires `/health`, `/model/info`, and `/generate` on top of that, translating `ValueError`s
from `generate()` into `422` HTTP responses; `schemas.py` defines the Pydantic request/response
shapes.

**Benchmarks** (`benchmarks/`) call `generate()` directly with different flags (`use_kv_cache`,
`n_kv_heads`) and time it with `time.perf_counter`, writing latency/tokens-per-second to JSON.

There is no database, no frontend, and no auth layer — the API is the full extent of the "serving"
side of this project.

---

## Tech Stack

- **Language:** Python 3.10+
- **Model / tensors:** PyTorch (CPU by default, CUDA used automatically if `torch.cuda.is_available()`)
- **API:** FastAPI + Uvicorn (`standard` extras), Pydantic for request/response validation
- **Testing:** pytest
- **HTTP client (dev/test dependency):** httpx
- **Packaging:** setuptools via `pyproject.toml`, editable install, one console script
  (`minillama-generate`)

No database, no ORM, no frontend framework, no containerization — deliberately, per the project scope.

---

## Project Structure

```
minillama-inference-lab/
├── minillama/
│   ├── config.py         # ModelConfig dataclass + validation (dim/heads/vocab sanity checks)
│   ├── tokenizer.py       # ByteTokenizer — UTF-8 byte-level encode/decode, vocab_size=256
│   ├── rmsnorm.py         # RMSNorm layer
│   ├── rope.py            # RoPE cache precomputation + apply_rope (Q/K only)
│   ├── attention.py       # Causal self-attention, MHA/GQA, KV-cache concat + repeat_interleave
│   ├── feedforward.py     # SwiGLU FFN
│   ├── block.py           # TransformerBlock (norm -> attn -> residual -> norm -> ffn -> residual)
│   ├── model.py           # MiniLLaMA: embedding, layers, final norm, lm_head, checkpoint loading
│   └── generation.py      # Sampling strategies + cached/non-cached decode loops
├── tests/                 # pytest suite, one file per milestone
├── scripts/
│   └── run_generate.py    # CLI entry point (also installed as `minillama-generate`)
├── benchmarks/
│   ├── benchmark_kv_cache.py       # KV-cache vs no-cache timing
│   ├── benchmark_gqa_vs_mha.py     # n_kv_heads=2 (GQA) vs n_kv_heads=4 (MHA) timing
│   └── results/                    # JSON output, overwritten on each run
├── api/
│   ├── main.py             # FastAPI app: /health, /model/info, /generate
│   ├── inference_service.py# Loads model/tokenizer once at startup, exposes singletons
│   └── schemas.py          # Pydantic request/response models
├── checkpoints/            # Empty (.gitkeep only) — no trained weights included
├── SPEC.md                 # Milestone-by-milestone build spec this project follows
├── pyproject.toml
└── requirements.txt
```

---

## Setup and Installation

Everything below runs from inside `minillama-inference-lab/`.

```bash
cd minillama-inference-lab
pip install -e .
```

This puts `minillama`, `api`, `scripts`, and `benchmarks` on the Python path via an editable
install, so imports work regardless of your current working directory.

If you'd rather not install the package:

```bash
pip install -r requirements.txt
```

and run everything with `python -m ...`, since `-m` adds the current directory to the import path.

**Environment variables:**

| Variable | Purpose |
|---|---|
| `MINILLAMA_CHECKPOINT_PATH` | Optional. If set, the FastAPI backend loads this `state_dict` at startup instead of using random weights. |

No API keys, no secrets, no `.env` file — the project doesn't call out to anything external.

---

## Usage

### Run tests

```bash
pytest
```

### Generate text (CLI)

```bash
python -m scripts.run_generate --prompt "Once upon a time" --max_new_tokens 32 --use_kv_cache
```

or, after an editable install:

```bash
minillama-generate --prompt "Once upon a time" --max_new_tokens 32 --use_kv_cache
```

> No checkpoint is loaded by default. The model prints a warning and generates random,
> non-meaningful text. Pass `--checkpoint checkpoints/my_model.pt` to load a real `state_dict`
> that matches `ModelConfig`.

`generate()` validates its arguments and raises `ValueError` on bad input (surfaced as HTTP `422`
from the API):

| Argument | Rule |
|---|---|
| `prompt` | must not be empty |
| `max_new_tokens` | must be `>= 0` |
| `temperature` | must be `> 0` |
| `top_p` | must be in `(0, 1]` |
| `top_k` | must be `>= 0` |
| `repetition_penalty` | must be `>= 1` |

### Run the API

```bash
python -m uvicorn api.main:app --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/model/info
curl.exe -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"prompt\":\"Once upon a time\",\"max_new_tokens\":16,\"use_kv_cache\":true,\"greedy\":true}"
```

`/generate` responds with `text`, `tokens_generated`, `latency_ms`, `tokens_per_second`, and
`used_kv_cache`.

### Run benchmarks

```bash
python benchmarks/benchmark_kv_cache.py
python benchmarks/benchmark_gqa_vs_mha.py
```

Results land in `benchmarks/results/*.json`, overwritten each run.

---

## Implementation Details

- **RoPE** is precomputed once as `cos`/`sin` tables of shape `[max_seq_len, head_dim//2]` and
  applied by rotating pairs of dimensions (`(x0, x1) -> (-x1, x0)`), scaled by cos/sin — the
  standard "rotate half" formulation. It's sliced by `start_pos` on every forward call so a
  KV-cached decode step gets the position embedding for its true position, not position 0.
- **GQA** is implemented by giving K/V projections fewer output heads (`n_kv_heads < n_heads`) and
  `repeat_interleave`-ing them up to `n_heads` only inside the attention score computation. The KV
  cache stores the *compact*, un-repeated K/V — repeating happens fresh every forward call, so the
  cache itself stays small.
- **KV cache** is a list of `(k, v)` tuples, one per layer, each `[B, n_kv_heads, seq_len, head_dim]`.
  Passing an empty list on the first call triggers "collect but don't consume" (prefill); passing
  the returned caches back in with `start_pos` advancing does single-token decode.
- **Causal masking** only bothers building an explicit `[T, past+T]` mask when `T > 1` (prefill or
  no-cache generation); single-token cached decode steps skip masking entirely since a lone new
  token can attend to everything already in the cache.
- **Loss** is computed as shifted next-token cross-entropy directly inside `MiniLLaMA.forward` when
  `targets` are passed — there's no separate training loop, since training is out of scope for this
  project.
- **Weight tying** (`tie_embeddings=True` by default) shares `lm_head.weight` with
  `tok_embeddings.weight`, the standard LLaMA-style parameter-saving trick.
- **Random-weight warning**: both `MiniLLaMA.forward` and `generate()` warn (once, via a
  `_warned_random` flag) if no checkpoint has been loaded, so it's never ambiguous whether output
  came from a trained model.

---

## Results / Outputs

Benchmark JSON is written to `benchmarks/results/`. The numbers below are one local CPU run,
committed as an illustrative snapshot rather than a guaranteed baseline — re-run the scripts to get
numbers for your own machine, and keep in mind the model underneath is tiny and untrained, so this
measures inference mechanics, not generation quality.

**KV cache vs no cache** (32 tokens, prompt: "Once upon a time"):

| Mode | Latency (ms) | Tokens/sec |
|------|-------------|------------|
| No KV cache | 257.2 | 124.4 |
| KV cache | 197.1 | 162.4 |
| **Speedup** | — | **1.31×** |

**MHA vs GQA** (32 tokens, KV cache enabled):

| Mode | n_kv_heads | Params | KV Elements | Latency (ms) | Tokens/sec |
|------|-----------|--------|-------------|-------------|------------|
| MHA | 4 | 459,392 | 24,576 | 238.9 | 133.9 |
| GQA | 2 | 426,624 | 12,288 | 194.2 | 164.8 |

---

## Limitations / Future Improvements

- No trained checkpoint ships with the repo — there's nothing here yet that generates coherent text
- Byte-level tokenizer only; no subword/BPE tokenizer
- Benchmarks reflect one local CPU and a tiny model — not representative of production-scale
  latency
- No frontend, no persistence layer, no auth — this is inference-mechanics code, not a service
- Would benefit from: a small trained checkpoint (even a toy one), a lightweight results dashboard,
  and eventually quantization support — none of which are implemented yet

---

## Why This Project

This is an from-scratch reimplementation of the core mechanics behind modern decoder-only LLMs —
RoPE, GQA, KV-cache decoding — written to be read and benchmarked rather than imported as a black
box. It demonstrates the ability to translate a paper-level architecture description into working,
tested PyTorch, reason about correctness at the tensor-shape level, and measure real performance
tradeoffs (cache vs no cache, GQA vs MHA) instead of asserting them.
