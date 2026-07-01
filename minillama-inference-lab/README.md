# MiniLLaMA Inference Lab

A CPU-first LLaMA-style decoder-only Transformer inference lab in PyTorch.

---

## What This Project Is

An architecture and inference benchmarking lab that implements LLaMA-style components from scratch:
RMSNorm, RoPE, SwiGLU FFN, Multi-Head Attention (MHA), Grouped-Query Attention (GQA), KV-cache
inference, and a FastAPI backend — all runnable on CPU with tiny default model sizes.

## What This Project Is Not

- **Not real LLaMA 3.** No Meta weights, no Hugging Face model loading.
- Default weights are **random and untrained** — generated text is not meaningful. No trained checkpoint is included by default.
- **Not a chatbot.** This is an inference mechanics and architecture lab, not a conversational AI system.
- Not a training framework. No large-scale training, LoRA, or RLHF.

---

## Features

- Byte-level tokenizer (vocab size 256)
- RMSNorm, RoPE (applied to Q and K only)
- Causal self-attention with MHA and GQA
- SwiGLU feed-forward network
- Autoregressive generation: greedy, temperature, top-k, top-p, repetition penalty
- KV-cache inference
- FastAPI `/generate` endpoint
- Benchmark scripts: KV-cache vs no-cache, MHA vs GQA

---

## Project Structure

All commands and paths below are relative to `minillama-inference-lab/`.

```
minillama-inference-lab/
├── minillama/          # core model (config, tokenizer, attention, FFN, model)
├── tests/              # pytest suite
├── scripts/            # CLI generation
├── benchmarks/         # benchmark scripts + results/
├── api/                # FastAPI backend
├── checkpoints/        # empty (.gitkeep only) — no trained weights included
├── SPEC.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## Quickstart

All commands run from inside `minillama-inference-lab/`. Installing the project in editable
mode (recommended) puts `minillama`, `api`, `scripts`, and `benchmarks` on the Python path
without relying on `sys.path` hacks or a specific working directory:

```bash
cd minillama-inference-lab
pip install -e .
pytest
python -m scripts.run_generate --prompt "Once upon a time" --max_new_tokens 16 --use_kv_cache
python -m uvicorn api.main:app --reload
```

The `minillama-generate` console command is also installed and equivalent to
`python -m scripts.run_generate`.

If you prefer not to install the package, `pip install -r requirements.txt` plus running
commands with `python -m ...` from inside `minillama-inference-lab/` also works, since `-m`
puts the current directory on the import path.

---

## Run Tests

```bash
pytest
```

---

## Run Generation CLI

```bash
python -m scripts.run_generate --prompt "Once upon a time" --max_new_tokens 32 --use_kv_cache
```

> **Note:** No trained checkpoint is loaded by default. The model prints a warning and generates
> random text. Output is not meaningful without a real trained checkpoint.

### Input Validation

`generate()` (and the CLI/API on top of it) validates its arguments and raises a clear
`ValueError` (`422` from the API) on bad input:

| Argument | Rule |
|---|---|
| `prompt` | must not be empty |
| `max_new_tokens` | must be `>= 0` |
| `temperature` | must be `> 0` |
| `top_p` | must be in `(0, 1]` |
| `top_k` | must be `>= 0` |
| `repetition_penalty` | must be `>= 1` |

### Checkpoint Loading

No trained checkpoint ships with this repo. If you have a PyTorch `state_dict` that matches
`ModelConfig`, load it with:

```bash
python -m scripts.run_generate --prompt "Once upon a time" --checkpoint checkpoints/my_model.pt
```

or programmatically:

```python
model = MiniLLaMA(ModelConfig())
model.load_checkpoint("checkpoints/my_model.pt")  # marks weights as loaded, silences the warning
```

The FastAPI backend loads a checkpoint at startup if the `MINILLAMA_CHECKPOINT_PATH`
environment variable is set:

```bash
MINILLAMA_CHECKPOINT_PATH=checkpoints/my_model.pt python -m uvicorn api.main:app --reload
```

---

## Run Benchmarks

Run from inside `minillama-inference-lab/`:

```bash
python benchmarks/benchmark_kv_cache.py
python benchmarks/benchmark_gqa_vs_mha.py
```

Results are saved to `benchmarks/results/`.

---

## Benchmark Results

> These numbers were captured on one local CPU setup and are committed as an illustrative
> snapshot, not a guaranteed baseline — timings vary across machines, CPU load, and PyTorch
> versions. Re-run the scripts above to regenerate them for your own machine; the JSON files
> in `benchmarks/results/` are overwritten each run.
> The benchmark uses a tiny random-weight model, so results measure inference mechanics, not model quality.

**KV Cache vs No Cache** (32 tokens, prompt: "Once upon a time"):

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

## FastAPI Usage

Start the server (from inside `minillama-inference-lab/`):

```bash
python -m uvicorn api.main:app --reload
```

Interactive API docs: `http://127.0.0.1:8000/docs`

**Windows PowerShell examples:**

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/model/info
curl.exe -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"prompt\":\"Once upon a time\",\"max_new_tokens\":16,\"use_kv_cache\":true,\"greedy\":true}"
```

Response fields: `text`, `tokens_generated`, `latency_ms`, `tokens_per_second`, `used_kv_cache`.

---

## Limitations

- Default weights are random — output is not meaningful without a trained checkpoint
- CPU-first; CUDA is optional if available
- Byte-level tokenizer (vocab=256) — not a subword tokenizer
- No real LLaMA 3 weights included
- No authentication, no database, no frontend

---

## Repository Status

**Completed:**
- Core model (RMSNorm, RoPE, SwiGLU, MHA/GQA, KV-cache, chunked cached decode)
- Autoregressive generation with all sampling modes and input validation
- Device-aware generation (CPU by default, CUDA if available)
- Shifted next-token loss (`logits[:, :-1]` predicts `targets[:, 1:]`)
- Checkpoint loading (`MiniLLaMA.load_checkpoint`, CLI `--checkpoint`, API `MINILLAMA_CHECKPOINT_PATH`)
- KV-cache and GQA vs MHA benchmarks
- FastAPI inference backend (`/health`, `/model/info`, `/generate`)
- Editable-install packaging (`pyproject.toml`, no `sys.path` hacks)

**Future Work:**
- Tiny Kaggle-trained checkpoint
- Frontend dashboard
- Quantization
- Docker packaging
