from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .inference_service import get_model, get_tokenizer, load_model_and_tokenizer
from .schemas import GenerateRequest, GenerateResponse, HealthResponse, ModelInfoResponse
from minillama.generation import generate


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_tokenizer()
    yield


app = FastAPI(title="MiniLLaMA Inference API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    model = get_model()
    cfg = model.config
    warning = (
        "Random untrained weights. Generated text is not meaningful."
        if not model._weights_loaded
        else ""
    )
    return ModelInfoResponse(
        weights_loaded=model._weights_loaded,
        warning=warning,
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        hidden_dim=cfg.hidden_dim,
        max_seq_len=cfg.max_seq_len,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    model = get_model()
    tokenizer = get_tokenizer()

    try:
        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            greedy=request.greedy,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            use_kv_cache=request.use_kv_cache,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return GenerateResponse(
        text=result["text"],
        tokens_generated=result["tokens_generated"],
        latency_ms=result["latency_ms"],
        tokens_per_second=result["tokens_per_second"],
        used_kv_cache=result["used_kv_cache"],
    )
