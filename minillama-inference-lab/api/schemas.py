from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=32, ge=0, le=128)
    greedy: bool = False
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0)
    use_kv_cache: bool = True


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    weights_loaded: bool
    warning: str
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    hidden_dim: int
    max_seq_len: int


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float
    used_kv_cache: bool
