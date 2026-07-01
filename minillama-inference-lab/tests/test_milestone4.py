import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["weights_loaded"] is False
    assert "Random untrained weights" in body["warning"]
    assert body["vocab_size"] == 256
    assert body["dim"] == 128
    assert body["n_layers"] == 2


def test_generate_returns_required_fields(client):
    r = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    for field in ("text", "tokens_generated", "latency_ms", "tokens_per_second", "used_kv_cache"):
        assert field in body


def test_generate_kv_cache_flag(client):
    r = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 4, "use_kv_cache": True})
    assert r.status_code == 200
    assert r.json()["used_kv_cache"] is True


def test_generate_invalid_temperature(client):
    r = client.post("/generate", json={"prompt": "hi", "temperature": 0.0})
    assert r.status_code == 422


def test_generate_invalid_top_p(client):
    r = client.post("/generate", json={"prompt": "hi", "top_p": 1.5})
    assert r.status_code == 422


def test_generate_invalid_repetition_penalty(client):
    r = client.post("/generate", json={"prompt": "hi", "repetition_penalty": 0.5})
    assert r.status_code == 422


def test_generate_empty_prompt(client):
    r = client.post("/generate", json={"prompt": ""})
    assert r.status_code == 422


def test_generate_invalid_top_k(client):
    r = client.post("/generate", json={"prompt": "hi", "top_k": -1})
    assert r.status_code == 422


def test_generate_invalid_max_new_tokens(client):
    r = client.post("/generate", json={"prompt": "hi", "max_new_tokens": -1})
    assert r.status_code == 422
