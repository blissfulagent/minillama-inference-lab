import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from minillama.config import ModelConfig
from minillama.model import MiniLLaMA
from minillama.tokenizer import ByteTokenizer

_model: MiniLLaMA | None = None
_tokenizer: ByteTokenizer | None = None


def load_model_and_tokenizer() -> None:
    global _model, _tokenizer
    config = ModelConfig()
    _model = MiniLLaMA(config)
    _model.eval()
    _tokenizer = ByteTokenizer()


def get_model() -> MiniLLaMA:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model_and_tokenizer() at startup.")
    return _model


def get_tokenizer() -> ByteTokenizer:
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not loaded. Call load_model_and_tokenizer() at startup.")
    return _tokenizer
