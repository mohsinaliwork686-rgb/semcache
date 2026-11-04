"""Turning questions into vectors with a model that runs on this machine."""

import importlib.util
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    """A sentence-embedding model run through fastembed, on ONNX Runtime rather than PyTorch.

    The model is downloaded and loaded the first time it is used, not when this is created; the
    default is about 90 MB. `cache_dir` is where the download is kept.
    """

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | None = None) -> None:
        # Checked now, so a missing install is found at start-up, not on the first question.
        if importlib.util.find_spec("fastembed") is None:
            raise ImportError(
                'the local embedding model needs fastembed: pip install "semcache[local]", '
                "or pass your own embedder"
            )
        self.model = model
        self._cache_dir = cache_dir
        self._loaded: Any = None
        self._lock = threading.Lock()

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack(list(self._load().embed(list(texts))))

    def _load(self) -> Any:
        with self._lock:
            if self._loaded is None:
                from fastembed import TextEmbedding

                self._loaded = TextEmbedding(self.model, cache_dir=self._cache_dir)
            return self._loaded
