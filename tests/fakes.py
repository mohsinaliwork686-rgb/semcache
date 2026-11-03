"""Stand-ins for the real clock and embedding model, so tests are instant and exact."""

import hashlib

import numpy as np


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeEmbedder:
    """Gives each known text its fixed vector, and any other text a stable random one.

    Random vectors in 64 dimensions are close to unrelated, so unknown texts never match.
    """

    def __init__(self, vectors: dict[str, list[float]] | None = None, dimensions: int = 64):
        # Short vectors are padded with zeros, so every vector has the same length.
        self.vectors = {
            text: np.pad(np.asarray(v, dtype=np.float32), (0, dimensions - len(v)))
            for text, v in (vectors or {}).items()
        }
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.stack([self._vector(text) for text in texts])

    def _vector(self, text: str) -> np.ndarray:
        if text in self.vectors:
            return self.vectors[text]
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(self.dimensions).astype(np.float32)


def at_angle(similarity: float) -> list[float]:
    """A 2-D vector whose cosine similarity to [1, 0] is exactly `similarity`."""
    return [similarity, float(np.sqrt(1 - similarity**2))]
