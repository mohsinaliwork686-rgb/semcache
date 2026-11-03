"""Cosine similarity between normalised vectors, and picking the best match."""

import numpy as np
from numpy.typing import ArrayLike


def normalize(vectors: ArrayLike) -> np.ndarray:
    """Scale each vector to length 1. A zero vector stays zero, so it scores 0 against anything."""
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.where(norms == 0, 1, norms)


def cosine_scores(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """The similarity of a normalised vector to each row of a matrix of normalised vectors."""
    # Rounding can push a score a hair past 1; a score above 1 would beat an exact match.
    return np.clip(matrix @ vector, -1.0, 1.0)


def best_match(scores: np.ndarray, live: np.ndarray | None = None) -> tuple[int, float] | None:
    """The row with the highest score among the live rows, or None if there are none.

    On a tie the earliest row wins, so the result does not depend on chance.
    """
    if live is not None:
        scores = np.where(live, scores, -np.inf)
    if scores.size == 0:
        return None
    index = int(np.argmax(scores))
    if scores[index] == -np.inf:
        return None
    return index, float(scores[index])
