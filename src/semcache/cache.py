"""The cache: an exact-text check first, then a search by meaning."""

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .similarity import normalize
from .store import Entry, MemoryStore, Store

# Anything that turns a list of strings into one vector per string.
Embedder = Callable[[Sequence[str]], ArrayLike]


@dataclass(frozen=True)
class Hit:
    """A stored answer, and the stored question it was matched to."""

    value: Any
    question: str
    similarity: float
    exact: bool


@dataclass(frozen=True)
class Stats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    expirations: int = 0
    errors: int = 0
    # Calls the cache let through untouched, such as a question that is not a string.
    skipped: int = 0

    @property
    def hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    @property
    def hit_rate(self) -> float:
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0


class SemanticCache:
    """Returns a stored answer when a new question means the same as one already answered.

    `threshold` is the lowest cosine similarity that counts as the same question. It has no
    default, because the right value depends on the embedding model and the questions; at 1.0
    only the exact same text matches. `ttl` is in seconds from when an answer is stored, and a
    hit does not extend it. `max_entries` limits the whole cache; the least recently used
    answers are dropped to stay within it.
    """

    def __init__(
        self,
        *,
        threshold: float,
        embedder: Embedder,
        ttl: float | None = None,
        max_entries: int | None = None,
        store: Store | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be above 0 and at most 1, not {threshold}")
        if ttl is not None and ttl <= 0:
            raise ValueError(f"ttl must be a positive number of seconds, not {ttl}")
        if max_entries is not None and max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, not {max_entries}")
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self._embedder = embedder
        self._store = store if store is not None else MemoryStore()
        self._clock = clock
        self._lock = threading.Lock()
        self._counts = dict.fromkeys((f.name for f in fields(Stats)), 0)

    def lookup(self, question: str, key: str = "") -> Hit | None:
        """The stored answer for this question, or None. `key` must match exactly."""
        hit, _ = self._find(question, key)
        return hit

    def store(self, question: str, value: Any, key: str = "") -> None:
        """Store an answer to this question under `key`."""
        self._put(question, value, key, vector=None)

    def cached(self, on: str | Callable | None = None) -> Callable:
        """Decorate a function so its results are cached here.

        `on` names the argument matched by meaning, and defaults to the first. Every other
        argument must match exactly: the same question to another model, or with another
        system prompt, never gets that call's answer.
        """
        from .decorator import wrap

        if callable(on):  # Used bare, as @cache.cached
            return wrap(self, on, None)
        return lambda func: wrap(self, func, on)

    def stats(self) -> Stats:
        with self._lock:
            return Stats(**self._counts)

    def clear(self) -> None:
        """Remove every stored answer. The statistics are kept."""
        self._store.clear()

    def _find(self, question: str, key: str) -> tuple[Hit | None, np.ndarray | None]:
        # Also returns the question's vector, so storing the answer after a miss does not
        # embed the same text twice.
        now = self._clock()
        entry = self._store.get_exact(key, question, now)
        if entry is not None:
            self._store.touch(entry, now)
            self._count(exact_hits=1)
            return Hit(entry.value, entry.question, 1.0, exact=True), None

        # Defined rather than left to floating point: identical text can embed to a score of
        # 0.99999994, so "at least 1.0" would sometimes miss its own question.
        if self.threshold >= 1:
            self._count(misses=1)
            return None, None

        vector = self._embed(question)
        found = self._store.search(key, vector, now)
        if found is not None and found[1] >= self.threshold:
            entry, similarity = found
            self._store.touch(entry, now)
            self._count(semantic_hits=1)
            return Hit(entry.value, entry.question, similarity, exact=False), vector
        self._count(misses=1)
        return None, vector

    def _put(self, question: str, value: Any, key: str, vector: np.ndarray | None) -> None:
        if vector is None:
            vector = self._embed(question)
        now = self._clock()
        entry = Entry(
            key=key,
            question=question,
            vector=vector,
            value=value,
            stored_at=now,
            expires_at=None if self.ttl is None else now + self.ttl,
        )
        evicted = self._store.add(entry, now, self.max_entries)
        self._count(stores=1, expirations=evicted.expired, evictions=evicted.least_recently_used)

    def _embed(self, question: str) -> np.ndarray:
        vectors = np.asarray(self._embedder([question]))
        if vectors.ndim != 2 or len(vectors) != 1:
            raise ValueError(f"the embedder returned shape {vectors.shape} for one question")
        return normalize(vectors[0])

    def _count(self, **amounts: int) -> None:
        with self._lock:
            for name, amount in amounts.items():
                self._counts[name] += amount
