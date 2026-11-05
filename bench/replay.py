"""Replaying the labelled questions through the cache, and scoring what it returns.

The cache under test is the library itself: a `SemanticCache` wrapped around a stand-in for the
model, so every lookup goes through the same code a user's call would.
"""

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from semcache import MemoryStore, SemanticCache

QUESTIONS = Path(__file__).parent / "questions.json"
NO_NEAR_MISS = "none"


@dataclass(frozen=True)
class Query:
    text: str
    intent: str
    # The kind of near-miss its intent was written to have, or "none".
    kind: str


@dataclass
class Counts:
    """What one replay did. Every query is exactly one of: a correct hit, a false hit, a miss."""

    queries: int = 0
    correct_hits: int = 0
    false_hits: int = 0
    misses: int = 0
    # Misses when an answer with the same intent was already stored.
    missed_chances: int = 0
    embed_calls: int = 0
    # Per kind of near-miss: how many queries of that kind, and how many got a wrong answer.
    queries_by_kind: dict[str, int] = field(default_factory=dict)
    false_hits_by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def hits(self) -> int:
        return self.correct_hits + self.false_hits


def load_queries(path: Path = QUESTIONS) -> list[Query]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Query(text, intent["id"], intent.get("near_miss", {}).get("kind", NO_NEAR_MISS))
        for intent in data["intents"]
        for text in intent["questions"]
    ]


def replay(queries: Sequence[Query], vectors: dict[str, np.ndarray], threshold: float) -> Counts:
    """Ask every query once, in this order, through a fresh cache.

    A miss "calls the model", whose answer is the query's own intent, and the cache stores it.
    A hit returns the intent of whichever stored question it matched, so a hit is false when
    that intent differs from the query's.
    """
    counts = Counts()

    def embed(texts: Sequence[str]) -> np.ndarray:
        counts.embed_calls += 1
        return np.stack([vectors[text] for text in texts])

    cache = SemanticCache(threshold=threshold, embedder=embed, store=MemoryStore())
    stored_intents: set[str] = set()
    called: list[Query] = []

    @cache.cached
    def model(question: str) -> str:
        called.append(current)
        return current.intent

    for current in queries:
        counts.queries += 1
        counts.queries_by_kind[current.kind] = counts.queries_by_kind.get(current.kind, 0) + 1
        before = len(called)
        answer = model(current.text)
        if len(called) > before:
            counts.misses += 1
            if current.intent in stored_intents:
                counts.missed_chances += 1
            stored_intents.add(current.intent)
        elif answer == current.intent:
            counts.correct_hits += 1
        else:
            counts.false_hits += 1
            counts.false_hits_by_kind[current.kind] = (
                counts.false_hits_by_kind.get(current.kind, 0) + 1
            )
    return counts


def shuffled(queries: Sequence[Query], seed: int) -> list[Query]:
    order = list(queries)
    random.Random(seed).shuffle(order)
    return order
