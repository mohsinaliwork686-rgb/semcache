"""What every store does, and the entry it keeps."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np


@dataclass(frozen=True, eq=False)
class Entry:
    """One stored answer. `key` holds every argument except the question, as one string; only
    entries with the same key are compared, so a different model never shares an answer."""

    key: str
    question: str
    vector: np.ndarray = field(repr=False)
    value: Any
    stored_at: float
    expires_at: float | None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def is_live(self, now: float) -> bool:
        return self.expires_at is None or now < self.expires_at


class Evicted(NamedTuple):
    """How many entries an add pushed out, and why."""

    expired: int = 0
    least_recently_used: int = 0


class Store(ABC):
    """Holds entries and searches them. Expired entries are never returned.

    `now` is passed in rather than read from a clock, so the cache decides what time it is.
    """

    @abstractmethod
    def get_exact(self, key: str, question: str, now: float) -> Entry | None:
        """The live entry for exactly this question under this key."""

    @abstractmethod
    def search(self, key: str, vector: np.ndarray, now: float) -> tuple[Entry, float] | None:
        """The live entry under this key most similar to a normalised vector, with its score."""

    @abstractmethod
    def add(self, entry: Entry, now: float, max_entries: int | None = None) -> Evicted:
        """Store an entry, replacing any with the same key and question.

        Expired entries are removed first. If the store still holds more than `max_entries`,
        the least recently used live entries are evicted until it does not.
        """

    @abstractmethod
    def touch(self, entry: Entry, now: float) -> None:
        """Record a hit, which counts as use for least-recently-used eviction."""

    @abstractmethod
    def __len__(self) -> int:
        """How many entries are held, including expired ones not yet removed."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every entry."""
