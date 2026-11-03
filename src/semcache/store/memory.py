"""A store that keeps everything in this process."""

import heapq
import threading
from collections import OrderedDict

import numpy as np

from ..similarity import best_match, cosine_scores
from .base import Entry, Evicted, Store


class _Partition:
    """The entries under one key, with their vectors packed into one matrix for searching."""

    def __init__(self, dimensions: int) -> None:
        self.vectors = np.empty((8, dimensions), dtype=np.float32)
        self.expires = np.empty(8, dtype=np.float64)
        self.entries: list[Entry] = []
        self.rows: dict[str, int] = {}
        self.by_question: dict[str, Entry] = {}

    def add(self, entry: Entry) -> None:
        if entry.vector.shape != self.vectors.shape[1:]:
            raise ValueError(
                f"vector has shape {entry.vector.shape}, but this key holds vectors of "
                f"shape {self.vectors.shape[1:]}; were two embedding models mixed?"
            )
        row = len(self.entries)
        if row == len(self.vectors):
            self.vectors = np.concatenate([self.vectors, np.empty_like(self.vectors)])
            self.expires = np.concatenate([self.expires, np.empty_like(self.expires)])
        self.vectors[row] = entry.vector
        self.expires[row] = np.inf if entry.expires_at is None else entry.expires_at
        self.entries.append(entry)
        self.rows[entry.id] = row
        self.by_question[entry.question] = entry

    def remove(self, entry: Entry) -> None:
        # Move the last row into the gap, so removing is O(1) and the matrix stays packed.
        row = self.rows.pop(entry.id)
        last = len(self.entries) - 1
        if row != last:
            moved = self.entries[last]
            self.entries[row] = moved
            self.vectors[row] = self.vectors[last]
            self.expires[row] = self.expires[last]
            self.rows[moved.id] = row
        self.entries.pop()
        del self.by_question[entry.question]


class MemoryStore(Store):
    """Search is one matrix multiplication per lookup. Safe to share between threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._partitions: dict[str, _Partition] = {}
        # Every live entry, least recently used first.
        self._recency: OrderedDict[str, Entry] = OrderedDict()
        # (expires_at, id), soonest first; ids of entries already removed are skipped.
        self._expiries: list[tuple[float, str]] = []

    def get_exact(self, key: str, question: str, now: float) -> Entry | None:
        with self._lock:
            partition = self._partitions.get(key)
            entry = partition and partition.by_question.get(question)
            return entry if entry and entry.is_live(now) else None

    def search(self, key: str, vector: np.ndarray, now: float) -> tuple[Entry, float] | None:
        with self._lock:
            partition = self._partitions.get(key)
            if partition is None:
                return None
            count = len(partition.entries)
            scores = cosine_scores(partition.vectors[:count], vector)
            found = best_match(scores, live=partition.expires[:count] > now)
            return found and (partition.entries[found[0]], found[1])

    def add(self, entry: Entry, now: float, max_entries: int | None = None) -> Evicted:
        with self._lock:
            partition = self._partitions.get(entry.key)
            if partition is None:
                partition = self._partitions[entry.key] = _Partition(len(entry.vector))
            replaced = partition.by_question.get(entry.question)
            if replaced is not None:
                partition.remove(replaced)
                del self._recency[replaced.id]
            partition.add(entry)
            self._recency[entry.id] = entry
            if entry.expires_at is not None:
                heapq.heappush(self._expiries, (entry.expires_at, entry.id))

            expired = self._remove_expired(now)
            evicted = 0
            while max_entries is not None and len(self._recency) > max_entries:
                _, oldest = self._recency.popitem(last=False)
                self._drop(oldest)
                evicted += 1
            return Evicted(expired, evicted)

    def touch(self, entry: Entry, now: float) -> None:
        with self._lock:
            if entry.id in self._recency:
                self._recency.move_to_end(entry.id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._recency)

    def clear(self) -> None:
        with self._lock:
            self._partitions.clear()
            self._recency.clear()
            self._expiries.clear()

    def _drop(self, entry: Entry) -> None:
        partition = self._partitions[entry.key]
        partition.remove(entry)
        if not partition.entries:
            del self._partitions[entry.key]

    def _remove_expired(self, now: float) -> int:
        removed = 0
        while self._expiries and self._expiries[0][0] <= now:
            _, entry_id = heapq.heappop(self._expiries)
            entry = self._recency.pop(entry_id, None)
            if entry is not None:
                self._drop(entry)
                removed += 1
        return removed
