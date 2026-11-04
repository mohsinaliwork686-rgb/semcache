"""A store in Redis, searched inside Redis with the vector sets added in Redis 8."""

import hashlib
import json
from typing import Any

import numpy as np

from .base import Entry, Evicted, Store

# The expiry stored for an answer that never expires (the last second of the year 9999), so
# one filter, ".expires > now", covers every entry.
_NEVER = 253_402_300_799

_POP_EXPIRED = """
local ids = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, 1000)
if #ids > 0 then redis.call('ZREM', KEYS[1], unpack(ids)) end
return ids
"""

# Atomic, so writers adding at the same moment never evict more than they need to.
_POP_LEAST_RECENTLY_USED = """
local extra = redis.call('ZCARD', KEYS[1]) - tonumber(ARGV[1])
if extra <= 0 then return {} end
return redis.call('ZPOPMIN', KEYS[1], extra)
"""

# KEYS: the entry, its vector set, its exact-question index, recency, expiries.
# ARGV: the entry's id, its question.
_REMOVE = """
if redis.call('HGET', KEYS[3], ARGV[2]) == ARGV[1] then redis.call('HDEL', KEYS[3], ARGV[2]) end
redis.call('VREM', KEYS[2], ARGV[1])
redis.call('DEL', KEYS[1])
redis.call('ZREM', KEYS[4], ARGV[1])
redis.call('ZREM', KEYS[5], ARGV[1])
return 1
"""

_TOUCH = """
if redis.call('ZSCORE', KEYS[1], ARGV[1]) then
  redis.call('ZADD', KEYS[1], redis.call('INCR', KEYS[2]), ARGV[1])
end
return 1
"""


class RedisStore(Store):
    """Keeps entries in Redis, so several processes or machines share one cache.

    Needs Redis 8 or later. Vectors are stored without quantisation and searched exactly, so
    scores match `MemoryStore`'s. Answers are stored as JSON, never pickled: unpickling data
    from a shared Redis would let anyone who can write to it run code. An answer that is not
    JSON-serialisable is refused.

    With many processes adding at once, the store can hold a few more than `max_entries` for
    a moment, between an add and the eviction that follows it.
    """

    def __init__(self, client: Any, name: str = "default") -> None:
        self.client = client
        self.prefix = f"semcache:{name}:"
        self._recency = self.prefix + "recency"
        self._ticks = self.prefix + "ticks"
        self._expiries = self.prefix + "expiries"
        self._pop_expired = client.register_script(_POP_EXPIRED)
        self._pop_least_recent = client.register_script(_POP_LEAST_RECENTLY_USED)
        self._remove_script = client.register_script(_REMOVE)
        self._touch_script = client.register_script(_TOUCH)
        self._version_checked = False

    @classmethod
    def from_url(cls, url: str, name: str = "default", **options: Any) -> "RedisStore":
        """Connect to the Redis at `url`. `name` keeps this cache apart from others there."""
        import redis

        return cls(redis.Redis.from_url(url, **options), name=name)

    def get_exact(self, key: str, question: str, now: float) -> Entry | None:
        self._check_version()
        entry_id = self.client.hget(self._exact_index(key), question)
        if entry_id is None:
            return None
        entry = self._load(entry_id.decode())
        return entry if entry and entry.is_live(now) else None

    def search(self, key: str, vector: np.ndarray, now: float) -> tuple[Entry, float] | None:
        self._check_version()
        reply = self.client.execute_command(
            "VSIM",
            self._vector_set(key),
            "FP32",
            _to_bytes(vector),
            "WITHSCORES",
            "COUNT",
            1,
            "FILTER",
            f".expires > {now!r}",
            "TRUTH",
        )
        # A dict of element to score, or a flat list; an empty list when the key is missing.
        if isinstance(reply, dict):
            pairs = list(reply.items())
        else:
            pairs = list(zip(reply[::2], reply[1::2], strict=True))
        if not pairs:
            return None
        entry_id, score = pairs[0]
        entry = self._load(entry_id.decode() if isinstance(entry_id, bytes) else entry_id)
        if entry is None:
            return None
        # Redis scores a match as (cosine + 1) / 2.
        return entry, min(1.0, max(-1.0, 2 * float(score) - 1))

    def add(self, entry: Entry, now: float, max_entries: int | None = None) -> Evicted:
        self._check_version()
        try:
            value = json.dumps(entry.value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"RedisStore keeps answers as JSON, and a {type(entry.value).__name__} is not "
                "JSON-serialisable; return text or plain data, or use MemoryStore"
            ) from error

        vector_set = self._vector_set(entry.key)
        if self.client.exists(vector_set):
            dimensions = int(self.client.execute_command("VDIM", vector_set))
            if dimensions != len(entry.vector):
                raise ValueError(
                    f"vector has {len(entry.vector)} dimensions, but this key holds vectors of "
                    f"{dimensions}; were two embedding models mixed?"
                )

        exact_index = self._exact_index(entry.key)
        replaced = self.client.hget(exact_index, entry.question)
        if replaced is not None:
            self._remove(replaced.decode())

        expires = _NEVER if entry.expires_at is None else entry.expires_at
        tick = self.client.incr(self._ticks)
        with self.client.pipeline(transaction=True) as pipe:
            pipe.execute_command(
                "VADD",
                vector_set,
                "FP32",
                _to_bytes(entry.vector),
                entry.id,
                "NOQUANT",
                "SETATTR",
                json.dumps({"expires": expires}),
            )
            pipe.hset(
                self.prefix + "entry:" + entry.id,
                mapping={
                    "key": entry.key,
                    "question": entry.question,
                    "value": value,
                    "stored_at": repr(entry.stored_at),
                    "expires_at": "" if entry.expires_at is None else repr(entry.expires_at),
                },
            )
            pipe.hset(exact_index, entry.question, entry.id)
            pipe.zadd(self._recency, {entry.id: tick})
            if entry.expires_at is not None:
                pipe.zadd(self._expiries, {entry.id: entry.expires_at})
            pipe.execute()

        expired = self._pop_expired(keys=[self._expiries], args=[repr(now)])
        for entry_id in expired:
            self._remove(entry_id.decode())
        evicted = 0
        if max_entries is not None:
            popped = self._pop_least_recent(keys=[self._recency], args=[max_entries])
            for entry_id in popped[::2]:
                self._remove(entry_id.decode())
                evicted += 1
        return Evicted(len(expired), evicted)

    def touch(self, entry: Entry, now: float) -> None:
        self._touch_script(keys=[self._recency, self._ticks], args=[entry.id])

    def __len__(self) -> int:
        return self.client.zcard(self._recency)

    def clear(self) -> None:
        batch = []
        for name in self.client.scan_iter(match=self.prefix + "*", count=500):
            batch.append(name)
            if len(batch) == 500:
                self.client.delete(*batch)
                batch.clear()
        if batch:
            self.client.delete(*batch)

    def _remove(self, entry_id: str) -> None:
        stored = self.client.hmget(self.prefix + "entry:" + entry_id, "key", "question")
        if stored[0] is None:
            # Already gone; it may still be listed for recency or expiry.
            self.client.zrem(self._recency, entry_id)
            self.client.zrem(self._expiries, entry_id)
            return
        key, question = stored[0].decode(), stored[1].decode()
        self._remove_script(
            keys=[
                self.prefix + "entry:" + entry_id,
                self._vector_set(key),
                self._exact_index(key),
                self._recency,
                self._expiries,
            ],
            args=[entry_id, question],
        )

    def _load(self, entry_id: str) -> Entry | None:
        data = self.client.hgetall(self.prefix + "entry:" + entry_id)
        if not data:
            return None
        expires_at = data[b"expires_at"].decode()
        return Entry(
            key=data[b"key"].decode(),
            question=data[b"question"].decode(),
            # The vector stays inside Redis; nothing reading an entry back needs it.
            vector=np.empty(0, dtype=np.float32),
            value=json.loads(data[b"value"]),
            stored_at=float(data[b"stored_at"]),
            expires_at=float(expires_at) if expires_at else None,
            id=entry_id,
        )

    def _vector_set(self, key: str) -> str:
        return self.prefix + "vectors:" + _short_hash(key)

    def _exact_index(self, key: str) -> str:
        return self.prefix + "questions:" + _short_hash(key)

    def _check_version(self) -> None:
        if self._version_checked:
            return
        version = self.client.info("server")["redis_version"]
        if int(version.split(".")[0]) < 8:
            raise RuntimeError(
                f"RedisStore needs Redis 8 or later, for vector sets; this server runs {version}"
            )
        self._version_checked = True


def _short_hash(key: str) -> str:
    # Keys can be long JSON; a hash keeps Redis key names short and free of odd characters.
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()
