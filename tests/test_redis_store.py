"""What only the Redis store has to get right. The shared suite in test_store.py runs on it too."""

import os
import threading
import uuid

import numpy as np
import pytest

from fakes import FakeEmbedder, at_angle
from semcache import MemoryStore, RedisStore, SemanticCache
from semcache.similarity import normalize
from semcache.store import Entry

NOW = 1_000.0


def entry(question, vector, *, key="k", value="answer", ttl=None):
    return Entry(
        key=key,
        question=question,
        vector=normalize(vector),
        value=value,
        stored_at=NOW,
        expires_at=None if ttl is None else NOW + ttl,
    )


def test_scores_match_the_memory_store(redis_store):
    rng = np.random.default_rng(7)
    memory = MemoryStore()
    stored = rng.standard_normal((40, 384))
    for n, vector in enumerate(stored):
        memory.add(entry(f"q{n}", vector), NOW)
        redis_store.add(entry(f"q{n}", vector), NOW)
    for query in rng.standard_normal((20, 384)):
        query = normalize(query)
        (in_memory, memory_score), (in_redis, redis_score) = (
            memory.search("k", query, NOW),
            redis_store.search("k", query, NOW),
        )
        assert in_redis.question == in_memory.question
        assert redis_score == pytest.approx(memory_score, abs=1e-6)


@pytest.mark.parametrize(
    "value", ["text", 42, 1.5, None, ["a", 1], {"answer": "yes", "sources": [1, 2]}]
)
def test_answers_come_back_as_they_went_in(redis_store, value):
    redis_store.add(entry("q", [1, 0], value=value), NOW)
    assert redis_store.get_exact("k", "q", NOW).value == value


def test_an_answer_that_is_not_json_is_refused(redis_store):
    with pytest.raises(TypeError, match="JSON"):
        redis_store.add(entry("q", [1, 0], value=object()), NOW)
    assert len(redis_store) == 0


def test_through_the_cache_a_non_json_answer_is_still_returned(redis_store):
    cache = SemanticCache(threshold=0.9, embedder=FakeEmbedder(), store=redis_store)

    @cache.cached
    def ask(question: str) -> object:
        return {1, 2}

    assert ask("q") == {1, 2}
    assert cache.stats().errors == 1


def test_searching_a_key_that_does_not_exist_finds_nothing_on_both_protocols(redis_store):
    # redis-py 8.1.0's own vsim() raises here under RESP3, its default (fixed after 8.1.0).
    for protocol in (2, 3):
        store = RedisStore.from_url(
            os.environ["REDIS_URL"], name=f"test-{uuid.uuid4().hex}", protocol=protocol
        )
        assert store.search("never stored", normalize([1, 0]), NOW) is None


def test_redis_older_than_8_is_refused_with_a_plain_reason(redis_store, monkeypatch):
    monkeypatch.setattr(redis_store.client, "info", lambda section: {"redis_version": "7.4.2"})
    with pytest.raises(RuntimeError, match="Redis 8 or later.*7.4.2"):
        redis_store.get_exact("k", "q", NOW)


def test_two_caches_on_one_redis_stay_apart(redis_store):
    other = RedisStore(redis_store.client, name=f"test-{uuid.uuid4().hex}")
    redis_store.add(entry("q", [1, 0]), NOW)
    assert other.get_exact("k", "q", NOW) is None
    assert other.search("k", normalize([1, 0]), NOW) is None
    other.clear()
    assert redis_store.get_exact("k", "q", NOW) is not None


def test_many_writers_at_once_leave_exactly_max_entries(redis_store):
    writers, per_writer, limit = 8, 40, 100
    evicted = []

    def write(writer):
        for n in range(per_writer):
            result = redis_store.add(
                entry(f"{writer}-{n}", [writer + 1, n + 1], key=f"k{writer}"),
                NOW,
                max_entries=limit,
            )
            evicted.append(result.least_recently_used)

    threads = [threading.Thread(target=write, args=(w,)) for w in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(redis_store) == limit
    assert sum(evicted) == writers * per_writer - limit


def test_the_cache_works_end_to_end_on_redis(redis_store):
    vectors = {"reset my password": [1, 0], "how do I reset my password": at_angle(0.95)}
    cache = SemanticCache(threshold=0.9, embedder=FakeEmbedder(vectors), store=redis_store)
    calls = []

    @cache.cached
    def ask(question: str) -> str:
        calls.append(question)
        return "Use the reset link."

    assert ask("reset my password") == "Use the reset link."
    assert ask("reset my password") == "Use the reset link."
    assert ask("how do I reset my password") == "Use the reset link."
    assert calls == ["reset my password"]
    stats = cache.stats()
    assert (stats.exact_hits, stats.semantic_hits, stats.misses) == (1, 1, 1)


def test_redis_being_down_never_breaks_the_call():
    # Nothing listens on port 1, so every Redis call fails at once.
    store = RedisStore.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)
    cache = SemanticCache(threshold=0.9, embedder=FakeEmbedder(), store=store)

    @cache.cached
    def ask(question: str) -> str:
        return "the model's answer"

    assert ask("q") == "the model's answer"
    assert cache.stats().errors == 1
