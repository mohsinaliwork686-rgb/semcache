"""The cache never breaks the call: when it fails, the function runs as if it were not there."""

import logging

import pytest

from fakes import FakeEmbedder
from semcache import MemoryStore, SemanticCache


class BrokenEmbedder:
    def __call__(self, texts):
        raise RuntimeError("the model would not load")


class BrokenStore(MemoryStore):
    """Works until `broken` is set, then every call fails the way a lost connection would."""

    broken = False

    def _check(self):
        if self.broken:
            raise ConnectionError("Redis is down")

    def get_exact(self, *args):
        self._check()
        return super().get_exact(*args)

    def search(self, *args):
        self._check()
        return super().search(*args)

    def add(self, *args):
        self._check()
        return super().add(*args)


@pytest.fixture
def store():
    return BrokenStore()


def test_a_broken_embedder_lets_the_call_through(caplog):
    cache = SemanticCache(threshold=0.9, embedder=BrokenEmbedder())

    @cache.cached
    def ask(question: str) -> str:
        return "the model's answer"

    with caplog.at_level(logging.WARNING, logger="semcache"):
        assert ask("q") == "the model's answer"
        assert ask("q") == "the model's answer"
    stats = cache.stats()
    assert (stats.errors, stats.misses, stats.stores) == (2, 0, 0)
    assert "the model would not load" in caplog.text


def test_a_store_that_goes_down_lets_the_call_through(store):
    cache = SemanticCache(threshold=0.9, embedder=FakeEmbedder(), store=store)
    calls = []

    @cache.cached
    def ask(question: str) -> str:
        calls.append(question)
        return "answer"

    ask("q")
    ask("q")
    assert calls == ["q"]
    store.broken = True
    assert ask("q") == "answer"
    assert calls == ["q", "q"]
    assert cache.stats().errors == 1, "a failed lookup is not followed by a failed store"


def test_a_store_that_fails_only_on_writes_still_returns_the_answer(store):
    cache = SemanticCache(threshold=0.9, embedder=FakeEmbedder(), store=store)

    def read_only(*args):
        raise ConnectionError("read-only replica")

    store.add = read_only

    @cache.cached
    def ask(question: str) -> str:
        return "answer"

    assert ask("q") == "answer"
    stats = cache.stats()
    assert (stats.misses, stats.stores, stats.errors) == (1, 0, 1)


def test_lookup_and_store_never_raise(store):
    cache = SemanticCache(threshold=0.9, embedder=FakeEmbedder(), store=store)
    store.broken = True
    cache.store("q", "answer")
    assert cache.lookup("q") is None
    assert cache.stats().errors == 2


def test_the_function_s_own_error_is_not_swallowed(store):
    cache = SemanticCache(threshold=0.9, embedder=FakeEmbedder(), store=store)

    @cache.cached
    def ask(question: str) -> str:
        raise TimeoutError("the model timed out")

    with pytest.raises(TimeoutError):
        ask("q")
    assert cache.stats().errors == 0
