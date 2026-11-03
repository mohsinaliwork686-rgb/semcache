import numpy as np
import pytest

from fakes import FakeClock, FakeEmbedder, at_angle
from semcache import SemanticCache, Stats


def make_cache(threshold=0.9, vectors=None, **options):
    embedder = FakeEmbedder(vectors)
    return SemanticCache(threshold=threshold, embedder=embedder, **options), embedder


def test_the_threshold_is_required():
    with pytest.raises(TypeError, match="threshold"):
        SemanticCache(embedder=FakeEmbedder())


@pytest.mark.parametrize(
    "options",
    [{"threshold": 0}, {"threshold": 1.1}, {"ttl": 0}, {"max_entries": 0}],
    ids=["threshold 0", "threshold above 1", "ttl 0", "max_entries 0"],
)
def test_impossible_settings_are_refused(options):
    with pytest.raises(ValueError):
        SemanticCache(**{"threshold": 0.9, "embedder": FakeEmbedder(), **options})


def test_an_exact_repeat_is_a_hit_without_embedding_again():
    cache, embedder = make_cache()
    cache.store("how do I reset my password", "Use the link.")
    calls = len(embedder.calls)
    hit = cache.lookup("how do I reset my password")
    assert (hit.value, hit.exact, hit.similarity) == ("Use the link.", True, 1.0)
    assert len(embedder.calls) == calls


def test_a_paraphrase_above_the_threshold_hits_and_below_it_misses():
    cache, _ = make_cache(
        threshold=0.9,
        vectors={"stored": [1, 0], "close": at_angle(0.95), "not close": at_angle(0.85)},
    )
    cache.store("stored", "answer")
    hit = cache.lookup("close")
    assert (hit.value, hit.question, hit.exact) == ("answer", "stored", False)
    assert hit.similarity == pytest.approx(0.95, abs=1e-6)
    assert cache.lookup("not close") is None


def test_threshold_1_is_an_exact_text_cache():
    # Two texts with the very same vector: only the stored text itself may hit.
    cache, embedder = make_cache(threshold=1.0, vectors={"stored": [1, 0], "twin": [1, 0]})
    cache.store("stored", "answer")
    calls = len(embedder.calls)
    assert cache.lookup("twin") is None
    assert cache.lookup("stored").value == "answer"
    assert len(embedder.calls) == calls, "an exact-only cache never needs to embed a lookup"


def test_a_different_key_never_shares_an_answer():
    cache, _ = make_cache()
    cache.store("summarise this", "small model's answer", key='{"model":"small"}')
    assert cache.lookup("summarise this", key='{"model":"large"}') is None
    assert cache.lookup("summarise this", key='{"model":"small"}') is not None


def test_an_answer_expires_ttl_seconds_after_it_was_stored():
    clock = FakeClock()
    cache, _ = make_cache(ttl=60, clock=clock)
    cache.store("q", "answer")
    clock.advance(59)
    assert cache.lookup("q") is not None
    clock.advance(1)
    assert cache.lookup("q") is None


def test_a_hit_does_not_extend_the_ttl():
    clock = FakeClock()
    cache, _ = make_cache(ttl=60, clock=clock)
    cache.store("q", "answer")
    for _ in range(5):
        clock.advance(10)
        assert cache.lookup("q") is not None
    clock.advance(10)
    assert cache.lookup("q") is None


def test_the_least_recently_used_answer_goes_when_full():
    clock = FakeClock()
    cache, _ = make_cache(max_entries=2, clock=clock)
    cache.store("a", 1)
    cache.store("b", 2)
    clock.advance(1)
    cache.lookup("a")
    cache.store("c", 3)
    assert cache.lookup("b") is None
    assert cache.lookup("a").value == 1
    assert cache.stats().evictions == 1


def test_the_statistics_add_up():
    clock = FakeClock()
    cache, _ = make_cache(vectors={"q": [1, 0], "q?": at_angle(0.99)}, ttl=10, clock=clock)
    assert cache.stats() == Stats()
    cache.lookup("q")  # miss
    cache.store("q", "answer")
    cache.lookup("q")  # exact hit
    cache.lookup("q?")  # semantic hit
    clock.advance(10)
    cache.store("later", "answer")  # removes the expired "q"
    stats = cache.stats()
    assert (stats.exact_hits, stats.semantic_hits, stats.misses) == (1, 1, 1)
    assert (stats.stores, stats.expirations, stats.evictions) == (2, 1, 0)
    assert stats.hits == 2
    assert stats.hit_rate == pytest.approx(2 / 3)


def test_an_embedder_returning_the_wrong_shape_is_caught():
    cache = SemanticCache(threshold=0.9, embedder=lambda texts: np.zeros(4))
    with pytest.raises(ValueError, match="shape"):
        cache.store("q", "answer")


def test_clear_forgets_the_answers_but_keeps_the_statistics():
    cache, _ = make_cache()
    cache.store("q", "answer")
    cache.clear()
    assert cache.lookup("q") is None
    assert cache.stats().stores == 1
