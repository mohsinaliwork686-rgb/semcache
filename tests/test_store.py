"""The suite every store must pass. Each test runs once per store."""

import pytest

from semcache.similarity import normalize
from semcache.store import Entry, Evicted, MemoryStore

NOW = 1_000.0


@pytest.fixture(params=[MemoryStore], ids=["memory"])
def store(request):
    store = request.param()
    yield store
    store.clear()


def entry(question, vector, *, key="k", value=None, at=NOW, ttl=None):
    return Entry(
        key=key,
        question=question,
        vector=normalize(vector),
        value=question.upper() if value is None else value,
        stored_at=at,
        expires_at=None if ttl is None else at + ttl,
    )


def test_an_exact_question_is_found_under_its_own_key_only(store):
    store.add(entry("reset password", [1, 0]), NOW)
    assert store.get_exact("k", "reset password", NOW).value == "RESET PASSWORD"
    assert store.get_exact("k", "Reset password", NOW) is None
    assert store.get_exact("other", "reset password", NOW) is None


def test_search_returns_the_closest_entry_and_its_score(store):
    store.add(entry("a", [1, 0]), NOW)
    store.add(entry("b", [1, 1]), NOW)
    store.add(entry("c", [0, 1]), NOW)
    found, score = store.search("k", normalize([1, 0.8]), NOW)
    assert found.question == "b"
    assert score == pytest.approx(0.9939, abs=1e-4)


def test_search_never_crosses_keys(store):
    store.add(entry("same model", [0, 1], key="model=small"), NOW)
    store.add(entry("other model", [1, 0], key="model=large"), NOW)
    found, _ = store.search("model=small", normalize([1, 0]), NOW)
    assert found.question == "same model"


def test_nothing_stored_finds_nothing(store):
    assert store.search("k", normalize([1, 0]), NOW) is None
    assert store.get_exact("k", "anything", NOW) is None
    assert len(store) == 0


def test_the_same_question_replaces_the_old_answer(store):
    store.add(entry("q", [1, 0], value="old"), NOW)
    store.add(entry("q", [1, 0], value="new"), NOW)
    assert len(store) == 1
    assert store.get_exact("k", "q", NOW).value == "new"
    assert store.search("k", normalize([1, 0]), NOW)[0].value == "new"


def test_an_entry_is_live_before_its_expiry_and_never_after(store):
    store.add(entry("q", [1, 0], ttl=60), NOW)
    assert store.get_exact("k", "q", NOW + 59.9) is not None
    assert store.search("k", normalize([1, 0]), NOW + 59.9) is not None
    assert store.get_exact("k", "q", NOW + 60) is None
    assert store.search("k", normalize([1, 0]), NOW + 60) is None


def test_an_expired_best_match_does_not_hide_a_live_one_below_it(store):
    store.add(entry("closest, expired", [1, 0], ttl=10), NOW)
    store.add(entry("a bit further, live", [1, 0.3]), NOW)
    found, _ = store.search("k", normalize([1, 0]), NOW + 20)
    assert found.question == "a bit further, live"


def test_adding_removes_expired_entries_first(store):
    store.add(entry("old", [1, 0], ttl=10), NOW)
    assert store.add(entry("new", [0, 1]), NOW + 20) == Evicted(expired=1)
    assert len(store) == 1


def test_when_full_the_least_recently_used_entry_goes(store):
    for name, vector in [("a", [1, 0]), ("b", [0, 1]), ("c", [1, 1])]:
        store.add(entry(name, vector), NOW, max_entries=3)
    assert store.add(entry("d", [1, -1]), NOW, max_entries=3) == Evicted(least_recently_used=1)
    assert store.get_exact("k", "a", NOW) is None
    assert len(store) == 3


def test_a_hit_protects_an_entry_from_eviction(store):
    for name, vector in [("a", [1, 0]), ("b", [0, 1])]:
        store.add(entry(name, vector), NOW, max_entries=2)
    store.touch(store.get_exact("k", "a", NOW), NOW + 1)
    store.add(entry("c", [1, 1]), NOW + 2, max_entries=2)
    assert store.get_exact("k", "a", NOW + 2) is not None
    assert store.get_exact("k", "b", NOW + 2) is None


def test_expired_entries_are_evicted_before_any_live_one(store):
    store.add(entry("live, used least recently", [1, 0]), NOW, max_entries=2)
    store.add(entry("expired", [0, 1], ttl=10), NOW + 1, max_entries=2)
    assert store.add(entry("new", [1, 1]), NOW + 20, max_entries=2) == Evicted(expired=1)
    assert store.get_exact("k", "live, used least recently", NOW + 20) is not None


def test_the_limit_covers_every_key_together(store):
    store.add(entry("a", [1, 0], key="one"), NOW, max_entries=2)
    store.add(entry("b", [1, 0], key="two"), NOW, max_entries=2)
    store.add(entry("c", [1, 0], key="three"), NOW, max_entries=2)
    assert len(store) == 2
    assert store.get_exact("one", "a", NOW) is None


def test_search_still_works_after_entries_are_removed(store):
    names = ["a", "b", "c", "d"]
    vectors = [[1, 0], [0, 1], [1, 1], [1, -1]]
    for name, vector in zip(names, vectors, strict=True):
        store.add(entry(name, vector), NOW, max_entries=4)
    store.add(entry("e", [-1, 0]), NOW, max_entries=4)  # evicts "a"
    for name, vector in zip(names[1:], vectors[1:], strict=True):
        found, score = store.search("k", normalize(vector), NOW)
        assert (found.question, score) == (name, pytest.approx(1.0))


def test_vectors_of_another_length_under_the_same_key_are_refused(store):
    store.add(entry("q", [1, 0]), NOW)
    with pytest.raises(ValueError, match="two embedding models"):
        store.add(entry("r", [1, 0, 0]), NOW)


def test_clear_removes_everything(store):
    store.add(entry("q", [1, 0]), NOW)
    store.clear()
    assert len(store) == 0
    assert store.get_exact("k", "q", NOW) is None
