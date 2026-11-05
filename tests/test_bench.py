"""The benchmark's scoring, on a set small enough to work out by hand."""

import numpy as np
import pytest

from bench.replay import Query, load_queries, replay, shuffled
from fakes import at_angle
from semcache.similarity import normalize

# Two intents that are near-misses of each other (A and B), and one unrelated (C).
A1 = Query("cancel before it ships", "cancel-before", "timing")
A2 = Query("cancel it, it hasn't shipped", "cancel-before", "timing")
B1 = Query("cancel after it ships", "cancel-after", "timing")
C1 = Query("gift card balance", "gift-card", "none")

VECTORS = {
    q.text: normalize(np.pad(np.asarray(v, dtype=np.float32), (0, 1)))
    for q, v in [
        (A1, [1, 0]),
        (A2, at_angle(0.95)),  # a paraphrase of A1
        (B1, at_angle(0.90)),  # close to A1, but needs a different answer
        (C1, [0, 1]),  # 0 against A1; never close to anything here
    ]
}
ORDER = [A1, A2, B1, C1]


def test_at_a_loose_threshold_the_near_miss_gets_the_wrong_answer():
    counts = replay(ORDER, VECTORS, threshold=0.85)
    # A1 misses and is stored; A2 hits A1 (right); B1 hits A1 (wrong); C1 misses.
    assert (counts.correct_hits, counts.false_hits, counts.misses) == (1, 1, 2)
    assert counts.false_hits_by_kind == {"timing": 1}
    assert counts.missed_chances == 0


def test_a_tighter_threshold_keeps_the_paraphrase_and_stops_the_false_hit():
    counts = replay(ORDER, VECTORS, threshold=0.92)
    assert (counts.correct_hits, counts.false_hits, counts.misses) == (1, 0, 3)


def test_at_threshold_1_only_exact_text_hits_so_the_paraphrase_is_a_missed_chance():
    counts = replay(ORDER, VECTORS, threshold=1.0)
    assert (counts.hits, counts.misses) == (0, 4)
    # A2 missed although A1, same intent, was already stored.
    assert counts.missed_chances == 1


def test_every_query_is_exactly_one_outcome():
    for threshold in (0.5, 0.85, 0.92, 1.0):
        counts = replay(ORDER, VECTORS, threshold)
        assert counts.correct_hits + counts.false_hits + counts.misses == len(ORDER)
        assert sum(counts.false_hits_by_kind.values()) == counts.false_hits
        assert counts.queries_by_kind == {"timing": 3, "none": 1}


def test_the_embedder_runs_once_per_lookup_and_never_again_to_store():
    assert replay(ORDER, VECTORS, threshold=0.85).embed_calls == 4
    # An exact-only cache embeds only to store its misses.
    assert replay(ORDER, VECTORS, threshold=1.0).embed_calls == 4


def test_the_order_depends_only_on_the_seed():
    assert shuffled(ORDER, 3) == shuffled(ORDER, 3)
    assert sorted(shuffled(ORDER, 3), key=ORDER.index) == ORDER


def test_the_real_question_set_loads_with_its_labels():
    queries = load_queries()
    assert len(queries) == 250
    assert {q.kind for q in queries} == {
        "amount",
        "negation",
        "swapped",
        "timing",
        "same-verb",
        "none",
    }
    assert sum(q.kind == "none" for q in queries) == 100


@pytest.mark.parametrize("threshold", [0.85, 0.92, 1.0])
def test_a_replay_gives_the_same_counts_every_time(threshold):
    assert vars(replay(ORDER, VECTORS, threshold)) == vars(replay(ORDER, VECTORS, threshold))
