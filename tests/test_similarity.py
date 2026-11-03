import numpy as np
import pytest

from semcache.similarity import best_match, cosine_scores, normalize


def score(a, b):
    return float(cosine_scores(normalize([a]), normalize(b))[0])


def test_identical_vectors_score_1_and_opposite_score_minus_1():
    assert score([3, 4], [3, 4]) == pytest.approx(1.0)
    assert score([3, 4], [-3, -4]) == pytest.approx(-1.0)
    assert score([1, 0], [0, 1]) == pytest.approx(0.0)


def test_length_does_not_matter():
    assert score([1, 1], [100, 100]) == pytest.approx(1.0)


def test_a_score_never_goes_past_1():
    vector = normalize(np.full(384, 0.1))
    scores = cosine_scores(np.stack([vector] * 50), vector)
    assert scores.max() <= 1.0


def test_a_zero_vector_scores_0_instead_of_nan():
    assert np.all(normalize([[0, 0], [0, 0]]) == 0)
    assert score([0, 0], [1, 0]) == 0.0


def test_the_best_of_several_is_picked():
    matrix = normalize([[1, 0], [1, 1], [0, 1]])
    assert best_match(cosine_scores(matrix, normalize([1, 0.9])))[0] == 1


def test_the_earliest_row_wins_a_tie():
    assert best_match(np.array([0.5, 0.9, 0.9]))[0] == 1


def test_rows_that_are_not_live_are_left_out():
    scores = np.array([0.9, 0.8, 0.3])
    assert best_match(scores, live=np.array([False, True, True])) == (1, pytest.approx(0.8))
    assert best_match(scores, live=np.array([False, False, False])) is None


def test_nothing_to_match_gives_none():
    assert best_match(np.array([])) is None
