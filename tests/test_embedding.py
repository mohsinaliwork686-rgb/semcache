import importlib.util
import os

import numpy as np
import pytest

from semcache import LocalEmbedder, SemanticCache
from semcache.similarity import normalize

needs_fastembed = pytest.mark.skipif(
    importlib.util.find_spec("fastembed") is None, reason='needs pip install "semcache[local]"'
)
# Loading the real model downloads about 90 MB, so it runs only when asked for (one CI job does).
real_model = pytest.mark.skipif(
    not os.environ.get("SEMCACHE_REAL_MODEL"), reason="set SEMCACHE_REAL_MODEL=1 to load the model"
)


def test_a_missing_fastembed_is_reported_at_once_with_the_fix(monkeypatch):
    monkeypatch.setattr("semcache.embedding.importlib.util.find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r"semcache\[local\]"):
        SemanticCache(threshold=0.9)


@needs_fastembed
def test_the_model_is_not_loaded_until_first_used():
    embedder = LocalEmbedder()
    cache = SemanticCache(threshold=0.9, embedder=embedder)
    cache.stats()
    assert embedder._loaded is None


@needs_fastembed
def test_the_default_embedder_is_the_local_model():
    assert isinstance(SemanticCache(threshold=0.9)._embedder, LocalEmbedder)


@real_model
@needs_fastembed
def test_the_real_model_puts_a_paraphrase_closer_than_a_different_question():
    vectors = normalize(
        LocalEmbedder()(
            [
                "how do I reset my password",
                "i forgot my password, help",
                "what is your refund policy for orders over $500",
            ]
        )
    )
    assert vectors.shape == (3, 384)
    paraphrase, different = vectors[0] @ vectors[1], vectors[0] @ vectors[2]
    assert paraphrase > 0.7 > different
    assert np.isfinite(vectors).all()
