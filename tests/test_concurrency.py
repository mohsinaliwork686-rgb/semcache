import random
import threading
from concurrent.futures import ThreadPoolExecutor

from fakes import FakeEmbedder
from semcache import SemanticCache


def test_many_threads_hitting_and_storing_at_once_add_up(store):
    cache = SemanticCache(threshold=0.9, embedder=FakeEmbedder(), store=store, max_entries=30)
    lock = threading.Lock()
    calls = 0

    @cache.cached
    def ask(question: str) -> str:
        nonlocal calls
        with lock:
            calls += 1
        return question.upper()

    questions = [f"question {n % 50}" for n in range(1_000)]
    random.Random(1).shuffle(questions)
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(ask, questions))

    assert answers == [q.upper() for q in questions], "no thread got another question's answer"
    stats = cache.stats()
    assert stats.hits + stats.misses == 1_000
    assert stats.misses == calls == stats.stores
    assert stats.errors == 0
    assert len(store) <= 30
