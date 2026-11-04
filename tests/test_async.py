import asyncio
import inspect
import threading

from fakes import FakeEmbedder, at_angle
from semcache import SemanticCache


def make_cache():
    vectors = {"reset my password": [1, 0], "how do I reset my password": at_angle(0.95)}
    return SemanticCache(threshold=0.9, embedder=FakeEmbedder(vectors))


def test_an_async_function_stays_async_and_is_cached():
    cache = make_cache()
    calls = []

    @cache.cached
    async def ask(question: str) -> str:
        calls.append(question)
        await asyncio.sleep(0)
        return f"answer to {question}"

    assert inspect.iscoroutinefunction(ask)

    async def main():
        first = await ask("reset my password")
        second = await ask("how do I reset my password")
        return first, second

    assert asyncio.run(main()) == ("answer to reset my password",) * 2
    assert calls == ["reset my password"]


def test_the_cache_work_runs_off_the_event_loop_thread():
    seen = []

    def embedder(texts):
        seen.append(threading.current_thread())
        return FakeEmbedder()(texts)

    cache = SemanticCache(threshold=0.9, embedder=embedder)

    @cache.cached
    async def ask(question: str) -> str:
        return "answer"

    async def main():
        await ask("q")
        return threading.current_thread()

    loop_thread = asyncio.run(main())
    assert seen and all(thread is not loop_thread for thread in seen)


def test_an_async_call_with_a_question_that_is_not_text_goes_straight_through():
    cache = make_cache()

    @cache.cached
    async def ask(question) -> str:
        return "answer"

    assert asyncio.run(ask(42)) == "answer"
    assert cache.stats().skipped == 1


def test_many_async_calls_at_once_add_up():
    cache = make_cache()

    @cache.cached
    async def ask(question: str) -> str:
        await asyncio.sleep(0)
        return question

    questions = [f"question {n % 20}" for n in range(200)]

    async def main():
        return await asyncio.gather(*(ask(q) for q in questions))

    assert asyncio.run(main()) == questions
    stats = cache.stats()
    assert stats.hits + stats.misses == 200
    assert stats.stores == stats.misses
    assert stats.errors == 0
