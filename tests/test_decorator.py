import pytest

from fakes import FakeEmbedder, at_angle
from semcache import SemanticCache, semcache


@pytest.fixture
def cache():
    vectors = {"reset my password": [1, 0], "how do I reset my password": at_angle(0.95)}
    return SemanticCache(threshold=0.9, embedder=FakeEmbedder(vectors))


def test_a_paraphrase_returns_the_stored_answer_without_calling(cache):
    calls = []

    @cache.cached
    def ask(question: str) -> str:
        calls.append(question)
        return f"answer to {question}"

    assert ask("reset my password") == "answer to reset my password"
    assert ask("how do I reset my password") == "answer to reset my password"
    assert calls == ["reset my password"]


def test_the_question_is_found_by_name_or_position(cache):
    calls = []

    @cache.cached(on="question")
    def ask(model: str, question: str) -> str:
        calls.append(question)
        return question

    ask("small", "reset my password")
    ask("small", question="reset my password")
    ask(question="reset my password", model="small")
    assert len(calls) == 1


def test_other_arguments_must_match_exactly(cache):
    calls = []

    @cache.cached
    def ask(question: str, model: str = "small", system: str = "") -> str:
        calls.append((question, model, system))
        return model

    assert ask("reset my password") == "small"
    assert ask("reset my password", model="large") == "large"
    assert ask("reset my password", system="Answer in French.") == "small"
    assert len(calls) == 3
    # A default and the same value passed in are the same call.
    ask("reset my password", "small")
    assert len(calls) == 3


def test_two_functions_on_one_cache_never_share(cache):
    @cache.cached
    def summarise(text: str) -> str:
        return "summary"

    @cache.cached
    def translate(text: str) -> str:
        return "translation"

    assert summarise("reset my password") == "summary"
    assert translate("reset my password") == "translation"


def test_a_question_that_is_not_text_goes_straight_through(cache):
    calls = []

    @cache.cached
    def ask(question) -> str:
        calls.append(question)
        return "answer"

    ask(42)
    ask(42)
    assert calls == [42, 42]
    assert cache.stats().skipped == 2
    assert len(cache._store) == 0


def test_an_argument_with_no_stable_text_form_is_not_cached(cache):
    calls = []

    @cache.cached
    def ask(question: str, client: object) -> str:
        calls.append(question)
        return "answer"

    ask("reset my password", object())
    ask("reset my password", object())
    assert len(calls) == 2
    assert cache.stats().skipped == 2


def test_an_exception_is_not_cached(cache):
    attempts = []

    @cache.cached
    def ask(question: str) -> str:
        attempts.append(question)
        if len(attempts) == 1:
            raise TimeoutError("the model timed out")
        return "answer"

    with pytest.raises(TimeoutError):
        ask("reset my password")
    assert ask("reset my password") == "answer"
    assert ask("reset my password") == "answer"
    assert len(attempts) == 2


def test_a_wrong_call_raises_the_function_s_own_error(cache):
    @cache.cached
    def ask(question: str) -> str:
        return "answer"

    with pytest.raises(TypeError):
        ask("one", "too many")


def test_naming_a_missing_argument_fails_at_once(cache):
    with pytest.raises(TypeError, match="no argument 'prompt'"):

        @cache.cached(on="prompt")
        def ask(question: str) -> str:
            return "answer"


def test_semcache_gives_the_function_its_own_cache():
    @semcache(threshold=0.9, embedder=FakeEmbedder(), ttl=60)
    def ask(question: str) -> str:
        """Ask the model."""
        return "answer"

    ask("q")
    ask("q")
    assert ask.cache.stats().exact_hits == 1
    assert ask.cache.ttl == 60
    assert (ask.__name__, ask.__doc__) == ("ask", "Ask the model.")
