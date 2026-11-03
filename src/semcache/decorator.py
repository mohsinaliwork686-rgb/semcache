"""Wrapping a function: finding the question among its arguments, and keying the rest."""

import functools
import inspect
import json
import time
from collections.abc import Callable
from typing import Any

from .cache import Embedder, SemanticCache
from .store import Store


def semcache(
    *,
    threshold: float,
    embedder: Embedder,
    ttl: float | None = None,
    max_entries: int | None = None,
    store: Store | None = None,
    on: str | None = None,
    clock: Callable[[], float] = time.time,
) -> Callable[[Callable], Callable]:
    """Cache a function's results by the meaning of one argument, the question.

    Gives the function its own `SemanticCache`, reachable as `function.cache`. See
    `SemanticCache` for the settings and `SemanticCache.cached` for `on`.
    """
    cache = SemanticCache(
        threshold=threshold,
        embedder=embedder,
        ttl=ttl,
        max_entries=max_entries,
        store=store,
        clock=clock,
    )
    return cache.cached(on=on)


def wrap(cache: SemanticCache, func: Callable, on: str | None) -> Callable:
    signature = inspect.signature(func)
    question_name = on if on is not None else next(iter(signature.parameters), None)
    if question_name not in signature.parameters:
        raise TypeError(f"{func.__qualname__}() has no argument {question_name!r} to match on")
    function_name = f"{func.__module__}.{func.__qualname__}"

    def split(args: tuple, kwargs: dict) -> tuple[str, str] | None:
        """The question, and every other argument as one exact-match key; None to skip."""
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            return None  # The function raises its own error for a bad call.
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        question = arguments.pop(question_name)
        if not isinstance(question, str):
            return None
        try:
            key = json.dumps(
                {"function": function_name, "arguments": arguments},
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            # An argument with no stable text form can't be keyed, and a guessed key could
            # give one caller another's answer.
            return None
        return question, key

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call = split(args, kwargs)
        if call is None:
            cache._count(skipped=1)
            return func(*args, **kwargs)
        question, key = call
        hit, vector = cache._find(question, key)
        if hit is not None:
            return hit.value
        value = func(*args, **kwargs)
        cache._put(question, value, key, vector)
        return value

    wrapper.cache = cache
    return wrapper
