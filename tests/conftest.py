import os
import uuid

import pytest

from semcache import MemoryStore, RedisStore


@pytest.fixture
def redis_store():
    """A RedisStore under a name no other test uses, emptied afterwards.

    Skipped when REDIS_URL is not set or Redis is not reachable, unless REQUIRE_REDIS is set,
    as it is in CI, where a missing Redis must fail rather than quietly skip.
    """
    url = os.environ.get("REDIS_URL")
    problem = None if url else "REDIS_URL is not set"
    store = None
    if url:
        store = RedisStore.from_url(url, name=f"test-{uuid.uuid4().hex}")
        try:
            store.client.ping()
        except Exception as error:
            problem = f"Redis at {url} is not reachable: {error}"
    if problem:
        if os.environ.get("REQUIRE_REDIS"):
            pytest.fail(problem)
        pytest.skip(problem)
    yield store
    store.clear()


@pytest.fixture(params=["memory", "redis"])
def store(request):
    """Each store in turn, so a test that takes `store` runs once per store."""
    if request.param == "redis":
        yield request.getfixturevalue("redis_store")
        return
    store = MemoryStore()
    yield store
    store.clear()
