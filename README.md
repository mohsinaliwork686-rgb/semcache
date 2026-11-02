# semcache

**Cache model answers by meaning, not exact wording. And see exactly where that goes wrong.**

> Week 7 of a 10-week build. In progress: see [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
> for the design.

---

## The problem

People ask a language model the same things all day, and every call costs time and money. A normal
cache does not help, because it matches the exact text, and nobody types a question the same way
twice. A semantic cache matches by meaning instead: if a new question is close enough to one it has
already answered, it returns that answer.

"Close enough" is the catch. These two questions are almost the same sentence:

```
what is your refund policy for orders over $500
what is your refund policy for orders under $50
```

They need opposite answers. Set the threshold loosely and the cache gives the second person the
first person's answer: wrong, instantly, and with no warning. Every semantic cache shows how much it
saves. Almost none shows where it starts doing this.

## What it does

A small Python library. One decorator wraps an existing function:

```python
from semcache import semcache


@semcache(threshold=0.9, ttl=3600)
def ask(question: str, model: str = "small") -> str:
    return call_the_model(question, model)
```

- **Matches by meaning**, with an exact-text check first so plain repeats cost nothing extra
- **Never mixes requests**: the same question to a different model, or with a different system
  prompt, never gets the other one's answer
- **Expiry and a size limit**: answers expire after a time to live, and the least recently used
  are dropped when it is full
- **In memory or in Redis**, using the vector sets added in Redis 8
- **Never breaks the call**: if the cache fails, the function runs as if it were not there

And a **benchmark** on a labelled set of support questions, built to include the near-misses that
fool a semantic cache. It shows, for every threshold, the hit rate, the money saved, and how many
wrong answers per thousand questions it would give. That chart will lead this README.

There is deliberately no default threshold. The right one depends on the embedding model and on
the questions, and the benchmark shows how to pick it.

## Status

| Day | Work | |
|-----|------|-|
| Mon | Plan and README | done |
| Tue | The labelled questions, the cache, the in-memory store | |
| Wed | Embeddings, the Redis store | |
| Thu | The benchmark and its chart, CI | |
| Fri | Publish | |

## Non-goals

- **Not a vector database.** A cache holds thousands of entries, not billions.
- **No streamed responses.** It caches whole answers.
- **No distributed invalidation.** Entries leave by age or by being least recently used.

## License

MIT. See [LICENSE](LICENSE).
