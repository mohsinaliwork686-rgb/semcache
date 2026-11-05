# Implementation plan — semcache

> Written Monday 2 November 2026, before any code. Week 7 of the 10-week programme.

## 1. Problem

Calling a language model is slow and costs money, and people ask the same things over and over.
A normal cache cannot help, because it matches the exact text, and two people almost never type a
question the same way:

```
"how do I reset my password"
"How do I reset my password?"      <- different text, a miss
"i forgot my password, help"       <- different text, a miss
```

A **semantic cache** matches by meaning instead. It turns each question into an embedding, finds
the most similar question it has answered before, and if the two are similar enough, returns the
stored answer without calling the model.

The catch is the word *enough*. Similar wording is not the same question:

```
"what is your refund policy for orders over $500"
"what is your refund policy for orders under $50"
```

At a loose threshold the cache answers the second with the answer to the first: wrong, instantly,
with no warning. That is a **false hit**, and it is worse than a miss.

Every semantic cache shows how much it saves. Almost none shows where it starts giving wrong
answers. `semcache` is a small library that does both, and its README leads with the chart of the
trade-off, including the part where the line goes bad.

## 2. Non-goals

- **Not a vector database.** Search is a straight comparison against every stored question. That
  is exact and fast for the thousands of entries a cache holds, and needs nothing installed.
- **No distributed invalidation.** Entries expire by age (TTL) or are pushed out when the cache is
  full (LRU). There is no "forget everything about refunds" across machines.
- **No streamed responses.** The wrapped function must return a whole value.
- **No request coalescing.** If two identical questions miss at the same moment, both call the
  model. Avoiding that is a different feature.
- **No default threshold** (see section 4).

## 3. What the user writes

The simple form wraps one function:

```python
from semcache import semcache


@semcache(threshold=0.9, ttl=3600)
def ask(question: str, model: str = "small") -> str:
    return call_the_model(question, model)
```

When the cache is shared, or needs a Redis store, it is created once and used as a decorator:

```python
from semcache import SemanticCache, RedisStore

cache = SemanticCache(
    threshold=0.9,
    ttl=3600,
    max_entries=10_000,
    store=RedisStore.from_url("redis://localhost:6379/0", name="support"),
)


@cache.cached(on="question")
async def ask(question: str, model: str = "small") -> str: ...


cache.stats()  # hits, misses, evictions, expirations, errors...
```

Sync and async functions both work. `lookup()` and `store()` are public too, for code that cannot
use a decorator; that is how `llm-router` could use it.

### Which argument is "the question"

`on=` names the argument that is matched by meaning. It defaults to the first parameter. **Every
other argument must match exactly.** The same question asked of a different model, or with a
different system prompt, is a different request, and must not get the other one's answer. Those
other arguments are turned into a stable key (JSON, sorted), and only entries with the same key are
searched. If the question argument is not a string, the call goes straight through and is counted
as skipped.

## 4. How a lookup works

1. **Exact match first.** If this exact text has been stored (under the same key), it is a hit.
   No embedding needed, so an exact repeat is as fast as a normal cache.
2. **Otherwise embed the question** and compare it with every live stored question under the same
   key. If the best similarity is at least the threshold, it is a hit.
3. **Otherwise call the function**, store the answer, and return it.

Similarity is cosine similarity between normalised vectors: 1 means the same direction, around 0
means unrelated.

**A threshold of 1.0 turns off meaning-matching**, leaving exactly a normal exact-text cache. That
is defined rather than left to floating point: identical text can embed to 0.99999994 instead of
1.0, so "at least 1.0" would otherwise sometimes miss its own question.

**The threshold is required.** There is no safe value that works everywhere: it depends on the
embedding model and on how close together the questions in a domain are. The benchmark will show
the same threshold landing in very different places on two models. A default would quietly pick a
point on that curve for the user; the README shows how to choose one instead.

**The cache never breaks the call.** If the embedder or the store fails (Redis down, a model that
will not load), the function is called as if there were no cache, and the error is counted.

## 5. The two decisions the roadmap left for Monday

### 5a. What counts as a false hit, and how the benchmark stays honest

The benchmark needs to know, for any two questions, whether one answer serves both. So it uses a
**labelled set of questions grouped by intent**: every question in a group is answered by the same
answer, and questions in different groups are not.

- About **50 intents, around 250 questions**, for a made-up online shop's support desk: orders,
  refunds, shipping, accounts, payments, product care.
- Each intent has a plain question and 3 to 5 rewordings: casual, misspelt, reordered, synonyms,
  extra words.
- **Near-miss intents are built in on purpose**, in pairs that share most of their words but need
  different answers: amounts ("over $500" / "under $50"), negation ("can I" / "can't I"), swapped
  things ("iPhone app" / "Android app"), timing ("before it ships" / "after it ships"), and the same
  verb about different things ("cancel my order" / "cancel my subscription").

A query **hits correctly** when the stored question it matched has the same intent. It is a
**false hit** when it matched a question with a different intent. A **missed chance** is a miss
when a question with the same intent was already stored.

The honesty rules, written down now so they cannot drift:

1. **The set is written and committed before any code that embeds it exists.** The git history
   shows the order.
2. **Labels are decided by reading, never by looking at a similarity score.**
3. **After the first run, a question can only change to fix a wrong label**, and every change goes
   into `bench/CHANGES.md` with the reason. Hard cases are never removed because they score badly.
4. **Results are reported per kind of near-miss**, not only overall, so a good average cannot hide
   one category that always fails. (`evalboard`'s whole point, applied to this week.)

The run itself:

- The questions are replayed in a shuffled order (fixed seeds, several runs, averaged). Each one is
  looked up; a miss "calls the model" and stores the answer.
- The model is simulated: a fixed cost and delay per call, stated in the results, so the numbers do
  not depend on a provider's price list or network. Embedding time is real, measured on the
  machine, and **subtracted from the time saved**, because the cache pays it on every lookup.
- For each threshold from 0.70 to 1.00, it reports: hit rate, **false hits per 1,000 queries**, the
  share of hits that were wrong, missed chances, time saved and money saved.
- It runs on **two embedding models**, to show that a threshold only means something for the model
  it was measured on.
- One chart: hit rate and false-hit rate against the threshold, with the point where the first
  false hit appears marked. It goes at the top of the README.

### 5b. How TTL and LRU work together

- **TTL counts from when the answer was stored.** A hit does not extend it: an answer goes stale
  with age, not with popularity.
- **An expired entry is never returned.** Expired entries are left out of the search, so the best
  *live* match wins; an expired entry that happens to be the closest does not turn a hit into a miss.
- **A hit counts as use for LRU.** Storing counts as use too.
- **When the cache is full and a new answer arrives, expired entries go first.** Only if it is still
  full is the least recently used live entry evicted. An expired entry never pushes out a live one.
- `max_entries` is one limit for the whole cache, across all exact-match keys.
- Time comes from a clock that is passed in (defaulting to the real one), so tests control it.

## 6. Stores

Both stores pass **the same test suite**, run once against each.

**`MemoryStore`.** Per key, a matrix of stored vectors and their entries. Search is one matrix
multiplication followed by a mask for expired rows. A lock makes it safe across threads. Values are
kept as the objects the function returned.

**`RedisStore`.** Uses **vector sets**, the data type added in Redis 8, so the search happens inside
Redis instead of downloading every vector. Checked today against Redis 8.10 with redis-py 8.1:

- `VSIM ... WITHSCORES TRUTH` does an exact search (not the approximate one), and its score is
  exactly (cosine + 1) / 2, so it converts back with no loss.
- Vectors are stored with `NOQUANT`. The default 8-bit storage moves scores by about 0.0001, which
  is enough to flip a result sitting right on the threshold, and would make the two stores disagree.
- Each entry carries its expiry time as an attribute, and `FILTER ".expires > now"` leaves expired
  entries out of the search, returning the best live one.
- Values are stored as **JSON, never pickle**: unpickling data from a shared Redis lets anyone who
  can write to it run code. A function whose return value is not JSON-serialisable is refused with a
  clear error when it is first stored.
- LRU order lives in a sorted set (entry, last used). The capacity is enforced on insert, in a
  transaction. Under many simultaneous writers it can briefly overshoot by about the number of
  writers; that is documented and tested rather than hidden.
- It checks the server version when it connects and says plainly if it is older than Redis 8.

## 7. Embeddings

**Decided today, against the roadmap: `fastembed` instead of `sentence-transformers`.** Same
models (it runs `all-MiniLM-L6-v2`, the one `ragsweep` used, and `bge-small-en-v1.5`), but through
ONNX Runtime instead of PyTorch. Found today:

- On this machine (an Intel Mac on macOS 12), pip offers nothing newer than PyTorch 2.2.2, from
  early 2024. Current `sentence-transformers` (6.2) accepts 2.2 at the oldest, so this machine sits
  one requirement bump away from not installing it at all. `fastembed` 0.9.1 installed here today,
  with both models listed (about 90 MB and 67 MB).
- ONNX Runtime has wheels for Python 3.11 to 3.14 on Linux, Windows and Apple Silicon. The one gap
  is Python 3.14 on an Intel Mac, which CI covers on other platforms.
- The install is far smaller than PyTorch.

The embedder is pluggable: anything that turns a list of strings into an array of vectors. The core
install needs only numpy; the local model comes with `pip install "semcache[local]"`, Redis with
`[redis]`, the benchmark's chart with `[bench]`. The README shows a three-line adapter for
`sentence-transformers` for anyone who already has it.

Tests use a fake embedder with fixed vectors, so they run instantly and never download a model.
One test, in one CI job, loads the real model to prove the wiring works.

## 8. Architecture

| Module | Job |
|--------|-----|
| `similarity.py` | **Pure.** Normalising vectors, cosine scores, picking the best match above a threshold |
| `embedding.py` | The embedder interface, and the `fastembed` one, loaded on first use |
| `store/base.py` | The store interface and the `Entry` record |
| `store/memory.py` | The in-memory store |
| `store/redis.py` | The Redis store, on vector sets |
| `cache.py` | `SemanticCache`: the exact layer, lookup, store, TTL, LRU, statistics, failing open |
| `decorator.py` | `semcache(...)` and `cache.cached(...)`: binding arguments, the exact-match key, sync and async |
| `bench/` | The labelled questions, the runner, the chart |

## 9. Build order

| Day | Work |
|-----|------|
| **Mon** | This plan and the README pitch, committed first. Then the project skeleton and CI |
| **Tue** | **The labelled question set first**, committed before any embedding code. Then `similarity.py`, `MemoryStore`, TTL and LRU, `SemanticCache` and the decorator, all tested with the fake embedder |
| **Wed** | `embedding.py`, the Redis store, the shared store test suite, async, failing open, the concurrency test |
| **Thu** | The benchmark runner, both models, the chart; CI with a Redis 8 service and a benchmark smoke run |
| **Fri** | README with the chart at the top, publish, portfolio, roadmap and CV |

## 10. Test plan

| Area | Must work | Must not break |
|------|-----------|----------------|
| similarity | identical vectors score 1, opposite −1; the best of several is picked | zero vectors; an empty store; ties |
| lookup | an exact repeat is a hit without embedding; a paraphrase above the threshold hits, below it misses | threshold 1.0 behaves exactly like an exact-text cache; another model or system prompt never shares an answer |
| TTL | an entry is a hit before its expiry and never after (clock passed in) | an expired best match does not hide a live one just below it; a hit does not extend the TTL |
| LRU | when full, the least recently used entry goes; a hit protects an entry | expired entries are evicted before any live one |
| stores | every store test passes on both stores | the Redis store's scores match the memory store's to within 1e-6 |
| decorator | sync and async functions; the question found by name or position | a non-string question goes straight through; an exception from the function is not cached |
| failure | Redis down or the embedder failing: the function still runs and the error is counted | the cache never raises into the caller's code |
| concurrency | many threads hitting and storing at once | the statistics add up exactly afterwards |
| benchmark | a tiny fixed set gives hand-worked hit and false-hit counts | every query is exactly one of a correct hit, a false hit or a miss; a replay with the same seed gives the same counts (changed 5 November, see below) |

**Changed on Thursday 5 November.** The benchmark row first said "raising the threshold never
adds hits". That is not true of a cache that stores only its misses: a higher threshold turns a
hit into a miss, the question is then stored, and a later paraphrase can hit it. For example, with
question 2 at 0.85 to question 1, and question 3 at 0.95 to question 2 but 0.75 to question 1, a
threshold of 0.8 gives a hit for question 2 only, while 0.9 gives a hit for question 3 only. The
row now tests what must hold instead.

## 11. Risks

| Risk | Mitigation |
|------|------------|
| A hand-written question set flatters the cache | The rules in 5a: written first, labelled by reading, changes logged, results per category |
| The first use downloads a model (~90 MB) | Loaded lazily, said in the README; unit tests never load it; CI caches it |
| redis-py 8.1.0 crashes on `VSIM ... WITHSCORES` against a missing key under RESP3 (redis-py 8's default). Found today; fixed on redis-py master in #4496, not yet released | The Redis store guards against the missing-key case itself, with a test, until a release has the fix |
| The two stores disagree right at the threshold | `NOQUANT` in Redis; the shared suite compares scores |
| The "money saved" number looks like a promise | Stated as a simulation, with the cost per call printed next to it |
| The PyPI name `semcache` is taken | Installed from the repository, like the earlier projects. A published release would need another name |

## 12. Definition of done

- [ ] Public repo, description and topics
- [ ] This plan, committed before the code
- [ ] The labelled question set committed before any embedding code
- [ ] README with the trade-off chart at the top, and how to choose a threshold
- [ ] Every row of the test plan covered; both stores run the same suite
- [ ] CI green: lint, tests on Python 3.11 to 3.14 with a real Redis 8, other operating systems, a benchmark smoke run
- [ ] At least 10 commits, one logical change each
