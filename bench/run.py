"""Run the benchmark: every threshold from 0.70 to 1.00, on each embedding model.

    python -m bench.run                 # both models, 5 shuffled runs each
    python -m bench.run --smoke         # one model, one run, three thresholds (CI)

Writes one JSON file per model to bench/results/. Needs `pip install -e ".[local]"`.
"""

import argparse
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from bench.replay import NO_NEAR_MISS, load_queries, replay, shuffled
from semcache import LocalEmbedder
from semcache.similarity import normalize

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
}
THRESHOLDS = [round(0.70 + step / 100, 2) for step in range(31)]
KINDS = ["amount", "negation", "swapped", "timing", "same-verb", NO_NEAR_MISS]

# The model is simulated, so the result doesn't depend on a provider's prices or network. These
# are assumptions, printed with every result: savings scale in proportion to them.
MODEL_SECONDS_PER_CALL = 1.0
MODEL_DOLLARS_PER_CALL = 0.001

RESULTS = Path(__file__).parent / "results"


def embed_all(model: str, texts: list[str]) -> tuple[dict[str, np.ndarray], float]:
    """Every question's vector, and the median time to embed one question on this machine."""
    embedder = LocalEmbedder(model)
    embedder(["warm up"])
    timings, vectors = [], {}
    for text in texts:
        start = time.perf_counter()
        vector = embedder([text])[0]
        timings.append(time.perf_counter() - start)
        vectors[text] = normalize(vector)
    return vectors, statistics.median(timings)


def sweep(model_key: str, runs: int, thresholds: list[float]) -> dict:
    queries = load_queries()
    vectors, embed_seconds = embed_all(MODELS[model_key], [q.text for q in queries])
    rows = []
    for threshold in thresholds:
        totals: dict[str, float] = {}
        by_kind_false: dict[str, int] = dict.fromkeys(KINDS, 0)
        by_kind_queries: dict[str, int] = dict.fromkeys(KINDS, 0)
        for seed in range(runs):
            counts = replay(shuffled(queries, seed), vectors, threshold)
            for name in ("queries", "correct_hits", "false_hits", "misses", "missed_chances"):
                totals[name] = totals.get(name, 0) + getattr(counts, name)
            totals["embed_calls"] = totals.get("embed_calls", 0) + counts.embed_calls
            for kind in KINDS:
                by_kind_false[kind] += counts.false_hits_by_kind.get(kind, 0)
                by_kind_queries[kind] += counts.queries_by_kind.get(kind, 0)

        n = totals["queries"]
        hits = totals["correct_hits"] + totals["false_hits"]
        model_calls_saved = hits / runs
        embedding_cost = totals["embed_calls"] / runs * embed_seconds
        rows.append(
            {
                "threshold": threshold,
                "hit_rate": hits / n,
                "correct_hit_rate": totals["correct_hits"] / n,
                "false_hits_per_1000": 1000 * totals["false_hits"] / n,
                "share_of_hits_wrong": totals["false_hits"] / hits if hits else 0.0,
                "missed_chances_per_1000": 1000 * totals["missed_chances"] / n,
                # Per run of all the questions, after paying for every embedding.
                "seconds_saved": model_calls_saved * MODEL_SECONDS_PER_CALL - embedding_cost,
                "dollars_saved": model_calls_saved * MODEL_DOLLARS_PER_CALL,
                "false_hits_per_1000_by_kind": {
                    kind: 1000 * by_kind_false[kind] / by_kind_queries[kind] for kind in KINDS
                },
            }
        )

    with_false_hits = [row["threshold"] for row in rows if row["false_hits_per_1000"] > 0]
    return {
        "model": MODELS[model_key],
        "questions": len(queries),
        "runs": runs,
        "embedding_ms_per_question": round(embed_seconds * 1000, 2),
        "simulated_model_call": {
            "seconds": MODEL_SECONDS_PER_CALL,
            "dollars": MODEL_DOLLARS_PER_CALL,
        },
        # The highest threshold at which any false hit happened.
        "first_false_hit_at": max(with_false_hits) if with_false_hits else None,
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "fastembed": version("fastembed"),
        "thresholds": rows,
    }


def print_summary(result: dict) -> None:
    print(
        f"\n{result['model']}: {result['questions']} questions x {result['runs']} runs, "
        f"{result['embedding_ms_per_question']} ms to embed one question"
    )
    print(f"first false hit at a threshold of {result['first_false_hit_at']}")
    print("threshold  hit rate  false hits/1000  wrong share  seconds saved/run")
    for row in result["thresholds"]:
        if round(row["threshold"] * 100) % 5 == 0:
            print(
                f"   {row['threshold']:.2f}    {row['hit_rate']:6.1%}"
                f"  {row['false_hits_per_1000']:15.1f}"
                f"  {row['share_of_hits_wrong']:10.1%}  {row['seconds_saved']:17.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="one model, one run, three thresholds")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    models, runs, thresholds = args.models, args.runs, THRESHOLDS
    if args.smoke:
        models, runs, thresholds = ["minilm"], 1, [0.8, 0.9, 1.0]

    args.out.mkdir(parents=True, exist_ok=True)
    for key in models:
        result = sweep(key, runs, thresholds)
        print_summary(result)
        if not args.smoke:
            (args.out / f"{key}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
