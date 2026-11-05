"""The README's chart: right and wrong answers from the cache, against the threshold, per model.

    python -m bench.chart

Reads bench/results/*.json and writes a light and a dark version to docs/images/. Needs
`pip install -e ".[bench]"`.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent.parent
RESULTS = Path(__file__).parent / "results"
OUT = ROOT / "docs" / "images"
PANELS = [("minilm", "all-MiniLM-L6-v2"), ("bge-small", "bge-small-en-v1.5")]

# Categorical slots 1 and 2, each stepped for its own surface; both pairs pass the palette
# validator (lightness, chroma, colour-blind separation, contrast) in their mode.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "right": "#2a78d6",
        "wrong": "#eb6834",
        "ink": "#0b0b0b",
        "ink_2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "right": "#3987e5",
        "wrong": "#d95926",
        "ink": "#f0efec",
        "ink_2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}


def load(key: str) -> dict:
    return json.loads((RESULTS / f"{key}.json").read_text())


def draw(theme: dict, path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.edgecolor": theme["axis"],
            "axes.labelcolor": theme["ink_2"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "text.color": theme["ink"],
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True, dpi=200)
    fig.patch.set_facecolor(theme["surface"])

    for ax, (key, name) in zip(axes, PANELS, strict=True):
        result = load(key)
        rows = result["thresholds"]
        x = [row["threshold"] for row in rows]
        right = [100 * row["correct_hit_rate"] for row in rows]
        wrong = [row["false_hits_per_1000"] / 10 for row in rows]

        ax.set_facecolor(theme["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

        ax.plot(
            x,
            right,
            color=theme["right"],
            linewidth=2,
            solid_capstyle="round",
            label="right answer from the cache",
        )
        ax.plot(
            x,
            wrong,
            color=theme["wrong"],
            linewidth=2,
            solid_capstyle="round",
            label="wrong answer from the cache",
        )

        # Where the last wrong answer happens, going up the thresholds.
        last = result["first_false_hit_at"]
        if last is not None:
            y = wrong[x.index(last)]
            ax.plot(
                [last],
                [y],
                marker="o",
                markersize=6,
                color=theme["wrong"],
                markeredgecolor=theme["surface"],
                markeredgewidth=2,
                zorder=3,
            )
            ax.annotate(
                f"still wrong\nat {last:.2f}",
                (last, y),
                xytext=(-6, 14),
                textcoords="offset points",
                ha="right",
                va="bottom",
                color=theme["ink_2"],
                fontsize=8,
            )

        ax.set_title(name, loc="left", color=theme["ink"], fontsize=10, pad=8)
        ax.set_xlim(0.695, 1.005)
        ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.2f}")
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
        ax.set_xlabel("similarity threshold")

    axes[0].set_ylabel("share of questions")
    axes[0].set_ylim(0, 55)
    legend = fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=8.5,
        labelcolor=theme["ink_2"],
        handlelength=1.6,
        bbox_to_anchor=(0.99, 1.0),
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        path = OUT / f"tradeoff-{mode}.png"
        draw(theme, path)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
