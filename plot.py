#!/usr/bin/env python3
"""Render the figures used in the writeup into figures/*.png.

    python plot.py

Numbers are the measured results (official ByteDance grader, gemini-2.5-flash judge):
- retrieval-off ablation & multi-model Δ: WideSearch 100 EN tasks, row-F1
  (Gemini & Mercury n=3; Haiku & gpt-5-mini n=3).
- contamination check: Gemini, public WideSearch vs fresh-2026 tasks, closed-book vs with-search.
- SealQA (LongSeal): accuracy by answer era, docs-given vs closed-book, mean of n=3.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)

SEARCH = "#2a78d6"   # retrieval
MEMORY = "#eb6834"   # parametric recall
INK = "#16202c"
MUTED = "#8a8880"
GRID = "#e3e2db"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
})


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_delta():
    """Headline: row-F1 gain from search (on - off), per model."""
    models = ["Claude Haiku-4.5", "Mercury-2", "gpt-5-mini", "Gemini-3.1-flash-lite"]
    delta = [0.189, 0.127, 0.032, -0.018]
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    y = range(len(models))
    colors = [SEARCH if d >= 0 else MEMORY for d in delta]
    ax.barh(list(y), delta, color=colors, height=0.62, zorder=3)
    ax.axvline(0, color="#c3c2b7", lw=1.2, zorder=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 0.24)
    ax.set_xlabel("row-F1 gain from turning search on  (search − closed-book)")
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    _clean(ax)
    for yi, d in zip(y, delta):
        ax.text(d + (0.006 if d >= 0 else -0.006), yi, f"{d:+.2f}",
                va="center", ha="left" if d >= 0 else "right",
                fontweight="bold", fontsize=11,
                color=SEARCH if d >= 0 else MEMORY)
    ax.set_title("What web search actually adds",
                 fontsize=14, fontweight="bold", loc="left", pad=24)
    ax.text(0, 1.06, "Every model gains from retrieval — except Gemini, which does worse with it.",
            transform=ax.transAxes, fontsize=10.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(f"{FIG}/delta_search_contribution.png", bbox_inches="tight")
    plt.close(fig)


def _grouped(ax, groups, series, title, sub, ymax=1.0):
    n = len(groups)
    w = 0.38
    x = range(n)
    (l1, v1, c1), (l2, v2, c2) = series
    b1 = ax.bar([i - w / 2 for i in x], v1, w, label=l1, color=c1, zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], v2, w, label=l2, color=c2, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    _clean(ax)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + ymax * 0.015, f"{h:.3f}".rstrip("0").rstrip("."),
                    ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=18)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=10.5, color=MUTED)


def fig_contamination():
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    _grouped(ax,
             ["Public WideSearch\n(facts in training data)", "Fresh 2026 tasks\n(after cutoff)"],
             [("With search", [0.290, 0.929], SEARCH),
              ("Closed-book (memory)", [0.305, 0.000], MEMORY)],
             "Gemini-3.1: search on vs off",
             "row-F1 · WideSearch official grader")
    fig.tight_layout()
    fig.savefig(f"{FIG}/contamination_check.png", bbox_inches="tight")
    plt.close(fig)


def fig_sealqa():
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    _grouped(ax,
             ["before 2024", "2024", "2025", "2026", "overall"],
             [("With docs (open-book)", [0.233, 0.185, 0.020, 0.062, 0.154], SEARCH),
              ("Closed-book (memory)", [0.313, 0.272, 0.080, 0.104, 0.223], MEMORY)],
             "SealQA (LongSeal) — Gemini, by answer era",
             "accuracy, mean of n=3 · closed-book ≥ with-docs everywhere",
             ymax=0.36)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sealqa_temporal.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_delta()
    fig_contamination()
    fig_sealqa()
    print("wrote:", ", ".join(sorted(os.listdir(FIG))))
