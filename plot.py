#!/usr/bin/env python3
"""Render the figures used in the writeup into figures/*.png.

    python plot.py

Numbers are the measured results (official ByteDance grader, gemini-2.5-flash judge):
- retrieval-off ablation & multi-model Δ: WideSearch 100 EN tasks, row-F1
  (Gemini & Mercury n=3; Haiku & gpt-5-mini n=3).
- contamination check: Gemini, public WideSearch vs fresh-2026 tasks, closed-book vs with-search.
- SealQA (LongSeal): accuracy by answer era, docs-given vs closed-book, mean of n=3.
"""
import csv
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
os.makedirs(FIG, exist_ok=True)


def _csv(rel):
    with open(os.path.join(DATA, rel)) as f:
        return list(csv.DictReader(f))

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
    disp = {"claude-haiku-4-5": "Claude Haiku-4.5", "mercury-2": "Mercury-2",
            "gpt-5-mini": "gpt-5-mini", "gemini-3.1-flash-lite": "Gemini-3.1-flash-lite"}
    rows = _csv("widesearch_by_model.csv")
    models = [disp.get(r["model"], r["model"]) for r in rows]
    delta = [round(float(r["search_f1"]) - float(r["closed_f1"]), 3) for r in rows]
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
    c = json.load(open(os.path.join(DATA, "fresh2026", "results.json")))["contamination_gemini_row_f1"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    _grouped(ax,
             ["Public WideSearch\n(facts in training data)", "Fresh 2026 tasks\n(after cutoff)"],
             [("With search", [c["public_with_search"], c["fresh_with_search"]], SEARCH),
              ("Closed-book (memory)", [c["public_closed_book"], c["fresh_closed_book"]], MEMORY)],
             "Gemini-3.1: search on vs off",
             "row-F1 · WideSearch official grader")
    fig.tight_layout()
    fig.savefig(f"{FIG}/contamination_check.png", bbox_inches="tight")
    plt.close(fig)


def fig_sealqa():
    """Gemini vs Mercury answering from memory (closed-book), by answer era.
    Gemini's lead is all in pre-cutoff eras; on 2026 the ranking flips."""
    rows = _csv("sealqa/by_era.csv")
    eras = [r["effective_year"] for r in rows]
    nq = [int(r["n"]) for r in rows]
    gem = [float(r["gemini_search_off"]) for r in rows]
    gem_sd = [float(r["gemini_search_off_sd"]) for r in rows]
    m2 = [float(r["m2_search_off"]) for r in rows]
    m2_sd = [float(r["m2_search_off_sd"]) for r in rows]
    GEM_C = "#5b6573"   # slate (incumbent)
    M2_C = SEARCH       # blue (ours)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    x = list(range(4))
    w = 0.38
    ax.axvspan(2.5, 3.5, color="#fff3ea", zorder=0)  # highlight 2026
    ekw = dict(ecolor="#0000004d", lw=1.1, capsize=3)
    ax.bar([i - w / 2 for i in x], gem, w, yerr=gem_sd, error_kw=ekw,
           label="Gemini-3.1-flash-lite", color=GEM_C, zorder=3)
    ax.bar([i + w / 2 for i in x], m2, w, yerr=m2_sd, error_kw=ekw,
           label="Mercury-2", color=M2_C, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\n(n={n})" for e, n in zip(eras, nq)])
    ax.set_ylim(0, 0.38)
    ax.set_ylabel("accuracy (closed-book / memory)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    _clean(ax)
    ax.annotate("ranking flips —\nMercury leads", xy=(3.19, 0.16), xytext=(2.1, 0.30),
                fontsize=10, color=M2_C, fontweight="bold", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=M2_C, lw=1.4))
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.set_title("SealQA — answering from memory (closed-book)",
                 fontsize=14, fontweight="bold", loc="left", pad=24)
    ax.text(0, 1.06, "Gemini's lead is all in pre-2024 answers; cross into post-cutoff 2026 and it flips.",
            transform=ax.transAxes, fontsize=10.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sealqa_temporal.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_delta()
    fig_contamination()
    fig_sealqa()
    print("wrote:", ", ".join(sorted(os.listdir(FIG))))
