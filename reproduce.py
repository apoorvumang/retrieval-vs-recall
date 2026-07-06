#!/usr/bin/env python3
"""Reproduce the headline WideSearch retrieval-off result from committed grades.

Reads the official-grader outputs in data/ (3 runs each of the full agentic-search
pipeline vs the same pipeline with search disabled) and prints the row-F1 table.

    python reproduce.py

No API keys needed — this recomputes the numbers from the committed grade files.
To regenerate those grades from scratch on any model, see README.md (BYO keys).
"""
import glob
import json
import math
import os

DATA = os.path.join(os.path.dirname(__file__), "data")


def load(kind):
    """Return list of per-run mean row-F1 for kind in {normal, closedbook}."""
    per_run = []
    for path in sorted(glob.glob(os.path.join(DATA, f"gemcmp-{kind}-r*.jsonl"))):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        per_run.append(sum(r["f1_by_row"] for r in rows) / len(rows))
    return per_run


def mean_sem(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd / math.sqrt(len(xs))


def main():
    normal, closed = load("normal"), load("closedbook")
    (nm, ns), (cm, cs) = mean_sem(normal), mean_sem(closed)
    print("WideSearch (100 EN tasks) — gemini-3.1-flash-lite — official ByteDance grader\n")
    print(f"  {'config':<26}{'row-F1 (mean ± SEM, n=3)':<28}per-run")
    print(f"  {'full retrieval (search on)':<26}{nm:.3f} ± {ns:.3f}{'':<15}{[round(x,3) for x in normal]}")
    print(f"  {'closed-book (search off)':<26}{cm:.3f} ± {cs:.3f}{'':<15}{[round(x,3) for x in closed]}")
    print(f"\n  Δ (closed − normal) = {cm - nm:+.3f}  "
          f"({(cm - nm) / nm * 100:+.1f}%)")
    print("\n  Disabling search does not reduce the score — retrieval's contribution to")
    print("  the headline metric is ~zero. The benchmark is measuring parametric recall.")


if __name__ == "__main__":
    main()
