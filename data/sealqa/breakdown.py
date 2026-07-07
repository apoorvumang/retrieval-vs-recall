#!/usr/bin/env python3
"""n=3 SealQA breakdown by effective_year with mean±std per cell."""
import json, os, statistics as st
from collections import defaultdict
from datasets import load_dataset

CONDS = ["gemini-open", "gemini-closed", "m2-open", "m2-closed"]
REPS = [1, 2, 3]
ds = load_dataset("vtllms/sealqa", "longseal", split="test")
yr = [str(y) for y in ds["effective_year"]]
YEARS = ["before 2024", "2024", "2025", "2026"]

def rep_correct(cond, r):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"{cond}-r{r}.jsonl")
    if not os.path.exists(p): return None
    rows = [json.loads(l) for l in open(p)]
    return [bool(x.get("correct")) for x in rows] if len(rows) == len(ds) else None

# per condition: list of per-rep correctness arrays
data = {c: [a for r in REPS if (a := rep_correct(c, r)) is not None] for c in CONDS}
for c in CONDS:
    print(f"{c}: {len(data[c])} reps loaded")

def cell(cond, mask):
    # mean±std of accuracy over reps, restricted to examples where mask[i] True
    idx = [i for i in range(len(ds)) if mask(i)]
    accs = [sum(1 for i in idx if arr[i]) / len(idx) for arr in data[cond]] if idx else []
    if not accs: return "   n/a   "
    m = st.mean(accs); s = st.stdev(accs) if len(accs) > 1 else 0.0
    return f"{m:.3f}±{s:.3f}"

def row(label, n, mask):
    print(f"{label:14} {n:>4}  " + "  ".join(f"{cell(c, mask):>12}" for c in CONDS))

print("\nSealQA longseal — accuracy mean±std (n=3)   [gemini temp0 -> ~0 std; m2 temp0.7 -> real std]\n")
print(f"{'effective_year':14} {'n':>4}  " + "  ".join(f"{c:>12}" for c in CONDS))
row("OVERALL", len(ds), lambda i: True)
for y in YEARS:
    row(y, yr.count(y), lambda i, y=y: yr[i] == y)
