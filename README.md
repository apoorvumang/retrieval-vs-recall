# retrieval-vs-recall

**What happens to search-benchmark scores when you turn the search off?**

Popular "agentic search" benchmarks are often built from facts that predate a model's
training cutoff — so a model can skip the retrieved pages and answer from memory. The
score *looks* like search skill; it's parametric recall. This repo shows the effect on
**WideSearch**, with the official ByteDance grader, and gives you the code to check it
on any model.

## The headline

On WideSearch (100 EN tasks), `gemini-3.1-flash-lite` scores **the same with search
turned off** as with the full agentic-search pipeline:

| config | row-F1 (mean ± SEM, n=3) |
|---|---|
| full retrieval (search on) | 0.282 ± 0.009 |
| **closed-book (search off)** | **0.300 ± 0.006** |
| Δ (closed − normal) | **+0.018 (+6.3%)** |

Same model, harness, and grader — the only change is whether search is allowed.
Reproduce it from the committed grades (no keys needed):

```bash
python reproduce.py
```

## The clearest single example — `ws_en_034`

The task asks for UK monthly house prices and says *"cite all statistics from
government websites"* — a retrieval task by construction.

- **With search:** the agent couldn't verify figures in its sources and returned a
  blank table → row-F1 **0.03**.
- **Closed-book:** it recalled plausible government figures from memory → row-F1 **0.86**.

The grader rewards the memorized answer and penalizes the faithful one. See
`data/examples.json` for this and four more.

## What's in here

```
reproduce.py            # recompute the headline table from committed grades
data/
  gemcmp-{normal,closedbook}-r{1,2,3}.jsonl   # official grades, n=3 each → the table
  examples.json         # 5 curated cases (incl. ws_en_034) with gold + both answers
lib/
  official_grade.py     # official ByteDance evaluate_single_query, judge = gemini-2.5-flash
  gate429.py            # Exa-429 sanity gate
docs/preview.html       # the writeup + charts
```

## Regenerating from scratch (bring your own keys)

The full agentic-search harness (planner → Exa search → synthesis, with a
`--closed-book` toggle) drives any OpenAI-compatible endpoint. It's being prepared
for release — see **Roadmap**. Grading uses the unchanged official grader:

```bash
git clone https://github.com/ByteDance-Seed/WideSearch /path/to/ws_official
export WS_OFFICIAL_ROOT=/path/to/ws_official
export GOOGLE_API_KEY=...   # judge (gemini-2.5-flash) + gemini model-under-test
export EXA_API_KEY=...      # search (normal runs only)
```

Copy `.env.example` to `.env` and fill in your keys.

## Roadmap

- [x] WideSearch retrieval-off ablation (this release)
- [ ] Live BYO-keys harness (`--closed-book` toggle) — pending a cleanup pass
- [ ] Fresh-2026 task set (post-cutoff events) — closed-book 0.000 → with-search 0.929
- [ ] Multi-model ablation (does every model memorize, or just some?)
- [ ] SealQA (LongSeal) temporal replication
