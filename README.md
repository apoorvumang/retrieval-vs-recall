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
  deep_widesearch_standalone.py   # the harness: planner -> Exa search -> synthesis, with --closed-book
  official_grade.py     # official ByteDance evaluate_single_query, judge = gemini-2.5-flash
  gate429.py            # Exa-429 sanity gate
docs/preview.html       # the writeup + charts
```

## Run it yourself (bring your own keys)

`lib/deep_widesearch_standalone.py` runs the staged loop — planner → Exa search →
synthesis — against any OpenAI-compatible endpoint. `--closed-book` skips search and
answers from the model's own memory. Copy `.env.example` to `.env` and fill in keys.

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=...   # model-under-test + judge (Gemini OpenAI-compat endpoint)
export EXA_API_KEY=...      # search (with-search runs only)

GBASE=https://generativelanguage.googleapis.com/v1beta/openai/
COMMON="--runner python-loop --source public-widesearch --language en \
  --variant deep_staged_v2 --search-type auto --num-results 5 --fanout 3 \
  --max-searches 4 --max-contents 0 --temperature 0.75 --concurrency 2 \
  --table-format markdown --reasoning-effort none \
  --llm-base-url $GBASE --model gemini-3.1-flash-lite --llm-api-key-env GOOGLE_API_KEY"

# with search vs closed-book (drop --limit for the full 100 tasks)
python lib/deep_widesearch_standalone.py $COMMON --limit 10 --output-dir runs/normal
python lib/deep_widesearch_standalone.py $COMMON --limit 10 --closed-book --output-dir runs/closedbook
```

Then grade with the **official** ByteDance grader (unchanged; we only swap the judge LLM):

```bash
git clone https://github.com/ByteDance-Seed/WideSearch /path/to/ws_official
export WS_OFFICIAL_ROOT=/path/to/ws_official
grade(){ WS_JUDGE_MODEL=gemini-2.5-flash WS_JUDGE_BASE_URL=$GBASE WS_JUDGE_KEY_ENV=GOOGLE_API_KEY \
  python lib/official_grade.py --preds runs/$1/results.jsonl --out grades-$1.jsonl; }
grade normal; grade closedbook
```

Swap `--model` / `--llm-base-url` / `--llm-api-key-env` for any other model — e.g.
`mercury-2` (Inception), `claude-haiku-4-5` (Anthropic, add `--no-response-format`),
or `gpt-5-mini` (OpenAI).

## Roadmap

- [x] WideSearch retrieval-off ablation (this release)
- [x] Live BYO-keys harness (`lib/deep_widesearch_standalone.py`, `--closed-book` toggle)
- [ ] Fresh-2026 task set (post-cutoff events) — closed-book 0.000 → with-search 0.929
- [ ] Multi-model ablation (does every model memorize, or just some?)
- [ ] SealQA (LongSeal) temporal replication
