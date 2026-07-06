# X thread — draft

> Rendered version with charts: `docs/preview.html`
> Tweet 1's multi-model chart is pending the gpt-5-mini + Haiku ablation run.

**1/** Popular "agentic search" benchmarks may be grading memory, not search.
On WideSearch we turned retrieval *off* — and the score didn't drop, it went *up*
(0.282 → 0.300 row-F1, n=3, official grader). If disabling search doesn't hurt, the
benchmark isn't measuring search. 🧵
[chart: retrieval-off, multiple models]

**2/** Why: WideSearch tasks are built from pre-cutoff facts — rankings, product
specs, house prices — already in the model's weights. So a model can ignore the
retrieved pages and just recall the answer. The score looks like search skill. It's recall.

**3/** The clean proof: run the same model closed-book vs with full search, on the
public set and on fresh post-cutoff tasks.
Public: memory 0.305 ≈ search 0.290 → retrieval adds nothing.
Fresh 2026: memory 0.000, search 0.929. When facts can't be memorized, search is everything.
[chart: contamination check]

**4/** One task makes it visceral. ws_en_034 asks for UK monthly house prices and says
"cite statistics from government websites."
• With search → couldn't verify sources, blank table → 0.03
• Closed-book → recalled plausible figures → 0.86
The benchmark rewards the memorized answer and penalizes the faithful one.
[chart: ws_en_034 table]

**5/** So we built a fresh slice from 2026 events — French Open, World Cup, Cannes,
Eurovision — all after the models' training cutoff. Same format; facts nothing could
have memorized.

**6/** On fresh data the illusion is gone — and the models are basically tied:
Gemini-3.1 0.929, Mercury-2 0.923. With nothing to memorize, all that's left is
retrieval + synthesis — and Mercury matches a frontier model.
[chart: fresh-data tie]

**7/** It's not just WideSearch. On SealQA (contamination-resistant, refreshed monthly)
Gemini scores higher answering from memory than from the retrieved documents — and the
gap is widest on older facts. (Caveat: SealQA's docs are adversarially noisy by design,
so it's a replication of the pattern, not the clean proof — that's WideSearch.)
[chart: SealQA by era]

**8/** If you benchmark search, three checks:
① Run a retrieval-off ablation. If the score barely moves, you're grading recall.
② Include post-cutoff items.
③ Verify gold is actually retrievable, not just true.

**9/** Full repro — the ablation, the fresh-2026 set, the official grader, every
example above. Bring your own keys and run it on any model:
https://github.com/apoorvumang/retrieval-vs-recall
