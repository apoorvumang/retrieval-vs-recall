#!/usr/bin/env python
"""Run the OFFICIAL ByteDance WideSearch grader on a standalone-harness results.jsonl.

We use the official `evaluate_single_query` UNCHANGED (tolerances: number_near/date_near,
norm_str, url_match, llm_judge per-column, + LLM semantic key/column alignment). The only
thing we replace is the LLM backend: their `llm_completion` (Volcengine/Ark + internal
config) -> our gpt-5-mini via the standard OpenAI client, using THEIR prompts verbatim.

Usage:
  python official_grade.py --preds <results.jsonl> --out <out.jsonl> [--workers 8] [--limit N]
  python official_grade.py --smoke      # one LLM call to verify the judge model works
"""
import argparse, json, os, sys, types, re
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.environ.get("WS_OFFICIAL_ROOT", "<WS_OFFICIAL_ROOT>")
JUDGE_MODEL = os.environ.get("WS_JUDGE_MODEL", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# 1) Stub the heavy / internal deps BEFORE importing the official grader.
#    src.utils.llm pulls in Volcengine/Ark + internal schema we don't have;
#    loguru/pandarallel are unused by evaluate_single_query's hot path.
# ---------------------------------------------------------------------------
from openai import OpenAI
# Judge model configurable: default gemini-2.5-flash (gpt-5-mini OpenAI quota dead).
# Set WS_JUDGE_BASE_URL (+ WS_JUDGE_KEY_ENV) to use a non-OpenAI OpenAI-compatible endpoint.
_base = os.environ.get("WS_JUDGE_BASE_URL")
_key = os.environ.get(os.environ.get("WS_JUDGE_KEY_ENV", "OPENAI_API_KEY"))
_client = OpenAI(api_key=_key, base_url=_base) if _base else OpenAI(api_key=_key)

class _Msg:
    def __init__(self, content): self.content = content

def _llm_completion(messages, tools=None, model_config_name="default_eval_config"):
    """Drop-in for src.utils.llm.llm_completion — uses gpt-5-mini, their prompts."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    last = None
    for attempt in range(5):
        try:
            resp = _client.chat.completions.create(model=JUDGE_MODEL, messages=messages)
            c = resp.choices[0].message.content
            if c:
                return _Msg(c)
            last = "empty content"
        except Exception as e:
            last = repr(e)[:200]
    sys.stderr.write(f"[llm_completion] failed after retries: {last}\n")
    return None

_fake_llm = types.ModuleType("src.utils.llm")
_fake_llm.llm_completion = _llm_completion
sys.modules["src.utils.llm"] = _fake_llm

_fake_loguru = types.ModuleType("loguru")
class _L:
    def __getattr__(self, n): return lambda *a, **k: None
_fake_loguru.logger = _L()
sys.modules["loguru"] = _fake_loguru

_fake_pp = types.ModuleType("pandarallel")
class _PP:
    @staticmethod
    def initialize(*a, **k): pass
_fake_pp.pandarallel = _PP()
sys.modules["pandarallel"] = _fake_pp

sys.path.insert(0, ROOT)
from src.evaluation.evaluation import evaluate_single_query  # noqa: E402
from src.evaluation.data_loader import (  # noqa: E402
    WideSearchDataLoaderHF, WideSearchResponse,
)


def smoke():
    print("JUDGE_MODEL =", JUDGE_MODEL)
    m = _llm_completion("Reply with exactly this markdown json:\n```json\n{\"idx_0\": 1}\n```")
    print("content:", None if m is None else repr(m.content[:300]))


def load_preds(path):
    """-> list of (instance_id, response_text_or_None)."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        md = r.get("query_metadata") or {}
        iid = md.get("instance_id")
        ans = r.get("answer") if r.get("ok") else None
        out.append((iid, ans))
    return out


def grade_one(loader, iid, ans):
    try:
        q = loader.load_query_by_instance_id(iid)
    except Exception as e:
        return {"instance_id": iid, "msg": f"query load fail: {e}", "f1_by_row": 0.0, "f1_by_item": 0.0,
                "recall_by_row": 0.0, "precision_by_row": 0.0, "recall_by_item": 0.0, "precision_by_item": 0.0, "score": 0.0}
    resp = None
    if ans:
        resp = WideSearchResponse(instance_id=iid, response=ans)
    res = evaluate_single_query(q, resp, eval_model_config_name="default_eval_config")
    return {
        "instance_id": iid, "score": res.score,
        "precision_by_row": res.precision_by_row, "recall_by_row": res.recall_by_row, "f1_by_row": res.f1_by_row,
        "precision_by_item": res.precision_by_item, "recall_by_item": res.recall_by_item, "f1_by_item": res.f1_by_item,
        "msg": (res.msg or "")[:200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        return smoke()

    print(f"[load] official HF dataset (gold + eval config) ...", flush=True)
    loader = WideSearchDataLoaderHF()
    preds = load_preds(a.preds)
    if a.limit:
        preds = preds[: a.limit]
    print(f"[load] {len(preds)} predictions from {a.preds}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(grade_one, loader, iid, ans): iid for iid, ans in preds}
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 10 == 0:
                print(f"  graded {done}/{len(preds)}", flush=True)

    results.sort(key=lambda r: r["instance_id"])
    if a.out:
        with open(a.out, "w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")

    n = len(results)
    def mean(k): return sum(r[k] for r in results) / n if n else 0.0
    produced = sum(1 for r in results if r["f1_by_item"] > 0 or "response is None" not in r["msg"])
    nonzero = sum(1 for r in results if r["f1_by_row"] > 0)
    print("\n===== OFFICIAL GRADER SUMMARY =====")
    print(f"preds       : {a.preds}")
    print(f"judge model : {JUDGE_MODEL}")
    print(f"n tasks      : {n}")
    print(f"row    P/R/F1: {mean('precision_by_row'):.4f} / {mean('recall_by_row'):.4f} / {mean('f1_by_row'):.4f}")
    print(f"item   P/R/F1: {mean('precision_by_item'):.4f} / {mean('recall_by_item'):.4f} / {mean('f1_by_item'):.4f}")
    print(f"exact-table  : {mean('score'):.4f}")
    print(f"tasks f1_row>0: {nonzero}/{n}")


if __name__ == "__main__":
    main()
