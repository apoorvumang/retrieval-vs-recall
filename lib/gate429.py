#!/usr/bin/env python
"""Run-level Exa-429 gate (Yan mandate 2026-06-24): after an eval, if the 429 rate
is >10%, the failed tasks MUST be retried (self-loop) before the run counts as
finished. This helper does the 3 sub-ops the wrapper loop needs.

  count   <results.jsonl>                          -> prints "<n429> <total>"
  subset  <results.jsonl> <src_queryset> <out>     -> writes queryset of ONLY the 429-failed instance_ids
  merge   <main_results>  <retry_results>          -> replaces 429 rows in main with retry rows (by instance_id), in place
"""
import json, sys

def is429(r): return "429" in (r.get("error") or "")
def iid(r): return (r.get("query_metadata") or {}).get("instance_id")

def main():
    cmd = sys.argv[1]
    if cmd == "count":
        rows = [json.loads(l) for l in open(sys.argv[2])]
        print(f"{sum(1 for r in rows if is429(r))} {len(rows)}")
    elif cmd == "subset":
        results, src, out = sys.argv[2], sys.argv[3], sys.argv[4]
        bad = {iid(json.loads(l)) for l in open(results) if is429(json.loads(l))}
        n = 0
        with open(out, "w") as f:
            for l in open(src):
                q = json.loads(l)
                qid = (q.get("metadata") or {}).get("instance_id") or q.get("instance_id")
                if qid in bad:
                    f.write(l); n += 1
        print(n)
    elif cmd == "merge":
        main_p, retry_p = sys.argv[2], sys.argv[3]
        retry = {iid(json.loads(l)): l for l in open(retry_p)}
        out = []
        for l in open(main_p):
            r = json.loads(l)
            rid = iid(r)
            out.append(retry[rid] if (is429(r) and rid in retry) else (l if l.endswith("\n") else l+"\n"))
        with open(main_p, "w") as f:
            for l in out:
                f.write(l if l.endswith("\n") else l+"\n")
        print("merged")
    else:
        sys.exit("usage: count|subset|merge")

if __name__ == "__main__":
    main()
