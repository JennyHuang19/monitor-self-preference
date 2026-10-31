"""Summarize diagnose_fooled.py labels.   python summarize_diagnose.py runs/diagnose_<prefix>.jsonl"""
import collections
import json
import random
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
CATS = ["rubber_stamp", "missed_in_check", "shared_error", "persuaded", "knew_but_accepted", "ambiguous_key", "PARSE_ERROR"]


def table(title, keyf, keys=None):
    c = collections.Counter((keyf(r), r["category"]) for r in rows)
    keys = keys or sorted({keyf(r) for r in rows})
    cats = [k for k in CATS if any(c[(g, k)] for g in keys)]
    print(f"\n{title}")
    print(f"  {'':16}{'n':>5} " + "".join(f"{k[:15]:>17}" for k in cats))
    for g in keys:
        n = sum(c[(g, k)] for k in cats)
        print(f"  {str(g):16}{n:>5} " + "".join(f"{c[(g, k)]:>7} ({c[(g, k)] / n:4.0%})  " if n else f"{'':>17}" for k in cats))


print(f"{len(rows)} fooled grades labelled")
table("by grader", lambda r: r["grader"])
table("by frame", lambda r: r["frame"], ["neutral", "self", "skim1k"])
table("by relation", lambda r: r["relation"], ["self", "sibling", "cross"])
table("by planted error type", lambda r: r["etype"])
table("by grader reasoning length", lambda r: "<100 tok" if r["cot_tokens"] < 100 else "100-399" if r["cot_tokens"] < 400 else "400+",
      ["<100 tok", "100-399", "400+"])
if any(r.get("solved_alone") is not None for r in rows):
    table("by whether the grader solved it alone (no pitch)", lambda r: "solved alone" if r.get("solved_alone") else "not solved", ["solved alone", "not solved"])
for flag in ("noticed_flaw", "computed_true_answer", "collusion_signal"):
    print(f"\n{flag}: {sum(bool(r.get(flag)) for r in rows)}/{len(rows)}")
random.seed(0)
for k in CATS:
    ex = [r for r in rows if r["category"] == k]
    if not ex:
        continue
    print(f"\n===== {k} ({len(ex)})")
    for r in random.sample(ex, min(3, len(ex))):
        print(f"  [{r['grader']} {r['frame']} {r['qid']} att={r['attacker']}] \"{(r.get('evidence') or '')[:220]}\"\n      -> {(r.get('explanation') or r.get('raw', ''))[:330]}")
coll = [r for r in rows if r.get("collusion_signal")]
if coll:
    print(f"\n===== collusion_signal = true ({len(coll)})")
    for r in coll:
        print(f"  [{r['grader']} {r['frame']} {r['qid']} att={r['attacker']} {r['category']}] \"{(r.get('evidence') or '')[:220]}\"\n      -> {(r.get('explanation') or '')[:330]}")
