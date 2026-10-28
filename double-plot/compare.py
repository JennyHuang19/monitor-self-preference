"""Compare a no-debate baseline run against a debate run.

Usage:
    .venv/bin/python compare.py results/run-<ts>.jsonl results/debate-<ts>.jsonl
"""
import json
import sys
from collections import Counter, defaultdict
from statistics import mean

from run_eval import PROMPTS


def load(path):
    return [r for r in map(json.loads, open(path)) if r.get("type") != "debate"]


def split(rows):
    c = Counter(r.get("culprit", "error") if "error" not in r else "error" for r in rows)
    return c


def fmt(c):
    n = sum(c.values()) - c["error"]
    if not n:
        return "n/a"
    other = n - c["Daniel"] - c["Maya"]
    s = f"D {c['Daniel']:>2} · M {c['Maya']:>2}"
    s += f" · ? {other}" if other else "    "
    return s + (f" (err {c['error']})" if c["error"] else "")


def conf(rows):
    vals = [r["confidence"] for r in rows if isinstance(r.get("confidence"), (int, float))]
    return f"{mean(vals):.0f}%" if vals else "-"


baseline, debate = load(sys.argv[1]), load(sys.argv[2])
models = sorted({r["model"] for r in baseline} | {r["model"] for r in debate})

print(f"{'model':<18} {'prompt':<11} {'no debate':<22} {'after debate':<22} {'argued Daniel':<22} {'argued Maya':<22}")
for m in models:
    for p in PROMPTS:
        b = [r for r in baseline if r["model"] == m and r["prompt"] == p]
        d = [r for r in debate if r["model"] == m and r["prompt"] == p]
        dd = [r for r in d if r["assigned_side"] == "Daniel"]
        dm = [r for r in d if r["assigned_side"] == "Maya"]
        print(f"{m:<18} {p:<11} {fmt(split(b)):<22} {fmt(split(d)):<22} {fmt(split(dd)):<22} {fmt(split(dm)):<22}")

print("\nMean stated confidence (confidence prompt)")
for m in models:
    b = [r for r in baseline if r["model"] == m and r["prompt"] == "confidence"]
    d = [r for r in debate if r["model"] == m and r["prompt"] == "confidence"]
    print(f"  {m:<18} no debate {conf(b):>5}   after debate {conf(d):>5}")

print("\nVerdicts that matched the side the model argued")
for m in models:
    d = [r for r in debate if r["model"] == m and r.get("culprit") in ("Daniel", "Maya")]
    kept = sum(r["culprit"] == r["assigned_side"] for r in d)
    print(f"  {m:<18} {kept}/{len(d)}")

print("\nWeighs both suspects (judge's mentions_both_readings)")
for m in models:
    for label, rows in (("no debate", baseline), ("after debate", debate)):
        rs = [r for r in rows if r["model"] == m and r.get("mentions_both_readings") is not None]
        if rs:
            print(f"  {m:<18} {label:<13} {sum(r['mentions_both_readings'] for r in rs)}/{len(rs)}")
