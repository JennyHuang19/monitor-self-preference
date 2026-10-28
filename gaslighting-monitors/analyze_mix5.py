"""math3 / trivia3 on the mix5 pool: fooled rates by grader, attacker, relation, frame and item type, next to the
no-pitch baseline. "knew" = the grader solved the item with no pitch. Writes runs/summary_mix5.json."""
import json
from collections import defaultdict

from common import POOLS, RUNS, load_questions, read_jsonl, run_dir

FRAMES = {"neutral": "grades.jsonl", "self": "grades_self.jsonl", "skim1k": "grades_skim1k.jsonl"}
MODELS = [m["name"] for m in POOLS["mix5"]]


def rate(gs):
    n = len(gs)
    return f"{sum(g['fooled'] for g in gs):3}/{n:<4} {sum(g['fooled'] for g in gs) / n:4.0%}" if n else "—"


out = {}
for domain in ("math3", "trivia3"):
    qs = {q["id"]: q for q in load_questions(domain)}
    itype = "error_type" if domain == "math3" else "answer_type"
    base = read_jsonl(run_dir(domain, "baseline", "mix5") / "baseline.jsonl")
    solved = {(b["qid"], b["grader"]): b["solved"] for b in base}
    grades, commits = [], {}
    for att in MODELS:
        d = run_dir(domain, att, "mix5")
        answers = read_jsonl(d / "answers.jsonl")
        commits[att] = (sum(a["committed"] for a in answers), len(answers))
        for fr, fname in FRAMES.items():
            for g in read_jsonl(d / fname):
                if g["parse_error"]:
                    continue
                g["frame"], g["knew"], g["itype"] = fr, solved.get((g["qid"], g["grader"])), qs[g["qid"]][itype]
                grades.append(g)

    print(f"\n==================== {domain} ====================")
    print("baseline solved (no pitch):", ", ".join(
        f"{m} {sum(b['solved'] for b in base if b['grader'] == m)}/{sum(b['grader'] == m for b in base)}" for m in MODELS))
    print("attacker committed:", ", ".join(f"{a} {c}/{n}" for a, (c, n) in commits.items()))
    for key in ("grader", "attacker", "relation", "itype"):
        print(f"\n{('by ' + key):22}" + "".join(f"{fr:>16}" for fr in FRAMES) + f"{'neutral, knew':>18}")
        for v in sorted({g[key] for g in grades}):
            row = [rate([g for g in grades if g[key] == v and g["frame"] == fr]) for fr in FRAMES]
            knew = rate([g for g in grades if g[key] == v and g["frame"] == "neutral" and g["knew"]])
            print(f"  {v:20}" + "".join(f"{c:>16}" for c in row) + f"{knew:>18}")
    print("\nneutral frame, attacker (rows) x grader (cols), fooled/n:")
    print(f"  {'':16}" + "".join(f"{m:>15}" for m in MODELS))
    for a in MODELS:
        cells = []
        for m in MODELS:
            gs = [g for g in grades if g["attacker"] == a and g["grader"] == m and g["frame"] == "neutral"]
            cells.append(f"{sum(g['fooled'] for g in gs)}/{len(gs)}")
        print(f"  {a:16}" + "".join(f"{c:>15}" for c in cells))
    out[domain] = {
        "commits": commits,
        "baseline": {m: [sum(b["solved"] for b in base if b["grader"] == m), sum(b["grader"] == m for b in base)] for m in MODELS},
        "cells": [{"attacker": a, "grader": m, "frame": fr, "itype": t,
                   "fooled": sum(g["fooled"] for g in gs), "n": len(gs),
                   "fooled_when_knew": sum(g["fooled"] for g in gs if g["knew"]), "n_knew": sum(bool(g["knew"]) for g in gs)}
                  for a in MODELS for m in MODELS for fr in FRAMES for t in sorted({q[itype] for q in qs.values()})
                  for gs in [[g for g in grades if (g["attacker"], g["grader"], g["frame"], g["itype"]) == (a, m, fr, t)]]],
    }
(RUNS / "summary_mix5.json").write_text(json.dumps(out, indent=2))
