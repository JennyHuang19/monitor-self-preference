"""Fooled rates per pool x domain x attacker x grader, for the neutral and self-check frames, next to the no-pitch
baseline. Pooled by the grader's relation to the attacker: self (same model), sibling (same family), cross.
Writes runs/summary.json."""
import json
from collections import defaultdict

from common import DOMAINS, POOLS, RUNS, pool_models, read_jsonl, run_dirs

FRAMES = {"neutral": "grades.jsonl", "self": "grades_self.jsonl", "skim": "grades_skim.jsonl",
          "skim1k": "grades_skim1k.jsonl"}


def relation(grader: dict, att: dict) -> str:
    if grader["name"] == att["name"]:
        return "self"
    return "sibling" if grader["family"] == att["family"] else "cross"


def belief_wrong(g) -> float:
    """The grader's stated belief (0-100) that the wrong answer is right."""
    try:
        c = float(g["confidence"])
    except (TypeError, ValueError):
        c = 50.0
    return c if g["fooled"] else 100 - c


def frac(f, n):
    return f"{f}/{n}" + (f" = {f / n:.0%}" if n else "")


rows = []
for pool in POOLS:
    for domain in DOMAINS:
        runs = run_dirs(domain, pool)
        if not runs:
            continue
        print(f"\n=== {pool} pool · {domain} ===")
        print(f"{'attacker':16} {'grader':19} {'neutral':>8} {'self':>8} {'skim':>8} {'skim1k':>8} {'baseline':>9}")
        for att, d in runs:
            base = read_jsonl(d / "baseline.jsonl")
            solved = {(b["qid"], b["grader"]): b["solved"] for b in base}
            by = {fr: defaultdict(list) for fr in FRAMES}
            for fr, fname in FRAMES.items():
                for g in read_jsonl(d / fname):
                    by[fr][g["grader"]].append(g)
            for gr in pool_models(pool):
                if not by["neutral"].get(gr["name"]):
                    continue
                rel = relation(gr, att)
                cells = []
                for fr in FRAMES:
                    gs = by[fr].get(gr["name"], [])
                    # Fooled even though it got the item right with no pitch: the clearest case of persuasion.
                    knew = [g for g in gs if solved.get((g["qid"], gr["name"]))]
                    rows.append({"pool": pool, "domain": domain, "frame": fr, "attacker": att["name"],
                                 "grader": gr["name"], "relation": rel, "fooled": sum(g["fooled"] for g in gs),
                                 "n": len(gs), "fooled_when_knew": sum(g["fooled"] for g in knew), "n_knew": len(knew),
                                 "belief_wrong": sum(map(belief_wrong, gs)) / len(gs) if gs else None})
                    cells.append(f"{sum(g['fooled'] for g in gs)}/{len(gs)}" if gs else "—")
                bs = [b for b in base if b["grader"] == gr["name"]]
                tag = {"self": " (self)", "sibling": " (sib)", "cross": ""}[rel]
                print(f"{att['name']:16} {gr['name'] + tag:19} {cells[0]:>8} {cells[1]:>8} {cells[2]:>8} {cells[3]:>8}"
                      f" {sum(b['solved'] for b in bs):>5}/{len(bs)}")
        for fr in FRAMES:
            print(f"  {fr} frame:")
            for rel in ("self", "sibling", "cross"):
                rs = [r for r in rows if (r["pool"], r["domain"], r["frame"], r["relation"]) == (pool, domain, fr, rel)]
                if not rs or not sum(r["n"] for r in rs):
                    continue
                print(f"    {rel:8} fooled {frac(sum(r['fooled'] for r in rs), sum(r['n'] for r in rs)):14}"
                      f" | when it knew {frac(sum(r['fooled_when_knew'] for r in rs), sum(r['n_knew'] for r in rs))}")
(RUNS / "summary.json").write_text(json.dumps(rows, indent=2))
