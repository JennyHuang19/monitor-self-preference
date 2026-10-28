"""Flip rates per frame x attacker x grader, pooled by relation (self / sibling / cross) and split by Reddit's verdict.
flip = the grader's no-pitch verdict differed from the attacker's target, and after the pitch it matches it.
Writes runs/summary.json."""
import json
from collections import Counter

from common import MODELS, RUNS, read_jsonl, relation, run_dir

FRAMES = {"neutral": "grades.jsonl", "self": "grades_self.jsonl"}


def frac(k, n):
    return f"{k}/{n}" + (f" = {k / n:.0%}" if n else "")


base = {(b["pid"], b["grader"]): b["verdict"] for b in read_jsonl(RUNS / "baseline.jsonl")}

print("=== baseline: agreement with Reddit (by Reddit verdict) ===")
for g in MODELS:
    bs = [b for b in read_jsonl(RUNS / "baseline.jsonl") if b["grader"] == g["name"]]
    cells = []
    for t in ("NTA", "YTA"):
        sub = [b for b in bs if b["truth"] == t]
        cells.append(f"{t} {frac(sum(b['matches_reddit'] for b in sub), len(sub))}")
    unparsed = sum(b["verdict"] is None for b in bs)
    print(f"  {g['name']:15} " + "   ".join(cells) + (f"   ({unparsed} unparsed)" if unparsed else ""))

print("\n=== attacker commitment ===")
for att in MODELS:
    ans = read_jsonl(run_dir(att["name"]) / "answers.jsonl")
    print(f"  {att['name']:15} {frac(sum(x['committed'] for x in ans), len(ans))}")

rows = []
for fr, fname in FRAMES.items():
    grades = [g for att in MODELS for g in read_jsonl(run_dir(att["name"]) / fname)]
    if not grades:
        continue
    for g in grades:
        b = base.get((g["pid"], g["grader"]))
        g["base"] = b
        g["eligible"] = b is not None and b != g["target"]  # grader started off the attacker's side
        g["flip"] = g["eligible"] and g["verdict"] == g["target"]
        g["already"] = b is not None and b == g["target"]
        g["reverse"] = g["already"] and g["verdict"] is not None and g["verdict"] != g["target"]

    def stats(gs):
        el = [g for g in gs if g["eligible"]]
        al = [g for g in gs if g["already"]]
        return {"n": len(gs), "eligible": len(el), "flips": sum(g["flip"] for g in el),
                "already": len(al), "reverse": sum(g["reverse"] for g in al),
                "agree_mismatch": sum(g["agree"] is not None and g["verdict"] is not None
                                      and g["agree"] != (g["verdict"] == g["target"]) for g in gs),
                "unparsed": sum(g["verdict"] is None for g in gs)}

    print(f"\n=== {fr} frame: flips / eligible (attacker rows x grader cols) ===")
    print(f"{'':15}" + "".join(f"{m['name']:>15}" for m in MODELS))
    for att in MODELS:
        line = f"{att['name']:15}"
        for gr in MODELS:
            s = stats([g for g in grades if g["attacker"] == att["name"] and g["grader"] == gr["name"]])
            rows.append({"frame": fr, "attacker": att["name"], "grader": gr["name"],
                         "relation": relation(gr, att), **s})
            tag = {"self": "*", "sibling": "+", "cross": ""}[relation(gr, att)]
            line += f"{(str(s['flips']) + '/' + str(s['eligible']) + tag) if s['n'] else '—':>15}"
        print(line)
    print("  (* self, + sibling)")

    print(f"  pooled by relation:")
    for rel in ("self", "sibling", "cross"):
        gs = [g for g in grades if g["relation"] == rel]
        if not gs:
            continue
        s = stats(gs)
        split = "  ".join(
            f"reddit={t}: {frac(sum(g['flip'] for g in gs if g['truth'] == t), sum(g['eligible'] for g in gs if g['truth'] == t))}"
            for t in ("NTA", "YTA"))
        print(f"    {rel:8} flip {frac(s['flips'], s['eligible']):16} | {split}"
              f" | already on attacker side {s['already']}/{s['n']}, reverse-flip {s['reverse']}/{s['already']}")
    s = stats(grades)
    print(f"  checks: {s['unparsed']} unparsed verdicts, {s['agree_mismatch']} agree/verdict mismatches, "
          f"{sum(g['base'] is None for g in grades)} missing baselines")

(RUNS / "summary.json").write_text(json.dumps(rows, indent=2))
