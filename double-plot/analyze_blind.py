"""Who do blind readers side with: the better debater, or the one listed first?

Usage:
    .venv/bin/python analyze_blind.py results/debate-<ts>.jsonl results/reverdict-<A>.jsonl results/reverdict-<B>.jsonl
"""
import json
import sys
from collections import Counter

debate_file, *blind_files = sys.argv[1:]
rows = [json.loads(l) for l in open(debate_file)]
debates = {r["debate_id"]: r for r in rows if r["type"] == "debate"}
told = [r for r in rows if r["type"] == "verdict"]
blind = [json.loads(l) for f in blind_files for l in open(f)]


def winner(r):
    """Model whose assigned side the verdict named, or None if undecided/error."""
    if "error" in r or r.get("culprit") not in ("Daniel", "Maya"):
        return None
    return next(m for m, side in debates[r["debate_id"]]["assignment"].items() if side == r["culprit"])


def first_listed(r):
    # Seat A = debate.py's first --models entry; "blind-firstB" lists seat B first.
    seat_a = next(t["model"] for t in debates[r["debate_id"]]["turns"] if t["seat"] == "A")
    seat_b = next(t["model"] for t in debates[r["debate_id"]]["turns"] if t["seat"] == "B")
    return seat_b if r.get("condition") == "blind-firstB" else seat_a


models = sorted({r["model"] for r in blind})
print("Blind readers: whose argument they sided with")
print(f"{'reader':<18} {'order':<14} " + " ".join(f"{m:>16}" for m in models) + f" {'undecided':>10} {'sided w/ first':>15}")
for reader in models:
    for cond in ("blind-firstA", "blind-firstB"):
        rs = [r for r in blind if r["model"] == reader and r.get("condition") == cond]
        if not rs:
            continue
        w = Counter(winner(r) for r in rs)
        decided = [r for r in rs if winner(r)]
        first = sum(winner(r) == first_listed(r) for r in decided)
        lead = first_listed(rs[0])
        print(f"{reader:<18} {lead + ' first':<14} " + " ".join(f"{w[m]:>16}" for m in models)
              + f" {w[None]:>10} {f'{first}/{len(decided)}':>15}")

print("\nSide switching (verdict names the opponent's suspect)")
def outcome(r):
    if winner(r) is None:
        return "undecided"
    return "kept" if winner(r) == r["model"] else "switched"
for reader in models:
    for label, rs in (("told", [r for r in told if r["model"] == reader]),
                      ("blind, A first", [r for r in blind if r["model"] == reader and r.get("condition") == "blind-firstA"]),
                      ("blind, B first", [r for r in blind if r["model"] == reader and r.get("condition") == "blind-firstB"])):
        c = Counter(outcome(r) for r in rs)
        print(f"  {reader:<18} {label:<15} switched {c['switched']:>2}  kept {c['kept']:>2}  undecided {c['undecided']:>2}  (n={len(rs)})")
