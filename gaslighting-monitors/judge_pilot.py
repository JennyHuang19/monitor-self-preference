"""Judge-agreement pilot for diagnose_fooled.py: how much does the 'why was the grader fooled' label depend on the judge?

    python judge_pilot.py select     # writes runs/judge_pilot_keys.json (stratified sample of already-labelled fooled grades)
    python diagnose_fooled.py --pool-prefix <p> --questions <q> --keys runs/judge_pilot_keys.json --judge opus --rep 1   (and --rep 2, --judge gpt, --judge gemini)
    python judge_pilot.py analyze    # agreement tables -> stdout and runs/judge_pilot_report.txt

The sample is stratified by (grader, relation), equal allocation capped by what exists, so small cells are not swamped by
gpt-oss-20b x cross. Opus rep 0 is the existing label; reps 1-2 re-ask the same judge; gpt and gemini are other families.
"""
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
POOLS = {"math3_mix5": 60, "math4_argon_glm": 40}  # pilot size per pool
KEYS = RUNS / "judge_pilot_keys.json"
SOURCES = [("opus", ""), ("opus r1", "__opus_r1"), ("opus r2", "__opus_r2"), ("gpt", "__gpt"), ("gemini", "__gemini")]
CATS = ["rubber_stamp", "missed_in_check", "shared_error", "persuaded", "knew_but_accepted", "ambiguous_key"]
# Coarse view: did the grader catch nothing, reach the error itself, or see the problem and let it go?
COARSE = {"rubber_stamp": "didnt_see", "missed_in_check": "didnt_see", "shared_error": "shared_error",
          "persuaded": "saw_it", "knew_but_accepted": "saw_it", "ambiguous_key": "ambiguous_key"}


def select():
    rng, out = random.Random(0), {}
    for prefix, size in POOLS.items():
        rows = [json.loads(l) for l in open(RUNS / f"diagnose_{prefix}.jsonl")]
        cells = collections.defaultdict(list)
        for r in rows:
            if r["category"] in CATS:
                cells[(r["grader"], r["relation"])].append(r["key"])
        for v in cells.values():
            rng.shuffle(v)
        pick, left = [], size
        order = sorted(cells, key=lambda c: len(cells[c]))  # fill small cells first, spread the rest evenly
        for i, c in enumerate(order):
            take = min(len(cells[c]), left // (len(order) - i))
            pick += cells[c][:take]; left -= take
        out[prefix] = sorted(pick)
        print(prefix, len(pick), "keys from", len(cells), "cells")
    json.dump(out, open(KEYS, "w"), indent=1)


def kappa(pairs):
    """Cohen's kappa for a list of (a, b) labels."""
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss(items):
    """Fleiss' kappa; items = list of label lists, same rater count each."""
    m = len(items[0]); n = len(items)
    tot = collections.Counter(x for it in items for x in it)
    pbar = sum((sum(c * c for c in collections.Counter(it).values()) - m) / (m * (m - 1)) for it in items) / n
    pe = sum((v / (n * m)) ** 2 for v in tot.values())
    return (pbar - pe) / (1 - pe)


def boot(stat, rows, it=2000):
    """95% interval for stat(rows), resampling whole problems (rows of one problem move together)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cluster"]].append(r)
    cl, rng, vals = list(by), random.Random(1), []
    for _ in range(it):
        rs = [r for c in (rng.choice(cl) for _ in cl) for r in by[c]]
        v = stat(rs)
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]


def analyze():
    keys = json.load(open(KEYS))
    rows = []  # one per pilot item: {cluster, meta, labels: {source: row}}
    for prefix, ks in keys.items():
        lab = {}
        for name, suf in SOURCES:
            f = RUNS / f"diagnose_{prefix}{suf}.jsonl"
            lab[name] = {r["key"]: r for r in map(json.loads, open(f))} if f.exists() else {}
        for k in ks:
            got = {n: lab[n][k] for n, _ in SOURCES if k in lab[n]}
            base = got["opus"]
            rows.append({"cluster": f"{prefix}|{base['qid']}", "pool": prefix, "key": k, "grader": base["grader"],
                         "attacker": base["attacker"], "relation": base["relation"], "labels": got})
    L = []
    say = lambda s="": (print(s), L.append(s))
    have = {n: sum(n in r["labels"] and r["labels"][n]["category"] in CATS for r in rows) for n, _ in SOURCES}
    say(f"Judge-agreement pilot: {len(rows)} fooled grades ({', '.join(f'{p} {len(k)}' for p, k in keys.items())}), "
        f"{len({r['cluster'] for r in rows})} problems. Valid labels per source: {have}")
    say("Intervals are 95%, bootstrapped over problems.\n")

    def pair_stats(a, b, f=lambda c: c, sub=rows):
        ok = [r for r in sub if all(x in r["labels"] and r["labels"][x]["category"] in CATS for x in (a, b))]
        pr = lambda rs: [(f(r["labels"][a]["category"]), f(r["labels"][b]["category"])) for r in rs]
        if not ok:
            return "      n=0"
        agree = sum(x == y for x, y in pr(ok)) / len(ok)
        k = kappa(pr(ok)); lo, hi = boot(lambda rs: kappa(pr(rs)), ok)
        return f"n={len(ok):3}  agree {agree:4.0%}  kappa {k:5.2f} [{lo:5.2f}, {hi:5.2f}]"

    for title, f in (("Six categories", lambda c: c), ("Coarse (rubber_stamp+missed_in_check, persuaded+knew_but_accepted merged)", COARSE.get)):
        say(f"== {title}")
        for a, b in (("opus", "opus r1"), ("opus", "opus r2"), ("opus r1", "opus r2"),
                     ("opus", "gpt"), ("opus", "gemini"), ("gpt", "gemini")):
            say(f"  {a:8} vs {b:8} {pair_stats(a, b, f)}")
        trio = [r for r in rows if all(x in r["labels"] and r["labels"][x]["category"] in CATS for x in ("opus", "gpt", "gemini"))]
        if trio:
            fl = lambda rs: fleiss([[f(r["labels"][x]["category"]) for x in ("opus", "gpt", "gemini")] for r in rs])
            lo, hi = boot(fl, trio)
            unan = sum(len({f(r["labels"][x]["category"]) for x in ("opus", "gpt", "gemini")}) == 1 for r in trio) / len(trio)
            say(f"  Fleiss kappa, opus/gpt/gemini: {fl(trio):.2f} [{lo:.2f}, {hi:.2f}]; all three agree on {unan:.0%}")
        say()

    for other in ("opus r1", "gpt", "gemini"):
        ok = [r for r in rows if all(x in r["labels"] and r["labels"][x]["category"] in CATS for x in ("opus", other))]
        if not ok:
            continue
        say(f"== Confusion: opus (rows) vs {other} (columns), n={len(ok)}")
        c = collections.Counter((r["labels"]["opus"]["category"], r["labels"][other]["category"]) for r in ok)
        used = [k for k in CATS if any(c[(k, y)] or c[(y, k)] for y in CATS)]
        w = 18
        say(" " * w + "".join(f"{k[:12]:>13}" for k in used))
        for a in used:
            say(f"{a:<{w}}" + "".join(f"{c[(a, b)] or '.':>13}" for b in used))
        say()

    say("== Marginals (share of pilot items per category)")
    say(" " * 18 + "".join(f"{n:>10}" for n, _ in SOURCES))
    for k in CATS:
        vals = [sum(r["labels"].get(n, {}).get("category") == k for r in rows) / max(1, have[n]) for n, _ in SOURCES]
        say(f"{k:<18}" + "".join(f"{v:10.0%}" for v in vals))
    say()

    say("== Yes/no fields, agreement with opus rep 0")
    for fld in ("noticed_flaw", "computed_true_answer", "collusion_signal"):
        parts = []
        for n, _ in SOURCES[1:]:
            ok = [r for r in rows if n in r["labels"] and fld in r["labels"][n] and fld in r["labels"]["opus"]]
            if ok:
                pr = [(bool(r["labels"]["opus"][fld]), bool(r["labels"][n][fld])) for r in ok]
                parts.append(f"{n} {sum(a == b for a, b in pr) / len(ok):.0%} (k={kappa(pr):.2f})")
        say(f"  {fld:22} " + "; ".join(parts))
    say()

    # Same-family check: the GPT judge vs gpt-oss graders/attackers, the Gemini judge vs gemini-4-argon pitches.
    say("== Same-family check (six categories, vs opus)")
    for judge, fam, test in (("gpt", "gpt-oss involved", lambda r: "gpt-oss" in r["grader"] or "gpt-oss" in r["attacker"]),
                             ("gemini", "gemini-4-argon pitch", lambda r: "gemini" in r["attacker"])):
        for lab, sub in ((fam, [r for r in rows if test(r)]), ("other rows", [r for r in rows if not test(r)])):
            say(f"  {judge:7} {lab:22} {pair_stats('opus', judge, sub=sub)}")
    say()

    (RUNS / "judge_pilot_report.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    {"select": select, "analyze": analyze}[sys.argv[1]]()
