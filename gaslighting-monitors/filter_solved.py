"""math4 item set for the argon_glm pool: only problems every pool model solves with no pitch.

    python filter_solved.py candidates [--carry N]   -> questions_math4_candidates.json
    python filter_solved.py filter [--samples 2] [--keep 100]  -> questions_math4.json

candidates: the math3 items every pool model solved in the existing single-sample baselines (glm-5.3 / glm-5.3-flash
from mix5, gemini-4-argon from closed2), plus every new math4 batch (runs/questions_math4_raw_<b>.json) whose item passes
check_items.check_math and isn't a duplicate. Ids are stable across reruns: m<NNN> for math3 p<NNN>, n<BB><KK> for item
KK of new batch BB, so baselines keyed on them stay valid as batches are added.

filter: keeps the candidates that every pool model solved in all of the first --samples no-pitch samples
(runs/math4_argon_glm__baseline/baseline.jsonl, from grade.py --baseline-only --baseline-samples N), in candidate order.
"""
import argparse
import json
from collections import defaultdict

from check_items import check_math
from common import HERE, POOLS, RUNS, norm_answer, parse_json, read_jsonl

POOL = "argon_glm"
CANDS = HERE / "questions_math4_candidates.json"
MODELS = [m["name"] for m in POOLS[POOL]]
# Where each pool model's earlier math3 baseline lives.
MATH3_BASE = {"glm-5.3": "math3_mix5__baseline", "glm-5.3-flash": "math3_mix5__baseline",
              "gemini-4-argon": "math3_closed2__baseline"}


def carried(n):
    solved = defaultdict(dict)
    for m, d in MATH3_BASE.items():
        for b in read_jsonl(RUNS / d / "baseline.jsonl"):
            if b["grader"] == m:
                solved[b["qid"]][m] = b["solved"]
    out = []
    for q in json.loads((HERE / "questions_math3.json").read_text()):
        if all(solved[q["id"]].get(m) for m in MODELS):
            out.append({**q, "id": "m" + q["id"][1:], "orig_id": f"math3:{q['id']}", "source": "math3", "domain": "math4"})
    return out[:n]


def new_items(seen):
    out = []
    files = sorted(RUNS.glob("questions_math4_raw_*.json"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))
    for f in files:
        b = int(f.stem.rsplit("_", 1)[1])
        try:
            got = parse_json(json.loads(f.read_text())["content"], first=True)["questions"]
        except Exception as e:
            print(f"{f.name}: unparsable ({e})")
            continue
        for k, q in enumerate(got, 1):
            q = {**q, "id": f"n{b:02d}{k:02d}", "orig_id": f"math4:batch{b}:{k}", "batch": b, "source": "new", "domain": "math4"}
            try:
                checks, printed = check_math(q)
            except Exception as e:  # missing field etc.
                checks, printed = {"well-formed": False}, repr(e)
            key = norm_answer(q.get("question"))
            checks["not a duplicate"] = key not in seen
            seen.add(key)
            if all(checks.values()):
                out.append(q)
            else:
                print(f"DROP {q['id']}: {[c for c, v in checks.items() if not v]} printed={printed!r}")
    return out, len(files)


def candidates(carry):
    old = carried(carry)
    seen = {norm_answer(q["question"]) for q in json.loads((HERE / "questions_math3.json").read_text())}
    new, nb = new_items(seen)
    CANDS.write_text(json.dumps(old + new, indent=2))
    print(f"{len(old)} carried from math3 + {len(new)} new (from {nb} batches) -> {CANDS.name}")


def filt(samples, keep):
    cands = json.loads(CANDS.read_text())
    got = defaultdict(dict)  # (qid, model) -> {sample: solved}
    for b in read_jsonl(RUNS / f"math4_{POOL}__baseline" / "baseline.jsonl"):
        got[(b["qid"], b["grader"])][b.get("sample", 0)] = b["solved"] and not b["parse_error"]
    done = [q for q in cands if all(len(got[(q["id"], m)]) >= samples for m in MODELS)]
    ok = lambda q, m: all(got[(q["id"], m)].get(s) for s in range(samples))
    print(f"{len(done)}/{len(cands)} candidates have {samples} baseline samples from every model")
    for src in ("math3", "new"):
        sub = [q for q in done if q["source"] == src]
        if sub:
            print(f"  {src:5} n={len(sub):3}  " + "  ".join(f"{m} {sum(ok(q, m) for q in sub)}" for m in MODELS)
                  + f"  all {sum(all(ok(q, m) for m in MODELS) for q in sub)}")
    passed = [q for q in done if all(ok(q, m) for m in MODELS)]
    (HERE / "questions_math4.json").write_text(json.dumps(passed[:keep], indent=2))
    print(f"{len(passed)} pass; kept {min(len(passed), keep)} -> questions_math4.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["candidates", "filter"])
    ap.add_argument("--carry", type=int, default=None, help="candidates: at most N carried math3 items")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--keep", type=int, default=100)
    a = ap.parse_args()
    candidates(a.carry) if a.mode == "candidates" else filt(a.samples, a.keep)
