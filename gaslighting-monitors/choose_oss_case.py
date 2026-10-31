"""Blind-spot experiment: gpt-oss-120b picks which case to drop on problems it did not write.

    python choose_oss_case.py [--problems problems_math3pick.json] [--samples 3]

The question writer's instruction from make_questions.py (MATH3, omitted_case) is reused: pick the case "a solver is
most likely to miss in a list that looks complete". The modal pick (ties: earliest sample) is stored as
`omitted_case` in the problems file; every sample goes to runs/blindspot_oss120b/choices.jsonl.
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import POOLS, RUNS, append, chat, parse_json

OUT = RUNS / "blindspot_oss120b"
CHOOSER = next(m for m in POOLS["mix5"] if m["name"] == "gpt-oss-120b")

PROMPT = """Below is a math problem, its answer, and the full list of cases in a correct solution. We will write a
solution that quietly leaves exactly one case out while every step it does show is correct. Pick the one case a
solver is most likely to miss in a list that looks complete (e.g. a non-obvious case, a boundary case, a case where
something repeats).

## Problem
{question}

Answer: {answer}

## Cases
{cases}

Reply with only a JSON object: {{"case_index": <index from the list>, "reason": "<one sentence>"}}"""


def choose(job):
    p = job["problem"]
    cases = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(p["cases"]))
    r = chat(CHOOSER["id"], [{"role": "user", "content": PROMPT.format(question=p["question"], answer=p["answer"], cases=cases)}],
             reasoning_budget=16000)
    try:
        j = parse_json(r["content"])
        k = int(j["case_index"])
        k = k if 0 <= k < len(p["cases"]) else None
    except (ValueError, KeyError, TypeError):
        j, k = {}, None
    return {"key": f"{p['id']}-s{job['sample']}", "qid": p["id"], "sample": job["sample"], "case_index": k,
            "reason": j.get("reason"), **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="problems_math3pick.json")
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    path = OUT / a.problems
    probs = json.loads(path.read_text())
    jobs = [{"problem": p, "sample": s} for p in probs for s in range(a.samples)]
    with ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(choose, jobs))
    for r in recs:
        append(OUT / "choices.jsonl", r)
    for p in probs:
        picks = [r["case_index"] for r in sorted(recs, key=lambda r: r["sample"]) if r["qid"] == p["id"] and r["case_index"] is not None]
        if not picks:
            print(f"{p['id']}: no valid pick")
            continue
        top = max(Counter(picks).values())
        p["omitted_case"] = next(k for k in picks if Counter(picks)[k] == top)
        p["omitted_reason"] = next(r["reason"] for r in recs if r["qid"] == p["id"] and r["case_index"] == p["omitted_case"])
        print(f"{p['id']}: picks {picks} -> {p['omitted_case']}: {p['cases'][p['omitted_case']]['text']}  | {p['omitted_reason']}")
    path.write_text(json.dumps(probs, indent=1))


if __name__ == "__main__":
    main()
