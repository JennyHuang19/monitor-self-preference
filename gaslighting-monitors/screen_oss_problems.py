"""Blind-spot experiment: screen candidate problems by how often each grader solves them with no pitch.

    python screen_oss_problems.py --ids r2-c0,r2-c4,r2-c5 [--graders gpt-oss-120b,kimi-k3,glm-5.3] [--samples 5]
    python screen_oss_problems.py --source math3 --ids p022,p036 --graders gpt-oss-120b

Each grader solves each problem with grade.py's FR_BASELINE prompt. A wrong answer equal to answer - value_k is
attributed to dropping case k (case values are distinct). Writes runs/blindspot_oss120b/screen.jsonl and prints, per
problem and grader, solved / tries and which cases were dropped. The target band for gpt-oss-120b is roughly 40-80%.
"""
import argparse
import json
from collections import defaultdict

from common import POOLS, RUNS, chat, read_jsonl, run_jobs
from grade import FR_BASELINE, _parse, _num

OUT = RUNS / "blindspot_oss120b"


def solve(job):
    g, p = job["grader"], job["problem"]
    r = chat(g["id"], [{"role": "user", "content": FR_BASELINE.format(question=p["question"])}], reasoning_budget=16000)
    parsed, err = _parse(r)
    ans = _num(parsed.get("answer"))
    drop = {str(p["answer"] - c["value"]): c["value"] for c in p["cases"] if isinstance(c, dict)}  # math3 cases are strings
    return {"qid": p["id"], "grader": g["name"], "sample": job["sample"], "answer": ans, "correct": str(p["answer"]),
            "solved": ans == str(p["answer"]), "dropped_case": drop.get(ans), "parse_error": err, **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--graders", default="gpt-oss-120b,kimi-k3,glm-5.3")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--source", choices=["candidates", "math3"], default="candidates")
    a = ap.parse_args()
    if a.source == "math3":  # Fable-written math3 problems (no per-case values, no writer's pick)
        cands = {q["id"]: q for q in json.loads((OUT.parent.parent / "questions_math3.json").read_text())}
    else:
        cands = {r["key"]: {"id": r["key"], **r["problem"]} for r in read_jsonl(OUT / "candidates.jsonl") if not r["parse_error"]}
    models = {m["name"]: m for m in POOLS["mix5"]}
    ids, graders = a.ids.split(","), a.graders.split(",")
    jobs = [{"key": f"{q}|{g}|s{s}", "grader": models[g], "problem": cands[q], "sample": s}
            for q in ids for g in graders for s in range(a.samples)]
    run_jobs(jobs, solve, OUT / "screen.jsonl")

    rows = defaultdict(list)
    for r in read_jsonl(OUT / "screen.jsonl"):
        rows[(r["qid"], r["grader"])].append(r)
    for q in ids:
        p = cands[q]
        pick = f"  writer's pick n={p['cases'][p['omitted_case']]['value']}" if "omitted_case" in p else ""
        print(f"\n{q}  answer {p['answer']}{pick}")
        for g in graders:
            rs = rows[(q, g)]
            wrong = [f"{r['answer']}" + (f"(drop {r['dropped_case']})" if r["dropped_case"] is not None else "")
                     for r in rs if not r["solved"]]
            print(f"   {g:<14} solved {sum(r['solved'] for r in rs)}/{len(rs)}   wrong: {', '.join(wrong) or '-'}")


if __name__ == "__main__":
    main()
