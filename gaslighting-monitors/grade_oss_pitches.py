"""Blind-spot experiment, step 3: each grader judges every clean pitch several times with the neutral prompt.

    python grade_oss_pitches.py [--pitches pitches_v3_opus.jsonl] [--graders gpt-oss-120b,kimi-k3,glm-5.3] [--samples 5]
        [--frame neutral|skim1k]

Uses grade.py's FR_GRADE (or FR_SKIM with the 1k thinking cap) prompt and grade_fields() parser with the same chat
settings as grade.grade(). Writes runs/blindspot_oss120b/grades_<pitches stem>[_skim1k].jsonl and prints fooled counts per grader x dropped case.
"""
import argparse
import json
from collections import defaultdict

from common import POOLS, RUNS, chat, read_jsonl, run_jobs
from grade import FR_GRADE, FR_SKIM, MAX_TOKENS, REASONING_BUDGET, grade_fields

OUT = RUNS / "blindspot_oss120b"


def grade(job):
    g, pitch, p = job["grader"], job["pitch"], job["problem"]
    tmpl = FR_SKIM if job["frame"] == "skim1k" else FR_GRADE
    r = chat(g["id"], [{"role": "user", "content": tmpl.format(question=p["question"], solution=pitch["content"])}],
             reasoning_budget=REASONING_BUDGET.get(job["frame"], 16000), max_tokens=MAX_TOKENS.get(job["frame"], 32000))
    return {"grader": g["name"], "pitch_key": pitch["key"], "writer": pitch.get("writer"), "dropped_value": pitch["dropped_value"],
            "chosen": pitch["chosen"], "claimed": pitch["target_wrong"], "correct": str(p["answer"]), "sample": job["sample"],
            "frame": job["frame"], **r, **grade_fields(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitches", default="pitches_v3_opus.jsonl")
    ap.add_argument("--problems", default="pilot_problem.json")
    ap.add_argument("--graders", default="gpt-oss-120b,kimi-k3,glm-5.3")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--frame", choices=["neutral", "skim1k"], default="neutral")
    a = ap.parse_args()
    data = json.loads((OUT / a.problems).read_text())
    probs = {p["id"]: p for p in ([data] if isinstance(data, dict) else data)}
    only = next(iter(probs)) if len(probs) == 1 else None  # pilot pitches carry no qid
    pitches = [r for r in read_jsonl(OUT / a.pitches) if r["route_ok"]]
    models = {m["name"]: m for m in POOLS["mix5"]}
    graders = [models[n] for n in a.graders.split(",")]
    out = OUT / f"grades_{a.pitches.removeprefix('pitches_').removesuffix('.jsonl')}{'' if a.frame == 'neutral' else '_' + a.frame}.jsonl"
    jobs = [{"key": f"{g['name']}|{pt['key']}|s{s}", "grader": g, "pitch": pt, "problem": probs[pt.get("qid", only)],
             "sample": s, "frame": a.frame} for g in graders for pt in pitches for s in range(a.samples)]
    run_jobs(jobs, grade, out)

    case_of = {pt["key"]: (pt.get("qid", only), pt["case_index"]) for pt in pitches}
    cells = defaultdict(lambda: [0, 0, 0])  # fooled, graded, unparsed
    for r in read_jsonl(out):
        c = cells[(r["grader"], *case_of[r["pitch_key"]])]
        if r["parse_error"]:
            c[2] += 1
        else:
            c[0] += r["fooled"]
            c[1] += 1
    print("\nfooled / graded (unparsed)   * = gpt-oss-120b's chosen case")
    for qid, p in probs.items():
        ks = sorted({k for q, k in case_of.values() if q == qid}, key=lambda k: (k != p["omitted_case"], k))
        heads = [f"{'*' if k == p['omitted_case'] else ''}c{k}(-{p['cases'][k]['value']})" for k in ks]
        print(f"\n{qid}  answer {p['answer']}")
        print(f"{'grader':<14}" + "".join(f"{h:>13}" for h in heads))
        for g in graders:
            row = ""
            for k in ks:
                f, n, u = cells[(g["name"], qid, k)]
                row += f"{f'{f}/{n}' + (f' ({u})' if u else ''):>13}"
            print(f"{g['name']:<14}{row}")


if __name__ == "__main__":
    main()
