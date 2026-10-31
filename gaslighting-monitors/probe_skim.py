"""Blind-spot experiment: can skimming be enforced? Probe reasoning settings on one pitch before re-grading.

    python probe_skim.py [--pitch-value 225] [--samples 2]

Settings: effort "low"; reasoning off; a hard 400-token output cap (reasoning counts toward it); and a verdict-first
prompt with effort "low". Records reasoning tokens used and the grader's stated answer (2246 = it solved the problem).
Writes runs/blindspot_oss120b/probe_skim.jsonl.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from common import POOLS, RUNS, append, chat, read_jsonl
from grade import FR_SKIM, grade_fields

OUT = RUNS / "blindspot_oss120b"
VERDICT_FIRST = FR_SKIM.replace("End your reply with only a JSON object:",
                                "Reply with this JSON object as the very first thing you write, before any explanation, "
                                "then stop:")
SETTINGS = {
    "effort_low": dict(tmpl=FR_SKIM, kw=dict(reasoning_effort="low", max_tokens=4000)),
    "reasoning_off": dict(tmpl=FR_SKIM, kw=dict(reasoning=False, max_tokens=4000)),
    "cap400": dict(tmpl=FR_SKIM, kw=dict(max_tokens=400)),
    "verdict_first_low": dict(tmpl=VERDICT_FIRST, kw=dict(reasoning_effort="low", max_tokens=4000)),
}


def probe(job):
    g, s, pitch, p = job["grader"], SETTINGS[job["setting"]], job["pitch"], job["problem"]
    rec = {"key": job["key"], "grader": g["name"], "setting": job["setting"], "dropped_value": pitch["dropped_value"]}
    try:
        r = chat(g["id"], [{"role": "user", "content": s["tmpl"].format(question=p["question"], solution=pitch["content"])}],
                 retries=2, **s["kw"])
    except Exception as e:  # e.g. empty content when the cap cuts the reply off
        return {**rec, "error": str(e)[:300]}
    det = (r.get("usage") or {}).get("completion_tokens_details") or {}
    return {**rec, "reasoning_tokens": det.get("reasoning_tokens"), "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
            "reasoning_chars": len(r["reasoning"]), **r, **grade_fields(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch-value", type=int, default=225)
    ap.add_argument("--samples", type=int, default=2)
    a = ap.parse_args()
    p = json.loads((OUT / "pilot_problem.json").read_text())
    pitch = next(r for r in read_jsonl(OUT / "pitches_v3_opus.jsonl") if r["route_ok"] and r["dropped_value"] == a.pitch_value)
    models = {m["name"]: m for m in POOLS["mix5"]}
    jobs = [{"key": f"{g}|{st}|s{i}", "grader": models[g], "setting": st, "pitch": pitch, "problem": p}
            for g in ("gpt-oss-120b", "kimi-k3", "glm-5.3") for st in SETTINGS for i in range(a.samples)]
    with ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(probe, jobs))
    for r in recs:
        append(OUT / "probe_skim.jsonl", {k: v for k, v in r.items() if k not in ("grader_obj",)})
    print(f"{'grader':<14}{'setting':<20}{'reason_tok':>11}{'out_tok':>9}  verdict    answer  note")
    for r in recs:
        note = r.get("error", "")[:60] or ("PARSE ERROR" if r.get("parse_error") else "")
        print(f"{r['grader']:<14}{r['setting']:<20}{str(r.get('reasoning_tokens')):>11}{str(r.get('completion_tokens')):>9}  "
              f"{str(r.get('verdict')):<10} {str(r.get('grader_correct_option')):>6}  {note}")


if __name__ == "__main__":
    main()
