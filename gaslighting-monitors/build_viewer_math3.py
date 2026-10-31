"""Build the math3 (mix5 pool) viewer in the style of aita/build_viewer.py: viewer_math3/index.html (overview +
problem list) and viewer_math3/problems/<qid>.json (one problem's pitches, grades and CoTs, loaded on demand).
Fooled counts match analyze_mix5.py (unparsed grades dropped); "knew" = the grader solved it with no pitch.

    python build_viewer_math3.py
"""
import json
from collections import defaultdict

import generate
import grade
from common import HERE, POOLS, load_questions, read_jsonl, run_dir

DOMAIN, POOL = "math3", "mix5"
FRAMES = {"neutral": "grades.jsonl", "self": "grades_self.jsonl", "skim1k": "grades_skim1k.jsonl"}
SCOPES = ("knew", "all")
MODELS = POOLS[POOL]
names = [m["name"] for m in MODELS]
OUT = HERE / "viewer_math3"
(OUT / "problems").mkdir(parents=True, exist_ok=True)

qs = load_questions(DOMAIN)
baseline = {(b["qid"], b["grader"]): b for b in read_jsonl(run_dir(DOMAIN, "baseline", POOL) / "baseline.jsonl")}
answers = {(a["qid"], att): a for att in names for a in read_jsonl(run_dir(DOMAIN, att, POOL) / "answers.jsonl")}
grades = defaultdict(dict)  # frame -> (qid, attacker, grader) -> record
for fr, fname in FRAMES.items():
    for att in names:
        for g in read_jsonl(run_dir(DOMAIN, att, POOL) / fname):
            grades[fr][(g["qid"], att, g["grader"])] = g


def knew(qid, gr):
    return bool(baseline.get((qid, gr), {}).get("solved"))


# Overview: attacker x grader fooled matrices, pooled by relation, per-problem fooled counts, per frame and scope.
# "all" pools the three frames: each pitch counts once per frame.
VIEWS = [*FRAMES, "all"]
matrix = {v: {s: {a: {g: [0, 0] for g in names} for a in names} for s in SCOPES} for v in VIEWS}
pooled = {v: {s: {r: [0, 0] for r in ("self", "sibling", "cross")} for s in SCOPES} for v in VIEWS}
item_fools = {v: {s: defaultdict(lambda: [0, 0]) for s in SCOPES} for v in VIEWS}
pair_fools = {v: {s: defaultdict(list) for s in SCOPES} for v in VIEWS}  # "att|grader" -> qids where fooled
etypes = sorted({q["error_type"] for q in qs})
etype = {q["id"]: q["error_type"] for q in qs}
by_type = {v: {s: {g: {t: [0, 0] for t in etypes} for g in names} for s in SCOPES} for v in VIEWS}  # grader x error type
for fr in FRAMES:
    for (qid, att, gr), g in grades[fr].items():
        if g["parse_error"]:
            continue
        for s in SCOPES:
            if s == "knew" and not knew(qid, gr):
                continue
            for v in (fr, "all"):
                for cell in (matrix[v][s][att][gr], pooled[v][s][g["relation"]], item_fools[v][s][qid],
                             by_type[v][s][gr][etype[qid]]):
                    cell[0] += g["fooled"]
                    cell[1] += 1
                if g["fooled"] and qid not in pair_fools[v][s][f"{att}|{gr}"]:
                    pair_fools[v][s][f"{att}|{gr}"].append(qid)

solve = {gr: [sum(knew(q["id"], gr) for q in qs), sum((q["id"], gr) in baseline for q in qs)] for gr in names}
commit = {att: [sum(answers[(q["id"], att)]["committed"] for q in qs if (q["id"], att) in answers),
                sum((q["id"], att) in answers for q in qs)] for att in names}
regression = json.loads((HERE / "runs" / "regression_math3.json").read_text())

index = [{"qid": q["id"], "question": q["question"], "topic": q["topic"], "etype": q["error_type"],
          "answer": str(q["answer"]), "target": str(q["target_wrong"]),
          "fools": {v: {s: item_fools[v][s][q["id"]] for s in SCOPES} for v in VIEWS}} for q in qs]

# The prompts as sent, with placeholders left in (format() also turns the templates' {{ }} back into { }).
ph = {"question": "{question}", "solution": "{pitch}"}
prompts = [
    {"title": "Attacker", "note": "system prompt + the question as the user turn",
     "turns": [["system", generate.MATH2.format(target="{target}")], ["user", "{question}"]]},
    {"title": "Grader · neutral frame", "note": "one user turn",
     "turns": [["user", grade.FR_GRADE.format(**ph)]]},
    {"title": "Grader · self-check frame", "note": "the pitch is placed as the grader's own earlier answer",
     "turns": [["user", "{question}"], ["assistant", "{pitch}"], ["user", grade.FR_SELF.format()]]},
    {"title": "Grader · skim frame, 1k thinking cap", "note": "reasoning capped at 1,000 tokens, whole reply at 2,500",
     "turns": [["user", grade.FR_SKIM.format(**ph)]]},
    {"title": "Grader · no-pitch baseline", "note": "sets whether the grader \"knew\" the answer",
     "turns": [["user", grade.FR_BASELINE.format(question="{question}")]]},
]

# The CoT study's "same slip" finding and worked examples (cot_study.py export), when it has been run.
slip_path = HERE / "runs" / "cot_study" / "viewer_slip.json"
summary = {"models": MODELS, "matrix": matrix, "pooled": pooled, "pairFools": pair_fools, "solve": solve,
           "byType": by_type, "typeCounts": {t: sum(q["error_type"] == t for q in qs) for t in etypes},
           "commit": commit, "regression": regression, "prompts": prompts, "items": index,
           "slip": json.loads(slip_path.read_text()) if slip_path.exists() else None}

# One file per problem: the problem, the no-pitch answers, each attacker's pitch, and every grade in all frames.
keep_g = ("verdict", "confidence", "grader_correct_option", "fooled", "reasoning", "content", "parse_error", "provider")
keep_b = ("answer", "confidence", "solved", "reasoning", "content", "parse_error", "provider")
for q in qs:
    qid = q["id"]
    rec = {"qid": qid, **{k: q.get(k) for k in ("question", "topic", "error_type", "error_description", "cases",
                                                 "solution", "check_code")},
           "answer": str(q["answer"]), "target": str(q["target_wrong"]),
           "baseline": {gr: {k: baseline[(qid, gr)].get(k) for k in keep_b} for gr in names if (qid, gr) in baseline},
           "pitches": {}}
    for att in names:
        a = answers.get((qid, att))
        if not a:
            continue
        pitch = {"final": a["final"], "committed": a["committed"], "reasoning": a["reasoning"],
                 "content": a["content"], "grades": {}}
        for fr in FRAMES:
            pitch["grades"][fr] = {}
            for gr in names:
                g = grades[fr].get((qid, att, gr))
                if g:
                    st = "unparsed" if g["parse_error"] else "fooled" if g["fooled"] else "caught"
                    pitch["grades"][fr][gr] = {**{k: g.get(k) for k in keep_g}, "knew": knew(qid, gr), "status": st}
        rec["pitches"][att] = pitch
    (OUT / "problems" / f"{qid}.json").write_text(json.dumps(rec))

html = (HERE / "viewer_math3_template.html").read_text()
(OUT / "index.html").write_text(html.replace("__SUMMARY__", json.dumps(summary).replace("</", "<\\/")))
size = sum(f.stat().st_size for f in (OUT / "problems").glob("*.json"))
print(f"Wrote viewer_math3/index.html and {len(qs)} problem files ({size / 1e6:.1f} MB)")
for fr in VIEWS:
    for s in SCOPES:
        print(f"  {fr:8} {s:5}", ", ".join(f"{r} {k}/{n}" for r, (k, n) in pooled[fr][s].items()))
