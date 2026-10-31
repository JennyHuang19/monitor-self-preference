"""Build the AITA flip viewer: viewer/index.html (overview + post list) and viewer/posts/<pid>.json (one post's
pitches, grades and CoTs, loaded on demand). Flip logic matches analyze.py.

    python build_viewer.py
"""
import json
from collections import defaultdict

import generate
import grade
from common import HERE, MODELS, RUNS, load_posts, read_jsonl, relation, run_dir

FRAMES = {"neutral": "grades.jsonl", "self": "grades_self.jsonl"}
OUT = HERE / "viewer"
(OUT / "posts").mkdir(parents=True, exist_ok=True)
names = [m["name"] for m in MODELS]


def status(base, verdict, target):
    """How the grader's verdict moved relative to its no-pitch verdict."""
    if verdict is None:
        return "unparsed"
    if base is None:
        return "no-baseline"
    if base != target:
        return "flipped" if verdict == target else "held"
    return "stayed" if verdict == target else "reversed"


posts = load_posts()
baseline = {(b["pid"], b["grader"]): b for b in read_jsonl(RUNS / "baseline.jsonl")}
answers = {(a["pid"], att): a for att in names for a in read_jsonl(run_dir(att) / "answers.jsonl")}
grades = defaultdict(dict)  # frame -> (pid, attacker, grader) -> record
for fr, fname in FRAMES.items():
    for att in names:
        for g in read_jsonl(run_dir(att) / fname):
            grades[fr][(g["pid"], att, g["grader"])] = g

# Overview: attacker x grader flip matrices, pooled by relation, and per-post flip counts.
matrix = {fr: {a: {g: [0, 0] for g in names} for a in names} for fr in FRAMES}
pooled = {fr: {r: [0, 0] for r in ("self", "sibling", "cross")} for fr in FRAMES}
post_flips = {fr: defaultdict(lambda: [0, 0]) for fr in FRAMES}
pair_flips = {fr: defaultdict(list) for fr in FRAMES}  # "att|grader" -> pids that flipped
for fr in FRAMES:
    for (pid, att, gr), g in grades[fr].items():
        b = baseline.get((pid, gr), {}).get("verdict")
        if b is None or b == g["target"] or g["verdict"] is None:  # unparsed grades are dropped
            continue
        flip = g["verdict"] == g["target"]
        m = matrix[fr][att][gr]
        m[0] += flip
        m[1] += 1
        rel = relation(next(x for x in MODELS if x["name"] == gr), next(x for x in MODELS if x["name"] == att))
        pooled[fr][rel][0] += flip
        pooled[fr][rel][1] += 1
        post_flips[fr][pid][0] += flip
        post_flips[fr][pid][1] += 1
        if flip:
            pair_flips[fr][f"{att}|{gr}"].append(pid)

base_agree = {}
for gr in names:
    row = {}
    for t in ("NTA", "YTA"):
        sub = [b for b in baseline.values() if b["grader"] == gr and b["truth"] == t]
        row[t] = [sum(b["matches_reddit"] for b in sub), len(sub)]
    base_agree[gr] = row

commit = {att: [sum(answers[(p["id"], att)]["committed"] for p in posts if (p["id"], att) in answers),
                sum((p["id"], att) in answers for p in posts)] for att in names}
controlled = json.loads((RUNS / "controlled.json").read_text())
regression = json.loads((RUNS / "regression.json").read_text())
cot = json.loads((RUNS / "cot_compare.json").read_text()) if (RUNS / "cot_compare.json").exists() else None

index = [{"pid": p["id"], "title": p["title"], "truth": p["verdict"].upper(), "score": p.get("score"),
          "flips": {fr: post_flips[fr][p["id"]] for fr in FRAMES}} for p in posts]

# The prompts as sent, with placeholders left in (format() also turns the templates' {{ }} back into { }).
ph = {"post": "{post}", "pitch": "{pitch}"}
prompts = [
    {"title": "Attacker", "note": "system prompt + the post as the user turn",
     "turns": [["system", generate.SYSTEM.format(target="{target}", gloss="{gloss: the poster is / is not the asshole}")],
               ["user", "{post}"]]},
    {"title": "Grader · neutral frame", "note": "one user turn",
     "turns": [["user", grade.NEUTRAL.format(**ph)]]},
    {"title": "Grader · self-check frame", "note": "the pitch is placed as the grader's own earlier answer",
     "turns": [["user", grade.SELF_ASK.format(**ph)], ["assistant", "{pitch}"], ["user", grade.SELF_VERIFY.format()]]},
    {"title": "Grader · no-pitch baseline", "note": "sets the verdict a flip is measured against",
     "turns": [["user", grade.BASELINE.format(**ph)]]},
]

summary = {"models": MODELS, "matrix": matrix, "pooled": pooled, "pairFlips": pair_flips, "baseAgree": base_agree,
           "commit": commit, "controlled": controlled, "regression": regression, "cot": cot, "prompts": prompts, "posts": index}

# One file per post: the post, the no-pitch verdicts, each attacker's pitch, and every grade in both frames.
keep = ("verdict", "agree", "confidence", "reasoning", "content", "parse_error", "provider")
for p in posts:
    pid = p["id"]
    rec = {"pid": pid, "title": p["title"], "text": p["text"], "truth": p["verdict"].upper(), "score": p.get("score"),
           "comments": [c for c in (p.get("comment1"), p.get("comment2")) if c],
           "baseline": {gr: {k: baseline[(pid, gr)].get(k) for k in keep} for gr in names if (pid, gr) in baseline},
           "pitches": {}}
    for att in names:
        a = answers.get((pid, att))
        if not a:
            continue
        pitch = {"target": a["target"], "final": a["final"], "committed": a["committed"],
                 "reasoning": a["reasoning"], "content": a["content"], "grades": {}}
        for fr in FRAMES:
            pitch["grades"][fr] = {}
            for gr in names:
                g = grades[fr].get((pid, att, gr))
                if g:
                    b = baseline.get((pid, gr), {}).get("verdict")
                    pitch["grades"][fr][gr] = {**{k: g.get(k) for k in keep}, "base": b,
                                               "status": status(b, g["verdict"], g["target"])}
        rec["pitches"][att] = pitch
    (OUT / "posts" / f"{pid}.json").write_text(json.dumps(rec))

html = (HERE / "viewer_template.html").read_text()
(OUT / "index.html").write_text(html.replace("__SUMMARY__", json.dumps(summary).replace("</", "<\\/")))
size = sum(f.stat().st_size for f in (OUT / "posts").glob("*.json"))
print(f"Wrote viewer/index.html and {len(posts)} post files ({size / 1e6:.1f} MB)")
