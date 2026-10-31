"""Build the frontier3 control page (true-side pitches) -> runs/<pool>/control/viewer/index.html + posts/<pid>.json.
Needs runs/<pool>/control/summary.json from analyze_control.py. Shown as the "Frontier control" tab of the AITA viewer.

    python build_control_viewer.py --pool frontier3
"""
import json

import generate
import grade
from common import CONTROL, MAIN_RUNS, MODELS, format_post, load_posts, read_jsonl

assert not CONTROL, "run without --control"
CTRL = MAIN_RUNS / "control"
OUT = CTRL / "viewer"
(OUT / "posts").mkdir(parents=True, exist_ok=True)
NAMES = [m["name"] for m in MODELS]
FRAMES = {"neutral": "grades.jsonl", "self": "grades_self.jsonl"}
S = json.loads((CTRL / "summary.json").read_text())
sub = json.loads((CTRL / "subset.json").read_text())
posts = {p["id"]: p for p in load_posts() if p["id"] in sub["truth"]}
base = {(b["pid"], b["grader"]): b for b in read_jsonl(CTRL / "baseline.jsonl")}
rerun = {(b["pid"], b["grader"]): b for b in read_jsonl(CTRL / "baseline_rerun.jsonl")}


def idx(root, fname, key):
    out = {}
    for att in NAMES:
        p = root / att / fname
        if p.exists():
            for r in read_jsonl(p):
                if r["pid"] in posts:
                    out[key(r)] = r
    return out


answers = {"true": idx(CTRL, "answers.jsonl", lambda r: (r["pid"], r["generator"])),
           "false": idx(MAIN_RUNS, "answers.jsonl", lambda r: (r["pid"], r["generator"]))}
grades = {c: {fr: idx(root, f, lambda r: (r["pid"], r["attacker"], r["grader"])) for fr, f in FRAMES.items()}
          for c, root in (("true", CTRL), ("false", MAIN_RUNS))}

keep = ("verdict", "agree", "confidence", "reasoning", "content")
post_list = []
for pid, p in posts.items():
    truth = p["verdict"].upper()
    push = sum(g["verdict"] is not None and g["verdict"] != truth
               for fr in FRAMES for g in grades["true"][fr].values() if g["pid"] == pid)
    flips = sum(g["verdict"] is not None and g["verdict"] != truth
                for fr in FRAMES for g in grades["false"][fr].values() if g["pid"] == pid)
    n = sum(1 for fr in FRAMES for g in grades["true"][fr].values() if g["pid"] == pid)
    post_list.append({"pid": pid, "title": p["title"], "truth": truth, "push": push, "flips": flips, "n": n,
                      "why": sub["why"][pid]})
    rec = {"pid": pid, "title": p["title"], "text": p["text"], "truth": truth,
           "baseline": {g: {"first": base[(pid, g)]["verdict"], "rerun": (rerun.get((pid, g)) or {}).get("verdict")}
                        for g in NAMES},
           "pitches": {}}
    for at in NAMES:
        rec["pitches"][at] = {}
        for c in ("true", "false"):
            x = answers[c].get((pid, at))
            if not x:
                continue
            rec["pitches"][at][c] = {
                "target": x["target"], "committed": x["committed"],
                # Full text for the control pitch; the false-side pitch is in the Frontier tab already.
                "content": x["content"] if c == "true" else None,
                "grades": {fr: {gr: {k: (g.get(k) if c == "true" or k in ("verdict", "agree", "confidence") else None)
                                     for k in keep}
                                for (pp, aa, gr), g in grades[c][fr].items() if pp == pid and aa == at}
                           for fr in FRAMES}}
    (OUT / "posts" / f"{pid}.json").write_text(json.dumps(rec))

data = {"models": NAMES, "summary": S, "posts": post_list,
        "prompts": {"attacker": generate.SYSTEM.format(target="{target}", gloss="{gloss}"),
                    "neutral": grade.NEUTRAL.format(post="{post}", pitch="{pitch}"),
                    "self_ask": grade.SELF_ASK.format(post="{post}"), "self_verify": grade.SELF_VERIFY}}
tpl = (CTRL.parent.parent.parent / "control_viewer_template.html").read_text()
(OUT / "index.html").write_text(tpl.replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/")))
print(f"wrote {OUT}/index.html and {len(post_list)} posts")
