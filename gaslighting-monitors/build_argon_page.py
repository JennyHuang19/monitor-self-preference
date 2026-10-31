"""Add the argon_glm math4 run as a second tab of the published math viewer
(claude.ai/artifact/KvVTmXRJi9EuRPGjFdLe8Z, the mix5 math3 page).

    python build_argon_page.py TEMPLATE.html OUT_DIR [--draws 2000]

TEMPLATE.html: the published page (Artifact read). Writes OUT_DIR/index.html (both tabs; the mix5 summary is kept
as is), OUT_DIR/argon/problems/<qid>.json (one per math4 problem) and OUT_DIR/argon/summary.json. The mix5 tab's
problems/<qid>.json files are already published and are not touched.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from common import HERE, POOLS, read_jsonl, run_dir
from mixed_logit import analyze

POOL, DOMAIN = "argon_glm", "math4"
FRAMES = {"neutral": "grades.jsonl", "self": "grades_self.jsonl", "skim1k": "grades_skim1k.jsonl"}
MODELS = POOLS[POOL]
NAMES = [m["name"] for m in MODELS]
BASE_KEYS = ("answer", "confidence", "solved", "reasoning", "content", "parse_error", "provider")
GRADE_KEYS = ("verdict", "confidence", "grader_correct_option", "fooled", "reasoning", "content", "parse_error", "provider")


def status(g):
    return "unparsed" if g["parse_error"] else "fooled" if g["fooled"] else "caught"


def build_data(draws):
    qs = json.loads((HERE / f"questions_{DOMAIN}.json").read_text())
    qids = {q["id"] for q in qs}
    base = defaultdict(list)
    for b in read_jsonl(run_dir(DOMAIN, "baseline", POOL) / "baseline.jsonl"):
        base[(b["qid"], b["grader"])].append(b)
    for v in base.values():
        v.sort(key=lambda b: b.get("sample", 0))
    knew = {k: all(b["solved"] for b in v) for k, v in base.items()}

    answers, grades = {}, []
    commit = {}
    for a in NAMES:
        d = run_dir(DOMAIN, a, POOL)
        ans = [x for x in read_jsonl(d / "answers.jsonl") if x["qid"] in qids]
        commit[a] = [sum(x["committed"] for x in ans), len(ans)]
        for x in ans:
            answers[(x["qid"], a)] = x
        for fr, f in FRAMES.items():
            for g in read_jsonl(d / f):
                if g["qid"] in qids:
                    grades.append({**g, "frame": fr, "knew": knew.get((g["qid"], g["grader"]), False)})
    ok = [g for g in grades if not g["parse_error"]]
    fr_all = list(FRAMES) + ["all"]
    inf = lambda g, fr, sc: (fr == "all" or g["frame"] == fr) and (sc == "all" or g["knew"])

    def kn(gs):
        return [sum(bool(g["fooled"]) for g in gs), len(gs)]

    matrix = {fr: {sc: {a: {g: kn([x for x in ok if x["attacker"] == a and x["grader"] == g and inf(x, fr, sc)])
                            for g in NAMES} for a in NAMES} for sc in ("knew", "all")} for fr in fr_all}
    pooled = {fr: {sc: {r: kn([x for x in ok if x["relation"] == r and inf(x, fr, sc)]) for r in ("self", "sibling", "cross")}
                   for sc in ("knew", "all")} for fr in fr_all}
    pair = {fr: {sc: {} for sc in ("knew", "all")} for fr in fr_all}
    for fr in fr_all:
        for sc in ("knew", "all"):
            for x in ok:
                if x["fooled"] and inf(x, fr, sc):
                    lst = pair[fr][sc].setdefault(f"{x['attacker']}|{x['grader']}", [])
                    if x["qid"] not in lst:
                        lst.append(x["qid"])
    etype = {q["id"]: q["error_type"] for q in qs}
    types = sorted(set(etype.values()))
    by_type = {fr: {sc: {g: {t: kn([x for x in ok if x["grader"] == g and etype[x["qid"]] == t and inf(x, fr, sc)])
                             for t in types} for g in NAMES} for sc in ("knew", "all")} for fr in fr_all}
    items = []
    for q in qs:
        gs = [x for x in ok if x["qid"] == q["id"]]
        items.append({"qid": q["id"], "question": q["question"], "topic": q["topic"], "etype": q["error_type"],
                      "answer": str(q["answer"]), "target": str(q["target_wrong"]), "source": q.get("source"),
                      "fools": {fr: {sc: kn([x for x in gs if inf(x, fr, sc)]) for sc in ("knew", "all")} for fr in fr_all}})
    solve = {g: [sum(knew.get((q["id"], g), False) for q in qs), len(qs)] for g in NAMES}

    # Candidate pool, for the lede: how many problems were screened and each model's pass rate on them.
    cands = json.loads((HERE / f"questions_{DOMAIN}_candidates.json").read_text())
    cand_ids = {q["id"] for q in cands}
    screened = [q for q in cands if all(base.get((q["id"], g)) for g in NAMES)]
    cand_solve = {g: [sum(knew.get((q["id"], g), False) for q in screened), len(screened)] for g in NAMES}
    screen = {"n_candidates": len(cands), "n_screened": len(screened), "n_pass": sum(
        all(knew.get((q["id"], g), False) for g in NAMES) for q in screened), "solve": cand_solve,
        "n_carried": sum(q.get("source") == "math3" for q in screened),
        "n_new": sum(q.get("source") == "new" for q in screened), "samples": 2, "n_kept": len(qs)}
    assert cand_ids >= qids

    rows = [{"fooled": x["fooled"], "attacker": x["attacker"], "grader": x["grader"], "frame": x["frame"],
             "relation": x["relation"], "solved": x["knew"], "qid": x["qid"]} for x in ok]
    reg = analyze(rows, "gemini-4-argon", draws=draws)
    summary = {"models": MODELS, "matrix": matrix, "pooled": pooled, "pairFools": pair, "solve": solve,
               "byType": by_type, "typeCounts": {t: sum(v == t for v in etype.values()) for t in types}, "commit": commit,
               "regression": {"frames": list(FRAMES), "pooled": {"all": {"n_rows": reg["n_rows"], "n_fooled": reg["n_fooled"],
                                                                           "model": reg}}},
               "items": items, "screen": screen, "unparsed": len(grades) - len(ok)}

    problems = {}
    for q in qs:
        bl = {}
        for g in NAMES:
            v = base.get((q["id"], g), [])
            if v:
                bl[g] = {**{k: v[0].get(k) for k in BASE_KEYS}, "solved": knew[(q["id"], g)],
                         "samples": [{"answer": b.get("answer"), "solved": b["solved"]} for b in v]}
        pitches = {}
        for a in NAMES:
            x = answers.get((q["id"], a))
            if not x:
                continue
            gr = defaultdict(dict)
            for g in grades:
                if g["qid"] == q["id"] and g["attacker"] == a:
                    gr[g["frame"]][g["grader"]] = {**{k: g.get(k) for k in GRADE_KEYS}, "knew": g["knew"], "status": status(g)}
            pitches[a] = {"final": x["final"], "committed": x["committed"], "reasoning": x.get("reasoning"),
                          "content": x["content"], "grades": gr}
        problems[q["id"]] = {"qid": q["id"], "question": q["question"], "topic": q["topic"], "error_type": q["error_type"],
                             "error_description": q.get("error_description"), "cases": q.get("cases", []),
                             "solution": q.get("solution"), "check_code": q.get("check_code"), "answer": str(q["answer"]),
                             "target": str(q["target_wrong"]), "orig_id": q.get("orig_id"), "baseline": bl,
                             "pitches": pitches}
    return summary, problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("out")
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "argon" / "problems").mkdir(parents=True, exist_ok=True)
    summary, problems = build_data(a.draws)
    for qid, p in problems.items():
        (out / "argon" / "problems" / f"{qid}.json").write_text(json.dumps(p))
    (out / "argon" / "summary.json").write_text(json.dumps(summary, indent=1))
    from page_patch import patch
    (out / "index.html").write_text(patch(Path(a.template).read_text(), summary))
    print(f"{len(problems)} problems; {summary['regression']['pooled']['all']['n_rows']} graded rows -> {out}")
