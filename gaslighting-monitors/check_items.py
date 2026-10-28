"""Check the math3 / trivia3 items, keep the first 100 that pass, and renumber them -> questions_<domain>.json.
Every item and its check results go to runs/questions_<domain>_checked.json.

math3: check_code prints the answer, target_wrong differs, omitted_case items list 6+ cases.
trivia3: fields present, target_wrong doesn't match the answer or an alias, number answers are integers.
Both: no duplicate questions.

    python check_items.py --domain math3|trivia3
Run it before any attacker/grader runs: renumbering changes the item ids.
"""
import json
import re
import subprocess
import sys

from common import HERE, RUNS, answer_matches, domain_args, norm_answer

N_KEEP = 100


def check_math(q):
    try:
        r = subprocess.run([sys.executable, "-I", "-c", q["check_code"]], capture_output=True, text=True, timeout=30)
        printed = r.stdout.strip()
    except subprocess.TimeoutExpired:
        printed = "TIMEOUT"
    return {"check_code prints answer": printed == str(q["answer"]),
            "integer answer": isinstance(q["answer"], int) and isinstance(q["target_wrong"], int),
            "target_wrong differs": q["target_wrong"] != q["answer"],
            "6+ cases": q.get("error_type") != "omitted_case" or len(q.get("cases", [])) >= 6}, printed


def check_trivia(q):
    keys = [q.get("answer"), *q.get("aliases", [])]
    return {"fields": all(q.get(k) for k in ("question", "answer", "target_wrong", "source")),
            "target_wrong differs": not answer_matches(q.get("target_wrong"), keys),
            "number answer is integer": q.get("answer_type") != "number"
                                        or bool(re.fullmatch(r"-?\d+", norm_answer(q.get("answer"))))}, None


if __name__ == "__main__":
    args = domain_args(__doc__)
    path = HERE / f"questions_{args.domain}.json"
    items = json.loads(path.read_text())
    seen, keep, report = set(), [], []
    for q in items:
        checks, printed = (check_math if args.domain == "math3" else check_trivia)(q)
        dup = norm_answer(q["question"]) in seen
        checks["not a duplicate"] = not dup
        seen.add(norm_answer(q["question"]))
        ok = all(checks.values())
        report.append({"orig_id": q["id"], "batch": q.get("batch"), "ok": ok, "printed": printed, "checks": checks, "item": q})
        if not ok:
            print(f"DROP {q['id']} (batch {q.get('batch')}): {[k for k, v in checks.items() if not v]} printed={printed!r}")
        elif len(keep) < N_KEEP:
            keep.append(q)
    prefix = "p" if args.domain == "math3" else "t"
    for i, q in enumerate(keep):
        q["orig_id"], q["id"] = q["id"], f"{prefix}{i + 1:03d}"
    (RUNS / f"questions_{args.domain}_checked.json").write_text(json.dumps(report, indent=2))
    path.write_text(json.dumps(keep, indent=2))
    print(f"{sum(r['ok'] for r in report)}/{len(items)} passed; kept {len(keep)} -> {path.name}")
    sys.exit(0 if len(keep) == N_KEEP else 1)
