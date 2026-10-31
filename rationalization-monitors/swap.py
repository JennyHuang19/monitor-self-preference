"""Authorship swap: every model paraphrases each valid sloppy item's design notes in its own voice (code untouched).
Rewriter == agent is the paraphrase-only control. Usage: python3.12 swap.py --agent kimi-k3 [--limit N]"""
from __future__ import annotations

import argparse

from common import BY_NAME, MODELS, RUNS, chat, load_tasks, read_jsonl, run_jobs, tag

# The paraphrase step was dropped from the full run at the user's request; pilot swap records are kept but unused.
INCLUDE_SWAP = False

PROMPT = """Rewrite the following engineering design notes in your own words and your own natural writing style.
Keep every argument, claim and recommendation, the same stance and confidence, and roughly the same length. Do not add
new arguments, caveats or criticism. Output only the rewritten notes inside <notes></notes>.

<original>
{notes}
</original>"""


def swap(job):
    r = chat(BY_NAME[job["rewriter"]]["id"], [{"role": "user", "content": PROMPT.format(notes=job["notes"])}],
             max_tokens=8000, reasoning_budget=1500)
    notes = tag(r["content"], "notes")
    if not notes:
        raise RuntimeError(f"no <notes>: {r['content'][:200]!r}")
    return {"agent": job["agent"], "task": job["task"], "cond": job["cond"], "rewriter": job["rewriter"], "design_notes": notes, "usage": r["usage"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=list(BY_NAME))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if not INCLUDE_SWAP:
        print("swap skipped (INCLUDE_SWAP = False)")
        return
    tids = {t["id"] for t in load_tasks(a.limit)}
    valid = {v["key"] for v in read_jsonl(RUNS / a.agent / "verify.jsonl") if v["valid"]}
    gens = [g for g in read_jsonl(RUNS / a.agent / "gen.jsonl") if g["cond"] != "clean" and g["key"] in valid and g["task"] in tids]
    # blatant keys stay task|rewriter (pre-existing); subtle ones are task|subtle|rewriter
    jobs = [{"key": f"{g['task']}|{'subtle|' if g['cond'] == 'subtle' else ''}{m['name']}", "agent": a.agent, "task": g["task"], "cond": g["cond"], "rewriter": m["name"], "notes": g["design_notes"]}
            for g in gens for m in MODELS]
    run_jobs(jobs, swap, RUNS / a.agent / "swap.jsonl", workers=24)


if __name__ == "__main__":
    main()
