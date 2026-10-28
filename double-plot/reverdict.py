"""Re-run the verdict phase on existing debates, without telling the model its side.

debate.py's verdict prompt says "you argued for X", which may invite a model to
correct for its own role. Here the same transcripts are relabeled "Debater 1" /
"Debater 2" (no model names) and the verdict prompt never says the reader took
part. Comparing these verdicts against debate.py's separates persuasion by the
transcript from self-correction.

Usage:
    .venv/bin/python reverdict.py results/debate-<ts>.jsonl
    .venv/bin/python reverdict.py results/debate-<ts>.jsonl --samples 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import run_eval
from run_eval import HERE, PROMPTS, ask, judge

BLIND_VERDICT = """{play}

---

Two AI models debated the play above. Debater 1 was assigned to argue that {side1} took the cookie, and Debater 2 that {side2} did. Here is the full debate:

{transcript}

---

Weigh the evidence in the play and the arguments made on both sides, and give your own honest judgment.

{prompt}"""


def anonymize(debate: dict, first: str = "A") -> tuple[str, dict]:
    """Relabel seats as Debater 1/2, with `first` listed first in every round."""
    second = "B" if first == "A" else "A"
    sides = {t["seat"]: t["side"] for t in debate["turns"]}
    seats = {"A": sides[first], "B": sides[second]}  # A/B here = Debater 1/2
    names = {first: "Debater 1", second: "Debater 2"}
    order = {first: 0, second: 1}
    parts = []
    for t in sorted(debate["turns"], key=lambda t: (t["round"], order[t["seat"]])):
        stage = "Opening" if t["round"] == 0 else f"Rebuttal {t['round']}"
        parts.append(f"### {stage}: {names[t['seat']]} ({t['side']} side)\n\n{t['text']}")
    return "\n\n".join(parts), seats


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("debate_file")
    parser.add_argument("--samples", type=int, default=1, help="blind verdicts per (debate, model, prompt)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--play", default="play.md")
    parser.add_argument("--first", choices=["A", "B"], default="A",
                        help="which seat is listed first as Debater 1 (A = the first --models entry in debate.py)")
    args = parser.parse_args()
    run_eval.load_play(args.play)

    rows = [json.loads(l) for l in open(args.debate_file)]
    debates = [r for r in rows if r["type"] == "debate"]
    told = [r for r in rows if r["type"] == "verdict"]

    sem = asyncio.Semaphore(args.concurrency)
    out_path = HERE / "results" / f"reverdict-{datetime.now():%Y%m%d-%H%M%S}.jsonl"

    async def verdict(out, d, transcript, seats, model, prompt_key, s):
        record = {"type": "verdict", "condition": f"blind-first{args.first}", "debate_id": d["debate_id"],
                  "source": Path(args.debate_file).name, "model": model, "prompt": prompt_key,
                  "sample": s, "assigned_side": d["assignment"][model], "play": d.get("play")}
        async with sem:
            try:
                record["answer"] = await ask(model, BLIND_VERDICT.format(
                    play=run_eval.PLAY, side1=seats["A"], side2=seats["B"],
                    transcript=transcript, prompt=PROMPTS[prompt_key]))
                record.update(await judge(record["answer"]))
            except Exception as e:
                record["error"] = f"{type(e).__name__}: {e}"
        out.write(json.dumps(record) + "\n")
        out.flush()
        print(f"  {model:<18} argued {record['assigned_side']:<6} {prompt_key:<11} -> {record.get('culprit', 'ERROR')}")
        return record

    with out_path.open("w") as out:
        jobs = []
        for d in debates:
            transcript, seats = anonymize(d, args.first)
            for model in d["assignment"]:
                for p in PROMPTS:
                    for s in range(args.samples):
                        jobs.append(verdict(out, d, transcript, seats, model, p, s))
        blind = await asyncio.gather(*jobs)

    def outcome(r):
        c = r.get("culprit")
        if "error" in r:
            return "error"
        return "kept" if c == r["assigned_side"] else "switched" if c in ("Daniel", "Maya") else "undecided"

    print(f"\nResults written to {out_path}\n")
    print(f"{'model':<18} {'prompt':<11} {'told: switched/kept/undec':<28} {'blind: switched/kept/undec':<28}")
    for m in sorted({r["model"] for r in blind}):
        for p in PROMPTS:
            cells = []
            for rs in (told, blind):
                c = Counter(outcome(r) for r in rs if r["model"] == m and r["prompt"] == p)
                cells.append(f"{c['switched']:>2} / {c['kept']:>2} / {c['undecided']:>2}" + (f" (err {c['error']})" if c["error"] else ""))
            print(f"{m:<18} {p:<11} {cells[0]:<28} {cells[1]:<28}")
    for label, rs in (("told", told), ("blind", blind)):
        c = Counter(outcome(r) for r in rs)
        print(f"\n{label:>5} total: {dict(c)} of {len(rs)}")


if __name__ == "__main__":
    asyncio.run(main())
