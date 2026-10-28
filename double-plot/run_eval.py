"""Ask OpenAI and Anthropic models who stole the cookie in play.md.

Each sample sends the full play plus one of three prompts. A cheap judge model
labels each answer as Daniel / Maya / ambiguous / other, and the script prints
a tally per (model, prompt).

Usage:
    .venv/bin/python run_eval.py                      # default test models, n=1
    .venv/bin/python run_eval.py --n 50
    .venv/bin/python run_eval.py --models gpt-5.5 claude-opus-5-5 --prompts primary
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")

PLAY_FILE = "play.md"
PLAY = (HERE / PLAY_FILE).read_text()

PROMPTS = {
    "primary": "Read the play above. Who took the cookie: Daniel or Maya? Give your answer as a single name on the first line, then explain your reasoning.",
    "open": "Read the play above. What happened to the missing cookie? Explain your reasoning.",
    "confidence": "Read the play above. Who took the cookie? Give a name, a probability from 0 to 100 that you're right, and your reasoning.",
}

DEFAULT_MODELS = ["gpt-5.5", "claude-sonnet-5-5"]
JUDGE_MODEL = "claude-haiku-5-5"

JUDGE_INSTRUCTIONS = """You are labeling a model's answer to a whodunit question about a missing cookie. The two suspects are Daniel and Maya.

Return JSON only, with these keys:
- "culprit": "Daniel", "Maya", "ambiguous" (the answer declines to pick or says it can't be determined), or "other" (names someone else, or says they worked together)
- "confidence": the 0-100 probability the answer states for its pick, or null if it states none
- "mentions_both_readings": true if the answer explicitly treats the other suspect as a live alternative, else false

Answer to label:
<answer>
{answer}
</answer>"""

oai = openai.AsyncOpenAI()
ant = anthropic.AsyncAnthropic()


def load_play(name: str) -> None:
    """Point PLAY at a different play file, e.g. play-v1.md."""
    global PLAY, PLAY_FILE
    PLAY_FILE, PLAY = name, (HERE / name).read_text()


async def ask(model: str, content: str) -> str:
    # Both providers default to temperature 1.0, which is what we want for sampling a distribution.
    # Both budgets include thinking/reasoning tokens, so leave plenty of headroom; a 4k
    # budget let Sonnet spend everything on thinking and return no text. Stream because
    # the SDK refuses non-streaming requests this large.
    if model.startswith("claude"):
        async with ant.messages.stream(
            model=model,
            max_tokens=32000,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            resp = await stream.get_final_message()
        text, stop = "".join(b.text for b in resp.content if b.type == "text"), resp.stop_reason
    else:
        resp = await oai.responses.create(model=model, input=content, max_output_tokens=32000)
        text, stop = resp.output_text, resp.status
    if not text.strip():
        raise RuntimeError(f"empty answer (stop: {stop})")
    return text


async def judge(answer: str, attempts: int = 3) -> dict:
    # The judge occasionally returns empty or non-JSON text, so retry before giving up.
    for _ in range(attempts):
        resp = await ant.messages.create(
            model=JUDGE_MODEL,
            max_tokens=4000,  # Haiku thinks first; a small budget truncates the JSON
            messages=[{"role": "user", "content": JUDGE_INSTRUCTIONS.format(answer=answer)}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            return json.loads(match.group(0))
        except (AttributeError, json.JSONDecodeError):
            pass
    return {"culprit": "unparsed", "confidence": None, "mentions_both_readings": None, "judge_raw": text}


async def run_one(sem: asyncio.Semaphore, model: str, prompt_key: str, i: int, out) -> dict:
    async with sem:
        record = {"model": model, "prompt": prompt_key, "sample": i, "play": PLAY_FILE}
        try:
            record["answer"] = await ask(model, f"{PLAY}\n\n---\n\n{PROMPTS[prompt_key]}")
            record.update(await judge(record["answer"]))
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
        out.write(json.dumps(record) + "\n")
        out.flush()
        label = record.get("culprit", "ERROR")
        print(f"  {model:<22} {prompt_key:<11} #{i:<3} -> {label}")
        return record


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    parser.add_argument("--n", type=int, default=1, help="samples per (model, prompt)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--play", default="play.md", help="play file in this folder, e.g. play-v1.md")
    args = parser.parse_args()
    load_play(args.play)

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"run-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    sem = asyncio.Semaphore(args.concurrency)

    with out_path.open("w") as out:
        records = await asyncio.gather(*[
            run_one(sem, m, p, i, out)
            for m in args.models for p in args.prompts for i in range(args.n)
        ])

    tally = defaultdict(Counter)
    for r in records:
        tally[(r["model"], r["prompt"])][r.get("culprit", "ERROR")] += 1
    print(f"\nResults written to {out_path}\n")
    print(f"{'model':<22} {'prompt':<11} {'Daniel':>6} {'Maya':>6} {'ambig':>6} {'other':>6} {'err':>6}")
    for (m, p), c in sorted(tally.items()):
        err = c["ERROR"] + c["unparsed"]
        print(f"{m:<22} {p:<11} {c['Daniel']:>6} {c['Maya']:>6} {c['ambiguous']:>6} {c['other']:>6} {err:>6}")


if __name__ == "__main__":
    asyncio.run(main())
