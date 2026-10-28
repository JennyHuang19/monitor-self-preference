"""AITA flip experiment: posts, models and run dirs. The OpenRouter client and JSONL helpers come from
../gaslighting-monitors/common.py (loaded under another name, since this file is also called common)."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)
DATA = HERE / "data" / "sample_75nta_25yta.json"

_spec = importlib.util.spec_from_file_location("gm_common", HERE.parent / "gaslighting-monitors" / "common.py")
gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gm)
chat, run_jobs, read_jsonl, parse_json, append = gm.chat, gm.run_jobs, gm.read_jsonl, gm.parse_json, gm.append

_ALL = {m["name"]: m for m in gm.GRADERS + gm.OPEN_MODELS}
MODELS = [_ALL[n] for n in ("glm-5.3", "glm-5.3-flash", "gpt-oss-120b", "gpt-oss-20b", "kimi-k3")]
VERDICTS = ("YTA", "NTA")


def model(name: str) -> dict:
    return next(m for m in MODELS if m["name"] == name)


def load_posts(limit: int | None = None) -> list[dict]:
    posts = json.loads(DATA.read_text())
    return posts[:limit] if limit else posts


def format_post(p: dict) -> str:
    # Title and body only: the Reddit comments state the verdict.
    return f"Title: {p['title']}\n\n{p['text']}"


def truth(p: dict) -> str:
    return p["verdict"].upper()


def target(p: dict) -> str:
    """The attacker argues for the opposite of Reddit's verdict."""
    return "NTA" if truth(p) == "YTA" else "YTA"


def relation(grader: dict, att: dict) -> str:
    if grader["name"] == att["name"]:
        return "self"
    return "sibling" if grader["family"] == att["family"] else "cross"


def run_dir(attacker_name: str) -> Path:
    d = RUNS / attacker_name
    d.mkdir(exist_ok=True)
    return d


def norm_verdict(v) -> str | None:
    v = str(v or "").strip().upper()
    return v if v in VERDICTS else None


def args(desc: str):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--attacker", choices=[m["name"] for m in MODELS], action="append",
                    help="repeatable; default: all five")
    ap.add_argument("--limit", type=int, default=None, help="only the first N posts (smoke test)")
    ap.add_argument("--frame", choices=["neutral", "self"], default="neutral", help="grade.py only")
    ap.add_argument("--baseline-only", action="store_true", help="grade.py: only run the no-pitch baseline")
    ap.add_argument("--reparse", action="store_true", help="grade.py: re-parse stored replies, no model calls")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.attackers = [model(n) for n in a.attacker] if a.attacker else MODELS
    return a
