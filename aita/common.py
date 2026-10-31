"""AITA flip experiment: posts, models and run dirs. The OpenRouter client and JSONL helpers come from
../gaslighting-monitors/common.py (loaded under another name, since this file is also called common)."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "sample_75nta_25yta.json"

_spec = importlib.util.spec_from_file_location("gm_common", HERE.parent / "gaslighting-monitors" / "common.py")
gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gm)
chat, run_jobs, read_jsonl, parse_json, append = gm.chat, gm.run_jobs, gm.read_jsonl, gm.parse_json, gm.append

_EXTRA = [{"name": "qwen3.8-flash", "id": "qwen/qwen3.8-flash", "family": "Qwen"},
          {"name": "kimi-k3.1-pro", "id": "moonshotai/kimi-k3.1-pro", "family": "Kimi"}]
_ALL = {m["name"]: m for m in gm.GRADERS + gm.OPEN_MODELS + _EXTRA}
# Strength tier, fixed before any of the new models were run: 1 flagship, 2 flash, 3 open-large, 4 open-small.
# Each tier holds models from 2+ families, so family match can vary at a fixed strength gap.
TIER = {"glm-5.3": 1, "kimi-k3": 1, "kimi-k3.1-pro": 1, "qwen3.8-max": 1,
        "glm-5.3-flash": 2, "qwen3.8-flash": 2,
        "gpt-oss-120b": 3, "gemma-4-31b": 3,
        "gpt-oss-20b": 4, "gemma-4-26b-a4b": 4}
MODELS = [_ALL[n] for n in ("glm-5.3", "glm-5.3-flash", "gpt-oss-120b", "gpt-oss-20b", "kimi-k3",
                            "qwen3.8-max", "qwen3.8-flash", "gemma-4-31b", "gemma-4-26b-a4b", "kimi-k3.1-pro")]
VERDICTS = ("YTA", "NTA")


def model(name: str) -> dict:
    return next(m for m in MODELS if m["name"] == name)


def load_posts(limit: int | None = None) -> list[dict]:
    posts = json.loads(DATA.read_text())
    if CONTROL:
        keep = set(json.loads((RUNS / "subset.json").read_text())["pids"])
        posts = [p for p in posts if p["id"] in keep]
    return posts[:limit] if limit else posts


def format_post(p: dict) -> str:
    # Title and body only: the Reddit comments state the verdict.
    return f"Title: {p['title']}\n\n{p['text']}"


def truth(p: dict) -> str:
    return p["verdict"].upper()


def target(p: dict) -> str:
    """The attacker argues for the opposite of Reddit's verdict (or for Reddit's verdict under --control)."""
    if CONTROL:
        return truth(p)
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
                    help="repeatable; default: every model in the pool")
    ap.add_argument("--limit", type=int, default=None, help="only the first N posts (smoke test)")
    ap.add_argument("--frame", choices=["neutral", "self"], default="neutral", help="grade.py only")
    ap.add_argument("--baseline-only", action="store_true", help="grade.py: only run the no-pitch baseline")
    ap.add_argument("--baseline-rerun", action="store_true",
                    help="grade.py: a second no-pitch sample -> baseline_rerun.jsonl (test-retest noise floor)")
    ap.add_argument("--reparse", action="store_true", help="grade.py: re-parse stored replies, no model calls")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pool", choices=list(POOLS), default="open", help="which models attack and grade (read at import)")
    ap.add_argument("--control", action="store_true", help="argue for Reddit's verdict on the control subset (read at import)")
    a = ap.parse_args()
    a.attackers = [model(n) for n in a.attacker] if a.attacker else MODELS
    return a
