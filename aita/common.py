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
run_jobs, read_jsonl, parse_json, append = gm.run_jobs, gm.read_jsonl, gm.parse_json, gm.append

import direct  # noqa: E402

# AITA-only frontier models (not in the gaslighting registry). These two go to the vendors' own APIs, keyed by
# their OpenRouter id; everything else (incl. gemini-4-argon, no Google key) goes through OpenRouter.
_FRONTIER3_EXTRA = [{"name": "gpt-6-astra", "id": "openai/gpt-6-astra", "family": "GPT"},
                    {"name": "claude-opus-4.8", "id": "anthropic/claude-opus-4.8", "family": "Claude"}]
DIRECT = {"openai/gpt-6-astra": ("openai", "gpt-6-astra"), "anthropic/claude-opus-4.8": ("anthropic", "claude-opus-4-8")}


def chat(model_id: str, messages: list[dict], *, max_tokens: int = 32000, **kw) -> dict:
    if model_id in DIRECT:
        return direct.chat(*DIRECT[model_id], messages, max_tokens=max_tokens)
    return gm.chat(model_id, messages, max_tokens=max_tokens, **kw)
_ALL = {m["name"]: m for m in gm.GRADERS + gm.OPEN_MODELS + gm.CLOSED2 + _FRONTIER3_EXTRA}
# open: the original five, runs in runs/. Other pools get runs/<pool>/.
POOLS = {"open": ("glm-5.3", "glm-5.3-flash", "gpt-oss-120b", "gpt-oss-20b", "kimi-k3"),
         "closed2": tuple(m["name"] for m in gm.CLOSED2),
         "frontier3": ("gpt-6-astra", "claude-opus-4.8", "gemini-4-argon")}
# Read --pool at import so analyze*.py (which take no other args) and the constants below follow it.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--pool", choices=list(POOLS), default="open")
POOL = _pre.parse_known_args()[0].pool
MODELS = [_ALL[n] for n in POOLS[POOL]]
RUNS = HERE / "runs" if POOL == "open" else HERE / "runs" / POOL
RUNS.mkdir(parents=True, exist_ok=True)
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
                    help="repeatable; default: every model in the pool")
    ap.add_argument("--limit", type=int, default=None, help="only the first N posts (smoke test)")
    ap.add_argument("--frame", choices=["neutral", "self"], default="neutral", help="grade.py only")
    ap.add_argument("--baseline-only", action="store_true", help="grade.py: only run the no-pitch baseline")
    ap.add_argument("--reparse", action="store_true", help="grade.py: re-parse stored replies, no model calls")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pool", choices=list(POOLS), default="open", help="which models attack and grade (read at import)")
    a = ap.parse_args()
    a.attackers = [model(n) for n in a.attacker] if a.attacker else MODELS
    return a
