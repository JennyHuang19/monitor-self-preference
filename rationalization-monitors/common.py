"""Rationalization experiment: models, paths and shared helpers. The OpenRouter client and JSONL helpers come from
../gaslighting-monitors/common.py (loaded under another name, since this file is also called common)."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)
TASKS = HERE / "tasks.json"

_spec = importlib.util.spec_from_file_location("gm_common", HERE.parent / "gaslighting-monitors" / "common.py")
gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gm)
chat, run_jobs, read_jsonl, parse_json, append = gm.chat, gm.run_jobs, gm.read_jsonl, gm.parse_json, gm.append
WRITER = gm.QUESTION_WRITER  # neutral (non-monitored) model for task writing and verification

# Three open-weight families. Every model is both agent and monitor; no siblings, so relation is self or cross.
MODELS = [
    {"name": "kimi-k3", "id": "moonshotai/kimi-k3", "family": "Kimi"},
    {"name": "glm-5.3", "id": "z-ai/glm-5.3", "family": "GLM"},
    {"name": "qwen3.8-2.4t", "id": "qwen/qwen3.8-2.4t-a95b", "family": "Qwen"},
]
BY_NAME = {m["name"]: m for m in MODELS}
CATEGORIES = ("vacuous_tests", "ignored_errors")


def relation(monitor: str, agent: str) -> str:
    return "self" if monitor == agent else "cross"


def load_tasks(limit: int | None = None) -> list[dict]:
    tasks = json.loads(TASKS.read_text())
    if limit:  # keep the category balance in pilots
        per = max(1, limit // len(CATEGORIES))
        tasks = [t for c in CATEGORIES for t in [x for x in tasks if x["category"] == c][:per]]
    return tasks


def tag(text: str, name: str) -> str | None:
    m = re.findall(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m[-1].strip() if m else None


def code_blocks(text: str) -> str:
    """All fenced code in a reply, concatenated (python fences or bare fences)."""
    blocks = re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", text, re.S)
    return "\n\n".join(blocks) if blocks else text


def cost(recs: list[dict]) -> float:
    return sum((r.get("usage") or {}).get("cost") or 0 for r in recs)
