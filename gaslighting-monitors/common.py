"""Shared OpenRouter client, model table and JSONL helpers (adapted from models-fill-in-gaps/tidy-stories/common.py)."""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)

QUESTION_WRITER = "anthropic/claude-fable-5.1"
GENERATOR = {"name": "glm-5.3", "id": "z-ai/glm-5.3", "family": "GLM"}
GRADERS = [
    {"name": "glm-5.3", "id": "z-ai/glm-5.3", "family": "GLM"},
    # Sibling of the GLM attacker: same family, different model. Only run on GLM's math pitches so far.
    {"name": "glm-5.3-flash", "id": "z-ai/glm-5.3-flash", "family": "GLM"},
    {"name": "kimi-k3", "id": "moonshotai/kimi-k3", "family": "Kimi"},
    {"name": "qwen3.8-max", "id": "qwen/qwen3.8-max-0902", "family": "Qwen"},
    {"name": "gpt-6.1-terra", "id": "openai/gpt-6.1-terra", "family": "GPT"},
]
N_SAMPLES = 3
# Weaker open-weight models: two families, each a sibling pair. Every model both attacks and grades.
OPEN_MODELS = [
    {"name": "gpt-oss-120b", "id": "openai/gpt-oss-120b", "family": "gpt-oss"},
    {"name": "gpt-oss-20b", "id": "openai/gpt-oss-20b", "family": "gpt-oss"},
    {"name": "gemma-4-31b", "id": "google/gemma-4-31b-it", "family": "Gemma"},
    {"name": "gemma-4-26b-a4b", "id": "google/gemma-4-26b-a4b-it", "family": "Gemma"},
]
# math3/trivia3 pool: two sibling pairs (GLM, gpt-oss) plus Kimi. Every model both attacks and grades.
MIX5 = [g for g in GRADERS if g["name"] in ("glm-5.3", "glm-5.3-flash", "kimi-k3")] + OPEN_MODELS[:2]
POOLS = {"frontier": GRADERS, "open": OPEN_MODELS, "mix5": MIX5}


def _load_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = HERE.parent / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"\'')
    raise RuntimeError("OPENROUTER_API_KEY not set and not in ../.env")


_key = _load_key()
_lock = threading.Lock()


def chat(model_id: str, messages: list[dict], *, reasoning: bool = True, max_tokens: int = 32000,
         temperature: float | None = None, json_mode: bool = False, retries: int = 6,
         reasoning_budget: int | None = None) -> dict:
    # reasoning_budget caps hidden thinking so it can't eat the whole max_tokens and leave no answer.
    reason = {"max_tokens": reasoning_budget} if reasoning_budget else {"enabled": reasoning}
    body: dict = {"model": model_id, "messages": messages, "max_tokens": max_tokens,
                  "reasoning": reason, "usage": {"include": True}}
    if temperature is not None:
        body["temperature"] = temperature
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {_key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    j = json.loads(r.read())
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:500]!r}")
            if "choices" not in j:
                raise RuntimeError(str(j)[:500])
            ch = j["choices"][0]
            msg = ch["message"]
            content = msg.get("content") or ""
            if not content.strip():
                raise RuntimeError(f"empty content (finish={ch.get('finish_reason')})")
            return {"content": content, "reasoning": msg.get("reasoning") or "",
                    "finish_reason": ch.get("finish_reason"), "provider": j.get("provider"),
                    "usage": j.get("usage", {}), "model": j.get("model")}
        except Exception as e:  # network errors, 429s, provider errors
            err = e
            print(f"  retry {attempt + 1} {model_id}: {str(e)[:200]}", flush=True)
            time.sleep(min(60, 2 ** attempt + random.random() * 2))
    raise RuntimeError(f"failed after {retries}: {err}")


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["key"] for l in path.read_text().splitlines() if l.strip()}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def append(path: Path, rec: dict) -> None:
    with _lock, path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run_jobs(jobs: list[dict], fn, out: Path, workers: int = 8) -> None:
    """jobs: dicts with a unique 'key'. fn(job) -> record dict. Skips keys already in out."""
    have = done_keys(out)
    todo = [j for j in jobs if j["key"] not in have]
    print(f"{out.name}: {len(todo)} to run ({len(have)} cached)", flush=True)
    n_ok = n_err = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, j): j for j in todo}
        for f in as_completed(futs):
            j = futs[f]
            try:
                rec = f.result()
                rec["key"] = j["key"]
                append(out, rec)
                n_ok += 1
                print(f"  ok {j['key']}", flush=True)
            except Exception as e:
                n_err += 1
                print(f"ERR {j['key']}: {str(e)[:200]}", flush=True)
    print(f"done: {n_ok} ok, {n_err} err", flush=True)


def parse_json(text: str, first: bool = False) -> dict:
    """Return the last JSON object in text (replies often contain other braces first: LaTeX, code, sets).
    first=True returns the first one instead, for replies that are one big object with nested objects."""
    dec = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for m in (starts if first else reversed(starts)):
        try:
            obj, _ = dec.raw_decode(text, m)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj:
            return obj
    raise ValueError("no JSON object found")


DOMAINS = ["math", "code", "trivia", "math2", "math3", "trivia3"]
DEFAULT_SAMPLES = {"math": 3, "code": 1, "trivia": 1, "math2": 1, "math3": 1, "trivia3": 1}
# Free-response domains: a claimed answer instead of a lettered option. Numeric ones are compared as integers.
FREE_RESPONSE = {"math2", "math3", "trivia3"}
NUMERIC_DOMAINS = {"math2", "math3"}


def norm_answer(x) -> str:
    """Loose form of a short answer for matching: lowercase, no markdown/punctuation/leading 'the', and a number
    reduced to its digits ('1,799.' -> '1799', 'about 4,810 m' -> '4810')."""
    s = re.sub(r"[*_`$\\]", "", str(x or "")).strip().lower()
    num = re.fullmatch(r"[^\d-]{0,12}?(-?[\d,]+(?:\.\d+)?)\s*[a-z%.]{0,12}\.?", s)
    if num:
        return num.group(1).replace(",", "").removesuffix(".0")
    s = re.sub(r"[^a-z0-9 ]+", " ", s.replace("&", " and "))
    return re.sub(r"^the ", "", re.sub(r"\s+", " ", s).strip())


def answer_matches(x, keys) -> bool:
    return x is not None and norm_answer(x) in {norm_answer(k) for k in keys}


def domain_args(desc: str):
    import argparse
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--domain", choices=DOMAINS, default="math")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--pool", choices=list(POOLS), default="frontier", help="which set of models attacks and grades")
    ap.add_argument("--attacker", choices=[g["name"] for p in POOLS.values() for g in p], default=None,
                    help="default: glm-5.3 for the frontier pool, the first open model for the open pool")
    ap.add_argument("--baseline-only", action="store_true", help="grade.py: only run the no-pitch baseline")
    ap.add_argument("--frame", choices=["neutral", "self", "skim", "skim1k"], default="neutral",
                    help="grade.py: neutral = judge a proposed answer; self = the pitch is shown as the grader's own earlier reply")
    ap.add_argument("--limit", type=int, default=None, help="generate/grade: only the first N items (pilots)")
    ap.add_argument("--batches", default=None, help="make_questions.py, math3/trivia3: comma-separated batch numbers")
    ap.add_argument("--reparse", action="store_true", help="grade.py: re-parse stored replies, no model calls")
    a = ap.parse_args()
    if a.samples is None:
        a.samples = DEFAULT_SAMPLES[a.domain]
    a.attacker = attacker(a.attacker or (GENERATOR["name"] if a.pool == "frontier" else OPEN_MODELS[0]["name"]), a.pool)
    return a


def pool_models(pool: str = "frontier") -> list[dict]:
    return POOLS[pool]


def attacker(name: str, pool: str = "frontier") -> dict:
    return next(g for g in POOLS[pool] if g["name"] == name)


def _dir_name(domain: str, attacker_name: str, pool: str) -> str:
    if pool != "frontier":
        return f"{domain}_{pool}__{attacker_name}"
    return domain if attacker_name == GENERATOR["name"] else f"{domain}__{attacker_name}"


def run_dir(domain: str, attacker_name: str = GENERATOR["name"], pool: str = "frontier") -> Path:
    """Frontier: GLM (the original attacker) keeps runs/<domain>/, others get runs/<domain>__<name>/.
    Other pools: runs/<domain>_<pool>__<name>/."""
    d = RUNS / _dir_name(domain, attacker_name, pool)
    d.mkdir(exist_ok=True)
    return d


def baseline_dir(domain: str, pool: str) -> Path:
    """Where the shared no-pitch baseline lives: the GLM run for frontier, a dedicated dir for the open pool."""
    if pool == "frontier":
        return run_dir(domain)
    return run_dir(domain, "baseline", pool)


def run_dirs(domain: str, pool: str = "frontier") -> list[tuple[dict, Path]]:
    """Every attacker in the pool that has a graded run for this domain (frontier: GLM first)."""
    models = POOLS[pool]
    if pool == "frontier":
        models = [GENERATOR] + [g for g in models if g["name"] != GENERATOR["name"]]
    out = []
    for g in models:
        d = RUNS / _dir_name(domain, g["name"], pool)
        if (d / "grades.jsonl").exists():
            out.append((attacker(g["name"], pool), d))
    return out


def load_questions(domain: str = "math", limit: int | None = None) -> list[dict]:
    return json.loads((HERE / f"questions_{domain}.json").read_text())[:limit]


def format_question(q: dict) -> str:
    if "answer" in q:  # math2/math3/trivia3: free response, no options
        return q["question"]
    if "options" not in q:  # code item
        return f"Specification:\n{q['spec']}\n\n```python\n{q['buggy_code']}\n```"
    opts = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
    return f"{q['question']}\n\n{opts}"
