"""Direct Anthropic / OpenAI clients with the same call shape and return dict as gm.chat (OpenRouter).
Keys come from the environment or ../.env (ANTHROPIC_API_KEY, OPENAI_API_KEY). Reasoning is on at high effort;
both APIs return only a summary of the hidden reasoning, stored under "reasoning"."""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent / ".env"


def _key(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in ENV.read_text().splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"\'')
    raise RuntimeError(f"{name} not set and not in {ENV}")


def _post(url: str, body: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:500]!r}")


def _anthropic(model: str, messages: list[dict], max_tokens: int) -> dict:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    body = {"model": model, "max_tokens": max_tokens, "messages": [m for m in messages if m["role"] != "system"],
            "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "high"}}
    if system:
        body["system"] = system
    j = _post("https://api.anthropic.com/v1/messages", body,
              {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"})
    blocks = j.get("content") or []
    return {"content": "".join(b.get("text", "") for b in blocks if b["type"] == "text"),
            "reasoning": "\n".join(b.get("thinking", "") for b in blocks if b["type"] == "thinking"),
            "finish_reason": j.get("stop_reason"), "provider": "Anthropic (direct)",
            "usage": j.get("usage", {}), "model": j.get("model")}


def _openai(model: str, messages: list[dict], max_tokens: int) -> dict:
    body = {"model": model, "input": messages, "max_output_tokens": max_tokens,
            "reasoning": {"effort": "high", "summary": "detailed"}}
    j = _post("https://api.openai.com/v1/responses", body, {"Authorization": f"Bearer {_key('OPENAI_API_KEY')}"})
    out = j.get("output") or []
    text = "".join(c.get("text", "") for o in out if o["type"] == "message" for c in o.get("content", [])
                   if c.get("type") == "output_text")
    summ = "\n".join(s.get("text", "") for o in out if o["type"] == "reasoning" for s in o.get("summary") or [])
    return {"content": text, "reasoning": summ, "finish_reason": j.get("status"),
            "incomplete": (j.get("incomplete_details") or {}).get("reason"), "provider": "OpenAI (direct)",
            "usage": j.get("usage", {}), "model": j.get("model")}


def chat(api: str, model: str, messages: list[dict], *, max_tokens: int = 32000, retries: int = 8) -> dict:
    fn = {"anthropic": _anthropic, "openai": _openai}[api]
    err = None
    for attempt in range(retries):
        try:
            r = fn(model, messages, max_tokens)
            if not r["content"].strip():
                raise RuntimeError(f"empty content (finish={r['finish_reason']})")
            return r
        except Exception as e:  # network errors, 429/529s
            err = e
            print(f"  retry {attempt + 1} {model}: {str(e)[:200]}", flush=True)
            time.sleep(min(120, 2 ** attempt + random.random() * 2))
    raise RuntimeError(f"failed after {retries}: {err}")
