"""Ask GPT and Gemini which agent in the transcript used more tokens.

Usage:
    python ask_models.py ["question"] [results.json]   # keys from ../.env
Gemini is called through OpenRouter.
Override model IDs with OPENAI_MODEL / GEMINI_MODEL if the defaults are wrong.
"""
import json
import os
import pathlib
import ssl
import sys
import urllib.request

import certifi

# python.org builds of Python ship without a CA bundle.
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

HERE = pathlib.Path(__file__).resolve().parent
TRANSCRIPT = (HERE / "agents-working-token-count.md").read_text()
QUESTION = sys.argv[1] if len(sys.argv) > 1 else "in solving the task, which agent in this script used more tokens?"
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "ask_models_results.json"
PROMPT = f"{TRANSCRIPT}\n\n{QUESTION}"

# Pull keys from the project-root .env without overriding the shell.
for line in (HERE.parent / ".env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-astra")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "google/gemini-4-argon")


def post(url, headers, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(req, timeout=300, context=SSL_CTX) as resp:
        return json.load(resp)


def ask_gpt():
    data = post(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        {"model": OPENAI_MODEL, "messages": [{"role": "user", "content": PROMPT}]},
    )
    return data["choices"][0]["message"]["content"]


def ask_gemini():
    data = post(
        "https://openrouter.ai/api/v1/chat/completions",
        {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        {"model": GEMINI_MODEL, "messages": [{"role": "user", "content": PROMPT}]},
    )
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    results = {}
    for name, fn in [(OPENAI_MODEL, ask_gpt), (GEMINI_MODEL, ask_gemini)]:
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = f"ERROR: {e!r}"
        print(f"\n===== {name} =====\n{results[name]}")
    (HERE / OUT_NAME).write_text(json.dumps(results, indent=2))
