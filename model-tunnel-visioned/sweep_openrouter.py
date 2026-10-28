"""Ask every question to every model on OpenRouter and save the answers.

Usage:
    python sweep_openrouter.py   # key from ../.env
"""
import json
import os
import pathlib
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import certifi

SSL_CTX = ssl.create_default_context(cafile=certifi.where())
HERE = pathlib.Path(__file__).resolve().parent
TRANSCRIPT = (HERE / "agents-working-token-count.md").read_text()

for line in (HERE.parent / ".env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

MODELS = [
    "openai/gpt-6.1-sol",
    "x-ai/grok-4.7",
    "moonshotai/kimi-k3.1-pro",
    "qwen/qwen3.8-max-prime",
    "z-ai/glm-5.3-prime",
    "mistralai/mistral-large-4-0",
]
QUESTIONS = {
    "tokens": "in solving the task, which agent in this script used more tokens?",
    "performance": "how well does each agent do in performing the task?",
    "agents_where": "Which agents were involved in performing the task and where did they perform their tasks?",
    "collaboration": "in what ways (if any) did the agents work together?",
}


def ask(model, question):
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({"model": model, "messages": [{"role": "user", "content": f"{TRANSCRIPT}\n\n{question}"}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600, context=SSL_CTX) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e!r}"


if __name__ == "__main__":
    jobs = [(m, q) for m in MODELS for q in QUESTIONS]
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda job: ask(job[0], QUESTIONS[job[1]]), jobs))
    results = {m: {} for m in MODELS}
    for (m, q), a in zip(jobs, answers):
        results[m][q] = a
    (HERE / "sweep_openrouter_results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {len(jobs)} answers to sweep_openrouter_results.json")
