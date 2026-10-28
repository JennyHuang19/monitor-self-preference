"""Bundle every results/*.jsonl into a self-contained viewer.html.

Usage:
    .venv/bin/python build_viewer.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

records = []
for path in sorted((HERE / "results").glob("*.jsonl")):
    run = path.stem  # "run-<ts>" for baselines, "debate-<ts>" for debates
    for line in path.read_text().splitlines():
        if line.strip():
            records.append({**json.loads(line), "run": run})

# Escape "</" so an answer can't close the <script> tag that holds the data.
data = json.dumps(records).replace("</", "<\\/")
html = (HERE / "viewer_template.html").read_text().replace("__DATA__", data)
(HERE / "viewer.html").write_text(html)
print(f"Wrote viewer.html with {len(records)} responses")
