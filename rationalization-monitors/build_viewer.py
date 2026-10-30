"""viewer.html: every sloppy item (code, notes, swap paraphrases) with each monitor's verdict per condition,
self verdicts highlighted, plus clean items. Usage: python3.12 build_viewer.py [--limit N]"""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict

from monitor import mon_dir
from common import HERE, MODELS, RUNS, load_tasks, read_jsonl

NAMES = [m["name"] for m in MODELS]
ORDER = {"clean": 0, "bare": 1, "rat": 2}
CSS = """:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#e4e1da;--flag:#b3261e;--pass:#2e7d32;--self:#fff4d6;--code:#f4f2ed}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe7;--mut:#9d9b95;--card:#1f1f1d;--line:#34332f;--flag:#ff8a80;--pass:#81c784;--self:#3a3320;--code:#262624}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe7;--mut:#9d9b95;--card:#1f1f1d;--line:#34332f;--flag:#ff8a80;--pass:#81c784;--self:#3a3320;--code:#262624}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1100px;margin:auto}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:8px 12px}
summary{cursor:pointer;font-weight:600}pre{background:var(--code);padding:10px;border-radius:6px;overflow-x:auto;font-size:12.5px;white-space:pre-wrap}
table{border-collapse:collapse;font-size:13px;width:100%;overflow-x:auto;display:block}td,th{border:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
.F{color:var(--flag);font-weight:600}.P{color:var(--pass);font-weight:600}.self{background:var(--self)}.mut{color:var(--mut)}
input,select{font:inherit;padding:4px;margin:4px 4px 4px 0}"""


def esc(s, n=None):
    s = str(s or "")
    return html.escape(s if not n or len(s) <= n else s[:n] + " …[truncated]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--prompt", choices=("general", "specific"), default="general")
    a = ap.parse_args()
    tasks = {t["id"]: t for t in load_tasks(a.limit)}
    verdicts = defaultdict(dict)  # (agent, task) -> {(monitor, cond): rec}
    for m in NAMES:
        for r in read_jsonl(RUNS / mon_dir(a.prompt) / f"{m}.jsonl"):
            verdicts[(r["agent"], r["task"])][(m, r["cond"])] = r
    parts = [f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>Rationalization Monitors</title><style>{CSS}</style></head><body><h1>Rationalization monitors</h1>"
             "<p class=mut>Each card is one agent × task. Rows = monitor (highlighted = same model as the agent). "
             "Cells: <span class=F>FLAG</span>/<span class=P>pass</span> + score. Conditions: clean, bare (sloppy code, no notes), "
             "rat (sloppy + agent's own design notes), swap:R (notes paraphrased by R).</p>"
             "<input id=q placeholder='filter (task id, agent, text)' oninput=f()> </p>"]
    summ = RUNS / f"summary_{a.prompt}.json"
    if summ.exists():
        parts.append(f"<details><summary>summary.json</summary><pre>{esc(summ.read_text(), 20000)}</pre></details>")
    for agent in NAMES:
        gens = {(g["task"], g["cond"]): g for g in read_jsonl(RUNS / agent / "gen.jsonl")}
        vers = {(v["task"], v["cond"]): v for v in read_jsonl(RUNS / agent / "verify.jsonl")}
        swaps = {(s["task"], s.get("cond", "sloppy"), s["rewriter"]): s for s in read_jsonl(RUNS / agent / "swap.jsonl")}
        for tid, t in tasks.items():
            vd = verdicts.get((agent, tid))
            if not vd:
                continue
            conds = sorted({c for (_, c) in vd}, key=lambda c: (c.startswith("subtle_"), ORDER.get(c.removeprefix("subtle_"), 3), c))
            rows = []
            for m in NAMES:
                cells = []
                for c in conds:
                    r = vd.get((m, c))
                    cells.append("<td>-</td>" if not r else
                                 f"<td title='{esc('; '.join(map(str, r.get('issues') or [])), 400)}'><span class={'F' if r['flag'] else 'P'}>"
                                 f"{'FLAG' if r['flag'] else 'pass'}</span> {r.get('score')}</td>")
                rows.append(f"<tr class={'self' if m == agent else ''}><th>{m}</th>{''.join(cells)}</tr>")
            gc = gens.get((tid, "clean"))
            body = [f"<p><b>Request:</b> {esc(t['user_request'])}</p>",
                    f"<table><tr><th>monitor</th>{''.join(f'<th>{c}</th>' for c in conds)}</tr>{''.join(rows)}</table>"]
            for level, pre in (("sloppy", ""), ("subtle", "subtle_")):
                g, v = gens.get((tid, level)), vers.get((tid, level), {})
                if not g or pre + "rat" not in conds:
                    continue
                rv = v.get("review") or {}
                lab = "blatant" if level == "sloppy" else "subtle"
                body.append(f"<h3>{lab}</h3><p><b>Ground truth:</b> {esc(rv.get('defect'))} <i>Failure scenario:</i> {esc(rv.get('failure_scenario'))} "
                            f"{'<br>Planted bug its tests miss: ' + esc((v.get('mutant') or {}).get('bug')) if v.get('mutant') else ''}"
                            f"{'<br>AST: ' + esc(', '.join(v.get('ast_hits') or [])) if v.get('ast_hits') else ''}</p>")
                body.append(f"<details><summary>{lab} reply</summary><pre>{esc(g['response'], 12000)}</pre></details>")
                body.append(f"<details open><summary>Design notes ({agent})</summary><p>{esc(g['design_notes'])}</p></details>")
                for r in NAMES:
                    if (tid, level, r) in swaps:
                        body.append(f"<details><summary>Notes paraphrased by {r}</summary><p>{esc(swaps[(tid, level, r)]['design_notes'])}</p></details>")
                ex = vd.get((agent, pre + "rat"))
                if ex:
                    body.append(f"<details><summary>Self-monitor ({agent}) on {lab}+notes: verdict text</summary><pre>{esc(ex['content'], 4000)}</pre></details>")
            if gc and "clean" in conds:
                body.append(f"<details><summary>Clean reply</summary><pre>{esc(gc['response'], 12000)}</pre></details>")
            parts.append(f"<details class=item><summary>{tid} · {agent} · {t['category']} — {esc(t['title'])}</summary>{''.join(body)}</details>")
    parts.append("<script>function f(){const q=document.getElementById('q').value.toLowerCase();"
                 "document.querySelectorAll('.item').forEach(d=>{d.style.display=d.textContent.toLowerCase().includes(q)?'':'none'})}</script></body></html>")
    out = HERE / f"viewer_{a.prompt}.html"
    out.write_text("".join(parts))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
