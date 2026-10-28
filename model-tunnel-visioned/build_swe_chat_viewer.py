"""Build a self-contained HTML viewer for a handful of SWE-chat sessions.

Usage:
    python3 build_swe_chat_viewer.py                      # random sample (default 8)
    python3 build_swe_chat_viewer.py --n 12 --seed 3
    python3 build_swe_chat_viewer.py --ids <session_id> <session_id> ...
    python3 build_swe_chat_viewer.py --agents "Claude Code" Codex

Writes swe_chat_viewer.html next to this script.
"""

import argparse
import json
import os
import random

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "SWE-chat")
OUT = os.path.join(HERE, "swe_chat_viewer.html")
MAX_TEXT = 20_000  # per-turn cap so the HTML stays a reasonable size
MAX_FILE_BYTES = 3_000_000

META_COLS = [
    "session_id", "agent", "model", "repo_id", "user_id", "branch", "created_at",
    "duration_seconds", "prompt_count", "turn_count", "tool_call_count",
    "input_tokens", "output_tokens", "cache_read_tokens", "files_touched_count",
    "agent_percentage", "user_persona", "session_success", "summary_intent",
    "summary_outcome", "summary_friction", "summary_learnings", "viewer_url_hf",
]


def clip(s):
    s = s if isinstance(s, str) else json.dumps(s, indent=2, ensure_ascii=False)
    if len(s) > MAX_TEXT:
        return s[:MAX_TEXT] + f"\n\n… [truncated {len(s) - MAX_TEXT:,} chars]"
    return s


def turn(kind, text, ts=None, name=None, sidechain=False):
    return {"kind": kind, "text": clip(text), "ts": ts, "name": name, "side": sidechain}


def blocks_text(content):
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict):
            if b.get("type") == "text":
                out.append(b.get("text", ""))
            elif b.get("type") == "image":
                out.append("[image]")
            else:
                out.append(json.dumps(b, ensure_ascii=False))
        else:
            out.append(str(b))
    return "\n".join(out)


def parse_claude_code(rows):
    turns = []
    for o in rows:
        t, ts, side = o.get("type"), o.get("timestamp"), bool(o.get("isSidechain"))
        msg = o.get("message") or {}
        if t == "user":
            content = msg.get("content")
            if isinstance(content, str):
                meta = o.get("isMeta") or content.lstrip().startswith(("<command-", "<local-command", "<system-reminder"))
                turns.append(turn("system" if meta else "user", content, ts, sidechain=side))
                continue
            for b in content or []:
                if b.get("type") == "tool_result":
                    txt = blocks_text(b.get("content"))
                    turns.append(turn("tool_result", txt, ts, "error" if b.get("is_error") else None, side))
                elif b.get("type") == "text":
                    turns.append(turn("user", b.get("text", ""), ts, sidechain=side))
                elif b.get("type") == "image":
                    turns.append(turn("user", "[image]", ts, sidechain=side))
        elif t == "assistant":
            for b in msg.get("content") or []:
                bt = b.get("type")
                if bt == "text":
                    turns.append(turn("assistant", b.get("text", ""), ts, sidechain=side))
                elif bt in ("thinking", "redacted_thinking"):
                    turns.append(turn("thinking", b.get("thinking") or "[redacted thinking]", ts, sidechain=side))
                elif bt == "tool_use":
                    turns.append(turn("tool_call", b.get("input"), ts, b.get("name"), side))
        elif t == "system" and o.get("content"):
            turns.append(turn("event", o["content"], ts, o.get("subtype")))
        elif t == "summary" and o.get("summary"):
            turns.append(turn("event", o["summary"], ts, "summary"))
    return turns


def parse_codex(rows):
    turns = []
    for o in rows:
        t, ts, p = o.get("type"), o.get("timestamp"), o.get("payload") or {}
        pt = p.get("type")
        if t == "event_msg":
            if pt == "user_message":
                turns.append(turn("user", p.get("message", ""), ts))
            elif pt in ("turn_aborted", "context_compacted"):
                turns.append(turn("event", p.get("reason", pt), ts, pt))
        elif t == "response_item":
            if pt == "message" and p.get("role") == "assistant":
                turns.append(turn("assistant", blocks_text(
                    [{"type": "text", "text": c.get("text", "")} for c in p.get("content") or []]), ts))
            elif pt == "reasoning":
                summ = "\n".join(s.get("text", "") for s in p.get("summary") or [])
                if summ.strip():
                    turns.append(turn("thinking", summ, ts))
            elif pt in ("function_call", "custom_tool_call"):
                args = p.get("arguments") or p.get("input") or ""
                try:
                    args = json.loads(args)
                except (TypeError, ValueError):
                    pass
                turns.append(turn("tool_call", args, ts, p.get("name")))
            elif pt in ("function_call_output", "custom_tool_call_output"):
                turns.append(turn("tool_result", p.get("output", ""), ts))
            elif pt == "web_search_call":
                turns.append(turn("tool_call", p.get("action"), ts, "web_search"))
    return turns


def parse_raw(rows):
    # Unknown agent format: show each event compactly so it's still browsable.
    return [turn("event", o, o.get("timestamp") if isinstance(o, dict) else None,
                 str(o.get("type")) if isinstance(o, dict) else None) for o in rows]


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def parse(agent, rows):
    if any(isinstance(r, dict) and r.get("type") in ("assistant", "user") and "message" in r for r in rows[:200]):
        return "claude_code", parse_claude_code(rows)
    if any(isinstance(r, dict) and r.get("type") == "response_item" for r in rows[:200]):
        return "codex", parse_codex(rows)
    return "raw", parse_raw(rows)


def jsonable(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--agents", nargs="*", default=["Claude Code", "Codex"])
    args = ap.parse_args()

    s = pd.read_parquet(os.path.join(DATA, "sessions.parquet"))
    s["path"] = s.transcript_path.fillna("").map(lambda p: os.path.join(DATA, p) if p else "")
    s = s[s.path.map(lambda p: bool(p) and os.path.isfile(p))]

    if args.ids:
        picked = s[s.session_id.isin(args.ids)]
    else:
        pool = s[s.agent.isin(args.agents) & s.prompt_count.between(2, 25)]
        pool = pool[pool.path.map(os.path.getsize) < MAX_FILE_BYTES]
        rng = random.Random(args.seed)
        per = max(1, args.n // max(1, len(args.agents)))
        picks = []
        for a in args.agents:
            ids = sorted(pool[pool.agent == a].session_id)
            picks += rng.sample(ids, min(per, len(ids)))
        rest = sorted(set(pool.session_id) - set(picks))
        picks += rng.sample(rest, max(0, min(args.n - len(picks), len(rest))))
        picked = s[s.session_id.isin(picks)]

    sessions = []
    for _, r in picked.iterrows():
        fmt, turns = parse(r.agent, load_rows(r.path))
        meta = {c: jsonable(r[c]) for c in META_COLS if c in r}
        meta["format"] = fmt
        sessions.append({"meta": meta, "turns": turns})
        print(f"{r.agent:12} {fmt:12} {len(turns):5} turns  {r.session_id}")

    with open(os.path.join(HERE, "swe_chat_viewer_template.html")) as f:
        html = f.read()
    payload = json.dumps(sessions, ensure_ascii=False).replace("</", "<\\/")
    with open(OUT, "w") as f:
        f.write(html.replace("__SESSIONS_JSON__", payload))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
