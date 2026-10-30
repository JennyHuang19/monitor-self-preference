"""Fooled rate vs grader reasoning length, for the viewer's Experiment Notes tab.

    python effort_chart.py   # writes runs/effort_chart.html (read by page_patch.add_notes)

Neutral + self-check grades pooled (the skim frame tells graders not to solve, so it is left out).
Reasoning length = the provider's reasoning_tokens. A point is drawn only when its bin has >= MIN_N grades.
"""
import collections
import glob
import json
from pathlib import Path

RUNS = Path(__file__).parent / "runs"
EXPS = [("math4_argon_glm", "Gemini-4-Argon + GLM Pair"), ("math3_mix5", "5 open models")]
# Color follows the model across both panels (categorical slots in fixed order).
SLOT = {"glm-5.3": 1, "glm-5.3-flash": 2, "gpt-oss-120b": 3, "gpt-oss-20b": 4, "kimi-k3": 5, "gemini-4-argon": 6}
BINS = [(0, 50, "0–49"), (50, 100, "50–99"), (100, 200, "100–199"), (200, 400, "200–399"),
        (400, 800, "400–799"), (800, 1600, "800–1.6k"), (1600, 10**9, "1.6k+")]
MIN_N = 10
W, H, L, R, T, B = 420, 250, 40, 12, 14, 46


def rt(r):
    return ((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0


def load(prefix):
    c = collections.Counter()
    for f in glob.glob(str(RUNS / f"{prefix}__*/grades*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["frame"] not in ("neutral", "self"):
                continue
            i = next(i for i, (lo, hi, _) in enumerate(BINS) if lo <= rt(r) < hi)
            c[(r["grader"], i, "n")] += 1
            c[(r["grader"], i, "f")] += bool(r["fooled"])
    return c


def panel(title, c, ymax):
    graders = sorted({g for g, _, _ in c}, key=SLOT.get)
    pw, ph = W - L - R, H - T - B
    x = lambda i: L + pw * (i + 0.5) / len(BINS)
    y = lambda v: T + ph * (1 - v / ymax)
    s = [f'<figure class="effort-panel"><figcaption>{title}</figcaption>'
         f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}: fooled rate by grader reasoning length">']
    for t in range(0, int(ymax * 100) + 1, 10):
        v = t / 100
        s.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                 f'<text class="tick" x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{t}%</text>')
    for i, (_, _, lab) in enumerate(BINS):
        s.append(f'<text class="tick" x="{x(i):.1f}" y="{H - B + 16}" text-anchor="middle">{lab}</text>')
    s.append(f'<text class="axis" x="{L + pw / 2:.1f}" y="{H - 8}" text-anchor="middle">reasoning tokens</text>')
    for g in graders:
        pts = [(i, c[(g, i, "f")], c[(g, i, "n")]) for i in range(len(BINS)) if c[(g, i, "n")] >= MIN_N]
        k = SLOT[g]
        if len(pts) > 1:
            d = " ".join(f'{"M" if j == 0 else "L"}{x(i):.1f},{y(f / n):.1f}' for j, (i, f, n) in enumerate(pts))
            s.append(f'<path class="ln" style="stroke:var(--s{k})" d="{d}"/>')
        for i, f, n in pts:
            tip = f"{g} · {BINS[i][2]} tokens · fooled {f}/{n} ({f / n:.0%})"
            s.append(f'<g class="pt" tabindex="0" data-tip="{tip}"><circle class="hit" cx="{x(i):.1f}" cy="{y(f / n):.1f}" r="11"/>'
                     f'<circle class="dot" style="fill:var(--s{k})" cx="{x(i):.1f}" cy="{y(f / n):.1f}" r="4"/></g>')
    s.append("</svg></figure>")
    return "".join(s)


def table(data):
    head = "".join(f'<th class="num">{lab}</th>' for _, _, lab in BINS)
    rows = []
    for title, c in data:
        rows.append(f'<tr><td colspan="{len(BINS) + 1}" class="label">{title}</td></tr>')
        for g in sorted({g for g, _, _ in c}, key=SLOT.get):
            cells = "".join(f'<td class="num">{c[(g, i, "f")]}/{c[(g, i, "n")]}</td>' if c[(g, i, "n")] else '<td class="num">—</td>'
                            for i in range(len(BINS)))
            rows.append(f"<tr><td>{g}</td>{cells}</tr>")
    return (f'<details><summary>Numbers (fooled / grades per bin)</summary><div class="scroll"><table>'
            f'<tr><th>Grader</th>{head}</tr>{"".join(rows)}</table></div></details>')


def main():
    data = [(title, load(prefix)) for prefix, title in EXPS]
    rates = [c[(g, i, "f")] / c[(g, i, "n")] for _, c in data for (g, i, k) in c if k == "n" and c[(g, i, "n")] >= MIN_N]
    ymax = max(0.1, -(-max(rates) * 10 // 1) / 10)  # round up to the next 10%
    models = sorted({g for _, c in data for g, _, _ in c}, key=SLOT.get)
    legend = "".join(f'<span><span class="sw" style="background:var(--s{SLOT[g]})"></span>{g}</span>' for g in models)
    html = (f'<h4>Fooled rate by reasoning length</h4>'
            f'<p class="small" style="margin:0">Neutral and self-check grades pooled. Each point is one grader in one reasoning-length bin; '
            f'bins with fewer than {MIN_N} grades are not drawn. Hover a point for its counts.</p>'
            f'<div class="effort-legend small">{legend}</div>'
            f'<div class="effort-grid">{"".join(panel(t, c, ymax) for t, c in data)}</div>'
            f'<div class="effort-tip" role="status" hidden></div>{table(data)}')
    (RUNS / "effort_chart.html").write_text(html)
    print(f"wrote {RUNS / 'effort_chart.html'} ({len(html)} chars), y max {ymax:.0%}")


if __name__ == "__main__":
    main()
