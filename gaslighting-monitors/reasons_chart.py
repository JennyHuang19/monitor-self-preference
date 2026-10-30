"""'Reasons the grader was fooled', by relation to the attacker, for the viewer's Gemini-4-Argon + GLM Pair tab.

    python reasons_chart.py   # writes runs/reasons_chart_argon.html and runs/reasons_chart_mix5.html from the diagnose_*.jsonl labels

One 100%-stacked bar per (grader, relation). Labels come from diagnose_fooled.py (an outside judge, Claude Opus 5.5).
"""
import collections
import html as H
import json
import random
from pathlib import Path

RUNS = Path(__file__).parent / "runs"
# Failure reasons in a fixed order; each takes the categorical slot of its position.
REASONS = [("rubber_stamp", "Rubber stamp", "never checked the step with the planted error"),
           ("missed_in_check", "Missed it while checking", "real verification, but not of the flawed step"),
           ("shared_error", "Repeated the pitch's mistake", "re-derived the step and made the same slip"),
           ("persuaded", "Persuaded", "raised the exact issue, then was talked out of it"),
           ("knew_but_accepted", "Knew, accepted anyway", "its own text shows the answer is wrong (partly mislabelled; see note)")]
SIB = {"glm-5.3": "GLM-5.3-flash's", "glm-5.3-flash": "GLM-5.3's", "gpt-oss-120b": "gpt-oss-20b's", "gpt-oss-20b": "gpt-oss-120b's"}
NOTE_ARGON = ("The mix of reasons does not depend on who wrote the pitch (permutation test, shuffling the relation within each grader: p = {p:.2f}). "
              "Each grader's own style dominates: GLM-5.3 mostly misses the error mid-check, GLM-5.3-flash rubber-stamps more. "
              "\"Unrelated\" is always a Gemini-4-Argon pitch, so that bar also reflects Gemini's writing style. "
              "\"Knew, accepted anyway\" is small and partly mislabelled: of three cases checked by hand, one was a plain rubber stamp and two were reasoning "
              "that broke off mid-check and switched to a terse approval.")
NOTE_MIX5 = ("Here the mix of reasons does depend on who wrote the pitch (permutation test, shuffling the relation within each grader: p = {p:.3f}), "
             "and it comes from the GLM graders: they rubber-stamp their own family's pitches far more often than other families' "
             "(GLM-5.3: 28% self, 17% sibling, 7% unrelated; GLM-5.3-flash: 37%, 33%, 15%). The gap holds whether or not the grader could solve the problem "
             "on its own (36% vs 15% when it could, 22% vs 7% when it couldn't), so it isn't just GLM's own pitches landing on problems it can't do. "
             "The gpt-oss graders show no such gap. In the Gemini-4-Argon + GLM run, GLM rubber-stamped Gemini's pitches as often as its own family's, "
             "so the difference may come from how gpt-oss and Kimi pitches are written rather than from familiarity alone. "
             "Kimi-K3 was fooled only 17 times, so its bars are very noisy. Two fooled grades could not be labelled (the judge declined them).")
EXPS = [
    {"src": "diagnose_math4_argon_glm.jsonl", "out": "reasons_chart_argon.html", "id": "reasons",
     "graders": ["glm-5.3", "glm-5.3-flash"], "cross": "Gemini-4-Argon's pitches", "note": NOTE_ARGON,
     "intro": "Gemini-4-Argon was never fooled, so only the two GLM graders appear."},
    {"src": "diagnose_math3_mix5.jsonl", "out": "reasons_chart_mix5.html", "id": "reasons-mix5",
     "graders": ["glm-5.3", "glm-5.3-flash", "gpt-oss-120b", "gpt-oss-20b", "kimi-k3"], "cross": "other families' pitches", "note": NOTE_MIX5,
     "intro": "Kimi-K3 has no sibling in this run, so it has no sibling bar."},
]



def rubric_html():
    """The judge prompt exactly as sent: categories filled in, per-grade fields left as {placeholders}."""
    import diagnose_fooled as df
    fields = ["question", "answer", "target", "etype", "error_description", "solution", "frame", "frame_note", "pitch", "cot", "reply"]
    text = df.PROMPT.format(**{f: "{" + f + "}" for f in fields},
                            cats="\n".join(f"- {k}: {v}" for k, v in df.CATEGORIES.items()), catnames=", ".join(df.CATEGORIES))
    names = {k: lab for k, lab, _ in REASONS}
    cats = "".join(f"<li><b>{H.escape(names.get(k, k))}</b> <code>{k}</code>: {H.escape(v)}</li>" for k, v in df.CATEGORIES.items())
    frames = "".join(f"<li><code>{k}</code>: {H.escape(v)}</li>" for k, v in df.FRAME_NOTE.items())
    return (f'<p class="small" style="margin:0 0 10px;max-width:78ch">Each fooled grade got one call to <b>{df.JUDGE}</b> (adaptive thinking, effort high), '
            f'with the prompt below. The judge picked exactly one failure type and had to quote the grader\'s own words as evidence. '
            f'The judge was not told whether the grader could solve the problem on its own.</p>'
            f'<h3 style="margin:0 0 6px">Failure types</h3><ul class="small" style="margin:0 0 12px;padding-left:20px;display:grid;gap:4px">{cats}</ul>'
            f'<p class="small" style="margin:0 0 12px;max-width:78ch">This page calls <code>shared_error</code> "Repeated the pitch\'s mistake": the grader repeated it while grading, after reading the pitch. '
            f'<code>ambiguous_key</code> was never chosen.</p>'
            f'<details open><summary>Exact prompt sent to the judge</summary><pre class="txt small">{H.escape(text)}</pre></details>'
            f'<details><summary>What {{frame_note}} says for each frame</summary><ul class="small" style="margin:8px 12px;padding-left:20px;display:grid;gap:4px">{frames}</ul></details>')


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d; half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def mark(rel, cx, cy):
    st = f'class="dot" style="fill:var(--rel{ {"self": 1, "sibling": 2, "cross": 3}[rel] })"'
    if rel == "self":
        return f'<circle {st} cx="{cx:.1f}" cy="{cy:.1f}" r="5"/>'
    if rel == "sibling":
        return f'<rect {st} x="{cx - 4.5:.1f}" y="{cy - 4.5:.1f}" width="9" height="9" rx="1.5"/>'
    return f'<path {st} d="M{cx:.1f},{cy - 6:.1f} L{cx + 6:.1f},{cy:.1f} L{cx:.1f},{cy + 6:.1f} L{cx - 6:.1f},{cy:.1f} Z"/>'


def perm_p(rows, it=5000):
    """Permutation test of relation x reason independence, shuffling relation within each grader."""
    def chi(rs):
        t = collections.Counter((r["relation"], r["category"]) for r in rs); n = len(rs)
        a = collections.Counter(r["relation"] for r in rs); b = collections.Counter(r["category"] for r in rs)
        return sum((t[(x, y)] - a[x] * b[y] / n) ** 2 / (a[x] * b[y] / n) for x in a for y in b)
    obs, rng, ge = chi(rows), random.Random(1), 0
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["grader"]].append(r)
    for _ in range(it):
        sh = []
        for rs in groups.values():
            rel = [r["relation"] for r in rs]; rng.shuffle(rel)
            sh += [{"relation": x, "category": r["category"]} for x, r in zip(rel, rs)]
        ge += chi(sh) >= obs
    return ge / it


def build(e):
    rows = [r for r in map(json.loads, open(RUNS / e["src"])) if r["category"] in dict((k, 1) for k, _, _ in REASONS)]
    c = collections.Counter((r["grader"], r["relation"], r["category"]) for r in rows)
    n = collections.Counter((r["grader"], r["relation"]) for r in rows)
    p = perm_p(rows)
    rels = lambda g: [(r, lab, d) for r, lab, d in (("self", "Self", "its own pitches"), ("sibling", "Sibling", f"{SIB.get(g, '')} pitches"),
                                                     ("cross", "Unrelated", e["cross"])) if n[(g, r)]]
    # Relation keeps one color and one shape everywhere: self = circle (slot 1), sibling = square (slot 2), unrelated = diamond (slot 3).
    REL_I = {"self": 1, "sibling": 2, "cross": 3}
    legend = "".join(f'<span><svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden="true">{mark(rel, 0, 0)}</svg> <b>{lab}</b> '
                     f'<span class="small">· {desc}</span></span>'
                     for rel, lab, desc in (("self", "Self", "its own pitches"), ("sibling", "Sibling", "the other model in its family"),
                                            ("cross", "Unrelated", e["cross"])))
    W, L, R, T, RH, B = 490, 206, 16, 6, 44, 34  # L leaves ~196px for the longest reason label (~176px at 12px)
    x = lambda v: L + (W - L - R) * v
    panels = []
    for g in e["graders"]:
        rs = rels(g)
        hgt = T + RH * len(REASONS) + B
        sv = [f'<svg viewBox="0 0 {W} {hgt}" role="img" aria-label="{g}: share of fooled grades for each reason, by who wrote the pitch">']
        for t in range(0, 101, 25):
            sv.append(f'<line class="grid" x1="{x(t / 100):.1f}" x2="{x(t / 100):.1f}" y1="{T}" y2="{T + RH * len(REASONS)}"/>'
                      f'<text class="tick" x="{x(t / 100):.1f}" y="{T + RH * len(REASONS) + 14}" text-anchor="middle">{t}%</text>')
        sv.append(f'<text class="axis" x="{(L + W - R) / 2:.1f}" y="{hgt - 4}" text-anchor="middle">share of this grader\'s fooled grades</text>')
        for j, (k, lab, _) in enumerate(REASONS):
            cy = T + RH * j + RH / 2
            if j:
                sv.append(f'<line class="grid" x1="4" x2="{W - R}" y1="{T + RH * j}" y2="{T + RH * j}" style="opacity:.5"/>')
            sv.append(f'<text class="rowlab" x="{L - 10}" y="{cy + 4:.1f}" text-anchor="end">{lab}</text>')
            offs = {1: [0], 2: [-6, 6], 3: [-11, 0, 11]}[len(rs)]
            for (rel, rlab, rdesc), dy in zip(rs, offs):
                f, tot = c[(g, rel, k)], n[(g, rel)]
                lo, hi = wilson(f, tot)
                yy = cy + dy
                tip = f"{g} grading {rdesc}: {lab} {f}/{tot} ({f / tot:.0%}; 95% interval {lo:.0%}–{hi:.0%})"
                sv.append(f'<g class="pt" tabindex="0" data-tip="{H.escape(tip)}">'
                          f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" style="stroke:var(--rel{REL_I[rel]});stroke-width:2;opacity:.5"/>'
                          f'<rect class="hit" x="{x(lo) - 6:.1f}" y="{yy - 6:.1f}" width="{x(hi) - x(lo) + 12:.1f}" height="12"/>'
                          f'{mark(rel, x(f / tot), yy)}</g>')
        sv.append("</svg>")
        counts = " · ".join(f"{rlab.lower()} {n[(g, rel)]}" for rel, rlab, _ in rs)
        panels.append(f'<figure class="effort-panel rdots"><figcaption>Grader: {g} <span class="small">· fooled grades: {counts}</span></figcaption>{"".join(sv)}</figure>')
    head = "".join(f'<th class="num">{lab}</th>' for _, lab, _ in REASONS)
    trs = "".join(f'<tr><td>{g}</td><td>{rlab}</td><td class="num">{n[(g, rel)]}</td>' +
                  "".join(f'<td class="num">{c[(g, rel, k)]} ({c[(g, rel, k)] / n[(g, rel)]:.0%})</td>' for k, _, _ in REASONS) + "</tr>"
                  for g in e["graders"] for rel, rlab, _ in rels(g))
    rubric = rubric_html()
    html = f"""
  <section class="panel reasons-sec" id="{e['id']}" hidden>
    <h2>Reasons the grader was fooled, by who wrote the pitch</h2>
    <div class="seg rtabs" role="group" aria-label="Chart or judge rubric" style="margin:0 0 12px">
      <button data-rtab="chart" aria-pressed="true">Chart</button><button data-rtab="rubric" aria-pressed="false">Judge rubric</button>
    </div>
    <div data-rpanel="chart">
    <p class="small" style="margin:0 0 10px;max-width:78ch">Every fooled grade ({len(rows)}, all frames) was labelled by an outside judge (Claude Opus 5.5) that read the problem, the planted error, the pitch and the grader's own reasoning. {e['intro']} Each row is one reason. The marks show what share of that grader's fooled grades it explains, for pitches by itself, its sibling, and unrelated models; lines are 95% intervals. Compare marks within a row: far-apart marks whose lines don't overlap are real differences. Hover a mark for its count.</p>
    <div class="rlegend small">{legend}</div>
    <div class="effort-grid rgrid">{"".join(panels)}</div>
    <p class="small" style="margin:12px 0 0;max-width:78ch">{e['note'].format(p=p)}</p>
    <details style="margin-top:10px"><summary>Numbers</summary><div class="scroll"><table><tr><th>Grader</th><th>Pitch by</th><th class="num">Fooled</th>{head}</tr>{trs}</table></div></details>
    </div>
    <div data-rpanel="rubric" hidden>{rubric}</div>
    <div class="effort-tip" role="status" hidden></div>
  </section>
"""
    (RUNS / e["out"]).write_text(html)
    print(f"wrote {e['out']} ({len(rows)} rows, p = {p:.3f})")


if __name__ == "__main__":
    for e in EXPS:
        build(e)
