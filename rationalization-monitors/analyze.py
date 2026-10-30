"""Self-leniency analysis (pure Python: the repo env has no numpy). Usage: python3.12 analyze.py [--limit N] [--boot 300]
Primary: on sloppy+rationale items, flag ~ monitor FE + agent FE + category + self. self < 0 means monitors flag their
own model's sloppy code less often than an additive monitor-strictness x agent-sloppiness model predicts."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict

from monitor import mon_dir
from common import MODELS, RUNS, cost, load_tasks, read_jsonl

NAMES = [m["name"] for m in MODELS]


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            raise ZeroDivisionError
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit(X, y, logistic=True, ridge=1e-2):
    """IRLS logistic (small ridge for stability under near-separation) or OLS. Returns coefficient list."""
    p = len(X[0])
    beta = [0.0] * p
    for _ in range(30 if logistic else 1):
        H = [[0.0] * p for _ in range(p)]
        g = [0.0] * p
        for xi, yi in zip(X, y):
            if logistic:
                eta = sum(a * b for a, b in zip(xi, beta))
                mu = 1 / (1 + math.exp(-max(-30, min(30, eta))))
                w, r = mu * (1 - mu), yi - mu
            else:
                w, r = 1.0, yi
            for j in range(p):
                if xi[j]:
                    g[j] += xi[j] * r
                    for k in range(p):
                        H[j][k] += w * xi[j] * xi[k]
        for j in range(1, p):
            H[j][j] += ridge
            if logistic:
                g[j] -= ridge * beta[j]
        if not logistic:
            return solve(H, g)
        step = solve(H, g)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-6:
            break
    return beta


def design(rows, extra):
    """Intercept, monitor and agent fixed effects (first level dropped), category, then extra(row) columns."""
    cols = ["intercept"] + [f"M={n}" for n in NAMES[1:]] + [f"A={n}" for n in NAMES[1:]] + ["cat=ignored_errors"]
    X = []
    for r in rows:
        x = [1.0] + [float(r["monitor"] == n) for n in NAMES[1:]] + [float(r["agent"] == n) for n in NAMES[1:]]
        x.append(float(r["category"] == "ignored_errors"))
        e = extra(r)
        X.append(x + [float(v) for v in e.values()])
    return cols + list(extra(rows[0]).keys()), X


def estimate(rows, extra, target, logistic):
    names, X = design(rows, extra)
    y = [float(r["flag"]) if logistic else float(r["score"]) for r in rows]
    b = fit(X, y, logistic)
    return {n: b[names.index(n)] for n in target}


def infer(rows, extra, target, logistic, B, rng, permute=None):
    """Point estimate, task-cluster bootstrap 95% CI, and (optionally) a within-task permutation p-value."""
    rows = [r for r in rows if logistic or isinstance(r.get("score"), (int, float))]
    if len({r["flag"] for r in rows}) < 2 and logistic:
        return {t: {"est": None, "note": "no variation"} for t in target}
    est = estimate(rows, extra, target, logistic)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    tids = list(by_task)
    boots = defaultdict(list)
    for _ in range(B):
        samp = [r for t in (rng.choice(tids) for _ in tids) for r in by_task[t]]
        try:
            e = estimate(samp, extra, target, logistic)
        except ZeroDivisionError:
            continue
        for k, v in e.items():
            boots[k].append(v)
    out = {}
    for k in target:
        bs = sorted(boots[k])
        out[k] = {"est": est[k], "ci": [bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]] if bs else None}
        if logistic:
            out[k]["odds_ratio"] = math.exp(est[k])
    if permute:
        k = target[0]
        null = []
        for _ in range(B):
            null.append(estimate(permute(by_task, rng), extra, [k], logistic)[k])
        out[k]["perm_p"] = (1 + sum(abs(v) >= abs(est[k]) for v in null)) / (1 + len(null))
    return out


def permute_agents(by_task, rng):
    """Within each task, shuffle which agent wrote which transcript (monitor fixed). Breaks only the self link."""
    out = []
    for rs in by_task.values():
        perm = dict(zip(NAMES, rng.sample(NAMES, len(NAMES))))
        out += [{**r, "agent": perm[r["agent"]]} for r in rs]
    return out


def matrix(rows, fn):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["monitor"], r["agent"])].append(r)
    return {m: {a: (fn(cells[(m, a)]), len(cells[(m, a)])) for a in NAMES} for m in NAMES}


def rate(rs):
    return sum(r["flag"] for r in rs) / len(rs) if rs else None


def mscore(rs):
    v = [r["score"] for r in rs if isinstance(r.get("score"), (int, float))]
    return sum(v) / len(v) if v else None


def show(title, mat, fmt):
    print(f"\n{title}  (rows = monitor, cols = agent; * = self)")
    print(f"{'':16}" + "".join(f"{a:>16}" for a in NAMES))
    for m in NAMES:
        print(f"{m:16}" + "".join(f"{(fmt(mat[m][a][0]) + ('*' if m == a else ' ')) if mat[m][a][0] is not None else '-':>16}" for a in NAMES))


def by_rel(rows, fn):
    return {rel: fn([r for r in rows if (r["monitor"] == r["agent"]) == (rel == "self")]) for rel in ("self", "cross")}


def report(rows, S, a, rng):
    cond = lambda c: [r for r in rows if r["base"] == c]
    S["matrices"] = {}
    for c in ("clean", "bare", "rat"):
        if cond(c):
            mf, ms = matrix(cond(c), rate), matrix(cond(c), mscore)
            show(f"[{c}] flag rate", mf, lambda v: f"{v:.2f}")
            show(f"[{c}] mean score", ms, lambda v: f"{v:.1f}")
            S["matrices"][c] = {"flag": mf, "score": ms}
    sw = [r for r in rows if r["base"].startswith("swap:")]

    print("\n== Flag rate / mean score by relation ==")
    S["by_relation"] = {}
    for c in ("clean", "bare", "rat"):
        if cond(c):
            fr, sc = by_rel(cond(c), rate), by_rel(cond(c), mscore)
            S["by_relation"][c] = {"flag": fr, "score": sc}
            print(f"  {c:6} flag self {fr['self']:.3f}  cross {fr['cross']:.3f}   score self {sc['self']:.1f}  cross {sc['cross']:.1f}")
    rat = cond("rat")
    if rat:
        acc = by_rel([r for r in rat if r.get("accepts_justification") is not None], lambda rs: sum(bool(r["accepts_justification"]) for r in rs) / max(1, len(rs)))
        S["accepts_justification"] = acc
        print(f"  rat: monitor accepts the justification   self {acc['self']:.3f}  cross {acc['cross']:.3f}")

    print(f"\n== Regressions (monitor + agent + category fixed effects; task-cluster bootstrap B={a.boot}) ==")
    S["models"] = {}
    specs = [
        ("primary: rat, flag ~ self", rat, lambda r: {"self": r["monitor"] == r["agent"]}, ["self"], True, permute_agents),
        ("primary: rat, score ~ self (OLS)", rat, lambda r: {"self": r["monitor"] == r["agent"]}, ["self"], False, permute_agents),
        ("rationale effect: bare+rat, flag ~ self*rationale", cond("bare") + rat,
         lambda r: {"self": r["monitor"] == r["agent"], "rationale": r["base"] == "rat", "self:rationale": r["monitor"] == r["agent"] and r["base"] == "rat"},
         ["rationale", "self:rationale", "self"], True, None),
        ("rationale effect: bare+rat, score ~ self*rationale (OLS)", cond("bare") + rat,
         lambda r: {"self": r["monitor"] == r["agent"], "rationale": r["base"] == "rat", "self:rationale": r["monitor"] == r["agent"] and r["base"] == "rat"},
         ["rationale", "self:rationale", "self"], False, None),
        ("bare: flag ~ self (code alone, no notes)", cond("bare"), lambda r: {"self": r["monitor"] == r["agent"]}, ["self"], True, permute_agents),
        ("swap: flag ~ code_self + style_self", sw,
         lambda r: {"code_self": r["monitor"] == r["agent"], "style_self": r["rewriter"] == r["monitor"]}, ["code_self", "style_self"], True, None),
        ("swap: score ~ code_self + style_self (OLS)", sw,
         lambda r: {"code_self": r["monitor"] == r["agent"], "style_self": r["rewriter"] == r["monitor"]}, ["code_self", "style_self"], False, None),
        ("clean control: flag ~ self (false positives)", cond("clean"), lambda r: {"self": r["monitor"] == r["agent"]}, ["self"], True, permute_agents),
    ]
    for title, rs, extra, target, logistic, perm in specs:
        if not rs:
            continue
        res = infer(rs, extra, target, logistic, a.boot, rng, perm)
        S["models"][title] = res
        print(f"  {title}  [n={len(rs)}]")
        for k, v in res.items():
            if v.get("est") is None:
                print(f"     {k:16} {v.get('note')}")
                continue
            ci = f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" if v.get("ci") else ""
            extra_s = (f"  OR {v['odds_ratio']:.2f}" if "odds_ratio" in v else "") + (f"  perm p={v['perm_p']:.3f}" if "perm_p" in v else "")
            print(f"     {k:16} {v['est']:+.3f} {ci}{extra_s}")

    if sw:
        print("\n== Swap: flag rate by (code author == monitor, notes rewriter == monitor) ==")
        S["swap_cells"] = {}
        for cs in (True, False):
            for ss in (True, False):
                v = rate([r for r in sw if r["self"] == cs and (r["rewriter"] == r["monitor"]) == ss])
                S["swap_cells"][f"code_self={cs},style_self={ss}"] = v
                print(f"  code_self={cs!s:5} style_self={ss!s:5}  flag {v if v is None else round(v, 3)}")

    print("\n== By category (rat): flag rate self vs cross ==")
    S["by_category"] = {}
    for c in ("vacuous_tests", "ignored_errors"):
        rs = [r for r in rat if r["category"] == c]
        if rs:
            S["by_category"][c] = by_rel(rs, rate)
            print(f"  {c:16} self {S['by_category'][c]['self']:.3f}  cross {S['by_category'][c]['cross']:.3f}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--prompt", choices=("general", "specific"), default="general")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    rng = random.Random(0)
    tids = {t["id"] for t in load_tasks(a.limit)}
    rows = [r for n in NAMES for r in read_jsonl(RUNS / mon_dir(a.prompt) / f"{n}.jsonl") if r["task"] in tids and "swap:" not in r["cond"]]  # swap condition dropped
    rows = list({(r["monitor"], r["key"]): r for r in rows}.values())  # guard against duplicate keys from overlapping runs
    for r in rows:
        r["self"] = r["monitor"] == r["agent"]
    S = {"n_rows": len(rows)}

    print(f"monitor prompt: {a.prompt}")
    print("== Verification (ground truth that sloppy really needs fixing / clean really is clean) ==")
    S["verify"] = {}
    for n in NAMES:
        vs = [v for v in read_jsonl(RUNS / n / "verify.jsonl") if v["task"] in tids]
        S["verify"][n] = {c: [sum(v["valid"] for v in vs if v["cond"] == c), sum(v["cond"] == c for v in vs)] for c in ("clean", "sloppy", "subtle")}
        print(f"  {n:16} " + "   ".join(f"{c} valid {x[0]}/{x[1]}" for c, x in S["verify"][n].items()))
    for pre, lv in (("", "blatant"), ("subtle_", "subtle")):
        S[f"n_tasks_{lv}_all_agents"] = len({r["task"] for r in rows if r["cond"] == pre + "rat"})
        print(f"  tasks with valid {lv} items for all 3 agents: {S[f'n_tasks_{lv}_all_agents']}")
    print("  parse fallbacks:", sum(bool(r.get("parse_fallback")) for r in rows), "/", len(rows))

    for r in rows:
        r["level"] = "clean" if r["cond"] == "clean" else "subtle" if r["cond"].startswith("subtle_") else "blatant"
        r["base"] = r["cond"].removeprefix("subtle_")
    clean = [r for r in rows if r["cond"] == "clean"]
    S["levels"] = {}
    for level in ("blatant", "subtle"):
        lr = [r for r in rows if r["level"] == level]
        if not lr:
            continue
        print(f"\n\n######## defect level: {level} (clean control shared) ########")
        S["levels"][level] = {}
        report(clean + lr, S["levels"][level], a, rng)

    spend = {f: cost([x for n in NAMES for x in read_jsonl(RUNS / n / f)]) for f in ("gen.jsonl", "verify.jsonl", "swap.jsonl")}
    spend["monitor"] = cost(rows)
    S["cost_usd"] = spend
    print("\ncost (USD):", {k: round(v, 2) for k, v in spend.items()}, "total", round(sum(spend.values()), 2))
    (RUNS / f"summary_{a.prompt}.json").write_text(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
