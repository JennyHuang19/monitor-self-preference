"""Family effect by logistic regression, the model in the AITA viewer:
  flip ~ attacker + grader + frame + grader x frame + relation + relation x frame + |Reddit-agreement gap| + |tier gap|
over every eligible (post, attacker, grader, frame) row (grader's no-pitch verdict was off the attacker's target).
Strength = grader's no-pitch agreement with Reddit (gap in units of 10 pp). Tier is fixed before the run; a term
that is constant in the pool (e.g. all tier 1 in frontier3) is dropped, as is a relation level with no rows.
Rows sharing (frame, attacker, grader) share every covariate, so the fit is a grouped binomial logit by Newton.
95% CIs: cluster bootstrap over posts (both frames of a post resampled together). Writes runs/<pool>/regression.json.
--no-gap drops the |Reddit-agreement gap| term (a sensitivity fit for pools where every grader has about the same
strength, so the gap is nearly collinear with self vs cross) and writes regression_nogap.json.

    python regression.py --pool frontier3 [--boot 2000] [--no-gap]
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict

from common import MODELS, POOL, RUNS, read_jsonl, relation, run_dir

FRAMES = {"neutral": "grades.jsonl", "self": "grades_self.jsonl"}
# 1 flagship, 2 flash, 3 open-large, 4 open-small (as in the viewer).
TIER = {"glm-5.3": 1, "kimi-k3": 1, "kimi-k3.1-pro": 1, "qwen3.8-max": 1, "glm-5.3-flash": 2, "qwen3.8-flash": 2,
        "gpt-oss-120b": 3, "gemma-4-31b": 3, "gpt-oss-20b": 4, "gemma-4-26b-a4b": 4,
        "gpt-6-astra": 1, "claude-opus-4.8": 1, "gemini-4-argon": 1, "gpt-6-sol": 1}
RIDGE = 1e-4
NO_GAP = "--no-gap" in __import__("sys").argv
NAMES = [m["name"] for m in MODELS]
BYNAME = {m["name"]: m for m in MODELS}

base = {(b["pid"], b["grader"]): b for b in read_jsonl(RUNS / "baseline.jsonl")}
strength = {}
for n in NAMES:
    bs = [b for b in base.values() if b["grader"] == n]
    strength[n] = 100 * sum(b["matches_reddit"] for b in bs) / len(bs)

# post -> {(frame, attacker, grader): [flips, n]}
by_pid: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for fr, fname in FRAMES.items():
    for att in NAMES:
        for g in read_jsonl(run_dir(att) / fname):
            b = base.get((g["pid"], g["grader"]))
            if b is None or b["verdict"] is None or b["verdict"] == g["target"]:
                continue
            c = by_pid[g["pid"]][(fr, att, g["grader"])]
            c[0] += g["verdict"] == g["target"]
            c[1] += 1
CELLS = sorted({k for d in by_pid.values() for k in d})
RELS = [r for r in ("self", "sibling") if any(relation(BYNAME[g], BYNAME[a]) == r for _, a, g in CELLS)]
USE_TIER = len({TIER[n] for n in NAMES}) > 1


def features(fr, a, g, rel=None, frame=None, interact=True):
    fr = frame or fr
    rel = rel or relation(BYNAME[g], BYNAME[a])
    s = fr == "self"
    x = {"intercept": 1.0}
    for n in NAMES[1:]:
        x[f"attacker {n}"] = float(a == n)
        x[f"grader {n}"] = float(g == n)
    x["frame self"] = float(s)
    for n in NAMES[1:]:
        x[f"grader {n} x frame self"] = float(g == n and s)
    for r in RELS:
        x[r] = float(rel == r)
    if not NO_GAP:
        x["|Reddit-agreement gap|"] = abs(strength[a] - strength[g]) / 10
    if USE_TIER:
        x["|tier gap|"] = float(abs(TIER[a] - TIER[g]))
    if interact:
        for r in RELS:
            x[f"{r} x self"] = float(rel == r and s)
    return x


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def sig(z):
    return 1 / (1 + math.exp(-z)) if z > -30 else math.exp(z)


def fit(counts, interact=True):
    """Grouped binomial logit. counts: {cell: [k, n]}. Returns (names, beta, loglik)."""
    rows = [(features(*c, interact=interact), k, n) for c, (k, n) in counts.items() if n]
    names = list(rows[0][0])
    X = [[x[v] for v in names] for x, _, _ in rows]
    beta = [0.0] * len(names)
    for _ in range(50):
        H = [[RIDGE * (i == j) for j in range(len(names))] for i in range(len(names))]
        grad = [-RIDGE * b for b in beta]
        for xi, (_, k, n) in zip(X, rows):
            p = sig(sum(a * b for a, b in zip(xi, beta)))
            w = n * p * (1 - p)
            for i, xa in enumerate(xi):
                if xa:
                    grad[i] += xa * (k - n * p)
                    for j, xb in enumerate(xi):
                        if xb:
                            H[i][j] += w * xa * xb
        step = solve(H, grad)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    ll = 0.0
    for xi, (_, k, n) in zip(X, rows):
        p = min(max(sig(sum(a * b for a, b in zip(xi, beta))), 1e-12), 1 - 1e-12)
        ll += k * math.log(p) + (n - k) * math.log(1 - p)
    return names, beta, ll


def predict(names, beta, x):
    return sig(sum(b * x.get(v, 0.0) for v, b in zip(names, beta)))


def summarize(counts, names, beta):
    """Odds ratios and average marginal effects (pp), weighted by eligible rows."""
    B = dict(zip(names, beta))
    out = {}

    def ame(fn, frames):
        num = den = 0.0
        for (fr, a, g), (k, n) in counts.items():
            if fr in frames and n:
                num += n * fn(fr, a, g)
                den += n
        return 100 * num / den

    for r in RELS:
        for fr in FRAMES:
            out[f"{fr}|OR {r} vs cross"] = math.exp(B[r] + (B[f"{r} x self"] if fr == "self" else 0))
            out[f"{fr}|pp {r} vs cross"] = ame(lambda f, a, g: predict(names, beta, features(f, a, g, rel=r)) -
                                               predict(names, beta, features(f, a, g, rel="cross")), {fr})
        out[f"pp {r} vs cross"] = ame(lambda f, a, g: predict(names, beta, features(f, a, g, rel=r)) -
                                      predict(names, beta, features(f, a, g, rel="cross")), set(FRAMES))
    out["pp frame self-check vs neutral"] = ame(lambda f, a, g: predict(names, beta, features(f, a, g, frame="self")) -
                                                predict(names, beta, features(f, a, g, frame="neutral")), set(FRAMES))
    if not NO_GAP:
        out["OR per 10 pp |Reddit-agreement gap|"] = math.exp(B["|Reddit-agreement gap|"])
    if USE_TIER:
        out["OR per unit |tier gap|"] = math.exp(B["|tier gap|"])
    return out


def chi2_sf(x, k):
    """Upper tail of chi-square with k df (regularized incomplete gamma, series / continued fraction)."""
    a, z = k / 2, x / 2
    if z <= 0:
        return 1.0
    if z < a + 1:
        s = t = 1 / a
        for n in range(1, 500):
            t *= z / (a + n)
            s += t
        return 1 - s * math.exp(-z + a * math.log(z) - math.lgamma(a))
    b, c, d = z + 1 - a, 1e300, 1 / (z + 1 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = 1 / (an * d + b)
        c = b + an / c
        h *= d * c
    return math.exp(-z + a * math.log(z) - math.lgamma(a)) * h


def total(pids):
    t = defaultdict(lambda: [0, 0])
    for p in pids:
        for c, (k, n) in by_pid[p].items():
            t[c][0] += k
            t[c][1] += n
    return t


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-gap", action="store_true")
    a = ap.parse_args()
    pids = sorted(by_pid)
    counts = total(pids)
    names, beta, ll = fit(counts)
    _, _, ll0 = fit(counts, interact=False)
    point = summarize(counts, names, beta)
    lr = 2 * (ll - ll0)
    # Overdispersion and the cells the model fits worst.
    pear, worst = 0.0, []
    for c, (k, n) in counts.items():
        p = predict(names, beta, features(*c))
        r2 = (k - n * p) ** 2 / (n * p * (1 - p))
        pear += r2
        worst.append([*c, relation(BYNAME[c[2]], BYNAME[c[1]]), k, n, n * p, r2])
    worst.sort(key=lambda w: -w[-1])
    same = [float(relation(BYNAME[g], BYNAME[at]) != "cross") for _, at, g in counts]
    gap = [abs(strength[at] - strength[g]) for _, at, g in counts]
    mu_s, mu_g = sum(same) / len(same), sum(gap) / len(gap)
    cov = sum((s - mu_s) * (g - mu_g) for s, g in zip(same, gap))
    corr = cov / math.sqrt(sum((s - mu_s) ** 2 for s in same) * sum((g - mu_g) ** 2 for g in gap) or 1)

    rng = random.Random(0)
    boots = defaultdict(list)
    for i in range(a.boot):
        t = total([rng.choice(pids) for _ in pids])
        try:
            n_, b_, _ = fit(t)
            for k, v in summarize(t, n_, b_).items():
                boots[k].append(v)
        except (ZeroDivisionError, OverflowError, IndexError):
            pass
    est = {}
    for k, v in point.items():
        bs = sorted(boots[k])
        est[k] = {"est": v, "lo": bs[int(0.025 * len(bs))], "hi": bs[int(0.975 * len(bs)) - 1]}

    n_rows = sum(n for _, n in counts.values())
    res = {"pool": POOL, "models": NAMES, "strength": strength, "tier": {n: TIER[n] for n in NAMES},
           "B": a.boot, "B_ok": len(boots[next(iter(point))]), "n_posts": len(pids), "n_rows": n_rows,
           "n_flips": sum(k for k, _ in counts.values()),
           "by frame": {fr: [sum(k for c, (k, n) in counts.items() if c[0] == fr),
                             sum(n for c, (k, n) in counts.items() if c[0] == fr)] for fr in FRAMES},
           "formula": "flip ~ attacker + grader + frame + grader × frame + relation + relation × frame" +
                      ("" if NO_GAP else " + |Reddit-agreement gap|") + (" + |tier gap|" if USE_TIER else "") +
                      ("" if USE_TIER else "   (|tier gap| dropped: every model is tier 1)") +
                      ("" if "sibling" in RELS else "   (no sibling pairs: relation is self vs cross)"),
           "corr same-family vs |Reddit-agreement gap|": corr,
           "LR relation x frame": {"stat": lr, "df": len(RELS), "p": chi2_sf(lr, len(RELS))},
           "coef": dict(zip(names, beta)), "pearson/df": pear / max(1, len(counts) - len(names)),
           "worst cells": worst[:6], "estimates": est}
    (RUNS / ("regression_nogap.json" if NO_GAP else "regression.json")).write_text(json.dumps(res, indent=2))

    print(f"\n=== regression ({POOL}): {res['formula']}")
    print(f"  {len(pids)} posts, {n_rows} eligible rows, {res['n_flips']} flips; by frame {res['by frame']}")
    print(f"  no-pitch Reddit agreement: " + ", ".join(f"{n} {s:.0f}%" for n, s in strength.items()))
    for k, v in est.items():
        print(f"  {k:42} {v['est']:+8.3f}  [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    print(f"  LR test relation x frame: stat {lr:.2f}, df {len(RELS)}, p {res['LR relation x frame']['p']:.3g}")
    print(f"  pearson/df {res['pearson/df']:.2f}; corr(same-family, |gap|) {corr:+.2f}; bootstrap fits ok "
          f"{res['B_ok']}/{a.boot}")
