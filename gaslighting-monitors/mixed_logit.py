"""Mixed-effects logistic regression of the math viewer, pure Python:
  fooled ~ attacker + grader + frame + relation [+ solved + relation x not solved] + (1 | problem)
fit by maximum likelihood, with the problem intercept integrated out by Gauss-Hermite quadrature.
The solved terms are dropped when every row has the same solved value (as in math4, where every grader solved every problem).

pp effects: the change in fooled rate (percentage points) from switching one term (relation, frame, solved) for every row,
averaged over rows and over the problem effect. 95% intervals: 2000 parametric draws from the fitted estimate's
asymptotic normal distribution (the "model-based intervals" of the viewer).

    from mixed_logit import analyze; analyze(rows, ref_model="glm-5.3")
rows: dicts with fooled, attacker, grader, frame, relation, solved, qid.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

FRAME_LABEL = {"self": "self-check", "skim1k": "skim1k"}
# Weak N(0, PRIOR_SD^2) penalty on every fixed effect except the intercept (log-odds scale), so a model that is never
# fooled (complete separation) gets a large but finite coefficient instead of diverging. 0 = plain maximum likelihood.
PRIOR_SD = 2.5
NODES = 24


def _gh(n):
    """Nodes and weights for E[f(Z)], Z ~ N(0,1), n even: Newton on the Hermite polynomial (Numerical Recipes gauher)."""
    xs, ws = [], []
    for i in range(n // 2):
        if i == 0:
            z = math.sqrt(2 * n + 1) - 1.85575 * (2 * n + 1) ** (-1 / 6)
        elif i == 1:
            z = xs[0] - 1.14 * n ** 0.426 / xs[0]
        elif i == 2:
            z = 1.86 * xs[1] - 0.86 * xs[0]
        elif i == 3:
            z = 1.91 * xs[2] - 0.91 * xs[1]
        else:
            z = 2 * xs[i - 1] - xs[i - 2]
        for _ in range(100):
            p1, p2 = math.pi ** -0.25, 0.0
            for j in range(1, n + 1):
                p3, p2 = p2, p1
                p1 = z * math.sqrt(2 / j) * p2 - math.sqrt((j - 1) / j) * p3
            pp = math.sqrt(2 * n) * p2
            z1, z = z, z - p1 / pp
            if abs(z - z1) < 1e-14:
                break
        xs.append(z)
        ws.append(2 / (pp * pp))
    nodes = [s * z * math.sqrt(2) for z in xs for s in (1, -1)]
    weights = [w / math.sqrt(math.pi) for w in ws for _ in (1, -1)]
    return nodes, weights


GH_Z, GH_W = _gh(NODES)


def sig(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def design(rows, ref_model, use_solved):
    models = sorted({r["attacker"] for r in rows} | {r["grader"] for r in rows})
    others = [m for m in models if m != ref_model]
    frames = [f for f in ("self", "skim1k") if any(r["frame"] == f for r in rows)]
    rels = [x for x in ("self", "sibling") if any(r["relation"] == x for r in rows)]
    names = ["intercept"] + [f"attacker {m}" for m in others] + [f"grader {m}" for m in others] + \
            [f"frame {FRAME_LABEL[f]}" for f in frames]
    if use_solved:
        names += ["solved"]
    names += [f"relation {x}" for x in rels]
    if use_solved:
        names += [f"relation {x} x not solved" for x in rels]

    def x_of(a, g, fr, rel, solved):
        v = [1.0] + [float(a == m) for m in others] + [float(g == m) for m in others] + [float(fr == f) for f in frames]
        if use_solved:
            v.append(float(solved))
        v += [float(rel == x) for x in rels]
        if use_solved:
            v += [float(rel == x and not solved) for x in rels]
        return v
    return names, x_of


def group(rows, x_of):
    """problem -> list of (x, k, n) over distinct covariate patterns."""
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        key = (r["attacker"], r["grader"], r["frame"], r["relation"], bool(r["solved"]))
        c = by[r["qid"]][key]
        c[0] += bool(r["fooled"])
        c[1] += 1
    return [[(x_of(*key), k, n) for key, (k, n) in pats.items()] for pats in by.values()]


def loglik_grad(theta, groups, fixed_sigma0=False):
    """theta = beta + [log sigma]. Returns log-likelihood and gradient."""
    p = len(theta) - (0 if fixed_sigma0 else 1)
    beta = theta[:p]
    sigma = 0.0 if fixed_sigma0 else math.exp(theta[p])
    zs, ws = ([0.0], [1.0]) if fixed_sigma0 else (GH_Z, GH_W)
    ll, grad = 0.0, [0.0] * len(theta)
    for pats in groups:
        etas = [sum(b * x for b, x in zip(beta, xv)) for xv, _, _ in pats]
        logs, gq = [], []
        for z, w in zip(zs, ws):
            s = math.log(w)
            g = [0.0] * len(theta)
            for (xv, k, n), eta in zip(pats, etas):
                e = eta + sigma * z
                # log Binomial kernel (the constant is the same for every theta)
                s += k * e - n * (e + math.log1p(math.exp(-e)) if e > 0 else math.log1p(math.exp(e)))
                res = k - n * sig(e)
                for j in range(p):
                    if xv[j]:
                        g[j] += res * xv[j]
                if not fixed_sigma0:
                    g[p] += res * z * sigma
            logs.append(s)
            gq.append(g)
        m = max(logs)
        wts = [math.exp(s - m) for s in logs]
        tot = sum(wts)
        ll += m + math.log(tot)
        for wt, g in zip(wts, gq):
            for j in range(len(theta)):
                grad[j] += wt / tot * g[j]
    if PRIOR_SD:
        for j in range(1, p):
            ll -= 0.5 * beta[j] ** 2 / PRIOR_SD ** 2
            grad[j] -= beta[j] / PRIOR_SD ** 2
    return ll, grad


def hessian(theta, groups, fixed_sigma0=False, h=1e-4):
    k = len(theta)
    H = [[0.0] * k for _ in range(k)]
    for j in range(k):
        tp, tm = theta[:], theta[:]
        tp[j] += h
        tm[j] -= h
        gp, gm = loglik_grad(tp, groups, fixed_sigma0)[1], loglik_grad(tm, groups, fixed_sigma0)[1]
        for i in range(k):
            H[i][j] = (gp[i] - gm[i]) / (2 * h)
    return [[(H[i][j] + H[j][i]) / 2 for j in range(k)] for i in range(k)]


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[c][c]:
                f = M[r][c] / M[c][c]
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] / M[i][i] for i in range(n)]


def inverse(A):
    n = len(A)
    cols = [solve(A, [float(i == j) for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def fit(groups, k_beta, fixed_sigma0=False):
    theta = [0.0] * k_beta + ([] if fixed_sigma0 else [math.log(0.8)])
    ll, g = loglik_grad(theta, groups, fixed_sigma0)
    for _ in range(100):
        H = hessian(theta, groups, fixed_sigma0)
        ridge = 1e-6
        while True:
            try:
                step = solve([[-H[i][j] + (ridge if i == j else 0) for j in range(len(theta))] for i in range(len(theta))], g)
            except ZeroDivisionError:
                step = None
            if step is not None:
                t = 1.0
                while t > 1e-6:
                    cand = [a + t * s for a, s in zip(theta, step)]
                    if not fixed_sigma0:
                        cand[-1] = max(cand[-1], math.log(1e-3))
                    ll2, g2 = loglik_grad(cand, groups, fixed_sigma0)
                    if ll2 >= ll - 1e-10:
                        break
                    t /= 2
                if t > 1e-6:
                    break
            ridge = ridge * 100 if ridge < 1e6 else None
            if ridge is None:
                return theta, ll, H
        done = abs(ll2 - ll) < 1e-9
        theta, ll, g = cand, ll2, g2
        if done:
            break
    return theta, ll, hessian(theta, groups, fixed_sigma0)


def chi2_sf(x, df):
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df == 2:
        return math.exp(-x / 2)
    raise ValueError(df)


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(max(s, 1e-14)) if i == j else s / L[j][j]
    return L


def analyze(rows, ref_model, draws=2000, seed=0):
    use_solved = len({bool(r["solved"]) for r in rows}) > 1
    names, x_of = design(rows, ref_model, use_solved)
    groups = group(rows, x_of)
    kb = len(names)
    theta, ll, H = fit(groups, kb)
    cov = inverse([[-h for h in row] for row in H])
    _, ll0, _ = fit(groups, kb, fixed_sigma0=True)
    se = [math.sqrt(max(cov[i][i], 0)) for i in range(len(theta))]
    coef = []
    for i, n in enumerate(names):
        z = theta[i] / se[i] if se[i] else 0
        coef.append({"term": n, "est": theta[i], "se": se[i], "p": 2 * norm_sf(abs(z)), "OR": math.exp(min(theta[i], 700)),
                     "OR lo": math.exp(min(theta[i] - 1.96 * se[i], 700)), "OR hi": math.exp(min(theta[i] + 1.96 * se[i], 700))})
    ls, ls_se = theta[-1], se[-1]
    sigma = math.exp(ls)
    out = {"formula": "fooled ~ attacker + grader + frame + relation" + (" + solved + relation x not solved" if use_solved else "")
                      + " + (1 | problem)",
           "use_solved": use_solved, "prior_sd": PRIOR_SD, "coef": coef, "loglik": ll, "n_params": len(theta), "AIC": 2 * len(theta) - 2 * ll,
           "sigma problem": {"est": sigma, "lo": math.exp(ls - 1.96 * ls_se), "hi": math.exp(ls + 1.96 * ls_se)},
           "ICC (latent)": sigma ** 2 / (sigma ** 2 + math.pi ** 2 / 3)}
    stat = max(0.0, 2 * (ll - ll0))
    out["LR stat (sigma=0)"], out["LR p (sigma=0)"] = stat, 0.5 * chi2_sf(stat, 1)  # boundary: 50:50 mixture
    if use_solved:  # LR test of the relation x not solved terms
        names_r, x_r = design(rows, ref_model, use_solved)
        keep = [i for i, n in enumerate(names_r) if "x not solved" not in n]
        groups_r = [[([xv[i] for i in keep], k, n) for xv, k, n in pats] for pats in groups]
        _, ll_r, _ = fit(groups_r, len(keep))
        st = max(0.0, 2 * (ll - ll_r))
        out["LR stat (no relation x solved)"], out["LR p (no relation x solved)"] = st, chi2_sf(st, 2)

    # pp effects: counterfactual rows, averaged over the problem effect by quadrature.
    pats = defaultdict(int)
    for r in rows:
        pats[(r["attacker"], r["grader"], r["frame"], r["relation"], bool(r["solved"]))] += 1
    pats = list(pats.items())
    total = sum(n for _, n in pats)

    def mean_rate(th, change):
        b, s = th[:kb], math.exp(th[kb])
        acc = 0.0
        for (a, g, fr, rel, sv), n in pats:
            kw = {"a": a, "g": g, "fr": fr, "rel": rel, "sv": sv, **change}
            eta = sum(bb * x for bb, x in zip(b, x_of(kw["a"], kw["g"], kw["fr"], kw["rel"], kw["sv"])))
            acc += n * sum(w * sig(eta + s * z) for z, w in zip(GH_Z, GH_W))
        return acc / total

    contrasts = {}
    for rel in ("self", "sibling"):
        if f"relation {rel}" in names:
            contrasts[f"pp {rel} vs cross"] = ({"rel": rel}, {"rel": "cross"}, None)
            if use_solved:
                contrasts[f"pp {rel} vs cross | solved"] = ({"rel": rel, "sv": True}, {"rel": "cross", "sv": True}, None)
                contrasts[f"pp {rel} vs cross | not solved"] = ({"rel": rel, "sv": False}, {"rel": "cross", "sv": False}, None)
    for fr in ("self", "skim1k"):
        if f"frame {FRAME_LABEL[fr]}" in names:
            contrasts[f"pp frame {FRAME_LABEL[fr]} vs neutral"] = ({"fr": fr}, {"fr": "neutral"}, None)
    if use_solved:
        contrasts["pp solved vs not solved"] = ({"sv": True}, {"sv": False}, None)
    rng = random.Random(seed)
    L = cholesky(cov)
    sims = []
    for _ in range(draws):
        e = [rng.gauss(0, 1) for _ in theta]
        sims.append([t + sum(L[i][j] * e[j] for j in range(i + 1)) for i, t in enumerate(theta)])
    for key, (c1, c0, _) in contrasts.items():
        est = 100 * (mean_rate(theta, c1) - mean_rate(theta, c0))
        vals = sorted(100 * (mean_rate(th, c1) - mean_rate(th, c0)) for th in sims)
        out[key] = {"est": est, "lo": vals[int(0.025 * draws)], "hi": vals[int(0.975 * draws) - 1]}
    if use_solved:
        for rel in ("self", "sibling"):
            i = names.index(f"relation {rel}")
            j = names.index(f"relation {rel} x not solved")
            e1, v1 = theta[i], cov[i][i]
            e2, v2 = theta[i] + theta[j], cov[i][i] + cov[j][j] + 2 * cov[i][j]
            for lab, e, v in ((" | solved", e1, v1), (" | not solved", e2, v2)):
                sd = math.sqrt(max(v, 0))
                out[f"OR {rel} vs cross{lab}"] = {"est": math.exp(e), "lo": math.exp(e - 1.96 * sd), "hi": math.exp(e + 1.96 * sd)}
    out["n_rows"], out["n_fooled"] = len(rows), sum(bool(r["fooled"]) for r in rows)
    return out
