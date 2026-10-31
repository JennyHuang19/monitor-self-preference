"""Family effect on the math3 mix5 run, from one mixed logistic regression over all three grading frames that holds
each model's strength fixed and lets each problem have its own baseline deceivability:

    logit P(fooled) = attacker FE + grader FE + frame FE + relation (self / sibling vs cross)
                      + u_problem,      u_problem ~ N(0, sigma^2)

Frame is the prompt type (neutral, self-check, skim1k); neutral is the reference, and its shift is the same for
every grader. The problem effect absorbs that some problems fool graders more than others, whoever attacks and grades.
"A x frame" adds relation x frame terms, and a likelihood-ratio test says whether the family effect differs by frame.
"A: same family" replaces self and sibling with one same-family indicator (self or sibling vs cross); a
likelihood-ratio test against model A says whether self and sibling differ.

Rows: one grade of a wrong pitch. Main fit uses only items the grader solved in its no-pitch baseline ("fooled when
it knew"), so a fooled grade means persuasion, not ignorance; the all-rows fit is reported too. The problem effect is
integrated out on a fixed grid; parameters are fit by maximum likelihood. 95% CIs are model-based (Wald, from the
observed information of the marginal likelihood): the problem effect models the clustering by problem, so no
bootstrap. pp effects average over the problem-effect distribution, with CIs from 4,000 draws of the parameters.
Writes runs/regression_math3.json.

    uv run --no-project --python 3.11 --with numpy --with scipy python fit_regression.py
"""
import json
import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from common import RUNS, read_jsonl

NAMES = ["glm-5.3", "glm-5.3-flash", "gpt-oss-120b", "gpt-oss-20b", "kimi-k3"]
FAM = {"glm-5.3": "GLM", "glm-5.3-flash": "GLM", "gpt-oss-120b": "gpt-oss", "gpt-oss-20b": "gpt-oss", "kimi-k3": "Kimi"}
FRAMES = {"neutral": "grades.jsonl", "self": "grades_self.jsonl", "skim1k": "grades_skim1k.jsonl"}
FR = list(FRAMES)  # FR[0] is the reference frame
FRAME_NAME = {"neutral": "neutral", "self": "self-check", "skim1k": "skim1k"}  # "self" alone reads as the relation
DRAWS = 4000
Z = np.linspace(-7, 7, 141)  # grid for the problem effect, in units of sigma
LOGW = -Z ** 2 / 2 - np.log(np.exp(-Z ** 2 / 2).sum())

baseline = read_jsonl(RUNS / "math3_mix5__baseline" / "baseline.jsonl")
solved = {(b["qid"], b["grader"]): b["solved"] for b in baseline}
strength = {n: 100 * sum(b["solved"] for b in baseline if b["grader"] == n) /
            sum(b["grader"] == n for b in baseline) for n in NAMES}
REL = {(a, g): "self" if a == g else "sibling" if FAM[a] == FAM[g] else "cross" for a in NAMES for g in NAMES}
CELLS = [(f, a, g) for f in FR for a in NAMES for g in NAMES]
MODELS_FIT = (("A: fixed effects", False, False), ("A: same family", False, True), ("A × frame", True, False))
RELS = {False: ("self", "sibling"), True: ("same family",)}  # relation columns: split, or pooled (self or sibling)


def columns(inter, family=False):
    """Column names, in design order: intercept, attacker FE, grader FE, frame FE, self and sibling (or one
    same-family column)[, relation x frame]. The first model and the neutral frame are the references."""
    return (["intercept"] + [f"attacker {n}" for n in NAMES[1:]] + [f"grader {n}" for n in NAMES[1:]]
            + [f"frame {FRAME_NAME[f]}" for f in FR[1:]]
            + list(RELS[family])
            + ([f"{r} x frame {FRAME_NAME[f]}" for r in ("self", "sibling") for f in FR[1:]] if inter else []))


def design(f, a, g, inter, family=False):
    x = [1.0] + [float(a == n) for n in NAMES[1:]] + [float(g == n) for n in NAMES[1:]] + [float(f == n) for n in FR[1:]]
    x += [float(REL[a, g] != "cross")] if family else [float(REL[a, g] == "self"), float(REL[a, g] == "sibling")]
    if inter:
        x += [float(REL[a, g] == r and f == fr) for r in ("self", "sibling") for fr in FR[1:]]
    return x


def negloglik(theta, X, y, G, sigma=None):
    """Minus the marginal log-likelihood (problem effect integrated out) and its gradient. theta = beta [+ log sigma];
    sigma=0 fits the plain logistic regression."""
    k = X.shape[1]
    beta = theta[:k]
    s = math.exp(theta[k]) if sigma is None else sigma
    E = (X @ beta)[:, None] + s * Z[None, :]  # rows x nodes
    lq = G @ (y[:, None] * E - np.logaddexp(0, E)) + LOGW  # problems x nodes
    m = lq.max(1, keepdims=True)
    ll = (m[:, 0] + np.log(np.exp(lq - m).sum(1))).sum()
    pi = np.exp(lq - m)
    pi /= pi.sum(1, keepdims=True)
    WR = (G.T @ pi) * (y[:, None] - 1 / (1 + np.exp(-E)))  # posterior weight x residual, rows x nodes
    grad = X.T @ WR.sum(1)
    if sigma is None:
        grad = np.append(grad, (WR * Z).sum() * s)
    return -ll, -grad


def fit(X, y, G, sigma=None):
    k = X.shape[1]
    th0 = np.zeros(k) if sigma is not None else np.append(np.zeros(k), math.log(0.5))
    r = minimize(negloglik, th0, args=(X, y, G, sigma), jac=True, method="L-BFGS-B",
                 options={"maxiter": 5000, "gtol": 1e-8, "ftol": 1e-14})
    th = r.x
    # Observed information by central differences of the analytic gradient.
    h = 1e-5
    H = np.array([(negloglik(th + h * e, X, y, G, sigma)[1] - negloglik(th - h * e, X, y, G, sigma)[1]) / (2 * h)
                  for e in np.eye(len(th))])
    V = np.linalg.inv((H + H.T) / 2)
    return th, -r.fun, V


def mean_p(eta, s):
    """P(fooled) averaged over the problem effect. eta: (..., cells)."""
    return (np.exp(LOGW)[None, :] / (1 + np.exp(-(eta[..., None] + s * Z)))).sum(-1)


def effects(thetas, Xc, nc, cols):
    """pp and OR effects for each parameter draw (rows of thetas): relation vs cross, frame vs neutral (on the
    cells that have them, weighted by rows)."""
    k = Xc.shape[1]
    B, s = thetas[:, :k], np.exp(thetas[:, k])
    eta = B @ Xc.T  # draws x cells
    ix = {c: i for i, c in enumerate(cols)}
    out = {}

    def pp(keep, drop):
        w = nc * keep
        off = eta - B[:, drop] @ Xc[:, drop].T
        diff = np.stack([mean_p(eta[i], s[i]) - mean_p(off[i], s[i]) for i in range(len(B))])
        return 100 * (diff * w).sum(1) / w.sum()

    for r in [r for r in ("self", "sibling", "same family") if r in ix]:
        out[f"OR {r} vs cross"] = np.exp(B[:, ix[r]])
        out[f"pp {r} vs cross"] = pp(Xc[:, ix[r]] == 1, [ix[r]])
    for f in FR[1:]:
        name = FRAME_NAME[f]
        drop = [i for c, i in ix.items() if c == f"frame {name}" or c.endswith(f" x frame {name}")]
        out[f"pp frame {name} vs {FR[0]}"] = pp(Xc[:, ix[f"frame {name}"]] == 1, drop)
    return out


res = {"strength": strength, "frames": FR, "pooled": {}}
rng = np.random.default_rng(0)
for subset in ("knew", "all"):
    rows = []  # (qid, frame, attacker, grader, fooled)
    for f, fname in FRAMES.items():
        for a in NAMES:
            for g in read_jsonl(RUNS / f"math3_mix5__{a}" / fname):
                if g.get("verdict") is None or (subset == "knew" and not solved.get((g["qid"], g["grader"]))):
                    continue
                rows.append((g["qid"], f, a, g["grader"], float(bool(g["fooled"]))))
    qids = sorted({r[0] for r in rows})
    qi = {q: i for i, q in enumerate(qids)}
    y = np.array([r[4] for r in rows])
    G = np.zeros((len(qids), len(rows)))
    G[[qi[r[0]] for r in rows], np.arange(len(rows))] = 1
    counts = {c: [0, 0] for c in CELLS}
    for _, f, a, g, yy in rows:
        counts[f, a, g][0] += int(yy)
        counts[f, a, g][1] += 1
    nc = np.array([counts[c][1] for c in CELLS], float)
    po = res["pooled"][subset] = {
        "n_items": len(qids), "n_rows": len(rows), "n_fooled": int(y.sum()),
        "by frame": {f: [sum(counts[c][0] for c in CELLS if c[0] == f), sum(counts[c][1] for c in CELLS if c[0] == f)]
                     for f in FR}}
    print(f"\n=== all frames, {subset} rows ({po['n_fooled']} fooled of {po['n_rows']}, {len(qids)} problems) ===")
    lls = {}
    for model, inter, family in MODELS_FIT:
        cols = columns(inter, family)
        X = np.array([design(f, a, g, inter, family) for _, f, a, g, _ in rows])
        th, ll, V = fit(X, y, G)
        lls[model] = ll
        if inter:
            lr = max(0.0, 2 * (ll - lls["A: fixed effects"]))
            df = 2 * (len(FR) - 1)
            po[model] = {"LR stat": lr, "df": df, "LR p (no interaction)": float(chi2.sf(lr, df))}
            print(f"  {model}: LR test of relation x frame: chi2 {lr:.2f} on {df} df, p = {chi2.sf(lr, df):.3f}")
            continue
        ll0 = fit(X, y, G, sigma=0.0)[1]
        lr0 = max(0.0, 2 * (ll - ll0))
        k = len(cols)
        se = np.sqrt(np.diag(V))
        sig, sig_se = math.exp(th[k]), se[k]  # se of log sigma
        coef = [{"term": c, "est": float(b), "se": float(e), "p": float(2 * norm.sf(abs(b / e))),
                 "OR": math.exp(b), "OR lo": math.exp(b - 1.96 * e), "OR hi": math.exp(b + 1.96 * e)}
                for c, b, e in zip(cols, th[:k], se[:k])]
        Xc = np.array([design(*c, inter, family) for c in CELLS])
        point = {kk: float(v[0]) for kk, v in effects(th[None, :], Xc, nc, cols).items()}
        draws = effects(rng.multivariate_normal(th, V, DRAWS), Xc, nc, cols)
        po[model] = {kk: {"est": point[kk], "lo": float(np.quantile(draws[kk], .025)),
                          "hi": float(np.quantile(draws[kk], .975))} for kk in point}
        po[model].update({
            "coef": coef, "loglik": ll, "n_params": k + 1, "AIC": 2 * (k + 1) - 2 * ll,
            "sigma problem": {"est": sig, "lo": math.exp(th[k] - 1.96 * sig_se), "hi": math.exp(th[k] + 1.96 * sig_se)},
            "ICC (latent)": sig ** 2 / (sig ** 2 + math.pi ** 2 / 3),
            "LR stat (sigma=0)": lr0, "LR p (sigma=0)": float(0.5 * chi2.sf(lr0, 1))})
        if family:  # nested in model A: same family is self and sibling with one shared coefficient
            lr = max(0.0, 2 * (lls["A: fixed effects"] - ll))
            po[model].update({"LR stat (self = sibling)": lr, "LR p (self = sibling)": float(chi2.sf(lr, 1))})
        print(f"  {model}  (log-lik {ll:.1f}, AIC {po[model]['AIC']:.1f})")
        print(f"    {'term':40} {'log-odds':>9} {'SE':>7} {'p':>7} {'OR':>7}  95% CI")
        for c in coef:
            print(f"    {c['term']:40} {c['est']:9.3f} {c['se']:7.3f} {c['p']:7.3f} {c['OR']:7.3f}  "
                  f"[{c['OR lo']:.3f}, {c['OR hi']:.3f}]")
        s_ = po[model]["sigma problem"]
        print(f"    problem effect sigma {s_['est']:.3f} [{s_['lo']:.3f}, {s_['hi']:.3f}], latent ICC "
              f"{po[model]['ICC (latent)']:.3f}, LR vs none {lr0:.1f} (p = {po[model]['LR p (sigma=0)']:.2g})")
        for kk in point:
            v = po[model][kk]
            print(f"    {kk:40} {v['est']:8.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]")
        if family:
            print(f"    LR test self = sibling: chi2 {po[model]['LR stat (self = sibling)']:.2f} on 1 df, "
                  f"p = {po[model]['LR p (self = sibling)']:.3f}")

print("\nbaseline solve rate (%):", {k: round(v, 1) for k, v in strength.items()})
(RUNS / "regression_math3.json").write_text(json.dumps(res, indent=2))
