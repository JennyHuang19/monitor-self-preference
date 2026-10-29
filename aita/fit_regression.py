"""Family effect from one logistic regression over both grading frames that holds each model's strength fixed.

    logit P(flip) = attacker FE + grader FE + frame FE + grader x frame + relation (self / sibling vs cross)
                    [+ |strength gap|]

Attacker FE absorbs how persuasive each attacker is overall; grader FE absorbs how easily each grader flips; frame FE
absorbs how much the prompt type (neutral vs self-check; neutral is the reference) moves flips overall, and
grader x frame lets each grader react to it its own way (without it the fit misfits badly: Pearson chi2/df 4.8 vs
1.6, mostly gpt-oss-20b flipping far more under self-check and Gemma far less). What is left
in `relation` is whether a grader flips more (H1) or less (H2) for pitches from its own family. Models B and C add
|gap|, the absolute strength difference, so that "close match" is not read as "same family". B measures strength by
no-pitch agreement with Reddit; C by the strength tier fixed before the run (common.TIER). A signed gap would be
absorbed exactly by the two FEs, so only |gap| is used. "A x frame" adds relation x frame terms, so the family effect
can differ by frame, and a likelihood-ratio test says whether it does.

No post effects (many posts never flip, which would separate). Rows from one post move together, so 95% CIs come
from a cluster bootstrap over posts; a resample takes both frames of a post together, since each frame grades the
same pitch. With no post terms the fit only depends on the 200 frame x attacker x grader cell counts, so each
bootstrap refit is a 200-row binomial GLM.

Everything is fit twice: on all 10 models ("pooled") and on the four open models only, gpt-oss and Gemma ("open"),
whose two families each have a sibling pair at matching tiers. Writes runs/regression.json.
"""
import json
import math
import random
from collections import defaultdict

from common import MODELS, RUNS, TIER, read_jsonl, relation, run_dir

FRAMES = {"neutral": "grades.jsonl", "self": "grades_self.jsonl"}
FR = list(FRAMES)  # FR[0] is the reference frame
FRAME_NAME = {"neutral": "neutral", "self": "self-check"}  # "self" alone reads as the relation
ALL = [m["name"] for m in MODELS]
NAMES = ALL  # the models in the current fit; analyze() sets it
B = 2000
RIDGE = 1e-3  # keeps a bootstrap fit finite if a resample leaves a whole row or column at zero flips

baseline = read_jsonl(RUNS / "baseline.jsonl")
base = {(b["pid"], b["grader"]): b["verdict"] for b in baseline}
# Strength proxy, fixed before looking at any pitch: % of posts where the no-pitch verdict matches Reddit.
strength = {n: 100 * sum(b["matches_reddit"] for b in baseline if b["grader"] == n) /
            sum(b["grader"] == n for b in baseline) for n in ALL}
REL = {(a, g): relation(next(m for m in MODELS if m["name"] == g), next(m for m in MODELS if m["name"] == a))
       for a in ALL for g in ALL}
PAIRS = [(a, g) for a in NAMES for g in NAMES]
CELLS = [(f, a, g) for f in FR for a, g in PAIRS]


# |gap| per model: B in units of 10 points of Reddit agreement, C in tiers.
GAPS = {"B": lambda a, g: abs(strength[a] - strength[g]) / 10, "C": lambda a, g: abs(TIER[a] - TIER[g])}
MODELS_FIT = (("A: fixed effects", None, False), ("B: + |strength gap|", "B", False), ("C: + |tier gap|", "C", False),
              ("A × frame", None, True))


def columns(with_gap, inter=False):
    """Column name -> index: intercept, attacker FE, grader FE (first model is the reference), frame FE, self,
    sibling[, |gap|][, relation x frame]."""
    names = (["intercept"] + [f"attacker {n}" for n in NAMES[1:]] + [f"grader {n}" for n in NAMES[1:]]
             + [f"frame {f}" for f in FR[1:]] + [f"grader {n} x frame {f}" for f in FR[1:] for n in NAMES[1:]]
             + ["self", "sibling"] + (["|gap|"] if with_gap else [])
             + ([f"{r} x {f}" for r in ("self", "sibling") for f in FR[1:]] if inter else []))
    return {c: i for i, c in enumerate(names)}


def design(f, a, g, with_gap, inter=False):
    x = [1.0] + [float(a == n) for n in NAMES[1:]] + [float(g == n) for n in NAMES[1:]] + [float(f == n) for n in FR[1:]]
    x += [float(f == fr and g == n) for fr in FR[1:] for n in NAMES[1:]]  # each grader reacts to the frame its own way
    x += [float(REL[a, g] == "self"), float(REL[a, g] == "sibling")]
    if with_gap:
        x.append(GAPS[with_gap](a, g))
    if inter:
        x += [float(REL[a, g] == r and f == fr) for r in ("self", "sibling") for fr in FR[1:]]
    return x


def sparse(x):
    return [(i, v) for i, v in enumerate(x) if v]  # the design is mostly zeros


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def log1pexp(x):
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def fit(counts, with_gap, inter=False, beta=None):
    """Binomial logistic regression by Newton steps. counts: (f, a, g) -> [flips, eligible]. Returns (beta, loglik)."""
    X = {c: sparse(design(*c, with_gap, inter)) for c in CELLS if counts[c][1]}
    k = len(design(*CELLS[0], with_gap, inter))
    beta = list(beta) if beta else [0.0] * k
    for _ in range(50):
        grad = [-RIDGE * b for b in beta]
        H = [[RIDGE * (i == j) for j in range(k)] for i in range(k)]
        for c, xs in X.items():
            y, n = counts[c]
            p = 1 / (1 + math.exp(-sum(beta[i] * v for i, v in xs)))
            w = n * p * (1 - p)
            for i, vi in xs:
                grad[i] += (y - n * p) * vi
                for j, vj in xs:
                    H[i][j] += w * vi * vj
        step = solve(H, grad)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    ll = 0.0
    for c, xs in X.items():
        y, n = counts[c]
        eta = sum(beta[i] * v for i, v in xs)
        ll -= y * log1pexp(-eta) + (n - y) * log1pexp(eta)
    return beta, ll


def expit(x):
    return 1 / (1 + math.exp(-x))


def pp(beta, counts, with_gap, inter, keep, drop):
    """Average change in flip probability, in points, on the cells `keep` selects if the `drop` columns were off."""
    num = den = 0.0
    for c in CELLS:
        n = counts[c][1]
        if not n or not keep(c):
            continue
        x = design(*c, with_gap, inter)
        eta = sum(b * v for b, v in zip(beta, x))
        num += n * (expit(eta) - expit(eta - sum(beta[i] * x[i] for i in drop)))
        den += n
    return 100 * num / den


def summarize(beta, counts, with_gap, inter=False):
    """Additive models: relation and frame effects (OR, and pp on the cells that have them). Interaction model: the
    self / sibling effect within each frame."""
    ix, out = columns(with_gap, inter), {}
    if not inter:
        for r in ("self", "sibling"):
            out[f"OR {r} vs cross"] = math.exp(beta[ix[r]])
            out[f"pp {r} vs cross"] = pp(beta, counts, with_gap, inter, lambda c: REL[c[1:]] == r, [ix[r]])
        for f in FR[1:]:
            # Each grader has its own frame shift, so there is no one frame OR: report the average shift in points.
            drop = [i for c, i in ix.items() if c == f"frame {f}" or c.endswith(f" x frame {f}")]
            out[f"pp frame {FRAME_NAME[f]} vs {FR[0]}"] = pp(beta, counts, with_gap, inter, lambda c: c[0] == f,
                                                             drop)
        if with_gap:
            out["OR per unit |gap|"] = math.exp(beta[ix["|gap|"]])
        return out
    for f in FR:
        for r in ("self", "sibling"):
            drop = [ix[r]] + ([ix[f"{r} x {f}"]] if f != FR[0] else [])
            out[f"{f}|OR {r} vs cross"] = math.exp(sum(beta[i] for i in drop))
            out[f"{f}|pp {r} vs cross"] = pp(beta, counts, with_gap, inter,
                                             lambda c: c[0] == f and REL[c[1:]] == r, drop)
    return out


def chi2_sf(x, df):
    """Upper tail of a chi-square with an even df."""
    return math.exp(-x / 2) * sum((x / 2) ** k / math.factorial(k) for k in range(df // 2))


def corr(xs, ys, ws):
    mx, my, sw = (sum(w * x for x, w in zip(xs, ws)) / sum(ws), sum(w * y for y, w in zip(ys, ws)) / sum(ws),
                  sum(ws))
    cov = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sw
    vx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws)) / sw
    vy = sum(w * (y - my) ** 2 for y, w in zip(ys, ws)) / sw
    return cov / math.sqrt(vx * vy)


per_post = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # pid -> (f, a, g) -> [flips, eligible]
for f, fname in FRAMES.items():
    for a in ALL:
        for g in read_jsonl(run_dir(a) / fname):
            b = base.get((g["pid"], g["grader"]))
            if b is None or b == g["target"] or g["verdict"] is None:  # unparsed grades are dropped
                continue
            cell = per_post[g["pid"]][(f, a, g["grader"])]
            cell[0] += g["verdict"] == g["target"]
            cell[1] += 1
pids = sorted(per_post)


def total(sample):
    t = {c: [0, 0] for c in CELLS}
    for p in sample:
        for c, (y, n) in per_post[p].items():
            if c in t:  # cells outside the current set of models are skipped
                t[c][0] += y
                t[c][1] += n
    return t


def ci(xs, v):
    xs = sorted(xs)
    return {"est": v, "lo": xs[int(0.025 * len(xs))], "hi": xs[int(0.975 * len(xs)) - 1]}


def analyze(names, label):
    """Fit every model in MODELS_FIT on the cells among `names`, with bootstrap CIs. Returns the summary dict."""
    global NAMES, PAIRS, CELLS
    NAMES, PAIRS = names, [(a, g) for a in names for g in names]
    CELLS = [(f, a, g) for f in FR for a, g in PAIRS]
    counts = total(pids)
    po = {"models": names, "n_posts": len(pids), "n_rows": sum(n for _, n in counts.values()),
          "n_flips": sum(y for y, _ in counts.values()),
          "by frame": {f: [sum(counts[c][0] for c in CELLS if c[0] == f),
                           sum(counts[c][1] for c in CELLS if c[0] == f)] for f in FR}}
    # How entangled same-family and |gap| are across the cells (weighted by eligible rows).
    ws = [counts[c][1] for c in CELLS]
    for k in GAPS:
        po[f"corr same-family vs |gap| {k}"] = corr([float(REL[c[1:]] != "cross") for c in CELLS],
                                                     [GAPS[k](*c[1:]) for c in CELLS], ws)
    print(f"\n##### {label}: {', '.join(names)}")
    print(f"=== both frames ({po['n_rows']} rows, {po['n_flips']} flips, {len(pids)} posts) ===")
    print("  by frame:", ", ".join(f"{f} {y}/{n}" for f, (y, n) in po["by frame"].items()))

    rng = random.Random(0)
    fits = {model: fit(counts, gap, inter) for model, gap, inter in MODELS_FIT}
    boots = {model: defaultdict(list) for model, _, _ in MODELS_FIT}
    for _ in range(B):
        bc = total([rng.choice(pids) for _ in pids])
        for model, gap, inter in MODELS_FIT:
            for k, v in summarize(fit(bc, gap, inter, fits[model][0])[0], bc, gap, inter).items():
                boots[model][k].append(v)

    for model, with_gap, inter in MODELS_FIT:
        beta = fits[model][0]
        point = summarize(beta, counts, with_gap, inter)
        print(f"\n=== {model} ===")
        if inter:
            lr = max(0.0, 2 * (fits[model][1] - fits["A: fixed effects"][1]))
            df = 2 * (len(FR) - 1)
            po[model] = {"LR stat": lr, "df": df, "LR p (no interaction)": chi2_sf(lr, df), "by frame": {}}
            for k, v in point.items():
                f, name = k.split("|")
                po[model]["by frame"].setdefault(f, {})[name] = ci(boots[model][k], v)
                print(f"  {f:8} {name:20} {v:8.3f}  [{po[model]['by frame'][f][name]['lo']:.3f}, "
                      f"{po[model]['by frame'][f][name]['hi']:.3f}]")
            print(f"  LR test of relation x frame: chi2 {lr:.2f} on {df} df, p = {po[model]['LR p (no interaction)']:.3f}")
            continue
        po[model] = {k: ci(boots[model][k], v) for k, v in point.items()}
        for k, v in po[model].items():
            print(f"  {k:30} {v['est']:8.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]")
        if with_gap is None:
            # Where the additive model misfits: each frame cell's Pearson contribution, largest first.
            worst = []
            for (f, a, g), (y, n) in counts.items():
                if n:
                    p = expit(sum(b * v for b, v in zip(beta, design(f, a, g, None))))
                    worst.append([f, a, g, REL[a, g], y, n, n * p, (y - n * p) ** 2 / (n * p * (1 - p))])
            worst.sort(key=lambda r: -r[7])
            dof = len(worst) - len(beta)
            po["pearson/df"] = sum(r[7] for r in worst) / dof
            po["worst cells"] = worst[:6]
            po["excess top6"] = sum(r[7] - 1 for r in worst[:6]) / (sum(r[7] for r in worst) - dof)
    print(f"\ncorr(same-family, |gap|) across cells: B {po['corr same-family vs |gap| B']:+.2f}, "
          f"C {po['corr same-family vs |gap| C']:+.2f}")
    print(f"Pearson chi2/df (model A): {po['pearson/df']:.2f}")
    return po


OPEN = ["gpt-oss-120b", "gpt-oss-20b", "gemma-4-31b", "gemma-4-26b-a4b"]
res = {"strength": strength, "tier": TIER, "B": B, "frames": FR,
       "pooled": analyze(ALL, "all 10 models"), "open": analyze(OPEN, "gpt-oss + Gemma only")}
print("\nstrength proxy (no-pitch agreement with Reddit, %):", {k: round(v, 1) for k, v in strength.items()})
(RUNS / "regression.json").write_text(json.dumps(res, indent=2))
