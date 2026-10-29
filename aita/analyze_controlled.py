"""Family effect with attacker strength and grader gullibility held fixed.

Pooled self/sibling/cross rates in analyze.py mix in main effects (e.g. gpt-oss graders flip a lot for everyone).
Here, using only the two families with sibling pairs (GLM, gpt-oss):
  interaction = flip(GLM->GLM) - flip(GLM->oss) - flip(oss->GLM) + flip(oss->oss)    [attacker->grader family]
  >0: graders flip more for same-family attackers (H1); <0: less (H2).
Also per grader: same-family attackers minus cross-family attackers; and self minus sibling.
95% CIs bootstrap over posts. Writes runs/controlled.json."""
import json
import math
import random
from collections import defaultdict

from common import MODELS, RUNS, read_jsonl, run_dir

FRAMES = {"neutral": "grades.jsonl", "self": "grades_self.jsonl"}
FAM = {m["name"]: m["family"] for m in MODELS}
PAIR = ("GLM", "gpt-oss")
B = 2000

base = {(b["pid"], b["grader"]): b["verdict"] for b in read_jsonl(RUNS / "baseline.jsonl")}


def cells(frame):
    """(pid) -> list of (attacker, grader, flipped) over eligible items."""
    by_pid = defaultdict(list)
    for att in MODELS:
        for g in read_jsonl(run_dir(att["name"]) / FRAMES[frame]):
            b = base.get((g["pid"], g["grader"]))
            if b is None or b == g["target"] or g["verdict"] is None:  # unparsed grades are dropped
                continue
            by_pid[g["pid"]].append((g["attacker"], g["grader"], g["verdict"] == g["target"]))
    return by_pid


def rate(rows, pred):
    k = n = 0
    for a, g, f in rows:
        if pred(a, g):
            n += 1
            k += f
    return k / n if n else float("nan")


def stats(rows):
    fam = lambda fa, fg: rate(rows, lambda a, g: FAM[a] == fa and FAM[g] == fg)
    out = {"interaction": fam("GLM", "GLM") - fam("GLM", "gpt-oss") - fam("gpt-oss", "GLM") + fam("gpt-oss", "gpt-oss")}
    # Same contrast on log-odds, which is less distorted by the GLM graders sitting near the floor.
    lo = lambda fa, fg: math.log((fam(fa, fg) + 0.005) / (1 - fam(fa, fg) + 0.005))
    out["interaction (log-odds)"] = lo("GLM", "GLM") - lo("GLM", "gpt-oss") - lo("gpt-oss", "GLM") + lo("gpt-oss", "gpt-oss")
    for fa in PAIR:
        for fg in PAIR:
            out[f"{fa}->{fg}"] = fam(fa, fg)
    diffs = []
    for gr in MODELS:
        same = rate(rows, lambda a, g: g == gr["name"] and FAM[a] == gr["family"])
        cross = rate(rows, lambda a, g: g == gr["name"] and FAM[a] != gr["family"])
        out[f"grader {gr['name']}: same-fam - cross"] = same - cross
        if gr["family"] in PAIR:
            diffs.append(same - cross)
    out["mean over GLM/gpt-oss graders: same-fam - cross"] = sum(diffs) / len(diffs)
    sib = [rate(rows, lambda a, g: g == gr["name"] and a == gr["name"]) -
           rate(rows, lambda a, g: g == gr["name"] and a != gr["name"] and FAM[a] == gr["family"])
           for gr in MODELS if gr["family"] in PAIR]
    out["mean over GLM/gpt-oss graders: self - sibling"] = sum(sib) / len(sib)
    return out


res = {}
rng = random.Random(0)
for frame in FRAMES:
    by_pid = cells(frame)
    if not by_pid:
        continue
    pids = sorted(by_pid)
    point = stats([r for p in pids for r in by_pid[p]])
    boots = defaultdict(list)
    for _ in range(B):
        rows = [r for p in (rng.choice(pids) for _ in pids) for r in by_pid[p]]
        for k, v in stats(rows).items():
            boots[k].append(v)
    print(f"\n=== {frame} frame (flip rates; 95% bootstrap CI over {len(pids)} posts) ===")
    res[frame] = {}
    for k, v in point.items():
        bs = sorted(x for x in boots[k] if x == x)
        if not bs:
            continue
        lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]
        res[frame][k] = {"est": v, "lo": lo, "hi": hi}
        print(f"  {k:52} {v:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
(RUNS / "controlled.json").write_text(json.dumps(res, indent=2))
