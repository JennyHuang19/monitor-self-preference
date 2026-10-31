"""Pick the 25-post control subset -> runs/<pool>/control/subset.json, and copy those posts' no-pitch baselines.

Only posts where every grader's baseline already matches Reddit (so pushback on a true-side pitch is defined for all
graders). 12 YTA + 13 NTA; main-run flipped posts first, the rest at random (seed 0).

    python select_control.py --pool frontier3
"""
import json
import random

from common import CONTROL, MAIN_RUNS, MODELS, read_jsonl, run_dir

N = {"YTA": 12, "NTA": 13}
assert not CONTROL, "run without --control: this reads the main run"

base = read_jsonl(MAIN_RUNS / "baseline.jsonl")
truth = {b["pid"]: b["truth"] for b in base}
bverd = {(b["pid"], b["grader"]): b["verdict"] for b in base}
all_agree = sorted(pid for pid in truth if all(bverd.get((pid, m["name"])) == truth[pid] for m in MODELS))

flipped = set()
for att in MODELS:
    for fname in ("grades.jsonl", "grades_self.jsonl"):
        for g in read_jsonl(run_dir(att["name"]) / fname):
            b = bverd.get((g["pid"], g["grader"]))
            if b is not None and b != g["target"] and g["verdict"] == g["target"]:
                flipped.add(g["pid"])

rng = random.Random(0)
picked, why = [], {}
for t, n in N.items():
    pool = [p for p in all_agree if truth[p] == t]
    fl = [p for p in pool if p in flipped]
    rest = [p for p in pool if p not in flipped]
    rng.shuffle(fl)
    rng.shuffle(rest)
    take = fl[:n] + rest[:max(0, n - len(fl))]
    picked += take
    why.update({p: ("main-run flip" if p in flipped else "random") for p in take})

out = MAIN_RUNS / "control"
out.mkdir(exist_ok=True)
(out / "subset.json").write_text(json.dumps({"pids": picked, "why": why, "truth": {p: truth[p] for p in picked}},
                                            indent=1))
keep = set(picked)
(out / "baseline.jsonl").write_text("".join(json.dumps(b) + "\n" for b in base if b["pid"] in keep))
for t in N:
    ps = [p for p in picked if truth[p] == t]
    print(f"{t}: {len(ps)} posts, {sum(why[p] == 'main-run flip' for p in ps)} with a main-run flip")
print(f"{sum(b['pid'] in keep for b in base)} baseline rows copied -> {out}")
