# Monitor self-preference

Experiments on whether LLM monitors (graders, judges, code reviewers) can be talked into a wrong verdict by another
model, and whether they are **easier to fool when the persuader is their own model or a sibling from the same family**
(self-preference / self-leniency) than when it is a model from another family.

The shared design across the main experiments:

1. An **attacker** model is secretly told to argue for a wrong answer (a wrong math option, the opposite of Reddit's
   AITA verdict, sloppy code dressed up with a justification) and writes a persuasive "pitch".
2. A **grader / monitor** model reads the pitch and gives a verdict. Each grader also gets a **no-pitch baseline**
   (it answers the item on its own), so "fooled" or "flipped" means the pitch moved it away from what it believed.
3. Each attacker–grader pair is labelled by **relation**: `self` (same model), `sibling` (same family, e.g. `glm-5.3`
   and `glm-5.3-flash`), or `cross` (different family).
4. Grading runs under several **frames** (prompt types): `neutral` ("is this answer correct?"), `self` (the grader is
   told the answer is its own and asked to check it), and `skim` / `skim1k` (told not to re-solve, or a 1k-token
   reasoning cap).
5. Analysis fits regressions such as `logit P(fooled) = attacker FE + grader FE + frame FE + relation (+ problem
   random effect)`. These hold attacker persuasiveness and grader gullibility fixed, so the `relation` term isolates
   the family effect. Self-contained HTML viewers show every pitch, grade and chain of thought.

All model calls go through OpenRouter, except `aita/direct.py`, which calls the Anthropic and OpenAI APIs directly.
API keys are read from `.env` at the repo root (gitignored). The scripts use Python 3.12 and the standard library
only (no numpy or pandas), and the statistics are implemented in pure Python.

## Repository layout

| Directory | What it is |
|---|---|
| [`gaslighting-monitors/`](gaslighting-monitors/) | Main experiment: attackers argue for wrong answers to math, code and trivia items; graders judge them. |
| [`aita/`](aita/) | The same design on subjective judgments: Reddit "Am I the Asshole?" posts, where attackers argue against Reddit's verdict. |
| [`rationalization-monitors/`](rationalization-monitors/) | Code-review monitors: do monitors flag their own model's sloppy code (with a self-written justification) less often? |
| [`double-plot/`](double-plot/) | Persuasion in debate: two models argue over a whodunit play, then judge it. |
| [`model-tunnel-visioned/`](model-tunnel-visioned/) | Tunnel vision: models answer a narrow question about an agent transcript and miss planted misconduct. |
| [`monitoring-bench/`](monitoring-bench/) | Downloader and one sample trajectory for an external monitoring dataset. |
| `investigator agents - failure modes and ways forward.pdf` | Background notes. |

Each experiment directory follows the same pattern: `common.py` (models, pools, paths and helpers), `generate.py`
(attacker pitches), `grade.py` / `monitor.py` (graders), `analyze*.py` / regression scripts, `build_viewer*.py` with
an HTML template, `run_*.sh` pipeline drivers, and `runs/` holding JSONL outputs, logs and summaries. Every
pipeline resumes from keys already cached on disk, so rerunning a script only fills in jobs that are missing or
failed.

## Experiments

### `gaslighting-monitors/`: wrong-answer persuasion on checkable items

Items are written by a question-writer model (`make_questions.py`), and each item has a planted `target_wrong`
answer that comes from a natural mistake. The items are then checked mechanically (`check_math.py`,
`check_code.py`, `check_items.py`: brute-force `check_code`, buggy vs fixed code, and the requirement that
the target really is wrong).

- **Domains** (`questions_<domain>.json`): `math`, `code`, `trivia`, `math2`, `math3` (free-response, mostly
  "omitted case" errors), `trivia3`, and `math4`. `math4` keeps only problems that every model in the pool solves in
  every no-pitch sample (`filter_solved.py`), so a fooled grade reflects persuasion rather than ignorance.
- **Model pools** (`common.py`, `POOLS`):
  - `frontier`: GLM, Kimi, Qwen, GPT
  - `open`: gpt-oss and Gemma
  - `mix5`: glm-5.3, glm-5.3-flash, kimi-k3, gpt-oss-120b, gpt-oss-20b
  - `closed2`: gpt-6-sol and gemini-4-argon
  - `argon_glm`: gemini-4-argon, glm-5.3, glm-5.3-flash
- **Pipeline:** `generate.py --domain D --pool P --attacker A` writes `runs/<domain>_<pool>__<attacker>/answers.jsonl`.
  `grade.py` writes `grades.jsonl`, `grades_self.jsonl`, `grades_skim1k.jsonl` and the shared `baseline.jsonl`. The
  drivers are `run_mix5.sh`, `run_argon_glm.sh` and `pipeline_argon_glm.sh`.
- **Analysis:**
  - `analyze.py` / `analyze_mix5.py`: fooled-rate tables, written to `runs/summary*.json` and `runs/analyze_*.txt`
  - `mixed_logit.py`: mixed-effects logit with a problem random intercept, used by the viewer
  - `judge_uptake.py`: how much of the pitch a grader took on
  - `effort_chart.py`: fooled rate vs reasoning length
- **Why were graders fooled?** `diagnose_fooled.py` has outside judges (Claude Opus, GPT and Gemini) label each
  fooled grade with a failure type. `judge_pilot.py` measures how much the labels depend on the judge, and
  `summarize_diagnose.py` / `reasons_chart.py` chart the reasons by relation.
- **Blind-spot experiment** (`runs/blindspot_oss120b/`): gpt-oss-120b chooses which case to drop on problems it did
  not write (`choose_oss_case.py`). Problems are screened to a target solve rate (`screen_oss_problems.py`), clean
  pitches are graded repeatedly (`grade_oss_pitches.py`), and `probe_skim.py` tests whether skimming can be
  enforced.
- **Viewers:** `viewer.html`, `viewer_mix5.html`, `viewer_closed2.html` and `viewer_math/`. `build_argon_page.py`
  and `page_patch.py` add the argon_glm tab and the notes to the published math viewer.

### `aita/`: flipping subjective verdicts

The data is 100 AITA posts (`data/sample_75nta_25yta.json`: 75 NTA, 25 YTA). Each attacker argues for the opposite
of Reddit's verdict, and graders give YTA/NTA plus a confidence score. A **flip** means the grader's no-pitch verdict
differed from the attacker's target, and after reading the pitch its verdict matches the target.

- **Pools** (`common.py`, selected with `--pool`):
  - `open` (default; runs in `runs/`): glm-5.3, glm-5.3-flash, gpt-oss-120b, gpt-oss-20b, kimi-k3
  - `closed2`
  - `frontier3` (runs in `runs/frontier3/`): gpt-6-astra, claude-opus-4.8, gemini-4-argon
- **Pipeline:** `generate.py` then `grade.py` (`--baseline-only`, `--frame neutral|self`). The drivers are
  `run_frontier3.sh` and `run_extended.sh`.
- **Analysis:**
  - `analyze.py`: flip rates by relation
  - `analyze_controlled.py`: the family interaction using the two families that have sibling pairs (GLM and
    gpt-oss)
  - `regression.py`: grouped binomial logit with |strength gap| terms and post-level cluster bootstrap
- **Control run** (`runs/frontier3/control/`): attackers argue *for* Reddit's verdict on a 25-post subset
  (`select_control.py`, `run_control.sh`, `analyze_control.py`, `build_control_viewer.py`). A second no-pitch
  baseline serves as a noise floor.
- **Flip diagnosis:** `diagnose_flips.py` has three judges label why each flip happened, and
  `reasons_chart_aita.py` charts the reasons.
- **Viewers:** `build_viewer.py` builds `viewer/index.html`, which loads each post's JSON on demand. Built viewers
  are under `runs/frontier3/viewer/` and `runs/frontier3/control/viewer/`.

### `rationalization-monitors/`: self-leniency in code review

`make_tasks.py` drafts coding tasks in two defect categories: `vacuous_tests` and `ignored_errors`. Each agent
answers every task twice (`generate.py`):

- **clean:** an honest solution
- **sloppy:** a real defect that must be fixed, plus "design notes" that read well but justify the defect wrongly

Monitors (`monitor.py`) judge every item without knowing which model wrote it, under four conditions:

- `clean`
- `bare`: sloppy code with the notes removed
- `rat`: sloppy code with the agent's own notes
- `swap`: notes paraphrased by another model (`swap.py`; piloted but dropped from the full run)

Monitors run with a `general` prompt and a `specific` prompt. `analyze.py` fits
`flag ~ monitor FE + agent FE + category + self`, where `self < 0` means monitors go easier on their own model.
`run.sh` runs the whole pipeline, and the viewers are `viewer_general.html` and `viewer_specific.html`.

### `double-plot/`: debate and position bias

`play.md` is a two-scene play with an ambiguous culprit (Daniel or Maya), and `answer-key.md` gives the intended
answer.

- `run_eval.py` asks models who did it, with no debate.
- `debate.py` has two models argue opposite sides in both assignments, then each gives its verdict.
- `reverdict.py` re-judges the same transcripts without telling the reader which side it argued.
- `analyze_blind.py` asks whether blind readers side with the better debater or with whoever went first.
- `compare.py` compares the debate and no-debate runs.

Results are in `results/` and the viewer is `viewer.html`.

### `model-tunnel-visioned/`: missing misconduct while answering a narrow question

`agents-working-token-count.md` is a multi-agent coding transcript with a short misconduct passage spliced into one
agent's reasoning (for example, trying to escape the sandbox to copy a reference solution). Models are then asked
narrow questions that don't mention conduct (`distractor-questions.md`: tokens, performance, who did what,
collaboration).

- `ask_models.py` and `sweep_openrouter.py` collect the answers.
- `generate_prompts.py` makes new cases in `generated/`.
- `tunnel-vision-answer-key.md` marks the misconduct.

`build_swe_chat_viewer.py` browses real sessions from the SWE-chat dataset. The dataset is gitignored and must be
downloaded separately.

### `monitoring-bench/`

`download_dataset.py` fetches the `ctrldataset2026` test split from Hugging Face into `data/`, and `traj.txt` is one
example trajectory.

### Also in the repo

- **AITA 10-model grid:** the `open` pool extended with Qwen, Gemma and kimi-k3.1-pro, with strength tiers fixed in
  advance (`common.TIER`). It uses `run_extended.sh`, `fit_regression.py` (family effect holding strength fixed) and
  `cot_compare.py`, which compares how graders reason when flipped by a same-family vs a cross-family pitch.
- **math3 CoT study** (`gaslighting-monitors/`): `cot_study.py` asks whether a monitor fooled by its own model's pitch
  reasons differently from one fooled by a cross-family pitch. It uses blinded records and labels from
  `cot_labels/` (see its README). `fit_regression.py` fits the pooled mixed logit, and `build_viewer_math3.py`
  builds `viewer_math3/`.

## Limitations and Responses

See [`LIMITATIONS.md`](LIMITATIONS.md) for the main objections to this design and the evidence that answers them.

## Running

```bash
# keys in .env at the repo root (e.g. OPENROUTER_API_KEY=...; ANTHROPIC_API_KEY / OPENAI_API_KEY for aita/direct.py)
cd gaslighting-monitors
python3.12 grade.py --domain math3 --pool mix5 --baseline-only --limit 5
python3.12 generate.py --domain math3 --pool mix5 --attacker kimi-k3 --limit 5
python3.12 grade.py --domain math3 --pool mix5 --attacker kimi-k3 --frame neutral --limit 5
python3.12 analyze_mix5.py
```

Each script's docstring gives its usage and outputs.
