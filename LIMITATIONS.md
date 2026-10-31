# Limitations and Responses

The numbers below come from committed result files, cited inline. Intervals are 95%. "OR" is the odds ratio for
being fooled or flipping, and "pp" is the change in that rate in percentage points, both relative to cross-family
pairs.

**1. "The family effect rests on too few families."** The AITA 10-model grid covers five families, each with a
flagship and a smaller sibling: GLM, gpt-oss, Qwen, Gemma and Kimi. The fit spans 16,813 graded rows, 3,310 flips
and 98 posts (`aita/runs/regression.log`).

With attacker, grader and frame fixed effects, graders flip more for their own model's pitch: OR 1.26 [1.13, 1.42],
+3.0 pp [1.4, 4.6]. They also flip more for a sibling's pitch: OR 1.18 [1.05, 1.34], +2.1 pp [0.6, 3.6].

The main alternative explanation is that same-family pairs are simply close in strength. To rule it out, strength
tiers were fixed *before* the new models ran, and every tier mixes families (`aita/common.py`, `TIER`). Controlling
for the tier gap leaves the estimates unchanged (OR 1.26 and 1.18), and the gap term itself is null (OR 1.00
[0.94, 1.06]).

Families were chosen for what the analysis needs, not for convenience:
- The chain-of-thought analyses need the grader's full reasoning. Open-weight models return it, while the Anthropic
  and OpenAI APIs return only a summary.
- A complete attacker × grader grid grows with the square of the number of models.

So closed frontier models (Claude, GPT, Gemini) were run in separate, smaller pools (frontier3, closed2, argon_glm).

*Sensitivity.* The result is weaker under a second strength control. That control measures strength as each
grader's no-pitch agreement with Reddit on the same posts, and adding it removes the pooled effect (self OR 0.99,
sibling OR 1.08). We treat the pre-registered tier gap as primary for two reasons. First, the agreement measure is
estimated from the very data being modelled. Second, it is correlated with family membership across cells
(r = −0.38), so it absorbs part of the effect it is meant to control for. The gpt-oss + Gemma subset, which has two
sibling pairs at matched tiers, keeps the effect under this control: self OR 2.21 [1.24, 4.07], sibling OR 2.25
[1.38, 3.83].

**2. "With one sample per item, flips could be sampling noise."** Each main item set has about 100 items:
- 100 AITA posts (98 analysed)
- 100 math3, 100 trivia3 and 100 math4 problems

Intervals resample whole items, so they account for clustering.

The noise has been measured directly. In the frontier3 control, a second no-pitch sample disagreed with the first on
1 of 75 grader–post pairs (1%) (`aita/runs/frontier3/control/results.txt`). This noise is also independent of who
wrote the pitch. Extra flips from resampling therefore land equally on self, sibling and cross pairs, which pulls
the family contrast toward zero rather than creating it.

math4 removes the issue by construction. It keeps only problems that every model in the pool solved in both of two
no-pitch samples: 106 of 293 candidates passed, and 100 were kept (`gaslighting-monitors/runs/filter_math4.txt`).

**3. "Graders flip because they are unstable, not because they are persuaded."** The frontier3 control tests this
directly. On 25 posts, attackers argued *for* Reddit's verdict instead of against it.

Graders almost never left a correct verdict when pushed toward it (0–7%). On the same posts they flipped to a false
pitch at 0–17%. The discrimination score, (1 − pushback) − flip, is 0.83–0.94 for every grader and frame, and every
lower CI bound is above 0.67. Graders are responding to which side the argument takes, not changing answers at
random.

**4. "Reddit verdicts are noisy ground truth."** The outcome never uses Reddit as the truth. A flip is measured
against the *grader's own* no-pitch verdict. Reddit's label only decides which side the attacker argues. Where it
matters (no-pitch agreement), the 10 models agree with Reddit on 76–91% of posts.

**5. "The diagnosis judge is related to the models it judges."**
- Judges never see model names or who wrote a pitch.
- The diagnosed math pools (mix5, argon_glm) contain no Claude grader, so the Opus judge has no family stake there.
- AITA flips are labelled by three judges from three different families, and reasons are split by their votes.

The judge-agreement pilot covered 100 fooled grades across 62 problems
(`gaslighting-monitors/runs/judge_pilot_report.txt`):
- Opus agrees with its own repeat labels at κ 0.80–0.89.
- On the coarse failure categories, judges from different families agree 89–92% of the time (Fleiss κ 0.56), and
  all three agree on 86%.

The fine six-way taxonomy is less stable across families (Fleiss κ 0.47). Most disagreement sits at the boundary
between "rubber stamp" and "missed in check", so claims about reasons should rest on the coarse categories.

**6. "Dropped parse failures could bias results."** Parse failures are too rare to matter:
- math3 and math4: 0.0–0.4% of grades in each relation, with no pattern by relation.
- AITA: 89 of about 16.8k grades.

**7. "Self-preference may just be style recognition."** Recognizing one's own writing style is a plausible way
self-preference *works*, so it isn't a confound to remove. A monitor that goes easier on text that sounds like its
own is the failure this repo studies. Telling style apart from authorship would take the paraphrase-swap control
(`rationalization-monitors/swap.py`), which was piloted but not run in full.
