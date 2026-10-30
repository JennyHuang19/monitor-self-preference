#!/bin/bash
# math4 on the argon_glm pool (gemini-4-argon, glm-5.3, glm-5.3-flash). Usage: ./run_argon_glm.sh [--limit N]
# Needs questions_math4_candidates.json (filter_solved.py candidates). Then:
# 1) no-pitch baseline, 2 samples per candidate x model, 2) keep the candidates every model solved both times,
# 3) one process per attacker: generate, then grade in each frame. Everything resumes from cached keys.
cd "$(dirname "$0")"
PY=python3.12
EXTRA="$*"
POOL=argon_glm
ATTACKERS="gemini-4-argon glm-5.3 glm-5.3-flash"
$PY grade.py --domain math4 --pool $POOL --baseline-only --baseline-samples 2 \
  --questions questions_math4_candidates.json > runs/base_math4_${POOL}.log 2>&1 || exit 1
$PY filter_solved.py filter --samples 2 || exit 1
for att in $ATTACKERS; do
  (
    $PY generate.py --domain math4 --pool $POOL --attacker $att $EXTRA
    for fr in neutral self skim1k; do
      $PY grade.py --domain math4 --pool $POOL --attacker $att --frame $fr $EXTRA
    done
  ) > runs/run_${POOL}_math4_${att}.log 2>&1 &
done
wait
echo ALL DONE
