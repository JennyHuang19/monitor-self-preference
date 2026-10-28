#!/bin/bash
# math3 + trivia3 on the mix5 pool. Usage: ./run_mix5.sh [--limit N]
# 1) the shared no-pitch baseline per domain, 2) one process per attacker x domain: generate, then grade in each frame.
# Everything resumes from cached keys, so re-running only fills in what's missing.
cd "$(dirname "$0")"
PY=python3.12
EXTRA="$*"
ATTACKERS="glm-5.3 glm-5.3-flash kimi-k3 gpt-oss-120b gpt-oss-20b"
for dom in math3 trivia3; do
  $PY grade.py --domain $dom --pool mix5 --baseline-only $EXTRA > runs/base_${dom}_mix5.log 2>&1 &
done
wait
for att in $ATTACKERS; do
  for dom in math3 trivia3; do
    (
      $PY generate.py --domain $dom --pool mix5 --attacker $att $EXTRA
      for fr in neutral self skim1k; do
        $PY grade.py --domain $dom --pool mix5 --attacker $att --frame $fr $EXTRA
      done
    ) > runs/run_mix5_${dom}_${att}.log 2>&1 &
  done
done
wait
echo ALL DONE
