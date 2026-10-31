#!/bin/bash
# Overnight frontier3 AITA run: attackers write pitches, all three grade (baseline + both frames), then analyze.
# Each stage runs twice: run_jobs resumes, so the second pass only retries jobs that errored.
cd "$(dirname "$0")"
P="python3.12"
for pass in 1 2; do
  echo "=== pass $pass $(date)"
  $P generate.py --pool frontier3 --workers 8
  $P grade.py --pool frontier3 --baseline-only --workers 8
  $P grade.py --pool frontier3 --frame neutral --workers 8
  $P grade.py --pool frontier3 --frame self --workers 8
done
$P analyze.py --pool frontier3 > runs/frontier3/results.txt
$P analyze_controlled.py --pool frontier3 >> runs/frontier3/results.txt
$P regression.py --pool frontier3 >> runs/frontier3/results.txt
echo "=== done $(date)"
