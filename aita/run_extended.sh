#!/bin/zsh
# Fill the 10x10 grid after adding Qwen, Gemma and kimi-k3.1-pro. Resumable: run_jobs skips keys already on disk.
#     zsh run_extended.sh > runs/extended.log 2>&1
cd "$(dirname "$0")"
PY=~/.local/bin/python3
NEW=(qwen3.8-max qwen3.8-flash gemma-4-31b gemma-4-26b-a4b kimi-k3.1-pro)
ALL=(glm-5.3 glm-5.3-flash gpt-oss-120b gpt-oss-20b kimi-k3 $NEW)

echo "== pitches from the new attackers"
for a in $NEW; do $PY generate.py --attacker $a --workers 8 > runs/log_gen_$a.txt 2>&1 & done
# Baseline has one shared file, so it runs in a single process alongside the pitches.
$PY grade.py --baseline-only --workers 16 > runs/log_baseline.txt 2>&1 &
wait
tail -n 1 runs/log_gen_*.txt runs/log_baseline.txt

echo "== grading: one process per (attacker, frame), each writing its own file"
for a in $ALL; do
  for f in neutral self; do
    $PY grade.py --attacker $a --frame $f --workers 6 > runs/log_grade_${a}_$f.txt 2>&1 &
  done
done
wait
tail -n 1 runs/log_grade_*.txt

echo "== analysis"
$PY analyze.py > runs/results.txt && $PY analyze_controlled.py && $PY fit_regression.py && $PY build_viewer.py
echo ALL_DONE
