#!/bin/bash
# Unattended math4 / argon_glm run: screen candidates as question batches arrive, top up with new batches until
# 100 problems pass the solve filter (at most MAX_ROUNDS extra rounds of 10 batches), then attack and grade.
# Log: runs/full_argon_glm.log. Safe to rerun: every stage resumes from cached keys.
cd "$(dirname "$0")"
PY=python3.12
POOL=argon_glm
MAX_ROUNDS=${MAX_ROUNDS:-3}
rounds=0
gen_running() { pgrep -u "$(id -u)" -f "make_questions.py --domain math4" > /dev/null; }
screen() {
  $PY filter_solved.py candidates || return 1
  $PY grade.py --domain math4 --pool $POOL --baseline-only --baseline-samples 2 \
    --questions questions_math4_candidates.json >> runs/base_math4_${POOL}.log 2>&1 || return 1
  $PY filter_solved.py filter --samples 2 | tee runs/filter_math4.txt
}
while true; do
  echo "== screen $(date)"
  screen || { echo "screen failed"; exit 1; }
  npass=$(awk '/ pass; kept /{print $1}' runs/filter_math4.txt)
  echo "pass: $npass"
  [ "${npass:-0}" -ge 100 ] && break
  if gen_running; then sleep 120; continue; fi
  # Generation idle and still short: one more round, unless this round's batches were just screened already.
  if [ "$rounds" -ge "$MAX_ROUNDS" ]; then echo "stopping top-ups after $rounds rounds with $npass passing"; break; fi
  last=$(ls runs/questions_math4_raw_*.json | sed 's/.*_raw_\([0-9]*\)\.json/\1/' | sort -n | tail -1)
  batches=$(seq -s, $((last + 1)) $((last + 10)))
  rounds=$((rounds + 1))
  echo "round $rounds: generating batches $batches"
  $PY make_questions.py --domain math4 --batches "$batches" >> runs/make_math4.log 2>&1 &
  sleep 60
done
echo "== attack + grade $(date)"
for att in gemini-4-argon glm-5.3 glm-5.3-flash; do
  (
    $PY generate.py --domain math4 --pool $POOL --attacker $att
    for fr in neutral self skim1k; do
      $PY grade.py --domain math4 --pool $POOL --attacker $att --frame $fr
    done
  ) > runs/run_${POOL}_math4_${att}.log 2>&1 &
done
wait
$PY analyze_mix5.py $POOL > runs/analyze_${POOL}.txt 2>&1
echo "ALL DONE $(date)"
