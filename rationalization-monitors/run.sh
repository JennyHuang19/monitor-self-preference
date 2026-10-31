#!/bin/bash
# Full rationalization pipeline. Usage: setsid ./run.sh [--limit N]   (CAP=350 USD spend guard)
# Each stage runs the 3 models in parallel; everything resumes from cached keys, so re-running fills in what's missing.
# Monitors run under two prompts: general (primary; not told which defects to look for) and specific (names them;
# skips the swap conditions).
cd "$(dirname "$0")"
PY=python3.12
EXTRA="$*"
CAP=${CAP:-350}
MODELS="kimi-k3 glm-5.3 qwen3.8-2.4t"
KEY=$(grep '^OPENROUTER_API_KEY=' ../.env | cut -d= -f2- | tr -d '"'"'")
used() { curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $KEY" | $PY -c "import json,sys;print(json.load(sys.stdin)['data']['usage'])"; }
START=$(used)
echo "key usage at start: \$$START (cap \$$CAP for this run)"
( while sleep 300; do u=$(used) || continue
    if $PY -c "import sys;sys.exit(0 if $u-$START>$CAP else 1)"; then echo "SPEND CAP HIT: \$$u (start \$$START); stopping"; kill -- -$$; fi
  done ) &
GUARD=$!
stage() {  # stage <script> <flag> [more args]
  s=$1; f=$2; shift 2
  for m in $MODELS; do $PY $s --$f $m "$@" $EXTRA > runs/${s%.py}_$m$(echo "$@" | tr -d ' -').log 2>&1 & done
  wait $(jobs -p | grep -v "^$GUARD$")
  echo "== $s $* done $(date +%T), spend \$$(used)"; tail -qn 4 runs/${s%.py}_*$(echo "$@" | tr -d ' -').log | grep -v "^  ok"
}
stage generate.py agent
stage generate.py agent   # second pass retries truncated / malformed replies
stage verify.py agent
stage swap.py agent
stage monitor.py monitor --prompt general
stage monitor.py monitor --prompt specific
for p in general specific; do
  $PY analyze.py --prompt $p $EXTRA > runs/analyze_$p.txt
  $PY build_viewer.py --prompt $p $EXTRA
done
echo "ALL DONE $(date), total spend this run \$$($PY -c "print(round($(used)-$START,2))")"
kill $GUARD
