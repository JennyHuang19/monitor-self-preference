#!/bin/bash
# frontier3 control: attackers argue FOR Reddit's verdict on the 25-post subset (select_control.py), all three grade
# in both frames, plus a second no-pitch baseline as a noise floor. Two passes so errored jobs retry.
cd "$(dirname "$0")"
P="python3.12"
A="--pool frontier3 --control --workers 8"
for pass in 1 2; do
  echo "=== pass $pass $(date)"
  $P generate.py $A
  $P grade.py $A --baseline-rerun
  $P grade.py $A --frame neutral
  $P grade.py $A --frame self
done
$P analyze_control.py --pool frontier3 > runs/frontier3/control/results.txt
echo "=== done $(date)"
