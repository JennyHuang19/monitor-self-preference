# Labels for the math3 CoT study

`cot_study.py analyze` and `cot_study.py export` read these labels with `thimble label NAME --csv`.

- `definitions.json`: every label's kind, its definition (the judge prompt for `prompt` labels), the one-sentence
  description, its values and the counts at the time of the study.
- `false_sum.py`, `ownership.py`, `hedging.py`: the code behind the `code` labels.

To recreate them:

    R=gaslighting-monitors/runs/cot_study/records.jsonl
    thimble label ownership --in $R --code gaslighting-monitors/cot_labels/ownership.py
    thimble label hedging   --in $R --code gaslighting-monitors/cot_labels/hedging.py
    thimble label pitch_false_sum --in 'gaslighting-monitors/runs/math3_mix5__*/answers.jsonl' \
        --where "r['committed']" --code gaslighting-monitors/cot_labels/false_sum.py

The prompt labels (`verify_depth`, `examined_error_step`, `found_then_dropped` over `records.jsonl`;
`pitch_shows_item` over `runs/cot_study/pitches.jsonl`) take `--prompt` with the definition in `definitions.json`.

`values/NAME.csv` holds each label's values as used for the published numbers (ref, value, confidence, by, checked).
