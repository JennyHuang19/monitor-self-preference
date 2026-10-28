"""Download the ctrldataset2026 test split from Hugging Face and save it locally."""

import argparse
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "neur26anonsub/ctrldataset2026"
SPLIT = "test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory to save the dataset to (default: monitoring-bench/data)",
    )
    args = parser.parse_args()

    ds = load_dataset(DATASET_ID, split=SPLIT)
    print(ds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / f"ctrldataset2026_{SPLIT}.jsonl"
    ds.to_json(jsonl_path, lines=True)
    print(f"Saved {len(ds)} rows to {jsonl_path}")


if __name__ == "__main__":
    main()
