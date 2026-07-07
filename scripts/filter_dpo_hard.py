#!/usr/bin/env python3
"""Filter DPO pairs: keep only those with department or urgency mismatch.

Pure symptom-only differences (same dept + same urgency) are too weak
as DPO signals and add noise. This script filters them out.

Usage:
    python filter_dpo_hard.py
"""

from __future__ import annotations

import json
from pathlib import Path

INPUT_TRAIN = "TriageMedical/dpo_train_2.json"
INPUT_VALID = "TriageMedical/dpo_valid_2.json"
OUTPUT_DIR = Path("data/dpo_v4")


def filter_pair(record: dict) -> tuple[bool, str]:
    """Return (keep, reason)."""
    try:
        c = json.loads(record["chosen"]["value"])
        r = json.loads(record["rejected"]["value"])
    except (json.JSONDecodeError, KeyError):
        return False, "parse_error"

    diffs = []
    if c["department"] != r["department"]:
        diffs.append("dept")
    if c["urgency"] != r["urgency"]:
        diffs.append("urgency")
    if diffs:
        return True, "+".join(diffs)
    return False, "symptom_only"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, input_path, output_name in [
        ("train", INPUT_TRAIN, "dpo_train.json"),
        ("valid", INPUT_VALID, "dpo_valid.json"),
    ]:
        in_file = Path(input_path)
        if not in_file.exists():
            print(f"SKIP: {in_file} not found")
            continue

        with in_file.open("r", encoding="utf-8") as f:
            records = json.load(f)

        kept = []
        reasons = {"dept": 0, "urgency": 0, "dept+urgency": 0, "symptom_only": 0, "parse_error": 0}
        for rec in records:
            keep, reason = filter_pair(rec)
            if keep:
                kept.append(rec)
            reasons[reason] = reasons.get(reason, 0) + 1

        out_path = OUTPUT_DIR / output_name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"[{label}] {len(records)} → {len(kept)} pairs kept")
        print(f"  dept only: {reasons['dept']}")
        print(f"  urgency only: {reasons['urgency']}")
        print(f"  dept+urgency: {reasons.get('dept+urgency', 0)}")
        print(f"  dropped (symptom_only): {reasons['symptom_only']}")
        print(f"  dropped (parse_error): {reasons['parse_error']}")
        print(f"  → {out_path}")

    print("\nDone. Output: data/dpo_v4/")


if __name__ == "__main__":
    main()
