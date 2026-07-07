"""Clean teacher-labeled triage data into SFT train/valid files."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import (
    ALLOWED_DEPARTMENTS,
    ALLOWED_URGENCY,
    DEPARTMENT_MAP,
    DROP_DEPARTMENTS,
    SYSTEM_PROMPT,
)
from .io_utils import read_json_list, write_json
from .schema import compact_triage_json, normalize_symptoms_for_schema, parse_json_object


def normalize_row(row: dict[str, Any], stats: Counter[str]) -> dict[str, Any] | None:
    input_text = row.get("input")
    if not isinstance(input_text, str) or not input_text.strip():
        stats["drop_bad_input"] += 1
        return None

    parsed = parse_json_object(row.get("output"))
    if parsed is None:
        stats["drop_bad_json"] += 1
        return None

    missing = {"department", "urgency", "symptoms"} - set(parsed)
    if missing:
        stats["drop_missing_keys"] += 1
        return None

    department = parsed.get("department")
    if not isinstance(department, str):
        stats["drop_bad_department_type"] += 1
        return None
    department = department.strip()
    if department in DEPARTMENT_MAP:
        stats[f"map_department:{department}->{DEPARTMENT_MAP[department]}"] += 1
        department = DEPARTMENT_MAP[department]
    elif department in DROP_DEPARTMENTS:
        stats[f"drop_department:{department}"] += 1
        return None
    elif department not in ALLOWED_DEPARTMENTS:
        stats[f"drop_unknown_department:{department}"] += 1
        return None

    urgency = parsed.get("urgency")
    if not isinstance(urgency, str) or urgency.strip() not in ALLOWED_URGENCY:
        stats["drop_bad_urgency"] += 1
        return None
    urgency = urgency.strip()

    symptoms = normalize_symptoms_for_schema(parsed.get("symptoms"))
    if symptoms is None:
        stats["drop_bad_symptoms"] += 1
        return None

    output = {"department": department, "urgency": urgency, "symptoms": symptoms}
    return {
        "instruction": SYSTEM_PROMPT,
        "input": input_text.strip(),
        "output": compact_triage_json(output),
    }


def deduplicate(rows: list[dict[str, Any]], stats: Counter[str]) -> list[dict[str, Any]]:
    by_input: dict[str, dict[str, Any]] = {}
    conflict_count = 0
    for row in rows:
        key = row["input"]
        existing = by_input.get(key)
        if existing is None:
            by_input[key] = row
            continue
        stats["drop_duplicate_input"] += 1
        if existing["output"] != row["output"]:
            conflict_count += 1
    stats["duplicate_output_conflicts"] = conflict_count
    return list(by_input.values())


def split_rows(
    rows: list[dict[str, Any]], valid_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    valid_size = max(1, round(len(shuffled) * valid_ratio))
    return shuffled[valid_size:], shuffled[:valid_size]


def build_clean_report(rows: list[dict[str, Any]], stats: Counter[str]) -> dict[str, Any]:
    dept_counter: Counter[str] = Counter()
    urgency_counter: Counter[str] = Counter()
    for row in rows:
        output = json.loads(row["output"])
        dept_counter[output["department"]] += 1
        urgency_counter[output["urgency"]] += 1
    return {
        "num_clean_rows": len(rows),
        "department_distribution": dict(dept_counter.most_common()),
        "urgency_distribution": dict(urgency_counter.most_common()),
        "cleaning_stats": dict(stats.most_common()),
    }


def clean_sft_data(
    input_files: list[str | Path],
    output_dir: str | Path = "data/processed",
    *,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    stats: Counter[str] = Counter()
    normalized_rows: list[dict[str, Any]] = []

    for input_file in input_files:
        path = Path(input_file)
        raw_rows = read_json_list(path)
        stats[f"source_rows:{path.name}"] = len(raw_rows)
        for row in raw_rows:
            normalized = normalize_row(row, stats)
            if normalized is not None:
                normalized_rows.append(normalized)

    all_clean = deduplicate(normalized_rows, stats)
    train_rows, valid_rows = split_rows(all_clean, valid_ratio, seed)

    write_json(output_dir / "sft_train.json", train_rows)
    write_json(output_dir / "sft_valid.json", valid_rows)
    write_json(output_dir / "sft_all_clean.json", all_clean)
    report = build_clean_report(all_clean, stats)
    write_json(output_dir / "sft_clean_report.json", report)

    return {
        "clean_rows": len(all_clean),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "output_dir": str(output_dir),
        "report": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean teacher-labeled data into SFT JSON files.")
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=["medical_triage_train_clean (1).json", "medical_triage_test_clean.json"],
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = clean_sft_data(args.input_files, args.output_dir, valid_ratio=args.valid_ratio, seed=args.seed)
    print(f"Clean rows: {result['clean_rows']}")
    print(f"Train rows: {result['train_rows']}")
    print(f"Valid rows: {result['valid_rows']}")
    print(f"Wrote outputs to: {result['output_dir']}")


if __name__ == "__main__":
    main()
