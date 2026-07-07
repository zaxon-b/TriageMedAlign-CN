#!/usr/bin/env python3
"""Validate cleaned SFT data for the medical triage project."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_DEPARTMENTS = {
    "儿科",
    "呼吸内科",
    "妇科",
    "心血管内科",
    "泌尿外科",
    "消化内科",
    "皮肤科",
    "眼科",
    "神经内科",
    "精神心理科",
    "耳鼻喉科",
    "骨科",
}

ALLOWED_URGENCY = {"高", "中", "低"}
REQUIRED_OUTPUT_KEYS = {"department", "urgency", "symptoms"}
REQUIRED_ROW_KEYS = {"instruction", "input", "output"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    return rows


def validate_file(path: Path) -> tuple[bool, dict[str, Any]]:
    rows = load_rows(path)
    errors: list[dict[str, Any]] = []
    input_counter: Counter[str] = Counter()
    dept_counter: Counter[str] = Counter()
    urgency_counter: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append({"idx": idx, "type": "row_not_object", "value": repr(row)[:300]})
            continue

        missing_row_keys = REQUIRED_ROW_KEYS - set(row)
        if missing_row_keys:
            errors.append({"idx": idx, "type": "missing_row_keys", "keys": sorted(missing_row_keys)})
            continue

        instruction = row.get("instruction")
        input_text = row.get("input")
        output_text = row.get("output")

        if not isinstance(instruction, str) or not instruction.strip():
            errors.append({"idx": idx, "type": "bad_instruction"})
        if not isinstance(input_text, str) or not input_text.strip():
            errors.append({"idx": idx, "type": "bad_input"})
        else:
            input_counter[input_text] += 1

        if not isinstance(output_text, str) or not output_text.strip():
            errors.append({"idx": idx, "type": "bad_output_text"})
            continue

        if "```" in output_text:
            errors.append({"idx": idx, "type": "markdown_fence_in_output"})

        try:
            output = json.loads(output_text)
        except json.JSONDecodeError as exc:
            errors.append({"idx": idx, "type": "output_not_json", "error": str(exc)})
            continue

        if not isinstance(output, dict):
            errors.append({"idx": idx, "type": "output_not_object", "value": repr(output)[:300]})
            continue

        output_keys = set(output)
        if output_keys != REQUIRED_OUTPUT_KEYS:
            errors.append(
                {
                    "idx": idx,
                    "type": "bad_output_keys",
                    "expected": sorted(REQUIRED_OUTPUT_KEYS),
                    "actual": sorted(output_keys),
                }
            )

        department = output.get("department")
        urgency = output.get("urgency")
        symptoms = output.get("symptoms")

        if department not in ALLOWED_DEPARTMENTS:
            errors.append({"idx": idx, "type": "bad_department", "value": department})
        else:
            dept_counter[department] += 1

        if urgency not in ALLOWED_URGENCY:
            errors.append({"idx": idx, "type": "bad_urgency", "value": urgency})
        else:
            urgency_counter[urgency] += 1

        if not isinstance(symptoms, list) or not symptoms:
            errors.append({"idx": idx, "type": "bad_symptoms", "value": symptoms})
        elif not all(isinstance(item, str) and item.strip() for item in symptoms):
            errors.append({"idx": idx, "type": "bad_symptom_item", "value": symptoms})

    duplicate_inputs = [text for text, count in input_counter.items() if count > 1]
    for input_text in duplicate_inputs[:20]:
        errors.append(
            {
                "idx": None,
                "type": "duplicate_input",
                "count": input_counter[input_text],
                "input": input_text[:200],
            }
        )

    report = {
        "file": str(path),
        "num_rows": len(rows),
        "num_errors": len(errors),
        "passed": len(errors) == 0,
        "department_distribution": dict(dept_counter.most_common()),
        "urgency_distribution": dict(urgency_counter.most_common()),
        "errors_preview": errors[:50],
    }
    return len(errors) == 0, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="+",
        help="Cleaned SFT JSON files to validate.",
    )
    parser.add_argument(
        "--report-path",
        default="data/processed/sft_validation_report.json",
        help="Where to write the validation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    all_passed = True

    for file_name in args.files:
        passed, report = validate_file(Path(file_name))
        reports.append(report)
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {file_name}: rows={report['num_rows']}, errors={report['num_errors']}")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"passed": all_passed, "files": reports}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Validation report written to: {report_path}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
