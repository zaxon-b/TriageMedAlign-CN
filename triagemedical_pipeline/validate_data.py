"""Validate SFT/DPO/GRPO data files."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_json_list, write_json
from .schema import validate_dpo_rows, validate_sft_rows


def validate_sft_file(path: str | Path) -> dict:
    report = validate_sft_rows(read_json_list(path))
    report["file"] = str(path)
    return report


def validate_dpo_file(path: str | Path) -> dict:
    report = validate_dpo_rows(read_json_list(path))
    report["file"] = str(path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TriageMedical SFT and DPO data files.")
    parser.add_argument("--sft", nargs="*", default=[], help="SFT JSON files to validate.")
    parser.add_argument("--dpo", nargs="*", default=[], help="DPO JSON files to validate.")
    parser.add_argument("--report-path", default="data/processed/pipeline_validation_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sft and not args.dpo:
        args.sft = [
            "data/processed/sft_train.json",
            "data/processed/sft_valid.json",
            "data/processed/sft_all_clean.json",
        ]
        args.dpo = ["dpo_train_2.json", "dpo_valid_2.json"]

    reports = {"sft": [], "dpo": []}
    all_passed = True

    for file_name in args.sft:
        report = validate_sft_file(file_name)
        reports["sft"].append(report)
        all_passed = all_passed and report["passed"]
        status = "PASS" if report["passed"] else "FAIL"
        print(f"[SFT {status}] {file_name}: rows={report['num_rows']}, errors={report['num_errors']}")

    for file_name in args.dpo:
        report = validate_dpo_file(file_name)
        reports["dpo"].append(report)
        all_passed = all_passed and report["passed"]
        status = "PASS" if report["passed"] else "FAIL"
        print(f"[DPO {status}] {file_name}: rows={report['num_rows']}, errors={report['num_errors']}")

    payload = {"passed": all_passed, **reports}
    write_json(args.report_path, payload)
    print(f"Validation report written to: {args.report_path}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
