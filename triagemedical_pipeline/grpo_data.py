"""Prepare SFT records for TRL GRPOTrainer."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_json_list, write_json, write_jsonl


def build_grpo_records(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        records.append(
            {
                "prompt": f"{row.get('instruction', '')}\n{row.get('input', '')}",
                "output": row.get("output", ""),
                "input": row.get("input", ""),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SFT JSON to GRPO prompt/output records.")
    parser.add_argument("--sft-data", default="data/processed/sft_train.json")
    parser.add_argument("--output-path", default="data/processed/grpo_train.json")
    parser.add_argument("--jsonl", action="store_true", help="Write JSONL instead of JSON list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_json_list(args.sft_data)
    records = build_grpo_records(rows)
    if args.jsonl or Path(args.output_path).suffix == ".jsonl":
        write_jsonl(args.output_path, records)
    else:
        write_json(args.output_path, records)
    print(f"Wrote {len(records)} GRPO records to {args.output_path}")


if __name__ == "__main__":
    main()
