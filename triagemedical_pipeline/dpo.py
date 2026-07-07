"""Build DPO preference data from predictions or synthetic corruptions."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import CONFUSABLE_DEPARTMENT_PAIRS, SYSTEM_PROMPT
from .eval_metrics import hard_jaccard
from .io_utils import read_json_list, write_json
from .schema import compact_triage_json, parse_json_object, validate_triage_obj


def parse_gold_output(raw: Any) -> dict[str, Any] | None:
    parsed = parse_json_object(raw)
    return parsed if validate_triage_obj(parsed)[0] else None


def is_meaningful_error(gold: dict[str, Any], pred: dict[str, Any], min_jaccard: float = 0.5) -> bool:
    if pred.get("department") != gold["department"]:
        return True
    if pred.get("urgency") != gold["urgency"]:
        return True
    return hard_jaccard(gold.get("symptoms", []), pred.get("symptoms", [])) < min_jaccard


def make_dpo_record(
    instruction: str,
    user_input: str,
    chosen_output: str,
    rejected_output: str,
    source: str,
) -> dict[str, Any]:
    return {
        "conversations": [
            {"from": "system", "value": instruction},
            {"from": "user", "value": user_input},
        ],
        "chosen": {"from": "assistant", "value": chosen_output},
        "rejected": {"from": "assistant", "value": rejected_output},
        "_source": source,
    }


def build_synthetic_pairs(
    rows: list[dict[str, Any]],
    *,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    random.seed(seed)
    pairs: list[dict[str, Any]] = []
    for row in rows:
        gold = parse_gold_output(row.get("output"))
        if gold is None:
            continue

        department = gold["department"]
        urgency = gold["urgency"]
        symptoms = list(gold["symptoms"])
        r = random.random()

        if r < 0.4 and department in CONFUSABLE_DEPARTMENT_PAIRS:
            rejected = dict(gold, department=CONFUSABLE_DEPARTMENT_PAIRS[department])
        elif r < 0.7:
            if urgency == "高":
                new_urgency = "低"
            elif urgency == "低":
                new_urgency = "高"
            else:
                new_urgency = random.choice(["高", "低"])
            rejected = dict(gold, urgency=new_urgency)
        else:
            if len(symptoms) >= 2:
                symptoms.pop(random.randrange(len(symptoms)))
            rejected = dict(gold, symptoms=symptoms)

        pairs.append(
            make_dpo_record(
                instruction=row.get("instruction", SYSTEM_PROMPT),
                user_input=row.get("input", ""),
                chosen_output=compact_triage_json(gold),
                rejected_output=json.dumps(rejected, ensure_ascii=False, separators=(",", ":")),
                source="synthetic",
            )
        )

    if max_samples is not None and len(pairs) > max_samples:
        pairs = random.sample(pairs, max_samples)
    return pairs


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pairs_from_predictions(
    prediction_path: str | Path,
    *,
    min_jaccard: float = 0.5,
    source: str = "prediction_errors",
) -> list[dict[str, Any]]:
    """Build DPO pairs from saved eval predictions.

    Supported prediction fields mirror the existing scripts/notebook:
    ``gold``, ``input``, ``parsed``/``parsed_prediction``, and ``response``.
    """
    pairs: list[dict[str, Any]] = []
    for row in _load_jsonl(prediction_path):
        gold = parse_gold_output(row.get("gold"))
        if gold is None:
            continue
        pred = row.get("parsed_prediction", row.get("parsed"))
        if pred is None:
            pred = parse_json_object(row.get("response"))

        pred_valid = validate_triage_obj(pred)[0] if pred is not None else False
        if pred is None:
            rejected_output = str(row.get("response", ""))
        elif not pred_valid:
            rejected_output = json.dumps(pred, ensure_ascii=False, separators=(",", ":"))
        elif is_meaningful_error(gold, pred, min_jaccard=min_jaccard):
            rejected_output = compact_triage_json(pred)
        else:
            continue

        pairs.append(
            make_dpo_record(
                instruction=SYSTEM_PROMPT,
                user_input=row.get("input", ""),
                chosen_output=compact_triage_json(gold),
                rejected_output=rejected_output,
                source=source,
            )
        )
    return pairs


def deduplicate_by_prompt(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for pair in pairs:
        key = json.dumps(pair["conversations"], ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


def split_pairs(
    pairs: list[dict[str, Any]], *, valid_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    valid_size = max(1, round(len(shuffled) * valid_ratio)) if shuffled else 0
    return shuffled[valid_size:], shuffled[:valid_size]


def strip_internal(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def save_dpo_pairs(
    pairs: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    pairs = deduplicate_by_prompt(pairs)
    train, valid = split_pairs(pairs, valid_ratio=valid_ratio, seed=seed)
    write_json(output_dir / "dpo_train.json", [strip_internal(row) for row in train])
    write_json(output_dir / "dpo_valid.json", [strip_internal(row) for row in valid])
    source_counts = Counter(row.get("_source", "unknown") for row in train)
    return {
        "num_pairs": len(pairs),
        "num_train": len(train),
        "num_valid": len(valid),
        "output_dir": str(output_dir),
        "source_distribution_train": dict(source_counts.most_common()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs.")
    parser.add_argument("--train-data", default="data/processed/sft_train.json")
    parser.add_argument("--predictions-jsonl", nargs="*", default=[])
    parser.add_argument("--output-dir", default="data/dpo")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-synthetic", type=int, default=1000)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--min-jaccard", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_pairs: list[dict[str, Any]] = []
    rows = read_json_list(args.train_data)

    if not args.skip_synthetic:
        synthetic = build_synthetic_pairs(rows, max_samples=args.max_synthetic, seed=args.seed)
        print(f"Generated synthetic pairs: {len(synthetic)}")
        all_pairs.extend(synthetic)

    if not args.synthetic_only:
        for path in args.predictions_jsonl:
            pairs = build_pairs_from_predictions(path, min_jaccard=args.min_jaccard)
            print(f"Generated prediction-error pairs from {path}: {len(pairs)}")
            all_pairs.extend(pairs)

    result = save_dpo_pairs(all_pairs, args.output_dir, valid_ratio=args.valid_ratio, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
