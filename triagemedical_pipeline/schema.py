"""Parsing and schema validation for SFT, DPO, GRPO, and predictions."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .constants import ALLOWED_DEPARTMENTS, ALLOWED_URGENCY, REQUIRED_OUTPUT_KEYS, REQUIRED_SFT_KEYS


def parse_json_object(value: Any, *, strip_markdown: bool = True) -> dict[str, Any] | None:
    """Parse a string/dict into a JSON object without extracting JSON from prose."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if strip_markdown:
        text = re.sub(r"```json|```", "", text).strip()
        text = re.sub(r"<\|im_end\|>|</s>", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def compact_triage_json(obj: dict[str, Any]) -> str:
    return json.dumps(
        {
            "department": obj["department"],
            "urgency": obj["urgency"],
            "symptoms": obj["symptoms"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def normalize_symptoms_for_schema(symptoms: Any) -> list[str] | None:
    if not isinstance(symptoms, list):
        return None
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in symptoms:
        if not isinstance(item, str):
            return None
        symptom = item.strip()
        if symptom and symptom not in seen:
            cleaned.append(symptom)
            seen.add(symptom)
    return cleaned or None


def validate_triage_obj(obj: Any) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not_object"
    if set(obj) != REQUIRED_OUTPUT_KEYS:
        return False, "bad_keys"
    if obj.get("department") not in ALLOWED_DEPARTMENTS:
        return False, "bad_department"
    if obj.get("urgency") not in ALLOWED_URGENCY:
        return False, "bad_urgency"
    symptoms = obj.get("symptoms")
    if not isinstance(symptoms, list) or not symptoms:
        return False, "bad_symptoms"
    if not all(isinstance(item, str) and item.strip() for item in symptoms):
        return False, "bad_symptom_item"
    return True, "ok"


def validate_sft_rows(rows: list[dict[str, Any]], *, check_duplicates: bool = True) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    input_counter: Counter[str] = Counter()
    dept_counter: Counter[str] = Counter()
    urgency_counter: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append({"idx": idx, "type": "row_not_object", "value": repr(row)[:300]})
            continue
        missing = REQUIRED_SFT_KEYS - set(row)
        if missing:
            errors.append({"idx": idx, "type": "missing_row_keys", "keys": sorted(missing)})
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

        output = parse_json_object(output_text)
        valid, reason = validate_triage_obj(output)
        if not valid:
            errors.append({"idx": idx, "type": reason, "value": output})
            continue
        dept_counter[output["department"]] += 1
        urgency_counter[output["urgency"]] += 1

    if check_duplicates:
        for input_text, count in input_counter.items():
            if count > 1:
                errors.append(
                    {"idx": None, "type": "duplicate_input", "count": count, "input": input_text[:200]}
                )

    return {
        "num_rows": len(rows),
        "num_errors": len(errors),
        "passed": len(errors) == 0,
        "department_distribution": dict(dept_counter.most_common()),
        "urgency_distribution": dict(urgency_counter.most_common()),
        "errors_preview": errors[:50],
    }


def validate_dpo_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    diff_counter: Counter[str] = Counter()
    rejected_schema_warnings: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append({"idx": idx, "type": "row_not_object"})
            continue
        for key in ("conversations", "chosen", "rejected"):
            if key not in row:
                errors.append({"idx": idx, "type": "missing_key", "key": key})
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            errors.append({"idx": idx, "type": "bad_conversations"})

        chosen = (row.get("chosen") or {}).get("value") if isinstance(row.get("chosen"), dict) else None
        rejected = (row.get("rejected") or {}).get("value") if isinstance(row.get("rejected"), dict) else None
        if not isinstance(rejected, str) or not rejected.strip():
            errors.append({"idx": idx, "type": "bad_rejected_text"})
            continue
        chosen_obj = parse_json_object(chosen)
        rejected_obj = parse_json_object(rejected)

        chosen_valid, chosen_reason = validate_triage_obj(chosen_obj)
        if not chosen_valid:
            errors.append({"idx": idx, "type": f"bad_chosen:{chosen_reason}", "value": chosen_obj})
        rejected_valid, rejected_reason = validate_triage_obj(rejected_obj)
        if rejected_obj is None:
            rejected_schema_warnings["rejected_not_json"] += 1
        elif not rejected_valid:
            rejected_schema_warnings[f"rejected_schema:{rejected_reason}"] += 1
        if chosen_valid and rejected_valid:
            diffs = []
            if chosen_obj["department"] != rejected_obj["department"]:
                diffs.append("department")
            if chosen_obj["urgency"] != rejected_obj["urgency"]:
                diffs.append("urgency")
            if chosen_obj["symptoms"] != rejected_obj["symptoms"]:
                diffs.append("symptoms")
            diff_counter["+".join(diffs) or "no_diff"] += 1

    return {
        "num_rows": len(rows),
        "num_errors": len(errors),
        "passed": len(errors) == 0,
        "diff_distribution": dict(diff_counter.most_common()),
        "rejected_schema_warnings": dict(rejected_schema_warnings.most_common()),
        "errors_preview": errors[:50],
    }
