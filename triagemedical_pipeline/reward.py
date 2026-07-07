"""Rule-based verifier reward for TRL GRPOTrainer."""

from __future__ import annotations

from typing import Any

from .constants import ALLOWED_DEPARTMENTS, ALLOWED_URGENCY, REQUIRED_OUTPUT_KEYS, URGENCY_REWARD_MATRIX
from .eval_metrics import normalize_symptom_list, symptom_f1
from .schema import parse_json_object, validate_triage_obj


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def _validate_schema(obj: Any) -> bool:
    return validate_triage_obj(obj)[0]


def medical_triage_reward(prompts, completions, **kwargs):
    """GRPO reward for structured medical triage.

    The function expects gold labels in ``kwargs["output"]`` as SFT-style JSON
    strings. It keeps the latest notebook/script design: strict schema gate,
    department exact match, risk-aware urgency reward, normalized symptom F1,
    and small penalties for markdown, duplicate symptoms, overlong symptoms,
    and high-risk downgrades.
    """
    gold_list = kwargs.get("output", [])
    rewards = []

    for completion, gold_raw in zip(completions, gold_list):
        gold = parse_json_object(gold_raw) if isinstance(gold_raw, str) else gold_raw
        if not _validate_schema(gold):
            rewards.append(0.0)
            continue

        raw_completion = _completion_to_text(completion).strip()
        has_markdown = "```" in raw_completion
        pred = parse_json_object(raw_completion)

        if pred is None:
            rewards.append(-1.0)
            continue
        if set(pred.keys()) != REQUIRED_OUTPUT_KEYS:
            rewards.append(-0.5)
            continue
        if pred.get("department") not in ALLOWED_DEPARTMENTS or pred.get("urgency") not in ALLOWED_URGENCY:
            rewards.append(-0.5)
            continue
        if not _validate_schema(pred):
            rewards.append(-0.5)
            continue

        r_format = 0.8 if has_markdown else 1.0
        r_dept = 1.0 if pred["department"] == gold["department"] else 0.0
        r_urg = URGENCY_REWARD_MATRIX.get((gold["urgency"], pred["urgency"]), 0.0)
        r_sym = symptom_f1(gold.get("symptoms", []), pred.get("symptoms", []))

        penalty = 0.0
        pred_sym_norm = normalize_symptom_list(pred.get("symptoms", []))
        gold_sym_norm = normalize_symptom_list(gold.get("symptoms", []))
        if len(pred_sym_norm) != len(set(pred_sym_norm)):
            penalty -= 0.05
        if len(pred_sym_norm) > max(6, len(gold_sym_norm) + 3):
            penalty -= 0.05
        if has_markdown:
            penalty -= 0.05
        if gold["urgency"] == "高" and pred["urgency"] == "低":
            penalty -= 0.40
        elif gold["urgency"] == "高" and pred["urgency"] == "中":
            penalty -= 0.10

        r_total = 0.05 * r_format + 0.25 * r_dept + 0.30 * r_urg + 0.40 * r_sym + penalty
        rewards.append(float(max(-1.0, min(1.0, r_total))))

    return rewards
