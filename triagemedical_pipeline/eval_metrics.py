"""Reusable evaluation metrics for structured medical triage predictions."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .constants import SYMPTOM_SYNONYMS
from .schema import parse_json_object, validate_triage_obj


def normalize_symptom(text: Any) -> str:
    symptom = str(text).strip()
    symptom = symptom.replace(" ", "")
    symptom = symptom.replace("，", "").replace(",", "")
    symptom = symptom.replace("。", "").replace(".", "")
    return SYMPTOM_SYNONYMS.get(symptom, symptom)


def normalize_symptom_list(symptoms: Any) -> list[str]:
    if not isinstance(symptoms, list):
        return []
    normalized: list[str] = []
    for item in symptoms:
        if not isinstance(item, str):
            continue
        symptom = normalize_symptom(item)
        if symptom:
            normalized.append(symptom)
    return normalized


def hard_jaccard(gold_symptoms: Any, pred_symptoms: Any) -> float:
    gold = {str(item).strip() for item in gold_symptoms if str(item).strip()} if isinstance(gold_symptoms, list) else set()
    pred = {str(item).strip() for item in pred_symptoms if str(item).strip()} if isinstance(pred_symptoms, list) else set()
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    return len(gold & pred) / len(gold | pred)


def normalized_symptom_scores(gold_symptoms: Any, pred_symptoms: Any) -> tuple[float, float]:
    gold_unique = list(dict.fromkeys(normalize_symptom_list(gold_symptoms)))
    pred_unique = list(dict.fromkeys(normalize_symptom_list(pred_symptoms)))
    if not gold_unique and not pred_unique:
        return 1.0, 1.0
    if not gold_unique or not pred_unique:
        return 0.0, 0.0

    used_gold: set[int] = set()
    hits = 0
    for pred_sym in pred_unique:
        for idx, gold_sym in enumerate(gold_unique):
            if idx in used_gold:
                continue
            exact = pred_sym == gold_sym
            containment = (
                len(pred_sym) >= 2
                and len(gold_sym) >= 2
                and (pred_sym in gold_sym or gold_sym in pred_sym)
            )
            if exact or containment:
                hits += 1
                used_gold.add(idx)
                break

    precision = hits / len(pred_unique) if pred_unique else 0.0
    recall = hits / len(gold_unique) if gold_unique else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    union = len(gold_unique) + len(pred_unique) - hits
    jaccard = hits / union if union else 0.0
    return f1, jaccard


def symptom_f1(gold_symptoms: Any, pred_symptoms: Any) -> float:
    return normalized_symptom_scores(gold_symptoms, pred_symptoms)[0]


def _macro_f1(labels: list[str], y_true: list[str], y_pred: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def compute_metrics_from_objects(pairs: list[tuple[dict[str, Any], Any]]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    y_true_dept: list[str] = []
    y_pred_dept: list[str] = []
    y_true_urgency: list[str] = []
    y_pred_urgency: list[str] = []
    hard_scores: list[float] = []
    normalized_f1_scores: list[float] = []
    normalized_jaccard_scores: list[float] = []
    e2e_correct = 0

    for gold, raw_pred in pairs:
        gold_valid, gold_reason = validate_triage_obj(gold)
        if not gold_valid:
            counters[f"skip_bad_gold:{gold_reason}"] += 1
            continue

        counters["evaluated"] += 1
        pred = parse_json_object(raw_pred) if isinstance(raw_pred, str) else raw_pred
        pred_valid, pred_reason = validate_triage_obj(pred)
        if pred is not None:
            counters["json_parse_ok"] += 1
        else:
            counters["json_parse_error"] += 1

        y_true_dept.append(gold["department"])
        y_true_urgency.append(gold["urgency"])
        if pred_valid:
            counters["schema_valid"] += 1
            y_pred_dept.append(pred["department"])
            y_pred_urgency.append(pred["urgency"])
            e2e_correct += int(pred["department"] == gold["department"] and pred["urgency"] == gold["urgency"])
            hard_scores.append(hard_jaccard(gold["symptoms"], pred["symptoms"]))
            f1, norm_j = normalized_symptom_scores(gold["symptoms"], pred["symptoms"])
            normalized_f1_scores.append(f1)
            normalized_jaccard_scores.append(norm_j)
        else:
            counters[f"schema_error:{pred_reason}"] += 1
            y_pred_dept.append("BAD_ANSWER")
            y_pred_urgency.append("BAD_ANSWER")

    evaluated = counters["evaluated"]
    schema_valid = counters["schema_valid"]
    dept_correct = sum(1 for t, p in zip(y_true_dept, y_pred_dept) if p != "BAD_ANSWER" and t == p)
    urgency_correct = sum(1 for t, p in zip(y_true_urgency, y_pred_urgency) if p != "BAD_ANSWER" and t == p)
    high_total = sum(1 for t in y_true_urgency if t == "高")
    high_correct = sum(1 for t, p in zip(y_true_urgency, y_pred_urgency) if t == "高" and p == "高")
    high_to_medium = sum(1 for t, p in zip(y_true_urgency, y_pred_urgency) if t == "高" and p == "中")
    high_to_low = sum(1 for t, p in zip(y_true_urgency, y_pred_urgency) if t == "高" and p == "低")

    return {
        "num_evaluated": evaluated,
        "json_parse_rate": counters["json_parse_ok"] / evaluated if evaluated else 0.0,
        "schema_valid_rate": schema_valid / evaluated if evaluated else 0.0,
        "department_accuracy_conditional": dept_correct / schema_valid if schema_valid else 0.0,
        "urgency_accuracy_conditional": urgency_correct / schema_valid if schema_valid else 0.0,
        "e2e_department_urgency_accuracy": e2e_correct / evaluated if evaluated else 0.0,
        "urgency_macro_f1": _macro_f1(["高", "中", "低"], y_true_urgency, y_pred_urgency),
        "high_urgency_recall": high_correct / high_total if high_total else 0.0,
        "high_urgency_total": high_total,
        "high_to_medium": high_to_medium,
        "high_to_low": high_to_low,
        "hard_jaccard": sum(hard_scores) / len(hard_scores) if hard_scores else 0.0,
        "normalized_symptom_f1": sum(normalized_f1_scores) / len(normalized_f1_scores)
        if normalized_f1_scores
        else 0.0,
        "normalized_symptom_jaccard": sum(normalized_jaccard_scores) / len(normalized_jaccard_scores)
        if normalized_jaccard_scores
        else 0.0,
        "counters": dict(counters),
    }
