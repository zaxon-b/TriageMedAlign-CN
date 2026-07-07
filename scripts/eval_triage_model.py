#!/usr/bin/env python3
"""Evaluate base or LoRA models on strict JSON medical triage generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # LoRA evaluation is optional.
    PeftModel = None


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
REQUIRED_KEYS = {"department", "urgency", "symptoms"}

STRONG_SYSTEM_PROMPT = (
    "你是一个中文医疗导诊助手。请根据患者主诉输出严格 JSON。"
    "department 必须从 ['儿科', '呼吸内科', '妇科', '心血管内科', '泌尿外科', "
    "'消化内科', '皮肤科', '眼科', '神经内科', '精神心理科', '耳鼻喉科', '骨科'] 中选择；"
    "urgency 必须是 ['高', '中', '低'] 之一；"
    "symptoms 必须是字符串数组。不要输出解释、Markdown 或多个 JSON。"
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    return rows


def parse_strict_json(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def validate_triage(obj: Any) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not_object"
    if set(obj) != REQUIRED_KEYS:
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


def symptom_jaccard(true_symptoms: list[str], pred_symptoms: Any) -> float:
    if not isinstance(pred_symptoms, list):
        return 0.0
    true_set = {str(item).strip() for item in true_symptoms if str(item).strip()}
    pred_set = {str(item).strip() for item in pred_symptoms if str(item).strip()}
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    return len(true_set & pred_set) / len(true_set | pred_set)


def build_prompt(
    tokenizer: AutoTokenizer,
    row: dict[str, Any],
    use_strong_prompt: bool,
) -> str:
    system_prompt = STRONG_SYSTEM_PROMPT if use_strong_prompt else row.get("instruction", STRONG_SYSTEM_PROMPT)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["input"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(Path(args.data_path))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.lora_path:
        if PeftModel is None:
            raise RuntimeError("peft is not installed, but --lora-path was provided")
        model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    counters: Counter[str] = Counter()
    dept_total = dept_correct = 0
    urgency_total = urgency_correct = 0
    e2e_correct = 0
    jaccard_sum = 0.0
    valid_pred_count = 0

    with predictions_path.open("w", encoding="utf-8") as pred_file:
        for idx, row in enumerate(tqdm(rows, desc="Evaluating")):
            gold = parse_strict_json(row.get("output"))
            gold_valid, gold_reason = validate_triage(gold)
            if not gold_valid:
                counters[f"skip_bad_gold:{gold_reason}"] += 1
                continue

            prompt = build_prompt(tokenizer, row, args.use_strong_prompt)
            response = generate_one(model, tokenizer, prompt, args.max_new_tokens)
            pred = parse_strict_json(response)
            pred_json_valid = pred is not None
            pred_schema_valid, pred_reason = validate_triage(pred)

            counters["evaluated"] += 1
            if pred_json_valid:
                counters["json_parse_ok"] += 1
            else:
                counters["json_parse_error"] += 1
            if pred_schema_valid:
                counters["schema_valid"] += 1
                valid_pred_count += 1
                dept_total += 1
                urgency_total += 1
                dept_correct += int(pred["department"] == gold["department"])
                urgency_correct += int(pred["urgency"] == gold["urgency"])
                jaccard_sum += symptom_jaccard(gold["symptoms"], pred["symptoms"])
            else:
                counters[f"schema_error:{pred_reason}"] += 1

            e2e_pass = (
                pred_schema_valid
                and pred["department"] == gold["department"]
                and pred["urgency"] == gold["urgency"]
            )
            e2e_correct += int(e2e_pass)

            pred_file.write(
                json.dumps(
                    {
                        "idx": idx,
                        "input": row["input"],
                        "gold": gold,
                        "response": response,
                        "parsed_prediction": pred,
                        "prediction_schema_valid": pred_schema_valid,
                        "prediction_error": None if pred_schema_valid else pred_reason,
                        "e2e_pass": e2e_pass,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    evaluated = counters["evaluated"]
    metrics = {
        "model": args.model_name_or_path,
        "lora_path": args.lora_path or "",
        "data_path": args.data_path,
        "num_input_rows": len(rows),
        "num_evaluated_rows": evaluated,
        "num_skipped_bad_gold": len(rows) - evaluated,
        "json_parse_rate": counters["json_parse_ok"] / evaluated if evaluated else 0.0,
        "schema_valid_rate": counters["schema_valid"] / evaluated if evaluated else 0.0,
        "department_accuracy_conditional": dept_correct / dept_total if dept_total else 0.0,
        "urgency_accuracy_conditional": urgency_correct / urgency_total if urgency_total else 0.0,
        "symptom_jaccard_conditional": jaccard_sum / valid_pred_count if valid_pred_count else 0.0,
        "e2e_department_urgency_accuracy": e2e_correct / evaluated if evaluated else 0.0,
        "counters": dict(counters),
        "predictions_path": str(predictions_path),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    csv_path = output_dir / "metrics.csv"
    scalar_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scalar_metrics))
        writer.writeheader()
        writer.writerow(scalar_metrics)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--data-path", default="medical_triage_test_clean.json")
    parser.add_argument("--output-dir", default="results/base_eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--use-strong-prompt",
        action="store_true",
        help="Use the strict 12-department prompt instead of row['instruction'].",
    )
    return parser.parse_args()


def main() -> None:
    metrics = evaluate(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
