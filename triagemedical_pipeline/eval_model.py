"""Model evaluation and saved-prediction metric utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .constants import SYSTEM_PROMPT
from .eval_metrics import compute_metrics_from_objects
from .io_utils import read_json_list, write_json, write_jsonl
from .schema import parse_json_object, validate_triage_obj


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metrics_from_prediction_file(path: str | Path) -> dict[str, Any]:
    pairs: list[tuple[dict[str, Any], Any]] = []
    for row in _load_jsonl(path):
        gold = parse_json_object(row.get("gold"))
        if gold is None and "output" in row:
            gold = parse_json_object(row.get("output"))
        pred = row.get("parsed_prediction", row.get("parsed"))
        if pred is None:
            pred = row.get("response")
        if gold is not None:
            pairs.append((gold, pred))
    return compute_metrics_from_objects(pairs)


def compare_prediction_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    baseline_name: str = "DPO",
    candidate_name: str = "GRPO",
) -> dict[str, Any]:
    baseline = _load_jsonl(baseline_path)
    candidate = _load_jsonl(candidate_path)
    n = min(len(baseline), len(candidate))
    counters = {
        "num_compared": n,
        "input_mismatch": 0,
        "both_e2e_correct": 0,
        "baseline_correct_candidate_wrong": 0,
        "baseline_wrong_candidate_correct": 0,
        "both_e2e_wrong": 0,
        "baseline_urgency_correct_candidate_wrong": 0,
        "baseline_urgency_wrong_candidate_correct": 0,
        "high_total": 0,
        "baseline_high_correct": 0,
        "candidate_high_correct": 0,
        "baseline_high_to_low": 0,
        "candidate_high_to_low": 0,
    }

    def status(item: dict[str, Any]) -> dict[str, Any]:
        gold = parse_json_object(item.get("gold")) or {}
        parsed = item.get("parsed_prediction", item.get("parsed"))
        if parsed is None:
            parsed = parse_json_object(item.get("response"))
        schema_valid = validate_triage_obj(parsed)[0]
        pred_dept = parsed.get("department") if schema_valid else "BAD_ANSWER"
        pred_urg = parsed.get("urgency") if schema_valid else "BAD_ANSWER"
        return {
            "gold_urg": gold.get("urgency"),
            "pred_urg": pred_urg,
            "dept_correct": pred_dept == gold.get("department"),
            "urgency_correct": pred_urg == gold.get("urgency"),
            "e2e_correct": pred_dept == gold.get("department") and pred_urg == gold.get("urgency"),
        }

    for idx in range(n):
        b_item = baseline[idx]
        c_item = candidate[idx]
        if b_item.get("input") != c_item.get("input"):
            counters["input_mismatch"] += 1
        b = status(b_item)
        c = status(c_item)
        if b["e2e_correct"] and c["e2e_correct"]:
            counters["both_e2e_correct"] += 1
        elif b["e2e_correct"] and not c["e2e_correct"]:
            counters["baseline_correct_candidate_wrong"] += 1
        elif not b["e2e_correct"] and c["e2e_correct"]:
            counters["baseline_wrong_candidate_correct"] += 1
        else:
            counters["both_e2e_wrong"] += 1

        if b["urgency_correct"] and not c["urgency_correct"]:
            counters["baseline_urgency_correct_candidate_wrong"] += 1
        elif not b["urgency_correct"] and c["urgency_correct"]:
            counters["baseline_urgency_wrong_candidate_correct"] += 1

        if b["gold_urg"] == "高":
            counters["high_total"] += 1
            counters["baseline_high_correct"] += int(b["pred_urg"] == "高")
            counters["candidate_high_correct"] += int(c["pred_urg"] == "高")
            counters["baseline_high_to_low"] += int(b["pred_urg"] == "低")
            counters["candidate_high_to_low"] += int(c["pred_urg"] == "低")

    counters["baseline_name"] = baseline_name
    counters["candidate_name"] = candidate_name
    counters["e2e_net_corrections"] = (
        counters["baseline_wrong_candidate_correct"] - counters["baseline_correct_candidate_wrong"]
    )
    counters["urgency_net_corrections"] = (
        counters["baseline_urgency_wrong_candidate_correct"]
        - counters["baseline_urgency_correct_candidate_wrong"]
    )
    return counters


class TriageModelEvaluator:
    """Light wrapper around Transformers generation for Base/SFT/DPO/GRPO eval."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        lora_path: str = "",
        data_path: str = "data/processed/sft_valid.json",
        output_dir: str = "results/eval",
        max_samples: int | None = None,
        max_new_tokens: int = 150,
        use_strong_prompt: bool = True,
        bf16: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.lora_path = lora_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        self.use_strong_prompt = use_strong_prompt
        self.bf16 = bf16

    def _build_prompt(self, tokenizer, row: dict[str, Any]) -> str:
        system_prompt = SYSTEM_PROMPT if self.use_strong_prompt else row.get("instruction", SYSTEM_PROMPT)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["input"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def run(self) -> dict[str, Any]:
        import torch
        from tqdm import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None

        rows = read_json_list(self.data_path)
        if self.max_samples is not None:
            rows = rows[: self.max_samples]

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16 if self.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.lora_path:
            if PeftModel is None:
                raise RuntimeError("peft is required when --lora-path is provided")
            model = PeftModel.from_pretrained(model, self.lora_path)
        model.eval()

        predictions: list[dict[str, Any]] = []
        pairs: list[tuple[dict[str, Any], Any]] = []
        for idx, row in enumerate(tqdm(rows, desc="Evaluating")):
            gold = parse_json_object(row.get("output"))
            if not validate_triage_obj(gold)[0]:
                continue
            prompt = self._build_prompt(tokenizer, row)
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            parsed = parse_json_object(response)
            schema_valid, schema_error = validate_triage_obj(parsed)
            predictions.append(
                {
                    "idx": idx,
                    "input": row["input"],
                    "gold": gold,
                    "response": response,
                    "parsed": parsed,
                    "schema_valid": schema_valid,
                    "schema_error": None if schema_valid else schema_error,
                }
            )
            pairs.append((gold, parsed if parsed is not None else response))

        metrics = compute_metrics_from_objects(pairs)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.output_dir / "predictions.jsonl", predictions)
        write_json(self.output_dir / "metrics.json", metrics)
        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model outputs or saved predictions.")
    parser.add_argument("--predictions-jsonl", default="", help="Compute metrics from saved predictions.jsonl.")
    parser.add_argument("--compare-baseline", default="")
    parser.add_argument("--compare-candidate", default="")
    parser.add_argument("--baseline-name", default="DPO")
    parser.add_argument("--candidate-name", default="GRPO")
    parser.add_argument("--model-name-or-path", default="")
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--data-path", default="data/processed/sft_valid.json")
    parser.add_argument("--output-dir", default="results/eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--use-strong-prompt", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_baseline and args.compare_candidate:
        report = compare_prediction_files(
            args.compare_baseline,
            args.compare_candidate,
            baseline_name=args.baseline_name,
            candidate_name=args.candidate_name,
        )
        write_json(output_dir / "paired_comparison.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.predictions_jsonl:
        metrics = metrics_from_prediction_file(args.predictions_jsonl)
        write_json(output_dir / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if not args.model_name_or_path:
        raise SystemExit("Provide --predictions-jsonl, --compare-* files, or --model-name-or-path.")

    evaluator = TriageModelEvaluator(
        args.model_name_or_path,
        lora_path=args.lora_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        use_strong_prompt=args.use_strong_prompt,
        bf16=args.bf16,
    )
    metrics = evaluator.run()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
