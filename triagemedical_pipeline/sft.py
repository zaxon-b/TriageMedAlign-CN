"""SFT data registration and LLaMA-Factory config helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .io_utils import write_json


def build_dataset_info(dataset_name: str, file_name: str) -> dict:
    return {
        dataset_name: {
            "file_name": file_name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }


def build_sft_yaml(
    *,
    dataset_name: str,
    output_dir: str,
    config_path: str | None = None,
    report_to: str = "none",
) -> str:
    cfg = load_config(config_path)
    model = cfg.model
    sft = cfg.sft
    return f"""### model
model_name_or_path: {model.base_model}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: {sft.lora_rank}
lora_alpha: {sft.lora_alpha}
lora_dropout: {sft.lora_dropout}

### dataset
dataset: {dataset_name}
template: {model.template}
cutoff_len: {sft.cutoff_len}
val_size: 0.1
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: 100
eval_steps: 100
eval_strategy: steps
plot_loss: true
overwrite_output_dir: true
report_to: {report_to}

### train
per_device_train_batch_size: {sft.per_device_train_batch_size}
gradient_accumulation_steps: {sft.gradient_accumulation_steps}
learning_rate: {sft.learning_rate}
num_train_epochs: {sft.num_train_epochs}
lr_scheduler_type: cosine
warmup_ratio: {sft.warmup_ratio}
bf16: true
flash_attn: fa2
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SFT dataset_info and training YAML.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-name", default="medical_sft_train")
    parser.add_argument("--file-name", default="sft_train.json")
    parser.add_argument("--dataset-info-out", default="configs/dataset_info_sft.json")
    parser.add_argument("--yaml-out", default="configs/train_sft_lora_r8.yaml")
    parser.add_argument("--output-dir", default="saves/qwen2.5-7b/lora/exp1_lora_r8_all")
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json(args.dataset_info_out, build_dataset_info(args.dataset_name, args.file_name))
    yaml_text = build_sft_yaml(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        config_path=args.config,
        report_to=args.report_to,
    )
    Path(args.yaml_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.yaml_out).write_text(yaml_text, encoding="utf-8")
    print(f"Wrote dataset info to {args.dataset_info_out}")
    print(f"Wrote SFT YAML to {args.yaml_out}")
    print(f"Train command: llamafactory-cli train {args.yaml_out}")


if __name__ == "__main__":
    main()
