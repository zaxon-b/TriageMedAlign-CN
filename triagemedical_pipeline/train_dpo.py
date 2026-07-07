"""Generate or launch LLaMA-Factory DPO training configs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .config import load_config
from .io_utils import write_json


def build_dpo_dataset_info(dataset_name: str, file_name: str) -> dict:
    return {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }


def build_dpo_yaml(
    *,
    dataset_name: str,
    output_dir: str,
    config_path: str | None = None,
    report_to: str = "none",
    run_name: str = "dpo_v4_r8_beta2_ftx01",
) -> str:
    cfg = load_config(config_path)
    model = cfg.model
    dpo = cfg.dpo
    return f"""### model
model_name_or_path: {model.base_model}
adapter_name_or_path: {model.sft_lora_path}

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: {dpo.lora_rank}
lora_alpha: {dpo.lora_alpha}
lora_dropout: {dpo.lora_dropout}
pref_beta: {dpo.pref_beta}
pref_loss: {dpo.pref_loss}
pref_ftx: {dpo.pref_ftx}

### dataset
dataset: {dataset_name}
template: {model.template}
cutoff_len: {dpo.cutoff_len}
val_size: 0.1
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: 200
eval_steps: 200
eval_strategy: steps
plot_loss: true
overwrite_output_dir: true
report_to: {report_to}
run_name: {run_name}

### train
per_device_train_batch_size: {dpo.per_device_train_batch_size}
gradient_accumulation_steps: {dpo.gradient_accumulation_steps}
learning_rate: {dpo.learning_rate}
num_train_epochs: {dpo.num_train_epochs}
lr_scheduler_type: cosine
warmup_ratio: {dpo.warmup_ratio}
bf16: true
flash_attn: fa2
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or launch DPO training config.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-name", default="dpo_medical_triage_v4")
    parser.add_argument("--file-name", default="dpo/dpo_train.json")
    parser.add_argument("--dataset-info-out", default="configs/dataset_info_dpo.json")
    parser.add_argument("--yaml-out", default="configs/train_dpo_v4.yaml")
    parser.add_argument("--output-dir", default="saves/qwen2.5-7b/lora/dpo_v4_r8_beta2_3epoch")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="dpo_v4_r8_beta2_ftx01")
    parser.add_argument("--run", action="store_true", help="Actually run llamafactory-cli train.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json(args.dataset_info_out, build_dpo_dataset_info(args.dataset_name, args.file_name))
    yaml_text = build_dpo_yaml(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        config_path=args.config,
        report_to=args.report_to,
        run_name=args.run_name,
    )
    Path(args.yaml_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.yaml_out).write_text(yaml_text, encoding="utf-8")
    command = ["llamafactory-cli", "train", args.yaml_out]
    print(f"Wrote dataset info to {args.dataset_info_out}")
    print(f"Wrote DPO YAML to {args.yaml_out}")
    print("Train command:", " ".join(command))
    if args.run:
        subprocess.run(command, check=True)
    else:
        print("Dry-run only. Add --run to start DPO training.")


if __name__ == "__main__":
    main()
