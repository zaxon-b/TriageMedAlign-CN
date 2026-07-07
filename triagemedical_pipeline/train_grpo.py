"""Generate or launch the TRL GRPO training entrypoint."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .config import load_config


def build_grpo_script(
    *,
    config_path: str | None = None,
    base_model: str | None = None,
    dpo_lora: str | None = None,
    merged_dir: str | None = None,
    data_path: str | None = None,
    output_dir: str | None = None,
) -> str:
    cfg = load_config(config_path)
    model = cfg.model
    grpo = cfg.grpo
    base_model = base_model or model.base_model
    dpo_lora = dpo_lora or model.dpo_lora_path
    merged_dir = merged_dir or model.dpo_merged_path
    data_path = data_path or cfg.paths.sft_train
    output_dir = output_dir or "outputs-grpo-v1"

    return f'''#!/usr/bin/env python3
"""Generated GRPO training entrypoint. This script starts a long GPU training job."""

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

cwd = Path.cwd()
if (cwd / "triagemedical_pipeline").exists():
    sys.path.insert(0, str(cwd))

from triagemedical_pipeline.reward import medical_triage_reward

BASE_MODEL = {base_model!r}
DPO_LORA = {dpo_lora!r}
MERGED_DIR = {merged_dir!r}
DATA_PATH = {data_path!r}
OUTPUT_DIR = {output_dir!r}


def build_prompt(tokenizer, row):
    messages = [
        {{"role": "system", "content": row.get("instruction", "")}},
        {{"role": "user", "content": row.get("input", "")}},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_dataset(tokenizer, path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    prompts = [build_prompt(tokenizer, row) for row in rows]
    outputs = [row["output"] for row in rows]
    return Dataset.from_dict({{"prompt": prompts, "output": outputs}})


if not os.path.exists(os.path.join(MERGED_DIR, "config.json")):
    print("Merging DPO LoRA into base model...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer_merge = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, DPO_LORA)
    merged = model.merge_and_unload()
    merged.save_pretrained(MERGED_DIR)
    tokenizer_merge.save_pretrained(MERGED_DIR)
    del base, model, merged, tokenizer_merge
    torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(tokenizer, DATA_PATH)
peft_config = LoraConfig(
    r={grpo.lora_rank},
    lora_alpha={grpo.lora_alpha},
    lora_dropout={grpo.lora_dropout},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
training_args = GRPOConfig(
    run_name="GRPO_r8",
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    num_generations={grpo.num_generations},
    max_completion_length={grpo.max_completion_length},
    temperature={grpo.temperature},
    beta={grpo.beta},
    per_device_train_batch_size={grpo.per_device_train_batch_size},
    gradient_accumulation_steps={grpo.gradient_accumulation_steps},
    learning_rate={grpo.learning_rate},
    num_train_epochs={grpo.num_train_epochs},
    lr_scheduler_type="cosine",
    warmup_ratio={grpo.warmup_ratio},
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    report_to=os.environ.get("TRIAGEMEDICAL_REPORT_TO", "none"),
    use_vllm=False,
)
trainer = GRPOTrainer(
    model=MERGED_DIR,
    processing_class=tokenizer,
    reward_funcs=medical_triage_reward,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "lora_adapter"))
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or launch TRL GRPO training script.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--script-out", default="configs/train_grpo_trl.py")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--dpo-lora", default=None)
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run", action="store_true", help="Actually run the generated GRPO script.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_text = build_grpo_script(
        config_path=args.config,
        base_model=args.base_model,
        dpo_lora=args.dpo_lora,
        merged_dir=args.merged_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    script_path = Path(args.script_out)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_text, encoding="utf-8")
    command = ["python", str(script_path)]
    print(f"Wrote GRPO training script to {script_path}")
    print("Train command:", " ".join(command))
    if args.run:
        subprocess.run(command, check=True)
    else:
        print("Dry-run only. Add --run to start GRPO training.")


if __name__ == "__main__":
    main()
