#!/usr/bin/env python3
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

BASE_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
DPO_LORA = 'saves/qwen2.5-7b/lora/dpo_v4_r8_beta2_3epoch'
MERGED_DIR = 'saves/qwen2.5-7b/dpo_merged'
DATA_PATH = 'data/processed/sft_train.json'
OUTPUT_DIR = 'outputs-grpo-v1'


def build_prompt(tokenizer, row):
    messages = [
        {"role": "system", "content": row.get("instruction", "")},
        {"role": "user", "content": row.get("input", "")},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_dataset(tokenizer, path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    prompts = [build_prompt(tokenizer, row) for row in rows]
    outputs = [row["output"] for row in rows]
    return Dataset.from_dict({"prompt": prompts, "output": outputs})


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
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
training_args = GRPOConfig(
    run_name="GRPO_r8",
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    num_generations=4,
    max_completion_length=256,
    temperature=0.9,
    beta=0.04,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-06,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
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
