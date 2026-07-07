import os

# ==========================================
# 第一步：配置你的 Wandb 密钥和项目名称
# ==========================================
# 1. 登录 https://wandb.ai/authorize 复制你的 API Key
# 2. 把下面这行的字符串替换成你真实的 API Key
os.environ["WANDB_API_KEY"] = "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"

# 自定义你在 Wandb 网页端想看到的项目名称（相当于建一个文件夹）
os.environ["WANDB_PROJECT"] = "TriageMedical-Qwen2.5"


# ==========================================
# 第二步：生成带有 Wandb 监控的 YAML 配置文件
# ==========================================
train_medical_dpo_v4 = """
### model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
adapter_name_or_path: saves/qwen2.5-7b/lora/exp1_lora_r8_all

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

# 🌟 DPO 核心参数（beta↑防遗忘 + ftx SFT正则化）🌟
pref_beta: 2.0
pref_loss: sigmoid
pref_ftx: 0.1

### dataset
dataset: dpo_medical_triage_v3
template: qwen
cutoff_len: 512
val_size: 0.1
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/qwen2.5-7b/lora/dpo_v4_r8_beta2
logging_steps: 10
save_steps: 200
eval_steps: 200
eval_strategy: steps
plot_loss: true
overwrite_output_dir: true

# 🌟 Wandb 配置 🌟
report_to: wandb
run_name: dpo_v4_r8_beta2_ftx01

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1.0e-6
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
flash_attn: fa2
"""

with open("/content/LLaMA-Factory/train_medical_dpo_v4.yaml", "w", encoding="utf-8") as f:
    f.write(train_medical_dpo_v4)

print("✅ Wandb 环境变量已设置，最新的 DPO v4 YAML 配置文件已生成！")
