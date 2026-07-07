#!/usr/bin/env python3
"""Build DPO preference data for medical triage from three sources.

Source 1 — Gold(32B) vs Base(7B-Instruct):  Base model errors as rejected.
Source 2 — Gold(32B) vs SFT(LoRA r=8):      SFT model errors as rejected.
Source 3 — Synthetic corruption:              Deliberately altered gold as rejected.

Works in Colab with vLLM; falls back to transformers for smaller envs.

Usage:
    # Full pipeline (needs GPU):
    python build_dpo_data.py --train-data data/processed/sft_train.json \
                             --base-model Qwen/Qwen2.5-7B-Instruct \
                             --lora-path /path/to/lora/checkpoint \
                             --output-dir data/dpo

    # Synthetic only (no GPU needed, for quick test):
    python build_dpo_data.py --train-data data/processed/sft_train.json \
                             --synthetic-only \
                             --output-dir data/dpo
"""

from __future__ import annotations  # 推迟注解求值，便于前向引用类型名

import argparse  # 解析命令行参数
import json  # 读写 JSON、构造去重键
import os  # 设置 vLLM 相关环境变量
import random  # 打乱、抽样、合成破坏策略
import sys  # 退出码、stdout/stderr workaround
from collections import Counter  # 统计各 _source 在 train/valid 中的数量
from pathlib import Path  # 跨平台路径拼接与存在性检查
from typing import Any  # JSON 行与解析结果的宽松类型

# ---------------------------------------------------------------------------
# Constants (mirror clean_sft_data.py)
# ---------------------------------------------------------------------------
# 与 clean_sft_data 中允许的科室/紧急度一致，便于 gold 与训练格式对齐。

ALLOWED_DEPARTMENTS = [  # 允许导诊输出的科室列表（顺序仅便于阅读）
    "儿科", "呼吸内科", "妇科", "心血管内科", "泌尿外科",
    "消化内科", "皮肤科", "眼科", "神经内科", "精神心理科", "耳鼻喉科", "骨科",
]
ALLOWED_URGENCY = ["高", "中", "低"]  # 允许的三档紧急度

# Confusable department pairs for synthetic corruption (Source 3).
# Each tuple: (source_department, confusable_department)
# 易混淆科室映射：合成负样本时把 gold 科室换成「看起来像」的另一科室。
CONFUSABLE_PAIRS = {
    "神经内科": "骨科",
    "骨科": "神经内科",
    "消化内科": "泌尿外科",
    "泌尿外科": "消化内科",
    "呼吸内科": "心血管内科",
    "心血管内科": "呼吸内科",
    "妇科": "泌尿外科",
    "儿科": "呼吸内科",
    "眼科": "神经内科",
    "皮肤科": "泌尿外科",
    "精神心理科": "神经内科",
    "耳鼻喉科": "呼吸内科",
}

# 与 SFT 数据中的 system 指令一致，保证 DPO 里 chosen/rejected 在同一任务分布下比较。
SYSTEM_PROMPT = (
    "你是一个中文医疗导诊助手。请根据患者主诉输出严格 JSON。"
    "department 必须从 ['儿科', '呼吸内科', '妇科', '心血管内科', '泌尿外科', "
    "'消化内科', '皮肤科', '眼科', '神经内科', '精神心理科', '耳鼻喉科', '骨科'] 中选择；"
    "urgency 必须是 ['高', '中', '低'] 之一；"
    "symptoms 必须是字符串数组。不要输出解释、Markdown 或多个 JSON。"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list[dict[str, Any]]:
    """从磁盘读取 JSON 文件，要求根元素为对象列表。"""
    with path.open("r", encoding="utf-8") as f:  # 按 UTF-8 打开，避免中文乱码
        data = json.load(f)  # 解析整个文件为 Python 对象
    if not isinstance(data, list):  # DPO/SFT 脚本约定：一行一个样本的列表结构
        raise ValueError(f"{path} must contain a JSON list")
    return data  # 返回样本行列表


def parse_gold_output(raw: Any) -> dict[str, Any] | None:
    """Parse and validate a gold JSON output string/dict."""
    if isinstance(raw, dict):  # 若已是 dict（部分数据可能预解析过）
        parsed = raw
    elif isinstance(raw, str):  # 常见：模型输出的 JSON 字符串，可能带 markdown 围栏
        text = raw.replace("```json", "").replace("```", "").strip()  # 去掉代码块标记与首尾空白
        try:
            parsed = json.loads(text)  # 解析为 Python dict
        except json.JSONDecodeError:  # 非法 JSON → 无法作为结构化标签使用
            return None
    else:  # 既不是 str 也不是 dict，无法解析
        return None
    if not isinstance(parsed, dict):  # 根必须是对象（含 department 等键）
        return None
    required = {"department", "urgency", "symptoms"}  # 导诊任务必需字段
    if not required.issubset(parsed.keys()):  # 缺任一字段则认为不可用
        return None
    return parsed  # 通过最小结构校验的 dict


def symptom_jaccard(a: list[str], b: list[str]) -> float:
    """症状列表的 Jaccard 相似度，用于判断预测与 gold 症状是否足够接近。"""
    set_a = {s.strip() for s in a if isinstance(s, str) and s.strip()}  # gold 侧：合法非空字符串 → 集合
    set_b = {s.strip() for s in b if isinstance(s, str) and s.strip()}  # 预测侧同样规范化
    if not set_a and not set_b:  # 两侧都空：视为完全一致
        return 1.0
    if not set_a or not set_b:  # 仅一侧空：无交集
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)  # |A∩B| / |A∪B|


def is_meaningful_error(
    gold: dict[str, Any], pred: dict[str, Any], min_jaccard: float = 0.5
) -> bool:
    """Return True if pred is meaningfully worse than gold (worth learning from)."""
    if pred.get("department") != gold["department"]:  # 科室错了 → 值得 DPO 纠正
        return True
    if pred.get("urgency") != gold["urgency"]:  # 紧急度错了 → 值得纠正
        return True
    g_sym = gold.get("symptoms", [])  # gold 症状列表（缺省当空列表）
    p_sym = pred.get("symptoms", [])
    if symptom_jaccard(g_sym, p_sym) < min_jaccard:  # 症状集合差异大 → 视为有意义错误
        return True
    return False  # 科室、紧急度一致且症状足够重叠 → 不当作偏好对


def make_dpo_record(
    instruction: str,
    user_input: str,
    chosen_output: str,
    rejected_output: str,
    source: str,
) -> dict[str, Any]:
    """Build one LLaMA-Factory-compatible DPO record."""
    return {
        "conversations": [  # 多轮对话前缀：system + user（与常见聊天模板一致）
            {"from": "system", "value": instruction},
            {"from": "user", "value": user_input},
        ],
        "chosen": {"from": "assistant", "value": chosen_output},  # 偏好回答（通常为 gold 紧凑 JSON）
        "rejected": {"from": "assistant", "value": rejected_output},  # 非偏好回答（模型错或合成错）
        "_source": source,  # 内部溯源标签；写入文件前会 strip 掉
    }


def gold_to_compact_json(gold: dict[str, Any]) -> str:
    """Serialize gold dict to compact JSON matching SFT training format."""
    return json.dumps(
        {"department": gold["department"], "urgency": gold["urgency"], "symptoms": gold["symptoms"]},
        ensure_ascii=False,  # 保留中文可读
        separators=(",", ":"),  # 无多余空格，与 SFT 紧凑格式一致
    )


# ---------------------------------------------------------------------------
# Source 3: Synthetic corruption
# ---------------------------------------------------------------------------

def build_synthetic_pairs(
    rows: list[dict[str, Any]], max_samples: int | None = None
) -> list[dict[str, Any]]:
    """Create DPO pairs by deliberately corrupting gold outputs."""
    pairs: list[dict[str, Any]] = []  # 累积合成的 (chosen, rejected) 记录
    random.seed(42)  # 固定种子，使可复现（与后面 sample 的种子一致策略）

    for row in rows:  # 遍历每条 SFT 行：用其 output 作 gold
        gold = parse_gold_output(row.get("output"))  # 解析教师标签
        if gold is None:  # 无法解析则跳过，不造伪偏好对
            continue

        department = gold["department"]  # 当前 gold 科室
        urgency = gold["urgency"]  # 当前 gold 紧急度
        symptoms = list(gold["symptoms"])  # copy：后面可能 pop，避免改到 gold 本体

        # Choose a corruption strategy at random
        r = random.random()  # [0,1) 均匀随机，驱动下面三分支概率

        if r < 0.4 and department in CONFUSABLE_PAIRS:  # 约 40%：且该科室有易混映射时用策略 A
            # Strategy A: swap department to confusable one
            rejected = dict(gold, department=CONFUSABLE_PAIRS[department])  # 浅拷贝并替换 department
        elif r < 0.7:  # 约 30%（累计到 0.7）：策略 B 改紧急度
            # Strategy B: flip urgency
            if urgency == "高":  # 高 → 低，制造危险低估
                new_urgency = "低"
            elif urgency == "低":  # 低 → 高，制造过度紧张
                new_urgency = "高"
            else:  # 中为「中」时随机拉到高或低
                new_urgency = random.choice(["高", "低"])
            rejected = dict(gold, urgency=new_urgency)
        else:  # 剩余约 30%：策略 C 改症状列表
            # Strategy C: modify symptoms
            if len(symptoms) >= 2:  # 至少两条才删一条，否则列表过短
                symptoms.pop(random.randrange(len(symptoms)))  # remove one：随机删一条症状
            rejected = dict(gold, symptoms=symptoms)  # 可能只删一条，也可能原样（len<2 时）

        chosen_json = gold_to_compact_json(gold)  # chosen：标准紧凑 JSON 字符串
        rejected_json = json.dumps(rejected, ensure_ascii=False, separators=(",", ":"))  # rejected：破坏后的紧凑 JSON

        pairs.append(
            make_dpo_record(
                instruction=row.get("instruction", SYSTEM_PROMPT),  # 行内若有自定义 instruction 则沿用
                user_input=row.get("input", ""),  # 患者主诉
                chosen_output=chosen_json,
                rejected_output=rejected_json,
                source="synthetic",  # 标记来源为合成
            )
        )

    if max_samples is not None and len(pairs) > max_samples:  # 超出上限则无放回随机下采样
        pairs = random.sample(pairs, max_samples)

    return pairs


# ---------------------------------------------------------------------------
# vLLM batch inference (Source 1 & 2)
# ---------------------------------------------------------------------------

def build_prompts_for_vllm(
    rows: list[dict[str, Any]], tokenizer: Any, instruction: str | None = None
) -> list[str]:
    """Build chat-template prompts for vLLM."""
    prompts = []  # 与 rows 等长的完整 prompt 字符串列表
    for row in rows:
        inst = instruction or row.get("instruction", SYSTEM_PROMPT)  # 优先函数传入的全局指令，否则用行内或默认
        messages = [  # ChatML 风格消息列表，交给 tokenizer 模板化
            {"role": "system", "content": inst},
            {"role": "user", "content": row["input"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # 拼出含生成起点的文本
        prompts.append(text)
    return prompts


def run_vllm_inference(
    prompts: list[str],
    model_name: str,
    max_tokens: int = 150,
    temperature: float = 0.1,
    gpu_memory_utilization: float = 0.85,
) -> list[str]:
    """Run vLLM batch inference and return decoded responses."""
    # Colab vLLM stability workaround
    os.environ.setdefault("VLLM_USE_V1", "0")  # 关闭 v1 引擎，减少部分环境下崩溃
    try:
        sys.stdout.fileno = lambda: 1  # 部分环境 fileno 异常时让 vLLM 仍能初始化
        sys.stderr.fileno = lambda: 2
    except Exception:
        pass  # 若替换失败则忽略，不影响主流程

    from vllm import LLM, SamplingParams  # 延迟导入：无 GPU 跑 synthetic-only 时可不装 vLLM

    print(f"  Loading vLLM engine for {model_name} ...")
    llm = LLM(
        model=model_name,  # HuggingFace Hub 或本地路径
        dtype="half",  # FP16，省显存
        max_model_len=2048,  # 最大上下文长度上限
        gpu_memory_utilization=gpu_memory_utilization,  # 预占显存比例，避免 OOM 或过低利用
        trust_remote_code=True,  # 部分模型需远程代码
        enforce_eager=True,  # 禁用部分 CUDA graph，换稳定性（Colab 常见）
    )
    sampling_params = SamplingParams(
        temperature=temperature,  # 略大于 0 时有一点随机性；配合 do_sample 在 vLLM 侧生效
        max_tokens=max_tokens,  # 导诊 JSON 较短即可
        stop=["<|im_end|>"],  # 遇到该停止串则截断（与部分聊天模板对齐）
    )

    print(f"  Generating {len(prompts)} responses ...")
    outputs = llm.generate(prompts, sampling_params)  # 批量生成，顺序与 prompts 一致
    return [o.outputs[0].text.strip() for o in outputs]  # 每条取第一个候选的文本并去空白


def run_model_dpo_pairs(
    rows: list[dict[str, Any]],
    model_name: str,
    max_samples: int | None,
    source_label: str,
    lora_path: str | None = None,
) -> list[dict[str, Any]]:
    """Source 1 / Source 2:  model inference → compare with gold → DPO pairs."""
    from transformers import AutoTokenizer  # 与模型同源的 chat 模板

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # 加载分词器

    # If LoRA, we need to load with peft — but vLLM doesn't support LoRA natively.
    # Fallback: use transformers + peft for LoRA inference (slower but correct).
    if lora_path:  # 指定了 LoRA：本脚本走 transformers+peft 逐条生成
        return _run_transformers_dpo_pairs(rows, model_name, lora_path, source_label, tokenizer, max_samples)

    # Base model: use vLLM for speed.
    prompts = build_prompts_for_vllm(rows, tokenizer, SYSTEM_PROMPT)  # 统一用默认 system prompt 组 batch
    responses = run_vllm_inference(prompts, model_name)  # 批量推理得到字符串输出

    pairs: list[dict[str, Any]] = []
    for row, response in zip(rows, responses):  # 严格按顺序对齐
        gold = parse_gold_output(row.get("output"))  # 数据中的教师 gold
        if gold is None:  # 无有效 gold 则无法构造偏好对
            continue
        pred = parse_gold_output(response)  # 尝试把模型输出解析为同结构 JSON
        if pred is None:
            # Can't parse → treat as very wrong → always keep
            pairs.append(
                make_dpo_record(
                    instruction=SYSTEM_PROMPT,
                    user_input=row["input"],
                    chosen_output=gold_to_compact_json(gold),
                    rejected_output=response,  # keep raw response as rejected：保留原始乱输出供学习
                    source=source_label,
                )
            )
        elif is_meaningful_error(gold, pred):  # 能解析但与 gold 差异足够大
            pairs.append(
                make_dpo_record(
                    instruction=SYSTEM_PROMPT,
                    user_input=row["input"],
                    chosen_output=gold_to_compact_json(gold),
                    rejected_output=gold_to_compact_json(pred),  # 结构化错误用紧凑 JSON 作 rejected
                    source=source_label,
                )
            )

    if max_samples is not None and len(pairs) > max_samples:  # 限制该来源条数
        random.seed(42)
        pairs = random.sample(pairs, max_samples)

    return pairs


def _run_transformers_dpo_pairs(
    rows: list[dict[str, Any]],
    model_name: str,
    lora_path: str,
    source_label: str,
    tokenizer: Any,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """LoRA inference via transformers + peft (slower fallback)."""
    import torch  # 张量设备与推理
    from tqdm import tqdm  # 长循环进度条
    from transformers import AutoModelForCausalLM  # 加载因果 LM
    from peft import PeftModel  # 将 LoRA 适配器挂到基座上

    print(f"  Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # A100/部分 GPU 上常用；省显存
        device_map="auto",  # 自动切多卡 / 单卡
        trust_remote_code=True,
    )
    print(f"  Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)  # 合并适配器权重逻辑由 peft 处理
    model.eval()  # 关闭 dropout，推理模式

    pairs: list[dict[str, Any]] = []
    for row in tqdm(rows, desc=f"  [{source_label}]"):  # 逐条：LoRA 无法像 vLLM 那样简单 batch 到同等规模
        gold = parse_gold_output(row.get("output"))
        if gold is None:
            continue

        messages = [  # 与 vLLM 路径一致的消息格式
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["input"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)  # 单条 batch=1，搬到模型所在 device

        with torch.no_grad():  # 推理不建计算图，省显存
            output_ids = model.generate(
                **inputs,  # input_ids、attention_mask 等
                max_new_tokens=150,  # 最多新生成 150 token
                do_sample=False,  # 贪心解码，稳定可复现
                pad_token_id=tokenizer.eos_token_id,  # 避免 pad_token 未设置警告
            )
        response_ids = output_ids[:, inputs.input_ids.shape[1]:]  # 去掉 prompt 部分，只保留新生成段
        response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()  # 解码为字符串

        pred = parse_gold_output(response)
        if pred is None:  # 与 vLLM 分支相同：解析失败则整段 raw 作为 rejected
            pairs.append(
                make_dpo_record(
                    instruction=SYSTEM_PROMPT,
                    user_input=row["input"],
                    chosen_output=gold_to_compact_json(gold),
                    rejected_output=response,
                    source=source_label,
                )
            )
        elif is_meaningful_error(gold, pred):
            pairs.append(
                make_dpo_record(
                    instruction=SYSTEM_PROMPT,
                    user_input=row["input"],
                    chosen_output=gold_to_compact_json(gold),
                    rejected_output=gold_to_compact_json(pred),
                    source=source_label,
                )
            )

    del model  # 显式释放大对象引用
    torch.cuda.empty_cache()  # 尽量归还 GPU 缓存

    if max_samples is not None and len(pairs) > max_samples:
        random.seed(42)
        pairs = random.sample(pairs, max_samples)

    return pairs


# ---------------------------------------------------------------------------
# Merge & split
# ---------------------------------------------------------------------------

def deduplicate_by_prompt(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by (system + user) prompt. Keep first occurrence."""
    seen: set[str] = set()  # 已出现过的 prompt 序列化键
    unique: list[dict[str, Any]] = []  # 去重后的列表（保留先出现的条目）
    dropped = 0  # 重复条数计数
    for p in pairs:
        # Build a stable key from the conversations
        key = json.dumps(p["conversations"], ensure_ascii=False, sort_keys=True)  # 键与字段顺序无关，稳定
        if key not in seen:  # 首次见到该 (system,user) 组合
            seen.add(key)
            unique.append(p)
        else:
            dropped += 1  # 同源或异源重复 prompt 则丢弃后续
    if dropped:
        print(f"  Dropped {dropped} duplicate prompt(s) across sources.")
    return unique


def split_and_save(
    pairs: list[dict[str, Any]],
    output_dir: Path,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Shuffle, split, and save DPO train/valid JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录（含父目录）

    shuffled = pairs[:]  # 浅拷贝列表，避免原地打乱影响调用方
    random.Random(seed).shuffle(shuffled)  # 用独立 Random 实例按 seed 打乱
    valid_size = max(1, round(len(shuffled) * valid_ratio))  # 验证集至少 1 条；按比例四舍五入
    valid = shuffled[:valid_size]  # 前段作为验证
    train = shuffled[valid_size:]  # 后段作为训练

    # Strip internal _source field for LLaMA-Factory compatibility
    def clean(rec: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in rec.items() if not k.startswith("_")}  # 去掉 _source 等内部字段

    train_clean = [clean(r) for r in train]  # 写入训练文件用的「干净」记录
    valid_clean = [clean(r) for r in valid]

    train_path = output_dir / "dpo_train.json"  # 训练偏好对
    valid_path = output_dir / "dpo_valid.json"  # 验证偏好对

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_clean, f, ensure_ascii=False, indent=2)  # 缩进 2 便于人工 diff
        f.write("\n")  # 文件末尾换行，符合 POSIX 习惯
    with valid_path.open("w", encoding="utf-8") as f:
        json.dump(valid_clean, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Summary report
    source_counts: Counter[str] = Counter()  # 统计各来源条数（train 用原名，valid 加后缀区分）
    for r in train:
        source_counts[r.get("_source", "unknown")] += 1  # 训练集内各 _source 计数
    for r in valid:
        source_counts[r.get("_source", "unknown") + "_valid"] += 1  # 验证集单独累计，打印时过滤掉

    print(f"\n{'='*50}")
    print(f"DPO data written to: {output_dir}")
    print(f"  dpo_train.json : {len(train)} pairs")
    print(f"  dpo_valid.json : {len(valid)} pairs")
    print(f"\nSource distribution (train):")
    for src, cnt in source_counts.most_common():  # 按计数从高到低打印
        if not src.endswith("_valid"):  # 只打印训练侧来源分布
            print(f"  {src}: {cnt}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DPO preference data for medical triage")
    # --- 数据路径与划分 ---
    p.add_argument("--train-data", default="data/processed/sft_train.json",
                   help="Cleaned SFT training data (JSON list).")
    p.add_argument("--output-dir", default="data/dpo")  # DPO json 输出目录
    p.add_argument("--valid-ratio", type=float, default=0.1)  # 验证集占比；至少 1 条见 split_and_save

    # Model settings
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Base model for Source 1 (vLLM inference).")  # Source1 与 Source2 共用基座名
    p.add_argument("--lora-path", default=None,
                   help="Path to LoRA adapter for Source 2. If None, skip Source 2.")  # 无路径则不做 gold vs SFT

    # Source toggles
    p.add_argument("--skip-source1", action="store_true", help="Skip Gold vs Base model pairs.")
    p.add_argument("--skip-source2", action="store_true", help="Skip Gold vs SFT model pairs.")
    p.add_argument("--skip-source3", action="store_true", help="Skip synthetic corruption pairs.")
    p.add_argument("--synthetic-only", action="store_true",
                   help="Only run Source 3 (no GPU needed).")  # 仅合成数据，早退不写 source1/2

    # Limits
    p.add_argument("--max-source1", type=int, default=None)  # Source1 最多保留条数；None 表示不截断
    p.add_argument("--max-source2", type=int, default=None)
    p.add_argument("--max-synthetic", type=int, default=1000)  # 合成对默认上限 1000

    # Misc
    p.add_argument("--seed", type=int, default=42)  # Python random 全局种子
    return p.parse_args()  # 解析 sys.argv


def main() -> None:
    args = parse_args()  # 读入 CLI
    random.seed(args.seed)  # 影响 shuffle、sample、合成中的随机（部分函数内部会再 seed(42)）

    data_path = Path(args.train_data)  # 训练 json 路径
    if not data_path.exists():  # 缺少输入则直接报错退出
        print(f"ERROR: training data not found: {data_path}")
        print("Run clean_sft_data.py first to generate data/processed/sft_train.json")
        sys.exit(1)

    rows = load_json(data_path)  # 加载 SFT 样本列表
    print(f"Loaded {len(rows)} training rows from {data_path}")

    all_pairs: list[dict[str, Any]] = []  # 汇总三种来源的 DPO 记录

    # ---- Source 3: Synthetic (no GPU) ----
    if not args.skip_source3:  # 默认构建合成偏好对
        print("\n--- Source 3: Synthetic corruption ---")
        syn = build_synthetic_pairs(rows, max_samples=args.max_synthetic)  # 受 max_synthetic 限制
        print(f"  Generated {len(syn)} synthetic pairs.")
        all_pairs.extend(syn)  # 拼到总列表

    if args.synthetic_only:  # 只要合成：跳过后续模型推理
        all_pairs = deduplicate_by_prompt(all_pairs)  # 按 prompt 去重
        split_and_save(all_pairs, Path(args.output_dir), args.valid_ratio, args.seed)  # 写 train/valid
        return  # 提前结束 main

    # ---- Source 1: Gold vs Base model ----
    if not args.skip_source1:  # 用基座模型跑推理，收集错误为 rejected
        print(f"\n--- Source 1: Gold vs Base model ({args.base_model}) ---")
        s1 = run_model_dpo_pairs(
            rows,
            model_name=args.base_model,
            max_samples=args.max_source1,
            source_label="gold_vs_base",  # 溯源标签
        )
        print(f"  Generated {len(s1)} pairs from base model errors.")
        all_pairs.extend(s1)

    # ---- Source 2: Gold vs SFT model ----
    if not args.skip_source2 and args.lora_path:  # 显式提供 LoRA 才跑 Source2
        print(f"\n--- Source 2: Gold vs SFT model ({args.lora_path}) ---")
        s2 = run_model_dpo_pairs(
            rows,
            model_name=args.base_model,  # LoRA 挂在该基座上
            lora_path=args.lora_path,
            max_samples=args.max_source2,
            source_label="gold_vs_sft",
        )
        print(f"  Generated {len(s2)} pairs from SFT model errors.")
        all_pairs.extend(s2)
    elif not args.skip_source2 and not args.lora_path:  # 用户想跑 source2 但没给路径 → 提示并跳过
        print("\n--- Source 2: SKIPPED (no --lora-path provided) ---")

    # ---- Merge & save ----
    all_pairs = deduplicate_by_prompt(all_pairs)  # 多源合并后按 prompt 去重
    split_and_save(all_pairs, Path(args.output_dir), args.valid_ratio, args.seed)


if __name__ == "__main__":  # 作为脚本直接运行时入口
    main()
