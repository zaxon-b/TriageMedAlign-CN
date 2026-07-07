#!/usr/bin/env python3
"""Clean teacher-labeled medical triage data for SFT.

The script reads the current train/test JSON files, validates and normalizes the
teacher output, removes duplicate inputs, then creates a fresh train/valid split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


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

DEPARTMENT_MAP = {
    "男科": "泌尿外科",
    "妇产科": "妇科",
}

DROP_DEPARTMENTS = {
    "内分泌科",
    "内分泌内科",
    "血液科",
    "风湿免疫科",
    "血管外科",
    "外科",
    "牙科",
    "乳腺外科",
    "内科",
    "口腔科",
    "肛肠科",
}

INSTRUCTION = (
    "你是一个中文医疗导诊助手。请根据患者主诉输出严格 JSON。"
    "department 必须从 ['儿科', '呼吸内科', '妇科', '心血管内科', '泌尿外科', "
    "'消化内科', '皮肤科', '眼科', '神经内科', '精神心理科', '耳鼻喉科', '骨科'] 中选择；"
    "urgency 必须是 ['高', '中', '低'] 之一；"
    "symptoms 必须是字符串数组。不要输出解释、Markdown 或多个 JSON。"
)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


# ============================================================
# 🔍 【第一步：JSON 解析】
# 把 Teacher 模型的原始输出文本 → 解析为 Python dict
# 处理可能包含的 markdown 代码块标记 (```json ... ```)
# ============================================================
def parse_output(raw_output: Any) -> dict[str, Any] | None:
    if not isinstance(raw_output, str):
        return None

    # 去除 markdown 代码块标记
    text = raw_output.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(text)  # 尝试解析 JSON
    except json.JSONDecodeError:
        return None  # 解析失败 → 整条丢弃

    if not isinstance(parsed, dict):
        return None  # 不是字典（如返回了数组）→ 丢弃
    return parsed


# ============================================================
# 🩺 【症状校验与归一化】
# 检查 symptoms 字段：必须是非空字符串数组，去空串、去重
# ============================================================
def normalize_symptoms(symptoms: Any) -> list[str] | None:
    if not isinstance(symptoms, list):  # 必须是数组
        return None

    cleaned: list[str] = []
    seen: set[str] = set()  # 用于去重
    for item in symptoms:
        if not isinstance(item, str):  # 每个元素必须是字符串
            return None
        symptom = item.strip()
        if not symptom:  # 跳过空字符串
            continue
        if symptom not in seen:  # 去重：如 ["头痛", "头痛"] → ["头痛"]
            cleaned.append(symptom)
            seen.add(symptom)

    if not cleaned:  # 症状列表不能为空
        return None
    return cleaned


# ============================================================
# 🧹 【单条数据的完整清洗流程】
# 依次进行：输入校验 → JSON解析 → 字段完整性 → 科室 → 紧急度 → 症状
# 任何一步失败就丢弃这条数据
# ============================================================
def normalize_row(row: dict[str, Any], stats: Counter[str]) -> dict[str, Any] | None:

    # ---------- 输入文本校验 ----------
    input_text = row.get("input")
    if not isinstance(input_text, str) or not input_text.strip():
        stats["drop_bad_input"] += 1
        return None

    # ---------- 🔍 JSON 解析（调用上面的 parse_output）----------
    parsed = parse_output(row.get("output"))
    if parsed is None:
        stats["drop_bad_json"] += 1  # JSON 解析失败 → 丢弃
        return None

    # ---------- 📋 字段完整性检查 ----------
    # 必须同时包含 department, urgency, symptoms 三个字段
    missing_keys = {"department", "urgency", "symptoms"} - set(parsed)
    if missing_keys:
        stats["drop_missing_keys"] += 1  # 缺字段 → 丢弃
        return None

    # ---------- 🏥 科室映射与丢弃 ----------
    department = parsed.get("department")
    if not isinstance(department, str):
        stats["drop_bad_department_type"] += 1
        return None
    department = department.strip()

    if department in DEPARTMENT_MAP:
        # 可纠正的科室：男科→泌尿外科, 妇产科→妇科
        stats[f"map_department:{department}->{DEPARTMENT_MAP[department]}"] += 1
        department = DEPARTMENT_MAP[department]
    elif department in DROP_DEPARTMENTS:
        # 黑名单科室（内分泌科、口腔科等 11 个）→ 直接丢弃
        stats[f"drop_department:{department}"] += 1
        return None
    elif department not in ALLOWED_DEPARTMENTS:
        # 既不在白名单也不在映射表 → 未知科室 → 丢弃
        stats[f"drop_unknown_department:{department}"] += 1
        return None

    # ---------- 🚨 紧急度校验 ----------
    # 必须是 {"高", "中", "低"} 之一
    urgency = parsed.get("urgency")
    if not isinstance(urgency, str) or urgency.strip() not in ALLOWED_URGENCY:
        stats["drop_bad_urgency"] += 1  # 不合法的紧急度 → 丢弃
        return None
    urgency = urgency.strip()

    # ---------- 🩺 症状校验（调用上面的 normalize_symptoms）----------
    symptoms = normalize_symptoms(parsed.get("symptoms"))
    if symptoms is None:
        stats["drop_bad_symptoms"] += 1  # 症状格式不合法 → 丢弃
        return None

    # ---------- ✅ 组装标准化输出 ----------
    normalized_output = {
        "department": department,
        "urgency": urgency,
        "symptoms": symptoms,
    }

    return {
        "instruction": INSTRUCTION,  # 统一替换为标准指令
        "input": input_text.strip(),
        # 紧凑 JSON（无空格），减少 token 数
        "output": json.dumps(normalized_output, ensure_ascii=False, separators=(",", ":")),
    }


# ============================================================
# 🔄 【Input 去重】
# 按患者主诉文本 (input) 精确匹配去重
# 同一主诉出现多次 → 只保留第一次出现的标注
# 同时统计「相同 input 但不同 output」的冲突数量
# ============================================================
def deduplicate(rows: list[dict[str, Any]], stats: Counter[str]) -> list[dict[str, Any]]:
    """Deduplicate by exact input, preferring rows from earlier files."""
    by_input: dict[str, dict[str, Any]] = {}  # key=主诉文本, value=整条数据
    conflict_count = 0

    for row in rows:
        key = row["input"]  # 以患者主诉作为去重 key
        existing = by_input.get(key)
        if existing is None:
            by_input[key] = row  # 首次出现 → 保留
            continue

        # 重复出现 → 丢弃
        stats["drop_duplicate_input"] += 1
        if existing["output"] != row["output"]:
            conflict_count += 1  # 标注冲突：同一主诉，不同标注结果

    stats["duplicate_output_conflicts"] = conflict_count
    return list(by_input.values())


def split_rows(
    rows: list[dict[str, Any]], valid_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    valid_size = max(1, round(len(shuffled) * valid_ratio))
    valid_rows = shuffled[:valid_size]
    train_rows = shuffled[valid_size:]
    return train_rows, valid_rows


def dump_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_report(path: Path, rows: list[dict[str, Any]], stats: Counter[str]) -> None:
    dept_counter: Counter[str] = Counter()
    urgency_counter: Counter[str] = Counter()

    for row in rows:
        output = json.loads(row["output"])
        dept_counter[output["department"]] += 1
        urgency_counter[output["urgency"]] += 1

    report = {
        "num_clean_rows": len(rows),
        "department_distribution": dict(dept_counter.most_common()),
        "urgency_distribution": dict(urgency_counter.most_common()),
        "cleaning_stats": dict(stats.most_common()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=[
            "medical_triage_train_clean (1).json",
            "medical_triage_test_clean.json",
        ],
        help="Input JSON files to clean and merge.",
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    stats: Counter[str] = Counter()

    # ========== 阶段1: 逐条清洗（JSON解析 + 字段校验 + 科室映射）==========
    normalized_rows: list[dict[str, Any]] = []
    for input_file in args.input_files:
        path = Path(input_file)
        raw_rows = load_json(path)
        stats[f"source_rows:{path.name}"] = len(raw_rows)
        for row in raw_rows:
            normalized = normalize_row(row, stats)  # 单条完整清洗
            if normalized is not None:
                normalized_rows.append(normalized)

    # ========== 阶段2: 全局去重 ==========
    deduped_rows = deduplicate(normalized_rows, stats)

    # ========== 阶段3: Train/Valid 切分（90%/10%）==========
    train_rows, valid_rows = split_rows(deduped_rows, args.valid_ratio, args.seed)

    # ========== 阶段4: 输出文件 ==========
    dump_json(output_dir / "sft_train.json", train_rows)
    dump_json(output_dir / "sft_valid.json", valid_rows)
    dump_json(output_dir / "sft_all_clean.json", deduped_rows)
    write_report(output_dir / "sft_clean_report.json", deduped_rows, stats)  # 清洗统计报告

    print(f"Clean rows: {len(deduped_rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Valid rows: {len(valid_rows)}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
