import json
import re


ALLOWED_DEPTS = {
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


# 与 eval_model_full_grpo.py 保持一致。
# 目的不是覆盖所有医学同义词，而是让 reward 和 eval 对常见口语/标签变体使用同一套口径。
SYMPTOM_SYNONYMS = {
    "发烧": "发热",
    "高烧": "发热",
    "低烧": "发热",
    "头疼": "头痛",
    "脑袋疼": "头痛",
    "肚子疼": "腹痛",
    "肚子痛": "腹痛",
    "下腹疼痛": "腹痛",
    "下腹痛": "腹痛",
    "上腹疼痛": "腹痛",
    "上腹痛": "腹痛",
    "胃疼": "胃痛",
    "拉肚子": "腹泻",
    "便稀": "腹泻",
    "小便疼": "尿痛",
    "尿疼": "尿痛",
    "排尿疼痛": "尿痛",
    "胸口痛": "胸痛",
    "胸口疼": "胸痛",
    "喉咙痛": "咽痛",
    "嗓子疼": "咽痛",
    "鼻塞流涕": "鼻塞",
    "流鼻涕": "流涕",
}


# 医疗导诊里，高危低判比低危高判更严重。
# 这个矩阵替代原来的简单距离分数，避免模型靠大量预测“中”刷分。
URGENCY_REWARD_MATRIX = {
    ("高", "高"): 1.0,
    ("高", "中"): 0.2,
    ("高", "低"): -1.0,
    ("中", "高"): 0.4,
    ("中", "中"): 1.0,
    ("中", "低"): 0.2,
    ("低", "高"): 0.0,
    ("低", "中"): 0.5,
    ("低", "低"): 1.0,
}


def parse_json_safe(text):
    if not isinstance(text, str):
        return None
    clean_text = re.sub(r"```json|```", "", text).strip()
    try:
        obj = json.loads(clean_text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _completion_to_text(completion):
    # TRL 不同版本 completion 可能是 str，也可能是 chat message list。
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def _validate_schema(obj):
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != REQUIRED_KEYS:
        return False
    if obj.get("department") not in ALLOWED_DEPTS:
        return False
    if obj.get("urgency") not in ALLOWED_URGENCY:
        return False
    symptoms = obj.get("symptoms")
    if not isinstance(symptoms, list) or not symptoms:
        return False
    if not all(isinstance(x, str) and x.strip() for x in symptoms):
        return False
    return True


def normalize_symptom(text):
    s = str(text).strip()
    s = s.replace(" ", "")
    s = s.replace("，", "").replace(",", "")
    s = s.replace("。", "").replace(".", "")
    return SYMPTOM_SYNONYMS.get(s, s)


def normalize_symptom_list(symptoms):
    if not isinstance(symptoms, list):
        return []
    normalized = []
    for item in symptoms:
        if not isinstance(item, str):
            continue
        symptom = normalize_symptom(item)
        if symptom:
            normalized.append(symptom)
    return normalized


def symptom_f1(gold_symptoms, pred_symptoms):
    gold_unique = list(dict.fromkeys(normalize_symptom_list(gold_symptoms)))
    pred_unique = list(dict.fromkeys(normalize_symptom_list(pred_symptoms)))

    if not gold_unique and not pred_unique:
        return 1.0
    if not gold_unique or not pred_unique:
        return 0.0

    used_gold = set()
    hits = 0
    for pred_sym in pred_unique:
        for idx, gold_sym in enumerate(gold_unique):
            if idx in used_gold:
                continue
            exact_match = pred_sym == gold_sym
            containment_match = (
                len(pred_sym) >= 2
                and len(gold_sym) >= 2
                and (pred_sym in gold_sym or gold_sym in pred_sym)
            )
            if exact_match or containment_match:
                hits += 1
                used_gold.add(idx)
                break

    precision = hits / len(pred_unique) if pred_unique else 0.0
    recall = hits / len(gold_unique) if gold_unique else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def medical_triage_reward(prompts, completions, **kwargs):
    """GRPO reward for medical triage.

    接口保持不变：TRL GRPOTrainer 会传入 prompts, completions 和数据集额外字段。

    评分逻辑：
    1. 严格 JSON/schema gate：解析失败或 schema 不合法直接给负分，避免坏格式也拿部分奖励。
    2. 科室 reward：exact match。
    3. 紧急度 reward：风险矩阵，高危低判强惩罚。
    4. 症状 reward：与 eval 中新增的 normalized symptom F1 同口径。
    5. 轻量 penalty：Markdown、重复症状、过长症状列表和高危降级额外扣分。
    """
    gold_list = kwargs.get("output", [])
    rewards = []

    for completion, gold_raw in zip(completions, gold_list):
        gold = parse_json_safe(gold_raw) if isinstance(gold_raw, str) else gold_raw
        if not _validate_schema(gold):
            rewards.append(0.0)
            continue

        raw_completion = _completion_to_text(completion).strip()
        has_markdown = "```" in raw_completion
        pred = parse_json_safe(raw_completion)

        if pred is None:
            rewards.append(-1.0)
            continue

        if not _validate_schema(pred):
            rewards.append(-0.5)
            continue

        r_format = 0.8 if has_markdown else 1.0
        r_dept = 1.0 if pred["department"] == gold["department"] else 0.0
        r_urg = URGENCY_REWARD_MATRIX.get((gold["urgency"], pred["urgency"]), 0.0)
        r_sym = symptom_f1(gold.get("symptoms", []), pred.get("symptoms", []))

        penalty = 0.0
        pred_sym_norm = normalize_symptom_list(pred.get("symptoms", []))
        gold_sym_norm = normalize_symptom_list(gold.get("symptoms", []))

        if len(pred_sym_norm) != len(set(pred_sym_norm)):
            penalty -= 0.05
        if len(pred_sym_norm) > max(6, len(gold_sym_norm) + 3):
            penalty -= 0.05
        if has_markdown:
            penalty -= 0.05
        if gold["urgency"] == "高" and pred["urgency"] == "低":
            penalty -= 0.40
        elif gold["urgency"] == "高" and pred["urgency"] == "中":
            penalty -= 0.10

        # 当前 eval 中格式已经接近满分，所以格式只占 5%。
        # 主要优化方向放在 urgency 风险和 symptoms 抽取，避免 GRPO 只强化分类而损失症状质量。
        r_total = (
            0.05 * r_format
            + 0.25 * r_dept
            + 0.30 * r_urg
            + 0.40 * r_sym
            + penalty
        )
        rewards.append(float(max(-1.0, min(1.0, r_total))))

    return rewards
