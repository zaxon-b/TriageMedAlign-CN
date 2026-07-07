import gc
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib import font_manager
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class Eval_model_full:
    """Evaluation class for structured Chinese medical triage.

    接口保持你给出的版本不变：
    - __init__ 参数不变
    - run() 不变
    - save_predictions(path) 不变

    新增功能：
    1. 紧急度 Macro-F1、混淆矩阵、高危召回、高危低判数量。
    2. 与 reward 同口径的 normalized symptom F1/Jaccard。
    3. compare_prediction_files() 静态方法，用于 DPO vs GRPO 成对比较。
    """

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

    # 与 scripts/medical_triage_grpo_reward.py 保持一致。
    # 这样 reward 中的症状归一化不会和 eval 完全脱节。
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

    STRONG_SYSTEM_PROMPT = (
        "你是一个中文医疗导诊助手。请根据患者主诉输出严格 JSON。"
        "department 必须从 ['儿科', '呼吸内科', '妇科', '心血管内科', '泌尿外科', "
        "'消化内科', '皮肤科', '眼科', '神经内科', '精神心理科', '耳鼻喉科', '骨科'] 中选择；"
        "urgency 必须是 ['高', '中', '低'] 之一；"
        "symptoms 必须是字符串数组。不要输出解释、Markdown 或多个 JSON。"
    )

    def __init__(
        self,
        mode,
        model_path,
        lora_path,
        eval_prompt,
        data_path="/content/LLaMA-Factory/data/sft_valid.json",
        num_test_samples=None,
        use_strong_prompt=True,
        max_new_tokens=150,
        enable_semantic_jaccard=True,
        semantic_threshold=0.85,
    ):
        print("new eval model with semantic jaccard + normalized symptom metrics")
        self.model_path = model_path
        self.lora_path = lora_path
        self.data_path = data_path
        self.num_test_samples = num_test_samples
        self.mode = mode
        self.tokenizer = None
        self.model = None
        self.test_data = None
        self.eval_prompt = eval_prompt
        self.use_strong_prompt = use_strong_prompt
        self.max_new_tokens = max_new_tokens
        self.enable_semantic_jaccard = enable_semantic_jaccard
        self.semantic_threshold = semantic_threshold
        self.embedding_model = None

        self.eval_params = {
            "y_true_dept": [],
            "y_pred_dept": [],
            "y_true_urgency": [],
            "y_pred_urgency": [],
            "y_true_symptoms": [],
            "y_pred_symptoms": [],
            "format_errors": 0,
            "schema_errors": 0,
            "bad_gold": 0,
            "json_parse_ok": 0,
            "schema_valid": 0,
            "e2e_dept_urg_correct": 0,
            "raw_predictions": [],
        }

    @classmethod
    def normalize_symptom(cls, text):
        s = str(text).strip()
        s = s.replace(" ", "")
        s = s.replace("，", "").replace(",", "")
        s = s.replace("。", "").replace(".", "")
        return cls.SYMPTOM_SYNONYMS.get(s, s)

    @classmethod
    def normalize_symptom_list(cls, symptoms):
        if not isinstance(symptoms, list):
            return []
        normalized = []
        for item in symptoms:
            if not isinstance(item, str):
                continue
            symptom = cls.normalize_symptom(item)
            if symptom:
                normalized.append(symptom)
        return normalized

    @classmethod
    def normalized_symptom_scores(cls, true_symptoms, pred_symptoms):
        """Return normalized F1 and Jaccard under the same matching rule as reward."""
        gold_unique = list(dict.fromkeys(cls.normalize_symptom_list(true_symptoms)))
        pred_unique = list(dict.fromkeys(cls.normalize_symptom_list(pred_symptoms)))

        if not gold_unique and not pred_unique:
            return 1.0, 1.0
        if not gold_unique or not pred_unique:
            return 0.0, 0.0

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
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        union = len(gold_unique) + len(pred_unique) - hits
        jaccard = hits / union if union > 0 else 0.0
        return f1, jaccard

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.mode == "lora":
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)

        self.model.eval()
        print(f">>> model load successfully from {self.model_path} <<<")

        if self.enable_semantic_jaccard:
            print(">>> loading embedding model (BAAI/bge-small-zh-v1.5) for Semantic Jaccard <<<")
            self.embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        if self.num_test_samples is not None and self.num_test_samples < len(all_data):
            random.seed(42)
            all_data = random.sample(all_data, self.num_test_samples)

        self.test_data = all_data
        print(f">>> load test data {len(all_data)} 条 <<<")

    def _extract_json_from_text(self, text):
        if not isinstance(text, str):
            return None
        clean_text = text.replace("```json", "").replace("```", "").strip()
        try:
            parsed_json = json.loads(clean_text)
        except Exception:
            return None
        return parsed_json if isinstance(parsed_json, dict) else None

    def _parse_gold_json(self, raw_output):
        parsed_json = self._extract_json_from_text(raw_output)
        is_valid, _ = self._validate_schema(parsed_json)
        return parsed_json if is_valid else None

    def _validate_schema(self, parsed_json):
        if not isinstance(parsed_json, dict):
            return False, "not_dict"
        if set(parsed_json.keys()) != self.REQUIRED_KEYS:
            return False, "bad_keys"
        if parsed_json.get("department") not in self.ALLOWED_DEPARTMENTS:
            return False, "bad_department"
        if parsed_json.get("urgency") not in self.ALLOWED_URGENCY:
            return False, "bad_urgency"
        symptoms = parsed_json.get("symptoms")
        if not isinstance(symptoms, list) or not symptoms:
            return False, "bad_symptoms"
        if not all(isinstance(x, str) and x.strip() for x in symptoms):
            return False, "bad_symptom_item"
        return True, "ok"

    def _build_prompt(self, row):
        system_prompt = self.STRONG_SYSTEM_PROMPT if self.use_strong_prompt else row["instruction"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["input"]},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _eval(self):
        for row in tqdm(self.test_data):
            true_json = self._parse_gold_json(row.get("output", ""))
            if true_json is None:
                self.eval_params["bad_gold"] += 1
                continue

            self.eval_params["y_true_dept"].append(true_json["department"])
            self.eval_params["y_true_urgency"].append(true_json["urgency"])
            self.eval_params["y_true_symptoms"].append(true_json["symptoms"])

            text = self._build_prompt(row)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            parsed_json = self._extract_json_from_text(response)
            schema_valid, schema_reason = self._validate_schema(parsed_json)

            pred_dept = parsed_json["department"] if schema_valid else "BAD_ANSWER"
            pred_urgency = parsed_json["urgency"] if schema_valid else "BAD_ANSWER"
            dept_correct = schema_valid and pred_dept == true_json["department"]
            urgency_correct = schema_valid and pred_urgency == true_json["urgency"]
            e2e_correct = dept_correct and urgency_correct

            self.eval_params["raw_predictions"].append(
                {
                    "input": row["input"],
                    "gold": true_json,
                    "response": response,
                    "parsed": parsed_json,
                    "schema_valid": schema_valid,
                    "schema_error": None if schema_valid else schema_reason,
                    "pred_department": pred_dept,
                    "pred_urgency": pred_urgency,
                    "dept_correct": dept_correct,
                    "urgency_correct": urgency_correct,
                    "e2e_dept_urg_correct": e2e_correct,
                    "high_to_low": true_json["urgency"] == "高" and pred_urgency == "低",
                }
            )

            if parsed_json is not None:
                self.eval_params["json_parse_ok"] += 1
            else:
                self.eval_params["format_errors"] += 1

            if schema_valid:
                self.eval_params["schema_valid"] += 1
                self.eval_params["y_pred_dept"].append(parsed_json["department"])
                self.eval_params["y_pred_urgency"].append(parsed_json["urgency"])
                self.eval_params["y_pred_symptoms"].append(parsed_json["symptoms"])
                if e2e_correct:
                    self.eval_params["e2e_dept_urg_correct"] += 1
            else:
                self.eval_params["schema_errors"] += 1
                self.eval_params["y_pred_dept"].append("BAD_ANSWER")
                self.eval_params["y_pred_urgency"].append("BAD_ANSWER")
                self.eval_params["y_pred_symptoms"].append([])

    def _print_urgency_metrics(self, valid_true_urg, valid_pred_urg):
        urg_labels = ["高", "中", "低"]
        report_dict = classification_report(
            valid_true_urg,
            valid_pred_urg,
            labels=urg_labels,
            zero_division=0,
            output_dict=True,
        )
        report_text = classification_report(
            valid_true_urg,
            valid_pred_urg,
            labels=urg_labels,
            zero_division=0,
        )
        print("[紧急度分类报告 conditional on schema valid]:")
        print(report_text)
        print(f"[紧急度 Macro-F1 conditional on schema valid]: {report_dict['macro avg']['f1-score'] * 100:.2f}%")

        all_true = self.eval_params["y_true_urgency"]
        all_pred = self.eval_params["y_pred_urgency"]
        high_total = sum(1 for x in all_true if x == "高")
        high_correct = sum(1 for t, p in zip(all_true, all_pred) if t == "高" and p == "高")
        high_to_mid = sum(1 for t, p in zip(all_true, all_pred) if t == "高" and p == "中")
        high_to_low = sum(1 for t, p in zip(all_true, all_pred) if t == "高" and p == "低")
        high_to_bad = sum(1 for t, p in zip(all_true, all_pred) if t == "高" and p == "BAD_ANSWER")

        print(f"[高紧急度样本数]: {high_total}")
        if high_total > 0:
            print(f"[高紧急度召回率 all samples]: {high_correct / high_total * 100:.2f}%")
        print(f"[高危 -> 中 数量]: {high_to_mid}")
        print(f"[高危 -> 低 数量]: {high_to_low}")
        print(f"[高危 -> BAD_ANSWER 数量]: {high_to_bad}")

        urg_cm = confusion_matrix(valid_true_urg, valid_pred_urg, labels=urg_labels)
        print("[紧急度混淆矩阵 conditional on schema valid]:")
        print("labels:", urg_labels)
        print(urg_cm)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            urg_cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=urg_labels,
            yticklabels=urg_labels,
            annot_kws={"size": 12},
        )
        plt.title(f"Urgency Confusion Matrix_{self.eval_prompt}", fontsize=14, pad=12)
        plt.xlabel("Predicted Urgency")
        plt.ylabel("True Urgency")
        plt.tight_layout()
        save_path = f"confusion_matrix_urgency_[{self.mode}]_{self.eval_prompt}.png"
        plt.savefig(save_path, dpi=300)
        print(f"紧急度混淆矩阵已保存至 {save_path}")

    def _print_symptom_metrics(self, valid_true_sym, valid_pred_sym):
        hard_jaccard_scores = []
        semantic_jaccard_scores = []
        normalized_f1_scores = []
        normalized_jaccard_scores = []

        for true_symptoms, pred_symptoms in zip(valid_true_sym, valid_pred_sym):
            set_t = (
                set(str(x).strip() for x in true_symptoms if str(x).strip())
                if isinstance(true_symptoms, list)
                else set()
            )
            set_p = (
                set(str(x).strip() for x in pred_symptoms if str(x).strip())
                if isinstance(pred_symptoms, list)
                else set()
            )

            if not set_t and not set_p:
                hard_jaccard_scores.append(1.0)
            elif not set_t or not set_p:
                hard_jaccard_scores.append(0.0)
            else:
                hard_jaccard_scores.append(len(set_t & set_p) / len(set_t | set_p))

            norm_f1, norm_jaccard = self.normalized_symptom_scores(true_symptoms, pred_symptoms)
            normalized_f1_scores.append(norm_f1)
            normalized_jaccard_scores.append(norm_jaccard)

            if self.enable_semantic_jaccard:
                if not set_t and not set_p:
                    semantic_jaccard_scores.append(1.0)
                elif not set_t or not set_p:
                    semantic_jaccard_scores.append(0.0)
                else:
                    gold_list = list(set_t)
                    pred_list = list(set_p)
                    gold_vecs = self.embedding_model.encode(gold_list)
                    pred_vecs = self.embedding_model.encode(pred_list)
                    sim_matrix = cosine_similarity(pred_vecs, gold_vecs)

                    hits = 0
                    matched_gold = set()
                    for i in range(len(pred_list)):
                        best_gold = int(np.argmax(sim_matrix[i]))
                        if sim_matrix[i][best_gold] >= self.semantic_threshold and best_gold not in matched_gold:
                            hits += 1
                            matched_gold.add(best_gold)

                    soft_union = len(gold_list) + len(pred_list) - hits
                    semantic_jaccard_scores.append(hits / soft_union if soft_union > 0 else 0.0)

        print(f"[症状提取重合度 (字面 Hard Jaccard)]: {np.mean(hard_jaccard_scores) * 100:.2f}%")
        print(f"[症状提取 Normalized F1 - reward 同口径]: {np.mean(normalized_f1_scores) * 100:.2f}%")
        print(f"[症状提取 Normalized Jaccard - reward 同口径]: {np.mean(normalized_jaccard_scores) * 100:.2f}%")
        if self.enable_semantic_jaccard:
            print(f"🌟 [症状提取重合度 (语义 Soft Jaccard)]: {np.mean(semantic_jaccard_scores) * 100:.2f}%")

    def _set_chinese_font(self):
        try:
            font_path = "SimHei.ttf"
            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

    def _plot(self):
        print("=" * 50)
        total = len(self.eval_params["y_true_dept"])
        if total == 0:
            print("没有可评估样本：可能是 gold 数据全部不符合 schema。")
            print(f"[跳过的坏 gold 数量]: {self.eval_params['bad_gold']}")
            print("=" * 50)
            return

        format_errors = self.eval_params["format_errors"]
        schema_valid = self.eval_params["schema_valid"]

        print(f"[评估样本数]: {total}")
        print(f"[跳过的坏 gold 数量]: {self.eval_params['bad_gold']}")
        print(f"[JSON 解析成功率]: {self.eval_params['json_parse_ok'] / total * 100:.2f}%")
        print(f"[格式错误数量]: {format_errors} / {total}")
        print(f"[格式错误率]: {format_errors / total * 100:.2f}%")
        print(f"[Schema 合法数量]: {schema_valid} / {total}")
        print(f"[Schema 合法率]: {schema_valid / total * 100:.2f}%")
        print(f"[端到端科室+紧急度准确率]: {self.eval_params['e2e_dept_urg_correct'] / total * 100:.2f}%")

        valid_data = [
            (td, pd, tu, pu, ts, ps)
            for td, pd, tu, pu, ts, ps in zip(
                self.eval_params["y_true_dept"],
                self.eval_params["y_pred_dept"],
                self.eval_params["y_true_urgency"],
                self.eval_params["y_pred_urgency"],
                self.eval_params["y_true_symptoms"],
                self.eval_params["y_pred_symptoms"],
            )
            if pd != "BAD_ANSWER"
        ]

        if len(valid_data) > 0:
            valid_true_dept, valid_pred_dept, valid_true_urg, valid_pred_urg, valid_true_sym, valid_pred_sym = zip(
                *valid_data
            )

            acc_dept = accuracy_score(valid_true_dept, valid_pred_dept)
            print(f"[科室准确率 conditional on schema valid]: {acc_dept * 100:.2f}%")
            print(classification_report(valid_true_dept, valid_pred_dept, zero_division=0))

            acc_urg = accuracy_score(valid_true_urg, valid_pred_urg)
            print(f"[紧急度准确率 conditional on schema valid]: {acc_urg * 100:.2f}%")
            self._set_chinese_font()
            self._print_urgency_metrics(valid_true_urg, valid_pred_urg)
            self._print_symptom_metrics(valid_true_sym, valid_pred_sym)

            labels = sorted(list(set(valid_true_dept) | set(valid_pred_dept)))
            cm = confusion_matrix(valid_true_dept, valid_pred_dept, labels=labels)

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                annot_kws={"size": 12},
            )
            plt.title(f"Confusion Matrix_{self.eval_prompt}", fontsize=18, pad=20)
            plt.xlabel("Predicted Department", fontsize=14)
            plt.ylabel("True Department", fontsize=14)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            save_path = f"confusion_matrix_[{self.mode}]_{self.eval_prompt}.png"
            plt.savefig(save_path, dpi=300)
            print(f"混淆矩阵已保存至 {save_path}")
        else:
            print("所有预测 schema 都错误，无法计算 conditional 指标。")

        print("=" * 50)

    def save_predictions(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in self.eval_params["raw_predictions"]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f">>> predictions saved to {path} <<<")

    @staticmethod
    def compare_prediction_files(
        baseline_path,
        candidate_path,
        baseline_name="DPO",
        candidate_name="GRPO",
        show_examples=3,
    ):
        """Compare two saved prediction jsonl files sample by sample."""

        def load_jsonl(path):
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows

        def status(item):
            gold = item.get("gold") if isinstance(item.get("gold"), dict) else {}
            parsed = item.get("parsed") if isinstance(item.get("parsed"), dict) else None
            schema_valid = bool(item.get("schema_valid")) and parsed is not None
            pred_dept = parsed.get("department") if schema_valid else "BAD_ANSWER"
            pred_urg = parsed.get("urgency") if schema_valid else "BAD_ANSWER"
            return {
                "gold_urg": gold.get("urgency"),
                "pred_urg": pred_urg,
                "dept_correct": pred_dept == gold.get("department"),
                "urgency_correct": pred_urg == gold.get("urgency"),
                "e2e_correct": pred_dept == gold.get("department") and pred_urg == gold.get("urgency"),
            }

        baseline_rows = load_jsonl(baseline_path)
        candidate_rows = load_jsonl(candidate_path)
        n = min(len(baseline_rows), len(candidate_rows))
        mismatch_inputs = 0
        b_right_c_right = 0
        b_right_c_wrong = 0
        b_wrong_c_right = 0
        b_wrong_c_wrong = 0
        b_urg_right_c_wrong = 0
        b_urg_wrong_c_right = 0
        high_total = 0
        b_high_correct = 0
        c_high_correct = 0
        b_high_to_low = 0
        c_high_to_low = 0
        improved = []
        regressed = []

        for i in range(n):
            b_item = baseline_rows[i]
            c_item = candidate_rows[i]
            if b_item.get("input") != c_item.get("input"):
                mismatch_inputs += 1

            b = status(b_item)
            c = status(c_item)

            if b["e2e_correct"] and c["e2e_correct"]:
                b_right_c_right += 1
            elif b["e2e_correct"] and not c["e2e_correct"]:
                b_right_c_wrong += 1
                regressed.append((i, b_item, c_item))
            elif not b["e2e_correct"] and c["e2e_correct"]:
                b_wrong_c_right += 1
                improved.append((i, b_item, c_item))
            else:
                b_wrong_c_wrong += 1

            if b["urgency_correct"] and not c["urgency_correct"]:
                b_urg_right_c_wrong += 1
            elif not b["urgency_correct"] and c["urgency_correct"]:
                b_urg_wrong_c_right += 1

            if b["gold_urg"] == "高":
                high_total += 1
                if b["pred_urg"] == "高":
                    b_high_correct += 1
                if c["pred_urg"] == "高":
                    c_high_correct += 1
                if b["pred_urg"] == "低":
                    b_high_to_low += 1
                if c["pred_urg"] == "低":
                    c_high_to_low += 1

        def pct(x, den):
            return x / den * 100 if den else 0.0

        print("=" * 50)
        print(f"[成对比较]: {baseline_name} vs {candidate_name}")
        print(f"[比较样本数]: {n}")
        print(f"[input 顺序不一致数量]: {mismatch_inputs}")
        print("-" * 50)
        print("[端到端 科室+紧急度 成对分析]")
        print(f"{baseline_name} 对, {candidate_name} 对: {b_right_c_right}")
        print(f"{baseline_name} 对, {candidate_name} 错: {b_right_c_wrong}")
        print(f"{baseline_name} 错, {candidate_name} 对: {b_wrong_c_right}")
        print(f"{baseline_name} 错, {candidate_name} 错: {b_wrong_c_wrong}")
        print(f"[净修正数量]: {b_wrong_c_right - b_right_c_wrong}")
        print("-" * 50)
        print("[紧急度 成对分析]")
        print(f"{baseline_name} 对, {candidate_name} 错: {b_urg_right_c_wrong}")
        print(f"{baseline_name} 错, {candidate_name} 对: {b_urg_wrong_c_right}")
        print(f"[紧急度净修正数量]: {b_urg_wrong_c_right - b_urg_right_c_wrong}")
        print("-" * 50)
        print("[高紧急度安全指标]")
        print(f"[高紧急度样本数]: {high_total}")
        print(f"[{baseline_name} 高紧急度召回率]: {pct(b_high_correct, high_total):.2f}%")
        print(f"[{candidate_name} 高紧急度召回率]: {pct(c_high_correct, high_total):.2f}%")
        print(f"[{baseline_name} 高危 -> 低 数量]: {b_high_to_low}")
        print(f"[{candidate_name} 高危 -> 低 数量]: {c_high_to_low}")

        if show_examples > 0:
            print("-" * 50)
            print(f"[{candidate_name} 修正 {baseline_name} 的样例 Top {show_examples}]")
            for idx, b_item, c_item in improved[:show_examples]:
                print(f"\n# sample {idx}")
                print("input:", b_item.get("input"))
                print("gold:", b_item.get("gold"))
                print(f"{baseline_name}:", b_item.get("parsed") or b_item.get("response"))
                print(f"{candidate_name}:", c_item.get("parsed") or c_item.get("response"))

            print("-" * 50)
            print(f"[{candidate_name} 相对 {baseline_name} 回退的样例 Top {show_examples}]")
            for idx, b_item, c_item in regressed[:show_examples]:
                print(f"\n# sample {idx}")
                print("input:", b_item.get("input"))
                print("gold:", b_item.get("gold"))
                print(f"{baseline_name}:", b_item.get("parsed") or b_item.get("response"))
                print(f"{candidate_name}:", c_item.get("parsed") or c_item.get("response"))
        print("=" * 50)

    def run(self):
        self._load_model()
        self._load_data()
        self._eval()
        self._plot()

        del self.model
        del self.tokenizer
        if self.embedding_model:
            del self.embedding_model
        torch.cuda.empty_cache()
        gc.collect()
        return
