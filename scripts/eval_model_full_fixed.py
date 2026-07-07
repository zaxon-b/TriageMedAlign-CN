import gc
import json
import random

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib import font_manager
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class Eval_model_full:
    # [MOD] 明确定义合法标签空间，eval 时会严格检查预测和 gold 是否符合 schema。
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

    # [MOD] 可选强约束 prompt。建议测 base model 时打开 use_strong_prompt=True。
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
        data_path="/content/LLaMA-Factory/data/medical_triage_test_clean.json",
        num_test_samples=None,
        use_strong_prompt=True,
        max_new_tokens=150,
    ):
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

        # [MOD] 增加更细的错误统计和端到端指标所需计数。
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

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        # [MOD] 原类没有使用 num_test_samples。这里固定 seed 后抽样，便于快速 smoke test。
        if self.num_test_samples is not None and self.num_test_samples < len(all_data):
            random.seed(42)
            all_data = random.sample(all_data, self.num_test_samples)

        self.test_data = all_data
        print(f">>> load test data {len(all_data)} 条 <<<")

    def _extract_json_from_text(self, text):
        # [MOD] 预测必须是严格 JSON。这里不从废话里截取 {...}，否则会低估格式错误率。
        if not isinstance(text, str):
            return None
        clean_text = text.replace("```json", "").replace("```", "").strip()
        try:
            parsed_json = json.loads(clean_text)
        except Exception:
            return None
        return parsed_json if isinstance(parsed_json, dict) else None

    def _parse_gold_json(self, raw_output):
        # [MOD] gold 也走严格 schema 检查；如果原始 test 里有越界科室，会被跳过并计入 bad_gold。
        parsed_json = self._extract_json_from_text(raw_output)
        is_valid, _ = self._validate_schema(parsed_json)
        return parsed_json if is_valid else None

    def _validate_schema(self, parsed_json):
        # [MOD] 新增 schema 校验：不仅检查 key，还检查合法科室、紧急度、症状数组。
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
        # [MOD] 可切换强约束 prompt。测试 base model 建议用强约束 prompt。
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
                    do_sample=False,  # [MOD] 明确贪心解码；temperature 在 do_sample=False 时没有意义，所以去掉。
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            parsed_json = self._extract_json_from_text(response)
            schema_valid, schema_reason = self._validate_schema(parsed_json)

            self.eval_params["raw_predictions"].append(
                {
                    "input": row["input"],
                    "gold": true_json,
                    "response": response,
                    "parsed": parsed_json,
                    "schema_valid": schema_valid,
                    "schema_error": None if schema_valid else schema_reason,
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

                # [MOD] 端到端指标：格式/schema 正确且科室、紧急度都正确才算成功。
                if (
                    parsed_json["department"] == true_json["department"]
                    and parsed_json["urgency"] == true_json["urgency"]
                ):
                    self.eval_params["e2e_dept_urg_correct"] += 1
            else:
                self.eval_params["schema_errors"] += 1
                self.eval_params["y_pred_dept"].append("BAD_ANSWER")
                self.eval_params["y_pred_urgency"].append("BAD_ANSWER")
                self.eval_params["y_pred_symptoms"].append([])

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
        print(
            f"[端到端科室+紧急度准确率]: "
            f"{self.eval_params['e2e_dept_urg_correct'] / total * 100:.2f}%"
        )

        # [MOD] conditional accuracy 只在 schema 合法样本上算，并明确标注 conditional，避免误读。
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
            report = classification_report(valid_true_dept, valid_pred_dept, zero_division=0)
            print(report)

            acc_urg = accuracy_score(valid_true_urg, valid_pred_urg)
            print(f"[紧急度准确率 conditional on schema valid]: {acc_urg * 100:.2f}%")

            jaccard_scores = []
            for t_sym, p_sym in zip(valid_true_sym, valid_pred_sym):
                set_t = set(str(x).strip() for x in t_sym if str(x).strip()) if isinstance(t_sym, list) else set()
                set_p = set(str(x).strip() for x in p_sym if str(x).strip()) if isinstance(p_sym, list) else set()
                if not set_t and not set_p:
                    jaccard_scores.append(1.0)
                elif not set_t or not set_p:
                    jaccard_scores.append(0.0)
                else:
                    jaccard_scores.append(len(set_t & set_p) / len(set_t | set_p))
            avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
            print(f"[症状提取重合度 conditional on schema valid]: {avg_jaccard * 100:.2f}%")

            try:
                font_path = "SimHei.ttf"
                font_manager.fontManager.addfont(font_path)
                prop = font_manager.FontProperties(fname=font_path)
                plt.rcParams["font.family"] = prop.get_name()
                plt.rcParams["axes.unicode_minus"] = False
            except Exception:
                pass

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
        # [MOD] 新增保存预测明细，便于检查 base model 为什么错。
        with open(path, "w", encoding="utf-8") as f:
            for item in self.eval_params["raw_predictions"]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f">>> predictions saved to {path} <<<")

    def run(self):
        self._load_model()
        self._load_data()
        self._eval()
        self._plot()

        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        return
