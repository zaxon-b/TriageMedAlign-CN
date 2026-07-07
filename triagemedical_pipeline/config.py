"""Configuration dataclasses and optional YAML/JSON loading."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PathConfig:
    raw_train: str = "medical_triage_train_clean (1).json"
    raw_test: str = "medical_triage_test_clean.json"
    processed_dir: str = "data/processed"
    sft_train: str = "data/processed/sft_train.json"
    sft_valid: str = "data/processed/sft_valid.json"
    sft_all: str = "data/processed/sft_all_clean.json"
    dpo_train: str = "dpo_train_2.json"
    dpo_valid: str = "dpo_valid_2.json"
    outputs_dir: str = "outputs"


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    sft_lora_path: str = "saves/qwen2.5-7b/lora/exp1_lora_r8_all"
    dpo_lora_path: str = "saves/qwen2.5-7b/lora/dpo_v4_r8_beta2_3epoch"
    dpo_merged_path: str = "saves/qwen2.5-7b/dpo_merged"
    grpo_lora_path: str = "outputs-grpo-v1/lora_adapter"
    template: str = "qwen"


@dataclass
class SFTConfig:
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    cutoff_len: int = 512
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    num_train_epochs: float = 3.0
    warmup_ratio: float = 0.1


@dataclass
class DPOConfig:
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    pref_beta: float = 2.0
    pref_loss: str = "sigmoid"
    pref_ftx: float = 0.1
    cutoff_len: int = 512
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-6
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1


@dataclass
class GRPOConfig:
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    num_generations: int = 4
    max_completion_length: int = 256
    temperature: float = 0.9
    beta: float = 0.04
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    num_train_epochs: int = 2
    warmup_ratio: float = 0.1


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML configs") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_config(path: str | Path | None = None) -> PipelineConfig:
    cfg = PipelineConfig()
    if path is None:
        return cfg
    data = _load_mapping(Path(path))
    for section_name, section_data in data.items():
        if not hasattr(cfg, section_name) or not isinstance(section_data, dict):
            continue
        section = getattr(cfg, section_name)
        for key, value in section_data.items():
            if hasattr(section, key):
                setattr(section, key, value)
    return cfg
