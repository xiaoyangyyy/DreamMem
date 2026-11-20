from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class DataConfig:
    seed: int = 123
    n_support: int = 8
    n_queries: int = 4
    near_miss_probability: float = 0.65
    support_label_noise: float = 0.0


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 96


@dataclass
class TrainConfig:
    seed: int = 17
    steps: int = 3000
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_accumulation: int = 1
    grad_clip: float = 1.0
    amp: bool = True
    eval_every: int = 250
    eval_batches: int = 8
    mask_schedule: str = "cosine"
    min_mask_ratio: float = 0.05
    concept_loss_weight: float = 1.0
    query_loss_weight: float = 1.0
    support_loss_weight: float = 1.0
    output_dir: str = "outputs/full"


@dataclass
class TrajectoryConfig:
    steps: int = 16
    samples: int = 6
    temperature: float = 1.0
    revision_probability: float = 0.35
    support_stages: list[int] = field(default_factory=lambda: [2, 4, 6])


@dataclass
class MemoryConfig:
    validation_weight: float = 4.0
    complexity_penalty: float = 0.08
    confidence_weight: float = 0.15
    frequency_weight: float = 0.2
    revision_bonus: float = 0.25
    generalization_bonus: float = 0.1
    formation_weight: float = 0.0
    specificity_bonus: float = 0.0
    generation_penalty: float = 0.0
    max_candidates: int = 48


@dataclass
class EvalConfig:
    seed: int = 900
    episodes: int = 256
    batch_size: int = 32
    bootstrap_samples: int = 1000
    tuning_trials: int = 80
    splits: list[str] = field(
        default_factory=lambda: ["seen", "unseen", "compositional"]
    )


@dataclass
class ExperimentConfig:
    name: str = "full"
    task_type: str = "concept"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    device: str = "cuda"
    seeds: list[int] = field(default_factory=lambda: [17, 29, 43])


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    known = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"Unknown {cls.__name__} field: {key}")
        default = known[key].default
        if is_dataclass(default):
            kwargs[key] = _construct(type(default), value)
        elif isinstance(value, dict) and is_dataclass(default):
            kwargs[key] = _construct(type(default), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    known = {f.name: f for f in fields(ExperimentConfig)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown experiment fields: {sorted(unknown)}")
    section_types = {
        "data": DataConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "trajectory": TrajectoryConfig,
        "memory": MemoryConfig,
        "eval": EvalConfig,
    }
    kwargs = {}
    for key, value in raw.items():
        kwargs[key] = _construct(section_types[key], value) if key in section_types else value
    return ExperimentConfig(**kwargs)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
