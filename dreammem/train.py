from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, config_to_dict, save_config
from .diffusion import corrupt_batch, masked_cross_entropy, masked_token_accuracy
from .encoding import SequenceEncoder
from .model import MaskedDiffusionTransformer
from .tasks import TaskGenerator, code_to_concept, concept_to_code
from .encoding import ATTRIBUTE_BASE, NIL


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_at(config: ExperimentConfig, step: int) -> float:
    train = config.train
    if step < train.warmup_steps:
        return train.learning_rate * (step + 1) / train.warmup_steps
    progress = (step - train.warmup_steps) / max(1, train.steps - train.warmup_steps)
    return train.learning_rate * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


def save_checkpoint(
    path: Path,
    model: MaskedDiffusionTransformer,
    config: ExperimentConfig,
    step: int,
    score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config_to_dict(config),
            "step": step,
            "score": score,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[MaskedDiffusionTransformer, ExperimentConfig, dict]:
    from .config import DataConfig, EvalConfig, MemoryConfig, ModelConfig, TrainConfig, TrajectoryConfig

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint["config"]
    config = ExperimentConfig(
        name=raw["name"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        trajectory=TrajectoryConfig(**raw["trajectory"]),
        memory=MemoryConfig(**raw["memory"]),
        eval=EvalConfig(**raw["eval"]),
        device=raw["device"],
        seeds=raw["seeds"],
    )
    model = MaskedDiffusionTransformer(config.model)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, config, checkpoint


@torch.no_grad()
def evaluate_one_shot(
    model: MaskedDiffusionTransformer,
    config: ExperimentConfig,
    split: str,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    from .encoding import MASK

    model.eval()
    generator = TaskGenerator(config.data)
    encoder = SequenceEncoder(config.data)
    totals = {"loss": 0.0, "query_accuracy": 0.0, "concept_accuracy": 0.0}
    episodes_seen = 0
    for batch_index in range(config.train.eval_batches):
        episodes = []
        for i in range(config.train.batch_size):
            index = batch_index * config.train.batch_size + i
            episodes.append(generator.sample(split, index, seed + index))
        batch = encoder.encode_batch(episodes, device=device)
        free_mask = batch.concept_mask | batch.query_mask
        inputs = batch.tokens.masked_fill(free_mask, MASK)
        ratio = torch.ones(len(episodes), device=device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs, ratio)
        loss = masked_cross_entropy(logits, batch.tokens, free_mask)
        query_accuracy = masked_token_accuracy(logits, batch.tokens, batch.query_mask)
        predictions = logits.argmax(dim=-1)[:, encoder.concept_positions]
        concept_correct = []
        for row, episode in enumerate(episodes):
            predicted = tuple(
                int(token) - ATTRIBUTE_BASE
                for token in predictions[row]
                if token not in (NIL, MASK) and int(token) >= ATTRIBUTE_BASE
            )
            concept_correct.append(float(predicted == episode.concept))
        totals["loss"] += float(loss) * len(episodes)
        totals["query_accuracy"] += float(query_accuracy) * len(episodes)
        totals["concept_accuracy"] += sum(concept_correct)
        episodes_seen += len(episodes)
    return {key: value / episodes_seen for key, value in totals.items()}


def train_model(config: ExperimentConfig, seed: int | None = None) -> Path:
    seed = config.train.seed if seed is None else seed
    set_seed(seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output_dir = Path(config.train.output_dir) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    generator = TaskGenerator(config.data)
    encoder = SequenceEncoder(config.data)
    model = MaskedDiffusionTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    use_amp = config.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(seed + 10_000)
    history_path = output_dir / "history.csv"
    best_score = -float("inf")
    started = time.time()
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "step",
                "train_loss",
                "val_loss",
                "val_query_accuracy",
                "val_concept_accuracy",
                "learning_rate",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for step in range(config.train.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for microstep in range(config.train.grad_accumulation):
                offset = step * config.train.batch_size * config.train.grad_accumulation
                episodes = [
                    generator.sample(
                        "train",
                        offset + microstep * config.train.batch_size + i,
                        seed=seed * 1_000_000 + offset + microstep * config.train.batch_size + i,
                    )
                    for i in range(config.train.batch_size)
                ]
                support_counts = [
                    2 + (offset + microstep * config.train.batch_size + i) % (config.data.n_support - 1)
                    for i in range(config.train.batch_size)
                ]
                batch = encoder.encode_batch(episodes, support_counts, device=device)
                corrupted = corrupt_batch(
                    batch,
                    torch_generator,
                    config.train.mask_schedule,
                    config.train.min_mask_ratio,
                )
                position_weights = torch.full_like(batch.tokens, config.train.support_loss_weight, dtype=torch.float32)
                position_weights = position_weights.masked_fill(
                    batch.query_mask,
                    config.train.query_loss_weight,
                ).masked_fill(
                    batch.concept_mask,
                    config.train.concept_loss_weight,
                )
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(corrupted.inputs, corrupted.mask_ratio)
                    loss = masked_cross_entropy(
                        logits,
                        corrupted.targets,
                        corrupted.loss_mask,
                        position_weights,
                    )
                    scaled_loss = loss / config.train.grad_accumulation
                scaler.scale(scaled_loss).backward()
                running_loss += float(loss.detach()) / config.train.grad_accumulation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            learning_rate = learning_rate_at(config, step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            scaler.step(optimizer)
            scaler.update()
            completed = step + 1
            if completed % config.train.eval_every == 0 or completed == config.train.steps:
                val_metrics = evaluate_one_shot(
                    model,
                    config,
                    "val",
                    seed + completed * 100,
                    device,
                )
                score = val_metrics["query_accuracy"] + val_metrics["concept_accuracy"]
                row = {
                    "step": completed,
                    "train_loss": running_loss,
                    "val_loss": val_metrics["loss"],
                    "val_query_accuracy": val_metrics["query_accuracy"],
                    "val_concept_accuracy": val_metrics["concept_accuracy"],
                    "learning_rate": learning_rate,
                    "elapsed_seconds": time.time() - started,
                }
                writer.writerow(row)
                history_file.flush()
                save_checkpoint(output_dir / "latest.pt", model, config, completed, score)
                if score > best_score:
                    best_score = score
                    save_checkpoint(output_dir / "best.pt", model, config, completed, score)
                print(json.dumps(row), flush=True)
    metadata = {
        "seed": seed,
        "steps_completed": config.train.steps,
        "parameters": model.parameter_count(),
        "device": str(device),
        "best_score": best_score,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir / "best.pt"
