from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .dcoder import DCoderAdapter, load_dcoder
from .dcoder_text import TextConceptEncoder
from .diffusion import corrupt_batch, masked_cross_entropy
from .task_factory import make_task_components
from .train import learning_rate_at, set_seed


def save_hf_checkpoint(
    model,
    tokenizer,
    source: Path,
    destination: Path,
    metadata: dict,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        destination,
        safe_serialization=True,
        max_shard_size="256MB",
    )
    tokenizer.save_pretrained(destination)
    for name in ("modeling_qwen2.py", "generation_utils.py", "__init__.py"):
        source_file = (source / name).resolve()
        destination_file = (destination / name).resolve()
        if source_file.exists() and source_file != destination_file:
            shutil.copy2(source_file, destination_file)
    (destination / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


@torch.no_grad()
def evaluate_conditional(
    model,
    encoder: TextConceptEncoder,
    config: ExperimentConfig,
    split: str,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    adapter = DCoderAdapter(model)
    generator = make_task_components(
        config,
        encoder.tokenizer,
        encoder.mask_token_id,
    )[0]
    query_correct = 0
    query_total = 0
    concept_correct = 0
    losses = []
    for index in range(config.train.eval_batches):
        episode = generator.sample(split, index, seed + index)
        batch = encoder.encode_batch([episode], 6, device=device)
        free = batch.concept_mask | batch.query_mask
        inputs = batch.tokens.masked_fill(free, encoder.mask_token_id)
        ratio = torch.ones(1, device=device)
        logits = adapter(inputs, ratio)
        losses.append(float(masked_cross_entropy(logits, batch.tokens, free)))
        query_correct += float((logits.argmax(-1) == batch.tokens)[batch.query_mask].sum())
        query_total += int(batch.query_mask.sum())
        predicted = encoder.decode_concept(
            logits.argmax(-1)[0, encoder.concept_positions].cpu().numpy()
        )
        concept_correct += float(predicted == episode.concept)
    return {
        "loss": float(np.mean(losses)),
        "query_accuracy": query_correct / query_total,
        "concept_accuracy": concept_correct / config.train.eval_batches,
    }


def train_dcoder(
    config: ExperimentConfig,
    model_path: str | Path = "models/open-dcoder-0.5B",
    seed: int = 17,
) -> Path:
    set_seed(seed)
    device = torch.device("cuda")
    model_path = Path(model_path)
    tokenizer, model = load_dcoder(model_path, device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    generator, encoder = make_task_components(
        config,
        tokenizer,
        int(model.generation_config.mask_token_id),
    )
    adapter = DCoderAdapter(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(seed + 10_000)
    output = Path(config.train.output_dir) / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    resume_metadata = output / "latest" / "experiment_metadata.json"
    start_step = 0
    if model_path.resolve() == resume_metadata.parent.resolve() and resume_metadata.exists():
        start_step = int(json.loads(resume_metadata.read_text(encoding="utf-8")).get("step", 0))
    best_score = -float("inf")
    best_metadata = output / "best" / "experiment_metadata.json"
    if start_step > 0 and best_metadata.exists():
        best_score = float(json.loads(best_metadata.read_text(encoding="utf-8"))["score"])
    started = time.time()
    history_path = output / "history.csv"
    history_exists = start_step > 0 and history_path.exists()
    with history_path.open("a" if history_exists else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "train_loss",
                "val_loss",
                "val_query_accuracy",
                "val_concept_accuracy",
                "learning_rate",
                "peak_memory_gib",
                "elapsed_seconds",
            ],
        )
        if not history_exists:
            writer.writeheader()
        for step in range(start_step, config.train.steps):
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            support_count = 2 + step % (config.data.n_support - 1)
            for microstep in range(config.train.grad_accumulation):
                index = step * config.train.grad_accumulation + microstep
                episode = generator.sample(
                    "train",
                    index,
                    seed * 1_000_000 + index,
                )
                batch = encoder.encode_batch([episode], support_count, device=device)
                corrupted = corrupt_batch(
                    batch,
                    torch_generator,
                    config.train.mask_schedule,
                    config.train.min_mask_ratio,
                    encoder.mask_token_id,
                )
                logits = adapter(corrupted.inputs, corrupted.mask_ratio)
                weights = torch.full_like(
                    batch.tokens,
                    config.train.support_loss_weight,
                    dtype=torch.float32,
                )
                weights = weights.masked_fill(
                    batch.query_mask,
                    config.train.query_loss_weight,
                ).masked_fill(
                    batch.concept_mask,
                    config.train.concept_loss_weight,
                )
                loss = masked_cross_entropy(
                    logits,
                    corrupted.targets,
                    corrupted.loss_mask,
                    weights,
                )
                (loss / config.train.grad_accumulation).backward()
                running_loss += float(loss.detach()) / config.train.grad_accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            learning_rate = learning_rate_at(config, step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            completed = step + 1
            if completed % config.train.eval_every == 0 or completed == config.train.steps:
                metrics = evaluate_conditional(
                    model,
                    encoder,
                    config,
                    "val",
                    seed + completed * 100,
                    device,
                )
                model.train()
                score = metrics["query_accuracy"] + metrics["concept_accuracy"]
                peak = torch.cuda.max_memory_allocated() / 1024**3
                row = {
                    "step": completed,
                    "train_loss": running_loss,
                    "val_loss": metrics["loss"],
                    "val_query_accuracy": metrics["query_accuracy"],
                    "val_concept_accuracy": metrics["concept_accuracy"],
                    "learning_rate": learning_rate,
                    "peak_memory_gib": peak,
                    "elapsed_seconds": time.time() - started,
                }
                writer.writerow(row)
                handle.flush()
                metadata = {"step": completed, "score": score, "seed": seed, **row}
                save_hf_checkpoint(model, tokenizer, model_path, output / "latest", metadata)
                if score > best_score:
                    best_score = score
                    save_hf_checkpoint(model, tokenizer, model_path, output / "best", metadata)
                print(json.dumps(row), flush=True)
    training = {
        "seed": seed,
        "steps_completed": config.train.steps,
        "best_score": best_score,
        "elapsed_seconds": time.time() - started,
        "base_model": str(model_path),
    }
    (output / "training.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    return output / "best"
