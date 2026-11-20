from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dreammem.dcoder import load_dcoder, revision_generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/open-dcoder-0.5B")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--revision-probability", type=float, default=0.0)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.6)
    parser.add_argument("--output", default="outputs/dcoder/trial.json")
    args = parser.parse_args()
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    tokenizer, model = load_dcoder(args.model, device)
    load_seconds = time.time() - started
    prompt = "Write a Python function that returns the Fibonacci number for n.\n\n"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    if args.revision_probability > 0:
        mask_token_id = int(model.generation_config.mask_token_id)
        masks = torch.full(
            (prompt_ids.shape[0], args.tokens),
            mask_token_id,
            device=device,
            dtype=torch.long,
        )
        input_ids = torch.cat((prompt_ids, masks), dim=1)
        sequence, history, attempted_revision_rates = revision_generate(
            model,
            input_ids,
            args.steps,
            args.temperature,
            args.revision_probability,
            seed=123,
        )
        history = list(history)
    else:
        attempted_revision_rates = []
        input_ids = prompt_ids
        with torch.no_grad():
            outputs = model.diffusion_generate(
                inputs=input_ids,
                max_new_tokens=args.tokens,
                steps=args.steps,
                temperature=args.temperature,
                alg=args.alg,
                alg_temp=args.alg_temp,
                output_history=True,
                return_dict_in_generate=True,
            )
        sequence = outputs.sequences
        history = list(outputs.history or [])
    generation_seconds = time.time() - started
    generated = tokenizer.decode(
        sequence[0][prompt_ids.shape[1] :],
        skip_special_tokens=True,
    )
    revision_rates = []
    committed_revision_rates = []
    replacement_rates = []
    mask_token_id = int(model.generation_config.mask_token_id)
    for previous, current in zip(history, history[1:]):
        changed = previous != current
        committed = previous != mask_token_id
        revision_rates.append(float(changed.float().mean()))
        committed_revision_rates.append(float((changed & committed).float().mean()))
        replacement_rates.append(
            float((changed & committed & (current != mask_token_id)).float().mean())
        )
    result = {
        "model": args.model,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "requested_steps": args.steps,
        "history_steps": len(history),
        "revision_rates": revision_rates,
        "committed_revision_rates": committed_revision_rates,
        "replacement_rates": replacement_rates,
        "attempted_revision_rates": attempted_revision_rates,
        "algorithm": "custom_remask" if args.revision_probability > 0 else args.alg,
        "revision_probability": args.revision_probability,
        "peak_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
        ),
        "prompt": prompt,
        "generated": generated,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
