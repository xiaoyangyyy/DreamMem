from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from transformers import AutoTokenizer


def load_dcoder(path: str | Path, device: torch.device):
    path = Path(path)
    if path.suffix == ".pt" or (path / "nanodiff_checkpoint.pt").exists():
        from .nanodiff_adapter import load_nanodiff

        return load_nanodiff(path, device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    spec = importlib.util.spec_from_file_location("dcoder_modeling", path / "modeling_qwen2.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load local diffusion model implementation")
    modeling = importlib.util.module_from_spec(spec)
    import transformers.utils

    def documentation_passthrough(*args, **kwargs):
        return lambda function: function

    transformers.utils.add_start_docstrings = documentation_passthrough
    transformers.utils.replace_return_docstrings = documentation_passthrough
    spec.loader.exec_module(modeling)

    def compatible_sdpa(query, key, value, attention_mask=None, dropout=0.0, is_causal=False, **kwargs):
        repetitions = query.shape[1] // key.shape[1]
        if repetitions > 1:
            key = key.repeat_interleave(repetitions, dim=1)
            value = value.repeat_interleave(repetitions, dim=1)
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            is_causal=is_causal,
        )
        return output, None

    modeling.ALL_ATTENTION_FUNCTIONS["sdpa"] = compatible_sdpa
    model = modeling.Qwen2ForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.generation_config.do_sample = True
    return tokenizer, model


class DCoderAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, causal: bool = False) -> None:
        super().__init__()
        self.model = model
        self.causal = causal

    def forward(self, tokens: torch.Tensor, mask_ratio: torch.Tensor, positions=None) -> torch.Tensor:
        if getattr(self.model.config, "backend", None) == "nanodiff":
            if self.causal:
                raise ValueError("nanoDiff does not expose an autoregressive attention mode")
            logits = self.model(tokens)
        else:
            pad_token_id = self.model.config.pad_token_id
            attention_mask = (tokens != pad_token_id).long() if pad_token_id is not None else None
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
                is_causal=self.causal,
            )
            logits = outputs.logits
        if positions is None:
            return torch.cat((logits[:, :1], logits[:, :-1]), dim=1)
        indices = [max(0, int(position) - 1) for position in positions]
        return logits[:, indices]

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self


@torch.no_grad()
def revision_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    steps: int,
    temperature: float,
    revision_probability: float,
    seed: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[float]]:
    mask_token_id = int(model.generation_config.mask_token_id)
    pad_token_id = model.config.pad_token_id
    free_mask = input_ids == mask_token_id
    if not free_mask.any():
        raise ValueError("Input must contain mask tokens")
    state = input_ids.clone()
    generator = torch.Generator(device=state.device)
    generator.manual_seed(seed)
    committed_confidence = torch.zeros_like(state, dtype=torch.float32)
    history = []
    attempted_revision_rates = []
    for step in range(steps):
        attention_mask = (
            (state != pad_token_id).long() if pad_token_id is not None else None
        )
        outputs = model(
            input_ids=state,
            attention_mask=attention_mask,
            is_causal=False,
        )
        logits = torch.cat((outputs.logits[:, :1], outputs.logits[:, :-1]), dim=1)
        logits = logits.float() / max(temperature, 1e-4)
        probabilities = logits.softmax(dim=-1)
        uniform = torch.rand(
            probabilities.shape,
            device=state.device,
            generator=generator,
        ).clamp_min(1e-6)
        gumbel = -torch.log((-torch.log(uniform)).clamp_min(1e-6))
        predicted = (logits + gumbel).argmax(dim=-1)
        confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
        if step == steps - 1:
            state = torch.where(free_mask, predicted, state)
            history.append(state.clone())
            break
        progress = (step + 1) / steps
        n_free = int(free_mask[0].sum())
        desired = max(1, int(torch.ceil(torch.tensor(progress * n_free))))
        update = (state == mask_token_id) & free_mask
        for row in range(state.shape[0]):
            committed = ((state[row] != mask_token_id) & free_mask[row]).sum()
            missing = int(desired - committed)
            if missing > 0:
                positions = torch.nonzero(update[row], as_tuple=False).flatten()
                order = torch.argsort(confidence[row, positions], descending=True)
                update[row, positions[order[missing:]]] = False
            committed_positions = torch.nonzero(
                (state[row] != mask_token_id) & free_mask[row],
                as_tuple=False,
            ).flatten()
            revise_count = int(len(committed_positions) * revision_probability * (1.0 - progress))
            if revise_count:
                jitter = torch.rand(
                    len(committed_positions),
                    device=state.device,
                    generator=generator,
                ) * 0.05
                order = torch.argsort(committed_confidence[row, committed_positions] + jitter)
                update[row, committed_positions[order[:revise_count]]] = True
        attempted_revision_rates.append(
            float((update & (state != mask_token_id)).float().mean())
        )
        state = torch.where(update, predicted, state)
        committed_confidence = torch.where(update, confidence, committed_confidence)
        history.append(state.clone())
    return state, history, attempted_revision_rates
