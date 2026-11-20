from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint as checkpoint_block


class GPT2TokenizerAdapter:
    def __init__(self) -> None:
        import tiktoken

        self.encoding = tiktoken.get_encoding("gpt2")

    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=self.encoding.encode(text))

    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer_backend.json").write_text(
            json.dumps({"backend": "tiktoken", "encoding": "gpt2"}, indent=2),
            encoding="utf-8",
        )


class NanoDiffModelMixin:
    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def forward(self, idx):
        _, length = idx.shape
        assert length <= self.config.block_size, f"sequence length {length} > block_size"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[:length].view(1, 1, length, -1).to(x.dtype)
        sin = self.rope_sin[:length].view(1, 1, length, -1).to(x.dtype)
        use_checkpoint = (
            getattr(self, "gradient_checkpointing", False)
            and self.training
            and torch.is_grad_enabled()
        )
        for block in self.blocks:
            if use_checkpoint:
                x = checkpoint_block(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        return self.lm_head(self.norm(x))

    def save_pretrained(self, destination, safe_serialization=True, max_shard_size="256MB") -> None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": self.state_dict(), "config": self.config},
            destination / "nanodiff_checkpoint.pt",
        )


def load_nanodiff(path: str | Path, device: torch.device):
    root = Path(__file__).resolve().parents[1] / "vendor" / "nanoDiff"
    sys.path.insert(0, str(root))
    from nanodiff.model import NanoDiff
    from nanodiff.utils import load_checkpoint

    path = Path(path)
    if path.is_dir():
        path = path / "nanodiff_checkpoint.pt"
    checkpoint = load_checkpoint(path, map_location="cpu")
    config = checkpoint["config"]
    config.backend = "nanodiff"
    model = NanoDiffModelMixin._mixin_model(NanoDiff, config)
    model.load_state_dict(checkpoint["model"])
    model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.generation_config = SimpleNamespace(mask_token_id=config.mask_token_id, do_sample=True)
    model.config.pad_token_id = None
    model.eval()
    return GPT2TokenizerAdapter(), model


def _mixin_model(cls, config):
    model = cls(config)
    model.forward = NanoDiffModelMixin.forward.__get__(model)
    model.gradient_checkpointing_enable = NanoDiffModelMixin.gradient_checkpointing_enable.__get__(model)
    model.save_pretrained = NanoDiffModelMixin.save_pretrained.__get__(model)
    return model


NanoDiffModelMixin._mixin_model = staticmethod(_mixin_model)
