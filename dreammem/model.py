from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig
from .encoding import VOCAB_SIZE


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = time.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class MaskedDiffusionTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.time_embedding = SinusoidalTimeEmbedding(config.d_model)
        self.time_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.SiLU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, mask_ratio: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("Expected tokens with shape [batch, sequence]")
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("Sequence exceeds configured maximum length")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        hidden = hidden + self.time_projection(self.time_embedding(mask_ratio))[:, None]
        hidden = self.encoder(hidden)
        hidden = self.final_norm(hidden)
        return torch.nn.functional.linear(
            hidden,
            self.token_embedding.weight,
            self.output_bias,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
