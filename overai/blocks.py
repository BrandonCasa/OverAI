"""Attention, compression, and embedding building blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth - 1):
            layers.extend(
                (
                    nn.Linear(current_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_dim),
                )
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class FourierTimeEmbedding(nn.Module):
    def __init__(
        self,
        number_of_time_values: int,
        model_dim: int,
        frequencies: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        frequency_values = 2.0 ** torch.arange(frequencies, dtype=torch.float32)
        self.frequency_values: torch.Tensor
        self.register_buffer("frequency_values", frequency_values, persistent=False)
        encoded_dim = number_of_time_values * frequencies * 2
        self.projection = MLP(encoded_dim, model_dim, model_dim, dropout=dropout)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        frequencies = self.frequency_values.to(dtype=timestamps.dtype)
        angles = timestamps.unsqueeze(-1) * frequencies
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return self.projection(encoded.flatten(start_dim=-2))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = MLP(dim, dim * 4, dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        attention_output, _ = self.attention(
            query=self.query_norm(queries),
            key=normalized_memory,
            value=normalized_memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        queries = queries + self.dropout(attention_output)
        return queries + self.dropout(self.ffn(self.ffn_norm(queries)))


class QueryCompressor(nn.Module):
    def __init__(
        self,
        dim: int,
        num_output_tokens: int,
        num_heads: int,
        layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_queries = nn.Parameter(
            torch.randn(1, num_output_tokens, dim) * 0.02
        )
        self.layers = nn.ModuleList(
            CrossAttentionBlock(dim, num_heads, dropout) for _ in range(layers)
        )

    def forward(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        queries = self.output_queries.expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, tokens, padding_mask)
        return queries


def _partition_windows(
    grid: torch.Tensor,
    window_height: int,
    window_width: int,
    shifted: bool,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    """Partition BHWD data into non-wrapping (optionally shifted) windows."""

    batch, height, width, dim = grid.shape
    top = window_height // 2 if shifted else 0
    left = window_width // 2 if shifted else 0
    bottom = (window_height - (height + top) % window_height) % window_height
    right = (window_width - (width + left) % window_width) % window_width

    padded = F.pad(grid, (0, 0, left, right, top, bottom))
    valid = torch.ones((batch, height, width, 1), dtype=torch.bool, device=grid.device)
    valid = F.pad(valid, (0, 0, left, right, top, bottom), value=False)
    padded_height, padded_width = padded.shape[1:3]

    windows = (
        padded.view(
            batch,
            padded_height // window_height,
            window_height,
            padded_width // window_width,
            window_width,
            dim,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_height * window_width, dim)
    )
    valid_windows = (
        valid.view(
            batch,
            padded_height // window_height,
            window_height,
            padded_width // window_width,
            window_width,
            1,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_height * window_width)
    )
    return windows, valid_windows, (top, left, padded_height, padded_width)


def _reverse_windows(
    windows: torch.Tensor,
    batch: int,
    height: int,
    width: int,
    window_height: int,
    window_width: int,
    padding: tuple[int, int, int, int],
) -> torch.Tensor:
    top, left, padded_height, padded_width = padding
    dim = windows.shape[-1]
    grid = (
        windows.view(
            batch,
            padded_height // window_height,
            padded_width // window_width,
            window_height,
            window_width,
            dim,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, padded_height, padded_width, dim)
    )
    return grid[:, top : top + height, left : left + width]


class SpatialVisionBlock(nn.Module):
    """Shifted local-window vision block with no boundary wraparound."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_height: int,
        window_width: int,
        shifted: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_height = window_height
        self.window_width = window_width
        self.shifted = shifted
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, dim * 4, dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = grid.shape
        residual_windows, valid, padding = _partition_windows(
            grid,
            self.window_height,
            self.window_width,
            self.shifted,
        )
        normalized = self.norm1(residual_windows)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        windows = residual_windows + self.dropout(attended)
        windows = windows + self.dropout(self.ffn(self.norm2(windows)))
        return _reverse_windows(
            windows,
            batch,
            height,
            width,
            self.window_height,
            self.window_width,
            padding,
        )
