"""Attention, compression, and embedding building blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ExportableLayerNorm(nn.LayerNorm):
    """LayerNorm that can retain FP32 accumulation inside an FP16 RTX graph."""

    export_fp32 = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.export_fp32:
            return super().forward(input)
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        normalized = F.layer_norm(
            input.float(), self.normalized_shape, weight, bias, self.eps
        )
        return normalized.to(dtype=input.dtype)


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
                    ExportableLayerNorm(hidden_dim),
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


class ExportableAttention(nn.Module):
    """Explicit multi-head attention that exports cleanly through ONNX.

    Separate projections support both self- and cross-attention while allowing
    PyTorch SDPA, TensorRT fused attention, and H100 flash kernels to select the
    best implementation for the active device.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        self_attention: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dimension must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.self_attention = self_attention
        self.export_attention_fp32 = False
        if self_attention:
            self.qkv_projection = nn.Linear(dim, dim * 3)
            self.query_projection = None
            self.key_value_projection = None
        else:
            self.qkv_projection = None
            self.query_projection = nn.Linear(dim, dim)
            self.key_value_projection = nn.Linear(dim, dim * 2)
        self.output_projection = nn.Linear(dim, dim)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.self_attention:
            assert self.qkv_projection is not None
            projected_query, projected_key, projected_value = self.qkv_projection(
                query
            ).chunk(3, dim=-1)
        else:
            assert self.query_projection is not None
            assert self.key_value_projection is not None
            projected_query = self.query_projection(query)
            projected_key, projected_value = self.key_value_projection(key).chunk(
                2, dim=-1
            )
        queries = self._heads(projected_query)
        keys = self._heads(projected_key)
        values = self._heads(projected_value)
        attention_dtype = queries.dtype
        if self.export_attention_fp32:
            queries = queries.float()
            keys = keys.float()
            values = values.float()
        attention_mask = None
        if key_padding_mask is not None:
            attention_mask = (~key_padding_mask).view(
                key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1]
            )
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.to(dtype=attention_dtype)
        attended = attended.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], self.dim
        )
        return self.output_projection(attended)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = ExportableLayerNorm(dim)
        self.memory_norm = ExportableLayerNorm(dim)
        self.attention = ExportableAttention(dim, num_heads, dropout)
        self.ffn_norm = ExportableLayerNorm(dim)
        self.ffn = MLP(dim, dim * 4, dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        attention_output = self.attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            memory_padding_mask,
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
    valid_template: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    """Partition BHWD data into non-wrapping (optionally shifted) windows."""

    batch, height, width, dim = grid.shape
    top = window_height // 2 if shifted else 0
    left = window_width // 2 if shifted else 0
    bottom = (window_height - (height + top) % window_height) % window_height
    right = (window_width - (width + left) % window_width) % window_width

    padded = F.pad(grid, (0, 0, left, right, top, bottom))
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
    if valid_template is None:
        valid = torch.ones(
            (batch, height, width, 1), dtype=torch.bool, device=grid.device
        )
        valid = F.pad(valid, (0, 0, left, right, top, bottom), value=False)
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
    else:
        valid_windows = (
            valid_template.unsqueeze(0)
            .expand(batch, -1, -1)
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
        grid_height: int,
        grid_width: int,
        shifted: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_height = window_height
        self.window_width = window_width
        self.shifted = shifted
        self.norm1 = ExportableLayerNorm(dim)
        self.attention = ExportableAttention(
            dim, num_heads, dropout, self_attention=True
        )
        self.norm2 = ExportableLayerNorm(dim)
        self.ffn = MLP(dim, dim * 4, dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        _, valid, _ = _partition_windows(
            torch.zeros(1, grid_height, grid_width, 1),
            window_height,
            window_width,
            shifted,
        )
        self.valid_windows: torch.Tensor
        self.register_buffer("valid_windows", valid, persistent=True)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = grid.shape
        residual_windows, valid, padding = _partition_windows(
            grid,
            self.window_height,
            self.window_width,
            self.shifted,
            self.valid_windows,
        )
        normalized = self.norm1(residual_windows)
        attended = self.attention(
            normalized, normalized, normalized, key_padding_mask=~valid
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
