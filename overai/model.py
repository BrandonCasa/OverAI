"""Executable hierarchical vision-memory imitation controller."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import (
    MLP,
    CrossAttentionBlock,
    ExportableLayerNorm,
    FourierTimeEmbedding,
    QueryCompressor,
    SpatialVisionBlock,
)
from .config import ModelConfig
from .types import (
    ControllerState,
    DecodedSlowAction,
    ExecutedActions,
    FastControllerState,
    FastPrediction,
    HierarchicalMemoryState,
    ObservationContext,
    ReplanOutput,
    RuntimeStepOutput,
    SlowPrediction,
    TimingContext,
)


def _fixed_2d_position(height: int, width: int, dim: int) -> torch.Tensor:
    if dim % 4:
        raise ValueError("vision_dim must be divisible by four for 2D position encoding")
    quarter = dim // 4
    frequencies = torch.exp(
        torch.arange(quarter, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0)) / max(quarter - 1, 1))
    )
    rows = torch.arange(height, dtype=torch.float32).unsqueeze(1) * frequencies
    columns = torch.arange(width, dtype=torch.float32).unsqueeze(1) * frequencies
    row_encoding = torch.cat((rows.sin(), rows.cos()), dim=1)
    column_encoding = torch.cat((columns.sin(), columns.cos()), dim=1)
    return torch.cat(
        (
            row_encoding[:, None, :].expand(-1, width, -1),
            column_encoding[None, :, :].expand(height, -1, -1),
        ),
        dim=-1,
    ).unsqueeze(0)


class SpatialVisionEncoder(nn.Module):
    """Encode a frame while preserving the configured spatial patch grid."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_projection = nn.Conv2d(
            cfg.input_channels,
            cfg.vision_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.position_encoding: torch.Tensor
        self.register_buffer(
            "position_encoding",
            _fixed_2d_position(cfg.grid_height, cfg.grid_width, cfg.vision_dim),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            SpatialVisionBlock(
                cfg.vision_dim,
                cfg.num_heads,
                cfg.window_height,
                cfg.window_width,
                cfg.grid_height,
                cfg.grid_width,
                shifted=bool(index % 2),
                dropout=cfg.dropout,
            )
            for index in range(cfg.vision_layers)
        )
        self.output_projection = nn.Linear(cfg.vision_dim, cfg.model_dim)
        self.output_norm = ExportableLayerNorm(cfg.model_dim)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 4 or frame.shape[1] != self.cfg.input_channels:
            raise ValueError(
                f"expected BCHW {self.cfg.channel_order} frame, got shape "
                f"{tuple(frame.shape)}"
            )
        if frame.shape[-2:] != (self.cfg.image_height, self.cfg.image_width):
            raise ValueError(
                "frame dimensions do not match config: "
                f"expected {(self.cfg.image_height, self.cfg.image_width)}, "
                f"got {tuple(frame.shape[-2:])}"
            )

        if frame.dtype == torch.uint8:
            frame = frame.to(dtype=self.patch_projection.weight.dtype).div_(127.5).sub_(1.0)
        elif not torch.is_floating_point(frame):
            raise TypeError("frames must be uint8 or floating point")
        else:
            frame = frame.mul(2.0).sub(1.0)

        patches = self.patch_projection(frame)
        grid = patches.permute(0, 2, 3, 1)
        grid = grid + self.position_encoding.to(dtype=grid.dtype)
        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training and grid.requires_grad:
                grid = checkpoint(block, grid, use_reentrant=False)
            else:
                grid = block(grid)
        return self.output_norm(self.output_projection(grid))


class ObservationContextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.encoder = MLP(4, cfg.model_dim, cfg.model_dim, dropout=cfg.dropout)

    def forward(self, context: ObservationContext) -> torch.Tensor:
        return self.encoder(
            torch.cat(
                (
                    context.health,
                    context.damage_event,
                    context.kill_event,
                    context.charge,
                ),
                dim=-1,
            )
        )


class ExecutedActionEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.movement_embedding = nn.Embedding(3, 32)
        self.buttons_encoder = MLP(cfg.num_buttons, 64, 64, dropout=cfg.dropout)
        self.axes_encoder = MLP(2, 64, 64, dropout=cfg.dropout)
        self.output = MLP(192, cfg.action_dim * 2, cfg.action_dim, dropout=cfg.dropout)

    def forward(self, actions: ExecutedActions) -> torch.Tensor:
        movement = self.movement_embedding(actions.movement.long()).flatten(-2)
        dtype = self.movement_embedding.weight.dtype
        buttons = self.buttons_encoder(actions.buttons.to(dtype=dtype))
        axes = self.axes_encoder(actions.axes.to(dtype=dtype))
        return self.output(torch.cat((movement, buttons, axes), dim=-1))


class PreviousTrajectoryEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.axis_value_projection = nn.Linear(2, cfg.model_dim)
        self.slow_value_projection = nn.Linear(6 + cfg.num_buttons, cfg.model_dim)
        self.compressor = QueryCompressor(
            cfg.model_dim,
            cfg.trajectory_summary_tokens,
            cfg.num_heads,
            layers=cfg.compressor_layers,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        previous_axis_trajectory: torch.Tensor,
        previous_slow_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        axis_tokens = self.axis_value_projection(previous_axis_trajectory)
        slow_tokens = self.slow_value_projection(previous_slow_trajectory)
        return self.compressor(torch.cat((axis_tokens, slow_tokens), dim=1))


def append_fifo(
    buffer: torch.Tensor,
    validity: torch.Tensor,
    new_entry: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if new_entry.shape != (buffer.shape[0], buffer.shape[2], buffer.shape[3]):
        raise ValueError(
            f"new FIFO entry has shape {tuple(new_entry.shape)}, expected "
            f"{(buffer.shape[0], buffer.shape[2], buffer.shape[3])}"
        )
    new_valid = torch.ones(
        validity.shape[0], 1, dtype=torch.bool, device=validity.device
    )
    return (
        torch.cat((buffer[:, 1:], new_entry.unsqueeze(1)), dim=1),
        torch.cat((validity[:, 1:], new_valid), dim=1),
    )


class HierarchicalTemporalMemory(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        compressor_args = {
            "dim": cfg.model_dim,
            "num_heads": cfg.num_heads,
            "layers": cfg.compressor_layers,
            "dropout": cfg.dropout,
        }
        self.frame_summary = QueryCompressor(
            num_output_tokens=cfg.recent_tokens_per_entry, **compressor_args
        )
        self.intermediate_compressor = QueryCompressor(
            num_output_tokens=cfg.intermediate_tokens_per_entry, **compressor_args
        )
        self.long_compressor = QueryCompressor(
            num_output_tokens=cfg.long_tokens_per_entry, **compressor_args
        )
        self.recent_age_embedding = nn.Embedding(cfg.recent_entries, cfg.model_dim)
        self.intermediate_age_embedding = nn.Embedding(
            cfg.intermediate_entries, cfg.model_dim
        )
        self.long_age_embedding = nn.Embedding(cfg.long_entries, cfg.model_dim)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> HierarchicalMemoryState:
        cfg = self.cfg

        def zeros(entries: int, tokens: int) -> torch.Tensor:
            return torch.zeros(
                batch_size,
                entries,
                tokens,
                cfg.model_dim,
                device=device,
                dtype=dtype,
            )

        def invalid(entries: int) -> torch.Tensor:
            return torch.zeros(batch_size, entries, dtype=torch.bool, device=device)

        return HierarchicalMemoryState(
            recent=zeros(cfg.recent_entries, cfg.recent_tokens_per_entry),
            intermediate=zeros(
                cfg.intermediate_entries, cfg.intermediate_tokens_per_entry
            ),
            long=zeros(cfg.long_entries, cfg.long_tokens_per_entry),
            recent_valid=invalid(cfg.recent_entries),
            intermediate_valid=invalid(cfg.intermediate_entries),
            long_valid=invalid(cfg.long_entries),
            frame_counter=0,
            intermediate_counter=0,
        )

    def update(
        self,
        current_grid: torch.Tensor,
        frame_metadata_tokens: torch.Tensor,
        state: HierarchicalMemoryState,
    ) -> HierarchicalMemoryState:
        batch_size = current_grid.shape[0]
        spatial_tokens = current_grid.reshape(
            batch_size, self.cfg.grid_tokens, self.cfg.model_dim
        )
        recent_entry = self.frame_summary(
            torch.cat((spatial_tokens, frame_metadata_tokens), dim=1)
        )
        recent, recent_valid = append_fifo(
            state.recent, state.recent_valid, recent_entry
        )
        frame_counter = state.frame_counter + 1
        intermediate = state.intermediate
        intermediate_valid = state.intermediate_valid
        long = state.long
        long_valid = state.long_valid
        intermediate_counter = state.intermediate_counter

        if frame_counter % self.cfg.frames_per_intermediate == 0:
            latest_recent = recent[:, -self.cfg.frames_per_intermediate :].flatten(1, 2)
            intermediate_entry = self.intermediate_compressor(latest_recent)
            intermediate, intermediate_valid = append_fifo(
                intermediate, intermediate_valid, intermediate_entry
            )
            intermediate_counter += 1
            if intermediate_counter % self.cfg.intermediate_per_long == 0:
                latest_intermediate = intermediate[
                    :, -self.cfg.intermediate_per_long :
                ].flatten(1, 2)
                previous_long = long[:, -1]
                long_entry = self.long_compressor(
                    torch.cat((latest_intermediate, previous_long), dim=1)
                )
                long, long_valid = append_fifo(long, long_valid, long_entry)

        return HierarchicalMemoryState(
            recent=recent,
            intermediate=intermediate,
            long=long,
            recent_valid=recent_valid,
            intermediate_valid=intermediate_valid,
            long_valid=long_valid,
            frame_counter=frame_counter,
            intermediate_counter=intermediate_counter,
        )

    def _add_age_embedding(
        self, memory: torch.Tensor, embedding: nn.Embedding
    ) -> torch.Tensor:
        # FIFO order is oldest to newest, so age zero belongs to the final entry.
        ages = torch.arange(memory.shape[1] - 1, -1, -1, device=memory.device)
        age_embedding = embedding(ages).view(1, memory.shape[1], 1, self.cfg.model_dim)
        return memory + age_embedding

    def read_tokens(
        self, state: HierarchicalMemoryState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recent = self._add_age_embedding(
            state.recent, self.recent_age_embedding
        ).flatten(1, 2)
        intermediate = self._add_age_embedding(
            state.intermediate, self.intermediate_age_embedding
        ).flatten(1, 2)
        long = self._add_age_embedding(state.long, self.long_age_embedding).flatten(
            1, 2
        )
        valid = torch.cat(
            (
                state.recent_valid.repeat_interleave(
                    self.cfg.recent_tokens_per_entry, dim=1
                ),
                state.intermediate_valid.repeat_interleave(
                    self.cfg.intermediate_tokens_per_entry, dim=1
                ),
                state.long_valid.repeat_interleave(
                    self.cfg.long_tokens_per_entry, dim=1
                ),
            ),
            dim=1,
        )
        return torch.cat((recent, intermediate, long), dim=1), ~valid


class SharedStateFusion(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.control_queries = nn.Parameter(
            torch.randn(1, cfg.control_query_tokens, cfg.model_dim) * 0.02
        )
        self.current_grid_layers = nn.ModuleList(
            CrossAttentionBlock(cfg.model_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.fusion_layers)
        )
        self.memory_layers = nn.ModuleList(
            CrossAttentionBlock(cfg.model_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.fusion_layers)
        )
        self.metadata_layers = nn.ModuleList(
            CrossAttentionBlock(cfg.model_dim, cfg.num_heads, cfg.dropout)
            for _ in range(2)
        )
        self.output_norm = ExportableLayerNorm(cfg.model_dim)

    def forward(
        self,
        current_grid: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        metadata_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch = current_grid.shape[0]
        grid_tokens = current_grid.reshape(batch, -1, current_grid.shape[-1])
        queries = self.control_queries.expand(batch, -1, -1)
        for layer in self.current_grid_layers:
            queries = layer(queries, grid_tokens)
        for layer in self.memory_layers:
            queries = layer(queries, memory_tokens, memory_padding_mask)
        for layer in self.metadata_layers:
            queries = layer(queries, metadata_tokens)
        return self.output_norm(queries)


class SlowControlDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.immediate_query = nn.Parameter(torch.randn(1, 1, cfg.model_dim) * 0.02)
        self.immediate_attention = CrossAttentionBlock(
            cfg.model_dim, cfg.num_heads, cfg.dropout
        )
        self.immediate_movement_head = nn.Linear(cfg.model_dim, 6)
        self.immediate_button_head = nn.Linear(cfg.model_dim, cfg.num_buttons)
        self.trajectory_queries = nn.Parameter(
            torch.randn(1, cfg.slow_horizon, cfg.model_dim) * 0.02
        )
        self.trajectory_decoder = nn.ModuleList(
            CrossAttentionBlock(cfg.model_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.decoder_layers)
        )
        self.trajectory_movement_head = nn.Linear(cfg.model_dim, 6)
        self.trajectory_button_head = nn.Linear(cfg.model_dim, cfg.num_buttons)

    def forward(self, shared_tokens: torch.Tensor) -> SlowPrediction:
        batch = shared_tokens.shape[0]
        immediate_token = self.immediate_attention(
            self.immediate_query.expand(batch, -1, -1), shared_tokens
        )[:, 0]
        trajectory = self.trajectory_queries.expand(batch, -1, -1)
        for layer in self.trajectory_decoder:
            trajectory = layer(trajectory, shared_tokens)
        return SlowPrediction(
            immediate_movement_logits=self.immediate_movement_head(immediate_token).view(
                batch, 2, 3
            ),
            immediate_button_logits=self.immediate_button_head(immediate_token),
            trajectory_movement_logits=self.trajectory_movement_head(trajectory).view(
                batch, self.trajectory_queries.shape[1], 2, 3
            ),
            trajectory_button_logits=self.trajectory_button_head(trajectory),
        )


class FastAxisDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        compressor_args = {
            "dim": cfg.model_dim,
            "num_output_tokens": 1,
            "num_heads": cfg.num_heads,
            "layers": cfg.compressor_layers,
            "dropout": cfg.dropout,
        }
        self.shared_state_pool = QueryCompressor(**compressor_args)
        self.previous_axis_trajectory_encoder = QueryCompressor(**compressor_args)
        self.axis_trajectory_projection = nn.Linear(2, cfg.model_dim)
        recurrent_input_dim = cfg.model_dim * 3 + cfg.action_dim
        self.recurrent_input = MLP(
            recurrent_input_dim,
            cfg.controller_dim,
            cfg.controller_dim,
            dropout=cfg.dropout,
        )
        self.recurrent_cell = nn.GRUCell(cfg.controller_dim, cfg.controller_dim)
        self.hidden_to_model = nn.Linear(cfg.controller_dim, cfg.model_dim)
        self.immediate_axis1_head = MLP(
            cfg.controller_dim, cfg.controller_dim, 1, dropout=cfg.dropout
        )
        self.immediate_axis2_head = MLP(
            cfg.controller_dim, cfg.controller_dim, 1, dropout=cfg.dropout
        )
        self.trajectory_queries = nn.Parameter(
            torch.randn(1, cfg.fast_horizon, cfg.model_dim) * 0.02
        )
        self.trajectory_decoder = nn.ModuleList(
            CrossAttentionBlock(cfg.model_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.decoder_layers)
        )
        self.axis1_trajectory_head = nn.Linear(cfg.model_dim, 1)
        self.axis2_trajectory_head = nn.Linear(cfg.model_dim, 1)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> FastControllerState:
        return FastControllerState(
            hidden=torch.zeros(
                batch_size, self.cfg.controller_dim, device=device, dtype=dtype
            ),
            previous_trajectory=torch.zeros(
                batch_size,
                self.cfg.fast_horizon,
                2,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(
        self,
        shared_tokens: torch.Tensor,
        executed_action_embedding: torch.Tensor,
        timing_embedding: torch.Tensor,
        state: FastControllerState,
    ) -> FastPrediction:
        batch = shared_tokens.shape[0]
        pooled_shared = self.shared_state_pool(shared_tokens)[:, 0]
        previous_tokens = self.axis_trajectory_projection(state.previous_trajectory)
        previous_summary = self.previous_axis_trajectory_encoder(previous_tokens)[:, 0]
        recurrent_input = self.recurrent_input(
            torch.cat(
                (
                    pooled_shared,
                    previous_summary,
                    executed_action_embedding,
                    timing_embedding,
                ),
                dim=-1,
            )
        )
        next_hidden = self.recurrent_cell(recurrent_input, state.hidden)
        immediate_axes = torch.cat(
            (
                torch.tanh(self.immediate_axis1_head(next_hidden)),
                torch.tanh(self.immediate_axis2_head(next_hidden)),
            ),
            dim=-1,
        )
        decoder_memory = torch.cat(
            (self.hidden_to_model(next_hidden).unsqueeze(1), shared_tokens), dim=1
        )
        trajectory = self.trajectory_queries.expand(batch, -1, -1)
        for layer in self.trajectory_decoder:
            trajectory = layer(trajectory, decoder_memory)
        axis_trajectory = torch.cat(
            (
                torch.tanh(self.axis1_trajectory_head(trajectory)),
                torch.tanh(self.axis2_trajectory_head(trajectory)),
            ),
            dim=-1,
        )
        next_state = FastControllerState(next_hidden, axis_trajectory)
        return FastPrediction(immediate_axes, axis_trajectory, next_state)


class HierarchicalImitationController(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision = SpatialVisionEncoder(cfg)
        self.observation_encoder = ObservationContextEncoder(cfg)
        self.executed_action_encoder = ExecutedActionEncoder(cfg)
        self.time_encoder = FourierTimeEmbedding(4, cfg.model_dim, dropout=cfg.dropout)
        self.trajectory_encoder = PreviousTrajectoryEncoder(cfg)
        self.memory = HierarchicalTemporalMemory(cfg)
        self.fusion = SharedStateFusion(cfg)
        self.slow_decoder = SlowControlDecoder(cfg)
        self.fast_decoder = FastAxisDecoder(cfg)
        self.action_to_model = nn.Linear(cfg.action_dim, cfg.model_dim)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ControllerState:
        return ControllerState(
            memory=self.memory.initial_state(batch_size, device, dtype),
            fast=self.fast_decoder.initial_state(batch_size, device, dtype),
            current_grid=None,
            shared_tokens=None,
            previous_axis_trajectory=torch.zeros(
                batch_size,
                self.cfg.fast_horizon,
                2,
                device=device,
                dtype=dtype,
            ),
            previous_slow_trajectory=torch.zeros(
                batch_size,
                self.cfg.slow_horizon,
                6 + self.cfg.num_buttons,
                device=device,
                dtype=dtype,
            ),
        )

    def build_metadata_tokens(
        self,
        observation_context: ObservationContext,
        executed_actions: ExecutedActions,
        timing: TimingContext,
        state: ControllerState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_embedding = self.observation_encoder(observation_context)
        action_embedding = self.executed_action_encoder(executed_actions)
        timing_embedding = self.time_encoder(
            torch.cat(
                (
                    timing.absolute_time,
                    timing.since_video_frame,
                    timing.since_slow_update,
                    timing.fast_delta_time,
                ),
                dim=-1,
            )
        )
        trajectory_tokens = self.trajectory_encoder(
            state.previous_axis_trajectory, state.previous_slow_trajectory
        )
        scalar_tokens = torch.stack(
            (
                context_embedding,
                self.action_to_model(action_embedding),
                timing_embedding,
            ),
            dim=1,
        )
        return (
            torch.cat((scalar_tokens, trajectory_tokens), dim=1),
            action_embedding,
            timing_embedding,
        )

    @staticmethod
    def _slow_trajectory(prediction: SlowPrediction) -> torch.Tensor:
        return torch.cat(
            (
                prediction.trajectory_movement_logits.flatten(-2),
                prediction.trajectory_button_logits,
            ),
            dim=-1,
        )

    def on_video_frame(
        self,
        frame: torch.Tensor,
        observation_context: ObservationContext,
        executed_actions: ExecutedActions,
        timing: TimingContext,
        state: ControllerState,
        run_slow_decoder: bool,
    ) -> ReplanOutput:
        current_grid = self.vision(frame)
        metadata, action_embedding, timing_embedding = self.build_metadata_tokens(
            observation_context, executed_actions, timing, state
        )
        memory_state = self.memory.update(current_grid, metadata, state.memory)
        memory_tokens, memory_padding_mask = self.memory.read_tokens(memory_state)
        shared_tokens = self.fusion(
            current_grid, memory_tokens, memory_padding_mask, metadata
        )
        fast_prediction = self.fast_decoder(
            shared_tokens, action_embedding, timing_embedding, state.fast
        )
        slow_prediction = self.slow_decoder(shared_tokens) if run_slow_decoder else None
        next_state = replace(
            state,
            memory=memory_state,
            fast=fast_prediction.next_state,
            current_grid=current_grid,
            shared_tokens=shared_tokens,
            previous_axis_trajectory=fast_prediction.axis_trajectory,
            previous_slow_trajectory=(
                self._slow_trajectory(slow_prediction)
                if slow_prediction is not None
                else state.previous_slow_trajectory
            ),
        )
        return ReplanOutput(slow_prediction, fast_prediction, next_state)

    def fast_tick_between_frames(
        self,
        observation_context: ObservationContext,
        executed_actions: ExecutedActions,
        timing: TimingContext,
        state: ControllerState,
    ) -> tuple[FastPrediction, ControllerState]:
        if state.shared_tokens is None:
            raise RuntimeError(
                "a video frame must be processed before a frame-free fast tick"
            )
        _, action_embedding, timing_embedding = self.build_metadata_tokens(
            observation_context, executed_actions, timing, state
        )
        fast_prediction = self.fast_decoder(
            state.shared_tokens, action_embedding, timing_embedding, state.fast
        )
        return fast_prediction, replace(
            state,
            fast=fast_prediction.next_state,
            previous_axis_trajectory=fast_prediction.axis_trajectory,
        )

    def slow_tick(
        self, state: ControllerState
    ) -> tuple[SlowPrediction, ControllerState]:
        if state.shared_tokens is None:
            raise RuntimeError("a video frame must be processed before a slow tick")
        prediction = self.slow_decoder(state.shared_tokens)
        return prediction, replace(
            state, previous_slow_trajectory=self._slow_trajectory(prediction)
        )


def decode_slow_action(prediction: SlowPrediction) -> DecodedSlowAction:
    return DecodedSlowAction(
        movement=prediction.immediate_movement_logits.argmax(dim=-1),
        buttons=torch.sigmoid(prediction.immediate_button_logits) >= 0.5,
    )


def derive_button_events(
    previous_button_states: torch.Tensor, current_button_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    previous = previous_button_states.bool()
    current = current_button_states.bool()
    return ~previous & current, previous & ~current


class RuntimeController:
    """Rate-derived scheduler; the caller applies the returned controls."""

    def __init__(self, model: HierarchicalImitationController) -> None:
        self.model = model
        self.fast_tick_index = 0

    def reset(self) -> None:
        self.fast_tick_index = 0

    def step(
        self,
        optional_new_frame: Optional[torch.Tensor],
        observation_context: ObservationContext,
        executed_actions: ExecutedActions,
        timing: TimingContext,
        state: ControllerState,
    ) -> RuntimeStepOutput:
        slow_due = self.fast_tick_index % self.model.cfg.fast_ticks_per_slow == 0
        if optional_new_frame is not None:
            output = self.model.on_video_frame(
                optional_new_frame,
                observation_context,
                executed_actions,
                timing,
                state,
                run_slow_decoder=slow_due,
            )
            state = output.state
            fast_prediction = output.fast
            slow_prediction = output.slow
        else:
            fast_prediction, state = self.model.fast_tick_between_frames(
                observation_context, executed_actions, timing, state
            )
            slow_prediction = None
            if slow_due:
                slow_prediction, state = self.model.slow_tick(state)

        discrete = (
            decode_slow_action(slow_prediction) if slow_prediction is not None else None
        )
        self.fast_tick_index += 1
        return RuntimeStepOutput(
            axes=fast_prediction.immediate_axes,
            discrete=discrete,
            slow_prediction=slow_prediction,
            fast_prediction=fast_prediction,
            state=state,
        )
