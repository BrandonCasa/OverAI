# OverAI

OverAI is a working PyTorch implementation of a hierarchical, streaming
imitation-learning controller for classic games. It learns only from recorded
human demonstrations: there is no reinforcement learning, reward model,
Q-learning, or online exploration.

The default model is the uploaded specification:

- 1280×720 planar red/blue input split into 40×40 patches, preserving a 32×18 grid; green is never reconstructed
- four shifted-window vision blocks with 6×8 local attention instead of quadratic global attention
- causal memory covering roughly 30 seconds at recent, intermediate, and long rates
- two-dimensional discrete movement and configurable button-state control at 5 Hz
- two bounded continuous axes at 60 Hz
- parallel 2-second trajectories (10 discrete states and 120 axis samples)
- fixed-shape CUDA BF16 training and RTX 4080-only FP16 TensorRT-RTX deployment

## What is implemented

The repository includes the two-channel model, streaming state, memory promotion logic,
rate-derived runtime scheduler, direct and horizon imitation losses, safe dataset
validation, incremental JPEG/PNG decoding, truncated backpropagation through time,
bfloat16 CUDA training, format-4 atomic checkpoints, Windows Graphics Capture and
Raw Input recording, dataset finalization with training-only axis calibration,
fixed-shape ONNX/TensorRT-RTX export, a 60 Hz RTX scheduler, SendInput control,
latency gates, and deterministic end-to-end tests.

Game integration is deliberately an adapter rather than hard-coded DOOM logic.
Each game exposes its screen and input mapping differently.
Implement `overai.runtime.GameAdapter` for the target executable, then use
`run_realtime`. Keep training and deployment mappings identical. Use this only in
offline or private-LAN games where automation is permitted.

## Install and verify

```powershell
uv sync
uv run python -m unittest discover -v
uv run overai-benchmark --model-config configs/rtx4080_720p.json
```

The checked-in `.python-version` pins Python 3.13 because the TensorRT-RTX wheel
does not support Python 3.14. After installing CUDA Toolkit 13.2 or newer 13.x, Visual Studio C++ Build
Tools, CMake/Ninja, TensorRT-RTX prerequisites, and Nsight Systems, provision the
isolated environment with `scripts/setup-rtx4080.ps1`. The script locates those tools
itself and refuses a non-RTX-4080 GPU.

For a quick end-to-end plumbing test, create a tiny synthetic episode and train
one batch:

```powershell
uv run overai-synthetic --output data/synthetic --seconds 4
uv run overai-synthetic --output data/synthetic-validation --seconds 4 `
  --split validation --episode-id synthetic-validation-001
uv run overai-train `
  --manifest data/synthetic/manifest.json `
  --validation-manifest data/synthetic-validation/manifest.json `
  --tiny `
  --history-seconds 0 `
  --optimization-seconds 1 `
  --stride-seconds 1 `
  --epochs 1 `
  --max-batches 1
```

The synthetic data verifies software plumbing only. It cannot establish gameplay
quality.

## Demonstration dataset

A dataset manifest points to independent episodes so train and validation games
can be split by whole episode (never by overlapping windows):

```json
{
  "version": 3,
  "split": "train",
  "channels": ["R", "B"],
  "axis_normalization": {
    "method": "clipped_linear_velocity_p99_5",
    "percentile": 99.5,
    "scale_counts_per_second": [4200.0, 3900.0]
  },
  "episodes": [
    {
      "id": "doom-e1m1-run-001",
      "red_frames": "episodes/doom-e1m1-run-001/frames_r",
      "blue_frames": "episodes/doom-e1m1-run-001/frames_b",
      "controls": "episodes/doom-e1m1-run-001/controls.pt"
    }
  ]
}
```

R and B are paired zero-padded grayscale JPEGs (quality 95) sampled at 30 Hz. The
`controls.pt` file is a tensor dictionary:

| Key | Shape | Rate | Meaning |
| --- | --- | --- | --- |
| `fast_timestamps` | `[T_fast]` | 60 Hz | Monotonic seconds |
| `frame_timestamps` | `[T_video]` | 30 Hz | Frame capture time |
| `slow_timestamps` | `[T_slow]` | 5 Hz | Discrete-control time |
| `raw_mouse_deltas` | `[T_fast, 2]` | 60 Hz | Integer Raw Input relative counts |
| `fast_durations` | `[T_fast]` | 60 Hz | Exact bin duration in seconds |
| `axes` | `[T_fast, 2]` | 60 Hz | Executed values in `[-1, 1]` |
| `movement` | `[T_slow, 2]` | 5 Hz | X: `LEFT=0, NONE=1, RIGHT=2`; Y: `REVERSE=0, NONE=1, FORWARD=2` |
| `buttons` | `[T_slow, num_buttons]` | 5 Hz | Binary held states |

Record executed inputs, not merely requested inputs, so the action-history
encoder sees what the game actually received.

Before committing to a long GPU run, validate each split independently:

```powershell
uv run overai-train `
  --manifest D:\datasets\doom\train.json `
  --model-config configs/rtx4080_720p.json `
  --validate-only
uv run overai-train `
  --manifest D:\datasets\doom\validation.json `
  --model-config configs/rtx4080_720p.json `
  --validate-only
```

A recorder can close after its last 60 Hz control sample but before the next
30 Hz frame or 5 Hz discrete sample. Dataset validation safely excludes only
that incomplete terminal fraction; it still rejects a missing complete interval
and reports the excluded count as `discarded_terminal_fast_ticks`.

## Windows recording and dataset finalization

Copy `configs/windows_control_profile.example.json`, then set the required
process, title regex, movement keys, the same number of buttons as `num_buttons`
in the model config, pause/emergency keys, and axis inversion.

```powershell
.venv\Scripts\overai-record.exe --profile configs\my-profile.json --split train --episode-id run-001 --output D:\overai\train
.venv\Scripts\overai-record.exe --profile configs\my-profile.json --split validation --episode-id val-001 --output D:\overai\validation
.venv\Scripts\overai-finalize-dataset.exe --train D:\overai\train --validation D:\overai\validation
```

Focus loss, pause, emergency stop, persistent timing gaps, or capture closure ends
a segment. Compatible source-resolution changes are normalized to the configured
model size. A brief missed capture callback reuses the last valid frame for at most
0.25 seconds; `reused_video_frames` records the exact count in `episode.json`.
JPEG encoding runs off the capture loop. Only validated episodes are added
atomically to manifests. Axis scales are the 99.5th percentile of nonzero absolute
training counts/second and are frozen for validation, training, and deployment.

## Local RTX 4080 training

The current RTX 4080 teacher profile uses 720p input with 576 visual tokens, six
512-wide vision blocks, a 640-wide shared/controller state, three fusion blocks,
two decoder blocks, and 121.0M parameters. The distillation target in
`configs/rtx4080_720p_new.json` preserves the same input, memory schedule, control
targets, rates, and two-second horizons while reducing the internals to four
256-wide vision blocks, a 320-wide state, two fusion blocks, one decoder block,
and 24.3M parameters. Its 6×8 windows keep visual attention local and its 12 shared
queries correspond to the 2-axis movement, 8-button, and 2-axis continuous
interface. The H100 profile matches the smaller learned architecture but disables
gradient checkpointing to favor throughput on the larger-memory GPU.
The Overwatch command uses
5 seconds of causal warm-up while optimizing two seconds at a time so the
current episode lengths contribute useful windows.
Warm-up is excluded from the gradient tape; the hierarchical memory still
conditions every optimized prediction. The optimization span is
split into 0.1-second truncated-backprop chunks to fit the 16 GB card with
batch size 1. BF16, gradient checkpointing, fused AdamW, TF32, and flash SDPA are
enabled where supported.

For the local Overwatch dataset, run the non-training readiness check first:

```powershell
.\scripts\train-overwatch-rtx4080.ps1
```

That command checks the exact GPU, BF16 support, model configuration, and both
manifests. It will not start training. Start the full run explicitly with:

```powershell
.\scripts\train-overwatch-rtx4080.ps1 -Train
```

The equivalent direct command is:

```powershell
uv run overai-train `
  --manifest C:\Users\brand\Documents\overai\Overwatch\train\train.json `
  --validation-manifest C:\Users\brand\Documents\overai\Overwatch\validation\validation.json `
  --model-config configs/rtx4080_720p.json `
  --output runs/overwatch-4080 `
  --batch-size 1 `
  --epochs 20 `
  --history-seconds 5 `
  --optimization-seconds 2 `
  --stride-seconds 2 `
  --tbptt-seconds 0.1 `
  --num-workers 2 `
  --checkpoint-every-steps 120 `
  --bf16
```

Optional `--compile-vision` can improve steady-state throughput, but first-run
compilation is slow. The training compiler keeps TorchInductor enabled while
disabling CUDA Graph capture: each TBPTT chunk retains several vision outputs until
one shared backward pass, so those outputs cannot safely be recycled as separate
CUDA Graph iterations. Resume with the helper so all memory-related settings remain
unchanged:

```powershell
.\scripts\train-overwatch-rtx4080.ps1 -Train -Resume runs\overwatch-4080\checkpoint_last.pt
```

Resume must keep the same model and dataset calibration. Add `-CompileVision`
only after the first uncompiled run is stable; compilation increases startup
time and can consume extra memory.

Use a separate episode manifest for validation. After every completed epoch, the
trainer evaluates only that held-out manifest with gradients disabled. It writes
training and validation metrics to `metrics.jsonl`, always updates
`checkpoint_last.pt`, and updates `checkpoint_best.pt` only when validation loss
improves. Reported metrics include discrete accuracy/F1, axis Huber and derivative
losses, and immediate-axis MAE. Held-out closed-loop gameplay scenarios remain a
separate final quality gate: decreasing imitation loss alone does not prove the
agent can recover from its own mistakes.

## Distill the current RTX 4080 checkpoint

Knowledge distillation trains a fresh, smaller student from the frozen outputs of
the existing model while keeping the original demonstration targets as a 30% hard-
label anchor. It does not overwrite or mutate the teacher checkpoint. The teacher
and student may use different internal widths and layer counts, but their image,
control, rate, and horizon contracts must match.

The Apex helper defaults to
`runs\apex-4080\checkpoint_last.pt` as the teacher and
`configs\rtx4080_720p_new.json` as the student. First run its read-only readiness
check; this validates both manifests, the format-4 checkpoint, frozen axis
calibration, control profile, and teacher/student compatibility:

```powershell
.\scripts\distill-apex-rtx4080.ps1
```

Start the full distillation explicitly:

```powershell
.\scripts\distill-apex-rtx4080.ps1 -Distill
```

The equivalent direct command is:

```powershell
.venv\Scripts\overai-distill.exe `
  --teacher-checkpoint runs\apex-4080\checkpoint_last.pt `
  --student-config configs\rtx4080_720p_new.json `
  --manifest C:\Users\brand\Documents\overai\Apex\train\train.json `
  --validation-manifest C:\Users\brand\Documents\overai\Apex\validation\validation.json `
  --output runs\apex-4080-distilled `
  --batch-size 1 `
  --epochs 10 `
  --history-seconds 5 `
  --optimization-seconds 2 `
  --stride-seconds 2 `
  --tbptt-seconds 0.1 `
  --num-workers 2 `
  --checkpoint-every-steps 120 `
  --bf16
```

The student run writes `checkpoint_last.pt`, validation-selected
`checkpoint_best.pt`, `metrics.jsonl`, and model/training/distillation config files
under `runs\apex-4080-distilled`. Checkpoints embed the teacher SHA-256 and
distillation settings. Resume an interrupted student run without changing those
settings or the teacher:

```powershell
.\scripts\distill-apex-rtx4080.ps1 -Distill `
  -Resume runs\apex-4080-distilled\checkpoint_last.pt
```

Use `-TeacherCheckpoint` to select `checkpoint_best.pt` or another run explicitly.
`-CompileVision` compiles only the student vision path and must be supplied again
when resuming. Distillation still needs the original train and held-out validation
episodes because the teacher supplies soft targets on those recorded states; the
validation metrics remain label-based. Export and benchmark
`runs\apex-4080-distilled\checkpoint_best.pt` through the normal RTX workflow
before treating the smaller model as a replacement.

## RTX 4080 deployment

Export on the deployment RTX 4080. This produces five fixed-shape graphs and
strongly typed FP16 TensorRT-RTX engines: ordinary video, intermediate promotion,
long promotion, between-frame fast control, and phase-correct slow control.

```powershell
.venv\Scripts\overai-export-rtx.exe --checkpoint runs\overwatch-4080\checkpoint_last.pt --output artifacts\overwatch-4080
.venv\Scripts\overai-benchmark-rtx.exe --artifact artifacts\overwatch-4080 --profile configs\my-profile.json --duration 600
.venv\Scripts\overai-run-rtx.exe --artifact artifacts\overwatch-4080 --profile configs\my-profile.json
```

Exports keep LayerNorm and attention SDPA/softmax in FP16 by default. If trained-
checkpoint parity identifies either family as precision-sensitive, rebuild with
`--fp32-layernorm`, `--fp32-attention`, or both; the overrides are recorded in the
artifact manifest.

The artifact embeds the checkpoint hash, model/training configuration, R/B channel
order, frozen axis scales, TensorRT-RTX version, and the exact RTX 4080 SM89 gate.
Inference uses persistent state and frame buffers plus TensorRT-RTX whole-graph CUDA
capture. Focus loss, pause, or emergency stop releases every injected held control.
PyTorch runtime remains a numerical reference and is not the supported production
backend.

The checked-in `windows_capture` backend is a functional Windows WGC/Raw Input
fallback for recorder and scheduler integration tests, but it performs BGRA-to-R/B
preprocessing through PyTorch CPU tensors. It is not the final zero-copy
D3D11-CUDA production capture backend and cannot satisfy the no-per-tick-allocation
acceptance gate. Building that native extension requires the separately downloaded
TensorRT-RTX Windows SDK (the PyPI wheel does not include its C++ headers), CUDA
Toolkit 13.2, and Visual Studio C++ Build Tools.

## PyTorch reference boundary

Subclass `GameAdapter` with five operations:

1. capture a 720p planar R/B frame;
2. expose the four causal context values;
3. report the previously executed controls;
4. apply two continuous axes;
5. apply a two-component discrete movement vector and held button states.

Load and run a checkpoint with:

```python
import torch

from overai.runtime import load_controller_checkpoint, run_realtime
from my_doom_adapter import DoomAdapter

device = torch.device("cuda")
model = load_controller_checkpoint("runs/overwatch-4080/checkpoint_last.pt", device)
run_realtime(model, DoomAdapter(), device=device)
```

The runtime returns new axes every 60 Hz tick, changes discrete controls only at
5 Hz, captures vision only at 30 Hz, and derives press/release transitions from
held states. Validate input focus, emergency release, pause behavior, capture
latency, and a complete private-LAN match before presenting the project as a
working game agent.
