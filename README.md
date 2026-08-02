# OverAI

OverAI is a working PyTorch implementation of a hierarchical, streaming
imitation-learning controller for classic games. It learns only from recorded
human demonstrations: there is no reinforcement learning, reward model,
Q-learning, or online exploration.

The default model is the uploaded specification:

- 1920×1080 RGB input split into 40×40 patches, preserving a 48×27 grid
- shifted 9×8 local-window vision attention instead of quadratic global attention
- causal memory covering roughly 30 seconds at recent, intermediate, and long rates
- two-dimensional discrete movement and six-button state control at 5 Hz
- two bounded continuous axes at 60 Hz
- parallel 2-second trajectories (10 discrete states and 120 axis samples)
- 112,269,820 trainable parameters

## What is implemented

The repository includes the model, streaming state, memory promotion logic,
rate-derived runtime scheduler, direct and horizon imitation losses, safe dataset
validation, incremental JPEG/PNG decoding, truncated backpropagation through time,
bfloat16 H100 training, atomic checkpoints and resume, a real-time game-adapter
boundary, a latency benchmark, and deterministic end-to-end tests.

Game integration is deliberately an adapter rather than hard-coded DOOM logic.
Each game exposes its screen, available telemetry, and input mapping differently.
Implement `overai.runtime.GameAdapter` for the target executable, then use
`run_realtime`. Keep training and deployment mappings identical. Use this only in
offline or private-LAN games where automation is permitted.

## Install and verify

```powershell
uv sync
uv run python -m unittest discover -v
uv run overai-benchmark --model-config configs/h100_1080p.json
```

The benchmark reports each scheduled path against its 30 Hz and 60 Hz deadline.
Run it on the deployment H100; passing on another GPU does not prove H100 timing.

For a quick end-to-end plumbing test, create a tiny synthetic episode and train
one batch:

```powershell
uv run overai-synthetic --output data/synthetic --seconds 4
uv run overai-train `
  --manifest data/synthetic/manifest.json `
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
  "version": 2,
  "episodes": [
    {
      "id": "doom-e1m1-run-001",
      "frames": "episodes/doom-e1m1-run-001/frames",
      "controls": "episodes/doom-e1m1-run-001/controls.pt"
    }
  ]
}
```

Frame files are zero-padded JPEG or PNG images sampled at exactly 30 Hz. The
`controls.pt` file is a tensor dictionary:

| Key | Shape | Rate | Meaning |
| --- | --- | --- | --- |
| `fast_timestamps` | `[T_fast]` | 60 Hz | Monotonic seconds |
| `frame_timestamps` | `[T_video]` | 30 Hz | Frame capture time |
| `slow_timestamps` | `[T_slow]` | 5 Hz | Discrete-control time |
| `health` | `[T_slow, 1]` | 5 Hz | Normalized context |
| `damage_events` | `[T_slow, 1]` | 5 Hz | Binary event |
| `kill_events` | `[T_slow, 1]` | 5 Hz | Binary event |
| `charge` | `[T_slow, 1]` | 5 Hz | Normalized context |
| `axes` | `[T_fast, 2]` | 60 Hz | Executed values in `[-1, 1]` |
| `movement` | `[T_slow, 2]` | 5 Hz | X: `LEFT=0, NONE=1, RIGHT=2`; Y: `REVERSE=0, NONE=1, FORWARD=2` |
| `buttons` | `[T_slow, 6]` | 5 Hz | Binary held states |

If a game cannot expose health, damage, kills, or charge, record zeros for that
channel and also return zeros from the deployment adapter. Context is sampled at
5 Hz and held across the intervening fast ticks. Do not infer future telemetry
into an earlier timestamp. Record executed inputs, not merely requested inputs,
so the action-history encoder sees what the game actually received.

Before committing to a long H100 run:

```powershell
uv run overai-train `
  --manifest D:\datasets\doom\train.json `
  --model-config configs/h100_1080p.json `
  --validate-only
```

## H100 training

The default H100 command uses 30 seconds of causal warm-up and optimizes two
seconds at a time. Warm-up is excluded from the gradient tape; the 30-second
memory still conditions every optimized prediction. The optimization span is
split into 0.2-second truncated-backprop chunks to bound activation memory.

```powershell
uv run overai-train `
  --manifest D:\datasets\doom\train.json `
  --model-config configs/h100_1080p.json `
  --output runs/doom-h100 `
  --batch-size 1 `
  --epochs 20 `
  --history-seconds 30 `
  --optimization-seconds 2 `
  --stride-seconds 2 `
  --tbptt-seconds 0.2 `
  --bf16
```

Optional `--compile-vision` can improve steady-state throughput, but first-run
compilation is slow. Resume without changing the model configuration:

```powershell
uv run overai-train `
  --manifest D:\datasets\doom\train.json `
  --model-config configs/h100_1080p.json `
  --output runs/doom-h100 `
  --resume runs/doom-h100/checkpoint_last.pt
```

Use a separate episode manifest for validation. Accuracy should include discrete
class accuracy/F1, axis Huber error, derivative error, and held-out closed-loop
gameplay scenarios. A decreasing imitation loss alone does not prove the agent
can recover from its own mistakes.

## Deployment boundary

Subclass `GameAdapter` with five operations:

1. capture a 1080p RGB frame;
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
model = load_controller_checkpoint("runs/doom-h100/checkpoint_last.pt", device)
run_realtime(model, DoomAdapter(), device=device)
```

The runtime returns new axes every 60 Hz tick, changes discrete controls only at
5 Hz, captures vision only at 30 Hz, and derives press/release transitions from
held states. Validate input focus, emergency release, pause behavior, capture
latency, and a complete private-LAN match before presenting the project as a
working game agent.
