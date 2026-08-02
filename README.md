# OverAI

Minimal `uv` project for NVIDIA CUDA PyTorch workloads.

```powershell
uv sync
uv run python main.py
```

PyTorch packages are resolved from the CUDA 13.0 wheel index. The remaining packages provide common transformer and attention-workflow support; PyTorch's built-in scaled-dot-product and flex attention are used instead of platform-specific compiled extensions.
