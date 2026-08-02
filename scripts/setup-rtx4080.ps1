$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it from https://docs.astral.sh/uv/ first."
}

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    throw "CUDA Toolkit 13.2 with nvcc on PATH is required. Install NVIDIA's pinned 13.2 Windows toolkit, then rerun this script."
}
$cudaVersion = & $nvcc.Source --version
if ($cudaVersion -notmatch "release 13\.2") {
    throw "CUDA Toolkit 13.2 is required; nvcc reported a different release."
}

if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    throw "Visual Studio 2022 C++ Build Tools are required. Install the Desktop development with C++ workload, then run this from its Developer PowerShell."
}

foreach ($tool in @("cmake", "ninja")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is required and must be on PATH."
    }
}

$env:UV_PROJECT_ENVIRONMENT = ".venv-rtx"
uv sync --python 3.13 --group recording --group rtx --group dev

& .venv-rtx\Scripts\python.exe -c @'
import torch
import tensorrt_rtx as trt

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4080"
assert torch.cuda.get_device_capability(0) == (8, 9)
print("RTX environment ready:", torch.__version__, trt.__version__)
'@
