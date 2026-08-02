$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it from https://docs.astral.sh/uv/ first."
}

$cudaRoot = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "^v13\.(?:[2-9]|[1-9]\d)" -and (Test-Path (Join-Path $_.FullName "bin\nvcc.exe")) } |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $cudaRoot) {
    throw "CUDA Toolkit 13.2 or newer 13.x with nvcc is required."
}
$env:PATH = "$(Join-Path $cudaRoot.FullName 'bin');$env:PATH"
$cudaVersion = (& (Join-Path $cudaRoot.FullName "bin\nvcc.exe") --version) -join "`n"
if ($cudaVersion -notmatch "release 13\.(?:[2-9]|[1-9]\d)") {
    throw "CUDA Toolkit 13.2 or newer 13.x is required; nvcc reported a different release."
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "Visual Studio C++ Build Tools are required. Install the Desktop development with C++ workload."
}
$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vsDevCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
if (-not $vsInstall -or -not (Test-Path $vsDevCmd)) {
    throw "Visual Studio C++ Build Tools with the Desktop development with C++ workload are required."
}
cmd.exe /c "`"$vsDevCmd`" -no_logo -arch=x64 && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
    }
}

$toolPaths = @(
    "C:\Program Files\CMake\bin",
    (Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja")
)
foreach ($toolPath in $toolPaths) {
    if (Test-Path $toolPath) {
        $env:PATH = "$toolPath;$env:PATH"
    }
}
foreach ($tool in @("cmake", "ninja", "cl")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is required but was not found after toolchain setup." }
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
