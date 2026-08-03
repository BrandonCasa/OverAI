param(
    [switch]$Distill,
    [switch]$CompileVision,
    [string]$TeacherCheckpoint = "",
    [string]$Resume = "",
    [int]$Epochs = 10
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$distiller = Join-Path $repoRoot ".venv\Scripts\overai-distill.exe"
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$studentConfig = Join-Path $repoRoot "configs\rtx4080_720p_new.json"
$trainManifest = "C:\Users\brand\Documents\overai\Apex\train\train.json"
$validationManifest = "C:\Users\brand\Documents\overai\Apex\validation\validation.json"
$output = Join-Path $repoRoot "runs\apex-4080-distilled"
if (-not $TeacherCheckpoint) {
    $TeacherCheckpoint = Join-Path $repoRoot "runs\apex-4080\checkpoint_last.pt"
}

foreach ($requiredPath in @(
    $distiller,
    $python,
    $studentConfig,
    $trainManifest,
    $validationManifest,
    $TeacherCheckpoint
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

& $python -c @'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
name = torch.cuda.get_device_name(0)
assert name == "NVIDIA GeForce RTX 4080", f"Expected NVIDIA GeForce RTX 4080, found {name}"
assert torch.cuda.is_bf16_supported(), "The selected GPU/PyTorch build does not support BF16"
print(f"Distillation GPU ready: {name}")
'@

$arguments = @(
    "--teacher-checkpoint", $TeacherCheckpoint,
    "--student-config", $studentConfig,
    "--manifest", $trainManifest,
    "--validation-manifest", $validationManifest,
    "--output", $output,
    "--batch-size", "6",
    "--epochs", "$Epochs",
    "--history-seconds", "5",
    "--optimization-seconds", "2",
    "--stride-seconds", "2",
    "--tbptt-seconds", "0.1",
    "--num-workers", "2",
    "--checkpoint-every-steps", "120",
    "--bf16"
)
if ($CompileVision) { $arguments += "--compile-vision" }
if ($Resume) { $arguments += @("--resume", $Resume) }
if (-not $Distill) { $arguments += "--validate-only" }

Push-Location $repoRoot
try {
    & $distiller @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Distillation command failed with exit code $LASTEXITCODE."
    }
    if (-not $Distill) {
        Write-Output "Readiness checks passed. Re-run with -Distill to start distillation."
    }
}
finally {
    Pop-Location
}
