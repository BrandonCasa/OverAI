param(
    [switch]$Train,
    [switch]$CompileVision,
    [string]$Resume = "",
    [int]$Epochs = 20
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $repoRoot ".venv\Scripts\overai-train.exe"
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$modelConfig = Join-Path $repoRoot "configs\rtx4080_720p.json"
$trainManifest = "C:\Users\brand\Documents\overai\Apex\train\train.json"
$validationManifest = "C:\Users\brand\Documents\overai\Apex\validation\validation.json"
$output = Join-Path $repoRoot "runs\apex-4080"

foreach ($requiredPath in @($trainer, $python, $modelConfig, $trainManifest, $validationManifest)) {
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
properties = torch.cuda.get_device_properties(0)
print(f"Training GPU ready: {name} ({properties.total_memory / 2**30:.1f} GiB)")
'@

Push-Location $repoRoot
try {
    & $trainer --manifest $trainManifest --model-config $modelConfig --history-seconds 5 --optimization-seconds 2 --stride-seconds 2 --validate-only
    if ($LASTEXITCODE -ne 0) { throw "Training dataset validation failed." }
    & $trainer --manifest $validationManifest --model-config $modelConfig --history-seconds 5 --optimization-seconds 2 --stride-seconds 2 --validate-only
    if ($LASTEXITCODE -ne 0) { throw "Validation dataset validation failed." }

    if (-not $Train) {
        Write-Output "Readiness checks passed. Re-run with -Train to start training."
        return
    }

    $arguments = @(
        "--manifest", $trainManifest,
        "--validation-manifest", $validationManifest,
        "--model-config", $modelConfig,
        "--output", $output,
        "--batch-size", "2",
        "--epochs", "$Epochs",
        "--history-seconds", "5",
        "--optimization-seconds", "2",
        "--stride-seconds", "2",
        "--tbptt-seconds", "0.1",
        "--num-workers", "4",
        "--checkpoint-every-steps", "120",
        "--bf16"
    )
    if ($CompileVision) { $arguments += "--compile-vision" }
    if ($Resume) { $arguments += @("--resume", $Resume) }
    & $trainer @arguments
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
