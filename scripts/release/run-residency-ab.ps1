param(
    [Parameter(Mandatory = $true)][string]$QwenSnapshot,
    [Parameter(Mandatory = $true)][string]$FluxSnapshot,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [int]$MinimumReductionMiB = 512
)

$ErrorActionPreference = "Stop"
$output = New-Item -ItemType Directory -Force -Path $OutputDirectory
$env:QWEN_IMAGE_SNAPSHOT = (Resolve-Path $QwenSnapshot).Path
$env:FLUX_DEV_DIR = (Resolve-Path $FluxSnapshot).Path
$env:CANDLE_GEN_OFFLOAD = $null

function Invoke-Probe {
    param(
        [string]$Package,
        [string]$Test,
        [string]$ModeVariable,
        [string]$Mode,
        [string]$OutputVariable,
        [string]$RgbName,
        [string]$LogName
    )

    if ($Mode) {
        Set-Item "env:$ModeVariable" $Mode
    } else {
        Remove-Item "env:$ModeVariable" -ErrorAction SilentlyContinue
    }
    Set-Item "env:$OutputVariable" (Join-Path $output $RgbName)
    & cargo test --locked -p $Package --features cuda $Test -- --ignored --nocapture *>&1 |
        Tee-Object -FilePath (Join-Path $output $LogName)
    if ($LASTEXITCODE -ne 0) {
        throw "$Package $Mode probe failed with exit code $LASTEXITCODE"
    }
}

Invoke-Probe candle-gen-qwen-image qwen_image_probed_generate_for_offload_ab `
    QWEN_OFFLOAD_MODE "" QWEN_OUT qwen-resident.rgb qwen-resident.log
Invoke-Probe candle-gen-qwen-image qwen_image_probed_generate_for_offload_ab `
    QWEN_OFFLOAD_MODE spec-sequential QWEN_OUT qwen-sequential.rgb qwen-sequential.log

Invoke-Probe candle-gen-flux flux_dev_probed_generate_for_offload_ab `
    FLUX_OFFLOAD_MODE "" FLUX_OUT flux-dev-resident.rgb flux-dev-resident.log
Invoke-Probe candle-gen-flux flux_dev_probed_generate_for_offload_ab `
    FLUX_OFFLOAD_MODE spec-sequential FLUX_OUT flux-dev-sequential.rgb flux-dev-sequential.log

$qwenResident = Join-Path $output qwen-resident.rgb
$qwenSequential = Join-Path $output qwen-sequential.rgb
$fluxResident = Join-Path $output flux-dev-resident.rgb
$fluxSequential = Join-Path $output flux-dev-sequential.rgb

& fc.exe /b $qwenResident $qwenSequential
if ($LASTEXITCODE -ne 0) { throw "Qwen resident and sequential output differ" }
& fc.exe /b $fluxResident $fluxSequential
if ($LASTEXITCODE -ne 0) { throw "FLUX resident and sequential output differ" }

python scripts/release/verify_residency_ab.py --model qwen-image `
    --resident (Join-Path $output qwen-resident.log) `
    --sequential (Join-Path $output qwen-sequential.log) `
    --min-reduction-mib $MinimumReductionMiB
if ($LASTEXITCODE -ne 0) { throw "Qwen VRAM comparison failed" }

python scripts/release/verify_residency_ab.py --model flux1_dev `
    --resident (Join-Path $output flux-dev-resident.log) `
    --sequential (Join-Path $output flux-dev-sequential.log) `
    --min-reduction-mib $MinimumReductionMiB
if ($LASTEXITCODE -ne 0) { throw "FLUX VRAM comparison failed" }

Get-FileHash $qwenResident, $qwenSequential, $fluxResident, $fluxSequential |
    Format-Table -AutoSize |
    Out-File (Join-Path $output checksums.sha256)
