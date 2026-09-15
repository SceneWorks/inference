param(
    [Parameter(Mandatory = $true)][string]$QwenSnapshot,
    [Parameter(Mandatory = $true)][string]$FluxSnapshot,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$InferenceRevision,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$SceneWorksRevision,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$QwenModelRevision,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$QwenModelInventorySha256,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$FluxModelRevision,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$FluxModelInventorySha256,
    [ValidateRange(0, [int]::MaxValue)][int]$MinimumReductionMiB = 512
)

$ErrorActionPreference = "Stop"
$output = New-Item -ItemType Directory -Force -Path $OutputDirectory
$env:QWEN_IMAGE_SNAPSHOT = (Resolve-Path $QwenSnapshot).Path
$env:FLUX_DEV_DIR = (Resolve-Path $FluxSnapshot).Path
$env:INFERENCE_REVISION = $InferenceRevision
$env:SCENEWORKS_REVISION = $SceneWorksRevision
$env:MEMORY_EXPECTED_ABI = "3"
$QwenCalibrationFingerprint = "qwen-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v2"

$checkedOutRevision = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $checkedOutRevision -ne $InferenceRevision) {
    throw "InferenceRevision must equal the checked-out inference HEAD ($checkedOutRevision)"
}
if (& git status --porcelain) {
    throw "The inference checkout must be clean before producing calibration evidence"
}

function Invoke-Probe {
    param(
        [string]$Package,
        [string]$Test,
        [string]$ModeVariable,
        [string]$Mode,
        [string]$OutputVariable,
        [string]$RgbName,
        [string]$LogName,
        [string]$ExpectedFingerprint,
        [string]$ModelRevision,
        [string]$ModelInventorySha256
    )

    if ($Mode) {
        Set-Item "env:$ModeVariable" $Mode
    } else {
        Remove-Item "env:$ModeVariable" -ErrorAction SilentlyContinue
    }
    Set-Item "env:$OutputVariable" (Join-Path $output $RgbName)
    $env:MEMORY_EXPECTED_FINGERPRINT = $ExpectedFingerprint
    $env:MEMORY_MODEL_REVISION = $ModelRevision
    $env:MEMORY_MODEL_INVENTORY_SHA256 = $ModelInventorySha256
    & cargo test --locked -p $Package --features cuda $Test -- --ignored --nocapture *>&1 |
        Tee-Object -FilePath (Join-Path $output $LogName)
    if ($LASTEXITCODE -ne 0) {
        throw "$Package $Mode probe failed with exit code $LASTEXITCODE"
    }
}

Invoke-Probe candle-gen-qwen-image qwen_image_probed_generate_for_offload_ab `
    QWEN_OFFLOAD_MODE "" QWEN_OUT qwen-resident.rgb qwen-resident.log $QwenCalibrationFingerprint `
    $QwenModelRevision $QwenModelInventorySha256
Invoke-Probe candle-gen-qwen-image qwen_image_probed_generate_for_offload_ab `
    QWEN_OFFLOAD_MODE request-staged QWEN_OUT qwen-sequential.rgb qwen-sequential.log $QwenCalibrationFingerprint `
    $QwenModelRevision $QwenModelInventorySha256

Invoke-Probe candle-gen-flux flux_dev_probed_generate_for_offload_ab `
    FLUX_OFFLOAD_MODE "" FLUX_OUT flux-dev-resident.rgb flux-dev-resident.log flux1-cuda-residency-v1 `
    $FluxModelRevision $FluxModelInventorySha256
Invoke-Probe candle-gen-flux flux_dev_probed_generate_for_offload_ab `
    FLUX_OFFLOAD_MODE request-staged FLUX_OUT flux-dev-sequential.rgb flux-dev-sequential.log flux1-cuda-residency-v1 `
    $FluxModelRevision $FluxModelInventorySha256

$qwenResident = Join-Path $output qwen-resident.rgb
$qwenSequential = Join-Path $output qwen-sequential.rgb
$fluxResident = Join-Path $output flux-dev-resident.rgb
$fluxSequential = Join-Path $output flux-dev-sequential.rgb

& fc.exe /b $qwenResident $qwenSequential
if ($LASTEXITCODE -ne 0) { throw "Qwen resident and sequential output differ" }
& fc.exe /b $fluxResident $fluxSequential
if ($LASTEXITCODE -ne 0) { throw "FLUX resident and sequential output differ" }

python scripts/release/verify_residency_ab.py --model qwen_image `
    --resident (Join-Path $output qwen-resident.log) `
    --sequential (Join-Path $output qwen-sequential.log) `
    --resident-output $qwenResident --sequential-output $qwenSequential `
    --expected-fingerprint $QwenCalibrationFingerprint --expected-abi 3 `
    --expected-model-revision $QwenModelRevision `
    --expected-model-inventory-sha256 $QwenModelInventorySha256 `
    --min-reduction-mib $MinimumReductionMiB
if ($LASTEXITCODE -ne 0) { throw "Qwen VRAM comparison failed" }

python scripts/release/verify_residency_ab.py --model flux1_dev `
    --resident (Join-Path $output flux-dev-resident.log) `
    --sequential (Join-Path $output flux-dev-sequential.log) `
    --resident-output $fluxResident --sequential-output $fluxSequential `
    --expected-fingerprint flux1-cuda-residency-v1 --expected-abi 3 `
    --expected-model-revision $FluxModelRevision `
    --expected-model-inventory-sha256 $FluxModelInventorySha256 `
    --min-reduction-mib $MinimumReductionMiB
if ($LASTEXITCODE -ne 0) { throw "FLUX VRAM comparison failed" }

Get-FileHash $qwenResident, $qwenSequential, $fluxResident, $fluxSequential |
    Format-Table -AutoSize |
    Out-File (Join-Path $output checksums.sha256)
