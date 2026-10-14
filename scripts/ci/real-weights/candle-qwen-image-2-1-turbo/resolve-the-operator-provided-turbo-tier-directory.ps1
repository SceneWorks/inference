$tiers = $env:QWEN21_TURBO_TIER_INPUT
if ([string]::IsNullOrWhiteSpace($tiers)) {
  Write-Output "no qwen_image_2_1_turbo_tier_snapshot input: turbo_validation_render_installed_tiers will not run"
  Add-Content -Path $env:GITHUB_ENV -Value 'QWEN21_TURBO_TIERS=0'
  exit 0
}
if (-not [System.IO.Path]::IsPathRooted($tiers) -or $tiers -notmatch '^[A-Za-z]:\\[A-Za-z0-9_.\\ -]+$') {
  Write-Output "::error::qwen_image_2_1_turbo_tier_snapshot must be an absolute local directory (letters, digits, _ . - space), got: $tiers"
  exit 1
}
foreach ($tier in @('q8', 'q4')) {
  foreach ($file in @('model_index.json', 'LICENSE', 'transformer\config.json', 'transformer\model.safetensors')) {
    $path = Join-Path (Join-Path $tiers $tier) $file
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
      Write-Output "::error::$path is missing: the directory must hold complete q8\ and q4\ Turbo tiers"
      exit 1
    }
  }
}
Add-Content -Path $env:GITHUB_ENV -Value "CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT=$tiers"
Add-Content -Path $env:GITHUB_ENV -Value 'QWEN21_TURBO_TIERS=1'
