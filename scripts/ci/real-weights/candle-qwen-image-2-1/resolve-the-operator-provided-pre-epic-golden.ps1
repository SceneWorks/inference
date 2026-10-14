$golden = $env:QWEN21_RENDER_GOLDEN_INPUT
if ([string]::IsNullOrWhiteSpace($golden)) {
  Write-Output "no qwen_image_2_1_render_golden input: the base renders are not compared with a pre-epic golden"
  exit 0
}
if (-not [System.IO.Path]::IsPathRooted($golden) -or $golden -notmatch '^[A-Za-z]:\\[A-Za-z0-9_.\\ -]+$') {
  Write-Output "::error::qwen_image_2_1_render_golden must be an absolute local directory (letters, digits, _ . - space), got: $golden"
  exit 1
}
foreach ($file in @(
    'qwen_image_2_1_candle_2048x2048_40steps_seed42.rgb',
    'qwen_image_2_1_candle_edit_2refs_1024x1024_8steps_seed42.rgb',
    'qwen_image_2_1_candle_edit_10refs_512x512_2steps_seed42.rgb')) {
  $path = Join-Path $golden $file
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    Write-Output "::error::$path is missing: copy the pre-epic qwen-image-2-1-cuda-evidence renders here"
    exit 1
  }
}
Add-Content -Path $env:GITHUB_ENV -Value "QWEN_IMAGE_2_1_RENDER_GOLDEN=$golden"
Add-Content -Path $env:GITHUB_ENV -Value 'QWEN_IMAGE_2_1_PREFIX_KV_CACHE=prefill'
