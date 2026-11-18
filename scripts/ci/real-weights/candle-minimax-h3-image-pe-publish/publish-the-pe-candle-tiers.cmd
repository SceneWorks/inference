set "HF_HUB_DISABLE_XET=1"
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
rem bf16/ is backend-neutral (both loaders read the dense convert_snapshot_to_bf16 layout), so this
rem job is its only publisher. publish_hf_tiers.py touches only the mapped prefixes.
if /i "%PUBLISH_PE_QUANT%"=="false" (
  "%REVIEWED_PYTHON%" scripts/ci/publish_hf_tiers.py --repo SceneWorks/minimax-h3-image-pe --map "%PUBLISH_SCRATCH%\bf16=bf16" --message "MiniMax-H3-Image-PE bf16 tier (inference %GITHUB_SHA%)" --report "%PUBLISH_REPORTS%\minimax-h3-image-pe-bf16.json" || exit /b 1
  exit /b 0
)
"%REVIEWED_PYTHON%" scripts/ci/publish_hf_tiers.py --repo SceneWorks/minimax-h3-image-pe --map "%PUBLISH_SCRATCH%\bf16=bf16" --map "%PUBLISH_SCRATCH%\candle-q8=candle-q8" --map "%PUBLISH_SCRATCH%\candle-q4=candle-q4" --message "MiniMax-H3-Image-PE bf16 and Candle q8/q4 tiers (inference %GITHUB_SHA%)" --report "%PUBLISH_REPORTS%\minimax-h3-image-pe-candle.json" || exit /b 1
exit /b 0
