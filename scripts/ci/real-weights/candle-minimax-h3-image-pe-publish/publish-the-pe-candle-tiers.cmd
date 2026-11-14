set "HF_HUB_DISABLE_XET=1"
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/ci/publish_hf_tiers.py --repo SceneWorks/minimax-h3-image-pe --map "%PUBLISH_SCRATCH%\candle-q8=candle-q8" --map "%PUBLISH_SCRATCH%\candle-q4=candle-q4" --message "SW-7104: MiniMax-H3-Image-PE Candle q8/q4 tiers (inference %GITHUB_SHA%)" --report "%PUBLISH_REPORTS%\minimax-h3-image-pe-candle.json" || exit /b 1
exit /b 0
