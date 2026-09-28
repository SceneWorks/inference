"%REVIEWED_PYTHON%" scripts/release/qwen38_bonsai_terminal.py qualify-snapshots --platform windows --binding BONSAI_QWEN38_SNAPSHOT=bonsai-qwen38-parent --binding BONSAI_MLX_SNAPSHOT=bonsai-mlx-2bit --binding BONSAI_GGUF_SNAPSHOT=bonsai-gguf --binding BONSAI_BASELINE_SNAPSHOT=bonsai-qwen3vl-baseline --output "%RUNNER_TEMP%\qwen38-bonsai-candle\snapshot-metadata-before.json"
set "STATUS=%ERRORLEVEL%"
type "%RUNNER_TEMP%\qwen38-bonsai-candle\snapshot-metadata-before.json"
exit /b %STATUS%
