if not exist "%RUNNER_TEMP%\qwen38-bonsai-candle" mkdir "%RUNNER_TEMP%\qwen38-bonsai-candle" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/qwen38_bonsai_terminal.py hardware --output "%RUNNER_TEMP%\qwen38-bonsai-candle\hardware-before.json" || exit /b 1
type "%RUNNER_TEMP%\qwen38-bonsai-candle\hardware-before.json"
