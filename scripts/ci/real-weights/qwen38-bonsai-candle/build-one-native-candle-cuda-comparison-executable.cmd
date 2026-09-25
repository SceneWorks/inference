call "%VCVARS%" || exit /b 1
cargo test --locked --release -p candle-llm --features cuda --test qwen38_bonsai --no-run --message-format=json > "%RUNNER_TEMP%\qwen38-bonsai-candle\build.jsonl" 2> "%RUNNER_TEMP%\qwen38-bonsai-candle\build.stderr" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/qwen38_bonsai_terminal.py resolve-binary --target qwen38_bonsai < "%RUNNER_TEMP%\qwen38-bonsai-candle\build.jsonl" > "%RUNNER_TEMP%\qwen38-bonsai-candle\binary.txt" || exit /b 1
