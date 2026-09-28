"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model qwen-image-2-1-cuda --snapshot "%CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model qwen-image-2-1-mlx --snapshot "%CANDLE_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT%" || exit /b 1
