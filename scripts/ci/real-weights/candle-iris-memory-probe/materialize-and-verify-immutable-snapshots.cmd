"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model iris-3b-cuda --snapshot "%IRIS_WEIGHTS_DIR%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model qwen3-vl-4b-instruct-cuda --snapshot "%IRIS_TEXT_ENCODER_DIR%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model openclip-convnext-xxlarge-cuda --snapshot "%IRIS_CRITIC_TRUNK_DIR%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model vgg16-tv-in1k-cuda --snapshot "%IRIS_LPIPS_VGG16_DIR%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model lpips-vgg-lin-cuda --snapshot "%IRIS_LPIPS_LIN_DIR%" || exit /b 1
