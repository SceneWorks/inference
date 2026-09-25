"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model sdxl-base-1.0 --snapshot "%SDXL_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model clip-vit-large-patch14 --snapshot "%SDXL_TOKENIZER_CLIP_L_DIR%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model clip-vit-bigg-14-laion2b --snapshot "%SDXL_TOKENIZER_CLIP_BIGG_DIR%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model sdxl-vae-fp16-fix --snapshot "%SDXL_VAE_FP16_FIX_DIR%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model realvisxl-v5 --snapshot "%REALVISXL_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model ip-adapter-plus-sdxl-vit-h --snapshot "%IP_ADAPTER_SNAPSHOT%" || exit /b 1
