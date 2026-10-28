"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model supra2-img-cuda --snapshot "%SUPRA2_SOURCE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model flan-t5-base-cuda --snapshot "%SUPRA2_T5_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model sd-vae-ft-mse-cuda --snapshot "%SUPRA2_VAE_SNAPSHOT%" || exit /b 1
