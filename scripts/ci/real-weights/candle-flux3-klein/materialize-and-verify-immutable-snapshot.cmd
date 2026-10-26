"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
rem The package is GATED (FLUX Kommunity License): the manifest row carries `requires_auth = true`,
rem so the fetch uses the runner's own configured Hugging Face credential and nothing is interpolated here.
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model flux3-image-klein-cuda --snapshot "%FLUX3_PACKAGE_DIR%" || exit /b 1
