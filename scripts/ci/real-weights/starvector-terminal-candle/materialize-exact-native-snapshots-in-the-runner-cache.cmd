"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model starvector-1b-im2svg --export STARVECTOR_1B_SNAPSHOT || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model starvector-8b-im2svg --export STARVECTOR_8B_SNAPSHOT || exit /b 1
