@echo off
setlocal
if not defined REVIEWED_PYTHON exit /b 1
set "SENSENOVA_HUB_LIB=%RUNNER_TEMP%\sensenova-hub-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%"
"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%SENSENOVA_HUB_LIB%" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%SENSENOVA_HUB_LIB%"
set "SENSENOVA_REFERENCE_VENV=%RUNNER_TEMP%\sensenova-venv-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%"
set "SENSENOVA_REFERENCE_PYTHON=%SENSENOVA_REFERENCE_VENV%/Scripts/python.exe"
uv venv --no-config --python "%REVIEWED_PYTHON%" "%SENSENOVA_REFERENCE_VENV%" || exit /b 1
uv pip install --no-config --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-first-match --only-binary=:all: --require-hashes --python "%SENSENOVA_REFERENCE_PYTHON%" -r .github/requirements/real-weights-sensenova-loop-reference-windows-x64-py312.txt || exit /b 1
"%REVIEWED_PYTHON%" scripts\ci\sensenova_loop_assets.py --backend candle-cuda --expected-sha "%GITHUB_SHA%" --work "%RUNNER_TEMP%\sensenova-reference-work-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%" --report "%SENSENOVA_REFERENCE_REPORT%" --reference-python "%SENSENOVA_REFERENCE_PYTHON%"
