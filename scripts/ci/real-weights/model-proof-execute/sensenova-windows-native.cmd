@echo off
setlocal
if not defined REVIEWED_PYTHON exit /b 1
set "SENSENOVA_HUB_LIB=%RUNNER_TEMP%\sensenova-hub-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%"
"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%SENSENOVA_HUB_LIB%" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%SENSENOVA_HUB_LIB%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "RUSTC_WRAPPER="
"%REVIEWED_PYTHON%" scripts\ci\sensenova_loop_assets.py --backend candle-cuda --expected-sha "%GITHUB_SHA%" --work "%RUNNER_TEMP%\sensenova-native-work-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%" --report "%SENSENOVA_NATIVE_REPORT%" --oracle-dir "%SENSENOVA_REFERENCE_REPORT%"
