@echo off
setlocal
if not defined REVIEWED_PYTHON exit /b 1
set "SENSENOVA_HUB_LIB=%RUNNER_TEMP%\sensenova-hub-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%"
if not "%~1"=="publish" "%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all --require-hashes --target "%SENSENOVA_HUB_LIB%" -r .github\requirements\real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%SENSENOVA_HUB_LIB%"
if "%~1"=="reference" (
  uv venv --no-config --python "%REVIEWED_PYTHON%" "%RUNNER_TEMP%\sensenova-venv-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%" || exit /b 1
  uv pip install --no-config --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-first-match --only-binary :all: --require-hashes --python "%RUNNER_TEMP%\sensenova-venv-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%\Scripts\python.exe" -r .github\requirements\real-weights-sensenova-loop-reference-windows-x64-py312.txt || exit /b 1
  "%REVIEWED_PYTHON%" scripts\ci\sensenova_loop_assets.py --backend candle-cuda --expected-sha "%GITHUB_SHA%" --work "%RUNNER_TEMP%\sensenova-reference-work-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%" --report "%SENSENOVA_REFERENCE_REPORT%" --reference-python "%RUNNER_TEMP%\sensenova-venv-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%\Scripts\python.exe"
  if errorlevel 1 exit /b 1
  exit /b 0
)
if "%~1"=="publish" (
  "%REVIEWED_PYTHON%" scripts\ci\sensenova_loop_publish.py || exit /b 1
  exit /b 0
)
if not "%~1"=="native" exit /b 1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "RUSTC_WRAPPER="
"%REVIEWED_PYTHON%" scripts\ci\sensenova_loop_assets.py --backend candle-cuda --expected-sha "%GITHUB_SHA%" --work "%RUNNER_TEMP%\sensenova-native-work-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%" --report "%SENSENOVA_NATIVE_REPORT%" --oracle-dir "%SENSENOVA_REFERENCE_REPORT%"
