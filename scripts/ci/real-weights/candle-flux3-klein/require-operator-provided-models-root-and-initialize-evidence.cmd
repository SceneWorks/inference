if "%CANDLE_GEN_MODELS_ROOT%"=="" (
  echo Missing repository variable CANDLE_GEN_MODELS_ROOT for the Windows CUDA runner.
  exit /b 1
)
if exist "%RUNNER_TEMP%\flux3-klein-cuda-evidence" rmdir /s /q "%RUNNER_TEMP%\flux3-klein-cuda-evidence"
mkdir "%RUNNER_TEMP%\flux3-klein-cuda-evidence" || exit /b 1
echo FLUX3_OUT=%RUNNER_TEMP%\flux3-klein-cuda-evidence>>"%GITHUB_ENV%"
