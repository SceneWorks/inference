if "%CANDLE_GEN_MODELS_ROOT%"=="" (
  echo Missing repository variable CANDLE_GEN_MODELS_ROOT for the Windows CUDA runner.
  exit /b 1
)
if exist "%RUNNER_TEMP%\supra2-cuda-evidence" rmdir /s /q "%RUNNER_TEMP%\supra2-cuda-evidence"
mkdir "%RUNNER_TEMP%\supra2-cuda-evidence" || exit /b 1
echo SUPRA2_OUT=%RUNNER_TEMP%\supra2-cuda-evidence>>"%GITHUB_ENV%"
echo SUPRA2_OUTPUT_DIR=%RUNNER_TEMP%\supra2-cuda-evidence\renders>>"%GITHUB_ENV%"
echo SUPRA2_BUNDLE=%RUNNER_TEMP%\supra2-bundle>>"%GITHUB_ENV%"
