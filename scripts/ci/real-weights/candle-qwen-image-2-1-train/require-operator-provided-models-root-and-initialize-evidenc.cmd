if "%CANDLE_GEN_MODELS_ROOT%"=="" (
  echo Missing repository variable CANDLE_GEN_MODELS_ROOT for the Windows CUDA runner.
  exit /b 1
)
if exist "%RUNNER_TEMP%\qwen-image-2-1-train-cuda-evidence" rmdir /s /q "%RUNNER_TEMP%\qwen-image-2-1-train-cuda-evidence"
mkdir "%RUNNER_TEMP%\qwen-image-2-1-train-cuda-evidence" || exit /b 1
echo QWEN21_TRAIN_OUT=%RUNNER_TEMP%\qwen-image-2-1-train-cuda-evidence>>"%GITHUB_ENV%"
