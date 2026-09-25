if "%CANDLE_GEN_MODELS_ROOT%"=="" (
  echo Missing repository variable CANDLE_GEN_MODELS_ROOT for the Windows CUDA runner.
  exit /b 1
)
if exist "%RUNNER_TEMP%\qwen-image-2-1-cuda-evidence" rmdir /s /q "%RUNNER_TEMP%\qwen-image-2-1-cuda-evidence"
mkdir "%RUNNER_TEMP%\qwen-image-2-1-cuda-evidence" || exit /b 1
echo QWEN_IMAGE_2_1_RENDER_OUT=%RUNNER_TEMP%\qwen-image-2-1-cuda-evidence>>"%GITHUB_ENV%"
