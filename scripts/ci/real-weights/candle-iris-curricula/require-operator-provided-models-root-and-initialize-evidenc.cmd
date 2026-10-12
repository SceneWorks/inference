if "%CANDLE_GEN_MODELS_ROOT%"=="" (
  echo Missing repository variable CANDLE_GEN_MODELS_ROOT for the Windows CUDA runner.
  exit /b 1
)
if exist "%RUNNER_TEMP%\iris-curricula-cuda-evidence" rmdir /s /q "%RUNNER_TEMP%\iris-curricula-cuda-evidence"
mkdir "%RUNNER_TEMP%\iris-curricula-cuda-evidence" || exit /b 1
echo IRIS_OUT=%RUNNER_TEMP%\iris-curricula-cuda-evidence>>"%GITHUB_ENV%"
