for %%I in ("%RUNNER_TEMP%\veda-s3-cuda-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%") do set "S3_OUTPUT_DIR=%%~fI"
for %%I in ("%RUNNER_TEMP%\veda-s3-cuda-target-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%") do set "CARGO_TARGET_DIR=%%~fI"
mkdir "%S3_OUTPUT_DIR%" || exit /b 1
echo S3_OUTPUT_DIR=%S3_OUTPUT_DIR%>>"%GITHUB_ENV%"
echo CARGO_TARGET_DIR=%CARGO_TARGET_DIR%>>"%GITHUB_ENV%"
>"%S3_OUTPUT_DIR%\preflight.json" echo {"instrument":"veda_s3_cuda","complete":false,"status":"tooling_not_yet_checked","runner":"%RUNNER_NAME%","host":"%COMPUTERNAME%","source_commit":"%GITHUB_SHA%","gpu_started":false,"device_test_started":false}
if not "%RUNNER_NAME%"=="cuda-windows" if not "%RUNNER_NAME%"=="cuda-windows-2" (
  echo Unexpected CUDA listener %RUNNER_NAME%
  exit /b 1
)
