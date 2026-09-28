set "BONSAI_GPU_RESERVATION=%LOCALAPPDATA%\SceneWorks\inference-reservations\cuda-device-0.json"
set "BONSAI_RESERVATION_TOKEN=%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%"
"%REVIEWED_PYTHON%" scripts/release/qwen38_bonsai_terminal.py reserve-gpu --gpu-index 0 --reservation "%BONSAI_GPU_RESERVATION%" --evidence "%RUNNER_TEMP%\qwen38-bonsai-candle\gpu-reservation.json" --token "%BONSAI_RESERVATION_TOKEN%" || exit /b 1
echo BONSAI_GPU_RESERVATION=%BONSAI_GPU_RESERVATION%>>"%GITHUB_ENV%"
echo BONSAI_RESERVATION_TOKEN=%BONSAI_RESERVATION_TOKEN%>>"%GITHUB_ENV%"
