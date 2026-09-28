call "%VCVARS%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/qwen38_bonsai_terminal.py check-gpu-reservation --gpu-index 0 --reservation "%BONSAI_GPU_RESERVATION%" --token "%BONSAI_RESERVATION_TOKEN%" --output "%RUNNER_TEMP%\qwen38-bonsai-candle\cuda-oracle-device-recheck.json" || exit /b 1
set "CUDA_ORACLE_LOG=%RUNNER_TEMP%\qwen38-bonsai-candle\cuda-packed-operator-oracles.log"
cargo test --locked -p candle-llm --features cuda --lib primitives::prism::tests::cuda_packed_operator_oracles_compile_and_execute_nvrtc -- --exact --nocapture > "%CUDA_ORACLE_LOG%" 2>&1 || (type "%CUDA_ORACLE_LOG%" & exit /b 1)
type "%CUDA_ORACLE_LOG%"
findstr /C:"test result: ok. 1 passed" "%CUDA_ORACLE_LOG%" >nul || exit /b 1
