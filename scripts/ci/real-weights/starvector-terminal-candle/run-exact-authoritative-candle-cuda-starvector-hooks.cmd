call "%VCVARS%" || exit /b 1
cargo test --locked -p candle-llm --features cuda starvector::tests::real_weight_provider_satisfies_shared_starvector_conformance -- --exact --ignored --nocapture > "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks\\candle-cuda-starvector-1b.log" 2>&1 || (type "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks\\candle-cuda-starvector-1b.log" & exit /b 1)
type "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks\\candle-cuda-starvector-1b.log"
findstr /C:"test result: ok. 1 passed" "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks\\candle-cuda-starvector-1b.log" >nul || exit /b 1
cargo test --locked -p candle-llm --features cuda starvector_8b::tests::real_weight_provider_satisfies_shared_starvector_conformance -- --exact --ignored --nocapture > "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks\\candle-cuda-starvector-8b.log" 2>&1 || (type "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks\\candle-cuda-starvector-8b.log" & exit /b 1)
type "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks\\candle-cuda-starvector-8b.log"
findstr /C:"test result: ok. 1 passed" "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks\\candle-cuda-starvector-8b.log" >nul || exit /b 1
