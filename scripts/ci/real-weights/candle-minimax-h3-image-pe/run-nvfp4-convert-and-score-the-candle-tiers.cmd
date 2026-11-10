call "%VCVARS%"
cargo test --locked --release -p candle-llm --features cuda --test minimax_h3_image_pe --no-run || exit /b 1
cargo build --locked --release -p candle-llm --example modelopt_nvfp4_to_bf16 --example prepare_snapshot || exit /b 1
set "MINIMAX_PE_PROMPTS=%MINIMAX_PE_SNAPSHOT%"
set "MINIMAX_PE_REFERENCE=%MINIMAX_PE_OUT%\nvfp4.json"
rem 1. The reference: the published NVFP4 weights, natively (W4A4 cuBLASLt / W4A16 decode GEMV).
call :run_tier nvfp4 "%MINIMAX_PE_SNAPSHOT%" || exit /b 1
rem 2. The tiers: NVFP4 -> bf16 on the CPU, then Candle GGML Q8 / Q4 from the bf16 snapshot.
if exist "%MINIMAX_PE_TIERS%" rmdir /s /q "%MINIMAX_PE_TIERS%"
mkdir "%MINIMAX_PE_TIERS%" || exit /b 1
cargo run --locked --release -p candle-llm --example modelopt_nvfp4_to_bf16 -- "%MINIMAX_PE_SNAPSHOT%" "%MINIMAX_PE_TIERS%\bf16" > "%MINIMAX_PE_OUT%\convert-bf16.log" 2>&1 || (type "%MINIMAX_PE_OUT%\convert-bf16.log" & exit /b 1)
type "%MINIMAX_PE_OUT%\convert-bf16.log"
for %%Q in (q8 q4) do (
  cargo run --locked --release -p candle-llm --example prepare_snapshot -- "%MINIMAX_PE_TIERS%\bf16" "%MINIMAX_PE_TIERS%\candle-%%Q" %%Q > "%MINIMAX_PE_OUT%\prepare-candle-%%Q.log" 2>&1 || (type "%MINIMAX_PE_OUT%\prepare-candle-%%Q.log" & exit /b 1)
  type "%MINIMAX_PE_OUT%\prepare-candle-%%Q.log"
)
set "FAILED=0"
call :run_tier candle-bf16 "%MINIMAX_PE_TIERS%\bf16" || set "FAILED=1"
call :run_tier candle-q8 "%MINIMAX_PE_TIERS%\candle-q8" || set "FAILED=1"
call :run_tier candle-q4 "%MINIMAX_PE_TIERS%\candle-q4" || set "FAILED=1"
if not "%FAILED%"=="0" exit /b 1
exit /b 0

rem %1 = tier label, %2 = snapshot to load. Each tier is its own process.
:run_tier
set "MINIMAX_PE_LABEL=%~1"
set "MINIMAX_PE_TIER=%~2"
set "MINIMAX_PE_OUTPUT=%MINIMAX_PE_OUT%\%~1.json"
set "log=%MINIMAX_PE_OUT%\%~1.log"
cargo test --locked --release -p candle-llm --features cuda --test minimax_h3_image_pe minimax_h3_image_pe_quality_against_nvfp4 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & echo ::error::%~1 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
