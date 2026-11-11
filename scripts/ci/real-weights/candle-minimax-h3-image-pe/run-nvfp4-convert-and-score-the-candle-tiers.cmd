call "%VCVARS%"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
if not defined MINIMAX_PE_GPU (echo ::error::no GPU was selected by the evidence step & exit /b 1)
set "CUDA_VISIBLE_DEVICES=%MINIMAX_PE_GPU%"
echo tier,gpu_index,gpu_name,free_mib,required_free_mib,verdict> "%MINIMAX_PE_OUT%\gpu-free-memory-per-tier.csv"
cargo test --locked --release -p candle-llm --features cuda --test minimax_h3_image_pe --no-run || exit /b 1
cargo build --locked --release -p candle-llm --example modelopt_nvfp4_to_bf16 --example prepare_snapshot || exit /b 1
set "MINIMAX_PE_PROMPTS=%MINIMAX_PE_SNAPSHOT%"
set "MINIMAX_PE_REFERENCE=%MINIMAX_PE_OUT%\nvfp4.json"
rem 1. The reference: the published NVFP4 weights, natively (W4A4 cuBLASLt / W4A16 decode GEMV).
call :run_tier nvfp4 24576 "%MINIMAX_PE_SNAPSHOT%" || exit /b 1
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
call :run_tier candle-bf16 63488 "%MINIMAX_PE_TIERS%\bf16" || set "FAILED=1"
call :run_tier candle-q8 36864 "%MINIMAX_PE_TIERS%\candle-q8" || set "FAILED=1"
call :run_tier candle-q4 24576 "%MINIMAX_PE_TIERS%\candle-q4" || set "FAILED=1"
if not "%FAILED%"=="0" exit /b 1
exit /b 0

rem %1 = tier label, %2 = free MiB the selected card must have, %3 = snapshot to load. Each tier is
rem its own process. The floors are conservative constants, not measurements: the weights plus the
rem request working set, in MiB. bf16 is about 53 GiB of weights + 5.9 GiB request = 62 GiB; q8 holds
rem about half the weights (27 + 6 = 33, rounded up to 36 GiB); q4 and the native NVFP4 reference
rem about a quarter (14 + 6 = 20, rounded up to 24 GiB). A tier the card cannot fit FAILS the job by
rem name - it is never skipped - and its refusal is a row in gpu-free-memory-per-tier.csv.
:run_tier
set "MINIMAX_PE_LABEL=%~1"
set "MINIMAX_PE_TIER=%~3"
set "TIER_FREE="
for /f "tokens=1 delims=, " %%F in ('nvidia-smi -i %MINIMAX_PE_GPU% --query-gpu^=memory.free --format^=csv^,noheader^,nounits') do set "TIER_FREE=%%F"
set "TIER_GPU_NAME="
for /f "tokens=1 delims=," %%N in ('nvidia-smi -i %MINIMAX_PE_GPU% --query-gpu^=name --format^=csv^,noheader') do set "TIER_GPU_NAME=%%N"
if not defined TIER_FREE (echo ::error::%~1 could not read free memory of GPU %MINIMAX_PE_GPU% & exit /b 1)
if %TIER_FREE% LSS %~2 (
  echo %~1,%MINIMAX_PE_GPU%,%TIER_GPU_NAME%,%TIER_FREE%,%~2,REFUSED>> "%MINIMAX_PE_OUT%\gpu-free-memory-per-tier.csv"
  echo ::error::tier %~1 REFUSED: GPU %MINIMAX_PE_GPU% %TIER_GPU_NAME% has %TIER_FREE% MiB free, the tier needs %~2 MiB - another tenant holds the card
  exit /b 1
)
echo %~1,%MINIMAX_PE_GPU%,%TIER_GPU_NAME%,%TIER_FREE%,%~2,RUN>> "%MINIMAX_PE_OUT%\gpu-free-memory-per-tier.csv"
set "MINIMAX_PE_OUTPUT=%MINIMAX_PE_OUT%\%~1.json"
set "log=%MINIMAX_PE_OUT%\%~1.log"
cargo test --locked --release -p candle-llm --features cuda --test minimax_h3_image_pe minimax_h3_image_pe_quality_against_nvfp4 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & echo ::error::%~1 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
