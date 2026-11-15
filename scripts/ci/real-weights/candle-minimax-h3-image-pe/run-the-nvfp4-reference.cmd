call "%VCVARS%"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
if not defined MINIMAX_PE_GPU (echo ::error::no GPU was selected by the evidence step & exit /b 1)
set "CUDA_VISIBLE_DEVICES=%MINIMAX_PE_GPU%"
echo tier,gpu_index,gpu_name,free_mib,required_free_mib,verdict> "%MINIMAX_PE_OUT%\gpu-free-memory-per-tier.csv"
cargo test --locked --release -p candle-llm --features cuda --test minimax_h3_image_pe --no-run || exit /b 1
set "MINIMAX_PE_PROMPTS=%MINIMAX_PE_SNAPSHOT%"
set "MINIMAX_PE_REFERENCE=%MINIMAX_PE_OUT%\nvfp4.json"
rem The reference: the published NVFP4 weights, natively (W4A4 cuBLASLt / W4A16 decode GEMV). Once it
rem passes, say so: the next step uploads nvfp4.json and the MLX job needs only that, so a later
rem Candle tier failing does not stop the Metal tiers.
call :run_tier nvfp4 "%MINIMAX_PE_SNAPSHOT%" || exit /b 1
echo nvfp4_reference=true>> "%GITHUB_OUTPUT%"
exit /b 0

rem %1 = tier label, %2 = snapshot to load. Each tier is its own process. The floor is DERIVED, not
rem guessed: the bytes of the snapshot's weight files on disk (read now, by scripts/ci/tier-weight-mib.ps1, so it
rem follows what was actually converted) + the request working set + a fixed margin, all in MiB.
rem   request: 5632 MiB = the 5902159696 bytes the engine's admission asked for on the fixed set's
rem            largest case (3531 prompt + 4096 new tokens) in run 42936821283.
rem   margin:  10240 MiB for the CUDA context, cuBLAS workspaces and allocator slack. Run 42936821283
rem            measured 5.3 to 9 GiB beyond weights + request on the bf16 tier (card had 63573 MiB
rem            free before the load, 3606 MiB left at admission against 5629 needed).
rem A tier the card cannot fit FAILS the job by name - it is never skipped - and its refusal is a
rem row in gpu-free-memory-per-tier.csv.
:run_tier
set "MINIMAX_PE_LABEL=%~1"
set "MINIMAX_PE_TIER=%~2"
set "TIER_FREE="
set "WEIGHT_MIB="
for /f "usebackq" %%W in (`powershell -NoProfile -File scripts\ci\tier-weight-mib.ps1 "%~2"`) do set "WEIGHT_MIB=%%W"
if not defined WEIGHT_MIB (echo ::error::%~1 could not size the weights under %~2 & exit /b 1)
set /a TIER_NEED=WEIGHT_MIB+5632+10240
for /f "tokens=1 delims=, " %%F in ('nvidia-smi -i %MINIMAX_PE_GPU% --query-gpu^=memory.free --format^=csv^,noheader^,nounits') do set "TIER_FREE=%%F"
set "TIER_GPU_NAME="
for /f "tokens=1 delims=," %%N in ('nvidia-smi -i %MINIMAX_PE_GPU% --query-gpu^=name --format^=csv^,noheader') do set "TIER_GPU_NAME=%%N"
if not defined TIER_FREE (echo ::error::%~1 could not read free memory of GPU %MINIMAX_PE_GPU% & exit /b 1)
if %TIER_FREE% LSS %TIER_NEED% (
  echo %~1,%MINIMAX_PE_GPU%,%TIER_GPU_NAME%,%TIER_FREE%,%TIER_NEED%,REFUSED>> "%MINIMAX_PE_OUT%\gpu-free-memory-per-tier.csv"
  echo ::error::tier %~1 REFUSED: GPU %MINIMAX_PE_GPU% %TIER_GPU_NAME% has %TIER_FREE% MiB free, the tier needs %TIER_NEED% MiB - another tenant holds the card
  exit /b 1
)
echo %~1,%MINIMAX_PE_GPU%,%TIER_GPU_NAME%,%TIER_FREE%,%TIER_NEED%,RUN>> "%MINIMAX_PE_OUT%\gpu-free-memory-per-tier.csv"
set "MINIMAX_PE_OUTPUT=%MINIMAX_PE_OUT%\%~1.json"
set "log=%MINIMAX_PE_OUT%\%~1.log"
cargo test --locked --release -p candle-llm --features cuda --test minimax_h3_image_pe minimax_h3_image_pe_quality_against_nvfp4 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & echo ::error::%~1 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
