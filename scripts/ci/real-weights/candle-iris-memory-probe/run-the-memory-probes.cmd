rem sc-25690 - the terminal CUDA memory measurement of Iris FULL fine-tuning (batch 1, 2 optimizer
rem steps + export): one process per shape, the device VRAM sampled by the test's VramProbe from
rem before the load (report-only). The card is the one with the MOST free memory at the start (the
rem inventory is recorded; another tenant may hold a card), selected with CUDA_VISIBLE_DEVICES in
rem PCI bus order. A CUDA OOM is recorded as "exceeds this card", never as a peak.
setlocal EnableDelayedExpansion
call "%VCVARS%"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv > "%IRIS_OUT%\cuda-device-inventory-before-probes.csv"
type "%IRIS_OUT%\cuda-device-inventory-before-probes.csv"
set "IRIS_PROBE_CARD="
set "IRIS_PROBE_FREE=0"
for /f "tokens=1,2 delims=, " %%A in ('nvidia-smi --query-gpu^=index^,memory.free --format^=csv^,noheader^,nounits') do (
  if %%B GTR !IRIS_PROBE_FREE! (set "IRIS_PROBE_CARD=%%A" & set "IRIS_PROBE_FREE=%%B")
)
if not defined IRIS_PROBE_CARD (echo ::error::no CUDA card found & exit /b 1)
echo selected card %IRIS_PROBE_CARD% with %IRIS_PROBE_FREE% MiB free>>"%IRIS_OUT%\summary.txt"
type "%IRIS_OUT%\summary.txt"
set "CUDA_VISIBLE_DEVICES=%IRIS_PROBE_CARD%"
set "TMPDIR=%RUNNER_TEMP%"
set "TEMP=%RUNNER_TEMP%"
set "IRIS_PROBE_STEPS=2"
set "IRIS_FAILED=0"
call :probe generation 1024 full
call :probe generation 1024 none
call :probe depth 1024 none
call :probe restoration 1024 none
call :probe generation 512 none
call :probe depth 512 none
call :probe restoration 512 none
type "%IRIS_OUT%\summary.txt"
exit /b %IRIS_FAILED%

rem %1 task, %2 size, %3 activation checkpointing
:probe
set "IRIS_PROBE_TASK=%~1"
set "IRIS_PROBE_SIZE=%~2"
set "IRIS_PROBE_AC=%~3"
set "log=%IRIS_OUT%\%~1-%~2-ac-%~3.log"
cargo test --locked --release -p candle-gen-iris --features cuda --test integration train_memory_probe::real_training_memory_probe -- --ignored --exact --nocapture --test-threads 1 > "%log%" 2>&1
set "rc=%ERRORLEVEL%"
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul
if not errorlevel 1 (
  for /f "delims=" %%L in ('findstr /C:"[[IRIS_MEMORY_PROBE]] vram" "%log%"') do echo [[IRIS_MEMORY_PROBE_CUDA]] %~1-%~2-ac-%~3 MEASURED %%L>>"%IRIS_OUT%\summary.txt"
  exit /b 0
)
findstr /I /C:"out of memory" /C:"CUDA_ERROR_OUT_OF_MEMORY" "%log%" >nul
if not errorlevel 1 (
  echo [[IRIS_MEMORY_PROBE_CUDA]] %~1-%~2-ac-%~3 EXCEEDS-CARD (CUDA out of memory on card %IRIS_PROBE_CARD%, %IRIS_PROBE_FREE% MiB free at start)>>"%IRIS_OUT%\summary.txt"
  exit /b 0
)
echo [[IRIS_MEMORY_PROBE_CUDA]] %~1-%~2-ac-%~3 FAILED rc=%rc%>>"%IRIS_OUT%\summary.txt"
set "IRIS_FAILED=1"
exit /b 0
