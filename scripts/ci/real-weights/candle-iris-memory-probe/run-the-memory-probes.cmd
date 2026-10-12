rem sc-25690 - the terminal CUDA memory measurement of Iris FULL fine-tuning (batch 1, 2 optimizer
rem steps + export): one process per shape. The test prints the driver pool's continuous counters
rem (USED_MEM_HIGH = live bytes, RESERVED_MEM_HIGH = the unit nvidia-smi reads) next to the polled
rem device-level VramProbe peak; the iris_memory_probe_row script classifies each log into ONE
rem summary row, every figure in decimal GB: MEASURED, AT-CARD-LIMIT (lower bound), EXCEEDS-CARD
rem (lower bound: CUDA out of memory) or FAILED. The card is the one with the MOST free memory at
rem the start (another tenant may hold a card), selected with CUDA_VISIBLE_DEVICES in PCI bus
rem order, and its memory is snapshotted again right before every probe. Report-only: the job is
rem red only for a FAILED or missing row.
rem
rem No command here sits inside a parenthesized block and no echoed text carries a parenthesis: an
rem unescaped closing parenthesis in an echo ends the enclosing block early, which is how the first
rem version of this script dropped every non-OOM failure.
setlocal EnableDelayedExpansion
call "%VCVARS%"
if not defined REVIEWED_PYTHON goto :no_python
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "IRIS_SUMMARY=%IRIS_OUT%\summary.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv > "%IRIS_OUT%\cuda-device-inventory-before-probes.csv"
type "%IRIS_OUT%\cuda-device-inventory-before-probes.csv"
set "IRIS_PROBE_CARD="
set "IRIS_PROBE_FREE=0"
for /f "tokens=1,2 delims=, " %%A in ('nvidia-smi --query-gpu^=index^,memory.free --format^=csv^,noheader^,nounits') do call :consider_card %%A %%B
if not defined IRIS_PROBE_CARD goto :no_card
"%REVIEWED_PYTHON%" scripts\ci\iris_memory_probe_row.py card --inventory "%IRIS_OUT%\cuda-device-inventory-before-probes.csv" --card %IRIS_PROBE_CARD% --summary "%IRIS_SUMMARY%"
if errorlevel 1 exit /b 1
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
type "%IRIS_SUMMARY%"
"%REVIEWED_PYTHON%" scripts\ci\iris_memory_probe_row.py expect --backend cuda --summary "%IRIS_SUMMARY%" generation-1024-ac-full generation-1024-ac-none depth-1024-ac-none restoration-1024-ac-none generation-512-ac-none depth-512-ac-none restoration-512-ac-none
if errorlevel 1 set "IRIS_FAILED=1"
exit /b %IRIS_FAILED%

:no_python
echo ::error::REVIEWED_PYTHON is not set - the reviewed CPython step did not run
exit /b 1

:no_card
echo ::error::no CUDA card found
exit /b 1

rem %1 card index, %2 its free MiB: keep the card with the most free memory
:consider_card
if %~2 GTR %IRIS_PROBE_FREE% set "IRIS_PROBE_CARD=%~1"
if %~2 GTR %IRIS_PROBE_FREE% set "IRIS_PROBE_FREE=%~2"
exit /b 0

rem %1 task, %2 size, %3 activation checkpointing
:probe
set "IRIS_PROBE_TASK=%~1"
set "IRIS_PROBE_SIZE=%~2"
set "IRIS_PROBE_AC=%~3"
set "name=%~1-%~2-ac-%~3"
set "log=%IRIS_OUT%\%name%.log"
set "snapshot=%IRIS_OUT%\%name%-card-before.csv"
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits -i %IRIS_PROBE_CARD% > "%snapshot%"
type "%snapshot%"
cargo test --locked --release -p candle-gen-iris --features cuda --test integration train_memory_probe::real_training_memory_probe -- --ignored --exact --nocapture --test-threads 1 > "%log%" 2>&1
set "rc=%ERRORLEVEL%"
type "%log%"
"%REVIEWED_PYTHON%" scripts\ci\iris_memory_probe_row.py cuda --name "%name%" --log "%log%" --rc %rc% --card-snapshot "%snapshot%" --card %IRIS_PROBE_CARD% --summary "%IRIS_SUMMARY%"
if errorlevel 1 set "IRIS_FAILED=1"
exit /b 0
