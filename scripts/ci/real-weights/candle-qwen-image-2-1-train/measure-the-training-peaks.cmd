call "%VCVARS%"
set "QWEN21_FAILED=0"
for %%C in (t2i-lora-256-dense t2i-lora-768-ckpt t2i-lokr-768-ckpt t2i-lora-1024-ckpt edit-lokr-768-ckpt-1ref edit-lora-768-ckpt-1ref) do call :run_one %%C || set "QWEN21_FAILED=1"
findstr /B /C:"QWEN21_TRAIN_PEAK " "%QWEN21_TRAIN_OUT%\*.log" > "%QWEN21_TRAIN_OUT%\training-peaks.txt"
type "%QWEN21_TRAIN_OUT%\training-peaks.txt"
if not "%QWEN21_FAILED%"=="0" exit /b 1
exit /b 0

:run_one
set "log=%QWEN21_TRAIN_OUT%\%~1.log"
set "QWEN21_TRAIN_CELL=%~1"
cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda --test integration train_peak_real_weights::training_peak_is_within_the_preflight_estimate -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & echo ::error::%~1 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test & exit /b 1)
exit /b 0
