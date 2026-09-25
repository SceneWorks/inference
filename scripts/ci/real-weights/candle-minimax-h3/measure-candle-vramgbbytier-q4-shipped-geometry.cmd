call "%VCVARS%"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
set "name=minimax_h3_vram_q4"
set "log=%RUNNER_TEMP%\minimax-h3-vram-q4.log"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --release --test integration vram_probe::%name% -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::minimax_h3_vram_q4 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
"%REVIEWED_PYTHON%" scripts/ci/validate_h3_vram_receipt.py --log "%log%" --tier q4 --out "%RUNNER_TEMP%\minimax-h3-vram-q4.json" || exit /b 1
type "%RUNNER_TEMP%\minimax-h3-vram-q4.json"
