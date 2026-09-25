call "%VCVARS%"
rem Record the card state IMMEDIATELY before the render: the probe's 1 GB idle-baseline
rem assertion is what rejects a contaminated number, and this is the evidence for why.
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
rem `name=` is the single-selection vocabulary
rem `test_minimax_h3_lanes_select_tests_that_exist_and_pin_their_run_count` binds against,
rem so this selection is checked to name a test that still EXISTS and is still `#[ignore]`d.
set "name=minimax_h3_vram_bf16"
set "log=%RUNNER_TEMP%\minimax-h3-vram-bf16.log"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --release --test integration vram_probe::%name% -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::minimax_h3_vram_bf16 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
rem The machine-parseable datum is the whole point of the run; a green step that produced
rem no `[[H3_VRAM]]` line has measured nothing anyone can transcribe into the manifest.
"%REVIEWED_PYTHON%" scripts/ci/validate_h3_vram_receipt.py --log "%log%" --tier bf16 --out "%RUNNER_TEMP%\minimax-h3-vram-bf16.json" || exit /b 1
type "%RUNNER_TEMP%\minimax-h3-vram-bf16.json"
