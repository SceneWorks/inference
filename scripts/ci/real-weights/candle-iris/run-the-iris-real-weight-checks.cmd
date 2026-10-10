call "%VCVARS%"
set "IRIS_FAILED=0"
call :run_one real_tokenizer_matches_the_pinned_ids || set "IRIS_FAILED=1"
call :run_one provider_renders_a_real_image || set "IRIS_FAILED=1"
call :run_restore real_small_restoration_matches_upstream || set "IRIS_FAILED=1"
call :run_restore real_release_scale_restoration_renders || set "IRIS_FAILED=1"
if not "%IRIS_FAILED%"=="0" exit /b 1
exit /b 0

:run_one
set "log=%IRIS_OUT%\%~1.log"
cargo test --locked --release -p candle-gen-iris --features cuda --test integration real_weights::%~1 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & echo ::error::%~1 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0

:run_restore
set "log=%IRIS_OUT%\%~1.log"
cargo test --locked --release -p candle-gen-iris --features cuda --test integration restoration_real_weights::%~1 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & echo ::error::%~1 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
