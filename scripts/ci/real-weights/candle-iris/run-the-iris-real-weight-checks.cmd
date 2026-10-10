call "%VCVARS%"
set "IRIS_FAILED=0"
call :run_one real_tokenizer_matches_the_pinned_ids || set "IRIS_FAILED=1"
call :run_one provider_renders_a_real_image || set "IRIS_FAILED=1"
call :run_one_in depth_real_weights real_depth_matches_the_upstream_reference || set "IRIS_FAILED=1"
if not "%IRIS_FAILED%"=="0" exit /b 1
exit /b 0

:run_one
call :run_one_in real_weights %~1
exit /b %ERRORLEVEL%

:run_one_in
set "log=%IRIS_OUT%\%~2.log"
cargo test --locked --release -p candle-gen-iris --features cuda --test integration %~1::%~2 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & echo ::error::%~2 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~2 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
