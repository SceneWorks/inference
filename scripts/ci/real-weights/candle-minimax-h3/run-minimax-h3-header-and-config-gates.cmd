call "%VCVARS%"
call :run_one declared_tensor_names_match_the_published_checkpoint || exit /b 1
call :run_one declared_encoder_shapes_match_the_published_checkpoint || exit /b 1
call :run_one declared_audio_tensor_names_match_the_published_checkpoint || exit /b 1
call :run_one published_audio_configs_reproduce_the_declared_geometry || exit /b 1
call :run_one stored_kaiser_filters_match_the_derivation_on_real_weights || exit /b 1
rem sc-19558 — the two DiT header gates. Header-only and device-free, so they cost the
rem materialize step above and nothing else; they were listed as NOT COVERED for exactly
rem one reason, that `transformer/` was in no row, and `minimax-h3-dit` is that row.
call :run_one declared_dit_tensor_names_match_the_published_checkpoint || exit /b 1
call :run_one the_adaln_projections_are_the_documented_twenty_six_gigabytes || exit /b 1
exit /b 0

:run_one
set "log=%RUNNER_TEMP%\minimax-h3-%~1.log"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --release --test integration real_weights::%~1 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
