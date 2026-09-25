call "%VCVARS%"
call :run_one real_weight_decode_produces_a_plausible_video || exit /b 1
call :run_one real_weight_multi_chunk_decode_blends_the_seam || exit /b 1
call :run_one real_weight_audio_decode_produces_a_plausible_stereo_track || exit /b 1
exit /b 0

:run_one
set "log=%RUNNER_TEMP%\minimax-h3-%~1.log"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --release --test integration real_weights::%~1 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
