call "%VCVARS%"
call :run_one conformance_output output_latent_to_waveform_finite_deterministic || exit /b 1
call :run_one conformance_output_44k output_44k_latent_to_waveform_finite_deterministic || exit /b 1
call :run_one mmdit_conformance mmdit_flow_shape_finite_deterministic || exit /b 1
call :run_one mmdit_conformance mmdit_sample_shape_finite_deterministic || exit /b 1
call :run_one conformance_output output_16k_matches_reference || exit /b 1
call :run_one mmdit_conformance mmdit_matches_reference || exit /b 1
call :run_one parity_reference assembly_matches_reference_waveform || exit /b 1
call :run_one parity_reference_44k assembly_44k_matches_reference_waveform || exit /b 1
call :run_one parity_reference_44k_decoder decoder_44k_matches_reference_mel_and_waveform || exit /b 1
exit /b 0

:run_one
set "log=%RUNNER_TEMP%\mmaudio-%~2.log"
cargo test --locked -p candle-audio-mmaudio --features cuda --release --test %~1 %~2 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~2 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
