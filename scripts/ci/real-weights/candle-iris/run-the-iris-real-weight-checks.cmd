call "%VCVARS%"
set "IRIS_FAILED=0"
call :run_one real_tokenizer_matches_the_pinned_ids real_weights::real_tokenizer_matches_the_pinned_ids --ignored || set "IRIS_FAILED=1"
call :run_one real_conditioning_and_forward_match_upstream real_weights::real_conditioning_and_forward_match_upstream --ignored || set "IRIS_FAILED=1"
call :run_one bf16_compute_stays_within_bf16_distance_of_fp32 dit_parity::bf16_compute_stays_within_bf16_distance_of_fp32 || set "IRIS_FAILED=1"
call :run_one windows_masks_and_layer_states_match_upstream text_parity::windows_masks_and_layer_states_match_upstream || set "IRIS_FAILED=1"
call :run_one provider_renders_a_real_image real_weights::provider_renders_a_real_image --ignored || set "IRIS_FAILED=1"
if not "%IRIS_FAILED%"=="0" exit /b 1
exit /b 0

rem %1 = evidence log name, %2 = test path in the integration binary, %3 = --ignored or empty.
:run_one
set "log=%IRIS_OUT%\%~1.log"
cargo test --locked --release -p candle-gen-iris --features cuda --test integration %~2 -- %~3 --exact --nocapture > "%log%" 2>&1 || (type "%log%" & echo ::error::%~2 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~2 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
