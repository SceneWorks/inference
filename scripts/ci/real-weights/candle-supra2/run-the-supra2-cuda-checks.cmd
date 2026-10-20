call "%VCVARS%"
set "SUPRA2_FAILED=0"
rem Weights-free parity and sampler checks on cuda:0 (each also runs its CPU leg on this box).
call :run_one tiny_dit_matches_upstream_module tiny_dit_parity::tiny_dit_matches_upstream_module || set "SUPRA2_FAILED=1"
call :run_one masked_context_rows_cannot_move_the_velocity tiny_dit_parity::masked_context_rows_cannot_move_the_velocity || set "SUPRA2_FAILED=1"
call :run_one cfg_and_plain_trajectories_follow_upstreams_euler_loop sampler::cfg_and_plain_trajectories_follow_upstreams_euler_loop || set "SUPRA2_FAILED=1"
call :run_one each_image_is_reproducible_from_its_own_seed sampler::each_image_is_reproducible_from_its_own_seed || set "SUPRA2_FAILED=1"
rem Real weights on cuda:0 against the committed torch golden.
call :run_one components_match_torch_reference real_weights::components_match_torch_reference --ignored || set "SUPRA2_FAILED=1"
call :run_one trajectories_match_torch_reference real_weights::trajectories_match_torch_reference --ignored || set "SUPRA2_FAILED=1"
call :run_one registry_renders_published_defaults real_weights::registry_renders_published_defaults --ignored || set "SUPRA2_FAILED=1"
if not "%SUPRA2_FAILED%"=="0" exit /b 1
exit /b 0

rem %1 = evidence log name, %2 = test path in the integration binary, %3 = --ignored or empty.
:run_one
set "log=%SUPRA2_OUT%\%~1.log"
cargo test --locked --release -p candle-gen-supra2 --features cuda --test integration %~2 -- %~3 --exact --nocapture > "%log%" 2>&1 || (type "%log%" & echo ::error::%~2 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~2 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
