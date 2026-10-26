call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\h3-veda-model-cuda.log"
rem Unit-test (`--lib`) target, not the integration binary; CPU arithmetic on a complete
rem synthetic DiT + JointDit built through the production loader.
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --lib dit::model::veda_controls_tests:: -- --ignored --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 2 -ExpectedNames "dit::model::veda_controls_tests::native_controls_capture_every_mode_packed_tier_and_adapter_stack|dit::model::veda_controls_tests::native_base_and_all_published_turbo_steps_reach_attention"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"dit::model::veda_controls_tests::native_controls_capture_every_mode_packed_tier_and_adapter_stack: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit::model::veda_controls_tests::native_base_and_all_published_turbo_steps_reach_attention: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --lib dit::model::veda_controls_tests:: -- --ignored --nocapture --test-threads=1 >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 2 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test dit::model::veda_controls_tests::native_controls_capture_every_mode_packed_tier_and_adapter_stack ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit::model::veda_controls_tests::native_base_and_all_published_turbo_steps_reach_attention ..." "%LOG%" >nul || exit /b 1
