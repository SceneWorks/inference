call "%VCVARS%"
set "QWEN21_FAILED=0"
call :run_one turbo_validation_prefix_kv_cache_velocity || set "QWEN21_FAILED=1"
if "%QWEN21_TURBO_TIERS%"=="1" (
  call :run_one turbo_validation_render_installed_tiers || set "QWEN21_FAILED=1"
) else (
  echo ::warning::turbo_validation_render_installed_tiers NOT RUN - no qwen_image_2_1_turbo_tier_snapshot input, so this run carries no Turbo installed-tier evidence
  echo NOT RUN: no qwen_image_2_1_turbo_tier_snapshot dispatch input> "%QWEN_IMAGE_2_1_RENDER_OUT%\turbo_validation_render_installed_tiers.NOT-RUN.txt"
)
if not "%QWEN21_FAILED%"=="0" exit /b 1
exit /b 0

:run_one
set "log=%QWEN_IMAGE_2_1_RENDER_OUT%\%~1.log"
cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda --test integration e2e_real_weights::%~1 -- --ignored --exact --nocapture > "%log%" 2>&1 || (type "%log%" & echo ::error::%~1 failed & exit /b 1)
type "%log%"
findstr /C:"test result: ok. 1 passed" "%log%" >nul || (echo ::error::%~1 did not run exactly one passing test - a rename would make this step vacuously green & exit /b 1)
exit /b 0
