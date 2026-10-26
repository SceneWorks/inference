call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\h3-cross-backend-cuda.log"
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration cross_backend:: -- --ignored --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 3 -ExpectedNames "cross_backend::dit_agrees_with_the_mlx_lane|cross_backend::the_dit_cross_backend_floor_is_also_the_mlx_lanes|cross_backend::a_dit_gated_ffn_half_swap_is_loud_against_the_mlx_record"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"cross_backend::dit_agrees_with_the_mlx_lane: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"cross_backend::the_dit_cross_backend_floor_is_also_the_mlx_lanes: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"cross_backend::a_dit_gated_ffn_half_swap_is_loud_against_the_mlx_record: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration cross_backend:: -- --ignored --nocapture --test-threads=1 >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 3 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::dit_agrees_with_the_mlx_lane ..." "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::the_dit_cross_backend_floor_is_also_the_mlx_lanes ..." "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::a_dit_gated_ffn_half_swap_is_loud_against_the_mlx_record ..." "%LOG%" >nul || exit /b 1
