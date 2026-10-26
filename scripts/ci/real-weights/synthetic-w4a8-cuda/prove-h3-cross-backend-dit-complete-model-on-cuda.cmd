call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\h3-cross-backend-cuda.log"
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration cross_backend:: -- --ignored --skip joint_denoise_agrees_with_the_mlx_lane --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 9 -ExpectedNames "cross_backend::dit_agrees_with_the_mlx_lane|cross_backend::the_dit_cross_backend_floor_is_also_the_mlx_lanes|cross_backend::a_dit_gated_ffn_half_swap_is_loud_against_the_mlx_record|cross_backend::video_decode_agrees_with_the_mlx_lane|cross_backend::video_encode_agrees_with_the_mlx_lane|cross_backend::audio_decode_agrees_with_the_mlx_lane|cross_backend::the_cross_backend_floor_is_the_mlx_lanes_own_reduced_precision|cross_backend::the_cross_backend_gate_cannot_resolve_an_eps_misplacement|cross_backend::a_gated_ffn_half_swap_is_loud_against_the_mlx_record"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"cross_backend::dit_agrees_with_the_mlx_lane: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"cross_backend::the_dit_cross_backend_floor_is_also_the_mlx_lanes: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"cross_backend::a_dit_gated_ffn_half_swap_is_loud_against_the_mlx_record: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"cross_backend::video_decode_agrees_with_the_mlx_lane: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"cross_backend::video_encode_agrees_with_the_mlx_lane: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"cross_backend::audio_decode_agrees_with_the_mlx_lane: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"cross_backend::the_cross_backend_floor_is_the_mlx_lanes_own_reduced_precision: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"cross_backend::the_cross_backend_gate_cannot_resolve_an_eps_misplacement: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"cross_backend::a_gated_ffn_half_swap_is_loud_against_the_mlx_record: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration cross_backend:: -- --ignored --skip joint_denoise_agrees_with_the_mlx_lane --nocapture --test-threads=1 >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 9 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::dit_agrees_with_the_mlx_lane ..." "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::the_dit_cross_backend_floor_is_also_the_mlx_lanes ..." "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::a_dit_gated_ffn_half_swap_is_loud_against_the_mlx_record ..." "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::video_decode_agrees_with_the_mlx_lane ..." "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::video_encode_agrees_with_the_mlx_lane ..." "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::audio_decode_agrees_with_the_mlx_lane ..." "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::the_cross_backend_floor_is_the_mlx_lanes_own_reduced_precision ..." "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::the_cross_backend_gate_cannot_resolve_an_eps_misplacement ..." "%LOG%" >nul || exit /b 1
findstr /C:"test cross_backend::a_gated_ffn_half_swap_is_loud_against_the_mlx_record ..." "%LOG%" >nul || exit /b 1
