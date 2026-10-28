call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\h3-pruned-cuda.log"
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration pruned_adaln:: -- --ignored --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 7 -ExpectedNames "pruned_adaln::native_pruned_adaln_matches_pinned_reference|pruned_adaln::native_blocks_and_final_outputs_match_the_pinned_reference|pruned_adaln::the_curve_cache_keys_on_the_schedule_and_the_basis|pruned_adaln::a_mixed_basis_block_stack_is_refused_before_any_projection|pruned_adaln::the_stock_path_is_untouched_and_refuses_curve_coordinates|pruned_adaln::load_from_dir_builds_the_pruned_model_with_its_identity_at_stored_precision|pruned_adaln::manual_unit_patch_and_mixed_reference_full_forward_matches_resident_control"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"pruned_adaln::native_pruned_adaln_matches_pinned_reference: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"pruned_adaln::native_blocks_and_final_outputs_match_the_pinned_reference: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"pruned_adaln::the_curve_cache_keys_on_the_schedule_and_the_basis: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"pruned_adaln::a_mixed_basis_block_stack_is_refused_before_any_projection: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"pruned_adaln::the_stock_path_is_untouched_and_refuses_curve_coordinates: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"pruned_adaln::load_from_dir_builds_the_pruned_model_with_its_identity_at_stored_precision: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"pruned_adaln::manual_unit_patch_and_mixed_reference_full_forward_matches_resident_control: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration pruned_adaln:: -- --ignored --nocapture --test-threads=1 >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 7 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test pruned_adaln::native_pruned_adaln_matches_pinned_reference ..." "%LOG%" >nul || exit /b 1
findstr /C:"test pruned_adaln::native_blocks_and_final_outputs_match_the_pinned_reference ..." "%LOG%" >nul || exit /b 1
findstr /C:"test pruned_adaln::the_curve_cache_keys_on_the_schedule_and_the_basis ..." "%LOG%" >nul || exit /b 1
findstr /C:"test pruned_adaln::a_mixed_basis_block_stack_is_refused_before_any_projection ..." "%LOG%" >nul || exit /b 1
findstr /C:"test pruned_adaln::the_stock_path_is_untouched_and_refuses_curve_coordinates ..." "%LOG%" >nul || exit /b 1
findstr /C:"test pruned_adaln::load_from_dir_builds_the_pruned_model_with_its_identity_at_stored_precision ..." "%LOG%" >nul || exit /b 1
findstr /C:"test pruned_adaln::manual_unit_patch_and_mixed_reference_full_forward_matches_resident_control ..." "%LOG%" >nul || exit /b 1
findstr /C:"H3_GEOMETRY_AUDIO_FULL_FORWARD backend=cuda cases=11" "%LOG%" >nul || exit /b 1
