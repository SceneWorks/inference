call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\h3-adapter-cuda.log"
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration adapter_topology:: -- --ignored --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 13 -ExpectedNames "adapter_topology::native_adapter_folds_on_the_pruned_curve|adapter_topology::body_deltas_are_identical_across_full_and_pruned_topologies|adapter_topology::full_topology_adaln_targets_are_refused_specifically_per_topology|adapter_topology::real_but_unaddressable_modules_name_the_exclusion|adapter_topology::an_adaln_only_file_with_no_body_targets_is_refused_specifically|adapter_topology::an_excluded_only_file_with_no_body_targets_is_refused_specifically|adapter_topology::an_adaln_only_lokr_with_no_body_targets_is_refused_specifically|adapter_topology::turbo_schedule_coordinates_drive_the_pruned_curve_without_a_thousand_scaling|adapter_topology::a_genuine_lokr_installs_on_the_pruned_curve_with_the_same_delta|adapter_topology::the_comfyui_converted_qkv_namespace_installs_identically_on_the_pruned_curve|adapter_topology::the_delta_is_independent_of_the_body_weights_it_folds_onto|adapter_topology::the_exact_dyadic_corpus_folds_exactly_on_both_topologies|adapter_topology::the_exact_lokr_corpus_folds_exactly_on_the_pruned_curve"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"adapter_topology::native_adapter_folds_on_the_pruned_curve: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::body_deltas_are_identical_across_full_and_pruned_topologies: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::full_topology_adaln_targets_are_refused_specifically_per_topology: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::real_but_unaddressable_modules_name_the_exclusion: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::an_adaln_only_file_with_no_body_targets_is_refused_specifically: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::an_excluded_only_file_with_no_body_targets_is_refused_specifically: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::an_adaln_only_lokr_with_no_body_targets_is_refused_specifically: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::turbo_schedule_coordinates_drive_the_pruned_curve_without_a_thousand_scaling: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::a_genuine_lokr_installs_on_the_pruned_curve_with_the_same_delta: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::the_comfyui_converted_qkv_namespace_installs_identically_on_the_pruned_curve: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::the_delta_is_independent_of_the_body_weights_it_folds_onto: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::the_exact_dyadic_corpus_folds_exactly_on_both_topologies: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adapter_topology::the_exact_lokr_corpus_folds_exactly_on_the_pruned_curve: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration adapter_topology:: -- --ignored --nocapture --test-threads=1 >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 13 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::native_adapter_folds_on_the_pruned_curve ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::body_deltas_are_identical_across_full_and_pruned_topologies ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::full_topology_adaln_targets_are_refused_specifically_per_topology ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::real_but_unaddressable_modules_name_the_exclusion ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::an_adaln_only_file_with_no_body_targets_is_refused_specifically ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::an_excluded_only_file_with_no_body_targets_is_refused_specifically ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::an_adaln_only_lokr_with_no_body_targets_is_refused_specifically ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::turbo_schedule_coordinates_drive_the_pruned_curve_without_a_thousand_scaling ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::a_genuine_lokr_installs_on_the_pruned_curve_with_the_same_delta ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::the_comfyui_converted_qkv_namespace_installs_identically_on_the_pruned_curve ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::the_delta_is_independent_of_the_body_weights_it_folds_onto ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::the_exact_dyadic_corpus_folds_exactly_on_both_topologies ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adapter_topology::the_exact_lokr_corpus_folds_exactly_on_the_pruned_curve ..." "%LOG%" >nul || exit /b 1
