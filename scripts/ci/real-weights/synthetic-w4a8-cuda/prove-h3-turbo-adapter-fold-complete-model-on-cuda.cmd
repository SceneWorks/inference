call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\h3-turbo-cuda.log"
rem The three real-weight `#[ignore]`s in this module are SKIPPED by name: they need provided
rem snapshots and belong to real-weights.yml, and `--ignored` alone would conscript them.
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration turbo_lora:: -- --ignored --skip manual_cuda_fabricated_lokr_install_receipt --skip exact_h3_trainer_file_receipt --skip manual_cuda_exact_h3_trainer_runtime_receipt --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 14 -ExpectedNames "turbo_lora::every_enumerated_target_resolves_through_the_model_tree|turbo_lora::adaln_and_the_norms_are_not_addressable|turbo_lora::the_install_folds_at_the_published_strength|turbo_lora::the_installed_fold_keeps_the_factor_dtype|turbo_lora::a_file_that_folds_nothing_is_refused_per_spec_not_in_aggregate|turbo_lora::an_unmatched_target_is_surfaced_by_name|turbo_lora::lokr_matches_an_independent_dense_kron_and_rejects_mutant_results|turbo_lora::fused_qkv_and_mlp_lokr_cover_the_complete_block_surface_numerically|turbo_lora::lokr_rejects_unknown_partial_unmatched_and_false_kind_claims|turbo_lora::direct_lokr_reaches_every_trunk_and_token_refiner_adapter_target|turbo_lora::a_peft_blob_supplies_the_alpha_and_its_rank_is_only_a_cross_check|turbo_lora::the_conversion_leaves_no_comfyui_module_behind|turbo_lora::clay_raw_refiner_header_conversion_preserves_factors_alpha_and_strict_unknowns|turbo_lora::the_render_seam_folds_the_staged_adapter_onto_the_dit"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"turbo_lora::every_enumerated_target_resolves_through_the_model_tree: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::adaln_and_the_norms_are_not_addressable: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::the_install_folds_at_the_published_strength: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::the_installed_fold_keeps_the_factor_dtype: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::a_file_that_folds_nothing_is_refused_per_spec_not_in_aggregate: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::an_unmatched_target_is_surfaced_by_name: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::lokr_matches_an_independent_dense_kron_and_rejects_mutant_results: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::fused_qkv_and_mlp_lokr_cover_the_complete_block_surface_numerically: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::lokr_rejects_unknown_partial_unmatched_and_false_kind_claims: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::direct_lokr_reaches_every_trunk_and_token_refiner_adapter_target: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::a_peft_blob_supplies_the_alpha_and_its_rank_is_only_a_cross_check: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::the_conversion_leaves_no_comfyui_module_behind: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::clay_raw_refiner_header_conversion_preserves_factors_alpha_and_strict_unknowns: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"turbo_lora::the_render_seam_folds_the_staged_adapter_onto_the_dit: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration turbo_lora:: -- --ignored --nocapture --test-threads=1 --skip manual_cuda_fabricated_lokr_install_receipt --skip exact_h3_trainer_file_receipt --skip manual_cuda_exact_h3_trainer_runtime_receipt >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 14 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::every_enumerated_target_resolves_through_the_model_tree ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::adaln_and_the_norms_are_not_addressable ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::the_install_folds_at_the_published_strength ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::the_installed_fold_keeps_the_factor_dtype ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::a_file_that_folds_nothing_is_refused_per_spec_not_in_aggregate ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::an_unmatched_target_is_surfaced_by_name ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::lokr_matches_an_independent_dense_kron_and_rejects_mutant_results ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::fused_qkv_and_mlp_lokr_cover_the_complete_block_surface_numerically ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::lokr_rejects_unknown_partial_unmatched_and_false_kind_claims ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::direct_lokr_reaches_every_trunk_and_token_refiner_adapter_target ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::a_peft_blob_supplies_the_alpha_and_its_rank_is_only_a_cross_check ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::the_conversion_leaves_no_comfyui_module_behind ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::clay_raw_refiner_header_conversion_preserves_factors_alpha_and_strict_unknowns ..." "%LOG%" >nul || exit /b 1
findstr /C:"test turbo_lora::the_render_seam_folds_the_staged_adapter_onto_the_dit ..." "%LOG%" >nul || exit /b 1
