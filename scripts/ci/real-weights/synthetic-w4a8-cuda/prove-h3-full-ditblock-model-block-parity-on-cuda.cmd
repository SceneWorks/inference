call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\h3-dit-parity-cuda.log"
rem Full DitBlock (attention + FFN) executions. Individual linear/AdaLN/packing
rem primitives in this module stay ordinary and are not selected by this filter.
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration dit_parity:: -- --ignored --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 14 -ExpectedNames "dit_parity::adaln_modulation_matches_the_reference|dit_parity::transformer_block_matches_the_reference|dit_parity::every_norm_epsilon_is_wired_through|dit_parity::reading_the_ffn_halves_gate_first_breaks_parity|dit_parity::the_modality_term_of_the_adaln_index_is_load_bearing|dit_parity::the_adaln_table_is_reshaped_before_it_is_chunked|dit_parity::every_block_weight_is_load_bearing|dit_parity::parity_residuals_bound_the_mutation_floor|dit_parity::weight_mapping_is_exhaustive_over_the_block_stack|dit_parity::attention_matches_the_reference_with_and_without_the_rotary|dit_parity::every_refiner_weight_is_load_bearing|dit_parity::qk_norm_runs_per_head_before_the_rotary|dit_parity::the_wrong_qkv_transform_breaks_parity|dit_parity::token_refiner_matches_the_reference"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"dit_parity::adaln_modulation_matches_the_reference: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::transformer_block_matches_the_reference: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::every_norm_epsilon_is_wired_through: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::reading_the_ffn_halves_gate_first_breaks_parity: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::the_modality_term_of_the_adaln_index_is_load_bearing: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::the_adaln_table_is_reshaped_before_it_is_chunked: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::every_block_weight_is_load_bearing: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::parity_residuals_bound_the_mutation_floor: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::weight_mapping_is_exhaustive_over_the_block_stack: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::attention_matches_the_reference_with_and_without_the_rotary: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::every_refiner_weight_is_load_bearing: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::qk_norm_runs_per_head_before_the_rotary: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::the_wrong_qkv_transform_breaks_parity: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_parity::token_refiner_matches_the_reference: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration dit_parity:: -- --ignored --nocapture --test-threads=1 >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 14 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::adaln_modulation_matches_the_reference ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::transformer_block_matches_the_reference ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::every_norm_epsilon_is_wired_through ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::reading_the_ffn_halves_gate_first_breaks_parity ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::the_modality_term_of_the_adaln_index_is_load_bearing ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::the_adaln_table_is_reshaped_before_it_is_chunked ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::every_block_weight_is_load_bearing ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::parity_residuals_bound_the_mutation_floor ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::weight_mapping_is_exhaustive_over_the_block_stack ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::attention_matches_the_reference_with_and_without_the_rotary ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::every_refiner_weight_is_load_bearing ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::qk_norm_runs_per_head_before_the_rotary ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::the_wrong_qkv_transform_breaks_parity ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_parity::token_refiner_matches_the_reference ..." "%LOG%" >nul || exit /b 1
