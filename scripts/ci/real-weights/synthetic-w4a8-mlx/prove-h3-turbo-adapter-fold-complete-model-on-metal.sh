set -euo pipefail
log="$RUNNER_TEMP/h3-turbo-mlx.log"
# The five real-weight `#[ignore]`s in this module are SKIPPED by name: they need provided
# snapshots and belong to real-weights.yml, and `--ignored` alone would conscript them
# into this synthetic route. Exactly the eleven synthetic model bodies must execute.
list_log="$log.list"
cargo test --locked -p mlx-gen-minimax-h3 --test integration turbo_lora:: -- --ignored --skip published_turbo_files_resolve_to_dit_targets_at_the_measured_alphas --skip the_real_turbo_lora_folds_onto_the_real_dit --skip turbo_render_records_a_measured_clip --skip exact_h3_trainer_file_receipt --skip manual_metal_exact_h3_trainer_runtime_receipt --list --color never >"$list_log" 2>&1
cat "$list_log"
test "$(awk '/: test$/{n++} END{print n+0}' "$list_log")" -eq 11
grep -Fxq "turbo_lora::the_three_published_alphas_fold_at_one_and_one_sixteenth: test" "$list_log"
grep -Fxq "turbo_lora::the_installed_fold_keeps_the_bf16_factor_dtype: test" "$list_log"
grep -Fxq "turbo_lora::the_user_strength_multiplies_the_alpha_fold: test" "$list_log"
grep -Fxq "turbo_lora::every_target_resolves_across_both_stacks: test" "$list_log"
grep -Fxq "turbo_lora::an_unmatched_target_is_loud: test" "$list_log"
grep -Fxq "turbo_lora::a_genuine_lokr_routes_to_the_shared_seam_and_the_turbo_files_do_not: test" "$list_log"
grep -Fxq "turbo_lora::h3_trainer_metadata_cannot_route_mixed_lokr_factors_around_validation: test" "$list_log"
grep -Fxq "turbo_lora::a_file_matching_nothing_names_the_expected_key_space: test" "$list_log"
grep -Fxq "turbo_lora::every_adapter_file_is_forced_and_gpu_verified_at_its_load_boundary: test" "$list_log"
grep -Fxq "turbo_lora::a_peft_blob_rank_disagreeing_with_the_factors_is_refused: test" "$list_log"
grep -Fxq "turbo_lora::the_conversion_leaves_no_comfyui_module_behind: test" "$list_log"
cargo test --locked -p mlx-gen-minimax-h3 --test integration turbo_lora:: -- --ignored --nocapture --test-threads=1 \
  --skip published_turbo_files_resolve_to_dit_targets_at_the_measured_alphas \
  --skip the_real_turbo_lora_folds_onto_the_real_dit \
  --skip turbo_render_records_a_measured_clip \
  --skip exact_h3_trainer_file_receipt \
  --skip manual_metal_exact_h3_trainer_runtime_receipt 2>&1 | tee "$log"
grep -Fq "test result: ok. 11 passed; 0 failed; 0 ignored" "$log"
grep -Fq "test turbo_lora::the_three_published_alphas_fold_at_one_and_one_sixteenth ..." "$log"
grep -Fq "test turbo_lora::the_installed_fold_keeps_the_bf16_factor_dtype ..." "$log"
grep -Fq "test turbo_lora::the_user_strength_multiplies_the_alpha_fold ..." "$log"
grep -Fq "test turbo_lora::every_target_resolves_across_both_stacks ..." "$log"
grep -Fq "test turbo_lora::an_unmatched_target_is_loud ..." "$log"
grep -Fq "test turbo_lora::a_genuine_lokr_routes_to_the_shared_seam_and_the_turbo_files_do_not ..." "$log"
grep -Fq "test turbo_lora::h3_trainer_metadata_cannot_route_mixed_lokr_factors_around_validation ..." "$log"
grep -Fq "test turbo_lora::a_file_matching_nothing_names_the_expected_key_space ..." "$log"
grep -Fq "test turbo_lora::every_adapter_file_is_forced_and_gpu_verified_at_its_load_boundary ..." "$log"
grep -Fq "test turbo_lora::a_peft_blob_rank_disagreeing_with_the_factors_is_refused ..." "$log"
grep -Fq "test turbo_lora::the_conversion_leaves_no_comfyui_module_behind ..." "$log"
