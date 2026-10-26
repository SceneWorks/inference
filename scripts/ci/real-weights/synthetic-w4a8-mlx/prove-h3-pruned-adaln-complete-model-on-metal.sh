set -euo pipefail
log="$RUNNER_TEMP/h3-pruned-mlx.log"
list_log="$log.list"
cargo test --locked -p mlx-gen-minimax-h3 --test integration pruned_adaln:: -- --ignored --list --color never >"$list_log" 2>&1
cat "$list_log"
test "$(awk '/: test$/{n++} END{print n+0}' "$list_log")" -eq 4
grep -Fxq "pruned_adaln::native_pruned_adaln_matches_pinned_reference: test" "$list_log"
grep -Fxq "pruned_adaln::full_model_live_coordinates_match_cache_and_refuse_foreign_basis: test" "$list_log"
grep -Fxq "pruned_adaln::full_model_detects_pruned_and_checks_pruning_identity: test" "$list_log"
grep -Fxq "pruned_adaln::manual_unit_patch_and_mixed_reference_full_forward_matches_resident_control: test" "$list_log"
cargo test --locked -p mlx-gen-minimax-h3 --test integration pruned_adaln:: -- --ignored --nocapture --test-threads=1 2>&1 | tee "$log"
grep -Fq "test result: ok. 4 passed; 0 failed; 0 ignored" "$log"
grep -Fq "test pruned_adaln::native_pruned_adaln_matches_pinned_reference ..." "$log"
grep -Fq "test pruned_adaln::full_model_live_coordinates_match_cache_and_refuse_foreign_basis ..." "$log"
grep -Fq "test pruned_adaln::full_model_detects_pruned_and_checks_pruning_identity ..." "$log"
grep -Fq "test pruned_adaln::manual_unit_patch_and_mixed_reference_full_forward_matches_resident_control ..." "$log"
grep -Fq "H3_GEOMETRY_AUDIO_FULL_FORWARD backend=metal cases=11" "$log"
