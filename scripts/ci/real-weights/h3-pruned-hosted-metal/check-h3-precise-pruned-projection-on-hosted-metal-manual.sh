set -euo pipefail
log="$RUNNER_TEMP/h3-pruned-hosted-metal.log"
list_log="$log.list"
cargo test --locked -p mlx-gen-minimax-h3 --test integration pruned_adaln::native_pruned_adaln_matches_pinned_reference -- --ignored --exact --list --color never >"$list_log" 2>&1
cat "$list_log"
test "$(awk '/: test$/{n++} END{print n+0}' "$list_log")" -eq 1
grep -Fxq "pruned_adaln::native_pruned_adaln_matches_pinned_reference: test" "$list_log"
cargo test --locked -p mlx-gen-minimax-h3 --test integration pruned_adaln::native_pruned_adaln_matches_pinned_reference -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$log"
grep -Fq "test result: ok. 1 passed; 0 failed; 0 ignored" "$log"
grep -Fq "test pruned_adaln::native_pruned_adaln_matches_pinned_reference ..." "$log"
grep -Fq "H3_PRUNED_ORACLE_DEVICE" "$log"
grep -Fq "H3_PRUNED_ORACLE block 0 modulation: max_abs=" "$log"
