set -euo pipefail
log="$RUNNER_TEMP/h3-adaln-evict-mlx.log"
list_log="$log.list"
cargo test --locked -p mlx-gen-minimax-h3 --test integration adaln_evict_memory:: -- --ignored --list --color never >"$list_log" 2>&1
cat "$list_log"
test "$(awk '/: test$/{n++} END{print n+0}' "$list_log")" -eq 1
grep -Fxq "adaln_evict_memory::the_evicted_adaln_weights_leave_device_memory: test" "$list_log"
cargo test --locked -p mlx-gen-minimax-h3 --test integration adaln_evict_memory:: -- --ignored --nocapture --test-threads=1 2>&1 | tee "$log"
grep -Fq "test result: ok. 1 passed; 0 failed; 0 ignored" "$log"
grep -Fq "test adaln_evict_memory::the_evicted_adaln_weights_leave_device_memory ..." "$log"
