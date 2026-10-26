set -euo pipefail
log="$RUNNER_TEMP/h3-bounded-mlx.log"
list_log="$log.list"
cargo test --locked -p mlx-gen-minimax-h3 --test integration bounded_attention:: -- --ignored --list --color never >"$list_log" 2>&1
cat "$list_log"
test "$(awk '/: test$/{n++} END{print n+0}' "$list_log")" -eq 1
grep -Fxq "bounded_attention::a_bounded_plan_threads_the_whole_block_stack: test" "$list_log"
cargo test --locked -p mlx-gen-minimax-h3 --test integration bounded_attention:: -- --ignored --nocapture --test-threads=1 2>&1 | tee "$log"
grep -Fq "test result: ok. 1 passed; 0 failed; 0 ignored" "$log"
grep -Fq "test bounded_attention::a_bounded_plan_threads_the_whole_block_stack ..." "$log"
