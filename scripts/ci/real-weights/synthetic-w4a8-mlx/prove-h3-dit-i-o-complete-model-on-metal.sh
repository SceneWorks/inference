set -euo pipefail
log="$RUNNER_TEMP/h3-dit-io-mlx.log"
list_log="$log.list"
cargo test --locked -p mlx-gen-minimax-h3 --test integration dit_io:: -- --ignored --list --color never >"$list_log" 2>&1
cat "$list_log"
test "$(awk '/: test$/{n++} END{print n+0}' "$list_log")" -eq 3
grep -Fxq "dit_io::the_model_embeds_the_context_before_refining_it: test" "$list_log"
grep -Fxq "dit_io::a_resident_dit_verifies_every_source_group_before_the_schedule_consumes_it: test" "$list_log"
grep -Fxq "dit_io::the_whole_model_reproduces_the_reference_velocity: test" "$list_log"
cargo test --locked -p mlx-gen-minimax-h3 --test integration dit_io:: -- --ignored --nocapture --test-threads=1 2>&1 | tee "$log"
grep -Fq "test result: ok. 3 passed; 0 failed; 0 ignored" "$log"
grep -Fq "test dit_io::the_model_embeds_the_context_before_refining_it ..." "$log"
grep -Fq "test dit_io::a_resident_dit_verifies_every_source_group_before_the_schedule_consumes_it ..." "$log"
grep -Fq "test dit_io::the_whole_model_reproduces_the_reference_velocity ..." "$log"
