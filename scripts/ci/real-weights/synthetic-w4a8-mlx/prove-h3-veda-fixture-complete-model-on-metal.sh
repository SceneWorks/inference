set -euo pipefail
log="$RUNNER_TEMP/h3-veda-fixture-mlx.log"
# Unit-test (`--lib`) target, not the integration binary. The device self-test is
# skipped by name: it is not a complete-model body and does not belong to this route.
list_log="$log.list"
cargo test --locked -p mlx-gen-minimax-h3 --lib veda_fixture:: -- --ignored --skip selected_kernel_actual_device_smoke --list --color never >"$list_log" 2>&1
cat "$list_log"
test "$(awk '/: test$/{n++} END{print n+0}' "$list_log")" -eq 2
grep -Fxq "veda_fixture::native_generator_learned_sparse_off_and_failure_paths: test" "$list_log"
grep -Fxq "veda_fixture::native_veda_controls_preserve_carriers_and_capture_published_schedules: test" "$list_log"
cargo test --locked -p mlx-gen-minimax-h3 --lib veda_fixture:: -- --ignored --nocapture --test-threads=1 --skip selected_kernel_actual_device_smoke 2>&1 | tee "$log"
grep -Fq "test result: ok. 2 passed; 0 failed; 0 ignored" "$log"
grep -Fq "test veda_fixture::native_generator_learned_sparse_off_and_failure_paths ..." "$log"
grep -Fq "test veda_fixture::native_veda_controls_preserve_carriers_and_capture_published_schedules ..." "$log"
