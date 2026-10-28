set -euo pipefail
log="$RUNNER_TEMP/w4a8-mlx.log"
# `--test-threads=1`: MLX stream/device state is thread-local; a parallel libtest run is
# not the configuration these native cases were written for.
list_log="$log.list"
cargo test --locked -p mlx-gen-minimax-h3 --test integration w4a8_compare:: -- --ignored --list --color never >"$list_log" 2>&1
cat "$list_log"
test "$(awk '/: test$/{n++} END{print n+0}' "$list_log")" -eq 3
grep -Fxq "w4a8_compare::native_forward_matches_the_external_reference: test" "$list_log"
grep -Fxq "w4a8_compare::invalid_operands_are_refused_by_the_published_contract: test" "$list_log"
grep -Fxq "w4a8_compare::mixed_w4a8_and_convrot_layers_keep_their_own_arithmetic: test" "$list_log"
cargo test --locked -p mlx-gen-minimax-h3 --test integration w4a8_compare:: -- --ignored --nocapture --test-threads=1 2>&1 | tee "$log"
# `pipefail` keeps the Cargo status through `tee`. The exact summary proves how many
# bodies ran and passed (0 ignored); the bare name prefix proves identity. Do NOT anchor
# on "<name> ... ok": with `--nocapture` the test's own output can land between the
# printed name and `ok`, so a passing run can split that line.
grep -Fq "test result: ok. 3 passed; 0 failed; 0 ignored" "$log"
grep -Fq "test w4a8_compare::native_forward_matches_the_external_reference ..." "$log"
grep -Fq "test w4a8_compare::invalid_operands_are_refused_by_the_published_contract ..." "$log"
grep -Fq "test w4a8_compare::mixed_w4a8_and_convrot_layers_keep_their_own_arithmetic ..." "$log"
