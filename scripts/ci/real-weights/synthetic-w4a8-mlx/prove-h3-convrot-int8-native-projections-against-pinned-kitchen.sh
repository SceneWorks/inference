set -euo pipefail
log="$RUNNER_TEMP/convrot-int8-mlx.log"
# sc-26691: the pinned Kitchen oracle (zero tolerance at BF16/F16) is the exactness contract the
# tiled INT8 GEMM must keep; it is the MLX twin of the synthetic-w4a8-cuda ConvRot step.
list_log="$log.list"
cargo test --locked -p mlx-gen-minimax-h3 --test integration convrot_int8:: -- --ignored --list --color never >"$list_log" 2>&1
cat "$list_log"
test "$(awk '/: test$/{n++} END{print n+0}' "$list_log")" -eq 1
grep -Fxq "convrot_int8::native_projections_match_pinned_kitchen: test" "$list_log"
cargo test --locked -p mlx-gen-minimax-h3 --test integration convrot_int8:: -- --ignored --nocapture --test-threads=1 2>&1 | tee "$log"
grep -Fq "test result: ok. 1 passed; 0 failed; 0 ignored" "$log"
grep -Fq "test convrot_int8::native_projections_match_pinned_kitchen ..." "$log"
