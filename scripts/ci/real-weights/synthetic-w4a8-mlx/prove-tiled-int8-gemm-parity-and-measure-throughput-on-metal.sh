set -euo pipefail
log="$RUNNER_TEMP/int8-gemm-mlx.log"
# sc-26691: tiled INT8 GEMM vs the serial reference kernels, bit for bit (random codes, the
# K=14336 saturated-accumulator case, packed W4A8, a published-checkpoint slice), plus the
# H3-geometry microbenchmark. The real slice is a complete cached checkpoint when one exists,
# else only its four selected projections fetched by byte range from the pinned revision.
slice_log="$log.slice"
python3.12 -I scripts/ci/h3_convrot_real_slice.py --output "$RUNNER_TEMP/h3-int8-gemm-real/slice.safetensors" | tee "$slice_log"
H3_INT8_GEMM_REAL_SAFETENSORS="$(tail -n 1 "$slice_log")"
test -f "$H3_INT8_GEMM_REAL_SAFETENSORS"
export H3_INT8_GEMM_REAL_SAFETENSORS
list_log="$log.list"
cargo test --locked -p mlx-gen-minimax-h3 --test integration int8_gemm_parity:: -- --ignored --list --color never >"$list_log" 2>&1
cat "$list_log"
test "$(awk '/: test$/{n++} END{print n+0}' "$list_log")" -eq 5
grep -Fxq "int8_gemm_parity::tiled_convrot_matches_serial_reference_on_random_codes: test" "$list_log"
grep -Fxq "int8_gemm_parity::extreme_accumulation_beyond_f32_integer_range_is_exact: test" "$list_log"
grep -Fxq "int8_gemm_parity::tiled_w4a8_matches_serial_reference_on_random_packed_codes: test" "$list_log"
grep -Fxq "int8_gemm_parity::real_checkpoint_slice_matches_serial_reference: test" "$list_log"
grep -Fxq "int8_gemm_parity::microbench_h3_projection_shapes: test" "$list_log"
cargo test --locked -p mlx-gen-minimax-h3 --test integration int8_gemm_parity:: -- --ignored --nocapture --test-threads=1 2>&1 | tee "$log"
grep -Fq "test result: ok. 5 passed; 0 failed; 0 ignored" "$log"
grep -Fq "test int8_gemm_parity::tiled_convrot_matches_serial_reference_on_random_codes ..." "$log"
grep -Fq "test int8_gemm_parity::extreme_accumulation_beyond_f32_integer_range_is_exact ..." "$log"
grep -Fq "test int8_gemm_parity::tiled_w4a8_matches_serial_reference_on_random_packed_codes ..." "$log"
grep -Fq "test int8_gemm_parity::real_checkpoint_slice_matches_serial_reference ..." "$log"
grep -Fq "test int8_gemm_parity::microbench_h3_projection_shapes ..." "$log"
grep -F "INT8_GEMM_" "$log" | tee -a "$GITHUB_STEP_SUMMARY"
