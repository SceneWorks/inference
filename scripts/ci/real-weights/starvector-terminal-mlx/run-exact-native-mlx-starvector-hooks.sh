set -o pipefail
cargo test --locked -p mlx-llm starvector_1b::tests::real_weight_provider_satisfies_shared_starvector_conformance -- --exact --ignored --nocapture 2>&1 | tee "$STARVECTOR_TERMINAL_DIR/hooks/mlx-starvector-1b.log"
grep -qE "test result: ok\\. 1 passed" "$STARVECTOR_TERMINAL_DIR/hooks/mlx-starvector-1b.log"
cargo test --locked -p mlx-llm starvector_8b::tests::real_weight_provider_satisfies_shared_starvector_conformance -- --exact --ignored --nocapture 2>&1 | tee "$STARVECTOR_TERMINAL_DIR/hooks/mlx-starvector-8b.log"
grep -qE "test result: ok\\. 1 passed" "$STARVECTOR_TERMINAL_DIR/hooks/mlx-starvector-8b.log"
