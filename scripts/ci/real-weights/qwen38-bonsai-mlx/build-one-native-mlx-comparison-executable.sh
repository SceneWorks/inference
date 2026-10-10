set -o pipefail
cargo test --locked --release -p mlx-llm --test integration --no-run --message-format=json 2>"$QWEN_BONSAI_OUTPUT_DIR/build.stderr" | tee "$QWEN_BONSAI_OUTPUT_DIR/build.jsonl" | python3.12 scripts/release/qwen38_bonsai_terminal.py resolve-binary --target integration > "$QWEN_BONSAI_OUTPUT_DIR/binary.txt"
test -x "$(cat "$QWEN_BONSAI_OUTPUT_DIR/binary.txt")"
