if python3.12 scripts/release/qwen38_bonsai_terminal.py qualify-snapshots --platform macos \
  --binding BONSAI_QWEN38_SNAPSHOT=bonsai-qwen38-parent \
  --binding BONSAI_MLX_SNAPSHOT=bonsai-mlx-2bit \
  --binding BONSAI_GGUF_SNAPSHOT=bonsai-gguf \
  --binding BONSAI_BASELINE_SNAPSHOT=bonsai-qwen3vl-baseline \
  --output "$QWEN_BONSAI_OUTPUT_DIR/snapshot-metadata-before.json"; then
  STATUS=0
else
  STATUS=$?
fi
cat "$QWEN_BONSAI_OUTPUT_DIR/snapshot-metadata-before.json"
exit "$STATUS"
