# mlx-llm affine Q8 / Q4 from the bf16 intermediate (SW-7099), then drop the intermediate.
for q in q8 q4; do
  "$CARGO_TARGET_DIR/release/examples/prepare_snapshot" "$PUBLISH_SCRATCH/pe/bf16" "$PUBLISH_SCRATCH/pe/mlx-$q" "$q"
done
rm -rf "$PUBLISH_SCRATCH/pe/bf16"
