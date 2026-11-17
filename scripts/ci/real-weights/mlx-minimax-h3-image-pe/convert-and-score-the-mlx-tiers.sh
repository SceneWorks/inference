# The reference is the CUDA job's native NVFP4 run (downloaded beside this job's evidence); the MLX
# tiers are scored against it exactly as the Candle tiers are. Conversion: NVFP4 -> bf16 on the
# CPU (`modelopt_nvfp4_to_bf16`), then MLX affine Q8 / Q4 (`prepare_snapshot`).
set -o pipefail
reference="$RUNNER_TEMP/minimax-pe-cuda-evidence/nvfp4.json"
test -s "$reference" || { echo "::error::the CUDA job's nvfp4.json reference is missing"; exit 1; }
cp "$reference" "$MINIMAX_PE_OUT/nvfp4.json"
bin_dir="$CARGO_TARGET_DIR/release/examples"
rm -rf "$MINIMAX_PE_TIERS"
mkdir -p "$MINIMAX_PE_TIERS"
"$bin_dir/modelopt_nvfp4_to_bf16" "$MINIMAX_PE_SNAPSHOT" "$MINIMAX_PE_TIERS/bf16" 2>&1 | tee "$MINIMAX_PE_OUT/convert-bf16.log"
for q in q8 q4; do
  "$bin_dir/prepare_snapshot" "$MINIMAX_PE_TIERS/bf16" "$MINIMAX_PE_TIERS/mlx-$q" "$q" 2>&1 | tee "$MINIMAX_PE_OUT/prepare-mlx-$q.log"
done
bin="$(cat "$RUNNER_TEMP/minimax-pe-mlx-test-binary")"
failed=0
for tier in bf16 mlx-q8 mlx-q4; do
  label="mlx-${tier#mlx-}"
  log="$MINIMAX_PE_OUT/$label.log"
  rc=0
  MINIMAX_PE_PROMPTS="$MINIMAX_PE_SNAPSHOT" MINIMAX_PE_TIER="$MINIMAX_PE_TIERS/$tier" \
    MINIMAX_PE_LABEL="$label" MINIMAX_PE_REFERENCE="$MINIMAX_PE_OUT/nvfp4.json" \
    MINIMAX_PE_OUTPUT="$MINIMAX_PE_OUT/$label.json" \
    "$bin" minimax_h3_image_pe::minimax_h3_image_pe_quality_against_nvfp4 --ignored --exact \
    --nocapture --test-threads 1 > "$log" 2>&1 || rc=$?
  cat "$log"
  if (( rc != 0 )); then
    echo "::error::$label failed (exit $rc)"
    failed=1
  elif ! grep -q "test result: ok. 1 passed" "$log"; then
    echo "::error::$label did not run exactly one passing test - a rename would make this step vacuously green"
    failed=1
  fi
done
if (( failed )); then
  exit 1
fi
