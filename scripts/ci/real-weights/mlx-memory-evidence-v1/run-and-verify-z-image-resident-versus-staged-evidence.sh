test "$(git rev-parse HEAD)" = "$INFERENCE_REVISION"
git diff --quiet
git diff --cached --quiet
test -z "$(git status --porcelain --untracked-files=normal)"
mkdir -p "$MEMORY_EVIDENCE_OUTPUT_DIR"
set -o pipefail
cargo test --locked --release -p mlx-gen-z-image \
  --test integration \
  sequential_residency_real_weights::sequential_bounds_peak_within_declared_decode_drift -- --ignored --exact --test-threads=1 --nocapture \
  2>&1 | tee "$MEMORY_EVIDENCE_LOG"
python3.12 scripts/release/verify_model_snapshot.py \
  --model z-image-turbo \
  --snapshot "$ZIMAGE_SNAPSHOT" \
  --inventory-output "$MEMORY_MODEL_INVENTORY_AFTER"
if ! cmp -s "$MEMORY_MODEL_INVENTORY" "$MEMORY_MODEL_INVENTORY_AFTER"; then
  echo "Z-Image snapshot content changed during the evidence run" >&2
  exit 1
fi
# sc-18149: the Sequential route's forced tiled decode (sc-13571) makes Exact parity
# structurally unsatisfiable for this A/B; the adjudicated contract is the measured
# mean-abs drift ceiling (+ p99 tail pin) the lane declares here, re-derived by the
# verifier from the bound artifacts rather than trusted from the harness. The isolator
# artifact (Resident + forced tiled decode) must be byte-identical to the staged output,
# which pins residency staging itself to exactness independently of the harness.
python3.12 scripts/release/verify_residency_ab.py \
  --model z_image_turbo \
  --resident "$MEMORY_EVIDENCE_LOG" \
  --sequential "$MEMORY_EVIDENCE_LOG" \
  --min-reduction-mib 512 \
  --expected-fingerprint z-image-mlx-independent-materialization-v4 \
  --expected-abi 3 \
  --expected-model-revision "$MEMORY_MODEL_REVISION" \
  --expected-model-inventory-sha256 "$MEMORY_MODEL_INVENTORY_SHA256" \
  --expected-parity tolerance:mean_abs_u8_subpixel:4.0 \
  --max-p99-abs-u8 13 \
  --resident-output "$MEMORY_EVIDENCE_OUTPUT_DIR/z_image_turbo-resident.rgb" \
  --sequential-output "$MEMORY_EVIDENCE_OUTPUT_DIR/z_image_turbo-staged.rgb" \
  --isolator-output "$MEMORY_EVIDENCE_OUTPUT_DIR/z_image_turbo-resident-tiled.rgb" \
  | tee "$MEMORY_EVIDENCE_OUTPUT_DIR/verifier-result.txt"
