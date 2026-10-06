set -o pipefail
failed=0
phase="${QWEN_IMAGE_2_1_LORA_PHASE:-full}"
case "$phase" in
  probe) export QWEN_IMAGE_2_1_PROBE_ONLY=1 QWEN_IMAGE_2_1_TRAINING_DIAGNOSTICS=1
         export QWEN_IMAGE_2_1_LORA_T2I_STEPS=2 QWEN_IMAGE_2_1_LORA_EDIT_STEPS=2 ;;
  diagnostic|q4-numeric|direction-protocol|edit|imports|full) ;;
  *) echo "unknown Qwen-Image 2.1 LoRA phase: $phase" >&2; exit 1 ;;
esac
echo "$phase" > "$QWEN_IMAGE_2_1_RENDER_OUT/selected-phase.txt"
run_one() {
  local name="$1" log="$QWEN_IMAGE_2_1_RENDER_OUT/$1.log"
  last_cargo_exit=0
  if cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
    lora_real_weights::"$name" -- --ignored --exact --nocapture --test-threads 1 2>&1 | tee "$log"; then
    :
  else
    local codes=("${PIPESTATUS[@]}")
    last_cargo_exit="${codes[0]}"
    # A capture/logging failure is a safety failure, even when Cargo returned zero.
    if [[ "${codes[1]}" != 0 ]]; then last_cargo_exit=1; fi
    echo "::error::$name failed" >&2
    return 1
  fi
  if ! grep -qE "test result: ok\. 1 passed" "$log"; then
    last_cargo_exit=1
    echo "::error::'$name' did not run exactly one passing test -- a rename would make this step vacuously green" >&2
    return 1
  fi
}
# Serial DIAGNOSTIC_ONLY style six captures and the fixed16 teacher-forced forwards.
# Retain both receipts if an independent style assertion fails; either failure keeps the phase red.
if [[ "$phase" == direction-protocol ]]; then
  if ! run_one diagnostic_reused_t2i_style_direction; then failed=1; fi
  printf '%s\n' "$last_cargo_exit" > "$QWEN_IMAGE_2_1_RENDER_OUT/direction-style.exit-code"
  # Continue ONLY on a completed safety-checked direction assertion failure (Rust exit101).
  # Missing controls, provenance/admission failures, crashes or incomplete retirement abort.
  if ! python3.12 -m scripts.ci.qwen21_direction_receipts --out "$QWEN_IMAGE_2_1_RENDER_OUT" --source "$GITHUB_SHA" --style-only --style-exit "$last_cargo_exit"; then
    echo "::error::Style stage was not safely completed; velocity stage refused" >&2; exit 1
  fi
  log="$QWEN_IMAGE_2_1_RENDER_OUT/direction-velocity.log"
  velocity_exit=0
  if cargo test --locked --release -p mlx-gen-qwen-image-2-1 --lib \
    conditioning_velocity_diagnostic::diagnostic_dense_q4_conditioning_velocity -- --ignored --exact --nocapture --test-threads 1 2>&1 | tee "$log"; then
    :
  else
    codes=("${PIPESTATUS[@]}")
    velocity_exit="${codes[0]}"
    if [[ "${codes[1]}" != 0 ]]; then velocity_exit=1; fi
    echo "::error::Dense/Q4 direction diagnostic failed" >&2; failed=1
  fi
  printf '%s\n' "$velocity_exit" > "$QWEN_IMAGE_2_1_RENDER_OUT/direction-velocity.exit-code"
  if ! grep -qE 'test result: ok\. 1 passed' "$log"; then
    echo "::error::Dense/Q4 direction diagnostic did not run exactly one passing test" >&2; failed=1
  fi
  if ! python3.12 -m scripts.ci.qwen21_direction_receipts --out "$QWEN_IMAGE_2_1_RENDER_OUT" --source "$GITHUB_SHA" --style-exit "$last_cargo_exit" --velocity-manifest "$QWEN_IMAGE_2_1_VELOCITY_MANIFEST"; then
    echo "::error::Direction diagnostic receipts are missing, partial or misattributed" >&2; failed=1
  fi
  exit "$failed"
fi
# Four fixed captures, no training or acceptance. Rust verifies replay pixels and admission.
if [[ "$phase" == q4-numeric ]]; then
  log="$QWEN_IMAGE_2_1_RENDER_OUT/q4-numeric.log"
  if ! cargo test --locked --release -p mlx-gen-qwen-image-2-1 --lib \
    q4_diagnostic::fixed_q4_numeric_diagnostic -- --ignored --exact --nocapture --test-threads 1 2>&1 | tee "$log"; then
    echo "::error::Q4 numeric diagnostic failed" >&2; exit 1
  fi
  if ! grep -qE 'test result: ok\. 1 passed' "$log"; then
    echo "::error::Q4 numeric diagnostic did not run exactly one passing test" >&2; exit 1
  fi
  exit 0
fi
# No training, full transfer campaign, stacks or public captures in this diagnostic.
# Its receipt is always DIAGNOSTIC_ONLY, even if every unchanged effect floor passes.
if [[ "$phase" == diagnostic ]]; then
  run_one diagnostic_reused_edit_adapter_semantics
  exit $?
fi
# IN ORDER: the stacking test consumes the adapters the two training tests write.
if [[ "$phase" == full || "$phase" == probe ]]; then
  run_one t2i_lora_trains_reloads_and_moves_every_tier || failed=1
fi
if [[ "$phase" != imports ]]; then
  run_one edit_lokr_trains_and_moves_two_reference_edits_every_tier || failed=1
fi
if [[ "$phase" == full || "$phase" == edit ]]; then
  run_one stacked_adapters_apply_with_independent_weights || failed=1
  export QWEN_IMAGE_2_1_CORRECTED_EDIT_ADAPTER="$QWEN_IMAGE_2_1_RENDER_OUT/adapters/qwen21_edit_lokr.safetensors"
fi
if [[ "$phase" != probe ]]; then
  run_one imported_adapters_move_t2i_and_two_reference_edit_every_tier || failed=1
fi
if [[ "$phase" != probe && -n "${QWEN_IMAGE_2_1_THIRD_PARTY_LORA:-}" ]]; then
  run_one third_party_lora_applies_strictly_and_moves_every_tier || failed=1
else
  echo "::notice::no qwen_image_2_1_third_party_lora input; the third-party adapter test was not selected"
fi
if [[ "$failed" != 0 ]]; then
  exit 1
fi
