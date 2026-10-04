set -o pipefail
failed=0
phase="${QWEN_IMAGE_2_1_LORA_PHASE:-full}"
case "$phase" in
  probe) export QWEN_IMAGE_2_1_PROBE_ONLY=1 QWEN_IMAGE_2_1_TRAINING_DIAGNOSTICS=1
         export QWEN_IMAGE_2_1_LORA_T2I_STEPS=2 QWEN_IMAGE_2_1_LORA_EDIT_STEPS=2 ;;
  edit|imports|full) ;;
  *) echo "unknown Qwen-Image 2.1 LoRA phase: $phase" >&2; exit 1 ;;
esac
echo "$phase" > "$QWEN_IMAGE_2_1_RENDER_OUT/selected-phase.txt"
run_one() {
  local name="$1" log="$QWEN_IMAGE_2_1_RENDER_OUT/$1.log"
  if ! cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
    lora_real_weights::"$name" -- --ignored --exact --nocapture --test-threads 1 2>&1 | tee "$log"; then
    echo "::error::$name failed" >&2
    return 1
  fi
  if ! grep -qE "test result: ok\. 1 passed" "$log"; then
    echo "::error::'$name' did not run exactly one passing test -- a rename would make this step vacuously green" >&2
    return 1
  fi
}
# IN ORDER: the stacking test consumes the adapters the two training tests write.
if [[ "$phase" == full || "$phase" == probe ]]; then
  run_one t2i_lora_trains_reloads_and_moves_every_tier || failed=1
fi
if [[ "$phase" != imports ]]; then
  run_one edit_lokr_trains_on_two_references_and_moves_every_tier || failed=1
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
