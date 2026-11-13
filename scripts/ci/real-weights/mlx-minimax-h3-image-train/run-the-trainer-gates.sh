set -o pipefail
export MINIMAX_H3_IMAGE_TRAINED_LORA_OUT="$RUNNER_TEMP/minimax-h3-image-trained-lora"
logs="$RUNNER_TEMP/minimax-h3-image-train-logs"
mkdir -p "$logs" "$MINIMAX_H3_IMAGE_TRAINED_LORA_OUT"
failed=()
# $1 the cargo test target selector, $2 the test path inside it
run_gate() {
  local target="$1" name="$2" status=0 log="$logs/${2##*::}.log"
  MODEL_INFERENCE_CASE_FQN="$name" cargo test --locked --release -p mlx-gen-minimax-h3 \
    --features model-inference-proof $target \
    "$name" -- --exact --ignored --nocapture 2>&1 | tee "$log" >&2 || status=$?
  if ((status)) || ! python3.12 -I scripts/ci/verify_model_inference_log.py --package mlx-gen-minimax-h3 \
    --backend mlx --fqn "$name" --log "$log"; then
    echo "::error::'$name' failed or lacks its registered proof evidence (exit $status)" >&2
    failed+=("$name")
  fi
}
run_gate --lib training::real_tests::streamed_backward_matches_resident_and_finite_differences_q4
run_gate "--test integration" h3_image_train_real::h3_image_train_t2i_saves_reproduces_and_resumes_q4
run_gate "--test integration" h3_image_train_real::h3_image_train_edit_saves_and_reproduces_q4
rm -rf "${MINIMAX_H3_IMAGE_TIERS_ROOT:?}"
if ((${#failed[@]})); then
  echo "::error::failed gates: ${failed[*]}" >&2
  exit 1
fi
