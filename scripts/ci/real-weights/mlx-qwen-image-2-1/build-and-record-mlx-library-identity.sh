set -o pipefail
if [[ "${QWEN_IMAGE_2_1_LORA_PHASE:-full}" == current-diagnostic ]]; then
  # The current route compiles only the reviewed family-local cfg(test) selector binary.
  cargo test --locked --release -p mlx-gen-qwen-image-2-1 --lib --no-run --message-format=json 2>&1 | tee "$QWEN_IMAGE_2_1_RENDER_OUT/lib-build-messages.jsonl"
  python3.12 scripts/ci/qwen21_mlx_build_identity.py --messages "$QWEN_IMAGE_2_1_RENDER_OUT/lib-build-messages.jsonl" --test-target lib --out "$QWEN_IMAGE_2_1_RENDER_OUT/mlx-lib-test-build-identity.json"
  exit 0
fi
cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration --no-run --message-format=json 2>&1 | tee "$QWEN_IMAGE_2_1_RENDER_OUT/build-messages.jsonl"
python3.12 scripts/ci/qwen21_mlx_build_identity.py --messages "$QWEN_IMAGE_2_1_RENDER_OUT/build-messages.jsonl" --out "$QWEN_IMAGE_2_1_RENDER_OUT/mlx-build-identity.json"
if [[ "${QWEN_IMAGE_2_1_LORA_PHASE:-full}" == q4-numeric || "${QWEN_IMAGE_2_1_LORA_PHASE:-full}" == direction-protocol ]]; then
  # The diagnostic hook exists only in the family-local cfg(test) library binary.
  cargo test --locked --release -p mlx-gen-qwen-image-2-1 --lib --no-run --message-format=json 2>&1 | tee "$QWEN_IMAGE_2_1_RENDER_OUT/lib-build-messages.jsonl"
  python3.12 scripts/ci/qwen21_mlx_build_identity.py --messages "$QWEN_IMAGE_2_1_RENDER_OUT/lib-build-messages.jsonl" --test-target lib --out "$QWEN_IMAGE_2_1_RENDER_OUT/mlx-lib-test-build-identity.json"
fi
