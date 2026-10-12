transfer_cache_args=()
if [[ -n "${QWEN21_TRANSFER_CACHE_BASE:-}" ]]; then
  transfer_cache_args=(--source-cache-root "$QWEN21_TRANSFER_CACHE_BASE")
fi

if [[ "${QWEN_IMAGE_2_1_LORA_PHASE:-full}" == direction-protocol ]]; then
  # Immutable original style donor plus separately named FAILED edit donor. No c233 replacement.
  python3.12 scripts/ci/qwen21_adapter_imports.py --manifest scripts/ci/qwen21_adapter_imports.json --destination "$QWEN_IMAGE_2_1_RENDER_OUT/direction-imports/adapters" "${transfer_cache_args[@]}"
  echo "QWEN_IMAGE_2_1_IMPORT_MANIFEST=$QWEN_IMAGE_2_1_RENDER_OUT/direction-imports/adapters/adapter-imports-resolved.json" >> "$GITHUB_ENV"
  python3.12 -m scripts.ci.qwen21_velocity_adapter --manifest scripts/ci/qwen21_velocity_adapter.json --destination "$QWEN_IMAGE_2_1_RENDER_OUT/velocity-input/adapters"
  echo "QWEN_IMAGE_2_1_VELOCITY_MANIFEST=$QWEN_IMAGE_2_1_RENDER_OUT/velocity-input/adapters/velocity-adapter-resolved.json" >> "$GITHUB_ENV"
  exit 0
fi
if [[ "${QWEN_IMAGE_2_1_LORA_PHASE:-full}" == diagnostic || "${QWEN_IMAGE_2_1_LORA_PHASE:-full}" == q4-numeric ]]; then
  # A failed terminal candidate's completed training may inform this diagnostic only.
  # The exact adapter/receipt hashes and original source/run are checked before model work.
  python3.12 -m scripts.ci.qwen21_diagnostic_adapter --manifest scripts/ci/qwen21_diagnostic_adapter.json --destination "$QWEN_IMAGE_2_1_RENDER_OUT/diagnostic/adapters"
  echo "QWEN_IMAGE_2_1_DIAGNOSTIC_MANIFEST=$QWEN_IMAGE_2_1_RENDER_OUT/diagnostic/adapters/adapter-imports-resolved.json" >> "$GITHUB_ENV"
  if [[ "$QWEN_IMAGE_2_1_LORA_PHASE" == q4-numeric ]]; then
    python3.12 -m scripts.ci.qwen21_q4_replay --manifest scripts/ci/qwen21_q4_replay.json --destination "$QWEN_IMAGE_2_1_RENDER_OUT/q4-replay"
    echo "QWEN_IMAGE_2_1_Q4_REPLAY_DIR=$QWEN_IMAGE_2_1_RENDER_OUT/q4-replay" >> "$GITHUB_ENV"
  fi
  exit 0
fi
python3.12 scripts/ci/qwen21_adapter_imports.py --manifest scripts/ci/qwen21_adapter_imports.json --destination "$QWEN_IMAGE_2_1_RENDER_OUT/imports/adapters" "${transfer_cache_args[@]}"
echo "QWEN_IMAGE_2_1_IMPORT_MANIFEST=$QWEN_IMAGE_2_1_RENDER_OUT/imports/adapters/adapter-imports-resolved.json" >> "$GITHUB_ENV"
mkdir -p "$QWEN_IMAGE_2_1_RENDER_OUT/adapters"
cp "$QWEN_IMAGE_2_1_RENDER_OUT/imports/adapters/qwen21_t2i_lora.safetensors" "$QWEN_IMAGE_2_1_RENDER_OUT/adapters/qwen21_t2i_lora.safetensors"
