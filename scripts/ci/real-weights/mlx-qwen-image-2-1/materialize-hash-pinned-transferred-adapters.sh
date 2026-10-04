if [[ "${QWEN_IMAGE_2_1_LORA_PHASE:-full}" == diagnostic ]]; then
  # A failed terminal candidate's completed training may inform this diagnostic only.
  # The exact adapter/receipt hashes and original source/run are checked before model work.
  python3.12 -m scripts.ci.qwen21_diagnostic_adapter --manifest scripts/ci/qwen21_diagnostic_adapter.json --destination "$QWEN_IMAGE_2_1_RENDER_OUT/diagnostic/adapters"
  echo "QWEN_IMAGE_2_1_DIAGNOSTIC_MANIFEST=$QWEN_IMAGE_2_1_RENDER_OUT/diagnostic/adapters/adapter-imports-resolved.json" >> "$GITHUB_ENV"
  exit 0
fi
python3.12 scripts/ci/qwen21_adapter_imports.py --manifest scripts/ci/qwen21_adapter_imports.json --destination "$QWEN_IMAGE_2_1_RENDER_OUT/imports/adapters"
echo "QWEN_IMAGE_2_1_IMPORT_MANIFEST=$QWEN_IMAGE_2_1_RENDER_OUT/imports/adapters/adapter-imports-resolved.json" >> "$GITHUB_ENV"
mkdir -p "$QWEN_IMAGE_2_1_RENDER_OUT/adapters"
cp "$QWEN_IMAGE_2_1_RENDER_OUT/imports/adapters/qwen21_t2i_lora.safetensors" "$QWEN_IMAGE_2_1_RENDER_OUT/adapters/qwen21_t2i_lora.safetensors"
