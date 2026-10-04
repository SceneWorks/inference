python3.12 scripts/ci/qwen21_adapter_imports.py --manifest scripts/ci/qwen21_adapter_imports.json --destination "$QWEN_IMAGE_2_1_RENDER_OUT/imports/adapters"
echo "QWEN_IMAGE_2_1_IMPORT_MANIFEST=$QWEN_IMAGE_2_1_RENDER_OUT/imports/adapters/adapter-imports-resolved.json" >> "$GITHUB_ENV"
mkdir -p "$QWEN_IMAGE_2_1_RENDER_OUT/adapters"
cp "$QWEN_IMAGE_2_1_RENDER_OUT/imports/adapters/qwen21_t2i_lora.safetensors" "$QWEN_IMAGE_2_1_RENDER_OUT/adapters/qwen21_t2i_lora.safetensors"
