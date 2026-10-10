if [[ -z "$MAGE_ORACLE_SEED_DIR" ]]; then
  echo "Mage oracle cache missed and MAGE_ORACLE_SEED_DIR is unset; refusing to run the multi-hour CPU producer on the shared Actions runner" >&2
  exit 1
fi
if [[ "$MAGE_ORACLE_SEED_DIR" != /* || ! -d "$MAGE_ORACLE_SEED_DIR" ]]; then
  echo "MAGE_ORACLE_SEED_DIR must be an existing absolute directory" >&2
  exit 1
fi
files=(
  mage_flow_te_golden.safetensors
  mage_flow_edit_golden.safetensors
  mage_flow_edit_base_golden.safetensors
  mage_flow_edit_turbo_golden.safetensors
  mage_flow_vae_f32_256.safetensors
  mage_flow_vae_f32_992.safetensors
  mage_flow_vae_f32_1024.safetensors
  mage_flow_vae_f32_2048.safetensors
  mage_flow_vae_f32_512x2048.safetensors
  mage_flow_vae_f32_768x1280.safetensors
  mage_flow_vae_f32_768x1152.safetensors
  mage_flow_dit_golden.safetensors
  mage_flow_e2e_golden.safetensors
  mage_flow_e2e_golden.png
  mage_flow_edit_golden.png
  mage_oracles_manifest.json
  mage_edit_oracle_manifest.json
  mage_edit_variants_manifest.json
  mage_candle_oracles_manifest.json
  mage_candle_transfer_manifest.json
)
mkdir -p "$MAGE_GOLDEN_DIR"
for file in "${files[@]}"; do
  source="$MAGE_ORACLE_SEED_DIR/$file"
  if [[ ! -f "$source" || -L "$source" ]]; then
    echo "Operator Mage oracle seed is missing a regular, non-symlink file: $source" >&2
    exit 1
  fi
  install -m 0644 "$source" "$MAGE_GOLDEN_DIR/$file"
done
edit_sha="$(shasum -a 256 "$MAGE_ORACLE_SEED_DIR/mage_edit_variants_manifest.json" | awk '{print $1}')"
transfer_sha="$(shasum -a 256 "$MAGE_ORACLE_SEED_DIR/mage_candle_transfer_manifest.json" | awk '{print $1}')"
if [[ ! "$edit_sha" =~ ^[0-9a-f]{64}$ || ! "$transfer_sha" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Could not bind the imported Mage seed manifest hashes" >&2
  exit 1
fi
{
  echo "edit-sha=$edit_sha"
  echo "transfer-sha=$transfer_sha"
  echo "imported=true"
} >> "$GITHUB_OUTPUT"
