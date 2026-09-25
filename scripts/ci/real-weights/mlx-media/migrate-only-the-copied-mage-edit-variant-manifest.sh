golden_root="$(cd "$MAGE_GOLDEN_DIR" && pwd -P)"
runner_root="$(cd "$RUNNER_TEMP" && pwd -P)"
seed_root="$(cd "$MAGE_ORACLE_SEED_DIR" && pwd -P)"
if [[ "$golden_root" != "$runner_root/"* ]]; then
  echo "Mage manifest migration output must remain under RUNNER_TEMP" >&2
  exit 1
fi
if [[ "$golden_root" == "$seed_root" || \
      "$MAGE_GOLDEN_DIR/mage_edit_variants_manifest.json" -ef "$MAGE_ORACLE_SEED_DIR/mage_edit_variants_manifest.json" ]]; then
  echo "Mage manifest migration must never mutate the persistent operator seed" >&2
  exit 1
fi
"$RUNNER_TEMP/mage-reference/bin/python" scripts/release/provision_mage_edit_variants.py \
  --gen "$MAGE_SNAPSHOT" \
  --edit "$MAGE_EDIT_SNAPSHOT" \
  --edit-base "$MAGE_EDIT_BASE_SNAPSHOT" \
  --edit-turbo "$MAGE_EDIT_TURBO_SNAPSHOT" \
  --output "$MAGE_GOLDEN_DIR" \
  --migrate-reference-environment-manifest-only
transfer_manifest="$MAGE_GOLDEN_DIR/mage_candle_transfer_manifest.json"
seed_transfer_manifest="$MAGE_ORACLE_SEED_DIR/mage_candle_transfer_manifest.json"
if [[ ! -f "$transfer_manifest" || -L "$transfer_manifest" ]]; then
  echo "Mage transfer manifest migration requires an exclusive temporary copy" >&2
  exit 1
fi
if [[ "$transfer_manifest" -ef "$seed_transfer_manifest" ]]; then
  echo "Mage transfer manifest migration must never mutate the persistent operator seed" >&2
  exit 1
fi
transfer_links="$(stat -f '%l' "$transfer_manifest")"
if [[ "$transfer_links" != "1" ]]; then
  echo "Mage transfer manifest migration requires exactly one hard link" >&2
  exit 1
fi
"$RUNNER_TEMP/mage-reference/bin/python" scripts/release/verify_mage_candle_transfer.py \
  --gen "$MAGE_SNAPSHOT" \
  --edit "$MAGE_EDIT_SNAPSHOT" \
  --edit-base "$MAGE_EDIT_BASE_SNAPSHOT" \
  --edit-turbo "$MAGE_EDIT_TURBO_SNAPSHOT" \
  --output "$MAGE_GOLDEN_DIR" \
  --migrate-edit-variant-manifest-hash-only
