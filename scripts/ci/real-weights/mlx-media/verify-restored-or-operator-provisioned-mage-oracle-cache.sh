"$RUNNER_TEMP/mage-reference/bin/python" \
  scripts/release/provision_mage_oracles.py \
  --snapshot "$MAGE_SNAPSHOT" \
  --edit-snapshot "$MAGE_EDIT_SNAPSHOT" \
  --output "$MAGE_GOLDEN_DIR" \
  --verify-only
"$RUNNER_TEMP/mage-reference/bin/python" \
  scripts/release/provision_mage_oracles.py \
  --snapshot "$MAGE_SNAPSHOT" \
  --edit-snapshot "$MAGE_EDIT_SNAPSHOT" \
  --output "$MAGE_GOLDEN_DIR" \
  --verify-edit-artifact
"$RUNNER_TEMP/mage-reference/bin/python" \
  scripts/release/provision_mage_edit_variants.py \
  --gen "$MAGE_SNAPSHOT" \
  --edit "$MAGE_EDIT_SNAPSHOT" \
  --edit-base "$MAGE_EDIT_BASE_SNAPSHOT" \
  --edit-turbo "$MAGE_EDIT_TURBO_SNAPSHOT" \
  --output "$MAGE_GOLDEN_DIR" \
  --verify-only
"$RUNNER_TEMP/mage-reference/bin/python" \
  scripts/release/verify_mage_candle_oracles.py \
  --snapshot "$MAGE_SNAPSHOT" \
  --edit-snapshot "$MAGE_EDIT_SNAPSHOT" \
  --output "$MAGE_GOLDEN_DIR"
"$RUNNER_TEMP/mage-reference/bin/python" \
  scripts/release/verify_mage_candle_transfer.py \
  --gen "$MAGE_SNAPSHOT" \
  --edit "$MAGE_EDIT_SNAPSHOT" \
  --edit-base "$MAGE_EDIT_BASE_SNAPSHOT" \
  --edit-turbo "$MAGE_EDIT_TURBO_SNAPSHOT" \
  --output "$MAGE_GOLDEN_DIR"
echo "MAGE_FLOW_TE_GOLDEN=$MAGE_GOLDEN_DIR/mage_flow_te_golden.safetensors" >> "$GITHUB_ENV"
