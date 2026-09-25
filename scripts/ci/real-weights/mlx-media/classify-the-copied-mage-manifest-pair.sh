if "$RUNNER_TEMP/mage-reference/bin/python" \
    scripts/release/provision_mage_edit_variants.py \
    --gen "$MAGE_SNAPSHOT" \
    --edit "$MAGE_EDIT_SNAPSHOT" \
    --edit-base "$MAGE_EDIT_BASE_SNAPSHOT" \
    --edit-turbo "$MAGE_EDIT_TURBO_SNAPSHOT" \
    --output "$MAGE_GOLDEN_DIR" \
    --verify-only && \
  "$RUNNER_TEMP/mage-reference/bin/python" \
    scripts/release/verify_mage_candle_transfer.py \
    --gen "$MAGE_SNAPSHOT" \
    --edit "$MAGE_EDIT_SNAPSHOT" \
    --edit-base "$MAGE_EDIT_BASE_SNAPSHOT" \
    --edit-turbo "$MAGE_EDIT_TURBO_SNAPSHOT" \
    --output "$MAGE_GOLDEN_DIR"; then
  echo "current=true" >> "$GITHUB_OUTPUT"
else
  # The migration command below accepts only the one exact legacy population. Any
  # unrelated validation failure therefore remains fail-closed rather than being blessed.
  echo "current=false" >> "$GITHUB_OUTPUT"
fi
