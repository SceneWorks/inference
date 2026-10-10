status=0
MAGE_EDIT_GOLDEN_NAME=mage_flow_edit_golden.safetensors \
  cargo test --locked --release -p mlx-gen-mage --test integration edit_real_weights:: \
  -- --ignored --nocapture || status=$?
MAGE_EDIT_SNAPSHOT="$MAGE_EDIT_BASE_SNAPSHOT" \
  MAGE_EDIT_GOLDEN_NAME=mage_flow_edit_base_golden.safetensors \
  cargo test --locked --release -p mlx-gen-mage --test integration \
  edit_real_weights::fixed_instruction_edit_matches_the_torch_reference -- --ignored --nocapture || status=$?
MAGE_EDIT_SNAPSHOT="$MAGE_EDIT_TURBO_SNAPSHOT" \
  MAGE_EDIT_GOLDEN_NAME=mage_flow_edit_turbo_golden.safetensors \
  cargo test --locked --release -p mlx-gen-mage --test integration \
  edit_real_weights::fixed_instruction_edit_matches_the_torch_reference -- --ignored --nocapture || status=$?
exit "$status"
