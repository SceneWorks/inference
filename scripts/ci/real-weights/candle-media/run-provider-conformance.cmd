call "%VCVARS%"
set "MAGE_CONFORMANCE_FAILED=0"
cargo test --locked -p candle-gen-z-image --features cuda --test integration conformance::z_image_conformance -- --ignored --nocapture || set "MAGE_CONFORMANCE_FAILED=1"
cargo test --locked --release -p candle-gen-mage --features cuda --test integration real_parity:: -- --ignored --nocapture || set "MAGE_CONFORMANCE_FAILED=1"
cargo test --locked --release -p candle-gen-mage --features cuda --test integration cuda_1024:: -- --ignored --nocapture || set "MAGE_CONFORMANCE_FAILED=1"
cargo test --locked --release -p candle-gen-mage --features cuda --test integration edit_parity:: -- --ignored --nocapture || set "MAGE_CONFORMANCE_FAILED=1"
for %%T in (q4 q8 bf16) do (
  set "MAGE_QUANT_TIER=%%T"
  cargo test --locked --release -p candle-gen-mage --features cuda --test integration quant_real_weights::registered_tier_matches_independent_oracle_and_vram_budget -- --ignored --nocapture || set "MAGE_CONFORMANCE_FAILED=1"
)
if not "%MAGE_CONFORMANCE_FAILED%"=="0" exit /b 1
