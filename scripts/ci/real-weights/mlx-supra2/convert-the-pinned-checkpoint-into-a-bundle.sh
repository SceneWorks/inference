# The install-time converter: verifies the pinned checkpoint and both component snapshots
# (sizes + SHA-256), then writes the native bundle the provider loads (whose pin contract re-checks
# the converted denoiser and conditioning against `gen_core::supra2::PINNED_*`).
rm -rf "$SUPRA2_BUNDLE"
set -o pipefail
cargo run --locked --release -p sceneworks-gen-core --example supra2_convert -- \
  "$SUPRA2_SOURCE_SNAPSHOT/model_final_ema.pt" "$SUPRA2_BUNDLE" "$SUPRA2_T5_SNAPSHOT" "$SUPRA2_VAE_SNAPSHOT" \
  2>&1 | tee "$SUPRA2_OUT/convert.log"
cp "$SUPRA2_BUNDLE/supra2_bundle.json" "$SUPRA2_OUT/supra2_bundle.json"
