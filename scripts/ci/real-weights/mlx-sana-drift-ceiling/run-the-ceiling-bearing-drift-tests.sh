test "$(git rev-parse HEAD)" = "$INFERENCE_REVISION"
git diff --quiet
git diff --cached --quiet
test -z "$(git status --porcelain --untracked-files=normal)"
mkdir -p "$SANA_DRIFT_OUTPUT_DIR"
set -o pipefail
# Two tests, run `--exact` so a rename cannot silently widen or empty the filter:
# the five-latent resample that owns the 6.0 ceiling, then the published-domain sweep
# that bounds every published edge, pins the drift/peak monotonicity, and records the
# out-of-domain overlap refusals (an admitted probe row asserts the same ceiling).
#
# Each invocation must prove it ran EXACTLY ONE passing test. A rename, a filter typo,
# or a lost `#[ignore]` all produce "0 passed" and exit 0 — a lane that is fully green
# while enforcing nothing (the sc-15520 review-round-2 guard the sibling ladder lanes
# carry). Per invocation rather than aggregate, so one invocation matching two tests
# can never cover for the other matching none.
out="$(cargo test --locked --release -p mlx-gen-sana \
  --test integration \
  memory_ladder_real_weights::the_tiled_decode_drift_is_resampled_across_production_latents -- --ignored --exact --test-threads=1 --nocapture \
  2>&1 | tee "$SANA_DRIFT_LOG" /dev/stderr)"
if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
  echo "::error::the resample test did not run exactly one passing test — a rename or a filter typo would make this step vacuously green" >&2
  exit 1
fi
out="$(cargo test --locked --release -p mlx-gen-sana \
  --test integration \
  memory_ladder_real_weights::the_published_decode_tile_domain_is_swept_against_the_whole_image_decode -- --ignored --exact --test-threads=1 --nocapture \
  2>&1 | tee -a "$SANA_DRIFT_LOG" /dev/stderr)"
if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
  echo "::error::the domain sweep did not run exactly one passing test — a rename or a filter typo would make this step vacuously green" >&2
  exit 1
fi
python3.12 scripts/release/verify_model_snapshot.py \
  --model sana-1600m-mlx \
  --snapshot "$SANA_LADDER_1600M" \
  --inventory-output "$SANA_MODEL_INVENTORY_AFTER"
if ! cmp -s "$SANA_MODEL_INVENTORY" "$SANA_MODEL_INVENTORY_AFTER"; then
  echo "SANA snapshot content changed during the drift run" >&2
  exit 1
fi
