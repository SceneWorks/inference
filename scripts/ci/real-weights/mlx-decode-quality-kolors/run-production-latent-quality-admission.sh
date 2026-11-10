set -euo pipefail
# Stamp each receipt with the closure derived from this checkout (sc-19728). Recorded for
# forensics, not matched against the build: measurements stand until they are remeasured.
DECODE_QUALITY_IMPLEMENTATION_FINGERPRINT="$(python3.12 scripts/ci/decode_quality_implementation_fingerprint.py)"
export DECODE_QUALITY_IMPLEMENTATION_FINGERPRINT
log="$RUNNER_TEMP/decode-quality-v2-kolors-kolors-${{ matrix.candidate.geometry }}.log"
out="$(cargo test --locked --release -p mlx-gen-kolors \
  --test integration decode_quality_admission::production_latent_quality_admission \
  -- --exact --ignored --test-threads=1 --nocapture 2>&1 | tee "$log" /dev/stderr)"
grep -qE "test result: ok\. 1 passed" <<<"$out"
python3.12 scripts/ci/collect_decode_quality_admission.py \
  --input "$log" --expected-policy-count 1 --expected-fixture-count 5 \
  --output "$RUNNER_TEMP/decode-quality-v2-kolors-kolors-${{ matrix.candidate.geometry }}-policies.json"
