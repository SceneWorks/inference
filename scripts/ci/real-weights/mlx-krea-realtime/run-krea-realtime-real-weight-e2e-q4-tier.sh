set -o pipefail
out="$(KREA_REALTIME_SNAPSHOT_DIR="$KREA_REALTIME_SNAPSHOT/q4" \
  cargo test --locked --release -p mlx-gen-krea-realtime --test integration generate_smoke:: \
  -- --ignored --nocapture --test-threads 1 \
  --skip kv_cache_residency_at_the_production_geometry \
  --skip long_clip_coherence_under_the_bounded_window \
  --skip s18_verdict_from_accumulated_cells \
  --skip s18_kv_tier_ab_from_accumulated_cells 2>&1 | tee /dev/stderr)"
if ! grep -qE "test result: ok\. 6 passed" <<<"$out"; then
  echo "::error::the Q4 e2e step did not run exactly six passing tests — a rename, a new #[ignore] test, or a skipped arm would otherwise change this lane's coverage silently" >&2
  exit 1
fi
