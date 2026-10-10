set -o pipefail
run_one() {
  local target="$1" name="$2" out
  if [[ "$target" == lib ]]; then
    out="$(KREA_REALTIME_SNAPSHOT_DIR="$KREA_REALTIME_SNAPSHOT/q4" \
      cargo test --locked --release -p mlx-gen-krea-realtime --lib \
      "$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  else
    out="$(KREA_REALTIME_SNAPSHOT_DIR="$KREA_REALTIME_SNAPSHOT/q4" \
      cargo test --locked --release -p mlx-gen-krea-realtime --test integration \
      generate_smoke::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  fi
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    return 1
  fi
}
run_one integration kv_cache_residency_at_the_production_geometry
run_one lib generate::tests::next_read_eviction_is_bit_identical_to_eager_max_window_retention
