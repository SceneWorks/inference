set -o pipefail
run_one() {
  local name="$1" out
  out="$(cargo test --locked --release -p mlx-gen-minimax-h3 --test integration \
    staged_residency::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    return 1
  fi
}
run_one the_video_decoder_hands_off_cleanly
run_one a_held_decoder_trips_both_handoff_gates
