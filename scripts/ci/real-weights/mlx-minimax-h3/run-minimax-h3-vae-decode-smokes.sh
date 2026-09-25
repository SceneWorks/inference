set -o pipefail
run_one() {
  local name="$1" out
  out="$(cargo test --locked --release -p mlx-gen-minimax-h3 --test integration \
    real_weights::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    return 1
  fi
}
run_one real_weight_decode_produces_a_plausible_video
run_one real_weight_multi_chunk_decode_blends_the_seam
run_one real_weight_audio_decode_produces_a_plausible_stereo_track
