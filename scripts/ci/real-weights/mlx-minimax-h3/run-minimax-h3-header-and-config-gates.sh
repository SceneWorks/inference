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
run_one declared_tensor_names_match_the_published_checkpoint
# sc-19445. The key-set proof above cannot see a tensor read at the wrong LEVEL: the
# shipped encoder is six levels of 128..1024 with downsamplers on 0-3 only and residual
# projections on 1/3/5 only, and the committed fixture is a four-level 32-channel toy
# whose spatial and temporal factor lists are the SAME list. This derives every one of
# the 118 encode shapes from the config and judges them against the published headers.
run_one declared_encoder_shapes_match_the_published_checkpoint
run_one declared_audio_tensor_names_match_the_published_checkpoint
run_one published_audio_configs_reproduce_the_declared_geometry
run_one te_layer_50_tap_is_exhaustive_and_the_tail_is_trimmable
run_one real_tokenizer_resolves_the_minimax_special_tokens
