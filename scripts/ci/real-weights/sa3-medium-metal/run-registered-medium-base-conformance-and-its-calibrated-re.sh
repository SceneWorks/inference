# sc-14546. This job already provisions `stable-audio-3-medium-base` for the identity
# gates, so the registered `stable_audio_3_medium_base` provider is validated here rather
# than in a job that would have to download the same 10.4 GB again.
#
# The render deliberately omits `steps`, `sampler` and `guidance` so the provider resolves
# this variant's own operating point — Euler / 50 / 7.0 — which is ~12.5x the example-work
# of the post-trained 8-step Pingpong default at the same duration. The sweep behind the
# enforced side-ratio floor runs at the render's own 10 s, so calibration and enforcement
# cannot drift apart.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test conformance registered_medium_base_provider_passes_full_audio_conformance -- --ignored --nocapture
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test conformance concurrent_medium_base_requests_are_deterministic_and_do_not_share_rng_state -- --ignored --nocapture
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test provider medium_base_stereo_width_floor_is_calibrated_across_prompts_and_seeds \
  -- --ignored --nocapture
SA3_MEDIUM_BASE_WAV_OUT="$RUNNER_TEMP/sa3-medium-base-metal.wav" \
  cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
    --test provider connected_medium_base_generation_at_its_own_defaults_is_stereo_finite_and_exact_length \
    -- --ignored --nocapture
shasum -a 256 "$RUNNER_TEMP/sa3-medium-base-metal.wav"
