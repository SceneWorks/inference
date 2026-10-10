# No `SA3_TEST_STEPS` and no sampler: the request omits `steps`, `sampler` and `guidance`
# so the provider resolves them, and the test asserts the resolved step count against the
# progress callbacks and proves the resolved sampler is Euler by rendering the same seed
# under both solvers.
SA3_SMALL_MUSIC_BASE_WAV_OUT="$RUNNER_TEMP/sa3-small-music-base-metal.wav" \
  cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
    --test provider connected_music_base_generation_at_its_own_defaults_is_stereo_finite_and_exact_length \
    -- --ignored --nocapture
SA3_SMALL_SFX_BASE_WAV_OUT="$RUNNER_TEMP/sa3-small-sfx-base-metal.wav" \
  cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
    --test provider connected_sfx_base_generation_at_its_own_defaults_is_stereo_finite_and_exact_length \
    -- --ignored --nocapture
shasum -a 256 "$RUNNER_TEMP/sa3-small-music-base-metal.wav" "$RUNNER_TEMP/sa3-small-sfx-base-metal.wav"
