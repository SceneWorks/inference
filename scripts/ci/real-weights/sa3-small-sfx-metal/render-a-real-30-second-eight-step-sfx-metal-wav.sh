SA3_TEST_DURATION=30 SA3_TEST_STEPS=8 \
SA3_SMALL_SFX_WAV_OUT="$RUNNER_TEMP/sa3-small-sfx-metal.wav" \
  cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
    --test provider connected_sfx_generation_is_stereo_finite_and_exact_length \
    -- --ignored --nocapture
shasum -a 256 "$RUNNER_TEMP/sa3-small-sfx-metal.wav"
