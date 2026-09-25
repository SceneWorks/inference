SA3_TEST_DURATION=30 SA3_TEST_STEPS=8 \
SA3_SMALL_MUSIC_WAV_OUT="$RUNNER_TEMP/sa3-small-music-metal.wav" \
  cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
    --test provider connected_short_generation_is_stereo_finite_and_exact_length \
    -- --ignored --nocapture
shasum -a 256 "$RUNNER_TEMP/sa3-small-music-metal.wav"
