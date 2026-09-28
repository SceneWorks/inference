# Medium's floor is derived from a 25-sample sweep that deliberately mixes music and SFX
# prompts, because medium is the only SA3 checkpoint registered for both. The sweep's
# defaults are the render's own 30 s / 8 steps.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test provider medium_stereo_width_floor_is_calibrated_across_prompts_and_seeds \
  -- --ignored --nocapture
