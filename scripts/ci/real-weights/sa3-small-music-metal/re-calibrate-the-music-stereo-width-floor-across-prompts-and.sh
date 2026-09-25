# sc-14544 added a per-window-median side-ratio assertion to the shared `assert_real_audio`
# helper, which the render step below enforces on this variant too. This sweep is the
# measurement behind it, at the render's own 30 s / 8 steps.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test provider music_stereo_width_floor_is_calibrated_across_prompts_and_seeds \
  -- --ignored --nocapture
