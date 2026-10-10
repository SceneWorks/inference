# The sweeps render at the variant's own resolved operating point — Euler / 50 / 7.0 — and
# at the same 10 s the render steps below enforce at, so calibration and enforcement cannot
# drift apart. `small-sfx-base` deliberately sweeps the *post-trained* SFX prompt list:
# its own shipped `demo_cond` is the music-base list, copy-pasted, and calibrating a Foley
# checkpoint on "Amen break 174 BPM" would measure the wrong distribution.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test provider music_base_stereo_width_floor_is_calibrated_across_prompts_and_seeds \
  -- --ignored --nocapture
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test provider sfx_base_stereo_width_floor_is_calibrated_across_prompts_and_seeds \
  -- --ignored --nocapture
