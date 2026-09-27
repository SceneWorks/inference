# The floor the render step below enforces is derived from this 25-sample sweep, so the
# calibration is re-checked rather than trusted. The sweep's own defaults are 30 s /
# 8 steps — the same duration and step count the render enforces at — so calibration and
# enforcement cannot drift apart. The CUDA job runs the identical sweep before its own
# render for the same reason: a floor measured on one backend and enforced on another is
# not a calibrated gate.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test provider sfx_stereo_width_floor_is_calibrated_across_prompts_and_seeds \
  -- --ignored --nocapture
