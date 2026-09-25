call "%VCVARS%"
REM The render below enforces SFX_SIDE_RATIO_FLOOR on this backend, so the floor is
REM calibrated on this backend, at the render's own 30 s / 8 steps. Enforcing a
REM Metal-measured margin on CUDA is the "unverified envelope" the sweep exists to
REM prevent; running the sweep here is what makes the enforcement below honest.
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test provider sfx_stereo_width_floor_is_calibrated_across_prompts_and_seeds -- --ignored --nocapture || exit /b 1
