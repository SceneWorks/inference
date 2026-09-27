call "%VCVARS%"
REM The render step below enforces the side-ratio floor on this backend, so the sweep
REM behind it runs on this backend at the render's own 30 s / 8 steps.
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test provider medium_stereo_width_floor_is_calibrated_across_prompts_and_seeds -- --ignored --nocapture || exit /b 1
