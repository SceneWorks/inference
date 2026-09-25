call "%VCVARS%"
REM Same reason as the SFX CUDA job: the render below enforces the per-window-median
REM side-ratio assertion on this backend, so the measurement behind it is taken on this
REM backend, at the render's own 30 s / 8 steps.
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test provider music_stereo_width_floor_is_calibrated_across_prompts_and_seeds -- --ignored --nocapture || exit /b 1
