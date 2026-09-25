call "%VCVARS%"
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test conformance registered_music_base_provider_passes_full_audio_conformance -- --ignored --nocapture || exit /b 1
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test conformance concurrent_music_base_requests_are_deterministic_and_do_not_share_rng_state -- --ignored --nocapture || exit /b 1
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test conformance registered_sfx_base_provider_passes_full_audio_conformance -- --ignored --nocapture || exit /b 1
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test conformance concurrent_sfx_base_requests_are_deterministic_and_do_not_share_rng_state -- --ignored --nocapture || exit /b 1
rem The render step below enforces both base side-ratio floors on this backend, so the
rem sweeps behind them run on this backend, at the render's own 10 s and the variants' own
rem resolved Euler / 50 / 7.0 operating point.
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test provider music_base_stereo_width_floor_is_calibrated_across_prompts_and_seeds -- --ignored --nocapture || exit /b 1
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test provider sfx_base_stereo_width_floor_is_calibrated_across_prompts_and_seeds -- --ignored --nocapture || exit /b 1
