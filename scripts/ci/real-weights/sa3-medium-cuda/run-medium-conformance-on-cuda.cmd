call "%VCVARS%"
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test conformance registered_medium_provider_passes_full_audio_conformance -- --ignored --nocapture || exit /b 1
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test conformance concurrent_medium_requests_are_deterministic_and_do_not_share_rng_state -- --ignored --nocapture || exit /b 1
