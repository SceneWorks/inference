call "%VCVARS%"
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test base_guidance -- --ignored --nocapture || exit /b 1
