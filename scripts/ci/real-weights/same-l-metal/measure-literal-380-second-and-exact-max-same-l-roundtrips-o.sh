cargo test --locked --release -p candle-audio-stable-audio-3 --features metal same_l_long_duration_roundtrip_resource_probe -- --ignored --nocapture
SA3_SAME_L_RESOURCE_SAMPLES=16777216 cargo test --locked --release -p candle-audio-stable-audio-3 --features metal same_l_long_duration_roundtrip_resource_probe -- --ignored --nocapture
