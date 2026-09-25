cargo test --locked --release -p candle-audio-stable-audio-3 --features metal same_l_short_standalone_and_embedded_match_every_band_layer -- --ignored --nocapture
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal same_l_long_durations_match_compact_band_boundaries -- --ignored --nocapture
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal same_l_variable_stride_padding_gather_and_noise_order_match_upstream -- --ignored --nocapture
