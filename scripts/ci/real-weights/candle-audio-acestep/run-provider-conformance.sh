# ACE-Step 1.5 music/song (sc-12842): the flow-matching DiT + Oobleck VAE producing a
# non-degenerate, rhythmic stereo mix (frame-energy variation, broadband spectrum, beat).
cargo test --locked --release -p candle-audio-acestep --test conformance acestep_conformance -- --ignored --nocapture
cargo test --locked --release -p candle-audio-acestep --test conformance acestep_music_wav_conformance -- --ignored --nocapture
