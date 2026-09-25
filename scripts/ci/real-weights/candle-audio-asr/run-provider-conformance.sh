# Whisper ASR (sc-12850): the first real Transcriber (audio→text, the Captioner-analog).
# Kokoro TTS → Whisper ASR round-trip — known text synthesized by kokoro_82m (snapshot
# above) transcribes back within a small CER, with monotonic segment timestamps.
cargo test --locked --release -p candle-audio-whisper --test conformance whisper_transcribes_kokoro_roundtrip_within_cer -- --ignored --nocapture
cargo test --locked --release -p candle-audio-whisper --test conformance whisper_greedy_transcription_is_deterministic -- --ignored --nocapture
# LAION CLAP audio embedder (sc-12851): the first real AudioEmbedder (semantic audio-text
# joint-space retrieval). Embeds a set of real clips spanning categories (a Kokoro speech
# clip from the snapshot above, a tone, and white noise) and asserts a TEXT query ranks its
# matching clip highest by cosine — the cross-modal ranking DoD — plus embedding determinism.
cargo test --locked --release -p candle-audio-clap --test conformance cross_modal_query_ranks_matching_clip_highest -- --ignored --nocapture
cargo test --locked --release -p candle-audio-clap --test conformance embedding_is_deterministic -- --ignored --nocapture
