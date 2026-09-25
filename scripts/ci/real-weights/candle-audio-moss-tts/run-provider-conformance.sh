# MOSS-TTSD-v0.5 multi-speaker dialogue TTS (sc-13360 AR brain + sc-13518 XY_Tokenizer
# codec). The Qwen3 backbone + 8-channel delay-pattern loop emits real, deterministic RVQ
# frames; a 2-speaker [S1]/[S2] script shapes the token stream vs a single-voice control;
# and the ported XY_Tokenizer codec renders those frames into one 24 kHz AudioTrack whose two
# speakers are acoustically distinct (measured via chatterbox_ve — cross-segment cosine below
# same-segment self-similarity), with a single-voice control and byte-identical re-synth. The
# non-English gate (task 12906) checks a Chinese prompt renders distinct real audio.
cargo test --locked --release -p candle-audio-moss-tts --test conformance moss_ttsd_emits_valid_delay_pattern_rvq_frames -- --ignored --nocapture
cargo test --locked --release -p candle-audio-moss-tts --test conformance moss_ttsd_two_speaker_script_shapes_the_token_stream -- --ignored --nocapture
cargo test --locked --release -p candle-audio-moss-tts --test conformance moss_ttsd_renders_multi_speaker_audio -- --ignored --nocapture
cargo test --locked --release -p candle-audio-moss-tts --test conformance moss_ttsd_renders_non_english -- --ignored --nocapture
