# Chatterbox voice embedder (sc-12844): the sc-12838 release gate. Uses the Kokoro
# snapshot above for the distinct reference voices it embeds.
cargo test --locked --release -p candle-audio-chatterbox-ve --test conformance chatterbox_ve_discriminates_speakers -- --ignored --nocapture
cargo test --locked --release -p candle-audio-chatterbox-ve --test conformance chatterbox_ve_wav_conformance -- --ignored --nocapture
# OpenVoice V2 voice conversion (sc-13223): the sc-12839 release gate. Converts a Kokoro
# source clip toward a DIFFERENT Kokoro target voice; the timbre shift is measured with
# chatterbox_ve above. sc-13233 adds a content-preservation gate: whisper_base (snapshot
# above) transcribes the converted clip and asserts its CER vs the known source script
# stays small — catching linguistic garbling the duration+timbre gate is blind to.
cargo test --locked --release -p candle-audio-openvoice --test conformance openvoice_v2_converts_toward_the_target_voice -- --ignored --nocapture
cargo test --locked --release -p candle-audio-openvoice --test conformance openvoice_v2_is_deterministic -- --ignored --nocapture
