# Chatterbox clone-TTS (sc-13222): the T3 speech-token LM stage on real t3_cfg.safetensors
# weights — a valid, non-degenerate, voice-responsive token sequence, plus the honest
# VoiceEmbedding-only boundary (a full clone WAV requires ReferenceAudio; sc-13239).
cargo test --locked --release -p candle-audio-chatterbox --test conformance chatterbox_t3_produces_valid_speech_tokens -- --ignored --nocapture
cargo test --locked --release -p candle-audio-chatterbox --test conformance chatterbox_t3_responds_to_the_reference_voice -- --ignored --nocapture
cargo test --locked --release -p candle-audio-chatterbox --test conformance chatterbox_generate_requires_reference_audio_for_a_full_clone -- --ignored --nocapture
# s3tokenizer (sc-13235) + T3 prompt-token wiring: the Whisper-v2 FSMN encoder + FSQ head
# tokenizes a reference at ≈25 Hz into valid FSQ codes, deterministically and content-
# responsively, and fills T3's reference conditioning prompt (previously orphaned — sc-13394).
cargo test --locked --release -p candle-audio-chatterbox --test conformance s3tokenizer_encodes_a_reference_at_25hz -- --ignored --nocapture
cargo test --locked --release -p candle-audio-chatterbox --test conformance chatterbox_reference_audio_fills_the_t3_prompt_tokens -- --ignored --nocapture
# s3tokenizer long-audio (>30s) sliding-window segmentation + merge_tokenized_segments
# (sc-13380): 30 s windows hopped by 26 s (4 s overlap), each tokenized independently and
# stitched so the overlap is counted once — a real >30 s clip tokenizes at 25 Hz across the
# seams and matches single-pass tokens over the non-boundary interior (continuity).
cargo test --locked --release -p candle-audio-chatterbox --test conformance s3tokenizer_windows_audio_longer_than_30s -- --ignored --nocapture
# CAMPPlus speaker encoder (sc-13236): the D-TDNN x-vector network derives DISCRIMINATIVE
# 192-d / 80-d flow speaker embeddings on real speaker_encoder.* weights (previously
# orphaned — sc-13394).
cargo test --locked --release -p candle-audio-chatterbox --test conformance campplus_derives_discriminative_x_vectors -- --ignored --nocapture
# S3Gen flow token->mel decoder (sc-13237): the UpsampleConformerEncoder + CausalConditionalCFM
# + ConditionalDecoder on real flow.* weights render a sane 80-bin log-mel (shape [80, ~2*n_tokens],
# finite, non-degenerate range) from real conditioning (s3tokenizer tokens + CAMPPlus 80-d spk +
# 24 kHz prompt mel) and are deterministic. Needs the full CHATTERBOX_SNAPSHOT (s3gen.safetensors).
cargo test --locked --release -p candle-audio-chatterbox --test conformance flow_synthesizes_a_sane_mel_from_speech_tokens -- --ignored --nocapture
# S3Gen HiFTNet vocoder (sc-13238): the NSF/iSTFT mel->waveform generator vocodes a real 24 kHz
# log-mel into a non-silent, finite, 480-samples/frame waveform, deterministically.
cargo test --locked --release -p candle-audio-chatterbox --test conformance hift_vocodes_a_real_mel_to_nonsilent_waveform -- --ignored --nocapture
# PerTh provenance watermarker (sc-13240): the natively-ported implicit watermarker on real
# weights (MIT; resolved from the SceneWorks/perth-implicit hub pin — sc-13443) embeds a
# RECOVERABLE (≈100% detection, clean≈0) and IMPERCEPTIBLE (psychoacoustically masked;
# SNR ≈16 dB) watermark, at both the model's native 32 kHz rate and Chatterbox's 24 kHz
# output path. Wired into generate() (sc-13239).
cargo test --locked --release -p candle-audio-chatterbox --test conformance perth_watermark_roundtrips_and_is_imperceptible -- --ignored --nocapture
# THE sc-12838 clone-WAV DoD (sc-13239): the epic-releasing gate. The full registry path
# (generate()) renders a real 24 kHz cloned-voice WAV from a reference clip + text that is
# non-silent, finite, token-proportional, speech-shaped, watermarked, AND voice-similar to
# the reference (cos(ve(out),ve(ref)) > cos(ve(out),ve(control)) by a material margin — it
# FAILS if the clone ignores the reference). Needs the full CHATTERBOX_SNAPSHOT; PerTh
# resolves from the SceneWorks/perth-implicit hub pin.
CHATTERBOX_WAV_OUT="$RUNNER_TEMP/chatterbox_clone_demo.wav" cargo test --locked --release -p candle-audio-chatterbox --test conformance chatterbox_clones_a_reference_voice_end_to_end -- --ignored --nocapture
