//! Real-weights demo for the Chatterbox voice embedder (sc-12844): synthesize three reference
//! clips with Kokoro (two distinct voices), embed each with the real `ve.safetensors`, and report
//! the same-speaker vs different-speaker cosine similarities plus the discriminative margin. Writes
//! the reference clips as WAVs for manual listening.
//!
//! Run (downloads Chatterbox `ve` ≈5.7 MB + uses the cached Kokoro snapshot):
//! ```sh
//! CHATTERBOX_VE_DEMO_OUT=/tmp/scratch cargo run --release \
//!     -p candle-audio-chatterbox-ve --example embed_demo
//! ```

use candle_audio::gen_core::{AudioParams, GenerationOutput, GenerationRequest, LoadSpec};
use candle_audio::wav::write_wav_pcm16;
use candle_audio_chatterbox_ve as ve;

fn kokoro_clip(text: &str, voice: &str) -> candle_audio::gen_core::AudioTrack {
    use candle_audio::gen_core::WeightsSource;
    let spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
        std::env::var("KOKORO_SNAPSHOT")
            .expect("set KOKORO_SNAPSHOT to a hexgrad/Kokoro-82M snapshot dir"),
    )));
    let gen = candle_audio_kokoro::load(&spec).expect("load kokoro");
    let req = GenerationRequest {
        prompt: text.to_string(),
        audio: Some(AudioParams {
            voice: Some(voice.to_string()),
            language: Some("en".to_string()),
            sample_rate: Some(24_000),
            ..Default::default()
        }),
        ..Default::default()
    };
    match gen.generate(&req, &mut |_| {}).expect("kokoro generate") {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected audio, got {other:?}"),
    }
}

fn main() {
    let out_dir = std::env::var("CHATTERBOX_VE_DEMO_OUT").unwrap_or_else(|_| ".".to_string());

    // Two distinct Kokoro voices → three reference clips (A said twice, B once).
    let a1 = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "af_heart",
    );
    let a2 = kokoro_clip(
        "She sells seashells by the seashore on a bright summer morning.",
        "af_heart",
    );
    let b1 = kokoro_clip(
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "am_michael",
    );

    // Honestly named: these are the Kokoro *reference* clips the embedder consumes, NOT a cloned
    // or converted output (this slice ships the embedder, not the clone generator / converter).
    let dir = std::path::Path::new(&out_dir);
    write_wav_pcm16(
        dir.join("chatterbox-ve-reference-af_heart.wav").as_path(),
        &a1,
    )
    .expect("write reference wav");
    write_wav_pcm16(
        dir.join("chatterbox-ve-control-am_michael.wav").as_path(),
        &b1,
    )
    .expect("write control wav");

    // Embed all three with the real Chatterbox voice encoder.
    let vespec = LoadSpec::new(candle_audio::gen_core::WeightsSource::File(
        std::path::PathBuf::from(std::env::var("CHATTERBOX_VE_SNAPSHOT").expect(
            "set CHATTERBOX_VE_SNAPSHOT to a ResembleAI/chatterbox snapshot dir holding ve.safetensors",
        ))
        .join(ve::WEIGHTS_FILE),
    ));
    let embedder = ve::load(&vespec).expect("load chatterbox_ve");
    let ea1 = embedder.embed(&a1).expect("embed a1");
    let ea2 = embedder.embed(&a2).expect("embed a2");
    let eb1 = embedder.embed(&b1).expect("embed b1");

    let same = ve::cosine_similarity(&ea1, &ea2);
    let cross1 = ve::cosine_similarity(&ea1, &eb1);
    let cross2 = ve::cosine_similarity(&ea2, &eb1);
    let cross = 0.5 * (cross1 + cross2);

    println!("embedding_dim               = {}", ea1.len());
    println!(
        "A1 duration (s)             = {:.2}",
        a1.samples.len() as f32 / 24_000.0
    );
    println!("same-speaker  cos(A1,A2)    = {same:.4}");
    println!("diff-speaker  cos(A1,B1)    = {cross1:.4}");
    println!("diff-speaker  cos(A2,B1)    = {cross2:.4}");
    println!("mean diff-speaker cos       = {cross:.4}");
    println!("discriminative margin       = {:.4}", same - cross);
    assert!(
        same > cross + 0.05,
        "voice encoder must place same-speaker clips closer than different-speaker \
         (same={same:.4}, cross={cross:.4})"
    );
    println!("OK: same-speaker clips are closer than different-speaker (real ve weights).");
}
