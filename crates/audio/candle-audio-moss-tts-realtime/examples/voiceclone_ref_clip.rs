//! Regenerate the committed MOSS-TTS-Realtime voice-clone reference clip (sc-17264).
//!
//! The two sc-14149/sc-14151 voice-clone gates need a reference speaker to clone. Before sc-17264
//! that came from `MOSS_VOICECLONE_REF`, a 24 kHz f32-LE clip that existed in no repository, on no
//! Hub, and in no provisioning script — so both tests were left out of the real-weight lane
//! entirely. This example renders that clip with **Kokoro** (`hexgrad/Kokoro-82M`, Apache-2.0), the
//! repository's sanctioned source of reference audio (see `candle-audio-chatterbox-ve`'s
//! `embed_demo`, which does the same for its speaker-embedding gate) — so no third-party audio
//! fixture is committed and there is no clip licence to clear.
//!
//! A rendered voice rather than a synthesized one (the sc-17270 codec clip is arithmetic). The gate
//! asserts the cloned output's CAMPPlus x-vector resembles the reference **more than the default
//! voice does** — a *relative* comparison, and measurement shows a synthetic reference also clears
//! it (0.530 vs 0.231). Real speech is used because it is the stronger signal: roughly double the
//! absolute clone-to-reference similarity (0.912 vs 0.719 for the default voice), and the clone
//! stays perfectly intelligible where a synthetic reference starts losing words (CER 0.000 vs
//! 0.070). A reference outside the model's distribution measures the metric's slope more than the
//! model's cloning.
//!
//! Kokoro emits 24 kHz mono natively, which is exactly the codec's rate — the clip reaches the
//! encoder without a resample.
//!
//! ```sh
//! KOKORO_SNAPSHOT=/path/to/Kokoro-82M/snapshots/<rev> \
//!   cargo run --release -p candle-audio-moss-tts-realtime --example voiceclone_ref_clip
//! ```
//!
//! Writes the clip and a provenance JSON beside it. Override the destination with
//! `MOSS_VOICECLONE_REF_OUT` (a directory).

use candle_audio::gen_core::{
    AudioParams, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use std::path::PathBuf;

/// The reference utterance. Phonetically broad and ~12 s at Kokoro's pace: an x-vector wants enough
/// voiced material to characterize a speaker, and the multi-turn gate compares embeddings of whole
/// turns against it.
const TEXT: &str = "The northern lights shimmered above the quiet harbour that evening, and every \
                    sailor paused to watch. She counted seven boats returning, their lanterns \
                    swaying gently against a cold blue sky.";

/// A Kokoro voice deliberately unlike MOSS-TTS-Realtime's default speaker — the gate's margin is
/// `sim(clone, ref) > sim(default, ref) + 0.05`, so a reference that already sounds like the
/// default voice would leave nothing to measure. Recorded in the metadata so a regeneration that
/// changes it is visible.
const VOICE: &str = "af_heart";

fn main() {
    let snapshot = std::env::var("KOKORO_SNAPSHOT")
        .expect("set KOKORO_SNAPSHOT to a hexgrad/Kokoro-82M snapshot dir");
    let out_dir = std::env::var("MOSS_VOICECLONE_REF_OUT").map_or_else(
        |_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("fixtures")
        },
        PathBuf::from,
    );

    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)));
    let generator = candle_audio_kokoro::load(&spec).expect("load kokoro");
    let request = GenerationRequest {
        prompt: TEXT.to_string(),
        audio: Some(AudioParams {
            voice: Some(VOICE.to_string()),
            language: Some("en".to_string()),
            sample_rate: Some(24_000),
            ..Default::default()
        }),
        ..Default::default()
    };
    let track = match generator
        .generate(&request, &mut |_| {})
        .expect("kokoro generate")
    {
        GenerationOutput::Audio(track) => track,
        other => panic!("expected Audio, got {other:?}"),
    };
    assert_eq!(
        track.sample_rate, 24_000,
        "clip must be at the codec's rate"
    );
    assert_eq!(track.channels, 1, "clip must be mono");

    // Peak-normalize so the committed clip has a predictable level regardless of what the voice
    // happened to produce; the x-vector is level-sensitive enough that "whatever came out" is not
    // a property worth freezing by accident.
    let peak = track.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    assert!(peak > 0.0, "kokoro returned a silent clip");
    let samples: Vec<f32> = track.samples.iter().map(|s| s * 0.9 / peak).collect();

    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for sample in &samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }

    std::fs::create_dir_all(&out_dir).expect("create fixture dir");
    let clip_path = out_dir.join("moss_voiceclone_ref_clip.f32");
    std::fs::write(&clip_path, &bytes).expect("write clip");

    let seconds = samples.len() as f32 / 24_000.0;
    let metadata = format!(
        "{{\n  \"story\": \"sc-17264\",\n  \"generator\": \
         \"crates/audio/candle-audio-moss-tts-realtime/examples/voiceclone_ref_clip.rs\",\n  \
         \"source\": {{\n    \"model\": \"hexgrad/Kokoro-82M\",\n    \"license\": \
         \"Apache-2.0\",\n    \"voice\": \"{VOICE}\",\n    \"text\": {text:?},\n    \"rendered\": \
         true,\n    \"third_party_audio\": false\n  }},\n  \"clip\": {{\n    \"format\": \"raw f32 \
         little-endian, mono\",\n    \"sample_rate\": 24000,\n    \"samples\": {samples_len},\n    \
         \"seconds\": {seconds:.3},\n    \"peak\": 0.9,\n    \"sha256\": \"{digest}\"\n  }}\n}}\n",
        text = TEXT,
        samples_len = samples.len(),
        digest = sha256_hex(&bytes),
    );
    let metadata_path = out_dir.join("moss_voiceclone_ref_metadata.json");
    std::fs::write(&metadata_path, metadata).expect("write metadata");

    println!(
        "wrote {} ({} bytes, {} samples, {seconds:.3} s, voice {VOICE})",
        clip_path.display(),
        bytes.len(),
        samples.len()
    );
    println!("wrote {}", metadata_path.display());
}

/// Minimal SHA-256 so the example carries no dependency just to stamp a digest.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(value);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}
