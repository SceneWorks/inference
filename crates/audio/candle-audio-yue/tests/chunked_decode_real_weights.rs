//! **Chunked decode ≡ whole-song decode on the real codec and Vocos weights** (epic sc-19373).
//!
//! The production codec decoder and Vocos upsampler evaluate a song
//! [`DECODE_CHUNK_FRAMES`] frames at a time with receptive-field context on each side, so the
//! decode stage's working set no longer grows with song length. Upstream decodes the whole song
//! at once; this check runs both over a 60 s synthetic 8-codebook grid (3 000 frames → 12 chunks,
//! 11 seams) on CPU/f32 and asserts the stitched output equals the whole-song output within the
//! measured reference noise floors (xcodec ≤ 1e-5, Vocos ≤ 2e-5 relative — the bounds of
//! `xcodec_parity` / `vocos_parity`). Mutations: a context narrower than the receptive field must
//! fail the same bound.
//!
//! ```text
//! YUE_XCODEC_SNAPSHOT=/path/to/xcodec-mini-infer \
//!   cargo test --release --locked -p candle-audio-yue --test chunked_decode_real_weights \
//!   -- --ignored --nocapture
//! ```
//!
//! Loads real weights and decodes 60 s whole-song for the comparison (~10 GB CPU working set):
//! run it under an external RSS watchdog.

use std::path::PathBuf;

use candle_audio_yue::candle_audio::candle_core::Device;
use candle_audio_yue::candle_audio::gen_core::CancelFlag;
use candle_audio_yue::codec::{
    CodecDecoder, XcodecDecoder, CHECKPOINT, DECODER_CONTEXT_FRAMES, DECODE_CHUNK_FRAMES,
};
use candle_audio_yue::tokens::{CodecFrames, CODEBOOK_SIZE, NUM_CODEBOOKS};
use candle_audio_yue::vocoder::{
    forward_chunked, vocos_context_frames, Track, Vocoder, VocosUpsampler,
};

/// 60 s at 50 frames/s.
const FRAMES: usize = 3_000;
// The grid must cross several chunk seams.
const _: () = assert!(FRAMES > 5 * DECODE_CHUNK_FRAMES);
/// The xcodec reference noise floor bound (`xcodec_parity::WAVE_MAX_REL`).
const XCODEC_MAX_REL: f64 = 1e-5;
/// The Vocos reference noise floor bound (`vocos_parity::WAVE_MAX_REL`).
const VOCOS_MAX_REL: f64 = 2e-5;

/// A deterministic pseudo-random grid (xorshift) over the full codebook range.
fn grid(frames: usize) -> CodecFrames {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    CodecFrames {
        codebooks: (0..NUM_CODEBOOKS)
            .map(|_| {
                (0..frames)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s % u64::from(CODEBOOK_SIZE)) as u32
                    })
                    .collect()
            })
            .collect(),
    }
}

/// `max|got − want| / max|want|`.
fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let peak = want.iter().fold(0f64, |m, &v| m.max(v.abs() as f64));
    let diff = got
        .iter()
        .zip(want)
        .fold(0f64, |m, (&a, &b)| m.max((a as f64 - b as f64).abs()));
    diff / peak
}

#[test]
#[ignore = "real weights: set YUE_XCODEC_SNAPSHOT to the staged xcodec-mini-infer snapshot; run with --ignored under an RSS watchdog"]
fn chunked_codec_and_vocos_equal_the_whole_song_decode_on_a_60s_grid() {
    let snapshot = PathBuf::from(
        std::env::var("YUE_XCODEC_SNAPSHOT")
            .expect("set YUE_XCODEC_SNAPSHOT to the xcodec-mini-infer snapshot dir"),
    );
    let device = Device::Cpu;
    let frames = grid(FRAMES);

    // xcodec: the production (chunked) decode vs one whole-song pass.
    let codec = XcodecDecoder::load(&snapshot.join(CHECKPOINT), &device).expect("load xcodec");
    let embed = codec.get_embed(&frames).expect("get_embed");
    let whole: Vec<f32> = codec
        .decode_embedding(&embed, &|| false)
        .unwrap()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let chunked = codec.decode(&frames, &CancelFlag::new()).unwrap().wave;
    let rel = max_rel(&chunked, &whole);
    println!("xcodec chunked vs whole: {rel:.3e} (bound {XCODEC_MAX_REL:.0e})");
    assert!(rel <= XCODEC_MAX_REL, "xcodec chunked diverges: {rel:.3e}");
    for context in [DECODER_CONTEXT_FRAMES / 2, 2] {
        let short = codec
            .decode_embedding_chunked(&embed, DECODE_CHUNK_FRAMES, context, &|| false)
            .unwrap()
            .unwrap();
        let m = max_rel(&short, &whole);
        println!("xcodec mutation context {context}: {m:.3e}");
        assert!(
            m > XCODEC_MAX_REL,
            "xcodec context {context} passed ({m:.3e})"
        );
    }
    drop((codec, whole, chunked));

    // Vocos: both decoders, production (chunked) vs whole-song.
    let vocos = VocosUpsampler::load(&snapshot, &device).expect("load both Vocos decoders");
    for track in [Track::Vocals, Track::Instrumental] {
        let dec = vocos.decoder(track);
        let whole = dec.forward(&embed).expect("whole-song vocos");
        let chunked = vocos.decode(track, &embed, &CancelFlag::new()).unwrap();
        let rel = max_rel(&chunked, &whole);
        println!("{track:?} vocos chunked vs whole: {rel:.3e} (bound {VOCOS_MAX_REL:.0e})");
        assert!(
            rel <= VOCOS_MAX_REL,
            "{track:?} vocos chunked diverges: {rel:.3e}"
        );
        let ctx = vocos_context_frames(dec.config());
        assert_eq!(ctx, 31, "8 ConvNeXt blocks, n_fft 3528, hop 882");
        // Cut into the backbone reach, then into the ISTFT overlap-add span.
        for context in [ctx / 2, 1] {
            let short = forward_chunked(dec, &embed, DECODE_CHUNK_FRAMES, context, &|| false)
                .unwrap()
                .unwrap();
            let m = max_rel(&short, &whole);
            println!("{track:?} vocos mutation context {context}: {m:.3e}");
            assert!(
                m > VOCOS_MAX_REL,
                "{track:?} context {context} passed ({m:.3e})"
            );
        }
    }
}
