//! Decode-stage working-set probe (not a test gate): runs the engine's decode stage — xcodec decode
//! of both tracks, Vocos 44.1 kHz upsample of both, limiter + low-band splice — over a synthetic
//! `SECONDS`-long 8-codebook grid, in the engine's order (codec released before the vocoders load),
//! printing each step's wall time. Measure the process's peak footprint externally
//! (`/usr/bin/time -l`) to compare chunk sizes; `whole` decodes each track in one pass, as upstream
//! does.
//!
//! ```text
//! cargo run --release -p candle-audio-yue --example decode_stage_peak -- \
//!   ~/.cache/sceneworks-yue-assets/xcodec-mini-infer 218 [whole|<chunk frames>]
//! ```
//!
//! Runs on the audio lane's default device (CPU, or Metal / CUDA with `--features metal|cuda`).
//! Run real weights under an external RSS watchdog.

use std::path::PathBuf;
use std::time::Instant;

use candle_audio_yue::candle_audio::gen_core::{CancelFlag, OutputLimiter};
use candle_audio_yue::codec::{CodecDecoder, XcodecDecoder, CHECKPOINT, DECODE_CHUNK_FRAMES};
use candle_audio_yue::splice::{post_process, TrackPair};
use candle_audio_yue::tokens::{CodecFrames, CODEBOOK_SIZE, FRAMES_PER_SECOND, NUM_CODEBOOKS};
use candle_audio_yue::vocoder::{Track, Vocoder, VocosUpsampler};

fn grid(frames: usize, seed: u64) -> CodecFrames {
    let mut s = seed | 1;
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = PathBuf::from(
        args.first()
            .expect("usage: <xcodec-mini-infer root> <seconds> [whole|<chunk frames>]"),
    );
    let seconds: usize = args.get(1).map_or(60, |s| s.parse().expect("seconds"));
    let chunk = match args.get(2).map(String::as_str) {
        None => DECODE_CHUNK_FRAMES,
        Some("whole") => usize::MAX,
        Some(n) => n.parse().expect("chunk frames"),
    };
    let frames = seconds * FRAMES_PER_SECOND as usize;
    let device = candle_audio_yue::candle_audio::default_device().expect("device");
    let cancel = CancelFlag::new();
    let grids = [grid(frames, 0x9E37_79B9), grid(frames, 0x7F4A_7C15)];
    eprintln!("{seconds} s = {frames} frames, chunk {chunk}, device {device:?}");

    let t = Instant::now();
    let codec = XcodecDecoder::load(&root.join(CHECKPOINT), &device)
        .expect("load xcodec")
        .with_chunk_frames(chunk);
    let decoded: Vec<_> = grids
        .iter()
        .map(|g| codec.decode(g, &cancel).expect("codec decode"))
        .collect();
    drop(codec);
    eprintln!("codec {:.1}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let vocos = VocosUpsampler::load(&root, &device)
        .expect("load vocos")
        .with_chunk_frames(chunk);
    let stems: Vec<Vec<f32>> = [Track::Vocals, Track::Instrumental]
        .iter()
        .zip(&decoded)
        .map(|(&track, d)| vocos.decode(track, &d.embedding, &cancel).expect("vocos"))
        .collect();
    drop(vocos);
    eprintln!("vocos {:.1}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let codec_rate = TrackPair {
        vocals: decoded[0].wave.clone(),
        instrumental: decoded[1].wave.clone(),
    };
    let vocoder_rate = TrackPair {
        vocals: stems[0].clone(),
        instrumental: stems[1].clone(),
    };
    let out = post_process(&codec_rate, &vocoder_rate, OutputLimiter::Clamp).expect("splice");
    eprintln!(
        "splice {:.1}s, mix {} samples",
        t.elapsed().as_secs_f64(),
        out.mix.len()
    );
}
