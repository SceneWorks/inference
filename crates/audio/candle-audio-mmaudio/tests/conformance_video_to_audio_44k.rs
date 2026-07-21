//! Real-weight video→audio (Foley) conformance for the shipping MMAudio **44.1 kHz** provider
//! (sc-13441): real pinned weights → registry load-by-id (`mmaudio_large_44k`) → a synchronized,
//! non-silent, deterministic, frame-dependent **44.1 kHz** soundtrack.
//!
//! Both tests are `#[ignore]`d and stage the five pinned checkpoints across three repos as named
//! `LoadSpec` components (sc-13666), resolved from env-pointed per-repo snapshot paths, falling back
//! to the audio lane's F-029 hub path (~9.5 GB across `hkchengrex/MMAudio`,
//! `apple/DFN5B-CLIP-ViT-H-14-384`, and `nvidia/bigvgan_v2_44khz_128band_512x`) — see [`common`] for
//! the env vars. The full 44k pipeline (the 1.03B large_44k_v2 MM-DiT plus CLIP ViT-H, Synchformer,
//! the 44k VAE, and NVIDIA BigVGAN v2) is heavy — see the crate PR notes for the memory envelope.
//!
//! ```text
//! cargo test --locked -p candle-audio-mmaudio --test conformance_video_to_audio_44k -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use candle_audio_mmaudio::candle_audio;
use candle_audio_mmaudio::candle_audio::candle_core::{safetensors, IndexOp};
use candle_audio_mmaudio::gen_core::{
    AudioParams, Conditioning, GenerationOutput, GenerationRequest, Image,
};
use gen_core_testkit::AudioProfile;

fn load_provider() -> Box<dyn candle_audio_mmaudio::gen_core::Generator> {
    candle_audio_mmaudio::provider_registry()
        .unwrap()
        .load(candle_audio_mmaudio::GENERATOR_ID_44K, &common::spec_44k())
        .expect("mmaudio_large_44k loads through the explicit registry")
}

/// Deterministic synthetic Foley clip: `n` frames of a moving bright bar (variant shifts phase/hue).
fn synthetic_clip(n: usize, w: u32, h: u32, variant: u8) -> Vec<Image> {
    (0..n)
        .map(|f| {
            let mut px = vec![0u8; (w * h * 3) as usize];
            let bar = ((f as u32 + variant as u32 * 3) * 7) % w;
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 3) as usize;
                    let on = x.abs_diff(bar) < 4;
                    px[i] = if on {
                        220
                    } else {
                        20 + (variant as u32 * 30 % 60) as u8
                    };
                    px[i + 1] = if on { 180 } else { 30 };
                    px[i + 2] = ((x + y + f as u32 * 11 + variant as u32 * 53) % 200) as u8;
                }
            }
            Image {
                width: w,
                height: h,
                pixels: px,
            }
        })
        .collect()
}

/// Real frames from a `raw_frames_8fps` `(T,H,W,3)` u8 tensor, if `MMAUDIO_FOLEY_FRAMES` is set.
fn real_clip() -> Option<Vec<Image>> {
    let path = std::env::var("MMAUDIO_FOLEY_FRAMES").ok()?;
    let device = candle_audio::default_device().ok()?;
    let t = safetensors::load(&path, &device).ok()?;
    let frames = t.get("raw_frames_8fps")?.clone();
    let (n, h, w, _c) = frames.dims4().ok()?;
    let mut out = Vec::with_capacity(n);
    for f in 0..n {
        let pixels: Vec<u8> = frames.i(f).ok()?.flatten_all().ok()?.to_vec1().ok()?;
        out.push(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        });
    }
    Some(out)
}

fn foley_request(frames: Vec<Image>, fps: u32, seed: u64) -> GenerationRequest {
    GenerationRequest {
        prompt: "typing on a keyboard".into(),
        fps: Some(fps),
        seed: Some(seed),
        steps: Some(25),
        conditioning: vec![Conditioning::VideoSync { frames }],
        audio: Some(AudioParams::default()),
        ..Default::default()
    }
}

/// The video→audio testkit gate against the REAL registered 44k provider: advertises VideoSync,
/// accepts + renders one non-silent, plausibly-long, byte-reproducible, frame-DEPENDENT track.
#[test]
#[ignore = "real weights: resolves ~9.5 GB of MMAudio 44k + DFN5B-CLIP + NVIDIA BigVGAN v2; heavy (1.03B DiT); run with --ignored"]
fn mmaudio_44k_video_to_audio_conformance() {
    let g = load_provider();
    let profile = AudioProfile {
        prompt: "typing on a keyboard".to_owned(),
        steps: 8,
        seed: 42,
        cancel_steps: 6,
        audio: AudioParams::default(),
    };
    gen_core_testkit::check_video_to_audio(g.as_ref(), &profile)
        .expect("MMAudio 44k passes the video→audio (Foley) contract");
    println!("check_video_to_audio (44k): PASS");
}

/// The real-Foley DoD at 44.1 kHz: fixed clip + prompt + seed → non-empty, 44.1 kHz, mono, finite,
/// NON-SILENT audio of a plausible length; byte-identical on re-synth; and a *different* clip yields
/// *different* audio (the video condition genuinely drives the output). Writes a WAV.
#[test]
#[ignore = "real weights: resolves ~9.5 GB; heavy (1.03B DiT + CLIP ViT-H); run with --ignored"]
fn mmaudio_44k_synced_foley_wav() {
    let g = load_provider();

    let (frames_a, fps) = match real_clip() {
        Some(f) => {
            println!("using REAL clip: {} frames @ 8 fps", f.len());
            (f, 8)
        }
        None => {
            println!("using synthetic clip: 12 frames @ 8 fps");
            (synthetic_clip(12, 64, 64, 0), 8)
        }
    };
    let expected_secs = frames_a.len() as f32 / fps as f32;

    let req = foley_request(frames_a.clone(), fps, 42);
    let out = g.generate(&req, &mut |_| {}).expect("generate Foley");
    let track = match out {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio, got {other:?}"),
    };
    assert_eq!(track.sample_rate, 44_100, "44k provider emits 44.1 kHz");
    assert_eq!(track.channels, 1);
    assert!(!track.samples.is_empty(), "empty track");
    assert!(
        track.samples.iter().all(|s| s.is_finite()),
        "non-finite samples"
    );

    let peak = track.samples.iter().fold(0f32, |m, s| m.max(s.abs()));
    let rms =
        (track.samples.iter().map(|s| s * s).sum::<f32>() / track.samples.len() as f32).sqrt();
    let secs = track.samples.len() as f32 / track.sample_rate as f32;
    println!(
        "Foley WAV (44k): {} samples ({secs:.3}s) peak={peak:.4} rms={rms:.4}",
        track.samples.len()
    );
    assert!(peak > 1e-3, "track is silent (peak={peak})");
    assert!(
        secs >= expected_secs * 0.25 && secs <= expected_secs * 4.0,
        "duration {secs:.3}s implausible for a {expected_secs:.3}s clip"
    );

    let out2 = g.generate(&req, &mut |_| {}).expect("re-synth");
    let track2 = match out2 {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio, got {other:?}"),
    };
    assert_eq!(
        track.samples, track2.samples,
        "re-synth is not byte-identical (seed/clip law)"
    );

    let frames_b = match real_clip() {
        Some(mut f) => {
            for img in &mut f {
                for p in &mut img.pixels {
                    *p = p.wrapping_sub(60);
                }
            }
            f
        }
        None => synthetic_clip(12, 64, 64, 200),
    };
    let req_b = foley_request(frames_b, fps, 42);
    let track_b = match g.generate(&req_b, &mut |_| {}).expect("generate clip B") {
        GenerationOutput::Audio(t) => t,
        other => panic!("expected Audio, got {other:?}"),
    };
    assert_ne!(
        track.samples, track_b.samples,
        "two different clips (same seed) produced identical audio — conditioning is ignored"
    );

    let out_path = std::env::var("MMAUDIO_WAV_OUT_44K")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("mmaudio_foley_44k.wav"));
    candle_audio::wav::write_wav_pcm16(&out_path, &track).expect("write WAV");
    println!("wrote {}", out_path.display());
}
