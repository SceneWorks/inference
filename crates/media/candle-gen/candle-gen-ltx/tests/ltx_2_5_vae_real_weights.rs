//! sc-18765 — **the candle LTX ports on REAL LTX-2.5 weights.**
//!
//! `ltx_2_5_vae_conformance.rs` proves the shipped candle loaders find every tensor an LTX-2.5
//! component checkpoint carries. These tests close the loop with the real weights: the 1.45 GB conv
//! VAE and the 0.36 GB audio VAE are memory-mapped **as shipped** — no conversion, no key remap, no
//! transpose — and driven through `LtxVideoVae`, `AudioDecoder` and `LtxVocoder`.
//!
//! `#[ignore]`d — needs the gated `Lightricks/LTX-2.5` weights. Deliberately **not** CUDA-gated: the
//! whole point is that these are the same ports on the same files, so the load and a small round
//! trip are meaningful on any device. Geometry is device-scaled:
//!
//!  - default: a 64x64x1 round trip, which a CPU can finish in seconds;
//!  - `LTX25_GEOM=WxHxF`: an explicit geometry;
//!  - `LTX25_FULL=1`: the acceptance geometries — the LTX-2.5 trainer's 960x544x89 and a SceneWorks
//!    768x512x25 bucket. Measured at ~9 minutes wall clock on this Mac's CPU, so it is affordable
//!    off the CUDA lane too.
//!
//! `LTX25_VAE_DIR` points at a directory holding the two component checkpoints; the tests never
//! search a cache for them (inference source may not reference HF caches —
//! `scripts/check-workspace.py`).
//!
//! ```text
//! LTX25_VAE_DIR=/path/to/Lightricks--LTX-2.5/vae \
//!   cargo test -p candle-gen-ltx --release --test ltx_2_5_vae_real_weights -- --ignored --nocapture
//! # on the CUDA lane, add --features cuda and LTX25_FULL=1
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen_ltx::audio_vae::AudioDecoder;
use candle_gen_ltx::config::{AudioVaeConfig, VocoderConfig, LATENT_CHANNELS};
use candle_gen_ltx::vae::LtxVideoVae;
use candle_gen_ltx::vocoder::LtxVocoder;

const CONV_VAE: &str = "ltx-2.5-video-vae-conv-bf16.safetensors";
const AUDIO_VAE: &str = "ltx-2.5-audio-vae-bf16.safetensors";

/// The directory holding the LTX-2.5 VAE component checkpoints, from `LTX25_VAE_DIR`. Explicit
/// only: inference source may not reach into a model cache to find weights.
fn vae_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("LTX25_VAE_DIR")?);
    dir.join(CONV_VAE).is_file().then_some(dir)
}

/// The CUDA device when the feature is on and a GPU is present, else CPU. The ports are
/// device-agnostic; only the affordable geometry differs.
fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    Device::Cpu
}

fn full_geometry() -> bool {
    std::env::var("LTX25_FULL").is_ok_and(|v| v == "1")
}

/// Band-limited `(1, 3, F, H, W)` content in [-1, 1] — a 32x-spatial autoencoder has nothing to
/// reconstruct from white noise, so the round trip is driven with structure it is built to carry.
fn synthetic_clip(frames: usize, h: usize, w: usize, device: &Device) -> Tensor {
    let mut data = Vec::with_capacity(3 * frames * h * w);
    for c in 0..3 {
        for f in 0..frames {
            let t = f as f32 / frames.max(2) as f32;
            for y in 0..h {
                let v = y as f32 / h as f32;
                for x in 0..w {
                    let u = x as f32 / w as f32;
                    let value =
                        0.55 * ((6.0 * u + 1.7 * t + c as f32).sin()) * ((4.0 * v - 2.3 * t).cos())
                            + 0.25 * (2.0 * v - 1.0)
                            + 0.10 * ((13.0 * u * v + 3.0 * t).sin());
                    data.push(value.clamp(-1.0, 1.0));
                }
            }
        }
    }
    Tensor::from_vec(data, (1, 3, frames, h, w), device).expect("clip")
}

fn psnr_db(got: &Tensor, want: &Tensor) -> f64 {
    let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(got.len(), want.len());
    let mse: f64 = got
        .iter()
        .zip(&want)
        .map(|(a, b)| {
            let e = *a as f64 - *b as f64;
            e * e
        })
        .sum::<f64>()
        / got.len() as f64;
    20.0 * 2.0f64.log10() - 10.0 * mse.max(1e-20).log10()
}

fn max_abs(t: &Tensor) -> f32 {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
}

#[test]
#[ignore = "sc-18765: needs the gated Lightricks/LTX-2.5 conv VAE (1.45 GB)"]
fn conv_vae_round_trips_on_the_unconverted_2_5_component_file() {
    let Some(dir) = vae_dir() else {
        eprintln!("skip: set LTX25_VAE_DIR to a directory holding the LTX-2.5 VAE components");
        return;
    };
    let device = device();
    let path = dir.join(CONV_VAE);
    let t = Instant::now();
    // The file AS SHIPPED — the VarBuilder is rooted at the checkpoint root, which is the single
    // difference from the 2.3 all-in-one path (`vb.pp("vae")`).
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &device)
            .expect("mmap 2.5 conv VAE")
    };
    let vae = LtxVideoVae::new_with_encoder(vb.clone(), vb, LATENT_CHANNELS, 4)
        .expect("the candle port must build straight off the LTX-2.5 conv VAE");
    eprintln!(
        "[candle 2.5 conv VAE] loaded {} on {device:?} in {:.1}s",
        path.display(),
        t.elapsed().as_secs_f64()
    );

    let geometries: Vec<(usize, usize, usize)> = match std::env::var("LTX25_GEOM") {
        // `WxHxF` — an explicit geometry, for putting a CPU run somewhere between the two presets.
        Ok(spec) => {
            let parts: Vec<usize> = spec
                .split(['x', 'X'])
                .map(|p| p.parse().expect("LTX25_GEOM must be WxHxF"))
                .collect();
            assert_eq!(parts.len(), 3, "LTX25_GEOM must be WxHxF, got {spec:?}");
            vec![(parts[0], parts[1], parts[2])]
        }
        Err(_) if full_geometry() => vec![(960, 544, 89), (768, 512, 25)],
        Err(_) => vec![(64, 64, 1)],
    };
    for &(w, h, frames) in &geometries {
        assert_eq!((frames - 1) % 8, 0, "LTX frames must be 1 + 8k");
        assert!(h % 32 == 0 && w % 32 == 0, "LTX geometry must be /32");
        let clip = synthetic_clip(frames, h, w, &device);

        let t = Instant::now();
        let latent = vae.encode(&clip).expect("encode");
        let encode_s = t.elapsed().as_secs_f64();
        assert_eq!(
            latent.dims(),
            [1, LATENT_CHANNELS, 1 + (frames - 1) / 8, h / 32, w / 32],
            "{w}x{h}x{frames}: the 32x32x8 latent geometry must be unchanged from 2.3"
        );

        let t = Instant::now();
        let round = vae.decode(&latent).expect("decode");
        let decode_s = t.elapsed().as_secs_f64();
        assert_eq!(round.dims(), clip.dims());

        let peak = max_abs(&round);
        assert!(
            peak.is_finite() && peak < 5.0,
            "{w}x{h}x{frames}: decode returned out-of-range values (max|v| = {peak})"
        );
        let psnr = psnr_db(&round, &clip);
        eprintln!(
            "[candle 2.5 conv VAE] {w}x{h}x{frames} latent{:?} encode {encode_s:.1}s \
             decode {decode_s:.1}s round-trip PSNR {psnr:.2} dB",
            latent.dims()
        );
        // Measured on the real LTX-2.5 conv VAE (2026-08-18, CPU, f32): **58.29 dB** at
        // 960x544x89 and **53.71 dB** at 768x512x25 — matching the MLX port's numbers at the same
        // geometries to two decimals, by a completely different route. A small tile has far less to
        // work with (48.11 dB at 64x64x1), so the bar sits below all of them and far above an
        // incompatible port, which lands in the single digits or negative.
        assert!(
            psnr > 20.0,
            "{w}x{h}x{frames}: round-trip PSNR {psnr:.2} dB is too low for a working VAE"
        );
    }
}

#[test]
#[ignore = "sc-18765: needs the gated Lightricks/LTX-2.5 audio VAE (0.36 GB)"]
fn audio_vae_and_vocoder_produce_stereo_audio_off_the_unconverted_2_5_component_file() {
    let Some(dir) = vae_dir() else {
        eprintln!("skip: set LTX25_VAE_DIR to a directory holding the LTX-2.5 VAE components");
        return;
    };
    let device = device();
    let path = dir.join(AUDIO_VAE);
    let audio_cfg = AudioVaeConfig::ltx_2_3();
    let voc_cfg = VocoderConfig::ltx_2_3();

    // Both roots are byte-for-byte the 2.3 ones: `audio_vae` for the decoder, the checkpoint root
    // for the vocoder (whose tensors carry their own `vocoder.` prefix).
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &device).expect("mmap 2.5 audio")
    };
    let decoder = AudioDecoder::load(&vb.pp("audio_vae"), &audio_cfg).expect("audio decoder");
    let vocoder = LtxVocoder::load(vb, &device, &voc_cfg).expect("vocoder");

    // ~1 s of audio on CPU, the full ~3.7 s AV-render length on the CUDA lane.
    let t_lat = if full_geometry() { 93 } else { 25 };
    let z_channels = audio_cfg.z_channels as usize;
    let latent = Tensor::zeros((1, z_channels, t_lat, 16), DType::F32, &device)
        .unwrap()
        .affine(0.0, 0.35)
        .unwrap();

    let t = Instant::now();
    let mel = decoder.decode(&latent).expect("audio vae decode");
    let mel_s = t.elapsed().as_secs_f64();
    assert_eq!(
        mel.dims(),
        [
            1,
            audio_cfg.out_ch as usize,
            4 * t_lat - 3,
            audio_cfg.mel_bins as usize
        ],
        "the audio VAE must produce a stereo mel of 4T-3 frames x mel_bins"
    );

    let t = Instant::now();
    let wav = vocoder.forward(&mel).expect("vocoder");
    let voc_s = t.elapsed().as_secs_f64();

    let core_upsample: i32 = voc_cfg.core.upsample_rates.iter().product();
    let core_samples = mel.dims()[2] * core_upsample as usize;
    let ratio = (voc_cfg.bwe_output_sample_rate / voc_cfg.bwe_input_sample_rate) as usize;
    assert_eq!(
        wav.dims(),
        [1, 2, core_samples * ratio],
        "the vocoder must emit stereo at {} Hz",
        voc_cfg.bwe_output_sample_rate
    );
    let core_seconds = core_samples as f64 / voc_cfg.bwe_input_sample_rate as f64;
    eprintln!(
        "[candle 2.5 audio] on {device:?}: latent T={t_lat} -> mel {:?} ({mel_s:.2}s) -> wav {:?} \
         ({voc_s:.2}s) | core stage {core_samples} samples @ {} Hz = {core_seconds:.3}s stereo | \
         output @ {} Hz",
        mel.dims(),
        wav.dims(),
        voc_cfg.bwe_input_sample_rate,
        voc_cfg.bwe_output_sample_rate,
    );
    let amplitude = max_abs(&wav);
    assert!(
        amplitude.is_finite() && amplitude > 1e-5 && amplitude <= 1.5,
        "the vocoder waveform is not a sane audio signal (max|v| = {amplitude})"
    );
}
