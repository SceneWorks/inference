//! sc-18765 — **LTX-2.5 conv video VAE + audio VAE on REAL weights, through the shipped ports.**
//!
//! `ltx_2_5_vae_conformance.rs` proves the 2.5 component checkpoints carry exactly the tensors the
//! 2.3 ports consume. That is a claim about the files. These tests are the claim about the *engine*:
//! the real 1.45 GB conv VAE and 0.36 GB audio VAE are converted through
//! [`mlx_gen_ltx::convert_vae_components`] and then driven by the **unmodified** `LtxVideoVae`,
//! `AudioDecoder` and `LtxVocoder`, end to end, at the LTX-2.5 trainer geometry (960×544×89) and at
//! a SceneWorks bucket (768×512).
//!
//! `#[ignore]`d — needs the gated `Lightricks/LTX-2.5` weights and a GPU. `LTX25_VAE_DIR` points at
//! a directory holding the two component checkpoints; the tests never search a cache for them
//! (inference source may not reference HF caches — `scripts/check-workspace.py`). Run:
//!
//! ```text
//! LTX25_VAE_DIR=/path/to/Lightricks--LTX-2.5/vae \
//! LTX25_SPLIT_DIR=/tmp/ltx25-split \
//!   cargo test -p mlx-gen-ltx --release --test ltx_2_5_vae_real_weights -- --ignored --nocapture
//! ```
//!
//! `LTX25_SPLIT_DIR` is optional and only caches the converted components between runs; with it
//! unset each test converts into its own temp dir.

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::memory::{get_peak_memory, reset_peak_memory};
use mlx_rs::ops::{abs, max as max_op, mean, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_ltx::audio_vae::AudioDecoder;
use mlx_gen_ltx::config::{AudioVaeConfig, LatentLogVar, LtxVaeConfig, VocoderConfig};
use mlx_gen_ltx::convert::convert_vae_components;
use mlx_gen_ltx::vae::LtxVideoVae;
use mlx_gen_ltx::vocoder::LtxVocoder;

const CONV_VAE: &str = "ltx-2.5-video-vae-conv-bf16.safetensors";
const AUDIO_VAE: &str = "ltx-2.5-audio-vae-bf16.safetensors";

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// The directory holding the LTX-2.5 VAE component checkpoints, from `LTX25_VAE_DIR`. Explicit
/// only: inference source may not reach into a model cache to find weights.
fn vae_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("LTX25_VAE_DIR")?);
    dir.join(CONV_VAE).is_file().then_some(dir)
}

/// Converted split components, cached in `LTX25_SPLIT_DIR` when set. Returns `None` (with a skip
/// message) when the 2.5 weights are not present. The `TempDir` guard, when used, must outlive the
/// returned path — hence the tuple.
fn split_components() -> Option<(PathBuf, Option<tempfile::TempDir>)> {
    let src = match vae_dir() {
        Some(d) => d,
        None => {
            eprintln!("skip: set LTX25_VAE_DIR to a directory holding {CONV_VAE} and {AUDIO_VAE}");
            return None;
        }
    };
    let (out, guard) = match std::env::var_os("LTX25_SPLIT_DIR") {
        Some(d) => (PathBuf::from(d), None),
        None => {
            let tmp = tempfile::tempdir().expect("tempdir");
            (tmp.path().to_path_buf(), Some(tmp))
        }
    };
    if out.join("vae_decoder.safetensors").is_file() && out.join("vocoder.safetensors").is_file() {
        eprintln!(
            "[convert] reusing cached split components at {}",
            out.display()
        );
        return Some((out, guard));
    }
    let t = Instant::now();
    let components = convert_vae_components(src.join(CONV_VAE), Some(src.join(AUDIO_VAE)), &out)
        .expect("convert the LTX-2.5 VAE components");
    eprintln!(
        "[convert] {:?} -> {} in {:.1}s",
        components,
        out.display(),
        t.elapsed().as_secs_f64()
    );
    Some((out, guard))
}

/// A smooth, band-limited test clip in [-1, 1]: `(1, 3, F, H, W)`. White noise is meaningless for a
/// 32×-spatial / 8×-temporal autoencoder — it has nothing to reconstruct — so the round trip is
/// driven with low-frequency structure the VAE is actually built to carry.
fn synthetic_clip(frames: i32, h: i32, w: i32) -> Array {
    let mut data = Vec::with_capacity((3 * frames * h * w) as usize);
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
    Array::from_slice(&data, &[1, 3, frames, h, w])
}

fn psnr_db(got: &Array, want: &Array) -> f32 {
    let diff = subtract(got, want).unwrap();
    let mse = mean(multiply(&diff, &diff).unwrap(), None)
        .unwrap()
        .item::<f32>();
    // Peak-to-peak of a [-1, 1] signal is 2.
    20.0 * (2.0f32).log10() - 10.0 * mse.max(1e-20).log10()
}

fn max_abs(a: &Array) -> f32 {
    max_op(abs(a).unwrap(), None).unwrap().item::<f32>()
}

#[test]
#[ignore = "sc-18765: needs the gated Lightricks/LTX-2.5 conv VAE (1.45 GB) + a GPU"]
fn conv_vae_round_trips_at_trainer_geometry_and_a_sceneworks_bucket() {
    let Some((dir, _guard)) = split_components() else {
        return;
    };
    let cfg = LtxVaeConfig::from_model_dir(&dir).expect("converted embedded_config.json");
    assert_eq!(
        cfg.latent_log_var,
        LatentLogVar::Uniform,
        "the 2.5 conv VAE declares a uniform log-variance head"
    );
    assert_eq!(cfg.patch_size, 4);
    assert_eq!(cfg.latent_channels, 128);

    let dec = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae_decoder");
    let enc = Weights::from_file(dir.join("vae_encoder.safetensors")).expect("vae_encoder");
    // Built through the SHIPPED constructor: a load error here is the drop-in claim failing.
    let vae =
        LtxVideoVae::from_weights(&dec, Some(&enc), &cfg).expect("build LtxVideoVae from 2.5");
    drop(dec);
    drop(enc);

    // (width, height, frames): the LTX-2.5 trainer validation geometry, and a SceneWorks bucket.
    for (w, h, frames) in [(960, 544, 89), (768, 512, 25)] {
        let clip = synthetic_clip(frames, h, w);
        clip.eval().unwrap();

        reset_peak_memory();
        let t = Instant::now();
        let latent = vae.encode(&clip).expect("encode");
        latent.eval().unwrap();
        let encode_s = t.elapsed().as_secs_f64();
        let encode_peak = get_peak_memory();

        let (t_lat, h_lat, w_lat) = (1 + (frames - 1) / 8, h / 32, w / 32);
        assert_eq!(
            latent.shape(),
            &[1, 128, t_lat, h_lat, w_lat],
            "{w}x{h}x{frames}: latent geometry (32x32x8 scale factors) must be unchanged from 2.3"
        );
        assert!(
            max_abs(&latent).is_finite(),
            "{w}x{h}x{frames}: encoder produced a non-finite latent"
        );

        reset_peak_memory();
        let t = Instant::now();
        let round = vae.decode(&latent).expect("decode");
        round.eval().unwrap();
        let decode_s = t.elapsed().as_secs_f64();
        let decode_peak = get_peak_memory();

        assert_eq!(
            round.shape(),
            clip.shape(),
            "{w}x{h}x{frames}: decode must return the input geometry"
        );
        let peak_abs = max_abs(&round);
        assert!(
            peak_abs.is_finite() && peak_abs < 5.0,
            "{w}x{h}x{frames}: decode returned out-of-range values (max|v| = {peak_abs})"
        );
        let psnr = psnr_db(&round, &clip);
        eprintln!(
            "[2.5 conv VAE] {w}x{h}x{frames} latent{:?}  encode {encode_s:.1}s peak {:.2} GiB  \
             decode {decode_s:.1}s peak {:.2} GiB  round-trip PSNR {psnr:.2} dB",
            latent.shape(),
            gib(encode_peak),
            gib(decode_peak),
        );
        // Measured on the real LTX-2.5 conv VAE (2026-08-18, M-series, f32): **58.29 dB** at
        // 960x544x89 and **53.71 dB** at 768x512x25. An incompatible port — wrong conv layout,
        // swapped per-channel statistics, a mis-sliced log-variance tail — lands in the single
        // digits or negative, so the bar sits well below the measurement and far above failure.
        assert!(
            psnr > 45.0,
            "{w}x{h}x{frames}: round-trip PSNR {psnr:.2} dB is too low for a working VAE"
        );
    }
}

#[test]
#[ignore = "sc-18765: needs the gated Lightricks/LTX-2.5 audio VAE (0.36 GB) + a GPU"]
fn audio_vae_and_vocoder_produce_correct_length_stereo_audio() {
    let Some((dir, _guard)) = split_components() else {
        return;
    };
    let audio_cfg = AudioVaeConfig::from_model_dir(&dir).expect("audio vae config");
    let voc_cfg = VocoderConfig::from_model_dir(&dir).expect("vocoder config");
    assert_eq!(
        audio_cfg,
        AudioVaeConfig::defaults(),
        "the 2.5 audio VAE structure must match the shipped port's"
    );
    assert!(voc_cfg.bwe.is_some() && voc_cfg.core.is_bigvgan());
    assert_eq!(voc_cfg.bwe_input_sample_rate, 16_000);
    assert_eq!(voc_cfg.bwe_output_sample_rate, 48_000);

    let audio_w = Weights::from_file(dir.join("audio_vae.safetensors")).expect("audio_vae");
    let voc_w = Weights::from_file(dir.join("vocoder.safetensors")).expect("vocoder");
    let decoder = AudioDecoder::from_weights(&audio_w, &audio_cfg).expect("build AudioDecoder");
    let vocoder = LtxVocoder::from_weights(&voc_w, &voc_cfg).expect("build LtxVocoder");
    drop(audio_w);
    drop(voc_w);

    // An AV render at 89 frames / 24 fps is ~3.67 s of audio. The audio latent runs at 16 kHz with
    // hop 160 ⇒ 100 mel frames/s, and the decoder expands the latent 4× along time, so `T` latent
    // steps carry `4T - 3` mel frames. Pick `T` for ~3.7 s and derive everything else from config.
    let t_lat = 93i32;
    let key = mlx_rs::random::key(18765).unwrap();
    let latent = mlx_rs::random::normal::<f32>(
        &[1, audio_cfg.z_channels, t_lat, 16],
        None,
        None,
        Some(&key),
    )
    .unwrap();

    reset_peak_memory();
    let t = Instant::now();
    let mel = decoder.decode(&latent).expect("audio vae decode");
    mel.eval().unwrap();
    let mel_s = t.elapsed().as_secs_f64();

    assert_eq!(
        mel.shape(),
        &[1, audio_cfg.out_ch, 4 * t_lat - 3, audio_cfg.mel_bins],
        "audio VAE must produce a stereo mel of 4T-3 frames x mel_bins"
    );
    let mel_frames = mel.shape()[2];

    let t = Instant::now();
    let wav = vocoder.forward(&mel).expect("vocoder forward");
    wav.eval().unwrap();
    let voc_s = t.elapsed().as_secs_f64();
    let peak = get_peak_memory();

    // The BWE stage resamples the core generator's output; both rates come from the config, so the
    // expected sample counts are derived, not hardcoded.
    let core_upsample: i32 = voc_cfg.core.upsample_rates.iter().product();
    let core_samples = mel_frames * core_upsample;
    let ratio = voc_cfg.bwe_output_sample_rate / voc_cfg.bwe_input_sample_rate;
    assert_eq!(
        wav.shape(),
        &[1, 2, core_samples * ratio],
        "vocoder must emit stereo at {} Hz for {mel_frames} mel frames",
        voc_cfg.bwe_output_sample_rate
    );

    let core_seconds = core_samples as f64 / voc_cfg.bwe_input_sample_rate as f64;
    let out_seconds = wav.shape()[2] as f64 / voc_cfg.bwe_output_sample_rate as f64;
    eprintln!(
        "[2.5 audio] latent T={t_lat} -> mel {:?} ({mel_s:.2}s) -> wav {:?} ({voc_s:.2}s) \
         peak {:.2} GiB | core stage {core_samples} samples @ {} Hz = {core_seconds:.3}s stereo | \
         output @ {} Hz = {out_seconds:.3}s stereo",
        mel.shape(),
        wav.shape(),
        gib(peak),
        voc_cfg.bwe_input_sample_rate,
        voc_cfg.bwe_output_sample_rate,
    );
    assert!(
        (core_seconds - out_seconds).abs() < 1e-6,
        "the BWE stage must preserve duration: {core_seconds}s core vs {out_seconds}s output"
    );

    let amplitude = max_abs(&wav);
    assert!(
        amplitude.is_finite() && amplitude > 1e-4 && amplitude <= 1.5,
        "vocoder waveform is not a sane audio signal (max|v| = {amplitude})"
    );
}

/// The DiffVAE file's conv encoder loads through the same port at `latent_log_var: constant`, and
/// produces the same normalized means as the conv VAE's `uniform` encoder would — the reference
/// keeps `normalize(conv_out[:, :128])` in both modes and differs only in what it does with the
/// discarded tail.
#[test]
#[ignore = "sc-18765: needs the gated Lightricks/LTX-2.5 DiffVAE (1.47 GB) + a GPU"]
fn diffvae_encoder_loads_at_constant_logvar_and_the_mode_is_checked_against_the_weights() {
    let Some(src) = vae_dir() else {
        eprintln!("skip: set LTX25_VAE_DIR to a directory holding the LTX-2.5 VAE components");
        return;
    };
    let diff = src.join("ltx-2.5-video-vae-bf16.safetensors");
    if !diff.is_file() {
        eprintln!("skip: {} not cached", diff.display());
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("diffvae");
    let components =
        convert_vae_components(&diff, None::<&Path>, &out).expect("convert the DiffVAE encoder");
    // sc-18766 added the NA diffusion decoder this file's decoder half becomes. What must NOT
    // appear is a `vae_decoder`: that component name is the conv stack's, and this file has none.
    assert_eq!(
        components,
        vec!["vae_encoder", "vae_diffusion_decoder"],
        "a CausalDiffusionVAE yields the conv encoder plus the NA diffusion decoder"
    );
    assert!(
        !out.join("vae_decoder.safetensors").exists(),
        "a CausalDiffusionVAE has no conv decoder to emit"
    );

    let cfg = LtxVaeConfig::from_model_dir(&out).expect("embedded_config.json");
    assert_eq!(cfg.latent_log_var, LatentLogVar::Constant);
    let enc = Weights::from_file(out.join("vae_encoder.safetensors")).expect("vae_encoder");

    // The 129-channel conv_out satisfies `constant` (and `uniform`); it must NOT satisfy
    // `per_channel`, which would need 256. This is the executed control on the width check — without
    // it the mode field would be decorative.
    let mut wrong = cfg.clone();
    wrong.latent_log_var = LatentLogVar::PerChannel;
    let err = match LtxVideoVae::encoder_only(&enc, &wrong) {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("a per_channel declaration must not accept a 129-channel head"),
    };
    assert!(
        err.contains("129") && err.contains("256"),
        "the refusal must name both widths, got: {err}"
    );

    let uniform = LtxVaeConfig {
        latent_log_var: LatentLogVar::Uniform,
        ..cfg.clone()
    };
    let constant_vae = LtxVideoVae::encoder_only(&enc, &cfg).expect("constant-mode encoder");
    assert!(constant_vae.has_encoder() && !constant_vae.has_decoder());
    let decode_err = match constant_vae.decode(&Array::zeros::<f32>(&[1, 128, 1, 2, 2]).unwrap()) {
        Err(e) => format!("{e}"),
        Ok(_) => panic!("an encoder-only VAE must not decode"),
    };
    assert!(
        decode_err.contains("encoder-only"),
        "the decode refusal must say why, got: {decode_err}"
    );

    let clip = synthetic_clip(9, 256, 256);
    let as_constant = constant_vae.encode(&clip).expect("encode");
    let as_uniform = LtxVideoVae::encoder_only(&enc, &uniform)
        .expect("uniform-mode encoder")
        .encode(&clip)
        .expect("encode");
    as_constant.eval().unwrap();
    as_uniform.eval().unwrap();
    assert_eq!(as_constant.shape(), &[1, 128, 2, 8, 8]);
    let delta = max_abs(&subtract(&as_constant, &as_uniform).unwrap());
    eprintln!("[2.5 DiffVAE encoder] constant-vs-uniform means max|Δ| = {delta:.3e}");
    assert_eq!(
        delta, 0.0,
        "the two log-variance modes must yield bit-identical means — they differ only in the \
         discarded tail"
    );
}
