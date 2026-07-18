//! AudioVideo **LoRA** e2e parity (sc-2687) — the production joint denoise WITH a full-surface LoRA.
//!
//! Belt-and-suspenders for the `AvDiT` LoRA path (the math is otherwise covered by composition: the
//! `LtxDiT` video gate in `lora_real_weights.rs` proves the per-Linear residual, and the `AvDiT`
//! routing gate there proves the full surface resolves). This runs the *actual* production path —
//! `apply_ltx_adapters` over `AvDiT` + `generate_av_latents` (joint 2-stage denoise) + video &
//! audio decode — and reproduces the reference `denoise_av`-with-`LoRALinear` golden
//! (`tools/dump_ltx_av_lora_e2e_golden.py`) for the same injected conditioning as the base AV e2e
//! golden (a clean A/B — only the LoRA differs). f32 regime (`quant_f32`), same as `av_e2e_parity`.
//!
//! **Lora-agnostic (sc-13019):** the golden records `lora_path` in its metadata (dump-time), this
//! test loads THAT lora (env `LTX_LORA_MULTI` overrides for a fresh A/B), and the structural
//! expectation derives from the lora file's own key inventory (`adapters::lora_target_inventory`) —
//! no hardcoded per-lora target counts. Committed golden: dumped with `DR34ML4Y_LTXXX_V2` on the
//! MLX 0.32.0 non-NAX env (the earlier Samantha-based golden encoded an asset-locked 0.31.2 A/B).
//! On 0.32.0 the free-running f32 chain is envelope-gated (cross-stack drift chaos-amplifies — see
//! `e2e_parity.rs`); the routing/structure gates stay exact.
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` weights (~20 GB) + the golden's LoRA. Run:
//! `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test av_lora_e2e_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, gt, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen_ltx::adapters::{apply_ltx_adapters, lora_target_inventory};
use mlx_gen_ltx::audio_vae::AudioDecoder;
use mlx_gen_ltx::config::{AudioVaeConfig, LtxConfig, LtxVaeConfig, SplitModel, VocoderConfig};
use mlx_gen_ltx::pipeline::{
    decode_audio_track, decode_to_frames, generate_av_latents, NUM_DENOISE_PASSES,
};
use mlx_gen_ltx::positions::{create_audio_position_grid, create_position_grid};
use mlx_gen_ltx::transformer::{AvDiT, Precision};
use mlx_gen_ltx::upsampler::LatentUpsampler;
use mlx_gen_ltx::vae::LtxVideoVae;
use mlx_gen_ltx::vocoder::LtxVocoder;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_av_lora_e2e_golden.safetensors"
);

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

/// The lora the golden was dumped with (`lora_path` metadata, recorded by the dump script);
/// `LTX_LORA_MULTI` overrides for a fresh A/B against a re-dumped golden. A `~/`-relative metadata
/// path expands against `$HOME` so the golden stays portable across home directories.
fn lora_path(golden: &Weights) -> std::path::PathBuf {
    if let Ok(p) = std::env::var("LTX_LORA_MULTI") {
        return p.into();
    }
    let p = golden
        .metadata("lora_path")
        .expect("golden lora_path metadata (re-dump with tools/dump_ltx_av_lora_e2e_golden.py)")
        .to_string();
    if let Some(rest) = p.strip_prefix("~/") {
        return std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(rest);
    }
    p.into()
}

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), want).unwrap()).unwrap();
    let denom = max_op(abs(want).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32(got), want).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(want).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

fn px_gt8(got: &Array, want: &Array) -> f32 {
    let a = got.as_dtype(Dtype::Float32).unwrap();
    let b = want.as_dtype(Dtype::Float32).unwrap();
    let diff = abs(subtract(&a, &b).unwrap()).unwrap();
    let over = gt(&diff, Array::from_int(8))
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    sum(&over, None).unwrap().item::<f32>() / (a.size() as f32) * 100.0
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 weights (~20 GB) + the golden's multi-surface LoRA (~2 GB)"]
fn av_lora_e2e_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("golden");
    let lora = lora_path(&g);
    // File-derived structural expectation (sc-13019): every target the lora carries must resolve
    // on the full AvDiT surface (see lora_real_weights.rs for the video-only complement).
    let (video_n, audio_n) = {
        let lw = Weights::from_file(&lora).expect("lora file (golden lora_path metadata)");
        let (v, a) = lora_target_inventory(&lw);
        assert!(
            !v.is_empty() && !a.is_empty(),
            "the golden's lora must be FULL-surface (video + audio/cross-modal); {} carries \
             video={} audio/cross={}",
            lora.display(),
            v.len(),
            a.len()
        );
        (v.len(), a.len())
    };

    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("config");
    let split = SplitModel::from_model_dir(&dir).expect("split_model.json");
    let tw = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
    let mut dit =
        AvDiT::from_weights(&tw, &cfg, Precision::quant_f32(split.bits, split.group)).expect("dit");
    // The actual production seam: install the full-surface LoRA onto AvDiT.
    let report = apply_ltx_adapters(
        &mut dit,
        &[AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora)],
        NUM_DENOISE_PASSES,
    )
    .expect("apply adapters");
    eprintln!(
        "av lora: applied={} skipped={} (file carries video={video_n} audio/cross={audio_n})",
        report.applied,
        report.skipped.len()
    );
    assert_eq!(
        report.applied,
        video_n + audio_n,
        "full video+audio+cross-modal surface must resolve"
    );
    assert!(
        report.skipped.is_empty(),
        "no target skipped: {:?}",
        report.skipped
    );
    // Three-way agreement: the REFERENCE dump must also have applied everything the file carries
    // (its `applied` count is recorded in the golden metadata at dump time).
    if let Some(dumped) = g.metadata("applied") {
        assert_eq!(
            dumped,
            (video_n + audio_n).to_string(),
            "the reference dump applied a different target count than the lora file carries — \
             stale golden or reference/port routing divergence"
        );
    }

    let upsampler = LatentUpsampler::from_weights(
        &Weights::from_file(dir.join("upsampler.safetensors")).expect("upsampler"),
    )
    .expect("upsampler");
    let vae_w = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae");
    let vae = LtxVideoVae::from_weights(&vae_w, None, &LtxVaeConfig::from_model_dir(&dir).unwrap())
        .expect("vae");
    let mean = vae_w.require("per_channel_statistics.mean").unwrap();
    let std = vae_w.require("per_channel_statistics.std").unwrap();
    let audio_decoder = AudioDecoder::from_weights(
        &Weights::from_file(dir.join("audio_vae.safetensors")).expect("audio_vae"),
        &AudioVaeConfig::from_model_dir(&dir).unwrap(),
    )
    .expect("audio decoder");
    let vcfg = VocoderConfig::from_model_dir(&dir).unwrap();
    let vocoder = LtxVocoder::from_weights(
        &Weights::from_file(dir.join("vocoder.safetensors")).expect("vocoder"),
        &vcfg,
    )
    .expect("vocoder");

    let pos1 = create_position_grid(1, 2, 4, 4);
    let pos2 = create_position_grid(1, 2, 8, 8);
    let apos = create_audio_position_grid(1, 9);

    let mut steps = 0;
    let (vlat, alat) = generate_av_latents(
        &dit,
        &upsampler,
        g.require("video_s1").unwrap(),
        &pos1,
        g.require("video_s2").unwrap(),
        &pos2,
        g.require("audio_s1").unwrap(),
        g.require("audio_s2").unwrap(),
        &apos,
        g.require("video_ctx").unwrap(),
        g.require("audio_ctx").unwrap(),
        mean,
        std,
        &[],
        None, // native distilled Euler (epic 7114 default path)
        0,
        &mlx_gen::CancelFlag::default(),
        &mut |_| steps += 1,
    )
    .expect("generate_av_latents");

    let pr_v = peak_rel(&vlat, g.require("video_latents").unwrap());
    let pr_a = peak_rel(&alat, g.require("audio_latents").unwrap());
    let frames = decode_to_frames(&vae, &vlat, &Default::default()).expect("decode video");
    let px = px_gt8(&frames, g.require("frames").unwrap());
    let track = decode_audio_track(
        &audio_decoder,
        &vocoder,
        &alat,
        vcfg.final_sample_rate() as u32,
        &Default::default(),
    )
    .expect("decode audio");
    let wav = g.require("waveform").unwrap();
    let wsh = wav.shape();
    let want_inter = wav
        .reshape(&[wsh[1], wsh[2]])
        .unwrap()
        .transpose_axes(&[1, 0])
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    let got_wav = Array::from_slice(&track.samples, &[track.samples.len() as i32]);
    let wav_pr = peak_rel(&got_wav, &want_inter);
    let wav_mr = mean_rel(&got_wav, &want_inter);
    eprintln!(
        "av lora e2e: video latents peak_rel {pr_v:.3e} / px>8 {px:.4}% | audio latents peak_rel \
         {pr_a:.3e} / waveform peak_rel {wav_pr:.3e} mean_rel {wav_mr:.3e}"
    );

    // The LoRA residual is a pure per-Linear add; at 0.31.2 the joint denoise reproduced the
    // reference LATENTS bit-exactly. sc-12896/sc-13019 (MLX 0.32.0): the f32 chain carries the
    // cross-stack per-forward ULP drift (see `e2e_parity.rs`), so the latents get tight measured
    // bounds instead of `== 0.0` — the routing/structure gates above stay exact. Measured
    // 2026-07-18 (DR34ML4Y golden, matched non-NAX 0.32.0): video latents peak_rel 3.697e-4 /
    // px>8 0.0000% | audio latents peak_rel 8.397e-5 | waveform peak_rel 4.353e-4 mean_rel
    // 2.075e-4. Latent bounds ~5× the measurements; a mis-scaled or mis-routed lora residual
    // lands O(1) on the latents. px/waveform keep their historical bounds (comfortably held).
    assert!(
        pr_v < 2e-3,
        "LoRA video latents diverged beyond the cross-stack bound: {pr_v:.3e} (sc-13019)"
    );
    assert!(
        pr_a < 5e-4,
        "LoRA audio latents diverged beyond the cross-stack bound: {pr_a:.3e} (sc-13019)"
    );
    assert!(px < 0.5, "video px>8 {px:.4}% too high");
    assert!(
        wav_pr < 1.5e-2,
        "audio waveform peak_rel {wav_pr:.3e} too high"
    );
    assert!(
        wav_mr < 5e-3,
        "audio waveform mean_rel {wav_mr:.3e} too high"
    );
    assert_eq!(track.channels, 2);
    assert_eq!(track.sample_rate, 48000);
}
