//! sc-18789 **A/B evidence run**: one fast-motion clip rendered with and without DFR on real
//! LTX-2.5 tier weights, outputs + a detail-retention metric written to disk.
//!
//! The DFR pipeline exists for detail retention under fast motion (v1.2.0 changelog: "holds detail
//! together noticeably better on fast motion"). This check renders the same seed/prompt/geometry
//! twice — the plain two-stage distilled path, then the DFR path (canvas keyframe slots + the
//! full-res detailing re-denoise + one temporal x2 round by default) — decodes both with the conv
//! VAE, and records per-arm **high-frequency (Laplacian) luma energy** over time-matched frames
//! plus sample frames as PPMs. The numbers and images are the A/B record; the assertions here are
//! shape/finiteness sanity, not a quality gate (measurements are once-per-epic evidence, not CI).
//!
//! **Fail-closed**: run explicitly (`--ignored`), a missing env var is a hard panic, never a
//! silent green.
//!
//! ```bash
//! LTX25_TIER_DIR=/Volumes/Models/scratch-tiers-sc18775/tiers \
//! LTX25_AB_OUT=/tmp/dfr_ab \
//! cargo test -p mlx-gen-ltx --release --test integration -- dfr_ab_fast_motion:: \
//!   --ignored --nocapture
//! ```
//!
//! Optional: `LTX25_AB_TIER` (default `q4`), `LTX25_AB_ROUNDS` (default `1`, `0..=2`).

use std::path::PathBuf;

use mlx_rs::memory::clear_cache;
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_ltx::config::{LtxConfig, LtxVaeConfig, SplitModel};
use mlx_gen_ltx::dfr::{generate_dfr_av_latents, DfrComponents, DfrRequest};
use mlx_gen_ltx::gemma4_te::Ltx25TextEncoder;
use mlx_gen_ltx::pipeline::{generate_av_latents, to_uint8_frames};
use mlx_gen_ltx::positions::{
    compute_audio_frames, create_audio_position_grid, create_position_grid,
};
use mlx_gen_ltx::tokenizer::Ltx25Tokenizer;
use mlx_gen_ltx::transformer::{AvDiT, Precision};
use mlx_gen_ltx::upsampler::LatentUpsampler;
use mlx_gen_ltx::vae::LtxVideoVae;

const PROMPT: &str = "a hummingbird hovering at a red trumpet flower, wings beating in a fast \
                      blur, rapid darting motion, sunlit garden, highly detailed feathers, \
                      fast camera pan following the bird";
const MAX_LEN: usize = 256;
const WIDTH: u32 = 448;
const HEIGHT: u32 = 256;
const FRAMES: i64 = 121;
const FPS: f32 = 24.0;
const SEED: u64 = 618;

fn required_env(name: &str) -> PathBuf {
    let Some(v) = std::env::var_os(name) else {
        panic!(
            "{name} must be set for the sc-18789 A/B run (fail-closed: this check never \
             silently passes)"
        );
    };
    PathBuf::from(v)
}

fn tier_dir() -> PathBuf {
    let tier = std::env::var("LTX25_AB_TIER").unwrap_or_else(|_| "q4".into());
    let dir = required_env("LTX25_TIER_DIR").join(tier);
    assert!(dir.is_dir(), "tier dir does not exist: {}", dir.display());
    dir
}

fn rounds() -> u32 {
    let r: u32 = std::env::var("LTX25_AB_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    assert!(r <= 2, "LTX25_AB_ROUNDS must be 0..=2, got {r}");
    r
}

fn seeded_noise(shape: &[i32], seed: u64) -> Array {
    let key = mlx_rs::random::key(seed).unwrap();
    mlx_rs::random::normal::<f32>(shape, None, None, Some(&key)).unwrap()
}

/// Luma at (y, x) of an RGB8 frame.
fn luma_at(frame: &[u8], w: usize, y: usize, x: usize) -> f64 {
    let p = (y * w + x) * 3;
    0.299 * frame[p] as f64 + 0.587 * frame[p + 1] as f64 + 0.114 * frame[p + 2] as f64
}

/// Motion-mask luma threshold: a pixel belongs to the moving region when its luma changed by more
/// than this between the sample frame and its predecessor (out of 255).
const MOTION_LUMA_THRESHOLD: f64 = 8.0;

/// Mean absolute discrete-Laplacian response over the luma plane, **restricted to the moving
/// region** — the pixels whose luma changed by more than [`MOTION_LUMA_THRESHOLD`] versus `prev`
/// (frame-difference gate). Detail retention under fast motion is what DFR exists for, so the
/// static background must not dilute the number. Returns `(masked_hf_energy, mask_coverage)`;
/// a frame with no moving pixels reports 0 energy at 0 coverage rather than a NaN.
fn motion_masked_hf_energy(frame: &[u8], prev: &[u8], h: usize, w: usize) -> (f64, f64) {
    let mut acc = 0.0;
    let mut n = 0usize;
    let mut total = 0usize;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            total += 1;
            let moved =
                (luma_at(frame, w, y, x) - luma_at(prev, w, y, x)).abs() > MOTION_LUMA_THRESHOLD;
            if !moved {
                continue;
            }
            let lap = 4.0 * luma_at(frame, w, y, x)
                - luma_at(frame, w, y - 1, x)
                - luma_at(frame, w, y + 1, x)
                - luma_at(frame, w, y, x - 1)
                - luma_at(frame, w, y, x + 1);
            acc += lap.abs();
            n += 1;
        }
    }
    if n == 0 {
        (0.0, 0.0)
    } else {
        (acc / n as f64, n as f64 / total as f64)
    }
}

fn write_ppm(path: &PathBuf, frame: &[u8], h: usize, w: usize) {
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    out.extend_from_slice(frame);
    std::fs::write(path, out).expect("write ppm");
}

/// Decode a video latent to uint8 frames and pull them to host memory as per-frame RGB8 buffers.
fn decode_frames(vae: &LtxVideoVae, latent: &Array) -> (Vec<Vec<u8>>, usize, usize) {
    let decoded = vae.decode(latent).expect("vae decode");
    let frames = to_uint8_frames(&decoded).expect("to_uint8_frames");
    eval([&frames]).expect("eval frames");
    let sh = frames.shape().to_vec(); // (F, H, W, 3)
    let (f, h, w) = (sh[0] as usize, sh[1] as usize, sh[2] as usize);
    let flat: Vec<u8> = frames.as_slice::<u8>().to_vec();
    let per = h * w * 3;
    (
        (0..f)
            .map(|i| flat[i * per..(i + 1) * per].to_vec())
            .collect(),
        h,
        w,
    )
}

#[test]
#[ignore = "sc-18789: real-weight A/B on the built LTX-2.5 tiers (LTX25_TIER_DIR) + Metal"]
fn dfr_ab_fast_motion_records_detail_retention() {
    let dir = tier_dir();
    let out_dir = required_env("LTX25_AB_OUT");
    std::fs::create_dir_all(&out_dir).expect("create output dir");
    let rounds = rounds();

    let cfg = LtxConfig::from_model_dir(&dir).expect("tier config");
    let split = SplitModel::from_model_dir(&dir).expect("split_model.json");
    let prec = Precision::quant_bf16(split.bits, split.group);
    assert!(
        cfg.use_keyframes_abs_pos_embedding,
        "the 2.5 checkpoint must carry the keyframe-slot marker — DFR is not optional for it"
    );

    // --- Text phase (encoder loads, encodes, drops before the DiT materializes) -----------------
    let te_path = dir.join("text_encoder.safetensors");
    let connector_w = Weights::from_file(dir.join("connector.safetensors")).expect("connector");
    let checkpoint = mlx_gen::gen_core::ltx_checkpoint::LtxCheckpointMetadata::from_file(
        dir.join("transformer.safetensors"),
    )
    .expect("transformer metadata");
    let te = Ltx25TextEncoder::from_packed_av(
        &checkpoint,
        &te_path,
        &connector_w,
        &cfg,
        prec,
        // sc-18798: this A/B measures DFR motion, not memory — keep the historical resident
        // encoder so nothing here is attributed to residency.
        mlx_gen::gen_core::OffloadPolicy::Resident,
    )
    .expect("build the 2.5 text encoder");
    let tok = Ltx25Tokenizer::from_packed_te_file(&te_path).expect("packed tokenizer");
    let (input_ids, mask) = tok.encode(PROMPT, MAX_LEN).expect("tokenize");
    let (_, _, video_ctx, audio_ctx) = te
        .encode_av_with_features(&input_ids, &mask)
        .expect("encode prompt");
    eval([&video_ctx, &audio_ctx]).expect("materialize contexts");
    drop(te);
    drop(connector_w);
    clear_cache();

    // --- Geometry + shared inputs ----------------------------------------------------------------
    let lf = ((FRAMES - 1) / 8 + 1) as i32; // 16
    let (h1, w1) = ((HEIGHT / 64) as i32, (WIDTH / 64) as i32); // 4 x 7
    let (h2, w2) = (2 * h1, 2 * w1);
    let af = compute_audio_frames(FRAMES as usize, FPS as f64);
    let pos1 = create_position_grid(1, lf as usize, h1 as usize, w1 as usize);
    let pos2 = create_position_grid(1, lf as usize, h2 as usize, w2 as usize);
    let audio_pos = create_audio_position_grid(1, af);
    let v1_noise = seeded_noise(&[1, 128, lf, h1, w1], SEED);
    let v2_noise = seeded_noise(&[1, 128, lf, h2, w2], SEED.wrapping_add(1));
    let a1_noise = seeded_noise(&[1, 8, af as i32, 16], SEED.wrapping_add(2));
    let a2_noise = seeded_noise(&[1, 8, af as i32, 16], SEED.wrapping_add(3));

    let vae_cfg = LtxVaeConfig::from_model_dir(&dir).expect("vae config");
    let dec_w = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae_decoder");
    let enc_w = Weights::from_file(dir.join("vae_encoder.safetensors")).expect("vae_encoder");
    let vae = LtxVideoVae::from_weights(&dec_w, Some(&enc_w), &vae_cfg).expect("conv vae");
    // The VAE per_channel_statistics double as the upsampler latent norm (same read the 2.3
    // engine load performs).
    let latent_mean = dec_w
        .require("per_channel_statistics.mean")
        .expect("latent mean")
        .clone();
    let latent_std = dec_w
        .require("per_channel_statistics.std")
        .expect("latent std")
        .clone();
    drop(dec_w);
    drop(enc_w);
    let spatial = LatentUpsampler::from_checkpoint(dir.join("spatial_upsampler.safetensors"))
        .expect("spatial");
    let temporal = LatentUpsampler::from_checkpoint(dir.join("temporal_upsampler.safetensors"))
        .expect("temporal");

    let dit_w = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
    let dit = AvDiT::from_weights(&dit_w, &cfg, prec).expect("build the 22B AvDiT");
    drop(dit_w);
    clear_cache();

    let cancel = CancelFlag::default();
    let mut steps = 0usize;
    let mut on_step = |_s: usize| {
        steps += 1;
        eprint!(".");
    };

    // --- Arm A: plain two-stage distilled (no DFR) -----------------------------------------------
    eprintln!("[A/B] baseline two-stage ({WIDTH}x{HEIGHT}, {FRAMES} frames)");
    let t0 = std::time::Instant::now();
    let (v_base, _a_base) = generate_av_latents(
        &dit,
        &spatial,
        &v1_noise,
        &pos1,
        &v2_noise,
        &pos2,
        &a1_noise,
        &a2_noise,
        &audio_pos,
        &video_ctx,
        &audio_ctx,
        &latent_mean,
        &latent_std,
        &[],
        None,
        SEED,
        &cancel,
        &mut on_step,
    )
    .expect("baseline two-stage");
    eval([&v_base]).expect("materialize baseline");
    eprintln!(
        "\n[A/B] baseline denoise: {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // --- Arm B: DFR (canvas slots + detailing re-denoise + temporal rounds) ---------------------
    let (canvas_frames, _segment, keyframe_positions) =
        mlx_gen::gen_core::ltx_dfr::resolve_canvas(FRAMES, 8).expect("canvas");
    assert_eq!(canvas_frames, FRAMES, "121 frames needs no canvas padding");
    let parts = DfrComponents {
        dit: &dit,
        spatial_upsampler: &spatial,
        temporal_upsampler: Some(&temporal),
        latent_mean: &latent_mean,
        latent_std: &latent_std,
        video_ctx: &video_ctx,
        audio_ctx: &audio_ctx,
        audio_pos: &audio_pos,
    };
    let req = DfrRequest {
        canvas_frames,
        requested_frames: FRAMES,
        keyframe_positions: &keyframe_positions,
        fps: FPS,
        seed: SEED,
        temporal_upsample_rounds: rounds,
        detailing_downscale: None,
        video_keyframes: &[],
    };
    eprintln!(
        "[A/B] DFR ({} keyframe slots, {rounds} temporal rounds)",
        keyframe_positions.len()
    );
    let t0 = std::time::Instant::now();
    let dfr = generate_dfr_av_latents(
        &parts,
        &req,
        &v1_noise,
        &pos1,
        &v2_noise,
        &pos2,
        &a1_noise,
        &a2_noise,
        &cancel,
        &mut on_step,
    )
    .expect("dfr pipeline");
    eval([&dfr.video_latent]).expect("materialize dfr");
    eprintln!("\n[A/B] dfr denoise: {:.1}s", t0.elapsed().as_secs_f64());
    assert_eq!(
        dfr.num_frames,
        (FRAMES - 1) * (1 << rounds) + 1,
        "rounds must multiply the frame contract"
    );
    assert_eq!(dfr.playback_fps, FPS * (1 << rounds) as f32);

    drop(dit);
    clear_cache();

    // --- Decode + metric -------------------------------------------------------------------------
    let (base_frames, bh, bw) = decode_frames(&vae, &v_base);
    let (dfr_frames, dh, dw) = decode_frames(&vae, &dfr.video_latent);
    assert_eq!((bh, bw), (dh, dw), "both arms decode at the target size");
    assert_eq!(base_frames.len(), FRAMES as usize);
    assert_eq!(dfr_frames.len(), dfr.num_frames as usize);

    // Time-matched samples: baseline frame t vs DFR frame t·2^rounds, each frame's motion mask
    // gated against its own predecessor on the same timeline (t ≥ 1 so a predecessor exists).
    let stride = 1usize << rounds;
    let samples: Vec<usize> = (1..=8)
        .map(|i| 1 + (i - 1) * (FRAMES as usize - 2) / 7)
        .collect();
    let mut base_hf = 0.0;
    let mut dfr_hf = 0.0;
    let mut base_cov = 0.0;
    let mut dfr_cov = 0.0;
    for (i, &t) in samples.iter().enumerate() {
        let b = &base_frames[t];
        let d = &dfr_frames[t * stride];
        let (bhf, bc) = motion_masked_hf_energy(b, &base_frames[t - 1], bh, bw);
        let (dhf, dc) = motion_masked_hf_energy(d, &dfr_frames[t * stride - stride], dh, dw);
        base_hf += bhf;
        dfr_hf += dhf;
        base_cov += bc;
        dfr_cov += dc;
        write_ppm(&out_dir.join(format!("base_f{t:03}_{i}.ppm")), b, bh, bw);
        write_ppm(
            &out_dir.join(format!("dfr_f{:03}_{i}.ppm", t * stride)),
            d,
            dh,
            dw,
        );
    }
    base_hf /= samples.len() as f64;
    dfr_hf /= samples.len() as f64;
    base_cov /= samples.len() as f64;
    dfr_cov /= samples.len() as f64;
    assert!(
        base_hf.is_finite() && base_hf > 0.0,
        "baseline has no moving detail at all"
    );
    assert!(
        dfr_hf.is_finite() && dfr_hf > 0.0,
        "dfr has no moving detail at all"
    );

    let summary = format!(
        "{{\"prompt\":{PROMPT:?},\"seed\":{SEED},\"size\":\"{WIDTH}x{HEIGHT}\",\
         \"frames_base\":{FRAMES},\"frames_dfr\":{},\"rounds\":{rounds},\
         \"metric\":\"motion-masked luma Laplacian (|dL|>{MOTION_LUMA_THRESHOLD} gate)\",\
         \"motion_hf_base\":{base_hf:.4},\"motion_hf_dfr\":{dfr_hf:.4},\
         \"motion_hf_ratio_dfr_over_base\":{:.4},\
         \"motion_coverage_base\":{base_cov:.4},\"motion_coverage_dfr\":{dfr_cov:.4}}}",
        dfr.num_frames,
        dfr_hf / base_hf
    );
    std::fs::write(out_dir.join("dfr_ab_metrics.json"), &summary).expect("write metrics");
    eprintln!("[A/B] {summary}");
    eprintln!(
        "[A/B] outputs in {} (P6 PPM frames + dfr_ab_metrics.json); total forwards observed: {steps}",
        out_dir.display()
    );
}
