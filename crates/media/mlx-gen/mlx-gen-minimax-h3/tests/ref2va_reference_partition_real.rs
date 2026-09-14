//! sc-23402 — the `ref2va` **reference partition** on real weights, two probes:
//!
//! 1. [`cold_second_load_token_embedding_incidence_probe`] — the field defect. On 2026-09-14 a
//!    `reference_to_video` render of `SceneWorks/minimax-h3-mlx@137ce668` (q4 tier, two 576x320
//!    plates) refused with `minimax-h3 te (ref2va token embedding): … every element is exactly
//!    zero at shape [1, 14801, 5120]` on one run and denoised on the next, same plan, same host.
//!    The route is a cold **second** big load in one process — the Qwen3-VL vision tower first,
//!    then the packed token table — and the sc-22414 stale-GPU-view defect zeroes the first Metal
//!    touch of a freshly loaded buffer. The probe performs exactly that sequence and stops at the
//!    token-table gather, in one of two modes:
//!
//!    * `MINIMAX_H3_COLD_MODE=raw` — the pre-sc-23402 path, spelled out: `mlx_gen::quant::embedding`
//!      straight to `TokenEmbedding::forward` on the default stream, no CPU check, no GPU-view
//!      verification (the `Dense`-only guard never matched a packed table). The printed `zero=`
//!      is the raw incidence; the probe does not fail on it, it *measures* it.
//!    * `MINIMAX_H3_COLD_MODE=screened` (default) — the shipped seams after sc-23402: the tower's
//!      read set verified through the map, `MiniMaxH3TextEncoder::from_weights` (which verifies
//!      its read set) and `screened_ref2va_token_embedding`. Any refusal fails the probe;
//!      `coherence::retries()` deltas say whether a divergence was observed and healed.
//!
//!    **One iteration per process**, and the iteration is only meaningful cold: the trigger is a
//!    page cache full of other files when the buffers are allocated (SceneWorks
//!    `docs/calibration/sc-18791/mac2-cold-load/RUNBOOK.md`). Drive it from a loop that sweeps
//!    the cache between processes and counts the `SC23402 …` summary lines — at p≈0.13 a clean
//!    streak of a handful proves nothing (22 clean forwards for 95 %).
//!
//! 2. [`reference_partition_step_cost_against_base`] — the shipped `generate`, one route per
//!    process (`MINIMAX_H3_COST_ROUTE=ref2va|t2va`), at the film-harness geometry (576x320, 124
//!    frames, 24 fps) for a short schedule, with every `Progress` event time-stamped so the
//!    per-step cost of `transformer_ref` under a two-plate presentation can be set against the
//!    base partition's, and the conditioning / DiT-load / decode phases can be read off the same
//!    line. The run cancels itself after the last step unless `MINIMAX_H3_COST_DECODE=1`.
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<root with tokenizer/, vae/, audio_vae/, FL2VA/> \
//! MINIMAX_H3_TE=<tier>/text_encoder MINIMAX_H3_DIT=<tier>/transformer \
//!   cargo test --release -p mlx-gen-minimax-h3 --test integration \
//!   ref2va_reference_partition_real:: -- --ignored --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::gen_core::{
    CancelFlag, Conditioning, Error as GenError, GenerationRequest, LoadPhase, LoadSpec, Progress,
    WeightsSource,
};
use mlx_gen::media::Image;
use mlx_gen_boogu::VisionTower;
use mlx_gen_minimax_h3::model::{load, DIT_COMPONENT, TEXT_ENCODER_COMPONENT};
use mlx_gen_minimax_h3::pipeline::{resolve_geometry, PATCH_SIZE, SPATIAL_STRIDE};
use mlx_gen_minimax_h3::reference::{normalize_reference_image, ReferencePresentation};
use mlx_gen_minimax_h3::text_encoder::{
    self as te, ConditioningDefect, MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MiniMaxH3Tokenizer,
    GROUP_SIZE, LM_PREFIX, VISION_PREFIX,
};
use mlx_gen_minimax_h3::{cost, AUDIO_OUTPUT_CHANNELS};
use mlx_rs::memory::{get_peak_memory, reset_peak_memory};

/// The SH010 prompt of the film-harness fixture the defect was observed on.
const PROMPT: &str = "A quiet, cluttered woodworking workshop in warm late-afternoon light. Tools \
                      hang on pegboard, sawdust on a long wooden workbench in the centre of frame. \
                      A courier in a blue jacket steps into the doorway on the left holding a \
                      small bright red parcel. Static wide shot, natural light, film grain.";

const WIDTH: u32 = 576;
const HEIGHT: u32 = 320;
const FRAMES: u32 = 124;

fn env(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_owned())
}

/// A deterministic 576x320 plate in the shape of the film-harness fixture plates: one flat colour
/// with a darker bench line across the lower third. The plates prove the plumbing, not likeness.
fn plate(seed: u8) -> Image {
    let (w, h) = (576u32, 320u32);
    let base = [
        40u8.wrapping_add(seed.wrapping_mul(37)),
        80u8.wrapping_add(seed.wrapping_mul(53)),
        120u8.wrapping_add(seed.wrapping_mul(71)),
    ];
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        let on_line = (200..212).contains(&y);
        for x in 0..w {
            let shade = if on_line { 60 } else { ((x * 5 / w) as u8) * 3 };
            for c in base {
                pixels.push(c.saturating_sub(shade));
            }
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// `MINIMAX_H3_REF_IMAGES` (comma-separated PNG/JPEG paths) or two synthetic plates.
fn reference_images() -> Vec<Image> {
    match std::env::var("MINIMAX_H3_REF_IMAGES") {
        Ok(list) if !list.trim().is_empty() => list
            .split(',')
            .map(|p| {
                let rgb = image::open(p.trim())
                    .unwrap_or_else(|e| panic!("{p}: {e}"))
                    .to_rgb8();
                Image {
                    width: rgb.width(),
                    height: rgb.height(),
                    pixels: rgb.into_raw(),
                }
            })
            .collect(),
        _ => vec![plate(1), plate(2)],
    }
}

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

fn gb(bytes: usize) -> f64 {
    bytes as f64 / 1e9
}

/// Probe 1 — see the module docs.
#[test]
#[ignore = "sc-23402: needs MINIMAX_H3_SNAPSHOT + MINIMAX_H3_TE and exclusive Metal access; run \
            one cold iteration per process"]
fn cold_second_load_token_embedding_incidence_probe() {
    let root = env("MINIMAX_H3_SNAPSHOT").expect("MINIMAX_H3_SNAPSHOT=<snapshot root>");
    let component = env("MINIMAX_H3_TE").expect("MINIMAX_H3_TE=<text_encoder component dir>");
    let mode = env_or("MINIMAX_H3_COLD_MODE", "screened");
    assert!(
        mode == "raw" || mode == "screened",
        "MINIMAX_H3_COLD_MODE must be raw|screened, got {mode}"
    );
    let tok = MiniMaxH3Tokenizer::from_snapshot(&root).expect("tokenizer");
    // The shipped route resizes every image reference to its own 2048 short edge before the
    // tower sees it — that is where a 576x320 plate becomes ~7 400 vision tokens.
    let images: Vec<Image> = reference_images()
        .iter()
        .map(|i| normalize_reference_image(i, SPATIAL_STRIDE as i32).expect("normalize"))
        .collect();
    let retries = mlx_gen::coherence::retries;

    reset_peak_memory();
    let started = Instant::now();
    let t = Instant::now();
    let mut w = te::map_shards(&component, true).expect("map the tier's shards");
    let t_map = secs(t);

    // --- the FIRST big load: the vision tower ------------------------------------------------
    let t = Instant::now();
    let tower = VisionTower::from_weights(&w, te::minimax_h3_vision_config(), VISION_PREFIX, 64)
        .expect("vision tower");
    let r0 = retries();
    if mode == "screened" {
        w.materialize_accessed().expect("tower read set verified");
    }
    let retries_tower = retries() - r0;
    let sources: Vec<&Image> = images.iter().collect();
    let grounded = te::run_vision(&tower, &sources).expect("tower forward");
    let mut forced: Vec<&mlx_rs::Array> = grounded.embeds.iter().collect();
    forced.extend(grounded.deepstack.iter().flatten());
    mlx_rs::transforms::eval(forced).expect("force the tower output");
    drop(tower);
    w.remove_prefix(VISION_PREFIX);
    let t_tower = secs(t);

    let presentation: Vec<ReferencePresentation> = grounded
        .counts
        .iter()
        .map(|&num_tokens| ReferencePresentation::Image { num_tokens })
        .collect();
    let (ids, mask, tags) = tok
        .encode_ref2va(PROMPT, &presentation)
        .expect("ref2va presentation");
    let tokens = ids.shape()[1];
    assert_eq!(tags.len() as i32, tokens);

    // --- the SECOND big load: the packed token table -----------------------------------------
    let (zero, refusal, retries_te, retries_embed, t_te_build, t_embed, t_forward) =
        match mode.as_str() {
            "raw" => {
                let t = Instant::now();
                let table =
                    mlx_gen::quant::embedding(&w, &format!("{LM_PREFIX}.embed_tokens"), GROUP_SIZE)
                        .expect("token table");
                let hidden = table.forward(&ids).expect("gather");
                let defect = te::inspect_conditioning("ref2va raw token embedding", &hidden)
                    .expect("screen");
                let t_embed = secs(t);
                assert_eq!(hidden.shape(), &[1, tokens, 5120]);
                let zero = matches!(&defect, Some(d) if d.defect == ConditioningDefect::AllZero);
                if let Some(d) = &defect {
                    println!("  raw lookup defect: {d}");
                }
                (zero, false, 0, 0, 0.0, t_embed, 0.0)
            }
            _ => {
                let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
                let r0 = retries();
                let t = Instant::now();
                let encoder = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg)
                    .expect("resident encoder");
                let t_te_build = secs(t);
                let retries_te = retries() - r0;
                assert!(encoder.token_table_is_quantized() || cfg.num_layers > 0);

                let r0 = retries();
                let t = Instant::now();
                let result = encoder.screened_ref2va_token_embedding(&ids);
                let t_embed = secs(t);
                let retries_embed = retries() - r0;
                let (zero, refusal) = match &result {
                    Ok(hidden) => {
                        assert_eq!(hidden.shape(), &[1, tokens, 5120]);
                        (false, false)
                    }
                    Err(e) => {
                        println!("  screened lookup refused: {e}");
                        (
                            e.to_string().contains("every element is exactly zero"),
                            true,
                        )
                    }
                };
                let t_forward = if std::env::var_os("MINIMAX_H3_FULL_FORWARD").is_some() {
                    let t = Instant::now();
                    let context = encoder
                        .forward_with_references(
                            &ids,
                            &mask,
                            &grounded.embeds,
                            &grounded.deepstack,
                            &grounded.grids,
                        )
                        .expect("full ref2va forward");
                    mlx_rs::transforms::eval([&context]).unwrap();
                    secs(t)
                } else {
                    0.0
                };
                (
                    zero,
                    refusal,
                    retries_te,
                    retries_embed,
                    t_te_build,
                    t_embed,
                    t_forward,
                )
            }
        };

    println!(
        "SC23402 mode={mode} zero={zero} refusal={refusal} tokens={tokens} \
         retries_tower={retries_tower} retries_te={retries_te} retries_embed={retries_embed} \
         t_map={t_map:.1} t_tower={t_tower:.1} t_te_build={t_te_build:.1} t_embed={t_embed:.1} \
         t_forward={t_forward:.1} t_total={:.1} peak_gb={:.2}",
        secs(started),
        gb(get_peak_memory())
    );
    if mode == "screened" {
        assert!(
            !refusal,
            "the screened ref2va token embedding must not refuse after sc-23402"
        );
    }
}

/// Probe 2 — see the module docs.
#[test]
#[ignore = "sc-23402: needs MINIMAX_H3_SNAPSHOT + MINIMAX_H3_DIT + MINIMAX_H3_TE and exclusive \
            Metal access"]
fn reference_partition_step_cost_against_base() {
    let root = env("MINIMAX_H3_SNAPSHOT").expect("MINIMAX_H3_SNAPSHOT=<snapshot root>");
    let dit = env("MINIMAX_H3_DIT").expect("MINIMAX_H3_DIT=<tier>/transformer");
    let text_encoder = env("MINIMAX_H3_TE").expect("MINIMAX_H3_TE=<tier>/text_encoder");
    let route = env_or("MINIMAX_H3_COST_ROUTE", "ref2va");
    assert!(
        route == "ref2va" || route == "t2va",
        "MINIMAX_H3_COST_ROUTE must be ref2va|t2va, got {route}"
    );
    let steps: u32 = env_or("MINIMAX_H3_COST_STEPS", "5").parse().expect("steps");
    let keep_decode = std::env::var_os("MINIMAX_H3_COST_DECODE").is_some();

    let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
        .with_component(DIT_COMPONENT, WeightsSource::Dir(dit))
        .with_component(TEXT_ENCODER_COMPONENT, WeightsSource::Dir(text_encoder));
    let generator = load(&spec).expect("load");

    let cancel = CancelFlag::default();
    let conditioning: Vec<Conditioning> = if route == "ref2va" {
        reference_images()
            .into_iter()
            .map(|image| Conditioning::Reference {
                image,
                strength: None,
            })
            .collect()
    } else {
        Vec::new()
    };
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: WIDTH,
        height: HEIGHT,
        frames: Some(FRAMES),
        steps: Some(steps),
        seed: Some(23402),
        cancel: cancel.clone(),
        conditioning,
        ..Default::default()
    };

    // The analytic sequence lengths the two routes pack, so the step ratio can be read against
    // them. Text tokens come from the tokenizer; a reference contributes its vision tokens to the
    // text rows AND the same count again as VAE latent rows (2048 short edge, /16 latent, 2x2
    // patch = /32 both ways).
    let geometry = resolve_geometry(WIDTH, HEIGHT, FRAMES as i32).expect("geometry");
    let tok = MiniMaxH3Tokenizer::from_snapshot(&root).expect("tokenizer");
    let (text_tokens, reference_rows) = if route == "ref2va" {
        let per_image: Vec<i32> = reference_images()
            .iter()
            .map(|i| {
                let n = normalize_reference_image(i, SPATIAL_STRIDE as i32).unwrap();
                ((n.width / 32) * (n.height / 32)) as i32
            })
            .collect();
        let presentation: Vec<ReferencePresentation> = per_image
            .iter()
            .map(|&n| ReferencePresentation::Image {
                num_tokens: n as usize,
            })
            .collect();
        let (ids, ..) = tok.encode_ref2va(PROMPT, &presentation).unwrap();
        (ids.shape()[1], per_image.iter().sum::<i32>())
    } else {
        (tok.encode_prompt(PROMPT).unwrap().0.shape()[1], 0)
    };
    let base_rows = cost::packed_seq_len(
        &geometry.joint,
        PATCH_SIZE,
        text_tokens,
        0,
        i32::from(AUDIO_OUTPUT_CHANNELS),
    )
    .expect("packed sequence length");
    let video_rows = cost::rows_per_latent_frame(&geometry.joint, PATCH_SIZE).unwrap()
        * i64::from(geometry.joint.num_latent_frames);
    println!(
        "  {route}: text rows {text_tokens}, video rows {video_rows} ({} latent frames), \
         reference latent rows {reference_rows}, packed sequence ≈ {}",
        geometry.joint.num_latent_frames,
        base_rows + i64::from(reference_rows)
    );

    let mut marks: Vec<(String, f64, usize)> = Vec::new();
    let started = Instant::now();
    reset_peak_memory();
    let mut on_progress = |p: Progress| {
        let label = match p {
            Progress::Loading(LoadPhase::TextEncoder) => "conditioning-start".to_owned(),
            Progress::Loading(LoadPhase::Renderer) => "dit-load-start".to_owned(),
            Progress::Step { current, total } => {
                if current == total && !keep_decode {
                    cancel.cancel();
                }
                format!("step-{current}/{total}")
            }
            Progress::Decoding => "decode-start".to_owned(),
        };
        marks.push((label, started.elapsed().as_secs_f64(), get_peak_memory()));
    };
    let outcome = generator.generate(&req, &mut on_progress);
    let total = started.elapsed().as_secs_f64();
    match (&outcome, keep_decode) {
        (Ok(_), _) => {}
        (Err(GenError::Canceled), false) => {}
        (Err(e), _) => panic!("{route} generate failed: {e}"),
    }

    println!("── sc-23402 step cost: route={route} {WIDTH}x{HEIGHT} / {FRAMES} frames / {steps} evaluations ──");
    println!(
        "  {:<22} {:>10} {:>10} {:>10}",
        "mark", "t (s)", "Δ (s)", "peak GB"
    );
    let mut previous = 0.0;
    let mut step_durations: Vec<f64> = Vec::new();
    let mut first_step: Option<f64> = None;
    let mut dit_load_start = 0.0;
    // A clock-free count of the `Step` events, for the completeness assertion below.
    let mut steps_reported = 0usize;
    for (label, t, peak) in &marks {
        println!(
            "  {label:<22} {t:>10.1} {:>10.1} {:>10.2}",
            t - previous,
            gb(*peak)
        );
        if label == "dit-load-start" {
            dit_load_start = *t;
        }
        if label.starts_with("step-") {
            steps_reported += 1;
            if first_step.is_none() {
                first_step = Some(t - dit_load_start);
            } else {
                step_durations.push(t - previous);
            }
        }
        previous = *t;
    }
    let mean = if step_durations.is_empty() {
        f64::NAN
    } else {
        step_durations.iter().sum::<f64>() / step_durations.len() as f64
    };
    println!(
        "SC23402COST route={route} steps={steps} dit_load_plus_step1={:.1} mean_step={mean:.1} \
         steps=[{}] total={total:.1} peak_gb={:.2} seq_rows={} text_rows={text_tokens} \
         reference_rows={reference_rows}",
        first_step.unwrap_or(f64::NAN),
        step_durations
            .iter()
            .map(|d| format!("{d:.1}"))
            .collect::<Vec<_>>()
            .join(","),
        gb(get_peak_memory()),
        base_rows + i64::from(reference_rows),
    );
    assert_every_evaluation_reported_a_step(steps_reported, steps);
}

/// The completeness check of probe 2, on two plain counts: the `Step` events observed and the
/// evaluations requested. Nothing here reads a clock.
fn assert_every_evaluation_reported_a_step(reported: usize, requested: u32) {
    assert_eq!(
        reported, requested as usize,
        "every evaluation must have reported a Step"
    );
}
