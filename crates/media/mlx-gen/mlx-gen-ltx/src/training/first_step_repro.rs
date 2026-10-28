#![cfg(test)]

use super::*;
use mlx_gen::media::Image;
use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
use std::path::PathBuf;

const RANK: i32 = 8;
const ALPHA: f32 = 8.0;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("LTX_BASE_DIR") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

/// The Gemma-3 TE dir passed as the (now-required) `LoadSpec::text_encoder` override. sc-13664
/// deleted the production `$LTX_GEMMA_DIR` / HF-cache fallback, so this real-weight harness supplies
/// the path explicitly: `LTX_GEMMA_DIR` is a **test-only** convenience env var, else the base
/// snapshot's sibling `gemma/` dir (the layout the self-contained LTX install ships).
fn gemma_override() -> WeightsSource {
    if let Ok(d) = std::env::var("LTX_GEMMA_DIR") {
        return WeightsSource::Dir(PathBuf::from(d));
    }
    WeightsSource::Dir(snapshot().join("gemma"))
}

/// A solid-colour `edge`×`edge` RGB source (the latent magnitude is irrelevant; the graph size —
/// driven by resolution — is the variable under test).
fn swatch(edge: u32) -> Image {
    let mut img = image::RgbImage::new(edge, edge);
    for px in img.pixels_mut() {
        *px = image::Rgb([180u8, 60, 90]);
    }
    Image {
        width: edge,
        height: edge,
        pixels: img.into_raw(),
    }
}

fn gb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Build a real LTX trainer + LoRA targets, encode one caption into a cached f32 context, then free
/// the Gemma TE + tokenizer (so the measured peaks reflect the post-free training working set, like
/// `train_impl`). Returns the trainer, the targets/params, the cached context, and the per-block
/// target grouping for the checkpointed path.
#[allow(clippy::type_complexity)]
fn build() -> (
    LtxTrainer,
    Vec<LtxLoraTarget>,
    LoraParams,
    Array,
    Vec<Vec<BlockLoraRef>>,
) {
    let te = gemma_override();
    let mut trainer = load_trainer_from_dir(&snapshot(), Some(&te))
        .expect("LTX-2.3 base snapshot (SceneWorks cache or $LTX_BASE_DIR) + Gemma TE");
    let suffixes: Vec<String> = DEFAULT_TARGET_SUFFIXES
        .iter()
        .map(|s| s.to_string())
        .collect();
    let (targets, params) = build_targets(
        &mut trainer.transformer,
        trainer.cfg.num_layers,
        &suffixes,
        RANK,
        7,
    )
    .unwrap();
    let n_layers = trainer.cfg.num_layers as usize;
    let block_targets = group_block_targets(&targets, n_layers);
    let ctx = {
        let te = trainer.text_encoder.as_ref().unwrap();
        let tok = trainer.tokenizer.as_ref().unwrap();
        let (ids, mask) = tok
            .encode("a solid colour swatch", MAX_PROMPT_TOKENS)
            .unwrap();
        to_dtype(&te.encode(&ids, &mask).unwrap(), Dtype::Float32).unwrap()
    };
    eval([&ctx]).unwrap();
    trainer.text_encoder = None;
    trainer.tokenizer = None;
    clear_cache();
    eprintln!(
        "[sc-4942] loaded LTX trainer (TE freed); {} LoRA targets; ctx {:?}",
        targets.len(),
        ctx.shape()
    );
    (trainer, targets, params, ctx, block_targets)
}

/// Run a single first training step at `edge` and report peak GPU memory across forward+backward.
/// Forces the backward (grad eval) — the real step-1 kill point. `checkpoint` selects the
/// block-checkpointed forward; the caller sets the SDPA-checkpoint flag on the transformer.
#[allow(clippy::too_many_arguments)]
fn one_step(
    trainer: &mut LtxTrainer,
    targets: &[LtxLoraTarget],
    params: &LoraParams,
    ctx: &Array,
    block_targets: &[Vec<BlockLoraRef>],
    edge: u32,
    checkpoint: bool,
    dtype: Dtype,
    tag: &str,
) -> Result<(f32, f64, Vec<i32>)> {
    let le = (edge / SPATIAL_SCALE as u32).max(1) as usize;
    let img = center_crop_square(&swatch(edge));
    let prep = preprocess_conditioning_image(&img, edge, edge)?;
    let latent = trainer.vae.encode(&prep)?;
    let clean = flatten_latent(&latent)?;
    eval([&clean])?;
    let positions = create_position_grid(1, 1, le, le);
    let noise = random::normal::<f32>(clean.shape(), None, None, Some(&random::key(1)?))?;
    eval([&noise])?;

    let ck = if checkpoint {
        Some(block_targets)
    } else {
        None
    };
    clear_cache();
    reset_peak_memory();
    let before = get_active_memory();
    let t0 = std::time::Instant::now();
    let (loss, grads) = compute_loss_grads(
        &mut trainer.transformer,
        params,
        targets,
        ALPHA,
        RANK as f32,
        &clean,
        ctx,
        &positions,
        0.5,
        &noise,
        false,
        None,
        ck,
        dtype,
    )?;
    // `compute_loss_grads` only forces the loss (forward). The real trainer forces the backward at
    // the step-1 optimizer `eval`; do the same here so the peak reflects the true working set.
    eval(grads.values())?;
    let secs = t0.elapsed().as_secs_f64();
    let peak = get_peak_memory();
    let shape = clean.shape().to_vec();
    eprintln!(
            "  [edge {edge:>4} {tag}] latent {shape:?}  loss {loss:.5}  active-before {:.2} GB  peak {:.2} GB  step {secs:.2}s",
            gb(before),
            gb(peak)
        );
    Ok((loss, gb(peak), shape))
}

/// Max relative grad diff between two param maps.
fn max_rel_diff(ga: &LoraParams, gb_: &LoraParams) -> f32 {
    let mut max_rel = 0f32;
    for (k, a) in ga {
        let b = gb_.get(k).expect("same keys");
        let num = a.subtract(b).unwrap().abs().unwrap().max(None).unwrap();
        let den = a.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
        max_rel = max_rel.max(num.item::<f32>() / den);
    }
    max_rel
}

/// Grads at `edge` for a given (checkpoint, dtype, sdpa) configuration, backward forced.
#[allow(clippy::too_many_arguments)]
fn grads_of(
    trainer: &mut LtxTrainer,
    targets: &[LtxLoraTarget],
    params: &LoraParams,
    ctx: &Array,
    block_targets: &[Vec<BlockLoraRef>],
    edge: u32,
    checkpoint: bool,
    dtype: Dtype,
) -> LoraParams {
    let le = (edge / SPATIAL_SCALE as u32).max(1) as usize;
    let img = center_crop_square(&swatch(edge));
    let prep = preprocess_conditioning_image(&img, edge, edge).unwrap();
    let latent = trainer.vae.encode(&prep).unwrap();
    let clean = flatten_latent(&latent).unwrap();
    let positions = create_position_grid(1, 1, le, le);
    let noise =
        random::normal::<f32>(clean.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
    eval([&clean, &noise]).unwrap();
    let ck = if checkpoint {
        Some(block_targets)
    } else {
        None
    };
    let (_l, g) = compute_loss_grads(
        &mut trainer.transformer,
        params,
        targets,
        ALPHA,
        RANK as f32,
        &clean,
        ctx,
        &positions,
        0.5,
        &noise,
        false,
        None,
        ck,
        dtype,
    )
    .unwrap();
    eval(g.values()).unwrap();
    g
}

/// sc-4942 — the always-on attention-segment checkpointing must not change the math: grads with the
/// SDPA checkpoint on must match the retained backward (flag off). Same decomposed attention,
/// recomputed instead of retained → (near-)bit-identical.
#[test]
#[ignore = "needs real LTX-2.3 + Gemma weights; run as its own process"]
fn attn_ckpt_grads_match_retained() {
    let (mut trainer, targets, params, ctx, bt) = build();
    let edge = 256u32; // small; the math is resolution-agnostic
    trainer.transformer.set_sdpa_checkpoint(false);
    let g_retained = grads_of(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        edge,
        false,
        Dtype::Float32,
    );
    trainer.transformer.set_sdpa_checkpoint(true);
    let g_ckpt = grads_of(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        edge,
        false,
        Dtype::Float32,
    );
    let max_rel = max_rel_diff(&g_retained, &g_ckpt);
    eprintln!("[sc-4942] attn-ckpt-vs-retained grad max relative diff: {max_rel:.2e}");
    assert!(
        max_rel < 1e-5,
        "attention-segment checkpointing must not change grads: max rel {max_rel:.2e}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED training::first_step_repro::attn_ckpt_grads_match_retained"
    );
}

/// sc-4942 — block (gradient) checkpointing must not change the math: the checkpointed forward+grads
/// must match the dense path within fp tolerance (same install + block forward, recompute-only).
/// This gate also catches the multi-output-VJP duplicate-cotangent bug (each checkpoint returns one
/// distinct array, so it should pass).
#[test]
#[ignore = "needs real LTX-2.3 + Gemma weights; run as its own process"]
fn block_ckpt_grads_match_dense() {
    let (mut trainer, targets, params, ctx, bt) = build();
    let edge = 256u32;
    trainer.transformer.set_sdpa_checkpoint(true);
    let g_dense = grads_of(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        edge,
        false,
        Dtype::Float32,
    );
    trainer.transformer.set_sdpa_checkpoint(false);
    let g_ckpt = grads_of(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        edge,
        true,
        Dtype::Float32,
    );
    let max_rel = max_rel_diff(&g_dense, &g_ckpt);
    eprintln!("[sc-4942] block-ckpt-vs-dense grad max relative diff: {max_rel:.2e}");
    assert!(
        max_rel < 5e-3,
        "block checkpointing must match dense within tolerance: max rel {max_rel:.2e}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED training::first_step_repro::block_ckpt_grads_match_dense"
    );
}

/// sc-4942 — the MEASURED finding that LTX trains f32, not bf16 (the rest of the family casts to
/// bf16; LTX deliberately does not — see the `compute_dtype` note in `train_impl`). bf16 *does*
/// shrink the working set (~30 %), but its gradient DECORRELATES from the f32 (quality) path:
/// global cosine 0.31–0.45, with the early/deep K projections of both attentions pointing
/// ~opposite — the 48-block distilled DiT's chaos-sensitivity (the same reason inference uses
/// `quant_f32`). This test pins that finding (asserts the decorrelation, NOT agreement) so a future
/// change that accidentally re-enables bf16 training is caught, and documents the memory delta that
/// makes the trade unattractive (f32 already fits the video tier). Runs f32 first (the cast is
/// destructive), then casts the same trainer to bf16.
#[test]
#[ignore = "needs real LTX-2.3 + Gemma weights; run as its own process"]
fn bf16_grads_decorrelate_justifying_f32() {
    let (mut trainer, targets, params, ctx, bt) = build();
    trainer.transformer.set_sdpa_checkpoint(true);

    // Memory A/B at 768 (activations dominate the peak).
    let (_, f32_peak, _) = one_step(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        768,
        false,
        Dtype::Float32,
        "attn-ckpt f32",
    )
    .expect("f32 step");
    // Grad reference at 256 in f32.
    let g_f32 = grads_of(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        256,
        false,
        Dtype::Float32,
    );

    trainer
        .transformer
        .cast_weights(Dtype::Bfloat16)
        .expect("cast");
    clear_cache();
    let g_bf16 = grads_of(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        256,
        false,
        Dtype::Bfloat16,
    );

    // Cosine between bf16 and f32 grads (both arrive f32 through the astype VJP). Gate on the GLOBAL
    // cosine (the concatenated gradient the optimizer follows) and the large-norm minimum; tiny-norm
    // params whose direction bf16 rounding scrambles contribute nothing to the update.
    let mut per: Vec<(String, f32, f32)> = Vec::new(); // (key, cos, na)
    let (mut gdot, mut gna2, mut gnb2) = (0f64, 0f64, 0f64);
    for (k, a) in &g_f32 {
        let b = g_bf16.get(k).expect("same keys");
        let dot = a.multiply(b).unwrap().sum(None).unwrap().item::<f32>();
        let na2 = a.square().unwrap().sum(None).unwrap().item::<f32>();
        let nb2 = b.square().unwrap().sum(None).unwrap().item::<f32>();
        gdot += dot as f64;
        gna2 += na2 as f64;
        gnb2 += nb2 as f64;
        let (na, nb) = (na2.sqrt(), nb2.sqrt());
        if na > 1e-12 && nb > 1e-12 {
            per.push((k.to_string(), dot / (na * nb), na));
        }
    }
    let global_cos = (gdot / (gna2.sqrt() * gnb2.sqrt())) as f32;
    per.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
    let max_norm = per.iter().map(|p| p.2).fold(0f32, f32::max);
    eprintln!("[sc-4942] bf16-vs-f32 grads: global cosine {global_cos:.5}; worst per-param:");
    for (k, c, na) in per.iter().take(5) {
        eprintln!(
            "    {k}: cos {c:.4}  |g| {na:.3e}  rel-norm {:.2e}",
            na / max_norm
        );
    }
    let min_large = per
        .iter()
        .filter(|p| p.2 >= 0.01 * max_norm)
        .map(|p| p.1)
        .fold(1f32, f32::min);
    eprintln!("[sc-4942] min cosine among params with |g| >= 1% of max: {min_large:.4}");
    // The FINDING (not a regression): bf16 grads do NOT track f32 on this distilled stack. Pin it
    // loosely (global cosine well under the family's >0.99 bar) so an accidental re-enable is caught.
    assert!(
        global_cos < 0.9,
        "expected bf16 to decorrelate from f32 on the LTX distilled stack (the reason LTX trains \
             f32); if this is now high, bf16 training may be viable — re-evaluate: {global_cos:.5}"
    );

    let (_, bf16_peak, _) = one_step(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        768,
        false,
        Dtype::Bfloat16,
        "attn-ckpt bf16",
    )
    .expect("bf16 step");
    // Informational: bf16 does shrink the working set — but f32 already fits the tier, so the
    // quality cost above is not worth taking.
    eprintln!(
            "[sc-4942] 768 peak f32 {f32_peak:.2} GB vs bf16 {bf16_peak:.2} GB ({:.0}%) — f32 fits the \
             video tier, so bf16's saving is not needed",
            100.0 * bf16_peak / f32_peak
        );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::first_step_repro::bf16_grads_decorrelate_justifying_f32");
}

/// sc-4942 — first-step peak sweep on the dense path (attention-segment checkpointing always on),
/// f32 then bf16, plus a block-ckpt point. These measured points are the basis of the
/// `projected_dense_peak_gb` guard fit — refit the constants if this prints materially different
/// numbers.
#[test]
#[ignore = "needs real LTX-2.3 + Gemma weights; run as its own process (may SIGKILL at large edge)"]
fn first_step_attn_ckpt_sweep() {
    let (mut trainer, targets, params, ctx, bt) = build();
    trainer.transformer.set_sdpa_checkpoint(true);
    eprintln!("[sc-4942] attn-ckpt dense sweep, f32:");
    for edge in [512u32, 768, 1024] {
        let _ = one_step(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            edge,
            false,
            Dtype::Float32,
            "attn-ckpt f32",
        )
        .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
    }
    eprintln!("[sc-4942] block-ckpt at 1024, f32:");
    trainer.transformer.set_sdpa_checkpoint(false);
    let _ = one_step(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        1024,
        true,
        Dtype::Float32,
        "blk-ckpt f32",
    )
    .map_err(|e| eprintln!("  blk-ckpt CATCHABLE error: {e}"));

    eprintln!("[sc-4942] casting weights to bf16…");
    trainer
        .transformer
        .cast_weights(Dtype::Bfloat16)
        .expect("cast");
    trainer.transformer.set_sdpa_checkpoint(true);
    clear_cache();
    eprintln!("[sc-4942] attn-ckpt dense sweep, bf16:");
    for edge in [512u32, 768, 1024] {
        let _ = one_step(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            edge,
            false,
            Dtype::Bfloat16,
            "attn-ckpt bf16",
        )
        .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
    }
    eprintln!("[sc-4942] block-ckpt + bf16 at 1024:");
    trainer.transformer.set_sdpa_checkpoint(false);
    let _ = one_step(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        1024,
        true,
        Dtype::Bfloat16,
        "blk-ckpt bf16",
    )
    .map_err(|e| eprintln!("  blk-ckpt bf16 CATCHABLE error: {e}"));
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED training::first_step_repro::first_step_attn_ckpt_sweep"
    );
}

/// sc-4942 — block checkpointing must drop the first-step peak below the dense path at production
/// resolution. Runs the dense step first (baseline), then the checkpointed step.
#[test]
#[ignore = "needs real LTX-2.3 + Gemma weights; run as its own process"]
fn block_ckpt_reduces_peak_vs_dense() {
    let (mut trainer, targets, params, ctx, bt) = build();
    trainer.transformer.set_sdpa_checkpoint(true);
    let (_, dense_peak, _) = one_step(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        1024,
        false,
        Dtype::Float32,
        "dense",
    )
    .expect("dense step");
    trainer.transformer.set_sdpa_checkpoint(false);
    let (_, ckpt_peak, _) = one_step(
        &mut trainer,
        &targets,
        &params,
        &ctx,
        &bt,
        1024,
        true,
        Dtype::Float32,
        "blk-ckpt",
    )
    .expect("checkpointed step");
    eprintln!(
        "[sc-4942] edge 1024  dense {dense_peak:.2} GB  ckpt {ckpt_peak:.2} GB  ({:.0}% reduction)",
        100.0 * (1.0 - ckpt_peak / dense_peak)
    );
    assert!(
            ckpt_peak < dense_peak,
            "block checkpointing must reduce the first-step peak: dense {dense_peak:.2} vs ckpt {ckpt_peak:.2}"
        );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::first_step_repro::block_ckpt_reduces_peak_vs_dense");
}
