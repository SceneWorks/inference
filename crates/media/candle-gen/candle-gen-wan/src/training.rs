//! The Candle **Wan2.2 LoRA trainer** — the Candle twin of `mlx-gen-wan`'s
//! `WanMoeTrainer`, implementing the backend-neutral [`gen_core::Trainer`](candle_gen::gen_core::train::Trainer)
//! with `backend = "candle"`. It retires the worker's Python torch `WanMoeLoraTrainer` (the dual-expert
//! path of epic 5164) and reuses the shared [`candle_gen::train`] harness the SDXL/Z-Image stories
//! established, building on [`crate::dit_train`]'s vendored trainable DiT.
//!
//! Registered for dense TI2V-5B and both A14B MoE variants. T2V-A14B retains its established
//! LoRA/LoKr contract; TI2V-5B and I2V-A14B are deliberately LoRA-only.
//!
//! ## Why Wan keeps a bespoke loop (sc-7787)
//!
//! The single-model flow-match driver ([`candle_gen::train::flow_match::run_flow_match_training`]) the
//! Z-Image/Lens/Krea trainers adopted assumes **one** DiT, optimizer, adapter set, and timestep range.
//! The Wan A14B is a **dual-expert MoE**: it alternates a high-noise expert (`transformer/`) and a
//! low-noise expert (`transformer_2/`), each with its own LoRA/LoKr, optimizer, LR schedule, timestep
//! band, and gradient-accumulation buffer, and emits an expert-suffixed save pair. That does not fit the
//! single-model driver cleanly, so Wan keeps the alternating loop below and consumes only the **Tier-1
//! shared helpers** from [`flow_match`] (sampling, batch math, validation, snapshot IO, adapter install
//! + optimizer step). Exactly the split sc-7787 sanctions.
//!
//! ## The Wan realities that shape it
//!
//! Cache → loop → save, on the **flow-match** objective. The two Wan-specific twists vs the Z-Image
//! trainer:
//!  1. **No velocity negation.** Wan feeds the transformer output to the flow-match step *without*
//!     negation (opposite of Z-Image's `noise_pred.neg()`), so the trainer regresses the **raw** DiT
//!     velocity toward `noise − x0` (`target = noise − x0`). The timestep fed to the DiT is `t · 1000`
//!     (the `[0, NUM_TRAIN_TIMESTEPS]` integer convention), not `1 − σ` — see `compute_loss_grads`.
//!  2. **MoE dual-expert.** The A14B denoises with a **high-noise** expert (`transformer/`, timestep
//!     ≥ `boundary·1000`) and a **low-noise** expert (`transformer_2/`, below it). Each gets its **own**
//!     LoRA/LoKr (separate factor map + optimizer + LR schedule + timestep band). Training **alternates**
//!     per step (odd → high, even → low), sampling that expert's band, and emits a `{stem}.high_noise` /
//!     `{stem}.low_noise` pair — what the inference loader ([`crate::adapters`] via [`crate::wan14b`])
//!     merges back onto the matching expert ([`MoeExpert`](candle_gen::gen_core::MoeExpert)).
//!
//! Caching: each still image is z16-VAE-encoded to a single-frame latent `[1, 16, 1, h, w]` (the
//! deterministic posterior **mean**, normalized) and its caption UMT5-encoded to `[1, 512, 4096]`
//! (zero-padded to 512, the same context surface inference feeds). The VAE + text encoder are dropped
//! after caching; the two experts are the working set.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, Tensor, Var};

use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::gen_core::train::subject_mask::{CropBox, PreparedSubjectMask};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{
    self, BucketSchedule, Image, LoadSpec, Modality, NetworkType, Progress, WeightsSource,
};
use candle_gen::train::checkpoint::file_stem;
use candle_gen::train::dataset::{bucket_edges, decode_square, square_image_tensor, SquareImage};
use candle_gen::train::flow_match::{
    self, prepared_subject_mask_weight, validate_flow_match_request, weighted_velocity_loss,
};
use candle_gen::train::flow_match::{combine_terms, StepLosses};
use candle_gen::train::gradient_checkpoint::checkpointed_backward;
use candle_gen::train::lora::{LoraHost, LoraSet};
use candle_gen::train::optim::{accumulate_grads, TrainOptimizer};
use candle_gen::train::perceptual::{
    plan_step, AuxAlternation, AuxLossSchedule, Parameterization, PerceptualPath, StepPlan,
};
use candle_gen::train::schedule::schedule_updates;
use candle_gen::train::taehv::TaehvConfig;
use candle_gen::{CandleError, Result};

use crate::config::{
    TextEncoderConfig, TransformerConfig, Vae16Config, VaeConfig, I2V_14B_BOUNDARY, MODEL_ID,
    MODEL_ID_I2V_14B, MODEL_ID_T2V_14B, NUM_TRAIN_TIMESTEPS, T2V_14B_BOUNDARY, T2V_14B_FLOW_SHIFT,
    VAE16_STRIDE_SPATIAL,
};
use crate::dit_train::{WanTransformerTrain, WAN_ATTN_TARGETS};
use crate::pipeline::{create_noise, frames_to_images};
use crate::rope::WanRope;
use crate::scheduler::flow_sigmas;
use crate::text_encoder::Umt5Encoder;
use crate::vae::WanVae;
use crate::vae16::WanVae16;

/// Error-message prefix shared by [`validate_flow_match_request`] and the component-IO helpers.
const LABEL: &str = "wan trainer";

/// Sample a timestep `t ∈ [lo, hi)` inside an expert's noise band: draw a unit `t_unit`
/// ([`flow_match::sample_unit_timestep`]) then affine-map it into `(lo, hi)`. The high-noise expert
/// samples `(boundary, 1)`, the low-noise `(0, boundary)` — the per-expert split the A14B trains. This
/// band map is Wan-specific (the other flow-match trainers train one model over the full `(0, 1)`), so
/// it stays local; only the unit draw is shared.
fn sample_band_timestep(
    timestep_type: &str,
    timestep_bias: &str,
    band: (f64, f64),
    seed: u64,
) -> f64 {
    let t_unit = flow_match::sample_unit_timestep(timestep_type, timestep_bias, seed) as f64;
    let (lo, hi) = band;
    (lo + t_unit * (hi - lo)).clamp(1e-3, 1.0 - 1e-3)
}

/// Which expert trains at `step` (1-based): odd steps → high-noise (`experts[0]`), even → low-noise
/// (`experts[1]`). The two experts alternate so both accumulate roughly `steps/2` micro-steps.
#[inline]
fn expert_index(step: u32, dual: bool) -> usize {
    if !dual || step % 2 == 1 {
        0
    } else {
        1
    }
}

/// The 0-based sample counter the expert training at `step` (1-based) is on — the value fed to the
/// [`BucketSchedule`] (see [`expert_cache_index`]).
///
/// The counter is the expert's own visit count `(step - 1) / 2`, **not** the raw step. Because the
/// experts alternate by step parity ([`expert_index`]), indexing by `step` couples the item parity
/// to the expert parity: for an even-sized dataset `(step - 1) % N` keeps the same parity forever, so
/// the high-noise expert would only ever see even-indexed items and the low-noise expert only
/// odd-indexed ones — each adapter silently training on a disjoint half of the user's images
/// (sc-11157 / F-082). Advancing by the per-expert visit count instead makes each expert walk the
/// full dataset in order (`0, 1, 2, …` cycling at `N`), independent of its parity.
#[inline]
fn expert_sample_counter(step: u32, dual: bool) -> usize {
    let experts = if dual { 2 } else { 1 };
    ((step - 1) / experts) as usize
}

/// The item-major cache entry (`item * n_buckets + bucket`, sc-2127) consumed at `step` (1-based):
/// the expert's own [`expert_sample_counter`] walked through `schedule`. With one resolution bucket
/// the schedule is `counter % n_items`, i.e. exactly the pre-bucket per-expert round-robin.
#[inline]
fn expert_cache_index(step: u32, dual: bool, schedule: &BucketSchedule) -> usize {
    schedule.cache_index(expert_sample_counter(step, dual))
}

/// One `(cos, sin)` RoPE table pair per resolution bucket (sc-2127), derived from the cached latent
/// geometry of the first item at each bucket (`cache[b]`; every item shares a bucket's geometry, so
/// the table for cache entry `i` is `ropes[i % n_buckets]`). Training stills ⇒ one latent frame.
fn bucket_rope_tables(
    cache: &[(Tensor, Tensor, Option<Tensor>)],
    n_buckets: usize,
    dit_cfg: &TransformerConfig,
    device: &Device,
) -> Result<Vec<(Tensor, Tensor)>> {
    let (pt, ph, pw) = dit_cfg.patch;
    let rope = WanRope::new(dit_cfg);
    cache
        .iter()
        .take(n_buckets)
        .map(|(x0, _, _)| {
            let (_, _, fl, hl, wl) = x0.dims5()?;
            Ok(rope.cos_sin(fl / pt, hl / ph, wl / pw, device)?)
        })
        .collect()
}

/// One item's item-major cache entries (sc-2127 × sc-24828): the decoded `square` encoded at each
/// bucket edge by `encode` (`[1, 3, edge, edge]` → clean `[1, C, 1, h, w]` latent), each paired with
/// its subject-mask loss weight on THAT bucket's latent grid (`None` when masked loss is off).
/// `decode_square` center-crops, so the mask is cropped with [`CropBox::center_square`].
fn encode_item_buckets(
    square: &SquareImage,
    edges: &[u32],
    mask: Option<&PreparedSubjectMask>,
    device: &Device,
    mut encode: impl FnMut(&Tensor) -> Result<Tensor>,
) -> Result<Vec<(Tensor, Option<Tensor>)>> {
    edges
        .iter()
        .map(|&edge| {
            let x0 = encode(&square_image_tensor(square, edge, device)?)?;
            let mask_weight = prepared_subject_mask_weight(
                LABEL,
                mask,
                CropBox::center_square,
                x0.dims(),
                device,
            )?;
            Ok((x0, mask_weight))
        })
        .collect()
}

#[inline]
fn pending_micro_count(micro_steps: u32, configured: u32) -> Option<u32> {
    let pending = micro_steps % configured.max(1);
    (pending > 0).then_some(pending)
}

#[inline]
fn use_checkpointed_backward(cfg: &TrainingConfig) -> bool {
    cfg.gradient_checkpointing
}

#[inline]
fn expert_schedule_inputs(steps: u32, expert_count: u32, warmup_steps: u32) -> (u32, u32) {
    ((steps / expert_count.max(1)).max(1), warmup_steps)
}

/// One micro-step's forward+backward over one expert's installed adapter `Var`s: build the noised
/// latent at `t`, predict the **raw** velocity through the (LoRA-adapted) DiT, regress it toward
/// `noise − x0`, return `(loss, grads)` keyed by `lora_vars`. `cos`/`sin` are the (constant,
/// per-resolution) RoPE tables. A free function so the tests can drive it against a tiny DiT.
///
/// `use_checkpoint` selects the **gradient-checkpointed** backward. This is not just a memory lever for
/// the 14B experts — it is **required**: candle's matmul backward materializes a gradient for the
/// *frozen* base weight as well as the activation, so a dense 40-block backward holds ~40 layers of f32
/// base-weight grads at once (tens of GB), OOMing even a 96 GB card with both experts resident. The
/// checkpointed path runs the (adapter-free) pre-main forward detached, then segments the per-block
/// stack so only one block's transient weight-grads are live at a time (see
/// [`WanTransformerTrain::main_block_segments`]). Both paths yield the same adapter grads (the
/// `dense_and_checkpoint_grads_match` test pins this on a tiny DiT).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &WanTransformerTrain,
    lora_vars: &[Var],
    x0: &Tensor,
    umt5: &Tensor,
    t: f64,
    noise: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mae: bool,
    mask_weight: Option<&Tensor>,
    compute_dtype: DType,
    use_checkpoint: bool,
    y_channels: usize,
) -> Result<(f32, GradStore)> {
    let (losses, grads) = compute_step_loss_grads(
        dit,
        lora_vars,
        x0,
        umt5,
        t,
        noise,
        cos,
        sin,
        mae,
        mask_weight,
        compute_dtype,
        use_checkpoint,
        y_channels,
        None,
    )?;
    Ok((losses.total, grads))
}

/// One planned perceptual step of an expert (epic 2123 E8): the shared path, the step's plan (with
/// its windows confined to the expert's band, [`plan_in_band`]) and its (item, bucket) entry.
struct AuxStep<'a> {
    path: &'a PerceptualPath,
    plan: StepPlan,
    entry: usize,
}

/// The step's loss terms on the expert's raw velocity `v` (epic 2123 E8): the (subject-mask
/// weighted) velocity regression when the diffusion term contributes, plus — on a planned step with
/// aux losses — the weighted perceptual term on the x0 estimate `x_t − t·v` (the raw velocity
/// regresses `noise − x0`), each latent frame decoded by the TAEHV decoder. No aux step ⇒ exactly the
/// legacy loss tensor.
#[allow(clippy::too_many_arguments)]
fn step_loss(
    v: &Tensor,
    target: &Tensor,
    x_t_latent: &Tensor,
    t: f64,
    mask_weight: Option<&Tensor>,
    mae: bool,
    aux: Option<&AuxStep<'_>>,
) -> Result<(Tensor, StepLosses)> {
    let (diffusion_on, aux_on) = aux.map_or((true, false), |a| {
        (a.plan.diffusion, !a.plan.aux.is_empty())
    });
    let diffusion = if diffusion_on {
        Some(weighted_velocity_loss(v, target, mask_weight, mae)?)
    } else {
        None
    };
    let aux_term = match aux {
        Some(a) if aux_on => {
            let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma: t as f32 }
                .recover_x0(x_t_latent, &v.to_dtype(DType::F32)?)?;
            a.path
                .aux_loss(&a.plan, a.entry, &latent_frames_nchw(&x0_hat)?)?
                .map(|t| t.weighted)
        }
        _ => None,
    };
    combine_terms(diffusion, aux_term)
}

/// [`compute_loss_grads`] with the step's planned perceptual terms (epic 2123 E8): an aux-only step
/// does not compute the diffusion term; both the dense and the checkpointed backward (the aux term
/// rides in the final checkpoint segment, after the velocity head) carry it. `aux = None` ⇒ the
/// legacy graph.
#[allow(clippy::too_many_arguments)]
fn compute_step_loss_grads(
    dit: &WanTransformerTrain,
    lora_vars: &[Var],
    x0: &Tensor,
    umt5: &Tensor,
    t: f64,
    noise: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mae: bool,
    mask_weight: Option<&Tensor>,
    compute_dtype: DType,
    use_checkpoint: bool,
    y_channels: usize,
    aux: Option<&AuxStep<'_>>,
) -> Result<(StepLosses, GradStore)> {
    let (x_t, target) = flow_match::build_batch(x0, noise, t)?;
    let x_t_latent = x_t.to_dtype(DType::F32)?;
    let mut x_t = x_t.to_dtype(compute_dtype)?;
    if y_channels > 0 {
        // The still image is the clean training target, not a separate I2V source. Match the MLX
        // trainer by zero-padding the inference-only source-conditioning channels: Wan LoRA targets
        // attention projections downstream of patch embedding, so their trained surface is independent
        // of these padded channels. A separately supplied `control_image_path` is rejected in
        // `validate` and can therefore never be silently discarded here.
        let (batch, _, frames, height, width) = x_t.dims5()?;
        let zeros = Tensor::zeros(
            (batch, y_channels, frames, height, width),
            compute_dtype,
            x_t.device(),
        )?;
        x_t = Tensor::cat(&[&x_t, &zeros], 1)?;
    }
    // Text context + timestep are adapter-free constants the blocks consume.
    let ctx = dit.embed_text(umt5)?;
    let timestep = t * NUM_TRAIN_TIMESTEPS as f64;

    if use_checkpoint {
        // The breakdown of the final segment's last evaluation, kept for reporting.
        let breakdown = std::cell::Cell::new(None);
        // Pre-main (patch + time embed) has no adapters → run it detached; no upstream grads to stitch.
        let (hidden, mctx) = dit.forward_pre_main(&x_t, timestep)?;
        let hidden_d = hidden.detach();
        let mut segs = dit.main_block_segments(&mctx, &ctx, cos, sin);
        // Final segment: head → raw velocity (NO negation) → flow-match regression (+ the step's aux
        // term) → [loss].
        let target_owned = target.clone();
        let mctx_ref = &mctx;
        let x_t_ref = &x_t_latent;
        let breakdown_ref = &breakdown;
        segs.push(Box::new(move |st: &[Tensor]| {
            let v = dit.velocity_out(&st[0], mctx_ref)?;
            let (loss, losses) = step_loss(&v, &target_owned, x_t_ref, t, mask_weight, mae, aux)
                .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
            breakdown_ref.set(Some(losses));
            Ok(vec![loss])
        }));
        let (loss_val, grads) =
            checkpointed_backward(&segs, std::slice::from_ref(&hidden_d), lora_vars)?;
        drop(segs);
        let losses = breakdown.get().ok_or_else(|| {
            CandleError::Msg(format!("{LABEL}: the final checkpoint segment never ran"))
        })?;
        Ok((
            StepLosses {
                total: loss_val,
                ..losses
            },
            grads,
        ))
    } else {
        // Dense backward (tiny models / tests only — see the `use_checkpoint` note re: OOM at scale).
        let v = dit.forward(&x_t, &ctx, timestep, cos, sin)?;
        let (loss, losses) = step_loss(&v, &target, &x_t_latent, t, mask_weight, mae, aux)?;
        let grads = loss.backward()?;
        Ok((losses, grads))
    }
}

/// A cached `[B, C, T, h, w]` latent (`T = 1` for a still) as the decoder's NCHW batch
/// `[B·T, C, h, w]` — each latent frame decoded independently.
/// Build every cache entry's perceptual reference once (item-major entries, each clean `[B, C, T,
/// h, w]` latent as the decoder's NCHW frame batch), handing the path the job's subject masks
/// first (sc-24832) so a mask-reading loss gets each entry's item mask on its decoded grid.
fn prepare_perceptual_references<'a>(
    path: &mut PerceptualPath,
    clean: impl Iterator<Item = &'a Tensor>,
    masks: Option<candle_gen::gen_core::train::subject_mask::PerceptualSubjectMasks>,
) -> Result<()> {
    path.attach_subject_masks(masks);
    for (entry, x0) in clean.enumerate() {
        path.ensure_reference(entry, &latent_frames_nchw(x0)?)?;
    }
    Ok(())
}

fn latent_frames_nchw(latent: &Tensor) -> Result<Tensor> {
    let (b, c, t, h, w) = latent.dims5()?;
    Ok(latent
        .permute((0, 2, 1, 3, 4))?
        .reshape((b * t, c, h, w))?
        .contiguous()?)
}

/// The TAEHV variant decoding the trainer's latent family: `taew2_1` for the z16 Wan 2.1 VAE
/// (T2V/I2V-A14B), `taew2_2` for the z48 Wan 2.2 VAE (TI2V-5B). Both decode the VAE's
/// per-channel-normalized posterior mean — exactly the cached latent the DiT regresses; TAEHV
/// applies no scale/shift of its own.
fn wan_decoder(variant: TrainVariant) -> candle_gen_perceptual::DecoderSpec {
    let config = TaehvConfig::for_wan_z_dim(variant.latent_channels())
        .expect("every Wan training variant is a z16 or z48 VAE");
    candle_gen_perceptual::DecoderSpec::Taehv {
        name: if config.latent_channels == 48 {
            "TAEW2.2"
        } else {
            "TAEW2.1"
        },
        config,
    }
}

/// The epic-2123 perceptual path through the shared builder: `None` when no aux loss is enabled.
fn load_perceptual_path(
    cfg: &TrainingConfig,
    variant: TrainVariant,
    device: &Device,
) -> Result<Option<PerceptualPath>> {
    candle_gen_perceptual::build_perceptual_path(
        cfg,
        &candle_gen_perceptual::AuxLossContext {
            label: LABEL,
            decoder: wan_decoder(variant),
            device,
            latent_lpips: None,
        },
    )
}

/// Epic 2123 E7: refuse a depth job whose resident experts (`base_bytes`, their on-disk weights —
/// the lower bound the aux models stack on) plus the TAEHV decoder + the losses' frozen models at
/// the largest bucket (one decoded frame per step, `entries` cached references) exceeds
/// `budget_bytes`. No-op when no aux loss is enabled; independent of the dense/checkpointed choice,
/// so it guards both paths.
fn check_perceptual_memory(
    cfg: &TrainingConfig,
    variant: TrainVariant,
    entries: usize,
    base_bytes: u64,
    budget_bytes: u64,
) -> Result<()> {
    let edge = bucket_edges(cfg).iter().copied().max().unwrap_or(0);
    let aux = candle_gen_perceptual::perceptual_footprint(
        cfg,
        &wan_decoder(variant),
        candle_gen_perceptual::AuxGeometry::image(edge, entries),
    );
    if aux == 0 {
        return Ok(());
    }
    flow_match::check_aux_memory(LABEL, base_bytes, aux, budget_bytes)
}

/// The dataset item `step` trains on (the item half of [`expert_cache_index`]'s `(item, bucket)`) —
/// what the perceptual alternation keys on (epic 2123 E8).
fn expert_item(step: u32, dual: bool, schedule: &BucketSchedule) -> usize {
    schedule.sample(expert_sample_counter(step, dual)).0
}

/// The perceptual alternation over `items` dataset items (epic 2123 E8): one window is one optimizer
/// update of every expert — `accum` micro-steps per expert, the experts interleaving by step parity,
/// so `accum · n_experts` global micro-steps — so each expert's update is all diffusion or all aux.
fn perceptual_alternation(items: usize, accum: u32, n_experts: usize) -> AuxAlternation {
    AuxAlternation::new(items, accum * n_experts as u32)
}

/// The perceptual plan of one micro-step with every aux loss's window confined to the routed
/// expert's noise `band` (epic 2123 E8): an expert only ever trains at noise levels inside its band,
/// so an aux-only step lands in `window ∩ band`, and an expert whose band misses a loss's window never
/// trains that loss (its claims fall through to diffusion). `t` is the band-sampled level; on an
/// aux-only step its position within the band is remapped into the confined window by the shared
/// [`plan_step`] policy, and a diffusion step keeps `t`. With the full band `(0, 1)` (the dense 5B)
/// this is exactly [`PerceptualPath::plan`].
fn plan_in_band(
    path: &PerceptualPath,
    key: u32,
    entry: usize,
    band: (f64, f64),
    t: f64,
) -> Result<StepPlan> {
    let (lo, hi) = (band.0 as f32, band.1 as f32);
    let t = t as f32;
    let schedules: Vec<AuxLossSchedule> = path
        .losses()
        .iter()
        .map(|l| {
            let s = l.schedule;
            let (a, b) = (s.t_min.max(lo), s.t_max.min(hi));
            if a <= b {
                AuxLossSchedule {
                    t_min: a,
                    t_max: b,
                    ..s
                }
            } else {
                AuxLossSchedule { weight: 0.0, ..s }
            }
        })
        .collect();
    let usable = (0..schedules.len())
        .map(|i| path.is_usable(entry, i))
        .collect::<Result<Vec<bool>>>()?;
    let u = if hi > lo {
        ((t - lo) / (hi - lo)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    // Whether the step is claimed depends only on `key`; a diffusion step keeps the sampled `t`.
    let claimed = plan_step(&schedules, key, u);
    let plan = if claimed.diffusion {
        plan_step(&schedules, key, t)
    } else {
        claimed
    };
    Ok(plan.without_skipped(|i| !usable[i]))
}

/// Tokenize + UMT5-encode `caption` → `[1, 512, 4096]` (f32, zero-padded to 512 — the same context
/// surface inference feeds; see [`crate::wan14b`]'s `encode`).
///
/// Shared Wan text-encode routine (sc-9000 / F-020). The trainer loads the UMT5 encoder at **bf16**
/// (unlike the inference providers' f32), so it passes `out_dtype = F32` to reproduce its prior
/// `.to_dtype(F32)` upcast of the bf16 embeds exactly.
fn encode_caption(
    tok: &TextTokenizer,
    te_cfg: &TextEncoderConfig,
    te: &Umt5Encoder,
    caption: &str,
    device: &Device,
) -> Result<Tensor> {
    crate::text_encode::umt5_encode_padded(
        tok,
        te_cfg,
        te,
        caption,
        device,
        DType::F32,
        "wan trainer",
    )
}

/// Insert `.{suffix}` before the extension of `file_name` (`a.safetensors` → `a.high_noise.safetensors`).
fn with_expert_suffix(file_name: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        return file_name.to_string();
    }
    match file_name.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}.{suffix}.{ext}"),
        None => format!("{file_name}.{suffix}"),
    }
}

/// One MoE expert's full trainable state: the (vendored) DiT with adapters installed, its optimizer +
/// LR schedule, its timestep band, and its own gradient-accumulation buffer + step counters.
struct ExpertState {
    dit: WanTransformerTrain,
    set: LoraSet,
    opt: TrainOptimizer,
    band: (f64, f64),
    accumulated: Option<GradStore>,
    micro: u32,
    update_idx: u32,
    total_updates: u32,
    warmup_updates: u32,
    /// Epic 2123 (sc-24827): this expert's adapter-noise RNG seed — the same per-expert seed its
    /// factors were initialised from, so the experts draw independent weight/gradient noise.
    noise_seed: u64,
    /// `"high_noise"` / `"low_noise"` — the saved-file suffix + the [`MoeExpert`] the inference loader
    /// merges this onto.
    suffix: &'static str,
}

/// Visit every preview adapter and preserve the first visitor failure. Preview rendering must not
/// continue after a model traversal fails: that would leave a partially frozen/thawed expert set
/// and hide a structural model error behind a best-effort preview warning (F-035 / sc-21704).
fn visit_preview_lora_hosts<'a, H: LoraHost + 'a>(
    hosts: impl IntoIterator<Item = &'a mut H>,
    visitor: &mut dyn FnMut(&mut candle_gen::train::lora::LoraLinear) -> Result<()>,
) -> Result<()> {
    for host in hosts {
        host.visit_lora_mut(visitor)?;
    }
    Ok(())
}

fn freeze_preview_lora_hosts<'a, H: LoraHost + 'a>(
    hosts: impl IntoIterator<Item = &'a mut H>,
) -> Result<()> {
    visit_preview_lora_hosts(hosts, &mut |ll| {
        ll.freeze_adapter();
        Ok(())
    })
}

fn thaw_preview_lora_hosts<'a, H: LoraHost + 'a>(
    hosts: impl IntoIterator<Item = &'a mut H>,
) -> Result<()> {
    visit_preview_lora_hosts(hosts, &mut |ll| {
        ll.thaw_adapter();
        Ok(())
    })
}

/// Cap on the number of preview prompts rendered per [`TrainingConfig::sample_every`] cadence
/// (sc-8650) — matches the shared `SAMPLE_PROMPT_CAP` the FlowMatchTrainer driver applies. A Wan still
/// is a full dual-expert denoise + z16 VAE decode, so this keeps the preview cost bounded.
const SAMPLE_PROMPT_CAP: usize = 4;

/// The Wan preview-sample render state (sc-8650) — everything [`render_one_preview`] needs to render a
/// single still frame on the **in-progress** experts (both carry their live LoRA/LoKr adapters), built
/// once in [`WanMoeTrainer::train_impl`]'s cache phase while the UMT5 encoder is still resident and
/// before the VAE encoder is dropped.
///
///  * `prompts` — the prompt strings (≤ [`SAMPLE_PROMPT_CAP`]) reported on each
///    [`TrainingProgress::Sample`], 1:1 with `umt5`.
///  * `umt5` — the per-prompt pre-encoded UMT5 caption embeds `[1, 512, 4096]` (f32, the exact context
///    surface [`encode_caption`] produces), projected per-expert inside the render via
///    [`WanTransformerTrain::embed_text`].
///  * `vae` — a resident **decode-only** [`WanVae16`] (the cache pass loads the VAE *with* the encoder
///    for latent caching, then drops it; the preview path needs only the decoder).
///  * `edge` — the square edge the seeded single-frame preview noise is shaped at: the largest
///    training bucket edge (`bucket_edges(cfg)` max — just the floored `cfg.resolution` when buckets
///    are off, the same edge the cached latents use).
struct WanSampleState {
    prompts: Vec<String>,
    umt5: Vec<Tensor>,
    vae: WanVae16,
    edge: u32,
}

/// Render ONE still preview frame (sc-8650) from the **in-progress** dual-expert A14B onto an RGB8
/// [`Image`], best-effort. This mirrors the inference T2V denoise ([`crate::wan14b`]'s `render`) but on
/// a **single-frame** latent (`F = 1`, the cheapest faithful still) and CFG-free.
///
/// **Dual-expert (not single-expert).** Both experts are resident with their live adapters during the
/// loop, so the preview runs the *real* MoE boundary-band schedule — the high-noise expert
/// (`experts[0]`) while the σ-derived integer timestep is `≥ boundary·1000`, the low-noise expert
/// (`experts[1]`) below it — exactly as inference switches them. That is the simplest faithful option
/// (a single-expert preview would denoise the whole trajectory through one band's adapter, which no
/// inference path does) and it exercises *both* trained adapters, so a preview reflects the full LoRA.
///
/// Conventions match [`compute_loss_grads`] / inference: the DiT consumes the **raw** velocity (no sign
/// flip) at timestep `σ · NUM_TRAIN_TIMESTEPS` (the `[0, 1000]` integer convention), driven over Wan's
/// native flow-σ schedule by [`candle_gen::run_flow_sampler`] with [`TimestepConvention::Sigma`]. The
/// schedule uses `steps = cfg.sample_steps.max(1)` and the T2V flow shift, and a fresh never-cancel
/// flag (a preview need not honor cancel mid-denoise; `req.cancel` is not threaded here).
///
/// CFG-free: training pre-encodes only the positive caption, so `cfg.sample_guidance_scale` is ignored
/// (the Wan trainer trains the velocity directly, no negative branch is cached).
fn render_one_preview(
    experts: &[ExpertState],
    state: &WanSampleState,
    index: usize,
    cfg: &TrainingConfig,
    seed: u64,
    device: &Device,
) -> Result<Image> {
    let dit_cfg = TransformerConfig::t2v_14b();
    let steps = cfg.sample_steps.max(1) as usize;
    // The experts are built at the bf16 compute dtype; their `forward` does NOT cast inputs (like
    // `compute_loss_grads`, which casts `x_t` → `compute_dtype`), so run the denoise in that dtype — the
    // F32 `create_noise` prior would otherwise hit an F32×BF16 matmul mismatch on the first block.
    let compute_dtype = flow_match::parse_compute_dtype(&cfg.train_dtype);

    // Single-frame latent geometry (F = 1) at the square training edge, z16 strides.
    let t_lat = 1usize;
    let h_lat = (state.edge / VAE16_STRIDE_SPATIAL) as usize;
    let w_lat = (state.edge / VAE16_STRIDE_SPATIAL) as usize;
    let (pt, ph, pw) = dit_cfg.patch;
    let (ppf, pph, ppw) = (t_lat / pt, h_lat / ph, w_lat / pw);
    let (cos, sin) = WanRope::new(&dit_cfg).cos_sin(ppf, pph, ppw, device)?;

    // Per-expert projected text context (each expert owns its own `condition_embedder.text_embedder`).
    let umt5 = &state.umt5[index];
    let ctx_high = experts[0].dit.embed_text(umt5)?;
    let ctx_low = experts[1].dit.embed_text(umt5)?;

    // Seeded [1, 16, 1, h_lat, w_lat] noise (the inference `create_noise`, frame axis pinned to 1).
    let noise = create_noise(seed, 16, t_lat, h_lat, w_lat, device)?.to_dtype(compute_dtype)?;

    // Wan's native flow-σ schedule (descending, trailing 0.0) integrated by the shared euler flow
    // sampler — the dual-expert MoE band switch lives inside the closure (boundary on σ·N).
    let boundary_ts = T2V_14B_BOUNDARY * NUM_TRAIN_TIMESTEPS as f64;
    let sigmas = flow_sigmas(steps, T2V_14B_FLOW_SHIFT);
    let cancel = CancelFlag::new();
    let mut on_progress = |_: Progress| {};
    let lat = candle_gen::run_flow_sampler(
        Some("euler"),
        TimestepConvention::Sigma,
        &sigmas,
        noise,
        seed,
        &cancel,
        &mut on_progress,
        None,
        |x, sigma| -> Result<Tensor> {
            // σ → integer timestep (σ·1000); MoE: high-noise expert at/above the boundary, low below.
            let ts = sigma as f64 * NUM_TRAIN_TIMESTEPS as f64;
            let (dit, ctx) = if ts >= boundary_ts {
                (&experts[0].dit, &ctx_high)
            } else {
                (&experts[1].dit, &ctx_low)
            };
            let v = dit.forward(x, ctx, ts, &cos, &sin)?; // raw velocity (no negation)
            Ok(v.to_dtype(compute_dtype)?)
        },
    )?;

    // Decode the single-frame latent [1,16,1,h,w] → [1,3,1,8h,8w]; the frame axis (dim 2) carries one
    // frame, so `frames_to_images` (the inference frame→Image conversion) yields exactly one Image. The
    // denoise ran in `compute_dtype`; the resident VAE is F32 → cast back before decode.
    let decoded = state.vae.decode(&lat.to_dtype(DType::F32)?)?; // [1, 3, 1, H, W] in [-1, 1]
    frames_to_images(&decoded)?
        .into_iter()
        .next()
        .ok_or_else(|| CandleError::Msg("wan: preview decode yielded no frame".into()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrainVariant {
    T2v14b,
    I2v14b,
    Ti2v5b,
}

impl TrainVariant {
    fn id(self) -> &'static str {
        match self {
            Self::T2v14b => MODEL_ID_T2V_14B,
            Self::I2v14b => MODEL_ID_I2V_14B,
            Self::Ti2v5b => MODEL_ID,
        }
    }

    fn descriptor(self) -> TrainerDescriptor {
        TrainerDescriptor {
            id: self.id(),
            family: "wan",
            backend: "candle",
            modality: Modality::Video,
            supports_lora: true,
            supports_lokr: self == Self::T2v14b,
            supports_control: false,
            supports_full_finetune: false,
            max_reference_images: 0,
            // Epic 2123 S2 (sc-24827): weight noise + gradient noise at the adapter optimizer
            // update.
            // sc-2127 (epic 2123): spatial multi-resolution buckets — one cached still latent per
            // bucket edge, per-bucket RoPE, sampled through `BucketSchedule`.
            // Subject-masked loss (sc-24828): every item is a still frame, weighted on both experts.
            // Depth anchoring (sc-24830): on every expert, dense and checkpointed — the shared
            // decoded-x0 perceptual path through TAEHV (taew2_1 z16 / taew2_2 z48), each expert's
            // aux steps confined to its own noise band.
            techniques: gen_core::train::TrainingTechniques {
                resolution_buckets: true,
                subject_mask_loss: true,
                depth_anchoring: true,
                // sc-24831: the face losses ride the same shared builder arms + x0 decoder.
                identity_loss: true,
                face_landmark_loss: true,
                // sc-24832: the body losses ride the same builder arms as depth anchoring
                // (decoded-x0 pixel losses through this trainer's x0 decoder).
                body_proportion_loss: true,
                body_shape_loss: true,
                normal_loss: true,
                ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
            },
        }
    }

    fn dit_config(self) -> TransformerConfig {
        match self {
            Self::T2v14b => TransformerConfig::t2v_14b(),
            Self::I2v14b => TransformerConfig::i2v_14b(),
            Self::Ti2v5b => TransformerConfig::ti2v_5b(),
        }
    }

    fn dual(self) -> bool {
        self != Self::Ti2v5b
    }

    fn latent_channels(self) -> usize {
        if self == Self::Ti2v5b {
            48
        } else {
            16
        }
    }

    fn boundary(self) -> f64 {
        match self {
            Self::T2v14b => T2V_14B_BOUNDARY,
            Self::I2v14b => I2V_14B_BOUNDARY,
            Self::Ti2v5b => 0.0,
        }
    }
}

enum TrainVae {
    Z16(WanVae16),
    Z48(WanVae),
}

impl TrainVae {
    fn load(variant: TrainVariant, root: &std::path::Path, device: &Device) -> Result<Self> {
        let vb = flow_match::component_vb(root, "vae", device, DType::F32, LABEL)?;
        Ok(match variant {
            TrainVariant::Ti2v5b => Self::Z48(WanVae::new_with_encoder(&VaeConfig::ti2v_5b(), vb)?),
            TrainVariant::T2v14b | TrainVariant::I2v14b => {
                Self::Z16(WanVae16::new_with_encoder(&Vae16Config::wan21(), vb)?)
            }
        })
    }

    fn encode(&self, video: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Self::Z16(vae) => vae.encode(video),
            Self::Z48(vae) => vae.encode(video),
        }?)
    }
}

/// Identity + capabilities of the candle Wan A14B trainer: LoRA + LoKr, `backend = "candle"`.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainVariant::T2v14b.descriptor()
}

pub fn trainer_descriptor_i2v_14b() -> TrainerDescriptor {
    TrainVariant::I2v14b.descriptor()
}

pub fn trainer_descriptor_ti2v_5b() -> TrainerDescriptor {
    TrainVariant::Ti2v5b.descriptor()
}

/// A loaded candle Wan A14B (T2V) MoE trainer. Loading is **lazy** — the heavy VAE / text-encoder / two
/// experts are built inside [`train`](Trainer::train) at the request's compute dtype.
pub struct WanMoeTrainer {
    descriptor: TrainerDescriptor,
    variant: TrainVariant,
    root: PathBuf,
    device: Device,
}

/// Construct the (lazy) candle Wan A14B trainer from a [`LoadSpec`] whose `weights` is the A14B snapshot
/// directory (`tokenizer/ text_encoder/ transformer/ transformer_2/ vae/`).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_variant(spec, TrainVariant::T2v14b)
}

pub fn load_trainer_i2v_14b(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_variant(spec, TrainVariant::I2v14b)
}

pub fn load_trainer_ti2v_5b(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_variant(spec, TrainVariant::Ti2v5b)
}

fn load_variant(spec: &LoadSpec, variant: TrainVariant) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(CandleError::Msg(format!(
                "{} trainer expects a snapshot directory, not a single .safetensors file",
                variant.id()
            )))
        }
    };
    if spec.quantize.is_some() {
        return Err(CandleError::Msg(format!(
            "{} trainer requires dense weights; explicit quantization is unsupported",
            variant.id()
        )));
    }
    let components: &[&str] = if variant.dual() {
        &["transformer", "transformer_2"]
    } else {
        &["transformer"]
    };
    for component in components {
        if packed_component(&root, component)? {
            return Err(CandleError::Msg(format!(
                "{} trainer requires dense weights; {component} is physically packed/quantized",
                variant.id()
            )));
        }
    }
    Ok(Box::new(WanMoeTrainer {
        descriptor: variant.descriptor(),
        variant,
        root,
        device: candle_gen::default_device()?,
    }))
}

fn packed_component(root: &std::path::Path, component: &str) -> Result<bool> {
    let component_dir = root.join(component);
    let config_path = component_dir.join("config.json");
    let bytes = match std::fs::read(&config_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => {
            return Err(CandleError::Msg(format!(
                "{LABEL}: read {}: {error}",
                config_path.display()
            )))
        }
    };
    if !bytes.is_empty() {
        let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
            CandleError::Msg(format!("{LABEL}: parse {}: {error}", config_path.display()))
        })?;
        if candle_gen::quant::PackedConfig::from_config(&config).is_some() {
            return Ok(true);
        }
    }
    if !component_dir.is_dir() {
        return Ok(false);
    }
    for entry in std::fs::read_dir(&component_dir).map_err(|error| {
        CandleError::Msg(format!(
            "{LABEL}: read {}: {error}",
            component_dir.display()
        ))
    })? {
        let path = entry
            .map_err(|error| CandleError::Msg(format!("{LABEL}: read directory entry: {error}")))?
            .path();
        if path.extension().and_then(|value| value.to_str()) != Some("safetensors") {
            continue;
        }
        // SAFETY: header-only, read-only mapping of a process-owned checkpoint file.
        let tensors =
            unsafe { candle_gen::candle_core::safetensors::MmapedSafetensors::new(&path)? };
        if tensors
            .tensors()
            .into_iter()
            .any(|(name, _)| name.ends_with(".scales"))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

// `register_trainer!` defines the explicit trainer constant and bridges the crate's rich `Result`
// into `gen_core::Result` via `Into::into`.
candle_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}
candle_gen::register_trainer! {
    pub(crate) const I2V_14B_TRAINER_REGISTRATION =
        trainer_descriptor_i2v_14b => load_trainer_i2v_14b
}
candle_gen::register_trainer! {
    pub(crate) const TI2V_5B_TRAINER_REGISTRATION =
        trainer_descriptor_ti2v_5b => load_trainer_ti2v_5b
}

impl Trainer for WanMoeTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Shared full-base-fine-tune floor (sc-14056): an adapter-only trainer must reject a
        // `full_finetune` request (typed `Unsupported`) rather than silently training a LoRA
        // adapter the caller did not ask for (F-006/F-055).
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
        // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        gen_core::train::validate_edit_request(self.descriptor(), req)?;
        if !self.descriptor.supports_lokr && req.config.network_type == NetworkType::Lokr {
            return Err(gen_core::Error::Unsupported(format!(
                "{} trainer is LoRA-only",
                self.variant.id()
            )));
        }
        if req.config.resume {
            return Err(gen_core::Error::Unsupported(format!(
                "{} trainer does not yet support resume",
                self.variant.id()
            )));
        }
        if req
            .items
            .iter()
            .any(|item| item.control_image_path.is_some())
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{} trainer does not consume per-item control/source images",
                self.variant.id()
            )));
        }
        if self.variant != TrainVariant::T2v14b
            && req.config.sample_every > 0
            && !req.config.sample_prompts.is_empty()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{} trainer does not support in-training previews",
                self.variant.id()
            )));
        }
        validate_flow_match_request(req, LABEL).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        // Epic 2123 E3: refuse an unsupported technique at the `train` entry point too, before
        // any loading/caching — a caller that skips `validate` must not get it silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        self.validate(req)?;
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl WanMoeTrainer {
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        validate_flow_match_request(req, LABEL)?;
        let cfg = &req.config;
        let device = &self.device;
        on_progress(TrainingProgress::Preparing);
        // sc-2127 — one training edge per resolution bucket (just the floored `resolution` when buckets
        // are off); the preview renders at the largest. Frame count is untouched (stills, T = 1).
        let edges = bucket_edges(cfg);
        let max_edge = edges.iter().copied().max().unwrap_or(0);
        let compute_dtype = flow_match::parse_compute_dtype(&cfg.train_dtype);
        let variant = self.variant;
        let dit_cfg = variant.dit_config();
        let dual = variant.dual();
        let y_channels = dit_cfg.in_channels - variant.latent_channels();

        // Epic 2123 E7: the depth-anchoring models count against the device budget on both paths
        // (no-op with nothing enabled), BEFORE any load or caching.
        if candle_gen_perceptual::any_aux_loss(cfg) {
            let components: &[&str] = if dual {
                &["transformer", "transformer_2"]
            } else {
                &["transformer"]
            };
            let base = components
                .iter()
                .map(|c| flow_match::component_bytes(&self.root, c, LABEL))
                .sum::<Result<u64>>()?;
            check_perceptual_memory(
                cfg,
                variant,
                req.items.len() * edges.len(),
                base,
                flow_match::device_training_budget_bytes(device, LABEL),
            )?;
        }
        // Epic 2123 depth anchoring (sc-24830): the frozen TAEHV + Depth-Anything-V2 load before the
        // caching pass, so a missing checkpoint fails fast.
        let mut perceptual = load_perceptual_path(cfg, variant, device)?;

        // --- load + cache: z16 VAE latent means + UMT5 caption embeds (both f32) ---
        on_progress(TrainingProgress::LoadingModel);
        let vae = TrainVae::load(variant, &self.root, device)?;
        let te_cfg = TextEncoderConfig::umt5_xxl();
        // Load the UMT5 encoder at bf16 for caching (11 GB vs 22 GB at f32) — it only produces the
        // cached caption embeds (kept f32 in the cache), and is dropped before the experts load.
        let text_encoder = Umt5Encoder::new(
            &te_cfg,
            flow_match::component_vb(&self.root, "text_encoder", device, DType::BF16, LABEL)?,
        )?;
        // Load+parse the UMT5 tokenizer ONCE, not once per dataset item / sample prompt (sc-8991 /
        // F-011). Byte-identical config to the old per-caption `from_file`, so the cached ids match.
        let tokenizer =
            crate::text_encode::build_umt5_tokenizer(&self.root, &te_cfg, "wan trainer")?;

        let total = req.items.len() as u32;
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127) of `(x0, caption, subject-mask
        // loss weight)` — the weight (broadcast to that bucket's `[1, C, 1, h, w]` latent) is `None`
        // unless subject-masked loss is on (sc-24828).
        let mut cache: Vec<(Tensor, Tensor, Option<Tensor>)> =
            Vec::with_capacity(req.items.len() * edges.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let cap = encode_caption(&tokenizer, &te_cfg, &text_encoder, &item.caption, device)?;
            let square = decode_square(&item.image_path)?; // decoded once, resized per bucket edge
                                                           // The item's subject mask, read + checked once (None when masked loss is off).
            let mask =
                PreparedSubjectMask::load_if_enabled(LABEL, item, cfg.subject_mask_loss.as_ref())?;
            let buckets = encode_item_buckets(&square, &edges, mask.as_ref(), device, |img| {
                let video = img.unsqueeze(2)?; // [1,3,1,edge,edge] (T=1 still frame)
                Ok(vae.encode(&video)?.to_dtype(DType::F32)?) // [1,16,1,h,w] normalized mean
            })?;
            for (x0, mask_weight) in buckets {
                cache.push((x0, cap.clone(), mask_weight));
            }
        }

        // --- preview-sample plan (sc-8650): pre-encode prompts + load a resident decode-only VAE ---
        // Built here, while the UMT5 encoder is still resident and before the cache VAE is dropped, so
        // `train_impl`'s loop can render previews from the in-progress experts (mirrors the inline
        // pattern the bespoke-loop families use; the FlowMatchTrainer families do this in `cache`).
        // `None` ⇒ sampling disabled (no cadence or no prompts) ⇒ the loop renders nothing.
        let sample_state: Option<WanSampleState> =
            if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled()
            {
                let mut umt5 = Vec::new();
                let mut prompts = Vec::new();
                for p in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                    // Same conditioning call the cache loop uses → identical [1, 512, 4096] surface.
                    umt5.push(encode_caption(
                        &tokenizer,
                        &te_cfg,
                        &text_encoder,
                        p,
                        device,
                    )?);
                    prompts.push(p.clone());
                }
                // Resident DECODE-ONLY z16 VAE (the cache VAE carried the encoder for latent caching; the
                // preview path needs only the decoder). Loaded f32, like inference's VAE.
                let preview_vae = WanVae16::new(
                    &Vae16Config::wan21(),
                    flow_match::component_vb(&self.root, "vae", device, DType::F32, LABEL)?,
                )?;
                Some(WanSampleState {
                    prompts,
                    umt5,
                    vae: preview_vae,
                    edge: max_edge,
                })
            } else {
                None
            };

        drop(text_encoder);
        drop(vae);
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            return Err(CandleError::Msg(
                "wan trainer: no usable dataset items".into(),
            ));
        }

        // RoPE tables per resolution bucket — every cached latent of one bucket shares its token
        // geometry (sc-2127; a single table when buckets are off).
        let ropes = bucket_rope_tables(&cache, edges.len(), &dit_cfg, device)?;
        // sc-2127: which cached (item, bucket) latent each expert visit trains on.
        let schedule =
            BucketSchedule::new(cache.len() / edges.len(), &cfg.training_buckets(), cfg.seed);
        // Epic 2123 E8: references once per (item, bucket) entry; alternation keyed on the real item
        // with one window per update of every expert (the trainer has no resume, so no replay).
        if let Some(path) = perceptual.as_mut() {
            // sc-24832: the job's subject masks (restricted normal loss) reach every reference,
            // cropped like the image and resampled onto its decoded size.
            let masks = candle_gen::gen_core::train::subject_mask::PerceptualSubjectMasks::load(
                "wan trainer",
                &req.items,
                cfg,
                edges.len(),
                CropBox::center_square,
            )?;
            prepare_perceptual_references(path, cache.iter().map(|(x0, _, _)| x0), masks)?;
        }
        let mut alternation = perceptual.as_ref().map(|_| {
            perceptual_alternation(
                cache.len() / edges.len(),
                cfg.gradient_accumulation.max(1),
                if dual { 2 } else { 1 },
            )
        });

        // --- build the two experts (transformer/ = high-noise, transformer_2/ = low-noise) ---
        let suffixes = flow_match::resolve_target_suffixes(cfg, &WAN_ATTN_TARGETS);
        let accum = cfg.gradient_accumulation.max(1);
        let weight_decay = flow_match::effective_weight_decay(cfg);
        let mae = flow_match::is_mae(cfg);
        let boundary = variant.boundary();
        let plans = if dual {
            vec![
                (
                    "transformer",
                    "high_noise",
                    (boundary, 1.0),
                    cfg.steps.div_ceil(2),
                ),
                ("transformer_2", "low_noise", (0.0, boundary), cfg.steps / 2),
            ]
        } else {
            vec![("transformer", "", (0.0, 1.0), cfg.steps)]
        };
        let expert_count = plans.len() as u32;
        let (expert_schedule_micro, expert_warmup) =
            expert_schedule_inputs(cfg.steps, expert_count, cfg.lr_warmup_steps);
        let mut experts: Vec<ExpertState> = Vec::with_capacity(plans.len());
        for (idx, (sub, suffix, band, _micro)) in plans.into_iter().enumerate() {
            let mut dit = WanTransformerTrain::new(
                &dit_cfg,
                flow_match::component_vb(&self.root, sub, device, compute_dtype, LABEL)?,
            )?;
            // Distinct per-expert seed (so the two adapters don't init identically), reproducible.
            let seed = cfg
                .seed
                .wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9));
            let set = flow_match::install_adapters(&mut dit, cfg, &suffixes, seed, device)?;
            let opt = TrainOptimizer::from_config(
                &cfg.optimizer,
                set.vars.clone(),
                cfg.learning_rate,
                weight_decay,
            )?;
            let (total_updates, warmup_updates) =
                schedule_updates(expert_schedule_micro, accum, expert_warmup);
            experts.push(ExpertState {
                dit,
                set,
                opt,
                band,
                accumulated: None,
                micro: 0,
                update_idx: 0,
                total_updates,
                warmup_updates,
                noise_seed: seed,
                suffix,
            });
        }

        // --- train loop (alternating experts) ---
        // Checkpointing is strictly request-controlled. Production 14B runs should opt in because
        // Candle's dense backward materializes frozen-base gradients, but a false flag must retain
        // the dense kernel rather than silently changing the requested execution semantics.
        let use_checkpoint = use_checkpointed_backward(cfg);
        let mut last_loss = 0.0f32;
        let mut steps_run = 0u32;
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let ei = expert_index(step, dual); // dual: odd → high; dense: the single expert
                                               // Index by the expert's own visit count, not the raw step — else on an even-sized
                                               // dataset each expert stays parity-locked to a disjoint half (sc-11157 / F-082).
            let ci = expert_cache_index(step, dual, &schedule);
            let (x0, cap, mask_weight) = &cache[ci];
            let (cos, sin) = &ropes[ci % edges.len()];
            let band = experts[ei].band;
            let mut t = sample_band_timestep(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                band,
                flow_match::timestep_seed(cfg.seed, step),
            );
            // Epic 2123 E8: plan the step (windows confined to this expert's band); an aux-only step
            // trains at the remapped noise level.
            let aux = match (perceptual.as_ref(), alternation.as_mut()) {
                (Some(path), Some(alt)) => {
                    let key = alt.key(step, expert_item(step, dual, &schedule));
                    let plan = plan_in_band(path, key, ci, band, t)?;
                    t = plan.noise_level as f64;
                    Some(AuxStep {
                        path,
                        plan,
                        entry: ci,
                    })
                }
                _ => None,
            };
            let noise = flow_match::sample_noise(
                x0.dims(),
                flow_match::noise_seed(cfg.seed, step),
                device,
            )?;
            let (losses, grads) = compute_step_loss_grads(
                &experts[ei].dit,
                &experts[ei].set.vars,
                x0,
                cap,
                t,
                &noise,
                cos,
                sin,
                mae,
                mask_weight.as_ref(),
                compute_dtype,
                use_checkpoint,
                y_channels,
                aux.as_ref(),
            )?;
            last_loss = losses.total;
            steps_run = step;

            let ex = &mut experts[ei];
            accumulate_grads(&mut ex.accumulated, grads, &ex.set.vars)?;
            ex.micro += 1;
            if ex.micro.is_multiple_of(accum) {
                apply_update(ex, accum, cfg)?;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            // --- preview samples (sc-8650) — render one still per prompt from the in-progress experts ---
            // Best-effort: a render `Err` logs + skips, never aborting the run (a flaky preview must not
            // fail a 14B training job). Both experts carry their live adapters, so `render_one_preview`
            // runs the partially-trained dual-expert MoE directly.
            if cfg.sample_every > 0 && step % cfg.sample_every == 0 {
                if let Some(state) = sample_state.as_ref() {
                    let total = state.prompts.len() as u32;
                    // Freeze BOTH experts' adapters to detached snapshots so the preview denoise runs
                    // graph-free (the factor `Var`s are otherwise tracked → the forward's activations are
                    // retained → OOM at full resolution). Restored right after so training keeps its grads.
                    freeze_preview_lora_hosts(experts.iter_mut().map(|ex| &mut ex.dit))?;
                    for (i, prompt) in state.prompts.iter().enumerate() {
                        if req.cancel.is_cancelled() {
                            break;
                        }
                        let seed = flow_match::sample_seed(cfg.seed, step, i);
                        match render_one_preview(&experts, state, i, cfg, seed, device) {
                            Ok(image) => on_progress(TrainingProgress::Sample {
                                step,
                                index: i as u32 + 1,
                                total,
                                prompt: prompt.clone(),
                                image,
                            }),
                            Err(e) => eprintln!(
                                "[sc-8650] wan: preview sample failed at step {step} (prompt {}): \
                                 {e} — skipping this preview, training continues",
                                i + 1
                            ),
                        }
                    }
                    thaw_preview_lora_hosts(experts.iter_mut().map(|ex| &mut ex.dit))?;
                }
            }

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                flow_match::create_output_dir(&req.output_dir)?;
                for ex in &experts {
                    let name = with_expert_suffix(
                        &format!("{}-step{step:06}.safetensors", file_stem(&req.file_name)),
                        ex.suffix,
                    );
                    flow_match::save_adapter(&ex.set, &HashMap::new(), &req.output_dir.join(name))?;
                }
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        if steps_run == 0 {
            return Err(CandleError::Canceled);
        }
        // Flush any expert's pending (sub-`accum`) accumulation so the final partial step is applied.
        for ex in &mut experts {
            if ex.accumulated.is_some() {
                let pending = pending_micro_count(ex.micro, accum)
                    .expect("an accumulated tail has at least one pending micro-step");
                apply_update(ex, pending, cfg)?;
            }
        }

        // --- save the high/low adapter pair; report the high-noise file as the primary path ---
        on_progress(TrainingProgress::Saving);
        flow_match::create_output_dir(&req.output_dir)?;
        let mut primary: Option<PathBuf> = None;
        for ex in &experts {
            let path = req
                .output_dir
                .join(with_expert_suffix(&req.file_name, ex.suffix));
            flow_match::save_adapter(&ex.set, &HashMap::new(), &path)?;
            if ex.suffix == "high_noise" || ex.suffix.is_empty() {
                primary = Some(path);
            }
        }
        Ok(TrainingOutput {
            adapter_path: primary.expect("a trained expert is always present"),
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Fire one optimizer update for `ex`: delegates the average-clip-(noise)-step to the shared
/// [`flow_match::apply_update`] (over `ex`'s own optimizer/accumulation/schedule and adapter-noise
/// seed), then advances the expert's update counter.
fn apply_update(ex: &mut ExpertState, micro_count: u32, cfg: &TrainingConfig) -> Result<()> {
    flow_match::apply_update(
        &mut ex.opt,
        &mut ex.accumulated,
        &ex.set,
        micro_count,
        cfg,
        ex.update_idx,
        ex.total_updates,
        ex.warmup_updates,
        ex.noise_seed,
    )?;
    ex.update_idx += 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_gen::gen_core::Quant;
    use candle_gen::train::lora::build_lora_targets;
    use candle_gen::train::optim::clip_grad_norm;

    /// A tiny Wan-shaped DiT (z16, head_dim 128, 1 head, 1 layer) — exercises the real flow-match
    /// forward+backward on CPU.
    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 16,
            out_channels: 16,
            num_layers: 1,
            num_heads: 1,
            head_dim: 128,
            dim: 128,
            ffn_dim: 256,
            freq_dim: 256,
            text_dim: 64,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 1024,
        }
    }

    fn bucket(resolution: u32, repeats: u32) -> gen_core::ResolutionBucket {
        gen_core::ResolutionBucket {
            resolution,
            repeats,
        }
    }

    /// The buckets-off schedule (one bucket at the legacy resolution) over `n` cached items.
    fn one_bucket(n: usize) -> BucketSchedule {
        BucketSchedule::new(n, &[bucket(512, 1)], 42)
    }

    /// sc-2127: with one resolution bucket the schedule-driven index is EXACTLY the pre-bucket
    /// per-expert formula `((step - 1) / experts) % N`, for the dual MoE and the dense variant, so
    /// a buckets-off run trains on the identical item sequence. (Mutation: feeding the schedule the
    /// raw `step - 1` instead of the per-expert counter fails this for `dual = true`.)
    #[test]
    fn one_bucket_schedule_reproduces_the_pre_bucket_expert_index() {
        for n in [1usize, 2, 3, 7, 10] {
            for dual in [true, false] {
                let experts = if dual { 2 } else { 1 };
                for repeats in [1u32, 5] {
                    let schedule = BucketSchedule::new(n, &[bucket(768, repeats)], 9);
                    for step in 1..=200u32 {
                        let old = (((step - 1) / experts) as usize) % n;
                        assert_eq!(
                            expert_cache_index(step, dual, &schedule),
                            old,
                            "n={n} dual={dual} repeats={repeats} step={step}"
                        );
                    }
                }
            }
        }
    }

    /// sc-2127: with buckets `[512×16, 768×4, 1024×1]` each expert, over one epoch of its own visits,
    /// trains every item at the 16:4:1 per-bucket mix, and every index lands in the item-major cache.
    /// (Mutation: feeding the schedule the raw `step - 1` instead of the per-expert counter breaks
    /// the dual counts.)
    #[test]
    fn multi_bucket_schedule_gives_each_expert_the_per_item_repeat_mix() {
        let (n_items, buckets) = (3usize, [bucket(512, 16), bucket(768, 4), bucket(1024, 1)]);
        let schedule = BucketSchedule::new(n_items, &buckets, 7);
        let epoch = schedule.epoch_len();
        assert_eq!(epoch, n_items * 21);
        for dual in [true, false] {
            let experts = if dual { 2 } else { 1 };
            for expert in 0..experts {
                let mut counts = vec![[0usize; 3]; n_items];
                for step in 1..=(epoch * experts) as u32 {
                    if expert_index(step, dual) != expert {
                        continue;
                    }
                    let ci = expert_cache_index(step, dual, &schedule);
                    assert!(ci < n_items * buckets.len());
                    counts[ci / buckets.len()][ci % buckets.len()] += 1;
                }
                for (item, c) in counts.iter().enumerate() {
                    assert_eq!(*c, [16, 4, 1], "dual={dual} expert={expert} item={item}");
                }
            }
        }
    }

    /// sc-24828 × sc-2127: with masked loss on and two buckets, each cached still latent
    /// `[1, C, 1, h, w]` carries a weight of ITS OWN shape (built on that bucket's grid), and the
    /// masked-out region is zero.
    #[test]
    fn subject_mask_weight_follows_each_buckets_latent() {
        use candle_gen::candle_core::IndexOp;
        use candle_gen::gen_core::train::TrainingItem;
        use candle_gen::gen_core::SubjectMaskLoss;
        let dir = tempfile::tempdir().unwrap();
        // 48×32 image → center square x ∈ [8, 40); the subject is that square's left half (x < 24).
        let image_path = dir.path().join("img.png");
        image::RgbImage::from_pixel(48, 32, image::Rgb([128, 64, 32]))
            .save(&image_path)
            .unwrap();
        let mask_path = dir.path().join("mask.png");
        image::GrayImage::from_fn(48, 32, |x, _| image::Luma([if x < 24 { 255 } else { 0 }]))
            .save(&mask_path)
            .unwrap();
        let item = TrainingItem {
            image_path,
            caption: String::new(),
            control_image_path: None,
            model_options: Default::default(),
            reference_image_paths: Vec::new(),
            subject_mask_path: Some(mask_path),
        };
        let cfg = SubjectMaskLoss {
            background_weight: 0.0,
            subject_weight: 1.0,
        };
        let mask = PreparedSubjectMask::load("t", &item, &cfg).unwrap();
        let square = decode_square(&item.image_path).unwrap();
        let dev = Device::Cpu;
        // A stand-in /8 still-frame encoder: `[1, 3, edge, edge]` → `[1, 3, 1, edge/8, edge/8]`.
        let encode = |img: &Tensor| Ok(img.avg_pool2d(8)?.unsqueeze(2)?);
        let entries = encode_item_buckets(&square, &[32, 64], Some(&mask), &dev, encode).unwrap();
        assert_eq!(entries.len(), 2);
        for ((x0, w), grid) in entries.iter().zip([4usize, 8]) {
            assert_eq!(x0.dims(), &[1, 3, 1, grid, grid]);
            let w = w.as_ref().expect("masked loss is on");
            assert_eq!(w.dims(), x0.dims(), "bucket {grid}: weight shape");
            let rows = w.i((0, 0, 0)).unwrap().to_vec2::<f32>().unwrap();
            for row in rows {
                for (x, v) in row.into_iter().enumerate() {
                    let want = if x < grid / 2 { 1.0 } else { 0.0 };
                    assert_eq!(v, want, "bucket {grid}: column {x}");
                }
            }
        }
        let off = encode_item_buckets(&square, &[32, 64], None, &dev, encode).unwrap();
        assert!(off.iter().all(|(_, w)| w.is_none()));
    }

    /// sc-2127: one RoPE table pair per bucket, shaped from that bucket's cached latent geometry —
    /// a 4×4 and an 8×8 latent get distinct tables equal to a direct `cos_sin` at their patch grid.
    /// (Mutation: building every table from `cache[0]` makes the 8×8 table the wrong length.)
    #[test]
    fn bucket_rope_tables_follow_each_buckets_latent_grid() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let cap = Tensor::zeros((1, 3, cfg.text_dim), DType::F32, &dev).unwrap();
        let latent = |hw: usize| Tensor::zeros((1, 16, 1, hw, hw), DType::F32, &dev).unwrap();
        // Two items × two buckets, item-major.
        let cache = vec![
            (latent(4), cap.clone(), None),
            (latent(8), cap.clone(), None),
            (latent(4), cap.clone(), None),
            (latent(8), cap.clone(), None),
        ];
        let ropes = bucket_rope_tables(&cache, 2, &cfg, &dev).unwrap();
        assert_eq!(ropes.len(), 2);
        for (b, hw) in [(0usize, 4usize), (1, 8)] {
            let (cos, sin) = WanRope::new(&cfg).cos_sin(1, hw / 2, hw / 2, &dev).unwrap();
            assert_eq!(ropes[b].0.dims(), cos.dims(), "bucket {b} cos");
            assert_eq!(ropes[b].1.dims(), sin.dims(), "bucket {b} sin");
        }
        assert_ne!(ropes[0].0.dims(), ropes[1].0.dims());
    }

    fn tiny_i2v_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 36,
            out_channels: 16,
            ..tiny_cfg()
        }
    }

    #[test]
    fn nondivisible_accumulation_uses_each_experts_actual_tail_count() {
        assert_eq!(pending_micro_count(5, 4), Some(1), "single-expert tail");
        assert_eq!(
            pending_micro_count(4, 4),
            None,
            "complete windows have no tail"
        );

        let mut visits = [0u32; 2];
        for step in 1..=10 {
            visits[expert_index(step, true)] += 1;
        }
        assert_eq!(visits, [5, 5]);
        assert_eq!(pending_micro_count(visits[0], 2), Some(1));
        assert_eq!(pending_micro_count(visits[1], 2), Some(1));
    }

    #[test]
    fn checkpoint_kernel_selection_is_strictly_opt_in_for_all_wan_training_cells() {
        for (variant, network) in [
            (TrainVariant::Ti2v5b, NetworkType::Lora),
            (TrainVariant::I2v14b, NetworkType::Lora),
            (TrainVariant::T2v14b, NetworkType::Lora),
            (TrainVariant::T2v14b, NetworkType::Lokr),
        ] {
            let mut config = TrainingConfig {
                network_type: network,
                ..Default::default()
            };
            assert!(
                !use_checkpointed_backward(&config),
                "{variant:?} {network:?}"
            );
            config.gradient_checkpointing = true;
            assert!(
                use_checkpointed_backward(&config),
                "{variant:?} {network:?}"
            );
        }
    }

    #[test]
    fn dual_experts_share_the_mlx_floor_divided_schedule_and_full_warmup() {
        assert_eq!(expert_schedule_inputs(21, 2, 7), (10, 7));
        assert_eq!(expert_schedule_inputs(1, 2, 3), (1, 3));
        let high = schedule_updates(10, 2, 7);
        let low = schedule_updates(10, 2, 7);
        assert_eq!(high, low);
        assert_eq!(high, (5, 4));
        assert!(
            (candle_gen::train::schedule::lr_multiplier(
                candle_gen::gen_core::train::LrSchedule::Constant,
                0,
                high.0,
                high.1,
            ) - 0.2)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn load_rejects_explicit_and_physical_packed_experts_before_training() {
        let empty = tempfile::tempdir().unwrap();
        let mut explicit = LoadSpec::new(WeightsSource::Dir(empty.path().into()));
        explicit.quantize = Some(Quant::Q8);
        assert!(load_variant(&explicit, TrainVariant::Ti2v5b)
            .err()
            .expect("explicit quantization must be rejected")
            .to_string()
            .contains("explicit quantization"));

        for (variant, component, config_marker) in [
            (TrainVariant::Ti2v5b, "transformer", true),
            (TrainVariant::I2v14b, "transformer_2", false),
        ] {
            let root = tempfile::tempdir().unwrap();
            let dir = root.path().join(component);
            std::fs::create_dir_all(&dir).unwrap();
            if config_marker {
                std::fs::write(
                    dir.join("config.json"),
                    r#"{"quantization":{"bits":8,"group_size":64}}"#,
                )
                .unwrap();
            } else {
                candle_gen::candle_core::safetensors::save(
                    &HashMap::from([(
                        "blocks.0.attn.to_q.scales".to_string(),
                        Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
                    )]),
                    dir.join("model.safetensors"),
                )
                .unwrap();
            }
            let spec = LoadSpec::new(WeightsSource::Dir(root.path().into()));
            let error = load_variant(&spec, variant)
                .err()
                .expect("physical packed expert must be rejected")
                .to_string();
            assert!(error.contains(component), "{variant:?}: {error}");
        }
    }

    /// Randomize every var in a fresh `VarMap` — `vb.get` raw tensors (notably the `patch_embedding`
    /// conv weight) default to ZERO-init, and a zero patch kernel makes `hidden ≡ 0`, which makes the
    /// LoRA adapters' inputs zero and their grads vacuously zero. Real training loads nonzero weights;
    /// the tiny tests must do the same to exercise the gradient path.
    fn randomize_base(vm: &VarMap, dev: &Device) {
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.1f32, v.dims(), dev).unwrap())
                .unwrap();
        }
    }

    fn tiny_inputs(
        cfg: &TransformerConfig,
        dev: &Device,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let x0 = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 1, 4, 4), dev).unwrap();
        let umt5 = Tensor::randn(0f32, 1f32, (1, 3, cfg.text_dim), dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, cfg.in_channels, 1, 4, 4), dev).unwrap();
        let (cos, sin) = WanRope::new(cfg).cos_sin(1, 2, 2, dev).unwrap();
        (x0, umt5, noise, cos, sin)
    }

    /// ISOLATION: does `build_lora_targets` + `set.vars[i].set(..)` propagate to the installed
    /// LoraLinear's forward, with NO Wan dit involved? (Pins the harness mechanism in this crate.)
    #[test]
    fn harness_factor_set_propagates_to_forward() {
        use candle_gen::candle_nn::{Linear, Module};
        use candle_gen::train::lora::{LoraHost, LoraLinear};
        struct H(LoraLinear);
        impl LoraHost for H {
            fn visit_lora_mut(
                &mut self,
                f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
            ) -> candle_gen::Result<()> {
                f(&mut self.0)
            }
        }
        let dev = Device::Cpu;
        let w = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
        let mut h = H(LoraLinear::from_linear(
            Linear::new(w, None),
            4,
            4,
            "to_q".into(),
        ));
        let set = build_lora_targets(&mut h, &["to_q".to_string()], 2, 4.0, 7, &dev).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 4), &dev).unwrap();
        let y0 =
            h.0.forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.5f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let y1 =
            h.0.forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
        assert_ne!(
            y0, y1,
            "setting set.vars must change the installed LoraLinear forward"
        );
    }

    #[test]
    fn preview_lora_visitor_failure_propagates() {
        struct FailingHost;
        impl LoraHost for FailingHost {
            fn visit_lora_mut(
                &mut self,
                _f: &mut dyn FnMut(
                    &mut candle_gen::train::lora::LoraLinear,
                ) -> candle_gen::Result<()>,
            ) -> candle_gen::Result<()> {
                Err(CandleError::Msg("injected Wan visitor failure".into()))
            }
        }

        let mut hosts = [FailingHost];
        let err = freeze_preview_lora_hosts(hosts.iter_mut()).unwrap_err();
        assert!(
            matches!(err, CandleError::Msg(ref message) if message == "injected Wan visitor failure"),
            "the visitor error must reach the training caller, got {err:?}"
        );
    }

    /// Band sampling is deterministic, in-band, and the high band lies above the low band.
    #[test]
    #[allow(clippy::manual_range_contains)] // the `±1e-6` tolerance reads clearer as explicit bounds
    fn band_timestep_is_in_band_and_ordered() {
        let hi = (T2V_14B_BOUNDARY, 1.0);
        let lo = (0.0, T2V_14B_BOUNDARY);
        for seed in [0u64, 1, 42, 9999] {
            let a = sample_band_timestep("sigmoid", "balanced", hi, seed);
            let b = sample_band_timestep("sigmoid", "balanced", hi, seed);
            assert_eq!(a, b, "same seed reproduces");
            assert!(
                a >= T2V_14B_BOUNDARY - 1e-6 && a < 1.0,
                "high band t out of range: {a}"
            );
            let l = sample_band_timestep("sigmoid", "balanced", lo, seed);
            assert!(
                l > 0.0 && l <= T2V_14B_BOUNDARY + 1e-6,
                "low band t out of range: {l}"
            );
        }
    }

    /// Regression for sc-11157 / F-082: the dataset item must NOT stay parity-locked to the
    /// alternating expert. Over a full pass, each expert must see EVERY item — most critically for
    /// the even-sized datasets (2, 10, 20 items) that are the common LoRA case — so neither adapter
    /// silently trains on a disjoint half. The pre-fix `(step - 1) % N` selector fails this: it kept
    /// the high-noise expert on even indices and the low-noise on odd indices forever.
    #[test]
    fn experts_each_cover_the_whole_dataset() {
        for &n in &[2usize, 4, 10, 20, 3, 7] {
            // Run enough steps for every expert to complete at least one full cycle of the dataset.
            let steps = (2 * n * 3) as u32;
            let mut hi_seen = std::collections::BTreeSet::new();
            let mut lo_seen = std::collections::BTreeSet::new();
            for step in 1..=steps {
                let idx = expert_cache_index(step, true, &one_bucket(n));
                match expert_index(step, true) {
                    0 => hi_seen.insert(idx),
                    _ => lo_seen.insert(idx),
                };
            }
            let full: std::collections::BTreeSet<usize> = (0..n).collect();
            assert_eq!(
                hi_seen, full,
                "high-noise expert must cover all {n} items, saw {hi_seen:?}"
            );
            assert_eq!(
                lo_seen, full,
                "low-noise expert must cover all {n} items, saw {lo_seen:?}"
            );
        }
    }

    /// Guards the exact even-size parity trap from the finding: with N even, the OLD selector
    /// `(step - 1) % N` collapses each expert onto one parity class. Assert the NEW selector breaks
    /// that lock by having the high-noise expert reach at least one ODD index (impossible under the
    /// old scheme) within N steps of its own progression.
    #[test]
    fn even_dataset_expert_is_not_parity_locked() {
        let n = 10usize;
        let mut hi_indices = Vec::new();
        for step in (1..=(4 * n as u32)).filter(|s| expert_index(*s, true) == 0) {
            hi_indices.push(expert_cache_index(step, true, &one_bucket(n)));
        }
        assert!(
            hi_indices.iter().any(|i| i % 2 == 1),
            "high-noise expert never reached an odd index on N={n}: {hi_indices:?} — parity lock persists"
        );
        assert!(
            hi_indices.iter().any(|i| i % 2 == 0),
            "high-noise expert never reached an even index: {hi_indices:?}"
        );
    }

    /// The keystone training gate: a real flow-match forward+backward over the tiny DiT with nonzero
    /// LoRA factors yields a finite loss and a gradient on **every** adapter `Var`.
    #[test]
    fn backward_reaches_lora_factors() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = WanTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = WAN_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        // Move B off its zero-init so both A and B grads are nonzero (a no-op-init adapter zeros A's grad).
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, umt5, noise, cos, sin) = tiny_inputs(&cfg, &dev);
        let (loss, grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &umt5,
            0.5,
            &noise,
            &cos,
            &sin,
            false,
            None,
            DType::F32,
            false,
            0,
        )
        .unwrap();
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        let mut saw_nonzero = false;
        for (i, v) in set.vars.iter().enumerate() {
            let g = grads
                .get(v.as_tensor())
                .unwrap_or_else(|| panic!("adapter var {i} has no gradient"));
            let gv = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(
                gv.iter().all(|x| x.is_finite()),
                "var {i} gradient has non-finite entries"
            );
            if gv.iter().any(|x| x.abs() > 1e-9) {
                saw_nonzero = true;
            }
        }
        assert!(
            saw_nonzero,
            "every adapter gradient was zero — backprop is not reaching the factors"
        );
        // 4 projections × 2 attentions (attn1 self + attn2 cross) × num_layers, ×2 factors.
        assert_eq!(set.vars.len(), 4 * 2 * cfg.num_layers * 2);
    }

    #[test]
    fn i2v_channel_padding_reaches_lora_factors() {
        let dev = Device::Cpu;
        let cfg = tiny_i2v_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = WanTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = WAN_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let x0 = Tensor::randn(0f32, 1f32, (1, 16, 1, 4, 4), &dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();
        let umt5 = Tensor::randn(0f32, 1f32, (1, 3, cfg.text_dim), &dev).unwrap();
        let (cos, sin) = WanRope::new(&cfg).cos_sin(1, 2, 2, &dev).unwrap();
        let (loss, grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &umt5,
            0.5,
            &noise,
            &cos,
            &sin,
            false,
            None,
            DType::F32,
            false,
            20,
        )
        .unwrap();
        assert!(loss.is_finite());
        assert!(set.vars.iter().any(|v| grads.get(v.as_tensor()).is_some()));
    }

    /// The correctness gate for the gradient-checkpointed backward (the path real training always uses):
    /// it must reproduce the dense `loss.backward()` grads (mod float reassociation) on the tiny DiT.
    /// All Wan adapters live in the checkpointed block stack (no retained pre-main adapters), so this
    /// spans every adapter target.
    #[test]
    fn dense_and_checkpoint_grads_match() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = WanTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = WAN_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, umt5, noise, cos, sin) = tiny_inputs(&cfg, &dev);
        let (loss_d, g_d) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &umt5,
            0.5,
            &noise,
            &cos,
            &sin,
            false,
            None,
            DType::F32,
            false,
            0,
        )
        .unwrap();
        let (loss_c, g_c) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &umt5,
            0.5,
            &noise,
            &cos,
            &sin,
            false,
            None,
            DType::F32,
            true,
            0,
        )
        .unwrap();
        assert!(
            (loss_d - loss_c).abs() < 1e-4,
            "loss: dense {loss_d} vs checkpoint {loss_c}"
        );
        let mut saw_nonzero = false;
        for (i, v) in set.vars.iter().enumerate() {
            let a = g_d
                .get(v.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let b = g_c
                .get(v.as_tensor())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-4,
                    "grad mismatch for var {i} (dense {x} vs checkpoint {y})"
                );
                if x.abs() > 1e-6 {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "expected nonzero adapter grads to compare");
    }

    /// sc-24828: subject-masked loss on both backward paths (the checkpointed one is what every
    /// real expert step runs). An all-ones map is the unweighted loss; an all-zero map zeroes the loss
    /// AND every adapter gradient (dense and checkpointed); a half map matches across paths.
    #[test]
    fn subject_mask_weight_reaches_both_backward_paths() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = WanTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = WAN_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, umt5, noise, cos, sin) = tiny_inputs(&cfg, &dev);
        let shape = x0.dims().to_vec();
        let map = |w: &[f32]| flow_match::subject_mask_weight(w, 4, 4, &shape, &dev).unwrap();
        let run = |weight: Option<&Tensor>, ckpt: bool| {
            compute_loss_grads(
                &dit,
                &set.vars,
                &x0,
                &umt5,
                0.5,
                &noise,
                &cos,
                &sin,
                false,
                weight,
                DType::F32,
                ckpt,
                0,
            )
            .unwrap()
        };
        let (plain, _) = run(None, false);
        let ones = map(&[1.0; 16]);
        assert!((run(Some(&ones), false).0 - plain).abs() < 1e-6);
        let zeros = map(&[0.0; 16]);
        for ckpt in [false, true] {
            let (loss, grads) = run(Some(&zeros), ckpt);
            assert_eq!(
                loss, 0.0,
                "ckpt={ckpt}: an all-background map must zero the loss"
            );
            for v in &set.vars {
                if let Some(g) = grads.get(v.as_tensor()) {
                    let g = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    assert!(
                        g.iter().all(|x| *x == 0.0),
                        "ckpt={ckpt}: nonzero adapter grad"
                    );
                }
            }
        }
        let half: Vec<f32> = (0..16).map(|i| if i % 4 < 2 { 1.0 } else { 0.0 }).collect();
        let half = map(&half);
        let (dense, _) = run(Some(&half), false);
        let (ckpt, _) = run(Some(&half), true);
        assert!(dense > 0.0 && dense < plain, "{dense} vs {plain}");
        assert!(
            (dense - ckpt).abs() < 1e-4,
            "dense {dense} vs checkpoint {ckpt}"
        );
        for d in [
            trainer_descriptor(),
            trainer_descriptor_i2v_14b(),
            trainer_descriptor_ti2v_5b(),
        ] {
            assert!(d.techniques.subject_mask_loss, "{}", d.id);
        }
    }

    /// A few optimizer steps on a fixed batch lower the loss — the step descends the flow-match
    /// objective end to end through the harness.
    #[test]
    fn one_optimizer_step_descends() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = WanTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = WAN_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, umt5, noise, cos, sin) = tiny_inputs(&cfg, &dev);
        let mut opt = TrainOptimizer::from_config("adamw", set.vars.clone(), 1e-2, 0.0).unwrap();
        let (loss0, mut grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &umt5,
            0.5,
            &noise,
            &cos,
            &sin,
            false,
            None,
            DType::F32,
            false,
            0,
        )
        .unwrap();
        for _ in 0..5 {
            clip_grad_norm(&mut grads, &set.vars, 1.0).unwrap();
            opt.step(&grads).unwrap();
            let (_l, g) = compute_loss_grads(
                &dit,
                &set.vars,
                &x0,
                &umt5,
                0.5,
                &noise,
                &cos,
                &sin,
                false,
                None,
                DType::F32,
                false,
                0,
            )
            .unwrap();
            grads = g;
        }
        let (loss1, _) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &umt5,
            0.5,
            &noise,
            &cos,
            &sin,
            false,
            None,
            DType::F32,
            false,
            0,
        )
        .unwrap();
        assert!(
            loss1 < loss0,
            "5 steps on a fixed batch should lower the loss: {loss0} -> {loss1}"
        );
    }

    /// The trainer resolves through the explicit family registry as the candle Wan
    /// A14B trainer; `load_trainer` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn trainer_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = crate::provider_registry()
            .unwrap()
            .load_trainer(MODEL_ID_T2V_14B, &spec)
            .expect("candle wan a14b trainer is registered");
        assert_eq!(t.descriptor().id, MODEL_ID_T2V_14B);
        assert_eq!(t.descriptor().backend, "candle");
        assert_eq!(t.descriptor().modality, Modality::Video);
        assert!(t.descriptor().supports_lora && t.descriptor().supports_lokr);
        assert!(t.descriptor().techniques.resolution_buckets);

        for id in [MODEL_ID_I2V_14B, MODEL_ID] {
            let t = crate::provider_registry()
                .unwrap()
                .load_trainer(id, &spec)
                .unwrap_or_else(|e| panic!("{id} trainer must resolve: {e}"));
            assert_eq!(t.descriptor().id, id);
            assert!(t.descriptor().supports_lora);
            assert!(!t.descriptor().supports_lokr);
            assert!(t.descriptor().techniques.resolution_buckets, "{id}");
        }
    }

    /// `validate` rejects an empty dataset, zero rank/steps, an unsupported optimizer, and unrecognized
    /// timestep/loss knobs — before any load (now via the shared `flow_match::validate_flow_match_request`).
    #[test]
    fn validate_rejects_bad_requests() {
        use candle_gen::gen_core::runtime::CancelFlag;
        use candle_gen::gen_core::train::TrainingItem;
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = crate::provider_registry()
            .unwrap()
            .load_trainer(MODEL_ID_T2V_14B, &spec)
            .unwrap();
        let base = TrainingRequest {
            items: vec![TrainingItem {
                image_path: "/img.png".into(),
                caption: "x".into(),
                control_image_path: None,
                model_options: Default::default(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            }],
            config: TrainingConfig::default(),
            output_dir: "/out".into(),
            file_name: "a.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        };
        assert!(t.validate(&base).is_ok());
        let bad = |mutate: &dyn Fn(&mut TrainingRequest)| {
            let mut r = base.clone();
            mutate(&mut r);
            assert!(t.validate(&r).is_err());
        };
        bad(&|r| r.items.clear());
        bad(&|r| r.config.rank = 0);
        bad(&|r| r.config.steps = 0);
        bad(&|r| r.config.optimizer = "lion".into());
        bad(&|r| r.config.timestep_type = "bogus".into());
        bad(&|r| r.config.loss_type = "huber".into());
        bad(&|r| r.items[0].control_image_path = Some("/source.png".into()));
        bad(&|r| r.config.resume = true);

        for id in [MODEL_ID_I2V_14B, MODEL_ID] {
            let t = crate::provider_registry()
                .unwrap()
                .load_trainer(id, &spec)
                .unwrap();
            assert!(t.validate(&base).is_ok());
            let mut conditioned = base.clone();
            conditioned.items[0].control_image_path = Some("/source.png".into());
            assert!(t.validate(&conditioned).is_err());
            let mut resume = base.clone();
            resume.config.resume = true;
            assert!(
                t.validate(&resume).is_err(),
                "{id} must fail closed on resume"
            );
            let mut lokr = base.clone();
            lokr.config.network_type = NetworkType::Lokr;
            let err = t.validate(&lokr).unwrap_err().to_string();
            assert!(err.contains("LoRA-only"), "{id}: {err}");
        }
    }

    /// The expert-suffix filename insertion lands before the extension.
    #[test]
    fn expert_suffix_naming() {
        assert_eq!(
            with_expert_suffix("mylora.safetensors", ""),
            "mylora.safetensors"
        );
        assert_eq!(
            with_expert_suffix("mylora.safetensors", "high_noise"),
            "mylora.high_noise.safetensors"
        );
        assert_eq!(
            with_expert_suffix("mylora.safetensors", "low_noise"),
            "mylora.low_noise.safetensors"
        );
        assert_eq!(
            with_expert_suffix("noext", "high_noise"),
            "noext.high_noise"
        );
    }

    #[test]
    fn dense_variant_uses_one_expert_and_walks_every_item() {
        let seen: Vec<usize> = (1..=8)
            .map(|step| {
                assert_eq!(expert_index(step, false), 0);
                expert_cache_index(step, false, &one_bucket(4))
            })
            .collect();
        assert_eq!(seen, [0, 1, 2, 3, 0, 1, 2, 3]);
    }
}

/// sc-24830 (epic 2123 depth anchoring) — the candle Wan step seam on the tiny DiT (z16, 1 layer,
/// latent `[1, 16, 1, 4, 4]`) with a random-init tiny-width TAEW2.1 and a random-init tiny
/// Depth-Anything-V2. Drives the same [`compute_step_loss_grads`] / [`plan_in_band`] /
/// [`perceptual_alternation`] `train_impl` runs. CPU.
#[cfg(test)]
mod depth_anchoring_tests {
    use super::*;
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_gen::train::lora::build_lora_targets;
    use candle_gen::train::perceptual::AuxLoss;
    use candle_gen::train::taehv::{splitmix_uniform, synthetic_taehv_weights, TaehvDecoder};

    fn tiny_cfg() -> TransformerConfig {
        TransformerConfig {
            in_channels: 16,
            out_channels: 16,
            num_layers: 1,
            num_heads: 1,
            head_dim: 128,
            dim: 128,
            ffn_dim: 256,
            freq_dim: 256,
            text_dim: 64,
            patch: (1, 2, 2),
            eps: 1e-6,
            rope_theta: 10000.0,
            rope_max_seq_len: 1024,
        }
    }

    fn schedule(t_min: f32, t_max: f32) -> AuxLossSchedule {
        AuxLossSchedule {
            weight: 0.1,
            t_min,
            t_max,
            every_n: 2,
        }
    }

    fn path_with(schedule: AuxLossSchedule) -> PerceptualPath {
        let tae = TaehvConfig {
            channels: [8, 6, 4, 4],
            ..TaehvConfig::taew2_1()
        };
        let dec = TaehvDecoder::from_weights(
            &synthetic_taehv_weights(&tae, 11, &Device::Cpu).unwrap(),
            tae,
        )
        .unwrap();
        let loss = candle_gen_depth::anchor::tiny_depth_anchor_loss(12, &Device::Cpu).unwrap();
        PerceptualPath::new(
            Some(Box::new(dec)),
            vec![AuxLoss {
                schedule,
                loss: Box::new(loss),
            }],
        )
        .unwrap()
    }

    struct Fixture {
        dit: WanTransformerTrain,
        set: LoraSet,
        x0: Tensor,
        umt5: Tensor,
        noise: Tensor,
        cos: Tensor,
        sin: Tensor,
    }

    /// Deterministic tiny DiT + inputs (seeded `splitmix_uniform`; candle's CPU `randn` is not).
    fn fixture() -> Fixture {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = WanTransformerTrain::new(&cfg, vb).unwrap();
        for (i, (_, v)) in vm.data().lock().unwrap().iter().enumerate() {
            let n = v.as_tensor().dims().get(1).copied().unwrap_or(1).max(1);
            v.set(
                &splitmix_uniform(
                    v.as_tensor().dims(),
                    500 + i as u64,
                    (1.0 / n as f64).sqrt(),
                    0.0,
                    &dev,
                )
                .unwrap(),
            )
            .unwrap();
        }
        let suffixes: Vec<String> = WAN_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        let x0 = splitmix_uniform(&[1, 16, 1, 4, 4], 1, 1.0, 0.0, &dev).unwrap();
        let umt5 = splitmix_uniform(&[1, 3, cfg.text_dim], 2, 1.0, 0.0, &dev).unwrap();
        let noise = splitmix_uniform(&[1, 16, 1, 4, 4], 3, 1.0, 0.0, &dev).unwrap();
        let (cos, sin) = WanRope::new(&cfg).cos_sin(1, 2, 2, &dev).unwrap();
        Fixture {
            dit,
            set,
            x0,
            umt5,
            noise,
            cos,
            sin,
        }
    }

    fn step(f: &Fixture, t: f64, ckpt: bool, aux: Option<&AuxStep<'_>>) -> (StepLosses, GradStore) {
        compute_step_loss_grads(
            &f.dit,
            &f.set.vars,
            &f.x0,
            &f.umt5,
            t,
            &f.noise,
            &f.cos,
            &f.sin,
            false,
            None,
            DType::F32,
            ckpt,
            0,
            aux,
        )
        .unwrap()
    }

    fn prepared(f: &Fixture, schedule: AuxLossSchedule) -> PerceptualPath {
        let mut p = path_with(schedule);
        p.ensure_reference(0, &latent_frames_nchw(&f.x0).unwrap())
            .unwrap();
        p
    }

    fn grad_bits(g: &GradStore, vars: &[Var]) -> Vec<Vec<u32>> {
        vars.iter()
            .map(|v| {
                g.get(v.as_tensor())
                    .map(|t| {
                        t.flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap()
                            .iter()
                            .map(|x| x.to_bits())
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect()
    }

    /// AC (a)+(b), dense and checkpointed: a depth step (key 2) computes no diffusion term, its
    /// total is the weighted depth term, and the zero-init LoRA-B factors get a nonzero finite
    /// gradient; a diffusion step (key 1) carries no depth term. Mutation: compute the diffusion
    /// term unconditionally in `step_loss` ⇒ red.
    #[test]
    fn depth_step_trains_the_lora_through_depth_only_on_both_paths() {
        let f = fixture();
        let p = prepared(&f, schedule(0.0, 1.0));
        for ckpt in [false, true] {
            let plan = plan_in_band(&p, 1, 0, (0.0, 1.0), 0.5).unwrap();
            let s1 = AuxStep {
                path: &p,
                plan,
                entry: 0,
            };
            let (diff, _) = step(&f, 0.5, ckpt, Some(&s1));
            assert_eq!(diff.aux, None, "ckpt={ckpt}");
            assert_eq!(Some(diff.total), diff.diffusion);
            let plan = plan_in_band(&p, 2, 0, (0.0, 1.0), 0.5).unwrap();
            assert!(!plan.diffusion);
            let t = plan.noise_level as f64;
            let s2 = AuxStep {
                path: &p,
                plan,
                entry: 0,
            };
            let (depth, g) = step(&f, t, ckpt, Some(&s2));
            assert_eq!(depth.diffusion, None, "ckpt={ckpt}");
            let a = depth.aux.expect("depth term");
            assert!(a > 0.0 && a.is_finite(), "ckpt={ckpt}: {a}");
            assert!((depth.total - a).abs() <= 1e-6 * a.abs(), "ckpt={ckpt}");
            let gb: f32 = f
                .set
                .vars
                .iter()
                .skip(1)
                .step_by(2)
                .map(|v| {
                    g.get(v.as_tensor())
                        .map(|t| {
                            t.abs()
                                .unwrap()
                                .sum_all()
                                .unwrap()
                                .to_scalar::<f32>()
                                .unwrap()
                        })
                        .unwrap_or(0.0)
                })
                .sum();
            assert!(gb > 0.0 && gb.is_finite(), "ckpt={ckpt}: LoRA-B grad {gb}");
        }
    }

    /// AC (c): depth off ⇒ bit-identical to the legacy step (its body reproduced here), and a
    /// diffusion-only planned step equals it too. Mutation: scale the diffusion loss (×1.0001) ⇒ red.
    #[test]
    fn depth_off_is_bit_identical_to_the_legacy_step() {
        assert!(load_perceptual_path(
            &TrainingConfig::default(),
            TrainVariant::T2v14b,
            &Device::Cpu
        )
        .unwrap()
        .is_none());
        let f = fixture();
        for v in f.set.vars.iter() {
            v.set(&(v.as_tensor().ones_like().unwrap() * 0.01).unwrap())
                .unwrap();
        }
        let (off, g_off) = step(&f, 0.5, false, None);
        let (x_t, target) = flow_match::build_batch(&f.x0, &f.noise, 0.5).unwrap();
        let ctx = f.dit.embed_text(&f.umt5).unwrap();
        let v = f
            .dit
            .forward(&x_t, &ctx, 0.5 * NUM_TRAIN_TIMESTEPS as f64, &f.cos, &f.sin)
            .unwrap();
        let loss = weighted_velocity_loss(&v, &target, None, false).unwrap();
        let legacy = loss.to_scalar::<f32>().unwrap();
        let g_legacy = loss.backward().unwrap();
        assert_eq!(off.total.to_bits(), legacy.to_bits());
        assert_eq!(
            grad_bits(&g_off, &f.set.vars),
            grad_bits(&g_legacy, &f.set.vars)
        );
        let p = prepared(&f, schedule(0.0, 1.0));
        let plan = plan_in_band(&p, 1, 0, (0.0, 1.0), 0.5).unwrap();
        let s1 = AuxStep {
            path: &p,
            plan,
            entry: 0,
        };
        let (on, g_on) = step(&f, 0.5, false, Some(&s1));
        assert_eq!(on, off);
        assert_eq!(
            grad_bits(&g_on, &f.set.vars),
            grad_bits(&g_off, &f.set.vars)
        );
    }

    /// Each expert trains its aux steps only inside its band: with the depth window `[0.2, 0.6]`
    /// the low-noise band `[0, 0.875]` trains depth inside the window while the high-noise band
    /// `[0.875, 1]` falls through to diffusion; the full band equals `PerceptualPath::plan`; and with
    /// the dual MoE + accumulation 2 every expert's update window is one step kind. Mutations: drop
    /// the band confinement ⇒ red; build the alternation with `accum` alone ⇒ red.
    #[test]
    fn aux_steps_stay_inside_each_experts_band_and_update_window() {
        let f = fixture();
        let p = prepared(&f, schedule(0.2, 0.6));
        let plan = plan_in_band(&p, 2, 0, (0.0, 0.875), 0.4375).unwrap();
        assert!(
            !plan.diffusion && (plan.noise_level - 0.4).abs() < 1e-6,
            "{plan:?}"
        );
        let plan = plan_in_band(&p, 2, 0, (0.875, 1.0), 0.9).unwrap();
        assert!(plan.diffusion && plan.aux.is_empty(), "{plan:?}");
        for key in 1..=4 {
            for t in [0.01f64, 0.5, 0.99] {
                assert_eq!(
                    plan_in_band(&p, key, 0, (0.0, 1.0), t).unwrap(),
                    p.plan(key, 0, t as f32).unwrap()
                );
            }
        }
        let items = 3;
        let mut p = path_with(schedule(0.0, 1.0));
        for e in 0..items {
            p.ensure_reference(e, &latent_frames_nchw(&f.x0).unwrap())
                .unwrap();
        }
        let sched = BucketSchedule::new(
            items,
            &[gen_core::ResolutionBucket {
                resolution: 64,
                repeats: 1,
            }],
            7,
        );
        let mut alt = perceptual_alternation(items, 2, 2);
        let mut kinds: [Vec<bool>; 2] = [Vec::new(), Vec::new()];
        for s in 1..=24u32 {
            let ei = expert_index(s, true);
            let band = if ei == 0 { (0.875, 1.0) } else { (0.0, 0.875) };
            let key = alt.key(s, expert_item(s, true, &sched));
            let plan = plan_in_band(
                &p,
                key,
                expert_cache_index(s, true, &sched),
                band,
                band.0 + 0.01,
            )
            .unwrap();
            kinds[ei].push(!plan.diffusion);
        }
        for (ei, k) in kinds.iter().enumerate() {
            for w in k.chunks(2) {
                assert_eq!(w[0], w[1], "expert {ei}: {k:?}");
            }
            assert!(
                k.contains(&true) && k.contains(&false),
                "expert {ei}: {k:?}"
            );
        }
    }

    /// AC (d), E7: depth grows the guarded footprint by TAEHV + DA2 (more for Large), the guard
    /// refuses at a synthetic budget between base and base+aux (it runs before the dense/checkpoint
    /// choice, so both paths), and the decoder follows the VAE (z48 ⇒ TAEW2.2). Mutation: compare
    /// `base` alone ⇒ red.
    #[test]
    fn memory_guard_counts_the_aux_models_and_the_decoder_follows_the_vae() {
        let mut on = TrainingConfig::default();
        on.depth_anchoring.schedule = schedule(0.0, 1.0);
        let fp = |c: &TrainingConfig, v: TrainVariant| {
            candle_gen_perceptual::perceptual_footprint(
                c,
                &wan_decoder(v),
                candle_gen_perceptual::AuxGeometry::image(1024, 1),
            )
        };
        let small = fp(&on, TrainVariant::T2v14b);
        on.depth_anchoring.model_size = gen_core::train::DepthModelSize::Large;
        let large = fp(&on, TrainVariant::T2v14b);
        assert!(small > 0 && large > small + (1u64 << 30), "{small} {large}");
        let base = 54u64 << 30;
        let v = TrainVariant::T2v14b;
        assert!(check_perceptual_memory(&TrainingConfig::default(), v, 1, base, base).is_ok());
        assert!(check_perceptual_memory(&on, v, 1, base, base + large / 2).is_err());
        assert!(check_perceptual_memory(&on, v, 1, base, base + large + (1 << 30)).is_ok());
        match wan_decoder(TrainVariant::Ti2v5b) {
            candle_gen_perceptual::DecoderSpec::Taehv { name, config } => {
                assert_eq!((name, config), ("TAEW2.2", TaehvConfig::taew2_2()))
            }
            _ => panic!("z48 decodes with TAEW2.2"),
        }
        match wan_decoder(TrainVariant::I2v14b) {
            candle_gen_perceptual::DecoderSpec::Taehv { name, config } => {
                assert_eq!((name, config), ("TAEW2.1", TaehvConfig::taew2_1()))
            }
            _ => panic!("z16 decodes with TAEW2.1"),
        }
    }

    /// AC (e): every Wan descriptor declares depth anchoring; a missing TAEHV checkpoint is a named
    /// error.
    #[test]
    fn descriptors_declare_depth_and_missing_decoder_is_named() {
        for d in [
            trainer_descriptor(),
            trainer_descriptor_i2v_14b(),
            trainer_descriptor_ti2v_5b(),
        ] {
            assert!(d.techniques.depth_anchoring, "{}", d.id);
        }
        let tmp = tempfile::tempdir().unwrap();
        let mut c = TrainingConfig::default();
        c.depth_anchoring.schedule = schedule(0.0, 1.0);
        c.perceptual_decoder_dir = Some(tmp.path().join("no-taehv"));
        c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
        let err = load_perceptual_path(&c, TrainVariant::Ti2v5b, &Device::Cpu)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("TAEW2.2"), "{err}");
    }
}

/// sc-24832: the Candle Wan (video) reference preparation hands each cache entry its item's
/// subject mask — cropped like the image and resampled onto that entry's decoded frame grid — to a
/// mask-reading loss. Mutation: drop `path.attach_subject_masks(masks)` from
/// `prepare_perceptual_references` ⇒ the probe sees no mask ⇒ red.
#[cfg(test)]
mod subject_mask_reference_tests {
    use super::*;
    use std::any::Any;
    use std::sync::{Arc, Mutex};

    use candle_gen::gen_core::train::subject_mask::PerceptualSubjectMasks;
    use candle_gen::train::perceptual::{AuxLoss, AuxLossSchedule, LossReference, PerceptualLoss};
    use candle_gen::train::taehv::{splitmix_uniform, synthetic_taehv_weights, TaehvDecoder};

    type Seen = Arc<Mutex<Vec<Option<(Vec<usize>, Vec<f32>)>>>>;

    /// Records the mask every reference receives.
    struct MaskProbe(Seen);
    impl PerceptualLoss for MaskProbe {
        fn name(&self) -> &'static str {
            "mask-probe"
        }
        fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
            self.reference_with_mask(clean, None)
        }
        fn reference_with_mask(
            &self,
            _clean: &Tensor,
            mask: Option<&Tensor>,
        ) -> Result<Option<LossReference>> {
            candle_gen::lock_recover(&self.0).push(mask.map(|m| {
                (
                    m.dims().to_vec(),
                    m.flatten_all().unwrap().to_vec1().unwrap(),
                )
            }));
            Ok(Some(Box::new(())))
        }
        fn loss(&self, live: &Tensor, _r: &dyn Any) -> Result<Tensor> {
            Ok(live.mean_all()?)
        }
    }

    fn write_png(
        dir: &std::path::Path,
        name: &str,
        f: impl Fn(u32, u32) -> u8,
    ) -> std::path::PathBuf {
        let p = dir.join(name);
        image::GrayImage::from_fn(12, 8, |x, y| image::Luma([f(x, y)]))
            .save(&p)
            .unwrap();
        p
    }

    #[test]
    fn the_video_reference_path_hands_each_entry_its_items_mask() {
        let dir = tempfile::tempdir().unwrap();
        let img = write_png(dir.path(), "img.png", |_, _| 128);
        let masks = [
            write_png(dir.path(), "right.png", |x, _| if x >= 6 { 255 } else { 0 }),
            write_png(dir.path(), "top.png", |_, y| if y < 4 { 255 } else { 0 }),
        ];
        let items: Vec<candle_gen::gen_core::train::TrainingItem> = masks
            .iter()
            .map(|m| {
                let mut it =
                    candle_gen::gen_core::train::TrainingItem::captioned(img.clone(), "c".into());
                it.subject_mask_path = Some(m.clone());
                it
            })
            .collect();
        let mut cfg = TrainingConfig::default();
        cfg.body_losses.normal.weight = 0.1;
        cfg.body_losses.normal_restrict_to_subject = true;
        let tae = TaehvConfig {
            channels: [8, 6, 4, 4],
            ..TaehvConfig::taew2_1()
        };
        let dec = TaehvDecoder::from_weights(
            &synthetic_taehv_weights(&tae, 11, &Device::Cpu).unwrap(),
            tae,
        )
        .unwrap();
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        let mut path = PerceptualPath::new(
            Some(Box::new(dec)),
            vec![AuxLoss {
                schedule: AuxLossSchedule {
                    weight: 0.1,
                    t_min: 0.0,
                    t_max: 1.0,
                    every_n: 1,
                },
                loss: Box::new(MaskProbe(seen.clone())),
            }],
        )
        .unwrap();
        let clean: Vec<Tensor> = (0..2u64)
            .map(|i| splitmix_uniform(&[1, 16, 1, 4, 4], 7 + i, 1.0, 0.0, &Device::Cpu).unwrap())
            .collect();
        let loaded =
            PerceptualSubjectMasks::load("t", &items, &cfg, 1, CropBox::center_square).unwrap();
        let expected = loaded.clone().unwrap();
        prepare_perceptual_references(&mut path, clean.iter(), loaded).unwrap();
        let seen = candle_gen::lock_recover(&seen);
        assert_eq!(seen.len(), 2);
        for (entry, got) in seen.iter().enumerate() {
            let (shape, values) = got.as_ref().expect("every reference gets its mask");
            assert_eq!(shape.len(), 2);
            let (h, w) = (shape[0], shape[1]);
            assert!(h > 0 && w > 0);
            assert_eq!(
                values,
                &expected.pixel_mask(entry, w, h).unwrap(),
                "entry {entry}"
            );
        }
        assert_ne!(seen[0], seen[1], "each entry carries its own item's mask");
    }
}
