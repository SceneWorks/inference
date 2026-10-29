//! Resolution-bucket geometry and the pre-flight memory guard.

use super::*;

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, as a function of the LTX
/// latent token count `s` (the trainer trains single-frame still latents, so `s = (edge/32)²`). With
/// attention-segment checkpointing always on (sc-4942) the seq² attention term is demoted to a single
/// layer's backward transient, so the measured f32 curve is essentially **linear** in `s`: the
/// constant is the resident base (the Q-packed 22B DiT after the Gemma TE is freed) plus the
/// f32-activation working set, the slope the per-token retained hidden activations across the 48
/// blocks.
///
/// CALIBRATED from `first_step_attn_ckpt_sweep` (128 GB Mac, rank 8 / 384 targets / batch 1, f32):
/// s=256 → 23.3 GB, s=576 → 31.5 GB, s=1024 → 42.6 GB (fit error < 0.2 GB). Refit if that harness
/// prints materially different numbers. (LTX trains f32 only — see the `compute_dtype` note in
/// `train_impl`; there is no bf16 production path to size.)
pub(super) fn projected_dense_peak_gb(s: f64) -> f64 {
    16.9 + 0.0251 * s
}

/// Latent tokens per side for a bucketed pixel `edge` (the VAE's ×32 spatial stride, floor 1).
pub(super) fn latent_edge_of(edge: u32) -> usize {
    (edge / SPATIAL_SCALE as u32).max(1) as usize
}

/// The latent edge the pre-flight guard and previews size for: the largest bucket's (epic 2123 E7).
pub(super) fn guard_latent_edge(edges: &[u32]) -> usize {
    edges.iter().map(|&e| latent_edge_of(e)).max().unwrap_or(1)
}

/// The item-major cache entry and its bucket (whose RoPE grid the step uses) for 1-based `step`
/// (sc-2127). With one bucket the entry is the pre-bucket `(step-1) % n_items` and the bucket is 0.
pub(super) fn step_entry(schedule: &BucketSchedule, step: u32) -> (usize, usize) {
    let entry = schedule.cache_index((step - 1) as usize);
    (entry, entry % schedule.n_buckets())
}

/// The bucket index of the largest edge in `edges` (the first on a tie; `0` for an empty list) —
/// the bucket training previews render at (sc-2127).
pub(super) fn largest_bucket(edges: &[usize]) -> usize {
    edges
        .iter()
        .enumerate()
        .fold(0, |best, (i, &e)| if e > edges[best] { i } else { best })
}

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead (sc-4942 — the sc-4874 mechanism).
/// `latent_edge` is the latent tokens per side (`edge/32`); the token count is `latent_edge²` (the
/// trainer trains single-frame still latents). The budget is MLX's reported memory limit (≈ the
/// device's recommended working set) × 0.85 for worker/host headroom. Only consulted when gradient
/// checkpointing is OFF. With resolution buckets the caller passes the LARGEST bucket's latent edge
/// (sc-2127 / epic 2123 E7): the per-step working set peaks at the biggest latent, and the cached
/// still latents (KBs–MBs per item) are not part of the projection.
///
/// Epic 2123 E7 (sc-24830): `extra_gb` is the training-time auxiliary models' footprint
/// ([`perceptual_footprint_gb`]); with it the guard also runs with gradient checkpointing on
/// (`checkpointed`), stacking on the resident base ([`checkpointed_baseline_gb`]).
pub(super) fn preflight_memory_guard(
    cfg: &TrainingConfig,
    latent_edge: usize,
    extra_gb: f64,
    checkpointed: bool,
) -> Result<()> {
    preflight_memory_guard_with_budget(cfg, latent_edge, extra_gb, checkpointed, get_memory_limit())
}

/// The checkpointed baseline the auxiliary-model guard stacks on: the resident-base term of
/// [`projected_dense_peak_gb`] (block checkpointing removes most of the activation term; the base
/// stays). A lower bound.
pub(super) fn checkpointed_baseline_gb() -> f64 {
    projected_dense_peak_gb(0.0)
}

/// [`preflight_memory_guard`] against an explicit budget (`budget_bytes`, the live MLX limit in
/// production) — so the guard's arithmetic is testable on any host. A checkpointed refusal goes
/// through the shared [`mlx_gen_perceptual::check_aux_memory`], naming `cfg`'s enabled aux losses.
pub(super) fn preflight_memory_guard_with_budget(
    cfg: &TrainingConfig,
    latent_edge: usize,
    extra_gb: f64,
    checkpointed: bool,
    budget_bytes: usize,
) -> Result<()> {
    let s = (latent_edge * latent_edge) as f64;
    let projected = if checkpointed {
        checkpointed_baseline_gb()
    } else {
        projected_dense_peak_gb(s)
    } + extra_gb;
    let budget_gb = budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if checkpointed {
        let px = latent_edge * SPATIAL_SCALE as usize;
        return mlx_gen_perceptual::check_aux_memory(
            "ltx_2_3 trainer",
            cfg,
            &format!("a checkpointed training step at resolution {px}"),
            extra_gb,
            projected,
            safe,
            &format!("{budget_gb:.0} GB MLX limit × 0.85"),
        );
    }
    if projected > safe {
        let px = latent_edge * SPATIAL_SCALE as usize;
        return Err(mlx_gen_perceptual::name_aux_losses(
            format!(
                "ltx_2_3 trainer: a dense first training step at resolution {px} needs ~{projected:.0} GB \
                 (the forward working set materializes in one allocation), exceeding this machine's ~{safe:.0} GB \
                 safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS would hard-kill the \
                 worker (SIGKILL) at the first step with no recoverable error (sc-4874/sc-4942). Enable Gradient \
                 Checkpointing (recomputes block activations in the backward) or reduce the training resolution."
            )
            .into(),
            cfg,
            extra_gb,
        ));
    }
    Ok(())
}
