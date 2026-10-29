use super::*;

/// The per-step loss breakdown [`compute_step_loss_grads`] returns (epic 2123 E8).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct StepLosses {
    /// The differentiated step loss.
    pub(super) total: f32,
    /// The diffusion (velocity-regression) term, `None` on an aux-only step.
    pub(super) diffusion: Option<f32>,
    /// The weighted aux-loss term, `None` when no aux loss contributed this step.
    pub(super) aux: Option<f32>,
}

/// One aux-loss step's view of the trainer's [`PerceptualPath`].
pub(super) struct AuxStep<'a> {
    pub(super) path: &'a PerceptualPath,
    pub(super) plan: &'a StepPlan,
    /// The step's (item, bucket) cache entry (selects its cached reference).
    pub(super) image: usize,
    /// The entry's latent tokens per side (to unpatchify the x0 estimate).
    pub(super) latent_edge: usize,
}

/// The LTX-2.5 loop's [`AuxDriver`] (sc-24830): each example's depth reference (its clean video
/// target tokens at the selected frames, decoded) once per job, and the alternation over the
/// loop's round-robin order (`(step − 1) mod n`, an `n`-step epoch) with `accumulation`
/// micro-steps per update.
pub(super) fn ltx25_aux_driver(
    path: PerceptualPath,
    cached: &[CachedLtx25Example],
    accumulation: u32,
    cancel: &mlx_gen::gen_core::runtime::CancelFlag,
) -> Result<AuxDriver> {
    let n = cached.len();
    AuxDriver::prepare_keyed(
        path,
        n,
        |index| {
            let example = &cached[index];
            let (Some(video), Some(depth)) = (example.video.as_ref(), example.depth.as_ref())
            else {
                return Err(format!(
                    "ltx_2_5 trainer: depth anchoring needs a generated video clip (example {index})"
                )
                .into());
            };
            depth.nchw(&video.clean)
        },
        n,
        accumulation,
        cancel,
    )
}

/// LTX-2.3's latent family for the shared aux-loss builder (epic 2123 E8): the 128-channel
/// LTX-2.3 VAE latent normalized by its per-channel statistics (`(μ − mean)/std`,
/// `LtxVideoVae::encode`), decoded by TAELTX2.3 — upstream's TAEHV checkpoint for LTX-2.3, which
/// takes exactly that normalized latent (its demo feeds `(z − latents_mean)/latents_std`). A still
/// image is one `T = 1` clip (8 grown frames, the last kept).
pub(super) fn ltx_decoder() -> mlx_gen_perceptual::DecoderSpec {
    mlx_gen_perceptual::DecoderSpec::Taehv {
        name: "TAELTX2.3",
        config: TaehvConfig::taeltx2_3(),
    }
}

/// Build the epic-2123 perceptual path through the shared builder: `None` when no aux loss is
/// enabled (nothing loads; every step is the plain diffusion step).
pub(super) fn load_perceptual_path(cfg: &TrainingConfig) -> Result<Option<PerceptualPath>> {
    load_perceptual_path_for(cfg, "ltx_2_3 trainer", ltx_decoder())
}

/// [`load_perceptual_path`] with an explicit error label and decoder (the LTX-2.5 route uses its
/// own: [`ltx25_decoder`]).
pub(super) fn load_perceptual_path_for(
    cfg: &TrainingConfig,
    label: &str,
    decoder: mlx_gen_perceptual::DecoderSpec,
) -> Result<Option<PerceptualPath>> {
    mlx_gen_perceptual::build_perceptual_path(
        cfg,
        &mlx_gen_perceptual::AuxLossContext {
            label,
            decoder,
            latent_lpips: None,
        },
    )
}

/// The LTX-2.3 loop's [`AuxDriver`] (epic 2123 E8): every (item, bucket) cache entry's perceptual
/// reference computed once, from its patchified clean latent unpatchified at its bucket's latent
/// edge (item-major: bucket = entry % n_buckets); the alternation over the schedule's epochs with
/// `accum` micro-steps per update.
pub(super) fn aux_driver(
    path: PerceptualPath,
    cache: &[(Array, Array, Option<Array>)],
    latent_edges: &[usize],
    schedule: &BucketSchedule,
    accum: u32,
    cancel: &mlx_gen::gen_core::runtime::CancelFlag,
) -> Result<AuxDriver> {
    AuxDriver::prepare(
        path,
        cache.len(),
        |entry| unflatten_latent(&cache[entry].0, latent_edges[entry % latent_edges.len()]),
        schedule,
        accum,
        cancel,
    )
}

/// Extra training memory (GB) the enabled perceptual losses add at the largest bucket pixel `edge`
/// (epic 2123 E7): TAELTX2.3 + the losses' frozen models plus `entries` cached references. `0`
/// when nothing is enabled.
pub(super) fn perceptual_footprint_gb(cfg: &TrainingConfig, edge: u32, entries: usize) -> f64 {
    mlx_gen_perceptual::perceptual_footprint_gb(
        cfg,
        &ltx_decoder(),
        mlx_gen_perceptual::AuxGeometry::image(edge, entries),
    )
}

/// The largest prepared video `(H, W)` latent grid and the frame count over `items` (sc-24830) —
/// read from each bundle's `videoShape` header metadata alone (no tensor data) before caching, for
/// the depth memory estimate.
pub(super) fn ltx25_video_grids(items: &[mlx_gen::TrainingItem]) -> Result<(i32, i32, i32)> {
    let mut max = (0, 0, 0);
    for item in items {
        let path = prepared_bundle_path(item)?;
        let metadata =
            mlx_gen::gen_core::weightsmeta::safetensors_file_metadata(path).map_err(|error| {
                mlx_gen::Error::Msg(format!(
                    "ltx_2_5 trainer: could not read prepared bundle header `{}`: {error}",
                    path.display()
                ))
            })?;
        let raw = metadata.get("videoShape").ok_or_else(|| {
            mlx_gen::Error::Msg(
                "ltx_2_5 trainer: depth anchoring needs the prepared bundle's `videoShape`".into(),
            )
        })?;
        let shape: Vec<i32> = serde_json::from_str(raw).map_err(|error| {
            mlx_gen::Error::Msg(format!(
                "ltx_2_5 trainer: prepared bundle `videoShape` is not a JSON shape: {error}"
            ))
        })?;
        if shape.len() != 5 {
            return Err(
                format!("ltx_2_5 trainer: videoShape must be [B,C,F,H,W], got {shape:?}").into(),
            );
        }
        max = (
            max.0.max(shape[2]),
            max.1.max(shape[3]),
            max.2.max(shape[4]),
        );
    }
    Ok(max)
}

/// Extra training memory (GB) LTX-2.5 depth anchoring adds (sc-24830, epic 2123 E7): TAELTX2.3 +
/// the losses' frozen models for `frames` selected latent frames per step at the `(h, w)` latent
/// grid (×32 pixels) — each frame's decode sized as a `T = 2` clip with its predecessor
/// ([`ltx25_decoder`]) — plus `entries` cached references. `0` when nothing is enabled.
pub(super) fn ltx25_perceptual_footprint_gb(
    cfg: &TrainingConfig,
    h: i32,
    w: i32,
    frames: u32,
    entries: usize,
) -> f64 {
    mlx_gen_perceptual::perceptual_footprint_gb(
        cfg,
        &ltx25_decoder(),
        mlx_gen_perceptual::AuxGeometry {
            height: (h as i64 * SPATIAL_SCALE) as u32,
            width: (w as i64 * SPATIAL_SCALE) as u32,
            frames,
            entries,
        },
    )
}

/// The LTX-2.5 depth memory guard (sc-24830, epic 2123 E7): refuse when the resident DiT base (the
/// calibrated resident term [`checkpointed_baseline_gb`] — a lower bound; the 2.5 trainer has no
/// fitted AV working-set curve) plus the aux models exceeds `budget_bytes × 0.85`, through the
/// shared [`mlx_gen_perceptual::check_aux_memory`] (naming `cfg`'s enabled aux losses).
pub(super) fn ltx25_aux_memory_guard(
    cfg: &TrainingConfig,
    extra_gb: f64,
    budget_bytes: usize,
) -> Result<()> {
    let projected = checkpointed_baseline_gb() + extra_gb;
    let budget_gb = budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    mlx_gen_perceptual::check_aux_memory(
        "ltx_2_5 trainer",
        cfg,
        &format!("a training step decoding the selected `{DEPTH_ANCHORING_FRAMES_KEY}` frames"),
        extra_gb,
        projected,
        safe,
        &format!("{budget_gb:.0} GB MLX limit × 0.85"),
    )
}
