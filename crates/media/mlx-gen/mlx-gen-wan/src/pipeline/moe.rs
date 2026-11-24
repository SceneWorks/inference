//! The A14B two-expert denoise: boundary crossing, ranges and the staged expert swap.

use super::*;

/// One MoE expert: a full transformer + its own (per-model) embedded contexts + guidance scale.
/// Wan2.2-A14B's "MoE" is two complete checkpoints, not token routing — each carries its own
/// `text_embedding`, so contexts are embedded per expert.
pub struct Expert<'a> {
    pub transformer: &'a WanTransformer,
    /// `embed_text` output for this expert (cond).
    pub ctx_cond: Array,
    /// `embed_text` output for this expert (uncond); `None` ⇒ CFG disabled for this expert.
    pub ctx_uncond: Option<Array>,
    /// This expert's guidance scale (the `low`/`high` of the dual `sample_guide_scale`).
    pub guidance: f32,
}

/// First step index whose integer timestep drops **below** `boundary_timestep` — the single high→low
/// MoE crossing (sc-12736, mirroring candle's `crossing_index`). Steps `0..k` run the high-noise
/// expert (`t ≥ boundary_timestep`), `k..steps` the low-noise expert. The flow-match integer timesteps
/// are monotonically non-increasing (built from a decreasing σ schedule), so this prefix/suffix split
/// is **exactly** the per-step `t ≥ boundary_timestep` choice the resident loop makes — returns
/// `timesteps.len()` if the boundary is never crossed (all high), `0` if it is crossed at the first
/// step (all low). Pure + GPU-free, so both the resident [`denoise_moe`] and the sequential
/// expert-swap share it and a unit test can pin the split against the per-step rule.
pub fn crossing_index(timesteps: &[f32], boundary_timestep: f32) -> usize {
    timesteps
        .iter()
        .position(|&t| t < boundary_timestep)
        .unwrap_or(timesteps.len())
}

/// Run the denoise steps `range` on a **single** expert through [`WanNativeDriver`], carrying the
/// request-owned [`mlx_gen::driver::DriverRunState`] solver history across compatible expert
/// crossings (`PIPE-3`, `PIPE-7`, `SW-6968`).
///
/// Both the resident MoE loop ([`denoise_moe`]) and the sequential expert-swap drive the
/// **identical** typed driver slice over their respective step ranges: when `range.start == 0`
/// the slice enters via `Start/Reset`, and when `range.start > 0` it continues via `Continue/Carry`
/// so the multistep `FlowUniPC` / `FlowDpmpp2m` history survives the expert swap bit-for-bit.
/// An empty `range` is a no-op (never builds an unused `StepCache`).
#[allow(clippy::too_many_arguments)]
pub fn denoise_expert_slice(
    kind: SolverKind,
    num_train_timesteps: usize,
    total_steps: usize,
    shift: f32,
    e: &Expert,
    grid: (usize, usize, usize),
    y: Option<&Array>,
    latents: &mut Array,
    range: std::ops::Range<usize>,
    phase: usize,
    cancel: &CancelFlag,
    driver_state: &mut mlx_gen::driver::DriverRunState,
    on_step: &mut dyn FnMut(usize),
) -> Result<()> {
    if range.is_empty() {
        return Ok(());
    }
    let cache = build_cache(e.transformer, &e.ctx_cond, e.ctx_uncond.as_ref(), grid)?;
    let driver = WanNativeDriver::new(
        "wan_native_moe",
        kind,
        num_train_timesteps,
        total_steps,
        shift,
        |x: &Array, _step: usize, t: f32| predict(e.transformer, x, t, &cache, e.guidance, y),
    );
    let (transition, history) = if range.start == 0 {
        (
            mlx_gen::driver::SliceTransition::Start,
            mlx_gen::driver::DriverHistoryPolicy::Reset,
        )
    } else {
        (
            mlx_gen::driver::SliceTransition::Continue,
            mlx_gen::driver::DriverHistoryPolicy::Carry,
        )
    };
    let mut on_progress = |progress: Progress| {
        if let Progress::Step { current, .. } = progress {
            on_step(current as usize);
        }
    };
    let mut hooks = mlx_gen::driver::StepHooks::new(&mut on_progress);
    let out = mlx_gen::driver::run_mlx_driver_slices(
        &driver,
        [mlx_gen::driver::SliceSpec {
            pass: 0,
            phase,
            span: range,
            transition,
            history,
            geometry: None,
            request_boundary_estimate: false,
        }],
        mlx_gen::gen_core::BoundaryState::XSigma(latents.clone()),
        total_steps,
        0,
        cancel,
        None,
        driver_state,
        &mut hooks,
        |_, out, _| Ok(out),
    )?;
    *latents = match out {
        mlx_gen::gen_core::BoundaryState::XSigma(x)
        | mlx_gen::gen_core::BoundaryState::CleanEstimate { latent: x, .. }
        | mlx_gen::gen_core::BoundaryState::RouteSpecific { state: x, .. } => x,
    };
    Ok(())
}

/// The dual-expert MoE denoise loop (Wan2.2-A14B). The **high-noise** expert runs while the integer
/// timestep is `≥ boundary_timestep` (`config.boundary · num_train_timesteps`, e.g. `0.875 · 1000 =
/// 875`) and the **low-noise** expert below it — switching the transformer, the per-expert contexts,
/// and the per-expert guidance together. Reduces to [`denoise`] when both experts are the same model.
///
/// This is the **resident** path: BOTH experts are held resident by the caller for the whole loop. The
/// single high→low boundary crossing is a prefix/suffix split ([`crossing_index`] + two
/// [`denoise_expert_slice`] calls sharing one request-owned [`mlx_gen::driver::DriverRunState`]
/// via `Carry`), so it is bit-exact to the single-scheduler choice — and drives the *identical*
/// per-step math the sequential expert-swap (sc-12736) drives, so the residency change stays
/// numerics-preserving.
///
/// `y` is the optional I2V-14B channel-concat conditioning `[20, F, H, W]` ([`build_i2v_y`]),
/// concatenated onto each forward's noise latent (see `predict`); `None` for T2V. It is constant
/// across steps and shared by both experts (the conditioning doesn't change with the noise level).
#[allow(clippy::too_many_arguments)]
pub fn denoise_moe(
    low: &Expert,
    high: &Expert,
    boundary_timestep: f32,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    y: Option<&Array>,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let sigmas = compute_sigmas(steps, shift, num_train_timesteps);
    let timesteps = crate::scheduler::integer_timesteps(&sigmas, num_train_timesteps);
    mlx_gen::video_execution::sampling(
        "wan",
        &format!("{kind:?}"),
        "flow_velocity",
        serde_json::json!({"low_guidance_bits":low.guidance.to_bits(),"high_guidance_bits":high.guidance.to_bits(),"shift_bits":shift.to_bits()}),
    );
    mlx_gen::video_execution::schedule("wan", &sigmas);

    // sc-2957: compiled elementwise glue (bit-exact, ~14% faster/step) — see `denoise`. Scoped +
    // restored on drop by the RAII guard (F-006/F-007) instead of leaking into later work.
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();

    // The grid is shared — the channel-concat `y` doesn't change F/H/W and each expert's contexts are
    // constant across steps. The single high→low crossing: high runs `0..k` (`t ≥ boundary`), low runs
    // `k..steps`. One request-owned `DriverRunState` carries multistep solver history across the split.
    let grid = low.transformer.patch_grid(init_noise);
    let k = crossing_index(&timesteps, boundary_timestep);
    let mut latents = init_noise.clone();
    let mut driver_state = mlx_gen::driver::DriverRunState::new();
    denoise_expert_slice(
        kind,
        num_train_timesteps,
        steps,
        shift,
        high,
        grid,
        y,
        &mut latents,
        0..k,
        0,
        cancel,
        &mut driver_state,
        on_step,
    )?;
    denoise_expert_slice(
        kind,
        num_train_timesteps,
        steps,
        shift,
        low,
        grid,
        y,
        &mut latents,
        k..timesteps.len(),
        1,
        cancel,
        &mut driver_state,
        on_step,
    )?;
    Ok(latents)
}

/// The dual-expert MoE denoise loop driven by a **curated unified solver** (epic 7114, sc-7121) — the
/// additive fold onto the shared gen-core solver library, alongside the native [`denoise_moe`]. Same
/// rationale as [`denoise_curated`]: the native `unipc`/`dpmpp2m` (flow-SNR) stay native (N1), and this
/// path serves the gen-core-only curated solvers. The boundary expert swap is applied inside the
/// predict closure (the integer timestep `(σ·num_train).trunc()` is compared to `boundary_timestep`),
/// so a multi-eval solver re-evaluates the correct expert at its intermediate σ.
#[allow(clippy::too_many_arguments)]
pub fn denoise_moe_curated(
    low: &Expert,
    high: &Expert,
    boundary_timestep: f32,
    sampler_name: &str,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    y: Option<&Array>,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();
    let grid = low.transformer.patch_grid(init_noise);
    let low_cache = build_cache(
        low.transformer,
        &low.ctx_cond,
        low.ctx_uncond.as_ref(),
        grid,
    )?;
    let high_cache = build_cache(
        high.transformer,
        &high.ctx_cond,
        high.ctx_uncond.as_ref(),
        grid,
    )?;
    let sigmas = compute_sigmas(steps, shift, num_train_timesteps);
    mlx_gen::video_execution::curated_sampling(
        "wan_curated",
        sampler_name,
        "flow_velocity",
        serde_json::json!({"low_guidance_bits":low.guidance.to_bits(),"high_guidance_bits":high.guidance.to_bits()}),
    );
    mlx_gen::video_execution::schedule("wan_curated", &sigmas);
    let nt = num_train_timesteps as f32;
    mlx_gen::run_flow_sampler(
        mlx_gen::SamplerRun {
            sampler_name: Some(sampler_name),
            sigmas: &sigmas,
            seed,
            cancel,
            on_progress,
        },
        mlx_gen::TimestepConvention::Sigma,
        init_noise.clone(),
        |x, sigma| {
            let t = (sigma * nt).trunc();
            let (e, cache) = if t >= boundary_timestep {
                (high, &high_cache)
            } else {
                (low, &low_cache)
            };
            predict(e.transformer, x, t, cache, e.guidance, y)
        },
    )
}

/// Sequential-residency twin of [`denoise_moe_curated`]. The curated solver remains in control of
/// every model evaluation (including Heun / DPM++ SDE sub-evaluations); this wrapper changes only
/// which expert is resident for that evaluation. Curated sigma trajectories are monotonically
/// descending, so there is at most one high→low transition. On that transition the outgoing expert
/// is dropped and its MLX buffers are flushed before the incoming expert is loaded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_moe_curated_swapped(
    boundary_timestep: f32,
    sampler_name: &str,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    y: Option<&Array>,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut load: impl FnMut(bool) -> Result<(WanTransformer, Array, Option<Array>, f32)>,
) -> Result<Array> {
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();
    let (transformer, cond, uncond, guidance) = load(true)?;
    let grid = transformer.patch_grid(init_noise);
    let cache = build_cache(&transformer, &cond, uncond.as_ref(), grid)?;
    // The first shifted Wan sigma is always above the configured A14B boundary. Keeping the
    // initially loaded high expert here avoids a second load before the first evaluation.
    let mut active = Some((transformer, cache, guidance, true));
    let sigmas = compute_sigmas(steps, shift, num_train_timesteps);
    mlx_gen::video_execution::curated_sampling(
        "wan_curated",
        sampler_name,
        "flow_velocity",
        serde_json::json!({"guidance_bits":guidance.to_bits()}),
    );
    mlx_gen::video_execution::schedule("wan_curated", &sigmas);
    let nt = num_train_timesteps as f32;
    let result = mlx_gen::run_flow_sampler(
        mlx_gen::SamplerRun {
            sampler_name: Some(sampler_name),
            sigmas: &sigmas,
            seed,
            cancel,
            on_progress,
        },
        mlx_gen::TimestepConvention::Sigma,
        init_noise.clone(),
        |x, sigma| {
            let t = (sigma * nt).trunc();
            let wants_high = t >= boundary_timestep;
            let has_high = active.as_ref().is_some_and(|(_, _, _, high)| *high);
            if wants_high != has_high {
                // A descending sigma schedule may cross high→low once, never low→high.
                if wants_high {
                    return Err(Error::Msg(
                        "wan: curated sampler timestep increased across the MoE boundary".into(),
                    ));
                }
                // SC-20686 Metal lane: the swap (drop, evict, load) is its own phase window.
                mlx_gen::sc20686::mark_phase("load");
                drop(active.take());
                mlx_rs::memory::clear_cache();
                let (transformer, cond, uncond, guidance) = load(false)?;
                let cache = build_cache(&transformer, &cond, uncond.as_ref(), grid)?;
                active = Some((transformer, cache, guidance, false));
            }
            let (transformer, cache, guidance, _) =
                active.as_ref().expect("curated expert is loaded");
            predict(transformer, x, t, cache, *guidance, y)
        },
    );
    drop(active);
    mlx_rs::memory::clear_cache();
    result
}

/// Drive the A14B high→low MoE expert swap so the two ~8-9 GB experts are **never co-resident**
/// (sc-12736, epic 12732 — the mlx Pillar-1 win, the residency twin of candle's `staged_expert_swap`):
/// load the high expert, `use_high` it over steps `0..k`, DROP it + `evict`, then load the low expert
/// and `use_low` it over `k..steps`. Each expert is loaded, used, then **explicitly dropped before the
/// next loads** — the evict-then-load discipline: MLX frees the dropped expert's arrays to its buffer
/// cache and `evict` (`mlx_rs::memory::clear_cache`) returns them to the OS *before* the incoming
/// expert materializes, so the peak is one expert, not both. A naive "load low, then drop high" would
/// momentarily co-reside both and the footprint would NOT drop; the ordering here is the whole point.
///
/// There is exactly ONE boundary crossing per denoise (`t ≥ boundary` ⇒ high, below ⇒ low; flow-match
/// timesteps decrease monotonically), so at most one swap. An expert whose step range is empty
/// (`k == 0` all-low, or `k == steps` all-high) is skipped entirely — it never loads.
///
/// Callers must `eval` any output that must survive the drop (the latents) *before* the expert is
/// dropped, or MLX's lazy graph would keep the expert's weights referenced and freeing it would reclaim
/// nothing — [`denoise_expert_slice`] already `eval`s the latents after every step, so by the time `use_high`
/// returns they are materialized and independent of the expert.
///
/// Generic over the expert type `E` and threaded state `St` so a weight-free unit test can pin the
/// never-co-resident property with a lightweight liveness witness — no GPU, no real weights — exactly as
/// the resident/sequential parity rests on [`denoise_expert_slice`]. The load closures receive `&mut St` so
/// they can emit their [`Progress::Loading`] before the (heavy) load; the use closures receive `&mut St`
/// to advance the shared scheduler/latents.
#[allow(clippy::too_many_arguments)]
pub fn staged_expert_swap<E, St>(
    k: usize,
    steps: usize,
    state: &mut St,
    load_high: impl FnOnce(&mut St) -> Result<E>,
    use_high: impl FnOnce(&E, &mut St) -> Result<()>,
    load_low: impl FnOnce(&mut St) -> Result<E>,
    use_low: impl FnOnce(&E, &mut St) -> Result<()>,
    mut evict: impl FnMut() -> Result<()>,
) -> Result<()> {
    if k > 0 {
        let high = load_high(state)?;
        use_high(&high, state)?;
        // Evict-then-load: drop the high expert (its arrays return to MLX's buffer cache) BEFORE the
        // low expert is ever loaded, then flush the cache to the OS — the never-co-resident invariant.
        drop(high);
        evict()?;
    }
    if k < steps {
        let low = load_low(state)?;
        use_low(&low, state)?;
        drop(low);
        evict()?;
    }
    Ok(())
}
