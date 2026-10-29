//! The dense, approximate, curated and TI2V denoise loops and their per-step cache.

use super::*;

/// Classifier-free guidance combine: `uncond + gs·(cond − uncond)`.
pub(super) fn cfg_combine(cond: &Array, uncond: &Array, gs: f32) -> Result<Array> {
    Ok(add(
        uncond,
        &multiply(&subtract(cond, uncond)?, scalar(gs))?,
    )?)
}

/// Per-generate caches for one transformer/expert, constant across every denoise step: the bf16 RoPE
/// `(cos, sin)` for the (fixed) grid + each block's cross-attention K/V for the (CFG-batched) context.
/// Mirrors the reference's `prepare_rope` / `prepare_cross_kv`, computed once before the loop.
pub(super) struct StepCache {
    /// Per-block cross-attention `(k, v)`, each `[batch, n, text_len, d]` (bf16).
    pub(super) cross_kv: Vec<(Array, Array)>,
    pub(super) cos: Array,
    pub(super) sin: Array,
    /// Forward batch width: 2 when CFG is on (cond+uncond stacked), else 1.
    pub(super) batch: usize,
    /// SC-20686 Metal-lane ownership of `cross_kv` (`None` unless a campaign is armed). Declared
    /// last so the K/V arrays are freed before the post-release allocator remnant is sampled.
    pub(super) _sc20686: Option<mlx_gen::sc20686::CacheSetGuard>,
}

/// SC-20686 operation names for the Wan cross-K/V cache lifecycle (Metal lane).
pub(crate) const SC20686_CROSS_KV_CREATE: &str = "WanTransformer::prepare_cross_kv";

pub(crate) const SC20686_CROSS_KV_RELEASE: &str = "StepCache::drop";

/// Build the per-expert [`StepCache`] from the embedded contexts + the (constant) RoPE grid. When CFG
/// is on (`ctx_uncond = Some`) the cond/uncond contexts are stacked on the batch axis so the cross-K/V
/// is `B=2`; otherwise `B=1`. The caches are evaluated once here (the reference's `mx.eval(cross_kv,
/// rope_cos_sin)`) so each per-step graph reuses them instead of recomputing.
pub(super) fn build_cache(
    transformer: &WanTransformer,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    grid: (usize, usize, usize),
) -> Result<StepCache> {
    let (context_batch, batch) = match ctx_uncond {
        Some(uncond) => (concatenate_axis(&[ctx_cond, uncond], 0)?, 2),
        None => (ctx_cond.clone(), 1),
    };
    mlx_gen::sc20686::mark_phase("prepare-cache");
    let cross_kv = transformer.prepare_cross_kv(&context_batch)?;
    let (cos, sin) = transformer.prepare_rope(grid)?;
    let mut to_eval: Vec<&Array> = vec![&cos, &sin];
    for (k, v) in &cross_kv {
        to_eval.push(k);
        to_eval.push(v);
    }
    mlx_rs::transforms::eval(to_eval)?;
    // Inert (`None`, no work) unless an SC-20686 campaign is armed on this thread.
    let sc20686 = mlx_gen::sc20686::register_cross_kv_set(
        &cross_kv,
        (grid.0 * grid.1 * grid.2) as u64,
        SC20686_CROSS_KV_CREATE,
        SC20686_CROSS_KV_RELEASE,
    )?;
    mlx_gen::sc20686::mark_denoise();
    Ok(StepCache {
        cross_kv,
        cos,
        sin,
        batch,
        _sc20686: sc20686,
    })
}

/// One denoise prediction reusing the precomputed [`StepCache`]: a single batched forward yielding
/// `[cond, uncond]`, combined as `uncond + gs·(cond − uncond)` when CFG is on, else the B=1 cond-only
/// forward.
///
/// `y` is the optional I2V channel-concat conditioning `[20, F, H, W]` (mirrors `WanModel.__call__`'s
/// `y`): when `Some`, it is concatenated **onto the channel axis after** the `[16, …]` noise latent —
/// `[noise(16), mask(4), z_video(16)]` → `[36, F, H, W]` — before patchify, exactly the channel order
/// the I2V-14B `patch_embedding` (in_dim 36) was trained on. The DiT prediction stays `out_dim = 16`,
/// so the scheduler step still consumes/produces the 16-channel latent.
pub(super) fn predict(
    transformer: &WanTransformer,
    latents: &Array,
    t: f32,
    cache: &StepCache,
    guidance: f32,
    y: Option<&Array>,
) -> Result<Array> {
    predict_approx(transformer, latents, t, cache, guidance, y, None, None)
}

/// [`predict`] carrying the sc-18322 denoise feature cache. `trunk: None` is exactly [`predict`].
#[allow(clippy::too_many_arguments)]
pub(super) fn predict_approx(
    transformer: &WanTransformer,
    latents: &Array,
    t: f32,
    cache: &StepCache,
    guidance: f32,
    y: Option<&Array>,
    trunk: Option<&mut crate::feature_cache::TrunkCache>,
    prune: Option<&crate::token_pruning::TokenPruner>,
) -> Result<Array> {
    let x = match y {
        Some(y) => concatenate_axis(&[latents, y], 0)?,
        None => latents.clone(),
    };
    let preds = transformer.forward_cached_approx(
        &x,
        t,
        &cache.cross_kv,
        &cache.cos,
        &cache.sin,
        cache.batch,
        trunk,
        prune,
    )?;
    if cache.batch == 2 {
        // preds[0] = cond (context row 0), preds[1] = uncond (row 1).
        cfg_combine(&preds[0], &preds[1], guidance)
    } else {
        preds
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("wan: B=1 forward produced no output".into()))
    }
}

/// The dense denoise loop (single model). `ctx_cond`/`ctx_uncond` are
/// [`WanTransformer::embed_text`] outputs; pass `ctx_uncond = None` for the CFG-disabled B=1 fast
/// path. `init_noise` is `[C, F, H, W]` f32. Returns the denoised latents `[out_dim, F, H, W]`
/// (f32). `on_step(i)` is called after each completed step.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    transformer: &WanTransformer,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_noise: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    denoise_approx(
        transformer,
        kind,
        num_train_timesteps,
        steps,
        shift,
        guidance,
        ctx_cond,
        ctx_uncond,
        init_noise,
        cancel,
        on_step,
        None,
        None,
    )
}

/// Refuse an approximate plan on a denoise route that does not implement it (sc-18322).
///
/// The routes that are **not** wired for the denoise feature cache — the TI2V mask-blend, the curated
/// unified solvers, and the MoE expert-swap paths — must refuse rather than run exactly, because
/// silently ignoring a selection is the defect the typed approximate surface exists to remove. Each
/// has a structural reason it is unwired, named in the message so a caller knows it is not a gap
/// waiting to be filled by a retry:
///
/// * **TI2V mask-blend** re-blends the conditioning latent into the trajectory after every step, so a
///   trunk residual captured at one step is not the quantity a later step needs.
/// * **Curated solvers** evaluate the model 1..N times per solver step (Heun twice, DPM++ SDE at a
///   midpoint), so "the previous step's residual" is ambiguous — and the step index the cache keys on
///   never reaches the predict closure.
/// * **MoE expert swap** changes transformers mid-trajectory, so a residual captured under the
///   high-noise expert is meaningless under the low-noise one, a rotation phase captured under one is
///   not the other's, and retaining either across the swap would keep the outgoing expert's weights
///   alive.
///
/// Each of those three has a live call site: `Wan::generate_impl`'s TI2V and curated arms, and
/// `Wan14b::generate_impl`, which covers every A14B route because every A14B route swaps experts. The
/// refusal names the plan's **actually selected** mechanisms rather than a fixed string, so a
/// pruning-only plan is not refused with the cache's name.
///
/// Unreachable today — the shared floor refuses every approximate selection before a provider is
/// called, so no plan reaching here can be anything but `Exact`. It exists so that the day a
/// characterization artifact makes selection possible, these routes fail closed by construction
/// instead of needing to be remembered.
pub fn refuse_unwired_approximation(
    id: &str,
    route: &str,
    plan: &mlx_gen::gen_core::ApproximationPlan,
) -> Result<()> {
    if plan.is_exact() {
        return Ok(());
    }
    Err(Error::Unsupported(format!(
        "{id}: the {route} denoise route does not implement {}. Leave approximation unset for this \
         route's exact denoise.",
        plan.describe_selection()
    )))
}

/// [`denoise`] carrying the sc-18322 **denoise feature cache**.
///
/// `trunk: None` is exactly [`denoise`]. When present, the cache decides per step whether the block
/// stack runs or its retained trunk residual is reapplied — and note where that decision is *not*: the
/// cancel check, the per-step `eval` and the progress callback below are straight-line statements in
/// this loop, outside the branch, so a reused step performs all three exactly as a recomputed one
/// does. That is the whole of the cancel-responsiveness discipline; see [`crate::feature_cache`].
#[allow(clippy::too_many_arguments)]
pub fn denoise_approx(
    transformer: &WanTransformer,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_noise: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
    mut trunk: Option<&mut crate::feature_cache::TrunkCache>,
    mut prune: Option<&mut crate::token_pruning::TokenPruner>,
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    // sc-2957: run the DiT's fusable elementwise glue (adaLN affine, gated residual, gated-GELU FFN,
    // RoPE rotation) through `mx.compile` — bit-exact (proven `max|Δ|=0` real + tiny, perf.rs /
    // compile_parity.rs) and ~14% faster/step at production geometry. Scoped + restored on drop by the
    // RAII guard (F-006/F-007) instead of leaking the render thread's setting into later work.
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();

    // Precompute the RoPE + cross-K/V caches once (grid + context are constant across steps).
    let grid = transformer.patch_grid(init_noise);
    let cache = build_cache(transformer, ctx_cond, ctx_uncond, grid)?;

    let mut latents = init_noise.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        // Honor the engine cancellation contract (sc-5551, the video sibling of chroma's sc-5514):
        // a video render runs minutes, so check before each step. The per-step `eval` below makes
        // this effective — without it MLX's lazy graph defers all compute to VAE decode and this
        // check would pass for every step.
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Declare the step to the feature cache before the forward: the step index lives here, in the
        // only loop that has one, so the cache never has to derive it from a timestep (sc-18322).
        if let Some(cache) = trunk.as_deref_mut() {
            cache.begin_step(i);
        }
        // Pruning declares the step for the same reason the cache does — it keys its per-block rotation
        // on it — and its warmup decision comes from the contract's own predicate, so a warmup step
        // passes `None` and runs the untouched block loop.
        if let Some(pruner) = prune.as_deref_mut() {
            pruner.begin_step(i);
        }
        let prune_this_step = match prune.as_deref() {
            Some(pruner) if pruner.prunes_step(i) => Some(pruner),
            _ => None,
        };
        let pred = predict_approx(
            transformer,
            &latents,
            t,
            &cache,
            guidance,
            None,
            trunk.as_deref_mut(),
            prune_this_step,
        )?;
        latents = sched.step(&pred, &latents)?;
        // Force evaluation each step to bound the lazy graph's peak memory (the reference's
        // per-step `mx.eval(latents)`).
        //
        // Unconditional, and outside the feature-cache branch three call levels down: a reused step is
        // materialized and progress-reported exactly like a recomputed one, so skipping the block stack
        // can never make the loop less cancel-responsive (sc-18322).
        mlx_rs::transforms::eval([&latents])?;
        on_step(i + 1);
    }
    Ok(latents)
}

/// The dense denoise loop driven by a **curated unified solver** (epic 7114, sc-7121) — the additive
/// fold onto the shared gen-core solver library, alongside the native [`denoise`]. Routes any curated
/// [`mlx_gen::Solver`] (`euler` / `euler_ancestral` / `heun` / `dpmpp_sde` / `ddim` / …) through
/// `mlx_gen::run_flow_sampler` over Wan's own shifted flow-σ schedule ([`compute_sigmas`]).
///
/// Wan's native `unipc`/`dpmpp2m` are a diffusers `FlowDPMSolver`/`FlowUniPC` in flow-SNR space
/// (`λ = log((1−σ)/σ)`), which the gen-core VE-space solvers (`λ = −ln σ`) do NOT reproduce — so the
/// native default stays on [`denoise`] (the N1 default-parity gate), and this path is selected only for
/// the gen-core-only curated solvers. The model is velocity-prediction over the FLOW
/// [`mlx_gen::TimestepConvention::Sigma`] convention, and Wan feeds the DiT the integer-valued timestep
/// `(σ·num_train).trunc()` (the predict closure maps σ → that timestep). `seed` drives the stochastic
/// solvers' per-step noise. Progress / cancel route through `run_flow_sampler`'s per-eval hook.
#[allow(clippy::too_many_arguments)]
pub fn denoise_curated(
    transformer: &WanTransformer,
    sampler_name: &str,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_noise: &Array,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();
    let grid = transformer.patch_grid(init_noise);
    let cache = build_cache(transformer, ctx_cond, ctx_uncond, grid)?;
    let sigmas = compute_sigmas(steps, shift, num_train_timesteps);
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
            // Wan feeds the DiT the integer-valued timestep `(σ·num_train).trunc()`, not raw σ.
            let t = (sigma * nt).trunc();
            predict(transformer, x, t, &cache, guidance, None)
        },
    )
}

/// One TI2V prediction with **per-token timesteps**, reusing the precomputed [`StepCache`]: a single
/// batched forward over the per-token timestep vector `t_tokens` `[1, L]` (mask-blend, sc-2680),
/// combined as `uncond + gs·(cond − uncond)` when CFG is on, else the B=1 cond-only forward. Mirrors
/// [`predict`] but routes through [`WanTransformer::forward_tokens_cached`].
pub(super) fn predict_tokens(
    transformer: &WanTransformer,
    latents: &Array,
    t_tokens: &Array,
    cache: &StepCache,
    guidance: f32,
) -> Result<Array> {
    let preds = transformer.forward_tokens_cached(
        latents,
        t_tokens,
        &cache.cross_kv,
        &cache.cos,
        &cache.sin,
        cache.batch,
    )?;
    if cache.batch == 2 {
        cfg_combine(&preds[0], &preds[1], guidance)
    } else {
        preds
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("wan: B=1 forward produced no output".into()))
    }
}

/// The image-conditioned TI2V-5B **mask-blend** denoise loop (port of `generate_wan.py`'s
/// `is_i2v_mask_blend` path, sc-2680). The first latent temporal frame is pinned to the encoded
/// image `z_img` and *frozen*: every step (1) builds the per-token timestep vector `t_tokens =
/// mask_tokens · t` (`0` for the first-frame tokens, so they carry timestep 0), (2) predicts the
/// noise with `predict_tokens`, (3) scheduler-steps, then (4) re-blends `latents = (1−mask)·z_img +
/// mask·latents` so the first frame stays the conditioning image while the rest denoise.
///
/// `init_latents` is the pre-blended `[C,F,H,W]` start `(1−mask)·z_img + mask·noise`; `z_img` is the
/// VAE-encoded image `[C,1,H,W]` (broadcasts over `F`); `mask` is `[C,F,H,W]` (`0` first frame, `1`
/// rest); `mask_tokens` is `[1,L]` (`0` first-frame tokens, `1` rest), `L` = the patch-token count.
#[allow(clippy::too_many_arguments)]
pub fn denoise_ti2v(
    transformer: &WanTransformer,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_latents: &Array,
    z_img: &Array,
    mask: &Array,
    mask_tokens: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    // sc-2957: compile the DiT's fusable elementwise glue (bit-exact, ~14% faster/step). The per-token
    // modulation shapes differ from T2V's, so `mx.compile` simply re-traces them once. Scoped +
    // restored on drop by the RAII guard (F-006/F-007) instead of leaking into later work.
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();

    // Precompute the RoPE + cross-K/V caches once (grid + context constant across steps), exactly like
    // [`denoise`]. The per-token timesteps change each step (the only per-step DiT input besides the
    // latent), so the time embedding is recomputed inside `forward_tokens_cached`.
    let grid = transformer.patch_grid(init_latents);
    let cache = build_cache(transformer, ctx_cond, ctx_uncond, grid)?;

    // `(1−mask)·z_img` — the frozen first-frame content (z_img broadcasts over F); precomputed once.
    let one_minus_mask = subtract(scalar(1.0), mask)?;
    let frozen = multiply(&one_minus_mask, z_img)?;

    let mut latents = init_latents.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        // Honor the engine cancellation contract — check before each (minutes-long) step (sc-5551).
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Per-token timesteps: 0 for the first-frame tokens (frozen), `t` for the rest. (The 5B's
        // seq_len equals the patch count, so no padding is needed — matches the reference.)
        let t_tokens = multiply(mask_tokens, scalar(t))?;
        let pred = predict_tokens(transformer, &latents, &t_tokens, &cache, guidance)?;
        latents = sched.step(&pred, &latents)?;
        // Re-apply the mask so the first frame stays pinned to the conditioning image.
        latents = add(&frozen, &multiply(mask, &latents)?)?;
        mlx_rs::transforms::eval([&latents])?;
        on_step(i + 1);
    }
    Ok(latents)
}
