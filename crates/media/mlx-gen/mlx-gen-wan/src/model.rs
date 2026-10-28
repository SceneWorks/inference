//! `mlx-gen-wan` model entries: the Wan2.2 **TI2V-5B** (`wan2_2_ti2v_5b`, dense, z48 VAE — fully
//! wired, sc-2680), the Wan2.2 **T2V-A14B** (`wan2_2_t2v_14b`, dual-expert MoE, z16 VAE — fully wired
//! here on the S1–S5 core), and the Wan2.2 **I2V-A14B** (`wan2_2_i2v_14b`, dual-expert MoE,
//! channel-concat image conditioning, in_dim 36 — sc-2681), plus their registry self-registration.
//!
//! The 5B [`Wan`] struct runs the complete dense pipeline (sc-2680) — [`Wan::generate`]: UMT5-XXL
//! encode → the dense [`crate::pipeline::denoise`] (T2V) or the [`denoise_ti2v`] per-token mask-blend (TI2V, single- or
//! multi-keyframe) → z48 VAE decode → RGB8 frames, with Q4/Q8 + LoRA. The shared [`Wan14b`] struct
//! serves both A14B variants — [`Wan14b::generate`] runs the
//! complete pipeline: UMT5-XXL encode → (I2V only) build the channel-concat conditioning `y` →
//! per-step dual-expert MoE denoise (boundary-switched high/low experts, [`denoise_moe`]) → z16 VAE
//! decode → RGB8 frames, **staging** each heavy component (T5, the two 27 GB experts, the VAE) in and
//! out to bound peak memory (mirrors `generate_wan.py`). The I2V variant differs only by the `y`
//! conditioning (the image's first-frame VAE latent + temporal mask, channel-concatenated to in_dim
//! 36) and the max-area resolution cap.

use std::path::PathBuf;

use mlx_gen::gen_core::{
    adapter_stack_resident_bytes, AdapterResidencyMode, ApproximationPlan, ApproximationSurface,
};
use mlx_gen::tiling::VaeTiling;
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterSpec, CancelFlag, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadPhase, LoadSpec, Modality, ModelDescriptor, MoeExpert,
    OffloadPolicy, Precision, Progress, Quant, Result, WeightsSource,
};
use mlx_rs::random;
use mlx_rs::Array;

use crate::adapters::{
    apply_wan_adapters_additive, merge_wan_adapters, reject_loha_on_packed, warn_skipped_adapters,
    WanLoraReport,
};
use crate::config::{WanModelConfig, WanQuant, MIN_SIZE};
use crate::feature_cache::TrunkCache;
use crate::pipeline::{
    align_dim, auto_tiling_budgeted_z16, build_i2v_y, build_ti2v_keyframe_z, build_ti2v_mask,
    crossing_index, decode_to_frames, decode_to_frames_22, denoise_approx, denoise_curated,
    denoise_moe, denoise_moe_curated, denoise_moe_curated_swapped, denoise_range, denoise_ti2v,
    frames_to_images, latent_shape, preflight_denoise_memory_guard, preprocess_ti2v_image,
    refuse_unwired_approximation, reject_off_grid, reject_over_area, resolve_sampler_knobs,
    seq_len, staged_expert_swap, ti2v_blend_init, Expert,
};
use crate::scheduler::{make_scheduler, SolverKind, WanScheduler};
use crate::text_encoder::encode_text_staged_for_tier;
use crate::token_pruning::TokenPruner;
use crate::transformer::WanTransformer;

/// Concrete z48 VAE assigned to the TI2V-5B route.
pub type Ti2vProviderVae = crate::vae22::Wan22Vae;
/// Concrete z16 VAE assigned to both A14B routes.
pub type A14bProviderVae = crate::vae::WanVae;

const fn provider_vae_stride(vae: VaeTiling) -> (usize, usize, usize) {
    (
        vae.temporal_scale as usize,
        vae.spatial_scale as usize,
        vae.spatial_scale as usize,
    )
}

/// Resolve the TI2V-5B route's load-bearing VAE geometry.
pub fn ti2v_vae_tiling(provider_id: &str) -> Option<mlx_gen::tiling::VaeTiling> {
    (provider_id == MODEL_ID).then_some(Ti2vProviderVae::VAE_TILING)
}

/// Resolve either A14B route's load-bearing VAE geometry.
pub fn a14b_vae_tiling(provider_id: &str) -> Option<mlx_gen::tiling::VaeTiling> {
    matches!(provider_id, MODEL_ID_T2V_14B | MODEL_ID_I2V_14B)
        .then_some(A14bProviderVae::VAE_TILING)
}

/// The curated unified solvers (epic 7114, sc-7121) every Wan generator exposes ADDITIVELY beyond its
/// native solvers — the gen-core-only solvers, routed through `run_flow_sampler` over Wan's own flow-σ
/// schedule ([`crate::pipeline::denoise_curated`] / [`denoise_moe_curated`]).
///
/// A14B residency note (sc-12795): under `Sequential`, these route the expert **per eval** inside the
/// solver callback. This preserves boundary-straddling sub-evaluations while making one high→low
/// evict-then-load swap, so only one expert is resident.
const WAN_CURATED_SAMPLERS: [&str; 4] = ["euler_ancestral", "heun", "dpmpp_sde", "ddim"];

/// Wan's native flow-SNR solvers ([`crate::scheduler::SolverKind`]), advertised under the curated
/// gen-core vocabulary (epic 7114 sc-7296). `uni_pc`/`dpmpp_2m`/`euler` route to the NATIVE flow-SNR
/// solver — NOT the gen-core VE-space `uni_pc`/`dpmpp_2m` (`λ = −ln σ`), which would not reproduce Wan's
/// diffusers FLOW-SNR (`λ = log((1−σ)/σ)`) parity. Sampler names are family labels in this framework
/// (cf. `euler` across prediction types), so Wan's native UniPC IS the `uni_pc` for Wan — the native
/// math is unchanged (the byte-exact N1 default).
const WAN_NATIVE_SAMPLERS: [&str; 3] = ["uni_pc", "euler", "dpmpp_2m"];

/// Legacy sampler spellings (pre-sc-7296) kept advertised so old recipes still validate + reproduce;
/// they map to the same native solvers via [`crate::scheduler::SolverKind::from_name`]. The SceneWorks
/// manifest surfaces only the curated names.
const WAN_LEGACY_SAMPLERS: [&str; 2] = ["unipc", "dpmpp2m"];

/// The dense-5B decode plan. A benchmark scope's fixed-tile control still wins outright — it exists to
/// hold tiles constant across a measurement campaign — and otherwise this defers to the carrier-aware
/// planner, which honors a worker-selected bounded-decode rung and falls back to
/// `auto_tiling_budgeted` when the request carries no memory strategy.
fn dense_decode_tiling(
    request: &GenerationRequest,
    width: u32,
    height: u32,
    out_frames: u32,
) -> Result<Option<mlx_gen::TilingConfig>> {
    match mlx_gen::diagnostics::benchmark_decode_control() {
        Some(control) => Ok(Some(control.tiling_config())),
        None if request.video_mode.as_deref() == Some("image_to_video")
            && request.memory.is_some() =>
        {
            crate::i2v_memory_strategy::decode_tiling(request, width, height, out_frames)
        }
        None => crate::memory_strategy::decode_tiling(request, width, height, out_frames),
    }
}

/// Wan's full per-generation sampler menu: native solvers (curated vocabulary) + the curated gen-core
/// fold-ins + the legacy aliases.
fn wan_samplers() -> Vec<&'static str> {
    let mut s = WAN_NATIVE_SAMPLERS.to_vec();
    s.extend(WAN_CURATED_SAMPLERS);
    s.extend(WAN_LEGACY_SAMPLERS);
    s
}

/// The native-only menu (curated vocabulary + legacy aliases, NO gen-core fold-ins) — for the VACE
/// path, which advertises its native solvers without the `run_flow_sampler` fold-ins.
pub(crate) fn wan_native_samplers() -> Vec<&'static str> {
    let mut s = WAN_NATIVE_SAMPLERS.to_vec();
    s.extend(WAN_LEGACY_SAMPLERS);
    s
}

/// Whether `req.sampler` selects a curated gen-core solver (routed through `run_flow_sampler`) rather
/// than a native Wan solver (handled by `scheduler.rs`). Used to branch the denoise dispatch.
pub(crate) fn is_wan_curated(name: Option<&str>) -> bool {
    matches!(name, Some(n) if WAN_CURATED_SAMPLERS.contains(&n))
}

/// Public provider id: `"wan2_2_ti2v_5b"`.
pub const MODEL_ID: &str = "wan2_2_ti2v_5b";

/// Stable identity + advertised capabilities for the Wan2.2 TI2V-5B (dense text+image→video).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::WAN_Z48_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "wan",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // 5B uses real CFG (guide 5.0) with the Chinese anti-artifact negative prompt, and
            // accepts a single image as the TI2V mask-blend conditioning reference. Keyframe =
            // Wan-native first_last_frame / multi-keyframe (epic 3040, sc-3357) via the same
            // mask-blend, pinning the listed latent frames instead of only frame 0.
            supports_negative_prompt: true,
            supports_guidance: true,
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::Keyframe],
            // Q4/Q8 (sc-2682) loads via `spec.quantize` (transformer-only); LoRA/LoKr merge onto the
            // single dense model at generate time (the reference `_loras_single` path — shared
            // untagged specs only, reusing the sc-2683/sc-2393 `merge_wan_adapters` seam).
            supports_lora: true,
            supports_lokr: true,
            samplers: wan_samplers(),
            // H/W align to patch×vae_stride = 32 (`reject_off_grid`); floor each side at MIN_SIZE = 480
            // (= 15·32 — the z48 vae22 renders garbage below a 15×15 latent grid, sc-10306/sc-12636),
            // matching candle; cap the long edge at 1280 (max_area 704×1280).
            supported_guidance_methods: vec![],
            min_size: MIN_SIZE,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: crate::memory_strategy::COMPONENT_PRECISION_FLOORS,
            // Cross-attention text K/V is cached across denoise steps.
            supports_kv_cache: true,
            // Wan pins a static `sample_shift` from config (not the empirical per-resolution mu).
            requires_sigma_shift: false,
            // Honors `OffloadPolicy::Sequential` (epic 12732, sc-12796): the dense path is already
            // staged (TE → DiT → z48 VAE, each loaded → used → dropped in turn), so Sequential adds no
            // expert swap (there are none) — it `clear_cache`-flushes the dead UMT5 TE / VAE-encoder
            // buffers off-GPU after each stage, so they don't linger in MLX's buffer cache (RSS /
            // wired-memory pressure) through denoise + decode. Advertised so the worker's fit-gate can
            // tell "bounds footprint here" from a no-op fallback.
            supports_sequential_offload: true,
            // TE → DiT → z48 VAE is phase-staged on every request; Sequential additionally flushes
            // dead allocator cache between those already-staged phases.
            unconditionally_engages_staged_residency: true,
            // The sc-18322 denoise feature cache, declared on the dense 5B alone: it is the one Wan
            // provider with a single transformer for the whole trajectory (no MoE expert swap to
            // invalidate a retained residual across) and a native one-forward-per-step loop with a step
            // index. Implemented in `crate::feature_cache`, wired into `pipeline::denoise_approx`.
            //
            // Declared is NOT selectable. No characterization artifact family exists yet, so the shared
            // floor refuses every selection against this surface — see `gen_core::approximation`. The
            // declaration is what makes the mechanism discoverable and what the terminal measurement
            // campaign narrows from these candidate intervals to the ones it can vouch for.
            approximation: approximation_surface(),
            ..Default::default()
        },
    }
}

/// The dense 5B's declared approximate-capability surface (sc-18322).
///
/// The candidate intervals are the mechanism's implemented operating points, not measured ones — the
/// domain shape carries no measurement claim, and the characterization binding (absent, and
/// unconstructible) is what refuses selection until one exists. Warmup is capped at 8 steps because the
/// native Wan trajectories run 20-50 steps and a warmup past a third of them leaves nothing to reuse.
pub(crate) fn approximation_surface() -> ApproximationSurface {
    ApproximationSurface::feature_cache(vec![2, 3, 4], 8).with_token_pruning(vec![2, 3, 4], 8)
}

/// The denoise feature cache a resolved plan asks for — the provider-side bridge from contract to
/// mechanism (sc-18322).
///
/// Deliberately two definitions rather than one with an inner `cfg`, so the production answer is
/// visibly and unconditionally `None`: [`TrunkCache`]'s only constructor is `#[cfg(test)]`, so a
/// production build cannot reach the uncharacterized path even if it somehow held a
/// [`ApproximationPlan::Approximate`]. That is the second of two independent locks — the first is the
/// contract refusing every approximate selection for want of a characterization artifact.
/// Both bridges first run [`refuse_inert_over_trajectory`], because the step count is only known here.
#[cfg(not(test))]
fn trunk_cache(id: &str, plan: &ApproximationPlan, steps: usize) -> Result<Option<TrunkCache>> {
    refuse_inert_over_trajectory(id, plan, steps)?;
    Ok(None)
}

#[cfg(test)]
fn trunk_cache(id: &str, plan: &ApproximationPlan, steps: usize) -> Result<Option<TrunkCache>> {
    refuse_inert_over_trajectory(id, plan, steps)?;
    Ok(TrunkCache::from_plan_for_test(plan))
}

/// The token pruner a resolved plan asks for — the second provider-side bridge, locked exactly like
/// [`trunk_cache`] (sc-18322). `total` is the generate's patchified token count, which the rotation
/// phases are built over.
#[cfg(not(test))]
fn token_pruner(
    id: &str,
    plan: &ApproximationPlan,
    steps: usize,
    _total: usize,
) -> Result<Option<TokenPruner>> {
    refuse_inert_over_trajectory(id, plan, steps)?;
    Ok(None)
}

#[cfg(test)]
fn token_pruner(
    id: &str,
    plan: &ApproximationPlan,
    steps: usize,
    total: usize,
) -> Result<Option<TokenPruner>> {
    refuse_inert_over_trajectory(id, plan, steps)?;
    TokenPruner::from_plan_for_test(plan, total)
}

/// Refuse an approximate policy that would engage on **no step** of this request's trajectory
/// (sc-18322).
///
/// The last way to encode *off* inside an `Approximate` plan. A warmup is gated against the declared
/// `max_warmup_steps` without knowing the step count — a request may leave `steps` to the model's default
/// — so a policy well inside its declared domain can still cover the whole trajectory and produce a
/// result bit-identical to the exact path. That is exactly the second-encoding-of-off defect that
/// rejecting a reuse interval of `1` exists to prevent, and letting it through here would make the
/// contract's own "an approximate plan changed the output" claim false.
///
/// This is the one place the step count and the policy are both in hand, so it is the one place the
/// refusal can be made. Unreachable today (no plan reaching a provider can be anything but `Exact`), and
/// it fails closed the day selection becomes possible.
pub(crate) fn refuse_inert_over_trajectory(
    id: &str,
    plan: &ApproximationPlan,
    steps: usize,
) -> Result<()> {
    if let Some(policy) = plan.feature_cache() {
        if policy.is_inert_over(steps) {
            return Err(Error::Unsupported(format!(
                "{id}: the selected denoise feature cache (interval {} steps, warmup {}) reuses \
                 nothing over this request's {steps} steps, so it would produce the exact result \
                 through the approximate path. Lower the warmup, raise the step count, or leave \
                 approximation unset.",
                policy.reuse_interval.steps(),
                policy.warmup.steps()
            )));
        }
    }
    if let Some(policy) = plan.token_pruning() {
        if policy.is_inert_over(steps) {
            return Err(Error::Unsupported(format!(
                "{id}: the selected token pruning (drop stride {}, warmup {}) prunes nothing over \
                 this request's {steps} steps, so it would produce the exact result through the \
                 approximate path. Lower the warmup, raise the step count, or leave approximation \
                 unset.",
                policy.drop_stride.stride(),
                policy.warmup.steps()
            )));
        }
    }
    Ok(())
}

/// The projection width the UMT5 text encoder packs to on a quantized tier: **Q8** (sc-12831). Q8 is
/// near-lossless for this drift-sensitive encoder — measured prompt-embedding cosine **0.9998** vs the
/// bf16 baseline on the real 5B TE, vs **0.976** at Q4 (a visible ~12° shift; the "biggest numeric-drift
/// risk" the candle sc-12757 finding flagged). Q8 already cuts the TE-encode active peak **11.83 → 7.72
/// GiB** — well under the epic's <16 GB-with-margin target — so we floor the TE at Q8 even when the DiT
/// tier is Q4: the user's Q4 *DiT* creative choice is untouched, and the TE stays regression-free (the
/// extra ~2 GiB a Q4 TE would save is not needed and not worth the drift).
pub(crate) const TE_QUANT_BITS: i32 = 8;

/// The effective UMT5 text-encoder quantization for a Wan tier (sc-12831). The DiT is packed on an
/// MLX-affine tier iff a pre-quantized snapshot manifest is present (`config.quantization`) **or** a
/// load-time `Q4`/`Q8` `spec.quantize` was requested (`Nvfp4` is candle-only and never reaches the MLX
/// Wan path — excluded so its `bits()` = 4 is not routed through `mlx quantize`). On such a tier the TE
/// packs to `TE_QUANT_BITS` (Q8), retiring the residual ~12 GiB f32-TE-encode active peak (sc-12796)
/// that no further component offload could lower — the epic's binding 5B constraint. On the bf16 tier
/// this is `None` (the TE stays dense / bit-exact). Shared by the 5B [`Wan`], the A14B [`Wan14b`], and
/// the VACE paths (`model_vace.rs`) — and, since sc-15203, by `mlx-gen-krea-realtime`, whose Krea
/// Realtime 14B tiers reuse this same stock-Wan UMT5-XXL encoder and therefore this same Q8 floor.
/// `pub` for that cross-crate reuse: the floor's rationale (and `TE_QUANT_BITS`) must live in exactly
/// one place rather than being re-derived per Wan-family provider.
pub fn effective_te_quant(config: &WanModelConfig, load_quant: Option<Quant>) -> Option<WanQuant> {
    let dit_affine_quantized =
        config.quantization.is_some() || matches!(load_quant, Some(Quant::Q4) | Some(Quant::Q8));
    dit_affine_quantized.then_some(WanQuant {
        bits: TE_QUANT_BITS,
        group_size: mlx_gen::quant::DEFAULT_GROUP_SIZE,
    })
}

/// The loaded Wan2.2 TI2V-5B (dense). Holds the resolved config + the snapshot directory; the heavy
/// components (UMT5 TE, the single 5B DiT, the z48 vae22) are **staged** inside [`Wan::generate`] —
/// loaded, used, then dropped in turn — to bound peak memory (mirrors `generate_wan.py`, which never
/// holds the T5 encoder + the 10 GB transformer resident at once).
pub struct Wan {
    descriptor: ModelDescriptor,
    config: WanModelConfig,
    root: PathBuf,
    /// LoRA/LoKr adapters merged onto the single dense model at generate time (the reference
    /// `_loras_single` path). Empty for a plain load. `moe_expert`-tagged specs are rejected (dense).
    adapters: Vec<AdapterSpec>,
    /// Optional Q4/Q8 quantization for the transformer (sc-2682). `None` = dense bf16 (or a
    /// pre-quantized snapshot, which `from_weights` builds packed from its `config.json` manifest).
    quant: Option<Quant>,
    /// Component-residency strategy for the dense render (epic 12732, sc-12796). [`OffloadPolicy::Resident`]
    /// (default) is the byte-identical pre-offload path — the staged components are dropped by scope but
    /// their freed buffers stay warm in MLX's cache. [`OffloadPolicy::Sequential`] additionally
    /// `clear_cache`-flushes each dead component (UMT5 TE, TI2V VAE-encoder, DiT) out to the OS as soon
    /// as it is dropped, so it is truly off-GPU (not just off the active set) for the stages that
    /// follow — bounding the unified-memory RSS / wired footprint. No expert swap (single dense DiT).
    /// Captured from [`LoadSpec::offload_policy`] at load.
    offload_policy: OffloadPolicy,
    /// Load-exact memory contract and numeric tier for the one calibrated plain Resident/Eager
    /// route. Other supported Wan loads remain available but deliberately publish no contract.
    memory_strategy: Option<mlx_gen::gen_core::MemoryProviderContract>,
    memory_tier: Option<mlx_gen::gen_core::MemoryNumericTier>,
    /// Exact provider-owned public-I2V artifact and adapter receipt, when the caller prepared one.
    i2v_memory: Option<crate::i2v_memory_strategy::PreparedWanI2vMemory>,
}

impl Wan {
    /// The resolved model config (exposed for tests).
    pub fn config(&self) -> &WanModelConfig {
        &self.config
    }

    /// Reject a `moe_expert`-tagged spec on the dense 5B (a misconfiguration — high/low tagging is
    /// only for the dual-expert A14B). The dense 5B takes only **shared** (untagged) specs — the
    /// reference's `_loras_single` (`--lora`, not `--lora-high/low`).
    fn check_dense_specs(&self) -> Result<()> {
        if self.adapters.iter().any(|s| s.moe_expert.is_some()) {
            return Err(Error::Msg(format!(
                "{}: `moe_expert` (high/low) tagging is only for the dual-expert A14B — the dense \
                 5B takes shared (untagged) adapters",
                self.descriptor.id
            )));
        }
        Ok(())
    }

    /// Enforce the "matched no module" error + surface partial skips after an adapter apply (fold OR
    /// additive) — shared by both paths so the message can't drift. A non-empty adapter set that
    /// applied nothing is a format/prefix misconfiguration; per-key skips are surfaced, not fatal.
    fn finalize_report(&self, report: &WanLoraReport) -> Result<()> {
        if report.applied == 0 {
            return Err(Error::Msg(format!(
                "{}: {} adapter file(s) matched no module — check the format (PEFT `lora_A/B` or \
                 kohya `lora_down/up`, `diffusion_model.`-prefixed Wan module names)",
                self.descriptor.id,
                self.adapters.len()
            )));
        }
        warn_skipped_adapters(self.descriptor.id, &report.skipped);
        Ok(())
    }

    /// **Dense-bf16 path.** Fold the load-time LoRA/LoKr adapters into the single dense model weight
    /// map in place, before the [`WanTransformer`] is built (the reference fold order: LoRA folds into
    /// the dense weight, then `spec.quantize` may quantize it). No-op without adapters. Reuses the
    /// sc-2683/sc-2393 [`merge_wan_adapters`] seam (`MoeExpert::High` ⇒ only the `moe_expert == None`
    /// pass fires, since all specs are untagged). Called only for a dense snapshot; a pre-quantized
    /// snapshot uses [`Wan::install_adapters_additive`] instead (sc-10045).
    fn merge_adapters(&self, w: &mut Weights) -> Result<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        self.check_dense_specs()?;
        let report = merge_wan_adapters(w, &self.adapters, MoeExpert::High)?;
        self.finalize_report(&report)
    }

    /// **Pre-quantized (packed Q4/Q8) path (sc-10045 / sc-10050).** Install the load-time adapters onto
    /// the already-built (packed) [`WanTransformer`] as forward-time residuals — the base stays packed,
    /// never dequantized. No-op without adapters. Plain LoRA and **LoKr** both apply here (LoKr via the
    /// structured deferred-Kronecker vec-trick, sc-10050 — no full delta materialized); only LoHa on a
    /// packed tier is rejected up front with an actionable error (deferred to sc-10051). Called only for
    /// a pre-quantized snapshot; a dense snapshot uses [`Wan::merge_adapters`] instead.
    fn install_adapters_additive(&self, dit: &mut WanTransformer) -> Result<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        self.check_dense_specs()?;
        reject_loha_on_packed(self.descriptor.id, &self.adapters)?;
        let report = apply_wan_adapters_additive(dit, &self.adapters, MoeExpert::High)?;
        self.finalize_report(&report)
    }
}

/// Load the Wan2.2 TI2V-5B from a converted MLX snapshot directory (`convert_wan.py` output:
/// `model.safetensors` + `t5_encoder.safetensors` + `vae.safetensors` + `tokenizer.json` +
/// `config.json`). The DiT runs bf16 GEMMs over an f32 residual (the S3 parity regime). Q4/Q8
/// (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot; LoRA/LoKr (sc-2683 / sc-2393)
/// merge onto the single dense model at generate time.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_ti2v_5b: expected a model directory (converted MLX snapshot), not a single file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_ti2v_5b: precision override is not wired (the DiT runs bf16 GEMMs over an f32 \
             residual stream — the parity regime)"
                .into(),
        ));
    }
    let config = WanModelConfig::from_model_dir(&root)?;
    if config.dual_model || !config.is_ti2v() {
        return Err(Error::Msg(format!(
            "wan2_2_ti2v_5b: config.json is not the dense TI2V-5B (model_type={}, dual_model={}); \
             expected the converted Wan2.2 TI2V-5B checkpoint (model_type=ti2v, dual_model=false)",
            config.model_type, config.dual_model
        )));
    }
    let quant = resolve_load_time_quant(MODEL_ID, &config, spec.quantize)?;
    let memory = crate::memory_strategy::contract_for_loaded(spec).map_err(Error::from)?;
    let (memory_strategy, memory_tier) = memory
        .map(|(contract, tier)| (Some(contract), Some(tier)))
        .unwrap_or((None, None));
    let i2v_memory = if spec.resolved_route.as_deref() == Some(MODEL_ID)
        && spec.prepared_file_pins().is_prepared()
    {
        Some(crate::i2v_memory_strategy::prepare(spec, MODEL_ID).map_err(Error::from)?)
    } else {
        None
    };
    Ok(Box::new(Wan {
        descriptor: descriptor(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
        offload_policy: spec.offload_policy,
        memory_strategy,
        memory_tier,
        i2v_memory,
    }))
}

// Hand-written so the load-exact SC-19236 contract is a property of the loaded provider. The
// descriptor/validation/generation arms are otherwise the same delegation emitted by
// `impl_generator!` for the other Wan routes.
impl Generator for Wan {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        // SC-20686 Metal lane: a direct `generate_impl` call unless a campaign is armed.
        mlx_gen::sc20686::observe_generation(
            &self.root,
            &req.cancel,
            self.descriptor.id,
            || mlx_gen::sc20686::RequestFacts::from_request(req),
            on_progress,
            |on_progress| self.generate_impl(req, on_progress),
        )
        .map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        self.i2v_memory
            .as_ref()
            .map(|prepared| &prepared.contract)
            .or(self.memory_strategy.as_ref())
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        if let Some(prepared) = &self.i2v_memory {
            return crate::i2v_memory_strategy::safety_check(prepared, context);
        }
        let (Some(contract), Some(tier)) = (self.memory_strategy.as_ref(), self.memory_tier) else {
            return if context.selection.strategy == mlx_gen::gen_core::MemoryStrategy::Resident {
                mlx_gen::gen_core::MemorySafetyDecision::Accept
            } else {
                mlx_gen::gen_core::MemorySafetyDecision::Reject {
                    reason: format!(
                        "{MODEL_ID} loaded route has no calibrated memory-strategy contract"
                    ),
                }
            };
        };
        crate::memory_strategy::safety_check(contract, tier, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        if let Some(prepared) = &self.i2v_memory {
            return crate::i2v_memory_strategy::begin_request(prepared, context);
        }
        let (Some(contract), Some(tier)) = (self.memory_strategy.as_ref(), self.memory_tier) else {
            return Ok(None);
        };
        crate::memory_strategy::begin_request(contract, tier, context)
    }
}

impl Wan {
    /// Validate body — kept on the crate's own [`mlx_gen::Error`] so `?` on the capability check
    /// lifts transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        // Shared capability floor: size range (the advertised `min_size` = MIN_SIZE = 480 is the
        // 5B coherence floor below which the z48 vae22 renders garbage, sc-10306/sc-12636 — NOT the
        // 32-px grid stride, which `reject_off_grid` enforces below; `max_size` caps the long edge),
        // count, guidance/negative/true_cfg, sampler (`unipc`/`euler`/`dpmpp2m`), scheduler, and
        // conditioning (`Reference`/`Keyframe`).
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if let Some(frames) = req.frames {
            // num_frames must be 1 + 4·k (one VAE temporal chunk + 4× per chunk).
            if frames % 4 != 1 {
                return Err(Error::Msg(format!(
                    "wan2_2_ti2v_5b: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
            if frames as usize > crate::MAX_WAN_FRAMES {
                return Err(Error::Msg(format!(
                    "{MODEL_ID}: num_frames {frames} exceeds the maximum {}",
                    crate::MAX_WAN_FRAMES
                )));
            }
        }
        // Reject an off-grid or over-area geometry rather than silently align-down refitting it — one
        // request means one geometry on both backends. sc-12607: the 32-px grid stride (candle rejects
        // via `is_multiple_of(SIZE_MULTIPLE)`). sc-12308: the 5B's OWN 901 120 area budget — its 32-px
        // grid makes 1280×704 the geometry it genuinely renders.
        let (dw, dh) = grid(&self.config, Ti2vProviderVae::VAE_TILING);
        reject_off_grid(MODEL_ID, req, dw, dh)?;
        reject_over_area(MODEL_ID, req, dw, dh, self.config.max_area)?;
        // sc-19571: the conditioning strengths now reach the mask builder, so their range is load
        // bearing — reject out-of-range/non-finite here rather than letting `1 − strength` become a
        // negative mask deep in the denoise.
        reject_out_of_range_strengths(MODEL_ID, req)?;
        // The TI2V mask-blend path (the `ti2v = Some(_)` branch of `generate_impl`) is entered by
        // Keyframe conditioning OR a Reference image — the contract checks below must cover both.
        let image_conditioned = !req.keyframes().is_empty() || i2v_reference(req).is_some();
        // F-074(b): a `Reference` image AND `Keyframe`s together previously fell into the keyframe
        // branch, silently dropping the Reference (undocumented precedence). The combination is
        // redundant — a Reference *is* a first-frame pin — so reject it; the first frame is
        // expressed as a Keyframe with frame_idx 0 alongside the others.
        if i2v_reference(req).is_some() && !req.keyframes().is_empty() {
            return Err(Error::Msg(
                "wan2_2_ti2v_5b: Reference and Keyframe conditioning cannot be combined (the \
                 keyframe path would silently drop the Reference) — express the first frame as a \
                 Keyframe with frame_idx 0"
                    .into(),
            ));
        }
        // F-074(a): the curated-sampler × image-conditioned-TI2V mask-blend rejection previously
        // fired in Stage 2 (denoise), AFTER the ~11 GB UMT5 load + VAE encode. Both inputs are known
        // at validate time, so reject here — the mask-blend path has no single-eval curated-sampler
        // hook. (The 14B sibling has no mask-blend path: its I2V is channel-concat, which the
        // curated solvers DO serve via `denoise_moe_curated`.)
        if is_wan_curated(req.sampler.as_deref()) && image_conditioned {
            return Err(Error::Msg(
                "wan2_2_ti2v_5b: curated samplers (euler_ancestral/heun/dpmpp_sde/ddim) are not \
                 supported with image-conditioned TI2V mask-blend — use unipc/euler/dpmpp2m"
                    .into(),
            ));
        }
        // F-015: `trim_first_frames` generates extra leading latent frames and drops them after
        // decode — but the mask-blend pins the Reference at latent frame 0 (into the discarded
        // prefix) and Keyframe indices resolve against the trim-EXTENDED grid (shifted vs the
        // delivered video). The reference (mlx_video generate_wan.py) documents trim as a 14B
        // first-frame-artifact fix and, run this way, trims its own pinned frame off — a degenerate
        // outcome, not a supported mode. Reject, mirroring the 14B's trim×I2V rejection.
        if req.trim_first_frames.unwrap_or(0) > 0 && image_conditioned {
            return Err(Error::Msg(
                "wan2_2_ti2v_5b: trim_first_frames is not supported with Reference/Keyframe \
                 conditioning (the pinned frame would be trimmed off, and keyframe positions \
                 would shift vs the delivered video)"
                    .into(),
            ));
        }
        Ok(())
    }

    /// The dense 5B pipeline (port of `generate_wan.py`'s single-model path, sc-2680) — **T2V** when
    /// no image is given, **TI2V** mask-blend when a `Reference` image is. Resolves request knobs,
    /// then **stages** the phases to bound memory: (1) UMT5 encode the prompt (+ neg, unless CFG is
    /// off); (1b, TI2V) load the z48 vae22, encode the conditioning image → `z_img`, build the
    /// first-frame mask + per-token mask, blend the noise init; (2) load the 5B DiT (merge adapters,
    /// quantize), embed the contexts, run the dense [`crate::pipeline::denoise`] (T2V) or [`denoise_ti2v`] mask-blend
    /// loop; (3) load the vae22 decoder → RGB8 frames. CFG runs with the single guidance scale.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // Reject anything outside the advertised surface before doing expensive work — in particular
        // an unknown `sampler`, which `solver_kind` would otherwise silently map to UniPC.
        self.validate(req)?;
        if let Some(prepared) = &self.i2v_memory {
            crate::i2v_memory_strategy::validate_active_request(prepared, req)
                .map_err(Error::from)?;
        }
        // The resolved approximate-capability plan (sc-18322). `validate` already ran the same
        // resolution at the shared floor, so this cannot fail here; resolving again is how the route
        // dispatch below gets a plan without re-reading the request and re-deriving a policy.
        //
        // Always `Exact` today — the contract refuses every approximate selection until a
        // quality-characterization artifact family exists — but the three unwired routes refuse a
        // non-exact plan **by name** rather than by omission, so the day selection becomes possible they
        // fail closed instead of silently running the exact denoise.
        let approximation = self
            .descriptor
            .capabilities
            .approximation_plan(self.descriptor.id, req)?;
        let cfg = &self.config;
        // Sequential offload (epic 12732, sc-12796): the dense render is already staged (TE → DiT → z48
        // VAE, each loaded → used → dropped in turn), so there is no expert swap. Under `Sequential`,
        // `clear_cache`-flush each dead component's buffers off-GPU as soon as it is dropped, so the
        // freed UMT5 TE / VAE-encoder don't linger in MLX's buffer cache (RSS / wired footprint) through
        // the denoise + decode that follow. `Resident` (default) leaves the cache warm — the
        // byte-identical pre-offload path (residency/lifetime change only, numerics untouched).
        let sequential = self.offload_policy == OffloadPolicy::Sequential
            || crate::i2v_memory_strategy::staged(req);

        // --- Resolve request knobs against config defaults ---
        let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
        let trim = req.trim_first_frames.unwrap_or(0) as usize;
        let vae_stride = provider_vae_stride(Ti2vProviderVae::VAE_TILING);
        let trim_out = trim * vae_stride.0; // discarded output frames = trim · 4
        let gen_frames = frames + trim_out;
        let (width, height) = resolve_capped_dims(req, cfg, Ti2vProviderVae::VAE_TILING);
        let (steps, shift, kind, seed) =
            resolve_sampler_knobs(req, cfg.sample_steps, cfg.sample_shift);
        // The 5B is dense → a single guidance scale (config Single(5.0), overridable per request).
        let guidance = cfg.sample_guide_scale.resolve_single(req.guidance);
        let cfg_disabled = guidance <= 1.0;
        let neg_prompt = req
            .negative_prompt
            .clone()
            .unwrap_or_else(|| cfg.sample_neg_prompt.clone());

        let lat = latent_shape(gen_frames, height, width, cfg.vae_z_dim, vae_stride)?;

        // sc-4986 — fail fast (catchable) if the DiT-denoise stage won't fit, before any heavy load.
        // sc-12796: this is policy-INdependent for the dense 5B (unlike the A14B's `moe_denoise_resident_bytes`,
        // whose resident-expert count is 1 under the swap vs 2 otherwise). The 5B holds exactly ONE DiT
        // resident during denoise under both `Resident` and `Sequential` — the TE/VAE are staged out
        // either way; `Sequential` only additionally `clear_cache`-flushes their dead buffers — so the
        // single-DiT byte count is already the correct budget for both policies.
        let adapter_bytes = adapter_stack_resident_bytes(
            &self.adapters,
            if self.config.quantization.is_some() {
                AdapterResidencyMode::Additive
            } else {
                AdapterResidencyMode::Folded
            },
        )
        .ok_or_else(|| {
            Error::Msg(format!(
                "{}: cannot size every additive adapter before the denoise fit gate",
                self.descriptor.id
            ))
        })?;
        preflight_denoise_memory_guard(
            self.descriptor.id,
            dit_resident_bytes(&[self.root.join("model.safetensors")], self.quant)
                .saturating_add(adapter_bytes),
            seq_len(lat, cfg.patch_size),
            cfg.dim,
            !cfg_disabled,
        )?;

        // sc-4998 — pick the memory-budgeted z48 vae22 decode tiling now, so an over-budget decode
        // (the 60 GB / 4.3 min blowup the px-threshold `auto` produced at moderate res) fails
        // catchably *before* the heavy denoise rather than OOM-ing in the post-loop decode stage.
        // sc-5039 — the decode runs **bf16** (visually lossless, cosine 0.999954 real-weight; lower
        // peak ⇒ the budget fits bigger tiles), so the plan uses the bf16 cost coefficient.
        let out_frames = 1 + (lat[1] - 1) * Ti2vProviderVae::VAE_TILING.temporal_scale;
        let out_height = lat[2] * Ti2vProviderVae::VAE_TILING.spatial_scale;
        let out_width = lat[3] * Ti2vProviderVae::VAE_TILING.spatial_scale;
        let decode_tiling =
            dense_decode_tiling(req, out_width as u32, out_height as u32, out_frames as u32)?;

        // --- Stage 1: UMT5 text encode (loaded → used → freed) ---
        let (context, context_null) = encode_text_staged_for_tier(
            &self.root,
            cfg,
            &req.prompt,
            &neg_prompt,
            cfg_disabled,
            self.quant,
        )?;
        // sc-12796: `encode_text_staged` already drops the UMT5 encoder before returning (only the small
        // `[L, dim]` contexts survive — the DiT projects them through its own `text_embedding` below).
        // Under `Sequential`, flush its ~11 GB bf16 weights (f32 activations — text_encoder.rs) out of
        // MLX's buffer cache to the OS now, so the TE is truly off-GPU for the whole denoise + decode.
        // `Resident` leaves the cache warm (byte-identical pre-offload behavior). `context`/`context_null`
        // are already eval'd.
        if sequential {
            mlx_rs::memory::clear_cache();
        }

        // Seeded init noise (f32) — shape matches the reference; exact RNG values differ across the
        // mlx-python/mlx-rs split (expected).
        let key = random::key(seed)?;
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        // --- Stage 1b (TI2V only): encode the conditioning image + build the mask-blend init ---
        // A `Reference` image → z48-VAE-encode to `z_img [z,1,h,w]`, build the first-frame mask
        // (`[z,T,h,w]`, 0 at frame 0) + per-token mask (`[1,L]`), and blend `(1−mask)·z_img +
        // mask·noise`. Without an image this is pure-noise T2V.
        // Channels-first `[z,1,h,w]` latent for one preprocessed TI2V image (z48-VAE-encode → reshape).
        let encode_kf = |vae: &Ti2vProviderVae, image: &Image| -> Result<Array> {
            let img_thwc = preprocess_ti2v_image(image, width, height)?; // [1,1,H,W,3]
            let z = vae.encode(&img_thwc)?; // [1,1,h,w,z]
            Ok(z.reshape(&z.shape()[1..])?.transpose_axes(&[3, 0, 1, 2])?) // [z,1,h,w]
        };
        let (t_lat, h_lat, w_lat) = (lat[1] as usize, lat[2] as usize, lat[3] as usize);
        let keyframes = req.keyframes();
        // **The seam** (sc-19571). Every conditioning strength the request carries is resolved into
        // `(latent_frame, strength)` pins HERE, once, and the mask builder below is driven by these
        // and by nothing else — so a strength cannot be "accepted at the boundary and defaulted on
        // the way in" the way it was when the builder took bare indices. `resolve_ti2v_pins` is pure
        // and unit-tested against non-default strengths (`ti2v_pins_carry_*`).
        let pins = resolve_ti2v_pins(req, t_lat)?;
        let (latents_init, ti2v) = if pins.is_empty() {
            (init_noise.clone(), None)
        } else {
            // Wan-native first_last_frame / multi-keyframe (sc-3357) or a single `Reference` image:
            // encode the conditioning frame(s), scatter them into the clean latent, then mask-blend.
            // The VAE encoder is scoped to this block so it drops before the mask build (sc-12796).
            let z = {
                let w = Weights::from_file(self.root.join("vae.safetensors"))?;
                let vae = Ti2vProviderVae::from_weights(&w)?;
                if keyframes.is_empty() {
                    // `[z,1,h,w]` — broadcasts over T_lat in the blend (the pin is frame 0).
                    let (image, _) = i2v_reference(req).expect("pins are non-empty");
                    encode_kf(&vae, image)?
                } else {
                    // Positional with `pins`: `resolve_ti2v_pins` walks `req.keyframes()` in the same
                    // request order, so `pins[i]` is this keyframe's resolved latent frame.
                    let mut frames: Vec<(Array, usize)> = Vec::with_capacity(keyframes.len());
                    for (kf, &(idx, _)) in keyframes.iter().zip(pins.iter()) {
                        frames.push((encode_kf(&vae, kf.image)?, idx));
                    }
                    build_ti2v_keyframe_z(&frames, cfg.vae_z_dim, t_lat, h_lat, w_lat)?
                }
            };
            let (mask, mask_tokens) =
                build_ti2v_mask(&pins, cfg.vae_z_dim, t_lat, h_lat, w_lat, cfg.patch_size);
            let latents = ti2v_blend_init(&z, &mask, &init_noise)?;
            mlx_rs::transforms::eval([&latents, &z])?;
            (latents, Some((z, mask, mask_tokens)))
        };
        // sc-12796: the TI2V/keyframe path above loads the z48 VAE **encoder** to encode the reference
        // image(s), then drops it at the brace (`latents_init`/`z` are eval'd and independent of it).
        // Under `Sequential`, flush the VAE encoder out of the cache before the DiT loads so the denoise
        // stage's footprint is the DiT alone, not the DiT + a cache-resident VAE encoder. Pure-noise T2V
        // (`ti2v` is `None`) loaded no VAE here — nothing to flush.
        if sequential && ti2v.is_some() {
            mlx_rs::memory::clear_cache();
        }

        // --- Stage 2: load the DiT, merge adapters + quantize, embed contexts, denoise (→ freed) ---
        let latents = {
            let mut w = Weights::from_file(self.root.join("model.safetensors"))?;
            // Adapter routing (sc-2683 / sc-2393 / sc-10045), two mutually-exclusive paths:
            //   • DENSE bf16 snapshot → FOLD LoRA/LoKr into the dense weights BEFORE building (the fork
            //     order: fold, then `spec.quantize` may quantize the merged weight). No-op w/o adapters.
            //   • PRE-QUANTIZED (packed Q4/Q8) snapshot → build packed first, then install LoRA as
            //     forward-time residuals on the built DiT (base stays packed; LoKr/LoHa rejected up
            //     front, deferred to sc-10050/sc-10051).
            let prequantized = self.config.quantization.is_some();
            if !prequantized {
                self.merge_adapters(&mut w)?;
            }
            let mut dit = WanTransformer::from_weights(&w, cfg)?;
            if let Some(q) = self.quant {
                dit.quantize(q.bits(), None)?;
            }
            if prequantized {
                self.install_adapters_additive(&mut dit)?;
            }
            let ctx_cond = dit.embed_text(&context)?;
            let ctx_uncond = match &context_null {
                Some(cn) => Some(dit.embed_text(cn)?),
                None => None,
            };
            let total = steps as u32;
            // Provider-neutral benchmark boundary: model construction, adapter/quant work, and
            // context embedding above are pre-denoise preparation. Emit only when the solver is
            // ready to consume its first latent, matching Qwen and SDXL.
            mlx_gen::diagnostics::record_phase_boundary(
                mlx_gen::diagnostics::BenchmarkPhaseBoundary::DenoiseStart,
            );
            // Curated unified solver (epic 7114, sc-7121): the gen-core-only solvers route through the
            // shared `denoise_curated`; the native unipc/euler/dpmpp2m stay on `scheduler.rs` (N1). The
            // image-conditioned TI2V mask-blend (per-token timesteps + a post-step re-blend) has no
            // single-eval curated-sampler hook, so it stays native-only.
            match (&ti2v, is_wan_curated(req.sampler.as_deref())) {
                (Some(_), true) => {
                    // Unreachable via requests — `validate_impl` rejects curated × mask-blend up
                    // front (F-074a); kept as a defensive backstop for direct callers.
                    return Err(Error::Msg(
                        "wan: curated samplers (euler_ancestral/heun/dpmpp_sde/ddim) are not \
                         supported with image-conditioned TI2V mask-blend — use unipc/euler/dpmpp2m"
                            .into(),
                    ));
                }
                (Some((z_img, mask, mask_tokens)), false) => {
                    // Unwired for the denoise feature cache: the mask-blend re-mixes the conditioning
                    // latent into the trajectory after every step (sc-18322).
                    refuse_unwired_approximation(
                        self.descriptor.id,
                        "TI2V mask-blend",
                        &approximation,
                    )?;
                    let mut on_step = |i: usize| {
                        on_progress(Progress::Step {
                            current: i as u32,
                            total,
                        })
                    };
                    denoise_ti2v(
                        &dit,
                        kind,
                        cfg.num_train_timesteps,
                        steps,
                        shift,
                        guidance,
                        &ctx_cond,
                        ctx_uncond.as_ref(),
                        &latents_init,
                        z_img,
                        mask,
                        mask_tokens,
                        &req.cancel,
                        &mut on_step,
                    )?
                }
                (None, true) => {
                    // Unwired: a curated solver evaluates the model 1..N times per solver step, so
                    // "the previous step's residual" has no single meaning (sc-18322).
                    refuse_unwired_approximation(
                        self.descriptor.id,
                        "curated unified solver",
                        &approximation,
                    )?;
                    denoise_curated(
                        &dit,
                        req.sampler.as_deref().expect("is_wan_curated ⇒ Some"),
                        cfg.num_train_timesteps,
                        steps,
                        shift,
                        guidance,
                        &ctx_cond,
                        ctx_uncond.as_ref(),
                        &latents_init,
                        seed,
                        &req.cancel,
                        on_progress,
                    )?
                }
                (None, false) => {
                    let mut on_step = |i: usize| {
                        on_progress(Progress::Step {
                            current: i as u32,
                            total,
                        })
                    };
                    // The one wired route: native dense T2V, one forward per step over one
                    // transformer, with the step index in hand (sc-18322). Neither `TrunkCache` nor
                    // `TokenPruner` has a production constructor, so both are `None` in every non-test
                    // build — the plan being `Exact` is the contract-level reason, this is the
                    // mechanism-level one. The step count is only known here, which is why the
                    // inert-over-the-trajectory refusal lives in these two bridges.
                    let grid = dit.patch_grid(&latents_init);
                    let mut trunk = trunk_cache(self.descriptor.id, &approximation, steps)?;
                    let mut prune = token_pruner(
                        self.descriptor.id,
                        &approximation,
                        steps,
                        grid.0 * grid.1 * grid.2,
                    )?;
                    denoise_approx(
                        &dit,
                        kind,
                        cfg.num_train_timesteps,
                        steps,
                        shift,
                        guidance,
                        &ctx_cond,
                        ctx_uncond.as_ref(),
                        &latents_init,
                        &req.cancel,
                        &mut on_step,
                        trunk.as_mut(),
                        prune.as_mut(),
                    )?
                }
            }
        };
        // sc-12796: the DiT drops at the brace above; `latents` is fully materialized (the per-step
        // `eval` in `denoise` / `denoise_ti2v` / `denoise_curated` detached it from the DiT weights), so
        // nothing live references the transformer. Under `Sequential`, flush the dropped DiT out of the
        // cache before the z48 VAE **decoder** loads, so the decode stage doesn't stack on a
        // cache-resident DiT. On the dense 5B the DiT is small (~10 GB bf16 / ~2.9 GB q4) but the z48
        // `vae22` decode is the heavy stage, so keeping its dead buffers off-GPU is the point.
        if sequential {
            mlx_rs::memory::clear_cache();
        }

        // --- Stage 3: z48 vae22 decode → RGB8 frames ---
        mlx_gen::diagnostics::record_phase_boundary(
            mlx_gen::diagnostics::BenchmarkPhaseBoundary::DecodeStart,
        );
        on_progress(Progress::Decoding);
        // Causal temporal decode: t_lat → 1 + (t_lat−1)·4 output frames (= gen_frames). The tiling
        // was chosen (and budget-checked) up front by `auto_tiling_budgeted` (sc-4998). sc-5039 casts
        // the decoder weights to bf16 (the conv-heavy body runs bf16, RMS_norm + denorm stay f32) —
        // `Wan22Vae` infers its compute dtype from the loaded weights. Decode-only: the conditioning
        // image was already encoded above through a separate f32 VAE load.
        let frames_u8 = {
            let mut w = Weights::from_file(self.root.join("vae.safetensors"))?;
            w.cast_all(mlx_rs::Dtype::Bfloat16)?;
            let vae = Ti2vProviderVae::from_weights(&w)?;
            decode_to_frames_22(&vae, &latents, decode_tiling.as_ref(), Some(&req.cancel))?
        };
        let mut images = frames_to_images(&frames_u8)?;
        // Discard the extra leading frames generated for `trim_first_frames`.
        if trim_out > 0 {
            images.drain(0..trim_out.min(images.len()));
        }

        let fps = req.fps.unwrap_or(cfg.sample_fps);
        Ok(GenerationOutput::Video {
            frames: images,
            fps,
            audio: None,
        })
    }
}

// The registration constants bridge the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_generators! {
    pub(crate) const TI2V_REGISTRATION = descriptor => load
}

// ===========================================================================================
// Wan2.2 T2V-A14B — dual-expert MoE text→video (the S1–S5 core, fully wired)
// ===========================================================================================

/// Public provider id for the dual-expert MoE T2V model: `"wan2_2_t2v_14b"`.
pub const MODEL_ID_T2V_14B: &str = "wan2_2_t2v_14b";

/// Stable identity + advertised capabilities for the Wan2.2 T2V-A14B (dual-expert MoE text→video).
pub fn descriptor_t2v_14b() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID_T2V_14B,
        family: "wan",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // CFG with the per-expert (low, high) guidance pair + the Chinese anti-artifact negative
            // prompt. Pure text→video: no image conditioning.
            supports_negative_prompt: true,
            supports_guidance: true,
            // LoRA + LoKr merge per-expert at generate time (sc-2683 / sc-2393, PEFT/kohya + LoKr,
            // MoE high/low); Q4/Q8 (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot.
            supports_lora: true,
            supports_lokr: true,
            samplers: wan_samplers(),
            // H/W align to patch×vae_stride = 16 (z16 VAE, spatial stride 8); long edge cap 1280.
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Cross-attention text K/V is cached across denoise steps (per expert).
            supports_kv_cache: true,
            // A14B honors `OffloadPolicy::Sequential` (epic 12732, sc-12736): the staged expert swap
            // holds only the ACTIVE MoE expert resident (never both) and frees the UMT5 TE / VAE
            // off-GPU during denoise, dropping the unified-memory peak to ~one expert. Advertised so
            // the worker's fit-gate can tell "bounds peak here" from a no-op fallback.
            supports_sequential_offload: true,
            // TE/VAE and the active expert are phase-staged even under Resident; Sequential adds the
            // stronger cache-flush/expert-residency controls described above.
            unconditionally_engages_staged_residency: true,
            ..Default::default()
        },
    }
}

/// The loaded Wan2.2 T2V-A14B. Holds the resolved config + the snapshot directory; the heavy
/// components (UMT5 TE, the two 14B experts, the z16 VAE) are **staged** inside
/// [`Wan14b::generate`] — loaded, used, then dropped in turn — to bound peak memory (mirrors
/// `generate_wan.py`, which never holds the T5 encoder and both 27 GB experts resident at once).
pub struct Wan14b {
    descriptor: ModelDescriptor,
    config: WanModelConfig,
    root: PathBuf,
    /// LoRA adapters merged onto the experts at generate time (sc-2683). Empty for a plain load;
    /// `moe_expert`-tagged specs route to the high/low expert (shared = both).
    adapters: Vec<AdapterSpec>,
    /// Optional Q4/Q8 quantization for the transformer experts (sc-2682). `None` = dense bf16 (or a
    /// pre-quantized snapshot, which `from_weights` builds packed from its `config.json` manifest —
    /// see [`resolve_load_time_quant`]). When `Some`, [`Wan14b::generate`] quantizes **each** expert
    /// independently after load (transformer-only: attn `q/k/v/o` + `ffn.fc1/fc2`; T5 + VAE stay f32).
    quant: Option<Quant>,
    /// Component-residency strategy for the denoise (epic 12732, sc-12736). [`OffloadPolicy::Resident`]
    /// (default) holds **both** ~8-9 GB MoE experts resident for the whole loop (the byte-identical
    /// pre-swap path). [`OffloadPolicy::Sequential`] runs the **expert swap** — only the ACTIVE expert
    /// is resident (high `0..k`, then evict-then-load the low `k..steps`, never co-resident) and the
    /// UMT5 TE / VAE encoder are `clear_cache`-freed off-GPU during denoise — bounding the unified-memory
    /// peak to ~one expert. Advertised via `supports_sequential_offload` so the worker's fit-gate can
    /// select it under a memory ceiling. Captured from [`LoadSpec::offload_policy`] at load.
    offload_policy: OffloadPolicy,
    /// Exact provider-owned public-I2V artifact and adapter receipt. Absent on direct T2V loads.
    i2v_memory: Option<crate::i2v_memory_strategy::PreparedWanI2vMemory>,
}

impl Wan14b {
    /// The resolved model config.
    pub fn config(&self) -> &WanModelConfig {
        &self.config
    }

    /// Enforce the "matched no module across either expert" error + surface the combined partial skips
    /// after an adapter apply (fold OR additive) — shared by both paths so the message can't drift. A
    /// non-empty adapter set that applied nothing across BOTH experts is a format/prefix
    /// misconfiguration; per-key skips (a target absent from this checkpoint) are surfaced as a
    /// warning, not fatal, mirroring the reference.
    fn finalize_dual_report(&self, low: WanLoraReport, high: WanLoraReport) -> Result<()> {
        if low.applied + high.applied == 0 {
            return Err(Error::Msg(format!(
                "{}: {} LoRA file(s) matched no module across either expert — check the format \
                 (expected PEFT `lora_A/B` or kohya `lora_down/up`, `diffusion_model.`-prefixed Wan \
                 module names)",
                self.descriptor.id,
                self.adapters.len()
            )));
        }
        let mut skipped = low.skipped;
        skipped.extend(high.skipped);
        skipped.sort();
        skipped.dedup();
        warn_skipped_adapters(self.descriptor.id, &skipped);
        Ok(())
    }

    /// **Dense-bf16 path.** Fold the load-time LoRA/LoKr adapters onto the two expert weight maps in
    /// place (sc-2683), before the [`WanTransformer`]s are built (the reference fold order: LoRA folds
    /// into the dense weight, then `spec.quantize` may quantize it). No-op without adapters (the
    /// no-adapter path is byte-identical). Shared specs merge onto both experts, `moe_expert`-tagged
    /// specs onto their own (the reference `(loras)+(loras_high/low)` split). Called only for a dense
    /// snapshot; a pre-quantized snapshot uses [`Wan14b::install_adapters_additive`] (sc-10045).
    fn merge_adapters(&self, low_w: &mut Weights, high_w: &mut Weights) -> Result<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        let low = merge_wan_adapters(low_w, &self.adapters, MoeExpert::Low)?;
        let high = merge_wan_adapters(high_w, &self.adapters, MoeExpert::High)?;
        self.finalize_dual_report(low, high)
    }

    /// **Pre-quantized (packed Q4/Q8) path (sc-10045).** Install the load-time LoRA adapters onto the
    /// two already-built (packed) expert [`WanTransformer`]s as forward-time residuals — the bases
    /// stay packed, never dequantized (removing the old `model.rs:614` rejection). No-op without
    /// adapters. Shared specs install onto both experts, `moe_expert`-tagged onto their own (same
    /// high/low routing as the fold path). Plain LoRA and **LoKr** both apply (LoKr via the structured
    /// deferred-Kronecker vec-trick, sc-10050); only LoHa on a packed tier is rejected up front with an
    /// actionable error (deferred to sc-10051). Called only for a pre-quantized snapshot; a dense
    /// snapshot uses [`Wan14b::merge_adapters`] instead.
    fn install_adapters_additive(
        &self,
        low_dit: &mut WanTransformer,
        high_dit: &mut WanTransformer,
    ) -> Result<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        reject_loha_on_packed(self.descriptor.id, &self.adapters)?;
        let low = apply_wan_adapters_additive(low_dit, &self.adapters, MoeExpert::Low)?;
        let high = apply_wan_adapters_additive(high_dit, &self.adapters, MoeExpert::High)?;
        self.finalize_dual_report(low, high)
    }

    /// Build **one** MoE expert's transformer, staged (sc-12736): load its
    /// `{high,low}_noise_model.safetensors`, apply *this* expert's adapters (dense FOLD before the
    /// build / packed ADDITIVE install after it), and quantize if requested — the per-expert half of
    /// the resident Stage-2 build, extracted so the sequential expert swap can materialize exactly one
    /// expert at a time. Returns the transformer plus its adapter report (`Some` iff adapters are
    /// configured); the caller feeds both experts' reports to [`Wan14b::finalize_dual_report`] for the
    /// cross-expert "matched no module across either expert" guard, exactly as the resident path does.
    /// The build is byte-identical to the resident per-expert build — the experts are independent
    /// weight files, so building one never touches the other (the numerics-preserving requirement).
    fn build_expert_staged(
        &self,
        which: MoeExpert,
    ) -> Result<(WanTransformer, Option<WanLoraReport>)> {
        let cfg = &self.config;
        let file = match which {
            MoeExpert::Low => "low_noise_model.safetensors",
            MoeExpert::High => "high_noise_model.safetensors",
        };
        let mut w = Weights::from_file(self.root.join(file))?;
        let prequantized = cfg.quantization.is_some();
        let has_adapters = !self.adapters.is_empty();
        // DENSE bf16 snapshot: FOLD this expert's LoRA/LoKr into the weights BEFORE building (fold,
        // then `spec.quantize` may quantize the merged weight) — mirrors `merge_adapters`, per expert.
        let fold_report = if !prequantized && has_adapters {
            Some(merge_wan_adapters(&mut w, &self.adapters, which)?)
        } else {
            None
        };
        let mut dit = WanTransformer::from_weights(&w, cfg)?;
        if let Some(q) = self.quant {
            dit.quantize(q.bits(), None)?;
        }
        // PRE-QUANTIZED (packed Q4/Q8) snapshot: install this expert's adapters as forward-time
        // residuals AFTER building (the bases stay packed) — mirrors `install_adapters_additive`.
        let report = if prequantized && has_adapters {
            reject_loha_on_packed(self.descriptor.id, &self.adapters)?;
            Some(apply_wan_adapters_additive(
                &mut dit,
                &self.adapters,
                which,
            )?)
        } else {
            fold_report
        };
        Ok((dit, report))
    }

    /// The **sequential-offload** dual-expert denoise (sc-12736, epic 12732) — the staged twin of the
    /// resident [`denoise_moe`] that keeps only the ACTIVE ~8-9 GB expert resident, dropping the
    /// unified-memory denoise peak to ~one expert on a Mac ceiling. High runs steps `0..k`
    /// (`t ≥ boundary_timestep`); then, via [`staged_expert_swap`]'s **evict-then-load** discipline, the
    /// high expert is dropped and its buffers `clear_cache`-flushed **before** the low expert loads,
    /// which runs `k..steps`. Flow-match timesteps decrease monotonically ⇒ exactly one boundary
    /// crossing ⇒ at most one swap; the two experts are **never co-resident** (the Pillar-1 win).
    ///
    /// **Parity:** one continuous scheduler advances across the swap and each expert drives the
    /// identical per-step [`denoise_range`] the resident [`denoise_moe`] drives, so the prefix/suffix
    /// split reproduces the resident per-step `t ≥ boundary` choice bit-for-bit — only residency /
    /// lifetime changes. Each expert projects the raw UMT5 `context`/`context_null` through its **own**
    /// `text_embedder` after it loads (the entangled per-expert projection), so the small raw contexts
    /// stay resident across the swap. Curated solvers use a separate per-eval swap because Heun and
    /// DPM++ SDE may cross the boundary during a sub-evaluation.
    #[allow(clippy::too_many_arguments)]
    fn denoise_moe_swapped(
        &self,
        context: &Array,
        context_null: &Array,
        low_gs: f32,
        high_gs: f32,
        kind: SolverKind,
        num_train_timesteps: usize,
        steps: usize,
        shift: f32,
        boundary_timestep: f32,
        init_noise: &Array,
        y: Option<&Array>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        // sc-2957 compiled elementwise glue, active for the whole (both-expert) denoise — see
        // `denoise_moe`. Scoped + restored on drop by the RAII guard.
        let _compile_glue = crate::transformer::CompileGlueGuard::enable();
        let mut sched = make_scheduler(kind, num_train_timesteps);
        sched.set_timesteps(steps, shift);
        let timesteps: Vec<f32> = sched.timesteps().to_vec();
        // High runs `0..k` (`t ≥ boundary`), low runs `k..steps` — the single high→low crossing.
        let k = crossing_index(&timesteps, boundary_timestep);
        let mut latents = init_noise.clone();
        let total = steps as u32;

        // Each expert's adapter report, carried out of its (dropped-before-the-next-loads) load closure
        // so the cross-expert "matched no module across either expert" guard runs once both are known.
        let high_report: std::cell::Cell<Option<WanLoraReport>> = std::cell::Cell::new(None);
        let low_report: std::cell::Cell<Option<WanLoraReport>> = std::cell::Cell::new(None);

        // Build a borrowed `Expert` (per-expert `text_embedder` projection of the raw contexts) and run
        // its step range through the shared scheduler/latents — the identical per-step math as resident.
        let denoise_expert = |dit: &WanTransformer,
                              guidance: f32,
                              range: std::ops::Range<usize>,
                              st: &mut SwapState| {
            let e = Expert {
                transformer: dit,
                ctx_cond: dit.embed_text(context)?,
                ctx_uncond: Some(dit.embed_text(context_null)?),
                guidance,
            };
            let grid = dit.patch_grid(init_noise);
            let op = &mut *st.on_progress;
            let mut on_step = |i: usize| {
                op(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_range(
                &mut *st.sched,
                &e,
                grid,
                y,
                &mut *st.latents,
                &timesteps,
                range,
                cancel,
                &mut on_step,
            )
        };

        let mut state = SwapState {
            sched: &mut *sched,
            latents: &mut latents,
            on_progress: &mut *on_progress,
        };
        staged_expert_swap(
            k,
            steps,
            &mut state,
            // load high
            |st| {
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                (st.on_progress)(Progress::Loading(LoadPhase::Renderer));
                let (dit, report) = self.build_expert_staged(MoeExpert::High)?;
                high_report.set(report);
                Ok(dit)
            },
            // use high over `0..k`
            |dit, st| denoise_expert(dit, high_gs, 0..k, st),
            // load low
            |st| {
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                (st.on_progress)(Progress::Loading(LoadPhase::Renderer));
                let (dit, report) = self.build_expert_staged(MoeExpert::Low)?;
                low_report.set(report);
                // Both experts' reports are now known → run the cross-expert zero-match guard before
                // the low denoise (fail fast). An expert that owned no steps (didn't load) contributes
                // a default (applied 0) report, so the guard still requires SOME loaded expert to have
                // matched. No-op without adapters. (For the A14B the schedule always spans ~999→0, so
                // both experts always own steps — `k` is strictly inside `0..steps`.)
                if !self.adapters.is_empty() {
                    self.finalize_dual_report(
                        low_report.take().unwrap_or_default(),
                        high_report.take().unwrap_or_default(),
                    )?;
                }
                Ok(dit)
            },
            // use low over `k..steps`
            |dit, st| denoise_expert(dit, low_gs, k..steps, st),
            // evict: flush the just-dropped expert's buffers to the OS BEFORE the next loads.
            || {
                mlx_rs::memory::clear_cache();
                Ok(())
            },
        )?;
        Ok(latents)
    }
}

/// The mutable denoise state threaded through [`staged_expert_swap`] (sc-12736): the continuous
/// scheduler, the in-place latents, and the progress sink. Held as `&mut` references so exclusive
/// access moves between the load/use closures via the `&mut SwapState` param (disjoint-field borrows)
/// rather than being captured by each closure — the borrow-checker-clean way to let the two swap
/// stages share the scheduler/latents/progress without a `RefCell`.
struct SwapState<'a> {
    sched: &'a mut dyn WanScheduler,
    latents: &'a mut Array,
    on_progress: &'a mut dyn FnMut(Progress),
}

/// Resolve the **load-time** quantization to apply in [`Wan14b::generate`], reconciling the requested
/// `spec.quantize` against a pre-quantized snapshot's `config.json` manifest (`cfg.quantization`).
///
/// A *pre-quantized* snapshot (manifest present) ships packed weights on disk → [`WanTransformer::
/// from_weights`] builds the experts quantized directly (the `loading.py` consume path), so **no**
/// load-time re-quantization is applied (returns `None`). A *dense bf16* snapshot honors
/// `spec.quantize` (quantized in-memory after load). A bits conflict is a hard error: the on-disk
/// manifest is authoritative, so we don't silently ignore (or re-quantize at) a different width.
/// (This is a deliberately *loud* "stored wins" — a pre-quantized snapshot at a different width is a
/// hard error here, not a silent downgrade.)
fn resolve_load_time_quant(
    id: &str,
    cfg: &WanModelConfig,
    requested: Option<Quant>,
) -> Result<Option<Quant>> {
    match (cfg.quantization, requested) {
        (Some(stored), Some(req)) if stored.bits != req.bits() => Err(Error::Msg(format!(
            "{id}: snapshot is pre-quantized {}-bit (config.json quantization block), but \
             spec.quantize requested {}-bit — the on-disk manifest is authoritative; drop the \
             precision override or convert a snapshot at the requested width",
            stored.bits,
            req.bits()
        ))),
        // Pre-quantized snapshot: `from_weights` builds it quantized; no load-time requant.
        (Some(_), _) => Ok(None),
        // Dense bf16 snapshot: quantize at load if requested.
        (None, req) => Ok(req),
    }
}

/// The z16 decode tiling for one A14B render, from the decoded output dims.
///
/// A BoundedDecode the route's own contract admitted has to be REACHED: the sealed edge/overlap
/// pair only ever comes out of [`crate::i2v_memory_strategy::decode_tiling`], never out of the auto
/// selector. The I2V arm reaches it through its `image_to_video` mode; the T2V route has no such
/// mode, so the `tile_vae_decode` knob is the second door — otherwise the T2V provider would
/// publish a BoundedDecode contract, admit the rung, and silently decode with the auto tile
/// (sc-22738). Everything else keeps the z16 selector, which is the A14B autoencoder's own cost
/// model.
pub(crate) fn a14b_decode_tiling(
    req: &GenerationRequest,
    out_width: i32,
    out_height: i32,
    out_frames: i32,
) -> Result<Option<mlx_gen::TilingConfig>> {
    let bounded_decode = req.memory.is_some_and(|memory| memory.tile_vae_decode);
    if bounded_decode
        || (req.video_mode.as_deref() == Some("image_to_video") && req.memory.is_some())
    {
        crate::i2v_memory_strategy::decode_tiling(
            req,
            out_width as u32,
            out_height as u32,
            out_frames as u32,
        )
    } else {
        auto_tiling_budgeted_z16(out_height, out_width, out_frames)
    }
}

/// Load the Wan2.2 T2V-A14B from a converted MLX snapshot directory (`convert_wan.py` output:
/// `low_noise_model.safetensors` + `high_noise_model.safetensors` + `t5_encoder.safetensors` +
/// `vae.safetensors` + `tokenizer.json` + `config.json`). LoRA adapters merge per-expert at generate
/// time (sc-2683); Q4/Q8 (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot. LoKr is the
/// sibling sc-2393.
pub fn load_t2v_14b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_t2v_14b: expected a model directory (converted MLX snapshot), not a single \
                 file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_t2v_14b: precision override is not wired (the experts run bf16 GEMMs over an \
             f32 residual stream — the parity regime)"
                .into(),
        ));
    }

    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.dual_model {
        return Err(Error::Msg(format!(
            "wan2_2_t2v_14b: config.json is not a dual-expert model (dual_model=false, \
             model_type={}); expected the converted Wan2.2 A14B MoE checkpoint",
            config.model_type
        )));
    }
    let quant = resolve_load_time_quant(MODEL_ID_T2V_14B, &config, spec.quantize)?;
    // The T2V-A14B route publishes the same pre-load memory surface as its I2V sibling (`lib.rs`
    // registers `i2v_memory_strategy::t2v_14b::MEMORY_{REGISTRATION,FIXTURE,BEHAVIOR}`), so the
    // LOADED provider has to seal the same receipt. Without it the registry declares a contract
    // the generator never publishes: `memory_strategy_contract()` answers `None`,
    // `memory_strategy_safety_check` refuses every optimized rung with "has no prepared I2V memory
    // receipt", and `begin_memory_strategy_request` opens no scope — so a selection the video gate
    // admits dies at the consumer's scope check (sc-22738).
    let i2v_memory = if spec.resolved_route.as_deref() == Some(MODEL_ID_T2V_14B)
        && spec.prepared_file_pins().is_prepared()
    {
        Some(crate::i2v_memory_strategy::prepare(spec, MODEL_ID_T2V_14B).map_err(Error::from)?)
    } else {
        None
    };
    Ok(Box::new(Wan14b {
        descriptor: descriptor_t2v_14b(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
        offload_policy: spec.offload_policy,
        i2v_memory,
    }))
}

impl Generator for Wan14b {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, request: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        self.validate_impl(request).map_err(Into::into)
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        // SC-20686 Metal lane: a direct `generate_impl` call unless a campaign is armed.
        mlx_gen::sc20686::observe_generation(
            &self.root,
            &request.cancel,
            self.descriptor.id,
            || mlx_gen::sc20686::RequestFacts::from_request(request),
            on_progress,
            |on_progress| self.generate_impl(request, on_progress),
        )
        .map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        self.i2v_memory.as_ref().map(|prepared| &prepared.contract)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        self.i2v_memory.as_ref().map_or_else(
            || {
                if context.selection.strategy == mlx_gen::gen_core::MemoryStrategy::Resident {
                    mlx_gen::gen_core::MemorySafetyDecision::Accept
                } else {
                    mlx_gen::gen_core::MemorySafetyDecision::Reject {
                        reason: format!(
                            "{} has no prepared I2V memory receipt",
                            self.descriptor.id
                        ),
                    }
                }
            },
            |prepared| crate::i2v_memory_strategy::safety_check(prepared, context),
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        match &self.i2v_memory {
            Some(prepared) => crate::i2v_memory_strategy::begin_request(prepared, context),
            None => Ok(None),
        }
    }
}

impl Wan14b {
    /// Validate body — kept on the crate's own [`mlx_gen::Error`] so `?` on the capability check
    /// lifts transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        let id = self.descriptor.id;
        // Shared capability floor: size range (the advertised `min_size` = patch×vae_stride = 16 is
        // the sub-tile lower bound; `max_size` caps the long edge), count, guidance/negative/true_cfg,
        // sampler (`unipc`/`euler`/`dpmpp2m`), scheduler, and conditioning (none for T2V, `Reference`
        // for I2V).
        self.descriptor.capabilities.validate_request(id, req)?;
        if let Some(frames) = req.frames {
            // num_frames must be 1 + 4·k (one VAE temporal chunk + 4× per chunk).
            if frames % 4 != 1 {
                return Err(Error::Msg(format!(
                    "{id}: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
            if frames as usize > crate::MAX_WAN_FRAMES {
                return Err(Error::Msg(format!(
                    "{id}: num_frames {frames} exceeds the maximum {}",
                    crate::MAX_WAN_FRAMES
                )));
            }
        }
        // Reject an off-grid or over-area geometry rather than silently align-down refitting it.
        // sc-12607: the 16-px grid stride (candle rejects via `is_multiple_of(SIZE_MULTIPLE_14B)`).
        // sc-12308: the area cap (T2V was uncapped entirely; I2V refit 1280×720 → 1264×704). Both A14B
        // variants share the 14B family's 921 600 budget.
        let (dw, dh) = grid(&self.config, A14bProviderVae::VAE_TILING);
        reject_off_grid(id, req, dw, dh)?;
        reject_over_area(id, req, dw, dh, self.config.max_area)?;
        // I2V channel-concat requires a single reference image (the first conditioning frame), and
        // does not support `trim_first_frames` (the reference builds `y` from `num_frames`, so an
        // extended noise length would mismatch the conditioning's temporal dim).
        if self.config.is_i2v_concat() {
            let Some((_, strength)) = i2v_reference(req) else {
                return Err(Error::Msg(format!(
                    "{id}: image-to-video requires a Reference conditioning image"
                )));
            };
            // sc-19571 — **refuse, do not ignore.** The A14B I2V conditions by CHANNEL-CONCAT: the
            // reference is VAE-encoded into `y` and concatenated onto every forward's latent, with
            // no denoise mask anywhere in the path. There is therefore nothing for a conditioning
            // strength to weight, and the 5B's `1 − strength` mask-blend cannot be extended here
            // (sc-19504). Accepting the knob and quietly rendering a full pin is the exact failure
            // this story exists to close, so a request that asks for anything other than a full pin
            // gets a typed error naming the mechanism.
            if strength != 1.0 {
                return Err(Error::Msg(format!(
                    "{id}: conditioning strength is not supported for A14B image-to-video (got \
                     {strength}) — the reference is channel-concatenated, not mask-blended, so \
                     there is no denoise mask to weight; use strength 1.0, or wan2_2_ti2v_5b, \
                     whose mask-blend honors it"
                )));
            }
            if req.trim_first_frames.unwrap_or(0) > 0 {
                return Err(Error::Msg(format!(
                    "{id}: trim_first_frames is not supported for I2V (the conditioning `y` is built \
                     from num_frames)"
                )));
            }
        }
        Ok(())
    }

    /// The full dual-expert MoE pipeline (port of `generate_wan.py`'s dual-model path) — serves both
    /// **T2V-A14B** and **I2V-A14B** (the struct's config selects). Resolves request knobs against the
    /// config defaults, then **stages** the phases to bound memory: (1) load UMT5, encode the prompt +
    /// negative prompt, drop the encoder; (1b, I2V only) load the z16 VAE encoder, build the
    /// channel-concat conditioning `y` from the reference image, drop it; (2) load both 14B experts,
    /// embed the contexts per expert, run the boundary-switched [`denoise_moe`] loop (with `y` for
    /// I2V), drop the experts; (3) load the z16 VAE, decode to RGB8 frames. CFG runs with the
    /// per-expert (low, high) guidance.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // Reject anything outside the advertised surface before doing expensive work — in particular
        // an unknown `sampler`, which `solver_kind` would otherwise silently map to UniPC.
        self.validate(req)?;
        if let Some(prepared) = &self.i2v_memory {
            crate::i2v_memory_strategy::validate_active_request(prepared, req)
                .map_err(Error::from)?;
        }
        // Every A14B route is an expert-swap route: the trajectory changes transformers at the
        // boundary, so a retained trunk residual or a rotation phase captured under the high-noise
        // expert is meaningless under the low-noise one (and retaining it would keep the outgoing
        // expert's weights alive). This provider therefore declares no approximate mechanism, and
        // refuses a non-exact plan **by name** here rather than by omission — resolving the plan is
        // what makes that refusal exist at all, and its absence was why the guard's doc claimed a
        // coverage it did not have (sc-18322).
        refuse_unwired_approximation(
            self.descriptor.id,
            "MoE expert swap",
            &self
                .descriptor
                .capabilities
                .approximation_plan(self.descriptor.id, req)?,
        )?;
        let cfg = &self.config;
        // Sequential offload (epic 12732, sc-12736): free the UMT5 TE / VAE off-GPU during denoise and
        // hold only the ACTIVE MoE expert resident (the expert swap). `Resident` (default) is the
        // byte-identical pre-swap path (both experts co-resident, no per-stage `clear_cache`).
        let sequential = self.offload_policy == OffloadPolicy::Sequential
            || crate::i2v_memory_strategy::staged(req);

        // --- Resolve request knobs against config defaults ---
        let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
        // trim_first_frames: generate `trim` extra leading temporal chunks (each = vae_stride_t = 4
        // latent frames → 4 output frames after the non-causal T→4T decode) and discard them after
        // decode, so the first kept frame sees a full temporal receptive field (port of
        // generate_wan.py). gen_frames stays 1+4k since frames is and we add a multiple of 4.
        let trim = req.trim_first_frames.unwrap_or(0) as usize;
        let vae_stride = provider_vae_stride(A14bProviderVae::VAE_TILING);
        let trim_out = trim * vae_stride.0; // discarded output frames = trim · 4
        let gen_frames = frames + trim * vae_stride.0;
        // validate() already rejected sub-tile + bad frame counts; round H/W down to the grid, then
        // enforce the model's max-area cap (I2V-14B / TI2V-5B: 704×1280; no-op for T2V's max_area 0).
        let (width, height) = resolve_capped_dims(req, cfg, A14bProviderVae::VAE_TILING);
        let (steps, shift, kind, seed) =
            resolve_sampler_knobs(req, cfg.sample_steps, cfg.sample_shift);
        // A scalar request `guidance` overrides both experts; otherwise use the config (low, high).
        let (low_gs, high_gs) = cfg.sample_guide_scale.resolve_dual(req.guidance);
        let neg_prompt = req
            .negative_prompt
            .clone()
            .unwrap_or_else(|| cfg.sample_neg_prompt.clone());

        // Init-noise latent geometry: [z_dim, t_lat, h_lat, w_lat] for the (possibly trim-extended)
        // generation length.
        let lat = latent_shape(gen_frames, height, width, cfg.vae_z_dim, vae_stride)?;

        // sc-4986 — fail fast (catchable) if the DiT-denoise stage won't fit, before any heavy load.
        // sc-12736/sc-12795: every `Sequential` route keeps only ONE expert resident; `Resident`
        // budgets BOTH so an oversized both-resident job is rejected before loading. The
        // denoise always runs cond+uncond (batch 2).
        let low_bytes =
            dit_resident_bytes(&[self.root.join("low_noise_model.safetensors")], self.quant);
        let high_bytes = dit_resident_bytes(
            &[self.root.join("high_noise_model.safetensors")],
            self.quant,
        );
        let adapter_mode = if self.config.quantization.is_some() {
            AdapterResidencyMode::Additive
        } else {
            AdapterResidencyMode::Folded
        };
        let (low_adapter_bytes, high_adapter_bytes) =
            wan14b_adapter_bytes_per_expert(&self.adapters, adapter_mode).ok_or_else(|| {
                Error::Msg(format!(
                    "{}: cannot size every additive adapter before the denoise fit gate",
                    self.descriptor.id
                ))
            })?;
        preflight_denoise_memory_guard(
            self.descriptor.id,
            wan14b_denoise_resident_bytes(
                self.offload_policy,
                req.sampler.as_deref(),
                low_bytes.saturating_add(low_adapter_bytes),
                high_bytes.saturating_add(high_adapter_bytes),
            ),
            seq_len(lat, cfg.patch_size),
            cfg.dim,
            true,
        )?;

        // --- Stage 1: UMT5 text encode (loaded → used → freed) ---
        // A14B always encodes both prompts (the dual-expert default), so `skip_neg = false`; unwrap
        // the always-present negative context back to the `Array` the MoE denoise expects.
        let (context, context_null) = encode_text_staged_for_tier(
            &self.root,
            cfg,
            &req.prompt,
            &neg_prompt,
            false,
            self.quant,
        )?;
        let context_null = context_null.expect("a14b always encodes the negative context");
        // sc-12736: `encode_text_staged` already drops the UMT5 encoder before returning (only the
        // small `[1, L, dim]` raw contexts survive — each expert projects them through its own
        // `text_embedder` below). Under `Sequential`, flush its ~11 GB f32 weights out of MLX's buffer
        // cache to the OS now, so the TE is truly off-GPU for the whole denoise. `Resident` leaves the
        // cache warm (byte-identical pre-swap behavior). `context`/`context_null` are already eval'd.
        if sequential {
            mlx_rs::memory::clear_cache();
        }

        // Seeded init noise (f32, no batch dim) — matches the reference's `mx.random.normal(shape)`
        // shape; exact seeded-RNG values differ across the mlx-python/mlx-rs split (expected). I2V
        // (like the reference) starts from pure noise — the image enters via the `y` channel-concat.
        let key = random::key(seed)?;
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        // --- Stage 1b (I2V only): build the channel-concat conditioning `y` (→ VAE encoder freed) ---
        // First frame = the reference image, the rest zero, VAE-encoded under a temporal mask →
        // `[20, T_lat, h_lat, w_lat]` (f32), concatenated onto each forward's noise latent in
        // `denoise_moe`. `frames` (not `gen_frames`) — validate() rejected `trim` for I2V.
        let y = if cfg.is_i2v_concat() {
            // sc-19571: the strength is deliberately NOT read here — the A14B I2V is channel-concat,
            // which has no denoise mask to weight, so `validate_impl` refuses a non-default strength
            // outright rather than accepting it and dropping it silently.
            let (image, _) = i2v_reference(req).ok_or_else(|| {
                Error::Msg(format!(
                    "{}: image-to-video requires a Reference conditioning image",
                    self.descriptor.id
                ))
            })?;
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = A14bProviderVae::from_weights(&w)?;
            let y = build_i2v_y(&vae, image, frames, height, width, vae_stride)?;
            mlx_rs::transforms::eval([&y])?;
            Some(y)
        } else {
            None
        };
        // sc-12736: the VAE encoder (loaded only to build `y`) drops at the brace above; `y` is eval'd
        // and independent of it. Under `Sequential`, flush the VAE out of the cache before the experts
        // load so the denoise peak is one expert, not one expert + a resident VAE. No-op for T2V.
        if sequential && y.is_some() {
            mlx_rs::memory::clear_cache();
        }

        // --- Stage 2: dual-expert MoE denoise (→ freed). Under `Sequential`, native solvers swap at
        // the step boundary and curated solvers at the exact model-evaluation boundary; both keep one
        // expert resident. `Resident` keeps both experts for byte-identical legacy behavior. ---
        let latents = if sequential && is_wan_curated(req.sampler.as_deref()) {
            let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32;
            let low_report: std::cell::Cell<Option<WanLoraReport>> = std::cell::Cell::new(None);
            let high_report: std::cell::Cell<Option<WanLoraReport>> = std::cell::Cell::new(None);
            on_progress(Progress::Loading(LoadPhase::Renderer));
            let latents = denoise_moe_curated_swapped(
                boundary_timestep,
                req.sampler.as_deref().expect("curated sampler"),
                cfg.num_train_timesteps,
                steps,
                shift,
                &init_noise,
                y.as_ref(),
                seed,
                &req.cancel,
                on_progress,
                |high| {
                    if req.cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
                    let which = if high {
                        MoeExpert::High
                    } else {
                        MoeExpert::Low
                    };
                    let (dit, report) = self.build_expert_staged(which)?;
                    let cond = dit.embed_text(&context)?;
                    let uncond = Some(dit.embed_text(&context_null)?);
                    if high {
                        high_report.set(report);
                    } else {
                        low_report.set(report);
                        if !self.adapters.is_empty() {
                            self.finalize_dual_report(
                                low_report.take().unwrap_or_default(),
                                high_report.take().unwrap_or_default(),
                            )?;
                        }
                    }
                    Ok((dit, cond, uncond, if high { high_gs } else { low_gs }))
                },
            )?;
            latents
        } else if sequential {
            let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32;
            self.denoise_moe_swapped(
                &context,
                &context_null,
                low_gs,
                high_gs,
                kind,
                cfg.num_train_timesteps,
                steps,
                shift,
                boundary_timestep,
                &init_noise,
                y.as_ref(),
                &req.cancel,
                on_progress,
            )?
        } else {
            let mut low_w = Weights::from_file(self.root.join("low_noise_model.safetensors"))?;
            let mut high_w = Weights::from_file(self.root.join("high_noise_model.safetensors"))?;
            // Adapter routing (sc-2683 / sc-2393 / sc-10045), two mutually-exclusive paths:
            //   • DENSE bf16 snapshot → FOLD LoRA/LoKr into each expert's dense weights BEFORE
            //     building (the fork order: fold, then `spec.quantize` may quantize the merged weight).
            //   • PRE-QUANTIZED (packed Q4/Q8) snapshot → build both experts packed first, then install
            //     LoRA as forward-time residuals per expert (bases stay packed; LoKr/LoHa rejected up
            //     front, deferred to sc-10050/sc-10051). Removes the old `model.rs:614` hard error.
            // Q4/Q8 (sc-2682) is transformer-only (attn q/k/v/o + ffn.fc1/fc2; T5 above + VAE below stay
            // f32 — the reference's quant scope), two routes:
            //   • pre-quantized snapshot (config.json `quantization` block) → `from_weights` already
            //     built the experts quantized from the on-disk packed weights (`self.quant` is None,
            //     resolved in load), so they load at the reduced ~Q4/Q8 size — the low-peak path;
            //   • dense bf16 snapshot + `spec.quantize` → quantize each expert in-memory after the fold.
            // Either way both experts are quantized independently.
            let prequantized = self.config.quantization.is_some();
            if !prequantized {
                self.merge_adapters(&mut low_w, &mut high_w)?;
            }
            let mut low_dit = WanTransformer::from_weights(&low_w, cfg)?;
            let mut high_dit = WanTransformer::from_weights(&high_w, cfg)?;
            if let Some(q) = self.quant {
                low_dit.quantize(q.bits(), None)?;
                high_dit.quantize(q.bits(), None)?;
            }
            if prequantized {
                self.install_adapters_additive(&mut low_dit, &mut high_dit)?;
            }

            // Each expert has its own text_embedding weights, so contexts are embedded per expert.
            let low = Expert {
                transformer: &low_dit,
                ctx_cond: low_dit.embed_text(&context)?,
                ctx_uncond: Some(low_dit.embed_text(&context_null)?),
                guidance: low_gs,
            };
            let high = Expert {
                transformer: &high_dit,
                ctx_cond: high_dit.embed_text(&context)?,
                ctx_uncond: Some(high_dit.embed_text(&context_null)?),
                guidance: high_gs,
            };
            let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32;
            let total = steps as u32;
            // Curated unified solver (epic 7114, sc-7121): the gen-core-only solvers route through
            // `denoise_moe_curated` (the boundary expert swap happens inside its predict closure); the
            // native unipc/euler/dpmpp2m stay on `scheduler.rs` (N1). Works for both T2V (`y = None`)
            // and I2V (`y = Some` channel-concat conditioning).
            if is_wan_curated(req.sampler.as_deref()) {
                denoise_moe_curated(
                    &low,
                    &high,
                    boundary_timestep,
                    req.sampler.as_deref().expect("is_wan_curated ⇒ Some"),
                    cfg.num_train_timesteps,
                    steps,
                    shift,
                    &init_noise,
                    y.as_ref(),
                    seed,
                    &req.cancel,
                    on_progress,
                )?
            } else {
                let mut on_step = |i: usize| {
                    on_progress(Progress::Step {
                        current: i as u32,
                        total,
                    })
                };
                denoise_moe(
                    &low,
                    &high,
                    boundary_timestep,
                    kind,
                    cfg.num_train_timesteps,
                    steps,
                    shift,
                    &init_noise,
                    y.as_ref(),
                    &req.cancel,
                    &mut on_step,
                )?
            }
        };

        // --- Stage 3: z16 VAE decode → RGB8 frames ---
        on_progress(Progress::Decoding);
        // sc-6894 F-009 — memory-**budgeted** (catchable) z16 decode tiling from the decoded output
        // dims (t_lat·4 frames after the non-causal decode), replacing the unbudgeted
        // `TilingConfig::auto` that could pick an over-budget tile and OOM the largest-resident model.
        // `Ok(None)` for small outputs → single-pass; an over-budget decode returns a catchable error
        // here instead of a SIGKILL in the decode. decode_to_frames re-checks `needs_tiling`.
        let out_frames = lat[1] * A14bProviderVae::VAE_TILING.temporal_scale;
        let out_height = lat[2] * A14bProviderVae::VAE_TILING.spatial_scale;
        let out_width = lat[3] * A14bProviderVae::VAE_TILING.spatial_scale;
        let tiling = a14b_decode_tiling(req, out_width, out_height, out_frames)?;
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = A14bProviderVae::from_weights(&w)?;
            decode_to_frames(&vae, &latents, tiling.as_ref(), Some(&req.cancel))?
        };
        let mut images = frames_to_images(&frames_u8)?;
        // Discard the extra leading frames generated for `trim_first_frames`.
        if trim_out > 0 {
            images.drain(0..trim_out.min(images.len()));
        }

        let fps = req.fps.unwrap_or(cfg.sample_fps);
        Ok(GenerationOutput::Video {
            frames: images,
            fps,
            audio: None,
        })
    }
}

mlx_gen::register_generators! {
    pub(crate) const T2V_14B_REGISTRATION = descriptor_t2v_14b => load_t2v_14b
}

// ===========================================================================================
// Wan2.2 I2V-A14B — dual-expert MoE image→video (channel-concat conditioning, in_dim 36)
// ===========================================================================================

/// Public provider id for the channel-concat I2V model: `"wan2_2_i2v_14b"`.
pub const MODEL_ID_I2V_14B: &str = "wan2_2_i2v_14b";

/// Estimated resident transformer bytes for the sc-4986 [`preflight_denoise_memory_guard`]: the
/// on-disk weight file size(s) scaled by the load-time quantization. A dense bf16 snapshot shrinks to
/// ~Q8/Q4 when `spec.quantize` runs at load; a pre-quantized snapshot has `quant == None` and its
/// files are already packed (ratio 1.0). The 14B MoE passes both expert files (both stay resident).
/// A missing file → 0 bytes for that file, so the guard under-counts rather than spuriously firing —
/// the real "snapshot incomplete" error then surfaces at the actual `Weights::from_file` load.
///
/// The two scaling arms are **mutually exclusive**, so the on-disk size is never double-discounted: a
/// load is *either* a dense bf16 snapshot with `spec.quantize` set (the bf16 file size is scaled down
/// by `ratio`) *or* a pre-packed Q4/Q8 snapshot with `quant == None` (the file is already the final
/// packed size, `ratio == 1.0`) — never a packed file scaled by a quant ratio again.
///
/// A path may also be a **shard directory** (the VACE lanes load `transformer/` /`transformer_2/`
/// diffusers shard dirs, sc-12459): its contents are summed under exactly the `Weights::from_dir`
/// selection — non-recursive, `*.safetensors` only, hidden entries (macOS AppleDouble
/// `._model.safetensors` sidecars) skipped via [`is_hidden_file`] — so the sizing matches what the
/// loader will actually map. Each entry is sized through `std::fs::metadata`, which **traverses
/// symlinks**: linked snapshots (the `assemble_wan_vace_*` `link == true` layout, convert.rs — the
/// default for local dev / the gated e2e) and HF-hub caches store every shard as a symlink into
/// `blobs/`, and sizing the link itself (`DirEntry::metadata`, ~dozens of bytes) would make the
/// preflight see none of the 27–54 GB expert weights and wave through requests that then die by
/// SIGKILL mid-load — the original F-008 failure mode. `pub(crate)` so the sc-12459 VACE preflight
/// reuses it.
///
/// [`is_hidden_file`]: mlx_gen::gen_core::weightsmeta::is_hidden_file
/// SC-20686 estimate-plus-reserve admission: exactly what a dense Wan route's generate-time fit gate
/// ([`preflight_denoise_memory_guard`]) prices for `req`, plus the output geometry its decode runs
/// at, resolved from the load spec, the snapshot's `config.json`/file sizes and the request alone --
/// no weights are loaded and MLX is never touched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenoiseFacts {
    pub(crate) resident_bytes: u64,
    /// The dense (pre-packing) bytes of the largest DiT expert a load-time quantization builds
    /// before packing it -- live beside its packs while they materialize; 0 for a packed tier.
    pub(crate) load_transient_bytes: u64,
    pub(crate) tokens: usize,
    pub(crate) dim: usize,
    pub(crate) cfg_batched: bool,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
}

/// [`DenoiseFacts`] of the TI2V-5B and the two A14B routes, mirroring their `generate_impl`.
pub(crate) fn dense_denoise_facts(
    route: &str,
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> Result<DenoiseFacts> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(Error::Msg(format!(
            "{route}: expected a model directory for the admission estimate"
        )));
    };
    let cfg = WanModelConfig::from_model_dir(root)?;
    let quant = resolve_load_time_quant(route, &cfg, spec.quantize)?;
    let adapter_mode = if cfg.quantization.is_some() {
        AdapterResidencyMode::Additive
    } else {
        AdapterResidencyMode::Folded
    };
    let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
    let trim = req.trim_first_frames.unwrap_or(0) as usize;
    let unsized_adapters = || {
        Error::Msg(format!(
            "{route}: cannot size every additive adapter for the estimate"
        ))
    };
    let dense = |file: &str| {
        if quant.is_some() {
            dit_resident_bytes(&[root.join(file)], None)
        } else {
            0
        }
    };
    let (vae, resident_bytes, cfg_batched, load_transient_bytes) = match route {
        MODEL_ID => {
            let adapters = adapter_stack_resident_bytes(&spec.adapters, adapter_mode)
                .ok_or_else(unsized_adapters)?;
            let guidance = cfg.sample_guide_scale.resolve_single(req.guidance);
            (
                Ti2vProviderVae::VAE_TILING,
                dit_resident_bytes(&[root.join("model.safetensors")], quant)
                    .saturating_add(adapters),
                guidance > 1.0,
                dense("model.safetensors"),
            )
        }
        MODEL_ID_T2V_14B | MODEL_ID_I2V_14B => {
            let low = dit_resident_bytes(&[root.join("low_noise_model.safetensors")], quant);
            let high = dit_resident_bytes(&[root.join("high_noise_model.safetensors")], quant);
            let (low_adapters, high_adapters) =
                wan14b_adapter_bytes_per_expert(&spec.adapters, adapter_mode)
                    .ok_or_else(unsized_adapters)?;
            (
                A14bProviderVae::VAE_TILING,
                wan14b_denoise_resident_bytes(
                    spec.offload_policy,
                    req.sampler.as_deref(),
                    low.saturating_add(low_adapters),
                    high.saturating_add(high_adapters),
                ),
                true,
                dense("low_noise_model.safetensors").max(dense("high_noise_model.safetensors")),
            )
        }
        other => {
            return Err(Error::Msg(format!(
                "{other}: not a dense Wan route for the admission estimate"
            )))
        }
    };
    let stride = provider_vae_stride(vae);
    let gen_frames = frames + trim * stride.0;
    let (width, height) = resolve_capped_dims(req, &cfg, vae);
    let latent = latent_shape(gen_frames, height, width, cfg.vae_z_dim, stride)?;
    Ok(DenoiseFacts {
        resident_bytes,
        load_transient_bytes,
        tokens: seq_len(latent, cfg.patch_size),
        dim: cfg.dim,
        cfg_batched,
        width,
        height,
        frames: u32::try_from(gen_frames)
            .map_err(|_| Error::Msg(format!("{route}: frame count overflows")))?,
    })
}

pub(crate) fn dit_resident_bytes(files: &[PathBuf], quant: Option<Quant>) -> u64 {
    fn weight_bytes_at(p: &std::path::Path) -> u64 {
        match std::fs::metadata(p) {
            Ok(m) if m.is_dir() => std::fs::read_dir(p)
                .map(|rd| {
                    rd.flatten()
                        .map(|e| e.path())
                        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
                        .filter(|p| !mlx_gen::gen_core::weightsmeta::is_hidden_file(p))
                        // `std::fs::metadata`, NOT `DirEntry::metadata`: the latter does not
                        // traverse symlinks, and linked snapshots / HF-hub caches store shards
                        // as symlinks into blobs/ — the link's own length is ~14 bytes.
                        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
                        .sum()
                })
                .unwrap_or(0),
            Ok(m) => m.len(),
            Err(_) => 0,
        }
    }
    let raw: u64 = files.iter().map(|p| weight_bytes_at(p)).sum();
    (raw as f64 * quant_resident_ratio(quant)) as u64
}

/// The resident fraction of a bf16 weight surface after load-time quantization.
pub(crate) fn quant_resident_ratio(quant: Option<Quant>) -> f64 {
    match quant.map(|q| q.bits()) {
        Some(4) => 0.30, // 4-bit affine: ~0.5 B/param + scales vs bf16 2 B/param
        Some(8) => 0.55, // 8-bit affine: ~1 B/param + scales
        _ => 1.0,
    }
}

/// The resident transformer bytes the sc-4986 [`preflight_denoise_memory_guard`] must budget for the
/// **dual-expert MoE denoise**, made policy + sampler aware (sc-12736). `low_bytes` / `high_bytes` are the two
/// experts' resident sizes ([`dit_resident_bytes`] per expert).
///
/// * **Expert swap** — `Sequential` with a native solver keeps only the ACTIVE expert resident, so
///   budget ONE expert (the larger of the two, conservatively). Without this the guard sums both and
///   **false-rejects** the swappable default job on exactly the one-expert-sized machines the worker
///   selects `Sequential` for — defeating the whole Pillar-1 win at the gate.
/// * **Both resident** — `Resident`, or a curated solver on callers without per-eval swapping, keeps
///   BOTH resident. Summing both makes the guard correctly
///   **reject** a both-resident job that won't fit — a catchable error before the load, not a SIGKILL
///   mid-denoise.
pub(crate) fn moe_denoise_resident_bytes(
    offload: OffloadPolicy,
    sampler: Option<&str>,
    low_bytes: u64,
    high_bytes: u64,
) -> u64 {
    let swaps = offload == OffloadPolicy::Sequential && !is_wan_curated(sampler);
    if swaps {
        // One resident expert (the swap). Take the larger so the guard never under-budgets.
        low_bytes.max(high_bytes)
    } else {
        // Both experts co-resident for the whole denoise.
        low_bytes + high_bytes
    }
}

/// Wan T2V/I2V A14B residency after sc-12795: both native and curated paths swap under
/// `Sequential`, while `Resident` retains both experts. VACE continues using
/// [`moe_denoise_resident_bytes`] because its curated path has not adopted per-eval swapping.
fn wan14b_denoise_resident_bytes(
    offload: OffloadPolicy,
    _sampler: Option<&str>,
    low_bytes: u64,
    high_bytes: u64,
) -> u64 {
    if offload == OffloadPolicy::Sequential {
        low_bytes.max(high_bytes)
    } else {
        low_bytes.saturating_add(high_bytes)
    }
}

/// Load-exact adapter residency for each Wan A14B expert. Packed Q4/Q8 snapshots retain additive
/// residuals; dense snapshots fold factors and therefore add no independent bytes. A shared adapter
/// is installed once on each expert, while an expert-tagged adapter contributes only to its target.
/// `None` keeps the preflight fail-closed when any additive source cannot be sized.
fn wan14b_adapter_bytes_per_expert(
    adapters: &[AdapterSpec],
    mode: AdapterResidencyMode,
) -> Option<(u64, u64)> {
    if adapters.is_empty() || mode == AdapterResidencyMode::Folded {
        return Some((0, 0));
    }
    adapters
        .iter()
        .try_fold((0_u64, 0_u64), |(low, high), adapter| {
            let bytes = mlx_gen::gen_core::weightsmeta::safetensors_path_bytes(&adapter.path);
            if bytes == 0 {
                return None;
            }
            Some(match adapter.moe_expert {
                None => (low.saturating_add(bytes), high.saturating_add(bytes)),
                Some(MoeExpert::Low) => (low.saturating_add(bytes), high),
                Some(MoeExpert::High) => (low, high.saturating_add(bytes)),
            })
        })
}

/// The single conditioning reference image for I2V (the first video frame) **and its resolved
/// conditioning strength**, if present.
///
/// sc-19571: this used to return the image alone, which is how the TI2V mask-blend came to build a
/// hard 0/1 pin no matter what the caller asked for. The per-`Reference` `strength` wins over the
/// request-level img2img `strength`, falling back to `1.0` (a full pin) — the identical resolution
/// candle's `candle_gen_wan` TI2V path performs, so the two lanes read one number the same way.
fn i2v_reference(req: &GenerationRequest) -> Option<(&Image, f32)> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, strength } => {
            Some((image, strength.or(req.strength).unwrap_or(1.0)))
        }
        _ => None,
    })
}

/// **The TI2V mask-blend seam** (sc-19571): resolve the request's image conditioning into the
/// `(latent_frame, strength)` pins that drive [`build_ti2v_mask`], and nothing else.
///
/// * `Keyframe`s (first_last_frame / multi-keyframe) → one pin each, **in request order**, so the
///   caller can zip the pins against `req.keyframes()` to pair each resolved index with its image.
///   `frame_idx` is a latent index with negative-from-end resolution (`-1` = last latent frame).
/// * otherwise a single `Reference` image → one pin at latent frame 0 with the strength
///   [`i2v_reference`] resolves.
/// * no image conditioning → no pins, i.e. pure-noise T2V.
///
/// Pure and total (no GPU, no weights), because the whole defect this closes was a strength that
/// was carried as far as the provider and then dropped on the way into the mask — which is exactly
/// the hop a unit test can only pin if it is expressible without a render.
fn resolve_ti2v_pins(req: &GenerationRequest, t_lat: usize) -> Result<Vec<(usize, f32)>> {
    let keyframes = req.keyframes();
    if keyframes.is_empty() {
        return Ok(i2v_reference(req)
            .map(|(_, strength)| vec![(0usize, strength)])
            .unwrap_or_default());
    }
    let mut pins = Vec::with_capacity(keyframes.len());
    for kf in &keyframes {
        let idx = if kf.frame_idx < 0 {
            t_lat as i32 + kf.frame_idx
        } else {
            kf.frame_idx
        };
        if idx < 0 || idx as usize >= t_lat {
            return Err(Error::Msg(format!(
                "wan2_2_ti2v_5b: keyframe latent frame index {} out of bounds for {t_lat} latent \
                 frames",
                kf.frame_idx
            )));
        }
        pins.push((idx as usize, kf.strength));
    }
    Ok(pins)
}

/// Range-gate every conditioning strength a Wan request can carry (sc-19571, mirroring candle's
/// `check_strength` and mlx-gen-ltx's F-054 gate).
///
/// `strength > 1` yields a **negative** denoise mask (`1 − strength`) → negative per-token timesteps
/// and extrapolating blends: silent garbage rather than a louder pin. `strength < 0` over-weights
/// the noise the same way. Now that the number actually reaches the mask, the range has to be a
/// request-boundary rejection instead of a value nobody read.
fn reject_out_of_range_strengths(id: &str, req: &GenerationRequest) -> Result<()> {
    let check = |label: &str, strength: f32| -> Result<()> {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(Error::Msg(format!(
                "{id}: {label} strength must be finite and in [0, 1] (got {strength})"
            )));
        }
        Ok(())
    };
    if let Some(strength) = req.strength {
        check("img2img", strength)?;
    }
    for c in &req.conditioning {
        match c {
            Conditioning::Reference {
                strength: Some(strength),
                ..
            } => check("Reference", *strength)?,
            Conditioning::Keyframe { strength, .. } => check("Keyframe", *strength)?,
            _ => {}
        }
    }
    Ok(())
}

/// Resolve the output `(width, height)` for a **dense** Wan path (5B TI2V, A14B): round the requested
/// dims down to the `patch · vae_stride` grid. Byte-identical block shared by the two dense
/// `generate_impl` bodies (F-010); the VACE paths align the same way inline and don't use this.
///
/// **sc-12308:** this used to also refit an over-`max_area` request via `best_output_size`, silently
/// handing back a geometry the caller never asked for. The cap is now enforced in `validate_impl` by
/// rejecting instead, so by the time this runs the area is already within budget — and since the
/// alignment here only ever rounds *down*, it cannot push a validated request back over.
///
/// **sc-12607:** `validate_impl` now *also* rejects an off-grid `width`/`height` (via
/// [`reject_off_grid`], matching candle), so a validated request is already grid-aligned and this
/// rounding is a defensive identity — it never silently snaps a geometry the caller chose.
fn resolve_capped_dims(
    req: &GenerationRequest,
    cfg: &WanModelConfig,
    vae: VaeTiling,
) -> (u32, u32) {
    let stride = provider_vae_stride(vae);
    let width = align_dim(req.width, cfg.patch_size.2, stride.2);
    let height = align_dim(req.height, cfg.patch_size.1, stride.1);
    (width, height)
}

/// The model's `(dw, dh)` pixel grid = `patch · vae_stride` — the lattice every Wan geometry sits on.
fn grid(cfg: &WanModelConfig, vae: VaeTiling) -> (u32, u32) {
    let stride = provider_vae_stride(vae);
    (
        (cfg.patch_size.2 * stride.2) as u32,
        (cfg.patch_size.1 * stride.1) as u32,
    )
}

/// Stable identity + advertised capabilities for the Wan2.2 I2V-A14B (dual-expert MoE image→video).
/// Identical to the T2V-A14B but advertises a single `Reference` conditioning image (the channel-
/// concat first frame) and the (3.5, 3.5) per-expert guidance.
pub fn descriptor_i2v_14b() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID_I2V_14B,
        family: "wan",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            // A single image is channel-concatenated as the first-frame conditioning (in_dim 36).
            conditioning: vec![ConditioningKind::Reference],
            // LoRA + LoKr merge per-expert at generate time (sc-2683 / sc-2393, PEFT/kohya + LoKr,
            // MoE high/low); Q4/Q8 (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot.
            supports_lora: true,
            supports_lokr: true,
            samplers: wan_samplers(),
            // H/W align to patch×vae_stride = 16 (z16 VAE, spatial stride 8); long edge cap 1280.
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: true,
            // A14B honors `OffloadPolicy::Sequential` (epic 12732, sc-12736): the staged expert swap
            // holds only the ACTIVE MoE expert resident (never both) and frees the UMT5 TE / VAE
            // off-GPU during denoise, dropping the unified-memory peak to ~one expert. Advertised so
            // the worker's fit-gate can tell "bounds peak here" from a no-op fallback.
            supports_sequential_offload: true,
            // TE/VAE and the active expert are phase-staged even under Resident; Sequential adds the
            // stronger cache-flush/expert-residency controls described above.
            unconditionally_engages_staged_residency: true,
            ..Default::default()
        },
    }
}

/// Load the Wan2.2 I2V-A14B from a converted MLX snapshot directory (same layout as the T2V-A14B:
/// `low_noise_model` + `high_noise_model` + `t5_encoder` + `vae` (with encoder) + `tokenizer.json` +
/// `config.json`). Requires `model_type == "i2v"` (in_dim 36) and a dual-expert checkpoint. LoRA
/// adapters merge per-expert at generate time (sc-2683); Q4/Q8 (sc-2682) loads via `spec.quantize`
/// or a pre-quantized snapshot. LoKr is the sibling sc-2393.
pub fn load_i2v_14b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_i2v_14b: expected a model directory (converted MLX snapshot), not a single \
                 file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_i2v_14b: precision override is not wired (the experts run bf16 GEMMs over an \
             f32 residual stream — the parity regime)"
                .into(),
        ));
    }

    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.is_i2v_concat() {
        return Err(Error::Msg(format!(
            "wan2_2_i2v_14b: config.json is not a channel-concat I2V model (model_type={}, \
             in_dim={}); expected the converted Wan2.2 I2V-A14B checkpoint (model_type=i2v, \
             in_dim=36)",
            config.model_type, config.in_dim
        )));
    }
    if !config.dual_model {
        return Err(Error::Msg(
            "wan2_2_i2v_14b: config.json is not a dual-expert model (dual_model=false); expected \
             the converted Wan2.2 I2V-A14B MoE checkpoint"
                .into(),
        ));
    }
    let quant = resolve_load_time_quant(MODEL_ID_I2V_14B, &config, spec.quantize)?;
    let i2v_memory = if spec.resolved_route.as_deref() == Some(MODEL_ID_I2V_14B)
        && spec.prepared_file_pins().is_prepared()
    {
        Some(crate::i2v_memory_strategy::prepare(spec, MODEL_ID_I2V_14B).map_err(Error::from)?)
    } else {
        None
    };
    Ok(Box::new(Wan14b {
        descriptor: descriptor_i2v_14b(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
        offload_policy: spec.offload_policy,
        i2v_memory,
    }))
}

mlx_gen::register_generators! {
    pub(crate) const I2V_14B_REGISTRATION = descriptor_i2v_14b => load_i2v_14b
}

#[cfg(test)]
mod tests;
