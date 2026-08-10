//! `KreaTurboControl` — the Krea 2 Turbo **pose-ControlNet** variant (sc-8465, epic 8459 S5): strict
//! pose conditioning via a trained control branch overlaid on the frozen Krea 2 Turbo base, registered
//! as its own `Generator` (`krea_2_turbo_control`). The MLX twin of the candle `Krea2Control` provider
//! (candle-gen-krea, sc-8464) — same 8-step CFG-free Turbo sampler, same single-forward residual
//! injection, same `control_scale` semantics.
//!
//! Identical to [`crate::model::Krea`] (Turbo) except a [`Krea2ControlBranch`] rides the DiT and
//! `generate` threads a VAE-encoded pose skeleton through it. [`load`] needs the base snapshot
//! (`spec.weights`, the DENSE `krea/Krea-2-Turbo` diffusers tree — NOT the packed Q4/Q8 turnkey the
//! plain `krea_2_turbo` gen uses, because the branch is a composable-forward overlay trained on the bf16
//! base) **and** the control overlay checkpoint (`spec.control`). Pose-only + dense bf16 (no quant, no
//! negative prompt, no guidance), mirroring the candle `krea_2_turbo_control` engine.

use mlx_gen::gen_core;
use mlx_gen::{
    require_base_snapshot, require_control, AcceptedControlKinds, ConditioningKind, ControlBranch,
    Error, GenerationOutput, GenerationRequest, Generator, LatentDecoder, LoadSpec,
    ModelDescriptor, Precision, Progress, Quant, Residency, Result, WeightsSource,
    BASE_SNAPSHOT_COMPONENT, VAE_COMPONENT,
};
use mlx_gen_wan::OwnedWanSingleFrameDecoder;

use mlx_gen::default_seed;

use mlx_rs::Array;
use std::path::Path;

use crate::config::Krea2Config;
use crate::control::Krea2ControlBranch;
use crate::pipeline::{maybe_apply_style_gain, KreaHeavy, KreaText, TurboOptions};

/// Registry id for the Krea 2 Turbo pose-ControlNet variant. Matches the SceneWorks worker's
/// `STRICT_CONTROL_ENGINES` `krea_2_turbo_control` id and the candle lane, so one id serves both
/// backends (the worker's OS/feature cfg picks the MLX vs candle provider).
pub const KREA_2_TURBO_CONTROL_ID: &str = "krea_2_turbo_control";

/// Turbo default steps (CFG-free distilled few-step), shared with the base [`crate::model`] Turbo path.
const DEFAULT_STEPS: u32 = 8;

/// Krea 2 Turbo pose-control identity + capabilities. Derived from the base Turbo [`crate::model::descriptor`]
/// so the shared surface (family/backend/samplers/size bounds/LoRA/mac_only/**supported_quants**) stays in
/// lockstep; the pose control lane only swaps the Turbo img2img `Reference` surface for a required `Control`
/// conditioning.
///
/// Q4/Q8 base (sc-11727, candle-gen PR #471): the earlier "dense-only" gate assumed a packed base would
/// desync the main-stream activations the residual was trained against. That assumption was refuted — a
/// q8 base is visually indistinguishable from bf16 and q4 keeps the pose-lock fully intact (GPU-proven on
/// the candle twin). mlx-gen's control lane reuses the same packed-capable
/// [`crate::transformer::Krea2Transformer`] as txt2img,
/// so a packed snapshot runs a **true packed forward** (real memory savings), not candle's dequant-to-bf16
/// VRAM overlay — so we inherit the base Turbo `&[Q4, Q8]` rather than blanking it.
pub fn descriptor() -> ModelDescriptor {
    let mut d = crate::model::descriptor();
    d.id = KREA_2_TURBO_CONTROL_ID;
    // Pose ControlNet: a required Control conditioning replaces Turbo's optional img2img Reference.
    d.capabilities.conditioning = vec![ConditioningKind::Control];
    // Pose only — this branch is a single-signal ControlNet, not a Fun-Union.
    d.control_kinds = Some(AcceptedControlKinds::Only(vec![mlx_gen::ControlKind::Pose]));
    d
}

/// A loaded Krea 2 Turbo pose-control generator: the cached descriptor + a component-residency strategy
/// (the base Turbo text phase + DiT/VAE + the pose control branch).
pub struct KreaTurboControl {
    descriptor: ModelDescriptor,
    memory_strategy: gen_core::MemoryProviderContract,
    loaded_precision: Precision,
    loaded_quant: Option<Quant>,
    /// Component-residency strategy (epic 10834 Phase 1, sc-11101; hoisted to the shared seam in
    /// sc-11125), selected from [`LoadSpec::offload_policy`]. `Resident` (default) holds the text phase +
    /// DiT + VAE + branch warm; `Sequential` holds only the per-phase loader closures and re-loads per
    /// generation in phase order (encode → **drop the text phase** → denoise/decode), bounding peak
    /// unified memory to `max(text, DiT+VAE+branch)`. Base weights may be dense bf16 or packed Q4/Q8
    /// (sc-11727); a packed base further lowers the `DiT+VAE+branch` term. The [`Residency`] seam owns the
    /// eval/drop/clear discipline, the stage-boundary cancel checks, and the error-safe cache flush.
    residency: Residency<KreaText, ControlHeavyOwned>,
}

/// The heavy render-phase components for pose control: the single-stream DiT + VAE ([`KreaHeavy`]) plus
/// the pose [`Krea2ControlBranch`] — which is a SECOND heavy component that stays on the heavy side (it
/// rides the DiT), mirroring qwen-image-control's `QwenControlHeavyOwned`. Owned by the `Resident`
/// components or a `Sequential` generate.
struct ControlHeavyOwned {
    heavy: KreaHeavy,
    branch: Krea2ControlBranch,
    alternate_decoder: Option<OwnedWanSingleFrameDecoder>,
    /// The effective base-DiT quant tier (`None` = dense bf16). Captured at load so the render-time
    /// geometry safety check can weigh the resident-weight footprint without re-deriving it from the
    /// (post-quant) modules.
    base_tier: Option<Quant>,
    /// The tier the pose branch was actually packed to at load (sc-15799): it follows the base tier,
    /// with the declared q4 → q8 floor, and is `None` only for a dense branch. Feeds the same geometry
    /// safety check.
    branch_tier: Option<Quant>,
    /// Architecture config, retained for the decode-tiling cost model (block/hidden/FFN widths).
    cfg: Krea2Config,
}

/// A borrow of the pose-control heavy components, so the render loop runs identically whether they are
/// held resident or were just loaded by the `Sequential` path.
struct ControlHeavyRef<'a> {
    heavy: &'a KreaHeavy,
    branch: &'a Krea2ControlBranch,
    alternate_decoder: Option<&'a OwnedWanSingleFrameDecoder>,
}

impl ControlHeavyOwned {
    fn as_ref(&self) -> ControlHeavyRef<'_> {
        ControlHeavyRef {
            heavy: &self.heavy,
            branch: &self.branch,
            alternate_decoder: self.alternate_decoder.as_ref(),
        }
    }
}

fn selected_decode_tiling(
    memory: Option<gen_core::GenerationMemory>,
) -> Result<Option<mlx_gen::tiling::TilingConfig>> {
    match memory {
        Some(memory) if memory.tile_vae_decode => Ok(Some(
            crate::memory::requested_control_decode_tiling(memory)?,
        )),
        _ => Ok(None),
    }
}

/// The exact output geometry requested for a pose-control render. Keeping this as a production seam
/// prevents the feasibility gate, control preprocessing, and render options from independently
/// resolving (or silently rewriting) width and height.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RequestedControlGeometry {
    width: u32,
    height: u32,
}

impl RequestedControlGeometry {
    fn from_request(req: &GenerationRequest) -> Self {
        Self {
            width: req.width,
            height: req.height,
        }
    }

    /// Dimensions supplied to the feasibility model. Render-facing access is available only after
    /// [`Self::require_feasible`] succeeds.
    fn feasibility_dimensions(self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn require_feasible(self, feasible: bool) -> Result<AdmittedControlGeometry> {
        crate::memory::require_control_geometry(self.width, self.height, feasible)?;
        Ok(AdmittedControlGeometry(self))
    }

    /// Run control preprocessing only after this exact requested geometry passes the feasibility
    /// gate. The callback seam keeps the ordering and dimension propagation weight-free-testable.
    fn prepare_after_feasibility<T>(
        self,
        feasible: bool,
        prepare: impl FnOnce(u32, u32) -> Result<T>,
    ) -> Result<(AdmittedControlGeometry, T)> {
        let admitted = self.require_feasible(feasible)?;
        let (width, height) = admitted.dimensions();
        let prepared = prepare(width, height)?;
        Ok((admitted, prepared))
    }
}

/// Requested geometry after the memory gate has admitted it. Both control preprocessing and the
/// render options consume this value, so neither can diverge from the dimensions that were checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdmittedControlGeometry(RequestedControlGeometry);

impl AdmittedControlGeometry {
    fn dimensions(self) -> (u32, u32) {
        (self.0.width, self.0.height)
    }

    fn turbo_options(self, req: &GenerationRequest, steps: usize, seed: u64) -> TurboOptions {
        TurboOptions {
            width: self.0.width,
            height: self.0.height,
            steps,
            seed,
            sampler: req.sampler.clone(),
            scheduler: req.scheduler.clone(),
            transformer_window_size: req
                .memory
                .and_then(|memory| memory.transformer_window_size)
                .map(|window| window as usize),
            memory: req.memory.unwrap_or_default(),
        }
    }
}

/// Construct a [`KreaTurboControl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`mlx_gen::WeightsSource::Dir`] `krea/Krea-2-Turbo` snapshot (`transformer/
/// text_encoder/ vae/ tokenizer/`) — a dense bf16 tier OR a pre-packed Q4/Q8 turnkey — and `spec.control`
/// (required) the converted MLX pose overlay (`control_step5000` — a single `.safetensors` `File`, or a
/// `Dir` of shards). Raw-trained LoRA/LoKr in `spec.adapters` install onto the base DiT (the branch is
/// never an adapter target). Quantization is supported (sc-11727): a pre-packed snapshot runs a true
/// packed forward directly, and a dense snapshot + `spec.quantize` quantizes the base at load — either way
/// only the base DiT/TE pack; the pose overlay itself stays the bf16 it was trained as.
///
/// Component residency (epic 10834 Phase 1, sc-11101; hoisted to the shared [`Residency::from_policy`]
/// seam in sc-11126, F-180): `Resident` (default) builds every component now and holds it warm;
/// `Sequential` keeps only the spec and re-loads per generate in phase order (encode → drop the text
/// phase → denoise/decode). Both use the same per-phase loaders, so the components are byte-identical.
/// Validity (base dir, control present, no quant, bf16) is checked up front either way.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // Fail fast — validate the whole spec up front for BOTH residencies (mirrors the pre-sc-11101 load
    // order): dense bf16, a base snapshot dir, the required control overlay, and no quant override.
    validate_control_spec(spec)?;
    if matches!(spec.weights, WeightsSource::File(_)) {
        return Ok(Box::new(build_native_krea_control_from_spec(spec)?));
    }
    let loaded_quant = crate::model::effective_base_quant_tier(spec, KREA_2_TURBO_CONTROL_ID)?;
    Ok(Box::new(KreaTurboControl {
        descriptor: descriptor(),
        memory_strategy: crate::memory_strategy::memory_strategy_contract(
            KREA_2_TURBO_CONTROL_ID,
            spec,
        )?,
        loaded_precision: spec.precision,
        loaded_quant,
        residency: build_control_residency(spec)?,
    }))
}

/// Build a Krea 2 Turbo **pose-control** generator around a **community single-file** DiT checkpoint
/// (the control sibling of [`crate::model::load_from_native_dit_file`], which serves the imported
/// t2i/img2img/edit surfaces): `dit_file` is a ComfyUI-exported Krea 2 DiT (native-mmdit keys, dense
/// bf16 or the descriptor-validated plain int8-per-row format), `base_snapshot_dir` a resident
/// turnkey snapshot supplying the shared text encoder / VAE / tokenizer / architecture config, and
/// `control` the trained pose control-branch overlay (`control_step5000`-style `.safetensors`). The
/// branch is architecturally independent of the DiT's weights — a `control_scale`-scaled RMS-clamped
/// residual folded onto the frozen base at load — so a same-shape imported fine-tune composes with it
/// exactly as the builtin Turbo DiT does.
///
/// Assembly: the DiT loads through the same fail-closed native path as the t2i entrypoint
/// ([`crate::loader::load_transformer_from_native_file`] — key-remap, coverage/shape validation
/// against the base tier's architecture config), `adapters` (Raw-trained LoRA/LoKr) install onto the
/// imported DiT before residency is finalized (the branch is never an adapter target), and the pose
/// branch loads from `control` exactly as the snapshot control load does. Branch tier follows the
/// base-DiT tier (sc-15799): the native loader dequantizes I8 files to bf16, so the in-memory base is
/// always DENSE — `control_branch_quant_bits(None)` keeps the branch dense, matching the snapshot
/// lane's dense-base behavior.
///
/// The memory contract keeps the `krea_2_turbo_control` provider identity and calibration fingerprint
/// so same-provider evidence resolves through the same selectors; the pinned File source can also
/// participate in the provider's staged lifecycle.
pub fn load_control_from_native_dit_file(
    dit_file: impl AsRef<Path>,
    base_snapshot_dir: impl AsRef<Path>,
    control: impl AsRef<Path>,
    adapters: &[mlx_gen::AdapterSpec],
) -> Result<Box<dyn Generator>> {
    let spec = LoadSpec::new(WeightsSource::File(dit_file.as_ref().to_path_buf()))
        .with_component(
            BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(base_snapshot_dir.as_ref().to_path_buf()),
        )
        .with_control(WeightsSource::File(control.as_ref().to_path_buf()))
        .with_adapters(adapters.to_vec());
    load(&spec)
}

/// The concrete-[`KreaTurboControl`] assembly behind [`load_control_from_native_dit_file`] (which
/// boxes the result) — the control twin of [`crate::model::build_native_krea`], returning the
/// concrete type so in-crate harnesses can exercise the assembly directly.
#[cfg(test)]
pub(crate) fn build_native_krea_control(
    dit_file: impl AsRef<Path>,
    base_snapshot_dir: impl AsRef<Path>,
    control: impl AsRef<Path>,
    adapters: &[mlx_gen::AdapterSpec],
) -> Result<KreaTurboControl> {
    let spec = LoadSpec::new(WeightsSource::File(dit_file.as_ref().to_path_buf()))
        .with_component(
            BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(base_snapshot_dir.as_ref().to_path_buf()),
        )
        .with_control(WeightsSource::File(control.as_ref().to_path_buf()))
        .with_adapters(adapters.to_vec());
    build_native_krea_control_from_spec(&spec)
}

fn build_native_krea_control_from_spec(spec: &LoadSpec) -> Result<KreaTurboControl> {
    validate_control_spec(spec)?;
    let WeightsSource::File(dit_file) = &spec.weights else {
        return Err(Error::Msg(format!(
            "{KREA_2_TURBO_CONTROL_ID}: imported control assembly requires a single-file DiT"
        )));
    };
    let base = require_base_snapshot(spec, KREA_2_TURBO_CONTROL_ID)?.to_path_buf();
    let streamable = matches!(spec.offload_policy, mlx_gen::OffloadPolicy::Sequential)
        && matches!(
            spec.load_shape,
            mlx_gen::gen_core::LoadShape::DeferredMaterialization
        )
        && !crate::model::adapters_have_diff_patch(&spec.adapters);
    let memory_strategy = crate::memory_strategy::native_memory_strategy_contract_from_spec(
        KREA_2_TURBO_CONTROL_ID,
        spec,
        &base,
        streamable,
    )?;
    let text_base = base.clone();
    let heavy_base = base;
    let heavy_spec = spec.clone();
    let heavy_dit = dit_file.clone();
    let residency = Residency::from_policy(
        spec.offload_policy,
        move || KreaText::from_snapshot(&text_base),
        move |_use_pid| load_native_control_heavy(&heavy_spec, &heavy_base, &heavy_dit, streamable),
    )?;
    Ok(KreaTurboControl {
        descriptor: descriptor(),
        memory_strategy,
        loaded_precision: spec.precision,
        loaded_quant: None,
        residency,
    })
}

fn load_native_control_heavy(
    spec: &LoadSpec,
    base: &Path,
    dit_file: &Path,
    streamable: bool,
) -> Result<ControlHeavyOwned> {
    let cfg = Krea2Config::from_snapshot(base)?;
    let dit =
        crate::loader::load_transformer_from_native_file_with_stream(dit_file, &cfg, streamable)?;
    let vae = crate::vae::load_vae(base)?;
    let mut heavy = KreaHeavy::from_parts(dit, vae);
    if !spec.adapters.is_empty() {
        heavy.apply_adapters(&spec.adapters)?;
    }
    let control = require_control(spec, KREA_2_TURBO_CONTROL_ID, "Krea 2 pose control overlay")?;
    let branch = Krea2ControlBranch::from_source(control, &cfg)?;
    let alternate_decoder = mlx_gen_wan::load_selected_single_frame_decoder(spec, &descriptor())?;
    Ok(ControlHeavyOwned {
        heavy,
        branch,
        alternate_decoder,
        base_tier: None,
        branch_tier: None,
        cfg,
    })
}

/// The up-front spec validation shared by [`load`] and [`build_control_residency`] (fail-fast for BOTH
/// residencies): bf16 activation precision, a base snapshot dir, and the required pose overlay. Quant is
/// allowed (sc-11727) — a pre-packed snapshot or a `spec.quantize` request both pack only the base DiT/TE,
/// leaving activations bf16 — so there is no quant rejection here.
fn validate_control_spec(spec: &LoadSpec) -> Result<()> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{KREA_2_TURBO_CONTROL_ID}: only the default bf16 activation precision is wired (drop the \
             precision override); Q4/Q8 weight packing is orthogonal and IS supported"
        )));
    }
    mlx_gen::gen_core::reject_unknown_components(
        spec,
        &[BASE_SNAPSHOT_COMPONENT, VAE_COMPONENT],
        KREA_2_TURBO_CONTROL_ID,
    )?;
    mlx_gen_wan::validate_selected_single_frame_decoder(spec, &descriptor())?;
    let _ = require_base_snapshot(spec, KREA_2_TURBO_CONTROL_ID)?;
    let _ = require_control(spec, KREA_2_TURBO_CONTROL_ID, "Krea 2 pose control overlay")?;
    if matches!(spec.weights, WeightsSource::File(_)) {
        if spec.quantize.is_some() {
            return Err(Error::Unsupported(format!(
                "{KREA_2_TURBO_CONTROL_ID}: imported DiT carries its numeric format; quantize is unsupported"
            )));
        }
        if spec.ip_adapter.is_some() || !spec.extra_controls.is_empty() || spec.pid.is_some() {
            return Err(Error::Unsupported(format!(
                "{KREA_2_TURBO_CONTROL_ID}: imported pose control does not accept IP-adapter, extra-control, or PiD overlays"
            )));
        }
    }
    Ok(())
}

/// The policy→[`Residency`] dispatch, routed through the single [`Residency::from_policy`] seam
/// (sc-11101; hoisted to the shared seam in sc-11126, F-180) so the `match offload_policy` lives in one
/// place rather than a bespoke per-crate copy. `Resident` eager-loads the text phase + heavy bundle now;
/// `Sequential` captures the two per-phase loaders and loads nothing now, deferring each to
/// [`Residency::run`]. Both go through the same [`crate::model::load_krea_text`] / [`load_control_heavy`],
/// so the `Resident` composition is byte-identical to the pre-seam one. Quant packs only the base DiT/TE
/// (sc-11727) and there is no re-quant across residencies (both share the load→adapt→quantize order), so
/// no F-181 concern; the pose branch carries no PiD overlay, so the seam's `use_pid` arg is unused. The
/// deferral is weight-free-testable: under
/// `Sequential` this touches no component weights, so a dispatch that mapped `Sequential → Resident`
/// (ignoring `offload_policy`) would eager-load here and fail the "Sequential defers" unit test.
fn build_control_residency(spec: &LoadSpec) -> Result<Residency<KreaText, ControlHeavyOwned>> {
    // Up-front fail-fast for both policies (precision + base dir + control present), so a direct call
    // (e.g. the F-180 unit test) rejects an invalid spec exactly as `load` does.
    validate_control_spec(spec)?;
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || {
            let root = require_base_snapshot(&spec_text, KREA_2_TURBO_CONTROL_ID)?;
            // Reuse the txt2img text loader so the Qwen3-VL TE packs identically (sc-11727): pre-packed
            // auto-detects, dense + `spec.quantize` quantizes at load, activations stay bf16.
            crate::model::load_krea_text(&spec_text, root, KREA_2_TURBO_CONTROL_ID)
        },
        move |_use_pid| {
            let root = require_base_snapshot(&spec_heavy, KREA_2_TURBO_CONTROL_ID)?;
            load_control_heavy(&spec_heavy, root)
        },
    )
}

/// Load the pose-control heavy phase — the base DiT + VAE ([`KreaHeavy`]), Raw-trained LoRA/LoKr on the
/// base DiT (the branch is never an adapter target), the optional base-DiT quantize, then the pose branch
/// (`spec.control`). Factored so `Sequential` loads these AFTER the text phase is dropped. `spec.control`
/// was already validated present by [`load`]; re-requiring it here keeps the `Sequential` per-generate
/// path self-contained.
///
/// Quant order mirrors the txt2img lane (sc-11727): install adapters, THEN quantize, so the residual
/// stacks over the (possibly already-packed) base. A pre-packed snapshot needs no `spec.quantize` — the
/// auto-detecting loader already built packed Linears in [`KreaHeavy::from_snapshot`], and
/// [`crate::model::load_time_quant_bits`] returns `None` (or errors on a bit mismatch).
///
/// The pose branch loads bf16 from its overlay (it carries no `.scales`), then is packed to the tier
/// **tier integrity** assigns it (sc-15799): [`crate::memory::control_branch_quant_bits`] of the base
/// width — q8 base ⇒ q8 branch, q4 base ⇒ q8 branch (the declared, measured floor), dense base ⇒ dense
/// branch. It is still a LOAD-TIME decision (a resident weight can't be re-packed mid-render) but it is
/// no longer a budget-gated *lever*: the sc-11748 gate kept a bf16 branch on any machine with headroom,
/// which left **~3.3 GB** of precision the user did not ask for resident on a q8 render — the branch's
/// projections are 3.30 B params ≈ 6.6 GB bf16 against ~3.3 GB packed at q8
/// ([`crate::memory`] / `gen_core::tier_integrity`). (NOT the 8.4 GB the catalog's `branchPackSaveGb`
/// once claimed: 8.4 exceeds the whole branch, so it was never a weight-side quantity; the key is
/// retracted and sc-16013 owns the re-measure.) `control_scale == 0` stays bit-exact to the base at any
/// tier.
fn load_control_heavy(spec: &LoadSpec, root: &Path) -> Result<ControlHeavyOwned> {
    let plan = crate::model::resolve_load_plan(spec, root, KREA_2_TURBO_CONTROL_ID)?;
    let streamable_transformer = matches!(spec.offload_policy, mlx_gen::OffloadPolicy::Sequential)
        && matches!(
            spec.load_shape,
            mlx_gen::gen_core::LoadShape::DeferredMaterialization
        )
        && !crate::model::adapters_have_diff_patch(&spec.adapters)
        && plan.load_time_quant_bits.is_none();
    let mut heavy = KreaHeavy::from_snapshot_with_stream(root, streamable_transformer)?;
    if !spec.adapters.is_empty() {
        heavy.apply_adapters(&spec.adapters)?;
    }
    if let Some(bits) = crate::model::load_time_quant_bits(spec, root, KREA_2_TURBO_CONTROL_ID)? {
        heavy.quantize(bits)?;
    }
    let control = require_control(spec, KREA_2_TURBO_CONTROL_ID, "Krea 2 pose control overlay")?;
    let cfg = Krea2Config::from_snapshot(root)?;
    let mut branch = Krea2ControlBranch::from_source(control, &cfg)?;
    // sc-15799 — tier integrity. The pose overlay always loads bf16 (it ships no `.scales`); pack it to
    // the tier the base tier implies, whether the base was packed at load or arrived pre-packed. No
    // device-budget reading: the branch's tier is a consequence of the user's tier choice, not a rung.
    let base_bits = crate::model::effective_base_quant_bits(spec, root, KREA_2_TURBO_CONTROL_ID)?;
    let base_tier = tier_from_bits(base_bits);
    let branch_bits = crate::memory::control_branch_quant_bits(base_bits);
    if let Some(bits) = branch_bits {
        branch.quantize(bits)?;
    }
    let branch_tier = tier_from_bits(branch_bits);
    let alternate_decoder = mlx_gen_wan::load_selected_single_frame_decoder(spec, &descriptor())?;
    Ok(ControlHeavyOwned {
        heavy,
        branch,
        alternate_decoder,
        base_tier,
        branch_tier,
        cfg,
    })
}

/// Map an effective quant width (`4`/`8`) to the [`Quant`] tier the decode-tiling cost model weighs; any
/// other width (or dense bf16) has no tier. Delegates to `crate::memory::tier_from_bits` rather than
/// mirroring it: the base and branch tiers must be read off ONE mapping now that the branch's tier is
/// derived from the base's (sc-15799).
fn tier_from_bits(bits: Option<i32>) -> Option<Quant> {
    bits.and_then(crate::memory::tier_from_bits)
}

/// The pose branch is pose-only (`Only([Pose])`) and defaults an unset `control_scale` to the S0 mid
/// value (0.6). All the resolve/validate-present boilerplate comes from the shared trait (sc-8241).
impl ControlBranch for KreaTurboControl {
    fn model_id(&self) -> &'static str {
        KREA_2_TURBO_CONTROL_ID
    }

    fn default_control_scale(&self) -> f32 {
        crate::control::DEFAULT_CONTROL_SCALE
    }
}

impl KreaTurboControl {
    fn calibration_fault(
        &self,
        req: &GenerationRequest,
        phase: gen_core::MemoryPhase,
    ) -> Result<()> {
        match req.memory {
            Some(memory) if memory.calibration_error_phase == Some(phase) => {
                Err(Error::Msg(format!(
                    "{KREA_2_TURBO_CONTROL_ID}: injected memory-strategy calibration error at {phase:?}"
                )))
            }
            _ => Ok(()),
        }
    }

    /// The rich-`Result` body behind [`Generator::generate`] (the crate's own [`mlx_gen::Error`] so `?`
    /// lifts `mlx_rs` device exceptions; the trait wrapper bridges into [`gen_core::Error`]). Renders
    /// `req.count` CFG-free Turbo images, one per pose per seed (`seed + n`), each pose-locked by the
    /// single-forward residual injection — through the residency (encode → drop text phase under
    /// `Sequential` → load heavy+branch → per-image render → free heavy).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        // The required pose skeleton + its resolved scale (unset → the 0.6 default; Some(0.0) inert).
        let (control_image, control_scale) = self.resolve_control(req)?;

        // Under `Resident` the phase-A Qwen3-VL encoder stays warm through the decode, so its footprint
        // co-resides in the decode peak; under `Sequential` it was dropped before the heavy phase. The
        // selected-shape geometry check needs to know which, so capture it here (before the
        // `residency.run` borrow) to avoid capturing `self` in the heavy closure below.
        let text_co_resident = !self.residency.is_sequential();

        // Phase A: prompt → context (sc-11101; sc-11125). Pose control is CFG-free (one context, no
        // negative), so the optional "text style" tap-reweight gain (sc-12009) applies to this single
        // conditional context — the same tap structure the txt2img/img2img encode carries (`None`/g≈1 is
        // a no-op). Under `Sequential` the shared seam loads the text phase, encodes, materializes, then
        // DROPS it + `clear_cache()` before the DiT/VAE/branch load below.
        self.residency.run(
            &req.cancel,
            req.use_pid,
            on_progress,
            |text: &KreaText| {
                let context =
                    maybe_apply_style_gain(text.encode(&req.prompt)?, req.text_style_gain)?;
                // `Residency::run` materializes only under Sequential. For calibration fault
                // injection, evaluate at the conditioning boundary here as well so Resident and
                // Sequential exercise the same physical phase without changing ordinary renders.
                if req.memory.is_some_and(|memory| {
                    memory.calibration_error_phase == Some(gen_core::MemoryPhase::Conditioning)
                }) {
                    mlx_rs::transforms::eval([&context])?;
                    self.calibration_fault(req, gen_core::MemoryPhase::Conditioning)?;
                }
                Ok(context)
            },
            // Materialize the context while the text phase is still alive (Sequential only).
            |ctx: Option<&Array>| {
                let Some(ctx) = ctx else { return Ok(()) };
                mlx_rs::transforms::eval([ctx])?;
                Ok(())
            },
            // Phase B: heavy render components (DiT + VAE + the pose branch). The render loop below runs
            // identically for both residencies.
            |heavy_owned, context, on_progress| {
                self.calibration_fault(req, gen_core::MemoryPhase::Denoise)?;
                let heavy = heavy_owned.as_ref();
                let safe_gib = mlx_gen::memory::safe_budget_gib();

                // Apply only the bounded-decode parameters selected upstream from promoted evidence.
                // Resident and staged-residency requests remain single-pass: this provider must not run
                // a second live-budget selector and silently upgrade either strategy to bounded decode.
                let decode_tiling = selected_decode_tiling(req.memory)?;
                let decode_tile_edge = decode_tiling
                    .as_ref()
                    .and_then(|tiling| tiling.spatial.as_ref())
                    .map(|spatial| spatial.tile_px as u32);

                // Feasibility is evaluated at the requested geometry only. No provider path may rewrite
                // width or height: an infeasible request is refused before control preprocessing or
                // render. There is currently no provider-owned current measurement bundle from which to
                // name a verified alternative, so `alternative` is truthfully absent.
                let requested_geometry = RequestedControlGeometry::from_request(req);
                let (requested_width, requested_height) =
                    requested_geometry.feasibility_dimensions();
                let feasible = crate::memory::control_geometry_fits(
                    safe_gib,
                    &heavy_owned.cfg,
                    heavy.branch.num_blocks(),
                    heavy_owned.base_tier,
                    heavy_owned.branch_tier,
                    requested_width,
                    requested_height,
                    text_co_resident,
                    decode_tile_edge,
                );
                // Hoist the count-invariant pose VAE encode + text prep OUT of the per-image loop
                // (F-073): both depend only on the (shared) context + pose + geometry, not the per-seed
                // noise. Build the plan ONCE; each seed reuses it via `render_control_from`.
                let (admitted_geometry, plan) =
                    requested_geometry.prepare_after_feasibility(feasible, |width, height| {
                        heavy
                            .heavy
                            .prepare_control(&context, control_image, width, height)
                    })?;

                let mut images = Vec::with_capacity(req.count as usize);
                for n in 0..req.count {
                    let opts = admitted_geometry.turbo_options(
                        req,
                        steps,
                        base_seed.wrapping_add(n as u64),
                    );
                    let img = heavy.heavy.render_control_from(
                        &plan,
                        heavy.branch,
                        control_scale,
                        heavy
                            .alternate_decoder
                            .map(|decoder| decoder as &dyn LatentDecoder),
                        decode_tiling.as_ref(),
                        req.memory.and_then(|memory| memory.calibration_error_phase),
                        &opts,
                        &req.cancel,
                        &req.preview,
                        on_progress,
                    )?;
                    images.push(img);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

impl Generator for KreaTurboControl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Capability floor (size/count/guidance/negative/conditioning-kind), then the shared
        // control-present check (a `Conditioning::Control` must be present).
        crate::model::validate_request(&self.descriptor, req)?;
        self.require_control_present(req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(
            &self.memory_strategy,
            self.loaded_precision,
            self.loaded_quant,
            context,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        crate::memory_strategy::begin_request(
            KREA_2_TURBO_CONTROL_ID,
            &self.memory_strategy,
            self.loaded_precision,
            self.loaded_quant,
            context,
        )
    }
}

// Explicit registration for `krea_2_turbo_control`. The `impl Generator` stays hand-written
// because `validate` adds the control-present check beyond the shared `validate_request`, so it is not
// the plain delegation `impl_generator!` expresses (the z-image control precedent).
mlx_gen::register_generators! {
    pub(crate) const CONTROL_REGISTRATION = descriptor => load;
    footprint = crate::model::component_footprint
}

pub const MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: KREA_2_TURBO_CONTROL_ID,
        contract: |spec| {
            crate::memory_strategy::memory_strategy_contract(KREA_2_TURBO_CONTROL_ID, spec)
        },
        safety_check: crate::memory_strategy::registered_safety_check,
    };
pub const MEMORY_BEHAVIOR_REGISTRATION: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: KREA_2_TURBO_CONTROL_ID,
        valid_fixtures: crate::memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            crate::memory_strategy::registered_begin_request(
                KREA_2_TURBO_CONTROL_ID,
                spec,
                contract,
                context,
            )
        },
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{Conditioning, ControlKind, Modality, OffloadPolicy, Quant, WeightsSource};

    #[test]
    fn decode_tiling_follows_only_the_selected_generation_memory() {
        assert!(selected_decode_tiling(None).unwrap().is_none());
        assert!(
            selected_decode_tiling(Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }))
            .unwrap()
            .is_none(),
            "staged residency must not be upgraded by a provider-local budget decision"
        );

        let selected = selected_decode_tiling(Some(gen_core::GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(crate::memory_strategy::DECODE_TILE_EDGE),
            decode_overlap: Some(crate::memory_strategy::DECODE_OVERLAP),
            ..Default::default()
        }))
        .unwrap()
        .expect("the upstream bounded-decode selection must be applied");
        let spatial = selected.spatial.unwrap();
        assert_eq!(spatial.tile_px, 512);
        assert_eq!(spatial.overlap_px, 64);
    }

    #[test]
    fn requested_geometry_drives_feasibility_preprocessing_and_render_options() {
        use std::cell::Cell;

        let mut req = GenerationRequest {
            width: 640,
            height: 384,
            sampler: Some("euler".into()),
            scheduler: Some("native".into()),
            ..Default::default()
        };

        let requested = RequestedControlGeometry::from_request(&req);
        assert_eq!(requested.feasibility_dimensions(), (640, 384));

        let preprocessing_called = Cell::new(false);
        let refusal = requested
            .prepare_after_feasibility(false, |_, _| {
                preprocessing_called.set(true);
                Ok(())
            })
            .expect_err("infeasible requested geometry must be refused before rendering");
        assert!(
            !preprocessing_called.get(),
            "control preprocessing must not run for refused geometry"
        );
        assert!(
            matches!(
                refusal,
                Error::GeometryRefused {
                    requested_width: 640,
                    requested_height: 384,
                    alternative: None,
                    ..
                }
            ),
            "refusal must retain the exact requested geometry"
        );

        let prepared_dimensions = Cell::new(None);
        let (admitted, ()) = requested
            .prepare_after_feasibility(true, |width, height| {
                prepared_dimensions.set(Some((width, height)));
                Ok(())
            })
            .expect("feasible requested geometry must be admitted unchanged");
        assert_eq!(prepared_dimensions.get(), Some((640, 384)));
        assert_eq!(admitted.dimensions(), (640, 384));

        // Render options must consume the admitted geometry, not re-read substitutable request
        // fields after the gate. This mutation makes that distinction observable without weights.
        req.width = 1024;
        req.height = 768;
        let opts = admitted.turbo_options(&req, 8, 42);
        assert_eq!((opts.width, opts.height), (640, 384));
        assert_eq!(opts.steps, 8);
        assert_eq!(opts.seed, 42);
        assert_eq!(opts.sampler.as_deref(), Some("euler"));
        assert_eq!(opts.scheduler.as_deref(), Some("native"));
        assert_eq!(opts.transformer_window_size, None);
    }

    #[test]
    fn descriptor_is_krea_2_turbo_control() {
        let d = descriptor();
        assert_eq!(d.id, "krea_2_turbo_control");
        assert_eq!(d.family, "krea_2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        // Pose ControlNet: Control conditioning (not the base Turbo Reference), CFG-free.
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(!d.capabilities.accepts(ConditioningKind::Reference));
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        // Q4/Q8 base is supported (sc-11727) — inherited from the base Turbo descriptor, no longer blanked.
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        // Shared surface stays in lockstep with the base Turbo descriptor.
        assert!(d.capabilities.mac_only);
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert_eq!(
            d.capabilities.max_count,
            crate::model::descriptor().capabilities.max_count
        );
    }

    #[test]
    fn native_control_load_missing_base_fails_at_config_read() {
        // The native control entrypoint proves the base tier is present FIRST (arch-config read),
        // exactly like the t2i native entrypoint — a bogus base fails there, before any tensor I/O.
        let e = load_control_from_native_dit_file(
            "/nonexistent-krea/dit.safetensors",
            "/nonexistent-krea",
            "/nonexistent-krea/control.safetensors",
            &[],
        )
        .err()
        .expect("missing base snapshot → err")
        .to_string();
        assert!(
            e.contains("config.json") || e.contains("read"),
            "expected the missing-base config-read error, got: {e}"
        );
    }

    #[test]
    fn native_control_load_accepts_adapters_without_early_rejection() {
        // Adapters thread through the native control assembly (parity with the snapshot control
        // load's `spec.adapters`) and must NOT be rejected at the door — with a bogus base the load
        // still fails first at the config read, never at an "adapters unsupported" guard.
        let adapters = vec![mlx_gen::AdapterSpec::new(
            std::path::PathBuf::from("/nonexistent-krea/style.safetensors"),
            1.0,
            mlx_gen::AdapterKind::Lora,
        )];
        let e = load_control_from_native_dit_file(
            "/nonexistent-krea/dit.safetensors",
            "/nonexistent-krea",
            "/nonexistent-krea/control.safetensors",
            &adapters,
        )
        .err()
        .expect("missing base snapshot → err")
        .to_string();
        assert!(
            !e.to_lowercase().contains("not yet supported")
                && !e.to_lowercase().contains("not supported"),
            "adapters must be accepted by the native control loader, got: {e}"
        );
    }

    #[test]
    fn native_control_valid_config_reaches_fail_closed_asset_sizing() {
        // A parseable arch config advances the load to the memory contract's fail-closed component
        // sizing, which requires the base text encoder / VAE and the pose overlay to actually hold
        // tensor bytes — a torn base or missing overlay fails loudly here, before any weight load.
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(root.join("transformer/config.json"), "{}").unwrap();

        let e = load_control_from_native_dit_file(
            root.join("missing-dit.safetensors"),
            &root,
            root.join("missing-control.safetensors"),
            &[],
        )
        .err()
        .expect("missing required components must fail")
        .to_string();
        assert!(
            e.contains("asset facts"),
            "expected the post-config asset-sizing stage, got: {e}"
        );
        assert!(!e.contains("config.json"), "config was valid, got: {e}");
    }

    #[test]
    fn load_rejects_missing_control_weights() {
        // A base dir but no `spec.control` → fail on the missing overlay (proving it is a hard
        // requirement — never a silent un-conditioned base).
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("pose control overlay"), "got: {err}");
    }

    #[test]
    fn load_rejects_single_file_without_companion_base() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/krea.safetensors".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains(BASE_SNAPSHOT_COMPONENT), "got: {err}");
    }

    #[test]
    fn load_accepts_quant_override() {
        // sc-11727: a quant override is NOT rejected (Q4/Q8 base is supported). Pin `Resident` so the load
        // deterministically eager-loads and fails on the missing weights — NOT on a dense-only quant guard.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()))
            .with_offload_policy(OffloadPolicy::Resident)
            .with_quant(Quant::Q8);
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(
            !err.contains("quantization is not supported")
                && !err.contains("Q4/Q8")
                && !err.contains("precision override"),
            "quant override must be accepted, not rejected by a dense-only guard; got: {err}"
        );
    }

    #[test]
    fn reachable_via_registry_by_id() {
        assert!(
            crate::provider_registry()
                .unwrap()
                .generators()
                .copied()
                .any(|r| (r.descriptor)().id == KREA_2_TURBO_CONTROL_ID),
            "id {KREA_2_TURBO_CONTROL_ID} not registered"
        );
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = crate::provider_registry()
            .unwrap()
            .load(KREA_2_TURBO_CONTROL_ID, &spec)
            .err()
            .expect("missing weights → err")
            .to_string();
        assert!(
            !err.contains("no generator registered"),
            "id not resolved: {err}"
        );
    }

    #[test]
    fn accepts_pose_rejects_other_control_kinds() {
        // A pose-only branch: the trait's resolve_control admits Pose and rejects Canny/Depth. Exercised
        // through the descriptor's accepted_control_kinds via a lightweight stub-free check on the enum.
        let accepted = AcceptedControlKinds::Only(vec![ControlKind::Pose]);
        assert!(accepted.accepts(&ControlKind::Pose));
        assert!(!accepted.accepts(&ControlKind::Canny));
        assert!(!accepted.accepts(&ControlKind::Depth));
        // Sanity: the trait wiring builds the same Conditioning::Control the worker feeds.
        let _ = Conditioning::Control {
            image: mlx_gen::media::Image {
                width: 8,
                height: 8,
                pixels: vec![0u8; 8 * 8 * 3],
            },
            kind: ControlKind::Pose,
            scale: Some(0.6),
        };
    }

    /// Real-weight harness for the native single-file + pose-branch assembly (the control twin of
    /// `model::native_load_folds_edit_adapter`): the discriminating check the GPU-free tests can't
    /// run (the branch fold needs a real DiT/base/overlay). Set `KREA_NATIVE_DIT` to a ComfyUI
    /// single-file Krea 2 DiT (e.g. a community fine-tune), `KREA_TURBO_DIR` to a resident turnkey
    /// snapshot tier (dense bf16), and `KREA_CONTROL_OVERLAY` to the converted pose control-branch
    /// overlay. Asserts the concrete assembly carries the `krea_2_turbo_control` identity with the
    /// dense-base ⇒ dense-branch tier consequence, so the imported composition is the builtin one
    /// with only the DiT swapped.
    #[test]
    #[ignore = "needs real weights: set KREA_NATIVE_DIT, KREA_TURBO_DIR, KREA_CONTROL_OVERLAY"]
    fn native_control_load_assembles_branch_on_imported_dit() {
        let dit = std::env::var("KREA_NATIVE_DIT").expect("set KREA_NATIVE_DIT");
        let base = std::env::var("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR");
        let overlay = std::env::var("KREA_CONTROL_OVERLAY").expect("set KREA_CONTROL_OVERLAY");
        let model = build_native_krea_control(&dit, &base, &overlay, &[])
            .expect("native single-file control load");
        assert_eq!(model.descriptor.id, KREA_2_TURBO_CONTROL_ID);
        assert_eq!(model.memory_strategy.provider_id, KREA_2_TURBO_CONTROL_ID);
        // The native loader materializes the imported DiT dense (I8 files dequantize at load), so
        // the assembly records no base tier and the branch stays the dense bf16 it was trained as.
        assert!(model.loaded_quant.is_none());
        assert_eq!(model.loaded_precision, Precision::Bf16);
        assert!(
            model
                .memory_strategy
                .resident_components()
                .iter()
                .any(|component| component.id == "pose_control_branch"),
            "the pose branch must be declared resident"
        );
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that Krea-Control's dispatch HONORS
    // `offload_policy` — not a smoke test. `build_control_residency` points at a non-existent base
    // snapshot *directory* (with a control overlay present, so the up-front precision/single-file/
    // missing-control/quant guards all pass). The discriminator is the deferral:
    //   * `Sequential` must capture the two loaders and touch NO component weights → `Ok`, and the built
    //     residency is `Sequential` (`is_sequential()`).
    //   * `Resident` must eager-load the text phase from that non-existent dir → `Err`.
    // A dispatch that ignored `offload_policy` and always built `Resident` (the F-172 bug class) would
    // eager-load under a `Sequential` request and turn the first assertion's `Ok` into an `Err` — this
    // test would fail. The A/B real-weight test is `#[ignore]`d; this runs by default.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/krea-control-residency-test-snapshot".into(),
        ))
        .with_control(WeightsSource::File(
            "/nonexistent/krea-control-residency-test-overlay.safetensors".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let res = build_control_residency(&missing_snapshot_spec(OffloadPolicy::Sequential))
            .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential residency (the deferred state machine)"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        let err = build_control_residency(&missing_snapshot_spec(OffloadPolicy::Resident))
            .err()
            .expect("Resident must eager-load and fail on a missing snapshot dir");
        // A load/IO error, not one of the up-front guards (which this valid Dir+control spec passes).
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file")
                && !msg.contains("precision override")
                && !msg.contains("pose control overlay")
                && !msg.contains("quantization is not supported"),
            "expected an eager-load failure, got an up-front guard: {msg}"
        );
    }
}
