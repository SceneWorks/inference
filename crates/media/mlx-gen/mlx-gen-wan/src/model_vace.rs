//! `mlx-gen-wan` Wan-VACE model entry (`wan_vace`, epic 3040 / sc-3388, S3 / sc-3436) — the native
//! port of diffusers `WanVACEPipeline`: a control video + per-frame mask (+ optional reference
//! images) → controllable video, covering **replace_person** (the SceneWorks worker's current Wan
//! `replace_person` path) plus **pose / depth / sketch control** and **extend / video_bridge** (the
//! Wan answer to sc-3357 / sc-3385).
//!
//! VACE is **mode-agnostic at the engine boundary**, exactly like diffusers `WanVACEPipeline`: the
//! worker builds the per-mode control video + mask (replace_person = the person-region-neutralized
//! clip + the person mask; pose/depth control = the render + an all-active mask; extend/bridge = the
//! source frames at the kept positions + a generated-span mask) and passes them as one
//! [`mlx_gen::Conditioning::ControlClip`]. The provider VAE-encodes the inactive/reactive split + unfolds the
//! mask into the 96-ch control latent ([`crate::vace::prepare_video_latents`] /
//! [`prepare_masks`]) and runs the CFG VACE denoise loop
//! ([`denoise_vace`]). Reference images (from [`mlx_gen::Conditioning::Reference`])
//! are encoded to leading latent frames and dropped after denoise (diffusers
//! `latents[:, :, num_reference_images:]`).
//!
//! **Snapshot layout** (the cutover, sc-3055, converts the diffusers VACE repo into this): the VACE
//! transformer in **diffusers tensor layout** (read directly by [`WanVaceTransformer`]) at
//! `model.safetensors` or a `transformer/` shard dir, plus the shared native-converted UMT5
//! (`t5_encoder.safetensors` + `tokenizer.json`) and z16 Wan VAE (`vae.safetensors`) — the same
//! components the base Wan 14B uses. **e2e is checkpoint-gated** (no VACE checkpoint in the local HF
//! cache yet — `tests/wanvace_e2e.rs`, `#[ignore]`); the engine pieces are validated component-wise
//! (S1 transformer structural parity, S2 conditioning byte-parity).

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterSpec, Capabilities, ConditioningKind, Error, GenerationOutput, GenerationRequest,
    Generator, Image, LoadPhase, LoadSpec, Modality, ModelDescriptor, MoeExpert, OffloadPolicy,
    Precision, Progress, Quant, Result, WeightsSource,
};
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::{random, Array, Dtype};

use crate::adapters::{merge_vace_adapters, merge_vace_adapters_expert, warn_skipped_adapters};
use crate::config::{WanModelConfig, WanVaceConfig};
use crate::model::{dit_resident_bytes, is_wan_curated, moe_denoise_resident_bytes};
use crate::pipeline::{
    align_dim, auto_tiling_budgeted_z16, crossing_index, decode_to_frames, frames_to_images,
    latent_shape, preflight_denoise_memory_guard, preprocess_i2v_image, reject_off_grid,
    reject_over_area, resolve_sampler_knobs, seq_len, staged_expert_swap,
};
use crate::scheduler::{make_scheduler, SolverKind, WanScheduler};
use crate::text_encoder::encode_text_staged_for_tier;
use crate::vace::{
    build_vace_control, denoise_vace, denoise_vace_moe, denoise_vace_range, prepare_masks,
    prepare_video_latents, vace_control_scales, WanVaceTransformer,
};

/// Concrete z16 VAE assigned to both VACE routes.
pub type ProviderVae = crate::vae::WanVae;

/// Resolve either VACE route's load-bearing VAE geometry.
pub fn vae_tiling(provider_id: &str) -> Option<mlx_gen::tiling::VaeTiling> {
    matches!(provider_id, MODEL_ID_VACE | MODEL_ID_VACE_FUN).then_some(ProviderVae::VAE_TILING)
}

/// Public provider id: `"wan_vace"`.
pub const MODEL_ID_VACE: &str = "wan_vace";

/// The Wan z16 VAE strides (the VACE checkpoints are Wan2.1-based): temporal 4, spatial 8, patch 2.
const VAE_T: usize = ProviderVae::VAE_TILING.temporal_scale as usize;
const VAE_S: usize = ProviderVae::VAE_TILING.spatial_scale as usize;

/// Upper bound on the control-clip frame count (sc-12459 / F-008): `vace_prep` sizes the control
/// video, mask, control-latent, and init-noise tensors directly from `clip.frames.len()`, so an
/// unbounded count (the gen-core capability floor only rejects a pathological 1 000 000) would drive
/// enormous allocations before any typed error. `1025` (= 1 + 4·256, ~64 s at Wan's 16 fps — far
/// above the 81/121-frame lengths the Wan checkpoints are trained on) mirrors the LTX lane's real
/// `MAX_FRAMES = 1025` ceiling; realistic-but-large requests below it are still bounded by the
/// sc-4986 [`preflight_denoise_memory_guard`] both VACE `generate_impl`s now run.
const MAX_CONTROL_FRAMES: usize = crate::MAX_WAN_FRAMES;

/// Drop the leading `num_ref` reference latent frames along the temporal axis (axis 1) — the diffusers
/// `latents[:, :, num_reference_images:]` slice both VACE variants apply after denoise, before the VAE
/// decode. `t_total` is the latent's temporal length; a no-op when `num_ref == 0` (F-010).
fn drop_reference_frames(latents: Array, num_ref: usize, t_total: i32) -> Result<Array> {
    if num_ref > 0 {
        let keep = Array::from_slice(
            &((num_ref as i32)..t_total).collect::<Vec<i32>>(),
            &[t_total - num_ref as i32],
        );
        Ok(latents.take_axis(&keep, 1)?)
    } else {
        Ok(latents)
    }
}

/// The pre-DiT setup shared by both VACE generators (F-072): everything from knob resolution through
/// the z16-VAE control latent and the seeded init noise — byte-identical between the single- and
/// dual-expert paths, which differ only in how they resolve `guidance` (single vs low/high) and thus
/// `cfg_disabled` (computed by each caller and passed in). Fields are exactly the locals each
/// `generate_impl` binds; the DiT stage and decode tail read them unchanged.
struct VacePrep {
    width: u32,
    height: u32,
    steps: usize,
    shift: f32,
    kind: SolverKind,
    context: Array,
    context_null: Option<Array>,
    control: Array,
    t_total: i32,
    scales: Vec<f32>,
    init_noise: Array,
    num_ref: usize,
}

/// Attention-token count of one VACE denoise forward, computed **from the request alone** (sc-12459
/// / F-008 — before the UMT5 encode, the VAE conditioning encode, or any weight load): the
/// control-latent grid `vace_prep` will materialize is `[·, t_lat + num_ref, H/8, W/8]` (the
/// `1 + (F−1)/4` z16 temporal latents plus one prepended latent frame per `Reference` image,
/// `prepare_video_latents`), and the init noise matches it — so the dense lanes' [`seq_len`] over
/// that grid is exactly the per-forward token count the denoise loop runs.
fn vace_denoise_tokens(config: &WanVaceConfig, req: &GenerationRequest) -> Result<usize> {
    let base = &config.base;
    // Same alignment `vace_prep` applies (round down to patch · VAE_S) — never larger than requested.
    let width = align_dim(req.width, base.patch_size.2, VAE_S);
    let height = align_dim(req.height, base.patch_size.1, VAE_S);
    let frames = req.control_clip().map(|c| c.frames.len()).unwrap_or(1);
    let num_ref = req
        .conditioning
        .iter()
        .filter(|c| matches!(c, mlx_gen::Conditioning::Reference { .. }))
        .count();
    let mut lat = latent_shape(frames, height, width, base.vae_z_dim, (VAE_T, VAE_S, VAE_S))?;
    lat[1] += num_ref as i32; // reference images prepend latent frames
    Ok(seq_len(lat, base.patch_size))
}

/// Run the shared VACE pre-DiT setup (Stages 1–2 + noise seeding). `cfg_disabled` is the caller's
/// guidance decision (single-expert `guidance ≤ 1.0`; dual-expert `low ≤ 1.0 && high ≤ 1.0`) — the
/// only place the two paths diverge before the DiT stage. Byte-identical to the code both
/// `generate_impl`s previously open-coded.
fn vace_prep(
    root: &Path,
    config: &WanVaceConfig,
    req: &GenerationRequest,
    cfg_disabled: bool,
    load_quant: Option<Quant>,
) -> Result<VacePrep> {
    let base = &config.base;
    let clip = req.control_clip().expect("validated present");

    // --- Resolve knobs ---
    // VACE aligns to patch · VAE_S, matching the dense paths' `resolve_capped_dims`. The max-area
    // cap is enforced (by rejection) in `validate_vace_clip`, so the alignment here only rounds down
    // and cannot push a validated request back over budget (sc-12308).
    let width = align_dim(req.width, base.patch_size.2, VAE_S);
    let height = align_dim(req.height, base.patch_size.1, VAE_S);
    let (steps, shift, kind, seed) =
        resolve_sampler_knobs(req, base.sample_steps, base.sample_shift);
    let neg_prompt = req
        .negative_prompt
        .clone()
        .unwrap_or_else(|| base.sample_neg_prompt.clone());

    // Control video [-1,1] + mask [0,1] (diffusers `clamp((m+1)/2)`), each [3, F, H, W].
    let control_video = preprocess_clip(clip.frames, width, height)?;
    let mask = preprocess_clip(clip.mask, width, height)?;
    let half = Array::from_slice(&[0.5f32], &[1]);
    let mask = multiply(&add(&mask, Array::from_slice(&[1.0f32], &[1]))?, &half)?; // (m+1)/2 ∈ [0,1]

    // Reference images (optional) → channels-first [3, H, W] each.
    let references: Vec<Array> = req
        .conditioning
        .iter()
        .filter_map(|c| match c {
            mlx_gen::Conditioning::Reference { image, .. } => Some(image),
            _ => None,
        })
        .map(|im| preprocess_i2v_image(im, width, height))
        .collect::<Result<_>>()?;
    let num_ref = references.len();

    // --- Stage 1: UMT5 text encode ---
    let (context, context_null) = encode_text_staged_for_tier(
        root,
        base,
        &req.prompt,
        &neg_prompt,
        cfg_disabled,
        load_quant,
    )?;

    // --- Stage 2: z16 VAE encode the control + mask → 96-ch control latent ---
    let control = {
        let w = Weights::from_file(root.join("vae.safetensors"))?;
        let vae = ProviderVae::from_weights(&w)?;
        let video_latents = prepare_video_latents(&vae, &control_video, Some(&mask), &references)?;
        let mask_latents = prepare_masks(&mask, VAE_T, VAE_S, base.patch_size.1, num_ref)?;
        let c = build_vace_control(&video_latents, &mask_latents)?;
        mlx_rs::transforms::eval([&c])?;
        c
    };
    // Control latent dims: [96, T_lat(+num_ref), h, w] → the noisy latent matches its frame/space.
    let csh = control.shape();
    let (t_total, h_lat, w_lat) = (csh[1], csh[2], csh[3]);
    // Per-vace-layer control_hidden_states_scale (diffusers `conditioning_scale`), broadcast from
    // the request (sc-3441). `None` ⇒ the diffusers default 1.0.
    //
    // sc-20261: `clip.masking_strength` is folded in here. VACE exposes ONE conditioning scale for
    // the whole hint stack, so the requested masking strength weights the mask/video control by
    // multiplying it — the mechanism the candle lane's dual-expert VACE-Fun route already used,
    // now shared by both MLX routes because they both run this `vace_prep`. `masking_strength =
    // 1.0` (the contract default) is the identity, so a default request is byte-identical to
    // pre-sc-20261. The `[0,1]` range this depends on is enforced in `validate_vace_clip`.
    //
    // The whole vector is resolved by the shared `vace.rs` seam rather than being assembled here,
    // so the honor wiring has one testable definition; `vace_prep_binds_scales_to_the_shared_
    // resolver` pins this line to it (a call site that rebuilt the vec inline would silently
    // un-honor the knob while every unit test on the seam stayed green).
    let scales = vace_control_scales(req, config.vace_layers.len());

    // Seeded init noise [z16, T_lat(+num_ref), h, w].
    let key = random::key(seed)?;
    let init_noise = random::normal::<f32>(
        &[base.vae_z_dim as i32, t_total, h_lat, w_lat],
        None,
        None,
        Some(&key),
    )?;

    Ok(VacePrep {
        width,
        height,
        steps,
        shift,
        kind,
        context,
        context_null,
        control,
        t_total,
        scales,
        init_noise,
        num_ref,
    })
}

/// The shared VACE decode tail (Stage 4, F-072): drop the leading reference latent frames, then
/// z16-VAE-decode the denoised latents → RGB8 frames → a `Video` output. Byte-identical to the code
/// both `generate_impl`s previously open-coded after their (single- vs dual-expert) DiT stage.
fn vace_decode_tail(
    root: &Path,
    base: &WanModelConfig,
    latents: Array,
    prep: &VacePrep,
    req: &GenerationRequest,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    // Drop the leading reference latent frames (diffusers `latents[:, :, num_reference_images:]`).
    let latents = drop_reference_frames(latents, prep.num_ref, prep.t_total)?;

    // --- Stage 4: z16 VAE decode → RGB8 frames ---
    on_progress(Progress::Decoding);
    // sc-6894 — the z16 VAE is non-causal in time (out_f = T_lat·VAE_T, ×4), NOT the causal 4·T−3
    // (task 6897); only the tiling heuristic reads out_frames. Budgeted, catchable selector (F-009).
    let latent_shape = latents.shape();
    let out_frames = latent_shape[1] * ProviderVae::VAE_TILING.temporal_scale;
    let out_height = latent_shape[2] * ProviderVae::VAE_TILING.spatial_scale;
    let out_width = latent_shape[3] * ProviderVae::VAE_TILING.spatial_scale;
    if (out_width, out_height) != (prep.width as i32, prep.height as i32) {
        return Err(Error::Msg(format!(
            "wan vace: z16 VAE geometry resolves {out_width}x{out_height}, expected {}x{}",
            prep.width, prep.height
        )));
    }
    let tiling = if req.memory.is_some_and(|memory| memory.tile_vae_decode) {
        crate::i2v_memory_strategy::decode_tiling(req, prep.width, prep.height, out_frames as u32)?
    } else {
        auto_tiling_budgeted_z16(out_height, out_width, out_frames)?
    };
    let frames_u8 = {
        let w = Weights::from_file(root.join("vae.safetensors"))?;
        let vae = ProviderVae::from_weights(&w)?;
        decode_to_frames(&vae, &latents, tiling.as_ref(), Some(&req.cancel))?
    };
    let images = frames_to_images(&frames_u8)?;

    let fps = req.fps.unwrap_or(base.sample_fps);
    Ok(GenerationOutput::Video {
        frames: images,
        fps,
        audio: None,
    })
}

/// Stable identity + advertised capabilities for `wan_vace`.
pub fn descriptor_vace() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID_VACE,
        family: "wan",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // CFG (guide 5.0) + the Chinese anti-artifact negative prompt. The control input is a
            // masked control clip (`ControlClip`, the universal VACE form the worker builds per
            // mode); optional `Reference` images become leading conditioning frames.
            supports_negative_prompt: true,
            supports_guidance: true,
            conditioning: vec![ConditioningKind::ControlClip, ConditioningKind::Reference],
            // Q4/Q8 is wired (sc-3440, `spec.quantize` → `WanVaceTransformer::quantize`, mirroring the
            // base Wan slice sc-2682). LoRA/LoKr is wired (sc-3439): `spec.adapters` merge onto the
            // dense diffusers-layout VACE weight map via `merge_vace_adapters` before `from_weights` +
            // quantize (the fork order — merge a dense adapter, then quantize), mirroring the base Wan
            // slices sc-2683 (LoRA) / sc-2393 (LoKr) on the diffusers host.
            supports_lora: true,
            supports_lokr: true,
            // sc-7296: curated gen-core vocabulary (`uni_pc`/`dpmpp_2m`) routed to Wan's native solvers
            // + legacy aliases; VACE advertises native solvers only (no `run_flow_sampler` fold-ins).
            samplers: crate::model::wan_native_samplers(),
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Not wired onto the shared `Residency` seam (F-176); Sequential is a no-op fallback.
            supports_sequential_offload: false,
            // The TE, scoped z16 VAE work, and VACE transformer are loaded/used/dropped as phases on
            // every request even though this provider exposes no selectable Sequential control.
            unconditionally_engages_staged_residency: true,
            ..Default::default()
        },
    }
}

/// The loaded Wan-VACE model. Holds the resolved config + snapshot dir; the heavy components (UMT5
/// TE, the z16 VAE, the VACE DiT) are **staged** inside [`WanVace::generate`] to bound peak memory.
pub struct WanVace {
    descriptor: ModelDescriptor,
    config: WanVaceConfig,
    root: PathBuf,
    /// Optional load-time Q4/Q8 quantization of the VACE DiT `_quantize_predicate` surface (sc-3440),
    /// applied in [`WanVace::generate`] after the dense transformer is built.
    quantize: Option<Quant>,
    /// LoRA/LoKr adapters merged onto the dense diffusers-layout VACE weight map (sc-3439), folded in
    /// [`WanVace::generate`] **before** `from_weights` + quantize (the fork order). Empty for a plain
    /// load — the no-adapter path is byte-identical to pre-sc-3439.
    adapters: Vec<AdapterSpec>,
    i2v_memory: Option<crate::i2v_memory_strategy::PreparedWanI2vMemory>,
}

impl WanVace {
    /// The resolved VACE config (exposed for tests).
    pub fn config(&self) -> &WanVaceConfig {
        &self.config
    }

    /// Merge the load-time LoRA/LoKr adapters onto the dense diffusers-layout VACE weight map in
    /// place (sc-3439), before [`WanVaceTransformer::from_weights`] is built and before `spec.quantize`
    /// quantizes (the fork order — merge a dense adapter, then quantize). No-op without adapters (the
    /// no-adapter weight map is byte-identical). VACE is a single dense transformer (no MoE), so it
    /// takes only **shared** (untagged) specs — a `moe_expert`-tagged spec (the dual-expert A14B
    /// surface) is a misconfiguration here, surfaced rather than silently honored. Reuses the
    /// diffusers-named [`merge_vace_adapters`] seam; per-key skips are reported (the reference warns
    /// on skip), and a non-empty spec list that matched **nothing** is a format/prefix error.
    fn merge_adapters(&self, w: &mut Weights) -> Result<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        if self.adapters.iter().any(|s| s.moe_expert.is_some()) {
            return Err(Error::Msg(format!(
                "{}: `moe_expert` (high/low) tagging is only for the dual-expert Wan2.2 A14B — the \
                 single dense VACE transformer takes shared (untagged) adapters",
                MODEL_ID_VACE
            )));
        }
        let report = merge_vace_adapters(w, &self.adapters)?;
        if report.applied == 0 {
            return Err(Error::Msg(format!(
                "{}: {} adapter file(s) matched no module — check the format (PEFT `lora_A/B` or \
                 kohya `lora_down/up`, diffusers `blocks.N.attn1/attn2.to_*` / `ffn.net.*` / \
                 `vace_blocks.*` module names, or native Wan names which are renamed to diffusers)",
                MODEL_ID_VACE,
                self.adapters.len()
            )));
        }
        warn_skipped_adapters(MODEL_ID_VACE, &report.skipped);
        Ok(())
    }
}

/// Resolve the `"wan_vace"` configuration from the snapshot directory.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "wan_vace: expected a model directory (converted snapshot), not a single file"
                    .into(),
            ))
        }
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan_vace: precision override is not wired (the DiT runs bf16 GEMMs over an f32 residual \
             stream — the parity regime)"
                .into(),
        ));
    }
    let config = WanVaceConfig::from_model_dir(&root)?;
    let i2v_memory = if spec.resolved_route.as_deref() == Some(MODEL_ID_VACE)
        && spec.prepared_file_pins().is_prepared()
    {
        Some(crate::i2v_memory_strategy::prepare(spec, MODEL_ID_VACE).map_err(Error::from)?)
    } else {
        None
    };
    Ok(Box::new(WanVace {
        descriptor: descriptor_vace(),
        config,
        root,
        quantize: spec.quantize,
        adapters: spec.adapters.clone(),
        i2v_memory,
    }))
}

/// Preprocess a list of frame [`Image`]s → a channels-first `[3, F, H, W]` clip in `[-1, 1]` (the
/// Wan VAE input convention), via the per-frame cover-fit lanczos resize + center-crop.
fn preprocess_clip(frames: &[Image], width: u32, height: u32) -> Result<Array> {
    if frames.is_empty() {
        return Err(Error::Msg("wan_vace: control clip has no frames".into()));
    }
    let planes: Vec<Array> = frames
        .iter()
        .map(|im| Ok(preprocess_i2v_image(im, width, height)?.expand_dims(1)?)) // [3,1,H,W]
        .collect::<Result<_>>()?;
    let refs: Vec<&Array> = planes.iter().collect();
    Ok(concatenate_axis(&refs, 1)?) // [3, F, H, W]
}

// F-072: `WanVace`'s validate body was byte-identical to the documented-shared `validate_vace_clip`
// (with `id = MODEL_ID_VACE`), which the dual-expert `WanVaceFun` already uses — so point both at it.
impl Generator for WanVace {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_vace_clip(&self.descriptor, MODEL_ID_VACE, &self.config, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        self.i2v_memory.as_ref().map(|prepared| &prepared.contract)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        self.i2v_memory.as_ref().map_or_else(
            || mlx_gen::gen_core::MemorySafetyDecision::Reject {
                reason: "wan_vace loaded route has no sealed memory contract".to_owned(),
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

impl WanVace {
    /// The VACE pipeline (port of diffusers `WanVACEPipeline.__call__`): stage the phases to bound
    /// memory — (1) UMT5 encode the prompt (+ neg unless CFG off); (2) load the z16 VAE, build the
    /// 96-ch control latent from the control clip + mask + reference images; (3) load the VACE DiT,
    /// run the CFG [`denoise_vace`] loop with per-vace-layer `control_hidden_states_scale`; (4) drop
    /// the reference latent frames and z16-VAE-decode → RGB8 frames.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        if let Some(prepared) = &self.i2v_memory {
            crate::i2v_memory_strategy::validate_active_request(prepared, req)?;
        }
        let base = &self.config.base;

        // sc-4986 / sc-12459 (F-008) — fail fast (catchable) if the DiT-denoise stage won't fit,
        // BEFORE the UMT5 encode, the VAE conditioning encode, and the VACE DiT load — the same
        // preflight the dense lanes run (model.rs). Batch factor 1 (`cfg_enabled: false`): VACE CFG
        // runs cond/uncond as two sequential B=1 forwards (vace.rs F-073), not the dense lanes'
        // batched B=2, so one forward's working set is the activation peak.
        //
        // Known under-fit: the 72 B/token/dim activation coefficient was fit on the DENSE forward
        // (real TI2V-5B measurements, sc-4986). A VACE forward additionally materializes the
        // per-vace-layer hint stack and the 96-ch control patch-embed, so this likely
        // under-estimates a VACE forward's working set by ~20-30% (partially offset by the guard's
        // 0.85 headroom): near-budget passes are optimistic, not guaranteed. Recalibrating the
        // coefficient requires real-weight VACE runs on hardware and is deliberately not attempted
        // here.
        preflight_denoise_memory_guard(
            self.descriptor.id,
            dit_resident_bytes(&[vace_transformer_weights_path(&self.root)], self.quantize),
            vace_denoise_tokens(&self.config, req)?,
            base.dim,
            false,
        )?;

        // Single-expert guidance: a scalar request `guidance` overrides the config; CFG off ⇒ ≤ 1.0.
        let guidance = base.sample_guide_scale.resolve_single(req.guidance);
        let cfg_disabled = guidance <= 1.0;

        // Stages 1–2 + noise seeding (shared with the dual-expert path, F-072).
        let prep = vace_prep(&self.root, &self.config, req, cfg_disabled, self.quantize)?;

        // --- Stage 3: load the VACE DiT, embed contexts, CFG denoise ---
        let latents = {
            let mut w = load_vace_transformer_weights(&self.root)?;
            // LoRA/LoKr (sc-3439): merge the diffusers-named adapters onto the dense bf16 weight map
            // BEFORE building the transformer and BEFORE quantizing — the fork order (a LoRA folds
            // into the dense weight, then `spec.quantize` quantizes the merged result). No-op without
            // adapters (the weight map stays byte-identical).
            self.merge_adapters(&mut w)?;
            let mut dit = WanVaceTransformer::from_weights(&w, &self.config, Dtype::Bfloat16)?;
            // Q4/Q8 (sc-3440): quantize the DiT `_quantize_predicate` surface in place after the dense
            // build (the diffusers VACE snapshot ships dense bf16 — no pre-quantized VACE snapshot).
            if let Some(q) = self.quantize {
                dit.quantize(q.bits(), None)?;
            }
            let total = prep.steps as u32;
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_vace(
                &dit,
                &prep.control,
                &prep.scales,
                prep.kind,
                base.num_train_timesteps,
                prep.steps,
                prep.shift,
                guidance,
                &prep.context,
                prep.context_null.as_ref(),
                &prep.init_noise,
                &req.cancel,
                &mut on_step,
            )?
        };

        // Stage 4: drop reference frames + z16-VAE decode → RGB8 (shared with the dual-expert path).
        vace_decode_tail(&self.root, base, latents, &prep, req, on_progress)
    }
}

/// Resolve where the VACE transformer weights live (diffusers layout) — the consolidated
/// `model.safetensors` when present, else the sharded `transformer/` dir. Shared by the loader and
/// the sc-12459 preflight's [`dit_resident_bytes`] so the two can't disagree on precedence; a
/// missing snapshot resolves to the (absent) single-file path, which sizes to 0 bytes at preflight
/// (the guard under-counts rather than spuriously firing) and errors loudly at the actual load.
fn vace_transformer_weights_path(root: &std::path::Path) -> PathBuf {
    let single = root.join("model.safetensors");
    if single.exists() {
        return single;
    }
    let shard_dir = root.join("transformer");
    if shard_dir.is_dir() {
        return shard_dir;
    }
    single
}

/// Load the VACE transformer weights (diffusers layout) — a consolidated `model.safetensors` or a
/// sharded `transformer/` dir, whichever the snapshot provides.
fn load_vace_transformer_weights(root: &std::path::Path) -> Result<Weights> {
    let path = vace_transformer_weights_path(root);
    if path.is_file() {
        return Weights::from_file(path);
    }
    if path.is_dir() {
        return Weights::from_dir(path);
    }
    Err(Error::Msg(format!(
        "wan_vace: no transformer weights at {} (expected model.safetensors or a transformer/ dir)",
        root.display()
    )))
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_generators! {
    pub(crate) const VACE_REGISTRATION = descriptor_vace => load
}

// ============================================================================================
// Wan2.2 VACE-Fun A14B — dual-expert (MoE) controllable video (sc-6604, epic 3456).
//
// The dual-expert sibling of `wan_vace`: `alibaba-pai/Wan2.2-VACE-Fun-A14B` is VACE-trained on the
// Wan2.2-T2V-A14B base, so it ships TWO `WanVACETransformer3DModel`s (high-noise `transformer/`,
// low-noise `transformer_2/`), each with its OWN per-expert `vace_blocks` + `vace_patch_embedding` +
// `text_embedder`, switched at the MoE `boundary_ratio` (0.875) — exactly the base Wan2.2-A14B MoE
// boundary swap ([`crate::pipeline::denoise_moe`]) applied to the VACE forward. The control
// conditioning (96-ch control latent + per-vace-layer scales) and the z16 VAE / UMT5 text encoder are
// identical to single-expert `wan_vace` (VACE-Fun is z16-VAE like Wan2.1 VACE); only the DiT stage is
// dual-expert. Reuses every host-side helper from the single-expert path verbatim.
// ============================================================================================

/// Public provider id: `"wan2_2_vace_fun_14b"`.
pub const MODEL_ID_VACE_FUN: &str = "wan2_2_vace_fun_14b";

/// Stable identity + advertised capabilities for `wan2_2_vace_fun_14b` (same surface as `wan_vace`:
/// a masked `ControlClip` + optional `Reference` images, CFG, Q4/Q8, LoRA/LoKr).
pub fn descriptor_vace_fun() -> ModelDescriptor {
    let mut descriptor = ModelDescriptor {
        required_components: &[],
        id: MODEL_ID_VACE_FUN,
        ..descriptor_vace()
    };
    descriptor.capabilities.supports_sequential_offload = true;
    descriptor
}

/// The loaded Wan2.2 VACE-Fun model (dual-expert). Mirrors [`WanVace`] but stages **two** transformers
/// (high + low) in [`WanVaceFun::generate`].
pub struct WanVaceFun {
    descriptor: ModelDescriptor,
    config: WanVaceConfig,
    root: PathBuf,
    quantize: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    offload_policy: OffloadPolicy,
    i2v_memory: Option<crate::i2v_memory_strategy::PreparedWanI2vMemory>,
}

impl WanVaceFun {
    /// The resolved VACE-Fun config (exposed for tests).
    pub fn config(&self) -> &WanVaceConfig {
        &self.config
    }

    /// Merge the load-time LoRA/LoKr adapters onto **one expert's** dense diffusers-layout weight map
    /// (sc-6604), before [`WanVaceTransformer::from_weights`] + quantize (the fork order). Shared
    /// (untagged) specs merge onto both experts; `moe_expert: high/low`-tagged specs route to their own
    /// (the dual-expert `(loras)+(loras_high/low)` split). The caller folds the "matched nothing across
    /// either expert" check across the two reports.
    fn merge_expert_adapters(&self, w: &mut Weights, expert: MoeExpert) -> Result<usize> {
        if self.adapters.is_empty() {
            return Ok(0);
        }
        let report = merge_vace_adapters_expert(w, &self.adapters, expert)?;
        warn_skipped_adapters(MODEL_ID_VACE_FUN, &report.skipped);
        Ok(report.applied)
    }

    /// Load, adapt, build, and quantize one VACE-Fun expert without materializing its sibling.
    fn build_expert_staged(&self, expert: MoeExpert) -> Result<(WanVaceTransformer, usize)> {
        let mut weights = load_vace_fun_expert_weights(&self.root, expert)?;
        let applied = self.merge_expert_adapters(&mut weights, expert)?;
        let mut transformer =
            WanVaceTransformer::from_weights(&weights, &self.config, Dtype::Bfloat16)?;
        if let Some(q) = self.quantize {
            transformer.quantize(q.bits(), None)?;
        }
        Ok((transformer, applied))
    }

    #[allow(clippy::too_many_arguments)]
    fn denoise_vace_moe_swapped(
        &self,
        control: &Array,
        scales: &[f32],
        kind: SolverKind,
        num_train_timesteps: usize,
        steps: usize,
        shift: f32,
        boundary_timestep: f32,
        guidance_low: f32,
        guidance_high: f32,
        ctx_cond: &Array,
        ctx_uncond: Option<&Array>,
        init_noise: &Array,
        cancel: &mlx_gen::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let mut sched = make_scheduler(kind, num_train_timesteps);
        sched.set_timesteps(steps, shift);
        let timesteps = sched.timesteps().to_vec();
        let k = crossing_index(&timesteps, boundary_timestep);
        if k == 0 || k == steps {
            return Err(Error::Msg(format!(
                "{}: denoise schedule must cross the expert boundary exactly once (boundary {}, \
                 crossing index {k}/{steps})",
                MODEL_ID_VACE_FUN, boundary_timestep
            )));
        }
        let mut latents = init_noise.clone();
        let total = steps as u32;
        let high_applied = std::cell::Cell::new(0usize);
        let low_applied = std::cell::Cell::new(0usize);

        struct VaceSwapState<'a> {
            sched: &'a mut dyn WanScheduler,
            latents: &'a mut Array,
            on_progress: &'a mut dyn FnMut(Progress),
        }

        let denoise_expert = |transformer: &WanVaceTransformer,
                              guidance: f32,
                              range: std::ops::Range<usize>,
                              state: &mut VaceSwapState| {
            let progress = &mut *state.on_progress;
            let mut on_step = |i: usize| {
                progress(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_vace_range(
                &mut *state.sched,
                transformer,
                control,
                scales,
                guidance,
                ctx_cond,
                ctx_uncond,
                &mut *state.latents,
                &timesteps,
                range,
                cancel,
                &mut on_step,
            )
        };

        let mut state = VaceSwapState {
            sched: &mut *sched,
            latents: &mut latents,
            on_progress,
        };
        staged_expert_swap(
            k,
            steps,
            &mut state,
            |state| {
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                (state.on_progress)(Progress::Loading(LoadPhase::Renderer));
                let (transformer, applied) = self.build_expert_staged(MoeExpert::High)?;
                high_applied.set(applied);
                Ok(transformer)
            },
            |transformer, state| denoise_expert(transformer, guidance_high, 0..k, state),
            |state| {
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                (state.on_progress)(Progress::Loading(LoadPhase::Renderer));
                let (transformer, applied) = self.build_expert_staged(MoeExpert::Low)?;
                low_applied.set(applied);
                if !self.adapters.is_empty() && high_applied.get() + low_applied.get() == 0 {
                    return Err(Error::Msg(format!(
                        "{}: {} adapter file(s) matched no module across either expert",
                        MODEL_ID_VACE_FUN,
                        self.adapters.len()
                    )));
                }
                Ok(transformer)
            },
            |transformer, state| denoise_expert(transformer, guidance_low, k..steps, state),
            || {
                mlx_rs::memory::clear_cache();
                Ok(())
            },
        )?;
        Ok(latents)
    }
}

/// Resolve the `"wan2_2_vace_fun_14b"` dual-expert VACE-Fun configuration.
pub fn load_vace_fun(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_vace_fun_14b: expected a model directory (converted dual-expert snapshot), \
                 not a single file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_vace_fun_14b: precision override is not wired (the DiT runs bf16 GEMMs over an \
             f32 residual stream — the parity regime)"
                .into(),
        ));
    }
    let config = WanVaceConfig::vace_fun_from_model_dir(&root)?;
    let i2v_memory = if spec.resolved_route.as_deref() == Some(MODEL_ID_VACE_FUN)
        && spec.prepared_file_pins().is_prepared()
    {
        Some(crate::i2v_memory_strategy::prepare(spec, MODEL_ID_VACE_FUN).map_err(Error::from)?)
    } else {
        None
    };
    Ok(Box::new(WanVaceFun {
        descriptor: descriptor_vace_fun(),
        config,
        root,
        quantize: spec.quantize,
        adapters: spec.adapters.clone(),
        offload_policy: spec.offload_policy,
        i2v_memory,
    }))
}

impl Generator for WanVaceFun {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_vace_clip(&self.descriptor, MODEL_ID_VACE_FUN, &self.config, req)
            .map_err(Into::into)
    }
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        self.i2v_memory.as_ref().map(|p| &p.contract)
    }
    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        self.i2v_memory.as_ref().map_or_else(
            || mlx_gen::gen_core::MemorySafetyDecision::Reject {
                reason: "wan2_2_vace_fun_14b loaded route has no sealed memory contract".to_owned(),
            },
            |p| crate::i2v_memory_strategy::safety_check(p, context),
        )
    }
    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        self.i2v_memory.as_ref().map_or(Ok(None), |p| {
            crate::i2v_memory_strategy::begin_request(p, context)
        })
    }
}

impl WanVaceFun {
    /// The dual-expert VACE pipeline: identical staging to [`WanVace::generate_impl`] (UMT5 encode →
    /// z16 VAE control latent → DiT denoise → drop reference frames → decode) with the DiT stage
    /// loading **both** experts and running the boundary-switched [`denoise_vace_moe`].
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        if let Some(prepared) = &self.i2v_memory {
            crate::i2v_memory_strategy::validate_active_request(prepared, req)?;
        }
        let base = &self.config.base;
        let sequential = self.offload_policy == OffloadPolicy::Sequential
            || crate::i2v_memory_strategy::staged(req);

        // sc-4986 / sc-12459 (F-008) — fail fast (catchable) if the DiT-denoise stage won't fit,
        // BEFORE the UMT5 encode, the VAE conditioning encode, and the 27–54 GB dual-expert load.
        // Both experts stay resident through the denoise (mirroring the dense A14B, model.rs), so
        // their weight files are summed. Batch factor 1 (`cfg_enabled: false`): VACE CFG runs
        // cond/uncond as two sequential B=1 forwards (vace.rs F-073), not the dense lanes' B=2.
        //
        // Known under-fit: the 72 B/token/dim activation coefficient was fit on the DENSE forward
        // (sc-4986); the VACE forward's extra hint stack + 96-ch control patch-embed likely push a
        // forward's working set ~20-30% above this estimate (partially offset by the 0.85
        // headroom) — see the fuller note at the single-expert call site
        // (`WanVace::generate_impl`). Recalibration is hardware-gated and not attempted here.
        let high_bytes = dit_resident_bytes(
            &[vace_fun_expert_weights_path(&self.root, MoeExpert::High)],
            self.quantize,
        );
        let low_bytes = dit_resident_bytes(
            &[vace_fun_expert_weights_path(&self.root, MoeExpert::Low)],
            self.quantize,
        );
        preflight_denoise_memory_guard(
            self.descriptor.id,
            moe_denoise_resident_bytes(
                self.offload_policy,
                req.sampler.as_deref(),
                low_bytes,
                high_bytes,
            ),
            vace_denoise_tokens(&self.config, req)?,
            base.dim,
            false,
        )?;

        // A scalar request `guidance` overrides both experts; otherwise the config (low, high) pair.
        let (low_gs, high_gs) = base.sample_guide_scale.resolve_dual(req.guidance);
        let cfg_disabled = low_gs <= 1.0 && high_gs <= 1.0;

        // Stages 1–2 + noise seeding (shared with the single-expert path, F-072).
        let prep = vace_prep(&self.root, &self.config, req, cfg_disabled, self.quantize)?;

        if sequential {
            // `vace_prep` materializes the raw text contexts and control latent before returning;
            // flush the dropped UMT5/VAE buffers so they do not overlap the active expert.
            mlx_rs::memory::clear_cache();
        }

        // --- Stage 3: native Sequential uses the one-resident expert swap. Curated solvers are not
        // currently advertised by VACE, but retain the conservative resident fallback if added. ---
        let latents = if sequential && !is_wan_curated(req.sampler.as_deref()) {
            let boundary_timestep = base.boundary * base.num_train_timesteps as f32;
            self.denoise_vace_moe_swapped(
                &prep.control,
                &prep.scales,
                prep.kind,
                base.num_train_timesteps,
                prep.steps,
                prep.shift,
                boundary_timestep,
                low_gs,
                high_gs,
                &prep.context,
                prep.context_null.as_ref(),
                &prep.init_noise,
                &req.cancel,
                on_progress,
            )?
        } else {
            // High-noise expert = `transformer/`; low-noise = `transformer_2/` (diffusers naming, the
            // same high/low split the base A14B converter uses, model.rs:806).
            let mut high_w = load_vace_fun_expert_weights(&self.root, MoeExpert::High)?;
            let mut low_w = load_vace_fun_expert_weights(&self.root, MoeExpert::Low)?;
            // LoRA/LoKr per expert (sc-6604) on the dense bf16 weights — BEFORE quantize (fork order).
            let applied = self.merge_expert_adapters(&mut high_w, MoeExpert::High)?
                + self.merge_expert_adapters(&mut low_w, MoeExpert::Low)?;
            if !self.adapters.is_empty() && applied == 0 {
                return Err(Error::Msg(format!(
                    "{}: {} adapter file(s) matched no module across either expert — check the format \
                     (PEFT `lora_A/B` or kohya `lora_down/up`, diffusers `blocks.N.attn1/attn2.to_*` / \
                     `ffn.net.*` / `vace_blocks.*` names) and the `moe_expert` (high/low) tag",
                    MODEL_ID_VACE_FUN,
                    self.adapters.len()
                )));
            }
            let mut high_dit =
                WanVaceTransformer::from_weights(&high_w, &self.config, Dtype::Bfloat16)?;
            let mut low_dit =
                WanVaceTransformer::from_weights(&low_w, &self.config, Dtype::Bfloat16)?;
            if let Some(q) = self.quantize {
                high_dit.quantize(q.bits(), None)?;
                low_dit.quantize(q.bits(), None)?;
            }
            let boundary_timestep = base.boundary * base.num_train_timesteps as f32;
            let total = prep.steps as u32;
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_vace_moe(
                &low_dit,
                &high_dit,
                &prep.control,
                &prep.scales,
                prep.kind,
                base.num_train_timesteps,
                prep.steps,
                prep.shift,
                boundary_timestep,
                low_gs,
                high_gs,
                &prep.context,
                prep.context_null.as_ref(),
                &prep.init_noise,
                &req.cancel,
                &mut on_step,
            )?
        };

        // Stage 4: drop reference frames + z16-VAE decode → RGB8 (shared with the single-expert path).
        vace_decode_tail(&self.root, base, latents, &prep, req, on_progress)
    }
}

/// Load one VACE-Fun expert's transformer weights (diffusers layout). Prefers a converted-snapshot
/// consolidated file (`high_noise_model.safetensors` / `low_noise_model.safetensors`, the names the
/// base-A14B converter writes — model.rs:806), else the raw diffusers shard dir (`transformer/` for
/// high, `transformer_2/` for low). Errors loudly when neither is present (no silent fallback).
fn load_vace_fun_expert_weights(root: &std::path::Path, expert: MoeExpert) -> Result<Weights> {
    let path = vace_fun_expert_weights_path(root, expert);
    if path.is_file() {
        return Weights::from_file(path);
    }
    if path.is_dir() {
        return Weights::from_dir(path);
    }
    let (label, single, dir) = vace_fun_expert_names(expert);
    Err(Error::Msg(format!(
        "wan2_2_vace_fun_14b: no {label}-noise expert weights at {} (expected {single} or a {dir}/ \
         dir)",
        root.display()
    )))
}

/// The `(label, consolidated-file, shard-dir)` names for one VACE-Fun expert (the base-A14B
/// converter's names / the raw diffusers layout — see [`load_vace_fun_expert_weights`]).
fn vace_fun_expert_names(expert: MoeExpert) -> (&'static str, &'static str, &'static str) {
    match expert {
        MoeExpert::High => ("high", "high_noise_model.safetensors", "transformer"),
        MoeExpert::Low => ("low", "low_noise_model.safetensors", "transformer_2"),
    }
}

/// Resolve where one VACE-Fun expert's weights live — consolidated file first, else the shard dir.
/// Shared by the loader and the sc-12459 preflight's [`dit_resident_bytes`] (same missing-snapshot
/// contract as [`vace_transformer_weights_path`]: resolves to the absent file → 0 preflight bytes,
/// loud error at the actual load).
fn vace_fun_expert_weights_path(root: &std::path::Path, expert: MoeExpert) -> PathBuf {
    let (_, single, dir) = vace_fun_expert_names(expert);
    let consolidated = root.join(single);
    if consolidated.exists() {
        return consolidated;
    }
    let shard_dir = root.join(dir);
    if shard_dir.is_dir() {
        return shard_dir;
    }
    consolidated
}

/// Shared control-clip validation for both VACE generators (single + dual expert): the capability
/// check plus the `ControlClip` presence + frame/mask-length + `1 + 4·k` frame-count contract.
fn validate_vace_clip(
    descriptor: &ModelDescriptor,
    id: &'static str,
    config: &WanVaceConfig,
    req: &GenerationRequest,
) -> Result<()> {
    descriptor.capabilities.validate_request(id, req)?;
    // sc-12607/sc-12308: reject an off-grid or over-area geometry rather than silently align-down
    // refitting it — candle hard-errors both on the same request (the widest of the three backend
    // behaviours was mlx's silent refit). Wan2.1's `vace-14B` aligns to `patch · VAE_S` = 16 like the
    // rest of the family and shares its `1280*720` budget.
    let grid = (config.base.patch_size.2 * VAE_S) as u32;
    reject_off_grid(id, req, grid, grid)?;
    reject_over_area(id, req, grid, grid, config.base.max_area)?;
    let clip = req.control_clip().ok_or_else(|| {
        Error::Msg(format!(
            "{id}: needs a ControlClip (the masked control video — the worker builds it per mode: \
             replace_person / pose-depth control / extend-bridge)"
        ))
    })?;
    // sc-20261 — `masking_strength` and `start_frame` were silently dropped on BOTH MLX VACE routes.
    // The candle lane's dual-expert sibling (`candle-gen-wan/src/model_vace_fun.rs`) already honored
    // the first and refused the second; these are that sibling's checks verbatim. MLX has no
    // separate `model_vace_fun.rs` — `WanVace` and `WanVaceFun` share this validator and `vace_prep`
    // — so wiring it here converges all four VACE providers on one answer.
    //
    // The range is load-bearing, not cosmetic: `masking_strength` now multiplies the per-vace-layer
    // conditioning scale (`weighted_control_scale`), so a negative or >1 value would invert or
    // over-drive every hint injection instead of weighting it.
    if !clip.masking_strength.is_finite() || !(0.0..=1.0).contains(&clip.masking_strength) {
        return Err(Error::Msg(format!(
            "{id}: masking_strength must be finite and in [0,1] (got {})",
            clip.masking_strength
        )));
    }
    if clip.start_frame != 0 {
        return Err(Error::Unsupported(format!(
            "{id} currently applies ControlClip only at start_frame=0"
        )));
    }
    // `mode` is already realized in the worker-rasterized control mask; all replacement modes
    // therefore use the same VACE mask path here (the sibling's rule, unchanged).
    if clip.frames.len() != clip.mask.len() {
        return Err(Error::Msg(format!(
            "{id}: control frames ({}) and mask frames ({}) length mismatch",
            clip.frames.len(),
            clip.mask.len()
        )));
    }
    if clip.frames.len() % VAE_T != 1 {
        return Err(Error::Msg(format!(
            "{id}: control clip frame count must be 1 + 4·k (got {})",
            clip.frames.len()
        )));
    }
    // sc-12459 (F-008): a real frame ceiling (the gen-core floor only rejects a pathological
    // 1 000 000) — see [`MAX_CONTROL_FRAMES`].
    if clip.frames.len() > MAX_CONTROL_FRAMES {
        return Err(Error::Msg(format!(
            "{id}: control clip frame count {} exceeds the maximum {MAX_CONTROL_FRAMES}",
            clip.frames.len()
        )));
    }
    let reference_count = req
        .conditioning
        .iter()
        .filter(|conditioning| matches!(conditioning, mlx_gen::Conditioning::Reference { .. }))
        .count();
    let combined = crate::combined_conditioning_latents(clip.frames.len(), reference_count)
        .ok_or_else(|| {
            Error::Msg(format!(
                "{id}: control/reference temporal conditioning size overflowed"
            ))
        })?;
    if combined > crate::MAX_WAN_CONDITIONING_LATENTS {
        let control_latents = 1 + (clip.frames.len() - 1) / VAE_T;
        return Err(Error::Msg(format!(
            "{id}: control clip uses {control_latents} latent frames and {reference_count} reference \
             images, totaling {combined}; the maximum combined temporal conditioning budget is {}",
            crate::MAX_WAN_CONDITIONING_LATENTS
        )));
    }
    Ok(())
}

// Explicit registration for the dual-expert VACE-Fun variant.
mlx_gen::register_generators! {
    pub(crate) const VACE_FUN_REGISTRATION = descriptor_vace_fun => load_vace_fun
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vace::weighted_control_scale;
    use mlx_gen::{Conditioning, OffloadPolicy, ReplacementMode};

    fn test_config() -> WanVaceConfig {
        // The 14B-defaults fallback config (dim 5120, patch (1,2,2), z16 / stride (4,8,8)) — the
        // same one `from_model_dir` returns for an empty dir.
        WanVaceConfig::from_model_dir(std::path::Path::new("/nonexistent-sc12459")).unwrap()
    }

    #[test]
    fn only_dual_expert_vace_advertises_sequential_offload() {
        assert!(!descriptor_vace().capabilities.supports_sequential_offload);
        let fun = descriptor_vace_fun();
        assert!(fun.capabilities.supports_sequential_offload);
        assert_eq!(fun.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
    }

    #[test]
    fn both_registered_vace_variants_declare_unconditional_phase_staging() {
        for descriptor in [descriptor_vace(), descriptor_vace_fun()] {
            assert_eq!(
                descriptor.capabilities.staged_residency_availability(),
                mlx_gen::StagedResidencyAvailability::UnconditionallyEngaged,
                "{} always stages heavyweight phases",
                descriptor.id
            );
        }
        assert!(!descriptor_vace().capabilities.supports_sequential_offload);
        assert!(
            descriptor_vace_fun()
                .capabilities
                .supports_sequential_offload
        );
    }

    #[test]
    fn vace_swap_ranges_match_resident_boundary_choice() {
        let timesteps = [999.0, 930.0, 875.0, 874.0, 500.0, 0.0];
        let boundary = 875.0;
        let k = crossing_index(&timesteps, boundary);
        assert_eq!(k, 3);

        let resident: Vec<_> = timesteps
            .iter()
            .map(|&t| if t >= boundary { "high" } else { "low" })
            .collect();
        let swapped: Vec<_> = (0..timesteps.len())
            .map(|i| if i < k { "high" } else { "low" })
            .collect();
        assert_eq!(resident, swapped);
        assert_eq!(resident.windows(2).filter(|w| w[0] != w[1]).count(), 1);
    }

    #[test]
    fn vace_cache_lifetime_is_bounded_to_one_resident_expert() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct Expert(Rc<Cell<usize>>);
        impl Drop for Expert {
            fn drop(&mut self) {
                self.0.set(self.0.get() - 1);
            }
        }
        struct Cache(Rc<Cell<usize>>);
        impl Drop for Cache {
            fn drop(&mut self) {
                self.0.set(self.0.get() - 1);
            }
        }

        let experts = Rc::new(Cell::new(0));
        let caches = Rc::new(Cell::new(0));
        let peak_experts = Rc::new(Cell::new(0));
        let peak_caches = Rc::new(Cell::new(0));
        let mut state = ();
        staged_expert_swap(
            2,
            4,
            &mut state,
            |_| {
                experts.set(experts.get() + 1);
                peak_experts.set(peak_experts.get().max(experts.get()));
                Ok(Expert(experts.clone()))
            },
            |_, _| {
                caches.set(caches.get() + 1);
                peak_caches.set(peak_caches.get().max(caches.get()));
                let _cache = Cache(caches.clone());
                Ok(())
            },
            |_| {
                assert_eq!((experts.get(), caches.get()), (0, 0));
                experts.set(1);
                peak_experts.set(peak_experts.get().max(experts.get()));
                Ok(Expert(experts.clone()))
            },
            |_, _| {
                caches.set(1);
                peak_caches.set(peak_caches.get().max(caches.get()));
                let _cache = Cache(caches.clone());
                Ok(())
            },
            || {
                assert_eq!((experts.get(), caches.get()), (0, 0));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!((peak_experts.get(), peak_caches.get()), (1, 1));
    }

    #[test]
    fn vace_fun_preflight_budgets_one_expert_only_for_native_sequential() {
        let (low, high) = (8, 11);
        assert_eq!(
            moe_denoise_resident_bytes(OffloadPolicy::Sequential, Some("unipc"), low, high),
            high
        );
        assert_eq!(
            moe_denoise_resident_bytes(OffloadPolicy::Resident, Some("unipc"), low, high),
            low + high
        );
        assert_eq!(
            moe_denoise_resident_bytes(
                OffloadPolicy::Sequential,
                Some("euler_ancestral"),
                low,
                high
            ),
            low + high
        );
    }

    fn img() -> Image {
        Image {
            width: 64,
            height: 64,
            pixels: vec![128u8; 64 * 64 * 3],
        }
    }

    fn clip_request(frames: usize, width: u32, height: u32, num_ref: usize) -> GenerationRequest {
        let mut conditioning = vec![Conditioning::ControlClip {
            frames: (0..frames).map(|_| img()).collect(),
            mask: (0..frames).map(|_| img()).collect(),
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::FaceOnly,
        }];
        for _ in 0..num_ref {
            conditioning.push(Conditioning::Reference {
                image: img(),
                strength: None,
            });
        }
        GenerationRequest {
            prompt: "x".into(),
            width,
            height,
            frames: Some(frames as u32),
            conditioning,
            ..Default::default()
        }
    }

    /// sc-12459 — the preflight token count matches the control-latent grid `vace_prep`
    /// materializes: `[·, (F−1)/4 + 1 + num_ref, H/8, W/8]` through the dense lanes' `seq_len`.
    #[test]
    fn vace_denoise_tokens_matches_control_latent_grid() {
        let cfg = test_config();
        // 64×64, 5 frames, no refs: h_lat = w_lat = 8, t_lat = 2 → 8·8·2 / (2·2) = 32 tokens.
        assert_eq!(
            vace_denoise_tokens(&cfg, &clip_request(5, 64, 64, 0)).unwrap(),
            32
        );
        // A reference image prepends one latent frame: t = 3 → 48 tokens.
        assert_eq!(
            vace_denoise_tokens(&cfg, &clip_request(5, 64, 64, 1)).unwrap(),
            48
        );
        // Dims align DOWN to the patch·VAE_S = 16 grid before the latent divide (never larger):
        // 79 → 64. And the frame axis follows 1 + 4·k: 9 frames → t_lat = 3.
        assert_eq!(
            vace_denoise_tokens(&cfg, &clip_request(9, 79, 64, 0)).unwrap(),
            48
        );
    }

    /// sc-12459 — `validate_vace_clip` enforces the real frame ceiling (the gen-core floor only
    /// rejects a pathological 1 000 000).
    #[test]
    fn validate_vace_clip_rejects_over_cap_frame_count() {
        let cfg = test_config();
        // 1029 = 1 + 4·257 satisfies the 1+4k grid but exceeds MAX_CONTROL_FRAMES = 1025.
        let req = clip_request(1029, 64, 64, 0);
        let err = validate_vace_clip(&descriptor_vace(), MODEL_ID_VACE, &cfg, &req)
            .expect_err("1029 frames must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("exceeds the maximum 1025"),
            "unexpected error: {msg}"
        );
        // The ceiling itself (1025 = 1 + 4·256) still validates.
        assert!(validate_vace_clip(
            &descriptor_vace(),
            MODEL_ID_VACE,
            &cfg,
            &clip_request(1025, 64, 64, 0)
        )
        .is_ok());
        // The dual-expert lane shares the same validator + ceiling.
        let err = validate_vace_clip(&descriptor_vace_fun(), MODEL_ID_VACE_FUN, &cfg, &req)
            .expect_err("dual-expert: 1029 frames must be rejected");
        assert!(err.to_string().contains("exceeds the maximum 1025"));
    }

    /// A `clip_request` with one ControlClip field overridden — the sc-20261 knobs.
    fn clip_request_with(
        masking_strength: f32,
        start_frame: i32,
        mode: ReplacementMode,
    ) -> GenerationRequest {
        let mut req = clip_request(5, 64, 64, 0);
        if let Conditioning::ControlClip {
            masking_strength: ms,
            start_frame: sf,
            mode: m,
            ..
        } = &mut req.conditioning[0]
        {
            *ms = masking_strength;
            *sf = start_frame;
            *m = mode;
        }
        req
    }

    /// sc-20261 — `masking_strength` is no longer silently dropped: `vace_prep` resolves the
    /// per-vace-layer conditioning scale through [`weighted_control_scale`], the same seam and the
    /// same arithmetic as the candle lane.
    ///
    /// The seam assertions use a **non-default** strength on purpose. `masking_strength = 1.0` is
    /// the identity for this mechanism, so an assert at the default would also pass against the old
    /// silently-dropping `req.control_scale.unwrap_or(1.0)` — a false green.
    #[test]
    fn masking_strength_weights_the_control_scale() {
        assert!((weighted_control_scale(Some(0.5), 0.4) - 0.2).abs() < f32::EPSILON);
        assert!((weighted_control_scale(None, 0.25) - 0.25).abs() < f32::EPSILON);
        // The pre-sc-20261 expression was `control_scale.unwrap_or(1.0)` alone; a non-default
        // strength must move the resolved scale off it, or the knob is still inert.
        assert_ne!(weighted_control_scale(Some(0.5), 0.4), 0.5);
        assert_ne!(weighted_control_scale(None, 0.25), 1.0);
        // The contract default is the identity, so a default request renders byte-identically.
        assert_eq!(weighted_control_scale(Some(0.75), 1.0), 0.75);
        assert_eq!(weighted_control_scale(None, 1.0), 1.0);

        // The range the seam depends on is enforced at validate rather than left to poison the
        // hint stack, on BOTH routes (they share this validator).
        let cfg = test_config();
        for (descriptor, id) in [
            (descriptor_vace(), MODEL_ID_VACE),
            (descriptor_vace_fun(), MODEL_ID_VACE_FUN),
        ] {
            let v = |ms: f32| {
                validate_vace_clip(
                    &descriptor,
                    id,
                    &cfg,
                    &clip_request_with(ms, 0, ReplacementMode::FaceOnly),
                )
            };
            // Non-default but in-range strengths are HONORED, not refused.
            assert!(v(0.4).is_ok(), "{id}: 0.4 must validate");
            assert!(v(0.0).is_ok(), "{id}: 0.0 must validate");
            for invalid in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
                let err = v(invalid).expect_err("out-of-range masking_strength must be rejected");
                assert!(
                    err.to_string().contains("masking_strength"),
                    "{id} names the field: {err}"
                );
            }
        }
    }

    /// sc-20261 (adversarial-review follow-up) — the same claim asserted at the **resolution
    /// `vace_prep` actually binds**: request in, per-vace-layer `scales` vector out. The scalar
    /// [`weighted_control_scale`] can be right while a call site bypasses it, so the seam under test
    /// is the whole vector.
    ///
    /// Non-default strengths throughout: at `masking_strength = 1.0` the resolved vector equals the
    /// pre-sc-20261 `req.control_scale.unwrap_or(1.0)` broadcast, so an assert at the default is a
    /// false green.
    #[test]
    fn resolved_control_scales_move_with_a_non_default_masking_strength() {
        let with = |control_scale: Option<f32>, strength: f32| {
            let mut req = clip_request_with(strength, 0, ReplacementMode::FaceOnly);
            req.control_scale = control_scale;
            req
        };
        // The pre-sc-20261 vector, for contrast — what a call site that dropped the knob resolves.
        let dropped = |req: &GenerationRequest, n: usize| vec![req.control_scale.unwrap_or(1.0); n];

        // Explicit control_scale × non-default strength.
        let req = with(Some(0.5), 0.4);
        let got = vace_control_scales(&req, 3);
        assert_eq!(got.len(), 3, "one scale per vace layer");
        assert!(
            got.iter().all(|s| (s - 0.2).abs() < f32::EPSILON),
            "{got:?}"
        );
        assert_ne!(got, dropped(&req, 3), "the knob must move the whole vector");

        // No explicit control_scale: the strength IS the scale, not the dropped 1.0.
        let req = with(None, 0.25);
        let got = vace_control_scales(&req, 2);
        assert!(
            got.iter().all(|s| (s - 0.25).abs() < f32::EPSILON),
            "{got:?}"
        );
        assert_ne!(got, dropped(&req, 2));

        // The contract default is the identity — a default request resolves byte-identically to
        // the pre-sc-20261 expression, so nothing already rendering changes.
        for cs in [Some(0.75_f32), None] {
            let req = with(cs, 1.0);
            assert_eq!(vace_control_scales(&req, 4), dropped(&req, 4));
        }
        // A request with no ControlClip at all falls back to the default strength.
        let mut bare = with(Some(0.4), 0.4);
        bare.conditioning.clear();
        bare.control_scale = Some(0.6);
        assert_eq!(vace_control_scales(&bare, 2), vec![0.6, 0.6]);
    }

    /// sc-20261 (adversarial-review follow-up) — bind the MLX VACE **call site** to
    /// [`vace_control_scales`].
    ///
    /// A unit test on the resolver cannot observe the call site: reverting `vace_prep`'s `scales`
    /// binding to the pre-sc-20261 `vec![req.control_scale.unwrap_or(1.0); n]` un-honors
    /// `masking_strength` while every arithmetic assertion above stays green. `vace_prep` is not
    /// drivable without real weights (the scales are resolved after the VAE encode), so the binding
    /// is pinned in the source instead — the same shape as `mlx-llm`'s
    /// `multi_frame_attention_has_no_quadratic_mask_allocation`. Both MLX VACE routes share this
    /// one `vace_prep`, so this pins the wiring for `wan_vace` and `wan2_2_vace_fun_14b` together.
    #[test]
    fn vace_prep_binds_scales_to_the_shared_resolver() {
        let source = include_str!("model_vace.rs");
        let body = source
            .split("#[cfg(test)]")
            .next()
            .expect("production body precedes the test module");
        assert!(
            body.contains("vace_control_scales(req, config.vace_layers.len())"),
            "`scales` must be resolved by the shared seam"
        );
        assert!(
            !body.contains("control_scale.unwrap_or("),
            "the pre-sc-20261 expression must not be reachable at the call site"
        );
    }

    /// sc-20261 — `start_frame` was silently dropped on both MLX VACE routes while the candle
    /// sibling refused it. Mirror the sibling: non-zero is the typed `Unsupported`, `0` validates.
    #[test]
    fn validate_refuses_non_zero_start_frame_like_the_sibling() {
        let cfg = test_config();
        for (descriptor, id) in [
            (descriptor_vace(), MODEL_ID_VACE),
            (descriptor_vace_fun(), MODEL_ID_VACE_FUN),
        ] {
            assert!(validate_vace_clip(
                &descriptor,
                id,
                &cfg,
                &clip_request_with(1.0, 0, ReplacementMode::FaceOnly)
            )
            .is_ok());
            let err = validate_vace_clip(
                &descriptor,
                id,
                &cfg,
                &clip_request_with(1.0, 1, ReplacementMode::FaceOnly),
            )
            .expect_err("start_frame != 0 must be refused");
            assert!(
                matches!(err, Error::Unsupported(_)),
                "{id}: typed Unsupported, got {err:?}"
            );
            assert!(
                err.to_string().contains("start_frame"),
                "{id} names it: {err}"
            );
        }
    }

    /// sc-20261 — `mode` is realized in the worker-rasterized mask, so every replacement mode keeps
    /// taking the same VACE mask path (the sibling's documented rule). Not a refusal.
    #[test]
    fn every_replacement_mode_still_validates() {
        let cfg = test_config();
        for mode in [
            ReplacementMode::FaceOnly,
            ReplacementMode::FullPersonKeepOutfit,
            ReplacementMode::FullPersonReplaceOutfit,
        ] {
            assert!(validate_vace_clip(
                &descriptor_vace(),
                MODEL_ID_VACE,
                &cfg,
                &clip_request_with(1.0, 0, mode)
            )
            .is_ok());
        }
    }

    /// sc-12607 — VACE renders on the 14B family's `patch(2)·VAE_S(8)` = 16-px grid; candle rejects an
    /// off-16 request, so mlx must too (it used to only `align_dim` it down in `vace_prep`). The reject
    /// fires before the ControlClip/frame checks, so an off-grid size is refused even with a valid clip.
    #[test]
    fn validate_vace_rejects_off_grid_size() {
        let cfg = test_config();
        let d = descriptor_vace();
        let v = |w, h| validate_vace_clip(&d, MODEL_ID_VACE, &cfg, &clip_request(5, w, h, 0));
        // 72 is off the 16-px grid (72 = 4.5·16) — rejected, not snapped to 64.
        let err = v(72, 64).expect_err("72 is off the 16-px grid").to_string();
        assert!(err.contains("multiples of 16"), "unexpected: {err}");
        // Off-grid on the height axis too.
        assert!(v(64, 72).is_err());
        // On-grid 64×64 still validates.
        assert!(v(64, 64).is_ok());
    }
}
