//! `mlx-gen-wan` model entries: the Wan2.2 **TI2V-5B** (`wan2_2_ti2v_5b`, dense, z48 VAE — fully
//! wired, sc-2680), the Wan2.2 **T2V-A14B** (`wan2_2_t2v_14b`, dual-expert MoE, z16 VAE — fully wired
//! here on the S1–S5 core), and the Wan2.2 **I2V-A14B** (`wan2_2_i2v_14b`, dual-expert MoE,
//! channel-concat image conditioning, in_dim 36 — sc-2681), plus their registry self-registration.
//!
//! The 5B [`Wan`] struct runs the complete dense pipeline (sc-2680) — [`Wan::generate`]: UMT5-XXL
//! encode → the dense [`denoise`] (T2V) or the [`denoise_ti2v`] per-token mask-blend (TI2V, single- or
//! multi-keyframe) → z48 VAE decode → RGB8 frames, with Q4/Q8 + LoRA. The shared [`Wan14b`] struct
//! serves both A14B variants — [`Wan14b::generate`] runs the
//! complete pipeline: UMT5-XXL encode → (I2V only) build the channel-concat conditioning `y` →
//! per-step dual-expert MoE denoise (boundary-switched high/low experts, [`denoise_moe`]) → z16 VAE
//! decode → RGB8 frames, **staging** each heavy component (T5, the two 27 GB experts, the VAE) in and
//! out to bound peak memory (mirrors `generate_wan.py`). The I2V variant differs only by the `y`
//! conditioning (the image's first-frame VAE latent + temporal mask, channel-concatenated to in_dim
//! 36) and the max-area resolution cap.

use std::path::PathBuf;

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
use crate::pipeline::{
    align_dim, auto_tiling_budgeted, auto_tiling_budgeted_z16, build_i2v_y, build_ti2v_keyframe_z,
    build_ti2v_mask, build_ti2v_multi_mask, crossing_index, decode_to_frames, decode_to_frames_22,
    denoise, denoise_curated, denoise_moe, denoise_moe_curated, denoise_moe_curated_swapped,
    denoise_range, denoise_ti2v, frames_to_images, latent_shape, preflight_denoise_memory_guard,
    preprocess_ti2v_image, reject_off_grid, reject_over_area, resolve_sampler_knobs, seq_len,
    staged_expert_swap, ti2v_blend_init, Expert,
};
use crate::scheduler::{make_scheduler, SolverKind, WanScheduler};
use crate::text_encoder::encode_text_staged_for_tier;
use crate::transformer::WanTransformer;
use crate::vae::WanVae;
use crate::vae22::Wan22Vae;

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
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::Keyframe],
            // Q4/Q8 (sc-2682) loads via `spec.quantize` (transformer-only); LoRA/LoKr merge onto the
            // single dense model at generate time (the reference `_loras_single` path — shared
            // untagged specs only, reusing the sc-2683/sc-2393 `merge_wan_adapters` seam).
            supports_lora: true,
            supports_lokr: true,
            samplers: wan_samplers(),
            schedulers: Vec::new(),
            // H/W align to patch×vae_stride = 32 (`reject_off_grid`); floor each side at MIN_SIZE = 480
            // (= 15·32 — the z48 vae22 renders garbage below a 15×15 latent grid, sc-10306/sc-12636),
            // matching candle; cap the long edge at 1280 (max_area 704×1280).
            supported_guidance_methods: vec![],
            min_size: MIN_SIZE,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
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
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
        },
    }
}

/// The projection width the UMT5 text encoder packs to on a quantized tier: **Q8** (sc-12831). Q8 is
/// near-lossless for this drift-sensitive encoder — measured prompt-embedding cosine **0.9998** vs the
/// bf16 baseline on the real 5B TE, vs **0.976** at Q4 (a visible ~12° shift; the "biggest numeric-drift
/// risk" the candle sc-12757 finding flagged). Q8 already cuts the TE-encode active peak **11.83 → 7.72
/// GiB** — well under the epic's <16 GB-with-margin target — so we floor the TE at Q8 even when the DiT
/// tier is Q4: the user's Q4 *DiT* creative choice is untouched, and the TE stays regression-free (the
/// extra ~2 GiB a Q4 TE would save is not needed and not worth the drift).
const TE_QUANT_BITS: i32 = 8;

/// The effective UMT5 text-encoder quantization for a Wan tier (sc-12831). The DiT is packed on an
/// MLX-affine tier iff a pre-quantized snapshot manifest is present (`config.quantization`) **or** a
/// load-time `Q4`/`Q8` `spec.quantize` was requested (`Nvfp4` is candle-only and never reaches the MLX
/// Wan path — excluded so its `bits()` = 4 is not routed through `mlx quantize`). On such a tier the TE
/// packs to [`TE_QUANT_BITS`] (Q8), retiring the residual ~12 GiB f32-TE-encode active peak (sc-12796)
/// that no further component offload could lower — the epic's binding 5B constraint. On the bf16 tier
/// this is `None` (the TE stays dense / bit-exact). Shared by the 5B [`Wan`], the A14B [`Wan14b`], and
/// the VACE paths (`model_vace.rs`).
pub(crate) fn effective_te_quant(
    config: &WanModelConfig,
    load_quant: Option<Quant>,
) -> Option<WanQuant> {
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
    Ok(Box::new(Wan {
        descriptor: descriptor(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
        offload_policy: spec.offload_policy,
    }))
}

mlx_gen::impl_generator!(Wan {
    validate: |s, req| s.validate_impl(req),
    generate: generate_impl,
});

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
        }
        // Reject an off-grid or over-area geometry rather than silently align-down refitting it — one
        // request means one geometry on both backends. sc-12607: the 32-px grid stride (candle rejects
        // via `is_multiple_of(SIZE_MULTIPLE)`). sc-12308: the 5B's OWN 901 120 area budget — its 32-px
        // grid makes 1280×704 the geometry it genuinely renders.
        let (dw, dh) = grid(&self.config);
        reject_off_grid(MODEL_ID, req, dw, dh)?;
        reject_over_area(MODEL_ID, req, dw, dh, self.config.max_area)?;
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
    /// quantize), embed the contexts, run the dense [`denoise`] (T2V) or [`denoise_ti2v`] mask-blend
    /// loop; (3) load the vae22 decoder → RGB8 frames. CFG runs with the single guidance scale.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // Reject anything outside the advertised surface before doing expensive work — in particular
        // an unknown `sampler`, which `solver_kind` would otherwise silently map to UniPC.
        self.validate(req)?;
        let cfg = &self.config;
        // Sequential offload (epic 12732, sc-12796): the dense render is already staged (TE → DiT → z48
        // VAE, each loaded → used → dropped in turn), so there is no expert swap. Under `Sequential`,
        // `clear_cache`-flush each dead component's buffers off-GPU as soon as it is dropped, so the
        // freed UMT5 TE / VAE-encoder don't linger in MLX's buffer cache (RSS / wired footprint) through
        // the denoise + decode that follow. `Resident` (default) leaves the cache warm — the
        // byte-identical pre-offload path (residency/lifetime change only, numerics untouched).
        let sequential = self.offload_policy == OffloadPolicy::Sequential;

        // --- Resolve request knobs against config defaults ---
        let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
        let trim = req.trim_first_frames.unwrap_or(0) as usize;
        let trim_out = trim * cfg.vae_stride.0; // discarded output frames = trim · 4
        let gen_frames = frames + trim_out;
        let (width, height) = resolve_capped_dims(req, cfg);
        let (steps, shift, kind, seed) =
            resolve_sampler_knobs(req, cfg.sample_steps, cfg.sample_shift);
        // The 5B is dense → a single guidance scale (config Single(5.0), overridable per request).
        let guidance = cfg.sample_guide_scale.resolve_single(req.guidance);
        let cfg_disabled = guidance <= 1.0;
        let neg_prompt = req
            .negative_prompt
            .clone()
            .unwrap_or_else(|| cfg.sample_neg_prompt.clone());

        let lat = latent_shape(gen_frames, height, width, cfg.vae_z_dim, cfg.vae_stride)?;

        // sc-4986 — fail fast (catchable) if the DiT-denoise stage won't fit, before any heavy load.
        // sc-12796: this is policy-INdependent for the dense 5B (unlike the A14B's `moe_denoise_resident_bytes`,
        // whose resident-expert count is 1 under the swap vs 2 otherwise). The 5B holds exactly ONE DiT
        // resident during denoise under both `Resident` and `Sequential` — the TE/VAE are staged out
        // either way; `Sequential` only additionally `clear_cache`-flushes their dead buffers — so the
        // single-DiT byte count is already the correct budget for both policies.
        preflight_denoise_memory_guard(
            self.descriptor.id,
            dit_resident_bytes(&[self.root.join("model.safetensors")], self.quant),
            seq_len(lat, cfg.patch_size),
            cfg.dim,
            !cfg_disabled,
        )?;

        // sc-4998 — pick the memory-budgeted z48 vae22 decode tiling now, so an over-budget decode
        // (the 60 GB / 4.3 min blowup the px-threshold `auto` produced at moderate res) fails
        // catchably *before* the heavy denoise rather than OOM-ing in the post-loop decode stage.
        // sc-5039 — the decode runs **bf16** (visually lossless, cosine 0.999954 real-weight; lower
        // peak ⇒ the budget fits bigger tiles), so the plan uses the bf16 cost coefficient.
        let decode_tiling =
            auto_tiling_budgeted(height as i32, width as i32, gen_frames as i32, true)?;

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
        let encode_kf = |vae: &Wan22Vae, image: &Image| -> Result<Array> {
            let img_thwc = preprocess_ti2v_image(image, width, height)?; // [1,1,H,W,3]
            let z = vae.encode(&img_thwc)?; // [1,1,h,w,z]
            Ok(z.reshape(&z.shape()[1..])?.transpose_axes(&[3, 0, 1, 2])?) // [z,1,h,w]
        };
        let (t_lat, h_lat, w_lat) = (lat[1] as usize, lat[2] as usize, lat[3] as usize);
        let keyframes = req.keyframes();
        let (latents_init, ti2v) = if !keyframes.is_empty() {
            // Wan-native first_last_frame / multi-keyframe (sc-3357): pin each Keyframe's latent frame
            // via the mask-blend (frame_idx is a latent index, negative-from-end → `-1` = last frame).
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = Wan22Vae::from_weights(&w)?;
            let mut frames: Vec<(Array, usize)> = Vec::with_capacity(keyframes.len());
            let mut indices: Vec<usize> = Vec::with_capacity(keyframes.len());
            for kf in &keyframes {
                let idx = if kf.frame_idx < 0 {
                    t_lat as i32 + kf.frame_idx
                } else {
                    kf.frame_idx
                };
                if idx < 0 || idx as usize >= t_lat {
                    return Err(Error::Msg(format!(
                        "wan2_2_ti2v_5b: keyframe latent frame index {} out of bounds for {t_lat} \
                         latent frames",
                        kf.frame_idx
                    )));
                }
                frames.push((encode_kf(&vae, kf.image)?, idx as usize));
                indices.push(idx as usize);
            }
            let z = build_ti2v_keyframe_z(&frames, cfg.vae_z_dim, t_lat, h_lat, w_lat)?;
            let (mask, mask_tokens) =
                build_ti2v_multi_mask(&indices, cfg.vae_z_dim, t_lat, h_lat, w_lat, cfg.patch_size);
            let latents = ti2v_blend_init(&z, &mask, &init_noise)?;
            mlx_rs::transforms::eval([&latents, &z])?;
            (latents, Some((z, mask, mask_tokens)))
        } else {
            match i2v_reference(req) {
                Some(image) => {
                    let z_img = {
                        let w = Weights::from_file(self.root.join("vae.safetensors"))?;
                        let vae = Wan22Vae::from_weights(&w)?;
                        encode_kf(&vae, image)?
                    };
                    let (mask, mask_tokens) =
                        build_ti2v_mask(cfg.vae_z_dim, t_lat, h_lat, w_lat, cfg.patch_size);
                    let latents = ti2v_blend_init(&z_img, &mask, &init_noise)?;
                    mlx_rs::transforms::eval([&latents, &z_img])?;
                    (latents, Some((z_img, mask, mask_tokens)))
                }
                None => (init_noise.clone(), None),
            }
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
                (None, true) => denoise_curated(
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
                )?,
                (None, false) => {
                    let mut on_step = |i: usize| {
                        on_progress(Progress::Step {
                            current: i as u32,
                            total,
                        })
                    };
                    denoise(
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
        on_progress(Progress::Decoding);
        // Causal temporal decode: t_lat → 1 + (t_lat−1)·4 output frames (= gen_frames). The tiling
        // was chosen (and budget-checked) up front by `auto_tiling_budgeted` (sc-4998). sc-5039 casts
        // the decoder weights to bf16 (the conv-heavy body runs bf16, RMS_norm + denorm stay f32) —
        // `Wan22Vae` infers its compute dtype from the loaded weights. Decode-only: the conditioning
        // image was already encoded above through a separate f32 VAE load.
        let frames_u8 = {
            let mut w = Weights::from_file(self.root.join("vae.safetensors"))?;
            w.cast_all(mlx_rs::Dtype::Bfloat16)?;
            let vae = Wan22Vae::from_weights(&w)?;
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
        id: MODEL_ID_T2V_14B,
        family: "wan",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // CFG with the per-expert (low, high) guidance pair + the Chinese anti-artifact negative
            // prompt. Pure text→video: no image conditioning.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            // LoRA + LoKr merge per-expert at generate time (sc-2683 / sc-2393, PEFT/kohya + LoKr,
            // MoE high/low); Q4/Q8 (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot.
            supports_lora: true,
            supports_lokr: true,
            samplers: wan_samplers(),
            schedulers: Vec::new(),
            // H/W align to patch×vae_stride = 16 (z16 VAE, spatial stride 8); long edge cap 1280.
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Cross-attention text K/V is cached across denoise steps (per expert).
            supports_kv_cache: true,
            requires_sigma_shift: false,
            // A14B honors `OffloadPolicy::Sequential` (epic 12732, sc-12736): the staged expert swap
            // holds only the ACTIVE MoE expert resident (never both) and frees the UMT5 TE / VAE
            // off-GPU during denoise, dropping the unified-memory peak to ~one expert. Advertised so
            // the worker's fit-gate can tell "bounds peak here" from a no-op fallback.
            supports_sequential_offload: true,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
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
    Ok(Box::new(Wan14b {
        descriptor: descriptor_t2v_14b(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
        offload_policy: spec.offload_policy,
    }))
}

mlx_gen::impl_generator!(Wan14b {
    validate: |s, req| s.validate_impl(req),
    generate: generate_impl,
});

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
        }
        // Reject an off-grid or over-area geometry rather than silently align-down refitting it.
        // sc-12607: the 16-px grid stride (candle rejects via `is_multiple_of(SIZE_MULTIPLE_14B)`).
        // sc-12308: the area cap (T2V was uncapped entirely; I2V refit 1280×720 → 1264×704). Both A14B
        // variants share the 14B family's 921 600 budget.
        let (dw, dh) = grid(&self.config);
        reject_off_grid(id, req, dw, dh)?;
        reject_over_area(id, req, dw, dh, self.config.max_area)?;
        // I2V channel-concat requires a single reference image (the first conditioning frame), and
        // does not support `trim_first_frames` (the reference builds `y` from `num_frames`, so an
        // extended noise length would mismatch the conditioning's temporal dim).
        if self.config.is_i2v_concat() {
            if i2v_reference(req).is_none() {
                return Err(Error::Msg(format!(
                    "{id}: image-to-video requires a Reference conditioning image"
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
        let cfg = &self.config;
        // Sequential offload (epic 12732, sc-12736): free the UMT5 TE / VAE off-GPU during denoise and
        // hold only the ACTIVE MoE expert resident (the expert swap). `Resident` (default) is the
        // byte-identical pre-swap path (both experts co-resident, no per-stage `clear_cache`).
        let sequential = self.offload_policy == OffloadPolicy::Sequential;

        // --- Resolve request knobs against config defaults ---
        let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
        // trim_first_frames: generate `trim` extra leading temporal chunks (each = vae_stride_t = 4
        // latent frames → 4 output frames after the non-causal T→4T decode) and discard them after
        // decode, so the first kept frame sees a full temporal receptive field (port of
        // generate_wan.py). gen_frames stays 1+4k since frames is and we add a multiple of 4.
        let trim = req.trim_first_frames.unwrap_or(0) as usize;
        let trim_out = trim * cfg.vae_stride.0; // discarded output frames = trim · 4
        let gen_frames = frames + trim * cfg.vae_stride.0;
        // validate() already rejected sub-tile + bad frame counts; round H/W down to the grid, then
        // enforce the model's max-area cap (I2V-14B / TI2V-5B: 704×1280; no-op for T2V's max_area 0).
        let (width, height) = resolve_capped_dims(req, cfg);
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
        let lat = latent_shape(gen_frames, height, width, cfg.vae_z_dim, cfg.vae_stride)?;

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
        preflight_denoise_memory_guard(
            self.descriptor.id,
            wan14b_denoise_resident_bytes(
                self.offload_policy,
                req.sampler.as_deref(),
                low_bytes,
                high_bytes,
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
            let image = i2v_reference(req).ok_or_else(|| {
                Error::Msg(format!(
                    "{}: image-to-video requires a Reference conditioning image",
                    self.descriptor.id
                ))
            })?;
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
            let y = build_i2v_y(&vae, image, frames, height, width, cfg.vae_stride)?;
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
        let out_frames = lat[1] * cfg.vae_stride.0 as i32;
        let tiling = auto_tiling_budgeted_z16(height as i32, width as i32, out_frames)?;
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
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
    let ratio = match quant.map(|q| q.bits()) {
        Some(4) => 0.30, // 4-bit affine: ~0.5 B/param + scales vs bf16 2 B/param
        Some(8) => 0.55, // 8-bit affine: ~1 B/param + scales
        _ => 1.0,
    };
    let raw: u64 = files.iter().map(|p| weight_bytes_at(p)).sum();
    (raw as f64 * ratio) as u64
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

/// The single conditioning reference image for I2V (the first video frame), if present.
fn i2v_reference(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
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
fn resolve_capped_dims(req: &GenerationRequest, cfg: &WanModelConfig) -> (u32, u32) {
    let width = align_dim(req.width, cfg.patch_size.2, cfg.vae_stride.2);
    let height = align_dim(req.height, cfg.patch_size.1, cfg.vae_stride.1);
    (width, height)
}

/// The model's `(dw, dh)` pixel grid = `patch · vae_stride` — the lattice every Wan geometry sits on.
fn grid(cfg: &WanModelConfig) -> (u32, u32) {
    (
        (cfg.patch_size.2 * cfg.vae_stride.2) as u32,
        (cfg.patch_size.1 * cfg.vae_stride.1) as u32,
    )
}

/// Stable identity + advertised capabilities for the Wan2.2 I2V-A14B (dual-expert MoE image→video).
/// Identical to the T2V-A14B but advertises a single `Reference` conditioning image (the channel-
/// concat first frame) and the (3.5, 3.5) per-expert guidance.
pub fn descriptor_i2v_14b() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID_I2V_14B,
        family: "wan",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // A single image is channel-concatenated as the first-frame conditioning (in_dim 36).
            conditioning: vec![ConditioningKind::Reference],
            // LoRA + LoKr merge per-expert at generate time (sc-2683 / sc-2393, PEFT/kohya + LoKr,
            // MoE high/low); Q4/Q8 (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot.
            supports_lora: true,
            supports_lokr: true,
            samplers: wan_samplers(),
            schedulers: Vec::new(),
            // H/W align to patch×vae_stride = 16 (z16 VAE, spatial stride 8); long edge cap 1280.
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: true,
            requires_sigma_shift: false,
            // A14B honors `OffloadPolicy::Sequential` (epic 12732, sc-12736): the staged expert swap
            // holds only the ACTIVE MoE expert resident (never both) and frees the UMT5 TE / VAE
            // off-GPU during denoise, dropping the unified-memory peak to ~one expert. Advertised so
            // the worker's fit-gate can tell "bounds peak here" from a no-op fallback.
            supports_sequential_offload: true,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
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
    Ok(Box::new(Wan14b {
        descriptor: descriptor_i2v_14b(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
        offload_policy: spec.offload_policy,
    }))
}

mlx_gen::register_generators! {
    pub(crate) const I2V_14B_REGISTRATION = descriptor_i2v_14b => load_i2v_14b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "x".into(),
            width,
            height,
            ..Default::default()
        }
    }

    #[test]
    fn resolve_capped_dims_aligns_down_only() {
        // 14B aligns to patch.{1,2}·vae_stride.{1,2} = 2·8 = 16, so 130 → 128 on both axes.
        let cfg = WanModelConfig::wan22_t2v_14b();
        assert_eq!(resolve_capped_dims(&req(130, 130), &cfg), (128, 128));
        // Already on-grid → unchanged.
        assert_eq!(resolve_capped_dims(&req(128, 256), &cfg), (128, 256));
        // The 5B's z48 VAE gives a 32-px grid instead.
        let five_b = WanModelConfig::wan22_ti2v_5b();
        assert_eq!(resolve_capped_dims(&req(720, 720), &five_b), (704, 704));
        assert_eq!(resolve_capped_dims(&req(512, 512), &five_b), (512, 512));
    }

    #[test]
    fn effective_te_quant_floors_at_q8_when_dit_is_quantized() {
        // sc-12831: the UMT5 TE packs to Q8 (the near-lossless floor) whenever the DiT is on an
        // MLX-affine quantized tier, and stays dense (None) otherwise. Pins the NON-default width (Q8,
        // not the DiT's) so tier-matching the DiT to Q4 — or dropping the Nvfp4 exclusion — goes red.
        let expect_q8 = Some(WanQuant {
            bits: 8,
            group_size: mlx_gen::quant::DEFAULT_GROUP_SIZE,
        });
        let mut dense = WanModelConfig::wan22_ti2v_5b();
        dense.quantization = None; // a dense bf16 snapshot

        // bf16 tier (dense snapshot, no load-time quant) → TE stays dense.
        assert_eq!(effective_te_quant(&dense, None), None);
        // Dense snapshot + a load-time Q4/Q8 `spec.quantize` → TE packs, ALWAYS at Q8 (not the DiT's 4).
        assert_eq!(effective_te_quant(&dense, Some(Quant::Q4)), expect_q8);
        assert_eq!(effective_te_quant(&dense, Some(Quant::Q8)), expect_q8);
        // Nvfp4 is candle-only (never an MLX affine width) → excluded, TE stays dense.
        assert_eq!(effective_te_quant(&dense, Some(Quant::Nvfp4)), None);

        // A pre-quantized q4 snapshot (`config.quantization` = bits 4) → the TE STILL floors at Q8, not 4.
        let mut prequant_q4 = dense.clone();
        prequant_q4.quantization = Some(WanQuant {
            bits: 4,
            group_size: 64,
        });
        assert_eq!(effective_te_quant(&prequant_q4, None), expect_q8);
    }

    /// sc-12308 — this replaces `resolve_capped_dims_enforces_max_area_cap`, which asserted that an
    /// over-area request SHRINKS. It no longer does: the cap is enforced by rejecting in
    /// `validate_impl`, so this function only aligns. Alignment rounds *down*, so it can never push
    /// a validated request back over budget — which is why the cap check can live at validate time.
    #[test]
    fn resolve_capped_dims_never_refits_over_area() {
        let cfg = WanModelConfig::wan22_ti2v_5b();
        // The geometry `validate_impl` would have rejected passes through UNCHANGED (bar alignment)
        // rather than being silently refit to something the caller never asked for.
        assert_eq!(resolve_capped_dims(&req(2048, 2048), &cfg), (2048, 2048));
        assert!(
            2048 * 2048 > cfg.max_area,
            "the guard belongs to validate, not to this function"
        );

        // The 14B family's canonical 720p is at its cap and is a fixed point of the alignment.
        let a14b = WanModelConfig::wan22_i2v_14b();
        assert_eq!(resolve_capped_dims(&req(1280, 720), &a14b), (1280, 720));
        assert_eq!(1280 * 720, a14b.max_area);
    }

    /// sc-12736 + sc-12796 (epic 12732): ALL three Wan2.2 MLX variants advertise
    /// `supports_sequential_offload` so the worker's fit-gate can tell "bounds footprint under
    /// `Sequential`" from a no-op fallback — the two A14B MoE variants via the expert swap (sc-12736),
    /// the dense TI2V-5B via the staged-component `clear_cache` flush (sc-12796). Mutation guard:
    /// flipping ANY of these back to `false` regresses the advertisement its offload path relies on.
    #[test]
    fn all_wan_mlx_variants_advertise_sequential_offload() {
        assert!(
            descriptor_t2v_14b()
                .capabilities
                .supports_sequential_offload,
            "T2V-A14B must advertise the expert-swap sequential offload"
        );
        assert!(
            descriptor_i2v_14b()
                .capabilities
                .supports_sequential_offload,
            "I2V-A14B must advertise the expert-swap sequential offload"
        );
        assert!(
            descriptor().capabilities.supports_sequential_offload,
            "the dense TI2V-5B must advertise sequential offload (sc-12796) — the staged TE/VAE \
             clear_cache flush bounds its unified-memory footprint"
        );
    }

    /// sc-12736: the sc-4986 pre-flight denoise guard's expert-byte accounting must track the ACTUAL
    /// resident-expert count of the path it guards — one under every `Sequential` swap and two under
    /// `Resident`. Getting this wrong either
    /// false-rejects the swappable job on a one-expert machine (summing when it should be one) or waves
    /// through a both-resident job that then OOMs (one when it should be two). Distinct per-
    /// expert sizes (`lo < hi`) so `max` vs `sum` can't alias.
    #[test]
    fn preflight_guard_budgets_one_expert_for_every_sequential_sampler() {
        use OffloadPolicy::{Resident, Sequential};
        let (lo, hi) = (8_000_000_000u64, 9_000_000_000u64);

        // Resident → BOTH experts, whatever the sampler (native or curated).
        for s in [Some("uni_pc"), Some("heun"), None] {
            assert_eq!(
                wan14b_denoise_resident_bytes(Resident, s, lo, hi),
                lo + hi,
                "Resident must budget both experts for sampler {s:?}"
            );
        }

        // Sequential → ONE expert (the larger) for native and curated per-eval swap paths.
        for s in [
            Some("uni_pc"),
            Some("euler"),
            Some("dpmpp_2m"),
            Some("unipc"),
            Some("dpmpp2m"),
            Some("euler_ancestral"),
            Some("heun"),
            Some("dpmpp_sde"),
            Some("ddim"),
            None,
        ] {
            assert_eq!(
                wan14b_denoise_resident_bytes(Sequential, s, lo, hi),
                hi,
                "Sequential must budget ONE expert for sampler {s:?}"
            );
        }

        // Discriminator: the swap budgets strictly less than the both-resident path (the whole point).
        assert!(
            wan14b_denoise_resident_bytes(Sequential, Some("uni_pc"), lo, hi)
                < wan14b_denoise_resident_bytes(Resident, Some("uni_pc"), lo, hi),
            "the swap must budget strictly less than both-resident"
        );
    }

    fn wan_5b() -> Wan {
        Wan {
            descriptor: descriptor(),
            config: WanModelConfig::wan22_ti2v_5b(),
            root: PathBuf::new(),
            adapters: vec![],
            quant: None,
            offload_policy: OffloadPolicy::Resident,
        }
    }

    fn wan_t2v_14b() -> Wan14b {
        Wan14b {
            descriptor: descriptor_t2v_14b(),
            config: WanModelConfig::wan22_t2v_14b(),
            root: PathBuf::new(),
            adapters: vec![],
            quant: None,
            offload_policy: OffloadPolicy::Resident,
        }
    }

    /// sc-12607 — the 5B's z48 VAE renders on a 32-px grid; candle rejects an off-32 request, so mlx
    /// must too (it used to silently align it *down* to the nearest tile). Mutation guard: delete the
    /// `reject_off_grid` call and `req(720, 480)` validates (720 is in range, its aligned area fits) —
    /// this `unwrap_err` then panics.
    #[test]
    fn validate_rejects_off_grid_size_5b() {
        let wan = wan_5b();
        // On-grid, in range, under the 901 120 px cap → validates.
        assert!(wan.validate_impl(&req(704, 480)).is_ok());
        // 720 is off the 32-px grid (720 = 22.5·32) — rejected, not snapped to 704.
        let err = wan.validate_impl(&req(720, 480)).unwrap_err().to_string();
        assert!(err.contains("multiples of 32"), "unexpected: {err}");
        // Off-grid on the height axis is rejected the same way.
        assert!(wan.validate_impl(&req(704, 500)).is_err());
    }

    /// sc-12636 — the 5B's z48 vae22 renders rainbow garbage below a 15×15 latent grid (`< 480` px per
    /// side), so candle floors its descriptor `min_size` at [`MIN_SIZE`] = 480 and mlx must match, or an
    /// on-grid sub-480 request that candle *rejects* is instead *rendered as garbage* on mlx. This is a
    /// SEPARATE axis from the 32-px grid stride above: the sizes here are all on-grid (multiples of 32),
    /// so only the `min_size` range check (`gen_core::Capabilities::validate_request`) can reject them.
    /// Mutation guard: revert the descriptor `min_size` to 32 (or drop [`MIN_SIZE`] below 480) and
    /// `req(384, 384)` / `req(480, 320)` validate — the `unwrap_err` / `is_err` below then panic.
    #[test]
    fn validate_rejects_below_min_size_floor_5b() {
        use crate::config::{MIN_SIZE, SIZE_MULTIPLE};
        let wan = wan_5b();
        // At the floor: 480 = 15·32, on-grid, well under the 901 120 px cap → validates.
        assert!(wan.validate_impl(&req(480, 480)).is_ok());
        // 384 = 12·32 is on-grid but below the 480 coherence floor → rejected by the range check.
        let err = wan.validate_impl(&req(384, 384)).unwrap_err().to_string();
        assert!(
            err.contains("outside supported range") && err.contains(&MIN_SIZE.to_string()),
            "expected a min_size range rejection naming {MIN_SIZE}, got: {err}"
        );
        // Sub-floor on a single axis (320 = 10·32) is rejected too — both sides must clear the floor.
        assert!(wan.validate_impl(&req(480, 320)).is_err());
        // The rejected sizes are on-grid, so it is the floor — not the stride — doing the rejecting.
        assert_eq!(384 % SIZE_MULTIPLE, 0);
        assert_eq!(320 % SIZE_MULTIPLE, 0);
    }

    /// sc-12607 — the 14B family renders on a 16-px grid (`SIZE_MULTIPLE_14B`); same reject-not-refit
    /// contract as candle. Exercised on the T2V variant (no Reference required).
    #[test]
    fn validate_rejects_off_grid_size_14b() {
        let wan = wan_t2v_14b();
        // On-grid (240 = 15·16, 176 = 11·16), well under the 921 600 px cap → validates.
        assert!(wan.validate_impl(&req(240, 176)).is_ok());
        // 170 is off the 16-px grid (170 = 10.625·16) — rejected.
        let err = wan.validate_impl(&req(240, 170)).unwrap_err().to_string();
        assert!(err.contains("multiples of 16"), "unexpected: {err}");
        // Off-grid on the width axis too.
        assert!(wan.validate_impl(&req(250, 176)).is_err());
    }

    /// sc-12607/sc-12409/sc-12587 — the worker's `pinned_engine_geometry` tie reads these `pub const`s
    /// to assert each model's manifest stride. They must equal the lattice `validate`/`resolve_capped_dims`
    /// actually enforce (the config-derived `grid`), or the tie would anchor a stride the engine doesn't
    /// use. Mutation guard: change `SIZE_MULTIPLE` to 16 (or the config's `vae_stride`) and this fails.
    #[test]
    fn pinned_stride_consts_match_the_enforced_lattice() {
        use crate::config::{SIZE_MULTIPLE, SIZE_MULTIPLE_14B};
        assert_eq!(
            grid(&WanModelConfig::wan22_ti2v_5b()),
            (SIZE_MULTIPLE, SIZE_MULTIPLE)
        );
        assert_eq!(
            grid(&WanModelConfig::wan22_t2v_14b()),
            (SIZE_MULTIPLE_14B, SIZE_MULTIPLE_14B)
        );
        assert_eq!(
            grid(&WanModelConfig::wan22_i2v_14b()),
            (SIZE_MULTIPLE_14B, SIZE_MULTIPLE_14B)
        );
    }

    // ---- sc-12459: `dit_resident_bytes` shard-dir sizing must match `Weights::from_dir` ----------

    /// A fresh, empty scratch dir for one sizing test (recreated per run; RUST_TEST_THREADS=1 is
    /// forced repo-wide, so no cross-test races on the shared temp root).
    fn sizing_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join("mlx_gen_wan_dit_sizing")
            .join(name);
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Pins the symlink traversal (adversarial-review issue on sc-12459): the `link == true`
    /// assembled snapshots (convert.rs, the default local-dev / gated-e2e layout) and HF-hub caches
    /// store each shard as a symlink into `blobs/`. `DirEntry::metadata` does NOT traverse the link
    /// (a symlinked shard would sum as ~14 bytes → the preflight sees none of the expert weights and
    /// the load dies by SIGKILL, the original F-008 failure); `std::fs::metadata` must size the
    /// blob target instead.
    #[test]
    fn dit_resident_bytes_dir_sums_symlinked_shards_at_target_size() {
        let root = sizing_dir("symlinked_shards");
        let blobs = root.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let blob = blobs.join("blob0");
        std::fs::write(&blob, vec![0u8; 4096]).unwrap();

        let shard_dir = root.join("transformer");
        std::fs::create_dir_all(&shard_dir).unwrap();
        // One symlinked shard (the linked-snapshot / HF-cache layout) + one regular shard.
        std::os::unix::fs::symlink(&blob, shard_dir.join("model-00001-of-00002.safetensors"))
            .unwrap();
        std::fs::write(
            shard_dir.join("model-00002-of-00002.safetensors"),
            vec![0u8; 2048],
        )
        .unwrap();

        assert_eq!(
            dit_resident_bytes(&[shard_dir], None),
            4096 + 2048,
            "symlinked shard must size as its blob target, not the link"
        );
    }

    /// The dir branch applies the exact `Weights::from_dir` selection: hidden AppleDouble sidecars
    /// (`._model.safetensors` — extension-only filters admit them) and non-`.safetensors` files
    /// contribute nothing, so the sizing matches what the loader will actually map.
    #[test]
    fn dit_resident_bytes_dir_skips_hidden_sidecars_and_non_safetensors() {
        let shard_dir = sizing_dir("hidden_sidecars");
        std::fs::write(shard_dir.join("model.safetensors"), vec![0u8; 1024]).unwrap();
        std::fs::write(shard_dir.join("._model.safetensors"), vec![0u8; 999]).unwrap();
        std::fs::write(shard_dir.join("config.json"), b"{}").unwrap();

        assert_eq!(
            dit_resident_bytes(&[shard_dir], None),
            1024,
            "only the real shard `Weights::from_dir` maps may be counted"
        );
    }

    // ---- sc-10045: adapter routing at the two merge_adapters SITES (5B `Wan` + A14B `Wan14b`) ------
    //
    // These drive the real private `install_adapters_additive` / `merge_adapters` methods that the
    // generate paths call — with a *tiny* synthetic `WanTransformer` (in = 64 so the block Linears
    // quantize at group_size 64), never the ~7GB checkpoint. They prove: (a) a plain LoRA AND a LoKr
    // install on a PACKED base with NO error (the old model.rs:614 rejection is gone; LoKr via the
    // structured deferred-Kronecker path, sc-10050) with the base STILL packed; (b) LoHa on a packed
    // base is the explicit typed error (deferred to sc-10051); (c) the dense fold path is untouched.

    use crate::config::WanModelConfig;
    use crate::transformer::WanTransformer;
    use mlx_gen::adapters::AdaptableHost;
    use mlx_gen::runtime::AdapterKind;
    use mlx_rs::ops::array_eq;
    use mlx_rs::{Array, Dtype};
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// A tiny 1-layer Wan config: dim 64 (in = 64 is a group_size multiple, so the block Linears pack
    /// to Q4/Q8), 2 heads, ffn 128. Not run through forward — only built + adapter-installed.
    fn tiny_cfg() -> WanModelConfig {
        let mut c = WanModelConfig::wan21_t2v_1_3b();
        c.dim = 64;
        c.ffn_dim = 128;
        c.num_heads = 2;
        c.num_layers = 1;
        c.freq_dim = 16;
        c.text_dim = 64;
        c.in_dim = 4;
        c.out_dim = 4;
        c.quantization = None; // built dense; the test packs the linears after build
        c
    }

    fn tmp_dir() -> PathBuf {
        let d = std::env::temp_dir().join("mlx_gen_wan_model_site_test");
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn bf16(vals: impl Iterator<Item = f32>, shape: &[i32]) -> Array {
        Array::from_slice(&vals.collect::<Vec<_>>(), shape)
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    }

    /// Build a tiny dense [`WanTransformer`] from synthesized weights matching [`tiny_cfg`]. Every
    /// tensor `from_weights` + `Block::load` requires is present; the per-block attn/ffn Linears carry
    /// `in = 64`/`128` so a subsequent `.quantize(bits)` packs them (group 64).
    fn tiny_transformer(cfg: &WanModelConfig) -> WanTransformer {
        let dim = cfg.dim as i32;
        let ffn = cfg.ffn_dim as i32;
        let mut entries: Vec<(String, Array)> = Vec::new();
        fn lin(entries: &mut Vec<(String, Array)>, name: &str, out: i32, inp: i32, seed: f32) {
            entries.push((
                format!("{name}.weight"),
                bf16(
                    (0..out * inp).map(|i| (i as f32 * 0.001 + seed).sin() * 0.05),
                    &[out, inp],
                ),
            ));
            entries.push((
                format!("{name}.bias"),
                bf16((0..out).map(|i| i as f32 * 0.001), &[out]),
            ));
        }
        // Embeddings + head (shapes plausible; not exercised by forward here).
        lin(
            &mut entries,
            "patch_embedding_proj",
            dim,
            cfg.in_dim as i32,
            0.1,
        );
        lin(
            &mut entries,
            "text_embedding_0",
            dim,
            cfg.text_dim as i32,
            0.2,
        );
        lin(&mut entries, "text_embedding_1", dim, dim, 0.3);
        lin(
            &mut entries,
            "time_embedding_0",
            dim,
            cfg.freq_dim as i32,
            0.4,
        );
        lin(&mut entries, "time_embedding_1", dim, dim, 0.5);
        lin(&mut entries, "time_projection", dim * 6, dim, 0.6);
        lin(&mut entries, "head.head", cfg.out_dim as i32, dim, 0.7);
        entries.push((
            "head.modulation".into(),
            bf16((0..2 * dim).map(|i| i as f32 * 0.001), &[1, 2, dim]),
        ));
        // Block 0.
        entries.push((
            "blocks.0.modulation".into(),
            bf16((0..6 * dim).map(|i| i as f32 * 0.001), &[1, 6, dim]),
        ));
        for attn in ["self_attn", "cross_attn"] {
            for (p, seed) in [("q", 1.1), ("k", 1.2), ("v", 1.3), ("o", 1.4)] {
                lin(
                    &mut entries,
                    &format!("blocks.0.{attn}.{p}"),
                    dim,
                    dim,
                    seed,
                );
            }
            entries.push((
                format!("blocks.0.{attn}.norm_q.weight"),
                bf16((0..dim).map(|_| 1.0), &[dim]),
            ));
            entries.push((
                format!("blocks.0.{attn}.norm_k.weight"),
                bf16((0..dim).map(|_| 1.0), &[dim]),
            ));
        }
        entries.push((
            "blocks.0.norm3.weight".into(),
            bf16((0..dim).map(|_| 1.0), &[dim]),
        ));
        entries.push((
            "blocks.0.norm3.bias".into(),
            bf16((0..dim).map(|_| 0.0), &[dim]),
        ));
        lin(&mut entries, "blocks.0.ffn.fc1", ffn, dim, 1.5);
        lin(&mut entries, "blocks.0.ffn.fc2", dim, ffn, 1.6);

        let path = tmp_dir().join("tiny_wan.safetensors");
        let refs: Vec<(&str, &Array)> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        let w = Weights::from_file(&path).unwrap();
        WanTransformer::from_weights(&w, cfg).unwrap()
    }

    /// A PEFT LoRA file targeting `blocks.0.self_attn.q` ([dim,dim]) — `diffusion_model.`-prefixed,
    /// A `[rank,dim]`, B `[dim,rank]`, no alpha.
    fn tiny_lora(name: &str, dim: i32, rank: i32) -> PathBuf {
        let a = bf16(
            (0..rank * dim).map(|i| (i as f32 * 0.01).sin() * 0.03),
            &[rank, dim],
        );
        let b = bf16(
            (0..dim * rank).map(|i| (i as f32 * 0.007).cos() * 0.03),
            &[dim, rank],
        );
        let path = tmp_dir().join(name);
        Array::save_safetensors(
            vec![
                ("diffusion_model.blocks.0.self_attn.q.lora_A.weight", &a),
                ("diffusion_model.blocks.0.self_attn.q.lora_B.weight", &b),
            ],
            None,
            &path,
        )
        .unwrap();
        path
    }

    /// A peft LoKr file (networkType=lokr) targeting `blocks.0.self_attn.q` ([64,64] = kron([8,8],[8,8])).
    fn tiny_lokr(name: &str) -> PathBuf {
        let w1 = Array::from_slice(
            &(0..64)
                .map(|i| (i as f32 * 0.03).sin() * 0.1)
                .collect::<Vec<_>>(),
            &[8, 8],
        );
        let w2 = Array::from_slice(
            &(0..64)
                .map(|i| (i as f32 * 0.05).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[8, 8],
        );
        let meta = HashMap::from([
            ("networkType".to_string(), "lokr".to_string()),
            ("alpha".to_string(), "8".to_string()),
            ("rank".to_string(), "8".to_string()),
        ]);
        let path = tmp_dir().join(name);
        Array::save_safetensors(
            vec![
                ("blocks.0.self_attn.q.lokr_w1", &w1),
                ("blocks.0.self_attn.q.lokr_w2", &w2),
            ],
            Some(&meta),
            &path,
        )
        .unwrap();
        path
    }

    fn lora_spec(path: PathBuf) -> AdapterSpec {
        AdapterSpec {
            path,
            scale: 0.8,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }
    }

    /// Probe the built transformer's `blocks.0.self_attn.q` linear's quant state, for the "stays
    /// packed" assertions (the site methods must never dequantize the base).
    fn q_is_quantized(dit: &mut WanTransformer) -> bool {
        dit.adaptable_mut(&["blocks", "0", "self_attn", "q"])
            .unwrap()
            .is_quantized()
    }

    /// Construct a bare pre-quantized `Wan` (5B) with a config that reports pre-quantized, so the
    /// generate path would route to the additive install. `bits` mirrors a Q4/Q8 snapshot.
    fn wan_5b_prequant(adapters: Vec<AdapterSpec>) -> Wan {
        let mut cfg = tiny_cfg();
        cfg.quantization = Some(crate::config::WanQuant {
            bits: 8,
            group_size: 64,
        });
        Wan {
            descriptor: descriptor(),
            config: cfg,
            root: tmp_dir(),
            adapters,
            quant: None,
            offload_policy: OffloadPolicy::Resident,
        }
    }

    fn wan_14b_prequant(adapters: Vec<AdapterSpec>) -> Wan14b {
        let mut cfg = tiny_cfg();
        cfg.dual_model = true;
        cfg.quantization = Some(crate::config::WanQuant {
            bits: 8,
            group_size: 64,
        });
        Wan14b {
            descriptor: descriptor_t2v_14b(),
            config: cfg,
            root: tmp_dir(),
            adapters,
            quant: None,
            offload_policy: OffloadPolicy::Resident,
        }
    }

    #[test]
    fn site_5b_packed_lora_installs_additively_no_error() {
        // 5B `Wan::install_adapters_additive`: a plain LoRA installs on a PACKED base with no error;
        // the base stays packed and the q forward-linear gains a residual.
        let cfg = tiny_cfg();
        let mut dit = tiny_transformer(&cfg);
        dit.quantize(8, None).unwrap();
        assert!(q_is_quantized(&mut dit), "base packed before install");

        let model = wan_5b_prequant(vec![lora_spec(tiny_lora(
            "site5b.safetensors",
            cfg.dim as i32,
            8,
        ))]);
        model
            .install_adapters_additive(&mut dit)
            .expect("plain LoRA on a packed 5B must install with no error");
        assert!(
            q_is_quantized(&mut dit),
            "base must STAY packed after install (no dequant)"
        );
    }

    #[test]
    fn site_5b_packed_lokr_installs_structurally_no_error() {
        // 5B (sc-10050): LoKr on a packed base now installs via the structured deferred-Kronecker path
        // with NO error (the sc-10045 interim rejection is gone), and the base STAYS packed.
        let cfg = tiny_cfg();
        let mut dit = tiny_transformer(&cfg);
        dit.quantize(8, None).unwrap();
        assert!(q_is_quantized(&mut dit), "base packed before install");

        let mut spec = lora_spec(tiny_lokr("site5b_lokr.safetensors"));
        spec.kind = AdapterKind::Lokr;
        let model = wan_5b_prequant(vec![spec]);
        model
            .install_adapters_additive(&mut dit)
            .expect("LoKr on a packed 5B must install structurally with no error (sc-10050)");
        assert!(
            q_is_quantized(&mut dit),
            "base must STAY packed after structured LoKr install (no dequant)"
        );
    }

    #[test]
    fn site_14b_packed_lora_installs_on_both_experts() {
        // A14B `Wan14b::install_adapters_additive`: a shared plain LoRA installs onto BOTH packed
        // experts with no error; both bases stay packed.
        let cfg = tiny_cfg();
        let mut low = tiny_transformer(&cfg);
        let mut high = tiny_transformer(&cfg);
        low.quantize(4, None).unwrap();
        high.quantize(4, None).unwrap();

        let model = wan_14b_prequant(vec![lora_spec(tiny_lora(
            "site14b.safetensors",
            cfg.dim as i32,
            8,
        ))]);
        model
            .install_adapters_additive(&mut low, &mut high)
            .expect("plain LoRA on packed A14B experts must install with no error");
        assert!(
            q_is_quantized(&mut low) && q_is_quantized(&mut high),
            "both experts stay packed"
        );
    }

    #[test]
    fn site_14b_packed_lokr_installs_structurally_no_error() {
        // A14B (sc-10050): a shared LoKr installs onto BOTH packed experts via the structured
        // deferred-Kronecker path with NO error; both bases stay packed.
        let cfg = tiny_cfg();
        let mut low = tiny_transformer(&cfg);
        let mut high = tiny_transformer(&cfg);
        low.quantize(4, None).unwrap();
        high.quantize(4, None).unwrap();

        let mut spec = lora_spec(tiny_lokr("site14b_lokr.safetensors"));
        spec.kind = AdapterKind::Lokr;
        let model = wan_14b_prequant(vec![spec]);
        model.install_adapters_additive(&mut low, &mut high).expect(
            "LoKr on packed A14B experts must install structurally with no error (sc-10050)",
        );
        assert!(
            q_is_quantized(&mut low) && q_is_quantized(&mut high),
            "both experts stay packed after structured LoKr install"
        );
    }

    #[test]
    fn site_5b_packed_loha_is_explicit_typed_error() {
        // 5B: LoHa on a packed base is still rejected (LoHa is sc-10051, out of scope) — the actionable
        // error points at the bf16 tier + sc-10051, not a panic, not the old "not yet wired".
        let cfg = tiny_cfg();
        let mut dit = tiny_transformer(&cfg);
        dit.quantize(8, None).unwrap();

        // A committed third-party LoHa fixture (detected by `hada_*` keys).
        let loha = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests/fixtures/sc3643_loha/linear.safetensors");
        let model = wan_5b_prequant(vec![lora_spec(loha)]);
        let err = model
            .install_adapters_additive(&mut dit)
            .expect_err("LoHa on a packed 5B must be rejected (sc-10051)");
        let msg = err.to_string();
        assert!(
            msg.contains("bf16") && msg.contains("sc-10051") && msg.contains("LoHa"),
            "actionable LoHa-deferral msg, got: {msg}"
        );
        assert!(
            !msg.contains("not yet wired"),
            "not the old generic message, got: {msg}"
        );
    }

    #[test]
    fn site_14b_dense_fold_still_works() {
        // Dense-path regression: `Wan14b::merge_adapters` (the fold path) still folds a plain LoRA onto
        // both dense expert weight maps, unchanged by sc-10045.
        let cfg = tiny_cfg();
        let lora = tiny_lora("site14b_dense.safetensors", cfg.dim as i32, 8);
        // Build the two dense expert weight maps from the tiny synthetic base.
        let base_path = {
            let mut c = cfg.clone();
            c.dual_model = true;
            let _ = tiny_transformer(&c); // side-effect: writes tiny_wan.safetensors
            tmp_dir().join("tiny_wan.safetensors")
        };
        let mut low_w = Weights::from_file(&base_path).unwrap();
        let mut high_w = Weights::from_file(&base_path).unwrap();
        let before = low_w
            .require("blocks.0.self_attn.q.weight")
            .unwrap()
            .clone();

        let model = wan_14b_prequant(vec![lora_spec(lora)]);
        // Use a DENSE config for the fold path (quantization None).
        let dense = Wan14b {
            config: {
                let mut c = model.config.clone();
                c.quantization = None;
                c
            },
            ..model
        };
        dense
            .merge_adapters(&mut low_w, &mut high_w)
            .expect("dense fold must apply");
        let after = low_w.require("blocks.0.self_attn.q.weight").unwrap();
        assert!(
            !array_eq(after, &before, false).unwrap().item::<bool>(),
            "dense fold must change the q weight"
        );
    }
}
