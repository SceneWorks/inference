//! # candle-gen-z-image
//!
//! The **Z-Image** (Tongyi `Z-Image-Turbo`) provider crate for [`candle-gen`](candle_gen) — the
//! candle (Windows/CUDA) sibling of `mlx-gen-z-image`. It implements the backend-neutral
//! [`gen_core::Generator`] contract and exposes its variants through an explicit family catalog.
//!
//! **txt2img (sc-3693):** [`ZImageGenerator::generate`] adapts the `candle-transformers` `z_image`
//! reference model (`pipeline`) through the contract: Qwen3 text encoder → DiT (flow-match Euler,
//! distilled 4-step, **no CFG**) → AutoencoderKL VAE, emitting `Progress` and honoring `req.cancel`,
//! with **deterministic CPU-seeded noise** (sc-3673) so output is launch-portable per seed. The
//! prompt's Qwen chat-template wrapping reuses gen-core's `TextTokenizer` — the same template the
//! mlx provider uses (the epic-3692 "carries over via gen-core" reuse).
//!
//! The descriptor advertises the wired surface — txt2img, LoRA/LoKr (sc-5166), and reference-guided
//! **img2img** latent-init (sc-11783, a single `Conditioning::Reference`) — but NOT the shapes candle
//! doesn't serve through the registry (Q4/Q8 quant; the bespoke `edit_image` masked-edit + strict-pose
//! control worker streams), so the worker routes those elsewhere rather than the candle backend silently
//! dropping them (the false-capability trap, exactly as the SDXL slice sc-3675 did). The descriptor's
//! `backend` is `"candle"` and `mac_only` is `false`.
//!
//! Z-Image-Turbo is guidance-distilled: no classifier-free guidance, no negative prompt; the wired
//! sampler is the model's static-shift-3.0 flow-match Euler schedule. See `pipeline` for the parity
//! choices reconciled against the macOS `mlx-gen-z-image` provider.

mod adapters;
// ComfyUI single-file → candle in-memory remap seam (epic 10451 Phase 2, sc-10668): the DiT
// fused-qkv split + leaf renames and the VAE ldm→diffusers key/shape remap that make a ComfyUI
// Z-Image install's separate component files loadable in place via `VarBuilder::from_tensors`.
mod comfyui;
// Crate-private shared plumbing (sc-9002 / F-022): the loader, VAE decode → RGB8, `[0,255] → [-1,1]`
// image preprocess, deterministic VAE-encode mean, seeded-noise prior, and the Qwen tokenizer policy —
// one home for what the three entry points (pipeline/edit/control) used to triplicate.
mod common;
mod dit;
mod memory_strategy;
mod pipeline;
// The packed-load seam (sc-9408, sc-9089 umbrella): re-exports the shared `candle_gen::quant::QLinear`
// (F-025 / sc-9005) + the thin dense-or-packed `QEmbedding` wrapper over the shared module, plus the
// vendored inference DiT + Qwen3 TE that build their projections from it. Used only when the snapshot is
// a pre-quantized MLX-packed tier (`SceneWorks/z-image-turbo-mlx`); a dense snapshot keeps the stock
// candle-transformers models.
mod packed_dit;
mod packed_te;
// The per-step latent-preview seam (epic 16948, sc-16957): the REUSED epic-16624 Z-Image 16-channel
// fit plus the `[1, 16, 1, h, w]` → `[1, 16, h, w]` frame-axis drop that reaches it. Public because
// [`control`] and [`edit`] are themselves public, name-driven entry points, so a consumer that stages
// its own denoise against them needs the same projector rather than a second copy of the fit.
pub mod preview;
mod quant;
mod training;

// Base (non-Turbo) `z_image` text-to-image generator (sc-8414, the candle sibling of mlx sc-8320).
// Registers its own engine id `z_image` alongside the Turbo `z_image_turbo` below; it
// reuses the identical DiT/VAE/encoder + [`pipeline`] components, differing only in the render path —
// real classifier-free guidance over the static **shift=6.0** flow-match schedule (vs Turbo's
// CFG-free 4-step shift-3.0 distillation). The Turbo path is completely untouched (additive).
pub mod base;

// Fun-ControlNet (strict-pose) provider (sc-5489, epic 5480) — VACE-style dual-injection control on
// the vendored DiT (`alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`). Invoked directly by the
// worker (a bespoke pose stream), not gen-core-registered — the `z_image_turbo` descriptor stays
// txt2img-only.
pub mod control;

// Z-Image Fun-ControlNet real-weight GPU validation (sc-5489) — env-driven, `#[ignore]`d integration
// test (with-control vs no-control pixel diff + mid-denoise cancel).
#[cfg(test)]
mod control_validate;

// Z-Image **img2img / edit** (sc-6595, epic 5480) — the candle sibling of the MLX `z_image_turbo`
// `Conditioning::Reference` route. A bespoke provider driven directly by the worker (a `z_image_edit` /
// `z_image_turbo`+`edit_image` stream), like the strict-pose control above; the registered
// `z_image_turbo` descriptor stays txt2img-only (it can't promise img2img through the registry path).
pub mod edit;

// Z-Image img2img real-weight GPU validation (sc-6595) — env-driven, `#[ignore]`d integration test
// (strength ablation + the strength-1.0 source round-trip + mid-denoise cancel).
#[cfg(test)]
mod edit_validate;

// Base (non-Turbo) `z_image` img2img/`Reference` real-weight GPU validation (sc-8646) — env-driven,
// `#[ignore]`d integration test driving the REGISTERED base generator through a `Conditioning::Reference`
// (strength ablation + the strength-1.0 source round-trip + prompt divergence).
#[cfg(test)]
mod base_img2img_validate;

// `z_image_turbo` img2img/`Reference` real-weight GPU validation (sc-11783) — env-driven, `#[ignore]`d
// integration test driving the REGISTERED Turbo generator through a `Conditioning::Reference` (strength
// ablation + the strength-1.0 source round-trip + prompt divergence), the CFG-free sibling of
// `base_img2img_validate`.
#[cfg(test)]
mod turbo_img2img_validate;

pub use adapters::{install_additive, merge_adapters, AdditiveReport, MergeReport};
// Base (non-Turbo) `z_image` generator (sc-8414). Its `descriptor`/`load`/`MODEL_ID` share the names
// of the Turbo model's free functions below, so reach them through the `base` module path (consumers
// use the registry id `"z_image"`).
pub use base::ZImageBaseGenerator;
pub use control::{ZImageControl, ZImageControlPaths, ZImageControlRequest, DEFAULT_CONTROL_SCALE};
pub use edit::{ZImageEdit, ZImageEditPaths, ZImageEditRequest, DEFAULT_EDIT_STRENGTH};

/// Add every registry-owned Candle Z-Image provider to an explicit media registry builder.
///
/// Bespoke control/edit utilities remain direct worker integrations and are intentionally absent.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(REGISTRATION)
        .register_generator(base::REGISTRATION)
        .register_encoder_contract_route(gen_core::EncoderContractRouteRegistration {
            route_id: "z_image_turbo_control",
            provider_id: MODEL_ID,
        })
        .register_encoder_contract_route(gen_core::EncoderContractRouteRegistration {
            route_id: "z_image_control",
            provider_id: base::MODEL_ID,
        })
        .register_imported_model(gen_core::ImportedModelRegistration {
            family: "z-image",
            source: gen_core::ImportedModelSource::ComfyUiTree,
            operation: gen_core::ImportedModelOperation::Generate,
            provider_id: MODEL_ID,
            required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
            inherit_adapters: true,
        });
    #[cfg(feature = "cuda")]
    let registry = register_memory_contract_surfaces(registry)
        .register_memory_behavior(TURBO_MEMORY_BEHAVIOR)
        .register_memory_behavior(BASE_MEMORY_BEHAVIOR)
        .register_memory_behavior(TURBO_CONTROL_MEMORY_BEHAVIOR)
        .register_memory_behavior(BASE_CONTROL_MEMORY_BEHAVIOR);
    registry.register_trainer(training::REGISTRATION)
}

/// Register the exhaustive weights-free memory-contract surface on every build platform.
pub fn register_memory_contract_surfaces(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(TURBO_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            provider_id: MODEL_ID,
            contract: weights_free_turbo_memory_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: MODEL_ID,
                contract: weights_free_turbo_surface_contract,
            },
        )
        .register_memory_strategy(BASE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            provider_id: base::MODEL_ID,
            contract: registered_base_memory_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: base::MODEL_ID,
                contract: weights_free_base_surface_contract,
            },
        )
        .register_composed_memory_strategy(TURBO_CONTROL_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: control_memory_contract_surface_specs,
            provider_id: "z_image_turbo_control",
            contract: registered_turbo_control_memory_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: "z_image_turbo_control",
                contract: weights_free_turbo_control_surface_contract,
            },
        )
        .register_composed_memory_strategy(BASE_CONTROL_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: control_memory_contract_surface_specs,
            provider_id: "z_image_control",
            contract: registered_base_control_memory_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: "z_image_control",
                contract: weights_free_base_control_surface_contract,
            },
        )
}

fn control_memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .map(|mut surface| {
            surface.spec.control = Some(gen_core::WeightsSource::Dir(
                "/__sceneworks_memory_contract_control_surface__".into(),
            ));
            surface
        })
        .collect()
}

/// Build the complete explicit Candle Z-Image provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, PidWeights, Progress, Quant, SizeFloor,
    WeightsSource, BASE_SNAPSHOT_COMPONENT, COMFYUI_TEXT_ENCODER_COMPONENT, COMFYUI_VAE_COMPONENT,
};
use candle_transformers::models::z_image::vae::Encoder as VaeEncoder;

use pipeline::{Components, Pipeline, DEFAULT_STEPS};

/// Registry id — matches the SceneWorks worker's `payload.model` (`MODEL_TABLE["z_image_turbo"]`)
/// and the macOS `mlx-gen-z-image` descriptor.
pub const MODEL_ID: &str = "z_image_turbo";

pub const TOKENIZER_CONTRACT: gen_core::EncoderTokenizerContract =
    gen_core::EncoderTokenizerContract {
        family: "qwen3",
        binding: gen_core::EncoderTokenizerBinding::RetainBase,
        artifact_candidates: &["tokenizer/tokenizer.json"],
        required_tokens: &[
            gen_core::EncoderRequiredToken {
                role: "qwen_endoftext",
                literal: "<|endoftext|>",
                id: 151_643,
                config_field: Some("bos_token_id"),
            },
            gen_core::EncoderRequiredToken {
                role: "qwen_im_start",
                literal: "<|im_start|>",
                id: 151_644,
                config_field: None,
            },
            gen_core::EncoderRequiredToken {
                role: "qwen_im_end",
                literal: "<|im_end|>",
                id: 151_645,
                config_field: Some("eos_token_id"),
            },
        ],
    };

pub const PROMPT_EXECUTIONS: &[gen_core::EncoderPromptExecutionContract] = &[
    gen_core::EncoderPromptExecutionContract {
        purpose: "z_image_prompt",
        template: gen_core::EncoderPromptTemplate::QwenInstruct,
        add_special_tokens: true,
        length: gen_core::EncoderPromptLengthPolicy::RightTruncate { max_tokens: 512 },
        padding: gen_core::EncoderPromptPadding::None,
        prefix_trim: 0,
    },
    gen_core::EncoderPromptExecutionContract {
        purpose: "z_image_empty_negative",
        template: gen_core::EncoderPromptTemplate::QwenInstruct,
        add_special_tokens: true,
        length: gen_core::EncoderPromptLengthPolicy::Unbounded,
        padding: gen_core::EncoderPromptPadding::None,
        prefix_trim: 0,
    },
];

pub const ENCODER_CONTRACT: gen_core::EncoderContract = gen_core::EncoderContract {
    architecture: "qwen3",
    hidden_size: 2560,
    intermediate_size: 9728,
    num_hidden_layers: 36,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    head_dim: 128,
    vocab_size: 151_936,
    output_width: 2560,
    loaded_hidden_layers: 36,
    requires_final_norm: false,
    requires_lm_head: false,
    hidden_activation: "silu",
    attention_dropout: gen_core::EncoderConfigFloat::new(0.0),
    rms_norm_eps: gen_core::EncoderConfigFloat::new(1e-6),
    qk_norm_eps: Some(gen_core::EncoderConfigFloat::new(1e-6)),
    rope_theta: gen_core::EncoderConfigFloat::new(1_000_000.0),
    max_position_embeddings: 40_960,
    attention_bias: gen_core::EncoderConfigBool::Required(false),
    tie_word_embeddings: gen_core::EncoderConfigBool::Required(true),
    tokenizer: TOKENIZER_CONTRACT,
    prompt_executions: PROMPT_EXECUTIONS,
    bos_token_id: Some(151_643),
    eos_token_id: Some(151_645),
    image_token_id: None,
    vision_start_token_id: None,
    vision_end_token_id: None,
    mrope_section: &[],
    mrope_interleaved: None,
    selected_hidden_layers: &[35],
    packing: Some(gen_core::EncoderPackingContract {
        group_size: 64,
        pack_embedding: true,
        pack_lm_head: false,
        supports_file: true,
    }),
    dense_storage_dtype_probe: None,
};

/// Z-Image works in latent space at /8 and the DiT patchifies that at /2, so both image dims must be
/// multiples of **16** for a clean patchify. Enforced in [`validate`](Generator::validate). Exposed as
/// the pinned-engine stride SceneWorks ties each advertised Z-Image bucket to (sc-12612), mirroring
/// `wan::config::SIZE_MULTIPLE_14B`; the base + edit modules import this same crate-root const so no
/// copy can drift from the check.
pub const SIZE_MULTIPLE: u32 = 16;

/// Process-global accelerated-attention runtime toggle (the Z-Image analogue of the SDXL flash-attn
/// switch, sc-3674). This switch was designed to decide whether a capable build actually *uses* the
/// DiT's fused attention dispatch (CUDA flash-attn / Metal SDPA), so the SceneWorks UI can expose it
/// (defaulted on) and the worker flips it from settings without recompiling. **sc-9032:** the
/// `flash-attn` cargo feature it was ANDed with was a no-op alias (`= ["cuda"]`, no fused dispatch
/// wired) and was removed; the pipeline now hard-codes the accelerated path off, so this toggle is
/// retained as public worker API but is inert until a real fused-dispatch slice re-gates it.
/// Default **on**.
static ACCEL_ATTN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable/disable accelerated attention for subsequently-loaded pipelines. Process-global; the worker
/// calls this from its backend setting at startup. Inert since sc-9032 removed the no-op `flash-attn`
/// feature — no fused dispatch is wired in (retained as worker API).
pub fn set_accel_attn(on: bool) {
    ACCEL_ATTN.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether accelerated attention is currently enabled (the runtime toggle, [`set_accel_attn`]). Since
/// sc-9032 the pipeline hard-codes the accelerated path off (the no-op `flash-attn` feature was
/// removed), so this returning `true` does not enable anything.
pub fn accel_attn_enabled() -> bool {
    ACCEL_ATTN.load(std::sync::atomic::Ordering::Relaxed)
}

/// A loaded candle Z-Image generator. Loading is **tensor-lazy**: `load` reads only the transformer's
/// small `config.json` tier marker, while the heavy components (Qwen3 encoder + DiT + VAE) are built
/// on the first [`generate`](Generator::generate) call and then **cached** in `components` so
/// back-to-back requests skip the disk re-read.
pub struct ZImageGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    text_encoder_source: Option<gen_core::ValidatedEncoderSource>,
    tokenizer_source: gen_core::ValidatedTokenizerSource,
    device: Device,
    dtype: DType,
    loaded_quant: Option<gen_core::Quant>,
    /// Serializes cache use with request-staged eviction. Without this guard a warm request could
    /// retain cloned component Arcs while a concurrent staged request attempted to shed the cache.
    lifecycle: Mutex<()>,
    /// LoRA/LoKr adapters merged into the DiT weights at component-load (sc-5166). Fixed for this
    /// generator instance; empty ⇒ the stock unadapted build.
    adapters: Vec<AdapterSpec>,
    /// Exact caller-prepared identities for every File source used by lazy component/PiD/adapter
    /// loading. Kept intact so later generate-time reads consume the cache-key tokens.
    file_pin_spec: LoadSpec,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    /// External ComfyUI component sources (epic 10451 Phase 2, sc-10668). `Some` ⇒ the pipeline builds
    /// the DiT/TE/VAE from the in-place remapped ComfyUI single-files rather than a diffusers snapshot
    /// dir. Set only by [`load_from_comfyui_components`]; the registry `load` leaves it `None`.
    comfyui: Option<std::sync::Arc<comfyui::ComfyuiSources>>,
    /// Executable shared-memory contract for registry-loaded CUDA providers. Bespoke ComfyUI loads
    /// deliberately leave this absent because their fused component source is not block-addressable.
    memory_strategy: Option<gen_core::MemoryProviderContract>,
    /// Cached components + the accel-attn flag they were built with. `Mutex` because `Generator` is
    /// shared and `generate` takes `&self`; the lock is held only to read/populate the cache, never
    /// across the denoise.
    components: Mutex<Option<(bool, Components)>>,
    /// Lazily-built, cached f32 VAE encoder for the img2img / `Reference` path (sc-11783). Built on the
    /// **first img2img request only** — a pure txt2img workload never populates it, so the txt2img cost is
    /// unchanged. Accel-independent (the encoder has no attention-dispatch toggle), so a single cached
    /// instance serves every request. The Turbo mirror of the base generator's `vae_encoder` (sc-8646).
    vae_encoder: Mutex<Option<Arc<VaeEncoder>>>,
}

impl ZImageGenerator {
    /// Get the cached components, loading (and caching) them on a miss. Keyed by the effective
    /// accel-attn setting (baked into the DiT config at build), so flipping [`set_accel_attn`] between
    /// calls rebuilds rather than serving a stale DiT.
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // sc-9032: the no-op `flash-attn` cargo feature (formerly ANDed here) was removed. No fused
        // CUDA flash-attn / Metal SDPA dispatch is wired behind a build feature, so `false` is
        // byte-identical to the old `cfg!(feature = "flash-attn") && accel_attn_enabled()` (which
        // always resolved false in every buildable config). `set_accel_attn` stays as worker API.
        let accel = false;
        let mut guard = candle_gen::lock_recover(&self.components);
        if let Some((cached_accel, comps)) = guard.as_ref() {
            if *cached_accel == accel {
                return Ok(comps.clone());
            }
        }
        let comps = pipe.load_components(accel)?;
        *guard = Some((accel, comps.clone()));
        Ok(comps)
    }

    /// Get the cached f32 VAE encoder for the img2img / `Reference` path (sc-11783), building it on a
    /// miss. Only ever called when a request carries a `Reference` at a strength that yields a non-empty
    /// denoise (`start_step > 0`), so a txt2img-only workload never builds it. Mirrors the base
    /// generator's `vae_encoder` (sc-8646).
    fn vae_encoder(&self, pipe: &Pipeline) -> gen_core::Result<Arc<VaeEncoder>> {
        // The inner `?` bridges the candle-side `load_vae_encoder` error into `gen_core::Error`.
        candle_gen::cached(&self.vae_encoder, || Ok(Arc::new(pipe.load_vae_encoder()?)))
    }
}

impl Generator for ZImageGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return gen_core::MemorySafetyDecision::Accept;
        };
        memory_strategy::admission_safety_check(MODEL_ID, contract, context, self.loaded_quant)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return Ok(None);
        };
        memory_strategy::validate_context(MODEL_ID, contract, context, self.loaded_quant)?;
        Ok(Some(Box::new(memory_strategy::request_scope(
            MODEL_ID,
            self.device.clone(),
            contract,
            context,
        )?)))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor: the descriptor advertises a single `Reference` (img2img, sc-11783)
        // but no guidance and no negative prompt, so guidance / negative / a MultiReference / any other
        // conditioning kind is rejected here (distilled-model honesty). A >1-`Reference` request is caught
        // by `resolve_reference` in `generate`.
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        // Model-specific floor on top (mirrors mlx-gen-z-image::validate_request).
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "z_image_turbo: prompt must not be empty".into(),
            ));
        }
        // An explicit `steps: Some(0)` would VAE-decode pure noise — reject loudly (txt2img-only).
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(
                "z_image_turbo: steps must be >= 1 (an explicit 0 renders undenoised noise)".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "z_image_turbo: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        // The rich-`CandleError` tail — including the typed `Canceled` — bridges into
        // `gen_core::Error` via `?`. The light `Pipeline` handle carries the snapshot/device; the
        // heavy components come from the cache.
        let pipe = match &self.comfyui {
            // In-place ComfyUI load (sc-10668): the DiT/VAE remap + verbatim Qwen3 TE.
            Some(sources) => Pipeline::load_comfyui_with_text_encoder(
                sources.clone(),
                self.text_encoder_source.clone(),
                self.tokenizer_source.clone(),
                &self.device,
                self.dtype,
                &self.adapters,
                self.pid_spec.clone(),
            ),
            None => Pipeline::load_with_text_encoder(
                &self.root,
                self.text_encoder_source.clone().ok_or_else(|| {
                    gen_core::Error::Msg(
                        "z_image_turbo: validated text encoder source is unavailable".into(),
                    )
                })?,
                &self.device,
                self.dtype,
                &self.adapters,
                self.pid_spec.clone(),
            ),
        };

        self.file_pin_spec.read_files_unchanged(
            self.file_pin_spec.file_source_paths(),
            || {
                if let Some(memory) = req.memory.as_ref().filter(|memory| {
                    memory.stage_residency
                        || memory.tile_vae_decode
                        || memory.chunk_attention
                        || memory.stream_transformer_blocks
                }) {
                    if !memory.stage_residency {
                        return Err(gen_core::Error::Unsupported(
                            "z_image_turbo: bounded decode, attention, and transformer residency require \
                             request-scoped staged residency"
                                .into(),
                        ));
                    }
                    if req.use_pid {
                        return Err(gen_core::Error::Unsupported(
                            "z_image_turbo: PiD decode is not supported under sequential residency; use the \
                             native VAE route or resident policy"
                                .into(),
                        ));
                    }
                    // A warm request may have populated either cache. Synchronize before releasing those
                    // weights, then let the request-owned three-stage route load/drop each phase in turn.
                    self.device
                        .synchronize()
                        .map_err(candle_gen::CandleError::from)?;
                    drop(candle_gen::lock_recover(&self.components).take());
                    drop(candle_gen::lock_recover(&self.vae_encoder).take());
                    let images = pipe.render_sequential(req, on_progress)?;
                    return Ok(GenerationOutput::Images(images));
                }
                let components = self.components(&pipe)?;

                // img2img / `Reference` (sc-11783): resolve the single reference + its effective
                // strength, and — when the strength yields a non-empty structure-preserving denoise
                // (`start_step > 0`) — VAE-encode it to the clean init latent. `resolve_reference`
                // errors on >1 reference; the capability floor in `validate` already rejects any
                // non-`Reference` conditioning. Mirrors the base generator + the shared
                // `render_base` img2img (sc-8646).
                let reference = pipeline::resolve_reference(req)?;
                let start_step = match &reference {
                    Some((_, strength)) => pipeline::init_time_step(
                        req.steps.map(|s| s as usize).unwrap_or(DEFAULT_STEPS),
                        *strength,
                    ),
                    None => 0,
                };
                let clean = if start_step > 0 {
                    let (image, _) = reference.expect("start_step > 0 implies a reference");
                    let encoder = self.vae_encoder(&pipe)?;
                    Some(pipe.encode_reference(&encoder, image, req.width, req.height)?)
                } else {
                    None
                };

                let images =
                    pipe.render(req, &components, clean.as_ref(), start_step, on_progress)?;
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

/// Z-Image-Turbo's identity + the wired surface: distilled txt2img (no CFG, no negative prompt) plus
/// LoRA/LoKr adapter merge (sc-5166) and reference-guided **img2img** latent-init (sc-11783, a single
/// `Conditioning::Reference`). Q4/Q8 quantization stays the Python fallback's job until candle wires it,
/// so the descriptor never promises a path `generate` can't serve. Two backend-correct deviations from
/// `mlx-gen-z-image`: `backend = "candle"` and `mac_only = false`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: Some(ENCODER_CONTRACT),
        denoiser_output_latent_space: Some(&candle_gen::gen_core::FLUX1_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "z-image",
        // The tensor backend whose provider crate registered this engine (sc-3723). MLX sets "mlx".
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Turbo is guidance-distilled: no CFG, no negative prompt.
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // img2img reference-guided latent-init (sc-11783): a single `Conditioning::Reference` seeds
            // the CFG-free denoise from the VAE-encoded reference (`render` + `encode_reference`, the
            // Turbo mirror of the base `z_image` img2img sc-8646). The strict-pose ControlNet + the
            // `edit_image` masked-edit surfaces stay bespoke worker streams (not registry-advertised).
            conditioning: vec![ConditioningKind::Reference],
            // LoRA/LoKr now wired (sc-5166): a trained adapter merges into the dense DiT weights at
            // load ([`crate::adapters::merge_adapters`]), closing the candle train→infer loop. Q4/Q8
            // quantization is still deferred (rejected at load, not silently dropped).
            supports_lora: true,
            supports_lokr: true,
            // Unified curated sampler/scheduler menu (epic 7114 P4, sc-7123). Z-Image-Turbo is
            // guidance-distilled (4 steps, `euler` recommended), but the curated integrators +
            // σ-schedules are exposed for ComfyUI parity; the default (`euler` over the native linear
            // flow-match schedule) is the byte-faithful N1 no-op.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            // candle is the Windows/CUDA backend — NOT Mac-only (the MLX provider sets this true).
            mac_only: false,
            supported_quants: &[],
            component_precision_floors: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: true,
            unconditionally_engages_staged_residency: false,
            // Per-step latent previews (epic 16948, sc-16957). Every Turbo render lane emits: the
            // resident and staged txt2img/img2img routes hand `run_flow_sampler` a projector hook, and
            // the name-driven control + edit providers' bespoke Euler loops emit directly. The
            // projection reuses the epic-16624 Z-Image 16-channel fit — see [`crate::preview`].
            supports_preview: true,
            supports_prompt_enhancement: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            // Chained denoise passes are not wired for this provider (sc-20415).
            supports_denoise_passes: false,
            max_speakers: None,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            size_floor: SizeFloor::RangeChecked,
            execution: Default::default(),
            approximation: Default::default(),
        },
    }
}

fn comfyui_sources_from_spec(
    spec: &LoadSpec,
) -> gen_core::Result<Option<std::sync::Arc<comfyui::ComfyuiSources>>> {
    let WeightsSource::File(_primary_path) = &spec.weights else {
        return Ok(None);
    };
    gen_core::reject_unknown_components(
        spec,
        &[
            BASE_SNAPSHOT_COMPONENT,
            COMFYUI_TEXT_ENCODER_COMPONENT,
            COMFYUI_VAE_COMPONENT,
        ],
        MODEL_ID,
    )?;
    let tokenizer_dir = gen_core::require_base_snapshot(spec, MODEL_ID)?.to_path_buf();
    let legacy_text_encoder = spec.components.get(COMFYUI_TEXT_ENCODER_COMPONENT);
    if spec.text_encoder.is_some() && legacy_text_encoder.is_some() {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: text encoder was supplied through both LoadSpec::text_encoder and legacy component '{COMFYUI_TEXT_ENCODER_COMPONENT}'"
        )));
    }
    let text_encoder = spec.text_encoder.as_ref().or(legacy_text_encoder);
    let vae = spec.components.get(COMFYUI_VAE_COMPONENT);
    let primary = spec
        .weights_file_pin()?
        .expect("File weights must resolve to a pin");
    let sources = match (text_encoder, vae) {
        (None, None) => comfyui::ComfyuiSources::combined(primary.clone(), tokenizer_dir)?,
        (Some(WeightsSource::File(text_encoder)), Some(WeightsSource::File(vae))) => {
            comfyui::ComfyuiSources::separate(
                primary.clone(),
                Some(spec.file_pin_for(text_encoder)?),
                spec.file_pin_for(vae)?,
                tokenizer_dir,
            )?
        }
        (Some(WeightsSource::Dir(_)), Some(WeightsSource::File(vae))) => {
            comfyui::ComfyuiSources::separate(
                primary.clone(),
                None,
                spec.file_pin_for(vae)?,
                tokenizer_dir,
            )?
        }
        (Some(_), None) => comfyui::ComfyuiSources::combined(primary.clone(), tokenizer_dir)?,
        (_, Some(WeightsSource::Dir(path))) => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: component '{COMFYUI_VAE_COMPONENT}' must be a file, not {}",
                path.display()
            )))
        }
        _ => {
            return Err(gen_core::Error::Msg(format!(
                "{MODEL_ID}: separate ComfyUI import requires a text encoder and '{COMFYUI_VAE_COMPONENT}', or neither for a combined checkpoint"
            )))
        }
    };
    Ok(Some(std::sync::Arc::new(sources)))
}

fn text_encoder_source_from_spec(
    spec: &LoadSpec,
    root: &Path,
) -> gen_core::Result<Option<gen_core::ValidatedEncoderSource>> {
    let legacy = spec.components.get(COMFYUI_TEXT_ENCODER_COMPONENT);
    if spec.text_encoder.is_some() && legacy.is_some() {
        return Err(gen_core::Error::Msg(format!(
            "{MODEL_ID}: text encoder was supplied through both LoadSpec::text_encoder and legacy component '{COMFYUI_TEXT_ENCODER_COMPONENT}'"
        )));
    }
    match spec.text_encoder.as_ref().or(legacy) {
        Some(source @ WeightsSource::File(_)) if matches!(spec.weights, WeightsSource::File(_)) => {
            ENCODER_CONTRACT
                .validate_comfyui_source_against_base(source, root)
                .map(Some)
        }
        Some(source) => ENCODER_CONTRACT
            .validate_source_against_base(source, root)
            .map(Some),
        None if matches!(spec.weights, WeightsSource::Dir(_)) => {
            ENCODER_CONTRACT.source_for_load(spec, root).map(Some)
        }
        None => Ok(None),
    }
}

fn tokenizer_source_from_spec(
    spec: &LoadSpec,
    root: &Path,
    text_encoder_source: Option<&gen_core::ValidatedEncoderSource>,
) -> gen_core::Result<gen_core::ValidatedTokenizerSource> {
    if let Some(source) = text_encoder_source {
        return source.tokenizer_source().cloned().ok_or_else(|| {
            gen_core::Error::Unsupported(
                "z-image validated text encoder has no retained tokenizer receipt".into(),
            )
        });
    }
    let WeightsSource::File(primary) = &spec.weights else {
        return Err(gen_core::Error::Unsupported(
            "z-image tokenizer receipt is unavailable for a non-file source".into(),
        ));
    };
    ENCODER_CONTRACT.validate_embedded_comfyui_file_against_base(
        primary,
        comfyui::COMBINED_TEXT_ENCODER_PREFIXES,
        root,
    )
}

/// Construct the (lazy) candle Z-Image generator from a [`LoadSpec`]. A [`WeightsSource::Dir`] points
/// at a complete `Tongyi-MAI/Z-Image-Turbo` diffusers snapshot. A [`WeightsSource::File`] selects an
/// imported combined checkpoint, or an imported DiT when the text encoder and VAE files are supplied
/// as named components; both File shapes require the tokenizer companion under
/// [`BASE_SNAPSHOT_COMPONENT`]. All shapes use this same registry provider. LoRA/LoKr adapters
/// are accepted and merged into the DiT at first `generate` (sc-5166); on-the-fly quantization and
/// control/IP-adapter overlays are still rejected — not wired, so refusing is more honest than
/// silently dropping them (the worker falls back to Python).
pub(crate) fn validate_load_spec(spec: &LoadSpec) -> gen_core::Result<()> {
    spec.validate_prepared_file_pins()?;
    let _ = comfyui_sources_from_spec(spec)?;
    let _ = gen_core::require_base_snapshot(spec, MODEL_ID)?;
    if matches!(spec.weights, WeightsSource::Dir(_)) {
        gen_core::reject_unknown_components(spec, &[], MODEL_ID)?;
    }
    // z-image loads a **pre-quantized MLX-packed tier** (`SceneWorks/z-image-turbo-mlx` q4/q8)
    // transparently when the snapshot dir carries a `quantization` block in its component `config.json`
    // (sc-9408, auto-detected at first `generate`) — no `spec.quantize` needed, the tier is already
    // quantized. `spec.quantize` is the *on-the-fly* quant of a *dense* tier, which z-image does not do
    // (the packed tier is the only quantized path), so it stays rejected — honest rather than
    // silently loading the dense tier at full precision.
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle z_image_turbo does not quantize a dense tier on the fly — point the weights dir at \
             a pre-quantized packed tier (SceneWorks/z-image-turbo-mlx q4/q8), which loads directly"
                .into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle z_image_turbo does not support control / IP-adapter overlays yet (txt2img only)"
                .into(),
        ));
    }
    if spec.identity.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle z_image_turbo does not support identity weights".into(),
        ));
    }
    let root = gen_core::require_base_snapshot(spec, MODEL_ID)?;
    let text_encoder_source = text_encoder_source_from_spec(spec, root)?;
    let _ = tokenizer_source_from_spec(spec, root, text_encoder_source.as_ref())?;
    let loaded_quant = memory_strategy::snapshot_quant_tier(spec, MODEL_ID)?;
    if let Some(source) = text_encoder_source.as_ref() {
        let load_time_quant =
            source.load_time_quant_bits(loaded_quant.map(Quant::bits), MODEL_ID)?;
        if let Some(bits) = load_time_quant {
            return Err(gen_core::Error::Unsupported(format!(
                "candle {MODEL_ID} requires a selected text encoder already packed at Q{bits}; this provider does not quantize a dense Z-Image encoder on the fly"
            )));
        }
    }
    Ok(())
}

pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    validate_load_spec(spec)?;
    let comfyui = comfyui_sources_from_spec(spec)?;
    let root = gen_core::require_base_snapshot(spec, MODEL_ID)?.to_path_buf();
    let text_encoder_source = text_encoder_source_from_spec(spec, &root)?;
    let tokenizer_source = tokenizer_source_from_spec(spec, &root, text_encoder_source.as_ref())?;
    let loaded_quant = memory_strategy::snapshot_quant_tier(spec, MODEL_ID)?;
    // Z-Image is a bf16 model; load at bf16 regardless of the CPU-default dtype. The device is the
    // backend selected at compile time (CUDA on Windows, Metal/CPU on Mac).
    let device = candle_gen::default_device()?;
    #[cfg(any(feature = "cuda", test))]
    let memory_strategy = Some(memory_strategy::provider_contract(MODEL_ID, spec)?);
    #[cfg(not(any(feature = "cuda", test)))]
    let memory_strategy = None;
    Ok(Box::new(ZImageGenerator {
        descriptor: descriptor(),
        root,
        text_encoder_source,
        tokenizer_source,
        device,
        dtype: DType::BF16,
        loaded_quant,
        lifecycle: Mutex::new(()),
        adapters: spec.adapters.clone(),
        file_pin_spec: spec.clone(),
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike quant/control above, it is not
        // rejected — `None` simply keeps the byte-exact native-VAE path.
        pid_spec: spec.pid.clone(),
        comfyui,
        memory_strategy,
        components: Mutex::new(None),
        vae_encoder: Mutex::new(None),
    }))
}

/// Construct an in-place **ComfyUI** Z-Image generator (epic 10451 Phase 2, sc-10668) from the three
/// separate ComfyUI component files + the directory holding our shipped `tokenizer/tokenizer.json` (the
/// one tiny file a ComfyUI tree does not ship). The DiT and VAE are key-remapped in memory at first
/// `generate` (`comfyui`) and the Qwen3 encoder loads verbatim — read in place, no copy, no
/// re-download. Dense bf16, plain fp8, and scalar-companion scaled-fp8 files all normalize to bf16
/// before assembly; unsupported packed integer formats remain typed load errors. `adapters` is the
/// ordered LoRA/LoKr stack applied to the remapped base DiT before generation.
///
/// Retained as a compatibility shim: it constructs the registry [`LoadSpec`] and delegates to
/// [`load`], so there is no second provider lifecycle.
pub fn load_from_comfyui_components(
    transformer_file: impl Into<PathBuf>,
    text_encoder_file: impl Into<PathBuf>,
    vae_file: impl Into<PathBuf>,
    tokenizer_dir: impl Into<PathBuf>,
    adapters: Vec<AdapterSpec>,
) -> gen_core::Result<Box<dyn Generator>> {
    let mut spec = LoadSpec::new(WeightsSource::File(transformer_file.into()))
        .with_component(
            BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(tokenizer_dir.into()),
        )
        .with_text_encoder(WeightsSource::File(text_encoder_file.into()))
        .with_component(COMFYUI_VAE_COMPONENT, WeightsSource::File(vae_file.into()))
        .with_adapters(adapters);
    spec.prepare_file_sources()?;
    load(&spec)
}

/// Construct a Z-Image generator from one fused community checkpoint containing transformer, Qwen3
/// text-encoder, and VAE tensors. The tokenizer remains a model-agnostic JSON asset supplied by
/// `tokenizer_dir`; `adapters` is applied to the remapped base DiT in order.
pub fn load_from_comfyui_checkpoint(
    checkpoint_file: impl Into<PathBuf>,
    tokenizer_dir: impl Into<PathBuf>,
    adapters: Vec<AdapterSpec>,
) -> gen_core::Result<Box<dyn Generator>> {
    let mut spec = LoadSpec::new(WeightsSource::File(checkpoint_file.into()))
        .with_component(
            BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(tokenizer_dir.into()),
        )
        .with_adapters(adapters);
    spec.prepare_file_sources()?;
    load(&spec)
}

// Link-time self-registration into gen-core's model registry. Linking this crate makes
// the explicit family and platform catalogs resolve the candle generator.
candle_gen::register_generators! { pub(crate) const REGISTRATION = descriptor => load }

fn registered_turbo_memory_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::provider_contract(MODEL_ID, spec)
}

fn registered_base_memory_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::provider_contract(base::MODEL_ID, spec)
}

fn weights_free_turbo_memory_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    // Q4/Q8 registry tiers are provisioned packed directories, not on-the-fly quant requests. Use
    // the weights-free route id to bypass source validation, then restore the registered identity.
    let mut contract = memory_strategy::provider_contract("z_image_turbo_contract_surface", spec)?;
    contract.provider_id = MODEL_ID.to_owned();
    Ok(contract)
}

fn weights_free_turbo_surface_contract(
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_surface_contract(MODEL_ID, surface)
}

fn weights_free_base_surface_contract(
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_surface_contract(base::MODEL_ID, surface)
}

fn weights_free_turbo_control_surface_contract(
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_control_surface_contract("z_image_turbo_control", surface)
}

fn weights_free_base_control_surface_contract(
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_control_surface_contract("z_image_control", surface)
}

fn registered_turbo_control_memory_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::control_contract("z_image_turbo_control", spec)
}

fn registered_base_control_memory_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::control_contract("z_image_control", spec)
}

const TURBO_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID,
    contract: registered_turbo_memory_contract,
    safety_check: memory_strategy::registered_safety_check,
};
#[cfg(any(feature = "cuda", test))]
const TURBO_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            memory_strategy::registered_begin_request(MODEL_ID, spec, contract, context)
        },
    };

const BASE_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: base::MODEL_ID,
    contract: registered_base_memory_contract,
    safety_check: memory_strategy::registered_safety_check,
};
#[cfg(feature = "cuda")]
const BASE_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: base::MODEL_ID,
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            memory_strategy::registered_begin_request(base::MODEL_ID, spec, contract, context)
        },
    };

const TURBO_CONTROL_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: "z_image_turbo_control",
        contract: registered_turbo_control_memory_contract,
        safety_check: memory_strategy::registered_safety_check,
    };

#[cfg(any(feature = "cuda", test))]
const TURBO_CONTROL_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: "z_image_turbo_control",
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            memory_strategy::registered_begin_request(
                "z_image_turbo_control",
                spec,
                contract,
                context,
            )
        },
    };

const BASE_CONTROL_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: "z_image_control",
        contract: registered_base_control_memory_contract,
        safety_check: memory_strategy::registered_safety_check,
    };

#[cfg(any(feature = "cuda", test))]
const BASE_CONTROL_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: "z_image_control",
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            memory_strategy::registered_begin_request("z_image_control", spec, contract, context)
        },
    };

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        Conditioning, ConditioningKind, Image, LoadShape, LoadSpec, MemoryBehaviorRoute,
        MemoryMode, MemoryNumericTier, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
        Precision, Quant, WeightsSource,
    };
    use std::path::Path;

    fn valid_model_root() -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("model root");
        gen_core_testkit::write_encoder_contract_fixture(
            &root.path().join("text_encoder"),
            ENCODER_CONTRACT,
        )
        .expect("valid encoder fixture");
        root
    }

    fn prefixed_encoder_fixture(
        root: &Path,
        contract: gen_core::EncoderContract,
        prefix: &str,
    ) -> PathBuf {
        use std::io::{Read as _, Write as _};

        let component = root.join("encoder");
        gen_core_testkit::write_encoder_contract_fixture(&component, contract)
            .expect("encoder fixture");
        let mut source = std::fs::File::open(component.join("model.safetensors"))
            .expect("encoder fixture weights");
        let mut header_len = [0_u8; 8];
        source.read_exact(&mut header_len).unwrap();
        let header_len = u64::from_le_bytes(header_len) as usize;
        let mut header_bytes = vec![0_u8; header_len];
        source.read_exact(&mut header_bytes).unwrap();
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&header_bytes).unwrap();
        let mut prefixed = serde_json::Map::new();
        let mut data_len = 0usize;
        for (name, value) in header {
            data_len = data_len.max(
                value["data_offsets"][1]
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap(),
            );
            prefixed.insert(format!("{prefix}{name}"), value);
        }
        prefixed.insert(
            "model.diffusion_model.block.weight".into(),
            serde_json::json!({
                "dtype": "U8", "shape": [1], "data_offsets": [data_len, data_len + 1]
            }),
        );
        data_len += 1;
        prefixed.insert(
            "first_stage_model.decoder.weight".into(),
            serde_json::json!({
                "dtype": "U8", "shape": [1], "data_offsets": [data_len, data_len + 1]
            }),
        );
        data_len += 1;
        let mut encoded = serde_json::to_vec(&prefixed).unwrap();
        while !(8 + encoded.len()).is_multiple_of(8) {
            encoded.push(b' ');
        }
        let path = root.join("combined.safetensors");
        let mut output = std::fs::File::create(&path).unwrap();
        output
            .write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        output.write_all(&encoded).unwrap();
        output
            .set_len(8 + encoded.len() as u64 + data_len as u64)
            .unwrap();
        path
    }

    #[test]
    fn control_memory_registrations_have_weights_free_behavior_seams() {
        let registry = gen_core::ProviderRegistryBuilder::new()
            .register_composed_memory_strategy(TURBO_CONTROL_MEMORY_REGISTRATION)
            .register_memory_behavior(TURBO_CONTROL_MEMORY_BEHAVIOR)
            .register_composed_memory_strategy(BASE_CONTROL_MEMORY_REGISTRATION)
            .register_memory_behavior(BASE_CONTROL_MEMORY_BEHAVIOR)
            .build()
            .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));

        gen_core_testkit::memory_strategy_registry_conformance(&registry, &spec);

        for contract in [
            registered_turbo_control_memory_contract(&spec).unwrap(),
            registered_base_control_memory_contract(&spec).unwrap(),
        ] {
            let fixtures = memory_strategy::registered_valid_fixture(
                &spec,
                &contract,
                MemoryStrategy::StagedResidency,
            )
            .unwrap();
            assert!(!fixtures.is_empty(), "{}", contract.provider_id);
            for fixture in fixtures {
                assert_eq!(fixture.context.mode, MemoryMode::ImageToImage);
                assert_eq!(fixture.context.geometry.reference_count, 1);
                assert_eq!(fixture.request.conditioning.len(), 1);
                assert!(matches!(
                    fixture.request.conditioning.as_slice(),
                    [Conditioning::Reference { .. }]
                ));
            }
        }
    }

    #[test]
    fn catalog_surfaces_publish_exact_prepacked_z_image_routes() {
        let registry = register_memory_contract_surfaces(
            gen_core::ProviderRegistryBuilder::new()
                .register_generator(REGISTRATION)
                .register_generator(base::REGISTRATION),
        )
        .build()
        .expect("Z-Image surface registry");
        let surfaces = registry
            .memory_contract_surfaces()
            .expect("weights-free Z-Image surfaces");

        for (provider_id, composed, fingerprint) in [
            (MODEL_ID, false, memory_strategy::CALIBRATION_FINGERPRINT),
            (
                base::MODEL_ID,
                false,
                memory_strategy::CALIBRATION_FINGERPRINT,
            ),
            (
                "z_image_turbo_control",
                true,
                memory_strategy::CONTROL_CALIBRATION_FINGERPRINT,
            ),
            (
                "z_image_control",
                true,
                memory_strategy::CONTROL_CALIBRATION_FINGERPRINT,
            ),
        ] {
            let provider_surfaces: Vec<_> = surfaces
                .iter()
                .filter(|surface| surface.contract.provider_id == provider_id)
                .collect();
            assert_eq!(provider_surfaces.len(), 12, "{provider_id}");
            let mut rung_four = 0;
            for surface in provider_surfaces {
                let expected =
                    surface.selector.load_shape == gen_core::LoadShape::DeferredMaterialization;
                let capability = surface
                    .contract
                    .capability(gen_core::MemoryStrategy::BoundedTransformerResidency)
                    .expect("rung four");
                assert_eq!(
                    capability.support,
                    if expected {
                        gen_core::MemoryStrategySupport::Implemented
                    } else {
                        gen_core::MemoryStrategySupport::Missing
                    },
                    "{}:{}",
                    provider_id,
                    surface.selector.id()
                );
                rung_four += usize::from(expected);
                assert_eq!(surface.composed, composed, "{provider_id}");
                assert_eq!(
                    surface.contract.calibration.as_ref().unwrap().fingerprint,
                    fingerprint,
                    "{provider_id}"
                );
                assert_eq!(
                    surface.contract.asset_facts,
                    gen_core::MemoryAssetFacts::default(),
                    "weights-free Z surfaces claim no asset bytes"
                );
                if composed {
                    assert!(
                        surface.spec.control.is_some(),
                        "{provider_id}:{} must carry a control source",
                        surface.selector.id()
                    );
                } else {
                    assert!(surface.spec.control.is_none(), "{provider_id}");
                }
            }
            assert_eq!(rung_four, 6, "{provider_id}");
        }
    }

    fn admission_context(
        contract: &gen_core::MemoryProviderContract,
        strategy: MemoryStrategy,
        quant: Option<Quant>,
    ) -> MemoryRunContext {
        gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant,
                component_precision_floors: &[],
            },
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap()
    }

    fn assert_admission_matrix(
        label: &str,
        context: &MemoryRunContext,
        admit: impl Fn(&MemoryRunContext) -> MemorySafetyDecision,
    ) {
        assert_eq!(
            admit(context),
            MemorySafetyDecision::Accept,
            "{label}: valid fixture"
        );

        let mut phases = context.clone();
        phases.has_phases = true;
        assert!(
            matches!(
                admit(&phases),
                MemorySafetyDecision::Reject { reason } if reason.contains("multi-phase")
            ),
            "{label}: multi-phase mutation"
        );

        let mut pid = context.clone();
        pid.use_pid = true;
        if context.selection.strategy.is_optimized() {
            assert!(
                matches!(
                    admit(&pid),
                    MemorySafetyDecision::Reject { reason } if reason.contains("PiD")
                ),
                "{label}: optimized PiD mutation"
            );
        } else {
            assert_eq!(
                admit(&pid),
                MemorySafetyDecision::Accept,
                "{label}: resident PiD remains admissible"
            );
        }

        let mut stale = context.clone();
        stale.calibration_fingerprint.push_str("-stale");
        assert!(
            matches!(
                admit(&stale),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("calibration handshake mismatch")
            ),
            "{label}: fingerprint mutation"
        );

        let mut wrong_tier = context.clone();
        wrong_tier.selection.tier.quant = None;
        assert!(
            matches!(
                admit(&wrong_tier),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("does not match loaded tier")
            ),
            "{label}: numeric-tier mutation"
        );
    }

    fn packed_q4_spec(tmp: &tempfile::TempDir) -> (LoadSpec, PathBuf) {
        let root = tmp.path().join("candle-z-image-sc-16600-admission");
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        gen_core_testkit::write_encoder_contract_fixture_with_quant(
            &root.join("text_encoder"),
            ENCODER_CONTRACT,
            Some(4),
        )
        .expect("valid packed encoder fixture");
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        spec.load_shape = LoadShape::DeferredMaterialization;
        (spec, root)
    }

    /// The seam under test: resolving `"z_image_turbo"` through the family registry returns this
    /// candle generator. `load`
    /// is tensor-lazy, so a nonexistent weights dir still resolves (the absent tier marker is dense).
    #[test]
    fn z_image_registers_and_resolves_as_candle() {
        let root = valid_model_root();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().into()));
        let g = crate::provider_registry()
            .unwrap()
            .load("z_image_turbo", &spec)
            .expect("candle z-image is registered");
        assert_eq!(g.descriptor().id, "z_image_turbo");
        assert_eq!(g.descriptor().family, "z-image");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn loaded_and_registered_admission_seams_cover_the_complete_context_gate() {
        let tmp = tempfile::tempdir().unwrap();
        let (spec, root) = packed_q4_spec(&tmp);

        for (label, generator) in [
            ("loaded turbo hook", load(&spec).unwrap()),
            ("loaded base hook", base::load(&spec).unwrap()),
        ] {
            let contract = generator
                .memory_strategy_contract()
                .expect("unit-test loads retain their CUDA memory contract");
            let context =
                admission_context(contract, MemoryStrategy::StagedResidency, Some(Quant::Q4));
            assert_admission_matrix(label, &context, |context| {
                generator.memory_strategy_safety_check(context)
            });
        }

        for registration in [
            TURBO_MEMORY_REGISTRATION,
            BASE_MEMORY_REGISTRATION,
            TURBO_CONTROL_MEMORY_REGISTRATION,
            BASE_CONTROL_MEMORY_REGISTRATION,
        ] {
            let contract = (registration.contract)(&spec).unwrap();
            assert_eq!(contract.provider_id, registration.provider_id);
            let strategy = if registration.provider_id.ends_with("_control") {
                MemoryStrategy::Resident
            } else {
                MemoryStrategy::StagedResidency
            };
            let context = admission_context(&contract, strategy, Some(Quant::Q4));
            assert_admission_matrix(registration.provider_id, &context, |context| {
                (registration.safety_check)(&spec, &contract, context)
            });
        }

        std::fs::remove_dir_all(root).ok();
    }

    /// The descriptor advertises the wired distilled txt2img + img2img surface: no CFG or negative
    /// prompt, a single reference conditioning kind, LoRA/LoKr, and not Mac-only.
    #[test]
    fn descriptor_advertises_only_wired_txt2img_surface() {
        let d = descriptor();
        assert!(
            !d.capabilities.supports_guidance,
            "turbo is guidance-distilled"
        );
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        // img2img reference-guided latent-init advertised (sc-11783) — a single `Reference`, NOT
        // MultiReference (that stays a bespoke worker shape).
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        // LoRA/LoKr wired (sc-5166) — merged into the DiT at load.
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        assert!(
            d.capabilities.supports_sequential_offload,
            "the generic txt2img/reference route stages Qwen3, DiT, and VAE"
        );
        assert!(d.capabilities.supported_quants.is_empty());
        assert_eq!(d.capabilities.min_size, 256);
        assert_eq!(d.capabilities.max_size, 2048);
        assert_eq!(d.capabilities.max_count, 8);
        // Curated sampler/scheduler menu (epic 7114 P4, sc-7123): full vocabulary, euler the default.
        assert_eq!(d.capabilities.samplers, candle_gen::curated_sampler_names());
        assert_eq!(
            d.capabilities.schedulers,
            candle_gen::curated_scheduler_names()
        );
    }

    #[test]
    fn imported_sources_share_the_registered_provider_capabilities() {
        assert!(
            descriptor().capabilities.supports_sequential_offload,
            "File and Dir loads now share the registry provider's staged lifecycle"
        );
    }

    /// A txt2img request passes validation; unsupported shapes are rejected clearly (not silently
    /// served). Uses the lazy generator so no weights are needed.
    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let root = valid_model_root();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().into()));
        let g = crate::provider_registry()
            .unwrap()
            .load("z_image_turbo", &spec)
            .unwrap();

        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());

        // img2img: a single `Reference` validates (sc-11783). A non-empty image isn't needed at the
        // validate floor — the VAE-encode happens in `generate`.
        let img2img = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: Some(0.6),
            }],
            ..Default::default()
        };
        assert!(g.validate(&img2img).is_ok(), "single Reference is img2img");

        for bad in [
            // empty prompt
            GenerationRequest::default(),
            // guidance on a distilled model (descriptor advertises no guidance)
            GenerationRequest {
                prompt: "x".into(),
                guidance: Some(5.0),
                ..Default::default()
            },
            // negative prompt (not supported)
            GenerationRequest {
                prompt: "x".into(),
                negative_prompt: Some("blurry".into()),
                ..Default::default()
            },
            // non-multiple-of-16 size
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                ..Default::default()
            },
            // explicit 0 steps
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
            // a MultiReference is NOT the advertised img2img surface (only a single `Reference` is)
            GenerationRequest {
                prompt: "x".into(),
                conditioning: vec![Conditioning::MultiReference {
                    images: vec![Image::default(), Image::default()],
                }],
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
        // Sanity: the img2img `Reference` kind is now advertised (sc-11783); MultiReference is not.
        assert!(descriptor()
            .capabilities
            .accepts(ConditioningKind::Reference));
        assert!(!descriptor()
            .capabilities
            .accepts(ConditioningKind::MultiReference));

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised Z-Image
        // bucket to. Pin the value and mutation-check that a multiple of 8 which is not SIZE_MULTIPLE
        // (16) is rejected with the stride error, and an on-stride size passes.
        assert_eq!(SIZE_MULTIPLE, 16);
        let off_stride = g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1000, // 125×8 — a multiple of 8 but not SIZE_MULTIPLE
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiples of 16"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1024, // 64×16 — on-stride
                ..Default::default()
            })
            .is_ok());
    }

    /// The request-scoped memory contract selects the staged route. A pre-cancel must return before
    /// touching the deliberately missing snapshot, proving the request authority is active rather than
    /// silently serving the resident cache path.
    #[test]
    fn request_staging_is_active_and_honors_pre_cancel_before_load() {
        let root = valid_model_root();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().into()));
        let generator = load(&spec).expect("lazy sequential generator");
        let cancel = gen_core::CancelFlag::default();
        cancel.cancel();
        let req = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            cancel,
            memory: Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(matches!(
            generator.generate(&req, &mut |_| {}),
            Err(gen_core::Error::Canceled)
        ));
    }

    /// PiD is a bespoke decoder whose student/caption stack does not yet have a phase-local loader.
    /// Sequential requests reject it explicitly before any snapshot access instead of retaining it
    /// through denoise and making the advertised peak false.
    #[test]
    fn request_staging_rejects_pid_explicitly_before_load() {
        let root = valid_model_root();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().into()));
        let generator = load(&spec).expect("lazy sequential generator");
        let req = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            use_pid: true,
            memory: Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let error = generator.generate(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(error, gen_core::Error::Unsupported(_)));
        assert!(error.to_string().contains("PiD"));
    }

    /// Quantization / control overlays are rejected at load as typed `Unsupported`, so the worker
    /// can fall back to Python rather than the backend silently dropping them. LoRA/LoKr are now
    /// wired (sc-5166), so a LoRA `LoadSpec` is **accepted** (lazily — the merge happens at generate).
    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant};
        let root = valid_model_root();
        let lora = LoadSpec::new(WeightsSource::Dir(root.path().into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load(&lora).is_ok(), "LoRA load is wired + lazy (sc-5166)");

        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));

        let control = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_control(WeightsSource::Dir("/ctrl".into()));
        assert!(matches!(
            load(&control).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }

    #[test]
    fn single_file_source_requires_the_base_snapshot_component() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains(BASE_SNAPSHOT_COMPONENT), "got: {err}");

        let dir = tempfile::tempdir().expect("temp dir");
        let checkpoint = prefixed_encoder_fixture(dir.path(), ENCODER_CONTRACT, "text_encoder.");
        gen_core_testkit::write_encoder_contract_tokenizer_fixture(dir.path(), ENCODER_CONTRACT)
            .unwrap();
        let complete = LoadSpec::new(WeightsSource::File(checkpoint)).with_component(
            BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(dir.path().to_path_buf()),
        );
        if let Err(error) = load(&complete) {
            panic!("complete File spec loads lazily: {error}");
        }
    }

    #[test]
    fn prepared_comfyui_component_replacement_fails_before_provider_load() {
        let dir = tempfile::tempdir().expect("temp dir");
        let dit = dir.path().join("dit.safetensors");
        let text = dir.path().join("text.safetensors");
        let vae = dir.path().join("vae.safetensors");
        for file in [&dit, &text, &vae] {
            std::fs::write(file, b"prepared bytes").unwrap();
        }
        let mut spec = LoadSpec::new(WeightsSource::File(dit))
            .with_component(
                BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(dir.path().join("tokenizer")),
            )
            .with_component(
                COMFYUI_TEXT_ENCODER_COMPONENT,
                WeightsSource::File(text.clone()),
            )
            .with_component(COMFYUI_VAE_COMPONENT, WeightsSource::File(vae));
        spec.prepare_file_sources().unwrap();

        std::fs::write(&text, b"replacement text encoder bytes").unwrap();
        let error = validate_load_spec(&spec)
            .expect_err("provider must reject a changed prepared component")
            .to_string();
        assert!(error.contains("changed after load"), "got: {error}");
    }

    /// The accel-attn runtime toggle defaults on and round-trips (what the worker/UI drive).
    #[test]
    fn accel_attn_toggle_roundtrips() {
        assert!(
            accel_attn_enabled(),
            "accel-attn runtime toggle defaults on"
        );
        set_accel_attn(false);
        assert!(!accel_attn_enabled());
        set_accel_attn(true);
        assert!(accel_attn_enabled());
    }
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let mut explicit_generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        explicit_generators.sort();
        assert_eq!(explicit_generators, ["z_image", "z_image_turbo"]);

        let explicit_trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(explicit_trainers, ["z_image_turbo"]);
        for id in [
            "z_image_turbo",
            "z_image",
            "z_image_turbo_control",
            "z_image_control",
        ] {
            assert_eq!(
                registry.provider_encoder_contract(id),
                Some(super::ENCODER_CONTRACT),
                "{id} must resolve through the provider-owned contract surface"
            );
        }
        assert_eq!(
            registry.provider_encoder_contract("z_image_control_typo"),
            None
        );
    }
}
