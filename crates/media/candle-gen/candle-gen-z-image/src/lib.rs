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

// Private opt-in model-proof plumbing; feature-OFF observers expand to the original expression.
#[macro_use]
#[path = "../../../../llm/test-support/model_proof.rs"]
mod model_proof;
const _: fn() = || {
    let _: std::result::Result<candle_core::Tensor, ()> =
        __model_proof_output!(std::result::Result::<candle_core::Tensor, ()>::Err(()));
};

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
// The per-step latent-preview seam (epic 16948, sc-16957): the REUSED epic-16624 Z-Image 16-channel
// fit plus the `[1, 16, 1, h, w]` → `[1, 16, h, w]` frame-axis drop that reaches it. Public because
// [`control`] and [`edit`] are themselves public, name-driven entry points, so a consumer that stages
// its own denoise against them needs the same projector rather than a second copy of the fit.
pub mod preview;
mod quant;
mod text_encoder;
mod training;
mod training_adapter;

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
mod control_registry;

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

/// Add every registry-owned Candle Z-Image provider to an explicit media registry builder via
/// one-call [`ProviderRegistryBuilder::register_route`](candle_gen::gen_core::ProviderRegistryBuilder::register_route)
/// (`SW-6979`, `REG-1`).
///
/// Bespoke control/edit utilities remain direct worker integrations and are intentionally absent.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_route(route_turbo())
        .register_route(route_base())
        .register_route(route_turbo_control())
        .register_route(route_base_control())
        .register_checkpoint_adapter(gen_core::CheckpointAdapterRegistration {
            backend_bindings: &[gen_core::CheckpointBackendBindingRegistration {
                backend: gen_core::CheckpointBackend::Candle,
                source: gen_core::ImportedModelSource::ComfyUiTree,
                operation: gen_core::ImportedModelOperation::Generate,
                provider_id: MODEL_ID,
                required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
                inherit_adapters: true,
            }],
            ..gen_core::Z_IMAGE_CHECKPOINT_ADAPTER
        })
        .register_trainer(training::REGISTRATION)
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
        .register_memory_behavior(TURBO_MEMORY_BEHAVIOR)
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
        .register_memory_behavior(BASE_MEMORY_BEHAVIOR)
        .register_memory_strategy(TURBO_CONTROL_MEMORY_REGISTRATION)
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
        .register_memory_behavior(TURBO_CONTROL_MEMORY_BEHAVIOR)
        .register_memory_strategy(BASE_CONTROL_MEMORY_REGISTRATION)
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
        .register_memory_behavior(BASE_CONTROL_MEMORY_BEHAVIOR)
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
    self, AdapterSpec, ConditioningKind, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, PidWeights, Progress, Quant, WeightsSource, BASE_SNAPSHOT_COMPONENT,
    COMFYUI_TEXT_ENCODER_COMPONENT, COMFYUI_VAE_COMPONENT,
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

/// Enable/disable accelerated attention: a thin wrapper over the **one** process-wide fused
/// attention switch, [`candle_gen::fused_attention::set_flash_attention_enabled`] (SW-6130), read per
/// call by every candle provider's shared attention helpers — so it is not Z-Image-scoped. Retained
/// as worker API. Default **on**. The Z-Image DiT's caption key-padding mask is trimmed exactly once
/// per prepared request (`dit::caption_padding_trim`), so its unmasked attention takes the fused kernel
/// on a flash build; a caption batch with unequal valid lengths keeps the masked, materialized path.
/// The Metal SDPA path remains off (sc-9032).
pub fn set_accel_attn(on: bool) {
    candle_gen::fused_attention::set_flash_attention_enabled(on);
}

/// Whether accelerated attention is currently enabled: the shared switch
/// ([`candle_gen::fused_attention::flash_attention_enabled`]) behind [`set_accel_attn`].
pub fn accel_attn_enabled() -> bool {
    candle_gen::fused_attention::flash_attention_enabled()
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
    fn generate_with_execution_report(
        &self,
        req: &candle_gen::gen_core::GenerationRequest,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        let plan = compile_turbo_trajectory(req)?;
        self.generate_planned_with_execution_report(req, &plan, None, on_progress)
    }

    fn generate_planned_with_execution_report(
        &self,
        req: &candle_gen::gen_core::GenerationRequest,
        plan: &candle_gen::gen_core::RenderPlan,
        scope: Option<&mut dyn candle_gen::gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_planned_with(self, req, plan, scope, on_progress, || {
            candle_gen::execution::unobserved_adapters(
                self.adapters.len(),
                "candle-gen-z-image/src/lib.rs",
            );
            candle_gen::execution::policy(
                "loaded/loaded_quant",
                format!("{:?}", self.loaded_quant).into(),
            );
        })
    }

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
        if req
            .negative_prompt
            .as_ref()
            .is_some_and(|negative| !negative.is_empty())
            && req.guidance.unwrap_or(0.0) == 0.0
        {
            return Err(gen_core::Error::Msg(
                "z_image_turbo: a negative prompt requires CFG different from 0.0".into(),
            ));
        }

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
        validate_low_res_start_for(&self.descriptor, req, false)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_with_report(req, on_progress)?
            .output
            .ok_or_else(|| {
                gen_core::Error::Msg("z_image_turbo: generator returned no output".into())
            })
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        let plan = compile_turbo_trajectory(req)?;
        self.generate_planned(req, &plan, None, on_progress)
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        self.validate(req)?;
        plan.check_request(&self.descriptor, req)?;
        if self.comfyui.is_none() {
            plan.check_load(&gen_core::RenderLoadBinding::from_spec(&self.file_pin_spec))?;
        }
        let run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        let resolved_pag = self.descriptor.capabilities.resolve_pag(MODEL_ID, req)?;
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
                    if req.use_pid() {
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
                    let (images, actual_steps) = if req.guidance.unwrap_or(0.0) != 0.0 || req.scheduler_shift.is_some() {
                        pipe.render_base_sequential(&experimental_cfg_request(req), Some(&run), resolved_pag.as_ref(), on_progress)?
                    } else {
                        pipe.render_sequential(req, Some(&run), resolved_pag.as_ref(), on_progress)?
                    };
                    let mut report = gen_core::GenerationReport::from_output(GenerationOutput::Images(images));
                    report.pag = resolved_pag.as_ref().map(|resolved| resolved.execution_report(actual_steps));
                    report.low_res_start = plan.low_res_start_execution_report("bf16", 2, "base_dit")?;
                    return Ok(report);
                }
                let components = self.components(&pipe)?;

                // img2img / `Reference` (sc-11783): resolve the single reference + its effective
                // strength, and — when the strength yields a non-empty structure-preserving denoise
                // (`start_step > 0`) — VAE-encode it to the clean init latent. `resolve_reference`
                // errors on >1 reference; the capability floor in `validate` already rejects any
                // non-`Reference` conditioning. Mirrors the base generator + the shared
                // `render_base` img2img (sc-8646).
                let reference = pipeline::resolve_reference(req)?;
                let steps = req.steps.map(|s| s as usize).unwrap_or(DEFAULT_STEPS);
                let start_step = match &reference {
                    Some((_, strength)) => {
                        if req.features.low_res_start.is_some() {
                            gen_core::resolve_img2img_init_time_step(steps, *strength)?
                        } else {
                            pipeline::init_time_step(steps, *strength)
                        }
                    }
                    None => 0,
                };
                let (init_w, init_h) = run.plan().phase_output_size(0);
                let clean = if start_step > 0 {
                    let (image, _) = reference.expect("start_step > 0 implies a reference");
                    let encoder = self.vae_encoder(&pipe)?;
                    Some(pipe.encode_reference(&encoder, image, init_w, init_h)?)
                } else {
                    None
                };

                let (images, actual_steps) = if req.guidance.unwrap_or(0.0) != 0.0 || req.scheduler_shift.is_some() {
                    let request = experimental_cfg_request(req);
                    pipe.render_base(&request, Some(&run), &components, clean.as_ref(), start_step, resolved_pag.as_ref(), on_progress)?
                } else {
                    pipe.render(req, Some(&run), &components, clean.as_ref(), start_step, resolved_pag.as_ref(), on_progress)?
                };
                let mut report = gen_core::GenerationReport::from_output(GenerationOutput::Images(images));
                report.pag = resolved_pag.as_ref().map(|resolved| resolved.execution_report(actual_steps));
                report.low_res_start = plan.low_res_start_execution_report("bf16", 2, "base_dit")?;
                Ok(report)
            },
        )
    }
}

pub(crate) fn validate_low_res_start_for(
    descriptor: &ModelDescriptor,
    req: &GenerationRequest,
    is_base: bool,
) -> gen_core::Result<()> {
    let reference = pipeline::resolve_reference(req)?;
    let default_steps = if is_base {
        pipeline::BASE_DEFAULT_STEPS
    } else {
        DEFAULT_STEPS
    };
    let steps = req.steps.map(|s| s as usize).unwrap_or(default_steps);
    let start_step = match &reference {
        Some((_, strength)) => {
            if req.features.low_res_start.is_some() {
                gen_core::resolve_img2img_init_time_step(steps, *strength)?
            } else {
                pipeline::init_time_step(steps, *strength)
            }
        }
        None => 0,
    };
    let executed_steps = steps.saturating_sub(start_step).max(1);
    if let Some(low_res) = req.features.low_res_start.as_ref() {
        low_res.resolve_for_descriptor(descriptor, req, executed_steps, SIZE_MULTIPLE)?;
    }
    Ok(())
}

pub(crate) fn compile_trajectory_for(
    req: &GenerationRequest,
    is_base: bool,
) -> gen_core::Result<gen_core::RenderPlan> {
    let descriptor = if is_base {
        base::descriptor()
    } else {
        descriptor()
    };
    descriptor
        .capabilities
        .validate_request(descriptor.id, req)?;
    if !is_base
        && req
            .negative_prompt
            .as_ref()
            .is_some_and(|negative| !negative.is_empty())
        && req.guidance.unwrap_or(0.0) == 0.0
    {
        return Err(gen_core::Error::Msg(
            "z_image_turbo: a negative prompt requires CFG different from 0.0".into(),
        ));
    }
    if req.prompt.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{}: prompt must not be empty",
            descriptor.id
        )));
    }
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!(
            "{}: steps must be >= 1 (an explicit 0 renders undenoised noise)",
            descriptor.id
        )));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(gen_core::Error::Msg(format!(
            "{}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            descriptor.id, req.width, req.height
        )));
    }
    let reference = pipeline::resolve_reference(req)?;
    let default_steps = if is_base {
        pipeline::BASE_DEFAULT_STEPS
    } else {
        DEFAULT_STEPS
    };
    let steps = req.steps.map(|s| s as usize).unwrap_or(default_steps);
    let start_step = match &reference {
        Some((_, strength)) => {
            if req.features.low_res_start.is_some() {
                gen_core::resolve_img2img_init_time_step(steps, *strength)?
            } else {
                pipeline::init_time_step(steps, *strength)
            }
        }
        None => 0,
    };
    let (full_sigmas, guidance) = if is_base {
        (
            pipeline::resolve_base_sigmas(req, steps)?,
            req.guidance.unwrap_or(4.0),
        )
    } else if req.guidance.unwrap_or(0.0) != 0.0 || req.scheduler_shift.is_some() {
        let eff_req = experimental_cfg_request(req);
        (
            pipeline::resolve_base_sigmas(&eff_req, steps)?,
            eff_req.guidance.unwrap_or(1.0),
        )
    } else {
        (pipeline::resolve_turbo_sigmas(req, steps)?, 1.0)
    };
    let start = start_step.min(full_sigmas.len().saturating_sub(1));
    let sigmas = full_sigmas[start..].to_vec();
    let executed_steps = sigmas.len().saturating_sub(1);
    let sampler = gen_core::sampling::canonical_solver_or_euler(req.sampler.as_deref()).name();
    gen_core::RenderPlan::packed_image_trajectory(
        &descriptor,
        req,
        gen_core::render::PackedImagePass {
            pixel_stride: SIZE_MULTIPLE,
            token_width: 16,
            sampler,
            guidance,
            sigmas,
            executed_steps,
            capture_sigma: 0.0,
        },
    )
}

pub(crate) fn compile_turbo_trajectory(
    req: &GenerationRequest,
) -> gen_core::Result<gen_core::RenderPlan> {
    compile_trajectory_for(req, false)
}

pub fn compile_turbo(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
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
    compile_turbo_trajectory(req)
        .map(|plan| plan.bind_load(spec))
        .map(Some)
}

pub(crate) fn compile_base_trajectory(
    req: &GenerationRequest,
) -> gen_core::Result<gen_core::RenderPlan> {
    compile_trajectory_for(req, true)
}

pub fn compile_base(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    gen_core::reject_unknown_components(spec, &[], base::MODEL_ID)?;
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle z_image does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle z_image does not support control / IP-adapter overlays yet (txt2img only)"
                .into(),
        ));
    }
    if spec.identity.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle z_image does not support identity weights".into(),
        ));
    }
    compile_base_trajectory(req)
        .map(|plan| plan.bind_load(spec))
        .map(Some)
}

const CONTROL_KINDS: &[gen_core::ControlKind] = &[
    gen_core::ControlKind::Pose,
    gen_core::ControlKind::Canny,
    gen_core::ControlKind::Depth,
];

pub(crate) fn route_base() -> gen_core::RouteRecord {
    let mut route = base::route_base();
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut bindings = route.bindings.with_render_plan(compile_base);
    #[cfg(feature = "cuda")]
    {
        bindings = bindings
            .with_memory_registration(BASE_MEMORY_REGISTRATION)
            .with_memory_fixture(
                gen_core::candle_memory_contract_surface_specs,
                registered_base_memory_contract,
            )
            .with_memory_surface_resolver(weights_free_base_surface_contract)
            .with_memory_behavior_registration(BASE_MEMORY_BEHAVIOR);
    }
    route.bindings = bindings;
    route
}

pub(crate) fn route_turbo_control() -> gen_core::RouteRecord {
    let mut route = route_turbo();
    route.route_id = "z_image_turbo_control";
    route.structure.conditioning.kinds = vec![ConditioningKind::Control];
    route.structure.conditioning.control_kinds =
        Some(gen_core::AcceptedControlKinds::Only(CONTROL_KINDS.to_vec()));
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresControl;
    route.structure.conditioning.modules = vec!["z_image_vace_control"];
    route.structure.weight_plan.requires_control_overlay = true;
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut bindings = gen_core::RouteBindings::generator(
        turbo_control_descriptor,
        control_registry::load_turbo_control,
    )
    .with_feature_route(Some(MODEL_ID), turbo_control_descriptor)
    .with_render_plan(compile_turbo_control);
    #[cfg(feature = "cuda")]
    {
        bindings = bindings
            .with_memory_registration(TURBO_CONTROL_MEMORY_REGISTRATION)
            .with_memory_fixture(
                control_memory_contract_surface_specs,
                registered_turbo_control_memory_contract,
            )
            .with_memory_surface_resolver(weights_free_turbo_control_surface_contract)
            .with_memory_behavior_registration(TURBO_CONTROL_MEMORY_BEHAVIOR);
    }
    route.bindings = bindings;
    route
}

pub(crate) fn route_base_control() -> gen_core::RouteRecord {
    let mut route = base::route_base();
    route.route_id = "z_image_control";
    route.structure.conditioning.kinds = vec![ConditioningKind::Control];
    route.structure.conditioning.control_kinds =
        Some(gen_core::AcceptedControlKinds::Only(CONTROL_KINDS.to_vec()));
    route.structure.conditioning.pass0_requirement =
        gen_core::Pass0ConditioningRequirement::RequiresControl;
    route.structure.conditioning.modules = vec!["z_image_vace_control"];
    route.structure.weight_plan.requires_control_overlay = true;
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut bindings = gen_core::RouteBindings::generator(
        base_control_descriptor,
        control_registry::load_base_control,
    )
    .with_feature_route(Some(base::MODEL_ID), base_control_descriptor)
    .with_render_plan(compile_base_control);
    #[cfg(feature = "cuda")]
    {
        bindings = bindings
            .with_memory_registration(BASE_CONTROL_MEMORY_REGISTRATION)
            .with_memory_fixture(
                control_memory_contract_surface_specs,
                registered_base_control_memory_contract,
            )
            .with_memory_surface_resolver(weights_free_base_control_surface_contract)
            .with_memory_behavior_registration(BASE_CONTROL_MEMORY_BEHAVIOR);
    }
    route.bindings = bindings;
    route
}

pub fn turbo_control_descriptor() -> ModelDescriptor {
    route_turbo_control().compute_descriptor()
}

pub fn base_control_descriptor() -> ModelDescriptor {
    route_base_control().compute_descriptor()
}

fn compile_control_for(
    spec: &LoadSpec,
    req: &GenerationRequest,
    is_base: bool,
) -> gen_core::Result<gen_core::RenderPlan> {
    let desc = if is_base {
        base_control_descriptor()
    } else {
        turbo_control_descriptor()
    };
    if req.features.low_res_start.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: low_res_start conflicts with pass-0 control conditioning (VACE control latents are full-resolution bound)",
            desc.id
        )));
    }
    if spec.control.is_none() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: requires a VACE control weight source in LoadSpec::control",
            desc.id
        )));
    }
    desc.capabilities.validate_request(desc.id, req)?;
    if req.prompt.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{}: prompt must not be empty",
            desc.id
        )));
    }
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!(
            "{}: steps must be >= 1",
            desc.id
        )));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(gen_core::Error::Msg(format!(
            "{}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            desc.id, req.width, req.height
        )));
    }
    let default_steps = if is_base {
        pipeline::BASE_DEFAULT_STEPS
    } else {
        DEFAULT_STEPS
    };
    let steps = req.steps.map(|s| s as usize).unwrap_or(default_steps);
    let sigmas = if is_base {
        pipeline::resolve_base_sigmas(req, steps)?
    } else {
        pipeline::resolve_turbo_sigmas(req, steps)?
    };
    let executed_steps = sigmas.len().saturating_sub(1);
    let sampler = gen_core::sampling::canonical_solver_or_euler(req.sampler.as_deref()).name();
    let guidance = if is_base {
        req.guidance.unwrap_or(4.0)
    } else {
        1.0
    };
    gen_core::RenderPlan::packed_image_trajectory(
        &desc,
        req,
        gen_core::render::PackedImagePass {
            pixel_stride: SIZE_MULTIPLE,
            token_width: 16,
            sampler,
            guidance,
            sigmas,
            executed_steps,
            capture_sigma: 0.0,
        },
    )
    .map(|plan| plan.bind_load(spec))
}

pub fn compile_turbo_control(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    compile_control_for(spec, req, false).map(Some)
}

pub fn compile_base_control(
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    compile_control_for(spec, req, true).map(Some)
}

/// Perturbed-Attention Guidance (PAG) capability descriptor for Z-Image Turbo and Edit.
pub fn turbo_pag_capability() -> gen_core::PagCapability {
    gen_core::PagCapability {
        num_blocks: 30,
        block_domain: std::borrow::Cow::Borrowed("layers"),
        default_layers: vec![15],
        default_start: 0.0,
        default_stop: 0.75,
        recommended_scale: 1.2,
        max_scale: gen_core::DEFAULT_PAG_MAX_SCALE,
    }
}

/// The experimental CFG renderer shares Turbo's weights and defaults. Candle Turbo's original
/// schedule is unshifted; retain it when only CFG is overridden, and honor an explicit shift.
fn experimental_cfg_request(req: &GenerationRequest) -> GenerationRequest {
    let mut request = req.clone();
    request.steps = Some(req.steps.unwrap_or(DEFAULT_STEPS as u32));
    request.guidance = Some(1.0 + req.guidance.unwrap_or(0.0));
    request.scheduler_shift = Some(req.scheduler_shift.unwrap_or(1.0));
    request
}

/// Primary typed route declaration for Candle Z-Image-Turbo (`REG-1`, `AP5`).
pub(crate) fn route_turbo() -> gen_core::RouteRecord {
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut bindings =
        gen_core::RouteBindings::generator(descriptor, load).with_render_plan(compile_turbo);
    #[cfg(feature = "cuda")]
    {
        bindings = bindings
            .with_memory_registration(TURBO_MEMORY_REGISTRATION)
            .with_memory_fixture(
                gen_core::candle_memory_contract_surface_specs,
                weights_free_turbo_memory_contract,
            )
            .with_memory_surface_resolver(weights_free_turbo_surface_contract)
            .with_memory_behavior_registration(TURBO_MEMORY_BEHAVIOR);
    }

    gen_core::RouteRecord {
        route_id: MODEL_ID,
        family: "z-image",
        backend: "candle",
        modality: Modality::Image,
        mac_only: false,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![gen_core::TextEncoderDeclaration::new("qwen3_4b", None)],
                encoder_contract: Some(ENCODER_CONTRACT),
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                kinds: vec![ConditioningKind::Reference],
                pass0_requirement: gen_core::Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: true,
                ..Default::default()
            },
            sampling: gen_core::SamplingSurface {
                samplers: candle_gen::curated_sampler_names(),
                schedulers: candle_gen::curated_scheduler_names(),
                supports_guidance: true,
                supports_true_cfg: false,
                ..Default::default()
            },
            denoiser: gen_core::ModelBinding::family_model(
                gen_core::FamilyModelSpec::new("z_image_turbo_dit")
                    .with_low_res_start()
                    .with_pag(turbo_pag_capability()),
            ),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("flux1_ae"),
                output_latent_space: Some(&candle_gen::gen_core::FLUX1_LATENT_SPACE),
                pixel_stride: Some(SIZE_MULTIPLE),
                token_width: Some(16),
                ..Default::default()
            },
            upsamplers: gen_core::UpsamplerSurface {
                components: vec!["pid_decoder"],
                max_temporal_upsample_rounds: 0,
            },
            adapters: gen_core::AdapterSurface::lora_and_lokr(
                gen_core::AdapterInstallPolicy::FusedMerge,
            ),
            residency: gen_core::ResidencySurface {
                lifetime_graph: vec![
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::PromptEncode,
                        ["qwen3_4b"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        ["z_image_turbo_dit"],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        ["flux1_ae"],
                        false,
                    ),
                ],
                supports_sequential_offload: true,
                ..Default::default()
            },
            weight_plan: gen_core::WeightPlanDeclaration {
                source_kind: gen_core::WeightSourceKind::DirectoryOrFile,
                supported_quants: &[],
                ..Default::default()
            },
            output: gen_core::OutputSurface {
                preview: gen_core::PreviewSupport::LatentProjection,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            ..Default::default()
        },
        limits: gen_core::ProductLimits {
            min_size: 256,
            max_size: 2048,
            size_floor: gen_core::SizeFloor::RangeChecked,
            supported_steps: gen_core::StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::DeterministicOffset,
            max_count: 8,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults {
            steps: Some(DEFAULT_STEPS as u32),
            guidance: Some(0.0),
            sampler: Some("euler"),
            dimensions: Some((1024, 1024)),
            ..Default::default()
        },
        bindings,
    }
}

/// Z-Image-Turbo's identity + the wired surface — computed from `route_turbo`.
pub fn descriptor() -> ModelDescriptor {
    route_turbo().compute_descriptor()
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
#[cfg(test)]
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
        // `data_len` is the encoder fixture's whole payload span, so this file is as big as the
        // fixture it re-prefixes. Flag it between the create and the extend (see `mark_sparse`).
        gen_core_testkit::mark_sparse(&path);
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
        let registry = register_memory_contract_surfaces(register_providers(
            gen_core::ProviderRegistryBuilder::new(),
        ))
        .build()
        .expect("Z-Image surface registry");
        let surfaces = registry
            .memory_contract_surfaces()
            .expect("weights-free Z-Image surfaces");

        for (provider_id, expects_control, fingerprint) in [
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
                assert!(!surface.composed, "{provider_id}");
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
                if expects_control {
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
            d.capabilities.supports_guidance,
            "Turbo supports explicit experimental CFG"
        );
        assert!(d.capabilities.supports_negative_prompt);
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
            memory: Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            features: candle_gen::gen_core::FeatureParams {
                pid: Some(candle_gen::gen_core::PidParams::enabled()),
                ..Default::default()
            },
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
            .expect_err("provider must reject a changed prepared component");
        assert!(
            matches!(
                error,
                gen_core::Error::Unsupported(ref reason)
                    if reason.starts_with("artifact seal mismatch after load: ")
            ),
            "got: {error:?}"
        );
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

    #[test]
    fn turbo_pag_capability_and_validation_match_contract() {
        use candle_gen::gen_core::PagConfig;

        let cap = turbo_pag_capability();
        assert_eq!(cap.num_blocks, 30);
        assert_eq!(cap.block_domain, "layers");
        assert_eq!(cap.default_layers, vec![15]);
        assert_eq!(cap.default_start, 0.0);
        assert_eq!(cap.default_stop, 0.75);
        assert_eq!(cap.recommended_scale, 1.2);
        assert_eq!(cap.max_scale, gen_core::DEFAULT_PAG_MAX_SCALE);

        let desc = descriptor();
        assert_eq!(desc.capabilities.pag.as_ref(), Some(&cap));

        let noop = GenerationRequest {
            prompt: "a test".into(),
            features: candle_gen::gen_core::FeatureParams {
                pag: Some(PagConfig {
                    scale: 0.0,
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        desc.capabilities.validate_request(MODEL_ID, &noop).unwrap();
        assert_eq!(
            desc.capabilities.resolve_pag(MODEL_ID, &noop).unwrap(),
            None
        );

        let active = GenerationRequest {
            prompt: "a test".into(),
            features: candle_gen::gen_core::FeatureParams {
                pag: Some(PagConfig {
                    scale: 1.2,
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        let resolved = desc
            .capabilities
            .resolve_pag(MODEL_ID, &active)
            .unwrap()
            .expect("active PAG resolves");
        assert_eq!(resolved.layers, vec![15]);
        assert_eq!(resolved.start, 0.0);
        assert_eq!(resolved.stop, 0.75);
        assert_eq!(resolved.execution_report(4).active_steps, 3);

        let oob = GenerationRequest {
            prompt: "a test".into(),
            features: candle_gen::gen_core::FeatureParams {
                pag: Some(PagConfig {
                    scale: 1.0,
                    layers: Some(vec![30]),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(desc.capabilities.validate_request(MODEL_ID, &oob).is_err());
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
        assert_eq!(
            explicit_generators,
            [
                "z_image",
                "z_image_control",
                "z_image_turbo",
                "z_image_turbo_control",
            ]
        );

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

    #[test]
    fn z_image_shared_driver_render_plans_and_low_res_start_conformance() {
        use candle_gen::gen_core::sampling::{FlowModelSampling, TimestepConvention};
        use candle_gen::gen_core::{
            BoundaryState, Conditioning, ControlKind, EstimateSource, GenerationRequest, Image,
            LoadSpec, LowResStartSpec, WeightsSource,
        };

        let registry = super::provider_registry().unwrap();
        let all_features = registry.all_route_feature_applicability();
        let plain_spec = LoadSpec::new(WeightsSource::Dir("/tmp/z-image".into()));
        let req = GenerationRequest {
            prompt: "a studio portrait".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            features: candle_gen::gen_core::FeatureParams {
                low_res_start: Some(LowResStartSpec::new(2, Some(0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        for id in ["z_image_turbo", "z_image"] {
            assert!(
                all_features[id]["low_res_start"].applicable,
                "{id} must support low_res_start"
            );
            let plan = registry
                .compile_render_plan(id, &plain_spec, &req)
                .unwrap()
                .expect("compiled plan");
            assert_eq!(plan.denoise_spans(), vec![0..2, 2..8]);
            assert_eq!(plan.phase_output_size(0), (512, 512));
            assert_eq!(plan.phase_output_size(1), (1024, 1024));
        }

        let img2img_req = GenerationRequest {
            prompt: "a studio portrait".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 64,
                    height: 64,
                    pixels: vec![128; 64 * 64 * 3],
                },
                strength: Some(0.5),
            }],
            features: candle_gen::gen_core::FeatureParams {
                low_res_start: Some(LowResStartSpec::new(1, Some(0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        let plan = registry
            .compile_render_plan("z_image", &plain_spec, &img2img_req)
            .unwrap()
            .expect("compiled img2img plan");
        assert_eq!(plan.executed_steps(), 4);
        assert_eq!(plan.denoise_spans(), vec![0..1, 1..4]);
        assert_eq!(plan.phase_output_size(0), (512, 512));

        let low_res_plan = *plan.low_res_start().unwrap();
        let sigma_k = plan.denoise_sigmas()[low_res_plan.k];
        let ms = FlowModelSampling::new(TimestepConvention::OneMinusSigma);
        let low_latent = vec![0.25f32; 16 * 64 * 64];
        let transitioned = candle_gen::gen_core::transition_clean_estimate_f32(
            BoundaryState::CleanEstimate {
                latent: &low_latent,
                source: EstimateSource::EndpointEvaluation,
            },
            1,
            16,
            8,
            &low_res_plan,
            sigma_k,
            &ms,
            plan.base_seed(),
            0,
            0,
            0,
        )
        .unwrap();
        assert_eq!(transitioned.len(), 16 * 128 * 128);
        assert!(transitioned.iter().all(|v| v.is_finite()));

        let control_spec = plain_spec
            .clone()
            .with_control(WeightsSource::Dir("/tmp/z-image-control".into()));
        let control_req = GenerationRequest {
            prompt: "a studio portrait".into(),
            width: 1024,
            height: 1024,
            steps: Some(8),
            conditioning: vec![Conditioning::Control {
                kind: ControlKind::Canny,
                image: Image {
                    width: 64,
                    height: 64,
                    pixels: vec![128; 64 * 64 * 3],
                },
                scale: Some(0.75),
            }],
            ..Default::default()
        };
        for control_id in ["z_image_turbo_control", "z_image_control"] {
            let feature_route = registry
                .feature_route_registrations()
                .find(|r| r.route_id == control_id)
                .expect("registered feature route");
            let planner = feature_route
                .render_plan
                .as_ref()
                .expect("control render plan");
            let plan = (planner.compile)(&control_spec, &control_req)
                .unwrap()
                .expect("compiled control plan");
            assert_eq!(plan.denoise_spans(), vec![0..8]);
            let mut bad = control_req.clone();
            bad.features.low_res_start = Some(LowResStartSpec::new(2, Some(0.5)));
            let err = (planner.compile)(&control_spec, &bad).unwrap_err();
            assert!(
                matches!(err, candle_gen::gen_core::Error::Unsupported(_)),
                "expected typed refusal for {control_id}, got {err:?}"
            );
            let app = &all_features[control_id]["low_res_start"];
            assert!(!app.applicable, "{control_id}");
            assert!(
                app.reasons.iter().any(|r| r.message.contains("control")),
                "{control_id}: {:?}",
                app.reasons
            );
        }
    }
}
