//! # candle-gen-ideogram
//!
//! The **Ideogram 4** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA)
//! sibling of `mlx-gen-ideogram`. Registers two engine ids:
//!
//! * **`ideogram_4`** — the quality variant: a 9.3B single-stream flow-matching DiT with **asymmetric
//!   two-DiT CFG** (a conditional + an unconditional transformer) + a Qwen3-VL-8B text encoder.
//!   `V4_QUALITY_48` default (48 steps, guidance 7.0).
//! * **`ideogram_4_turbo`** — the same base + the bundled ostris **TurboTime LoRA** installed at load
//!   (single DiT, CFG-free, ~8 steps; guidance inert).
//!
//! **Reuse:** Ideogram's VAE is the FLUX.2 `AutoencoderKLFlux2`, reused verbatim from
//! [`candle_gen_flux2`] (`Flux2Vae`), exactly as the MLX provider reuses `mlx-gen-flux2`. The Qwen3-VL
//! text path ([`text_encoder`]) is adapted from flux2's Qwen3 encoder (θ=5e6, 13 interleaved
//! captured states). The single-stream DiT + the denoise pipeline are ported here.
//!
//! **Status:** both `ideogram_4` (quality, asymmetric two-DiT CFG) and `ideogram_4_turbo` (CFG-free
//! single DiT + the bundled TurboTime LoRA merged at load, [`adapters`]) are wired end-to-end —
//! Qwen3-VL text encode → single-stream DiT → VAE decode — for **T2I** (sc-6596) and **edit**
//! (sc-6598: img2img/Remix `Reference` + mask inpaint `Mask`, via the FLUX.2 VAE encoder). `backend =
//! "candle"`, `mac_only = false`.

pub mod adapters;
pub mod config;
pub mod loader;
mod memory_strategy;
pub mod pipeline;
pub mod preview;
pub mod quant;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, PidWeights, Progress, Quant,
    WeightsSource,
};

pub use adapters::TurboLoraReport;
/// Re-export the pinned width/height stride at the crate root so SceneWorks can tie each advertised
/// Ideogram image bucket to `candle_gen_ideogram::SIZE_MULTIPLE` (sc-12612) instead of a hand-copied
/// literal.
pub use config::SIZE_MULTIPLE;

/// Release a component only after the backend fence succeeds. On a failed fence the allocation is
/// intentionally retained because queued device work may still reference it.
pub(crate) fn synchronized_release<T, E>(
    component: T,
    synchronize: impl FnOnce() -> Result<(), E>,
) -> Result<(), E> {
    match synchronize() {
        Ok(()) => {
            drop(component);
            Ok(())
        }
        Err(error) => {
            std::mem::forget(component);
            Err(error)
        }
    }
}

use config::{MODEL_ID, MODEL_ID_TURBO, RES_MAX, RES_MIN};
use pipeline::Components;

/// A lazily-loaded Ideogram 4 generator. `turbo` selects the CFG-free single-DiT + TurboTime LoRA
/// path; otherwise the asymmetric two-DiT quality path. Components are loaded on the first
/// `generate` and cached.
pub struct Ideogram4Generator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    turbo: bool,
    device: Device,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    adapters: Vec<AdapterSpec>,
    components: Mutex<Option<Arc<Components>>>,
    lifecycle: Mutex<()>,
    memory_contract: gen_core::MemoryProviderContract,
    load_receipt: Option<memory_strategy::IdeogramLoadReceipt>,
    memory_admission: memory_strategy::AdmissionRegistry,
}

impl Ideogram4Generator {
    fn components(&self) -> gen_core::Result<Arc<Components>> {
        candle_gen::cached(&self.components, || {
            let components = if self.turbo {
                pipeline::load_components_turbo(
                    &self.root,
                    &self.device,
                    self.pid_spec.as_ref(),
                    &self.adapters,
                )?
            } else {
                pipeline::load_components(
                    &self.root,
                    &self.device,
                    self.pid_spec.as_ref(),
                    &self.adapters,
                )?
            };
            Ok(Arc::new(components))
        })
    }
}

impl Generator for Ideogram4Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_contract)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        if let Some(receipt) = &self.load_receipt {
            if let Err(error) = receipt.ensure_unchanged() {
                self.memory_admission.clear_approval();
                return gen_core::MemorySafetyDecision::Reject {
                    reason: error.to_string(),
                };
            }
        }
        match memory_strategy::validate_context(
            self.descriptor.id,
            &self.memory_contract,
            context,
            self.load_receipt
                .as_ref()
                .map(|receipt| receipt.route.tier)
                .unwrap_or_else(
                    || match self.root.file_name().and_then(|name| name.to_str()) {
                        Some("q4") => Some(Quant::Q4),
                        Some("q8") => Some(Quant::Q8),
                        _ => None,
                    },
                ),
        ) {
            Ok(()) => match self.memory_admission.approve(context) {
                Ok(()) => gen_core::MemorySafetyDecision::Accept,
                Err(error) => gen_core::MemorySafetyDecision::Reject {
                    reason: error.to_string(),
                },
            },
            Err(error) => {
                self.memory_admission.clear_approval();
                gen_core::MemorySafetyDecision::Reject {
                    reason: error.to_string(),
                }
            }
        }
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        if let Some(receipt) = &self.load_receipt {
            receipt.ensure_unchanged()?;
        }
        let tier = self
            .load_receipt
            .as_ref()
            .map(|receipt| receipt.route.tier)
            .unwrap_or(None);
        memory_strategy::validate_context(
            self.descriptor.id,
            &self.memory_contract,
            context,
            tier,
        )?;
        Ok(Some(Box::new(
            memory_strategy::IdeogramMemoryScope::new_bound(
                self.descriptor.id,
                self.device.clone(),
                &self.memory_contract,
                context,
                self.memory_admission.clone(),
            )?,
        )))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{}: prompt must not be empty",
                self.descriptor.id
            )));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!(
                "{}: steps must be >= 1",
                self.descriptor.id
            )));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "{}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                self.descriptor.id, req.width, req.height
            )));
        }
        // Edit: an inpaint `Mask` is meaningless without a source `Reference` to keep/blend against
        // (the capability floor admits both kinds individually; this enforces the pairing). Multiple
        // references / masks are caught in `resolve_edit` at generate time.
        let has_ref = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Reference { .. }));
        let has_mask = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Mask { .. }));
        if has_mask && !has_ref {
            return Err(gen_core::Error::Msg(format!(
                "{}: an inpaint mask requires a reference (source) image",
                self.descriptor.id
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
        self.memory_admission.consume_for_generate(req)?;
        if let Some(receipt) = &self.load_receipt {
            receipt.ensure_unchanged()?;
        }
        let memory = req.memory.unwrap_or_default();
        if memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: bounded decode, attention, and transformer residency are Missing",
                self.descriptor.id
            )));
        }
        let images = if memory.stage_residency {
            if let Some(warm) = candle_gen::lock_recover(&self.components).take() {
                synchronized_release(warm, || self.device.synchronize())
                    .map_err(gen_core::Error::backend)?;
            }
            pipeline::render_staged(
                &self.root,
                self.turbo,
                self.pid_spec.as_ref(),
                &self.adapters,
                req,
                &self.device,
                on_progress,
            )?
        } else {
            let comps = self.components()?;
            pipeline::render(&comps, req, &self.device, on_progress)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

/// Ideogram 4 (quality) descriptor — asymmetric two-DiT CFG; no text negative prompt (the negative
/// branch is the trained unconditional DiT). Edit (sc-6598): img2img/Remix via a source `Reference`,
/// and mask inpaint via a `Mask` (white = repaint) alongside the `Reference`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&candle_gen::gen_core::FLUX2_PACKED_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "ideogram",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_guidance: true,
            // Edit (sc-6303/6330 → candle sc-6598): one img2img/inpaint source Reference + optional
            // inpaint Mask. No control/pose/multi-reference. Works in both quality and turbo.
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::Mask],
            supports_lora: true,
            supports_lokr: true,
            schedulers: vec!["flow_match_euler"],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: 8,
            // sc-9607: advertise the packed tiers so the worker's `resolve_quant` / A-B quant toggle
            // engages off-Mac (the resolved q4/q8 turnkey subdir self-describes; `build` no-ops the
            // requested quant — see below). Both quality + turbo share this via `descriptor_turbo`.
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Per-step latent previews (epic 16948, sc-16955). Ideogram drives no shared sampler, so
            // its bespoke flow-match loop emits directly (`crate::preview`); the VAE it loads is the
            // FLUX.2 one tensor-for-tensor, so it reuses that fit rather than introducing one.
            // `candle-gen-catalog`'s guard derives this flag from the sources — including from a
            // bespoke crate's direct emission call — so it cannot run ahead of or behind the wiring.
            supports_preview: true,
            ..Default::default()
        },
    }
}

/// Ideogram 4 Turbo descriptor — same base, CFG-free single DiT + the bundled TurboTime LoRA;
/// guidance is inert (`supports_guidance = false`).
pub fn descriptor_turbo() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = MODEL_ID_TURBO;
    d.capabilities.supports_guidance = false;
    d
}

fn build(
    spec: &LoadSpec,
    descriptor: ModelDescriptor,
    turbo: bool,
) -> gen_core::Result<Box<dyn Generator>> {
    let provider_id = descriptor.id;
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{} expects a snapshot directory (transformer/ [unconditional_transformer/] \
                 text_encoder/ vae/ tokenizer/), not a single .safetensors file",
                descriptor.id
            )));
        }
    };
    // User LoRA/LoKr stacks are installed with the bundled TurboTime adapter (turbo) or on both CFG
    // DiTs (quality). ControlNet+IP-Adapter overlays remain separate unsupported combinations.
    // img2img/Remix + mask inpaint edit is NOT a LoadSpec overlay —
    // it arrives as per-request `Reference`/`Mask` conditioning (sc-6598), handled in the pipeline, so
    // it is unaffected by these load-time rejects.
    // sc-9607: `spec.quantize` (Q4/Q8) is ACCEPTED and is a no-op. The per-tier turnkey is already
    // MLX-packed; `loader::linear_detect` builds each `QLinear` (shared `AdaptLinear`) with a packed
    // base straight from the packed parts (sc-9412), so the resolved q4/q8 subdir self-describes its
    // tier and no on-the-fly quant
    // pass runs (there is no post-load `.quantize()` to double-quantize). Advertising `supported_quants`
    // lets the worker's A-B tier toggle engage; the requested quant is carried for the recipe only.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not support ControlNet / IP-Adapter overlays (img2img/mask edit is \
             request-time Reference/Mask conditioning, which is supported)",
            descriptor.id
        )));
    }
    let device = candle_gen::default_device()?;
    let (load_receipt, memory_contract) =
        match memory_strategy::IdeogramLoadReceipt::capture(provider_id, spec) {
            Ok(receipt) => {
                let contract = memory_strategy::contract_from_receipt(provider_id, spec, &receipt);
                (Some(receipt), contract)
            }
            #[cfg(test)]
            Err(_) if !root.exists() => {
                let mut fixture_spec = spec.clone();
                fixture_spec.resolved_route = Some(provider_id.to_owned());
                (
                    None,
                    memory_strategy::uncalibrated_contract(provider_id, &fixture_spec)?,
                )
            }
            Err(error) => return Err(error),
        };
    Ok(Box::new(Ideogram4Generator {
        descriptor,
        root,
        turbo,
        device,
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike control/IP above, it is not
        // rejected — `None` simply keeps the byte-exact native-VAE path.
        pid_spec: spec.pid.clone(),
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
        lifecycle: Mutex::new(()),
        memory_contract,
        load_receipt,
        memory_admission: memory_strategy::AdmissionRegistry::new(provider_id),
    }))
}

/// Construct a lazy candle Ideogram 4 (quality) generator. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a candle-readable (bf16) Ideogram 4 snapshot.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor(), false)
}

/// Construct a lazy candle Ideogram 4 **Turbo** generator (CFG-free single DiT + bundled TurboTime
/// LoRA). The snapshot must additionally carry [`config::TURBO_LORA_FILE`].
pub fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor_turbo(), true)
}

candle_gen::register_generators! {
    pub(crate) const QUALITY_REGISTRATION = descriptor => load
}
candle_gen::register_generators! {
    pub(crate) const TURBO_REGISTRATION = descriptor_turbo => load_turbo
}

/// Add all Candle Ideogram providers to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(QUALITY_REGISTRATION)
        .register_generator(TURBO_REGISTRATION);
    #[cfg(feature = "cuda")]
    let registry = register_memory_contract_surfaces(registry)
        .register_memory_behavior(QUALITY_MEMORY_BEHAVIOR)
        .register_memory_behavior(TURBO_MEMORY_BEHAVIOR);
    registry
}

fn quality_memory_contract(spec: &LoadSpec) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::provider_contract(MODEL_ID, spec)
}

fn turbo_memory_contract(spec: &LoadSpec) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::provider_contract(MODEL_ID_TURBO, spec)
}

fn quality_weights_free_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_contract(MODEL_ID, spec)
}

fn turbo_weights_free_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_contract(MODEL_ID_TURBO, spec)
}

fn quality_surface_contract(
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_surface_contract(MODEL_ID, surface)
}

fn turbo_surface_contract(
    surface: &gen_core::MemoryContractSurfaceSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    memory_strategy::weights_free_surface_contract(MODEL_ID_TURBO, surface)
}

const QUALITY_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID,
    contract: quality_memory_contract,
    safety_check: memory_strategy::registered_safety_check,
};

const TURBO_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID_TURBO,
    contract: turbo_memory_contract,
    safety_check: memory_strategy::registered_safety_check,
};

#[cfg(feature = "cuda")]
const QUALITY_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            memory_strategy::registered_begin_request(MODEL_ID, spec, contract, context)
        },
    };

#[cfg(feature = "cuda")]
const TURBO_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID_TURBO,
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            memory_strategy::registered_begin_request(MODEL_ID_TURBO, spec, contract, context)
        },
    };

pub fn register_memory_contract_surfaces(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(QUALITY_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            provider_id: MODEL_ID,
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            contract: quality_weights_free_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: MODEL_ID,
                contract: quality_surface_contract,
            },
        )
        .register_memory_strategy(TURBO_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            provider_id: MODEL_ID_TURBO,
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            contract: turbo_weights_free_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: MODEL_ID_TURBO,
                contract: turbo_surface_contract,
            },
        )
}

/// Build the complete explicit Candle Ideogram provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit, ["ideogram_4", "ideogram_4_turbo"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::Image;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct DropProbe(Arc<AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn synchronized_release_drops_after_success_and_retains_after_failed_fence() {
        let dropped = Arc::new(AtomicUsize::new(0));
        synchronized_release(DropProbe(Arc::clone(&dropped)), || Ok::<_, ()>(())).unwrap();
        assert_eq!(dropped.load(Ordering::SeqCst), 1);

        let error = synchronized_release(DropProbe(Arc::clone(&dropped)), || Err::<(), _>("fence"))
            .expect_err("failed synchronization must surface");
        assert_eq!(error, "fence");
        assert_eq!(
            dropped.load(Ordering::SeqCst),
            1,
            "in-flight storage must be retained rather than dropped"
        );
    }

    #[test]
    fn registers_both_ids_as_candle() {
        for id in [MODEL_ID, MODEL_ID_TURBO] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            let g = crate::provider_registry()
                .unwrap()
                .load(id, &spec)
                .unwrap_or_else(|_| panic!("{id} is registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "ideogram");
            assert_eq!(g.descriptor().backend, "candle");
            assert!(!g.descriptor().capabilities.mac_only);
        }
    }

    #[test]
    fn descriptor_surfaces() {
        let q = descriptor();
        assert!(q.capabilities.supports_guidance);
        assert!(!q.capabilities.supports_negative_prompt);
        // sc-9607: advertises the packed tiers so the worker A-B toggle engages (turbo inherits it).
        assert_eq!(q.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(
            descriptor_turbo().capabilities.supported_quants,
            &[Quant::Q4, Quant::Q8]
        );
        // Edit surface (sc-6598): img2img Reference + inpaint Mask.
        assert!(q.capabilities.accepts(ConditioningKind::Reference));
        assert!(q.capabilities.accepts(ConditioningKind::Mask));
        assert!(!q.capabilities.accepts(ConditioningKind::Control));
        let t = descriptor_turbo();
        assert_eq!(t.id, MODEL_ID_TURBO);
        assert!(!t.capabilities.supports_guidance);
        // Turbo shares the edit surface.
        assert!(t.capabilities.accepts(ConditioningKind::Reference));
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let ok = GenerationRequest {
            prompt: "a neon city skyline at dusk".into(),
            guidance: Some(7.0),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised Ideogram
        // bucket to. Pin the value and mutation-check that a size which is a multiple of 8 (the VAE
        // scale) but not SIZE_MULTIPLE (16) is still rejected with the stride error, and an on-stride
        // size passes.
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

    #[test]
    fn validate_edit_conditioning_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let img = |w: u32, h: u32| Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        };
        let base = GenerationRequest {
            prompt: "a fox".into(),
            width: 512,
            height: 512,
            ..Default::default()
        };
        // img2img: a single source Reference is accepted.
        assert!(g
            .validate(&GenerationRequest {
                conditioning: vec![Conditioning::Reference {
                    image: img(512, 512),
                    strength: Some(0.6),
                }],
                ..base.clone()
            })
            .is_ok());
        // inpaint: Reference + Mask is accepted.
        assert!(g
            .validate(&GenerationRequest {
                conditioning: vec![
                    Conditioning::Reference {
                        image: img(512, 512),
                        strength: None,
                    },
                    Conditioning::Mask {
                        image: img(512, 512)
                    },
                ],
                ..base.clone()
            })
            .is_ok());
        // A Mask without a Reference is rejected (pairing).
        let e = g
            .validate(&GenerationRequest {
                conditioning: vec![Conditioning::Mask {
                    image: img(512, 512),
                }],
                ..base.clone()
            })
            .unwrap_err()
            .to_string();
        assert!(e.contains("requires a reference"), "got: {e}");
        // An out-of-surface Control conditioning is rejected by the capability floor.
        assert!(g
            .validate(&GenerationRequest {
                conditioning: vec![Conditioning::Control {
                    image: img(512, 512),
                    kind: candle_gen::gen_core::ControlKind::Pose,
                    scale: Some(1.0),
                }],
                ..base
            })
            .is_err());
    }

    #[test]
    fn both_variants_reach_pipeline() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-ideogram-snap".into()));
        let req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        // Both quality and turbo are wired — generate() passes the capability gate and fails on the
        // missing snapshot dir (a load error, NOT Unsupported), proving the pipeline path is reached.
        for id in [MODEL_ID, MODEL_ID_TURBO] {
            let g = crate::provider_registry().unwrap().load(id, &spec).unwrap();
            let err = g.generate(&req, &mut |_| {}).unwrap_err();
            assert!(
                !matches!(err, gen_core::Error::Unsupported(_)),
                "{id} should reach the pipeline, got {err:?}"
            );
        }
    }

    #[test]
    fn load_accepts_adapters_and_rejects_single_file() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let file = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        assert!(load(&file).is_err());
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(load(&lora).is_ok());
    }
}
