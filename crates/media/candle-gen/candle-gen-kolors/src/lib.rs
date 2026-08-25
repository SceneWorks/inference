//! # candle-gen-kolors
//!
//! The **Kolors** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling
//! of `mlx-gen-kolors`. It implements the backend-neutral [`gen_core::Generator`] contract and
//! exposes the candle Kolors generator through its explicit family catalog.
//!
//! **txt2img + source-image img2img:** Kolors is a bilingual (Chinese/English) SDXL-family model —
//! the SDXL UNet + SDXL VAE with a **ChatGLM3-6B** text encoder in place of dual CLIP. `pipeline`
//! runs it through the contract: ChatGLM3 encode (penultimate hidden state → cross-attention
//! context, last-token last-layer state → pooled add-embedding) → either seeded noise or a
//! VAE-encoded reference plus the selected schedule tail → the Kolors UNet (real CFG over the
//! leading-Euler 1100-step schedule) → the SDXL VAE, emitting `Progress`, honoring `req.cancel`,
//! with **deterministic CPU-seeded noise** (sc-3673) so output is launch-portable per seed.
//!
//! The descriptor advertises the wired surface — txt2img, single-reference img2img, and packed
//! **Q4/Q8** MLX-tier inference (sc-10819, epic 9083), plus user LoRA/LoKr on the SDXL-family UNet.
//! ControlNet-pose and
//! IP-Adapter remain separately wired bespoke Candle providers and are deliberately rejected by the
//! registered base/img2img loader rather than silently dropped. `backend` is `"candle"` and
//! `mac_only` is `false`.

mod chatglm3;
// Shared Kolors pipeline scaffolding (sc-9001 / F-021): the time_ids / initial-noise / decode /
// CFG-batched-encode / curated-σ-prior blocks that were copy-pasted across the three entry points.
mod common;
mod config;
pub mod memory_strategy;
mod pipeline;
// Per-step latent preview wiring (epic 16948, sc-16954). No coefficients of its own — Kolors shares
// the SDXL four-channel latent space (one byte-identical VAE file, `scaling_factor` 0.13025) and
// projects through `candle_gen_sdxl::preview` rather than restating the fit.
pub mod preview;
mod sampler;
mod tokenizer;
mod training;
mod unet;

// IP-Adapter-Plus reference-image (identity) provider (sc-5488, epic 5480) — CLIP ViT-L/14-336 image
// tokens injected into the vendored SDXL `UNet2DConditionModel` (candle-gen-sdxl) alongside the
// encoder_hid_proj-projected ChatGLM3 text path, denoised with the Kolors leading-Euler sampler.
// Invoked directly by the worker (a bespoke reference stream), not gen-core-registered.
pub mod ip_provider;

// ControlNet (strict-pose) provider (sc-5489, epic 5480) — a rendered OpenPose skeleton drives the
// `Kwai-Kolors/Kolors-ControlNet-Pose` SDXL-family `ControlNetModel`, whose per-block residuals are
// added into the vendored SDXL UNet (no IP installed). Invoked directly by the worker (a bespoke pose
// stream), not gen-core-registered.
pub mod control;

// Kolors IP-Adapter-Plus real-weight GPU validation (sc-5488) — env-driven, `#[ignore]`d integration
// test (the Kolors sibling of the SDXL IP-Adapter Phase-5 harness).
#[cfg(test)]
mod ip_validate;

// Kolors ControlNet (strict-pose) real-weight GPU validation (sc-5489) — env-driven, `#[ignore]`d
// integration test (with-control vs no-control pixel diff + mid-denoise cancel).
#[cfg(test)]
mod control_validate;

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, Conditioning, GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor,
    PidWeights, Progress, WeightsSource,
};

pub use config::{descriptor, MODEL_ID, SIZE_MULTIPLE};
pub use control::{KolorsControl, KolorsControlPaths, KolorsControlRequest, DEFAULT_CONTROL_SCALE};
pub use ip_provider::{
    IpAdapterKolors, IpAdapterKolorsPaths, IpAdapterKolorsRequest, DEFAULT_IP_ADAPTER_SCALE,
};
use sampler::NUM_TRAIN_TIMESTEPS;

use pipeline::{Components, Pipeline};

/// A loaded candle Kolors generator. Loading is **lazy**: `load` does no file I/O (registry
/// introspection against a missing path still resolves), and the heavy components (ChatGLM3 + UNet +
/// VAE) are built on the first [`generate`](Generator::generate) call and then cached.
pub struct KolorsGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    adapters: Vec<candle_gen::gen_core::AdapterSpec>,
    components: Mutex<Option<Components>>,
    /// Serializes cache eviction and staged component lifetimes.  A staged request is authoritative:
    /// it first retires any warm resident set and cannot repopulate it.
    lifecycle: Mutex<()>,
    memory_contract: gen_core::MemoryProviderContract,
    load_seal: Option<memory_strategy::KolorsLoadSeal>,
    memory_admission: memory_strategy::AdmissionRegistry,
}

impl KolorsGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // `?` bridges the candle-side `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

fn evict_warm_for_staged<T>(slot: &Mutex<Option<T>>) -> Option<T> {
    candle_gen::lock_recover(slot).take()
}

fn release_warm_after_synchronize<T>(device: &Device, component: T) -> gen_core::Result<()> {
    device.synchronize().map_err(gen_core::Error::backend)?;
    drop(component);
    Ok(())
}

impl Generator for KolorsGenerator {
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
        let tier = match &self.load_seal {
            Some(seal) => match seal.ensure_unchanged() {
                Ok(()) => seal.tier(),
                Err(error) => {
                    self.memory_admission.clear_approval();
                    return gen_core::MemorySafetyDecision::Reject {
                        reason: error.to_string(),
                    };
                }
            },
            None => {
                self.memory_admission.clear_approval();
                return gen_core::MemorySafetyDecision::Reject {
                    reason: "kolors: physical load seal is unavailable".into(),
                };
            }
        };
        match memory_strategy::validate_context(&self.memory_contract, context, tier) {
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
        let seal = self.load_seal.as_ref().ok_or_else(|| {
            gen_core::Error::Unsupported("kolors: physical load seal is unavailable".into())
        })?;
        seal.ensure_unchanged()?;
        let tier = seal.tier();
        memory_strategy::validate_context(&self.memory_contract, context, tier)?;
        Ok(Some(Box::new(memory_strategy::KolorsMemoryScope::new(
            self.device.clone(),
            &self.memory_contract,
            context,
            self.memory_admission.clone(),
        )?)))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor accepts only the registered Reference img2img conditioning.
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        let reference_count = req
            .conditioning
            .iter()
            .filter(|conditioning| matches!(conditioning, Conditioning::Reference { .. }))
            .count();
        if reference_count > 1 {
            return Err(gen_core::Error::Msg(
                "kolors: multiple reference images are not supported".into(),
            ));
        }
        if req.strength.is_some() && reference_count == 0 {
            return Err(gen_core::Error::Unsupported(
                "kolors: img2img strength requires Reference conditioning".into(),
            ));
        }
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(
                "kolors: prompt must not be empty".into(),
            ));
        }
        // `steps == 0` would VAE-decode undenoised noise; `steps > NUM_TRAIN_TIMESTEPS` collapses the
        // leading schedule (every timestep maps to one value). Reject both (the sampler errors too).
        if let Some(steps) = req.steps {
            if steps == 0 || steps as usize > NUM_TRAIN_TIMESTEPS {
                return Err(gen_core::Error::Msg(format!(
                    "kolors: steps must be in 1..={NUM_TRAIN_TIMESTEPS} (got {steps})"
                )));
            }
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "kolors: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
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
        // A pre-canceled request must not trigger the lazy multi-gigabyte component load. Render
        // has additional checkpoints after request setup, at every image, and before decode.
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        self.memory_admission.consume_for_generate(req)?;
        if let Some(seal) = &self.load_seal {
            seal.ensure_unchanged()?;
        }
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        let memory = req.memory.unwrap_or_default();
        if memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks {
            return Err(gen_core::Error::Unsupported(
                "kolors: Decode, Attention, and Transformer memory strategies are Missing".into(),
            ));
        }
        let pipe = Pipeline::load(
            &self.root,
            &self.device,
            self.pid_spec.clone(),
            self.adapters.clone(),
        )
        .with_load_seal(self.load_seal.clone());
        let images = if memory.stage_residency {
            if let Some(warm) = evict_warm_for_staged(&self.components) {
                release_warm_after_synchronize(&self.device, warm)?;
            } else {
                self.device
                    .synchronize()
                    .map_err(gen_core::Error::backend)?;
            }
            pipe.render_staged(req, on_progress)?
        } else {
            let components = self.components(&pipe)?;
            pipe.render(req, &components, on_progress)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

/// Construct the (lazy) candle Kolors generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `Kwai-Kolors/Kolors-diffusers` snapshot OR a packed
/// `SceneWorks/kolors-mlx` q4/q8 tier (`text_encoder/`, `tokenizer/`, `unet/`, `vae/`, with
/// `tokenizer/tokenizer.json` materialized). A packed tier is auto-detected from disk (sc-10819), so
/// `spec.quantize` is an advisory no-op. LoRA adapters and control / IP-adapter overlays are still
/// rejected — none are wired on the candle lane, so refusing is more honest than silently dropping them.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "kolors expects a Kolors-diffusers snapshot directory (text_encoder/ tokenizer/ \
                 unet/ vae/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    // Packed q4/q8 MLX tiers are wired end-to-end (sc-10819, epic 9083): the tier is packed-detected
    // from disk (`unet/` & `text_encoder/` `config.json` `quantization` blocks; see
    // `pipeline::load_components`), so the `LoadSpec::quantize` overlay is an advisory no-op on an
    // already-packed tier — exactly as SDXL/boogu/flux2-dev treat it. No reject here.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle kolors registered base/img2img generator does not accept control / IP-adapter \
             overlays; use the dedicated Kolors control or IP-Adapter provider"
                .into(),
        ));
    }
    let device = candle_gen::default_device()?;
    let (load_seal, memory_contract) = if root.exists() {
        let seal = memory_strategy::KolorsLoadSeal::capture_load(
            &root,
            &spec.adapters,
            spec.pid.as_ref(),
        )?;
        let contract = seal.contract().clone();
        (Some(seal), contract)
    } else {
        // Registry/catalog introspection intentionally permits a missing lazy root. It cannot begin
        // an admitted request: physical-tier and receipt validation both require the real source.
        (None, memory_strategy::provider_contract())
    };
    Ok(Box::new(KolorsGenerator {
        descriptor: descriptor(),
        root,
        device,
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike adapters/quant/control above,
        // it is not rejected — `None` simply keeps the byte-exact native-VAE path.
        pid_spec: spec.pid.clone(),
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
        lifecycle: Mutex::new(()),
        memory_contract,
        load_seal,
        memory_admission: memory_strategy::AdmissionRegistry::new(),
    }))
}

// Link-time self-registration into gen-core's model registry. Linking this crate makes
// the explicit family and platform catalogs resolve the candle generator.
candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

/// Add the Candle Kolors provider to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(REGISTRATION)
        .register_trainer(training::TRAINER_REGISTRATION);
    // Unconditional (no `cfg(feature = "cuda")`): the memory seams below construct on any platform
    // and open no device, so a selector can price Kolors before load on CPU catalogs too.
    register_memory_contract_surfaces(registry)
}

/// Register every Kolors memory route's weights-free contract and behavior surface.
///
/// Three routes, two shapes. The registered `kolors` generator takes an ordinary
/// `register_memory_strategy`. The bespoke IP-Adapter and strict-pose compositions are assembled
/// from typed path structs by the worker and have no generator descriptor, so they take
/// `register_composed_memory_strategy` — the seam gen-core provides for a generator-less route
/// (`z_image_control` is the same shape). An ordinary registration would make
/// `ProviderRegistryBuilder::build` fail with "has no matching generator registration", and the
/// catalog's `BESPOKE_MEMORY_ROUTE_WAIVERS` cannot hold them either: that table is keyed on
/// `BESPOKE_UTILITY_CRATES` (descriptor-less crates), and `candle-gen-kolors` ships a descriptor.
pub fn register_memory_contract_surfaces(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(memory_strategy::memory_registration())
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: memory_strategy::surface_specs,
            provider_id: MODEL_ID,
            contract: memory_strategy::weights_free_contract,
        })
        .register_memory_behavior(memory_strategy::MEMORY_BEHAVIOR)
        .register_composed_memory_strategy(memory_strategy::IP_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: memory_strategy::surface_specs,
            provider_id: memory_strategy::IP_PROVIDER_ID,
            contract: memory_strategy::ip_composed_contract,
        })
        .register_memory_behavior(memory_strategy::IP_MEMORY_BEHAVIOR)
        .register_composed_memory_strategy(memory_strategy::CONTROL_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: memory_strategy::surface_specs,
            provider_id: memory_strategy::CONTROL_PROVIDER_ID,
            contract: memory_strategy::control_composed_contract,
        })
        .register_memory_behavior(memory_strategy::CONTROL_MEMORY_BEHAVIOR)
}

/// Build the complete explicit Candle Kolors provider catalog.
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

        assert_eq!(explicit, ["kolors"]);
        let trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(trainers, ["kolors"]);
    }

    /// sc-20762 review (BLOCKER 4): the crate used to register **no** memory route, so a selector
    /// could not price any Kolors route before load. `register_providers` alone — no CUDA gate, no
    /// catalog help — must now carry all three routes with their weights-free contract fixtures and
    /// behavior seams, and every one must survive executable conformance.
    #[test]
    fn register_providers_alone_carries_every_kolors_memory_route() {
        use candle_gen::gen_core::{LoadSpec, WeightsSource};
        use std::collections::BTreeSet;

        let expected = BTreeSet::from([
            super::MODEL_ID,
            super::memory_strategy::IP_PROVIDER_ID,
            super::memory_strategy::CONTROL_PROVIDER_ID,
        ]);
        let registry = super::provider_registry().unwrap();
        assert_eq!(
            registry
                .memory_strategy_registrations()
                .map(|registration| registration.provider_id)
                .collect::<BTreeSet<_>>(),
            expected
        );
        assert_eq!(
            registry
                .memory_contract_fixture_registrations()
                .map(|registration| registration.provider_id)
                .collect::<BTreeSet<_>>(),
            expected
        );
        assert_eq!(
            registry
                .memory_behavior_registrations()
                .map(|registration| registration.provider_id)
                .collect::<BTreeSet<_>>(),
            expected
        );

        gen_core_testkit::memory_contract_surface_registry_conformance(&registry);
        gen_core_testkit::memory_strategy_registry_conformance(
            &registry,
            &LoadSpec::new(WeightsSource::Dir("/__weights_free__".into())),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{AdapterKind, AdapterSpec, Conditioning, Image, Modality, Quant};

    #[test]
    fn kolors_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .expect("candle kolors is registered");
        assert_eq!(g.descriptor().id, "kolors");
        assert_eq!(g.descriptor().family, "kolors");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn validate_accepts_txt2img_and_single_reference_img2img() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();

        let ok = GenerationRequest {
            prompt: "一只猫 / a cat holding a lit candle".into(),
            guidance: Some(5.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());

        let reference = GenerationRequest {
            prompt: "restyle the source as watercolor".into(),
            strength: Some(0.45),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 8,
                    height: 8,
                    pixels: vec![127; 8 * 8 * 3],
                },
                strength: Some(0.6),
            }],
            ..Default::default()
        };
        assert!(g.validate(&reference).is_ok());

        for bad in [
            GenerationRequest::default(), // empty prompt
            GenerationRequest {
                prompt: "x".into(),
                width: 1020, // not a multiple of 8
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(NUM_TRAIN_TIMESTEPS as u32 + 1),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                conditioning: vec![
                    Conditioning::Reference {
                        image: Image::default(),
                        strength: None,
                    },
                    Conditioning::Reference {
                        image: Image::default(),
                        strength: None,
                    },
                ],
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                strength: Some(0.5),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised Kolors
        // bucket to. Pin the value and mutation-check that a size which is a multiple of 4 but not
        // SIZE_MULTIPLE (8) is still rejected with the stride error, and an on-stride size passes.
        assert_eq!(SIZE_MULTIPLE, 8);
        let off_stride = g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1020, // 255×4 — a multiple of 4 but not SIZE_MULTIPLE
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiples of 8"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1024, // 128×8 — on-stride
                ..Default::default()
            })
            .is_ok());
    }

    #[test]
    fn pre_cancelled_zero_strength_native_and_curated_skip_component_load() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();

        for sampler in [None, Some("dpmpp_2m".to_string())] {
            let request = GenerationRequest {
                prompt: "edit the reference".into(),
                width: 512,
                height: 512,
                steps: Some(10),
                sampler,
                conditioning: vec![Conditioning::Reference {
                    image: Image {
                        width: 8,
                        height: 8,
                        pixels: vec![127; 8 * 8 * 3],
                    },
                    strength: Some(0.0),
                }],
                ..Default::default()
            };
            request.cancel.cancel();
            let err = g
                .generate(&request, &mut |_| {})
                .expect_err("pre-cancellation must win before the missing snapshot is opened");
            assert!(matches!(err, gen_core::Error::Canceled));
        }
    }

    /// sc-7124: the curated ε/DDPM menu is advertised, so `validate` accepts a curated sampler +
    /// scheduler pair (the worker may send one) and the native `euler_discrete` default, while still
    /// rejecting an unadvertised name — the shared `validate_request` only passes a named sampler that is
    /// in `descriptor().samplers`. GPU-free (lazy generator).
    #[test]
    fn validate_accepts_curated_sampler_and_scheduler() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();

        // The native default is still accepted.
        let native = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("euler_discrete".into()),
            ..Default::default()
        };
        assert!(g.validate(&native).is_ok());

        // A curated ε/DDPM sampler + curated scheduler validate OK.
        let curated = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("dpmpp_2m".into()),
            scheduler: Some("karras".into()),
            ..Default::default()
        };
        assert!(g.validate(&curated).is_ok());

        // An unadvertised sampler is still rejected (not silently downgraded).
        let bogus = GenerationRequest {
            prompt: "x".into(),
            sampler: Some("not_a_sampler".into()),
            ..Default::default()
        };
        assert!(g.validate(&bogus).is_err());
    }

    #[test]
    fn load_accepts_adapters_and_rejects_single_file() {
        // `/snap` exists on Ubuntu hosts, which would turn this lazy-loader test into an
        // unintended physical-receipt probe. Keep the fixture guaranteed missing while preserving
        // the exact Kolors repository/revision/tier path shape.
        let temp = tempfile::tempdir().unwrap();
        let root = temp
            .path()
            .join("models--SceneWorks--kolors-mlx")
            .join("snapshots")
            .join(memory_strategy::KOLORS_REVISION)
            .join("bf16");
        let lora =
            LoadSpec::new(WeightsSource::Dir(root.clone())).with_adapters(vec![AdapterSpec::new(
                "/lora.safetensors".into(),
                1.0,
                AdapterKind::Lora,
            )]);
        assert!(load(&lora).is_ok());

        // sc-10819: a packed q4/q8 tier is auto-detected from disk, so `quantize` is NO LONGER a load
        // reject (contrast the LoRA/control overlays above). Load is lazy (no file I/O), so a quant-only
        // spec at a nonexistent dir succeeds — the packed tier is resolved on the first `generate`.
        let quant = LoadSpec::new(WeightsSource::Dir(root)).with_quant(Quant::Q8);
        assert!(
            load(&quant).is_ok(),
            "a quant spec must not be rejected — packed tiers are wired (sc-10819)"
        );

        let single = LoadSpec::new(WeightsSource::File("/x.safetensors".into()));
        let err = load(&single).err().expect("err").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
