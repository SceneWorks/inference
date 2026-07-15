//! # candle-gen-kolors
//!
//! The **Kolors** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling
//! of `mlx-gen-kolors`. It implements the backend-neutral [`gen_core::Generator`] contract and
//! exposes the candle Kolors generator through its explicit family catalog.
//!
//! **txt2img:** Kolors is a bilingual (Chinese/English) SDXL-family T2I model — the SDXL UNet + SDXL
//! VAE with a **ChatGLM3-6B** text encoder in place of dual CLIP. `pipeline` runs it through the
//! contract: ChatGLM3 encode (penultimate hidden state → cross-attention context, last-token
//! last-layer state → pooled add-embedding) → the Kolors UNet (real CFG over the leading-Euler
//! 1100-step schedule) → the SDXL VAE, emitting `Progress`, honoring `req.cancel`, with
//! **deterministic CPU-seeded noise** (sc-3673) so output is launch-portable per seed.
//!
//! The descriptor advertises the wired surface — txt2img (negative prompt + CFG guidance) and packed
//! **Q4/Q8** MLX-tier inference (sc-10819, epic 9083) — but NOT LoRA/LoKr, ControlNet-pose, or
//! IP-Adapter (all wired in the mlx provider), so the worker routes the rest to the Python fallback
//! rather than the candle backend silently dropping a control (the false-capability trap, exactly as
//! the SDXL / FLUX / Z-Image / Chroma slices did). `backend` is `"candle"` and `mac_only` is `false`.

mod chatglm3;
// Shared Kolors pipeline scaffolding (sc-9001 / F-021): the time_ids / initial-noise / decode /
// CFG-batched-encode / curated-σ-prior blocks that were copy-pasted across the three entry points.
mod common;
mod config;
mod pipeline;
mod sampler;
mod tokenizer;
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
    self, GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor, PidWeights,
    Progress, WeightsSource,
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
    components: Mutex<Option<Components>>,
}

impl KolorsGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        // `?` bridges the candle-side `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

impl Generator for KolorsGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // The shared capability floor (count/size range, negative_prompt + guidance; since the
        // descriptor advertises NO conditioning, any conditioning entry is rejected here).
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
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
        let pipe = Pipeline::load(&self.root, &self.device, self.pid_spec.clone());
        let components = self.components(&pipe)?;
        let images = pipe.render(req, &components, on_progress)?;
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
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle kolors does not support LoRA/LoKr yet — refusing to silently drop the adapters"
                .into(),
        ));
    }
    // Packed q4/q8 MLX tiers are wired end-to-end (sc-10819, epic 9083): the tier is packed-detected
    // from disk (`unet/` & `text_encoder/` `config.json` `quantization` blocks; see
    // `pipeline::load_components`), so the `LoadSpec::quantize` overlay is an advisory no-op on an
    // already-packed tier — exactly as SDXL/boogu/flux2-dev treat it. No reject here.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle kolors does not support control / IP-adapter overlays yet (txt2img only)"
                .into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(KolorsGenerator {
        descriptor: descriptor(),
        root,
        device,
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike adapters/quant/control above,
        // it is not rejected — `None` simply keeps the byte-exact native-VAE path.
        pid_spec: spec.pid.clone(),
        components: Mutex::new(None),
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
    registry.register_generator(REGISTRATION)
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
    fn validate_accepts_txt2img_and_rejects_unsupported() {
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
                conditioning: vec![Conditioning::Reference {
                    image: Image::default(),
                    strength: None,
                }],
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
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
    fn load_rejects_unwired_surfaces_and_single_file() {
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(matches!(
            load(&lora).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));

        // sc-10819: a packed q4/q8 tier is auto-detected from disk, so `quantize` is NO LONGER a load
        // reject (contrast the LoRA/control overlays above). Load is lazy (no file I/O), so a quant-only
        // spec at a nonexistent dir succeeds — the packed tier is resolved on the first `generate`.
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(
            load(&quant).is_ok(),
            "a quant spec must not be rejected — packed tiers are wired (sc-10819)"
        );

        let single = LoadSpec::new(WeightsSource::File("/x.safetensors".into()));
        let err = load(&single).err().expect("err").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
