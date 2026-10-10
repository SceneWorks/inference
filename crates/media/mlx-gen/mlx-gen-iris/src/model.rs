//! `Iris3b` — the Iris-3B text-to-image [`Generator`]: descriptor, load, request validation and the
//! staged generate (text encoder → backbone), registered as [`MODEL_ID`].

use std::path::Path;

use gen_core::iris::{
    reject_unhonored_generation_controls, GenerationParams, GenerationResources, IrisConfig,
    FAMILY, GENERATION_MODEL_ID,
};
use mlx_gen::weights::Weights;
use mlx_gen::{
    default_seed, gen_core, Capabilities, Error, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, Modality, ModelDescriptor, Precision, Progress, Residency, Result, SizeFloor,
};
use mlx_rs::transforms::eval;
use mlx_rs::Dtype;

use crate::dit::IrisDiT;
use crate::pipeline::{denoise, encode, noise, to_image, Conditioning};
use crate::text_encoder::IrisTextEncoder;

/// Registry id (the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = GENERATION_MODEL_ID;
/// Output sides must be multiples of the patch size (load refuses a patch that does not divide it).
pub const SIZE_MULTIPLE: u32 = gen_core::iris::SIZE_MULTIPLE;
/// Smallest side: one patch.
pub const MIN_SIZE: u32 = 16;
/// Largest side served (the release renders ~1 MP; 2048² is the pixel-space attention ceiling
/// this route admits).
pub const MAX_SIZE: u32 = 2048;
/// Images per request.
pub const MAX_COUNT: u32 = 8;

/// Identity + capabilities, constructible without weights.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        // Pixel space: there is no latent and no VAE.
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // CFG against the negative prompt run through the same template (default "").
            supports_negative_prompt: true,
            // `guidance` is upstream's `cfg_scale` (default 3.0; 1.0 = CFG off).
            supports_guidance: true,
            supports_true_cfg: false,
            // The FlowDPM-Solver++ (order 2, lower-order final) is the only integrator; no
            // sampler/scheduler names are advertised, so any requested one is refused.
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            max_count: MAX_COUNT,
            mac_only: true,
            supported_quants: &[],
            supports_sequential_offload: true,
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SIZE_MULTIPLE,
            },
            ..Default::default()
        },
    }
}

/// The backbone half of the generation task.
pub struct Heavy {
    pub dit: IrisDiT,
}

/// A loaded Iris-3B generator.
pub struct Iris3b {
    descriptor: ModelDescriptor,
    config: IrisConfig,
    residency: Residency<IrisTextEncoder, Heavy>,
}

/// The compute dtype a load spec selects: the release's bf16 autocast by default, upstream's FP32
/// CPU path on an explicit `Precision::Fp32`.
pub fn compute_dtype(spec: &LoadSpec) -> Dtype {
    match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    }
}

/// Load the backbone from its directory (`config.yaml` already parsed) and evaluate every
/// parameter before returning, so no forward ever reads a weight lazily.
pub fn load_backbone(dir: &Path, config: &IrisConfig, compute: Dtype) -> Result<IrisDiT> {
    let path = dir.join(gen_core::iris::BACKBONE_WEIGHTS_FILE);
    let mut weights = Weights::from_file(&path)
        .map_err(|e| Error::Msg(format!("iris: loading {}: {e}", path.display())))?;
    // The release ships FP32. Cast every matrix to the compute dtype ONE TENSOR AT A TIME and
    // drop its source handle, so the load peak is the compute-dtype model plus a single FP32
    // tensor rather than both full copies. Vectors (norm gains, biases) and the text position
    // table stay as stored; the modules decide their own dtype.
    let mut keys: Vec<String> = weights.keys().map(str::to_owned).collect();
    keys.sort();
    for key in keys {
        let Some(array) = weights.remove(&key) else {
            continue;
        };
        let array = if array.ndim() >= 2 && key != "y_pos_embedding" {
            array.as_dtype(compute)?
        } else {
            array
        };
        eval([&array])?;
        weights.insert(key, array);
    }
    let dit = IrisDiT::from_weights(&weights, &config.model, compute)?;
    // Every derived parameter is evaluated before any forward can read it.
    let arrays = dit.arrays();
    for group in arrays.chunks(64) {
        eval(group.iter().copied())?;
    }
    drop(weights);
    Ok(dit)
}

fn refuse_unsupported_spec(spec: &LoadSpec) -> Result<()> {
    let refusals: [(&str, bool); 8] = [
        ("quantize", spec.quantize.is_some()),
        ("adapters", !spec.adapters.is_empty()),
        ("text_encoder", spec.text_encoder.is_some()),
        ("control", spec.control.is_some()),
        ("extra_controls", !spec.extra_controls.is_empty()),
        ("ip_adapter", spec.ip_adapter.is_some()),
        ("pid", spec.pid.is_some()),
        ("identity", spec.identity.is_some()),
    ];
    for (field, set) in refusals {
        if set {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: LoadSpec::{field} is not supported by the Iris-3B generation route \
                 (the text encoder is the '{}' component)",
                gen_core::iris::TEXT_ENCODER_COMPONENT
            )));
        }
    }
    Ok(())
}

/// Construct the generator from its task resources: `spec.weights` = the backbone directory
/// (`config.yaml` + `model.safetensors`), `spec.components["text_encoder"]` = the
/// `Qwen/Qwen3-VL-4B-Instruct` snapshot. `Resident` (default) loads both now; `Sequential` loads
/// the encoder, encodes, drops it, then loads the backbone, per request.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    refuse_unsupported_spec(spec)?;
    let resources = GenerationResources::from_spec(spec, MODEL_ID)?;
    let config = IrisConfig::from_dir(&resources.backbone_dir)?;
    config.validate_supported()?;
    let compute = compute_dtype(spec);
    let te_dir = resources.text_encoder_dir.clone();
    let te_cfg = config.text_encoder.clone();
    let bb_dir = resources.backbone_dir.clone();
    let bb_cfg = config.clone();
    let residency = Residency::from_policy(
        spec.offload_policy,
        move || IrisTextEncoder::load(&te_dir, &te_cfg),
        move |_use_pid| {
            Ok(Heavy {
                dit: load_backbone(&bb_dir, &bb_cfg, compute)?,
            })
        },
    )?;
    Ok(Box::new(Iris3b {
        descriptor: descriptor(),
        config,
        residency,
    }))
}

impl Iris3b {
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate_impl(req)?;
        let params = GenerationParams::resolve(req, default_seed());
        let total = (params.steps as u32) * req.count;
        let flow = self.config.flow.clone();
        self.residency.run(
            &req.cancel,
            false,
            on_progress,
            |te: &IrisTextEncoder| -> Result<Conditioning> {
                let conditioning = encode(te, &req.prompt, &params)?;
                Ok(conditioning)
            },
            |_| Ok(()),
            |heavy: &Heavy, conditioning: Conditioning, on_progress| {
                let channels = heavy.dit.config().in_channels;
                let mut samples = Vec::with_capacity(req.count as usize);
                for i in 0..req.count {
                    let seed = params.seed.wrapping_add(i as u64);
                    let z = noise(seed, channels, params.width, params.height)?;
                    let done = i * params.steps as u32;
                    let x = denoise(
                        &heavy.dit,
                        &flow,
                        &conditioning,
                        &z,
                        &params,
                        &req.cancel,
                        |step| {
                            on_progress(Progress::Step {
                                current: done + step as u32,
                                total,
                            })
                        },
                    )?;
                    samples.push(x);
                }
                // Pixel space: the "decode" is only the [−1, 1] → RGB8 conversion.
                on_progress(Progress::Decoding);
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let images = samples.iter().map(to_image).collect::<Result<Vec<_>>>()?;
                mlx_rs::memory::clear_cache();
                Ok(GenerationOutput::Images(images))
            },
        )
    }

    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        reject_unhonored_generation_controls(MODEL_ID, req)?;
        Ok(())
    }
}

impl Generator for Iris3b {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::WeightsSource;

    #[test]
    fn descriptor_is_pixel_space_and_honest() {
        let d = descriptor();
        assert_eq!(d.id, "iris_3b");
        assert_eq!(d.family, "iris");
        assert!(d.denoiser_output_latent_space.is_none());
        assert!(d.capabilities.samplers.is_empty());
        assert!(!d.capabilities.supports_true_cfg);
    }

    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/iris-not-on-disk".into()));
        spec.quantize = Some(mlx_gen::Quant::Q4);
        match load(&spec) {
            Err(Error::Unsupported(m)) => assert!(m.contains("quantize"), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
    }

    #[test]
    fn a_missing_text_encoder_is_a_load_error() {
        let dir = tempfile::tempdir().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        let err = load(&spec).err().expect("missing text encoder must fail");
        assert!(err.to_string().contains("text_encoder"), "{err}");
    }
}
