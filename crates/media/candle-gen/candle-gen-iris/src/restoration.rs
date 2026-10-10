//! `iris_3b_restore` — the Iris-3B restoration / upscaling [`Transform`] on Candle (story
//! sc-25683), the CUDA sibling of `mlx-gen-iris`'s restorer with the same surface field for field.
//!
//! The restoration task is the `upscaler/` export: its own fine-tuned backbone (`config.yaml` with a
//! `task: restoration` section + `model.safetensors`) and the shipped empty-prompt states
//! (`empty_prompt.safetensors`). It runs **without the text encoder** (E4).
//!
//! The pixel path around the forward (input budget, bicubic upsample, small-image enlargement,
//! padding, 50 %-overlap tiling with upstream's Gaussian fusion, wavelet colour fix, quantization)
//! is the backend-neutral [`gen_core::iris::restoration`] driver shared with the MLX twin; this crate
//! supplies only each tile's one-step velocity.
//!
//! [`gen_core::iris::restoration`]: candle_gen::gen_core::iris::restoration

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::iris::restoration::{
    self as contract, plan_request, restore, RestorationPlan, RestorationResources,
    RestorationSettings, TileGeometry, DEFAULT_SCALE, INPUT_BUDGET,
};
use candle_gen::gen_core::iris::{IrisConfig, FAMILY};
use candle_gen::gen_core::{
    self, Image, LoadSpec, Progress, Transform, TransformCapabilities, TransformDescriptor,
    TransformRegistration, TransformRequest,
};
use candle_gen::{CandleError as Error, Result};

use crate::dit::{IrisDiT, TextBatch};
use crate::model::{compute_dtype, load_backbone};

/// Registry id of the restoration transform.
pub const MODEL_ID: &str = contract::MODEL_ID;

/// Identity + capabilities, constructible without weights (the MLX twin's surface, `mac_only`
/// false).
pub fn descriptor() -> TransformDescriptor {
    TransformDescriptor {
        id: MODEL_ID,
        family: FAMILY,
        backend: "candle",
        capabilities: TransformCapabilities {
            scale: true,
            min_edge: false,
            resolution: false,
            // Upstream has no ceiling; the planner refuses only a zero-pixel or overflowing output.
            max_scale: f32::MAX,
            is_diffusion: false,
            supports_strength: false,
            mac_only: false,
            default_scale: DEFAULT_SCALE as f32,
            input_budget: Some(INPUT_BUDGET),
            supports_color_fix: true,
        },
    }
}

/// A loaded Iris-3B restorer.
pub struct IrisRestorer {
    descriptor: TransformDescriptor,
    config: IrisConfig,
    settings: RestorationSettings,
    dit: IrisDiT,
    /// `[1, T, L, D]` f32 empty-prompt states on the backbone's device.
    states: Tensor,
    mask: Vec<Vec<i32>>,
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
                "{MODEL_ID}: LoadSpec::{field} is not supported by the Iris-3B restoration route"
            )));
        }
    }
    Ok(())
}

impl IrisRestorer {
    /// Load the restoration export (`spec.weights` = the `upscaler/` directory) onto the build's
    /// device ([`candle_gen::default_device`]). `Precision::Bf16` (default) mirrors the release's
    /// CUDA autocast; `Precision::Fp32` is upstream's CPU path. `OffloadPolicy` is advisory and this
    /// route has a single heavy component, so it is always resident.
    pub fn load(spec: &LoadSpec) -> Result<Self> {
        Self::load_on(spec, &candle_gen::default_device()?)
    }

    /// [`load`](Self::load) onto an explicit device.
    pub fn load_on(spec: &LoadSpec, device: &Device) -> Result<Self> {
        refuse_unsupported_spec(spec)?;
        let resources = RestorationResources::from_spec(spec, MODEL_ID)?;
        let prompt = resources.empty_prompt()?;
        let dit = load_backbone(
            &resources.dir,
            &resources.config,
            compute_dtype(spec),
            device,
        )?;
        let states = Tensor::from_vec(prompt.embeddings, prompt.shape.to_vec(), device)?;
        Ok(Self {
            descriptor: descriptor(),
            config: resources.config,
            settings: resources.settings,
            dit,
            states,
            mask: vec![prompt.mask],
        })
    }

    /// The export's tile geometry (the release: 1024-px tiles, 16-px patches).
    pub fn geometry(&self) -> TileGeometry {
        self.settings.geometry(&self.config)
    }

    /// Plan a request against the loaded export.
    pub fn plan(&self, req: &TransformRequest) -> Result<RestorationPlan> {
        Ok(plan_request(req, self.geometry(), MODEL_ID)?)
    }

    /// One tile's velocity: `[3, h, w]` CHW f32 in `[-1, 1]` → `v` (same layout, f32), the backbone
    /// run once at `sigma · num_train_timesteps` on the empty-prompt states.
    pub fn velocity(&self, tile: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
        let device = self.dit.device();
        let x = Tensor::from_slice(tile, (1, 3, height, width), device)?;
        let t = Tensor::from_slice(&[self.settings.model_time(&self.config)], 1, device)?;
        let text = TextBatch {
            states: &self.states,
            mask: &self.mask,
        };
        let v = self.dit.forward(&x, &t, &text)?;
        Ok(v.to_dtype(DType::F32)?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?)
    }

    fn apply_impl(
        &self,
        req: &TransformRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let plan = self.plan(req)?;
        Ok(restore(
            &req.image,
            &plan,
            self.settings.sigma,
            &req.cancel,
            on_progress,
            &mut |tile, h, w| self.velocity(tile, h, w).map_err(Into::into),
        )?)
    }
}

impl Transform for IrisRestorer {
    fn descriptor(&self) -> &TransformDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TransformRequest) -> gen_core::Result<()> {
        plan_request(req, self.geometry(), MODEL_ID).map(|_| ())
    }

    fn apply(
        &self,
        req: &TransformRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<Image> {
        self.apply_impl(req, on_progress).map_err(Into::into)
    }
}

/// Registry loader.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Transform>> {
    Ok(Box::new(IrisRestorer::load(spec)?))
}

pub(crate) const REGISTRATION: TransformRegistration = TransformRegistration { descriptor, load };

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::WeightsSource;

    #[test]
    fn descriptor_declares_the_release_surface() {
        let d = descriptor();
        assert_eq!(d.id, "iris_3b_restore");
        assert_eq!(d.backend, "candle");
        let c = &d.capabilities;
        assert!(c.scale && !c.min_edge && !c.resolution && !c.mac_only);
        assert_eq!(c.default_scale, 4.0);
        assert!(c.supports_color_fix && !c.is_diffusion && !c.supports_strength);
    }

    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/iris-not-on-disk".into()));
        spec.quantize = Some(gen_core::Quant::Q4);
        match IrisRestorer::load_on(&spec, &Device::Cpu) {
            Err(Error::Unsupported(m)) => assert!(m.contains("quantize"), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
    }
}
