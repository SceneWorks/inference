//! `iris_3b_restore` — the Iris-3B restoration / upscaling [`Transform`] on MLX (story sc-25683).
//!
//! The restoration task is the `upscaler/` export: its own fine-tuned backbone (`config.yaml` with a
//! `task: restoration` section + `model.safetensors`) and the shipped empty-prompt states
//! (`empty_prompt.safetensors`). It runs **without the text encoder** (E4): the conditioning is the
//! empty-prompt stack, read once at load.
//!
//! Everything around the forward — the input budget, the bicubic upsample, small-image enlargement,
//! padding, 50 %-overlap tiling with upstream's Gaussian fusion, the wavelet colour fix and the 8-bit
//! quantization — is the backend-neutral [`gen_core::iris::restoration`] driver, shared with the
//! Candle twin. This crate supplies only the one-step velocity of each tile.
//!
//! [`gen_core::iris::restoration`]: mlx_gen::gen_core::iris::restoration

use gen_core::iris::restoration::{
    self as contract, plan_request, restore, RestorationPlan, RestorationResources,
    RestorationSettings, TileGeometry, DEFAULT_SCALE, INPUT_BUDGET,
};
use gen_core::iris::{IrisConfig, FAMILY};
use mlx_gen::{
    gen_core, Error, Image, LoadSpec, Progress, Result, Transform, TransformCapabilities,
    TransformDescriptor, TransformRegistration, TransformRequest,
};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::dit::{IrisDiT, TextBatch};
use crate::model::{compute_dtype, load_backbone};

/// Registry id of the restoration transform.
pub const MODEL_ID: &str = contract::MODEL_ID;

/// Identity + capabilities, constructible without weights.
pub fn descriptor() -> TransformDescriptor {
    TransformDescriptor {
        id: MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        capabilities: capabilities(true),
    }
}

/// The restorer's request surface (shared with the Candle twin but for `mac_only`).
pub(crate) fn capabilities(mac_only: bool) -> TransformCapabilities {
    TransformCapabilities {
        // `TargetSize::Scale` — any positive finite factor; the output is `floor(input · scale)`.
        scale: true,
        min_edge: false,
        resolution: false,
        // Upstream has no ceiling: compute grows with the output area (the planner refuses only a
        // zero-pixel or overflowing output).
        max_scale: f32::MAX,
        // One deterministic forward per tile: no seed, no strength.
        is_diffusion: false,
        supports_strength: false,
        mac_only,
        default_scale: DEFAULT_SCALE as f32,
        input_budget: Some(INPUT_BUDGET),
        supports_color_fix: true,
    }
}

/// A loaded Iris-3B restorer.
pub struct IrisRestorer {
    descriptor: TransformDescriptor,
    config: IrisConfig,
    settings: RestorationSettings,
    dit: IrisDiT,
    /// `[1, T, L, D]` f32 empty-prompt states.
    states: Array,
    mask: Vec<Vec<i32>>,
}

/// Refuse every load knob the restoration route does not implement, by name.
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
    /// Load the restoration export (`spec.weights` = the `upscaler/` directory). The release
    /// computes under bf16 autocast (`Precision::Bf16`, default); `Precision::Fp32` is upstream's
    /// CPU path. `OffloadPolicy` is advisory and this route has a single heavy component, so it is
    /// always resident.
    pub fn load(spec: &LoadSpec) -> Result<Self> {
        refuse_unsupported_spec(spec)?;
        let resources = RestorationResources::from_spec(spec, MODEL_ID)?;
        let prompt = resources.empty_prompt()?;
        let dit = load_backbone(&resources.dir, &resources.config, compute_dtype(spec))?;
        let shape: Vec<i32> = prompt.shape.iter().map(|&d| d as i32).collect();
        let states = Array::from_slice(&prompt.embeddings, &shape);
        eval([&states])?;
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
        let x = Array::from_slice(tile, &[1, 3, height as i32, width as i32]);
        let t = Array::from_slice(&[self.settings.model_time(&self.config)], &[1]);
        let text = TextBatch {
            states: &self.states,
            mask: &self.mask,
        };
        let v = self.dit.forward(&x, &t, &text)?.as_dtype(Dtype::Float32)?;
        eval([&v])?;
        let n = v.shape().iter().product::<i32>();
        Ok(v.reshape(&[n])?.as_slice::<f32>().to_vec())
    }

    fn apply_impl(
        &self,
        req: &TransformRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let plan = self.plan(req)?;
        let image = restore(
            &req.image,
            &plan,
            self.settings.sigma,
            &req.cancel,
            on_progress,
            &mut |tile, h, w| self.velocity(tile, h, w).map_err(Into::into),
        );
        mlx_rs::memory::clear_cache();
        Ok(image?)
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
    use mlx_gen::WeightsSource;

    #[test]
    fn descriptor_declares_the_release_surface() {
        let d = descriptor();
        assert_eq!(d.id, "iris_3b_restore");
        assert_eq!(d.family, "iris");
        let c = &d.capabilities;
        assert!(c.scale && !c.min_edge && !c.resolution);
        assert_eq!(c.default_scale, 4.0);
        assert!(c.supports_color_fix && !c.is_diffusion && !c.supports_strength);
        assert_eq!(c.input_budget, Some(INPUT_BUDGET));
    }

    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/iris-not-on-disk".into()));
        spec.quantize = Some(mlx_gen::Quant::Q4);
        match IrisRestorer::load(&spec) {
            Err(Error::Unsupported(m)) => assert!(m.contains("quantize"), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
    }
}
