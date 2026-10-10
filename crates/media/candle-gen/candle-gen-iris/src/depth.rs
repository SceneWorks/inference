//! Iris-3B **monocular depth** on Candle (`iris3b/downstream/depth.py` @
//! [`crate::UPSTREAM_CODE_REVISION`], story sc-25682) — the twin of `mlx_gen_iris::depth`,
//! [`DEPTH_MODEL_ID`] (`iris_3b_depth`).
//!
//! `IrisDepth` is the backbone with its input projections widened by one channel, a zero channel
//! concatenated after RGB, one forward at model time `flow.num_train_timesteps` conditioned on the
//! export's shipped empty-prompt states, and a 1×1 convolution reducing the 3-channel output to one.
//! **No text encoder is loaded** (E4): the task's resource closure is its `depth/` export only
//! ([`TaskExport`]), and a text encoder, the generation checkpoint or the restoration export is a
//! typed refusal.
//!
//! The source preprocessing, the resize back and the result/metadata types are the backend-neutral
//! [`gen_core::iris::depth`]; this module is the tensor forward.
//!
//! [`gen_core::iris::depth`]: candle_gen::gen_core::iris::depth

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::iris::depth::{
    estimate_with, DepthMetadata, DepthOutput, DepthRequest, IrisDepthEstimator,
    DEPTH_EXTRA_INPUT_CHANNELS,
};
use candle_gen::gen_core::iris::downstream::{EmptyPrompt, TaskExport};
use candle_gen::gen_core::iris::IrisTask;
use candle_gen::gen_core::{self, LoadSpec, Progress};
use candle_gen::{CandleError as Error, Result};

use crate::dit::{IrisDiT, TextBatch};
use crate::model::compute_dtype;
use crate::nn::{expect_shape, safetensors_keys, Checkpoint, Linear};

/// Model id of the depth task.
pub const DEPTH_MODEL_ID: &str = gen_core::iris::DEPTH_MODEL_ID;

/// The export's key prefix of the widened backbone (`IrisDepth.pixel`).
const BACKBONE_PREFIX: &str = "pixel.";
/// `IrisDepth.depth_reducer` (`nn.Conv2d(in_channels, 1, 1)`).
const REDUCER_PREFIX: &str = "depth_reducer.";

/// A loaded Iris-3B depth model.
pub struct IrisDepth {
    export: TaskExport,
    dit: IrisDiT,
    /// The 1×1 conv as a `[1, C]` linear layer, in the compute dtype.
    reducer: Linear,
    /// `[1, T, L, D]` f32 empty-prompt states (on the model's device) and their `[T]` mask.
    embeddings: Tensor,
    mask: Vec<i32>,
}

fn refuse_unsupported_spec(spec: &LoadSpec) -> Result<()> {
    let refusals: [(&str, bool); 7] = [
        ("quantize", spec.quantize.is_some()),
        ("adapters", !spec.adapters.is_empty()),
        ("control", spec.control.is_some()),
        ("extra_controls", !spec.extra_controls.is_empty()),
        ("ip_adapter", spec.ip_adapter.is_some()),
        ("pid", spec.pid.is_some()),
        ("identity", spec.identity.is_some()),
    ];
    for (field, set) in refusals {
        if set {
            return Err(Error::Unsupported(format!(
                "{DEPTH_MODEL_ID}: LoadSpec::{field} is not supported by the Iris-3B depth task"
            )));
        }
    }
    Ok(())
}

/// Load the depth task on the build's device ([`candle_gen::default_device`]: `cuda:0` on a CUDA
/// build): `spec.weights` is the `depth/` export directory (`config.yaml` with `task.name: depth`,
/// `model.safetensors`, `empty_prompt.safetensors`); nothing else may be staged. `spec.precision`
/// selects upstream's CUDA bf16 autocast (default) or its FP32 CPU path.
pub fn load_depth(spec: &LoadSpec) -> Result<IrisDepth> {
    refuse_unsupported_spec(spec)?;
    let export = TaskExport::from_spec(spec, IrisTask::Depth, DEPTH_MODEL_ID)?;
    load_depth_export(export, compute_dtype(spec), &candle_gen::default_device()?)
}

/// [`load_depth`] from an already-resolved export at an explicit compute dtype and device.
pub fn load_depth_export(export: TaskExport, compute: DType, device: &Device) -> Result<IrisDepth> {
    let model = &export.config.model;
    let path = export.weights_path();
    if let Some(key) = safetensors_keys(&path)?
        .into_iter()
        .find(|k| !k.starts_with(BACKBONE_PREFIX) && !k.starts_with(REDUCER_PREFIX))
    {
        return Err(Error::Msg(format!(
            "iris: the depth export {} carries `{key}`, which is neither a `{BACKBONE_PREFIX}*` \
             backbone key nor the depth reducer",
            path.display()
        )));
    }
    let backbone = Checkpoint::open_prefixed(&path, device, BACKBONE_PREFIX)
        .map_err(|e| Error::Msg(format!("iris: loading {}: {e}", path.display())))?;
    let dit = IrisDiT::from_checkpoint_widened(
        &backbone,
        model,
        compute,
        device,
        model.in_channels + DEPTH_EXTRA_INPUT_CHANNELS,
    )?;
    drop(backbone);

    let reducer = Checkpoint::open_prefixed(&path, device, REDUCER_PREFIX)?;
    let weight = reducer.take("weight")?;
    let bias = reducer.take("bias")?;
    let leftover = reducer.unused_keys();
    if !leftover.is_empty() {
        return Err(Error::Msg(format!(
            "iris: the depth reducer carries unexpected `{REDUCER_PREFIX}{}`",
            leftover[0]
        )));
    }
    expect_shape(
        "depth_reducer.weight",
        &weight,
        &[1, model.in_channels, 1, 1],
    )?;
    expect_shape("depth_reducer.bias", &bias, &[1])?;
    let reducer = Linear::from_parts(
        weight.reshape((1, model.in_channels))?.to_dtype(compute)?,
        Some(bias.to_dtype(compute)?),
    );

    let prompt: EmptyPrompt = export.empty_prompt()?;
    let embeddings = Tensor::from_vec(prompt.embeddings, prompt.shape.to_vec(), device)?;
    Ok(IrisDepth {
        dit,
        reducer,
        embeddings,
        mask: prompt.mask,
        export,
    })
}

impl IrisDepth {
    pub fn export(&self) -> &TaskExport {
        &self.export
    }

    pub fn compute_dtype(&self) -> DType {
        self.dit.compute_dtype()
    }

    pub fn device(&self) -> &Device {
        self.dit.device()
    }

    /// `IrisDepth.forward`: `[B, 3, H, W]` RGB in `[-1, 1]` (H and W on the patch grid, any device)
    /// → `[B, 1, H, W]` relative log depth (f32, on the model's device).
    pub fn forward(&self, rgb: &Tensor) -> Result<Tensor> {
        let (b, _, h, w) = rgb.dims4()?;
        let device = self.device();
        let rgb = rgb.to_device(device)?.to_dtype(DType::F32)?;
        let zero = Tensor::zeros((b, DEPTH_EXTRA_INPUT_CHANNELS, h, w), DType::F32, device)?;
        let x = Tensor::cat(&[&rgb, &zero], 1)?;
        let t_model = self.export.config.flow.num_train_timesteps as f32;
        let t = Tensor::full(t_model, b, device)?;
        let (_, tl, l, d) = self.embeddings.dims4()?;
        let states = self.embeddings.broadcast_as((b, tl, l, d))?.contiguous()?;
        let mask = vec![self.mask.clone(); b];
        let out = self.dit.forward(
            &x,
            &t,
            &TextBatch {
                states: &states,
                mask: &mask,
            },
        )?;
        // depth_reducer: a 1×1 conv over the channels — under upstream's autocast a bf16 conv, so
        // it runs in the compute dtype like every other projection.
        let c = out.dim(1)?;
        let pixels = out
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((b * h * w, c))?;
        let reduced = self.reducer.forward(&pixels)?;
        Ok(reduced
            .reshape((b, h, w, 1))?
            .permute((0, 3, 1, 2))?
            .contiguous()?
            .to_dtype(DType::F32)?)
    }

    /// One `[3, h, w]` host input → the `h × w` host prediction.
    fn predict(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>> {
        let (h, w) = (height as usize, width as usize);
        let rgb = Tensor::from_slice(input, (1, 3, h, w), self.device())?;
        let depth = self.forward(&rgb)?;
        Ok(depth
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?)
    }
}

impl IrisDepthEstimator for IrisDepth {
    fn backend(&self) -> &'static str {
        "candle"
    }

    fn estimate(
        &self,
        req: &DepthRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<DepthOutput> {
        let compute = match self.compute_dtype() {
            DType::F32 => "float32",
            _ => "bfloat16",
        };
        estimate_with(
            req,
            self.export.config.model.patch_size,
            |plan| {
                DepthMetadata::new(
                    "candle",
                    self.export.config_sha256.clone(),
                    compute,
                    req.resolution,
                    plan,
                )
            },
            on_progress,
            |input, plan| {
                self.predict(input, plan.model_width, plan.model_height)
                    .map_err(Into::into)
            },
        )
    }
}

/// Load the depth task as a boxed backend-neutral estimator (see [`load_depth`]).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn IrisDepthEstimator>> {
    Ok(Box::new(load_depth(spec)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::WeightsSource;

    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/iris-depth-not-on-disk".into()));
        spec.quantize = Some(gen_core::Quant::Q4);
        match load_depth(&spec) {
            Err(Error::Unsupported(m)) => assert!(m.contains("quantize"), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
    }
}
