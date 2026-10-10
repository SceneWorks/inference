//! Iris-3B **monocular depth** on MLX (`iris3b/downstream/depth.py` @ [`crate::UPSTREAM_CODE_REVISION`],
//! story sc-25682) — [`DEPTH_MODEL_ID`] (`iris_3b_depth`).
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

use gen_core::iris::depth::{
    estimate_with, DepthMetadata, DepthOutput, DepthRequest, IrisDepthEstimator,
    DEPTH_EXTRA_INPUT_CHANNELS,
};
use gen_core::iris::downstream::{EmptyPrompt, TaskExport};
use gen_core::iris::IrisTask;
use mlx_gen::weights::Weights;
use mlx_gen::{gen_core, Error, LoadSpec, Progress, Result};
use mlx_rs::ops::{addmm, concatenate_axis, zeros};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::dit::{IrisDiT, TextBatch};
use crate::model::compute_dtype;

/// Model id of the depth task.
pub const DEPTH_MODEL_ID: &str = gen_core::iris::DEPTH_MODEL_ID;

/// The export's key prefix of the widened backbone (`IrisDepth.pixel`).
const BACKBONE_PREFIX: &str = "pixel.";
/// `IrisDepth.depth_reducer` (`nn.Conv2d(in_channels, 1, 1)`).
const REDUCER_WEIGHT: &str = "depth_reducer.weight";
const REDUCER_BIAS: &str = "depth_reducer.bias";

/// A loaded Iris-3B depth model.
pub struct IrisDepth {
    export: TaskExport,
    dit: IrisDiT,
    /// `[1, C]` (the 1×1 conv kernel) and `[1]`, in the compute dtype.
    reducer_weight: Array,
    reducer_bias: Array,
    /// `[1, T, L, D]` f32 empty-prompt states and their `[T]` mask.
    embeddings: Array,
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

/// Load the depth task: `spec.weights` is the `depth/` export directory (`config.yaml` with
/// `task.name: depth`, `model.safetensors`, `empty_prompt.safetensors`); nothing else may be staged.
/// `spec.precision` selects upstream's CUDA bf16 autocast (default) or its FP32 CPU path. Every
/// parameter is evaluated before this returns.
pub fn load_depth(spec: &LoadSpec) -> Result<IrisDepth> {
    refuse_unsupported_spec(spec)?;
    let export = TaskExport::from_spec(spec, IrisTask::Depth, DEPTH_MODEL_ID)?;
    load_depth_export(export, compute_dtype(spec))
}

/// [`load_depth`] from an already-resolved export at an explicit compute dtype.
pub fn load_depth_export(export: TaskExport, compute: Dtype) -> Result<IrisDepth> {
    let model = &export.config.model;
    let path = export.weights_path();
    let mut source = Weights::from_file(&path)
        .map_err(|e| Error::Msg(format!("iris: loading {}: {e}", path.display())))?;
    let mut keys: Vec<String> = source.keys().map(str::to_owned).collect();
    keys.sort();
    // Split the export into the backbone (`pixel.*`, prefix stripped) and the reducer, casting each
    // backbone matrix to the compute dtype one tensor at a time (the generation loader's policy).
    let mut backbone = Weights::empty();
    let (mut reducer_weight, mut reducer_bias) = (None, None);
    for key in keys {
        let Some(array) = source.remove(&key) else {
            continue;
        };
        // Read the file bytes on MLX's CPU stream first: a GPU cast over an unread `Load` makes the
        // Metal command buffer wait on the disk, which on a cold page cache trips the GPU watchdog
        // (sc-24245).
        mlx_rs::transforms::eval_pending_loads([&array])?;
        if let Some(name) = key.strip_prefix(BACKBONE_PREFIX) {
            let array = if array.ndim() >= 2 && name != "y_pos_embedding" {
                array.as_dtype(compute)?
            } else {
                array
            };
            eval([&array])?;
            backbone.insert(name, array);
        } else if key == REDUCER_WEIGHT {
            reducer_weight = Some(array);
        } else if key == REDUCER_BIAS {
            reducer_bias = Some(array);
        } else {
            return Err(Error::Msg(format!(
                "iris: the depth export {} carries `{key}`, which is neither a `{BACKBONE_PREFIX}*` \
                 backbone key nor the depth reducer",
                path.display()
            )));
        }
    }
    let missing = |key: &str| Error::Msg(format!("iris: the depth export is missing `{key}`"));
    let reducer_weight = reducer_weight.ok_or_else(|| missing(REDUCER_WEIGHT))?;
    let reducer_bias = reducer_bias.ok_or_else(|| missing(REDUCER_BIAS))?;
    crate::nn::expect_shape(
        REDUCER_WEIGHT,
        &reducer_weight,
        &[1, model.in_channels as i32, 1, 1],
    )?;
    crate::nn::expect_shape(REDUCER_BIAS, &reducer_bias, &[1])?;
    let reducer_weight = reducer_weight
        .reshape(&[1, model.in_channels as i32])?
        .as_dtype(compute)?;
    let reducer_bias = reducer_bias.as_dtype(compute)?;

    let dit = IrisDiT::from_weights_widened(
        &backbone,
        model,
        compute,
        model.in_channels + DEPTH_EXTRA_INPUT_CHANNELS,
    )?;
    drop(backbone);

    let prompt: EmptyPrompt = export.empty_prompt()?;
    let shape: Vec<i32> = prompt.shape.iter().map(|&d| d as i32).collect();
    let embeddings = Array::from_slice(&prompt.embeddings, &shape);

    let mut arrays = dit.arrays();
    arrays.extend([&reducer_weight, &reducer_bias, &embeddings]);
    for group in arrays.chunks(64) {
        eval(group.iter().copied())?;
    }
    Ok(IrisDepth {
        dit,
        reducer_weight,
        reducer_bias,
        embeddings,
        mask: prompt.mask,
        export,
    })
}

impl IrisDepth {
    pub fn export(&self) -> &TaskExport {
        &self.export
    }

    pub fn compute_dtype(&self) -> Dtype {
        self.dit.compute_dtype()
    }

    /// `IrisDepth.forward`: `[B, 3, H, W]` RGB in `[-1, 1]` (f32, H and W on the patch grid) →
    /// `[B, 1, H, W]` relative log depth (f32).
    pub fn forward(&self, rgb: &Array) -> Result<Array> {
        let sh = rgb.shape();
        let (b, h, w) = (sh[0], sh[2], sh[3]);
        let x = concatenate_axis(
            &[
                &rgb.as_dtype(Dtype::Float32)?,
                &zeros::<f32>(&[b, DEPTH_EXTRA_INPUT_CHANNELS as i32, h, w])?,
            ],
            1,
        )?;
        let t_model = self.export.config.flow.num_train_timesteps as f32;
        let t = Array::from_slice(&vec![t_model; b as usize], &[b]);
        let e = self.embeddings.shape();
        let states = mlx_rs::ops::broadcast_to(&self.embeddings, &[b, e[1], e[2], e[3]])?;
        let mask = vec![self.mask.clone(); b as usize];
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
        let c = out.shape()[1];
        let pixels = out
            .transpose_axes(&[0, 2, 3, 1])?
            .reshape(&[-1, c])?
            .as_dtype(self.compute_dtype())?;
        let reduced = addmm(
            &self.reducer_bias,
            &pixels,
            self.reducer_weight.t(),
            1.0,
            1.0,
        )?;
        Ok(reduced.reshape(&[b, 1, h, w])?.as_dtype(Dtype::Float32)?)
    }

    fn estimate_impl(
        &self,
        req: &DepthRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<DepthOutput> {
        let compute = match self.compute_dtype() {
            Dtype::Float32 => "float32",
            _ => "bfloat16",
        };
        estimate_with(
            req,
            self.export.config.model.patch_size,
            |plan| {
                DepthMetadata::new(
                    "mlx",
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

    /// One `[3, h, w]` host input → the `h × w` host prediction.
    fn predict(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>> {
        let (h, w) = (height as i32, width as i32);
        let rgb = Array::from_slice(input, &[1, 3, h, w]);
        let depth = self.forward(&rgb)?;
        eval([&depth])?;
        let values = depth.reshape(&[h * w])?.as_slice::<f32>().to_vec();
        drop(depth);
        mlx_rs::memory::clear_cache();
        Ok(values)
    }
}

impl IrisDepthEstimator for IrisDepth {
    fn backend(&self) -> &'static str {
        "mlx"
    }

    fn estimate(
        &self,
        req: &DepthRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<DepthOutput> {
        self.estimate_impl(req, on_progress)
    }
}

/// Load the depth task as a boxed backend-neutral estimator (see [`load_depth`]).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn IrisDepthEstimator>> {
    Ok(Box::new(load_depth(spec)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::WeightsSource;

    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/iris-depth-not-on-disk".into()));
        spec.quantize = Some(mlx_gen::Quant::Q4);
        match load_depth(&spec) {
            Err(Error::Unsupported(m)) => assert!(m.contains("quantize"), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
    }
}
