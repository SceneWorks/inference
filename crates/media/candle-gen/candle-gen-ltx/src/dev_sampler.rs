//! The non-distilled LTX-2.5 stage-one execution contract.
//!
//! A split LTX-2.5 transformer carries an explicit safetensors `variant` identity.  The dev
//! checkpoint is not compatible with the eight-transition distilled trajectory: it runs thirty
//! guided transitions and evaluates the joint DiT four times at each transition.

use candle_gen::candle_core::{DType, Error, Result, Tensor};
use candle_gen::gen_core::ltx_checkpoint::{LtxBundle, LtxComponent};
use serde::{Deserialize, Serialize};

use crate::config::STAGE1_SIGMAS;
use crate::params::{GuiderParams, LTX_2_5_PARAMS};

/// Transformer identity declared on the split transformer safetensors file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TransformerVariant {
    Distilled,
    Dev,
}

impl TransformerVariant {
    pub const METADATA_KEY: &'static str = "variant";

    /// Parse the exact checkpoint identity.  There is intentionally no fallback: treating a dev
    /// transformer as distilled makes it render with the wrong sampler while still producing a
    /// plausible looking clip.
    pub fn from_metadata(value: Option<&str>) -> Result<Self> {
        match value {
            Some("distilled") => Ok(Self::Distilled),
            Some("dev") => Ok(Self::Dev),
            Some(other) => Err(Error::Msg(format!(
                "ltx_2_5: unsupported transformer variant {other:?}; expected 'distilled' or 'dev'"
            ))),
            None => Err(Error::Msg(
                "ltx_2_5: split transformer metadata is missing required 'variant'; refusing to default a checkpoint identity".into(),
            )),
        }
    }

    /// Read the identity from the one authoritative component: the split transformer itself.
    pub fn from_bundle(bundle: &LtxBundle) -> Result<Self> {
        let transformer = bundle
            .require(LtxComponent::Transformer)
            .map_err(|error| Error::Msg(error.to_string()))?;
        Self::from_metadata(transformer.metadata().raw_value(Self::METADATA_KEY))
    }

    pub const fn is_dev(self) -> bool {
        matches!(self, Self::Dev)
    }

    pub const fn id(self) -> &'static str {
        match self {
            Self::Distilled => "distilled",
            Self::Dev => "dev",
        }
    }
}

/// The attention-level branch a dev transition runs.  These are deliberately separate from a
/// whole-block selector: STG keeps text cross-attention and both FFNs live at its one skipped
/// self-attention block, while modality isolation keeps self/text/FF live at every block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevBranch {
    Conditional,
    NegativeText,
    StgPerturbed,
    ModalityIsolated,
}

/// Typed variant-specific stage-one schedule.  `sigmas.len() - 1` is the exact number of DiT
/// transitions; callers never accept an arbitrary request-side re-sampling of either checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionPlan {
    pub variant: TransformerVariant,
    pub sigmas: Vec<f32>,
    pub stg_blocks: &'static [u32],
}

impl ExecutionPlan {
    pub fn for_variant(variant: TransformerVariant) -> Self {
        match variant {
            TransformerVariant::Distilled => Self {
                variant,
                sigmas: STAGE1_SIGMAS.to_vec(),
                stg_blocks: &[],
            },
            TransformerVariant::Dev => Self {
                variant,
                sigmas: (0..=LTX_2_5_PARAMS.num_inference_steps)
                    .map(|index| 1.0 - index as f32 / LTX_2_5_PARAMS.num_inference_steps as f32)
                    .collect(),
                stg_blocks: LTX_2_5_PARAMS.video_guider.stg_blocks,
            },
        }
    }

    pub fn transitions(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
    }

    /// Drive the per-transition branch choreography independently from weights.  The actual
    /// Candle sampler uses the same ordering below its tensor operations, so this is a
    /// mutation-sensitive structural witness rather than a default-value test.
    pub fn execute<T, F>(&self, mut forward: F) -> Result<Vec<T>>
    where
        F: FnMut(DevBranch, f32, &[u32]) -> Result<T>,
    {
        let mut outputs = Vec::new();
        for &sigma in self.sigmas.iter().take(self.transitions()) {
            match self.variant {
                TransformerVariant::Distilled => {
                    outputs.push(forward(DevBranch::Conditional, sigma, &[])?);
                }
                TransformerVariant::Dev => {
                    outputs.push(forward(DevBranch::Conditional, sigma, &[])?);
                    outputs.push(forward(DevBranch::NegativeText, sigma, &[])?);
                    outputs.push(forward(DevBranch::StgPerturbed, sigma, self.stg_blocks)?);
                    outputs.push(forward(DevBranch::ModalityIsolated, sigma, &[])?);
                }
            }
        }
        Ok(outputs)
    }
}

/// Combine the four whole-prediction denoised estimates in f32, then apply upstream's 0.7
/// whole-sample standard-deviation rescale and restore the model dtype.
pub fn combine_guidance(
    conditional: &Tensor,
    negative_text: &Tensor,
    perturbed: &Tensor,
    isolated_modality: &Tensor,
    params: GuiderParams,
) -> Result<Tensor> {
    let shape = conditional.dims();
    for (name, value) in [
        ("negative-text", negative_text),
        ("STG-perturbed", perturbed),
        ("modality-isolated", isolated_modality),
    ] {
        if value.dims() != shape {
            return Err(Error::Msg(format!(
                "ltx_2_5: {name} guidance prediction shape {:?} differs from conditional {:?}",
                value.dims(),
                shape
            )));
        }
    }
    let dtype = conditional.dtype();
    let conditional = conditional.to_dtype(DType::F32)?;
    let negative_text = negative_text.to_dtype(DType::F32)?;
    let perturbed = perturbed.to_dtype(DType::F32)?;
    let isolated_modality = isolated_modality.to_dtype(DType::F32)?;
    let cfg = ((&conditional - &negative_text)? * (params.cfg_scale - 1.0) as f64)?;
    let stg = ((&conditional - &perturbed)? * params.stg_scale as f64)?;
    let modality = ((&conditional - &isolated_modality)? * (params.modality_scale - 1.0) as f64)?;
    let guided = (((&conditional + &cfg)? + &stg)? + &modality)?;
    if params.rescale_scale == 0.0 {
        return guided.to_dtype(dtype);
    }
    let conditional_std = whole_prediction_std(&conditional)?;
    let guided_std = whole_prediction_std(&guided)?;
    let ratio = conditional_std.broadcast_div(&guided_std.maximum(1e-12f64)?)?;
    let factor = ((ratio * params.rescale_scale as f64)? + (1.0 - params.rescale_scale) as f64)?;
    guided.broadcast_mul(&factor)?.to_dtype(dtype)
}

/// Torch's default `std` is the sample standard deviation (`correction = 1`) over the entire
/// prediction, not a per-channel/per-token approximation.
fn whole_prediction_std(prediction: &Tensor) -> Result<Tensor> {
    let count = prediction.elem_count();
    if count < 2 {
        return Err(Error::Msg(
            "ltx_2_5: guidance rescale requires at least two prediction elements".into(),
        ));
    }
    let mean = prediction.mean_all()?;
    let centered = prediction.broadcast_sub(&mean)?;
    ((centered.sqr()?.sum_all()? / (count - 1) as f64)?).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NATIVE_STEPS;
    use candle_gen::gen_core::ltx_checkpoint::LtxBundleBuilder;

    fn transformer_bundle(variant: Option<&str>) -> LtxBundle {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transformer.safetensors");
        let metadata = variant
            .map(|variant| format!(r#""variant":"{variant}""#))
            .unwrap_or_default();
        let header = format!(
            r#"{{"__metadata__":{{{metadata}}},"weight":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&0_f32.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        // The builder parses metadata at construction, so the resolved bundle remains valid after
        // this helper's temporary directory is dropped.
        LtxBundleBuilder::new()
            .with_component(LtxComponent::Transformer, path)
            .build()
            .unwrap()
    }

    #[test]
    fn variant_identity_is_exact_and_never_defaults() {
        assert!(TransformerVariant::from_metadata(None).is_err());
        assert!(TransformerVariant::from_metadata(Some("turbo")).is_err());
        assert_eq!(
            TransformerVariant::from_metadata(Some("dev")).unwrap(),
            TransformerVariant::Dev
        );
        assert_eq!(
            TransformerVariant::from_metadata(Some("distilled")).unwrap(),
            TransformerVariant::Distilled
        );
    }

    #[test]
    fn split_transformer_metadata_is_the_provider_identity_source() {
        assert_eq!(
            TransformerVariant::from_bundle(&transformer_bundle(Some("dev"))).unwrap(),
            TransformerVariant::Dev
        );
        assert!(TransformerVariant::from_bundle(&transformer_bundle(None)).is_err());
        assert!(TransformerVariant::from_bundle(&transformer_bundle(Some("unknown"))).is_err());
    }

    #[test]
    fn dev_plan_runs_four_evaluations_for_each_of_thirty_transitions() {
        let plan = ExecutionPlan::for_variant(TransformerVariant::Dev);
        assert_eq!(plan.transitions(), 30);
        assert_eq!(plan.stg_blocks, &[28]);
        let mut calls = Vec::new();
        plan.execute(|branch, sigma, skipped| {
            calls.push((branch, sigma, skipped.to_vec()));
            Ok(())
        })
        .unwrap();
        assert_eq!(calls.len(), 120);
        for chunk in calls.chunks_exact(4) {
            assert_eq!(chunk[0].0, DevBranch::Conditional);
            assert_eq!(chunk[1].0, DevBranch::NegativeText);
            assert_eq!(chunk[2], (DevBranch::StgPerturbed, chunk[2].1, vec![28]));
            assert_eq!(chunk[3].0, DevBranch::ModalityIsolated);
        }
        assert_eq!(
            ExecutionPlan::for_variant(TransformerVariant::Distilled).transitions(),
            NATIVE_STEPS as usize
        );
    }

    #[test]
    fn guidance_combination_uses_every_branch_and_rescales_whole_prediction() -> Result<()> {
        let device = candle_gen::candle_core::Device::Cpu;
        let conditional = Tensor::from_vec(vec![4f32, 8., 12., 16.], (1, 4), &device)?;
        let negative = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 4), &device)?;
        let perturbed = Tensor::from_vec(vec![2f32, 4., 6., 8.], (1, 4), &device)?;
        let isolated = Tensor::from_vec(vec![3f32, 6., 9., 12.], (1, 4), &device)?;
        let no_rescale = GuiderParams {
            cfg_scale: 3.0,
            stg_scale: 1.0,
            stg_blocks: &[28],
            rescale_scale: 0.0,
            modality_scale: 3.0,
        };
        assert_eq!(
            combine_guidance(&conditional, &negative, &perturbed, &isolated, no_rescale)?
                .to_vec2::<f32>()?,
            vec![vec![14.0, 28.0, 42.0, 56.0]]
        );
        let rescaled = combine_guidance(
            &conditional,
            &negative,
            &perturbed,
            &isolated,
            GuiderParams {
                rescale_scale: 0.7,
                ..no_rescale
            },
        )?;
        let expected = 0.7 * 5.163_978_f32 / 18.073_923_f32 + 0.3;
        assert!((rescaled.to_vec2::<f32>()?[0][3] - 56.0 * expected).abs() < 1e-4);
        Ok(())
    }
}
