//! The non-distilled LTX-2.5 execution contract.  This is deliberately separate from the baked
//! distilled schedule: a dev checkpoint must execute CFG/STG branches for all thirty transitions,
//! never borrow the eight-step path merely because both checkpoint generations share an architecture.

use std::collections::BTreeMap;

use mlx_gen::{Error, Result};

use crate::params::LTX_2_5_PARAMS;

/// Transformer identity declared by the split transformer's safetensors metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformerVariant {
    Distilled,
    Dev,
}

impl TransformerVariant {
    pub const METADATA_KEY: &'static str = "variant";

    pub fn from_metadata(metadata: &BTreeMap<String, String>) -> Result<Self> {
        match metadata.get(Self::METADATA_KEY).map(String::as_str) {
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

    pub const fn is_dev(self) -> bool {
        matches!(self, Self::Dev)
    }
}

/// Read the one authoritative identity from the resolved transformer component.  A bundle whose
/// components disagree is rejected by the split resolver; this still refuses an absent or unknown
/// transformer tag instead of turning a dev checkpoint into distilled generation.
pub fn from_bundle(
    bundle: &mlx_gen::gen_core::ltx_checkpoint::LtxBundle,
) -> Result<TransformerVariant> {
    let transformer =
        bundle.require(mlx_gen::gen_core::ltx_checkpoint::LtxComponent::Transformer)?;
    let mut metadata = BTreeMap::new();
    if let Some(value) = transformer
        .metadata()
        .raw_value(TransformerVariant::METADATA_KEY)
    {
        metadata.insert(
            TransformerVariant::METADATA_KEY.to_string(),
            value.to_string(),
        );
    }
    TransformerVariant::from_metadata(&metadata)
}

/// Exact branch a dev DiT evaluation must run.  STG is the conditional branch with its declared
/// layer set skipped; CFG combines it with the unconditional and full conditional branches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevBranch {
    Unconditional,
    Conditional,
    StgConditional,
}

/// A typed sampler plan consumed by the MLX LTX generation loop.
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
                sigmas: crate::pipeline::STAGE1_SIGMAS.to_vec(),
                stg_blocks: &[],
            },
            TransformerVariant::Dev => Self {
                variant,
                // LTX's dev row is thirty denoise transitions from sigma 1 to zero.  Keep the
                // generated grid in this typed execution seam rather than reusing the distilled
                // waypoint table; callers can neither shorten it nor silently alias it.
                sigmas: (0..=LTX_2_5_PARAMS.num_inference_steps)
                    .map(|i| 1.0 - i as f32 / LTX_2_5_PARAMS.num_inference_steps as f32)
                    .collect(),
                stg_blocks: LTX_2_5_PARAMS.video_guider.stg_blocks,
            },
        }
    }

    pub fn transitions(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
    }

    /// Drive the exact per-step branch choreography.  The caller supplies the backend-native
    /// DiT forward; this function owns the control-flow invariant and is testable without weights.
    pub fn execute<T, F>(&self, mut forward: F) -> Result<Vec<T>>
    where
        F: FnMut(DevBranch, f32, &[u32]) -> Result<T>,
    {
        let mut out = Vec::new();
        for &sigma in self.sigmas.iter().take(self.transitions()) {
            match self.variant {
                TransformerVariant::Distilled => {
                    out.push(forward(DevBranch::Conditional, sigma, &[])?)
                }
                TransformerVariant::Dev => {
                    out.push(forward(DevBranch::Unconditional, sigma, &[])?);
                    out.push(forward(DevBranch::Conditional, sigma, &[])?);
                    out.push(forward(DevBranch::StgConditional, sigma, self.stg_blocks)?);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::NATIVE_STEPS;

    #[test]
    fn dev_is_thirty_step_three_branch_stg_execution_not_distilled_alias() {
        let dev = ExecutionPlan::for_variant(TransformerVariant::Dev);
        let distilled = ExecutionPlan::for_variant(TransformerVariant::Distilled);
        assert_eq!(dev.transitions(), 30);
        assert_eq!(distilled.transitions(), NATIVE_STEPS as usize);
        assert_ne!(dev.sigmas, distilled.sigmas);
        let mut calls = Vec::new();
        dev.execute(|branch, _, skipped| {
            calls.push((branch, skipped.to_vec()));
            Ok(())
        })
        .unwrap();
        assert_eq!(calls.len(), 90);
        assert!(calls
            .iter()
            .any(|(branch, skipped)| *branch == DevBranch::StgConditional && skipped == &[28]));
    }

    #[test]
    fn variant_metadata_never_defaults_on_unknown_or_missing_identity() {
        assert!(TransformerVariant::from_metadata(&BTreeMap::new()).is_err());
        let mut bad = BTreeMap::new();
        bad.insert("variant".into(), "turbo".into());
        assert!(TransformerVariant::from_metadata(&bad).is_err());
        bad.insert("variant".into(), "dev".into());
        assert_eq!(
            TransformerVariant::from_metadata(&bad).unwrap(),
            TransformerVariant::Dev
        );
    }
}
