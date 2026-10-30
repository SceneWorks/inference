//! **Activation checkpointing** of the Iris trainers (story sc-25689): upstream's
//! `models/ac.py` policies (`train.activation_checkpointing`, `train.ac_selective_every`) and the
//! regions they recompute, shared by the MLX and Candle trainers.
//!
//! A *region* is one trunk block or one PiT pixel block; the index counts within its own stack
//! (`should_checkpoint(policy, block_index, every)`), so the trunk and the pixel refiner are
//! strided independently. A checkpointed region keeps only its inputs through the forward and
//! recomputes its interior during the backward, so the gradient is that of the plain graph.
//!
//! Native semantic mapping (E3):
//! * `full` / `selective_layer` — exactly upstream's region set. MLX recomputes each region through
//!   MLX's own checkpoint transform (`mx.checkpoint`, the native counterpart of
//!   `torch.utils.checkpoint`); Candle, which has no checkpoint primitive, runs the region outside
//!   the tape in the forward and replays it on the tape during a segmented backward (chain rule over
//!   the region boundary), producing the same gradient.
//! * `selective_op` — upstream enters every region but keeps the outputs of the aten matmul /
//!   attention / convolution kernels through torch's dispatcher-level selective-save cache. Neither
//!   MLX nor Candle has a per-op save policy for its autodiff, so natively every region is entered
//!   and recomputed whole (the `full` region set). The gradient — the only training-visible quantity
//!   — is identical (upstream documents the policy as changing memory/compute only, never the
//!   update); what differs is the memory/recompute operating point.

use gen_core::{Error, Result};

/// `train.activation_checkpointing`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcPolicy {
    None,
    Full,
    SelectiveOp,
    SelectiveLayer { every: usize },
}

/// `ac.POLICIES`.
pub const POLICIES: [&str; 4] = ["none", "full", "selective_op", "selective_layer"];

impl AcPolicy {
    /// `validate_policy` + the `selective_layer` stride.
    pub fn parse(policy: &str, every: usize) -> Result<Self> {
        match policy {
            "none" => Ok(Self::None),
            "full" => Ok(Self::Full),
            "selective_op" => Ok(Self::SelectiveOp),
            "selective_layer" => {
                if every < 1 {
                    return Err(Error::Msg(format!(
                        "iris training: activation_checkpointing='selective_layer' recomputes \
                         every Nth block and needs ac_selective_every >= 1, got {every}"
                    )));
                }
                Ok(Self::SelectiveLayer { every })
            }
            other => Err(Error::Msg(format!(
                "iris training: unknown activation_checkpointing '{other}' ({})",
                POLICIES.join(" | ")
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Full => "full",
            Self::SelectiveOp => "selective_op",
            Self::SelectiveLayer { .. } => "selective_layer",
        }
    }

    /// `should_checkpoint`: whether region `block_index` (0-based within its stack) recomputes.
    pub fn should_checkpoint(self, block_index: usize) -> bool {
        match self {
            Self::None => false,
            Self::Full | Self::SelectiveOp => true,
            Self::SelectiveLayer { every } => block_index.is_multiple_of(every),
        }
    }

    /// Whether any region recomputes.
    pub fn active(self) -> bool {
        self != Self::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policies_select_upstreams_regions() {
        assert!(!AcPolicy::parse("none", 2).unwrap().should_checkpoint(0));
        assert!(AcPolicy::parse("full", 2).unwrap().should_checkpoint(5));
        assert!(AcPolicy::parse("selective_op", 2)
            .unwrap()
            .should_checkpoint(3));
        let sl = AcPolicy::parse("selective_layer", 3).unwrap();
        let picked: Vec<usize> = (0..8).filter(|&i| sl.should_checkpoint(i)).collect();
        assert_eq!(picked, vec![0, 3, 6]);
        assert!(AcPolicy::parse("selective_layer", 0).is_err());
        assert!(AcPolicy::parse("block", 2).is_err());
    }
}
