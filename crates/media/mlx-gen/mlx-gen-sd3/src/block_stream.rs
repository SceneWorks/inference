//! Reopenable one-block windows for the SD3.5 MMDiT.

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::Sd3Arch;
use crate::transformer::JointBlock;

#[derive(Clone)]
pub(crate) struct Sd3BlockStream {
    inventory: crate::artifact_inventory::Sd3ArtifactInventory,
    arch: Sd3Arch,
}

impl Sd3BlockStream {
    pub(crate) fn new(
        inventory: crate::artifact_inventory::Sd3ArtifactInventory,
        arch: Sd3Arch,
    ) -> Self {
        Self { inventory, arch }
    }

    pub(crate) fn blocks(&self) -> usize {
        self.arch.num_layers
    }

    pub(crate) fn open(&self) -> Result<Weights> {
        self.inventory.ensure_unchanged()?;
        let weights = Weights::from_dir(self.inventory.transformer_dir()).map_err(|error| {
            Error::Msg(format!(
                "sd3 block stream: open transformer snapshot: {error}"
            ))
        })?;
        self.inventory.ensure_unchanged()?;
        Ok(weights)
    }

    pub(crate) fn materialize(&self, weights: &mut Weights, index: usize) -> Result<JointBlock> {
        if index >= self.blocks() {
            return Err(Error::Msg(format!(
                "sd3 block stream: block {index} is outside the {}-block stack",
                self.blocks()
            )));
        }
        let block = JointBlock::from_weights(
            weights,
            index,
            index + 1 == self.blocks(),
            self.arch.is_dual_attention_block(index),
            self.arch.num_heads as i32,
            self.arch.head_dim as i32,
        )?;
        weights.remove_accessed();
        Ok(block)
    }

    pub(crate) fn verify_materialized_window(&self) -> Result<()> {
        self.inventory.ensure_unchanged()
    }
}

pub(crate) fn evict_resident_blocks(blocks: &mut Vec<JointBlock>, expected: usize) -> Result<()> {
    if blocks.len() != expected {
        return Err(Error::Msg(format!(
            "sd3: cannot finalize {expected}-block stream from {} resident blocks",
            blocks.len()
        )));
    }
    blocks.clear();
    Ok(())
}
