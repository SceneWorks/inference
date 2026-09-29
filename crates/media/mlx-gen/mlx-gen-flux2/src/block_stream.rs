//! Reopenable FLUX.2 Klein transformer block windows.
//!
//! The non-block trunk remains resident.  When rung 4 is selected the two block stacks are
//! reconstructed independently from the exact transformer directory, evaluated, and released one
//! window at a time.  No converted or temporary artifact is created.

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::{Flux2Config, Flux2Quant};
use crate::transformer::{DoubleBlock, SingleBlock};

#[derive(Clone)]
pub(crate) struct Flux2BlockStream {
    inventory: crate::artifact_inventory::KleinArtifactInventory,
    cfg: Flux2Config,
    quant: Option<Flux2Quant>,
}

impl Flux2BlockStream {
    pub(crate) fn new(
        inventory: crate::artifact_inventory::KleinArtifactInventory,
        cfg: Flux2Config,
        quant: Option<Flux2Quant>,
    ) -> Self {
        Self {
            inventory,
            cfg,
            quant,
        }
    }

    pub(crate) fn double_blocks(&self) -> usize {
        self.cfg.num_double_layers
    }
    pub(crate) fn single_blocks(&self) -> usize {
        self.cfg.num_single_layers
    }

    pub(crate) fn open(&self) -> Result<Weights> {
        self.inventory
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))?;
        let view = Weights::from_dir(self.inventory.transformer_dir()).map_err(|error| {
            Error::Msg(format!(
                "flux2 block stream: open transformer snapshot: {error}"
            ))
        })?;
        self.inventory
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))?;
        Ok(view)
    }

    pub(crate) fn verify_materialized_window(&self) -> Result<()> {
        self.inventory
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))
    }

    pub(crate) fn materialize_double(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<DoubleBlock> {
        if index >= self.double_blocks() {
            return Err(Error::Msg(format!(
                "flux2 block stream: double block {index} is outside the {}-block stack",
                self.double_blocks()
            )));
        }
        crate::loader::alias_transformer_double_block(view, index);
        let block = DoubleBlock::from_weights(
            view,
            &format!("transformer_blocks.{index}"),
            self.cfg.num_heads as i32,
            self.cfg.head_dim as i32,
            self.quant,
        )?;
        view.remove_accessed();
        Ok(block)
    }

    pub(crate) fn materialize_single(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<SingleBlock> {
        if index >= self.single_blocks() {
            return Err(Error::Msg(format!(
                "flux2 block stream: single block {index} is outside the {}-block stack",
                self.single_blocks()
            )));
        }
        let block = SingleBlock::from_weights(
            view,
            &format!("single_transformer_blocks.{index}"),
            self.cfg.num_heads as i32,
            self.cfg.head_dim as i32,
            self.quant,
        )?;
        view.remove_accessed();
        Ok(block)
    }
}

pub(crate) fn evict_resident_blocks<Joint, Single>(
    joint: &mut Vec<Joint>,
    single: &mut Vec<Single>,
    expected_joint: usize,
    expected_single: usize,
) -> Result<()> {
    if joint.len() != expected_joint || single.len() != expected_single {
        return Err(Error::Msg(format!(
            "flux2: cannot finalize double/single stream {expected_joint}/{expected_single} from resident stacks {}/{}",
            joint.len(), single.len()
        )));
    }
    joint.clear();
    single.clear();
    Ok(())
}
