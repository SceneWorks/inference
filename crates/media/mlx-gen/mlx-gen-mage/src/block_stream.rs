//! Snapshot-backed materialization of Mage-Flow's uniform 12-block DiT stack.

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::config::MageFlowConfig;
use crate::transformer_block::MageTransformerBlock;

#[derive(Clone, Default)]
struct BlockAdapters(Vec<(String, Vec<Adapter>)>);

impl BlockAdapters {
    fn install(&self, block: &mut MageTransformerBlock) -> Result<()> {
        for (path, adapters) in &self.0 {
            let segments = path.split('.').collect::<Vec<_>>();
            let target = block.adaptable_mut(&segments).ok_or_else(|| {
                Error::Msg(format!(
                    "mage_flow block stream: missing adapter target `{path}`"
                ))
            })?;
            target.set_adapters(adapters.clone());
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct MageBlockStream {
    source: WeightsSource,
    cfg: MageFlowConfig,
    quant_bits: Option<i32>,
    adapters: Vec<BlockAdapters>,
}

impl MageBlockStream {
    pub(crate) fn new(source: WeightsSource, cfg: MageFlowConfig) -> Self {
        let depth = cfg.depth;
        Self {
            source,
            cfg,
            quant_bits: None,
            adapters: vec![BlockAdapters::default(); depth],
        }
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.cfg.depth
    }

    pub(crate) fn set_quant_bits(&mut self, bits: i32) {
        self.quant_bits = Some(bits);
    }

    pub(crate) fn capture_adapters(&mut self, blocks: &mut [MageTransformerBlock]) {
        self.adapters = blocks
            .iter_mut()
            .map(|block| {
                let mut captured = Vec::new();
                for path in block.adaptable_paths() {
                    let segments = path.split('.').collect::<Vec<_>>();
                    if let Some(target) = block.adaptable_mut(&segments) {
                        if !target.adapters().is_empty() {
                            captured.push((path, target.adapters().to_vec()));
                        }
                    }
                }
                BlockAdapters(captured)
            })
            .collect();
    }

    pub(crate) fn open(&self) -> Result<Weights> {
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    pub(crate) fn materialize(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<MageTransformerBlock> {
        if index >= self.n_blocks() {
            return Err(Error::Msg(format!(
                "mage_flow block stream: block {index} is outside the {}-block stack",
                self.n_blocks()
            )));
        }
        let mut block = MageTransformerBlock::from_weights(
            view,
            &format!("transformer_blocks.{index}"),
            &self.cfg,
        )?;
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            block.quantize(bits)?;
        }
        self.adapters[index].install(&mut block)?;
        Ok(block)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BlockWindow<'a> {
    pub(crate) plan: mlx_gen::block_residency::BlockPlan,
    pub(crate) cancel: &'a mlx_gen::CancelFlag,
}
