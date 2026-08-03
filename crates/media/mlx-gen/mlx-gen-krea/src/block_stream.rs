//! Family-side bounded-residency loader for Krea 2's 28-block DiT trunk (SC-16352).
//!
//! Window scheduling and teardown stay in [`mlx_gen::block_residency::run_windowed`]. This module
//! only knows how to reopen Krea's transformer snapshot, rebuild one exact `SingleStreamBlock`, replay
//! the resident block's quantization and forward-time adapters, and drain the lazy view of the tensors
//! that constructor read.

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::config::Krea2Config;
use crate::transformer::block::SingleStreamBlock;

#[derive(Clone, Default)]
struct BlockAdapters {
    per_path: Vec<(String, Vec<Adapter>)>,
}

impl BlockAdapters {
    fn install(&self, block: &mut SingleStreamBlock) -> Result<()> {
        for (path, adapters) in &self.per_path {
            let segments: Vec<&str> = path.split('.').collect();
            let target = block.adaptable_mut(&segments).ok_or_else(|| {
                Error::Msg(format!(
                    "krea block stream: adapter target `{path}` is absent from a materialized block"
                ))
            })?;
            target.set_adapters(adapters.clone());
        }
        Ok(())
    }
}

/// Re-openable description of Krea's uniform `transformer_blocks` stack.
#[derive(Clone)]
pub(crate) struct KreaBlockStream {
    source: WeightsSource,
    cfg: Krea2Config,
    quant_bits: Option<i32>,
    adapters: Vec<BlockAdapters>,
    #[cfg(test)]
    test_blocks: Option<Vec<SingleStreamBlock>>,
    #[cfg(test)]
    test_materializations: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
}

impl KreaBlockStream {
    pub(crate) fn new(source: WeightsSource, cfg: Krea2Config) -> Self {
        let n_blocks = cfg.num_layers;
        Self {
            source,
            cfg,
            quant_bits: None,
            adapters: vec![BlockAdapters::default(); n_blocks],
            #[cfg(test)]
            test_blocks: None,
            #[cfg(test)]
            test_materializations: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        cfg: Krea2Config,
        blocks: Vec<SingleStreamBlock>,
        materializations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        let mut stream = Self::new(WeightsSource::File(std::path::PathBuf::new()), cfg);
        stream.test_blocks = Some(blocks);
        stream.test_materializations = Some(materializations);
        stream
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.cfg.num_layers
    }

    pub(crate) fn set_quant_bits(&mut self, bits: i32) {
        self.quant_bits = Some(bits);
    }

    /// Capture only forward-time residuals. Diff-patch loads never construct a stream: their full
    /// weight deltas mutate the resident base and cannot be reconstructed from the pristine snapshot.
    pub(crate) fn capture_adapters(&mut self, blocks: &mut [SingleStreamBlock]) {
        self.adapters = blocks
            .iter_mut()
            .map(|block| {
                let mut per_path = Vec::new();
                for path in block.adaptable_paths() {
                    let segments: Vec<&str> = path.split('.').collect();
                    if let Some(target) = block.adaptable_mut(&segments) {
                        let adapters = target.adapters();
                        if !adapters.is_empty() {
                            per_path.push((path, adapters.to_vec()));
                        }
                    }
                }
                BlockAdapters { per_path }
            })
            .collect();
    }

    pub(crate) fn open(&self) -> Result<Weights> {
        #[cfg(test)]
        if self.test_blocks.is_some() {
            return Ok(Weights::empty());
        }
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    pub(crate) fn materialize(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<SingleStreamBlock> {
        if index >= self.n_blocks() {
            return Err(Error::Msg(format!(
                "krea block stream: block {index} is out of range for a {}-block stack",
                self.n_blocks()
            )));
        }
        #[cfg(test)]
        if let Some(blocks) = &self.test_blocks {
            if let Some(counter) = &self.test_materializations {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            return blocks.get(index).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "krea test block stream: block {index} is absent from a {}-block fixture",
                    blocks.len()
                ))
            });
        }
        let cfg = &self.cfg;
        let mut block = SingleStreamBlock::from_weights(
            view,
            &format!("transformer_blocks.{index}"),
            cfg.num_attention_heads as i32,
            cfg.num_kv_heads as i32,
            cfg.attention_head_dim as i32,
            cfg.hidden_size as i32,
            cfg.norm_eps,
        )?;
        // `Array` handles are refcounted. Removing the exact read set prevents the view from retaining
        // a second handle after the materialized window is dropped.
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            block.quantize(bits)?;
        }
        if let Some(adapters) = self.adapters.get(index) {
            adapters.install(&mut block)?;
        }
        Ok(block)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BlockWindow<'a> {
    pub(crate) plan: mlx_gen::block_residency::BlockPlan,
    pub(crate) cancel: &'a mlx_gen::CancelFlag,
}
