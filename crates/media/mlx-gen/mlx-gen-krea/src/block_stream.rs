//! Family-side bounded-residency loader for Krea 2's 28-block DiT trunk (SC-16352).
//!
//! Window scheduling and teardown stay in [`mlx_gen::block_residency::run_windowed`]. This module
//! only knows how to reopen Krea's transformer snapshot, rebuild one exact `SingleStreamBlock`, replay
//! the resident block's quantization and forward-time adapters, and drain the lazy view of the tensors
//! that constructor read.

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, PinnedWeightsFile, Result, WeightsSource};

use crate::config::Krea2Config;
use crate::transformer::block::SingleStreamBlock;

static NATIVE_WINDOW_REOPENS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static BLOCK_MATERIALIZATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Process-wide diagnostics for Krea's bounded transformer-residency path.
///
/// These counters intentionally observe completed operations rather than requested flags: a native
/// reopen increments only after the pinned File was revalidated and parsed, and a materialization
/// increments only after the exact block and any forward-time adapters were assembled. They let
/// on-device acceptance distinguish a real block-window run from staged component residency alone.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BlockStreamDiagnostics {
    pub native_window_reopens: u64,
    pub block_materializations: u64,
}

/// Reset the process-wide block-stream diagnostics before a serialized on-device measurement.
pub fn reset_block_stream_diagnostics() {
    NATIVE_WINDOW_REOPENS.store(0, std::sync::atomic::Ordering::Relaxed);
    BLOCK_MATERIALIZATIONS.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Snapshot completed native-window operations since the last reset.
pub fn block_stream_diagnostics() -> BlockStreamDiagnostics {
    BlockStreamDiagnostics {
        native_window_reopens: NATIVE_WINDOW_REOPENS.load(std::sync::atomic::Ordering::Relaxed),
        block_materializations: BLOCK_MATERIALIZATIONS.load(std::sync::atomic::Ordering::Relaxed),
    }
}

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
    source: KreaBlockSource,
    cfg: Krea2Config,
    quant_bits: Option<i32>,
    adapters: Vec<BlockAdapters>,
    #[cfg(test)]
    test_blocks: Option<Vec<SingleStreamBlock>>,
    #[cfg(test)]
    test_materializations: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
}

#[derive(Clone)]
enum KreaBlockSource {
    Diffusers(WeightsSource),
    /// Native/ComfyUI keys are normalized on every fresh view. The retained source path is the
    /// extension-bearing loader path and is lstat/target pinned; it is never canonicalized to an HF
    /// cache blob.
    Native(Box<PinnedWeightsFile>),
}

impl KreaBlockStream {
    pub(crate) fn new(source: WeightsSource, cfg: Krea2Config) -> Self {
        let n_blocks = cfg.num_layers;
        Self {
            source: KreaBlockSource::Diffusers(source),
            cfg,
            quant_bits: None,
            adapters: vec![BlockAdapters::default(); n_blocks],
            #[cfg(test)]
            test_blocks: None,
            #[cfg(test)]
            test_materializations: None,
        }
    }

    pub(crate) fn new_native(source: PinnedWeightsFile, cfg: Krea2Config) -> Self {
        let n_blocks = cfg.num_layers;
        Self {
            source: KreaBlockSource::Native(Box::new(source)),
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
            KreaBlockSource::Diffusers(WeightsSource::Dir(dir)) => Weights::from_dir(dir),
            KreaBlockSource::Diffusers(WeightsSource::File(file)) => Weights::from_file(file),
            KreaBlockSource::Native(file) => file.read_unchanged(|path| {
                let weights = crate::loader::normalized_native_weights_lazy(path)?;
                NATIVE_WINDOW_REOPENS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(weights)
            }),
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
            let block = blocks.get(index).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "krea test block stream: block {index} is absent from a {}-block fixture",
                    blocks.len()
                ))
            })?;
            if let Some(counter) = &self.test_materializations {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            BLOCK_MATERIALIZATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(block);
        }
        if let KreaBlockSource::Native(file) = &self.source {
            return file.read_unchanged(|_| self.materialize_from_view(view, index, true));
        }
        self.materialize_from_view(view, index, false)
    }

    fn materialize_from_view(
        &self,
        view: &mut Weights,
        index: usize,
        source_guarded: bool,
    ) -> Result<SingleStreamBlock> {
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
        if source_guarded {
            // Evaluate only this block's exact read set before the native pin's post-check. Evaluating
            // the whole normalized map would make File reopening physically correct but memory-bound
            // in name only; the accessed subset preserves the real windowed implementation.
            view.materialize_accessed()?;
        }
        // `Array` handles are refcounted. Removing the exact read set prevents the view from retaining
        // a second handle after the materialized window is dropped.
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            block.quantize(bits)?;
        }
        if let Some(adapters) = self.adapters.get(index) {
            adapters.install(&mut block)?;
        }
        BLOCK_MATERIALIZATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(block)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BlockWindow<'a> {
    pub(crate) plan: mlx_gen::block_residency::BlockPlan,
    pub(crate) cancel: &'a mlx_gen::CancelFlag,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_window_reopen_rejects_a_changed_pinned_file_before_parsing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("imported-krea.safetensors");
        std::fs::write(&file, b"initial invalid fixture").expect("write initial fixture");
        let pinned = PinnedWeightsFile::pin(&file).expect("pin fixture");
        let stream = KreaBlockStream::new_native(pinned, Krea2Config::turbo());

        std::fs::write(&file, b"replacement invalid fixture with a different size")
            .expect("replace fixture");
        let error = match stream.open() {
            Ok(_) => panic!("the next transformer window must reject changed bytes"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("pinned weights") && error.contains("changed after load"),
            "got: {error}"
        );
    }
}
