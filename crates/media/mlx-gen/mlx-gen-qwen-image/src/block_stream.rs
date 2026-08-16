//! Ladder rung 4 — family-side bounded residency for Qwen-Image's uniform DiT stack (SC-16353).
//!
//! The window lifecycle is the shared [`mlx_gen::block_residency::run_windowed`] primitive. This
//! module owns only the Qwen-specific reconstruction: reopen the diffusers checkpoint, apply the
//! production key remap, rebuild one [`QwenTransformerBlock`], drain exactly the tensors its
//! constructor read, then replay the resident block's quantization and adapters.

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::transformer::QwenTransformerBlock;

#[derive(Clone, Default)]
struct BlockAdapters {
    per_path: Vec<(String, Vec<Adapter>)>,
}

impl BlockAdapters {
    fn install(&self, block: &mut QwenTransformerBlock) -> Result<()> {
        for (path, adapters) in &self.per_path {
            let segments: Vec<&str> = path.split('.').collect();
            let target = block.adaptable_mut(&segments).ok_or_else(|| {
                Error::Msg(format!(
                    "qwen-image block stream: adapter target `{path}` is absent from the rebuilt block"
                ))
            })?;
            target.set_adapters(adapters.clone());
        }
        Ok(())
    }
}

/// Re-openable description of the 60-block Qwen-Image DiT trunk.
#[derive(Clone)]
pub(crate) struct QwenBlockStream {
    source: WeightsSource,
    base: String,
    num_heads: i32,
    head_dim: i32,
    n_blocks: usize,
    quant_bits: Option<i32>,
    adapters: Vec<BlockAdapters>,
}

impl QwenBlockStream {
    pub(crate) fn new(
        source: WeightsSource,
        base: impl Into<String>,
        num_heads: i32,
        head_dim: i32,
        n_blocks: usize,
    ) -> Self {
        Self {
            source,
            base: base.into(),
            num_heads,
            head_dim,
            n_blocks,
            quant_bits: None,
            adapters: vec![BlockAdapters::default(); n_blocks],
        }
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    pub(crate) fn set_quant_bits(&mut self, bits: i32) {
        self.quant_bits = Some(bits);
    }

    #[cfg(test)]
    fn has_adapters(&self) -> bool {
        self.adapters.iter().any(|block| !block.per_path.is_empty())
    }

    /// Capture the exact forward-time residuals installed on the resident blocks.
    pub(crate) fn capture_adapters(&mut self, blocks: &mut [QwenTransformerBlock]) {
        self.adapters = blocks
            .iter_mut()
            .map(|block| {
                let mut per_path = Vec::new();
                for path in block.adaptable_paths() {
                    let segments: Vec<&str> = path.split('.').collect();
                    // SC-18319 — a capture walks EVERY projection, so it resolves through the PROBE
                    // half and takes the `&mut` only where something is actually installed.
                    if block
                        .adaptable_facts(&segments)
                        .is_some_and(|f| f.adapter_count > 0)
                    {
                        let adapters = block
                            .adaptable_mut(&segments)
                            .expect("resolved through the probe above")
                            .adapters()
                            .to_vec();
                        per_path.push((path, adapters));
                    }
                }
                BlockAdapters { per_path }
            })
            .collect();
    }

    /// Open the same lazily-backed checkpoint view used by the resident loader, including its
    /// diffusers-to-internal aliases.
    pub(crate) fn open(&self) -> Result<Weights> {
        let mut weights = match &self.source {
            WeightsSource::Dir(path) => Weights::from_dir(path),
            WeightsSource::File(path) => Weights::from_file(path),
        }?;
        crate::loader::remap_transformer_keys(&mut weights);
        Ok(weights)
    }

    pub(crate) fn materialize(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<QwenTransformerBlock> {
        if index >= self.n_blocks {
            return Err(Error::Msg(format!(
                "qwen-image block stream: block {index} is out of range for {} blocks",
                self.n_blocks
            )));
        }
        let prefix = format!("{}.{index}", self.base);
        let mut block =
            QwenTransformerBlock::from_weights(view, &prefix, self.num_heads, self.head_dim)?;
        // Array handles are refcounted. Removing exactly the accessed keys is what lets dropping a
        // completed window release its materialized buffers instead of retaining a second reference.
        view.remove_accessed();
        crate::loader::remove_transformer_source_aliases(view, &format!("{prefix}."));
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::adapters::Adapter;
    use mlx_rs::Array;

    fn fixture(tmp: &tempfile::TempDir) -> std::path::PathBuf {
        let path = tmp.path().join(format!(
            "qwen-block-stream-{}.safetensors",
            std::process::id()
        ));
        let w8 = Array::from_slice(&vec![0.1f32; 64], &[8, 8]);
        let b8 = Array::from_slice(&[0.0f32; 8], &[8]);
        let n4 = Array::from_slice(&[1.0f32; 4], &[4]);
        let w48x8 = Array::from_slice(&vec![0.1f32; 48 * 8], &[48, 8]);
        let b48 = Array::from_slice(&[0.0f32; 48], &[48]);
        let w16x8 = Array::from_slice(&vec![0.1f32; 16 * 8], &[16, 8]);
        let b16 = Array::from_slice(&[0.0f32; 16], &[16]);
        let w8x16 = Array::from_slice(&vec![0.1f32; 8 * 16], &[8, 16]);
        fn linear<'a>(
            named: &mut Vec<(String, &'a Array)>,
            prefix: &str,
            weight: &'a Array,
            bias: &'a Array,
        ) {
            named.push((format!("transformer_blocks.0.{prefix}.weight"), weight));
            named.push((format!("transformer_blocks.0.{prefix}.bias"), bias));
        }
        let mut named: Vec<(String, &Array)> = Vec::new();
        linear(&mut named, "img_mod.1", &w48x8, &b48);
        linear(&mut named, "txt_mod.1", &w48x8, &b48);
        for prefix in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_out.0",
            "attn.to_add_out",
        ] {
            linear(&mut named, prefix, &w8, &b8);
        }
        for prefix in [
            "attn.norm_q",
            "attn.norm_k",
            "attn.norm_added_q",
            "attn.norm_added_k",
        ] {
            named.push((format!("transformer_blocks.0.{prefix}.weight"), &n4));
        }
        for stream in ["img_mlp", "txt_mlp"] {
            linear(&mut named, &format!("{stream}.net.0.proj"), &w16x8, &b16);
            linear(&mut named, &format!("{stream}.net.2"), &w8x16, &b8);
        }
        let refs: Vec<(&str, &Array)> = named
            .iter()
            .map(|(name, array)| (name.as_str(), *array))
            .collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        path
    }

    #[test]
    fn stream_reopens_remaps_and_drains_only_the_materialized_block() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp);
        let stream = QwenBlockStream::new(
            WeightsSource::File(path.clone()),
            "transformer_blocks",
            2,
            4,
            1,
        );
        let mut view = stream.open().unwrap();
        let before = view.keys().count();
        let _block = stream.materialize(&mut view, 0).unwrap();
        assert!(before > 0);
        assert_eq!(
            view.keys().count(),
            0,
            "every fixture tensor was read and drained"
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn out_of_range_block_is_rejected() {
        let stream = QwenBlockStream::new(
            WeightsSource::File("/nonexistent".into()),
            "transformer_blocks",
            2,
            4,
            1,
        );
        let mut view = Weights::empty();
        assert!(stream.materialize(&mut view, 1).is_err());
    }

    #[test]
    fn captured_adapters_are_reinstalled_on_the_matching_block() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp);
        let mut stream = QwenBlockStream::new(
            WeightsSource::File(path.clone()),
            "transformer_blocks",
            2,
            4,
            1,
        );
        let mut view = stream.open().unwrap();
        let mut resident = stream.materialize(&mut view, 0).unwrap();
        resident
            .adaptable_mut(&["attn", "to_q"])
            .unwrap()
            .push(Adapter::Lora {
                a: Array::from_slice(&[0.5_f32; 16], &[8, 2]),
                b: Array::from_slice(&[0.25_f32; 16], &[2, 8]),
                scale: 1.0,
            });
        stream.capture_adapters(std::slice::from_mut(&mut resident));
        assert!(stream.has_adapters());

        let mut view = stream.open().unwrap();
        let mut rebuilt = stream.materialize(&mut view, 0).unwrap();
        assert_eq!(
            rebuilt
                .adaptable_mut(&["attn", "to_q"])
                .unwrap()
                .adapters()
                .len(),
            1,
            "streamed block must replay the resident block's forward-time adapter"
        );
        std::fs::remove_file(path).ok();
    }
}
