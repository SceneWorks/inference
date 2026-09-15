//! Ladder rung 4 — **bounded transformer residency** for the ChatGLM3-6B conditioning tower
//! (SC-15521), i.e. the `TransformerComponent::TextEncoder` scope.
//!
//! The window lifecycle is NOT implemented here: it is [`mlx_gen::block_residency::run_windowed`]
//! (SC-15750), the shared primitive every other adopter drives. This module is only the family-side
//! half — "how do I rebuild GLM block *n* from the `text_encoder/` snapshot, carrying the same tier
//! the resident block carries".
//!
//! The U-Net half of Kolors' rung 4 is `mlx_gen_sdxl::block_stream`, reached through the re-exported
//! `UNet2DConditionModel`. This module is the *other* scope, and Kolors is the first provider on the
//! ladder to implement it.
//!
//! ## Why a text-encoder window is worth implementing here and is not on SDXL
//!
//! `gen_core::TransformerComponent::TextEncoder` has existed since SC-15449 and no MLX provider had
//! ever populated it, for a good reason: on every prior adopter the encoder is small, rung 1 sheds
//! it before the denoise, and the conditioning phase therefore never carries the request peak — so
//! windowing it would bound a phase that is not the maximum, which the epic is explicit is not a
//! saving.
//!
//! Kolors breaks that. Its tower is **ChatGLM3-6B**: 11.63 GiB dense, larger than the U-Net at every
//! tier, and at the `bf16` tier's smallest advertised output the conditioning phase **is** the
//! peak-bearing one (11.3599 GiB against a 11.3599 GiB request peak — measured by
//! `tests/memory_ladder_real_weights.rs::the_text_encoder_window_scope_cannot_move_the_request_peak`).
//! One cell of nine, but a cell a caller can actually ask for.
//!
//! ## The cost side is the opposite shape to the DiT's
//!
//! `gen_core`'s own docs name it: the encoder re-materializes **once per generation** while the
//! U-Net re-materializes once per denoise **step**. So the same mechanism has a completely different
//! price here — 28 block re-opens against a phase that runs once, versus 70 across eleven sub-stacks
//! against a phase that runs 4–50 times. That is why the two scopes are separately selectable rather
//! than folded into one flag.
//!
//! ## No adapters, and the guard that keeps that true
//!
//! Kolors' LoRA/LoKr surface targets the **U-Net** only (`apply_sdxl_adapters_with`), and the
//! IP-Adapter installs decoupled K/V into the U-Net's cross-attention. Nothing is installed on a
//! `GlmBlock` at load, so — unlike `mlx_gen_sdxl::block_stream` — there is no forward-time state to
//! capture and replay. That is a fact about today's loader, not a law, so
//! [`GlmBlockStream::materialize`] rebuilds a block through the **same** `GlmBlock::from_weights`
//! constructor the resident stack uses: if a future story installs per-block state, the streamed
//! block loses it in exactly the way a reviewer can see, and the byte-identity assertion in
//! `transformer_window_sweep_and_streamed_output_identity` fails loudly rather than rendering a
//! plausible wrong image.

use mlx_rs::Dtype;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::chatglm3::GlmBlock;

/// Everything needed to rebuild one ChatGLM3 block from the `text_encoder/` snapshot, on demand.
///
/// Held by [`ChatGlmModel`](crate::chatglm3::ChatGlmModel) and consumed by its windowed encode.
/// Cheap to clone: a path, two scalars and an enum.
#[derive(Clone)]
pub(crate) struct GlmBlockStream {
    /// The re-openable `text_encoder/` directory. A directory rather than a file because the `bf16`
    /// tier ships the tower as **three shards plus an index**, which `Weights::from_dir` resolves
    /// and a single-file source could not.
    source: WeightsSource,
    n_blocks: usize,
    dtype: Dtype,
    /// The load-time quantization the resident stack received, replayed per materialized block so a
    /// streamed block is byte-identical to its resident twin. `None` for a dense tier. On a
    /// pre-quantized (packed) snapshot this is recorded but is a documented no-op — the same no-op
    /// the resident path performed, because `ChatGlmLinear::load` packed-detects off disk.
    quant_bits: Option<i32>,
}

impl GlmBlockStream {
    pub(crate) fn new(
        source: WeightsSource,
        n_blocks: usize,
        dtype: Dtype,
        quant_bits: Option<i32>,
    ) -> Self {
        Self {
            source,
            n_blocks,
            dtype,
            quant_bits,
        }
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Open a fresh lazy view of the `text_encoder/` snapshot. Called once per window by
    /// [`run_windowed`](mlx_gen::block_residency::run_windowed).
    ///
    /// `Weights::from_dir` is lazy per tensor, so a re-open costs the shard headers' JSON parse and
    /// not the tier. On the sharded `bf16` tower that is three parses per window — the reason the
    /// text-encoder scope's cost side is stated separately from the U-Net's.
    pub(crate) fn open(&self) -> Result<Weights> {
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    /// Materialize block `index` out of `view` — quantized exactly like its resident twin — then
    /// drain the view of precisely the tensors that block read.
    pub(crate) fn materialize(&self, view: &mut Weights, index: usize) -> Result<GlmBlock> {
        if index >= self.n_blocks {
            return Err(Error::Msg(format!(
                "kolors chatglm3 block stream: block {index} is out of range for a {}-block stack",
                self.n_blocks
            )));
        }
        let mut block = GlmBlock::from_weights(view, index, None, self.dtype)?;
        // LOAD-BEARING (SC-15750): the view keeps its own refcounted handle to every tensor the
        // constructor cloned. Draining exactly the accessed keys is what makes the window's drop a
        // real release rather than a no-op that still produces correct hidden states.
        //
        // Exact rather than by prefix, for the same reason `mlx_gen_sdxl::block_stream` drains
        // exactly: a `remove_prefix` would drain the same bytes here but would also hide a
        // constructor that silently stopped reading a tensor.
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            block.quantize(bits)?;
        }
        Ok(block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chatglm3::ChatGlmConfig;
    use mlx_rs::Array;

    const HIDDEN: i32 = 64;
    const KV: i32 = 2;
    const HEADS: i32 = 4;

    /// A tiny two-block ChatGLM3 encoder written to a temp `.safetensors`, so the stream can be
    /// exercised without the 4-12 GiB tower. Widths are group-64-divisible so the quantization
    /// replay test can pack them.
    fn fixture(tmp: &tempfile::TempDir, tag: &str, n_blocks: usize) -> std::path::PathBuf {
        let dir = tmp.path().join(format!("kolors-glm-stream-{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let mut named: Vec<(String, Array)> = Vec::new();
        let mut push = |key: String, shape: Vec<i32>| {
            let n: i32 = shape.iter().product();
            let values: Vec<f32> = (0..n).map(|k| ((k % 97) as f32) * 0.01).collect();
            named.push((key, Array::from_slice(&values, &shape)));
        };
        let head_dim = HIDDEN / HEADS;
        let qkv_out = HIDDEN + 2 * KV * head_dim;
        for i in 0..n_blocks {
            let b = format!("encoder.layers.{i}");
            push(format!("{b}.input_layernorm.weight"), vec![HIDDEN]);
            push(format!("{b}.post_attention_layernorm.weight"), vec![HIDDEN]);
            push(
                format!("{b}.self_attention.query_key_value.weight"),
                vec![qkv_out, HIDDEN],
            );
            push(
                format!("{b}.self_attention.query_key_value.bias"),
                vec![qkv_out],
            );
            push(
                format!("{b}.self_attention.dense.weight"),
                vec![HIDDEN, HIDDEN],
            );
            push(
                format!("{b}.mlp.dense_h_to_4h.weight"),
                vec![2 * HIDDEN, HIDDEN],
            );
            push(
                format!("{b}.mlp.dense_4h_to_h.weight"),
                vec![HIDDEN, HIDDEN],
            );
        }
        let refs: Vec<(&str, &Array)> = named.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        path
    }

    fn stream(path: &std::path::Path, n: usize) -> GlmBlockStream {
        GlmBlockStream::new(
            WeightsSource::File(path.to_path_buf()),
            n,
            Dtype::Float32,
            None,
        )
    }

    /// The drain must remove EXACTLY the block's own tensors — not a whole prefix.
    ///
    /// The distinction is load-bearing: exact draining leaves an un-read key visible, which is what
    /// the second half asserts. A prefix drain would erase the evidence of a constructor that
    /// silently stopped reading a tensor.
    #[test]
    fn block_stream_drains_exactly_what_the_block_read() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "drain", 2);
        let s = stream(&path, 2);
        let mut view = s.open().unwrap();
        let before = view.len();
        s.materialize(&mut view, 0).unwrap();
        let after = view.len();
        assert!(
            after < before,
            "materializing block 0 must drain its tensors"
        );
        assert!(view
            .get("encoder.layers.1.self_attention.dense.weight")
            .is_some());
        assert!(view
            .get("encoder.layers.0.self_attention.dense.weight")
            .is_none());
        assert_eq!(
            view.keys()
                .filter(|k| k.starts_with("encoder.layers.0."))
                .count(),
            0,
            "every block-0 tensor the constructor read must have been drained"
        );
        assert_eq!(before - after, before / 2);

        // Mutation half: a tensor the constructor never reads is NOT drained.
        let mut view = s.open().unwrap();
        view.insert("encoder.layers.0.unread_by_the_block", Array::from_f32(1.0));
        s.materialize(&mut view, 0).unwrap();
        assert!(
            view.get("encoder.layers.0.unread_by_the_block").is_some(),
            "an un-read key must survive an exact drain"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// A fresh view per window is what keeps the bound: two windows must not share a map that
    /// accumulates.
    #[test]
    fn each_window_opens_an_independent_view() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "reopen", 2);
        let s = stream(&path, 2);
        let mut first = s.open().unwrap();
        let full = first.len();
        s.materialize(&mut first, 0).unwrap();
        assert!(first.len() < full);
        let second = s.open().unwrap();
        assert_eq!(second.len(), full, "a new window must see the whole stack");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn out_of_range_blocks_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "range", 2);
        let s = stream(&path, 2);
        let mut view = s.open().unwrap();
        let err = match s.materialize(&mut view, 2) {
            Ok(_) => panic!("block 2 of a 2-block stack must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("out of range"), "{err}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// A materialized block replays the recorded quantization. "Is quantized" alone would pass on a
    /// different tier, so the streamed block is compared against a RESIDENT block packed the same
    /// way — the property that actually matters.
    #[test]
    fn a_materialized_block_replays_the_recorded_quantization() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "quant", 1);
        let mut s = stream(&path, 1);
        let mut view = s.open().unwrap();
        let dense = s.materialize(&mut view, 0).unwrap();
        assert!(!dense.is_quantized(), "the dense tier must stay dense");

        s.quant_bits = Some(4);
        let mut view = s.open().unwrap();
        let packed = s.materialize(&mut view, 0).unwrap();
        assert!(
            packed.is_quantized(),
            "the replay must pack every projection"
        );

        let view = s.open().unwrap();
        let mut twin = GlmBlock::from_weights(&view, 0, None, Dtype::Float32).unwrap();
        twin.quantize(4).unwrap();
        assert!(
            packed.matches_quantized_params(&twin),
            "codes / scales / biases / group / bits must match the resident block quantized the \
             same way"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// The stream's block count is the config's layer count. A stream that under-reported it would
    /// silently window a prefix of the tower and leave the tail unrun.
    #[test]
    fn the_stack_depth_is_the_configured_layer_count() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(ChatGlmConfig::chatglm3_6b().num_layers, 28);
        let path = fixture(&tmp, "depth", 2);
        assert_eq!(stream(&path, 2).n_blocks(), 2);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
