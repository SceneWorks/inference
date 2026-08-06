//! Ladder rung 4 — **bounded transformer residency** for the SANA Linear-DiT trunk (SC-15523).
//!
//! The window lifecycle is NOT implemented here: it is
//! [`mlx_gen::block_residency::run_windowed`] (SC-15750, hoisted to
//! [`gen_core::block_window`](mlx_gen::gen_core::block_window)), the shared primitive that also
//! serves Z-Image / Chroma / SDXL / Krea. This module is only the family-side half — "how do I
//! rebuild SANA block *n* from the snapshot so it is the same block the resident stack holds".
//!
//! ## Why re-reading per window is nearly free
//!
//! `Array::load_safetensors` is **lazy per tensor**: opening the snapshot once per window costs a
//! JSON header parse, not 2 GB. Only the tensors a window's blocks actually read are materialized,
//! and [`Weights::remove_accessed`] then drops the view's own reference to exactly those, so the
//! window's residency is the blocks themselves and nothing else.
//!
//! `remove_accessed` is deliberate rather than decorative. The shared primitive documents that
//! `apply` must take tensors OUT of the view instead of cloning them, because `Array` is refcounted
//! and a clone left in the map keeps the materialized buffer alive. SANA's block constructor takes
//! `&Weights` and clones (it is the same constructor the resident path uses, and forking it would be
//! a second way to build a block — precisely the divergence hazard this epic exists to remove), so
//! the drain runs immediately after each block instead.
//!
//! ## What SANA does NOT have to replay, and why that is checked rather than asserted in prose
//!
//! Z-Image replays a recorded load-time quantization and a captured adapter list onto every
//! materialized block. SANA has neither:
//!
//! - **Quantization** is *packed-detected from disk* ([`crate::quant::lin`] keys off the on-disk
//!   `{base}.scales`), so a block rebuilt from the same file is packed identically by construction.
//! - **Adapters** are refused at load (`mlx-gen-sana` advertises `supports_lora: false` and
//!   `load_components` rejects a non-empty `spec.adapters`), so there is no forward-time residual to
//!   capture.
//!
//! What remains is the class those two mechanisms exist to prevent, and it is *not* absent here: a
//! streamed block built from a **different config** than its resident twin silently drops the
//! tensors that config gates. SANA-Sprint's `qk_norm = "rms_norm_across_heads"` is exactly such a
//! gate — `attn1.norm_q/k` and `attn2.norm_q/k` are read only when
//! [`SanaTransformerConfig::qk_norm`] is set, so a base-config stream over Sprint weights would
//! produce a block that is *present but wrong* rather than one that fails loudly. The stream
//! therefore carries the trunk's own config, and
//! `a_base_config_stream_over_sprint_weights_is_present_but_wrong` pins that it is observable.

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::SanaTransformerConfig;
use crate::transformer::SanaBlock;

/// Everything needed to rebuild one SANA transformer block from its snapshot, on demand.
///
/// Held by [`SanaTransformer`](crate::transformer::SanaTransformer) and consumed by the windowed
/// forward. Cheap to clone: a path, a config, and a count.
#[derive(Clone)]
pub(crate) struct SanaBlockStream {
    /// The re-openable `transformer/` directory. An in-memory load has no such source and simply
    /// does not construct a stream.
    source: PathBuf,
    cfg: SanaTransformerConfig,
    n_blocks: usize,
}

impl SanaBlockStream {
    /// Declare a streamable stack rooted at the snapshot's `transformer/` directory.
    pub(crate) fn new(source: impl AsRef<Path>, cfg: SanaTransformerConfig) -> Self {
        let n_blocks = cfg.num_layers.max(0) as usize;
        Self {
            source: source.as_ref().to_path_buf(),
            cfg,
            n_blocks,
        }
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Open a fresh lazy view of the trunk's weights. Called once per window by
    /// [`run_windowed`](mlx_gen::block_residency::run_windowed).
    ///
    /// It must be FRESH: a view retained across windows keeps every materialized buffer alive
    /// through its own map, so the release frees nothing.
    pub(crate) fn open(&self) -> Result<Weights> {
        Weights::from_dir(&self.source)
    }

    /// Materialize block `index` out of `view`, then drain the view of precisely the tensors that
    /// block read.
    pub(crate) fn materialize(&self, view: &mut Weights, index: usize) -> Result<SanaBlock> {
        if index >= self.n_blocks {
            return Err(Error::Msg(format!(
                "sana block stream: block {index} is out of range for a {}-block stack",
                self.n_blocks
            )));
        }
        let block = SanaBlock::load(view, &format!("transformer_blocks.{index}"), &self.cfg)?;
        // LOAD-BEARING (SC-15750): the view keeps its own refcounted handle to every tensor the
        // constructor cloned. Draining exactly the accessed keys is what makes the window's drop a
        // real release rather than a no-op that still produces correct images.
        view.remove_accessed();
        Ok(block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    /// A trunk config narrowed to a tiny synthetic block, so the stream can be exercised without a
    /// real snapshot. `qk_norm` selects the Sprint gate.
    fn cfg(qk_norm: bool) -> SanaTransformerConfig {
        SanaTransformerConfig {
            num_attention_heads: 2,
            attention_head_dim: 4, // inner_dim = 8
            num_layers: 2,
            num_cross_attention_heads: 2,
            cross_attention_head_dim: 4,
            caption_channels: 8,
            mlp_ratio: 1.0,
            qk_norm,
            ..SanaTransformerConfig::sana_1600m()
        }
    }

    fn write_fixture(tag: &str, cfg: &SanaTransformerConfig) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("sana-block-stream-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let inner = cfg.inner_dim();
        let hidden = (cfg.mlp_ratio * inner as f32) as i32;
        let mut named: Vec<(String, Array)> = Vec::new();
        let mut push = |name: String, shape: Vec<i32>, seed: f32| {
            let n: i32 = shape.iter().product();
            let values: Vec<f32> = (0..n).map(|k| (k as f32 * 0.017 + seed).sin()).collect();
            named.push((name, Array::from_slice(&values, &shape)));
        };
        for i in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{i}");
            let s = i as f32;
            push(format!("{p}.scale_shift_table"), vec![6, inner], s);
            for leaf in ["to_q", "to_k", "to_v"] {
                push(format!("{p}.attn1.{leaf}.weight"), vec![inner, inner], s);
            }
            push(format!("{p}.attn1.to_out.0.weight"), vec![inner, inner], s);
            push(format!("{p}.attn1.to_out.0.bias"), vec![inner], s);
            for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                push(format!("{p}.attn2.{leaf}.weight"), vec![inner, inner], s);
                push(format!("{p}.attn2.{leaf}.bias"), vec![inner], s);
            }
            if cfg.qk_norm {
                for attn in ["attn1", "attn2"] {
                    for leaf in ["norm_q", "norm_k"] {
                        push(format!("{p}.{attn}.{leaf}.weight"), vec![inner], s + 0.5);
                    }
                }
            }
            push(
                format!("{p}.ff.conv_inverted.weight"),
                vec![2 * hidden, inner, 1, 1],
                s,
            );
            push(format!("{p}.ff.conv_inverted.bias"), vec![2 * hidden], s);
            push(
                format!("{p}.ff.conv_depth.weight"),
                vec![2 * hidden, 1, 3, 3],
                s,
            );
            push(format!("{p}.ff.conv_depth.bias"), vec![2 * hidden], s);
            push(
                format!("{p}.ff.conv_point.weight"),
                vec![inner, hidden, 1, 1],
                s,
            );
        }
        let refs: Vec<(&str, &Array)> = named.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, dir.join("model.safetensors")).unwrap();
        dir
    }

    fn inputs(cfg: &SanaTransformerConfig) -> (Array, Array, Array) {
        let inner = cfg.inner_dim();
        let n = 4;
        let m = 3;
        let fill = |count: i32, seed: f32| {
            Array::from_slice(
                &(0..count)
                    .map(|k| (k as f32 * 0.031 + seed).cos())
                    .collect::<Vec<f32>>(),
                &[count],
            )
        };
        (
            fill(n * inner, 0.1).reshape(&[1, n, inner]).unwrap(),
            fill(m * inner, 0.2).reshape(&[1, m, inner]).unwrap(),
            fill(6 * inner, 0.3).reshape(&[1, 6 * inner]).unwrap(),
        )
    }

    fn run(block: &SanaBlock, cfg: &SanaTransformerConfig) -> Vec<f32> {
        let (hidden, caption, temb) = inputs(cfg);
        let out = block
            .forward(
                &hidden,
                &caption,
                None,
                &temb,
                2,
                2,
                mlx_gen::attention::AttentionBudget::UNBOUNDED,
            )
            .unwrap();
        out.as_slice::<f32>().to_vec()
    }

    /// The drain must remove EXACTLY the block's own tensors — not a whole prefix.
    ///
    /// The distinction is load-bearing: `remove_prefix("transformer_blocks.0.")` would drain the
    /// same bytes here but would also hide a constructor that silently stopped reading a tensor.
    /// Exact draining leaves an un-read key visible, which is what the second half asserts.
    #[test]
    fn block_stream_drains_exactly_what_the_block_read() {
        let cfg = cfg(false);
        let dir = write_fixture("drain", &cfg);
        let stream = SanaBlockStream::new(&dir, cfg.clone());
        let mut view = stream.open().unwrap();
        let before = view.len();
        stream.materialize(&mut view, 0).unwrap();
        let after = view.len();
        assert!(
            after < before,
            "materializing block 0 must drain its tensors"
        );
        assert!(view.get("transformer_blocks.1.attn1.to_q.weight").is_some());
        assert!(view.get("transformer_blocks.0.attn1.to_q.weight").is_none());
        assert_eq!(
            view.keys()
                .filter(|k| k.starts_with("transformer_blocks.0."))
                .count(),
            0,
            "every block-0 tensor the constructor read must have been drained"
        );
        assert_eq!(before - after, before / 2);

        // Mutation half: a tensor the constructor never reads is NOT drained, so a constructor that
        // quietly drops a read stays observable instead of being wiped along with the rest.
        let mut view = stream.open().unwrap();
        view.insert(
            "transformer_blocks.0.unread_by_the_block",
            Array::from_f32(1.0),
        );
        stream.materialize(&mut view, 0).unwrap();
        assert!(
            view.get("transformer_blocks.0.unread_by_the_block")
                .is_some(),
            "an un-read key must survive an exact drain"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// A fresh view per window is what keeps the bound: re-opening returns the full key set.
    #[test]
    fn each_window_opens_an_independent_view() {
        let cfg = cfg(false);
        let dir = write_fixture("reopen", &cfg);
        let stream = SanaBlockStream::new(&dir, cfg);
        let mut first = stream.open().unwrap();
        let full = first.len();
        stream.materialize(&mut first, 0).unwrap();
        assert!(first.len() < full);
        assert_eq!(
            stream.open().unwrap().len(),
            full,
            "a new window must see the whole stack"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// A streamed block is the resident block: same constructor, same tensors, same arithmetic.
    #[test]
    fn a_materialized_block_matches_its_resident_twin() {
        for qk_norm in [false, true] {
            let cfg = cfg(qk_norm);
            let dir = write_fixture(&format!("twin-{qk_norm}"), &cfg);
            let stream = SanaBlockStream::new(&dir, cfg.clone());
            let view = stream.open().unwrap();
            let resident = SanaBlock::load(&view, "transformer_blocks.1", &cfg).unwrap();
            let mut windowed_view = stream.open().unwrap();
            let streamed = stream.materialize(&mut windowed_view, 1).unwrap();
            assert_eq!(
                run(&resident, &cfg),
                run(&streamed, &cfg),
                "a streamed block must reproduce its resident twin exactly (qk_norm={qk_norm})"
            );
            std::fs::remove_dir_all(dir).ok();
        }
    }

    /// **The present-but-wrong class.** A stream carrying the wrong config over Sprint weights
    /// silently skips the `qk_norm` RMSNorm gates: every tensor it *does* read is present, nothing
    /// fails, and the block is wrong. This asserts the difference is observable, which is what makes
    /// carrying the trunk's own config load-bearing rather than decorative.
    #[test]
    fn a_base_config_stream_over_sprint_weights_is_present_but_wrong() {
        let sprint = cfg(true);
        let dir = write_fixture("present-but-wrong", &sprint);

        let sprint_stream = SanaBlockStream::new(&dir, sprint.clone());
        let mut view = sprint_stream.open().unwrap();
        let correct = run(&sprint_stream.materialize(&mut view, 0).unwrap(), &sprint);

        let base_stream = SanaBlockStream::new(&dir, cfg(false));
        let mut view = base_stream.open().unwrap();
        let wrong = run(&base_stream.materialize(&mut view, 0).unwrap(), &sprint);

        assert_ne!(
            correct, wrong,
            "a base-config stream over Sprint weights must not silently reproduce the Sprint block"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// A MISSING tensor fails loudly rather than defaulting.
    #[test]
    fn a_missing_tensor_fails_loudly() {
        let sprint = cfg(true);
        let dir = write_fixture("missing", &sprint);
        // A Sprint stream over BASE weights requires `norm_q`/`norm_k`, which a base fixture lacks.
        let base_dir = write_fixture("missing-base", &cfg(false));
        let stream = SanaBlockStream::new(&base_dir, sprint);
        let mut view = stream.open().unwrap();
        let error = match stream.materialize(&mut view, 0) {
            Ok(_) => panic!("a Sprint stream over base weights must fail"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("norm_q"),
            "the failure must name the missing tensor, got: {error}"
        );
        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_dir_all(base_dir).ok();
    }

    #[test]
    fn out_of_range_blocks_are_rejected() {
        let cfg = cfg(false);
        let dir = write_fixture("range", &cfg);
        let stream = SanaBlockStream::new(&dir, cfg);
        let mut view = stream.open().unwrap();
        let error = match stream.materialize(&mut view, 2) {
            Ok(_) => panic!("block 2 of a 2-block stack must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("out of range"), "{error}");
        std::fs::remove_dir_all(dir).ok();
    }
}
