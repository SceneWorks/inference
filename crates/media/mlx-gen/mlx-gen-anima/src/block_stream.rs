//! Ladder rung 4 — **bounded transformer residency** for the Anima Cosmos-Predict2 DiT (SC-15524).
//!
//! The window lifecycle is NOT implemented here: it is
//! [`mlx_gen::block_residency::run_windowed`] (SC-15750), the shared primitive Z-Image, Qwen and
//! Mage-Flow also drive. This module is only the family-side half — "how do I rebuild Anima block
//! *n* from the checkpoint, carrying the same tier and the same adapters the resident block
//! carries" — which is the part that genuinely differs per family.
//!
//! ## Why re-reading per window is nearly free here too
//!
//! [`Weights::from_file`] goes through `Array::load_safetensors_with_metadata`, which is **lazy per
//! tensor**: the handles exist, the bytes do not, until something is evaluated. So re-opening the
//! 4.18 GB `anima-{variant}-v1.0.safetensors` once per window costs a header parse, not 4.18 GB.
//! Only the tensors a window's blocks actually read are materialized, and
//! [`Weights::remove_accessed`] then drops the view's own reference to exactly those, so a window's
//! residency is the blocks themselves and nothing else.
//!
//! `remove_accessed` is load-bearing rather than decorative: `Array` is refcounted, and
//! [`crate::transformer::Block::from_weights`] **clones** out of the view (it is the same
//! constructor the resident path uses — forking it would be a second way to build a block, exactly
//! the divergence hazard this epic exists to remove). Draining after each block is what turns the
//! window's drop into a real release. The drain is *exact* — `require` records every key the
//! constructor read and only those are removed — so a constructor read that never happened stays
//! visible instead of being wiped along with the rest.
//!
//! ## The Anima-specific wrinkle: one file, two models
//!
//! Anima's DiT checkpoint bundles BOTH the Cosmos DiT (`{prefix}.blocks.*`, `{prefix}.x_embedder.*`,
//! …) and the `AnimaTextConditioner` (`{prefix}.llm_adapter.*`). The conditioner is *not* windowed:
//! it is 6 small blocks that run once per prompt in the conditioning phase, not 28 blocks run once
//! per denoise step. Windowing the DiT out of the same file is unaffected — the stream addresses
//! `{prefix}.blocks.{n}` and never touches `llm_adapter.*` — but it is why the stream records the
//! detected root `prefix` rather than assuming `net` (turbo/aesthetic use
//! `model.diffusion_model`).
//!
//! ## Adapter replay
//!
//! A window rebuilt from the checkpoint would silently drop the LoRA/LoKr pairs installed at load —
//! identity-free output, no error. Anima installs every adapter as a **forward-time residual**
//! (`apply_adapters_strict` only ever `push`es an [`Adapter`]; it never folds a delta into a base),
//! so replaying the captured per-block list reproduces the resident block exactly rather than
//! approximating it. [`Adapter`] holds refcounted `Array` factors, so the capture is a handle copy —
//! a rank-16 LoRA's factors, not an `[out, in]` delta.
//!
//! The conditioner's 60 `llm_adapter.*` targets are untouched by this: they live on the resident
//! `AnimaTextConditioner`, which rung 4 never sheds.

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::config::DitConfig;
use crate::transformer::Block;

/// The adapters installed on ONE block, addressed by the block-local dotted path
/// (`"self_attn.q_proj"`, `"mlp.layer1"`, `"adaln_modulation_mlp.2"`, …).
///
/// Captured from the resident blocks after [`crate::adapters::apply_anima_adapters`] has run, so the
/// streamed block replays exactly what the resident block ended up holding — not a second derivation
/// of which adapters should have landed where.
#[derive(Clone, Default)]
pub(crate) struct BlockAdapters {
    per_path: Vec<(String, Vec<Adapter>)>,
}

impl BlockAdapters {
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.per_path.is_empty()
    }

    fn install(&self, block: &mut Block) -> Result<()> {
        for (path, adapters) in &self.per_path {
            let segs: Vec<&str> = path.split('.').collect();
            let target = block.adaptable_mut(&segs).ok_or_else(|| {
                Error::Msg(format!(
                    "anima block stream: adapter target `{path}` is absent from a materialized \
                     block — the streamed block tree does not match the resident one"
                ))
            })?;
            target.set_adapters(adapters.clone());
        }
        Ok(())
    }
}

/// Everything needed to rebuild one Anima DiT block from its checkpoint, on demand.
///
/// Held by [`CosmosDiT`](crate::CosmosDiT) and consumed by the windowed forward. Cheap to clone: a
/// path, a prefix, a `Copy` config and refcounted adapter factors.
#[derive(Clone)]
pub(crate) struct AnimaBlockStream {
    /// The re-openable checkpoint. Anima resolves this to the variant's exact
    /// `diffusion_models/anima-{variant}-v1.0.safetensors`, so it is a `File` on every registry
    /// load — never the caller's raw `LoadSpec::weights`, which may be the `split_files/` root.
    source: WeightsSource,
    /// Key prefix for the stack, already joined with the detected checkpoint root, so block `n` is
    /// `{base}.{n}` (`net.blocks.3`, `model.diffusion_model.blocks.3`, …).
    base: String,
    cfg: DitConfig,
    n_blocks: usize,
    /// Per-block captured adapters, indexed by block. Empty when no adapter is installed.
    adapters: Vec<BlockAdapters>,
}

impl AnimaBlockStream {
    /// Declare a streamable stack. `prefix` is the detected DiT root (`net` /
    /// `model.diffusion_model`) — the same prefix [`crate::transformer::CosmosDiT::from_weights`]
    /// read the resident stack under, so a streamed block reads the identical keys.
    pub(crate) fn new(source: WeightsSource, prefix: &str, cfg: DitConfig) -> Self {
        let base = mlx_gen::weights::join(prefix, "blocks");
        Self {
            source,
            base,
            cfg,
            n_blocks: cfg.num_layers,
            adapters: vec![BlockAdapters::default(); cfg.num_layers],
        }
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    #[cfg(test)]
    pub(crate) fn has_adapters(&self) -> bool {
        self.adapters.iter().any(|a| !a.is_empty())
    }

    /// Capture the adapters currently installed on `blocks` so materialized blocks carry them too.
    pub(crate) fn capture_adapters(&mut self, blocks: &mut [Block]) {
        self.adapters = blocks
            .iter_mut()
            .map(|block| {
                let paths = block.adaptable_paths();
                let mut per_path = Vec::new();
                for path in paths {
                    let segs: Vec<&str> = path.split('.').collect();
                    // SC-18319 — walk every path through the PROBE half and take the `&mut` only for
                    // the paths that actually hold something. This walk visits EVERY projection in
                    // the block, so resolving it `&mut`-first would unfuse every
                    // `FusedQkvProjection` in the stack during a capture that copies nothing.
                    if block
                        .adaptable_facts(&segs)
                        .is_some_and(|f| f.adapter_count > 0)
                    {
                        let adapters = block
                            .adaptable_mut(&segs)
                            .expect("resolved through the probe above")
                            .adapters()
                            .to_vec();
                        per_path.push((path.clone(), adapters));
                    }
                }
                BlockAdapters { per_path }
            })
            .collect();
    }

    /// Open a fresh lazy view of the checkpoint. Called once per window by
    /// [`run_windowed`](mlx_gen::block_residency::run_windowed).
    pub(crate) fn open(&self) -> Result<Weights> {
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    /// Materialize block `index` out of `view`, adapted exactly like its resident twin, then drain
    /// the view of precisely the tensors that block read.
    pub(crate) fn materialize(&self, view: &mut Weights, index: usize) -> Result<Block> {
        if index >= self.n_blocks {
            return Err(Error::Msg(format!(
                "anima block stream: block {index} is out of range for a {}-block stack",
                self.n_blocks
            )));
        }
        let prefix = format!("{}.{index}", self.base);
        let mut block = Block::from_weights(view, &prefix, &self.cfg)?;
        // LOAD-BEARING (SC-15750): the view keeps its own refcounted handle to every tensor the
        // constructor cloned. Draining exactly the accessed keys is what makes the window's drop a
        // real release rather than a no-op that still produces correct images.
        view.remove_accessed();
        if let Some(adapters) = self.adapters.get(index) {
            adapters.install(&mut block)?;
        }
        Ok(block)
    }
}

/// The plan + cancellation a windowed forward runs under. A thin `Copy` pair so the forward
/// signatures do not grow two more positional arguments.
#[derive(Clone, Copy)]
pub(crate) struct BlockWindow<'a> {
    pub(crate) plan: mlx_gen::block_residency::BlockPlan,
    pub(crate) cancel: &'a mlx_gen::CancelFlag,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    /// A tiny two-block Anima stack written to a temp `.safetensors`, so the stream can be exercised
    /// without the licensed 4.18 GB checkpoint.
    fn cfg() -> DitConfig {
        DitConfig {
            num_attention_heads: 2,
            attention_head_dim: 4,
            num_layers: 2,
            text_embed_dim: 8,
            adaln_lora_dim: 4,
            ..DitConfig::anima()
        }
    }

    fn fixture(
        tmp: &tempfile::TempDir,
        tag: &str,
        prefix: &str,
        cfg: DitConfig,
    ) -> std::path::PathBuf {
        let hidden = cfg.hidden_size() as i32;
        let adaln = cfg.adaln_lora_dim as i32;
        let ctx = cfg.text_embed_dim as i32;
        let head = cfg.attention_head_dim as i32;
        let ff = (hidden as f32 * cfg.mlp_ratio) as i32;
        let dir = tmp.path().join(format!("anima-block-stream-{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let mut named: Vec<(String, Array)> = Vec::new();
        let mut push = |name: String, shape: Vec<i32>, salt: f32| {
            let n: i32 = shape.iter().product();
            let values: Vec<f32> = (0..n).map(|k| (k as f32 + salt) * 0.01).collect();
            named.push((name, Array::from_slice(&values, &shape)));
        };
        for i in 0..cfg.num_layers {
            let salt = i as f32;
            for attn in ["self_attn", "cross_attn"] {
                let kv_in = if attn == "cross_attn" { ctx } else { hidden };
                push(
                    format!("{prefix}.blocks.{i}.{attn}.q_proj.weight"),
                    vec![hidden, hidden],
                    salt,
                );
                push(
                    format!("{prefix}.blocks.{i}.{attn}.k_proj.weight"),
                    vec![hidden, kv_in],
                    salt,
                );
                push(
                    format!("{prefix}.blocks.{i}.{attn}.v_proj.weight"),
                    vec![hidden, kv_in],
                    salt,
                );
                push(
                    format!("{prefix}.blocks.{i}.{attn}.output_proj.weight"),
                    vec![hidden, hidden],
                    salt,
                );
                push(
                    format!("{prefix}.blocks.{i}.{attn}.q_norm.weight"),
                    vec![head],
                    salt,
                );
                push(
                    format!("{prefix}.blocks.{i}.{attn}.k_norm.weight"),
                    vec![head],
                    salt,
                );
            }
            push(
                format!("{prefix}.blocks.{i}.mlp.layer1.weight"),
                vec![ff, hidden],
                salt,
            );
            push(
                format!("{prefix}.blocks.{i}.mlp.layer2.weight"),
                vec![hidden, ff],
                salt,
            );
            for norm in [
                "adaln_modulation_self_attn",
                "adaln_modulation_cross_attn",
                "adaln_modulation_mlp",
            ] {
                push(
                    format!("{prefix}.blocks.{i}.{norm}.1.weight"),
                    vec![adaln, hidden],
                    salt,
                );
                push(
                    format!("{prefix}.blocks.{i}.{norm}.2.weight"),
                    vec![3 * hidden, adaln],
                    salt,
                );
            }
        }
        let refs: Vec<(&str, &Array)> = named.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        path
    }

    fn stream(path: &std::path::Path, prefix: &str) -> AnimaBlockStream {
        AnimaBlockStream::new(WeightsSource::File(path.to_path_buf()), prefix, cfg())
    }

    /// The drain must remove EXACTLY the block's own tensors — not a whole prefix.
    ///
    /// `remove_prefix` would drain the same bytes here, but would also hide a constructor that
    /// silently stopped reading a tensor: the block would be built from a stale default and the wipe
    /// would erase the evidence. Exact draining leaves an un-read key visible, which is what the
    /// second half asserts.
    #[test]
    fn block_stream_drains_exactly_what_the_block_read() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "drain", "net", cfg());
        let stream = stream(&path, "net");
        let mut view = stream.open().unwrap();
        let before = view.len();
        stream.materialize(&mut view, 0).unwrap();
        let after = view.len();
        assert!(
            after < before,
            "materializing block 0 must drain its tensors"
        );
        assert!(view.get("net.blocks.1.self_attn.q_proj.weight").is_some());
        assert!(view.get("net.blocks.0.self_attn.q_proj.weight").is_none());
        assert_eq!(
            view.keys()
                .filter(|k| k.starts_with("net.blocks.0."))
                .count(),
            0,
            "every block-0 tensor the constructor read must have been drained"
        );
        assert_eq!(before - after, before / 2);

        // Mutation half: a key the constructor never reads is NOT drained, so a constructor that
        // quietly drops a read stays observable instead of being wiped along with the rest.
        let mut view = stream.open().unwrap();
        view.insert("net.blocks.0.unread_by_the_block", Array::from_f32(1.0));
        stream.materialize(&mut view, 0).unwrap();
        assert!(
            view.get("net.blocks.0.unread_by_the_block").is_some(),
            "an un-read key must survive an exact drain"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// A fresh view per window is what keeps the bound: two windows must not share a map that
    /// accumulates.
    #[test]
    fn each_window_opens_an_independent_view() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "reopen", "net", cfg());
        let stream = stream(&path, "net");
        let mut first = stream.open().unwrap();
        let full = first.len();
        stream.materialize(&mut first, 0).unwrap();
        assert!(first.len() < full);
        assert_eq!(
            stream.open().unwrap().len(),
            full,
            "a new window must see the whole stack"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// Turbo and aesthetic checkpoints root the DiT at `model.diffusion_model`, not `net`. The
    /// stream records the detected prefix, so a materialized block reads the same keys the resident
    /// load did on EVERY variant — a hardcoded `net` would work on base and silently fail on the
    /// other two.
    #[test]
    fn the_detected_checkpoint_prefix_is_carried_into_the_stream() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "prefix", "model.diffusion_model", cfg());
        let stream = stream(&path, "model.diffusion_model");
        let mut view = stream.open().unwrap();
        stream.materialize(&mut view, 1).unwrap();
        assert!(view
            .get("model.diffusion_model.blocks.1.mlp.layer1.weight")
            .is_none());

        // And the wrong prefix is a typed error, not a silently wrong block.
        let wrong = super::AnimaBlockStream::new(
            WeightsSource::File(path.clone()),
            "net",
            super::super::config::DitConfig {
                num_layers: 2,
                ..cfg()
            },
        );
        let mut view = wrong.open().unwrap();
        assert!(wrong.materialize(&mut view, 0).is_err());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn out_of_range_blocks_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "range", "net", cfg());
        let stream = stream(&path, "net");
        let mut view = stream.open().unwrap();
        let err = match stream.materialize(&mut view, 2) {
            Ok(_) => panic!("block 2 of a 2-block stack must be rejected"),
            Err(error) => error.to_string(),
        };
        assert!(err.contains("out of range"), "{err}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// Captured adapters travel onto a materialized block, per block.
    ///
    /// This is the SC-15524 restatement of the adapter-replay discipline: without it a windowed
    /// render silently drops every LoRA — identity-free output and no error — which is precisely the
    /// failure mode `anima-turbo-lora-v0.2` (508 targets) would exhibit as "the turbo LoRA stopped
    /// working" with nothing in the logs.
    #[test]
    fn captured_adapters_are_reinstalled_per_block() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "adapters", "net", cfg());
        let mut stream = stream(&path, "net");
        let view = stream.open().unwrap();
        let mut resident: Vec<Block> = (0..2)
            .map(|i| Block::from_weights(&view, &format!("net.blocks.{i}"), &cfg()).unwrap())
            .collect();
        let hidden = cfg().hidden_size() as i32;
        let a = Array::from_slice(&vec![0.5_f32; (hidden * 2) as usize], &[hidden, 2]);
        let b = Array::from_slice(&vec![0.25_f32; (2 * hidden) as usize], &[2, hidden]);
        resident[1]
            .adaptable_mut(&["self_attn", "q_proj"])
            .unwrap()
            .push(Adapter::Lora { a, b, scale: 1.0 });
        stream.capture_adapters(&mut resident);
        assert!(stream.has_adapters());

        let mut view = stream.open().unwrap();
        let mut zero = stream.materialize(&mut view, 0).unwrap();
        let mut view = stream.open().unwrap();
        let mut one = stream.materialize(&mut view, 1).unwrap();
        assert!(zero
            .adaptable_mut(&["self_attn", "q_proj"])
            .unwrap()
            .adapters()
            .is_empty());
        assert_eq!(
            one.adaptable_mut(&["self_attn", "q_proj"])
                .unwrap()
                .adapters()
                .len(),
            1
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
