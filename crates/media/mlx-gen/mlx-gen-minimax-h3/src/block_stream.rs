//! Rung 4 — **bounded transformer residency** for MiniMax-H3 on MLX (sc-18662).
//!
//! Window scheduling, loop order, release discipline and the cancellation contract stay in
//! [`gen_core::block_window`](mlx_gen::gen_core::block_window) /
//! [`mlx_gen::block_residency::run_windowed`]. What lives here is the
//! family's own half: how a window of MiniMax-H3 blocks is reconstructed from a staged component
//! directory, and the two places this model differs from every image family that has adopted the
//! rung before it.
//!
//! # Difference 1 — the AdaLN projection must NOT be in the window
//!
//! `adaln_proj` is **39.3 % of the DiT's bytes** ([`crate::memory_strategy::ADALN_EVICTED_BYTES`]
//! against [`crate::memory_strategy::DIT_BF16_BYTES`]) and sc-17145's precompute already consumes it
//! exactly once per *request*, keeping a modulation table in its place. A window that re-read it
//! would re-materialize 26.02 GB per step for a table already held — and produce **bit-identical
//! output**, so no correctness assertion anywhere could see it.
//!
//! [`DitBlock::from_weights_body_only`] is therefore what a denoise window materializes: ten tensors
//! per block, not twelve. `window_bytes_body_only_is_the_residency_bound` (this module's own unit
//! tests) pins the difference in
//! bytes rather than trusting the loader's shape.
//!
//! # Difference 2 — the precompute itself has to be windowed
//!
//! Every prior adopter's rung-4 trunk is *only* walked per step. This one is walked twice with
//! different residency needs: once per request over `adaln_proj` (the precompute) and once per step
//! over the block bodies (the denoise). A deferred load that windowed only the second would still
//! peak at the whole 50-block projection stack during the first, which is the larger of the two.
//!
//! [`precompute_adaln_windowed`] runs the projection pass through the same shared driver, holding
//! `window` projections at a time and forcing each window's tables before it releases. The eval is
//! load-bearing for the reason [`gen_core::block_window`](mlx_gen::gen_core::block_window) gives:
//! MLX is lazy, and an unevaluated
//! table still references the projection it came from.
//!
//! # The per-window cost obligation
//!
//! [`gen_core::memory_strategy::MemoryWindowMaterialization`](mlx_gen::gen_core::memory_strategy::MemoryWindowMaterialization)
//! requires a window to be a transfer of bytes already in the accelerator's form. This satisfies it structurally, and by construction
//! rather than by assertion: [`Weights::from_dir`] hands out lazily-mapped MLX handles, and the two
//! loaders a window calls — [`crate::quant::lin`] and `as_dtype` — perform **no conversion** on the
//! staged tier. A packed `q4`/`q8` base is used in its published affine form (`cast_weights` no-ops
//! on one, deliberately), and a `bf16` tier is already at the DiT's compute dtype so the cast is an
//! identity. There is no repack in this path, which is the ~26x defect SC-15791 measured on the
//! Candle side.
//!
//! # What is NOT windowed, and why that is arithmetic rather than omission
//!
//! The 17 input/output projections (`crate::dit::heads`) and the 21-tensor token refiner stay
//! resident across the whole request. Together they are 38 of the DiT's 638 tensors; the 600 block
//! tensors this module streams are the residency. The refiner additionally runs **once per request**
//! rather than once per step, so windowing it would pay a re-materialization to bound bytes that a
//! single window already bounds.
//!
//! # Adapters ride along with every rebuilt block (SW-7096)
//!
//! A window rebuilds each block from the staged tier, so a forward-time residual installed on a
//! resident block would be lost. [`DitBlockStream`] therefore owns a [`StreamedBlockAdapters`]
//! overlay: the strict install ([`crate::adapters::apply_minimax_h3_adapters`]) folds every
//! `transformer_blocks.{i}.…` factor into a shape-only slot instead of into a block, and
//! [`DitBlockStream::materialize`] pushes that slot's residuals onto the freshly rebuilt block
//! before it runs. The residual form is the same one the resident path installs —
//! `base(x) + Σ adapter.residual(x)` over an untouched (dense or packed) base — so the streamed
//! output with a LoRA is the resident output with that LoRA, on every tier.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::block_residency::{run_windowed, BlockPlan};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result};

use crate::dit::adaln::{AdaLnCache, TimestepSchedule};
use crate::dit::block::{AdaLnModulation, AdaLnProjection, DitBlock};
use crate::dit::config::MiniMaxH3DitConfig;
use crate::text_encoder::{MiniMaxH3TeConfig, Qwen3DecoderLayer};

/// Reopenable description of a staged MiniMax-H3 DiT's block stack.
///
/// Holds a path and a dtype and nothing else — deliberately no [`Weights`] handle. A view retained
/// across windows keeps every materialized buffer alive through its own map, so the release frees
/// nothing;
/// [`gen_core::block_window::BlockWindowBackend::open_view`](mlx_gen::gen_core::block_window::BlockWindowBackend::open_view)
/// says so, and `a_retained_view_defeats_the_bound` would be the test that catches it if this
/// struct grew one.
#[derive(Clone, Debug)]
pub struct DitBlockStream {
    dir: PathBuf,
    dtype: Dtype,
    cfg: MiniMaxH3DitConfig,
    /// The configured adapters' residuals for the streamed blocks, re-applied to each block as a
    /// window rebuilds it. Empty on an un-adapted render.
    adapters: StreamedBlockAdapters,
}

/// The adapter residuals of a streamed block stack, keyed by `(block, leaf)` (SW-7096).
///
/// Each slot is an [`AdaptableLinear`] over a **shape-only** base: a lazy zero array of the
/// leaf's logical `[out, in]` shape that is never evaluated, so it costs no memory. The shape is
/// what the shared LoKr install reads (`AdaptableLinear::base_shape`); the forward never runs on a
/// slot. Only the residuals pushed onto a slot are used — [`DitBlockStream::materialize`] copies
/// them onto the rebuilt block's matching leaf.
#[derive(Clone, Default)]
pub struct StreamedBlockAdapters {
    slots: BTreeMap<(usize, &'static str), AdaptableLinear>,
}

impl std::fmt::Debug for StreamedBlockAdapters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamedBlockAdapters")
            .field("slots", &self.slots.len())
            .field("residuals", &self.residual_count())
            .finish()
    }
}

impl StreamedBlockAdapters {
    /// Residuals held across every slot.
    pub fn residual_count(&self) -> usize {
        self.slots.values().map(|slot| slot.adapters().len()).sum()
    }

    /// Whether no residual is held.
    pub fn is_empty(&self) -> bool {
        self.residual_count() == 0
    }

    /// Evaluate every held residual's own arrays (never a base). Called once at load, so a
    /// window's per-block eval never pulls a lazy adapter-file read into the block stream.
    pub fn materialize(&self) -> Result<()> {
        for slot in self.slots.values() {
            slot.materialize_adapters()?;
        }
        Ok(())
    }
}

/// The streamed block stack as an adapter target (SW-7096): `transformer_blocks.{i}.<leaf>` resolves
/// to block `i`'s slot. Any model that streams its blocks through a [`DitBlockStream`] — H3 video's
/// deferred DiT today, the image DiT's once it streams — routes its block paths here, so the
/// strict install lands every block residual in the overlay that [`DitBlockStream::materialize`]
/// re-applies.
impl AdaptableHost for DitBlockStream {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["transformer_blocks", i, rest @ ..] => {
                let index: usize = i.parse().ok()?;
                self.adapter_slot_mut(index, rest)
            }
            _ => None,
        }
    }
}

/// The logical `[out, in]` shape of one adaptable block leaf, or `None` for a path that is not one
/// of [`crate::adapters::BLOCK_TARGETS`].
fn block_leaf_shape(cfg: &MiniMaxH3DitConfig, leaf: &str) -> Option<(&'static str, [i32; 2])> {
    let hidden = cfg.hidden_size;
    let inner = cfg.inner_dim();
    let ffn = cfg.ffn_dim;
    let (name, shape) = match leaf {
        "attn.to_q" => ("attn.to_q", [inner, hidden]),
        "attn.to_k" => ("attn.to_k", [inner, hidden]),
        "attn.to_v" => ("attn.to_v", [inner, hidden]),
        "attn.to_out.0" => ("attn.to_out.0", [hidden, inner]),
        "ff.net.0.proj" => ("ff.net.0.proj", [2 * ffn, hidden]),
        "ff.net.2" => ("ff.net.2", [hidden, ffn]),
        _ => return None,
    };
    Some((name, shape))
}

impl DitBlockStream {
    /// Describe the block stack staged at `dir` — the same directory
    /// [`crate::dit::MiniMaxH3Dit::load_dir`] resolves, so a tiered install streams its own tier.
    pub fn new(dir: impl AsRef<Path>, dtype: Dtype, cfg: MiniMaxH3DitConfig) -> Result<Self> {
        let dir = dir.as_ref();
        if !dir.join("config.json").is_file() {
            return Err(Error::Msg(format!(
                "minimax-h3 block stream: {} has no config.json; a window cannot verify the tier it \
                 is re-opening",
                dir.display()
            )));
        }
        cfg.validate()?;
        Ok(Self {
            dir: dir.to_path_buf(),
            dtype,
            cfg,
            adapters: StreamedBlockAdapters::default(),
        })
    }

    /// The adapter slot for block `index`'s leaf `leaf` (`["attn", "to_q"]` …), created on first
    /// use. `None` for an out-of-range block or a leaf that is not adapter-addressable, which the
    /// strict install reports as unmatched — exactly what the resident block would answer.
    pub fn adapter_slot_mut(
        &mut self,
        index: usize,
        leaf: &[&str],
    ) -> Option<&mut AdaptableLinear> {
        if index >= self.n_blocks() {
            return None;
        }
        let (name, [out, inp]) = block_leaf_shape(&self.cfg, &leaf.join("."))?;
        match self.adapters.slots.entry((index, name)) {
            std::collections::btree_map::Entry::Occupied(slot) => Some(slot.into_mut()),
            std::collections::btree_map::Entry::Vacant(slot) => {
                let base = Array::zeros::<f32>(&[out, inp]).ok()?;
                Some(slot.insert(AdaptableLinear::dense(base, None)))
            }
        }
    }

    /// The adapter residuals this stream re-applies to every rebuilt block.
    pub fn adapters(&self) -> &StreamedBlockAdapters {
        &self.adapters
    }

    /// Push block `index`'s held residuals onto its freshly rebuilt `block`.
    fn install_adapters(&self, block: &mut DitBlock, index: usize) -> Result<()> {
        for ((_, leaf), slot) in self.adapters.slots.range((index, "")..(index + 1, "")) {
            if slot.adapters().is_empty() {
                continue;
            }
            let segs: Vec<&str> = leaf.split('.').collect();
            let target = block.adaptable_mut(&segs).ok_or_else(|| {
                Error::Msg(format!(
                    "minimax-h3 block stream: block {index} has no adaptable leaf {leaf}"
                ))
            })?;
            for adapter in slot.adapters() {
                target.push(adapter.clone());
            }
        }
        Ok(())
    }

    /// Blocks in the streamed stack — `num_layers`, 50 shipped.
    pub fn n_blocks(&self) -> usize {
        self.cfg.num_layers.max(0) as usize
    }

    pub fn config(&self) -> &MiniMaxH3DitConfig {
        &self.cfg
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// A **fresh** lazily-mapped view of the staged directory. One per window.
    ///
    /// MLX maps per tensor and materializes on first read, so this is cheap regardless of how many
    /// shards the tier ships: the whole 14-shard 66 GB bf16 partition opens at kilobytes of handles.
    pub fn open(&self) -> Result<Weights> {
        Weights::from_dir(&self.dir).map_err(|error| {
            Error::Msg(format!(
                "minimax-h3 block stream: reopening {}: {error}",
                self.dir.display()
            ))
        })
    }

    /// Reconstruct block `index`'s **body** out of `view`, draining its source handles.
    ///
    /// Draining is not tidiness: `Array` is refcounted, so a handle left in the view keeps the
    /// materialized buffer alive for the rest of the window and the bound silently does not hold —
    /// the failure `run_windowed`'s own doc warns about.
    pub fn materialize(&self, view: &mut Weights, index: usize) -> Result<DitBlock> {
        let prefix = format!("transformer_blocks.{index}");
        let mut block = DitBlock::from_weights_body_only(view, &prefix, &self.cfg, self.dtype)?;
        // sc-23402: force + GPU-verify this block's ten source tensors (sc-22414) before its
        // prefix is drained — one block per eval, which is also the sc-23108 submission bound.
        view.materialize_accessed()?;
        view.remove_prefix(&format!("{prefix}."));
        // SW-7096: the rebuilt block carries the configured adapters, exactly as the resident one.
        self.install_adapters(&mut block, index)?;
        Ok(block)
    }

    /// Reconstruct block `index`'s AdaLN projection out of `view`, draining its source handles.
    ///
    /// Used only by [`precompute_adaln_windowed`], once per request. The denoise path must never
    /// call this — see the module docs.
    pub fn materialize_adaln(&self, view: &mut Weights, index: usize) -> Result<AdaLnProjection> {
        let prefix = format!("transformer_blocks.{index}.adaln_proj.linear");
        let projection = AdaLnProjection::from_weights(view, &prefix, &self.cfg, self.dtype)?;
        // sc-23402: the projection's two source tensors, forced and GPU-verified (sc-22414) before
        // the precompute consumes them and the prefix is drained.
        view.materialize_accessed()?;
        view.remove_prefix(&format!("{prefix}."));
        Ok(projection)
    }

    /// The block plan this stream runs at `window`.
    pub fn plan(&self, window: usize) -> Result<BlockPlan> {
        Ok(BlockPlan::new(self.n_blocks(), window)?)
    }
}

/// Run the whole 50-block AdaLN projection pass under a bounded window (sc-18662).
///
/// The resident twin is [`AdaLnCache::precompute`], which requires the stack to already be loaded.
/// This is the [`LoadShape::DeferredMaterialization`](mlx_gen::gen_core::LoadShape) path: at no point
/// are more than `plan.window()` projections materialized, and each window's tables are forced
/// before its weights are released — without that the tables are unevaluated graph nodes still
/// referencing the projection, and the window would free nothing.
///
/// There is **no eviction step and nothing to evict**: a deferred load never held the projections, so
/// the cache this returns is the same object the resident path produces *after* its 26.02 GB drop.
/// That is why it returns no released-bytes figure where `precompute_and_evict` does — reporting a
/// release that did not happen is the over-declared-saving direction the contract refuses.
pub fn precompute_adaln_windowed(
    stream: &DitBlockStream,
    plan: &BlockPlan,
    cancel: &CancelFlag,
    schedule: TimestepSchedule,
    temb: &Array,
) -> Result<AdaLnCache> {
    if plan.n_blocks() != stream.n_blocks() {
        return Err(Error::Msg(format!(
            "minimax-h3 adaln window: the plan covers {} blocks but the stream has {}",
            plan.n_blocks(),
            stream.n_blocks()
        )));
    }
    let layers: Vec<AdaLnModulation> = run_windowed(
        plan,
        cancel,
        Vec::with_capacity(stream.n_blocks()),
        || stream.open(),
        |mut acc: Vec<AdaLnModulation>, view: &mut Weights, range: Range<usize>| {
            for index in range {
                let projection = stream.materialize_adaln(view, index)?;
                acc.push(projection.forward(temb).map_err(|error| {
                    Error::Msg(format!("minimax-h3 adaln window: layer {index}: {error}"))
                })?);
                // The projection drops here, at the end of its iteration, rather than at the end of
                // the window: a `window > 1` pass otherwise holds every projection in the window
                // plus its table, which is the residency this exists to bound.
            }
            Ok(acc)
        },
        // LOAD-BEARING, and it must cover the tables produced **so far** rather than only this
        // window's: `AdaLnModulation` is six lazy arrays, and one left unevaluated pins its
        // projection's 520 MB (bf16) past the release. Re-evaluating an already-materialized array
        // is free. Keep each eval to one block: a large window must not combine cold reads into
        // the same Metal submission and reintroduce the resident-stack timeout (sc-23108).
        |acc: &Vec<AdaLnModulation>| {
            for layer in acc {
                mlx_rs::transforms::eval(layer.tables())?;
            }
            Ok(())
        },
    )?;
    if layers.len() != stream.n_blocks() {
        return Err(Error::Msg(format!(
            "minimax-h3 adaln window: projected {} of {} layers",
            layers.len(),
            stream.n_blocks()
        )));
    }
    AdaLnCache::from_layers(schedule, layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The residency bound is a claim about **bytes**, so it is asserted against the two name lists
    /// rather than against the loader's shape: a window materializes the ten body tensors and
    /// neither of the two `adaln_proj` ones.
    ///
    /// A regression here — a window that started pulling `adaln_proj` again — would re-materialize
    /// 39.3 % of the DiT per step and produce **bit-identical output**, so nothing else in the suite
    /// could see it.
    #[test]
    fn window_bytes_body_only_is_the_residency_bound() {
        let all = DitBlock::names("transformer_blocks.7");
        let body = DitBlock::body_names("transformer_blocks.7");
        assert_eq!(all.len(), 12, "the published block ships twelve tensors");
        assert_eq!(body.len(), 10);
        for dropped in [
            "transformer_blocks.7.adaln_proj.linear.weight",
            "transformer_blocks.7.adaln_proj.linear.bias",
        ] {
            assert!(all.contains(&dropped.to_owned()), "premise: {dropped}");
            assert!(
                !body.contains(&dropped.to_owned()),
                "a denoise window must not read {dropped}"
            );
        }
        // And the body list is the whole remainder, not a subset that quietly drops a norm.
        let mut expected: Vec<String> = all
            .into_iter()
            .filter(|n| !n.contains(".adaln_proj."))
            .collect();
        expected.sort();
        let mut got = body;
        got.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn a_missing_tier_marker_is_refused_before_any_window_opens() {
        let empty = tempfile::tempdir().expect("temp dir");
        let error =
            DitBlockStream::new(empty.path(), Dtype::Bfloat16, MiniMaxH3DitConfig::default())
                .unwrap_err()
                .to_string();
        assert!(error.contains("config.json"), "{error}");
    }

    /// SW-7096: the stream is an adapter host over exactly the six adaptable leaves of each of its
    /// blocks, at each leaf's logical shape; everything else is unmatched, as on a resident block.
    #[test]
    fn stream_adapter_slots_cover_exactly_the_block_leaves() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        let cfg = MiniMaxH3DitConfig::default();
        let mut stream = DitBlockStream::new(dir.path(), Dtype::Bfloat16, cfg.clone()).unwrap();
        for (leaf, shape) in [
            ("attn.to_q", [7168, 5376]),
            ("attn.to_k", [7168, 5376]),
            ("attn.to_v", [7168, 5376]),
            ("attn.to_out.0", [5376, 7168]),
            ("ff.net.0.proj", [28672, 5376]),
            ("ff.net.2", [5376, 14336]),
        ] {
            let path = format!("transformer_blocks.49.{leaf}");
            let segs: Vec<&str> = path.split('.').collect();
            let slot = stream
                .adaptable_mut(&segs)
                .unwrap_or_else(|| panic!("{path}"));
            assert_eq!(slot.base_shape(), shape.to_vec(), "{path}");
        }
        for unmatched in [
            "transformer_blocks.50.attn.to_q",
            "transformer_blocks.0.adaln_proj.linear",
            "transformer_blocks.0.attn.qkv_proj",
            "token_refiner.refiner_blocks.0.attn.to_q",
            "proj_in",
        ] {
            let segs: Vec<&str> = unmatched.split('.').collect();
            assert!(stream.adaptable_mut(&segs).is_none(), "{unmatched}");
        }
        assert!(
            stream.adapters().is_empty(),
            "creating slots holds no residual"
        );
        // Residuals pushed onto a slot are what `materialize` re-applies to the rebuilt block
        // (proven against real weights in `tests/h3_image_lora_real.rs`, which ordinary CI does
        // not run because it rebuilds blocks).
        let q = ["transformer_blocks", "3", "attn", "to_q"];
        for rank in [2, 3] {
            stream
                .adaptable_mut(&q)
                .unwrap()
                .push(mlx_gen::adapters::Adapter::Lora {
                    a: Array::zeros::<f32>(&[5376, rank]).unwrap(),
                    b: Array::zeros::<f32>(&[rank, 7168]).unwrap(),
                    scale: 0.5,
                });
        }
        assert_eq!(stream.adapters().residual_count(), 2);
        assert_eq!(stream.adaptable_mut(&q).unwrap().adapters().len(), 2);
        let cloned = stream.clone();
        assert_eq!(
            cloned.adapters().residual_count(),
            2,
            "a cloned stream keeps the overlay"
        );
    }

    #[test]
    fn the_plan_covers_the_shipped_fifty_blocks() {
        let cfg = MiniMaxH3DitConfig::default();
        assert_eq!(cfg.num_layers, 50);
        let plan = BlockPlan::new(cfg.num_layers as usize, 1).unwrap();
        assert_eq!(plan.window_count(), 50);
        assert!(plan.is_bounded());
        assert!(!BlockPlan::new(cfg.num_layers as usize, 50)
            .unwrap()
            .is_bounded());
    }
}

// ── The text-encoder arm (sc-18662, `TransformerComponent::Both`) ────────────────────────────────

/// Reopenable description of a staged Qwen3-VL-32B condition encoder's decoder stack.
///
/// The DiT's twin, and deliberately a separate type rather than a generic over both: the two stacks
/// differ in every detail a window touches — key prefix (`model.language_model.layers.{i}` against
/// `transformer_blocks.{i}`), constructor arity, and which tensors a layer even has. A shared
/// generic would abstract over four fields and hide the one thing worth reading, which is *what a
/// window of this stack costs*.
///
/// **This arm is why rung 4 can be declared
/// [`TransformerComponent::Both`](mlx_gen::gen_core::memory_strategy::TransformerComponent::Both)**
/// (`gen_core::memory_strategy`). Declaring `Dit` alone would leave the conditioning phase — which
/// was the model's binding stage until sc-19120's packed tiers, and is still the taller of the two
/// at every tier — entirely untouched.
#[derive(Clone, Debug)]
pub struct TeBlockStream {
    dir: PathBuf,
    prefix: String,
    cfg: MiniMaxH3TeConfig,
    n_layers: usize,
}

impl TeBlockStream {
    /// Describe the decoder stack staged at `dir` under `prefix`.
    ///
    /// `n_layers` is the **tapped** depth (`select_hidden`, 50 shipped), not the checkpoint's 64:
    /// `MiniMaxH3TextEncoder` runs layers `0..select_hidden` and never touches the rest, so a window
    /// schedule over 64 would re-materialize fourteen layers that contribute nothing.
    pub fn new(dir: impl AsRef<Path>, prefix: &str, cfg: MiniMaxH3TeConfig) -> Result<Self> {
        let dir = dir.as_ref();
        if !dir.join("config.json").is_file() {
            return Err(Error::Msg(format!(
                "minimax-h3 te stream: {} has no config.json; a window cannot verify the tier it is \
                 re-opening",
                dir.display()
            )));
        }
        let n_layers = cfg.out_layer()? + 1;
        Ok(Self {
            dir: dir.to_path_buf(),
            prefix: prefix.to_owned(),
            cfg,
            n_layers,
        })
    }

    /// Tapped decoder layers — `select_hidden`, 50 shipped.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// A **fresh** lazily-mapped view. One per window, for the reason [`DitBlockStream::open`] gives.
    pub fn open(&self) -> Result<Weights> {
        Weights::from_dir(&self.dir).map_err(|error| {
            Error::Msg(format!(
                "minimax-h3 te stream: reopening {}: {error}",
                self.dir.display()
            ))
        })
    }

    /// Reconstruct decoder layer `index` out of `view`, draining its source handles.
    pub fn materialize(&self, view: &mut Weights, index: usize) -> Result<Qwen3DecoderLayer> {
        let prefix = format!("{}.layers.{index}", self.prefix);
        let layer = Qwen3DecoderLayer::from_weights(
            view,
            &prefix,
            self.cfg.num_heads,
            self.cfg.num_kv_heads,
            self.cfg.head_dim,
            self.cfg.rms_norm_eps,
        )?;
        // sc-23402: force + GPU-verify this layer's source tensors (sc-22414) before the window
        // runs it and the prefix is drained — the windowed twin of the resident constructor's
        // `materialize_accessed` in `text_encoder::encoder`.
        view.materialize_accessed()?;
        view.remove_prefix(&format!("{prefix}."));
        Ok(layer)
    }

    /// The block plan this stream runs at `window`.
    pub fn plan(&self, window: usize) -> Result<BlockPlan> {
        Ok(BlockPlan::new(self.n_layers, window)?)
    }
}
