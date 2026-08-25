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

use std::ops::Range;
use std::path::{Path, PathBuf};

use mlx_rs::{Array, Dtype};

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
        })
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
        let block = DitBlock::from_weights_body_only(view, &prefix, &self.cfg, self.dtype)?;
        view.remove_prefix(&format!("{prefix}."));
        Ok(block)
    }

    /// Reconstruct block `index`'s AdaLN projection out of `view`, draining its source handles.
    ///
    /// Used only by [`precompute_adaln_windowed`], once per request. The denoise path must never
    /// call this — see the module docs.
    pub fn materialize_adaln(&self, view: &mut Weights, index: usize) -> Result<AdaLnProjection> {
        let prefix = format!("transformer_blocks.{index}.adaln_proj.linear");
        let projection = AdaLnProjection::from_weights(view, &prefix, &self.cfg, self.dtype)?;
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
        // is free.
        |acc: &Vec<AdaLnModulation>| {
            let flat: Vec<&Array> = acc.iter().flat_map(AdaLnModulation::tables).collect();
            mlx_rs::transforms::eval(flat).map_err(Into::into)
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
        view.remove_prefix(&format!("{prefix}."));
        Ok(layer)
    }

    /// The block plan this stream runs at `window`.
    pub fn plan(&self, window: usize) -> Result<BlockPlan> {
        Ok(BlockPlan::new(self.n_layers, window)?)
    }
}
