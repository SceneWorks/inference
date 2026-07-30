//! Ladder rung 4 — **bounded transformer residency** for the MLX Z-Image DiT (SC-15754).
//!
//! The window lifecycle itself is NOT implemented here: it is
//! [`mlx_gen::block_residency::run_windowed`] (SC-15750), the shared primitive that also serves
//! Mage-Flow / Qwen / FLUX. This module is only the family-side half — "how do I rebuild Z-Image
//! block *n* from the snapshot, with the same quantization and the same adapters the resident block
//! carries" — which is the part that genuinely differs per family.
//!
//! ## Why re-reading per window is nearly free
//!
//! [`Array::load_safetensors`](mlx_rs::Array::load_safetensors_with_metadata) is **lazy per tensor**:
//! SC-15744 measured 1073 handles at 0.0 MiB until something is evaluated. So opening the snapshot
//! once per window costs a JSON header parse, not 3.4 GB. Only the tensors a window's blocks actually
//! read are materialized, and [`Weights::remove_accessed`] then drops the view's own reference to
//! exactly those, so the window's residency is the blocks themselves and nothing else.
//!
//! `remove_accessed` is deliberate rather than decorative. SC-15750's primitive documents that `apply`
//! must take tensors OUT of the view instead of cloning them, because `Array` is refcounted and a
//! clone left in the map keeps the materialized buffer alive. The Z-Image block constructors take
//! `&Weights` and clone (they are the same constructors the resident path uses, and forking them would
//! be a second way to build a block — precisely the divergence hazard this epic exists to remove), so
//! the drain runs immediately after each block instead: `from_weights` records every key it read
//! through `require`/`get`, and `remove_accessed` removes exactly that set. A constructor read that
//! never happened therefore cannot be drained, which is what makes
//! [`block_stream_drains_exactly_what_the_block_read`] a real check rather than a tautology.
//!
//! ## The resident-blocks precondition, and why it is enforced rather than documented
//!
//! Streaming bounds residency only if the *resident* `layers` are not also materialized. Under
//! [`OffloadPolicy::Sequential`](mlx_gen::OffloadPolicy) they are not: the heavy bundle is rebuilt per
//! generate, its blocks are fresh unevaluated handles, and a streaming forward never touches them, so
//! they stay at ~0 bytes. Under `Resident` the same generator serves many requests, so after the first
//! ordinary render the blocks ARE materialized and a streaming forward would add a window on top of
//! them — strictly more memory and strictly slower.
//!
//! Rung 1 has a load-time-vs-request-time seam of the same shape and resolves it by silently yielding
//! resident behaviour (see [`crate::memory_strategy`]). Rung 4 does NOT: `generate` rejects a
//! `stream_transformer_blocks` request on a non-`Sequential` generator with a typed error. A silent
//! degradation there would be a *false green* — the calibration harness would record a rung-4 run that
//! saved nothing, which is exactly the failure mode SC-15750's mutation check exists to prevent, one
//! layer up.

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::transformer_block::{ZImageBlockConfig, ZImageTransformerBlock};

/// The adapters installed on ONE block, addressed by the block-local dotted path
/// (`"attention.to_q"`, `"feed_forward.w1"`, `"adaLN_modulation.0"`, …).
///
/// Captured from the resident blocks after [`crate::adapters::apply_z_image_adapters`] has run, and
/// re-installed verbatim on each materialized block. Z-Image installs every adapter as a *forward-time
/// residual* (`apply_adapters_strict` only ever `push`es an [`Adapter`]; it never folds a delta into a
/// base the way the diff-patch path does), so replaying the captured list reproduces the resident
/// block exactly rather than approximating it. [`Adapter`] holds refcounted `Array` factors, so the
/// capture is a handle copy — a rank-16 LoRA's factors, not a `[out, in]` delta.
#[derive(Clone, Default)]
pub(crate) struct BlockAdapters {
    per_path: Vec<(String, Vec<Adapter>)>,
}

impl BlockAdapters {
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.per_path.is_empty()
    }

    /// Re-install the captured adapters on a freshly materialized block.
    fn install(&self, block: &mut ZImageTransformerBlock) -> Result<()> {
        for (path, adapters) in &self.per_path {
            let segs: Vec<&str> = path.split('.').collect();
            let target = block.adaptable_mut(&segs).ok_or_else(|| {
                Error::Msg(format!(
                    "z-image block stream: adapter target `{path}` is absent from a materialized \
                     block — the streamed block tree does not match the resident one"
                ))
            })?;
            target.set_adapters(adapters.clone());
        }
        Ok(())
    }
}

/// Everything needed to rebuild one Z-Image transformer block stack from its snapshot, on demand.
///
/// Held by [`ZImageTransformer`](crate::ZImageTransformer) (and by the ControlNet variant for its own
/// `control_layers`), and consumed by the windowed forward. Cheap to clone: a path, a few scalars, and
/// refcounted adapter factors.
#[derive(Clone)]
pub(crate) struct ZImageBlockStream {
    source: WeightsSource,
    /// Key prefix for the stack (`"layers"`, `"control_layers"`), already joined with any checkpoint
    /// prefix, so block `n` is `{base}.{n}`.
    base: String,
    cfg: ZImageBlockConfig,
    n_blocks: usize,
    /// The load-time quantization the resident stack received, replayed per materialized block so the
    /// streamed weights are byte-identical to the resident ones. `None` for a dense/bf16 tier. On a
    /// pre-quantized (packed) snapshot this is recorded but is a documented no-op — the same no-op the
    /// resident path performed.
    quant_bits: Option<i32>,
    /// Per-block captured adapters, indexed by block. Empty when no adapter is installed.
    adapters: Vec<BlockAdapters>,
    /// Whether the ControlNet block variant is being streamed (it carries the extra `before_proj` /
    /// `after_proj` projections). Base stacks are `false`.
    control: bool,
}

impl ZImageBlockStream {
    /// Declare a streamable stack. `source` must be re-openable — a snapshot directory or a single
    /// `.safetensors` file. An in-memory / ComfyUI-normalized `Weights` has no re-openable source, so
    /// those loads simply do not construct one and rung 4 is declared unavailable for them.
    pub(crate) fn new(
        source: WeightsSource,
        base: impl Into<String>,
        cfg: ZImageBlockConfig,
        n_blocks: usize,
        control: bool,
    ) -> Self {
        Self {
            source,
            base: base.into(),
            cfg,
            n_blocks,
            quant_bits: None,
            adapters: vec![BlockAdapters::default(); n_blocks],
            control,
        }
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// The block shape this stream materializes at — so a test can build the resident twin from the
    /// same derivation rather than a second spelling of it.
    #[cfg(test)]
    pub(crate) fn cfg_for_test(&self) -> &ZImageBlockConfig {
        &self.cfg
    }

    #[cfg(test)]
    pub(crate) fn has_adapters(&self) -> bool {
        self.adapters.iter().any(|a| !a.is_empty())
    }

    /// Record the load-time quantization so a materialized block reproduces it.
    pub(crate) fn set_quant_bits(&mut self, bits: i32) {
        self.quant_bits = Some(bits);
    }

    /// Capture the adapters currently installed on `blocks` so materialized blocks carry them too.
    ///
    /// Called after the resident stack has been adapted, so the two paths cannot diverge on *which*
    /// adapters, *which* targets, or *what order* — the streamed block replays exactly what the
    /// resident block ended up holding.
    pub(crate) fn capture_adapters(&mut self, blocks: &mut [ZImageTransformerBlock]) {
        self.adapters = blocks
            .iter_mut()
            .map(|block| {
                let paths = block.adaptable_paths();
                let mut per_path = Vec::new();
                for path in paths {
                    let segs: Vec<&str> = path.split('.').collect();
                    if let Some(target) = block.adaptable_mut(&segs) {
                        let adapters = target.adapters();
                        if !adapters.is_empty() {
                            per_path.push((path.clone(), adapters.to_vec()));
                        }
                    }
                }
                BlockAdapters { per_path }
            })
            .collect();
    }

    /// Open a fresh lazy view of the stack's weights. Called once per window by
    /// [`run_windowed`](mlx_gen::block_residency::run_windowed).
    pub(crate) fn open(&self) -> Result<Weights> {
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    fn check(&self, index: usize, control: bool) -> Result<String> {
        if index >= self.n_blocks {
            return Err(Error::Msg(format!(
                "z-image block stream: block {index} is out of range for a {}-block stack",
                self.n_blocks
            )));
        }
        if control != self.control {
            return Err(Error::Msg(format!(
                "z-image block stream `{}` is a {} stack; it cannot materialize a {} block",
                self.base,
                if self.control { "control" } else { "base" },
                if control { "control" } else { "base" },
            )));
        }
        Ok(format!("{}.{index}", self.base))
    }

    /// Materialize base block `index` out of `view`, quantized and adapted exactly like its resident
    /// twin, then drain the view of precisely the tensors that block read.
    pub(crate) fn materialize(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<ZImageTransformerBlock> {
        let prefix = self.check(index, false)?;
        let mut block = ZImageTransformerBlock::from_weights(view, &prefix, self.cfg)?;
        // LOAD-BEARING (SC-15750): the view keeps its own refcounted handle to every tensor the
        // constructor cloned. Draining exactly the accessed keys is what makes the window's drop a
        // real release rather than a no-op that still produces correct images.
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            block.quantize(bits)?;
        }
        if let Some(adapters) = self.adapters.get(index) {
            adapters.install(&mut block)?;
        }
        Ok(block)
    }

    /// Materialize ControlNet block `index` — the base submodules plus the `after_proj` hint
    /// projection and, on block 0, the `before_proj` seed.
    ///
    /// No adapter replay: [`AdaptableHost for ZImageControlTransformer`](crate::ZImageControlTransformer)
    /// routes every adapter path to the composed **base** DiT, because the fork trains LoRA/LoKr on the
    /// base transformer and never on the ControlNet branch. So a control block has no adapters to
    /// carry, and `capture_adapters` is never called on a control stream.
    pub(crate) fn materialize_control(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<crate::control_transformer_block::ZImageControlBlock> {
        let prefix = self.check(index, true)?;
        let mut block = crate::control_transformer_block::ZImageControlBlock::from_weights(
            view,
            &prefix,
            self.cfg,
            index == 0,
        )?;
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            block.quantize(bits)?;
        }
        Ok(block)
    }
}

/// The plan + cancellation a windowed forward runs under. A thin pair so the forward signatures do not
/// grow two more positional arguments, and so `Copy` keeps the per-step closures cheap.
#[derive(Clone, Copy)]
pub(crate) struct BlockWindow<'a> {
    pub(crate) plan: mlx_gen::block_residency::BlockPlan,
    pub(crate) cancel: &'a mlx_gen::CancelFlag,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    fn cfg() -> ZImageBlockConfig {
        ZImageBlockConfig {
            dim: 8,
            n_heads: 2,
            norm_eps: 1e-5,
        }
    }

    /// A two-block synthetic stack written to a temp `.safetensors`, so the stream can be exercised
    /// without a real snapshot.
    fn fixture(tag: &str, n_blocks: usize) -> std::path::PathBuf {
        fixture_at(tag, n_blocks, cfg().dim)
    }

    /// [`fixture`] at an explicit width, so the quantization test can use a group-64-divisible one.
    fn fixture_at(tag: &str, n_blocks: usize, dim: i32) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("zimage-block-stream-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let mut named: Vec<(String, Array)> = Vec::new();
        for i in 0..n_blocks {
            for (leaf, shape) in [
                ("attention.to_q.weight", vec![dim, dim]),
                ("attention.to_k.weight", vec![dim, dim]),
                ("attention.to_v.weight", vec![dim, dim]),
                ("attention.to_out.0.weight", vec![dim, dim]),
                ("attention.norm_q.weight", vec![dim / 2]),
                ("attention.norm_k.weight", vec![dim / 2]),
                ("feed_forward.w1.weight", vec![dim, dim]),
                ("feed_forward.w2.weight", vec![dim, dim]),
                ("feed_forward.w3.weight", vec![dim, dim]),
                ("attention_norm1.weight", vec![dim]),
                ("attention_norm2.weight", vec![dim]),
                ("ffn_norm1.weight", vec![dim]),
                ("ffn_norm2.weight", vec![dim]),
                ("adaLN_modulation.0.weight", vec![4 * dim, dim]),
            ] {
                let n: i32 = shape.iter().product();
                let values: Vec<f32> = (0..n).map(|k| (k as f32 + i as f32) * 0.01).collect();
                named.push((
                    format!("layers.{i}.{leaf}"),
                    Array::from_slice(&values, &shape),
                ));
            }
        }
        let refs: Vec<(&str, &Array)> = named.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        path
    }

    fn stream(path: &std::path::Path, n: usize) -> ZImageBlockStream {
        ZImageBlockStream::new(
            WeightsSource::File(path.to_path_buf()),
            "layers",
            cfg(),
            n,
            false,
        )
    }

    /// The drain must remove EXACTLY the block's own tensors — not a whole prefix.
    ///
    /// The distinction is load-bearing. `remove_prefix("layers.0.")` would drain the same bytes here
    /// but would also hide a constructor that silently stopped reading a tensor: the block would be
    /// built from a stale default and the wipe would erase the evidence. Exact draining leaves an
    /// un-read key visible, which is what the second half asserts — and it is not hypothetical, it is
    /// how the first draft of this fixture was caught spelling `q_norm` where the block reads
    /// `norm_q`.
    #[test]
    fn block_stream_drains_exactly_what_the_block_read() {
        let path = fixture("drain", 2);
        let s = stream(&path, 2);
        let mut view = s.open().unwrap();
        let before = view.len();
        s.materialize(&mut view, 0).unwrap();
        let after = view.len();
        assert!(
            after < before,
            "materializing block 0 must drain its tensors"
        );
        // Block 1's tensors are untouched, and block 0's are gone.
        assert!(view.get("layers.1.attention.to_q.weight").is_some());
        assert!(view.get("layers.0.attention.to_q.weight").is_none());
        // Nothing of block 0 survived: the fixture holds the block's whole read set and no more.
        assert_eq!(
            view.keys().filter(|k| k.starts_with("layers.0.")).count(),
            0,
            "every block-0 tensor the constructor read must have been drained"
        );
        assert_eq!(before - after, before / 2);

        // Mutation half: a tensor the constructor never reads is NOT drained, so a constructor that
        // quietly drops a read stays observable instead of being wiped along with the rest.
        let mut view = s.open().unwrap();
        view.insert("layers.0.unread_by_the_block", Array::from_f32(1.0));
        s.materialize(&mut view, 0).unwrap();
        assert!(
            view.get("layers.0.unread_by_the_block").is_some(),
            "an un-read key must survive an exact drain"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// A fresh view per window is what keeps the bound: two windows must not share a map that
    /// accumulates. Re-opening returns the full key set every time.
    #[test]
    fn each_window_opens_an_independent_view() {
        let path = fixture("reopen", 2);
        let s = stream(&path, 2);
        let mut first = s.open().unwrap();
        let full = first.len();
        s.materialize(&mut first, 0).unwrap();
        assert!(first.len() < full);
        let second = s.open().unwrap();
        assert_eq!(second.len(), full, "a new window must see the whole stack");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// **The quantization replay, which had no coverage anywhere.**
    ///
    /// `set_quant_bits` → `block.quantize(bits)` is what makes a streamed block byte-identical to its
    /// resident twin on a **dense snapshot loaded with `spec.quantize`** — a real configuration
    /// (`mlx_gen::quant::needs_load_time_quant` returning `true` is exactly it). Nothing reached it:
    /// the DiT tests never quantize, and the real-weights suite runs against a *pre-quantized tier
    /// dir* where `quantize()` is a documented no-op. So the replay was correct-by-construction and
    /// unverified, while a doc comment claimed otherwise.
    ///
    /// It is tested here rather than at the DiT level because the whole-transformer `quantize()` also
    /// touches the timestep embedder, whose tiny fixture in-features are not divisible by the group
    /// size — the mechanism under test is per-block, and this is where it lives.
    #[test]
    fn a_materialized_block_replays_the_recorded_quantization() {
        // Group-64 packing needs in-features divisible by 64, so this fixture is 64-wide.
        let path = fixture_at("quant", 1, 64);
        let mut s = ZImageBlockStream::new(
            WeightsSource::File(path.clone()),
            "layers",
            ZImageBlockConfig {
                dim: 64,
                n_heads: 2,
                norm_eps: 1e-5,
            },
            1,
            false,
        );

        // No bits recorded → the materialized block stays dense, like its unquantized twin.
        let mut view = s.open().unwrap();
        let mut dense = s.materialize(&mut view, 0).unwrap();
        assert!(!dense
            .adaptable_mut(&["attention", "to_q"])
            .unwrap()
            .is_quantized());

        // Bits recorded (what `ZImageTransformer::quantize` does) → the block comes back packed.
        s.set_quant_bits(4);
        let mut view = s.open().unwrap();
        let mut packed = s.materialize(&mut view, 0).unwrap();
        for path in ["attention.to_q", "feed_forward.w1", "adaLN_modulation.0"] {
            let segs: Vec<&str> = path.split('.').collect();
            assert!(
                packed.adaptable_mut(&segs).unwrap().is_quantized(),
                "{path} must be quantized by the replay"
            );
        }

        // And it matches a RESIDENT block quantized the same way — the property that matters, since
        // "is quantized" alone would pass on a different tier or group size.
        let view = s.open().unwrap();
        let mut resident =
            ZImageTransformerBlock::from_weights(&view, "layers.0", *s.cfg_for_test()).unwrap();
        resident.quantize(4).unwrap();
        let (rw, rs, rb, _, rg, rbits) = resident
            .adaptable_mut(&["attention", "to_q"])
            .unwrap()
            .quantized_params()
            .map(|(w, s, b, bias, g, bits)| {
                (w.clone(), s.clone(), b.clone(), bias.cloned(), g, bits)
            })
            .expect("resident is quantized");
        let (pw, ps, pb, _, pg, pbits) = packed
            .adaptable_mut(&["attention", "to_q"])
            .unwrap()
            .quantized_params()
            .map(|(w, s, b, bias, g, bits)| {
                (w.clone(), s.clone(), b.clone(), bias.cloned(), g, bits)
            })
            .expect("streamed is quantized");
        assert_eq!((rg, rbits), (pg, pbits), "group size / bits must match");
        for (a, b, label) in [
            (&rw, &pw, "codes"),
            (&rs, &ps, "scales"),
            (&rb, &pb, "biases"),
        ] {
            assert!(
                mlx_rs::ops::all_close(a, b, None, None, None)
                    .unwrap()
                    .item::<bool>(),
                "{label} differ between the resident and streamed block"
            );
        }
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn out_of_range_blocks_are_rejected() {
        let path = fixture("range", 2);
        let s = stream(&path, 2);
        let mut view = s.open().unwrap();
        let err = match s.materialize(&mut view, 2) {
            Ok(_) => panic!("block 2 of a 2-block stack must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("out of range"), "{err}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// Captured adapters travel onto a materialized block. The capture is by block-local path, so a
    /// LoRA installed on block 1 must NOT appear on block 0.
    #[test]
    fn captured_adapters_are_reinstalled_per_block() {
        let path = fixture("adapters", 2);
        let mut s = stream(&path, 2);

        // Build the two resident blocks and install a LoRA on block 1 only.
        let view = s.open().unwrap();
        let mut resident: Vec<ZImageTransformerBlock> = (0..2)
            .map(|i| {
                ZImageTransformerBlock::from_weights(&view, &format!("layers.{i}"), cfg()).unwrap()
            })
            .collect();
        let a = Array::from_slice(&vec![0.5f32; (cfg().dim * 2) as usize], &[cfg().dim, 2]);
        let b = Array::from_slice(&vec![0.25f32; (2 * cfg().dim) as usize], &[2, cfg().dim]);
        resident[1]
            .adaptable_mut(&["attention", "to_q"])
            .unwrap()
            .push(Adapter::Lora {
                a: a.clone(),
                b: b.clone(),
                scale: 1.0,
            });
        s.capture_adapters(&mut resident);
        assert!(s.has_adapters());

        let mut view = s.open().unwrap();
        let b0 = s.materialize(&mut view, 0).unwrap();
        let mut view = s.open().unwrap();
        let mut b1 = s.materialize(&mut view, 1).unwrap();
        let mut b0 = b0;
        assert!(b0
            .adaptable_mut(&["attention", "to_q"])
            .unwrap()
            .adapters()
            .is_empty());
        assert_eq!(
            b1.adaptable_mut(&["attention", "to_q"])
                .unwrap()
                .adapters()
                .len(),
            1
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
