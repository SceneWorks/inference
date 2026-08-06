//! Ladder rung 4 — **bounded transformer residency** for the Chroma1 MMDiT (SC-15520).
//!
//! The window lifecycle is NOT implemented here: it is
//! [`mlx_gen::block_residency::run_windowed`] (SC-15750), the shared primitive Z-Image, Qwen,
//! FLUX.1, Anima and Kolors also drive. This module is only the family-side half — "how do I rebuild
//! Chroma block *n* from the snapshot, carrying the same tier and the same adapters the resident
//! block carries" — which is the part that genuinely differs per family.
//!
//! ## Two sub-stacks, one cadence
//!
//! Chroma inherits FLUX's split trunk: 19 `transformer_blocks` (double / joint stream) followed by
//! 38 `single_transformer_blocks`. They are windowed **independently** over the same cadence,
//! because the carried activation changes shape between them (`(encoder, hidden)` before the
//! concatenation, one `joint` tensor after it) and a single plan cannot span both.
//!
//! ## Why re-reading per window is nearly free
//!
//! [`Weights::from_dir`] goes through `Array::load_safetensors_with_metadata`, which is **lazy per
//! tensor**: the handles exist, the bytes do not, until something is evaluated. Re-opening
//! `transformer/` once per window costs a header parse, not 5.18 GB. Only the tensors a window's
//! blocks actually read are materialized, and [`Weights::remove_accessed`] then drops the view's own
//! reference to exactly those. On this family that drain is measured **inert** on the request peak
//! (see [`ChromaBlockStream::materialize_double`]) — a fresh view is opened per window, so its
//! references die at the window boundary regardless — and it is kept as the shared discipline rather
//! than claimed as load-bearing on a sibling's evidence.
//!
//! ## What rung 4 does NOT bound here
//!
//! The pruned-adaLN trunk stays resident: `x_embedder`, `context_embedder`, `proj_out` and the whole
//! dense `distilled_guidance_layer` Approximator (a 5-deep residual stack plus its per-layer norm
//! vectors, deliberately kept dense at every tier). Those run once per step and are not block
//! weights; the window addresses `transformer_blocks.{n}` / `single_transformer_blocks.{n}` and
//! never touches them.
//!
//! ## Adapter replay
//!
//! A window rebuilt from the snapshot would silently drop the LoRA/LoKr residuals installed at load
//! — a plausible wrong image with no error anywhere. Chroma installs every adapter as a
//! forward-time residual on an [`AdaptableLinear`], so replaying the captured per-block list
//! reproduces the resident block exactly rather than approximating it, and a block that comes back
//! **without** a pair the resident stack had is refused loudly ([`BlockAdapters::install`] and the
//! `adapters_expected` guard below).
//!
//! Production admission is stricter still — [`crate::memory_strategy::structurally_streamable`]
//! withdraws rung 4 for any spec carrying adapters — but the replay is implemented and tested
//! regardless, because "the route is currently refused" is not a mechanism and the refusal is one
//! contract edit away from being relaxed.

use mlx_gen::adapters::Adapter;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::config::ChromaTransformerConfig;
use crate::transformer::{DoubleBlock, SingleBlock, StreamBlock};

/// The adapters installed on ONE block, addressed by the block-local dotted path
/// (`"attn.to_q"`, `"ff.net.0.proj"`, `"proj_mlp"`, …).
///
/// Captured from the resident blocks after [`crate::adapters::apply_chroma_adapters`] has run, so a
/// streamed block replays exactly what the resident block ended up holding — never a second
/// derivation of which adapters should have landed where.
#[derive(Clone, Default)]
pub(crate) struct BlockAdapters {
    per_path: Vec<(String, Vec<Adapter>)>,
}

impl BlockAdapters {
    pub(crate) fn is_empty(&self) -> bool {
        self.per_path.is_empty()
    }

    fn capture<B: StreamBlock>(block: &mut B) -> Self {
        let mut per_path = Vec::new();
        for path in B::ADAPTER_PATHS {
            let segments: Vec<&str> = path.split('.').collect();
            if let Some(target) = block.adapter_target(&segments) {
                let adapters = target.adapters();
                if !adapters.is_empty() {
                    per_path.push(((*path).to_owned(), adapters.to_vec()));
                }
            }
        }
        Self { per_path }
    }

    fn install<B: StreamBlock>(&self, block: &mut B, what: &str) -> Result<()> {
        for (path, adapters) in &self.per_path {
            let segments: Vec<&str> = path.split('.').collect();
            let target = block.adapter_target(&segments).ok_or_else(|| {
                Error::Msg(format!(
                    "chroma block stream: adapter target `{path}` is absent from a materialized \
                     {what} — the streamed block tree does not match the resident one"
                ))
            })?;
            target.set_adapters(adapters.clone());
        }
        Ok(())
    }

    /// Verify the replay actually landed: every captured path must come back carrying exactly the
    /// captured adapter count.
    ///
    /// This is the **present-but-wrong** guard, and it is a different failure from a missing one. A
    /// `set_adapters` that resolved to the wrong leaf, or a constructor that rebuilt a leaf after the
    /// install, leaves the block with an adapter list that is present and short (or empty) at the
    /// path that matters — which a bare "did anything get installed" check waves through. Counted
    /// per path, so a partially-replayed block is refused rather than rendered.
    fn verify<B: StreamBlock>(&self, block: &mut B, what: &str) -> Result<()> {
        for (path, adapters) in &self.per_path {
            let segments: Vec<&str> = path.split('.').collect();
            let installed = block.adapter_target(&segments).map(|t| t.adapters().len());
            if installed != Some(adapters.len()) {
                return Err(Error::Msg(format!(
                    "chroma block stream: {what} came back with {installed:?} adapters at `{path}` \
                     after replaying {} — a streamed block that silently loses part of its LoRA \
                     renders a plausible, wrong image",
                    adapters.len()
                )));
            }
        }
        Ok(())
    }
}

/// Everything needed to rebuild one Chroma DiT block from its snapshot, on demand.
///
/// Held by [`ChromaTransformer`](crate::transformer::ChromaTransformer) and consumed by the windowed
/// forward. Cheap to clone: a path, a `Copy` config and refcounted adapter factors.
#[derive(Clone)]
pub(crate) struct ChromaBlockStream {
    /// The re-openable `transformer/` component directory — never the caller's raw
    /// `LoadSpec::weights`, which is the snapshot root.
    source: WeightsSource,
    cfg: ChromaTransformerConfig,
    double: Vec<BlockAdapters>,
    single: Vec<BlockAdapters>,
    /// Whether the resident stacks carried ANY adapter. A materialized block whose capture is empty
    /// while this is set is refused rather than rendered without its residuals.
    adapters_expected: bool,
}

impl ChromaBlockStream {
    pub(crate) fn new(source: WeightsSource, cfg: ChromaTransformerConfig) -> Self {
        Self {
            source,
            cfg,
            double: vec![BlockAdapters::default(); cfg.num_layers],
            single: vec![BlockAdapters::default(); cfg.num_single_layers],
            adapters_expected: false,
        }
    }

    pub(crate) fn double_blocks(&self) -> usize {
        self.cfg.num_layers
    }

    pub(crate) fn single_blocks(&self) -> usize {
        self.cfg.num_single_layers
    }

    #[cfg(test)]
    pub(crate) fn expects_adapters(&self) -> bool {
        self.adapters_expected
    }

    /// Capture the adapters currently installed on the resident stacks, so materialized blocks carry
    /// them too. Called once, after the resident stack has been built, quantized and adapted, so the
    /// two paths cannot diverge on *which* state, *which* target or *what order*.
    pub(crate) fn capture_adapters(
        &mut self,
        double: &mut [DoubleBlock],
        single: &mut [SingleBlock],
    ) {
        self.double = double.iter_mut().map(BlockAdapters::capture).collect();
        self.single = single.iter_mut().map(BlockAdapters::capture).collect();
        self.adapters_expected = self
            .double
            .iter()
            .chain(self.single.iter())
            .any(|adapters| !adapters.is_empty());
    }

    /// Open a fresh lazy view of the `transformer/` component. Called once per window by
    /// [`run_windowed`](mlx_gen::block_residency::run_windowed).
    pub(crate) fn open(&self) -> Result<Weights> {
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    /// Materialize double block `index` out of `view`, adapted exactly like its resident twin, then
    /// drain the view of precisely the tensors that block read.
    pub(crate) fn materialize_double(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<DoubleBlock> {
        if index >= self.cfg.num_layers {
            return Err(Error::Msg(format!(
                "chroma block stream: double block {index} is outside the {}-block stack",
                self.cfg.num_layers
            )));
        }
        let mut block = DoubleBlock::from_view(view, index, &self.cfg).map_err(|error| {
            Error::Msg(format!(
                "chroma block stream: materialize double block {index}: {error}"
            ))
        })?;
        // The shared drain (SC-15750): the view keeps its own refcounted handle to every tensor the
        // constructor cloned, so draining exactly the accessed keys is what lets the window's drop
        // release them. **Measured on this family it does not move the request peak** — removing
        // both calls leaves the sweep at 14.6932 GiB — because `run_windowed` opens a *fresh* view
        // per window here, so the view's own references die with it at the window boundary anyway,
        // and because the request peak is set by the decode phase rather than by the window. It
        // stays because it is the shared contract's discipline and because it is what makes the
        // bound hold at a configuration where the window IS the binding term; it is recorded as
        // measured-inert here rather than claimed load-bearing on inheritance.
        view.remove_accessed();
        self.replay(
            &self.double[index],
            &mut block,
            &format!("double block {index}"),
        )?;
        Ok(block)
    }

    /// Materialize single block `index`, as [`Self::materialize_double`].
    pub(crate) fn materialize_single(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<SingleBlock> {
        if index >= self.cfg.num_single_layers {
            return Err(Error::Msg(format!(
                "chroma block stream: single block {index} is outside the {}-block stack",
                self.cfg.num_single_layers
            )));
        }
        let mut block = SingleBlock::from_view(view, index, &self.cfg).map_err(|error| {
            Error::Msg(format!(
                "chroma block stream: materialize single block {index}: {error}"
            ))
        })?;
        view.remove_accessed();
        self.replay(
            &self.single[index],
            &mut block,
            &format!("single block {index}"),
        )?;
        Ok(block)
    }

    fn replay<B: StreamBlock>(
        &self,
        captured: &BlockAdapters,
        block: &mut B,
        what: &str,
    ) -> Result<()> {
        if captured.is_empty() && self.adapters_expected {
            // The resident stack HAD adapters and this block's capture is empty. That is not a
            // recoverable state: the streamed block would run its linears without their residuals
            // while its neighbours ran with them — a plausible, wrong image and no error anywhere.
            return Err(Error::Msg(format!(
                "chroma block stream: {what} has no captured adapters, but this transformer was \
                 armed with adapters installed. A streamed block without its residuals would \
                 silently drop part of the LoRA."
            )));
        }
        captured.install(block, what)?;
        captured.verify(block, what)
    }
}

/// The plan pair + cancellation a windowed forward runs under. A `Copy` bundle so the forward
/// signatures do not grow three more positional arguments.
#[derive(Clone, Copy)]
pub(crate) struct ChromaBlockWindow<'a> {
    pub(crate) double: mlx_gen::block_residency::BlockPlan,
    pub(crate) single: mlx_gen::block_residency::BlockPlan,
    pub(crate) cancel: &'a mlx_gen::CancelFlag,
}

impl<'a> ChromaBlockWindow<'a> {
    pub(crate) fn new(
        double_blocks: usize,
        single_blocks: usize,
        size: usize,
        cancel: &'a mlx_gen::CancelFlag,
    ) -> Result<Self> {
        Ok(Self {
            double: mlx_gen::block_residency::BlockPlan::new(double_blocks, size)?,
            single: mlx_gen::block_residency::BlockPlan::new(single_blocks, size)?,
            cancel,
        })
    }
}

/// Evict both resident block stacks after the deferred stream has captured their exact shape. The
/// embedders, Approximator and `proj_out` remain resident.
pub(crate) fn evict_resident_blocks(
    double: &mut Vec<DoubleBlock>,
    single: &mut Vec<SingleBlock>,
    expected_double: usize,
    expected_single: usize,
) -> Result<()> {
    if double.len() != expected_double || single.len() != expected_single {
        return Err(Error::Msg(format!(
            "chroma: cannot finalize a {expected_double}/{expected_single} block stream from \
             resident stacks {}/{}",
            double.len(),
            single.len()
        )));
    }
    double.clear();
    single.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::{DOUBLE_ADAPTER_PATHS, SINGLE_ADAPTER_PATHS};
    use mlx_rs::Array;

    /// A tiny two-double/two-single Chroma stack written to a temp `.safetensors`, so the stream is
    /// exercisable without the 14 GB tier.
    fn cfg() -> ChromaTransformerConfig {
        ChromaTransformerConfig {
            num_layers: 2,
            num_single_layers: 2,
            num_attention_heads: 2,
            attention_head_dim: 4,
            joint_attention_dim: 8,
            ..ChromaTransformerConfig::default()
        }
    }

    fn tensor(shape: Vec<i32>, salt: f32) -> Array {
        let n: i32 = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|k| (k as f32 + salt) * 0.01).collect();
        Array::from_slice(&values, &shape)
    }

    fn fixture(tmp: &tempfile::TempDir, tag: &str) -> std::path::PathBuf {
        let cfg = cfg();
        let inner = cfg.inner_dim() as i32;
        let head = cfg.attention_head_dim as i32;
        let dir = tmp.path().join(format!(
            "chroma-block-stream-{tag}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let mut named: Vec<(String, Array)> = Vec::new();
        for i in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{i}");
            let salt = i as f32;
            for leaf in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
            ] {
                named.push((
                    format!("{p}.{leaf}.weight"),
                    tensor(vec![inner, inner], salt),
                ));
                named.push((format!("{p}.{leaf}.bias"), tensor(vec![inner], salt)));
            }
            for norm in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
                named.push((format!("{p}.attn.{norm}.weight"), tensor(vec![head], salt)));
            }
            for ff in ["ff", "ff_context"] {
                named.push((
                    format!("{p}.{ff}.net.0.proj.weight"),
                    tensor(vec![4 * inner, inner], salt),
                ));
                named.push((
                    format!("{p}.{ff}.net.0.proj.bias"),
                    tensor(vec![4 * inner], salt),
                ));
                named.push((
                    format!("{p}.{ff}.net.2.weight"),
                    tensor(vec![inner, 4 * inner], salt),
                ));
                named.push((format!("{p}.{ff}.net.2.bias"), tensor(vec![inner], salt)));
            }
        }
        for i in 0..cfg.num_single_layers {
            let p = format!("single_transformer_blocks.{i}");
            let salt = 10.0 + i as f32;
            for leaf in ["attn.to_q", "attn.to_k", "attn.to_v"] {
                named.push((
                    format!("{p}.{leaf}.weight"),
                    tensor(vec![inner, inner], salt),
                ));
                named.push((format!("{p}.{leaf}.bias"), tensor(vec![inner], salt)));
            }
            for norm in ["norm_q", "norm_k"] {
                named.push((format!("{p}.attn.{norm}.weight"), tensor(vec![head], salt)));
            }
            named.push((
                format!("{p}.proj_mlp.weight"),
                tensor(vec![4 * inner, inner], salt),
            ));
            named.push((format!("{p}.proj_mlp.bias"), tensor(vec![4 * inner], salt)));
            named.push((
                format!("{p}.proj_out.weight"),
                tensor(vec![inner, 5 * inner], salt),
            ));
            named.push((format!("{p}.proj_out.bias"), tensor(vec![inner], salt)));
        }
        let refs: Vec<(&str, &Array)> = named.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        path
    }

    fn stream(path: &std::path::Path) -> ChromaBlockStream {
        ChromaBlockStream::new(WeightsSource::File(path.to_path_buf()), cfg())
    }

    fn resident(stream: &ChromaBlockStream) -> (Vec<DoubleBlock>, Vec<SingleBlock>) {
        let view = stream.open().unwrap();
        let cfg = cfg();
        let double = (0..cfg.num_layers)
            .map(|i| DoubleBlock::load(&view, i, &cfg).unwrap())
            .collect();
        let single = (0..cfg.num_single_layers)
            .map(|i| SingleBlock::load(&view, i, &cfg).unwrap())
            .collect();
        (double, single)
    }

    fn lora() -> Adapter {
        let inner = cfg().inner_dim() as i32;
        Adapter::Lora {
            a: Array::from_slice(&vec![0.5f32; (inner * 2) as usize], &[inner, 2]),
            b: Array::from_slice(&vec![0.25f32; (2 * inner) as usize], &[2, inner]),
            scale: 1.0,
        }
    }

    /// The drain must remove EXACTLY the block's own tensors, so a constructor that silently stopped
    /// reading one leaves that key visible instead of having the evidence wiped by a prefix sweep.
    #[test]
    fn the_drain_is_exact_and_a_fresh_view_is_independent() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "drain");
        let s = stream(&path);
        let mut view = s.open().unwrap();
        let before = view.len();
        let _ = s.materialize_double(&mut view, 0).unwrap();
        let after = view.len();
        assert!(after < before, "the window must drain what it read");
        assert!(
            view.get("transformer_blocks.1.attn.to_q.weight").is_some(),
            "draining block 0 must not touch block 1"
        );
        // A fresh view is independent of the drained one.
        let mut fresh = s.open().unwrap();
        assert_eq!(fresh.len(), before);
        let _ = s.materialize_double(&mut fresh, 1).unwrap();
        let _ = s.materialize_single(&mut fresh, 0).unwrap();
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn out_of_range_blocks_are_refused_by_name() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "range");
        let s = stream(&path);
        let mut view = s.open().unwrap();
        let double = match s.materialize_double(&mut view, 2) {
            Ok(_) => panic!("a block past the stack must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(double.contains("double block 2"), "{double}");
        let single = match s.materialize_single(&mut view, 9) {
            Ok(_) => panic!("a block past the stack must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(single.contains("single block 9"), "{single}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// Captured residuals travel onto a materialized block **per block** — a LoRA installed on
    /// double block 1 must not appear on double block 0, and must not leak into the single stack.
    #[test]
    fn captured_adapters_are_reinstalled_per_block() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "adapters");
        let mut s = stream(&path);
        let (mut double, mut single) = resident(&s);
        double[1]
            .adaptable_mut(&["attn", "to_q"])
            .unwrap()
            .push(lora());
        single[0].adaptable_mut(&["proj_mlp"]).unwrap().push(lora());
        s.capture_adapters(&mut double, &mut single);
        assert!(s.expects_adapters());

        let mut view = s.open().unwrap();
        let mut d1 = s.materialize_double(&mut view, 1).unwrap();
        assert_eq!(
            d1.adaptable_mut(&["attn", "to_q"])
                .unwrap()
                .adapters()
                .len(),
            1
        );
        let mut view = s.open().unwrap();
        let mut s0 = s.materialize_single(&mut view, 0).unwrap();
        assert_eq!(s0.adaptable_mut(&["proj_mlp"]).unwrap().adapters().len(), 1);
        assert!(s0
            .adaptable_mut(&["attn", "to_q"])
            .unwrap()
            .adapters()
            .is_empty());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// **The missing-capture guard.** A block whose capture is empty while the stack was armed must
    /// be refused, not rendered without its residuals.
    #[test]
    fn a_block_missing_its_captured_adapters_errors_loudly() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "missing");
        let mut s = stream(&path);
        let (mut double, mut single) = resident(&s);
        double[0]
            .adaptable_mut(&["attn", "to_q"])
            .unwrap()
            .push(lora());
        s.capture_adapters(&mut double, &mut single);
        assert!(s.expects_adapters());
        // Block 0 replays.
        let mut view = s.open().unwrap();
        s.materialize_double(&mut view, 0).unwrap();
        // Block 1 has no capture, and the stream refuses rather than dropping the residual.
        let mut view = s.open().unwrap();
        let error = match s.materialize_double(&mut view, 1) {
            Ok(_) => panic!("a block with no captured adapters must be refused when armed"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("no captured adapters") && error.contains("silently drop"),
            "the refusal must name the hazard, got: {error}"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// **The present-but-wrong guard**, the silent class both a missing-capture check and a bare
    /// "did anything install" check miss: the capture names a path, the install runs, and the block
    /// still comes back with the wrong adapter count at that path.
    #[test]
    fn a_present_but_wrong_replay_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "wrong");
        let s = stream(&path);
        let mut view = s.open().unwrap();
        let mut block = DoubleBlock::load(&view, 0, &cfg()).unwrap();
        view.remove_accessed();

        // (a) A capture naming a leaf that is NOT part of the block tree: `install` cannot resolve it
        //     and must refuse rather than leaving the block silently short.
        let bogus = BlockAdapters {
            per_path: vec![("attn.to_out.7".to_owned(), vec![lora()])],
        };
        let error = match bogus.install(&mut block, "double block 0") {
            Ok(()) => panic!("an unresolvable adapter target must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("is absent from a materialized"), "{error}");

        // (b) A replay that resolves but does not take. `install` succeeds; the block is then left
        //     holding a DIFFERENT (present, wrong) adapter set, and only the per-path count check
        //     catches it.
        let captured = BlockAdapters {
            per_path: vec![("attn.to_q".to_owned(), vec![lora(), lora()])],
        };
        captured.install(&mut block, "double block 0").unwrap();
        captured
            .verify(&mut block, "double block 0")
            .expect("a faithful replay verifies");
        block
            .adaptable_mut(&["attn", "to_q"])
            .unwrap()
            .set_adapters(vec![lora()]);
        let error = match captured.verify(&mut block, "double block 0") {
            Ok(()) => panic!("a present-but-wrong replay must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("came back with Some(1) adapters at `attn.to_q`"),
            "{error}"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// With no adapter installed anywhere, materialization is unaffected — the guard must not fire
    /// on the ordinary clean path (the mutation-discriminating other half of the two tests above).
    #[test]
    fn no_adapters_installed_means_no_guard() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "clean");
        let mut s = stream(&path);
        let (mut double, mut single) = resident(&s);
        s.capture_adapters(&mut double, &mut single);
        assert!(!s.expects_adapters());
        let mut view = s.open().unwrap();
        for i in 0..cfg().num_layers {
            let mut block = s.materialize_double(&mut view, i).unwrap();
            assert!(block
                .adaptable_mut(&["attn", "to_q"])
                .unwrap()
                .adapters()
                .is_empty());
        }
        let mut view = s.open().unwrap();
        for i in 0..cfg().num_single_layers {
            s.materialize_single(&mut view, i).unwrap();
        }
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// The block-local adapter path lists must exactly match what `adaptable_mut` resolves, in BOTH
    /// directions: a path dropped from a list means rung 4 stops replaying that adapter, and the
    /// streamed render quietly loses part of the LoRA.
    #[test]
    fn every_listed_adapter_path_resolves_and_nothing_else_does() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "paths");
        let s = stream(&path);
        let (mut double, mut single) = resident(&s);
        for listed in DOUBLE_ADAPTER_PATHS {
            let segments: Vec<&str> = listed.split('.').collect();
            assert!(
                double[0].adaptable_mut(&segments).is_some(),
                "listed double path `{listed}` does not resolve"
            );
        }
        for listed in SINGLE_ADAPTER_PATHS {
            let segments: Vec<&str> = listed.split('.').collect();
            assert!(
                single[0].adaptable_mut(&segments).is_some(),
                "listed single path `{listed}` does not resolve"
            );
        }
        for absent in ["attn.to_out", "ff.net.0", "norm_q", "attn.to_add"] {
            let segments: Vec<&str> = absent.split('.').collect();
            assert!(
                double[0].adaptable_mut(&segments).is_none(),
                "`{absent}` resolves but is absent from DOUBLE_ADAPTER_PATHS"
            );
        }
        for absent in ["attn.to_out.0", "ff.net.2", "proj"] {
            let segments: Vec<&str> = absent.split('.').collect();
            assert!(
                single[0].adaptable_mut(&segments).is_none(),
                "`{absent}` resolves but is absent from SINGLE_ADAPTER_PATHS"
            );
        }
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// A snapshot with the trunk as well as the two block stacks, so a whole
    /// [`crate::transformer::ChromaTransformer`] can be built without the 14 GB tier. Shapes are
    /// nominal — `from_weights` validates presence and the pruned-adaLN invariant, not dimensions.
    fn trunk_fixture(tag: &str) -> (std::path::PathBuf, ChromaTransformerConfig) {
        let path = fixture(tag);
        let mut cfg = cfg();
        cfg.approximator_layers = 2;
        cfg.approximator_hidden_dim = 16;
        cfg.approximator_num_channels = 8;
        let inner = cfg.inner_dim() as i32;
        let hidden = cfg.approximator_hidden_dim as i32;

        let existing = mlx_gen::weights::Weights::from_file(&path).unwrap();
        let mut named: Vec<(String, Array)> = existing
            .keys()
            .map(|k| (k.to_owned(), existing.get(k).unwrap().clone()))
            .collect();
        let linear = |named: &mut Vec<(String, Array)>, prefix: &str, out: i32, input: i32| {
            named.push((format!("{prefix}.weight"), tensor(vec![out, input], 1.0)));
            named.push((format!("{prefix}.bias"), tensor(vec![out], 1.0)));
        };
        linear(&mut named, "x_embedder", inner, cfg.in_channels as i32);
        linear(
            &mut named,
            "context_embedder",
            inner,
            cfg.joint_attention_dim as i32,
        );
        linear(&mut named, "proj_out", cfg.in_channels as i32, inner);
        let p = "distilled_guidance_layer";
        linear(
            &mut named,
            &format!("{p}.in_proj"),
            hidden,
            4 * cfg.approximator_num_channels as i32,
        );
        linear(&mut named, &format!("{p}.out_proj"), inner, hidden);
        for i in 0..cfg.approximator_layers {
            linear(
                &mut named,
                &format!("{p}.layers.{i}.linear_1"),
                hidden,
                hidden,
            );
            linear(
                &mut named,
                &format!("{p}.layers.{i}.linear_2"),
                hidden,
                hidden,
            );
            named.push((format!("{p}.norms.{i}.weight"), tensor(vec![hidden], 1.0)));
        }
        let refs: Vec<(&str, &Array)> = named.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        (path, cfg)
    }

    /// **`quantize` after the stream is armed must refuse, not silently pack nothing** (review of
    /// PR #496).
    ///
    /// The production order is quantize → adapt → arm, and it is correct. But
    /// `ChromaTransformer::quantize` is `pub`, and after arming both stacks are evicted, so its
    /// loops iterate empty vectors: it would return `Ok(())` having packed nothing while every
    /// streamed block was rebuilt dense from a snapshot the caller believes it quantized.
    #[test]
    fn quantizing_after_the_stream_is_armed_is_refused() {
        let (path, cfg) = trunk_fixture("quant-order");
        let build = || {
            let view = mlx_gen::weights::Weights::from_file(&path).unwrap();
            crate::transformer::ChromaTransformer::from_weights(view, cfg)
                .expect("build the tiny transformer")
        };

        let mut armed = build().with_block_stream(WeightsSource::File(path.clone()));
        armed.finalize_block_stream().expect("arm and evict");
        assert_eq!(armed.resident_block_counts(), (0, 0));
        let error = match armed.quantize(8) {
            Ok(()) => panic!("quantize after arming packed nothing and reported success"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("after the block stream is armed"),
            "the refusal must name the ordering, got: {error}"
        );

        // The discriminating control: an UNARMED transformer never reaches this guard. Its
        // `quantize` may still fail on these nominal shapes, but never with the ordering message —
        // without this the assertion above would pass on a `quantize` that always errored.
        let unarmed = build().quantize(8).err().map(|e| e.to_string());
        assert!(
            !unarmed
                .as_deref()
                .is_some_and(|e| e.contains("after the block stream is armed")),
            "an unarmed transformer must not hit the ordering guard, got: {unarmed:?}"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn eviction_is_all_or_nothing_and_shape_checked() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = cfg();
        let path = fixture(&tmp, "evict");
        let s = stream(&path);
        let (mut double, mut single) = resident(&s);
        assert!(evict_resident_blocks(
            &mut double,
            &mut single,
            cfg.num_layers + 1,
            cfg.num_single_layers
        )
        .is_err());
        assert_eq!(
            double.len(),
            cfg.num_layers,
            "a rejected shape must not partially evict"
        );
        evict_resident_blocks(
            &mut double,
            &mut single,
            cfg.num_layers,
            cfg.num_single_layers,
        )
        .unwrap();
        assert!(double.is_empty() && single.is_empty());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn the_two_sub_stacks_get_independent_plans() {
        let cancel = mlx_gen::CancelFlag::default();
        let window = ChromaBlockWindow::new(19, 38, 1, &cancel).unwrap();
        assert_eq!(window.double.n_blocks(), 19);
        assert_eq!(window.double.window_count(), 19);
        assert_eq!(window.single.n_blocks(), 38);
        assert_eq!(window.single.window_count(), 38);
        let wide = ChromaBlockWindow::new(19, 38, 5, &cancel).unwrap();
        assert_eq!(
            wide.double.window_count(),
            4,
            "19 = 3 full windows + a tail"
        );
        assert_eq!(
            wide.single.window_count(),
            8,
            "38 = 7 full windows + a tail"
        );
        assert!(ChromaBlockWindow::new(19, 38, 0, &cancel).is_err());
    }
}
