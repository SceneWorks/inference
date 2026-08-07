//! Ladder rung 4 — **bounded transformer residency** for the Wan DiT block stack (sc-15528).
//!
//! The window lifecycle is NOT here: it is [`mlx_gen::block_residency::run_windowed`] (sc-15750), the
//! shared primitive that also serves Z-Image, Chroma, Qwen and Mage-Flow. This module is only the
//! family-side half — *"how do I rebuild Wan block `n` from the snapshot, with the same quantization
//! the resident block carries"* — which is the part that genuinely differs per family.
//!
//! ## Why re-reading per window is nearly free
//!
//! `Array::load_safetensors` is **lazy per tensor**: the handles exist, the bytes do not. Re-opening
//! `low_noise_model.safetensors` once per window costs a JSON header parse, not 8.4 GB. Only the
//! tensors a window's blocks actually read are materialized, and [`Weights::remove_accessed`] then
//! drops the view's own reference to exactly those, so the window's residency is the blocks
//! themselves and nothing else.
//!
//! ## What this buys on a **dual-expert** config, and why that is the whole point (sc-16354)
//!
//! Every `dual_model` Wan config is `num_layers: 40`, and the Bernini renderer holds **both** experts
//! live across the entire denoise (`bernini.rs` refuses a single-expert snapshot outright). A naive
//! per-expert window bounds the active expert and leaves the idle one's 40 blocks fully resident, so
//! it would buy at most half of what the arithmetic suggests. A **deferred** transformer holds zero
//! blocks: the idle expert costs nothing because there is nothing to be idle. That is why rung 4 on
//! this family is declared over the whole `2 x 40 = 80`-block trunk rather than per expert, and why
//! the request-scope block count a Bernini window is validated against is 80, not 40.
//!
//! ## The cost side, priced honestly
//!
//! A window plan is re-walked **once per forward**, and the guided-velocity modes run several full
//! forwards per step (`VitMode::VaeTxtVitWapg`, the `bernini_image` default, runs 4). At 40 steps that
//! is `40 blocks x 40 steps x 4 passes = 6400` window materializations per render against ONE expert
//! — an order of magnitude above the 240x that `gen_core::block_window`'s own docs use as their
//! cautionary arithmetic. That is a latency consequence to price into the chosen window size, not a
//! correctness problem, and it is the reason the published domain here starts at a wide cadence rather
//! than at 1.
//!
//! ## Adapters are refused, not replayed
//!
//! Z-Image and Chroma install adapters as **forward-time residuals**, so a streamed block can replay
//! the captured list and reproduce its resident twin exactly. Wan does not: `apply_wan_adapters`
//! **merges deltas into the weight map at load** (see [`crate::adapters`]). Re-reading the base file
//! per window would therefore silently produce an *un-adapted* block — correct-looking output from the
//! wrong weights, the exact silent class this epic exists to remove. [`WanBlockStream::new`] refuses
//! to construct over a non-empty adapter set instead, and
//! [`a_merged_adapter_install_refuses_to_stream`](self) pins it.

use std::ops::Range;

use mlx_gen::attention::AttentionBudget;
use mlx_gen::block_residency::BlockPlan;
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterSpec, CancelFlag, Error, Result, WeightsSource};

use crate::config::WanModelConfig;
use crate::transformer::Block;

/// Everything needed to rebuild one Wan transformer block from its snapshot, on demand.
///
/// Cheap to clone: a path, a config and two scalars.
#[derive(Clone)]
pub struct WanBlockStream {
    source: WeightsSource,
    cfg: WanModelConfig,
    /// The load-time quantization the resident stack received, replayed per materialized block so the
    /// streamed weights are byte-identical to the resident ones. `None` for a dense/bf16 tier. On a
    /// **pre-quantized (packed)** snapshot — every shipped Bernini tier — the packed weights come
    /// straight out of `Block::load` via `cfg.quantization`, and this replay is the same documented
    /// no-op the resident path performed.
    quant_bits: Option<i32>,
    /// Ladder rung 3, replayed onto each materialized block so a rung-3 + rung-4 composition executes
    /// the same attention on the streamed path as on the resident one.
    attn_budget: AttentionBudget,
}

impl WanBlockStream {
    /// Declare a streamable stack over a re-openable `source`.
    ///
    /// `adapters` is the load spec's adapter set and **must be empty**: see the module docs. An
    /// in-memory / normalized `Weights` has no re-openable source, so those loads simply do not
    /// construct a stream and rung 4 is declared unavailable for them.
    pub fn new(
        source: WeightsSource,
        cfg: WanModelConfig,
        adapters: &[AdapterSpec],
    ) -> Result<Self> {
        if !adapters.is_empty() {
            return Err(Error::Unsupported(format!(
                "wan block stream: {} adapter file(s) are installed, and Wan MERGES adapter deltas \
                 into the weight map at load rather than installing forward-time residuals. A \
                 streamed block re-read from the base snapshot would silently carry none of them. \
                 Bounded transformer residency is refused on an adapted load.",
                adapters.len()
            )));
        }
        Ok(Self {
            source,
            cfg,
            quant_bits: None,
            attn_budget: AttentionBudget::UNBOUNDED,
        })
    }

    /// The stack depth this stream materializes — read from the model config, never a constant, so a
    /// `num_layers` overlay in `config.json` cannot desynchronize the plan from the checkpoint.
    pub fn n_blocks(&self) -> usize {
        self.cfg.num_layers
    }

    /// Record the load-time quantization so a materialized block reproduces it.
    pub fn set_quant_bits(&mut self, bits: i32) {
        self.quant_bits = Some(bits);
    }

    /// Record rung 3's attention budget so a materialized block executes the same attention.
    pub fn set_attention_budget(&mut self, budget: AttentionBudget) {
        self.attn_budget = budget;
    }

    /// Open a fresh lazy view of the stack's weights. Called once per window by
    /// [`mlx_gen::block_residency::run_windowed`].
    pub fn open(&self) -> Result<Weights> {
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    /// Materialize block `index` out of `view`, quantized and budgeted exactly like its resident twin,
    /// then drain the view of precisely the tensors that block read.
    pub(crate) fn materialize(&self, view: &mut Weights, index: usize) -> Result<Block> {
        if index >= self.n_blocks() {
            return Err(Error::Msg(format!(
                "wan block stream: block {index} is out of range for a {}-block stack",
                self.n_blocks()
            )));
        }
        let mut block = Block::load(view, index, &self.cfg)?;
        // LOAD-BEARING (sc-15750): the view keeps its own refcounted handle to every tensor the
        // constructor cloned. Draining exactly the accessed keys is what makes the window's drop a
        // real release rather than a no-op that still produces correct images.
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            block.quantize(bits, None)?;
        }
        block.set_attention_budget(self.attn_budget);
        Ok(block)
    }

    /// Precompute every block's cross-attention K/V **through the window plan**.
    ///
    /// [`WanTransformer::prepare_cross_kv`](crate::WanTransformer::prepare_cross_kv) reads every
    /// block's `cross_attn.{k,v,norm_k}`, so a deferred stack cannot serve it from a zero-block
    /// transformer. Walking the same plan is what keeps the one-time build bounded: the caches it
    /// produces are `[B, n, text_len, d]` per block — small next to the weights — and they are
    /// legitimately resident for the whole denoise, because they are reused across every step and
    /// every guidance pass. Rebuilding them per window per forward would multiply a once-per-generate
    /// cost by the pass count for no residency saving.
    pub fn prepare_cross_kv_windowed(
        &self,
        context_batch: &mlx_rs::Array,
        plan: &BlockPlan,
        cancel: &CancelFlag,
    ) -> Result<Vec<(mlx_rs::Array, mlx_rs::Array)>> {
        let out = mlx_gen::block_residency::run_windowed(
            plan,
            cancel,
            Vec::with_capacity(self.n_blocks()),
            || self.open(),
            |mut acc: Vec<(mlx_rs::Array, mlx_rs::Array)>,
             view: &mut Weights,
             range: Range<usize>| {
                for index in range {
                    let block = self.materialize(view, index)?;
                    acc.push(block.prepare_kv(context_batch)?);
                }
                Ok(acc)
            },
            |acc: &Vec<(mlx_rs::Array, mlx_rs::Array)>| {
                // The carried K/V are unevaluated graph nodes still referencing the window's block
                // weights; forcing them here is what lets the window's drop free anything at all.
                let arrays: Vec<&mlx_rs::Array> = acc.iter().flat_map(|(k, v)| [k, v]).collect();
                if !arrays.is_empty() {
                    mlx_rs::transforms::eval(arrays)?;
                }
                Ok(())
            },
        )?;
        if out.len() != self.n_blocks() {
            return Err(Error::Msg(format!(
                "wan block stream: windowed cross-K/V produced {} entries for a {}-block stack",
                out.len(),
                self.n_blocks()
            )));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> WanModelConfig {
        let mut c = WanModelConfig::wan21_t2v_1_3b();
        c.dim = 128;
        c.num_heads = 1;
        c.num_layers = 2;
        c.ffn_dim = 256;
        c.freq_dim = 256;
        c.text_dim = 32;
        c.text_len = 8;
        c.in_dim = 16;
        c.out_dim = 16;
        c.vae_z_dim = 16;
        c
    }

    fn fixture() -> std::path::PathBuf {
        std::path::PathBuf::from(format!(
            "{}/tests/fixtures/s5_low.safetensors",
            env!("CARGO_MANIFEST_DIR")
        ))
    }

    /// Wan merges adapter deltas at load, so a streamed block re-read from the base file would carry
    /// none of them. The refusal is the whole guard — there is no replay to test, and a silent
    /// un-adapted stream is the failure this exists to prevent.
    #[test]
    fn a_merged_adapter_install_refuses_to_stream() {
        let adapters = vec![AdapterSpec {
            path: "/nonexistent/lora.safetensors".into(),
            scale: 1.0,
            kind: mlx_gen::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }];
        let error = match WanBlockStream::new(WeightsSource::File(fixture()), tiny_cfg(), &adapters)
        {
            Ok(_) => panic!("an adapted load must not produce a stream"),
            Err(error) => error,
        };
        let text = error.to_string();
        assert!(
            text.contains("MERGES adapter deltas") && text.contains("refused"),
            "the refusal must name the mechanism and the decision, got: {text}"
        );
        // The discriminating control: the same source with no adapters DOES construct, so the
        // assertion above is about the adapter set rather than about the fixture path.
        WanBlockStream::new(WeightsSource::File(fixture()), tiny_cfg(), &[])
            .expect("an unadapted load must stream");
    }

    /// The stack depth comes from the model config, so a `num_layers` overlay moves the plan with the
    /// checkpoint instead of leaving a hardcoded 40 pointing past the end of a shorter stack.
    #[test]
    fn the_stack_depth_comes_from_the_model_config() {
        let mut cfg = tiny_cfg();
        cfg.num_layers = 7;
        let stream = WanBlockStream::new(WeightsSource::File(fixture()), cfg, &[]).expect("stream");
        assert_eq!(stream.n_blocks(), 7);
    }

    /// Materializing block `n` drains EXACTLY the tensors that block read: block 1's keys survive
    /// draining block 0, which is what makes the drain a real check rather than a tautology.
    #[test]
    fn the_drain_is_exact_and_out_of_range_is_refused() {
        let cfg = tiny_cfg();
        let stream = WanBlockStream::new(WeightsSource::File(fixture()), cfg, &[]).expect("stream");
        let mut view = stream.open().expect("open");
        let before = view.len();
        let _block0 = match stream.materialize(&mut view, 0) {
            Ok(block) => block,
            Err(error) => panic!("block 0: {error}"),
        };
        let after_first = view.len();
        assert!(
            after_first < before,
            "materializing block 0 must drain its own tensors ({before} -> {after_first})"
        );
        assert!(
            view.get("blocks.1.modulation").is_some(),
            "draining block 0 must not remove block 1's tensors"
        );
        let _block1 = match stream.materialize(&mut view, 1) {
            Ok(block) => block,
            Err(error) => panic!("block 1: {error}"),
        };
        assert!(view.len() < after_first, "block 1's drain must remove more");

        let error = match stream.materialize(&mut view, 2) {
            Ok(_) => panic!("block 2 is past a 2-block stack"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("out of range"), "{error}");
    }

    /// **The parity contract, executed.** A windowed forward must be *bit-identical* to the resident
    /// one — that is what licenses `MemoryParityContract::Exact` on rung 4 rather than a tolerance.
    ///
    /// Both cadences are checked (1 and 2 over a 2-block stack) so a plan whose ragged tail or whose
    /// window boundary reordered the block application would redden, and the cross-K/V is built BOTH
    /// ways so the windowed cache build is covered by the same equality rather than assumed.
    ///
    /// This is weights-free: it runs on the checked-in 2-block fixture, so the mechanism is gated in
    /// CI rather than only on an operator's Mac.
    #[test]
    fn a_windowed_forward_is_bit_identical_to_the_resident_one() {
        let cfg = tiny_cfg();
        let weights = Weights::from_file(fixture()).expect("fixture");
        let resident = crate::WanTransformer::from_weights(&weights, &cfg).expect("resident");
        let deferred =
            crate::WanTransformer::from_weights_deferred(&weights, &cfg).expect("deferred");
        assert!(!resident.is_deferred() && deferred.is_deferred());
        assert_eq!(
            deferred.num_blocks(),
            cfg.num_layers,
            "a deferred stack must still report the checkpoint's depth"
        );

        let stream =
            WanBlockStream::new(WeightsSource::File(fixture()), cfg.clone(), &[]).expect("stream");
        let cancel = CancelFlag::default();

        // The fixture's own conditioning + latent, embedded exactly as the denoise loop embeds them,
        // so this drives the real seam rather than a synthetic shape that happens to type-check.
        let ctx = weights.require("ctx_cond").expect("ctx_cond").clone();
        let embedded = resident.embed_text(&ctx).expect("embed_text");
        let cross_resident = resident.prepare_cross_kv(&embedded).expect("cross kv");
        let cross_windowed = stream
            .prepare_cross_kv_windowed(
                &embedded,
                &BlockPlan::new(cfg.num_layers, 1).expect("plan"),
                &cancel,
            )
            .expect("windowed cross kv");
        for (index, (a, b)) in cross_resident.iter().zip(cross_windowed.iter()).enumerate() {
            assert_eq!(
                bytes(&a.0),
                bytes(&b.0),
                "block {index} windowed cross-K differs from the resident one"
            );
            assert_eq!(bytes(&a.1), bytes(&b.1), "block {index} cross-V differs");
        }

        let latent = weights.require("init_noise").expect("init_noise").clone();
        let (tokens, grid) = resident
            .patch_embed_tokens(&latent)
            .expect("patch_embed_tokens");
        let (cos, sin) = resident.prepare_rope(grid).expect("rope");

        let expected = resident
            .forward_packed(&tokens, 500.0, &cross_resident, &cos, &sin)
            .expect("resident forward");
        for window in [1_usize, 2] {
            let plan = BlockPlan::new(cfg.num_layers, window).expect("plan");
            let got = deferred
                .forward_packed_windowed(
                    &tokens,
                    500.0,
                    &cross_windowed,
                    &cos,
                    &sin,
                    &stream,
                    &plan,
                    &cancel,
                )
                .expect("windowed forward");
            assert_eq!(
                bytes(&expected),
                bytes(&got),
                "window {window} must be bit-identical to the resident forward"
            );
        }
    }

    /// A plan or a stream sized for a different stack is refused rather than silently walking a
    /// prefix — the shape that would leave the tail blocks un-applied and still produce an image.
    #[test]
    fn a_mismatched_plan_or_cross_kv_is_refused() {
        let cfg = tiny_cfg();
        let weights = Weights::from_file(fixture()).expect("fixture");
        let deferred =
            crate::WanTransformer::from_weights_deferred(&weights, &cfg).expect("deferred");
        let stream =
            WanBlockStream::new(WeightsSource::File(fixture()), cfg.clone(), &[]).expect("stream");
        let cancel = CancelFlag::default();
        let tokens = mlx_rs::Array::zeros::<f32>(&[1, 4, cfg.dim as i32]).expect("tokens");
        let (cos, sin) = deferred.prepare_rope((1, 2, 2)).expect("rope");
        let kv = (
            mlx_rs::Array::zeros::<f32>(&[1, 1, 1, 1]).expect("k"),
            mlx_rs::Array::zeros::<f32>(&[1, 1, 1, 1]).expect("v"),
        );

        let short_plan = BlockPlan::new(1, 1).expect("plan");
        let error = match deferred.forward_packed_windowed(
            &tokens,
            0.0,
            &[kv.clone(), kv.clone()],
            &cos,
            &sin,
            &stream,
            &short_plan,
            &cancel,
        ) {
            Ok(_) => panic!("a 1-block plan over a 2-block stack must be refused"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("covers 1 blocks"), "{error}");

        let plan = BlockPlan::new(cfg.num_layers, 1).expect("plan");
        let error = match deferred.forward_packed_windowed(
            &tokens,
            0.0,
            &[kv],
            &cos,
            &sin,
            &stream,
            &plan,
            &cancel,
        ) {
            Ok(_) => panic!("one cross-K/V pair for a 2-block stack must be refused"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("one cross-K/V pair per block"),
            "{error}"
        );
    }

    /// Raw bytes of an evaluated array, for bit-identity assertions.
    fn bytes(a: &mlx_rs::Array) -> Vec<u8> {
        let a = a.as_dtype(mlx_rs::Dtype::Float32).expect("f32");
        a.eval().expect("eval");
        a.as_slice::<f32>()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    /// A fresh view per window is independent: the drain of one view cannot starve the next.
    #[test]
    fn each_window_opens_an_independent_view() {
        let stream =
            WanBlockStream::new(WeightsSource::File(fixture()), tiny_cfg(), &[]).expect("stream");
        let mut first = stream.open().expect("open");
        assert!(stream.materialize(&mut first, 0).is_ok(), "block 0");
        let second = stream.open().expect("reopen");
        assert!(
            second.get("blocks.0.modulation").is_some(),
            "a re-opened view must carry block 0 again"
        );
    }
}
