//! The **NVFP4 precision seam** for the Krea 2 single-stream DiT trunk (sc-12110, epic 11037).
//!
//! [`crate::transformer::Krea2Transformer`] loads its projections through
//! [`crate::loader::linear_detect`] by default — dense bf16, or the MLX-packed q4/q8 dequant-on-forward
//! [`crate::quant::QLinear`]. This module adds the seam that lets the SAME trunk serve those projections
//! through [`Nvfp4Linear`] instead — the sc-11041 packed-forward NVFP4 path — so a **real Krea 2 Turbo
//! denoise** can be run end-to-end on the FP4 tensor cores and compared against both epic baselines.
//!
//! This is a direct port of `candle-gen-sana`'s `nvfp4_dit` (sc-11045), which established the pattern.
//! Krea is the epic's **validation vehicle** (sc-12110): Michael redirected SC#1/SC#2 here because
//! SANA's Mix-FFN is convolutional — its linears are 0.20% of block time, capping any end-to-end
//! multiple at ~1.002× — whereas **Krea's DiT is 100% linear GEMM with zero `Conv2d`**, so the NVFP4
//! lane reaches essentially all parameterized compute.
//!
//! Three things live here:
//!
//! 1. [`DitPlan`] — how to serve the trunk's projections: the byte-unchanged default (dense/packed via
//!    `linear_detect`), or NVFP4 under a [`Nvfp4Quant`] regime (the sc-11038 per-layer mixed policy, or
//!    a blanket W4A4/W4A16 for a controlled bench).
//! 2. [`LayerRole`] — the structural facts about a projection that its dotted key cannot carry,
//!    threaded from the loader (which knows the trunk's topology) into the shared substring policy.
//! 3. [`ActProbe`] — a per-layer, per-step **activation-outlier sparsity** recorder, so the
//!    benign→W4A4 / outlier→W4A16 partition can be re-derived against **Krea's** naming from live
//!    activations rather than inherited from SANA's.
//!
//! # Why Krea needs [`LayerRole`] more than SANA did
//!
//! The shared policy ([`ActPrecision::for_outlier_layer_with`]) is substring-based and was tuned on
//! SANA's diffusers naming. **Three of its anchors do not exist in Krea's checkpoint**, and every gap
//! fails in the *unsafe* direction (an outlier-carrying layer silently landing on W4A4):
//!
//! * **`attn2` / `cross_att*`** — Krea has **no cross-attention at all**. It is a *single-stream* DiT:
//!   the fused text context is **concatenated onto the image token sequence** (`combined = [ctx ; img]`)
//!   and read by ordinary self-attention. There is no projection named `attn2` to guard.
//! * **`caption_projection`** — Krea's text→DiT ingest is named `txt_in.linear_{1,2}`, fed by the
//!   `text_fusion` stack that aggregates the raw Qwen3-VL hidden states. Neither matches.
//! * **`proj_out`** — Krea's trunk head is `final_layer.linear`. [`names_final_proj`] cannot fire on it.
//!   (Krea's *only* `proj_out` is a control-branch layer nested under `blocks.{i}`, which the anchor
//!   correctly declines — verified, and the reason the anchor is safe to leave alone here.)
//!
//! So Krea states these facts explicitly through [`LayerRole`] rather than sharpening substrings
//! until they happen to fit a second provider — the seam sc-11045 built for exactly this, and the lesson
//! sc-12140 records. **Measured vindication:** `final_layer.linear` really does measure
//! [`OutlierClass::Dense`] on real activations (crush **909×**). It is guarded *only* because the loader
//! states [`LayerRole::final_proj`]; the name-only anchor would have left it on W4A4.
//!
//! # The finding that is not about naming at all (sc-12110)
//!
//! Measuring the real trunk did not just expose naming gaps — it refuted the policy's underlying
//! *model* of where massive activations live. The sc-11038 policy assumes they arrive with the
//! **caption** and can be contained by guarding a named block. On Krea, the first measurement under
//! that assumption gave **209 layers at W4A4, of which 59 measured Dense** — and the violations were
//! concentrated in the compute bulk, not the caption path:
//!
//! * **`ff.down` was Dense in 28/28 blocks** and **`attn.to_out.0` in 21/28** — 45 of the 59. Both read
//!   a **post-nonlinearity intermediate** (`silu(gate(x))·up(x)`; `attn_out·sigmoid(gate(x))`), i.e. a
//!   product of two unbounded branches with no normalization before the next GEMM.
//! * Every projection reading a **normalized block input** (`attn.to_{q,k,v,gate}`, `ff.{gate,up}`) was
//!   benign from block 4 onward.
//!
//! So the real rule on Krea is **normalized inputs are benign; post-nonlinearity intermediates are
//! not** — orthogonal to captions, and invisible on SANA because SANA's FFN is a `GLUMBConv` and never
//! entered the linear lane. That is [`LayerRole::is_post_nonlinearity`], and it is why the partition is
//! re-derived here by measurement instead of inherited.
//!
//! **What is and is not quantized.** The seam covers the trunk's GEMM projections: the 28 single-stream
//! blocks' `attn.{to_q,to_k,to_v,to_gate,to_out.0}` + `ff.{gate,up,down}`, the `text_fusion`
//! layerwise/refiner blocks' equivalents, `img_in`, `txt_in.linear_{1,2}` and `final_layer.linear` —
//! 260 projections on Krea 2 Turbo. It deliberately does **not** cover:
//!
//! * the timestep / modulation embedders (`time_embed.linear_{1,2}`, `time_mod_proj`), whose `[B, …]`
//!   batch-1 shapes give the FP4 GEMM nothing to win while M-padding to 16 would dominate (the same
//!   exclusion SANA made);
//! * `text_fusion.projector`, a `[1, num_layers]` collapse whose `N = 1` is ineligible for the cuBLASLt
//!   FP4 path anyway (it would fall back at runtime; excluding it keeps the report honest).
//!
//! Everything degrades cleanly: an [`Nvfp4Linear`] on a non-`sm_120` device, on CPU, or on a non-cuda
//! build transparently serves the dequant→bf16 fallback (sc-11041), so a `DitPlan::nvfp4(..)` trunk
//! still *runs* everywhere — it just does not light the FP4 cores. That is the SC#4 Blackwell-only
//! gate, observed at model level by [`Nvfp4Report::fp4_lit`].

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::lock_recover;
use candle_gen::quant::{ActPrecision, Nvfp4Linear, Nvfp4Regime, OutlierClass, OutlierSparsity};

/// How the trunk should serve one projection's activations when running NVFP4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Quant {
    /// The **sc-11038 mixed-precision policy** (the shipping default): the outlier-carrying class runs
    /// W4A16 (bf16 activation), the benign compute-bulk runs W4A4. Classified by
    /// [`ActPrecision::for_outlier_layer_with`] on the projection's dotted key, plus the structural
    /// facts Krea's loader threads through [`LayerRole`] — see [`DitPlan::act_for`].
    Mixed,
    /// Blanket W4A4 on every eligible projection — ignores the outlier policy. For a controlled
    /// throughput/stability bench of the FP4 compute path, **not** a shipping regime.
    BlanketW4A4,
    /// Blanket W4A16 on every eligible projection — the NVFP4 *storage* tier (weights packed, bf16
    /// activation, no FP4 compute). The stability-fallback default.
    BlanketW4A16,
}

/// The **structural facts** about a Krea projection that its dotted key cannot carry, threaded from the
/// loader — which knows the trunk's topology — into the shared substring policy (sc-11045 pattern,
/// sc-12110 for Krea).
///
/// This is the seam that keeps the policy honest across providers. Rather than widen a substring until
/// it happens to fit Krea too (and mis-fire on a third provider), the provider states the fact. Every
/// flag defaults to `false`, i.e. "an ordinary interior compute-bulk projection".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayerRole {
    /// This projection lives in Krea's **first or last** single-stream transformer block — the edges the
    /// sc-11038 policy keeps on bf16 activation. The shared policy's `blocks.0.` clause does match
    /// Krea's `transformer_blocks.0.`, but its `last_block`/`final_block` markers do not match
    /// `transformer_blocks.27.`, so the last-block half is applied here where the block count is known.
    pub is_edge_block: bool,
    /// This projection is the trunk's **final output projection** (the head) — Krea's
    /// `final_layer.linear` `[6144 → 64]`.
    ///
    /// **Krea cannot rely on the name-only fallback here** ([`names_final_proj`] anchors on a trailing
    /// `proj_out` segment, and Krea's head is not spelled that way), so the loader threads this
    /// explicitly. That is precisely the defect class sc-12140 records: a name-only anchor that silently
    /// does not fire leaves the trunk head — measured Dense on SANA, crush 438× — on W4A4.
    pub is_final_proj: bool,
    /// This projection **reads text-encoder-derived context** — Krea's analogue of the class the shared
    /// policy names `caption_projection` + `attn2`, neither of which exists here (see the [module
    /// docs](self)).
    ///
    /// Krea's caption-reading surface is the `text_fusion` stack (which consumes the **raw stacked
    /// Qwen3-VL hidden states** — the massive-activation source itself) and the `txt_in.linear_{1,2}`
    /// ingest that projects the fused context into DiT width. On SANA the equivalent class measured
    /// Dense with per-block crush ratios up to 5124×, so it is guarded here by default and the guard is
    /// **checked by measurement**, not assumed — see `nvfp4_krea_dit_real_activation_outlier_sparsity`.
    ///
    /// Note this guard is nearly free for SC#1: `text_fusion` is 4 blocks at width 2560 and `txt_in` is
    /// 2 layers, against 28 single-stream blocks at width 6144 — a rounding error of the trunk's GEMM
    /// time. The compute bulk (`transformer_blocks.{i}`) is **not** in this class and is the layer set
    /// SC#1 actually rides on.
    ///
    /// **Measured, not assumed** (sc-12110): `text_fusion.layerwise_blocks.0.attn.to_out.0` measures
    /// Dense with a **40145× crush** — the largest in the trunk. The guard is earning its keep.
    pub is_context_read: bool,
    /// This projection reads a **post-nonlinearity intermediate** activation rather than a normalized
    /// block input — Krea's `attn.to_out.0` (which reads the sigmoid-gated attention output) and
    /// `ff.down` (which reads `silu(gate(x)) · up(x)`).
    ///
    /// # This is sc-12110's central partition finding, and it has no SANA precedent
    ///
    /// The sc-11038 policy assumed massive activations enter a DiT through the **caption** and are
    /// therefore containable by guarding a *named block* (`attn2`, `caption_projection`). On Krea that
    /// assumption fails structurally, and the first measurement on real activations showed it:
    /// **209 layers assigned W4A4, of which 59 measured Dense** — and 45 of those 59 were exactly these
    /// two leaves, recurring in essentially every block:
    ///
    /// | leaf | activation it reads | Dense blocks | worst crush |
    /// |---|---|---:|---:|
    /// | `ff.down` | `silu(gate(x)) · up(x)` | **28 / 28** | 3107× |
    /// | `attn.to_out.0` | `attn_out · sigmoid(gate(x))` | 21 / 28 | 686× |
    /// | `ff.gate` / `ff.up` | RMSNorm(x) — a *block input* | 6 / 56 | — |
    /// | `attn.to_{q,k,v,gate}` | RMSNorm(x) — a *block input* | 3 / 28 each | — |
    ///
    /// The pattern is not about captions at all: **normalized block inputs are benign; products of two
    /// unbounded nonlinear branches are not.** A SwiGLU intermediate multiplies two learned projections
    /// with no normalization between them and the next GEMM, so its dynamic range is the *product* of
    /// two heavy tails — which is precisely the sc-7702 mechanism (one outlier crushes its 16-block's
    /// co-located channels to E2M1 zero). SANA never surfaced this because its FFN is a `GLUMBConv`,
    /// i.e. not in the linear lane at all.
    ///
    /// **Why this is good news for SC#1 anyway.** The guarded leaves are the *low-N* ones: `ff.down` is
    /// `[6144 ← 16384]` (N=6144) and `to_out.0` is `[6144 ← 6144]`. The layers that stay on W4A4 include
    /// `ff.gate`/`ff.up` at `[16384 ← 6144]` — **N=16384**, the widest GEMMs in the trunk and exactly
    /// the ones the ~1/N quantizer-amortization argument depends on. The partition removes the layers
    /// that would have collapsed while keeping the ones the throughput case rests on.
    pub is_post_nonlinearity: bool,
}

impl LayerRole {
    /// An interior compute-bulk projection: not an edge block, not the head, not context-reading.
    pub fn interior() -> Self {
        Self::default()
    }

    /// An interior projection in Krea's first/last single-stream transformer block.
    pub fn edge_block(is_edge_block: bool) -> Self {
        Self {
            is_edge_block,
            ..Self::default()
        }
    }

    /// The trunk's **final output projection** (`final_layer.linear`).
    pub fn final_proj() -> Self {
        Self {
            is_final_proj: true,
            ..Self::default()
        }
    }

    /// A projection that reads text-encoder-derived context (`text_fusion.*`, `txt_in.*`).
    pub fn context_read() -> Self {
        Self {
            is_context_read: true,
            ..Self::default()
        }
    }

    /// A projection that reads a post-nonlinearity intermediate (`attn.to_out.0`, `ff.down`).
    pub fn post_nonlinearity() -> Self {
        Self {
            is_post_nonlinearity: true,
            ..Self::default()
        }
    }

    /// True iff `leaf` names a projection whose **input** is a post-nonlinearity intermediate rather
    /// than a normalized block input — see [`Self::is_post_nonlinearity`].
    ///
    /// Structural, so it holds for the `text_fusion` blocks as well as the single-stream ones (both are
    /// built from the same `GatedAttention` + `SwiGlu` modules).
    fn names_post_nonlinearity(name: &str) -> bool {
        name.ends_with(".attn.to_out.0") || name.ends_with(".ff.down")
    }

    /// The role the shipping loader assigns `name` on a Krea trunk of `num_layers` single-stream blocks
    /// — the **single source of truth** for the trunk's topology facts.
    ///
    /// Shared by [`crate::transformer::Krea2Transformer::load_planned`] and the validation harness, so a
    /// report can never cross the measured class against a *different* partition than the one the loader
    /// actually built (the drift sc-11045's Sana harness invited by re-deriving roles inline).
    pub fn for_krea_layer(name: &str, num_layers: usize) -> Self {
        let leading_edge = (0..KREA_LEADING_EDGE_BLOCKS)
            .any(|i| name.starts_with(&format!("transformer_blocks.{i}.")));
        Self {
            is_edge_block: leading_edge
                || name.starts_with(&format!(
                    "transformer_blocks.{}.",
                    num_layers.saturating_sub(1)
                )),
            is_final_proj: name == "final_layer.linear",
            is_context_read: name.starts_with("text_fusion.") || name.starts_with("txt_in."),
            is_post_nonlinearity: Self::names_post_nonlinearity(name),
        }
    }
}

/// Krea 2 Turbo's single-stream block count — the [`DitPlan::num_layers`] default, so a plan built
/// without a config still names the right last block for the shipping model.
const DEFAULT_NUM_LAYERS: usize = 28;

/// How many **leading** single-stream blocks are held at W4A16 (blocks `0..KREA_LEADING_EDGE_BLOCKS`).
///
/// **Four, from measurement — not from the spike's prose** (sc-12110). The sc-11038 policy said "first
/// **two** & last"; on real Krea activations that is not enough. Probing the baseline trunk across a
/// live denoise, the leading blocks carry caption-derived outliers on their *block inputs* — not just
/// on the post-nonlinearity sites the rest of the trunk shows:
///
/// | block | Dense leaves at W4A4 |
/// |---|---|
/// | 1 | all 8 (`attn.to_{q,k,v,gate,out.0}` + `ff.{gate,up,down}`; min benign 0.722, crush 686×) |
/// | 2 | 6 (`attn.to_{q,k,v,gate,out.0}` + `ff.down`; min benign 0.962) |
/// | 3 | 6 (same set; min benign 0.973) |
/// | 4+ | 2 (`attn.to_out.0` + `ff.down` only — the post-nonlinearity class) |
///
/// So the caption's massive activations wash out of the *block inputs* by block 4, and blocks 0–3 are
/// guarded wholesale. This is Krea-specific and structural: it is a **single-stream** DiT — the text
/// context is concatenated onto the image sequence rather than read through a separate cross-attention
/// block, so the caption's activations enter the compute bulk directly and decay along the stack. There
/// is no `attn2` to guard instead, which is exactly why the shared substring policy cannot express this.
const KREA_LEADING_EDGE_BLOCKS: usize = 4;

/// How to serve the trunk's projections (sc-12110). Default: the pre-existing `linear_detect` path —
/// the byte-unchanged baseline (dense bf16, or MLX-packed q4/q8).
#[derive(Clone)]
pub struct DitPlan {
    quant: Option<Nvfp4Quant>,
    probe: Option<Arc<ActProbe>>,
    checked: bool,
    /// The trunk's single-stream block count, used to name the **last** edge block. Set from the config
    /// by [`crate::transformer::Krea2Transformer::load_planned`] so the loader and any harness agree.
    num_layers: usize,
}

impl Default for DitPlan {
    fn default() -> Self {
        Self {
            quant: None,
            probe: None,
            checked: false,
            num_layers: DEFAULT_NUM_LAYERS,
        }
    }
}

impl DitPlan {
    /// The baseline trunk — exactly what [`crate::transformer::Krea2Transformer::load`] builds.
    pub fn baseline() -> Self {
        Self::default()
    }

    /// Bind the plan to a trunk of `num_layers` single-stream blocks (so `is_edge_block` names the right
    /// last block). Called by the loader from the config; a harness building a plan by hand for the
    /// shipping Turbo trunk can rely on the [`DEFAULT_NUM_LAYERS`] default.
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// The trunk block count this plan is bound to (see [`Self::with_num_layers`]).
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// The [`LayerRole`] this plan assigns `name`, derived from the trunk topology it is bound to.
    pub fn role_for(&self, name: &str) -> LayerRole {
        LayerRole::for_krea_layer(name, self.num_layers)
    }

    /// The activation precision this plan assigns `name`, deriving the [`LayerRole`] from the trunk
    /// topology — **the form the loader uses**, so the role is never stated twice.
    pub fn act_for_layer(&self, name: &str) -> ActPrecision {
        self.act_for(name, self.role_for(name))
    }

    /// Serve every eligible projection through [`Nvfp4Linear`] under `quant`.
    pub fn nvfp4(quant: Nvfp4Quant) -> Self {
        Self {
            quant: Some(quant),
            ..Self::default()
        }
    }

    /// Attach an [`ActProbe`]: every projection records the outlier sparsity of its **input
    /// activation** on each forward. Works on the baseline plan too — that is how the *unperturbed* real
    /// activations are captured (the spike's residual gate wants the true activation distribution, not
    /// one already shaped by quantization).
    pub fn with_probe(mut self, probe: Arc<ActProbe>) -> Self {
        self.probe = Some(probe);
        self
    }

    /// Route every NVFP4 projection through [`Nvfp4Linear::forward_checked`] — the sc-11044 NaN/inf
    /// guard — so a non-finite tensor **fails loud at the layer that produced it**, on every step,
    /// rather than silently propagating through the denoise (SC#3). Costs one scalar reduction per
    /// projection per forward; off by default.
    pub fn checked(mut self) -> Self {
        self.checked = true;
        self
    }

    /// True iff this plan serves projections through NVFP4.
    pub fn is_nvfp4(&self) -> bool {
        self.quant.is_some()
    }

    /// The probe attached to this plan, if any.
    pub(crate) fn probe(&self) -> Option<&Arc<ActProbe>> {
        self.probe.as_ref()
    }

    /// The activation precision this plan assigns `name`, given the structural facts Krea's loader knows
    /// and a dotted key cannot carry.
    ///
    /// [`ActPrecision::for_outlier_layer_with`] carries the shared substring policy; the three Krea
    /// facts are threaded in here because **only the provider knows them** and all three of the shared
    /// policy's corresponding anchors miss on Krea's naming (see the [module docs](self)):
    ///
    /// * `is_edge_block` — Krea's **last** block (`transformer_blocks.27.`), which the shared policy's
    ///   `last_block`/`final_block` markers do not match.
    /// * `is_final_proj` — Krea's head is `final_layer.linear`; [`names_final_proj`] anchors on a
    ///   trailing `proj_out` segment and will **not** fire (sc-12140).
    /// * `is_context_read` — Krea's `text_fusion.*` / `txt_in.*`; the shared policy's `caption_projection`
    ///   and `attn2` / `cross_att*` anchors have no Krea counterpart (it is a single-stream DiT with no
    ///   cross-attention).
    ///
    /// Public so a validation harness can ask what the shipping policy *would* assign a layer while
    /// probing a **baseline** trunk (sc-12110's partition gate measures unquantized activations, then
    /// crosses the measured class against this assumed one).
    pub fn act_for(&self, name: &str, role: LayerRole) -> ActPrecision {
        match self.quant {
            Some(Nvfp4Quant::BlanketW4A4) => ActPrecision::W4A4,
            Some(Nvfp4Quant::BlanketW4A16) => ActPrecision::W4A16,
            // Mixed: the shared policy, plus the structural facts only the loader knows. Each of these
            // three was verified against real Krea activations (sc-12110) — none is inherited belief:
            //   * is_edge_block      → blocks 0..3 measured Dense on their block INPUTS (crush 686×);
            //   * is_context_read    → text_fusion `to_out.0` measured Dense at crush 40145×;
            //   * is_post_nonlinearity → `ff.down` measured Dense in 28/28 blocks.
            Some(Nvfp4Quant::Mixed) => {
                if role.is_edge_block || role.is_context_read || role.is_post_nonlinearity {
                    ActPrecision::W4A16
                } else {
                    ActPrecision::for_outlier_layer_with(name, role.is_final_proj)
                }
            }
            None => ActPrecision::W4A16,
        }
    }
}

/// One recorded activation measurement: the sparsity of the tensor entering `layer` at `step`.
#[derive(Clone, Debug)]
pub struct ActRecord {
    /// The projection's dotted key (e.g. `transformer_blocks.7.attn.to_q`).
    pub layer: String,
    /// The denoise step index the recorder was set to ([`ActProbe::set_step`]).
    pub step: usize,
    /// The activation-precision the plan assigned this projection — so a report can cross the
    /// *measured* class against the *assumed* partition.
    pub act: ActPrecision,
    /// The measured outlier sparsity of the input activation.
    pub sparsity: OutlierSparsity,
}

/// Records per-layer, per-step activation-outlier sparsity across a live denoise (sc-11045 pattern).
///
/// The spike (sc-11038) established that NVFP4 W4A4 damage scales with activation-outlier **sparsity**
/// and partitioned layers on that basis — but only ever measured *synthetic* activations, and sc-11045
/// only ever measured **SANA's** layers. This probe re-closes that gate on Krea: attach it to a
/// [`DitPlan`], run a real denoise, then read [`Self::records`] to see whether every layer the policy
/// sends to W4A4 actually measures W4A4-viable **on Krea's naming and topology**.
///
/// Instrumentation, not a hot path: each measurement moves the activation to host f32
/// ([`OutlierSparsity::from_tensor`]), so a probed denoise runs far slower than an unprobed one. Never
/// attach a probe to a timed run.
#[derive(Default)]
pub struct ActProbe {
    step: AtomicUsize,
    tau: Mutex<f32>,
    records: Mutex<Vec<ActRecord>>,
}

impl ActProbe {
    /// A probe at [`OutlierSparsity::DEFAULT_TAU`], step 0.
    pub fn new() -> Self {
        Self {
            step: AtomicUsize::new(0),
            tau: Mutex::new(OutlierSparsity::DEFAULT_TAU),
            records: Mutex::new(Vec::new()),
        }
    }

    /// A probe with an explicit outlier multiplier `tau`.
    pub fn with_tau(tau: f32) -> Self {
        Self {
            tau: Mutex::new(tau),
            ..Self::new()
        }
    }

    /// Stamp subsequent measurements with denoise step `step`. The caller drives this from its sampler
    /// loop (the trunk itself has no notion of a step).
    pub fn set_step(&self, step: usize) {
        self.step.store(step, Ordering::Relaxed);
    }

    /// Every measurement recorded so far, in capture order.
    pub fn records(&self) -> Vec<ActRecord> {
        lock_recover(&self.records).clone()
    }

    /// Drop all recorded measurements (keeps the step/tau settings).
    pub fn clear(&self) {
        lock_recover(&self.records).clear();
    }

    /// Measure `x` and file it under `layer` at the current step. Errors from the measurement are
    /// propagated — a probe that cannot measure should fail the run, not silently under-report.
    fn record(&self, layer: &str, act: ActPrecision, x: &Tensor) -> Result<()> {
        let tau = *lock_recover(&self.tau);
        let sparsity = OutlierSparsity::from_tensor(x, tau)?;
        lock_recover(&self.records).push(ActRecord {
            layer: layer.to_string(),
            step: self.step.load(Ordering::Relaxed),
            act,
            sparsity,
        });
        Ok(())
    }
}

/// A trunk projection served through [`Nvfp4Linear`], plus its instrumentation (name / probe /
/// NaN-guard flag) — the `Nvfp4` arm of [`crate::quant::QLinear`].
pub struct Nvfp4Proj {
    inner: Box<Nvfp4Linear>,
    name: String,
    probe: Option<Arc<ActProbe>>,
    checked: bool,
    act: ActPrecision,
}

impl Nvfp4Proj {
    pub(crate) fn new(inner: Nvfp4Linear, name: &str, plan: &DitPlan, act: ActPrecision) -> Self {
        Self {
            inner: Box::new(inner),
            name: name.to_string(),
            probe: plan.probe.clone(),
            checked: plan.checked,
            act,
        }
    }

    /// `y = x·Wᵀ (+ b)` through the NVFP4 path. Records the input activation first when a probe is
    /// attached, then runs the forward (through the NaN guard when the plan asked for it).
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(p) = &self.probe {
            p.record(&self.name, self.act, x)?;
        }
        if self.checked {
            self.inner.forward_checked(x)
        } else {
            self.inner.forward(x)
        }
    }

    /// The underlying NVFP4 linear (for report accounting).
    pub(crate) fn linear(&self) -> &Nvfp4Linear {
        &self.inner
    }
}

/// A **probe-only** wrapper over a baseline projection: records the input activation, then delegates.
///
/// This is how the partition gate measures *unperturbed* activations — the baseline trunk's real
/// distribution, unshaped by any quantization. `act` is stamped with what the **shipping mixed policy
/// would assign**, so a summary can cross measured-vs-assumed without re-deriving roles.
pub struct ProbedProj {
    inner: Box<crate::quant::QLinear>,
    name: String,
    probe: Arc<ActProbe>,
    act: ActPrecision,
}

impl ProbedProj {
    pub(crate) fn new(
        inner: crate::quant::QLinear,
        name: &str,
        probe: Arc<ActProbe>,
        act: ActPrecision,
    ) -> Self {
        Self {
            inner: Box::new(inner),
            name: name.to_string(),
            probe,
            act,
        }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.probe.record(&self.name, self.act, x)?;
        self.inner.forward(x)
    }
}

/// Model-level NVFP4 accounting over a built trunk (sc-12110 SC#6 / SC#4).
///
/// Byte-accounting, not `nvidia-smi`: it sums the *actual* resident weight buffers the trunk holds, so
/// it is immune to GPU contention and to allocator/workspace noise. That is the same technique
/// sc-11041 used to prove SC#6 at layer level, lifted to the whole model.
#[derive(Clone, Debug, Default)]
pub struct Nvfp4Report {
    /// Projections served through [`Nvfp4Linear`].
    pub n_quantized: usize,
    /// Of those, how many actually run the FP4 tensor-core GEMM (`sm_120` + W4A4 + eligible shape).
    /// Zero on non-Blackwell — the observable form of the SC#4 gate.
    pub fp4_lit: usize,
    /// Of those, how many serve the dequant→bf16 path (W4A16 override, or the capability fallback).
    pub dequant_bf16: usize,
    /// Summed **packed NVFP4 footprint** (E2M1 nibbles + UE4M3 block scales) of the quantized weights.
    ///
    /// A property of the **format**, not of the run: the packed host container is retained in every
    /// regime, so this is identical whether or not anything is packed on-device. Use
    /// [`Self::resident_bytes`] for what the run actually costs in VRAM.
    pub nvfp4_bytes: usize,
    /// Summed bf16 footprint those same weights would occupy dense — the SC#6 comparison baseline.
    pub bf16_bytes: usize,
    /// Summed bytes resident on-device for the **W4A4 (FP4-regime)** weights
    /// ([`Nvfp4Linear::resident_device_bytes`]). Only populated on a cuda build; zero when no layer
    /// resolved to the packed FP4 path.
    pub resident_fp4_bytes: usize,
    /// Summed bytes resident on-device for the **W4A16 / fallback (dequant→bf16)** weights
    /// ([`Nvfp4Linear::resident_dequant_bf16_bytes`]) — dense bf16, i.e. **no footprint win at all**.
    pub dequant_bf16_bytes: usize,
}

impl Nvfp4Report {
    /// Bytes the trunk's quantized projections **actually hold resident on-device** for their weights:
    /// packed FP4 buffers for the W4A4 layers **plus dense bf16** for every W4A16 / fallback layer.
    ///
    /// This is the honest SC#6 number, and it is **regime-aware** — a run with nothing on the packed
    /// path reports the full bf16 residency, as it should.
    pub fn resident_bytes(&self) -> usize {
        self.resident_fp4_bytes + self.dequant_bf16_bytes
    }

    /// **The SC#6 number: resident on-device weight bytes as a fraction of the dense bf16 footprint.**
    ///
    /// ~0.28 only when every projection is on the packed W4A4 path; **1.0** for a blanket-W4A16 run
    /// (dense bf16 resident, nothing packed on-device); in between under the mixed policy, in
    /// proportion to how much of the trunk the outlier class holds at bf16.
    pub fn footprint_ratio(&self) -> f64 {
        if self.bf16_bytes == 0 {
            0.0
        } else {
            self.resident_bytes() as f64 / self.bf16_bytes as f64
        }
    }

    /// The **packed format's** footprint ratio (~0.28 at ~4.5 eff bits/wt) — a property of the NVFP4
    /// container, independent of which regime the layers resolved to.
    ///
    /// Correct for "is the packing ~4.5 bits/weight?"; **wrong** for "what does this run cost in
    /// VRAM?" — that is [`Self::footprint_ratio`].
    pub fn packed_footprint_ratio(&self) -> f64 {
        if self.bf16_bytes == 0 {
            0.0
        } else {
            self.nvfp4_bytes as f64 / self.bf16_bytes as f64
        }
    }

    /// Effective bits per weight implied by the **packed NVFP4 format** (target ≈ 4.5).
    pub fn effective_bits(&self) -> f64 {
        // bf16_bytes / 2 == weight count.
        let weights = self.bf16_bytes / 2;
        if weights == 0 {
            0.0
        } else {
            self.nvfp4_bytes as f64 * 8.0 / weights as f64
        }
    }

    /// Fraction of the quantized projections actually serving the packed FP4 path.
    pub fn fp4_lit_fraction(&self) -> f64 {
        if self.n_quantized == 0 {
            0.0
        } else {
            self.fp4_lit as f64 / self.n_quantized as f64
        }
    }

    /// Fold one NVFP4 projection into the report.
    pub(crate) fn add(&mut self, l: &Nvfp4Linear) {
        self.n_quantized += 1;
        match l.regime() {
            Nvfp4Regime::Fp4W4A4 => self.fp4_lit += 1,
            Nvfp4Regime::DequantBf16 => self.dequant_bf16 += 1,
        }
        self.nvfp4_bytes += l.nvfp4_footprint_bytes();
        self.bf16_bytes += l.bf16_footprint_bytes();
        // Regime-aware residency: each layer contributes ONLY what its resolved regime actually holds
        // on-device — packed FP4 buffers, or the dense bf16 dequant. Never both, never the host
        // container (sc-11045 review, MAJOR 3).
        #[cfg(feature = "cuda")]
        {
            self.resident_fp4_bytes += l.resident_device_bytes().unwrap_or(0);
        }
        self.dequant_bf16_bytes += l.resident_dequant_bf16_bytes().unwrap_or(0);
    }
}

/// A per-layer summary of the probe's records, aggregated across steps (the sc-11045 residual gate,
/// re-run on Krea by sc-12110).
#[derive(Clone, Debug)]
pub struct LayerSparsitySummary {
    pub layer: String,
    /// The activation precision the policy assigned.
    pub act: ActPrecision,
    /// Steps measured for this layer.
    pub steps: usize,
    /// The **worst** (lowest) benign fraction seen across steps — the gate is a worst-case question.
    pub min_benign_fraction: f64,
    /// Mean benign fraction across steps.
    pub mean_benign_fraction: f64,
    /// The class implied by the worst step.
    pub worst_class: OutlierClass,
    /// Largest per-block crush ratio seen across steps.
    pub max_crush_ratio: f32,
}

impl LayerSparsitySummary {
    /// True iff a layer the policy sends to **W4A4** measured W4A4-viable at its worst step — i.e. the
    /// partition held for this layer. Layers assigned W4A16 are vacuously fine (they never run W4A4).
    pub fn partition_holds(&self) -> bool {
        match self.act {
            ActPrecision::W4A4 => !matches!(self.worst_class, OutlierClass::Dense),
            ActPrecision::W4A16 => true,
        }
    }
}

/// Aggregate raw [`ActRecord`]s into one worst-case summary per layer, sorted by layer name.
pub fn summarize(records: &[ActRecord]) -> Vec<LayerSparsitySummary> {
    use std::collections::BTreeMap;
    let mut by_layer: BTreeMap<&str, Vec<&ActRecord>> = BTreeMap::new();
    for r in records {
        by_layer.entry(r.layer.as_str()).or_default().push(r);
    }
    by_layer
        .into_iter()
        .map(|(layer, rs)| {
            let steps = rs.len();
            let min_benign = rs
                .iter()
                .map(|r| r.sparsity.benign_fraction)
                .fold(f64::INFINITY, f64::min);
            let mean_benign =
                rs.iter().map(|r| r.sparsity.benign_fraction).sum::<f64>() / steps as f64;
            let max_crush = rs
                .iter()
                .map(|r| r.sparsity.max_crush_ratio)
                .fold(0f32, f32::max);
            // The worst step's class: rebuild it from the worst benign fraction via the same floors.
            let worst_class = if min_benign >= OutlierSparsity::BENIGN_FLOOR {
                OutlierClass::Benign
            } else if min_benign >= OutlierSparsity::DENSE_FLOOR {
                OutlierClass::Sparse
            } else {
                OutlierClass::Dense
            };
            LayerSparsitySummary {
                layer: layer.to_string(),
                act: rs[0].act,
                steps,
                min_benign_fraction: min_benign,
                mean_benign_fraction: mean_benign,
                worst_class,
                max_crush_ratio: max_crush,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_plan_is_not_nvfp4_and_blanket_plans_force_their_regime() {
        assert!(!DitPlan::baseline().is_nvfp4());
        let w4a4 = DitPlan::nvfp4(Nvfp4Quant::BlanketW4A4);
        assert!(w4a4.is_nvfp4());
        // A blanket plan ignores the outlier policy — even for a role the policy would flag.
        assert_eq!(
            w4a4.act_for("txt_in.linear_1", LayerRole::context_read()),
            ActPrecision::W4A4
        );
        assert_eq!(
            w4a4.act_for("final_layer.linear", LayerRole::final_proj()),
            ActPrecision::W4A4
        );
        assert_eq!(
            DitPlan::nvfp4(Nvfp4Quant::BlanketW4A16)
                .act_for("transformer_blocks.7.attn.to_q", LayerRole::interior()),
            ActPrecision::W4A16
        );
    }

    #[test]
    fn mixed_plan_applies_the_measured_partition_to_kreas_naming() {
        let p = DitPlan::nvfp4(Nvfp4Quant::Mixed);
        // The **measured** benign compute-bulk → W4A4: normalized block inputs, in the interior blocks.
        // `ff.gate`/`ff.up` are the N=16384 GEMMs the SC#1 case rests on — 50/56 measured Benign.
        for leaf in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_gate",
            "ff.gate",
            "ff.up",
        ] {
            let name = format!("transformer_blocks.7.{leaf}");
            assert_eq!(
                p.act_for_layer(&name),
                ActPrecision::W4A4,
                "{name} reads a normalized block input and must ride W4A4"
            );
        }
        // The post-nonlinearity class → W4A16. `ff.down` measured Dense in 28/28 blocks and
        // `attn.to_out.0` in 21/28; guarding them is sc-12110's central partition fix.
        for leaf in ["attn.to_out.0", "ff.down"] {
            let name = format!("transformer_blocks.7.{leaf}");
            assert_eq!(
                p.act_for_layer(&name),
                ActPrecision::W4A16,
                "{name} reads a post-nonlinearity intermediate and MUST be guarded"
            );
        }
        // The leading edge is blocks 0..3 — measured, wider than the spike's "first two".
        for i in 0..KREA_LEADING_EDGE_BLOCKS {
            assert_eq!(
                p.act_for_layer(&format!("transformer_blocks.{i}.attn.to_q")),
                ActPrecision::W4A16,
                "leading block {i} measured Dense on its block inputs"
            );
        }
        // ...and block 4 is where the block inputs become benign again.
        assert_eq!(
            p.act_for_layer("transformer_blocks.4.attn.to_q"),
            ActPrecision::W4A4
        );
        // Krea's LAST block — which the shared substring policy cannot name — via `is_edge_block`.
        assert_eq!(
            p.act_for_layer("transformer_blocks.27.attn.to_q"),
            ActPrecision::W4A16
        );
        // Krea's caption-reading class — no `attn2` / `caption_projection` exists to match — via
        // `is_context_read`.
        for name in [
            "txt_in.linear_1",
            "txt_in.linear_2",
            "text_fusion.layerwise_blocks.1.attn.to_q",
            "text_fusion.refiner_blocks.1.ff.down",
        ] {
            assert_eq!(
                p.act_for_layer(name),
                ActPrecision::W4A16,
                "{name} reads text-encoder context and must be guarded"
            );
        }
        // Krea's final head — `final_layer.linear`, which the shared name anchor will NOT infer
        // (sc-12140) — via `is_final_proj`. Measured Dense, crush 909×.
        assert_eq!(p.act_for_layer("final_layer.linear"), ActPrecision::W4A16);
        // `img_in` is the image ingest and measured perfectly Benign (1.00000, crush 0.0) — it stays in
        // the lane. A guard that swept it up "to be safe" would be cost with no evidence.
        assert_eq!(p.act_for_layer("img_in"), ActPrecision::W4A4);
    }

    /// **The measured partition, pinned as a whole** (sc-12110): on the shipping Turbo trunk the mixed
    /// policy must send exactly the layers that measured W4A4-viable to W4A4 — no more, no less.
    ///
    /// This is the arithmetic behind the run's `139/260 fp4-lit`: 23 interior blocks (4..=26) × 6 benign
    /// leaves + `img_in`. If someone widens or narrows a guard, this count moves and the test says so.
    #[test]
    fn measured_partition_yields_the_expected_w4a4_surface() {
        let p = DitPlan::nvfp4(Nvfp4Quant::Mixed);
        let n = DEFAULT_NUM_LAYERS;
        let mut names: Vec<String> = vec![
            "img_in".into(),
            "txt_in.linear_1".into(),
            "txt_in.linear_2".into(),
        ];
        for i in 0..n {
            for leaf in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_gate",
                "attn.to_out.0",
                "ff.gate",
                "ff.up",
                "ff.down",
            ] {
                names.push(format!("transformer_blocks.{i}.{leaf}"));
            }
        }
        for kind in ["layerwise_blocks", "refiner_blocks"] {
            for i in 0..2 {
                for leaf in [
                    "attn.to_q",
                    "attn.to_k",
                    "attn.to_v",
                    "attn.to_gate",
                    "attn.to_out.0",
                    "ff.gate",
                    "ff.up",
                    "ff.down",
                ] {
                    names.push(format!("text_fusion.{kind}.{i}.{leaf}"));
                }
            }
        }
        names.push("final_layer.linear".into());
        assert_eq!(names.len(), 260, "the lane's surface");

        let w4a4: Vec<&String> = names
            .iter()
            .filter(|n| p.act_for_layer(n) == ActPrecision::W4A4)
            .collect();
        // 23 interior blocks (4..=26) × {to_q, to_k, to_v, to_gate, ff.gate, ff.up} + img_in.
        let interior_blocks = n - KREA_LEADING_EDGE_BLOCKS - 1;
        assert_eq!(interior_blocks, 23);
        assert_eq!(
            w4a4.len(),
            interior_blocks * 6 + 1,
            "W4A4 surface changed — re-run the partition gate before accepting this: {:?}",
            w4a4.iter().take(5).collect::<Vec<_>>()
        );
        // Nothing post-nonlinearity, context-reading, edge, or head may be in there.
        for name in &w4a4 {
            assert!(
                !name.ends_with(".ff.down") && !name.ends_with(".attn.to_out.0"),
                "{name} is a post-nonlinearity site and measured Dense — it must not ride W4A4"
            );
            assert!(!name.starts_with("text_fusion.") && !name.starts_with("txt_in."));
            assert_ne!(*name, "final_layer.linear");
        }
    }

    /// **The sc-12140 defect, pinned for Krea.** The shared name-only fallback anchors on a trailing
    /// `proj_out` segment. Krea's head is `final_layer.linear`, so the fallback does **not** fire — the
    /// loader MUST thread `LayerRole::final_proj()`. If someone ever drops that threading, this test
    /// fails rather than the trunk head silently landing on W4A4.
    #[test]
    fn final_head_is_only_guarded_because_the_loader_states_it() {
        let p = DitPlan::nvfp4(Nvfp4Quant::Mixed);
        // Without the explicit role, the name alone leaves Krea's head on the compute path.
        assert_eq!(
            p.act_for("final_layer.linear", LayerRole::interior()),
            ActPrecision::W4A4,
            "precondition: Krea's head name is NOT inferable by the shared anchor"
        );
        // With it, guarded.
        assert_eq!(
            p.act_for("final_layer.linear", LayerRole::final_proj()),
            ActPrecision::W4A16
        );
    }

    /// The role assignment is derived from the trunk's topology in ONE place, so the loader and the
    /// validation harness cannot drift apart.
    #[test]
    fn for_krea_layer_names_the_edges_head_and_context_class() {
        let n = 28usize;
        // Leading edge: blocks 0..3 (measured — wider than the spike's "first two").
        assert!(LayerRole::for_krea_layer("transformer_blocks.0.attn.to_q", n).is_edge_block);
        assert!(LayerRole::for_krea_layer("transformer_blocks.3.attn.to_q", n).is_edge_block);
        assert!(!LayerRole::for_krea_layer("transformer_blocks.4.attn.to_q", n).is_edge_block);
        // Trailing edge: the last block.
        assert!(LayerRole::for_krea_layer("transformer_blocks.27.ff.gate", n).is_edge_block);
        // An interior normalized-input projection: no flag set at all.
        assert_eq!(
            LayerRole::for_krea_layer("transformer_blocks.14.ff.gate", n),
            LayerRole::interior()
        );
        // The post-nonlinearity class — sc-12110's central finding.
        assert_eq!(
            LayerRole::for_krea_layer("transformer_blocks.14.ff.down", n),
            LayerRole::post_nonlinearity()
        );
        assert_eq!(
            LayerRole::for_krea_layer("transformer_blocks.14.attn.to_out.0", n),
            LayerRole::post_nonlinearity()
        );
        // The head — stated, never inferred (sc-12140).
        assert_eq!(
            LayerRole::for_krea_layer("final_layer.linear", n),
            LayerRole::final_proj()
        );
        // The caption-reading class.
        assert_eq!(
            LayerRole::for_krea_layer("txt_in.linear_1", n),
            LayerRole::context_read()
        );
        // A text-fusion post-nonlinearity site carries BOTH facts (it is context-reading *and* reads a
        // post-nonlinearity intermediate — it measured the trunk's worst crush at 40145×).
        let r = LayerRole::for_krea_layer("text_fusion.refiner_blocks.0.attn.to_out.0", n);
        assert!(r.is_context_read && r.is_post_nonlinearity);
        // `img_in` is the image ingest — compute-bulk, measured perfectly benign.
        assert_eq!(
            LayerRole::for_krea_layer("img_in", n),
            LayerRole::interior()
        );
    }

    /// A prefix match on `transformer_blocks.2.` must not be satisfied by `transformer_blocks.27.` —
    /// the trailing dot is load-bearing. (With `num_layers = 3` the last block is `2`; block 27 does not
    /// exist, but the guard is that a *prefix* of a longer index never aliases.)
    #[test]
    fn edge_block_prefix_does_not_alias_longer_indices() {
        // Last block of a 3-block trunk is `transformer_blocks.2.`; `.27.` must NOT match it.
        assert_eq!(
            LayerRole::for_krea_layer("transformer_blocks.27.attn.to_q", 3),
            LayerRole::interior(),
            "`transformer_blocks.2` must not prefix-match `transformer_blocks.27`"
        );
    }

    /// The **packed-format** ratio: a property of the NVFP4 container, true in every regime.
    #[test]
    fn report_packed_ratio_and_effective_bits_are_nvfp4_scale() {
        // 4096×4096 weight: bf16 = 33_554_432 B; NVFP4 ≈ nibbles (8_388_608) + scales (1_048_576).
        let r = Nvfp4Report {
            nvfp4_bytes: 9_437_184,
            bf16_bytes: 33_554_432,
            ..Default::default()
        };
        assert!((r.packed_footprint_ratio() - 0.28125).abs() < 1e-6);
        assert!((r.effective_bits() - 4.5).abs() < 1e-6);
    }

    /// **The SC#6 ratio is regime-aware**: it reports what the run holds in VRAM, not what the packed
    /// container weighs (sc-11045 review, MAJOR 3 — carried over so Krea's SC#6 claim cannot regress to
    /// the regime-blind form).
    #[test]
    fn footprint_ratio_is_regime_aware_and_never_claims_fp4_for_a_bf16_run() {
        let bf16 = 33_554_432usize;
        let packed = 9_437_184usize;

        let w4a4 = Nvfp4Report {
            nvfp4_bytes: packed,
            bf16_bytes: bf16,
            resident_fp4_bytes: packed,
            dequant_bf16_bytes: 0,
            ..Default::default()
        };
        assert!((w4a4.footprint_ratio() - 0.28125).abs() < 1e-6);

        // Blanket W4A16 / capability fallback: NOTHING packed on-device; every weight resident as dense
        // bf16. The honest answer is 1.0 — the regime buys stability, not footprint.
        let w4a16 = Nvfp4Report {
            nvfp4_bytes: packed,
            bf16_bytes: bf16,
            resident_fp4_bytes: 0,
            dequant_bf16_bytes: bf16,
            ..Default::default()
        };
        assert!(
            (w4a16.footprint_ratio() - 1.0).abs() < 1e-6,
            "a W4A16 run holds dense bf16 — it must NEVER report an NVFP4 footprint (got {:.4})",
            w4a16.footprint_ratio()
        );
        assert!((w4a16.packed_footprint_ratio() - 0.28125).abs() < 1e-6);
    }

    #[test]
    fn summarize_reports_worst_case_and_partition_verdict() {
        let mk = |layer: &str, step: usize, act, benign: f64| ActRecord {
            layer: layer.to_string(),
            step,
            act,
            sparsity: OutlierSparsity {
                total_blocks: 1000,
                outlier_blocks: ((1.0 - benign) * 1000.0).round() as usize,
                benign_fraction: benign,
                robust_scale: 1.0,
                max_crush_ratio: 10.0,
                tau: 20.0,
            },
        };
        let recs = vec![
            mk("a", 0, ActPrecision::W4A4, 0.999),
            mk("a", 1, ActPrecision::W4A4, 0.996), // worst step still benign
            mk("b", 0, ActPrecision::W4A4, 0.999),
            mk("b", 1, ActPrecision::W4A4, 0.5), // collapses at step 1 → partition broken
        ];
        let s = summarize(&recs);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].layer, "a");
        assert_eq!(s[0].steps, 2);
        assert!((s[0].min_benign_fraction - 0.996).abs() < 1e-9);
        assert_eq!(s[0].worst_class, OutlierClass::Benign);
        assert!(s[0].partition_holds());
        // `b` is assigned W4A4 but measures Dense at its worst step — the gate must catch it.
        assert_eq!(s[1].worst_class, OutlierClass::Dense);
        assert!(!s[1].partition_holds());
    }
}
