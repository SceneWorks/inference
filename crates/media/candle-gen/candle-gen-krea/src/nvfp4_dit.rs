//! The **NVFP4 precision seam** for the Krea 2 single-stream DiT trunk (sc-12110, epic 11037).
//!
//! [`crate::transformer::Krea2Transformer`] loads its projections through
//! [`crate::loader::linear_detect`] by default — dense bf16, or the MLX-packed q4/q8 dequant-on-forward
//! [`crate::quant::QLinear`]. This module adds the seam that lets the SAME trunk serve those projections
//! through [`candle_gen::quant::Nvfp4Linear`] instead — the sc-11041 packed-forward NVFP4 path — so a **real Krea 2 Turbo
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
//! 2. [`KreaSite`] / [`LayerRole`] / [`ExecutionRole`] — the provider-owned **role table**: an
//!    exhaustive structural classification of every projection the lane serves, the structural facts
//!    that classification implies, and the execution role (packed W4A4 or dense BF16) those facts
//!    select. [`DitPlan::representation`] is the whole decision table, capability facts included.
//! 3. [`ActProbe`] — a per-layer, per-step **activation-outlier sparsity** recorder, so the
//!    benign→W4A4 / outlier→W4A16 partition can be re-derived against **Krea's** naming from live
//!    activations rather than inherited from SANA's.
//!
//! # Why Krea owns its role table outright (sc-12121)
//!
//! The shared policy (`candle_gen::quant::ActPrecision::for_outlier_layer_with`) is substring-based and
//! was tuned on SANA's diffusers naming. **Every one of its anchors misses on Krea's checkpoint**, and
//! every gap fails in the *unsafe* direction (an outlier-carrying layer silently landing on W4A4):
//!
//! * **`attn2` / `cross_att*`** — Krea has **no cross-attention at all**. It is a *single-stream* DiT:
//!   the fused text context is **concatenated onto the image token sequence** (`combined = [ctx ; img]`)
//!   and read by ordinary self-attention. There is no projection named `attn2` to guard.
//! * **`caption_projection`** — Krea's text→DiT ingest is named `txt_in.linear_{1,2}`, fed by the
//!   `text_fusion` stack that aggregates the raw Qwen3-VL hidden states. Neither matches.
//! * **`proj_out`** — Krea's trunk head is `final_layer.linear`, so the shared crate's name-only
//!   `proj_out` anchor cannot fire on it.
//!   (Krea's *only* `proj_out` is a control-branch layer nested under `blocks.{i}`, which that anchor
//!   correctly declines — verified, and the reason it is safe to leave alone in its own crate.)
//!
//! sc-12110 threaded Krea's facts *into* that policy through [`LayerRole`]. **sc-12121 removes the
//! policy from the Krea path entirely**: production Krea selection calls neither the shared substring
//! heuristic nor its `names_final_proj` fallback. A dotted key is classified into an exhaustive
//! [`KreaSite`] role table, and the role — not the spelling — selects the execution role. A name the
//! table does not recognise is **not** silently an interior W4A4 projection any more: it resolves to
//! [`DenseReason::Unclassified`] and takes the dense BF16 fallback, because "a guard that quietly
//! failed to fire" is the exact defect class (sc-12140) this story exists to delete.
//!
//! **Measured vindication:** `final_layer.linear` really does measure
//! [`candle_gen::quant::OutlierClass::Dense`] on real activations (crush **909×**). It is guarded
//! because the role table names it [`KreaSite::TrunkHead`]; no name-only anchor in any crate would
//! have fired on it.
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
//! Everything degrades cleanly: an [`candle_gen::quant::Nvfp4Linear`] on a non-`sm_120` device, on CPU, or on a non-cuda
//! build transparently serves the dequant→bf16 fallback (sc-11041), so a `DitPlan::nvfp4(..)` trunk
//! still *runs* everywhere — it just does not light the FP4 cores. That is the SC#4 Blackwell-only
//! gate, observed at model level by [`Nvfp4Report::fp4_lit`].

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::lock_recover;
use candle_gen::quant::{
    ActPrecision, AdaptLinear, Nvfp4Context, Nvfp4Fallback, Nvfp4Linear, Nvfp4Regime, OutlierClass,
    OutlierSparsity, NVFP4_BLOCK, NVFP4_K_ALIGN, NVFP4_N_ALIGN,
};

/// How the trunk should serve one projection's activations when running NVFP4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Quant {
    /// The **sc-11038 mixed-precision policy** (the shipping default): the outlier-carrying class runs
    /// W4A16 (bf16 activation), the benign compute-bulk runs W4A4. Classified **entirely** by Krea's
    /// own role table — [`KreaSite`] → [`LayerRole`] → [`ExecutionRole`] — with no substring
    /// heuristic anywhere in the path (sc-12121); see [`DitPlan::execution_role`].
    Mixed,
    /// Blanket W4A4 on every eligible projection — ignores the outlier policy. For a controlled
    /// throughput/stability bench of the FP4 compute path, **not** a shipping regime.
    BlanketW4A4,
    /// Blanket W4A16 on every eligible projection — the NVFP4 *storage* tier (weights packed, bf16
    /// activation, no FP4 compute). The stability-fallback default.
    BlanketW4A16,
}

/// One of the eight GEMM leaves a Krea attention + SwiGLU block exposes to the NVFP4 lane
/// (sc-12121). Both the single-stream `transformer_blocks.{i}` and the `text_fusion` blocks are built
/// from the same `GatedAttention` + `SwiGlu` modules, so both carry exactly this leaf set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockLeaf {
    /// `attn.to_q` — reads the RMSNorm'd block input.
    AttnQ,
    /// `attn.to_k` — reads the RMSNorm'd block input.
    AttnK,
    /// `attn.to_v` — reads the RMSNorm'd block input.
    AttnV,
    /// `attn.to_gate` — reads the RMSNorm'd block input (Krea's gate branch; no SANA analogue).
    AttnGate,
    /// `attn.to_out.0` — reads `attn_out · sigmoid(gate(x))`, a **post-nonlinearity** intermediate.
    AttnOut,
    /// `ff.gate` — reads the RMSNorm'd block input. `[16384 ← 6144]`, one of the two widest GEMMs.
    FfGate,
    /// `ff.up` — reads the RMSNorm'd block input. `[16384 ← 6144]`.
    FfUp,
    /// `ff.down` — reads `silu(gate(x)) · up(x)`, a **post-nonlinearity** intermediate.
    FfDown,
}

/// Define [`BlockLeaf`]'s key accessors from **one** list of `(variant, module, module-relative
/// suffix, ordinal)` rows (sc-12121).
///
/// The point is that there is exactly one place the eight loader strings are spelled: `attn`/`ff`
/// `load_planned` call [`BlockLeaf::module_leaf`] rather than passing literals of their own, so the
/// role table and the loader cannot drift apart. Every generated body is an **exhaustive** `match`
/// with no wildcard arm, so a leaf added to the enum without a row here does not compile.
macro_rules! block_leaf_table {
    ($( $variant:ident => $module:literal, $leaf:literal, $ordinal:literal );* $(;)?) => {
        impl BlockLeaf {
            /// The module this leaf is loaded by — `attn` (`GatedAttention`) or `ff` (`SwiGlu`),
            /// spelled as the block's `join(prefix, ..)` segment.
            pub fn module(self) -> &'static str {
                match self { $( Self::$variant => $module, )* }
            }

            /// The dotted suffix this leaf is loaded under **relative to its module's prefix** — the
            /// exact literal `GatedAttention::load_planned` / `SwiGlu::load_planned` hand to
            /// `linear_detect_planned`, because those loaders read it from here.
            pub fn module_leaf(self) -> &'static str {
                match self { $( Self::$variant => $leaf, )* }
            }

            /// The dotted suffix this leaf is loaded under relative to its **block** prefix —
            /// [`Self::module`] and [`Self::module_leaf`] composed at compile time.
            pub fn leaf_key(self) -> &'static str {
                match self { $( Self::$variant => concat!($module, ".", $leaf), )* }
            }

            /// A dense index for this leaf, by exhaustive match — the count source
            /// `block_leaf_all_is_every_variant` crosses against [`Self::ALL`]'s length.
            #[cfg(test)]
            fn ordinal(self) -> usize {
                match self { $( Self::$variant => $ordinal, )* }
            }

            /// The leaf with the given [`Self::ordinal`], if any — the inverse the anchor test walks
            /// to reconstruct [`Self::ALL`] without reading the array literal.
            #[cfg(test)]
            fn from_ordinal(i: usize) -> Option<Self> {
                $( if i == $ordinal { return Some(Self::$variant); } )*
                None
            }
        }
    };
}

block_leaf_table! {
    AttnQ    => "attn", "to_q",     0;
    AttnK    => "attn", "to_k",     1;
    AttnV    => "attn", "to_v",     2;
    AttnGate => "attn", "to_gate",  3;
    AttnOut  => "attn", "to_out.0", 4;
    FfGate   => "ff",   "gate",     5;
    FfUp     => "ff",   "up",       6;
    FfDown   => "ff",   "down",     7;
}

impl BlockLeaf {
    /// Every leaf, in the order a block loads them — the coverage test's enumeration source.
    ///
    /// Anchored by `block_leaf_all_is_every_variant`: an array literal is not an exhaustive
    /// construct, so the test reconstructs this list from the exhaustive `ordinal` / `from_ordinal`
    /// pair (test-only, hence not linkable here) and asserts the two agree.
    pub const ALL: [Self; 8] = [
        Self::AttnQ,
        Self::AttnK,
        Self::AttnV,
        Self::AttnGate,
        Self::AttnOut,
        Self::FfGate,
        Self::FfUp,
        Self::FfDown,
    ];

    /// The leaves `GatedAttention::load_planned` loads, in load order — the loader's own enumeration,
    /// so its five projection keys come from this table too.
    pub const ATTN: [Self; 5] = [
        Self::AttnQ,
        Self::AttnK,
        Self::AttnV,
        Self::AttnGate,
        Self::AttnOut,
    ];

    /// The leaves `SwiGlu::load_planned` loads, in load order.
    pub const FF: [Self; 3] = [Self::FfGate, Self::FfUp, Self::FfDown];

    /// The leaf named by a block-relative dotted suffix, or `None` for a suffix that is not one of the
    /// lane's eight GEMM leaves.
    fn from_leaf_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|leaf| leaf.leaf_key() == key)
    }

    /// True iff this leaf's **input** is a post-nonlinearity intermediate rather than a normalized
    /// block input — see [`LayerRole::is_post_nonlinearity`], sc-12110's central partition finding.
    ///
    /// Deliberately an **exhaustive** `match` rather than a `matches!`: a `matches!` would silently
    /// default a newly-added leaf to `false`, i.e. route an unexamined post-nonlinearity projection
    /// straight into the packed W4A4 lane — the unsafe direction. Written this way, adding a leaf
    /// without answering this question does not compile.
    pub fn reads_post_nonlinearity(self) -> bool {
        match self {
            Self::AttnOut | Self::FfDown => true,
            Self::AttnQ
            | Self::AttnK
            | Self::AttnV
            | Self::AttnGate
            | Self::FfGate
            | Self::FfUp => false,
        }
    }
}

/// **The Krea role table** (sc-12121): every site in the trunk the NVFP4 lane serves, named
/// structurally rather than by substring.
///
/// [`Self::classify`] is the single parse from a dotted key to a site, and it is **total by
/// refusal**: a key that is not one of these sites returns `None`, which
/// [`LayerRole::for_krea_layer`] turns into [`DenseReason::Unclassified`] — the dense BF16 fallback.
/// Nothing reaches the packed W4A4 lane by failing to match a guard.
///
/// The sites deliberately mirror the loader's call graph (`Krea2Transformer::load_front`,
/// `TextFusionTransformer::load_planned`, `Krea2Block::load_planned`), which is why the coverage test
/// can enumerate the whole 260-projection surface from this enum and assert it round-trips.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KreaSite {
    /// `img_in` — the image-token ingest. Measured perfectly Benign (1.00000, crush 0.0).
    ImageIngest,
    /// `txt_in.linear_{1,2}` — the fused-context ingest into DiT width, Krea's `caption_projection`
    /// analogue.
    ContextIngest,
    /// `text_fusion.{layerwise,refiner}_blocks.{i}.{leaf}` — the stack that aggregates the **raw**
    /// Qwen3-VL hidden states, i.e. the massive-activation source itself.
    TextFusion {
        /// Which of the block's eight GEMM leaves this projection is.
        leaf: BlockLeaf,
    },
    /// `transformer_blocks.{index}.{leaf}` — the single-stream compute bulk.
    Block {
        /// The block index as spelled in the key.
        index: usize,
        /// Which of the block's eight GEMM leaves this projection is.
        leaf: BlockLeaf,
    },
    /// `final_layer.linear` — the trunk head `[6144 → 64]`. Measured Dense, crush 909×.
    TrunkHead,
}

impl KreaSite {
    /// The site `name` denotes, or `None` when the role table does not name it.
    ///
    /// Structural throughout: fixed keys are matched whole, and block keys are **parsed** (prefix,
    /// then an integer index, then one of the eight leaf suffixes) rather than sniffed for
    /// substrings. `transformer_blocks.27.attn.to_q` and `transformer_blocks.2.attn.to_q` are
    /// different sites because the parse says so, not because a trailing dot happened to be in the
    /// pattern.
    pub fn classify(name: &str) -> Option<Self> {
        match name {
            "img_in" => return Some(Self::ImageIngest),
            "txt_in.linear_1" | "txt_in.linear_2" => return Some(Self::ContextIngest),
            "final_layer.linear" => return Some(Self::TrunkHead),
            _ => {}
        }
        if let Some(rest) = name.strip_prefix("transformer_blocks.") {
            let (index, leaf) = rest.split_once('.')?;
            return Some(Self::Block {
                index: index.parse().ok()?,
                leaf: BlockLeaf::from_leaf_key(leaf)?,
            });
        }
        let rest = name.strip_prefix("text_fusion.")?;
        for kind in ["layerwise_blocks.", "refiner_blocks."] {
            let Some(rest) = rest.strip_prefix(kind) else {
                continue;
            };
            let (index, leaf) = rest.split_once('.')?;
            index.parse::<usize>().ok()?;
            return Some(Self::TextFusion {
                leaf: BlockLeaf::from_leaf_key(leaf)?,
            });
        }
        None
    }
}

/// The **execution role** a Krea projection is assigned: which of the two representations the trunk
/// actually serves it as (sc-12121, epic E5).
///
/// There are only two, because there are only two things `Nvfp4Linear` can be: the packed W4A4
/// FP4-tensor-core path, or a dense BF16 weight. **W4A16 is dense BF16** — it dequantizes the packed
/// weight once at construction and holds the full dense footprint — so it is reported here as
/// [`Self::DenseBf16`], never as "native NVFP4".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionRole {
    /// Packed W4A4: the weight stays in its NVFP4 container on-device and the cuBLASLt FP4 GEMM runs.
    PackedW4A4,
    /// Dense BF16, for the stated reason. Covers both the W4A16 outlier override and every
    /// capability fallback — they hold the same bytes and run the same GEMM.
    DenseBf16(DenseReason),
}

impl ExecutionRole {
    /// True iff this role is the packed W4A4 lane.
    pub fn is_packed_w4a4(self) -> bool {
        matches!(self, Self::PackedW4A4)
    }

    /// The activation precision this role asks [`Nvfp4Linear`] for.
    pub fn act_precision(self) -> ActPrecision {
        match self {
            Self::PackedW4A4 => ActPrecision::W4A4,
            Self::DenseBf16(_) => ActPrecision::W4A16,
        }
    }
}

/// Why a projection is served dense BF16 rather than packed W4A4 (sc-12121).
///
/// The variants are listed in the order [`DitPlan::representation`] settles them, which is the order
/// the real pipeline settles them: the checkpoint's own declaration and geometry first (they are
/// answered at *plan* time by the residency policy), then the device floor, then the plan's requested
/// regime, then the structural role, then the construction-time fused-quantizer probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseReason {
    /// The producer marked the layer `full_precision_matrix_mult` — "do not run a quantized matmul
    /// here". Answered by the residency policy at plan time.
    FullPrecisionDeclared,
    /// The checkpoint does not offer this projection as NVFP4 at all: Kitchen's Krea profile
    /// deliberately exports sensitive layers as dense BF16 and the import preserves that.
    PreservedDense,
    /// ComfyUI padded the stored grid (`logical_shape != stored_shape`). The packed container has
    /// nowhere to record the unpad, so the repacked operand would contract over padding.
    PaddedStorage,
    /// The stored/packed grid is not one the cuBLASLt FP4 GEMM accepts (padded K not a multiple of
    /// `NVFP4_K_ALIGN`, or N not a multiple of `NVFP4_N_ALIGN`).
    ShapeIneligible,
    /// The bound device is not at the NVFP4 `sm_120` floor (CPU, a pre-Blackwell GPU, or a non-cuda
    /// build).
    NoNvfp4Hardware,
    /// The plan does not serve this trunk through NVFP4, or asks for blanket W4A16 — the NVFP4
    /// *storage* tier, which is dense BF16 resident by construction.
    DenseRegimeRequested,
    /// The role table does not name this projection, so nothing structural is known about its
    /// activations. The fallback is dense BF16 on purpose: an unrecognised key must never reach the
    /// packed lane by default (sc-12140).
    Unclassified,
    /// The trunk head (`final_layer.linear`) — measured Dense, crush 909×.
    TrunkHead,
    /// Reads text-encoder-derived context (`text_fusion.*`, `txt_in.*`) — measured Dense, crush up
    /// to 40145×.
    ContextRead,
    /// Reads a post-nonlinearity intermediate (`attn.to_out.0`, `ff.down`) — sc-12110's central
    /// finding, Dense in 28/28 blocks for `ff.down`.
    PostNonlinearity,
    /// Sits in a leading (`0..4`) or the trailing single-stream block, where the caption's massive
    /// activations still reach the *block inputs*.
    EdgeBlock,
    /// The fused NVFP4 activation quantizer will not compile on this device. W4A4 through the
    /// unfused reference chain measured **0.01×** vs dense bf16, so the honest answer is W4A16
    /// (sc-12078) — settled at construction, never per forward.
    NoFusedQuantizer,
    /// The trunk's shared cuBLASLt context is bound to a **different** device than this projection
    /// (`Nvfp4Fallback::DeviceMismatch`). A runtime accident of context sharing, **not** predictable
    /// from any per-key capability fact — [`Nvfp4Capability`] cannot model it, so the prediction is
    /// reconciled against the constructed layer's own reported cause instead (sc-12121 review fix).
    DeviceMismatch,
    /// Staging the FP4 weight onto the device failed (allocation/driver accident,
    /// `Nvfp4Fallback::StagingFailed`). Like [`Self::DeviceMismatch`], invisible to a plan-time
    /// probe and reconciled from the constructed layer.
    StagingFailed,
}

/// The [`DenseReason`] a construction-time [`Nvfp4Fallback`] corresponds to, or `None` when the cause
/// is one [`DitPlan::representation`] already predicts from [`Nvfp4Capability`] (sc-12121 review fix).
///
/// This exists for exactly the two causes no capability fact can see — a shared-context/device
/// mismatch and a weight-staging failure. Both are legitimate transparent degradations that predate
/// this story; without this mapping `check_representation` would abort a whole trunk load and blame
/// the role table for a driver accident. Every other cause stays `None`, so a genuine
/// prediction/construction disagreement is still a hard error.
pub fn dense_reason_for_fallback(cause: Nvfp4Fallback) -> Option<DenseReason> {
    match cause {
        Nvfp4Fallback::DeviceMismatch => Some(DenseReason::DeviceMismatch),
        Nvfp4Fallback::StagingFailed => Some(DenseReason::StagingFailed),
        // Predicted by `Nvfp4Capability`: `nvfp4_device`, `layout_native`, `fused_quantizer`, and the
        // plan's own regime. A disagreement on any of these is a real defect, not an accident.
        Nvfp4Fallback::W4A16Requested
        | Nvfp4Fallback::NotCudaDevice
        | Nvfp4Fallback::ShapeIneligible
        | Nvfp4Fallback::NoDeviceHandle
        | Nvfp4Fallback::NoFusedQuantizer => None,
    }
}

/// The device- and checkpoint-level facts that can force dense BF16 **regardless of the structural
/// role** (sc-12121). Every field is read off something real — the compiled logical plan, the probed
/// device residency, the shared cuBLASLt context — never guessed from a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4Capability {
    /// The checkpoint offers this projection as NVFP4 (false for Kitchen's preserved-dense layers).
    pub checkpoint_offers_nvfp4: bool,
    /// The producer declared `full_precision_matrix_mult` for this layer.
    pub full_precision_declared: bool,
    /// The stored grid is the layer itself (`logical_shape == stored_shape`).
    pub storage_unpadded: bool,
    /// The grid is one the cuBLASLt FP4 GEMM accepts.
    pub layout_native: bool,
    /// The bound device is at the NVFP4 `sm_120` floor.
    pub nvfp4_device: bool,
    /// The fused NVFP4 activation quantizer compiled on this device.
    pub fused_quantizer: bool,
}

/// True iff a **dense** `[out, in]` projection packs to a grid the cuBLASLt FP4 GEMM accepts
/// (sc-12121) — the NVFP4 *validation* lane's twin of
/// [`candle_gen::logical_weights::nvfp4_layout_is_native`], which answers the same question for a
/// checkpoint's already-stored grid.
///
/// The two differ in exactly one place and it matters: `Nvfp4Tensor::pack` pads the contraction to
/// [`NVFP4_BLOCK`] (16), and `Nvfp4Linear`'s shape gate then tests that **padded** width against
/// [`NVFP4_K_ALIGN`] (32). So `in_features = 48` is eligible while `in_features = 16` is not, and
/// reading the raw `cols` here would have disagreed with the layer it is predicting.
pub fn dense_shape_is_fp4_eligible(rows: usize, cols: usize) -> bool {
    let cols_padded = cols.div_ceil(NVFP4_BLOCK) * NVFP4_BLOCK;
    cols_padded.is_multiple_of(NVFP4_K_ALIGN) && rows.is_multiple_of(NVFP4_N_ALIGN)
}

impl Nvfp4Capability {
    /// Everything available — the **only** combination that can reach the packed W4A4 lane.
    pub const ELIGIBLE: Self = Self {
        checkpoint_offers_nvfp4: true,
        full_precision_declared: false,
        storage_unpadded: true,
        layout_native: true,
        nvfp4_device: true,
        fused_quantizer: true,
    };

    /// [`Self::ELIGIBLE`] on a device below the NVFP4 floor — a CPU lane, a pre-Blackwell GPU, or a
    /// non-cuda build (which also has no fused quantizer).
    pub const NO_HARDWARE: Self = Self {
        nvfp4_device: false,
        fused_quantizer: false,
        ..Self::ELIGIBLE
    };
}

/// The **structural facts** about a Krea projection, derived from the role table
/// ([`KreaSite`]) — the form the loader and the validation harness both consume (sc-11045 pattern,
/// sc-12110 for Krea, made role-table-derived and total by sc-12121).
///
/// Every flag defaults to `false`, i.e. "an ordinary interior compute-bulk projection". The one flag
/// that is *not* a fact about the trunk is [`Self::is_unclassified`]: it records that the role table
/// did not name the key at all.
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
    /// **No name-only anchor in any crate fires on this key** (the shared crate's fallback anchors on
    /// a trailing `proj_out` segment, and Krea's head is not spelled that way), which is why the role
    /// table names [`KreaSite::TrunkHead`] outright. That is precisely the defect class sc-12140
    /// records: a name-only anchor that silently does not fire leaves the trunk head — measured Dense
    /// on SANA, crush 438×, and on Krea, crush 909× — on W4A4.
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
    /// The role table ([`KreaSite::classify`]) does not name this projection's key.
    ///
    /// This is not a fact about the trunk; it is the absence of one, and it is recorded rather than
    /// defaulted away. Before sc-12121 an unrecognised key fell through every guard and landed on
    /// W4A4 — the same silent-miss failure mode as a substring anchor that does not fire. Now it
    /// selects [`DenseReason::Unclassified`], i.e. dense BF16.
    pub is_unclassified: bool,
}

impl LayerRole {
    /// An interior compute-bulk projection: not an edge block, not the head, not context-reading.
    pub fn interior() -> Self {
        Self::default()
    }

    /// A projection the role table does not name — dense BF16 by [`DenseReason::Unclassified`].
    pub fn unclassified() -> Self {
        Self {
            is_unclassified: true,
            ..Self::default()
        }
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

    /// The role the shipping loader assigns `name` on a Krea trunk of `num_layers` single-stream blocks
    /// — the **single source of truth** for the trunk's topology facts.
    ///
    /// Shared by [`crate::transformer::Krea2Transformer::load_planned`] and the validation harness, so a
    /// report can never cross the measured class against a *different* partition than the one the loader
    /// actually built (the drift sc-11045's Sana harness invited by re-deriving roles inline).
    ///
    /// A key the role table does not name yields [`Self::unclassified`], **not** an interior
    /// projection (sc-12121).
    pub fn for_krea_layer(name: &str, num_layers: usize) -> Self {
        match KreaSite::classify(name) {
            Some(site) => Self::for_site(site, num_layers),
            None => Self::unclassified(),
        }
    }

    /// The structural facts a [`KreaSite`] implies on a trunk of `num_layers` single-stream blocks —
    /// the role table proper (sc-12121). Exhaustive over the enum, so adding a site is a compile
    /// error until its facts are stated.
    pub fn for_site(site: KreaSite, num_layers: usize) -> Self {
        match site {
            // The image ingest measured perfectly Benign (1.00000, crush 0.0): compute bulk.
            KreaSite::ImageIngest => Self::interior(),
            KreaSite::ContextIngest => Self::context_read(),
            KreaSite::TextFusion { leaf } => Self {
                is_context_read: true,
                is_post_nonlinearity: leaf.reads_post_nonlinearity(),
                ..Self::default()
            },
            KreaSite::Block { index, leaf } => Self {
                is_edge_block: index < KREA_LEADING_EDGE_BLOCKS || index + 1 == num_layers.max(1),
                is_post_nonlinearity: leaf.reads_post_nonlinearity(),
                ..Self::default()
            },
            KreaSite::TrunkHead => Self::final_proj(),
        }
    }

    /// The [`ExecutionRole`] these structural facts select, **before** any device or checkpoint
    /// capability fact (sc-12121). The capability-aware form is [`DitPlan::representation`].
    ///
    /// The precedence is the reporting order only — the facts are not mutually exclusive (a
    /// `text_fusion` `ff.down` is both context-reading and post-nonlinearity), and every one of them
    /// selects dense BF16, so which is named cannot change the representation.
    pub fn execution_role(&self) -> ExecutionRole {
        if self.is_unclassified {
            ExecutionRole::DenseBf16(DenseReason::Unclassified)
        } else if self.is_final_proj {
            ExecutionRole::DenseBf16(DenseReason::TrunkHead)
        } else if self.is_context_read {
            ExecutionRole::DenseBf16(DenseReason::ContextRead)
        } else if self.is_post_nonlinearity {
            ExecutionRole::DenseBf16(DenseReason::PostNonlinearity)
        } else if self.is_edge_block {
            ExecutionRole::DenseBf16(DenseReason::EdgeBlock)
        } else {
            ExecutionRole::PackedW4A4
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
    /// The **one** cuBLASLt handle every NVFP4 projection in this trunk shares (sc-12274).
    ///
    /// Set once by [`crate::transformer::Krea2Transformer::load_planned`], which is also where
    /// `num_layers` is bound — the plan is already the value threaded to every
    /// [`crate::loader::linear_detect_planned`] call, so it is the natural carrier and no intermediate
    /// signature changes.
    ///
    /// Empty ([`Nvfp4Context::none`]) on the baseline plan, on CPU, and below `sm_120` — all of which
    /// simply take the dequant→bf16 fallback. Before this, every W4A4 layer built its own handle and
    /// its own eager 32 MiB workspace: **6.6 GiB across the 260-projection blanket-W4A4 trunk**, none of
    /// it visible to the weights-only SC#6 sum (measured — the real footprint was 0.603×, not 0.2813×).
    ctx: Nvfp4Context,
}

impl Default for DitPlan {
    fn default() -> Self {
        Self {
            quant: None,
            probe: None,
            checked: false,
            num_layers: DEFAULT_NUM_LAYERS,
            ctx: Nvfp4Context::none(),
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
    /// shipping Turbo trunk can rely on the `DEFAULT_NUM_LAYERS` default.
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// The trunk block count this plan is bound to (see [`Self::with_num_layers`]).
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Bind the **one shared** cuBLASLt context every NVFP4 projection of this trunk will use
    /// (sc-12274). Called by [`crate::transformer::Krea2Transformer::load_planned`] with a context
    /// built once from the weights' device; without it each layer would build its own handle and pay
    /// its own 32 MiB workspace.
    pub fn with_nvfp4_context(mut self, ctx: Nvfp4Context) -> Self {
        self.ctx = ctx;
        self
    }

    /// The shared cuBLASLt context (see [`Self::with_nvfp4_context`]) — what the loader hands to
    /// [`candle_gen::quant::Nvfp4Linear::from_dense_in`].
    pub fn nvfp4_context(&self) -> &Nvfp4Context {
        &self.ctx
    }

    /// The [`LayerRole`] this plan assigns `name`, derived from the trunk topology it is bound to.
    pub fn role_for(&self, name: &str) -> LayerRole {
        LayerRole::for_krea_layer(name, self.num_layers)
    }

    /// The activation precision this plan assigns `name`, deriving the [`LayerRole`] from the trunk
    /// topology — **the form the loader uses**, so the role is never stated twice.
    pub fn act_for_layer(&self, name: &str) -> ActPrecision {
        self.act_for(self.role_for(name))
    }

    /// The [`ExecutionRole`] this plan assigns `name` from its structural role alone — the
    /// capability-blind half of [`Self::representation`].
    pub fn execution_role_for_layer(&self, name: &str) -> ExecutionRole {
        self.execution_role(self.role_for(name))
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

    /// The [`ExecutionRole`] this plan assigns a projection with structural role `role`
    /// (sc-12121) — **the production Krea selection**, and the reason no substring heuristic reaches
    /// it: `role` is the only input, so a dotted key cannot participate in the decision at all.
    ///
    /// Each structural fact was verified against real Krea activations (sc-12110) — none is inherited
    /// belief:
    ///
    /// * `is_edge_block` — blocks 0..3 and the last measured Dense on their block **inputs** (crush
    ///   686×); the shared crate's `last_block`/`final_block` markers never matched
    ///   `transformer_blocks.27.` anyway.
    /// * `is_context_read` — `text_fusion` `to_out.0` measured Dense at crush 40145×; the shared
    ///   crate's `caption_projection` / `attn2` / `cross_att*` anchors have no Krea counterpart (it
    ///   is a single-stream DiT with no cross-attention).
    /// * `is_post_nonlinearity` — `ff.down` measured Dense in 28/28 blocks.
    /// * `is_final_proj` — `final_layer.linear`, measured Dense at crush 909×, which no name-only
    ///   anchor fires on (sc-12140).
    /// * `is_unclassified` — a key the role table does not name, which must not reach the packed lane
    ///   by default.
    pub fn execution_role(&self, role: LayerRole) -> ExecutionRole {
        match self.quant {
            Some(Nvfp4Quant::BlanketW4A4) => ExecutionRole::PackedW4A4,
            Some(Nvfp4Quant::BlanketW4A16) | None => {
                ExecutionRole::DenseBf16(DenseReason::DenseRegimeRequested)
            }
            Some(Nvfp4Quant::Mixed) => role.execution_role(),
        }
    }

    /// The activation precision this plan assigns a projection with structural role `role`.
    ///
    /// Public so a validation harness can ask what the shipping policy *would* assign a layer while
    /// probing a **baseline** trunk (sc-12110's partition gate measures unquantized activations, then
    /// crosses the measured class against this assumed one).
    pub fn act_for(&self, role: LayerRole) -> ActPrecision {
        self.execution_role(role).act_precision()
    }

    /// **The decision table** (sc-12121): the representation `name` actually resolves to under the
    /// checkpoint and device facts `cap` carries.
    ///
    /// Precedence is the order the real pipeline settles these questions, so the reported
    /// [`DenseReason`] names the stage that actually decided:
    ///
    /// 1. `full_precision_matrix_mult`, a preserved-dense layer, padded storage, an ineligible grid,
    ///    and a device below the `sm_120` floor are all settled at **plan** time by
    ///    `CandleCodecResidency` — such a row is priced (and materialized) dense, so no packed
    ///    container ever reaches `Nvfp4Linear`.
    /// 2. The plan's requested regime (baseline / blanket W4A16 → dense BF16 by request).
    /// 3. The **structural role** — the outlier-sensitive classes, which run W4A16, i.e. dense BF16.
    /// 4. The construction-time fused-quantizer probe inside `Nvfp4Linear` (sc-12078).
    ///
    /// Only a projection that clears every one of them is served [`ExecutionRole::PackedW4A4`].
    pub fn representation(&self, name: &str, cap: Nvfp4Capability) -> ExecutionRole {
        self.representation_for_role(self.role_for(name), cap)
    }

    /// [`Self::representation`] over an already-derived structural role.
    pub fn representation_for_role(&self, role: LayerRole, cap: Nvfp4Capability) -> ExecutionRole {
        let dense = |reason| ExecutionRole::DenseBf16(reason);
        if cap.full_precision_declared {
            return dense(DenseReason::FullPrecisionDeclared);
        }
        if !cap.checkpoint_offers_nvfp4 {
            return dense(DenseReason::PreservedDense);
        }
        if !cap.storage_unpadded {
            return dense(DenseReason::PaddedStorage);
        }
        if !cap.layout_native {
            return dense(DenseReason::ShapeIneligible);
        }
        if !cap.nvfp4_device {
            return dense(DenseReason::NoNvfp4Hardware);
        }
        match self.execution_role(role) {
            ExecutionRole::DenseBf16(reason) => dense(reason),
            ExecutionRole::PackedW4A4 if !cap.fused_quantizer => {
                dense(DenseReason::NoFusedQuantizer)
            }
            ExecutionRole::PackedW4A4 => ExecutionRole::PackedW4A4,
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
    /// The NVFP4 base wrapped in the ONE shared additive linear (sc-11091 / sc-21483), so a user
    /// LoRA/LoKr rides *unmerged* alongside the packed forward exactly as it does on the dense and
    /// MLX-packed arms. With no adapter attached this is byte-identical to the bare
    /// [`Nvfp4Linear`] forward, so the SC#1/SC#2/SC#6 benches are unchanged.
    inner: Box<AdaptLinear>,
    name: String,
    probe: Option<Arc<ActProbe>>,
    checked: bool,
    act: ActPrecision,
}

impl Nvfp4Proj {
    pub(crate) fn new(inner: Nvfp4Linear, name: &str, plan: &DitPlan, act: ActPrecision) -> Self {
        Self {
            inner: Box::new(AdaptLinear::from_nvfp4(inner)),
            name: name.to_string(),
            probe: plan.probe.clone(),
            checked: plan.checked,
            act,
        }
    }

    /// The adapter-capable host, for the additive installer (sc-21483).
    pub(crate) fn adapt(&self) -> &AdaptLinear {
        &self.inner
    }

    pub(crate) fn adapt_mut(&mut self) -> &mut AdaptLinear {
        &mut self.inner
    }

    /// `y = x·Wᵀ (+ b)` through the NVFP4 path. Records the input activation first when a probe is
    /// attached, then runs the forward (through the NaN guard when the plan asked for it).
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(p) = &self.probe {
            p.record(&self.name, self.act, x)?;
        }
        let y = self.inner.forward(x)?;
        if self.checked {
            // The sc-11044 NaN guard, applied to the projection's FULL output — base plus any
            // additive residual — so an adapter that collapses the signal fails just as loud as a
            // W4A4 collapse would. Same single sum-of-squares reduction as
            // `Nvfp4Linear::forward_checked`, which this replaces now that the residual stack sits
            // between the packed GEMM and the caller.
            let energy = y
                .to_dtype(candle_gen::candle_core::DType::F32)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?;
            if !energy.is_finite() {
                return Err(candle_gen::candle_core::Error::Msg(format!(
                    "krea NVFP4 `{}`: non-finite output (NaN/inf) from the {:?} regime — W4A4 \
                     signal collapse, a bad activation, or a diverging additive residual; failing \
                     loud (sc-11044 NaN guard)",
                    self.name,
                    self.inner
                        .base_nvfp4()
                        .expect("an Nvfp4Proj always holds an NVFP4 base")
                        .regime(),
                )));
            }
        }
        Ok(y)
    }

    /// The underlying NVFP4 linear (for report accounting).
    pub(crate) fn linear(&self) -> &Nvfp4Linear {
        self.inner
            .base_nvfp4()
            .expect("an Nvfp4Proj always holds an NVFP4 base")
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
    /// (`Nvfp4Linear::resident_device_bytes`). Only populated on a cuda build; zero when no layer
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
    use std::collections::BTreeSet;

    /// **`ALL` is every variant** (sc-12121 review fix). `pub const ALL: [Self; 8]` is an array
    /// literal, not an exhaustive construct: a leaf added to the enum but left out of it would be
    /// invisible to the coverage walk while still classifying — the "no leaf without a role-table
    /// row" claim would quietly stop holding.
    ///
    /// So `ALL` is crossed against the exhaustive [`BlockLeaf::ordinal`] match (which a new variant
    /// cannot compile without extending) in both directions: every ordinal `0..ALL.len()` names a
    /// leaf, and every leaf's ordinal is its position in `ALL`. Add a ninth variant and the macro
    /// forces it an ordinal; whatever ordinal it gets, either it collides (RED here) or it is
    /// `ALL.len()`, and `from_ordinal(ALL.len())` then returns `Some` — also RED.
    #[test]
    fn block_leaf_all_is_every_variant() {
        for (i, leaf) in BlockLeaf::ALL.iter().enumerate() {
            assert_eq!(leaf.ordinal(), i, "{leaf:?} is not at its ordinal in ALL");
            assert_eq!(BlockLeaf::from_ordinal(i), Some(*leaf));
        }
        assert_eq!(
            BlockLeaf::from_ordinal(BlockLeaf::ALL.len()),
            None,
            "the exhaustive ordinal match names more leaves than `BlockLeaf::ALL` lists — a variant \
             was added to the enum without being added to ALL"
        );
        // Every key round-trips, and the eight keys are distinct.
        for leaf in BlockLeaf::ALL {
            assert_eq!(BlockLeaf::from_leaf_key(leaf.leaf_key()), Some(leaf));
            assert_eq!(
                leaf.leaf_key(),
                format!("{}.{}", leaf.module(), leaf.module_leaf()),
                "leaf_key must be module + module_leaf composed"
            );
        }
        let keys: BTreeSet<&str> = BlockLeaf::ALL.iter().map(|l| l.leaf_key()).collect();
        assert_eq!(keys.len(), BlockLeaf::ALL.len(), "duplicate leaf keys");
        // ATTN + FF partition ALL, and each sub-list is one module.
        let split: Vec<BlockLeaf> = BlockLeaf::ATTN
            .into_iter()
            .chain(BlockLeaf::FF)
            .collect::<Vec<_>>();
        assert_eq!(split, BlockLeaf::ALL.to_vec());
        assert!(BlockLeaf::ATTN.iter().all(|l| l.module() == "attn"));
        assert!(BlockLeaf::FF.iter().all(|l| l.module() == "ff"));
        // The post-nonlinearity partition is the sc-12110 finding, pinned.
        let post: BTreeSet<&str> = BlockLeaf::ALL
            .iter()
            .filter(|l| l.reads_post_nonlinearity())
            .map(|l| l.leaf_key())
            .collect();
        assert_eq!(
            post,
            BTreeSet::from(["attn.to_out.0", "ff.down"]),
            "the post-nonlinearity class moved"
        );
    }

    /// **The loader's leaf strings ARE the role table's** (sc-12121 review fix).
    ///
    /// The original review found the coverage walk building block keys from `BlockLeaf::leaf_key`
    /// while claiming to spell them the way the loader does — the table checked against itself. The
    /// fix made `GatedAttention::load_planned` / `SwiGlu::load_planned` read their suffixes from
    /// [`BlockLeaf::module_leaf`], so there is one source. This test proves the loop closes against
    /// something the table does **not** own:
    ///
    /// 1. `testfix::tiny_transformer` writes a real safetensors file whose keys are spelled by hand
    ///    (`{prefix}.attn.to_q`, `{prefix}.ff.down`, …) and loads a real trunk through the real
    ///    loader — so a table typo makes the load fail outright.
    /// 2. `native_mapping::BLOCK_DIFFUSERS_LEAVES` is the checkpoint namespace's own independent
    ///    enumeration of per-block leaves; the eight GEMM leaves must each appear there.
    ///
    /// Mutate any one of the eight `module_leaf` literals and both halves go RED.
    #[test]
    fn loader_leaf_literals_match_the_role_table() {
        // (2) — the checkpoint namespace's list, which nothing in `nvfp4_dit` produces.
        let checkpoint: BTreeSet<&str> = crate::native_mapping::BLOCK_DIFFUSERS_LEAVES
            .iter()
            .copied()
            .collect();
        for leaf in BlockLeaf::ALL {
            let weight_key = format!("{}.weight", leaf.leaf_key());
            assert!(
                checkpoint.contains(weight_key.as_str()),
                "{weight_key} is not a per-block leaf the Krea checkpoint namespace names — the \
                 role table and the checkpoint's key spelling have drifted"
            );
        }

        // (1) — a real load through the real loader, keys spelled independently by the fixture.
        let tmp = tempfile::tempdir().unwrap();
        let (dit, _cfg) = crate::testfix::tiny_transformer(&tmp);
        let loaded: BTreeSet<String> = dit
            .nvfp4_layer_names()
            .into_iter()
            .filter_map(|n| n.strip_prefix("transformer_blocks.0.").map(str::to_string))
            .collect();
        let table: BTreeSet<String> = BlockLeaf::ALL
            .iter()
            .map(|l| l.leaf_key().to_string())
            .collect();
        assert_eq!(
            loaded, table,
            "the block's loaded projection keys and `BlockLeaf::ALL` disagree — a key that stops \
             classifying silently downgrades to DenseReason::Unclassified"
        );
    }

    /// **Only the two unpredictable causes are excused** (sc-12121 review fix).
    ///
    /// `dense_reason_for_fallback` is the whole width of the hole `check_representation` allows in
    /// its refusal, so the set of causes it maps must be exactly the two `Nvfp4Capability` cannot
    /// model. Widen it to a cause the capability facts DO model — say `NoFusedQuantizer`, which
    /// `Nvfp4Capability::fused_quantizer` predicts — and a real prediction defect stops being an
    /// error, which is the coverage lie this story exists to delete.
    #[test]
    fn only_the_unpredictable_fallback_causes_are_excused() {
        let excused: Vec<Nvfp4Fallback> = [
            Nvfp4Fallback::W4A16Requested,
            Nvfp4Fallback::NotCudaDevice,
            Nvfp4Fallback::ShapeIneligible,
            Nvfp4Fallback::NoDeviceHandle,
            Nvfp4Fallback::DeviceMismatch,
            Nvfp4Fallback::NoFusedQuantizer,
            Nvfp4Fallback::StagingFailed,
        ]
        .into_iter()
        .filter(|c| dense_reason_for_fallback(*c).is_some())
        .collect();
        assert_eq!(
            excused,
            vec![Nvfp4Fallback::DeviceMismatch, Nvfp4Fallback::StagingFailed],
            "exactly the two runtime accidents no per-key capability probe can see may be excused"
        );
        assert_eq!(
            dense_reason_for_fallback(Nvfp4Fallback::DeviceMismatch),
            Some(DenseReason::DeviceMismatch)
        );
        assert_eq!(
            dense_reason_for_fallback(Nvfp4Fallback::StagingFailed),
            Some(DenseReason::StagingFailed)
        );
    }

    #[test]
    fn baseline_plan_is_not_nvfp4_and_blanket_plans_force_their_regime() {
        assert!(!DitPlan::baseline().is_nvfp4());
        let w4a4 = DitPlan::nvfp4(Nvfp4Quant::BlanketW4A4);
        assert!(w4a4.is_nvfp4());
        // A blanket plan ignores the outlier policy — even for a role the policy would flag.
        assert_eq!(w4a4.act_for(LayerRole::context_read()), ActPrecision::W4A4);
        assert_eq!(w4a4.act_for(LayerRole::final_proj()), ActPrecision::W4A4);
        assert_eq!(
            DitPlan::nvfp4(Nvfp4Quant::BlanketW4A16).act_for(LayerRole::interior()),
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
    /// Every dotted key the NVFP4 lane serves on a Krea trunk of `n` single-stream blocks — built
    /// from the role table's own leaf enumeration, in the order the loader visits them.
    ///
    /// **This is the table's inverse, and says so.** The per-block suffixes come from
    /// [`BlockLeaf::leaf_key`], not from a second copy of the loader's literals — because since
    /// sc-12121's review fix the loader has no literals of its own: `GatedAttention::load_planned`
    /// and `SwiGlu::load_planned` call [`BlockLeaf::module_leaf`] and [`BlockLeaf::module`]. There is
    /// one source of the eight strings, so "the loader's surface" and "the table's inverse" are the
    /// same set by construction. What crosses the two is
    /// `loader_leaf_literals_match_the_role_table`, which walks a real loaded block's projection keys
    /// and asserts they are exactly `BlockLeaf::ALL.map(leaf_key)`; this helper then supplies the
    /// non-block keys (`img_in`, `txt_in.*`, `final_layer.linear`) the same way.
    fn krea_lane_surface(n: usize) -> Vec<String> {
        let leaves = || BlockLeaf::ALL.into_iter().map(BlockLeaf::leaf_key);
        let mut names: Vec<String> = vec![
            "img_in".into(),
            "txt_in.linear_1".into(),
            "txt_in.linear_2".into(),
        ];
        for i in 0..n {
            names.extend(leaves().map(|leaf| format!("transformer_blocks.{i}.{leaf}")));
        }
        for kind in ["layerwise_blocks", "refiner_blocks"] {
            for i in 0..2 {
                names.extend(leaves().map(|leaf| format!("text_fusion.{kind}.{i}.{leaf}")));
            }
        }
        names.push("final_layer.linear".into());
        names
    }

    #[test]
    fn measured_partition_yields_the_expected_w4a4_surface() {
        let p = DitPlan::nvfp4(Nvfp4Quant::Mixed);
        let n = DEFAULT_NUM_LAYERS;
        let names = krea_lane_surface(n);
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

    /// **The sc-12140 defect, pinned for Krea — now as a role-table property (sc-12121).** The head is
    /// guarded because [`KreaSite::classify`] names it [`KreaSite::TrunkHead`], not because any name
    /// anchor fires on it. If someone ever drops that row from the table, the key stops classifying
    /// and this test fails — rather than the trunk head silently landing on W4A4.
    #[test]
    fn final_head_is_guarded_by_the_role_table_not_by_its_name() {
        let p = DitPlan::nvfp4(Nvfp4Quant::Mixed);
        assert_eq!(
            KreaSite::classify("final_layer.linear"),
            Some(KreaSite::TrunkHead)
        );
        assert_eq!(p.act_for_layer("final_layer.linear"), ActPrecision::W4A16);
        assert_eq!(
            p.execution_role_for_layer("final_layer.linear"),
            ExecutionRole::DenseBf16(DenseReason::TrunkHead)
        );
        // Hand the plan a role that does NOT state the head, and the same key rides W4A4 — proof the
        // decision comes from the role, not from the spelling.
        assert_eq!(
            p.act_for(LayerRole::interior()),
            ActPrecision::W4A4,
            "no name anchor participates: an interior role is an interior projection"
        );
    }

    /// **AC1 coverage: the role table names every projection the lane serves, and nothing else
    /// reaches the packed lane** (sc-12121).
    ///
    /// The surface is enumerated from the enum itself ([`BlockLeaf::ALL`] × the block sites plus the
    /// three fixed sites), so a leaf added to a block without a role-table row cannot compile, and a
    /// key the loader passes that the table does not name cannot classify.
    #[test]
    fn every_krea_projection_receives_an_explicit_role() {
        let n = DEFAULT_NUM_LAYERS;
        let plan = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(n);
        for name in krea_lane_surface(n) {
            let site = KreaSite::classify(&name).unwrap_or_else(|| {
                panic!("`{name}` is served by the lane but has no role-table row")
            });
            let role = LayerRole::for_krea_layer(&name, n);
            assert!(
                !role.is_unclassified,
                "`{name}` classified as {site:?} but produced an unclassified role"
            );
            assert_eq!(
                role,
                LayerRole::for_site(site, n),
                "`{name}`: the key-parse and the site-table must agree"
            );
            // Every projection gets one of exactly two execution roles — no third state, no default.
            let got = plan.execution_role_for_layer(&name);
            assert!(
                got.is_packed_w4a4() || matches!(got, ExecutionRole::DenseBf16(_)),
                "`{name}` has no execution role"
            );
            assert_eq!(got.act_precision(), plan.act_for_layer(&name));
        }
        // Every site variant and every leaf is exercised by that enumeration (an exhaustive match
        // here is what makes "exhaustive" a compile-time property rather than a claim).
        for leaf in BlockLeaf::ALL {
            for site in [
                KreaSite::Block { index: 7, leaf },
                KreaSite::TextFusion { leaf },
            ] {
                match site {
                    KreaSite::Block { .. } | KreaSite::TextFusion { .. } => {}
                    KreaSite::ImageIngest | KreaSite::ContextIngest | KreaSite::TrunkHead => {}
                }
                assert_eq!(
                    LayerRole::for_site(site, DEFAULT_NUM_LAYERS).is_post_nonlinearity,
                    leaf.reads_post_nonlinearity()
                );
            }
        }
    }

    /// **An unrecognised key takes the dense BF16 fallback, never the packed lane** (sc-12121).
    ///
    /// This is the failure mode the substring anchors had: a guard that does not fire leaves an
    /// outlier-carrying layer on W4A4. The role table's answer to "I do not know this key" is dense.
    #[test]
    fn an_unclassified_key_falls_back_to_dense_bf16() {
        let p = DitPlan::nvfp4(Nvfp4Quant::Mixed);
        for name in [
            // Plausible-looking keys that are NOT lane sites: the excluded embedders, the
            // `text_fusion` projector, a control-branch `proj_out`, a leaf that does not exist.
            "time_embed.linear_1",
            "time_mod_proj",
            "text_fusion.projector",
            "transformer_blocks.3.control.proj_out",
            "transformer_blocks.3.attn.to_out",
            "transformer_blocks.x.attn.to_q",
            "text_fusion.layerwise_blocks.q.ff.up",
            "",
        ] {
            assert_eq!(KreaSite::classify(name), None, "`{name}` must not classify");
            assert!(LayerRole::for_krea_layer(name, DEFAULT_NUM_LAYERS).is_unclassified);
            assert_eq!(
                p.execution_role_for_layer(name),
                ExecutionRole::DenseBf16(DenseReason::Unclassified),
                "`{name}` is unknown and must fall back to dense bf16"
            );
            assert_eq!(p.act_for_layer(name), ActPrecision::W4A16);
        }
    }

    /// **AC2: the packed-versus-dense decision table.** Each capability miss selects dense BF16 with
    /// the reason that names the stage which actually decided, and only the fully eligible row on a
    /// benign structural role reaches packed W4A4.
    #[test]
    fn capability_misses_each_select_dense_bf16_with_their_own_reason() {
        let p = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(DEFAULT_NUM_LAYERS);
        let benign = "transformer_blocks.7.attn.to_q";
        // The one row that reaches the packed lane.
        assert_eq!(
            p.representation(benign, Nvfp4Capability::ELIGIBLE),
            ExecutionRole::PackedW4A4
        );

        let cases: [(Nvfp4Capability, DenseReason); 6] = [
            (
                Nvfp4Capability {
                    full_precision_declared: true,
                    ..Nvfp4Capability::ELIGIBLE
                },
                DenseReason::FullPrecisionDeclared,
            ),
            (
                Nvfp4Capability {
                    checkpoint_offers_nvfp4: false,
                    ..Nvfp4Capability::ELIGIBLE
                },
                DenseReason::PreservedDense,
            ),
            (
                Nvfp4Capability {
                    storage_unpadded: false,
                    ..Nvfp4Capability::ELIGIBLE
                },
                DenseReason::PaddedStorage,
            ),
            (
                Nvfp4Capability {
                    layout_native: false,
                    ..Nvfp4Capability::ELIGIBLE
                },
                DenseReason::ShapeIneligible,
            ),
            (Nvfp4Capability::NO_HARDWARE, DenseReason::NoNvfp4Hardware),
            (
                Nvfp4Capability {
                    fused_quantizer: false,
                    ..Nvfp4Capability::ELIGIBLE
                },
                DenseReason::NoFusedQuantizer,
            ),
        ];
        for (cap, reason) in cases {
            assert_eq!(
                p.representation(benign, cap),
                ExecutionRole::DenseBf16(reason),
                "{cap:?} must select dense bf16 as {reason:?}"
            );
        }

        // The outlier-sensitive structural classes select dense BF16 even on a fully eligible device.
        for (name, reason) in [
            ("final_layer.linear", DenseReason::TrunkHead),
            ("txt_in.linear_1", DenseReason::ContextRead),
            (
                "text_fusion.refiner_blocks.0.attn.to_q",
                DenseReason::ContextRead,
            ),
            (
                "transformer_blocks.7.ff.down",
                DenseReason::PostNonlinearity,
            ),
            ("transformer_blocks.0.attn.to_q", DenseReason::EdgeBlock),
            ("transformer_blocks.27.ff.up", DenseReason::EdgeBlock),
            ("nothing.the.table.names", DenseReason::Unclassified),
        ] {
            assert_eq!(
                p.representation(name, Nvfp4Capability::ELIGIBLE),
                ExecutionRole::DenseBf16(reason),
                "{name} is outlier-sensitive by structure and must not ride W4A4"
            );
        }

        // A blanket-W4A4 bench ignores the ROLE but never the capability facts…
        let blanket = DitPlan::nvfp4(Nvfp4Quant::BlanketW4A4);
        assert_eq!(
            blanket.representation("final_layer.linear", Nvfp4Capability::ELIGIBLE),
            ExecutionRole::PackedW4A4
        );
        assert_eq!(
            blanket.representation("final_layer.linear", Nvfp4Capability::NO_HARDWARE),
            ExecutionRole::DenseBf16(DenseReason::NoNvfp4Hardware)
        );
        // …and the storage tier / baseline are dense bf16 by request, on any device.
        for plan in [
            DitPlan::nvfp4(Nvfp4Quant::BlanketW4A16),
            DitPlan::baseline(),
        ] {
            assert_eq!(
                plan.representation(benign, Nvfp4Capability::ELIGIBLE),
                ExecutionRole::DenseBf16(DenseReason::DenseRegimeRequested)
            );
        }
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
