//! The **NVFP4 precision seam** for the SANA Linear-DiT trunk (sc-11045, epic 11037).
//!
//! [`crate::transformer::SanaTransformer`] loads its projections dense (f32 [`Linear`]) by default.
//! This module adds the seam that lets the SAME trunk serve those projections through
//! [`Nvfp4Linear`] instead — the sc-11041 packed-forward NVFP4 path — so a **real Sana-1.6B denoise**
//! can be run end-to-end on the FP4 tensor cores and compared against the dense f32 baseline.
//!
//! Two things live here:
//!
//! 1. [`DitPlan`] — how to serve the trunk's projections: dense (the byte-unchanged default), or
//!    NVFP4 under a [`Nvfp4Quant`] regime (the sc-11038 per-layer mixed policy, or a blanket
//!    W4A4/W4A16 for a controlled bench).
//! 2. [`ActProbe`] — a per-layer, per-step **activation-outlier sparsity** recorder. This is the
//!    empirical gate the sc-11038 spike could not close with synthetic activations: it measures
//!    [`OutlierSparsity`] of the *real* activation entering every projection at every denoise step, so
//!    the benign→W4A4 / outlier→W4A16 partition can be checked against a live model instead of
//!    assumed.
//!
//! **What is and is not quantized.** The seam covers the trunk's `nn.Linear` GEMMs: the self-attention
//! (`attn1`) and cross-attention (`attn2`) q/k/v/out projections, `caption_projection.linear_{1,2}`,
//! and `proj_out` — 163 projections on SANA-1.6B (20 blocks × 8 + 3). It deliberately does **not**
//! cover:
//!
//! * the **Mix-FFN**, which in SANA is a `GLUMBConv` built from *convolutions* (1×1 inverted, 3×3
//!   depthwise, 1×1 point), not linears — so a meaningful slice of the trunk's FLOPs sits outside the
//!   NVFP4 lane by construction (an honest limit on any end-to-end multiple; see the crate README);
//! * the timestep / guidance embedders (`[B, 256] → [B, dim]`, batch-1 shapes), where the FP4 GEMM has
//!   nothing to win and M-padding to 16 would dominate.
//!
//! Everything degrades cleanly: an [`Nvfp4Linear`] on a non-`sm_120` device, on CPU, or on a non-cuda
//! build transparently serves the dequant→bf16 fallback (sc-11041), so a `DitPlan::nvfp4(..)` trunk
//! still *runs* everywhere — it just does not light the FP4 cores. That is the SC#4 Blackwell-only
//! gate, observed at model level by [`Nvfp4Report::fp4_lit`].

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{Result, Tensor};
use candle_gen::candle_nn::{Linear, Module};
use candle_gen::lock_recover;
use candle_gen::quant::{
    ActPrecision, Nvfp4Context, Nvfp4Linear, Nvfp4Regime, OutlierClass, OutlierSparsity,
};

/// How the trunk should serve one projection's activations when running NVFP4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Quant {
    /// The **sc-11038 mixed-precision policy** (the shipping default): the outlier-carrying class runs
    /// W4A16 (bf16 activation), the benign compute-bulk runs W4A4. Classified by
    /// [`ActPrecision::for_outlier_layer`] on the projection's dotted key, plus SANA's first/last
    /// transformer block (which the shared substring policy cannot name — see
    /// [`DitPlan::act_for`]).
    Mixed,
    /// Blanket W4A4 on every eligible projection — ignores the outlier policy. For a controlled
    /// throughput/stability bench of the FP4 compute path, **not** a shipping regime.
    BlanketW4A4,
    /// Blanket W4A16 on every eligible projection — the NVFP4 *storage* tier (weights packed, bf16
    /// activation, no FP4 compute). The stability-fallback default.
    BlanketW4A16,
}

/// The **structural facts** about a projection that its dotted key cannot carry, threaded from the
/// loader — which knows the trunk's topology — into the shared substring policy (sc-11045).
///
/// This is the seam that keeps the policy honest. A name like `proj_out` is ambiguous across the
/// workspace (SANA's trunk head vs LTX's per-block `ff.proj_out`); rather than sharpen the substring
/// until it happens to fit SANA, the provider states the fact. Both flags default to `false`, i.e.
/// "an ordinary interior projection".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayerRole {
    /// This projection lives in SANA's **first or last** transformer block — the edges the spike
    /// sc-11038 policy keeps on bf16 activation. Not nameable by the shared policy, whose
    /// `last_block`/`final_block` markers do not match SANA's `transformer_blocks.{i}` keys.
    pub is_edge_block: bool,
    /// This projection is the trunk's **final output projection** (the head) — SANA's top-level
    /// `proj_out`, measured Dense (crush 438×) on real activations, *not* a per-block layer that
    /// merely spells itself `proj_out`.
    pub is_final_proj: bool,
}

impl LayerRole {
    /// An interior projection: neither an edge block nor the final head.
    pub fn interior() -> Self {
        Self::default()
    }

    /// An interior projection in SANA's first/last transformer block.
    pub fn edge_block(is_edge_block: bool) -> Self {
        Self {
            is_edge_block,
            is_final_proj: false,
        }
    }

    /// The trunk's final output projection.
    pub fn final_proj() -> Self {
        Self {
            is_edge_block: false,
            is_final_proj: true,
        }
    }
}

/// How to serve the trunk's projections (sc-11045). Default: dense f32 — the byte-unchanged baseline.
#[derive(Clone, Default)]
pub struct DitPlan {
    quant: Option<Nvfp4Quant>,
    probe: Option<Arc<ActProbe>>,
    checked: bool,
    /// The **one** cuBLASLt handle every NVFP4 projection in this trunk shares (sc-12274).
    ///
    /// Set once by [`crate::transformer::SanaTransformer::from_weights_planned`]. Empty
    /// ([`Nvfp4Context::none`]) on the dense plan, on CPU, and below `sm_120` — all of which take the
    /// dequant→bf16 fallback. Before this, every W4A4 layer built its own handle with its own eager
    /// 32 MiB workspace; SANA's 163 projections meant ~5.1 GiB of duplicated scratch, invisible to the
    /// weights-only SC#6 sum. Measured on Krea's 260-projection trunk: real footprint 0.603×, reported
    /// 0.2813×.
    ctx: Nvfp4Context,
}

impl DitPlan {
    /// The dense f32 trunk — exactly what [`crate::transformer::SanaTransformer::from_weights`] builds.
    pub fn dense() -> Self {
        Self::default()
    }

    /// Bind the **one shared** cuBLASLt context every NVFP4 projection of this trunk will use
    /// (sc-12274). Called by [`crate::transformer::SanaTransformer::from_weights_planned`].
    pub fn with_nvfp4_context(mut self, ctx: Nvfp4Context) -> Self {
        self.ctx = ctx;
        self
    }

    /// The shared cuBLASLt context — what the loader hands to
    /// [`candle_gen::quant::Nvfp4Linear::from_dense_in`].
    pub fn nvfp4_context(&self) -> &Nvfp4Context {
        &self.ctx
    }

    /// Serve every eligible projection through [`Nvfp4Linear`] under `quant`.
    pub fn nvfp4(quant: Nvfp4Quant) -> Self {
        Self {
            quant: Some(quant),
            ..Self::default()
        }
    }

    /// Attach an [`ActProbe`]: every projection records the outlier sparsity of its **input
    /// activation** on each forward. Works on the dense plan too — that is how the *unperturbed* real
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

    /// The activation precision this plan assigns `name`, given the structural facts SANA's loader
    /// knows and a dotted key cannot carry.
    ///
    /// [`ActPrecision::for_outlier_layer_with`] carries the shared substring policy
    /// (caption_projection, the cross-attention block, `blocks.0.`). Two facts are threaded in here
    /// instead, because only the provider knows them:
    ///
    /// * `is_edge_block` — SANA's **last** block. The shared policy matches literal
    ///   `last_block`/`final_block` markers, while SANA keys are `transformer_blocks.{i}`, so the
    ///   last-block half of the spike's first/last rule is applied here, where the block count is known.
    /// * `is_final_proj` — SANA's **final output projection**. The shared policy will not fire on a
    ///   per-block layer that merely spells itself `proj_out` (LTX's `ff.proj_out`, Flux/Chroma's
    ///   `single_transformer_blocks.{i}.proj_out` — the benign W4A4 compute bulk), so the trunk head
    ///   is named explicitly rather than inferred from a substring (sc-11045 review, MAJOR 1).
    ///
    /// Public so a validation harness can ask what the shipping policy *would* assign a layer while
    /// probing a **dense** trunk (sc-11045's residual gate measures unquantized activations, then
    /// crosses the measured class against this assumed one).
    pub fn act_for(&self, name: &str, role: LayerRole) -> ActPrecision {
        match self.quant {
            Some(Nvfp4Quant::BlanketW4A4) => ActPrecision::W4A4,
            Some(Nvfp4Quant::BlanketW4A16) => ActPrecision::W4A16,
            // Mixed: the shared policy, plus the structural facts only the loader knows.
            Some(Nvfp4Quant::Mixed) => {
                if role.is_edge_block {
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
    /// The projection's dotted key (e.g. `transformer_blocks.7.attn2.to_k`).
    pub layer: String,
    /// The denoise step index the recorder was set to ([`ActProbe::set_step`]).
    pub step: usize,
    /// The activation-precision the plan assigned this projection — so a report can cross the
    /// *measured* class against the *assumed* partition.
    pub act: ActPrecision,
    /// The measured outlier sparsity of the input activation.
    pub sparsity: OutlierSparsity,
}

/// Records per-layer, per-step activation-outlier sparsity across a live denoise (sc-11045).
///
/// The spike (sc-11038) established that NVFP4 W4A4 damage scales with activation-outlier **sparsity**
/// and partitioned layers on that basis — but only ever measured *synthetic* activations. This probe
/// closes that gate by measuring the real thing: attach it to a [`DitPlan`], run a real denoise, then
/// read [`Self::records`] to see whether every layer the policy sends to W4A4 actually measures
/// W4A4-viable.
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

/// One trunk projection, served either dense (f32 [`Linear`]) or through [`Nvfp4Linear`].
///
/// The dense arm is the pre-existing behaviour verbatim; the NVFP4 arm is the sc-11041 packed-forward
/// path (which itself falls back to dequant→bf16 off `sm_120`).
pub(crate) enum SanaProj {
    Dense(Linear),
    Nvfp4(Box<Nvfp4Linear>),
}

/// A trunk projection plus its instrumentation (name / probe / NaN-guard flag).
pub(crate) struct Proj {
    inner: SanaProj,
    name: String,
    probe: Option<Arc<ActProbe>>,
    checked: bool,
    act: ActPrecision,
}

impl Proj {
    pub(crate) fn new(inner: SanaProj, name: &str, plan: &DitPlan, act: ActPrecision) -> Self {
        Self {
            inner,
            name: name.to_string(),
            probe: plan.probe.clone(),
            checked: plan.checked,
            act,
        }
    }

    /// `y = x·Wᵀ (+ b)`. Records the input activation first when a probe is attached, then runs the
    /// dense or NVFP4 forward (the latter through the NaN guard when the plan asked for it).
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(p) = &self.probe {
            p.record(&self.name, self.act, x)?;
        }
        match &self.inner {
            SanaProj::Dense(l) => l.forward(x),
            SanaProj::Nvfp4(l) => {
                if self.checked {
                    l.forward_checked(x)
                } else {
                    l.forward(x)
                }
            }
        }
    }

    /// The NVFP4 leg, when this projection is quantized.
    fn nvfp4(&self) -> Option<&Nvfp4Linear> {
        match &self.inner {
            SanaProj::Nvfp4(l) => Some(l),
            SanaProj::Dense(_) => None,
        }
    }
}

/// Model-level NVFP4 accounting over a built trunk (sc-11045 SC#6 / SC#4).
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
    /// [`Self::resident_bytes`] for what the run actually costs in VRAM (sc-11045 review, MAJOR 3).
    pub nvfp4_bytes: usize,
    /// Summed bf16 footprint those same weights would occupy dense — the SC#6 comparison baseline.
    pub bf16_bytes: usize,
    /// Summed bytes resident on-device for the **W4A4 (FP4-regime)** weights
    /// (`Nvfp4Linear::resident_device_bytes`). Only populated on a cuda build; zero when no layer
    /// resolved to the packed FP4 path.
    pub resident_fp4_bytes: usize,
    /// Summed bytes resident on-device for the **W4A16 / fallback (dequant→bf16)** weights
    /// ([`Nvfp4Linear::resident_dequant_bf16_bytes`]) — dense bf16, i.e. **no footprint win at all**.
    ///
    /// This is the field that makes the report regime-aware. Every `DequantBf16` layer materializes a
    /// full dense bf16 weight at construction and holds it for life; counting only `nvfp4_bytes` let a
    /// W4A16 run — with *nothing* packed on device — report a 0.28× "NVFP4 footprint".
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
    ///
    /// **Regime-aware since the sc-11045 review (MAJOR 3).** This previously divided
    /// [`Self::nvfp4_bytes`] — the *host* packed container, accumulated for every layer regardless of
    /// regime — by `bf16_bytes`, so it returned ~0.2822 **even for a W4A16 leg that reported 163/163
    /// dequant→bf16 and had not packed a single byte on-device**. It could not fail, and it reported an
    /// NVFP4 footprint for a run that had none.
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

    /// Effective bits per weight implied by the **packed NVFP4 format** (target ≈ 4.5). Like
    /// [`Self::packed_footprint_ratio`], a statement about the container, not about residency.
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

    fn add(&mut self, l: &Nvfp4Linear) {
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

/// Accumulate a report over a trunk's projections.
pub(crate) fn report_over<'a>(projections: impl Iterator<Item = &'a Proj>) -> Nvfp4Report {
    let mut r = Nvfp4Report::default();
    for p in projections {
        if let Some(l) = p.nvfp4() {
            r.add(l);
        }
    }
    r
}

/// A per-layer summary of the probe's records, aggregated across steps (sc-11045 residual gate).
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
    fn dense_plan_is_not_nvfp4_and_blanket_plans_force_their_regime() {
        assert!(!DitPlan::dense().is_nvfp4());
        let w4a4 = DitPlan::nvfp4(Nvfp4Quant::BlanketW4A4);
        assert!(w4a4.is_nvfp4());
        // A blanket plan ignores the outlier policy — even for a name the policy would flag.
        assert_eq!(
            w4a4.act_for("transformer_blocks.3.attn2.to_k", LayerRole::interior()),
            ActPrecision::W4A4
        );
        assert_eq!(
            DitPlan::nvfp4(Nvfp4Quant::BlanketW4A16)
                .act_for("transformer_blocks.3.attn1.to_q", LayerRole::interior()),
            ActPrecision::W4A16
        );
    }

    #[test]
    fn mixed_plan_applies_the_spike_partition_including_sanas_last_block() {
        let p = DitPlan::nvfp4(Nvfp4Quant::Mixed);
        // Benign compute-bulk → W4A4.
        assert_eq!(
            p.act_for("transformer_blocks.7.attn1.to_q", LayerRole::interior()),
            ActPrecision::W4A4
        );
        // Outlier class by the shared substring policy: cross-attn K/V + caption projection.
        assert_eq!(
            p.act_for("transformer_blocks.7.attn2.to_k", LayerRole::interior()),
            ActPrecision::W4A16
        );
        assert_eq!(
            p.act_for("transformer_blocks.7.attn2.to_v", LayerRole::interior()),
            ActPrecision::W4A16
        );
        assert_eq!(
            p.act_for("caption_projection.linear_1", LayerRole::interior()),
            ActPrecision::W4A16
        );
        // First block via the shared policy's `blocks.0.` rule.
        assert_eq!(
            p.act_for("transformer_blocks.0.attn1.to_q", LayerRole::interior()),
            ActPrecision::W4A16
        );
        // SANA's LAST block — which the shared substring policy cannot name — via `is_edge_block`.
        assert_eq!(
            p.act_for(
                "transformer_blocks.19.attn1.to_q",
                LayerRole::edge_block(true)
            ),
            ActPrecision::W4A16
        );
        // SANA's final head — which the shared policy deliberately will NOT infer from the substring
        // (it would also match LTX/Flux/Chroma per-block `proj_out` layers) — via `is_final_proj`.
        assert_eq!(
            p.act_for("proj_out", LayerRole::final_proj()),
            ActPrecision::W4A16
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

    /// **The SC#6 ratio is regime-aware** (sc-11045 review, MAJOR 3): it reports what the run holds in
    /// VRAM, not what the packed container weighs.
    ///
    /// The bug this pins: `footprint_ratio()` used to be `nvfp4_bytes / bf16_bytes`, and `nvfp4_bytes`
    /// accumulates the *host* packed container for every layer regardless of regime. So a W4A16 leg
    /// reporting 163/163 dequant→bf16 — nothing packed on-device, every weight resident as dense bf16
    /// — still printed "footprint 437.56 MiB … ratio 0.2822". The ratio could not fail, and it claimed
    /// an NVFP4 footprint for a run that had none.
    #[test]
    fn footprint_ratio_is_regime_aware_and_never_claims_fp4_for_a_bf16_run() {
        let bf16 = 33_554_432usize;
        let packed = 9_437_184usize;

        // (a) Blanket W4A4: everything staged packed on-device, nothing dequantized. ~0.28×.
        let w4a4 = Nvfp4Report {
            nvfp4_bytes: packed,
            bf16_bytes: bf16,
            resident_fp4_bytes: packed,
            dequant_bf16_bytes: 0,
            ..Default::default()
        };
        assert!((w4a4.footprint_ratio() - 0.28125).abs() < 1e-6);

        // (b) Blanket W4A16 / capability fallback: NOTHING packed on-device; every weight resident as
        // dense bf16. The honest answer is 1.0 — the regime buys stability, not footprint.
        let w4a16 = Nvfp4Report {
            nvfp4_bytes: packed, // the host container is still there — and still irrelevant to VRAM
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
        // The old regime-blind formula is still available, correctly labelled as a format claim.
        assert!((w4a16.packed_footprint_ratio() - 0.28125).abs() < 1e-6);

        // (c) Mixed: half the bytes packed, half held dense bf16 → between the two.
        let mixed = Nvfp4Report {
            nvfp4_bytes: packed,
            bf16_bytes: bf16,
            resident_fp4_bytes: packed / 2,
            dequant_bf16_bytes: bf16 / 2,
            ..Default::default()
        };
        let expected = (packed / 2 + bf16 / 2) as f64 / bf16 as f64;
        assert!((mixed.footprint_ratio() - expected).abs() < 1e-6);
        assert!(mixed.footprint_ratio() > w4a4.footprint_ratio());
        assert!(mixed.footprint_ratio() < w4a16.footprint_ratio());
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
