//! The NVFP4 activation-outlier probe (sc-11045) and its per-layer summary, shared by the SANA and
//! Krea NVFP4 DiT plans (sc-27413: previously one verbatim copy per provider).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use candle_core::{Result, Tensor};

use crate::lock_recover;
use crate::quant::{ActPrecision, OutlierClass, OutlierSparsity};

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

/// Records per-layer, per-step activation-outlier sparsity across a live denoise (sc-11045; shared by
/// the SANA and Krea NVFP4 trunks, sc-27413).
///
/// The spike (sc-11038) established that NVFP4 W4A4 damage scales with activation-outlier **sparsity**
/// and partitioned layers on that basis — but only ever measured *synthetic* activations. This probe
/// closes that gate by measuring the real thing: attach it to a provider's NVFP4 DiT plan, run a real
/// denoise, then read [`Self::records`] to see whether every layer the policy sends to W4A4 actually
/// measures W4A4-viable on that model's naming and topology.
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
    pub fn record(&self, layer: &str, act: ActPrecision, x: &Tensor) -> Result<()> {
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
