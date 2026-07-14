//! NVFP4 activation-outlier **sparsity** instrumentation (sc-11044, epic 11037).
//!
//! This implements the empirical gate the master-gate spike (sc-11038) could only emulate: given the
//! activations flowing through the FP4 path, measure — per layer — **how many NVFP4 16-blocks carry a
//! massive-activation outlier**. That sparsity is the quantity that decides whether W4A4 is safe on a
//! layer (spike sc-11038 / the sc-7702 collapse mechanism):
//!
//! > an activation outlier sharing a 16-block crushes its co-located channels to E2M1 zero — damage
//! > scales with outlier **sparsity**: benign ≈0.99 cosine, ~2 sparse "massive activations" ≈0.984,
//! > 8 → ≈0.966, dense → collapse (0.0).
//!
//! The mechanism is per-block: NVFP4 gives each 16-element block a single UE4M3 scale set by the block
//! **amax**. If one element in a block is ~100–1000× the others (a "massive activation"), the block
//! scale is pinned to that outlier and the other 15 co-located channels round to the coarse E2M1 grid
//! at that huge scale — several of them collapse to zero. So the right per-layer metric is the
//! **fraction of blocks that are clean** (no massive outlier); a layer whose activations put an
//! outlier in *most* blocks (dense) collapses under W4A4 and must stay bf16-activation (W4A16).
//!
//! This module is **backend-neutral** — it operates on a materialized host `f32` slice (or a [`Tensor`]
//! moved to CPU) — so it compiles and is unit-tested on the CPU lane, and can be called on a live
//! denoise's activations on the GPU rig (moving each captured activation to host for the small
//! reduction). It does **not** quantize; it only measures the sparsity that governs the partition.

use candle_core::{Result, Tensor};

/// The W4A4 stability class the measured outlier sparsity implies for a layer (sc-11044).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutlierClass {
    /// Almost every 16-block is clean (benign fraction ≥ [`OutlierSparsity::BENIGN_FLOOR`]). W4A4 is
    /// safe here — this is the compute-bulk (self-attn + FF) the ~2× win rides.
    Benign,
    /// A minority of blocks carry an outlier ([`OutlierSparsity::DENSE_FLOOR`] ≤ benign fraction <
    /// [`OutlierSparsity::BENIGN_FLOOR`]). W4A4 degrades but does not collapse — usable with a quality
    /// budget; the spike's "sparse massive activations" regime.
    Sparse,
    /// Outliers are dense (benign fraction < [`OutlierSparsity::DENSE_FLOOR`]). W4A4 collapses (the
    /// sc-7702 mechanism); this layer must stay **W4A16** (bf16 activation) — the outlier class.
    Dense,
}

/// Per-layer activation-outlier sparsity over the NVFP4 16-blocks (sc-11044). Built from a layer's
/// activation tensor; the load-bearing field is [`Self::benign_fraction`] (fraction of 16-blocks with
/// **no** massive-activation outlier), which classifies the layer's W4A4 stability ([`Self::class`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlierSparsity {
    /// Total number of 16-element blocks measured (`rows · ceil(cols / 16)`; a short final block is
    /// counted).
    pub total_blocks: usize,
    /// Blocks containing at least one element whose magnitude exceeds `tau · robust_scale` — a
    /// massive-activation outlier.
    pub outlier_blocks: usize,
    /// `1 - outlier_blocks / total_blocks` — the fraction of clean blocks. The spike's sparsity metric
    /// (≈0.99 benign, 0.98–0.97 sparse, → collapse dense).
    pub benign_fraction: f64,
    /// The robust per-tensor magnitude scale the outlier threshold is relative to (the median of the
    /// nonzero `|x|`, robust to the outliers themselves).
    pub robust_scale: f32,
    /// The largest per-block dynamic range (`block_amax / block_median_abs`) seen — how hard the worst
    /// block crushes its co-located channels.
    pub max_crush_ratio: f32,
    /// The outlier multiplier used (`tau`): an element is an outlier iff `|x| > tau · robust_scale`.
    pub tau: f32,
}

impl OutlierSparsity {
    /// NVFP4 block size along the contraction axis (16 elements share one UE4M3 scale).
    pub const BLOCK: usize = 16;

    /// Default "massive activation" multiplier: `|x| > 20 · median(|x|)` is an outlier. Massive
    /// activations are typically 10²–10³× the typical magnitude; 20× robustly separates them from the
    /// bulk without flagging ordinary heavy tails.
    pub const DEFAULT_TAU: f32 = 20.0;

    /// Benign-fraction floor for [`OutlierClass::Benign`] (W4A4-safe). Above this almost every block is
    /// clean — the spike's benign regime.
    pub const BENIGN_FLOOR: f64 = 0.995;

    /// Benign-fraction floor for [`OutlierClass::Sparse`]; below it the layer is [`OutlierClass::Dense`]
    /// and W4A4 collapses (must stay W4A16). Set at the knee where the spike saw the collapse steepen.
    pub const DENSE_FLOOR: f64 = 0.98;

    /// Measure the outlier sparsity of a `[rows, cols]` row-major activation slice, blocking each row
    /// into 16-element NVFP4 blocks along `cols`. `tau` is the outlier multiplier (see
    /// [`Self::DEFAULT_TAU`]).
    pub fn from_slice(data: &[f32], rows: usize, cols: usize, tau: f32) -> Self {
        assert_eq!(data.len(), rows * cols, "data length must be rows * cols");

        // Robust per-tensor scale = median of the nonzero magnitudes (robust to the outliers we are
        // trying to detect; a plain mean/RMS would be inflated by them).
        let mut mags: Vec<f32> = data.iter().map(|v| v.abs()).filter(|&m| m > 0.0).collect();
        let robust_scale = if mags.is_empty() {
            0.0
        } else {
            mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            mags[mags.len() / 2]
        };
        let threshold = tau * robust_scale;

        let blocks_per_row = cols.div_ceil(Self::BLOCK);
        let total_blocks = rows * blocks_per_row;
        let mut outlier_blocks = 0usize;
        let mut max_crush_ratio = 0f32;

        for r in 0..rows {
            let row = &data[r * cols..r * cols + cols];
            for b in 0..blocks_per_row {
                let c0 = b * Self::BLOCK;
                let c1 = (c0 + Self::BLOCK).min(cols);
                let block = &row[c0..c1];
                let mut amax = 0f32;
                let mut hit = false;
                let mut babs: Vec<f32> = Vec::with_capacity(block.len());
                for &v in block {
                    let a = v.abs();
                    amax = amax.max(a);
                    babs.push(a);
                    // An outlier only "crushes" a block if it shares the block with other channels;
                    // threshold > 0 guards the degenerate all-zero-scale tensor.
                    if threshold > 0.0 && a > threshold {
                        hit = true;
                    }
                }
                if hit {
                    outlier_blocks += 1;
                    // Crush ratio = block amax / block median magnitude (dynamic range within the block).
                    babs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let med = babs[babs.len() / 2];
                    if med > 0.0 {
                        max_crush_ratio = max_crush_ratio.max(amax / med);
                    }
                }
            }
        }

        let benign_fraction = if total_blocks == 0 {
            1.0
        } else {
            1.0 - outlier_blocks as f64 / total_blocks as f64
        };

        Self {
            total_blocks,
            outlier_blocks,
            benign_fraction,
            robust_scale,
            max_crush_ratio,
            tau,
        }
    }

    /// Measure from a `[.., cols]` activation [`Tensor`] (any device/dtype) — the trailing dim is the
    /// contraction axis blocked into 16s; all leading dims flatten into rows. Materialized to a host
    /// `f32` slice for the reduction (instrumentation, not a hot path).
    pub fn from_tensor(x: &Tensor, tau: f32) -> Result<Self> {
        let cols = x.dims().last().copied().unwrap_or(0);
        let rows = x.elem_count().checked_div(cols).unwrap_or(0);
        let data = x
            .to_dtype(candle_core::DType::F32)?
            .to_device(&candle_core::Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(Self::from_slice(&data, rows, cols, tau))
    }

    /// The W4A4 stability class implied by [`Self::benign_fraction`] (see [`OutlierClass`]).
    pub fn class(&self) -> OutlierClass {
        if self.benign_fraction >= Self::BENIGN_FLOOR {
            OutlierClass::Benign
        } else if self.benign_fraction >= Self::DENSE_FLOOR {
            OutlierClass::Sparse
        } else {
            OutlierClass::Dense
        }
    }

    /// True iff this layer's measured sparsity is compatible with running **W4A4** (benign or sparse,
    /// not dense-collapse). The partition "holds" for a benign-classified (W4A4) layer iff this is true.
    pub fn w4a4_viable(&self) -> bool {
        !matches!(self.class(), OutlierClass::Dense)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prng(seed: &mut u64) -> f32 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// A benign activation (no massive outliers) reads as ~1.0 benign fraction → Benign → W4A4-viable.
    #[test]
    fn benign_activation_is_w4a4_viable() {
        let (rows, cols) = (64, 256);
        let mut seed = 0xBEEF_0001u64;
        let data: Vec<f32> = (0..rows * cols).map(|_| prng(&mut seed) * 0.3).collect();
        let s = OutlierSparsity::from_slice(&data, rows, cols, OutlierSparsity::DEFAULT_TAU);
        assert_eq!(s.class(), OutlierClass::Benign, "benign fraction = {}", s.benign_fraction);
        assert!(s.w4a4_viable());
        assert!(s.benign_fraction > 0.995);
    }

    /// A few sparse massive activations (one per handful of blocks) → Sparse: W4A4 still viable, but the
    /// benign fraction has dropped off 1.0 and the crush ratio is large.
    #[test]
    fn sparse_massive_activations_are_sparse_class() {
        let (rows, cols) = (64, 256); // 16 blocks/row · 64 = 1024 blocks
        let mut seed = 0xBEEF_0002u64;
        let mut data: Vec<f32> = (0..rows * cols).map(|_| prng(&mut seed) * 0.3).collect();
        // Inject ~1% of blocks with a massive outlier (one element ~200× the bulk median).
        for r in 0..rows {
            if r % 6 == 0 {
                data[r * cols + 3] = 80.0; // lands in block 0 of that row
            }
        }
        let s = OutlierSparsity::from_slice(&data, rows, cols, OutlierSparsity::DEFAULT_TAU);
        assert!(s.outlier_blocks > 0, "should detect injected outliers");
        assert!(
            s.benign_fraction < OutlierSparsity::BENIGN_FLOOR
                && s.benign_fraction >= OutlierSparsity::DENSE_FLOOR,
            "sparse benign fraction {} not in [{}, {})",
            s.benign_fraction,
            OutlierSparsity::DENSE_FLOOR,
            OutlierSparsity::BENIGN_FLOOR
        );
        assert_eq!(s.class(), OutlierClass::Sparse);
        assert!(s.w4a4_viable());
        assert!(s.max_crush_ratio > 50.0, "crush ratio {}", s.max_crush_ratio);
    }

    /// Dense outliers (an outlier in most blocks — a caption/cross-attn-style feature) → Dense: W4A4
    /// collapses, so it is NOT W4A4-viable (must stay W4A16).
    #[test]
    fn dense_outliers_flag_collapse_not_viable() {
        let (rows, cols) = (64, 256);
        let mut seed = 0xBEEF_0003u64;
        let mut data: Vec<f32> = (0..rows * cols).map(|_| prng(&mut seed) * 0.3).collect();
        // Put a massive outlier in (almost) every block — a persistent-channel "massive activation".
        let blocks_per_row = cols / OutlierSparsity::BLOCK;
        for r in 0..rows {
            for b in 0..blocks_per_row {
                data[r * cols + b * OutlierSparsity::BLOCK + 2] = 120.0;
            }
        }
        let s = OutlierSparsity::from_slice(&data, rows, cols, OutlierSparsity::DEFAULT_TAU);
        assert!(s.benign_fraction < OutlierSparsity::DENSE_FLOOR, "bf = {}", s.benign_fraction);
        assert_eq!(s.class(), OutlierClass::Dense);
        assert!(!s.w4a4_viable(), "dense-outlier layer must not be W4A4-viable");
    }

    /// The metric runs through a `[.., cols]` tensor and agrees with the slice path.
    #[test]
    fn from_tensor_matches_slice() -> Result<()> {
        let (rows, cols) = (8, 64);
        let mut seed = 0xBEEF_0004u64;
        let data: Vec<f32> = (0..rows * cols).map(|_| prng(&mut seed)).collect();
        let t = Tensor::from_vec(data.clone(), (rows, cols), &candle_core::Device::Cpu)?;
        let a = OutlierSparsity::from_tensor(&t, OutlierSparsity::DEFAULT_TAU)?;
        let b = OutlierSparsity::from_slice(&data, rows, cols, OutlierSparsity::DEFAULT_TAU);
        assert_eq!(a.total_blocks, b.total_blocks);
        assert_eq!(a.outlier_blocks, b.outlier_blocks);
        Ok(())
    }
}
