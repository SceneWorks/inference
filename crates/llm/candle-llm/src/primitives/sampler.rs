//! Token sampling.
//!
//! The Candle port of `mlx-llm`'s sampler (the union of the mlx-gen samplers): temperature + top-p +
//! top-k + repetition/presence penalties, with greedy as the default. The math is host-side and identical to
//! the reference — a stabilised, **unnormalised** `exp((logit - max)/T)` weight per token, heap-based
//! nucleus selection, and a categorical inverse-CDF draw from the pluggable [`TokenRng`]. Greedy
//! (`temperature <= 0`) with no penalty and no constraint takes the on-device argmax fast path
//! (a single-element host transfer); otherwise logits are pulled to host f32.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use candle_core::{DType, Tensor};

use crate::error::Result;

/// A pluggable random source for the categorical draw. Greedy decoding never touches it, so an
/// unused RNG produces bit-identical (deterministic) output.
pub trait TokenRng {
    /// The next sample in `[0, 1)`.
    fn next_f32(&mut self) -> f32;
}

/// SplitMix64 — the deterministic, seedable PRNG the mlx-gen stacks use. Reproduced verbatim (same
/// constants, same `next_f32` 24-bit mantissa) so seeded runs match the reference engines.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// The golden-ratio increment.
    pub const INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(Self::INCREMENT);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl TokenRng for SplitMix64 {
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Sampling knobs. [`Default`] is greedy (deterministic argmax).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature. `<= 0` ⇒ greedy argmax.
    pub temperature: f32,
    /// Nucleus threshold in `(0, 1]`. `>= 1` disables top-p.
    pub top_p: f32,
    /// Keep only the `top_k` highest-logit tokens before nucleus selection. `0` disables top-k.
    pub top_k: usize,
    /// CTRL/HF repetition penalty. `1.0` disables it.
    pub repetition_penalty: f32,
    /// How many recent history tokens the repetition penalty looks back over.
    pub repetition_context: usize,
    /// Additive penalty subtracted once from every token present anywhere in the history.
    pub presence_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context: 0,
            presence_penalty: 0.0,
        }
    }
}

impl SamplingParams {
    /// True when this configuration is pure greedy with no penalty and (caller-checked) no
    /// constraint mask — eligible for the on-device argmax fast path (and, in the speculative
    /// engine, for device-resident drafts).
    pub fn is_plain_greedy(&self) -> bool {
        self.temperature <= 0.0 && self.repetition_penalty == 1.0 && self.presence_penalty == 0.0
    }
}

/// Sample the next token id from `logits`.
///
/// * `logits` — `[vocab]` or `[1, vocab]` for the current (last) position.
/// * `history` — token ids already in the sequence (prompt + generated), for the repetition penalty.
/// * `allowed` — an optional per-vocab mask (e.g. a JSON-constraint grammar); `false` (or
///   out-of-range) entries are forced to `-inf`.
pub fn sample(
    logits: &Tensor,
    history: &[i32],
    params: &SamplingParams,
    rng: &mut impl TokenRng,
    allowed: Option<&[bool]>,
) -> Result<i32> {
    // Fast path: pure greedy, no penalty, no constraint -> on-device argmax (1-element transfer).
    if params.is_plain_greedy() && allowed.is_none() {
        return argmax_device(logits);
    }

    let v = penalized_logits(logits, history, params, allowed)?;

    // Greedy after mask/penalty have been applied to the host logits.
    if params.temperature <= 0.0 {
        return Ok(argmax_host(&v));
    }

    let weights = nucleus_weights(&v, params);
    let total: f32 = weights.iter().map(|x| x.1).sum();
    if total <= 0.0 || !total.is_finite() {
        return Ok(argmax_host(&v)); // everything masked / -inf; deterministic fallback
    }

    // Categorical inverse-CDF draw over the (unnormalised) weights.
    let mut target = rng.next_f32() * total;
    for (i, w) in &weights {
        target -= *w;
        if target <= 0.0 {
            return Ok(*i as i32);
        }
    }
    Ok(weights.last().map(|x| x.0).unwrap_or(0) as i32)
}

/// The shaped candidate distribution [`sample`] would draw from for a **stochastic**
/// (positive-`temperature`) configuration: `(token_id, unnormalised_weight)` after the constraint
/// mask, repetition penalty, temperature, top-k, and top-p — the distribution speculative decoding
/// (stories 7259 / 7260) feeds to the backend-neutral acceptance sampler. Empty when all masked out.
pub fn shaped_candidates(
    logits: &Tensor,
    history: &[i32],
    params: &SamplingParams,
    allowed: Option<&[bool]>,
) -> Result<Vec<(i32, f32)>> {
    let v = penalized_logits(logits, history, params, allowed)?;
    Ok(nucleus_weights(&v, params)
        .into_iter()
        .map(|(i, w)| (i as i32, w))
        .collect())
}

/// Pull `logits` to host f32 and apply the constraint mask plus repetition and presence penalties
/// (the position-shaping shared by [`sample`] and [`shaped_candidates`]). Repetition is applied
/// first, then presence, matching the MLX backend's combined-penalty order.
fn penalized_logits(
    logits: &Tensor,
    history: &[i32],
    params: &SamplingParams,
    allowed: Option<&[bool]>,
) -> Result<Vec<f32>> {
    super::host_sync::note_host_sync(); // whole-vocab device->host transfer
    let mut v: Vec<f32> = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    penalize_host(&mut v, history, params, allowed);
    Ok(v)
}

/// Pull every row of an all-positions logits tensor `[1, n, vocab]` (or `[n, vocab]`) to host f32
/// in **one** device->host transfer — the speculative engine's verify decision (sc-24130): the
/// `n = K + 1` rows come over together, then each row is shaped on host with
/// [`sample_host`] / [`shaped_candidates_host`], so a verify step costs one sync however wide it
/// is (the per-row [`sample`] would cost `K + 1`).
pub fn logits_rows_host(logits: &Tensor) -> Result<Vec<Vec<f32>>> {
    let (n, vocab) = match logits.dims() {
        [1, n, vocab] | [n, vocab] => (*n, *vocab),
        dims => {
            return Err(crate::error::Error::Msg(format!(
                "logits_rows_host: expected [1, n, vocab] or [n, vocab], got {dims:?}"
            )))
        }
    };
    super::host_sync::note_host_sync(); // one n x vocab transfer
    let flat = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    Ok(flat
        .chunks_exact(vocab)
        .take(n)
        .map(<[f32]>::to_vec)
        .collect())
}

/// On-device argmax of every row of `[1, n, vocab]` (or `[n, vocab]`) logits, brought to host as
/// **one** `n`-element transfer — the greedy verify decision's whole device->host traffic. Row `i`
/// ties break to the lowest index like [`argmax_device`], so a row's result equals the
/// single-row fast path's.
pub fn argmax_rows_device(logits: &Tensor) -> Result<Vec<i32>> {
    let rows = argmax_rows_tensor(logits)?;
    super::host_sync::note_host_sync(); // one n-element transfer
    Ok(rows
        .to_vec1::<u32>()?
        .into_iter()
        .map(|t| t as i32)
        .collect())
}

/// On-device argmax of every row of `[1, n, vocab]` (or `[n, vocab]`) logits as a `[n]` `u32`
/// tensor that **stays on the device** — no host transfer. A greedy proposer keeps its draft
/// tokens here and feeds them back as device ids, so drafting issues no sync; the tokens reach the
/// host later inside the verify decision's single transfer.
pub fn argmax_rows_tensor(logits: &Tensor) -> Result<Tensor> {
    let (n, vocab) = match logits.dims() {
        [1, n, vocab] | [n, vocab] => (*n, *vocab),
        [vocab] => (1, *vocab),
        dims => {
            return Err(crate::error::Error::Msg(format!(
                "argmax_rows_tensor: expected [1, n, vocab], [n, vocab] or [vocab], got {dims:?}"
            )))
        }
    };
    Ok(logits.reshape((n, vocab))?.argmax(1)?)
}

/// [`sample`] over a logits row already on the host (from [`logits_rows_host`]): identical
/// shaping and draw, no device transfer. `row` is consumed (penalties are applied in place).
pub fn sample_host(
    mut row: Vec<f32>,
    history: &[i32],
    params: &SamplingParams,
    rng: &mut impl TokenRng,
    allowed: Option<&[bool]>,
) -> i32 {
    penalize_host(&mut row, history, params, allowed);
    if params.temperature <= 0.0 {
        return argmax_host(&row);
    }
    let weights = nucleus_weights(&row, params);
    let total: f32 = weights.iter().map(|x| x.1).sum();
    if total <= 0.0 || !total.is_finite() {
        return argmax_host(&row);
    }
    let mut target = rng.next_f32() * total;
    for (i, w) in &weights {
        target -= *w;
        if target <= 0.0 {
            return *i as i32;
        }
    }
    weights.last().map(|x| x.0).unwrap_or(0) as i32
}

/// [`shaped_candidates`] over a logits row already on the host (from [`logits_rows_host`]).
pub fn shaped_candidates_host(
    mut row: Vec<f32>,
    history: &[i32],
    params: &SamplingParams,
    allowed: Option<&[bool]>,
) -> Vec<(i32, f32)> {
    penalize_host(&mut row, history, params, allowed);
    nucleus_weights(&row, params)
        .into_iter()
        .map(|(i, w)| (i as i32, w))
        .collect()
}

/// The constraint mask plus repetition and presence penalties over host logits, in place (the
/// position-shaping shared by every sampling entry point). Repetition is applied first, then
/// presence, matching the MLX backend's combined-penalty order.
fn penalize_host(
    v: &mut [f32],
    history: &[i32],
    params: &SamplingParams,
    allowed: Option<&[bool]>,
) {
    // Constraint mask: forbid disallowed ids.
    if let Some(mask) = allowed {
        for (i, val) in v.iter_mut().enumerate() {
            if i >= mask.len() || !mask[i] {
                *val = f32::NEG_INFINITY;
            }
        }
    }

    // Repetition penalty (Keskar et al. 2019 / HF CTRL formulation) over the recent window.
    if params.repetition_penalty != 1.0 && params.repetition_context > 0 {
        let start = history.len().saturating_sub(params.repetition_context);
        for &tok in &history[start..] {
            if let Some(slot) = usize::try_from(tok).ok().and_then(|i| v.get_mut(i)) {
                *slot = if *slot < 0.0 {
                    *slot * params.repetition_penalty
                } else {
                    *slot / params.repetition_penalty
                };
            }
        }
    }

    // Presence penalty is additive and count-independent. It covers the full prompt + generated
    // history, unlike the independently windowed CTRL repetition penalty. Invalid token ids are
    // ignored, and a repeated id is adjusted exactly once.
    if params.presence_penalty != 0.0 {
        let mut seen = HashSet::with_capacity(history.len());
        for &tok in history {
            if seen.insert(tok) {
                if let Some(slot) = usize::try_from(tok).ok().and_then(|i| v.get_mut(i)) {
                    *slot -= params.presence_penalty;
                }
            }
        }
    }
}

/// Temperature + top-k + top-p shaping into `(index, unnormalised_weight)` candidates. Assumes
/// `params.temperature > 0`; returns empty when all logits are masked (`-inf`).
fn nucleus_weights(v: &[f32], params: &SamplingParams) -> Vec<(usize, f32)> {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Vec::new();
    }
    let inv_t = 1.0 / params.temperature;
    let mut weights: Vec<(usize, f32)> = v
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, ((x - max) * inv_t).exp()))
        .collect();

    // top-k: keep the k highest-weight tokens (descending weight, ties to lower index).
    if params.top_k > 0 && params.top_k < weights.len() {
        weights.select_nth_unstable_by(params.top_k - 1, |a, b| weight_desc_index_asc(*a, *b));
        weights.truncate(params.top_k);
    }

    // top-p nucleus.
    if params.top_p < 1.0 {
        weights = nucleus_select(&weights, params.top_p);
    }
    weights
}

/// On-device argmax of a `[vocab]` / `[1, vocab]` logits row. Greedy fast path — avoids pulling the
/// full vocabulary to the host. Ties break to the lowest index (matching the host scan), so greedy
/// decoding is bit-identical whichever path is taken.
pub fn argmax_device(logits: &Tensor) -> Result<i32> {
    let flat = logits.flatten_all()?;
    let idx = flat.argmax(0)?;
    super::host_sync::note_host_sync(); // 1-element device->host transfer
    Ok(idx.to_scalar::<u32>()? as i32)
}

/// Host-side argmax over an f32 slice; first maximum wins (ties → lowest index).
pub fn argmax_host(v: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best as i32
}

/// Descending by weight, ascending by index on ties — a total order so selection is deterministic.
fn weight_desc_index_asc(a: (usize, f32), b: (usize, f32)) -> Ordering {
    b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
}

/// Heap-ordered nucleus: pop highest-weight tokens until the cumulative weight reaches
/// `top_p * total`, always keeping at least one. Ties break to lower index.
fn nucleus_select(weights: &[(usize, f32)], top_p: f32) -> Vec<(usize, f32)> {
    let total: f32 = weights.iter().map(|x| x.1).sum();
    let threshold = top_p.max(0.0) * total;
    let mut heap: BinaryHeap<ByWeight> = weights.iter().map(|&(i, w)| ByWeight(i, w)).collect();
    let mut kept = Vec::new();
    let mut cum = 0.0f32;
    while let Some(ByWeight(i, w)) = heap.pop() {
        kept.push((i, w));
        cum += w;
        if cum >= threshold {
            break;
        }
    }
    kept
}

/// Max-heap ordering by weight; ties resolve so the *lower* index is popped first (deterministic).
struct ByWeight(usize, f32);

impl PartialEq for ByWeight {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for ByWeight {}
impl PartialOrd for ByWeight {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ByWeight {
    fn cmp(&self, other: &Self) -> Ordering {
        // Larger weight is "greater" (popped first). On equal weight, the lower index is "greater"
        // so it pops first — reverse the index comparison.
        self.1
            .total_cmp(&other.1)
            .then_with(|| other.0.cmp(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn logits(v: &[f32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), (1, v.len()), &Device::Cpu).unwrap()
    }

    #[test]
    fn splitmix64_matches_known_sequence() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220A8397B1DCDAF);
        assert_eq!(rng.next_u64(), 0x6E789E6AA1B965F4);
    }

    #[test]
    fn next_f32_is_in_unit_interval() {
        let mut rng = SplitMix64::new(42);
        for _ in 0..1000 {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x), "{x}");
        }
    }

    #[test]
    fn greedy_picks_argmax() {
        let mut rng = SplitMix64::new(0);
        let l = logits(&[0.1, 5.0, 0.2, -1.0]);
        let t = sample(&l, &[], &SamplingParams::default(), &mut rng, None).unwrap();
        assert_eq!(t, 1);
    }

    #[test]
    fn argmax_device_matches_host() {
        let l = logits(&[0.1, 5.0, 0.2, 9.9, 3.0]);
        assert_eq!(argmax_device(&l).unwrap(), 3);
        assert_eq!(argmax_host(&[0.1, 5.0, 0.2, 9.9, 3.0]), 3);
    }

    #[test]
    fn host_row_entry_points_match_the_tensor_ones_with_one_transfer() {
        let rows = Tensor::from_vec(
            vec![0.1f32, 5.0, 0.2, 9.9, 3.0, 2.0, 1.0, 4.0, 0.5, 4.0],
            (1, 2, 5),
            &Device::Cpu,
        )
        .unwrap();
        let before = crate::primitives::host_sync_count();
        let host = logits_rows_host(&rows).unwrap();
        assert_eq!(crate::primitives::host_sync_count() - before, 1);
        assert_eq!(host.len(), 2);
        assert_eq!(host[0], vec![0.1, 5.0, 0.2, 9.9, 3.0]);
        let before = crate::primitives::host_sync_count();
        assert_eq!(argmax_rows_device(&rows).unwrap(), vec![3, 2]); // ties -> lowest index
        assert_eq!(crate::primitives::host_sync_count() - before, 1);
        let before = crate::primitives::host_sync_count();
        let kept = argmax_rows_tensor(&rows).unwrap();
        assert_eq!(crate::primitives::host_sync_count() - before, 0);
        assert_eq!(kept.to_vec1::<u32>().unwrap(), vec![3, 2]);

        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 3,
            repetition_penalty: 1.3,
            repetition_context: 4,
            presence_penalty: 0.2,
        };
        let row = logits(&[1.0, 2.0, 3.0, 0.5, -1.0, 4.0]);
        let history = [1, 5, 5];
        let mask = [true, true, true, true, false, true];
        let mut a = SplitMix64::new(9);
        let mut b = SplitMix64::new(9);
        for _ in 0..16 {
            let via_tensor = sample(&row, &history, &params, &mut a, Some(&mask)).unwrap();
            let host_row = logits_rows_host(&row).unwrap().remove(0);
            let via_host = sample_host(host_row, &history, &params, &mut b, Some(&mask));
            assert_eq!(via_tensor, via_host);
        }
        assert_eq!(
            shaped_candidates(&row, &history, &params, Some(&mask)).unwrap(),
            shaped_candidates_host(
                logits_rows_host(&row).unwrap().remove(0),
                &history,
                &params,
                Some(&mask)
            )
        );
        // Greedy on host with a penalty is the penalized argmax, like the tensor path.
        let greedy = SamplingParams {
            repetition_penalty: 5.0,
            repetition_context: 8,
            ..Default::default()
        };
        let l = logits(&[2.0, 1.5, 0.5]);
        assert_eq!(
            sample_host(
                logits_rows_host(&l).unwrap().remove(0),
                &[0, 0, 0],
                &greedy,
                &mut SplitMix64::new(0),
                None
            ),
            1
        );
        let bad = Tensor::zeros((2, 2, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(logits_rows_host(&bad).is_err());
    }

    #[test]
    fn argmax_ties_break_to_lowest_index() {
        assert_eq!(argmax_host(&[1.0, 5.0, 5.0, 2.0]), 1);
    }

    #[test]
    fn sampling_is_deterministic_for_fixed_seed() {
        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            ..Default::default()
        };
        let l = logits(&[1.0, 2.0, 3.0, 0.5, -1.0, 4.0]);
        let mut a = SplitMix64::new(123);
        let mut b = SplitMix64::new(123);
        let ta: Vec<i32> = (0..20)
            .map(|_| sample(&l, &[], &params, &mut a, None).unwrap())
            .collect();
        let tb: Vec<i32> = (0..20)
            .map(|_| sample(&l, &[], &params, &mut b, None).unwrap())
            .collect();
        assert_eq!(ta, tb);
    }

    #[test]
    fn different_seeds_can_diverge() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            ..Default::default()
        };
        let l = logits(&[1.0, 1.1, 0.9, 1.05, 0.95, 1.2]);
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(0x9E37_79B9);
        let ta: Vec<i32> = (0..40)
            .map(|_| sample(&l, &[], &params, &mut a, None).unwrap())
            .collect();
        let tb: Vec<i32> = (0..40)
            .map(|_| sample(&l, &[], &params, &mut b, None).unwrap())
            .collect();
        assert_ne!(ta, tb);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            ..Default::default()
        };
        let l = logits(&[0.0, 1.0, 5.0, 2.0]);
        let mut rng = SplitMix64::new(3);
        for _ in 0..30 {
            assert_eq!(sample(&l, &[], &params, &mut rng, None).unwrap(), 2);
        }
    }

    #[test]
    fn constraint_mask_forces_allowed_token() {
        // argmax is index 2, but the mask only allows index 0.
        let params = SamplingParams::default();
        let l = logits(&[0.1, 0.2, 9.0, 0.3]);
        let mask = [true, false, false, false];
        let mut rng = SplitMix64::new(0);
        assert_eq!(sample(&l, &[], &params, &mut rng, Some(&mask)).unwrap(), 0);
    }

    #[test]
    fn repetition_penalty_suppresses_recent_token() {
        // Token 0 has the top logit but is heavily penalised by recent history -> token 1 wins.
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 5.0,
            repetition_context: 8,
            ..Default::default()
        };
        let l = logits(&[2.0, 1.5, 0.5]);
        let history = [0, 0, 0];
        let mut rng = SplitMix64::new(0);
        let t = sample(&l, &history, &params, &mut rng, Some(&[true, true, true])).unwrap();
        assert_eq!(t, 1);
    }

    #[test]
    fn presence_penalty_is_additive_once_per_seen_token() {
        let params = SamplingParams {
            presence_penalty: 0.75,
            ..Default::default()
        };
        let l = logits(&[2.0, 1.5, 0.5]);
        let adjusted = penalized_logits(&l, &[0, 0, 0, 2, -1, 99], &params, None).unwrap();
        assert_eq!(adjusted, vec![1.25, 1.5, -0.25]);

        // Token 0 began as the argmax. A single presence subtraction makes token 1 win even
        // though token 0 appears three times; its count does not multiply the penalty.
        let mut rng = SplitMix64::new(0);
        assert_eq!(sample(&l, &[0, 0, 0], &params, &mut rng, None).unwrap(), 1);
    }

    #[test]
    fn presence_and_repetition_use_backend_parity_order() {
        let params = SamplingParams {
            repetition_penalty: 2.0,
            repetition_context: 8,
            presence_penalty: 0.5,
            ..Default::default()
        };
        let l = logits(&[4.0, -2.0]);
        let adjusted = penalized_logits(&l, &[0, 1], &params, None).unwrap();
        // CTRL transform first: [4 / 2, -2 * 2], then additive presence: [-.5, -.5].
        assert_eq!(adjusted, vec![1.5, -4.5]);
    }
}
