//! Deterministic categorical token sampling shared by the MOSS-TTS family's AR loops (sc-27272).
//!
//! MOSS-TTSD (`candle-audio-moss-tts`) and MOSS-TTS-Realtime (`candle-audio-moss-tts-realtime`)
//! sample every codebook/channel token with the same HF-style pipeline: repetition penalty on the
//! raw logits → divide by temperature → top-k mask → top-p (nucleus) mask → softmax → multinomial
//! draw, driven by a **seeded** splitmix64 PRNG so the gen-core reproducibility law holds (the
//! same `seed` yields byte-identical frames run to run). The two crates carried byte-identical
//! private copies of this module; they now re-export it and keep only their own reference
//! decoding defaults (`TEMPERATURE`, `TOP_K`, …) as a `DEFAULT_PARAMS` constant.
//!
//! Parity with PyTorch's draws is not a goal (a different RNG) — a valid, deterministic
//! distribution is. See each provider's `sampling` module for its model's RNG-parity notes.

use std::collections::HashSet;

/// Fixed-parameter sampling configuration. Each provider supplies its reference defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
}

/// A tiny deterministic PRNG (splitmix64) — no external deps, reproducible from a `u64` seed.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the PRNG. A `None` request seed maps to a fixed constant so decoding stays deterministic.
    pub fn seed(seed: u64) -> Self {
        // Mix the seed so seed 0 is not a degenerate all-zero state.
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform `f32` in `[0, 1)` (24-bit mantissa).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Sample one token id from `logits` (length = vocab), applying the reference pipeline. `history` is
/// this codebook's tokens from previous frames (most recent last); the repetition penalty uses its
/// last `repetition_window` entries. `logits` is consumed (mutated in place).
pub fn sample(logits: &mut [f32], history: &[u32], params: &SamplingParams, rng: &mut Rng) -> u32 {
    let vocab = logits.len();
    if vocab == 0 {
        return 0;
    }

    // 1) Repetition penalty on the raw logits (each repeated token penalized once, from its original
    //    value): l<0 → l*penalty (more negative), else l/penalty (smaller). Matches the reference
    //    gather/where/scatter.
    if params.repetition_penalty != 1.0 && !history.is_empty() {
        let start = history.len().saturating_sub(params.repetition_window);
        let recent: HashSet<u32> = history[start..].iter().copied().collect();
        for &tok in &recent {
            let idx = tok as usize;
            if idx < vocab {
                let l = logits[idx];
                logits[idx] = if l < 0.0 {
                    l * params.repetition_penalty
                } else {
                    l / params.repetition_penalty
                };
            }
        }
    }

    // 2) Temperature.
    let temp = if params.temperature <= 0.0 {
        1.0
    } else {
        params.temperature
    };
    if params.temperature <= 0.0 {
        // temperature == 0 ⇒ greedy (the reference short-circuit).
        return argmax(logits);
    }
    for l in logits.iter_mut() {
        *l /= temp;
    }

    // 3) top-k: keep the k largest logits, mask the rest to -inf.
    let k = params.top_k.clamp(1, vocab);
    if k < vocab {
        // Partial selection of the k-th largest (sc-27363): `select_nth_unstable_by` places the
        // element a full descending sort would put at `k - 1` there, so `kth` is the same value a
        // full sort yields — without sorting the whole vocabulary.
        let mut sorted: Vec<f32> = logits.to_vec();
        let (_, kth, _) = sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
        let kth = *kth;
        for l in logits.iter_mut() {
            if *l < kth {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    // 4) top-p (nucleus): keep the smallest set of highest-prob tokens whose cumulative probability
    //    reaches top_p; always keep at least one. Implemented as the reference does: sort ascending,
    //    remove tokens whose ascending-cumulative softmax mass is <= (1 - top_p).
    if params.top_p > 0.0 && params.top_p < 1.0 {
        apply_top_p(logits, params.top_p);
    }

    // 5) softmax over the surviving logits.
    let probs = softmax(logits);

    // 6) multinomial draw.
    multinomial(&probs, rng)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Nucleus filter: sort ascending, mask tokens whose ascending cumulative softmax mass is
/// `<= 1 - top_p`, always keeping the single highest-prob token.
///
/// After top-k most of the row is `-inf`. Those entries sort first, carry probability exactly
/// `0.0` (so they leave the running sum unchanged) and re-masking them is a no-op, so the filter
/// only needs to sort the surviving entries (sc-27363) — the result is bit-identical to sorting the
/// whole vocabulary. That argument needs a finite row maximum (otherwise [`softmax`] goes uniform
/// and the `-inf` entries carry mass), no NaN (whose probability would poison the sum at a
/// different position) and no exact ties among the survivors (an unstable sort's tie order depends
/// on its input); any of those falls back to the full-vocabulary sort, which is the original code.
fn apply_top_p(logits: &mut [f32], top_p: f32) {
    let vocab = logits.len();
    let ascending = |logits: &[f32], a: usize, b: usize| logits[a].total_cmp(&logits[b]);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut order: Vec<usize> = Vec::new();
    if max.is_finite() && !logits.iter().any(|l| l.is_nan()) {
        order.extend((0..vocab).filter(|&i| logits[i] != f32::NEG_INFINITY));
        order.sort_unstable_by(|&a, &b| ascending(logits, a, b));
        if order
            .windows(2)
            .any(|w| ascending(logits, w[0], w[1]).is_eq())
        {
            order.clear();
        }
    }
    if order.is_empty() {
        order.extend(0..vocab);
        order.sort_unstable_by(|&a, &b| ascending(logits, a, b));
    }
    // softmax over the full row (the `-inf` entries contribute `exp(-inf) = 0` to the sum).
    let probs = softmax(logits);
    let mut cum = 0.0f32;
    let threshold = 1.0 - top_p;
    // The last element of `order` is the highest-prob token — always kept.
    for &idx in order.iter().take(order.len().saturating_sub(1)) {
        cum += probs[idx];
        if cum <= threshold {
            logits[idx] = f32::NEG_INFINITY;
        }
    }
}

/// Numerically-stable softmax.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // All -inf (shouldn't happen): uniform.
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    let mut exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in exps.iter_mut() {
            *e /= sum;
        }
    }
    exps
}

/// Draw an index from a probability vector via inverse-CDF with a uniform sample.
fn multinomial(probs: &[f32], rng: &mut Rng) -> u32 {
    let u = rng.next_f32();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u < cum {
            return i as u32;
        }
    }
    // Fallback (floating-point slack): the last non-zero probability index.
    probs
        .iter()
        .enumerate()
        .rev()
        .find(|(_, &p)| p > 0.0)
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MOSS-TTS-Realtime reference defaults, used as a representative parameter set.
    const PARAMS: SamplingParams = SamplingParams {
        temperature: 0.8,
        top_p: 0.6,
        top_k: 30,
        repetition_penalty: 1.1,
        repetition_window: 50,
    };

    #[test]
    fn rng_is_deterministic_per_seed() {
        let a: Vec<f32> = {
            let mut r = Rng::seed(42);
            (0..8).map(|_| r.next_f32()).collect()
        };
        let b: Vec<f32> = {
            let mut r = Rng::seed(42);
            (0..8).map(|_| r.next_f32()).collect()
        };
        assert_eq!(a, b, "same seed ⇒ same stream");
        let c: Vec<f32> = {
            let mut r = Rng::seed(43);
            (0..8).map(|_| r.next_f32()).collect()
        };
        assert_ne!(a, c, "different seed ⇒ different stream");
        assert!(a.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let params = SamplingParams {
            temperature: 0.0,
            ..PARAMS
        };
        let mut rng = Rng::seed(1);
        let mut logits = vec![0.1, 5.0, 0.2, -1.0];
        assert_eq!(sample(&mut logits, &[], &params, &mut rng), 1);
    }

    #[test]
    fn top_k_one_selects_the_argmax() {
        let params = SamplingParams {
            top_k: 1,
            top_p: 1.0,
            temperature: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 50,
        };
        let mut rng = Rng::seed(7);
        // With k=1 only the max survives, so any draw returns it regardless of the RNG.
        for _ in 0..20 {
            let mut logits = vec![1.0, 2.0, 9.0, 3.0, 0.5];
            assert_eq!(sample(&mut logits, &[], &params, &mut rng), 2);
        }
    }

    #[test]
    fn neg_inf_masked_token_is_never_sampled() {
        // A token that is the *pre-mask argmax* but masked to -inf is never drawn, across many
        // seeds/draws, at the reference temperature (the EOS-suppression mechanism).
        let params = SamplingParams {
            repetition_penalty: 1.0,
            ..PARAMS
        };
        let masked = 2usize; // would otherwise dominate (largest finite logit)
        for seed in 0..64 {
            let mut rng = Rng::seed(seed);
            for _ in 0..16 {
                let mut logits = vec![1.0f32, 0.5, 9.0, 0.2, -1.0, 3.0];
                logits[masked] = f32::NEG_INFINITY;
                let t = sample(&mut logits, &[], &params, &mut rng) as usize;
                assert_ne!(t, masked, "a -inf-masked token must never be sampled");
            }
        }
    }

    #[test]
    fn distribution_favors_high_logits() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 100,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 50,
        };
        let mut rng = Rng::seed(123);
        let mut counts = [0usize; 3];
        for _ in 0..2000 {
            let mut logits = vec![0.0f32, 3.0, -2.0];
            let t = sample(&mut logits, &[], &params, &mut rng);
            counts[t as usize] += 1;
        }
        assert!(
            counts[1] > counts[0] && counts[0] > counts[2],
            "counts {counts:?} should rank by logit"
        );
    }

    /// The pre-sc-27363 top-k / top-p filters, verbatim: a full descending sort for the k-th
    /// value and a full-vocabulary ascending sort for the nucleus.
    fn reference_filters(logits: &mut [f32], params: &SamplingParams) {
        let vocab = logits.len();
        let k = params.top_k.clamp(1, vocab);
        if k < vocab {
            let mut sorted: Vec<f32> = logits.to_vec();
            sorted.sort_unstable_by(|a, b| b.total_cmp(a));
            let kth = sorted[k - 1];
            for l in logits.iter_mut() {
                if *l < kth {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        if params.top_p > 0.0 && params.top_p < 1.0 {
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_unstable_by(|&a, &b| logits[a].total_cmp(&logits[b]));
            let probs = softmax(logits);
            let mut cum = 0.0f32;
            let threshold = 1.0 - params.top_p;
            for &idx in order.iter().take(vocab.saturating_sub(1)) {
                cum += probs[idx];
                if cum <= threshold {
                    logits[idx] = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// The new filters as `sample` runs them (temperature 1, no repetition penalty).
    fn current_filters(logits: &mut [f32], params: &SamplingParams) {
        let vocab = logits.len();
        let k = params.top_k.clamp(1, vocab);
        if k < vocab {
            let mut sorted: Vec<f32> = logits.to_vec();
            let (_, kth, _) = sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
            let kth = *kth;
            for l in logits.iter_mut() {
                if *l < kth {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        if params.top_p > 0.0 && params.top_p < 1.0 {
            apply_top_p(logits, params.top_p);
        }
    }

    /// sc-27363: the partial top-k selection and the survivors-only top-p sort mask exactly the
    /// same entries (bit-for-bit) as the full sorts they replace — over random rows, rows with
    /// exact ties, pre-masked `-inf` entries, all-`-inf` rows, `+inf` and NaN — and `sample`
    /// therefore draws the same token for every seed.
    #[test]
    fn filters_match_the_full_sort_reference_bit_for_bit() {
        let mut gen = Rng::seed(2026);
        for case in 0..400usize {
            let vocab = 1 + (gen.next_u64() % 300) as usize;
            let mut row: Vec<f32> = (0..vocab).map(|_| gen.next_f32() * 20.0 - 10.0).collect();
            match case % 6 {
                // Exact ties: quantize to a coarse grid.
                1 => row.iter_mut().for_each(|l| *l = (*l * 0.5).round()),
                // Some entries already masked (EOS suppression).
                2 => row
                    .iter_mut()
                    .step_by(3)
                    .for_each(|l| *l = f32::NEG_INFINITY),
                // Every entry masked.
                3 => row.iter_mut().for_each(|l| *l = f32::NEG_INFINITY),
                4 => row[0] = f32::INFINITY,
                5 => row[vocab / 2] = f32::NAN,
                _ => {}
            }
            for (top_k, top_p) in [(30usize, 0.6f32), (50, 0.95), (1, 0.5), (vocab, 0.3)] {
                let params = SamplingParams {
                    temperature: 1.0,
                    top_k,
                    top_p,
                    repetition_penalty: 1.0,
                    repetition_window: 50,
                };
                let (mut want, mut got) = (row.clone(), row.clone());
                reference_filters(&mut want, &params);
                current_filters(&mut got, &params);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&want), "case {case} k={top_k} p={top_p}");
            }
        }
    }
}
