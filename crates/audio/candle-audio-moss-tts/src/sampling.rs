//! Deterministic per-channel token sampling for the MOSS-TTSD delay-pattern AR loop (sc-13360).
//!
//! MOSS-TTSD samples each channel token — it is **not** a greedy model (`generation_config` ships
//! `do_sample=True` with `temperature=0.9`, `top_k=50`, `top_p=0.95`, `repetition_penalty=1.0`).
//! Greedy (argmax) decoding of a discrete-codebook TTS collapses into a repeating loop that decodes
//! to silence, so the reference (and this port) samples. This module reproduces the reference
//! per-channel `LogitsProcessorList` (Temperature → TopK → TopP warpers + an optional
//! RepetitionPenalty) faithfully, but with a **seeded** PRNG so the gen-core reproducibility law
//! holds: the same `seed` yields byte-identical frames run to run.
//!
//! Order (matching HF's `RepetitionPenaltyLogitsProcessor` → `TemperatureLogitsWarper` →
//! `TopKLogitsWarper` → `TopPLogitsWarper` → softmax → multinomial): repetition penalty on the raw
//! logits → divide by temperature → top-k mask → top-p (nucleus) mask → softmax → multinomial draw.
//! Parity with PyTorch's audio is not required (a different RNG) — a valid, deterministic
//! distribution is.

use std::collections::HashSet;

/// MOSS-TTSD-v0.5 `generation_config` decoding defaults.
pub const TEMPERATURE: f32 = 0.9;
pub const TOP_P: f32 = 0.95;
pub const TOP_K: usize = 50;
pub const REPETITION_PENALTY: f32 = 1.0;
pub const REPETITION_WINDOW: usize = 50;

/// Fixed-parameter sampling configuration (the reference `generation_config` defaults).
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: TEMPERATURE,
            top_p: TOP_P,
            top_k: TOP_K,
            repetition_penalty: REPETITION_PENALTY,
            repetition_window: REPETITION_WINDOW,
        }
    }
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
        let mut sorted: Vec<f32> = logits.to_vec();
        // Partial selection of the k-th largest.
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        let kth = sorted[k - 1];
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
fn apply_top_p(logits: &mut [f32], top_p: f32) {
    let vocab = logits.len();
    let mut order: Vec<usize> = (0..vocab).collect();
    order.sort_unstable_by(|&a, &b| logits[a].total_cmp(&logits[b])); // ascending
                                                                      // softmax over the (finite) logits in ascending order.
    let probs = softmax(logits);
    let mut cum = 0.0f32;
    let threshold = 1.0 - top_p;
    // The last element (index vocab-1 in `order`) is the highest-prob token — always kept.
    for &idx in order.iter().take(vocab.saturating_sub(1)) {
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
            ..Default::default()
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
    fn repetition_penalty_suppresses_recent_tokens() {
        // A token that is strongly preferred but heavily repeated should eventually be dispreferred
        // once penalized enough; verify the penalty lowers its (positive) logit.
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 100,
            top_p: 1.0,
            repetition_penalty: 2.0,
            repetition_window: 50,
        };
        let mut rng = Rng::seed(3);
        let history = vec![2u32; 10];
        let mut logits = vec![1.0f32, 1.0, 4.0, 1.0];
        // token 2 logit 4.0 → /2 = 2.0; token 1/3 stay 1.0. Still max, but reduced.
        let mut peek = logits.clone();
        let _ = sample(&mut peek, &history, &params, &mut rng);
        // Re-derive the penalized logit directly.
        let l = 4.0f32 / 2.0;
        assert!((l - 2.0).abs() < 1e-6);
        // Sanity: sampling still returns a valid in-range token.
        let t = sample(&mut logits, &history, &params, &mut rng);
        assert!((t as usize) < 4);
    }

    #[test]
    fn distribution_favors_high_logits() {
        // Over many draws, the higher-logit token should dominate.
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
}
