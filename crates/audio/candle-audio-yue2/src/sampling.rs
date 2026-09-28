//! The YuE2 token sampler: token-range masks, stop ids, the windowed repetition penalty,
//! temperature / top-k / top-p, classifier-free guidance and the categorical draw (sc-22991).
//!
//! Ported from the pinned upstream `src/yue2/sampling.py` (`window_penalty`, `distribution`, the
//! CFG line of `generate_tokens`) and the protocol constants of `src/yue2/protocol.py`. The order
//! and arithmetic are the released ones, which differ from a generic LLM sampler (and from
//! `candle_llm`'s, which is therefore not reused here):
//!
//! 1. the phase's **allow mask** is added (`-inf` outside it): the ABC phase allows the ordinary
//!    text ids `[0, EOD)` plus `ABC_END`; the semantic phase allows the 32 768 codec ids plus
//!    `MUSIC_END`;
//! 2. before `min_tokens` outputs the phase's end id is barred;
//! 3. the **windowed repetition penalty** divides a positive score (multiplies a negative one) by
//!    `penalty^count`, where `count` is how often the id occurs among the last `penalty_window`
//!    **generated** tokens — compounded per occurrence, never over the prefix;
//! 4. temperature `0` stops here and takes the argmax (first maximum); otherwise the scores are
//!    divided by the temperature, everything below the `top_k`-th score is removed (ties at the
//!    threshold survive), and when `top_p < 1` the sorted tail whose preceding mass already exceeds
//!    `top_p` is removed — always keeping the top **one** entry, or the top **three** under the
//!    historical `cot = off` arithmetic.
//!
//! # Precision
//!
//! Upstream feeds the sampler F32 scores (`logits.float()`), except for the semantic stage of
//! `cot = off`, which keeps the model dtype throughout (`legacy_off`). [`Arith`] reproduces that:
//! [`Arith::Bf16`] rounds after every elementwise operation exactly where PyTorch's BF16 CPU
//! kernels do (measured against torch 2.10, see `tests/fixtures/README.md`): a Python-float
//! multiplier/divisor stays F32 inside the op, a Python-float comparison threshold and the penalty
//! base are rounded to BF16 first, softmax is computed in F32 and rounded, and the cumulative sum
//! accumulates in F32 and rounds each output. [`Arith::F32`] accumulates the cumulative sum in F64,
//! as PyTorch does on the CPU. CFG mixes the branches in the **model** dtype before the sampler
//! upcasts (`uncond + scale · (cond − uncond)`).
//!
//! # The draw
//!
//! A stochastic step draws one uniform `u ∈ [0, 1)` from a [`TokenRng`] and returns the first id
//! (in vocabulary order) whose cumulative probability exceeds `u`. The distribution is the
//! released one; the random stream is not PyTorch's (a seed does not reproduce a torch run, and is
//! not a cross-platform bit-exact guarantee — epic E9). Parity tests inject the draws instead.

use candle_audio::gen_core;
pub use candle_llm::primitives::{SplitMix64, TokenRng};

pub use crate::protocol::Sampling;
use crate::protocol::{ABC_END, CODEC_OFFSET, CODEC_SIZE, EOD, MUSIC_END};

/// The largest protocol id a sampler mask names (the last codec id).
pub(crate) const VOCAB_MAX_PROTOCOL_ID: u32 = CODEC_OFFSET + CODEC_SIZE - 1;

/// The two autoregressive stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The symbolic score (ABC text).
    Abc,
    /// The semantic codec tokens.
    Semantic,
}

impl Phase {
    /// The id that ends this phase.
    pub fn end_token(self) -> u32 {
        match self {
            Phase::Abc => ABC_END,
            Phase::Semantic => MUSIC_END,
        }
    }

    /// Whether this phase may emit `id` (its allow mask, end id included).
    pub fn allows(self, id: u32) -> bool {
        match self {
            Phase::Abc => id < EOD || id == ABC_END,
            Phase::Semantic => {
                (CODEC_OFFSET..CODEC_OFFSET + CODEC_SIZE).contains(&id) || id == MUSIC_END
            }
        }
    }
}

/// The arithmetic a score row is computed in (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arith {
    /// IEEE single precision (cumulative sums in double).
    F32,
    /// PyTorch's BF16 CPU arithmetic: F32 inside each op, rounded to BF16 after it.
    Bf16,
}

impl Arith {
    /// Round an F32 result to this arithmetic's storage type (round-to-nearest-even for BF16).
    pub fn round(self, x: f32) -> f32 {
        match self {
            Arith::F32 => x,
            Arith::Bf16 => round_bf16(x),
        }
    }
}

/// Round to the nearest BF16 (ties to even), returned as the F32 holding it. NaN stays NaN.
pub fn round_bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xFFFF_0000)
}

/// Classifier-free guidance, upstream's `unconditional + cfg_scale * (conditional -
/// unconditional)` in the model dtype's arithmetic (`scale` is a Python float, so it stays F32
/// inside the multiply).
pub fn cfg_mix(conditional: &[f32], unconditional: &[f32], scale: f64, arith: Arith) -> Vec<f32> {
    let s = scale as f32;
    conditional
        .iter()
        .zip(unconditional)
        .map(|(&c, &u)| arith.round(u + arith.round(s * arith.round(c - u))))
        .collect()
}

/// Upstream `distribution`: shape one logits row into the scores the draw samples from (see the
/// module docs for the order). `history` is the tokens generated so far in this phase (never the
/// prefix); `step` is the 0-based output index; `legacy_off` selects the historical `cot = off`
/// top-p rule (keep three). `logits` must already be representable in `arith`.
pub fn distribution(
    logits: &[f32],
    sampling: &Sampling,
    history: &[u32],
    step: usize,
    phase: Phase,
    arith: Arith,
    legacy_off: bool,
) -> Vec<f32> {
    let mut scores: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if phase.allows(i as u32) {
                x + 0.0
            } else {
                x + f32::NEG_INFINITY
            }
        })
        .collect();
    let end = phase.end_token() as usize;
    if step < sampling.min_tokens() as usize && end < scores.len() {
        scores[end] = f32::NEG_INFINITY;
    }
    let recent = &history[history
        .len()
        .saturating_sub(sampling.penalty_window() as usize)..];
    window_penalty(&mut scores, recent, sampling.repetition_penalty(), arith);
    if sampling.temperature() == 0.0 {
        return scores;
    }
    if sampling.temperature() != 1.0 {
        let t = sampling.temperature() as f32;
        for s in &mut scores {
            *s = arith.round(*s / t);
        }
    }
    top_k(&mut scores, sampling.top_k() as usize);
    if sampling.top_p() < 1.0 {
        top_p(
            &mut scores,
            sampling.top_p(),
            arith,
            if legacy_off { 3 } else { 1 },
        );
    }
    scores
}

/// Upstream `window_penalty`: `alpha = penalty ** count` per id over `recent`, then
/// `where(s < 0, s * alpha, s / alpha)`. A penalty of exactly `1` or an empty window is a no-op.
fn window_penalty(scores: &mut [f32], recent: &[u32], penalty: f64, arith: Arith) {
    if penalty == 1.0 || recent.is_empty() {
        return;
    }
    let mut counts = std::collections::BTreeMap::<usize, u32>::new();
    for &id in recent {
        *counts.entry(id as usize).or_default() += 1;
    }
    // The Python-float base meets a model-dtype exponent tensor, so it is cast to that dtype.
    let base = arith.round(penalty as f32);
    for (id, count) in counts {
        let Some(s) = scores.get_mut(id) else {
            continue;
        };
        let alpha = arith.round(base.powf(count as f32));
        *s = if *s < 0.0 {
            arith.round(*s * alpha)
        } else {
            arith.round(*s / alpha)
        };
    }
}

/// Remove every score below the `k`-th highest (duplicates count toward `k`; ties at the threshold
/// survive).
fn top_k(scores: &mut [f32], k: usize) {
    let k = k.min(scores.len());
    if k == 0 {
        return;
    }
    let mut sorted = scores.to_vec();
    let (_, kth, _) = sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    let threshold = *kth;
    for s in scores.iter_mut() {
        if *s < threshold {
            *s = f32::NEG_INFINITY;
        }
    }
}

/// Upstream's nucleus rule over the descending scores: remove entry `j` when
/// `cumsum(p)[j] − p[j] > top_p`, never the first `keep` entries.
fn top_p(scores: &mut [f32], top_p: f64, arith: Arith, keep: usize) {
    let mut order: Vec<usize> = (0..scores.len())
        .filter(|&i| scores[i] > f32::NEG_INFINITY)
        .collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let values: Vec<f32> = order.iter().map(|&i| scores[i]).collect();
    let probs = softmax_values(&values, arith);
    let threshold = arith.round(top_p as f32);
    let mut acc64 = 0.0f64;
    let mut acc32 = 0.0f32;
    for (j, (&i, &p)) in order.iter().zip(&probs).enumerate() {
        let cum = match arith {
            Arith::F32 => {
                acc64 += p as f64;
                acc64 as f32
            }
            Arith::Bf16 => {
                acc32 += p;
                round_bf16(acc32)
            }
        };
        if j >= keep && arith.round(cum - p) > threshold {
            scores[i] = f32::NEG_INFINITY;
        }
    }
}

/// Softmax of `values` (all finite, or `-inf` for a zero weight): `exp(x − max) · (1 / Σ)` in F32,
/// rounded to `arith`.
fn softmax_values(values: &[f32], arith: Arith) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|&v| (v - max).exp()).collect();
    let inv = 1.0 / exps.iter().sum::<f32>();
    exps.into_iter().map(|e| arith.round(e * inv)).collect()
}

/// The released final step: `scores.softmax(-1)` in `arith`.
pub fn probabilities(scores: &[f32], arith: Arith) -> Vec<f32> {
    softmax_values(scores, arith)
}

/// The first index of the maximum score (PyTorch `argmax` tie rule).
pub fn argmax(scores: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &s) in scores.iter().enumerate() {
        if s > scores[best] {
            best = i;
        }
    }
    best as u32
}

/// Inverse-CDF draw: the first id (in vocabulary order) with positive probability whose
/// cumulative probability (F64) exceeds `u · Σp`. `None` when no id has positive probability.
pub fn draw(probabilities: &[f32], u: f32) -> Option<u32> {
    let total: f64 = probabilities.iter().map(|&p| p as f64).sum();
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    let target = u as f64 * total;
    let mut acc = 0.0f64;
    let mut last = None;
    for (i, &p) in probabilities.iter().enumerate() {
        if p > 0.0 {
            acc += p as f64;
            last = Some(i as u32);
            if acc > target {
                return last;
            }
        }
    }
    last
}

/// One step's token from shaped `scores`: the argmax at temperature `0` (no draw consumed),
/// otherwise one draw from `rng` over `probabilities(scores)`.
pub fn next_token(
    scores: &[f32],
    sampling: &Sampling,
    arith: Arith,
    rng: &mut dyn TokenRng,
) -> gen_core::Result<u32> {
    if sampling.temperature() == 0.0 {
        return Ok(argmax(scores));
    }
    let probs = probabilities(scores, arith);
    draw(&probs, rng.next_f32()).ok_or_else(|| {
        gen_core::Error::Msg("YuE2 sampler: the shaped distribution has no mass".into())
    })
}

/// A validated [`Sampling`] from all seven values (test construction through the protocol's own
/// validation).
#[cfg(test)]
pub(crate) fn test_sampling(
    temperature: f64,
    top_p: f64,
    top_k: i64,
    repetition_penalty: f64,
    penalty_window: i64,
    min_tokens: i64,
    max_tokens: i64,
) -> Sampling {
    Sampling::semantic_default()
        .with_overrides(&crate::protocol::SamplingOverrides {
            temperature: Some(temperature),
            top_p: Some(top_p),
            top_k: Some(top_k),
            repetition_penalty: Some(repetition_penalty),
            penalty_window: Some(penalty_window),
            min_tokens: Some(min_tokens),
            max_tokens: Some(max_tokens),
        })
        .expect("valid test sampling")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ABC_START, MUSIC_START, VOCAB_SIZE};

    const VOCAB: usize = VOCAB_SIZE as usize;

    fn greedy(min_tokens: i64) -> Sampling {
        test_sampling(0.0, 1.0, 1, 1.0, 1, min_tokens, min_tokens.max(1))
    }

    #[test]
    fn phase_masks_are_the_protocol_ranges() {
        assert!(Phase::Abc.allows(0) && Phase::Abc.allows(EOD - 1) && Phase::Abc.allows(ABC_END));
        for id in [EOD, ABC_START, MUSIC_END, CODEC_OFFSET, VOCAB_SIZE - 1] {
            assert!(!Phase::Abc.allows(id), "abc allowed {id}");
        }
        assert!(Phase::Semantic.allows(CODEC_OFFSET));
        assert!(Phase::Semantic.allows(CODEC_OFFSET + CODEC_SIZE - 1));
        assert!(Phase::Semantic.allows(MUSIC_END));
        for id in [0, EOD - 1, ABC_END, MUSIC_START, CODEC_OFFSET + CODEC_SIZE] {
            assert!(!Phase::Semantic.allows(id), "semantic allowed {id}");
        }
    }

    /// Greedy picks the best allowed id even when a disallowed id scores higher, and the end id is
    /// barred until `min_tokens` outputs.
    #[test]
    fn masks_and_min_tokens_steer_greedy() {
        let mut logits = vec![0.0f32; VOCAB];
        logits[EOD as usize] = 50.0; // disallowed in both phases
        logits[MUSIC_END as usize] = 10.0;
        logits[CODEC_OFFSET as usize + 7] = 5.0;
        logits[42] = 9.0; // text: allowed only in ABC
        let s = distribution(
            &logits,
            &greedy(2),
            &[],
            0,
            Phase::Semantic,
            Arith::F32,
            false,
        );
        assert_eq!(argmax(&s), CODEC_OFFSET + 7, "end barred before min_tokens");
        let s = distribution(
            &logits,
            &greedy(2),
            &[],
            2,
            Phase::Semantic,
            Arith::F32,
            false,
        );
        assert_eq!(argmax(&s), MUSIC_END, "end allowed at min_tokens");
        logits[ABC_END as usize] = 8.0;
        let s = distribution(&logits, &greedy(0), &[], 0, Phase::Abc, Arith::F32, false);
        assert_eq!(argmax(&s), 42);
    }

    /// The penalty compounds per occurrence inside the window and ignores older history.
    #[test]
    fn window_penalty_compounds_per_occurrence_within_the_window() {
        let mut scores = vec![2.0f32, -2.0, 2.0, 1.0];
        window_penalty(&mut scores, &[0, 0, 1, 3], 2.0, Arith::F32);
        assert_eq!(scores, vec![0.5, -4.0, 2.0, 0.5]);
        let sampling = test_sampling(0.0, 1.0, 1, 2.0, 2, 0, 1);
        let mut logits = vec![0.0f32; VOCAB];
        logits[10] = 4.0;
        logits[11] = 3.0;
        // 10 is outside the 2-token window: unpenalized, still the argmax.
        let s = distribution(
            &logits,
            &sampling,
            &[10, 11, 11],
            3,
            Phase::Abc,
            Arith::F32,
            false,
        );
        assert_eq!((s[10], s[11]), (4.0, 0.75));
        // Inside the window it is halved below 11's 3.0.
        let s = distribution(
            &logits,
            &sampling,
            &[12, 10],
            3,
            Phase::Abc,
            Arith::F32,
            false,
        );
        assert_eq!(argmax(&s), 11);
    }

    #[test]
    fn top_k_keeps_ties_and_top_p_keeps_the_head() {
        let mut s = vec![5.0f32, 4.0, 4.0, 4.0, 1.0];
        top_k(&mut s, 2);
        assert_eq!(s, vec![5.0, 4.0, 4.0, 4.0, f32::NEG_INFINITY]);
        // One dominant entry: top_p = 0.1 keeps exactly the head (keep = 1) or three (legacy).
        let base = vec![10.0f32, 1.0, 0.9, 0.8, 0.7];
        let mut one = base.clone();
        top_p(&mut one, 0.1, Arith::F32, 1);
        assert_eq!(one.iter().filter(|x| x.is_finite()).count(), 1);
        let mut three = base;
        top_p(&mut three, 0.1, Arith::F32, 3);
        assert_eq!(three.iter().filter(|x| x.is_finite()).count(), 3);
    }

    #[test]
    fn bf16_rounding_is_nearest_even() {
        assert_eq!(round_bf16(1.0), 1.0);
        assert_eq!(round_bf16(1.2), 1.203125);
        assert_eq!(round_bf16(0.95), 0.94921875);
        // Exactly halfway between 1.0 and 1.0078125 rounds to the even (1.0).
        assert_eq!(round_bf16(1.0 + 2f32.powi(-8)), 1.0);
        assert_eq!(round_bf16(1.0 + 3.0 * 2f32.powi(-8)), 1.015625);
        assert!(round_bf16(f32::NAN).is_nan());
        assert_eq!(round_bf16(f32::NEG_INFINITY), f32::NEG_INFINITY);
    }

    #[test]
    fn draw_is_the_inverse_cdf_in_vocabulary_order() {
        let p = [0.0f32, 0.25, 0.0, 0.5, 0.25];
        assert_eq!(draw(&p, 0.0), Some(1));
        assert_eq!(draw(&p, 0.2499), Some(1));
        assert_eq!(draw(&p, 0.25), Some(3));
        assert_eq!(draw(&p, 0.7499), Some(3));
        assert_eq!(draw(&p, 0.75), Some(4));
        assert_eq!(draw(&p, 0.999_999_9), Some(4));
        assert_eq!(draw(&[0.0, 0.0], 0.5), None);
    }

    #[test]
    fn cfg_mix_is_the_released_formula_in_the_model_dtype() {
        let got = cfg_mix(&[2.0, -1.0], &[1.0, 1.0], 1.5, Arith::F32);
        assert_eq!(got, vec![2.5, -2.0]);
        // Scale 1 reproduces the conditional row exactly.
        assert_eq!(
            cfg_mix(&[0.3, 7.0], &[1.1, -2.0], 1.0, Arith::F32),
            vec![0.3, 7.0]
        );
        // BF16: every intermediate is rounded.
        let (c, u) = (round_bf16(1.3), round_bf16(0.7));
        let want = round_bf16(u + round_bf16(1.01f32 * round_bf16(c - u)));
        assert_eq!(cfg_mix(&[c], &[u], 1.01, Arith::Bf16), vec![want]);
    }
}
