//! Token sampling.
//!
//! This is the union of the three hand-rolled mlx-gen samplers (prompt-refine: temperature + top-p;
//! JoyCaption: + repetition penalty; sensenova: + top-k + on-device greedy argmax), unified so each
//! reference reduces to a special case:
//!
//! * prompt-refine parity: `top_k = 0`, `repetition_penalty = 1.0` → temperature + top-p.
//! * JoyCaption parity: `top_k = 0`, `repetition_penalty = 1.05` → penalty + temperature + top-p.
//! * sensenova parity: `repetition_penalty = 1.0`, `top_k > 0` → temperature + top-k + top-p.
//!
//! The math matches the references (except the nucleus mass, accumulated in f64 — see
//! `nucleus_select`): a stabilised, **unnormalised** `exp((logit - max)/T)` weight per token
//! (equivalent to softmax for nucleus selection and for the inverse-CDF draw, since both scale by
//! the total), heap-based nucleus selection (verified against a full sort), and a categorical
//! inverse-CDF draw from the pluggable [`TokenRng`]. Greedy (`temperature <= 0`) with
//! no penalty and no constraint takes the on-device argmax fast path (a single-element host
//! transfer) like sensenova's `decode_argmax`; otherwise [`sample`] pulls the logits to host f32.
//! [`sample`] is the **host reference** and stays so: the mlx-gen stacks' seeded parity rides it.
//!
//! **The shared decode sampler** (epic sc-24432, story sc-24439) is [`sample_with_path`]: every MLX
//! decode loop (the speculative engine, the plain / prefix-cached loop, the batch and continuous
//! lanes) draws through it. It routes each draw by [`sampler_path`] — the backend-neutral
//! [`core_llm::Sampling::sampler_path`] policy plus the degenerate-temperature guard Candle applies:
//!
//! * **device** — plain greedy (the on-device argmax) and temperature / top-k / top-p
//!   ([`sample_device`]): the draw is a lazy one-element array ([`SampledToken::Device`]); only the
//!   chosen id ever crosses to the host, and only when a caller resolves it. The device rule is
//!   Candle's: the same weights, top-k and top-p as weight thresholds, and an **index-order**
//!   inverse-CDF draw from one host uniform of the seeded [`TokenRng`] — the host reference's
//!   distribution, not its per-seed draws (the reference walks the nucleus in heap order), so
//!   seeded non-greedy output differs from the pre-sc-24439 host sampler.
//! * **host** — a constraint mask or a repetition / presence penalty (both read host state: the
//!   grammar and the history window) and a temperature whose reciprocal is not a finite, non-zero
//!   scale copy the row to the host, and the path says why. The host path shapes the row as
//!   [`sample`] does but draws by the device rule (index order, one uniform), so a seed lands on
//!   the same token either way; only the [`sample`] reference keeps the heap-order walk.
//!
//! Every device-to-host read the sampling seam makes is counted per thread ([`host_transfers`]), so
//! "one token id per step" is asserted, not assumed.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use mlx_rs::ops::indexing::{argmax, argmax_axis, IndexOp};
use mlx_rs::ops::{cumsum, exp, maximum, minimum, sort, sum};
use mlx_rs::{Array, Dtype};

use core_llm::{HostSampleReason, SamplerPath};

use crate::error::{Error, Result};
use crate::primitives::nn::input_ids;

// ---------------------------------------------------------------------------------------------
// Host-transfer accounting.
// ---------------------------------------------------------------------------------------------

/// Device-to-host reads the sampling seam made on the current thread: how many reads, and how many
/// elements they carried. A device draw read back costs one read of one element; a host draw
/// copies the whole logits row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostTransfers {
    /// Reads (host synchronizations on a sampled value or a logits row).
    pub reads: u64,
    /// Elements those reads carried.
    pub elements: u64,
}

impl HostTransfers {
    /// The reads made since `earlier` (a previous [`host_transfers`] snapshot on this thread).
    pub fn since(self, earlier: HostTransfers) -> HostTransfers {
        HostTransfers {
            reads: self.reads - earlier.reads,
            elements: self.elements - earlier.elements,
        }
    }
}

thread_local! {
    static HOST_TRANSFERS: Cell<HostTransfers> = const {
        Cell::new(HostTransfers { reads: 0, elements: 0 })
    };
}

/// This thread's running [`HostTransfers`] total. Take a snapshot before and after a run and
/// subtract ([`HostTransfers::since`]).
pub fn host_transfers() -> HostTransfers {
    HOST_TRANSFERS.with(Cell::get)
}

fn note_host_read(elements: usize) {
    HOST_TRANSFERS.with(|t| {
        let mut v = t.get();
        v.reads += 1;
        v.elements += elements as u64;
        t.set(v);
    });
}

// ---------------------------------------------------------------------------------------------
// A drawn token.
// ---------------------------------------------------------------------------------------------

/// One drawn token: already on the host, or still device-resident — a single-element integer
/// array the next forward can consume before it is ever read back. A decode loop resolves a device
/// token to the host only where it must (the stop-token check, the history / penalty window, the
/// constraint, the emitted event) and feeds the next step's forward from the array itself, so a
/// pipelined loop can enqueue step `t + 1` before step `t`'s id is read.
#[derive(Clone, Debug)]
pub enum SampledToken {
    /// The id, on the host.
    Host(i32),
    /// A one-element integer array holding the id, on the device (int32 for the shared sampler's
    /// draws, so reading it back queues no op).
    Device(Array),
}

impl SampledToken {
    /// The id on the host. A device token is evaluated and read back — one counted
    /// [`host_transfers`] read of one element.
    pub fn resolve(&self) -> Result<i32> {
        match self {
            SampledToken::Host(id) => Ok(*id),
            SampledToken::Device(id) => {
                // Read the drawn array itself: a cast or reshape here would be a new op queued on
                // the stream *behind* any step already enqueued after this draw, and the read
                // would wait for that whole step — undoing the pipelining.
                let value = if id.dtype() == Dtype::Int32 {
                    id.item::<i32>()
                } else {
                    id.as_dtype(Dtype::Int32)?.item::<i32>()
                };
                note_host_read(1);
                Ok(value)
            }
        }
    }

    /// The `[1, 1]` int32 input ids of the forward that consumes this token — built from the
    /// device array without reading it back.
    pub fn input(&self) -> Result<Array> {
        match self {
            SampledToken::Host(id) => Ok(input_ids(&[*id])),
            SampledToken::Device(id) => Ok(id.as_dtype(Dtype::Int32)?.reshape(&[1, 1])?),
        }
    }

    /// Whether the id is still on the device.
    pub fn is_device(&self) -> bool {
        matches!(self, SampledToken::Device(_))
    }
}

/// A pluggable random source for the categorical draw. Greedy decoding never touches it, so an
/// unused RNG produces bit-identical (deterministic) output.
pub trait TokenRng {
    /// The next sample in `[0, 1)`.
    fn next_f32(&mut self) -> f32;
}

/// SplitMix64 — the deterministic, seedable PRNG the mlx-gen stacks use. Reproduced verbatim
/// (same constants, same `next_f32` 24-bit mantissa) so seeded runs match the reference engines.
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
    /// Additive once-per-seen-token presence penalty over the full prompt + generated history.
    pub presence_penalty: f32,
    /// CTRL/HF repetition penalty. `1.0` disables it. Applied **once per distinct id** in the
    /// window (Hugging Face `RepetitionPenaltyLogitsProcessor` gathers and scatters, so a repeated
    /// id is divided / multiplied once, never compounded by its count).
    pub repetition_penalty: f32,
    /// How many recent history tokens the repetition penalty looks back over.
    pub repetition_context: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            repetition_context: 0,
        }
    }
}

impl SamplingParams {
    /// True when this configuration is pure greedy with no penalty and (caller-checked) no
    /// constraint mask — eligible for the on-device argmax fast path.
    pub fn is_plain_greedy(&self) -> bool {
        self.temperature <= 0.0 && !self.is_penalized()
    }

    /// True when a repetition or presence penalty reads the history window — a host-path draw.
    pub fn is_penalized(&self) -> bool {
        self.presence_penalty != 0.0 || self.repetition_penalty != 1.0
    }
}

/// Sample the next token id from `logits` on the **host reference** (see the module docs; the
/// decode loops draw through [`sample_with_path`]).
///
/// This is the heap-order reference kept for the mlx-gen media pipelines' seeded parity with their
/// upstream references; mlx-llm decode loops must draw through [`draw_token`] /
/// [`sample_with_path`] instead.
///
/// * `logits` — `[vocab]` or `[1, vocab]` for the current (last) position.
/// * `history` — token ids already in the sequence (prompt + generated), for the repetition
///   penalty. Pass an empty slice when the penalty is disabled.
/// * `allowed` — an optional per-vocab mask (e.g. from a JSON-constraint grammar); `false` (or
///   out-of-range) entries are forced to `-inf`.
pub fn sample(
    logits: &Array,
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
        return Ok(argmax_host(&v)); // a NaN / -inf row; deterministic fallback
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

/// Draw one token on the **shared decode sampler**, returning the draw and where it happened
/// ([`sampler_path`]): the device path returns the lazy on-device argmax (plain greedy) or
/// [`sample_device`]'s draw (temperature / top-k / top-p) as [`SampledToken::Device`] — nothing is
/// read back until the caller resolves it; the host path copies the row, applies the mask and
/// penalties, and draws by the same rule (see the module docs). The path is
/// the branch that ran, never re-derived from the request later, so a decode report measures it
/// (epic sc-24432, stories sc-24434 / sc-24439).
///
/// A stochastic device draw consumes exactly one [`TokenRng::next_f32`], as a non-degenerate host
/// draw does, and greedy none. A constrained row whose mask allows no token is
/// [`Error::NoAllowedToken`] — never a silent draw of a forbidden id.
pub fn sample_with_path(
    logits: &Array,
    history: &[i32],
    params: &SamplingParams,
    rng: &mut impl TokenRng,
    allowed: Option<&[bool]>,
) -> Result<(SampledToken, SamplerPath)> {
    let path = sampler_path(params, allowed.is_some());
    let token = match path {
        SamplerPath::Device if params.temperature <= 0.0 => {
            SampledToken::Device(argmax(logits.reshape(&[-1])?, None)?.as_dtype(Dtype::Int32)?)
        }
        SamplerPath::Device => SampledToken::Device(sample_device(logits, params, rng.next_f32())?),
        SamplerPath::Host(_) => SampledToken::Host(sample_host_index_order(
            logits, history, params, rng, allowed,
        )?),
    };
    Ok((token, path))
}

/// The shared sampler's host path: the reference's mask, penalties and shaped weights, drawn by
/// [`sample_device`]'s rule — an index-order inverse CDF over the kept weights from one uniform, the
/// uniform consumed by every stochastic row (degenerate ones too). A seed therefore draws the same
/// token on the host and on the device (rounding knife-edges and threshold ties aside), so a
/// permissive constraint or a zero-strength penalty never changes a seeded run.
fn sample_host_index_order(
    logits: &Array,
    history: &[i32],
    params: &SamplingParams,
    rng: &mut impl TokenRng,
    allowed: Option<&[bool]>,
) -> Result<i32> {
    let v = penalized_logits(logits, history, params, allowed)?;
    if params.temperature <= 0.0 {
        return Ok(argmax_host(&v));
    }
    let u = rng.next_f32();
    if degenerate_temperature(params.temperature) || v.iter().any(|x| x.is_nan()) {
        return Ok(argmax_host(&v));
    }
    let mut weights = nucleus_weights(&v, params);
    weights.sort_unstable_by_key(|x| x.0);
    let total: f64 = weights.iter().map(|x| f64::from(x.1)).sum();
    if weights.is_empty() || !(total > 0.0 && total.is_finite()) {
        return Ok(argmax_host(&v));
    }
    let target = f64::from(u) * total;
    let mut running = 0.0f64;
    for &(i, w) in &weights {
        running += f64::from(w);
        if w > 0.0 && running > target {
            return Ok(i as i32);
        }
    }
    Ok(weights.last().map_or(0, |x| x.0) as i32)
}

/// [`sample_with_path`], read back: the shared decode sampler for a loop that needs every id on the
/// host before its next step (the plain, batch and continuous loops). A device draw costs one
/// one-element read.
pub fn draw_token(
    logits: &Array,
    history: &[i32],
    params: &SamplingParams,
    rng: &mut impl TokenRng,
    allowed: Option<&[bool]>,
) -> Result<i32> {
    sample_with_path(logits, history, params, rng, allowed)?
        .0
        .resolve()
}

/// Where [`sample_with_path`] draws for `params`: the backend-neutral
/// [`core_llm::Sampling::sampler_path`] policy — a constraint mask or a penalty reads host state,
/// so it takes the host — plus Candle's degenerate-temperature guard: a positive temperature whose
/// reciprocal is not a finite, non-zero scale (subnormal, `+inf`, NaN) cannot shape device weights
/// and takes the host reference ([`HostSampleReason::DegenerateTemperature`]). Everything else,
/// greedy or stochastic, draws on the device — unless the device sampler is switched off
/// ([`DEVICE_SAMPLER`](crate::switches::DEVICE_SAMPLER)), when it takes the host reference
/// ([`HostSampleReason::Reference`]).
pub fn sampler_path(params: &SamplingParams, constrained: bool) -> SamplerPath {
    if constrained {
        SamplerPath::Host(HostSampleReason::Constraint)
    } else if params.is_penalized() {
        SamplerPath::Host(HostSampleReason::Penalty)
    } else if degenerate_temperature(params.temperature) {
        SamplerPath::Host(HostSampleReason::DegenerateTemperature)
    } else if !crate::switches::DEVICE_SAMPLER.enabled() {
        SamplerPath::Host(HostSampleReason::Reference)
    } else {
        SamplerPath::Device
    }
}

/// A positive (or NaN) temperature whose reciprocal is not a usable finite, non-zero scale.
fn degenerate_temperature(temperature: f32) -> bool {
    if temperature <= 0.0 {
        return false; // greedy
    }
    let inv_t = 1.0 / temperature;
    !inv_t.is_finite() || inv_t == 0.0
}

/// Draw one token from a `[vocab]` / `[1, vocab]` logits row **on the device**, given the uniform
/// `u` in `[0, 1)`: temperature, top-k and top-p with no penalty and no mask (the host reference
/// handles those). Returns a lazy scalar int32 array holding the id; nothing is synchronized.
///
/// The rule is Candle's device sampler's (`candle-llm` `sample_device`): the reference's stabilised
/// weights `exp((logit - max) / T)`; top-k and top-p applied as **weight thresholds** (the `k`-th
/// largest weight, and the lightest weight of the heaviest-first prefix whose mass reaches
/// `top_p` of the top-k mass — found from one ascending sort); then an inverse-CDF draw over the
/// kept weights walked in **vocabulary order**. The kept set is the reference's except at an exact
/// weight tie on a threshold, where every tied token is kept (the reference keeps the lower index);
/// the walk order differs from the reference's heap order, so the distribution matches while
/// per-seed draws do not. A row whose weights degenerate (a NaN, a `+inf` maximum) falls back to
/// its argmax, as the reference does.
pub fn sample_device(logits: &Array, params: &SamplingParams, u: f32) -> Result<Array> {
    let x = logits.reshape(&[-1])?.as_dtype(Dtype::Float32)?;
    let vocab = x.shape()[0];
    let max = x.max(None)?;
    let inv_t = Array::from_f32(1.0 / params.temperature);
    let w = exp(x.subtract(&max)?.multiply(&inv_t)?)?;
    let zero = Array::from_f32(0.0);

    // The weight floor top-k and top-p impose, from one ascending sort of the weights.
    let top_k = params.top_k > 0 && params.top_k < vocab as usize;
    let top_p = params.top_p < 1.0;
    let floor = if top_k || top_p {
        let asc = sort(&w)?;
        let mut floor = None;
        let mut kept = asc.clone();
        if top_k {
            let kth = asc.index(vocab - params.top_k as i32);
            kept = mlx_rs::ops::r#where(&asc.ge(&kth)?, &asc, &zero)?;
            floor = Some(kth);
        }
        if top_p {
            // Heaviest-first: an entry is in the nucleus while the mass strictly heavier than it is
            // still below `top_p` of the total; the heaviest entry is always kept.
            let heavier_or_equal = cumsum(&kept, 0, true, true)?;
            let heavier = heavier_or_equal.subtract(&kept)?;
            let threshold = sum(&kept, None)?.multiply(Array::from_f32(params.top_p.max(0.0)))?;
            let in_nucleus = heavier.lt(&threshold)?;
            // The heaviest entry bounds the floor from above, built on the device: no per-draw
            // vocab-length host mask.
            let lightest = minimum(
                mlx_rs::ops::r#where(&in_nucleus, &asc, Array::from_f32(f32::INFINITY))?
                    .min(None)?,
                asc.index(vocab - 1),
            )?;
            floor = Some(match floor {
                Some(kth) => maximum(&kth, &lightest)?,
                None => lightest,
            });
        }
        floor
    } else {
        None
    };
    let kept = match floor {
        Some(floor) => mlx_rs::ops::r#where(&w.ge(&floor)?, &w, &zero)?,
        None => w,
    };

    // Index-order inverse CDF: the first id whose running mass exceeds `u` of the total. The target
    // is held strictly below the total so rounding can never walk past the last kept id.
    let running = cumsum(&kept, 0, None, None)?;
    let total = running.index(vocab - 1);
    let below_total = total.multiply(Array::from_f32(1.0 - f32::EPSILON))?;
    let target = minimum(total.multiply(Array::from_f32(u))?, &below_total)?;
    let drawn = sum(running.le(&target)?, None)?.as_dtype(Dtype::Int32)?;
    let usable = total.is_finite()?.logical_and(&total.gt(&zero)?)?;
    let fallback = argmax(&x, None)?.as_dtype(Dtype::Int32)?;
    Ok(mlx_rs::ops::r#where(&usable, &drawn, &fallback)?)
}

/// The shaped candidate distribution `sample` would draw from for a **stochastic** (`temperature >
/// 0`) configuration: `(token_id, unnormalised_weight)` after the constraint mask, repetition
/// penalty, temperature, top-k, and top-p — the distribution speculative decoding (stories 7171 /
/// 7172) feeds to the backend-neutral acceptance sampler. Empty when everything is masked out.
pub fn shaped_candidates(
    logits: &Array,
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

/// Pull `logits` to host f32 and apply the constraint mask + repetition penalty (the position-shaping
/// shared by `sample` and [`shaped_candidates`]). A mask that allows no id of the row is
/// [`Error::NoAllowedToken`].
fn penalized_logits(
    logits: &Array,
    history: &[i32],
    params: &SamplingParams,
    allowed: Option<&[bool]>,
) -> Result<Vec<f32>> {
    let lf = logits.as_dtype(Dtype::Float32)?;
    let mut v: Vec<f32> = lf.as_slice::<f32>().to_vec();
    note_host_read(v.len());
    if let Some(mask) = allowed {
        if !mask.iter().take(v.len()).any(|&a| a) {
            return Err(Error::NoAllowedToken { vocab: v.len() });
        }
    }
    penalize_host(&mut v, history, params, allowed);
    Ok(v)
}

/// The constraint mask plus repetition and presence penalties over host logits, in place (the
/// position-shaping [`penalized_logits`] applies after the device read).
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

    // Repetition penalty (Keskar et al. 2019 / HF CTRL formulation) over the recent window,
    // once per distinct id: HF gathers each id's logit and scatters the transformed value back,
    // so an id repeated in the window is penalised exactly once.
    if params.repetition_penalty != 1.0 && params.repetition_context > 0 {
        let start = history.len().saturating_sub(params.repetition_context);
        let mut penalized = std::collections::HashSet::with_capacity(history.len() - start);
        for &tok in history[start..].iter().filter(|&&t| penalized.insert(t)) {
            if let Some(slot) = usize::try_from(tok).ok().and_then(|i| v.get_mut(i)) {
                *slot = if *slot < 0.0 {
                    *slot * params.repetition_penalty
                } else {
                    *slot / params.repetition_penalty
                };
            }
        }
    }
    // Presence penalty is additive and count-independent: every id seen anywhere in the prompt or
    // generated history is adjusted exactly once, even when it occurs repeatedly.
    if params.presence_penalty != 0.0 {
        let mut seen = std::collections::HashSet::with_capacity(history.len());
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

/// On-device argmax of a `[vocab]` / `[1, vocab]` logits row. Greedy fast path — avoids pulling
/// the full vocabulary to the host. Ties break to the lowest index (matching MLX and the host
/// scan), so greedy decoding is bit-identical whichever path is taken.
pub fn argmax_device(logits: &Array) -> Result<i32> {
    let flat = logits.reshape(&[-1])?;
    let idx = argmax(&flat, None)?;
    let id = idx.item::<u32>() as i32;
    note_host_read(1);
    Ok(id)
}

/// On-device argmax of **every** row of a `[.., n, vocab]` logits block, brought to the host in one
/// transfer — the speculative verify step's greedy decision (epic sc-24432, story sc-24434), where
/// the per-row [`argmax_device`] would pay `n` transfers. Ties break to the lowest index, exactly as
/// [`argmax_device`] does per row, so the chosen ids are identical.
pub fn argmax_rows_device(logits: &Array) -> Result<Vec<i32>> {
    let vocab = logits.shape().last().copied().unwrap_or(0);
    let rows = logits.reshape(&[-1, vocab])?;
    let idx = argmax_axis(&rows, -1, None)?;
    let ids: Vec<i32> = idx.as_slice::<u32>().iter().map(|&t| t as i32).collect();
    note_host_read(ids.len());
    Ok(ids)
}

/// The target distribution a speculative acceptance test ([`core_llm::speculative::accept_token`])
/// checks a draft against: [`shaped_candidates`], or — when nothing survives the shaping (a NaN or
/// `+inf` maximum) — the point mass on the penalized row's argmax, which is the token [`sample`]
/// itself commits for such a row. Never empty, so a degenerate row can never accept a draft (or
/// commit a fallback id) the row did not choose. A constrained row whose mask allows no token is
/// [`Error::NoAllowedToken`], exactly as in [`sample`].
pub fn acceptance_target(
    logits: &Array,
    history: &[i32],
    params: &SamplingParams,
    allowed: Option<&[bool]>,
) -> Result<Vec<(i32, f32)>> {
    let v = penalized_logits(logits, history, params, allowed)?;
    let weights = nucleus_weights(&v, params);
    let total: f32 = weights.iter().map(|x| x.1).sum();
    if weights.is_empty() || total <= 0.0 || !total.is_finite() {
        return Ok(vec![(argmax_host(&v), 1.0)]);
    }
    Ok(weights.into_iter().map(|(i, w)| (i as i32, w)).collect())
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
/// `top_p * total`, always keeping at least one. Equivalent to a descending sort + prefix for
/// distinct weights (the references verify this against a full sort); ties break to lower index.
/// The mass is accumulated in f64 (sc-24133, matching `candle-llm`'s `nucleus_select`): an f32
/// running sum over a wide vocabulary drifts by more than a tail token's weight and moved the
/// nucleus boundary by rounding alone, so knife-edge boundaries may differ from the f32 mlx-gen
/// references by one token.
fn nucleus_select(weights: &[(usize, f32)], top_p: f32) -> Vec<(usize, f32)> {
    let total: f64 = weights.iter().map(|x| f64::from(x.1)).sum();
    let threshold = f64::from(top_p.max(0.0)) * total;
    let mut heap: BinaryHeap<ByWeight> = weights.iter().map(|&(i, w)| ByWeight(i, w)).collect();
    let mut kept = Vec::new();
    let mut cum = 0.0f64;
    while let Some(ByWeight(i, w)) = heap.pop() {
        kept.push((i, w));
        cum += f64::from(w);
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

    fn logits(v: &[f32]) -> Array {
        Array::from_slice(v, &[1, v.len() as i32])
    }

    #[test]
    fn splitmix64_matches_known_sequence() {
        // SplitMix64(0) reference outputs (standard constants).
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
    fn top_p_restricts_to_nucleus() {
        // One token dominates; top_p just above it keeps essentially only that token.
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.5,
            ..Default::default()
        };
        let l = logits(&[10.0, 0.0, 0.0, 0.0]);
        let mut rng = SplitMix64::new(7);
        for _ in 0..50 {
            let t = sample(&l, &[], &params, &mut rng, None).unwrap();
            assert_eq!(t, 0);
        }
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

    /// A constrained row whose mask allows nothing is a typed error from every sampling entry
    /// point — greedy, stochastic and the speculative acceptance target — never the old silent
    /// fallback to id 0 (a forbidden token). An allowed id past the row does not count.
    #[test]
    fn a_fully_masked_row_is_a_named_error() {
        let l = logits(&[0.1, 0.2, 9.0, 0.3]);
        let stochastic = SamplingParams {
            temperature: 0.8,
            ..Default::default()
        };
        for mask in [
            &[false; 4][..],
            &[false, false, false, false, true][..],
            &[][..],
        ] {
            for params in [SamplingParams::default(), stochastic] {
                let mut rng = SplitMix64::new(0);
                let err = sample_with_path(&l, &[], &params, &mut rng, Some(mask)).unwrap_err();
                assert!(
                    matches!(err, Error::NoAllowedToken { vocab: 4 }),
                    "{mask:?} {params:?}: {err}"
                );
                let err = acceptance_target(&l, &[], &params, Some(mask)).unwrap_err();
                assert!(
                    matches!(err, Error::NoAllowedToken { vocab: 4 }),
                    "{mask:?} {params:?}: {err}"
                );
            }
        }
        // One allowed id is enough.
        let mut rng = SplitMix64::new(0);
        let mask = [false, false, false, true];
        let t = sample(&l, &[], &SamplingParams::default(), &mut rng, Some(&mask)).unwrap();
        assert_eq!(t, 3);
    }

    /// The reported path is the branch that ran: the on-device argmax only for a plain greedy,
    /// unconstrained draw; otherwise the host, with the reason the row came there.
    /// sc-24446 (E5): with the device sampler switched off (`MLX_LLM_DEVICE_SAMPLER`; here its
    /// thread-scoped layer) a request that would draw on the device takes the host reference.
    #[test]
    fn the_device_sampler_switch_off_routes_to_the_host_reference() {
        let stochastic = SamplingParams {
            temperature: 0.8,
            ..Default::default()
        };
        for params in [SamplingParams::default(), stochastic] {
            assert_eq!(sampler_path(&params, false), SamplerPath::Device);
            let off =
                crate::switches::DEVICE_SAMPLER.scoped(false, || sampler_path(&params, false));
            assert_eq!(off, SamplerPath::Host(HostSampleReason::Reference));
        }
    }

    #[test]
    fn sample_with_path_reports_the_branch_that_ran() {
        let l = logits(&[0.1, 0.2, 9.0, 0.3]);
        let mut rng = SplitMix64::new(0);
        let greedy = SamplingParams::default();
        let penalized = SamplingParams {
            presence_penalty: 0.5,
            ..Default::default()
        };
        let stochastic = SamplingParams {
            temperature: 0.8,
            ..Default::default()
        };
        let degenerate = SamplingParams {
            temperature: 1e-39,
            ..Default::default()
        };
        let mask = [true; 4];
        for (params, allowed, path) in [
            (greedy, None, SamplerPath::Device),
            (stochastic, None, SamplerPath::Device),
            (
                degenerate,
                None,
                SamplerPath::Host(HostSampleReason::DegenerateTemperature),
            ),
            (
                greedy,
                Some(&mask[..]),
                SamplerPath::Host(HostSampleReason::Constraint),
            ),
            (
                penalized,
                None,
                SamplerPath::Host(HostSampleReason::Penalty),
            ),
            (
                stochastic,
                Some(&mask[..]),
                SamplerPath::Host(HostSampleReason::Constraint),
            ),
        ] {
            let (token, got) = sample_with_path(&l, &[], &params, &mut rng, allowed).unwrap();
            assert_eq!(got, path, "{params:?} {allowed:?}");
            assert_eq!(sampler_path(&params, allowed.is_some()), path);
            // A device draw is returned unread; a host draw is already on the host.
            assert_eq!(token.is_device(), path == SamplerPath::Device, "{params:?}");
        }
        for t in [f32::INFINITY, f32::NAN] {
            let params = SamplingParams {
                temperature: t,
                ..Default::default()
            };
            assert_eq!(
                sampler_path(&params, false),
                SamplerPath::Host(HostSampleReason::DegenerateTemperature)
            );
        }
    }

    /// The host walk the device rule reproduces: the reference's shaped weights, walked in
    /// vocabulary order, the first id whose running mass exceeds `u` of the total — with the
    /// running mass at the chosen boundary, so a caller can skip rounding knife-edges.
    fn index_order_reference(v: &[f32], params: &SamplingParams, u: f32) -> (i32, f64) {
        let mut weights = nucleus_weights(v, params);
        weights.sort_unstable_by_key(|x| x.0);
        let total: f64 = weights.iter().map(|x| f64::from(x.1)).sum();
        let target = f64::from(u) * total;
        let mut cum = 0.0f64;
        for (i, w) in &weights {
            cum += f64::from(*w);
            if cum > target {
                return (
                    *i as i32,
                    ((cum - target).min(target - (cum - f64::from(*w)))) / total,
                );
            }
        }
        unreachable!("u < 1 always lands inside the mass")
    }

    fn device_draw(v: &[f32], params: &SamplingParams, u: f32) -> i32 {
        let before = host_transfers();
        let id = SampledToken::Device(sample_device(&logits(v), params, u).unwrap())
            .resolve()
            .unwrap();
        // Only the chosen id crossed to the host.
        assert_eq!(
            host_transfers().since(before),
            HostTransfers {
                reads: 1,
                elements: 1
            }
        );
        id
    }

    const ROW: [f32; 12] = [
        1.3, -0.4, 2.1, 0.7, 2.0, -1.5, 0.2, 1.9, 0.9, -0.1, 1.1, 0.4,
    ];

    fn knob_sets() -> Vec<SamplingParams> {
        let with = |temperature, top_p, top_k| SamplingParams {
            temperature,
            top_p,
            top_k,
            ..Default::default()
        };
        vec![
            with(0.7, 0.9, 0),
            with(1.0, 1.0, 0),
            with(0.8, 1.0, 5),
            with(1.3, 0.75, 7),
            with(0.5, 0.0, 0),
        ]
    }

    /// The device rule is the index-order inverse CDF over the reference's kept weights: for every
    /// knob set and a sweep of uniforms, the device id equals the host walk's (rounding
    /// knife-edges excepted), and never falls outside the reference's kept set.
    #[test]
    fn the_device_draw_is_the_index_order_walk_over_the_reference_weights() {
        for params in knob_sets() {
            let kept: Vec<i32> = shaped_candidates(&logits(&ROW), &[], &params, None)
                .unwrap()
                .into_iter()
                .map(|x| x.0)
                .collect();
            for i in 0..200 {
                let u = (i as f32 + 0.5) / 200.0;
                let got = device_draw(&ROW, &params, u);
                assert!(
                    kept.contains(&got),
                    "{params:?} u={u}: {got} not kept {kept:?}"
                );
                let (want, margin) = index_order_reference(&ROW, &params, u);
                if margin > 1e-5 {
                    assert_eq!(got, want, "{params:?} u={u}");
                }
            }
        }
    }

    /// AC2: over many seeds, the device sampler's draws follow the host sampler's distribution
    /// (chi-square against the reference's normalised shaped weights), for temperature 0.7 /
    /// top-p 0.9 and the other knob sets.
    #[test]
    fn device_draws_match_the_host_distribution_chi_square() {
        let n = 6_000u64;
        for params in knob_sets() {
            let reference = shaped_candidates(&logits(&ROW), &[], &params, None).unwrap();
            let total: f64 = reference.iter().map(|x| f64::from(x.1)).sum();
            let mut counts = vec![0u64; ROW.len()];
            for seed in 0..n {
                let mut rng = SplitMix64::new(0x2443_9000 + seed);
                let (token, path) =
                    sample_with_path(&logits(&ROW), &[], &params, &mut rng, None).unwrap();
                assert_eq!(path, SamplerPath::Device);
                counts[token.resolve().unwrap() as usize] += 1;
            }
            let mut chi2 = 0.0f64;
            for &(t, w) in &reference {
                let expected = n as f64 * f64::from(w) / total;
                chi2 += (counts[t as usize] as f64 - expected).powi(2) / expected;
            }
            let outside: u64 = (0..ROW.len())
                .filter(|t| !reference.iter().any(|x| x.0 as usize == *t))
                .map(|t| counts[t])
                .sum();
            assert_eq!(
                outside, 0,
                "{params:?}: drew outside the kept set {counts:?}"
            );
            // 99.9 % chi-square critical values by degrees of freedom (kept - 1).
            const CRITICAL: [f64; 12] = [
                0.0, 10.83, 13.82, 16.27, 18.47, 20.52, 22.46, 24.32, 26.12, 27.88, 29.59, 31.26,
            ];
            let df = reference.len() - 1;
            assert!(
                chi2 < CRITICAL[df] || df == 0,
                "{params:?}: chi-square {chi2:.2} (df {df}) over {counts:?}"
            );
        }
    }

    /// One seed, one token: a permissive constraint (host path) draws exactly what the device
    /// draws for every uniform (knife-edges excepted), so a constraint that forbids nothing never
    /// changes a seeded run.
    #[test]
    fn the_host_path_draws_the_device_token_for_the_same_uniform() {
        let allow = [true; ROW.len()];
        for params in knob_sets() {
            for i in 0..100 {
                let u = (i as f32 + 0.37) / 100.0;
                let (want, margin) = index_order_reference(&ROW, &params, u);
                let host = sample_host_index_order(
                    &logits(&ROW),
                    &[],
                    &params,
                    &mut Fixed(u),
                    Some(&allow),
                )
                .unwrap();
                assert_eq!(host, want, "{params:?} u={u}");
                if margin > 1e-5 {
                    assert_eq!(device_draw(&ROW, &params, u), host, "{params:?} u={u}");
                }
            }
        }
    }

    /// A [`TokenRng`] that always returns one uniform.
    struct Fixed(f32);

    impl TokenRng for Fixed {
        fn next_f32(&mut self) -> f32 {
            self.0
        }
    }

    /// Degenerate rows fall back to the argmax on the device, as the reference does, and top-p 0
    /// keeps only the heaviest id.
    #[test]
    fn a_degenerate_device_row_falls_back_to_the_argmax() {
        let params = SamplingParams {
            temperature: 0.9,
            top_p: 0.8,
            ..Default::default()
        };
        let pos_inf = [0.1, f32::INFINITY, 0.3, 0.2];
        assert_eq!(device_draw(&pos_inf, &params, 0.3), 1);
        let greedy_nucleus = SamplingParams {
            temperature: 0.9,
            top_p: 0.0,
            ..Default::default()
        };
        for u in [0.0, 0.5, 0.999] {
            assert_eq!(device_draw(&ROW, &greedy_nucleus, u), 2);
        }
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
    fn repetition_penalty_is_applied_once_per_distinct_id() {
        // Hugging Face `RepetitionPenaltyLogitsProcessor` gathers each id's logit and scatters the
        // transformed value back: an id repeated in the window is penalised once, not per count.
        // Host-only (no MLX array), so it needs no Metal device.
        let params = SamplingParams {
            repetition_penalty: 2.0,
            repetition_context: 8,
            ..Default::default()
        };
        let mut row = vec![4.0, -2.0, 3.0];
        penalize_host(&mut row, &[0, 0, 0, 1, 1], &params, None);
        assert_eq!(row, vec![2.0, -4.0, 3.0]);
    }

    #[test]
    fn presence_penalty_applies_once_to_every_seen_token() {
        let params = SamplingParams {
            temperature: 0.0,
            presence_penalty: 0.75,
            ..Default::default()
        };
        let l = logits(&[2.0, 1.5, 0.5]);
        let mut rng = SplitMix64::new(0);
        assert_eq!(sample(&l, &[0], &params, &mut rng, None).unwrap(), 1);
        assert_eq!(
            sample(&l, &[0, 0, 0], &params, &mut rng, None).unwrap(),
            1,
            "repeat count must not multiply a presence penalty"
        );
    }

    #[test]
    fn repetition_transform_precedes_additive_presence_penalty() {
        let params = SamplingParams {
            temperature: 0.0,
            presence_penalty: 0.75,
            repetition_penalty: 2.0,
            repetition_context: 1,
            ..Default::default()
        };
        // Repetition-first transforms token 0 from 2.0 -> 1.0, then presence subtracts to 0.25,
        // so token 1 wins at 0.5. Additive-first would produce (2.0 - 0.75) / 2 = 0.625 and
        // incorrectly keep token 0.
        let l = logits(&[2.0, 0.5]);
        let mut rng = SplitMix64::new(0);
        assert_eq!(sample(&l, &[0], &params, &mut rng, None).unwrap(), 1);
    }

    #[test]
    fn nucleus_matches_full_sort_for_distinct_weights() {
        // Cross-check the heap nucleus against a brute-force descending sort + prefix.
        let weights: Vec<(usize, f32)> = vec![(0, 0.05), (1, 0.4), (2, 0.25), (3, 0.2), (4, 0.1)];
        for &top_p in &[0.3f32, 0.5, 0.7, 0.9, 0.99] {
            let got = nucleus_select(&weights, top_p);
            let mut sorted = weights.clone();
            sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
            let total: f32 = weights.iter().map(|x| x.1).sum();
            let threshold = top_p * total;
            let mut expected = Vec::new();
            let mut cum = 0.0;
            for &(i, w) in &sorted {
                expected.push((i, w));
                cum += w;
                if cum >= threshold {
                    break;
                }
            }
            let gi: Vec<usize> = got.iter().map(|x| x.0).collect();
            let ei: Vec<usize> = expected.iter().map(|x| x.0).collect();
            assert_eq!(gi, ei, "top_p={top_p}");
        }
    }

    /// The heaviest weight always bounds the top-p floor, even when the nucleus is empty
    /// (`top_p = 0`): the floor is the maximum weight, so every token tied at the maximum stays
    /// kept (the documented threshold-tie rule) and the uniform picks among them.
    #[test]
    fn an_empty_nucleus_keeps_every_token_tied_at_the_maximum() {
        let row = [2.0, 0.1, 2.0, -1.0];
        let params = SamplingParams {
            temperature: 0.7,
            top_p: 0.0,
            ..Default::default()
        };
        assert_eq!(device_draw(&row, &params, 0.25), 0);
        assert_eq!(device_draw(&row, &params, 0.75), 2);
    }

    /// The device draw builds nothing vocab-sized on the host: every per-draw array is derived on
    /// the device from the logits (only scalar knobs and the uniform cross host→device). A source
    /// scan, since MLX exposes no upload counter.
    #[test]
    fn the_device_draw_uploads_no_host_built_array() {
        let src = include_str!("sampler.rs");
        let start = src.find("pub fn sample_device(").expect("sample_device");
        let body = &src[start..];
        let body = &body[..body.find("\n}\n").expect("end of sample_device")];
        for host_built in ["from_slice", "from_iter", "collect", "vec!", "Vec<"] {
            assert!(
                !body.contains(host_built),
                "sample_device builds a host array per draw (`{host_built}`)"
            );
        }
    }
}
