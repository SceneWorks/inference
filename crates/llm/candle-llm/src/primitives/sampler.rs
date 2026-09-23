//! Token sampling.
//!
//! The Candle port of `mlx-llm`'s sampler (the union of the mlx-gen samplers): temperature + top-p +
//! top-k + repetition/presence penalties, with greedy as the default.
//!
//! **Two implementations, one distribution** (epic sc-24128, story sc-24133):
//!
//! * the **host reference** ([`sample_host`]) — a stabilised, **unnormalised**
//!   `exp((logit - max)/T)` weight per token, heap-based nucleus selection, and a categorical
//!   inverse-CDF draw from the pluggable [`TokenRng`]. It pulls the whole logits row to the host.
//! * the **device sampler** ([`sample_device`], CUDA) — the same weights, with top-k and top-p
//!   found as radix-selected thresholds on the device (no vocabulary sort) and the categorical
//!   draw made on the device from a counter-based SplitMix64 stream bit-identical to the host one.
//!   Only the chosen token id crosses to the host.
//!
//! [`sample`] routes each draw through [`sampler_path`]: greedy is the on-device argmax
//! (unchanged); temperature / top-k / top-p go to the device sampler when the device has one; a
//! constraint mask or a repetition/presence penalty (the backend-neutral policy,
//! [`core_llm::Sampling::sampler_path`]) and a device without the kernel take the host reference.
//! Every decision is counted per thread ([`super::host_sync`]) so the request's
//! [`DecodeRecord`](crate::decode::DecodeRecord) names the path and why — never a silent downgrade.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use candle_core::{DType, Device, Tensor};

pub use core_llm::{HostSampleReason, SamplerPath};

use super::host_sync::{note_host_sync, note_logits_to_host, note_sampler_path};
use crate::error::{Error, Result};

/// A pluggable random source for the categorical draw. Greedy decoding never touches it, so an
/// unused RNG produces bit-identical (deterministic) output.
pub trait TokenRng {
    /// The next sample in `[0, 1)`.
    fn next_f32(&mut self) -> f32;

    /// Reserve the next `n` draws of a **counter-based** stream so an accelerator can compute them
    /// itself, returning the stream state *before* the reservation: draw `i` (0-based) is
    /// SplitMix64's output function of `state + (i + 1) * SplitMix64::INCREMENT`. The generator
    /// advances past the reserved draws. `None` (the default) for a generator that is not
    /// counter-based — the device sampler then draws its uniform here and passes it to the kernel.
    fn reserve_counter_draws(&mut self, n: u64) -> Option<u64> {
        let _ = n;
        None
    }
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

    /// The current state (the value the next draw advances from).
    pub fn state(&self) -> u64 {
        self.0
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

    /// SplitMix64 is counter-based: its state after `n` draws is `state + n * INCREMENT` and each
    /// output is a pure function of that state, so a device can generate the identical stream.
    fn reserve_counter_draws(&mut self, n: u64) -> Option<u64> {
        let base = self.0;
        self.0 = self.0.wrapping_add(n.wrapping_mul(Self::INCREMENT));
        Some(base)
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
    /// constraint mask — eligible for the on-device argmax fast path.
    fn is_plain_greedy(&self) -> bool {
        self.temperature <= 0.0 && self.repetition_penalty == 1.0 && self.presence_penalty == 0.0
    }

    /// The backend-neutral view of these knobs, for the shared routing policy.
    pub fn to_core(&self) -> core_llm::Sampling {
        core_llm::Sampling {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            presence_penalty: self.presence_penalty,
            repetition_penalty: self.repetition_penalty,
            repetition_context: self.repetition_context,
        }
    }
}

/// The largest vocabulary the device sampler serves: its fixed-point weight sums (`w * 2^40`) stay
/// below `2^63` up to `2^23` tokens.
pub const DEVICE_SAMPLER_MAX_VOCAB: usize = 1 << 23;

thread_local! {
    static REFERENCE_SAMPLER: Cell<bool> = const { Cell::new(false) };
}

/// Run `f` with this thread's stochastic draws forced onto the host reference sampler — the
/// pre-sc-24133 behaviour, kept for parity checks and device-vs-host benchmarks. Greedy is
/// unaffected (it was, and stays, the device argmax). Draws made inside report
/// `host` / [`HostSampleReason::Reference`]. Restored on exit, including by unwinding.
pub fn with_reference_sampler<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            REFERENCE_SAMPLER.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(REFERENCE_SAMPLER.with(|c| c.replace(true)));
    f()
}

/// Whether `device` has the device sampler for a `vocab`-wide row: a CUDA device in a
/// `cuda`-feature build. CPU and Metal answer `false` and use the host reference.
pub fn device_sampler_available(device: &Device, vocab: usize) -> bool {
    cfg!(feature = "cuda") && device.is_cuda() && (1..=DEVICE_SAMPLER_MAX_VOCAB).contains(&vocab)
}

/// Where [`sample`] draws for `params` on `device` over a `vocab`-wide row, with or without a
/// constraint mask. The request half of the decision is the backend-neutral policy
/// ([`core_llm::Sampling::sampler_path`]); the device half is [`device_sampler_available`].
pub fn sampler_path(
    device: &Device,
    vocab: usize,
    params: &SamplingParams,
    constrained: bool,
) -> SamplerPath {
    let policy = params.to_core().sampler_path(constrained);
    if policy != SamplerPath::Device {
        return policy;
    }
    if params.temperature <= 0.0 {
        return SamplerPath::Device; // greedy: the on-device argmax, on every device
    }
    if REFERENCE_SAMPLER.with(Cell::get) {
        return SamplerPath::Host(HostSampleReason::Reference);
    }
    if device_sampler_available(device, vocab) {
        SamplerPath::Device
    } else {
        SamplerPath::Host(HostSampleReason::DeviceUnavailable)
    }
}

/// Sample the next token id from `logits`.
///
/// * `logits` — `[vocab]` or `[1, vocab]` for the current (last) position.
/// * `history` — token ids already in the sequence (prompt + generated), for the repetition penalty.
/// * `allowed` — an optional per-vocab mask (e.g. a JSON-constraint grammar); `false` (or
///   out-of-range) entries are forced to `-inf`.
///
/// Routed by [`sampler_path`]: the device path transfers one token id (one host sync) and never
/// the logits; the host path is [`sample_host`].
pub fn sample(
    logits: &Tensor,
    history: &[i32],
    params: &SamplingParams,
    rng: &mut impl TokenRng,
    allowed: Option<&[bool]>,
) -> Result<i32> {
    let path = sampler_path(
        logits.device(),
        logits.elem_count(),
        params,
        allowed.is_some(),
    );
    match path {
        SamplerPath::Device if params.temperature <= 0.0 => {
            note_sampler_path(path);
            argmax_device(logits)
        }
        SamplerPath::Device => {
            let row = logits.flatten_all()?.unsqueeze(0)?;
            let id = sample_device(&row, params, rng)?; // notes the device draw
            note_host_sync(); // the chosen id: a 1-element device->host transfer
            Ok(id.get(0)?.to_scalar::<u32>()? as i32)
        }
        SamplerPath::Host(_) => {
            note_sampler_path(path);
            sample_host(logits, history, params, rng, allowed)
        }
    }
}

/// Draw one token per row of `logits` (`[rows, vocab]`, or `[vocab]` for one row) **without
/// copying the logits to the host**: the result is a `u32` `[rows]` tensor of token ids on the
/// logits' device, and nothing is synchronized — the caller decides when (and whether) to read
/// it. This is the seam speculative decoding samples through.
///
/// Temperature, top-k and top-p only: penalties and constraint masks are host-path features
/// ([`sampler_path`] routes them there). `temperature <= 0` is the per-row argmax and consumes no
/// randomness. Otherwise row `r` uses the `r`-th next draw of `rng`; with a counter-based
/// generator ([`SplitMix64`]) the device computes that uniform itself, bit-identical to
/// [`TokenRng::next_f32`], so a seed reproduces its draws.
///
/// On a device without the kernel ([`device_sampler_available`] is `false`) the same selection
/// rule runs on the host — the rows are copied, and the copy and the host path are counted.
pub fn sample_device(
    logits: &Tensor,
    params: &SamplingParams,
    rng: &mut impl TokenRng,
) -> Result<Tensor> {
    let rows = match logits.rank() {
        1 => logits.unsqueeze(0)?,
        2 => logits.clone(),
        r => {
            return Err(Error::Msg(format!(
                "sample_device: logits must be [vocab] or [rows, vocab], got rank {r}"
            )))
        }
    };
    let (n_rows, vocab) = rows.dims2()?;
    let device = rows.device().clone();
    if params.temperature <= 0.0 {
        for _ in 0..n_rows {
            note_sampler_path(SamplerPath::Device);
        }
        return Ok(rows.argmax(1)?);
    }
    #[cfg(feature = "cuda")]
    if device_sampler_available(&device, vocab) {
        let x = rows.to_dtype(DType::F32)?.contiguous()?;
        let ids = match rng.reserve_counter_draws(n_rows as u64) {
            Some(state) => cuda::sample_rows(&x, params, state, None)?,
            None => {
                let mut out = Vec::with_capacity(n_rows);
                for r in 0..n_rows {
                    let u = rng.next_f32();
                    out.push(cuda::sample_rows(
                        &x.narrow(0, r, 1)?.contiguous()?,
                        params,
                        0,
                        Some(u),
                    )?);
                }
                Tensor::cat(&out, 0)?
            }
        };
        for _ in 0..n_rows {
            note_sampler_path(SamplerPath::Device);
        }
        return Ok(ids);
    }

    // No device kernel: the same index-order selection rule on the host.
    let _ = vocab;
    let host = rows.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    note_host_sync();
    let mut ids = Vec::with_capacity(n_rows);
    for row in &host {
        note_logits_to_host();
        note_sampler_path(SamplerPath::Host(HostSampleReason::DeviceUnavailable));
        ids.push(select_in_index_order(row, params, rng.next_f32()) as u32);
    }
    Ok(Tensor::from_vec(ids, n_rows, &device)?)
}

/// `n` uniforms in `[0, 1)` as an f32 `[n]` tensor on `device`, bit-identical to `n` successive
/// [`TokenRng::next_f32`] calls on `rng` (which advances past them). On CUDA with a counter-based
/// generator they are generated on the device — no transfer. The uniform source for a
/// speculative acceptance test run on the device.
pub fn uniform_device(rng: &mut impl TokenRng, n: usize, device: &Device) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        if let Some(state) = rng.reserve_counter_draws(n as u64) {
            return cuda::uniform(state, n, device);
        }
    }
    let host: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    Ok(Tensor::from_vec(host, n, device)?)
}

/// The device sampler's selection rule on the host: the reference's shaped weights, walked in
/// **index order** for the inverse-CDF draw (the same distribution as [`sample_host`]'s walk;
/// the device kernel walks in index order too, so a seeded draw lands on the same token up to
/// rounding). Any NaN, or nothing left, falls back to the argmax as the reference does.
fn select_in_index_order(v: &[f32], params: &SamplingParams, u: f32) -> i32 {
    if v.iter().any(|x| x.is_nan()) {
        return argmax_host(v);
    }
    let mut weights = nucleus_weights(v, params);
    if weights.is_empty() {
        return argmax_host(v);
    }
    weights.sort_unstable_by_key(|x| x.0);
    let total: f64 = weights.iter().map(|x| f64::from(x.1)).sum();
    let target = f64::from(u) * total;
    let mut cum = 0.0f64;
    for (i, w) in &weights {
        cum += f64::from(*w);
        if *w > 0.0 && cum > target {
            return *i as i32;
        }
    }
    argmax_host(v)
}

/// The host reference sampler: pull the logits row to host f32, apply the constraint mask and the
/// penalties, then temperature + top-k + top-p and a categorical inverse-CDF draw from `rng` — the
/// distribution [`sample_device`] reproduces. Plain greedy (no penalty, no mask) still takes the
/// on-device argmax.
pub fn sample_host(
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
///
/// A host read of the whole row: counted as a host sampling decision with
/// [`HostSampleReason::SpeculativeDistribution`].
pub fn shaped_candidates(
    logits: &Tensor,
    history: &[i32],
    params: &SamplingParams,
    allowed: Option<&[bool]>,
) -> Result<Vec<(i32, f32)>> {
    note_sampler_path(SamplerPath::Host(HostSampleReason::SpeculativeDistribution));
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
    note_host_sync(); // whole-vocab device->host transfer
    note_logits_to_host();
    let mut v: Vec<f32> = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;

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
    Ok(v)
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
    note_host_sync(); // 1-element device->host transfer
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
/// `top_p * total`, always keeping at least one. Ties break to lower index. The mass is accumulated
/// in f64 (sc-24133): an f32 running sum over a wide vocabulary drifts by more than a tail token's
/// weight, which moved the nucleus boundary by rounding alone (on a 5000-token row the f32 sum
/// crossed the threshold one token early, 5e-6 of the mass before the exact crossing). The device
/// sampler sums in exact fixed point.
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

/// The CUDA device sampler: NVRTC-compiled kernels (`sampler_cuda.cu`), one block per row.
/// Why a kernel: at the pinned Candle rev, `arg_sort_last_dim` sorts a row inside one block's
/// shared memory (a 248k vocabulary does not fit) and `cumsum` is a triangular matmul
/// (quadratic in the vocabulary), so top-k / top-p cannot be expressed with Candle ops.
#[cfg(feature = "cuda")]
mod cuda {
    use super::SamplingParams;
    use crate::error::Result;
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::WrapErr;
    use candle_core::op::BackpropOp;
    use candle_core::{CpuStorage, CudaStorage, CustomOp1, Device, Layout, Shape, Storage, Tensor};

    const SOURCE: &str = include_str!("sampler_cuda.cu");
    const MODULE: &str = "candle_llm_sampler_v1";
    /// Must equal `SAMPLER_THREADS` in the kernel source (block reductions assume 32 full warps).
    const THREADS: u32 = 1024;
    static PTX: std::sync::OnceLock<std::result::Result<String, String>> =
        std::sync::OnceLock::new();

    fn ptx() -> candle_core::Result<&'static str> {
        match PTX.get_or_init(|| {
            candle_core::cuda_backend::cudarc::nvrtc::compile_ptx(SOURCE)
                .map(|ptx| ptx.to_src())
                .map_err(|e| format!("sampler CUDA nvrtc compile failed: {e}"))
        }) {
            Ok(ptx) => Ok(ptx),
            Err(error) => candle_core::bail!("{error}"),
        }
    }

    struct SampleRows {
        inv_t: f32,
        top_k: u32,
        top_p: f32,
        state: u64,
        /// `>= 0` replaces the counter-derived uniform (single-row, non-counter RNG).
        u_override: f32,
    }

    impl CustomOp1 for SampleRows {
        fn name(&self) -> &'static str {
            "candle-llm-sample-rows"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("device sampler invoked on CPU")
        }

        fn cuda_fwd(
            &self,
            storage: &CudaStorage,
            layout: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let dev = storage.device().clone();
            let (rows, vocab) = layout.shape().dims2()?;
            let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
                candle_core::Error::Msg("device sampler input must be contiguous".into())
            })?;
            let x = storage.as_cuda_slice::<f32>()?.slice(start..end);
            let mut out = unsafe { dev.alloc::<u32>(rows) }?;
            let function =
                dev.get_or_load_custom_func("candle_llm_sample_rows_f32", MODULE, ptx()?)?;
            let vocab = vocab as u32;
            let config = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = function.builder();
            builder
                .arg(&x)
                .arg(&vocab)
                .arg(&self.inv_t)
                .arg(&self.top_k);
            builder
                .arg(&self.top_p)
                .arg(&self.state)
                .arg(&self.u_override)
                .arg(&mut out);
            unsafe { builder.launch(config) }.w()?;
            Ok((CudaStorage::wrap_cuda_slice(out, dev), Shape::from(rows)))
        }
    }

    /// Token ids `u32 [rows]` for contiguous f32 `x: [rows, vocab]` on a CUDA device.
    pub(super) fn sample_rows(
        x: &Tensor,
        params: &SamplingParams,
        state: u64,
        u_override: Option<f32>,
    ) -> Result<Tensor> {
        let op = SampleRows {
            inv_t: 1.0 / params.temperature,
            top_k: u32::try_from(params.top_k).unwrap_or(u32::MAX),
            top_p: params.top_p,
            state,
            u_override: u_override.unwrap_or(-1.0),
        };
        Ok(x.apply_op1_no_bwd(&op)?)
    }

    /// `n` SplitMix64 uniforms from `state`, generated on `device`.
    pub(super) fn uniform(state: u64, n: usize, device: &Device) -> Result<Tensor> {
        let dev = device.as_cuda_device()?.clone();
        let mut out = unsafe { dev.alloc::<f32>(n) }?;
        if n > 0 {
            let function =
                dev.get_or_load_custom_func("candle_llm_splitmix_uniform_f32", MODULE, ptx()?)?;
            let count = n as u32;
            let mut builder = function.builder();
            builder.arg(&state).arg(&count).arg(&mut out);
            unsafe { builder.launch(LaunchConfig::for_num_elems(count)) }.w()?;
        }
        let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev));
        Ok(Tensor::from_storage(storage, n, BackpropOp::none(), false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn logits(v: &[f32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), (1, v.len()), &Device::Cpu).unwrap()
    }

    /// SplitMix64's output function on an already-advanced state — what the device kernel computes.
    fn splitmix_output(mut z: u64) -> f32 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32) / ((1u64 << 24) as f32)
    }

    #[test]
    fn reserved_counter_draws_are_the_next_f32_stream() {
        let mut reserved = SplitMix64::new(77);
        let mut stepped = SplitMix64::new(77);
        let base = reserved.reserve_counter_draws(5).unwrap();
        for i in 0..5u64 {
            let device_side =
                splitmix_output(base.wrapping_add((i + 1).wrapping_mul(SplitMix64::INCREMENT)));
            assert_eq!(device_side, stepped.next_f32(), "draw {i}");
        }
        assert_eq!(
            reserved.state(),
            stepped.state(),
            "advanced past the reservation"
        );
        assert_eq!(reserved.next_f32(), stepped.next_f32());
    }

    #[test]
    fn plain_greedy_is_one_device_draw_and_one_sync() {
        let span_syncs = crate::primitives::host_sync_count();
        let before = crate::primitives::sampler_counters();
        let mut rng = SplitMix64::new(0);
        let l = logits(&[0.1, 5.0, 0.2]);
        assert_eq!(
            sample(&l, &[], &SamplingParams::default(), &mut rng, None).unwrap(),
            1
        );
        let after = crate::primitives::sampler_counters();
        assert_eq!(crate::primitives::host_sync_count() - span_syncs, 1);
        assert_eq!(after.device_draws - before.device_draws, 1);
        assert_eq!(after.logits_to_host, before.logits_to_host);
        assert_eq!(rng.state(), 0, "greedy consumes no randomness");
    }

    #[test]
    fn sample_device_greedy_is_the_row_argmax_without_randomness() {
        let rows =
            Tensor::from_vec(vec![0.0f32, 3.0, 1.0, 9.0, 2.0, 2.0], (2, 3), &Device::Cpu).unwrap();
        let mut rng = SplitMix64::new(4);
        let ids = sample_device(&rows, &SamplingParams::default(), &mut rng).unwrap();
        assert_eq!(ids.to_vec1::<u32>().unwrap(), vec![1, 0]);
        assert_eq!(rng.state(), 4);
    }

    #[test]
    fn reference_override_is_scoped_and_labelled() {
        let params = SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            ..Default::default()
        };
        let path = with_reference_sampler(|| sampler_path(&Device::Cpu, 8, &params, false));
        assert_eq!(path, SamplerPath::Host(HostSampleReason::Reference));
        assert_eq!(
            sampler_path(&Device::Cpu, 8, &params, false),
            SamplerPath::Host(HostSampleReason::DeviceUnavailable),
            "restored after the scope"
        );
        let greedy = with_reference_sampler(|| {
            sampler_path(&Device::Cpu, 8, &SamplingParams::default(), false)
        });
        assert_eq!(greedy, SamplerPath::Device, "greedy is unaffected");
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
