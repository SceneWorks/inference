//! SC-20677 minimal compressed-domain KV candidate adapters.
//!
//! Epic 20669 compares three representations at matched physical bytes and matched quality
//! before any one of them is productized: the existing SC-20675/SC-20676 packed group-affine
//! cache, a two-stage TurboQuant-style packed RVQ ([`rvq`]), and asymmetric RaBitQ with packed
//! one-bit keys and nibble-packed four-bit values ([`rabitq`]). This module owns only what a fair
//! comparison needs: encode/append, one compressed-domain attention dispatch per representation,
//! exact physical byte accounting, the SC-20674 pre-mutation route/fallback vocabulary, and an
//! independent fp32 dequantize-then-attend oracle. Decoder routing, paging, serialization and the
//! rest of the SC-20674 lifecycle belong to whichever candidate the ADR selects.
//!
//! Both new representations work in a randomized-Hadamard rotated domain. The query is rotated
//! once (`[B,Hq,S_q,D]`), the Metal kernel reconstructs one key/value element at a time in
//! registers inside an online softmax, and the rotated output (`[B,Hq,S_q,D]`) is rotated back
//! once. No dense historical K/V and no `S_q x S_kv` score tensor exists at any point; the
//! [`probe_materialization`] guard measures that physically from the MLX allocator.
//!
//! Mechanisms are ported from VeloxQuant-MLX v0.65.0 (commit
//! [`VELOXQUANT_COMMIT`], MIT, see the crate `NOTICE`); no upstream performance or quality claim
//! is inherited.

pub mod compare;
pub mod rabitq;
pub mod rvq;

use std::f64::consts::PI;

use mlx_rs::{Array, Dtype};
use serde::Serialize;

use crate::error::{Error, Result};
use crate::primitives::kv_cache::{CacheRoute, PackedAttentionMask};
use crate::primitives::packed_group_affine_kv::DenseFallbackEvent;
use crate::primitives::sampler::SplitMix64;

/// Frozen upstream source the mechanisms were ported from (SC-20672 audit).
pub const VELOXQUANT_COMMIT: &str = "54989ee223611627592f7f9bd925e924658f1f22";
pub const VELOXQUANT_TAG: &str = "v0.65.0";

/// Head dimensions the candidate kernels accept: a power of two for the Walsh-Hadamard rotation,
/// at most eight register slots per lane, and the same qualified set as the SC-20676 reader.
pub fn candidate_head_dimension_supported(head_dim: usize) -> bool {
    matches!(head_dim, 64 | 128 | 256)
}

/// Attention request presented to a candidate before any dispatch. `backend` must be
/// `"mlx-metal"`; everything else is a dense fallback with a stable reason.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CandidateAttentionRequest {
    pub backend: String,
    pub batch: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub query_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
    #[serde(serialize_with = "serialize_mask")]
    pub mask: PackedAttentionMask,
    pub scale: f32,
}

fn serialize_mask<S: serde::Serializer>(
    mask: &PackedAttentionMask,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    serializer.serialize_str(&mask_label(*mask))
}

pub fn mask_label(mask: PackedAttentionMask) -> String {
    match mask {
        PackedAttentionMask::None => "none".into(),
        PackedAttentionMask::Causal => "causal".into(),
        PackedAttentionMask::SlidingWindow(window) => format!("window:{window}"),
        PackedAttentionMask::Additive => "additive".into(),
    }
}

/// Stable, observable reasons a candidate declines the compressed route before mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateFallbackReason {
    UnsupportedBackend,
    EmptyQuery,
    EmptyCache,
    QueryLongerThanCache,
    UnsupportedHeadDimension,
    UnsupportedGqaRatio,
    GeometryMismatch,
    AdditiveMask,
    EmptySlidingWindow,
    InvalidScale,
    /// The SC-20676 group-affine reader hard-codes `rsqrt(D)`; any other scale is dense.
    UnsupportedScale,
}

impl CandidateFallbackReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedBackend => "backend is not mlx-metal",
            Self::EmptyQuery => "empty query",
            Self::EmptyCache => "empty compressed cache",
            Self::QueryLongerThanCache => "query length exceeds cached length",
            Self::UnsupportedHeadDimension => "head dimension must be 64, 128, or 256",
            Self::UnsupportedGqaRatio => "query heads are not a multiple of KV heads",
            Self::GeometryMismatch => {
                "request batch/KV-head/head-dimension/length differs from cache"
            }
            Self::AdditiveMask => "additive mask requires dense fallback",
            Self::EmptySlidingWindow => "empty sliding window",
            Self::InvalidScale => "attention scale must be finite and positive",
            Self::UnsupportedScale => "reader supports only scale = rsqrt(head_dim)",
        }
    }
}

/// Shared capability check for the rotated-domain candidates. Returns the first failing reason.
pub fn candidate_request_support(
    request: &CandidateAttentionRequest,
    cache: CacheGeometry,
) -> std::result::Result<(), CandidateFallbackReason> {
    use CandidateFallbackReason as R;
    if request.backend != "mlx-metal" {
        return Err(R::UnsupportedBackend);
    }
    if request.query_len == 0 {
        return Err(R::EmptyQuery);
    }
    if cache.logical_len == 0 {
        return Err(R::EmptyCache);
    }
    if !candidate_head_dimension_supported(request.head_dim) {
        return Err(R::UnsupportedHeadDimension);
    }
    if request.kv_heads == 0 || !request.query_heads.is_multiple_of(request.kv_heads) {
        return Err(R::UnsupportedGqaRatio);
    }
    if request.batch != cache.batch
        || request.kv_heads != cache.kv_heads
        || request.head_dim != cache.head_dim
        || request.kv_len != cache.logical_len
    {
        return Err(R::GeometryMismatch);
    }
    if request.query_len > cache.logical_len {
        return Err(R::QueryLongerThanCache);
    }
    match request.mask {
        PackedAttentionMask::Additive => return Err(R::AdditiveMask),
        PackedAttentionMask::SlidingWindow(0) => return Err(R::EmptySlidingWindow),
        _ => {}
    }
    if !request.scale.is_finite() || request.scale <= 0.0 {
        return Err(R::InvalidScale);
    }
    Ok(())
}

/// Record a declined route exactly as the SC-20675 cache does: the event names the operation and
/// reason and the cache state is untouched.
pub fn decline(
    events: &mut Vec<DenseFallbackEvent>,
    operation: &str,
    reason: CandidateFallbackReason,
    logical_len: usize,
    allocated_bytes: usize,
) -> CacheRoute {
    events.push(DenseFallbackEvent {
        operation: operation.into(),
        reason: reason.as_str().into(),
        logical_len,
        allocated_bytes,
    });
    CacheRoute::DenseFallback {
        reason: reason.as_str().into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheGeometry {
    pub batch: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub logical_len: usize,
}

/// Exact physical storage of one candidate. Every field is derived from the live buffers, never
/// from a bit-width formula; tests assert the closed-form per-token cost against these.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CandidateRepresentation {
    pub family: String,
    pub config: String,
    pub identity: String,
    pub version: u32,
    pub batch: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub logical_len: usize,
    pub layout: String,
    pub key_code_bytes: usize,
    pub key_metadata_bytes: usize,
    pub value_code_bytes: usize,
    pub value_metadata_bytes: usize,
    /// Unquantized staging retained by the representation (group-affine's incomplete f32 key
    /// token group); zero for the rotated candidates.
    pub dense_staging_bytes: usize,
    /// Rotation signs and codebooks shared by every token of the cache.
    pub shared_constant_bytes: usize,
    /// Codes + metadata + staging + shared constants: the representation compared at matched
    /// budgets.
    pub representation_bytes: usize,
    /// Host staging vector capacity (a POC upload buffer, reported separately).
    pub host_allocated_bytes: usize,
    /// Live MLX array payload (`Array::nbytes`) of the device-resident representation.
    pub device_bytes: usize,
}

impl CandidateRepresentation {
    pub fn payload_bytes(&self) -> usize {
        self.key_code_bytes
            + self.key_metadata_bytes
            + self.value_code_bytes
            + self.value_metadata_bytes
            + self.dense_staging_bytes
    }
}

/// Dispatch shape of one candidate's attention call, reported beside its timing so a reader can
/// see what the numbers compare.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KernelProfile {
    pub kernel: String,
    pub threads_per_threadgroup: usize,
    pub simd_groups: usize,
    pub gpu_family: String,
    pub threadgroup_barriers_per_kv_token: usize,
    /// Where the online-softmax running max/normalizer live.
    pub softmax_state: String,
    /// MLX ops dispatched around the kernel on every attend (query/output rotation and casts).
    pub extra_mlx_ops_per_attend: usize,
    /// Whether `attend` itself performs host-side argument staging before the kernel.
    pub host_staging_inside_attend: bool,
}

/// The rotated candidates share one geometry: a single 32-lane SIMD group per query row with
/// softmax state in registers (`simd_sum` only), plus `astype·multiply·hadamard` on the query and
/// `hadamard·multiply·astype` on the output.
pub(crate) fn rotated_kernel_profile(kernel: &str) -> KernelProfile {
    KernelProfile {
        kernel: kernel.into(),
        threads_per_threadgroup: 32,
        simd_groups: 1,
        gpu_family: "any-apple (fixed one-SIMD-group geometry)".into(),
        threadgroup_barriers_per_kv_token: 0,
        softmax_state: "registers (simd_sum)".into(),
        extra_mlx_ops_per_attend: 6,
        host_staging_inside_attend: false,
    }
}

/// A minimal comparable compressed KV representation for one attention layer.
pub trait CompressedKvCandidate {
    fn family(&self) -> &'static str;
    fn config(&self) -> String;
    fn representation(&self) -> CandidateRepresentation;
    /// Decide the route before any mutation or dispatch; declining records a fallback event.
    fn preflight(&mut self, request: &CandidateAttentionRequest) -> CacheRoute;
    /// Append `[B,Hkv,step,D]` row-major keys and values.
    fn append(&mut self, keys: &[f32], values: &[f32], step: usize) -> Result<()>;
    fn trim(&mut self, len: usize) -> Result<()>;
    /// Upload/refresh the device-resident representation. Separated from [`Self::attend`] so
    /// attend timing and transient memory exclude staging.
    fn sync_device(&mut self) -> Result<()>;
    /// Compressed-domain attention. Callers must have passed [`Self::preflight`].
    fn attend(&mut self, query: &Array, request: &CandidateAttentionRequest) -> Result<Array>;
    /// The same attention with any per-dispatch host staging hoisted out, when the reader has
    /// such staging and it is separable. `None` means [`Self::attend`] already excludes it.
    fn attend_excluding_staging(
        &mut self,
        _query: &Array,
        _request: &CandidateAttentionRequest,
    ) -> Option<Result<Array>> {
        None
    }
    fn kernel_profile(&self) -> KernelProfile;
    /// Oracle-only full reconstruction `[B,Hkv,S,D]` in the original domain. Counted.
    fn dequantize_dense_for_oracle(&mut self) -> Result<(Vec<f32>, Vec<f32>)>;
    /// The query the representation's score estimator effectively uses (identity unless the
    /// method also quantizes the query).
    fn oracle_query(
        &self,
        query: &[f32],
        _request: &CandidateAttentionRequest,
    ) -> Result<Vec<f32>> {
        Ok(query.to_vec())
    }
    fn full_cache_dequantizations(&self) -> usize;
    fn fallback_events(&self) -> &[DenseFallbackEvent];
}

// ---------------------------------------------------------------------------------------------
// Randomized Hadamard rotation.
// ---------------------------------------------------------------------------------------------

/// `y = H (s ⊙ x) / sqrt(D)` with Sylvester-order `H` and deterministic ±1 signs `s`.
#[derive(Clone, Debug, PartialEq)]
pub struct HadamardRotation {
    signs: Vec<f32>,
}

impl HadamardRotation {
    pub fn new(head_dim: usize, seed: u64) -> Result<Self> {
        if !head_dim.is_power_of_two() {
            return Err(Error::Unsupported(
                "Hadamard rotation requires a power-of-two head dimension".into(),
            ));
        }
        let mut rng = SplitMix64::new(seed);
        Ok(Self {
            signs: (0..head_dim)
                .map(|_| if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 })
                .collect(),
        })
    }

    pub fn signs(&self) -> &[f32] {
        &self.signs
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y: Vec<f32> = x.iter().zip(&self.signs).map(|(x, s)| x * s).collect();
        fwht_normalized(&mut y);
        y
    }

    pub fn inverse(&self, y: &[f32]) -> Vec<f32> {
        let mut x = y.to_vec();
        fwht_normalized(&mut x);
        x.iter_mut().zip(&self.signs).for_each(|(x, s)| *x *= s);
        x
    }

    /// Rotate the last axis of an MLX array (query side).
    pub fn forward_mlx(&self, x: &Array, signs: &Array) -> Result<Array> {
        let scale = 1.0 / (self.signs.len() as f32).sqrt();
        Ok(x.as_dtype(Dtype::Float32)?
            .multiply(signs)?
            .hadamard_transform(Some(scale))?)
    }

    /// Inverse-rotate the last axis of an MLX array (output side).
    pub fn inverse_mlx(&self, y: &Array, signs: &Array) -> Result<Array> {
        let scale = 1.0 / (self.signs.len() as f32).sqrt();
        Ok(y.hadamard_transform(Some(scale))?.multiply(signs)?)
    }
}

fn fwht_normalized(values: &mut [f32]) {
    let n = values.len();
    let mut half = 1;
    while half < n {
        for block in (0..n).step_by(half * 2) {
            for i in block..block + half {
                let (a, b) = (values[i], values[i + half]);
                values[i] = a + b;
                values[i + half] = a - b;
            }
        }
        half *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    values.iter_mut().for_each(|v| *v *= scale);
}

// ---------------------------------------------------------------------------------------------
// Lloyd-Max scalar codebooks (VeloxQuant `math/lloyd_max.py`, `codebooks/strategies.py`).
// ---------------------------------------------------------------------------------------------

/// Sorted scalar codebook with midpoint decision boundaries.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarCodebook {
    centroids: Vec<f32>,
    boundaries: Vec<f32>,
}

impl ScalarCodebook {
    pub fn from_centroids(mut centroids: Vec<f32>) -> Self {
        centroids.sort_by(f32::total_cmp);
        let boundaries = centroids.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
        Self {
            centroids,
            boundaries,
        }
    }

    /// Lloyd-Max levels for `N(0, 1/d)` over `±6σ` (upstream `lloyd_max_gaussian`).
    pub fn gaussian(bits: u32, head_dim: usize) -> Self {
        let sigma = 1.0 / (head_dim as f64).sqrt();
        let norm = 1.0 / (sigma * (2.0 * PI).sqrt());
        Self::from_centroids(lloyd_max(
            |x| norm * (-(x * x) / (2.0 * sigma * sigma)).exp(),
            6.0 * sigma,
            1 << bits,
        ))
    }

    /// Lloyd-Max levels for the Laplacian stage-2 residual of TurboQuant RVQ, with the upstream
    /// default scale `sqrt(1/d) * sqrt(3π)/2 * 4^-b / sqrt(2)` over `±8·scale`.
    pub fn laplacian_residual(bits: u32, head_dim: usize) -> Self {
        let sigma_q =
            (1.0 / head_dim as f64).sqrt() * ((3.0 * PI).sqrt() / 2.0) * 4f64.powi(-(bits as i32));
        let scale = (sigma_q / 2f64.sqrt()).max(1e-6);
        Self::from_centroids(lloyd_max(
            |x| (-x.abs() / scale).exp() / (2.0 * scale),
            8.0 * scale,
            1 << bits,
        ))
    }

    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    pub fn quantize(&self, x: f32) -> u32 {
        self.boundaries.partition_point(|b| *b < x) as u32
    }

    pub fn value(&self, code: u32) -> f32 {
        self.centroids[code as usize]
    }
}

fn lloyd_max(pdf: impl Fn(f64) -> f64, half_support: f64, levels: usize) -> Vec<f32> {
    const QUAD: usize = 10_000;
    const ITERATIONS: usize = 500;
    const TOLERANCE: f64 = 1e-9;
    let lo = -half_support;
    let dx = 2.0 * half_support / (QUAD - 1) as f64;
    let grid: Vec<(f64, f64)> = (0..QUAD)
        .map(|i| {
            let x = lo + dx * i as f64;
            (x, pdf(x))
        })
        .collect();
    let mut centroids: Vec<f64> = (0..levels)
        .map(|i| lo + 2.0 * half_support * i as f64 / (levels.max(2) - 1) as f64)
        .collect();
    for _ in 0..ITERATIONS {
        let boundaries: Vec<f64> = centroids.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
        let mut mass = vec![0.0f64; levels];
        let mut moment = vec![0.0f64; levels];
        // Trapezoid integration within each Voronoi cell; a grid segment that straddles a
        // decision boundary is split there so symmetric densities stay exactly symmetric.
        let mut accumulate = |cell: usize, (x0, p0): (f64, f64), (x1, p1): (f64, f64)| {
            let width = x1 - x0;
            mass[cell] += (p0 + p1) * 0.5 * width;
            moment[cell] += (x0 * p0 + x1 * p1) * 0.5 * width;
        };
        let mut cell = 0;
        for pair in grid.windows(2) {
            let mut start = pair[0];
            while cell < boundaries.len() && boundaries[cell] <= start.0 {
                cell += 1;
            }
            while cell < boundaries.len() && boundaries[cell] < pair[1].0 {
                let split = (boundaries[cell], pdf(boundaries[cell]));
                accumulate(cell, start, split);
                start = split;
                cell += 1;
            }
            accumulate(cell, start, pair[1]);
        }
        let mut shift = 0.0f64;
        for i in 0..levels {
            if mass[i] >= 1e-12 {
                let next = moment[i] / mass[i];
                shift = shift.max((next - centroids[i]).abs());
                centroids[i] = next;
            }
        }
        if shift < TOLERANCE {
            break;
        }
    }
    centroids.into_iter().map(|c| c as f32).collect()
}

// ---------------------------------------------------------------------------------------------
// Oracle and physical dense-materialization guard.
// ---------------------------------------------------------------------------------------------

/// Independent fp32 attention over dense `[B,Hkv,S,D]` K/V (SC-20676's reference semantics).
pub fn dense_attention_oracle(
    request: &CandidateAttentionRequest,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
) -> Result<Vec<f32>> {
    use crate::primitives::packed_attention::{attention_f32_masked, PackedAttentionShape};
    let shape = PackedAttentionShape {
        batch: request.batch,
        query_heads: request.query_heads,
        kv_heads: request.kv_heads,
        query_len: request.query_len,
        kv_len: request.kv_len,
        head_dim: request.head_dim,
    };
    let expected = request.batch * request.kv_heads * request.kv_len * request.head_dim;
    if keys.len() != expected || values.len() != expected {
        return Err(Error::Config("oracle K/V shape mismatch".into()));
    }
    let (heads, len, dim) = (request.kv_heads, request.kv_len, request.head_dim);
    let at = |b: usize, h: usize, s: usize, d: usize| ((b * heads + h) * len + s) * dim + d;
    attention_f32_masked(
        shape,
        query,
        |b, h, s, d| keys[at(b, h, s, d)],
        |b, h, s, d| values[at(b, h, s, d)],
        request.scale,
        request.mask,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MaterializationVerdict {
    /// At least one check was discriminating for this geometry and none tripped.
    Compressed,
    /// Transient allocation reached a dense K (or V) reconstruction or an `S_q x S_kv` score tensor.
    DenseMaterialized,
    /// The geometry is too small for either threshold to exceed the allowed query/output workspace.
    Inconclusive,
}

/// Allocator-measured transient memory of one attention call against the two dense shapes
/// the compressed route must never allocate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct MaterializationProbe {
    pub transient_bytes: u64,
    /// One of dense K or dense V at fp16 (`B·Hkv·S·D·2`).
    pub dense_reconstruction_threshold_bytes: u64,
    /// Score tensor at fp16 (`B·Hq·S_q·S·2`).
    pub score_matrix_threshold_bytes: u64,
    /// Query/output-sized workspace the rotated route legitimately allocates, plus page slack.
    pub allowed_workspace_bytes: u64,
    pub verdict: MaterializationVerdict,
}

impl MaterializationProbe {
    pub fn classify(request: &CandidateAttentionRequest, transient_bytes: u64) -> Self {
        let dense =
            (request.batch * request.kv_heads * request.kv_len * request.head_dim * 2) as u64;
        let score =
            (request.batch * request.query_heads * request.query_len * request.kv_len * 2) as u64;
        let query_f32 =
            (request.batch * request.query_heads * request.query_len * request.head_dim * 4) as u64;
        let allowed = 16 * query_f32 + 256 * 1024;
        let checks =
            [dense, score].map(|threshold| (threshold > allowed, transient_bytes >= threshold));
        let verdict = if checks
            .iter()
            .any(|(discriminating, tripped)| *discriminating && *tripped)
        {
            MaterializationVerdict::DenseMaterialized
        } else if checks.iter().any(|(discriminating, _)| *discriminating) {
            MaterializationVerdict::Compressed
        } else {
            MaterializationVerdict::Inconclusive
        };
        Self {
            transient_bytes,
            dense_reconstruction_threshold_bytes: dense,
            score_matrix_threshold_bytes: score,
            allowed_workspace_bytes: allowed,
            verdict,
        }
    }
}

/// Run one attention call and measure its transient allocation high-water mark from MLX.
/// Inputs must already be evaluated so their residency is part of the baseline.
pub fn probe_materialization(
    request: &CandidateAttentionRequest,
    attend: impl FnOnce() -> Result<Array>,
) -> Result<(Array, MaterializationProbe)> {
    mlx_rs::memory::reset_peak_memory();
    let before = mlx_rs::memory::get_active_memory() as u64;
    let output = attend()?;
    output.eval()?;
    let peak = mlx_rs::memory::get_peak_memory() as u64;
    Ok((
        output,
        MaterializationProbe::classify(request, peak.saturating_sub(before)),
    ))
}

/// Pack `bits`-wide codes LSB-first into `u32` words (upstream `_pack_indices` layout).
pub(crate) fn pack_words(codes: &[u32], bits: u32, out: &mut Vec<u32>) {
    let per_word = (32 / bits) as usize;
    for chunk in codes.chunks(per_word) {
        let word = chunk
            .iter()
            .enumerate()
            .fold(0u32, |word, (i, code)| word | (code << (i as u32 * bits)));
        out.push(word);
    }
}

pub(crate) fn unpack_word_code(words: &[u32], index: usize, bits: u32) -> u32 {
    let per_word = (32 / bits) as usize;
    (words[index / per_word] >> ((index % per_word) as u32 * bits)) & ((1 << bits) - 1)
}

pub(crate) fn mlx_dims(dims: &[usize]) -> Result<Vec<i32>> {
    dims.iter()
        .map(|d| {
            i32::try_from(*d)
                .map_err(|_| Error::Unsupported("candidate dimension exceeds MLX i32".into()))
        })
        .collect()
}

pub(crate) fn validate_step(
    geometry: CacheGeometry,
    keys: &[f32],
    values: &[f32],
    step: usize,
) -> Result<()> {
    let expected = geometry.batch * geometry.kv_heads * step * geometry.head_dim;
    if step == 0 || keys.len() != expected || values.len() != expected {
        return Err(Error::Config("candidate append shape mismatch".into()));
    }
    if keys.iter().chain(values).any(|v| !v.is_finite()) {
        return Err(Error::Config("candidate append requires finite K/V".into()));
    }
    i32::try_from(geometry.logical_len + step)
        .map_err(|_| Error::Config("candidate length exceeds MLX i32".into()))?;
    Ok(())
}

/// Iterate `[B,Hkv,step,D]` input in token-major order: `(token, row, slice)`.
pub(crate) fn token_major_rows<'a>(
    geometry: CacheGeometry,
    data: &'a [f32],
    step: usize,
) -> impl Iterator<Item = &'a [f32]> + 'a {
    let rows = geometry.batch * geometry.kv_heads;
    let dim = geometry.head_dim;
    (0..step).flat_map(move |token| {
        (0..rows).map(move |row| {
            let start = (row * step + token) * dim;
            &data[start..start + dim]
        })
    })
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub fn gaussian(seed: u64, len: usize) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        let mut uniform = move || ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        (0..len)
            .map(|_| ((-2.0 * uniform().ln()).sqrt() * (2.0 * PI * uniform()).cos()) as f32)
            .collect()
    }

    pub fn request(
        batch: usize,
        query_heads: usize,
        kv_heads: usize,
        query_len: usize,
        kv_len: usize,
        head_dim: usize,
    ) -> CandidateAttentionRequest {
        CandidateAttentionRequest {
            backend: "mlx-metal".into(),
            batch,
            query_heads,
            kv_heads,
            query_len,
            kv_len,
            head_dim,
            mask: PackedAttentionMask::Causal,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn hadamard_rotation_is_orthonormal_and_invertible() {
        let rotation = HadamardRotation::new(64, 7).unwrap();
        let x = gaussian(1, 64);
        let y = rotation.forward(&x);
        let norm = |v: &[f32]| v.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm(&x) - norm(&y)).abs() < 1e-4);
        assert!(max_abs_diff(&rotation.inverse(&y), &x) < 1e-5);
        assert!(HadamardRotation::new(96, 7).is_err());
    }

    #[test]
    fn lloyd_max_codebooks_are_symmetric_and_sorted() {
        for bits in 1..=4 {
            for book in [
                ScalarCodebook::gaussian(bits, 128),
                ScalarCodebook::laplacian_residual(bits, 128),
            ] {
                let c = book.centroids();
                assert_eq!(c.len(), 1 << bits);
                assert!(c.windows(2).all(|w| w[0] < w[1]));
                for i in 0..c.len() {
                    assert!((c[i] + c[c.len() - 1 - i]).abs() < 1e-5, "{c:?}");
                }
            }
        }
        // b=1 Gaussian Lloyd-Max is ±σ·sqrt(2/π).
        let one = ScalarCodebook::gaussian(1, 64);
        let expected = (2.0 / PI).sqrt() as f32 / 8.0;
        assert!((one.centroids()[1] - expected).abs() < 1e-4);
        assert_eq!(one.quantize(-1.0), 0);
        assert_eq!(one.quantize(1.0), 1);
    }

    #[test]
    fn word_packing_round_trips_every_bit_width() {
        for bits in 1..=4u32 {
            let codes: Vec<u32> = (0..67u32).map(|i| (i * 7 + 3) % (1 << bits)).collect();
            let mut words = Vec::new();
            pack_words(&codes, bits, &mut words);
            assert_eq!(words.len(), 67usize.div_ceil((32 / bits) as usize));
            for (i, code) in codes.iter().enumerate() {
                assert_eq!(unpack_word_code(&words, i, bits), *code);
            }
        }
    }

    #[test]
    fn materialization_classifier_trips_on_dense_or_score_and_refuses_tiny_geometry() {
        let decode = request(1, 4, 2, 1, 4096, 128);
        let dense = MaterializationProbe::classify(&decode, 0).dense_reconstruction_threshold_bytes;
        assert_eq!(dense, 4096 * 2 * 128 * 2);
        assert_eq!(
            MaterializationProbe::classify(&decode, 64 * 1024).verdict,
            MaterializationVerdict::Compressed
        );
        assert_eq!(
            MaterializationProbe::classify(&decode, dense).verdict,
            MaterializationVerdict::DenseMaterialized
        );
        let tiny = request(1, 1, 1, 1, 4, 64);
        assert_eq!(
            MaterializationProbe::classify(&tiny, u64::MAX).verdict,
            MaterializationVerdict::Inconclusive
        );
    }
}
