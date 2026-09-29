//! SC-20677 matched-budget comparison harness.
//!
//! Runs the existing group-affine reader (SC-20675/SC-20676) and every [`super::rvq`] and
//! [`super::rabitq`] configuration over the same K/V and query, then emits one JSON report with
//! provenance, exact resident bytes, attention error against an exact fp32 attention over the
//! input K/V, implementation parity against an independent fp32 dequantize-then-attend oracle,
//! allocator-measured transient memory, timing, and the matched-budget / matched-quality picks.
//!
//! Real captured K/V is a one-command step once a capture exists (safetensors with `q`
//! `[B,Hq,S_q,D]`, `k`/`v` `[B,Hkv,S,D]`; optional string metadata `scale`, `mask`, `model`,
//! `layer` is recorded):
//!
//! ```text
//! cargo run --release -p mlx-llm --bin sc20677_kv_candidates -- \
//!     --kv /path/llama-layer12.safetensors --kv /path/qwen-layer20.safetensors \
//!     --warm-iterations 20 --out /path/sc20677-comparison.json
//! ```

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use mlx_rs::{Array, Dtype};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::rabitq::{RabitqConfig, RabitqKvCandidate};
use super::rvq::{RvqConfig, RvqKvCandidate};
use super::{
    candidate_head_dimension_supported, candidate_request_support, decline, dense_attention_oracle,
    mlx_dims, probe_materialization, CacheGeometry, CandidateAttentionRequest,
    CandidateFallbackReason, CandidateRepresentation, CompressedKvCandidate, MaterializationProbe,
    VELOXQUANT_COMMIT, VELOXQUANT_TAG,
};
use crate::error::{Error, Result};
use crate::primitives::kv_cache::{CacheRoute, PackedAttentionMask};
use crate::primitives::packed_group_affine_kv::{
    CompiledKernelHandle, DenseFallbackEvent, PackedGroupAffineKvCache, StagedReaderLayer,
    PACKED_METAL_QUANT_GROUP_SIZE,
};
use crate::primitives::packed_metal::{PackedMask, PackedMetalKernel};
use crate::primitives::sampler::SplitMix64;

pub const REPORT_SCHEMA: &str = "sc-20677-kv-candidate-comparison/v1";
const GROUP_AFFINE_IDENTITY: &str = "sc-20677-group-affine-b2-g32";

// ---------------------------------------------------------------------------------------------
// Group-affine adapter over the existing SC-20675 cache and retained SC-20676 reader.
// ---------------------------------------------------------------------------------------------

pub struct GroupAffineCandidate {
    cache: PackedGroupAffineKvCache,
    kernel: Arc<PackedMetalKernel>,
    staged_arguments: Option<StagedReaderLayer>,
    geometry: CacheGeometry,
    fallback_events: Vec<DenseFallbackEvent>,
}

impl GroupAffineCandidate {
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(batch: usize, kv_heads: usize, head_dim: usize) -> Result<Self> {
        if !candidate_head_dimension_supported(head_dim) {
            return Err(Error::Unsupported(
                CandidateFallbackReason::UnsupportedHeadDimension
                    .as_str()
                    .into(),
            ));
        }
        let mut cache = PackedGroupAffineKvCache::new(
            GROUP_AFFINE_IDENTITY,
            1,
            batch,
            kv_heads,
            head_dim,
            PACKED_METAL_QUANT_GROUP_SIZE,
        )?;
        let kernel = Arc::new(PackedMetalKernel::for_identity(GROUP_AFFINE_IDENTITY)?);
        cache.bind_compiled_handle(CompiledKernelHandle::new(kernel.clone()))?;
        Ok(Self {
            cache,
            kernel,
            staged_arguments: None,
            geometry: CacheGeometry {
                batch,
                kv_heads,
                head_dim,
                logical_len: 0,
            },
            fallback_events: Vec::new(),
        })
    }

    fn support(
        &self,
        request: &CandidateAttentionRequest,
    ) -> std::result::Result<(), CandidateFallbackReason> {
        candidate_request_support(request, self.geometry)?;
        let expected = 1.0 / (request.head_dim as f32).sqrt();
        if (request.scale - expected).abs() > expected * 1e-6 {
            return Err(CandidateFallbackReason::UnsupportedScale);
        }
        Ok(())
    }
}

impl CompressedKvCandidate for GroupAffineCandidate {
    fn family(&self) -> &'static str {
        "group-affine"
    }

    fn config(&self) -> String {
        format!("b2-g{PACKED_METAL_QUANT_GROUP_SIZE}")
    }

    fn representation(&self) -> CandidateRepresentation {
        let [key_codes, key_metadata, staging, value_codes, value_metadata] =
            self.cache.layer_component_bytes(0).unwrap_or([0; 5]);
        let payload = key_codes + key_metadata + staging + value_codes + value_metadata;
        CandidateRepresentation {
            family: self.family().into(),
            config: self.config(),
            identity: GROUP_AFFINE_IDENTITY.into(),
            version: self.cache.representation().version,
            batch: self.geometry.batch,
            kv_heads: self.geometry.kv_heads,
            head_dim: self.geometry.head_dim,
            logical_len: self.geometry.logical_len,
            layout: format!(
                "{}; {}",
                self.cache.representation().key_grouping,
                self.cache.representation().value_grouping
            ),
            key_code_bytes: key_codes,
            key_metadata_bytes: key_metadata,
            value_code_bytes: value_codes,
            value_metadata_bytes: value_metadata,
            dense_staging_bytes: staging,
            shared_constant_bytes: 0,
            representation_bytes: payload,
            host_allocated_bytes: self.cache.host_allocated_payload_bytes(),
            device_bytes: self.cache.retained_device_packed_logical_bytes(),
        }
    }

    fn preflight(&mut self, request: &CandidateAttentionRequest) -> CacheRoute {
        match self.support(request) {
            Ok(()) => CacheRoute::ExperimentalPacked,
            Err(reason) => {
                let bytes = self.representation().representation_bytes;
                decline(
                    &mut self.fallback_events,
                    "attend",
                    reason,
                    self.geometry.logical_len,
                    bytes,
                )
            }
        }
    }

    fn append(&mut self, keys: &[f32], values: &[f32], step: usize) -> Result<()> {
        super::validate_step(self.geometry, keys, values, step)?;
        self.cache.append(0, keys, values, step)?;
        self.geometry.logical_len = self.cache.logical_len();
        self.staged_arguments = None;
        Ok(())
    }

    fn trim(&mut self, len: usize) -> Result<()> {
        self.cache.trim(len)?;
        self.geometry.logical_len = len;
        self.staged_arguments = None;
        Ok(())
    }

    /// The SC-20676 reader stages its device layer inside the first dispatch.
    fn sync_device(&mut self) -> Result<()> {
        Ok(())
    }

    fn attend(&mut self, query: &Array, request: &CandidateAttentionRequest) -> Result<Array> {
        if let Err(reason) = self.support(request) {
            return Err(Error::Unsupported(format!(
                "group-affine attend without passing preflight: {}",
                reason.as_str()
            )));
        }
        let mask = group_affine_mask(request)?;
        self.cache.dispatch_packed(0, query, mask)
    }

    /// Sync the host-reference mirror once (untimed) and dispatch the retained SC-20676 kernel on
    /// it directly, so the per-dispatch mirror sync (delta upload plus the bounded pending K group
    /// re-upload) is excluded.
    fn attend_excluding_staging(
        &mut self,
        query: &Array,
        request: &CandidateAttentionRequest,
    ) -> Option<Result<Array>> {
        Some((|| {
            let mask = group_affine_mask(request)?;
            if self.staged_arguments.is_none() {
                self.staged_arguments = Some(self.cache.staged_reader_arguments(0)?);
            }
            let staged = self.staged_arguments.as_ref().expect("staged");
            self.kernel.dispatch(&staged.args(query, mask))
        })())
    }

    fn kernel_profile(&self) -> super::KernelProfile {
        let tuning = self.kernel.tuning_profile(self.geometry.head_dim);
        super::KernelProfile {
            kernel: "sc20676_split_kv_simdgroup".into(),
            threads_per_threadgroup: tuning.map_or(0, |t| t.threads),
            simd_groups: tuning.map_or(0, |t| t.simd_groups),
            gpu_family: tuning.map_or("unsupported".into(), |t| t.gpu_family.into()),
            // Barriers occur only in the once-per-threadgroup SIMD-group merge, never per token.
            threadgroup_barriers_per_kv_token: 0,
            softmax_state: "registers per SIMD group (simd_sum), merged once through threadgroup \
                            memory; split-KV partials merged by a reduce pass"
                .into(),
            extra_mlx_ops_per_attend: 0,
            // The host-reference cache syncs its device mirror inside `dispatch_packed`: appended
            // deltas plus a re-upload of the bounded (< one group) pending K tail.
            host_staging_inside_attend: true,
        }
    }

    fn dequantize_dense_for_oracle(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        self.cache.record_dense_dequantization();
        let (_, keys, values) = self.cache.evaluated_dense_layer(0)?;
        Ok((keys, values))
    }

    fn full_cache_dequantizations(&self) -> usize {
        self.cache.full_cache_dequantizations()
    }

    fn fallback_events(&self) -> &[DenseFallbackEvent] {
        &self.fallback_events
    }
}

fn group_affine_mask(request: &CandidateAttentionRequest) -> Result<PackedMask> {
    Ok(match request.mask {
        PackedAttentionMask::None => PackedMask::None,
        PackedAttentionMask::Causal => PackedMask::Causal,
        // Clamp as the rotated candidates do: a window >= kv_len admits exactly the causal set.
        PackedAttentionMask::SlidingWindow(window) => {
            PackedMask::SlidingWindow(window.min(request.kv_len))
        }
        PackedAttentionMask::Additive => {
            return Err(Error::Unsupported(
                CandidateFallbackReason::AdditiveMask.as_str().into(),
            ))
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Inputs.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum CaseSource {
    Synthetic {
        seed: u64,
        generator: String,
    },
    Captured {
        path: String,
        sha256: String,
        metadata: BTreeMap<String, String>,
    },
}

/// One comparison input: dense `[B,Hq,S_q,D]` query and `[B,Hkv,S,D]` K/V in f32.
#[derive(Clone, Debug)]
pub struct KvCase {
    pub source: CaseSource,
    pub request: CandidateAttentionRequest,
    pub query: Vec<f32>,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SyntheticShape {
    pub batch: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub query_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
}

fn gaussian(rng: &mut SplitMix64) -> f32 {
    let mut uniform = || ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let (u, v) = (uniform(), uniform());
    ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
}

/// Gaussian K/V/Q with the structure that stresses KV quantizers: keys carry a per-head mean
/// offset and every sixteenth channel is a 6x outlier, as post-RoPE LLM keys typically do.
pub fn synthetic_case(
    shape: SyntheticShape,
    seed: u64,
    mask: PackedAttentionMask,
    scale: Option<f32>,
) -> KvCase {
    let mut rng = SplitMix64::new(seed);
    let rows = shape.batch * shape.kv_heads;
    let mut keys = Vec::with_capacity(rows * shape.kv_len * shape.head_dim);
    for _ in 0..rows {
        let offsets: Vec<f32> = (0..shape.head_dim)
            .map(|_| 0.5 * gaussian(&mut rng))
            .collect();
        for _ in 0..shape.kv_len {
            for (d, offset) in offsets.iter().enumerate() {
                let outlier = if d % 16 == 0 { 6.0 } else { 1.0 };
                keys.push(offset + outlier * gaussian(&mut rng));
            }
        }
    }
    let values = (0..keys.len()).map(|_| gaussian(&mut rng)).collect();
    let query = (0..shape.batch * shape.query_heads * shape.query_len * shape.head_dim)
        .map(|_| gaussian(&mut rng))
        .collect();
    KvCase {
        source: CaseSource::Synthetic {
            seed,
            generator: "gaussian; K per-head mean 0.5σ; every 16th K channel 6x".into(),
        },
        request: CandidateAttentionRequest {
            backend: "mlx-metal".into(),
            batch: shape.batch,
            query_heads: shape.query_heads,
            kv_heads: shape.kv_heads,
            query_len: shape.query_len,
            kv_len: shape.kv_len,
            head_dim: shape.head_dim,
            mask,
            scale: scale.unwrap_or(1.0 / (shape.head_dim as f32).sqrt()),
        },
        query,
        keys,
        values,
    }
}

pub fn parse_mask(text: &str) -> Result<PackedAttentionMask> {
    match text {
        "none" => Ok(PackedAttentionMask::None),
        "causal" => Ok(PackedAttentionMask::Causal),
        "additive" => Ok(PackedAttentionMask::Additive),
        other => other
            .strip_prefix("window:")
            .and_then(|window| window.parse().ok())
            .map(PackedAttentionMask::SlidingWindow)
            .ok_or_else(|| Error::Config(format!("unknown mask `{other}`"))),
    }
}

fn host_tensor(tensors: &BTreeMap<String, Array>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let array = tensors
        .get(name)
        .ok_or_else(|| Error::MissingTensor(format!("captured K/V file lacks `{name}`")))?;
    let shape = array
        .shape()
        .iter()
        .map(|d| usize::try_from(*d).map_err(|_| Error::Config("negative dimension".into())))
        .collect::<Result<Vec<_>>>()?;
    if shape.len() != 4 {
        return Err(Error::Config(format!("`{name}` must be rank 4")));
    }
    let data = array.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
    Ok((shape, data))
}

/// Load a captured `(q, k, v)` safetensors file. `mask`/`scale` override file metadata.
pub fn load_captured_case(
    path: &Path,
    mask: Option<PackedAttentionMask>,
    scale: Option<f32>,
) -> Result<KvCase> {
    let bytes = std::fs::read(path)?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let (arrays, metadata) = Array::load_safetensors_with_metadata(path)
        .map_err(|e| Error::Msg(format!("load_safetensors {}: {e}", path.display())))?;
    let tensors: BTreeMap<String, Array> = arrays.into_iter().collect();
    let metadata: BTreeMap<String, String> = metadata.into_iter().collect();
    let (q_shape, query) = host_tensor(&tensors, "q")?;
    let (k_shape, keys) = host_tensor(&tensors, "k")?;
    let (v_shape, values) = host_tensor(&tensors, "v")?;
    if k_shape != v_shape || q_shape[0] != k_shape[0] || q_shape[3] != k_shape[3] {
        return Err(Error::Config(
            "captured q/k/v shapes disagree on batch or head dimension".into(),
        ));
    }
    if keys
        .iter()
        .chain(&values)
        .chain(&query)
        .any(|v| !v.is_finite())
    {
        return Err(Error::Config(
            "captured q/k/v contain non-finite values".into(),
        ));
    }
    let mask = match mask {
        Some(mask) => mask,
        None => metadata
            .get("mask")
            .map(|m| parse_mask(m))
            .transpose()?
            .unwrap_or(PackedAttentionMask::Causal),
    };
    let scale = match scale {
        Some(scale) => scale,
        None => match metadata.get("scale") {
            Some(text) => text
                .parse()
                .map_err(|_| Error::Config(format!("metadata scale `{text}` is not a float")))?,
            None => 1.0 / (q_shape[3] as f32).sqrt(),
        },
    };
    Ok(KvCase {
        source: CaseSource::Captured {
            path: path.display().to_string(),
            sha256,
            metadata,
        },
        request: CandidateAttentionRequest {
            backend: "mlx-metal".into(),
            batch: q_shape[0],
            query_heads: q_shape[1],
            kv_heads: k_shape[1],
            query_len: q_shape[2],
            kv_len: k_shape[2],
            head_dim: q_shape[3],
            mask,
            scale,
        },
        query,
        keys,
        values,
    })
}

// ---------------------------------------------------------------------------------------------
// Per-candidate measurement.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ErrorMetrics {
    pub max_abs_error: f64,
    pub relative_l2_error: f64,
}

pub fn error_metrics(actual: &[f32], expected: &[f32]) -> ErrorMetrics {
    let (mut diff, mut energy, mut max) = (0.0f64, 0.0f64, 0.0f64);
    for (a, e) in actual.iter().zip(expected) {
        let d = f64::from(*a) - f64::from(*e);
        diff += d * d;
        energy += f64::from(*e) * f64::from(*e);
        max = max.max(d.abs());
    }
    ErrorMetrics {
        max_abs_error: max,
        relative_l2_error: if energy > 0.0 {
            (diff / energy).sqrt()
        } else {
            diff.sqrt()
        },
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Timing {
    pub encode_ms: f64,
    /// `None` when the reader stages its device representation inside the first dispatch.
    pub device_sync_ms: Option<f64>,
    /// First dispatch, including JIT compilation and any lazy staging.
    pub attend_cold_ms: f64,
    pub attend_warm_median_ms: f64,
    pub attend_warm_min_ms: f64,
    /// Warm median with the reader's per-dispatch host staging hoisted out (group-affine only;
    /// the rotated candidates stage nothing inside `attend`).
    pub attend_excluding_staging_warm_median_ms: Option<f64>,
    pub warm_iterations: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CandidateResult {
    pub family: String,
    pub config: String,
    /// `compressed`, `dense-fallback` (declined before mutation) or `error` (a bug, recorded).
    pub route: String,
    pub fallback_reason: Option<String>,
    pub error: Option<String>,
    pub representation: Option<CandidateRepresentation>,
    pub kernel: Option<super::KernelProfile>,
    pub bytes_per_token_per_kv_head: Option<f64>,
    pub compression_vs_dense_fp16: Option<f64>,
    pub timing: Option<Timing>,
    pub quality_vs_exact: Option<ErrorMetrics>,
    pub parity_vs_dequantize_oracle: Option<ErrorMetrics>,
    pub materialization: Option<MaterializationProbe>,
    pub full_cache_dequantizations_during_attend: Option<usize>,
}

impl CandidateResult {
    fn declined(family: &str, config: &str, reason: String) -> Self {
        Self {
            family: family.into(),
            config: config.into(),
            route: "dense-fallback".into(),
            fallback_reason: Some(reason),
            error: None,
            representation: None,
            kernel: None,
            bytes_per_token_per_kv_head: None,
            compression_vs_dense_fp16: None,
            timing: None,
            quality_vs_exact: None,
            parity_vs_dequantize_oracle: None,
            materialization: None,
            full_cache_dequantizations_during_attend: None,
        }
    }

    pub fn is_compressed(&self) -> bool {
        self.route == "compressed"
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

pub fn dense_fp16_kv_bytes(request: &CandidateAttentionRequest) -> usize {
    request.batch * request.kv_heads * request.kv_len * request.head_dim * 2 * 2
}

/// Measure one candidate. Returns a declined row when preflight falls back; a failure after the
/// compressed route was accepted is recorded as `route = "error"` rather than hidden.
pub fn measure_candidate(
    case: &KvCase,
    exact: &[f32],
    mut candidate: Box<dyn CompressedKvCandidate>,
    warm_iterations: usize,
) -> CandidateResult {
    let family = candidate.family().to_string();
    let config = candidate.config();
    let request = &case.request;
    let started = Instant::now();
    if let Err(error) = candidate.append(&case.keys, &case.values, request.kv_len) {
        return CandidateResult::declined(&family, &config, format!("encode declined: {error}"));
    }
    let encode_ms = elapsed_ms(started);
    if let CacheRoute::DenseFallback { reason } = candidate.preflight(request) {
        return CandidateResult::declined(&family, &config, reason);
    }
    match measure_accepted(case, exact, candidate.as_mut(), encode_ms, warm_iterations) {
        Ok(result) => result,
        Err(error) => CandidateResult {
            route: "error".into(),
            error: Some(error.to_string()),
            ..CandidateResult::declined(&family, &config, String::new())
        },
    }
}

fn measure_accepted(
    case: &KvCase,
    exact: &[f32],
    candidate: &mut dyn CompressedKvCandidate,
    encode_ms: f64,
    warm_iterations: usize,
) -> Result<CandidateResult> {
    let request = &case.request;
    let started = Instant::now();
    candidate.sync_device()?;
    let sync_ms = elapsed_ms(started);
    let query = Array::from_slice(
        &case.query,
        &mlx_dims(&[
            request.batch,
            request.query_heads,
            request.query_len,
            request.head_dim,
        ])?,
    );
    query.eval()?;
    let dequantizations_before = candidate.full_cache_dequantizations();

    let started = Instant::now();
    let cold = candidate.attend(&query, request)?;
    cold.eval()?;
    let attend_cold_ms = elapsed_ms(started);

    let (probed, materialization) =
        probe_materialization(request, || candidate.attend(&query, request))?;
    let output = probed.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();

    let mut warm = Vec::with_capacity(warm_iterations);
    for _ in 0..warm_iterations {
        let started = Instant::now();
        candidate.attend(&query, request)?.eval()?;
        warm.push(elapsed_ms(started));
    }
    warm.sort_by(f64::total_cmp);
    let mut excluding_staging = Vec::new();
    if let Some(first) = candidate.attend_excluding_staging(&query, request) {
        first?.eval()?; // stages the reader arguments once, outside the timed loop
        for _ in 0..warm_iterations {
            let started = Instant::now();
            if let Some(output) = candidate.attend_excluding_staging(&query, request) {
                output?.eval()?;
            }
            excluding_staging.push(elapsed_ms(started));
        }
        excluding_staging.sort_by(f64::total_cmp);
    }
    let dequantizations_during_attend =
        candidate.full_cache_dequantizations() - dequantizations_before;
    let representation = candidate.representation();

    let (dense_keys, dense_values) = candidate.dequantize_dense_for_oracle()?;
    let oracle_query = candidate.oracle_query(&case.query, request)?;
    let oracle = dense_attention_oracle(request, &oracle_query, &dense_keys, &dense_values)?;

    let vectors = (request.batch * request.kv_heads * request.kv_len) as f64;
    Ok(CandidateResult {
        family: candidate.family().into(),
        config: candidate.config(),
        route: "compressed".into(),
        fallback_reason: None,
        error: None,
        bytes_per_token_per_kv_head: Some(representation.payload_bytes() as f64 / vectors),
        compression_vs_dense_fp16: Some(
            dense_fp16_kv_bytes(request) as f64 / representation.representation_bytes as f64,
        ),
        representation: Some(representation),
        kernel: Some(candidate.kernel_profile()),
        timing: Some(Timing {
            encode_ms,
            device_sync_ms: (candidate.family() != "group-affine").then_some(sync_ms),
            attend_cold_ms,
            attend_warm_median_ms: warm.get(warm.len() / 2).copied().unwrap_or(f64::NAN),
            attend_warm_min_ms: warm.first().copied().unwrap_or(f64::NAN),
            attend_excluding_staging_warm_median_ms: excluding_staging
                .get(excluding_staging.len() / 2)
                .copied(),
            warm_iterations,
        }),
        quality_vs_exact: Some(error_metrics(&output, exact)),
        parity_vs_dequantize_oracle: Some(error_metrics(&output, &oracle)),
        materialization: Some(materialization),
        full_cache_dequantizations_during_attend: Some(dequantizations_during_attend),
    })
}

/// `(family, config, candidate-or-construction-refusal)`.
pub type CandidateEntry = (String, String, Result<Box<dyn CompressedKvCandidate>>);

/// The comparison set (E5): the incumbent group-affine reader, packed RVQ at 1..=3 bits per
/// stage for K and V independently, and both RaBitQ score estimators.
pub fn candidate_set(batch: usize, kv_heads: usize, head_dim: usize) -> Vec<CandidateEntry> {
    let mut set: Vec<CandidateEntry> = Vec::new();
    set.push((
        "group-affine".into(),
        format!("b2-g{PACKED_METAL_QUANT_GROUP_SIZE}"),
        GroupAffineCandidate::new(batch, kv_heads, head_dim)
            .map(|c| Box::new(c) as Box<dyn CompressedKvCandidate>),
    ));
    for key_bits in 1..=3 {
        for value_bits in 1..=3 {
            let config = RvqConfig {
                key_bits,
                value_bits,
            };
            set.push((
                "packed-rvq".into(),
                config.label(),
                RvqKvCandidate::new(config, batch, kv_heads, head_dim)
                    .map(|c| Box::new(c) as Box<dyn CompressedKvCandidate>),
            ));
        }
    }
    for config in RabitqConfig::all() {
        set.push((
            "rabitq".into(),
            config.label(),
            RabitqKvCandidate::new(config, batch, kv_heads, head_dim)
                .map(|c| Box::new(c) as Box<dyn CompressedKvCandidate>),
        ));
    }
    set
}

// ---------------------------------------------------------------------------------------------
// Matching.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Pick {
    pub config: String,
    pub representation_bytes: usize,
    pub relative_l2_error: f64,
    /// `representation_bytes / budget_bytes` for a budget match: picks are the best fit
    /// *within* the budget, which for coarse bit grids can be well below it.
    pub budget_fraction: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BudgetMatch {
    pub budget_bytes: usize,
    /// Per family: the lowest-error compressed configuration within the budget.
    pub picks: BTreeMap<String, Option<Pick>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QualityMatch {
    pub target_relative_l2_error: f64,
    /// Per family: the smallest compressed configuration meeting the target.
    pub picks: BTreeMap<String, Option<Pick>>,
}

fn picks(results: &[CandidateResult]) -> BTreeMap<String, Vec<Pick>> {
    let mut by_family: BTreeMap<String, Vec<Pick>> = BTreeMap::new();
    for result in results {
        by_family.entry(result.family.clone()).or_default();
        if let (true, Some(rep), Some(quality)) = (
            result.is_compressed(),
            &result.representation,
            &result.quality_vs_exact,
        ) {
            by_family
                .get_mut(&result.family)
                .expect("inserted")
                .push(Pick {
                    config: result.config.clone(),
                    representation_bytes: rep.representation_bytes,
                    relative_l2_error: quality.relative_l2_error,
                    budget_fraction: None,
                });
        }
    }
    by_family
}

pub fn match_budgets(results: &[CandidateResult], budgets: &[usize]) -> Vec<BudgetMatch> {
    let families = picks(results);
    budgets
        .iter()
        .map(|&budget_bytes| BudgetMatch {
            budget_bytes,
            picks: families
                .iter()
                .map(|(family, options)| {
                    let best = options
                        .iter()
                        .filter(|p| p.representation_bytes <= budget_bytes)
                        .min_by(|a, b| a.relative_l2_error.total_cmp(&b.relative_l2_error))
                        .map(|pick| Pick {
                            budget_fraction: Some(
                                pick.representation_bytes as f64 / budget_bytes as f64,
                            ),
                            ..pick.clone()
                        });
                    (family.clone(), best)
                })
                .collect(),
        })
        .collect()
}

pub fn match_quality(results: &[CandidateResult], targets: &[f64]) -> Vec<QualityMatch> {
    let families = picks(results);
    targets
        .iter()
        .map(|&target| QualityMatch {
            target_relative_l2_error: target,
            picks: families
                .iter()
                .map(|(family, options)| {
                    let best = options
                        .iter()
                        .filter(|p| p.relative_l2_error <= target)
                        .min_by_key(|p| p.representation_bytes)
                        .cloned();
                    (family.clone(), best)
                })
                .collect(),
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Report.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Provenance {
    pub inference_git_sha: String,
    pub inference_dirty: bool,
    pub veloxquant_commit: String,
    pub veloxquant_tag: String,
    pub mlx_rs_lock_source: Option<String>,
    pub os: String,
    pub arch: String,
    pub cpu_brand: Option<String>,
    pub generated_unix_seconds: u64,
    pub command_line: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CaseReport {
    pub source: CaseSource,
    pub request: CandidateAttentionRequest,
    pub dense_fp16_kv_bytes: usize,
    pub exact_reference: String,
    pub results: Vec<CandidateResult>,
    pub budget_matches: Vec<BudgetMatch>,
    pub quality_matches: Vec<QualityMatch>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComparisonReport {
    pub schema: String,
    pub provenance: Provenance,
    pub caveats: Vec<String>,
    pub cases: Vec<CaseReport>,
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub fn provenance(command_line: Vec<String>) -> Result<Provenance> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .ok_or_else(|| Error::Msg("inference root".into()))?;
    let root_text = root.to_string_lossy();
    let sha = command_output("git", &["-C", &root_text, "rev-parse", "HEAD"])
        .filter(|sha| sha.len() == 40)
        .ok_or_else(|| Error::Msg("cannot resolve inference git SHA for provenance".into()))?;
    let dirty = command_output("git", &["-C", &root_text, "status", "--porcelain"])
        .is_none_or(|status| !status.is_empty());
    let mlx_rs_lock_source = std::fs::read_to_string(root.join("Cargo.lock"))
        .ok()
        .and_then(|lock| {
            let block = lock
                .split("[[package]]")
                .find(|block| block.contains("name = \"pmetal-mlx-rs\""))?;
            block
                .lines()
                .find_map(|line| line.strip_prefix("source = "))
                .map(|source| source.trim_matches('"').to_string())
        });
    Ok(Provenance {
        inference_git_sha: sha,
        inference_dirty: dirty,
        veloxquant_commit: VELOXQUANT_COMMIT.into(),
        veloxquant_tag: VELOXQUANT_TAG.into(),
        mlx_rs_lock_source,
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        cpu_brand: command_output("sysctl", &["-n", "machdep.cpu.brand_string"]),
        generated_unix_seconds: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        command_line,
    })
}

/// Matched-budget points: the incumbent's bytes plus any requested budgets, sorted and unique.
pub fn budget_points(incumbent: Option<usize>, extra: &[usize]) -> Vec<usize> {
    let mut budgets: Vec<usize> = incumbent.into_iter().chain(extra.iter().copied()).collect();
    budgets.sort_unstable();
    budgets.dedup();
    budgets
}

/// Interpretation limits every report carries.
pub fn report_caveats() -> Vec<String> {
    vec![
        "timing is not kernel-for-kernel: the group-affine incumbent is the SC-20676 reader \
         (split-KV SIMD-group kernel with register softmax state, no per-token threadgroup \
         barriers, and a per-dispatch mirror sync inside attend that re-uploads the bounded \
         pending key group; see attend_excluding_staging_warm_median_ms), while packed-rvq/rabitq use one SIMD group with register state and add 6 \
         MLX rotation/cast ops per attend. Each result's `kernel` block records its geometry."
            .into(),
        "the dense-materialization probe only trips on a transient at least one whole dense K \
         (or V) at fp16 or a whole S_q x S_kv fp16 score tensor; a chunked reconstruction below \
         those sizes would evade it, so the probe complements, not replaces, source review and \
         the full_cache_dequantizations counter."
            .into(),
        "budget picks are the best configuration within the budget; see budget_fraction for \
         how much of the budget each pick uses."
            .into(),
    ]
}

/// Compare every candidate on one case.
pub fn compare_case(
    case: &KvCase,
    warm_iterations: usize,
    extra_budgets: &[usize],
    extra_targets: &[f64],
) -> Result<CaseReport> {
    let request = &case.request;
    let exact = dense_attention_oracle(request, &case.query, &case.keys, &case.values)?;
    let results: Vec<CandidateResult> =
        candidate_set(request.batch, request.kv_heads, request.head_dim)
            .into_iter()
            .map(|(family, config, candidate)| match candidate {
                Ok(candidate) => measure_candidate(case, &exact, candidate, warm_iterations),
                Err(error) => CandidateResult::declined(
                    &family,
                    &config,
                    format!("construction declined: {error}"),
                ),
            })
            .collect();
    let incumbent = results
        .iter()
        .find(|r| r.family == "group-affine" && r.is_compressed());
    let budgets = budget_points(
        incumbent
            .and_then(|r| r.representation.as_ref())
            .map(|rep| rep.representation_bytes),
        extra_budgets,
    );
    let targets: Vec<f64> = incumbent
        .and_then(|r| r.quality_vs_exact)
        .map(|q| q.relative_l2_error)
        .into_iter()
        .chain(extra_targets.iter().copied())
        .collect();
    Ok(CaseReport {
        source: case.source.clone(),
        request: request.clone(),
        dense_fp16_kv_bytes: dense_fp16_kv_bytes(request),
        exact_reference: "fp32 CPU attention over the unquantized input K/V".into(),
        budget_matches: match_budgets(&results, &budgets),
        quality_matches: match_quality(&results, &targets),
        results,
    })
}

// ---------------------------------------------------------------------------------------------
// CLI.
// ---------------------------------------------------------------------------------------------

pub const USAGE: &str = "\
sc20677_kv_candidates: compare group-affine, packed RVQ and RaBitQ KV representations.

Inputs (captured files take precedence; with none, one synthetic case runs):
  --kv PATH               captured safetensors with q [B,Hq,Sq,D], k/v [B,Hkv,S,D]; repeatable
  --batch N --query-heads N --kv-heads N --query-len N --kv-len N --head-dim N --seed N
                          synthetic shape (default 1, 8, 2, 1, 1024, 128, 20677)
Options:
  --mask none|causal|window:N   (default: file metadata `mask`, else causal)
  --scale F                     (default: file metadata `scale`, else 1/sqrt(D))
  --warm-iterations N           (default 10)
  --budget-bytes N              extra matched-budget point; repeatable
  --target-rel-error F          extra matched-quality point; repeatable
  --out PATH                    write the JSON report (default: stdout)

Real-weight run (one command once captures exist):
  cargo run --release -p mlx-llm --bin sc20677_kv_candidates -- \\
      --kv llama-layer12.safetensors --kv qwen-layer20.safetensors \\
      --warm-iterations 20 --out sc20677-comparison.json
";

#[derive(Clone, Debug, PartialEq)]
pub struct CliOptions {
    pub captured: Vec<String>,
    pub synthetic: SyntheticShape,
    pub seed: u64,
    pub mask: Option<PackedAttentionMask>,
    pub scale: Option<f32>,
    pub warm_iterations: usize,
    pub budgets: Vec<usize>,
    pub targets: Vec<f64>,
    pub out: Option<String>,
    pub help: bool,
}

pub fn parse_cli(args: &[String]) -> Result<CliOptions> {
    let mut options = CliOptions {
        captured: Vec::new(),
        synthetic: SyntheticShape {
            batch: 1,
            query_heads: 8,
            kv_heads: 2,
            query_len: 1,
            kv_len: 1024,
            head_dim: 128,
        },
        seed: 20677,
        mask: None,
        scale: None,
        warm_iterations: 10,
        budgets: Vec::new(),
        targets: Vec::new(),
        out: None,
        help: false,
    };
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        if flag == "--help" || flag == "-h" {
            options.help = true;
            continue;
        }
        let value = iter
            .next()
            .ok_or_else(|| Error::Config(format!("{flag} needs a value")))?;
        let number = |value: &str| -> Result<usize> {
            value
                .parse()
                .map_err(|_| Error::Config(format!("{flag} expects an integer, got `{value}`")))
        };
        match flag.as_str() {
            "--kv" => options.captured.push(value.clone()),
            "--batch" => options.synthetic.batch = number(value)?,
            "--query-heads" => options.synthetic.query_heads = number(value)?,
            "--kv-heads" => options.synthetic.kv_heads = number(value)?,
            "--query-len" => options.synthetic.query_len = number(value)?,
            "--kv-len" => options.synthetic.kv_len = number(value)?,
            "--head-dim" => options.synthetic.head_dim = number(value)?,
            "--seed" => options.seed = number(value)? as u64,
            "--mask" => options.mask = Some(parse_mask(value)?),
            "--scale" => {
                options.scale = Some(value.parse().map_err(|_| {
                    Error::Config(format!("--scale expects a float, got `{value}`"))
                })?)
            }
            "--warm-iterations" => options.warm_iterations = number(value)?,
            "--budget-bytes" => options.budgets.push(number(value)?),
            "--target-rel-error" => options.targets.push(value.parse().map_err(|_| {
                Error::Config(format!("--target-rel-error expects a float, got `{value}`"))
            })?),
            "--out" => options.out = Some(value.clone()),
            other => return Err(Error::Config(format!("unknown flag `{other}`\n\n{USAGE}"))),
        }
    }
    let shape = options.synthetic;
    if [
        shape.batch,
        shape.query_heads,
        shape.kv_heads,
        shape.query_len,
        shape.kv_len,
        shape.head_dim,
    ]
    .contains(&0)
    {
        return Err(Error::Config(
            "synthetic dimensions must be positive".into(),
        ));
    }
    Ok(options)
}

/// Entry point for `src/bin/sc20677_kv_candidates.rs`.
pub fn cli(args: &[String]) -> Result<()> {
    let options = parse_cli(args)?;
    if options.help {
        print!("{USAGE}");
        return Ok(());
    }
    let cases = if options.captured.is_empty() {
        vec![synthetic_case(
            options.synthetic,
            options.seed,
            options.mask.unwrap_or(PackedAttentionMask::Causal),
            options.scale,
        )]
    } else {
        options
            .captured
            .iter()
            .map(|path| load_captured_case(Path::new(path), options.mask, options.scale))
            .collect::<Result<Vec<_>>>()?
    };
    let mut command_line = vec!["sc20677_kv_candidates".to_string()];
    command_line.extend(args.iter().cloned());
    let report = ComparisonReport {
        schema: REPORT_SCHEMA.into(),
        provenance: provenance(command_line)?,
        caveats: report_caveats(),
        cases: cases
            .iter()
            .map(|case| {
                compare_case(
                    case,
                    options.warm_iterations,
                    &options.budgets,
                    &options.targets,
                )
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| Error::Msg(format!("serialize report: {e}")))?;
    match options.out {
        Some(path) => std::fs::write(path, json + "\n")?,
        None => println!("{json}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::rabitq::{RabitqMagnitude, RabitqScore};
    use super::super::MaterializationVerdict;
    use super::*;

    /// Per-config ceilings on relative L2 attention error vs exact fp32 for the fixed synthetic
    /// case in `synthetic_comparison_runs_every_candidate_compressed_and_matches_budgets`
    /// (measured values plus ~10% headroom; the kernels are deterministic). An encoder/codebook regression that parity cannot
    /// see (parity decodes the same codes) trips these. RaBitQ attention error is dominated by its
    /// 1-bit keys, so value-side RaBitQ regressions are caught by the value round-trip SNR test.
    const QUALITY_CEILINGS: &[(&str, f64)] = &[
        ("b2-g32", 0.82),
        ("k1x2-v1x2", 0.62),
        ("k1x2-v2x2", 0.49),
        ("k1x2-v3x2", 0.46),
        ("k2x2-v1x2", 0.54),
        ("k2x2-v2x2", 0.41),
        ("k2x2-v3x2", 0.33),
        ("k3x2-v1x2", 0.49),
        ("k3x2-v2x2", 0.28),
        ("k3x2-v3x2", 0.20),
        ("k1-hamming-l1mean-v4", 0.87),
        ("k1-hamming-unbiased-v4", 1.70),
        ("k1-fpq-l1mean-v4", 0.76),
        ("k1-fpq-unbiased-v4", 1.03),
    ];

    #[test]
    fn budget_points_are_sorted_and_unique() {
        assert_eq!(
            budget_points(Some(96), &[200, 50, 96, 200]),
            vec![50, 96, 200]
        );
        assert_eq!(budget_points(None, &[3, 1]), vec![1, 3]);
    }

    /// A sliding window wider than the cache (even beyond MLX i32) admits the causal set: every
    /// family must accept it and match its causal output exactly.
    #[cfg(target_os = "macos")]
    #[test]
    fn oversized_sliding_window_is_clamped_and_staging_excluded_dispatch_matches() {
        let shape = SyntheticShape {
            batch: 1,
            query_heads: 2,
            kv_heads: 1,
            query_len: 2,
            kv_len: 40,
            head_dim: 64,
        };
        let causal = synthetic_case(shape, 5, PackedAttentionMask::Causal, None);
        let wide = CandidateAttentionRequest {
            mask: PackedAttentionMask::SlidingWindow(usize::MAX),
            ..causal.request.clone()
        };
        let query = Array::from_slice(&causal.query, &[1, 2, 2, 64]);
        for (family, config, candidate) in candidate_set(1, 1, 64) {
            let mut candidate = candidate.unwrap();
            candidate
                .append(&causal.keys, &causal.values, shape.kv_len)
                .unwrap();
            assert_eq!(
                candidate.preflight(&wide),
                CacheRoute::ExperimentalPacked,
                "{family}/{config}"
            );
            candidate.sync_device().unwrap();
            let expected = candidate.attend(&query, &causal.request).unwrap();
            let actual = candidate.attend(&query, &wide).unwrap();
            // Only the group-affine reader stages inside `attend`; its staging-excluded dispatch
            // must produce the identical output.
            match candidate.attend_excluding_staging(&query, &causal.request) {
                Some(hoisted) => {
                    assert_eq!(family, "group-affine");
                    assert_eq!(
                        hoisted.unwrap().as_slice::<f32>(),
                        expected.as_slice::<f32>()
                    );
                }
                None => assert_ne!(family, "group-affine"),
            }
            assert_eq!(
                actual.as_slice::<f32>(),
                expected.as_slice::<f32>(),
                "{family}/{config}"
            );
        }
    }

    fn row(family: &str, config: &str, bytes: usize, error: f64) -> CandidateResult {
        CandidateResult {
            route: "compressed".into(),
            fallback_reason: None,
            representation: Some(CandidateRepresentation {
                representation_bytes: bytes,
                ..CandidateRepresentation::default()
            }),
            quality_vs_exact: Some(ErrorMetrics {
                max_abs_error: error,
                relative_l2_error: error,
            }),
            ..CandidateResult::declined(family, config, String::new())
        }
    }

    #[test]
    fn group_affine_component_bytes_are_exact_and_sum_to_stored_bytes() {
        let (rows, dim, tokens) = (2, 64, 40);
        let data: Vec<f32> = (0..rows * tokens * dim).map(|i| (i % 13) as f32).collect();
        let mut cache = PackedGroupAffineKvCache::new("bytes", 1, 1, rows, dim, 32).unwrap();
        cache.append(0, &data, &data, tokens).unwrap();
        let [key_codes, key_metadata, staging, value_codes, value_metadata] =
            cache.layer_component_bytes(0).unwrap();
        assert_eq!(
            key_codes,
            rows * 32 * dim / 4,
            "one complete 32-token group"
        );
        assert_eq!(
            key_metadata,
            rows * dim * 2 * 2,
            "f16 scale + zero per channel"
        );
        assert_eq!(staging, 8 * rows * dim * 4, "eight pending f32 key tokens");
        assert_eq!(value_codes, tokens * rows * dim / 4);
        assert_eq!(value_metadata, tokens * rows * (dim / 32) * 2 * 2);
        assert_eq!(
            key_codes + key_metadata + staging + value_codes + value_metadata,
            cache.logical_stored_bytes()
        );
    }

    #[test]
    fn budget_and_quality_matching_pick_per_family() {
        let results = vec![
            row("group-affine", "b2-g32", 96, 0.10),
            row("packed-rvq", "k1x2-v1x2", 68, 0.20),
            row("packed-rvq", "worse-within-budget", 80, 0.90),
            row("packed-rvq", "k2x2-v1x2", 100, 0.08),
            row("packed-rvq", "k2x2-v2x2", 132, 0.03),
            row("rabitq", "k1-fpq-unbiased-v4", 84, 0.30),
            CandidateResult::declined("rabitq", "k1-hamming-l1mean-v4", "declined".into()),
        ];
        let budget = &match_budgets(&results, &[96])[0];
        assert_eq!(
            budget.picks["group-affine"].as_ref().unwrap().config,
            "b2-g32"
        );
        assert_eq!(
            budget.picks["packed-rvq"].as_ref().unwrap().config,
            "k1x2-v1x2"
        );
        let rabitq = budget.picks["rabitq"].as_ref().unwrap();
        assert_eq!(rabitq.config, "k1-fpq-unbiased-v4");
        assert_eq!(rabitq.budget_fraction, Some(84.0 / 96.0));
        assert_eq!(
            budget.picks["packed-rvq"].as_ref().unwrap().budget_fraction,
            Some(68.0 / 96.0)
        );
        let quality = &match_quality(&results, &[0.10])[0];
        assert_eq!(
            quality.picks["packed-rvq"].as_ref().unwrap().config,
            "k2x2-v1x2"
        );
        assert_eq!(quality.picks["rabitq"], None);
        assert_eq!(
            quality.picks["packed-rvq"]
                .as_ref()
                .unwrap()
                .budget_fraction,
            None
        );
    }

    #[test]
    fn cli_parses_flags_masks_and_rejects_unknown() {
        let args = |s: &str| s.split_whitespace().map(String::from).collect::<Vec<_>>();
        let options = parse_cli(&args(
            "--kv a.safetensors --kv b.safetensors --mask window:64 --budget-bytes 10",
        ))
        .unwrap();
        assert_eq!(options.captured, vec!["a.safetensors", "b.safetensors"]);
        assert_eq!(options.mask, Some(PackedAttentionMask::SlidingWindow(64)));
        assert_eq!(options.budgets, vec![10]);
        assert!(parse_cli(&args("--bogus 1")).is_err());
        assert!(parse_cli(&args("--kv-len 0")).is_err());
        assert!(parse_mask("window:x").is_err());
    }

    #[test]
    fn error_metrics_are_relative_l2_and_max_abs() {
        let metrics = error_metrics(&[1.0, 2.0], &[1.0, 1.0]);
        assert_eq!(metrics.max_abs_error, 1.0);
        assert!((metrics.relative_l2_error - (0.5f64).sqrt()).abs() < 1e-12);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn synthetic_comparison_runs_every_candidate_compressed_and_matches_budgets() {
        let shape = SyntheticShape {
            batch: 1,
            query_heads: 4,
            kv_heads: 2,
            query_len: 2,
            kv_len: 64,
            head_dim: 64,
        };
        let case = synthetic_case(shape, 7, PackedAttentionMask::Causal, None);
        let report = compare_case(&case, 1, &[], &[]).unwrap();
        assert_eq!(report.results.len(), 1 + 9 + 4);
        let error = |config: &str| {
            report
                .results
                .iter()
                .find(|r| r.config == config)
                .and_then(|r| r.quality_vs_exact)
                .unwrap()
                .relative_l2_error
        };
        for result in &report.results {
            println!(
                "{}/{}: quality {:?}",
                result.family, result.config, result.quality_vs_exact
            );
        }
        // More RVQ bits must strictly reduce attention error.
        assert!(error("k3x2-v3x2") < error("k2x2-v2x2"));
        assert!(error("k2x2-v2x2") < error("k1x2-v1x2"));
        for result in &report.results {
            let ceiling = QUALITY_CEILINGS
                .iter()
                .find(|(config, _)| *config == result.config)
                .map(|(_, ceiling)| *ceiling)
                .unwrap_or_else(|| panic!("no ceiling for {}", result.config));
            let quality = result.quality_vs_exact.unwrap().relative_l2_error;
            assert!(
                quality < ceiling,
                "{}/{}: relative L2 {quality} >= ceiling {ceiling}",
                result.family,
                result.config
            );
            let kernel = result.kernel.as_ref().unwrap();
            let group_affine = result.family == "group-affine";
            assert_eq!(kernel.host_staging_inside_attend, group_affine);
            assert_eq!(kernel.threadgroup_barriers_per_kv_token, 0);
            assert_eq!(
                kernel.extra_mlx_ops_per_attend,
                if group_affine { 0 } else { 6 }
            );
            assert!(result.is_compressed(), "{result:?}");
            assert_eq!(result.full_cache_dequantizations_during_attend, Some(0));
            let parity = result.parity_vs_dequantize_oracle.unwrap();
            assert!(
                parity.max_abs_error < 2e-3,
                "{}/{}: {parity:?}",
                result.family,
                result.config
            );
        }
        let incumbent = report.results[0].representation.as_ref().unwrap();
        assert_eq!(
            report.budget_matches[0].budget_bytes,
            incumbent.representation_bytes
        );
        assert_eq!(report.quality_matches.len(), 1);
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["request"]["mask"], "causal");
        assert_eq!(
            json["results"][0]["kernel"]["gpu_family"],
            "conservative-unknown-apple"
        );
    }

    /// SC-20671/SC-20676 detection reused: the cache-owned full-dequantization counter plus the
    /// allocator-measured transient against dense-reconstruction geometry. A deliberately
    /// materializing attention (dequantize the whole cache, then dense SDPA) must trip both,
    /// while every family's compressed route must trip neither.
    #[cfg(target_os = "macos")]
    #[test]
    fn deliberate_dense_reconstruction_is_detected_and_compressed_routes_are_not() {
        use crate::primitives::attention::{sdpa, AttnMask};
        let shape = SyntheticShape {
            batch: 1,
            query_heads: 4,
            kv_heads: 2,
            query_len: 1,
            kv_len: 4096,
            head_dim: 128,
        };
        let case = synthetic_case(shape, 11, PackedAttentionMask::Causal, None);
        let request = &case.request;
        let query = Array::from_slice(&case.query, &[1, 4, 1, 128]);
        query.eval().unwrap();
        let families: Vec<Box<dyn CompressedKvCandidate>> = vec![
            Box::new(GroupAffineCandidate::new(1, 2, 128).unwrap()),
            Box::new(
                RvqKvCandidate::new(
                    RvqConfig {
                        key_bits: 2,
                        value_bits: 2,
                    },
                    1,
                    2,
                    128,
                )
                .unwrap(),
            ),
            Box::new(
                RabitqKvCandidate::new(
                    RabitqConfig {
                        score: RabitqScore::FullPrecisionQuery,
                        magnitude: RabitqMagnitude::Unbiased,
                    },
                    1,
                    2,
                    128,
                )
                .unwrap(),
            ),
        ];
        for mut candidate in families {
            candidate
                .append(&case.keys, &case.values, request.kv_len)
                .unwrap();
            assert_eq!(candidate.preflight(request), CacheRoute::ExperimentalPacked);
            candidate.sync_device().unwrap();
            // Warm: JIT and the group-affine lazy device staging are not attention transients.
            candidate.attend(&query, request).unwrap().eval().unwrap();
            let (_, compressed) =
                probe_materialization(request, || candidate.attend(&query, request)).unwrap();
            assert_eq!(
                compressed.verdict,
                MaterializationVerdict::Compressed,
                "{}: {compressed:?}",
                candidate.family()
            );
            assert_eq!(candidate.full_cache_dequantizations(), 0);

            let (_, dense) = probe_materialization(request, || {
                let (keys, values) = candidate.dequantize_dense_for_oracle()?;
                let keys = Array::from_slice(&keys, &[1, 2, 4096, 128]);
                let values = Array::from_slice(&values, &[1, 2, 4096, 128]);
                sdpa(&query, &keys, &values, request.scale, AttnMask::Causal)
            })
            .unwrap();
            assert_eq!(
                dense.verdict,
                MaterializationVerdict::DenseMaterialized,
                "{}: {dense:?}",
                candidate.family()
            );
            assert_eq!(candidate.full_cache_dequantizations(), 1);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn captured_file_round_trips_through_the_loader_with_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("capture.safetensors");
        let case = synthetic_case(
            SyntheticShape {
                batch: 1,
                query_heads: 2,
                kv_heads: 1,
                query_len: 1,
                kv_len: 8,
                head_dim: 64,
            },
            3,
            PackedAttentionMask::Causal,
            None,
        );
        let q = Array::from_slice(&case.query, &[1, 2, 1, 64])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let k = Array::from_slice(&case.keys, &[1, 1, 8, 64]);
        let v = Array::from_slice(&case.values, &[1, 1, 8, 64]);
        let metadata: std::collections::HashMap<String, String> = [
            ("scale".to_string(), "0.25".to_string()),
            ("mask".into(), "none".into()),
        ]
        .into();
        Array::save_safetensors([("q", &q), ("k", &k), ("v", &v)], &metadata, &path).unwrap();
        let loaded = load_captured_case(&path, None, None).unwrap();
        assert_eq!(loaded.request.scale, 0.25);
        assert_eq!(loaded.request.mask, PackedAttentionMask::None);
        assert_eq!((loaded.request.query_heads, loaded.request.kv_len), (2, 8));
        assert_eq!(loaded.keys, case.keys);
        let CaseSource::Captured {
            sha256, metadata, ..
        } = &loaded.source
        else {
            panic!("captured source")
        };
        assert_eq!(sha256.len(), 64);
        assert_eq!(metadata["mask"], "none");
        let overridden =
            load_captured_case(&path, Some(PackedAttentionMask::Causal), Some(0.5)).unwrap();
        assert_eq!(overridden.request.mask, PackedAttentionMask::Causal);
        assert_eq!(overridden.request.scale, 0.5);
    }
}
