//! Asymmetric RaBitQ KV candidate: packed 1-bit keys, nibble-packed 4-bit values (SC-20677).
//!
//! Keys follow upstream `rabitq_encode`: `y = H(s ⊙ k)/sqrt(D)`, one sign bit per coordinate
//! (`y >= 0 -> 1`, little-endian within a byte) and `mag = L1(y)/D`, so `k̂_rot = mag · sign`.
//! Upstream ships no RaBitQ serving cache and its fused decode takes a single global scalar value
//! codebook, which cannot track per-token value magnitudes. The value side here uses the same
//! rotation, an fp16 per-vector norm and a 16-level Lloyd-Max `N(0,1/d)` codebook over the unit
//! rotated vector, nibble-packed exactly as upstream `rabitq_pack_values` (low nibble = even
//! channel).
//!
//! Two score estimators are compared because upstream's kernel mechanism is not the only
//! credible one:
//! * [`RabitqScore::BinarizedQuery`] — upstream `rabitq_fused_attend`: the query is binarized
//!   too and the score is `(D - 2·popcount(q_bits ^ k_bits)) · L1(q_rot)/D · mag · scale`.
//! * [`RabitqScore::FullPrecisionQuery`] — `<q_rot, sign> · mag · scale` with the fp32 query.
//!
//! Byte layout: key bits `[S,B,Hkv,D/8]` u8, key magnitudes `[S,B,Hkv]` f16, value nibbles
//! `[S,B,Hkv,D/2]` u8, value norms `[S,B,Hkv]` f16.

use half::f16;
use mlx_rs::fast::{MetalKernel, OutputArg};
use mlx_rs::{Array, Dtype};

use super::rvq::mask_template;
use super::{
    candidate_head_dimension_supported, candidate_request_support, decline, mlx_dims,
    token_major_rows, validate_step, CacheGeometry, CandidateAttentionRequest,
    CandidateRepresentation, CompressedKvCandidate, HadamardRotation, ScalarCodebook,
};
use crate::error::{Error, Result};
use crate::primitives::kv_cache::CacheRoute;
use crate::primitives::packed_group_affine_kv::DenseFallbackEvent;

const VERSION: u32 = 1;
const ROTATION_SEED: u64 = 0x5c20_677b;
const VALUE_BITS: u32 = 4;

pub(crate) const MSL_HEADER: &str = "#include <metal_stdlib>\nusing namespace metal;\n";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RabitqScore {
    BinarizedQuery,
    FullPrecisionQuery,
}

impl RabitqScore {
    pub fn label(self) -> &'static str {
        match self {
            Self::BinarizedQuery => "k1-hamming-v4",
            Self::FullPrecisionQuery => "k1-fpq-v4",
        }
    }
    fn template(self) -> i32 {
        match self {
            Self::BinarizedQuery => 0,
            Self::FullPrecisionQuery => 1,
        }
    }
}

pub struct RabitqKvCandidate {
    score: RabitqScore,
    geometry: CacheGeometry,
    rotation: HadamardRotation,
    value_codebook: ScalarCodebook,
    key_bits: Vec<u8>,
    key_mags: Vec<f16>,
    value_nibbles: Vec<u8>,
    value_norms: Vec<f16>,
    constants: Vec<Array>,
    device: Option<Vec<Array>>,
    kernel: Option<MetalKernel>,
    fallback_events: Vec<DenseFallbackEvent>,
    full_cache_dequantizations: usize,
}

impl std::fmt::Debug for RabitqKvCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RabitqKvCandidate")
            .field("score", &self.score)
            .field("geometry", &self.geometry)
            .finish_non_exhaustive()
    }
}

impl RabitqKvCandidate {
    pub fn new(score: RabitqScore, batch: usize, kv_heads: usize, head_dim: usize) -> Result<Self> {
        if batch == 0 || kv_heads == 0 || !candidate_head_dimension_supported(head_dim) {
            return Err(Error::Unsupported(format!(
                "RaBitQ candidate geometry unsupported: {}",
                super::CandidateFallbackReason::UnsupportedHeadDimension.as_str()
            )));
        }
        let rotation = HadamardRotation::new(head_dim, ROTATION_SEED)?;
        let value_codebook = ScalarCodebook::gaussian(VALUE_BITS, head_dim);
        let constants = vec![
            Array::from_slice(rotation.signs(), &[head_dim as i32]),
            Array::from_slice(
                value_codebook.centroids(),
                &[value_codebook.centroids().len() as i32],
            ),
        ];
        Ok(Self {
            score,
            geometry: CacheGeometry {
                batch,
                kv_heads,
                head_dim,
                logical_len: 0,
            },
            rotation,
            value_codebook,
            key_bits: Vec::new(),
            key_mags: Vec::new(),
            value_nibbles: Vec::new(),
            value_norms: Vec::new(),
            constants,
            device: None,
            kernel: None,
            fallback_events: Vec::new(),
            full_cache_dequantizations: 0,
        })
    }

    fn rows(&self) -> usize {
        self.geometry.batch * self.geometry.kv_heads
    }

    fn encode_key(&self, x: &[f32], bits: &mut Vec<u8>, mags: &mut Vec<f16>) -> Result<()> {
        let y = self.rotation.forward(x);
        let mag = f16::from_f32(y.iter().map(|v| v.abs()).sum::<f32>() / y.len() as f32);
        if !mag.is_finite() {
            return Err(Error::Unsupported(
                "RaBitQ key magnitude exceeds f16".into(),
            ));
        }
        for chunk in y.chunks(8) {
            bits.push(
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u8, |byte, (t, v)| byte | (u8::from(*v >= 0.0) << t)),
            );
        }
        mags.push(mag);
        Ok(())
    }

    fn encode_value(&self, x: &[f32], nibbles: &mut Vec<u8>, norms: &mut Vec<f16>) -> Result<()> {
        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let stored = f16::from_f32(norm);
        if !stored.is_finite() {
            return Err(Error::Unsupported("RaBitQ value norm exceeds f16".into()));
        }
        let inverse = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let y = self.rotation.forward(x);
        for pair in y.chunks(2) {
            let low = self.value_codebook.quantize(pair[0] * inverse) as u8;
            let high = self.value_codebook.quantize(pair[1] * inverse) as u8;
            nibbles.push(low | (high << 4));
        }
        norms.push(stored);
        Ok(())
    }

    fn rotated_key(&self, index: usize) -> Vec<f32> {
        let nb = self.geometry.head_dim / 8;
        let mag = self.key_mags[index].to_f32();
        (0..self.geometry.head_dim)
            .map(|d| {
                if (self.key_bits[index * nb + d / 8] >> (d % 8)) & 1 == 1 {
                    mag
                } else {
                    -mag
                }
            })
            .collect()
    }

    fn rotated_value(&self, index: usize) -> Vec<f32> {
        let half = self.geometry.head_dim / 2;
        let norm = self.value_norms[index].to_f32();
        (0..self.geometry.head_dim)
            .map(|d| {
                let byte = self.value_nibbles[index * half + d / 2];
                let code = (byte >> ((d % 2) * 4)) & 0xf;
                self.value_codebook.value(u32::from(code)) * norm
            })
            .collect()
    }

    fn kernel(&mut self) -> Result<&MetalKernel> {
        if self.kernel.is_none() {
            self.kernel = Some(MetalKernel::with_options(
                "sc20677_rabitq_online",
                &[
                    "q",
                    "params",
                    "k_bits",
                    "k_mag",
                    "v_nibbles",
                    "v_norm",
                    "v_cents",
                ],
                &["out"],
                RABITQ_MSL,
                MSL_HEADER,
                true,
                false,
            )?);
        }
        Ok(self.kernel.as_ref().expect("initialized"))
    }
}

impl CompressedKvCandidate for RabitqKvCandidate {
    fn family(&self) -> &'static str {
        "rabitq"
    }

    fn config(&self) -> String {
        self.score.label().into()
    }

    fn representation(&self) -> CandidateRepresentation {
        let shared_constant_bytes = self.constants.iter().map(Array::nbytes).sum::<usize>();
        let key_code_bytes = self.key_bits.len();
        let key_metadata_bytes = self.key_mags.len() * 2;
        let value_code_bytes = self.value_nibbles.len();
        let value_metadata_bytes = self.value_norms.len() * 2;
        CandidateRepresentation {
            family: self.family().into(),
            config: self.config(),
            identity: format!("sc-20677-rabitq-{}", self.score.label()),
            version: VERSION,
            batch: self.geometry.batch,
            kv_heads: self.geometry.kv_heads,
            head_dim: self.geometry.head_dim,
            logical_len: self.geometry.logical_len,
            layout: "token-major K bits [S,B,Hkv,D/8] u8 + mag f16; V nibbles [S,B,Hkv,D/2] u8 + norm f16"
                .into(),
            key_code_bytes,
            key_metadata_bytes,
            value_code_bytes,
            value_metadata_bytes,
            dense_staging_bytes: 0,
            shared_constant_bytes,
            representation_bytes: key_code_bytes
                + key_metadata_bytes
                + value_code_bytes
                + value_metadata_bytes
                + shared_constant_bytes,
            host_allocated_bytes: self.key_bits.capacity()
                + self.key_mags.capacity() * 2
                + self.value_nibbles.capacity()
                + self.value_norms.capacity() * 2,
            device_bytes: self
                .device
                .as_ref()
                .map_or(0, |arrays| arrays.iter().map(Array::nbytes).sum::<usize>())
                + shared_constant_bytes,
        }
    }

    fn preflight(&mut self, request: &CandidateAttentionRequest) -> CacheRoute {
        match candidate_request_support(request, self.geometry) {
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
        validate_step(self.geometry, keys, values, step)?;
        let (mut bits, mut mags, mut nibbles, mut norms) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for row in token_major_rows(self.geometry, keys, step) {
            self.encode_key(row, &mut bits, &mut mags)?;
        }
        for row in token_major_rows(self.geometry, values, step) {
            self.encode_value(row, &mut nibbles, &mut norms)?;
        }
        self.key_bits.extend(bits);
        self.key_mags.extend(mags);
        self.value_nibbles.extend(nibbles);
        self.value_norms.extend(norms);
        self.geometry.logical_len += step;
        self.device = None;
        Ok(())
    }

    fn trim(&mut self, len: usize) -> Result<()> {
        if len > self.geometry.logical_len {
            return Err(Error::Config("trim exceeds logical length".into()));
        }
        let vectors = len * self.rows();
        let dim = self.geometry.head_dim;
        self.key_bits.truncate(vectors * dim / 8);
        self.key_mags.truncate(vectors);
        self.value_nibbles.truncate(vectors * dim / 2);
        self.value_norms.truncate(vectors);
        self.geometry.logical_len = len;
        self.device = None;
        Ok(())
    }

    fn sync_device(&mut self) -> Result<()> {
        if self.device.is_some() {
            return Ok(());
        }
        let (s, b, h, d) = (
            self.geometry.logical_len,
            self.geometry.batch,
            self.geometry.kv_heads,
            self.geometry.head_dim,
        );
        if s == 0 {
            return Err(Error::Config("cannot upload an empty RaBitQ cache".into()));
        }
        let arrays = vec![
            Array::from_slice(&self.key_bits, &mlx_dims(&[s, b, h, d / 8])?),
            Array::from_slice(&self.key_mags, &mlx_dims(&[s, b, h])?),
            Array::from_slice(&self.value_nibbles, &mlx_dims(&[s, b, h, d / 2])?),
            Array::from_slice(&self.value_norms, &mlx_dims(&[s, b, h])?),
        ];
        for array in arrays.iter().chain(&self.constants) {
            array.eval()?;
        }
        self.device = Some(arrays);
        Ok(())
    }

    fn attend(&mut self, query: &Array, request: &CandidateAttentionRequest) -> Result<Array> {
        if let Err(reason) = candidate_request_support(request, self.geometry) {
            return Err(Error::Unsupported(format!(
                "RaBitQ attend without passing preflight: {}",
                reason.as_str()
            )));
        }
        if query.shape()
            != mlx_dims(&[
                request.batch,
                request.query_heads,
                request.query_len,
                request.head_dim,
            ])?
            .as_slice()
        {
            return Err(Error::Config(
                "RaBitQ query shape differs from request".into(),
            ));
        }
        self.sync_device()?;
        let (mask_mode, window) = mask_template(request.mask)?;
        let dim = request.head_dim;
        let signs = self.constants[0].clone();
        let codebook = self.constants[1].clone();
        let rotated = self.rotation.forward_mlx(query, &signs)?;
        let params = Array::from_slice(&[request.scale], &[1]);
        let queries = request.batch * request.query_heads * request.query_len;
        let grid_x = i32::try_from(queries * 32)
            .map_err(|_| Error::Unsupported("RaBitQ Metal grid exceeds i32".into()))?;
        let device = self.device.clone().expect("synced");
        let score_mode = self.score.template();
        let out = self
            .kernel()?
            .apply()
            .input(&rotated)
            .input(&params)
            .inputs(device.iter())
            .input(&codebook)
            .output(OutputArg {
                shape: rotated.shape().to_vec(),
                dtype: Dtype::Float32,
            })
            .grid(grid_x, 1, 1)
            .thread_group(32, 1, 1)
            .template_arg("D", dim as i32)
            .template_arg("VPT", (dim / 32) as i32)
            .template_arg("NB", (dim / 8) as i32)
            .template_arg("SCORE_MODE", score_mode)
            .template_arg("MASK_MODE", mask_mode)
            .template_arg("WINDOW", window)
            .run()?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("RaBitQ kernel returned no output".into()))?;
        Ok(self
            .rotation
            .inverse_mlx(&out, &signs)?
            .as_dtype(query.dtype())?)
    }

    fn dequantize_dense_for_oracle(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        self.full_cache_dequantizations += 1;
        let (rows, len, dim) = (
            self.rows(),
            self.geometry.logical_len,
            self.geometry.head_dim,
        );
        let mut keys = vec![0.0f32; rows * len * dim];
        let mut values = vec![0.0f32; rows * len * dim];
        for token in 0..len {
            for row in 0..rows {
                let index = token * rows + row;
                let target = (row * len + token) * dim;
                keys[target..target + dim]
                    .copy_from_slice(&self.rotation.inverse(&self.rotated_key(index)));
                values[target..target + dim]
                    .copy_from_slice(&self.rotation.inverse(&self.rotated_value(index)));
            }
        }
        Ok((keys, values))
    }

    /// The binarized estimator scores against `q̂ = R^T(L1(q_rot)/D · sign(q_rot))`.
    fn oracle_query(&self, query: &[f32], request: &CandidateAttentionRequest) -> Vec<f32> {
        match self.score {
            RabitqScore::FullPrecisionQuery => query.to_vec(),
            RabitqScore::BinarizedQuery => query
                .chunks(request.head_dim)
                .flat_map(|row| {
                    let rotated = self.rotation.forward(row);
                    let scale = rotated.iter().map(|v| v.abs()).sum::<f32>() / rotated.len() as f32;
                    let estimate: Vec<f32> = rotated
                        .iter()
                        .map(|v| if *v >= 0.0 { scale } else { -scale })
                        .collect();
                    self.rotation.inverse(&estimate)
                })
                .collect(),
        }
    }

    fn full_cache_dequantizations(&self) -> usize {
        self.full_cache_dequantizations
    }

    fn fallback_events(&self) -> &[DenseFallbackEvent] {
        &self.fallback_events
    }
}

/// Same one-SIMD-group-per-query geometry as the RVQ and SC-20676 readers. In binarized mode lane
/// `l < D/8` owns query sign byte `l`; `simd_sum(popcount(...))` is the Hamming distance.
const RABITQ_MSL: &str = r#"
    const uint lane = thread_position_in_threadgroup.x;
    const uint query = threadgroup_position_in_grid.x;
    const uint SQ = q_shape[2];
    const uint HQ = q_shape[1];
    const uint S = k_bits_shape[0];
    const uint B = k_bits_shape[1];
    const uint HKV = k_bits_shape[2];
    const uint qi = query % SQ;
    const uint qh = (query / SQ) % HQ;
    const uint b = query / (SQ * HQ);
    const uint kh = qh / (HQ / HKV);
    const float scale = params[0];
    const uint q_base = ((b * HQ + qh) * SQ + qi) * D;

    float qv[VPT];
    float acc[VPT];
    float l1 = 0.0f;
    for (uint o = 0; o < VPT; ++o) {
        qv[o] = q[q_base + lane + o * 32];
        l1 += fabs(qv[o]);
        acc[o] = 0.0f;
    }
    const float q_scale = simd_sum(l1) / float(D);
    uint q_byte = 0;
    if (lane < NB) {
        for (uint t = 0; t < 8; ++t) {
            q_byte |= (q[q_base + lane * 8 + t] >= 0.0f ? 1u : 0u) << t;
        }
    }
    float running_max = -INFINITY;
    float running_norm = 0.0f;
    const uint qpos = S - SQ + qi;
    for (uint s = 0; s < S; ++s) {
        if ((MASK_MODE == 1 && s > qpos) ||
            (MASK_MODE == 2 && (s > qpos || s + WINDOW <= qpos))) continue;
        const uint row = (s * B + b) * HKV + kh;
        float estimate;
        if (SCORE_MODE == 0) {
            const uint ham = lane < NB ? popcount(q_byte ^ uint(k_bits[row * NB + lane])) : 0u;
            estimate = (float(D) - 2.0f * float(simd_sum(ham))) * q_scale;
        } else {
            float partial = 0.0f;
            for (uint o = 0; o < VPT; ++o) {
                const uint d = lane + o * 32;
                const bool positive = ((k_bits[row * NB + d / 8] >> (d % 8)) & 1u) != 0u;
                partial += positive ? qv[o] : -qv[o];
            }
            estimate = simd_sum(partial);
        }
        const float score = estimate * float(k_mag[row]) * scale;
        const float next_max = max(running_max, score);
        const float rescale = exp(running_max - next_max);
        const float weight = exp(score - next_max);
        running_norm = running_norm * rescale + weight;
        running_max = next_max;
        const float vnorm = float(v_norm[row]);
        for (uint o = 0; o < VPT; ++o) {
            const uint d = lane + o * 32;
            const uint code = (uint(v_nibbles[row * (D / 2) + d / 2]) >> ((d & 1u) * 4u)) & 15u;
            acc[o] = acc[o] * rescale + weight * v_cents[code] * vnorm;
        }
    }
    for (uint o = 0; o < VPT; ++o) {
        out[q_base + lane + o * 32] = running_norm > 0.0f ? acc[o] / running_norm : 0.0f;
    }
"#;

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::{dense_attention_oracle, CandidateFallbackReason};
    use super::*;
    use crate::primitives::kv_cache::PackedAttentionMask;

    #[test]
    fn byte_accounting_is_exact_per_token() {
        for dim in [64, 128, 256] {
            let (batch, heads, len) = (2, 3, 7);
            let n = batch * heads * len * dim;
            let mut cache =
                RabitqKvCandidate::new(RabitqScore::FullPrecisionQuery, batch, heads, dim).unwrap();
            cache.append(&gaussian(1, n), &gaussian(2, n), len).unwrap();
            let rep = cache.representation();
            let vectors = batch * heads * len;
            assert_eq!(rep.key_code_bytes, vectors * dim / 8);
            assert_eq!(rep.key_metadata_bytes, vectors * 2);
            assert_eq!(rep.value_code_bytes, vectors * dim / 2);
            assert_eq!(rep.value_metadata_bytes, vectors * 2);
            assert_eq!(rep.shared_constant_bytes, dim * 4 + 16 * 4);
            assert_eq!(
                rep.representation_bytes,
                vectors * (dim / 8 + 2 + dim / 2 + 2) + dim * 4 + 64
            );
        }
    }

    #[test]
    fn key_bits_follow_upstream_sign_and_little_endian_convention() {
        let mut cache = RabitqKvCandidate::new(RabitqScore::BinarizedQuery, 1, 1, 64).unwrap();
        let key = gaussian(5, 64);
        cache.append(&key, &key, 1).unwrap();
        let rotated = cache.rotation.forward(&key);
        for (d, value) in rotated.iter().enumerate() {
            let bit = (cache.key_bits[d / 8] >> (d % 8)) & 1;
            assert_eq!(bit == 1, *value >= 0.0, "channel {d}");
        }
        let l1 = rotated.iter().map(|v| v.abs()).sum::<f32>() / 64.0;
        assert_eq!(cache.key_mags[0], f16::from_f32(l1));
    }

    #[test]
    fn preflight_rejects_additive_and_unsupported_geometry_without_mutation() {
        let mut cache = RabitqKvCandidate::new(RabitqScore::FullPrecisionQuery, 1, 2, 64).unwrap();
        cache
            .append(&gaussian(3, 2 * 3 * 64), &gaussian(4, 2 * 3 * 64), 3)
            .unwrap();
        let before = cache.representation();
        let additive = CandidateAttentionRequest {
            mask: PackedAttentionMask::Additive,
            ..request(1, 2, 2, 1, 3, 64)
        };
        assert_eq!(
            cache.preflight(&additive),
            CacheRoute::DenseFallback {
                reason: CandidateFallbackReason::AdditiveMask.as_str().into()
            }
        );
        assert_eq!(
            cache.preflight(&request(1, 3, 2, 1, 3, 64)),
            CacheRoute::DenseFallback {
                reason: CandidateFallbackReason::UnsupportedGqaRatio.as_str().into()
            }
        );
        assert_eq!(cache.representation(), before);
        assert_eq!(cache.fallback_events().len(), 2);
        assert!(RabitqKvCandidate::new(RabitqScore::BinarizedQuery, 1, 2, 512).is_err());
    }

    #[cfg(target_os = "macos")]
    fn metal_parity(score: RabitqScore, request: CandidateAttentionRequest) {
        let n = request.batch * request.kv_heads * request.kv_len * request.head_dim;
        let keys = gaussian(31, n);
        let values = gaussian(32, n);
        let query_host = gaussian(
            33,
            request.batch * request.query_heads * request.query_len * request.head_dim,
        );
        let mut cache =
            RabitqKvCandidate::new(score, request.batch, request.kv_heads, request.head_dim)
                .unwrap();
        cache.append(&keys, &values, request.kv_len).unwrap();
        assert_eq!(cache.preflight(&request), CacheRoute::ExperimentalPacked);
        let query = Array::from_slice(
            &query_host,
            &mlx_dims(&[
                request.batch,
                request.query_heads,
                request.query_len,
                request.head_dim,
            ])
            .unwrap(),
        );
        let actual = cache
            .attend(&query, &request)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(cache.full_cache_dequantizations(), 0);
        let (dense_keys, dense_values) = cache.dequantize_dense_for_oracle().unwrap();
        let oracle_query = cache.oracle_query(&query_host, &request);
        let expected =
            dense_attention_oracle(&request, &oracle_query, &dense_keys, &dense_values).unwrap();
        let diff = max_abs_diff(&actual, &expected);
        assert!(diff < 2e-4, "{score:?} {request:?}: max abs diff {diff}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matches_fp32_dequantize_then_attend_oracle_for_both_estimators() {
        for score in [RabitqScore::BinarizedQuery, RabitqScore::FullPrecisionQuery] {
            metal_parity(score, request(2, 4, 2, 3, 29, 64));
            metal_parity(score, request(1, 8, 2, 1, 17, 128));
            metal_parity(
                score,
                CandidateAttentionRequest {
                    mask: PackedAttentionMask::SlidingWindow(4),
                    scale: 0.2,
                    ..request(1, 2, 1, 2, 11, 256)
                },
            );
            metal_parity(
                score,
                CandidateAttentionRequest {
                    mask: PackedAttentionMask::None,
                    ..request(1, 2, 2, 5, 9, 64)
                },
            );
        }
    }
}
