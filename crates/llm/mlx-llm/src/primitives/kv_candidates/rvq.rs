//! Two-stage TurboQuant-style packed RVQ KV candidate (SC-20677).
//!
//! Upstream `TurboQuantRVQKVCache` packs two `b`-bit key index streams plus an fp16 norm, but
//! rebuilds all of K on every fetch and keeps fp16 V; its standalone fused kernel instead reads
//! *unpacked* byte indices, ignores the key norm, and needs a separate value codebook (SC-20672).
//! Neither satisfies E3. This adapter keeps the upstream quantizer — randomized Hadamard
//! rotation, a Lloyd-Max `N(0,1/d)` stage-1 codebook and a Laplacian stage-2 residual codebook
//! over unit-norm rotated vectors — and applies it to both K and V with genuinely packed `u32`
//! streams. Attention decodes `(c1[i1] + c2[i2]) · norm` per element inside an online softmax in
//! the rotated domain, so the only dense tensors are the rotated query and output.
//!
//! Device layout is token-major so append is a suffix write: codes `[S,B,Hkv,W]` (`u32`,
//! `W = ceil(D / floor(32/b))`), norms `[S,B,Hkv]` (`f16`).

use half::f16;
use mlx_rs::fast::{MetalKernel, OutputArg};
use mlx_rs::{Array, Dtype};

use super::{
    candidate_head_dimension_supported, candidate_request_support, decline, mlx_dims, pack_words,
    token_major_rows, unpack_word_code, validate_step, CacheGeometry, CandidateAttentionRequest,
    CandidateRepresentation, CompressedKvCandidate, HadamardRotation, ScalarCodebook,
};
use crate::error::{Error, Result};
use crate::primitives::kv_cache::{CacheRoute, PackedAttentionMask};
use crate::primitives::packed_group_affine_kv::DenseFallbackEvent;

const VERSION: u32 = 1;
const ROTATION_SEED: u64 = 0x5c20_677a;

/// Bits per stage for keys and values; total code bits per coordinate are `2·bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RvqConfig {
    pub key_bits: u32,
    pub value_bits: u32,
}

impl RvqConfig {
    pub fn label(self) -> String {
        format!("k{}x2-v{}x2", self.key_bits, self.value_bits)
    }

    fn validate(self) -> Result<()> {
        if !(1..=4).contains(&self.key_bits) || !(1..=4).contains(&self.value_bits) {
            return Err(Error::Unsupported(
                "RVQ bits per stage must be 1..=4".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn words_per_vector(head_dim: usize, bits: u32) -> usize {
    head_dim.div_ceil((32 / bits) as usize)
}

/// Two packed code streams plus one norm per cached vector.
#[derive(Clone, Debug, Default)]
struct RvqStream {
    stage1: Vec<u32>,
    stage2: Vec<u32>,
    norms: Vec<f16>,
}

impl RvqStream {
    fn truncate(&mut self, vectors: usize, words: usize) {
        self.stage1.truncate(vectors * words);
        self.stage2.truncate(vectors * words);
        self.norms.truncate(vectors);
    }
    fn code_bytes(&self) -> usize {
        (self.stage1.len() + self.stage2.len()) * 4
    }
    fn metadata_bytes(&self) -> usize {
        self.norms.len() * 2
    }
    fn allocated_bytes(&self) -> usize {
        (self.stage1.capacity() + self.stage2.capacity()) * 4 + self.norms.capacity() * 2
    }
}

struct RvqCodec {
    bits: u32,
    words: usize,
    stage1: ScalarCodebook,
    stage2: ScalarCodebook,
}

impl RvqCodec {
    fn new(bits: u32, head_dim: usize) -> Self {
        Self {
            bits,
            words: words_per_vector(head_dim, bits),
            stage1: ScalarCodebook::gaussian(bits, head_dim),
            stage2: ScalarCodebook::laplacian_residual(bits, head_dim),
        }
    }

    fn encode(&self, rotation: &HadamardRotation, x: &[f32], out: &mut RvqStream) -> Result<()> {
        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let stored = f16::from_f32(norm);
        if !stored.is_finite() {
            return Err(Error::Unsupported("RVQ vector norm exceeds f16".into()));
        }
        let rotated = rotation.forward(x);
        let inverse = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let mut first = Vec::with_capacity(x.len());
        let mut second = Vec::with_capacity(x.len());
        for y in rotated {
            let unit = y * inverse;
            let code1 = self.stage1.quantize(unit);
            first.push(code1);
            second.push(self.stage2.quantize(unit - self.stage1.value(code1)));
        }
        pack_words(&first, self.bits, &mut out.stage1);
        pack_words(&second, self.bits, &mut out.stage2);
        out.norms.push(stored);
        Ok(())
    }

    /// Rotated-domain reconstruction of vector `index`.
    fn decode_rotated(&self, stream: &RvqStream, index: usize, head_dim: usize) -> Vec<f32> {
        let words = &stream.stage1[index * self.words..(index + 1) * self.words];
        let words2 = &stream.stage2[index * self.words..(index + 1) * self.words];
        let norm = stream.norms[index].to_f32();
        (0..head_dim)
            .map(|d| {
                (self.stage1.value(unpack_word_code(words, d, self.bits))
                    + self.stage2.value(unpack_word_code(words2, d, self.bits)))
                    * norm
            })
            .collect()
    }
}

struct RvqDevice {
    arrays: Vec<Array>,
}

/// Minimal packed-RVQ K/V representation for one attention layer.
pub struct RvqKvCandidate {
    config: RvqConfig,
    geometry: CacheGeometry,
    rotation: HadamardRotation,
    key_codec: RvqCodec,
    value_codec: RvqCodec,
    keys: RvqStream,
    values: RvqStream,
    constants: Vec<Array>,
    device: Option<RvqDevice>,
    kernel: Option<MetalKernel>,
    fallback_events: Vec<DenseFallbackEvent>,
    full_cache_dequantizations: usize,
}

impl std::fmt::Debug for RvqKvCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RvqKvCandidate")
            .field("config", &self.config)
            .field("geometry", &self.geometry)
            .finish_non_exhaustive()
    }
}

impl RvqKvCandidate {
    pub fn new(config: RvqConfig, batch: usize, kv_heads: usize, head_dim: usize) -> Result<Self> {
        config.validate()?;
        if batch == 0 || kv_heads == 0 || !candidate_head_dimension_supported(head_dim) {
            return Err(Error::Unsupported(format!(
                "RVQ candidate geometry unsupported: {}",
                super::CandidateFallbackReason::UnsupportedHeadDimension.as_str()
            )));
        }
        let rotation = HadamardRotation::new(head_dim, ROTATION_SEED)?;
        let key_codec = RvqCodec::new(config.key_bits, head_dim);
        let value_codec = RvqCodec::new(config.value_bits, head_dim);
        let constants = vec![
            Array::from_slice(rotation.signs(), &[head_dim as i32]),
            codebook_array(&key_codec.stage1),
            codebook_array(&key_codec.stage2),
            codebook_array(&value_codec.stage1),
            codebook_array(&value_codec.stage2),
        ];
        Ok(Self {
            config,
            geometry: CacheGeometry {
                batch,
                kv_heads,
                head_dim,
                logical_len: 0,
            },
            rotation,
            key_codec,
            value_codec,
            keys: RvqStream::default(),
            values: RvqStream::default(),
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

    fn kernel(&mut self) -> Result<&MetalKernel> {
        if self.kernel.is_none() {
            self.kernel = Some(MetalKernel::with_options(
                "sc20677_rvq_online",
                &[
                    "q", "params", "k_codes1", "k_codes2", "k_norm", "v_codes1", "v_codes2",
                    "v_norm", "kc1", "kc2", "vc1", "vc2",
                ],
                &["out"],
                RVQ_MSL,
                super::rabitq::MSL_HEADER,
                true,
                false,
            )?);
        }
        Ok(self.kernel.as_ref().expect("initialized"))
    }
}

fn codebook_array(book: &ScalarCodebook) -> Array {
    Array::from_slice(book.centroids(), &[book.centroids().len() as i32])
}

impl CompressedKvCandidate for RvqKvCandidate {
    fn family(&self) -> &'static str {
        "packed-rvq"
    }

    fn config(&self) -> String {
        self.config.label()
    }

    fn representation(&self) -> CandidateRepresentation {
        let shared_constant_bytes = self.constants.iter().map(Array::nbytes).sum::<usize>();
        let payload = self.keys.code_bytes()
            + self.keys.metadata_bytes()
            + self.values.code_bytes()
            + self.values.metadata_bytes();
        CandidateRepresentation {
            family: self.family().into(),
            config: self.config(),
            identity: format!("sc-20677-packed-rvq-{}", self.config.label()),
            version: VERSION,
            batch: self.geometry.batch,
            kv_heads: self.geometry.kv_heads,
            head_dim: self.geometry.head_dim,
            logical_len: self.geometry.logical_len,
            layout: "token-major codes [S,B,Hkv,W] u32 x2 stages; norms [S,B,Hkv] f16; K and V"
                .into(),
            key_code_bytes: self.keys.code_bytes(),
            key_metadata_bytes: self.keys.metadata_bytes(),
            value_code_bytes: self.values.code_bytes(),
            value_metadata_bytes: self.values.metadata_bytes(),
            dense_staging_bytes: 0,
            shared_constant_bytes,
            representation_bytes: payload + shared_constant_bytes,
            host_allocated_bytes: self.keys.allocated_bytes() + self.values.allocated_bytes(),
            device_bytes: self.device.as_ref().map_or(0, |device| {
                device.arrays.iter().map(Array::nbytes).sum::<usize>()
            }) + shared_constant_bytes,
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
        // Stage into scratch so an encode failure leaves the cache unchanged.
        let mut staged_keys = RvqStream::default();
        let mut staged_values = RvqStream::default();
        for row in token_major_rows(self.geometry, keys, step) {
            self.key_codec
                .encode(&self.rotation, row, &mut staged_keys)?;
        }
        for row in token_major_rows(self.geometry, values, step) {
            self.value_codec
                .encode(&self.rotation, row, &mut staged_values)?;
        }
        for (target, staged) in [
            (&mut self.keys, staged_keys),
            (&mut self.values, staged_values),
        ] {
            target.stage1.extend(staged.stage1);
            target.stage2.extend(staged.stage2);
            target.norms.extend(staged.norms);
        }
        self.geometry.logical_len += step;
        self.device = None;
        Ok(())
    }

    fn trim(&mut self, len: usize) -> Result<()> {
        if len > self.geometry.logical_len {
            return Err(Error::Config("trim exceeds logical length".into()));
        }
        let vectors = len * self.rows();
        self.keys.truncate(vectors, self.key_codec.words);
        self.values.truncate(vectors, self.value_codec.words);
        self.geometry.logical_len = len;
        self.device = None;
        Ok(())
    }

    fn sync_device(&mut self) -> Result<()> {
        if self.device.is_some() {
            return Ok(());
        }
        let (s, b, h) = (
            self.geometry.logical_len,
            self.geometry.batch,
            self.geometry.kv_heads,
        );
        if s == 0 {
            return Err(Error::Config("cannot upload an empty RVQ cache".into()));
        }
        let key_shape = mlx_dims(&[s, b, h, self.key_codec.words])?;
        let value_shape = mlx_dims(&[s, b, h, self.value_codec.words])?;
        let norm_shape = mlx_dims(&[s, b, h])?;
        let arrays = vec![
            Array::from_slice(&self.keys.stage1, &key_shape),
            Array::from_slice(&self.keys.stage2, &key_shape),
            Array::from_slice(&self.keys.norms, &norm_shape),
            Array::from_slice(&self.values.stage1, &value_shape),
            Array::from_slice(&self.values.stage2, &value_shape),
            Array::from_slice(&self.values.norms, &norm_shape),
        ];
        for array in arrays.iter().chain(&self.constants) {
            array.eval()?;
        }
        self.device = Some(RvqDevice { arrays });
        Ok(())
    }

    fn attend(&mut self, query: &Array, request: &CandidateAttentionRequest) -> Result<Array> {
        if let Err(reason) = candidate_request_support(request, self.geometry) {
            return Err(Error::Unsupported(format!(
                "RVQ attend without passing preflight: {}",
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
            return Err(Error::Config("RVQ query shape differs from request".into()));
        }
        self.sync_device()?;
        let (mask_mode, window) = mask_template(request.mask, request.kv_len)?;
        let dim = request.head_dim;
        let signs = self.constants[0].clone();
        let rotated = self.rotation.forward_mlx(query, &signs)?;
        let params = Array::from_slice(&[request.scale], &[1]);
        let (key_bits, value_bits) = (self.config.key_bits, self.config.value_bits);
        let (key_words, value_words) = (self.key_codec.words, self.value_codec.words);
        let queries = request.batch * request.query_heads * request.query_len;
        let grid_x = i32::try_from(queries * 32)
            .map_err(|_| Error::Unsupported("RVQ Metal grid exceeds i32".into()))?;
        let device = self.device.as_ref().expect("synced").arrays.clone();
        let constants = self.constants.clone();
        let out = self
            .kernel()?
            .apply()
            .input(&rotated)
            .input(&params)
            .inputs(device.iter())
            .inputs(constants[1..].iter())
            .output(OutputArg {
                shape: rotated.shape().to_vec(),
                dtype: Dtype::Float32,
            })
            .grid(grid_x, 1, 1)
            .thread_group(32, 1, 1)
            .template_arg("D", dim as i32)
            .template_arg("VPT", (dim / 32) as i32)
            .template_arg("KB", key_bits as i32)
            .template_arg("VB", value_bits as i32)
            .template_arg("KW", key_words as i32)
            .template_arg("VW", value_words as i32)
            .template_arg("MASK_MODE", mask_mode)
            .template_arg("WINDOW", window)
            .run()?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("RVQ kernel returned no output".into()))?;
        Ok(self
            .rotation
            .inverse_mlx(&out, &signs)?
            .as_dtype(query.dtype())?)
    }

    fn kernel_profile(&self, _request: &super::CandidateAttentionRequest) -> super::KernelProfile {
        super::rotated_kernel_profile("sc20677_rvq_online")
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
                let k = self
                    .rotation
                    .inverse(&self.key_codec.decode_rotated(&self.keys, index, dim));
                let v = self.rotation.inverse(&self.value_codec.decode_rotated(
                    &self.values,
                    index,
                    dim,
                ));
                keys[target..target + dim].copy_from_slice(&k);
                values[target..target + dim].copy_from_slice(&v);
            }
        }
        Ok((keys, values))
    }

    fn full_cache_dequantizations(&self) -> usize {
        self.full_cache_dequantizations
    }

    fn fallback_events(&self) -> &[DenseFallbackEvent] {
        &self.fallback_events
    }
}

/// A window of at least `kv_len` keys admits exactly the causal set, so it is clamped to
/// `kv_len` (already validated to fit MLX `i32`) instead of failing after preflight accepted it.
pub(crate) fn mask_template(mask: PackedAttentionMask, kv_len: usize) -> Result<(i32, i32)> {
    match mask {
        PackedAttentionMask::None => Ok((0, 0)),
        PackedAttentionMask::Causal => Ok((1, 0)),
        PackedAttentionMask::SlidingWindow(window) if window > 0 => Ok((
            2,
            i32::try_from(window.min(kv_len))
                .map_err(|_| Error::Unsupported("sliding window exceeds i32".into()))?,
        )),
        PackedAttentionMask::SlidingWindow(_) => {
            Err(Error::Unsupported("empty sliding window".into()))
        }
        PackedAttentionMask::Additive => Err(Error::Unsupported(
            "additive mask requires dense fallback".into(),
        )),
    }
}

/// One 32-lane SIMD group per query row (the SC-20676 conservative geometry, so timing compares
/// representations rather than tuning). Lane `l` owns channels `l + 32·o`.
const RVQ_MSL: &str = r#"
    const uint lane = thread_position_in_threadgroup.x;
    const uint query = threadgroup_position_in_grid.x;
    const uint SQ = q_shape[2];
    const uint HQ = q_shape[1];
    const uint S = k_codes1_shape[0];
    const uint B = k_codes1_shape[1];
    const uint HKV = k_codes1_shape[2];
    const uint qi = query % SQ;
    const uint qh = (query / SQ) % HQ;
    const uint b = query / (SQ * HQ);
    const uint kh = qh / (HQ / HKV);
    const uint K_EPW = 32 / KB;
    const uint V_EPW = 32 / VB;
    const uint K_MASK = (1u << KB) - 1u;
    const uint V_MASK = (1u << VB) - 1u;
    const float scale = params[0];
    const uint q_base = ((b * HQ + qh) * SQ + qi) * D;

    float qv[VPT];
    float acc[VPT];
    for (uint o = 0; o < VPT; ++o) {
        qv[o] = q[q_base + lane + o * 32];
        acc[o] = 0.0f;
    }
    float running_max = -INFINITY;
    float running_norm = 0.0f;
    const uint qpos = S - SQ + qi;
    for (uint s = 0; s < S; ++s) {
        if ((MASK_MODE == 1 && s > qpos) ||
            (MASK_MODE == 2 && (s > qpos || s + WINDOW <= qpos))) continue;
        const uint row = (s * B + b) * HKV + kh;
        float partial = 0.0f;
        for (uint o = 0; o < VPT; ++o) {
            const uint d = lane + o * 32;
            const uint c1 = (k_codes1[row * KW + d / K_EPW] >> ((d % K_EPW) * KB)) & K_MASK;
            const uint c2 = (k_codes2[row * KW + d / K_EPW] >> ((d % K_EPW) * KB)) & K_MASK;
            partial += qv[o] * (kc1[c1] + kc2[c2]);
        }
        const float score = simd_sum(partial) * float(k_norm[row]) * scale;
        const float next_max = max(running_max, score);
        const float rescale = exp(running_max - next_max);
        const float weight = exp(score - next_max);
        running_norm = running_norm * rescale + weight;
        running_max = next_max;
        const float vnorm = float(v_norm[row]);
        for (uint o = 0; o < VPT; ++o) {
            const uint d = lane + o * 32;
            const uint c1 = (v_codes1[row * VW + d / V_EPW] >> ((d % V_EPW) * VB)) & V_MASK;
            const uint c2 = (v_codes2[row * VW + d / V_EPW] >> ((d % V_EPW) * VB)) & V_MASK;
            acc[o] = acc[o] * rescale + weight * (vc1[c1] + vc2[c2]) * vnorm;
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

    fn filled(
        config: RvqConfig,
        batch: usize,
        kv_heads: usize,
        len: usize,
        dim: usize,
    ) -> (RvqKvCandidate, Vec<f32>, Vec<f32>) {
        let n = batch * kv_heads * len * dim;
        let keys = gaussian(11, n);
        let values = gaussian(12, n);
        let mut cache = RvqKvCandidate::new(config, batch, kv_heads, dim).unwrap();
        cache.append(&keys, &values, len).unwrap();
        (cache, keys, values)
    }

    #[test]
    fn byte_accounting_is_exact_per_token_and_matches_buffers() {
        for (kb, vb, dim) in [(1, 1, 64), (2, 1, 128), (3, 2, 128), (2, 2, 256)] {
            let config = RvqConfig {
                key_bits: kb,
                value_bits: vb,
            };
            let (batch, heads, len) = (2, 3, 5);
            let (cache, _, _) = filled(config, batch, heads, len, dim);
            let rep = cache.representation();
            let vectors = batch * heads * len;
            let key_words = dim.div_ceil((32 / kb) as usize);
            let value_words = dim.div_ceil((32 / vb) as usize);
            assert_eq!(rep.key_code_bytes, vectors * 2 * key_words * 4);
            assert_eq!(rep.value_code_bytes, vectors * 2 * value_words * 4);
            assert_eq!(rep.key_metadata_bytes, vectors * 2);
            assert_eq!(rep.value_metadata_bytes, vectors * 2);
            let constants = dim * 4 + ((2 << kb) + (2 << vb)) * 4;
            assert_eq!(rep.shared_constant_bytes, constants);
            assert_eq!(rep.representation_bytes, rep.payload_bytes() + constants);
            assert_eq!(rep.device_bytes, constants, "nothing uploaded yet");
        }
    }

    #[test]
    fn host_round_trip_is_within_rvq_distortion_and_trim_is_a_prefix() {
        let config = RvqConfig {
            key_bits: 2,
            value_bits: 2,
        };
        let (mut cache, keys, _) = filled(config, 1, 2, 6, 128);
        let (dense_keys, _) = cache.dequantize_dense_for_oracle().unwrap();
        assert_eq!(cache.full_cache_dequantizations(), 1);
        let err: f32 = keys
            .iter()
            .zip(&dense_keys)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let energy: f32 = keys.iter().map(|a| a * a).sum();
        // Upstream reports ~13 dB SNR for b=2 on Gaussian input; require at least 12 dB.
        assert!(err / energy < 0.063, "relative MSE {}", err / energy);

        let mut prefix = RvqKvCandidate::new(config, 1, 2, 128).unwrap();
        let first = |data: &[f32]| {
            let mut out = Vec::new();
            for row in 0..2 {
                out.extend_from_slice(&data[row * 6 * 128..row * 6 * 128 + 4 * 128]);
            }
            out
        };
        prefix.append(&first(&keys), &first(&keys), 4).unwrap();
        cache.trim(4).unwrap();
        assert_eq!(cache.keys.stage1, prefix.keys.stage1);
        assert_eq!(cache.keys.norms, prefix.keys.norms);
        assert_eq!(
            cache.representation().key_code_bytes,
            prefix.representation().key_code_bytes
        );
    }

    /// Value-side round trip: upstream reports ~7.5 dB (b=1) and ~13 dB (b=2) for two-stage RVQ
    /// on Gaussian input; the mean over 64 vectors must meet that and no vector may exceed twice it. Rows span a 100x magnitude range so a dropped per-vector normalization
    /// or a mis-scaled codebook cannot hide, and more bits must strictly help.
    #[test]
    fn value_round_trip_meets_rvq_snr_and_improves_with_bits() {
        let (rows, len, dim) = (4, 16, 128);
        let mut values = gaussian(19, rows * len * dim);
        for (i, v) in values.iter_mut().enumerate() {
            *v *= [0.05f32, 0.5, 1.0, 5.0][(i / dim) % 4];
        }
        let mut previous = f32::INFINITY;
        for (bits, bound) in [(1, 0.20f32), (2, 0.063), (3, 0.025)] {
            let mut cache = RvqKvCandidate::new(
                RvqConfig {
                    key_bits: 1,
                    value_bits: bits,
                },
                1,
                rows,
                dim,
            )
            .unwrap();
            cache.append(&values, &values, len).unwrap();
            let (_, decoded) = cache.dequantize_dense_for_oracle().unwrap();
            let per_vector: Vec<f32> = values
                .chunks(dim)
                .zip(decoded.chunks(dim))
                .map(|(original, decoded)| {
                    let error: f32 = original
                        .iter()
                        .zip(decoded)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    error / original.iter().map(|a| a * a).sum::<f32>()
                })
                .collect();
            let mean = per_vector.iter().sum::<f32>() / per_vector.len() as f32;
            let worst = per_vector.iter().copied().fold(0.0, f32::max);
            println!("RVQ value bits {bits}: mean relative MSE {mean}, worst {worst}");
            assert!(mean < bound, "value bits {bits}: mean relative MSE {mean}");
            assert!(
                worst < 2.0 * bound,
                "value bits {bits}: worst relative MSE {worst}"
            );
            assert!(
                mean < previous,
                "value bits {bits} did not improve on fewer bits"
            );
            previous = mean;
        }
    }

    #[test]
    fn preflight_declines_before_mutation_with_stable_reasons() {
        let config = RvqConfig {
            key_bits: 1,
            value_bits: 1,
        };
        let mut empty = RvqKvCandidate::new(config, 1, 2, 64).unwrap();
        assert_eq!(
            empty.preflight(&request(1, 4, 2, 1, 0, 64)),
            CacheRoute::DenseFallback {
                reason: CandidateFallbackReason::EmptyCache.as_str().into()
            }
        );
        let (mut cache, _, _) = filled(config, 1, 2, 4, 64);
        let before = cache.representation();
        let base = request(1, 4, 2, 1, 4, 64);
        let cases: Vec<(CandidateAttentionRequest, CandidateFallbackReason)> = vec![
            (
                CandidateAttentionRequest {
                    backend: "candle-cuda".into(),
                    ..base.clone()
                },
                CandidateFallbackReason::UnsupportedBackend,
            ),
            (
                CandidateAttentionRequest {
                    query_len: 0,
                    ..base.clone()
                },
                CandidateFallbackReason::EmptyQuery,
            ),
            (
                CandidateAttentionRequest {
                    head_dim: 96,
                    ..base.clone()
                },
                CandidateFallbackReason::UnsupportedHeadDimension,
            ),
            (
                CandidateAttentionRequest {
                    query_heads: 3,
                    ..base.clone()
                },
                CandidateFallbackReason::UnsupportedGqaRatio,
            ),
            (
                CandidateAttentionRequest {
                    kv_len: 5,
                    ..base.clone()
                },
                CandidateFallbackReason::GeometryMismatch,
            ),
            (
                CandidateAttentionRequest {
                    query_len: 5,
                    ..base.clone()
                },
                CandidateFallbackReason::QueryLongerThanCache,
            ),
            (
                CandidateAttentionRequest {
                    mask: PackedAttentionMask::Additive,
                    ..base.clone()
                },
                CandidateFallbackReason::AdditiveMask,
            ),
            (
                CandidateAttentionRequest {
                    mask: PackedAttentionMask::SlidingWindow(0),
                    ..base.clone()
                },
                CandidateFallbackReason::EmptySlidingWindow,
            ),
            (
                CandidateAttentionRequest {
                    scale: f32::NAN,
                    ..base.clone()
                },
                CandidateFallbackReason::InvalidScale,
            ),
        ];
        for (index, (request, reason)) in cases.iter().enumerate() {
            assert_eq!(
                cache.preflight(request),
                CacheRoute::DenseFallback {
                    reason: reason.as_str().into()
                },
                "case {index}"
            );
            assert_eq!(cache.fallback_events()[index].reason, reason.as_str());
        }
        assert_eq!(cache.representation(), before, "preflight never mutates");
        assert_eq!(cache.preflight(&base), CacheRoute::ExperimentalPacked);
        assert!(RvqKvCandidate::new(config, 1, 2, 96).is_err());
    }

    #[cfg(target_os = "macos")]
    fn metal_parity(config: RvqConfig, request: CandidateAttentionRequest) {
        let n = request.batch * request.kv_heads * request.kv_len * request.head_dim;
        let keys = gaussian(21, n);
        let values = gaussian(22, n);
        let query_host = gaussian(
            23,
            request.batch * request.query_heads * request.query_len * request.head_dim,
        );
        let mut cache =
            RvqKvCandidate::new(config, request.batch, request.kv_heads, request.head_dim).unwrap();
        // Two chunks exercise append ordering.
        let split = request.kv_len / 2;
        let chunk = |data: &[f32], start: usize, end: usize| {
            let mut out = Vec::new();
            for row in 0..request.batch * request.kv_heads {
                let base = row * request.kv_len * request.head_dim;
                out.extend_from_slice(
                    &data[base + start * request.head_dim..base + end * request.head_dim],
                );
            }
            out
        };
        cache
            .append(&chunk(&keys, 0, split), &chunk(&values, 0, split), split)
            .unwrap();
        cache
            .append(
                &chunk(&keys, split, request.kv_len),
                &chunk(&values, split, request.kv_len),
                request.kv_len - split,
            )
            .unwrap();
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
        let output = cache.attend(&query, &request).unwrap();
        let actual = output.as_slice::<f32>().to_vec();
        assert_eq!(cache.full_cache_dequantizations(), 0);
        let (dense_keys, dense_values) = cache.dequantize_dense_for_oracle().unwrap();
        let expected =
            dense_attention_oracle(&request, &query_host, &dense_keys, &dense_values).unwrap();
        let diff = max_abs_diff(&actual, &expected);
        assert!(diff < 2e-4, "{config:?} {request:?}: max abs diff {diff}");
        let rep = cache.representation();
        assert_eq!(rep.device_bytes, rep.representation_bytes);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matches_fp32_dequantize_then_attend_oracle_with_gqa_masks_and_bits() {
        let base = request(2, 4, 2, 3, 37, 64);
        for (kb, vb) in [(1, 1), (2, 1), (1, 3), (2, 2)] {
            metal_parity(
                RvqConfig {
                    key_bits: kb,
                    value_bits: vb,
                },
                base.clone(),
            );
        }
        let config = RvqConfig {
            key_bits: 2,
            value_bits: 2,
        };
        for mask in [
            PackedAttentionMask::None,
            PackedAttentionMask::SlidingWindow(5),
        ] {
            metal_parity(
                config,
                CandidateAttentionRequest {
                    mask,
                    ..base.clone()
                },
            );
        }
        metal_parity(config, request(1, 8, 2, 1, 19, 128));
        metal_parity(
            config,
            CandidateAttentionRequest {
                scale: 0.3,
                ..request(1, 2, 1, 2, 9, 256)
            },
        );
    }
}
