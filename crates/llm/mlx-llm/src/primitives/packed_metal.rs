//! Retained MLX Metal reader for SC-20675's packed K/V layout.
//!
//! K codes are token-group packed (`[B,H,ceil(S/group),D]`) and V codes are
//! channel-group packed (`[B,H,S,ceil(D/group)]`). The kernel streams keys and
//! values and keeps only one output accumulator per thread; it never constructs
//! dense historical K/V or a score matrix.
use crate::error::{Error, Result};
use mlx_rs::fast::{MetalKernel, OutputArg};
use mlx_rs::{Array, Dtype};

/// Mask forms the retained reader can prove without allocating a score matrix.  Arbitrary
/// additive masks deliberately select the observable dense fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedMask {
    Causal,
    SlidingWindow(usize),
    AdditiveUnsupported,
}

const HEADER: &str = r#"#include <metal_stdlib>
using namespace metal;
"#;

const MSL: &str = r#"
    const uint b = thread_position_in_grid.y;
    const uint query = threadgroup_position_in_grid.x;
    const uint qh = query / q_shape[2];
    const uint qi = query % q_shape[2];
    const uint kh = qh / (q_shape[1] / v_codes_shape[1]);
    const uint d = thread_position_in_threadgroup.x;
    if (qi >= q_shape[2] || d >= q_shape[3]) return;
    float outv = 0.0f, norm = 0.0f, maxv = -INFINITY;
    for (uint ks = 0; ks < v_codes_shape[2]; ++ks) {
        const uint qpos = v_codes_shape[2] - q_shape[2] + qi;
        if ((MASK_MODE == 1 && ks > qpos) ||
            (MASK_MODE == 2 && (ks > qpos || ks + WINDOW <= qpos))) continue;
        float dot = 0.0f;
        for (uint j = 0; j < q_shape[3]; ++j) {
            const uint qidx = ((b*q_shape[1]+qh)*q_shape[2]+qi)*q_shape[3]+j;
            const uint kg = ks / GROUP;
            const uint kbit = (ks % GROUP) * q_shape[3] + j;
            const uint kc = (k_codes[((b*v_codes_shape[1]+kh)*((v_codes_shape[2]+GROUP-1)/GROUP)+kg)*K_WORDS + kbit/4] >> ((kbit%4)*2)) & 3;
            const float kval = k_zero[((b*v_codes_shape[1]+kh)*((v_codes_shape[2]+GROUP-1)/GROUP)+kg)*q_shape[3]+j] + float(k_scale[((b*v_codes_shape[1]+kh)*((v_codes_shape[2]+GROUP-1)/GROUP)+kg)*q_shape[3]+j]) * float(kc);
            dot += float(q[qidx]) * kval;
        }
        const float score = dot * rsqrt(float(q_shape[3]));
        const float next_max = max(maxv, score);
        const float rescale = (maxv == -INFINITY) ? 0.0f : exp(maxv - next_max);
        const float w = exp(score - next_max);
        norm = norm * rescale + w;
        maxv = next_max;
        const uint vbit = d;
        const uint vc = (v_codes[((b*v_codes_shape[1]+kh)*v_codes_shape[2]+ks)*V_WORDS+vbit/4] >> ((vbit%4)*2)) & 3;
        outv = outv * rescale + w * (v_zero[((b*v_codes_shape[1]+kh)*v_codes_shape[2]+ks)*((q_shape[3]+GROUP-1)/GROUP)+d/GROUP] + float(v_scale[((b*v_codes_shape[1]+kh)*v_codes_shape[2]+ks)*((q_shape[3]+GROUP-1)/GROUP)+d/GROUP]) * float(vc));
    }
    out[((b*q_shape[1]+qh)*q_shape[2]+qi)*q_shape[3]+d] = outv / norm;
"#;

/// Retained kernel object; MLX performs cold compilation on first `.run()` and
/// reuses the same compiled pipeline for subsequent dispatches.
pub struct PackedMetalKernel {
    kernel: MetalKernel,
    identity: String,
}

impl crate::primitives::packed_group_affine_kv::RetainedPackedKernel for PackedMetalKernel {
    fn cache_identity(&self) -> &str {
        &self.identity
    }

    fn backend(&self) -> &str {
        "mlx-metal"
    }

    fn retained_host_bytes_estimate(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    fn dispatch(
        &self,
        query: &Array,
        k_codes: &Array,
        k_scale: &Array,
        k_zero: &Array,
        v_codes: &Array,
        v_scale: &Array,
        v_zero: &Array,
        mask: PackedMask,
    ) -> Result<Array> {
        PackedMetalKernel::dispatch(
            self, query, k_codes, k_scale, k_zero, v_codes, v_scale, v_zero, mask,
        )
    }
}

impl std::fmt::Debug for PackedMetalKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedMetalKernel").finish_non_exhaustive()
    }
}

impl PackedMetalKernel {
    pub fn new() -> Result<Self> {
        Self::for_identity("sc-20676-packed-group-affine-v1")
    }

    /// Construct the retained reader for one cache identity.  The identity is part of the
    /// compiled-handle binding, preventing a pipeline from being reused with another cache's
    /// layout or quantization contract.
    pub fn for_identity(identity: impl Into<String>) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::with_options(
                "sc20676_group_affine_online",
                &[
                    "q", "k_codes", "k_scale", "k_zero", "v_codes", "v_scale", "v_zero",
                ],
                &["out"],
                MSL,
                HEADER,
                true,
                false,
            )?,
            identity: identity.into(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        q: &Array,
        k_codes: &Array,
        k_scale: &Array,
        k_zero: &Array,
        v_codes: &Array,
        v_scale: &Array,
        v_zero: &Array,
        mask: PackedMask,
    ) -> Result<Array> {
        let shape = q.shape();
        if k_codes.ndim() != 4 || v_codes.ndim() != 4 {
            return Err(Error::Unsupported("SC-20676 packed buffer rank".into()));
        }
        let q_shape = shape;
        let kc_shape = k_codes.shape();
        let ks_shape = k_scale.shape();
        let kz_shape = k_zero.shape();
        let vc_shape = v_codes.shape();
        let vs_shape = v_scale.shape();
        let vz_shape = v_zero.shape();
        if q_shape.len() != 4
            || kc_shape.len() != 4
            || ks_shape.len() != 4
            || kz_shape.len() != 4
            || vc_shape.len() != 4
            || vs_shape.len() != 4
            || vz_shape.len() != 4
            || kc_shape[0] != q_shape[0]
            || kc_shape[1] == 0
            || q_shape[1] % kc_shape[1] != 0
            || ks_shape != [kc_shape[0], kc_shape[1], kc_shape[2], q_shape[3]]
            || kz_shape != ks_shape
            || vc_shape[0] != q_shape[0]
            || vc_shape[1] != kc_shape[1]
            || vc_shape[2] == 0
            || q_shape[2] > vc_shape[2]
            || kc_shape[2] != (vc_shape[2] + 3) / 4
            || vs_shape
                != [
                    vc_shape[0],
                    vc_shape[1],
                    vc_shape[2],
                    (q_shape[3] + 4 - 1) / 4,
                ]
            || vz_shape != vs_shape
            || kc_shape[3] != ((4 * q_shape[3] + 3) / 4)
            || vc_shape[3] != (q_shape[3] + 3) / 4
            || k_codes.dtype() != Dtype::Uint8
            || v_codes.dtype() != Dtype::Uint8
            || k_scale.dtype() != Dtype::Float16
            || k_zero.dtype() != Dtype::Float16
            || v_scale.dtype() != Dtype::Float16
            || v_zero.dtype() != Dtype::Float16
            || !matches!(q.dtype(), Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32)
        {
            return Err(Error::Unsupported(
                "SC-20676 packed buffers do not match query/cache geometry".into(),
            ));
        }
        let (mask_mode, window) = match mask {
            PackedMask::Causal => (1, 0),
            PackedMask::SlidingWindow(window) if window > 0 => (
                2,
                i32::try_from(window)
                    .map_err(|_| Error::Unsupported("sliding window exceeds i32".into()))?,
            ),
            PackedMask::SlidingWindow(_) => {
                return Err(Error::Unsupported("empty sliding window".into()))
            }
            PackedMask::AdditiveUnsupported => {
                return Err(Error::Unsupported(
                    "additive mask requires dense fallback".into(),
                ))
            }
        };
        if shape[0] == 0 || shape[1] == 0 || shape[2] == 0 || !matches!(shape[3], 64 | 128 | 256) {
            return Err(Error::Unsupported("SC-20676 packed Metal geometry".into()));
        }
        let out = self
            .kernel
            .apply()
            .input(q)
            .input(k_codes)
            .input(k_scale)
            .input(k_zero)
            .input(v_codes)
            .input(v_scale)
            .input(v_zero)
            .output(OutputArg {
                shape: shape.to_vec(),
                dtype: q.dtype(),
            })
            .grid(shape[1] * shape[2] * shape[3], shape[0], 1)
            .thread_group(shape[3], 1, 1)
            .template_arg("GROUP", 4)
            .template_arg("K_WORDS", (4 * shape[3] + 3) / 4)
            .template_arg("V_WORDS", (shape[3] + 3) / 4)
            .template_arg("MASK_MODE", mask_mode)
            .template_arg("WINDOW", window)
            .run()?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("SC-20676 kernel returned no output".into()))?;
        Ok(out)
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    fn half(values: Vec<f32>, shape: &[i32]) -> Array {
        Array::from_slice(&values, shape)
            .as_dtype(Dtype::Float16)
            .unwrap()
    }

    fn run_uniform_score_case(mask: PackedMask) -> Vec<f32> {
        const Q_HEADS: usize = 4;
        const KV_HEADS: usize = 2;
        const SQ: usize = 2;
        const SKV: usize = 5;
        const D: usize = 64;
        const GROUPS: usize = D / 4;
        let q = Array::from_slice(&vec![0.0f32; Q_HEADS * SQ * D], &[1, 4, 2, 64]);
        let key_codes =
            Array::from_slice(&vec![0u8; KV_HEADS * SKV.div_ceil(4) * D], &[1, 2, 2, 64]);
        let key_scale = half(vec![0.0; KV_HEADS * SKV.div_ceil(4) * D], &[1, 2, 2, 64]);
        let key_zero = half(vec![0.0; KV_HEADS * SKV.div_ceil(4) * D], &[1, 2, 2, 64]);
        let value_codes = Array::from_slice(&[0u8; KV_HEADS * SKV * GROUPS], &[1, 2, 5, 16]);
        let value_scale = half(vec![0.0; KV_HEADS * SKV * GROUPS], &[1, 2, 5, 16]);
        let mut value_zeros = vec![0.0; KV_HEADS * SKV * GROUPS];
        for kh in 0..KV_HEADS {
            for token in 0..SKV {
                for group in 0..GROUPS {
                    value_zeros[(kh * SKV + token) * GROUPS + group] =
                        kh as f32 * 10.0 + token as f32;
                }
            }
        }
        let value_zero = half(value_zeros, &[1, 2, 5, 16]);
        let output = PackedMetalKernel::new()
            .unwrap()
            .dispatch(
                &q,
                &key_codes,
                &key_scale,
                &key_zero,
                &value_codes,
                &value_scale,
                &value_zero,
                mask,
            )
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        output.eval().unwrap();
        output.as_slice::<f32>().to_vec()
    }

    #[test]
    fn real_metal_kernel_jits_with_gqa_and_exact_causal_and_sliding_boundaries() {
        const SQ: usize = 2;
        const D: usize = 64;
        for (mask, expected_tokens) in [
            (
                PackedMask::Causal,
                vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3, 4]],
            ),
            (PackedMask::SlidingWindow(2), vec![vec![2, 3], vec![3, 4]]),
        ] {
            let values = run_uniform_score_case(mask);
            for qh in 0..4 {
                let kh = qh / 2;
                for (qi, tokens) in expected_tokens.iter().enumerate() {
                    let expected = kh as f32 * 10.0
                        + tokens.iter().map(|&token| token as f32).sum::<f32>()
                            / tokens.len() as f32;
                    for d in 0..D {
                        let actual = values[(qh * SQ + qi) * D + d];
                        assert!(
                            (actual - expected).abs() <= 1e-5,
                            "qh={qh} qi={qi} d={d}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sliding_window_larger_than_msl_i32_fails_before_dispatch() {
        let q = Array::from_slice(&[0.0f32; 64], &[1, 1, 1, 64]);
        let key_codes = Array::from_slice(&[0u8; 64], &[1, 1, 1, 64]);
        let key_scale = half(vec![0.0; 64], &[1, 1, 1, 64]);
        let key_zero = half(vec![0.0; 64], &[1, 1, 1, 64]);
        let value_codes = Array::from_slice(&[0u8; 16], &[1, 1, 1, 16]);
        let value_scale = half(vec![0.0; 16], &[1, 1, 1, 16]);
        let value_zero = half(vec![0.0; 16], &[1, 1, 1, 16]);
        let error = PackedMetalKernel::new()
            .unwrap()
            .dispatch(
                &q,
                &key_codes,
                &key_scale,
                &key_zero,
                &value_codes,
                &value_scale,
                &value_zero,
                PackedMask::SlidingWindow(usize::MAX),
            )
            .unwrap_err();
        assert!(error.to_string().contains("sliding window exceeds i32"));
    }

    #[test]
    fn real_metal_kernel_accepts_every_advertised_query_dtype_and_head_dimension() {
        for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
            for dimension in [64usize, 128, 256] {
                let q = Array::from_slice(&vec![0.0f32; dimension], &[1, 1, 1, dimension as i32])
                    .as_dtype(dtype)
                    .unwrap();
                let key_codes =
                    Array::from_slice(&vec![0u8; dimension], &[1, 1, 1, dimension as i32]);
                let key_scale = half(vec![0.0; dimension], &[1, 1, 1, dimension as i32]);
                let key_zero = half(vec![0.0; dimension], &[1, 1, 1, dimension as i32]);
                let value_words = dimension.div_ceil(4);
                let value_codes =
                    Array::from_slice(&vec![0u8; value_words], &[1, 1, 1, value_words as i32]);
                let value_scale = half(vec![0.0; value_words], &[1, 1, 1, value_words as i32]);
                let value_zero = half(vec![0.0; value_words], &[1, 1, 1, value_words as i32]);
                let output = PackedMetalKernel::new()
                    .unwrap()
                    .dispatch(
                        &q,
                        &key_codes,
                        &key_scale,
                        &key_zero,
                        &value_codes,
                        &value_scale,
                        &value_zero,
                        PackedMask::Causal,
                    )
                    .unwrap()
                    .as_dtype(Dtype::Float32)
                    .unwrap();
                output.eval().unwrap();
                assert!(
                    output
                        .as_slice::<f32>()
                        .iter()
                        .all(|value| value.abs() <= f32::EPSILON),
                    "dtype={dtype:?} dimension={dimension}"
                );
            }
        }
    }
}
