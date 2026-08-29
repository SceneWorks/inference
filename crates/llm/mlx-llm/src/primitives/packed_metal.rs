//! Retained MLX Metal reader for SC-20675's packed K/V layout.
//!
//! K codes are token-group packed (`[B,H,ceil(S/group),D]`) and V codes are
//! channel-group packed (`[B,H,S,ceil(D/group)]`). The kernel streams keys and
//! values and keeps only one output accumulator per thread; it never constructs
//! dense historical K/V or a score matrix.
use crate::error::{Error, Result};
use mlx_rs::fast::{MetalKernel, OutputArg};
use mlx_rs::Array;

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
    const uint qh = thread_position_in_grid.x / q_shape[2];
    const uint qi = thread_position_in_grid.x % q_shape[2];
    const uint kh = qh / (q_shape[1] / kv_shape[1]);
    const uint d = thread_position_in_threadgroup.x;
    if (qi >= q_shape[2] || d >= q_shape[3]) return;
    float outv = 0.0f, norm = 0.0f, maxv = -INFINITY;
    for (uint ks = 0; ks < kv_shape[2]; ++ks) {
        const uint qpos = kv_shape[2] - q_shape[2] + qi;
        if ((MASK_MODE == 1 && ks > qpos) ||
            (MASK_MODE == 2 && ks + WINDOW < qpos)) continue;
        float dot = 0.0f;
        for (uint j = 0; j < q_shape[3]; ++j) {
            const uint qidx = ((b*q_shape[1]+qh)*q_shape[2]+qi)*q_shape[3]+j;
            const uint kg = ks / GROUP;
            const uint kbit = (ks % GROUP) * q_shape[3] + j;
            const uint kc = (k_codes[((b*kv_shape[1]+kh)*((kv_shape[2]+GROUP-1)/GROUP)+kg)*K_WORDS + kbit/4] >> ((kbit%4)*2)) & 3;
            const float kval = k_zero[((b*kv_shape[1]+kh)*((kv_shape[2]+GROUP-1)/GROUP)+kg)*q_shape[3]+j] + float(k_scale[((b*kv_shape[1]+kh)*((kv_shape[2]+GROUP-1)/GROUP)+kg)*q_shape[3]+j]) * float(kc);
            dot += float(q[qidx]) * kval;
        }
        const float score = dot * rsqrt(float(q_shape[3]));
        const float next_max = max(maxv, score);
        const float rescale = (maxv == -INFINITY) ? 0.0f : exp(maxv - next_max);
        const float w = exp(score - next_max);
        norm = norm * rescale + w;
        maxv = next_max;
        const uint vbit = d;
        const uint vc = (v_codes[((b*kv_shape[1]+kh)*kv_shape[2]+ks)*V_WORDS+vbit/4] >> ((vbit%4)*2)) & 3;
        outv = outv * rescale + w * (v_zero[((b*kv_shape[1]+kh)*kv_shape[2]+ks)*((q_shape[3]+GROUP-1)/GROUP)+d/GROUP] + float(v_scale[((b*kv_shape[1]+kh)*kv_shape[2]+ks)*((q_shape[3]+GROUP-1)/GROUP)+d/GROUP]) * float(vc));
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

    fn retained_bytes(&self) -> usize {
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
        {
            return Err(Error::Unsupported(
                "SC-20676 packed buffers do not match query/cache geometry".into(),
            ));
        }
        let (mask_mode, window) = match mask {
            PackedMask::Causal => (1, 0),
            PackedMask::SlidingWindow(window) if window > 0 => (2, window as i32),
            PackedMask::SlidingWindow(_) => {
                return Err(Error::Unsupported("empty sliding window".into()))
            }
            PackedMask::AdditiveUnsupported => {
                return Err(Error::Unsupported(
                    "additive mask requires dense fallback".into(),
                ))
            }
        };
        if !matches!(shape[3], 64 | 128 | 256) || shape[1] == 0 {
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
            .grid(shape[1] * shape[2], shape[0], 1)
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
