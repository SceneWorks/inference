//! Retained MLX Metal reader for SC-20675's packed K/V layout.
//!
//! K codes are token-group packed (`[B,H,ceil(S/group),D]`) and V codes are
//! channel-group packed (`[B,H,S,ceil(D/group)]`). The kernel streams keys and
//! values and keeps only one output accumulator per thread; it never constructs
//! dense historical K/V or a score matrix.
use crate::error::{Error, Result};
use crate::primitives::packed_group_affine_kv::{
    packed_metal_head_dimension_supported, PACKED_CODES_PER_BYTE, PACKED_METAL_QUANT_GROUP_SIZE,
};
use mlx_rs::fast::{MetalKernel, OutputArg};
use mlx_rs::{Array, Dtype};

fn checked_shape(shape: &[i32], tensor: &str) -> Result<[usize; 4]> {
    let [batch, heads, tokens, width] = shape else {
        return Err(Error::Unsupported(format!("SC-20676 {tensor} buffer rank")));
    };
    [*batch, *heads, *tokens, *width]
        .map(|dimension| {
            usize::try_from(dimension).map_err(|_| {
                Error::Unsupported(format!("SC-20676 {tensor} buffer has a negative dimension"))
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .and_then(|dimensions| {
            dimensions
                .try_into()
                .map_err(|_| Error::Msg("internal SC-20676 shape conversion failed".into()))
        })
}

fn checked_div_ceil(value: usize, divisor: usize, name: &str) -> Result<usize> {
    if divisor == 0 {
        return Err(Error::Msg(format!("SC-20676 {name} divisor is zero")));
    }
    Ok(value / divisor + usize::from(!value.is_multiple_of(divisor)))
}

fn checked_msl_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::Unsupported(format!("SC-20676 {name} exceeds MSL i32 range")))
}

/// Mask forms the retained reader can prove without allocating a score matrix.  Arbitrary
/// additive masks deliberately select the observable dense fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedMask {
    None,
    Causal,
    SlidingWindow(usize),
    AdditiveUnsupported,
}

/// Explicit GPU-family tuning boundary. Unknown Apple GPUs use one SIMD group and stride over D;
/// qualified recent families may use one thread per channel with 2/4/8 cooperating SIMD groups.
/// No family outside this enum is silently assigned an aggressive geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PackedMetalGpuFamily {
    #[default]
    ConservativeUnknownApple,
    Apple7OrNewer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedMetalTuning {
    threads: usize,
    simd_groups: usize,
    values_per_thread: usize,
}

/// Dispatch geometry selected by the retained packed Metal reader. Evidence harnesses expose this
/// profile alongside the probed device name so performance receipts cannot silently change tuning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedMetalTuningProfile {
    pub gpu_family: &'static str,
    pub threads: usize,
    pub simd_groups: usize,
    pub values_per_thread: usize,
}

impl PackedMetalGpuFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConservativeUnknownApple => "conservative-unknown-apple",
            Self::Apple7OrNewer => "apple7-or-newer",
        }
    }

    fn tuning(self, head_dimension: usize) -> Option<PackedMetalTuning> {
        if !packed_metal_head_dimension_supported(head_dimension) {
            return None;
        }
        let threads = match self {
            Self::ConservativeUnknownApple => 32,
            Self::Apple7OrNewer => head_dimension,
        };
        Some(PackedMetalTuning {
            threads,
            simd_groups: threads / 32,
            values_per_thread: head_dimension.div_ceil(threads),
        })
    }
}

const HEADER: &str = r#"#include <metal_stdlib>
using namespace metal;

template <typename T> struct sc20676_device_element;
template <typename T> struct sc20676_device_element<device T*> { using type = T; };
"#;

const MSL: &str = r#"
    const uint b = thread_position_in_grid.y;
    const uint query = thread_position_in_grid.x / THREADS;
    const uint qh = query / q_shape[2];
    const uint qi = query % q_shape[2];
    const uint kh = qh / (q_shape[1] / v_codes_shape[1]);
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid & 31;
    const uint simd_id = tid >> 5;
    if (qi >= q_shape[2] || tid >= THREADS) return;

    threadgroup float dot_partials[8];
    threadgroup float shared_norm;
    threadgroup float shared_max;
    threadgroup float shared_rescale;
    threadgroup float shared_weight;
    if (tid == 0) {
        shared_norm = 0.0f;
        shared_max = -INFINITY;
        shared_rescale = 0.0f;
        shared_weight = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float outv[VALUES_PER_THREAD];
    for (uint owned = 0; owned < VALUES_PER_THREAD; ++owned) outv[owned] = 0.0f;
    for (uint ks = 0; ks < v_codes_shape[2]; ++ks) {
        const uint qpos = v_codes_shape[2] - q_shape[2] + qi;
        if ((MASK_MODE == 1 && ks > qpos) ||
            (MASK_MODE == 2 && (ks > qpos || ks + WINDOW <= qpos))) continue;

        float partial = 0.0f;
        for (uint j = tid; j < q_shape[3]; j += THREADS) {
            const uint qidx = ((b*q_shape[1]+qh)*q_shape[2]+qi)*q_shape[3]+j;
            const uint kg = ks / GROUP;
            const uint kbit = (ks % GROUP) * q_shape[3] + j;
            const uint kc = (k_codes[((b*v_codes_shape[1]+kh)*((v_codes_shape[2]+GROUP-1)/GROUP)+kg)*K_WORDS + kbit/CODES_PER_BYTE] >> ((kbit%CODES_PER_BYTE)*2)) & 3;
            const float kval = k_zero[((b*v_codes_shape[1]+kh)*((v_codes_shape[2]+GROUP-1)/GROUP)+kg)*q_shape[3]+j] + float(k_scale[((b*v_codes_shape[1]+kh)*((v_codes_shape[2]+GROUP-1)/GROUP)+kg)*q_shape[3]+j]) * float(kc);
            partial += float(q[qidx]) * kval;
        }
        const float simd_partial = simd_sum(partial);
        if (lane == 0) dot_partials[simd_id] = simd_partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0) {
            const float group_partial = lane < SIMD_GROUPS ? dot_partials[lane] : 0.0f;
            const float dot = simd_sum(group_partial);
            if (lane == 0) {
                const float score = dot * rsqrt(float(q_shape[3]));
                const float next_max = max(shared_max, score);
                shared_rescale = shared_max == -INFINITY ? 0.0f : exp(shared_max - next_max);
                shared_weight = exp(score - next_max);
                shared_norm = shared_norm * shared_rescale + shared_weight;
                shared_max = next_max;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint owned = 0; owned < VALUES_PER_THREAD; ++owned) {
            const uint d = tid + owned * THREADS;
            if (d >= q_shape[3]) continue;
            const uint vc = (v_codes[((b*v_codes_shape[1]+kh)*v_codes_shape[2]+ks)*V_WORDS+d/CODES_PER_BYTE] >> ((d%CODES_PER_BYTE)*2)) & 3;
            const uint metadata = ((b*v_codes_shape[1]+kh)*v_codes_shape[2]+ks)*((q_shape[3]+GROUP-1)/GROUP)+d/GROUP;
            const float value = v_zero[metadata] + float(v_scale[metadata]) * float(vc);
            outv[owned] = outv[owned] * shared_rescale + shared_weight * value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint owned = 0; owned < VALUES_PER_THREAD; ++owned) {
        const uint d = tid + owned * THREADS;
        if (d < q_shape[3]) {
            using OutputT = typename sc20676_device_element<decltype(out)>::type;
            out[((b*q_shape[1]+qh)*q_shape[2]+qi)*q_shape[3]+d] = static_cast<OutputT>(outv[owned] / shared_norm);
        }
    }
"#;

/// Retained kernel object; MLX performs cold compilation on first `.run()` and
/// reuses the same compiled pipeline for subsequent dispatches.
pub struct PackedMetalKernel {
    kernel: MetalKernel,
    identity: String,
    gpu_family: PackedMetalGpuFamily,
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
        f.debug_struct("PackedMetalKernel")
            .field("gpu_family", &self.gpu_family)
            .finish_non_exhaustive()
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
        Self::for_identity_and_family(identity, PackedMetalGpuFamily::ConservativeUnknownApple)
    }

    /// Bind a cache identity to an explicit GPU-family tuning profile. Callers may select the
    /// qualified recent-family profile only after their device probe; unknown devices retain the
    /// conservative one-SIMD-group geometry.
    pub fn for_identity_and_family(
        identity: impl Into<String>,
        gpu_family: PackedMetalGpuFamily,
    ) -> Result<Self> {
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
            gpu_family,
        })
    }

    pub fn gpu_family(&self) -> PackedMetalGpuFamily {
        self.gpu_family
    }

    pub fn tuning_profile(&self, head_dimension: usize) -> Option<PackedMetalTuningProfile> {
        self.gpu_family
            .tuning(head_dimension)
            .map(|tuning| PackedMetalTuningProfile {
                gpu_family: self.gpu_family.as_str(),
                threads: tuning.threads,
                simd_groups: tuning.simd_groups,
                values_per_thread: tuning.values_per_thread,
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
        let q_shape = q.shape();
        let kc_shape = k_codes.shape();
        let ks_shape = k_scale.shape();
        let kz_shape = k_zero.shape();
        let vc_shape = v_codes.shape();
        let vs_shape = v_scale.shape();
        let vz_shape = v_zero.shape();
        let q_dimensions = checked_shape(q_shape, "query")?;
        let kc_dimensions = checked_shape(kc_shape, "key codes")?;
        let ks_dimensions = checked_shape(ks_shape, "key scales")?;
        let kz_dimensions = checked_shape(kz_shape, "key zeros")?;
        let vc_dimensions = checked_shape(vc_shape, "value codes")?;
        let vs_dimensions = checked_shape(vs_shape, "value scales")?;
        let vz_dimensions = checked_shape(vz_shape, "value zeros")?;
        let [batch, query_heads, query_tokens, head_dimension] = q_dimensions;
        let key_words = checked_div_ceil(
            PACKED_METAL_QUANT_GROUP_SIZE
                .checked_mul(head_dimension)
                .ok_or_else(|| Error::Unsupported("SC-20676 key word size overflow".into()))?,
            PACKED_CODES_PER_BYTE,
            "key word size",
        )?;
        let value_words =
            checked_div_ceil(head_dimension, PACKED_CODES_PER_BYTE, "value word size")?;
        let key_groups = checked_div_ceil(
            vc_dimensions[2],
            PACKED_METAL_QUANT_GROUP_SIZE,
            "key group count",
        )?;
        let value_groups = checked_div_ceil(
            head_dimension,
            PACKED_METAL_QUANT_GROUP_SIZE,
            "value group count",
        )?;
        if batch == 0
            || query_heads == 0
            || query_tokens == 0
            || !packed_metal_head_dimension_supported(head_dimension)
            || kc_dimensions[0] != batch
            || kc_dimensions[1] == 0
            || query_heads % kc_dimensions[1] != 0
            || ks_dimensions
                != [
                    kc_dimensions[0],
                    kc_dimensions[1],
                    kc_dimensions[2],
                    head_dimension,
                ]
            || kz_dimensions != ks_dimensions
            || vc_dimensions[0] != batch
            || vc_dimensions[1] != kc_dimensions[1]
            || vc_dimensions[2] == 0
            || query_tokens > vc_dimensions[2]
            || kc_dimensions[2] != key_groups
            || vs_dimensions
                != [
                    vc_dimensions[0],
                    vc_dimensions[1],
                    vc_dimensions[2],
                    value_groups,
                ]
            || vz_dimensions != vs_dimensions
            || kc_dimensions[3] != key_words
            || vc_dimensions[3] != value_words
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
            PackedMask::None => (0, 0),
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
        let tuning = self.gpu_family.tuning(head_dimension).ok_or_else(|| {
            Error::Unsupported(
                "SC-20676 has no conservative tuning for this device/geometry".into(),
            )
        })?;
        let threads = checked_msl_i32(tuning.threads, "thread-group width")?;
        let grid_x = q_shape[1]
            .checked_mul(q_shape[2])
            .and_then(|value| value.checked_mul(threads))
            .ok_or_else(|| Error::Unsupported("SC-20676 Metal grid dimension overflow".into()))?;
        let group = checked_msl_i32(PACKED_METAL_QUANT_GROUP_SIZE, "quantization group")?;
        let codes_per_byte = checked_msl_i32(PACKED_CODES_PER_BYTE, "codes per byte")?;
        let key_words = checked_msl_i32(key_words, "key words")?;
        let value_words = checked_msl_i32(value_words, "value words")?;
        let simd_groups = checked_msl_i32(tuning.simd_groups, "SIMD-group count")?;
        let values_per_thread = checked_msl_i32(tuning.values_per_thread, "values per thread")?;
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
                shape: q_shape.to_vec(),
                dtype: q.dtype(),
            })
            .grid(grid_x, q_shape[0], 1)
            .thread_group(threads, 1, 1)
            .template_arg("GROUP", group)
            .template_arg("CODES_PER_BYTE", codes_per_byte)
            .template_arg("K_WORDS", key_words)
            .template_arg("V_WORDS", value_words)
            .template_arg("THREADS", threads)
            .template_arg("SIMD_GROUPS", simd_groups)
            .template_arg("VALUES_PER_THREAD", values_per_thread)
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

    #[test]
    fn gpu_family_tuning_is_conservative_and_geometry_explicit() {
        for dimension in [64, 128, 256] {
            let conservative = PackedMetalGpuFamily::ConservativeUnknownApple
                .tuning(dimension)
                .unwrap();
            assert_eq!(conservative.threads, 32);
            assert_eq!(conservative.simd_groups, 1);
            assert_eq!(conservative.values_per_thread, dimension / 32);

            let recent = PackedMetalGpuFamily::Apple7OrNewer
                .tuning(dimension)
                .unwrap();
            assert_eq!(recent.threads, dimension);
            assert_eq!(recent.simd_groups, dimension / 32);
            assert_eq!(recent.values_per_thread, 1);
        }
        assert!(PackedMetalGpuFamily::ConservativeUnknownApple
            .tuning(96)
            .is_none());
        assert_eq!(
            PackedMetalGpuFamily::ConservativeUnknownApple.as_str(),
            "conservative-unknown-apple"
        );
        assert_eq!(
            PackedMetalGpuFamily::Apple7OrNewer.as_str(),
            "apple7-or-newer"
        );
    }

    #[test]
    fn sliding_window_larger_than_msl_i32_fails_before_dispatch() {
        let q = Array::from_slice(&[0.0f32; 64], &[1, 1, 1, 64]);
        let key_codes = Array::from_slice(&[0u8; 512], &[1, 1, 1, 512]);
        let key_scale = half(vec![0.0; 64], &[1, 1, 1, 64]);
        let key_zero = half(vec![0.0; 64], &[1, 1, 1, 64]);
        let value_codes = Array::from_slice(&[0u8; 16], &[1, 1, 1, 16]);
        let value_scale = half(vec![0.0; 2], &[1, 1, 1, 2]);
        let value_zero = half(vec![0.0; 2], &[1, 1, 1, 2]);
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
}
