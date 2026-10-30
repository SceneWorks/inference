//! Native packed **W4A8 (asym_w4a8_int8)** projections, not a dense dequantization in disguise
//! (sc-26685).
//!
//! What is resident is the published packed form and nothing else: `I8 [out, in/2]` two-int4 codes,
//! `F8_E4M3 [out, in/16]` group scales, `F32 [out]` channel scales and one `F32 [16]` codebook, plus
//! the optional dense bias. The int4→int8 expansion happens **inside** the tiled GEMM's weight-tile
//! load ([`crate::int8_gemm`], sc-26691), so a projection never holds a dense `[out, in]` int8 (or
//! float) shadow of its weight — nor, any more, even a decoded column-tile scratch.
//!
//! # The arithmetic is the pinned Kitchen chain, in three pieces
//!
//! 1. **Activation** — the pinned fused ConvRot quantization of
//!    `int8_linear.cu::quantize_int8_rowwise_convrot_kernel`: `convrot_fht_f32` replays the
//!    regular (non-Walsh) `H₄` butterfly of `convrot_groupsize = 256` in **F32** and in the kernel's
//!    own `+` order (strides `1,4,16,64`, `0.5` per stage), then the row-wise int8 quantizer takes
//!    the row absmax/scale from those F32 values and applies Kitchen's source-dtype cast chain to
//!    the numerator, the divisor and the quotient (`quant_div_float_to_float`), with `rint` and the
//!    `[-128, 127]` clamp. It runs **once per row, before any weight tile is decoded**: the pinned
//!    `launch_w4a8_gemm` computes `xq`/`xs` before the per-tile decode, and a per-tile
//!    re-quantization would change the result. The W4A8 quantizer is its own kernel (`QUANT`,
//!    registered as `sc_w4a8_int8_quant_v1`): its numerator rounding is the W4A8 authority's and
//!    must not be conflated with the S4 ConvRot codec's kernel, which is fed an already-rounded
//!    rotation.
//! 2. **Weight** — `int8 = round_ties_even(codebook[q] · fp8_e4m3(s_rel))`, clamp `[-127, 127]` (a
//!    different clamp from the activation side), decoded as each 16-code group is staged into the
//!    GEMM's threadgroup tile. The even column is the **low** nibble of the stored byte, exactly as
//!    [`gen_core::w4a8_int8::dequant_packed_to_int8`] reads it on the host.
//! 3. **Epilogue** — the S4 epilogue verbatim: `((acc · xs) · s_channel) + bias`, exact int32
//!    accumulation, one rounding to the activation dtype at the store.
//!
//! The expected fixture comes from the pinned external comfy-kitchen executable (see the epic
//! source-of-truth / `local://sc26685-w4a8-primary-evidence.md`), never a host re-derivation; this
//! file schedules only the dequant + reused S4 quant/GEMM legs — it does not re-implement the oracle.
//!
//! # `.weight_s_rel` is loaded as raw E4M3 bytes, not as a float
//!
//! MLX has no fp8 storage dtype: its safetensors reader maps `F8_E4M3` to `uint8`
//! (`mlx/io/safetensors.cpp`), which is what [`Weights`] hands out. The group scales therefore stay
//! their on-disk bytes, [`W4A8Int8::validate_s_rel`] checks exactly those bytes, and the dequant
//! kernel decodes them — a scale tensor never enters the float domain on the way in.

use crate::{attention::slice_axis, weights::Weights, Result};
use gen_core::w4a8_int8::{W4A8Int8, MAX_DESCRIPTOR_BYTES};
use mlx_rs::{
    ops::{add, concatenate_axis, multiply, negative, subtract},
    transforms::eval,
    Array, Dtype,
};
#[cfg(any(test, feature = "test-support"))]
use {mlx_rs::fast::MetalKernel, std::cell::RefCell};

/// Output-column width of the decoded weight tile the H3 memory model still charges
/// (`gen_core::h3_memory::NativeBackend::Mlx { tile_columns }`).
///
/// Since sc-26691 the forward decodes inside the tiled GEMM's threadgroup loads and materializes no
/// decoded weight scratch at all, so this is now a conservative **upper bound** in that structural
/// model, not an allocation [`W4A8Linear::forward`] performs. It is kept rather than zeroed so the
/// published working-set accounting (and the artifacts generated from it) does not move in a kernel
/// change; the arithmetic never depended on it.
pub const TILE_COLS: usize = 256;

/// The ORIGINAL serial W4A8 kernels (sc-26685), kept only as the parity reference for the tiled
/// path in [`crate::int8_gemm`] (sc-26691); never on a production route.
///
/// The pinned activation quantizer of `int8_linear.cu::quantize_int8_rowwise_convrot_kernel`, fed
/// [`convrot_fht_f32`]'s F32 rotated row. `x` is F32 by construction; `T` is the *activation* dtype,
/// so the abmax/scale are computed on the unrounded F32 values and only
/// `quant_div_float_to_float<T>`'s three casts see the dtype: the numerator `T(rot)`, the divisor
/// `T(scale)`, and the quotient `T(rot / divisor)`.
///
/// This is NOT the S4 ConvRot quantizer: that codec's authority rotates in the source-dtype
/// dense path, so its input is already dtype-rounded and its kernel has no numerator cast. Sharing a
/// kernel name across two different sources would let whichever library compiled first define both.
#[cfg(any(test, feature = "test-support"))]
const REFERENCE_QUANT: &str = r#"
uint row = thread_position_in_grid.x;
if (row >= M) return;
float peak = 0.0f;
for (uint j=0; j<K; ++j) { float v=abs(float(x[row*K+j])); if (!isnan(v)) peak=max(peak,v); }
peak = min(peak, DTYPE==1 ? 3.38953139e38f : DTYPE==2 ? 65504.0f : 3.402823466e38f);
// quant_div_float_to_float<T>: T(numerator) / T(scale), quotient rounded to T, before nearbyintf.
float scale = max(peak * (1.0f / 127.0f), 1.0e-30f);
s[row] = scale;
float divisor = float(T(scale));
for (uint j=0; j<K; ++j) {
    float z = float(T(float(T(float(x[row*K+j]))) / divisor));
    q[row*K+j] = char(isnan(z) ? -128.0f : clamp(rint(z), -128.0f, 127.0f));
}
"#;
/// Packed `I8 [out, in/2]` → int8 `[tile, in]`, the `w4a8_gemm.cu` `dequant_int4_grouped_to_int8`
/// arithmetic: `int8 = round_ties_even(codebook[q] · fp8_e4m3fn(s_rel))`, clamp `[-127, 127]`.
#[cfg(any(test, feature = "test-support"))]
const REFERENCE_DEQUANT: &str = r#"
uint idx = thread_position_in_grid.x;
if (idx >= uint(WORK)) return;
uint k = uint(K);
uint c = idx % k;
uint col = idx / k;
uint out = uint(OUT0) + col;
// Frozen dialect: two int4 codes per stored byte, even column in the LOW nibble.
uint packed_byte = uint(uchar(packed[out * (k / 2u) + c / 2u]));
uint nibble = (c & 1u) == 0u ? (packed_byte & 0x0Fu) : (packed_byte >> 4);
// One E4M3FN group scale per 16 columns (`gen_core::w4a8_int8::GROUP_SIZE`).
uint scale_code = uint(uchar(srel[out * (k / 16u) + c / 16u]));
// OCP E4M3FN byte -> f32, bit-identical to `gen_core::comfy_quant::fp8_e4m3fn_to_f32`.
uint exponent = (scale_code >> 3) & 0x0Fu;
uint mantissa = scale_code & 0x07u;
float scale;
if (exponent == 0x0Fu && mantissa == 0x07u) {
    // NaN code: `validate_s_rel` refuses it at load, and the host oracle's `NaN as i8` is 0.
    scale = 0.0f;
} else {
    float magnitude = exponent == 0u ? float(mantissa) * 0.125f * 0.015625f
                                     : (1.0f + float(mantissa) * 0.125f)
                                           * exp2(float(exponent) - 7.0f);
    scale = (scale_code & 0x80u) == 0u ? magnitude : -magnitude;
}
w[col * k + c] = char(clamp(rint(codebook[nibble] * scale), -127.0f, 127.0f));
"#;

#[cfg(any(test, feature = "test-support"))]
thread_local! {
    // MetalKernel is thread-affine; never place one in a Send provider/model base.
    // These handles capture no arrays and share MLX's existing compiled-pipeline cache.
    static REFERENCE_KERNELS: RefCell<Option<(MetalKernel, MetalKernel, MetalKernel)>> = const { RefCell::new(None) };
}

#[cfg(any(test, feature = "test-support"))]
fn reference_kernels<T>(
    f: impl FnOnce(&MetalKernel, &MetalKernel, &MetalKernel) -> Result<T>,
) -> Result<T> {
    crate::nn::prepare_retained_compilation_thread();
    REFERENCE_KERNELS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.is_none() {
            *cache = Some((
                MetalKernel::new(
                    "sc_w4a8_int8_quant_v1",
                    &["x"],
                    &["q", "s"],
                    REFERENCE_QUANT,
                )?,
                MetalKernel::new(
                    "sc_convrot_int8_project_v1",
                    &["q", "w", "sx", "sw", "b"],
                    &["y"],
                    crate::convrot_int8::REFERENCE_PROJECT,
                )?,
                MetalKernel::new(
                    "sc_w4a8_int8_dequant_v1",
                    &["packed", "srel", "codebook"],
                    &["w"],
                    REFERENCE_DEQUANT,
                )?,
            ));
        }
        let (quant, project, dequant) = cache.as_ref().expect("initialized above");
        f(quant, project, dequant)
    })
}

/// MLX reports shapes as `i32`; the descriptor's geometry is `usize`.
fn dims(shape: &[i32]) -> Vec<usize> {
    shape.iter().map(|&v| v as usize).collect()
}

/// The pinned fused ConvRot butterfly, in F32, in the kernel's own addition order.
///
/// `int8_linear.cu::quantize_int8_rowwise_convrot_kernel` never multiplies by a materialized
/// rotation: it lands the row in shared memory as F32 and runs four radix-4 stages (strides
/// `1,4,16,64`), each `0.5f * h4_row_dot(d, x0, x1, x2, x3)` — three left-to-right binary32 adds and
/// one exact `* 0.5` — block-diagonally over `group`-wide blocks. The row absmax and `x_scale` are
/// then taken from those **F32** values, before `quant_div_float_to_float<T>` rounds anything to the
/// activation dtype.
///
/// A dense `matmul` by the same Hadamard basis is NOT equivalent: MLX re-associates the 256-term
/// dot, and a rotation performed in the source dtype rounds the rotated values first, so the peak
/// (and the divisor derived from it) moves by a dtype ulp — the review's BF16 `[1, 2^-8, 0, …]` row
/// has an exact rotated peak of `257/4096` that a BF16 rotation stores as `0.0625`, selecting a
/// different int8 code. Every add below is its own elementwise binary32 op, so the per-element order
/// is `((a+b)+c)+(-d)`, `((a+b)-c)+d`, `((a-b)+c)+d`, `((-a)+b+c)+d`, matching `h4_row_dot`.
fn convrot_fht_f32(x: &Array, rows: i32, k: i32, group: i32) -> Result<Array> {
    if !(group > 0 && group.count_ones() == 1 && group.trailing_zeros().is_multiple_of(2))
        || k % group != 0
    {
        return Err("W4A8 ConvRot group must be a power of four dividing K".into());
    }
    let mut cur = x.as_dtype(Dtype::Float32)?.reshape(&[rows, k])?;
    let half = Array::from_slice(&[0.5f32], &[]);
    let mut stride = 1i32;
    while stride < group {
        let blocks = rows * k / (4 * stride);
        let view = cur.reshape(&[blocks, 4, stride])?;
        let a = slice_axis(&view, 1, 0, 1)?;
        let b = slice_axis(&view, 1, 1, 1)?;
        let c = slice_axis(&view, 1, 2, 1)?;
        let d = slice_axis(&view, 1, 3, 1)?;
        let ab = add(&a, &b)?;
        let o0 = subtract(&add(&ab, &c)?, &d)?;
        let o1 = add(&subtract(&ab, &c)?, &d)?;
        let o2 = add(&add(&subtract(&a, &b)?, &c)?, &d)?;
        let o3 = add(&add(&add(&negative(&a)?, &b)?, &c)?, &d)?;
        let stage = concatenate_axis(
            &[
                multiply(&o0, &half)?,
                multiply(&o1, &half)?,
                multiply(&o2, &half)?,
                multiply(&o3, &half)?,
            ],
            1,
        )?;
        cur = stage.reshape(&[rows, k])?;
        stride *= 4;
    }
    Ok(cur)
}

/// Is this `.comfy_quant` payload the frozen `asym_w4a8_int8` dialect? A **probe**, not a validator:
/// everything it could get wrong is still refused by [`W4A8Int8::parse`] in
/// [`W4A8Linear::from_weights`]. A descriptor naming any other dialect (`convrot-int8-native`,
/// `int8_tensorwise`, …) is not this loader's to interpret, so the base falls through to the codec
/// that owns it.
fn is_w4a8_descriptor(payload: &[u8]) -> bool {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(payload) else {
        return false;
    };
    value.get("format").and_then(serde_json::Value::as_str) == Some("asym_w4a8_int8")
}

/// Compact native packed-W4A8 base, shareable by adapter-bearing projections.
#[derive(Clone)]
pub struct W4A8Linear {
    pub(crate) descriptor: W4A8Int8,
    packed: Array,
    codebook: Array,
    s_rel: Array,
    s_channel: Array,
    pub(crate) bias: Option<Array>,
}

impl W4A8Linear {
    /// Validated geometry cannot be mutated independently of the resident buffers.
    pub fn descriptor(&self) -> W4A8Int8 {
        self.descriptor
    }

    /// Admit only the frozen dialect plus its packed operands. No dense weight is ever built here:
    /// the codes stay packed and the int4→int8 expansion happens per tile in [`Self::forward`].
    ///
    /// Returns `Ok(None)` when `.comfy_quant` names another dialect (or is absent and nothing about
    /// the operands is W4A8), so the ConvRot/affine/dense fallbacks still run.
    pub fn from_weights(w: &Weights, base: &str, bias: bool) -> Result<Option<Self>> {
        let dk = format!("{base}.comfy_quant");
        let Some(payload) = w.get(&dk) else {
            // W4A8 has no implicit form: its `.weight` is two int4 codes per byte and would be read
            // as a dense int8 weight. Refuse loudly rather than fall through to a loader that would
            // misread this operand set.
            if w.get(&format!("{base}.weight_codebook")).is_some()
                || w.get(&format!("{base}.weight_s_rel")).is_some()
                || w.get(&format!("{base}.weight_s_channel")).is_some()
            {
                return Err(format!(
                    "{base}: W4A8 operands require an authoritative .comfy_quant descriptor"
                )
                .into());
            }
            return Ok(None);
        };
        if payload.dtype() != Dtype::Uint8
            || payload.ndim() != 1
            || payload.size() == 0
            || payload.size() > MAX_DESCRIPTOR_BYTES
        {
            return Err(format!("{base}: .comfy_quant must be a bounded U8 vector").into());
        }
        eval([payload])?;
        let descriptor_bytes = payload.as_slice::<u8>();
        if !is_w4a8_descriptor(descriptor_bytes) {
            return Ok(None);
        }
        if w.get(&format!("{base}.scales")).is_some()
            || w.get(&format!("{base}.biases")).is_some()
            || w.get(&format!("{base}.weight_scale")).is_some()
            || w.get(&format!("{base}.weight_correction")).is_some()
        {
            return Err(format!(
                "{base}: contradictory operands for a packed W4A8 base (no affine scales/biases, \
                 no per-row weight_scale, no weight_correction)"
            )
            .into());
        }
        let codes = w.require(&format!("{base}.weight"))?;
        let codebook = w.require(&format!("{base}.weight_codebook"))?;
        let s_rel = w.require(&format!("{base}.weight_s_rel"))?;
        let s_channel = w.require(&format!("{base}.weight_s_channel"))?;
        if codes.dtype() != Dtype::Int8
            || codebook.dtype() != Dtype::Float32
            || s_channel.dtype() != Dtype::Float32
            || s_rel.dtype() != Dtype::Uint8
        {
            return Err(format!(
                "{base}: W4A8 requires I8 packed codes, U8 F8_E4M3 s_rel bytes, and F32 \
                 codebook/s_channel without casting"
            )
            .into());
        }
        eval([codebook, s_rel, s_channel])?;
        let descriptor = W4A8Int8::parse(
            descriptor_bytes,
            &dims(codes.shape()),
            &dims(s_rel.shape()),
            &dims(s_channel.shape()),
            &dims(codebook.shape()),
        )
        .map_err(|e| crate::Error::Msg(format!("{base}: {e}")))?;
        descriptor
            .validate_s_rel(s_rel.as_slice::<u8>())
            .map_err(crate::Error::Msg)?;
        descriptor
            .validate_s_channel(s_channel.as_slice::<f32>())
            .map_err(crate::Error::Msg)?;
        descriptor
            .validate_codebook(codebook.as_slice::<f32>())
            .map_err(crate::Error::Msg)?;
        let bias = if bias {
            Some(w.require(&format!("{base}.bias"))?.clone())
        } else {
            None
        };
        if bias.as_ref().is_some_and(|b| {
            b.shape() != [descriptor.out_features as i32]
                || !matches!(b.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        }) {
            return Err(format!("{base}: invalid W4A8 bias").into());
        }
        // No rotation matrix is retained: the pinned W4A8 activation leg is a F32 butterfly inside
        // the quantizer ([`convrot_fht_f32`]), so a resident `[256, 256]` basis would be dead weight
        // *and* the wrong arithmetic (see the module doc).
        Ok(Some(Self {
            descriptor,
            packed: codes.clone(),
            codebook: codebook.clone(),
            s_rel: s_rel.clone(),
            s_channel: s_channel.clone(),
            bias,
        }))
    }

    /// Validated activation geometry, the pinned F32 butterfly and the F32 epilogue bias, shared by
    /// [`Self::forward`] and the test-only reference.
    fn prepare(&self, x: &Array) -> Result<(Array, Array, usize)> {
        let d = self.descriptor;
        if x.shape().last() != Some(&(d.in_features as i32))
            || !matches!(x.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err("W4A8 INT8 activation shape/dtype mismatch".into());
        }
        let rows = x.size() / d.in_features;
        if rows == 0
            || rows > i32::MAX as usize
            // The butterfly shapes and the kernel's `row*K` strides are i32-indexed.
            || x.size() > i32::MAX as usize
            || rows
                .checked_mul(d.out_features)
                .is_none_or(|s| s > u32::MAX as usize)
            || d.in_features
                .checked_mul(TILE_COLS)
                .is_none_or(|s| s > i32::MAX as usize)
        {
            return Err("W4A8 INT8 activation exceeds native index geometry".into());
        }
        // The pinned F32 butterfly, not a dense rotation and not a source-dtype one: the row
        // absmax/scale must come from these unrounded values (see [`convrot_fht_f32`]).
        let rotated = convrot_fht_f32(
            x,
            rows as i32,
            d.in_features as i32,
            d.rotation_group as i32,
        )?;
        // The reference adds `bias.float()` in the F32 epilogue: convert the stored bias STRAIGHT to
        // F32. A `b.as_dtype(x.dtype())` hop would round a BF16/F16-stored bias first and lose
        // precision the reference keeps (a BF16 `-0.999` would become `-1.0`).
        let bias = match &self.bias {
            Some(b) => b.as_dtype(Dtype::Float32)?,
            None => mlx_rs::ops::zeros::<f32>(&[d.out_features as i32])?,
        };
        Ok((rotated, bias, rows))
    }

    fn output_shape(&self, x: &Array) -> Vec<i32> {
        let mut shape = x.shape().to_vec();
        *shape.last_mut().unwrap() = self.descriptor.out_features as i32;
        shape
    }

    /// Packed W4A8 integer dot: one full-row ConvRot + int8 activation quantization, then ONE tiled
    /// GEMM ([`crate::int8_gemm`]) that decodes the packed weight inside its tile loads and applies
    /// the S4 int8 epilogue `((acc · xs) · s_channel) + bias`.
    ///
    /// The activation codes and scales are computed once for the full row before any weight code is
    /// decoded, as the pinned `launch_w4a8_gemm` does; accumulation is exact int32.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let d = self.descriptor;
        let (rotated, bias, rows) = self.prepare(x)?;
        let (codes, scales) = crate::int8_gemm::quantize_rows(
            &rotated,
            rows,
            d.in_features,
            x.dtype(),
            crate::int8_gemm::Numerator::W4A8,
        )?;
        let y = crate::int8_gemm::project(
            &codes,
            &scales,
            crate::int8_gemm::Weight::W4A8 {
                packed: &self.packed,
                s_rel: &self.s_rel,
                codebook: &self.codebook,
            },
            &self.s_channel,
            &bias,
            rows,
            d.out_features,
            d.in_features,
            x.dtype(),
        )?;
        Ok(y.reshape(&self.output_shape(x))?)
    }

    /// The original serial sc-26685 kernels (per-row quantizer, `[TILE_COLS, in]` decode scratch,
    /// one SIMD group per output element): the parity reference for [`Self::forward`]. Test support
    /// only; never on a production route.
    #[cfg(any(test, feature = "test-support"))]
    pub fn forward_reference(&self, x: &Array) -> Result<Array> {
        let d = self.descriptor;
        let k = d.in_features as i32;
        let n = d.out_features as i32;
        let (rotated, bias, rows) = self.prepare(x)?;
        let m = rows as i32;
        reference_kernels(|quant, project, dequant| {
            let mut q = quant
                .apply()
                .inputs([&rotated])
                .output_shape([m, k], Dtype::Int8)
                .output_shape([m], Dtype::Float32)
                .template_arg("T", x.dtype())
                .template_arg("M", m)
                .template_arg("K", k)
                .template_arg(
                    "DTYPE",
                    match x.dtype() {
                        Dtype::Bfloat16 => 1i32,
                        Dtype::Float16 => 2,
                        _ => 0,
                    },
                )
                .grid(m, 1, 1)
                .thread_group(m.min(256), 1, 1)
                .run()?
                .into_iter();
            let codes = q.next().ok_or("W4A8 quantizer returned no codes")?;
            let scales = q.next().ok_or("W4A8 quantizer returned no scales")?;
            let mut tiles: Vec<Array> = Vec::new();
            let mut out0 = 0i32;
            while out0 < n {
                let cols = TILE_COLS.min((n - out0) as usize) as i32;
                let scratch = dequant
                    .apply()
                    .inputs([&self.packed, &self.s_rel, &self.codebook])
                    .output_shape([cols, k], Dtype::Int8)
                    .template_arg("K", k)
                    .template_arg("OUT0", out0)
                    .template_arg("WORK", cols * k)
                    .grid(cols * k, 1, 1)
                    .thread_group((cols * k).min(256), 1, 1)
                    .run()?
                    .remove(0);
                let channel = slice_axis(&self.s_channel, 0, out0, cols)?;
                let tile_bias = slice_axis(&bias, 0, out0, cols)?;
                let mut tile = project
                    .apply()
                    .inputs([&codes, &scratch, &scales, &channel, &tile_bias])
                    .output_shape([m, cols], x.dtype())
                    .template_arg("T", x.dtype())
                    .template_arg("K", k)
                    .template_arg("N", cols)
                    .grid(32, cols, m)
                    .thread_group(32, 1, 1)
                    .run()?;
                let tile = tile.remove(0);
                eval([&tile])?;
                tiles.push(tile);
                out0 += cols;
            }
            let flat = if tiles.len() == 1 {
                tiles.pop().expect("one tile above")
            } else {
                concatenate_axis(&tiles, 1)?
            };
            Ok(flat.reshape(&self.output_shape(x))?)
        })
    }

    /// Retained native arrays only; no dense weight set can be materialized through this API.
    pub fn materialize(&self) -> Result<()> {
        eval(
            [&self.packed, &self.s_rel, &self.s_channel, &self.codebook]
                .into_iter()
                .chain(self.bias.iter()),
        )?;
        Ok(())
    }

    /// Projection-owned resident bytes (packed codes + the three scale/codebook companions + bias).
    /// No rotation matrix is resident — the activation leg is the F32 butterfly computed per call.
    pub fn nbytes(&self) -> usize {
        self.descriptor.resident_bytes() + self.bias.as_ref().map_or(0, Array::nbytes)
    }

    /// Decoded-weight scratch a single [`Self::forward`] allocates: none since sc-26691 — the packed
    /// codes are decoded into the GEMM's threadgroup tiles. ([`TILE_COLS`] stays the memory model's
    /// conservative upper bound.)
    pub fn tile_scratch_bytes(&self) -> usize {
        0
    }
}
