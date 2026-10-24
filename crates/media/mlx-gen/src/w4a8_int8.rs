//! Native packed **W4A8 (asym_w4a8_int8)** projections, not a dense dequantization in disguise
//! (sc-26685).
//!
//! What is resident is the published packed form and nothing else: `I8 [out, in/2]` two-int4 codes,
//! `F8_E4M3 [out, in/16]` group scales, `F32 [out]` channel scales and one `F32 [16]` codebook, plus
//! the optional dense bias. The int4→int8 expansion happens **inside** [`W4A8Linear::forward`] on a
//! bounded `[TILE_COLS, in_features]` scratch that is a transient kernel output, so a projection
//! never holds a dense `[out, in]` int8 (or float) shadow of its weight.
//!
//! # The arithmetic is the pinned Kitchen chain, in three pieces
//!
//! 1. **Activation** — the S4 [`convrot_int8`](crate::convrot_int8) path verbatim: the regular
//!    (non-Walsh) `H₄` Kronecker rotation of `convrot_groupsize = 256` (normalized `1/16`) applied in
//!    the source dtype, then the row-wise int8 quantizer (`rint`, clamp `[-128, 127]`) with Kitchen's
//!    bf16 cast chain. The same `QUANT` kernel name *and* source as the ConvRot codec are used, so
//!    MLX compiles and caches ONE library for both. It runs **once per row, before any weight tile is
//!    decoded**: the pinned `launch_w4a8_codebook_gemm_chunked` computes its `xq`/`xs` outside the
//!    chunk loop, and a per-tile re-quantization would silently change the result.
//! 2. **Weight** — `int8 = round_ties_even(codebook[q] · fp8_e4m3(s_rel))`, clamp `[-127, 127]` (a
//!    different clamp from the activation side), decoded one output-column tile at a time
//!    ([`TILE_COLS`]). The even column is the **low** nibble of the stored byte, exactly as
//!    [`gen_core::w4a8_int8::dequant_packed_to_int8`] reads it on the host.
//! 3. **Epilogue** — the S4 `PROJECT` kernel verbatim: `((acc · xs) · s_channel) + bias`, int32
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
use gen_core::w4a8_int8::{regular_hadamard, W4A8Int8, MAX_DESCRIPTOR_BYTES};
use mlx_rs::{fast::MetalKernel, ops::matmul, transforms::eval, Array, Dtype};
use std::cell::RefCell;

/// Output-column width of one weight tile. The dequant kernel expands exactly `TILE_COLS ·
/// in_features` packed codes into a transient int8 scratch of that many bytes, so the decoded weight
/// is bounded no matter how wide the projection is.
///
/// The arithmetic is tile-invariant — every output column accumulates its whole `in_features` in one
/// pass and takes its own `s_channel`/bias in the epilogue — so this is a memory/dispatch trade-off,
/// never a numerics knob. Chunking is over **output columns** only: the activation codes and scales
/// are computed once for the full row and reused by every tile (see the module docs).
pub const TILE_COLS: usize = 256;

/// Identical to `convrot_int8.rs::QUANT` — same MLX kernel name *and* same source, so both codecs
/// share one compiled library. Keep the two strings in lockstep; a divergence would not miscompute
/// (each is its own kernel) but would make MLX clear and rebuild the library whenever a model mixes
/// the two codecs.
const QUANT: &str = r#"
uint row = thread_position_in_grid.x;
if (row >= M) return;
float peak = 0.0f;
for (uint j=0; j<K; ++j) { float v=abs(float(x[row*K+j])); if (!isnan(v)) peak=max(peak,v); }
peak = min(peak, DTYPE==1 ? 3.38953139e38f : DTYPE==2 ? 65504.0f : 3.402823466e38f);
// Kitchen rounds scale and quotient to the activation dtype before nearbyintf.
float scale = max(peak * (1.0f / 127.0f), 1.0e-30f);
s[row] = scale;
float divisor = float(T(scale));
for (uint j=0; j<K; ++j) {
    float z = float(T(float(x[row*K+j]) / divisor));
    q[row*K+j] = char(isnan(z) ? -128.0f : clamp(rint(z), -128.0f, 127.0f));
}
"#;
/// Identical to `convrot_int8.rs::PROJECT` (same kernel name and source — see [`QUANT`]). Run once
/// per output-column tile, with `N`/`sw`/`b` scoped to that tile, which is exactly one row of the
/// pinned `((acc · xs) · s_channel) + bias` epilogue.
const PROJECT: &str = r#"
uint lane = thread_position_in_threadgroup.x;
uint out = thread_position_in_grid.y;
uint row = thread_position_in_grid.z;
int acc = 0;
for (uint j=lane; j<K; j+=32) acc += int(q[row*K+j]) * int(w[out*K+j]);
acc = simd_sum(acc);
if (lane == 0) {
    float value = (float(acc) * sx[row]) * sw[out];
    y[row*N+out] = T(value + b[out]);
}
"#;
/// Packed `I8 [out, in/2]` → int8 `[tile, in]`, the `w4a8_gemm.cu` `dequant_int4_grouped_to_int8`
/// arithmetic: `int8 = round_ties_even(codebook[q] · fp8_e4m3fn(s_rel))`, clamp `[-127, 127]`.
const DEQUANT: &str = r#"
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

thread_local! {
    // MetalKernel is thread-affine; never place one in a Send provider/model base.
    // These handles capture no arrays and share MLX's existing compiled-pipeline cache.
    static KERNELS: RefCell<Option<(MetalKernel, MetalKernel, MetalKernel)>> = const { RefCell::new(None) };
}

fn kernels<T>(f: impl FnOnce(&MetalKernel, &MetalKernel, &MetalKernel) -> Result<T>) -> Result<T> {
    crate::nn::prepare_retained_compilation_thread();
    KERNELS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.is_none() {
            *cache = Some((
                MetalKernel::new("sc_convrot_int8_quant_v1", &["x"], &["q", "s"], QUANT)?,
                MetalKernel::new(
                    "sc_convrot_int8_project_v1",
                    &["q", "w", "sx", "sw", "b"],
                    &["y"],
                    PROJECT,
                )?,
                MetalKernel::new(
                    "sc_w4a8_int8_dequant_v1",
                    &["packed", "srel", "codebook"],
                    &["w"],
                    DEQUANT,
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
    rotation: Array,
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
        // The rotation is the shared, per-load constant the S4 loader builds: `convrot_rotation`
        // hands the SAME array to every projection of the model, and the basis itself comes from the
        // contract (`regular_hadamard`), so the two codecs cannot drift apart on the signs.
        let g = descriptor.rotation_group;
        let rotation = w.convrot_rotation(g, || {
            Array::from_slice(&regular_hadamard(g), &[g as i32, g as i32])
        });
        Ok(Some(Self {
            descriptor,
            packed: codes.clone(),
            codebook: codebook.clone(),
            s_rel: s_rel.clone(),
            s_channel: s_channel.clone(),
            bias,
            rotation,
        }))
    }

    /// Packed W4A8 integer dot: one full-row ConvRot + int8 activation quantization, then a bounded
    /// packed→int8 weight decode per output-column tile and the S4 int8 epilogue
    /// `((acc · xs) · s_channel) + bias` on that tile.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let d = self.descriptor;
        if x.shape().last() != Some(&(d.in_features as i32))
            || !matches!(x.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err("W4A8 INT8 activation shape/dtype mismatch".into());
        }
        let k = d.in_features as i32;
        let n = d.out_features as i32;
        let g = d.rotation_group as i32;
        let rows = x.size() / d.in_features;
        if rows == 0
            || rows > i32::MAX as usize
            || x.size() > u32::MAX as usize
            || rows
                .checked_mul(d.out_features)
                .is_none_or(|s| s > u32::MAX as usize)
            || d.in_features
                .checked_mul(TILE_COLS)
                .is_none_or(|s| s > i32::MAX as usize)
        {
            return Err("W4A8 INT8 activation exceeds native index geometry".into());
        }
        let m = rows as i32;
        // Rotation in the source dtype, as Kitchen's _rotate_activation, NOT a permanent float weight.
        let rotated =
            matmul(&x.reshape(&[-1, g])?, &self.rotation.as_dtype(x.dtype())?)?.reshape(&[m, k])?;
        let bias = match &self.bias {
            Some(b) => b.as_dtype(x.dtype())?.as_dtype(Dtype::Float32)?,
            None => mlx_rs::ops::zeros::<f32>(&[n])?,
        };
        kernels(|quant, project, dequant| {
            // Quantize ONCE for the full row, before any weight tile is touched: the pinned chunked
            // GEMM computes xq/xs outside its chunk loop and every tile reuses these two arrays.
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
                // Transient [cols, in] int8 weight scratch — the ONLY decoded weight that exists.
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
                tiles.push(tile.remove(0));
                out0 += cols;
            }
            // Each tile is already its own `[m, cols]` output; only the final join is a copy.
            let flat = if tiles.len() == 1 {
                tiles.pop().expect("one tile above")
            } else {
                mlx_rs::ops::concatenate_axis(&tiles, 1)?
            };
            let mut shape = x.shape().to_vec();
            *shape.last_mut().unwrap() = n;
            Ok(flat.reshape(&shape)?)
        })
    }

    /// Retained native arrays only; no dense weight set can be materialized through this API.
    pub fn materialize(&self) -> Result<()> {
        eval(
            [
                &self.packed,
                &self.s_rel,
                &self.s_channel,
                &self.codebook,
                &self.rotation,
            ]
            .into_iter()
            .chain(self.bias.iter()),
        )?;
        Ok(())
    }

    /// Projection-owned resident bytes (packed codes + the three scale/codebook companions + bias);
    /// the shared rotation constant is per-load and not counted here.
    pub fn nbytes(&self) -> usize {
        self.descriptor.resident_bytes() + self.bias.as_ref().map_or(0, Array::nbytes)
    }

    /// The transient a single [`Self::forward`] holds at once: one `[TILE_COLS, in_features]` int8
    /// weight scratch. Not resident — it is a kernel output freed with the turn — but a peak or
    /// residency budget that ignores it under-counts what a forward actually allocates, so it is
    /// reported rather than left implicit.
    pub fn tile_scratch_bytes(&self) -> usize {
        TILE_COLS * self.descriptor.in_features
    }
}
