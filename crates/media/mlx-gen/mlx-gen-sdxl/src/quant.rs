//! Packed (pre-quantized) weight loading — the consume side of [`crate::convert`].
//!
//! A pre-quantized Q4/Q8 snapshot stores each quantized Linear as the packed triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`. The `lin` loader
//! **auto-detects** it by the presence of `{base}.scales` and builds the quantized module directly —
//! so a published Q4 snapshot loads packed with no dense fp16/f32 transient. A dense snapshot (no
//! `.scales`) loads dense exactly as before, so the same loader serves both.
//!
//! SDXL quantizes **three** components (the fork's `nn.quantize`, wired in [`crate::model::load`]):
//! the U-Net's true Linears (time/add-embedding MLPs, every cross-attention transformer's
//! attention + GEGLU FFN + `proj_in`/`proj_out`, each ResNet's `time_emb_proj`, plus the Kolors
//! `encoder_hid_proj` when present) and BOTH CLIP text encoders (q/k/v/out + mlp fc1/fc2 + the TE2
//! `text_projection`). The **convs stay dense** (`conv_in`/`conv_out`, resnet `conv1`/`conv2`, the
//! up/down samplers, and the 1×1-`conv_shortcut`-as-Linear — sc-3329), the norms stay dense, and the
//! CLIP token/position embeddings stay dense (gather lookups, not matmuls) — the converter's
//! per-component predicates match this scope exactly. The **VAE is never quantized** (it runs f32),
//! so all three tiers ship a dense VAE.
//!
//! This is the Group-B per-crate template (sc-8669), a thin wrapper over `mlx_gen::quant::lin` — plus
//! `lin_geglu_half`, the one SDXL-specific twist: the GEGLU `ff.net.0.proj` is stored on disk as a
//! single `[2·hidden, D]` tensor and row-split into the value/gate halves at load. Because
//! group-wise affine quantization is per-row independent, packing the whole `[2·hidden, D]` and
//! row-slicing the packed triple `[lo:hi]` on load is byte-identical to the load-time split-then-
//! quantize the dense path runs.

use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Load-time Q4/Q8 of one SDXL-family Linear, evaluated **one projection at a time** (sc-24245, the
/// `mlx_gen::quant::quantize_map` precedent): read the dense source first, then force the packed
/// result. A GPU `quantize` over a not-yet-read safetensors `Load` makes the Metal command buffer
/// wait on the disk read — past the GPU watchdog on a cold page cache — and the fused-LDM route
/// builds its components straight from lazy file arrays. Forcing the packed triple right away bounds
/// the dense transient to one projection. An already-packed base is left untouched (never read), as
/// [`AdaptableLinear::quantize`] leaves it, so a streamed pre-packed trunk stays lazy.
///
/// Inside [`guarded_load_quantize`] (the heavy bundle's load-time pack: the U-Net with its installed
/// IP K/V pairs, the ControlNet branches) each projection's pack is a checked iOS load quantum
/// ([`crate::quantum::LOAD_QUANTIZE`], epic 16774), priced by [`quantize_bytes`].
pub(crate) fn quantize_linear(linear: &mut AdaptableLinear, bits: i32) -> Result<()> {
    let Some((weight, bias)) = linear.dense_weight() else {
        return Ok(());
    };
    let bytes = quantize_bytes(weight, bias, bits);
    // Actual borrowed owner observations. These page-rounded logical payload sums never
    // establish unique backing or a physical credit.
    let payload =
        |array: &Array| mlx_gen::ios_quantum::allocated_bytes(crate::quantum::bytes(array));
    mlx_gen::ios_quantum::diagnostic_metadata(
        "sdxl",
        "quantize-linear-before",
        &[
            ("schema", 1),
            ("elements", weight.size() as u64),
            ("bits", bits.max(0) as u64),
            ("logical_bytes", payload(weight) + bias.map_or(0, payload)),
            ("budget", bytes),
            ("physical_certificate", 0),
        ],
    );
    if GUARDED_LOAD_QUANTIZE.with(std::cell::Cell::get) {
        mlx_gen::ios_quantum::before(crate::quantum::LOAD_QUANTIZE, bytes)?;
    }
    linear.materialize_weights()?;
    linear.quantize(bits, None)?;
    linear.materialize_weights()?;
    if let Some((codes, scales, offsets, bias, _, packed_bits)) = linear.quantized_params() {
        mlx_gen::ios_quantum::diagnostic_metadata(
            "sdxl",
            "quantize-linear-after",
            &[
                ("schema", 1),
                ("bits", packed_bits.max(0) as u64),
                (
                    "logical_bytes",
                    payload(codes) + payload(scales) + payload(offsets) + bias.map_or(0, payload),
                ),
                ("physical_certificate", 0),
            ],
        );
    }
    Ok(())
}

/// The transient of one projection's load-time pack ([`AdaptableLinear::quantize`]) beside its
/// resident dense `weight` `[out, in]` (+ `bias`): the bf16 casts it packs from and the packed
/// triple (`bits`-wide u32 codes, a bf16 scale and bias per [`GROUP_SIZE`] group), each its own
/// buffer.
pub(crate) fn quantize_bytes(weight: &Array, bias: Option<&Array>, bits: i32) -> u64 {
    let elements = weight.size() as u64;
    let bias = bias.map_or(0, |b| b.size() as u64 * 2);
    quantize_bytes_from_metadata(elements, bias, bits)
}

/// The SAME existing raw pack budget over metadata; `bias` is its BF16 byte extent. No Array,
/// readiness/allocator query or physical-allocation certificate is involved.
pub(crate) fn quantize_bytes_from_metadata(elements: u64, bias: u64, bits: i32) -> u64 {
    let packed = elements
        .saturating_mul(bits.max(0) as u64)
        .div_ceil(8)
        .saturating_add(elements.div_ceil(GROUP_SIZE as u64).saturating_mul(4));
    elements
        .saturating_mul(2)
        .saturating_add(bias)
        .saturating_add(packed)
        .saturating_add(mlx_gen::ios_quantum::page_slack(5))
}

thread_local! {
    static GUARDED_LOAD_QUANTIZE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run the heavy bundle's load-time pack (`model::load_heavy`) with every projection's pack checked
/// as a [`crate::quantum::LOAD_QUANTIZE`] quantum. Restored on return or unwind.
pub(crate) fn guarded_load_quantize<T>(body: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            GUARDED_LOAD_QUANTIZE.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(GUARDED_LOAD_QUANTIZE.with(|flag| flag.replace(true)));
    body()
}

/// Private projection callback over the ACTUAL module quant walk. Runtime delegates to the same
/// quantize_linear body; header pricing records/applies typed pack effects without fake Arrays.
pub(crate) type QuantizeLinear<'a> = dyn FnMut(&str, &mut AdaptableLinear, i32) -> Result<()> + 'a;

pub(crate) fn child_path(prefix: &str, child: &str) -> String {
    if prefix.is_empty() {
        child.to_owned()
    } else {
        format!("{prefix}.{child}")
    }
}

/// Group size the converter writes — the codebase-wide `mlx_gen::quant::DEFAULT_GROUP_SIZE` (64).
pub(crate) const GROUP_SIZE: i32 = 64;

/// Load `{base}` as an [`AdaptableLinear`] at SDXL's [`GROUP_SIZE`] — packed when `{base}.scales`
/// is present (a pre-quantized snapshot), else dense. The shared [`mlx_gen::quant::lin`].
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}

/// Actual base-model GEGLU row selector, shared by construction and header normalization.
/// Adapter UP routing intentionally has a different existing odd-row floor split.
pub(crate) fn geglu_half_ranges(output_rows: i32) -> Result<[(i32, i32); 2]> {
    if output_rows % 2 != 0 {
        return Err(mlx_gen::Error::Msg(format!(
            "sdxl GEGLU: ff.net.0.proj has odd output dim {output_rows}; cannot split value/gate halves",
        )));
    }
    let half = output_rows / 2;
    Ok([(0, half), (half, output_rows)])
}

/// Load the value/gate **half** of a GEGLU `ff.net.0.proj` — rows `[lo, hi)` of the stored
/// `[2·hidden, D]` Linear (bias included). Packed when `{base}.scales` is present (row-slice the u32
/// codes + scales + biases + dense bias along axis 0 — valid because group-wise affine quantization
/// is per-row independent, so this is byte-identical to the dense path's split-then-quantize), else
/// row-slice the dense weight + bias. `base` is `…transformer_blocks.{k}.ff.net.0.proj`.
pub(crate) fn lin_geglu_half(w: &Weights, base: &str, lo: i32, hi: i32) -> Result<AdaptableLinear> {
    let idx = Array::from_slice(&(lo..hi).collect::<Vec<i32>>(), &[hi - lo]);
    let row_slice = |a: &Array| -> Result<Array> { Ok(a.take_axis(&idx, 0)?) };
    let bias_full = w.require(&format!("{base}.bias"))?;
    let bias = Some(row_slice(bias_full)?);
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let biases = w.require(&format!("{base}.biases"))?;
        // F-011: the shared derivation validates the untrusted packed shapes (2-D, non-zero in
        // dim, bits ∈ {4, 8}) with a typed error instead of a panic/divide-by-zero.
        let bits = mlx_gen::quant::packed_bits(wq, scales, GROUP_SIZE)?;
        return Ok(AdaptableLinear::from_quantized_parts(
            row_slice(wq)?,
            row_slice(scales)?,
            row_slice(biases)?,
            bias,
            GROUP_SIZE,
            bits,
        ));
    }
    Ok(AdaptableLinear::dense(
        row_slice(w.require(&format!("{base}.weight"))?)?,
        bias,
    ))
}

/// Header-only projection geometry at the adapter-install boundary. A load-time pack happens
/// AFTER adapter installation, whereas an existing affine pack takes the residual install branch.
/// Keeping both states is necessary even when their final requested tier is identical.
#[derive(Clone, Debug)]
pub(crate) struct HeaderLinear {
    pub shape: [i32; 2],
    pub source_packed_bits: Option<i32>,
    pub weight_dtype: Dtype,
    pub scales_dtype: Option<Dtype>,
    pub affine_biases_dtype: Option<Dtype>,
    pub packed_codes_shape: Option<Vec<usize>>,
    pub packed_scales_shape: Option<Vec<usize>>,
    pub packed_biases_shape: Option<Vec<usize>>,
}

pub(crate) fn header_dtype(dtype: mlx_gen::gen_core::weightsmeta::Dtype) -> Result<Dtype> {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    Ok(match dtype {
        Stored::BOOL => Dtype::Bool,
        Stored::U8 => Dtype::Uint8,
        Stored::I8 => Dtype::Int8,
        Stored::U16 => Dtype::Uint16,
        Stored::I16 => Dtype::Int16,
        Stored::U32 => Dtype::Uint32,
        Stored::I32 => Dtype::Int32,
        Stored::U64 => Dtype::Uint64,
        Stored::I64 => Dtype::Int64,
        Stored::F16 => Dtype::Float16,
        Stored::BF16 => Dtype::Bfloat16,
        Stored::F32 => Dtype::Float32,
        // The actual pinned native safetensors decoder stores these as uint8, NOT decoded fp8.
        Stored::F8_E4M3 | Stored::F8_E8M0 => Dtype::Uint8,
        Stored::F64 | Stored::F8_E5M2 => {
            return Err(mlx_gen::Error::Unsupported(
                "sdxl native safetensors loader does not accept this source dtype".into(),
            ))
        }
    })
}

/// `loader::cast_for_load`'s actual iOS leaf cast, after the native source dtype conversion.
pub(crate) fn loaded_header_dtype(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    header: &mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader,
) -> Result<Dtype> {
    loaded_dtype_in_component(headers.iter().any(|h| h.name.ends_with(".scales")), header)
}

/// [`loaded_header_dtype`] given the component-wide packed flag (any `.scales` name), which is a
/// property of the component's target NAMES and therefore identical in every fused source order.
pub(crate) fn loaded_dtype_in_component(
    packed: bool,
    header: &mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader,
) -> Result<Dtype> {
    let stored = header_dtype(header.dtype)?;
    if !packed
        || (header.is_float()
            && !header.name.ends_with(".scales")
            && !header.name.ends_with(".biases"))
    {
        Ok(crate::model::DTYPE)
    } else {
        Ok(stored)
    }
}

/// `quant::lin`'s actual `.scales` selector and `packed_bits` validator, without an Array. The
/// component-wide packed flag follows `loader::cast_for_load`; this exporter describes the iOS
/// load, whose floating dense leaves are cast even when the component is prepacked.
pub(crate) fn linear_from_headers(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    base: &str,
) -> Result<HeaderLinear> {
    let require = |suffix: &str| {
        headers
            .iter()
            .find(|h| h.name == format!("{base}.{suffix}"))
            .ok_or_else(|| mlx_gen::Error::MissingTensor(format!("{base}.{suffix}")))
    };
    let shape = |h: &mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader| -> Result<Vec<i32>> {
        h.shape
            .iter()
            .map(|d| {
                i32::try_from(*d).map_err(|_| {
                    mlx_gen::Error::Msg(
                        "sdxl header normalization: projection dimension exceeds native i32".into(),
                    )
                })
            })
            .collect()
    };
    let weight = require("weight")?;
    let wshape = shape(weight)?;
    if let Some(scales) = headers.iter().find(|h| h.name == format!("{base}.scales")) {
        let sshape = shape(scales)?;
        let wd = header_dtype(weight.dtype)?;
        let sd = header_dtype(scales.dtype)?;
        let bits = mlx_gen::quant::packed_bits_from_metadata(&wshape, wd, &sshape, sd, GROUP_SIZE)?;
        let biases = require("biases")?;
        // The runtime loads this exact companion, not an inferred bf16 replacement.
        let bd = header_dtype(biases.dtype)?;
        return Ok(HeaderLinear {
            shape: [
                wshape[0],
                sshape[1].checked_mul(GROUP_SIZE).ok_or_else(|| {
                    mlx_gen::Error::Msg("sdxl header input width overflow".into())
                })?,
            ],
            source_packed_bits: Some(bits),
            weight_dtype: wd,
            scales_dtype: Some(sd),
            affine_biases_dtype: Some(bd),
            packed_codes_shape: Some(weight.shape.clone()),
            packed_scales_shape: Some(scales.shape.clone()),
            packed_biases_shape: Some(biases.shape.clone()),
        });
    }
    let [out, input] = wshape.as_slice() else {
        return Err(mlx_gen::Error::Msg(
            "sdxl header Linear weight is not a matrix".into(),
        ));
    };
    let dtype = loaded_header_dtype(headers, weight)?;
    Ok(HeaderLinear {
        shape: [*out, *input],
        source_packed_bits: None,
        weight_dtype: dtype,
        scales_dtype: None,
        affine_biases_dtype: None,
        packed_codes_shape: None,
        packed_scales_shape: None,
        packed_biases_shape: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::Dtype;

    /// The invariant that makes [`lin_geglu_half`]'s packed row-slice byte-identical to the dense
    /// split-then-quantize: group-wise affine quantization is per-row, so row-slicing axis 0 of the
    /// packed triple commutes with the split. Pack a `[4, 128]` weight whole, row-slice `[0:2]` of
    /// the packed codes, and compare to packing just rows `[0:2]` of the dense weight — byte-equal.
    #[test]
    fn packed_row_slice_commutes_with_split() {
        let full = Array::from_slice(
            &(0..4 * 128).map(|i| (i as f32).cos()).collect::<Vec<_>>(),
            &[4, 128],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let (wq, _sc, _bi) = quantize(&full, GROUP_SIZE, 4).unwrap();
        let idx = Array::from_slice(&[0i32, 1], &[2]);
        let sliced_wq = wq.take_axis(&idx, 0).unwrap();
        let half = full.take_axis(&idx, 0).unwrap();
        let (ewq, _, _) = quantize(&half, GROUP_SIZE, 4).unwrap();
        assert!(
            eq(&sliced_wq, &ewq)
                .unwrap()
                .all(None)
                .unwrap()
                .item::<bool>(),
            "packed row-slice must equal packing the sliced rows"
        );
    }
}
