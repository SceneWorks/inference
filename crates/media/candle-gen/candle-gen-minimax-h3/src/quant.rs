//! Packed (pre-quantized) weight loading on the **candle** lane — the twin of
//! `mlx_gen_minimax_h3::quant` (sc-20267, epic sc-17137).
//!
//! A `q4` / `q8` tier stores each packed Linear as the MLX affine triple `{base}.weight` (u32
//! codes) + `{base}.scales` + `{base}.biases`. [`lin`] **auto-detects** it on the presence of
//! `{base}.scales` and builds the quantized base directly through the shared
//! [`candle_gen::quant::QLinear::from_packed_gs`] seam, so a published tier loads packed with **no
//! dense bf16 transient** — which at 66.28 GB is the difference between loading and being
//! OOM-killed. A dense `bf16` tier has no `.scales` and loads dense through the identical call, so
//! one loader serves every tier and nothing reads a manifest to decide.
//!
//! # Why this wrapper exists rather than `QLinear::linear_detect`
//!
//! The shared detect entry points ([`candle_gen::quant::QLinear::linear_detect`],
//! [`candle_gen::quant::AdaptLinear::linear_detect`]) are fed by a
//! [`candle_gen::candle_nn::VarBuilder`]. This crate's DiT loads through
//! [`candle_gen::Weights`] — a flat key→`Tensor` map — so those entry points are not callable
//! here. This module is the `Weights`-fed shape of the same detect, over the same shared
//! [`candle_gen::quant::AdaptLinear`] base, mirroring the `Weights`-fed `linear_detect` the
//! ideogram loader (sc-11104) already carries.
//!
//! **Scale fidelity.** [`candle_gen::Weights`] coerces every *floating* tensor to the compute dtype
//! at load (u32 code streams are left alone — the property this detect depends on). The published
//! tiers' scales/biases are **bf16** (`mlx_gen::quant::quantize_map` casts before packing), so that
//! coercion is lossless at bf16 and a widening at f32. The repack rounds them to `f16` inside the
//! `Q4_1` block regardless.
//!
//! # Which loaders route through here, and that split is a correctness property
//!
//! The converter's two dense-by-policy sets leave a specific set of tensors dense **in every tier**:
//! `mlx_gen_minimax_h3::convert::DENSE_BY_POLICY` (the DiT's float32 I/O heads, `context_embedder`,
//! `norm_out.linear` and every norm) and `TE_DENSE_BY_POLICY` (the text encoder's Qwen3 decoder
//! norms, the vision LayerNorms, `pos_embed` and `patch_embed.proj`). Those sets are exactly what
//! [`crate::dit::heads`] and [`crate::text_encoder`]'s norm loads read with a raw `Weights::require`,
//! so those loaders must *not* gain a packed path: a `require` that met a packed tensor would load
//! u32 codes where a float is expected (sc-14980's Mage `pos_embed` failure). [`guard_dense`] makes
//! that a loud, attributable load error instead.
//!
//! The loaders that **do** pack are the complete set the MLX lane packs, on both components:
//!
//! * DiT — [`crate::dit::layers::LinearNoBias`] and [`crate::dit::block::AdaLnProjection`].
//! * Text encoder (sc-20267) — the seven Qwen3 decoder projections behind `text_encoder`'s single
//!   `Proj` loader, plus the token table through [`embed`].
//!
//! The Qwen3-VL **vision tower** is the one `TE_PACK_SUFFIXES` member this crate does not serve: it
//! is consumed from `candle_gen_boogu::vision::VisionTower` through boogu's own loader, which
//! already refuses a packed tower loudly. So it is fail-loud rather than silent, but the candle lane
//! cannot yet load the vision shard of a packed text-encoder tier.
//!
//! # The group size is IMPORTED, never re-declared here
//!
//! The tiers pack at 64 (`mlx_gen_minimax_h3::convert::GROUP_SIZE`, the codebase-wide default), and
//! this module reads that number as [`candle_gen::quant::MLX_GROUP_SIZE`] — the very constant the
//! shared repack is written against. It is deliberately **not** re-exported under a crate-local
//! `pub const`: the repack *infers* the packed bit width from the weight/scales column ratio
//! assuming this group size, so a second declaration that drifted would derive an illegal width
//! from a perfectly good artifact instead of failing cleanly. That second declaration is also
//! exactly what `scripts/check-workspace.py`'s cross-backend geometry gate exists to catch.

use candle_gen::candle_core::quantized::GgmlDType;
use candle_gen::candle_core::{DType, Tensor};
use candle_gen::candle_nn::{Embedding, Linear};
use candle_gen::quant::{AdaptLinear, QEmbedding, QLinear, MLX_GROUP_SIZE as GROUP_SIZE};
use candle_gen::{CandleError, Result, Weights};

/// A loaded projection plus the device bytes its **frozen base** holds.
///
/// The byte count is recorded at load because it is not recoverable afterwards: a packed base never
/// materializes a dense weight, and [`candle_gen::quant::AdaptLinear`] exposes no accessor for the
/// resident `QTensor`. The MLX twin reads the same quantity back off its `AdaptableLinear`
/// (`mlx_gen_minimax_h3::quant::nbytes`); this is the same number, measured on the container candle
/// actually keeps resident (the repacked GGUF blocks) rather than on the MLX triple that produced
/// it.
#[derive(Debug)]
pub struct TieredLinear {
    /// The shared residual-capable projection — a frozen dense **or packed** base.
    pub linear: AdaptLinear,
    /// Device bytes the frozen base holds (quantized weight + any dense bias, or the dense weight +
    /// bias).
    pub base_bytes: usize,
}

/// Bytes a dense tensor occupies on its device.
fn dense_bytes(t: &Tensor) -> usize {
    t.elem_count() * t.dtype().size_in_bytes()
}

/// Bytes a GGUF-quantized `[out, in]` weight occupies — `blocks · type_size`, the resident container
/// the packed path keeps (the MLX scales/biases are folded INTO those blocks by the repack, so they
/// are not a separate term).
fn quant_bytes(dtype: GgmlDType, out: usize, in_features: usize) -> usize {
    (out * in_features / dtype.block_size()) * dtype.type_size()
}

/// Component label for the DiT's load errors — [`crate::dit`].
pub const DIT: &str = "dit";

/// Component label for the condition encoder's load errors — [`crate::text_encoder`].
pub const TEXT_ENCODER: &str = "te";

/// Refuse a **packed** tensor at a loader that has no packed path (sc-14980).
///
/// The dense-by-policy sets — [`crate::dit::heads`] on the DiT, the four norms in
/// [`crate::text_encoder`] — are read with a raw `Weights::require`, which hands back the u32 code
/// stream with no complaint at all: the shape check downstream may or may not catch it, and nothing
/// in the message would say *why*. This names the offending `.scales` key and says the tensor is
/// packed, so a tier that ever packed one fails at load rather than rendering garbage.
///
/// `component` is [`DIT`] or [`TEXT_ENCODER`]. It is a parameter rather than a constant in the
/// format string because both components route through this one guard, and a text-encoder norm
/// refused under a message that says `dit` — and that recites the DiT's dense-by-policy set — sends
/// the reader to the wrong converter list.
pub fn guard_dense(w: &Weights, component: &str, base: &str) -> Result<()> {
    let scales = format!("{base}.scales");
    if w.contains(&scales) {
        let policy = if component == TEXT_ENCODER {
            "`TE_DENSE_BY_POLICY` keeps the Qwen3 decoder norms, the vision LayerNorms, `pos_embed` \
             and `patch_embed.proj` dense in every tier"
        } else {
            "`DENSE_BY_POLICY` keeps the float32 I/O heads, `context_embedder`, `norm_out.linear` \
             and every norm dense in every tier"
        };
        return Err(CandleError::Msg(format!(
            "minimax-h3 {component} {base}: `{scales}` is present — this weight is MLX-PACKED, but \
             {base} is loaded by a dense-only path ({policy}). Reading its u32 codes as a float is \
             silent garbage (sc-14980). A tier that packs this tensor must add a real packed path \
             here first."
        )));
    }
    Ok(())
}

/// Load `{base}` as a shared [`candle_gen::quant::AdaptLinear`] — **packed** when `{base}.scales` is
/// present (built straight from the MLX triple at [`GROUP_SIZE`], no dense weight materialized),
/// else **dense** from `{base}.weight` cast to `dtype`.
///
/// `bias` additionally loads the dense `{base}.bias` (the quantization's own `{base}.biases` is
/// distinct and always loaded on the packed path). `base` is the full dotted key prefix (e.g.
/// `transformer_blocks.0.attn.to_out.0`), so the `.scales`/`.biases` siblings survive any
/// `to_out.0`-style nesting — build the base string first, then detect.
///
/// The logical `[out, in]` is recovered from the **scales** shape `[out, in/group]`, never from the
/// packed code column count: Q4 and Q8 pack a different number of codes per u32 word, so the codes
/// cannot answer the question and the scales can.
///
/// `component` is [`DIT`] or [`TEXT_ENCODER`] — see [`guard_dense`] for why it is a parameter.
pub fn lin(
    w: &Weights,
    component: &str,
    base: &str,
    bias: bool,
    dtype: DType,
) -> Result<TieredLinear> {
    let weight_key = format!("{base}.weight");
    let scales_key = format!("{base}.scales");
    let bias_key = format!("{base}.bias");

    if w.contains(&scales_key) {
        let wq = w.require(&weight_key)?;
        if wq.dtype() != DType::U32 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 {component} {base}: `{scales_key}` marks this weight packed, but \
                 `{weight_key}` loaded as {:?} rather than U32 — the packed code stream must reach \
                 the repack at its native width, and a float cast has already destroyed it",
                wq.dtype()
            )));
        }
        let scales = w.require(&scales_key)?;
        let biases = w.require(&format!("{base}.biases"))?;
        let dense_bias = if bias {
            Some(w.require(&bias_key)?.to_dtype(dtype)?)
        } else {
            None
        };
        let (out, groups) = scales.dims2()?;
        let in_features = groups * GROUP_SIZE;
        let bias_bytes = dense_bias.as_ref().map(dense_bytes).unwrap_or(0);
        let device = wq.device().clone();
        let packed =
            QLinear::from_packed_gs(&wq, &scales, &biases, dense_bias, GROUP_SIZE, &device)?;
        let quant_dtype = packed.quant_dtype().ok_or_else(|| {
            CandleError::Msg(format!(
                "minimax-h3 {component} {base}: the packed repack produced a dense projection"
            ))
        })?;
        return Ok(TieredLinear {
            base_bytes: quant_bytes(quant_dtype, out, in_features) + bias_bytes,
            linear: AdaptLinear::from_packed(packed, in_features, out),
        });
    }

    let weight = w.require(&weight_key)?.to_dtype(dtype)?;
    let dense_bias = if bias {
        Some(w.require(&bias_key)?.to_dtype(dtype)?)
    } else {
        None
    };
    let (out, in_features) = weight.dims2()?;
    let base_bytes = dense_bytes(&weight) + dense_bias.as_ref().map(dense_bytes).unwrap_or(0);
    Ok(TieredLinear {
        linear: AdaptLinear::from_dense(Linear::new(weight, dense_bias), in_features, out),
        base_bytes,
    })
}

/// A loaded token embedding plus the device bytes its table holds.
///
/// The embedding analogue of [`TieredLinear`], and recorded at load for the same reason: a packed
/// table never materializes densely and [`candle_gen::quant::QEmbedding`] exposes no byte accessor.
pub struct TieredEmbedding {
    /// The shared dense-or-packed token table.
    pub embedding: QEmbedding,
    /// Device bytes the table holds (the GGUF blocks when packed, the dense table otherwise).
    pub base_bytes: usize,
    /// The table's `hidden` width — recoverable from a packed table only via its scales, so it is
    /// captured here rather than re-derived at the call site.
    pub hidden: usize,
}

/// Load `{base}.weight` as a shared [`candle_gen::quant::QEmbedding`] — **packed** when
/// `{base}.scales` is present, else **dense**.
///
/// The token table is a genuine pack target: `mlx_gen_minimax_h3::convert::TE_PACK_SUFFIXES` names
/// `.embed_tokens`, so a published packed text-encoder tier ships this tensor as u32 codes. A raw
/// `Weights::require` would hand those codes back as if they were a float table and report nothing
/// at all — the sc-14980 class — so the detect has to live here rather than at the call site.
///
/// `dtype` is the **store** dtype the gathered rows come back as, matching the dense path exactly
/// (`candle-gen-boogu`'s Qwen3-VL precedent, sc-9410): the caller widens to the compute dtype after
/// the gather, not before.
///
/// # The packed forward dequantizes the whole table
///
/// [`candle_gen::quant::QEmbedding`]'s quantized forward dequantizes the full `[vocab, hidden]`
/// table before index-selecting, so the packed path trades a permanent dense table for a transient
/// one. That is the shared seam's established behaviour (boogu runs the identical path for this same
/// Qwen3-VL tower) and it is bounded: conditioning runs **once per render**, not once per step. The
/// resident saving is real; the transient is the documented cost of it.
pub fn embed(w: &Weights, base: &str, dtype: DType) -> Result<TieredEmbedding> {
    let weight_key = format!("{base}.weight");
    let scales_key = format!("{base}.scales");

    if w.contains(&scales_key) {
        let wq = w.require(&weight_key)?;
        if wq.dtype() != DType::U32 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 te {base}: `{scales_key}` marks this table packed, but `{weight_key}` \
                 loaded as {:?} rather than U32 — the packed code stream must reach the repack at \
                 its native width, and a float cast has already destroyed it",
                wq.dtype()
            )));
        }
        let scales = w.require(&scales_key)?;
        let biases = w.require(&format!("{base}.biases"))?;
        let (_vocab, groups) = scales.dims2()?;
        let hidden = groups * GROUP_SIZE;
        let device = wq.device().clone();
        let embedding =
            QEmbedding::from_packed_dtype_gs(&wq, &scales, &biases, &device, dtype, GROUP_SIZE)?;
        let base_bytes = match &embedding {
            QEmbedding::Quantized { table, .. } => table.storage_size_in_bytes(),
            QEmbedding::Dense(_) => {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 te {base}: the packed repack produced a dense embedding"
                )))
            }
        };
        return Ok(TieredEmbedding {
            embedding,
            base_bytes,
            hidden,
        });
    }

    let weight = w.require(&weight_key)?.to_dtype(dtype)?;
    let (_vocab, hidden) = weight.dims2()?;
    let base_bytes = dense_bytes(&weight);
    Ok(TieredEmbedding {
        embedding: QEmbedding::Dense(Embedding::new(weight, hidden)),
        base_bytes,
        hidden,
    })
}

#[cfg(test)]
pub(crate) mod testkit {
    //! Synthetic MLX packed triples, so the two wired loaders can be exercised without a real tier.

    use super::*;
    use candle_gen::candle_core::Device;

    /// A deterministic MLX **Q4** pack at [`GROUP_SIZE`] of an `[out, in]` weight: per-element
    /// 4-bit codes → u32 words (LSB-first nibbles), one `(scale, bias)` per group of 64.
    ///
    /// Returns `(wq [out, in/8] u32, scales [out, in/64], biases [out, in/64], grid [out, in])` —
    /// the packed triple a tier ships, plus the exact affine grid `scale·q + bias` it decodes to.
    /// The grid is the **dense reference** the packed forward is measured against: it is what the
    /// tier's producer quantized, so agreement with it is the property that matters (a dense
    /// checkpoint of the *unquantized* weight is a different tensor and would only measure
    /// quantization error).
    pub fn q4_pack(
        out_dim: usize,
        in_dim: usize,
        phase: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let dev = Device::Cpu;
        assert_eq!(in_dim % GROUP_SIZE, 0, "in_dim must be a multiple of 64");
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| (((i + phase) * 7 + i / 13) % 16) as u8)
            .collect();
        let groups_per_row = in_dim / GROUP_SIZE;
        let groups = out_dim * groups_per_row;
        let scales: Vec<f32> = (0..groups)
            .map(|k| 0.0625 * (((k + phase) % 7) as f32 + 1.0))
            .collect();
        let biases: Vec<f32> = (0..groups)
            .map(|k| -0.5 - 0.125 * ((k + phase) % 5) as f32)
            .collect();
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let k = row * groups_per_row + col / GROUP_SIZE;
                scales[k] * codes[i] as f32 + biases[k]
            })
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        (
            Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap(),
            Tensor::from_vec(scales, (out_dim, groups_per_row), &dev).unwrap(),
            Tensor::from_vec(biases, (out_dim, groups_per_row), &dev).unwrap(),
            Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap(),
        )
    }

    /// **Relative max-abs-diff** — `max|a-b| / max|b|`, the crate's established measure. Never
    /// cosine: cosine is scale-invariant, so it is structurally blind to a mis-decoded group scale,
    /// which is the defect class the packed path can produce.
    pub fn rel_max_abs(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.dims(), b.dims(), "shape");
        let max_abs = |t: &Tensor| -> f32 {
            t.abs()
                .expect("abs")
                .flatten_all()
                .expect("flat")
                .max(0)
                .expect("max")
                .to_vec0::<f32>()
                .expect("scalar")
        };
        let d = max_abs(&(a - b).expect("difference"));
        let scale = max_abs(b);
        if scale == 0.0 {
            d
        } else {
            d / scale
        }
    }

    /// Insert the packed triple for `{base}` into a `Weights`-bound key map, returning the affine
    /// grid the packed weight decodes to.
    pub fn insert_packed(
        map: &mut std::collections::HashMap<String, Tensor>,
        base: &str,
        out_dim: usize,
        in_dim: usize,
        phase: usize,
    ) -> Tensor {
        let (wq, scales, biases, grid) = q4_pack(out_dim, in_dim, phase);
        map.insert(format!("{base}.weight"), wq);
        map.insert(format!("{base}.scales"), scales);
        map.insert(format!("{base}.biases"), biases);
        grid
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::*;
    use super::*;
    use candle_gen::candle_core::Device;
    use std::collections::HashMap;

    /// The detect fires on the `.scales` sibling, keeps the base quantized, and never materializes
    /// a dense weight — the whole point of the packed path.
    #[test]
    fn a_scales_sibling_selects_the_packed_base() {
        let mut map = HashMap::new();
        insert_packed(&mut map, "p", 32, 128, 3);
        let w = Weights::from_map(map);
        let loaded = lin(&w, DIT, "p", false, DType::F32).expect("packed load");
        assert!(loaded.linear.is_packed(), "the `.scales` sibling must pack");
        assert_eq!(loaded.linear.base_shape(), (32, 128), "[out, in] recovered");
        assert_eq!(
            loaded.linear.quant_dtype(),
            Some(GgmlDType::Q4_1),
            "the MLX affine Q4 pack repacks losslessly into Q4_1"
        );
        assert_eq!(
            loaded.linear.matmul_strategy(),
            Some(candle_gen::quant::MatmulStrategy::DequantDense),
            "sc-7702: the packed forward must dequantize the WEIGHT, never quantize the activation"
        );
        // Resident bytes are the Q4_1 blocks, not a dense f32 weight.
        assert_eq!(loaded.base_bytes, (32 * 128 / 32) * 20);
        assert!(loaded.base_bytes < 32 * 128 * 4);
    }

    /// The packed forward reproduces a dense projection built from the **same dequantized grid** —
    /// gated on relative max-abs, never cosine.
    #[test]
    fn the_packed_forward_matches_its_dense_grid() {
        let (out, in_) = (32usize, 128usize);
        let mut map = HashMap::new();
        let grid = insert_packed(&mut map, "p", out, in_, 5);
        let w = Weights::from_map(map);
        let packed = lin(&w, DIT, "p", false, DType::F32).expect("packed load");

        let mut dense_map = HashMap::new();
        dense_map.insert("p.weight".to_string(), grid);
        let dense =
            lin(&Weights::from_map(dense_map), DIT, "p", false, DType::F32).expect("dense load");
        assert!(!dense.linear.is_packed(), "no `.scales` ⇒ dense");

        let x = Tensor::from_vec(
            (0..3 * in_)
                .map(|i| (i as f32 * 0.13).sin())
                .collect::<Vec<_>>(),
            (3, in_),
            &Device::Cpu,
        )
        .unwrap();
        let got = packed.linear.forward_upcast(&x).expect("packed forward");
        let want = dense.linear.forward_upcast(&x).expect("dense forward");
        let drift = rel_max_abs(&got, &want);
        println!("[quant] packed vs dense-grid rel-max-abs = {drift:.3e}");
        assert!(
            drift < 5e-3,
            "the Q4_1 repack is lossless up to the f16 scale/bias cast; got {drift:.3e}"
        );
    }

    /// A packed tensor at a **dense-only** loader is refused loudly, naming the key and the reason —
    /// the sc-14980 class this whole split exists to close.
    #[test]
    fn guard_dense_refuses_a_packed_tensor() {
        let mut map = HashMap::new();
        insert_packed(&mut map, "proj_in", 32, 128, 1);
        let w = Weights::from_map(map);
        let err = guard_dense(&w, DIT, "proj_in").expect_err("a packed head must be refused");
        let msg = err.to_string();
        assert!(msg.contains("proj_in.scales"), "{msg}");
        assert!(msg.contains("MLX-PACKED"), "{msg}");
        // The COMPONENT is attributed, and the message recites the DiT's policy list rather than
        // the text encoder's — the two share this guard, so a mislabelled refusal sends the reader
        // to the wrong converter constant.
        assert!(msg.contains("minimax-h3 dit proj_in"), "{msg}");
        assert!(msg.contains("DENSE_BY_POLICY"), "{msg}");
        assert!(
            !msg.contains("TE_DENSE_BY_POLICY"),
            "a DiT refusal must not cite the text encoder's list: {msg}"
        );

        // A dense weight set passes the same guard untouched.
        let mut dense = HashMap::new();
        dense.insert(
            "proj_in.weight".to_string(),
            Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap(),
        );
        guard_dense(&Weights::from_map(dense), DIT, "proj_in").expect("a dense head must pass");
    }

    /// A `.scales` sibling whose `.weight` is NOT a u32 code stream is a typed load error, not a
    /// silent repack of whatever floats happened to be there.
    #[test]
    fn a_packed_marker_over_a_float_weight_is_refused() {
        let mut map = HashMap::new();
        insert_packed(&mut map, "p", 32, 128, 2);
        // Overwrite the codes with a float tensor of the same shape.
        map.insert(
            "p.weight".to_string(),
            Tensor::zeros((32, 16), DType::F32, &Device::Cpu).unwrap(),
        );
        let err = lin(&Weights::from_map(map), DIT, "p", false, DType::F32)
            .expect_err("a float weight under a packed marker must be refused");
        assert!(err.to_string().contains("rather than U32"), "{err}");
    }
}
