//! [`Weights`]-fed stored-GGML helpers (SW-7285) over the shared
//! [`candle_layers::stored_ggml`] implementation.
//!
//! The `VarBuilder`-fed loaders (`lin_gs`, `AdaptLinear::linear_detect*`, the shared embeddings)
//! and the [`crate::candle_layers::CandleWeightSource`]-fed `linear_from_weights_gs` classify and
//! build stored GGML blocks in `candle_layers` itself. A family whose loader is fed by
//! [`Weights`] — a flat key → `Tensor` map with checkpoint-codec awareness — uses these: they add
//! the one rule a `Weights` map needs on top, that an **encoded** key (FP8 / MXFP8 / NVFP4 /
//! `.comfy_quant`) is ruled out by its source format before the tensor is fetched, because
//! `Weights::require` on an encoded key decodes it dense — exactly the transient a stored tier
//! exists to avoid. `candle-gen-minimax-h3`'s DiT / text-encoder loaders route through here.

pub use candle_layers::stored_ggml::{
    config_declares_ggml_storage, guard_dense_vb, qtensor_from_stored, stored_block_geometry,
    stored_block_tensor, stored_ggml_qtensor_vb, stored_ggml_vb, stored_ggml_vb_key,
    validate_stored, StoredGgmlRefusal, GGML_STORAGE, SHARED_CONTEXT,
};

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::Device;

use crate::{CandleError, Result, Weights};

/// The GGML block type and logical `[rows, cols]` of `{base}.weight` when it is a **stored GGML
/// block tensor** (`U8 [rows, cols / 32, block_bytes]`). `None` for a dense, MLX-packed or
/// checkpoint-encoded weight, which other loaders own.
pub fn stored_ggml_storage(w: &Weights, base: &str) -> Option<(GgmlDType, usize, usize)> {
    let key = format!("{base}.weight");
    if !w.contains(&key)
        || w.source_format_id(&key).is_some()
        || w.contains(&format!("{base}.comfy_quant"))
    {
        return None;
    }
    let t = w.require(&key).ok()?;
    stored_block_tensor(&t)
}

/// Rebuild `{base}.weight`'s stored GGML blocks as a resident [`QTensor`] — the blocks used
/// exactly as published, never dequantized, re-quantized or routed through the
/// saved-quantization cache. `None` when the weight is not a stored block tensor.
///
/// Refuses ([`validate_stored`]) a stored weight that also carries an MLX `.scales` sibling and one
/// whose blocks decode to a shape other than `shape`. `context` attributes the refusal (e.g.
/// `minimax-h3 dit`). The destination is `device`, else the host-packed set's target, else the
/// stored tensor's own device.
pub fn stored_ggml_qtensor(
    w: &Weights,
    context: &str,
    base: &str,
    shape: Option<(usize, usize)>,
    device: Option<&Device>,
) -> Result<Option<QTensor>> {
    let Some(geometry) = stored_ggml_storage(w, base) else {
        return Ok(None);
    };
    validate_stored(
        context,
        base,
        geometry,
        w.contains(&format!("{base}.scales")),
        shape,
    )
    .map_err(|e| CandleError::Msg(e.to_string()))?;
    let stored = w.require(&format!("{base}.weight"))?;
    let device = device
        .or_else(|| w.host_packed_target())
        .cloned()
        .unwrap_or_else(|| stored.device().clone());
    let qt = qtensor_from_stored(&stored, &device)
        .map_err(|e| CandleError::Msg(format!("{context} {base}: {e}")))?;
    Ok(Some(qt))
}

/// Refuse a stored GGML block at a [`Weights`]-fed **dense-only** loader
/// ([`StoredGgmlRefusal::DenseOnly`]). A loader that also refuses MLX `.scales` keeps its own
/// family-specific message for that half (it names the converter's dense-by-policy list).
pub fn guard_dense_stored(w: &Weights, context: &str, base: &str) -> Result<()> {
    if let Some((dtype, ..)) = stored_ggml_storage(w, base) {
        return Err(CandleError::Msg(
            StoredGgmlRefusal::DenseOnly {
                context: context.to_owned(),
                base: base.to_owned(),
                dtype,
            }
            .to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{pack_mlx_affine, MLX_GROUP_SIZE};
    use candle_core::{DType, Tensor};
    use std::collections::HashMap;

    fn triple(out: usize, inp: usize, bits: usize) -> (Tensor, Tensor, Tensor) {
        let grid = Tensor::arange(0f32, (out * inp) as f32, &Device::Cpu)
            .unwrap()
            .reshape((out, inp))
            .unwrap()
            .affine(0.0017, -0.3)
            .unwrap()
            .cos()
            .unwrap();
        pack_mlx_affine(&grid, bits, MLX_GROUP_SIZE).unwrap()
    }

    fn stored(out: usize, inp: usize, bits: usize) -> Tensor {
        let (wq, s, b) = triple(out, inp, bits);
        let qt =
            crate::quant::repack_packed_weight_uncached(&wq, &s, &b, MLX_GROUP_SIZE, &Device::Cpu)
                .unwrap();
        crate::candle_layers::llm_projection::to_ggml_block_tensor(&qt).unwrap()
    }

    /// The `CandleWeightSource`-fed shared loader builds a stored block as stored, refuses a
    /// `.scales` sibling, and the `Weights` helpers classify, build and guard the same tensor.
    /// Construction only — no forward runs.
    #[test]
    fn weights_fed_paths_build_and_refuse_stored_blocks() {
        let mut map = HashMap::new();
        map.insert("p.weight".to_owned(), stored(8, 128, 8));
        let w = Weights::from_map(map.clone());
        assert_eq!(
            stored_ggml_storage(&w, "p"),
            Some((GgmlDType::Q8_0, 8, 128))
        );
        let q = crate::quant::linear_from_weights(&w, "p", false).unwrap();
        assert_eq!(q.quant_dtype(), Some(GgmlDType::Q8_0));
        let qt = stored_ggml_qtensor(&w, "t", "p", Some((8, 128)), None)
            .unwrap()
            .unwrap();
        assert_eq!(qt.shape().dims2().unwrap(), (8, 128));
        let err = stored_ggml_qtensor(&w, "t", "p", Some((4, 128)), None).unwrap_err();
        assert!(err.to_string().contains("expected (4, 128)"), "{err}");
        let err = guard_dense_stored(&w, "t", "p").unwrap_err();
        assert!(err.to_string().contains("dense-only"), "{err}");

        let (_, s, b) = triple(8, 128, 8);
        map.insert("p.scales".to_owned(), s);
        map.insert("p.biases".to_owned(), b);
        let both = Weights::from_map(map);
        let err = crate::quant::linear_from_weights(&both, "p", false).unwrap_err();
        assert!(err.to_string().contains("never both"), "{err}");

        // A dense weight passes the guard and is not classified.
        let mut dense = HashMap::new();
        dense.insert(
            "d.weight".to_owned(),
            Tensor::zeros((8, 4, 34), DType::F32, &Device::Cpu).unwrap(),
        );
        let dense = Weights::from_map(dense);
        assert_eq!(stored_ggml_storage(&dense, "d"), None);
        guard_dense_stored(&dense, "t", "d").unwrap();
    }

    /// The stored-GGML tier's forward through the SHARED loaders equals the repacked MLX tier's:
    /// both dequantize identical blocks, so the outputs are bit-identical (q4 and q8, `VarBuilder`-
    /// and `Weights`-fed). Executes the dequant-dense matmul, so it is an explicit proof only.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn the_shared_stored_ggml_forward_equals_the_repacked_forward() {
        let (out, inp) = (16usize, 128usize);
        let x = Tensor::from_vec(
            (0..3 * inp)
                .map(|i| (i as f32 * 0.31).sin())
                .collect::<Vec<_>>(),
            (3, inp),
            &Device::Cpu,
        )
        .unwrap();
        let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for bits in [4usize, 8] {
            let (wq, s, b) = triple(out, inp, bits);
            let mut mlx = HashMap::new();
            mlx.insert("p.weight".to_owned(), wq);
            mlx.insert("p.scales".to_owned(), s);
            mlx.insert("p.biases".to_owned(), b);
            let mut ggml = HashMap::new();
            ggml.insert("p.weight".to_owned(), stored(out, inp, bits));

            let vb = |m: &HashMap<String, Tensor>| {
                candle_nn::VarBuilder::from_tensors(m.clone(), DType::F32, &Device::Cpu)
            };
            let want = crate::quant::lin_gs(&vb(&mlx), "p", inp, out, false, MLX_GROUP_SIZE)
                .unwrap()
                .forward(&x)
                .unwrap();
            let got = crate::quant::lin_gs(&vb(&ggml), "p", inp, out, false, MLX_GROUP_SIZE)
                .unwrap()
                .forward(&x)
                .unwrap();
            let want = flat(want);
            assert_eq!(flat(got), want, "q{bits}: VarBuilder-fed shared path");
            let got = crate::quant::linear_from_weights(&Weights::from_map(ggml), "p", false)
                .unwrap()
                .forward(&x)
                .unwrap();
            assert_eq!(flat(got), want, "q{bits}: Weights-fed shared path");
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::stored_ggml::tests::the_shared_stored_ggml_forward_equals_the_repacked_forward");
    }
}
