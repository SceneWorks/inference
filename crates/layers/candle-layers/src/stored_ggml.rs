//! **Stored GGML block tiers** on the shared packed-linear path (SW-7285).
//!
//! SceneWorks hosts Candle q4 / q8 tiers in two forms:
//!
//! * the MLX affine triple — `{base}.weight` (U32 codes) + `{base}.scales` + `{base}.biases` — which
//!   the shared loaders repack to GGML at first load ([`crate::quant::repack_packed_weight`], routed
//!   through the saved-quantization cache); and
//! * a **Candle-native GGML tier**, where the converter has already done that repack offline and
//!   persisted the blocks: ONE `{base}.weight`, dtype `U8`, shape `[out, in / 32, block_bytes]`
//!   (`Q4_1` = 20 B, `Q8_0` = 34 B per 32 weights), with no `.scales` / `.biases`. The component's
//!   `config.json` carries `quantization = {"bits": N, "storage": "ggml"}` ([`GGML_STORAGE`]).
//!
//! This module is the one implementation of the second form's detect, validation and build, shared
//! by the `VarBuilder`-fed loaders ([`crate::quant::lin_gs`], [`crate::quant::embedding_dtype_gs`],
//! [`crate::AdaptLinear::linear_detect_gs`]), the [`crate::CandleWeightSource`]-fed
//! [`crate::quant::linear_from_weights_gs`], the `Weights`-fed helpers in `candle-gen`
//! (`candle_gen::quant::stored_ggml`, which `candle-gen-minimax-h3` routes through), and the
//! mapped-set family loaders (Ideogram, Boogu / Mage) through [`stored_ggml_mmap_host`]. Every one
//! reads the blocks on the host and uploads only the final [`QTensor`].
//!
//! A stored block tensor is classified **from its header alone** — `U8`, rank 3, and a last axis
//! equal to one of the persisted block sizes ([`crate::llm_projection::ggml_block_storage`]) — and is
//! built into a resident [`QTensor`] exactly as stored: never dequantized, re-quantized, or routed
//! through the saved-quantization cache. Three malformed shapes are refused with a typed
//! [`StoredGgmlRefusal`] rather than loaded: a projection carrying BOTH a stored block and a
//! `.scales` sibling, a block whose decoded `[rows, cols]` disagrees with the shape the model
//! expects, and a stored block reaching a **dense-only** loader (the sc-14980 class: reading GGML
//! bytes as a float is silent garbage).

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::llm_projection::{from_ggml_block_tensor, ggml_block_storage};

/// The `quantization.storage` value a Candle-native tier's component `config.json` carries: every
/// packed projection is persisted as a stored GGML block tensor rather than an MLX affine triple.
/// The same tag `candle-llm`'s prepared tiers use.
pub const GGML_STORAGE: &str = gen_core::stored_ggml::GGML_STORAGE;

/// Whether a component `config.json` declares a Candle-native GGML tier
/// (`quantization.storage == "ggml"`). Informational only: every loader classifies each tensor by
/// its header, so a mislabelled config can neither hide nor invent a stored block.
pub fn config_declares_ggml_storage(cfg: &serde_json::Value) -> bool {
    gen_core::stored_ggml::config_declares_ggml_storage(cfg)
}

/// The GGML block type and logical `(rows, cols)` of a **rank-3** stored GGML block tensor with the
/// given stored dtype and shape; `None` for anything else (dense, MLX codes, a stacked rank-4 MoE
/// tensor, a checkpoint codec).
pub fn stored_block_geometry(dtype: DType, shape: &[usize]) -> Option<(GgmlDType, usize, usize)> {
    if shape.len() != 3 {
        return None;
    }
    ggml_block_storage(dtype == DType::U8, shape)
}

/// [`stored_block_geometry`] of a live tensor.
pub fn stored_block_tensor(t: &Tensor) -> Option<(GgmlDType, usize, usize)> {
    stored_block_geometry(t.dtype(), t.dims())
}

/// [`stored_block_geometry`] read off a safetensors **header** (no payload): the tier validators and
/// memory pricers that walk `gen_core` headers classify a stored block by this one rule. Rank 3 only;
/// a stacked rank-4 MoE tensor is [`stored_stacked_header_geometry`].
pub fn stored_header_geometry(
    header: &gen_core::SafetensorsTensorHeader,
) -> Option<(GgmlDType, usize, usize)> {
    if header.shape.len() != 3 {
        return None;
    }
    ggml_block_storage(
        header.dtype == gen_core::weightsmeta::Dtype::U8,
        &header.shape,
    )
}

/// A **stacked** stored block header (`U8 [E, out, in / 32, block_bytes]`, the fused-MoE form):
/// `(dtype, experts, out, in)`. `None` for any other tensor.
pub fn stored_stacked_header_geometry(
    header: &gen_core::SafetensorsTensorHeader,
) -> Option<(GgmlDType, usize, usize, usize)> {
    let [experts, out, ..] = header.shape[..] else {
        return None;
    };
    if header.shape.len() != 4 {
        return None;
    }
    let (dtype, _, cols) = ggml_block_storage(
        header.dtype == gen_core::weightsmeta::Dtype::U8,
        &header.shape,
    )?;
    Some((dtype, experts, out, cols))
}

/// The nominal weight bit width a persisted GGML block type stands for — the `bits` a tier's
/// `quantization` marker declares for it (`Q4_*` → 4, `Q8_0` → 8). `None` for a block type no
/// SceneWorks q4/q8 tier persists.
pub fn stored_ggml_bits(dtype: GgmlDType) -> Option<usize> {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q4K => Some(4),
        GgmlDType::Q8_0 => Some(8),
        _ => None,
    }
}

/// A typed refusal of a malformed stored GGML projection. `context` names the loader (e.g.
/// `minimax-h3 dit`, `stored-ggml`) so the message attributes the failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StoredGgmlRefusal {
    /// `{base}.weight` is a stored block AND `{base}.scales` exists: the two forms are mutually
    /// exclusive, and silently preferring one would load whichever a converter bug left behind.
    #[error(
        "{context} {base}: `{base}.weight` is a stored {dtype:?} GGML block tensor but \
         `{base}.scales` is also present — a tier stores a projection either as GGML blocks or as \
         an MLX affine triple, never both"
    )]
    BothForms {
        context: String,
        base: String,
        dtype: GgmlDType,
    },
    /// The stored blocks decode to a `[rows, cols]` other than the model's `[out, in]`.
    #[error(
        "{context} {base}: stored {dtype:?} blocks decode to [{rows}, {cols}], expected {expected:?}"
    )]
    ShapeMismatch {
        context: String,
        base: String,
        dtype: GgmlDType,
        rows: usize,
        cols: usize,
        expected: (usize, usize),
    },
    /// A stored block reached a loader that has no quantized path.
    #[error(
        "{context} {base}: `{base}.weight` is a stored {dtype:?} GGML block tensor, but {base} is \
         loaded by a dense-only path. Reading GGML blocks as a float is silent garbage (sc-14980); \
         the Candle tier converter must leave this tensor dense."
    )]
    DenseOnly {
        context: String,
        base: String,
        dtype: GgmlDType,
    },
}

impl From<StoredGgmlRefusal> for candle_core::Error {
    fn from(refusal: StoredGgmlRefusal) -> Self {
        candle_core::Error::wrap(refusal)
    }
}

/// Validate a classified stored block before it is built: refuse a `.scales` sibling
/// ([`StoredGgmlRefusal::BothForms`]) and a decoded shape other than `expected`
/// ([`StoredGgmlRefusal::ShapeMismatch`]).
pub fn validate_stored(
    context: &str,
    base: &str,
    (dtype, rows, cols): (GgmlDType, usize, usize),
    has_scales: bool,
    expected: Option<(usize, usize)>,
) -> std::result::Result<(), StoredGgmlRefusal> {
    if has_scales {
        return Err(StoredGgmlRefusal::BothForms {
            context: context.to_owned(),
            base: base.to_owned(),
            dtype,
        });
    }
    if let Some(expected) = expected {
        if (rows, cols) != expected {
            return Err(StoredGgmlRefusal::ShapeMismatch {
                context: context.to_owned(),
                base: base.to_owned(),
                dtype,
                rows,
                cols,
                expected,
            });
        }
    }
    Ok(())
}

/// Rebuild a rank-3 stored block tensor as a resident [`QTensor`] on `device`, blocks used exactly
/// as stored. Never consults the saved-quantization cache.
pub fn qtensor_from_stored(stored: &Tensor, device: &Device) -> Result<QTensor> {
    from_ggml_block_tensor(stored, device).map_err(|e| match e {
        crate::llm_projection::Error::Candle(inner) => inner,
        other => candle_core::Error::Msg(other.to_string()),
    })
}

/// Label the shared `VarBuilder` / `CandleWeightSource` loaders attribute their refusals to.
pub const SHARED_CONTEXT: &str = "stored-ggml";

/// The stored-block geometry of `key` in `vb`, read from the backend's tensor header **without
/// loading the payload**. A backend that does not report the stored dtype (or a codec-planned
/// tensor) is never classified as a stored block: its dense read keeps its own shape check, which a
/// rank-3 block tensor fails loudly rather than silently.
pub fn stored_ggml_vb_key(vb: &VarBuilder, key: &str) -> Result<Option<(GgmlDType, usize, usize)>> {
    if !vb.contains_tensor(key) {
        return Ok(None);
    }
    let Some(info) = vb.tensor_info(key)? else {
        return Ok(None);
    };
    if info.source_packed || info.padded_shape.is_some() {
        return Ok(None);
    }
    Ok(info
        .dtype
        .and_then(|dtype| stored_block_geometry(dtype, info.shape.dims())))
}

/// [`stored_ggml_vb_key`] for `{base}.weight`.
pub fn stored_ggml_vb(vb: &VarBuilder, base: &str) -> Result<Option<(GgmlDType, usize, usize)>> {
    stored_ggml_vb_key(vb, &format!("{base}.weight"))
}

/// Build `{base}.weight`'s stored GGML blocks from `vb` as a resident [`QTensor`] on `device`, or
/// `None` when it is not a stored block tensor. The blocks are read on the host (no device round
/// trip) and uploaded once as the final `QTensor`. Refuses ([`validate_stored`]) a `.scales`
/// sibling and a decoded shape other than `expected`.
pub fn stored_ggml_qtensor_vb(
    vb: &VarBuilder,
    context: &str,
    base: &str,
    expected: Option<(usize, usize)>,
    device: &Device,
) -> Result<Option<QTensor>> {
    let Some(geometry) = stored_ggml_vb(vb, base)? else {
        return Ok(None);
    };
    validate_stored(
        context,
        base,
        geometry,
        vb.contains_tensor(&format!("{base}.scales")),
        expected,
    )?;
    let stored = vb
        .clone()
        .set_device(Device::Cpu)
        .get_unchecked_dtype(&format!("{base}.weight"), DType::U8)?;
    Ok(Some(qtensor_from_stored(&stored, device)?))
}

/// The stored-block geometry of `{base}.weight` in a mapped safetensors set, read from its
/// **header** (no payload). `None` when the key is absent or is not a stored block tensor.
pub fn stored_ggml_mmap_geometry(
    st: &MmapedSafetensors,
    base: &str,
) -> Option<(GgmlDType, usize, usize)> {
    let view = st.get(&format!("{base}.weight")).ok()?;
    let dtype = DType::try_from(view.dtype()).ok()?;
    stored_block_geometry(dtype, view.shape())
}

/// `{base}.weight` of a mapped safetensors set read **on the host** when its header marks it a
/// stored GGML block tensor, after refusing a `.scales` sibling ([`validate_stored`]); `None`
/// otherwise. The mapped twin of [`stored_ggml_qtensor_vb`]'s host read: the caller builds the
/// resident [`QTensor`] with [`qtensor_from_stored`] on its destination, so the block bytes reach a
/// device once, as that final buffer, and never as a device copy read straight back.
pub fn stored_ggml_mmap_host(
    st: &MmapedSafetensors,
    context: &str,
    base: &str,
) -> Result<Option<Tensor>> {
    let Some(geometry) = stored_ggml_mmap_geometry(st, base) else {
        return Ok(None);
    };
    validate_stored(
        context,
        base,
        geometry,
        st.get(&format!("{base}.scales")).is_ok(),
        None,
    )?;
    Ok(Some(st.load(&format!("{base}.weight"), &Device::Cpu)?))
}

/// Refuse a stored GGML block at a **dense-only** `VarBuilder` loader. `vb` is already prefixed to
/// the projection (`vb.pp(base)`), so the probed key is `weight`, the convention of
/// `candle_nn::linear` and friends.
pub fn guard_dense_vb(vb: &VarBuilder) -> Result<()> {
    if let Some((dtype, ..)) = stored_ggml_vb_key(vb, "weight")? {
        return Err(StoredGgmlRefusal::DenseOnly {
            context: SHARED_CONTEXT.to_owned(),
            base: vb.prefix(),
            dtype,
        }
        .into());
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod testkit {
    //! Tiny synthetic stored-GGML fixtures. Building them repacks a synthetic MLX triple (host
    //! format conversion only — no forward, no matmul).
    use super::*;
    use crate::llm_projection::to_ggml_block_tensor;
    use crate::repack::pack_mlx_affine;

    /// The MLX affine triple of a deterministic `[out, in]` grid at group 64 and `bits`.
    pub fn triple(out: usize, inp: usize, bits: usize) -> (Tensor, Tensor, Tensor) {
        let grid = Tensor::arange(0f32, (out * inp) as f32, &Device::Cpu)
            .unwrap()
            .reshape((out, inp))
            .unwrap()
            .affine(0.0021, -0.4)
            .unwrap()
            .sin()
            .unwrap();
        pack_mlx_affine(&grid, bits, crate::repack::MLX_GROUP_SIZE).unwrap()
    }

    /// The stored block tensor a Candle-native tier ships for that triple: exactly the blocks the
    /// load-time repack builds, computed without the cache.
    pub fn stored(out: usize, inp: usize, bits: usize) -> Tensor {
        let (wq, s, b) = triple(out, inp, bits);
        let qt = crate::quant::repack_packed_weight_uncached(
            &wq,
            &s,
            &b,
            crate::repack::MLX_GROUP_SIZE,
            &Device::Cpu,
        )
        .unwrap();
        to_ggml_block_tensor(&qt).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::*;
    use super::*;
    use std::collections::HashMap;

    fn vb(map: HashMap<String, Tensor>) -> VarBuilder<'static> {
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    #[test]
    fn header_detection_classifies_only_rank3_u8_blocks() {
        assert_eq!(
            stored_block_geometry(DType::U8, &[32, 4, 20]),
            Some((GgmlDType::Q4_1, 32, 128))
        );
        assert_eq!(
            stored_block_geometry(DType::U8, &[32, 4, 34]),
            Some((GgmlDType::Q8_0, 32, 128))
        );
        // A float tensor of the same shape, a rank-2 U8 code matrix, a stacked rank-4 tensor and an
        // unknown block size are all something else.
        assert_eq!(stored_block_geometry(DType::BF16, &[32, 4, 20]), None);
        assert_eq!(stored_block_geometry(DType::U8, &[32, 80]), None);
        assert_eq!(stored_block_geometry(DType::U8, &[2, 32, 4, 20]), None);
        assert_eq!(stored_block_geometry(DType::U8, &[32, 4, 21]), None);
    }

    #[test]
    fn header_helpers_classify_stored_blocks_and_their_bits() {
        use gen_core::weightsmeta::Dtype;
        let header = |dtype, shape: &[usize]| gen_core::SafetensorsTensorHeader {
            name: "p.weight".into(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: shape.iter().product::<usize>() as u64,
        };
        assert_eq!(
            stored_header_geometry(&header(Dtype::U8, &[8, 4, 34])),
            Some((GgmlDType::Q8_0, 8, 128))
        );
        assert_eq!(
            stored_header_geometry(&header(Dtype::U32, &[8, 4, 34])),
            None
        );
        assert_eq!(
            stored_header_geometry(&header(Dtype::U8, &[2, 8, 4, 20])),
            None
        );
        assert_eq!(
            stored_stacked_header_geometry(&header(Dtype::U8, &[2, 8, 4, 20])),
            Some((GgmlDType::Q4_1, 2, 8, 128))
        );
        assert_eq!(
            stored_stacked_header_geometry(&header(Dtype::U8, &[8, 4, 20])),
            None
        );
        // The contract-layer mirror (`gen_core`'s header predicate) agrees with this module's
        // classification for every persisted block type.
        for dtype in crate::llm_projection::STORED_GGML {
            let h = header(Dtype::U8, &[8, 4, dtype.type_size()]);
            assert!(h.is_stored_ggml_block(), "{dtype:?}");
            assert_eq!(
                h.stored_ggml_block_bits(),
                stored_ggml_bits(dtype),
                "{dtype:?}"
            );
            assert!(stored_header_geometry(&h).is_some(), "{dtype:?}");
        }
        assert_eq!(stored_ggml_bits(GgmlDType::Q4_1), Some(4));
        assert_eq!(stored_ggml_bits(GgmlDType::Q8_0), Some(8));
        assert_eq!(stored_ggml_bits(GgmlDType::F16), None);
    }

    #[test]
    fn var_builder_probe_reads_the_header_dtype() {
        let mut map = HashMap::new();
        map.insert("p.weight".to_owned(), stored(8, 128, 4));
        map.insert(
            "d.weight".to_owned(),
            Tensor::zeros((8, 4, 20), DType::F32, &Device::Cpu).unwrap(),
        );
        let vb = vb(map);
        assert_eq!(
            stored_ggml_vb(&vb, "p").unwrap(),
            Some((GgmlDType::Q4_1, 8, 128))
        );
        assert_eq!(stored_ggml_vb(&vb, "d").unwrap(), None, "float, not U8");
        assert_eq!(stored_ggml_vb(&vb, "missing").unwrap(), None);
    }

    #[test]
    fn config_marker_is_recognised() {
        let ggml: serde_json::Value =
            serde_json::from_str(r#"{"quantization": {"bits": 4, "storage": "ggml"}}"#).unwrap();
        let mlx: serde_json::Value =
            serde_json::from_str(r#"{"quantization": {"bits": 4, "group_size": 64}}"#).unwrap();
        assert!(config_declares_ggml_storage(&ggml));
        assert!(!config_declares_ggml_storage(&mlx));
    }

    #[test]
    fn refusals_are_typed() {
        let both = validate_stored("t", "p", (GgmlDType::Q4_1, 8, 128), true, None).unwrap_err();
        assert!(matches!(both, StoredGgmlRefusal::BothForms { .. }));
        assert!(both.to_string().contains("never both"));
        let shape =
            validate_stored("t", "p", (GgmlDType::Q4_1, 8, 128), false, Some((8, 64))).unwrap_err();
        assert!(matches!(
            shape,
            StoredGgmlRefusal::ShapeMismatch {
                rows: 8,
                cols: 128,
                ..
            }
        ));
        validate_stored("t", "p", (GgmlDType::Q4_1, 8, 128), false, Some((8, 128))).unwrap();
    }

    #[test]
    fn dense_only_var_builder_loaders_refuse_stored_blocks() {
        let mut map = HashMap::new();
        map.insert("p.weight".to_owned(), stored(8, 128, 8));
        let vb = vb(map);
        let err = guard_dense_vb(&vb.pp("p")).unwrap_err().to_string();
        assert!(err.contains("stored Q8_0 GGML block"), "{err}");
        assert!(err.contains("dense-only"), "{err}");
        // The shared dense constructors carry the same guard.
        let err = crate::QLinear::linear_no_bias(128, 8, vb.pp("p"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("dense-only"), "{err}");
        let err = crate::AdaptLinear::linear_no_bias(128, 8, vb.pp("p"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("dense-only"), "{err}");
    }

    /// Counts every artifact a load asks the saved-quantization cache for — the same observable
    /// `candle-gen-minimax-h3`'s tier render reports. Produced artifacts land in a temp dir.
    struct CountingCache {
        dir: tempfile::TempDir,
        requests: std::sync::atomic::AtomicUsize,
    }
    struct Artifact(std::path::PathBuf);
    impl gen_core::QuantizationArtifact for Artifact {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl gen_core::QuantizationArtifactCache for CountingCache {
        fn materialize(
            &self,
            key: &gen_core::QuantizationArtifactKey,
            produce: &mut dyn FnMut(&std::path::Path) -> gen_core::Result<()>,
        ) -> gen_core::Result<Box<dyn gen_core::QuantizationArtifact>> {
            let n = self
                .requests
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let path = self.dir.path().join(format!("{n}-{}", key.digest()));
            produce(&path)?;
            Ok(Box::new(Artifact(path)))
        }
    }
    fn counting_cache() -> std::sync::Arc<CountingCache> {
        std::sync::Arc::new(CountingCache {
            dir: tempfile::tempdir().unwrap(),
            requests: std::sync::atomic::AtomicUsize::new(0),
        })
    }
    fn requests(cache: &CountingCache) -> usize {
        cache.requests.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Every shared `VarBuilder` loader builds a stored block as the packed base, and none of them
    /// asks the saved-quantization cache for anything; the MLX triple of the same weight (control)
    /// does. Construction only — no forward runs.
    #[test]
    fn shared_var_builder_loaders_build_stored_blocks_without_a_cache_write() {
        for bits in [4usize, 8] {
            let want = if bits == 4 {
                GgmlDType::Q4_1
            } else {
                GgmlDType::Q8_0
            };
            let cache = counting_cache();
            let _scope = gen_core::QuantizationCacheContext::new(cache.clone()).install();
            let mut map = HashMap::new();
            map.insert("p.weight".to_owned(), stored(8, 128, bits));
            map.insert(
                "p.bias".to_owned(),
                Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap(),
            );
            map.insert("e.weight".to_owned(), stored(16, 128, bits));
            let tier = vb(map);

            let q = crate::quant::lin_gs(&tier, "p", 128, 8, true, 64).unwrap();
            assert_eq!(q.quant_dtype(), Some(want));
            assert_eq!(
                q.matmul_strategy(),
                Some(crate::quant::MatmulStrategy::DequantDense)
            );
            let q = crate::QLinear::linear_detect(128, 8, &tier, "p", false).unwrap();
            assert_eq!(q.quant_dtype(), Some(want));
            let a = crate::AdaptLinear::linear_detect(128, 8, &tier, "p", true).unwrap();
            assert!(a.is_packed());
            assert_eq!(a.base_shape(), (8, 128));
            let a = crate::AdaptLinear::detect(&tier, "p").unwrap();
            assert!(a.is_packed());
            assert_eq!(a.base_shape(), (8, 128));
            let e = crate::quant::embedding_dtype_gs(&tier, "e", 16, 128, 64, DType::F32).unwrap();
            assert!(e.is_quantized());
            assert_eq!(
                e.resident_storage().unwrap().device_table_bytes,
                0,
                "stored rows stay host-resident"
            );
            assert_eq!(
                requests(&cache),
                0,
                "q{bits}: a stored block is never cached"
            );

            // Control: the MLX triple of the same tier goes through the cache once.
            let (wq, sc, bi) = triple(8, 128, bits);
            let mut mlx = HashMap::new();
            mlx.insert("p.weight".to_owned(), wq);
            mlx.insert("p.scales".to_owned(), sc);
            mlx.insert("p.biases".to_owned(), bi);
            let q = crate::quant::lin_gs(&vb(mlx), "p", 128, 8, false, 64).unwrap();
            assert_eq!(q.quant_dtype(), Some(want));
            assert_eq!(requests(&cache), 1, "q{bits}: the repack is cached");
        }
    }

    /// The stored bytes are exactly what the MLX tier's repack builds — so the two tiers' forwards
    /// dequantize identical blocks (the forward itself is the ignored proof in `candle-gen`).
    #[test]
    fn stored_blocks_are_byte_identical_to_the_repacked_triple() {
        for bits in [4usize, 8] {
            let mut map = HashMap::new();
            map.insert("p.weight".to_owned(), stored(8, 128, bits));
            let q = stored_ggml_qtensor_vb(&vb(map), "t", "p", Some((8, 128)), &Device::Cpu)
                .unwrap()
                .unwrap();
            let (wq, sc, bi) = triple(8, 128, bits);
            let repacked = crate::quant::repack_packed_weight_uncached(
                &wq,
                &sc,
                &bi,
                crate::repack::MLX_GROUP_SIZE,
                &Device::Cpu,
            )
            .unwrap();
            assert_eq!(q.dtype(), repacked.dtype());
            assert_eq!(q.data().unwrap(), repacked.data().unwrap());
        }
    }

    #[test]
    fn shared_loaders_refuse_malformed_stored_projections() {
        // Shape disagreement with the model's [out, in].
        let mut map = HashMap::new();
        map.insert("p.weight".to_owned(), stored(8, 128, 4));
        let err = crate::quant::lin_gs(&vb(map), "p", 128, 16, false, 64)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected (16, 128)"), "{err}");

        // Both forms present.
        let mut map = HashMap::new();
        map.insert("p.weight".to_owned(), stored(8, 128, 4));
        let (_, sc, bi) = triple(8, 128, 4);
        map.insert("p.scales".to_owned(), sc);
        map.insert("p.biases".to_owned(), bi);
        let both = vb(map);
        for err in [
            crate::quant::lin_gs(&both, "p", 128, 8, false, 64)
                .unwrap_err()
                .to_string(),
            crate::AdaptLinear::linear_detect(128, 8, &both, "p", false)
                .unwrap_err()
                .to_string(),
        ] {
            assert!(err.contains("never both"), "{err}");
        }

        // Embedding shape mismatch.
        let mut map = HashMap::new();
        map.insert("e.weight".to_owned(), stored(16, 128, 8));
        let err = crate::quant::embedding(&vb(map), "e", 32, 128)
            .err()
            .expect("vocab mismatch")
            .to_string();
        assert!(err.contains("expected (32, 128)"), "{err}");
    }

    /// End to end through a real safetensors header: the mmap backend reports the stored dtype, so
    /// the probe classifies the block without reading its payload, and the load builds it as stored.
    #[test]
    fn mmapped_tier_is_classified_from_its_header() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.safetensors");
        let mut map = HashMap::new();
        map.insert("blocks.0.q.weight".to_owned(), stored(8, 128, 4));
        map.insert(
            "blocks.0.norm.weight".to_owned(),
            Tensor::ones(128, DType::BF16, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&map, &file).unwrap();
        // SAFETY: the file is private to this test and unmodified while mapped.
        let mapped = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&file], DType::BF16, &Device::Cpu).unwrap()
        };
        let cache = counting_cache();
        let _scope = gen_core::QuantizationCacheContext::new(cache.clone()).install();
        assert_eq!(
            stored_ggml_vb(&mapped, "blocks.0.q").unwrap(),
            Some((GgmlDType::Q4_1, 8, 128))
        );
        let q = crate::QLinear::linear_detect(128, 8, &mapped, "blocks.0.q", false).unwrap();
        assert_eq!(q.quant_dtype(), Some(GgmlDType::Q4_1));
        assert_eq!(stored_ggml_vb(&mapped, "blocks.0.norm").unwrap(), None);
        assert_eq!(requests(&cache), 0);

        // The mapped-set helper the `MappedWeights`-fed family loaders share: classified from the
        // header, read on the host, and built once on the destination.
        // SAFETY: as above.
        let st = unsafe { MmapedSafetensors::new(&file).unwrap() };
        assert_eq!(
            stored_ggml_mmap_geometry(&st, "blocks.0.q"),
            Some((GgmlDType::Q4_1, 8, 128))
        );
        assert_eq!(stored_ggml_mmap_geometry(&st, "blocks.0.norm"), None);
        let host = stored_ggml_mmap_host(&st, "t", "blocks.0.q")
            .unwrap()
            .expect("a stored block tensor");
        assert!(host.device().is_cpu());
        assert_eq!(host.dims(), [8, 4, 20]);
        let built = qtensor_from_stored(&host, &Device::Cpu).unwrap();
        let want = qtensor_from_stored(&map["blocks.0.q.weight"], &Device::Cpu).unwrap();
        assert_eq!(built.data().unwrap(), want.data().unwrap());
        assert!(stored_ggml_mmap_host(&st, "t", "blocks.0.norm")
            .unwrap()
            .is_none());
    }
}
