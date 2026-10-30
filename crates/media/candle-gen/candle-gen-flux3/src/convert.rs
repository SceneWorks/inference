//! Native Candle GGML conversion of the ORIGINAL dense FLUX3 DiT (sc-26241).
//!
//! The source rules, streamed deterministic layout, atomic publication, full-checksum receipt and
//! EXTERNAL encoder/autoencoder placement are the backend-neutral
//! [`gen_core_flux3::native_convert`]. The one Candle-specific step is the pack: each source FP8
//! target projection is quantized from its stored dense values by Candle's own GGML quantizer, so
//! the bytes are exactly what [`crate::nn::Loader`] reads back into a `QTensor`.
use std::path::Path;

use candle_gen::candle_core::quantized::{GgmlDType, QTensor};
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{self, CancelFlag};
use gen_core_flux3::artifacts::NativeDitStorage;
use gen_core_flux3::conversion::ConversionReceipt;
pub use gen_core_flux3::native_convert::ConversionSource;
use gen_core_flux3::native_convert::{convert_native, Packed};

type CoreResult<T> = gen_core::Result<T>;

fn invalid(detail: impl std::fmt::Display) -> gen_core::Error {
    gen_core::Error::Unsupported(format!("flux3 ggml conversion: {detail}"))
}

/// The GGML block type a native storage selects; anything else has no Candle GGML converter.
pub fn ggml_dtype(storage: NativeDitStorage) -> CoreResult<GgmlDType> {
    match storage {
        NativeDitStorage::GgmlQ4_1 => Ok(GgmlDType::Q4_1),
        NativeDitStorage::GgmlQ8_0 => Ok(GgmlDType::Q8_0),
        other => Err(invalid(format!(
            "{} is not a Candle GGML storage; convert with that backend's own converter",
            other.key()?
        ))),
    }
}

/// Quantize one dense `[n, k]` projection to its GGML row bytes (`[n, k/32 · block]`).
pub fn quantize_rows(
    raw: &[u8],
    dtype: &str,
    shape: &[usize],
    ggml: GgmlDType,
) -> CoreResult<Vec<u8>> {
    let dtype = match dtype {
        "BF16" => DType::BF16,
        "F32" => DType::F32,
        other => {
            return Err(invalid(format!(
                "target stored as {other}; only original BF16/F32 dense weights are converted"
            )))
        }
    };
    let tensor = Tensor::from_raw_buffer(raw, dtype, shape, &Device::Cpu)
        .map_err(|e| invalid(format!("decode source tensor: {e}")))?;
    let q = QTensor::quantize(&tensor, ggml).map_err(|e| invalid(format!("quantize: {e}")))?;
    Ok(q.data()
        .map_err(|e| invalid(format!("packed bytes: {e}")))?
        .into_owned())
}

/// Convert `source` into a NEW owned GGML package at `output` (which must not exist).
pub fn convert_ggml(
    source: &ConversionSource,
    storage: NativeDitStorage,
    output: &Path,
    cancel: &CancelFlag,
) -> CoreResult<ConversionReceipt> {
    let ggml = ggml_dtype(storage)?;
    convert_native(source, storage, output, cancel, |target, tensor, raw| {
        if let [_, k] = tensor.shape.as_slice() {
            storage.check_contraction_width(target, *k)?;
        }
        Ok(Packed::from([(
            target.to_owned(),
            quantize_rows(raw, &tensor.dtype, &tensor.shape, ggml)?,
        )]))
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
    use gen_core_flux3::conversion::ComponentRole;
    use gen_core_flux3::DIT_WEIGHTS_FILE;
    use std::path::PathBuf;

    fn fixture_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-flux3/tests/fixtures/tiny-snapshot")
    }

    /// A private copy of the tiny dense package plus the license files a real package ships.
    pub(crate) fn source_copy(dir: &Path) -> ConversionSource {
        let root = dir.join("source");
        std::fs::create_dir_all(root.join("text_encoder")).unwrap();
        for entry in ["config.json", "model.safetensors", "ae.safetensors"] {
            std::fs::copy(fixture_root().join(entry), root.join(entry)).unwrap();
        }
        for entry in std::fs::read_dir(fixture_root().join("text_encoder")).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(
                entry.path(),
                root.join("text_encoder").join(entry.file_name()),
            )
            .unwrap();
        }
        std::fs::write(root.join("LICENSE.md"), b"license").unwrap();
        std::fs::write(root.join("NOTICE.md"), b"notice").unwrap();
        ConversionSource {
            root,
            repository: gen_core_flux3::DISTILLED_WEIGHTS_REPO.into(),
            revision: gen_core_flux3::DISTILLED_WEIGHTS_REVISION.into(),
        }
    }

    /// The written GGML rows are exactly Candle's own quantizer output, and the package is
    /// deterministic with its unchanged encoder/AE external.
    ///
    /// *Mutations that red this:* writing a different block type than the marker, quantizing a
    /// rounded copy, or copying the encoder/AE into the package.
    #[test]
    fn ggml_conversion_writes_candles_exact_blocks_deterministically() {
        let dir = tempfile::tempdir().unwrap();
        let source = source_copy(dir.path());
        let src = gen_core_flux3::safetensors_stream::SafetensorsSource::open(
            &source.root.join(DIT_WEIGHTS_FILE),
        )
        .unwrap();
        for storage in [NativeDitStorage::GgmlQ4_1, NativeDitStorage::GgmlQ8_0] {
            let key = storage.key().unwrap();
            let one = dir.path().join(format!("{key}-a"));
            let two = dir.path().join(format!("{key}-b"));
            let r1 = convert_ggml(&source, storage, &one, &CancelFlag::new()).unwrap();
            let r2 = convert_ggml(&source, storage, &two, &CancelFlag::new()).unwrap();
            assert_eq!(r1, r2);
            assert_eq!(
                std::fs::read(one.join(DIT_WEIGHTS_FILE)).unwrap(),
                std::fs::read(two.join(DIT_WEIGHTS_FILE)).unwrap()
            );
            assert_eq!(
                gen_core_flux3::conversion::open_conversion(&one).unwrap(),
                r1
            );
            r1.verify_external(&source.root).unwrap();
            assert!(!one.join("ae.safetensors").exists());
            assert_eq!(NativeDitStorage::from_package(&one).unwrap(), storage);
            let cfg = gen_core_flux3::resolve_checkpoint(&one).unwrap();
            storage
                .validate_headers(
                    &cfg.dit,
                    &safetensors_path_tensor_headers(one.join(DIT_WEIGHTS_FILE)).unwrap(),
                )
                .unwrap();
            let dit = r1
                .components
                .iter()
                .find(|c| c.role == ComponentRole::Dit)
                .unwrap();
            assert_eq!(
                dit.quantized_targets.len(),
                storage.packed_targets(&cfg.dit).unwrap().len()
            );
            let out = gen_core_flux3::safetensors_stream::SafetensorsSource::open(
                &one.join(DIT_WEIGHTS_FILE),
            )
            .unwrap();
            let name = "single_blocks.0.mlp_out.weight";
            let tensor = src.get(name).unwrap();
            let want = quantize_rows(
                &src.read(name).unwrap(),
                &tensor.dtype,
                &tensor.shape,
                ggml_dtype(storage).unwrap(),
            )
            .unwrap();
            assert_eq!(out.read(name).unwrap(), want);
        }
    }

    #[test]
    fn non_candle_storages_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let source = source_copy(dir.path());
        assert!(convert_ggml(
            &source,
            NativeDitStorage::MlxAffine {
                bits: 4,
                group_size: 64
            },
            &dir.path().join("mlx"),
            &CancelFlag::new()
        )
        .is_err());
    }
}
