//! Reuse the packed result of the shared GGML weight fold across loads and process restarts.
use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use std::borrow::Cow;

pub fn quantize_cached(src: &Tensor, dtype: GgmlDType, device: &Device) -> Result<QTensor> {
    let context = gen_core::QuantizationCacheContext::capture();
    let Some(cache) = context.cache() else {
        return QTensor::quantize_onto(src, dtype, device);
    };
    let cpu = src.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let owned = {
        let covers_storage = {
            let (storage, layout) = cpu.storage_and_layout();
            let length = match &*storage {
                candle_core::Storage::Cpu(storage) => storage.as_slice::<f32>()?.len(),
                _ => 0,
            };
            layout.start_offset() == 0 && cpu.is_contiguous() && length == cpu.elem_count()
        };
        if covers_storage {
            cpu
        } else {
            cpu.force_contiguous()?
        }
    };
    let key = {
        let (storage, _) = owned.storage_and_layout();
        let candle_core::Storage::Cpu(storage) = &*storage else {
            candle_core::bail!("quantization identity requires CPU storage");
        };
        gen_core::QuantizationArtifactKey::new(
            &format!("candle-ggml-{dtype:?}-f32-v1"),
            src.dims(),
            bytemuck::cast_slice(storage.as_slice::<f32>()?),
            "ggml",
        )
    };
    let mut produced = None;
    let artifact = cache
        .materialize(&key, &mut |path| {
            let tensor =
                QTensor::quantize_onto(&owned, dtype, device).map_err(gen_core::Error::backend)?;
            std::fs::write(
                path,
                tensor.data().map_err(gen_core::Error::backend)?.as_ref(),
            )?;
            produced = Some(tensor);
            Ok(())
        })
        .map_err(|error| candle_core::Error::Msg(format!("saved weight quantization: {error}")))?;
    if let Some(tensor) = produced {
        return Ok(tensor);
    }
    let bytes = std::fs::read(artifact.path())?;
    // Candle copies/uploads the borrowed payload during this call; retain its backing buffer.
    let storage = QStorage::from_data(Cow::Borrowed(&bytes), device, dtype)?;
    QTensor::new(storage, src.shape())
}

/// Cache the source-device variant used by embeddings and legacy projection loaders.
pub fn quantize_on_source_device_cached(src: &Tensor, dtype: GgmlDType) -> Result<QTensor> {
    if gen_core::QuantizationCacheContext::capture()
        .cache()
        .is_none()
    {
        return QTensor::quantize(src, dtype);
    }
    let cpu = src.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    quantize_cached(&cpu, dtype, src.device())
}
