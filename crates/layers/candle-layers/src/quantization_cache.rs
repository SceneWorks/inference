//! Reuse the packed result of the shared GGML weight fold across loads and process restarts.
use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use std::borrow::Cow;

/// Install the common backend bridge after the request's backend-neutral cache context.
/// Core quantizer calls then reuse saved weights too, without changing individual model callers.
pub fn install_runtime_cache_context() -> impl Drop {
    let cache = gen_core::QuantizationCacheContext::capture()
        .cache()
        .cloned()
        .map(|cache| {
            std::sync::Arc::new(RuntimeCache(cache))
                as std::sync::Arc<
                    dyn candle_core::quantized::runtime_cache::RuntimeQuantizationCache,
                >
        });
    candle_core::quantized::runtime_cache::install(cache)
}
struct RuntimeCache(std::sync::Arc<dyn gen_core::QuantizationArtifactCache>);
impl candle_core::quantized::runtime_cache::RuntimeQuantizationCache for RuntimeCache {
    fn quantize(
        &self,
        source: &Tensor,
        dtype: GgmlDType,
        device: &Device,
        mode: &'static str,
        imatrix: Option<&[f32]>,
        produce: &mut dyn FnMut() -> Result<QTensor>,
    ) -> Result<QTensor> {
        quantize_with_cache(
            self.0.as_ref(),
            source,
            dtype,
            device,
            mode,
            imatrix,
            produce,
        )
    }
    fn repack_affine(
        &self,
        codes: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        group_size: usize,
        device: &Device,
        produce: &mut dyn FnMut() -> Result<QTensor>,
    ) -> Result<QTensor> {
        repack_affine_with_cache(
            self.0.as_ref(),
            codes,
            scales,
            biases,
            group_size,
            device,
            produce,
        )
    }
}

pub fn quantize_cached(src: &Tensor, dtype: GgmlDType, device: &Device) -> Result<QTensor> {
    let context = gen_core::QuantizationCacheContext::capture();
    let Some(cache) = context.cache() else {
        return QTensor::quantize_onto(src, dtype, device);
    };
    quantize_with_cache(
        cache.as_ref(),
        src,
        dtype,
        device,
        "cpu-upload",
        None,
        || QTensor::quantize_onto(&cpu_quantization_input(src)?, dtype, device),
    )
}

fn cpu_quantization_input(src: &Tensor) -> Result<Tensor> {
    let cpu = src.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let covers_storage = {
        let (storage, layout) = cpu.storage_and_layout();
        let length = match &*storage {
            candle_core::Storage::Cpu(storage) => storage.as_slice::<f32>()?.len(),
            _ => 0,
        };
        layout.start_offset() == 0 && cpu.is_contiguous() && length == cpu.elem_count()
    };
    if covers_storage {
        Ok(cpu)
    } else {
        cpu.force_contiguous()
    }
}

fn quantize_with_cache(
    cache: &dyn gen_core::QuantizationArtifactCache,
    src: &Tensor,
    dtype: GgmlDType,
    device: &Device,
    mode: &str,
    imatrix: Option<&[f32]>,
    produce: impl FnOnce() -> Result<QTensor>,
) -> Result<QTensor> {
    let origin = src.checkpoint_source_identity();
    let source = if origin.is_some() {
        "checkpoint"
    } else {
        "content"
    };
    let recipe = format!(
        "candle-{source}-ggml-{dtype:?}-{mode}-imatrix{}-v2",
        imatrix.is_some()
    );
    let identify = |input: &[u8]| {
        gen_core::QuantizationArtifactKey::from_slices(
            &recipe,
            src.dims(),
            [input, imatrix.map(bytemuck::cast_slice).unwrap_or(&[])],
            "ggml",
        )
    };
    let key = if let Some(source) = &origin {
        identify(source)
    } else {
        // Contiguous CPU weights are hashed in their existing storage. Other layouts/devices
        // require only the normal CPU/F32 staging, never a second copy just to form the key.
        let cpu = src
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?;
        let (storage, layout) = cpu.storage_and_layout();
        let candle_core::Storage::Cpu(storage) = &*storage else {
            candle_core::bail!("quantization identity requires CPU storage");
        };
        let values = storage.as_slice::<f32>()?;
        let start = layout.start_offset();
        identify(bytemuck::cast_slice(
            &values[start..start + cpu.elem_count()],
        ))
    };
    materialize_runtime_weight(cache, &key, src.dims(), dtype, device, produce)
}

/// Source-device callers use the same early cache lookup, before any CPU/F32 staging.
pub fn quantize_on_source_device_cached(src: &Tensor, dtype: GgmlDType) -> Result<QTensor> {
    let context = gen_core::QuantizationCacheContext::capture();
    let Some(cache) = context.cache() else {
        return QTensor::quantize(src, dtype);
    };
    quantize_with_cache(
        cache.as_ref(),
        src,
        dtype,
        src.device(),
        "cpu-upload",
        None,
        || QTensor::quantize_onto(&cpu_quantization_input(src)?, dtype, src.device()),
    )
}

/// Shared affine repack cache. Identity lookup precedes dequantization, float expansion and
/// weight hashing for every caller, including models that pass raw checkpoint tensors.
/// Per-projection widths come from the operands; mixed Q4/Q8 components need no special route.
pub(crate) fn repack_affine_cached(
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
    device: &Device,
    produce: impl FnOnce() -> Result<QTensor>,
) -> Result<QTensor> {
    let context = gen_core::QuantizationCacheContext::capture();
    if let Some(cache) = context.cache() {
        return repack_affine_with_cache(
            cache.as_ref(),
            wq,
            scales,
            biases,
            group_size,
            device,
            produce,
        );
    }
    if let Some(cache) = candle_core::quantized::runtime_cache::for_tensor(wq) {
        let mut produce = Some(produce);
        return candle_core::quantized::runtime_cache::without_cache(|| {
            cache.repack_affine(wq, scales, biases, group_size, device, &mut || {
                produce.take().ok_or_else(|| {
                    candle_core::Error::Msg("affine producer invoked twice".into())
                })?()
            })
        });
    }
    produce()
}

fn repack_affine_with_cache(
    cache: &dyn gen_core::QuantizationArtifactCache,
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
    device: &Device,
    produce: impl FnOnce() -> Result<QTensor>,
) -> Result<QTensor> {
    let mut identity = Vec::new();
    let (recipe, parts) = match [wq, scales, biases].map(Tensor::checkpoint_source_identity) {
        [Some(codes), Some(scales_source), Some(biases_source)] => {
            ("checkpoint", vec![codes, scales_source, biases_source])
        }
        _ => {
            // Mutable/in-memory inputs have no immutable source name. Identify their compact
            // operands, never an expanded float weight, so adapters cannot reuse stale codes.
            let codes = wq
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            let scale = scales
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let bias = biases
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            (
                "content",
                vec![
                    bytemuck::cast_slice(&codes).to_vec(),
                    bytemuck::cast_slice(&scale).to_vec(),
                    bytemuck::cast_slice(&bias).to_vec(),
                ],
            )
        }
    };
    for part in parts {
        identity.extend_from_slice(&(part.len() as u64).to_le_bytes());
        identity.extend_from_slice(&part);
    }
    let (rows, groups) = scales.dims2()?;
    let cols = groups
        .checked_mul(group_size)
        .ok_or_else(|| candle_core::Error::Msg("packed dimension overflow".into()))?;
    if cols == 0 {
        candle_core::bail!("affine projection has an empty contraction");
    }
    let words = wq.dims2()?.1;
    let bits = words
        .checked_mul(32)
        .and_then(|n| n.checked_div(cols))
        .filter(|bits| matches!(bits, 4 | 8) && words.checked_mul(32) == cols.checked_mul(*bits))
        .ok_or_else(|| candle_core::Error::Msg("unsupported affine projection width".into()))?;
    let dtype = if bits == 4 {
        GgmlDType::Q4_1
    } else {
        GgmlDType::Q8_0
    };
    let shape = [rows, cols];
    let key = gen_core::QuantizationArtifactKey::new(
        &format!("candle-{recipe}-affine-g{group_size}-b{bits}-v1"),
        &shape,
        &identity,
        "ggml",
    );
    materialize_runtime_weight(cache, &key, &shape, dtype, device, produce)
}

fn materialize_runtime_weight(
    cache: &dyn gen_core::QuantizationArtifactCache,
    key: &gen_core::QuantizationArtifactKey,
    shape: &[usize],
    dtype: GgmlDType,
    device: &Device,
    produce: impl FnOnce() -> Result<QTensor>,
) -> Result<QTensor> {
    let mut produce = Some(produce);
    for attempt in 0..2 {
        let mut produced = None;
        let artifact = cache
            .materialize(key, &mut |path| {
                let tensor = produce.take().ok_or_else(|| {
                    gen_core::Error::Msg("runtime weight producer invoked twice".into())
                })?;
                let tensor = candle_core::quantized::runtime_cache::without_cache(tensor)
                    .map_err(gen_core::Error::backend)?;
                std::fs::write(
                    path,
                    tensor.data().map_err(gen_core::Error::backend)?.as_ref(),
                )?;
                produced = Some(tensor);
                Ok(())
            })
            .map_err(|e| candle_core::Error::Msg(format!("saved runtime weight: {e}")))?;
        if let Some(tensor) = produced {
            return Ok(tensor);
        }
        // Completed artifacts are trusted. Read runtime bytes directly; no source/payload
        // digest pass belongs here. The lease protects the file until its upload completes.
        let read = std::fs::read(artifact.path())
            .map_err(candle_core::Error::from)
            .and_then(|bytes| {
                let elements = shape
                    .iter()
                    .try_fold(1usize, |n, &d| n.checked_mul(d))
                    .ok_or_else(|| {
                        candle_core::Error::Msg("runtime weight shape overflow".into())
                    })?;
                let expected = (elements / dtype.block_size())
                    .checked_mul(dtype.type_size())
                    .ok_or_else(|| {
                        candle_core::Error::Msg("runtime weight size overflow".into())
                    })?;
                if !elements.is_multiple_of(dtype.block_size()) || bytes.len() != expected {
                    candle_core::bail!(
                        "runtime weight cannot be decoded: expected {expected} bytes, got {}",
                        bytes.len()
                    );
                }
                QTensor::new(
                    QStorage::from_data(Cow::Borrowed(&bytes), device, dtype)?,
                    shape,
                )
            });
        drop(artifact);
        match read {
            Ok(tensor) => return Ok(tensor),
            Err(error) => {
                if attempt != 0
                    || !cache
                        .invalidate(key)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                {
                    return Err(error);
                }
            }
        }
    }
    unreachable!("cache retry is bounded")
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{QuantizationArtifact, QuantizationArtifactCache, QuantizationArtifactKey};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct Artifact(PathBuf);
    impl QuantizationArtifact for Artifact {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    struct Cache {
        root: tempfile::TempDir,
        writes: AtomicUsize,
        hits: AtomicUsize,
    }
    impl Cache {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                root: tempfile::tempdir().unwrap(),
                writes: AtomicUsize::new(0),
                hits: AtomicUsize::new(0),
            })
        }
    }
    impl QuantizationArtifactCache for Cache {
        fn materialize(
            &self,
            key: &QuantizationArtifactKey,
            produce: &mut dyn FnMut(&Path) -> gen_core::Result<()>,
        ) -> gen_core::Result<Box<dyn QuantizationArtifact>> {
            let path = self.root.path().join(key.digest());
            if path.is_file() {
                self.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                produce(&path)?;
                self.writes.fetch_add(1, Ordering::Relaxed);
            }
            Ok(Box::new(Artifact(path)))
        }
        fn invalidate(&self, key: &QuantizationArtifactKey) -> gen_core::Result<bool> {
            std::fs::remove_file(self.root.path().join(key.digest()))?;
            Ok(true)
        }
    }
    fn fixture(path: &Path, bits: usize, group: usize, delta: f32) -> Result<()> {
        let grid = Tensor::arange(0f32, 256f32, &Device::Cpu)?
            .reshape((2, 128))?
            .affine(0.003, delta as f64)?;
        let (codes, scales, biases) = crate::repack::pack_mlx_affine(&grid, bits, group)?;
        candle_core::safetensors::save(
            &HashMap::from([
                ("p.weight".to_owned(), codes),
                ("p.scales".to_owned(), scales),
                ("p.biases".to_owned(), biases),
                ("dense.weight".to_owned(), grid.to_dtype(DType::BF16)?),
            ]),
            path,
        )?;
        Ok(())
    }
    fn packed(parts: &HashMap<String, Tensor>, group: usize) -> Result<QTensor> {
        crate::quant::repack_packed_weight(
            &parts["p.weight"],
            &parts["p.scales"],
            &parts["p.biases"],
            group,
            &Device::Cpu,
        )
    }
    fn data(weight: &QTensor) -> Vec<u8> {
        weight.data().unwrap().into_owned()
    }

    #[test]
    fn raw_checkpoint_callers_reuse_runtime_bytes_for_q4_q8_and_group32_group64() -> Result<()> {
        for bits in [4, 8] {
            for group in [32, 64] {
                let source = tempfile::tempdir()?;
                let file = source.path().join("model.safetensors");
                fixture(&file, bits, group, 0.)?;
                let cache = Cache::new();
                let _scope = gen_core::QuantizationCacheContext::new(cache.clone()).install();
                let first = candle_core::safetensors::load(&file, &Device::Cpu)?;
                let expected = data(&packed(&first, group)?);
                drop(first);
                // The cache is trusted, including after a process reopens the source. A source-byte
                // digest would miss this hit; reading/expanding a float grid would produce new bytes.
                fixture(&file, bits, group, 3.)?;
                let reopened = candle_core::safetensors::load(&file, &Device::Cpu)?;
                assert_eq!(data(&packed(&reopened, group)?), expected);
                assert_eq!(cache.writes.load(Ordering::Relaxed), 1);
                assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
            }
        }
        Ok(())
    }

    #[test]
    fn dense_runtime_quantization_hits_before_float_expansion_and_source_hashing() -> Result<()> {
        let source = tempfile::tempdir()?;
        let file = source.path().join("model.safetensors");
        fixture(&file, 8, 64, 0.)?;
        let cache = Cache::new();
        let _scope = gen_core::QuantizationCacheContext::new(cache.clone()).install();
        let first = candle_core::safetensors::load(&file, &Device::Cpu)?;
        let expected = data(&quantize_cached(
            &first["dense.weight"],
            GgmlDType::Q8_0,
            &Device::Cpu,
        )?);
        fixture(&file, 8, 64, 7.)?;
        let reopened = candle_core::safetensors::load(&file, &Device::Cpu)?;
        assert_eq!(
            data(&quantize_cached(
                &reopened["dense.weight"],
                GgmlDType::Q8_0,
                &Device::Cpu
            )?),
            expected
        );
        assert_eq!(cache.writes.load(Ordering::Relaxed), 1);
        assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
        // Value-changing adapter/training work must not inherit the immutable source identity.
        let adapted = (&reopened["dense.weight"] + 1.)?;
        assert!(adapted.checkpoint_source_identity().is_none());
        assert_ne!(
            data(&quantize_cached(&adapted, GgmlDType::Q8_0, &Device::Cpu)?),
            expected
        );
        Ok(())
    }

    #[test]
    fn mutable_contiguous_subviews_hash_only_their_logical_storage_range() -> Result<()> {
        let source = Tensor::arange(0f32, 256f32, &Device::Cpu)?.reshape((4, 64))?;
        let first = source.narrow(0, 1, 1)?;
        let second = source.narrow(0, 2, 1)?;
        let expected = data(&QTensor::quantize_onto(
            &first.force_contiguous()?,
            GgmlDType::Q8_0,
            &Device::Cpu,
        )?);
        let cache = Cache::new();
        let _scope = gen_core::QuantizationCacheContext::new(cache.clone()).install();
        assert_eq!(
            data(&quantize_cached(&first, GgmlDType::Q8_0, &Device::Cpu)?),
            expected
        );
        assert_ne!(
            data(&quantize_cached(&second, GgmlDType::Q8_0, &Device::Cpu)?),
            expected
        );
        assert_eq!(
            data(&quantize_cached(
                &first.force_contiguous()?,
                GgmlDType::Q8_0,
                &Device::Cpu
            )?),
            expected
        );
        assert_eq!(cache.writes.load(Ordering::Relaxed), 2);
        assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn views_casts_and_mutations_have_correct_shared_source_identities() -> Result<()> {
        let source = tempfile::tempdir()?;
        let file = source.path().join("model.safetensors");
        fixture(&file, 8, 64, 0.)?;
        let parts = candle_core::safetensors::load(&file, &Device::Cpu)?;
        let matrix = &parts["dense.weight"];
        let view = matrix.narrow(0, 0, 1)?;
        assert_ne!(
            view.checkpoint_source_identity(),
            matrix.narrow(0, 1, 1)?.checkpoint_source_identity()
        );
        assert_ne!(
            matrix.checkpoint_source_identity(),
            matrix.to_dtype(DType::F32)?.checkpoint_source_identity()
        );
        let reloaded = candle_core::safetensors::load(&file, &Device::Cpu)?;
        assert_eq!(
            matrix.to_dtype(DType::F32)?.checkpoint_source_identity(),
            reloaded["dense.weight"]
                .to_dtype(DType::F32)?
                .checkpoint_source_identity()
        );
        matrix.zero_set()?;
        assert!(matrix.checkpoint_source_identity().is_none());
        assert!(view.checkpoint_source_identity().is_none());
        Ok(())
    }

    #[test]
    fn direct_core_quantizers_inherit_the_request_cache_without_model_opt_ins() -> Result<()> {
        let source = tempfile::tempdir()?;
        let file = source.path().join("model.safetensors");
        fixture(&file, 8, 64, 0.)?;
        let cache = Cache::new();
        let _scope = gen_core::QuantizationCacheContext::new(cache.clone()).install();
        let _backend = install_runtime_cache_context();
        let first = candle_core::safetensors::load(&file, &Device::Cpu)?;
        let ordinary = data(&QTensor::quantize(&first["dense.weight"], GgmlDType::Q8_0)?);
        let uploaded = data(&QTensor::quantize_onto(
            &first["dense.weight"],
            GgmlDType::Q8_0,
            &Device::Cpu,
        )?);
        let importance = vec![1f32; 256];
        let matrix = data(&QTensor::quantize_imatrix(
            &first["dense.weight"].reshape((1, 256))?,
            &importance,
            GgmlDType::Q4K,
        )?);
        let matrix_upload = data(&QTensor::quantize_imatrix_onto(
            &first["dense.weight"].reshape((1, 256))?,
            &importance,
            GgmlDType::Q4K,
            &Device::Cpu,
        )?);
        fixture(&file, 8, 64, 9.)?;
        let second = candle_core::safetensors::load(&file, &Device::Cpu)?;
        assert_eq!(
            data(&QTensor::quantize(
                &second["dense.weight"],
                GgmlDType::Q8_0
            )?),
            ordinary
        );
        assert_eq!(
            data(&QTensor::quantize_onto(
                &second["dense.weight"],
                GgmlDType::Q8_0,
                &Device::Cpu
            )?),
            uploaded
        );
        assert_eq!(
            data(&QTensor::quantize_imatrix(
                &second["dense.weight"].reshape((1, 256))?,
                &importance,
                GgmlDType::Q4K
            )?),
            matrix
        );
        assert_eq!(
            data(&QTensor::quantize_imatrix_onto(
                &second["dense.weight"].reshape((1, 256))?,
                &importance,
                GgmlDType::Q4K,
                &Device::Cpu
            )?),
            matrix_upload
        );
        assert_eq!(cache.writes.load(Ordering::Relaxed), 4);
        assert_eq!(cache.hits.load(Ordering::Relaxed), 4);
        Ok(())
    }

    #[test]
    fn mapped_and_owned_sources_keep_the_request_cache_across_loader_threads() -> Result<()> {
        let source = tempfile::tempdir()?;
        let file = source.path().join("model.safetensors");
        fixture(&file, 8, 64, 0.)?;
        let cache = Cache::new();
        let (mapped, owned) = {
            let _scope = gen_core::QuantizationCacheContext::new(cache.clone()).install();
            let _backend = install_runtime_cache_context();
            let mapped = unsafe { candle_core::safetensors::MmapedSafetensors::new(&file)? };
            let mut owned = candle_core::safetensors::MmapedSafetensors::from_buffers(vec![Ok(
                std::fs::read(&file)?,
            )])?;
            owned.bind_source_paths(std::slice::from_ref(&file));
            (mapped, owned)
        };
        let run = |source: candle_core::safetensors::MmapedSafetensors| {
            std::thread::spawn(move || -> Result<Vec<u8>> {
                let weight = source.load("dense.weight", &Device::Cpu)?;
                Ok(data(&QTensor::quantize_onto(
                    &weight,
                    GgmlDType::Q8_0,
                    &Device::Cpu,
                )?))
            })
            .join()
            .unwrap()
        };
        let expected = run(mapped)?;
        assert_eq!(run(owned)?, expected);
        assert_eq!(cache.writes.load(Ordering::Relaxed), 1);
        assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn unreadable_runtime_artifact_rebuilds_once_without_a_validation_pass() -> Result<()> {
        let source = tempfile::tempdir()?;
        let file = source.path().join("model.safetensors");
        fixture(&file, 8, 64, 0.)?;
        let cache = Cache::new();
        let _scope = gen_core::QuantizationCacheContext::new(cache.clone()).install();
        let parts = candle_core::safetensors::load(&file, &Device::Cpu)?;
        let expected = data(&packed(&parts, 64)?);
        let entry = std::fs::read_dir(cache.root.path())?
            .next()
            .unwrap()?
            .path();
        std::fs::write(entry, b"bad")?;
        assert_eq!(data(&packed(&parts, 64)?), expected);
        assert_eq!(cache.writes.load(Ordering::Relaxed), 2);
        Ok(())
    }
}
