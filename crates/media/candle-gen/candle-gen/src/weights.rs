//! A small safetensors key→`Tensor` map for non-`VarBuilder` weight loads — the candle twin of
//! `mlx_gen::weights::Weights` (sc-5491).
//!
//! Most pipelines build the stock UNet/VAE/DiT through a `VarBuilder`, but several loads want a raw
//! key→`Tensor` map (cast to the compute dtype on load) instead: the SDXL IP-Adapter Resampler mixes
//! a learned-`latents` tensor with fused-projection Linears, the ControlNet adds per-residual
//! zero-convs, and the FLUX-family IP-Adapter / PuLID EVA-CLIP towers load the same way. Because that
//! surface is shared across unrelated provider crates (SDXL, FLUX, PuLID), it lives here in the
//! `candle-gen` core crate rather than in any one pipeline crate (F-060, sc-9044) — `candle-gen-sdxl`
//! re-exports it from its old `weights` path for source compatibility.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{safetensors as cst, DType, Tensor};

use crate::candle_core::Device;
use crate::{CandleError, Result};

/// Coerce a loaded tensor to the compute `dtype`, but only when it is a FLOATING tensor.
///
/// Integer buffers — e.g. the CLIP image encoder's I64 `position_ids` (`h94/IP-Adapter`
/// `models/image_encoder`) — are left as-is: casting an int index buffer to f16 is meaningless, and
/// on CUDA (sm_120) the int→f16 cast kernel isn't compiled, so `to_dtype` there fails with
/// `DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")` (sc-5488). The consumers here only
/// `require()` the float weights, so an untouched buffer is simply never read.
fn coerce_float(v: Tensor, dtype: DType) -> Result<Tensor> {
    let is_float = matches!(
        v.dtype(),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64
    );
    if is_float && v.dtype() != dtype {
        Ok(v.to_dtype(dtype)?)
    } else {
        Ok(v)
    }
}

/// A loaded checkpoint weight map (every tensor coerced to the requested compute dtype on load).
pub struct Weights {
    map: HashMap<String, Tensor>,
    int8: std::sync::Mutex<Option<crate::quant::Int8Context>>,
    convrot_rotations: std::sync::Mutex<HashMap<usize, Tensor>>,
    /// The device the set is *destined for* when some tensors were deliberately left on the host
    /// (see [`Weights::from_files_filtered_host_packed`]). `None` for every ordinary constructor.
    target: Option<Device>,
}

/// Decode a little-endian F32 plane into its `f32` values, for value-validating native scale and
/// codebook operands directly off the mmap without materializing a `Tensor` copy.
fn f32_le_planes(bytes: &[u8]) -> Vec<f32> {
    cst::read_source_bytes(bytes, |bytes| {
        bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect()
    })
}
impl Weights {
    /// The device this weight set is resident on — every constructor loads all tensors onto one
    /// `device`, so any tensor's device answers for the set. `None` only for an empty map.
    ///
    /// Added for sc-12274: a loader that wants **one** shared per-device resource for the whole trunk
    /// (e.g. `candle_gen::quant::Nvfp4Context`, one cuBLASLt handle instead of one per projection)
    /// needs the device up front, before it starts fetching individual weights.
    pub fn device(&self) -> Option<&Device> {
        self.map.values().next().map(|t| t.device())
    }

    /// Load every tensor from a `.safetensors` file onto `device`, casting to `dtype` (f16 in
    /// production, f32 for CPU parity), matching how `mlx-gen-sdxl` casts the IP-Adapter bundle to the
    /// UNet dtype before building.
    pub fn from_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let raw = cst::load(path, device)?;
        let mut map = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            map.insert(k, coerce_float(v, dtype)?);
        }
        #[cfg(feature = "model-inference-proof")]
        gen_core::model_input_proof::record_reader_paths(
            "candle-gen::Weights::from_file",
            gen_core::model_input_proof::InputRole::ModelWeights,
            &[path],
        );
        Ok(Self::from_map(map))
    }

    /// Load and MERGE every `.safetensors` file in `files` into one weight map, in the given order
    /// (each tensor coerced to `dtype` exactly like [`from_file`](Self::from_file)).
    ///
    /// A key that appears in more than one shard is a HARD ERROR (naming the key + offending shard) —
    /// not a silent last-file-wins overwrite. In a normal sharded checkpoint every tensor lives in
    /// exactly one shard, so a cross-shard duplicate is abnormal (a mis-sharded / double-listed
    /// checkpoint, or a stray `.safetensors` polluting the dir); letting a stray file shadow the real
    /// weights with no diagnostic is exactly the footgun F-064 (sc-9050) closed for the sibling
    /// `candle-gen-sam3` / `candle-gen-depth` loaders, so `from_files` mirrors that policy.
    ///
    /// Callers pass a deterministically sorted shard list (see
    /// [`crate::loader::sorted_safetensors`]); this is the shard-aware path for snapshots that
    /// ship the checkpoint across multiple `*.safetensors` instead of a single file (F-037, sc-9021).
    pub fn from_files(files: &[impl AsRef<Path>], device: &Device, dtype: DType) -> Result<Self> {
        #[cfg(feature = "model-inference-proof")]
        let mut __input_paths = Vec::with_capacity(files.len());
        let mut map = HashMap::new();
        for path in files {
            let path = path.as_ref();
            #[cfg(feature = "model-inference-proof")]
            __input_paths.push(path);
            let raw = cst::load(path, device)?;
            for (k, v) in raw {
                let v = coerce_float(v, dtype)?;
                if map.insert(k.clone(), v).is_some() {
                    return Err(CandleError::Msg(format!(
                        "duplicate tensor key {k:?} while merging shard {}: a checkpoint's tensors \
                         must each live in exactly one .safetensors shard — this snapshot has {k:?} \
                         in more than one file (mis-sharded checkpoint or a stray .safetensors in \
                         the dir)",
                        path.display()
                    )));
                }
            }
        }
        #[cfg(feature = "model-inference-proof")]
        gen_core::model_input_proof::record_reader_paths(
            "candle-gen::Weights::from_files",
            gen_core::model_input_proof::InputRole::ModelWeights,
            &__input_paths,
        );
        Ok(Self::from_map(map))
    }

    /// Load only the tensors whose key starts with one of `prefixes`, via a header-only mmap
    /// (sc-8990 / F-010), casting floats to `dtype` exactly as [`from_file`](Self::from_file).
    ///
    /// The `openai/clip-vit-large-patch14` snapshot ships the *full* `CLIPModel` in one file, so the
    /// image embedder's old `from_file` materialized the entire checkpoint — including the unused
    /// `text_model.*` tower — on the device. Restricting to the needed prefixes (`vision_model.` +
    /// `visual_projection.`) drops that transient. Each retained tensor is byte-identical to what
    /// `from_file` would have produced for the same key.
    pub fn from_file_filtered(
        path: &Path,
        device: &Device,
        dtype: DType,
        prefixes: &[&str],
    ) -> Result<Self> {
        let st = crate::loader::mmap_safetensors(path)?;
        let mut map = HashMap::new();
        for (k, _view) in st.tensors() {
            if !prefixes.iter().any(|p| k.starts_with(p)) {
                continue;
            }
            // Load just this one tensor's bytes (header-only mmap), then re-cast floats to the compute
            // dtype via the shared `coerce_float` helper — identical per-tensor handling to
            // `from_file`, so retained values are byte-equal.
            let v = coerce_float(st.load(&k, device)?, dtype)?;
            map.insert(k, v);
        }
        #[cfg(feature = "model-inference-proof")]
        gen_core::model_input_proof::record_reader_paths(
            "candle-gen::Weights::from_file_filtered",
            gen_core::model_input_proof::InputRole::ModelWeights,
            &[path],
        );
        Ok(Self::from_map(map))
    }

    /// Shard-aware [`from_file_filtered`](Self::from_file_filtered): apply the same header-only,
    /// prefix-restricted read to EVERY `.safetensors` in `files` and merge the results, so a resharded
    /// snapshot whose prefix-matched tensors are split across shards loads their full union (F-037,
    /// sc-9021) while still never materializing the unmatched towers (sc-8990 / F-010).
    ///
    /// Duplicate-key policy matches [`from_files`](Self::from_files): a key present in more than one
    /// shard is a HARD ERROR, not a silent last-file-wins overwrite (F-064, sc-9050).
    pub fn from_files_filtered(
        files: &[impl AsRef<Path>],
        device: &Device,
        dtype: DType,
        prefixes: &[&str],
    ) -> Result<Self> {
        #[cfg(feature = "model-inference-proof")]
        let mut __input_paths = Vec::with_capacity(files.len());
        let mut map = HashMap::new();
        for path in files {
            let path = path.as_ref();
            #[cfg(feature = "model-inference-proof")]
            __input_paths.push(path);
            let st = crate::loader::mmap_safetensors(path)?;
            for (k, _view) in st.tensors() {
                if !prefixes.iter().any(|p| k.starts_with(p)) {
                    continue;
                }
                let v = coerce_float(st.load(&k, device)?, dtype)?;
                if map.insert(k.clone(), v).is_some() {
                    return Err(CandleError::Msg(format!(
                        "duplicate tensor key {k:?} while merging shard {}: a checkpoint's tensors \
                         must each live in exactly one .safetensors shard — this snapshot has {k:?} \
                         in more than one file (mis-sharded checkpoint or a stray .safetensors in \
                         the dir)",
                        path.display()
                    )));
                }
            }
        }
        #[cfg(feature = "model-inference-proof")]
        gen_core::model_input_proof::record_reader_paths(
            "candle-gen::Weights::from_files_filtered",
            gen_core::model_input_proof::InputRole::ModelWeights,
            &__input_paths,
        );
        Ok(Self::from_map(map))
    }

    /// [`from_files_filtered`](Self::from_files_filtered) that keeps every MLX packed affine triple
    /// (`{base}.weight` u32 codes + `{base}.scales` + `{base}.biases`, detected by the `.scales`
    /// sibling) **on the host** and puts everything else on `device` (sc-26691).
    ///
    /// Repacking a triple into its GGML block layout is host work (`quant::repack`), so uploading the
    /// triple to a CUDA device only for the repack to copy it straight back made a 24 B-weight q4 text
    /// encoder load at ~40 MB/s. The consumer ([`crate::quant::linear_from_weights_gs`], the packed
    /// loaders) uploads only the final repacked buffer to [`host_packed_target`](Self::host_packed_target).
    /// Tensor values are identical to `from_files_filtered`; only their residency differs, so a
    /// caller must reach packed triples through a packed-aware loader and take the destination from
    /// `host_packed_target`, not from a tensor.
    pub fn from_files_filtered_host_packed(
        files: &[impl AsRef<Path>],
        device: &Device,
        dtype: DType,
        prefixes: &[&str],
    ) -> Result<Self> {
        let mut stores = Vec::with_capacity(files.len());
        let mut packed_bases: std::collections::HashSet<String> = std::collections::HashSet::new();
        for path in files {
            let st = crate::loader::mmap_safetensors(path.as_ref())?;
            for (k, _view) in st.tensors() {
                if prefixes.iter().any(|p| k.starts_with(p)) {
                    if let Some(base) = k.strip_suffix(".scales") {
                        packed_bases.insert(base.to_owned());
                    }
                }
            }
            stores.push(st);
        }
        let host = Device::Cpu;
        let mut map = HashMap::new();
        for (path, st) in files.iter().zip(&stores) {
            let path = path.as_ref();
            for (k, _view) in st.tensors() {
                if !prefixes.iter().any(|p| k.starts_with(p)) {
                    continue;
                }
                let in_triple = [".weight", ".scales", ".biases"].iter().any(|s| {
                    k.strip_suffix(s)
                        .is_some_and(|base| packed_bases.contains(base))
                });
                let v = coerce_float(st.load(&k, if in_triple { &host } else { device })?, dtype)?;
                if map.insert(k.clone(), v).is_some() {
                    return Err(CandleError::Msg(format!(
                        "duplicate tensor key {k:?} while merging shard {}: a checkpoint's tensors \
                         must each live in exactly one .safetensors shard — this snapshot has \
                         {k:?} in more than one file (mis-sharded checkpoint or a stray \
                         .safetensors in the dir)",
                        path.display()
                    )));
                }
            }
        }
        let mut w = Self::from_map(map);
        w.target = Some(device.clone());
        Ok(w)
    }

    /// The destination device of a [`from_files_filtered_host_packed`](Self::from_files_filtered_host_packed)
    /// set, whose packed triples sit on the host so the triple's own tensors cannot name it. `None`
    /// for every other constructor — consumers then keep using the weight tensor's device.
    pub fn host_packed_target(&self) -> Option<&Device> {
        self.target.as_ref()
    }

    /// Fetch a required tensor, erroring (not panicking) when a checkpoint is missing a key.
    pub fn require(&self, key: &str) -> Result<Tensor> {
        self.map
            .get(key)
            .cloned()
            .ok_or_else(|| CandleError::Msg(format!("missing tensor: {key}")))
    }

    /// Whether `key` is present (e.g. the ControlNet's optional `encoder_hid_proj`).
    pub fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Iterate the tensor keys (drives the `ip_adapter.{n}` index discovery in the SDXL
    /// `load_ip_kv_pairs` loader).
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    /// Build directly from an in-memory map — tests (including cross-crate ones, e.g. the FLUX
    /// IP-Adapter image-encoder fixtures, sc-5872) construct synthetic weights without a file, and a
    /// caller can assemble a checkpoint programmatically.
    pub fn from_map(map: HashMap<String, Tensor>) -> Self {
        Self {
            map,
            int8: std::sync::Mutex::new(None),
            convrot_rotations: std::sync::Mutex::new(HashMap::new()),
            target: None,
        }
    }

    /// Native INT8-aware shard loading. Candle has no signed-byte dtype; preserve an I8
    /// weight's exact two's-complement bytes as U8, not an 8x I64/dense floating expansion.
    /// All floating operands retain their stored dtype, including F32 heads, timestep tables,
    /// BF16 reduced projections and F16 folded biases. Compute conversion belongs to the
    /// projection, not the checkpoint reader; one read preserves source identity and precision.
    pub fn from_files_native_int8(files: &[impl AsRef<Path>], device: &Device) -> Result<Self> {
        // Header/descriptor/scale pass first: no tensor, CUDA handle or rotation is allocated
        // until every native projection is validated, even when companions live in other shards.
        let mut stores = Vec::with_capacity(files.len());
        let mut headers = HashMap::new();
        for (index, path) in files.iter().enumerate() {
            let st = crate::loader::mmap_safetensors(path.as_ref())?;
            for (key, view) in st.tensors() {
                if key.ends_with(".comfy_quant")
                    && (view.dtype() != safetensors::Dtype::U8
                        || view.shape().len() != 1
                        || view.data().len() > gen_core::convrot_int8::MAX_DESCRIPTOR_BYTES)
                {
                    return Err(CandleError::Msg(format!(
                        "{key}: descriptor must be a bounded U8 vector"
                    )));
                }
                if headers
                    .insert(key.clone(), (index, view.dtype(), view.shape().to_vec()))
                    .is_some()
                {
                    return Err(CandleError::Msg(format!("duplicate tensor key {key:?}")));
                }
            }
            stores.push(st);
        }
        for (key, (di, _, _)) in headers.iter().filter(|(k, _)| k.ends_with(".comfy_quant")) {
            let base = key.trim_end_matches(".comfy_quant");
            let payload = stores[*di].get(key)?;
            let declared: serde_json::Value =
                cst::read_tensor_bytes(&payload, |bytes| serde_json::from_slice(bytes))
                    .map_err(|e| CandleError::Msg(format!("{key}: {e}")))?;
            let is_w4a8 = declared.get("format").and_then(serde_json::Value::as_str)
                == Some("asym_w4a8_int8");
            if is_w4a8 {
                // W4A8 (sc-26685): two int4 codes per I8 byte, F8_E4M3 per-group s_rel, F32
                // per-channel s_channel, and a F32[16] codebook. The F8 payload is preserved as
                // raw bytes (U8) exactly like the I8 codes — never a generic floating coercion.
                let s_rel = format!("{base}.weight_s_rel");
                let s_channel = format!("{base}.weight_s_channel");
                let codebook = format!("{base}.weight_codebook");
                let weight = format!("{base}.weight");
                let (_, wd, ws) = headers
                    .get(&weight)
                    .ok_or_else(|| CandleError::Msg(format!("{key}: missing weight")))?;
                let (sri, srd, srs) = headers
                    .get(&s_rel)
                    .ok_or_else(|| CandleError::Msg(format!("{key}: missing weight_s_rel")))?;
                let (sci, scd, scs) = headers
                    .get(&s_channel)
                    .ok_or_else(|| CandleError::Msg(format!("{key}: missing weight_s_channel")))?;
                let (cbi, cbd, cbs) = headers
                    .get(&codebook)
                    .ok_or_else(|| CandleError::Msg(format!("{key}: missing weight_codebook")))?;
                if *wd != safetensors::Dtype::I8
                    || *srd != safetensors::Dtype::F8_E4M3
                    || *scd != safetensors::Dtype::F32
                    || *cbd != safetensors::Dtype::F32
                {
                    return Err(CandleError::Msg(format!(
                        "{key}: native W4A8 requires I8 weight, F8_E4M3 weight_s_rel, F32 \
                         weight_s_channel and F32 weight_codebook"
                    )));
                }
                if headers.contains_key(&format!("{base}.scales"))
                    || headers.contains_key(&format!("{base}.biases"))
                    || headers.contains_key(&format!("{base}.weight_scale"))
                    || headers.contains_key(&format!("{base}.weight_correction"))
                {
                    return Err(CandleError::Msg(format!(
                        "{key}: contradictory affine / ConvRot / W4A8 operands"
                    )));
                }
                let descriptor = cst::read_tensor_bytes(&payload, |data| {
                    gen_core::w4a8_int8::W4A8Int8::parse(data, ws, srs, scs, cbs)
                })
                .map_err(CandleError::Msg)?;
                cst::read_tensor_bytes(&stores[*sri].get(&s_rel)?, |bytes| {
                    descriptor.validate_s_rel(bytes)
                })
                .map_err(CandleError::Msg)?;
                descriptor
                    .validate_s_channel(&f32_le_planes(stores[*sci].get(&s_channel)?.data()))
                    .map_err(CandleError::Msg)?;
                descriptor
                    .validate_codebook(&f32_le_planes(stores[*cbi].get(&codebook)?.data()))
                    .map_err(CandleError::Msg)?;
                continue;
            }
            let weight = format!("{base}.weight");
            let scale = format!("{base}.weight_scale");
            let (_, wd, ws) = headers
                .get(&weight)
                .ok_or_else(|| CandleError::Msg(format!("{key}: missing weight")))?;
            let (si, sd, ss) = headers
                .get(&scale)
                .ok_or_else(|| CandleError::Msg(format!("{key}: missing weight_scale")))?;
            if *wd != safetensors::Dtype::I8 || *sd != safetensors::Dtype::F32 {
                return Err(CandleError::Msg(format!(
                    "{key}: native ConvRot requires original I8 storage and F32 scales"
                )));
            }
            if headers.contains_key(&format!("{base}.scales"))
                || headers.contains_key(&format!("{base}.biases"))
            {
                return Err(CandleError::Msg(format!(
                    "{key}: contradictory affine and ConvRot operands"
                )));
            }
            let descriptor = cst::read_tensor_bytes(&payload, |data| {
                gen_core::convrot_int8::ConvRotInt8::parse(data, ws, ss)
            })
            .map_err(CandleError::Msg)?;
            let scales = stores[*si].get(&scale)?;
            descriptor
                .validate_scale_values(f32_le_planes(scales.data()))
                .map_err(CandleError::Msg)?;
        }
        for (key, (_, stored_dtype, _)) in &headers {
            let native = (key.ends_with(".weight") && *stored_dtype == safetensors::Dtype::I8)
                || key.ends_with(".weight_scale")
                || key.ends_with(".weight_s_rel")
                || key.ends_with(".weight_s_channel")
                || key.ends_with(".weight_codebook");
            if native {
                let base = key.rsplit_once('.').expect("suffix above contains a dot").0;
                if !headers.contains_key(&format!("{base}.comfy_quant")) {
                    return Err(CandleError::Msg(format!(
                        "{key}: native INT8/W4A8 requires .comfy_quant"
                    )));
                }
            }
        }
        let mut map = HashMap::with_capacity(headers.len());
        for st in &stores {
            for (key, view) in st.tensors() {
                let tensor = if view.dtype() == safetensors::Dtype::I8 {
                    Tensor::from_raw_buffer(view.data(), DType::U8, view.shape(), device)?
                } else {
                    st.load(&key, device)?
                };
                map.insert(key, tensor);
            }
        }
        Ok(Self::from_map(map))
    }

    /// One lazy cuBLASLt handle/workspace for every native INT8 projection in this map.
    pub fn int8_context(&self, device: &Device) -> Result<crate::quant::Int8Context> {
        let mut cached = self.int8.lock().unwrap_or_else(|p| p.into_inner());
        if cached.is_none() {
            *cached = Some(crate::quant::Int8Context::new(device)?);
        }
        Ok(cached.as_ref().expect("initialized above").clone())
    }

    /// Share one immutable rotation per group on this map's device (no per-layer matrix copies).
    pub fn convrot_rotation(&self, group: usize, device: &Device) -> Result<Tensor> {
        let mut cache = self
            .convrot_rotations
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if let Some(r) = cache.get(&group) {
            if !r.device().same_device(device) {
                return Err(CandleError::Msg(
                    "ConvRot map contains multiple devices".into(),
                ));
            }
            return Ok(r.clone());
        }
        let r = crate::quant::regular_hadamard(group, device)?;
        cache.insert(group, r.clone());
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write_st(path: &Path, name: &str, value: f32) {
        let t = Tensor::new(&[value], &Device::Cpu).unwrap();
        let mut m = HashMap::new();
        m.insert(name.to_string(), t);
        candle_core::safetensors::save(&m, path).unwrap();
    }

    fn scratch_dir(tmp: &tempfile::TempDir, tag: &str) -> PathBuf {
        let d = tmp.path().join(format!("candle_gen_weights_test_{tag}"));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// `from_files` merges every shard into one map — the UNION of disjoint keys is visible (the
    /// shard-aware path for F-037).
    #[test]
    fn from_files_merges_disjoint_shards() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = scratch_dir(&tmp, "merge");
        let a = dir.join("model-00001-of-00002.safetensors");
        let b = dir.join("model-00002-of-00002.safetensors");
        write_st(&a, "a.weight", 1.0);
        write_st(&b, "b.weight", 2.0);
        let w = Weights::from_files(&[a, b], &Device::Cpu, DType::F32).unwrap();
        assert_eq!(
            w.require("a.weight").unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0]
        );
        assert_eq!(
            w.require("b.weight").unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A key present in more than one shard is a HARD ERROR naming the key — not a silent
    /// last-file-wins overwrite (align with the F-064 / sc-9050 duplicate-key policy that the sibling
    /// sam3/depth loaders enforce). A stray or mis-sharded file must not silently shadow real weights.
    #[test]
    fn from_files_errors_on_duplicate() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = scratch_dir(&tmp, "dup");
        let first = dir.join("a.safetensors");
        let last = dir.join("b.safetensors");
        write_st(&first, "shared", 10.0);
        write_st(&last, "shared", 20.0);
        match Weights::from_files(&[first, last], &Device::Cpu, DType::F32) {
            Err(CandleError::Msg(m)) => assert!(
                m.contains("duplicate tensor key") && m.contains("shared"),
                "expected a duplicate-key error naming the key, got: {m}"
            ),
            Err(e) => panic!("expected a duplicate-key CandleError::Msg, got: {e}"),
            Ok(_) => panic!("expected a duplicate-key error, but from_files succeeded"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn native_int8_loader_preserves_all_source_float_precisions() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("mixed.safetensors");
        let device = Device::Cpu;
        let source = Tensor::new(&[1.0001f32, -0.1234567, -0.0], &device).unwrap();
        let mut map = HashMap::new();
        for (name, dtype) in [
            ("head", DType::F32),
            ("projection", DType::BF16),
            ("bias", DType::F16),
        ] {
            map.insert(name.to_owned(), source.to_dtype(dtype).unwrap());
        }
        cst::save(&map, &file).unwrap();
        let loaded = Weights::from_files_native_int8(&[file], &device).unwrap();
        for (name, original) in map {
            let actual = loaded.require(&name).unwrap();
            assert_eq!(actual.dtype(), original.dtype(), "{name}");
            let bits = |tensor: &Tensor| {
                tensor
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&actual), bits(&original), "{name}");
        }
    }

    /// `from_file_filtered` materializes only the prefix-matched keys (byte-identical to `from_file`
    /// for those keys) and drops everything else — e.g. the unused CLIP text tower for the image path.
    #[test]
    fn from_file_filtered_keeps_only_matching_prefixes() {
        let dev = Device::Cpu;
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let file = dir.join("model.safetensors");

        let mut map = HashMap::new();
        map.insert(
            "vision_model.post_layernorm.weight".to_string(),
            Tensor::new(&[1.0f32, 2.0], &dev).unwrap(),
        );
        map.insert(
            "visual_projection.weight".to_string(),
            Tensor::new(&[3.0f32, 4.0], &dev).unwrap(),
        );
        map.insert(
            "text_model.embeddings.token_embedding.weight".to_string(),
            Tensor::new(&[9.0f32, 9.0], &dev).unwrap(),
        );
        cst::save(&map, &file).unwrap();

        let w = Weights::from_file_filtered(
            &file,
            &dev,
            DType::F32,
            &["vision_model.", "visual_projection."],
        )
        .unwrap();

        // Kept: the two vision-side prefixes, values intact.
        assert!(w.contains("vision_model.post_layernorm.weight"));
        assert_eq!(
            w.require("visual_projection.weight")
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![3.0, 4.0]
        );
        // Dropped: the unused text tower is never materialized.
        assert!(!w.contains("text_model.embeddings.token_embedding.weight"));
    }

    /// `from_files_filtered` unions the prefix-matched keys across shards (F-037 × F-010) and still
    /// drops the unmatched tower; a cross-shard duplicate is the same hard error as `from_files`.
    #[test]
    fn from_files_filtered_unions_shards_and_drops_unmatched() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = scratch_dir(&tmp, "filter_shards");
        let a = dir.join("model-00001-of-00002.safetensors");
        let b = dir.join("model-00002-of-00002.safetensors");
        // Vision tensors split across two shards; a text-tower tensor (co-resident in shard b) that
        // must be dropped. `write_st` writes a single-tensor file, so shard b is saved directly.
        write_st(&a, "vision_model.a.weight", 1.0);
        let mut mb = HashMap::new();
        mb.insert(
            "visual_projection.weight".to_string(),
            Tensor::new(&[2.0f32], &Device::Cpu).unwrap(),
        );
        mb.insert(
            "text_model.dead.weight".to_string(),
            Tensor::new(&[9.0f32], &Device::Cpu).unwrap(),
        );
        cst::save(&mb, &b).unwrap();

        let w = Weights::from_files_filtered(
            &[a, b],
            &Device::Cpu,
            DType::F32,
            &["vision_model.", "visual_projection."],
        )
        .unwrap();
        assert_eq!(
            w.require("vision_model.a.weight")
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0]
        );
        assert!(w.contains("visual_projection.weight"));
        assert!(!w.contains("text_model.dead.weight"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// sc-26691: the host-packed loader returns the same tensor values as `from_files_filtered`,
    /// records the destination, and (on CPU, where residency cannot differ) still unions shards.
    #[test]
    fn host_packed_loader_matches_filtered_values_and_records_target() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = scratch_dir(&tmp, "host_packed");
        let a = dir.join("a.safetensors");
        let mut m = HashMap::new();
        m.insert(
            "lm.q.weight".to_string(),
            Tensor::new(&[0x1234_5678u32, 7], &Device::Cpu).unwrap(),
        );
        m.insert(
            "lm.q.scales".to_string(),
            Tensor::new(&[0.5f32], &Device::Cpu).unwrap(),
        );
        m.insert(
            "lm.q.biases".to_string(),
            Tensor::new(&[0.25f32], &Device::Cpu).unwrap(),
        );
        m.insert(
            "lm.n.weight".to_string(),
            Tensor::new(&[3.0f32], &Device::Cpu).unwrap(),
        );
        m.insert(
            "other.x".to_string(),
            Tensor::new(&[9.0f32], &Device::Cpu).unwrap(),
        );
        cst::save(&m, &a).unwrap();
        let hp =
            Weights::from_files_filtered_host_packed(&[&a], &Device::Cpu, DType::F32, &["lm."])
                .unwrap();
        let plain =
            Weights::from_files_filtered(&[&a], &Device::Cpu, DType::F32, &["lm."]).unwrap();
        assert!(hp.host_packed_target().is_some() && plain.host_packed_target().is_none());
        assert!(!hp.contains("other.x"));
        for k in ["lm.q.weight", "lm.q.scales", "lm.q.biases", "lm.n.weight"] {
            let (x, y) = (hp.require(k).unwrap(), plain.require(k).unwrap());
            assert_eq!(x.dtype(), y.dtype());
            assert_eq!(x.to_vec1::<f32>().ok(), y.to_vec1::<f32>().ok(), "{k}");
        }
        assert_eq!(
            hp.require("lm.q.weight").unwrap().to_vec1::<u32>().unwrap(),
            vec![0x1234_5678, 7]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A cross-shard DUPLICATE under a matched prefix is a HARD ERROR naming the key — the same F-064
    /// (sc-9050) policy `from_files` enforces, extended to the filtered/header-only path.
    #[test]
    fn from_files_filtered_errors_on_cross_shard_duplicate() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = scratch_dir(&tmp, "filter_dup");
        let a = dir.join("model-00001-of-00002.safetensors");
        let b = dir.join("model-00002-of-00002.safetensors");
        // Same matched-prefix key in BOTH shards → must not silently last-file-wins.
        write_st(&a, "vision_model.shared.weight", 1.0);
        write_st(&b, "vision_model.shared.weight", 2.0);
        match Weights::from_files_filtered(&[a, b], &Device::Cpu, DType::F32, &["vision_model."]) {
            Err(CandleError::Msg(m)) => assert!(
                m.contains("duplicate tensor key") && m.contains("vision_model.shared.weight"),
                "expected a duplicate-key error naming the key, got: {m}"
            ),
            Err(e) => panic!("expected a duplicate-key CandleError::Msg, got: {e}"),
            Ok(_) => panic!("expected a duplicate-key error, but from_files_filtered succeeded"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
