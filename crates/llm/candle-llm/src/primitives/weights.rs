//! Safetensors weight loading.
//!
//! [`Weights`] is a flat name → `Tensor` map loaded from a single file or a sharded HF snapshot
//! directory (`model-00001-of-0000N.safetensors`, …) onto a chosen [`Device`]. Models look tensors
//! up by their HF key via [`Weights::require`] / [`Weights::get`]. The Candle port of `mlx-llm`'s
//! `Weights` (Candle reads safetensors into device tensors directly).

use candle_checkpoint::ProjectionSource;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use candle_core::safetensors::{Load, MmapedSafetensors};
use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::primitives::quant::ggml_block_storage;

/// Whether `name` is part of a ModelOpt NVFP4 projection triple (SW-7099) that
/// [`Weights::from_dir`] keeps on the host: a companion scale ([`core_llm::modelopt::is_companion`])
/// or a rank-2 U8 `.weight` (packed E2M1 codes — a dense projection is never U8, and a stored GGML
/// block tensor is rank 3).
pub(crate) fn modelopt_host_tensor(name: &str, is_u8: bool, shape: &[usize]) -> bool {
    core_llm::modelopt::is_companion(name)
        || (is_u8 && shape.len() == 2 && name.ends_with(".weight"))
}

/// A loaded set of named weight tensors, plus the device they live on.
pub struct Weights {
    tensors: HashMap<String, Tensor>,
    device: Device,
    encoded: HashMap<String, EncodedRow>,
    source: Option<ProjectionSource>,
    native_census: std::sync::Mutex<super::ProjectionCensus>,
}

// Models may address a logical key through a different prefix (e.g. a companion MTP head).
// Keep the original reader key and placement with the row. Only coefficient consumers decode it.
struct EncodedRow {
    key: String,
    model_shape: Option<[usize; 2]>,
    device: Device,
    dense: OnceLock<Tensor>,
}

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weights")
            .field("tensors", &self.len())
            .field("encoded", &self.encoded.len())
            .field("device", &self.device)
            .finish()
    }
}

impl Weights {
    /// Construct directly from an in-memory map (used by converters and tests).
    pub fn from_map(tensors: HashMap<String, Tensor>, device: Device) -> Self {
        Self {
            tensors,
            device,
            encoded: HashMap::new(),
            source: None,
            native_census: Default::default(),
        }
    }

    /// Adopt a pre-authenticated provider component without forcing its stored projections
    /// through a dense coefficient map.
    pub fn from_checkpoint_parts(
        tensors: HashMap<String, Tensor>,
        source: Option<ProjectionSource>,
        device: Device,
    ) -> Self {
        let mut out = Self::from_map(tensors, device);
        if let Some(source) = source {
            out.encoded = Self::encoded_rows(&source, &out.device, None);
            out.source = Some(source);
        }
        out
    }

    /// Load every tensor from a single `.safetensors` file onto `device`.
    pub fn from_file(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let weights = Self::mapped(&[path.to_owned()], device, None, false)?;
        #[cfg(feature = "model-inference-proof")]
        crate::model_input_proof::record_reader_paths(
            "candle-llm::primitives::Weights::from_file",
            crate::model_input_proof::InputRole::ModelWeights,
            &[path],
        );
        Ok(weights)
    }

    /// Load one shard onto the host with [`Weights::from_dir`]'s placement rules — a ModelOpt
    /// NVFP4 triple as raw bytes (E4M3 scales as U8 codes). The ModelOpt converter reads shards
    /// through this (SW-7099).
    pub fn from_file_host_raw(path: impl AsRef<Path>) -> Result<Self> {
        // Snapshot conversion operates on physical bytes, without interpreting source metadata.
        let path = path.as_ref();
        let mapped = unsafe { MmapedSafetensors::new(path)? };
        Ok(Self::from_map(
            Self::plain_tensors(&mapped, &Device::Cpu, None, true, None)?,
            Device::Cpu,
        ))
    }

    /// Load and merge every `*.safetensors` shard in a snapshot directory onto `device`.
    ///
    /// A prepared tier's stored GGML block tensors (sc-19375: `U8` `[.., rows, blocks,
    /// block_bytes]`, see [`from_ggml_block_tensor`](crate::primitives::quant::from_ggml_block_tensor))
    /// are read onto the **host** instead: the loader rebuilds each as a `QTensor` on `device` straight from those
    /// bytes, so a copy of them on `device` would only be a second, transient device residency of
    /// every quantized projection. Load admission prices them on the host accordingly.
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = dir.as_ref();
        let mut shards: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        if shards.is_empty() {
            return Err(Error::Msg(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        shards.sort();
        Self::from_shards(shards, device)
    }

    pub(crate) fn from_shards(shards: Vec<std::path::PathBuf>, device: &Device) -> Result<Self> {
        let weights = Self::mapped(&shards, device, None, true)?;
        #[cfg(feature = "model-inference-proof")]
        crate::model_input_proof::record_reader_paths(
            "candle-llm::primitives::Weights::from_dir",
            crate::model_input_proof::InputRole::ModelWeights,
            &shards,
        );
        Ok(weights)
    }

    /// Load a snapshot directory **pipeline-sharded**: each tensor is placed on the device chosen by
    /// `place` (keyed by its HF name), so a model too large for one GPU can be split across several
    /// without ever staging the whole thing on a single card. Each shard file is read to host memory,
    /// its tensors are moved to their target devices, and the host copy is dropped — so peak *device*
    /// memory is per-GPU-shard and peak *host* memory is one shard file. `device` records the model's
    /// home device (where embeddings / the first layer live); it is the value [`Weights::device`]
    /// reports and the decoder treats as its input device.
    pub fn from_dir_sharded(
        dir: impl AsRef<Path>,
        device: Device,
        place: impl Fn(&str) -> Device,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let mut shards: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();
        if shards.is_empty() {
            return Err(Error::Msg(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        shards.sort();
        let weights = Self::mapped(&shards, &device, Some(&place), false)?;
        #[cfg(feature = "model-inference-proof")]
        crate::model_input_proof::record_reader_paths(
            "candle-llm::primitives::Weights::from_dir_sharded",
            crate::model_input_proof::InputRole::ModelWeights,
            &shards,
        );
        Ok(weights)
    }

    /// ModelOpt uses row-major NVFP4 block scales and already has its own native importer.
    /// Preserve its original byte layout alongside any independently described checkpoint rows.
    pub(crate) fn checkpoint_source(
        files: &[PathBuf],
        device: &Device,
    ) -> Result<std::result::Result<ProjectionSource, MmapedSafetensors>> {
        // SAFETY: the caller holds the checkpoint immutable for the lifetime of its loaded weights.
        let mapped = unsafe { MmapedSafetensors::multi(files)? };
        let modelopt = mapped
            .tensors()
            .into_iter()
            .filter_map(|(name, view)| {
                let stem = name.strip_suffix(".weight")?;
                if view.shape().len() == 2
                    && DType::try_from(view.dtype()).ok() == Some(DType::U8)
                    && mapped.get(&format!("{stem}.comfy_quant")).is_err()
                    && mapped
                        .get(&format!("{stem}.weight_scale"))
                        .ok()
                        .is_some_and(|s| {
                            s.shape().len() == 2
                                && (DType::try_from(s.dtype()).ok() == Some(DType::F8E4M3)
                                    || (DType::try_from(s.dtype()).ok() == Some(DType::U8)
                                        && mapped.get(&format!("{stem}.weight_scale_2")).is_ok()))
                        })
                {
                    Some(name)
                } else {
                    None
                }
            })
            .collect::<std::collections::HashSet<_>>();
        Ok(ProjectionSource::from_mapped_selected(
            mapped,
            files,
            device,
            |name| !modelopt.contains(name),
        )?)
    }

    fn mapped(
        files: &[PathBuf],
        device: &Device,
        place: Option<&dyn Fn(&str) -> Device>,
        host_packed: bool,
    ) -> Result<Self> {
        match Self::checkpoint_source(files, device)? {
            Err(plain) => Ok(Self::from_map(
                Self::plain_tensors(&plain, device, place, host_packed, None)?,
                device.clone(),
            )),
            Ok(source) => {
                let plan = source.reader().plan();
                let claimed = plan
                    .all_physical_keys()
                    .collect::<std::collections::HashSet<_>>();
                let tensors =
                    Self::plain_tensors(source.mmap(), device, place, host_packed, Some(&claimed))?;
                let encoded = Self::encoded_rows(&source, device, place);
                Ok(Self {
                    tensors,
                    device: device.clone(),
                    encoded,
                    source: Some(source),
                    native_census: Default::default(),
                })
            }
        }
    }

    fn encoded_rows(
        source: &ProjectionSource,
        device: &Device,
        place: Option<&dyn Fn(&str) -> Device>,
    ) -> HashMap<String, EncodedRow> {
        source
            .reader()
            .plan()
            .tensors
            .iter()
            .map(|row| {
                let key = &row.logical_key;
                (
                    key.clone(),
                    EncodedRow {
                        model_shape: None,
                        key: key.clone(),
                        device: place.map_or_else(|| device.clone(), |p| p(key)),
                        dense: OnceLock::new(),
                    },
                )
            })
            .collect()
    }

    fn plain_tensors(
        mapped: &MmapedSafetensors,
        device: &Device,
        place: Option<&dyn Fn(&str) -> Device>,
        host_packed: bool,
        claimed: Option<&std::collections::HashSet<&str>>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        for (name, view) in mapped.tensors() {
            if claimed.is_some_and(|keys| keys.contains(name.as_str())) {
                continue;
            }
            let is_u8 = DType::try_from(view.dtype()).is_ok_and(|d| d == DType::U8);
            let tensor = if host_packed && modelopt_host_tensor(&name, is_u8, view.shape()) {
                let dtype = match DType::try_from(view.dtype())? {
                    DType::F8E4M3 => DType::U8,
                    other => other,
                };
                candle_core::safetensors::read_tensor_bytes(&view, |data| {
                    Tensor::from_raw_buffer(data, dtype, view.shape(), &Device::Cpu)
                })?
            } else if host_packed && ggml_block_storage(is_u8, view.shape()).is_some() {
                view.load(&Device::Cpu)?
            } else {
                let target = place.map_or_else(|| device.clone(), |p| p(&name));
                view.load(&target)?
            };
            tensors.insert(name, tensor);
        }
        Ok(tensors)
    }

    pub(crate) fn encoded_device(&self, key: &str) -> Option<&Device> {
        self.encoded.get(key).map(|r| &r.device)
    }

    pub(crate) fn tensor_device(&self, key: &str) -> Result<&Device> {
        if let Some(row) = self.encoded.get(key) {
            return Ok(&row.device);
        }
        Ok(self.require(key)?.device())
    }

    pub(crate) fn bind_matrix_shape(&mut self, key: &str, shape: [usize; 2]) -> Result<()> {
        if let Some(row) = self.encoded.get_mut(key) {
            self.source
                .as_ref()
                .expect("owned checkpoint")
                .reader()
                .validate_matrix_shape(&row.key, shape)?;
            row.model_shape = Some(shape);
        }
        Ok(())
    }

    pub(crate) fn shape(&self, key: &str) -> Option<&[usize]> {
        if let Some(row) = self.encoded.get(key) {
            if let Some(shape) = &row.model_shape {
                return Some(shape.as_slice());
            }
            return Some(&self.source.as_ref()?.reader().planned(&row.key)?.shape);
        }
        self.tensors.get(key).map(Tensor::dims)
    }

    pub(crate) fn add_key_prefix(&mut self, prefix: &str) {
        self.tensors = self
            .tensors
            .drain()
            .map(|(k, v)| (format!("{prefix}{k}"), v))
            .collect();
        self.encoded = self
            .encoded
            .drain()
            .map(|(k, v)| (format!("{prefix}{k}"), v))
            .collect();
    }

    /// A tied LM head may use the encoded matrix while lookup still needs real coefficients.
    /// Without a native operator the caller shares the embedding's existing dense storage.
    pub(crate) fn native_projection(
        &self,
        key: &str,
        dtype: DType,
        device: &Device,
        bias: Option<Tensor>,
    ) -> Result<Option<super::Projection>> {
        let Some(row) = self.encoded.get(key) else {
            return Ok(None);
        };
        use candle_nn::var_builder::SimpleBackend;
        let op = self
            .source
            .as_ref()
            .expect("encoded row belongs to source")
            .native_linear_weight(
                &row.key,
                row.model_shape.map(|[n, k]| (n, k)),
                dtype,
                device,
            )?
            .map(|op| super::Projection::from_native(candle_nn::Linear::from_native(op, bias)))
            .transpose()?;
        Ok(self.record_native(op))
    }

    pub(crate) fn native_projection_slice(
        &self,
        key: &str,
        dtype: DType,
        device: &Device,
        first_row: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Option<super::Projection>> {
        let Some(row) = self.encoded.get(key) else {
            return Ok(None);
        };
        let source = self.source.as_ref().expect("encoded row belongs to source");
        let op = source
            .native_linear_slice(&row.key, device, dtype, first_row, rows, cols)?
            .map(super::Projection::from_native)
            .transpose()?;
        Ok(self.record_native(op))
    }

    pub(crate) fn native_flattened_projection(
        &self,
        key: &str,
        source_shape: &[usize],
        dtype: DType,
        bias: Option<Tensor>,
    ) -> Result<Option<super::Projection>> {
        use candle_nn::var_builder::SimpleBackend;
        let Some(row) = self.encoded.get(key) else {
            return Ok(None);
        };
        let source = self.source.as_ref().expect("encoded row belongs to source");
        let op = source
            .native_linear_weight_flattened(&row.key, source_shape, dtype, &row.device)?
            .map(|op| super::Projection::from_native(candle_nn::Linear::from_native(op, bias)))
            .transpose()?;
        Ok(self.record_native(op))
    }

    fn record_native(&self, op: Option<super::Projection>) -> Option<super::Projection> {
        if let Some(op) = &op {
            self.native_census
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .record(op);
        }
        op
    }

    pub(crate) fn native_census(&self) -> super::ProjectionCensus {
        *self.native_census.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Load the publisher's projection, or its codec-defined dense fallback for this device.
    /// Resolving this before any raw tensor lookup keeps codes and companions together.
    pub(crate) fn checkpoint_projection(
        &self,
        key: &str,
        dtype: DType,
        device: &Device,
        bias: Option<Tensor>,
    ) -> Result<Option<super::Projection>> {
        let Some(row) = self.encoded.get(key) else {
            return Ok(None);
        };
        let source = self
            .source
            .as_ref()
            .expect("encoded row belongs to the source");
        if let Some(op) = self.native_projection(key, dtype, device, bias.clone())? {
            return Ok(Some(op));
        }
        let dense = source.on_device(device)?.reader().read_dense_with_shape(
            &row.key,
            row.model_shape.map(|[n, k]| (n, k)),
            dtype,
            device,
        )?;
        Ok(Some(super::Projection::load_with_bias(dense, bias, None)?))
    }

    pub(crate) fn linear(
        &self,
        key: &str,
        dtype: DType,
        bias: Option<Tensor>,
    ) -> Result<super::Projection> {
        if let Some(p) =
            self.checkpoint_projection(key, dtype, self.tensor_device(key)?, bias.clone())?
        {
            return Ok(p);
        }
        super::Projection::load_with_bias(self.require(key)?.to_dtype(dtype)?, bias, None)
    }

    /// Preserve the published dtype for plain matrices in models that use it as compute dtype.
    pub(crate) fn linear_dtype(&self, key: &str) -> Result<DType> {
        Ok(match self.plain(key) {
            Some(t) => t.dtype(),
            None => crate::compute_dtype(self.tensor_device(key)?),
        })
    }

    pub(crate) fn plain(&self, key: &str) -> Option<&Tensor> {
        self.tensors.get(key)
    }

    pub(crate) fn encoded_dense(
        &self,
        key: &str,
        dtype: DType,
        device: Option<&Device>,
    ) -> Result<Option<Tensor>> {
        let Some(row) = self.encoded.get(key) else {
            return Ok(None);
        };
        let device = device.unwrap_or(&row.device);
        Ok(Some(
            self.source
                .as_ref()
                .expect("encoded row belongs to source")
                .on_device(device)?
                .reader()
                .read_dense_with_shape(
                    &row.key,
                    row.model_shape.map(|[n, k]| (n, k)),
                    dtype,
                    device,
                )?,
        ))
    }

    /// Fetch actual coefficients. Native matrix consumers use the source directly.
    pub fn require(&self, key: &str) -> Result<&Tensor> {
        if let Some(row) = self.encoded.get(key) {
            if let Some(t) = row.dense.get() {
                return Ok(t);
            }
            let t = self
                .source
                .as_ref()
                .expect("encoded row belongs to source")
                .on_device(&row.device)?
                .reader()
                .read_dense_with_shape(
                    &row.key,
                    row.model_shape.map(|[n, k]| (n, k)),
                    DType::F32,
                    &row.device,
                )?;
            let _ = row.dense.set(t);
            return Ok(row.dense.get().expect("initialized"));
        }
        self.tensors
            .get(key)
            .ok_or_else(|| Error::MissingTensor(key.to_owned()))
    }

    /// Fetch a tensor by key if present.
    pub fn get(&self, key: &str) -> Option<&Tensor> {
        self.require(key).ok()
    }

    /// A missing optional coefficient and a malformed encoded coefficient are distinct.
    pub(crate) fn optional(&self, key: &str) -> Result<Option<&Tensor>> {
        self.contains(key).then(|| self.require(key)).transpose()
    }

    /// Whether a key is present.
    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key) || self.encoded.contains_key(key)
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len() + self.encoded.len()
    }

    /// Whether no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty() && self.encoded.is_empty()
    }

    /// All loaded tensor keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors
            .keys()
            .chain(self.encoded.keys())
            .map(|s| s.as_str())
    }

    /// The device the tensors live on (the decoder reads this to pick its compute dtype).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Consume into the underlying `name → Tensor` map (used by the snapshot writer, which mutates
    /// then re-serializes the tensors via `candle_core::safetensors::save`).
    /// Preserves the original infallible map conversion for plain/in-memory checkpoints.
    /// A source-native model may have no dense coefficients (ConvRot); for external
    /// checkpoints use [`Self::try_into_map`] to propagate that error.
    pub fn into_map(self) -> HashMap<String, Tensor> {
        self.try_into_map()
            .expect("checkpoint cannot be represented as a dense tensor map")
    }

    /// Explicit coefficient export. Releases the mapping after the logical weights are read.
    pub fn try_into_map(mut self) -> Result<HashMap<String, Tensor>> {
        for (name, row) in self.encoded {
            let t = match row.dense.into_inner() {
                Some(t) => t,
                None => self
                    .source
                    .as_ref()
                    .expect("encoded row belongs to source")
                    .on_device(&row.device)?
                    .reader()
                    .read_dense_with_shape(
                        &row.key,
                        row.model_shape.map(|[n, k]| (n, k)),
                        DType::F32,
                        &row.device,
                    )?,
            };
            self.tensors.insert(name, t);
        }
        Ok(self.tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_and_get_on_in_memory_map() {
        let mut m = HashMap::new();
        m.insert(
            "a.weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0], (2,), &Device::Cpu).unwrap(),
        );
        let w = Weights::from_map(m, Device::Cpu);
        assert_eq!(w.len(), 1);
        assert!(w.contains("a.weight"));
        assert!(w.require("a.weight").is_ok());
        assert!(w.get("missing").is_none());
        assert!(matches!(w.require("missing"), Err(Error::MissingTensor(_))));
    }

    #[test]
    fn sharded_checkpoint_keeps_scale_and_codes_together_through_prefix_mapping() {
        let dir = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let codes = Tensor::from_raw_buffer(&[0x38; 16], DType::F8E4M3, &[4, 4], &dev).unwrap();
        candle_core::safetensors::save(
            &HashMap::from([("proj.weight", codes)]),
            dir.path().join("a.safetensors"),
        )
        .unwrap();
        let descriptor = br#"{"format":"float8_e4m3fn"}"#;
        candle_core::safetensors::save(
            &HashMap::from([
                ("proj.weight_scale", Tensor::new(2f32, &dev).unwrap()),
                (
                    "proj.comfy_quant",
                    Tensor::new(descriptor.as_slice(), &dev).unwrap(),
                ),
            ]),
            dir.path().join("b.safetensors"),
        )
        .unwrap();
        let mut w = Weights::from_dir_sharded(dir.path(), dev.clone(), |_| dev.clone()).unwrap();
        assert_eq!(w.keys().collect::<Vec<_>>(), ["proj.weight"]);
        assert!(w.plain("proj.weight").is_none());
        w.add_key_prefix("mtp.");
        assert_eq!(w.shape("mtp.proj.weight"), Some([4, 4].as_slice()));
        let op = w.linear("mtp.proj.weight", DType::F32, None).unwrap();
        assert_eq!(op.resident_bytes(), Some(64));
        assert_eq!(
            w.require("mtp.proj.weight")
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![2.; 16]
        );
    }

    #[test]
    fn save_then_load_roundtrip() {
        // sc-17755: guard the fixture root so it leaves on `Drop`, panic or not.
        let guard = tempfile::Builder::new()
            .prefix("candle-llm-weights-test-")
            .tempdir()
            .expect("fixture temp dir");
        let dir = guard.path();
        let path = dir.join("model.safetensors");
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        let mut map = HashMap::new();
        map.insert("w".to_string(), a);
        candle_core::safetensors::save(&map, &path).unwrap();

        let w = Weights::from_file(&path, &Device::Cpu).unwrap();
        assert_eq!(w.require("w").unwrap().dims(), &[2, 2]);

        let w2 = Weights::from_dir(dir, &Device::Cpu).unwrap();
        assert!(w2.contains("w"));
    }
}
