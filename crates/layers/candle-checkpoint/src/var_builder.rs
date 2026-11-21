//! Checkpoint-aware stock VarBuilder. One mapped source owns the original codes and companions.
//! Constructors asking for a Linear select the native GEMM; coefficient reads use the reference
//! codec. Plain tensors retain Candle's original lookup, shape and dtype behavior.
use crate::logical_weights::{baseline_codec_registry, CandleCodecResidency, LogicalWeightReader};
use candle_core::safetensors::{self as st, MmapedSafetensors};
use candle_core::{DType, Device, Error, Result, Shape, Tensor};
use candle_nn::linear::NativeLinearWeight;
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};
use gen_core::checkpoint_codec::LogicalKeyMapping;
use gen_core::SafetensorsTensorHeader;
use safetensors::Dtype;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

struct SourceMapping;
impl LogicalKeyMapping for SourceMapping {
    fn mapping_id(&self) -> &'static str {
        "candle-constructor-v1"
    }
    fn logical_key(&self, key: &str) -> Option<String> {
        Some(key.to_owned())
    }
    fn accepts_descriptor_shape(&self) -> bool {
        true
    }
    fn inline_fp8_scales(&self) -> bool {
        true
    }
}

/// The same source can serve stock builders and typed models that collect a raw tensor map.
#[derive(Clone)]
pub struct ProjectionSource(Arc<LogicalWeightReader>, Arc<SourcePlan>);
struct SourcePlan {
    headers: Vec<SafetensorsTensorHeader>,
    descriptors: BTreeMap<String, Vec<u8>>,
    metadata: Option<String>,
    other_devices: std::sync::Mutex<Vec<Arc<LogicalWeightReader>>>,
}
impl SourcePlan {
    fn open(&self, st: Arc<MmapedSafetensors>, dev: &Device) -> Result<Arc<LogicalWeightReader>> {
        let policy = CandleCodecResidency::probe(dev);
        let plan = gen_core::compile_logical_weight_plan_with_metadata(
            &self.headers,
            &self.descriptors,
            self.metadata.as_deref(),
            &SourceMapping,
            baseline_codec_registry(),
            &policy,
        )
        .map_err(Error::wrap)?;
        Ok(Arc::new(LogicalWeightReader::from_mapped(
            st,
            plan,
            dev,
            policy.native_execution_capability(),
        )))
    }
}
impl ProjectionSource {
    pub fn reader(&self) -> &LogicalWeightReader {
        &self.0
    }

    pub fn on_device(&self, dev: &Device) -> Result<Self> {
        if self.0.device().same_device(dev) {
            return Ok(self.clone());
        }
        let mut cache = self
            .1
            .other_devices
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(reader) = cache.iter().find(|reader| reader.device().same_device(dev)) {
            return Ok(Self(reader.clone(), self.1.clone()));
        }
        let reader = self.1.open(self.0.st.clone(), dev)?;
        cache.push(reader.clone());
        Ok(Self(reader, self.1.clone()))
    }

    pub fn mmap(&self) -> &MmapedSafetensors {
        &self.0.st
    }
    pub fn native_linear(
        &self,
        key: &str,
        dtype: DType,
        dev: &Device,
        bias: Option<Tensor>,
    ) -> Result<Option<candle_nn::Linear>> {
        Ok(self
            .native_linear_weight(key, None, dtype, dev)?
            .map(|op| candle_nn::Linear::from_native(op, bias)))
    }

    pub fn native_linear_transposed(
        &self,
        key: &str,
        shape: (usize, usize),
        dtype: DType,
        dev: &Device,
        bias: Option<Tensor>,
    ) -> Result<Option<candle_nn::Linear>> {
        Ok(self
            .native_linear_weight_transposed(key, shape, dtype, dev)?
            .map(|op| candle_nn::Linear::from_native(op, bias)))
    }

    /// Return the original mmap for a plain checkpoint. Detection inspects tensor headers only.
    pub fn from_mapped(
        st: MmapedSafetensors,
        files: &[PathBuf],
        dev: &Device,
    ) -> Result<std::result::Result<Self, MmapedSafetensors>> {
        Self::from_mapped_selected(st, files, dev, |_| true)
    }

    /// Restrict the logical plan to a component's selected physical keys. Companion names
    /// are captured by matrix base across shards, even if the projection caller selected just weight.
    pub fn from_mapped_selected(
        st: MmapedSafetensors,
        files: &[PathBuf],
        dev: &Device,
        selected: impl Fn(&str) -> bool,
    ) -> Result<std::result::Result<Self, MmapedSafetensors>> {
        let views: BTreeMap<_, _> = st.tensors().into_iter().collect();
        let mut bases = BTreeSet::new();
        for (name, view) in &views {
            if !selected(name) {
                continue;
            }
            if let Some(base) = name.strip_suffix(".comfy_quant") {
                bases.insert(base.to_owned());
            }
            if let Some(base) = name.strip_suffix(".weight") {
                if matches!(view.dtype(), Dtype::F8_E4M3 | Dtype::F8_E5M2)
                    || (matches!(view.dtype(), Dtype::I8 | Dtype::U8)
                        && views.contains_key(&format!("{base}.weight_scale")))
                {
                    bases.insert(base.to_owned());
                }
            }
        }
        let bare_fp8 = views.iter().any(|(key, view)| {
            selected(key)
                && matches!(view.dtype(), Dtype::F8_E4M3 | Dtype::F8_E5M2)
                && !matches!(
                    key.rsplit('.').next(),
                    Some("weight_s_rel" | "weight_scale" | "weight_scale_2" | "scales")
                )
        });
        if bases.is_empty() && !bare_fp8 {
            return Ok(Err(st));
        }
        let mut headers = Vec::new();
        let mut descriptors = BTreeMap::new();
        for (name, view) in &views {
            let base = name.rsplit_once('.').map(|(b, _)| b);
            let part = base.is_some_and(|b| bases.contains(b));
            let float8 = matches!(view.dtype(), Dtype::F8_E4M3 | Dtype::F8_E5M2);
            let companion = matches!(
                name.rsplit('.').next(),
                Some(
                    "comfy_quant"
                        | "weight_scale"
                        | "weight_scale_2"
                        | "input_scale"
                        | "scales"
                        | "biases"
                        | "scale_weight"
                        | "scale_input"
                        | "weight_s_rel"
                        | "weight_s_channel"
                        | "weight_codebook"
                )
            );
            if !(part && (companion || name.ends_with(".weight")))
                && !(float8 && !companion && selected(name))
                && name != "scaled_fp8"
            {
                continue;
            }
            if name.ends_with(".comfy_quant") {
                if view.data().len() > 65_536 {
                    candle_core::bail!("{name}: oversized checkpoint descriptor");
                }
                descriptors.insert(
                    name.clone(),
                    st::read_tensor_bytes(view, |bytes| bytes.to_vec()),
                );
            }
            headers.push(SafetensorsTensorHeader {
                name: name.clone(),
                dtype: gen_core::weightsmeta::Dtype::parse(&format!("{:?}", view.dtype()))
                    .ok_or_else(|| Error::Msg(format!("{name}: unsupported {:?}", view.dtype())))?,
                shape: view.shape().to_vec(),
                data_bytes: view.data().len() as u64,
            });
        }
        let mut layers = serde_json::Map::new();
        for path in files {
            if let Some(raw) =
                gen_core::safetensors_path_quantization_metadata(path).map_err(Error::wrap)?
            {
                gen_core::comfy_quant::parse_quantization_metadata(&raw).map_err(Error::wrap)?;
                let parsed: serde_json::Value = serde_json::from_str(&raw).map_err(Error::wrap)?;
                for (name, entry) in parsed["layers"]
                    .as_object()
                    .expect("validated layers")
                    .iter()
                {
                    if !bases.contains(name)
                        && !bases.iter().any(|b| {
                            b.strip_suffix(name.as_str())
                                .is_some_and(|prefix| prefix.ends_with('.'))
                        })
                    {
                        continue;
                    }
                    if layers
                        .insert(name.clone(), entry.clone())
                        .is_some_and(|previous| previous != *entry)
                    {
                        candle_core::bail!("conflicting shard descriptors for {name}");
                    }
                }
            }
        }
        let meta = (!layers.is_empty()).then(|| serde_json::json!({"layers":layers}).to_string());
        let definition = Arc::new(SourcePlan {
            headers,
            descriptors,
            metadata: meta,
            other_devices: Default::default(),
        });
        let reader = definition.open(Arc::new(st), dev)?;
        Ok(Ok(Self(reader, definition)))
    }
}

#[cfg(feature = "cuda")]
enum NativeOp {
    Fp8(candle_quant_kernels::fp8_linear::Fp8Linear),
    ConvRot(candle_quant_kernels::convrot_int8::ConvRotLinear),
}

#[cfg(feature = "cuda")]
struct SourceLinear {
    source: ProjectionSource,
    key: String,
    dtype: DType,
    op: NativeOp,
    format_id: &'static str,
    resident_bytes: u64,
    transpose: bool,
}
#[cfg(feature = "cuda")]
impl NativeLinearWeight for SourceLinear {
    fn forward(&self, x: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        match &self.op {
            NativeOp::ConvRot(op) => match bias {
                Some(b) => op.forward_with_bias(x, b),
                None => op.forward(x),
            },
            NativeOp::Fp8(op) => {
                let y = op.forward(x)?.to_dtype(x.dtype())?;
                match bias {
                    Some(b) => y.broadcast_add(&b.to_dtype(x.dtype())?),
                    None => Ok(y),
                }
            }
        }
    }
    fn compute_dtype(&self) -> DType {
        self.dtype
    }
    fn relocated(&self, dev: &Device) -> Result<candle_nn::Linear> {
        let source = self.source.on_device(dev)?;
        let shape = Some(self.dims());
        let op = source.native_oriented(&self.key, shape, self.dtype, dev, self.transpose)?;
        if let Some(op) = op {
            return Ok(candle_nn::Linear::from_native(op, None));
        }
        let dense = source.reader().read_dense_with_shape(
            &self.key,
            if self.transpose { None } else { shape },
            self.dtype,
            dev,
        )?;
        Ok(candle_nn::Linear::new(
            if self.transpose {
                dense.t()?.contiguous()?
            } else {
                dense
            },
            None,
        ))
    }
    fn format_id(&self) -> &'static str {
        self.format_id
    }
    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
    fn dense_weight(&self) -> Result<Tensor> {
        let t = self.source.0.dense_coefficients(
            &self.key,
            if self.transpose {
                None
            } else {
                Some(self.dims())
            },
            self.dtype,
            self.device(),
        )?;
        if self.transpose {
            t.t()?.contiguous()
        } else {
            Ok(t)
        }
    }
    fn dims(&self) -> (usize, usize) {
        match &self.op {
            NativeOp::Fp8(op) => op.dims(),
            NativeOp::ConvRot(op) => (op.descriptor().out_features, op.descriptor().in_features),
        }
    }
    fn device(&self) -> &Device {
        match &self.op {
            NativeOp::Fp8(op) => op.device(),
            NativeOp::ConvRot(op) => op.device(),
        }
    }
}

impl ProjectionSource {
    fn native_oriented(
        &self,
        name: &str,
        shape: Option<(usize, usize)>,
        dtype: DType,
        dev: &Device,
        transpose: bool,
    ) -> Result<Option<Arc<dyn NativeLinearWeight>>> {
        if !self.reader().device().same_device(dev) {
            return self
                .on_device(dev)?
                .native_oriented(name, shape, dtype, dev, transpose);
        }
        #[cfg(feature = "cuda")]
        {
            let convrot = if transpose {
                None
            } else {
                self.0.convrot_projection(name, None)?
            };
            let op = if let Some(base) = convrot {
                NativeOp::ConvRot(base)
            } else if let Some(base) = self
                .0
                .fp8_projection_oriented(name, None, shape, transpose)?
            {
                NativeOp::Fp8(base)
            } else {
                return Ok(None);
            };
            let layer = SourceLinear {
                source: self.clone(),
                key: name.to_owned(),
                dtype,
                op,
                transpose,
                format_id: self
                    .0
                    .planned(name)
                    .expect("constructed planned projection")
                    .codec_id,
                resident_bytes: self
                    .0
                    .materialized_bytes(name)
                    .expect("constructed projection recorded"),
            };
            if !layer.device().same_device(dev) {
                candle_core::bail!("{name}: source and requested execution device differ");
            }
            Ok(Some(Arc::new(layer)))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (name, shape, dtype, dev, transpose);
            Ok(None)
        }
    }
}

impl SimpleBackend for ProjectionSource {
    fn native_linear_weight(
        &self,
        key: &str,
        shape: Option<(usize, usize)>,
        dtype: DType,
        dev: &Device,
    ) -> Result<Option<Arc<dyn NativeLinearWeight>>> {
        self.native_oriented(key, shape, dtype, dev, false)
    }
    fn native_linear_weight_transposed(
        &self,
        key: &str,
        shape: (usize, usize),
        dtype: DType,
        dev: &Device,
    ) -> Result<Option<Arc<dyn NativeLinearWeight>>> {
        self.native_oriented(key, Some(shape), dtype, dev, true)
    }
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        if self.0.planned(name).is_some() {
            return self.on_device(dev)?.reader().read_dense(name, dtype, dev);
        }
        self.0.st.load(name, dev)?.to_dtype(dtype)
    }
    fn get(&self, shape: Shape, name: &str, _: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let t = if self.reader().planned(name).is_some() {
            let dims = match shape.dims() {
                [n, k] => Some((*n, *k)),
                _ => None,
            };
            self.on_device(dev)?
                .reader()
                .read_dense_with_shape(name, dims, dtype, dev)?
        } else {
            self.get_unchecked(name, dtype, dev)?
        };
        if t.shape() != &shape {
            return Err(Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: shape,
                got: t.shape().clone(),
            }
            .bt());
        }
        Ok(t)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.0.st.get(name).is_ok()
    }
}

/// Mapped weights for models that infer projection sizes from checkpoint keys.
pub enum MappedWeights {
    Checkpoint(ProjectionSource),
    Plain(MmapedSafetensors),
}

impl MappedWeights {
    pub fn new(st: MmapedSafetensors, paths: &[PathBuf], device: &Device) -> Result<Self> {
        Ok(match ProjectionSource::from_mapped(st, paths, device)? {
            Ok(source) => Self::Checkpoint(source),
            Err(plain) => Self::Plain(plain),
        })
    }
    pub fn mmap(&self) -> &MmapedSafetensors {
        match self {
            Self::Checkpoint(src) => src.mmap(),
            Self::Plain(st) => st,
        }
    }
    pub fn read(&self, key: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        match self {
            Self::Checkpoint(src) => src.get_unchecked(key, dtype, dev),
            Self::Plain(st) => st.load(key, dev)?.to_dtype(dtype),
        }
    }
    pub fn linear(
        &self,
        base: &str,
        bias: bool,
        dtype: DType,
        dev: &Device,
    ) -> Result<candle_nn::Linear> {
        let w = format!("{base}.weight");
        let b = bias
            .then(|| self.read(&format!("{base}.bias"), dtype, dev))
            .transpose()?;
        if let Self::Checkpoint(src) = self {
            if let Some(layer) = src.native_linear(&w, dtype, dev, b.clone())? {
                return Ok(layer);
            }
        }
        Ok(candle_nn::Linear::new(self.read(&w, dtype, dev)?, b))
    }
}

pub fn mmap_var_builder(
    files: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    // SAFETY: callers must keep the supplied checkpoint files unmodified while mapped.
    let st = unsafe { MmapedSafetensors::multi(files)? };
    let backend: Box<dyn SimpleBackend> = match ProjectionSource::from_mapped(st, files, device)? {
        Ok(source) => Box::new(source),
        Err(plain) => Box::new(plain),
    };
    Ok(VarBuilder::from_backend(backend, dtype, device.clone()))
}
