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

    /// Return the original mmap for a plain checkpoint. Detection inspects tensor headers only.
    pub fn from_mapped(
        st: MmapedSafetensors,
        files: &[PathBuf],
        dev: &Device,
    ) -> Result<std::result::Result<Self, MmapedSafetensors>> {
        let views: BTreeMap<_, _> = st.tensors().into_iter().collect();
        let mut bases = BTreeSet::new();
        for (name, view) in &views {
            if let Some(base) = name.strip_suffix(".comfy_quant") {
                bases.insert(base.to_owned());
            }
            if let Some(base) = name.strip_suffix(".weight") {
                if matches!(view.dtype(), Dtype::F8_E4M3 | Dtype::F8_E5M2)
                    || (view.dtype() == Dtype::I8
                        && views.contains_key(&format!("{base}.weight_scale")))
                {
                    bases.insert(base.to_owned());
                }
            }
        }
        if bases.is_empty() {
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
                && !(float8 && !companion)
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
    fn format_id(&self) -> &'static str {
        self.format_id
    }
    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
    fn dense_weight(&self) -> Result<Tensor> {
        self.source
            .0
            .read_dense(&self.key, self.dtype, self.device())
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

impl SimpleBackend for ProjectionSource {
    fn native_linear_weight(
        &self,
        name: &str,
        shape: Option<(usize, usize)>,
        dtype: DType,
        dev: &Device,
    ) -> Result<Option<Arc<dyn NativeLinearWeight>>> {
        if !self.reader().device().same_device(dev) {
            return self
                .on_device(dev)?
                .native_linear_weight(name, shape, dtype, dev);
        }
        #[cfg(feature = "cuda")]
        {
            let op = if let Some(base) = self.0.convrot_projection(name, None)? {
                NativeOp::ConvRot(base)
            } else if let Some(base) = self.0.fp8_projection_with_shape(name, None, shape)? {
                NativeOp::Fp8(base)
            } else {
                return Ok(None);
            };
            let layer = SourceLinear {
                source: self.clone(),
                key: name.to_owned(),
                dtype,
                op,
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
            let _ = (name, shape, dtype, dev);
            Ok(None)
        }
    }
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        if self.0.planned(name).is_some() {
            return self.0.read_dense(name, dtype, dev);
        }
        self.0.st.load(name, dev)?.to_dtype(dtype)
    }
    fn get(&self, shape: Shape, name: &str, _: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let t = self.get_unchecked(name, dtype, dev)?;
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
