//! Checkpoint-aware stock VarBuilder. One mapped source owns the original codes and companions.
//! Constructors asking for a Linear select the native GEMM; coefficient reads use the reference
//! codec. Plain tensors retain Candle's original lookup, shape and dtype behavior.
use crate::logical_weights::{baseline_codec_registry, CandleCodecResidency, LogicalWeightReader};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Error, Result, Shape, Tensor};
use candle_nn::linear::NativeLinearWeight;
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};
use gen_core::checkpoint_codec::LogicalKeyMapping;
use gen_core::checkpoint_codec::ProjectionSourceHeaders;
use safetensors::Dtype;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Default)]
struct SourceMapping {
    legacy_convrot: bool,
    transforms: BTreeMap<String, gen_core::checkpoint_codec::LogicalTransformDeclaration>,
}
impl LogicalKeyMapping for SourceMapping {
    fn legacy_convrot_geometry(&self) -> bool {
        self.legacy_convrot
    }
    fn mapping_id(&self) -> &'static str {
        "candle-constructor-v1"
    }
    fn logical_key(&self, key: &str) -> Option<String> {
        Some(key.to_owned())
    }
    fn logical_transform(
        &self,
        key: &str,
    ) -> Option<gen_core::checkpoint_codec::LogicalTransformDeclaration> {
        self.transforms.get(key).cloned()
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
    mapping: SourceMapping,
    files: Vec<PathBuf>,
    headers: ProjectionSourceHeaders,
    other_devices: std::sync::Mutex<Vec<Arc<LogicalWeightReader>>>,
}
impl SourcePlan {
    fn open(&self, st: Arc<MmapedSafetensors>, dev: &Device) -> Result<Arc<LogicalWeightReader>> {
        let policy = CandleCodecResidency::probe(dev);
        let plan = self
            .headers
            .compile(&self.mapping, baseline_codec_registry(), &policy)
            .map_err(Error::wrap)?;
        Ok(Arc::new(LogicalWeightReader::from_mapped(
            st,
            plan,
            dev,
            policy.native_execution_capability(),
            &self.files,
        )))
    }
}
impl ProjectionSource {
    pub fn selected_var_builder(
        self,
        keys: impl IntoIterator<Item = String>,
        dtype: DType,
        dev: &Device,
    ) -> VarBuilder<'static> {
        let names = keys
            .into_iter()
            .map(|key| (key.clone(), (key.clone(), SourceView::Whole(key))))
            .collect();
        VarBuilder::from_backend(
            Box::new(Remapped {
                source: MappedWeights::Checkpoint(self),
                names,
            }),
            dtype,
            dev.clone(),
        )
    }

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

    pub fn into_reader(self) -> (Arc<MmapedSafetensors>, Arc<LogicalWeightReader>) {
        (self.0.st.clone(), self.0)
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

    /// A streamed block keeps the original encoded operand on the host until its window opens.
    /// The destination policy selects the codec; an unsupported destination uses the caller's
    /// ordinary dense preparation. No execution receipt is published until the device binds it.
    pub fn host_fp8_linear(
        &self,
        key: &str,
        shape: Option<(usize, usize)>,
        dtype: DType,
        destination: &Device,
    ) -> Result<Option<Arc<dyn NativeLinearWeight>>> {
        #[cfg(feature = "cuda")]
        {
            let source = self.on_device(destination)?;
            if !source.reader().has_native_projection(key) {
                return Ok(None);
            }
            let row = source.reader().planned(key).expect("selected logical row");
            let dims = match (shape, row.shape.as_slice()) {
                (Some((n, k)), _) => (n, k),
                (None, [n, k]) if row.undeclared_padded_storage().is_none() => (*n, *k),
                _ => {
                    candle_core::bail!("{key}: host projection requires logical matrix dimensions")
                }
            };
            source
                .reader()
                .validate_matrix_shape(key, [dims.0, dims.1])?;
            let operand = source
                .reader()
                .prepare_operand(key, Some(dims), &Device::Cpu)?;
            let layer = SourceLinear {
                key: key.into(),
                dtype,
                format_id: row.codec_id,
                resident_bytes: operand.resident_bytes(),
                transpose: false,
                first_row: None,
                op: NativeOp::HostFp8 { operand, dims },
                source,
            };
            Ok(Some(Arc::new(layer)))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (key, shape, dtype, destination);
            Ok(None)
        }
    }

    /// Select a contiguous physical row range from a stack of equally shaped scalar-FP8 matrices.
    pub fn native_linear_slice(
        &self,
        key: &str,
        dev: &Device,
        dtype: DType,
        first_row: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Option<candle_nn::Linear>> {
        #[cfg(feature = "cuda")]
        {
            let source = self.on_device(dev)?;
            let Some((op, resident_bytes)) =
                source.0.fp8_matrix_slice(key, first_row, rows, cols)?
            else {
                return Ok(None);
            };
            let format_id = op.format_id();
            let layer = SourceLinear {
                source,
                key: key.to_owned(),
                dtype,
                op: NativeOp::Fp8(op),
                format_id,
                resident_bytes,
                transpose: false,
                first_row: Some(first_row),
            };
            Ok(Some(candle_nn::Linear::from_native(Arc::new(layer), None)))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (key, dev, dtype, first_row, rows, cols);
            Ok(None)
        }
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
        Self::from_mapped_for(st, files, dev, selected, SourceMapping::default())
    }
    fn from_mapped_for(
        st: MmapedSafetensors,
        files: &[PathBuf],
        dev: &Device,
        selected: impl Fn(&str) -> bool,
        mapping: SourceMapping,
    ) -> Result<std::result::Result<Self, MmapedSafetensors>> {
        // A plain checkpoint needs only the mmap's existing index. This is a conservative
        // fast path: only the portable selector resolves companion ownership, metadata and
        // physical key spelling across shards.
        let tensors = st.tensors();
        if !tensors.iter().any(|(name, view)| {
            (selected(name)
                && matches!(
                    view.dtype(),
                    Dtype::F8_E4M3 | Dtype::F8_E5M2 | Dtype::I8 | Dtype::U8
                ))
                || gen_core::checkpoint_codec::projection_companion_base(name).is_some()
        }) {
            return Ok(Err(st));
        }
        let tables = st.quantization_tables()?;
        let mut headers = BTreeMap::new();
        for (name, view) in tensors {
            let header = gen_core::weightsmeta::SafetensorsTensorHeader {
                name: name.clone(),
                dtype: gen_core::weightsmeta::Dtype::parse(&format!("{:?}", view.dtype()))
                    .ok_or_else(|| {
                        Error::Msg(format!(
                            "unsupported source dtype {:?} for {name}",
                            view.dtype()
                        ))
                    })?,
                shape: view.shape().to_vec(),
                data_bytes: view.data().len() as u64,
            };
            if headers.insert(name.clone(), header).is_some() {
                candle_core::bail!("duplicate tensor across checkpoint shards: {name}")
            }
        }
        let Some(headers) = ProjectionSourceHeaders::select(headers, selected, &tables, |names| {
            names
                .iter()
                .map(|&name| {
                    let view = st
                        .get(name)
                        .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
                    if view.data().len() > 65_536 {
                        return Err(format!("{name}: descriptor exceeds 65536 bytes").into());
                    }
                    Ok((name.to_owned(), view.data().to_vec()))
                })
                .collect()
        })
        .map_err(Error::wrap)?
        else {
            return Ok(Err(st));
        };
        let definition = Arc::new(SourcePlan {
            mapping,
            files: files.to_vec(),
            headers,
            other_devices: Default::default(),
        });
        let reader = definition.open(Arc::new(st), dev)?;
        Ok(Ok(Self(reader, definition)))
    }
}

#[cfg(feature = "cuda")]
enum NativeOp {
    HostFp8 {
        operand: crate::logical_weights::LogicalTensor,
        dims: (usize, usize),
    },
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
    first_row: Option<usize>,
}
#[cfg(feature = "cuda")]
impl SourceLinear {
    fn slice_coefficients(
        &self,
        source: &ProjectionSource,
        dev: &Device,
        first: usize,
    ) -> Result<Tensor> {
        let (rows, cols) = self.dims();
        source
            .reader()
            .dense_matrix_slice(&self.key, first, rows, cols, self.dtype, dev)
    }
}
#[cfg(feature = "cuda")]
impl NativeLinearWeight for SourceLinear {
    fn forward(&self, x: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        match &self.op {
            NativeOp::HostFp8 { .. } => candle_core::bail!(
                "{}: streamed host projection must enter its execution window",
                self.key
            ),
            NativeOp::ConvRot(op) => match bias {
                Some(b) => op.forward_with_bias(x, b),
                None => op.forward(x),
            },
            NativeOp::Fp8(op) => op.forward_with_bias(x, bias),
        }
    }
    fn checkpoint_bias_dtype(&self) -> Option<DType> {
        self.source.reader().projection_bias_dtype(&self.key)
    }
    fn compute_dtype(&self) -> DType {
        self.dtype
    }
    fn relocated(&self, dev: &Device) -> Result<candle_nn::Linear> {
        let source = self.source.on_device(dev)?;
        if let NativeOp::HostFp8 { operand, dims } = &self.op {
            if source.reader().has_native_projection(&self.key) {
                let op = source.reader().bind_fp8_operand(
                    &self.key,
                    operand,
                    None,
                    Some(*dims),
                    false,
                )?;
                let layer = SourceLinear {
                    op: NativeOp::Fp8(op),
                    source,
                    key: self.key.clone(),
                    dtype: self.dtype,
                    format_id: self.format_id,
                    resident_bytes: self.resident_bytes,
                    transpose: false,
                    first_row: None,
                };
                return Ok(candle_nn::Linear::from_native(Arc::new(layer), None));
            }
        }
        if let Some(first) = self.first_row {
            let (rows, cols) = self.dims();
            if let Some(layer) =
                source.native_linear_slice(&self.key, dev, self.dtype, first, rows, cols)?
            {
                return Ok(layer);
            }
            return Ok(candle_nn::Linear::new(
                self.slice_coefficients(&source, dev, first)?,
                None,
            ));
        }
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
    fn retained_coefficients(&self, t: &Tensor) -> Result<()> {
        self.source
            .reader()
            .record_coefficient_cache(&self.key, self.first_row, t)
    }
    fn dense_weight(&self) -> Result<Tensor> {
        if let Some(first) = self.first_row {
            return self.slice_coefficients(&self.source, self.device(), first);
        }
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
            NativeOp::HostFp8 { dims, .. } => *dims,
            NativeOp::Fp8(op) => op.dims(),
            NativeOp::ConvRot(op) => (op.descriptor().out_features, op.descriptor().in_features),
        }
    }
    fn device(&self) -> &Device {
        match &self.op {
            NativeOp::HostFp8 { .. } => &Device::Cpu,
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
                first_row: None,
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
    fn tensor_info(&self, name: &str) -> Result<Option<candle_nn::var_builder::TensorInfo>> {
        if let Some(row) = self.reader().planned(name) {
            Ok(Some(candle_nn::var_builder::TensorInfo {
                shape: row.shape.as_slice().into(),
                padded_shape: matches!(
                    row.codec,
                    gen_core::checkpoint_codec::TensorCodecSpec::Mxfp8 {
                        logical_shape_declared: false,
                        ..
                    }
                )
                .then_some(gen_core::mxfp8_padded_shape),
                source_packed: row.residency.mode
                    == gen_core::checkpoint_codec::ResidencyMode::Packed,
            }))
        } else {
            SimpleBackend::tensor_info(self.mmap(), name)
        }
    }
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
    fn native_linear_weight_flattened(
        &self,
        name: &str,
        shape: &[usize],
        dtype: DType,
        dev: &Device,
    ) -> Result<Option<Arc<dyn NativeLinearWeight>>> {
        let Some(row) = self.reader().planned(name) else {
            return Ok(None);
        };
        if shape.len() < 2 {
            candle_core::bail!(
                "{name}: expected patch kernel {shape:?}, got {:?}",
                row.shape
            );
        }
        let cols = shape[1..]
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| Error::Msg(format!("{name}: contraction overflow")))?;
        if row.shape.len() == 2 {
            self.reader()
                .validate_matrix_shape(name, [shape[0], cols])?;
            return self.native_linear_weight(name, Some((shape[0], cols)), dtype, dev);
        }
        if row.shape != shape {
            candle_core::bail!(
                "{name}: expected patch kernel {shape:?}, got {:?}",
                row.shape
            );
        }
        Ok(self
            .native_linear_slice(name, dev, dtype, 0, shape[0], cols)?
            .and_then(|layer| layer.native_weight()))
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
        self.linear_shaped(base, bias, dtype, dev, None)
    }

    /// Bind an explicitly shaped projection, including a flattened patch projection.
    /// A checkpoint may store the original spatial kernel or its canonical 2-D matrix.
    pub fn linear_shaped(
        &self,
        base: &str,
        bias: bool,
        dtype: DType,
        dev: &Device,
        shape: Option<&[usize]>,
    ) -> Result<candle_nn::Linear> {
        let key = format!("{base}.weight");
        let bias_dtype = match self {
            Self::Checkpoint(src) => src.reader().projection_bias_dtype(&key).unwrap_or(dtype),
            Self::Plain(_) => dtype,
        };
        let bias = bias
            .then(|| self.read(&format!("{base}.bias"), bias_dtype, dev))
            .transpose()?;
        let dims = shape
            .map(|shape| {
                let [n, tail @ ..] = shape else {
                    candle_core::bail!("{key}: expected a projection shape")
                };
                if tail.is_empty() {
                    candle_core::bail!("{key}: missing contraction dimensions");
                }
                let k = tail
                    .iter()
                    .try_fold(1usize, |k, &d| k.checked_mul(d))
                    .ok_or_else(|| Error::Msg(format!("{key}: contraction overflow")))?;
                Ok((*n, k))
            })
            .transpose()?;
        if let Self::Checkpoint(src) = self {
            if let (Some((n, k)), Some(row)) = (dims, src.reader().planned(&key)) {
                if row.shape.len() == 2 {
                    src.reader().validate_matrix_shape(&key, [n, k])?;
                }
            }
            let op = match shape {
                Some(shape) if shape.len() > 2 => {
                    src.native_linear_weight_flattened(&key, shape, dtype, dev)?
                }
                _ => src.native_linear_weight(&key, dims, dtype, dev)?,
            };
            if let Some(op) = op {
                return Ok(candle_nn::Linear::from_native(op, bias));
            }
            if let Some(row) = src.reader().planned(&key) {
                if row.shape.len() == 2 {
                    let weight = src.reader().read_dense_with_shape(&key, dims, dtype, dev)?;
                    if dims.is_some_and(|(n, k)| weight.dims() != [n, k]) {
                        candle_core::bail!("{key}: expected {dims:?}, got {:?}", weight.dims());
                    }
                    return Ok(candle_nn::Linear::new(weight, bias));
                }
            }
        }
        let mut weight = self.read(&key, dtype, dev)?;
        if let (Some(shape), Some((n, k))) = (shape, dims) {
            if weight.dims() != shape && weight.dims() != [n, k] {
                candle_core::bail!("{key}: expected {shape:?}, got {:?}", weight.dims());
            }
            weight = weight.reshape((n, k))?;
        }
        Ok(candle_nn::Linear::new(weight, bias))
    }
}

/// One provider-declared matrix view. Names and geometry are the provider's responsibility.
/// Source interpretation, packed row slicing and residency are the shared reader's.
#[derive(Debug, Clone)]
pub enum SourceView {
    Whole(String),
    Transposed(String),
    Reshape {
        name: String,
        shape: Vec<usize>,
    },
    Rows {
        name: String,
        first: usize,
        rows: usize,
    },
    Matrix {
        name: String,
        rows: usize,
        cols: usize,
    },
}
impl SourceView {
    fn name(&self) -> &str {
        match self {
            Self::Whole(name)
            | Self::Transposed(name)
            | Self::Reshape { name, .. }
            | Self::Rows { name, .. }
            | Self::Matrix { name, .. } => name,
        }
    }
}
struct Remapped {
    source: MappedWeights,
    names: BTreeMap<String, (String, SourceView)>,
}
impl Remapped {
    fn lookup(&self, name: &str) -> Result<&(String, SourceView)> {
        self.names.get(name).ok_or_else(|| {
            Error::CannotFindTensor {
                path: name.to_owned(),
            }
            .bt()
        })
    }
    fn logical<'a>(source: &ProjectionSource, physical: &'a str, name: &'a str) -> &'a str {
        if source
            .reader()
            .planned(name)
            .is_some_and(|r| r.physical_key == physical)
        {
            name
        } else {
            physical
        }
    }
    fn read(
        &self,
        name: &str,
        shape: Option<&Shape>,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        let (physical, view) = self.lookup(name)?;
        if let MappedWeights::Checkpoint(src) = &self.source {
            let logical = Self::logical(src, physical, name);
            if src.reader().planned(logical).is_some() {
                let requested = match shape.map(Shape::dims) {
                    Some(&[rows, cols]) => Some((rows, cols)),
                    _ => None,
                };
                let source_dims = match view {
                    SourceView::Whole(_) | SourceView::Rows { .. } => requested,
                    SourceView::Transposed(_) => requested.map(|(n, k)| (k, n)),
                    SourceView::Matrix { rows, cols, .. } => Some((*rows, *cols)),
                    SourceView::Reshape { .. } => None,
                };
                let t = src.on_device(dev)?.reader().read_dense_with_shape(
                    logical,
                    source_dims,
                    dtype,
                    dev,
                )?;
                return match view {
                    SourceView::Matrix { rows, cols, .. } => t.reshape((*rows, *cols)),
                    SourceView::Transposed(_) => t.t()?.contiguous(),
                    SourceView::Reshape { shape, .. } => t.reshape(shape.as_slice()),
                    _ => Ok(t),
                };
            }
        }
        let t = self.source.read(physical, dtype, dev)?;
        match view {
            SourceView::Whole(_) => Ok(t),
            SourceView::Rows { first, rows, .. } => t.narrow(0, *first, *rows)?.contiguous(),
            SourceView::Matrix { rows, cols, .. } => t.reshape((*rows, *cols)),
            SourceView::Transposed(_) => t.t()?.contiguous(),
            SourceView::Reshape { shape, .. } => t.reshape(shape.as_slice()),
        }
    }
}
impl SimpleBackend for Remapped {
    fn tensor_info(&self, name: &str) -> Result<Option<candle_nn::var_builder::TensorInfo>> {
        let Some((physical, view)) = self.names.get(name) else {
            return Ok(None);
        };
        let source = match &self.source {
            MappedWeights::Checkpoint(src) => {
                src.tensor_info(Self::logical(src, physical, name))?
            }
            MappedWeights::Plain(st) => SimpleBackend::tensor_info(st, physical)?,
        };
        let Some(source) = source else {
            return Ok(None);
        };
        let shape = match view {
            SourceView::Whole(_) => source.shape,
            SourceView::Matrix { rows, cols, .. } => (*rows, *cols).into(),
            SourceView::Reshape { shape, .. } => shape.as_slice().into(),
            SourceView::Rows { rows, .. } => {
                let mut dims = source.shape.dims().to_vec();
                let first = dims
                    .first_mut()
                    .ok_or_else(|| Error::Msg(format!("{name}: row view of scalar")))?;
                *first = *rows;
                dims.into()
            }
            SourceView::Transposed(_) => {
                let (rows, cols) = source.shape.dims2()?;
                (cols, rows).into()
            }
        };
        Ok(Some(candle_nn::var_builder::TensorInfo {
            shape,
            source_packed: source.source_packed,
            padded_shape: matches!(view, SourceView::Whole(_) | SourceView::Transposed(_))
                .then_some(source.padded_shape)
                .flatten(),
        }))
    }
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        self.read(name, None, dtype, dev)
    }
    fn get(&self, shape: Shape, name: &str, _: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let t = self.read(name, Some(&shape), dtype, dev)?;
        if t.shape() != &shape {
            return Err(Error::UnexpectedShape {
                msg: format!("remapped weight {name}"),
                expected: shape,
                got: t.shape().clone(),
            }
            .bt());
        }
        Ok(t)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.names.contains_key(name)
    }
    fn native_linear_weight(
        &self,
        name: &str,
        shape: Option<(usize, usize)>,
        dtype: DType,
        dev: &Device,
    ) -> Result<Option<Arc<dyn NativeLinearWeight>>> {
        let Some((physical, view)) = self.names.get(name) else {
            return Ok(None);
        };
        let MappedWeights::Checkpoint(src) = &self.source else {
            return Ok(None);
        };
        if matches!(view, SourceView::Transposed(_)) {
            let stored = src.mmap().get(physical)?;
            let [cols, rows] = stored.shape() else {
                candle_core::bail!("{name}: transposed projection requires a matrix");
            };
            let dims = shape.unwrap_or((*rows, *cols));
            if let Some(expected) = shape {
                if src.reader().planned(physical).is_some() {
                    src.reader()
                        .validate_matrix_shape(physical, [expected.1, expected.0])?;
                } else if expected != (*rows, *cols) {
                    candle_core::bail!("{name}: transposed projection has shape [{rows},{cols}], expected {expected:?}");
                }
            }
            return src.native_linear_weight_transposed(physical, dims, dtype, dev);
        }
        if matches!(view, SourceView::Reshape { .. }) {
            // Matrix contractions use SourceView::Matrix, which declares the exact geometry.
            return Ok(None);
        }
        if let SourceView::Matrix { rows, cols, .. } = view {
            if shape.is_some_and(|s| s != (*rows, *cols)) {
                candle_core::bail!(
                    "{name}: requested projection shape disagrees with checkpoint matrix"
                );
            }
            if src.mmap().get(physical)?.shape().len() != 2 {
                return Ok(src
                    .native_linear_slice(physical, dev, dtype, 0, *rows, *cols)?
                    .and_then(|layer| layer.native_weight()));
            }
        }
        src.native_linear_weight(Self::logical(src, physical, name), shape, dtype, dev)
    }
}

/// Bind a catalog's namespace/shape conversion to an existing mmap, retaining packed matrix operands.
pub fn remapped_var_builder(
    st: MmapedSafetensors,
    files: &[PathBuf],
    dtype: DType,
    dev: &Device,
    views: BTreeMap<String, Vec<SourceView>>,
) -> Result<VarBuilder<'static>> {
    remapped_var_builder_with_source(st, files, dtype, dev, views).map(|(builder, _)| builder)
}

/// Also report whether the shared planner bound an encoded source in the selected namespace.
pub fn remapped_var_builder_with_source(
    st: MmapedSafetensors,
    files: &[PathBuf],
    dtype: DType,
    dev: &Device,
    views: BTreeMap<String, Vec<SourceView>>,
) -> Result<(VarBuilder<'static>, bool)> {
    use gen_core::checkpoint_codec::{
        LogicalTransformDeclaration, LogicalTransformOutput, RowRange,
    };
    let mut mapping = SourceMapping::default();
    let mut names = BTreeMap::new();
    for (physical, outputs) in &views {
        let mut rows = Vec::new();
        for view in outputs {
            if names
                .insert(view.name().to_owned(), (physical.clone(), view.clone()))
                .is_some()
            {
                candle_core::bail!("multiple checkpoint tensors map to {}", view.name());
            }
            if let SourceView::Rows {
                name,
                first,
                rows: n,
            } = view
            {
                rows.push(LogicalTransformOutput {
                    logical_key: name.clone(),
                    rows: Some(RowRange::new(*first, *n)),
                    half_swap: false,
                });
            }
        }
        if !rows.is_empty() {
            mapping.transforms.insert(
                physical.to_owned(),
                LogicalTransformDeclaration {
                    source_logical_shape: Some(st.get(physical)?.shape().to_vec()),
                    outputs: rows,
                },
            );
        }
    }
    let source = match ProjectionSource::from_mapped_for(
        st,
        files,
        dev,
        |k| views.contains_key(k),
        mapping,
    )? {
        Ok(src) => MappedWeights::Checkpoint(src),
        Err(st) => MappedWeights::Plain(st),
    };
    let encoded = matches!(&source, MappedWeights::Checkpoint(_));
    Ok((
        VarBuilder::from_backend(Box::new(Remapped { source, names }), dtype, dev.clone()),
        encoded,
    ))
}

pub fn mmap_var_builder(
    files: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    mmap_var_builder_with_geometry(files, dtype, device, false)
}
pub fn mmap_var_builder_with_geometry(
    files: &[PathBuf],
    dtype: DType,
    device: &Device,
    legacy_convrot: bool,
) -> Result<VarBuilder<'static>> {
    // SAFETY: callers must keep the supplied checkpoint files unmodified while mapped.
    let st = unsafe { MmapedSafetensors::multi(files)? };
    let backend: Box<dyn SimpleBackend> = match ProjectionSource::from_mapped_for(
        st,
        files,
        device,
        |_| true,
        SourceMapping {
            legacy_convrot,
            ..Default::default()
        },
    )? {
        Ok(source) => Box::new(source),
        Err(plain) => Box::new(plain),
    };
    Ok(VarBuilder::from_backend(backend, dtype, device.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn file_builder_recognizes_metadata_only_convrot_on_underscore_projection() {
        use safetensors::tensor::{serialize_to_file, TensorView};

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("mixed.safetensors");
        let codes = [0u8; 16 * 64];
        let scales = 1f32.to_le_bytes().repeat(16);
        let dense = 3.5f32.to_le_bytes();
        let key = "attn.in_proj_weight";
        let spec = serde_json::json!({"layers": {"attn.in_proj": {
            "format": "int8_tensorwise", "convrot": true, "per_row": true, "convrot_groupsize": 64
        }}});
        serialize_to_file(
            [
                (
                    key,
                    TensorView::new(Dtype::I8, vec![16, 64], &codes).unwrap(),
                ),
                (
                    "attn.in_proj.weight_scale",
                    TensorView::new(Dtype::F32, vec![16, 1], &scales).unwrap(),
                ),
                (
                    "dense.weight",
                    TensorView::new(Dtype::F32, vec![1, 1], &dense).unwrap(),
                ),
            ],
            Some(std::collections::HashMap::from([(
                "_quantization_metadata".to_owned(),
                spec.to_string(),
            )])),
            &file,
        )
        .unwrap();
        let files = [file];
        // A component may select just the unrelated plain parameter in this same mapped file.
        // SAFETY: this test owns the file and keeps it unchanged while mapped.
        let st = unsafe { MmapedSafetensors::multi(&files).unwrap() };
        let plain = ProjectionSource::from_mapped_selected(st, &files, &Device::Cpu, |key| {
            key == "dense.weight"
        })
        .unwrap()
        .err()
        .expect("unrelated dense selection must stay plain");
        assert_eq!(
            plain
                .load("dense.weight", &Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            [[3.5]]
        );
        // The full normal constructor must dispatch to the codec planner and refuse the CPU's
        // missing native integer operator before a plain Candle reader sees the original I8 codes.
        let error = mmap_var_builder(&files, DType::F32, &Device::Cpu)
            .err()
            .expect("ConvRot cannot execute on the CPU")
            .to_string();
        assert!(
            error.contains(key) && error.contains("ConvRot INT8 requires native"),
            "{error}"
        );
    }

    #[test]
    fn failed_convrot_projection_does_not_publish_packed_execution() {
        use safetensors::tensor::TensorView;
        let codes = [0u8; 16 * 64];
        let scales = 1f32.to_le_bytes().repeat(16);
        let bias_value = 1.0f32 + 1.0 / 4096.0;
        let bias = bias_value.to_le_bytes().repeat(16);
        let desc = br#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":64}"#;
        let bytes = safetensors::tensor::serialize(
            [
                (
                    "p.weight",
                    TensorView::new(Dtype::I8, vec![16, 64], &codes).unwrap(),
                ),
                (
                    "p.bias",
                    TensorView::new(Dtype::F32, vec![16], &bias).unwrap(),
                ),
                (
                    "p.weight_scale",
                    TensorView::new(Dtype::F32, vec![16, 1], &scales).unwrap(),
                ),
                (
                    "p.comfy_quant",
                    TensorView::new(Dtype::U8, vec![desc.len()], desc).unwrap(),
                ),
            ],
            None,
        )
        .unwrap();
        let scope = gen_core::CheckpointReceiptScope::default();
        scope.with(|| {
            let st = MmapedSafetensors::from_buffers([Ok(bytes)]).unwrap();
            let policy = CandleCodecResidency::from_cuda_compute_cap((12, 0));
            let headers = st
                .tensors()
                .into_iter()
                .map(
                    |(name, view)| gen_core::weightsmeta::SafetensorsTensorHeader {
                        name,
                        dtype: gen_core::weightsmeta::Dtype::parse(&format!("{:?}", view.dtype()))
                            .unwrap(),
                        shape: view.shape().to_vec(),
                        data_bytes: view.data().len() as u64,
                    },
                )
                .collect::<Vec<_>>();
            let descriptors = BTreeMap::from([("p.comfy_quant".into(), desc.to_vec())]);
            let plan = gen_core::checkpoint_codec::plan_projection_source_headers(
                &headers,
                &descriptors,
                |_| true,
                &[],
                baseline_codec_registry(),
                &policy,
            )
            .unwrap()
            .unwrap();
            let reader = LogicalWeightReader::from_mapped(
                Arc::new(st),
                plan,
                &Device::Cpu,
                policy.native_execution_capability(),
                &[],
            );
            for dtype in [DType::BF16, DType::F16, DType::F32] {
                assert_eq!(
                    reader.projection_bias_dtype("p.weight").unwrap_or(dtype),
                    DType::F32
                );
                let bias = reader
                    .retained_projection_bias("p.weight", "p.bias", &Device::Cpu)
                    .unwrap()
                    .unwrap();
                assert_eq!(bias.dtype(), DType::F32);
                assert_eq!(bias.to_vec1::<f32>().unwrap(), [bias_value; 16]);
            }
            let bad_bias = Tensor::zeros(15, DType::F32, &Device::Cpu).unwrap();
            let err = reader
                .convrot_projection("p.weight", Some(bad_bias))
                .unwrap_err()
                .to_string();
            assert!(err.contains("invalid ConvRot bias"), "{err}");
            assert!(scope.try_snapshots().unwrap().is_empty());
        });
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn prepared_fp8_window_keeps_source_operands_until_device_binding() {
        use safetensors::tensor::TensorView;
        // Simulate only the destination policy. Every tensor actually allocated by this test
        // is on CPU; no CUDA handle, runtime kernel or module forward is invoked.
        for (encoding, format) in [
            (
                Dtype::F8_E4M3,
                gen_core::checkpoint_codec::FP8_E4M3_SCALAR_CODEC.codec_id,
            ),
            (
                Dtype::F8_E5M2,
                gen_core::checkpoint_codec::FP8_E5M2_SCALAR_CODEC.codec_id,
            ),
        ] {
            let codes = [0x3c; 16 * 32];
            let weight_scale = 0.5_f32.to_le_bytes();
            let input_scale = 2.0_f32.to_le_bytes();
            let bytes = safetensors::tensor::serialize(
                [
                    (
                        "proj.weight",
                        TensorView::new(encoding, vec![16, 32], &codes).unwrap(),
                    ),
                    (
                        "proj.weight_scale",
                        TensorView::new(Dtype::F32, vec![], &weight_scale).unwrap(),
                    ),
                    (
                        "proj.input_scale",
                        TensorView::new(Dtype::F32, vec![], &input_scale).unwrap(),
                    ),
                ],
                None,
            )
            .unwrap();
            let scope = gen_core::CheckpointReceiptScope::default();
            scope.with(|| {
                let st = MmapedSafetensors::from_buffers([Ok(bytes)]).unwrap();
                let cpu = ProjectionSource::from_mapped(st, &[], &Device::Cpu)
                    .unwrap()
                    .ok()
                    .unwrap();
                let policy = CandleCodecResidency::from_cuda_compute_cap((12, 0));
                let plan = cpu
                    .1
                    .headers
                    .compile(&cpu.1.mapping, baseline_codec_registry(), &policy)
                    .unwrap();
                let target = ProjectionSource(
                    Arc::new(LogicalWeightReader::from_mapped(
                        cpu.0.st.clone(),
                        plan,
                        &Device::Cpu,
                        policy.native_execution_capability(),
                        &[],
                    )),
                    cpu.1,
                );
                assert!(target
                    .host_fp8_linear("proj.weight", Some((32, 16)), DType::F32, &Device::Cpu)
                    .is_err());
                let prepared = target
                    .host_fp8_linear("proj.weight", Some((16, 32)), DType::F32, &Device::Cpu)
                    .unwrap()
                    .unwrap();
                assert_eq!(prepared.format_id(), format);
                assert_eq!(prepared.resident_bytes(), 16 * 32 + 8);
                assert_eq!(prepared.dims(), (16, 32));
                assert_eq!(prepared.compute_dtype(), DType::F32);
                let raw = target
                    .reader()
                    .prepare_operand("proj.weight", Some((16, 32)), &Device::Cpu)
                    .unwrap();
                let crate::logical_weights::LogicalTensor::PackedScalarFp8 {
                    codes,
                    weight_scale,
                    input_scale,
                    ..
                } = raw
                else {
                    panic!("expected original FP8 operand")
                };
                assert_eq!(weight_scale, 0.5);
                assert_eq!(input_scale, Some(2.0));
                assert_eq!(codes.device().location(), Device::Cpu.location());
                assert_eq!(
                    codes.dtype(),
                    if encoding == Dtype::F8_E5M2 {
                        DType::U8
                    } else {
                        DType::F8E4M3
                    }
                );
                assert!(scope.try_snapshots().unwrap().is_empty());
            });
        }
    }

    #[test]
    fn remapped_dense_mxfp8_uses_the_callers_unpadded_dimensions() {
        use safetensors::tensor::TensorView;
        let codes: Vec<_> = (0..32 * 32)
            .map(|i| {
                if i / 32 < 3 && i % 32 < 5 {
                    0x38 + (i / 32) as u8
                } else {
                    0x7e
                }
            })
            .collect();
        let scales = [127; 128 * 4];
        let descriptor = br#"{"format":"mxfp8"}"#;
        let bytes = safetensors::tensor::serialize(
            [
                (
                    "old.weight",
                    TensorView::new(Dtype::F8_E4M3, vec![32, 32], &codes).unwrap(),
                ),
                (
                    "old.weight_scale",
                    TensorView::new(Dtype::U8, vec![128, 4], &scales).unwrap(),
                ),
                (
                    "old.comfy_quant",
                    TensorView::new(Dtype::U8, vec![descriptor.len()], descriptor).unwrap(),
                ),
            ],
            None,
        )
        .unwrap();
        for (view, shape, expected) in [
            (
                SourceView::Whole("new.weight".into()),
                (3, 5),
                vec![vec![1.; 5], vec![1.125; 5], vec![1.25; 5]],
            ),
            (
                SourceView::Transposed("new.weight".into()),
                (5, 3),
                vec![vec![1., 1.125, 1.25]; 5],
            ),
        ] {
            let st = MmapedSafetensors::from_buffers([Ok(bytes.clone())]).unwrap();
            let scope = gen_core::CheckpointReceiptScope::default();
            let vb = scope
                .with(|| {
                    remapped_var_builder(
                        st,
                        &[],
                        DType::F32,
                        &Device::Cpu,
                        BTreeMap::from([("old.weight".into(), vec![view])]),
                    )
                })
                .unwrap();
            assert!(vb.get_unchecked("new.weight").is_err());
            assert!(vb.get((63, 66), "new.weight").is_err());
            assert!(scope.try_snapshots().unwrap().is_empty());
            let layer = candle_nn::linear_no_bias(shape.1, shape.0, vb.pp("new")).unwrap();
            let facts = scope.try_snapshots().unwrap();
            assert_eq!(facts.len(), 1);
            assert_eq!(facts[0].resident_bytes(), 3 * 5 * 4);
            assert_eq!(
                layer.try_weight().unwrap().to_vec2::<f32>().unwrap(),
                expected
            );
        }
    }
}
