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
    transforms: BTreeMap<String, gen_core::checkpoint_codec::LogicalTransformDeclaration>,
}
impl LogicalKeyMapping for SourceMapping {
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
        // The common dense path needs only the mmap's existing index. Source schemas and
        // companion/metadata scope are validated once by the portable shard planner.
        if !st.tensors().into_iter().any(|(k, v)| {
            (selected(&k) && matches!(v.dtype(), Dtype::F8_E4M3 | Dtype::F8_E5M2))
                || k.ends_with(".comfy_quant")
                || (k.ends_with(".weight_scale")
                    && st
                        .get(&format!("{}.weight", k.trim_end_matches(".weight_scale")))
                        .is_ok())
        }) {
            return Ok(Err(st));
        }
        let mut tables = Vec::new();
        for path in files {
            if let Some(table) =
                gen_core::safetensors_path_quantization_metadata(path).map_err(Error::wrap)?
            {
                tables.push(table);
            }
        }
        let Some(headers) =
            ProjectionSourceHeaders::read(files, selected, &tables).map_err(Error::wrap)?
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
            .dense_coefficients(&self.key, None, self.dtype, dev)?
            .reshape(((), cols))?
            .narrow(0, first, rows)?
            .force_contiguous()
    }
}
#[cfg(feature = "cuda")]
impl NativeLinearWeight for SourceLinear {
    fn forward(&self, x: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        match &self.op {
            NativeOp::ConvRot(op) => match bias {
                Some(b) => op.forward_with_bias(x, b),
                None => op.forward(x),
            },
            NativeOp::Fp8(op) => op.forward_with_bias(x, bias),
        }
    }
    fn compute_dtype(&self) -> DType {
        self.dtype
    }
    fn relocated(&self, dev: &Device) -> Result<candle_nn::Linear> {
        let source = self.source.on_device(dev)?;
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
            if self
                .reader()
                .planned(name)
                .is_some_and(|r| matches!(r.codec, gen_core::TensorCodecSpec::RowwiseFp8 { .. }))
                && (dtype != DType::BF16 || transpose)
            {
                return Ok(None);
            }
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
        if row.shape == [shape[0], cols] {
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

/// One provider-declared matrix view. Names and geometry are the provider's responsibility.
/// Source interpretation, packed row slicing and residency are the shared reader's.
#[derive(Debug, Clone)]
pub enum SourceView {
    Whole(String),
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
            Self::Whole(name) | Self::Rows { name, .. } | Self::Matrix { name, .. } => name,
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
}
impl SimpleBackend for Remapped {
    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> Result<Tensor> {
        let (physical, view) = self.lookup(name)?;
        if let MappedWeights::Checkpoint(src) = &self.source {
            let logical = Self::logical(src, physical, name);
            if src.reader().planned(logical).is_some() {
                let t = src
                    .on_device(dev)?
                    .reader()
                    .read_dense(logical, dtype, dev)?;
                return match view {
                    SourceView::Matrix { rows, cols, .. } => t.reshape((*rows, *cols)),
                    _ => Ok(t),
                };
            }
        }
        let t = self.source.read(physical, dtype, dev)?;
        match view {
            SourceView::Whole(_) => Ok(t),
            SourceView::Rows { first, rows, .. } => t.narrow(0, *first, *rows)?.contiguous(),
            SourceView::Matrix { rows, cols, .. } => t.reshape((*rows, *cols)),
        }
    }
    fn get(&self, shape: Shape, name: &str, _: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let t = self.get_unchecked(name, dtype, dev)?;
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
    Ok(VarBuilder::from_backend(
        Box::new(Remapped { source, names }),
        dtype,
        dev.clone(),
    ))
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
