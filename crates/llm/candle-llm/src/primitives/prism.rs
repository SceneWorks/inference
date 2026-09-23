//! Compact Prism/Bonsai affine-2 and GGUF ternary operators.
//!
//! The packed checkpoint remains the resident weight. CPU forwards decode one bounded row at a
//! time; CUDA forwards dispatch a packed matrix-vector/matrix kernel which decodes inside the dot
//! product. Neither path materializes a full dense model weight.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use core_llm::{
    apply_hadamard_forward_in_place, apply_hadamard_inverse_in_place, GdnLayout,
    PrismHadamardMetadata, PrismPackedKind, PrismPackedMatrixRef, PrismTransformRole,
    PRISM_GROUP_SIZE,
};

use crate::error::{Error, Result};

fn prism_err(context: &str, error: impl std::fmt::Display) -> Error {
    Error::Config(format!("Prism {context}: {error}"))
}

/// GGUF row gather applied before a packed projection is exposed to the Qwen decoder.
///
/// Prism's GGUF stores GDN value heads as `[repetition, key-group, unit]`; Qwen's decoder consumes
/// `[key-group, repetition, unit]`. `prefix` leaves q/k rows ahead of the value rows untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnRowMap {
    pub prefix: usize,
    pub groups: usize,
    pub repetitions: usize,
    pub unit: usize,
}

impl GdnRowMap {
    pub(crate) fn source_row(self, logical_row: usize) -> Result<usize> {
        if logical_row < self.prefix {
            return Ok(logical_row);
        }
        let logical = logical_row - self.prefix;
        let span = self
            .groups
            .checked_mul(self.repetitions)
            .and_then(|v| v.checked_mul(self.unit))
            .ok_or_else(|| Error::Config("Prism GDN row-map overflow".into()))?;
        if self.groups == 0 || self.repetitions == 0 || self.unit == 0 || logical >= span {
            return Err(Error::Config(format!(
                "Prism GDN row {logical_row} outside prefix {} + span {span}",
                self.prefix
            )));
        }
        let group = logical / (self.repetitions * self.unit);
        let rem = logical % (self.repetitions * self.unit);
        let repetition = rem / self.unit;
        let lane = rem % self.unit;
        Ok(self.prefix + (repetition * self.groups + group) * self.unit + lane)
    }

    pub(crate) fn validate(self, rows: usize) -> Result<()> {
        let span = self
            .groups
            .checked_mul(self.repetitions)
            .and_then(|v| v.checked_mul(self.unit))
            .ok_or_else(|| Error::Config("Prism GDN row-map overflow".into()))?;
        if self.prefix.checked_add(span) != Some(rows) {
            return Err(Error::Config(format!(
                "Prism GDN row-map geometry {} + {span} != {rows}",
                self.prefix
            )));
        }
        Ok(())
    }
}

#[derive(Clone)]
#[allow(dead_code)] // Device tensors are consumed only in `cfg(feature = "cuda")` builds.
enum PackedStorage {
    /// MLX affine 2-bit: 16 two-bit codes per u32 plus one scale per group. Published biases are
    /// validated as `-scale` at load and are therefore not retained separately.
    MlxAffine2 {
        words: Option<Tensor>,
        scales: Option<Tensor>,
        host_words: Option<Arc<[u32]>>,
        host_scales: Option<Arc<[f32]>>,
    },
    /// Native GGUF PQ2_0/PTQ1_0 blocks, retained byte-for-byte.
    Gguf {
        kind: PrismPackedKind,
        bytes: Option<Tensor>,
        host_bytes: Option<Arc<[u8]>>,
    },
}

#[derive(Clone, Copy)]
struct PackedGeometry {
    rows: usize,
    input_width: usize,
    row_map: Option<GdnRowMap>,
    gdn: Option<GdnLayout>,
}

/// One compact resident `[output_rows, input_width]` matrix.
#[derive(Clone)]
pub struct PrismPackedWeight {
    name: String,
    rows: usize,
    input_width: usize,
    storage: PackedStorage,
    role: PrismTransformRole,
    signs: Option<Arc<[i8]>>,
    block_size: usize,
    gdn: Option<GdnLayout>,
    row_map: Option<GdnRowMap>,
    device: Device,
}

impl std::fmt::Debug for PrismPackedWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrismPackedWeight")
            .field("name", &self.name)
            .field("rows", &self.rows)
            .field("input_width", &self.input_width)
            .field("role", &self.role)
            .field("block_size", &self.block_size)
            .field("gdn", &self.gdn)
            .field("row_map", &self.row_map)
            .finish_non_exhaustive()
    }
}

impl PrismPackedWeight {
    #[allow(clippy::too_many_arguments)]
    pub fn from_mlx_affine2(
        name: impl Into<String>,
        words: Tensor,
        scales: Tensor,
        biases: &Tensor,
        metadata: &PrismHadamardMetadata,
        row_map: Option<GdnRowMap>,
        gdn: Option<GdnLayout>,
    ) -> Result<Self> {
        let name = name.into();
        if words.device().location() != scales.device().location() {
            return Err(Error::Config(format!(
                "Prism MLX affine-2 {name} stores codes and scales on different devices"
            )));
        }
        let device = words.device().clone();
        let (rows, packed_cols) = words.dims2()?;
        let (scale_rows, scale_cols) = scales.dims2()?;
        let input_width = packed_cols
            .checked_mul(16)
            .ok_or_else(|| Error::Config("Prism MLX input width overflow".into()))?;
        if words.dtype() != DType::U32
            || input_width == 0
            || input_width % PRISM_GROUP_SIZE != 0
            || scale_rows != rows
            || scale_cols != input_width / PRISM_GROUP_SIZE
            || biases.dims2()? != (scale_rows, scale_cols)
        {
            return Err(Error::Config(format!(
                "Prism MLX affine-2 geometry for {name}: weight {:?} {:?}, scales {:?}, biases {:?}",
                words.dtype(),
                words.shape(),
                scales.shape(),
                biases.shape()
            )));
        }
        let scale_host = scales
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let bias_host = biases
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if scale_host
            .iter()
            .zip(&bias_host)
            .any(|(&scale, &bias)| !scale.is_finite() || !bias.is_finite() || bias != -scale)
        {
            return Err(Error::Config(format!(
                "Prism MLX affine-2 {name} requires finite biases exactly equal to -scales"
            )));
        }
        let word_host = if device.is_cpu() {
            Some(Arc::from(words.flatten_all()?.to_vec1::<u32>()?))
        } else {
            None
        };
        Self::new(
            name,
            PackedGeometry {
                rows,
                input_width,
                row_map,
                gdn,
            },
            PackedStorage::MlxAffine2 {
                words: (!device.is_cpu()).then_some(words),
                scales: if device.is_cpu() {
                    None
                } else {
                    Some(scales.to_dtype(DType::F32)?)
                },
                host_words: word_host,
                host_scales: device.is_cpu().then(|| Arc::from(scale_host)),
            },
            metadata,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_gguf(
        name: impl Into<String>,
        kind: PrismPackedKind,
        dimensions: &[usize],
        data: Vec<u8>,
        metadata: &PrismHadamardMetadata,
        row_map: Option<GdnRowMap>,
        gdn: Option<GdnLayout>,
        device: &Device,
    ) -> Result<Self> {
        let name = name.into();
        let matrix = PrismPackedMatrixRef::from_gguf(kind, dimensions, &data)
            .map_err(|e| prism_err(&format!("GGUF matrix {name}"), e))?;
        let rows = matrix.output_rows();
        let input_width = matrix.input_width();
        let (bytes, host_bytes) = if device.is_cpu() {
            (None, Some(Arc::from(data)))
        } else {
            (
                Some(Tensor::from_vec(
                    data,
                    (rows * input_width / PRISM_GROUP_SIZE * kind.block_bytes(),),
                    device,
                )?),
                None,
            )
        };
        Self::new(
            name,
            PackedGeometry {
                rows,
                input_width,
                row_map,
                gdn,
            },
            PackedStorage::Gguf {
                kind,
                bytes,
                host_bytes,
            },
            metadata,
            device.clone(),
        )
    }

    fn new(
        name: String,
        geometry: PackedGeometry,
        storage: PackedStorage,
        metadata: &PrismHadamardMetadata,
        device: Device,
    ) -> Result<Self> {
        let PackedGeometry {
            rows,
            input_width,
            row_map,
            gdn,
        } = geometry;
        metadata
            .validate()
            .map_err(|e| prism_err("Hadamard metadata", e))?;
        let classified = metadata
            .classify_weight(&name, input_width)
            .map_err(|e| prism_err(&format!("weight {name}"), e))?;
        if let Some(map) = row_map {
            map.validate(rows)?;
        }
        if gdn.is_some() && classified.role != PrismTransformRole::Forward {
            return Err(Error::Config(format!(
                "Prism GDN activation reorder on {name} requires a forward Hadamard transform"
            )));
        }
        Ok(Self {
            name,
            rows,
            input_width,
            storage,
            role: classified.role,
            signs: classified.signs.map(|s| Arc::<[i8]>::from(s.to_vec())),
            block_size: metadata.block_size,
            gdn,
            row_map,
            device,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn input_width(&self) -> usize {
        self.input_width
    }

    pub fn role(&self) -> PrismTransformRole {
        self.role
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.dim(x.rank() - 1)? != self.input_width {
            return Err(Error::Config(format!(
                "Prism {} input width {} != {}",
                self.name,
                x.dim(x.rank() - 1)?,
                self.input_width
            )));
        }
        if x.device().location() != self.device.location() {
            return Err(Error::Config(format!(
                "Prism {} activation and weight are on different devices",
                self.name
            )));
        }
        let original_shape = x.dims().to_vec();
        let tokens = x.elem_count() / self.input_width;
        let transformed = self.transform_forward(x)?;
        let flat = transformed
            .to_dtype(DType::F32)?
            .reshape((tokens, self.input_width))?
            .contiguous()?;
        let output = if flat.device().is_cuda() {
            #[cfg(feature = "cuda")]
            {
                self.cuda_matmul(&flat)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(Error::Unsupported(
                    "Prism CUDA weight reached a build without the cuda feature".into(),
                ));
            }
        } else if flat.device().is_cpu() {
            self.cpu_matmul(&flat)?
        } else {
            return Err(Error::Unsupported(
                "Prism packed weights currently support CPU and CUDA devices".into(),
            ));
        };
        let mut out_shape = original_shape;
        *out_shape.last_mut().expect("rank is nonzero") = self.rows;
        Ok(output.reshape(out_shape)?.to_dtype(x.dtype())?)
    }

    /// Decode selected embedding rows without expanding the full vocabulary matrix. The published
    /// inverse embedding contract is applied in the required order: normalized H, then signs.
    pub fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        if self.role != PrismTransformRole::Inverse {
            return Err(Error::Config(format!(
                "Prism {} is not declared as an inverse embedding",
                self.name
            )));
        }
        if ids.device().location() != self.device.location() {
            return Err(Error::Config(format!(
                "Prism {} ids and weight are on different devices",
                self.name
            )));
        }
        let decoded = if ids.device().is_cuda() {
            #[cfg(feature = "cuda")]
            {
                self.cuda_embedding(ids)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(Error::Unsupported(
                    "Prism CUDA embedding reached a build without the cuda feature".into(),
                ));
            }
        } else if ids.device().is_cpu() {
            self.cpu_embedding(ids)?
        } else {
            return Err(Error::Unsupported(
                "Prism packed embeddings currently support CPU and CUDA devices".into(),
            ));
        };
        let transformed = self.transform_inverse(&decoded)?;
        let mut shape = ids.dims().to_vec();
        shape.push(self.input_width);
        Ok(transformed.reshape(shape)?)
    }

    fn transform_forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.role == PrismTransformRole::None {
            return Ok(x.clone());
        }
        if self.role != PrismTransformRole::Forward {
            return Err(Error::Config(format!(
                "Prism inverse transform is valid only for embeddings ({})",
                self.name
            )));
        }
        if x.device().is_cpu() {
            let shape = x.dims().to_vec();
            let mut values = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let signs = self.signs.as_deref().ok_or_else(|| {
                Error::Config(format!("Prism {} has no explicit signs", self.name))
            })?;
            for row in values.chunks_exact_mut(self.input_width) {
                apply_hadamard_forward_in_place(row, signs, self.block_size, self.gdn)
                    .map_err(|e| prism_err(&format!("forward transform {}", self.name), e))?;
            }
            return Ok(Tensor::from_vec(values, shape, x.device())?.to_dtype(x.dtype())?);
        }
        self.transform_forward_tensor(x)
    }

    fn transform_inverse(&self, x: &Tensor) -> Result<Tensor> {
        let signs = self
            .signs
            .as_deref()
            .ok_or_else(|| Error::Config(format!("Prism {} has no explicit signs", self.name)))?;
        if x.device().is_cpu() {
            let shape = x.dims().to_vec();
            let mut values = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            for row in values.chunks_exact_mut(self.input_width) {
                apply_hadamard_inverse_in_place(row, signs, self.block_size)
                    .map_err(|e| prism_err(&format!("inverse transform {}", self.name), e))?;
            }
            return Ok(Tensor::from_vec(values, shape, x.device())?.to_dtype(x.dtype())?);
        }
        let h = fwht_tensor(x, self.block_size)?;
        multiply_signs(&h, signs)
    }

    fn transform_forward_tensor(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        if let Some(gdn) = self.gdn {
            out = gdn_reorder_tensor(&out, gdn)?;
        }
        let signs = self
            .signs
            .as_deref()
            .ok_or_else(|| Error::Config(format!("Prism {} has no explicit signs", self.name)))?;
        fwht_tensor(&multiply_signs(&out, signs)?, self.block_size)
    }

    fn cpu_matmul(&self, x: &Tensor) -> Result<Tensor> {
        let values = x.flatten_all()?.to_vec1::<f32>()?;
        let tokens = values.len() / self.input_width;
        let mut output = vec![0f32; tokens * self.rows];
        let mut row = vec![0f32; self.input_width];
        for logical_row in 0..self.rows {
            let stored_row = match self.row_map {
                Some(map) => map.source_row(logical_row)?,
                None => logical_row,
            };
            self.decode_row(stored_row, &mut row)?;
            for token in 0..tokens {
                let input = &values[token * self.input_width..(token + 1) * self.input_width];
                output[token * self.rows + logical_row] =
                    input.iter().zip(&row).map(|(a, b)| a * b).sum();
            }
        }
        Ok(Tensor::from_vec(output, (tokens, self.rows), x.device())?)
    }

    fn cpu_embedding(&self, ids: &Tensor) -> Result<Tensor> {
        let ids = ids.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?;
        let count = ids.len();
        let mut output = vec![0f32; count * self.input_width];
        for (at, id) in ids.into_iter().enumerate() {
            let row = usize::try_from(id).map_err(|_| {
                Error::Config(format!("Prism {} negative embedding id {id}", self.name))
            })?;
            if row >= self.rows {
                return Err(Error::Config(format!(
                    "Prism {} embedding id {row} outside {} rows",
                    self.name, self.rows
                )));
            }
            self.decode_row(
                row,
                &mut output[at * self.input_width..(at + 1) * self.input_width],
            )?;
        }
        Ok(Tensor::from_vec(
            output,
            (count, self.input_width),
            &self.device,
        )?)
    }

    fn decode_row(&self, row: usize, output: &mut [f32]) -> Result<()> {
        match &self.storage {
            PackedStorage::MlxAffine2 {
                host_words,
                host_scales,
                ..
            } => {
                let host_words = host_words
                    .as_deref()
                    .ok_or_else(|| Error::Config("Prism CPU codes are unavailable".into()))?;
                let host_scales = host_scales
                    .as_deref()
                    .ok_or_else(|| Error::Config("Prism CPU scales are unavailable".into()))?;
                let packed_cols = self.input_width / 16;
                let scale_cols = self.input_width / PRISM_GROUP_SIZE;
                let words = &host_words[row * packed_cols..(row + 1) * packed_cols];
                let scales = &host_scales[row * scale_cols..(row + 1) * scale_cols];
                for (col, value) in output.iter_mut().enumerate() {
                    let code = ((words[col / 16] >> (2 * (col % 16))) & 3) as f32;
                    *value = (code - 1.0) * scales[col / PRISM_GROUP_SIZE];
                }
            }
            PackedStorage::Gguf {
                kind, host_bytes, ..
            } => {
                let host_bytes = host_bytes
                    .as_deref()
                    .ok_or_else(|| Error::Config("Prism CPU GGUF bytes are unavailable".into()))?;
                PrismPackedMatrixRef::from_gguf(*kind, &[self.input_width, self.rows], host_bytes)
                    .and_then(|matrix| matrix.decode_row_into(row, output))
                    .map_err(|e| prism_err(&format!("decode {}", self.name), e))?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_matmul(&self, x: &Tensor) -> Result<Tensor> {
        cuda::matmul(self, x)
    }

    #[cfg(feature = "cuda")]
    fn cuda_embedding(&self, ids: &Tensor) -> Result<Tensor> {
        cuda::embedding(self, ids)
    }
}

/// Packed matrices keyed by the exact tensor name used by a decoder.
#[derive(Clone, Default)]
pub struct PrismRegistry {
    weights: HashMap<String, Arc<PrismPackedWeight>>,
}

impl PrismRegistry {
    pub fn new(weights: HashMap<String, Arc<PrismPackedWeight>>) -> Result<Self> {
        if weights.is_empty() {
            return Err(Error::Config("Prism registry is empty".into()));
        }
        if weights.iter().any(|(name, weight)| name != &weight.name) {
            return Err(Error::Config(
                "Prism registry key does not match packed weight name".into(),
            ));
        }
        Ok(Self { weights })
    }

    pub fn get(&self, name: &str) -> Option<&Arc<PrismPackedWeight>> {
        self.weights.get(name)
    }

    pub fn require(&self, name: &str) -> Result<Arc<PrismPackedWeight>> {
        self.get(name)
            .cloned()
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

fn multiply_signs(x: &Tensor, signs: &[i8]) -> Result<Tensor> {
    let width = x.dim(x.rank() - 1)?;
    if signs.len() != width {
        return Err(Error::Config(format!(
            "Prism sign width {} != activation width {width}",
            signs.len()
        )));
    }
    let signs = signs.iter().map(|&v| v as f32).collect::<Vec<_>>();
    let signs = Tensor::from_vec(signs, (width,), x.device())?.to_dtype(x.dtype())?;
    Ok(x.broadcast_mul(&signs)?)
}

fn fwht_tensor(x: &Tensor, block_size: usize) -> Result<Tensor> {
    let shape = x.dims().to_vec();
    let width = *shape
        .last()
        .ok_or_else(|| Error::Config("Prism FWHT needs a non-scalar activation".into()))?;
    if block_size == 0 || !block_size.is_power_of_two() || width % block_size != 0 {
        return Err(Error::Config(format!(
            "Prism FWHT width {width} is not divisible by power-of-two block {block_size}"
        )));
    }
    let rows = x.elem_count() / width;
    let blocks = width / block_size;
    let mut out = x.reshape((rows * blocks, block_size))?;
    let mut step = 1usize;
    while step < block_size {
        let pairs = block_size / (step * 2);
        let paired = out.reshape((rows * blocks, pairs, 2, step))?;
        let a = paired.narrow(2, 0, 1)?;
        let b = paired.narrow(2, 1, 1)?;
        out = Tensor::cat(&[&(a.broadcast_add(&b)?), &(a.broadcast_sub(&b)?)], 2)?
            .reshape((rows * blocks, block_size))?;
        step *= 2;
    }
    Ok(out
        .affine((block_size as f64).sqrt().recip(), 0.0)?
        .reshape(shape)?)
}

fn gdn_reorder_tensor(x: &Tensor, layout: GdnLayout) -> Result<Tensor> {
    let width = x.dim(x.rank() - 1)?;
    if layout.width() != width {
        return Err(Error::Config(format!(
            "Prism GDN activation width {width} != {}",
            layout.width()
        )));
    }
    let mut gather = vec![0u32; width];
    for head_dim in 0..layout.head_dim {
        for group in 0..layout.groups {
            for repetition in 0..layout.repetitions {
                let source = (head_dim * layout.groups + group) * layout.repetitions + repetition;
                let destination =
                    (head_dim * layout.repetitions + repetition) * layout.groups + group;
                gather[destination] = source as u32;
            }
        }
    }
    let gather = Tensor::from_vec(gather, (width,), x.device())?;
    Ok(x.index_select(&gather, x.rank() - 1)?)
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::WrapErr;
    use candle_core::{CpuStorage, CudaStorage, CustomOp2, CustomOp3, Layout, Shape};

    /// The Prism packed-operator kernels, compiled through the shared nvrtc compile-once seam
    /// (sc-24137 / sc-23990): once per device, failure cached, no build.rs.
    const PRISM_SRC: candle_quant_kernels::KernelSource = candle_quant_kernels::KernelSource {
        name: "candle_llm_prism_packed_v1",
        src: include_str!("prism_cuda.cu"),
        cc_floor: (7, 0),
    };

    fn function(
        dev: &candle_core::CudaDevice,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaFunction> {
        PRISM_SRC
            .compiled(dev)
            .and_then(|kernel| kernel.function(name))
            .map_err(|e| candle_core::Error::Msg(format!("Prism CUDA kernel: {e}")))
    }

    fn map_args(map: Option<GdnRowMap>) -> (u32, u32, u32, u32) {
        map.map(|m| {
            (
                m.prefix as u32,
                m.groups as u32,
                m.repetitions as u32,
                m.unit as u32,
            )
        })
        .unwrap_or((0, 0, 0, 0))
    }

    struct MlxMatmul {
        tokens: usize,
        rows: usize,
        width: usize,
        map: Option<GdnRowMap>,
    }

    impl CustomOp3 for MlxMatmul {
        fn name(&self) -> &'static str {
            "prism-mlx-affine2-matmul"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("Prism MLX CUDA matmul invoked on CPU")
        }

        fn cuda_fwd(
            &self,
            sx: &CudaStorage,
            lx: &Layout,
            sw: &CudaStorage,
            lw: &Layout,
            ss: &CudaStorage,
            ls: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let dev = sx.device().clone();
            let x = contiguous_cuda::<f32>(sx, lx)?;
            let words = contiguous_cuda::<u32>(sw, lw)?;
            let scales = contiguous_cuda::<f32>(ss, ls)?;
            let mut output = unsafe { dev.alloc::<f32>(self.tokens * self.rows) }?;
            let function = function(&dev, "prism_mlx_affine2_matmul_f32")?;
            let (prefix, groups, repetitions, unit) = map_args(self.map);
            let (tokens, rows, width) = (self.tokens as u32, self.rows as u32, self.width as u32);
            let config = LaunchConfig {
                grid_dim: ((self.tokens * self.rows) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
            };
            let stream = dev.cuda_stream();
            let mut builder = stream.launch_builder(&function);
            builder.arg(&x).arg(&words).arg(&scales).arg(&mut output);
            builder
                .arg(&tokens)
                .arg(&rows)
                .arg(&width)
                .arg(&prefix)
                .arg(&groups)
                .arg(&repetitions)
                .arg(&unit);
            unsafe { builder.launch(config) }.w()?;
            Ok((
                CudaStorage::wrap_cuda_slice(output, dev),
                Shape::from((self.tokens, self.rows)),
            ))
        }
    }

    struct GgufMatmul {
        kind: PrismPackedKind,
        tokens: usize,
        rows: usize,
        width: usize,
        map: Option<GdnRowMap>,
    }

    impl CustomOp2 for GgufMatmul {
        fn name(&self) -> &'static str {
            "prism-gguf-matmul"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("Prism GGUF CUDA matmul invoked on CPU")
        }

        fn cuda_fwd(
            &self,
            sx: &CudaStorage,
            lx: &Layout,
            sw: &CudaStorage,
            lw: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let dev = sx.device().clone();
            let x = contiguous_cuda::<f32>(sx, lx)?;
            let packed = contiguous_cuda::<u8>(sw, lw)?;
            let mut output = unsafe { dev.alloc::<f32>(self.tokens * self.rows) }?;
            let kernel = match self.kind {
                PrismPackedKind::Pq2_0 => "prism_pq2_matmul_f32",
                PrismPackedKind::Ptq1_0 => "prism_ptq_matmul_f32",
            };
            let function = function(&dev, kernel)?;
            let (prefix, groups, repetitions, unit) = map_args(self.map);
            let (tokens, rows, width) = (self.tokens as u32, self.rows as u32, self.width as u32);
            let config = LaunchConfig {
                grid_dim: ((self.tokens * self.rows) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
            };
            let stream = dev.cuda_stream();
            let mut builder = stream.launch_builder(&function);
            builder.arg(&x).arg(&packed).arg(&mut output);
            builder
                .arg(&tokens)
                .arg(&rows)
                .arg(&width)
                .arg(&prefix)
                .arg(&groups)
                .arg(&repetitions)
                .arg(&unit);
            unsafe { builder.launch(config) }.w()?;
            Ok((
                CudaStorage::wrap_cuda_slice(output, dev),
                Shape::from((self.tokens, self.rows)),
            ))
        }
    }

    struct MlxEmbedding {
        count: usize,
        rows: usize,
        width: usize,
    }

    impl CustomOp3 for MlxEmbedding {
        fn name(&self) -> &'static str {
            "prism-mlx-affine2-embedding"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("Prism MLX CUDA embedding invoked on CPU")
        }

        fn cuda_fwd(
            &self,
            si: &CudaStorage,
            li: &Layout,
            sw: &CudaStorage,
            lw: &Layout,
            ss: &CudaStorage,
            ls: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let dev = si.device().clone();
            let ids = contiguous_cuda::<i64>(si, li)?;
            let words = contiguous_cuda::<u32>(sw, lw)?;
            let scales = contiguous_cuda::<f32>(ss, ls)?;
            launch_embedding(
                &dev,
                "prism_mlx_affine2_embedding_f32",
                &ids,
                EmbeddingPacked::Mlx(&words, &scales),
                self.count,
                self.rows,
                self.width,
            )
        }
    }

    struct GgufEmbedding {
        kind: PrismPackedKind,
        count: usize,
        rows: usize,
        width: usize,
    }

    impl CustomOp2 for GgufEmbedding {
        fn name(&self) -> &'static str {
            "prism-gguf-embedding"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("Prism GGUF CUDA embedding invoked on CPU")
        }

        fn cuda_fwd(
            &self,
            si: &CudaStorage,
            li: &Layout,
            sw: &CudaStorage,
            lw: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let dev = si.device().clone();
            let ids = contiguous_cuda::<i64>(si, li)?;
            let packed = contiguous_cuda::<u8>(sw, lw)?;
            let kernel = match self.kind {
                PrismPackedKind::Pq2_0 => "prism_pq2_embedding_f32",
                PrismPackedKind::Ptq1_0 => "prism_ptq_embedding_f32",
            };
            launch_embedding(
                &dev,
                kernel,
                &ids,
                EmbeddingPacked::Gguf(&packed),
                self.count,
                self.rows,
                self.width,
            )
        }
    }

    fn contiguous_cuda<
        'a,
        T: candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + candle_core::cuda_backend::CudaDType,
    >(
        storage: &'a CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, T>> {
        let slice = storage.as_cuda_slice::<T>()?;
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Prism CUDA input must be contiguous".into()))?;
        Ok(slice.slice(start..end))
    }

    enum EmbeddingPacked<'a> {
        Mlx(
            &'a candle_core::cuda_backend::cudarc::driver::CudaView<'a, u32>,
            &'a candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>,
        ),
        Gguf(&'a candle_core::cuda_backend::cudarc::driver::CudaView<'a, u8>),
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_embedding(
        dev: &candle_core::CudaDevice,
        kernel: &str,
        ids: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, i64>,
        packed: EmbeddingPacked<'_>,
        count: usize,
        rows: usize,
        width: usize,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let mut output = unsafe { dev.alloc::<f32>(count * width) }?;
        let function = function(dev, kernel)?;
        let (count_u, rows_u, width_u) = (count as u32, rows as u32, width as u32);
        let config = LaunchConfig::for_num_elems((count * width) as u32);
        let stream = dev.cuda_stream();
        let mut builder = stream.launch_builder(&function);
        builder.arg(ids);
        match packed {
            EmbeddingPacked::Mlx(words, scales) => {
                builder.arg(words).arg(scales);
            }
            EmbeddingPacked::Gguf(bytes) => {
                builder.arg(bytes);
            }
        }
        builder
            .arg(&mut output)
            .arg(&count_u)
            .arg(&rows_u)
            .arg(&width_u);
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, dev.clone()),
            Shape::from((count, width)),
        ))
    }

    pub(super) fn matmul(weight: &PrismPackedWeight, x: &Tensor) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        match &weight.storage {
            PackedStorage::MlxAffine2 { words, scales, .. } => Ok(x.apply_op3_no_bwd(
                words
                    .as_ref()
                    .ok_or_else(|| Error::Config("missing Prism CUDA codes".into()))?,
                scales
                    .as_ref()
                    .ok_or_else(|| Error::Config("missing Prism CUDA scales".into()))?,
                &MlxMatmul {
                    tokens,
                    rows: weight.rows,
                    width: weight.input_width,
                    map: weight.row_map,
                },
            )?),
            PackedStorage::Gguf { kind, bytes, .. } => Ok(x.apply_op2_no_bwd(
                bytes
                    .as_ref()
                    .ok_or_else(|| Error::Config("missing Prism CUDA bytes".into()))?,
                &GgufMatmul {
                    kind: *kind,
                    tokens,
                    rows: weight.rows,
                    width: weight.input_width,
                    map: weight.row_map,
                },
            )?),
        }
    }

    pub(super) fn embedding(weight: &PrismPackedWeight, ids: &Tensor) -> Result<Tensor> {
        let ids = ids.to_dtype(DType::I64)?.contiguous()?;
        let count = ids.elem_count();
        match &weight.storage {
            PackedStorage::MlxAffine2 { words, scales, .. } => Ok(ids.apply_op3_no_bwd(
                words
                    .as_ref()
                    .ok_or_else(|| Error::Config("missing Prism CUDA codes".into()))?,
                scales
                    .as_ref()
                    .ok_or_else(|| Error::Config("missing Prism CUDA scales".into()))?,
                &MlxEmbedding {
                    count,
                    rows: weight.rows,
                    width: weight.input_width,
                },
            )?),
            PackedStorage::Gguf { kind, bytes, .. } => Ok(ids.apply_op2_no_bwd(
                bytes
                    .as_ref()
                    .ok_or_else(|| Error::Config("missing Prism CUDA bytes".into()))?,
                &GgufEmbedding {
                    kind: *kind,
                    count,
                    rows: weight.rows,
                    width: weight.input_width,
                },
            )?),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    fn metadata(name: &str, role: PrismTransformRole, signs: Vec<i8>) -> PrismHadamardMetadata {
        let mut signs_by_width = BTreeMap::new();
        signs_by_width.insert(signs.len(), signs);
        let mut forward_weight_names = BTreeSet::new();
        let mut inverse_weight_names = BTreeSet::new();
        match role {
            PrismTransformRole::Forward => {
                forward_weight_names.insert(name.into());
            }
            PrismTransformRole::Inverse => {
                inverse_weight_names.insert(name.into());
            }
            PrismTransformRole::None => {}
        }
        PrismHadamardMetadata {
            block_size: 128,
            signs_by_width,
            forward_weight_names,
            inverse_weight_names,
            gdn_v_grouped: true,
        }
    }

    fn mlx_weight(name: &str, codes: &[u8], role: PrismTransformRole) -> PrismPackedWeight {
        assert_eq!(codes.len(), 128);
        let mut words = vec![0u32; 8];
        for (lane, &code) in codes.iter().enumerate() {
            words[lane / 16] |= u32::from(code) << (2 * (lane % 16));
        }
        let device = Device::Cpu;
        PrismPackedWeight::from_mlx_affine2(
            name,
            Tensor::from_vec(words, (1, 8), &device).unwrap(),
            Tensor::from_vec(vec![0.5f32], (1, 1), &device).unwrap(),
            &Tensor::from_vec(vec![-0.5f32], (1, 1), &device).unwrap(),
            &metadata(
                name,
                role,
                (0..128).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect(),
            ),
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn mlx_affine2_matches_dense_reference_without_expansion() {
        let codes = (0..128).map(|i| (i % 3) as u8).collect::<Vec<_>>();
        let weight = mlx_weight("plain.weight", &codes, PrismTransformRole::None);
        let input = (0..128).map(|i| i as f32 / 64.0 - 1.0).collect::<Vec<_>>();
        let expected = input
            .iter()
            .zip(&codes)
            .map(|(&x, &code)| x * (f32::from(code) - 1.0) * 0.5)
            .sum::<f32>();
        let actual = weight
            .forward(&Tensor::from_vec(input, (1, 128), &Device::Cpu).unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[0][0];
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }

    #[test]
    fn pq2_and_ptq1_match_decoded_dense_rows() {
        for kind in [PrismPackedKind::Pq2_0, PrismPackedKind::Ptq1_0] {
            let mut bytes = vec![0u8; kind.block_bytes()];
            let scale_at = if kind == PrismPackedKind::Pq2_0 {
                0
            } else {
                26
            };
            bytes[scale_at..scale_at + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            let codes = if kind == PrismPackedKind::Pq2_0 {
                &mut bytes[2..]
            } else {
                &mut bytes[..26]
            };
            for (i, byte) in codes.iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(37);
            }
            let mut dense = vec![0f32; 128];
            PrismPackedMatrixRef::from_gguf(kind, &[128, 1], &bytes)
                .unwrap()
                .decode_row_into(0, &mut dense)
                .unwrap();
            let input = (0..128).map(|i| (i % 11) as f32 - 5.0).collect::<Vec<_>>();
            let expected = input.iter().zip(&dense).map(|(a, b)| a * b).sum::<f32>();
            let weight = PrismPackedWeight::from_gguf(
                "plain.weight",
                kind,
                &[128, 1],
                bytes,
                &metadata("plain.weight", PrismTransformRole::None, vec![1; 128]),
                None,
                None,
                &Device::Cpu,
            )
            .unwrap();
            let actual = weight
                .forward(&Tensor::from_vec(input, (1, 128), &Device::Cpu).unwrap())
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()[0][0];
            assert!(
                (actual - expected).abs() < 1e-4,
                "{kind:?}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn optional_ungrouped_gdn_reorder_precedes_sign_and_hadamard() {
        let name = "model.layers.0.linear_attn.ssm_out.weight";
        let codes = (0..128)
            .map(|i| if i % 7 == 0 { 2u8 } else { 1u8 })
            .collect::<Vec<_>>();
        let mut signs = vec![1i8; 128];
        signs[1] = -1;
        signs[65] = -1;
        let meta = metadata(name, PrismTransformRole::Forward, signs.clone());
        let device = Device::Cpu;
        let mut words = vec![0u32; 8];
        for (lane, &code) in codes.iter().enumerate() {
            words[lane / 16] |= u32::from(code) << (2 * (lane % 16));
        }
        let layout = GdnLayout {
            head_dim: 32,
            groups: 2,
            repetitions: 2,
        };
        let weight = PrismPackedWeight::from_mlx_affine2(
            name,
            Tensor::from_vec(words, (1, 8), &device).unwrap(),
            Tensor::from_vec(vec![1f32], (1, 1), &device).unwrap(),
            &Tensor::from_vec(vec![-1f32], (1, 1), &device).unwrap(),
            &meta,
            None,
            Some(layout),
        )
        .unwrap();
        let input = (0..128).map(|i| i as f32 - 60.0).collect::<Vec<_>>();
        let mut transformed = input.clone();
        apply_hadamard_forward_in_place(&mut transformed, &signs, 128, Some(layout)).unwrap();
        let expected = transformed
            .iter()
            .zip(&codes)
            .map(|(&x, &code)| x * (f32::from(code) - 1.0))
            .sum::<f32>();
        let actual = weight
            .forward(&Tensor::from_vec(input, (1, 128), &device).unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[0][0];
        assert!((actual - expected).abs() < 2e-3, "{actual} != {expected}");
    }

    #[test]
    fn published_grouped_ssm_out_does_not_permute_activation() {
        let name = "model.layers.0.linear_attn.ssm_out.weight";
        let codes = (0..128)
            .map(|i| if i % 7 == 0 { 2u8 } else { 1u8 })
            .collect::<Vec<_>>();
        let signs = (0..128)
            .map(|i| if i % 5 == 0 { -1i8 } else { 1 })
            .collect::<Vec<_>>();
        let mut words = vec![0u32; 8];
        for (lane, &code) in codes.iter().enumerate() {
            words[lane / 16] |= u32::from(code) << (2 * (lane % 16));
        }
        let device = Device::Cpu;
        let weight = PrismPackedWeight::from_mlx_affine2(
            name,
            Tensor::from_vec(words, (1, 8), &device).unwrap(),
            Tensor::from_vec(vec![1f32], (1, 1), &device).unwrap(),
            &Tensor::from_vec(vec![-1f32], (1, 1), &device).unwrap(),
            &metadata(name, PrismTransformRole::Forward, signs.clone()),
            None,
            None,
        )
        .unwrap();
        let input = (0..128).map(|i| i as f32 - 60.0).collect::<Vec<_>>();
        let mut transformed = input.clone();
        apply_hadamard_forward_in_place(&mut transformed, &signs, 128, None).unwrap();
        let expected = transformed
            .iter()
            .zip(&codes)
            .map(|(&x, &code)| x * (f32::from(code) - 1.0))
            .sum::<f32>();
        let actual = weight
            .forward(&Tensor::from_vec(input, (1, 128), &device).unwrap())
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[0][0];
        assert!((actual - expected).abs() < 2e-3, "{actual} != {expected}");
    }

    #[test]
    fn inverse_embedding_applies_hadamard_before_signs() {
        let name = "model.embed_tokens.weight";
        let codes = (0..128)
            .map(|i| if i == 0 { 2 } else { 1 })
            .collect::<Vec<_>>();
        let weight = mlx_weight(name, &codes, PrismTransformRole::Inverse);
        let mut expected = codes
            .iter()
            .map(|&c| (f32::from(c) - 1.0) * 0.5)
            .collect::<Vec<_>>();
        let signs = (0..128)
            .map(|i| if i % 3 == 0 { -1 } else { 1 })
            .collect::<Vec<_>>();
        apply_hadamard_inverse_in_place(&mut expected, &signs, 128).unwrap();
        let actual = weight
            .embedding(&Tensor::from_vec(vec![0u32], (1,), &Device::Cpu).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "lane {i}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn malformed_affine_and_row_metadata_fail_closed() {
        let device = Device::Cpu;
        let meta = metadata("plain.weight", PrismTransformRole::None, vec![1; 128]);
        let error = PrismPackedWeight::from_mlx_affine2(
            "plain.weight",
            Tensor::zeros((1, 8), DType::U32, &device).unwrap(),
            Tensor::ones((1, 1), DType::F32, &device).unwrap(),
            &Tensor::zeros((1, 1), DType::F32, &device).unwrap(),
            &meta,
            None,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("biases exactly equal"));
        let error = PrismPackedWeight::from_gguf(
            "plain.weight",
            PrismPackedKind::Pq2_0,
            &[128, 1],
            vec![0; 34],
            &meta,
            Some(GdnRowMap {
                prefix: 0,
                groups: 1,
                repetitions: 2,
                unit: 1,
            }),
            None,
            &device,
        )
        .unwrap_err();
        assert!(error.to_string().contains("row-map geometry"));
    }

    #[test]
    fn runtime_cuda_source_requires_no_host_or_sdk_headers() {
        // Production NVRTC compilation intentionally supplies no include directories. Keep this
        // invariant test available on CPU CI; the CUDA oracle separately compiles and runs every
        // packed operator on the actual device.
        let source = include_str!("prism_cuda.cu");
        assert!(!source.lines().any(|line| {
            let directive = line
                .trim_start()
                .strip_prefix('#')
                .unwrap_or("")
                .trim_start();
            directive.starts_with("include")
        }));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_packed_operator_oracles_compile_and_execute_nvrtc() {
        fn rotate(mut values: Vec<f32>, signs: &[i8], inverse: bool) -> Vec<f32> {
            if !inverse {
                for (value, sign) in values.iter_mut().zip(signs) {
                    *value *= f32::from(*sign);
                }
            }
            let mut step = 1;
            while step < values.len() {
                for base in (0..values.len()).step_by(step * 2) {
                    for lane in 0..step {
                        let a = values[base + lane];
                        let b = values[base + step + lane];
                        values[base + lane] = a + b;
                        values[base + step + lane] = a - b;
                    }
                }
                step *= 2;
            }
            let scale = (values.len() as f32).sqrt().recip();
            for value in &mut values {
                *value *= scale;
            }
            if inverse {
                for (value, sign) in values.iter_mut().zip(signs) {
                    *value *= f32::from(*sign);
                }
            }
            values
        }
        fn gguf_rows(kind: PrismPackedKind, rows: usize) -> (Vec<u8>, Vec<Vec<f32>>) {
            let mut bytes = Vec::new();
            let mut dense = Vec::new();
            for row in 0..rows {
                let scale = 0.125 * (row + 1) as f32;
                let mut block = vec![0u8; kind.block_bytes()];
                match kind {
                    PrismPackedKind::Pq2_0 => {
                        block[..2]
                            .copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                        for (i, value) in block[2..].iter_mut().enumerate() {
                            *value = (i as u8).wrapping_mul(29).wrapping_add(row as u8 * 7);
                        }
                    }
                    PrismPackedKind::Ptq1_0 => {
                        for (i, value) in block[..26].iter_mut().enumerate() {
                            *value = (i as u8).wrapping_mul(37).wrapping_add(row as u8 * 11);
                        }
                        block[26..]
                            .copy_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                    }
                }
                let mut decoded = vec![0.0f32; 128];
                match kind {
                    PrismPackedKind::Pq2_0 => {
                        for (lane, value) in decoded.iter_mut().enumerate() {
                            let code = (block[2 + lane / 4] >> (2 * (lane % 4))) & 3;
                            *value = (f32::from(code) - 1.0) * scale;
                        }
                    }
                    PrismPackedKind::Ptq1_0 => {
                        const POW3: [u16; 5] = [1, 3, 9, 27, 81];
                        let mut lane = 0;
                        for &(lo, hi, trits) in &[(0, 16, 5), (16, 24, 5), (24, 26, 4)] {
                            for &power in POW3.iter().take(trits) {
                                for &byte in &block[lo..hi] {
                                    let code =
                                        ((((u16::from(byte) * power) & 255) * 3) >> 8) as f32;
                                    decoded[lane] = (code - 1.0) * scale;
                                    lane += 1;
                                }
                            }
                        }
                    }
                }
                bytes.extend_from_slice(&block);
                dense.push(decoded);
            }
            (bytes, dense)
        }
        fn expected_matmul(
            inputs: &[Vec<f32>],
            dense: &[Vec<f32>],
            signs: &[i8],
            map: GdnRowMap,
        ) -> Vec<Vec<f32>> {
            inputs
                .iter()
                .map(|input| {
                    let rotated = rotate(input.clone(), signs, false);
                    (0..dense.len())
                        .map(|logical| {
                            let stored = map.source_row(logical).unwrap();
                            rotated.iter().zip(&dense[stored]).map(|(a, b)| a * b).sum()
                        })
                        .collect()
                })
                .collect()
        }
        fn assert_close(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
            for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                for (col, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        (actual - expected).abs() < 3e-3,
                        "[{row},{col}] {actual} != {expected}"
                    );
                }
            }
        }

        let device = Device::new_cuda(0).expect("CUDA is required for the Prism NVRTC fixture");
        let rows = 4usize;
        let width = 128usize;
        let signs = (0..width)
            .map(|i| if i % 5 == 0 || i % 11 == 0 { -1i8 } else { 1 })
            .collect::<Vec<_>>();
        let map = GdnRowMap {
            prefix: 0,
            groups: 2,
            repetitions: 2,
            unit: 1,
        };
        let inputs = vec![
            (0..width)
                .map(|i| i as f32 / 71.0 - 0.8)
                .collect::<Vec<_>>(),
            (0..width)
                .map(|i| (i % 13) as f32 / 9.0 - 0.4)
                .collect::<Vec<_>>(),
        ];
        let flat_inputs = inputs.iter().flatten().copied().collect::<Vec<_>>();

        let name = "cuda.forward.weight";
        let mut words = vec![0u32; rows * width / 16];
        let scales = (0..rows)
            .map(|row| 0.125 * (row + 1) as f32)
            .collect::<Vec<_>>();
        let mut dense = vec![vec![0.0f32; width]; rows];
        for row in 0..rows {
            for col in 0..width {
                let code = ((row * 3 + col * 5) % 4) as u32;
                words[row * width / 16 + col / 16] |= code << (2 * (col % 16));
                dense[row][col] = (code as f32 - 1.0) * scales[row];
            }
        }
        let affine = PrismPackedWeight::from_mlx_affine2(
            name,
            Tensor::from_vec(words.clone(), (rows, width / 16), &device).unwrap(),
            Tensor::from_vec(scales.clone(), (rows, 1), &device).unwrap(),
            &Tensor::from_vec(
                scales.iter().map(|s| -*s).collect::<Vec<_>>(),
                (rows, 1),
                &device,
            )
            .unwrap(),
            &metadata(name, PrismTransformRole::Forward, signs.clone()),
            Some(map),
            None,
        )
        .unwrap();
        let actual = affine
            .forward(&Tensor::from_vec(flat_inputs.clone(), (2, width), &device).unwrap())
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_close(&actual, &expected_matmul(&inputs, &dense, &signs, map));

        let embed_name = "cuda.embedding.weight";
        let embedding = PrismPackedWeight::from_mlx_affine2(
            embed_name,
            Tensor::from_vec(words, (rows, width / 16), &device).unwrap(),
            Tensor::from_vec(scales.clone(), (rows, 1), &device).unwrap(),
            &Tensor::from_vec(
                scales.iter().map(|s| -*s).collect::<Vec<_>>(),
                (rows, 1),
                &device,
            )
            .unwrap(),
            &metadata(embed_name, PrismTransformRole::Inverse, signs.clone()),
            None,
            None,
        )
        .unwrap();
        let actual = embedding
            .embedding(&Tensor::from_vec(vec![3u32, 1], (2,), &device).unwrap())
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let expected = vec![
            rotate(dense[3].clone(), &signs, true),
            rotate(dense[1].clone(), &signs, true),
        ];
        assert_close(&actual, &expected);

        for kind in [PrismPackedKind::Pq2_0, PrismPackedKind::Ptq1_0] {
            let (bytes, dense) = gguf_rows(kind, rows);
            let forward_name = format!("cuda.{kind:?}.forward.weight");
            let forward = PrismPackedWeight::from_gguf(
                &forward_name,
                kind,
                &[width, rows],
                bytes.clone(),
                &metadata(&forward_name, PrismTransformRole::Forward, signs.clone()),
                Some(map),
                None,
                &device,
            )
            .unwrap();
            let actual = forward
                .forward(&Tensor::from_vec(flat_inputs.clone(), (2, width), &device).unwrap())
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            assert_close(&actual, &expected_matmul(&inputs, &dense, &signs, map));

            let inverse_name = format!("cuda.{kind:?}.embedding.weight");
            let inverse = PrismPackedWeight::from_gguf(
                &inverse_name,
                kind,
                &[width, rows],
                bytes,
                &metadata(&inverse_name, PrismTransformRole::Inverse, signs.clone()),
                None,
                None,
                &device,
            )
            .unwrap();
            let actual = inverse
                .embedding(&Tensor::from_vec(vec![2u32, 0], (2,), &device).unwrap())
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let expected = vec![
                rotate(dense[2].clone(), &signs, true),
                rotate(dense[0].clone(), &signs, true),
            ];
            assert_close(&actual, &expected);
        }
    }
}
