//! Compact Prism/Bonsai affine-2 and GGUF ternary operators (`SW-6957` / `SW-F335`).
//!
//! The packed checkpoint remains the resident weight. CPU forwards decode one bounded row at a
//! time; CUDA forwards dispatch a packed matrix-vector/matrix kernel which decodes inside the dot
//! product. Neither path materializes a full dense model weight.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
#[cfg(feature = "cuda")]
use core_llm::DecodeBackend;
use core_llm::{
    apply_hadamard_forward_in_place, apply_hadamard_inverse_in_place, GdnLayout,
    PrismHadamardMetadata, PrismPackedKind, PrismPackedMatrixRef, PrismTransformRole,
    PRISM_GROUP_SIZE,
};

use crate::llm_projection::{Error, Result};

/// A CUDA kernel's `u32` launch argument or block dimension, refused rather than wrapped when
/// `n` does not fit (sc-27400): a wrapped size silently launches a partial grid.
pub fn launch_u32(n: usize, what: &str) -> candle_core::Result<u32> {
    u32::try_from(n)
        .map_err(|_| candle_core::Error::Msg(format!("CUDA launch: {what} {n} does not fit u32")))
}

/// A CUDA grid's x dimension (at most `2^31 - 1` blocks), refused rather than wrapped
/// (sc-27400).
pub fn launch_grid_x(n: usize, what: &str) -> candle_core::Result<u32> {
    if n > i32::MAX as usize {
        candle_core::bail!(
            "CUDA launch: {what} {n} exceeds the grid x limit {}",
            i32::MAX
        );
    }
    launch_u32(n, what)
}

/// `a * b` elements for a kernel's output or grid, refused on overflow (sc-27400).
pub fn launch_elems(a: usize, b: usize, what: &str) -> candle_core::Result<usize> {
    a.checked_mul(b).ok_or_else(|| {
        candle_core::Error::Msg(format!("CUDA launch: {what} {a} x {b} overflows usize"))
    })
}

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
    /// Map a logical decoder output row to its stored row in the packed Prism matrix.
    pub fn source_row(self, logical_row: usize) -> Result<usize> {
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

    /// Validate that `prefix + groups * repetitions * unit == rows`.
    pub fn validate(self, rows: usize) -> Result<()> {
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

#[doc(hidden)]
#[derive(Clone)]
#[allow(dead_code)] // Device tensors are consumed only in `cfg(feature = "cuda")` builds.
pub enum PackedStorage {
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
    #[doc(hidden)]
    pub storage: PackedStorage,
    role: PrismTransformRole,
    signs: Option<Arc<[i8]>>,
    /// `signs` as an F32 `[input_width]` tensor on `device`, built once at load: the tensor
    /// rotation path multiplies by it every forward without a host upload (which would also
    /// break CUDA-graph capture).
    device_signs: Option<Tensor>,
    /// The GDN activation reorder's `[input_width]` U32 gather index on `device`, built once at
    /// load for the same reason.
    gdn_gather: Option<Tensor>,
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
        let device_signs = classified
            .signs
            .map(|signs| upload_signs(signs, &device))
            .transpose()?;
        let gdn_gather = gdn
            .map(|layout| gdn_gather_index(layout, &device))
            .transpose()?;
        Ok(Self {
            name,
            rows,
            input_width,
            storage,
            role: classified.role,
            signs: classified.signs.map(|s| Arc::<[i8]>::from(s.to_vec())),
            device_signs,
            gdn_gather,
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
                __model_proof_output!(self.cuda_matmul(&flat))?
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(Error::Unsupported(
                    "Prism CUDA weight reached a build without the cuda feature".into(),
                ));
            }
        } else if flat.device().is_cpu() {
            __model_proof_output!(self.cpu_matmul(&flat))?
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
                __model_proof_output!(self.cuda_embedding(ids))?
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(Error::Unsupported(
                    "Prism CUDA embedding reached a build without the cuda feature".into(),
                ));
            }
        } else if ids.device().is_cpu() {
            __model_proof_output!(self.cpu_embedding(ids))?
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

    /// The forward rotation of `x` (GDN reorder, signs, block Hadamard). On the CPU it runs in
    /// f32 on the host and returns `x`'s dtype; on a device it is [`transform_forward_tensor`]
    /// (Self::transform_forward_tensor), which returns **f32** when the fused CUDA rotation ran (its
    /// output cast is fused — the caller widens to f32 next anyway) and `x`'s dtype otherwise.
    #[doc(hidden)]
    pub fn transform_forward(&self, x: &Tensor) -> Result<Tensor> {
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

    #[doc(hidden)]
    pub fn transform_inverse(&self, x: &Tensor) -> Result<Tensor> {
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
        if let Some(y) = self.fused_rotation(x, Rotation::Inverse)? {
            return Ok(y);
        }
        self.transform_inverse_tensor(x)
    }

    /// The device-tensor inverse rotation (normalized H, then signs): the op chain, the oracle
    /// of the fused inverse.
    #[doc(hidden)]
    pub fn transform_inverse_tensor(&self, x: &Tensor) -> Result<Tensor> {
        let h = fwht_tensor(x, self.block_size)?;
        multiply_signs(&h, self.resident_signs()?)
    }

    /// The device rotation: the fused CUDA kernel where it serves `x` (sc-24440), otherwise the
    /// op chain ([`transform_forward_chain`](Self::transform_forward_chain)).
    #[doc(hidden)]
    pub fn transform_forward_tensor(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(y) = self.fused_rotation(x, Rotation::Forward)? {
            return Ok(y);
        }
        self.transform_forward_chain(x)
    }

    /// The forward rotation as the device-tensor op chain — GDN reorder, sign product,
    /// `log2(block)` butterfly stages and the normalizing affine, each a stored tensor of `x`'s
    /// dtype. The path every non-CUDA device and every input the fused kernel refuses takes, and
    /// the oracle the fused kernel is bit-identical to.
    #[doc(hidden)]
    pub fn transform_forward_chain(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        if let Some(gather) = &self.gdn_gather {
            out = gdn_reorder_tensor(&out, gather)?;
        }
        fwht_tensor(
            &multiply_signs(&out, self.resident_signs()?)?,
            self.block_size,
        )
    }

    /// The load-time device copy of `signs`.
    fn resident_signs(&self) -> Result<&Tensor> {
        self.device_signs
            .as_ref()
            .ok_or_else(|| Error::Config(format!("Prism {} has no explicit signs", self.name)))
    }

    /// The fused CUDA rotation (sc-24440) of `x` in `rotation`'s order, or `None` when it does
    /// not serve the input — not CUDA, a dtype other than f32 / bf16 / f16, a Hadamard block
    /// wider than [`FUSED_ROTATION_MAX_BLOCK`], or the fused-kernel switch off — recorded in the
    /// fused-primitive tally either way (`crate::fused`). The forward returns f32 (the
    /// forward's cast fused), the inverse `x`'s dtype; both bit-identical to the op chain.
    ///
    /// It reads the load-time device signs and GDN gather (`resident_signs`, `gdn_gather`):
    /// nothing is uploaded per forward, so the rotation is graph-capturable.
    fn fused_rotation(&self, x: &Tensor, rotation: Rotation) -> Result<Option<Tensor>> {
        if !x.device().is_cuda() {
            return Ok(None);
        }
        #[cfg(feature = "cuda")]
        {
            use super::fused::{note_fused, note_reference};
            if !super::fused::try_fused() {
                return Ok(None);
            }
            // The defaults table's Candle CUDA row (E5) may keep the rotation on the op chain.
            let fused_rotation_default = if x.device().is_cuda() {
                DecodeBackend::CandleCuda
            } else if x.device().is_metal() {
                DecodeBackend::CandleMetal
            } else {
                DecodeBackend::CandleCpu
            }
            .defaults()
            .fused_rotation;
            if !fused_rotation_default {
                note_reference(super::fused::REASON_DISABLED);
                return Ok(None);
            }
            if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
                note_reference("dtype");
                return Ok(None);
            }
            if self.block_size > FUSED_ROTATION_MAX_BLOCK {
                note_reference("shape");
                return Ok(None);
            }
            let gather = match rotation {
                Rotation::Forward => self.gdn_gather.as_ref(),
                Rotation::Inverse => None,
            };
            let y = cuda::rotate(
                x,
                self.resident_signs()?,
                gather,
                self.block_size,
                rotation_affine_factor(self.block_size, x.dtype())
                    .ok_or_else(|| Error::Unsupported(format!("{:?} rotation", x.dtype())))?,
                rotation,
            )?;
            note_fused();
            Ok(Some(y))
        }
        #[cfg(not(feature = "cuda"))]
        {
            // A CUDA tensor cannot exist without the feature.
            let _ = rotation;
            Ok(None)
        }
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

/// Widest Hadamard block the fused CUDA rotation serves: one block's values live in its shared
/// memory as f32 (32 KiB here, under every device's 48 KiB default).
pub const FUSED_ROTATION_MAX_BLOCK: usize = 8192;

/// `1 / sqrt(block)` as `dtype` holds it — exactly the factor candle's `affine` hands its kernel
/// (`T::from_f64`), which the fused rotation must multiply by to stay bit-identical to the chain.
/// `None` for a dtype the fused rotation does not serve.
#[doc(hidden)]
pub fn rotation_affine_factor(block: usize, dtype: DType) -> Option<f32> {
    let factor = (block as f64).sqrt().recip();
    match dtype {
        DType::F32 => Some(factor as f32),
        DType::BF16 => Some(half::bf16::from_f64(factor).to_f32()),
        DType::F16 => Some(half::f16::from_f64(factor).to_f32()),
        _ => None,
    }
}

/// The order of a rotation: the forward (activation) transform or the inverse (embedding) one.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotation {
    /// GDN gather, signs, then the normalized Hadamard.
    Forward,
    /// The normalized Hadamard, then signs.
    Inverse,
}

thread_local! {
    /// Host→device uploads of a Prism sign vector or GDN gather index on this thread (test
    /// builds): they happen once per weight at load, never per forward.
    #[doc(hidden)]
    pub static ROTATION_UPLOADS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Upload `signs` once, as the F32 `[width]` tensor the rotation multiplies by.
#[doc(hidden)]
pub fn upload_signs(signs: &[i8], device: &Device) -> Result<Tensor> {
    ROTATION_UPLOADS.with(|n| n.set(n.get() + 1));
    let signs = signs.iter().map(|&v| f32::from(v)).collect::<Vec<_>>();
    let width = signs.len();
    Ok(Tensor::from_vec(signs, (width,), device)?)
}

#[doc(hidden)]
pub fn multiply_signs(x: &Tensor, signs: &Tensor) -> Result<Tensor> {
    let width = x.dim(x.rank() - 1)?;
    if signs.dim(0)? != width {
        return Err(Error::Config(format!(
            "Prism sign width {} != activation width {width}",
            signs.dim(0)?
        )));
    }
    Ok(x.broadcast_mul(&signs.to_dtype(x.dtype())?)?)
}

#[doc(hidden)]
pub fn fwht_tensor(x: &Tensor, block_size: usize) -> Result<Tensor> {
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

/// The GDN activation reorder's gather index, uploaded once per weight at load.
#[doc(hidden)]
pub fn gdn_gather_index(layout: GdnLayout, device: &Device) -> Result<Tensor> {
    ROTATION_UPLOADS.with(|n| n.set(n.get() + 1));
    let width = layout.width();
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
    Ok(Tensor::from_vec(gather, (width,), device)?)
}

#[doc(hidden)]
pub fn gdn_reorder_tensor(x: &Tensor, gather: &Tensor) -> Result<Tensor> {
    let width = x.dim(x.rank() - 1)?;
    if gather.dim(0)? != width {
        return Err(Error::Config(format!(
            "Prism GDN activation width {width} != {}",
            gather.dim(0)?
        )));
    }
    Ok(x.index_select(gather, x.rank() - 1)?)
}

/// The Prism packed-operator kernels, compiled through the shared nvrtc compile-once seam
/// (sc-24137 / sc-23990): once per device, failure cached, no build.rs.
pub const PRISM_SRC: candle_quant_kernels::KernelSource = candle_quant_kernels::KernelSource {
    name: "candle_llm_prism_packed_v1",
    src: include_str!("prism_cuda.cu"),
    cc_floor: (7, 0),
};

#[cfg(feature = "cuda")]
#[doc(hidden)]
// Every `unsafe` block here carries the invariant it relies on (sc-27399, sc-27400).
#[warn(clippy::undocumented_unsafe_blocks)]
pub mod cuda {
    use super::*;
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::WrapErr;
    use candle_core::{CpuStorage, CudaStorage, CustomOp2, CustomOp3, Layout, Shape};

    fn function(
        dev: &candle_core::CudaDevice,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaFunction> {
        PRISM_SRC
            .compiled(dev)
            .and_then(|kernel| kernel.function(name))
            .map_err(|e| candle_core::Error::Msg(format!("Prism CUDA kernel: {e}")))
    }

    fn map_args(map: Option<GdnRowMap>) -> candle_core::Result<(u32, u32, u32, u32)> {
        map.map_or(Ok((0, 0, 0, 0)), |m| {
            Ok((
                launch_u32(m.prefix, "Prism GDN row-map prefix")?,
                launch_u32(m.groups, "Prism GDN row-map groups")?,
                launch_u32(m.repetitions, "Prism GDN row-map repetitions")?,
                launch_u32(m.unit, "Prism GDN row-map unit")?,
            ))
        })
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
            let elems = launch_elems(self.tokens, self.rows, "Prism matmul tokens x rows")?;
            // SAFETY: one block per output element writes it, so the uninitialised buffer is
            // fully overwritten before it is read.
            let mut output = unsafe { dev.alloc::<f32>(elems) }?;
            let function = function(&dev, "prism_mlx_affine2_matmul_f32")?;
            let (prefix, groups, repetitions, unit) = map_args(self.map)?;
            let (tokens, rows, width) = (
                launch_u32(self.tokens, "Prism matmul tokens")?,
                launch_u32(self.rows, "Prism matmul rows")?,
                launch_u32(self.width, "Prism matmul width")?,
            );
            let config = LaunchConfig {
                grid_dim: (launch_grid_x(elems, "Prism matmul blocks")?, 1, 1),
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
            // SAFETY: the arguments match `prism_mlx_affine2_matmul_f32` in `prism_cuda.cu` in
            // order and type; `tokens * rows` blocks fit the kernel's u32 job index (checked
            // above), and each reads in-bounds rows of the contiguous inputs.
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
            let elems = launch_elems(self.tokens, self.rows, "Prism matmul tokens x rows")?;
            // SAFETY: one block per output element writes it, so the uninitialised buffer is
            // fully overwritten before it is read.
            let mut output = unsafe { dev.alloc::<f32>(elems) }?;
            let kernel = match self.kind {
                PrismPackedKind::Pq2_0 => "prism_pq2_matmul_f32",
                PrismPackedKind::Ptq1_0 => "prism_ptq_matmul_f32",
            };
            let function = function(&dev, kernel)?;
            let (prefix, groups, repetitions, unit) = map_args(self.map)?;
            let (tokens, rows, width) = (
                launch_u32(self.tokens, "Prism matmul tokens")?,
                launch_u32(self.rows, "Prism matmul rows")?,
                launch_u32(self.width, "Prism matmul width")?,
            );
            let config = LaunchConfig {
                grid_dim: (launch_grid_x(elems, "Prism matmul blocks")?, 1, 1),
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
            // SAFETY: the arguments match the `PRISM_GGUF_MATMUL` kernels in `prism_cuda.cu` in
            // order and type; `tokens * rows` blocks fit the kernel's u32 job index (checked
            // above), and each reads in-bounds rows of the contiguous inputs.
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
        let elems = launch_elems(count, width, "Prism embedding count x width")?;
        // SAFETY: one thread per output element writes it, so the uninitialised buffer is fully
        // overwritten before it is read.
        let mut output = unsafe { dev.alloc::<f32>(elems) }?;
        let function = function(dev, kernel)?;
        let (count_u, rows_u, width_u) = (
            launch_u32(count, "Prism embedding count")?,
            launch_u32(rows, "Prism embedding rows")?,
            launch_u32(width, "Prism embedding width")?,
        );
        let config =
            LaunchConfig::for_num_elems(launch_u32(elems, "Prism embedding output elements")?);
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
        // SAFETY: the arguments match the `*_embedding_f32` kernels in `prism_cuda.cu` in order
        // and type; `count * width` fits their u32 element index (checked above), and an
        // out-of-range id writes NaN instead of reading past the table.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, dev.clone()),
            Shape::from((count, width)),
        ))
    }

    /// The fused block-Hadamard rotation (`prism_rotate` in `prism_cuda.cu`) of `x` `[.., width]`
    /// (f32 / bf16 / f16): one launch, one block per (row, Hadamard block). `signs` is F32
    /// `[width]`; `gather` (forward only) the U32 `[width]` GDN source-column index; `mul` the
    /// affine factor as `x`'s dtype holds it
    /// ([`rotation_affine_factor`](super::rotation_affine_factor)). The forward returns F32 (the
    /// chain's trailing cast fused), the inverse `x`'s dtype.
    pub fn rotate(
        x: &Tensor,
        signs: &Tensor,
        gather: Option<&Tensor>,
        block: usize,
        mul: f32,
        rotation: Rotation,
    ) -> Result<Tensor> {
        let Device::Cuda(dev) = x.device() else {
            return Err(Error::Unsupported(
                "the fused Prism rotation is CUDA-only".into(),
            ));
        };
        let shape = x.dims().to_vec();
        let width = *shape
            .last()
            .ok_or_else(|| Error::Config("Prism rotation needs a non-scalar activation".into()))?;
        if block == 0 || !block.is_power_of_two() || width % block != 0 || signs.dim(0)? != width {
            return Err(Error::Config(format!(
                "Prism rotation: width {width}, block {block}, {} signs",
                signs.dim(0)?
            )));
        }
        let rows = x.elem_count() / width;
        let x = x.contiguous()?;
        let signs = signs.to_dtype(DType::F32)?.contiguous()?;
        let gather = gather.map(|g| g.contiguous()).transpose()?;
        let dtype_code = match x.dtype() {
            DType::F32 => 0i32,
            DType::BF16 => 1,
            DType::F16 => 2,
            other => {
                return Err(Error::Unsupported(format!(
                    "the fused Prism rotation serves f32 / bf16 / f16, not {other:?}"
                )))
            }
        };
        let out_dtype = match rotation {
            Rotation::Forward => DType::F32,
            Rotation::Inverse => x.dtype(),
        };
        let out_code = i32::from(out_dtype != DType::F32);
        let inverse = i32::from(rotation == Rotation::Inverse);
        let function = function(dev, "prism_rotate")?;
        let config = LaunchConfig {
            grid_dim: (
                launch_grid_x(width / block, "Prism rotation blocks")?,
                launch_u32(rows, "Prism rotation rows")?,
                1,
            ),
            block_dim: ((block / 2).clamp(1, 256) as u32, 1, 1),
            shared_mem_bytes: launch_u32(
                block.saturating_mul(std::mem::size_of::<f32>()),
                "Prism rotation shared bytes",
            )?,
        };
        let (width_u, block_u) = (
            launch_u32(width, "Prism rotation width")?,
            launch_u32(block, "Prism rotation block")?,
        );
        let stream = dev.cuda_stream();
        let (xs, xl) = x.storage_and_layout();
        let (ss, sl) = signs.storage_and_layout();
        let candle_core::Storage::Cuda(xs) = &*xs else {
            return Err(Error::Unsupported(
                "Prism rotation input is not on CUDA".into(),
            ));
        };
        let candle_core::Storage::Cuda(ss) = &*ss else {
            return Err(Error::Unsupported(
                "Prism rotation signs are not on CUDA".into(),
            ));
        };
        let signs_view = ss.as_cuda_slice::<f32>()?.slice(sl.start_offset()..);
        let gather_storage = gather.as_ref().map(|g| g.storage_and_layout());
        let gather_view = match &gather_storage {
            Some((gs, gl)) => match &**gs {
                candle_core::Storage::Cuda(gs) => {
                    Some(gs.as_cuda_slice::<u32>()?.slice(gl.start_offset()..))
                }
                _ => {
                    return Err(Error::Unsupported(
                        "Prism rotation gather index is not on CUDA".into(),
                    ))
                }
            },
            None => None,
        };
        let null = 0u64;
        macro_rules! launch {
            ($t:ty, $o:ty) => {{
                let xv = xs.as_cuda_slice::<$t>()?.slice(xl.start_offset()..);
                // SAFETY: the kernel writes every output element.
                let output = unsafe { dev.alloc::<$o>(rows * width) }?;
                let mut builder = stream.launch_builder(&function);
                builder.arg(&xv).arg(&signs_view);
                match &gather_view {
                    Some(g) => builder.arg(g),
                    None => builder.arg(&null),
                };
                builder
                    .arg(&output)
                    .arg(&width_u)
                    .arg(&block_u)
                    .arg(&mul)
                    .arg(&dtype_code)
                    .arg(&out_code)
                    .arg(&inverse);
                // SAFETY: matches `prism_rotate` in `prism_cuda.cu`.
                unsafe { builder.launch(config) }.w()?;
                Tensor::from_storage(
                    candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                    Shape::from(shape.clone()),
                    candle_core::op::BackpropOp::none(),
                    false,
                )
            }};
        }
        Ok(match (x.dtype(), out_dtype) {
            (DType::F32, _) => launch!(f32, f32),
            (DType::BF16, DType::F32) => launch!(half::bf16, f32),
            (DType::BF16, _) => launch!(half::bf16, half::bf16),
            (DType::F16, DType::F32) => launch!(half::f16, f32),
            _ => launch!(half::f16, half::f16),
        })
    }

    pub fn matmul(weight: &PrismPackedWeight, x: &Tensor) -> Result<Tensor> {
        let tokens = x.dim(0)?;
        match &weight.storage {
            PackedStorage::MlxAffine2 { words, scales, .. } => Ok(__model_proof_output!(x
                .apply_op3_no_bwd(
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
                ))?),
            PackedStorage::Gguf { kind, bytes, .. } => Ok(__model_proof_output!(x
                .apply_op2_no_bwd(
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
                ))?),
        }
    }

    pub fn embedding(weight: &PrismPackedWeight, ids: &Tensor) -> Result<Tensor> {
        let ids = ids.to_dtype(DType::I64)?.contiguous()?;
        let count = ids.elem_count();
        match &weight.storage {
            PackedStorage::MlxAffine2 { words, scales, .. } => Ok(__model_proof_output!(ids
                .apply_op3_no_bwd(
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
                ))?),
            PackedStorage::Gguf { kind, bytes, .. } => Ok(__model_proof_output!(ids
                .apply_op2_no_bwd(
                    bytes
                        .as_ref()
                        .ok_or_else(|| Error::Config("missing Prism CUDA bytes".into()))?,
                    &GgufEmbedding {
                        kind: *kind,
                        count,
                        rows: weight.rows,
                        width: weight.input_width,
                    },
                ))?),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdn_row_map_preserves_prefix_and_transposes_group_repetition_grid() {
        let map = GdnRowMap {
            prefix: 4,
            groups: 2,
            repetitions: 3,
            unit: 2,
        };
        map.validate(4 + 2 * 3 * 2).unwrap();
        assert!(map.validate(15).is_err());

        for r in 0..4 {
            assert_eq!(map.source_row(r).unwrap(), r);
        }
        // logical row 4 -> group 0, rep 0, lane 0 -> stored 4
        assert_eq!(map.source_row(4).unwrap(), 4);
        // logical row 6 -> group 0, rep 1, lane 0 -> stored 4 + (1 * 2 + 0) * 2 = 8
        assert_eq!(map.source_row(6).unwrap(), 8);
        // logical row 10 -> group 1, rep 0, lane 0 -> stored 4 + (0 * 2 + 1) * 2 = 6
        assert_eq!(map.source_row(10).unwrap(), 6);
        assert!(map.source_row(16).is_err());
    }

    #[test]
    fn launch_params_refuse_instead_of_wrapping() {
        assert_eq!(launch_u32(7, "rows").unwrap(), 7);
        assert_eq!(launch_u32(u32::MAX as usize, "rows").unwrap(), u32::MAX);
        assert!(launch_u32(u32::MAX as usize + 1, "rows").is_err());
        assert_eq!(
            launch_grid_x(i32::MAX as usize, "grid").unwrap(),
            i32::MAX as u32
        );
        assert!(launch_grid_x(i32::MAX as usize + 1, "grid").is_err());
        assert_eq!(launch_elems(3, 5, "tokens x rows").unwrap(), 15);
        assert!(launch_elems(usize::MAX, 2, "tokens x rows").is_err());
    }

    #[test]
    fn runtime_cuda_source_requires_no_host_or_sdk_headers() {
        let source = PRISM_SRC.src;
        assert!(!source.lines().any(|line| {
            let directive = line
                .trim_start()
                .strip_prefix('#')
                .unwrap_or("")
                .trim_start();
            directive.starts_with("include")
        }));
    }
}
