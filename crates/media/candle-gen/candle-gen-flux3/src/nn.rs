//! The leaf modules of the FLUX 3 DiT graph and the precision policy they share — the Candle twin
//! of `mlx-gen-flux3`'s `nn` module, op for op.
//!
//! **Precision policy** (mirrors upstream's CUDA path, `torch.autocast(bfloat16)` over bf16
//! parameters, and degenerates to the oracle's FP32 CPU path when the compute dtype is f32):
//!
//! * every [`Linear`] casts its input to the compute dtype and returns the compute dtype
//!   (autocast's matmul rule) — its weight is stored in the compute dtype;
//! * `F.layer_norm` is on autocast's f32 list: [`layer_norm_f32`] normalizes in f32 and returns
//!   f32, so the modulated input of a block is f32 until the next Linear casts it back;
//! * the q/k RMSNorm + RoPE ([`rmsnorm_rope`]) runs in f32 and returns the dtype it was given;
//! * attention takes compute-dtype q/k/v and returns the compute dtype
//!   (`scaled_dot_product_attention` is on autocast's low-precision list), with its logits,
//!   softmax and `p·v` in f32 as the fused kernels keep them — see [`packed_attention`]; the
//!   residual streams and the modulation vectors are outputs of Linears, so they stay in the
//!   compute dtype;
//! * every other elementwise op follows ordinary **type promotion** — candle has none, so the
//!   binary helpers here ([`add`], [`mul`], [`cat`]) promote to f32 whenever their operands
//!   disagree, exactly as torch (and MLX) do.
//!
//! The Candle CPU backend has no half-precision GEMM, so there a bf16 compute dtype runs each GEMM
//! and each attention kernel in f32 over bf16-rounded operands and rounds the result back to bf16 —
//! what a bf16 GEMM computes (f32 accumulation, one rounding of the result) — the same rule the
//! sibling `candle-gen-iris` applies ([`gemm_dtype`]).

use candle_gen::candle_core::quantized::QMatMul;
use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::VarBuilder;
use candle_gen::{CandleError as Error, Result};

/// LayerNorm eps of every `layernorm_modulation` (`F.layer_norm(..., eps=1e-6)`).
pub const LN_EPS: f64 = 1e-6;
/// RMSNorm eps of the q/k norms (`rmsnorm_rope(..., eps=1e-6)`).
pub const RMS_EPS: f32 = 1e-6;

/// The dtype two operands meet at: themselves when they agree, f32 otherwise (the only two dtypes
/// this graph carries are f32 and the compute dtype).
fn common(a: DType, b: DType) -> DType {
    if a == b {
        a
    } else {
        DType::F32
    }
}

/// `x` at `dtype` (a no-op clone when it already is).
pub fn to(x: &Tensor, dtype: DType) -> Result<Tensor> {
    Ok(if x.dtype() == dtype {
        x.clone()
    } else {
        x.to_dtype(dtype)?
    })
}

/// Promoting broadcast `a + b`.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dt = common(a.dtype(), b.dtype());
    Ok(to(a, dt)?.broadcast_add(&to(b, dt)?)?)
}

/// Promoting broadcast `a · b`.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dt = common(a.dtype(), b.dtype());
    Ok(to(a, dt)?.broadcast_mul(&to(b, dt)?)?)
}

/// Concatenate along `axis`, promoting every part to the common dtype first.
pub fn cat(parts: &[&Tensor], axis: usize) -> Result<Tensor> {
    let dt = parts
        .iter()
        .skip(1)
        .fold(parts[0].dtype(), |acc, p| common(acc, p.dtype()));
    let parts = parts
        .iter()
        .map(|p| to(p, dt))
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::cat(&parts, axis)?)
}

/// Split the last axis into `n` equal chunks (`tensor.chunk(n, dim=-1)`).
pub fn chunk_last(x: &Tensor, n: usize) -> Result<Vec<Tensor>> {
    Ok(x.chunk(n, D::Minus1)?)
}

/// `F.silu` in the input dtype.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_gen::candle_nn::ops::silu(x)?)
}

/// The dtype a GEMM over `t`'s dtype runs in on `t`'s device: itself, except a half-precision
/// operand on the Candle CPU backend (which has no bf16/f16 matmul kernel) accumulates in f32 and
/// is rounded back by the caller.
pub fn gemm_dtype(t: &Tensor) -> DType {
    kernel_dtype(t.device(), t.dtype())
}

/// See [`gemm_dtype`].
pub fn kernel_dtype(device: &Device, dtype: DType) -> DType {
    if device.is_cpu() && matches!(dtype, DType::BF16 | DType::F16) {
        DType::F32
    } else {
        dtype
    }
}

/// Builds modules from the package's `model.safetensors` (an mmap, tensors read one at a time at
/// the compute dtype) on one device.
pub struct Loader<'a> {
    pub vb: &'a VarBuilder<'static>,
    pub compute: DType,
    pub device: &'a Device,
    /// Native GGML targets of a converted package; `None` for the dense package.
    pub packed: Option<&'a PackedSource>,
    /// Publisher FP8R/NVFP4 targets of a frozen BFL quantized package (CUDA only).
    pub publisher: Option<&'a PublisherSource>,
}

/// The raw GGML rows of a converted package's packed targets, read straight into a `QTensor`
/// (never through the compute-dtype `VarBuilder`, which would cast the code bytes as numbers).
pub struct PackedSource {
    pub(crate) path: std::path::PathBuf,
    pub(crate) targets: std::collections::BTreeSet<String>,
    pub(crate) ggml: candle_gen::candle_core::quantized::GgmlDType,
    pub(crate) storage: gen_core_flux3::artifacts::NativeDitStorage,
}

impl PackedSource {
    /// The GGML `storage` rows of `path`; `targets` is the storage's exact packed partition.
    pub fn new(
        path: &std::path::Path,
        storage: gen_core_flux3::artifacts::NativeDitStorage,
        targets: std::collections::BTreeSet<String>,
    ) -> Result<Self> {
        Ok(Self {
            path: path.to_path_buf(),
            targets,
            ggml: crate::convert::ggml_dtype(storage)?,
            storage,
        })
    }

    /// The raw U8 block rows of one packed target, through the audited loader seam.
    fn rows(&self, key: &str) -> Result<(Vec<u8>, usize, usize)> {
        let t = candle_gen::load_one_tensor(&self.path, key, DType::U8, &Device::Cpu, "flux3")?;
        let (n, bytes_per_row) = t.dims2()?;
        let block = self.ggml.type_size();
        if bytes_per_row % block != 0 {
            return Err(Error::Msg(format!(
                "flux3: {key} rows are {bytes_per_row} B, not whole {block}-byte GGML blocks"
            )));
        }
        let k = bytes_per_row / block * self.ggml.block_size();
        Ok((t.flatten_all()?.to_vec1::<u8>()?, n, k))
    }
}

/// The publisher FP8R / NVFP4 targets of a frozen BFL quantized package, staged AS SHIPPED onto
/// the frozen-recipe CUDA kernels (`candle_quant_kernels::flux3_quant`). There is no CPU or Metal
/// path: on those builds construction is a typed refusal, never a dequantized substitute.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct PublisherSource {
    pub(crate) reader: gen_core_flux3::safetensors_stream::SafetensorsSource,
    pub(crate) targets: std::collections::BTreeSet<String>,
    pub(crate) storage: gen_core_flux3::artifacts::NativeDitStorage,
    pub(crate) device: Device,
    #[cfg(feature = "cuda")]
    pub(crate) lt: std::sync::Arc<candle_quant_kernels::cublaslt::CublasLt>,
}

impl PublisherSource {
    /// Validate the device for `storage` (CUDA with the format's compute floor) and open `path`.
    pub fn new(
        path: &std::path::Path,
        storage: gen_core_flux3::artifacts::NativeDitStorage,
        targets: std::collections::BTreeSet<String>,
        device: &Device,
    ) -> Result<Self> {
        let reader = gen_core_flux3::safetensors_stream::SafetensorsSource::open(path)
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        #[cfg(feature = "cuda")]
        {
            use gen_core_flux3::quantization::DitFormat;
            if !device.is_cuda() {
                return Err(Error::Unsupported(
                    "flux3: publisher FP8R/NVFP4 kernels run on CUDA only".into(),
                ));
            }
            let lt = std::sync::Arc::new(candle_quant_kernels::cublaslt::CublasLt::new(device)?);
            let meets = match storage {
                gen_core_flux3::artifacts::NativeDitStorage::Publisher(DitFormat::Fp8Rowwise) => {
                    lt.meets_fp8_floor()?
                }
                gen_core_flux3::artifacts::NativeDitStorage::Publisher(DitFormat::Nvfp4) => {
                    lt.meets_nvfp4_floor()?
                }
                _ => false,
            };
            if !meets {
                return Err(Error::Unsupported(format!(
                    "flux3: this CUDA device {:?} is below the {} compute floor; install a tier \
                     this device runs (no dequantized substitute)",
                    lt.compute_cap()?,
                    storage
                        .key()
                        .map_err(|e| Error::Unsupported(e.to_string()))?
                )));
            }
            Ok(Self {
                reader,
                targets,
                storage,
                device: device.clone(),
                lt,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (reader, targets, device);
            Err(Error::Unsupported(format!(
                "flux3: {} publisher kernels need the CUDA build; install the dense or native \
                 q4/q8 tier on this backend",
                storage
                    .key()
                    .map_err(|e| Error::Unsupported(e.to_string()))?
            )))
        }
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn raw(&self, name: &str) -> Result<Vec<u8>> {
        self.reader
            .read(name)
            .map_err(|e| Error::Unsupported(e.to_string()))
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn scalar(&self, name: &str) -> Result<f32> {
        let bytes = self.raw(name)?;
        let bytes: [u8; 4] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| Error::Unsupported(format!("flux3: {name} is not one f32")))?;
        Ok(f32::from_le_bytes(bytes))
    }

    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn linear(&self, key: &str) -> Result<Linear> {
        let stem = key
            .strip_suffix(".weight")
            .ok_or_else(|| Error::Unsupported(format!("flux3: {key} is not a weight")))?;
        let shape = self
            .reader
            .get(key)
            .map_err(|e| Error::Unsupported(e.to_string()))?
            .shape
            .clone();
        let (out, input) = match (self.storage, shape.as_slice()) {
            (
                gen_core_flux3::artifacts::NativeDitStorage::Publisher(
                    gen_core_flux3::quantization::DitFormat::Nvfp4,
                ),
                [n, half],
            ) => (*n, half * 2),
            (_, [n, k]) => (*n, *k),
            _ => return Err(Error::Unsupported(format!("flux3: {key} is not a matrix"))),
        };
        #[cfg(feature = "cuda")]
        {
            use gen_core_flux3::artifacts::NativeDitStorage;
            use gen_core_flux3::quantization::DitFormat;
            let kernel = match self.storage {
                NativeDitStorage::Publisher(DitFormat::Fp8Rowwise) => {
                    let scales: Vec<f32> = self
                        .raw(&format!("{stem}.weight_scale"))?
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    PublisherKernel::Fp8 {
                        weight: self.lt.stage_fp8_bytes(&self.raw(key)?, out, input)?,
                        scales: self.lt.stage_f32(&scales)?,
                        lt: self.lt.clone(),
                    }
                }
                NativeDitStorage::Publisher(DitFormat::Nvfp4) => PublisherKernel::Nvfp4 {
                    weight: self.lt.flux3_stage_nvfp4_weight(
                        &self.raw(key)?,
                        &self.raw(&format!("{stem}.weight_scale"))?,
                        out,
                        input,
                        self.scalar(&format!("{stem}.weight_scale_2"))?,
                    )?,
                    input_scale: self.scalar(&format!("{stem}.input_scale"))?,
                    lt: self.lt.clone(),
                },
                _ => return Err(Error::Unsupported("flux3: not a publisher storage".into())),
            };
            Ok(Linear {
                base: Base::Publisher(PublisherBase {
                    kernel,
                    storage: self.storage,
                    out,
                    input,
                    device: self.device.clone(),
                }),
                adapted: None,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (stem, out, input);
            Err(Error::Unsupported(
                "flux3: publisher kernels need the CUDA build".into(),
            ))
        }
    }
}

/// A publisher-format projection on its frozen-recipe CUDA kernel. BF16 in, BF16 out.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct PublisherBase {
    kernel: PublisherKernel,
    storage: gen_core_flux3::artifacts::NativeDitStorage,
    out: usize,
    input: usize,
    device: Device,
}

#[cfg(feature = "cuda")]
enum PublisherKernel {
    Fp8 {
        lt: std::sync::Arc<candle_quant_kernels::cublaslt::CublasLt>,
        weight: candle_quant_kernels::cublaslt::DevFp8,
        scales: candle_gen::candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    },
    Nvfp4 {
        lt: std::sync::Arc<candle_quant_kernels::cublaslt::CublasLt>,
        weight: candle_quant_kernels::cublaslt::DevNvfp4,
        input_scale: f32,
    },
}

/// Uninhabited off CUDA: no publisher projection can exist there.
#[cfg(not(feature = "cuda"))]
enum PublisherKernel {}

impl PublisherBase {
    fn forward_chunk(&self, chunk: &Tensor) -> Result<Tensor> {
        match &self.kernel {
            #[cfg(feature = "cuda")]
            PublisherKernel::Fp8 { lt, weight, scales } => {
                Ok(lt.flux3_fp8_rowwise_linear(chunk, weight, scales)?)
            }
            #[cfg(feature = "cuda")]
            PublisherKernel::Nvfp4 {
                lt,
                weight,
                input_scale,
            } => Ok(lt.flux3_nvfp4_static_linear(chunk, weight, *input_scale)?),
            #[cfg(not(feature = "cuda"))]
            _ => {
                let _ = chunk;
                unreachable!("PublisherKernel is uninhabited off CUDA")
            }
        }
    }
}

impl Loader<'_> {
    /// Whether the checkpoint carries `key`.
    pub fn contains(&self, key: &str) -> bool {
        self.vb.contains_tensor(key)
    }

    fn take(&self, key: &str) -> Result<Tensor> {
        if !self.vb.contains_tensor(key) {
            return Err(Error::Msg(format!(
                "flux3: the DiT checkpoint is missing `{key}` (config.json and model.safetensors \
                 disagree)"
            )));
        }
        Ok(self.vb.get_unchecked(key)?)
    }

    /// A tensor kept in f32 (the q/k RMSNorm scales).
    pub fn f32(&self, key: &str) -> Result<Tensor> {
        to(&self.take(key)?, DType::F32)
    }

    /// A bias-less `nn.Linear` (`{prefix}.weight`, `[out, in]`). Dense packages store it in the
    /// compute dtype; a converted package's packed target is held as its GGML `QTensor`.
    pub fn linear(&self, prefix: &str) -> Result<Linear> {
        let key = format!("{prefix}.weight");
        if let Some(packed) = self.packed.filter(|p| p.targets.contains(&key)) {
            return Linear::from_ggml_rows(packed, &key, self.compute, self.device);
        }
        if let Some(publisher) = self.publisher.filter(|p| p.targets.contains(&key)) {
            return publisher.linear(&key);
        }
        let weight = to(&self.take(&key)?, self.compute)?;
        if weight.rank() != 2 {
            return Err(Error::Msg(format!(
                "flux3: {prefix}.weight is rank {}, a Linear needs [out, in]",
                weight.rank()
            )));
        }
        Ok(Linear {
            base: Base::Dense(weight),
            adapted: None,
        })
    }
}

enum Base {
    Dense(Tensor),
    /// A native GGML projection. The `QMatMul` is constructed DIRECTLY as its `QTensor` arm:
    /// `QMatMul::from_arc` would honour `CANDLE_DEQUANTIZE_ALL[_F16]` and silently swap in a
    /// dense weight. Activations stay row-independent, so row chunking is exact.
    Ggml {
        matmul: QMatMul,
        storage: gen_core_flux3::artifacts::NativeDitStorage,
        out: usize,
        input: usize,
        compute: DType,
        device: Device,
    },
    /// A publisher FP8R/NVFP4 projection on its frozen-recipe CUDA kernel (BF16 compute).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Publisher(PublisherBase),
}

/// `nn.Linear(bias=False)`: `x · Wᵀ` in the compute dtype.
pub struct Linear {
    base: Base,
    adapted: Option<candle_gen::quant::AdaptLinear>,
}

impl Linear {
    /// A linear layer from an already-shaped `[out, in]` weight, stored as given.
    pub fn from_weight(weight: Tensor) -> Self {
        Self {
            base: Base::Dense(weight),
            adapted: None,
        }
    }

    fn from_ggml_rows(
        packed: &PackedSource,
        key: &str,
        compute: DType,
        device: &Device,
    ) -> Result<Self> {
        let (bytes, out, input) = packed.rows(key)?;
        packed
            .storage
            .check_contraction_width(key, input)
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        let qtensor = candle_gen::candle_core::quantized::ggml_file::qtensor_from_ggml(
            packed.ggml,
            &bytes,
            vec![out, input],
            device,
        )?;
        Ok(Self {
            base: Base::Ggml {
                matmul: QMatMul::QTensor(std::sync::Arc::new(qtensor)),
                storage: packed.storage,
                out,
                input,
                compute,
                device: device.clone(),
            },
            adapted: None,
        })
    }

    /// A packed projection over an in-memory `QTensor` (tests and converters).
    pub fn from_packed_qtensor(
        qtensor: candle_gen::candle_core::quantized::QTensor,
        storage: gen_core_flux3::artifacts::NativeDitStorage,
        compute: DType,
    ) -> Result<Self> {
        let (out, input) = qtensor.shape().dims2()?;
        storage
            .check_contraction_width("packed projection", input)
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        let device = qtensor.device();
        Ok(Self {
            base: Base::Ggml {
                matmul: QMatMul::QTensor(std::sync::Arc::new(qtensor)),
                storage,
                out,
                input,
                compute,
                device,
            },
            adapted: None,
        })
    }

    /// `x` of any leading shape `[.., in]` → `[.., out]` in the compute dtype.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = self.dtype();
        let x = to(x, dtype)?;
        let dims = x.dims().to_vec();
        let (last, lead) = dims
            .split_last()
            .ok_or_else(|| Error::Msg("flux3: a Linear input needs at least one axis".into()))?;
        let rows: usize = lead.iter().product();
        let (out_dim, _) = self.out_in()?;
        let flat = x.reshape((rows, *last))?;
        let gemm = kernel_dtype(flat.device(), dtype);
        let y = match &self.base {
            Base::Dense(weight) => {
                let y = to(&flat, gemm)?.matmul(&to(weight, gemm)?.t()?)?;
                to(&y, dtype)?
            }
            Base::Ggml {
                matmul,
                storage,
                out,
                input,
                compute,
                ..
            } => {
                if *last != *input {
                    return Err(Error::Msg(format!(
                        "flux3: packed Linear expects [.., {input}], got [.., {last}]"
                    )));
                }
                packed_forward(
                    matmul,
                    *storage,
                    &flat,
                    *out,
                    *compute,
                    gen_core_flux3::packed_gemm::PACKED_GEMM_TRANSIENT_BUDGET,
                )?
            }
            Base::Publisher(publisher) => {
                if *last != publisher.input {
                    return Err(Error::Msg(format!(
                        "flux3: packed Linear expects [.., {}], got [.., {last}]",
                        publisher.input
                    )));
                }
                publisher_forward(publisher, &flat)?
            }
        };
        let y = if let Some(adapted) = &self.adapted {
            adapted.apply_residuals(&to(&flat, gemm)?, y)?
        } else {
            y
        };
        let mut shape = lead.to_vec();
        shape.push(out_dim);
        __model_proof_output!(Ok(y.reshape(shape)?))
    }

    pub(crate) fn prepare_frozen_group(
        layers: &mut [&mut Self],
    ) -> Result<Option<candle_gen::train::projection::FrozenProjectionPack>> {
        // Preserve the CPU half-precision oracle and every packed/native base.
        if layers.iter().any(|l| !matches!(&l.base, Base::Dense(w) if !(w.track_op() || w.device().is_cpu() && w.dtype() != DType::F32))) {
            return Ok(None);
        }
        for layer in layers.iter_mut() {
            layer.adapted_mut()?;
        }
        let mut adapted = layers
            .iter_mut()
            .map(|l| l.adapted.as_mut().unwrap())
            .collect::<Vec<_>>();
        let pack =
            candle_gen::train::projection::FrozenProjectionPack::prepare_adapt(&mut adapted)?;
        if pack.is_some() {
            for layer in layers {
                layer.base = Base::Dense(
                    layer
                        .adapted
                        .as_ref()
                        .unwrap()
                        .dense_base()
                        .unwrap()
                        .weight()
                        .clone(),
                );
            }
        }
        Ok(pack)
    }

    pub(crate) fn forward_frozen_group(
        pack: &candle_gen::train::projection::FrozenProjectionPack,
        x: &Tensor,
        layers: &[&Self],
    ) -> Result<Vec<Tensor>> {
        let adapted = layers
            .iter()
            .map(|l| l.adapted.as_ref().expect("prepared dense facade"))
            .collect::<Vec<_>>();
        Ok(pack.forward_adapt(&to(x, layers[0].dtype())?, &adapted)?)
    }

    fn adapted_mut(&mut self) -> Result<&mut candle_gen::quant::AdaptLinear> {
        if self.adapted.is_none() {
            let (output, input) = self.out_in()?;
            self.adapted = Some(match &self.base {
                Base::Dense(weight) => candle_gen::quant::AdaptLinear::from_dense(
                    candle_gen::candle_nn::Linear::new(weight.clone(), None),
                    input,
                    output,
                ),
                // Residual storage only: the packed base is never dequantized or folded.
                Base::Ggml {
                    compute, device, ..
                } => candle_gen::quant::AdaptLinear::for_external_base(
                    input, output, *compute, device,
                )?,
                Base::Publisher(publisher) => candle_gen::quant::AdaptLinear::for_external_base(
                    input,
                    output,
                    DType::BF16,
                    &publisher.device,
                )?,
            });
        }
        Ok(self.adapted.as_mut().expect("initialized"))
    }

    /// Attach a validated bottleneck residual without mutating the base weight. On a packed
    /// base it rides as an additive residual over the strict packed GEMM output.
    pub fn push_lora(&mut self, down: Tensor, up: Tensor, scale: f64) -> Result<()> {
        self.adapted_mut()?
            .push_lora_checked(down.t()?.contiguous()?, up.t()?.contiguous()?, scale)
    }

    /// Training-only storage-sharing factors. Unlike inference's prepared-factor cache,
    /// `TrainableLora` reads the Var leaves on EVERY forward and transposes inside that graph.
    pub fn push_trainable_lora(&mut self, down: Tensor, up: Tensor, scale: f64) -> Result<()> {
        if self.is_quantized() {
            return Err(Error::Unsupported(
                "flux3 training: a packed GGML base projection has no differentiable dense weight; \
                 train on the original dense package"
                    .into(),
            ));
        }
        let (output, input) = self.out_in()?;
        let rank = down.dim(0)?;
        if down.dims() != [rank, input]
            || up.dims() != [output, rank]
            || rank == 0
            || !scale.is_finite()
        {
            return Err(Error::Unsupported(
                "flux3 training: invalid trainable LoRA shape/gain".into(),
            ));
        }
        self.adapted_mut()?.push_trainable_lora(down, up, scale);
        Ok(())
    }

    /// `[out, in]` of the projection.
    pub fn out_in(&self) -> Result<(usize, usize)> {
        match &self.base {
            Base::Dense(weight) => Ok(weight.dims2()?),
            Base::Ggml { out, input, .. } => Ok((*out, *input)),
            Base::Publisher(publisher) => Ok((publisher.out, publisher.input)),
        }
    }

    /// The dense weight. A packed base has none; it is never dequantized to answer this.
    pub fn weight(&self) -> Result<&Tensor> {
        match &self.base {
            Base::Dense(weight) => Ok(weight),
            Base::Ggml { storage, .. } | Base::Publisher(PublisherBase { storage, .. }) => {
                Err(Error::Unsupported(format!(
                    "flux3: {} projection has no dense weight; dequantizing it is refused",
                    storage
                        .key()
                        .map_err(|e| Error::Unsupported(e.to_string()))?
                )))
            }
        }
    }

    /// Whether the base is held in a packed native format.
    pub fn is_quantized(&self) -> bool {
        matches!(self.base, Base::Ggml { .. } | Base::Publisher(_))
    }

    /// The compute (input and output) dtype.
    pub fn dtype(&self) -> DType {
        match &self.base {
            Base::Dense(weight) => weight.dtype(),
            Base::Ggml { compute, .. } => *compute,
            Base::Publisher(_) => DType::BF16,
        }
    }
}

/// A publisher-format projection over BF16 rows, in the same exact row-chunk plan as the GGML path
/// (each row's activation quantization is independent, so chunking changes no output bit).
fn publisher_forward(publisher: &PublisherBase, flat: &Tensor) -> Result<Tensor> {
    use gen_core_flux3::packed_gemm;
    let (rows, input) = flat.dims2()?;
    let per_row = packed_gemm::transient_bytes_per_row(publisher.storage, input, publisher.out, 2)
        .map_err(|e| Error::Unsupported(e.to_string()))?;
    // The fixed per-launch padding (NVFP4) comes out of the budget first, so the WHOLE launch fits.
    let overhead = packed_gemm::launch_overhead_bytes(publisher.storage, input, publisher.out, 2)
        .map_err(|e| Error::Unsupported(e.to_string()))?;
    let plan = packed_gemm::plan_row_chunks(
        rows,
        input,
        publisher.out,
        per_row,
        packed_gemm::PACKED_GEMM_TRANSIENT_BUDGET.saturating_sub(overhead),
    )
    .map_err(|e| Error::Unsupported(e.to_string()))?;
    let flat = to(flat, DType::BF16)?;
    let mut parts = Vec::with_capacity(plan.len());
    for (start, len) in plan {
        let chunk = if len == rows {
            flat.contiguous()?
        } else {
            flat.narrow(0, start, len)?.contiguous()?
        };
        parts.push(publisher.forward_chunk(&chunk)?);
    }
    match parts.len() {
        0 => Ok(Tensor::zeros(
            (0, publisher.out),
            DType::BF16,
            flat.device(),
        )?),
        1 => Ok(parts.pop().expect("one chunk")),
        _ => Ok(Tensor::cat(&parts, 0)?),
    }
}

/// ONE planned launch's rows, widened to f32 AFTER narrowing: the f32 copy is a per-chunk
/// transient of exactly `len · k` elements (what `transient_bytes_per_row` prices), never a
/// whole-input copy held across every launch.
fn widened_chunk(flat: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    let rows = flat.dim(0)?;
    let chunk = if start == 0 && len == rows {
        flat.contiguous()?
    } else {
        flat.narrow(0, start, len)?.contiguous()?
    };
    to(&chunk, DType::F32)
}

/// The strict packed GEMM: the compute-dtype rows widened exactly to f32 (the quantized kernels'
/// input dtype), run in planned row chunks, and rounded ONCE to the compute dtype.
fn packed_forward(
    matmul: &QMatMul,
    storage: gen_core_flux3::artifacts::NativeDitStorage,
    flat: &Tensor,
    out: usize,
    compute: DType,
    budget: u64,
) -> Result<Tensor> {
    use candle_gen::candle_core::Module;
    use gen_core_flux3::packed_gemm;
    let (rows, input) = flat.dims2()?;
    let per_row =
        packed_gemm::transient_bytes_per_row(storage, input, out, compute.size_in_bytes() as u64)
            .map_err(|e| Error::Unsupported(e.to_string()))?;
    let plan = packed_gemm::plan_row_chunks(rows, input, out, per_row, budget)
        .map_err(|e| Error::Unsupported(e.to_string()))?;
    let mut parts = Vec::with_capacity(plan.len());
    for (start, len) in plan {
        let chunk = widened_chunk(flat, start, len)?;
        parts.push(to(&matmul.forward(&chunk)?, compute)?);
    }
    match parts.len() {
        0 => Ok(Tensor::zeros((0, out), compute, flat.device())?),
        1 => Ok(parts.pop().expect("one chunk")),
        _ => Ok(Tensor::cat(&parts, 0)?),
    }
}

/// `F.layer_norm(x, (dim,), eps)` without affine parameters, computed and returned in **f32**
/// (autocast's policy for `layer_norm`): biased variance over the last axis, as torch.
pub fn layer_norm_f32(x: &Tensor) -> Result<Tensor> {
    let x = to(x, DType::F32)?.contiguous()?;
    let dim = x.dim(D::Minus1)? as f64;
    let mean = (x.sum_keepdim(D::Minus1)? / dim)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = (centered.sqr()?.sum_keepdim(D::Minus1)? / dim)?;
    Ok(centered.broadcast_div(&(var + LN_EPS)?.sqrt()?)?)
}

/// `layernorm_modulation(x, scale, shift) = (1 + scale) · layer_norm(x) + shift`. The `1` is
/// created in `scale`'s dtype (torch's weak Python scalar): `1 + scale` rounds in the compute
/// dtype, and the product with the f32 norm promotes to f32, exactly as upstream. `scale` and
/// `shift` are `[N, hidden]` or `[1, hidden]`.
pub fn layernorm_modulation(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let one_plus = (scale + 1.0)?;
    let norm = layer_norm_f32(x)?;
    add(&mul(&norm, &one_plus)?, shift)
}

/// Rotation tables `(cos, sin)`, `[n, pairs]` f32, for [`rmsnorm_rope`] — the
/// [`gen_core_flux3::rope_cos_sin`] tables of the tokens, in token order.
///
/// [`gen_core_flux3::rope_cos_sin`]: gen_core_flux3::rope_cos_sin
pub struct Rope {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl Rope {
    /// The tables of `ids` under the DiT's `axes_dim` / `theta`, on `device`.
    pub fn new(ids: &[[f32; 4]], axes_dim: &[usize], theta: f64, device: &Device) -> Result<Self> {
        let (cos, sin) = gen_core_flux3::rope_cos_sin(ids, axes_dim, theta);
        let pairs: usize = axes_dim.iter().map(|d| d / 2).sum();
        let n = ids.len();
        Ok(Self {
            cos: Tensor::from_vec(cos, (n, pairs), device)?,
            sin: Tensor::from_vec(sin, (n, pairs), device)?,
        })
    }
}

/// `rmsnorm_rope(x, scale, freqs)`: per-head RMSNorm in f32 with an f32 gain, then the
/// adjacent-pair complex rotation in f32, returned in `x`'s dtype. `x` is `[B, H, L, D]`;
/// `rope` holds `[L, D/2]` tables.
pub fn rmsnorm_rope(x: &Tensor, scale: &Tensor, rope: &Rope) -> Result<Tensor> {
    rmsnorm_rope_impl(x, scale, rope, false)
}

/// Differentiable RMSNorm + RoPE. Candle's pinned `rms_norm` calls `apply_op2_no_bwd`,
/// so using it here would sever gradients to q/k adapters AND preceding blocks.
/// The pinned `rms_norm_slow` is the primitive-op equivalent with a real backward graph.
pub fn rmsnorm_rope_training(x: &Tensor, scale: &Tensor, rope: &Rope) -> Result<Tensor> {
    rmsnorm_rope_impl(x, scale, rope, true)
}

fn rmsnorm_rope_impl(x: &Tensor, scale: &Tensor, rope: &Rope, training: bool) -> Result<Tensor> {
    let (b, h, l, d) = x.dims4()?;
    let pairs = d / 2;
    let xf = to(x, DType::F32)?.contiguous()?;
    let normed = if training {
        candle_gen::train::norm::rms_norm_diff(&xf, scale, RMS_EPS as f64)?
    } else {
        candle_gen::candle_nn::ops::rms_norm(&xf, scale, RMS_EPS)?
    };
    let xr = normed.reshape((b, h, l, pairs, 2))?;
    let re = xr.narrow(4, 0, 1)?.squeeze(4)?;
    let im = xr.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = rope.cos.reshape((1, 1, l, pairs))?;
    let sin = rope.sin.reshape((1, 1, l, pairs))?;
    // freqs = [[cos, −sin], [sin, cos]]: out0 = cos·x0 − sin·x1, out1 = sin·x0 + cos·x1.
    let out_re = re.broadcast_mul(&cos)?.sub(&im.broadcast_mul(&sin)?)?;
    let out_im = re.broadcast_mul(&sin)?.add(&im.broadcast_mul(&cos)?)?;
    let out = Tensor::stack(&[out_re, out_im], 4)?.reshape((b, h, l, d))?;
    to(&out, x.dtype())
}

/// `packed_attention(q, k, v, bounds)`: per-sample SDPA over `[1, H, L, D]` (samples never attend
/// each other); returns `[L, H·D]` in token order and the compute dtype. An empty bound
/// contributes nothing (upstream skips it and leaves zeros, which the single stream never reads).
/// Every launch goes through the shared i32-overflow-safe query-row chunking.
///
/// The operands are rounded to the compute dtype and the attention math — the `q·kᵀ` scores, the
/// softmax and the `p·v` product — runs in **f32**, rounding once at the output. That is what the
/// fused attention kernels upstream's production path runs (cuDNN under autocast: bf16 operands,
/// f32 logits and softmax, one output rounding) and what the MLX twin's Metal SDPA computes; a
/// composed bf16 path that rounded the logits to bf16 before the softmax (a 0.125 ulp at a
/// 16-magnitude logit, 13 % on its probability) measures outside upstream's own bf16 distance on
/// the real DiT (sc-26235, 7.2 % vs 2.5 % of peak at 64²).
pub fn packed_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    bounds: &[(usize, usize)],
    compute: DType,
) -> Result<Tensor> {
    packed_attention_impl(q, k, v, bounds, compute, false)
}

/// Training-only differentiable softmax. The pinned `softmax_last_dim` explicitly
/// uses `apply_op1_no_bwd`, so inference's callback cannot carry q/k gradients.
pub fn packed_attention_training(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    bounds: &[(usize, usize)],
    compute: DType,
) -> Result<Tensor> {
    packed_attention_impl(q, k, v, bounds, compute, true)
}

fn packed_attention_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    bounds: &[(usize, usize)],
    compute: DType,
    training: bool,
) -> Result<Tensor> {
    let (_, h, l, d) = q.dims4()?;
    let kernel = DType::F32;
    let prep = |a: &Tensor| -> Result<Tensor> { Ok(to(&to(a, compute)?, kernel)?.contiguous()?) };
    let (q, k, v) = (prep(q)?, prep(k)?, prep(v)?);
    let scale = 1.0 / (d as f64).sqrt();
    let one = |start: usize, end: usize| -> Result<Tensor> {
        let (q, k, v) = if start == 0 && end == l {
            (q.clone(), k.clone(), v.clone())
        } else {
            (
                q.narrow(2, start, end - start)?.contiguous()?,
                k.narrow(2, start, end - start)?.contiguous()?,
                v.narrow(2, start, end - start)?.contiguous()?,
            )
        };
        let o = if training {
            candle_gen::fused_attention::training_sdpa_budgeted_bhsd(
                &q,
                &k,
                &v,
                scale,
                None,
                |scores| candle_gen::candle_nn::ops::softmax(scores, D::Minus1),
                candle_gen::ATTN_SCORES_BUDGET,
            )?
        } else {
            candle_gen::sdpa_budgeted_bhsd(
                &q,
                &k,
                &v,
                scale,
                None,
                softmax_last_dim,
                candle_gen::ATTN_SCORES_BUDGET,
            )?
        };
        Ok(to(&o, compute)?
            .transpose(1, 2)?
            .reshape((end - start, h * d))?)
    };
    let mut chunks = Vec::with_capacity(bounds.len());
    for &(start, end) in bounds {
        if end > start {
            chunks.push(one(start, end)?);
        }
    }
    match chunks.len() {
        0 => Err(Error::Msg(
            "flux3: packed attention over no tokens (every sample bound is empty)".into(),
        )),
        1 => Ok(chunks.pop().expect("one chunk")),
        _ => Ok(Tensor::cat(&chunks, 0)?),
    }
}

/// Shape check with a named error (checkpoint/config disagreement surfaces at load, by key).
pub fn expect_shape(name: &str, a: &Tensor, want: &[usize]) -> Result<()> {
    if a.dims() != want {
        return Err(Error::Msg(format!(
            "flux3: {name} has shape {:?}, the config implies {want:?}",
            a.dims()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::quantized::{GgmlDType, QTensor};
    use candle_gen::testkit::tensor_peak_rel;
    use gen_core_flux3::artifacts::NativeDitStorage;

    fn seeded(n: usize, k: usize, salt: f32) -> Tensor {
        let data: Vec<f32> = (0..n * k)
            .map(|i| (i as f32 * 0.37 + salt).sin() * 0.8 + (i as f32 * 0.011).cos() * 0.2)
            .collect();
        Tensor::from_vec(data, (n, k), &Device::Cpu).unwrap()
    }

    fn packed(w: &Tensor, storage: NativeDitStorage, compute: DType) -> (Linear, Tensor) {
        let ggml = crate::convert::ggml_dtype(storage).unwrap();
        let q = QTensor::quantize(w, ggml).unwrap();
        let dequant = QTensor::dequantize(&q, &Device::Cpu).unwrap();
        (
            Linear::from_packed_qtensor(q, storage, compute).unwrap(),
            dequant,
        )
    }

    /// The packed GEMM is the dense GEMM over the dequantized weight up to the dynamic Q8
    /// activation quantization the GGML kernels apply — and returns the compute dtype.
    ///
    /// *Mutations that red this:* skipping the f32 widening (CPU qmatmul rejects bf16), dropping
    /// the final rounding to the compute dtype, or transposing the packed weight.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn packed_forward_matches_the_dense_dequantized_reference() {
        let (n, k) = (48, 128);
        let w = seeded(n, k, 0.3);
        let x = seeded(7, k, 1.7).reshape((1, 7, k)).unwrap();
        for storage in [NativeDitStorage::GgmlQ4_1, NativeDitStorage::GgmlQ8_0] {
            for compute in [DType::F32, DType::BF16] {
                let (linear, dequant) = packed(&w, storage, compute);
                let y = linear.forward(&x).unwrap();
                assert_eq!(y.dims(), &[1, 7, n]);
                assert_eq!(y.dtype(), compute);
                let xr = to(&x, compute).unwrap().to_dtype(DType::F32).unwrap();
                let reference = xr
                    .reshape((7, k))
                    .unwrap()
                    .matmul(&dequant.t().unwrap())
                    .unwrap()
                    .reshape((1, 7, n))
                    .unwrap();
                let err = tensor_peak_rel(&y, &reference, None);
                assert!(err < 0.02, "{storage:?}/{compute:?}: {err}");
                assert!(linear.is_quantized());
                assert!(linear.weight().is_err(), "no dequantized dense weight");
                assert_eq!(linear.out_in().unwrap(), (n, k));
            }
        }
        // Q4_1 at an odd block count is refused, never run through the paired CPU kernel.
        let odd = QTensor::quantize(&seeded(8, 96, 0.1), GgmlDType::Q4_1).unwrap();
        assert!(Linear::from_packed_qtensor(odd, NativeDitStorage::GgmlQ4_1, DType::F32).is_err());

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED nn::tests::packed_forward_matches_the_dense_dequantized_reference");
    }

    /// CUDA: the GGML Q4_1/Q8_0 packed path stays packed ON DEVICE (a `QMatMul::QTensor` with CUDA
    /// storage, never a dequantized dense weight) and matches the dense dequantized reference up to
    /// the Q8_1 activation quantization the CUDA kernels apply (same bound as the CPU test).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn cuda_packed_ggml_path_stays_packed_and_matches_the_dequantized_reference() {
        let dev = Device::new_cuda(0).expect("CUDA device 0 (CUDA_VISIBLE_DEVICES selects GPU1)");
        let (n, k) = (48, 128);
        let w = seeded(n, k, 0.3);
        let x = seeded(7, k, 1.7);
        for storage in [NativeDitStorage::GgmlQ4_1, NativeDitStorage::GgmlQ8_0] {
            let ggml = crate::convert::ggml_dtype(storage).unwrap();
            let q_cpu = QTensor::quantize(&w, ggml).unwrap();
            let dequant = QTensor::dequantize(&q_cpu, &Device::Cpu).unwrap();
            let bytes = q_cpu.data().unwrap().into_owned();
            let q_gpu = candle_gen::candle_core::quantized::ggml_file::qtensor_from_ggml(
                ggml,
                &bytes,
                vec![n, k],
                &dev,
            )
            .unwrap();
            for compute in [DType::F32, DType::BF16] {
                let linear = Linear::from_packed_qtensor(
                    candle_gen::candle_core::quantized::ggml_file::qtensor_from_ggml(
                        ggml,
                        &bytes,
                        vec![n, k],
                        &dev,
                    )
                    .unwrap(),
                    storage,
                    compute,
                )
                .unwrap();
                let Base::Ggml { matmul, device, .. } = &linear.base else {
                    panic!("packed base")
                };
                assert!(device.is_cuda());
                match matmul {
                    QMatMul::QTensor(t) => assert!(t.device().is_cuda(), "packed codes on device"),
                    _ => panic!("{storage:?}: the packed path must not hold a dense weight"),
                }
                let y = linear
                    .forward(&x.to_device(&dev).unwrap())
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap();
                assert_eq!(y.dtype(), compute);
                let xr = to(&x, compute).unwrap().to_dtype(DType::F32).unwrap();
                let reference = xr.matmul(&dequant.t().unwrap()).unwrap();
                let err = tensor_peak_rel(&y, &reference, None);
                eprintln!("cuda {storage:?}/{compute:?}: peak-relative error {err:e}");
                assert!(err < 0.02, "{storage:?}/{compute:?}: {err}");
            }
            drop(q_gpu);
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED nn::tests::cuda_packed_ggml_path_stays_packed_and_matches_the_dequantized_reference");
    }

    /// The f32 widening is per planned launch: each chunk's f32 storage holds exactly its own
    /// `len · k` elements, so the priced per-row transient is the real one.
    ///
    /// *Mutation that reds this:* widening the whole `[rows, k]` input before narrowing (each
    /// chunk would then be a view over full-size f32 storage).
    #[test]
    fn packed_chunks_widen_only_their_own_rows() {
        let k = 64;
        let x = seeded(10, k, 2.1).to_dtype(DType::BF16).unwrap();
        for (start, len) in [(0, 3), (3, 3), (9, 1), (0, 10)] {
            let chunk = widened_chunk(&x, start, len).unwrap();
            assert_eq!(chunk.dims(), &[len, k]);
            assert_eq!(chunk.dtype(), DType::F32);
            let (storage, _) = chunk.storage_and_layout();
            let elements = match &*storage {
                candle_gen::candle_core::Storage::Cpu(cpu) => cpu.as_slice::<f32>().unwrap().len(),
                _ => unreachable!("CPU test"),
            };
            assert_eq!(
                elements,
                len * k,
                "chunk {start}+{len}: f32 copy is per chunk"
            );
        }
    }

    /// Row chunking changes no bit: every row's activation quantization is independent.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn row_chunked_packed_forward_is_bit_identical() {
        let w = seeded(32, 64, 0.9);
        let x = seeded(10, 64, 2.1);
        for storage in [NativeDitStorage::GgmlQ4_1, NativeDitStorage::GgmlQ8_0] {
            let (linear, _) = packed(&w, storage, DType::F32);
            let Base::Ggml { matmul, .. } = &linear.base else {
                panic!("packed base")
            };
            let per_row =
                gen_core_flux3::packed_gemm::transient_bytes_per_row(storage, 64, 32, 4).unwrap();
            let whole = packed_forward(matmul, storage, &x, 32, DType::F32, u64::MAX).unwrap();
            let chunked = packed_forward(matmul, storage, &x, 32, DType::F32, per_row * 3).unwrap();
            let a = whole.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(a, b);
            assert!(packed_forward(matmul, storage, &x, 32, DType::F32, per_row - 1).is_err());
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED nn::tests::row_chunked_packed_forward_is_bit_identical");
    }

    /// LoRA on a packed base is an additive residual over the strict packed output; the base is
    /// never folded, dequantized or trained.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn lora_on_a_packed_base_is_an_exact_additive_residual() {
        let (n, k, r) = (32, 64, 4);
        let w = seeded(n, k, 0.5);
        let x = seeded(3, k, 0.2);
        let down = seeded(r, k, 3.3);
        let up = seeded(n, r, 4.4);
        let (mut linear, _) = packed(&w, NativeDitStorage::GgmlQ8_0, DType::F32);
        let base = linear.forward(&x).unwrap();
        linear.push_lora(down.clone(), up.clone(), 0.75).unwrap();
        let adapted = linear.forward(&x).unwrap();
        let residual = (x
            .matmul(&down.t().unwrap())
            .unwrap()
            .matmul(&up.t().unwrap())
            .unwrap()
            * 0.75)
            .unwrap();
        let want = (&base + &residual).unwrap();
        assert!(tensor_peak_rel(&adapted, &want, None) < 1e-5);
        assert!(linear.is_quantized(), "the base stays packed");
        assert!(matches!(
            linear.push_trainable_lora(down, up, 1.0),
            Err(Error::Unsupported(_))
        ));
        assert!(linear
            .push_lora(seeded(r, k + 32, 0.0), seeded(n, r, 0.0), 1.0)
            .is_err());

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED nn::tests::lora_on_a_packed_base_is_an_exact_additive_residual");
    }

    /// Source guard (sc-26241): nothing in this workspace may flip candle's global DMMV switch
    /// (`FORCE_DMMV` dequantizes whole weights on CUDA) or route a FLUX3 packed weight through a
    /// constructor that honours `CANDLE_DEQUANTIZE_ALL`. The pieces are joined at run time so this
    /// test's own source never matches.
    #[test]
    fn no_dmmv_switch_or_dequantizing_qmatmul_constructor_is_reachable() {
        let force = ["set_force", "_dmmv("].concat();
        let env = ["CANDLE_DEQUANTIZE", "_ALL"].concat();
        let banned_here = [
            ["QMatMul::from", "_arc("].concat(),
            ["QMatMul::from", "_qtensor("].concat(),
            [".dequantize", "("].concat(),
            [".dequantize_f16", "("].concat(),
            [".forward_via", "_f16("].concat(),
        ];
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let mut stack = vec![crates.clone()];
        let mut scanned = 0;
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                if path.is_dir() {
                    if name != "vendor" && name != "target" && !name.starts_with('.') {
                        stack.push(path);
                    }
                } else if name.ends_with(".rs") {
                    let text = std::fs::read_to_string(&path).unwrap();
                    scanned += 1;
                    assert!(
                        !text.contains(&force),
                        "{} flips FORCE_DMMV",
                        path.display()
                    );
                    assert!(
                        !text.contains(&format!("set_var(\"{env}")),
                        "{} sets {env}",
                        path.display()
                    );
                }
            }
        }
        assert!(scanned > 100, "the scan must cover the workspace crates");
        let own = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for entry in std::fs::read_dir(&own).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                for banned in &banned_here {
                    assert!(
                        !text.contains(banned.as_str()),
                        "{} uses {banned}",
                        path.display()
                    );
                }
            }
        }
    }

    #[test]
    fn promotion_follows_torch() {
        let dev = Device::Cpu;
        let a = Tensor::new(&[1f32, 2.0], &dev).unwrap();
        let b = a.to_dtype(DType::BF16).unwrap();
        assert_eq!(add(&a, &b).unwrap().dtype(), DType::F32);
        assert_eq!(add(&b, &b).unwrap().dtype(), DType::BF16);
        assert_eq!(cat(&[&b, &a], 0).unwrap().dtype(), DType::F32);
        assert_eq!(mul(&b, &b).unwrap().dtype(), DType::BF16);
    }

    /// `layer_norm_f32` is torch's `F.layer_norm` (biased variance, eps inside the sqrt) and
    /// returns f32 whatever it was given.
    #[test]
    fn layer_norm_is_torch_layer_norm_in_f32() {
        let dev = Device::Cpu;
        let x = Tensor::new(&[[1f32, 2.0, 4.0, 7.0]], &dev).unwrap();
        let y = layer_norm_f32(&x.to_dtype(DType::BF16).unwrap()).unwrap();
        assert_eq!(y.dtype(), DType::F32);
        let got = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // mean 3.5, biased var 5.25
        let want: Vec<f32> = [1f32, 2.0, 4.0, 7.0]
            .iter()
            .map(|v| (v - 3.5) / (5.25f32 + 1e-6).sqrt())
            .collect();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{got:?} vs {want:?}");
        }
    }

    /// `(1 + scale)` is formed in the modulation's dtype: under bf16 it rounds there, which is
    /// visible against the f32 form for a scale below bf16's resolution at 1.
    #[test]
    fn one_plus_scale_rounds_in_the_modulation_dtype() {
        let dev = Device::Cpu;
        let x = Tensor::new(&[[1f32, -1.0]], &dev).unwrap();
        let shift = Tensor::zeros((1, 2), DType::BF16, &dev).unwrap();
        let scale = Tensor::new(&[[0.001f32, 0.001]], &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let out = layernorm_modulation(&x, &scale, &shift).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // bf16(1 + 0.001) == 1.0 exactly (bf16 has 8 significand bits), so the norm is unscaled.
        let norm = layer_norm_f32(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got, norm);
    }

    #[test]
    fn rope_at_position_zero_is_identity_and_rotates_elsewhere() {
        let dev = Device::Cpu;
        let ids = [[0.0f32, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let rope = Rope::new(&ids, &[2, 2, 2, 2], 10_000.0, &dev).unwrap();
        let scale = Tensor::ones(8, DType::F32, &dev).unwrap();
        let x = Tensor::new(&[1f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &dev)
            .unwrap()
            .reshape((1, 1, 1, 8))
            .unwrap();
        let x2 = Tensor::cat(&[&x, &x], 2).unwrap();
        let y = rmsnorm_rope(&x2, &scale, &rope).unwrap();
        let rows = y.reshape((2, 8)).unwrap().to_vec2::<f32>().unwrap();
        // RMS of [1,0,1,0,1,0,1,0] is sqrt(0.5): every odd slot is 0 at position 0.
        let r = (0.5f32).sqrt().recip();
        for (i, v) in rows[0].iter().enumerate() {
            let want = if i % 2 == 0 { r } else { 0.0 };
            assert!((v - want).abs() < 1e-5, "{:?}", rows[0]);
        }
        // Position 1 on the last axis rotates the last axis's pair (slots 6, 7) by 1 rad.
        assert!((rows[1][6] - r * 1f32.cos()).abs() < 1e-5);
        assert!((rows[1][7] - r * 1f32.sin()).abs() < 1e-5);
        assert!((rows[1][0] - r).abs() < 1e-5, "other axes stay at 0");
    }

    /// Per-sample bounds: a token never attends across its bound.
    ///
    /// *Mutation that reds this:* attending over the whole packed stream.
    #[test]
    fn packed_attention_is_per_sample() {
        let dev = Device::Cpu;
        let n = 4;
        let data: Vec<f32> = (0..n * 2).map(|i| (i as f32 * 0.7).sin()).collect();
        let q = Tensor::from_vec(data.clone(), (1, 1, n, 2), &dev).unwrap();
        let k =
            Tensor::from_vec(data.iter().map(|v| v * 1.3).collect(), (1, 1, n, 2), &dev).unwrap();
        let v = Tensor::from_vec(
            data.iter().map(|v| v * 0.5 + 1.0).collect(),
            (1, 1, n, 2),
            &dev,
        )
        .unwrap();
        let split = packed_attention(&q, &k, &v, &[(0, 2), (2, 4)], DType::F32).unwrap();
        let whole = packed_attention(&q, &k, &v, &[(0, 4)], DType::F32).unwrap();
        assert_eq!(split.dims(), &[4, 2]);
        let first_alone = packed_attention(
            &q.narrow(2, 0, 2).unwrap(),
            &k.narrow(2, 0, 2).unwrap(),
            &v.narrow(2, 0, 2).unwrap(),
            &[(0, 2)],
            DType::F32,
        )
        .unwrap();
        let d = (split.narrow(0, 0, 2).unwrap() - &first_alone)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d < 1e-6, "sample 0 is the standalone attention ({d})");
        let diff = (&split - &whole)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-3, "the bounds must change the result ({diff})");
        assert!(matches!(
            packed_attention(&q, &k, &v, &[(1, 1)], DType::F32),
            Err(Error::Msg(_))
        ));
    }
}
