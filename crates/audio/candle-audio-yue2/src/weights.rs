//! Loaded YuE2 matmul weights (sc-22995): dense, GGML block-quantized, or — on CUDA, in the
//! experimental FP8 AR mode only — FP8 E4M3 ([`crate::fp8`]); and the loader that reads each
//! checkpoint tensor in the representation its tier stores ([`crate::precision`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use candle_audio::candle_core::quantized::{QMatMul, QTensor};
use candle_audio::candle_core::{DType, Device, Module, Tensor};
use candle_llm::primitives::linear;
use candle_llm::primitives::quant::{from_ggml_block_tensor, to_ggml_block_tensor};
use candle_nn::VarBuilder;

use crate::precision::{Backend, Storage};

type CResult<T> = candle_audio::candle_core::Result<T>;

fn candle_err(e: impl std::fmt::Display) -> candle_audio::candle_core::Error {
    candle_audio::candle_core::Error::Msg(e.to_string())
}

/// The backend a device belongs to (for residency pricing).
pub fn backend_of(device: &Device) -> Backend {
    match device {
        Device::Cpu => Backend::Cpu,
        Device::Cuda(_) => Backend::Cuda,
        Device::Metal(_) => Backend::Metal,
    }
}

/// A matmul weight `[out, in]` as loaded.
pub enum Proj {
    /// A dense weight in the compute dtype.
    Dense(Tensor),
    /// A GGML block-quantized weight, multiplied by Candle's quantized matmul.
    Ggml(Arc<QTensor>),
    /// An FP8 E4M3 weight of the experimental FP8 AR mode (CUDA only).
    #[cfg(feature = "cuda")]
    Fp8(Box<crate::fp8::Fp8Weight>),
}

impl std::fmt::Debug for Proj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Proj::Dense(t) => f
                .debug_tuple("Dense")
                .field(&t.dtype())
                .field(&t.dims())
                .finish(),
            Proj::Ggml(q) => f
                .debug_tuple("Ggml")
                .field(&q.dtype())
                .field(&q.shape().dims())
                .finish(),
            #[cfg(feature = "cuda")]
            Proj::Fp8(w) => f.debug_tuple("Fp8").field(&w.shape()).finish(),
        }
    }
}

impl Proj {
    /// `x @ weightᵀ (+ bias)`, in `x`'s dtype. A GGML weight takes an F32 activation (Candle's
    /// quantized matmul), so the activation is converted in and the result converted back.
    pub fn forward(&self, x: &Tensor, bias: Option<&Tensor>) -> CResult<Tensor> {
        match self {
            Proj::Dense(w) => linear(x, w, bias).map_err(candle_err),
            Proj::Ggml(q) => {
                let dtype = x.dtype();
                let y = QMatMul::QTensor(Arc::clone(q))
                    .forward(&x.to_dtype(DType::F32)?.contiguous()?)?
                    .to_dtype(dtype)?;
                match bias {
                    Some(b) => y.broadcast_add(&b.to_dtype(dtype)?),
                    None => Ok(y),
                }
            }
            #[cfg(feature = "cuda")]
            Proj::Fp8(w) => w.forward(x, bias),
        }
    }

    /// The same weight on `device`: a dense tensor is copied, a GGML weight is rebuilt there from
    /// its exact blocks (never dequantized). An FP8 weight never moves (the FP8 mode is restored to
    /// BF16 before anything offloads).
    pub fn to_device(&self, device: &Device) -> CResult<Self> {
        match self {
            Proj::Dense(t) => Ok(Proj::Dense(t.to_device(device)?)),
            Proj::Ggml(q) => {
                let stored = to_ggml_block_tensor(q).map_err(candle_err)?;
                Ok(Proj::Ggml(Arc::new(
                    from_ggml_block_tensor(&stored, device).map_err(candle_err)?,
                )))
            }
            #[cfg(feature = "cuda")]
            Proj::Fp8(_) => Err(candle_err(
                "YuE2: an FP8 AR weight cannot move between devices; restore the BF16 AR path first",
            )),
        }
    }

    /// The device the weight lives on.
    pub fn device(&self) -> Device {
        match self {
            Proj::Dense(t) => t.device().clone(),
            Proj::Ggml(q) => q.device(),
            #[cfg(feature = "cuda")]
            Proj::Fp8(w) => w.device().clone(),
        }
    }

    /// The logical `[out, in]` shape.
    pub fn dims(&self) -> Vec<usize> {
        match self {
            Proj::Dense(t) => t.dims().to_vec(),
            Proj::Ggml(q) => q.shape().dims().to_vec(),
            #[cfg(feature = "cuda")]
            Proj::Fp8(w) => {
                let (n, k) = w.shape();
                vec![n, k]
            }
        }
    }

    /// How the weight is held: `bf16`/`float32` for a dense one, the GGML label, or `fp8_e4m3`.
    pub fn label(&self) -> &'static str {
        match self {
            Proj::Dense(t) => match t.dtype() {
                DType::BF16 => "bf16",
                DType::F32 => "float32",
                DType::F16 => "float16",
                _ => "dense_other",
            },
            Proj::Ggml(q) => Storage::Ggml(q.dtype()).label(),
            #[cfg(feature = "cuda")]
            Proj::Fp8(_) => "fp8_e4m3",
        }
    }

    /// Resident bytes on its device: a dense tensor's elements; a GGML weight's blocks plus, on
    /// CUDA, the row padding Candle allocates past them (its storage API reports the payload
    /// only); an FP8 weight's bytes plus its F32 scale.
    pub fn resident_bytes(&self) -> u64 {
        match self {
            Proj::Dense(t) => (t.elem_count() * t.dtype().size_in_bytes()) as u64,
            Proj::Ggml(q) => crate::precision::loaded_bytes(
                Storage::Ggml(q.dtype()),
                q.shape().dims(),
                backend_of(&q.device()),
                DType::F32,
            ),
            #[cfg(feature = "cuda")]
            Proj::Fp8(w) => w.resident_bytes(),
        }
    }

    /// The dense tensor, when the weight is dense.
    pub fn dense(&self) -> Option<&Tensor> {
        match self {
            Proj::Dense(t) => Some(t),
            _ => None,
        }
    }
}

/// Reads checkpoint tensors in the representation the loaded tier stores them in.
///
/// `storage` is the tier's precision plan by tensor name (`None`: the released BF16 checkpoint,
/// every tensor dense). A dense tensor is read through `vb` (converted to the compute dtype on the
/// model device); a GGML block tensor is read raw on the host and rebuilt on the model device
/// exactly as stored.
#[derive(Clone)]
pub(crate) struct Loader<'a> {
    vb: VarBuilder<'a>,
    storage: Option<Arc<BTreeMap<String, Storage>>>,
}

impl<'a> Loader<'a> {
    /// A loader over `vb` for the tier whose plan is `storage` (`None` = all BF16).
    pub(crate) fn new(vb: VarBuilder<'a>, storage: Option<Arc<BTreeMap<String, Storage>>>) -> Self {
        Self { vb, storage }
    }

    /// The compute dtype.
    pub(crate) fn dtype(&self) -> DType {
        self.vb.dtype()
    }

    /// The model device.
    pub(crate) fn device(&self) -> &Device {
        self.vb.device()
    }

    fn planned(&self, name: &str) -> CResult<Storage> {
        match &self.storage {
            None => Ok(Storage::Bf16),
            Some(plan) => plan.get(name).copied().ok_or_else(|| {
                candle_err(format!(
                    "YuE2 tier: `{name}` is not in the verified precision plan"
                ))
            }),
        }
    }

    /// A dense tensor of `shape` (`name` is the full checkpoint name). Asking for a tensor the tier
    /// stores quantized is an error.
    pub(crate) fn tensor(
        &self,
        name: &str,
        shape: impl Into<candle_audio::candle_core::Shape>,
    ) -> CResult<Tensor> {
        match self.planned(name)? {
            Storage::Bf16 => self.vb.get(shape, name),
            Storage::Ggml(d) => Err(candle_err(format!(
                "YuE2 tier: `{name}` is stored {d:?}; it is not a dense tensor"
            ))),
        }
    }

    /// The matmul weight `name` of logical shape `(out, in)`.
    pub(crate) fn matrix(&self, name: &str, (out, inp): (usize, usize)) -> CResult<Proj> {
        match self.planned(name)? {
            Storage::Bf16 => Ok(Proj::Dense(self.vb.get((out, inp), name)?)),
            storage @ Storage::Ggml(dtype) => {
                let (_, want) = storage.stored(&[out, inp]);
                let host = self.vb.clone().set_device(Device::Cpu);
                let stored = host.get_unchecked_dtype(name, DType::U8)?;
                if stored.dims() != want.as_slice() {
                    return Err(candle_err(format!(
                        "YuE2 tier: `{name}` is stored {:?}, expected the {dtype:?} blocks {want:?} \
                         of a [{out}, {inp}] weight",
                        stored.dims()
                    )));
                }
                let q = from_ggml_block_tensor(&stored, self.device()).map_err(candle_err)?;
                if q.dtype() != dtype {
                    return Err(candle_err(format!(
                        "YuE2 tier: `{name}` holds {:?} blocks, the plan says {dtype:?}",
                        q.dtype()
                    )));
                }
                Ok(Proj::Ggml(Arc::new(q)))
            }
        }
    }
}
