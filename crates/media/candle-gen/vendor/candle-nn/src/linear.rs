//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```no_run
//! use candle::{Tensor, Device::Cpu};
//! use candle_nn::{Linear, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Cpu)?;
//! let layer = Linear::new(w, None); // Use no bias.
//! let xs = Tensor::new(&[[10f32, 100.]], &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(ys.to_vec2::<f32>()?, &[[210.0, 430.0, 650.0]]);
//! # Ok(()) }
//! ```
use candle::{Result, Tensor};

use std::sync::{Arc, OnceLock};

/// A frozen, checkpoint-native matrix implementing unbiased x Wᵀ.
/// Operators requiring the actual dense matrix can request its decoded weight explicitly.
pub trait NativeLinearWeight: Send + Sync {
    fn forward(&self, x: &Tensor, bias: Option<&Tensor>) -> Result<Tensor>;
    fn dense_weight(&self) -> Result<Tensor>;
    /// Called when the owning Linear caches a coefficient copy in addition to this operand.
    fn retained_coefficients(&self, _tensor: &Tensor) -> Result<()> {
        Ok(())
    }
    fn compute_dtype(&self) -> candle::DType {
        candle::DType::BF16
    }
    fn relocated(&self, device: &candle::Device) -> Result<Linear> {
        candle::bail!(
            "{} does not support relocation to {device:?}",
            self.format_id()
        )
    }
    fn format_id(&self) -> &'static str;
    fn resident_bytes(&self) -> u64;
    fn dims(&self) -> (usize, usize);
    fn device(&self) -> &candle::Device;
}

#[derive(Clone)]
enum Weight {
    Dense(Tensor),
    Native {
        projection: Arc<dyn NativeLinearWeight>,
        decoded: Arc<OnceLock<Tensor>>,
    },
}

#[derive(Clone)]
pub struct Linear {
    weight: Weight,
    bias: Option<Tensor>,
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("dimensions", &self.weight_dims())
            .field("native", &self.is_native())
            .field("bias", &self.bias)
            .finish()
    }
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            weight: Weight::Dense(weight),
            bias,
        }
    }

    /// Read the checkpoint's own matrix geometry. Dimensioned constructors continue validating
    /// the caller's expectation; this form is for architectures whose matrix width is inferred.
    pub fn from_var_builder(vb: &crate::VarBuilder<'_>, with_bias: bool) -> Result<Self> {
        let bias = with_bias.then(|| vb.get_unchecked("bias")).transpose()?;
        if let Some(op) = vb.native_linear_weight("weight")? {
            return Ok(Self::from_native(op, bias));
        }
        let weight = vb.get_unchecked("weight")?;
        weight.dims2()?;
        Ok(Self::new(weight, bias))
    }

    pub fn from_native(projection: Arc<dyn NativeLinearWeight>, bias: Option<Tensor>) -> Self {
        Self {
            weight: Weight::Native {
                projection,
                decoded: Arc::new(OnceLock::new()),
            },
            bias,
        }
    }

    pub fn is_native(&self) -> bool {
        matches!(self.weight, Weight::Native { .. })
    }

    pub fn native_weight(&self) -> Option<Arc<dyn NativeLinearWeight>> {
        match &self.weight {
            Weight::Native { projection, .. } => Some(projection.clone()),
            Weight::Dense(_) => None,
        }
    }
    pub fn native_format_id(&self) -> Option<&'static str> {
        match &self.weight {
            Weight::Native { projection, .. } => Some(projection.format_id()),
            Weight::Dense(_) => None,
        }
    }
    pub fn native_resident_bytes(&self) -> Option<u64> {
        match &self.weight {
            Weight::Native {
                projection,
                decoded,
            } => Some(
                projection.resident_bytes()
                    + decoded
                        .get()
                        .map_or(0, |t| (t.elem_count() * t.dtype().size_in_bytes()) as u64),
            ),
            Weight::Dense(_) => None,
        }
    }

    pub fn weight_dims(&self) -> Result<(usize, usize)> {
        match &self.weight {
            Weight::Dense(t) => t.dims2(),
            Weight::Native { projection, .. } => Ok(projection.dims()),
        }
    }

    pub fn device(&self) -> &candle::Device {
        match &self.weight {
            Weight::Dense(t) => t.device(),
            Weight::Native { projection, .. } => projection.device(),
        }
    }

    /// Obtain the true, decoded weight for operators needing its individual coefficients.
    pub fn try_weight(&self) -> Result<&Tensor> {
        match &self.weight {
            Weight::Dense(t) => Ok(t),
            Weight::Native {
                projection,
                decoded,
            } => {
                if decoded.get().is_none() {
                    let _ = decoded.set(projection.dense_weight()?);
                }
                let tensor = decoded.get().expect("decoded weight initialized above");
                projection.retained_coefficients(tensor)?;
                Ok(tensor)
            }
        }
    }

    /// Upstream compatibility accessor. Use try_weight for fallible native materialization.
    pub fn weight(&self) -> &Tensor {
        self.try_weight()
            .expect("failed to materialize the linear weight")
    }

    pub fn weight_dtype(&self) -> candle::DType {
        match &self.weight {
            Weight::Dense(t) => t.dtype(),
            Weight::Native { projection, .. } => projection.compute_dtype(),
        }
    }
    pub fn with_bias(mut self, bias: Option<Tensor>) -> Self {
        self.bias = bias;
        self
    }
    pub fn to_device(&self, device: &candle::Device) -> Result<Self> {
        if self.device().same_device(device) {
            return Ok(self.clone());
        }
        let base = match &self.weight {
            Weight::Dense(t) => Self::new(t.to_device(device)?, None),
            Weight::Native { projection, .. } => projection.relocated(device)?,
        };
        Ok(base.with_bias(
            self.bias
                .as_ref()
                .map(|b| b.to_device(device))
                .transpose()?,
        ))
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl super::Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let weight = match &self.weight {
            Weight::Dense(weight) => weight,
            Weight::Native { projection, .. } => {
                return projection.forward(x, self.bias.as_ref());
            }
        };
        // When possible, we avoid using a broadcasted matmul as it is much slower
        // than the standard matmul for the cuda and cpu backends.
        let x = match *x.dims() {
            [b1, b2, m, k] => {
                if x.is_contiguous() {
                    let w = weight.t()?;
                    x.reshape((b1 * b2 * m, k))?
                        .matmul(&w)?
                        .reshape((b1, b2, m, ()))?
                } else {
                    let w = weight.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&w)?
                }
            }
            [bsize, m, k] => {
                if x.is_contiguous() {
                    let w = weight.t()?;
                    x.reshape((bsize * m, k))?
                        .matmul(&w)?
                        .reshape((bsize, m, ()))?
                } else {
                    let w = weight.broadcast_left(bsize)?.t()?;
                    x.matmul(&w)?
                }
            }
            _ => {
                let w = weight.t()?;
                x.matmul(&w)?
            }
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

/// Create or initialize a new linear layer.
///
/// This uses some default names for weights and biases, namely `"weight"` and `"bias"`.
pub fn linear(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    let mut layer = linear_no_bias(in_dim, out_dim, vb.clone())?;
    let bound = 1. / (in_dim as f64).sqrt();
    let bs = vb.get_with_hints(
        out_dim,
        "bias",
        crate::Init::Uniform {
            lo: -bound,
            up: bound,
        },
    )?;
    layer.bias = Some(bs);
    Ok(layer)
}

/// Create or initialize a new linear layer without biases.
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    if let Some(native) = vb.native_linear_weight_with_shape("weight", Some((out_dim, in_dim)))? {
        if native.dims() != (out_dim, in_dim) {
            candle::bail!(
                "native weight {} expected [{out_dim},{in_dim}], got {:?}",
                vb.prefix(),
                native.dims()
            );
        }
        return Ok(Linear::from_native(native, None));
    }
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws, None))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: crate::VarBuilder,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}
