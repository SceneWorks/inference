//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```rust
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
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn dense_weight(&self) -> Result<Tensor>;
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
                Ok(decoded.get().expect("decoded weight initialized above"))
            }
        }
    }

    /// Upstream compatibility accessor. Use try_weight for fallible native materialization.
    pub fn weight(&self) -> &Tensor {
        self.try_weight()
            .expect("failed to materialize the linear weight")
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
                let y = projection.forward(x)?;
                return match &self.bias {
                    Some(b) => y.broadcast_add(&b.to_dtype(y.dtype())?),
                    None => Ok(y),
                };
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
    if let Some(native) = vb.native_linear_weight("weight")? {
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
