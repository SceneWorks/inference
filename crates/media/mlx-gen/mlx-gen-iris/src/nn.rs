//! The leaf modules of the Iris graph and the precision policy they share.
//!
//! **Precision policy** (mirrors upstream's CUDA path, `torch.autocast(bfloat16)` over FP32
//! parameters, and degenerates to upstream's FP32 CPU path when the compute dtype is f32):
//!
//! * every `Linear` casts its input to the compute dtype and returns the compute dtype (autocast's
//!   matmul rule) — its weight and bias are stored in the compute dtype;
//! * every `RMSNorm` normalises in f32 and multiplies by an f32 gain, so it returns f32 (upstream's
//!   `RMSNorm` computes in fp32 and its fp32 gain promotes the product);
//! * attention runs in the compute dtype (`scaled_dot_product_attention` is on autocast's
//!   low-precision list); RoPE rotates in f32 and returns the dtype it was given;
//! * every other elementwise op follows ordinary type promotion, exactly as torch does, so the
//!   residual streams stay f32 where upstream's do and the PiT pixel stream stays in the compute
//!   dtype where upstream's does.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::gen_core::iris;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::fast::{rms_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{concatenate_axis, stack_axis};
use mlx_rs::{Array, Dtype};

/// Builds modules from a checkpoint at one compute dtype.
pub struct Loader<'a> {
    pub weights: &'a Weights,
    pub compute: Dtype,
    /// The adapted projections of a load that installed LoRA/LoKr adapters (see
    /// [`AdaptedLinears`]); `None` builds every projection bare from `weights`.
    pub adapted: Option<&'a AdaptedLinears>,
}

/// The checkpoint's projections after an adapter install, keyed by upstream module path
/// (`blocks.3.attn_proj`): each is the [`AdaptableLinear`] the install acted on — the dense base
/// (with any diff-patch delta folded in) plus its stack of forward-time adapter residuals. The
/// [`Loader`] hands a projection out of here instead of rebuilding it from the raw tensor, and
/// records which paths it consumed, so an adapter that landed on a 2-D tensor the graph does not
/// read as a projection is a load error rather than a silently dropped residual.
pub struct AdaptedLinears {
    linears: RefCell<BTreeMap<String, AdaptableLinear>>,
    adapted: BTreeSet<String>,
    consumed: RefCell<BTreeSet<String>>,
}

impl AdaptedLinears {
    /// `linears` are every projection the install could reach; `adapted` the paths it changed.
    pub fn new(linears: BTreeMap<String, AdaptableLinear>, adapted: BTreeSet<String>) -> Self {
        Self {
            linears: RefCell::new(linears),
            adapted,
            consumed: RefCell::new(BTreeSet::new()),
        }
    }

    /// Adapted paths no module consumed as a projection — must be empty after a model build.
    pub fn unconsumed(&self) -> Vec<String> {
        let consumed = self.consumed.borrow();
        self.adapted.difference(&consumed).cloned().collect()
    }
}

impl Loader<'_> {
    fn require(&self, key: &str) -> Result<&Array> {
        self.weights.require(key)
    }

    /// A tensor kept in f32 (norm gains, adaLN biases, the text position table).
    pub fn f32(&self, key: &str) -> Result<Array> {
        Ok(self.require(key)?.as_dtype(Dtype::Float32)?)
    }

    pub fn linear(&self, prefix: &str, bias: bool) -> Result<Linear> {
        if let Some(adapted) = self.adapted {
            if let Some(inner) = adapted.linears.borrow_mut().remove(prefix) {
                // Same keys a bare build reads (so the checkpoint's unused-key accounting holds).
                self.require(&format!("{prefix}.weight"))?;
                if bias {
                    self.require(&format!("{prefix}.bias"))?;
                }
                if inner.bias().is_some() != bias {
                    return Err(Error::Msg(format!(
                        "iris: the projection {prefix} has a bias mismatch after the adapter install"
                    )));
                }
                adapted.consumed.borrow_mut().insert(prefix.to_owned());
                // Every adapter factor is read and evaluated here, at the load boundary, so no
                // forward ever waits on a lazy adapter read.
                inner.materialize_adapters()?;
                return Ok(Linear {
                    inner,
                    compute: self.compute,
                });
            }
        }
        let weight = self
            .require(&format!("{prefix}.weight"))?
            .as_dtype(self.compute)?;
        let bias = if bias {
            Some(
                self.require(&format!("{prefix}.bias"))?
                    .as_dtype(self.compute)?,
            )
        } else {
            None
        };
        Ok(Linear {
            inner: AdaptableLinear::dense(weight, bias),
            compute: self.compute,
        })
    }

    pub fn norm(&self, prefix: &str, eps: f32) -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: self.f32(&format!("{prefix}.weight"))?,
            eps,
        })
    }
}

/// `nn.Linear` (`[out, in]` weight, stored in the compute dtype), as the repo's
/// [`AdaptableLinear`]: the bare forward is `addmm(b, x, Wᵀ)` / `x·Wᵀ` exactly as before adapters
/// existed, and an installed LoRA/LoKr adds its residual `Σ adapter(x)` on top (LoKr through the
/// structured Kronecker product, never a materialized `[out, in]` delta).
pub struct Linear {
    inner: AdaptableLinear,
    compute: Dtype,
}

impl Linear {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = x.as_dtype(self.compute)?;
        self.inner.forward(&x)
    }

    /// The dense base parameters (adapter factors are evaluated at load by the [`Loader`]).
    pub fn arrays(&self) -> Vec<&Array> {
        match self.inner.dense_weight() {
            Some((weight, bias)) => std::iter::once(weight).chain(bias).collect(),
            None => Vec::new(),
        }
    }
}

/// `iris3b.nn.norms.RMSNorm`: f32 normalisation, f32 gain.
pub struct RmsNorm {
    weight: Array,
    eps: f32,
}

impl RmsNorm {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        Ok(rms_norm(
            x.as_dtype(Dtype::Float32)?,
            &self.weight,
            self.eps,
        )?)
    }

    pub fn arrays(&self) -> Vec<&Array> {
        vec![&self.weight]
    }
}

/// `x · (1 + scale) + shift`. The `1` is created in `scale`'s dtype — torch's weak Python scalar —
/// so a bf16 stream stays bf16 (an f32 `Array` scalar would promote it).
pub fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array> {
    let one = Array::from_f32(1.0).as_dtype(scale.dtype())?;
    Ok(x.multiply(&scale.add(&one)?)?.add(shift)?)
}

/// Split the last axis into `n` equal chunks (`tensor.chunk(n, dim=-1)`).
pub fn chunk_last(x: &Array, n: i32) -> Result<Vec<Array>> {
    let axis = x.ndim() as i32 - 1;
    Ok(x.split(n, axis)?)
}

/// Rotation tables `(cos, sin)`, `[n, pairs]` f32, for `apply_rope`.
pub struct Rope {
    pub cos: Array,
    pub sin: Array,
}

impl Rope {
    fn from_angles(angles: &[f32], n: usize, pairs: usize) -> Self {
        let cos: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
        let sin: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
        let shape = [n as i32, pairs as i32];
        Self {
            cos: Array::from_slice(&cos, &shape),
            sin: Array::from_slice(&sin, &shape),
        }
    }

    /// `rope_2d(head_dim, height, width, theta, scale, aspect="isotropic", frame_pairs=0)`:
    /// `head_dim/4` frequencies, interleaved (x, y) per pair along the last axis, row-major grid,
    /// both axes at one step `scale / (max(h, w) − 1)`.
    pub fn grid_2d(head_dim: usize, height: usize, width: usize, theta: f32, scale: f32) -> Self {
        let angles = iris::rope_2d_angles(head_dim, height, width, theta, scale);
        Self::from_angles(&angles, height * width, 2 * (head_dim / 4))
    }

    /// `rope_1d(head_dim, length, theta)`: standard 1-D RoPE over integer positions.
    pub fn line_1d(head_dim: usize, length: usize, theta: f32) -> Self {
        let angles = iris::rope_1d_angles(head_dim, length, theta);
        Self::from_angles(&angles, length, head_dim / 2)
    }

    /// Rotate `x` `[B, N, H, head_dim]` by complex multiplication over **adjacent** channel pairs
    /// (`view_as_complex`), in f32; returns `x`'s dtype.
    pub fn apply(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n, h, d) = (sh[0], sh[1], sh[2], sh[3]);
        let pairs = d / 2;
        let xf = x.as_dtype(Dtype::Float32)?.reshape(&[b, n, h, pairs, 2])?;
        let parts = xf.split(2, 4)?;
        let (re, im) = (parts[0].squeeze_axes(&[4])?, parts[1].squeeze_axes(&[4])?);
        let cos = self.cos.reshape(&[1, n, 1, pairs])?;
        let sin = self.sin.reshape(&[1, n, 1, pairs])?;
        let out_re = re.multiply(&cos)?.subtract(&im.multiply(&sin)?)?;
        let out_im = re.multiply(&sin)?.add(&im.multiply(&cos)?)?;
        let out = stack_axis(&[out_re, out_im], 4)?.reshape(&[b, n, h, d])?;
        Ok(out.as_dtype(x.dtype())?)
    }
}

/// Grouped-query SDPA over `[B, N, heads, head_dim]` q and `[B, N, kv_heads, head_dim]` k/v, in
/// the compute dtype; returns `[B, N, heads·head_dim]`. `mask` is additive `[B|1, 1, N, N]`.
pub fn attention(
    q: &Array,
    k: &Array,
    v: &Array,
    compute: Dtype,
    mask: Option<&Array>,
) -> Result<Array> {
    let sh = q.shape();
    let (b, n, h, d) = (sh[0], sh[1], sh[2], sh[3]);
    let t =
        |a: &Array| -> Result<Array> { Ok(a.as_dtype(compute)?.transpose_axes(&[0, 2, 1, 3])?) };
    let (q, k, v) = (t(q)?, t(k)?, t(v)?);
    let scale = 1.0 / (d as f32).sqrt();
    let out = match mask {
        Some(m) => scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            ScaledDotProductAttentionMask::Array(&m.as_dtype(compute)?),
            None,
        )?,
        None => scaled_dot_product_attention(&q, &k, &v, scale, None, None)?,
    };
    Ok(out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?)
}

/// `TransformerTextEmbedder`'s key mask: real keys OR the diagonal, additive `[B, 1, T, T]`.
pub fn key_padding_mask(mask: &[Vec<i32>], tokens: usize) -> Array {
    let data = iris::key_padding_additive(mask, tokens);
    Array::from_slice(&data, &[mask.len() as i32, 1, tokens as i32, tokens as i32])
}

/// Concatenate along `axis`.
pub fn cat(parts: &[&Array], axis: i32) -> Result<Array> {
    Ok(concatenate_axis(parts, axis)?)
}

/// Shape check with a named error (checkpoint/config disagreement surfaces at load, by key).
pub fn expect_shape(name: &str, a: &Array, want: &[i32]) -> Result<()> {
    if a.shape() != want {
        return Err(Error::Msg(format!(
            "iris: {name} has shape {:?}, the config implies {want:?}",
            a.shape()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `x * (1 + scale) + shift` keeps a bf16 stream bf16 (the Python `1` is a weak
    /// scalar); the `1` here must not promote the PiT stream to f32 either.
    #[test]
    fn modulate_keeps_a_bf16_stream_in_bf16() {
        let bf16 = |v: &[f32]| {
            Array::from_slice(v, &[1, v.len() as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let out = modulate(&bf16(&[1.0, 2.0]), &bf16(&[0.5, 0.5]), &bf16(&[1.0, -1.0])).unwrap();
        assert_eq!(out.dtype(), Dtype::Bfloat16);
        let got: Vec<f32> = out.as_dtype(Dtype::Float32).unwrap().as_slice().to_vec();
        assert_eq!(got, vec![2.5, 0.5]);
    }
}
