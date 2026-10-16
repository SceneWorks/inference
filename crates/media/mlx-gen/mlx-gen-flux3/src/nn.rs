//! The leaf modules of the FLUX 3 DiT graph and the precision policy they share.
//!
//! **Precision policy** (mirrors upstream's CUDA path, `torch.autocast(bfloat16)` over bf16
//! parameters, and degenerates to the oracle's FP32 CPU path when the compute dtype is f32):
//!
//! * every [`Linear`] casts its input to the compute dtype and returns the compute dtype
//!   (autocast's matmul rule) — its weight is stored in the compute dtype;
//! * `F.layer_norm` is on autocast's f32 list: [`layer_norm_f32`] normalizes in f32 and returns
//!   f32, so the modulated input of a block is f32 until the next Linear casts it back;
//! * the q/k RMSNorm + RoPE (`rmsnorm_rope`) runs in f32 and returns the dtype it was given;
//! * attention runs in the compute dtype (`scaled_dot_product_attention` is on autocast's
//!   low-precision list); the residual streams and the modulation vectors are outputs of
//!   Linears, so they stay in the compute dtype;
//! * every other elementwise op follows ordinary type promotion, exactly as torch does.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{concatenate_axis, stack_axis};
use mlx_rs::{Array, Dtype};

/// LayerNorm eps of every `layernorm_modulation` (`F.layer_norm(..., eps=1e-6)`).
pub const LN_EPS: f32 = 1e-6;
/// RMSNorm eps of the q/k norms (`rmsnorm_rope(..., eps=1e-6)`).
pub const RMS_EPS: f32 = 1e-6;

/// Builds modules from a checkpoint at one compute dtype.
pub struct Loader<'a> {
    pub weights: &'a Weights,
    pub compute: Dtype,
}

impl Loader<'_> {
    fn require(&self, key: &str) -> Result<&Array> {
        self.weights.require(key)
    }

    /// A tensor kept in f32 (the q/k RMSNorm scales).
    pub fn f32(&self, key: &str) -> Result<Array> {
        Ok(self.require(key)?.as_dtype(Dtype::Float32)?)
    }

    /// A bias-less `nn.Linear` (`{prefix}.weight`, `[out, in]`), stored in the compute dtype.
    pub fn linear(&self, prefix: &str) -> Result<Linear> {
        let weight = self
            .require(&format!("{prefix}.weight"))?
            .as_dtype(self.compute)?;
        Ok(Linear {
            inner: AdaptableLinear::dense(weight, None),
            compute: self.compute,
        })
    }
}

/// `nn.Linear(bias=False)` as the repo's [`AdaptableLinear`]: the bare forward is `x·Wᵀ`, and an
/// installed LoRA (sc-26239) adds its residual on top.
pub struct Linear {
    inner: AdaptableLinear,
    compute: Dtype,
}

impl Linear {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = x.as_dtype(self.compute)?;
        self.inner.forward(&x)
    }

    /// Attach a preflight-validated additive bottleneck without fusing the base.
    pub fn push_lora(&mut self, down: Array, up: Array, scale: f64) -> Result<()> {
        let adapter = mlx_gen::adapters::Adapter::Lora {
            a: down.transpose_axes(&[1, 0])?.as_dtype(self.compute)?,
            b: up.transpose_axes(&[1, 0])?.as_dtype(self.compute)?,
            scale: scale as f32,
        };
        adapter.materialize()?;
        self.inner.push(adapter);
        Ok(())
    }

    /// Replace the factors inside the functional autograd closure. No evaluation,
    /// serialization or merged delta: each trace receives the current A/B arrays.
    /// Normal inference residual arithmetic is deliberate: source mixed precision and
    /// generation reload must use the same cast/alpha/rank convention.
    pub fn replace_training_lora(&mut self, down: Array, up: Array, scale: f64) -> Result<()> {
        let (output, input) = self.out_in()?;
        let rank = down.shape().first().copied().unwrap_or(0);
        if rank <= 0
            || down.shape() != [rank, input]
            || up.shape() != [output, rank]
            || !scale.is_finite()
        {
            return Err(Error::Unsupported(
                "flux3 training: invalid trainable LoRA shape/gain".into(),
            ));
        }
        self.inner
            .set_adapters(vec![mlx_gen::adapters::Adapter::Lora {
                a: down.transpose_axes(&[1, 0])?.as_dtype(self.compute)?,
                b: up.transpose_axes(&[1, 0])?.as_dtype(self.compute)?,
                scale: scale as f32,
            }]);
        Ok(())
    }

    /// `[out, in]` of the dense base.
    pub fn out_in(&self) -> Result<(i32, i32)> {
        let (weight, _) = self
            .inner
            .dense_weight()
            .ok_or_else(|| Error::Msg("flux3: a Linear base is not dense".into()))?;
        let sh = weight.shape();
        Ok((sh[0], sh[1]))
    }

    /// The dense base parameters.
    pub fn arrays(&self) -> Vec<&Array> {
        match self.inner.dense_weight() {
            Some((weight, bias)) => std::iter::once(weight).chain(bias).collect(),
            None => Vec::new(),
        }
    }
}

/// `F.layer_norm(x, (dim,), eps)` without affine parameters, computed and returned in **f32**
/// (autocast's policy for `layer_norm`).
pub fn layer_norm_f32(x: &Array) -> Result<Array> {
    Ok(layer_norm(
        &x.as_dtype(Dtype::Float32)?,
        None,
        None,
        LN_EPS,
    )?)
}

/// `layernorm_modulation(x, scale, shift) = (1 + scale) · layer_norm(x) + shift`. The `1` is
/// created in `scale`'s dtype (torch's weak Python scalar): `1 + scale` rounds in the compute
/// dtype, and the product with the f32 norm promotes to f32, exactly as upstream.
pub fn layernorm_modulation(x: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    let one = Array::from_f32(1.0).as_dtype(scale.dtype())?;
    let norm = layer_norm_f32(x)?;
    Ok(norm.multiply(&scale.add(&one)?)?.add(shift)?)
}

/// `rmsnorm_rope(x, scale, freqs)`: per-head RMSNorm in f32 with an f32 gain, then the adjacent-pair
/// complex rotation in f32, returned in `x`'s dtype. `x` is `[B, H, L, D]`; `cos`/`sin` are
/// `[L, D/2]` (the [`gen_core::flux3::rope_cos_sin`] tables of the tokens, in token order).
///
/// [`gen_core::flux3::rope_cos_sin`]: mlx_gen::gen_core::flux3::rope_cos_sin
pub fn rmsnorm_rope(x: &Array, scale: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, h, l, d) = (sh[0], sh[1], sh[2], sh[3]);
    let pairs = d / 2;
    let normed = rms_norm(&x.as_dtype(Dtype::Float32)?, scale, RMS_EPS)?;
    let xf = normed.reshape(&[b, h, l, pairs, 2])?;
    let parts = xf.split(2, 4)?;
    let (re, im) = (parts[0].squeeze_axes(&[4])?, parts[1].squeeze_axes(&[4])?);
    let cos = cos.reshape(&[1, 1, l, pairs])?;
    let sin = sin.reshape(&[1, 1, l, pairs])?;
    // freqs = [[cos, −sin], [sin, cos]]: out0 = cos·x0 − sin·x1, out1 = sin·x0 + cos·x1.
    let out_re = re.multiply(&cos)?.subtract(&im.multiply(&sin)?)?;
    let out_im = re.multiply(&sin)?.add(&im.multiply(&cos)?)?;
    let out = stack_axis(&[out_re, out_im], 4)?.reshape(&[b, h, l, d])?;
    Ok(out.as_dtype(x.dtype())?)
}

/// `packed_attention(q, k, v, bounds)`: per-sample SDPA over `[1, H, L, D]` (samples never attend
/// each other), in the compute dtype; returns `[L, H·D]` in token order. An empty bound
/// contributes nothing (upstream skips it and leaves zeros, which the single stream never reads).
pub fn packed_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    bounds: &[(i32, i32)],
    compute: Dtype,
) -> Result<Array> {
    let sh = q.shape();
    let (h, l, d) = (sh[1], sh[2], sh[3]);
    let scale = 1.0 / (d as f32).sqrt();
    let (q, k, v) = (
        q.as_dtype(compute)?,
        k.as_dtype(compute)?,
        v.as_dtype(compute)?,
    );
    let one = |start: i32, end: i32| -> Result<Array> {
        let (q, k, v) = if start == 0 && end == l {
            (q.clone(), k.clone(), v.clone())
        } else {
            let slice = |a: &Array| -> Result<Array> {
                let mut parts = a.split_axis(&[start, end], 2)?;
                Ok(parts.swap_remove(1))
            };
            (slice(&q)?, slice(&k)?, slice(&v)?)
        };
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        Ok(o.transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[end - start, h * d])?)
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
        1 => Ok(chunks.pop().unwrap()),
        _ => {
            let refs: Vec<&Array> = chunks.iter().collect();
            Ok(concatenate_axis(&refs, 0)?)
        }
    }
}

/// Split the last axis into `n` equal chunks (`tensor.chunk(n, dim=-1)`).
pub fn chunk_last(x: &Array, n: i32) -> Result<Vec<Array>> {
    let axis = x.ndim() as i32 - 1;
    Ok(x.split(n, axis)?)
}

/// `F.silu` in the input dtype.
pub fn silu(x: &Array) -> Result<Array> {
    mlx_gen::nn::silu(x)
}

/// Concatenate along `axis`.
pub fn cat(parts: &[&Array], axis: i32) -> Result<Array> {
    Ok(concatenate_axis(parts, axis)?)
}

/// Shape check with a named error (checkpoint/config disagreement surfaces at load, by key).
pub fn expect_shape(name: &str, a: &Array, want: &[i32]) -> Result<()> {
    if a.shape() != want {
        return Err(Error::Msg(format!(
            "flux3: {name} has shape {:?}, the config implies {want:?}",
            a.shape()
        )));
    }
    Ok(())
}
