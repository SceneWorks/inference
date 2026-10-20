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
    /// MLX affine targets of a converted package; `None` for the dense package.
    pub affine: Option<&'a AffineTargets>,
}

/// The packed `<stem>.weight` targets of an MLX affine package and their exact parameters.
pub struct AffineTargets {
    pub targets: std::collections::BTreeSet<String>,
    pub bits: i32,
    pub group_size: i32,
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
        if let Some(affine) = self
            .affine
            .filter(|a| a.targets.contains(&format!("{prefix}.weight")))
        {
            let wq = self.require(&format!("{prefix}.weight"))?;
            let scales = self.require(&format!("{prefix}.scales"))?;
            // The packed shapes must encode EXACTLY the declared bits at the declared group.
            let bits = mlx_gen::quant::packed_bits(wq, scales, affine.group_size)?;
            if bits != affine.bits {
                return Err(Error::Unsupported(format!(
                    "flux3: {prefix} packs {bits} bits, the package declares {}",
                    affine.bits
                )));
            }
            return Ok(Linear {
                inner: AdaptableLinear::from_quantized_parts(
                    wq.clone(),
                    scales.as_dtype(self.compute)?,
                    self.require(&format!("{prefix}.biases"))?
                        .as_dtype(self.compute)?,
                    None,
                    affine.group_size,
                    bits,
                ),
                compute: self.compute,
            });
        }
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
        if self.is_quantized() {
            return Err(Error::Unsupported(
                "flux3 training: a packed MLX affine base has no differentiable dense weight; \
                 train on the original dense package"
                    .into(),
            ));
        }
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

    /// `[out, in]` of the base (a packed base reports its logical, unpacked shape).
    pub fn out_in(&self) -> Result<(i32, i32)> {
        let sh = self.inner.base_shape();
        match sh.as_slice() {
            [out, input] => Ok((*out, *input)),
            other => Err(Error::Msg(format!(
                "flux3: a Linear base has shape {other:?}"
            ))),
        }
    }

    /// Whether the base is a packed MLX affine projection.
    pub fn is_quantized(&self) -> bool {
        self.inner.dense_weight().is_none()
    }

    /// Every base parameter (dense weight, or packed codes + scales + biases), for eval.
    pub fn arrays(&self) -> Vec<&Array> {
        if let Some((weight, bias)) = self.inner.dense_weight() {
            return std::iter::once(weight).chain(bias).collect();
        }
        match self.inner.quantized_params() {
            Some((wq, scales, biases, bias, _, _)) => {
                [wq, scales, biases].into_iter().chain(bias).collect()
            }
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

#[cfg(test)]
mod affine_tests {
    use super::*;
    use mlx_gen::gen_core::flux3::affine;

    fn seeded(n: usize, salt: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 0.37 + salt).sin() * 0.8 + (i as f32 * 0.011).cos() * 0.2)
            .collect()
    }

    fn bf16_array(bits: &[u16], shape: &[i32]) -> Array {
        let values: Vec<f32> = bits.iter().map(|b| affine::bf16_value(*b)).collect();
        Array::from_slice(&values, shape)
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    }

    /// Hosted-only (MLX does not build on Windows). The backend-neutral converter's pack runs
    /// through MLX's own `quantized_matmul` exactly as its dequantized reference, and reconstructs
    /// within one affine step of `mx.quantize` on the same weights (byte identity is NOT claimed).
    ///
    /// *Mutations that red this:* packing codes high-bits-first, swapping scales and biases, or
    /// casting the `u32` codes in the loader.
    #[test]
    fn gen_core_affine_pack_runs_through_mlx_quantized_matmul() {
        let (n, k) = (32usize, 128usize);
        let w = seeded(n * k, 0.3);
        let x = Array::from_slice(&seeded(3 * k, 1.7), &[3, k as i32]);
        for (bits, group) in [(4u32, 64u32), (8, 32)] {
            let packed = affine::quantize(&w, n, k, bits, group).unwrap();
            let words = (k * bits as usize / 32) as i32;
            let groups = (k / group as usize) as i32;
            let linear = Linear {
                inner: AdaptableLinear::from_quantized_parts(
                    Array::from_slice(&packed.codes, &[n as i32, words]),
                    bf16_array(&packed.scales, &[n as i32, groups])
                        .as_dtype(Dtype::Float32)
                        .unwrap(),
                    bf16_array(&packed.biases, &[n as i32, groups])
                        .as_dtype(Dtype::Float32)
                        .unwrap(),
                    None,
                    group as i32,
                    bits as i32,
                ),
                compute: Dtype::Float32,
            };
            assert!(linear.is_quantized());
            assert_eq!(linear.out_in().unwrap(), (n as i32, k as i32));
            let y = linear.forward(&x).unwrap();
            let reference = Array::from_slice(
                &affine::dequantize(&packed, n, k, bits, group).unwrap(),
                &[n as i32, k as i32],
            );
            let want = x
                .matmul(reference.transpose_axes(&[1, 0]).unwrap())
                .unwrap();
            let diff = y
                .subtract(&want)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            let peak = want.abs().unwrap().max(None).unwrap().item::<f32>();
            assert!(diff <= 1e-4 * peak.max(1.0), "q{bits}: {diff} vs {peak}");
            // Same weights through mx.quantize: both reconstruct within one affine step.
            let dense = Array::from_slice(&w, &[n as i32, k as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let (mq, ms, mb) = mlx_rs::ops::quantize(dense, group as i32, bits as i32).unwrap();
            let theirs = mlx_rs::ops::dequantize(mq, ms, &mb, group as i32, bits as i32)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap();
            let gap = theirs
                .subtract(reference)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            let step = packed
                .scales
                .iter()
                .map(|s| affine::bf16_value(*s).abs())
                .fold(0f32, f32::max);
            assert!(gap <= 1.5 * step, "q{bits}: {gap} vs step {step}");
        }
    }
}
