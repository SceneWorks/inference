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

    /// A bias-less `nn.Linear` (`{prefix}.weight`, `[out, in]`), stored in the compute dtype.
    pub fn linear(&self, prefix: &str) -> Result<Linear> {
        let weight = to(&self.take(&format!("{prefix}.weight"))?, self.compute)?;
        if weight.rank() != 2 {
            return Err(Error::Msg(format!(
                "flux3: {prefix}.weight is rank {}, a Linear needs [out, in]",
                weight.rank()
            )));
        }
        Ok(Linear {
            weight,
            adapted: None,
        })
    }
}

/// `nn.Linear(bias=False)`: `x · Wᵀ` in the compute dtype (the weight's).
pub struct Linear {
    weight: Tensor,
    adapted: Option<candle_gen::quant::AdaptLinear>,
}

impl Linear {
    /// A linear layer from an already-shaped `[out, in]` weight, stored as given.
    pub fn from_weight(weight: Tensor) -> Self {
        Self {
            weight,
            adapted: None,
        }
    }

    /// `x` of any leading shape `[.., in]` → `[.., out]` in the weight's dtype.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = self.weight.dtype();
        let x = to(x, dtype)?;
        let dims = x.dims().to_vec();
        let (last, lead) = dims
            .split_last()
            .ok_or_else(|| Error::Msg("flux3: a Linear input needs at least one axis".into()))?;
        let rows: usize = lead.iter().product();
        let out_dim = self.weight.dim(0)?;
        let flat = x.reshape((rows, *last))?;
        let gemm = gemm_dtype(&self.weight);
        let y = to(&flat, gemm)?.matmul(&to(&self.weight, gemm)?.t()?)?;
        let y = to(&y, dtype)?;
        let y = if let Some(adapted) = &self.adapted {
            adapted.apply_residuals(&to(&flat, gemm)?, y)?
        } else {
            y
        };
        let mut shape = lead.to_vec();
        shape.push(out_dim);
        Ok(y.reshape(shape)?)
    }

    /// Attach a validated bottleneck residual without mutating the base weight.
    pub fn push_lora(&mut self, down: Tensor, up: Tensor, scale: f64) -> Result<()> {
        if self.adapted.is_none() {
            let (output, input) = self.out_in()?;
            self.adapted = Some(candle_gen::quant::AdaptLinear::from_dense(
                candle_gen::candle_nn::Linear::new(self.weight.clone(), None),
                input,
                output,
            ));
        }
        self.adapted
            .as_mut()
            .expect("initialized")
            .push_lora_checked(down.t()?.contiguous()?, up.t()?.contiguous()?, scale)
    }

    /// Training-only storage-sharing factors. Unlike inference's prepared-factor cache,
    /// `TrainableLora` reads the Var leaves on EVERY forward and transposes inside that graph.
    pub fn push_trainable_lora(&mut self, down: Tensor, up: Tensor, scale: f64) -> Result<()> {
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
        if self.adapted.is_none() {
            self.adapted = Some(candle_gen::quant::AdaptLinear::from_dense(
                candle_gen::candle_nn::Linear::new(self.weight.clone(), None),
                input,
                output,
            ));
        }
        self.adapted
            .as_mut()
            .expect("initialized")
            .push_trainable_lora(down, up, scale);
        Ok(())
    }

    /// `[out, in]` of the weight.
    pub fn out_in(&self) -> Result<(usize, usize)> {
        Ok(self.weight.dims2()?)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn dtype(&self) -> DType {
        self.weight.dtype()
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
/// [`gen_core::flux3::rope_cos_sin`] tables of the tokens, in token order.
///
/// [`gen_core::flux3::rope_cos_sin`]: candle_gen::gen_core::flux3::rope_cos_sin
pub struct Rope {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl Rope {
    /// The tables of `ids` under the DiT's `axes_dim` / `theta`, on `device`.
    pub fn new(ids: &[[f32; 4]], axes_dim: &[usize], theta: f64, device: &Device) -> Result<Self> {
        let (cos, sin) = candle_gen::gen_core::flux3::rope_cos_sin(ids, axes_dim, theta);
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
        candle_gen::candle_nn::ops::rms_norm_slow(&xf, scale, RMS_EPS)?
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
        let o = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            None,
            |scores| {
                if training {
                    candle_gen::candle_nn::ops::softmax(scores, D::Minus1)
                } else {
                    softmax_last_dim(scores)
                }
            },
            candle_gen::ATTN_SCORES_BUDGET,
        )?;
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
