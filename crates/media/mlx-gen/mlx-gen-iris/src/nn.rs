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
use mlx_gen::{Error, Quant, Result};
use mlx_rs::fast::{rms_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{concatenate_axis, pad, stack_axis};
use mlx_rs::{Array, Dtype};

/// Group size of every quantized backbone projection (sc-26266): the codebase default, which the
/// release's projection widths (2560, 768, 256, 64, …) divide.
pub const QUANT_GROUP_SIZE: usize = mlx_gen::quant::DEFAULT_GROUP_SIZE as usize;

/// The load-time quantization of one backbone build (sc-26266): the affine width of the `q8` /
/// `q4` tier.
///
/// A projection whose input width is not a multiple of [`QUANT_GROUP_SIZE`] is packed over a
/// zero-padded input. The release has exactly one such family — the SwiGLU down projections
/// (`*.mlp.w2`, `[2560, 6826]`, ~1.2 GB of the 5.6 GB bf16 backbone). Zero-padding their input to
/// the next group multiple is exact (`[x, 0] · [W, 0]ᵀ = x · Wᵀ`) and lets them quantize like
/// every other projection — adapted or not: an adapter residual on such a projection reads the
/// unpadded input while the packed base reads the padded one ([`Linear::forward`]), so the tier is
/// one layout for the whole backbone whatever adapters a load installs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackboneQuant {
    /// Affine bits per weight (4 or 8).
    pub bits: i32,
}

impl BackboneQuant {
    /// The quantization a load with `quant` applies: `None` is the dense tier. Only the affine
    /// tiers exist on this backend; NVFP4 is a typed refusal.
    pub fn for_load(quant: Option<Quant>) -> Result<Option<Self>> {
        let bits = match quant {
            None => return Ok(None),
            Some(Quant::Q4) => 4,
            Some(Quant::Q8) => 8,
            Some(other) => {
                return Err(Error::Unsupported(format!(
                    "iris: the MLX Iris-3B route ships the q8 and q4 affine tiers; {other:?} is \
                     not one of them"
                )))
            }
        };
        Ok(Some(Self { bits }))
    }
}

/// How one `[out, in]` projection is held under a [`BackboneQuant`] — the single rule
/// [`Loader::linear`] applies and the memory contract prices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionLayout {
    /// Stored in the compute dtype.
    Dense,
    /// Affine-packed at `bits` over an input zero-padded to `padded_in` (`== in` when aligned).
    Quantized { bits: i32, padded_in: usize },
}

impl ProjectionLayout {
    /// The layout of a projection with `in_features` inputs. A projection narrower than one group
    /// (the release's 3-, 12- and 16-wide pixel and pooling heads, a few KB in all) stays dense.
    pub fn of(in_features: usize, quant: Option<BackboneQuant>) -> Self {
        let Some(quant) = quant else {
            return Self::Dense;
        };
        if in_features < QUANT_GROUP_SIZE {
            return Self::Dense;
        }
        let padded_in = in_features.div_ceil(QUANT_GROUP_SIZE) * QUANT_GROUP_SIZE;
        Self::Quantized {
            bits: quant.bits,
            padded_in,
        }
    }

    /// Resident bytes of an `[out, in]` weight held this way, `dense_width` bytes per element when
    /// dense. A packed weight is `bits / 8` bytes per (padded) element plus one bf16 scale and one
    /// bf16 bias per group — the layout `mlx_rs::ops::quantize` produces.
    pub fn weight_bytes(self, out: u64, in_features: u64, dense_width: u64) -> u64 {
        match self {
            Self::Dense => out * in_features * dense_width,
            Self::Quantized { bits, padded_in } => {
                let padded_in = padded_in as u64;
                out * padded_in * bits as u64 / 8 + out * (padded_in / QUANT_GROUP_SIZE as u64) * 4
            }
        }
    }
}

/// Builds modules from a checkpoint at one compute dtype.
pub struct Loader<'a> {
    pub weights: &'a Weights,
    pub compute: Dtype,
    /// The adapted projections of a load that installed LoRA/LoKr adapters (see
    /// [`AdaptedLinears`]); `None` builds every projection bare from `weights`.
    pub adapted: Option<&'a AdaptedLinears>,
    /// The load-time quantization of every projection ([`ProjectionLayout::of`]); `None` is the
    /// dense tier.
    pub quant: Option<BackboneQuant>,
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
            if let Some(mut inner) = adapted.linears.borrow_mut().remove(prefix) {
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
                // The adapters stay forward-time residuals over the packed base, reading the
                // checkpoint-width input; an unaligned base is rebuilt over its zero-padded input
                // with the same adapter stack ([`Linear::forward`] feeds each its own width).
                let in_features = inner.base_shape()[1] as usize;
                let mut in_pad = 0;
                if let ProjectionLayout::Quantized { bits, padded_in } =
                    ProjectionLayout::of(in_features, self.quant)
                {
                    in_pad = (padded_in - in_features) as i32;
                    if in_pad > 0 {
                        let (weight, bias) = inner.dense_weight().ok_or_else(|| {
                            Error::Msg(format!(
                                "iris: the adapted projection {prefix} is not dense before its \
                                 load-time quantization"
                            ))
                        })?;
                        let weight = pad(weight, &[(0, 0), (0, in_pad)][..], None, None)?;
                        let mut padded = AdaptableLinear::dense(weight, bias.cloned());
                        padded.set_adapters(inner.adapters().to_vec());
                        inner = padded;
                    }
                    inner.quantize(bits, Some(QUANT_GROUP_SIZE as i32))?;
                }
                return Ok(Linear {
                    inner,
                    compute: self.compute,
                    in_pad,
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
        let in_features = *weight.shape().last().unwrap_or(&0) as usize;
        match ProjectionLayout::of(in_features, self.quant) {
            ProjectionLayout::Dense => Ok(Linear {
                inner: AdaptableLinear::dense(weight, bias),
                compute: self.compute,
                in_pad: 0,
            }),
            ProjectionLayout::Quantized { bits, padded_in } => {
                let in_pad = (padded_in - in_features) as i32;
                let weight = if in_pad > 0 {
                    pad(&weight, &[(0, 0), (0, in_pad)][..], None, None)?
                } else {
                    weight
                };
                let mut inner = AdaptableLinear::dense(weight, bias);
                inner.quantize(bits, Some(QUANT_GROUP_SIZE as i32))?;
                Ok(Linear {
                    inner,
                    compute: self.compute,
                    in_pad,
                })
            }
        }
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
    /// Zero columns appended to the input before a packed base whose input width is not a group
    /// multiple (`ProjectionLayout::Quantized::padded_in`); `0` for every other projection. The
    /// adapter residuals read the unpadded input.
    in_pad: i32,
}

impl Linear {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = x.as_dtype(self.compute)?;
        if self.in_pad > 0 {
            let mut widths = vec![(0, 0); x.ndim()];
            *widths.last_mut().expect("a projection input has an axis") = (0, self.in_pad);
            return self
                .inner
                .forward_split(&pad(&x, &widths[..], None, None)?, &x);
        }
        self.inner.forward(&x)
    }

    /// The base parameters — dense weight and bias, or the packed codes, scales, biases and bias
    /// (adapter factors are evaluated at load by the [`Loader`]).
    pub fn arrays(&self) -> Vec<&Array> {
        if let Some((weight, bias)) = self.inner.dense_weight() {
            return std::iter::once(weight).chain(bias).collect();
        }
        match self.inner.quantized_params() {
            Some((codes, scales, biases, bias, _, _)) => {
                [codes, scales, biases].into_iter().chain(bias).collect()
            }
            None => Vec::new(),
        }
    }

    /// The checkpoint's `[out, in]`, whatever the projection is held as.
    pub fn logical_shape(&self) -> [i32; 2] {
        let shape = self.inner.base_shape();
        [shape[0], shape[1] - self.in_pad]
    }

    /// Whether the base is affine-packed.
    pub fn is_quantized(&self) -> bool {
        self.inner.is_quantized()
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
#[derive(Clone)]
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
        Self::grid_2d_aspect(head_dim, height, width, theta, scale, false)
    }

    /// [`Self::grid_2d`] under `model.rope_aspect` (`square = true`: each axis normalized to
    /// `[0, scale]` on its own — `gen_core::iris::rope_2d_angles_aspect`).
    pub fn grid_2d_aspect(
        head_dim: usize,
        height: usize,
        width: usize,
        theta: f32,
        scale: f32,
        square: bool,
    ) -> Self {
        let angles = iris::rope_2d_angles_aspect(head_dim, height, width, theta, scale, square);
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
    expect_dims(name, a.shape(), want)
}

/// [`expect_shape`] over a shape already in hand (a projection's [`Linear::logical_shape`]).
pub fn expect_dims(name: &str, have: &[i32], want: &[i32]) -> Result<()> {
    if have != want {
        return Err(Error::Msg(format!(
            "iris: {name} has shape {have:?}, the config implies {want:?}"
        )));
    }
    Ok(())
}
