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
use mlx_gen::ios_quantum::Scores;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Quant, Result};
use mlx_rs::fast::rms_norm;
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

    fn has(&self, key: &str) -> bool {
        self.weights.keys().any(|k| k == key)
    }

    /// The pre-quantized projection `prefix` of a packed artifact ([`crate::packed`]): the stored
    /// codes / scales / biases and the bias at bf16 — exactly what [`pack_projection`] builds from
    /// the dense weight at load — and the zero columns its input is padded by. `None` when the
    /// projection is stored dense.
    fn packed(&self, prefix: &str, bias: bool) -> Result<Option<(AdaptableLinear, i32)>> {
        let scales_key = format!("{prefix}.scales");
        if !self.has(&scales_key) {
            return Ok(None);
        }
        let codes = self.require(&format!("{prefix}.weight"))?.clone();
        let scales = self.require(&scales_key)?.clone();
        let biases = self.require(&format!("{prefix}.biases"))?.clone();
        let bias = if bias {
            Some(
                self.require(&format!("{prefix}.bias"))?
                    .as_dtype(Dtype::Bfloat16)?,
            )
        } else {
            None
        };
        let group = QUANT_GROUP_SIZE as i32;
        let bits = mlx_gen::quant::packed_bits(&codes, &scales, group)?;
        let padded_in = scales.shape()[1] * group;
        let in_key = format!("{prefix}{}", crate::packed::IN_FEATURES_SUFFIX);
        let in_features = if self.has(&in_key) {
            self.require(&in_key)?.item::<i32>()
        } else {
            padded_in
        };
        if in_features <= 0 || in_features > padded_in {
            return Err(Error::Msg(format!(
                "iris: the packed projection {prefix} records {in_features} input features over \
                 a {padded_in}-wide pack"
            )));
        }
        Ok(Some((
            AdaptableLinear::from_quantized_parts(codes, scales, biases, bias, group, bits),
            padded_in - in_features,
        )))
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
                // A packed artifact's base is stored packed: the install acted on a zero-stride
                // placeholder of its logical shape, and the adapters ride on the stored pack.
                if let Some((mut packed, in_pad)) = self.packed(prefix, bias)? {
                    packed.set_adapters(inner.adapters().to_vec());
                    return Ok(Linear {
                        inner: packed,
                        compute: self.compute,
                        in_pad,
                    });
                }
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
                    inner = pack_projection(prefix, &inner, bits)?;
                }
                return Ok(Linear {
                    inner,
                    compute: self.compute,
                    in_pad,
                });
            }
        }
        if let Some((inner, in_pad)) = self.packed(prefix, bias)? {
            return Ok(Linear {
                inner,
                compute: self.compute,
                in_pad,
            });
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
                let inner = pack_projection(prefix, &AdaptableLinear::dense(weight, bias), bits)?;
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

/// Pack a dense projection at `bits` (group [`QUANT_GROUP_SIZE`]), keeping its adapter stack —
/// the one quantizer of every backbone projection, adapted or not (sc-26266).
///
/// The weight and bias are held at bf16 (the stored dtype of the tier's scales and biases,
/// as `AdaptableLinear::quantize` holds them), and every code is rounded to the bf16 scales and
/// biases **as stored** ([`mlx_llm::primitives::quant::quantize_on_stored_grid`]), not to the
/// f32 grid MLX's `quantize` rounds to before it casts the scales. Same bytes and layout; on the
/// release's projections the q8 weight error drops from 0.86 % to 0.55 % (the f32-grid
/// optimum) — MLX's mismatch is ~1.6 x the rounding the 8 bits pay for. At q4 the grid step
/// is 17 x wider and the two agree to the noise.
pub fn pack_projection(
    prefix: &str,
    dense: &AdaptableLinear,
    bits: i32,
) -> Result<AdaptableLinear> {
    let (weight, bias) = dense.dense_weight().ok_or_else(|| {
        Error::Msg(format!(
            "iris: the projection {prefix} is not dense before its load-time quantization"
        ))
    })?;
    let weight = weight.as_dtype(Dtype::Bfloat16)?;
    let bias = bias.map(|b| b.as_dtype(Dtype::Bfloat16)).transpose()?;
    let group = QUANT_GROUP_SIZE as i32;
    let (codes, scales, biases) =
        mlx_llm::primitives::quant::quantize_on_stored_grid(&weight, group, bits)
            .map_err(|e| Error::Msg(format!("iris: packing {prefix}: {e}")))?;
    let mut packed =
        AdaptableLinear::from_quantized_parts(codes, scales, biases, bias, group, bits);
    packed.set_adapters(dense.adapters().to_vec());
    Ok(packed)
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

    /// Output width of the loaded projection.
    pub fn out_features(&self) -> u64 {
        self.logical_shape()[0] as u64
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

    /// Width of the loaded gain (the normalized axis).
    pub fn width(&self) -> u64 {
        self.weight.size() as u64
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
    attention_budgeted(q, k, v, compute, mask, mlx_gen::AttentionPlan::UNBOUNDED)
}

/// Query-row bounded SDPA. Each chunk retains every key/value head, including GQA groups.
/// This axis is tolerance-equivalent (not bit-exact); evaluated chunks cut the lazy graph.
pub fn attention_budgeted(
    q: &Array,
    k: &Array,
    v: &Array,
    compute: Dtype,
    mask: Option<&Array>,
    plan: mlx_gen::AttentionPlan<'_>,
) -> Result<Array> {
    let sh = q.shape();
    let (b, n, h, d) = (sh[0], sh[1], sh[2], sh[3]);
    let t =
        |a: &Array| -> Result<Array> { Ok(a.as_dtype(compute)?.transpose_axes(&[0, 2, 1, 3])?) };
    let (q, k, v) = (t(q)?, t(k)?, t(v)?);
    let scale = 1.0 / (d as f32).sqrt();
    let mask = mask.map(|m| m.as_dtype(compute)).transpose()?;
    let out = mlx_gen::sdpa_budgeted_bhsd(&q, &k, &v, scale, mask.as_ref(), plan)?;
    Ok(out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, n, h * d])?)
}

/// How the scores of one [`attention_budgeted`] call over `query_rows` queries at `head_dim` exist
/// in memory on the CURRENT default device — the iOS quantum accounting's score term.
///
/// Every Iris attention calls `mlx_rs::fast::scaled_dot_product_attention` (once, or once per
/// query chunk). On the GPU, MLX 0.32 runs it as a fused kernel that streams the scores —
/// `sdpa_full` for more than 8 query rows at head widths 64/80/128, `sdpa_vector` for at most 8
/// rows at 64/96/128/256 — so no score tensor exists ([`Scores::Fused`]); Iris's real head widths
/// are 128 (trunk, refiner, PiT, the Qwen3-VL tower) and 80 (layerwise adapter), all fused.
/// Anywhere else (the CPU, or a width the kernels reject) MLX falls back to `q·kᵀ → softmax → ·v`:
/// the logits `[B, H, rows, keys]` at the compute width, the softmax running in place on them
/// (one buffer), plus the masked copy of the logits when `masked`.
pub fn sdpa_scores(head_dim: u64, query_rows: u64, compute_bytes: u64, masked: bool) -> Scores {
    let gpu = mlx_rs::Device::try_default().is_ok_and(|d| d != mlx_rs::Device::cpu());
    let fused = if query_rows > 8 {
        matches!(head_dim, 64 | 80 | 128)
    } else {
        matches!(head_dim, 64 | 96 | 128 | 256)
    };
    if gpu && fused {
        Scores::Fused
    } else {
        Scores::Materialized {
            live: 1 + u64::from(masked),
            element_bytes: compute_bytes,
        }
    }
}

/// Whether [`sdpa_scores`] prices the CPU fallback (which also materializes `q · scale`, and
/// under grouped-query attention broadcasts k and v over the query groups).
pub fn sdpa_falls_back(scores: Scores) -> bool {
    matches!(scores, Scores::Materialized { .. })
}

/// Bytes per element of a compute dtype (f32 4, bf16/f16 2).
pub fn dtype_bytes(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::Float32 => 4,
        _ => 2,
    }
}

/// One named production attention site. Debug observations report the *executed* shared kernel
/// chunk count, not a predicted count, so dropping a plan cannot pass a parity-only test.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_at_site(
    q: &Array,
    k: &Array,
    v: &Array,
    compute: Dtype,
    mask: Option<&Array>,
    plan: mlx_gen::AttentionPlan<'_>,
    site: &'static str,
) -> Result<Array> {
    #[cfg(debug_assertions)]
    diagnostics::before(site);
    let out = attention_budgeted(q, k, v, compute, mask, plan)?;
    #[cfg(debug_assertions)]
    diagnostics::record(site, plan.budget, mlx_gen::attention::last_chunk_count());
    #[cfg(not(debug_assertions))]
    let _ = site;
    Ok(out)
}

/// Debug-build-only observation seam for CPU regression tests. Absent from release builds.
#[cfg(debug_assertions)]
#[doc(hidden)]
pub mod diagnostics {
    use mlx_gen::{AttentionBudget, CancelFlag};
    use std::cell::RefCell;
    #[derive(Clone, Debug)]
    pub struct Observation {
        pub site: &'static str,
        pub budget: AttentionBudget,
        pub chunks: usize,
    }
    #[derive(Default)]
    struct State {
        enabled: bool,
        observations: Vec<Observation>,
        cancel_at: Option<(&'static str, CancelFlag)>,
    }
    thread_local! { static STATE: RefCell<State> = RefCell::new(State::default()); }
    /// Opt in to observation and clear prior test state. Normal debug consumers record nothing.
    pub fn reset() {
        STATE.with(|s| {
            *s.borrow_mut() = State {
                enabled: true,
                ..Default::default()
            }
        });
    }
    pub fn observations() -> Vec<Observation> {
        STATE.with(|s| s.borrow().observations.clone())
    }
    pub fn cancel_at(site: &'static str, flag: CancelFlag) {
        STATE.with(|s| s.borrow_mut().cancel_at = Some((site, flag)));
    }
    pub(super) fn before(site: &'static str) {
        STATE.with(|s| {
            if let Some((at, flag)) = &s.borrow().cancel_at {
                if *at == site {
                    flag.cancel();
                }
            }
        });
    }
    pub(super) fn record(site: &'static str, budget: AttentionBudget, chunks: usize) {
        STATE.with(|s| {
            let mut state = s.borrow_mut();
            if state.enabled {
                state.observations.push(Observation {
                    site,
                    budget,
                    chunks,
                });
            }
        });
    }
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
