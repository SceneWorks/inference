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

use gen_core_iris as iris;
use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Quant, Result};
use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{concatenate_axis, pad, stack_axis};
use mlx_rs::{Array, Dtype};

/// Group size of every quantized backbone projection (sc-26266): the codebase default, which the
/// release's projection widths (2560, 768, 256, 64, …) divide.
pub const QUANT_GROUP_SIZE: usize = mlx_gen::quant::DEFAULT_GROUP_SIZE as usize;

/// Recheck file-backed pending constructor and stored-grid evaluation allocations after build.
/// The earlier joint check is not a reservation against intervening owned allocations.
pub(crate) const ADAPTED_BASE_EVAL_QUANTUM: &str = "Iris adapted base evaluation";

/// Called by the file-backed model loader immediately before evaluating the built DiT parameters.
/// The byte count comes from the same captured-header constructor law as the early load guard.
/// Its raw base inputs were evaluated by read_backbone_tensor; accepted Iris adapter files attach
/// residuals without replacing that base (header_inventory rejects diff-patch files). Loader::linear
/// has already evaluated and synchronized those residuals, so no adapter producer feeds a base cast.
/// The captured source plan also includes still-pending stored-grid outputs/workspace and excludes
/// its already-active build scalars. It does not certify arbitrary caller-supplied Array graphs or
/// oversized incoming backing. Even a zero-target build reaches this real evaluation boundary.
pub(crate) fn before_adapted_base_eval(pending_constructor_bytes: u64) -> Result<()> {
    mlx_gen::ios_quantum::before(ADAPTED_BASE_EVAL_QUANTUM, pending_constructor_bytes)
}

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
    load_facts: BTreeMap<String, crate::adapters::ProjectionAdapterLoadFacts>,
}

impl AdaptedLinears {
    /// All known-file accepted residual materialization allocations still potentially pending.
    /// Source file inputs are already evaluated. Diff/merged-base adapters are refused upstream.
    pub(crate) fn pending_materialization_bytes(&self) -> Result<u64> {
        self.load_facts.values().try_fold(0u64, |sum, facts| {
            sum.checked_add(facts.materialize_scratch_bytes()?)
                .ok_or_else(|| {
                    Error::Unsupported("Iris pending adapter materialization sum overflow".into())
                })
        })
    }

    /// `linears` are every projection the install could reach; `adapted` the paths it changed.
    pub fn new(linears: BTreeMap<String, AdaptableLinear>, adapted: BTreeSet<String>) -> Self {
        Self {
            linears: RefCell::new(linears),
            adapted,
            consumed: RefCell::new(BTreeSet::new()),
            load_facts: BTreeMap::new(),
        }
    }

    pub(crate) fn with_load_facts(
        mut self,
        facts: BTreeMap<String, crate::adapters::ProjectionAdapterLoadFacts>,
    ) -> Self {
        self.load_facts = facts;
        self
    }

    /// The actual ordered payload inventory before the Loader moves the projections into the
    /// graph. Keys are the same upstream projection paths used by the header-only inventory.
    pub fn adapter_facts(&self) -> crate::adapters::ProjectionAdapterFacts {
        self.linears
            .borrow()
            .iter()
            .filter(|(_, linear)| !linear.adapters().is_empty())
            .map(|(path, linear)| {
                (
                    path.clone(),
                    linear
                        .adapters()
                        .iter()
                        .map(mlx_gen::adapters::facts::AdapterFacts::of)
                        .collect(),
                )
            })
            .collect()
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
    fn packed(
        &self,
        prefix: &str,
        bias: bool,
        resolved_width: Option<i32>,
    ) -> Result<Option<(AdaptableLinear, i32)>> {
        let scales_key = format!("{prefix}.scales");
        if !self.has(&scales_key) {
            crate::packed::check_loaded_projection(self.weights, prefix, None)?;
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
        crate::packed::check_loaded_projection(self.weights, prefix, Some(bits))?;
        let padded_in = scales.shape()[1] * group;
        let in_key = format!("{prefix}{}", crate::packed::IN_FEATURES_SUFFIX);
        let in_features = if self.has(&in_key) {
            let source = self.require(&in_key)?; // Preserve exact consumed-key accounting.
            match resolved_width {
                // The install already evaluated this immutable marker with the original
                // try_item conversion. Reuse the actual value, not a second conversion buffer.
                Some(width) => width,
                None => source.item::<i32>(), // Original bare packed-loader conversion path.
            }
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
                // Raw file reads and graph scalar construction are a separate guarded quantum.
                // This is the ACTUAL lazy reconstruction boundary: all new casts, factor products,
                // kron/Hadamard/einsum outputs are checked BEFORE evaluation allocates them. The
                // already-held base/source tensors are deliberately absent from this scratch.
                let load = adapted.load_facts.get(prefix);
                if let Some(load) = load {
                    mlx_gen::ios_quantum::before(
                        crate::adapters::ADAPTER_MATERIALIZE_QUANTUM,
                        load.materialize_scratch_bytes()?,
                    )?;
                }
                // Every payload is evaluated at load, never on the first forward.
                inner.materialize_adapters()?;
                if load.is_some() {
                    let arrays: Vec<&Array> =
                        inner.adapters().iter().flat_map(|a| a.arrays()).collect();
                    mlx_gen::ios_quantum::finish(
                        crate::adapters::ADAPTER_MATERIALIZE_QUANTUM,
                        &arrays,
                    )?;
                }
                // A packed artifact's base is stored packed: the install acted on a zero-stride
                // placeholder of its logical shape, and the adapters ride on the stored pack.
                if let Some((mut packed, in_pad)) =
                    self.packed(prefix, bias, Some(inner.base_shape()[1]))?
                {
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
        if let Some((inner, in_pad)) = self.packed(prefix, bias, None)? {
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
/// existed, and an installed LoRA/LoKr/LoHa adds its residual `Σ adapter(x)` on top. Iris's dense
/// install host reconstructs LoKr/LoHa as materialized `[out, in]` deltas even over a packed base;
/// packing the base does not change or re-install the adapter stack.
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

    /// All retained parameters: the dense/packed base AND the materialized adapter-owned
    /// payloads. Transpose views are listed once; reconstruction-only source factors are absent.
    pub fn arrays(&self) -> Vec<&Array> {
        let mut arrays: Vec<&Array> = if let Some((weight, bias)) = self.inner.dense_weight() {
            std::iter::once(weight).chain(bias).collect()
        } else {
            match self.inner.quantized_params() {
                Some((codes, scales, biases, bias, _, _)) => {
                    [codes, scales, biases].into_iter().chain(bias).collect()
                }
                None => Vec::new(),
            }
        };
        arrays.extend(self.inner.adapters().iter().flat_map(|a| a.arrays()));
        arrays
    }

    pub fn adapter_facts(&self) -> Vec<mlx_gen::adapters::facts::AdapterFacts> {
        self.inner
            .adapters()
            .iter()
            .map(mlx_gen::adapters::facts::AdapterFacts::of)
            .collect()
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

    /// Zero columns appended to the input before an unaligned packed base (`0` otherwise).
    pub fn input_pad(&self) -> u64 {
        self.in_pad.max(0) as u64
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
    /// `[0, scale]` on its own — `gen_core_iris::rope_2d_angles_aspect`).
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

/// Whether the current default device runs MLX's fused kernels (`rms_norm`, SDPA) — any device but
/// the CPU, whose backend runs their unfused fallbacks. On the GPU, MLX 0.32 runs every Iris
/// attention as a fused kernel that streams its scores (`sdpa_full` above 8 query rows at head
/// widths 64/80/128, `sdpa_vector` at most 8 at 64/96/128/256; Iris's are 128 — trunk, refiner,
/// PiT, the Qwen3-VL tower — and 80, the layerwise adapter), so no score tensor exists there.
pub fn fused_kernels() -> bool {
    mlx_rs::Device::try_default().is_ok_and(|d| d != mlx_rs::Device::cpu())
}

/// Bytes per element of a compute dtype (f32 4, bf16/f16 2).
pub fn dtype_bytes(dtype: Dtype) -> u64 {
    match dtype {
        Dtype::Float32 => 4,
        _ => 2,
    }
}

/// A sub-layer barrier of one iOS block quantum (sc-26270): on a physical iOS device (and in the
/// CPU fixture that runs its guards) `arrays` are materialized before the block builds its next
/// sub-layer. MLX allocates an op's output when it encodes it and keeps the op's inputs alive until
/// its command buffer completes — with up to ten command buffers encoded ahead — so one lazy block
/// graph can hold every intermediate of the block at once; these barriers bound a block quantum's
/// peak to its widest sub-layer, which is what the counted prices in [`crate::dit`] sum. An eval
/// never changes a value, and off device the graph is exactly the historical one.
pub(crate) fn barrier(arrays: &[&Array]) -> Result<()> {
    if mlx_gen::ios_quantum::enabled() {
        mlx_rs::transforms::eval(arrays.iter().copied())?;
        #[cfg(test)]
        BARRIER_HOOK.with(|hook| {
            if let Some(hook) = hook.borrow_mut().as_mut() {
                hook();
            }
        });
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// Run after every sub-layer barrier's evaluation — the quantum proofs' seam to measure each
    /// sub-layer from its own start (`crate::quantum_proofs`). Test builds only.
    pub(crate) static BARRIER_HOOK: RefCell<Option<Box<dyn FnMut()>>> = const { RefCell::new(None) };
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

/// One source graph serves fresh-A runtime budgets and the conservative-U cumulative ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantAllocationStage {
    Build,
    Evaluate,
}

pub(crate) fn visit_stored_grid_stages(
    rows: u64,
    input: u64,
    group: u64,
    bits: u64,
    metal: bool,
    mut allocate: impl FnMut(QuantAllocationStage, u64) -> Result<()>,
) -> Result<()> {
    let overflow = || Error::Unsupported("Iris stored-grid allocation arithmetic overflow".into());
    if group == 0 || !input.is_multiple_of(group) || !matches!(bits, 2 | 4 | 8) {
        return Err(overflow());
    }
    let n = rows.checked_mul(input).ok_or_else(overflow)?;
    let groups = n / group;
    let words = n.checked_mul(bits).ok_or_else(overflow)? / 32;
    let mut emit = |stage, width: u64, elements: u64, count: u64| {
        let request = width.checked_mul(elements).ok_or_else(overflow)?;
        for _ in 0..count {
            allocate(stage, request)?;
        }
        Ok::<(), Error>(())
    };
    use QuantAllocationStage::{Build, Evaluate};
    emit(Evaluate, 2, n, 1)?; // conditional Quantize input contiguity copy
    emit(Evaluate, 4, words, 1)?;
    emit(Evaluate, 2, groups, 2)?; // quantized scale/bias siblings, even discarded codes count
    emit(Evaluate, 4, n, 1)?; // weight -> f32
    emit(Evaluate, 4, groups, 2)?; // scale and bias -> f32
    emit(Evaluate, 4, n, 5)?; // subtract, divide, round, max, min
    emit(Build, 4, 1, 2)?; // clip endpoint from_f32 constructors
    emit(Evaluate, 4, n, 1)?; // uint32 cast
    emit(Build, 4, 32 / bits, 1)?; // place Array::from_slice
    emit(Evaluate, 4, n, 1)?; // disjoint-field multiplication
                              // Last-axis extent <=16. output==1 implies input<=16, so the native all-reduce
                              // input>4096 branch is impossible. Preserve CPU zero-output vs Metal minimum-4 semantics.
    let output = words.checked_mul(4).ok_or_else(overflow)?;
    allocate(Evaluate, if metal { output.max(4) } else { output })
}

pub(crate) fn visit_stored_grid_allocations(
    rows: u64,
    input: u64,
    group: u64,
    bits: u64,
    metal: bool,
    mut allocate: impl FnMut(u64) -> Result<()>,
) -> Result<()> {
    visit_stored_grid_stages(rows, input, group, bits, metal, |_, bytes| allocate(bytes))
}

pub(crate) fn stored_grid_allocation_parts(
    rows: u64,
    input: u64,
    group: u64,
    bits: u64,
) -> Result<(u64, u64)> {
    let mut build = 0u64;
    let mut evaluate = 0u64;
    visit_stored_grid_stages(rows, input, group, bits, true, |stage, bytes| {
        let bytes =
            mlx_gen::ios_quantum::fresh_metal_allocation_bytes(bytes, crate::retention::PAGE_BYTES)
                .ok_or_else(|| {
                    Error::Unsupported("Iris stored-grid fresh allocation overflow".into())
                })?;
        let sum = if stage == QuantAllocationStage::Build {
            &mut build
        } else {
            &mut evaluate
        };
        *sum = sum
            .checked_add(bytes)
            .ok_or_else(|| Error::Unsupported("Iris stored-grid fresh sum overflow".into()))?;
        Ok(())
    })?;
    Ok((build, evaluate))
}

pub(crate) fn stored_grid_allocation_bytes(
    rows: u64,
    input: u64,
    group: u64,
    bits: u64,
) -> Result<u64> {
    let (build, evaluate) = stored_grid_allocation_parts(rows, input, group, bits)?;
    build
        .checked_add(evaluate)
        .ok_or_else(|| Error::Unsupported("Iris stored-grid complete sum overflow".into()))
}

pub(crate) const PENDING_CONSTRUCTION_QUANTUM: &str = "Iris pending construction";

/// A real check immediately before constructor graph creation. The closure is intentionally
/// lazy: disabled execution performs NONE of the new budget-only facts/overflow arithmetic.
pub(crate) fn before_pending_construction(plan: impl FnOnce() -> Result<u64>) -> Result<()> {
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(());
    }
    mlx_gen::ios_quantum::before(PENDING_CONSTRUCTION_QUANTUM, plan()?)
}
