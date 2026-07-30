//! Shared Stable Audio 3 transformer primitives.
//!
//! Tensor names and arithmetic mirror frozen upstream commit
//! `124e8a799f57a1f665495ecb72e547d0a62867f1`.

use candle_audio::candle_core::{bail, DType, Device, Result, Tensor, D};
use candle_nn::{linear, linear_b, linear_no_bias, Init, Linear, Module, VarBuilder};

use crate::config::{FeedForwardConfig, NormConfig, NormType, QkNorm};

pub struct DynamicTanh {
    alpha: Tensor,
    gamma: Tensor,
    beta: Tensor,
}

impl DynamicTanh {
    pub fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get_with_hints(1, "alpha", Init::Const(4.0))?,
            gamma: vb.get_with_hints(dim, "gamma", Init::Const(1.0))?,
            beta: vb.get_with_hints(dim, "beta", Init::Const(0.0))?,
        })
    }

    pub fn from_tensors(alpha: Tensor, gamma: Tensor, beta: Tensor) -> Self {
        Self { alpha, gamma, beta }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.alpha)?
            .tanh()?
            .broadcast_mul(&self.gamma)?
            .broadcast_add(&self.beta)
    }
}

enum NormInner {
    Layer { gamma: Tensor, beta: Tensor },
    Rms { gamma: Tensor },
    Dyt(DynamicTanh),
}

/// Upstream LayerNorm/RMSNorm/DynamicTanh including `fix_scale` and `force_fp32`.
pub struct Norm {
    inner: NormInner,
    eps: f64,
    force_fp32: bool,
}

impl Norm {
    pub fn load(kind: NormType, dim: usize, cfg: &NormConfig, vb: VarBuilder) -> Result<Self> {
        let inner = match kind {
            NormType::LayerNorm => NormInner::Layer {
                // A fixed scale is still a persistent state-dict buffer upstream.
                gamma: vb.get_with_hints(dim, "gamma", Init::Const(1.0))?,
                beta: vb.get_with_hints(dim, "beta", Init::Const(0.0))?,
            },
            NormType::RmsNorm => NormInner::Rms {
                gamma: vb.get_with_hints(dim, "gamma", Init::Const(1.0))?,
            },
            NormType::Dyt => NormInner::Dyt(DynamicTanh::load(dim, vb)?),
        };
        let _ = cfg.fix_scale;
        Ok(Self {
            inner,
            eps: cfg.eps,
            force_fp32: cfg.force_fp32,
        })
    }

    fn load_torch_layer_norm(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            inner: NormInner::Layer {
                gamma: vb.get_with_hints(dim, "weight", Init::Const(1.0))?,
                beta: vb.get_with_hints(dim, "bias", Init::Const(0.0))?,
            },
            eps,
            force_fp32: false,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let NormInner::Dyt(dyt) = &self.inner {
            return dyt.forward(x);
        }
        let original = x.dtype();
        let x = if self.force_fp32 && original != DType::F32 {
            x.to_dtype(DType::F32)?
        } else {
            x.clone()
        };
        let y = match &self.inner {
            NormInner::Layer { gamma, beta } => {
                let gamma = gamma.to_dtype(x.dtype())?;
                let beta = beta.to_dtype(x.dtype())?;
                let mean = x.mean_keepdim(D::Minus1)?;
                let centered = x.broadcast_sub(&mean)?;
                let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
                centered
                    .broadcast_div(&(var + self.eps)?.sqrt()?)?
                    .broadcast_mul(&gamma)?
                    .broadcast_add(&beta)?
            }
            NormInner::Rms { gamma } => {
                let gamma = gamma.to_dtype(x.dtype())?;
                let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
                x.broadcast_div(&(ms + self.eps)?.sqrt()?)?
                    .broadcast_mul(&gamma)?
            }
            NormInner::Dyt(_) => unreachable!(),
        };
        if y.dtype() == original {
            Ok(y)
        } else {
            y.to_dtype(original)
        }
    }
}

/// fp32 half-split RoPE (`[x1,x2] -> [-x2,x1]`), not the interleaved Llama variant.
pub struct RotaryEmbedding {
    inv_freq: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, device: &Device) -> Result<Self> {
        if dim < 2 || !dim.is_multiple_of(2) {
            bail!("rotary dimension must be positive and even, got {dim}")
        }
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| (1.0 / 10_000f64.powf(i as f64 / dim as f64)) as f32)
            .collect();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv, dim / 2, device)?,
        })
    }

    /// Load the persisted upstream `rope.inv_freq` buffer.
    ///
    /// SAME checkpoints serialize this buffer for every transformer block. Loading it instead of
    /// silently reconstructing the usual base-10,000 values makes the checkpoint the source of
    /// truth and rejects shape-incompatible rotary state.
    pub fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        if dim < 2 || !dim.is_multiple_of(2) {
            bail!("rotary dimension must be positive and even, got {dim}")
        }
        Ok(Self {
            // RoPE is intentionally evaluated in fp32 even when model weights use fp16.
            // Request the persisted buffer at its compute dtype instead of inheriting the
            // VarBuilder dtype; otherwise fp32 positions and fp16 frequencies cannot matmul.
            inv_freq: vb.get_with_hints_dtype(
                dim / 2,
                "inv_freq",
                Default::default(),
                DType::F32,
            )?,
        })
    }

    pub fn frequencies(&self, len: usize) -> Result<Tensor> {
        self.frequencies_range(0, len)
    }

    /// Build absolute-position frequencies for a bounded sequence range.
    ///
    /// SAME-L attention projects query tiles and their key/value halos independently.  Generating
    /// the table from the global range keeps RoPE positions identical to the unchunked model without
    /// retaining a full-sequence table in every layer.
    pub fn frequencies_range(&self, start: usize, len: usize) -> Result<Tensor> {
        let end = start
            .checked_add(len)
            .ok_or_else(|| candle_audio::candle_core::Error::Msg("RoPE range overflow".into()))?;
        let start = u32::try_from(start)
            .map_err(|_| candle_audio::candle_core::Error::Msg("RoPE start exceeds u32".into()))?;
        let end = u32::try_from(end)
            .map_err(|_| candle_audio::candle_core::Error::Msg("RoPE end exceeds u32".into()))?;
        let positions = Tensor::arange(start, end, self.inv_freq.device())?.to_dtype(DType::F32)?;
        let freqs = positions
            .unsqueeze(1)?
            .matmul(&self.inv_freq.unsqueeze(0)?)?;
        Tensor::cat(&[&freqs, &freqs], 1)
    }
}

pub fn apply_rotary(t: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let original = t.dtype();
    let seq = t.dim(D::Minus2)?;
    let rot_dim = freqs.dim(D::Minus1)?;
    let freq_seq = freqs.dim(D::Minus2)?;
    if rot_dim > t.dim(D::Minus1)? || seq > freq_seq || rot_dim % 2 != 0 {
        bail!(
            "invalid RoPE shapes {:?} and {:?}",
            t.shape(),
            freqs.shape()
        )
    }
    let t32 = t.to_dtype(DType::F32)?;
    let freqs = freqs
        .narrow(D::Minus2, freq_seq - seq, seq)?
        .to_dtype(DType::F32)?;
    let rotating = t32.narrow(D::Minus1, 0, rot_dim)?;
    let half = rot_dim / 2;
    let first = rotating.narrow(D::Minus1, 0, half)?;
    let second = rotating.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
    let y = rotating
        .broadcast_mul(&freqs.cos()?)?
        .broadcast_add(&rotated.broadcast_mul(&freqs.sin()?)?)?;
    let y = if rot_dim < t32.dim(D::Minus1)? {
        Tensor::cat(
            &[
                &y,
                &t32.narrow(D::Minus1, rot_dim, t32.dim(D::Minus1)? - rot_dim)?,
            ],
            D::Minus1,
        )?
    } else {
        y
    };
    y.to_dtype(original)?.contiguous()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedForwardActivation {
    Silu,
    SinPi,
}

pub struct FeedForward {
    input: Linear,
    output: Linear,
    inner_dim: usize,
    glu: bool,
    activation: FeedForwardActivation,
}

impl FeedForward {
    /// Load from a `FeedForward` root (`ff.0.proj`, `ff.2` in the state dict).
    pub fn load(dim: usize, cfg: &FeedForwardConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.use_conv {
            bail!("Stable Audio 3 shipped transformer feed-forwards are linear, not Conv1d")
        }
        let inner_dim = (dim as f64 * cfg.mult) as usize;
        let input_dim = if cfg.glu { inner_dim * 2 } else { inner_dim };
        let input_vb = if cfg.glu {
            vb.pp("ff.0.proj")
        } else {
            vb.pp("ff.0")
        };
        Ok(Self {
            input: linear_b(dim, input_dim, !cfg.no_bias, input_vb)?,
            output: linear_b_initialized(
                inner_dim,
                dim,
                !cfg.no_bias,
                cfg.zero_init_output,
                vb.pp("ff.2"),
            )?,
            inner_dim,
            glu: cfg.glu,
            activation: if cfg.sinusoidal {
                FeedForwardActivation::SinPi
            } else {
                FeedForwardActivation::Silu
            },
        })
    }

    fn activate(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation {
            FeedForwardActivation::Silu => candle_nn::ops::silu(x),
            FeedForwardActivation::SinPi => (x * std::f64::consts::PI)?.sin(),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let projected = self.input.forward(x)?;
        let hidden = if self.glu {
            let values = projected.narrow(D::Minus1, 0, self.inner_dim)?;
            let gate = projected.narrow(D::Minus1, self.inner_dim, self.inner_dim)?;
            values.broadcast_mul(&self.activate(&gate)?)?
        } else {
            self.activate(&projected)?
        };
        self.output.forward(&hidden)
    }
}

pub struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    pub fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale: vb.get_with_hints(dim, "scale", Init::Const(1e-5))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.scale)
    }
}

fn linear_b_initialized(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    zero_init: bool,
    vb: VarBuilder,
) -> Result<Linear> {
    if !zero_init {
        return linear_b(in_dim, out_dim, bias, vb);
    }
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", Init::Const(0.0))?;
    let bias = bias
        .then(|| vb.get_with_hints(out_dim, "bias", Init::Const(0.0)))
        .transpose()?;
    Ok(Linear::new(weight, bias))
}

fn to_heads(x: &Tensor, heads: usize, dim_head: usize) -> Result<Tensor> {
    let (batch, seq, _) = x.dims3()?;
    x.reshape((batch, seq, heads, dim_head))?
        .transpose(1, 2)?
        .contiguous()
}

fn from_heads(x: &Tensor) -> Result<Tensor> {
    let (batch, heads, seq, dim) = x.dims4()?;
    x.transpose(1, 2)?.reshape((batch, seq, heads * dim))
}

fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return x.contiguous();
    }
    let (b, h, n, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, groups, n, d))?
        .reshape((b, h * groups, n, d))?
        .contiguous()
}

fn combine_differential(ordinary: &Tensor, differential: &Tensor) -> Result<Tensor> {
    ordinary - differential
}

fn ada_ln_gate(gate: &Tensor) -> Result<Tensor> {
    candle_nn::ops::sigmoid(&gate.affine(-1.0, 1.0)?)
}

enum Projection {
    SelfFused(Linear),
    Cross { q: Linear, kv: Linear },
}

enum QkNormLayer {
    None,
    L2(f64),
    Norm { q: Norm, k: Norm },
}

pub struct Attention {
    projection: Projection,
    to_out: Linear,
    qk_norm: QkNormLayer,
    num_heads: usize,
    kv_heads: usize,
    dim_head: usize,
    differential: bool,
    causal: bool,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        dim: usize,
        dim_head: usize,
        dim_context: Option<usize>,
        qk_kind: QkNorm,
        qk_eps: f64,
        differential: bool,
        zero_init_output: bool,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let context_dim = dim_context.unwrap_or(dim);
        if !dim.is_multiple_of(dim_head) || !context_dim.is_multiple_of(dim_head) {
            bail!("attention dimensions must be divisible by head dimension")
        }
        let projection = if dim_context.is_some() {
            Projection::Cross {
                q: linear_no_bias(dim, dim * if differential { 2 } else { 1 }, vb.pp("to_q"))?,
                kv: linear_no_bias(
                    context_dim,
                    context_dim * if differential { 3 } else { 2 },
                    vb.pp("to_kv"),
                )?,
            }
        } else {
            Projection::SelfFused(linear_no_bias(
                dim,
                dim * if differential { 5 } else { 3 },
                vb.pp("to_qkv"),
            )?)
        };
        let qk_norm = match qk_kind {
            QkNorm::None => QkNormLayer::None,
            QkNorm::L2 => QkNormLayer::L2(qk_eps),
            QkNorm::Ln => QkNormLayer::Norm {
                // Attention uses torch.nn.LayerNorm, whose state-dict keys differ from the
                // custom block LayerNorm's gamma/beta keys.
                q: Norm::load_torch_layer_norm(dim_head, qk_eps, vb.pp("q_norm"))?,
                k: Norm::load_torch_layer_norm(dim_head, qk_eps, vb.pp("k_norm"))?,
            },
            kind => {
                let norm_kind = match kind {
                    QkNorm::Rms => NormType::RmsNorm,
                    QkNorm::Dyt => NormType::Dyt,
                    _ => unreachable!(),
                };
                let cfg = NormConfig {
                    fix_scale: false,
                    force_fp32: false,
                    eps: qk_eps,
                };
                QkNormLayer::Norm {
                    q: Norm::load(norm_kind, dim_head, &cfg, vb.pp("q_norm"))?,
                    k: Norm::load(norm_kind, dim_head, &cfg, vb.pp("k_norm"))?,
                }
            }
        };
        Ok(Self {
            projection,
            to_out: linear_b_initialized(dim, dim, false, zero_init_output, vb.pp("to_out"))?,
            qk_norm,
            num_heads: dim / dim_head,
            kv_heads: context_dim / dim_head,
            dim_head,
            differential,
            causal,
        })
    }

    fn normalize(&self, q: Tensor, k: Tensor) -> Result<(Tensor, Tensor)> {
        match &self.qk_norm {
            QkNormLayer::None => Ok((q, k)),
            QkNormLayer::L2(eps) => {
                let qn = q
                    .sqr()?
                    .sum_keepdim(D::Minus1)?
                    .sqrt()?
                    .clamp(*eps, f64::INFINITY)?;
                let kn = k
                    .sqr()?
                    .sum_keepdim(D::Minus1)?
                    .sqrt()?
                    .clamp(*eps, f64::INFINITY)?;
                Ok((q.broadcast_div(&qn)?, k.broadcast_div(&kn)?))
            }
            QkNormLayer::Norm { q: qn, k: kn } => Ok((qn.forward(&q)?, kn.forward(&k)?)),
        }
    }

    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        padding_mask: Option<&Tensor>,
        additive_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let groups = self.num_heads / self.kv_heads;
        let k = repeat_kv(k, groups)?;
        let mut v = repeat_kv(v, groups)?;
        if let Some(mask) = padding_mask {
            v = v.broadcast_mul(
                &mask
                    .unsqueeze(1)?
                    .unsqueeze(D::Minus1)?
                    .to_dtype(v.dtype())?,
            )?;
        }
        let scale = 1.0 / (self.dim_head as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = additive_mask {
            scores = scores.broadcast_add(mask)?;
        }
        if self.causal && q.dim(2)? > 1 {
            let n = q.dim(2)?;
            let m = k.dim(2)?;
            let mut values = vec![0f32; n * m];
            for i in 0..n {
                for j in (i + 1)..m {
                    values[i * m + j] = f32::NEG_INFINITY;
                }
            }
            let mask = Tensor::from_vec(values, (1, 1, n, m), q.device())?;
            scores = scores.broadcast_add(&mask.to_dtype(scores.dtype())?)?;
        }
        candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        rotary: Option<&Tensor>,
        rotary_k: Option<&Tensor>,
        padding_mask: Option<&Tensor>,
        additive_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let source = context.unwrap_or(x);
        let (mut q, mut k, v, mut q_diff, mut k_diff) = match &self.projection {
            Projection::SelfFused(proj) => {
                let all = proj.forward(x)?;
                let dim = self.num_heads * self.dim_head;
                let q = to_heads(
                    &all.narrow(D::Minus1, 0, dim)?,
                    self.num_heads,
                    self.dim_head,
                )?;
                let k = to_heads(
                    &all.narrow(D::Minus1, dim, dim)?,
                    self.num_heads,
                    self.dim_head,
                )?;
                let v = to_heads(
                    &all.narrow(D::Minus1, 2 * dim, dim)?,
                    self.num_heads,
                    self.dim_head,
                )?;
                let qd = self
                    .differential
                    .then(|| {
                        to_heads(
                            &all.narrow(D::Minus1, 3 * dim, dim)?,
                            self.num_heads,
                            self.dim_head,
                        )
                    })
                    .transpose()?;
                let kd = self
                    .differential
                    .then(|| {
                        to_heads(
                            &all.narrow(D::Minus1, 4 * dim, dim)?,
                            self.num_heads,
                            self.dim_head,
                        )
                    })
                    .transpose()?;
                (q, k, v, qd, kd)
            }
            Projection::Cross { q: qp, kv } => {
                let q_all = qp.forward(x)?;
                let kv_all = kv.forward(source)?;
                let q_dim = self.num_heads * self.dim_head;
                let kv_dim = self.kv_heads * self.dim_head;
                let q = to_heads(
                    &q_all.narrow(D::Minus1, 0, q_dim)?,
                    self.num_heads,
                    self.dim_head,
                )?;
                let k = to_heads(
                    &kv_all.narrow(D::Minus1, 0, kv_dim)?,
                    self.kv_heads,
                    self.dim_head,
                )?;
                let qd = self
                    .differential
                    .then(|| {
                        to_heads(
                            &q_all.narrow(D::Minus1, q_dim, q_dim)?,
                            self.num_heads,
                            self.dim_head,
                        )
                    })
                    .transpose()?;
                let (kd, v_offset) = if self.differential {
                    (
                        Some(to_heads(
                            &kv_all.narrow(D::Minus1, kv_dim, kv_dim)?,
                            self.kv_heads,
                            self.dim_head,
                        )?),
                        2 * kv_dim,
                    )
                } else {
                    (None, kv_dim)
                };
                let v = to_heads(
                    &kv_all.narrow(D::Minus1, v_offset, kv_dim)?,
                    self.kv_heads,
                    self.dim_head,
                )?;
                (q, k, v, qd, kd)
            }
        };
        (q, k) = self.normalize(q, k)?;
        if let (Some(qd), Some(kd)) = (q_diff.take(), k_diff.take()) {
            let normalized = self.normalize(qd, kd)?;
            q_diff = Some(normalized.0);
            k_diff = Some(normalized.1);
        }
        if let Some(freqs) = rotary {
            let q_len = q.dim(2)?;
            let k_len = k.dim(2)?;
            let (qf, kf) = if let Some(kf) = rotary_k {
                (freqs.clone(), kf.clone())
            } else if q_len >= k_len {
                (freqs.clone(), (freqs * (q_len as f64 / k_len as f64))?)
            } else {
                ((freqs * (k_len as f64 / q_len as f64))?, freqs.clone())
            };
            q = apply_rotary(&q, &qf)?;
            k = apply_rotary(&k, &kf)?;
            if let Some(qd) = q_diff.as_mut() {
                *qd = apply_rotary(qd, &qf)?;
            }
            if let Some(kd) = k_diff.as_mut() {
                *kd = apply_rotary(kd, &kf)?;
            }
        }
        let mut out = self.attend(&q, &k, &v, padding_mask, additive_mask)?;
        if let (Some(qd), Some(kd)) = (&q_diff, &k_diff) {
            // Stable Audio 3 differential attention is a direct subtraction. There is no lambda.
            out =
                combine_differential(&out, &self.attend(qd, kd, &v, padding_mask, additive_mask)?)?;
        }
        self.to_out.forward(&from_heads(&out)?)
    }

    /// Execute one bounded query tile of self-attention against its key/value halo.
    ///
    /// The fused projection is evaluated only for the query rows and halo rows.  Unused projection
    /// components are discarded, but no full-sequence Q/K/V tensor or score matrix is materialized.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_self_band_tile(
        &self,
        query_input: &Tensor,
        key_value_input: &Tensor,
        query_start: usize,
        key_start: usize,
        left: usize,
        right: usize,
        query_rotary: &Tensor,
        key_rotary: &Tensor,
    ) -> Result<Tensor> {
        if self.causal {
            bail!("band self-attention does not support causal mode")
        }
        let Projection::SelfFused(projection) = &self.projection else {
            bail!("band self-attention requires the fused self-attention projection")
        };
        let query_all = projection.forward(query_input)?;
        let key_value_all = projection.forward(key_value_input)?;
        let dim = self.num_heads * self.dim_head;
        let mut q = to_heads(
            &query_all.narrow(D::Minus1, 0, dim)?,
            self.num_heads,
            self.dim_head,
        )?;
        let mut k = to_heads(
            &key_value_all.narrow(D::Minus1, dim, dim)?,
            self.num_heads,
            self.dim_head,
        )?;
        let v = to_heads(
            &key_value_all.narrow(D::Minus1, 2 * dim, dim)?,
            self.num_heads,
            self.dim_head,
        )?;
        let mut q_diff = self
            .differential
            .then(|| {
                to_heads(
                    &query_all.narrow(D::Minus1, 3 * dim, dim)?,
                    self.num_heads,
                    self.dim_head,
                )
            })
            .transpose()?;
        let mut k_diff = self
            .differential
            .then(|| {
                to_heads(
                    &key_value_all.narrow(D::Minus1, 4 * dim, dim)?,
                    self.num_heads,
                    self.dim_head,
                )
            })
            .transpose()?;
        (q, k) = self.normalize(q, k)?;
        if let (Some(qd), Some(kd)) = (q_diff.take(), k_diff.take()) {
            let (qd, kd) = self.normalize(qd, kd)?;
            q_diff = Some(qd);
            k_diff = Some(kd);
        }
        q = apply_rotary(&q, query_rotary)?;
        k = apply_rotary(&k, key_rotary)?;
        if let Some(qd) = q_diff.as_mut() {
            *qd = apply_rotary(qd, query_rotary)?;
        }
        if let Some(kd) = k_diff.as_mut() {
            *kd = apply_rotary(kd, key_rotary)?;
        }
        let mask = sliding_window_additive_mask_with_offsets(
            query_start,
            query_input.dim(1)?,
            key_start,
            key_value_input.dim(1)?,
            left,
            right,
            q.dtype(),
            q.device(),
        )?;
        let mut out = self.attend(&q, &k, &v, None, Some(&mask))?;
        if let (Some(qd), Some(kd)) = (&q_diff, &k_diff) {
            out = combine_differential(&out, &self.attend(qd, kd, &v, None, Some(&mask))?)?;
        }
        self.to_out.forward(&from_heads(&out)?)
    }
}

/// Learned memory-token prepend/extend/trim seam used by the shipped small and medium DiTs.
pub struct MemoryTokens {
    tokens: Tensor,
}

impl MemoryTokens {
    pub fn load(count: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            tokens: vb.get_with_hints(
                (count, dim),
                "memory_tokens",
                Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
            )?,
        })
    }

    pub fn prepend(
        &self,
        x: &Tensor,
        padding_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (batch, _, dim) = x.dims3()?;
        let count = self.tokens.dim(0)?;
        let memory = self.tokens.unsqueeze(0)?.expand((batch, count, dim))?;
        let x = Tensor::cat(&[&memory, x], 1)?;
        let mask = match padding_mask {
            Some(mask) => {
                let valid = Tensor::ones((batch, count), mask.dtype(), mask.device())?;
                Some(Tensor::cat(&[&valid, mask], 1)?)
            }
            None => None,
        };
        Ok((x, mask))
    }

    pub fn trim(&self, x: &Tensor) -> Result<Tensor> {
        let count = self.tokens.dim(0)?;
        x.narrow(1, count, x.dim(1)? - count)
    }
}

pub struct LocalConditioning {
    first: Linear,
    second: Linear,
}

impl LocalConditioning {
    pub fn load(input_dim: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            first: linear(input_dim, dim, vb.pp("0"))?,
            second: linear_b_initialized(dim, dim, true, true, vb.pp("2"))?,
        })
    }

    pub fn forward(&self, input: &Tensor, target_len: usize) -> Result<Tensor> {
        let mut x = self
            .second
            .forward(&candle_nn::ops::silu(&self.first.forward(input)?)?)?;
        let len = x.dim(1)?;
        if len < target_len {
            let (batch, _, dim) = x.dims3()?;
            let zeros = Tensor::zeros((batch, target_len - len, dim), x.dtype(), x.device())?;
            x = Tensor::cat(&[&zeros, &x], 1)?;
        } else if len > target_len {
            x = x.narrow(1, len - target_len, target_len)?;
        }
        Ok(x)
    }
}

/// Independent masks for one attention branch.
#[derive(Clone, Copy, Default)]
pub struct AttentionMasks<'a> {
    /// `[batch, key_len]` zero/one validity mask. Matching upstream, invalid keys zero `v`.
    pub key_padding: Option<&'a Tensor>,
    /// Additive score mask broadcastable to `[batch, heads, query_len, key_len]`.
    pub additive: Option<&'a Tensor>,
}

/// Independent self- and cross-attention masks for an assembled transformer block.
#[derive(Clone, Copy, Default)]
pub struct TransformerBlockMasks<'a> {
    pub self_attention: AttentionMasks<'a>,
    pub cross_attention: AttentionMasks<'a>,
}

/// Build upstream's exact band mask: zero for `j-i ∈ [-left, right]`, `-inf` outside.
pub fn sliding_window_additive_mask(
    query_len: usize,
    key_len: usize,
    left: usize,
    right: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    sliding_window_additive_mask_with_offsets(0, query_len, 0, key_len, left, right, dtype, device)
}

/// Build a local band mask using absolute query/key offsets.
#[allow(clippy::too_many_arguments)]
pub fn sliding_window_additive_mask_with_offsets(
    query_start: usize,
    query_len: usize,
    key_start: usize,
    key_len: usize,
    left: usize,
    right: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut values = vec![0f32; query_len * key_len];
    for i in 0..query_len {
        for j in 0..key_len {
            let query = query_start.saturating_add(i);
            let key = key_start.saturating_add(j);
            if key.saturating_add(left) < query || key > query.saturating_add(right) {
                values[i * key_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(values, (query_len, key_len), device)?.to_dtype(dtype)
}

/// One bounded query tile and the key/value halo required to evaluate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandAttentionTile {
    pub query_start: usize,
    pub query_len: usize,
    pub key_start: usize,
    pub key_len: usize,
}

/// Pure allocation plan used by the SAME-L runtime and its no-dense-fallback structural tests.
pub fn band_attention_plan(
    sequence_len: usize,
    left: usize,
    right: usize,
    query_tile: usize,
) -> Result<Vec<BandAttentionTile>> {
    if query_tile == 0 {
        bail!("band attention query tile must be non-zero")
    }
    let mut tiles = Vec::with_capacity(sequence_len.div_ceil(query_tile));
    for query_start in (0..sequence_len).step_by(query_tile) {
        let query_end = sequence_len.min(query_start.saturating_add(query_tile));
        let key_start = query_start.saturating_sub(left);
        let key_end = sequence_len.min(query_end.saturating_add(right));
        tiles.push(BandAttentionTile {
            query_start,
            query_len: query_end - query_start,
            key_start,
            key_len: key_end - key_start,
        });
    }
    Ok(tiles)
}

pub struct TransformerBlock {
    pre_norm: Norm,
    self_attn: Attention,
    cross: Option<(Norm, Attention)>,
    ff_norm: Norm,
    ff: FeedForward,
    self_scale: Option<LayerScale>,
    cross_scale: Option<LayerScale>,
    ff_scale: Option<LayerScale>,
    to_scale_shift_gate: Option<Tensor>,
    local: Option<LocalConditioning>,
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        dim: usize,
        dim_head: usize,
        dim_context: Option<usize>,
        norm_type: NormType,
        norm_cfg: &NormConfig,
        qk_norm: QkNorm,
        qk_eps: f64,
        differential: bool,
        causal: bool,
        ff_cfg: &FeedForwardConfig,
        zero_init_branch_outputs: bool,
        global_conditioning: bool,
        local_cond_dim: Option<usize>,
        layer_scale: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Upstream makes LayerScale and zero-initialized branch outputs mutually exclusive.
        let zero_init_branch_outputs = zero_init_branch_outputs && !layer_scale;
        let cross = dim_context
            .map(|context_dim| -> Result<_> {
                Ok((
                    Norm::load(norm_type, dim, norm_cfg, vb.pp("cross_attend_norm"))?,
                    Attention::load(
                        dim,
                        dim_head,
                        Some(context_dim),
                        qk_norm,
                        qk_eps,
                        differential,
                        zero_init_branch_outputs,
                        causal,
                        vb.pp("cross_attn"),
                    )?,
                ))
            })
            .transpose()?;
        Ok(Self {
            pre_norm: Norm::load(norm_type, dim, norm_cfg, vb.pp("pre_norm"))?,
            self_attn: Attention::load(
                dim,
                dim_head,
                None,
                qk_norm,
                qk_eps,
                differential,
                zero_init_branch_outputs,
                causal,
                vb.pp("self_attn"),
            )?,
            cross,
            ff_norm: Norm::load(norm_type, dim, norm_cfg, vb.pp("ff_norm"))?,
            ff: {
                let mut effective = ff_cfg.clone();
                effective.zero_init_output = zero_init_branch_outputs;
                FeedForward::load(dim, &effective, vb.pp("ff"))?
            },
            self_scale: layer_scale
                .then(|| LayerScale::load(dim, vb.pp("self_attn_scale")))
                .transpose()?,
            cross_scale: (layer_scale && dim_context.is_some())
                .then(|| LayerScale::load(dim, vb.pp("cross_attn_scale")))
                .transpose()?,
            ff_scale: layer_scale
                .then(|| LayerScale::load(dim, vb.pp("ff_scale")))
                .transpose()?,
            to_scale_shift_gate: global_conditioning
                .then(|| {
                    vb.get_with_hints(
                        6 * dim,
                        "to_scale_shift_gate",
                        Init::Randn {
                            mean: 0.0,
                            stdev: 1.0 / (dim as f64).sqrt(),
                        },
                    )
                })
                .transpose()?,
            local: local_cond_dim
                .map(|input| LocalConditioning::load(input, dim, vb.pp("to_local_embed")))
                .transpose()?,
        })
    }

    fn scale(scale: &Option<LayerScale>, x: &Tensor) -> Result<Tensor> {
        match scale {
            Some(scale) => scale.forward(x),
            None => Ok(x.clone()),
        }
    }

    /// Project one block's local conditioning for provenance-locked parity traces.
    pub fn project_local(&self, input: &Tensor, target_len: usize) -> Result<Option<Tensor>> {
        self.local
            .as_ref()
            .map(|local| local.forward(input, target_len))
            .transpose()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        global_cond: Option<&Tensor>,
        local_cond: Option<&Tensor>,
        rotary: Option<&Tensor>,
        cross_rotary: Option<&Tensor>,
        masks: TransformerBlockMasks<'_>,
    ) -> Result<Tensor> {
        let dim = x.dim(D::Minus1)?;
        let mut x = x.clone();
        if let (Some(base), Some(global)) = (&self.to_scale_shift_gate, global_cond) {
            let modulation = base.broadcast_add(global)?.unsqueeze(1)?;
            let part = |i| modulation.narrow(D::Minus1, i * dim, dim);
            let (ss, shs, gs, sf, shf, gf) =
                (part(0)?, part(1)?, part(2)?, part(3)?, part(4)?, part(5)?);
            let normalized = self
                .pre_norm
                .forward(&x)?
                .broadcast_mul(&(ss + 1.0)?)?
                .broadcast_add(&shs)?;
            let attn = self.self_attn.forward(
                &normalized,
                None,
                rotary,
                None,
                masks.self_attention.key_padding,
                masks.self_attention.additive,
            )?;
            // Exact upstream gate: sigmoid(1 - gate), not sigmoid(gate).
            let gate_self = ada_ln_gate(&gs)?;
            x = (&x + Self::scale(&self.self_scale, &attn.broadcast_mul(&gate_self)?)?)?;
            if let (Some((norm, cross)), Some(context)) = (&self.cross, context) {
                let (query_rope, key_rope) = if cross_rotary.is_some() {
                    (rotary, cross_rotary)
                } else {
                    (None, None)
                };
                let h = cross.forward(
                    &norm.forward(&x)?,
                    Some(context),
                    query_rope,
                    key_rope,
                    masks.cross_attention.key_padding,
                    masks.cross_attention.additive,
                )?;
                x = (&x + Self::scale(&self.cross_scale, &h)?)?;
            }
            if let (Some(local), Some(input)) = (&self.local, local_cond) {
                x = (&x + local.forward(input, x.dim(1)?)?)?;
            }
            let normalized = self
                .ff_norm
                .forward(&x)?
                .broadcast_mul(&(sf + 1.0)?)?
                .broadcast_add(&shf)?;
            let ff = self.ff.forward(&normalized)?;
            let gate_ff = ada_ln_gate(&gf)?;
            x = (&x + Self::scale(&self.ff_scale, &ff.broadcast_mul(&gate_ff)?)?)?;
        } else {
            let attn = self.self_attn.forward(
                &self.pre_norm.forward(&x)?,
                None,
                rotary,
                None,
                masks.self_attention.key_padding,
                masks.self_attention.additive,
            )?;
            x = (&x + Self::scale(&self.self_scale, &attn)?)?;
            if let (Some((norm, cross)), Some(context)) = (&self.cross, context) {
                let (query_rope, key_rope) = if cross_rotary.is_some() {
                    (rotary, cross_rotary)
                } else {
                    (None, None)
                };
                let h = cross.forward(
                    &norm.forward(&x)?,
                    Some(context),
                    query_rope,
                    key_rope,
                    masks.cross_attention.key_padding,
                    masks.cross_attention.additive,
                )?;
                x = (&x + Self::scale(&self.cross_scale, &h)?)?;
            }
            if let (Some(local), Some(input)) = (&self.local, local_cond) {
                x = (&x + local.forward(input, x.dim(1)?)?)?;
            }
            let ff = self.ff.forward(&self.ff_norm.forward(&x)?)?;
            x = (&x + Self::scale(&self.ff_scale, &ff)?)?;
        }
        Ok(x)
    }

    /// Bounded global sliding-window execution for SAME-L.
    ///
    /// Every normalization, attention projection, score matrix, and feed-forward intermediate is
    /// evaluated per query tile.  The returned hidden state remains globally packed so all twelve
    /// layers share absolute sequence positions.
    pub fn forward_band(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        left: usize,
        right: usize,
        query_tile: usize,
    ) -> Result<Tensor> {
        if self.cross.is_some() || self.to_scale_shift_gate.is_some() || self.local.is_some() {
            bail!("SAME band blocks support only unconditioned self-attention")
        }
        let sequence_len = x.dim(1)?;
        let plan = band_attention_plan(sequence_len, left, right, query_tile)?;
        let mut attended = Vec::with_capacity(plan.len());
        for tile in &plan {
            let query = x.narrow(1, tile.query_start, tile.query_len)?;
            let keys = x.narrow(1, tile.key_start, tile.key_len)?;
            let normalized_query = self.pre_norm.forward(&query)?;
            let normalized_keys = self.pre_norm.forward(&keys)?;
            let query_rope = rope.frequencies_range(tile.query_start, tile.query_len)?;
            let key_rope = rope.frequencies_range(tile.key_start, tile.key_len)?;
            let attention = self.self_attn.forward_self_band_tile(
                &normalized_query,
                &normalized_keys,
                tile.query_start,
                tile.key_start,
                left,
                right,
                &query_rope,
                &key_rope,
            )?;
            attended.push((&query + Self::scale(&self.self_scale, &attention)?)?);
        }
        let attended_refs: Vec<_> = attended.iter().collect();
        let attended = Tensor::cat(&attended_refs, 1)?;

        let mut outputs = Vec::with_capacity(plan.len());
        for tile in &plan {
            let input = attended.narrow(1, tile.query_start, tile.query_len)?;
            let ff = self.ff.forward(&self.ff_norm.forward(&input)?)?;
            outputs.push((&input + Self::scale(&self.ff_scale, &ff)?)?);
        }
        let output_refs: Vec<_> = outputs.iter().collect();
        Tensor::cat(&output_refs, 1)
    }
}

/// Upstream's strict decoder rule: `(depth - block_index) < sinusoidal_blocks`.
pub fn is_sinusoidal_block(block_index: usize, depth: usize, threshold: usize) -> bool {
    depth.saturating_sub(block_index) < threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn same_l_exact_max_band_plan_is_bounded() {
        let plan = band_attention_plan(69_632, 17, 17, 1024).unwrap();
        assert_eq!(plan.len(), 68);
        assert_eq!(plan.first().unwrap().query_start, 0);
        assert_eq!(plan.last().unwrap().query_start, 68_608);
        assert_eq!(plan.iter().map(|tile| tile.key_len).max(), Some(1058));
        assert!(plan
            .iter()
            .all(|tile| tile.query_len <= 1024 && tile.key_len <= 1058));
        assert!(plan
            .iter()
            .all(|tile| tile.query_len * tile.key_len < 69_632usize.pow(2)));

        for len in [1, 17, 18, 35, 36, 1023, 1024, 1025, 2051] {
            let tiles = band_attention_plan(len, 5, 9, 1024).unwrap();
            assert_eq!(tiles.iter().map(|tile| tile.query_len).sum::<usize>(), len);
            assert_eq!(tiles.first().unwrap().query_start, 0);
            assert_eq!(
                tiles.last().unwrap().query_start + tiles.last().unwrap().query_len,
                len
            );
        }
        assert!(band_attention_plan(8, 1, 1, 0).is_err());
    }

    #[test]
    fn tiled_band_matches_dense_mask_for_ordinary_and_differential_attention() {
        let device = Device::Cpu;
        let batch = 2;
        let sequence = 37;
        let dim = 8;
        let dim_head = 4;
        let left = 3;
        let right = 5;
        let query_tile = 8;
        let values: Vec<f32> = (0..batch * sequence * dim)
            .map(|index| ((index as f32 * 0.013).sin() + (index as f32 * 0.007).cos()) * 0.25)
            .collect();
        let x = Tensor::from_vec(values, (batch, sequence, dim), &device).unwrap();
        let rope = RotaryEmbedding::new(dim_head / 2, &device).unwrap();
        let full_rope = rope.frequencies(sequence).unwrap();
        let dense_mask =
            sliding_window_additive_mask(sequence, sequence, left, right, DType::F32, &device)
                .unwrap();

        for differential in [false, true] {
            let map = VarMap::new();
            let attention = Attention::load(
                dim,
                dim_head,
                None,
                QkNorm::Dyt,
                1e-3,
                differential,
                false,
                false,
                VarBuilder::from_varmap(&map, DType::F32, &device).pp("attention"),
            )
            .unwrap();
            let dense = attention
                .forward(&x, None, Some(&full_rope), None, None, Some(&dense_mask))
                .unwrap();
            let plan = band_attention_plan(sequence, left, right, query_tile).unwrap();
            let mut parts = Vec::with_capacity(plan.len());
            for tile in plan {
                let query = x.narrow(1, tile.query_start, tile.query_len).unwrap();
                let keys = x.narrow(1, tile.key_start, tile.key_len).unwrap();
                let query_rope = rope
                    .frequencies_range(tile.query_start, tile.query_len)
                    .unwrap();
                let key_rope = rope
                    .frequencies_range(tile.key_start, tile.key_len)
                    .unwrap();
                parts.push(
                    attention
                        .forward_self_band_tile(
                            &query,
                            &keys,
                            tile.query_start,
                            tile.key_start,
                            left,
                            right,
                            &query_rope,
                            &key_rope,
                        )
                        .unwrap(),
                );
            }
            let refs: Vec<_> = parts.iter().collect();
            let tiled = Tensor::cat(&refs, 1).unwrap();
            let error = max_abs_diff(&dense, &tiled);
            assert!(
                error <= 2e-5,
                "differential={differential}: dense/band max abs {error}"
            );
        }
    }

    #[test]
    fn same_l_marks_exactly_indices_five_through_eleven() {
        let selected: Vec<_> = (0..12).filter(|&i| is_sinusoidal_block(i, 12, 8)).collect();
        assert_eq!(selected, vec![5, 6, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn rope_is_half_split_and_fp32() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 1, 4), &dev).unwrap();
        let f = Tensor::from_vec(
            vec![
                std::f32::consts::FRAC_PI_2,
                0.,
                std::f32::consts::FRAC_PI_2,
                0.,
            ],
            (1, 4),
            &dev,
        )
        .unwrap();
        let y = apply_rotary(&x, &f)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((y[0] + 3.).abs() < 1e-5);
        assert!((y[1] - 2.).abs() < 1e-5);
        assert!((y[2] - 1.).abs() < 1e-5);
        assert!((y[3] - 4.).abs() < 1e-5);
    }

    #[test]
    fn f16_loaded_rope_supports_ranged_band_positions() {
        let dev = Device::Cpu;
        let inv_freq = Tensor::from_vec(vec![1f32, 0.01], 2, &dev).unwrap();
        let tensors = std::collections::HashMap::from([("inv_freq".to_owned(), inv_freq)]);
        let rope =
            RotaryEmbedding::load(4, VarBuilder::from_tensors(tensors, DType::F16, &dev)).unwrap();
        let x = Tensor::from_vec(
            (0..2 * 11 * 4)
                .map(|index| (index as f32 * 0.013).sin())
                .collect::<Vec<_>>(),
            (2, 11, 4),
            &dev,
        )
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();

        for tile in band_attention_plan(11, 3, 5, 4).unwrap() {
            let frequencies = rope
                .frequencies_range(tile.key_start, tile.key_len)
                .unwrap();
            assert_eq!(frequencies.dtype(), DType::F32);
            let rotated = apply_rotary(
                &x.narrow(1, tile.key_start, tile.key_len)
                    .unwrap()
                    .unsqueeze(1)
                    .unwrap(),
                &frequencies,
            )
            .unwrap();
            assert_eq!(rotated.dtype(), DType::F16);
            assert_eq!(rotated.dims(), &[2, 1, tile.key_len, 4]);
        }
    }

    #[test]
    fn corrected_dyt_differential_and_gate_equations_are_exact() {
        let dev = Device::Cpu;
        let dyt = DynamicTanh::from_tensors(
            Tensor::from_vec(vec![4f32], 1, &dev).unwrap(),
            Tensor::from_vec(vec![1f32, 2.], 2, &dev).unwrap(),
            Tensor::from_vec(vec![0f32, 0.5], 2, &dev).unwrap(),
        );
        let x = Tensor::from_vec(vec![0.25f32, -0.25], (1, 1, 2), &dev).unwrap();
        let y = dyt
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((y[0] - 1f32.tanh()).abs() < 1e-6);
        assert!((y[1] - (-2f32 * 1f32.tanh() + 0.5)).abs() < 1e-6);

        let ordinary = Tensor::from_vec(vec![3f32, -2.], 2, &dev).unwrap();
        let differential = Tensor::from_vec(vec![1f32, 4.], 2, &dev).unwrap();
        assert_eq!(
            combine_differential(&ordinary, &differential)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![2.0, -6.0]
        );

        let gate = ada_ln_gate(&Tensor::zeros(1, DType::F32, &dev).unwrap())
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((gate - 0.731_058_6).abs() < 1e-6);
        assert!((gate - 0.5).abs() > 0.2);
    }

    #[test]
    fn varmap_initialization_honors_zero_branches_and_layer_scale_override() {
        let dev = Device::Cpu;
        let input = Tensor::ones((1, 3, 4), DType::F32, &dev).unwrap();

        let ff_zero_map = VarMap::new();
        let ff_zero = FeedForward::load(
            4,
            &FeedForwardConfig {
                mult: 2.0,
                zero_init_output: true,
                ..Default::default()
            },
            VarBuilder::from_varmap(&ff_zero_map, DType::F32, &dev).pp("ff"),
        )
        .unwrap();
        assert_eq!(
            ff_zero
                .forward(&input)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
            0.0
        );

        let ff_live_map = VarMap::new();
        let ff_live = FeedForward::load(
            4,
            &FeedForwardConfig {
                mult: 2.0,
                zero_init_output: false,
                ..Default::default()
            },
            VarBuilder::from_varmap(&ff_live_map, DType::F32, &dev).pp("ff"),
        )
        .unwrap();
        assert!(
            ff_live
                .forward(&input)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                > 0.0
        );

        let attention_map = VarMap::new();
        let attention = Attention::load(
            4,
            2,
            None,
            QkNorm::None,
            1e-6,
            false,
            true,
            false,
            VarBuilder::from_varmap(&attention_map, DType::F32, &dev).pp("attn"),
        )
        .unwrap();
        assert_eq!(
            attention
                .forward(&input, None, None, None, None, None)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
            0.0
        );

        let block_map = VarMap::new();
        let block = TransformerBlock::load(
            4,
            2,
            None,
            NormType::LayerNorm,
            &NormConfig::default(),
            QkNorm::None,
            1e-6,
            false,
            false,
            &FeedForwardConfig {
                mult: 2.0,
                ..Default::default()
            },
            true,
            false,
            None,
            false,
            VarBuilder::from_varmap(&block_map, DType::F32, &dev).pp("block"),
        )
        .unwrap();
        let block_output = block
            .forward(
                &input,
                None,
                None,
                None,
                None,
                None,
                TransformerBlockMasks::default(),
            )
            .unwrap();
        assert_eq!(
            block_output.to_vec3::<f32>().unwrap(),
            input.to_vec3::<f32>().unwrap()
        );

        let scaled_map = VarMap::new();
        let _scaled = TransformerBlock::load(
            4,
            2,
            None,
            NormType::LayerNorm,
            &NormConfig::default(),
            QkNorm::None,
            1e-6,
            false,
            false,
            &FeedForwardConfig {
                mult: 2.0,
                ..Default::default()
            },
            true,
            false,
            None,
            true,
            VarBuilder::from_varmap(&scaled_map, DType::F32, &dev).pp("scaled"),
        )
        .unwrap();
        let vars = scaled_map.data().lock().unwrap();
        for key in ["scaled.self_attn.to_out.weight", "scaled.ff.ff.2.weight"] {
            assert!(
                vars[key]
                    .as_tensor()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    > 0.0,
                "{key} must not be zero-init when LayerScale is enabled"
            );
        }
        for key in ["scaled.self_attn_scale.scale", "scaled.ff_scale.scale"] {
            let values = vars[key].as_tensor().to_vec1::<f32>().unwrap();
            assert!(values.iter().all(|value| (*value - 1e-5).abs() < 1e-9));
        }
        drop(vars);

        let learned_map = VarMap::new();
        let learned_vb = VarBuilder::from_varmap(&learned_map, DType::F32, &dev);
        let _memory = MemoryTokens::load(2, 4, learned_vb.pp("transformer")).unwrap();
        let _global_block = TransformerBlock::load(
            4,
            2,
            None,
            NormType::LayerNorm,
            &NormConfig::default(),
            QkNorm::None,
            1e-6,
            false,
            false,
            &FeedForwardConfig {
                mult: 2.0,
                ..Default::default()
            },
            true,
            true,
            None,
            false,
            learned_vb.pp("block"),
        )
        .unwrap();
        let vars = learned_map.data().lock().unwrap();
        for key in ["transformer.memory_tokens", "block.to_scale_shift_gate"] {
            assert!(
                vars[key]
                    .as_tensor()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    > 0.0,
                "{key} must use upstream's random initialization"
            );
        }
    }
}
