//! The frozen MOSS graph. The small geometry constructor is private to numerical tests.
use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::{self, Module};
use candle_gen::gen_core::{moss_image::MossImageConfig, CancelFlag};
use candle_gen::train::projection::FrozenProjectionPack;
use candle_gen::weights::Weights;
use candle_gen::{CandleError, Result};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub dim: usize,
    pub ffn: usize,
    pub heads: usize,
    pub layers: usize,
    pub text: usize,
    pub freq: usize,
    pub channels: usize,
}
impl From<&MossImageConfig> for Geometry {
    fn from(c: &MossImageConfig) -> Self {
        Self {
            dim: c.dim,
            ffn: c.ffn_dim,
            heads: c.num_heads,
            layers: c.num_layers,
            text: c.text_dim,
            freq: c.freq_dim,
            channels: c.latent_channels,
        }
    }
}

pub(crate) fn check_cancel(cancel: &CancelFlag) -> Result<()> {
    if cancel.is_cancelled() {
        Err(CandleError::Canceled)
    } else {
        Ok(())
    }
}

pub(super) fn ordinary_time_silu(x: &Tensor) -> Result<Tensor> {
    // Reuse the edit graph's author nn.SiLU boundary only for BF16.
    // Other dtype arithmetic remains unchanged.
    if x.dtype() == DType::BF16 {
        edit::silu(x)
    } else {
        Ok(candle_nn::ops::silu(x)?)
    }
}

struct LowRank {
    down: Tensor,
    up: Tensor,
    scale: f64,
}
struct Linear {
    base: candle_nn::Linear,
    low_rank: Option<LowRank>,
    residuals: Vec<crate::adapters::Residual>,
    residuals_frozen: bool,
    edit_adapter: bool,
    edit_fused_bias: bool,
}
impl Linear {
    fn load(
        w: &Weights,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
        adapter: Option<&Weights>,
        consumed: &mut BTreeSet<String>,
    ) -> Result<Self> {
        let weight_name = format!("{name}.weight");
        let weight = w.require(&weight_name)?;
        if weight.dims() != [output, input] {
            return Err(CandleError::Msg(format!("MOSS shape {weight_name}")));
        }
        consumed.insert(weight_name);
        let b = if bias {
            let key = format!("{name}.bias");
            let value = w.require(&key)?;
            if value.dims() != [output] {
                return Err(CandleError::Msg(format!("MOSS shape {key}")));
            }
            consumed.insert(key);
            Some(value)
        } else {
            None
        };
        let low_rank = if let Some(a) = adapter {
            let prefix = format!("diffusion_model.{name}");
            let down = a.require(&format!("{prefix}.lora_down.weight"))?;
            let up = a.require(&format!("{prefix}.lora_up.weight"))?;
            let alpha = a
                .require(&format!("{prefix}.alpha"))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let rank = down.dim(0)?;
            if down.dims() != [rank, input]
                || up.dims() != [output, rank]
                || alpha.len() != 1
                || !alpha[0].is_finite()
                || rank == 0
                || down.dtype() != DType::F32
                || up.dtype() != DType::F32
            {
                return Err(CandleError::Msg(format!(
                    "MOSS invalid low rank projection {prefix}"
                )));
            }
            Some(LowRank {
                down,
                up,
                scale: f64::from(alpha[0]) / rank as f64,
            })
        } else {
            None
        };
        Ok(Self {
            base: candle_nn::Linear::new(weight, b),
            low_rank,
            residuals: Vec::new(),
            residuals_frozen: false,
            edit_adapter: false,
            edit_fused_bias: false,
        })
    }
    fn prepare_frozen_group(layers: &mut [&mut Self]) -> Result<Option<FrozenProjectionPack>> {
        if layers.iter().any(|l| {
            l.low_rank.is_some()
                || l.edit_fused_bias
                || (l.dtype() == DType::BF16 && l.base.weight().device().is_cpu())
        }) {
            return Ok(None);
        }
        let mut facades = layers
            .iter()
            .map(|l| {
                let (out, input) = l.base.weight().dims2()?;
                Ok(candle_gen::quant::AdaptLinear::from_dense(
                    l.base.clone(),
                    input,
                    out,
                ))
            })
            .collect::<candle_gen::candle_core::Result<Vec<_>>>()?;
        let pack =
            FrozenProjectionPack::prepare_adapt(&mut facades.iter_mut().collect::<Vec<_>>())?;
        if pack.is_some() {
            for (layer, facade) in layers.iter_mut().zip(&facades) {
                layer.base = facade.dense_base().unwrap().clone();
            }
        }
        Ok(pack)
    }

    fn forward_frozen_group(
        pack: &FrozenProjectionPack,
        x: &Tensor,
        layers: &[&Self],
    ) -> Result<Vec<Tensor>> {
        let facades = layers
            .iter()
            .map(|l| {
                let (out, input) = l.base.weight().dims2()?;
                Ok(candle_gen::quant::AdaptLinear::from_dense(
                    l.base.clone(),
                    input,
                    out,
                ))
            })
            .collect::<candle_gen::candle_core::Result<Vec<_>>>()?;
        pack.forward_adapt(x, &facades.iter().collect::<Vec<_>>())?
            .into_iter()
            .zip(layers)
            .map(|(y, l)| l.apply_residuals(x, y))
            .collect()
    }

    fn dtype(&self) -> DType {
        self.base.weight().dtype()
    }
    fn base_forward(&self, x: &Tensor) -> Result<Tensor> {
        // Candle's CPU matmul has no BF16 kernel. This explicit FP32 accumulation over the
        // already-rounded BF16 operands, followed by BF16 rounding, is the CPU numerical oracle.
        // Editing also widens biased GPU projections until after bias addition: the frozen
        // source rounds its fused linear once, not once before and again after the bias.
        if self.dtype() == DType::BF16
            && (x.device().is_cpu() || (self.edit_fused_bias && self.base.bias().is_some()))
        {
            let linear = candle_nn::Linear::new(
                self.base.weight().to_dtype(DType::F32)?,
                self.base
                    .bias()
                    .map(|b| b.to_dtype(DType::F32))
                    .transpose()?,
            );
            Ok(linear
                .forward(&x.to_dtype(DType::F32)?)?
                .to_dtype(DType::BF16)?)
        } else {
            Ok(self.base.forward(x)?)
        }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.apply_residuals(x, self.forward_dmd(x)?)
    }
    fn apply_residuals(&self, x: &Tensor, mut output: Tensor) -> Result<Tensor> {
        for residual in &self.residuals {
            let frozen;
            let residual = if self.residuals_frozen {
                frozen = residual.detached();
                &frozen
            } else {
                residual
            };
            if let Some(value) = residual.forward(x)? {
                output = output.to_dtype(DType::F32)?.add(&value)?;
            }
        }
        Ok(output)
    }
    fn forward_dmd(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(a) = &self.low_rank {
            if self.edit_adapter {
                let base = self.base_forward(&x.to_dtype(self.dtype())?)?;
                let project = |input: &Tensor, weight: &Tensor| -> Result<Tensor> {
                    let dtype = weight.dtype();
                    let input = input.to_dtype(dtype)?;
                    if input.device().is_cpu() && dtype == DType::BF16 {
                        Ok(candle_nn::Linear::new(weight.to_dtype(DType::F32)?, None)
                            .forward(&input.to_dtype(DType::F32)?)?
                            .to_dtype(dtype)?)
                    } else {
                        Ok(candle_nn::Linear::new(weight.clone(), None).forward(&input)?)
                    }
                };
                let residual = project(&project(x, &a.down)?, &a.up)?;
                return Ok(base.add(&(residual * a.scale)?.to_dtype(base.dtype())?)?);
            }
            let base = self
                .base_forward(&x.to_dtype(self.dtype())?)?
                .to_dtype(DType::F32)?;
            let low =
                candle_nn::Linear::new(a.down.clone(), None).forward(&x.to_dtype(DType::F32)?)?;
            let low = candle_nn::Linear::new(a.up.clone(), None).forward(&low)?;
            Ok(base.add(&(low * a.scale)?)?)
        } else {
            self.base_forward(&x.to_dtype(self.dtype())?)
        }
    }
}

struct Norm(Tensor);
impl Norm {
    fn load(w: &Weights, name: &str, dim: usize, keys: &mut BTreeSet<String>) -> Result<Self> {
        let key = format!("{name}.weight");
        let t = w.require(&key)?;
        if t.dims() != [dim] {
            return Err(CandleError::Msg(format!("MOSS norm shape {key}")));
        }
        keys.insert(key);
        Ok(Self(t))
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normalized = candle_gen::train::norm::rms_normalize_f32(x, 1e-6)?;
        // PyTorch promotes the normalized activation when the norm weight is an FP32 island.
        let dtype = if x.dtype() == DType::F32 || self.0.dtype() == DType::F32 {
            DType::F32
        } else {
            x.dtype()
        };
        Ok(normalized
            .to_dtype(x.dtype())?
            .to_dtype(dtype)?
            .broadcast_mul(&self.0.to_dtype(dtype)?)?)
    }
}
pub(crate) fn situ(hidden: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let a = ((gate / 4.0)?.tanh()? * 4.0)?;
    let a = a.mul(&candle_nn::ops::sigmoid(gate)?)?;
    Ok(a.mul(&((hidden / 25.0)?.tanh()? * 25.0)?)?)
}

/// Source FP32 frequency construction, coordinates and sin/cos, then activation cast.
pub(crate) fn rope(
    height: usize,
    width: usize,
    text: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let bands = head_dim / 8;
    let frequencies: Vec<f32> = (0..bands)
        .map(|i| 1.0 / 10000_f32.powf(i as f32 / bands as f32))
        .collect();
    let mut values = Vec::with_capacity((height * width + text) * head_dim);
    for n in 0..height * width + text {
        let coordinate = if n < height * width {
            [
                0.0,
                text as f32,
                n.div_euclid(width) as f32 - (height - 1) as f32 / 2.0,
                (n % width) as f32 - (width - 1) as f32 / 2.0,
            ]
        } else {
            [0.0, (n - height * width) as f32, 0.0, 0.0]
        };
        for trig in 0..2 {
            for c in coordinate {
                for f in &frequencies {
                    let p = c * f;
                    values.push(if trig == 0 { p.sin() } else { p.cos() });
                }
            }
        }
    }
    Ok(Tensor::from_vec(values, (1, height * width + text, head_dim), device)?.to_dtype(dtype)?)
}
fn rotate(x: &Tensor, rope: &Tensor) -> Result<Tensor> {
    let half = x.dim(3)? / 2;
    let sin = rope.narrow(2, 0, half)?.unsqueeze(2)?;
    let cos = rope.narrow(2, half, half)?.unsqueeze(2)?;
    let sin = Tensor::cat(&[&sin, &sin], 3)?;
    let cos = Tensor::cat(&[&cos, &cos], 3)?;
    let paired = Tensor::cat(
        &[&x.narrow(3, half, half)?.neg()?, &x.narrow(3, 0, half)?],
        3,
    )?;
    // Source multiplication promotes BF16 rope when a widened DMD Q/K enters.
    let dtype = x.dtype();
    Ok(x.broadcast_mul(&cos.to_dtype(dtype)?)?
        .add(&paired.broadcast_mul(&sin.to_dtype(dtype)?)?)?)
}
fn attention_qk_operands(
    q: &Tensor,
    k: &Tensor,
    value_dtype: DType,
    r: &Tensor,
    heads: usize,
) -> Result<(Tensor, Tensor)> {
    let (b, s, d) = q.dims3()?;
    let width = d / heads;
    let q = rotate(&q.reshape((b, s, heads, width))?, r)?
        .transpose(1, 2)?
        .to_dtype(value_dtype)?;
    let k = rotate(&k.reshape((b, s, heads, width))?, r)?
        .transpose(1, 2)?
        .to_dtype(value_dtype)?;
    // CUDA batched GEMM needs packed heads. BF16 -> FP32 already packs these views,
    // but DMD's FP32 -> FP32 is a no-op and retains the interleaved head strides.
    // Pack K before its final matrix transpose, which cuBLAS supports directly.
    Ok((
        q.to_dtype(DType::F32)?.contiguous()?,
        k.to_dtype(DType::F32)?.contiguous()?.transpose(2, 3)?,
    ))
}
fn attention_value_operand(v: &Tensor, heads: usize) -> Result<Tensor> {
    let (b, s, d) = v.dims3()?;
    // The value GEMM has the same FP32 no-op-cast layout hazard as Q/K.
    Ok(v.reshape((b, s, heads, d / heads))?
        .transpose(1, 2)?
        .to_dtype(DType::F32)?
        .contiguous()?)
}
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, r: &Tensor, heads: usize) -> Result<Tensor> {
    let (b, s, d) = q.dims3()?;
    let output_dtype = v.dtype();
    // Explicit FP32 score and value accumulation, then round to the activation dtype.
    let (qh, kt) = attention_qk_operands(q, k, output_dtype, r, heads)?;
    let vh = attention_value_operand(v, heads)?;
    if let Some(out) = candle_gen::fused_attention::try_training_sdpa_bhsd(
        &qh,
        &kt.transpose(2, 3)?,
        &vh,
        1.0 / ((d / heads) as f64).sqrt(),
        None,
    )? {
        return Ok(out
            .to_dtype(output_dtype)?
            .transpose(1, 2)?
            .reshape((b, s, d))?);
    }
    let scores = (qh.matmul(&kt)? / ((d / heads) as f64).sqrt())?;
    // The optimized inference kernel uses apply_op1_no_bwd and severs every Q/K adapter
    // gradient. Recorded training/recomputation graphs must use differentiable FP32 softmax.
    let p = if scores.track_op() {
        candle_nn::ops::softmax(&scores, candle_gen::candle_core::D::Minus1)?
    } else {
        candle_nn::ops::softmax_last_dim(&scores)?
    };
    Ok(p.matmul(&vh)?
        .to_dtype(output_dtype)?
        .transpose(1, 2)?
        .reshape((b, s, d))?)
}
struct TextBlock {
    n1: Norm,
    n2: Norm,
    qn: Norm,
    kn: Norm,
    qkv: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    heads: usize,
}
impl TextBlock {
    fn load(w: &Weights, p: &str, g: Geometry, keys: &mut BTreeSet<String>) -> Result<Self> {
        Ok(Self {
            n1: Norm::load(w, &format!("{p}.norm1"), g.dim, keys)?,
            n2: Norm::load(w, &format!("{p}.norm2"), g.dim, keys)?,
            qn: Norm::load(w, &format!("{p}.q_norm"), g.dim / g.heads, keys)?,
            kn: Norm::load(w, &format!("{p}.k_norm"), g.dim / g.heads, keys)?,
            qkv: Linear::load(w, &format!("{p}.qkv"), g.dim, 3 * g.dim, true, None, keys)?,
            o: Linear::load(w, &format!("{p}.proj"), g.dim, g.dim, true, None, keys)?,
            gate: Linear::load(w, &format!("{p}.ffn_gate"), g.dim, g.ffn, true, None, keys)?,
            up: Linear::load(w, &format!("{p}.ffn_up"), g.dim, g.ffn, true, None, keys)?,
            down: Linear::load(w, &format!("{p}.ffn_down"), g.ffn, g.dim, true, None, keys)?,
            heads: g.heads,
        })
    }
    fn forward(&self, x: &Tensor, r: &Tensor) -> Result<Tensor> {
        self.forward_with_edit_precision(x, r, false)
    }
    fn forward_with_edit_precision(&self, x: &Tensor, r: &Tensor, edit: bool) -> Result<Tensor> {
        let qkv = self.qkv.forward(&self.n1.forward(x)?)?;
        let (b, s, three) = qkv.dims3()?;
        let d = three / 3;
        let q = self
            .qn
            .forward(
                &qkv.narrow(2, 0, d)?
                    .reshape((b, s, self.heads, d / self.heads))?,
            )?
            .reshape((b, s, d))?;
        let k = self
            .kn
            .forward(
                &qkv.narrow(2, d, d)?
                    .reshape((b, s, self.heads, d / self.heads))?,
            )?
            .reshape((b, s, d))?;
        let attend = if edit {
            edit::text_attention
        } else {
            attention
        };
        let x =
            x.add(
                &self
                    .o
                    .forward(&attend(&q, &k, &qkv.narrow(2, 2 * d, d)?, r, self.heads)?)?,
            )?;
        let y = self.n2.forward(&x)?;
        let activation = if edit { edit::edit_situ } else { situ };
        Ok(x.add(
            &self
                .down
                .forward(&activation(&self.up.forward(&y)?, &self.gate.forward(&y)?)?)?,
        )?)
    }
}
struct Block {
    projection_pack: Option<FrozenProjectionPack>,
    n1: Norm,
    n2: Norm,
    qn: Norm,
    kn: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    up_gate: Linear,
    down: Linear,
    heads: usize,
}
impl Block {
    fn load(
        w: &Weights,
        p: &str,
        g: Geometry,
        a: Option<&Weights>,
        keys: &mut BTreeSet<String>,
    ) -> Result<Self> {
        Ok(Self {
            projection_pack: None,
            n1: Norm::load(w, &format!("{p}.norm1"), g.dim, keys)?,
            n2: Norm::load(w, &format!("{p}.norm2"), g.dim, keys)?,
            qn: Norm::load(w, &format!("{p}.q_norm"), g.dim / g.heads, keys)?,
            kn: Norm::load(w, &format!("{p}.k_norm"), g.dim / g.heads, keys)?,
            q: Linear::load(w, &format!("{p}.proj.q"), g.dim, g.dim, false, a, keys)?,
            k: Linear::load(w, &format!("{p}.proj.k"), g.dim, g.dim, false, a, keys)?,
            v: Linear::load(w, &format!("{p}.proj.v"), g.dim, g.dim, false, a, keys)?,
            o: Linear::load(w, &format!("{p}.proj.o"), g.dim, g.dim, false, a, keys)?,
            up_gate: Linear::load(
                w,
                &format!("{p}.proj.ffn_up_gate"),
                g.dim,
                2 * g.ffn,
                false,
                a,
                keys,
            )?,
            down: Linear::load(
                w,
                &format!("{p}.proj.ffn_down"),
                g.ffn,
                g.dim,
                false,
                a,
                keys,
            )?,
            heads: g.heads,
        })
    }
    fn forward(&self, x: &Tensor, r: &Tensor, m: &Tensor) -> Result<Tensor> {
        let part = |i| m.i((.., i, ..))?.unsqueeze(1);
        let dtype = if x.dtype() == DType::F32 || m.dtype() == DType::F32 {
            DType::F32
        } else {
            x.dtype()
        };
        let y = self
            .n1
            .forward(x)?
            .to_dtype(dtype)?
            .broadcast_mul(&(part(1)?.to_dtype(dtype)? + 1.0)?)?
            .broadcast_add(&part(0)?.to_dtype(dtype)?)?
            .to_dtype(self.q.dtype())?;
        let (b, s, d) = y.dims3()?;
        let (q, k, v) = if let Some(pack) = &self.projection_pack {
            candle_gen::train::profile::cohort_reuse("moss:frozen-projection-pack");
            let mut ys =
                Linear::forward_frozen_group(pack, &y, &[&self.q, &self.k, &self.v])?.into_iter();
            (ys.next().unwrap(), ys.next().unwrap(), ys.next().unwrap())
        } else {
            (
                self.q.forward(&y)?,
                self.k.forward(&y)?,
                self.v.forward(&y)?,
            )
        };
        let q = self
            .qn
            .forward(&q.reshape((b, s, self.heads, d / self.heads))?)?
            .reshape((b, s, d))?;
        let k = self
            .kn
            .forward(&k.reshape((b, s, self.heads, d / self.heads))?)?
            .reshape((b, s, d))?;
        let projected = self.o.forward(&attention(&q, &k, &v, r, self.heads)?)?;
        // Selective custom targets can widen Q without widening O (or vice versa). Promote
        // the actual multiplication/addition operands, never infer O's dtype from Q.
        let promoted = |a: DType, b: DType| {
            if a == DType::F32 || b == DType::F32 {
                DType::F32
            } else {
                a
            }
        };
        let dtype = promoted(projected.dtype(), m.dtype());
        let residual = projected
            .to_dtype(dtype)?
            .broadcast_mul(&part(2)?.to_dtype(dtype)?)?;
        let dtype = promoted(x.dtype(), residual.dtype());
        let x = x.to_dtype(dtype)?.add(&residual.to_dtype(dtype)?)?;
        let normalized = self.n2.forward(&x)?;
        let dtype = promoted(normalized.dtype(), m.dtype());
        let y = normalized
            .to_dtype(dtype)?
            .broadcast_mul(&(part(4)?.to_dtype(dtype)? + 1.0)?)?
            .broadcast_add(&part(3)?.to_dtype(dtype)?)?
            .to_dtype(self.q.dtype())?;
        let hg = self.up_gate.forward(&y)?;
        let f = hg.dim(2)? / 2;
        let residual = self
            .down
            .forward(&situ(&hg.narrow(2, 0, f)?, &hg.narrow(2, f, f)?)?)?;
        let dtype = promoted(residual.dtype(), m.dtype());
        let residual = residual
            .to_dtype(dtype)?
            .broadcast_mul(&part(5)?.to_dtype(dtype)?)?;
        let dtype = promoted(x.dtype(), residual.dtype());
        Ok(x.to_dtype(dtype)?.add(&residual.to_dtype(dtype)?)?)
    }
}

pub(crate) struct Transformer {
    resident_bytes: u64,
    training_context: HashMap<candle_gen::candle_core::TensorId, Tensor>,
    training_rope: Vec<((usize, usize, usize, DType), Tensor)>,
    g: Geometry,
    video: Linear,
    text: Linear,
    text_blocks: Vec<TextBlock>,
    time1: Linear,
    time2: Linear,
    shared: Linear,
    head: Linear,
    layers: Vec<Block>,
    norm: Norm,
    final_linear: Linear,
}
impl Transformer {
    pub(crate) fn training_preparation_bytes(&self, inputs: &[(&Tensor, usize, usize)]) -> u64 {
        let mut ids = std::collections::HashSet::new();
        let mut geometry = std::collections::HashSet::new();
        let mut bytes = 0usize;
        for (context, h, w) in inputs {
            let tokens = context.dims()[1];
            if ids.insert(context.id()) {
                bytes += tokens * self.g.dim * 4;
            }
            if geometry.insert((*h, *w, tokens)) {
                bytes += (h * w + tokens)
                    * (self.g.dim / self.g.heads)
                    * self.video.dtype().size_in_bytes();
            }
        }
        bytes as u64
    }

    /// Called once after adapter installation/resume, only for existing cached captions.
    /// Uncached text remains uncached; previews and misses never extend run residency.
    pub(crate) fn prepare_training(&mut self, inputs: &[(&Tensor, usize, usize)]) -> Result<()> {
        for block in &mut self.layers {
            if block.projection_pack.is_none() {
                block.projection_pack =
                    Linear::prepare_frozen_group(&mut [&mut block.q, &mut block.k, &mut block.v])?;
            }
        }
        self.training_context.clear();
        self.training_rope.clear();
        for (context, h, w) in inputs {
            let tokens = context.dim(1)?;
            let key = (*h, *w, tokens, self.video.dtype());
            if !self.training_rope.iter().any(|(k, _)| *k == key) {
                self.training_rope.push((
                    key,
                    rope(
                        *h,
                        *w,
                        tokens,
                        self.g.dim / self.g.heads,
                        self.video.dtype(),
                        self.device(),
                    )?,
                ));
            }
            if !self.training_context.contains_key(&context.id()) {
                let r = &self
                    .training_rope
                    .iter()
                    .find(|(k, _)| *k == key)
                    .unwrap()
                    .1;
                let mut text = self.project_context(context)?;
                for block in &self.text_blocks {
                    text = block.forward(&text, &r.narrow(1, h * w, tokens)?)?;
                }
                if text.track_op() {
                    return Err(CandleError::Msg(
                        "MOSS cannot cache trainable text conditioning".into(),
                    ));
                }
                self.training_context.insert(context.id(), text);
            }
        }
        Ok(())
    }

    pub(crate) fn channels(&self) -> usize {
        self.g.channels
    }
    pub(crate) fn device(&self) -> &Device {
        self.video.base.weight().device()
    }
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.resident_bytes
            + self
                .training_context
                .values()
                .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
                .sum::<u64>()
            + self
                .training_rope
                .iter()
                .map(|(_, t)| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
                .sum::<u64>()
            + self
                .layers
                .iter()
                .map(|b| {
                    [&b.q, &b.k, &b.v, &b.o, &b.up_gate, &b.down]
                        .iter()
                        .flat_map(|l| &l.residuals)
                        .map(crate::adapters::Residual::bytes)
                        .sum::<u64>()
                })
                .sum::<u64>()
    }
    pub(crate) fn load(w: &Weights, g: Geometry, adapter: Option<&Weights>) -> Result<Self> {
        let mut keys = BTreeSet::new();
        let mut result = Self {
            resident_bytes: 0,
            training_context: HashMap::new(),
            training_rope: Vec::new(),
            g,
            video: Linear::load(
                w,
                "video_embedder",
                g.channels,
                g.dim,
                true,
                None,
                &mut keys,
            )?,
            text: Linear::load(
                w,
                "text_embedder.proj",
                g.text,
                g.dim,
                true,
                None,
                &mut keys,
            )?,
            text_blocks: (0..2)
                .map(|i| TextBlock::load(w, &format!("text_embedder.blocks.{i}"), g, &mut keys))
                .collect::<Result<_>>()?,
            time1: Linear::load(w, "time_embedding.0", g.freq, g.dim, true, None, &mut keys)?,
            time2: Linear::load(w, "time_embedding.2", g.dim, g.dim, true, None, &mut keys)?,
            shared: Linear::load(
                w,
                "shared_block_modulation",
                g.dim,
                6 * g.dim,
                true,
                None,
                &mut keys,
            )?,
            head: Linear::load(
                w,
                "video_head_modulation",
                g.dim,
                2 * g.dim,
                true,
                None,
                &mut keys,
            )?,
            layers: (0..g.layers)
                .map(|i| Block::load(w, &format!("layers.{i}"), g, adapter, &mut keys))
                .collect::<Result<_>>()?,
            norm: Norm::load(w, "final_norm_video", g.dim, &mut keys)?,
            final_linear: Linear::load(
                w,
                "final_linear_video",
                g.dim,
                g.channels,
                false,
                None,
                &mut keys,
            )?,
        };
        for key in &keys {
            let t = w.require(key)?;
            result.resident_bytes += (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        }
        if let Some(adapter) = adapter {
            for key in adapter.keys().filter(|k| !k.ends_with(".alpha")) {
                let t = adapter.require(key)?;
                result.resident_bytes += (t.elem_count() * t.dtype().size_in_bytes()) as u64;
            }
        }
        // Derived RoPE bands are checkpoint-compatible but are reconstructed in source FP32.
        keys.insert("rope.bands".into());
        let extra: Vec<_> = w.keys().filter(|k| !keys.contains(*k)).collect();
        if !extra.is_empty() {
            return Err(CandleError::Msg(format!(
                "MOSS unconsumed transformer tensors {extra:?}"
            )));
        }
        Ok(result)
    }
    #[cfg(test)]
    pub(crate) fn block_stream_dtype(&self) -> Result<DType> {
        let x = Tensor::ones((1, 2, self.g.dim), self.video.dtype(), self.device())?;
        let modulation = Tensor::ones((1, 6, self.g.dim), self.video.dtype(), self.device())?;
        let rotations = rope(
            1,
            1,
            1,
            self.g.dim / self.g.heads,
            self.video.dtype(),
            self.device(),
        )?;
        Ok(self.layers[0].forward(&x, &rotations, &modulation)?.dtype())
    }
    #[cfg(test)]
    pub(crate) fn adapter_scales(&self) -> Vec<f64> {
        self.layers
            .iter()
            .flat_map(|b| [&b.q, &b.k, &b.v, &b.o, &b.up_gate, &b.down])
            .flat_map(|l| &l.residuals)
            .map(crate::adapters::Residual::scale)
            .collect()
    }
    pub(crate) fn projection_shapes(&self) -> std::collections::BTreeMap<String, (usize, usize)> {
        let mut shapes = std::collections::BTreeMap::new();
        for (i, b) in self.layers.iter().enumerate() {
            for (name, l) in [
                ("q", &b.q),
                ("k", &b.k),
                ("v", &b.v),
                ("o", &b.o),
                ("ffn_up_gate", &b.up_gate),
                ("ffn_down", &b.down),
            ] {
                shapes.insert(
                    format!("layers.{i}.proj.{name}"),
                    (l.base.weight().dims()[0], l.base.weight().dims()[1]),
                );
            }
        }
        shapes
    }
    pub(crate) fn push_residual(
        &mut self,
        path: &str,
        residual: crate::adapters::Residual,
    ) -> Result<()> {
        for (i, b) in self.layers.iter_mut().enumerate() {
            for (name, l) in [
                ("q", &mut b.q),
                ("k", &mut b.k),
                ("v", &mut b.v),
                ("o", &mut b.o),
                ("ffn_up_gate", &mut b.up_gate),
                ("ffn_down", &mut b.down),
            ] {
                if path == format!("layers.{i}.proj.{name}") {
                    l.residuals.push(residual);
                    return Ok(());
                }
            }
        }
        Err(CandleError::Msg(format!(
            "unknown MOSS transformer projection {path}"
        )))
    }
    pub(crate) fn validate_residuals(&self) -> Result<()> {
        for block in &self.layers {
            for linear in [
                &block.q,
                &block.k,
                &block.v,
                &block.o,
                &block.up_gate,
                &block.down,
            ] {
                for residual in &linear.residuals {
                    residual.validate()?;
                }
            }
        }
        Ok(())
    }
    pub(crate) fn freeze_residuals(&mut self, freeze: bool) {
        for block in &mut self.layers {
            for linear in [
                &mut block.q,
                &mut block.k,
                &mut block.v,
                &mut block.o,
                &mut block.up_gate,
                &mut block.down,
            ] {
                linear.residuals_frozen = freeze;
            }
        }
    }
    pub(crate) fn clear_residuals(&mut self) {
        for b in &mut self.layers {
            for l in [
                &mut b.q,
                &mut b.k,
                &mut b.v,
                &mut b.o,
                &mut b.up_gate,
                &mut b.down,
            ] {
                l.residuals.clear();
            }
        }
    }
    /// Reuse the shared seeded factor initializer and lossless serializer without using its
    /// activation-dtype residual. The short-lived host proxies share frozen base handles only;
    /// the executable MOSS stack below retains raw FP32 master factors from the resulting set.
    pub(crate) fn training_adapter(
        &mut self,
        c: &candle_gen::gen_core::TrainingConfig,
    ) -> Result<candle_gen::train::lora::LoraSet> {
        use candle_gen::train::lora::{
            build_adapt_lokr_targets_peft, build_adapt_lora_targets, AdaptLoraHost,
        };
        struct InitHost(Vec<(String, candle_gen::quant::AdaptLinear)>);
        impl AdaptLoraHost for InitHost {
            fn visit_adapt_lora_mut(
                &mut self,
                f: &mut dyn FnMut(&str, &mut candle_gen::quant::AdaptLinear) -> Result<()>,
            ) -> Result<()> {
                for (name, l) in &mut self.0 {
                    f(name, l)?;
                }
                Ok(())
            }
        }
        let mut entries = Vec::new();
        for (i, b) in self.layers.iter().enumerate() {
            for (name, l) in [
                ("q", &b.q),
                ("k", &b.k),
                ("v", &b.v),
                ("o", &b.o),
                ("ffn_up_gate", &b.up_gate),
                ("ffn_down", &b.down),
            ] {
                entries.push((
                    format!("layers.{i}.proj.{name}"),
                    candle_gen::quant::AdaptLinear::from_dense(
                        l.base.clone(),
                        l.base.weight().dims()[1],
                        l.base.weight().dims()[0],
                    ),
                ));
            }
        }
        let paths = candle_gen::gen_core::moss_image_training::target_paths(
            self.g.layers,
            &c.lora_target_modules,
        )?;
        let mut host = InitHost(entries);
        let set = match c.network_type {
            candle_gen::gen_core::NetworkType::Lora => {
                build_adapt_lora_targets(&mut host, &paths, c.rank, c.alpha, c.seed, self.device())?
            }
            candle_gen::gen_core::NetworkType::Lokr => build_adapt_lokr_targets_peft(
                &mut host,
                &paths,
                c.rank,
                c.alpha,
                c.decompose_factor,
                c.seed,
                self.device(),
            )?,
        };
        let lokr = set.kind == candle_gen::train::lora::AdapterKind::Lokr;
        let mut groups =
            std::collections::BTreeMap::<String, std::collections::BTreeMap<String, Tensor>>::new();
        let suffixes: &[&str] = if lokr {
            &candle_gen::gen_core::weightsmeta::LOKR_SUFFIXES
        } else {
            &[".lora_A.weight", ".lora_B.weight"]
        };
        for (name, var) in set.named_vars() {
            let (path, suffix) =
                candle_gen::gen_core::weightsmeta::split_factor_key(&name, suffixes)
                    .ok_or_else(|| CandleError::Msg("invalid MOSS factor key".into()))?;
            groups
                .entry(path.into())
                .or_default()
                .insert(suffix.into(), var.as_tensor().clone());
        }
        self.clear_residuals();
        let shapes = self.projection_shapes();
        // Portable MLX/import scaling divides in FP32, including non-binary alpha/rank.
        // Widen only the rounded scalar for Candle's affine API, not the division itself.
        let scale = f64::from(set.alpha / set.rank as f32);
        for (path, group) in groups {
            let (out, input) = shapes[&path];
            self.push_residual(
                &path,
                crate::adapters::Residual::from_group(&group, out, input, lokr, scale)?,
            )?;
        }
        Ok(set)
    }
    /// Full production training graph. Recomputed boundary leaves preserve the derivatives of
    /// frozen backbone operations to earlier adapter factors, not just the last adapted block.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn loss_and_grads(
        &self,
        x: &Tensor,
        time: f32,
        context: &Tensor,
        target: &Tensor,
        mask: Option<&Tensor>,
        mae: bool,
        checkpoint: bool,
        vars: &[candle_gen::candle_core::Var],
        cancel: &CancelFlag,
    ) -> Result<(f32, candle_gen::candle_core::backprop::GradStore)> {
        use candle_gen::train::gradient_checkpoint::{checkpointed_backward, Segment};
        check_cancel(cancel)?;
        let (batch, c, h, w) = x.dims4()?;
        let (cb, tokens, td) = context.dims3()?;
        if batch != 1
            || cb != 1
            || tokens == 0
            || td != self.g.text
            || c != self.g.channels
            || target.dims() != x.dims()
        {
            return Err(CandleError::Msg(
                "MOSS training latent/context/target geometry".into(),
            ));
        }
        let half = self.g.freq / 2;
        let features = if self.g.freq == 256 {
            candle_gen::gen_core::moss_image::moss_time_features(time)?.to_vec()
        } else {
            let phase: Vec<_> = (0..half)
                .map(|i| f64::from(time) * 10000_f64.powf(-(i as f64) / half as f64))
                .collect();
            phase
                .iter()
                .map(|p| p.cos() as f32)
                .chain(phase.iter().map(|p| p.sin() as f32))
                .collect()
        };
        let t = Tensor::from_vec(features, (1, self.g.freq), x.device())?
            .to_dtype(self.time1.dtype())?;
        let t = self
            .time2
            .forward(&ordinary_time_silu(&self.time1.forward(&t)?)?)?
            .to_dtype(DType::F32)?;
        let m = self
            .shared
            .forward(&t.to_dtype(self.shared.dtype())?)?
            .reshape((1, 6, self.g.dim))?;
        let head = self
            .head
            .forward(&t.to_dtype(self.head.dtype())?)?
            .reshape((1, 2, self.g.dim))?;
        let video = x
            .reshape((1, c, h * w))?
            .transpose(1, 2)?
            .to_dtype(self.video.dtype())?
            .contiguous()?;
        let video = self.video.forward(&video)?;
        let r = match self
            .training_rope
            .iter()
            .find(|(key, _)| *key == (h, w, tokens, video.dtype()))
        {
            Some((_, r)) => {
                candle_gen::train::profile::cohort_reuse("moss:geometry-storage-hit");
                r.clone()
            }
            None => rope(
                h,
                w,
                tokens,
                self.g.dim / self.g.heads,
                video.dtype(),
                x.device(),
            )?,
        };
        let text = if let Some(text) = self.training_context.get(&context.id()) {
            candle_gen::train::profile::cohort_reuse("moss:frozen-caption-storage-hit");
            text.clone()
        } else {
            let mut text = self.project_context(context)?;
            for block in &self.text_blocks {
                text = block.forward(&text, &r.narrow(1, h * w, tokens)?)?;
            }
            text
        };
        let y = Tensor::cat(&[video, text], 1)?.detach();
        let finish = |y: &Tensor| -> Result<Tensor> {
            let y = self
                .norm
                .forward(&y.narrow(1, 0, h * w)?.to_dtype(self.norm.0.dtype())?)?;
            let dtype = if y.dtype() == DType::F32 || head.dtype() == DType::F32 {
                DType::F32
            } else {
                y.dtype()
            };
            let y = y
                .to_dtype(dtype)?
                .broadcast_mul(&(head.i((.., 1, ..))?.unsqueeze(1)?.to_dtype(dtype)? + 1.)?)?
                .broadcast_add(&head.i((.., 0, ..))?.unsqueeze(1)?.to_dtype(dtype)?)?;
            let pred = __model_proof_output!(self
                .final_linear
                .forward(&y.to_dtype(self.final_linear.dtype())?))?
            .transpose(1, 2)?
            .reshape((1, c, h, w))?;
            Ok(candle_gen::train::flow_match::weighted_velocity_loss(
                &pred, target, mask, mae,
            )?)
        };
        let result = if checkpoint {
            let mut segments: Vec<Segment<'_>> = Vec::new();
            for block in &self.layers {
                let r = &r;
                let m = &m;
                segments.push(Box::new(move |state| {
                    check_cancel(cancel)
                        .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
                    Ok(vec![block.forward(&state[0], r, m).map_err(|e| {
                        candle_gen::candle_core::Error::Msg(e.to_string())
                    })?])
                }));
            }
            segments.push(Box::new(|state| {
                Ok(vec![finish(&state[0]).map_err(|e| {
                    candle_gen::candle_core::Error::Msg(e.to_string())
                })?])
            }));
            checkpointed_backward(&segments, &[y], vars)?
        } else {
            let mut y = y;
            for block in &self.layers {
                check_cancel(cancel)?;
                y = block.forward(&y, &r, &m)?;
            }
            let loss = finish(&y)?;
            (loss.to_scalar::<f32>()?, loss.backward()?)
        };
        if !result.0.is_finite() {
            return Err(CandleError::Msg("MOSS nonfinite loss".into()));
        }
        for var in vars {
            if let Some(g) = result.1.get(var.as_tensor()) {
                crate::adapters::finite(g, "training gradient")?;
            }
        }
        check_cancel(cancel)?;
        Ok(result)
    }
    pub(crate) fn project_context(&self, context: &Tensor) -> Result<Tensor> {
        self.text.forward(&context.to_dtype(self.text.dtype())?)
    }

    pub(crate) fn forward(
        &self,
        x: &Tensor,
        time: f32,
        context: &Tensor,
        mask: &[u8],
        skip: &[u32],
        cancel: &CancelFlag,
    ) -> Result<Tensor> {
        check_cancel(cancel)?;
        let (batch, c, h, w) = x.dims4()?;
        let (cb, s, td) = context.dims3()?;
        if c != self.g.channels
            || cb != batch
            || td != self.g.text
            || mask.len() != batch * s
            || skip.iter().any(|i| *i as usize >= self.g.layers)
        {
            return Err(CandleError::Msg("MOSS forward geometry/mask/skip".into()));
        }
        let features = if self.g.freq == 256 {
            candle_gen::gen_core::moss_image::moss_time_features(time)?.to_vec()
        } else {
            let half = self.g.freq / 2;
            let phase: Vec<_> = (0..half)
                .map(|i| f64::from(time) * 10000_f64.powf(-(i as f64) / half as f64))
                .collect();
            phase
                .iter()
                .map(|p| p.cos() as f32)
                .chain(phase.iter().map(|p| p.sin() as f32))
                .collect()
        };
        let t = Tensor::from_vec(features, (1, self.g.freq), x.device())?
            .to_dtype(self.time1.dtype())?;
        let t = self
            .time2
            .forward(&ordinary_time_silu(&self.time1.forward(&t)?)?)?
            .to_dtype(DType::F32)?;
        let m = self
            .shared
            .forward(&t.to_dtype(self.shared.dtype())?)?
            .reshape((1, 6, self.g.dim))?;
        let head = self
            .head
            .forward(&t.to_dtype(self.head.dtype())?)?
            .reshape((1, 2, self.g.dim))?;
        let mut result = Vec::with_capacity(batch);
        for b in 0..batch {
            check_cancel(cancel)?;
            let indices: Vec<u32> = mask[b * s..(b + 1) * s]
                .iter()
                .enumerate()
                .filter_map(|(i, v)| (*v == 1).then_some(i as u32))
                .collect();
            if indices.is_empty() || mask[b * s..(b + 1) * s].iter().any(|v| *v > 1) {
                return Err(CandleError::Msg("MOSS empty/nonbinary mask".into()));
            }
            let ctx = context.narrow(0, b, 1)?.index_select(
                &Tensor::from_vec(indices.clone(), indices.len(), x.device())?,
                1,
            )?;
            let video = x
                .narrow(0, b, 1)?
                .reshape((1, c, h * w))?
                .transpose(1, 2)?
                .to_dtype(self.video.dtype())?
                .contiguous()?;
            let video = self.video.forward(&video)?;
            let r = rope(
                h,
                w,
                indices.len(),
                self.g.dim / self.g.heads,
                video.dtype(),
                x.device(),
            )?;
            let mut text = self.project_context(&ctx)?;
            for block in &self.text_blocks {
                check_cancel(cancel)?;
                text = block.forward(&text, &r.narrow(1, h * w, indices.len())?)?;
                x.device().synchronize()?;
                check_cancel(cancel)?;
            }
            let mut y = Tensor::cat(&[video, text], 1)?;
            for (i, block) in self.layers.iter().enumerate() {
                check_cancel(cancel)?;
                if !skip.contains(&(i as u32)) {
                    y = block.forward(&y, &r, &m)?;
                    x.device().synchronize()?;
                    check_cancel(cancel)?;
                }
            }
            let y = self
                .norm
                .forward(&y.narrow(1, 0, h * w)?.to_dtype(self.norm.0.dtype())?)?;
            let dtype = if y.dtype() == DType::F32 || head.dtype() == DType::F32 {
                DType::F32
            } else {
                y.dtype()
            };
            let y = y
                .to_dtype(dtype)?
                .broadcast_mul(&(head.i((.., 1, ..))?.unsqueeze(1)?.to_dtype(dtype)? + 1.0)?)?
                .broadcast_add(&head.i((.., 0, ..))?.unsqueeze(1)?.to_dtype(dtype)?)?;
            result.push(
                self.final_linear
                    .forward(&y.to_dtype(self.final_linear.dtype())?)?
                    .transpose(1, 2)?
                    .reshape((1, c, h, w))?,
            );
        }
        Ok(Tensor::cat(&result, 0)?)
    }
}

/// Promote only source-mandated FP32 islands, then replace all six published tensors.
pub(crate) fn install_replacements(base: &Weights, replacements: &Weights) -> Result<Weights> {
    let mut map = HashMap::new();
    for key in base.keys() {
        let mut t = base.require(key)?;
        if [
            "time_embedding.",
            "shared_block_modulation.",
            "video_head_modulation.",
        ]
        .iter()
        .any(|p| key.starts_with(p))
        {
            t = t.to_dtype(DType::F32)?;
        }
        if replacements.contains(key) {
            t = replacements.require(key)?;
        }
        map.insert(key.clone(), t);
    }
    Ok(Weights::from_map(map))
}

#[cfg(test)]
pub(crate) fn test_projection(
    w: &Weights,
    name: &str,
    dim: usize,
    adapter: Option<&Weights>,
    input: &Tensor,
) -> Result<Tensor> {
    Linear::load(w, name, dim, dim, false, adapter, &mut BTreeSet::new())?.forward(input)
}

#[cfg(test)]
pub(crate) fn test_norm(x: &Tensor, weight: Tensor) -> Result<Tensor> {
    Norm(weight).forward(x)
}

#[cfg(test)]
mod attention_layout_tests {
    use super::*;

    const SEQUENCE: usize = 5;
    const HEADS: usize = 3;
    const WIDTH: usize = 8;

    fn inputs(
        device: &Device,
        batch: usize,
        qk_dtype: DType,
        value_dtype: DType,
        rope_dtype: DType,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let input = |offset: usize, dtype| {
            let values: Vec<f32> = (0..batch * SEQUENCE * HEADS * WIDTH)
                .map(|i| (((i * 7 + offset) % 29) as f32 - 14.) / 32.)
                .collect();
            Tensor::from_vec(values, (batch, SEQUENCE, HEADS * WIDTH), device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
        };
        (
            input(0, qk_dtype),
            input(3, qk_dtype),
            input(11, value_dtype),
            rope(1, 2, 3, WIDTH, rope_dtype, device).unwrap(),
        )
    }

    fn policies() -> [(DType, DType, DType); 4] {
        [
            (DType::BF16, DType::BF16, DType::BF16),
            (DType::F32, DType::BF16, DType::BF16),
            (DType::F32, DType::F32, DType::BF16),
            (DType::F32, DType::F32, DType::F32),
        ]
    }

    #[test]
    fn production_attention_operands_pack_each_head_for_both_matmuls() {
        for batch in [1, 2] {
            for (qk_dtype, value_dtype, rope_dtype) in policies() {
                let (q, k, v, r) = inputs(&Device::Cpu, batch, qk_dtype, value_dtype, rope_dtype);
                let (q, kt) = attention_qk_operands(&q, &k, v.dtype(), &r, HEADS).unwrap();
                let v = attention_value_operand(&v, HEADS).unwrap();
                // These are the actual operands consumed by attention, not a CPU numerical
                // proxy: CPU matmul accepts the interleaved strides that CUDA rejects.
                let packed = [HEADS * SEQUENCE * WIDTH, SEQUENCE * WIDTH, WIDTH, 1];
                assert_eq!(q.stride(), packed, "Q must pack its head matrices");
                assert_eq!(v.stride(), packed, "V must pack its head matrices");
                assert_eq!(
                    kt.stride(),
                    [HEADS * SEQUENCE * WIDTH, SEQUENCE * WIDTH, 1, WIDTH],
                    "K must pack its head matrices before its cuBLAS-supported transpose"
                );
                assert_eq!(q.dims(), [batch, HEADS, SEQUENCE, WIDTH]);
                assert_eq!(kt.dims(), [batch, HEADS, WIDTH, SEQUENCE]);
                assert_eq!(v.dims(), q.dims());
                for operand in [&q, &kt, &v] {
                    assert_eq!(operand.dtype(), DType::F32);
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an explicitly admitted CUDA device; no model weights"]
    fn production_attention_cuda_accepts_ordinary_and_dmd_layouts() {
        let device = Device::new_cuda(0).expect("admitted logical CUDA device 0");
        for batch in [1, 2] {
            for (qk_dtype, value_dtype, rope_dtype) in policies() {
                let (q, k, v, r) = inputs(&device, batch, qk_dtype, value_dtype, rope_dtype);
                let actual = attention(&q, &k, &v, &r, HEADS).unwrap();
                let (q, k, v, r) = inputs(&Device::Cpu, batch, qk_dtype, value_dtype, rope_dtype);
                let expected = attention(&q, &k, &v, &r, HEADS).unwrap();
                assert_eq!(actual.dims(), [batch, SEQUENCE, HEADS * WIDTH]);
                assert_eq!(actual.dtype(), value_dtype);
                let flatten = |x: Tensor| {
                    x.to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                };
                let actual = flatten(actual);
                let expected = flatten(expected);
                assert_eq!(actual.len(), expected.len());
                let tolerance = if value_dtype == DType::BF16 {
                    0.005
                } else {
                    0.0001
                };
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert!(actual.is_finite() && expected.is_finite());
                    assert!((actual - expected).abs() <= tolerance);
                }
            }
        }
    }
}

mod edit;
pub(crate) use edit::EditConditioning;
