//! UNet/ControlNet pricing hierarchy. Every retained parameter and activation is metadata;
//! constructors mirror the actual lookup/layout/pack order without native arrays or model graphs.
use mlx_gen::{Error, Result};
use mlx_rs::Dtype;
use mlx_gen::adapters::facts::TensorFacts as Tensor;
use mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader;
use crate::config::UNetConfig;
use crate::quantum as q;
use crate::plan::SdxlForwardPlan;
use super::block::BlockSpec;

pub(crate) type MetaVisit<'a> = dyn FnMut(&str, &mut q::LinearWork, i32) -> Result<()> + 'a;

/// Local convenience for existing fact values, not a native tensor interface or payload factory.
pub(crate) trait FactView {
    fn shape(&self) -> &[i32];
    fn dtype(&self) -> Dtype;
    fn size(&self) -> usize;
    fn reshape(&self, shape: &[i32]) -> Result<Tensor>;
}
impl FactView for Tensor {
    fn shape(&self) -> &[i32] { &self.shape }
    fn dtype(&self) -> Dtype { self.dtype }
    fn size(&self) -> usize { self.elements() as usize }
    fn reshape(&self, shape: &[i32]) -> Result<Tensor> {
        let result = tensor(shape, self.dtype)?;
        if result.elements() != self.elements() {
            return Err(Error::Msg("sdxl metadata reshape element counts differ".into()));
        }
        Ok(result)
    }
}
pub(crate) fn tensor(shape: &[i32], dtype: Dtype) -> Result<Tensor> {
    // Native dimensions are signed i32. Negative explicit extents are not accepted by zeros.
    if shape.iter().any(|&d| d < 0) { return Err(Error::Msg("sdxl metadata tensor has a negative dimension".into())); }
    Ok(Tensor { shape: shape.to_vec(), dtype, strides: Vec::new() })
}
pub(crate) fn tensor_bytes(tensor: &Tensor) -> u64 { tensor.bytes() }
fn tensor_elements(tensor: &Tensor) -> u64 { tensor.elements() }
pub(crate) fn element_bytes(tensor: &Tensor) -> u64 { q::dtype_bytes(tensor.dtype) }
pub(crate) fn promote(input: Dtype, tensors: &[&Tensor]) -> Dtype {
    tensors.iter().fold(input, |d, t| Dtype::from_promoting_types(d, t.dtype))
}
fn linear_out(linear: &q::LinearWork) -> u64 { linear.output_width() }
pub(crate) fn nchw_to_nhwc(weight: &Tensor) -> Result<Tensor> {
    if weight.shape.len() != 4 { return Err(Error::Msg("sdxl metadata convolution transpose requires four axes".into())); }
    tensor(&[weight.shape[0], weight.shape[2], weight.shape[3], weight.shape[1]], weight.dtype)
}

pub(crate) struct HeaderWeights {
    tensors: std::collections::BTreeMap<String, Tensor>,
}
impl HeaderWeights {
    pub(crate) fn new(headers: &[SafetensorsTensorHeader]) -> Result<Self> {
        let mut tensors = std::collections::BTreeMap::new();
        // The native safetensors decoder visits every dtype before model constructor lookups.
        for header in headers { crate::quant::header_dtype(header.dtype)?; }
        for header in headers {
            tensors.insert(header.name.clone(), crate::text_encoder::clip_header_tensor(headers, &header.name)?);
        }
        Ok(Self { tensors })
    }
    /// The VAE loader's explicit cast_all policy, including dense integer leaves. Do not infer
    /// the UNet's packed-component exception or substitute the provider's DTYPE for this dtype.
    pub(crate) fn new_all_cast(headers: &[SafetensorsTensorHeader], dtype: Dtype) -> Result<Self> {
        let mut weights = Self::new(headers)?;
        for tensor in weights.tensors.values_mut() { tensor.dtype = dtype; }
        Ok(weights)
    }
    pub(crate) fn require(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name).ok_or_else(|| Error::MissingTensor(name.to_owned()))
    }
    pub(crate) fn get(&self, name: &str) -> Option<&Tensor> { self.tensors.get(name) }
}
fn header_linear(weights: &HeaderWeights, path: &str, biased: bool) -> Result<q::LinearWork> {
    // Exactly quant::lin's lookup order over the constructor's ALREADY cast leaves. This also
    // serves the VAE's optional resnet time projection under its distinct cast_all policy.
    let bias = if biased { Some(weights.require(&format!("{path}.bias"))?.clone()) } else { None };
    if let Some(scales) = weights.get(&format!("{path}.scales")) {
        let codes = weights.require(&format!("{path}.weight"))?;
        let bits = mlx_gen::quant::packed_bits_from_metadata(&codes.shape, codes.dtype,
            &scales.shape, scales.dtype, crate::quant::GROUP_SIZE)?;
        let offsets = weights.require(&format!("{path}.biases"))?;
        let mut work = q::LinearWork::dense(codes.clone(), bias);
        work.base.base_shape = vec![scales.shape[0], scales.shape[1] * crate::quant::GROUP_SIZE];
        work.base.is_quantized = true; work.bits = Some(bits);
        work.weights = vec![codes.clone(), scales.clone(), offsets.clone()];
        Ok(work)
    } else { Ok(q::LinearWork::dense(weights.require(&format!("{path}.weight"))?.clone(), bias)) }
}
fn row_slice(tensor: &Tensor, lo: i32, hi: i32) -> Result<Tensor> {
    let rows = tensor.shape.first().copied().ok_or_else(|| Error::Msg("sdxl metadata take axis0 on a scalar".into()))?;
    if lo < 0 || hi < lo || hi > rows { return Err(Error::Msg("sdxl metadata take row index out of bounds".into())); }
    let mut shape = tensor.shape.clone(); shape[0] = hi - lo;
    self::tensor(&shape, tensor.dtype)
}
fn header_geglu_half(weights: &HeaderWeights, base: &str, lo: i32, hi: i32) -> Result<q::LinearWork> {
    // Match lin_geglu_half: bias and its row selection precede packed selection/validation.
    let bias = Some(row_slice(weights.require(&format!("{base}.bias"))?, lo, hi)?);
    if let Some(scales) = weights.get(&format!("{base}.scales")) {
        let codes = weights.require(&format!("{base}.weight"))?;
        let offsets = weights.require(&format!("{base}.biases"))?;
        let bits = mlx_gen::quant::packed_bits_from_metadata(&codes.shape, codes.dtype,
            &scales.shape, scales.dtype, crate::quant::GROUP_SIZE)?;
        let codes = row_slice(codes, lo, hi)?;
        let scales = row_slice(scales, lo, hi)?;
        let offsets = row_slice(offsets, lo, hi)?;
        let mut result = q::LinearWork::dense(codes.clone(), bias);
        result.base.base_shape = vec![scales.shape[0], scales.shape[1] * crate::quant::GROUP_SIZE];
        result.base.is_quantized = true; result.bits = Some(bits);
        result.weights = vec![codes, scales, offsets];
        return Ok(result);
    }
    Ok(q::LinearWork::dense(row_slice(weights.require(&format!("{base}.weight"))?, lo, hi)?, bias))
}

struct SinusoidFacts { _sigmas: Tensor, _host_bytes: u64 }
impl SinusoidFacts {
    fn timestep(dim: i32) -> Result<Self> {
        // Exact constructor table shape/dtype, including its retained arange/scalar host graph.
        Ok(Self { _sigmas: tensor(&[dim / 2], Dtype::Float32)?,
            _host_bytes: super::SinusoidalPositionalEncoding::timestep_table_bytes(dim) })
    }
}
#[derive(Clone)]
pub(crate) struct StreamFacts {
    prefix: String, headers: Vec<SafetensorsTensorHeader>, n_blocks: usize,
    dtype: Dtype, quant_bits: Option<i32>, bytes: std::sync::OnceLock<Vec<u64>>,
}
impl StreamFacts {
    fn streamed_bytes(&self, index: usize) -> Result<u64> {
        // Preserve the native stream's first-price validation point, rather than introducing an
        // eager arm-time refusal before a configured adapter outcome has been selected.
        if self.bytes.get().is_none() {
            let bytes = crate::block_stream::streamed_block_bytes_from_headers(
                &self.headers, &self.prefix, self.n_blocks, self.dtype, self.quant_bits)?;
            let _ = self.bytes.set(bytes);
        }
        self.bytes.get().and_then(|bytes| bytes.get(index).copied()).ok_or_else(|| Error::Msg(format!(
            "sdxl block stream `{}`: block {index} is out of range for a {}-block sub-stack", self.prefix, self.n_blocks)))
    }
}


#[derive(Clone)]
pub(crate) struct ResnetFacts {
    pub(super) norm1_w: Tensor, pub(super) norm1_b: Tensor,
    pub(super) conv1: q::ConvWork, pub(super) norm2_w: Tensor, pub(super) norm2_b: Tensor,
    pub(super) conv2: q::ConvWork, pub(super) time_emb_proj: Option<q::LinearWork>,
    pub(super) shortcut: Option<q::LinearWork>,
}
#[derive(Clone)]
pub(crate) struct TimeFacts { pub(super) linear1: q::LinearWork, pub(super) linear2: q::LinearWork }
#[derive(Clone)]
pub(crate) struct AttentionFacts {
    pub(super) q: q::LinearWork, pub(super) k: q::LinearWork, pub(super) v: q::LinearWork,
    pub(super) out: q::LinearWork, pub(super) num_heads: i32, pub(super) head_dim: i32,
    pub(super) to_k_ip: Option<q::LinearWork>, pub(super) to_v_ip: Option<q::LinearWork>,
}
#[derive(Clone)]
pub(crate) struct TransformerBlockFacts {
    pub(super) norm1_w: Tensor, pub(super) norm1_b: Tensor,
    pub(super) norm2_w: Tensor, pub(super) norm2_b: Tensor,
    pub(super) norm3_w: Tensor, pub(super) norm3_b: Tensor,
    pub(super) attn1: AttentionFacts, pub(super) attn2: AttentionFacts,
    pub(super) linear1: q::LinearWork, pub(super) linear2: q::LinearWork, pub(super) linear3: q::LinearWork,
}
#[derive(Clone)]
pub(crate) struct TransformerFacts {
    pub(super) norm_w: Tensor, pub(super) norm_b: Tensor,
    pub(super) proj_in: q::LinearWork, pub(super) blocks: Vec<TransformerBlockFacts>,
    pub(super) proj_out: q::LinearWork, pub(super) block_stream: Option<StreamFacts>,
}
#[derive(Clone)]
pub(crate) struct BlockFacts {
    pub(super) resnets: Vec<ResnetFacts>, pub(super) attentions: Option<Vec<TransformerFacts>>,
    pub(super) downsample: Option<q::ConvWork>, pub(super) upsample: Option<q::ConvWork>,
}
pub(crate) struct UNetFacts {
    pub(super) conv_in: q::ConvWork, _timesteps: SinusoidFacts,
    pub(super) time_embedding: TimeFacts, _add_time_proj: SinusoidFacts,
    pub(super) add_embedding: TimeFacts, pub(super) down_blocks: Vec<BlockFacts>,
    pub(super) mid_resnet0: ResnetFacts, pub(super) mid_transformer: TransformerFacts,
    pub(super) mid_resnet1: ResnetFacts, pub(super) up_blocks: Vec<BlockFacts>,
    pub(super) conv_norm_out_w: Tensor, pub(super) conv_norm_out_b: Tensor,
    pub(super) conv_out: q::ConvWork, pub(super) encoder_hid_proj: Option<q::LinearWork>,
}
#[derive(Clone)]
pub(crate) struct ConvFacts { pub(super) weight: Tensor, pub(super) bias: Tensor, pub(super) stride: i32, pub(super) pad: i32 }
#[derive(Clone)]
pub(crate) struct CondFacts { pub(super) conv_in: ConvFacts, pub(super) blocks: Vec<ConvFacts>, pub(super) conv_out: ConvFacts }
pub(crate) struct ControlFacts {
    pub(super) conv_in: ConvFacts, _timesteps: SinusoidFacts,
    pub(super) time_embedding: TimeFacts, _add_time_proj: SinusoidFacts,
    pub(super) add_embedding: TimeFacts, pub(super) down_blocks: Vec<BlockFacts>,
    pub(super) mid_resnet0: ResnetFacts, pub(super) mid_transformer: TransformerFacts,
    pub(super) mid_resnet1: ResnetFacts, pub(super) cond_embedding: CondFacts,
    pub(super) down_zero: Vec<ConvFacts>, pub(super) mid_zero: ConvFacts,
    pub(super) encoder_hid_proj: Option<q::LinearWork>,
}

impl ResnetFacts {
pub fn from_weights(w: &HeaderWeights, prefix: &str) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        let shortcut = match w.get(&format!("{prefix}.conv_shortcut.weight")) {
            Some(sw) => {
                // [out, in, 1, 1] → [out, in] Linear over NHWC channels.
                let sh = sw.shape();
                let w2 = sw.reshape(&[sh[0], sh[1]])?;
                let b = w.require(&format!("{prefix}.conv_shortcut.bias"))?.clone();
                Some(q::LinearWork::dense(w2, Some(b)))
            }
            None => None,
        };
        // VAE resnets carry no time embedding (`temb_channels=None`). The U-Net's `time_emb_proj` is
        // quantized (packed-detect, sc-8746); the VAE resnets never reach this crate's quantize path.
        // Detect presence by the dense `.weight` OR a packed `.scales` (a pre-quantized snapshot has
        // no dense `.weight` key beyond the u32 codes, but always carries `.scales`).
        let has_time_proj = w.get(&format!("{prefix}.time_emb_proj.weight")).is_some()
            || w.get(&format!("{prefix}.time_emb_proj.scales")).is_some();
        let time_emb_proj = if has_time_proj {
            Some(header_linear(
                w,
                &format!("{prefix}.time_emb_proj"),
                true,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            conv1: q::ConvWork::new(nchw_to_nhwc(&g("conv1.weight")?)?, Some(g("conv1.bias")?)),
            time_emb_proj,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            conv2: q::ConvWork::new(nchw_to_nhwc(&g("conv2.weight")?)?, Some(g("conv2.bias")?)),
            shortcut,
        })
    }

pub(crate) fn quantize_using(&mut self, prefix: &str, bits: i32, visit: &mut MetaVisit<'_>) -> Result<()> {
        if let Some(t) = &mut self.time_emb_proj {
            visit(&crate::quant::child_path(prefix, "time_emb_proj"), t, bits)?;
        }
        // `shortcut` is the 1×1 `conv_shortcut` stored as a Linear (for the NHWC channel-matmul
        // forward), not a true Linear — so it stays DENSE like every other conv in the U-Net
        // (`conv_in`/`conv_out`, resnet `conv1`/`conv2`, the up/down samplers; see
        // `UNetFacts::quantize`). It sits directly on the resnet residual/skip path, so
        // int8 rounding error there injects straight into the residual stream and, at 1024², compounds
        // across the denoise loop into a runaway latent outlier that blows out the VAE → a flat image
        // (sc-3329). The defect is the int8 rounding itself, not a dtype/overflow issue (f32 scales +
        // f32 activations reproduce it identically), and it stayed sub-threshold at ≤512² so sc-2641
        // missed it. The `mlx_sd` reference also stores `conv_shortcut` as `nn.Linear` and quantizes
        // it, so leaving it dense is a deliberate, correct divergence from a reference that shares the
        // bug. (LoRA still merges into `shortcut` as a reshaped conv delta — that path is unaffected.)
        Ok(())
    }

pub(crate) fn quantum_scratch_with_effects(&self, x: &Tensor, temb: Option<&Tensor>, effects: &crate::quantum::EffectScope<'_>) -> Result<u64> {
        #[cfg(test)]
        {
            trace_facts(crate::quantum::UNET_RESNET, x);
            trace_facts("resnet conv1", self.conv1.weight());
            trace_facts("resnet conv2", self.conv2.weight());
        }
        let act_in = element_bytes(x);
        let act_out = match (&self.time_emb_proj, temb) {
            (Some(proj), Some(t)) => {
                let projected = effects.linear_metadata("time_emb_proj", proj)?.output_dtype(t.dtype());
                crate::quantum::dtype_bytes(Dtype::from_promoting_types(x.dtype(), projected))
            }
            _ => act_in,
        };
        let conv1 = effects.conv_metadata("conv1", &self.conv1);
        let conv2 = effects.conv_metadata("conv2", &self.conv2);
        Ok(crate::quantum::resnet_bytes_from_metadata(
            x.shape(), act_in, act_out,
            &conv1.weight.shape, crate::quantum::dtype_bytes(conv1.weight.dtype),
            &conv2.weight.shape, crate::quantum::dtype_bytes(conv2.weight.dtype),
            crate::quantum::backend(),
        )
        .saturating_add(self.promotion_scratch_with_effects(x, temb, effects)?))
    }

pub(crate) fn promotion_scratch_with_effects(&self, x: &Tensor, temb: Option<&Tensor>, effects: &crate::quantum::EffectScope<'_>) -> Result<u64> {
        use crate::quantum as q;
        let casts = |input: Dtype, arrays: &[&Tensor]| {
            let output = promote(input, arrays);
            arrays
                .iter()
                .map(|a| {
                    if a.dtype() != output {
                        tensor_elements(a) * q::dtype_bytes(output)
                    } else {
                        0
                    }
                })
                .sum::<u64>()
        };
        let conv1 = effects.conv_metadata("conv1", &self.conv1);
        let conv2 = effects.conv_metadata("conv2", &self.conv2);
        let n1 = promote(x.dtype(), &[&self.norm1_w, &self.norm1_b]);
        let c1 = Dtype::from_promoting_types(n1, conv1.weight.dtype);
        let mut mid = self
            .conv1
            .bias()
            .map_or(c1, |b| Dtype::from_promoting_types(c1, b.dtype()));
        let mut extra = casts(x.dtype(), &[&self.norm1_w, &self.norm1_b]);
        if let Some(b) = self.conv1.bias() {
            extra += casts(c1, &[b]);
        }
        if let (Some(p), Some(t)) = (&self.time_emb_proj, temb) {
            let projection = effects.linear_metadata("time_emb_proj", p)?;
            extra += projection.operand_bytes(t.shape(), t.dtype());
            mid = Dtype::from_promoting_types(mid, projection.output_dtype(t.dtype()));
        }
        extra += casts(mid, &[&self.norm2_w, &self.norm2_b]);
        let n2 = promote(mid, &[&self.norm2_w, &self.norm2_b]);
        let c2 = Dtype::from_promoting_types(n2, conv2.weight.dtype);
        if let Some(b) = self.conv2.bias() {
            extra += casts(c2, &[b]);
        }
        if let Some(shortcut) = &self.shortcut {
            let mut shape = x.shape().to_vec();
            *shape.last_mut().expect("NHWC input") = conv1.weight.shape[3];
            extra += effects.shortcut_metadata("conv_shortcut", shortcut).operand_bytes(&shape, x.dtype());
        }
        Ok(extra)
    }

pub(crate) fn output_dtype_with_effects(&self, input: Dtype, temb: Option<Dtype>, effects: &crate::quantum::EffectScope<'_>) -> Result<Dtype> {
        let mut arrays: Vec<&Tensor> = vec![&self.norm1_w, &self.norm1_b];
        arrays.extend(self.conv1.bias());
        arrays.extend([&self.norm2_w, &self.norm2_b]);
        arrays.extend(self.conv2.bias());
        let conv1 = effects.conv_metadata("conv1", &self.conv1);
        let conv2 = effects.conv_metadata("conv2", &self.conv2);
        let mut y = Dtype::from_promoting_types(promote(input, &arrays),
            Dtype::from_promoting_types(conv1.weight.dtype, conv2.weight.dtype));
        if let (Some(proj), Some(t)) = (&self.time_emb_proj, temb) {
            y = Dtype::from_promoting_types(y, effects.linear_metadata("time_emb_proj", proj)?.output_dtype(t));
        }
        let residual = match &self.shortcut {
            Some(sc) => {
                Dtype::from_promoting_types(input, effects.shortcut_metadata("conv_shortcut", sc).output_dtype(input))
            }
            None => input,
        };
        Ok(Dtype::from_promoting_types(residual, y))
    }

pub(crate) fn output_shape_with_effects(
        &self, input: &[i32], skip: Option<&[i32]>, effects: &crate::quantum::EffectScope<'_>,
    ) -> Result<Vec<i32>> {
        let mut input = input.to_vec();
        if let Some(skip) = skip {
            if input[..3] != skip[..3] { return Err(mlx_gen::Error::Msg("sdxl resnet skip concat axes disagree".into())); }
            input[3] = input[3].checked_add(skip[3]).ok_or_else(|| mlx_gen::Error::Msg("sdxl resnet concat width overflow".into()))?;
        }
        let first = effects.conv_metadata("conv1", &self.conv1).output_shape(&input, 1, 1)?;
        let output = effects.conv_metadata("conv2", &self.conv2).output_shape(&first, 1, 1)?;
        let residual = match &self.shortcut {
            Some(shortcut) => effects.shortcut_metadata("conv_shortcut", shortcut).output_shape(&input)?,
            None => input,
        };
        crate::adapters::broadcast_shape(&output, &residual)
    }

pub(crate) fn conv_weights(&self) -> (&Tensor, &Tensor) {
        (self.conv1.weight(), self.conv2.weight())
    }

}

impl TimeFacts {
pub fn from_weights(w: &HeaderWeights, prefix: &str) -> Result<Self> {
        // Packed-detect (sc-8746): both MLP Linears are quantized in [`Self::quantize`], so a
        // pre-quantized snapshot stores their packed triple — `crate::quant::lin` loads packed or
        // dense automatically.
        Ok(Self {
            linear1: header_linear(w, &format!("{prefix}.linear_1"), true)?,
            linear2: header_linear(w, &format!("{prefix}.linear_2"), true)?,
        })
    }

pub(crate) fn quantize_using(&mut self, prefix: &str, bits: i32, visit: &mut MetaVisit<'_>) -> Result<()> {
        visit(&crate::quant::child_path(prefix, "linear_1"), &mut self.linear1, bits)?;
        visit(&crate::quant::child_path(prefix, "linear_2"), &mut self.linear2, bits)?;
        Ok(())
    }

pub(crate) fn output_dtype_with_effects(&self, x: Dtype, effects: &crate::quantum::EffectScope<'_>) -> Result<Dtype> {
        let first = effects.linear_metadata("linear_1", &self.linear1)?.output_dtype(x);
        let first = Dtype::from_promoting_types(x, first);
        Ok(Dtype::from_promoting_types(first,
            effects.linear_metadata("linear_2", &self.linear2)?.output_dtype(first)))
    }

pub(crate) fn projection_scratch(&self, batch: i32, input: Dtype, effects: &crate::quantum::EffectScope<'_>) -> Result<u64> {
        let first = effects.linear_metadata("linear_1", &self.linear1)?;
        let second = effects.linear_metadata("linear_2", &self.linear2)?;
        let shape = [batch, *first.base.base_shape.last().unwrap_or(&0)];
        let hidden = first.output_dtype(input);
        let intermediate = [batch, first.output_width() as i32];
        let output = second.output_dtype(hidden);
        let plane = |dtype, width| mlx_gen::ios_quantum::tensor_bytes(
            crate::quantum::dtype_bytes(dtype), &[batch.max(0) as u64, width]);
        Ok(first.operand_bytes(&shape, input)
            .saturating_add(plane(hidden, first.output_width()).saturating_mul(3))
            .saturating_add(second.operand_bytes(&intermediate, hidden))
            .saturating_add(plane(output, second.output_width()))
            .saturating_add(crate::quantum::page_slack(4)))
    }

pub(crate) fn out_features_with_effects(&self, effects: &crate::quantum::EffectScope<'_>) -> Result<i32> {
        Ok(effects.linear_metadata("linear_2", &self.linear2)?.output_width() as i32)
    }

}

impl AttentionFacts {
fn from_weights(w: &HeaderWeights, prefix: &str, model_dims: i32, num_heads: i32) -> Result<Self> {
        // Packed-detect (sc-8746): q/k/v (bias-free) + to_out.0 (biased) are all quantized in
        // [`Self::quantize`], so `crate::quant::lin` loads their packed triple or the dense weight.
        let no_bias = |n: &str| header_linear(w, &format!("{prefix}.{n}"), false);
        let head_dim = model_dims / num_heads;
        Ok(Self {
            q: no_bias("to_q")?,
            k: no_bias("to_k")?,
            v: no_bias("to_v")?,
            out: header_linear(w, &format!("{prefix}.to_out.0"), true)?,
            num_heads,
            head_dim,
            
            to_k_ip: None,
            to_v_ip: None,
            
        })
    }

fn install_ip(&mut self, k_ip: Tensor, v_ip: Tensor) {
        self.to_k_ip = Some(q::LinearWork::dense(k_ip, None));
        self.to_v_ip = Some(q::LinearWork::dense(v_ip, None));
    }

fn quantize_using(&mut self, prefix: &str, bits: i32, visit: &mut MetaVisit<'_>) -> Result<()> {
        for (name, lin) in [("to_q", &mut self.q), ("to_k", &mut self.k),
            ("to_v", &mut self.v), ("to_out.0", &mut self.out)] {
            visit(&crate::quant::child_path(prefix, name), lin, bits)?;
        }
        for (name, lin) in [("to_k_ip", &mut self.to_k_ip), ("to_v_ip", &mut self.to_v_ip)] {
            if let Some(lin) = lin { visit(&crate::quant::child_path(prefix, name), lin, bits)?; }
        }
        Ok(())
    }

}

impl TransformerBlockFacts {
pub(crate) fn from_weights(
        w: &HeaderWeights,
        prefix: &str,
        model_dims: i32,
        num_heads: i32,
    ) -> Result<Self> {
        // GEGLU: `ff.net.0.proj` is one `[2*hidden, D]` Linear on disk, row-split into value/gate
        // halves. Determine `2*hidden` from the packed `.scales` grid (rows) when present, else from
        // the dense weight — the split rows are identical either way (sc-8746). The packed row-slice
        // is byte-identical to the dense split-then-quantize (quantization is per-row).
        let ff_proj = format!("{prefix}.ff.net.0.proj");
        let two_h = match w.get(&format!("{ff_proj}.scales")) {
            Some(scales) => scales.shape()[0],
            None => w.require(&format!("{ff_proj}.weight"))?.shape()[0],
        };
        let [(lo1, hi1), (lo2, hi2)] = crate::quant::geglu_half_ranges(two_h)?;
        // Value/gate halves — the same actual row selector serves header normalization.
        let linear1 = header_geglu_half(w, &ff_proj, lo1, hi1)?;
        let linear2 = header_geglu_half(w, &ff_proj, lo2, hi2)?;
        // FFN output (`ff.net.2`) is a plain quantized Linear.
        let linear3 = header_linear(w, &format!("{prefix}.ff.net.2"), true)?;
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            norm3_w: g("norm3.weight")?,
            norm3_b: g("norm3.bias")?,
            attn1: AttentionFacts::from_weights(
                w,
                &format!("{prefix}.attn1"),
                model_dims,
                num_heads,
            )?,
            attn2: AttentionFacts::from_weights(
                w,
                &format!("{prefix}.attn2"),
                model_dims,
                num_heads,
            )?,
            linear1,
            linear2,
            linear3,
        })
    }

pub(crate) fn quantize_using(&mut self, prefix: &str, bits: i32, visit: &mut MetaVisit<'_>) -> Result<()> {
        self.attn1.quantize_using(&crate::quant::child_path(prefix, "attn1"), bits, visit)?;
        self.attn2.quantize_using(&crate::quant::child_path(prefix, "attn2"), bits, visit)?;
        visit(&crate::quant::child_path(prefix, "ff.linear1"), &mut self.linear1, bits)?;
        visit(&crate::quant::child_path(prefix, "ff.linear2"), &mut self.linear2, bits)?;
        visit(&crate::quant::child_path(prefix, "ff.linear3"), &mut self.linear3, bits)?;
        Ok(())
    }

fn install_ip(&mut self, k_ip: Tensor, v_ip: Tensor) {
        self.attn2.install_ip(k_ip, v_ip);
    }

pub(crate) fn has_ip(&self) -> bool {
        self.attn2.to_k_ip.is_some() && self.attn2.to_v_ip.is_some()
    }

pub(crate) fn quantum_scratch_from_facts(
        &self,
        batch: i32,
        context_batch: i32,
        tokens: i32,
        context_tokens: i32,
        ip: bool,
        x: Dtype,
        attention: mlx_gen::attention::AttentionPlan<'_>,
        activation: Dtype,
        ff_width: u64,
    ) -> u64 {
        let heads = self.attn1.num_heads;
        let act = crate::quantum::dtype_bytes(activation);
        let dim = self.norm1_w.shape()[0] as u64;
        // Entered narrower than the projections promote to: the self-attention's q/k/v each cast
        // the narrow LayerNorm output up (`quantized_matmul`'s `astype(x, result_type)`).
        let casts = if act > crate::quantum::dtype_bytes(x) {
            mlx_gen::ios_quantum::tensor_bytes(act, &[3, batch.max(1) as u64, tokens as u64, dim])
                .saturating_add(crate::quantum::page_slack(3))
        } else {
            0
        };
        // The two attentions are planned independently by `sdpa_budgeted_bhsd`; the self-attention
        // (the larger key axis) has the smaller chunk, the cross-attention the larger — each prices
        // at its own rows, so take the plan of the larger score domain (rows × keys).
        let self_rows = attention
            .budget
            .query_block(batch, heads, tokens, tokens)
            .clamp(1, tokens.max(1));
        let cross_rows = attention
            .budget
            .query_block(batch, heads, tokens, context_tokens)
            .clamp(1, tokens.max(1));
        let rows = if (self_rows as i64) * (tokens as i64)
            >= (cross_rows as i64) * (context_tokens as i64)
        {
            self_rows
        } else {
            cross_rows
        };
        casts.saturating_add(crate::quantum::transformer_block_bytes(
            batch,
            context_batch,
            tokens,
            dim,
            heads,
            self.attn1.head_dim,
            context_tokens,
            ff_width,
            rows,
            ip && self.has_ip(),
            act,
            crate::quantum::backend(),
        ))
    }

pub(crate) fn projection_work(
        &self, batch: i32, context_batch: i32, tokens: i32, hidden_width: i32,
        context_tokens: i32, memory_width: i32,
        x: Dtype, memory: Dtype, ip: Option<(i32, i32, Dtype)>,
        effects: &crate::quantum::EffectScope<'_>,
    ) -> Result<(Dtype, u64)> {
        // Native projections read ACTIVATIONS, not the pre-install borrowed weight's axes.
        // Selected effects may have changed a dense base before its actual load-time pack.
        let dim = hidden_width;
        let operand = |_family: &str, _role: &str, linear: &crate::quantum::LinearWork,
            shape: &[i32], dtype: Dtype| {
            let bytes = linear.operand_bytes(shape, dtype);
            #[cfg(test)]
            super::transformer::PROJECTION_OPERAND_LOG.with(|log| {
                if let Some(records) = log.borrow_mut().as_mut() {
                    records.push((format!("{_family}.{_role}"), shape.to_vec(), dtype, bytes));
                }
            });
            bytes
        };
        let mut scratch = 0u64;
        let mut hidden = x;
        for (name, attention, norm_w, norm_b, cross) in [
            ("attn1", &self.attn1, &self.norm1_w, &self.norm1_b, false),
            ("attn2", &self.attn2, &self.norm2_w, &self.norm2_b, true),
        ] {
            let scope = effects.child(name);
            let norm = promote(hidden, &[norm_w, norm_b]);
            let query = [batch, tokens, dim];
            let key = [if cross { context_batch } else { batch },
                if cross { context_tokens } else { tokens },
                if cross { memory_width } else { hidden_width }];
            let key_dtype = if cross { memory } else { norm };
            let q = scope.linear_metadata("to_q", &attention.q)?;
            let k = scope.linear_metadata("to_k", &attention.k)?;
            let v = scope.linear_metadata("to_v", &attention.v)?;
            scratch = scratch.saturating_add(operand(name, "to_q", &q, &query, norm))
                .saturating_add(operand(name, "to_k", &k, &key, key_dtype))
                .saturating_add(operand(name, "to_v", &v, &key, key_dtype));
            let mut attended = Dtype::from_promoting_types(q.output_dtype(norm),
                Dtype::from_promoting_types(k.output_dtype(key_dtype), v.output_dtype(key_dtype)));
            if cross {
                if let Some((ip_tokens, ip_width, ip_dtype)) = ip {
                    for (name, projection) in [("to_k_ip", &attention.to_k_ip), ("to_v_ip", &attention.to_v_ip)] {
                        if let Some(projection) = projection {
                            let work = scope.linear_metadata(name, projection)?;
                            scratch = scratch.saturating_add(operand("attn2", name, &work,
                                &[context_batch, ip_tokens, ip_width], ip_dtype));
                            attended = Dtype::from_promoting_types(attended, work.output_dtype(ip_dtype));
                        }
                    }
                }
            }
            let out = scope.linear_metadata("to_out.0", &attention.out)?;
            scratch = scratch.saturating_add(operand(name, "to_out.0", &out,
                &[batch, tokens, attention.num_heads * attention.head_dim], attended));
            hidden = Dtype::from_promoting_types(hidden, out.output_dtype(attended));
        }
        let norm = promote(hidden, &[&self.norm3_w, &self.norm3_b]);
        let value = effects.linear_metadata("ff.linear1", &self.linear1)?;
        let gate = effects.linear_metadata("ff.linear2", &self.linear2)?;
        let output = effects.linear_metadata("ff.linear3", &self.linear3)?;
        let input = [batch, tokens, dim];
        scratch = scratch.saturating_add(operand("ff", "linear1", &value, &input, norm))
            .saturating_add(operand("ff", "linear2", &gate, &input, norm));
        let product = Dtype::from_promoting_types(value.output_dtype(norm), gate.output_dtype(norm));
        scratch = scratch.saturating_add(operand("ff", "linear3", &output,
            &[batch, tokens, value.output_width() as i32], product));
        Ok((Dtype::from_promoting_types(hidden, output.output_dtype(product)), scratch))
    }

pub(crate) fn context_tokens(&self, memory: &Tensor, ip: Option<(&Tensor, f32)>) -> i32 {
        let ip_tokens = match ip {
            Some((tokens, _)) if self.has_ip() => tokens.shape()[1],
            _ => 0,
        };
        memory.shape()[1] + ip_tokens
    }

}

impl TransformerFacts {
pub fn from_weights(
        w: &HeaderWeights,
        prefix: &str,
        model_dims: i32,
        num_heads: i32,
        num_layers: i32,
    ) -> Result<Self> {
        let blocks = (0..num_layers)
            .map(|i| {
                TransformerBlockFacts::from_weights(
                    w,
                    &format!("{prefix}.transformer_blocks.{i}"),
                    model_dims,
                    num_heads,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            norm_w: w.require(&format!("{prefix}.norm.weight"))?.clone(),
            norm_b: w.require(&format!("{prefix}.norm.bias"))?.clone(),
            // Packed-detect (sc-8746): SDXL uses linear `proj_in`/`proj_out`, both quantized.
            proj_in: header_linear(w, &format!("{prefix}.proj_in"), true)?,
            blocks,
            proj_out: header_linear(w, &format!("{prefix}.proj_out"), true)?,
            block_stream: None,
        })
    }

pub(crate) fn quantize_using(&mut self, prefix: &str, bits: i32, visit: &mut MetaVisit<'_>) -> Result<()> {
        // Preserve the actual pack order: BOTH projections precede the transformer blocks.
        visit(&crate::quant::child_path(prefix, "proj_in"), &mut self.proj_in, bits)?;
        visit(&crate::quant::child_path(prefix, "proj_out"), &mut self.proj_out, bits)?;
        for (index, b) in self.blocks.iter_mut().enumerate() {
            b.quantize_using(&crate::quant::child_path(prefix, &format!("transformer_blocks.{index}")), bits, visit)?;
        }
        Ok(())
    }

pub(crate) fn block_price_with_effects(
        &self,
        index: usize,
        y: &Tensor,
        encoder_x: &Tensor,
        ip: Option<(&Tensor, f32)>,
        plan: SdxlForwardPlan<'_>,
        effects: &crate::quantum::EffectScope<'_>,
    ) -> Result<u64> {
        let block = &self.blocks[index];
        let scope = effects.child(&format!("transformer_blocks.{index}"));
        let context_batch = plan.conditioning_batch(y.shape()[0]).max(encoder_x.shape()[0]);
        let (activation, operands) = block.projection_work(y.shape()[0], context_batch, y.shape()[1], y.shape()[2],
            encoder_x.shape()[1], encoder_x.shape()[2], y.dtype(), encoder_x.dtype(),
            ip.map(|(a, _)| (a.shape()[1], a.shape()[2], a.dtype())), &scope)?;
        let activations = block.quantum_scratch_from_facts(y.shape()[0], context_batch, y.shape()[1],
            block.context_tokens(encoder_x, ip), ip.is_some(), y.dtype(), plan.attention,
            activation, scope.linear_metadata("ff.linear1", &block.linear1)?.output_width());
        let weights = match &self.block_stream {
            Some(stream) => stream.streamed_bytes(index)?,
            None => 0,
        };
        Ok(activations.saturating_add(operands).saturating_add(weights))
    }

pub(crate) fn projection_budgets_with_effects(
        &self,
        x: &Tensor,
        encoder_x: &Tensor,
        ip: Option<(&Tensor, f32)>,
        plan: SdxlForwardPlan<'_>,
        effects: &crate::quantum::EffectScope<'_>,
    ) -> Result<(u64, u64, i32, Dtype, Dtype)> {
        let input = effects.linear_metadata("proj_in", &self.proj_in)?;
        let output = effects.linear_metadata("proj_out", &self.proj_out)?;
        let s = x.shape();
        let shape = [s[0], s[1] * s[2], s[3]];
        let norm = promote(x.dtype(), &[&self.norm_w, &self.norm_b]);
        let inner = input.output_width() as i32;
        let mut hidden = input.output_dtype(norm);
        let groups = crate::quantum::group_norm_buffers(crate::quantum::backend(), element_bytes(x));
        let plane = |dtype, width| mlx_gen::ios_quantum::tensor_bytes(crate::quantum::dtype_bytes(dtype),
            &[s[0] as u64, s[1] as u64, s[2] as u64, width]);
        let proj_in = plane(x.dtype(), s[3] as u64).saturating_mul(groups)
            .saturating_add(input.operand_bytes(&shape, norm))
            .saturating_add(plane(hidden, inner as u64))
            .saturating_add(crate::quantum::page_slack(groups + 1));
        let entered = hidden;
        for (index, block) in self.blocks.iter().enumerate() {
            hidden = block.projection_work(s[0], plan.conditioning_batch(s[0]).max(encoder_x.shape()[0]),
                s[1] * s[2], inner, encoder_x.shape()[1], encoder_x.shape()[2], hidden, encoder_x.dtype(),
                ip.map(|(a, _)| (a.shape()[1], a.shape()[2], a.dtype())),
                &effects.child(&format!("transformer_blocks.{index}")))?.0;
        }
        let projected = output.output_dtype(hidden);
        let summed = Dtype::from_promoting_types(projected, x.dtype());
        let widened = u64::from(summed != projected);
        let proj_out = plane(projected, output.output_width())
            .saturating_add(output.operand_bytes(&[s[0], s[1] * s[2], inner], hidden))
            .saturating_add(plane(summed, output.output_width()).saturating_mul(widened))
            .saturating_add(crate::quantum::page_slack(1 + widened));
        Ok((proj_in, proj_out, inner, entered, summed))
    }

pub(crate) fn price_forward_with_effects(
        &self,
        x: &Tensor,
        encoder_x: &Tensor,
        ip: Option<(&Tensor, f32)>,
        plan: SdxlForwardPlan<'_>,
        out: &mut crate::quantum::Quanta,
        effects: &crate::quantum::EffectScope<'_>,
    ) -> Result<Tensor> {
        let (proj_in, proj_out, inner, mut hidden, summed) =
            self.projection_budgets_with_effects(x, encoder_x, ip, plan, effects)?;
        out.push((crate::quantum::TRANSFORMER_PROJ_IN, proj_in));
        let s = x.shape();
        let inner_plan = plan.with_window(None);
        for (index, block) in self.blocks.iter().enumerate() {
            let y = tensor(&[s[0], s[1] * s[2], inner], hidden)?;
            out.push((crate::quantum::TRANSFORMER_BLOCK,
                self.block_price_with_effects(index, &y, encoder_x, ip, inner_plan, effects)?));
            hidden = block.projection_work(s[0], plan.conditioning_batch(s[0]).max(encoder_x.shape()[0]),
                s[1] * s[2], inner, encoder_x.shape()[1], encoder_x.shape()[2], hidden, encoder_x.dtype(),
                ip.map(|(a, _)| (a.shape()[1], a.shape()[2], a.dtype())),
                &effects.child(&format!("transformer_blocks.{index}")))?.0;
        }
        out.push((crate::quantum::TRANSFORMER_PROJ_OUT, proj_out));
        tensor(s, summed)
    }

pub fn install_ip(&mut self, pairs: &mut impl Iterator<Item = (Tensor, Tensor)>) -> Result<()> {
        for block in &mut self.blocks {
            let (k_ip, v_ip) = pairs
                .next()
                .ok_or_else(|| mlx_gen::Error::Msg("ip_adapter: not enough K/V pairs".into()))?;
            block.install_ip(k_ip, v_ip);
        }
        Ok(())
    }

}

impl BlockFacts {
pub fn from_weights(w: &HeaderWeights, spec: &BlockSpec) -> Result<Self> {
        let resnets = (0..spec.num_resnets)
            .map(|j| ResnetFacts::from_weights(w, &format!("{}.resnets.{j}", spec.prefix)))
            .collect::<Result<Vec<_>>>()?;
        let attentions = if spec.add_cross_attention {
            Some(
                (0..spec.num_resnets)
                    .map(|j| {
                        TransformerFacts::from_weights(
                            w,
                            &format!("{}.attentions.{j}", spec.prefix),
                            spec.out_channels,
                            spec.num_heads,
                            spec.transformer_layers,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        let downsample = if spec.add_downsample {
            Some(q::ConvWork::new(
                nchw_to_nhwc(w.require(&format!("{}.downsamplers.0.conv.weight", spec.prefix))?)?,
                Some(
                    w.require(&format!("{}.downsamplers.0.conv.bias", spec.prefix))?
                        .clone(),
                ),
            ))
        } else {
            None
        };
        let upsample = if spec.add_upsample {
            Some(q::ConvWork::new(
                nchw_to_nhwc(w.require(&format!("{}.upsamplers.0.conv.weight", spec.prefix))?)?,
                Some(
                    w.require(&format!("{}.upsamplers.0.conv.bias", spec.prefix))?
                        .clone(),
                ),
            ))
        } else {
            None
        };
        Ok(Self {
            resnets,
            attentions,
            downsample,
            upsample,
        })
    }

pub(crate) fn quantize_using(&mut self, prefix: &str, bits: i32, visit: &mut MetaVisit<'_>) -> Result<()> {
        for (index, r) in self.resnets.iter_mut().enumerate() {
            r.quantize_using(&crate::quant::child_path(prefix, &format!("resnets.{index}")), bits, visit)?;
        }
        if let Some(attns) = &mut self.attentions {
            for (index, a) in attns.iter_mut().enumerate() {
                a.quantize_using(&crate::quant::child_path(prefix, &format!("attentions.{index}")), bits, visit)?;
            }
        }
        Ok(())
    }

pub(crate) fn price_forward_with_effects(
        &self,
        x: &Tensor,
        encoder_x: &Tensor,
        temb: &Tensor,
        mut residuals: Option<&mut Vec<Tensor>>,
        ip: Option<(&Tensor, f32)>,
        plan: crate::plan::SdxlForwardPlan<'_>,
        out: &mut crate::quantum::Quanta,
        outer_held: u64,
        mut owners: Option<&mut Vec<u64>>,
        effects: &crate::quantum::EffectScope<'_>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut x = x.clone();
        let mut output_states = Vec::with_capacity(self.resnets.len() + 1);
        for i in 0..self.resnets.len() {
            let skip = match residuals.as_deref_mut() {
                Some(res) => Some(res.pop().ok_or_else(|| {
                    mlx_gen::Error::Msg("sdxl unet pricing: residual stack underflow".into())
                })?),
                None => None,
            };
            let resnet = &self.resnets[i];
            if let Some(owners) = owners.as_deref_mut() {
                owners.push(
                    outer_held + output_states.iter().map(tensor_bytes).sum::<u64>(),
                );
            }
            out.push((
                crate::quantum::UNET_RESNET,
                resnet.quantum_scratch_with_effects(&x, Some(temb), &effects.child(&format!("resnets.{i}")))?,
            ));
            let input = skip.as_ref().map_or(x.dtype(), |s| {
                Dtype::from_promoting_types(x.dtype(), s.dtype())
            });
            let shape = resnet.output_shape_with_effects(x.shape(), skip.as_ref().map(Tensor::shape),
                &effects.child(&format!("resnets.{i}")))?;
            x = tensor(
                &shape,
                resnet.output_dtype_with_effects(input, Some(temb.dtype()), &effects.child(&format!("resnets.{i}")))?,
            )?;
            if let Some(attns) = &self.attentions {
                let first = out.len();
                // The caller's resnet output remains borrowed for the whole attention call;
                // only PREVIOUS output_states are already in the retained stack at this point.
                let held = outer_held
                    + output_states.iter().map(tensor_bytes).sum::<u64>()
                    + tensor_bytes(&x);
                x = attns[i].price_forward_with_effects(&x, encoder_x, ip, plan, out, &effects.child(&format!("attentions.{i}")))?;
                if let Some(owners) = owners.as_deref_mut() {
                    owners.extend(std::iter::repeat_n(held, out.len() - first));
                }
            }
            output_states.push(x.clone());
        }
        if let Some(c) = &self.downsample {
            if let Some(owners) = owners.as_deref_mut() {
                owners.push(
                    outer_held + output_states.iter().map(tensor_bytes).sum::<u64>(),
                );
            }
            let conv = effects.conv_metadata("downsamplers.0.conv", c);
            out.push((crate::quantum::UNET_DOWNSAMPLE, sampler_scratch(&x, &conv, false)));
            x = tensor(&conv.output_shape(x.shape(), 2, 1)?, conv.output_dtype(x.dtype()))?;
            output_states.push(x.clone());
        }
        if let Some(c) = &self.upsample {
            if let Some(owners) = owners.as_deref_mut() {
                owners.push(
                    outer_held + output_states.iter().map(tensor_bytes).sum::<u64>(),
                );
            }
            let conv = effects.conv_metadata("upsamplers.0.conv", c);
            out.push((crate::quantum::UNET_UPSAMPLE, sampler_scratch(&x, &conv, true)));
            let s = x.shape();
            let input = [s[0], 2 * s[1], 2 * s[2], s[3]];
            x = tensor(&conv.output_shape(&input, 1, 1)?, conv.output_dtype(x.dtype()))?;
            output_states.push(x.clone());
        }
        Ok((x, output_states))
    }

pub fn install_ip(&mut self, pairs: &mut impl Iterator<Item = (Tensor, Tensor)>) -> Result<()> {
        if let Some(attns) = &mut self.attentions {
            for a in attns {
                a.install_ip(pairs)?;
            }
        }
        Ok(())
    }

}

impl UNetFacts {
pub fn from_weights(w: &HeaderWeights, cfg: &UNetConfig) -> Result<Self> {
        let n = cfg.num_blocks();
        let boc = &cfg.block_out_channels;
        let temb_dim_src = boc[0]; // sinusoidal timestep width

        // The same pure block selector drives header-only adapter target enumeration.
        let mut down_blocks = Vec::with_capacity(n);
        for i in 0..n {
            let prefix = format!("down_blocks.{i}");
            down_blocks.push(BlockFacts::from_weights(
                w,
                &super::adapter_block_spec(cfg, i, false, &prefix),
            )?);
        }

        // Mid: resnet, transformer, resnet (the vendored mid_blocks.0/1/2).
        let mid_resnet0 = ResnetFacts::from_weights(w, "mid_block.resnets.0")?;
        let mid_transformer = TransformerFacts::from_weights(
            w,
            "mid_block.attentions.0",
            *boc.last().unwrap(),
            *cfg.num_attention_heads.last().unwrap(),
            *cfg.transformer_layers_per_block.last().unwrap(),
        )?;
        let mid_resnet1 = ResnetFacts::from_weights(w, "mid_block.resnets.1")?;

        let mut up_blocks = Vec::with_capacity(n);
        for k in 0..n {
            let prefix = format!("up_blocks.{k}");
            up_blocks.push(BlockFacts::from_weights(
                w,
                &super::adapter_block_spec(cfg, k, true, &prefix),
            )?);
        }

        Ok(Self {
            conv_in: q::ConvWork::new(
                nchw_to_nhwc(w.require("conv_in.weight")?)?,
                Some(w.require("conv_in.bias")?.clone()),
            ),
            _timesteps: SinusoidFacts::timestep(temb_dim_src)?,
            time_embedding: TimeFacts::from_weights(w, "time_embedding")?,
            _add_time_proj: SinusoidFacts::timestep(
                cfg.addition_time_embed_dim.unwrap_or(256),
            )?,
            add_embedding: TimeFacts::from_weights(w, "add_embedding")?,
            down_blocks,
            mid_resnet0,
            mid_transformer,
            mid_resnet1,
            up_blocks,
            conv_norm_out_w: w.require("conv_norm_out.weight")?.clone(),
            conv_norm_out_b: w.require("conv_norm_out.bias")?.clone(),
            conv_out: q::ConvWork::new(
                nchw_to_nhwc(w.require("conv_out.weight")?)?,
                Some(w.require("conv_out.bias")?.clone()),
            ),
            // Kolors `encoder_hid_proj` (4096→2048). Auto-detected: absent for SDXL → `None`.
            // Packed-detect (sc-8746): quantized in [`Self::quantize`], and Kolors carries a bias, so
            // load it packed-or-dense via `crate::quant::lin`. Presence is keyed on the dense
            // `.weight` OR the packed `.scales` (a Q4/Q8 snapshot has no dense weight beyond codes).
            encoder_hid_proj: if w.get("encoder_hid_proj.weight").is_some()
                || w.get("encoder_hid_proj.scales").is_some()
            {
                Some(header_linear(
                    w,
                    "encoder_hid_proj",
                    w.get("encoder_hid_proj.bias").is_some(),
                )?)
            } else {
                None
            },
        })
    }

pub(crate) fn quantize_using(&mut self, bits: i32, visit: &mut MetaVisit<'_>) -> Result<()> {
        self.time_embedding.quantize_using("time_embedding", bits, visit)?;
        self.add_embedding.quantize_using("add_embedding", bits, visit)?;
        if let Some(proj) = &mut self.encoder_hid_proj {
            visit("encoder_hid_proj", proj, bits)?;
        }
        for (index, b) in self.down_blocks.iter_mut().enumerate() {
            b.quantize_using(&format!("down_blocks.{index}"), bits, visit)?;
        }
        self.mid_resnet0.quantize_using("mid_block.resnets.0", bits, visit)?;
        self.mid_transformer.quantize_using("mid_block.attentions.0", bits, visit)?;
        self.mid_resnet1.quantize_using("mid_block.resnets.1", bits, visit)?;
        for (index, b) in self.up_blocks.iter_mut().enumerate() {
            b.quantize_using(&format!("up_blocks.{index}"), bits, visit)?;
        }
        Ok(())
    }

pub(crate) fn price_forward_with_effects(
        &self,
        x: &Tensor,
        encoder_x: &Tensor,
        text_emb: &Tensor,
        control: Option<&(Vec<Tensor>, Tensor)>,
        ip: Option<(&Tensor, f32)>,
        plan: crate::plan::SdxlForwardPlan<'_>,
        pending: crate::quantum::ControlMergeWork,
        out: &mut crate::quantum::Quanta,
        effects: &crate::quantum::EffectScope<'_>,
    ) -> Result<Dtype> {
        let batch = x.shape()[0];
        let dtype = x.dtype();
        let conv_in = effects.conv_metadata("conv_in", &self.conv_in);
        out.push((crate::quantum::UNET_STEM, stem_work_bytes(x, encoder_x, &conv_in,
            self.encoder_hid_proj.as_ref(), 0, crate::quantum::backend())
            .saturating_add(stem_projection_scratch(&self.time_embedding,
                &self.add_embedding, self.encoder_hid_proj.as_ref(), plan.conditioning_batch(batch),
                dtype, text_emb.dtype(), encoder_x, effects)?)));
        let projected = match &self.encoder_hid_proj {
            Some(proj) => {
                let e = encoder_x.shape();
                Some(tensor(
                    &effects.linear_metadata("encoder_hid_proj", proj)?.output_shape(&[e[0], e[1], e[2]])?,
                    effects.linear_metadata("encoder_hid_proj", proj)?.output_dtype(encoder_x.dtype()),
                )?)
            }
            None => None,
        };
        let encoder_x = projected.as_ref().unwrap_or(encoder_x);
        let temb = tensor(
            &[
                plan.conditioning_batch(batch),
                self.time_embedding.out_features_with_effects(&effects.child("time_embedding"))?,
            ],
            text_time_temb_dtype_with_effects(
                &self.time_embedding,
                &self.add_embedding,
                text_emb.dtype(),
                dtype,
                effects,
            )?,
        )?;
        let mut x = tensor(
            &conv_in.output_shape(x.shape(), 1, 1)?, conv_in.output_dtype(dtype),
        )?;
        let mut residuals: Vec<Tensor> = vec![x.clone()];
        for (index, block) in self.down_blocks.iter().enumerate() {
            let (o, res) = block.price_forward_with_effects(&x, encoder_x, &temb, None, ip, plan, out, 0, None, &effects.child(&format!("down_blocks.{index}")))?;
            x = o;
            residuals.extend(res);
        }
        if let Some((down, _)) = control {
            let scratch = residuals
                .iter()
                .zip(down)
                .map(|(x, c)| q::control_injection_bytes_metadata(x, c))
                .sum::<u64>()
                .saturating_add(pending.down);
            out.push((crate::quantum::CONTROL_DOWN_INJECTION, scratch));
            for (residual, c) in residuals.iter_mut().zip(down) {
                *residual = tensor(
                    residual.shape(),
                    Dtype::from_promoting_types(residual.dtype(), c.dtype()),
                )?;
            }
        }
        for (index, (resnet, attention)) in [
            (&self.mid_resnet0, Some(&self.mid_transformer)),
            (&self.mid_resnet1, None),
        ].into_iter().enumerate() {
            out.push((
                crate::quantum::UNET_RESNET,
                resnet.quantum_scratch_with_effects(&x, Some(&temb), &effects.child(&format!("mid_block.resnets.{index}")))?,
            ));
            x = tensor(
                &resnet.output_shape_with_effects(x.shape(), None, &effects.child(&format!("mid_block.resnets.{index}")))?,
                resnet.output_dtype_with_effects(x.dtype(), Some(temb.dtype()), &effects.child(&format!("mid_block.resnets.{index}")))?,
            )?;
            if let Some(attention) = attention {
                x = attention.price_forward_with_effects(&x, encoder_x, ip, plan, out, &effects.child("mid_block.attentions.0"))?;
            }
        }
        if let Some((_, mid)) = control {
            out.push((
                crate::quantum::CONTROL_MID_INJECTION,
                q::control_injection_bytes_metadata(&x, mid).saturating_add(pending.mid),
            ));
            x = tensor(
                x.shape(),
                Dtype::from_promoting_types(x.dtype(), mid.dtype()),
            )?;
        }
        for (index, block) in self.up_blocks.iter().enumerate() {
            let (o, _) =
                block.price_forward_with_effects(&x, encoder_x, &temb, Some(&mut residuals), ip, plan, out, 0, None, &effects.child(&format!("up_blocks.{index}")))?;
            x = o;
        }
        let conv_out = effects.conv_metadata("conv_out", &self.conv_out);
        out.push((crate::quantum::UNET_CONV_OUT, crate::quantum::norm_conv_bytes(
            &conv_out.call(x.shape(), element_bytes(&x), 1, 1), crate::quantum::backend())));
        // Validate the final biased convolution's actual broadcast geometry, not its old base shape.
        conv_out.output_shape(x.shape(), 1, 1)?;
        Ok(conv_out.output_dtype(promote(x.dtype(),
            &[&self.conv_norm_out_w, &self.conv_norm_out_b])))
    }

pub fn install_ip_adapter(&mut self, pairs: Vec<(Tensor, Tensor)>) -> Result<()> {
        let expected = pairs.len();
        let mut it = pairs.into_iter();
        for b in &mut self.down_blocks {
            b.install_ip(&mut it)?;
        }
        for b in &mut self.up_blocks {
            b.install_ip(&mut it)?;
        }
        self.mid_transformer.install_ip(&mut it)?;
        let leftover = it.count();
        if leftover != 0 {
            return Err(mlx_gen::Error::Msg(format!(
                "ip_adapter: {leftover} of {expected} K/V pairs unused (cross-attn count mismatch)"
            )));
        }
        Ok(())
    }

}

impl ConvFacts {
fn load(w: &HeaderWeights, prefix: &str, stride: i32, pad: i32) -> Result<Self> {
        Ok(Self {
            weight: nchw_to_nhwc(w.require(&format!("{prefix}.weight"))?)?,
            bias: w.require(&format!("{prefix}.bias"))?.clone(),
            stride,
            pad,
        })
    }

fn call(&self, x: &Tensor) -> crate::quantum::ConvCall {
        crate::quantum::ConvCall::from_metadata(
            x.shape(),
            element_bytes(x),
            &self.weight.shape, element_bytes(&self.weight),
            self.stride,
            self.pad,
        )
    }

fn price_output(&self, x: &Tensor) -> Result<Tensor> {
        let s = x.shape();
        let w = self.weight.shape();
        let out = |n: i32, k: i32| (n + 2 * self.pad - k) / self.stride.max(1) + 1;
        tensor(
            &[s[0], out(s[1], w[1]), out(s[2], w[2]), w[0]],
            promote(x.dtype(), &[&self.weight, &self.bias]),
        )
    }

}

impl CondFacts {
fn load(w: &HeaderWeights) -> Result<Self> {
        let p = "controlnet_cond_embedding";
        let conv_in = ConvFacts::load(w, &format!("{p}.conv_in"), 1, 1)?;
        // blocks 0..6: even = stride 1, odd = stride 2 (the down-projection convs).
        let blocks = (0..6)
            .map(|i| {
                let stride = if i % 2 == 0 { 1 } else { 2 };
                ConvFacts::load(w, &format!("{p}.blocks.{i}"), stride, 1)
            })
            .collect::<Result<Vec<_>>>()?;
        let conv_out = ConvFacts::load(w, &format!("{p}.conv_out"), 1, 1)?;
        Ok(Self {
            conv_in,
            blocks,
            conv_out,
        })
    }

fn price_forward(&self, control: &Tensor, out: &mut crate::quantum::Quanta) -> Result<Tensor> {
        let mut e = control.clone();
        for (layer, silu) in std::iter::once((&self.conv_in, true))
            .chain(self.blocks.iter().map(|b| (b, true)))
            .chain(std::iter::once((&self.conv_out, false)))
        {
            out.push((
                crate::quantum::CONTROL_COND_CONV,
                crate::quantum::single_conv_bytes(
                    &layer.call(&e),
                    1, // Same live-input term as the conditioning-embedding guard.
                    u64::from(silu),
                    crate::quantum::backend(),
                ),
            ));
            e = layer.price_output(&e)?;
        }
        Ok(e)
    }

}

impl ControlFacts {
pub fn from_weights(w: &HeaderWeights, cfg: &UNetConfig) -> Result<Self> {
        let n = cfg.num_blocks();
        let boc = &cfg.block_out_channels;

        let mut down_blocks = Vec::with_capacity(n);
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            down_blocks.push(BlockFacts::from_weights(
                w,
                &super::adapter_block_spec(cfg, i, false, &format!("down_blocks.{i}")),
            )?);
        }

        let mid_resnet0 = ResnetFacts::from_weights(w, "mid_block.resnets.0")?;
        let mid_transformer = TransformerFacts::from_weights(
            w,
            "mid_block.attentions.0",
            *boc.last().unwrap(),
            *cfg.num_attention_heads.last().unwrap(),
            *cfg.transformer_layers_per_block.last().unwrap(),
        )?;
        let mid_resnet1 = ResnetFacts::from_weights(w, "mid_block.resnets.1")?;

        // Nine down zero-convs (controlnet_down_blocks.{0..8}) — 1×1, stride 1, no pad.
        let down_zero = (0..9)
            .map(|i| ConvFacts::load(w, &format!("controlnet_down_blocks.{i}"), 1, 0))
            .collect::<Result<Vec<_>>>()?;
        let mid_zero = ConvFacts::load(w, "controlnet_mid_block", 1, 0)?;

        Ok(Self {
            conv_in: ConvFacts::load(w, "conv_in", 1, 1)?,
            _timesteps: SinusoidFacts::timestep(boc[0])?,
            time_embedding: TimeFacts::from_weights(w, "time_embedding")?,
            _add_time_proj: SinusoidFacts::timestep(
                cfg.addition_time_embed_dim.unwrap_or(256),
            )?,
            add_embedding: TimeFacts::from_weights(w, "add_embedding")?,
            down_blocks,
            mid_resnet0,
            mid_transformer,
            mid_resnet1,
            cond_embedding: CondFacts::load(w)?,
            down_zero,
            mid_zero,
            // Kolors `encoder_hid_proj` (4096→2048). Auto-detected: absent for SDXL → `None`.
            // Packed-detect (sc-8746): quantized in [`Self::quantize`], keyed on the dense `.weight`
            // OR the packed `.scales` (a Q4/Q8 snapshot has no dense weight beyond the u32 codes).
            encoder_hid_proj: if w.get("encoder_hid_proj.weight").is_some()
                || w.get("encoder_hid_proj.scales").is_some()
            {
                Some(header_linear(
                    w,
                    "encoder_hid_proj",
                    w.get("encoder_hid_proj.bias").is_some(),
                )?)
            } else {
                None
            },
        })
    }

pub(crate) fn price_embed_cond(
        &self,
        control: &Tensor,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        self.cond_embedding.price_forward(control, out)
    }

pub(crate) fn price_forward_with_effects(
        &self,
        x: &Tensor,
        cond_embed: &Tensor,
        encoder_x: &Tensor,
        text_emb: &Tensor,
        plan: crate::plan::SdxlForwardPlan<'_>,
        out: &mut crate::quantum::Quanta,
        mut owners: Option<&mut Vec<u64>>,
        effects: &crate::quantum::EffectScope<'_>,
    ) -> Result<(Vec<Tensor>, Tensor)> {
        let plan =
            crate::plan::SdxlForwardPlan::with_attention(plan.attention).with_cfg_row(plan.cfg_row);
        let batch = x.shape()[0];
        let dtype = x.dtype();
        if let Some(owners) = owners.as_deref_mut() {
            owners.push(0);
        }
        out.push((
            crate::quantum::CONTROL_STEM,
            self.stem_scratch(x, encoder_x, cond_embed).saturating_add(
                stem_projection_scratch(&self.time_embedding,
                    &self.add_embedding, self.encoder_hid_proj.as_ref(), plan.conditioning_batch(batch),
                    dtype, text_emb.dtype(), encoder_x, effects)?),
        ));
        let projected = match &self.encoder_hid_proj {
            Some(proj) => {
                let e = encoder_x.shape();
                Some(tensor(
                    &[e[0], e[1], effects.linear_metadata("encoder_hid_proj", proj)?.output_width() as i32],
                    effects.linear_metadata("encoder_hid_proj", proj)?.output_dtype(encoder_x.dtype()),
                )?)
            }
            None => None,
        };
        let encoder_x = projected.as_ref().unwrap_or(encoder_x);
        let temb = tensor(
            &[
                plan.conditioning_batch(batch),
                self.time_embedding.out_features_with_effects(&effects.child("time_embedding"))?,
            ],
            text_time_temb_dtype_with_effects(
                &self.time_embedding,
                &self.add_embedding,
                text_emb.dtype(),
                dtype,
                effects,
            )?,
        )?;
        let stem = self.conv_in.price_output(x)?;
        let mut x = tensor(
            stem.shape(),
            Dtype::from_promoting_types(stem.dtype(), cond_embed.dtype()),
        )?;
        let fixed =
            tensor_bytes(&temb) + projected.as_ref().map_or(0, tensor_bytes);
        let mut residuals: Vec<Tensor> = vec![x.clone()];
        for (index, block) in self.down_blocks.iter().enumerate() {
            let held = fixed + residuals.iter().map(tensor_bytes).sum::<u64>();
            let (o, res) = block.price_forward_with_effects(
                &x,
                encoder_x,
                &temb,
                None,
                None,
                plan,
                out,
                held,
                owners.as_deref_mut(),
                &effects.child(&format!("down_blocks.{index}")),
            )?;
            x = o;
            residuals.extend(res);
        }
        let mut mid_distinct = false;
        for (index, (resnet, attention)) in [
            (&self.mid_resnet0, Some(&self.mid_transformer)),
            (&self.mid_resnet1, None),
        ].into_iter().enumerate() {
            if let Some(owners) = owners.as_deref_mut() {
                owners.push(
                    fixed
                        + residuals.iter().map(tensor_bytes).sum::<u64>()
                        + if mid_distinct {
                            tensor_bytes(&x)
                        } else {
                            0
                        },
                );
            }
            out.push((
                crate::quantum::UNET_RESNET,
                resnet.quantum_scratch_with_effects(&x, Some(&temb), &effects.child(&format!("mid_block.resnets.{index}")))?,
            ));
            let s = x.shape();
            x = tensor(
                &[s[0], s[1], s[2], resnet.conv_weights().1.shape()[0]],
                resnet.output_dtype_with_effects(x.dtype(), Some(temb.dtype()), &effects.child(&format!("mid_block.resnets.{index}")))?,
            )?;
            mid_distinct = true;
            if let Some(attention) = attention {
                let first = out.len();
                let held = fixed
                    + residuals.iter().map(tensor_bytes).sum::<u64>()
                    + tensor_bytes(&x);
                x = attention.price_forward_with_effects(&x, encoder_x, None, plan, out, &effects.child("mid_block.attentions.0"))?;
                if let Some(owners) = owners.as_deref_mut() {
                    owners.extend(std::iter::repeat_n(held, out.len() - first));
                }
            }
        }
        check_residual_zero_pairing(residuals.len(), self.down_zero.len())?;
        let scaled = |z: &ConvFacts, r: &Tensor, out: &mut crate::quantum::Quanta| {
            out.push((
                crate::quantum::CONTROL_ZERO_CONV,
                crate::quantum::single_conv_bytes(&z.call(r), 0, 0, crate::quantum::backend()),
            ));
            let o = z.price_output(r)?;
            tensor(o.shape(), Dtype::from_promoting_types(o.dtype(), dtype))
        };
        let retained = fixed
            + residuals.iter().map(tensor_bytes).sum::<u64>()
            + tensor_bytes(&x);
        let mut down = Vec::with_capacity(residuals.len());
        for (r, z) in residuals.iter().zip(&self.down_zero) {
            if let Some(owners) = owners.as_deref_mut() {
                owners.push(retained + down.iter().map(tensor_bytes).sum::<u64>());
            }
            down.push(scaled(z, r, out)?);
        }
        if let Some(owners) = owners.as_deref_mut() {
            owners.push(retained + down.iter().map(tensor_bytes).sum::<u64>());
        }
        let mid = scaled(&self.mid_zero, &x, out)?;
        Ok((down, mid))
    }

pub(crate) fn quantize_using(&mut self, bits: i32, visit: &mut MetaVisit<'_>) -> Result<()> {
        // ControlFacts's encoder_hid_proj is LAST, unlike the UNet walk. Keep the actual order.
        self.time_embedding.quantize_using("time_embedding", bits, visit)?;
        self.add_embedding.quantize_using("add_embedding", bits, visit)?;
        for (index, b) in self.down_blocks.iter_mut().enumerate() {
            b.quantize_using(&format!("down_blocks.{index}"), bits, visit)?;
        }
        self.mid_resnet0.quantize_using("mid_block.resnets.0", bits, visit)?;
        self.mid_transformer.quantize_using("mid_block.attentions.0", bits, visit)?;
        self.mid_resnet1.quantize_using("mid_block.resnets.1", bits, visit)?;
        if let Some(proj) = &mut self.encoder_hid_proj {
            visit("encoder_hid_proj", proj, bits)?;
        }
        Ok(())
    }

fn stem_scratch(&self, x: &Tensor, encoder_x: &Tensor, cond_embed: &Tensor) -> u64 {
        stem_bytes(
            x,
            encoder_x,
            &self.conv_in.weight,
            self.encoder_hid_proj.as_ref(),
            0,
            crate::quantum::backend(),
        )
        // The selected conditioning row and the promoted conv+conditioning sum remain live
        // through this quantum. Unlike the plain stem these are not donated convolution outputs.
        .saturating_add(
            2 * mlx_gen::ios_quantum::tensor_bytes(
                crate::quantum::dtype_bytes(Dtype::from_promoting_types(
                    x.dtype(),
                    cond_embed.dtype(),
                )),
                &[
                    x.shape()[0] as u64,
                    cond_embed.shape()[1] as u64,
                    cond_embed.shape()[2] as u64,
                    cond_embed.shape()[3] as u64,
                ],
            ),
        )
    }

}

pub(crate) fn stem_work_bytes(
    x: &Tensor, encoder_x: &Tensor, conv_in: &crate::quantum::ConvWork,
    encoder_hid_proj: Option<&q::LinearWork>, extra_out: u64, backend: crate::quantum::Backend,
) -> u64 {
    let call = conv_in.call(x.shape(), element_bytes(x), 1, 1);
    let conv = crate::quantum::single_conv_bytes(&call, 0, extra_out, backend);
    let e = encoder_x.shape();
    let context = encoder_hid_proj.map_or(0, |proj| {
        (e[0].max(0) as u64)
            .saturating_mul(e[1].max(0) as u64)
            .saturating_mul(linear_out(proj))
            .saturating_mul(element_bytes(encoder_x))
    });
    let temb_width = 4 * conv_in.weight.shape[0].max(0) as u64;
    let temb =
        mlx_gen::ios_quantum::elementwise_bytes((e[0].max(0) as u64).saturating_mul(temb_width), 8);
    let upstream = mlx_gen::ios_quantum::elementwise_bytes(x.size() as u64, 3);
    conv.saturating_add(context)
        .saturating_add(temb)
        .saturating_add(upstream)
}

pub(crate) fn sampler_scratch(x: &Tensor, work: &crate::quantum::ConvWork, up: bool) -> u64 {
    let mut shape = x.shape().to_vec();
    if up { shape[1] *= 2; shape[2] *= 2; }
    let act = element_bytes(x);
    let call = work.call(&shape, act, if up { 1 } else { 2 }, 1);
    crate::quantum::single_conv_bytes(&call, u64::from(up), 0, crate::quantum::backend())
        .saturating_add(work.bias_bytes(x.dtype(), call.out_bytes() / act))
}

pub(crate) fn text_time_temb_dtype_with_effects(
    time_embedding: &TimeFacts,
    add_embedding: &TimeFacts,
    text_emb_dtype: Dtype,
    dtype: Dtype,
    effects: &crate::quantum::EffectScope<'_>,
) -> Result<Dtype> {
    Ok(Dtype::from_promoting_types(
        time_embedding.output_dtype_with_effects(dtype, &effects.child("time_embedding"))?,
        add_embedding.output_dtype_with_effects(Dtype::from_promoting_types(text_emb_dtype, dtype),
            &effects.child("add_embedding"))?,
    ))
}

pub(crate) fn stem_projection_scratch(
    time: &TimeFacts, add: &TimeFacts, encoder: Option<&q::LinearWork>,
    batch: i32, dtype: Dtype, text: Dtype, memory: &Tensor,
    effects: &crate::quantum::EffectScope<'_>,
) -> Result<u64> {
    let mut bytes = time.projection_scratch(batch, dtype, &effects.child("time_embedding"))?
        .saturating_add(add.projection_scratch(batch, Dtype::from_promoting_types(text, dtype),
            &effects.child("add_embedding"))?);
    if let Some(encoder) = encoder {
        let work = effects.linear_metadata("encoder_hid_proj", encoder)?;
        bytes = bytes.saturating_add(work.operand_bytes(memory.shape(), memory.dtype()))
            .saturating_add(mlx_gen::ios_quantum::tensor_bytes(
                crate::quantum::dtype_bytes(work.output_dtype(memory.dtype())),
                &[memory.shape()[0].max(0) as u64, memory.shape()[1].max(0) as u64, work.output_width()]));
    }
    Ok(bytes)
}

fn check_residual_zero_pairing(residuals: usize, zero_convs: usize) -> Result<()> {
    if residuals != zero_convs {
        return Err(mlx_gen::Error::Msg(format!(
            "sdxl controlnet: {residuals} down residuals but {zero_convs} zero-conv heads — the \
             control branch geometry does not match the loaded down_zero convs"
        )));
    }
    Ok(())
}

impl ResnetFacts {
pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut q::LinearWork> {
        match path {
            ["time_emb_proj"] => self.time_emb_proj.as_mut(),
            // conv_shortcut is a 1×1 conv stored as a Linear; a conv LoRA merges into it as a
            // reshaped 2-D delta (sc-2919, routed by the SDXL adapter's conv path).
            ["conv_shortcut"] => self.shortcut.as_mut(),
            _ => None,
        }
    }
pub(crate) fn adaptable_conv_mut(&mut self, path: &[&str]) -> Option<&mut q::ConvWork> {
        match path {
            ["conv1"] => Some(&mut self.conv1),
            ["conv2"] => Some(&mut self.conv2),
            _ => None,
        }
    }
}

impl AttentionFacts {
    pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut q::LinearWork> {
        match path {
            ["to_q"] => Some(&mut self.q),
            ["to_k"] => Some(&mut self.k),
            ["to_v"] => Some(&mut self.v),
            ["to_out", "0"] => Some(&mut self.out),
            _ => None,
        }
    }
}

impl TransformerBlockFacts {
pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut q::LinearWork> {
        match path {
            ["attn1", rest @ ..] => self.attn1.adaptable_mut(rest),
            ["attn2", rest @ ..] => self.attn2.adaptable_mut(rest),
            // GEGLU FF (sc-2671 complete coverage). The diffusers `ff.net.0.proj` is row-split into
            // `linear1` (value) + `linear2` (gate) and `ff.net.2` is `linear3`; the SDXL adapter
            // merge translates those diffusers FF keys into these internal `ff.linearN` names (and
            // row-splits a `ff.net.0.proj` delta across linear1/linear2). Unreachable under the
            // vendored coverage (the merge gates FF keys out there).
            ["ff", "linear1"] => Some(&mut self.linear1),
            ["ff", "linear2"] => Some(&mut self.linear2),
            ["ff", "linear3"] => Some(&mut self.linear3),
            _ => None,
        }
    }
}

impl TransformerFacts {
pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut q::LinearWork> {
        match path {
            ["proj_in"] => Some(&mut self.proj_in),
            ["proj_out"] => Some(&mut self.proj_out),
            ["transformer_blocks", k, rest @ ..] => self
                .blocks
                .get_mut(k.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }
}

impl BlockFacts {
pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut q::LinearWork> {
        match path {
            ["resnets", j, rest @ ..] => self
                .resnets
                .get_mut(j.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["attentions", j, rest @ ..] => self
                .attentions
                .as_mut()?
                .get_mut(j.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }
pub(crate) fn adaptable_conv_mut(&mut self, path: &[&str]) -> Option<&mut q::ConvWork> {
        match path {
            ["resnets", j, rest @ ..] => self
                .resnets
                .get_mut(j.parse::<usize>().ok()?)?
                .adaptable_conv_mut(rest),
            ["downsamplers", "0", "conv"] => self.downsample.as_mut(),
            ["upsamplers", "0", "conv"] => self.upsample.as_mut(),
            _ => None,
        }
    }
}

impl UNetFacts {
pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut q::LinearWork> {
        match path {
            ["down_blocks", i, rest @ ..] => self
                .down_blocks
                .get_mut(i.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["up_blocks", k, rest @ ..] => self
                .up_blocks
                .get_mut(k.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            // mid_block (sc-2671 complete coverage). Routable here, but the vendored coverage path
            // gates mid_block/ff keys out so the faithful 515-module merge is unaffected; only the
            // opt-in complete coverage actually merges into these.
            ["mid_block", "attentions", "0", rest @ ..] => self.mid_transformer.adaptable_mut(rest),
            ["mid_block", "resnets", "0", rest @ ..] => self.mid_resnet0.adaptable_mut(rest),
            ["mid_block", "resnets", "1", rest @ ..] => self.mid_resnet1.adaptable_mut(rest),
            _ => None,
        }
    }
pub(crate) fn adaptable_conv_mut(&mut self, path: &[&str]) -> Option<&mut q::ConvWork> {
        match path {
            ["conv_in"] => Some(&mut self.conv_in),
            ["conv_out"] => Some(&mut self.conv_out),
            ["down_blocks", i, rest @ ..] => self
                .down_blocks
                .get_mut(i.parse::<usize>().ok()?)?
                .adaptable_conv_mut(rest),
            ["up_blocks", k, rest @ ..] => self
                .up_blocks
                .get_mut(k.parse::<usize>().ok()?)?
                .adaptable_conv_mut(rest),
            ["mid_block", "resnets", "0", rest @ ..] => self.mid_resnet0.adaptable_conv_mut(rest),
            ["mid_block", "resnets", "1", rest @ ..] => self.mid_resnet1.adaptable_conv_mut(rest),
            _ => None,
        }
    }
}

impl UNetFacts {
    pub(crate) fn from_headers(headers: &[SafetensorsTensorHeader], cfg: &UNetConfig) -> Result<Self> {
        Self::from_weights(&HeaderWeights::new(headers)?, cfg)
    }
    pub(crate) fn arm_block_streams(&mut self, headers: &[SafetensorsTensorHeader], dtype: Dtype,
        quant_bits: Option<i32>) -> Result<()> {
        for (index, block) in self.down_blocks.iter_mut().enumerate() {
            block.arm_streams(headers, &format!("down_blocks.{index}"), dtype, quant_bits)?;
        }
        self.mid_transformer.arm_stream(headers, "mid_block.attentions.0", dtype, quant_bits)?;
        for (index, block) in self.up_blocks.iter_mut().enumerate() {
            block.arm_streams(headers, &format!("up_blocks.{index}"), dtype, quant_bits)?;
        }
        Ok(())
    }
}
impl ControlFacts {
    pub(crate) fn from_headers(headers: &[SafetensorsTensorHeader], cfg: &UNetConfig) -> Result<Self> {
        Self::from_weights(&HeaderWeights::new(headers)?, cfg)
    }
}
impl BlockFacts {
    fn arm_streams(&mut self, headers: &[SafetensorsTensorHeader], prefix: &str,
        dtype: Dtype, quant_bits: Option<i32>) -> Result<()> {
        if let Some(attentions) = &mut self.attentions {
            for (index, attention) in attentions.iter_mut().enumerate() {
                attention.arm_stream(headers, &format!("{prefix}.attentions.{index}"), dtype, quant_bits)?;
            }
        }
        Ok(())
    }
}
impl TransformerFacts {
    fn arm_stream(&mut self, headers: &[SafetensorsTensorHeader], prefix: &str,
        dtype: Dtype, quant_bits: Option<i32>) -> Result<()> {
        self.block_stream = Some(StreamFacts { prefix: prefix.to_owned(), headers: headers.to_vec(),
            n_blocks: self.blocks.len(), dtype, quant_bits, bytes: std::sync::OnceLock::new() });
        Ok(())
    }
}
pub(crate) fn ip_pairs_from_headers(headers: &[SafetensorsTensorHeader]) -> Result<Vec<(Tensor, Tensor)>> {
    let weights = HeaderWeights::new(headers)?;
    crate::ip_adapter::ip_kv_indices(weights.tensors.keys().map(String::as_str))?.into_iter().map(|index| {
        let key = weights.require(&format!("ip_adapter.{index}.to_k_ip.weight"))?.clone();
        let value = weights.require(&format!("ip_adapter.{index}.to_v_ip.weight"))?.clone();
        Ok((key, value))
    }).collect()
}

fn stem_bytes(x: &Tensor, encoder_x: &Tensor, conv_in: &Tensor,
    encoder_hid_proj: Option<&q::LinearWork>, extra_out: u64, backend: q::Backend) -> u64 {
    stem_work_bytes(x, encoder_x, &q::ConvWork { weight: conv_in.clone(), bias: None },
        encoder_hid_proj, extra_out, backend)
}

#[cfg(test)]
fn trace_facts(label: &str, tensor: &Tensor) { eprintln!("METADATA {label}: shape={:?} dtype={:?}", tensor.shape, tensor.dtype); }

pub(crate) fn add(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    tensor(&crate::adapters::broadcast_shape(&left.shape, &right.shape)?,
        Dtype::from_promoting_types(left.dtype, right.dtype))
}

impl ResnetFacts {
    pub(crate) fn quantum_scratch(&self, x: &Tensor, temb: Option<&Tensor>) -> u64 {
        self.quantum_scratch_with_effects(x, temb, &q::EffectScope::native())
            .expect("constructor metadata projection facts are infallible")
    }
    pub(crate) fn output_dtype(&self, input: Dtype, temb: Option<Dtype>) -> Dtype {
        self.output_dtype_with_effects(input, temb, &q::EffectScope::native())
            .expect("constructor metadata projection facts are infallible")
    }
    pub(crate) fn price_tiled_vae(
        &self,
        x: &Tensor,
        tile_edge: i32,
        labels: crate::quantum::VaeLabels,
        out: &mut crate::quantum::Quanta,
    ) -> Result<Tensor> {
        let s = x.shape();
        let (cin, cout) = (self.conv1.weight.shape[self.conv1.weight.shape.len() - 1], self.conv1.weight.shape[0]);
        out.push((
            labels.tiled_stage,
            crate::quantum::tiled_stage_bytes(s, cin, cout, element_bytes(x), true),
        ));
        let tile = |input: &[i32], channels: i32, width: u64, weight: &Tensor| {
            let (th, tw) = crate::vae::max_conv_tile(input[1], input[2], tile_edge);
            crate::quantum::tile_bytes_from_metadata(
                &[input[0], th, tw, channels],
                crate::quantum::TileSource { full_width: input[2], normalized: true },
                width, &weight.shape, element_bytes(weight), false, crate::quantum::backend(),
            )
        };
        out.push((
            labels.tile,
            tile(
                s,
                s[3],
                element_bytes(x),
                self.conv1.weight(),
            ),
        ));
        let mut conv1 = vec![&self.norm1_w, &self.norm1_b, self.conv1.weight()];
        conv1.extend(self.conv1.bias());
        let y = promote(x.dtype(), &conv1);
        let mid = [s[0], s[1], s[2], self.conv2.weight().shape()[3]];
        out.push((
            labels.tile,
            tile(
                &mid,
                mid[3],
                crate::quantum::dtype_bytes(y),
                self.conv2.weight(),
            ),
        ));
        tensor(
            &[s[0], s[1], s[2], cout],
            self.output_dtype(x.dtype(), None),
        )
    }
}

#[cfg(test)]
mod attention_adapter_tests {
    use super::*;

    #[test]
    fn metadata_adapter_targets_match_native_paths_and_mutate_only_selected_linear() {
        let linear = |input| q::LinearWork::dense(tensor(&[64, input], Dtype::Float16).unwrap(), None);
        let base = AttentionFacts {
            q: linear(64), k: linear(128), v: linear(192), out: linear(256),
            num_heads: 1, head_dim: 64,
            to_k_ip: Some(linear(320)), to_v_ip: Some(linear(384)),
        };
        // Independent native leaf-to-field inventory. Pointer equality is ONLY Rust metadata
        // object identity, never an Array/NativeData backing or physical-allocation certificate.
        let targets: &[(&[&str], usize)] = &[
            (&["to_q"], 0), (&["to_k"], 1), (&["to_v"], 2), (&["to_out", "0"], 3),
        ];
        for &(path, selected) in targets {
            let mut trial = base.clone();
            let expected = match selected {
                0 => &mut trial.q, 1 => &mut trial.k, 2 => &mut trial.v, 3 => &mut trial.out,
                _ => unreachable!(),
            } as *mut q::LinearWork;
            let actual = trial.adaptable_mut(path).expect("native metadata adapter target") as *mut q::LinearWork;
            assert_eq!(actual, expected, "literal native target must return its actual mutable field: {path:?}");
            trial.adaptable_mut(path).unwrap().pack(8).unwrap();
            let states = [trial.q.bits, trial.k.bits, trial.v.bits, trial.out.bits,
                trial.to_k_ip.as_ref().unwrap().bits, trial.to_v_ip.as_ref().unwrap().bits];
            for (index, state) in states.into_iter().enumerate() {
                assert_eq!(state, if index == selected { Some(8) } else { None },
                    "selected-world pack must mutate one actual field, not a clone/alias/default: {path:?}/{index}");
            }
        }
        for path in [vec![], vec!["to_out"], vec!["to_out", "1"], vec!["to_q", "extra"],
                     vec!["to_k_ip"], vec!["to_v_ip"], vec!["unknown"]] {
            assert!(base.clone().adaptable_mut(&path).is_none(), "not a native adaptable leaf: {path:?}");
        }
    }
}
