//! A native candle port of the **HuBERT** base encoder (`transformers.HubertModel`, post-LN
//! variant: `feat_extract_norm = "group"`, `do_stable_layer_norm = false`) — xcodec's semantic
//! branch for the YuE ICL reference encoder (sc-19379).
//!
//! ```text
//!   wave [B, N] ─▶ feature encoder: conv(k10 s5) → GroupNorm(C groups) → GELU,
//!                                   6 × (conv(k3/k3/k3/k3/k2/k2, s2) → GELU)      [B, 512, T]
//!               ─▶ feature projection: LayerNorm(512) → Linear(512 → 768)         [B, T, 768]
//!               ─▶ encoder: + GELU(pos-conv(k128, groups 16, weight_norm dim 2)[..T])
//!                           → LayerNorm → 12 × post-LN transformer layer
//! ```
//!
//! [`Hubert::mean_hidden_states`] is what xcodec's `SoundStream.get_regress_target` consumes: the
//! element-wise mean of all 13 `output_hidden_states` (the encoder input after its LayerNorm, then
//! every layer's output). Dropout, LayerDrop and SpecAugment masking are training-only and absent.
//!
//! The architecture constants (conv strides, 12 heads, 16 positional-conv groups, `eps = 1e-5`) are
//! those of xcodec's `semantic_ckpts/hf_1_325000/config.json`; every width comes from the tensors
//! and is cross-checked, so a mismatched checkpoint is an error rather than a silent misread.

use std::collections::HashMap;

use candle_audio::candle_core::{Result, Tensor, D};
use candle_audio::neural_codec::fold_weight_norm_dim;
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module};

/// Feature-encoder conv strides (`config.conv_stride`); kernels are read from the weights and
/// checked against [`CONV_KERNELS`].
pub const CONV_STRIDES: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];
/// Feature-encoder conv kernels (`config.conv_kernel`).
pub const CONV_KERNELS: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
/// Attention heads (`config.num_attention_heads`).
pub const NUM_HEADS: usize = 12;
/// Positional-conv groups (`config.num_conv_pos_embedding_groups`).
pub const POS_CONV_GROUPS: usize = 16;
/// LayerNorm / GroupNorm epsilon (`config.layer_norm_eps`; GroupNorm's torch default is the same).
pub const EPS: f64 = 1e-5;
/// Rows of queries attended at once — bounds the `[heads, rows, T]` score block for long clips.
const QUERY_CHUNK: usize = 256;

struct Weights<'a> {
    map: &'a HashMap<String, Tensor>,
    prefix: &'a str,
}

impl Weights<'_> {
    fn get(&self, name: &str) -> Result<Tensor> {
        let full = format!("{}{name}", self.prefix);
        self.map.get(&full).cloned().ok_or_else(|| {
            candle_audio::candle_core::Error::Msg(format!("HuBERT: missing {full:?}"))
        })
    }

    fn linear(&self, name: &str) -> Result<Linear> {
        Ok(Linear::new(
            self.get(&format!("{name}.weight"))?,
            Some(self.get(&format!("{name}.bias"))?),
        ))
    }

    fn layer_norm(&self, name: &str) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.get(&format!("{name}.weight"))?,
            self.get(&format!("{name}.bias"))?,
            EPS,
        ))
    }
}

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ln: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
    final_ln: LayerNorm,
}

impl Layer {
    /// `HubertEncoderLayer.forward` (post-LN): `h = LN(h + attn(h)); h = LN(h + ff(h))`.
    fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let (b, t, c) = h.dims3()?;
        let hd = c / NUM_HEADS;
        let heads = |x: Tensor| -> Result<Tensor> {
            x.reshape((b, t, NUM_HEADS, hd))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = heads(self.q.forward(h)?)?;
        let k = heads(self.k.forward(h)?)?;
        let v = heads(self.v.forward(h)?)?;
        let kt = k.transpose(2, 3)?.contiguous()?;
        let scale = (hd as f64).powf(-0.5);
        let mut parts = Vec::with_capacity(t.div_ceil(QUERY_CHUNK));
        let mut start = 0;
        while start < t {
            let rows = QUERY_CHUNK.min(t - start);
            let qc = q.narrow(2, start, rows)?;
            let scores = (qc.matmul(&kt)? * scale)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            parts.push(probs.matmul(&v)?);
            start += rows;
        }
        let attn = Tensor::cat(&parts, 2)?
            .transpose(1, 2)?
            .reshape((b, t, c))?;
        let h = self.ln.forward(&(h + self.out.forward(&attn)?)?)?;
        let ff = self.ff_out.forward(&self.ff_in.forward(&h)?.gelu_erf()?)?;
        self.final_ln.forward(&(h + ff)?)
    }
}

/// The HuBERT encoder (inference only).
pub struct Hubert {
    convs: Vec<Conv1d>,
    group_norm: (Tensor, Tensor),
    proj_ln: LayerNorm,
    proj: Linear,
    pos_conv: Conv1d,
    pos_remove: usize,
    enc_ln: LayerNorm,
    layers: Vec<Layer>,
}

impl Hubert {
    /// Load from a tensor map whose HuBERT tensors sit under `prefix` (e.g. `"semantic_model."`
    /// inside the xcodec checkpoint). Layers are counted from the map.
    pub fn load(map: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let w = Weights { map, prefix };
        let mut convs = Vec::with_capacity(CONV_STRIDES.len());
        for (i, (&stride, &kernel)) in CONV_STRIDES.iter().zip(&CONV_KERNELS).enumerate() {
            let weight = w.get(&format!("feature_extractor.conv_layers.{i}.conv.weight"))?;
            let k = weight.dim(2)?;
            if k != kernel {
                candle_audio::candle_core::bail!("HuBERT: conv layer {i} kernel {k} != {kernel}");
            }
            convs.push(Conv1d::new(
                weight,
                None,
                Conv1dConfig {
                    stride,
                    ..Default::default()
                },
            ));
        }
        let group_norm = (
            w.get("feature_extractor.conv_layers.0.layer_norm.weight")?,
            w.get("feature_extractor.conv_layers.0.layer_norm.bias")?,
        );

        let v = w.get("encoder.pos_conv_embed.conv.weight_v")?;
        let g = w.get("encoder.pos_conv_embed.conv.weight_g")?;
        let pos_weight = fold_weight_norm_dim(&v, &g, 2)?;
        let (hidden, per_group, pos_k) = pos_weight.dims3()?;
        if per_group * POS_CONV_GROUPS != hidden {
            candle_audio::candle_core::bail!(
                "HuBERT: positional conv {:?} is not {POS_CONV_GROUPS}-grouped over {hidden}",
                pos_weight.dims()
            );
        }
        if !hidden.is_multiple_of(NUM_HEADS) {
            candle_audio::candle_core::bail!(
                "HuBERT: hidden {hidden} not divisible by {NUM_HEADS}"
            );
        }
        let pos_conv = Conv1d::new(
            pos_weight,
            Some(w.get("encoder.pos_conv_embed.conv.bias")?),
            Conv1dConfig {
                padding: pos_k / 2,
                groups: POS_CONV_GROUPS,
                ..Default::default()
            },
        );

        let mut layers = Vec::new();
        while map.contains_key(&format!(
            "{prefix}encoder.layers.{}.attention.q_proj.weight",
            layers.len()
        )) {
            let l = format!("encoder.layers.{}", layers.len());
            layers.push(Layer {
                q: w.linear(&format!("{l}.attention.q_proj"))?,
                k: w.linear(&format!("{l}.attention.k_proj"))?,
                v: w.linear(&format!("{l}.attention.v_proj"))?,
                out: w.linear(&format!("{l}.attention.out_proj"))?,
                ln: w.layer_norm(&format!("{l}.layer_norm"))?,
                ff_in: w.linear(&format!("{l}.feed_forward.intermediate_dense"))?,
                ff_out: w.linear(&format!("{l}.feed_forward.output_dense"))?,
                final_ln: w.layer_norm(&format!("{l}.final_layer_norm"))?,
            });
        }
        if layers.is_empty() {
            candle_audio::candle_core::bail!("HuBERT: no encoder layers under {prefix:?}");
        }
        let proj = w.linear("feature_projection.projection")?;
        if proj.weight().dim(0)? != hidden {
            candle_audio::candle_core::bail!(
                "HuBERT: projection emits {} but the encoder is {hidden} wide",
                proj.weight().dim(0)?
            );
        }
        Ok(Self {
            convs,
            group_norm,
            proj_ln: w.layer_norm("feature_projection.layer_norm")?,
            proj,
            pos_conv,
            pos_remove: usize::from(pos_k.is_multiple_of(2)),
            enc_ln: w.layer_norm("encoder.layer_norm")?,
            layers,
        })
    }

    /// Frames the feature encoder yields for `samples` input samples.
    pub fn frames_for(samples: usize) -> usize {
        CONV_STRIDES
            .iter()
            .zip(&CONV_KERNELS)
            .fold(
                samples,
                |n, (&s, &k)| if n < k { 0 } else { (n - k) / s + 1 },
            )
    }

    /// Samples one feature frame advances (the product of [`CONV_STRIDES`]).
    pub const HOP: usize = 320;
    /// Samples one feature frame reads (the stack's receptive field).
    pub const RECEPTIVE_FIELD: usize = 400;

    /// The feature encoder `[1, N]` → `[1, C, T]`, evaluated `chunk_frames` output frames at a time
    /// so the sample-rate activations never exceed one chunk (the stack has no padding, so frame
    /// `f` reads exactly samples `f·320 .. f·320 + 400` and chunks tile the output with no seam).
    /// The first layer's GroupNorm normalizes each channel over the **whole** clip, so its
    /// statistics come from two chunked passes over the first conv (mean, then the centered
    /// second moment, accumulated in float64) before the chunks are normalized.
    pub fn features(&self, wave: &Tensor, chunk_frames: usize) -> Result<Tensor> {
        let n = wave.dim(D::Minus1)?;
        let frames = Self::frames_for(n);
        if frames == 0 {
            candle_audio::candle_core::bail!("HuBERT: {n} samples is shorter than one frame");
        }
        let chunk = chunk_frames.max(1);
        let (k0, s0) = (CONV_KERNELS[0], CONV_STRIDES[0]);
        let t0 = (n - k0) / s0 + 1;
        let per_chunk = chunk.saturating_mul(Self::HOP / s0);
        let conv0 = |u0: usize, u1: usize| -> Result<Tensor> {
            let len = (u1 - 1) * s0 + k0 - u0 * s0;
            self.convs[0].forward(&wave.narrow(D::Minus1, u0 * s0, len)?.unsqueeze(1)?)
        };
        let channel_sums =
            |y: &Tensor| -> Result<Vec<f32>> { y.sum(D::Minus1)?.flatten_all()?.to_vec1() };
        let mut mean = Vec::new();
        for u0 in (0..t0).step_by(per_chunk) {
            let sums = channel_sums(&conv0(u0, u0.saturating_add(per_chunk).min(t0))?)?;
            mean.resize(sums.len(), 0f64);
            mean.iter_mut()
                .zip(sums)
                .for_each(|(m, v)| *m += f64::from(v));
        }
        mean.iter_mut().for_each(|m| *m /= t0 as f64);
        let c = mean.len();
        let mean_t = Tensor::from_vec(
            mean.iter().map(|&m| m as f32).collect::<Vec<_>>(),
            (1, c, 1),
            wave.device(),
        )?;
        let mut var = vec![0f64; c];
        for u0 in (0..t0).step_by(per_chunk) {
            let y = conv0(u0, u0.saturating_add(per_chunk).min(t0))?.broadcast_sub(&mean_t)?;
            var.iter_mut()
                .zip(channel_sums(&y.sqr()?)?)
                .for_each(|(v, s)| *v += f64::from(s));
        }
        let inv_std: Vec<f32> = var
            .iter()
            .map(|&v| (1.0 / (v / t0 as f64 + EPS).sqrt()) as f32)
            .collect();
        let inv_std = Tensor::from_vec(inv_std, (1, c, 1), wave.device())?;
        let (gw, gb) = (
            self.group_norm.0.reshape((1, c, 1))?,
            self.group_norm.1.reshape((1, c, 1))?,
        );

        let mut parts = Vec::with_capacity(frames.div_ceil(chunk));
        for f0 in (0..frames).step_by(chunk) {
            let f1 = f0.saturating_add(chunk).min(frames);
            let len = (f1 - 1) * Self::HOP + Self::RECEPTIVE_FIELD - f0 * Self::HOP;
            let mut x = self.convs[0]
                .forward(&wave.narrow(D::Minus1, f0 * Self::HOP, len)?.unsqueeze(1)?)?;
            x = x
                .broadcast_sub(&mean_t)?
                .broadcast_mul(&inv_std)?
                .broadcast_mul(&gw)?
                .broadcast_add(&gb)?
                .gelu_erf()?;
            for conv in &self.convs[1..] {
                x = conv.forward(&x)?.gelu_erf()?;
            }
            parts.push(x);
        }
        Tensor::cat(&parts, D::Minus1)
    }

    /// The mean of all `1 + layers` hidden states for a `[1, N]` waveform → `[1, T, hidden]`, the
    /// feature encoder run `chunk_frames` frames at a time ([`Self::features`]). `cancel` is polled
    /// before each transformer layer; a tripped poll returns `None`.
    pub fn mean_hidden_states(
        &self,
        wave: &Tensor,
        chunk_frames: usize,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Option<Tensor>> {
        let x = self
            .features(wave, chunk_frames)?
            .transpose(1, 2)?
            .contiguous()?;
        let h = self.proj.forward(&self.proj_ln.forward(&x)?)?;

        let pos = self.pos_conv.forward(&h.transpose(1, 2)?.contiguous()?)?;
        let t = pos.dim(D::Minus1)? - self.pos_remove;
        let pos = pos.narrow(D::Minus1, 0, t)?.gelu_erf()?.transpose(1, 2)?;
        let mut h = self.enc_ln.forward(&(h + pos)?)?;

        let mut sum = h.clone();
        for layer in &self.layers {
            if cancel() {
                return Ok(None);
            }
            h = layer.forward(&h)?;
            sum = (sum + &h)?;
        }
        Ok(Some((sum / (self.layers.len() + 1) as f64)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_encoder_frames_follow_the_conv_stack() {
        // 1 s at 16 kHz padded by 160 each side (xcodec's framing) → 50 frames.
        assert_eq!(Hubert::frames_for(16_000 + 320), 50);
        assert_eq!(Hubert::frames_for(400), 1);
        assert_eq!(Hubert::frames_for(399), 0);
        assert_eq!(Hubert::frames_for(0), 0);
        assert_eq!(CONV_STRIDES.iter().product::<usize>(), Hubert::HOP);
        assert_eq!(Hubert::frames_for(Hubert::RECEPTIVE_FIELD), 1);
        assert_eq!(Hubert::frames_for(Hubert::RECEPTIVE_FIELD + Hubert::HOP), 2);
        assert_eq!(
            Hubert::frames_for(Hubert::RECEPTIVE_FIELD + Hubert::HOP - 1),
            1
        );
    }
}
