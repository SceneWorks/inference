//! The **UpsampleConformerEncoder** — the token-side network of the S3Gen flow (sc-13237). A
//! faithful native-candle port of Chatterbox's `models/s3gen/transformer/upsample_encoder.py`
//! (CosyVoice2, itself modified from wenet/ESPnet): a linear input embed with ESPnet **relative
//! positional encoding**, a causal `PreLookaheadLayer`, **6** relative-position Conformer blocks, a
//! `Upsample1D` (25 Hz → 50 Hz, `token_mel_ratio = 2`), a second linear embed, **4** more Conformer
//! blocks, and a final LayerNorm.
//!
//! These are the `flow.encoder.*` tensors (206). Each Conformer block here is configured with
//! `macaron_style = False`, `use_cnn_module = False`, so it reduces to `{norm_mha → rel-pos
//! self-attention, norm_ff → SwiGLU-free Swish FFN}` — exactly the tensors present
//! (`self_attn.{linear_q,linear_k,linear_v,linear_out,linear_pos,pos_bias_u,pos_bias_v}`,
//! `feed_forward.{w_1,w_2}`, `norm_mha`, `norm_ff`); there is deliberately no conv module or macaron
//! FFN.
//!
//! ## Faithfulness notes (verified against the reference + the pinned `flow.encoder.*` shapes)
//!
//! - **embed / up_embed** (`LinearNoSubsampling`): `LayerNorm(Linear(x))` (eps 1e-5), then the
//!   ESPnet rel-pos encoding scales the hidden by `xscale = sqrt(output_size)` and yields a length
//!   `2T−1` positional embedding.
//! - **RelPositionMultiHeadedAttention** (`rel_selfattn`, Transformer-XL §3.3): `matrix_ac =
//!   (q+pos_bias_u)·kᵀ`, `matrix_bd = rel_shift((q+pos_bias_v)·pᵀ)`, `scores = (ac+bd)/√d_k`. Single
//!   utterance ⇒ the padding mask is all-ones and omitted.
//! - **PreLookaheadLayer**: right-pad 3, `conv1` (k=4) + leaky-ReLU, left-pad 2, `conv2` (k=3),
//!   residual.
//! - **Upsample1D**: nearest ×2, left-pad 4, `conv` (k=5, stride 1) — output length exactly `2T`.
//! - Conformer layer norms use eps **1e-12** (wenet), the embed/after LayerNorms eps **1e-5**.

use candle_audio::candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{
    conv1d, layer_norm, linear, linear_no_bias, Conv1d, Conv1dConfig, LayerNorm, Linear, Module,
    VarBuilder,
};

/// Encoder width (`output_size`).
pub const ENC_DIM: usize = 512;
/// Attention heads.
pub const ENC_HEADS: usize = 8;
/// Per-head dim (`ENC_DIM / ENC_HEADS`).
pub const ENC_HEAD_DIM: usize = ENC_DIM / ENC_HEADS;
/// Position-wise FFN width (`linear_units`).
const ENC_FFN: usize = 2048;
/// Pre-upsample Conformer blocks (`num_blocks`).
const ENC_BLOCKS: usize = 6;
/// Post-upsample Conformer blocks (hard-coded to 4 in the reference).
const ENC_UP_BLOCKS: usize = 4;
/// PreLookaheadLayer look-ahead length.
const PRE_LOOKAHEAD: usize = 3;
/// Token→mel upsample stride.
const UP_STRIDE: usize = 2;

/// ESPnet relative positional embedding for a `t`-length sequence: `[1, 2t−1, ENC_DIM]` with row
/// `r` holding relative position `p = (t−1) − r` (positions `t−1 … 0 … −(t−1)`), even lanes `sin(p·
/// div)`, odd lanes `cos(p·div)`, `div_j = 10000^(−2j/ENC_DIM)`.
fn rel_pos_emb(t: usize, device: &Device) -> CandleResult<Tensor> {
    let d = ENC_DIM;
    let half = d / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| (10_000f64).powf(-2.0 * j as f64 / d as f64))
        .collect();
    let len = 2 * t - 1;
    let mut data = vec![0f32; len * d];
    for r in 0..len {
        let p = (t as f64 - 1.0) - r as f64;
        for (j, &f) in inv_freq.iter().enumerate() {
            let a = p * f;
            data[r * d + 2 * j] = a.sin() as f32;
            data[r * d + 2 * j + 1] = a.cos() as f32;
        }
    }
    Tensor::from_vec(data, (1, len, d), device)
}

/// The `LinearNoSubsampling` embed: `LayerNorm(Linear(x))` then the rel-pos `xscale = sqrt(dim)`.
struct Embed {
    linear: Linear,
    norm: LayerNorm,
    xscale: f64,
}

impl Embed {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        // `out` is a Sequential(Linear[0], LayerNorm[1], Dropout[2]).
        let out = vb.pp("out");
        Ok(Self {
            linear: linear(ENC_DIM, ENC_DIM, out.pp("0"))?,
            norm: layer_norm(ENC_DIM, 1e-5, out.pp("1"))?,
            xscale: (ENC_DIM as f64).sqrt(),
        })
    }

    /// `[B, T, ENC_DIM]` → (scaled hidden `[B, T, ENC_DIM]`, pos_emb `[1, 2T−1, ENC_DIM]`).
    fn forward(&self, x: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let x = self.norm.forward(&self.linear.forward(x)?)?;
        let x = (x * self.xscale)?;
        let pe = rel_pos_emb(x.dim(1)?, x.device())?;
        Ok((x, pe))
    }
}

/// Relative-position multi-head self-attention (Transformer-XL / ESPnet `rel_pos`).
struct RelSelfAttn {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor, // [H, d_k]
    pos_bias_v: Tensor,
}

impl RelSelfAttn {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            linear_q: linear(ENC_DIM, ENC_DIM, vb.pp("linear_q"))?,
            linear_k: linear(ENC_DIM, ENC_DIM, vb.pp("linear_k"))?,
            linear_v: linear(ENC_DIM, ENC_DIM, vb.pp("linear_v"))?,
            linear_out: linear(ENC_DIM, ENC_DIM, vb.pp("linear_out"))?,
            linear_pos: linear_no_bias(ENC_DIM, ENC_DIM, vb.pp("linear_pos"))?,
            pos_bias_u: vb.get((ENC_HEADS, ENC_HEAD_DIM), "pos_bias_u")?,
            pos_bias_v: vb.get((ENC_HEADS, ENC_HEAD_DIM), "pos_bias_v")?,
        })
    }

    /// The ESPnet `rel_shift`: `[B, H, T, 2T−1]` → `[B, H, T, T]`.
    fn rel_shift(x: &Tensor) -> CandleResult<Tensor> {
        let (b, h, t, n) = x.dims4()?; // n = 2T-1
        let zero = Tensor::zeros((b, h, t, 1), x.dtype(), x.device())?;
        let x = Tensor::cat(&[&zero, x], 3)?; // [B, H, T, 2T]
        let x = x.reshape((b, h, n + 1, t))?; // [B, H, 2T, T]
        let x = x.narrow(2, 1, n)?; // [B, H, 2T-1, T]
        let x = x.reshape((b, h, t, n))?; // view_as original
        x.narrow(3, 0, n / 2 + 1)?.contiguous() // [B, H, T, T]
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> CandleResult<Tensor> {
        let (b, t, _) = x.dims3()?;
        let to_heads = |proj: &Tensor| -> CandleResult<Tensor> {
            proj.reshape((b, t, ENC_HEADS, ENC_HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()
        };
        // q kept in (B, T, H, d_k) for the bias adds; k/v in (B, H, T, d_k).
        let q = self
            .linear_q
            .forward(x)?
            .reshape((b, t, ENC_HEADS, ENC_HEAD_DIM))?;
        let k = to_heads(&self.linear_k.forward(x)?)?;
        let v = to_heads(&self.linear_v.forward(x)?)?;

        // Positional projection: [1, 2T-1, D] -> [1, H, 2T-1, d_k].
        let p_len = pos_emb.dim(1)?;
        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((1, p_len, ENC_HEADS, ENC_HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;

        // (q + bias) broadcast over time, then to (B, H, T, d_k).
        let bias_u = self.pos_bias_u.reshape((1, 1, ENC_HEADS, ENC_HEAD_DIM))?;
        let bias_v = self.pos_bias_v.reshape((1, 1, ENC_HEADS, ENC_HEAD_DIM))?;
        let q_u = q.broadcast_add(&bias_u)?.transpose(1, 2)?.contiguous()?; // [B, H, T, d_k]
        let q_v = q.broadcast_add(&bias_v)?.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (ENC_HEAD_DIM as f64).sqrt();
        let matrix_ac = q_u.matmul(&k.transpose(2, 3)?.contiguous()?)?; // [B, H, T, T]
        let matrix_bd = q_v.broadcast_matmul(&p.transpose(2, 3)?.contiguous()?)?; // [B, H, T, 2T-1]
        let matrix_bd = Self::rel_shift(&matrix_bd)?; // [B, H, T, T]
        let scores = ((matrix_ac + matrix_bd)? * scale)?;
        let attn = softmax_last_dim(&scores)?;
        let ctx = attn
            .matmul(&v)? // [B, H, T, d_k]
            .transpose(1, 2)?
            .reshape((b, t, ENC_DIM))?;
        self.linear_out.forward(&ctx)
    }
}

/// Swish (SiLU) position-wise FFN (`w_2(swish(w_1(x)))`).
struct FeedForward {
    w1: Linear,
    w2: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            w1: linear(ENC_DIM, ENC_FFN, vb.pp("w_1"))?,
            w2: linear(ENC_FFN, ENC_DIM, vb.pp("w_2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.w2.forward(&self.w1.forward(x)?.silu()?)
    }
}

/// One pre-norm Conformer block (macaron/conv disabled): self-attn + FFN.
struct ConformerLayer {
    norm_mha: LayerNorm,
    self_attn: RelSelfAttn,
    norm_ff: LayerNorm,
    feed_forward: FeedForward,
}

impl ConformerLayer {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            norm_mha: layer_norm(ENC_DIM, 1e-12, vb.pp("norm_mha"))?,
            self_attn: RelSelfAttn::load(vb.pp("self_attn"))?,
            norm_ff: layer_norm(ENC_DIM, 1e-12, vb.pp("norm_ff"))?,
            feed_forward: FeedForward::load(vb.pp("feed_forward"))?,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> CandleResult<Tensor> {
        let att = self
            .self_attn
            .forward(&self.norm_mha.forward(x)?, pos_emb)?;
        let x = (x + att)?;
        let ff = self.feed_forward.forward(&self.norm_ff.forward(&x)?)?;
        x + ff
    }
}

/// The causal `PreLookaheadLayer` (channels 512): a look-ahead conv then a causal conv, residual.
struct PreLookahead {
    conv1: Conv1d, // kernel PRE_LOOKAHEAD+1, no padding
    conv2: Conv1d, // kernel 3, no padding
}

impl PreLookahead {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let cfg = Conv1dConfig::default(); // padding 0, stride 1
        Ok(Self {
            conv1: conv1d(ENC_DIM, ENC_DIM, PRE_LOOKAHEAD + 1, cfg, vb.pp("conv1"))?,
            conv2: conv1d(ENC_DIM, ENC_DIM, 3, cfg, vb.pp("conv2"))?,
        })
    }

    /// `[B, T, C]` → `[B, T, C]` (length preserved).
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let h = x.transpose(1, 2)?.contiguous()?; // [B, C, T]
                                                  // look-ahead: right-pad by PRE_LOOKAHEAD, conv1 (k=4) -> length preserved, leaky-relu.
        let h = h.pad_with_zeros(2, 0, PRE_LOOKAHEAD)?;
        let h = self.conv1.forward(&h)?;
        let h = candle_nn::ops::leaky_relu(&h, 0.01)?;
        // causal: left-pad by 2, conv2 (k=3) -> length preserved.
        let h = h.pad_with_zeros(2, 2, 0)?;
        let h = self.conv2.forward(&h)?;
        let h = h.transpose(1, 2)?.contiguous()?; // [B, T, C]
        x + h
    }
}

/// The `Upsample1D` (stride 2): nearest-interp ×2, left-pad 4, conv (k=5, stride 1) → length `2T`.
struct UpLayer {
    conv: Conv1d, // kernel 2*stride+1 = 5
}

impl UpLayer {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            conv: conv1d(
                ENC_DIM,
                ENC_DIM,
                2 * UP_STRIDE + 1,
                Conv1dConfig::default(),
                vb.pp("conv"),
            )?,
        })
    }

    /// `[B, C, T]` → `[B, C, 2T]`.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let t = x.dim(2)?;
        let up = x.upsample_nearest1d(t * UP_STRIDE)?; // [B, C, 2T]
        let up = up.pad_with_zeros(2, UP_STRIDE * 2, 0)?; // left-pad 4 -> [B, C, 2T+4]
        self.conv.forward(&up) // k=5 -> [B, C, 2T]
    }
}

/// The assembled UpsampleConformerEncoder (`flow.encoder.*`).
pub struct UpsampleConformerEncoder {
    embed: Embed,
    pre_lookahead: PreLookahead,
    encoders: Vec<ConformerLayer>,
    up_layer: UpLayer,
    up_embed: Embed,
    up_encoders: Vec<ConformerLayer>,
    after_norm: LayerNorm,
}

impl UpsampleConformerEncoder {
    /// Build from a `flow.encoder.*`-rooted [`VarBuilder`].
    pub fn load(vb: VarBuilder) -> CandleResult<Self> {
        let mut encoders = Vec::with_capacity(ENC_BLOCKS);
        for i in 0..ENC_BLOCKS {
            encoders.push(ConformerLayer::load(vb.pp("encoders").pp(i))?);
        }
        let mut up_encoders = Vec::with_capacity(ENC_UP_BLOCKS);
        for i in 0..ENC_UP_BLOCKS {
            up_encoders.push(ConformerLayer::load(vb.pp("up_encoders").pp(i))?);
        }
        Ok(Self {
            embed: Embed::load(vb.pp("embed"))?,
            pre_lookahead: PreLookahead::load(vb.pp("pre_lookahead_layer"))?,
            encoders,
            up_layer: UpLayer::load(vb.pp("up_layer"))?,
            up_embed: Embed::load(vb.pp("up_embed"))?,
            up_encoders,
            after_norm: layer_norm(ENC_DIM, 1e-5, vb.pp("after_norm"))?,
        })
    }

    /// `[B, T, ENC_DIM]` token embeddings → `[B, 2T, ENC_DIM]` upsampled encoding.
    pub fn forward(&self, tokens: &Tensor) -> CandleResult<Tensor> {
        let (mut x, mut pos_emb) = self.embed.forward(tokens)?;
        x = self.pre_lookahead.forward(&x)?;
        for layer in &self.encoders {
            x = layer.forward(&x, &pos_emb)?;
        }
        // Upsample ×2 in [B, C, T] layout.
        let x_ct = x.transpose(1, 2)?.contiguous()?;
        let x_ct = self.up_layer.forward(&x_ct)?;
        x = x_ct.transpose(1, 2)?.contiguous()?; // [B, 2T, C]
        let (x2, pe2) = self.up_embed.forward(&x)?;
        x = x2;
        pos_emb = pe2;
        for layer in &self.up_encoders {
            x = layer.forward(&x, &pos_emb)?;
        }
        self.after_norm.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::DType;
    use candle_nn::VarMap;

    #[test]
    fn rel_pos_emb_is_centered_and_correct_length() {
        let dev = Device::Cpu;
        let t = 5usize;
        let pe = rel_pos_emb(t, &dev).unwrap();
        assert_eq!(pe.dims(), &[1, 2 * t - 1, ENC_DIM]);
        // Centre row (index t-1) is relative position 0 → sin=0, cos=1 on every lane.
        let center: Vec<f32> = pe
            .narrow(1, t - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for j in 0..ENC_DIM / 2 {
            assert!(center[2 * j].abs() < 1e-6, "sin lane {j} at p=0");
            assert!(
                (center[2 * j + 1] - 1.0).abs() < 1e-6,
                "cos lane {j} at p=0"
            );
        }
    }

    #[test]
    fn rel_shift_maps_2t_minus_1_to_t() {
        let dev = Device::Cpu;
        let (b, h, t) = (1usize, 2usize, 4usize);
        let n = 2 * t - 1;
        let x = Tensor::arange(0f32, (b * h * t * n) as f32, &dev)
            .unwrap()
            .reshape((b, h, t, n))
            .unwrap();
        let y = RelSelfAttn::rel_shift(&x).unwrap();
        assert_eq!(y.dims(), &[b, h, t, t]);
        assert!(y
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    /// Wire the whole encoder from synthetic weights and confirm the 25→50 Hz upsample ratio and
    /// the 512-wide output. Real weights are exercised by the `#[ignore]`d conformance test.
    #[test]
    fn encoder_upsamples_25hz_to_50hz_with_synthetic_weights() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        materialize_encoder_shapes(&vb);
        let enc = UpsampleConformerEncoder::load(vb).unwrap();

        let t = 7usize;
        let tokens = Tensor::randn(0f32, 1.0, (1, t, ENC_DIM), &dev).unwrap();
        let out = enc.forward(&tokens).unwrap();
        assert_eq!(out.dims(), &[1, 2 * t, ENC_DIM], "output must be 2T frames");
        assert!(out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    /// Materialize every `flow.encoder.*` tensor at its real shape.
    fn materialize_encoder_shapes(vb: &VarBuilder) {
        let embed = |vb: &VarBuilder| {
            let out = vb.pp("out");
            let _ = out.get((ENC_DIM, ENC_DIM), "0.weight").unwrap();
            let _ = out.get(ENC_DIM, "0.bias").unwrap();
            let _ = out.get(ENC_DIM, "1.weight").unwrap();
            let _ = out.get(ENC_DIM, "1.bias").unwrap();
        };
        let conformer = |vb: &VarBuilder| {
            let sa = vb.pp("self_attn");
            for name in ["linear_q", "linear_k", "linear_v", "linear_out"] {
                let _ = sa
                    .get((ENC_DIM, ENC_DIM), &format!("{name}.weight"))
                    .unwrap();
                let _ = sa.get(ENC_DIM, &format!("{name}.bias")).unwrap();
            }
            let _ = sa.get((ENC_DIM, ENC_DIM), "linear_pos.weight").unwrap();
            let _ = sa.get((ENC_HEADS, ENC_HEAD_DIM), "pos_bias_u").unwrap();
            let _ = sa.get((ENC_HEADS, ENC_HEAD_DIM), "pos_bias_v").unwrap();
            let _ = vb
                .get((ENC_FFN, ENC_DIM), "feed_forward.w_1.weight")
                .unwrap();
            let _ = vb.get(ENC_FFN, "feed_forward.w_1.bias").unwrap();
            let _ = vb
                .get((ENC_DIM, ENC_FFN), "feed_forward.w_2.weight")
                .unwrap();
            let _ = vb.get(ENC_DIM, "feed_forward.w_2.bias").unwrap();
            for norm in ["norm_mha", "norm_ff"] {
                let _ = vb.get(ENC_DIM, &format!("{norm}.weight")).unwrap();
                let _ = vb.get(ENC_DIM, &format!("{norm}.bias")).unwrap();
            }
        };
        embed(&vb.pp("embed"));
        embed(&vb.pp("up_embed"));
        let pl = vb.pp("pre_lookahead_layer");
        let _ = pl
            .get((ENC_DIM, ENC_DIM, PRE_LOOKAHEAD + 1), "conv1.weight")
            .unwrap();
        let _ = pl.get(ENC_DIM, "conv1.bias").unwrap();
        let _ = pl.get((ENC_DIM, ENC_DIM, 3), "conv2.weight").unwrap();
        let _ = pl.get(ENC_DIM, "conv2.bias").unwrap();
        let ul = vb.pp("up_layer");
        let _ = ul
            .get((ENC_DIM, ENC_DIM, 2 * UP_STRIDE + 1), "conv.weight")
            .unwrap();
        let _ = ul.get(ENC_DIM, "conv.bias").unwrap();
        for i in 0..ENC_BLOCKS {
            conformer(&vb.pp("encoders").pp(i));
        }
        for i in 0..ENC_UP_BLOCKS {
            conformer(&vb.pp("up_encoders").pp(i));
        }
        let _ = vb.get(ENC_DIM, "after_norm.weight").unwrap();
        let _ = vb.get(ENC_DIM, "after_norm.bias").unwrap();
    }
}
