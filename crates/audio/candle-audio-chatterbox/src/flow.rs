//! The **S3Gen flow** — `CausalMaskedDiffWithXvec` + `CausalConditionalCFM` (sc-13237). The third
//! of S3Gen's four networks: it turns 25 Hz speech tokens (plus a reference prompt) into an 80-bin
//! 50 Hz mel via flow-matching. A faithful native-candle port of Chatterbox's `models/s3gen/flow.py`,
//! `flow_matching.py`, and `decoder.py` (CosyVoice2 / Matcha-TTS).
//!
//! It is the `flow.*` block of `s3gen.safetensors` (**1121 tensors**). Its pieces:
//!
//! - **token embedding** `Embedding(6561 → 512)` (`flow.input_embedding`);
//! - the [`UpsampleConformerEncoder`](crate::flow_encoder) (`flow.encoder.*`, output 512, 8 heads,
//!   6+4 blocks, 25 Hz → 50 Hz) + `encoder_proj` `Linear(512 → 80)` (`flow.encoder_proj`), giving
//!   the CFM's conditioning mean `mu`;
//! - the **CausalConditionalCFM** Euler flow-matching sampler (`n_timesteps = 10`, **cosine**
//!   t-schedule, classifier-free guidance `inference_cfg_rate = 0.7`) over
//! - a **ConditionalDecoder** U-Net estimator (`flow.decoder.estimator.*`): input 320 channels
//!   (`x 80 ++ mu 80 ++ spk 80 ++ cond 80`), a causal ResNet down-block, **12** DiT mid-blocks (each
//!   a ResnetBlock1D + 4 self-attention transformer blocks, 8 heads), a causal ResNet up-block with
//!   the skip, and a final block + `Conv1d(256 → 80)` head predicting the 80-bin velocity field.
//!
//! The speaker conditioning `spk_embed_80` is the L2-normalized 192→80 `flow.spk_embed_affine_layer`
//! projection of the CAMPPlus x-vector — produced by [`Campplus::spk_embed_flow`](crate::campplus)
//! (which owns `flow.spk_embed_affine_layer`, the remaining 2 of the 1121 `flow.*` tensors), so it
//! is consumed here directly.
//!
//! Single-utterance inference has no padding, so the reference's all-ones masks are no-ops and are
//! omitted throughout (numerically identical). The decoder is built with `causal = True`,
//! `channels = [256]`, so its "down/up-sample" convs are length-preserving causal `Conv1d(256, 256,
//! 3)` and the whole estimator preserves the mel time axis.

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_audio::{AudioError, Result};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{
    conv1d, embedding, layer_norm, linear, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear,
    Module, VarBuilder,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::path::Path;

use crate::config::{S3GenConfig, SPEECH_VOCAB_SIZE};
use crate::flow_encoder::{UpsampleConformerEncoder, ENC_DIM};
use crate::mel24::Mel24Extractor;
use crate::s3gen::S3GEN_WEIGHTS_FILE;

/// Mel width (`out_channels`).
const MEL_DIM: usize = 80;
/// Decoder hidden width (`channels[0]`).
const DEC_CH: usize = 256;
/// Decoder input channels (`x ++ mu ++ spk ++ cond` = 4 × 80).
const DEC_IN: usize = 4 * MEL_DIM;
/// Time-embedding width (`channels[0] * 4`).
const TIME_EMB_DIM: usize = DEC_CH * 4;
/// Number of DiT mid-blocks.
const MID_BLOCKS: usize = 12;
/// Transformer blocks per down/mid/up block.
const N_TB: usize = 4;
/// Transformer attention heads.
const TB_HEADS: usize = 8;
/// Transformer per-head dim (`attention_head_dim`).
const TB_HEAD_DIM: usize = 64;
/// Transformer inner attention width (`heads · head_dim`).
const TB_INNER: usize = TB_HEADS * TB_HEAD_DIM;
/// Transformer FFN inner width (`dim · 4`).
const TB_FFN: usize = DEC_CH * 4;

// =================================================================================================
// Timestep embedding (Matcha SinusoidalPosEmb + TimestepEmbedding).
// =================================================================================================

/// `SinusoidalPosEmb(dim=DEC_IN)` at `scale = 1000`: `cat[sin(1000·t·inv), cos(1000·t·inv)]` with
/// `inv_j = exp(−j · ln(10000)/(half−1))`, half = dim/2. Host f64 math for one scalar `t`.
fn sinusoidal_pos_emb(t: f64, device: &Device) -> CandleResult<Tensor> {
    let dim = DEC_IN; // 320
    let half = dim / 2;
    let scale = 1000.0f64;
    let step = (10_000f64).ln() / (half as f64 - 1.0);
    let mut out = vec![0f32; dim];
    for j in 0..half {
        let inv = (-(j as f64) * step).exp();
        let a = scale * t * inv;
        out[j] = a.sin() as f32; // sin half first
        out[half + j] = a.cos() as f32; // cos half second
    }
    Tensor::from_vec(out, (1, dim), device)
}

/// `TimestepEmbedding`: `linear_2(silu(linear_1(·)))` (`DEC_IN → TIME_EMB_DIM → TIME_EMB_DIM`).
struct TimeMlp {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimeMlp {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            linear_1: linear(DEC_IN, TIME_EMB_DIM, vb.pp("linear_1"))?,
            linear_2: linear(TIME_EMB_DIM, TIME_EMB_DIM, vb.pp("linear_2"))?,
        })
    }
    fn forward(&self, t: &Tensor) -> CandleResult<Tensor> {
        self.linear_2.forward(&self.linear_1.forward(t)?.silu()?)
    }
}

// =================================================================================================
// Causal 1-D conv blocks (Matcha CausalBlock1D / CausalResnetBlock1D).
// =================================================================================================

/// `nn.Conv1d` with left-only causal padding of `kernel − 1` (the reference `CausalConv1d`).
struct CausalConv1d {
    conv: Conv1d,
    left_pad: usize,
}

impl CausalConv1d {
    fn load(in_ch: usize, out_ch: usize, kernel: usize, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            conv: conv1d(in_ch, out_ch, kernel, Conv1dConfig::default(), vb)?,
            left_pad: kernel - 1,
        })
    }
    /// `[B, in, T]` → `[B, out, T]` (length preserved).
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.conv.forward(&x.pad_with_zeros(2, self.left_pad, 0)?)
    }
}

/// `CausalBlock1D`: `Mish(LayerNorm(CausalConv1d(x)))` (LayerNorm over channels via transpose).
struct CausalBlock1D {
    conv: CausalConv1d,
    norm: LayerNorm,
}

impl CausalBlock1D {
    fn load(in_ch: usize, out_ch: usize, vb: VarBuilder) -> CandleResult<Self> {
        // block = Sequential(CausalConv1d[0], Transpose, LayerNorm[2], Transpose, Mish).
        let conv = CausalConv1d::load(in_ch, out_ch, 3, vb.pp("block").pp("0"))?;
        let norm = layer_norm(out_ch, 1e-5, vb.pp("block").pp("2"))?;
        Ok(Self { conv, norm })
    }
    /// `[B, in, T]` → `[B, out, T]`.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let h = self.conv.forward(x)?; // [B, out, T]
        let h = self.norm.forward(&h.transpose(1, 2)?.contiguous()?)?; // LN over channels
        let h = h.transpose(1, 2)?.contiguous()?; // [B, out, T]
        candle_nn::ops::mish(&h)
    }
}

/// `CausalResnetBlock1D`: `block2(block1(x) + mlp(t)) + res_conv(x)`.
struct ResnetBlock {
    mlp: Linear, // mlp = Sequential(Mish, Linear(time_emb, dim_out)); Mish applied on t externally
    block1: CausalBlock1D,
    block2: CausalBlock1D,
    res_conv: Conv1d, // 1x1
}

impl ResnetBlock {
    fn load(dim: usize, dim_out: usize, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            mlp: linear(TIME_EMB_DIM, dim_out, vb.pp("mlp").pp("1"))?,
            block1: CausalBlock1D::load(dim, dim_out, vb.pp("block1"))?,
            block2: CausalBlock1D::load(dim_out, dim_out, vb.pp("block2"))?,
            res_conv: conv1d(dim, dim_out, 1, Conv1dConfig::default(), vb.pp("res_conv"))?,
        })
    }

    /// `x [B, dim, T]`, `t_emb [B, TIME_EMB_DIM]` → `[B, dim_out, T]`.
    fn forward(&self, x: &Tensor, t_emb: &Tensor) -> CandleResult<Tensor> {
        let h = self.block1.forward(x)?;
        // mlp = Linear(Mish(t)); add the [B, dim_out, 1] time bias, broadcast over time.
        let t = self
            .mlp
            .forward(&candle_nn::ops::mish(t_emb)?)?
            .unsqueeze(D::Minus1)?;
        let h = h.broadcast_add(&t)?;
        let h = self.block2.forward(&h)?;
        h + self.res_conv.forward(x)?
    }
}

// =================================================================================================
// BasicTransformerBlock (diffusers/matcha): self-attention + GELU FFN, plain LayerNorm.
// =================================================================================================

struct SelfAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
}

impl SelfAttention {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            to_q: candle_nn::linear_no_bias(DEC_CH, TB_INNER, vb.pp("to_q"))?,
            to_k: candle_nn::linear_no_bias(DEC_CH, TB_INNER, vb.pp("to_k"))?,
            to_v: candle_nn::linear_no_bias(DEC_CH, TB_INNER, vb.pp("to_v"))?,
            to_out: linear(TB_INNER, DEC_CH, vb.pp("to_out").pp("0"))?,
        })
    }

    /// `[B, T, DEC_CH]` self-attention → `[B, T, DEC_CH]`.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, t, _) = x.dims3()?;
        let heads = |proj: &Tensor| -> CandleResult<Tensor> {
            proj.reshape((b, t, TB_HEADS, TB_HEAD_DIM))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = heads(&self.to_q.forward(x)?)?;
        let k = heads(&self.to_k.forward(x)?)?;
        let v = heads(&self.to_v.forward(x)?)?;
        let scale = 1.0 / (TB_HEAD_DIM as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let att = softmax_last_dim(&att)?;
        let ctx = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, TB_INNER))?;
        self.to_out.forward(&ctx)
    }
}

/// `FeedForward` with `activation_fn="gelu"`: `Linear(1024←256) → GELU → Linear(256←1024)`.
struct FeedForwardGelu {
    proj: Linear, // net.0.proj
    out: Linear,  // net.2
}

impl FeedForwardGelu {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            proj: linear(DEC_CH, TB_FFN, vb.pp("net").pp("0").pp("proj"))?,
            out: linear(TB_FFN, DEC_CH, vb.pp("net").pp("2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.out.forward(&self.proj.forward(x)?.gelu_erf()?)
    }
}

/// One `BasicTransformerBlock`: `x += attn1(norm1(x)); x += ff(norm3(x))`.
struct TransformerBlock {
    norm1: LayerNorm,
    attn1: SelfAttention,
    norm3: LayerNorm,
    ff: FeedForwardGelu,
}

impl TransformerBlock {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(DEC_CH, 1e-5, vb.pp("norm1"))?,
            attn1: SelfAttention::load(vb.pp("attn1"))?,
            norm3: layer_norm(DEC_CH, 1e-5, vb.pp("norm3"))?,
            ff: FeedForwardGelu::load(vb.pp("ff"))?,
        })
    }
    /// `[B, T, DEC_CH]` → `[B, T, DEC_CH]`.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = (x + self.attn1.forward(&self.norm1.forward(x)?)?)?;
        let ff = self.ff.forward(&self.norm3.forward(&x)?)?;
        x + ff
    }
}

/// A `(ResnetBlock, [TransformerBlock; N_TB])` stage, run in the `[B, C, T]` / `[B, T, C]` dance.
struct UNetStage {
    resnet: ResnetBlock,
    transformers: Vec<TransformerBlock>,
}

impl UNetStage {
    fn load(dim: usize, dim_out: usize, vb: VarBuilder) -> CandleResult<Self> {
        let resnet = ResnetBlock::load(dim, dim_out, vb.pp("0"))?;
        let mut transformers = Vec::with_capacity(N_TB);
        for i in 0..N_TB {
            transformers.push(TransformerBlock::load(vb.pp("1").pp(i))?);
        }
        Ok(Self {
            resnet,
            transformers,
        })
    }

    /// `x [B, dim, T]` → `[B, dim_out, T]`.
    fn forward(&self, x: &Tensor, t_emb: &Tensor) -> CandleResult<Tensor> {
        let x = self.resnet.forward(x, t_emb)?; // [B, dim_out, T]
        let mut h = x.transpose(1, 2)?.contiguous()?; // [B, T, dim_out]
        for tb in &self.transformers {
            h = tb.forward(&h)?;
        }
        h.transpose(1, 2)?.contiguous() // [B, dim_out, T]
    }
}

// =================================================================================================
// ConditionalDecoder — the CFM velocity estimator (flow.decoder.estimator.*).
// =================================================================================================

/// The U-Net velocity estimator: `time_mlp`, one down-stage, `MID_BLOCKS` mid-stages, one up-stage
/// (with the down skip), a final causal block, and the `Conv1d(256 → 80)` head.
pub struct ConditionalDecoder {
    time_mlp: TimeMlp,
    down: UNetStage,
    down_sample: CausalConv1d,
    mid: Vec<UNetStage>,
    up: UNetStage,
    up_sample: CausalConv1d,
    final_block: CausalBlock1D,
    final_proj: Conv1d, // 1x1, 256 -> 80
}

impl ConditionalDecoder {
    /// Build from a `flow.decoder.estimator.*`-rooted [`VarBuilder`].
    pub fn load(vb: VarBuilder) -> CandleResult<Self> {
        let time_mlp = TimeMlp::load(vb.pp("time_mlp"))?;
        let down = UNetStage::load(DEC_IN, DEC_CH, vb.pp("down_blocks").pp("0"))?;
        let down_sample =
            CausalConv1d::load(DEC_CH, DEC_CH, 3, vb.pp("down_blocks").pp("0").pp("2"))?;
        let mut mid = Vec::with_capacity(MID_BLOCKS);
        for i in 0..MID_BLOCKS {
            mid.push(UNetStage::load(DEC_CH, DEC_CH, vb.pp("mid_blocks").pp(i))?);
        }
        // up resnet takes 2×DEC_CH in (skip concat).
        let up = UNetStage::load(2 * DEC_CH, DEC_CH, vb.pp("up_blocks").pp("0"))?;
        let up_sample = CausalConv1d::load(DEC_CH, DEC_CH, 3, vb.pp("up_blocks").pp("0").pp("2"))?;
        let final_block = CausalBlock1D::load(DEC_CH, DEC_CH, vb.pp("final_block"))?;
        let final_proj = conv1d(
            DEC_CH,
            MEL_DIM,
            1,
            Conv1dConfig::default(),
            vb.pp("final_proj"),
        )?;
        Ok(Self {
            time_mlp,
            down,
            down_sample,
            mid,
            up,
            up_sample,
            final_block,
            final_proj,
        })
    }

    /// One velocity prediction. `x, mu, cond` are `[B, 80, T]`; `spks` is `[B, 80]`; `t` a scalar in
    /// `[0, 1]`. Returns `[B, 80, T]`.
    pub fn forward(
        &self,
        x: &Tensor,
        mu: &Tensor,
        t: f64,
        spks: &Tensor,
        cond: &Tensor,
    ) -> CandleResult<Tensor> {
        let device = x.device();
        let t_emb = self.time_mlp.forward(&sinusoidal_pos_emb(t, device)?)?; // [1, TIME_EMB_DIM]

        // Channel-concat [x, mu, spks(repeat over T), cond] -> [B, 320, T].
        let tlen = x.dim(2)?;
        let spks_rep = spks
            .unsqueeze(D::Minus1)?
            .broadcast_as((spks.dim(0)?, MEL_DIM, tlen))?;
        let mut h = Tensor::cat(&[x, mu, &spks_rep, cond], 1)?.contiguous()?;

        // Down stage + skip.
        h = self.down.forward(&h, &t_emb)?;
        let skip = h.clone();
        h = self.down_sample.forward(&h)?;

        // Mid stages.
        for stage in &self.mid {
            h = stage.forward(&h, &t_emb)?;
        }

        // Up stage: concat skip along channels, then upsample conv.
        h = Tensor::cat(&[&h, &skip], 1)?.contiguous()?;
        h = self.up.forward(&h, &t_emb)?;
        h = self.up_sample.forward(&h)?;

        h = self.final_block.forward(&h)?;
        self.final_proj.forward(&h)
    }
}

// =================================================================================================
// CausalConditionalCFM — cosine-schedule Euler flow-matching with classifier-free guidance.
// =================================================================================================

/// The cosine flow-matching schedule: `t_span = 1 − cos(linspace(0,1,n+1) · π/2)` (`n` Euler steps).
/// Pure/host-testable.
#[derive(Debug, Clone)]
pub struct CfmSchedule {
    t_span: Vec<f64>,
}

impl CfmSchedule {
    /// Build the `n_timesteps`-step cosine schedule.
    pub fn cosine(n_timesteps: usize) -> Self {
        let n = n_timesteps.max(1);
        let t_span = (0..=n)
            .map(|k| {
                let t = k as f64 / n as f64;
                1.0 - (t * 0.5 * std::f64::consts::PI).cos()
            })
            .collect();
        Self { t_span }
    }

    /// Number of Euler steps.
    pub fn num_steps(&self) -> usize {
        self.t_span.len() - 1
    }

    /// The scheduled time at node `k` (`t_span[k]`).
    pub fn t(&self, k: usize) -> f64 {
        self.t_span[k]
    }

    /// The Euler step size at step `k` (`t_span[k+1] − t_span[k]`).
    pub fn dt(&self, k: usize) -> f64 {
        self.t_span[k + 1] - self.t_span[k]
    }
}

/// The `CausalConditionalCFM` decoder: the cosine schedule over a [`ConditionalDecoder`] estimator,
/// integrated with the Euler solver and classifier-free guidance `inference_cfg_rate`.
pub struct CausalConditionalCfm {
    estimator: ConditionalDecoder,
    inference_cfg_rate: f64,
    steps: usize,
}

impl CausalConditionalCfm {
    fn new(estimator: ConditionalDecoder, cfg: &S3GenConfig) -> Self {
        Self {
            estimator,
            inference_cfg_rate: cfg.cfm_inference_cfg_rate,
            steps: cfg.cfm_steps,
        }
    }

    /// Solve the ODE from seeded standard-normal noise. `mu, cond` are `[1, 80, T]`, `spks` `[1, 80]`.
    /// Returns the generated mel `[1, 80, T]`.
    fn solve(&self, mu: &Tensor, spks: &Tensor, cond: &Tensor, seed: u64) -> CandleResult<Tensor> {
        let device = mu.device();
        let (_, feats, tlen) = mu.dims3()?;
        // z ~ N(0, I), seeded (the gen-core reproducibility law).
        let mut rng = StdRng::seed_from_u64(seed);
        let noise: Vec<f32> = (0..feats * tlen)
            .map(|_| rng.sample(StandardNormal))
            .collect();
        let mut x = Tensor::from_vec(noise, (1, feats, tlen), device)?;

        // Zeroed conditioning for the unconditioned CFG pass.
        let mu_zero = mu.zeros_like()?;
        let spks_zero = spks.zeros_like()?;
        let cond_zero = cond.zeros_like()?;

        let schedule = CfmSchedule::cosine(self.steps);
        let cfg = self.inference_cfg_rate;
        for k in 0..schedule.num_steps() {
            let t = schedule.t(k);
            let dt = schedule.dt(k);
            let dphi_cond = self.estimator.forward(&x, mu, t, spks, cond)?;
            let dphi_uncond = self
                .estimator
                .forward(&x, &mu_zero, t, &spks_zero, &cond_zero)?;
            // dxdt = (1 + rate)·cond − rate·uncond.
            let dxdt = ((dphi_cond * (1.0 + cfg))? - (dphi_uncond * cfg)?)?;
            x = (x + (dxdt * dt)?)?;
        }
        Ok(x)
    }
}

// =================================================================================================
// Flow — CausalMaskedDiffWithXvec top-level.
// =================================================================================================

/// The assembled S3Gen flow: token embedding + [`UpsampleConformerEncoder`] + `encoder_proj` +
/// [`CausalConditionalCfm`]. Produces an 80-bin mel from speech tokens, a reference prompt (tokens +
/// mel), and the 80-d flow speaker embedding.
pub struct Flow {
    input_embedding: Embedding,
    encoder: UpsampleConformerEncoder,
    encoder_proj: Linear,
    cfm: CausalConditionalCfm,
    mel_extractor: Mel24Extractor,
    device: Device,
    cfg: S3GenConfig,
}

impl Flow {
    /// Load the flow from a Chatterbox snapshot directory (reads `s3gen.safetensors`, prefix
    /// `flow.*`; `flow.spk_embed_affine_layer` is owned by [`crate::campplus`] and not read here).
    pub fn from_snapshot(dir: &Path) -> Result<Self> {
        let path = dir.join(S3GEN_WEIGHTS_FILE);
        if !path.is_file() {
            return Err(AudioError::Msg(format!(
                "flow: {} missing (the flow weights live in the S3Gen checkpoint)",
                path.display()
            )));
        }
        let device = candle_audio::default_device_metal_incompatible()?;
        // SAFETY: mmap of a provider-resolved, pinned-SHA safetensors file — the shared idiom.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&path), DType::F32, &device)?
        };
        Ok(Self::new(&S3GenConfig::DEFAULT, vb.pp("flow"), device)?)
    }

    /// Build from a `flow.*`-rooted [`VarBuilder`].
    pub fn new(cfg: &S3GenConfig, vb: VarBuilder, device: Device) -> CandleResult<Self> {
        let input_embedding = embedding(SPEECH_VOCAB_SIZE, ENC_DIM, vb.pp("input_embedding"))?;
        let encoder = UpsampleConformerEncoder::load(vb.pp("encoder"))?;
        let encoder_proj = linear(ENC_DIM, MEL_DIM, vb.pp("encoder_proj"))?;
        let estimator = ConditionalDecoder::load(vb.pp("decoder").pp("estimator"))?;
        let cfm = CausalConditionalCfm::new(estimator, cfg);
        Ok(Self {
            input_embedding,
            encoder,
            encoder_proj,
            cfm,
            mel_extractor: Mel24Extractor::new(cfg),
            device,
            cfg: *cfg,
        })
    }

    /// The 24 kHz prompt-mel extractor (also used to derive `prompt_mel` from a reference clip).
    pub fn mel_extractor(&self) -> &Mel24Extractor {
        &self.mel_extractor
    }

    /// The configured device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Render an 80-bin mel from speech tokens, a reference prompt (prompt tokens + prompt mel), and
    /// the 80-d flow speaker embedding.
    ///
    /// - `speech_tokens`: the 25 Hz S3 speech tokens to synthesize (`[0, 6560]`).
    /// - `prompt_tokens`: the reference clip's 25 Hz speech tokens (the s3tokenizer output).
    /// - `prompt_mel`: the reference clip's 24 kHz 80-bin mel `[n_prompt_frames, 80]`
    ///   (from [`Mel24Extractor`]).
    /// - `spk_embed_80`: the L2-normalized 192→80 flow speaker embedding
    ///   ([`Campplus::spk_embed_flow`](crate::campplus)).
    ///
    /// Returns the generated mel `[80, T_mel]` (`T_mel ≈ 2 × speech_tokens.len()`), the frames for
    /// the new tokens only (the prompt-mel prefix is dropped, mirroring the reference).
    pub fn inference(
        &self,
        speech_tokens: &[u32],
        prompt_tokens: &[u32],
        prompt_mel: &Tensor,
        spk_embed_80: &[f32],
        seed: u64,
    ) -> Result<Tensor> {
        if speech_tokens.is_empty() {
            return Err(AudioError::Msg("flow: no speech tokens to render".into()));
        }
        if spk_embed_80.len() != MEL_DIM {
            return Err(AudioError::Msg(format!(
                "flow: spk_embed must be {MEL_DIM}-d (the flow speaker embedding), got {}",
                spk_embed_80.len()
            )));
        }
        let (mel_len1, feat_dim) = prompt_mel.dims2()?;
        if feat_dim != MEL_DIM {
            return Err(AudioError::Msg(format!(
                "flow: prompt_mel must be [frames, {MEL_DIM}], got [_, {feat_dim}]"
            )));
        }

        // Align: make prompt tokens ≈ mel_len1 / 2 (the reference's "mel_len = 2 · token_len" fixup;
        // it truncates the prompt tokens to prompt_mel_frames // 2).
        let n_prompt = prompt_tokens.len().min(mel_len1 / 2);
        let prompt = &prompt_tokens[..n_prompt];

        // Validate the token range (defensive: the flow embedding is [0, 6560]).
        for &tk in prompt.iter().chain(speech_tokens) {
            if tk as usize >= SPEECH_VOCAB_SIZE {
                return Err(AudioError::Msg(format!(
                    "flow: token {tk} out of range (>= {SPEECH_VOCAB_SIZE}); drop special tokens first"
                )));
            }
        }

        // Token embedding over [prompt ++ speech].
        let mut ids: Vec<u32> = Vec::with_capacity(prompt.len() + speech_tokens.len());
        ids.extend_from_slice(prompt);
        ids.extend_from_slice(speech_tokens);
        let n_tok = ids.len();
        let ids = Tensor::from_vec(ids, (1, n_tok), &self.device)?;
        let token_emb = self.input_embedding.forward(&ids)?; // [1, T, 512]

        // Encoder + projection → mu [1, 80, 2T].
        let h = self.encoder.forward(&token_emb)?; // [1, 2T, 512]
        let total = h.dim(1)?; // 2T
        if mel_len1 >= total {
            return Err(AudioError::Msg(format!(
                "flow: prompt mel ({mel_len1} frames) >= encoder output ({total} frames)"
            )));
        }
        let mel_len2 = total - mel_len1;
        let mu = self
            .encoder_proj
            .forward(&h)?
            .transpose(1, 2)?
            .contiguous()?; // [1, 80, 2T]

        // conds: prompt mel at the front, zeros after; [1, 80, 2T].
        let prompt_mel_b = prompt_mel.unsqueeze(0)?; // [1, mel_len1, 80]
        let pad = Tensor::zeros((1, mel_len2, MEL_DIM), DType::F32, &self.device)?;
        let cond = Tensor::cat(&[&prompt_mel_b, &pad], 1)?
            .transpose(1, 2)?
            .contiguous()?; // [1, 80, 2T]

        // Speaker embedding [1, 80].
        let spks = Tensor::from_vec(spk_embed_80.to_vec(), (1, MEL_DIM), &self.device)?;

        // Flow-matching solve, then drop the prompt-mel prefix.
        let feat = self.cfm.solve(&mu, &spks, &cond, seed)?; // [1, 80, 2T]
        let feat = feat.narrow(2, mel_len1, mel_len2)?; // [1, 80, mel_len2]
        Ok(feat.squeeze(0)?) // [80, mel_len2]
    }

    /// The S3Gen config in use.
    pub fn config(&self) -> &S3GenConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn cosine_schedule_matches_reference_shape() {
        let s = CfmSchedule::cosine(10);
        assert_eq!(s.num_steps(), 10);
        // t_span = 1 - cos(linspace(0,1,11) * pi/2): starts at 0, ends at 1, monotone increasing.
        assert!((s.t(0) - 0.0).abs() < 1e-12);
        assert!((s.t(10) - 1.0).abs() < 1e-12);
        for k in 0..10 {
            assert!(s.t(k + 1) > s.t(k), "schedule must be increasing");
            assert!(s.dt(k) > 0.0);
        }
        // Reference value at k=1: 1 - cos(0.05*pi).
        let want = 1.0 - (0.1f64 * 0.5 * std::f64::consts::PI).cos();
        assert!((s.t(1) - want).abs() < 1e-12);
        // The steps sum to 1 (the full [0,1] integration span).
        let sum: f64 = (0..10).map(|k| s.dt(k)).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cfg_blend_is_guidance_at_rate_0_7() {
        // dxdt = (1+r)·cond − r·uncond, r = 0.7.
        let dev = Device::Cpu;
        let cond = Tensor::full(2.0f32, (1, 4, 3), &dev).unwrap();
        let uncond = Tensor::full(1.0f32, (1, 4, 3), &dev).unwrap();
        let r = 0.7f64;
        let dxdt = ((cond * (1.0 + r)).unwrap() - (uncond * r).unwrap()).unwrap();
        let v = dxdt.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        // (1.7)·2 − 0.7·1 = 3.4 − 0.7 = 2.7.
        assert!((v - 2.7).abs() < 1e-5);
    }

    #[test]
    fn sinusoidal_pos_emb_layout_is_sin_then_cos() {
        let dev = Device::Cpu;
        // t = 0 → all sin = 0, all cos = 1.
        let e = sinusoidal_pos_emb(0.0, &dev).unwrap();
        assert_eq!(e.dims(), &[1, DEC_IN]);
        let v = e.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let half = DEC_IN / 2;
        assert!(v[..half].iter().all(|&s| s.abs() < 1e-6), "sin half at t=0");
        assert!(
            v[half..].iter().all(|&c| (c - 1.0).abs() < 1e-6),
            "cos half at t=0"
        );
    }

    /// Estimator block accounting on synthetic weights: 320 → 80 velocity field, length preserved.
    #[test]
    fn estimator_maps_320_to_80_preserving_length() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        materialize_estimator_shapes(&vb);
        let est = ConditionalDecoder::load(vb).unwrap();

        let tlen = 6usize;
        let x = Tensor::randn(0f32, 1.0, (1, MEL_DIM, tlen), &dev).unwrap();
        let mu = Tensor::randn(0f32, 1.0, (1, MEL_DIM, tlen), &dev).unwrap();
        let spks = Tensor::randn(0f32, 1.0, (1, MEL_DIM), &dev).unwrap();
        let cond = Tensor::zeros((1, MEL_DIM, tlen), DType::F32, &dev).unwrap();
        let out = est.forward(&x, &mu, 0.3, &spks, &cond).unwrap();
        assert_eq!(
            out.dims(),
            &[1, MEL_DIM, tlen],
            "velocity field is [1, 80, T]"
        );
        assert!(out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    // ---- synthetic-weight materializers (shapes exactly match flow.decoder.estimator.*) ----

    fn causal_block(vb: &VarBuilder, in_ch: usize, out_ch: usize) {
        let _ = vb.get((out_ch, in_ch, 3), "block.0.weight").unwrap();
        let _ = vb.get(out_ch, "block.0.bias").unwrap();
        let _ = vb.get(out_ch, "block.2.weight").unwrap();
        let _ = vb.get(out_ch, "block.2.bias").unwrap();
    }

    fn resnet(vb: &VarBuilder, dim: usize, dim_out: usize) {
        causal_block(&vb.pp("block1"), dim, dim_out);
        causal_block(&vb.pp("block2"), dim_out, dim_out);
        let _ = vb.get((dim_out, TIME_EMB_DIM), "mlp.1.weight").unwrap();
        let _ = vb.get(dim_out, "mlp.1.bias").unwrap();
        let _ = vb.get((dim_out, dim, 1), "res_conv.weight").unwrap();
        let _ = vb.get(dim_out, "res_conv.bias").unwrap();
    }

    fn transformer(vb: &VarBuilder) {
        let a = vb.pp("attn1");
        for name in ["to_q", "to_k", "to_v"] {
            let _ = a
                .get((TB_INNER, DEC_CH), &format!("{name}.weight"))
                .unwrap();
        }
        let _ = a.get((DEC_CH, TB_INNER), "to_out.0.weight").unwrap();
        let _ = a.get(DEC_CH, "to_out.0.bias").unwrap();
        let _ = vb.get((TB_FFN, DEC_CH), "ff.net.0.proj.weight").unwrap();
        let _ = vb.get(TB_FFN, "ff.net.0.proj.bias").unwrap();
        let _ = vb.get((DEC_CH, TB_FFN), "ff.net.2.weight").unwrap();
        let _ = vb.get(DEC_CH, "ff.net.2.bias").unwrap();
        for norm in ["norm1", "norm3"] {
            let _ = vb.get(DEC_CH, &format!("{norm}.weight")).unwrap();
            let _ = vb.get(DEC_CH, &format!("{norm}.bias")).unwrap();
        }
    }

    fn stage(vb: &VarBuilder, dim: usize, dim_out: usize) {
        resnet(&vb.pp("0"), dim, dim_out);
        for i in 0..N_TB {
            transformer(&vb.pp("1").pp(i));
        }
    }

    fn materialize_estimator_shapes(vb: &VarBuilder) {
        let _ = vb
            .get((TIME_EMB_DIM, DEC_IN), "time_mlp.linear_1.weight")
            .unwrap();
        let _ = vb.get(TIME_EMB_DIM, "time_mlp.linear_1.bias").unwrap();
        let _ = vb
            .get((TIME_EMB_DIM, TIME_EMB_DIM), "time_mlp.linear_2.weight")
            .unwrap();
        let _ = vb.get(TIME_EMB_DIM, "time_mlp.linear_2.bias").unwrap();
        // down block: resnet(320->256) + 4 transformers + downsample conv.
        let d = vb.pp("down_blocks").pp("0");
        stage(&d, DEC_IN, DEC_CH);
        let _ = d.get((DEC_CH, DEC_CH, 3), "2.weight").unwrap();
        let _ = d.get(DEC_CH, "2.bias").unwrap();
        // 12 mid blocks.
        for i in 0..MID_BLOCKS {
            stage(&vb.pp("mid_blocks").pp(i), DEC_CH, DEC_CH);
        }
        // up block: resnet(512->256) + 4 transformers + upsample conv.
        let u = vb.pp("up_blocks").pp("0");
        stage(&u, 2 * DEC_CH, DEC_CH);
        let _ = u.get((DEC_CH, DEC_CH, 3), "2.weight").unwrap();
        let _ = u.get(DEC_CH, "2.bias").unwrap();
        causal_block(&vb.pp("final_block"), DEC_CH, DEC_CH);
        let _ = vb.get((MEL_DIM, DEC_CH, 1), "final_proj.weight").unwrap();
        let _ = vb.get(MEL_DIM, "final_proj.bias").unwrap();
    }
}
