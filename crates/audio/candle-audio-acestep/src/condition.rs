//! The ACE-Step 1.5 **`AceStepConditionEncoder`** (sc-12842) — assembles the DiT cross-attention
//! context from the prompt, lyric, and timbre streams, and owns the `silence_latent` buffer that
//! seeds the text-to-music source latents.
//!
//! Weight layout (verified against the pinned `condition_encoder/*.safetensors`):
//!
//! ```text
//!   text_projector            Linear(text_hidden 1024 → hidden 2048, no bias)
//!   lyric_encoder.embed_tokens Linear(1024 → 2048, bias)  ← projects the Qwen lyric embeddings
//!   lyric_encoder.layers.N     8× bidirectional block (to_q/to_k/to_v/to_out.0 + norm_q/norm_k,
//!                                 GQA 16/8 head_dim 128, RoPE, SwiGLU MLP, RMSNorms)
//!   lyric_encoder.norm         RMSNorm
//!   timbre_encoder.embed_tokens Linear(64 → 2048, bias)   ← real timbre latents (audio-to-audio)
//!   timbre_encoder.layers.N     4× bidirectional block
//!   timbre_encoder.norm         RMSNorm
//!   silence_latent            [1, 15000, 64]               ← tiled/cropped to the src latents
//! ```
//!
//! The fused context is `cat([text_proj(prompt), lyric_encoder(lyrics), timbre_encoder(timbre)])`
//! along the sequence, in the reference's pack order [lyric | timbre | text]. For pure
//! text-to-music the timbre stream is a fixed 30 s slice of the encoded `silence_latent`
//! (no reference audio); an absent lyric stream (instrumental) is simply omitted.

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use candle_audio::candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{linear, linear_b, rms_norm, Linear, Module, RmsNorm, VarBuilder};

use crate::config::ConditionEncoderConfig;

fn to_heads(x: &Tensor, num_heads: usize, head_dim: usize) -> CandleResult<Tensor> {
    let (b, l, _) = x.dims3()?;
    x.reshape((b, l, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()
}

fn from_heads(x: &Tensor) -> CandleResult<Tensor> {
    let (b, h, l, d) = x.dims4()?;
    x.transpose(1, 2)?.reshape((b, l, h * d))
}

fn repeat_kv(x: &Tensor, groups: usize) -> CandleResult<Tensor> {
    if groups == 1 {
        return x.contiguous();
    }
    let (b, h, l, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, groups, l, d))?
        .reshape((b, h * groups, l, d))?
        .contiguous()
}

fn rope_tables(
    head_dim: usize,
    len: usize,
    theta: f64,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);
    for pos in 0..len {
        for j in 0..half {
            let inv = 1.0 / theta.powf(2.0 * j as f64 / head_dim as f64);
            let a = pos as f64 * inv;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (len, half), device)?,
        Tensor::from_vec(sin, (len, half), device)?,
    ))
}

struct EncoderLayer {
    input_layernorm: RmsNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    post_attention_layernorm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    theta: f64,
}

impl EncoderLayer {
    fn new(cfg: &ConditionEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d = cfg.head_dim;
        let hidden = cfg.hidden_size;
        let sa = vb.pp("self_attn");
        Ok(Self {
            input_layernorm: rms_norm(hidden, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            to_q: linear_b(hidden, cfg.num_attention_heads * d, false, sa.pp("to_q"))?,
            to_k: linear_b(hidden, cfg.num_key_value_heads * d, false, sa.pp("to_k"))?,
            to_v: linear_b(hidden, cfg.num_key_value_heads * d, false, sa.pp("to_v"))?,
            to_out: linear_b(
                cfg.num_attention_heads * d,
                hidden,
                false,
                sa.pp("to_out.0"),
            )?,
            norm_q: rms_norm(d, cfg.rms_norm_eps, sa.pp("norm_q"))?,
            norm_k: rms_norm(d, cfg.rms_norm_eps, sa.pp("norm_k"))?,
            post_attention_layernorm: rms_norm(
                hidden,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            gate_proj: linear_b(hidden, cfg.intermediate_size, false, vb.pp("mlp.gate_proj"))?,
            up_proj: linear_b(hidden, cfg.intermediate_size, false, vb.pp("mlp.up_proj"))?,
            down_proj: linear_b(cfg.intermediate_size, hidden, false, vb.pp("mlp.down_proj"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: d,
            theta: cfg.rope_theta,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let device = x.device();
        let len = x.dim(1)?;
        let h = self.input_layernorm.forward(x)?;
        let q = self.norm_q.forward(&to_heads(
            &self.to_q.forward(&h)?,
            self.num_heads,
            self.head_dim,
        )?)?;
        let k = self.norm_k.forward(&to_heads(
            &self.to_k.forward(&h)?,
            self.num_kv_heads,
            self.head_dim,
        )?)?;
        let v = to_heads(&self.to_v.forward(&h)?, self.num_kv_heads, self.head_dim)?;
        let (cos, sin) = rope_tables(self.head_dim, len, self.theta, device)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // Bidirectional (encoder) attention — no causal mask.
        let att = candle_nn::ops::softmax_last_dim(&(q.matmul(&k.transpose(2, 3)?)? * scale)?)?;
        let attn = self
            .to_out
            .forward(&from_heads(&att.matmul(&v.contiguous()?)?)?)?;
        let x = (x + attn)?;
        let h = self.post_attention_layernorm.forward(&x)?;
        let ff = self
            .down_proj
            .forward(&(self.gate_proj.forward(&h)?.silu()? * self.up_proj.forward(&h)?)?)?;
        x + ff
    }
}

/// One `{lyric,timbre}_encoder` sub-stack: an input projection, N bidirectional layers, a norm.
struct Encoder {
    embed_tokens: Linear,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
}

impl Encoder {
    fn new(
        cfg: &ConditionEncoderConfig,
        in_dim: usize,
        n: usize,
        prefix: &str,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let root = vb.pp(prefix);
        let embed_tokens = linear(in_dim, cfg.hidden_size, root.pp("embed_tokens"))?;
        let vb_l = root.pp("layers");
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            layers.push(EncoderLayer::new(cfg, vb_l.pp(i))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, root.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Project raw inputs through `embed_tokens`, then run the contextual encoder stack.
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let mut x = self.embed_tokens.forward(input)?;
        for l in &self.layers {
            x = l.forward(&x)?;
        }
        self.norm.forward(&x)
    }
}

/// Pinned ACE-Step 1.5 text-to-music timbre window: 30 seconds × 25 latent frames/second.
pub const DEFAULT_FIXED_TIMBRE_FRAMES: usize = 750;

/// The assembled condition encoder.
pub struct ConditionEncoder {
    text_projector: Linear,
    lyric_encoder: Encoder,
    timbre_encoder: Encoder,
    silence_latent: Tensor, // [1, T0, acoustic]
    fixed_timbre_frames: usize,
    fixed_timbre_context: OnceLock<Tensor>,
    #[cfg(test)]
    fixed_timbre_compute_count: AtomicUsize,
    cfg: ConditionEncoderConfig,
}

impl ConditionEncoder {
    /// Compatibility constructor using the pinned checkpoint's 30-second fixed timbre window.
    pub fn new(cfg: &ConditionEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        Self::new_with_fixed_timbre_frames(cfg, DEFAULT_FIXED_TIMBRE_FRAMES, vb)
    }

    /// Construct with the fixed timbre frame count bound to encoder state.
    pub fn new_with_fixed_timbre_frames(
        cfg: &ConditionEncoderConfig,
        fixed_timbre_frames: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let text_projector = linear_b(cfg.text_hidden_dim, h, false, vb.pp("text_projector"))?;
        let lyric_encoder = Encoder::new(
            cfg,
            cfg.text_hidden_dim,
            cfg.num_lyric_encoder_hidden_layers,
            "lyric_encoder",
            vb.clone(),
        )?;
        let timbre_encoder = Encoder::new(
            cfg,
            cfg.timbre_hidden_dim,
            cfg.num_timbre_encoder_hidden_layers,
            "timbre_encoder",
            vb.clone(),
        )?;
        let silence_latent = vb.get_unchecked("silence_latent")?;
        Ok(Self {
            text_projector,
            lyric_encoder,
            timbre_encoder,
            silence_latent,
            fixed_timbre_frames,
            fixed_timbre_context: OnceLock::new(),
            #[cfg(test)]
            fixed_timbre_compute_count: AtomicUsize::new(0),
            cfg: cfg.clone(),
        })
    }

    /// The source latents `[1, latent_len, acoustic]` for text-to-music: the learned
    /// `silence_latent` tiled/cropped to the requested length.
    pub fn src_latents(&self, latent_len: usize, device: &Device) -> CandleResult<Tensor> {
        let (_, t0, c) = self.silence_latent.dims3()?;
        if latent_len <= t0 {
            self.silence_latent
                .narrow(1, 0, latent_len)?
                .to_device(device)
        } else {
            let reps = latent_len.div_ceil(t0);
            let tiled = Tensor::cat(&vec![&self.silence_latent; reps], 1)?;
            tiled
                .narrow(1, 0, latent_len)?
                .to_device(device)?
                .reshape((1, latent_len, c))
        }
    }

    /// Build the DiT cross-attention context `[1, S, hidden]` from the prompt hidden states, the
    /// lyric token embeddings (Qwen embedding lookup), and the pooled text-to-music timbre context.
    ///
    /// Stream order is the reference's `_pack_sequences` order — **lyric, then timbre, then
    /// text** — not text-first. Cross-attention itself is permutation-invariant over the context
    /// (permuting K and V rows identically permutes the softmax weights identically, and no
    /// positional encoding is applied to this sequence), so this ordering is not load-bearing for
    /// the maths; it is matched so the packed context is bit-comparable against the reference and
    /// so any future change that *does* become order-sensitive — a context RoPE, a positional
    /// bias, or attention-mask packing — starts from the correct layout.
    /// Compatibility entry point retaining the original arbitrary `timbre_frames` argument. Calls
    /// matching this encoder's bound fixed frame count use the cache; other frame counts preserve
    /// the previous behavior by computing their timbre context without caching it.
    pub fn encode(
        &self,
        text_hidden: &Tensor,
        lyric_embeds: Option<&Tensor>,
        timbre_frames: usize,
    ) -> CandleResult<Tensor> {
        if timbre_frames == self.fixed_timbre_frames {
            return self.encode_cached(text_hidden, lyric_embeds);
        }
        let timbre = self.compute_timbre_context(timbre_frames, text_hidden.device())?;
        self.fuse_context(text_hidden, lyric_embeds, &timbre)
    }

    /// Build the dynamic lyric/text context around the cached fixed pooled timbre row.
    pub fn encode_cached(
        &self,
        text_hidden: &Tensor,
        lyric_embeds: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let timbre = get_or_compute(&self.fixed_timbre_context, || {
            #[cfg(test)]
            self.fixed_timbre_compute_count
                .fetch_add(1, Ordering::Relaxed);
            self.compute_timbre_context(self.fixed_timbre_frames, text_hidden.device())
        })?;
        self.fuse_context(text_hidden, lyric_embeds, timbre)
    }

    fn compute_timbre_context(
        &self,
        timbre_frames: usize,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let timbre_in = self.src_latents(timbre_frames, device)?;
        let timbre = self.timbre_encoder.forward(&timbre_in)?;
        timbre.narrow(1, 0, 1) // CLS-like pooling: first position.
    }

    fn fuse_context(
        &self,
        text_hidden: &Tensor,
        lyric_embeds: Option<&Tensor>,
        timbre: &Tensor,
    ) -> CandleResult<Tensor> {
        let mut parts: Vec<Tensor> = Vec::new();
        if let Some(lyric) = lyric_embeds {
            parts.push(self.lyric_encoder.forward(lyric)?);
        }
        // Text-to-music timbre. The reference does NOT feed the learned `special_token` here — its
        // `AceStepTimbreEncoder::forward` never reads that parameter. It projects a fixed 30 s slice
        // of the VAE-encoded `silence_latent` through `embed_tokens`, runs the encoder stack, and
        // CLS-pools row 0. Feeding a bare 1-row token instead puts the timbre encoder far out of
        // distribution; the reference's own source notes that an OOD timbre input "produces
        // drone-like audio (observed on all text2music outputs)", which is exactly the broadband
        // drone this port emitted over every generated track.
        parts.push(timbre.clone());
        parts.push(self.text_projector.forward(text_hidden)?);
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 1)
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}

/// Stable fallible `OnceLock` pattern: inspect, compute with `?`, then publish without panicking.
/// A lost initialization race may discard one computed value, while every later call reuses the
/// published value.
fn get_or_compute<T, E>(
    cache: &OnceLock<T>,
    compute: impl FnOnce() -> std::result::Result<T, E>,
) -> std::result::Result<&T, E> {
    if let Some(value) = cache.get() {
        return Ok(value);
    }
    let built = compute()?;
    Ok(cache.get_or_init(|| built))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::{DType, Device};
    use std::collections::HashMap;

    #[test]
    fn repeated_encodes_compute_fixed_timbre_context_once() {
        let device = Device::Cpu;
        let cfg = ConditionEncoderConfig {
            hidden_size: 2,
            intermediate_size: 2,
            head_dim: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            num_lyric_encoder_hidden_layers: 0,
            num_timbre_encoder_hidden_layers: 0,
            text_hidden_dim: 2,
            timbre_hidden_dim: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            sliding_window: 8,
        };
        let mut weights = HashMap::new();
        weights.insert(
            "text_projector.weight".into(),
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap(),
        );
        weights.insert(
            "lyric_encoder.embed_tokens.weight".into(),
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap(),
        );
        weights.insert(
            "timbre_encoder.embed_tokens.weight".into(),
            Tensor::from_vec(vec![1.0_f32, 2.0], (2, 1), &device).unwrap(),
        );
        for prefix in ["lyric_encoder", "timbre_encoder"] {
            weights.insert(
                format!("{prefix}.embed_tokens.bias"),
                Tensor::zeros(2, DType::F32, &device).unwrap(),
            );
            weights.insert(
                format!("{prefix}.norm.weight"),
                Tensor::ones(2, DType::F32, &device).unwrap(),
            );
        }
        weights.insert(
            "silence_latent".into(),
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (1, 4, 1), &device).unwrap(),
        );
        let encoder = ConditionEncoder::new_with_fixed_timbre_frames(
            &cfg,
            3,
            VarBuilder::from_tensors(weights, DType::F32, &device),
        )
        .unwrap();
        let text_a = Tensor::from_vec(vec![1.0_f32, 0.0], (1, 1, 2), &device).unwrap();
        let lyric_a = Tensor::from_vec(vec![1.0_f32, 0.0], (1, 1, 2), &device).unwrap();
        let text_b = Tensor::from_vec(vec![0.0_f32, 1.0], (1, 1, 2), &device).unwrap();
        let lyric_b = Tensor::from_vec(vec![0.0_f32, 1.0], (1, 1, 2), &device).unwrap();

        // The compatibility API retains arbitrary frame behavior and does not poison the fixed
        // cache when the caller asks for a different timbre window.
        encoder.encode(&text_a, Some(&lyric_a), 2).unwrap();
        assert_eq!(
            encoder.fixed_timbre_compute_count.load(Ordering::Relaxed),
            0
        );

        let first = encoder.encode_cached(&text_a, Some(&lyric_a)).unwrap();
        let second = encoder.encode_cached(&text_b, Some(&lyric_b)).unwrap();
        assert_eq!(first.dims3().unwrap(), (1, 3, 2));
        assert_eq!(second.dims3().unwrap(), (1, 3, 2));
        assert_eq!(
            encoder.fixed_timbre_compute_count.load(Ordering::Relaxed),
            1
        );

        let rows = |context: &Tensor, row: usize| {
            context
                .narrow(1, row, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_ne!(
            rows(&first, 0),
            rows(&second, 0),
            "lyrics must stay dynamic"
        );
        assert_eq!(
            rows(&first, 1),
            rows(&second, 1),
            "only pooled timbre is cached"
        );
        assert_ne!(rows(&first, 2), rows(&second, 2), "text must stay dynamic");
    }
}
