//! The **T3** text→speech-token LM (sc-13222) — a faithful candle port of Chatterbox's
//! `models/t3/t3.py`: a Llama-520M backbone (30 layers, 16 heads, Llama-3 RoPE) driven on custom
//! `inputs_embeds` (the backbone's own `embed_tokens` is unused), with Chatterbox's text/speech
//! embeddings, learned positional embeddings, conditioning encoder (`T3CondEnc` — speaker
//! projection + emotion advisor + a Perceiver resampler over the prompt speech tokens), and the
//! speech-token head, decoded autoregressively with classifier-free guidance.
//!
//! Weight keys mirror `t3_cfg.safetensors` exactly: `tfmr.*` (the Llama backbone), `text_emb`,
//! `speech_emb`, `text_head`, `speech_head`, `text_pos_emb.emb`, `speech_pos_emb.emb`, and
//! `cond_enc.{spkr_enc,emotion_adv_fc,perceiver.*}`.
//!
//! The decode loop threads cooperative cancellation and per-step progress (the reference's HF
//! `generate` loop), and samples deterministically from a seeded RNG (the gen-core seed law).

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, ops::softmax_last_dim, rms_norm, Embedding,
    LayerNorm, Linear, Module, RmsNorm, VarBuilder,
};
use rand::rngs::StdRng;
use rand::Rng;

use crate::config::T3Config;

/// The speech-token conditioning bundle assembled by the provider from the request conditioning
/// (the port of `T3Cond`). The speaker vector is the `chatterbox_ve` 256-d embedding; the prompt
/// speech tokens are the s3tokenizer codes of the reference clip (may be empty when only a bare
/// voice embedding is supplied — the Perceiver then sees an empty prompt, matching the reference's
/// `cond_prompt_speech_emb is None` branch).
pub struct T3Cond {
    /// `[256]` speaker embedding (raw `chatterbox_ve` vector).
    pub speaker_emb: Vec<f32>,
    /// The reference clip's S3 speech tokens (`speech_cond_prompt_len` of them, or empty).
    pub cond_prompt_speech_tokens: Vec<u32>,
    /// The emotion-advisor scalar (`exaggeration`, default 0.5).
    pub emotion_adv: f32,
}

/// One Llama decoder layer (RMSNorm → self-attention with RoPE → RMSNorm → SwiGLU MLP), holding a
/// per-layer KV cache for autoregressive decode.
struct Layer {
    input_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    post_attention_layernorm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    kv: Option<(Tensor, Tensor)>,
}

impl Layer {
    fn new(cfg: &T3Config, vb: VarBuilder) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let d = cfg.head_dim;
        let heads = cfg.num_attention_heads;
        let attn = vb.pp("self_attn");
        let mlp = vb.pp("mlp");
        Ok(Self {
            input_layernorm: rms_norm(h, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            q_proj: linear_no_bias(h, heads * d, attn.pp("q_proj"))?,
            k_proj: linear_no_bias(h, heads * d, attn.pp("k_proj"))?,
            v_proj: linear_no_bias(h, heads * d, attn.pp("v_proj"))?,
            o_proj: linear_no_bias(heads * d, h, attn.pp("o_proj"))?,
            post_attention_layernorm: rms_norm(
                h,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            gate_proj: linear_no_bias(h, cfg.intermediate_size, mlp.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, cfg.intermediate_size, mlp.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, h, mlp.pp("down_proj"))?,
            num_heads: heads,
            head_dim: d,
            kv: None,
        })
    }

    fn reset_cache(&mut self) {
        self.kv = None;
    }

    /// Forward one chunk of `l` positions starting at absolute offset `offset`. `cos`/`sin` are the
    /// full RoPE tables; `mask` is the additive causal mask over `[offset+l, offset+l]` restricted
    /// to the new rows, or `None` for a single-token step.
    fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        offset: usize,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let q = self
            .q_proj
            .forward(&h)?
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(&h)?
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(&h)?
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // RoPE over the positions this chunk occupies (`offset..offset+l`).
        let cos_l = cos.narrow(0, offset, l)?;
        let sin_l = sin.narrow(0, offset, l)?;
        let q = candle_nn::rotary_emb::rope(&q, &cos_l, &sin_l)?;
        let mut k = candle_nn::rotary_emb::rope(&k, &cos_l, &sin_l)?;
        let mut v = v;

        // Append to the KV cache.
        if let Some((pk, pv)) = &self.kv {
            k = Tensor::cat(&[pk, &k], 2)?.contiguous()?;
            v = Tensor::cat(&[pv, &v], 2)?.contiguous()?;
        }
        self.kv = Some((k.clone(), v.clone()));

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        if let Some(m) = mask {
            att = att.broadcast_add(m)?;
        }
        let att = softmax_last_dim(&att)?;
        let out =
            att.matmul(&v)?
                .transpose(1, 2)?
                .reshape((b, l, self.num_heads * self.head_dim))?;
        let attn_out = self.o_proj.forward(&out)?;
        let x = (residual + attn_out)?;

        // SwiGLU MLP.
        let residual = &x;
        let h = self.post_attention_layernorm.forward(&x)?;
        let gate = self.gate_proj.forward(&h)?.silu()?;
        let up = self.up_proj.forward(&h)?;
        let mlp = self.down_proj.forward(&(gate * up)?)?;
        residual + mlp
    }
}

/// The Perceiver resampler over the prompt speech embeddings (`cond_enc.perceiver`): a shared-`norm`
/// cross/self attention (`AttentionBlock2`) that maps a variable-length prompt `[B, N, 1024]` to a
/// fixed `[B, 32, 1024]` via a learned query, then a self-attention refine.
struct Perceiver {
    pre_attention_query: Tensor, // [1, 32, 1024]
    norm: LayerNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    proj_out: Linear,
    num_heads: usize,
    channels: usize,
}

impl Perceiver {
    fn new(dim: usize, vb: VarBuilder) -> CandleResult<Self> {
        let attn = vb.pp("attn");
        Ok(Self {
            pre_attention_query: vb.get((1, 32, dim), "pre_attention_query")?,
            norm: layer_norm(dim, 1e-5, attn.pp("norm"))?,
            to_q: linear(dim, dim, attn.pp("to_q"))?,
            to_k: linear(dim, dim, attn.pp("to_k"))?,
            to_v: linear(dim, dim, attn.pp("to_v"))?,
            proj_out: linear(dim, dim, attn.pp("proj_out"))?,
            num_heads: 4,
            channels: dim,
        })
    }

    fn split(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let hd = self.channels / self.num_heads;
        x.reshape((b, l, self.num_heads, hd))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// `AttentionBlock2.forward(x1, x2)`: q from norm(x1), k/v from norm(x2), attention, proj_out,
    /// residual on x1.
    fn attn_block(&self, x1: &Tensor, x2: &Tensor) -> CandleResult<Tensor> {
        let x1n = self.norm.forward(x1)?;
        let x2n = self.norm.forward(x2)?;
        let q = self.split(&self.to_q.forward(&x1n)?)?;
        let k = self.split(&self.to_k.forward(&x2n)?)?;
        let v = self.split(&self.to_v.forward(&x2n)?)?;
        let hd = self.channels / self.num_heads;
        let scale = (hd as f64).powf(-0.5);
        let att = softmax_last_dim(&(q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?)?;
        let (b, _, l1, _) = q.dims4()?;
        let out = att
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, l1, self.channels))?;
        let h = self.proj_out.forward(&out)?;
        x1 + h
    }

    fn forward(&self, h: &Tensor) -> CandleResult<Tensor> {
        let b = h.dim(0)?;
        let query = self
            .pre_attention_query
            .broadcast_as((b, 32, self.channels))?;
        let pre_att = self.attn_block(&query, h)?;
        self.attn_block(&pre_att, &pre_att)
    }
}

/// The full T3 LM.
pub struct T3 {
    cfg: T3Config,
    text_emb: Embedding,
    speech_emb: Embedding,
    text_pos_emb: Tensor,   // [text_pos_len, dim]
    speech_pos_emb: Tensor, // [speech_pos_len, dim]
    speech_head: Linear,
    spkr_enc: Linear,
    emotion_adv_fc: Linear,
    perceiver: Perceiver,
    layers: Vec<Layer>,
    norm: RmsNorm,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl T3 {
    /// Build the T3 LM from a `t3_cfg.safetensors` VarBuilder.
    pub fn new(cfg: &T3Config, vb: VarBuilder) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let text_emb = embedding(cfg.text_tokens_dict_size, h, vb.pp("text_emb"))?;
        let speech_emb = embedding(cfg.speech_tokens_dict_size, h, vb.pp("speech_emb"))?;
        let text_pos_emb = vb
            .pp("text_pos_emb")
            .pp("emb")
            .get((cfg.text_pos_len(), h), "weight")?;
        let speech_pos_emb = vb
            .pp("speech_pos_emb")
            .pp("emb")
            .get((cfg.speech_pos_len(), h), "weight")?;
        let speech_head = linear_no_bias(h, cfg.speech_tokens_dict_size, vb.pp("speech_head"))?;
        let cond = vb.pp("cond_enc");
        let spkr_enc = linear(cfg.speaker_embed_size, h, cond.pp("spkr_enc"))?;
        let emotion_adv_fc = linear_no_bias(1, h, cond.pp("emotion_adv_fc"))?;
        let perceiver = Perceiver::new(h, cond.pp("perceiver"))?;

        let tfmr = vb.pp("tfmr");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Layer::new(cfg, tfmr.pp("layers").pp(i))?);
        }
        let norm = rms_norm(h, cfg.rms_norm_eps, tfmr.pp("norm"))?;

        let (cos, sin) =
            llama3_rope_tables(cfg, cfg.max_position_embeddings.min(8192), vb.device())?;

        Ok(Self {
            cfg: *cfg,
            text_emb,
            speech_emb,
            text_pos_emb,
            speech_pos_emb,
            speech_head,
            spkr_enc,
            emotion_adv_fc,
            perceiver,
            layers,
            norm,
            cos,
            sin,
            device: vb.device().clone(),
        })
    }

    /// The conditioning prefix embeddings `[1, len_cond, dim]` (the port of
    /// `T3CondEnc.forward`): speaker projection, then (optionally Perceiver-resampled) prompt
    /// speech embeddings, then the emotion-advisor row.
    fn prepare_conditioning(&self, cond: &T3Cond) -> CandleResult<Tensor> {
        let h = self.cfg.hidden_size;
        let spk = Tensor::from_vec(
            cond.speaker_emb.clone(),
            (1, self.cfg.speaker_embed_size),
            &self.device,
        )?;
        let cond_spkr = self.spkr_enc.forward(&spk)?.reshape((1, 1, h))?;

        let mut parts = vec![cond_spkr];
        if !cond.cond_prompt_speech_tokens.is_empty() {
            let n = cond.cond_prompt_speech_tokens.len();
            let ids =
                Tensor::from_vec(cond.cond_prompt_speech_tokens.clone(), (1, n), &self.device)?;
            let prompt_emb = self.speech_emb.forward(&ids)?; // [1, n, dim]
            let resampled = self.perceiver.forward(&prompt_emb)?; // [1, 32, dim]
            parts.push(resampled);
        }
        let emo = Tensor::from_vec(vec![cond.emotion_adv], (1, 1), &self.device)?;
        let cond_emotion = self.emotion_adv_fc.forward(&emo)?.reshape((1, 1, h))?;
        parts.push(cond_emotion);

        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 1)
    }

    fn reset_cache(&mut self) {
        for l in &mut self.layers {
            l.reset_cache();
        }
    }

    fn backbone(&mut self, embeds: &Tensor, offset: usize) -> CandleResult<Tensor> {
        let (_, l, _) = embeds.dims3()?;
        let mask = if l > 1 {
            Some(causal_mask(l, offset, &self.device, embeds.dtype())?)
        } else {
            None
        };
        let mut h = embeds.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, &self.cos, &self.sin, offset, mask.as_ref())?;
        }
        self.norm.forward(&h)
    }

    /// Autoregressively decode speech tokens for `text_tokens` (already SOT/EOT-wrapped) under the
    /// `cond` conditioning, with classifier-free guidance (`cfg_weight`). Returns the raw speech
    /// token ids (including any trailing stop token). `progress(step)` is called each decode step;
    /// `cancel()` aborts (returning `Ok(None)`).
    #[allow(clippy::too_many_arguments)]
    pub fn inference(
        &mut self,
        cond: &T3Cond,
        text_tokens: &[u32],
        cfg_weight: f32,
        temperature: f32,
        top_p: f32,
        min_p: f32,
        repetition_penalty: f32,
        max_new_tokens: usize,
        rng: &mut StdRng,
        progress: &mut dyn FnMut(usize),
        cancel: &dyn Fn() -> bool,
    ) -> CandleResult<Option<Vec<u32>>> {
        self.reset_cache();
        let use_cfg = cfg_weight > 0.0;
        let batch = if use_cfg { 2 } else { 1 };

        // Conditioning prefix (shared across the CFG batch).
        let cond_emb = self.prepare_conditioning(cond)?; // [1, len_cond, dim]

        // Text embeddings + learned positions. For CFG the uncond row zeroes the text embeddings.
        let text_ids =
            Tensor::from_vec(text_tokens.to_vec(), (1, text_tokens.len()), &self.device)?;
        let mut text_emb = self.text_emb.forward(&text_ids)?; // [1, T, dim]
        let text_pos = self
            .text_pos_emb
            .narrow(0, 0, text_tokens.len())?
            .unsqueeze(0)?;
        text_emb = text_emb.broadcast_add(&text_pos)?;

        // Initial speech token: a single start-of-speech, position 0.
        let start = self.cfg.start_speech_token;
        let bos_ids = Tensor::from_vec(vec![start], (1, 1), &self.device)?;
        let bos_emb = self
            .speech_emb
            .forward(&bos_ids)?
            .broadcast_add(&self.speech_pos_emb.narrow(0, 0, 1)?.unsqueeze(0)?)?;

        // Assemble the per-row prefix and stack into the CFG batch.
        let rows: Vec<Tensor> = (0..batch)
            .map(|r| {
                let te = if use_cfg && r == 1 {
                    text_emb.zeros_like()?
                } else {
                    text_emb.clone()
                };
                Tensor::cat(&[&cond_emb, &te, &bos_emb], 1)
            })
            .collect::<CandleResult<_>>()?;
        let refs: Vec<&Tensor> = rows.iter().collect();
        let mut inputs = Tensor::cat(&refs, 0)?; // [batch, len_cond+T+1, dim]

        let mut offset = 0usize;
        let mut generated: Vec<u32> = Vec::new();
        let mut seen = vec![false; self.cfg.speech_tokens_dict_size];

        for step in 0..max_new_tokens {
            if cancel() {
                return Ok(None);
            }
            let hidden = self.backbone(&inputs, offset)?;
            offset += inputs.dim(1)?;
            // Logits of the last position for each CFG row.
            let last = hidden.narrow(1, hidden.dim(1)? - 1, 1)?; // [batch, 1, dim]
            let logits = self.speech_head.forward(&last)?.squeeze(1)?; // [batch, V]

            let combined = if use_cfg {
                let condl = logits.narrow(0, 0, 1)?;
                let uncondl = logits.narrow(0, 1, 1)?;
                (&condl + ((&condl - &uncondl)? * cfg_weight as f64)?)?
            } else {
                logits
            };
            let mut lg: Vec<f32> = combined
                .reshape((self.cfg.speech_tokens_dict_size,))?
                .to_vec1()?;

            apply_repetition_penalty(&mut lg, &seen, repetition_penalty);
            if temperature != 1.0 && temperature > 0.0 {
                for x in lg.iter_mut() {
                    *x /= temperature;
                }
            }
            let mut probs = softmax_vec(&lg);
            apply_min_p(&mut probs, min_p);
            apply_top_p(&mut probs, top_p);
            let next = sample_multinomial(&probs, rng);

            progress(step + 1);
            generated.push(next);
            if next as usize == self.cfg.stop_speech_token as usize {
                break;
            }
            seen[next as usize] = true;

            // Next-token embedding at fixed position `step + 1`.
            let nid = Tensor::from_vec(vec![next], (1, 1), &self.device)?;
            let pos = self.speech_pos_emb.narrow(0, step + 1, 1)?.unsqueeze(0)?;
            let ne = self.speech_emb.forward(&nid)?.broadcast_add(&pos)?; // [1,1,dim]
            inputs = if use_cfg {
                Tensor::cat(&[&ne, &ne], 0)?
            } else {
                ne
            };
        }
        Ok(Some(generated))
    }
}

/// Llama-3 RoPE cos/sin tables `[seq, head_dim/2]` with the standard long-context frequency remap.
fn llama3_rope_tables(
    cfg: &T3Config,
    seq: usize,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let d = cfg.head_dim;
    let half = d / 2;
    let sc = cfg.rope_scaling;
    let low_wavelen = sc.original_max_position_embeddings as f64 / sc.low_freq_factor;
    let high_wavelen = sc.original_max_position_embeddings as f64 / sc.high_freq_factor;
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let freq = 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / d as f64);
        let wavelen = 2.0 * std::f64::consts::PI / freq;
        let adj = if wavelen > low_wavelen {
            freq / sc.factor
        } else if wavelen < high_wavelen {
            freq
        } else {
            let smooth = (sc.original_max_position_embeddings as f64 / wavelen
                - sc.low_freq_factor)
                / (sc.high_freq_factor - sc.low_freq_factor);
            (1.0 - smooth) * freq / sc.factor + smooth * freq
        };
        inv_freq.push(adj);
    }
    let mut cos = Vec::with_capacity(seq * half);
    let mut sin = Vec::with_capacity(seq * half);
    for pos in 0..seq {
        for &f in &inv_freq {
            let a = pos as f64 * f;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq, half), device)?,
        Tensor::from_vec(sin, (seq, half), device)?,
    ))
}

/// Additive causal mask for `l` new query rows at absolute `offset` attending over `offset+l` keys.
fn causal_mask(l: usize, offset: usize, device: &Device, dtype: DType) -> CandleResult<Tensor> {
    let total = offset + l;
    let mut data = vec![0f32; l * total];
    for i in 0..l {
        let qpos = offset + i;
        for j in 0..total {
            if j > qpos {
                data[i * total + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, l, total), device)?.to_dtype(dtype)
}

/// Divide (positive-logit) / multiply (negative-logit) the logits of already-seen tokens by
/// `penalty` — the HF `RepetitionPenaltyLogitsProcessor`.
fn apply_repetition_penalty(logits: &mut [f32], seen: &[bool], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for (i, l) in logits.iter_mut().enumerate() {
        if seen[i] {
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum > 0.0 {
        for e in exp.iter_mut() {
            *e /= sum;
        }
    }
    exp
}

/// Min-p filtering: zero out tokens with probability below `min_p · max_prob`, then renormalize.
fn apply_min_p(probs: &mut [f32], min_p: f32) {
    if min_p <= 0.0 {
        return;
    }
    let max = probs.iter().copied().fold(0.0f32, f32::max);
    let thresh = min_p * max;
    let mut sum = 0.0;
    for p in probs.iter_mut() {
        if *p < thresh {
            *p = 0.0;
        }
        sum += *p;
    }
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// Nucleus (top-p) filtering: keep the smallest set of tokens whose cumulative probability first
/// reaches `top_p`, zero the rest, renormalize.
fn apply_top_p(probs: &mut [f32], top_p: f32) {
    if top_p >= 1.0 || top_p <= 0.0 {
        return;
    }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0f32;
    let mut keep = vec![false; probs.len()];
    for &i in &idx {
        keep[i] = true;
        cum += probs[i];
        if cum >= top_p {
            break;
        }
    }
    let mut sum = 0.0;
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep[i] {
            *p = 0.0;
        }
        sum += *p;
    }
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

/// Sample one index from a (normalized) probability vector using the seeded RNG.
fn sample_multinomial(probs: &[f32], rng: &mut StdRng) -> u32 {
    let r: f32 = rng.random_range(0.0..1.0);
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Drop T3's special/BOS/EOS speech tokens (ids `>= SPEECH_VOCAB_SIZE = 6561`), yielding the real
/// S3 speech-token sequence S3Gen consumes (the reference's `speech_tokens[speech_tokens < 6561]`).
pub fn strip_special_speech_tokens(tokens: &[u32]) -> Vec<u32> {
    tokens
        .iter()
        .copied()
        .filter(|&t| (t as usize) < crate::config::SPEECH_VOCAB_SIZE)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_removes_special_tokens() {
        // 6561 = start_speech, 6562 = stop_speech, both dropped; real codes kept.
        let out = strip_special_speech_tokens(&[6561, 3, 100, 6560, 6562, 8000]);
        assert_eq!(out, vec![3, 100, 6560]);
    }

    #[test]
    fn softmax_and_top_p_are_consistent() {
        let p = softmax_vec(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let mut q = p.clone();
        apply_top_p(&mut q, 0.5);
        // top_p keeps only the largest until cumulative >= 0.5 (here the top element ~0.665).
        assert!(q[2] > 0.0 && q[0] == 0.0);
        assert!((q.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn min_p_prunes_low_probability_mass() {
        let mut p = vec![0.7f32, 0.2, 0.1];
        apply_min_p(&mut p, 0.5); // threshold = 0.35 → keep only 0.7
        assert!(p[0] > 0.0 && p[1] == 0.0 && p[2] == 0.0);
    }

    #[test]
    fn repetition_penalty_pushes_seen_tokens_down() {
        let mut l = vec![2.0f32, -2.0, 1.0];
        apply_repetition_penalty(&mut l, &[true, true, false], 2.0);
        assert_eq!(l, vec![1.0, -4.0, 1.0]);
    }

    #[test]
    fn llama3_rope_tables_have_expected_shape() {
        let cfg = T3Config::LLAMA_520M;
        let (cos, sin) = llama3_rope_tables(&cfg, 16, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[16, cfg.head_dim / 2]);
        assert_eq!(sin.dims(), &[16, cfg.head_dim / 2]);
    }
}
