//! The MOSS-TTSD **backbone** (sc-13360) — a standard Qwen3 causal LM driving `channels` (8)
//! tied per-codebook prediction heads.
//!
//! At every sequence position the input is a **`channels`-wide** token: channel 0 is a text/speech
//! id and channels 1..channels-1 are the remaining audio codebooks. The backbone embeds each channel
//! through its own embedding (`model.embedding_list.{i}`), **sums** them, and runs the Qwen3 decoder
//! stack (`model.language_model.*`), returning the post-final-norm hidden state. Because
//! `tie_word_embeddings` is set, each channel's prediction head is its own embedding matrix
//! (`logits_i = hidden · embedding_list[i]ᵀ`) — channel 0 over the full text vocab (152697), each
//! audio channel over its 1025-wide codebook vocab. This mirrors the reference
//! `MossTTSDModel._prepare_multi_modal_inputs` + `Qwen3Model` + tied `MossTTSDForCausalLM.lm_heads`
//! exactly, weight-for-weight.
//!
//! The KV-cache prefill/step machinery (and the M=1 gemv duplicate-row trick that keeps a cached
//! single-token step byte-identical to full recompute) is the sibling MOSS-TTS-Realtime idiom
//! (sc-13417), reused here since the decoder stack is the identical Qwen3 layer.

use candle_audio::candle_core::{IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Module, RmsNorm, VarBuilder};

use crate::blocks::{causal_mask, rope_tables, rope_tables_at, Layer, LayerKv};
use crate::config::MossTtsdConfig;

/// One backbone input position: the `channels` (8) codebook ids — channel 0 is the text/speech id,
/// channels 1..channels-1 the remaining audio codebooks.
pub type Frame = Vec<u32>;

/// The backbone's KV cache (sc-13417): one growing [`LayerKv`] slot per decoder layer plus the
/// number of positions already cached (the RoPE offset for the next [`Backbone::step`]). A fresh
/// cache is required per AR run.
pub struct BackboneCache {
    layers: Vec<LayerKv>,
    offset: usize,
}

impl BackboneCache {
    fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![LayerKv::default(); num_layers],
            offset: 0,
        }
    }

    /// Positions currently cached (0 before the first prefill).
    pub fn offset(&self) -> usize {
        self.offset
    }
}

/// The Qwen3 backbone: the multi-channel input embeddings + the decoder stack + final norm, plus the
/// tied per-channel prediction heads (the embedding matrices themselves).
pub struct Backbone {
    /// Per-channel embeddings: `embeds[0]` over the text vocab, `embeds[1..]` over the codebook vocab.
    embeds: Vec<Embedding>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    head_dim: usize,
    rope_theta: f64,
    hidden_size: usize,
    channels: usize,
    device: candle_audio::candle_core::Device,
}

impl Backbone {
    /// Build the backbone from the checkpoint (`model.embedding_list.*` + `model.language_model.*`).
    /// `vb` is the snapshot-root VarBuilder (the full-model tensor namespace).
    pub fn new(cfg: &MossTtsdConfig, vb: VarBuilder) -> CandleResult<Self> {
        let model = vb.pp("model");
        let mut embeds = Vec::with_capacity(cfg.channels);
        // Channel 0: text + speech-codebook-0, over the full text vocab.
        embeds.push(candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            model.pp("embedding_list.0"),
        )?);
        // Channels 1..channels-1: the remaining audio codebooks, over the codebook vocab.
        for c in 1..cfg.channels {
            embeds.push(candle_nn::embedding(
                cfg.speech_vocab_size,
                cfg.hidden_size,
                model.pp(format!("embedding_list.{c}")),
            )?);
        }

        let block = cfg.block();
        let vb_l = model.pp("language_model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Layer::new(&block, vb_l.pp(i))?);
        }
        let norm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            model.pp("language_model.norm"),
        )?;
        Ok(Self {
            embeds,
            layers,
            norm,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            hidden_size: cfg.hidden_size,
            channels: cfg.channels,
            device: vb.device().clone(),
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The summed multi-channel input embeddings for a frame sequence → `[1, T, H]`.
    fn embed(&self, frames: &[Frame]) -> CandleResult<Tensor> {
        let t = frames.len();
        let mut sum: Option<Tensor> = None;
        for (c, embed) in self.embeds.iter().enumerate() {
            let ids: Vec<u32> = frames.iter().map(|f| f[c]).collect();
            let ids_t = Tensor::from_vec(ids, (1, t), &self.device)?;
            let e = embed.forward(&ids_t)?;
            sum = Some(match sum {
                Some(s) => (s + e)?,
                None => e,
            });
        }
        Ok(sum.expect("at least one channel"))
    }

    /// Full-sequence forward over `frames` → the post-norm hidden states `[1, T, H]`.
    pub fn forward(&self, frames: &[Frame]) -> CandleResult<Tensor> {
        let t = frames.len();
        let mut h = self.embed(frames)?;
        let (cos, sin) = rope_tables(&self.device, t, self.head_dim, self.rope_theta)?;
        let mask = if t > 1 {
            Some(causal_mask(&self.device, t, h.dtype())?)
        } else {
            None
        };
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, mask.as_ref())?;
        }
        self.norm.forward(&h)
    }

    /// Full-sequence forward, returning only the **last** position's hidden state `[1, 1, H]` — the
    /// stateless recompute path the [`kv_cache_matches_full_recompute`](Self) test compares against.
    pub fn forward_last(&self, frames: &[Frame]) -> CandleResult<Tensor> {
        let h = self.forward(frames)?;
        let t = h.dim(1)?;
        h.i((.., t - 1..t, ..))?.contiguous()
    }

    /// A fresh [`BackboneCache`] sized for this backbone's decoder stack.
    pub fn new_cache(&self) -> BackboneCache {
        BackboneCache::new(self.layers.len())
    }

    /// Run the decoder stack over `h` (`[1, T, H]` input embeddings), appending `T` new positions at
    /// `cache.offset..cache.offset + T`, then advance the offset and return the last position's
    /// post-norm hidden state `[1, 1, H]`. A single-token step (`T == 1`) is run as two duplicate
    /// rows to dodge Candle's M=1 gemv path (see [`crate::blocks`]); only the trailing real position
    /// is appended to the cache, so the result is byte-identical to [`forward_last`](Self::forward_last).
    fn run_cached(&self, h: Tensor, cache: &mut BackboneCache) -> CandleResult<Tensor> {
        let t = h.dim(1)?;
        let (mut h, cos, sin, mask) = if t == 1 {
            let h2 = Tensor::cat(&[&h, &h], 1)?;
            let (c1, s1) = rope_tables_at(
                &self.device,
                cache.offset,
                1,
                self.head_dim,
                self.rope_theta,
            )?;
            let cos = Tensor::cat(&[&c1, &c1], 0)?;
            let sin = Tensor::cat(&[&s1, &s1], 0)?;
            (h2, cos, sin, None)
        } else {
            let (cos, sin) = rope_tables_at(
                &self.device,
                cache.offset,
                t,
                self.head_dim,
                self.rope_theta,
            )?;
            let mask = if t > 1 {
                Some(causal_mask(&self.device, t, h.dtype())?)
            } else {
                None
            };
            (h, cos, sin, mask)
        };
        for (layer, slot) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward_cached(&h, &cos, &sin, mask.as_ref(), slot, t)?;
        }
        cache.offset += t;
        let h = self.norm.forward(&h)?;
        let last = h.dim(1)?;
        h.i((.., last - 1..last, ..))?.contiguous()
    }

    /// Prefill the prompt `frames` into a fresh `cache` (positions `0..frames.len()`, causal),
    /// returning the last prompt position's hidden state `[1, 1, H]`.
    pub fn prefill(&self, frames: &[Frame], cache: &mut BackboneCache) -> CandleResult<Tensor> {
        let h = self.embed(frames)?;
        self.run_cached(h, cache)
    }

    /// Append one new `frame` (the just-emitted position fed back) to `cache` and return its hidden
    /// state `[1, 1, H]`.
    pub fn step(&self, frame: &Frame, cache: &mut BackboneCache) -> CandleResult<Tensor> {
        let h = self.embed(std::slice::from_ref(frame))?;
        self.run_cached(h, cache)
    }

    /// The per-channel prediction logits at a `[1, 1, H]` hidden state, one `Vec<f32>` per channel
    /// (tied heads: `logits_c = hidden · embedding_list[c]ᵀ`). Channel 0 is `vocab_size` wide; each
    /// audio channel is `speech_vocab_size` wide.
    pub fn channel_logits(&self, hidden: &Tensor) -> CandleResult<Vec<Vec<f32>>> {
        let h = hidden.reshape((1, self.hidden_size))?; // [1, H]
        let mut out = Vec::with_capacity(self.channels);
        for embed in &self.embeds {
            let w = embed.embeddings(); // [V, H]
            let logits = h.matmul(&w.t()?)?; // [1, V]
            out.push(logits.reshape(((),))?.to_vec1::<f32>()?);
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_audio::candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    /// A structurally-faithful tiny config (real Qwen3 block shapes, small dims) — no real weights.
    pub(crate) fn tiny_cfg() -> MossTtsdConfig {
        MossTtsdConfig::from_json(
            r#"{
              "model_type": "moss_ttsd",
              "architectures": ["MossTTSDForCausalLM"],
              "vocab_size": 40, "hidden_size": 16, "intermediate_size": 32,
              "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
              "head_dim": 4, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
              "attention_bias": false, "bos_token_id": 6,
              "channels": 4, "speech_vocab_size": 8, "speech_pad_token": 7,
              "speech_token_range": [20, 40], "speech_eos_token": 39
            }"#,
        )
        .unwrap()
    }

    /// Build a backbone over a deterministically-seeded VarMap (exercises the real weight paths).
    pub(crate) fn tiny_backbone(cfg: &MossTtsdConfig) -> Backbone {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let backbone = Backbone::new(cfg, vb.clone()).unwrap();
        for (i, (_, var)) in varmap.data().lock().unwrap().iter().enumerate() {
            let t = var.as_tensor();
            let n = t.shape().elem_count();
            let vals: Vec<f32> = (0..n)
                .map(|j| (((i * 31 + j * 17) % 13) as f64 * 0.03 - 0.18) as f32)
                .collect();
            var.set(&Tensor::from_vec(vals, t.shape(), &Device::Cpu).unwrap())
                .unwrap();
        }
        backbone
    }

    fn frame(cfg: &MossTtsdConfig, text: u32, seed: u32) -> Frame {
        let mut f = vec![text];
        for c in 1..cfg.channels {
            f.push(
                (seed.wrapping_mul(7).wrapping_add(c as u32 * 3)) % cfg.speech_vocab_size as u32,
            );
        }
        f
    }

    /// The KV-cache path (prefill + steps) is numerically equivalent to the stateless full-sequence
    /// recompute at every AR position (sc-13417) — modulo float accumulation order.
    #[test]
    fn kv_cache_matches_full_recompute() {
        const TOL: f32 = 1e-4;
        let assert_close = |a: &Tensor, b: &Tensor, ctx: &str| {
            let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(av.len(), bv.len(), "{ctx}: length mismatch");
            let max = av
                .iter()
                .zip(&bv)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max);
            assert!(max < TOL, "{ctx}: max abs diff {max} exceeds {TOL}");
        };
        let cfg = tiny_cfg();
        let bb = tiny_backbone(&cfg);

        let prompt: Vec<Frame> = (0..5)
            .map(|k| frame(&cfg, k as u32 + 3, k as u32 + 1))
            .collect();
        let fed: Vec<Frame> = (0..6)
            .map(|k| {
                frame(
                    &cfg,
                    cfg.text_pad_id % cfg.vocab_size as u32,
                    (k as u32 + 100) * 5,
                )
            })
            .collect();

        let mut all = prompt.clone();
        let mut cache = bb.new_cache();
        let cached0 = bb.prefill(&prompt, &mut cache).unwrap();
        let recompute0 = bb.forward_last(&all).unwrap();
        assert_close(&cached0, &recompute0, "prefill vs forward_last");
        assert_eq!(cache.offset(), prompt.len());

        for (i, f) in fed.iter().enumerate() {
            all.push(f.clone());
            let cached = bb.step(f, &mut cache).unwrap();
            let recompute = bb.forward_last(&all).unwrap();
            assert_close(&cached, &recompute, &format!("step {i}"));
            assert_eq!(cache.offset(), all.len());
        }
    }

    /// The tied per-channel heads have the right width (channel 0 = text vocab, audio channels =
    /// codebook vocab) and produce finite logits.
    #[test]
    fn channel_logits_have_per_channel_widths() {
        let cfg = tiny_cfg();
        let bb = tiny_backbone(&cfg);
        let prompt: Vec<Frame> = (0..3)
            .map(|k| frame(&cfg, k as u32 + 1, k as u32 + 2))
            .collect();
        let hidden = bb.forward_last(&prompt).unwrap();
        let logits = bb.channel_logits(&hidden).unwrap();
        assert_eq!(logits.len(), cfg.channels);
        assert_eq!(
            logits[0].len(),
            cfg.vocab_size,
            "channel 0 spans the text vocab"
        );
        for (c, row) in logits.iter().enumerate().skip(1) {
            assert_eq!(
                row.len(),
                cfg.speech_vocab_size,
                "audio channel {c} spans the codebook vocab"
            );
        }
        assert!(logits.iter().all(|row| row.iter().all(|v| v.is_finite())));
    }
}
