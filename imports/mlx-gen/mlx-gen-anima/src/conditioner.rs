//! `AnimaTextConditioner` — transcribed from diffusers
//! `condition_embedder_anima.py::AnimaTextConditioner`. Maps T5 token ids (learned query tokens via
//! `nn.Embedding(32128, 1024)`) + Qwen3 hidden states → the DiT's `encoder_hidden_states`. 6 blocks of
//! `[self-attn (RoPE θ=10000) → cross-attn into Qwen3 states → GELU MLP]`, then `out_proj` + RMSNorm,
//! mask-multiply, and **right-pad to 512** so the DiT always sees exactly 512 text tokens.
//!
//! Weight keys are the `net.llm_adapter.`-**stripped** names (`blocks.N.*`, `embed.weight`,
//! `out_proj.*`, `norm.weight`) — identical to the diffusers `AnimaTextConditioner` state dict, which
//! the Anima convert script loads with `strict=True` (no rename).

use mlx_rs::error::Result as MlxResult;
use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, zeros_dtype};
use mlx_rs::transforms::checkpoint;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{prefixed_paths, AdaptableHost, AdaptableLinear};
use mlx_gen::nn::{apply_text_rope, gelu_exact, TextRope, TokenEmbedding};
use mlx_gen::weights::{join, Weights};
use mlx_gen::Result;

use crate::config::ConditionerConfig;

fn lin(w: &Weights, name: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, name, bias, mlx_gen::quant::DEFAULT_GROUP_SIZE)
}

/// `(cos, sin)` RoPE tables, one pair per sequence length.
type Rope<'a> = (&'a Array, &'a Array);

/// `AnimaTextConditionerAttention`: q/k/v/o projections (no bias), per-head q/k RMSNorm (eps 1e-6),
/// half-split RoPE. Query positions and key positions can differ (cross-attn: target vs source).
struct Attention {
    q_proj: AdaptableLinear,
    k_proj: AdaptableLinear,
    v_proj: AdaptableLinear,
    o_proj: AdaptableLinear,
    q_norm: Array,
    k_norm: Array,
    heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
    /// sc-10576 — SDPA-segment gradient checkpointing (see the DiT `Attention::ckpt_sdpa`). The
    /// conditioner is never whole-block checkpointed (it runs in the traced graph so its 60 adapter
    /// factors get gradients), so — like z-image's refiner leg — its attention segment ckpt is left ON
    /// during training to bound the retained `[heads, S, S]` probability matrix. Numerically identical
    /// to the retained backward; inference never sets it.
    ckpt_sdpa: bool,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &ConditionerConfig) -> Result<Self> {
        let hd = cfg.head_dim() as i32;
        Ok(Self {
            q_proj: lin(w, &join(prefix, "q_proj"), false)?,
            k_proj: lin(w, &join(prefix, "k_proj"), false)?,
            v_proj: lin(w, &join(prefix, "v_proj"), false)?,
            o_proj: lin(w, &join(prefix, "o_proj"), false)?,
            q_norm: w.require(&join(prefix, "q_norm.weight"))?.clone(),
            k_norm: w.require(&join(prefix, "k_norm.weight"))?.clone(),
            heads: cfg.num_attention_heads as i32,
            head_dim: hd,
            scale: (hd as f32).powf(-0.5),
            eps: cfg.norm_eps,
            ckpt_sdpa: false,
        })
    }

    fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.ckpt_sdpa = on;
    }

    /// `hidden`: `[B, Sq, D]` (query source). `encoder`: `Some([B, Sk, Ctx])` (cross) or `None`
    /// (self → `hidden`). `q_rope`/`k_rope` apply half-split RoPE with (possibly) different positions.
    /// Batch is 1 in this pipeline, so all tokens are real ⇒ no attention mask needed.
    fn forward(
        &self,
        hidden: &Array,
        encoder: Option<&Array>,
        q_rope: Rope,
        k_rope: Rope,
    ) -> Result<Array> {
        let hsh = hidden.shape();
        let (b, sq) = (hsh[0], hsh[1]);
        let kv_src = encoder.unwrap_or(hidden);
        let sk = kv_src.shape()[1];

        let q = self
            .q_proj
            .forward(hidden)?
            .reshape(&[b, sq, self.heads, self.head_dim])?;
        let k = self
            .k_proj
            .forward(kv_src)?
            .reshape(&[b, sk, self.heads, self.head_dim])?;
        let v = self
            .v_proj
            .forward(kv_src)?
            .reshape(&[b, sk, self.heads, self.head_dim])?;

        let q = rms_norm(&q, &self.q_norm, self.eps)?;
        let k = rms_norm(&k, &self.k_norm, self.eps)?;
        let q = apply_text_rope(&q, q_rope.0, q_rope.1)?;
        let k = apply_text_rope(&k, k_rope.0, k_rope.1)?;

        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        let o = if self.ckpt_sdpa {
            // sc-10576: checkpoint the SDPA (q/k/v threaded, scale captured) — recompute the seq²
            // attention in the backward instead of retaining it. Numerically identical.
            let scale = self.scale;
            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                Ok(vec![scaled_dot_product_attention(
                    &inp[0], &inp[1], &inp[2], scale, None, None,
                )?])
            });
            seg(&[q, k, v])?.into_iter().next().ok_or_else(|| {
                mlx_gen::Error::Msg("anima conditioner: checkpoint SDPA produced no output".into())
            })?
        } else {
            scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?
        };
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, sq, self.heads * self.head_dim])?;
        self.o_proj.forward(&o)
    }
}

/// `AnimaTextConditionerBlock`: pre-RMSNorm self-attn → cross-attn → GELU MLP (all residual).
struct Block {
    norm_self_attn: Array,
    self_attn: Attention,
    norm_cross_attn: Array,
    cross_attn: Attention,
    norm_mlp: Array,
    mlp_in: AdaptableLinear,  // mlp.0 (bias)
    mlp_out: AdaptableLinear, // mlp.2 (bias)
    eps: f32,
}

impl Block {
    fn from_weights(w: &Weights, prefix: &str, cfg: &ConditionerConfig) -> Result<Self> {
        Ok(Self {
            norm_self_attn: w.require(&join(prefix, "norm_self_attn.weight"))?.clone(),
            self_attn: Attention::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            norm_cross_attn: w.require(&join(prefix, "norm_cross_attn.weight"))?.clone(),
            cross_attn: Attention::from_weights(w, &join(prefix, "cross_attn"), cfg)?,
            norm_mlp: w.require(&join(prefix, "norm_mlp.weight"))?.clone(),
            mlp_in: lin(w, &join(prefix, "mlp.0"), true)?,
            mlp_out: lin(w, &join(prefix, "mlp.2"), true)?,
            eps: cfg.norm_eps,
        })
    }

    fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.self_attn.set_sdpa_checkpoint(on);
        self.cross_attn.set_sdpa_checkpoint(on);
    }

    fn forward(
        &self,
        hidden: &Array,
        encoder: &Array,
        target_rope: Rope,
        source_rope: Rope,
    ) -> Result<Array> {
        // self-attn: q and k both use the target (T5 query-token) positions.
        let normed = rms_norm(hidden, &self.norm_self_attn, self.eps)?;
        let h = add(
            hidden,
            &self
                .self_attn
                .forward(&normed, None, target_rope, target_rope)?,
        )?;
        // cross-attn: q uses target positions, k uses source (Qwen3) positions.
        let normed = rms_norm(&h, &self.norm_cross_attn, self.eps)?;
        let h = add(
            &h,
            &self
                .cross_attn
                .forward(&normed, Some(encoder), target_rope, source_rope)?,
        )?;
        // GELU MLP.
        let normed = rms_norm(&h, &self.norm_mlp, self.eps)?;
        let mlp = self
            .mlp_out
            .forward(&gelu_exact(&self.mlp_in.forward(&normed)?)?)?;
        Ok(add(&h, &mlp)?)
    }
}

/// The full `AnimaTextConditioner`.
pub struct AnimaTextConditioner {
    embed: TokenEmbedding,
    blocks: Vec<Block>,
    out_proj: AdaptableLinear, // bias
    norm: Array,
    rope: TextRope,
    cfg: ConditionerConfig,
}

impl AnimaTextConditioner {
    /// `prefix` is the checkpoint root of the adapter (`"net.llm_adapter"`).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: ConditionerConfig) -> Result<Self> {
        let embed = mlx_gen::quant::embedding(
            w,
            &join(prefix, "embed"),
            mlx_gen::quant::DEFAULT_GROUP_SIZE,
        )?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::from_weights(
                w,
                &join(prefix, &format!("blocks.{i}")),
                &cfg,
            )?);
        }
        Ok(Self {
            embed,
            blocks,
            out_proj: lin(w, &join(prefix, "out_proj"), true)?,
            norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            rope: TextRope::new(cfg.head_dim() as i32, cfg.rope_theta),
            cfg,
        })
    }

    /// `source_hidden`: `[B, Ss, source_dim]` Qwen3 states (already mask-multiplied), in `dtype`.
    /// `target_ids`: `[B, St]` int32 T5 token ids (`St <= min_sequence_length`). Returns
    /// `[B, min_sequence_length(=512), target_dim]` in `dtype`.
    pub fn forward(
        &self,
        source_hidden: &Array,
        target_ids: &Array,
        dtype: Dtype,
    ) -> Result<Array> {
        self.forward_weighted(source_hidden, target_ids, None, dtype)
    }

    /// [`forward`](Self::forward) with **ComfyUI-style prompt weighting** applied to the T5
    /// query-token path (sc-10566). `target_weights`, when present, is a per-target-token weight
    /// vector aligned 1:1 with `target_ids` (the real, unpadded T5 sequence): each token's full
    /// `target_dim` OUTPUT vector is scaled by its scalar weight **before** the right-pad to 512 —
    /// exactly ComfyUI's `out = self.llm_adapter(...); out = out * t5xxl_weights`
    /// (`comfy/ldm/anima/model.py:198-206`, with `t5xxl_weights` reshaped to `[1, St, 1]` in
    /// `comfy/model_base.py:1470`). The Qwen source path is untouched (its weights are pinned to
    /// `1.0` upstream), and an all-`1.0` weight vector is a strict no-op (identical to [`forward`]).
    pub fn forward_weighted(
        &self,
        source_hidden: &Array,
        target_ids: &Array,
        target_weights: Option<&[f32]>,
        dtype: Dtype,
    ) -> Result<Array> {
        let ssh = source_hidden.shape();
        let (b, ss) = (ssh[0], ssh[1]);
        let st = target_ids.shape()[1];

        // learned query tokens from T5 ids (in_proj is Identity: model_dim == target_dim).
        let mut hidden = self.embed.forward(target_ids)?.as_dtype(dtype)?; // [B, St, D]

        let (t_cos, t_sin) = self.rope.forward(st)?; // target positions
        let (s_cos, s_sin) = self.rope.forward(ss)?; // source positions
        let target_rope = (&t_cos, &t_sin);
        let source_rope = (&s_cos, &s_sin);

        for block in &self.blocks {
            hidden = block.forward(&hidden, source_hidden, target_rope, source_rope)?;
        }
        let mut hidden = rms_norm(
            &self.out_proj.forward(&hidden)?,
            &self.norm,
            self.cfg.norm_eps,
        )?;

        // Prompt weighting: scale each T5 query-token OUTPUT vector by its weight (T5 path only).
        // Skip entirely when every weight is 1.0 so the unweighted path stays bit-identical.
        if let Some(w) = target_weights {
            if w.iter().any(|&x| x != 1.0) {
                // Align the weight vector to St, padding/truncating with 1.0 (fail gracefully on a
                // length mismatch rather than panic on a shape error).
                let st_usize = st as usize;
                let mut wv = vec![1.0f32; st_usize];
                let take = w.len().min(st_usize);
                wv[..take].copy_from_slice(&w[..take]);
                let warr = Array::from_slice(&wv, &[1, st, 1]).as_dtype(dtype)?; // [1, St, 1]
                hidden = multiply(&hidden, &warr)?; // broadcast over B and target_dim
            }
        }

        // Right-pad to min_sequence_length so the DiT always sees exactly 512 text tokens.
        let min = self.cfg.min_sequence_length as i32;
        if st < min {
            let pad = zeros_dtype(&[b, min - st, self.cfg.target_dim as i32], dtype)?;
            Ok(concatenate_axis(&[&hidden, &pad], 1)?)
        } else {
            Ok(hidden)
        }
    }

    pub fn config(&self) -> &ConditionerConfig {
        &self.cfg
    }

    /// Toggle SDPA-segment gradient checkpointing (sc-10576) on every conditioner block. Training-only:
    /// the trainer leaves this ON (the conditioner is never whole-block checkpointed), matching the
    /// z-image refiner leg. Numerically identical to the retained backward; inference never calls it.
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        for b in &mut self.blocks {
            b.set_sdpa_checkpoint(on);
        }
    }
}

// ---- LoRA/LoKr adapter surface (sc-10521) --------------------------------------------------------
//
// The **turbo** LoRA (`anima-turbo-lora-v0.2`) addresses 60 = 6×10 `llm_adapter.*` targets that a
// DiT-only injection walk would silently drop — the sc-10274 (eros distill LoRA) failure class. (Those
// 60 `lora_B` are zero-initialized in the turbo file, so `B·A ≡ 0` and dropping them is numerically
// inert for *it*; routing must still be enforced for future non-zero conditioner LoRAs and the shipped
// `anima-rl-v0.1`, which also carries these 60 targets.) The conditioner MUST therefore be a
// first-class injectable target. The trained files address it (after the loader strips the
// `diffusion_model.` prefix and the top-level host strips the `llm_adapter.` segment) by the
// conditioner's own module names: `blocks.N.{self_attn,cross_attn}.{q_proj,k_proj,v_proj,o_proj}` and
// `blocks.N.mlp.{0,2}` (no norm targets).

impl AdaptableHost for Attention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["q_proj"] => Some(&mut self.q_proj),
            ["k_proj"] => Some(&mut self.k_proj),
            ["v_proj"] => Some(&mut self.v_proj),
            ["o_proj"] => Some(&mut self.o_proj),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["q_proj", "k_proj", "v_proj", "o_proj"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl AdaptableHost for Block {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["self_attn", rest @ ..] => self.self_attn.adaptable_mut(rest),
            ["cross_attn", rest @ ..] => self.cross_attn.adaptable_mut(rest),
            ["mlp", "0"] => Some(&mut self.mlp_in),
            ["mlp", "2"] => Some(&mut self.mlp_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.extend(prefixed_paths("self_attn", &self.self_attn));
        out.extend(prefixed_paths("cross_attn", &self.cross_attn));
        out.push("mlp.0".to_string());
        out.push("mlp.2".to_string());
        out
    }
}

/// The `AnimaTextConditioner` adapter host: `blocks.N.*` route into the per-block leaves (the 60 = 6×10
/// adapter targets the turbo LoRA carries); `out_proj` is routable too (the shipped LoRAs never target
/// it). Only block targets are enumerated for the kohya table.
impl AdaptableHost for AnimaTextConditioner {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["blocks", n, rest @ ..] => self
                .blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["out_proj"] => Some(&mut self.out_proj),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, b) in self.blocks.iter().enumerate() {
            out.extend(prefixed_paths(&format!("blocks.{i}"), b));
        }
        out
    }
}

// -------------------------------------------------------------------------------------------------
// Test-only structural constructor
// -------------------------------------------------------------------------------------------------

/// Build the conditioner module tree with **placeholder** (1×1) weights so the target-enumeration
/// guard (sc-10522) can exercise the real [`AdaptableHost::adaptable_paths`] surface without the
/// licensed checkpoint or any Metal compute — enumeration only walks the tree; nothing is evaluated.
#[cfg(test)]
mod structural {
    use super::*;

    fn ph_lin() -> AdaptableLinear {
        AdaptableLinear::dense(Array::from_slice(&[0.0f32], &[1, 1]), None)
    }
    fn ph_norm() -> Array {
        Array::from_slice(&[1.0f32], &[1])
    }

    impl AnimaTextConditioner {
        /// A weight-free conditioner with `cfg.num_layers` structurally-complete blocks (6 for Anima),
        /// for the path-enumeration guard. Placeholder tensors only; nothing is evaluated.
        pub(crate) fn structural(cfg: ConditionerConfig) -> Self {
            let heads = cfg.num_attention_heads as i32;
            let head_dim = cfg.head_dim() as i32;
            let attn = || Attention {
                q_proj: ph_lin(),
                k_proj: ph_lin(),
                v_proj: ph_lin(),
                o_proj: ph_lin(),
                q_norm: ph_norm(),
                k_norm: ph_norm(),
                heads,
                head_dim,
                scale: 1.0,
                eps: cfg.norm_eps,
                ckpt_sdpa: false,
            };
            let blocks = (0..cfg.num_layers)
                .map(|_| Block {
                    norm_self_attn: ph_norm(),
                    self_attn: attn(),
                    norm_cross_attn: ph_norm(),
                    cross_attn: attn(),
                    norm_mlp: ph_norm(),
                    mlp_in: ph_lin(),
                    mlp_out: ph_lin(),
                    eps: cfg.norm_eps,
                })
                .collect();
            Self {
                embed: TokenEmbedding::Dense(Array::from_slice(&[0.0f32], &[1, 1])),
                blocks,
                out_proj: ph_lin(),
                norm: ph_norm(),
                rope: TextRope::new(head_dim, cfg.rope_theta),
                cfg,
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Test-only SYNTHETIC constructor — real (small) random weights, evaluable on Metal (sc-10576)
// -------------------------------------------------------------------------------------------------

/// Real-weight conditioner for the grad-parity test (sc-10576): a dimension-parametric build with
/// small random weights so the 6-block conditioner forward+backward runs on Metal without the licensed
/// checkpoint. Mirrors [`CosmosDiT::synthetic`]. `cfg` sets every dim; use a tiny `cfg` for speed.
#[cfg(test)]
pub(crate) mod synthetic {
    use super::*;
    use mlx_rs::random;

    pub(crate) struct Rng(pub u64);

    impl Rng {
        fn lin(&mut self, out: i32, in_f: i32, bias: bool) -> AdaptableLinear {
            self.0 = self.0.wrapping_add(1);
            let key = random::key(self.0).unwrap();
            let w = random::normal::<f32>(&[out, in_f], None, None, Some(&key)).unwrap();
            let w = multiply(&w, Array::from_slice(&[0.05f32], &[1])).unwrap();
            let b = bias.then(|| Array::zeros::<f32>(&[out]).unwrap());
            AdaptableLinear::dense(w, b)
        }

        fn norm(&mut self, d: i32) -> Array {
            Array::ones::<f32>(&[d]).unwrap()
        }

        fn attn(&mut self, heads: i32, head_dim: i32, kv_in: i32, eps: f32) -> Attention {
            let d = heads * head_dim;
            Attention {
                q_proj: self.lin(d, d, false),
                k_proj: self.lin(d, kv_in, false),
                v_proj: self.lin(d, kv_in, false),
                o_proj: self.lin(d, d, false),
                q_norm: self.norm(head_dim),
                k_norm: self.norm(head_dim),
                heads,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
                eps,
                ckpt_sdpa: false,
            }
        }
    }

    impl AnimaTextConditioner {
        /// A synthetic conditioner with real random weights. `cfg` sets every dim (`model_dim`,
        /// `source_dim` for the cross-attn kv, `target_vocab_size` for the T5 embedding).
        pub(crate) fn synthetic(cfg: ConditionerConfig, seed: u64) -> Self {
            let d = cfg.model_dim as i32;
            let heads = cfg.num_attention_heads as i32;
            let head_dim = cfg.head_dim() as i32;
            let src = cfg.source_dim as i32;
            let target = cfg.target_dim as i32;
            let ff = (cfg.mlp_ratio * d as f32).round() as i32;
            let vocab = cfg.target_vocab_size as i32;
            let eps = cfg.norm_eps;

            let mut r = Rng(seed);
            // Learned T5 query-token embedding [vocab, model_dim].
            let embed_rand =
                random::normal::<f32>(&[vocab, d], None, None, Some(&random::key(seed).unwrap()))
                    .unwrap();
            let embed_w = multiply(&embed_rand, Array::from_slice(&[0.05f32], &[1])).unwrap();
            let blocks = (0..cfg.num_layers)
                .map(|_| Block {
                    norm_self_attn: r.norm(d),
                    self_attn: r.attn(heads, head_dim, d, eps), // self: kv from hidden (model_dim)
                    norm_cross_attn: r.norm(d),
                    cross_attn: r.attn(heads, head_dim, src, eps), // cross: kv from Qwen source_dim
                    norm_mlp: r.norm(d),
                    mlp_in: r.lin(ff, d, true),
                    mlp_out: r.lin(d, ff, true),
                    eps,
                })
                .collect();
            Self {
                embed: TokenEmbedding::Dense(embed_w),
                blocks,
                out_proj: r.lin(target, d, true),
                norm: r.norm(target),
                rope: TextRope::new(head_dim, cfg.rope_theta),
                cfg,
            }
        }
    }
}
