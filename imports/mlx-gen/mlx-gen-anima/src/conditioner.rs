//! `AnimaTextConditioner` — transcribed from diffusers
//! `condition_embedder_anima.py::AnimaTextConditioner`. Maps T5 token ids (learned query tokens via
//! `nn.Embedding(32128, 1024)`) + Qwen3 hidden states → the DiT's `encoder_hidden_states`. 6 blocks of
//! `[self-attn (RoPE θ=10000) → cross-attn into Qwen3 states → GELU MLP]`, then `out_proj` + RMSNorm,
//! mask-multiply, and **right-pad to 512** so the DiT always sees exactly 512 text tokens.
//!
//! Weight keys are the `net.llm_adapter.`-**stripped** names (`blocks.N.*`, `embed.weight`,
//! `out_proj.*`, `norm.weight`) — identical to the diffusers `AnimaTextConditioner` state dict, which
//! the Anima convert script loads with `strict=True` (no rename).

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, zeros_dtype};
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
        })
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
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
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
        let hidden = rms_norm(
            &self.out_proj.forward(&hidden)?,
            &self.norm,
            self.cfg.norm_eps,
        )?;

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
