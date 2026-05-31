//! The full Qwen-Image MMDiT. Port of the fork's `QwenTransformer`: project image latents and the
//! (RMSNorm'd) text embeddings into the inner dim, build the timestep conditioning + 3D RoPE, run
//! 60 dual-stream blocks, then `AdaLayerNormContinuous` + `proj_out` back to patch space.
//!
//! Weight keys follow the fork's *internal* module tree (e.g. `transformer_blocks.{i}.img_mod_linear`,
//! `…attn.attn_to_out.0`, `…img_ff.mlp_in`); the on-disk diffusers→internal remapping lands with
//! the full-model assembly (e2e slice). Per-block weights are exercised by the synthetic-weight
//! block parity test; the full 60-layer forward is validated end-to-end against the image golden.

use mlx_rs::fast::rms_norm;
use mlx_rs::Array;

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::time_text_embed::TimeTextEmbed;
use super::{AdaLayerNormContinuous, QwenRope3d, QwenTransformerBlock};

pub struct QwenTransformerConfig {
    pub in_channels: i32,
    pub out_channels: i32,
    pub num_layers: usize,
    pub num_heads: i32,
    pub head_dim: i32,
    pub joint_attention_dim: i32,
    pub patch_size: i32,
    pub txt_norm_eps: f32,
}

impl QwenTransformerConfig {
    pub fn qwen_image() -> Self {
        Self {
            in_channels: 64,
            out_channels: 16,
            num_layers: 60,
            num_heads: 24,
            head_dim: 128,
            joint_attention_dim: 3584,
            patch_size: 2,
            txt_norm_eps: 1e-6,
        }
    }

    pub fn inner_dim(&self) -> i32 {
        self.num_heads * self.head_dim
    }
}

pub struct QwenTransformer {
    img_in_w: Array,
    img_in_b: Array,
    txt_norm_w: Array,
    txt_in_w: Array,
    txt_in_b: Array,
    time_text_embed: TimeTextEmbed,
    blocks: Vec<QwenTransformerBlock>,
    norm_out: AdaLayerNormContinuous,
    proj_out_w: Array,
    proj_out_b: Array,
    rope: QwenRope3d,
    eps: f32,
}

impl QwenTransformer {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &QwenTransformerConfig) -> Result<Self> {
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(QwenTransformerBlock::from_weights(
                w,
                &p(&format!("transformer_blocks.{i}")),
                cfg.num_heads,
                cfg.head_dim,
            )?);
        }
        Ok(Self {
            img_in_w: w.require(&p("img_in.weight"))?.clone(),
            img_in_b: w.require(&p("img_in.bias"))?.clone(),
            txt_norm_w: w.require(&p("txt_norm.weight"))?.clone(),
            txt_in_w: w.require(&p("txt_in.weight"))?.clone(),
            txt_in_b: w.require(&p("txt_in.bias"))?.clone(),
            time_text_embed: TimeTextEmbed::from_weights(w, &p("time_text_embed"))?,
            blocks,
            norm_out: AdaLayerNormContinuous::from_weights(w, &p("norm_out"))?,
            proj_out_w: w.require(&p("proj_out.weight"))?.clone(),
            proj_out_b: w.require(&p("proj_out.bias"))?.clone(),
            rope: QwenRope3d::qwen_image(),
            eps: cfg.txt_norm_eps,
        })
    }

    /// `hidden_states`: packed image latents `[B, latent_h·latent_w, in_channels]`.
    /// `encoder_hidden_states`: text features `[B, txt_seq, joint_attention_dim]`.
    /// `timestep`: the scheduler sigma (flow-match time) for this step. Returns the velocity
    /// `[B, latent_h·latent_w, patch²·out_channels]`.
    pub fn forward(
        &self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        encoder_hidden_states_mask: Option<&Array>,
        timestep: f32,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<Array> {
        let b = hidden_states.shape()[0];
        let img_seq = hidden_states.shape()[1];
        let txt_seq = encoder_hidden_states.shape()[1];

        let mut hidden = linear(hidden_states, &self.img_in_w, &self.img_in_b)?;
        let encoder = rms_norm(encoder_hidden_states, &self.txt_norm_w, self.eps)?;
        let mut encoder = linear(&encoder, &self.txt_in_w, &self.txt_in_b)?;

        let ts = Array::from_slice(&vec![timestep; b as usize], &[b]);
        let text_emb = self.time_text_embed.forward(&ts)?;

        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope.forward(latent_h, latent_w, txt_seq as usize)?;
        let mask = build_joint_mask(encoder_hidden_states_mask, b, img_seq)?;

        for block in &self.blocks {
            let (e, h) = block.forward(
                &hidden,
                &encoder,
                &text_emb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                mask.as_ref(),
            )?;
            encoder = e;
            hidden = h;
        }

        let hidden = self.norm_out.forward(&hidden, &text_emb)?;
        linear(&hidden, &self.proj_out_w, &self.proj_out_b)
    }
}

/// Additive joint mask `[B, 1, 1, txt+img]` (text keys masked where padded; image keys always
/// attended). Returns `None` when no text token is padded (the fork's all-ones short-circuit).
fn build_joint_mask(txt_mask: Option<&Array>, b: i32, img_seq: i32) -> Result<Option<Array>> {
    let Some(m) = txt_mask else {
        return Ok(None);
    };
    let mvals = m.as_slice::<i32>();
    if mvals.iter().all(|&v| v == 1) {
        return Ok(None);
    }
    let txt_seq = m.shape()[1];
    let joint = txt_seq + img_seq;
    let mut data = vec![0f32; (b * joint) as usize];
    for bi in 0..b {
        for j in 0..joint {
            let valid = j >= txt_seq || mvals[(bi * txt_seq + j) as usize] == 1;
            if !valid {
                data[(bi * joint + j) as usize] = -1e9;
            }
        }
    }
    Ok(Some(Array::from_slice(&data, &[b, 1, 1, joint])))
}
