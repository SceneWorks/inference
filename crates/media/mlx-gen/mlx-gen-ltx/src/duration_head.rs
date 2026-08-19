//! `DurationHead` (sc-18774) — MLX port of the reference regression head that predicts a shot's
//! duration (log-seconds, exponentiated to seconds here) from frozen `Embeddings1DConnector`
//! outputs. Port of `Lightricks/LTX-2` @ `d151147788a9284cca791edc6ce898007e727fe6`
//! `packages/ltx-core/src/ltx_core/duration_head/duration_head.py` (`DurationHead`,
//! `AttentionPooler`).
//!
//! Architecture (15 tensors, `model_patches/ltx-2.5-duration-head-bf16.safetensors`; hyperparameters
//! pinned in [`gen_core::duration_head::hparams`] — the checkpoint's own `config.duration_head`
//! metadata section ships empty, see that module's docs):
//!
//! * `video_input_proj` / `audio_input_proj` — per-modality `Linear(cross_attention_dim,
//!   pooler_hidden_dim)`, each followed by an additive learnable modality embedding
//!   (`{video,audio}_modality_emb`, shape `(pooler_hidden_dim,)`) so the pooler can tell the two
//!   streams apart once concatenated.
//! * `attention_pooler` — one learnable query token (`query_tokens`, `(1, 256)`) cross-attends the
//!   concatenated, modality-tagged tokens via a standard `torch.nn.MultiheadAttention` (4 heads,
//!   `batch_first=True`): a single fused `in_proj_weight`/`in_proj_bias` (`(3·256, 256)` /
//!   `(3·256,)`) that this port splits into separate Q/K/V projections at load time, plus
//!   `out_proj`.
//! * `mlp_hidden` (`Linear(256, 256)`, tanh-approximate GELU) → `mlp_out` (`Linear(256, 1)`) → the
//!   log-duration; [`DurationHead::forward`] exponentiates before returning, matching upstream
//!   (`log_duration.exp()`).
//!
//! Modality-agnostic (upstream: "pass either or both of {audio, video} connector outputs") and
//! never sees packed/multi-sample sequences or an attention mask — the connector already replaces
//! padded positions with learnable registers and marks the result fully attendable (see
//! `crate::connector`), so every token this module receives is valid.
//!
//! Runs entirely in **f32**: the checkpoint is tiny (~4 MB) and, like the vocoder/audio decoder, is
//! a "quality island" utility component rather than part of the hot denoise loop, so there is no
//! reason to route it through the bf16-SDPA-shape landmine documented in `crate::connector` — f32
//! throughout sidesteps that class of bug entirely rather than reasoning about whether this head's
//! particular shape (4 heads, head_dim 64, no mask) happens to trigger it.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::nn::gelu_approximate;
use mlx_rs::ops::{add, concatenate_axis, exp, tile};
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::duration_head::hparams;
use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// The loaded `DurationHead`. All weights are held in f32 (see module docs).
pub struct DurationHead {
    video_proj_w: Array,
    video_proj_b: Array,
    video_mod_emb: Array,
    audio_proj_w: Array,
    audio_proj_b: Array,
    audio_mod_emb: Array,
    query_tokens: Array,
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    out_proj_w: Array,
    out_proj_b: Array,
    mlp_hidden_w: Array,
    mlp_hidden_b: Array,
    mlp_out_w: Array,
    mlp_out_b: Array,
}

impl DurationHead {
    /// Build from a `Weights` map holding the `duration_head.*`-prefixed tensors (i.e. the
    /// as-shipped `ltx-2.5-duration-head-bf16.safetensors`, loaded whole via [`Weights::from_file`]).
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let get = |key: &str| -> Result<Array> {
            w.require(&format!("duration_head.{key}"))?
                .as_dtype(Dtype::Float32)
                .map_err(Error::from)
        };
        let hd = hparams::POOLER_HIDDEN_DIM as i32;
        // `nn.MultiheadAttention`'s fused in_proj: rows [0:hd)=Q, [hd:2hd)=K, [2hd:3hd)=V.
        let in_proj_w = get("attention_pooler.cross_attn.in_proj_weight")?; // (3*hd, hd)
        let in_proj_b = get("attention_pooler.cross_attn.in_proj_bias")?; // (3*hd,)
        let w_parts = in_proj_w.split_axis(&[hd, 2 * hd], 0)?;
        let b_parts = in_proj_b.split_axis(&[hd, 2 * hd], 0)?;
        Ok(Self {
            video_proj_w: get("video_input_proj.weight")?,
            video_proj_b: get("video_input_proj.bias")?,
            video_mod_emb: get("video_modality_emb")?,
            audio_proj_w: get("audio_input_proj.weight")?,
            audio_proj_b: get("audio_input_proj.bias")?,
            audio_mod_emb: get("audio_modality_emb")?,
            query_tokens: get("attention_pooler.query_tokens")?,
            q_w: w_parts[0].clone(),
            q_b: b_parts[0].clone(),
            k_w: w_parts[1].clone(),
            k_b: b_parts[1].clone(),
            v_w: w_parts[2].clone(),
            v_b: b_parts[2].clone(),
            out_proj_w: get("attention_pooler.cross_attn.out_proj.weight")?,
            out_proj_b: get("attention_pooler.cross_attn.out_proj.bias")?,
            mlp_hidden_w: get("mlp_hidden.weight")?,
            mlp_hidden_b: get("mlp_hidden.bias")?,
            mlp_out_w: get("mlp_out.weight")?,
            mlp_out_b: get("mlp_out.bias")?,
        })
    }

    /// Predict duration in **seconds** (already exponentiated, matching upstream
    /// `DurationHead.forward`). `video_tokens`: `(B, T_v, 4096)`, or `None`; `audio_tokens`:
    /// `(B, T_a, 2048)`, or `None`. At least one must be given. Returns `(B,)`.
    pub fn forward(&self, video_tokens: Option<&Array>, audio_tokens: Option<&Array>) -> Result<Array> {
        if video_tokens.is_none() && audio_tokens.is_none() {
            return Err(Error::Msg(
                "ltx duration_head: forward requires at least one of video_tokens / audio_tokens"
                    .into(),
            ));
        }
        let mut groups: Vec<Array> = Vec::with_capacity(2);
        if let Some(v) = video_tokens {
            let proj = linear(&v.as_dtype(Dtype::Float32)?, &self.video_proj_w, &self.video_proj_b)?;
            groups.push(add(&proj, &self.video_mod_emb)?);
        }
        if let Some(a) = audio_tokens {
            let proj = linear(&a.as_dtype(Dtype::Float32)?, &self.audio_proj_w, &self.audio_proj_b)?;
            groups.push(add(&proj, &self.audio_mod_emb)?);
        }
        let tokens = if groups.len() == 1 {
            groups.into_iter().next().expect("len checked above")
        } else {
            let refs: Vec<&Array> = groups.iter().collect();
            concatenate_axis(&refs, 1)?
        };

        let sh = tokens.shape();
        let (b, t) = (sh[0], sh[1]);
        let hd = hparams::POOLER_HIDDEN_DIM as i32;
        let nq = hparams::NUM_QUERIES as i32;
        let nh = hparams::NUM_POOLER_HEADS as i32;
        let head_dim = hd / nh;

        let queries = tile(&self.query_tokens.reshape(&[1, nq, hd])?, &[b, 1, 1])?;
        let q = linear(&queries, &self.q_w, &self.q_b)?;
        let k = linear(&tokens, &self.k_w, &self.k_b)?;
        let v = linear(&tokens, &self.v_w, &self.v_b)?;
        let q = q
            .reshape(&[b, nq, nh, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[b, t, nh, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[b, t, nh, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attn = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?; // (b, nh, nq, head_dim)
        let attn = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, nq, hd])?;
        let pooled = linear(&attn, &self.out_proj_w, &self.out_proj_b)?;

        let pooled_flat = pooled.reshape(&[b, nq * hd])?;
        let hidden = gelu_approximate(linear(
            &pooled_flat,
            &self.mlp_hidden_w,
            &self.mlp_hidden_b,
        )?)?;
        let log_duration = linear(&hidden, &self.mlp_out_w, &self.mlp_out_b)?.reshape(&[b])?; // squeeze(-1)
        Ok(exp(&log_duration)?)
    }

    /// Predict duration in seconds for a **single-item batch** (mirrors upstream
    /// `DurationPredictor.__call__`'s own restriction — it only ever runs against one caption).
    pub fn predict_seconds(
        &self,
        video_tokens: Option<&Array>,
        audio_tokens: Option<&Array>,
    ) -> Result<f32> {
        let seconds = self.forward(video_tokens, audio_tokens)?;
        if seconds.shape() != [1] {
            return Err(Error::Msg(format!(
                "ltx duration_head: predict_seconds only supports a single-item batch, got shape {:?}",
                seconds.shape()
            )));
        }
        Ok(seconds.item::<f32>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A tiny, deterministic synthetic `DurationHead` (not the real checkpoint) — enough to exercise
    /// shapes, the modality-agnostic branches, and the reachability seam without needing real
    /// weights. Real-weight numeric golden parity lives in `tests/duration_head_golden.rs`.
    fn synthetic_weights() -> Weights {
        let hd = hparams::POOLER_HIDDEN_DIM as i32;
        let video_dim = hparams::VIDEO_CROSS_ATTENTION_DIM as i32;
        let audio_dim = hparams::AUDIO_CROSS_ATTENTION_DIM as i32;
        let mut m = HashMap::new();
        let fill = |shape: &[i32], scale: f32| -> Array {
            let n: i32 = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * scale).collect();
            Array::from_slice(&data, shape)
        };
        m.insert(
            "duration_head.video_input_proj.weight".into(),
            fill(&[hd, video_dim], 0.001),
        );
        m.insert(
            "duration_head.video_input_proj.bias".into(),
            fill(&[hd], 0.01),
        );
        m.insert(
            "duration_head.video_modality_emb".into(),
            fill(&[hd], 0.01),
        );
        m.insert(
            "duration_head.audio_input_proj.weight".into(),
            fill(&[hd, audio_dim], 0.001),
        );
        m.insert(
            "duration_head.audio_input_proj.bias".into(),
            fill(&[hd], 0.01),
        );
        m.insert(
            "duration_head.audio_modality_emb".into(),
            fill(&[hd], 0.01),
        );
        m.insert(
            "duration_head.attention_pooler.query_tokens".into(),
            fill(&[hparams::NUM_QUERIES as i32, hd], 0.01),
        );
        m.insert(
            "duration_head.attention_pooler.cross_attn.in_proj_weight".into(),
            fill(&[3 * hd, hd], 0.005),
        );
        m.insert(
            "duration_head.attention_pooler.cross_attn.in_proj_bias".into(),
            fill(&[3 * hd], 0.001),
        );
        m.insert(
            "duration_head.attention_pooler.cross_attn.out_proj.weight".into(),
            fill(&[hd, hd], 0.005),
        );
        m.insert(
            "duration_head.attention_pooler.cross_attn.out_proj.bias".into(),
            fill(&[hd], 0.001),
        );
        m.insert(
            "duration_head.mlp_hidden.weight".into(),
            fill(&[hparams::MLP_HIDDEN as i32, hd], 0.005),
        );
        m.insert(
            "duration_head.mlp_hidden.bias".into(),
            fill(&[hparams::MLP_HIDDEN as i32], 0.001),
        );
        m.insert(
            "duration_head.mlp_out.weight".into(),
            fill(&[1, hparams::MLP_HIDDEN as i32], 0.01),
        );
        m.insert("duration_head.mlp_out.bias".into(), fill(&[1], 0.001));
        Weights::from_map(m)
    }

    fn probe(shape: &[i32]) -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.037).sin() * 0.5).collect();
        Array::from_slice(&data, shape)
    }

    #[test]
    fn forward_requires_at_least_one_modality() {
        let head = DurationHead::from_weights(&synthetic_weights()).unwrap();
        assert!(head.forward(None, None).is_err());
    }

    #[test]
    fn forward_is_modality_agnostic_video_only() {
        let head = DurationHead::from_weights(&synthetic_weights()).unwrap();
        let video = probe(&[1, 5, hparams::VIDEO_CROSS_ATTENTION_DIM as i32]);
        let seconds = head.predict_seconds(Some(&video), None).unwrap();
        assert!(seconds.is_finite() && seconds > 0.0, "seconds={seconds}");
    }

    #[test]
    fn forward_is_modality_agnostic_audio_only() {
        let head = DurationHead::from_weights(&synthetic_weights()).unwrap();
        let audio = probe(&[1, 4, hparams::AUDIO_CROSS_ATTENTION_DIM as i32]);
        let seconds = head.predict_seconds(None, Some(&audio)).unwrap();
        assert!(seconds.is_finite() && seconds > 0.0, "seconds={seconds}");
    }

    #[test]
    fn forward_accepts_both_modalities_at_once() {
        let head = DurationHead::from_weights(&synthetic_weights()).unwrap();
        let video = probe(&[1, 5, hparams::VIDEO_CROSS_ATTENTION_DIM as i32]);
        let audio = probe(&[1, 4, hparams::AUDIO_CROSS_ATTENTION_DIM as i32]);
        let seconds = head.predict_seconds(Some(&video), Some(&audio)).unwrap();
        assert!(seconds.is_finite() && seconds > 0.0, "seconds={seconds}");
    }

    #[test]
    fn predict_seconds_rejects_a_multi_item_batch() {
        let head = DurationHead::from_weights(&synthetic_weights()).unwrap();
        let video = probe(&[2, 5, hparams::VIDEO_CROSS_ATTENTION_DIM as i32]);
        assert!(head.predict_seconds(Some(&video), None).is_err());
    }

    /// Reachability at the real-component level (sc-18774 acceptance): wiring
    /// `gen_core::duration_head::resolve_request_num_frames`'s injected predictor to this REAL (if
    /// synthetic-weighted) `DurationHead::forward` proves the opt-in seam reaches all the way into
    /// the network's forward pass, not just a mock closure.
    #[test]
    fn opt_in_seam_reaches_the_real_forward_pass() {
        use mlx_gen::gen_core::duration_head::{resolve_request_num_frames, AutoDurationRange};

        let head = DurationHead::from_weights(&synthetic_weights()).unwrap();
        let video = probe(&[1, 5, hparams::VIDEO_CROSS_ATTENTION_DIM as i32]);
        let mut calls = 0u32;
        let mut predict = || -> mlx_gen::gen_core::Result<f32> {
            calls += 1;
            head.predict_seconds(Some(&video), None)
                .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))
        };
        let frames = resolve_request_num_frames(
            None,
            Some(AutoDurationRange::default()),
            24.0,
            mlx_gen::gen_core::duration_head::TEMPORAL_GRID,
            &mut predict,
        )
        .unwrap();
        assert_eq!(calls, 1, "the real forward pass must be reached exactly once");
        let frames = frames.expect("auto-duration opted in");
        assert_eq!((frames - 1) % 8, 0, "frames={frames}");

        // An explicit duration wins and never touches the head, even with the real forward wired in.
        calls = 0;
        let mut predict2 = || -> mlx_gen::gen_core::Result<f32> {
            calls += 1;
            head.predict_seconds(Some(&video), None)
                .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))
        };
        let explicit = resolve_request_num_frames(
            Some(65),
            Some(AutoDurationRange::default()),
            24.0,
            mlx_gen::gen_core::duration_head::TEMPORAL_GRID,
            &mut predict2,
        )
        .unwrap();
        assert_eq!(explicit, Some(65));
        assert_eq!(calls, 0, "explicit duration must never reach the head");
    }
}
