//! `DurationHead` (sc-18774) — candle port of the reference regression head that predicts a shot's
//! duration (log-seconds, exponentiated to seconds here) from frozen `Embeddings1DConnector`
//! outputs. Mirror of `mlx_gen_ltx::duration_head` — see that module's docs for the full upstream
//! reference (`Lightricks/LTX-2` @ `d151147788a9284cca791edc6ce898007e727fe6`,
//! `packages/ltx-core/src/ltx_core/duration_head/duration_head.py`), the architecture (per-modality
//! input projections + additive modality embeddings → a 4-head `AttentionPooler` cross-attending one
//! learnable query → a small MLP → log-duration → `exp`), and the pinned hyperparameters
//! ([`gen_core::duration_head::hparams`], shared with the MLX port so both backends read from one
//! source of truth).
//!
//! Runs entirely in **f32** (same rationale as `mlx_gen_ltx::duration_head`: a tiny "quality island"
//! utility component, not part of the hot denoise loop, so there is no reason to pay for bf16
//! activations here).

use candle_gen::candle_core::{DType, Device, Error, Result, Tensor};
use candle_gen::gen_core::duration_head::hparams;
use candle_gen::weights::Weights;

/// `y = x·Wᵀ + b`, broadcasting over any leading batch dims (`w`: `[out, in]`, PyTorch layout).
fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    x.broadcast_matmul(&w.t()?)?.broadcast_add(b)
}

/// The loaded `DurationHead`. All weights are held in f32 (see module docs).
pub struct DurationHead {
    video_proj_w: Tensor,
    video_proj_b: Tensor,
    video_mod_emb: Tensor,
    audio_proj_w: Tensor,
    audio_proj_b: Tensor,
    audio_mod_emb: Tensor,
    query_tokens: Tensor,
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    out_proj_w: Tensor,
    out_proj_b: Tensor,
    mlp_hidden_w: Tensor,
    mlp_hidden_b: Tensor,
    mlp_out_w: Tensor,
    mlp_out_b: Tensor,
    device: Device,
}

impl DurationHead {
    /// Build from a `Weights` map holding the `duration_head.*`-prefixed tensors (i.e. the
    /// as-shipped `ltx-2.5-duration-head-bf16.safetensors`, loaded whole via
    /// [`Weights::from_file`]).
    pub fn from_weights(w: &Weights, device: &Device) -> Result<Self> {
        let get = |key: &str| -> Result<Tensor> {
            w.require(&format!("duration_head.{key}"))
                .map_err(|e| Error::Msg(e.to_string()))?
                .to_dtype(DType::F32)?
                .contiguous()
        };
        let hd = hparams::POOLER_HIDDEN_DIM;
        // `nn.MultiheadAttention`'s fused in_proj: rows [0:hd)=Q, [hd:2hd)=K, [2hd:3hd)=V.
        let in_proj_w = get("attention_pooler.cross_attn.in_proj_weight")?; // (3*hd, hd)
        let in_proj_b = get("attention_pooler.cross_attn.in_proj_bias")?; // (3*hd,)
        let q_w = in_proj_w.narrow(0, 0, hd)?.contiguous()?;
        let k_w = in_proj_w.narrow(0, hd, hd)?.contiguous()?;
        let v_w = in_proj_w.narrow(0, 2 * hd, hd)?.contiguous()?;
        let q_b = in_proj_b.narrow(0, 0, hd)?.contiguous()?;
        let k_b = in_proj_b.narrow(0, hd, hd)?.contiguous()?;
        let v_b = in_proj_b.narrow(0, 2 * hd, hd)?.contiguous()?;
        Ok(Self {
            video_proj_w: get("video_input_proj.weight")?,
            video_proj_b: get("video_input_proj.bias")?,
            video_mod_emb: get("video_modality_emb")?,
            audio_proj_w: get("audio_input_proj.weight")?,
            audio_proj_b: get("audio_input_proj.bias")?,
            audio_mod_emb: get("audio_modality_emb")?,
            query_tokens: get("attention_pooler.query_tokens")?,
            q_w,
            q_b,
            k_w,
            k_b,
            v_w,
            v_b,
            out_proj_w: get("attention_pooler.cross_attn.out_proj.weight")?,
            out_proj_b: get("attention_pooler.cross_attn.out_proj.bias")?,
            mlp_hidden_w: get("mlp_hidden.weight")?,
            mlp_hidden_b: get("mlp_hidden.bias")?,
            mlp_out_w: get("mlp_out.weight")?,
            mlp_out_b: get("mlp_out.bias")?,
            device: device.clone(),
        })
    }

    /// Predict duration in **seconds** (already exponentiated, matching upstream
    /// `DurationHead.forward`). `video_tokens`: `(B, T_v, 4096)`, or `None`; `audio_tokens`:
    /// `(B, T_a, 2048)`, or `None`. At least one must be given. Returns `(B,)`.
    pub fn forward(&self, video_tokens: Option<&Tensor>, audio_tokens: Option<&Tensor>) -> Result<Tensor> {
        if video_tokens.is_none() && audio_tokens.is_none() {
            return Err(Error::Msg(
                "ltx duration_head: forward requires at least one of video_tokens / audio_tokens"
                    .into(),
            ));
        }
        let mut groups: Vec<Tensor> = Vec::with_capacity(2);
        if let Some(v) = video_tokens {
            let proj = linear(&v.to_dtype(DType::F32)?, &self.video_proj_w, &self.video_proj_b)?;
            groups.push(proj.broadcast_add(&self.video_mod_emb)?);
        }
        if let Some(a) = audio_tokens {
            let proj = linear(&a.to_dtype(DType::F32)?, &self.audio_proj_w, &self.audio_proj_b)?;
            groups.push(proj.broadcast_add(&self.audio_mod_emb)?);
        }
        let tokens = if groups.len() == 1 {
            groups.into_iter().next().expect("len checked above")
        } else {
            Tensor::cat(&groups, 1)?
        };

        let (b, t, _) = tokens.dims3()?;
        let hd = hparams::POOLER_HIDDEN_DIM;
        let nq = hparams::NUM_QUERIES;
        let nh = hparams::NUM_POOLER_HEADS;
        let head_dim = hd / nh;

        let queries = self
            .query_tokens
            .reshape((1, nq, hd))?
            .broadcast_as((b, nq, hd))?
            .contiguous()?;
        let q = linear(&queries, &self.q_w, &self.q_b)?;
        let k = linear(&tokens, &self.k_w, &self.k_b)?;
        let v = linear(&tokens, &self.v_w, &self.v_b)?;
        let q = q.reshape((b, nq, nh, head_dim))?.transpose(1, 2)?.contiguous()?; // (b,nh,nq,hd)
        let k = k.reshape((b, t, nh, head_dim))?.transpose(1, 2)?.contiguous()?; // (b,nh,t,hd)
        let v = v.reshape((b, t, nh, head_dim))?.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let weights = candle_gen::candle_nn::ops::softmax_last_dim(&scores)?;
        let attn = weights.matmul(&v)?; // (b,nh,nq,head_dim)
        let attn = attn.transpose(1, 2)?.reshape((b, nq, hd))?.contiguous()?;
        let pooled = linear(&attn, &self.out_proj_w, &self.out_proj_b)?;

        let pooled_flat = pooled.reshape((b, nq * hd))?;
        let hidden = linear(&pooled_flat, &self.mlp_hidden_w, &self.mlp_hidden_b)?.gelu()?; // tanh-approx GELU
        let log_duration = linear(&hidden, &self.mlp_out_w, &self.mlp_out_b)?.reshape(b)?; // squeeze(-1)
        log_duration.exp()
    }

    /// Predict duration in seconds for a **single-item batch** (mirrors upstream
    /// `DurationPredictor.__call__`'s own restriction — it only ever runs against one caption).
    pub fn predict_seconds(
        &self,
        video_tokens: Option<&Tensor>,
        audio_tokens: Option<&Tensor>,
    ) -> Result<f32> {
        let seconds = self.forward(video_tokens, audio_tokens)?;
        if seconds.dims() != [1] {
            return Err(Error::Msg(format!(
                "ltx duration_head: predict_seconds only supports a single-item batch, got shape {:?}",
                seconds.dims()
            )));
        }
        seconds.reshape(())?.to_scalar::<f32>()
    }

    /// The device this head's weights are resident on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A tiny, deterministic synthetic `DurationHead` (not the real checkpoint) — enough to exercise
    /// shapes, the modality-agnostic branches, and the reachability seam without needing real
    /// weights. Real-weight numeric golden parity lives in `tests/duration_head_golden.rs`.
    fn synthetic_weights(device: &Device) -> Weights {
        let hd = hparams::POOLER_HIDDEN_DIM;
        let video_dim = hparams::VIDEO_CROSS_ATTENTION_DIM;
        let audio_dim = hparams::AUDIO_CROSS_ATTENTION_DIM;
        let fill = |shape: &[usize], scale: f32| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * scale).collect();
            Tensor::from_vec(data, shape, device).unwrap()
        };
        let mut m = HashMap::new();
        m.insert(
            "duration_head.video_input_proj.weight".into(),
            fill(&[hd, video_dim], 0.001),
        );
        m.insert(
            "duration_head.video_input_proj.bias".into(),
            fill(&[hd], 0.01),
        );
        m.insert("duration_head.video_modality_emb".into(), fill(&[hd], 0.01));
        m.insert(
            "duration_head.audio_input_proj.weight".into(),
            fill(&[hd, audio_dim], 0.001),
        );
        m.insert(
            "duration_head.audio_input_proj.bias".into(),
            fill(&[hd], 0.01),
        );
        m.insert("duration_head.audio_modality_emb".into(), fill(&[hd], 0.01));
        m.insert(
            "duration_head.attention_pooler.query_tokens".into(),
            fill(&[hparams::NUM_QUERIES, hd], 0.01),
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
            fill(&[hparams::MLP_HIDDEN, hd], 0.005),
        );
        m.insert(
            "duration_head.mlp_hidden.bias".into(),
            fill(&[hparams::MLP_HIDDEN], 0.001),
        );
        m.insert(
            "duration_head.mlp_out.weight".into(),
            fill(&[1, hparams::MLP_HIDDEN], 0.01),
        );
        m.insert("duration_head.mlp_out.bias".into(), fill(&[1], 0.001));
        Weights::from_map(m)
    }

    fn probe(shape: &[usize], device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.037).sin() * 0.5).collect();
        Tensor::from_vec(data, shape, device).unwrap()
    }

    #[test]
    fn forward_requires_at_least_one_modality() {
        let device = Device::Cpu;
        let head = DurationHead::from_weights(&synthetic_weights(&device), &device).unwrap();
        assert!(head.forward(None, None).is_err());
    }

    #[test]
    fn forward_is_modality_agnostic_video_only() {
        let device = Device::Cpu;
        let head = DurationHead::from_weights(&synthetic_weights(&device), &device).unwrap();
        let video = probe(&[1, 5, hparams::VIDEO_CROSS_ATTENTION_DIM], &device);
        let seconds = head.predict_seconds(Some(&video), None).unwrap();
        assert!(seconds.is_finite() && seconds > 0.0, "seconds={seconds}");
    }

    #[test]
    fn forward_is_modality_agnostic_audio_only() {
        let device = Device::Cpu;
        let head = DurationHead::from_weights(&synthetic_weights(&device), &device).unwrap();
        let audio = probe(&[1, 4, hparams::AUDIO_CROSS_ATTENTION_DIM], &device);
        let seconds = head.predict_seconds(None, Some(&audio)).unwrap();
        assert!(seconds.is_finite() && seconds > 0.0, "seconds={seconds}");
    }

    #[test]
    fn forward_accepts_both_modalities_at_once() {
        let device = Device::Cpu;
        let head = DurationHead::from_weights(&synthetic_weights(&device), &device).unwrap();
        let video = probe(&[1, 5, hparams::VIDEO_CROSS_ATTENTION_DIM], &device);
        let audio = probe(&[1, 4, hparams::AUDIO_CROSS_ATTENTION_DIM], &device);
        let seconds = head.predict_seconds(Some(&video), Some(&audio)).unwrap();
        assert!(seconds.is_finite() && seconds > 0.0, "seconds={seconds}");
    }

    #[test]
    fn predict_seconds_rejects_a_multi_item_batch() {
        let device = Device::Cpu;
        let head = DurationHead::from_weights(&synthetic_weights(&device), &device).unwrap();
        let video = probe(&[2, 5, hparams::VIDEO_CROSS_ATTENTION_DIM], &device);
        assert!(head.predict_seconds(Some(&video), None).is_err());
    }

    /// Reachability at the real-component level (sc-18774 acceptance): wiring
    /// `gen_core::duration_head::resolve_request_num_frames`'s injected predictor to this REAL (if
    /// synthetic-weighted) `DurationHead::forward` proves the opt-in seam reaches all the way into
    /// the network's forward pass, not just a mock closure.
    #[test]
    fn opt_in_seam_reaches_the_real_forward_pass() {
        use candle_gen::gen_core::duration_head::{resolve_request_num_frames, AutoDurationRange};

        let device = Device::Cpu;
        let head = DurationHead::from_weights(&synthetic_weights(&device), &device).unwrap();
        let video = probe(&[1, 5, hparams::VIDEO_CROSS_ATTENTION_DIM], &device);
        let mut calls = 0u32;
        let mut predict = || -> candle_gen::gen_core::Result<f32> {
            calls += 1;
            head.predict_seconds(Some(&video), None)
                .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))
        };
        let frames = resolve_request_num_frames(
            None,
            Some(AutoDurationRange::default()),
            24.0,
            candle_gen::gen_core::duration_head::TEMPORAL_GRID,
            &mut predict,
        )
        .unwrap();
        assert_eq!(calls, 1, "the real forward pass must be reached exactly once");
        let frames = frames.expect("auto-duration opted in");
        assert_eq!((frames - 1) % 8, 0, "frames={frames}");

        // An explicit duration wins and never touches the head, even with the real forward wired in.
        calls = 0;
        let mut predict2 = || -> candle_gen::gen_core::Result<f32> {
            calls += 1;
            head.predict_seconds(Some(&video), None)
                .map_err(|e| candle_gen::gen_core::Error::Msg(e.to_string()))
        };
        let explicit = resolve_request_num_frames(
            Some(65),
            Some(AutoDurationRange::default()),
            24.0,
            candle_gen::gen_core::duration_head::TEMPORAL_GRID,
            &mut predict2,
        )
        .unwrap();
        assert_eq!(explicit, Some(65));
        assert_eq!(calls, 0, "explicit duration must never reach the head");
    }
}
