//! The Chatterbox speaker encoder itself — a faithful candle port of Resemble AI's
//! `VoiceEncoder` (`ve.safetensors`): a 3-layer LSTM (`40 → 256`) followed by a `256 → 256`
//! projection and a ReLU, L2-normalized, producing a 256-d GE2E-style speaker embedding.
//!
//! ## Weight layout (`ve.safetensors`)
//!
//! ```text
//!   lstm.weight_ih_l{0,1,2}  lstm.weight_hh_l{0,1,2}  lstm.bias_ih_l{0,1,2}  lstm.bias_hh_l{0,1,2}
//!   proj.weight  proj.bias
//!   similarity_weight  similarity_bias   (GE2E loss params — inference-irrelevant, ignored)
//! ```
//!
//! candle-nn's `LSTM` reads the exact PyTorch key/shape/gate layout (gate order `i,f,g,o`), so
//! each layer maps directly under the `lstm` prefix with the matching `layer_idx`.

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::rnn::{LSTMConfig, RNN};
use candle_nn::{VarBuilder, LSTM};

use crate::config;

/// The loaded speaker encoder: three stacked forward LSTM layers + a linear projection.
pub struct SpeakerEncoder {
    layers: Vec<LSTM>,
    proj_w: Tensor,
    proj_b: Tensor,
    device: Device,
}

impl SpeakerEncoder {
    /// Build the encoder from a `ve`-shaped [`VarBuilder`] (the safetensors root). Reads exactly
    /// the `lstm.*` and `proj.*` tensors; the GE2E `similarity_*` scalars are not read.
    pub fn new(vb: VarBuilder, device: Device) -> CandleResult<Self> {
        let lstm_vb = vb.pp("lstm");
        let mut layers = Vec::with_capacity(config::NUM_LAYERS);
        for layer_idx in 0..config::NUM_LAYERS {
            let in_dim = if layer_idx == 0 {
                config::N_MELS
            } else {
                config::HIDDEN
            };
            let cfg = LSTMConfig {
                layer_idx,
                ..LSTMConfig::default()
            };
            layers.push(candle_nn::lstm(
                in_dim,
                config::HIDDEN,
                cfg,
                lstm_vb.clone(),
            )?);
        }
        let proj_w = vb.get((config::EMBED_DIM, config::HIDDEN), "proj.weight")?;
        let proj_b = vb.get(config::EMBED_DIM, "proj.bias")?;
        Ok(Self {
            layers,
            proj_w,
            proj_b,
            device,
        })
    }

    /// Embed one mel-frame block `[T, N_MELS]` into a raw (un-normalized) `[EMBED_DIM]` vector:
    /// stacked-LSTM forward, take the top layer's final hidden state, project, ReLU. This is the
    /// per-partial-utterance forward (Resemblyzer `forward` up to the L2-norm, applied by the
    /// caller after averaging).
    fn forward_block(&self, mel_frames: &[Vec<f32>]) -> CandleResult<Vec<f32>> {
        let t = mel_frames.len();
        let flat: Vec<f32> = mel_frames.iter().flatten().copied().collect();
        // [1, T, N_MELS]
        let mut hidden =
            Tensor::from_vec(flat, (1, t, config::N_MELS), &self.device)?.to_dtype(DType::F32)?;
        // Run each LSTM layer over the sequence, feeding the stacked per-timestep hidden states
        // to the next; keep the final layer's last hidden state.
        let mut final_h: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let states = layer.seq(&hidden)?;
            let last = states.last().expect("non-empty sequence");
            if i + 1 == self.layers.len() {
                final_h = Some(last.h().clone());
            } else {
                hidden = layer.states_to_tensor(&states)?;
            }
        }
        let h = final_h.expect("at least one LSTM layer"); // [1, HIDDEN]
                                                           // proj: h @ Wᵀ + b, then ReLU.
        let projected = h.matmul(&self.proj_w.t()?)?.broadcast_add(&self.proj_b)?;
        let projected = projected.relu()?;
        projected.flatten_all()?.to_vec1::<f32>()
    }

    /// Embed a full reference clip's mel frames `[T, N_MELS]` into the final L2-normalized
    /// `[EMBED_DIM]` speaker vector, averaging over ~1.6 s partial utterances the way
    /// Resemblyzer's `embed_utterance` does (whole-clip forward when shorter than one partial).
    pub fn embed_mel_frames(&self, mel_frames: &[Vec<f32>]) -> CandleResult<Vec<f32>> {
        let n = mel_frames.len();
        let part = config::PARTIALS_N_FRAMES;
        let raw = if n <= part {
            self.forward_block(mel_frames)?
        } else {
            // 50%-overlap partial windows covering the whole clip (last window right-aligned).
            let step = (part / 2).max(1);
            let mut starts: Vec<usize> = (0..=n.saturating_sub(part)).step_by(step).collect();
            if *starts.last().unwrap_or(&0) != n - part {
                starts.push(n - part);
            }
            let mut acc = vec![0.0f32; config::EMBED_DIM];
            for &s in &starts {
                let block = &mel_frames[s..s + part];
                let e = self.forward_block(block)?;
                for (a, v) in acc.iter_mut().zip(e) {
                    *a += v;
                }
            }
            let inv = 1.0 / starts.len() as f32;
            acc.iter().map(|v| v * inv).collect()
        };
        Ok(l2_normalize(&raw))
    }
}

/// L2-normalize a vector (zero-safe: an all-zero vector is returned unchanged).
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Cosine similarity of two equal-length vectors (used by the discriminative conformance test).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_norm() {
        let v = l2_normalize(&[3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        assert_eq!(l2_normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn cosine_bounds() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }
}
