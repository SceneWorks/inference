//! LTX-2.3 text encoder — the full S1 path producing `video_embeddings` from token ids. Port of
//! mlx-gen-ltx `text_encoder.rs` (the 2.3 per-token-RMS feature path):
//!   Gemma-3-12B (49 hidden states) → `norm_and_concat_per_token_rms` (3840×49 = 188160)
//!   → `×√(4096/3840)` → `video_aggregate_embed` Linear (188160 → 4096)
//!   → `Embeddings1DConnector` → `video_embeddings` `[1, L, 4096]`.
//!
//! The projection lives at the checkpoint's top level (`text_embedding_projection.video_aggregate_
//! embed.*`); the connector under `model.diffusion_model.video_embeddings_connector.*`. Runs bf16.
//!
//! sc-18763: the math below (`normed_hidden`) is the V2 (`PER_TOKEN_RMS`) caption feature
//! extractor, and it's the ONLY one this crate implements — see
//! [`crate::config::TextEncoderFeatureVersion`] for the upstream-mirroring detection function.
//! Construction (both [`LtxTextEncoder::new`] and [`LtxTextEncoder::new_av`], via `require_v2`
//! below) validates against [`crate::config::TextEncoderFeatureConfig::ltx_2_3`], which is
//! currently a **hardcoded, compile-time-checked constant** (this crate has no
//! `embedded_config.json` reader yet) — NOT a live per-checkpoint config-driven check the way the
//! mlx side is. Real config-driven validation on this backend is being wired through sc-18757's
//! config-threading surface; see the module note in `config.rs` for detail.

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;

use crate::config::{
    ConnectorConfig, GemmaConfig, TextEncoderFeatureConfig, TextEncoderFeatureVersion,
};
use crate::connector::Connector;
use crate::gemma::GemmaEncoder;
use crate::quant::{qlinear, QLinear};

const RMS_EPS: f64 = 1e-6;

/// The audio text head (sc-5495): a separate aggregate projection (188160 → 2048) + rescale +
/// `audio_embeddings_connector`, sharing the same Gemma hidden states as the video head.
struct AudioHead {
    aggregate: QLinear, // [2048, 188160] + bias (packed-detected, sc-9417)
    rescale: f64,       // √(2048 / 3840)
    connector: Connector,
}

pub struct LtxTextEncoder {
    gemma: GemmaEncoder,
    aggregate: QLinear, // [4096, 188160] + bias (packed-detected, sc-9417)
    rescale: f64,       // √(4096 / 3840)
    connector: Connector,
    audio: Option<AudioHead>,
    hidden_size: usize,
    device: Device,
}

impl LtxTextEncoder {
    /// `gemma_vb` rooted at `language_model.model.`; `proj_vb` rooted at the checkpoint top level
    /// (for `text_embedding_projection.*`); `dit_vb` rooted at `model.diffusion_model.` (for the
    /// connector).
    pub fn new(
        gemma_vb: VarBuilder,
        proj_vb: VarBuilder,
        dit_vb: VarBuilder,
        gemma_cfg: &GemmaConfig,
        conn_cfg: &ConnectorConfig,
    ) -> Result<Self> {
        require_v2()?;
        let device = gemma_vb.device().clone();
        let gemma = GemmaEncoder::new(gemma_vb, gemma_cfg)?;
        // Packed-detecting aggregate projection (sc-9417): dense in the hosted tier, but routed through
        // the shared packed-detect for the "linear_detect everywhere" superset. `out_dim` (the connector
        // inner dim, 4096) drives the rescale — read from config, not the weight shape, so the packed
        // path (no dense weight) needs no shape probe.
        let out_dim = conn_cfg.inner_dim();
        let aggregate = qlinear(
            &proj_vb,
            "text_embedding_projection.video_aggregate_embed",
            true,
        )?;
        let rescale = (out_dim as f64 / gemma_cfg.hidden_size as f64).sqrt();
        let connector = Connector::new(dit_vb, conn_cfg)?;
        Ok(Self {
            gemma,
            aggregate,
            rescale,
            connector,
            audio: None,
            hidden_size: gemma_cfg.hidden_size,
            device,
        })
    }

    /// As [`Self::new`] but also loads the **audio** text head (sc-5495): the
    /// `audio_aggregate_embed` projection (188160 → 2048) + the `audio_embeddings_connector`. Enables
    /// [`Self::encode_both`] for the AudioVideo path.
    #[allow(clippy::too_many_arguments)]
    pub fn new_av(
        gemma_vb: VarBuilder,
        proj_vb: VarBuilder,
        dit_vb: VarBuilder,
        gemma_cfg: &GemmaConfig,
        conn_cfg: &ConnectorConfig,
        audio_conn_cfg: &ConnectorConfig,
    ) -> Result<Self> {
        let mut me = Self::new(
            gemma_vb,
            proj_vb.clone(),
            dit_vb.clone(),
            gemma_cfg,
            conn_cfg,
        )?;
        // Audio aggregate projection (188160 → 2048); `out_dim` = the audio connector inner dim.
        let out_dim = audio_conn_cfg.inner_dim();
        let aggregate = qlinear(
            &proj_vb,
            "text_embedding_projection.audio_aggregate_embed",
            true,
        )?;
        let rescale = (out_dim as f64 / gemma_cfg.hidden_size as f64).sqrt();
        let connector =
            Connector::new_with_prefix(dit_vb, audio_conn_cfg, "audio_embeddings_connector")?;
        me.audio = Some(AudioHead {
            aggregate,
            rescale,
            connector,
        });
        Ok(me)
    }

    /// `norm_and_concat_per_token_rms`: stack the 49 hidden states `[1,L,3840,49]`, RMS-normalize each
    /// `(token, layer)` slice over the 3840 hidden dim, flatten dim-major/layer-minor `[1,L,188160]`,
    /// zero the padded positions.
    fn normed_hidden(&self, hiddens: &[Tensor], mask01: &[u32]) -> Result<Tensor> {
        let refs: Vec<&Tensor> = hiddens.iter().collect();
        let enc = Tensor::stack(&refs, 3)?; // (1, L, 3840, 49)
        let (b, l, _, n) = enc.dims4()?;
        let var = enc.sqr()?.mean_keepdim(2)?; // (1, L, 1, 49)
        let inv = (var + RMS_EPS)?.sqrt()?.recip()?;
        let normed = enc.broadcast_mul(&inv)?;
        let normed = normed.reshape((b, l, self.hidden_size * n))?; // (1, L, 188160)
                                                                    // Zero padded token positions.
        let mask: Vec<f32> = mask01.iter().map(|&m| m as f32).collect();
        let mask = Tensor::from_vec(mask, (1, l, 1), &self.device)?.to_dtype(DType::BF16)?;
        normed.broadcast_mul(&mask)
    }

    /// Encode `input_ids` `[1,L]` (u32) + `mask01` (1 for valid, left-padded) → `video_embeddings`
    /// `[1, L, 4096]` (bf16).
    pub fn encode(&self, input_ids: &Tensor, mask01: &[u32]) -> Result<Tensor> {
        Ok(self.encode_with_features(input_ids, mask01)?.1)
    }

    /// Like [`Self::encode`] but also returns the pre-connector `video_features` `[1, L, 4096]`
    /// (bf16) — the feature-extractor output entering the connector (post-projection, post-norm).
    /// sc-18763: this is the "connector input" the acceptance-criterion golden-parity gate checks;
    /// mirrors mlx-gen-ltx `text_encoder.rs`'s `encode_with_features`.
    pub fn encode_with_features(
        &self,
        input_ids: &Tensor,
        mask01: &[u32],
    ) -> Result<(Tensor, Tensor)> {
        let hiddens = self.gemma.forward(input_ids, mask01)?; // 49 × (1,L,3840)
        let normed = self.normed_hidden(&hiddens, mask01)?;
        let scaled = (normed * self.rescale)?;
        let features = self.aggregate.forward(&scaled)?; // (1,L,4096)
        let nv = mask01.iter().filter(|&&m| m != 0).count();
        let embeddings = self.connector.forward(&features, nv)?;
        Ok((features, embeddings))
    }

    /// Encode once and project BOTH the video (4096) and audio (2048) contexts, sharing the Gemma
    /// hidden states + per-token-RMS concat (sc-5495). Requires [`Self::new_av`]. Returns
    /// `(video_embeddings [1,L,4096], audio_embeddings [1,L,2048])` (bf16).
    pub fn encode_both(&self, input_ids: &Tensor, mask01: &[u32]) -> Result<(Tensor, Tensor)> {
        let (_, _, video, audio_ctx) = self.encode_both_with_features(input_ids, mask01)?;
        Ok((video, audio_ctx))
    }

    /// Like [`Self::encode_both`] but also returns the pre-connector `(video_features,
    /// audio_features)` — the feature-extractor outputs entering each connector (post-projection,
    /// post-norm). Requires [`Self::new_av`]. Mirrors mlx-gen-ltx `text_encoder.rs`'s
    /// `encode_av_with_features`.
    pub fn encode_both_with_features(
        &self,
        input_ids: &Tensor,
        mask01: &[u32],
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let audio = self.audio.as_ref().ok_or_else(|| {
            candle_gen::candle_core::Error::Msg(
                "ltx: audio text head not loaded (use new_av)".into(),
            )
        })?;
        let hiddens = self.gemma.forward(input_ids, mask01)?;
        let normed = self.normed_hidden(&hiddens, mask01)?;
        let nv = mask01.iter().filter(|&&m| m != 0).count();
        let v_feat = self.aggregate.forward(&(normed.clone() * self.rescale)?)?;
        let video = self.connector.forward(&v_feat, nv)?;
        let a_feat = audio.aggregate.forward(&(normed * audio.rescale)?)?;
        let audio_ctx = audio.connector.forward(&a_feat, nv)?;
        Ok((v_feat, a_feat, video, audio_ctx))
    }
}

/// sc-18763: reject construction unless the crate's caption-feature-extractor selection resolves
/// to V2 — `normed_hidden` above is the V2 math unconditionally, and running it against anything
/// else would silently produce plausible-looking, wrong conditioning. `new_av` delegates to `new`,
/// so this one call site covers both constructors.
///
/// This currently checks a **hardcoded, compile-time constant**
/// ([`TextEncoderFeatureConfig::ltx_2_3`]), not a live per-checkpoint config value — this crate
/// doesn't yet thread real config through construction (sc-18757 is wiring that). It still catches
/// a real class of bug (someone editing that constant away from what upstream's detection accepts
/// without noticing), just not a per-checkpoint one yet. Do not describe this as "config-driven"
/// the way the mlx backend's genuinely-JSON-driven check is (sc-18763 coordinator review).
fn require_v2() -> Result<()> {
    let cfg = TextEncoderFeatureConfig::ltx_2_3()?;
    if cfg.version != TextEncoderFeatureVersion::V2 {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "ltx: text encoder requires the V2 (PER_TOKEN_RMS) caption feature extractor; config \
             selected {:?}, which this port does not implement",
            cfg.version
        )));
    }
    Ok(())
}

#[cfg(test)]
mod version_gate_tests {
    use super::*;

    #[test]
    fn require_v2_accepts_the_shipped_ltx_2_3_flags() {
        require_v2().expect("the hardcoded LTX-2.3 flags must resolve to V2");
    }
}
