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
//! extractor, and it's the ONLY one this crate implements. Construction (both
//! [`LtxTextEncoder::new`] and [`LtxTextEncoder::new_av`], via `require_v2` below) validates
//! against [`AvConfig::ltx_2_3`](crate::config::AvConfig::ltx_2_3)'s `caption_feature_version` —
//! the same production-canonical LTX-2.3 constant the AvDiT build path uses, itself validated
//! through `gen_core::ltx_checkpoint::caption_feature_version` (sc-18757's shared, upstream-
//! mirroring detector both backends fold onto as of the 2026-08-19 coordinator review — this
//! crate previously carried its own duplicate detection logic here).
//!
//! This is still a **compile-time constant check**, not a per-checkpoint one: `LtxTextEncoder`'s
//! constructors don't yet take a loaded `AvConfig`/bundle (that threading is a separate, larger
//! surface change outside this story's scope), so the gate below only catches the constant
//! drifting away from what the shared detector accepts — not a real checkpoint's own config. Real
//! per-checkpoint threading for this specific gate remains open, matching `AvConfig`'s own
//! module-level note on split-bundle loading (sc-18757).

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::ltx_checkpoint::CaptionFeatureVersion;

use crate::config::{AvConfig, ConnectorConfig, GemmaConfig};
use crate::connector::Connector;
use crate::gemma::GemmaEncoder;
use crate::quant::{qlinear, QLinear};

const RMS_EPS: f64 = 1e-6;

/// The audio text head (sc-5495): a separate aggregate projection (188160 → 2048) + rescale +
/// `audio_embeddings_connector`, sharing the same Gemma hidden states as the video head.
///
/// `pub(crate)` so [`crate::gemma4_te`] (LTX-2.5, sc-18770) builds the SAME head off the SAME keys
/// rather than duplicating the projection math. The head is backbone-agnostic — it consumes the
/// per-token-RMS `normed` tensor and knows nothing about which Gemma generation produced it.
pub(crate) struct AudioHead {
    pub(crate) aggregate: QLinear, // [2048, 188160] + bias (packed-detected, sc-9417)
    pub(crate) rescale: f64,       // √(2048 / 3840)
    pub(crate) connector: Connector,
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
    /// The math itself lives in [`per_token_rms_normed_hidden`], shared with LTX-2.5's Gemma 4
    /// encoder (sc-18770) — same caption feature version, same extractor.
    fn normed_hidden(&self, hiddens: &[Tensor], mask01: &[u32]) -> Result<Tensor> {
        per_token_rms_normed_hidden(hiddens, mask01, self.hidden_size, &self.device)
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
/// ([`AvConfig::ltx_2_3`](crate::config::AvConfig::ltx_2_3)'s `caption_feature_version`), not a
/// live per-checkpoint config value — `LtxTextEncoder`'s constructors don't yet take a loaded
/// `AvConfig`, so this is not the same thing as the split-bundle, per-checkpoint reads
/// `AvConfig::from_bundle` performs elsewhere in this crate (sc-18757). It still catches a real
/// class of bug (someone editing the constant away from what the shared detector accepts without
/// noticing), just not a per-checkpoint one yet. Do not describe this as "config-driven" the way
/// the mlx backend's genuinely-JSON-driven check is (sc-18763 coordinator review).
fn require_v2() -> Result<()> {
    require_v2_version(AvConfig::ltx_2_3().caption_feature_version)
}

/// The gate itself, over a caption-feature version the caller supplies.
///
/// Split out (sc-18770) so LTX-2.5 runs the SAME gate rather than a parallel copy of it — and, on
/// the 2.5 path, feeds it a **genuinely per-checkpoint** value (`AvConfig::from_bundle`'s
/// `caption_feature_version`, resolved by the shared detector off the loaded transformer config)
/// instead of the compile-time constant `require_v2` above is stuck with. LTX-2.3's behavior is
/// unchanged: it still passes the constant, and still catches only constant drift.
pub(crate) fn require_v2_version(version: CaptionFeatureVersion) -> Result<()> {
    if version != CaptionFeatureVersion::V2 {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "ltx: text encoder requires the V2 (PER_TOKEN_RMS) caption feature extractor; config \
             selected {version:?}, which this port does not implement"
        )));
    }
    Ok(())
}

/// `norm_and_concat_per_token_rms`: stack the hidden states `[1,L,hidden,n]`, RMS-normalize each
/// `(token, layer)` slice over the hidden dim, flatten dim-major/layer-minor `[1,L,hidden×n]`, zero
/// the padded positions.
///
/// This is the V2 (`PER_TOKEN_RMS`) caption feature extractor, shared verbatim by LTX-2.3's Gemma 3
/// encoder and LTX-2.5's Gemma 4 encoder (sc-18770): the math is a property of the caption feature
/// version, not of the backbone, and both checkpoints declare V2. Generic over the state count — it
/// reads `n` off the stacked tensor — so a backbone with a different depth flows through unchanged.
pub(crate) fn per_token_rms_normed_hidden(
    hiddens: &[Tensor],
    mask01: &[u32],
    hidden_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let refs: Vec<&Tensor> = hiddens.iter().collect();
    let enc = Tensor::stack(&refs, 3)?; // (1, L, hidden, n)
    let (b, l, _, n) = enc.dims4()?;
    let var = enc.sqr()?.mean_keepdim(2)?; // (1, L, 1, n)
    let inv = (var + RMS_EPS)?.sqrt()?.recip()?;
    let normed = enc.broadcast_mul(&inv)?;
    let normed = normed.reshape((b, l, hidden_size * n))?; // (1, L, hidden × n)
                                                           // Zero padded token positions.
    let mask: Vec<f32> = mask01.iter().map(|&m| m as f32).collect();
    let mask = Tensor::from_vec(mask, (1, l, 1), device)?.to_dtype(DType::BF16)?;
    normed.broadcast_mul(&mask)
}

/// The `aggregate_embed` projection plus its `rescale_norm` scalar, packed-detected per tensor.
///
/// `out_dim` comes from the connector config rather than a weight-shape probe, because the packed
/// path has no dense weight to probe (the same reason [`LtxTextEncoder::new`] reads it that way).
pub(crate) fn load_aggregate(
    proj_vb: &VarBuilder,
    key: &str,
    out_dim: usize,
    hidden_size: usize,
) -> Result<(QLinear, f64)> {
    let aggregate = qlinear(proj_vb, key, true)?;
    let rescale = (out_dim as f64 / hidden_size as f64).sqrt();
    Ok((aggregate, rescale))
}

#[cfg(test)]
mod version_gate_tests {
    use super::*;

    #[test]
    fn require_v2_accepts_the_shipped_ltx_2_3_flags() {
        require_v2().expect("the hardcoded LTX-2.3 flags must resolve to V2");
    }
}
