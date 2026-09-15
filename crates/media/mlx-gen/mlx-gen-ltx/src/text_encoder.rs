//! LTX-2.3 text encoder — the full S1 path producing `video_embeddings` from token ids.
//!
//! Port of `text_encoder.py::LTX2TextEncoder.encode` (the 2.3 "v2" / per-token-RMS feature path):
//!   Gemma-3-12B (49 hidden states) → `norm_and_concat_per_token_rms` (3840×49 = 188160)
//!   → `rescale_norm(√(out/hidden))` → `video_aggregate_embed` Linear (188160 → 4096)
//!   → `Embeddings1DConnector` → `video_embeddings` (1, L, 4096).
//!
//! Runs **bf16** end-to-end to match the reference (gemma-3-12b-it-bf16 + bf16 activations).
//! `video_aggregate_embed.{weight,bias}` and the connector both live in `connector.safetensors`
//! (`text_embedding_projection.video_aggregate_embed.*`, `video_embeddings_connector.*`).
//!
//! The **AudioVideo** path (sc-2684) reuses the shared Gemma hiddens + per-token-RMS `normed_hidden`
//! and adds a parallel **audio** head: `text_embedding_projection.audio_aggregate_embed` (→ 2048) +
//! `audio_embeddings_connector` (8 layers, dim 2048 = 32×64). Built only by `from_weights_av`;
//! the video-only `from_weights` leaves it `None`.
//!
//! sc-18763: `normed_hidden` below is the V2 (`PER_TOKEN_RMS`) caption feature extractor, and it's
//! the ONLY one this crate implements. That was previously implicit — nothing checked the loaded
//! model actually selected V2 before running this math. [`LtxTextEncoder::from_weights`] /
//! [`LtxTextEncoder::from_weights_av`] now require `ltx_cfg.caption_feature_version == V2` (see
//! `gen_core::ltx_checkpoint::caption_feature_version`, sc-18757's shared, upstream-mirroring
//! detector both backends fold onto as of the 2026-08-19 coordinator review), erroring loudly
//! instead of silently running V2 math against a V1-shaped checkpoint.

use mlx_rs::ops::{add, mean_axes, multiply, rsqrt, stack_axis};
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::ltx_checkpoint::CaptionFeatureVersion;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::LtxConfig;
use crate::connector::Connector;
use crate::gemma::{GemmaConfig, GemmaModel, GemmaQuant};
use crate::transformer::{Linear, Precision};

const RMS_EPS: f32 = 1e-6;

/// One modality's feature-extractor head: the `aggregate_embed` Linear (188160 → out_dim) + its
/// `rescale_norm` scalar + the `Embeddings1DConnector`.
///
/// `pub(crate)` so [`crate::gemma4_te`] (LTX-2.5, sc-18770) builds the SAME head off the SAME
/// `connector.safetensors` keys rather than duplicating the projection math. The head is
/// backbone-agnostic — it consumes the per-token-RMS `normed` tensor and knows nothing about which
/// Gemma generation produced the hidden states.
pub(crate) struct FeatureHead {
    /// The `aggregate_embed` Linear (188160 → out_dim). A [`Linear`] rather than a bare weight so
    /// the LTX-2.5 tiers can pack it: at 4096×188160 and 2048×188160 the two heads are 2.31 GB of
    /// bf16 between them (sc-18775), and a `Linear` binds the packed triple or a dense weight from
    /// the same call, keyed on whether `{prefix}.scales` is present. LTX-2.3's dense
    /// `connector.safetensors` therefore loads exactly as before.
    aggregate: Linear,
    rescale: Array, // √(out_dim / hidden) scalar in `dtype`
    connector: Connector,
}

impl FeatureHead {
    /// `rescale_norm(normed) → aggregate_embed → connector`. `normed` is the shared masked
    /// per-token-RMS `(1, L, 188160)`; `mask01` the `(1, L)` 1/0 attention mask.
    pub(crate) fn forward(&self, normed: &Array, mask01: &Array) -> Result<(Array, Array)> {
        let features = self.aggregate.forward(&multiply(normed, &self.rescale)?)?;
        let embeddings = self.connector.forward(&features, mask01)?;
        Ok((features, embeddings))
    }
}

/// The LTX-2.3 text encoder (Gemma backbone + per-token-RMS feature extractor + connector). Carries
/// a video head always and an optional audio head (sc-2684 AudioVideo path).
pub struct LtxTextEncoder {
    gemma: GemmaModel,
    video: FeatureHead,
    audio: Option<FeatureHead>,
    dtype: Dtype,
}

impl LtxTextEncoder {
    /// Build the **video-only** encoder from the Gemma weights + the LTX `connector.safetensors`.
    ///
    /// `prec` carries the compute dtype (bf16 to match the reference) **and** the checkpoint's quant
    /// geometry. `gemma_quant` selectively quantizes the Gemma backbone (from its snapshot
    /// `config.json`; `None` ⇒ dense bf16 — the default `…-bf16` snapshot). The connector and the
    /// feature-extractor Linear are dense-or-quantized per tensor: LTX-2.3 ships them dense in
    /// `connector.safetensors` and takes the dense arm, while an LTX-2.5 `q4`/`q8` tier packs them
    /// (sc-18775) and takes the quantized arm — the same `{prefix}.scales` predicate the DiT uses.
    pub fn from_weights(
        gemma_w: &Weights,
        connector_w: &Weights,
        gemma_cfg: GemmaConfig,
        gemma_quant: Option<GemmaQuant>,
        ltx_cfg: &LtxConfig,
        prec: Precision,
    ) -> Result<Self> {
        require_v2(ltx_cfg)?;
        let gemma = GemmaModel::from_weights(gemma_w, gemma_cfg, gemma_quant)?;
        let video = load_video_head(connector_w, gemma_cfg.hidden_size, ltx_cfg, prec)?;
        Ok(Self {
            gemma,
            video,
            audio: None,
            dtype: prec.dtype(),
        })
    }

    /// Build the **AudioVideo** encoder (sc-2684): the video head + the audio head
    /// (`audio_aggregate_embed` + `audio_embeddings_connector`, dim 2048 = 32 × 64).
    pub fn from_weights_av(
        gemma_w: &Weights,
        connector_w: &Weights,
        gemma_cfg: GemmaConfig,
        gemma_quant: Option<GemmaQuant>,
        ltx_cfg: &LtxConfig,
        prec: Precision,
    ) -> Result<Self> {
        require_v2(ltx_cfg)?;
        let gemma = GemmaModel::from_weights(gemma_w, gemma_cfg, gemma_quant)?;
        let video = load_video_head(connector_w, gemma_cfg.hidden_size, ltx_cfg, prec)?;
        let audio = load_audio_head(connector_w, gemma_cfg.hidden_size, ltx_cfg, prec)?;
        Ok(Self {
            gemma,
            video,
            audio: Some(audio),
            dtype: prec.dtype(),
        })
    }

    /// The Gemma-3 backbone, exposed so the prompt enhancer (sc-2845) can reuse the **already-loaded**
    /// text-encoder weights as an autoregressive LLM (the reference `enhance_t2v` path runs generation
    /// on `self.language_model`).
    pub fn gemma(&self) -> &GemmaModel {
        &self.gemma
    }

    /// Encode `(1, L)` token ids + `(1, L)` attention mask → `video_embeddings` `(1, L, 4096)`.
    pub fn encode(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        Ok(self.encode_with_features(input_ids, attention_mask)?.1)
    }

    /// Like [`encode`](Self::encode) but also returns the pre-connector `video_features` (the
    /// feature-extractor output) for stage localization.
    pub fn encode_with_features(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
    ) -> Result<(Array, Array)> {
        let hiddens = self.gemma.forward(input_ids, attention_mask)?; // 49 × (1, L, 3840)
        let normed = self.normed_hidden(&hiddens, attention_mask)?;
        self.video.forward(&normed, attention_mask)
    }

    /// AudioVideo encode (sc-2684): `(video_embeddings (1,L,4096), audio_embeddings (1,L,2048))`.
    /// Errors if this encoder was not built with [`Self::from_weights_av`].
    pub fn encode_av(&self, input_ids: &Array, attention_mask: &Array) -> Result<(Array, Array)> {
        let (_, _, ve, ae) = self.encode_av_with_features(input_ids, attention_mask)?;
        Ok((ve, ae))
    }

    /// AudioVideo encode returning `(video_features, audio_features, video_embeddings,
    /// audio_embeddings)` — the pre-connector features included for stage localization.
    pub fn encode_av_with_features(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
    ) -> Result<(Array, Array, Array, Array)> {
        let audio = self.audio.as_ref().ok_or_else(|| {
            Error::Msg("ltx_2_3: text encoder built without the audio head".into())
        })?;
        let hiddens = self.gemma.forward(input_ids, attention_mask)?;
        let normed = self.normed_hidden(&hiddens, attention_mask)?;
        let (vf, ve) = self.video.forward(&normed, attention_mask)?;
        let (af, ae) = audio.forward(&normed, attention_mask)?;
        Ok((vf, af, ve, ae))
    }

    /// `norm_and_concat_per_token_rms` — the shared masked per-token RMS `(1, L, 188160)`.
    /// Each `(token, layer)` slice over the 3840 hidden dim is RMS-normalized independently, the 49
    /// layers are concatenated **dim-major / layer-minor** (`d*49 + layer`, via stack+reshape), and
    /// padded positions are zeroed. The video / audio heads then rescale + aggregate this off it.
    ///
    /// The math itself lives in [`per_token_rms_normed_hidden`], shared with LTX-2.5's Gemma 4
    /// encoder (sc-18770) — same caption feature version, same extractor.
    fn normed_hidden(&self, hiddens: &[Array], attention_mask: &Array) -> Result<Array> {
        per_token_rms_normed_hidden(hiddens, attention_mask, self.dtype)
    }
}

/// The `aggregate_embed` Linear plus its `rescale_norm` scalar.
///
/// `out_dim` comes from the **bias**, not the weight: a quantized `aggregate_embed` stores its
/// weight packed as U32 with the input axis folded into the packing, so reading `shape()[0]` off
/// the packed tensor would still be right but reading it off the bias is right in both layouts
/// and needs no knowledge of the packing.
///
/// `hidden_size` is the backbone's hidden dim (Gemma 3 and Gemma 4 both ship 3840 here, but the
/// rescale is derived rather than assumed) — taken as a scalar rather than a `GemmaConfig` so
/// LTX-2.5's Gemma 4 path (sc-18770), whose config comes from `mlx_llm::ModelConfig`, shares this
/// loader instead of copying it.
pub(crate) fn load_aggregate(
    connector_w: &Weights,
    key_prefix: &str,
    hidden_size: i32,
    prec: Precision,
) -> Result<(Linear, Array)> {
    let bias_key = format!("{key_prefix}.bias");
    let out_dim = connector_w
        .get(&bias_key)
        .ok_or_else(|| Error::MissingTensor(bias_key.clone()))?
        .shape()[0];
    let aggregate = Linear::load(connector_w, key_prefix, prec)?;
    let rescale = Array::from_slice(&[(out_dim as f32 / hidden_size as f32).sqrt()], &[1])
        .as_dtype(prec.dtype())?;
    Ok((aggregate, rescale))
}

/// The video feature head: `text_embedding_projection.video_aggregate_embed` + the
/// `video_embeddings_connector` at the checkpoint's declared connector dims.
pub(crate) fn load_video_head(
    connector_w: &Weights,
    hidden_size: i32,
    ltx_cfg: &LtxConfig,
    prec: Precision,
) -> Result<FeatureHead> {
    let (aggregate, rescale) = load_aggregate(
        connector_w,
        "text_embedding_projection.video_aggregate_embed",
        hidden_size,
        prec,
    )?;
    let connector =
        Connector::from_weights(connector_w, "video_embeddings_connector.", ltx_cfg, prec)?;
    Ok(FeatureHead {
        aggregate,
        rescale,
        connector,
    })
}

/// The audio feature head: `text_embedding_projection.audio_aggregate_embed` + the
/// `audio_embeddings_connector`, which shares the checkpoint's layer count / theta / register
/// max-pos but runs at the audio connector dims (32 × 64 = 2048).
pub(crate) fn load_audio_head(
    connector_w: &Weights,
    hidden_size: i32,
    ltx_cfg: &LtxConfig,
    prec: Precision,
) -> Result<FeatureHead> {
    let (aggregate, rescale) = load_aggregate(
        connector_w,
        "text_embedding_projection.audio_aggregate_embed",
        hidden_size,
        prec,
    )?;
    let connector = Connector::from_weights_dims(
        connector_w,
        "audio_embeddings_connector.",
        ltx_cfg.connector_num_layers,
        ltx_cfg.audio_connector_num_attention_heads,
        ltx_cfg.audio_connector_attention_head_dim,
        ltx_cfg.positional_embedding_theta,
        ltx_cfg.connector_positional_embedding_max_pos,
        ltx_cfg.connector_ff_bias,
        prec,
    )?;
    Ok(FeatureHead {
        aggregate,
        rescale,
        connector,
    })
}

/// `norm_and_concat_per_token_rms` — the shared masked per-token RMS `(1, L, hidden × n_states)`.
///
/// Each `(token, layer)` slice over the hidden dim is RMS-normalized independently, the states are
/// concatenated **dim-major / layer-minor** (`d*n + layer`, via stack+reshape), and padded positions
/// are zeroed.
///
/// This is the V2 (`PER_TOKEN_RMS`) caption feature extractor, shared verbatim by LTX-2.3's Gemma 3
/// encoder and LTX-2.5's Gemma 4 encoder (sc-18770): the math is a property of the caption feature
/// version, not of the backbone, and both checkpoints declare V2. Generic over the state count and
/// hidden size — it reads both off the stacked tensor — so a backbone with a different depth flows
/// through unchanged.
pub(crate) fn per_token_rms_normed_hidden(
    hiddens: &[Array],
    attention_mask: &Array,
    dtype: Dtype,
) -> Result<Array> {
    let refs: Vec<&Array> = hiddens.iter().collect();
    let encoded = stack_axis(&refs, 3)?; // (1, L, hidden, n)
    let sh = encoded.shape();
    let (b, l) = (sh[0], sh[1]);
    // per-token RMS over the hidden dim (axis 2), per layer.
    let var = mean_axes(&multiply(&encoded, &encoded)?, &[2], true)?; // (1, L, 1, n)
    let eps = Array::from_slice(&[RMS_EPS], &[1]).as_dtype(dtype)?;
    let normed = multiply(&encoded, &rsqrt(&add(&var, &eps)?)?)?;
    let normed = normed.reshape(&[b, l, -1])?; // (1, L, hidden × n), dim-major/layer-minor
                                               // zero padded token positions (multiply by the 0/1 mask == where(mask, x, 0)).
    let mask = attention_mask.reshape(&[b, l, 1])?.as_dtype(dtype)?;
    Ok(multiply(&normed, &mask)?)
}

/// sc-18763: reject construction against anything but a V2-selected config. `normed_hidden` above
/// is the V2 math unconditionally — running it against a V1-shaped checkpoint would silently
/// produce plausible-looking, wrong conditioning (the exact failure mode the story called out).
pub(crate) fn require_v2(ltx_cfg: &LtxConfig) -> Result<()> {
    if ltx_cfg.caption_feature_version != CaptionFeatureVersion::V2 {
        return Err(Error::Msg(format!(
            "ltx: text encoder requires the V2 (PER_TOKEN_RMS) caption feature extractor; config \
             selected {:?}, which this port does not implement",
            ltx_cfg.caption_feature_version
        )));
    }
    Ok(())
}

#[cfg(test)]
mod version_gate_tests {
    use super::*;
    use crate::config::LtxConfig;

    // `LtxConfig::validated` (the fallible caption-detection step) is private to `crate::config`,
    // so these synthetic configs set the field directly rather than routing through it — `require_v2`
    // only cares about the field's value, not how a real load path resolved it. The detection logic
    // itself (`gen_core::ltx_checkpoint::caption_feature_version`) has its own coverage in
    // `crate::config`'s tests and in `gen_core::ltx_checkpoint`'s own test module.

    fn v1_config() -> LtxConfig {
        let mut cfg = LtxConfig::video_only_defaults();
        cfg.caption_feature_version = CaptionFeatureVersion::V1;
        cfg
    }

    fn v2_config() -> LtxConfig {
        let mut cfg = LtxConfig::video_only_defaults();
        cfg.caption_feature_version = CaptionFeatureVersion::V2;
        cfg
    }

    #[test]
    fn require_v2_accepts_v2_config() {
        require_v2(&v2_config()).expect("V2 config must be accepted");
    }

    #[test]
    fn require_v2_rejects_v1_config_loudly() {
        let err = require_v2(&v1_config()).expect_err("V1 config must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("V2"), "unexpected error message: {msg}");
        assert!(msg.contains("V1"), "unexpected error message: {msg}");
    }
}
