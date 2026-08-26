//! LTX-2.5 text encoder — the Gemma 4 backbone from the **shared LLM crate** plus LTX's own
//! projection + feature-extraction heads (sc-18770). The candle twin of `mlx-gen-ltx`'s
//! `gemma4_te.rs`.
//!
//! ```text
//! packed gemma4-12b-with-proj TE  →  candle_llm::models::CausalLm (Gemma 4 unified, 48 layers)
//!   → 49 hidden states (1, L, 3840)
//!   → per-token-RMS concat (3840 × 49 = 188160)          [shared with LTX-2.3, V2 extractor]
//!   → ×√(out/hidden) → {video,audio}_aggregate_embed     [LTX-specific]
//!   → Embeddings1DConnector                              [LTX-specific]
//!   → video_embeddings (1, L, 4096) / audio_embeddings (1, L, 2048)
//! ```
//!
//! **What is LTX-specific and therefore lives here** is deliberately small: the two
//! `text_embedding_projection.*_aggregate_embed` heads, the dual-modality wiring, the packed-asset
//! load path, and the `gemma_source_checkpoint.gemma_version` assertion. Everything about *being a
//! Gemma 4 decoder* — the dual head dims, the dual RoPE schedules, `attention_k_eq_v`, the
//! `layer_scalar` buffers, KV sharing — belongs to `candle-llm` and is not restated here.
//!
//! **LTX-2.3 is untouched.** [`crate::gemma`]'s Gemma 3 encoder and
//! [`LtxTextEncoder`](crate::text_encoder::LtxTextEncoder) keep running 2.3 exactly as before; this
//! module shares their feature-extractor math and head loaders rather than replacing them.
//!
//! # The version gate, threaded properly
//!
//! LTX-2.3's `require_v2` on this backend is a documented compromise: it checks a **compile-time
//! constant**, because `LtxTextEncoder`'s constructors never took a loaded config. This module does
//! not inherit that compromise — it runs the same shared gate
//! (`text_encoder::require_v2_version`, crate-private) against the
//! [`AvConfig::caption_feature_version`](crate::config::AvConfig) resolved by
//! `AvConfig::from_bundle` off the real transformer config. Wired *through* the existing gate, not
//! around it, and fed a real per-checkpoint value.

use std::path::Path;

use candle_gen::candle_core::{Device, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::gemma_assets::GemmaAssets;
use candle_gen::gen_core::ltx_checkpoint::{
    check_gemma_version, GemmaEncoderIdentity, GemmaVersionCheck, LtxCheckpointMetadata,
};

use candle_llm::config::ModelConfig;
use candle_llm::models::CausalLm;
use candle_llm::primitives::Weights as LlmWeights;

use crate::config::{AvConfig, ConnectorConfig};
use crate::connector::Connector;
use crate::quant::QLinear;
use crate::text_encoder::{
    load_aggregate, per_token_rms_normed_hidden, require_v2_version, AudioHead,
};

/// Map a `gen_core` error into candle's `Result`, which is what this crate's tensor paths speak.
///
/// The widening direction only works one way (`CandleError: From<candle_core::Error>` but not the
/// reverse — the orphan rule forbids it), so a `gen_core::Error` arriving here is stringified into
/// `candle_core::Error::Msg`. The message is preserved verbatim, which is what the R4 assertion
/// needs: the mismatch text from `check_gemma_version` reaches the caller intact.
fn from_gen_core(e: candle_gen::gen_core::Error) -> candle_gen::candle_core::Error {
    candle_gen::candle_core::Error::Msg(e.to_string())
}

/// The LTX-2.5 text encoder: a shared Gemma 4 decoder plus LTX's video head and audio head.
pub struct Ltx25TextEncoder {
    model: CausalLm,
    aggregate: QLinear,
    rescale: f64,
    connector: Connector,
    audio: Option<AudioHead>,
    hidden_size: usize,
    device: Device,
}

impl Ltx25TextEncoder {
    /// Build the **AudioVideo** encoder.
    ///
    /// * `checkpoint` — the transformer's metadata, which declares `gemma_source_checkpoint`. The
    ///   version assertion needs both halves of the pair, so there is no constructor that omits it;
    ///   an escape hatch here would silently disable the assertion R4 requires.
    /// * `te_path` — the packed `gemma4-12b-with-proj` encoder (config, weights and HF assets all
    ///   inside one safetensors file).
    /// * `proj_vb` — rooted at the checkpoint top level, for `text_embedding_projection.*`.
    /// * `dit_vb` — rooted at `model.diffusion_model.`, for the connectors.
    /// * `av_cfg` — the loaded `AvConfig`; its `caption_feature_version` is the real, per-checkpoint
    ///   value the V2 gate is fed.
    ///
    /// The tier converter (sc-18775) moves `text_embedding_projection.*` **out** of the packed
    /// encoder and into `connector.safetensors`, so `proj_vb` resolves the heads from the same file
    /// and the same keys as LTX-2.3's.
    #[allow(clippy::too_many_arguments)]
    pub fn from_packed_av(
        checkpoint: &LtxCheckpointMetadata,
        te_path: &Path,
        proj_vb: VarBuilder,
        dit_vb: VarBuilder,
        av_cfg: &AvConfig,
        conn_cfg: &ConnectorConfig,
        audio_conn_cfg: &ConnectorConfig,
    ) -> Result<Self> {
        // (1) The caption feature extractor must be the V2 (PER_TOKEN_RMS) one this port implements
        // — checked against the loaded checkpoint's own config, not a constant.
        require_v2_version(av_cfg.caption_feature_version)?;

        // (2) R4: the encoder must be the one this checkpoint was trained against. A mismatch is a
        // hard error from `gen_core`, never a warning and never a fallback.
        assert_gemma_version_for(checkpoint, te_path)?;

        let device = proj_vb.device().clone();

        // (3) The Gemma 4 config travels inside the packed encoder's `__metadata__.gemma_config`.
        // `ModelConfig::from_json` rebinds to `text_config` for Gemma 4 (`nests_text_config`), which
        // is also where a tier's `quantization` block is stamped — a block written above it would be
        // invisible and the packed encoder would load as if dense.
        let config = gemma_config_value(te_path)?;
        let cfg = ModelConfig::from_json(&config).map_err(from_llm)?;
        let hidden_size = cfg.hidden_size as usize;

        // (4) Load through the same seam the shipped tiers were validated on (PR #820).
        let weights = LlmWeights::from_file(te_path, &device).map_err(from_llm)?;
        let model = CausalLm::from_weights(&weights, "", cfg).map_err(from_llm)?;

        let (aggregate, rescale) = load_aggregate(
            &proj_vb,
            "text_embedding_projection.video_aggregate_embed",
            conn_cfg.inner_dim(),
            hidden_size,
        )?;
        let connector = Connector::new(dit_vb.clone(), conn_cfg)?;

        let (audio_aggregate, audio_rescale) = load_aggregate(
            &proj_vb,
            "text_embedding_projection.audio_aggregate_embed",
            audio_conn_cfg.inner_dim(),
            hidden_size,
        )?;
        let audio_connector =
            Connector::new_with_prefix(dit_vb, audio_conn_cfg, "audio_embeddings_connector")?;

        Ok(Self {
            model,
            aggregate,
            rescale,
            connector,
            audio: Some(AudioHead {
                aggregate: audio_aggregate,
                rescale: audio_rescale,
                connector: audio_connector,
            }),
            hidden_size,
            device,
        })
    }

    /// The Gemma 4 backbone, exposed so the LTX-2.5 prompt enhancer (sc-18764) can drive the
    /// **already-loaded** encoder as an autoregressive LM instead of loading a second copy.
    pub fn model(&self) -> &CausalLm {
        &self.model
    }

    /// The hidden-state stack.
    ///
    /// candle is eager, so unlike the MLX twin there is no lazy-graph hazard to defend against here
    /// — the states are materialized as they are produced. The method exists so both backends read
    /// the same way at the call sites below, and so the cross-backend gate has one named seam.
    fn hidden_states(&self, input_ids: &Tensor) -> Result<Vec<Tensor>> {
        let mut cache = self.model.new_cache();
        self.model
            .hidden_states(input_ids, &mut cache, 0)
            .map_err(from_llm)
    }

    /// Encode `input_ids` `[1,L]` (u32) + `mask01` → `video_embeddings` `[1, L, 4096]` (bf16).
    pub fn encode(&self, input_ids: &Tensor, mask01: &[u32]) -> Result<Tensor> {
        Ok(self.encode_with_features(input_ids, mask01)?.1)
    }

    /// Like [`Self::encode`] but also returns the pre-connector `video_features` — the
    /// **connector input** (post-projection, post-norm), which is the tensor the parity gates check.
    pub fn encode_with_features(
        &self,
        input_ids: &Tensor,
        mask01: &[u32],
    ) -> Result<(Tensor, Tensor)> {
        let hiddens = self.hidden_states(input_ids)?;
        let normed = per_token_rms_normed_hidden(&hiddens, mask01, self.hidden_size, &self.device)?;
        let features = self.aggregate.forward(&(normed * self.rescale)?)?;
        let nv = mask01.iter().filter(|&&m| m != 0).count();
        let embeddings = self.connector.forward(&features, nv)?;
        Ok((features, embeddings))
    }

    /// AudioVideo encode: `(video_embeddings [1,L,4096], audio_embeddings [1,L,2048])`.
    pub fn encode_both(&self, input_ids: &Tensor, mask01: &[u32]) -> Result<(Tensor, Tensor)> {
        let (_, _, video, audio) = self.encode_both_with_features(input_ids, mask01)?;
        Ok((video, audio))
    }

    /// Like [`Self::encode_both`] but also returns the two **connector inputs**
    /// `(video_features, audio_features)`, mirroring `mlx-gen-ltx`'s `encode_av_with_features` so
    /// the cross-backend gate compares like with like.
    pub fn encode_both_with_features(
        &self,
        input_ids: &Tensor,
        mask01: &[u32],
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let audio = self.audio.as_ref().ok_or_else(|| {
            candle_gen::candle_core::Error::Msg(
                "ltx_2_5: text encoder built without the audio head".into(),
            )
        })?;
        let hiddens = self.hidden_states(input_ids)?;
        let normed = per_token_rms_normed_hidden(&hiddens, mask01, self.hidden_size, &self.device)?;
        let nv = mask01.iter().filter(|&&m| m != 0).count();
        let v_feat = self.aggregate.forward(&(normed.clone() * self.rescale)?)?;
        let video = self.connector.forward(&v_feat, nv)?;
        let a_feat = audio.aggregate.forward(&(normed * audio.rescale)?)?;
        let audio_ctx = audio.connector.forward(&a_feat, nv)?;
        Ok((v_feat, a_feat, video, audio_ctx))
    }
}

/// Lift a `candle-llm` engine error into candle's `Result`, preserving the message.
fn from_llm(e: candle_llm::Error) -> candle_gen::candle_core::Error {
    candle_gen::candle_core::Error::Msg(e.to_string())
}

/// Read the packed encoder's `__metadata__.gemma_config` as JSON.
fn gemma_config_value(te_path: &Path) -> Result<serde_json::Value> {
    let assets = GemmaAssets::load(te_path).map_err(from_gen_core)?;
    serde_json::from_str(assets.config_json()).map_err(|e| {
        candle_gen::candle_core::Error::Msg(format!(
            "ltx_2_5: the Gemma config packed in {} is not valid JSON: {e}",
            te_path.display()
        ))
    })
}

/// R4: assert the packed encoder is the one this checkpoint declares, failing loudly on mismatch.
///
/// Delegates the comparison to `gen_core::ltx_checkpoint::check_gemma_version` — upstream's
/// `encoder_configurator._check_gemma_version` — rather than re-implementing the rules. The returned
/// [`GemmaVersionCheck`] is deliberately **not** discarded. Both non-`Matched` outcomes are
/// legitimate answers for *some* pair and so cannot be rejected by gen-core, but neither is
/// acceptable on the 2.5 path:
///
/// * [`GemmaVersionCheck::Gemma3Legacy`] — the encoder declares `model_type: gemma3` against a
///   pre-2.4 checkpoint. That is exactly the LTX-2.3 pair, i.e. this loader was pointed at the 2.3
///   Gemma 3 snapshot. (A *Gemma 4* encoder against a pre-2.4 checkpoint never reaches here —
///   gen-core rejects the `model_type` mismatch itself.)
/// * [`GemmaVersionCheck::SkippedNoDeclaredVersion`] — nothing on either side to compare, so the
///   encoder identity is unverified. Upstream warns and continues; R4 says fail.
fn assert_gemma_version_for(checkpoint: &LtxCheckpointMetadata, te_path: &Path) -> Result<()> {
    let encoder = GemmaEncoderIdentity::from_single_file(te_path).map_err(from_gen_core)?;
    match check_gemma_version(checkpoint, &encoder).map_err(from_gen_core)? {
        GemmaVersionCheck::Matched(_) => Ok(()),
        GemmaVersionCheck::Gemma3Legacy => Err(candle_gen::candle_core::Error::Msg(format!(
            "ltx_2_5: the text encoder at {} declares model_type gemma3 and was paired with a \
             pre-2.4 checkpoint, so the pair resolved down the LTX-2.3 route. That pair is valid \
             for LTX-2.3, but this is the 2.5 encoder path, which requires a Gemma 4 encoder and a \
             2.5 transformer — load it through LtxTextEncoder instead",
            te_path.display()
        ))),
        GemmaVersionCheck::SkippedNoDeclaredVersion => {
            Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx_2_5: neither the checkpoint nor the Gemma config at {} declares a \
                 gemma_version, so the encoder identity cannot be verified. A 2.5 text encoder must \
                 declare gemma_version and be paired with a checkpoint that declares \
                 gemma_source_checkpoint",
                te_path.display()
            )))
        }
    }
}

#[cfg(test)]
mod version_assertion_tests {
    use super::*;
    use candle_gen::candle_core::DType;
    use std::collections::BTreeMap;

    const GEMMA4_VERSION: &str = "gemma4-12b-ltx-v1";

    /// Write a safetensors file carrying only `__metadata__` — enough for the identity read, which
    /// gen-core documents as config-only. Byte-for-byte the same fixture shape `mlx-gen-ltx`'s twin
    /// uses, so both backends assert against identical inputs.
    fn write_te(dir: &Path, name: &str, gemma_config: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        let header =
            serde_json::json!({ "__metadata__": { "gemma_config": gemma_config } }).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(&path, bytes).expect("write packed te");
        path
    }

    fn gemma4_config(version: &str) -> String {
        serde_json::json!({ "model_type": "gemma4_unified", "gemma_version": version }).to_string()
    }

    fn checkpoint(model_version: &str, gemma_version: Option<&str>) -> LtxCheckpointMetadata {
        let mut raw = BTreeMap::new();
        raw.insert("model_version".to_string(), model_version.to_string());
        if let Some(v) = gemma_version {
            raw.insert(
                "gemma_source_checkpoint".to_string(),
                serde_json::json!({ "ltx_version": model_version, "gemma_version": v }).to_string(),
            );
        }
        LtxCheckpointMetadata::from_raw(Path::new("transformer.safetensors"), raw)
            .expect("checkpoint metadata")
    }

    #[test]
    fn a_matching_gemma_version_is_accepted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let te = write_te(dir.path(), "te.safetensors", &gemma4_config(GEMMA4_VERSION));
        assert_gemma_version_for(&checkpoint("2.5.0", Some(GEMMA4_VERSION)), &te)
            .expect("the declared pair must be accepted");
    }

    #[test]
    fn a_mismatched_gemma_version_fails_loudly() {
        let dir = tempfile::tempdir().expect("tempdir");
        let te = write_te(
            dir.path(),
            "te.safetensors",
            &gemma4_config("gemma4-12b-ltx-v2"),
        );
        let err = assert_gemma_version_for(&checkpoint("2.5.0", Some(GEMMA4_VERSION)), &te)
            .expect_err("a gemma_version mismatch must be a hard error, never a warning");
        let msg = err.to_string();
        assert!(
            msg.contains("gemma4-12b-ltx-v1") && msg.contains("gemma4-12b-ltx-v2"),
            "the error must name BOTH versions so the mismatch is diagnosable: {msg}"
        );
    }

    /// Pointing the 2.5 loader at the LTX-2.3 Gemma 3 snapshot is the case that reaches the
    /// `Gemma3Legacy` arm: gen-core answers `Gemma3Legacy` (correct — that pair IS valid 2.3), so
    /// only this layer can refuse it.
    #[test]
    fn the_2_3_gemma3_pair_is_refused_by_the_2_5_loader() {
        let dir = tempfile::tempdir().expect("tempdir");
        let te = write_te(
            dir.path(),
            "te.safetensors",
            &serde_json::json!({ "model_type": "gemma3" }).to_string(),
        );
        let err = assert_gemma_version_for(&checkpoint("2.3.0", None), &te)
            .expect_err("the 2.3 pair must not be accepted by the 2.5 loader");
        assert!(
            err.to_string().contains("LTX-2.3"),
            "the error must name the route it resolved down: {err}"
        );
    }

    #[test]
    fn an_undeclared_gemma_version_is_rejected_rather_than_skipped() {
        let dir = tempfile::tempdir().expect("tempdir");
        let te = write_te(
            dir.path(),
            "te.safetensors",
            &serde_json::json!({ "model_type": "gemma4_unified" }).to_string(),
        );
        let err = assert_gemma_version_for(&checkpoint("2.5.0", None), &te)
            .expect_err("an unverifiable identity must not silently pass");
        assert!(
            err.to_string().contains("gemma_version"),
            "the error must name what was missing: {err}"
        );
    }

    /// The V2 gate must reject a non-V2 selection. The 2.5 path feeds it a real per-checkpoint
    /// value, so this is the assertion standing between a V1-shaped config and V2 math silently
    /// producing plausible-looking, wrong conditioning.
    #[test]
    fn the_v2_gate_rejects_a_v1_selection() {
        use candle_gen::gen_core::ltx_checkpoint::CaptionFeatureVersion;
        require_v2_version(CaptionFeatureVersion::V2).expect("V2 must be accepted");
        let err = require_v2_version(CaptionFeatureVersion::V1).expect_err("V1 must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("V2") && msg.contains("V1"),
            "unexpected error message: {msg}"
        );
    }

    /// The masking half of the shared V2 extractor, asserted identically to the MLX twin: padded
    /// positions are zeroed and every `(token, layer)` slice normalizes independently.
    #[test]
    fn the_shared_extractor_zeroes_padding_and_normalizes_per_token_per_layer() {
        let device = Device::Cpu;
        let (hidden, states, len) = (4usize, 3usize, 2usize);
        let hiddens: Vec<Tensor> = (0..states)
            .map(|s| {
                Tensor::from_vec(
                    vec![(s + 1) as f32; len * hidden],
                    (1, len, hidden),
                    &device,
                )
                .and_then(|t| t.to_dtype(DType::BF16))
                .expect("hidden state")
            })
            .collect();
        // Token 0 is padding, token 1 is valid.
        let mask01 = vec![0u32, 1];

        let normed =
            per_token_rms_normed_hidden(&hiddens, &mask01, hidden, &device).expect("extractor");
        assert_eq!(normed.dims3().expect("rank 3"), (1, len, hidden * states));

        let flat = normed
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("host copy");
        let width = hidden * states;
        assert!(
            flat[..width].iter().all(|v| *v == 0.0),
            "padded positions must be zeroed, got {:?}",
            &flat[..width]
        );
        assert!(
            flat[width..].iter().all(|v| (*v - 1.0).abs() < 1e-2),
            "each per-token-RMS slice must normalize to ~1, got {:?}",
            &flat[width..]
        );
    }
}
