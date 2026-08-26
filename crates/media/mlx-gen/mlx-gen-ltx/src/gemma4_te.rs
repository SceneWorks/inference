//! LTX-2.5 text encoder — the Gemma 4 backbone from the **shared LLM crate** plus LTX's own
//! projection + feature-extraction heads (sc-18770).
//!
//! ```text
//! packed gemma4-12b-with-proj TE  →  mlx_llm::CausalLm (Gemma 4 unified, 48 layers)
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
//! `layer_scalar` buffers, KV sharing — belongs to `mlx-llm` and is not restated here. This module
//! is the first `mlx-gen-* → mlx-llm` consumer, the direction `mlx-llm/Cargo.toml` has documented
//! as the intended inversion since that crate was created.
//!
//! **LTX-2.3 is untouched.** [`crate::gemma`]'s Gemma 3 backbone and [`crate::text_encoder`]'s
//! [`LtxTextEncoder`](crate::text_encoder::LtxTextEncoder) keep running 2.3 exactly as before;
//! this module shares their feature-extractor math and head loaders rather than replacing them.
//!
//! # Two hazards this module exists to get right
//!
//! 1. **MLX laziness on the hidden-state stack.** `hidden_states` returns 49 handles onto ONE
//!    unevaluated graph. Forcing the last one submits every weight page-in and every layer's
//!    matmuls as a single Metal command buffer, which the real 26.3 GB encoder exceeds — it comes
//!    back as `kIOGPUCommandBufferCallbackErrorSubmissionsIgnored`, and the poisoned process fails
//!    every *later* submission too. `Ltx25TextEncoder::hidden_states_in_order` walks the stack in
//!    order so each layer is its own command buffer. See the `hidden_states_from_embeds` doc in
//!    `mlx-llm` for the measurement.
//! 2. **Cold mmapped weights.** `Weights::from_file` is lazy; the first graph over a cold tier drags
//!    the whole file into one buffer and trips the watchdog. [`materialize_in_batches`] forces the
//!    tensors resident in bounded submissions before anything builds a graph over them.

use std::path::Path;

use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::gemma_assets::GemmaAssets;
use mlx_gen::gen_core::ltx_checkpoint::{
    check_gemma_version, GemmaEncoderIdentity, GemmaVersionCheck, LtxCheckpointMetadata,
    LtxComponent,
};
use mlx_gen::gen_core::LtxBundle;
use mlx_gen::weights::Weights as GenWeights;
use mlx_gen::{Error, Result};

use mlx_llm::config::ModelConfig;
use mlx_llm::primitives::Weights as LlmWeights;
use mlx_llm::CausalLm;

use crate::config::LtxConfig;
use crate::text_encoder::{
    load_audio_head, load_video_head, per_token_rms_normed_hidden, require_v2, FeatureHead,
};
use crate::transformer::Precision;

/// Byte budget for one `eval` submission when forcing a cold tier resident.
///
/// The same 512 MB bound `crate::tiers` uses for its conversion evals and `mlx-llm`'s tier-quality
/// measurement uses for its loads. Large enough that materialization is not dominated by submission
/// overhead, small enough that every buffer stays well inside the GPU watchdog's budget.
const EVAL_BATCH_BYTES: usize = 512 * 1024 * 1024;

/// Lift an `mlx-llm` engine error into this crate's error type, **preserving the typed variants**.
///
/// `Unsupported` and `Canceled` are contract-load-bearing on the gen-core side (the worker and the
/// conformance testkit match on them), so they must cross this seam as themselves rather than being
/// stringified into `Msg` — the same rule `gen_core::Error::backend` documents.
fn from_llm(e: mlx_llm::Error) -> Error {
    match e {
        mlx_llm::Error::Unsupported(m) => Error::Unsupported(m),
        mlx_llm::Error::Canceled => Error::Canceled,
        mlx_llm::Error::MissingTensor(k) => Error::MissingTensor(k),
        mlx_llm::Error::Io(e) => Error::Io(e),
        other => Error::Msg(other.to_string()),
    }
}

/// Force a checkpoint's mmapped tensors resident in **bounded batches**, before anything builds a
/// graph over them.
///
/// See the module docs, hazard 2. Left lazy, the first `eval` of a forward pulls the whole cold file
/// — 7.7 GB at q4, 24 GB at bf16 — into a single Metal command buffer and the watchdog kills it with
/// `kIOGPUCommandBufferCallbackErrorTimeout`. One killed buffer poisons the process, so the failure
/// is not even local to the load that caused it.
pub fn materialize_in_batches(weights: &LlmWeights) -> Result<()> {
    let keys: Vec<String> = weights.keys().map(str::to_string).collect();
    let mut batch: Vec<Array> = Vec::new();
    let mut bytes = 0usize;
    for key in &keys {
        let array = weights.require(key).map_err(from_llm)?.clone();
        bytes = bytes.saturating_add(array.nbytes());
        batch.push(array);
        if bytes >= EVAL_BATCH_BYTES {
            eval(batch.iter())?;
            batch.clear();
            bytes = 0;
        }
    }
    if !batch.is_empty() {
        eval(batch.iter())?;
    }
    Ok(())
}

/// The LTX-2.5 text encoder: a shared Gemma 4 decoder plus LTX's video head and optional audio head.
pub struct Ltx25TextEncoder {
    model: CausalLm,
    video: FeatureHead,
    audio: Option<FeatureHead>,
    dtype: Dtype,
}

impl Ltx25TextEncoder {
    /// Build the **AudioVideo** encoder from a resolved split bundle.
    ///
    /// This is the production entry point, and the only one that exists, because it is the only one
    /// that can perform the version assertion: [`check_gemma_version`] compares the *transformer's*
    /// declared `gemma_source_checkpoint.gemma_version` against the *encoder's* own `gemma_version`,
    /// so it needs both halves. Offering a bundle-free constructor would be an escape hatch that
    /// silently disables the assertion, which is exactly the failure R4 forbids.
    ///
    /// `connector_w` carries the LTX heads. Note the tier converter (sc-18775) moves
    /// `text_embedding_projection.*` **out** of the packed encoder and into `connector.safetensors`,
    /// so the heads load from the same file and the same keys as LTX-2.3's — verified against the
    /// built tiers, not assumed.
    pub fn from_bundle_av(
        bundle: &LtxBundle,
        connector_w: &GenWeights,
        ltx_cfg: &LtxConfig,
        prec: Precision,
    ) -> Result<Self> {
        let te = bundle.require(LtxComponent::TextEncoder)?;
        let checkpoint = bundle
            .require(LtxComponent::Transformer)?
            .metadata()
            .clone();
        Self::from_packed_av(&checkpoint, te.path(), connector_w, ltx_cfg, prec)
    }

    /// As [`Self::from_bundle_av`] but with the checkpoint metadata and encoder path supplied
    /// directly — for the tier path, where `connector.safetensors` itself carries the
    /// `gemma_source_checkpoint` stamp, and for tests that build a synthetic pair.
    ///
    /// Still performs the full version assertion; there is no way to reach the decoder without it.
    pub fn from_packed_av(
        checkpoint: &LtxCheckpointMetadata,
        te_path: &Path,
        connector_w: &GenWeights,
        ltx_cfg: &LtxConfig,
        prec: Precision,
    ) -> Result<Self> {
        // (1) The caption feature extractor must be the V2 (PER_TOKEN_RMS) one this port implements.
        // Same genuinely config-driven gate LTX-2.3 runs, reading the loaded checkpoint's config.
        require_v2(ltx_cfg)?;

        // (2) R4: the encoder must be the one this checkpoint was trained against. A mismatch is a
        // hard error from `gen_core`, never a warning and never a fallback.
        assert_gemma_version_for(checkpoint, te_path)?;

        // (3) The Gemma 4 config travels inside the packed encoder's `__metadata__.gemma_config`.
        // `ModelConfig::from_json` rebinds to `text_config` for Gemma 4 (`nests_text_config`), which
        // is also where the tier's `quantization` block is stamped — a block written above it would
        // be invisible and the packed encoder would load as if dense.
        let identity_config = gemma_config_value(te_path)?;
        let cfg = ModelConfig::from_json(&identity_config).map_err(from_llm)?;

        // (4) Load through the same seam the shipped tiers were validated on (sc-18775 / PR #820):
        // `Weights::from_file` + `CausalLm::from_weights`, with a bounded materialization between
        // them so the cold file never becomes one command buffer.
        let weights = LlmWeights::from_file(te_path).map_err(from_llm)?;
        materialize_in_batches(&weights)?;
        let hidden_size = cfg.hidden_size;
        let model = CausalLm::from_weights(&weights, "", cfg).map_err(from_llm)?;

        let video = load_video_head(connector_w, hidden_size, ltx_cfg, prec)?;
        let audio = load_audio_head(connector_w, hidden_size, ltx_cfg, prec)?;
        Ok(Self {
            model,
            video,
            audio: Some(audio),
            dtype: prec.dtype(),
        })
    }

    /// The Gemma 4 backbone, exposed so the LTX-2.5 prompt enhancer (sc-18764) can drive the
    /// **already-loaded** encoder as an autoregressive LM instead of loading a second 26 GB copy.
    pub fn model(&self) -> &CausalLm {
        &self.model
    }

    /// The hidden-state stack, **evaluated in order**.
    ///
    /// See the module docs, hazard 1: the returned handles are one lazy graph, and forcing the tail
    /// alone submits the whole 48-layer stack as a single Metal command buffer. Walking the slice in
    /// order splits it into one buffer per layer. A consumer that streams the states — which is what
    /// a feature extractor does — gets this for free; this method makes it unconditional rather than
    /// depending on how the caller happens to touch the result.
    fn hidden_states_in_order(&self, input_ids: &Array) -> Result<Vec<Array>> {
        let mut cache = self.model.new_cache();
        let states = self
            .model
            .hidden_states(input_ids, &mut cache, 0)
            .map_err(from_llm)?;
        for state in &states {
            eval([state])?;
        }
        Ok(states)
    }

    /// Encode `(1, L)` token ids + `(1, L)` attention mask → `video_embeddings` `(1, L, 4096)`.
    pub fn encode(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        Ok(self.encode_with_features(input_ids, attention_mask)?.1)
    }

    /// Like [`Self::encode`] but also returns the pre-connector `video_features` — the
    /// **connector input** (post-projection, post-norm), which is the tensor the parity gates check.
    pub fn encode_with_features(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
    ) -> Result<(Array, Array)> {
        let hiddens = self.hidden_states_in_order(input_ids)?;
        let normed = per_token_rms_normed_hidden(&hiddens, attention_mask, self.dtype)?;
        self.video.forward(&normed, attention_mask)
    }

    /// AudioVideo encode: `(video_embeddings (1,L,4096), audio_embeddings (1,L,2048))`.
    pub fn encode_av(&self, input_ids: &Array, attention_mask: &Array) -> Result<(Array, Array)> {
        let (_, _, ve, ae) = self.encode_av_with_features(input_ids, attention_mask)?;
        Ok((ve, ae))
    }

    /// AudioVideo encode returning `(video_features, audio_features, video_embeddings,
    /// audio_embeddings)` — the two **connector inputs** included, mirroring LTX-2.3's
    /// `encode_av_with_features` so the cross-version and cross-backend gates compare like with like.
    pub fn encode_av_with_features(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
    ) -> Result<(Array, Array, Array, Array)> {
        let audio = self.audio.as_ref().ok_or_else(|| {
            Error::Msg("ltx_2_5: text encoder built without the audio head".into())
        })?;
        let hiddens = self.hidden_states_in_order(input_ids)?;
        let normed = per_token_rms_normed_hidden(&hiddens, attention_mask, self.dtype)?;
        let (vf, ve) = self.video.forward(&normed, attention_mask)?;
        let (af, ae) = audio.forward(&normed, attention_mask)?;
        Ok((vf, af, ve, ae))
    }
}

/// Read the packed encoder's `__metadata__.gemma_config` as JSON.
fn gemma_config_value(te_path: &Path) -> Result<serde_json::Value> {
    let assets = GemmaAssets::load(te_path)?;
    serde_json::from_str(assets.config_json()).map_err(|e| {
        Error::Msg(format!(
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
    let encoder = GemmaEncoderIdentity::from_single_file(te_path)?;
    match check_gemma_version(checkpoint, &encoder)? {
        GemmaVersionCheck::Matched(_) => Ok(()),
        GemmaVersionCheck::Gemma3Legacy => Err(Error::Msg(format!(
            "ltx_2_5: the text encoder at {} declares model_type gemma3 and was paired with a \
             pre-2.4 checkpoint, so the pair resolved down the LTX-2.3 route. That pair is valid \
             for LTX-2.3, but this is the 2.5 encoder path, which requires a Gemma 4 encoder and a \
             2.5 transformer — load it through LtxTextEncoder instead",
            te_path.display()
        ))),
        GemmaVersionCheck::SkippedNoDeclaredVersion => Err(Error::Msg(format!(
            "ltx_2_5: neither the checkpoint nor the Gemma config at {} declares a gemma_version, \
             so the encoder identity cannot be verified. A 2.5 text encoder must declare \
             gemma_version and be paired with a checkpoint that declares gemma_source_checkpoint",
            te_path.display()
        ))),
    }
}

#[cfg(test)]
mod version_assertion_tests {
    use super::*;
    use std::collections::BTreeMap;

    const GEMMA4_VERSION: &str = "gemma4-12b-ltx-v1";

    /// Write a safetensors file carrying only `__metadata__` — enough for the identity read, which
    /// gen-core documents as config-only and which deliberately does NOT require the full asset
    /// pack (so an incomplete pack reports a version mismatch rather than a missing tokenizer).
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
        // A DIFFERENT Gemma 4 build than the transformer was trained against.
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

    /// Pointing the 2.5 loader at the LTX-2.3 Gemma 3 snapshot is the case that actually reaches
    /// the `Gemma3Legacy` arm: gen-core answers `Gemma3Legacy` (correct — that pair IS valid 2.3),
    /// so only this layer can refuse it. Without the refusal the 2.5 encoder path would build a
    /// Gemma 4 decoder over Gemma 3 weights.
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

    /// A *Gemma 4* encoder against a pre-2.4 checkpoint is refused one layer down, by gen-core's
    /// `model_type` check, before `Gemma3Legacy` is ever returned. Pinned here so the split of
    /// responsibility is explicit and a future gen-core change that loosened it would be caught.
    #[test]
    fn a_gemma4_encoder_against_a_pre_2_4_checkpoint_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let te = write_te(dir.path(), "te.safetensors", &gemma4_config(GEMMA4_VERSION));
        let err = assert_gemma_version_for(&checkpoint("2.3.0", None), &te)
            .expect_err("a Gemma 4 encoder must not be accepted against a pre-2.4 checkpoint");
        assert!(
            err.to_string().contains("gemma3"),
            "gen-core's model_type refusal must be the one that fires: {err}"
        );
    }

    /// Neither side declares a version: `check_gemma_version` returns `SkippedNoDeclaredVersion`
    /// (upstream warns and continues). For a 2.5 encoder that is unverifiable identity, not a pass.
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

    /// The masking half of the shared V2 extractor: padded positions are zeroed, and every
    /// `(token, layer)` slice is RMS-normalized over the hidden dim independently.
    #[test]
    fn the_shared_extractor_zeroes_padding_and_normalizes_per_token_per_layer() {
        let (hidden, states, len) = (4usize, 3usize, 2usize);
        // state[s] is filled with (s + 1), so a per-slice normalization is distinguishable from a
        // whole-tensor one: each layer's magnitude differs but each must normalize to ~1.
        let hiddens: Vec<Array> = (0..states)
            .map(|s| {
                Array::from_slice(
                    &vec![(s + 1) as f32; len * hidden],
                    &[1, len as i32, hidden as i32],
                )
            })
            .collect();
        // Token 0 is padding, token 1 is valid.
        let mask = Array::from_slice(&[0.0f32, 1.0], &[1, len as i32]);

        let normed =
            per_token_rms_normed_hidden(&hiddens, &mask, Dtype::Float32).expect("extractor");
        assert_eq!(normed.shape(), &[1, len as i32, (hidden * states) as i32]);
        eval([&normed]).expect("eval");
        let flat = normed.as_slice::<f32>();

        let width = hidden * states;
        assert!(
            flat[..width].iter().all(|v| *v == 0.0),
            "padded positions must be zeroed, got {:?}",
            &flat[..width]
        );
        assert!(
            flat[width..].iter().all(|v| (*v - 1.0).abs() < 1e-3),
            "each per-token-RMS slice must normalize to ~1, got {:?}",
            &flat[width..]
        );
    }

    /// The typed error variants gen-core's contract matches on must cross the mlx-llm seam as
    /// themselves. Laundering `Canceled` into `Msg` would break the worker's cancellation contract.
    #[test]
    fn typed_llm_errors_cross_the_seam_without_being_stringified() {
        assert!(matches!(
            from_llm(mlx_llm::Error::Canceled),
            Error::Canceled
        ));
        assert!(matches!(
            from_llm(mlx_llm::Error::Unsupported("nope".into())),
            Error::Unsupported(_)
        ));
        assert!(matches!(
            from_llm(mlx_llm::Error::MissingTensor("k".into())),
            Error::MissingTensor(_)
        ));
    }
}
