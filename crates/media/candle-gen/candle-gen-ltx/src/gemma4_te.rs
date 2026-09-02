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
//!
//! # The padding mask
//!
//! The tokenizer *left-pads* to `max_length`, and `CausalLm`'s default masking is causal only — no
//! padding component — so every valid token would attend the pad run and all 49 hidden states,
//! hence the video and audio features, would be wrong. Wrong but finite, non-zero, correctly shaped
//! and still prompt-separated, which is why only a numeric oracle or an explicit pad-invariance
//! test catches it. `causal_padding_mask` builds LTX-2.3's `valid(i, j) = j <= i && mask01[j] != 0`
//! rule as an additive mask and `masked_hidden_states` threads it through
//! `CausalLm::hidden_states_with_mask`.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::gemma_assets::GemmaAssets;
use candle_gen::gen_core::ltx_checkpoint::{
    check_gemma_version, GemmaEncoderIdentity, GemmaVersionCheck, LtxCheckpointMetadata,
    LtxComponent,
};
use candle_gen::gen_core::LtxBundle;

use candle_llm::config::ModelConfig;
use candle_llm::models::CausalLm;
use candle_llm::primitives::{AttnMask, Weights as LlmWeights};

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

/// Additive-mask fill for a blocked `(query, key)` pair.
///
/// The same large finite negative `candle-llm`'s own masks use, deliberately rather than LTX-2.3's
/// bf16-min fill: a Gemma 4 `sliding_attention` layer **sums** this mask with its window band
/// (`LlamaLayer::combined_sliding_mask`), and a bf16-min fill plus a band is one rounding step from
/// `inf`. After the softmax the two are indistinguishable — `exp(-1e30)` is already exactly 0 — so
/// this changes no number, only the composition headroom.
const MASK_NEG: f32 = -1e30;

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
    /// Build the **AudioVideo** encoder from a resolved split bundle — the candle twin of
    /// `mlx-gen-ltx`'s `Ltx25TextEncoder::from_bundle_av`, and the production entry point.
    ///
    /// The bundle is what makes the version assertion possible at all: [`check_gemma_version`]
    /// compares the *transformer's* declared `gemma_source_checkpoint.gemma_version` against the
    /// *encoder's* own `gemma_version`, so it needs both halves. This resolves them from one
    /// object rather than making every caller pair them up by hand (and get it wrong once).
    #[allow(clippy::too_many_arguments)]
    pub fn from_bundle_av(
        bundle: &LtxBundle,
        proj_vb: VarBuilder,
        dit_vb: VarBuilder,
        av_cfg: &AvConfig,
        conn_cfg: &ConnectorConfig,
        audio_conn_cfg: &ConnectorConfig,
    ) -> Result<Self> {
        let te = bundle
            .require(LtxComponent::TextEncoder)
            .map_err(from_gen_core)?;
        let checkpoint = bundle
            .require(LtxComponent::Transformer)
            .map_err(from_gen_core)?
            .metadata()
            .clone();
        Self::from_packed_av(
            &checkpoint,
            te.path(),
            proj_vb,
            dit_vb,
            av_cfg,
            conn_cfg,
            audio_conn_cfg,
        )
    }

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
        let mut me = Self::from_packed_video(
            checkpoint,
            te_path,
            proj_vb.clone(),
            dit_vb.clone(),
            av_cfg,
            conn_cfg,
        )?;

        let (audio_aggregate, audio_rescale) = load_aggregate(
            &proj_vb,
            "text_embedding_projection.audio_aggregate_embed",
            audio_conn_cfg.inner_dim(),
            me.hidden_size,
        )?;
        let audio_connector =
            Connector::new_with_prefix(dit_vb, audio_conn_cfg, "audio_embeddings_connector")?;
        me.audio = Some(AudioHead {
            aggregate: audio_aggregate,
            rescale: audio_rescale,
            connector: audio_connector,
        });
        Ok(me)
    }

    /// The **video-only** encoder, mirroring LTX-2.3's [`new`] vs [`new_av`] split
    /// (`crate::text_encoder::LtxTextEncoder`) and `mlx-gen-ltx`'s `from_packed_video`.
    ///
    /// A checkpoint whose transformer carries no `audio_embeddings_connector` / no
    /// `audio_aggregate_embed` — a video-only 2.5 build — has no audio head to load, and
    /// [`Self::from_packed_av`] would fail on the missing tensor rather than produce a usable
    /// encoder. [`Self::encode`] / [`Self::encode_with_features`] work exactly as on the AV
    /// encoder; [`Self::encode_both`] / [`Self::encode_both_with_features`] name this constructor
    /// in their error.
    ///
    /// [`new`]: crate::text_encoder::LtxTextEncoder::new
    /// [`new_av`]: crate::text_encoder::LtxTextEncoder::new_av
    pub fn from_packed_video(
        checkpoint: &LtxCheckpointMetadata,
        te_path: &Path,
        proj_vb: VarBuilder,
        dit_vb: VarBuilder,
        av_cfg: &AvConfig,
        conn_cfg: &ConnectorConfig,
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

        // (4) Load through the same seam the shipped tiers were validated on (PR #820). A quantized
        // source is staged on the host: uploading the complete MLX-affine encoder first would retain
        // it while the decoder accumulates its resident QTensor form, producing a near-2x undeclared
        // device-memory peak. The dense tier keeps its direct-to-device load path unchanged.
        let stored_quant = cfg.quantization;
        let model = if stored_quant.is_some() {
            let weights = LlmWeights::from_file(te_path, &Device::Cpu).map_err(from_llm)?;
            CausalLm::from_weights_on_device(&weights, "", cfg, stored_quant, &device)
        } else {
            let weights = LlmWeights::from_file(te_path, &device).map_err(from_llm)?;
            CausalLm::from_weights(&weights, "", cfg)
        }
        .map_err(from_llm)?;

        let (aggregate, rescale) = load_aggregate(
            &proj_vb,
            "text_embedding_projection.video_aggregate_embed",
            conn_cfg.inner_dim(),
            hidden_size,
        )?;
        let connector = Connector::new(dit_vb, conn_cfg)?;

        Ok(Self {
            model,
            aggregate,
            rescale,
            connector,
            audio: None,
            hidden_size,
            device,
        })
    }

    /// The Gemma 4 backbone, exposed so the LTX-2.5 prompt enhancer (sc-18764) can drive the
    /// **already-loaded** encoder as an autoregressive LM instead of loading a second copy.
    pub fn model(&self) -> &CausalLm {
        &self.model
    }

    /// The hidden-state stack under the caller's padding mask.
    ///
    /// candle is eager, so unlike the MLX twin there is no lazy-graph hazard to defend against here
    /// — the states are materialized as they are produced. The method exists so both backends read
    /// the same way at the call sites below, and so the cross-backend gate has one named seam.
    fn hidden_states(&self, input_ids: &Tensor, mask01: &[u32]) -> Result<Vec<Tensor>> {
        masked_hidden_states(&self.model, input_ids, mask01, &self.device)
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
        let hiddens = self.hidden_states(input_ids, mask01)?;
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
                "ltx_2_5: text encoder built without the audio head — it was constructed with \
                 from_packed_video, which is the video-only entry point; use from_bundle_av / \
                 from_packed_av for the AudioVideo path"
                    .into(),
            )
        })?;
        let hiddens = self.hidden_states(input_ids, mask01)?;
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

/// The additive **causal + left-padding** attention mask `[1, 1, L, L]`, in `dtype`.
///
/// `valid(i, j) = j <= i && mask01[j] != 0` keeps (`0`); everything else blocks ([`MASK_NEG`]) —
/// byte-for-byte the rule LTX-2.3 applies on both backends (`crate::gemma`'s
/// `causal_padding_mask`, and its MLX twin).
///
/// See [`masked_hidden_states`] for why a causal-only mask is not enough.
fn causal_padding_mask(mask01: &[u32], dtype: DType, device: &Device) -> Result<Tensor> {
    let l = mask01.len();
    let mut data = vec![0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            let valid = j <= i && mask01[j] != 0;
            data[i * l + j] = if valid { 0.0 } else { MASK_NEG };
        }
    }
    Tensor::from_vec(data, (1, 1, l, l), device)?.to_dtype(dtype)
}

/// The `model`'s hidden-state stack for `input_ids` under `mask01`.
///
/// The tokenizer left-pads to `max_length`, so a purely causal mask lets every valid token attend
/// the whole pad run and all 49 hidden states — hence the video and audio features — come out
/// wrong. They also stay finite, non-zero, correctly shaped and prompt-separated, so nothing but a
/// numeric oracle or a pad-invariance test notices. This threads the same additive causal+padding
/// mask LTX-2.3 uses.
///
/// Free-standing rather than a method so the weights-free regression can drive it over a tiny
/// synthetic [`CausalLm`], with no packed encoder and no feature heads.
fn masked_hidden_states(
    model: &CausalLm,
    input_ids: &Tensor,
    mask01: &[u32],
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mask = causal_padding_mask(mask01, model.compute_dtype(), device)?;
    let mut cache = model.new_cache();
    model
        .hidden_states_with_mask(input_ids, &mut cache, 0, AttnMask::Additive(&mask))
        .map_err(from_llm)
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

/// The padding-mask regression (sc-18770 review issue 1), weights-free — the candle twin of
/// `mlx-gen-ltx`'s `padding_mask_tests`, asserting the same property in the same terms.
///
/// The bug this exists to catch: `CausalLm::hidden_states` masks causally and nothing else, so
/// under the tokenizer's *left* padding every valid token attends the pad run. The resulting
/// hidden states — and every feature built from them — are wrong while remaining finite, non-zero,
/// correctly shaped and prompt-separated, i.e. invisible to every other assertion in this crate.
#[cfg(test)]
mod padding_mask_tests {
    use super::*;
    use candle_llm::primitives::Weights as LlmWeightsMap;
    use std::collections::HashMap;

    const HIDDEN: usize = 32;
    const VOCAB: usize = 48;
    const HEAD_DIM: usize = 8;
    const LAYERS: usize = 3;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const INTER: usize = 64;

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("host copy")
    }

    /// A tiny **real** `CausalLm` over synthetic weights — no packed encoder, no download, and the
    /// same seam `from_packed_av` builds through, so the mask genuinely flows into attention.
    fn tiny_model(device: &Device) -> CausalLm {
        let cfg_json = serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
            "vocab_size": VOCAB, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "tie_word_embeddings": true
        });
        let cfg = ModelConfig::from_json(&cfg_json).expect("tiny config");

        // Deterministic, dependency-free PRNG: the values only need to be distinct and O(0.1).
        let mut state = 0x5C1_8770_u64;
        let mut randn = |rows: usize, cols: usize| -> Tensor {
            let data: Vec<f32> = (0..rows * cols)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.4
                })
                .collect();
            Tensor::from_vec(data, (rows, cols), device).expect("randn")
        };

        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("model.embed_tokens.weight".into(), randn(VOCAB, HIDDEN));
        m.insert(
            "model.norm.weight".into(),
            Tensor::ones(HIDDEN, DType::F32, device).expect("norm"),
        );
        for i in 0..LAYERS {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            m.insert(
                p("self_attn.q_proj.weight"),
                randn(HEADS * HEAD_DIM, HIDDEN),
            );
            m.insert(
                p("self_attn.k_proj.weight"),
                randn(KV_HEADS * HEAD_DIM, HIDDEN),
            );
            m.insert(
                p("self_attn.v_proj.weight"),
                randn(KV_HEADS * HEAD_DIM, HIDDEN),
            );
            m.insert(
                p("self_attn.o_proj.weight"),
                randn(HIDDEN, HEADS * HEAD_DIM),
            );
            m.insert(p("mlp.gate_proj.weight"), randn(INTER, HIDDEN));
            m.insert(p("mlp.up_proj.weight"), randn(INTER, HIDDEN));
            m.insert(p("mlp.down_proj.weight"), randn(HIDDEN, INTER));
            m.insert(
                p("input_layernorm.weight"),
                Tensor::ones(HIDDEN, DType::F32, device).expect("in ln"),
            );
            m.insert(
                p("post_attention_layernorm.weight"),
                Tensor::ones(HIDDEN, DType::F32, device).expect("post ln"),
            );
        }
        CausalLm::from_weights(&LlmWeightsMap::from_map(m, device.clone()), "", cfg)
            .expect("tiny CausalLm")
    }

    fn ids(v: &[u32], device: &Device) -> Tensor {
        Tensor::from_vec(v.to_vec(), (1, v.len()), device).expect("ids")
    }

    /// The mask rule itself: `valid(i, j) = j <= i && mask01[j] != 0`, everything else blocked.
    #[test]
    fn the_mask_blocks_pad_columns_and_the_causal_upper_triangle() {
        let device = Device::Cpu;
        let m = causal_padding_mask(&[0, 0, 1, 1], DType::F32, &device).expect("mask");
        assert_eq!(m.dims4().expect("rank 4"), (1, 1, 4, 4));
        let v = host(&m);
        for i in 0..4usize {
            for j in 0..4usize {
                let want_open = j <= i && j >= 2;
                let got = v[i * 4 + j];
                if want_open {
                    assert_eq!(got, 0.0, "({i},{j}) must be attendable");
                } else {
                    assert!(got < -1e29, "({i},{j}) must be blocked, got {got}");
                }
            }
        }
        // The two pad *columns* are blocked for EVERY query row, including the pad rows themselves —
        // this is the component `AttnMask::Causal` does not have, and the whole point of the fix.
        for i in 0..4usize {
            assert!(v[i * 4] < -1e29 && v[i * 4 + 1] < -1e29, "pad columns open");
        }
    }

    /// **The regression.** Two sequences that differ *only* in their left-pad token ids must
    /// produce bit-identical hidden states — and therefore bit-identical extractor output — at
    /// every valid position.
    ///
    /// Replace `AttnMask::Additive(&mask)` in [`masked_hidden_states`] with `AttnMask::Causal` and
    /// this fails: the valid tokens then attend the pad run, whose keys and values differ between
    /// the two variants.
    #[test]
    fn left_pad_ids_do_not_reach_the_valid_positions() {
        let device = Device::Cpu;
        let model = tiny_model(&device);
        const PADS: usize = 3;
        const LEN: usize = 8;

        // Identical valid tail, different pad ids. `mask01` is the same for both.
        let a = ids(&[0, 0, 0, 3, 9, 14, 2, 7], &device);
        let b = ids(&[41, 17, 5, 3, 9, 14, 2, 7], &device);
        let mask01 = [0u32, 0, 0, 1, 1, 1, 1, 1];

        let ha = masked_hidden_states(&model, &a, &mask01, &device).expect("stack a");
        let hb = masked_hidden_states(&model, &b, &mask01, &device).expect("stack b");
        assert_eq!(ha.len(), LAYERS + 1);

        // Sanity: the pad rows themselves DO differ, or the two inputs would be indistinguishable
        // and the assertion below would hold for a trivial reason.
        let pad_delta = host(&ha[LAYERS])
            .iter()
            .zip(host(&hb[LAYERS]).iter())
            .take(PADS * HIDDEN)
            .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()));
        assert!(
            pad_delta > 0.0,
            "the two inputs must actually differ at the pad positions"
        );

        for (i, (sa, sb)) in ha.iter().zip(hb.iter()).enumerate() {
            let (va, vb) = (host(sa), host(sb));
            for t in PADS..LEN {
                let (lo, hi) = (t * HIDDEN, (t + 1) * HIDDEN);
                assert_eq!(
                    &va[lo..hi],
                    &vb[lo..hi],
                    "hidden state {i}, valid position {t} must not depend on the pad ids — the \
                     padding component of the attention mask was dropped"
                );
            }
        }

        // ...and the same through the extractor, which is what the feature heads consume. The
        // extractor zeroes padding in bf16 (the real compute dtype on GPU; this tiny model runs on
        // CPU, where `compute_dtype` is f32), so the stacks are cast first — bit-identical inputs
        // stay bit-identical through the cast.
        let bf16 = |v: &[Tensor]| -> Vec<Tensor> {
            v.iter()
                .map(|t| t.to_dtype(DType::BF16).expect("bf16"))
                .collect()
        };
        let na =
            per_token_rms_normed_hidden(&bf16(&ha), &mask01, HIDDEN, &device).expect("normed a");
        let nb =
            per_token_rms_normed_hidden(&bf16(&hb), &mask01, HIDDEN, &device).expect("normed b");
        let width = HIDDEN * (LAYERS + 1);
        let (va, vb) = (host(&na), host(&nb));
        for t in PADS..LEN {
            assert_eq!(
                &va[t * width..(t + 1) * width],
                &vb[t * width..(t + 1) * width],
                "extractor output at valid position {t} must not depend on the pad ids"
            );
        }
    }

    /// The mask is not a no-op dressed up as one: an encoder that attends the pad run produces
    /// *different* valid-position states from one that does not. Without this, the test above would
    /// also pass against a model whose attention ignored the mask entirely.
    #[test]
    fn attending_the_pad_run_actually_changes_the_valid_positions() {
        let device = Device::Cpu;
        let model = tiny_model(&device);
        let a = ids(&[0, 0, 0, 3, 9, 14, 2, 7], &device);
        let mask01 = [0u32, 0, 0, 1, 1, 1, 1, 1];

        let masked = masked_hidden_states(&model, &a, &mask01, &device).expect("masked");
        let mut cache = model.new_cache();
        let causal_only = model
            .hidden_states(&a, &mut cache, 0)
            .expect("causal-only stack");

        let delta = host(&masked[LAYERS])
            .iter()
            .zip(host(&causal_only[LAYERS]).iter())
            .skip(3 * HIDDEN)
            .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()));
        assert!(
            delta > 1e-4,
            "causal-only masking must measurably differ from causal+padding at the valid \
             positions (got {delta}) — otherwise the regression above proves nothing"
        );
    }
}

#[cfg(test)]
mod version_assertion_tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

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

    /// A `__metadata__`-only transformer file: the `gemma_source_checkpoint` stamp the assertion
    /// reads, plus a `config.transformer` section so the bundle builder classifies it correctly.
    /// Byte-for-byte the fixture `mlx-gen-ltx`'s twin writes.
    fn write_transformer(dir: &Path, model_version: &str, gemma_version: &str) -> PathBuf {
        let path = dir.join("transformer.safetensors");
        let header = serde_json::json!({
            "__metadata__": {
                "model_version": model_version,
                "config": serde_json::json!({ "transformer": { "num_layers": 48 } }).to_string(),
                "gemma_source_checkpoint": serde_json::json!({
                    "ltx_version": model_version,
                    "gemma_version": gemma_version,
                })
                .to_string(),
            }
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(&path, bytes).expect("write transformer");
        path
    }

    /// `from_bundle_av` — the production entry point, and the constructor the MLX twin has had
    /// since sc-18770 — must actually reach the version assertion through the bundle: resolve both
    /// components, take the *transformer's* metadata and the *text encoder's* path, and hand them
    /// to the gate.
    ///
    /// Weights-free: both files are `__metadata__`-only safetensors and the mismatch fires before a
    /// single tensor is read, so no `VarBuilder` is ever dereferenced.
    #[test]
    fn from_bundle_av_reaches_the_version_assertion_through_the_bundle() {
        use candle_gen::candle_nn::VarBuilder;
        use candle_gen::gen_core::ltx_checkpoint::{CaptionFeatureVersion, LtxBundleBuilder};

        let dir = tempfile::tempdir().expect("tempdir");
        // The encoder declares a DIFFERENT Gemma 4 build than the transformer was trained against.
        let te = write_te(
            dir.path(),
            "text_encoder.safetensors",
            &gemma4_config("gemma4-12b-ltx-v2"),
        );
        let transformer = write_transformer(dir.path(), "2.5.0", GEMMA4_VERSION);

        let bundle = LtxBundleBuilder::new()
            .with_component(LtxComponent::TextEncoder, &te)
            .with_component(LtxComponent::Transformer, &transformer)
            .build()
            .expect("synthetic 2.5 bundle");

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mut av_cfg = AvConfig::ltx_2_3();
        av_cfg.caption_feature_version = CaptionFeatureVersion::V2;

        let msg = match Ltx25TextEncoder::from_bundle_av(
            &bundle,
            vb.clone(),
            vb,
            &av_cfg,
            &ConnectorConfig::ltx_2_3(),
            &ConnectorConfig::ltx_2_3_audio(),
        ) {
            Ok(_) => panic!(
                "a gemma_version mismatch must be a hard error on the bundle path too, but \
                 from_bundle_av returned an encoder"
            ),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("gemma4-12b-ltx-v1") && msg.contains("gemma4-12b-ltx-v2"),
            "the bundle path must reach the same version assertion, naming both sides: {msg}"
        );
        assert!(
            msg.contains("text_encoder.safetensors"),
            "the assertion must be run against the ENCODER's path, not the transformer's: {msg}"
        );
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
