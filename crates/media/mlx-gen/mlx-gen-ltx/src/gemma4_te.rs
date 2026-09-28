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
//! # Three hazards this module exists to get right
//!
//! 1. **MLX laziness on the hidden-state stack.** `hidden_states` returns 49 handles onto ONE
//!    unevaluated graph. Forcing the last one submits every weight page-in and every layer's
//!    matmuls as a single Metal command buffer, which the real 26.3 GB encoder exceeds — it comes
//!    back as `kIOGPUCommandBufferCallbackErrorSubmissionsIgnored`, and the poisoned process fails
//!    every *later* submission too. `Ltx25TextEncoder::hidden_states_in_order` walks the stack in
//!    order so each layer is its own command buffer. See the `hidden_states_from_embeds` doc in
//!    `mlx-llm` for the measurement.
//! 2. **Cold mmapped weights.** `Weights::from_file` is lazy; the first graph over a cold tier drags
//!    the whole file into one buffer and trips the watchdog. Under
//!    [`OffloadPolicy::Resident`] [`materialize_in_batches`] forces the tensors resident in bounded
//!    submissions before anything builds a graph over them. Under [`OffloadPolicy::Sequential`] it
//!    is deliberately **not** called — see `load_backbone`.
//! 3. **The padding mask.** The tokenizer *left-pads* to `max_length`, and `CausalLm`'s default
//!    masking is causal only — no padding component — so every valid token would attend the pad
//!    run and all 49 hidden states, hence the video and audio features, would be wrong. Wrong but
//!    finite, non-zero, correctly shaped and still prompt-separated, which is why only a numeric
//!    oracle or an explicit pad-invariance test catches it. `causal_padding_mask` builds LTX-2.3's
//!    `valid(i, j) = j <= i && mask01[j] != 0` rule as an additive mask and
//!    `masked_hidden_states_in_order` threads it through `CausalLm::hidden_states_with_mask`.
//!
//! # Residency, and why the text phase is the one worth bounding (sc-18798)
//!
//! On LTX the **text phase binds the peak**, and 2.5 makes it worse: measured on 2.3 q4, the bf16
//! TE is ≈24.6 GiB against a q4 DiT's ≈10.6 GiB, and 2.5's encoder is 26.3 GB. Bounding the DiT
//! harder cannot move a TE-bound peak, so the levers that matter are on this side.
//!
//! There are **two orthogonal ones, and conflating them is the mistake**:
//!
//! * *Component staging* — TE and AvDiT never co-resident. LTX already does this unconditionally
//!   (epic 10975 / sc-10976), it is not a selectable control, and this module does not change it.
//!   The descriptor still declares `supports_sequential_offload: false` with
//!   `unconditionally_engages_staged_residency: true`, and that stays true.
//! * *Intra-encoder residency* — whether the 48 decoder layers are all resident **while the text
//!   phase runs**. That is what [`OffloadPolicy::Sequential`] selects here, through
//!   `mlx_llm::residency`. Staging bounds the peak to `max(text, everything_else)`; when the text
//!   phase *is* the maximum, only this lever touches it. Z-Image measured the same split: streaming
//!   the encoder at all took conditioning 8.489 → 2.718 GiB, while the tunable component scope
//!   moved the request peak 0.0 %.
//!
//! ## Per-tier TE quantization, and why `q4` is where this pays most
//!
//! A tier's text encoder is **not** always packed at the tier's width.
//! [`crate::tiers::TEXT_ENCODER_Q4_QUALITY`] carries the decision and sc-18775's measured numbers:
//! over the 49 hidden states this extractor concatenates, q4 lands at worst cos 0.889414 /
//! rel-L2 0.53488 against a cos > 0.97 / rel-L2 < 0.30 bar and **fails**; q8 lands at 0.999086 /
//! 0.04320 against cos > 0.995 / rel-L2 < 0.10 and **passes**. Judging on the final layer alone
//! would have wrongly passed q4 (0.99995) — worst-case over all 49 is the whole point, because the
//! extractor per-token-RMS normalizes the concatenated stack and scales a low-norm middle layer's
//! error straight back into the conditioning the DiT sees.
//!
//! So `q4` ships the encoder **dense**, declared as
//! [`DenseReason::BelowQualityBar`](crate::tiers::DenseReason::BelowQualityBar) — a measured,
//! declared exception to the whole-pipeline tier contract (R5) rather than an unexplained dense
//! component. The consequence *for residency*, which is what this module has to get right:
//!
//! **the `q4` tier's text phase carries a bf16 encoder, so `q4` is the tier where the text phase
//! binds hardest and where streaming buys the most** — the opposite of the intuition that the
//! smallest tier has the smallest text phase. It is also why a `q4` tier is only ~2 % smaller than
//! a `q8` one rather than ~40 %: the encoder it ships is the larger of the two. Exact on-disk tier
//! sizes are sc-18781's manifest footprints, measured there rather than restated here.

use std::path::Path;

use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::gemma_assets::GemmaAssets;
use mlx_gen::gen_core::ltx_checkpoint::{
    check_gemma_version, GemmaEncoderIdentity, GemmaVersionCheck, LtxCheckpointMetadata,
    LtxComponent,
};
use mlx_gen::gen_core::LtxBundle;
use mlx_gen::gen_core::OffloadPolicy;
use mlx_gen::weights::Weights as GenWeights;
use mlx_gen::{Error, Result};

use mlx_llm::config::ModelConfig;
use mlx_llm::primitives::{AttnMask, Weights as LlmWeights};
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

/// Additive-mask fill for a blocked `(query, key)` pair.
///
/// The same large finite negative `mlx-llm`'s own masks use, deliberately rather than LTX-2.3's
/// bf16-min fill: a Gemma 4 `sliding_attention` layer **sums** this mask with its window band
/// (`LlamaLayer::windowed`), and a bf16-min fill plus a band is one rounding step from `inf`. After
/// the softmax the two are indistinguishable — `exp(-1e30)` is already exactly 0 — so this changes
/// no number, only the composition headroom.
const MASK_NEG: f32 = -1e30;

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
        mlx_llm::Error::IncoherentLoad {
            name,
            bytes,
            cpu,
            gpu,
            attempts,
        } => Error::IncoherentLoad {
            name,
            bytes,
            cpu,
            gpu,
            attempts,
        },
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
        policy: OffloadPolicy,
    ) -> Result<Self> {
        let te = bundle.require(LtxComponent::TextEncoder)?;
        let checkpoint = bundle
            .require(LtxComponent::Transformer)?
            .metadata()
            .clone();
        Self::from_packed_av(&checkpoint, te.path(), connector_w, ltx_cfg, prec, policy)
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
        policy: OffloadPolicy,
    ) -> Result<Self> {
        let (model, hidden_size) = load_backbone(checkpoint, te_path, ltx_cfg, policy)?;
        let video = load_video_head(connector_w, hidden_size, ltx_cfg, prec)?;
        let audio = load_audio_head(connector_w, hidden_size, ltx_cfg, prec)?;
        Ok(Self {
            model,
            video,
            audio: Some(audio),
            dtype: prec.dtype(),
        })
    }

    /// The **video-only** encoder, mirroring LTX-2.3's [`from_weights`] vs [`from_weights_av`] split
    /// (`crate::text_encoder::LtxTextEncoder`).
    ///
    /// A checkpoint whose transformer carries no `audio_embeddings_connector` / no
    /// `audio_aggregate_embed` — a video-only 2.5 build — has no audio head to load, and
    /// [`Self::from_packed_av`] would fail on the missing tensor rather than produce a usable
    /// encoder. [`Self::encode`] / [`Self::encode_with_features`] work exactly as on the AV
    /// encoder; [`Self::encode_av`] / [`Self::encode_av_with_features`] name this constructor in
    /// their error.
    ///
    /// [`from_weights`]: crate::text_encoder::LtxTextEncoder::from_weights
    /// [`from_weights_av`]: crate::text_encoder::LtxTextEncoder::from_weights_av
    pub fn from_packed_video(
        checkpoint: &LtxCheckpointMetadata,
        te_path: &Path,
        connector_w: &GenWeights,
        ltx_cfg: &LtxConfig,
        prec: Precision,
        policy: OffloadPolicy,
    ) -> Result<Self> {
        let (model, hidden_size) = load_backbone(checkpoint, te_path, ltx_cfg, policy)?;
        let video = load_video_head(connector_w, hidden_size, ltx_cfg, prec)?;
        Ok(Self {
            model,
            video,
            audio: None,
            dtype: prec.dtype(),
        })
    }

    /// The Gemma 4 backbone, exposed so the LTX-2.5 prompt enhancer (sc-18764) can drive the
    /// **already-loaded** encoder as an autoregressive LM instead of loading a second 26 GB copy.
    pub fn model(&self) -> &CausalLm {
        &self.model
    }

    /// The hidden-state stack under the caller's padding mask, **evaluated in order**.
    ///
    /// See the module docs, hazard 1 (evaluation order) and hazard 3 (the padding mask).
    fn hidden_states_in_order(&self, input_ids: &Array, mask01: &Array) -> Result<Vec<Array>> {
        masked_hidden_states_in_order(&self.model, input_ids, mask01)
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
        let hiddens = self.hidden_states_in_order(input_ids, attention_mask)?;
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
            Error::Msg(
                "ltx_2_5: text encoder built without the audio head — it was constructed with \
                 from_packed_video, which is the video-only entry point; use from_bundle_av / \
                 from_packed_av for the AudioVideo path"
                    .into(),
            )
        })?;
        let hiddens = self.hidden_states_in_order(input_ids, attention_mask)?;
        let normed = per_token_rms_normed_hidden(&hiddens, attention_mask, self.dtype)?;
        let (vf, ve) = self.video.forward(&normed, attention_mask)?;
        let (af, ae) = audio.forward(&normed, attention_mask)?;
        Ok((vf, af, ve, ae))
    }
}

/// The gated, materialized Gemma 4 backbone plus its hidden size — everything both constructors do
/// before they diverge on which feature heads to load.
///
/// Kept as one function so neither entry point can skip a gate: a video-only constructor that
/// forgot [`require_v2`] or [`assert_gemma_version_for`] would be exactly the escape hatch R4
/// forbids.
fn load_backbone(
    checkpoint: &LtxCheckpointMetadata,
    te_path: &Path,
    ltx_cfg: &LtxConfig,
    policy: OffloadPolicy,
) -> Result<(CausalLm, i32)> {
    // (1) The caption feature extractor must be the V2 (PER_TOKEN_RMS) one this port implements.
    // Same genuinely config-driven gate LTX-2.3 runs, reading the loaded checkpoint's config.
    require_v2(ltx_cfg)?;

    // (2) R4: the encoder must be the one this checkpoint was trained against. A mismatch is a
    // hard error from `gen_core`, never a warning and never a fallback.
    assert_gemma_version_for(checkpoint, te_path)?;

    // (3) The Gemma 4 config travels inside the packed encoder's `__metadata__.gemma_config`.
    // `ModelConfig::from_json` rebinds to `text_config` for Gemma 4 (`nests_text_config`), which
    // is also where the tier's `quantization` block is stamped — a block written above it would
    // be invisible and the packed encoder would load as if dense. The stream reads that same
    // `cfg.quantization` per materialized layer, so a packed tier streams packed.
    let identity_config = gemma_config_value(te_path)?;
    let cfg = ModelConfig::from_json(&identity_config).map_err(from_llm)?;
    let hidden_size = cfg.hidden_size;

    // (4) Residency. `Sequential` (sc-18798) materializes one decoder layer at a time from
    // `te_path` and drops it, bounding the text phase's weight peak to a single layer instead of
    // all 48.
    //
    // Note what is deliberately NOT called on that branch: `materialize_in_batches` forces the
    // *whole* file resident, which is precisely what the stream exists to avoid — running both
    // would leave the memory exactly where it started and hide it behind a stream that looks
    // engaged. The cold-file hazard it exists for (hazard 2 in the module docs) is handled
    // differently under the stream, and better: the per-layer `eval` inside
    // `mlx_llm::residency::SequentialStack::run_layer` already bounds every submission to one
    // layer's weights, which is a tighter bound than the 512 MB batching.
    let model = match policy {
        OffloadPolicy::Sequential => {
            CausalLm::from_file_sequential(te_path, "", cfg, None).map_err(from_llm)?
        }
        // The seam the shipped tiers were validated on (sc-18775 / PR #820): `Weights::from_file`
        // + `CausalLm::from_weights`, with a bounded materialization between them so the cold file
        // never becomes one command buffer.
        OffloadPolicy::Resident => {
            let weights = LlmWeights::from_file(te_path).map_err(from_llm)?;
            materialize_in_batches(&weights)?;
            CausalLm::from_weights(&weights, "", cfg).map_err(from_llm)?
        }
    };
    Ok((model, hidden_size))
}

/// The additive **causal + left-padding** attention mask `[1, 1, L, L]`, in `dtype`.
///
/// `valid(i, j) = j <= i && mask01[j] != 0` keeps (`0`); everything else blocks ([`MASK_NEG`]) —
/// byte-for-byte the rule LTX-2.3 applies on both backends (`crate::gemma`'s
/// `causal_padding_mask`, and its candle twin).
///
/// See [`masked_hidden_states_in_order`] for why a causal-only mask is not enough.
fn causal_padding_mask(mask01: &Array, dtype: Dtype) -> Result<Array> {
    let sh = mask01.shape();
    if sh.len() != 2 || sh[0] != 1 {
        return Err(Error::Msg(format!(
            "ltx_2_5: the attention mask must be (1, L), got {sh:?}"
        )));
    }
    let l = sh[1];
    let host = mask01.as_dtype(Dtype::Int32)?;
    eval([&host])?;
    let m = host.as_slice::<i32>();
    let mut data = vec![0f32; (l * l) as usize];
    for i in 0..l as usize {
        for j in 0..l as usize {
            let valid = j <= i && m[j] != 0;
            data[i * l as usize + j] = if valid { 0.0 } else { MASK_NEG };
        }
    }
    Array::from_slice(&data, &[1, 1, l, l])
        .as_dtype(dtype)
        .map_err(Error::from)
}

/// The `model`'s hidden-state stack for `input_ids` under `mask01`, **evaluated in order**.
///
/// Two things this gets right that the bare `CausalLm::hidden_states` does not:
///
/// 1. **Padding.** The tokenizer left-pads to `max_length`, so a purely causal mask lets every
///    valid token attend the whole pad run and all 49 hidden states — hence the video and audio
///    features — come out wrong. They also stay finite, non-zero, correctly shaped and
///    prompt-separated, so nothing but a numeric oracle or a pad-invariance test notices. This
///    threads the same additive causal+padding mask LTX-2.3 uses.
/// 2. **Evaluation order.** The returned handles are one lazy graph; forcing the tail alone
///    submits all 48 layers as a single Metal command buffer (module docs, hazard 1). Walking the
///    slice in order splits it into one buffer per layer.
///
/// Free-standing rather than a method so the weights-free regression can drive it over a tiny
/// synthetic [`CausalLm`], with no packed encoder and no feature heads.
fn masked_hidden_states_in_order(
    model: &CausalLm,
    input_ids: &Array,
    mask01: &Array,
) -> Result<Vec<Array>> {
    let mask = causal_padding_mask(mask01, model.compute_dtype())?;
    let mut cache = model.new_cache();
    let states = model
        .hidden_states_with_mask(input_ids, &mut cache, 0, AttnMask::Additive(&mask))
        .map_err(from_llm)?;
    for state in &states {
        eval([state])?;
    }
    Ok(states)
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

/// The padding-mask regression (sc-18770 review issue 1), weights-free.
///
/// The bug this exists to catch: `CausalLm::hidden_states` masks causally and nothing else, so
/// under the tokenizer's *left* padding every valid token attends the pad run. The resulting
/// hidden states — and every feature built from them — are wrong while remaining finite, non-zero,
/// correctly shaped and prompt-separated, i.e. invisible to every other assertion in this crate.
#[cfg(test)]
mod padding_mask_tests {
    use super::*;
    use mlx_rs::ops::indexing::TryIndexOp;
    use std::collections::HashMap;

    const HIDDEN: i32 = 32;
    const VOCAB: i32 = 48;
    const HEAD_DIM: i32 = 8;
    const LAYERS: usize = 3;
    const HEADS: i32 = 4;
    const KV_HEADS: i32 = 2;
    const INTER: i32 = 64;

    fn host(a: &Array) -> Vec<f32> {
        let f = a.as_dtype(Dtype::Float32).expect("f32");
        eval([&f]).expect("eval");
        f.as_slice::<f32>().to_vec()
    }

    /// A tiny **real** `CausalLm` over synthetic weights — no packed encoder, no download, and the
    /// same seam `from_packed_av` builds through, so the mask genuinely flows into attention.
    fn tiny_model() -> CausalLm {
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
        let mut randn = |shape: &[i32]| -> Array {
            let n: i32 = shape.iter().product();
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.4
                })
                .collect();
            Array::from_slice(&data, shape)
        };

        let mut m: HashMap<String, Array> = HashMap::new();
        m.insert("model.embed_tokens.weight".into(), randn(&[VOCAB, HIDDEN]));
        m.insert(
            "model.norm.weight".into(),
            Array::ones::<f32>(&[HIDDEN]).expect("norm"),
        );
        for i in 0..LAYERS {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            m.insert(
                p("self_attn.q_proj.weight"),
                randn(&[HEADS * HEAD_DIM, HIDDEN]),
            );
            m.insert(
                p("self_attn.k_proj.weight"),
                randn(&[KV_HEADS * HEAD_DIM, HIDDEN]),
            );
            m.insert(
                p("self_attn.v_proj.weight"),
                randn(&[KV_HEADS * HEAD_DIM, HIDDEN]),
            );
            m.insert(
                p("self_attn.o_proj.weight"),
                randn(&[HIDDEN, HEADS * HEAD_DIM]),
            );
            m.insert(p("mlp.gate_proj.weight"), randn(&[INTER, HIDDEN]));
            m.insert(p("mlp.up_proj.weight"), randn(&[INTER, HIDDEN]));
            m.insert(p("mlp.down_proj.weight"), randn(&[HIDDEN, INTER]));
            m.insert(
                p("input_layernorm.weight"),
                Array::ones::<f32>(&[HIDDEN]).expect("in ln"),
            );
            m.insert(
                p("post_attention_layernorm.weight"),
                Array::ones::<f32>(&[HIDDEN]).expect("post ln"),
            );
        }
        CausalLm::from_weights(&LlmWeights::from_map(m), "", cfg).expect("tiny CausalLm")
    }

    fn ids(v: &[i32]) -> Array {
        Array::from_slice(v, &[1, v.len() as i32])
    }

    /// The mask rule itself: `valid(i, j) = j <= i && mask01[j] != 0`, everything else blocked.
    #[test]
    fn the_mask_blocks_pad_columns_and_the_causal_upper_triangle() {
        let mask01 = ids(&[0, 0, 1, 1]);
        let m = causal_padding_mask(&mask01, Dtype::Float32).expect("mask");
        assert_eq!(m.shape(), &[1, 1, 4, 4]);
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
    /// Replace `AttnMask::Additive(&mask)` in [`masked_hidden_states_in_order`] with
    /// `AttnMask::Causal` and this fails: the valid tokens then attend the pad run, whose keys and
    /// values differ between the two variants.
    #[test]
    fn left_pad_ids_do_not_reach_the_valid_positions() {
        let model = tiny_model();
        const PADS: usize = 3;
        const LEN: usize = 8;

        // Identical valid tail, different pad ids. `mask01` is the same for both.
        let a = ids(&[0, 0, 0, 3, 9, 14, 2, 7]);
        let b = ids(&[41, 17, 5, 3, 9, 14, 2, 7]);
        let mask01 = ids(&[0, 0, 0, 1, 1, 1, 1, 1]);

        let ha = masked_hidden_states_in_order(&model, &a, &mask01).expect("stack a");
        let hb = masked_hidden_states_in_order(&model, &b, &mask01).expect("stack b");
        assert_eq!(ha.len(), LAYERS + 1);

        // Sanity: the pad rows themselves DO differ, or the two inputs would be indistinguishable
        // and the assertion below would hold for a trivial reason.
        let pad_delta = host(&ha[LAYERS])
            .iter()
            .zip(host(&hb[LAYERS]).iter())
            .take(PADS * HIDDEN as usize)
            .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()));
        assert!(
            pad_delta > 0.0,
            "the two inputs must actually differ at the pad positions"
        );

        for (i, (sa, sb)) in ha.iter().zip(hb.iter()).enumerate() {
            let va = host(sa);
            let vb = host(sb);
            for t in PADS..LEN {
                let lo = t * HIDDEN as usize;
                let hi = lo + HIDDEN as usize;
                assert_eq!(
                    &va[lo..hi],
                    &vb[lo..hi],
                    "hidden state {i}, valid position {t} must not depend on the pad ids — the \
                     padding component of the attention mask was dropped"
                );
            }
        }

        // ...and the same through the extractor, which is what the feature heads consume.
        let na = per_token_rms_normed_hidden(&ha, &mask01, Dtype::Float32).expect("normed a");
        let nb = per_token_rms_normed_hidden(&hb, &mask01, Dtype::Float32).expect("normed b");
        let width = HIDDEN as usize * (LAYERS + 1);
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
        let model = tiny_model();
        let a = ids(&[0, 0, 0, 3, 9, 14, 2, 7]);
        let mask01 = ids(&[0, 0, 0, 1, 1, 1, 1, 1]);

        let masked = masked_hidden_states_in_order(&model, &a, &mask01).expect("masked");
        let mut cache = model.new_cache();
        let causal_only = model
            .hidden_states(&a, &mut cache, 0)
            .expect("causal-only stack");

        let delta = host(&masked[LAYERS])
            .iter()
            .zip(host(&causal_only[LAYERS]).iter())
            .skip(3 * HIDDEN as usize)
            .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()));
        assert!(
            delta > 1e-4,
            "causal-only masking must measurably differ from causal+padding at the valid \
             positions (got {delta}) — otherwise the regression above proves nothing"
        );
    }

    /// The `[0]` entry is the embeddings, so its pad rows differ by construction; the assertion
    /// above is about every *later* entry. Pinned separately so a future "just compare entry 0"
    /// simplification cannot quietly weaken it.
    #[test]
    fn the_embedding_entry_is_the_one_place_pad_ids_legitimately_show() {
        let model = tiny_model();
        let a = ids(&[0, 0, 0, 3, 9, 14, 2, 7]);
        let b = ids(&[41, 17, 5, 3, 9, 14, 2, 7]);
        let mask01 = ids(&[0, 0, 0, 1, 1, 1, 1, 1]);
        let ha = masked_hidden_states_in_order(&model, &a, &mask01).expect("a");
        let hb = masked_hidden_states_in_order(&model, &b, &mask01).expect("b");
        let e0 = ha[0].try_index((0, 0)).expect("row");
        let e1 = hb[0].try_index((0, 0)).expect("row");
        let delta = host(&e0)
            .iter()
            .zip(host(&e1).iter())
            .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()));
        assert!(delta > 0.0, "different pad ids must embed differently");
    }
}

#[cfg(test)]
mod version_assertion_tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

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

    /// A `__metadata__`-only transformer file: the `gemma_source_checkpoint` stamp the assertion
    /// reads, plus a `config.transformer` section so the bundle builder classifies it correctly.
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

    /// `from_bundle_av` — the documented production entry point — must actually reach the version
    /// assertion through the bundle: resolve both components, take the *transformer's* metadata and
    /// the *text encoder's* path, and hand them to the gate.
    ///
    /// Weights-free: both files are `__metadata__`-only safetensors, and the mismatch fires before
    /// a single tensor is read. Mutating `from_bundle_av` to read the *encoder's* metadata instead
    /// of the transformer's (or to pass the transformer's path as `te_path`) makes this fail.
    #[test]
    fn from_bundle_av_reaches_the_version_assertion_through_the_bundle() {
        use mlx_gen::gen_core::ltx_checkpoint::LtxBundleBuilder;

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

        let mut cfg = LtxConfig::video_only_defaults();
        cfg.caption_feature_version = mlx_gen::gen_core::ltx_checkpoint::CaptionFeatureVersion::V2;

        let msg = match Ltx25TextEncoder::from_bundle_av(
            &bundle,
            &GenWeights::empty(),
            &cfg,
            Precision::quant_bf16(8, 32),
            OffloadPolicy::Resident,
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

    /// The 4-layer `gemma4_unified` fixture `mlx-llm`'s decoder goldens are built from — a real
    /// Gemma 4 config and a matching complete weight set, shared rather than re-invented so this
    /// fixture cannot drift from the loader it is fed to.
    const DECODER_GOLDENS: &str =
        include_str!("../../../../llm/testdata/gemma4/gemma4_decoder_goldens.json");

    /// Write a **complete** packed text encoder: the goldens' weights plus the `__metadata__`
    /// `gemma_config` (with its `gemma_version` stamp) that [`gemma_config_value`] and the identity
    /// gate read. Unlike [`write_te`] this one is loadable, which is what a residency test needs.
    fn write_loadable_te(dir: &Path, gemma_version: &str) -> (PathBuf, serde_json::Value) {
        let goldens: serde_json::Value =
            serde_json::from_str(DECODER_GOLDENS).expect("parse gemma4 decoder goldens");
        let mut cfg = goldens["config"].clone();
        cfg["gemma_version"] = serde_json::json!(gemma_version);

        let mut arrays: Vec<(String, Array)> = Vec::new();
        for (key, entry) in goldens["weights"].as_object().expect("weights object") {
            let shape: Vec<i32> = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();
            let data: Vec<f32> = entry["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            arrays.push((key.clone(), Array::from_slice(&data, &shape)));
        }

        // The packed-asset floor `GemmaAssets::from_single_file` enforces: the tokenizer as a U8
        // payload tensor, and the two required sidecars — which it accepts as `__metadata__`
        // strings (the ComfyUI-pack fallback), so they need no tensors here.
        arrays.push(("tokenizer_json".to_string(), Array::from_slice(b"{}", &[2])));
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("gemma_config".to_string(), cfg.to_string());
        metadata.insert("tokenizer_config.json".to_string(), "{}".to_string());
        metadata.insert("processor_config.json".to_string(), "{}".to_string());

        let path = dir.join("text_encoder.safetensors");
        let refs: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, Some(&metadata), &path).expect("write packed te");
        (path, goldens["config"].clone())
    }

    /// R9 reachability: the `OffloadPolicy` a caller hands
    /// [`Ltx25TextEncoder`](super::Ltx25TextEncoder) must actually **reach the loader selection**,
    /// not merely be accepted and dropped.
    ///
    /// This is the sc-18456/18457 failure class — a rung declared, planned and even measured, but
    /// unreachable because the engine-keyed load seam never named it. Here the seam is
    /// [`load_backbone`]'s `match policy`, so that is where the check goes, rather than on the
    /// encoder's output (which is identical either way, by design).
    ///
    /// The observation is `mlx_llm::CausalLm::stream_observation`: `Some` only when the streaming
    /// loader was constructed. Note it distinguishes the two policies *before any forward runs* —
    /// this is a claim about the loader, not about what a pass did.
    ///
    /// MUTATION: make `load_backbone` ignore `policy` and always take the `Resident` arm — the
    /// Sequential half goes RED. Making it always take the Sequential arm reddens the Resident
    /// half. Both halves are needed: a one-sided check passes a loader wired to one branch.
    #[test]
    fn offload_policy_reaches_the_backbone_loader_selection() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (te, _cfg_json) = write_loadable_te(dir.path(), GEMMA4_VERSION);
        let ckpt = checkpoint("2.5.0", Some(GEMMA4_VERSION));

        let mut cfg = LtxConfig::video_only_defaults();
        cfg.caption_feature_version = mlx_gen::gen_core::ltx_checkpoint::CaptionFeatureVersion::V2;

        let (resident, _) = load_backbone(&ckpt, &te, &cfg, OffloadPolicy::Resident)
            .expect("the resident backbone loads");
        assert!(
            resident.stream_observation().is_none(),
            "OffloadPolicy::Resident must not construct the streaming loader"
        );

        let (streamed, _) = load_backbone(&ckpt, &te, &cfg, OffloadPolicy::Sequential)
            .expect("the sequential backbone loads");
        let obs = streamed.stream_observation().expect(
            "OffloadPolicy::Sequential must reach CausalLm::from_file_sequential — a policy that \
             is accepted and then dropped is the sc-18456 unreachable-rung class",
        );
        assert_eq!(
            obs.passes(),
            0,
            "the loader selection is observable before any forward runs"
        );
    }
}
