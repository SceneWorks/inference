//! The end-to-end Lens-Turbo / Lens T2I pipeline (sc-3173) — wires the four ported components into a
//! single `generate`: [`LensTokenizer`] → the gpt-oss
//! [`LensTextEncoder`] (multi-layer capture + the
//! `txt_offset = 97` slice) → the [`LensTransformer`] denoising DiT (with
//! the [`schedule`] flow-match sigmas + norm-rescaled CFG) → the Flux.2
//! [`vae::decode`](crate::vae) shim.
//!
//! A faithful port of `_vendor/lens/pipeline.py::LensPipeline.__call__`. The two model variants
//! (`lens`, `lens_turbo`) share this code and arch — they differ only in their sampling defaults
//! (registered in [`crate::registry`]).
//!
//! ## Parity-critical details (from the reference `__call__`)
//! - **Encode → offset slice → align.** Positives and negatives are each encoded to the four captured
//!   gpt-oss layers, sliced at `input_ids[97:]`, then zero-padded to a shared `S_txt` and stacked
//!   `[pos; neg]` along the batch axis for the joint CFG forward. An **empty** negative is the
//!   unconditional branch: zero text features + an all-`false` mask (no text tokens), *not* a second
//!   encode (`encode_prompt`).
//! - **Joint CFG batch.** Each step runs the DiT once over `B·2` (here `B = 1`): `hidden = [x; x]`,
//!   `encoder_features = [pos; neg]`. The output splits `cond, uncond`; the per-step guidance is the
//!   **norm-rescaled** CFG (`cfg_rescale`, delegating to the shared `gen_core::guidance`).
//! - **…except when CFG is off** (sc-17616). At guidance exactly `1.0` — the `lens_turbo` default —
//!   the combine reduces to `cond`, so the uncond half is neither encoded nor batched: the encode
//!   returns cond-only `[1, …]` conditioning and the denoise runs a `B = 1` forward. That decision is
//!   made ONCE, by `needs_joint_cfg` inside `assemble_conditioning`; the denoise lanes consume the
//!   batch they are given rather than narrowing a `[pos; neg]` stack themselves.
//! - **Timestep.** The transformer is fed the *shifted sigma* directly (the reference `timestep /
//!   1000`, where `scheduler.timesteps = sigma · 1000`) — i.e. [`schedule::timesteps`].
//! - **Latents.** `[B, latent_h · latent_w, 128]`, `latent_{h,w} = {height,width} / 16`; the denoise
//!   is the core flow-match Euler step.

use mlx_rs::ops::{concatenate_axis, split, split_sections};
use mlx_rs::{Array, Dtype};

use mlx_gen::attention::AttentionPlan;
use mlx_gen::gen_core;
use mlx_gen::scheduler::compute_mu;
use mlx_gen::weights::Weights;
use mlx_gen::{
    resolve_flow_schedule, run_flow_sampler_with_latent_hook, CancelFlag, Error, Image,
    MlxLatentOps, PreviewSink, Progress, Quant, Result, TimestepConvention,
};
use mlx_gen_flux2::{load_vae, Flux2Vae};

use crate::config::GptOssConfig;
use crate::dit::{LensDitConfig, LensTransformer};
use crate::schedule::{self, lens_schedule};
use crate::text::{LensTokenizer, TXT_OFFSET};
use crate::text_encoder::encoder::LensTextEncoder;
use crate::vae;

/// The VAE downsample factor (`vae_scale_factor`): a Lens latent cell maps to a 16×16 pixel tile
/// (Flux.2's 8× conv VAE composed with the 2× DiT patchify).
pub const VAE_SCALE_FACTOR: u32 = 16;

/// Whether a denoise step still needs the joint `[cond; uncond]` (B=2) DiT forward, or can run the
/// conditional half alone (B=1). At exactly `guidance == 1.0` (the turbo default) the CFG combine
/// `uncond + 1·(cond − uncond)` reduces to `cond` and the norm-rescale's ratio `‖cond‖/‖comb‖`
/// reduces to `1`, so the entire uncond branch (its own 48-block DiT forward) is dead weight — skip
/// it. Extracted as a pure predicate so the (otherwise real-weight-only) batching decision is unit
/// testable (F-020).
///
/// This is also the **single** predicate that decides the conditioning batch (sc-17616): the encode
/// (`LensText::encode_prompt*`) and the denoise lanes both derive their batching from it over the same
/// `guidance`, so the conditioning a lane receives always already has the batch that lane expects. The
/// lanes therefore do NOT re-narrow — see [`LensText::encode_prompt_windowed`].
///
/// NOTE (parity): this is NOT bit-identical to the un-skipped B=2 output. The un-skipped path computes
/// `uncond + 1.0·(cond − uncond)` (a lossy subtract-then-add over the genuinely-different uncond DiT
/// output) then multiplies by a ratio within ~1 ULP of 1; the skipped path feeds `cond` in for both
/// operands, so the residual (~1 ULP, far below the ~1e-2 DiT parity tolerance) is dropped. The
/// arithmetic still routes through `cfg_rescale(cond, cond, 1.0)` so the rescale op-order is preserved.
pub(crate) fn needs_joint_cfg(guidance: f32) -> bool {
    guidance != 1.0
}

/// Whether the encoder mask marks **every** text position valid, in which case the DiT's joint
/// attention mask is identically zero and can be skipped entirely (sc-17719).
///
/// `build_joint_mask` turns `text_valid` into an additive `(valid − 1)·1e9`, so an all-ones mask is an
/// all-zero `[B, 1, 1, img_len + txt_len]` tensor that is then broadcast-added to the **full** score
/// matrix `[B, heads, q, k]` in every one of the 48 blocks, every step. At 256² that is small; at
/// 2048² the score matrix is ~16.4 k × 16.4 k per head, so adding a known-zero to it is the single
/// largest piece of pure waste in the denoise. `LensTransformer::forward` documents `text_valid: None`
/// as the skip path — this is the predicate that lets the lanes take it.
///
/// After sc-17616 the all-ones case is the *common* one: a CFG-off encode returns the cond-only row
/// (`encode_one` produces `ones([1, s])` for a single unpadded prompt), and an empty negative pads
/// nothing. A mask only carries zeros when the two prompts differ in length and the shorter is padded.
///
/// Costs one host sync per render — evaluated once outside the step loop, never per step.
fn text_mask_is_all_valid(encoder_mask: &Array) -> Result<bool> {
    Ok(mlx_rs::ops::min(encoder_mask, None)?.item::<f32>() == 1.0)
}

/// Reject conditioning whose batch does not match the batch this guidance selects (sc-17616).
///
/// The encode and the denoise lanes each derive their batching from [`needs_joint_cfg`] over the
/// guidance scale, so they agree by construction whenever the caller passes the same scale to both.
/// The lanes take the conditioning and the scale as separate arguments, though, and several are
/// `pub` — so a caller *can* still encode at one guidance and render at another. Checking here turns
/// that into a named error at the stage boundary instead of a raw MLX shape message from somewhere
/// inside the first block's joint attention.
fn check_conditioning_batch(
    encoder_features: &[Array],
    encoder_mask: &Array,
    joint: bool,
    guidance_scale: f32,
) -> Result<()> {
    let want = if joint { 2 } else { 1 };
    let got = encoder_mask.shape()[0];
    let feats = encoder_features
        .first()
        .map(|f| f.shape()[0])
        .unwrap_or(got);
    if got != want || feats != want {
        return Err(Error::Msg(format!(
            "lens denoise: conditioning batch (features {feats}, mask {got}) does not match \
             guidance {guidance_scale}, which needs batch {want} — encode the prompt with the same \
             guidance the denoise runs (sc-17616)"
        )));
    }
    Ok(())
}

/// Norm-rescaled classifier-free guidance for Lens — the per-token (channel-axis `[-1]`) geometry of
/// `[B, seq, C]` predictions, delegating to the backend-neutral [`gen_core::guidance::cfg_rescale`]
/// (epic 7434 P3, sc-7441). Replaces the bespoke `schedule::cfg_rescale`: the math now lives once in
/// gen-core, with the MLX [`MlxLatentOps`] supplying the ops. Byte-identical to the retired function
/// (same combine op-order, `1e-12` floor, `where(‖comb‖>0, …, 1)` guard). `shape` is unused on the
/// MLX backend (the `Array` carries its own shape), so an empty slice is passed.
fn cfg_rescale(cond: &Array, uncond: &Array, guidance: f32) -> Result<Array> {
    Ok(gen_core::guidance::cfg_rescale(
        &MlxLatentOps,
        cond,
        uncond,
        guidance,
        &[],
        &[-1],
    )?)
}

/// Default harmony-preamble date (`Current date:`). The preamble is the first [`TXT_OFFSET`] tokens,
/// which are **sliced off** before the DiT conditioning, so its `date` line never reaches the image
/// path — a fixed constant keeps generation deterministic regardless of wall-clock. (The Python
/// worker passes the live date; the value is image-irrelevant.)
pub const DEFAULT_DATE: &str = "2025-01-01";

/// Default reasoner decode temperature (F-105) — the vendor `PromptReasoner.__init__` default `0.7`
/// (stochastic sampling). `0.0` selects deterministic greedy decode. Only consulted when
/// [`GenerateOptions::enable_reasoner`] is set; image parity is unaffected (the reasoner only rewrites
/// the prompt text before encoding).
pub const DEFAULT_REASONER_TEMPERATURE: f32 = 0.7;

/// Options for a single [`LensPipeline::generate`] call.
pub struct GenerateOptions<'a> {
    pub prompt: &'a str,
    /// Empty ⇒ the unconditional branch (zero text features), matching the reference default `""`.
    pub negative_prompt: &'a str,
    /// Output pixels — both must be divisible by [`VAE_SCALE_FACTOR`] (use [`crate::resolution`]).
    pub height: u32,
    pub width: u32,
    pub num_steps: usize,
    pub guidance_scale: f32,
    /// Curated sampler name (epic 7114 sc-7305); `None` ⇒ the engine default `euler`, byte-equivalent
    /// to the legacy flow-match loop (N1). The legacy `flow_match_euler` alias / any unknown name falls
    /// back to `euler` (N3).
    pub sampler: Option<&'a str>,
    /// Curated scheduler name (epic 7114 sc-7305); `None` ⇒ the native empirical-μ flow-match schedule
    /// (the byte-exact N1 default). The legacy `flow_match` alias / any unknown name falls back to it (N3).
    pub scheduler: Option<&'a str>,
    pub seed: u64,
    /// Harmony-preamble `Current date:` (image-irrelevant for the encode path; the **reasoner** uses
    /// it for its own preamble — see [`DEFAULT_DATE`]).
    pub date: &'a str,
    /// Refine the prompt through the attached local [`LensReasoner`](crate::reasoner::LensReasoner)
    /// before encoding (sc-3176, the vendor `enable_reasoner`). Requires
    /// [`attach_reasoner`](LensPipeline::attach_reasoner); off by default.
    pub enable_reasoner: bool,
    /// Reasoner decode temperature (F-105); consulted only when `enable_reasoner` is set. `> 0` samples
    /// (seeded by [`seed`](Self::seed)) — the vendor `PromptReasoner` behavior; `0.0` is deterministic
    /// greedy. Defaults to [`DEFAULT_REASONER_TEMPERATURE`] (the vendor `0.7`). The reasoner only
    /// rewrites the prompt text, so this does not affect image sampling.
    pub reasoner_temperature: f32,
}

/// The **text-encode phase** of a Lens pipeline (sc-11030): the tokenizer + the gpt-oss MoE text
/// encoder. The encoder is the LARGEST component (the ~20 B-param gpt-oss, ~13 GB q4 — comparable to
/// or larger than the DiT), so it is the one dropped first under `OffloadPolicy::Sequential`: encode →
/// **drop** → denoise/decode bounds peak unified memory to `max(text-encoder, DiT+VAE)`.
pub struct LensText {
    tokenizer: LensTokenizer,
    encoder: LensTextEncoder,
    num_text_layers: usize,
    dtype: Dtype,
}

/// The **heavy render phase** of a Lens pipeline (sc-11030): the denoising DiT + the Flux.2 VAE. Held
/// after the text encoder is dropped under `Sequential`; both residencies run the identical
/// [`render`](LensHeavy::render) body, so a `Sequential` job is byte-identical to `Resident`.
pub struct LensHeavy {
    transformer: LensTransformer,
    vae: Flux2Vae,
    dtype: Dtype,
}

/// A loaded Lens pipeline: the [`LensText`] + [`LensHeavy`] phases (sc-11030 split) held together,
/// plus the **optional** local prompt reasoner (sc-3176 — `None` unless
/// [`attach_reasoner`](LensPipeline::attach_reasoner)d, so the default pipeline carries no extra
/// gpt-oss footprint for an off-by-default feature). This is the direct struct API used by the e2e
/// parity tests + the trainer; the registry generator ([`crate::registry`]) holds the two phases
/// separately so it can drop the encoder mid-job for `Sequential` residency.
pub struct LensPipeline {
    text: LensText,
    heavy: LensHeavy,
    reasoner: Option<crate::reasoner::LensReasoner>,
}

impl LensPipeline {
    /// Load all four components from a `microsoft/Lens-Turbo` (or `microsoft/Lens`) snapshot directory
    /// at `dtype` (bf16 production / f32 tight-gate). The snapshot is the diffusers multi-component
    /// tree: `tokenizer/tokenizer.json`, `text_encoder/`, `transformer/`, `vae/`. The VAE always runs
    /// f32 internally (the shared Flux.2 decoder).
    pub fn load(snapshot_dir: impl AsRef<std::path::Path>, dtype: Dtype) -> Result<Self> {
        Self::load_quant(snapshot_dir, dtype, None)
    }

    /// As [`load`](Self::load) but quantizes the gpt-oss encoder's MoE experts to Q4/Q8 (sc-3172) so
    /// the encoder loads at `~12 GB` instead of `~40 GB` bf16 — the per-layer dequant is the only
    /// transient. The DiT and VAE stay dense (DiT quant is sc-3175); the dominant footprint is the
    /// 20 B-param encoder, so quantizing it is the memory win.
    pub fn load_quant(
        snapshot_dir: impl AsRef<std::path::Path>,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let root = snapshot_dir.as_ref();
        Ok(Self {
            text: LensText::load(root, dtype, quant)?,
            heavy: LensHeavy::load(root, dtype)?,
            reasoner: None,
        })
    }

    /// Attach a local prompt [`LensReasoner`](crate::reasoner::LensReasoner) (sc-3176), enabling the
    /// `enable_reasoner` path in [`generate`](Self::generate). Loaded separately (its own gpt-oss copy)
    /// so the base pipeline pays nothing for this off-by-default feature.
    pub fn attach_reasoner(&mut self, reasoner: crate::reasoner::LensReasoner) {
        self.reasoner = Some(reasoner);
    }
}

impl LensText {
    /// Whether this phase was constructed with the Sequential-only re-openable encoder stack.
    pub fn is_streamable(&self) -> bool {
        self.encoder.is_streamable()
    }

    /// Load the text-encode phase (sc-11030) — the tokenizer + the gpt-oss MoE text encoder — from a
    /// Lens snapshot's `tokenizer/` + `text_encoder/`. `quant` (Q4/Q8) quantizes the encoder's MoE
    /// experts at load (sc-3172): the ~38 GB / 20 B-param bulk → ~12 GB, the per-layer dequant the only
    /// transient. `num_text_layers` is the arch constant (the DiT text-conditioning layer count), so
    /// this loads without touching the DiT — the split that lets `Sequential` drop the encoder before
    /// the DiT is loaded.
    pub fn load(root: &std::path::Path, dtype: Dtype, quant: Option<Quant>) -> Result<Self> {
        Self::load_with_streaming(root, dtype, quant, false)
    }

    /// Sequential-only loader for rung 4's text-encoder component scope. Resident callers continue
    /// through [`Self::load`] and retain the warm layer stack across requests.
    pub fn load_streamable(
        root: &std::path::Path,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        Self::load_with_streaming(root, dtype, quant, true)
    }

    fn load_with_streaming(
        root: &std::path::Path,
        dtype: Dtype,
        quant: Option<Quant>,
        streamable: bool,
    ) -> Result<Self> {
        let tokenizer = LensTokenizer::from_file(root.join("tokenizer").join("tokenizer.json"))?;
        let enc_cfg = GptOssConfig::lens();
        let text_encoder_dir = root.join("text_encoder");
        let enc_w = Weights::from_dir(&text_encoder_dir)?;
        let encoder = if streamable {
            LensTextEncoder::from_streamable_source(
                enc_w,
                mlx_gen::WeightsSource::Dir(text_encoder_dir),
                &enc_cfg,
                dtype,
                crate::text_encoder::encoder::DEFAULT_SELECTED_LAYERS.to_vec(),
                quant,
            )?
        } else {
            LensTextEncoder::from_weights_quant(enc_w, &enc_cfg, dtype, quant)?
        };
        Ok(Self {
            tokenizer,
            encoder,
            num_text_layers: LensDitConfig::lens().num_text_layers,
            dtype,
        })
    }

    /// Encode one prompt to its per-layer DiT text features (sliced at [`TXT_OFFSET`]) + the valid
    /// mask. Returns `(features, mask)` where `features` is `num_text_layers × [1, S, 2880]` and
    /// `mask` is `[1, S]` (all-`1`; a single prompt is unpadded). When the rendered prompt is ≤ the
    /// offset (never, for real prompts) the features collapse to length 0 (`_get_text_embeddings`).
    fn encode_one(
        &self,
        prompt: &str,
        date: &str,
        cancel: Option<&CancelFlag>,
        window: Option<usize>,
    ) -> Result<(Vec<Array>, Array)> {
        let out = self.tokenizer.encode(prompt, date)?;
        let l = out.ids.len();
        let input_ids = Array::from_slice(&out.ids, &[1, l as i32]);
        let layers = match window {
            Some(window) => self.encoder.encode_windowed(
                &input_ids,
                window,
                cancel.unwrap_or(&CancelFlag::default()),
            )?,
            None => self.encoder.encode(&input_ids, cancel)?,
        }; // num_text_layers × [1, L, 2880]

        let offset = TXT_OFFSET as i32;
        if l as i32 > offset {
            let s = l as i32 - offset;
            // `[:, offset:, :]` — split at `offset` along the sequence axis, keep the tail.
            let features = layers
                .iter()
                .map(|f| Ok(split_sections(f, &[offset], 1)?[1].clone()))
                .collect::<Result<Vec<_>>>()?;
            // Single unpadded prompt ⇒ every retained token is valid.
            let mask = mlx_rs::ops::ones::<f32>(&[1, s])?;
            Ok((features, mask))
        } else {
            // `input_ids` shorter than the offset (never for a real prompt): length-0 features.
            let dim = layers[0].shape()[2];
            let features = (0..self.num_text_layers)
                .map(|_| Ok(mlx_rs::ops::zeros::<f32>(&[1, 0, dim])?.as_dtype(self.dtype)?))
                .collect::<Result<Vec<_>>>()?;
            let mask = mlx_rs::ops::zeros::<f32>(&[1, 0])?;
            Ok((features, mask))
        }
    }

    /// Encode positives + negatives and assemble the joint CFG batch (`encode_prompt` +
    /// `_align_text_features` + the `[pos; neg]` stack). Returns `(encoder_features, encoder_mask)`
    /// where each feature layer is `[2, S_txt, 2880]` and the mask is `[2, S_txt]` (`1` = valid) — or
    /// batch **1** when `guidance_scale` disables CFG, see [`Self::encode_prompt_windowed`].
    pub fn encode_prompt(
        &self,
        prompt: &str,
        negative_prompt: &str,
        date: &str,
        guidance_scale: f32,
        cancel: Option<&CancelFlag>,
    ) -> Result<(Vec<Array>, Array)> {
        self.encode_prompt_windowed(prompt, negative_prompt, date, guidance_scale, cancel, None)
    }

    /// [`Self::encode_prompt`] with an optional rung-4 text-encoder window. The option is resolved at
    /// the generator boundary so direct struct/training callers remain resident by default.
    ///
    /// **The CFG batching is decided here, once** (sc-17616), mirroring `candle-gen-lens`. When
    /// `needs_joint_cfg` is false for `guidance_scale` (i.e. exactly `1.0`, the `lens_turbo` default)
    /// the joint combine reduces to `cond`, so the uncond half is neither encoded nor batched: each
    /// feature layer comes back `[1, S_txt, 2880]` and the mask `[1, S_txt]`. The denoise lanes select
    /// their B=1 forward from the **same** predicate over the **same** `guidance_scale`, so they consume
    /// this shaping directly and never re-narrow — a lane that forgets to narrow, or narrows the
    /// features without the mask, is the sc-14195 defect shape and can no longer arise from a lane.
    ///
    /// Taking the scale rather than a pre-derived `bool` (the one deliberate divergence from the candle
    /// twin) is what keeps that true: caller and lanes cannot disagree about a value neither of them
    /// derives. A caller passing *different* scales to the encode and the denoise is still possible —
    /// the denoise lanes reject that at the stage boundary with a named error
    /// (`check_conditioning_batch`).
    pub fn encode_prompt_windowed(
        &self,
        prompt: &str,
        negative_prompt: &str,
        date: &str,
        guidance_scale: f32,
        cancel: Option<&CancelFlag>,
        window: Option<usize>,
    ) -> Result<(Vec<Array>, Array)> {
        let positive = self.encode_one(prompt, date, cancel, window)?;
        assemble_conditioning(positive, guidance_scale, self.dtype, || {
            // Honor a cancel between the positive and negative MoE encodes (F-019) — each is a full
            // ~24-layer 20B forward, so the gap between them is a meaningful cancellation point.
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(Error::Canceled);
            }
            // Empty negative ⇒ the unconditional branch: zero text features matching the positive
            // shape + an all-`false` (all-zero) mask. A non-empty negative is encoded normally.
            if negative_prompt.trim().is_empty() {
                return Ok(None);
            }
            self.encode_one(negative_prompt, date, cancel, window)
                .map(Some)
        })
    }
}

/// Assemble the DiT conditioning from the encoded positive, applying the **single** CFG batching gate
/// (sc-17616). This is where the `[pos; neg]`-vs-cond-only decision lives for every lane.
///
/// - `!needs_joint_cfg(guidance_scale)` ⇒ cond-only: returns the positive alone, batch 1, and
///   `encode_negative` is **never called** — no wasted ~20 B-param uncond encode.
/// - otherwise ⇒ the joint batch: `encode_negative` supplies the uncond stream (`None` ⇒ the empty
///   negative's zero features + zero mask), both sides zero-pad to `S_txt = max(s_pos, s_neg)`, and
///   features and mask are stacked **together** so the mask can never describe a different batch than
///   the features it accompanies (the sc-14195 defect shape).
///
/// Split out from [`LensText::encode_prompt_windowed`] so the gate is unit-testable on synthetic
/// arrays — the encoder itself needs the 12 GB real snapshot. `pub(crate)` so the trainer's
/// preview pre-encode goes through this same gate instead of hand-rolling a second copy of it.
pub(crate) fn assemble_conditioning(
    positive: (Vec<Array>, Array),
    guidance_scale: f32,
    dtype: Dtype,
    encode_negative: impl FnOnce() -> Result<Option<(Vec<Array>, Array)>>,
) -> Result<(Vec<Array>, Array)> {
    let (pos_feats, pos_mask) = positive;
    if !needs_joint_cfg(guidance_scale) {
        // CFG off: cond-only conditioning — skip the uncond encode and the batch entirely.
        let features = pos_feats
            .iter()
            .map(|f| Ok(f.as_dtype(dtype)?))
            .collect::<Result<Vec<_>>>()?;
        return Ok((features, pos_mask));
    }
    let s_pos = pos_feats[0].shape()[1];

    let (neg_feats, neg_mask) = match encode_negative()? {
        Some(encoded) => encoded,
        None => {
            let zeros = pos_feats
                .iter()
                .map(mlx_rs::ops::zeros_like)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            (zeros, mlx_rs::ops::zeros_like(&pos_mask)?)
        }
    };
    let s_neg = neg_feats[0].shape()[1];

    // Pad both to a shared S_txt = max(s_pos, s_neg).
    let target = s_pos.max(s_neg);
    let pos_feats = pad_features(&pos_feats, s_pos, target)?;
    let neg_feats = pad_features(&neg_feats, s_neg, target)?;
    let pos_mask = pad_mask(&pos_mask, s_pos, target)?;
    let neg_mask = pad_mask(&neg_mask, s_neg, target)?;

    // Stack [pos; neg] along the batch axis → the joint CFG forward.
    let mut encoder_features = Vec::with_capacity(pos_feats.len());
    for (pf, nf) in pos_feats.iter().zip(neg_feats.iter()) {
        encoder_features.push(concatenate_axis(&[pf, nf], 0)?.as_dtype(dtype)?);
    }
    let encoder_mask = concatenate_axis(&[&pos_mask, &neg_mask], 0)?; // [2, S_txt]
    Ok((encoder_features, encoder_mask))
}

impl LensHeavy {
    /// Load the heavy render phase (sc-11030) — the denoising DiT + the Flux.2 VAE, dense — from a Lens
    /// snapshot's `transformer/` + `vae/`. DiT quant (sc-3175) and LoRA/LoKr (sc-3174) are applied
    /// **after** load via [`quantize_dit`](Self::quantize_dit) / [`apply_adapters`](Self::apply_adapters)
    /// (the registry orchestrates the adapters-then-quantize order); the VAE always runs f32 internally.
    /// Independent of the text encoder (separate weight files, RNG-free), so the `Sequential` path
    /// rebuilds a byte-identical bundle after the encoder is dropped.
    pub fn load(root: &std::path::Path, dtype: Dtype) -> Result<Self> {
        let dit_cfg = LensDitConfig::lens();
        let dit_w = Weights::from_dir(root.join("transformer"))?;
        let transformer = LensTransformer::from_weights(&dit_w, &dit_cfg, dtype)?;
        let vae = load_vae(root)?;
        Ok(Self {
            transformer,
            vae,
            dtype,
        })
    }

    /// Deferred DiT loader used only by request-scoped rung 4. The VAE remains resident within the
    /// heavy phase; the transformer retains only its front/back projections and a reopenable block
    /// source.
    pub fn load_streamable(
        root: &std::path::Path,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let cfg = LensDitConfig::lens();
        let transformer_dir = root.join("transformer");
        let weights = Weights::from_dir(&transformer_dir)?;
        let transformer = LensTransformer::from_streamable_source(
            weights,
            mlx_gen::WeightsSource::Dir(transformer_dir),
            &cfg,
            dtype,
            quant,
        )?;
        Ok(Self {
            transformer,
            vae: load_vae(root)?,
            dtype,
        })
    }

    /// The denoising loop over pre-encoded conditioning + an initial latent, on the engine **default**
    /// sampler/scheduler (`euler` over the native empirical-μ flow-match schedule). Exposed for the e2e
    /// parity gate (which injects the reference's initial latents to factor out cross-RNG noise). A thin
    /// wrapper over [`denoise_with_sampler`](Self::denoise_with_sampler) forcing the default — `euler`
    /// reproduces the legacy `FlowMatchEuler::step` loop within the N1 parity tolerance (the shared
    /// `flow_match_euler_step` + gen-core keystone equivalence).
    ///
    /// - `encoder_features`: `num_text_layers × [2, S_txt, 2880]` (`[pos; neg]`).
    /// - `encoder_mask`: `[2, S_txt]` (`1` = valid).
    /// - `init_latents`: `[1, latent_h · latent_w, 128]`.
    ///
    /// Resolve the descending flow-match σ schedule for a generation (`num_steps + 1` entries, trailing
    /// `0.0`) — the exact vector [`denoise_with_sampler`](Self::denoise_with_sampler) feeds the solver.
    /// Factored out so the registry can compute the PiD `from_ldm` early-stop capture plan (sc-8048)
    /// against the same σ trajectory the denoise runs, then truncate the loop to `keep`.
    pub fn resolve_sigmas(
        &self,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        scheduler_name: Option<&str>,
    ) -> Vec<f32> {
        let native = lens_schedule(num_steps, latent_h, latent_w).sigmas;
        let mu = compute_mu(latent_h * latent_w, num_steps);
        resolve_flow_schedule(scheduler_name, mu, num_steps, &native)
    }

    /// Returns the final latents `[1, latent_h · latent_w, 128]` (patch-space; feed to [`vae::decode`]).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        init_latents: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Array> {
        self.denoise_with_sampler(
            encoder_features,
            encoder_mask,
            init_latents,
            latent_h,
            latent_w,
            num_steps,
            guidance_scale,
            None, // default sampler — `euler` == the legacy flow-match loop
            None, // native empirical-μ schedule
            0,    // seed unused by the deterministic default; stochastic solvers key off it
            cancel,
            on_step,
        )
    }

    /// The denoising loop routed through the unified curated-sampler framework (epic 7114 sc-7305): the
    /// per-generation `sampler` (integration method) + `scheduler` (σ schedule) knobs. Lens is
    /// rectified-flow (FLOW prediction) fed the **raw shifted sigma** as its timestep
    /// ([`TimestepConvention::Sigma`]); each step's velocity is the norm-rescaled joint-CFG combination
    /// (one DiT forward over `[pos; neg]`, `cfg_rescale`) — the body of the legacy loop, now
    /// driven by the curated solver.
    ///
    /// - `sampler_name`: a curated solver name (`euler` / `heun` / `dpmpp_2m` / `uni_pc` / …). `None`,
    ///   the legacy `flow_match_euler` alias, or any unknown name falls back to `euler` (N3) — the
    ///   byte-equivalent default.
    /// - `scheduler_name`: a curated scheduler name (`karras` / `exponential` / …). `None`, the legacy
    ///   `flow_match` alias, or any unknown name returns the native schedule verbatim (the N1 byte-exact
    ///   default).
    /// - `seed`: drives the per-step noise of the stochastic solvers (`euler_ancestral` / `dpmpp_sde` /
    ///   `lcm`); the deterministic solvers ignore it.
    ///
    /// The full-schedule entry; [`denoise_with_sampler_keep`](Self::denoise_with_sampler_keep) is the
    /// PiD `from_ldm` early-stop (sc-8048) variant that additionally truncates the descending schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_with_sampler(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        init_latents: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        sampler_name: Option<&str>,
        scheduler_name: Option<&str>,
        seed: u64,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Array> {
        self.denoise_with_sampler_keep(
            encoder_features,
            encoder_mask,
            init_latents,
            latent_h,
            latent_w,
            num_steps,
            guidance_scale,
            sampler_name,
            scheduler_name,
            seed,
            None, // full schedule — no PiD early-stop capture
            cancel,
            on_step,
        )
    }

    /// Preview-aware sibling of [`Self::denoise_with_sampler`].
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_with_sampler_with_preview(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        init_latents: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        sampler_name: Option<&str>,
        scheduler_name: Option<&str>,
        seed: u64,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
        preview: &PreviewSink,
    ) -> Result<Array> {
        self.denoise_with_sampler_keep_with_preview(
            encoder_features,
            encoder_mask,
            init_latents,
            latent_h,
            latent_w,
            num_steps,
            guidance_scale,
            sampler_name,
            scheduler_name,
            seed,
            None,
            cancel,
            on_step,
            preview,
        )
    }

    /// PiD `from_ldm` early-stop (sc-8048) denoise: as [`denoise_with_sampler`](Self::denoise_with_sampler)
    /// but truncates the descending flow-match schedule to `keep` (`Some(k)` → run over `sigmas[..k]`,
    /// leaving the latent at the capture σ `sigmas[k-1]`; `None` → the full schedule, byte-identical).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_with_sampler_keep(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        init_latents: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        sampler_name: Option<&str>,
        scheduler_name: Option<&str>,
        seed: u64,
        keep: Option<usize>,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Array> {
        self.denoise_with_sampler_keep_with_preview(
            encoder_features,
            encoder_mask,
            init_latents,
            latent_h,
            latent_w,
            num_steps,
            guidance_scale,
            sampler_name,
            scheduler_name,
            seed,
            keep,
            cancel,
            on_step,
            &PreviewSink::default(),
        )
    }

    /// Preview-aware sibling of [`Self::denoise_with_sampler_keep`].
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_with_sampler_keep_with_preview(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        init_latents: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        sampler_name: Option<&str>,
        scheduler_name: Option<&str>,
        seed: u64,
        keep: Option<usize>,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
        preview: &PreviewSink,
    ) -> Result<Array> {
        self.denoise_with_sampler_keep_with_preview_memory(
            encoder_features,
            encoder_mask,
            init_latents,
            latent_h,
            latent_w,
            num_steps,
            guidance_scale,
            sampler_name,
            scheduler_name,
            seed,
            keep,
            cancel,
            on_step,
            preview,
            AttentionPlan::UNBOUNDED,
            None,
        )
    }

    /// Request-scoped denoise body with the shared attention plan and optional 48-block DiT window.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn denoise_with_sampler_keep_with_preview_memory(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        init_latents: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        sampler_name: Option<&str>,
        scheduler_name: Option<&str>,
        seed: u64,
        keep: Option<usize>,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
        preview: &PreviewSink,
        attention: AttentionPlan<'_>,
        transformer_window: Option<usize>,
    ) -> Result<Array> {
        // The native Lens schedule is the byte-exact N1 default: empirical-μ flow-match sigmas (length
        // num_steps + 1, trailing 0). A curated `scheduler_name` re-shapes σ over the SAME empirical μ
        // (`compute_mu(seq_len, steps)`), so `karras` / `exponential` / … stay consistent with Lens's
        // resolution-/step-dependent shift instead of degrading to a linear ramp; an unset / legacy /
        // unknown name returns this native schedule verbatim.
        let sigmas = self.resolve_sigmas(latent_h, latent_w, num_steps, scheduler_name);
        // PiD `from_ldm` early-stop (sc-8048): `keep` truncates the descending schedule so the denoise
        // stops at the achieved-σ step and hands PiD the partially-denoised latent (`sigmas[keep-1]` is
        // the capture σ; Lens is `vp_frame=false` so the schedule σ *is* the degrade σ). `None` → the
        // full schedule → byte-identical clean path. Lens is pure T2I (no img2img blend), so the slice
        // is `..keep` and there is no `start_step` offset.
        let sigmas: &[f32] = match keep {
            Some(k) => &sigmas[..k],
            None => &sigmas,
        };

        let dtype = self.dtype;
        let transformer = &self.transformer;
        // FLOW velocity field: the norm-rescaled joint-CFG combination over one DiT forward. Lens feeds
        // the raw shifted sigma as the timestep directly (Sigma convention), matching the legacy loop.
        // At guidance 1.0 (the turbo default) the uncond DiT forward is dead weight — run the cond
        // half alone (B=1) and route it through `cfg_rescale(cond, cond, 1.0)` (F-020). The
        // conditioning already arrives cond-only in that case: `LensText::encode_prompt*` gates on the
        // SAME `needs_joint_cfg` over the SAME guidance (sc-17616), so this lane consumes it as given
        // rather than narrowing a `[pos; neg]` stack itself.
        let joint = needs_joint_cfg(guidance_scale);
        check_conditioning_batch(encoder_features, encoder_mask, joint, guidance_scale)?;
        // sc-17719: an all-valid mask is an all-zero additive term over the whole score matrix in every
        // block, every step. Resolve once, here, and hand the DiT its documented `None` skip path.
        let text_valid = (!text_mask_is_all_valid(encoder_mask)?).then_some(encoder_mask);
        let predict = |latents: &Array, sigma: f32| -> Result<Array> {
            if joint {
                // Joint CFG batch: duplicate the latent (cond/uncond share x_t), one DiT call.
                let hidden = concatenate_axis(&[latents, latents], 0)?; // [2, seq, 128]
                let timestep = Array::from_slice(&[sigma, sigma], &[2]).as_dtype(dtype)?;
                let noise = transformer.forward_with_memory(
                    &hidden,
                    encoder_features,
                    text_valid,
                    &timestep,
                    1,
                    latent_h,
                    latent_w,
                    attention,
                    transformer_window,
                    cancel,
                )?;
                // chunk(2) → cond (positive, batch 0), uncond (negative, batch 1).
                let parts = split(&noise, 2, 0)?;
                cfg_rescale(&parts[0], &parts[1], guidance_scale)
            } else {
                let timestep = Array::from_slice(&[sigma], &[1]).as_dtype(dtype)?;
                let cond = transformer.forward_with_memory(
                    latents,
                    encoder_features,
                    text_valid,
                    &timestep,
                    1,
                    latent_h,
                    latent_w,
                    attention,
                    transformer_window,
                    cancel,
                )?;
                cfg_rescale(&cond, &cond, guidance_scale)
            }
        };

        // Adapt the framework's `Progress` callback to the pipeline's (completed, total) step callback.
        let mut on_progress = |p: Progress| {
            if let Progress::Step { current, total } = p {
                on_step(current as usize, total as usize);
            }
        };

        // Route through the unified curated-sampler framework (epic 7114 P3 seam): cancellation, the
        // per-step `eval` (sc-5399 — bounds the lazy graph so a mid-render cancel lands within ~1 model
        // eval), and progress are handled inside `run_flow_sampler`. `euler` reproduces the legacy
        // `FlowMatchEuler::step` loop within the N1 parity tolerance.
        let previews = mlx_gen::preview::PreviewCounter::new(sigmas);
        run_flow_sampler_with_latent_hook(
            sampler_name,
            TimestepConvention::Sigma,
            sigmas,
            init_latents.as_dtype(dtype)?,
            seed,
            cancel,
            &mut on_progress,
            |latents, sigma| {
                mlx_gen_flux2::preview::emit_flux_preview(
                    preview,
                    &previews,
                    sigmas,
                    sigma,
                    latents,
                    latent_h as i32,
                    latent_w as i32,
                    &self.vae,
                );
            },
            predict,
        )
    }

    /// Render one image from pre-encoded conditioning (sc-11030): seed → init latents → denoise →
    /// decode → [`Image`]. The **single render body** shared by
    /// [`LensPipeline::generate_with_progress`] and the registry generator's `Resident` / `Sequential`
    /// paths, so a `Sequential` job — which has already dropped the text encoder before this runs —
    /// produces byte-identical output to `Resident`. The init noise is drawn from the global RNG
    /// reseeded with `seed` here, so hoisting the encode out of the count loop cannot perturb it.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        sampler: Option<&str>,
        scheduler: Option<&str>,
        seed: u64,
        keep: Option<usize>,
        pid_decoder: Option<&dyn mlx_gen::LatentDecoder>,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize),
    ) -> Result<Image> {
        self.render_with_preview(
            encoder_features,
            encoder_mask,
            latent_h,
            latent_w,
            num_steps,
            guidance_scale,
            sampler,
            scheduler,
            seed,
            keep,
            pid_decoder,
            cancel,
            on_step,
            &PreviewSink::default(),
        )
    }

    /// Preview-aware sibling of [`Self::render`].
    #[allow(clippy::too_many_arguments)]
    pub fn render_with_preview(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        sampler: Option<&str>,
        scheduler: Option<&str>,
        seed: u64,
        keep: Option<usize>,
        pid_decoder: Option<&dyn mlx_gen::LatentDecoder>,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize),
        preview: &PreviewSink,
    ) -> Result<Image> {
        let seq_len = (latent_h * latent_w) as i32;
        mlx_rs::random::seed(seed)?;
        let init = mlx_rs::random::normal::<f32>(&[1, seq_len, 128], None, None, None)?;
        let latents = self.denoise_with_sampler_keep_with_preview(
            encoder_features,
            encoder_mask,
            &init,
            latent_h,
            latent_w,
            num_steps,
            guidance_scale,
            sampler,
            scheduler,
            seed,
            keep,
            cancel,
            &mut |cur, _total| on_step(cur),
            preview,
        )?;
        // `vae::decode` returns NHWC [1, H, W, 3] (native) or [1, 4H, 4W, 3] (PiD, 4× SR), both in
        // [-1,1] — `decoded_to_image` handles either.
        let decoded = vae::decode(&self.vae, &latents, latent_h, latent_w, pid_decoder)?;
        decoded_to_image(&decoded)
    }
}

impl LensPipeline {
    /// Text-encode the positive + negative prompts — delegates to the [`LensText`] phase (kept on
    /// `LensPipeline` for the e2e parity gate + the direct struct API).
    ///
    /// `guidance_scale` must be the value the matching denoise call receives: it selects the
    /// conditioning batch (joint `[pos; neg]` vs cond-only) that the denoise lane expects (sc-17616).
    pub fn encode_prompt(
        &self,
        prompt: &str,
        negative_prompt: &str,
        date: &str,
        guidance_scale: f32,
        cancel: Option<&CancelFlag>,
    ) -> Result<(Vec<Array>, Array)> {
        self.text
            .encode_prompt(prompt, negative_prompt, date, guidance_scale, cancel)
    }

    /// The default-sampler denoise — delegates to the [`LensHeavy`] phase (e2e parity gate).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        init_latents: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Array> {
        self.heavy.denoise(
            encoder_features,
            encoder_mask,
            init_latents,
            latent_h,
            latent_w,
            num_steps,
            guidance_scale,
            cancel,
            on_step,
        )
    }

    /// The curated-sampler denoise — delegates to the [`LensHeavy`] phase (e2e parity gate).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_with_sampler(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        init_latents: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        sampler_name: Option<&str>,
        scheduler_name: Option<&str>,
        seed: u64,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Array> {
        self.heavy.denoise_with_sampler(
            encoder_features,
            encoder_mask,
            init_latents,
            latent_h,
            latent_w,
            num_steps,
            guidance_scale,
            sampler_name,
            scheduler_name,
            seed,
            cancel,
            on_step,
        )
    }

    /// Generate a single image (no cancellation / progress). Draws the initial latents from the
    /// global RNG seeded with `opts.seed`. Native-VAE decode (no PiD overlay).
    pub fn generate(&self, opts: &GenerateOptions) -> Result<Image> {
        self.generate_with_progress(opts, None, None, &CancelFlag::default(), &mut |_| {})
    }

    /// Generate a single image, threading a cancel flag and a per-step progress callback
    /// (`on_step(completed_step)`). The registry loops `count` with per-image seeds over this.
    ///
    /// `pid_decoder`: an optional PiD super-resolving decoder (epic 7840, sc-7847) that replaces the
    /// native Flux.2 VAE decode (4× SR). `None` → the byte-exact VAE path. `keep`: the PiD `from_ldm`
    /// early-stop truncation (sc-8048) — `Some(k)` runs the denoise over `sigmas[..k]` so the latent
    /// sits at the capture σ; `None` runs the full schedule (byte-identical clean path).
    pub fn generate_with_progress(
        &self,
        opts: &GenerateOptions,
        pid_decoder: Option<&dyn mlx_gen::LatentDecoder>,
        keep: Option<usize>,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize),
    ) -> Result<Image> {
        if !opts.width.is_multiple_of(VAE_SCALE_FACTOR)
            || !opts.height.is_multiple_of(VAE_SCALE_FACTOR)
        {
            return Err(Error::Msg(format!(
                "lens: height/width must be divisible by {VAE_SCALE_FACTOR} (got {}x{})",
                opts.height, opts.width
            )));
        }
        if opts.num_steps == 0 {
            return Err(Error::Msg("lens: num_steps must be >= 1".into()));
        }
        let latent_h = (opts.height / VAE_SCALE_FACTOR) as usize;
        let latent_w = (opts.width / VAE_SCALE_FACTOR) as usize;

        // Optional prompt refinement via the local reasoner (sc-3176) — before encoding.
        let refined;
        let prompt = if opts.enable_reasoner {
            let reasoner = self.reasoner.as_ref().ok_or_else(|| {
                Error::Msg(
                    "lens: enable_reasoner set but no reasoner attached (call attach_reasoner)"
                        .into(),
                )
            })?;
            refined = reasoner.refine(
                opts.prompt,
                crate::reasoner::DEFAULT_MAX_NEW_TOKENS,
                opts.date,
                opts.reasoner_temperature,
                opts.seed,
                Some(cancel),
            )?;
            &refined
        } else {
            opts.prompt
        };

        let (encoder_features, encoder_mask) = self.text.encode_prompt(
            prompt,
            opts.negative_prompt,
            opts.date,
            opts.guidance_scale,
            Some(cancel),
        )?;

        // One render body (sc-11030): seed → init latents → denoise → decode. Byte-identical to the
        // pre-split inline path — `render` reseeds the global RNG from `opts.seed` immediately before
        // the init-noise draw, so the encode never perturbs it.
        self.heavy.render(
            &encoder_features,
            &encoder_mask,
            latent_h,
            latent_w,
            opts.num_steps,
            opts.guidance_scale,
            opts.sampler,
            opts.scheduler,
            opts.seed,
            keep,
            pid_decoder,
            cancel,
            on_step,
        )
    }

    /// The loaded VAE — delegates to the [`LensHeavy`] phase (e2e parity test decode step).
    pub fn vae(&self) -> &Flux2Vae {
        self.heavy.vae()
    }
}

impl LensHeavy {
    /// The loaded VAE (for the e2e parity test's decode step).
    pub fn vae(&self) -> &Flux2Vae {
        &self.vae
    }

    pub(crate) fn into_vae(self) -> Flux2Vae {
        self.vae
    }

    /// Apply LoRA/LoKr adapters to the DiT (sc-3174) — stacked, mixed, strict (errors on an unmatched
    /// target). A LoRA trained on base `microsoft/Lens` applies to `Lens-Turbo` (same architecture).
    pub fn apply_adapters(&mut self, specs: &[mlx_gen::AdapterSpec]) -> Result<()> {
        crate::adapters::apply_lens_adapters(&mut self.transformer, specs)?;
        Ok(())
    }

    /// Quantize the DiT's linears to Q4/Q8 (sc-3175 — the complement to the encoder MoE-expert quant in
    /// [`LensText::load`](LensText::load)). Call **after** [`apply_adapters`](Self::apply_adapters): the
    /// adapters are forward-time residuals over the now-quantized base, so quantize-after-merge is the
    /// correct order (the registry orchestrates it).
    pub fn quantize_dit(&mut self, quant: Quant) -> Result<()> {
        self.transformer.quantize(quant.bits())
    }
}

/// Render one preview sample (sc-5637) from the **in-progress training adapter** already installed on
/// `transformer`: seeded init latents → flow-match Euler norm-rescaled-CFG denoise → VAE decode →
/// [`Image`]. A stripped [`LensPipeline::denoise`] + decode for the trainer (which holds the raw DiT +
/// VAE, not a `LensPipeline` — the gpt-oss encoder is freed after caching). `encoder_features`/
/// `encoder_mask` are the pre-encoded conditioning, batched to match `guidance_scale` exactly as
/// [`LensText::encode_prompt_windowed`] would (sc-17616): the joint CFG batch (`[2, …]` = positive then
/// empty-negative) when [`needs_joint_cfg`], and cond-only (`[1, …]`) when not. The trainer's
/// `sample_caps` pre-encode applies the same gate over the same `cfg.sample_guidance_scale`, so this
/// lane consumes the batch as given and never re-narrows.
/// `dtype` is the trainer compute dtype. `cancel` is the training request's flag (F-077): a cancel
/// during a preview burst interrupts this render's own denoise mid-loop, not just between prompts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_sample(
    transformer: &LensTransformer,
    vae: &Flux2Vae,
    encoder_features: &[Array],
    encoder_mask: &Array,
    seed: u64,
    edge: u32,
    num_steps: usize,
    guidance_scale: f32,
    dtype: Dtype,
    cancel: &CancelFlag,
) -> Result<Image> {
    let latent = (edge / VAE_SCALE_FACTOR) as usize;
    let seq_len = (latent * latent) as i32;
    let schedule = lens_schedule(num_steps.max(1), latent, latent);
    let timesteps = schedule::timesteps(&schedule);
    mlx_rs::random::seed(seed)?;
    let init = mlx_rs::random::normal::<f32>(&[1, seq_len, 128], None, None, None)?;
    let mut latents = init.as_dtype(dtype)?;
    // At guidance 1.0 the uncond DiT forward is dead weight; run the cond half alone (B=1) and route
    // through `cfg_rescale(cond, cond, 1.0)` (F-020; see `needs_joint_cfg`). The caller pre-encoded the
    // conditioning under the same gate, so it is already cond-only here (sc-17616).
    let joint = needs_joint_cfg(guidance_scale);
    check_conditioning_batch(encoder_features, encoder_mask, joint, guidance_scale)?;
    // sc-17719 — see `text_mask_is_all_valid`. The trainer's preview captions are single unpadded
    // prompts, so this is always the skip path in practice; resolved once, outside the step loop.
    let text_valid = (!text_mask_is_all_valid(encoder_mask)?).then_some(encoder_mask);
    for (i, &sigma) in timesteps.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let noise_pred = if joint {
            // Joint CFG batch: duplicate the latent (cond/uncond share x_t), one DiT call (mirrors
            // `LensPipeline::denoise`).
            let hidden = concatenate_axis(&[&latents, &latents], 0)?;
            let timestep = Array::from_slice(&[sigma, sigma], &[2]).as_dtype(dtype)?;
            let noise = transformer.forward(
                &hidden,
                encoder_features,
                text_valid,
                &timestep,
                1,
                latent,
                latent,
            )?;
            let parts = split(&noise, 2, 0)?;
            cfg_rescale(&parts[0], &parts[1], guidance_scale)?
        } else {
            let timestep = Array::from_slice(&[sigma], &[1]).as_dtype(dtype)?;
            let cond = transformer.forward(
                &latents,
                encoder_features,
                text_valid,
                &timestep,
                1,
                latent,
                latent,
            )?;
            cfg_rescale(&cond, &cond, guidance_scale)?
        };
        latents = schedule.step(&latents, &noise_pred, i)?;
        latents.eval()?;
    }
    let decoded = vae::decode(vae, &latents, latent, latent, None)?; // training preview: native VAE
    decoded_to_image(&decoded)
}

/// Zero-pad each `[B, cur, C]` feature layer along the sequence axis to length `target`.
fn pad_features(features: &[Array], cur: i32, target: i32) -> Result<Vec<Array>> {
    if cur == target {
        return Ok(features.to_vec());
    }
    let pad = target - cur;
    features
        .iter()
        .map(|f| {
            let (b, c) = (f.shape()[0], f.shape()[2]);
            let z = mlx_rs::ops::zeros::<f32>(&[b, pad, c])?.as_dtype(f.dtype())?;
            Ok(concatenate_axis(&[f, &z], 1)?)
        })
        .collect()
}

/// Zero-pad a `[B, cur]` mask along the sequence axis to length `target`.
fn pad_mask(mask: &Array, cur: i32, target: i32) -> Result<Array> {
    if cur == target {
        return Ok(mask.clone());
    }
    let pad = target - cur;
    let b = mask.shape()[0];
    let z = mlx_rs::ops::zeros::<f32>(&[b, pad])?;
    Ok(concatenate_axis(&[mask, &z], 1)?)
}

/// Convert a decoded image `[1, H, W, 3]` (NHWC) in `[-1, 1]` to an RGB8 [`Image`]
/// (`((x·0.5+0.5).clamp(0,1)·255).round()`), matching the reference `_to_pil` quantization.
pub(crate) fn decoded_to_image(decoded: &Array) -> Result<Image> {
    let x = decoded.as_dtype(Dtype::Float32)?;
    let half = Array::from_f32(0.5);
    let x = mlx_rs::ops::add(&mlx_rs::ops::multiply(&x, &half)?, &half)?;
    let x = mlx_rs::ops::clip(&x, (0.0, 1.0))?;
    let x = mlx_rs::ops::round(&mlx_rs::ops::multiply(&x, Array::from_f32(255.0))?, 0)?;
    let sh = x.shape();
    // One image per call: reject B>1 instead of silently keeping only batch 0, and size in usize /
    // flatten via -1 to avoid the u32/i32 product overflow at large resolutions (F-068).
    if sh[0] != 1 {
        return Err(Error::Msg(format!(
            "lens decoded_to_image: expected batch size 1, got {}",
            sh[0]
        )));
    }
    let (h, w, c) = (sh[1] as usize, sh[2] as usize, sh[3] as usize);
    let n = h * w * c;
    let flat = x.reshape(&[-1])?;
    let pixels: Vec<u8> = flat.as_slice::<f32>()[..n]
        .iter()
        .map(|&v| v as u8)
        .collect();
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use mlx_rs::{Array, Dtype};

    use super::{
        assemble_conditioning, check_conditioning_batch, needs_joint_cfg, text_mask_is_all_valid,
    };

    /// A synthetic encode result: `layers` feature layers of `[1, s, c]` filled with `fill`, plus the
    /// all-valid `[1, s]` mask a single unpadded prompt produces.
    fn encoded(s: i32, c: i32, fill: f32) -> (Vec<Array>, Array) {
        let feats = (0..2)
            .map(|_| Array::from_slice(&vec![fill; (s * c) as usize], &[1, s, c]))
            .collect();
        (feats, mlx_rs::ops::ones::<f32>(&[1, s]).unwrap())
    }

    fn values(a: &Array) -> Vec<f32> {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .flatten(None, None)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    /// sc-17616 — the CFG-off gate lives at the encoder: cond-only conditioning, and the ~20 B-param
    /// uncond encode is never run. Against the pre-sc-17616 code (which always stacked `[pos; neg]` and
    /// left the narrowing to each denoise lane) both assertions fail.
    #[test]
    fn cfg_off_conditioning_is_cond_only_and_never_encodes_the_negative() {
        let called = Cell::new(false);
        let (features, mask) =
            assemble_conditioning(encoded(3, 4, 1.5), 1.0, Dtype::Float32, || {
                called.set(true);
                Ok(Some(encoded(3, 4, -2.0)))
            })
            .expect("assemble");

        assert!(
            !called.get(),
            "CFG off must not encode the negative — that 20B forward is dead weight"
        );
        assert_eq!(mask.shape(), &[1, 3], "the mask must follow the features");
        for f in &features {
            assert_eq!(f.shape(), &[1, 3, 4], "CFG off ⇒ cond-only (batch 1)");
            assert!(values(f).iter().all(|&v| v == 1.5), "row 0 is the positive");
        }
    }

    /// The joint path is unchanged: `[pos; neg]` in that order, and the mask is stacked to the SAME
    /// batch as the features.
    #[test]
    fn joint_cfg_stacks_the_negative_below_the_positive() {
        let (features, mask) =
            assemble_conditioning(encoded(3, 4, 1.5), 5.0, Dtype::Float32, || {
                Ok(Some(encoded(3, 4, -2.0)))
            })
            .expect("assemble");

        assert_eq!(mask.shape(), &[2, 3]);
        for f in &features {
            assert_eq!(f.shape(), &[2, 3, 4]);
            let v = values(f);
            let (pos, neg) = v.split_at(12);
            assert!(pos.iter().all(|&x| x == 1.5), "row 0 must be the positive");
            assert!(neg.iter().all(|&x| x == -2.0), "row 1 must be the negative");
        }
    }

    /// A negative LONGER than the positive pads both sides to `S_txt = max(s_pos, s_neg)` — and the
    /// mask is padded alongside the features, marking the positive's pad tail invalid. Narrowing or
    /// padding the features without the mask is exactly what sc-14195 fixed in candle SDXL.
    #[test]
    fn joint_cfg_pads_the_mask_alongside_the_features() {
        let (features, mask) =
            assemble_conditioning(encoded(2, 4, 1.5), 5.0, Dtype::Float32, || {
                Ok(Some(encoded(3, 4, -2.0)))
            })
            .expect("assemble");

        assert_eq!(mask.shape(), &[2, 3], "mask padded to the shared S_txt");
        for f in &features {
            assert_eq!(f.shape(), &[2, 3, 4], "features padded to the shared S_txt");
        }
        assert_eq!(
            values(&mask),
            vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            "the positive's pad tail must be masked invalid, the negative's kept valid"
        );
    }

    /// A caller that encodes at one guidance and denoises at another is the one way the sc-14195 shape
    /// can still reach a lane (the conditioning and the scale are separate arguments, and several
    /// lanes are `pub`). It must be a named error at the stage boundary, both directions, not a raw
    /// MLX shape message from inside the first block's joint attention.
    #[test]
    fn a_conditioning_batch_that_contradicts_the_guidance_is_rejected() {
        let joint = assemble_conditioning(encoded(3, 4, 1.5), 5.0, Dtype::Float32, || {
            Ok(Some(encoded(3, 4, -2.0)))
        })
        .expect("assemble");
        let cond_only = assemble_conditioning(encoded(3, 4, 1.5), 1.0, Dtype::Float32, || Ok(None))
            .expect("assemble");

        // Matched pairs pass.
        check_conditioning_batch(&joint.0, &joint.1, true, 5.0)
            .expect("joint conditioning at CFG on");
        check_conditioning_batch(&cond_only.0, &cond_only.1, false, 1.0)
            .expect("cond-only conditioning at CFG off");

        // Joint conditioning handed to a B=1 lane — the sc-14195 shape.
        let err = check_conditioning_batch(&joint.0, &joint.1, false, 1.0)
            .expect_err("batch-2 conditioning must be rejected by the B=1 lane");
        assert!(
            err.to_string().contains("does not match guidance"),
            "expected a named conditioning/guidance error, got: {err}"
        );

        // …and the mirror: cond-only conditioning handed to the joint lane.
        check_conditioning_batch(&cond_only.0, &cond_only.1, true, 5.0)
            .expect_err("batch-1 conditioning must be rejected by the joint lane");
    }

    /// sc-17719 — the predicate that lets a lane skip an identically-zero attention mask. It must key
    /// on "every position valid", not on the batch: a single zero anywhere is load-bearing padding.
    #[test]
    fn the_attention_mask_is_skippable_exactly_when_every_text_position_is_valid() {
        let ones = mlx_rs::ops::ones::<f32>(&[1, 4]).unwrap();
        assert!(text_mask_is_all_valid(&ones).unwrap());
        // Batch 2, both rows fully valid (equal-length prompts) — still skippable.
        assert!(text_mask_is_all_valid(&mlx_rs::ops::ones::<f32>(&[2, 4]).unwrap()).unwrap());

        // One padded tail position ⇒ the mask is load-bearing and must be kept.
        let padded = Array::from_slice(&[1.0f32, 1.0, 1.0, 0.0], &[1, 4]);
        assert!(!text_mask_is_all_valid(&padded).unwrap());
        // The empty-negative uncond row is all-zero, so a guided empty-negative batch keeps its mask.
        let empty_neg = Array::from_slice(&[1.0f32, 1.0, 0.0, 0.0], &[2, 2]);
        assert!(!text_mask_is_all_valid(&empty_neg).unwrap());
    }

    /// The empty negative stays the unconditional branch (zero features + zero mask), still gated on
    /// CFG being on.
    #[test]
    fn empty_negative_is_the_zero_unconditional_branch() {
        let (features, mask) =
            assemble_conditioning(encoded(2, 4, 1.5), 5.0, Dtype::Float32, || Ok(None))
                .expect("assemble");

        assert_eq!(mask.shape(), &[2, 2]);
        assert_eq!(values(&mask), vec![1.0, 1.0, 0.0, 0.0]);
        for f in &features {
            let v = values(f);
            assert!(v[..8].iter().all(|&x| x == 1.5));
            assert!(v[8..].iter().all(|&x| x == 0.0), "uncond features are zero");
        }
    }

    #[test]
    fn joint_cfg_only_needed_above_unit_guidance() {
        // The turbo default (1.0) skips the dead uncond branch; every other guidance keeps B=2.
        assert!(
            !needs_joint_cfg(1.0),
            "guidance 1.0 must skip the uncond half"
        );
        assert!(
            needs_joint_cfg(5.0),
            "base guidance 5.0 must keep joint CFG"
        );
        assert!(needs_joint_cfg(1.5));
        assert!(needs_joint_cfg(0.0));
        // Not near-equal: a guidance a hair off 1.0 still runs the joint path (the combine no longer
        // reduces to `cond`), so the skip is a strict `== 1.0` gate.
        assert!(needs_joint_cfg(1.0 + f32::EPSILON));
    }
}
