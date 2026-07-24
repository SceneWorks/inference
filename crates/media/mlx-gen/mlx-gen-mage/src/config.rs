//! Mage-Flow shape/hyperparameter constants and the `transformer/config.json` reader.
//!
//! **This module is the single source of truth every other module in the crate reads from.** It is
//! owned by the scaffold story (sc-14037) precisely so the four parallel P1 ports — text encoder
//! (sc-14038), VAE (sc-14039), DiT (sc-14040), watermarked noise (sc-14104) — consume constants
//! from here instead of each transcribing them from the reference, and never contend over the same
//! file.
//!
//! Every value below is transcribed from the published `microsoft/Mage-Flow*` component configs
//! (all six repos ship **byte-identical** `transformer/config.json`, `vae/config.json` and
//! `scheduler/scheduler_config.json` — see `tests/config_conformance.rs`, which pins their SHA-256)
//! or from the vendored reference at `crates/media/mlx-gen/_vendor/mage_flow/`, cited `file:line`.
//!
//! ## The config-strip trap
//!
//! The reference DiT reads **only nine** fields out of `transformer/config.json`
//! (`pipeline.py:729-737` strips everything else into `_meta` before constructing
//! [`MageFlowParams`](../_vendor/mage_flow/models/mage_flow.py)); every other published key —
//! `theta`, `mlp_ratio`, `static_shift`, `depth_single_blocks`, `qkv_bias`, `guidance_embed`, … —
//! is **hardcoded in Python** and silently ignored if the JSON disagrees. Modelling that faithfully
//! matters: a future repo shipping `"theta": 5000` would change nothing in the reference, so a port
//! that *did* read it would diverge. [`MageFlowConfig`] therefore carries exactly the nine consumed
//! fields, the code-hardcoded values live as `const`s here, and
//! [`MageFlowConfig::from_transformer_config_json`] **verifies** the published-but-ignored keys
//! still agree with those consts rather than reading them — so a drifting checkpoint fails loudly
//! instead of being silently misinterpreted.

use mlx_gen::{Error, Result};

// ---------------------------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------------------------

/// Registry family id shared by all six Mage-Flow variants. Every model id is prefixed with it,
/// matching the image-family convention (`flux2_dev`, `krea_2_raw`, `z_image_turbo`, …).
pub const FAMILY: &str = "mage_flow";

// ---------------------------------------------------------------------------------------------
// DiT — the nine fields the reference actually reads (`pipeline.py:729-737`)
// ---------------------------------------------------------------------------------------------

/// Shape/hyperparameters of the Mage-Flow NR-MMDiT — **exactly** the nine
/// `transformer/config.json` fields the reference passes to `MageFlowParams`
/// (`_vendor/mage_flow/models/mage_flow.py:46-56`). Everything else the model needs is a
/// module-level `const` in this file, because that is where the reference keeps it too.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MageFlowConfig {
    /// VAE latent channels entering `img_in` (`Linear(in_channels → hidden_size)`), **128**.
    pub in_channels: i32,
    /// Channels emitted by `proj_out`, **128**. `patch_size == 1`, so `proj_out` is
    /// `Linear(hidden_size → out_channels)`.
    pub out_channels: i32,
    /// Text-conditioning width (the Qwen3-VL LM hidden size), **2560**. Enters the DiT as
    /// `RMSNorm(context_in_dim, eps=1e-6) → Linear(context_in_dim → hidden_size)`
    /// (`mage_flow.py:74-75`).
    pub context_in_dim: i32,
    /// Joint-stream model width, **3072**.
    pub hidden_size: i32,
    /// Attention heads, **24** ⇒ [`head_dim`](Self::head_dim) 128.
    pub num_heads: i32,
    /// Dual-stream block count, **12**. `depth_single_blocks` is 0 — there is no single-stream
    /// tail (see [`DEPTH_SINGLE_BLOCKS`]).
    pub depth: usize,
    /// msrope per-axis head-dim split `(frame, height, width)`, **[16, 56, 56]**; must sum to
    /// [`head_dim`](Self::head_dim) (the reference asserts this at `mage_flow.py:70`).
    pub axes_dim: Vec<i32>,
    /// Gradient checkpointing — training-only, published as `false`.
    pub checkpoint: bool,
    /// Latent patch size, **1**: latents are flattened token-per-latent-cell with no patchify.
    pub patch_size: i32,
}

// ---------------------------------------------------------------------------------------------
// DiT — values hardcoded in the reference's *code*, NOT read from config.json
// ---------------------------------------------------------------------------------------------

/// msrope base frequency. Hardcoded at `mage_flow.py:72`
/// (`MageFlowEmbedRope(theta=10000, …)`), *not* read from the config's `"theta"` key.
pub const ROPE_THETA: f32 = 10_000.0;

/// `scale_rope=True` (`mage_flow.py:72`): the height/width axes use **centred** coordinates
/// `-(L - L/2) … L/2 - 1` rather than `0 … L-1` (`mage_layers.py:194-203`). The frame axis is not
/// centred — it is the segment's position in `img_shapes` (`mage_layers.py:171`, `:192`).
pub const SCALE_ROPE: bool = true;

/// Size of the precomputed positive (and mirrored negative) msrope frequency table, per axis
/// (`mage_layers.py:110-111`). Bounds the packed sequence, not the resolution.
pub const ROPE_TABLE_LEN: i32 = 4096;

/// FFN expansion on **both** streams. Hardcoded (`mage_layers.py:547`, `:557`); the config's
/// `"mlp_ratio"` key is stripped before the model sees it.
pub const MLP_RATIO: f32 = 4.0;

/// The DiT MLP is a **`gelu-approximate` FeedForward, NOT SwiGLU** (`mage_layers.py:547`, `:557`).
/// This corrects the epic's original z-image reuse assumption: z-image uses SwiGLU, Mage does not,
/// on either stream. Kept as a named constant so the DiT port (sc-14040) cannot silently inherit
/// the sibling's activation.
pub const FFN_ACTIVATION: &str = "gelu-approximate";

/// Attention QKV projections carry a bias.
pub const QKV_BIAS: bool = true;

/// No distilled-guidance embedding: Mage uses **real CFG** (a second unconditional forward), which
/// is why `cfg > 1.0` builds a negative branch at all (`pipeline.py:326`, `:535`).
pub const GUIDANCE_EMBED: bool = false;

/// **Zero** single-stream blocks: all [`MageFlowConfig::depth`] blocks are dual-stream (SD3-style
/// joint self-attention with modality-specific norms/projections).
pub const DEPTH_SINGLE_BLOCKS: usize = 0;

/// `eps` for every `LayerNorm(elementwise_affine=False)` / `RMSNorm` in the DiT
/// (`mage_flow.py:74`, `:90`; `mage_layers.py`).
pub const NORM_EPS: f32 = 1e-6;

/// Joint-attention concatenation order — `[text, image]`, `causal=False` (`mage_layers.py:424`,
/// `:490`). Rotary embeddings are applied to the **image** q/k only (`:421-422`), matching the
/// published `apply_text_rotary_emb: false`.
pub const TEXT_STREAM_FIRST: bool = true;

/// Sinusoidal timestep-embedding width feeding `TimestepEmbedding(→ hidden_size)`
/// (`mage_layers.py:89-96`).
pub const FREQUENCY_EMBEDDING_SIZE: i32 = 256;

/// `Timesteps(..., scale=1000, ...)`: the DiT is fed the scheduler **sigma ∈ [0, 1]** directly
/// (`pipeline.py:189`), scaled by this inside the embedder — not a 0..1000 timestep index.
pub const TIMESTEP_SCALE: f32 = 1000.0;

/// `Timesteps(..., max_period=10000, flip_sin_to_cos=True, downscale_freq_shift=0)`.
pub const TIMESTEP_MAX_PERIOD: f32 = 10_000.0;

/// The sinusoidal frequency table is **deliberately rounded to bf16** before use
/// (`mage_layers.py:45`) — the model was trained with that rounding, so an f32 table is a
/// divergence, not an improvement.
pub const TIMESTEP_FREQS_BF16: bool = true;

// ---------------------------------------------------------------------------------------------
// Scheduler — `scheduler/scheduler_config.json` (identical in all six repos)
// ---------------------------------------------------------------------------------------------

/// `FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)`.
pub const NUM_TRAIN_TIMESTEPS: u32 = 1000;

/// Static (resolution-independent) flow-match shift, **6.0**, for *every* variant including
/// 4-step Turbo: `σ ↦ 6σ / (1 + 5σ)` with `use_dynamic_shifting=false` (`pipeline.py:37-50`).
/// There is no distilled Turbo timestep table — Turbo is the same ladder evaluated at N=4.
pub const STATIC_SHIFT: f32 = 6.0;

/// `use_dynamic_shifting` — false, so the shift never depends on the token count.
pub const USE_DYNAMIC_SHIFTING: bool = false;

// ---------------------------------------------------------------------------------------------
// VAE — `vae/config.json` + the reference codec
// ---------------------------------------------------------------------------------------------

/// Mage-VAE latent channels (== [`MageFlowConfig::in_channels`]).
pub const LATENT_CHANNELS: i32 = 128;

/// Mage-VAE spatial downsample factor.
pub const VAE_DOWNSAMPLE_FACTOR: u32 = 16;

/// **There is no latent scale or shift.** Latents feed `img_in = Linear(128 → 3072)` raw
/// (`mage_flow.py:73`, `:109`); `MageVAE.encode`/`decode` normalise nothing
/// (`mage_vae.py:615-633`) and no `scaling_factor`/`shift_factor`/`latents_mean`/`latents_std`
/// exists anywhere in the reference or the published configs. Declared as a constant so the seam
/// is greppable and a port cannot quietly re-introduce a FLUX/SD-style constant.
pub const LATENT_SCALE_SHIFT: Option<(f32, f32)> = None;

/// Published `vae/config.json` value — **and a trap.** The reference pipeline never honours it:
/// `ModelConfig.vae_sample_posterior` defaults to `true` (`mage_flow.py:35`) and `load_from_repo`
/// does not override it, so the edit path *samples* the posterior (`pipeline.py:499`) while the
/// published config says `false`. sc-14039/sc-14048 must reproduce the sampling behaviour, not the
/// config value.
pub const VAE_CONFIG_SAMPLE_POSTERIOR: bool = false;

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

/// Every requested side must be a multiple of this (`_make_divisible_by_16`, `pipeline.py:92`) —
/// it is the VAE's downsample factor, and `patch_size == 1` adds no further stride.
pub const SIZE_MULTIPLE: u32 = VAE_DOWNSAMPLE_FACTOR;

/// Native-resolution range, per side (README / paper): 512–2048, any aspect up to 4:1.
pub const MIN_SIZE: u32 = 512;
/// See [`MIN_SIZE`].
pub const MAX_SIZE: u32 = 2048;

// ---------------------------------------------------------------------------------------------
// Text encoder — `text_encoder/config.json` (Qwen3-VL-4B)
// ---------------------------------------------------------------------------------------------

/// Qwen3-VL **language-model** hyperparameters — the only path text-to-image conditioning uses.
/// The vision tower (`vision_config`) is required for the *edit* conditioning path (sc-14048) and
/// is deliberately absent here.
///
/// Conditioning is the **final (36th) hidden state AFTER the final RMSNorm** — *not* the
/// penultimate layer (that is the z-image convention, and it is wrong for Mage by a full RMSNorm
/// plus one decoder layer; see `_vendor/MAGE_FLOW_GAPS.md` GAP 1, measured bit-exact). The pooled
/// `vec` the reference also computes is discarded by the DiT (`mage_flow.py:116`), so a port needs
/// no pooled text vector at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenVlTextConfig {
    pub hidden_size: i32,
    pub num_layers: usize,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub vocab_size: i32,
    /// Interleaved M-RoPE section split `(t, h, w)`.
    pub mrope_section: [i32; 3],
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
}

impl QwenVlTextConfig {
    /// The published `text_encoder/config.json → text_config` of every `microsoft/Mage-Flow*` repo.
    pub const fn mage_flow() -> Self {
        Self {
            hidden_size: 2560,
            num_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            // Decoupled from `hidden_size / num_attention_heads` (32 × 128 = 4096 ≠ 2560).
            head_dim: 128,
            intermediate_size: 9728,
            vocab_size: 151_936,
            mrope_section: [24, 20, 20],
            attention_bias: false,
            // `tie_word_embeddings: true` and the checkpoint ships **no** `lm_head` tensor — the
            // tied matrix *is* `embed_tokens.weight`. Omitting the lm_head therefore saves zero
            // bytes of resident weights (sc-14105); do not book a memory saving for it.
            tie_word_embeddings: true,
        }
    }
}

/// `rms_norm_eps` of the Qwen3-VL LM.
pub const TE_RMS_NORM_EPS: f32 = 1e-6;
/// `rope_theta` of the Qwen3-VL LM.
pub const TE_ROPE_THETA: f64 = 5_000_000.0;
/// `hidden_act` of the Qwen3-VL LM: SwiGLU with a SiLU gate. (The **DiT** does not share this —
/// see [`FFN_ACTIVATION`].)
pub const TE_HIDDEN_ACT: &str = "silu";

/// System-prompt tokens dropped from the front of the encoded sequence for the **generation**
/// template (`utils.py:55`, `PROMPT_TEMPLATE["mage-flow"].start_idx`).
pub const DROP_IDX_GEN: usize = 34;
/// System-prompt tokens dropped for the **edit** template (`utils.py:64`).
pub const DROP_IDX_EDIT: usize = 64;

/// The verbatim generation chat template (`utils.py:47-56`); `{}` is the user prompt. Encoding
/// this template and dropping [`DROP_IDX_GEN`] tokens is what produces the DiT's `txt` stream.
pub const PROMPT_TEMPLATE_GEN: &str = "<|im_start|>system\nDescribe the image by detailing the \
     color, shape, size, texture, quantity, text, spatial relationships of the objects and \
     background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

/// The verbatim edit chat template (`utils.py:57-65`); `{}` is the edit instruction body built by
/// `_edit_prompt_body` (`pipeline.py:390-393`).
pub const PROMPT_TEMPLATE_EDIT: &str =
    "<|im_start|>system\nDescribe the key features of the input \
     image (color, shape, size, texture, objects, background), then explain how the user's text \
     instruction should alter or modify the image. Generate a new image that meets the user's \
     requirements while maintaining consistency with the original input where appropriate.\
     <|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

/// Long-edge cap applied to a reference image before it reaches the Qwen3-VL **vision** tower on
/// the edit path (`pipeline.py:425`, `:533`). The same image is VAE-encoded at *target*
/// resolution separately — this cap applies to the VL conditioning only.
pub const VL_COND_LONG_EDGE: u32 = 384;

// ---------------------------------------------------------------------------------------------
// Gaussian-Shading watermark (sc-14104)
// ---------------------------------------------------------------------------------------------

/// Default Gaussian-Shading key (`mage_latent.py:19`). The reference also honours a
/// `MAGEFLOW_GS_KEY` env var and a `~/.mageflow/gs_key` keyfile; **neither is ported** — this
/// repository derives no paths and reads no production env side channels (`check-workspace.py`
/// epic-13657 guardrail), so sc-14104 must surface the key through `LoadSpec`/the request instead.
pub const GS_DEFAULT_KEY: u64 = 20_260_720;

/// Watermark payload (`mage_latent.py:10`), SHA-256-expanded to [`GS_MESSAGE_BITS`].
pub const GS_PAYLOAD: &str = "MageFlow";

/// Watermark message length in bits (`mage_latent.py:20`).
pub const GS_MESSAGE_BITS: usize = 256;

// ---------------------------------------------------------------------------------------------
// Config reader
// ---------------------------------------------------------------------------------------------

/// The published-but-code-hardcoded keys this crate verifies rather than reads, as
/// `(json key, expected literal)` pairs rendered for the mismatch message.
const IGNORED_BUT_PINNED: &[&str] = &[
    "theta",
    "mlp_ratio",
    "static_shift",
    "depth_single_blocks",
    "qkv_bias",
    "guidance_embed",
];

impl MageFlowConfig {
    /// The production config shipped by all six `microsoft/Mage-Flow*` repos.
    pub fn mage_flow() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            context_in_dim: 2560,
            hidden_size: 3072,
            num_heads: 24,
            depth: 12,
            axes_dim: vec![16, 56, 56],
            checkpoint: false,
            patch_size: 1,
        }
    }

    /// Per-head attention width, `hidden_size / num_heads` (128 in production).
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_heads
    }

    /// Read the nine consumed fields out of a `transformer/config.json` body.
    ///
    /// Fields are **required**, never defaulted: a silently-defaulted `depth` or `axes_dim` would
    /// produce a plausible-looking model with the wrong geometry. In addition, the
    /// published-but-ignored keys in `IGNORED_BUT_PINNED` are *verified* against this module's
    /// constants — the reference hardcodes them in code, so a checkpoint that disagrees would be
    /// silently misinterpreted rather than rejected.
    pub fn from_transformer_config_json(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            Error::Msg(format!(
                "mage_flow: transformer/config.json is invalid: {e}"
            ))
        })?;

        let cfg = Self {
            in_channels: req_i64(&v, "in_channels")? as i32,
            out_channels: req_i64(&v, "out_channels")? as i32,
            context_in_dim: req_i64(&v, "context_in_dim")? as i32,
            hidden_size: req_i64(&v, "hidden_size")? as i32,
            num_heads: req_i64(&v, "num_heads")? as i32,
            depth: usize::try_from(req_i64(&v, "depth")?).map_err(|_| {
                Error::Msg("mage_flow: transformer/config.json depth is negative".into())
            })?,
            axes_dim: req_axes_dim(&v)?,
            checkpoint: req_bool(&v, "checkpoint")?,
            patch_size: req_i64(&v, "patch_size")? as i32,
        };
        cfg.validate()?;
        verify_hardcoded(&v)?;
        Ok(cfg)
    }

    /// The invariants the reference asserts or relies on: positive dims, an integral head split,
    /// `sum(axes_dim) == head_dim` (`mage_flow.py:70`), and a 3-axis msrope.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("in_channels", self.in_channels),
            ("out_channels", self.out_channels),
            ("context_in_dim", self.context_in_dim),
            ("hidden_size", self.hidden_size),
            ("num_heads", self.num_heads),
            ("patch_size", self.patch_size),
        ] {
            if value <= 0 {
                return Err(Error::Msg(format!(
                    "mage_flow: transformer config {name} must be > 0 (got {value})"
                )));
            }
        }
        if self.depth == 0 {
            return Err(Error::Msg(
                "mage_flow: transformer config depth must be > 0".into(),
            ));
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(Error::Msg(format!(
                "mage_flow: hidden_size {} is not divisible by num_heads {}",
                self.hidden_size, self.num_heads
            )));
        }
        if self.axes_dim.len() != 3 {
            return Err(Error::Msg(format!(
                "mage_flow: axes_dim must have 3 entries (frame, height, width); got {:?}",
                self.axes_dim
            )));
        }
        let sum: i32 = self.axes_dim.iter().sum();
        if sum != self.head_dim() {
            return Err(Error::Msg(format!(
                "mage_flow: axes_dim {:?} sums to {sum} but head_dim is {} \
                 (the reference asserts sum(axes_dim) == attention_head_dim)",
                self.axes_dim,
                self.head_dim()
            )));
        }
        Ok(())
    }
}

fn req<'a>(v: &'a serde_json::Value, key: &str) -> Result<&'a serde_json::Value> {
    v.get(key).ok_or_else(|| {
        Error::Msg(format!(
            "mage_flow: transformer/config.json is missing required key '{key}'"
        ))
    })
}

fn req_i64(v: &serde_json::Value, key: &str) -> Result<i64> {
    req(v, key)?.as_i64().ok_or_else(|| {
        Error::Msg(format!(
            "mage_flow: transformer/config.json '{key}' is not an integer"
        ))
    })
}

fn req_bool(v: &serde_json::Value, key: &str) -> Result<bool> {
    req(v, key)?.as_bool().ok_or_else(|| {
        Error::Msg(format!(
            "mage_flow: transformer/config.json '{key}' is not a boolean"
        ))
    })
}

fn req_axes_dim(v: &serde_json::Value) -> Result<Vec<i32>> {
    let arr = req(v, "axes_dim")?.as_array().ok_or_else(|| {
        Error::Msg("mage_flow: transformer/config.json 'axes_dim' is not an array".into())
    })?;
    arr.iter()
        .map(|entry| {
            entry
                .as_i64()
                .map(|n| n as i32)
                .ok_or_else(|| Error::Msg("mage_flow: axes_dim entries must be integers".into()))
        })
        .collect()
}

/// Reject a checkpoint whose published values for the code-hardcoded keys disagree with this
/// crate's constants. Absent keys are fine (they are not part of the consumed surface); a
/// *present and different* value is a real divergence the reference would silently ignore.
fn verify_hardcoded(v: &serde_json::Value) -> Result<()> {
    let mismatch = |key: &str, got: String, want: String| {
        Error::Msg(format!(
            "mage_flow: transformer/config.json '{key}' is {got}, but this port hardcodes {want} \
             (the reference hardcodes it in code and ignores the config value, so a differing \
             checkpoint would be silently misinterpreted); expected pinned keys: {IGNORED_BUT_PINNED:?}"
        ))
    };
    if let Some(theta) = v.get("theta").and_then(serde_json::Value::as_f64) {
        if theta as f32 != ROPE_THETA {
            return Err(mismatch("theta", theta.to_string(), ROPE_THETA.to_string()));
        }
    }
    if let Some(ratio) = v.get("mlp_ratio").and_then(serde_json::Value::as_f64) {
        if ratio as f32 != MLP_RATIO {
            return Err(mismatch(
                "mlp_ratio",
                ratio.to_string(),
                MLP_RATIO.to_string(),
            ));
        }
    }
    if let Some(shift) = v.get("static_shift").and_then(serde_json::Value::as_f64) {
        if shift as f32 != STATIC_SHIFT {
            return Err(mismatch(
                "static_shift",
                shift.to_string(),
                STATIC_SHIFT.to_string(),
            ));
        }
    }
    if let Some(n) = v
        .get("depth_single_blocks")
        .and_then(serde_json::Value::as_i64)
    {
        if n as usize != DEPTH_SINGLE_BLOCKS {
            return Err(mismatch(
                "depth_single_blocks",
                n.to_string(),
                DEPTH_SINGLE_BLOCKS.to_string(),
            ));
        }
    }
    if let Some(b) = v.get("qkv_bias").and_then(serde_json::Value::as_bool) {
        if b != QKV_BIAS {
            return Err(mismatch("qkv_bias", b.to_string(), QKV_BIAS.to_string()));
        }
    }
    if let Some(b) = v.get("guidance_embed").and_then(serde_json::Value::as_bool) {
        if b != GUIDANCE_EMBED {
            return Err(mismatch(
                "guidance_embed",
                b.to_string(),
                GUIDANCE_EMBED.to_string(),
            ));
        }
    }
    Ok(())
}
