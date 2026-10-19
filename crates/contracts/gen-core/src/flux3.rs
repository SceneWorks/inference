//! Backend-neutral contract for the **FLUX 3 Image Klein** family (epic sc-26233; S1 = sc-26234,
//! the distilled text-to-image route): frozen upstream identity, the release package layout and
//! its `config.json`, the Qwen3-VL text window, the flow-map schedule, the packed-token id and RoPE
//! tables, and the generation control surface.
//!
//! Everything here is host-side (no tensors), so the MLX provider (`mlx-gen-flux3`), a Candle twin
//! (sc-26235) and the SceneWorks worker read one definition. The tensor work lives in the backend
//! crates.
//!
//! # Frozen upstream
//!
//! * Inference code: GitHub `black-forest-labs/flux-3-image-klein` @ [`UPSTREAM_CODE_REVISION`]
//!   (`src/flux3/{dit,autoencoder,text_encoder,sampling,pipeline}.py`). **The executable source
//!   is the authority** wherever it and the package's `config.json` disagree: the source never
//!   reads `config.json` (its widths are module constants), so the file is descriptive. The one
//!   known disagreement is the text window: `text_encoder.py` clamps the prompt at
//!   [`TEXT_MAX_LENGTH`] = 4096 tokens while `config.json` says `max_tokens: 8192`;
//!   [`TextEncoderConfig::window_max_length`] applies the source's clamp.
//! * Distilled package: Hugging Face `black-forest-labs/flux-3-image-klein` @
//!   [`DISTILLED_WEIGHTS_REVISION`] — `config.json` + `model.safetensors` (the 5.6B flow-map DiT,
//!   bf16), `ae.safetensors` (the autoencoder, f32) and `text_encoder/` (the Qwen3-VL-4B-Instruct
//!   language model, bf16). The bundled `text_encoder/` shards are byte-identical to
//!   [`TEXT_ENCODER_UPSTREAM_REPO`] @ [`TEXT_ENCODER_UPSTREAM_REVISION`] (same LFS sha256 for both
//!   shards, same tokenizer and config), which is why that snapshot may be provisioned as the
//!   `text_encoder` component and why the licence row is Qwen's.
//! * The base checkpoint (`flux-3-image-klein-base`, sc-26239) is pinned separately below and
//!   samples with CFG50/3.5. References and boxes are sc-26236/7's. The prompt upsampler
//!   (`flux-3-image-klein-pup`, sc-26238) is the [`pup`] submodule: a separate, optional
//!   package named by the [`pup::UPSAMPLER_COMPONENT`] component.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::generator::GenerationRequest;
use crate::runtime::{LoadSpec, WeightsSource};
use crate::{Error, Result};

pub mod adapters;
pub mod affine;
pub mod artifacts;
pub mod conversion;
pub mod image_input;
pub mod layouts;
pub mod native_convert;
pub mod packed_gemm;
pub mod phase_resource;
pub mod pup;
pub mod quantization;
pub mod reference_workspace;
pub mod references;
pub mod safetensors_stream;
pub mod training;

// ---------------------------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------------------------

/// GitHub `black-forest-labs/flux-3-image-klein` commit the port mirrors.
pub const UPSTREAM_CODE_REPO: &str = "black-forest-labs/flux-3-image-klein";
/// The frozen inference-code commit.
pub const UPSTREAM_CODE_REVISION: &str = "45e57c9fb06f9751cdced29a65c5d2171003be05";
/// Hugging Face repository of the distilled package (DiT + autoencoder + text encoder).
pub const DISTILLED_WEIGHTS_REPO: &str = "black-forest-labs/flux-3-image-klein";
/// Pinned revision of the distilled package.
pub const DISTILLED_WEIGHTS_REVISION: &str = "8454017f9ebaada30c998fccd9f67a80c7580f68";
/// sha256 of the pinned package's `model.safetensors` (the HF LFS object; also recorded in the
/// package's own `manifest.json`).
pub const DISTILLED_DIT_SHA256: &str =
    "c3e9813cdd049c831c836b2e9db61150e49a93fc1a4c63b256494642fee0bd7c";
/// sha256 of the pinned package's `ae.safetensors`.
pub const DISTILLED_AUTOENCODER_SHA256: &str =
    "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3";
/// The upstream repository the bundled `text_encoder/` is a byte-identical copy of.
pub const TEXT_ENCODER_UPSTREAM_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
/// Its revision: the pinned package's `text_encoder/model-0000N-of-00002.safetensors` carry the
/// same LFS sha256 as this revision's shards (`30a01a05…3339a9` and `046296a2…002a6`), and
/// every non-LFS file has the same git blob oid as this revision's tree (`config.json`
/// `22be8bb7…`, `tokenizer.json` `c6cc1014…`, `tokenizer_config.json` `d3d37632…`,
/// `chat_template.json` `e49e75bd…`, `merges.txt`, `vocab.json`, `generation_config.json`,
/// `model.safetensors.index.json`, both preprocessor configs) — evidence in
/// `docs/licensing/sc-26234-flux-kommunity-licence-evidence.md`.
pub const TEXT_ENCODER_UPSTREAM_REVISION: &str = "ebb281ec70b05090aa6165b016eac8ec08e71b17";

/// Registry id of the distilled text-to-image route (the SceneWorks worker's `payload.model`).
pub const DISTILLED_MODEL_ID: &str = "flux3_klein";
/// Registry id of the CFG base checkpoint.
pub const BASE_MODEL_ID: &str = "flux3_klein_base";
/// Frozen base package (same bundled encoder and autoencoder identities as distilled).
pub const BASE_WEIGHTS_REPO: &str = "black-forest-labs/flux-3-image-klein-base";
pub const BASE_WEIGHTS_REVISION: &str = "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4";
/// Pinned Hub API LFS SHA256 (metadata identity, not a local-download attestation).
pub const BASE_DIT_SHA256: &str =
    "5c921759dc03663dbb67df11d8e3d317d2d5b893b31b969c95dc77ebf861b19c";
/// Same LFS object as distilled; the fourteen encoder files also have identical git blob ids.
pub const BASE_AUTOENCODER_SHA256: &str = DISTILLED_AUTOENCODER_SHA256;
/// Frozen `pipeline.MODELS["base"]` sampling defaults.
pub const BASE_DEFAULT_STEPS: u32 = 50;
/// Frozen base CFG scale.
pub const BASE_DEFAULT_GUIDANCE: f32 = 3.5;
/// Descriptor family shared by every FLUX 3 Image Klein route and backend.
pub const FAMILY: &str = "flux3";

/// `LoadSpec::components` key carrying the text-encoder directory. Optional: when absent the
/// package's own `text_encoder/` (`config.json` `text_encoder.path`) is read.
pub const TEXT_ENCODER_COMPONENT: &str = "text_encoder";
/// `LoadSpec::components` key carrying the autoencoder file (or a directory holding it).
/// Optional: when absent the package's own `ae.safetensors` (`config.json` `vae.path`) is read.
pub const AUTOENCODER_COMPONENT: &str = "autoencoder";
/// The package description file.
pub const CONFIG_FILE: &str = "config.json";
/// The DiT weights file (upstream `checkpoint.safetensors_file`).
pub const DIT_WEIGHTS_FILE: &str = "model.safetensors";
/// Files the text-encoder resource must carry besides its safetensors.
pub const TEXT_ENCODER_REQUIRED_FILES: [&str; 3] =
    ["config.json", "tokenizer.json", "tokenizer_config.json"];

// ---------------------------------------------------------------------------------------------
// Source constants (`dit.py`, `autoencoder.py`, `text_encoder.py`, `sampling.py`, `pipeline.py`)
// ---------------------------------------------------------------------------------------------

/// Pixels per packed latent token on each side (`autoencoder.DOWNSAMPLING_RATIO`: 8× VAE × 2×2
/// patchify). Output sides must be multiples of it.
pub const DOWNSAMPLING_RATIO: u32 = 16;
/// Smallest output side (`pipeline.MIN_SIDE`).
pub const MIN_SIDE: u32 = 64;
/// Largest output side (`pipeline.MAX_SIDE`).
pub const MAX_SIDE: u32 = 4096;
/// Default output size (`sampler.size` / `pipeline.default_size`).
pub const DEFAULT_SIZE: u32 = 1024;
/// The distilled checkpoint's step count (`pipeline.MODELS["distilled"]["steps"]`).
pub const DEFAULT_STEPS: u32 = 16;
/// The schedule shape parameter (`Pipeline.generate(alpha=6.93)`), fixed upstream.
pub const DEFAULT_ALPHA: f32 = 6.93;
/// Image token ids scale their (row, col) axes by this (`pipeline.IMAGE_ROPE_SCALE`).
pub const IMAGE_ROPE_SCALE: f32 = 0.5;
/// The one integrator of the flow-map checkpoint (`sampling.denoise_flow_map`): an Euler step per
/// schedule interval, with the DiT told the target time of the jump.
pub const SAMPLER: &str = "euler";
/// Every sampler name the distilled route advertises.
pub const SAMPLERS: [&str; 1] = [SAMPLER];
/// The 1-based hidden-state indices (HF `output_hidden_states` numbering) stacked into the text
/// conditioning (`text_encoder.OUTPUT_LAYERS`).
pub const TEXT_OUTPUT_LAYERS: [usize; 8] = [4, 8, 12, 16, 20, 24, 28, 32];
/// The source's prompt clamp (`text_encoder.MAX_LENGTH`).
pub const TEXT_MAX_LENGTH: usize = 4096;
/// The text window is padded up to a multiple of this (`text_encoder.PAD_MULTIPLE`).
pub const TEXT_PAD_MULTIPLE: usize = 80;
/// Position id upstream assigns every pad token (`positions.masked_fill(mask == 0, 1)`).
pub const TEXT_PAD_POSITION: i32 = 1;
/// The Qwen3-VL chat template around a single user turn with the generation prompt
/// (`apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)`
/// of the pinned `chat_template.json`, which adds no system turn).
pub const CHAT_PREFIX: &str = "<|im_start|>user\n";
/// See [`CHAT_PREFIX`].
pub const CHAT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// The pad token the pinned tokenizer names (`tokenizer_config.json` `pad_token`), used by the
/// encoder for the window's trailing pads.
pub const PAD_TOKEN: &str = "<|endoftext|>";
/// Timestep embedding width (`dit.timestep_embedding(dim=256)`).
pub const TIMESTEP_EMBED_DIM: usize = 256;
/// Timestep embedding period (`dit.timestep_embedding(max_period=10000)`).
pub const TIMESTEP_MAX_PERIOD: f64 = 10_000.0;
/// The DiT's `MODES`, sorted — the order tokens take inside one sample of the single stream.
pub const MODES: [&str; 3] = ["image", "image_cond", "text"];

/// The full prompt text the encoder tokenizes.
pub fn chat_text(prompt: &str) -> String {
    format!("{CHAT_PREFIX}{prompt}{CHAT_SUFFIX}")
}

// ---------------------------------------------------------------------------------------------
// config.json
// ---------------------------------------------------------------------------------------------

/// `config.json` `dit` — the DiT widths (`dit.py` module constants of the release).
#[derive(Clone, Debug, PartialEq)]
pub struct DitConfig {
    pub hidden: usize,
    pub heads: usize,
    /// Per-mode early blocks.
    pub depth: usize,
    /// Shared single-stream blocks.
    pub depth_single: usize,
    pub mlp_ratio: f64,
    /// RoPE dims per id axis (sum = head dim).
    pub axes_dim: Vec<usize>,
    pub theta: f64,
    pub image_in_channels: usize,
    pub image_cond_in_channels: usize,
    pub text_in_channels: usize,
    pub out_modes: Vec<String>,
    pub vec_in_dim: Option<usize>,
    pub image_rope_scale: f64,
    /// `true`: the checkpoint carries `ref_time_in` and samples without CFG (the distilled
    /// release); `false`: a CFG checkpoint (the base release, sc-26239).
    pub flow_map: bool,
}

impl DitConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    pub fn mlp_hidden(&self) -> usize {
        (self.hidden as f64 * self.mlp_ratio) as usize
    }
}

/// `config.json` `text_encoder`.
#[derive(Clone, Debug, PartialEq)]
pub struct TextEncoderConfig {
    /// Package-relative directory of the encoder snapshot.
    pub path: String,
    pub tokenizer: String,
    /// 1-based hidden-state indices stacked into the conditioning.
    pub layers: Vec<usize>,
    pub pad_multiple: usize,
    /// As written in the file (8192 in the release). Not the window's clamp — see
    /// [`Self::window_max_length`].
    pub max_tokens: usize,
    pub dtype: String,
}

impl TextEncoderConfig {
    /// The prompt clamp the frozen source applies: `min(max_tokens, TEXT_MAX_LENGTH)`. The source's
    /// `text_encoder.py` hard-codes 4096 and never reads `config.json`, so the file's 8192 is not
    /// what the release renders with.
    pub fn window_max_length(&self) -> usize {
        self.max_tokens.min(TEXT_MAX_LENGTH)
    }
}

/// `config.json` `vae`.
#[derive(Clone, Debug, PartialEq)]
pub struct VaeConfig {
    /// Package-relative autoencoder file.
    pub path: String,
    pub dtype: String,
}

/// `config.json` `sampler`.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerConfig {
    pub steps: u32,
    /// `None`: a flow map, sampled without CFG.
    pub guidance: Option<f64>,
    pub alpha: f64,
    pub size: [u32; 2],
}

/// The parsed package `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct Flux3Config {
    pub format_version: u64,
    pub model: String,
    /// `None` for the bf16 package; a quantized variant names its format (sc-26241).
    pub quantization: Option<String>,
    pub torch_dtype: String,
    pub dit: DitConfig,
    pub text_encoder: TextEncoderConfig,
    pub vae: VaeConfig,
    pub sampler: SamplerConfig,
}

fn field<'a>(v: &'a serde_json::Value, key: &str, path: &str) -> Result<&'a serde_json::Value> {
    v.get(key)
        .ok_or_else(|| Error::Msg(format!("config.json: missing `{path}{key}`")))
}

fn usize_of(v: &serde_json::Value, path: &str) -> Result<usize> {
    v.as_u64().map(|n| n as usize).ok_or_else(|| {
        Error::Msg(format!(
            "config.json: `{path}` is not a non-negative integer"
        ))
    })
}

fn f64_of(v: &serde_json::Value, path: &str) -> Result<f64> {
    v.as_f64()
        .filter(|f| f.is_finite())
        .ok_or_else(|| Error::Msg(format!("config.json: `{path}` is not a finite number")))
}

fn string_of(v: &serde_json::Value, path: &str) -> Result<String> {
    v.as_str()
        .map(str::to_owned)
        .ok_or_else(|| Error::Msg(format!("config.json: `{path}` is not a string")))
}

fn usize_list(v: &serde_json::Value, path: &str) -> Result<Vec<usize>> {
    v.as_array()
        .ok_or_else(|| Error::Msg(format!("config.json: `{path}` is not a list")))?
        .iter()
        .map(|e| usize_of(e, path))
        .collect()
}

impl Flux3Config {
    /// Read `dir/config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join(CONFIG_FILE);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Msg(format!("flux3: read {}: {e}", path.display())))?;
        Self::parse(&text).map_err(|e| Error::Msg(format!("flux3: {}: {e}", path.display())))
    }

    /// Parse the JSON text. Every key the native graph depends on is required; nothing is
    /// defaulted, so a package of another format version fails here by key name.
    pub fn parse(text: &str) -> Result<Self> {
        let root: serde_json::Value =
            serde_json::from_str(text).map_err(|e| Error::Msg(format!("config.json: {e}")))?;
        let dit = field(&root, "dit", "")?;
        let in_channels = field(dit, "in_channels", "dit.")?;
        let te = field(&root, "text_encoder", "")?;
        let vae = field(&root, "vae", "")?;
        let sampler = field(&root, "sampler", "")?;
        let size = field(sampler, "size", "sampler.")?
            .as_array()
            .filter(|a| a.len() == 2)
            .ok_or_else(|| Error::Msg("config.json: `sampler.size` is not [w, h]".into()))?;
        let guidance = match field(sampler, "guidance", "sampler.")? {
            serde_json::Value::Null => None,
            v => Some(f64_of(v, "sampler.guidance")?),
        };
        let quantization = match field(&root, "quantization", "")? {
            serde_json::Value::Null => None,
            v if v.is_object() => artifacts::NativeDitStorage::from_marker(v)?
                .config_marker_name()
                .map(str::to_owned),
            v => Some(string_of(v, "quantization")?),
        };
        let vec_in_dim = match field(dit, "vec_in_dim", "dit.")? {
            serde_json::Value::Null => None,
            v => Some(usize_of(v, "dit.vec_in_dim")?),
        };
        Ok(Self {
            format_version: field(&root, "format_version", "")?
                .as_u64()
                .ok_or_else(|| {
                    Error::Msg("config.json: `format_version` is not an integer".into())
                })?,
            model: string_of(field(&root, "model", "")?, "model")?,
            quantization,
            torch_dtype: string_of(field(&root, "torch_dtype", "")?, "torch_dtype")?,
            dit: DitConfig {
                hidden: usize_of(field(dit, "hidden", "dit.")?, "dit.hidden")?,
                heads: usize_of(field(dit, "heads", "dit.")?, "dit.heads")?,
                depth: usize_of(field(dit, "depth", "dit.")?, "dit.depth")?,
                depth_single: usize_of(field(dit, "depth_single", "dit.")?, "dit.depth_single")?,
                mlp_ratio: f64_of(field(dit, "mlp_ratio", "dit.")?, "dit.mlp_ratio")?,
                axes_dim: usize_list(field(dit, "axes_dim", "dit.")?, "dit.axes_dim")?,
                theta: f64_of(field(dit, "theta", "dit.")?, "dit.theta")?,
                image_in_channels: usize_of(
                    field(in_channels, "image", "dit.in_channels.")?,
                    "dit.in_channels.image",
                )?,
                image_cond_in_channels: usize_of(
                    field(in_channels, "image_cond", "dit.in_channels.")?,
                    "dit.in_channels.image_cond",
                )?,
                text_in_channels: usize_of(
                    field(in_channels, "text", "dit.in_channels.")?,
                    "dit.in_channels.text",
                )?,
                out_modes: field(dit, "out_modes", "dit.")?
                    .as_array()
                    .ok_or_else(|| Error::Msg("config.json: `dit.out_modes` is not a list".into()))?
                    .iter()
                    .map(|m| string_of(m, "dit.out_modes"))
                    .collect::<Result<_>>()?,
                vec_in_dim,
                image_rope_scale: f64_of(
                    field(dit, "image_rope_scale", "dit.")?,
                    "dit.image_rope_scale",
                )?,
                flow_map: field(dit, "flow_map", "dit.")?.as_bool().ok_or_else(|| {
                    Error::Msg("config.json: `dit.flow_map` is not a bool".into())
                })?,
            },
            text_encoder: TextEncoderConfig {
                path: string_of(field(te, "path", "text_encoder.")?, "text_encoder.path")?,
                tokenizer: string_of(
                    field(te, "tokenizer", "text_encoder.")?,
                    "text_encoder.tokenizer",
                )?,
                layers: usize_list(field(te, "layers", "text_encoder.")?, "text_encoder.layers")?,
                pad_multiple: usize_of(
                    field(te, "pad_multiple", "text_encoder.")?,
                    "text_encoder.pad_multiple",
                )?,
                max_tokens: usize_of(
                    field(te, "max_tokens", "text_encoder.")?,
                    "text_encoder.max_tokens",
                )?,
                dtype: string_of(field(te, "dtype", "text_encoder.")?, "text_encoder.dtype")?,
            },
            vae: VaeConfig {
                path: string_of(field(vae, "path", "vae.")?, "vae.path")?,
                dtype: string_of(field(vae, "dtype", "vae.")?, "vae.dtype")?,
            },
            sampler: SamplerConfig {
                steps: usize_of(field(sampler, "steps", "sampler.")?, "sampler.steps")? as u32,
                guidance,
                alpha: f64_of(field(sampler, "alpha", "sampler.")?, "sampler.alpha")?,
                size: [
                    usize_of(&size[0], "sampler.size")? as u32,
                    usize_of(&size[1], "sampler.size")? as u32,
                ],
            },
        })
    }

    /// Refuse any package the native graph does not implement, naming the key — never a silent
    /// mis-render. The supported set is the released bf16 architecture's switches; every width
    /// and depth is read from the file.
    pub fn validate_supported(&self) -> Result<()> {
        self.validate_graph(false)
    }

    /// Validate graph/sampler semantics AFTER a caller has preflighted the exact
    /// native artifact/backend/tier with `artifacts::NativeDitStorage`. Legacy
    /// dense-only callers keep `validate_supported` and cannot misread packed codes.
    pub fn validate_native_storage_graph(&self) -> Result<()> {
        self.validate_graph(true)
    }

    fn validate_graph(&self, allow_native_storage: bool) -> Result<()> {
        let d = &self.dit;
        let checks: [(&str, bool, String); 14] = [
            (
                "format_version",
                self.format_version == 1,
                self.format_version.to_string(),
            ),
            (
                // Packed codes are accepted ONLY through the explicit-storage
                // graph entry point; the historical dense entry point still refuses them.
                "quantization",
                self.quantization.is_none()
                    || (allow_native_storage
                        && matches!(
                            self.quantization.as_deref(),
                            Some("fp8r" | "nvfp4" | "mlx-affine" | "ggml-q4_1" | "ggml-q8_0")
                        )),
                format!("{:?}", self.quantization),
            ),
            (
                "torch_dtype",
                self.torch_dtype == "bfloat16",
                self.torch_dtype.clone(),
            ),
            (
                "dit.hidden",
                d.heads > 0 && d.hidden > 0 && d.hidden.is_multiple_of(d.heads),
                d.hidden.to_string(),
            ),
            (
                "dit.axes_dim",
                d.axes_dim.len() == 4
                    && d.axes_dim.iter().all(|&a| a > 0 && a.is_multiple_of(2))
                    && d.axes_dim.iter().sum::<usize>() == d.head_dim(),
                format!("{:?}", d.axes_dim),
            ),
            (
                "dit.mlp_ratio",
                d.mlp_ratio.is_finite() && d.mlp_ratio > 0.0 && d.mlp_hidden() > 0,
                d.mlp_ratio.to_string(),
            ),
            (
                "dit.depth",
                d.depth > 0 && d.depth_single > 0,
                format!("{}/{}", d.depth, d.depth_single),
            ),
            (
                "dit.out_modes",
                d.out_modes == ["image"],
                format!("{:?}", d.out_modes),
            ),
            (
                "dit.vec_in_dim",
                d.vec_in_dim.is_none(),
                format!("{:?}", d.vec_in_dim),
            ),
            (
                "dit.theta",
                d.theta.is_finite() && d.theta > 0.0,
                d.theta.to_string(),
            ),
            (
                "dit.image_rope_scale",
                d.image_rope_scale.is_finite() && d.image_rope_scale > 0.0,
                d.image_rope_scale.to_string(),
            ),
            (
                "text_encoder.layers",
                !self.text_encoder.layers.is_empty()
                    && self.text_encoder.layers.windows(2).all(|w| w[0] < w[1])
                    && self.text_encoder.layers[0] > 0,
                format!("{:?}", self.text_encoder.layers),
            ),
            (
                "text_encoder.pad_multiple",
                self.text_encoder.pad_multiple > 0 && self.text_encoder.window_max_length() > 0,
                self.text_encoder.pad_multiple.to_string(),
            ),
            (
                "sampler.steps",
                self.sampler.steps > 0
                    && self.sampler.alpha.is_finite()
                    && self.sampler.alpha > 0.0,
                self.sampler.steps.to_string(),
            ),
        ];
        for (key, ok, value) in checks {
            if !ok {
                return Err(Error::Unsupported(format!(
                    "flux3: config.json `{key}` = {value} is not a package the native graph \
                     implements"
                )));
            }
        }
        // A flow-map DiT samples without CFG and vice versa (`Pipeline.__init__`'s own check).
        if self.dit.flow_map != self.sampler.guidance.is_none() {
            return Err(Error::Unsupported(format!(
                "flux3: config.json `dit.flow_map` = {} but `sampler.guidance` = {:?}; a flow map \
                 samples without CFG and a CFG checkpoint needs a guidance",
                self.dit.flow_map, self.sampler.guidance
            )));
        }
        // The base checkpoint's defaults are the frozen source's `pipeline.MODELS["base"]`
        // (50 steps, CFG 3.5), not whatever a package's descriptive config.json claims. A config
        // that disagrees is not the pinned release, so refuse it rather than pick either value.
        if !self.dit.flow_map
            && (self.sampler.steps != BASE_DEFAULT_STEPS
                || self.sampler.guidance != Some(f64::from(BASE_DEFAULT_GUIDANCE)))
        {
            return Err(Error::Unsupported(format!(
                "flux3: base config.json sampler = {} steps / guidance {:?}, but the frozen base \
                 defaults are {BASE_DEFAULT_STEPS} steps / guidance {BASE_DEFAULT_GUIDANCE}; this is \
                 not the pinned base package",
                self.sampler.steps, self.sampler.guidance
            )));
        }
        Ok(())
    }

    /// The text conditioning width one token row carries: `layers × encoder hidden size`. The
    /// encoder's width is only known from its own snapshot, so the DiT's `in_channels.text` is
    /// checked against it at load.
    pub fn text_in_channels_for(&self, encoder_hidden: usize) -> usize {
        self.text_encoder.layers.len() * encoder_hidden
    }
}

// ---------------------------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------------------------

/// The resolved, existence-checked resources of the distilled text-to-image route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationResources {
    /// The package root (`config.json` + `model.safetensors`).
    pub package_dir: PathBuf,
    /// The Qwen3-VL snapshot directory.
    pub text_encoder_dir: PathBuf,
    /// The autoencoder safetensors file.
    pub autoencoder_file: PathBuf,
}

impl GenerationResources {
    /// Resolve and check the route's resources from a load spec: `spec.weights` is the package
    /// root; `components["text_encoder"]` (a directory) and `components["autoencoder"]` (a file,
    /// or a directory holding `ae.safetensors`) override the package's bundled copies. Every
    /// missing piece is a load-time error naming the path and the resource it belongs to.
    pub fn from_spec(spec: &LoadSpec, config: &Flux3Config, model_id: &str) -> Result<Self> {
        crate::control::reject_unknown_components(
            spec,
            &[
                TEXT_ENCODER_COMPONENT,
                AUTOENCODER_COMPONENT,
                pup::UPSAMPLER_COMPONENT,
            ],
            model_id,
        )?;
        let package_dir = match &spec.weights {
            WeightsSource::Dir(dir) => dir.clone(),
            WeightsSource::File(file) => {
                return Err(Error::Msg(format!(
                    "{model_id}: the weights resource must be the {DISTILLED_WEIGHTS_REPO} package \
                     directory ({CONFIG_FILE} + {DIT_WEIGHTS_FILE}), not the single file {}",
                    file.display()
                )))
            }
        };
        let text_encoder_dir = match spec.components.get(TEXT_ENCODER_COMPONENT) {
            Some(WeightsSource::Dir(dir)) => dir.clone(),
            Some(WeightsSource::File(file)) => {
                return Err(Error::Msg(format!(
                    "{model_id}: the '{TEXT_ENCODER_COMPONENT}' component must be the Qwen3-VL \
                     snapshot directory, not the single file {}",
                    file.display()
                )))
            }
            None => package_dir.join(&config.text_encoder.path),
        };
        let autoencoder_file = match spec.components.get(AUTOENCODER_COMPONENT) {
            Some(WeightsSource::File(file)) => file.clone(),
            Some(WeightsSource::Dir(dir)) => dir.join(&config.vae.path),
            None => package_dir.join(&config.vae.path),
        };
        let resources = Self {
            package_dir,
            text_encoder_dir,
            autoencoder_file,
        };
        resources.check(model_id)?;
        Ok(resources)
    }

    /// Existence check of every file the route reads. Shard completeness is checked by the
    /// backend when it resolves the shard index.
    pub fn check(&self, model_id: &str) -> Result<()> {
        require_dir(&self.package_dir, model_id, "package")?;
        for name in [CONFIG_FILE, DIT_WEIGHTS_FILE] {
            require_file(&self.package_dir.join(name), model_id, "package")?;
        }
        require_file(&self.autoencoder_file, model_id, "autoencoder")?;
        require_dir(&self.text_encoder_dir, model_id, "text encoder")?;
        for name in TEXT_ENCODER_REQUIRED_FILES {
            require_file(&self.text_encoder_dir.join(name), model_id, "text encoder")?;
        }
        let index = self.text_encoder_dir.join("model.safetensors.index.json");
        let single = self.text_encoder_dir.join("model.safetensors");
        if !index.is_file() && !single.is_file() {
            return Err(Error::Msg(format!(
                "{model_id}: the text encoder resource {} has neither \
                 model.safetensors.index.json nor model.safetensors — the {TEXT_ENCODER_UPSTREAM_REPO} \
                 snapshot is incomplete",
                self.text_encoder_dir.display()
            )));
        }
        Ok(())
    }

    pub fn dit_weights(&self) -> PathBuf {
        self.package_dir.join(DIT_WEIGHTS_FILE)
    }
}

fn require_dir(dir: &Path, model_id: &str, resource: &str) -> Result<()> {
    if dir.is_dir() {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "{model_id}: the {resource} resource directory {} does not exist",
            dir.display()
        )))
    }
}

fn require_file(path: &Path, model_id: &str, resource: &str) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "{model_id}: the {resource} resource is incomplete — {} is missing",
            path.display()
        )))
    }
}

// ---------------------------------------------------------------------------------------------
// Geometry and the packed-token tables
// ---------------------------------------------------------------------------------------------

/// `autoencoder.latent_shape`: `(h, w)` tokens of a `width × height` image.
pub fn latent_shape(width: u32, height: u32) -> Result<(usize, usize)> {
    if width == 0
        || height == 0
        || !width.is_multiple_of(DOWNSAMPLING_RATIO)
        || !height.is_multiple_of(DOWNSAMPLING_RATIO)
    {
        return Err(Error::Msg(format!(
            "flux3: width and height must be positive multiples of {DOWNSAMPLING_RATIO} (got \
             {width}x{height})"
        )));
    }
    Ok((
        (height / DOWNSAMPLING_RATIO) as usize,
        (width / DOWNSAMPLING_RATIO) as usize,
    ))
}

/// `autoencoder.latent_ids(h, w)`: the integer `(t, row, col, l) = (0, row, col, 0)` id of every
/// token, row-major — the order the autoencoder packs and unpacks tokens in.
pub fn latent_ids(h: usize, w: usize) -> Vec<[i32; 4]> {
    let mut ids = Vec::with_capacity(h * w);
    for row in 0..h {
        for col in 0..w {
            ids.push([0, row as i32, col as i32, 0]);
        }
    }
    ids
}

/// `pipeline.scaled_ids`: the DiT's image ids — the integer ids as f32 with the (row, col) axes
/// scaled by `image_rope_scale` (0.5 in the release).
pub fn scaled_image_ids(ids: &[[i32; 4]], image_rope_scale: f32) -> Vec<[f32; 4]> {
    ids.iter()
        .map(|id| {
            [
                id[0] as f32,
                id[1] as f32 * image_rope_scale,
                id[2] as f32 * image_rope_scale,
                id[3] as f32,
            ]
        })
        .collect()
}

/// `TextEncoder.encode`'s ids: `(0, 0, 0, position)` per window token.
pub fn text_ids(length: usize) -> Vec<[f32; 4]> {
    (0..length).map(|p| [0.0, 0.0, 0.0, p as f32]).collect()
}

/// `dit.rope_embed`'s rotation tables as `(cos, sin)`, `[N, Σ axes_dim/2]` row-major: for id axis
/// `i` with `dim = axes_dim[i]`, pair `k` rotates by `pos_i / theta^(2k/dim)`. Computed in f64 and
/// rounded once to f32, exactly as upstream (`torch.arange(…, float64)`, `einsum` promoting the
/// f32 ids to f64, `.float()` on the cos/sin). Pairs are adjacent channels `(2j, 2j+1)` of the
/// head; axis tables are concatenated in axis order.
pub fn rope_cos_sin(ids: &[[f32; 4]], axes_dim: &[usize], theta: f64) -> (Vec<f32>, Vec<f32>) {
    let pairs: usize = axes_dim.iter().map(|d| d / 2).sum();
    let mut cos = Vec::with_capacity(ids.len() * pairs);
    let mut sin = Vec::with_capacity(ids.len() * pairs);
    for id in ids {
        for (axis, &dim) in axes_dim.iter().enumerate() {
            let pos = id[axis] as f64;
            for k in 0..dim / 2 {
                let scale = (2 * k) as f64 / dim as f64;
                let angle = pos * (1.0 / theta.powf(scale));
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }
    (cos, sin)
}

/// `dit.timestep_embedding`'s frequencies `exp(−ln(max_period)·k/half)`, `half = dim/2`, in f32
/// (upstream builds them with f32 tensor ops).
pub fn timestep_freqs(dim: usize, max_period: f64) -> Vec<f32> {
    let half = dim / 2;
    let neg_log = -(max_period.ln()) as f32;
    (0..half)
        .map(|k| (neg_log * k as f32 / half as f32).exp())
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Text window
// ---------------------------------------------------------------------------------------------

/// `TextEncoder.encode`'s window of one prompt: the template-wrapped prompt's ids clamped at
/// `max_length`, right-padded with the pad id to the next multiple of `pad_multiple` (never
/// beyond `max_length`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextWindow {
    /// `length` ids: the real tokens then pads.
    pub input_ids: Vec<i32>,
    /// `length` 0/1 flags (upstream's `attention_mask`).
    pub mask: Vec<i32>,
    /// `length` positions: `0..real` for the real tokens, [`TEXT_PAD_POSITION`] for every pad.
    pub positions: Vec<i32>,
    /// Real tokens kept (after the clamp).
    pub real: usize,
    /// Prompt tokens the clamp dropped.
    pub truncated_tokens: usize,
}

impl TextWindow {
    pub fn length(&self) -> usize {
        self.input_ids.len()
    }
}

/// `min(ceil(real / pad_multiple) · pad_multiple, max_length)`.
pub fn padded_length(real: usize, pad_multiple: usize, max_length: usize) -> usize {
    (real.div_ceil(pad_multiple) * pad_multiple).min(max_length)
}

/// Assemble the window from the template-wrapped prompt's token ids (`tokenizer(text)` with the
/// pinned tokenizer, which adds no BOS/EOS).
pub fn assemble_window(
    ids: &[i32],
    pad_id: i32,
    pad_multiple: usize,
    max_length: usize,
) -> Result<TextWindow> {
    if pad_multiple == 0 || max_length == 0 {
        return Err(Error::Msg(format!(
            "flux3: a text window needs pad_multiple > 0 and max_length > 0 (got {pad_multiple}, \
             {max_length})"
        )));
    }
    let real = ids.len().min(max_length);
    let length = padded_length(real, pad_multiple, max_length);
    let mut input_ids = ids[..real].to_vec();
    input_ids.resize(length, pad_id);
    let mut mask = vec![1; real];
    mask.resize(length, 0);
    let positions = (0..length)
        .map(|p| {
            if p < real {
                p as i32
            } else {
                TEXT_PAD_POSITION
            }
        })
        .collect();
    Ok(TextWindow {
        input_ids,
        mask,
        positions,
        real,
        truncated_tokens: ids.len() - real,
    })
}

/// The additive attention mask upstream's padded forward applies, `[length, length]` row-major:
/// query `i` sees key `j` iff `j <= i` and `mask[j] == 1` (the pads are masked as keys for every
/// query, including the pad queries themselves, which therefore all attend the same real keys).
pub fn padded_causal_additive(mask: &[i32]) -> Vec<f32> {
    let n = mask.len();
    let mut data = vec![f32::NEG_INFINITY; n * n];
    for i in 0..n {
        for j in 0..=i {
            if mask[j] != 0 {
                data[i * n + j] = 0.0;
            }
        }
    }
    data
}

// ---------------------------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------------------------

/// `torch.linspace(1.0, 0.0, n)` in f32, including torch's symmetric evaluation (`start + i·step`
/// for the first half, `end − (n − 1 − i)·step` for the second), so the grid is bit-identical.
fn linspace_1_to_0(n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![1.0];
    }
    let step = (0.0f32 - 1.0f32) / (n - 1) as f32;
    let halfway = n / 2;
    (0..n)
        .map(|i| {
            if i < halfway {
                1.0f32 + step * i as f32
            } else {
                0.0f32 - step * (n - 1 - i) as f32
            }
        })
        .collect()
}

/// `sampling.timesteps(steps, alpha)`: the `steps + 1` schedule points
/// `alpha·t / (1 + (alpha − 1)·t)` over `t = linspace(1, 0, steps + 1)`, in f32 as upstream
/// (`alpha − 1` is evaluated as a double scalar and applied to the f32 grid).
pub fn flow_map_schedule(steps: u32, alpha: f32) -> Result<Vec<f32>> {
    if steps == 0 {
        return Err(Error::Msg("flux3: steps must be >= 1".into()));
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(Error::Msg(format!(
            "flux3: alpha {alpha} must be a finite value > 0"
        )));
    }
    let alpha_minus_one = (alpha as f64 - 1.0) as f32;
    Ok(linspace_1_to_0(steps as usize + 1)
        .into_iter()
        .map(|t| alpha * t / (1.0 + alpha_minus_one * t))
        .collect())
}

// ---------------------------------------------------------------------------------------------
// Generation control surface
// ---------------------------------------------------------------------------------------------

/// The resolved parameters of one distilled flow-map or base CFG text-to-image request.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationParams {
    pub prompt: String,
    /// CFG scale for the base checkpoint; absent for the distilled flow map.
    pub guidance: Option<f32>,
    /// Unconditional row's caption (empty by default), encoded independently of the prompt.
    pub negative_prompt: String,
    pub steps: u32,
    pub alpha: f32,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
}

/// Upstream `generate.py`'s seed for a request that carries none:
/// `int(hashlib.sha256(caption.encode()).hexdigest(), 16) % 2**32`, i.e. the low 32 bits of
/// the prompt's SHA-256 read big-endian — an unseeded request reproduces per prompt.
pub fn prompt_seed(prompt: &str) -> u64 {
    let digest = Sha256::digest(prompt.as_bytes());
    u64::from(u32::from_be_bytes([
        digest[28], digest[29], digest[30], digest[31],
    ]))
}

impl GenerationParams {
    /// Resolve a completed PUP/bypass caption exactly once. The original request retains its
    /// lossless layout and bindings for validation/provenance; only this internal clone consumes
    /// them, because its prompt already contains the locked annotation table.
    pub fn resolve_caption(
        req: &GenerationRequest,
        caption: String,
        config: &Flux3Config,
    ) -> Result<Self> {
        layouts::compile_request(req)?;
        pup::validate_caption_size(&caption, "effective caption")?;
        let mut effective = req.clone();
        effective.prompt = caption;
        effective.flux3_layout = None;
        effective.flux3_layout_bindings.clear();
        Self::resolve(&effective, prompt_seed(&effective.prompt), config)
    }

    /// Resolve a request against the loaded package. Base CFG consumes guidance (default3.5)
    /// and independently encoded negative conditioning; distilled is a flow map:
    /// any guidance is refused (upstream `generate`: "samples without CFG; drop the guidance"; the
    /// descriptor advertises no guidance axis, so `1.0` is not a special case), as is a negative
    /// prompt (nothing would consume it); the sampler is
    /// the flow-map Euler step and the schedule's `alpha` is the release constant.
    pub fn resolve(
        req: &GenerationRequest,
        default_seed: u64,
        config: &Flux3Config,
    ) -> Result<Self> {
        if req.true_cfg.is_some() {
            return Err(Error::Unsupported("flux3: the separate true_cfg axis is not supported; use guidance for the base checkpoint's CFG scale".into()));
        }
        if let Some(g) = req.guidance.filter(|_| config.dit.flow_map) {
            return Err(Error::Unsupported(format!(
                "flux3: the distilled checkpoint samples without CFG; drop the guidance (got {g})"
            )));
        }
        if config.dit.flow_map
            && req
                .negative_prompt
                .as_deref()
                .is_some_and(|n| !n.is_empty())
        {
            return Err(Error::Unsupported(
                "flux3: the distilled checkpoint samples without CFG, so a negative prompt has no \
                 effect; drop it"
                    .into(),
            ));
        }
        match req.sampler.as_deref() {
            None | Some(SAMPLER) => {}
            Some(other) => {
                return Err(Error::Unsupported(format!(
                    "flux3: sampler {other:?} is not the flow-map integrator (use {SAMPLER:?})"
                )))
            }
        }
        if let Some(scheduler) = &req.scheduler {
            return Err(Error::Unsupported(format!(
                "flux3: scheduler {scheduler:?}: the flow-map schedule has no scheduler name"
            )));
        }
        if let Some(shift) = req.scheduler_shift {
            return Err(Error::Unsupported(format!(
                "flux3: scheduler_shift {shift}: the schedule's alpha is fixed at {DEFAULT_ALPHA} \
                 by the release"
            )));
        }
        let guidance = if config.dit.flow_map {
            None
        } else {
            // Frozen `pipeline.MODELS["base"]`; validate_supported refuses a disagreeing config.
            let g = req.guidance.unwrap_or(BASE_DEFAULT_GUIDANCE);
            if !g.is_finite() || g < 0.0 {
                return Err(Error::Unsupported(
                    "flux3: CFG guidance must be finite and non-negative".into(),
                ));
            }
            Some(g)
        };
        let default_steps = if config.dit.flow_map {
            config.sampler.steps
        } else {
            BASE_DEFAULT_STEPS
        };
        let steps = req.steps.unwrap_or(default_steps);
        if steps == 0 {
            return Err(Error::Msg("flux3: steps must be >= 1".into()));
        }
        let (width, height) = references::request_plan(req)?
            .map(|plan| plan.output.canvas)
            .unwrap_or((req.width, req.height));
        latent_shape(width, height)?;
        Ok(Self {
            prompt: req.prompt.clone(),
            guidance,
            negative_prompt: req.negative_prompt.clone().unwrap_or_default(),
            steps,
            alpha: config.sampler.alpha as f32,
            seed: req.seed.unwrap_or(default_seed),
            width,
            height,
        })
    }

    pub fn schedule(&self) -> Result<Vec<f32>> {
        flow_map_schedule(self.steps, self.alpha)
    }

    pub fn latent_shape(&self) -> Result<(usize, usize)> {
        latent_shape(self.width, self.height)
    }
}

/// Validate a request's full control surface against the loaded package, with the model id
/// prefixed on every error.
pub fn validate_generation_request(
    model_id: &str,
    req: &GenerationRequest,
    config: &Flux3Config,
) -> Result<GenerationParams> {
    GenerationParams::resolve(req, 0, config).map_err(|e| match e {
        Error::Unsupported(m) => Error::Unsupported(format!("{model_id}: {m}")),
        Error::Msg(m) => Error::Msg(format!("{model_id}: {m}")),
        other => other,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pinned package's `config.json` (revision [`DISTILLED_WEIGHTS_REVISION`]), verbatim.
    const RELEASE_CONFIG: &str = r#"{
  "format_version": 1,
  "model": "flux-3-image-klein",
  "quantization": null,
  "torch_dtype": "bfloat16",
  "dit": {
    "hidden": 3072,
    "heads": 24,
    "depth": 5,
    "depth_single": 28,
    "mlp_ratio": 3.0,
    "axes_dim": [32, 32, 32, 32],
    "theta": 10000,
    "in_channels": {"image": 128, "image_cond": 128, "text": 20480},
    "out_modes": ["image"],
    "vec_in_dim": null,
    "image_rope_scale": 0.5,
    "flow_map": true
  },
  "text_encoder": {
    "path": "text_encoder",
    "tokenizer": "text_encoder",
    "layers": [4, 8, 12, 16, 20, 24, 28, 32],
    "pad_multiple": 80,
    "max_tokens": 8192,
    "dtype": "bfloat16"
  },
  "vae": {
    "path": "ae.safetensors",
    "dtype": "float32"
  },
  "sampler": {
    "steps": 16,
    "guidance": null,
    "alpha": 6.93,
    "size": [1024, 1024]
  }
}"#;

    fn release() -> Flux3Config {
        Flux3Config::parse(RELEASE_CONFIG).unwrap()
    }

    #[test]
    fn completed_layout_caption_is_consumed_once_without_rechecking_a_double_table() {
        let original = layouts::Layout::from_json(serde_json::json!({"version":1,"boxes":[{"id":"a","bbox":[0.1,0.2,0.7,0.8],"desc":"x".repeat(9000)}]})).unwrap();
        let req = GenerationRequest {
            prompt: "scene".into(),
            flux3_layout: Some(original.clone()),
            ..Default::default()
        };
        validate_generation_request(DISTILLED_MODEL_ID, &req, &release()).unwrap();
        let input = references::upsampler_input(&req).unwrap();
        let bypass = pup::upsample(None, &input, &Default::default(), &req.cancel, &mut |_| {
            panic!("bypass")
        })
        .unwrap();
        assert!(bypass.caption.chars().count() < pup::MAX_EFFECTIVE_CAPTION_CHARS);
        // A completed enhanced caption can be longer than the bypass, but still below the cap.
        for caption in [
            bypass.caption.clone(),
            format!("{} {}", "enhanced ".repeat(300), bypass.caption),
        ] {
            let params =
                GenerationParams::resolve_caption(&req, caption.clone(), &release()).unwrap();
            assert_eq!(params.prompt, caption);
            assert_eq!(params.seed, prompt_seed(&caption));
            assert_eq!(req.flux3_layout.as_ref(), Some(&original));
        }
    }

    /// The release package agrees with the frozen source's module constants on every width the
    /// native graph reads from the file — and the one place it disagrees (the text clamp) is
    /// resolved toward the source.
    #[test]
    fn release_config_matches_the_frozen_source_constants() {
        let c = release();
        c.validate_supported().unwrap();
        assert_eq!(c.dit.hidden, 3072);
        assert_eq!(c.dit.heads, 24);
        assert_eq!(c.dit.head_dim(), 128);
        assert_eq!(c.dit.depth, 5);
        assert_eq!(c.dit.depth_single, 28);
        assert_eq!(c.dit.mlp_hidden(), 9216);
        assert_eq!(c.dit.axes_dim, [32, 32, 32, 32]);
        assert_eq!(c.dit.theta, 10_000.0);
        assert_eq!(c.dit.image_in_channels, 128);
        assert_eq!(c.dit.image_cond_in_channels, 128);
        assert_eq!(c.dit.text_in_channels, 20480);
        assert_eq!(c.text_in_channels_for(2560), 20480);
        assert_eq!(c.dit.image_rope_scale as f32, IMAGE_ROPE_SCALE);
        assert!(c.dit.flow_map);
        assert_eq!(c.text_encoder.layers, TEXT_OUTPUT_LAYERS);
        assert_eq!(c.text_encoder.pad_multiple, TEXT_PAD_MULTIPLE);
        assert_eq!(c.text_encoder.max_tokens, 8192);
        assert_eq!(c.text_encoder.window_max_length(), TEXT_MAX_LENGTH);
        assert_eq!(c.sampler.steps, DEFAULT_STEPS);
        assert_eq!(c.sampler.alpha as f32, DEFAULT_ALPHA);
        assert_eq!(c.sampler.guidance, None);
        assert_eq!(c.sampler.size, [DEFAULT_SIZE, DEFAULT_SIZE]);
        assert_eq!(c.vae.path, "ae.safetensors");
        assert_eq!(c.text_encoder.path, "text_encoder");
    }

    /// *Mutation that reds this:* dropping any row of `validate_supported`.
    #[test]
    fn unsupported_packages_are_refused_by_key() {
        let cases: [(&str, &str, &str); 6] = [
            (
                "\"quantization\": null",
                "\"quantization\": \"fp8r\"",
                "quantization",
            ),
            ("\"flow_map\": true", "\"flow_map\": false", "dit.flow_map"),
            (
                "\"out_modes\": [\"image\"]",
                "\"out_modes\": [\"image\", \"depth\"]",
                "dit.out_modes",
            ),
            (
                "\"axes_dim\": [32, 32, 32, 32]",
                "\"axes_dim\": [32, 32, 64]",
                "dit.axes_dim",
            ),
            (
                "\"torch_dtype\": \"bfloat16\"",
                "\"torch_dtype\": \"float8\"",
                "torch_dtype",
            ),
            (
                "\"vec_in_dim\": null",
                "\"vec_in_dim\": 768",
                "dit.vec_in_dim",
            ),
        ];
        for (from, to, key) in cases {
            let text = RELEASE_CONFIG.replacen(from, to, 1);
            assert_ne!(text, RELEASE_CONFIG, "{key}: the mutation applied");
            let config = Flux3Config::parse(&text).unwrap();
            match config.validate_supported() {
                Err(Error::Unsupported(m)) => assert!(m.contains(key), "{key}: {m}"),
                other => panic!("{key}: expected a typed refusal, got {other:?}"),
            }
        }
        // A missing key is a parse error naming it.
        let text = RELEASE_CONFIG.replacen("\"pad_multiple\": 80,", "", 1);
        let err = Flux3Config::parse(&text).unwrap_err().to_string();
        assert!(err.contains("text_encoder.pad_multiple"), "{err}");
    }

    /// `sampling.timesteps` at the release defaults, bit for bit (torch f32, recorded from the
    /// frozen source on CPU). *Mutation that reds this:* evaluating the grid in f64, dropping
    /// torch's symmetric linspace, or folding `alpha − 1` into f32 before the subtraction.
    #[test]
    fn flow_map_schedule_is_bit_identical_to_upstream() {
        let want_16: [u32; 17] = [
            1065353216, 1065193357, 1065014351, 1064812539, 1064583266, 1064320518, 1064016384,
            1063660259, 1063237552, 1062727674, 1062100557, 1061310512, 1060284570, 1058898530,
            1056880300, 1050790776, 0,
        ];
        let got: Vec<u32> = flow_map_schedule(16, DEFAULT_ALPHA)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, want_16);
        let want_4: [u32; 5] = [1065353216, 1064583266, 1063237552, 1060284570, 0];
        let got: Vec<u32> = flow_map_schedule(4, DEFAULT_ALPHA)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, want_4);
        assert_eq!(flow_map_schedule(1, DEFAULT_ALPHA).unwrap(), [1.0, 0.0]);
        let want_50_tail: [u32; 4] = [1052804440, 1050477946, 1046834813, 1040040406];
        let got = flow_map_schedule(50, DEFAULT_ALPHA).unwrap();
        assert_eq!(got.len(), 51);
        let tail: Vec<u32> = got[46..50].iter().map(|v| v.to_bits()).collect();
        assert_eq!(tail, want_50_tail);
        assert!(flow_map_schedule(0, DEFAULT_ALPHA).is_err());
        assert!(flow_map_schedule(4, 0.0).is_err());
    }

    #[test]
    fn text_window_pads_to_80_and_clamps_at_4096() {
        assert_eq!(padded_length(1, 80, 4096), 80);
        assert_eq!(padded_length(80, 80, 4096), 80);
        assert_eq!(padded_length(81, 80, 4096), 160);
        assert_eq!(padded_length(4096, 80, 4096), 4096);
        let ids: Vec<i32> = (100..107).collect();
        let w = assemble_window(&ids, 7, 80, 4096).unwrap();
        assert_eq!(w.length(), 80);
        assert_eq!(w.real, 7);
        assert_eq!(&w.input_ids[..7], &ids[..]);
        assert!(w.input_ids[7..].iter().all(|&id| id == 7));
        assert_eq!(w.mask.iter().sum::<i32>(), 7);
        assert_eq!(&w.positions[..7], &[0, 1, 2, 3, 4, 5, 6]);
        assert!(w.positions[7..].iter().all(|&p| p == TEXT_PAD_POSITION));
        assert_eq!(w.truncated_tokens, 0);
        // The clamp: 5000 prompt tokens keep 4096, drop 904, pad none.
        let long: Vec<i32> = (0..5000).collect();
        let w = assemble_window(&long, 7, 80, 4096).unwrap();
        assert_eq!((w.length(), w.real, w.truncated_tokens), (4096, 4096, 904));
        // An exact multiple pads nothing.
        let w = assemble_window(&vec![1; 160], 7, 80, 4096).unwrap();
        assert_eq!((w.length(), w.real), (160, 160));
        assert!(assemble_window(&ids, 7, 0, 4096).is_err());
    }

    /// Pads are masked as keys for every query (including the pad queries), real keys are visible
    /// causally. *Mutation that reds this:* dropping the `mask[j] != 0` condition.
    #[test]
    fn padded_causal_mask_hides_pad_keys_from_every_query() {
        let m = padded_causal_additive(&[1, 1, 0, 0]);
        let row = |i: usize| &m[i * 4..i * 4 + 4];
        assert_eq!(
            row(0),
            [0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]
        );
        assert_eq!(row(1), [0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        assert_eq!(row(2), [0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        assert_eq!(row(3), [0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
    }

    #[test]
    fn ids_follow_the_autoencoder_and_pipeline_conventions() {
        assert_eq!(latent_shape(1024, 1024).unwrap(), (64, 64));
        assert_eq!(latent_shape(1536, 1024).unwrap(), (64, 96));
        assert!(latent_shape(1000, 1024).is_err());
        assert!(latent_shape(0, 16).is_err());
        let ids = latent_ids(2, 3);
        assert_eq!(
            ids,
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 2, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 2, 0]
            ]
        );
        let scaled = scaled_image_ids(&ids, IMAGE_ROPE_SCALE);
        assert_eq!(scaled[5], [0.0, 0.5, 1.0, 0.0]);
        assert_eq!(
            text_ids(3),
            [[0.0; 4], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]]
        );
        assert_eq!(
            chat_text("a fox"),
            "<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// `rope` at the release geometry: axis tables concatenated in axis order, `dim/2` pairs per
    /// axis, angle `pos / theta^(2k/dim)` in f64 rounded to f32.
    #[test]
    fn rope_tables_follow_upstream_axis_layout() {
        let ids = [[0.0, 1.5, 2.0, 0.0], [0.0, 0.0, 0.0, 3.0]];
        let (cos, sin) = rope_cos_sin(&ids, &[32, 32, 32, 32], 10_000.0);
        assert_eq!(cos.len(), 2 * 64);
        assert_eq!(sin.len(), 2 * 64);
        // token 0: axis 0 (pos 0) ⇒ cos 1 / sin 0 for its 16 pairs
        assert!(cos[..16].iter().all(|&c| c == 1.0) && sin[..16].iter().all(|&s| s == 0.0));
        // axis 1 (pos 1.5), pair 0 ⇒ angle 1.5
        assert_eq!(cos[16], (1.5f64).cos() as f32);
        assert_eq!(sin[16], (1.5f64).sin() as f32);
        // axis 1, pair 3 ⇒ angle 1.5 / 10000^(6/32)
        let omega = 1.0 / 10_000f64.powf(6.0 / 32.0);
        assert_eq!(cos[16 + 3], (1.5 * omega).cos() as f32);
        // token 1: axis 3 (pos 3), pair 1 ⇒ angle 3 / 10000^(2/32)
        let omega = 1.0 / 10_000f64.powf(2.0 / 32.0);
        assert_eq!(sin[64 + 48 + 1], (3.0 * omega).sin() as f32);
        let freqs = timestep_freqs(TIMESTEP_EMBED_DIM, TIMESTEP_MAX_PERIOD);
        assert_eq!(freqs.len(), 128);
        assert_eq!(freqs[0], 1.0);
        assert!((freqs[127] - (-(10_000f32.ln()) * 127.0 / 128.0).exp()).abs() < 1e-9);
    }

    fn request() -> GenerationRequest {
        GenerationRequest {
            prompt: "a red fox".into(),
            width: 1024,
            height: 768,
            ..Default::default()
        }
    }

    #[test]
    fn base_config_disagreeing_with_frozen_defaults_is_refused() {
        let mut base = release();
        base.dit.flow_map = false;
        base.sampler.steps = BASE_DEFAULT_STEPS;
        base.sampler.guidance = Some(3.5);
        base.validate_supported().unwrap();
        for (steps, guidance) in [(16, Some(3.5)), (50, Some(4.0)), (28, Some(2.5))] {
            let mut wrong = base.clone();
            wrong.sampler.steps = steps;
            wrong.sampler.guidance = guidance;
            assert!(
                matches!(wrong.validate_supported(), Err(Error::Unsupported(_))),
                "{steps}/{guidance:?}"
            );
        }
        // Defaults come from the frozen constants even when the config field is ignored.
        let mut drifted = base.clone();
        drifted.sampler.steps = 7;
        drifted.sampler.guidance = Some(9.0);
        let p = GenerationParams::resolve(&request(), 0, &drifted).unwrap();
        assert_eq!(
            (p.steps, p.guidance),
            (BASE_DEFAULT_STEPS, Some(BASE_DEFAULT_GUIDANCE))
        );
    }

    #[test]
    fn base_uses_cfg_defaults_and_explicit_negative_conditioning() {
        let mut base = release();
        base.dit.flow_map = false;
        base.sampler.steps = 50;
        base.sampler.guidance = Some(3.5);
        let p = GenerationParams::resolve(&request(), 19, &base).unwrap();
        assert_eq!(p.steps, 50);
        assert_eq!(p.guidance, Some(3.5));
        assert_eq!(p.negative_prompt, "");
        assert_eq!(p.schedule().unwrap().len(), 51);
        let explicit = GenerationRequest {
            steps: Some(7),
            guidance: Some(1.0),
            negative_prompt: Some("blur, watermark".into()),
            ..request()
        };
        let p = GenerationParams::resolve(&explicit, 19, &base).unwrap();
        assert_eq!(p.steps, 7);
        assert_eq!(p.guidance, Some(1.0));
        assert_eq!(p.negative_prompt, "blur, watermark");
        let separate_true_cfg = GenerationRequest {
            true_cfg: Some(4.25),
            ..request()
        };
        assert!(matches!(
            GenerationParams::resolve(&separate_true_cfg, 19, &base),
            Err(Error::Unsupported(_))
        ));
        for guidance in [f32::NAN, f32::INFINITY, -1.0] {
            let req = GenerationRequest {
                guidance: Some(guidance),
                ..request()
            };
            assert!(matches!(
                GenerationParams::resolve(&req, 0, &base),
                Err(Error::Unsupported(_))
            ));
        }
    }

    #[test]
    fn params_resolve_the_release_defaults() {
        let p = GenerationParams::resolve(&request(), 7, &release()).unwrap();
        assert_eq!(p.steps, 16);
        assert_eq!(p.alpha, DEFAULT_ALPHA);
        assert_eq!(p.seed, 7);
        assert_eq!(p.latent_shape().unwrap(), (48, 64));
        assert_eq!(p.schedule().unwrap().len(), 17);
        let explicit = GenerationRequest {
            steps: Some(4),
            seed: Some(3),
            sampler: Some("euler".into()),
            ..request()
        };
        let p = GenerationParams::resolve(&explicit, 7, &release()).unwrap();
        assert_eq!((p.steps, p.seed), (4, 3));
    }

    /// Upstream `generate.py`: `int(sha256(caption).hexdigest(), 16) % 2**32`, checked against
    /// `hashlib` for three captions. *Mutation that reds this:* little-endian bytes, or any
    /// other 4 bytes of the digest.
    #[test]
    fn prompt_seed_is_upstreams_sha256_low_32_bits() {
        assert_eq!(prompt_seed("a red fox in the snow"), 2_515_539_735);
        assert_eq!(prompt_seed(""), 2_018_687_061);
        assert_eq!(
            prompt_seed("a red fox sitting in fresh snow at golden hour, soft light, photograph"),
            2_007_615_934
        );
        assert!(prompt_seed("anything") <= u64::from(u32::MAX));
    }

    /// *Mutation that reds this:* dropping any refusal arm of `resolve`.
    #[test]
    fn controls_the_flow_map_cannot_honour_are_typed_refusals() {
        let cases: Vec<(&str, GenerationRequest)> = vec![
            (
                "guidance",
                GenerationRequest {
                    guidance: Some(3.5),
                    ..request()
                },
            ),
            (
                "guidance 1.0 (no guidance axis exists; the descriptor says so too)",
                GenerationRequest {
                    guidance: Some(1.0),
                    ..request()
                },
            ),
            (
                "negative prompt",
                GenerationRequest {
                    negative_prompt: Some("blurry".into()),
                    ..request()
                },
            ),
            (
                "sampler",
                GenerationRequest {
                    sampler: Some("dpmpp_2m".into()),
                    ..request()
                },
            ),
            (
                "scheduler",
                GenerationRequest {
                    scheduler: Some("karras".into()),
                    ..request()
                },
            ),
            (
                "scheduler_shift",
                GenerationRequest {
                    scheduler_shift: Some(3.0),
                    ..request()
                },
            ),
        ];
        for (what, req) in cases {
            match validate_generation_request("flux3_klein", &req, &release()) {
                Err(Error::Unsupported(m)) => assert!(m.starts_with("flux3_klein:"), "{what}: {m}"),
                other => panic!("{what}: expected a typed refusal, got {other:?}"),
            }
        }
        let off_grid = GenerationRequest {
            width: 1000,
            ..request()
        };
        assert!(matches!(
            validate_generation_request("flux3_klein", &off_grid, &release()),
            Err(Error::Msg(_))
        ));
        let mut base = release();
        base.dit.flow_map = false;
        base.sampler.steps = BASE_DEFAULT_STEPS;
        base.sampler.guidance = Some(3.5);
        base.validate_supported().unwrap();
        assert_eq!(
            GenerationParams::resolve(&request(), 0, &base)
                .unwrap()
                .guidance,
            Some(3.5)
        );
    }

    #[test]
    fn resources_resolve_the_bundled_components_and_their_overrides() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path();
        std::fs::write(dir.join(CONFIG_FILE), RELEASE_CONFIG).unwrap();
        std::fs::write(dir.join(DIT_WEIGHTS_FILE), b"x").unwrap();
        let config = release();
        let spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()));
        // Nothing bundled yet: the autoencoder is named first.
        let err = GenerationResources::from_spec(&spec, &config, "m")
            .unwrap_err()
            .to_string();
        assert!(err.contains("ae.safetensors"), "{err}");
        std::fs::write(dir.join("ae.safetensors"), b"x").unwrap();
        let err = GenerationResources::from_spec(&spec, &config, "m")
            .unwrap_err()
            .to_string();
        assert!(err.contains("text encoder"), "{err}");
        let te = dir.join("text_encoder");
        std::fs::create_dir(&te).unwrap();
        for name in TEXT_ENCODER_REQUIRED_FILES {
            std::fs::write(te.join(name), b"{}").unwrap();
        }
        let err = GenerationResources::from_spec(&spec, &config, "m")
            .unwrap_err()
            .to_string();
        assert!(err.contains("model.safetensors.index.json"), "{err}");
        std::fs::write(te.join("model.safetensors"), b"x").unwrap();
        let resources = GenerationResources::from_spec(&spec, &config, "m").unwrap();
        assert_eq!(resources.text_encoder_dir, te);
        assert_eq!(resources.autoencoder_file, dir.join("ae.safetensors"));
        assert_eq!(resources.dit_weights(), dir.join(DIT_WEIGHTS_FILE));

        // Overrides: a separate encoder snapshot and an autoencoder file elsewhere.
        let other = tempfile::tempdir().unwrap();
        let te2 = other.path().join("qwen");
        std::fs::create_dir(&te2).unwrap();
        for name in TEXT_ENCODER_REQUIRED_FILES {
            std::fs::write(te2.join(name), b"{}").unwrap();
        }
        std::fs::write(te2.join("model.safetensors.index.json"), b"{}").unwrap();
        let ae2 = other.path().join("ae-elsewhere.safetensors");
        std::fs::write(&ae2, b"x").unwrap();
        let mut spec = spec.clone();
        spec.components.insert(
            TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(te2.clone()),
        );
        spec.components.insert(
            AUTOENCODER_COMPONENT.into(),
            WeightsSource::File(ae2.clone()),
        );
        let resources = GenerationResources::from_spec(&spec, &config, "m").unwrap();
        assert_eq!(resources.text_encoder_dir, te2);
        assert_eq!(resources.autoencoder_file, ae2);
        // An unknown component key and a single-file weights source are refused.
        let mut bad = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()));
        bad.components
            .insert("upsampler".into(), WeightsSource::Dir(te2));
        assert!(GenerationResources::from_spec(&bad, &config, "m").is_err());
        let file = LoadSpec::new(WeightsSource::File(dir.join(DIT_WEIGHTS_FILE)));
        assert!(GenerationResources::from_spec(&file, &config, "m").is_err());
    }
}
