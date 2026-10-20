//! Supra2-IMG — backend-neutral contract, install-time converter and host-side sampling policy
//! (epic sc-24212, story sc-24213).
//!
//! Upstream: `SupraLabs/Supra2-IMG` at the frozen revision [`SUPRA2_SOURCE`], reference
//! implementation `inference.py`: a ~104M-parameter rectified-flow DiT conditioned on
//! `google/flan-t5-base` (128-token context, attention mask) decoded by `stabilityai/sd-vae-ft-mse`
//! (`latent / 0.18215`). Published interface: 256×256 text-to-image with prompt, seed, guidance
//! (CFG, enabled only when `guidance > 1`), steps and image count; defaults seed 0, CFG 3.0, 50 steps.
//!
//! Everything both backends must agree on lives here, tensor-free, so the MLX provider and the
//! Candle/CUDA port share one implementation:
//!
//! * the architecture geometry ([`Supra2Config`]) and the exact denoiser tensor inventory,
//! * the revision pins of every component ([`SUPRA2_SOURCE`], [`FLAN_T5_BASE`], [`SD_VAE`]),
//! * the **converter** ([`convert_pinned_checkpoint`]) turning the upstream `torch.save` checkpoint
//!   into a validated native bundle ([`BUNDLE_MANIFEST`], [`DENOISER_FILE`], [`CONDITIONING_FILE`]),
//!   run once at install time on the user's machine — nothing is published,
//! * tokenization + truncation + mask ([`Supra2Tokenizer`]),
//! * request resolution ([`Supra2Resolved`]), the uniform Euler flow schedule, the sinusoidal
//!   timestep features, the CFG rule, deterministic per-image initial noise, and the
//!   `torchvision.save_image` pixel quantization.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::pth::{PthArchive, PthDtype, PthTensor, PthValue};
use crate::{Error, GenerationRequest, Result};

/// Registry id of the text-to-image provider.
pub const PROVIDER_ID: &str = "supra2_img";
/// Provider family.
pub const FAMILY: &str = "supra2";
/// `LoadSpec::components` id carrying the `google/flan-t5-base` snapshot directory.
pub const TEXT_ENCODER_COMPONENT: &str = "text_encoder";
/// `LoadSpec::components` id carrying the `stabilityai/sd-vae-ft-mse` snapshot directory.
pub const VAE_COMPONENT: &str = "vae";

/// The only published output resolution (square).
pub const IMAGE_SIZE: u32 = 256;
/// Recommended defaults (`inference.py` argparse defaults and the model card).
pub const DEFAULT_STEPS: u32 = 50;
pub const DEFAULT_GUIDANCE: f32 = 3.0;
pub const DEFAULT_SEED: u64 = 0;
/// Images per request. Upstream's `--n` has no cap; this bounds one request's latent batch.
pub const MAX_IMAGES: u32 = 64;
/// SD VAE latent scaling (`VAE_SCALE`).
pub const VAE_SCALE: f32 = 0.18215;
/// Sampler / scheduler names advertised on the descriptor.
pub const SAMPLER: &str = "euler";
pub const SCHEDULER: &str = "supra2_uniform_flow";

/// Supra2 DiT geometry. [`Supra2Config::published`] is the frozen checkpoint; the struct is
/// dimension-parametric so tiny parity fixtures run through the same code.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Supra2Config {
    pub latent_channels: usize,
    pub latent_size: usize,
    pub patch: usize,
    pub d_model: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_hidden: usize,
    pub ctx_dim: usize,
    pub ctx_len: usize,
    pub freq_dim: usize,
}

impl Supra2Config {
    /// `inference.py` constants: 32² latent, patch 2, D 576, depth 14, 9 heads, MLP ×4, Flan-T5-Base
    /// 768-wide context, 128 tokens, 256 sinusoidal frequencies.
    pub const fn published() -> Self {
        Self {
            latent_channels: 4,
            latent_size: 32,
            patch: 2,
            d_model: 576,
            depth: 14,
            num_heads: 9,
            mlp_hidden: 2304,
            ctx_dim: 768,
            ctx_len: 128,
            freq_dim: 256,
        }
    }

    pub fn num_tokens(&self) -> usize {
        (self.latent_size / self.patch).pow(2)
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    pub fn patch_dim(&self) -> usize {
        self.latent_channels * self.patch * self.patch
    }

    /// Elements of one image's latent `[C, S, S]`.
    pub fn latent_numel(&self) -> usize {
        self.latent_channels * self.latent_size * self.latent_size
    }

    fn validate(&self) -> Result<()> {
        let positive = [
            self.latent_channels,
            self.latent_size,
            self.patch,
            self.d_model,
            self.depth,
            self.num_heads,
            self.mlp_hidden,
            self.ctx_dim,
            self.ctx_len,
            self.freq_dim,
        ];
        if positive.contains(&0)
            || !self.latent_size.is_multiple_of(self.patch)
            || !self.d_model.is_multiple_of(self.num_heads)
            || !self.freq_dim.is_multiple_of(2)
        {
            return Err(Error::Msg(format!("supra2: invalid geometry {self:?}")));
        }
        Ok(())
    }

    /// The exact `SupraDiT.state_dict()` inventory: every key upstream's `load_state_dict(strict=True)`
    /// requires, with its PyTorch shape.
    pub fn expected_tensors(&self) -> Vec<(String, Vec<usize>)> {
        let d = self.d_model;
        let mut t = vec![
            ("pos_embed".to_string(), vec![1, self.num_tokens(), d]),
            ("x_embed.weight".into(), vec![d, self.patch_dim()]),
            ("x_embed.bias".into(), vec![d]),
            ("t_embed.mlp.0.weight".into(), vec![d, self.freq_dim]),
            ("t_embed.mlp.0.bias".into(), vec![d]),
            ("t_embed.mlp.2.weight".into(), vec![d, d]),
            ("t_embed.mlp.2.bias".into(), vec![d]),
            ("ctx_proj.weight".into(), vec![d, self.ctx_dim]),
            ("ctx_proj.bias".into(), vec![d]),
        ];
        for i in 0..self.depth {
            let p = |s: &str| format!("blocks.{i}.{s}");
            t.extend([
                (p("self_attn.qkv.weight"), vec![3 * d, d]),
                (p("self_attn.qkv.bias"), vec![3 * d]),
                (p("self_attn.proj.weight"), vec![d, d]),
                (p("self_attn.proj.bias"), vec![d]),
                (p("cross_attn.q.weight"), vec![d, d]),
                (p("cross_attn.q.bias"), vec![d]),
                (p("cross_attn.kv.weight"), vec![2 * d, d]),
                (p("cross_attn.kv.bias"), vec![2 * d]),
                (p("cross_attn.proj.weight"), vec![d, d]),
                (p("cross_attn.proj.bias"), vec![d]),
                (p("mlp.0.weight"), vec![self.mlp_hidden, d]),
                (p("mlp.0.bias"), vec![self.mlp_hidden]),
                (p("mlp.2.weight"), vec![d, self.mlp_hidden]),
                (p("mlp.2.bias"), vec![d]),
                (p("adaln.1.weight"), vec![6 * d, d]),
                (p("adaln.1.bias"), vec![6 * d]),
            ]);
        }
        t.extend([
            ("final.linear.weight".to_string(), vec![self.patch_dim(), d]),
            ("final.linear.bias".into(), vec![self.patch_dim()]),
            ("final.adaln.1.weight".into(), vec![2 * d, d]),
            ("final.adaln.1.bias".into(), vec![2 * d]),
        ]);
        t
    }

    /// Denoiser parameter count (104.1M for the published geometry).
    pub fn parameter_count(&self) -> usize {
        self.expected_tensors()
            .iter()
            .map(|(_, s)| s.iter().product::<usize>())
            .sum()
    }
}

// ------------------------------------------------------------------------------------------------
// Revision pins
// ------------------------------------------------------------------------------------------------

/// One exact file of a pinned Hugging Face revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinnedFile {
    pub path: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

/// A component pinned to an exact Hugging Face revision. `files` are the files the runtime reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentPin {
    pub repo: &'static str,
    pub revision: &'static str,
    pub license: &'static str,
    pub files: &'static [PinnedFile],
}

impl ComponentPin {
    /// Sum of the pinned files — the download footprint of the files this provider reads.
    pub fn download_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }

    /// Cheap load-time check: every pinned file exists with its pinned size.
    pub fn check_sizes(&self, dir: &Path) -> Result<()> {
        for f in self.files {
            let path = dir.join(f.path);
            let meta = std::fs::metadata(&path).map_err(|e| {
                Error::Unsupported(format!(
                    "supra2: {}@{} is missing `{}` in {}: {e}",
                    self.repo,
                    self.revision,
                    f.path,
                    dir.display()
                ))
            })?;
            if meta.len() != f.bytes {
                return Err(Error::Unsupported(format!(
                    "supra2: {} has {} bytes, but {}@{} pins {} bytes",
                    path.display(),
                    meta.len(),
                    self.repo,
                    self.revision,
                    f.bytes
                )));
            }
        }
        Ok(())
    }

    /// Install-time check: sizes and full SHA-256 of every pinned file.
    pub fn verify(&self, dir: &Path) -> Result<()> {
        self.check_sizes(dir)?;
        for f in self.files {
            let path = dir.join(f.path);
            let digest = sha256_file(&path)?;
            if digest != f.sha256 {
                return Err(Error::Unsupported(format!(
                    "supra2: {} has sha256 {digest}, but {}@{} pins {}",
                    path.display(),
                    self.repo,
                    self.revision,
                    f.sha256
                )));
            }
        }
        Ok(())
    }
}

/// The Supra2 denoiser checkpoint (Apache-2.0).
pub const SUPRA2_SOURCE: ComponentPin = ComponentPin {
    repo: "SupraLabs/Supra2-IMG",
    revision: "10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b",
    license: "Apache-2.0",
    files: &[PinnedFile {
        path: "model_final_ema.pt",
        bytes: 416_651_529,
        sha256: "e96aa0537e0fd120af82b3a606dd50c90ae079ac7aa97b38a8da4882daeeee48",
    }],
};

/// Flan-T5-Base text encoder + tokenizer (Apache-2.0).
pub const FLAN_T5_BASE: ComponentPin = ComponentPin {
    repo: "google/flan-t5-base",
    revision: "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    license: "Apache-2.0",
    files: &[
        PinnedFile {
            path: "config.json",
            bytes: 1_404,
            sha256: "7c1853dbfa0e4aac093eb109a358b6ab25fe86b7c15185a91322f0ed26f0f940",
        },
        PinnedFile {
            path: "model.safetensors",
            bytes: 990_345_061,
            sha256: "1dfb70afdcedceb9f9fae2f9b68e004ad934361fb35b9b2bd50b45ea90790fc8",
        },
        PinnedFile {
            path: "tokenizer.json",
            bytes: 2_424_064,
            sha256: "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a",
        },
    ],
};

/// SD VAE ft-MSE decoder (MIT).
pub const SD_VAE: ComponentPin = ComponentPin {
    repo: "stabilityai/sd-vae-ft-mse",
    revision: "31f26fdeee1355a5c34592e401dd41e45d25a493",
    license: "MIT",
    files: &[
        PinnedFile {
            path: "config.json",
            bytes: 547,
            sha256: "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e",
        },
        PinnedFile {
            path: "diffusion_pytorch_model.safetensors",
            bytes: 334_643_276,
            sha256: "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815",
        },
    ],
};

/// SHA-256 of a file, streamed.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| Error::Msg(format!("supra2: open {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

// ------------------------------------------------------------------------------------------------
// Converted bundle
// ------------------------------------------------------------------------------------------------

/// Bundle manifest file name.
pub const BUNDLE_MANIFEST: &str = "supra2_bundle.json";
/// Denoiser weights: F32 safetensors, upstream `state_dict` key names verbatim.
pub const DENOISER_FILE: &str = "denoiser.safetensors";
/// Checkpoint-provided unconditional conditioning: `uncond_text` F32 `[ctx_len, ctx_dim]` and
/// `uncond_mask` F32 `[ctx_len]` (0/1). Absent when the checkpoint stores none (the pipeline then
/// encodes the empty string, exactly as upstream's fallback).
pub const CONDITIONING_FILE: &str = "conditioning.safetensors";
/// Manifest schema tag.
pub const BUNDLE_SCHEMA: &str = "sceneworks.supra2.bundle/v1";

/// A file written into the bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleFile {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
}

/// Identity of the converted source checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleSource {
    /// `Some` when the source matched [`SUPRA2_SOURCE`] byte-for-byte.
    pub repo: Option<String>,
    pub revision: Option<String>,
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
    /// Which state dict was converted: `ema` when present, else `model` (upstream's preference).
    pub weights_key: String,
}

/// A component the bundle expects alongside it (`LoadSpec::components`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleComponent {
    pub component: String,
    pub repo: String,
    pub revision: String,
    pub license: String,
    pub download_bytes: u64,
}

/// `supra2_bundle.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BundleManifest {
    pub schema: String,
    pub provider_id: String,
    pub source: BundleSource,
    pub config: Supra2Config,
    pub vae_scale: f32,
    pub denoiser: BundleFile,
    pub denoiser_tensors: usize,
    /// `Some` when the checkpoint stores unconditional embeddings.
    pub conditioning: Option<BundleFile>,
    pub components: Vec<BundleComponent>,
}

/// A validated bundle directory.
#[derive(Clone, Debug)]
pub struct Supra2Bundle {
    pub dir: PathBuf,
    pub manifest: BundleManifest,
}

impl Supra2Bundle {
    pub fn denoiser_path(&self) -> PathBuf {
        self.dir.join(&self.manifest.denoiser.file)
    }

    pub fn conditioning_path(&self) -> Option<PathBuf> {
        self.manifest
            .conditioning
            .as_ref()
            .map(|c| self.dir.join(&c.file))
    }
}

/// Converter options. Production uses [`convert_pinned_checkpoint`].
#[derive(Clone, Debug)]
pub struct ConvertOptions {
    pub config: Supra2Config,
    /// Require the source to match [`SUPRA2_SOURCE`] byte-for-byte.
    pub require_pinned_source: bool,
}

/// Convert the frozen upstream `model_final_ema.pt` (verified against [`SUPRA2_SOURCE`]) into a
/// validated native bundle in `out_dir`.
pub fn convert_pinned_checkpoint(source: &Path, out_dir: &Path) -> Result<BundleManifest> {
    convert_checkpoint(
        source,
        out_dir,
        &ConvertOptions {
            config: Supra2Config::published(),
            require_pinned_source: true,
        },
    )
}

fn tensor_f32_bytes(name: &str, t: &PthTensor) -> Result<Vec<u8>> {
    let values: Vec<f32> = match t.dtype {
        PthDtype::F32 => return Ok(t.data.clone()),
        PthDtype::BF16 | PthDtype::F16 => t.to_f32()?,
        PthDtype::I64 => {
            return Err(Error::Unsupported(format!(
                "supra2: denoiser tensor `{name}` is int64"
            )))
        }
    };
    Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
}

fn all_finite(bytes: &[u8]) -> bool {
    bytes
        .chunks_exact(4)
        .all(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite())
}

fn as_usize_shape(shape: &[i64]) -> Vec<usize> {
    shape.iter().map(|&d| d as usize).collect()
}

fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<u8>)]) -> Result<()> {
    let bad = |e: safetensors::SafeTensorError| Error::Msg(format!("supra2: safetensors: {e}"));
    let views = tensors
        .iter()
        .map(|(name, shape, data)| {
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), data)
                .map(|v| (name.clone(), v))
                .map_err(bad)
        })
        .collect::<Result<Vec<_>>>()?;
    let tmp = path.with_extension("safetensors.partial");
    safetensors::serialize_to_file(views, &None, &tmp).map_err(bad)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn bundle_file(dir: &Path, file: &str) -> Result<BundleFile> {
    let path = dir.join(file);
    Ok(BundleFile {
        file: file.to_string(),
        bytes: std::fs::metadata(&path)?.len(),
        sha256: sha256_file(&path)?,
    })
}

/// Convert a Supra2 `torch.save` checkpoint into `out_dir`, validating it against `opts.config`:
///
/// * the state dict is `ema` when present, else `model` (else the top object), exactly upstream's
///   selection, and must match [`Supra2Config::expected_tensors`] key-for-key and shape-for-shape
///   (upstream's `load_state_dict(strict=True)`), with every value finite;
/// * `config.patch` (when present) must equal the geometry's patch; `config.ctx_len` (default 128)
///   must equal its context length;
/// * stored `uncond_text`/`uncond_mask` (when present) must be `[ctx_len, ctx_dim]`/`[ctx_len]`,
///   finite, with a 0/1 mask selecting at least one token.
///
/// Writes [`DENOISER_FILE`], [`CONDITIONING_FILE`] (when stored) and [`BUNDLE_MANIFEST`] last, so a
/// partial conversion never leaves a manifest behind.
pub fn convert_checkpoint(
    source: &Path,
    out_dir: &Path,
    opts: &ConvertOptions,
) -> Result<BundleManifest> {
    let cfg = opts.config;
    cfg.validate()?;
    let source_bytes = std::fs::metadata(source)
        .map_err(|e| Error::Msg(format!("supra2: source {}: {e}", source.display())))?
        .len();
    let source_sha = sha256_file(source)?;
    let pinned = SUPRA2_SOURCE.files[0];
    let matches_pin = source_bytes == pinned.bytes && source_sha == pinned.sha256;
    if opts.require_pinned_source && !matches_pin {
        return Err(Error::Unsupported(format!(
            "supra2: {} (sha256 {source_sha}, {source_bytes} bytes) is not {}@{}/{}",
            source.display(),
            SUPRA2_SOURCE.repo,
            SUPRA2_SOURCE.revision,
            pinned.path
        )));
    }
    let mut archive = PthArchive::open(source)?;
    let root = archive.root().clone();
    let (weights_key, weights) = match (root.get("ema"), root.get("model")) {
        (Some(ema @ PthValue::Dict(_)), _) => ("ema", ema.clone()),
        (_, Some(model @ PthValue::Dict(_))) => ("model", model.clone()),
        _ => ("", root.clone()),
    };
    let ckpt_cfg = root.get("config").cloned();
    let int_cfg = |key: &str| match ckpt_cfg.as_ref().and_then(|c| c.get(key)) {
        Some(PthValue::Int(n)) => Ok(Some(*n)),
        None => Ok(None),
        Some(other) => Err(Error::Unsupported(format!(
            "supra2: checkpoint config `{key}` is {other:?}, expected an int"
        ))),
    };
    if let Some(patch) = int_cfg("patch")? {
        if patch as usize != cfg.patch {
            return Err(Error::Unsupported(format!(
                "supra2: checkpoint PATCH={patch} != geometry PATCH={}",
                cfg.patch
            )));
        }
    }
    let ctx_len = int_cfg("ctx_len")?.unwrap_or(cfg.ctx_len as i64);
    if ctx_len as usize != cfg.ctx_len {
        return Err(Error::Unsupported(format!(
            "supra2: checkpoint ctx_len={ctx_len} != geometry ctx_len={}",
            cfg.ctx_len
        )));
    }

    // Denoiser: exact inventory, in canonical order.
    let expected = cfg.expected_tensors();
    let specs: BTreeMap<String, crate::pth::PthTensorSpec> = weights
        .flatten_tensors()
        .into_iter()
        .map(|(k, s)| (k, s.clone()))
        .collect();
    let mut missing = Vec::new();
    for (name, shape) in &expected {
        match specs.get(name) {
            None => missing.push(name.clone()),
            Some(spec) if as_usize_shape(&spec.shape) != *shape => {
                return Err(Error::Unsupported(format!(
                    "supra2: `{name}` has shape {:?}, expected {shape:?}",
                    spec.shape
                )))
            }
            Some(_) => {}
        }
    }
    if !missing.is_empty() {
        return Err(Error::MissingTensor(format!(
            "supra2 checkpoint lacks {missing:?}"
        )));
    }
    let expected_names: std::collections::BTreeSet<&str> =
        expected.iter().map(|(n, _)| n.as_str()).collect();
    let unexpected: Vec<&String> = specs
        .keys()
        .filter(|k| !expected_names.contains(k.as_str()))
        .collect();
    if !unexpected.is_empty() {
        return Err(Error::Unsupported(format!(
            "supra2: checkpoint has unexpected tensors {unexpected:?}"
        )));
    }
    let mut denoiser = Vec::with_capacity(expected.len());
    for (name, shape) in &expected {
        let t = archive.read_tensor(name, &specs[name])?;
        let data = tensor_f32_bytes(name, &t)?;
        if !all_finite(&data) {
            return Err(Error::Unsupported(format!(
                "supra2: `{name}` has non-finite values"
            )));
        }
        denoiser.push((name.clone(), shape.clone(), data));
    }

    // Stored unconditional conditioning.
    let uncond = match (
        ckpt_cfg.as_ref().and_then(|c| c.get("uncond_text")),
        ckpt_cfg.as_ref().and_then(|c| c.get("uncond_mask")),
    ) {
        (None, None) => None,
        (Some(PthValue::Tensor(text)), Some(PthValue::Tensor(mask))) => {
            if as_usize_shape(&text.shape) != [cfg.ctx_len, cfg.ctx_dim]
                || as_usize_shape(&mask.shape) != [cfg.ctx_len]
            {
                return Err(Error::Unsupported(format!(
                    "supra2: stored unconditional shapes {:?}/{:?}, expected [{}, {}]/[{}]",
                    text.shape, mask.shape, cfg.ctx_len, cfg.ctx_dim, cfg.ctx_len
                )));
            }
            let text_bytes =
                tensor_f32_bytes("uncond_text", &archive.read_tensor("uncond_text", text)?)?;
            if !all_finite(&text_bytes) {
                return Err(Error::Unsupported(
                    "supra2: stored uncond_text has non-finite values".into(),
                ));
            }
            let mask_values = archive.read_tensor("uncond_mask", mask)?.to_f32()?;
            if mask_values.iter().any(|&m| m != 0.0 && m != 1.0)
                || !mask_values.contains(&1.0)
            {
                return Err(Error::Unsupported(
                    "supra2: stored uncond_mask must be 0/1 and select at least one token".into(),
                ));
            }
            let mask_bytes: Vec<u8> = mask_values.iter().flat_map(|v| v.to_le_bytes()).collect();
            Some(vec![
                (
                    "uncond_text".to_string(),
                    vec![cfg.ctx_len, cfg.ctx_dim],
                    text_bytes,
                ),
                ("uncond_mask".to_string(), vec![cfg.ctx_len], mask_bytes),
            ])
        }
        _ => {
            return Err(Error::Unsupported(
                "supra2: checkpoint config must store both uncond_text and uncond_mask tensors, or neither"
                    .into(),
            ))
        }
    };

    std::fs::create_dir_all(out_dir)?;
    let _ = std::fs::remove_file(out_dir.join(BUNDLE_MANIFEST));
    write_safetensors(&out_dir.join(DENOISER_FILE), &denoiser)?;
    drop(denoiser);
    let conditioning = match uncond {
        Some(tensors) => {
            write_safetensors(&out_dir.join(CONDITIONING_FILE), &tensors)?;
            Some(bundle_file(out_dir, CONDITIONING_FILE)?)
        }
        None => {
            let _ = std::fs::remove_file(out_dir.join(CONDITIONING_FILE));
            None
        }
    };
    let component = |id: &str, pin: &ComponentPin| BundleComponent {
        component: id.to_string(),
        repo: pin.repo.to_string(),
        revision: pin.revision.to_string(),
        license: pin.license.to_string(),
        download_bytes: pin.download_bytes(),
    };
    let manifest = BundleManifest {
        schema: BUNDLE_SCHEMA.to_string(),
        provider_id: PROVIDER_ID.to_string(),
        source: BundleSource {
            repo: matches_pin.then(|| SUPRA2_SOURCE.repo.to_string()),
            revision: matches_pin.then(|| SUPRA2_SOURCE.revision.to_string()),
            file: source
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            bytes: source_bytes,
            sha256: source_sha,
            weights_key: weights_key.to_string(),
        },
        config: cfg,
        vae_scale: VAE_SCALE,
        denoiser: bundle_file(out_dir, DENOISER_FILE)?,
        denoiser_tensors: expected.len(),
        conditioning,
        components: vec![
            component(TEXT_ENCODER_COMPONENT, &FLAN_T5_BASE),
            component(VAE_COMPONENT, &SD_VAE),
        ],
    };
    let json = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| Error::Msg(format!("supra2: manifest: {e}")))?;
    let tmp = out_dir.join(format!("{BUNDLE_MANIFEST}.partial"));
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, out_dir.join(BUNDLE_MANIFEST))?;
    Ok(manifest)
}

/// Read and validate a converted bundle: schema, provider id, geometry, and the size and SHA-256 of
/// every converted file (so a truncated or edited bundle never loads).
pub fn read_bundle(dir: &Path) -> Result<Supra2Bundle> {
    let path = dir.join(BUNDLE_MANIFEST);
    let bytes = std::fs::read(&path).map_err(|e| {
        Error::Unsupported(format!(
            "supra2: {} is not a converted Supra2 bundle ({e}); run the install-time converter",
            dir.display()
        ))
    })?;
    let manifest: BundleManifest = serde_json::from_slice(&bytes)
        .map_err(|e| Error::Unsupported(format!("supra2: invalid {}: {e}", path.display())))?;
    if manifest.schema != BUNDLE_SCHEMA || manifest.provider_id != PROVIDER_ID {
        return Err(Error::Unsupported(format!(
            "supra2: {} declares schema `{}` / provider `{}`, expected `{BUNDLE_SCHEMA}` / `{PROVIDER_ID}`",
            path.display(),
            manifest.schema,
            manifest.provider_id
        )));
    }
    manifest.config.validate()?;
    if manifest.vae_scale != VAE_SCALE
        || manifest.denoiser_tensors != manifest.config.expected_tensors().len()
    {
        return Err(Error::Unsupported(format!(
            "supra2: {} declares an inconsistent VAE scale or tensor count",
            path.display()
        )));
    }
    for file in std::iter::once(&manifest.denoiser).chain(manifest.conditioning.as_ref()) {
        if file.file.contains('/') || file.file.contains('\\') || file.file.contains("..") {
            return Err(Error::Unsupported(format!(
                "supra2: bundle file `{}` must be a plain file name",
                file.file
            )));
        }
        let actual = bundle_file(dir, &file.file)?;
        if actual != *file {
            return Err(Error::Unsupported(format!(
                "supra2: bundle file {} does not match its manifest (bytes {} vs {}, sha256 {} vs {})",
                dir.join(&file.file).display(),
                actual.bytes,
                file.bytes,
                actual.sha256,
                file.sha256
            )));
        }
    }
    Ok(Supra2Bundle {
        dir: dir.to_path_buf(),
        manifest,
    })
}

// ------------------------------------------------------------------------------------------------
// Tokenization
// ------------------------------------------------------------------------------------------------

/// One tokenized prompt, padded to the context length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Supra2Tokens {
    /// `ctx_len` ids (right-padded with the pad id).
    pub ids: Vec<i32>,
    /// `ctx_len` attention mask (`1` = real token).
    pub mask: Vec<i32>,
    /// Untruncated token count including `</s>` (upstream's `prompt tokens=` log line).
    pub prompt_tokens: usize,
    /// Whether the prompt exceeded the context and was truncated (upstream's WARNING).
    pub truncated: bool,
}

/// Flan-T5 tokenization with upstream's policy: `padding="max_length", truncation=True,
/// max_length=ctx_len`. Truncation runs **before** the post-processor, so a truncated prompt still
/// ends in `</s>` — exactly what the HF fast tokenizer does.
pub struct Supra2Tokenizer {
    inner: tokenizers::Tokenizer,
    pad_id: i32,
}

fn tok_err(e: tokenizers::Error) -> Error {
    Error::Msg(format!("supra2 tokenizer: {e}"))
}

impl Supra2Tokenizer {
    /// Load `tokenizer.json` from the Flan-T5 component directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_file(&dir.join("tokenizer.json"))
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let mut inner = tokenizers::Tokenizer::from_file(path).map_err(tok_err)?;
        inner.with_padding(None);
        inner.with_truncation(None).map_err(tok_err)?;
        let pad_id = inner
            .token_to_id("<pad>")
            .ok_or_else(|| Error::Msg("supra2 tokenizer: no `<pad>` token".into()))?
            as i32;
        Ok(Self { inner, pad_id })
    }

    /// Tokenize one prompt to exactly `ctx_len` ids + mask.
    pub fn encode(&self, prompt: &str, ctx_len: usize) -> Result<Supra2Tokens> {
        let full = self.inner.encode(prompt, true).map_err(tok_err)?;
        let prompt_tokens = full.get_ids().len();
        let ids: Vec<i32> = if prompt_tokens > ctx_len {
            let mut truncating = self.inner.clone();
            truncating
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length: ctx_len,
                    strategy: tokenizers::TruncationStrategy::LongestFirst,
                    direction: tokenizers::TruncationDirection::Right,
                    stride: 0,
                }))
                .map_err(tok_err)?;
            let enc = truncating.encode(prompt, true).map_err(tok_err)?;
            enc.get_ids().iter().map(|&i| i as i32).collect()
        } else {
            full.get_ids().iter().map(|&i| i as i32).collect()
        };
        if ids.len() > ctx_len {
            return Err(Error::Msg(format!(
                "supra2 tokenizer: produced {} ids for a {ctx_len}-token context",
                ids.len()
            )));
        }
        let real = ids.len();
        let mut ids = ids;
        ids.resize(ctx_len, self.pad_id);
        let mut mask = vec![1; real];
        mask.resize(ctx_len, 0);
        Ok(Supra2Tokens {
            ids,
            mask,
            prompt_tokens,
            truncated: prompt_tokens > ctx_len,
        })
    }
}

// ------------------------------------------------------------------------------------------------
// Request policy + sampling math
// ------------------------------------------------------------------------------------------------

/// A validated request with the published defaults applied.
#[derive(Clone, Debug, PartialEq)]
pub struct Supra2Resolved {
    pub steps: u32,
    pub guidance: f32,
    /// `guidance > 1.0` — upstream's `use_cfg`.
    pub use_cfg: bool,
    /// Seed of image 0; image `i` uses [`image_seed`]`(seed, i)`.
    pub seed: u64,
    pub count: u32,
}

impl Supra2Resolved {
    /// Apply defaults and refuse what the published interface does not have. Call after the
    /// descriptor's shared `Capabilities::validate_request` floor.
    pub fn resolve(req: &GenerationRequest) -> Result<Self> {
        if req.width != IMAGE_SIZE || req.height != IMAGE_SIZE {
            return Err(Error::Unsupported(format!(
                "supra2: native resolution is {IMAGE_SIZE}x{IMAGE_SIZE} only (got {}x{})",
                req.width, req.height
            )));
        }
        if req
            .negative_prompt
            .as_deref()
            .is_some_and(|n| !n.is_empty())
        {
            return Err(Error::Unsupported(
                "supra2: negative prompts are not part of the published interface (CFG uses the checkpoint's unconditional embedding)".into(),
            ));
        }
        if !req.prompt_batch.is_empty() || !req.conditioning.is_empty() {
            return Err(Error::Unsupported(
                "supra2: one text prompt per request; prompt batches and image conditioning are unsupported".into(),
            ));
        }
        let steps = req.steps.unwrap_or(DEFAULT_STEPS);
        if steps == 0 {
            return Err(Error::Unsupported(
                "supra2: steps must be at least 1".into(),
            ));
        }
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        if !guidance.is_finite() {
            return Err(Error::Unsupported("supra2: guidance must be finite".into()));
        }
        let count = req.count.max(1);
        if count > MAX_IMAGES {
            return Err(Error::Unsupported(format!(
                "supra2: at most {MAX_IMAGES} images per request (got {count})"
            )));
        }
        Ok(Self {
            steps,
            guidance,
            use_cfg: cfg_enabled(guidance),
            seed: req.seed.unwrap_or(DEFAULT_SEED),
            count,
        })
    }
}

/// Upstream's guidance branching: CFG runs only when `cfg > 1.0`.
pub fn cfg_enabled(guidance: f32) -> bool {
    guidance > 1.0
}

/// `v_uncond + g · (v_cond − v_uncond)` (host reference of the device-side mix).
pub fn cfg_mix(v_cond: f32, v_uncond: f32, guidance: f32) -> f32 {
    v_uncond + guidance * (v_cond - v_uncond)
}

/// Euler step size `1 / steps` (f32, as torch broadcasts the Python float into the f32 latent).
pub fn euler_dt(steps: u32) -> f32 {
    (1.0f64 / steps as f64) as f32
}

/// Uniform flow times `t_i = i · (1 / steps)`, `i = 0..steps` (noise at `t = 0`, data at `t = 1`):
/// upstream's `torch.full((n,), i * dt)` with `dt` a Python float.
pub fn euler_timesteps(steps: u32) -> Vec<f32> {
    let dt = 1.0f64 / steps as f64;
    (0..steps).map(|i| (i as f64 * dt) as f32).collect()
}

/// `TimestepEmbedder._sinusoidal(t)` for one scalar `t`: `[cos(args), sin(args)]` with
/// `freqs_j = exp(−ln(10000) · j / half)` and `args = t · freqs · 1000`, in float32 as torch runs it.
pub fn timestep_features(t: f32, freq_dim: usize) -> Vec<f32> {
    let half = freq_dim / 2;
    let neg_log = (-(10000.0f64.ln())) as f32;
    let mut cos = Vec::with_capacity(half);
    let mut sin = Vec::with_capacity(half);
    for j in 0..half {
        let freq = (neg_log * j as f32 / half as f32).exp();
        let arg = t * freq * 1000.0f32;
        cos.push(arg.cos());
        sin.push(arg.sin());
    }
    cos.extend(sin);
    if freq_dim % 2 == 1 {
        cos.push(0.0);
    }
    cos
}

/// Seed of image `index` in a request seeded with `seed`: each output asset is reproducible on its
/// own (`seed + index`).
pub fn image_seed(seed: u64, index: u32) -> u64 {
    seed.wrapping_add(u64::from(index))
}

/// Deterministic standard-normal initial latent for one image (SplitMix64 + Box–Muller), identical
/// on every backend so MLX and Candle integrate the same `z0` for the same seed. Upstream draws from
/// torch's CPU/CUDA generators, which neither native backend reproduces; seeds are therefore
/// reproducible per backend-neutral generator, not bit-compatible with `torch.manual_seed`.
pub fn initial_noise(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let r = (-2.0 * uniform().max(1e-300).ln()).sqrt();
        let theta = std::f64::consts::TAU * uniform();
        out.push((r * theta.cos()) as f32);
        if out.len() < len {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

/// Upstream's pixel path: `(x.clamp(-1, 1) + 1) / 2`, then `torchvision.utils.save_image`'s
/// `mul(255).add_(0.5).clamp_(0, 255).to(uint8)`.
pub fn quantize_pixel(x: f32) -> u8 {
    let unit = (x.clamp(-1.0, 1.0) + 1.0) / 2.0;
    (unit * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Convert one decoded NHWC `[H, W, 3]` image (VAE output range ~[-1, 1]) to interleaved RGB8.
pub fn quantize_rgb(decoded_hwc: &[f32]) -> Vec<u8> {
    decoded_hwc.iter().map(|&x| quantize_pixel(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_geometry_matches_the_model_card() {
        let c = Supra2Config::published();
        assert_eq!(c.num_tokens(), 256);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.expected_tensors().len(), 237);
        // Model card: 104.1M parameters (`sum(p.numel() for p in SupraDiT().parameters())`).
        assert_eq!(c.parameter_count(), 104_094_736);
    }

    #[test]
    fn schedule_matches_upstream_euler_loop() {
        let t = euler_timesteps(4);
        assert_eq!(t, vec![0.0, 0.25, 0.5, 0.75]);
        assert_eq!(euler_dt(4), 0.25);
        let t50 = euler_timesteps(50);
        assert_eq!(t50.len(), 50);
        assert_eq!(t50[49], (49.0f64 * (1.0 / 50.0)) as f32);
    }

    #[test]
    fn guidance_branching_is_strictly_above_one() {
        assert!(!cfg_enabled(1.0));
        assert!(!cfg_enabled(0.5));
        assert!(cfg_enabled(1.0001));
        assert_eq!(cfg_mix(2.0, 1.0, 3.0), 4.0);
    }

    #[test]
    fn timestep_features_at_zero_are_cos_one_sin_zero() {
        let f = timestep_features(0.0, 256);
        assert_eq!(f.len(), 256);
        assert!(f[..128].iter().all(|&v| v == 1.0));
        assert!(f[128..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn noise_is_deterministic_standard_normal_per_image_seed() {
        let a = initial_noise(image_seed(0, 1), 4096);
        assert_eq!(a, initial_noise(1, 4096));
        assert_ne!(a, initial_noise(0, 4096));
        let mean = a.iter().map(|&v| v as f64).sum::<f64>() / a.len() as f64;
        let var = a.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(
            mean.abs() < 0.05 && (var - 1.0).abs() < 0.08,
            "{mean} {var}"
        );
    }

    #[test]
    fn pixel_quantization_matches_save_image() {
        assert_eq!(quantize_pixel(-2.0), 0);
        assert_eq!(quantize_pixel(1.5), 255);
        assert_eq!(quantize_pixel(0.0), 128); // 127.5 + 0.5 truncates to 128
        assert_eq!(quantize_pixel(-1.0 + 2.0 / 255.0 * 0.49), 0);
    }

    #[test]
    fn resolve_applies_published_defaults_and_refuses_off_interface_requests() {
        let base = GenerationRequest {
            prompt: "x".into(),
            width: 256,
            height: 256,
            count: 1,
            ..Default::default()
        };
        let r = Supra2Resolved::resolve(&base).unwrap();
        assert_eq!(
            (r.steps, r.guidance, r.use_cfg, r.seed, r.count),
            (50, 3.0, true, 0, 1)
        );
        let off = Supra2Resolved::resolve(&GenerationRequest {
            guidance: Some(1.0),
            ..base.clone()
        })
        .unwrap();
        assert!(!off.use_cfg);
        for bad in [
            GenerationRequest {
                width: 512,
                ..base.clone()
            },
            GenerationRequest {
                negative_prompt: Some("blurry".into()),
                ..base.clone()
            },
            GenerationRequest {
                steps: Some(0),
                ..base.clone()
            },
            GenerationRequest {
                count: MAX_IMAGES + 1,
                ..base.clone()
            },
            GenerationRequest {
                prompt_batch: vec!["a".into()],
                ..base.clone()
            },
        ] {
            assert!(matches!(
                Supra2Resolved::resolve(&bad),
                Err(Error::Unsupported(_))
            ));
        }
    }

    #[test]
    fn component_pins_declare_download_footprint() {
        assert_eq!(SUPRA2_SOURCE.download_bytes(), 416_651_529);
        assert_eq!(FLAN_T5_BASE.download_bytes(), 992_770_529);
        let vae_bytes = SD_VAE.download_bytes();
        assert_eq!(vae_bytes, 334_643_823);
    }
}
