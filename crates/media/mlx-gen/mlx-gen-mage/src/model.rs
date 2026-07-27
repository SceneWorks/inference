//! The six Mage-Flow variants, their [`ModelDescriptor`]s, and the explicit registration
//! constants and the registered RL [`Generator`] implementation.
//!
//! ## The variant matrix
//!
//! `microsoft` publishes **six** repositories whose `transformer/`, `vae/`, `scheduler/` and
//! `text_encoder/` JSON configs are byte-identical; only the transformer *weights* and the
//! README's default `steps`/`cfg` differ. **No config flag distinguishes a variant, or even
//! generation from editing** — the edit path differs purely by input-sequence assembly, so the
//! same backbone serves both. Turbo and Edit-Turbo are full distilled checkpoints, not LoRAs.
//!
//! | id | repo | task | steps | cfg |
//! | --- | --- | --- | --- | --- |
//! | `mage_flow` | `microsoft/Mage-Flow` (RL) | gen | 20 | 5.0 |
//! | `mage_flow_base` | `microsoft/Mage-Flow-Base` | gen | 30 | 5.0 |
//! | `mage_flow_turbo` | `microsoft/Mage-Flow-Turbo` | gen | 4 | 1.0 (off) |
//! | `mage_flow_edit` | `microsoft/Mage-Flow-Edit` | edit | 30 | 5.0 |
//! | `mage_flow_edit_base` | `microsoft/Mage-Flow-Edit-Base` | edit | 30 | 5.0 |
//! | `mage_flow_edit_turbo` | `microsoft/Mage-Flow-Edit-Turbo` | edit | 4 | 1.0 (off) |
//!
//! Each variant has its own id (rather than one id plus a switch), keeping the variant part
//! of the worker's model cache key.
//!
//! All generation and edit IDs are composed into the shipped platform catalog after their owning
//! stories validated the shared production paths and checkpoint-specific defaults.
//!
//! [`mlx_gen_catalog::provider_registry`]: https://docs.rs/mlx-gen-catalog

use mlx_gen::{
    Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant, Result,
    WeightsSource,
};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::config::{FAMILY, MAX_SIZE, MIN_SIZE, SIZE_MULTIPLE};
use crate::pipeline::MageComponentDirs;
use crate::{resolve_gs_key, GenerationSample, MageFlowPipeline};

/// Which published checkpoint a registered id serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MageVariant {
    /// `microsoft/Mage-Flow` — the Diffusion-NFT RL checkpoint, 20 steps.
    Rl,
    /// `microsoft/Mage-Flow-Base` — the training target, 30 steps.
    Base,
    /// `microsoft/Mage-Flow-Turbo` — Decoupled-DMD distilled, 4 steps, CFG off.
    Turbo,
    /// `microsoft/Mage-Flow-Edit` — instruction editing, 30 steps.
    Edit,
    /// `microsoft/Mage-Flow-Edit-Base` — instruction editing, 30 steps.
    EditBase,
    /// `microsoft/Mage-Flow-Edit-Turbo` — distilled instruction editing, 4 steps, CFG off.
    EditTurbo,
}

impl MageVariant {
    /// Registry id — always prefixed with [`FAMILY`].
    pub const fn id(self) -> &'static str {
        match self {
            Self::Rl => "mage_flow",
            Self::Base => "mage_flow_base",
            Self::Turbo => "mage_flow_turbo",
            Self::Edit => "mage_flow_edit",
            Self::EditBase => "mage_flow_edit_base",
            Self::EditTurbo => "mage_flow_edit_turbo",
        }
    }

    /// The upstream Hugging Face repository this variant's weights come from.
    pub const fn upstream_repo(self) -> &'static str {
        match self {
            Self::Rl => "microsoft/Mage-Flow",
            Self::Base => "microsoft/Mage-Flow-Base",
            Self::Turbo => "microsoft/Mage-Flow-Turbo",
            Self::Edit => "microsoft/Mage-Flow-Edit",
            Self::EditBase => "microsoft/Mage-Flow-Edit-Base",
            Self::EditTurbo => "microsoft/Mage-Flow-Edit-Turbo",
        }
    }

    /// `true` for the instruction-editing checkpoints, which consume reference images.
    pub const fn is_edit(self) -> bool {
        matches!(self, Self::Edit | Self::EditBase | Self::EditTurbo)
    }

    /// `true` for the Decoupled-DMD distilled checkpoints (4 steps, CFG off).
    pub const fn is_distilled(self) -> bool {
        matches!(self, Self::Turbo | Self::EditTurbo)
    }

    /// Published default step count, used when a request omits `steps`.
    pub const fn default_steps(self) -> u32 {
        match self {
            Self::Rl => 20,
            Self::Base | Self::Edit | Self::EditBase => 30,
            Self::Turbo | Self::EditTurbo => 4,
        }
    }

    /// Published default guidance scale. The distilled variants default to **1.0**, at which the
    /// reference builds no unconditional branch at all (`pipeline.py:326`, `:535`) — so CFG is
    /// genuinely off, not merely weightless.
    pub const fn default_cfg(self) -> f32 {
        if self.is_distilled() {
            1.0
        } else {
            5.0
        }
    }

    /// Every variant, in registration order.
    pub const ALL: [MageVariant; 6] = [
        Self::Rl,
        Self::Base,
        Self::Turbo,
        Self::Edit,
        Self::EditBase,
        Self::EditTurbo,
    ];
}

/// Every registered Mage-Flow id, in registration order.
pub const MODEL_IDS: [&str; 6] = [
    "mage_flow",
    "mage_flow_base",
    "mage_flow_turbo",
    "mage_flow_edit",
    "mage_flow_edit_base",
    "mage_flow_edit_turbo",
];

/// Maximum homogeneous output count exposed through the platform request surface.
pub const MAX_COUNT: u32 = 8;

/// Immutable upstream revision used to establish the Turbo checkpoint fingerprint.
pub const TURBO_SNAPSHOT_REVISION: &str = "8523c9d1ae3cbe2148241e4769c918d0ab158ef8";
/// Immutable upstream revision used to establish the Base checkpoint fingerprint.
pub const BASE_SNAPSHOT_REVISION: &str = "59a9cfd58cf6ecef28245852c6bdace3f12428a2";
/// Immutable upstream revision used to establish the Edit-Base checkpoint fingerprint.
pub const EDIT_BASE_SNAPSHOT_REVISION: &str = "8654a7bc0283ab2946385230b5b2eb944e0b76ea";
/// Immutable upstream revision used to establish the Edit-Turbo checkpoint fingerprint.
pub const EDIT_TURBO_SNAPSHOT_REVISION: &str = "14427bd7627d3a25436497a5939e1096f6a0d523";
/// Immutable upstream revision used to establish the primary Edit checkpoint fingerprint.
pub const EDIT_SNAPSHOT_REVISION: &str = "b01d524f86498b7dabcc4b3572c6d264d786a16e";
// Every identity tensor is a **bias**, deliberately (sc-14980). Biases are never quantized, so a
// bias fingerprint is byte-identical in the dense flat snapshot and in every pre-quantized
// `<tier>/transformer/` artifact — one pinned hash per variant verifies all three tiers, with no
// per-tier constant and no weakening of the check on the packed path.
//
// Turbo previously pinned `img_in.weight`, which the Q4/Q8 packs rewrite into u32 codes; that hash
// could not survive a tier artifact. `transformer_blocks.0.attn.add_k_proj.bias` replaces it and is
// strictly stronger: measured over all six published checkpoints its first 4096 bytes yield **six
// distinct** digests, so this one tensor discriminates every variant (`img_in.bias`, by contrast,
// collides Base with RL and Edit-Base with Edit).
const TURBO_IDENTITY_TENSOR: &str = "transformer_blocks.0.attn.add_k_proj.bias";
const BASE_IDENTITY_TENSOR: &str = "transformer_blocks.0.attn.add_k_proj.bias";
const EDIT_IDENTITY_TENSOR: &str = "transformer_blocks.0.attn.add_k_proj.bias";
const TURBO_IDENTITY_BYTES: usize = 4096;
const BASE_IDENTITY_BYTES: usize = 4096;
const EDIT_IDENTITY_BYTES: usize = 4096;
const TURBO_IDENTITY_SHA256: &str =
    "52d3e3d2bcbb655f4575b71757081da3406dd13e5c58ef73173e070ff1c4767f";
const BASE_IDENTITY_SHA256: &str =
    "c6597b08e4efe45f7bbb5d2470c68e7975d71ca26dce13a1fb34db18ca6a9e3e";
const EDIT_BASE_IDENTITY_SHA256: &str =
    "bb53a04c20e5df443bb093c3f24027f9391f6d65e3edd60ed96546b050db717b";
const EDIT_TURBO_IDENTITY_SHA256: &str =
    "d387be05845ea0e0fc6b2bec5c05bccb3808c25a0123d9e2b3459e2e7f9705df";
const EDIT_IDENTITY_SHA256: &str =
    "bd24b2009764136298499d60750ded8ebdfa7950981d116e9937588471b2ecab";

/// Build a variant's weights-free descriptor.
///
/// Capability fields that later stories own are left at their conservative `Default` (`false` /
/// empty) rather than pre-announced: quant tiers are sc-14046, LoRA/LoKr routing is sc-14057, and
/// the curated sampler/scheduler menus are sc-14041's once the flow-match loop exists. A
/// descriptor is a promise to the worker, so the scaffold promises only what it can point at in
/// the published configs.
pub fn descriptor_for(variant: MageVariant) -> ModelDescriptor {
    ModelDescriptor {
        // The text encoder (8.875 GB) and VAE (0.345 GB) are BIT-IDENTICAL across all six Mage
        // variants — only the 8.232 GB DiT differs — so the SceneWorks mirrors host them once in a
        // shared components repo and stage them as caller-provisioned co-requisite dirs
        // (sc-14979): 58.65 GB for a full six-variant install instead of 105.04 GB. The DiT still
        // arrives as the base `WeightsSource::Dir`. A spec that stages neither falls back to the
        // flat published layout — see `resolve_component_dirs`.
        required_components: REQUIRED_COMPONENTS,
        id: variant.id(),
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Real CFG (`guidance_embed: false`) on the undistilled checkpoints; the distilled
            // ones run at cfg 1.0, where the reference never builds the negative branch.
            supports_negative_prompt: !variant.is_distilled(),
            supports_guidance: !variant.is_distilled(),
            conditioning: if variant.is_edit() {
                vec![
                    ConditioningKind::Reference,
                    ConditioningKind::MultiReference,
                ]
            } else {
                Vec::new()
            },
            // Q4/Q8 tiers are sc-14046; `&[]` means dense-only, which is what the scaffold is.
            supported_quants: &[Quant::Q4, Quant::Q8],
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            // A platform request has one geometry/prompt and `count` independent seeds. The
            // pipeline additionally exposes heterogeneous geometry/prompt packs directly.
            max_count: MAX_COUNT,
            mac_only: true,
            ..Default::default()
        },
    }
}

/// Construct a Mage-Flow generator from a [`LoadSpec`].
///
pub fn load(variant: MageVariant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 || !spec.adapters.is_empty() {
        return Err(Error::Unsupported(
            "mage_flow variants support bf16/Q4/Q8 checkpoints without adapters".into(),
        ));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "mage_flow expects a diffusers snapshot directory".into(),
            ))
        }
    };
    match variant {
        MageVariant::Base => verify_checkpoint_identity(
            root,
            variant,
            BASE_SNAPSHOT_REVISION,
            BASE_IDENTITY_TENSOR,
            BASE_IDENTITY_BYTES,
            &[3072],
            BASE_IDENTITY_SHA256,
        )?,
        MageVariant::Turbo => verify_checkpoint_identity(
            root,
            variant,
            TURBO_SNAPSHOT_REVISION,
            TURBO_IDENTITY_TENSOR,
            TURBO_IDENTITY_BYTES,
            &[3072],
            TURBO_IDENTITY_SHA256,
        )?,
        MageVariant::EditBase => verify_checkpoint_identity(
            root,
            variant,
            EDIT_BASE_SNAPSHOT_REVISION,
            EDIT_IDENTITY_TENSOR,
            EDIT_IDENTITY_BYTES,
            &[3072],
            EDIT_BASE_IDENTITY_SHA256,
        )?,
        MageVariant::Edit => verify_checkpoint_identity(
            root,
            variant,
            EDIT_SNAPSHOT_REVISION,
            EDIT_IDENTITY_TENSOR,
            EDIT_IDENTITY_BYTES,
            &[3072],
            EDIT_IDENTITY_SHA256,
        )?,
        MageVariant::EditTurbo => verify_checkpoint_identity(
            root,
            variant,
            EDIT_TURBO_SNAPSHOT_REVISION,
            EDIT_IDENTITY_TENSOR,
            EDIT_IDENTITY_BYTES,
            &[3072],
            EDIT_TURBO_IDENTITY_SHA256,
        )?,
        _ => {}
    }
    let dirs = resolve_component_dirs(root, spec)?;
    assemble(variant, spec, dirs)
}

/// Construct a Mage-Flow generator from a caller-owned **fine-tuned transformer** (sc-15036,
/// epic 14034 F6) — the artifact a full base fine-tune (sc-14056) writes.
///
/// Two things distinguish this from [`load`], and both are forced by what a fine-tune *is*:
///
/// 1. **`spec.weights` is the fine-tuned `transformer/` component directory itself** (a
///    `config.json` + `diffusion_pytorch_model.safetensors` pair, exactly what the trainer's
///    `save_full_checkpoint` emits), NOT a diffusers snapshot root. A training run produces the
///    DiT alone; it never re-emits the text encoder or VAE, so there is no snapshot root to point
///    at and no flat-layout sibling to fall back to. Both shared components must therefore be
///    caller-staged in [`LoadSpec::components`] — normally the installed base model's own
///    `text_encoder/` + `vae/`, which a fine-tune leaves untouched and is numerically paired with
///    by construction. A missing one is a typed error here rather than a mid-load "No such file".
/// 2. **The pinned-checkpoint identity verification is skipped.** That guard exists to catch one
///    *published* variant's snapshot staged under another published variant's id, and it works by
///    hashing a prefix of `transformer_blocks.0.attn.add_k_proj.bias`. A full fine-tune trains
///    every DiT weight including that bias, so the guard would reject the user's own trained
///    checkpoint **by construction** — it cannot distinguish "fine-tuned from Base" from "the
///    wrong published checkpoint", because the caller is the only one who knows. `variant` states
///    which published checkpoint the run started from, and with it the architecture, the
///    sampling regime (steps / CFG / distillation) and the edit-vs-generate input assembly the
///    fine-tune inherits.
///
/// Deliberately kept off the registry `load(id, spec)` path (like Krea's
/// `load_from_native_dit_file`): a fine-tune is a caller-owned artifact at an arbitrary path, not
/// a published id, so it is reached through this explicit API rather than by resolving an id.
pub fn load_finetuned(variant: MageVariant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 || !spec.adapters.is_empty() {
        return Err(Error::Unsupported(
            "mage_flow fine-tuned checkpoints load as bf16/Q4/Q8 without adapters".into(),
        ));
    }
    let transformer = match &spec.weights {
        WeightsSource::Dir(dir) => dir.clone(),
        WeightsSource::File(file) => {
            return Err(Error::Msg(format!(
                "mage_flow: a fine-tuned checkpoint is a transformer DIRECTORY (config.json + \
                 diffusion_pytorch_model.safetensors), got the file {}",
                file.display()
            )))
        }
    };
    mlx_gen::gen_core::reject_unknown_components(spec, REQUIRED_COMPONENTS, FAMILY)?;
    let staged = |id: &str| -> Result<std::path::PathBuf> {
        match spec.components.get(id) {
            Some(WeightsSource::Dir(dir)) => Ok(dir.clone()),
            Some(WeightsSource::File(file)) => Err(Error::Msg(format!(
                "mage_flow: the '{id}' component must be staged as a directory, got the file {}",
                file.display()
            ))),
            // No flat-layout fallback: a fine-tune dir has no component siblings, so silently
            // probing `<transformer>/text_encoder` would only turn a staging bug into a confusing
            // deep load failure.
            None => Err(Error::Msg(format!(
                "mage_flow: loading a fine-tuned transformer requires the '{id}' component to be \
                 staged from the installed base model — a training run produces the transformer \
                 alone"
            ))),
        }
    };
    let dirs = MageComponentDirs {
        transformer,
        text_encoder: staged(COMPONENT_TEXT_ENCODER)?,
        vae: staged(COMPONENT_VAE)?,
    };
    assemble(variant, spec, dirs)
}

/// Build the pipeline + generator from already-resolved component dirs — the half [`load`] and
/// [`load_finetuned`] share once each has decided *where* the components live.
fn assemble(
    variant: MageVariant,
    spec: &LoadSpec,
    dirs: MageComponentDirs,
) -> Result<Box<dyn Generator>> {
    let part = if variant.is_edit() {
        crate::vae::VaePart::Both
    } else {
        crate::vae::VaePart::Decode
    };
    let pipeline = MageFlowPipeline::load_components(&dirs, spec.quantize.map(Quant::bits), part)?;
    Ok(Box::new(MageFlow {
        variant,
        descriptor: descriptor_for(variant),
        tier: spec.quantize,
        pipeline,
    }))
}

/// The caller-provisioned component ids Mage-Flow advertises (sc-14979).
pub const COMPONENT_TEXT_ENCODER: &str = "text_encoder";
/// The caller-provisioned VAE component id (sc-14979).
pub const COMPONENT_VAE: &str = "vae";
/// Both shared components, in descriptor order.
pub const REQUIRED_COMPONENTS: &[&str] = &[COMPONENT_TEXT_ENCODER, COMPONENT_VAE];

/// Resolve where each component's weights live for this load.
///
/// **Split layout (the SceneWorks mirrors, sc-14980/sc-14979).** `spec.weights` is the variant's
/// per-tier dir (`<variant snapshot>/<tier>/`), holding the DiT alone; the text encoder and VAE —
/// bit-identical across all six variants — are staged by the caller in [`LoadSpec::components`] as
/// exact component dirs resolved from the shared components mirror. Six installs cost 58.65 GB
/// instead of 105.04 GB.
///
/// **Flat layout (upstream snapshots, existing installs, arbitrary user paths).** No components are
/// staged and every component sits directly under `spec.weights`. This fallback is why the split is
/// not a breaking change: a repo/revision without tier subdirs, and every `#[ignore]`d real-weights
/// test that points at a raw `microsoft/Mage-Flow*` snapshot, keeps loading unchanged.
///
/// The two are distinguished per component, not globally, so a partially-staged spec is still
/// coherent. Unknown component ids are rejected rather than ignored.
pub(crate) fn resolve_component_dirs(root: &Path, spec: &LoadSpec) -> Result<MageComponentDirs> {
    mlx_gen::gen_core::reject_unknown_components(spec, REQUIRED_COMPONENTS, FAMILY)?;
    let staged = |id: &str, fallback: &str| -> Result<std::path::PathBuf> {
        match spec.components.get(id) {
            Some(WeightsSource::Dir(dir)) => Ok(dir.clone()),
            Some(WeightsSource::File(file)) => Err(Error::Msg(format!(
                "mage_flow: the '{id}' component must be staged as a directory, got the file {}",
                file.display()
            ))),
            None => Ok(root.join(fallback)),
        }
    };
    Ok(MageComponentDirs {
        transformer: root.join("transformer"),
        text_encoder: staged(COMPONENT_TEXT_ENCODER, "text_encoder")?,
        vae: staged(COMPONENT_VAE, "vae")?,
    })
}

/// Verify bytes from a weight-bearing tensor, not a path or model-card label.
///
/// All Mage-Flow variants share byte-identical configs and tensor schemas, so those cannot detect
/// one checkpoint accidentally routed under another variant id. The caller supplies a tensor,
/// byte count, and hash pinned to the immutable upstream revision. Base deliberately uses an
/// attention bias because its `img_in.weight` prefix is byte-identical to RL's.
fn verify_checkpoint_identity(
    root: &Path,
    variant: MageVariant,
    revision: &str,
    tensor_name: &str,
    identity_bytes: usize,
    expected_shape: &[u64],
    expected_sha256: &str,
) -> Result<()> {
    let id = variant.id();
    let path = root
        .join("transformer")
        .join("diffusion_pytorch_model.safetensors");
    let mut file = std::fs::File::open(&path).map_err(|error| {
        Error::Msg(format!(
            "{id}: cannot open transformer checkpoint {}: {error}",
            path.display()
        ))
    })?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > 1_048_576 {
        return Err(Error::Msg(format!(
            "{id}: invalid safetensors header length {header_len}"
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let metadata: serde_json::Value = serde_json::from_slice(&header).map_err(|error| {
        Error::Msg(format!(
            "{id}: invalid safetensors header in {}: {error}",
            path.display()
        ))
    })?;
    let tensor = metadata
        .get(tensor_name)
        .ok_or_else(|| Error::Msg(format!("{id}: missing {tensor_name}")))?;
    if tensor.get("dtype").and_then(serde_json::Value::as_str) != Some("BF16")
        || tensor.get("shape").and_then(serde_json::Value::as_array)
            != Some(
                &expected_shape
                    .iter()
                    .map(|&dimension| serde_json::json!(dimension))
                    .collect::<Vec<_>>(),
            )
    {
        return Err(Error::Msg(format!(
            "{id}: {tensor_name} has the wrong dtype or shape"
        )));
    }
    let offsets = tensor
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::Msg(format!("{id}: {tensor_name} has no data offsets")))?;
    let start = offsets.first().and_then(serde_json::Value::as_u64);
    let end = offsets.get(1).and_then(serde_json::Value::as_u64);
    let (Some(start), Some(end)) = (start, end) else {
        return Err(Error::Msg(format!(
            "{id}: {tensor_name} has invalid data offsets"
        )));
    };
    if end.saturating_sub(start) < identity_bytes as u64 {
        return Err(Error::Msg(format!(
            "{id}: {tensor_name} is too short for identity verification"
        )));
    }
    file.seek(SeekFrom::Start(8 + header_len + start))?;
    let mut bytes = vec![0u8; identity_bytes];
    file.read_exact(&mut bytes)?;
    let got = format!("{:x}", Sha256::digest(bytes));
    if got != expected_sha256 {
        return Err(Error::Msg(format!(
            "{id}: checkpoint fingerprint mismatch for {tensor_name} \
             (expected revision {revision}, got sha256 {got}); \
             another Mage-Flow checkpoint cannot serve the {id} id"
        )));
    }
    Ok(())
}

pub struct MageFlow {
    variant: MageVariant,
    descriptor: ModelDescriptor,
    tier: Option<Quant>,
    pipeline: MageFlowPipeline,
}

impl Generator for MageFlow {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_generation_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        crate::memory::ensure_generation_fits(
            self.tier,
            req.width,
            req.height,
            req.count,
            crate::memory::production_safe_budget_gb()?,
        )?;
        if req.cancel.is_cancelled() {
            return Err(mlx_gen::gen_core::Error::Canceled);
        }
        let steps = req.steps.unwrap_or(self.variant.default_steps());
        let cfg = req.guidance.unwrap_or(self.variant.default_cfg());
        let seed = req.seed.unwrap_or(0) as i64;
        let key = resolve_gs_key(None)?;
        let negative_prompt = req.negative_prompt.as_deref().unwrap_or(" ");
        if self.variant.is_edit() {
            let references = edit_references(req)?;
            let mut images = Vec::with_capacity(req.count as usize);
            for index in 0..req.count {
                let trace = self.pipeline.edit_trace(
                    &req.prompt,
                    negative_prompt,
                    &references,
                    req.height,
                    req.width,
                    steps as usize,
                    cfg,
                    seed.wrapping_add(index as i64),
                    &key,
                    false,
                    on_progress,
                )?;
                mlx_rs::transforms::eval([&trace.image_u8]).map_err(Error::from)?;
                images.push(Image {
                    width: req.width,
                    height: req.height,
                    pixels: trace
                        .image_u8
                        .try_as_slice::<u8>()
                        .map_err(|error| {
                            Error::Msg(format!(
                                "mage_flow edit: RGB8 output is not host-readable: {error}"
                            ))
                        })?
                        .to_vec(),
                });
            }
            return Ok(GenerationOutput::Images(images));
        }
        let samples = (0..req.count)
            .map(|index| GenerationSample {
                prompt: &req.prompt,
                negative_prompt,
                height: req.height,
                width: req.width,
                seed: seed.wrapping_add(index as i64),
            })
            .collect::<Vec<_>>();
        let traces = self
            .pipeline
            .generate_batch_trace(&samples, steps as usize, cfg, &key, false, on_progress)?
            .samples;
        let mut images = Vec::with_capacity(traces.len());
        for trace in traces {
            mlx_rs::transforms::eval([&trace.image_u8]).map_err(Error::from)?;
            let pixels = trace
                .image_u8
                .try_as_slice::<u8>()
                .map_err(|e| {
                    Error::Msg(format!("mage_flow: RGB8 output is not host-readable: {e}"))
                })?
                .to_vec();
            images.push(Image {
                width: req.width,
                height: req.height,
                pixels,
            });
        }
        Ok(GenerationOutput::Images(images))
    }
}

fn edit_references(req: &GenerationRequest) -> Result<Vec<image::RgbImage>> {
    let mut images = Vec::new();
    for conditioning in &req.conditioning {
        match conditioning {
            Conditioning::Reference { image, .. } => images.push(image),
            Conditioning::MultiReference { images: refs } => images.extend(refs),
            _ => {}
        }
    }
    if images.is_empty() {
        return Err(Error::Msg(
            "mage_flow edit: Reference or MultiReference conditioning is required".into(),
        ));
    }
    images
        .into_iter()
        .map(|image| {
            image::RgbImage::from_raw(image.width, image.height, image.pixels.clone()).ok_or_else(
                || Error::Msg("mage_flow edit: reference image is not valid RGB8".into()),
            )
        })
        .collect()
}

fn validate_generation_request(
    descriptor: &ModelDescriptor,
    req: &GenerationRequest,
) -> mlx_gen::gen_core::Result<()> {
    descriptor
        .capabilities
        .validate_request(descriptor.id, req)?;
    if !req.width.is_multiple_of(REQUIRED_SIZE_MULTIPLE)
        || !req.height.is_multiple_of(REQUIRED_SIZE_MULTIPLE)
    {
        return Err(mlx_gen::gen_core::Error::Msg(format!(
            "mage_flow dimensions must be divisible by {REQUIRED_SIZE_MULTIPLE}"
        )));
    }
    Ok(())
}

/// Every side of a Mage-Flow request must be a multiple of this (the VAE's 16× downsample;
/// `patch_size == 1` adds no further stride). Re-exported at the model layer because SceneWorks
/// pins each advertised resolution bucket to an engine stride constant.
pub const REQUIRED_SIZE_MULTIPLE: u32 = SIZE_MULTIPLE;

/// Per-component on-disk footprint used by the worker's staged-residency fit gate.
///
/// Mage-Flow quantizes all three weight-bearing components, so the accounting must follow the
/// selected snapshot tree rather than a transformer-only approximation. Missing/unreadable
/// subdirectories contribute zero bytes here; checkpoint identity/load validation separately rejects
/// missing required components before generation.
///
/// **Resolves through [`resolve_component_dirs`], not the spec's weights root** (sc-15154). On the
/// SPLIT layout the root is the variant's per-tier dir and holds the DiT *alone* — the text encoder
/// and VAE are staged in [`LoadSpec::components`] from the shared mirror. Summing subdirs of the
/// root therefore reported the DiT's bytes as the whole model: 2.33 GB for a q4 tier whose real
/// install is 7.00 GB, and 8.23 GB for bf16's 17.46 GB. The worker's fit gate adds a flat activation
/// headroom to this number, so the shortfall surfaced as an over-budget message quoting a figure
/// that tracked neither the tier's weights nor its measured peak. The flat-layout fallback is
/// unchanged: with nothing staged, `resolve_component_dirs` returns `root/<component>` exactly as
/// before.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    let mlx_gen::WeightsSource::Dir(root) = &spec.weights else {
        return Err(mlx_gen::gen_core::Error::Msg(
            "mage_flow: per-component footprint requires a snapshot directory, not a single \
             .safetensors file"
                .to_owned(),
        ));
    };
    let dirs = resolve_component_dirs(root, spec)?;
    Ok(mlx_gen::PerComponentBytes {
        text_encoder: mlx_gen::safetensors_path_bytes(dirs.text_encoder),
        dit: mlx_gen::safetensors_path_bytes(dirs.transformer),
        vae: mlx_gen::safetensors_path_bytes(dirs.vae),
    })
}

macro_rules! mage_registrations {
    ( $( $variant:ident => ( $descriptor_fn:ident, $load_fn:ident, $registration:ident ) ),+ $(,)? ) => {
        $(
            /// This variant's weights-free descriptor (see [`descriptor_for`]).
            pub fn $descriptor_fn() -> ModelDescriptor {
                descriptor_for(MageVariant::$variant)
            }

            fn $load_fn(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
                load(MageVariant::$variant, spec)
            }

            mlx_gen::register_generators! {
                pub const $registration = $descriptor_fn => $load_fn;
                footprint = component_footprint
            }
        )+

        /// The explicit registration constants, in variant order — the surface a catalog crate
        /// composes.
        pub const REGISTRATIONS: &[mlx_gen::registry::ModelRegistration] = &[ $($registration),+ ];
    };
}

mage_registrations! {
    Rl => (descriptor, load_rl, REGISTRATION),
    Base => (descriptor_base, load_base, REGISTRATION_BASE),
    Turbo => (descriptor_turbo, load_turbo, REGISTRATION_TURBO),
    Edit => (descriptor_edit, load_edit, REGISTRATION_EDIT),
    EditBase => (descriptor_edit_base, load_edit_base, REGISTRATION_EDIT_BASE),
    EditTurbo => (descriptor_edit_turbo, load_edit_turbo, REGISTRATION_EDIT_TURBO),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-15154 — the footprint must follow the SPLIT layout's staged components, not the tier dir.
    ///
    /// The discriminating case: the same fake tier tree scanned with and without the components
    /// staged. A footprint that sums subdirs of `spec.weights` scores the split spec at the DiT's
    /// bytes alone and cannot tell the two specs apart, which is exactly what made the worker's
    /// over-budget message quote a figure unrelated to the tier's real install.
    #[test]
    fn the_footprint_counts_staged_components_not_just_the_tier_dir() {
        let root = std::env::temp_dir().join(format!(
            "mage-footprint-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        let write = |dir: std::path::PathBuf, bytes: usize| {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), vec![0u8; bytes]).unwrap();
            dir
        };
        // SPLIT: the variant tier dir holds the DiT; the shared mirror holds the TE + VAE.
        let tier = root.join("q4");
        write(tier.join("transformer"), 300);
        let te = write(root.join("shared/q4/text_encoder"), 700);
        let vae = write(root.join("shared/q4/vae"), 50);

        let split = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(tier.clone()))
            .with_component(COMPONENT_TEXT_ENCODER, mlx_gen::WeightsSource::Dir(te))
            .with_component(COMPONENT_VAE, mlx_gen::WeightsSource::Dir(vae));
        let got = component_footprint(&split).unwrap();
        assert_eq!(
            (got.dit, got.text_encoder, got.vae),
            (300, 700, 50),
            "the staged text encoder and VAE are part of what this tier loads"
        );

        // ...and the same spec with nothing staged sees only the DiT — the pre-fix behavior, kept
        // here so the assertion above is visibly about the staging and not about the tree.
        let unstaged = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(tier));
        let got = component_footprint(&unstaged).unwrap();
        assert_eq!((got.dit, got.text_encoder, got.vae), (300, 0, 0));

        // FLAT (upstream snapshots / legacy installs): every component under the root, nothing
        // staged. Unchanged by this fix.
        let flat = root.join("flat");
        write(flat.join("transformer"), 300);
        write(flat.join("text_encoder"), 700);
        write(flat.join("vae"), 50);
        let got = component_footprint(&mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(flat)))
            .unwrap();
        assert_eq!((got.dit, got.text_encoder, got.vae), (300, 700, 50));

        std::fs::remove_dir_all(root).ok();
    }

    /// A minimal but structurally valid safetensors file carrying exactly the Base identity tensor
    /// (`transformer_blocks.0.attn.add_k_proj.bias`, BF16, `[3072]`) filled with `fill` — enough for
    /// [`verify_checkpoint_identity`] to parse the header, seek, and hash. Nothing loads these
    /// weights; the tests below only exercise which *guard* fires.
    fn load_error(result: Result<Box<dyn Generator>>, context: &str) -> String {
        match result {
            Ok(_) => panic!("{context}"),
            Err(error) => error.to_string(),
        }
    }

    fn write_identity_only_checkpoint(path: &Path, fill: u8) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let payload = 3072 * 2; // BF16 [3072]
        let header = format!(
            "{{\"{BASE_IDENTITY_TENSOR}\":{{\"dtype\":\"BF16\",\"shape\":[3072],\"data_offsets\":[0,{payload}]}}}}"
        );
        let mut bytes = Vec::with_capacity(8 + header.len() + payload);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend(std::iter::repeat_n(fill, payload));
        std::fs::write(path, bytes).unwrap();
    }

    /// sc-15036 — `load_finetuned` must get PAST the pinned-checkpoint identity guard that `load`
    /// enforces, because a full base fine-tune (sc-14056) rewrites every DiT weight *including*
    /// `transformer_blocks.0.attn.add_k_proj.bias`, so `load` rejects the user's own trained
    /// checkpoint by construction.
    ///
    /// Discriminating in both directions on ONE fabricated checkpoint:
    ///   * `load(Base, …)` must fail with the **fingerprint-mismatch** message — delete the guard
    ///     and this half fails;
    ///   * `load_finetuned(Base, …)` must fail with something else entirely (it reaches component
    ///     staging) — route `load_finetuned` back through the guard and this half fails.
    #[test]
    fn load_finetuned_bypasses_the_pinned_checkpoint_identity_guard() {
        let root = std::env::temp_dir().join(format!(
            "mage-finetuned-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        let transformer = root.join("transformer");
        write_identity_only_checkpoint(
            &transformer.join("diffusion_pytorch_model.safetensors"),
            0x5a,
        );

        // `load` sees a snapshot ROOT and hashes `<root>/transformer/…`: the fill is not the pinned
        // Base fingerprint, so the guard fires.
        let published = load_error(
            load(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone())),
            ),
            "a checkpoint whose identity tensor moved must not load as published Base",
        );
        assert!(
            published.contains("checkpoint fingerprint mismatch"),
            "expected the identity guard to fire, got: {published}"
        );

        // `load_finetuned` is handed the SAME root — deliberately, so the mutation "delegate to
        // `load`" is caught: under it this call would report the fingerprint mismatch above. The
        // real entrypoint treats the path as the transformer dir itself and never opens
        // `<path>/transformer`, so it gets past identity and fails later, at the actual load.
        let staged = std::env::temp_dir().join("mage-finetuned-nonexistent-component");
        let finetuned = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone()))
                    .with_component(COMPONENT_TEXT_ENCODER, WeightsSource::Dir(staged.clone()))
                    .with_component(COMPONENT_VAE, WeightsSource::Dir(staged)),
            ),
            "the fabricated checkpoint has no real components to load",
        );
        assert!(
            !finetuned.contains("checkpoint fingerprint"),
            "load_finetuned must not enforce the published-checkpoint fingerprint, got: {finetuned}"
        );
        // ...and the transformer dir it DID read is the one it was handed.
        assert!(
            transformer.is_dir(),
            "fixture sanity: the nested published-layout transformer dir exists"
        );

        std::fs::remove_dir_all(root).ok();
    }

    /// sc-15036 — the shared components are REQUIRED for a fine-tune and there is deliberately no
    /// flat-layout fallback: a training run emits the transformer alone, so probing
    /// `<transformer>/text_encoder` would turn a staging bug into a confusing deep load failure.
    /// Each missing id must be named. Also pins that a FILE weights source is refused (a fine-tune
    /// is a directory).
    #[test]
    fn load_finetuned_requires_both_shared_components_to_be_staged() {
        let root = std::env::temp_dir().join(format!(
            "mage-finetuned-components-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let dir = |name: &str| WeightsSource::Dir(root.join(name));

        let bare = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone())),
            ),
            "no components staged",
        );
        assert!(
            bare.contains(COMPONENT_TEXT_ENCODER),
            "the missing component must be named, got: {bare}"
        );

        let vae_only = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::Dir(root.clone()))
                    .with_component(COMPONENT_TEXT_ENCODER, dir("te")),
            ),
            "the VAE is still missing",
        );
        assert!(
            vae_only.contains(COMPONENT_VAE),
            "the missing VAE must be named, got: {vae_only}"
        );

        let as_file = load_error(
            load_finetuned(
                MageVariant::Base,
                &LoadSpec::new(WeightsSource::File(
                    root.join("diffusion_pytorch_model.safetensors"),
                )),
            ),
            "a fine-tune is a transformer directory, not a single file",
        );
        assert!(
            as_file.contains("transformer DIRECTORY"),
            "expected the directory-shape refusal, got: {as_file}"
        );

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn variant_table_matches_the_published_defaults() {
        // Pinned against the six model cards (epic sc-14034 ground-truth reference). Deliberately
        // asserts the *non-uniform* values — RL 20 vs Base 30 vs Turbo 4 — so a table that
        // collapsed to a single default could not pass.
        let table: Vec<(&str, bool, bool, u32, f32)> = MageVariant::ALL
            .iter()
            .map(|v| {
                (
                    v.id(),
                    v.is_edit(),
                    v.is_distilled(),
                    v.default_steps(),
                    v.default_cfg(),
                )
            })
            .collect();
        assert_eq!(
            table,
            vec![
                ("mage_flow", false, false, 20, 5.0),
                ("mage_flow_base", false, false, 30, 5.0),
                ("mage_flow_turbo", false, true, 4, 1.0),
                ("mage_flow_edit", true, false, 30, 5.0),
                ("mage_flow_edit_base", true, false, 30, 5.0),
                ("mage_flow_edit_turbo", true, true, 4, 1.0),
            ]
        );
    }

    #[test]
    fn registrations_cover_every_variant_in_order() {
        let ids: Vec<&str> = REGISTRATIONS
            .iter()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        assert_eq!(ids, MODEL_IDS.to_vec());
        assert_eq!(
            ids,
            MageVariant::ALL.iter().map(|v| v.id()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn edit_variants_advertise_reference_conditioning_and_gen_variants_do_not() {
        let edit = descriptor_for(MageVariant::Edit);
        assert_eq!(
            edit.capabilities.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference
            ]
        );
        let gen = descriptor_for(MageVariant::Rl);
        assert!(gen.capabilities.conditioning.is_empty());
        // Distillation, not task, drives the CFG surface.
        assert!(gen.capabilities.supports_guidance);
        assert!(
            !descriptor_for(MageVariant::Turbo)
                .capabilities
                .supports_guidance
        );
        assert!(
            descriptor_for(MageVariant::Edit)
                .capabilities
                .supports_guidance
        );
    }

    #[test]
    fn every_edit_variant_enters_the_full_multimodal_loader() {
        let spec = LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()));
        for variant in [
            MageVariant::Edit,
            MageVariant::EditBase,
            MageVariant::EditTurbo,
        ] {
            let err = load(variant, &spec)
                .err()
                .expect("missing edit snapshot must fail");
            assert!(
                !matches!(err, Error::Unsupported(_)),
                "{} must enter the multimodal component loader: {err}",
                variant.id(),
            );
        }
    }

    #[test]
    fn edit_variant_defaults_and_cfg_surfaces_are_exact() {
        let base = descriptor_edit_base();
        assert_eq!(base.id, "mage_flow_edit_base");
        assert_eq!(MageVariant::EditBase.default_steps(), 30);
        assert_eq!(MageVariant::EditBase.default_cfg(), 5.0);
        assert!(base.capabilities.supports_guidance);
        assert!(base.capabilities.supports_negative_prompt);

        let turbo = descriptor_edit_turbo();
        assert_eq!(turbo.id, "mage_flow_edit_turbo");
        assert_eq!(MageVariant::EditTurbo.default_steps(), 4);
        assert_eq!(MageVariant::EditTurbo.default_cfg(), 1.0);
        assert!(!crate::pipeline::uses_cfg(
            MageVariant::EditTurbo.default_cfg()
        ));
        assert!(!turbo.capabilities.supports_guidance);
        assert!(!turbo.capabilities.supports_negative_prompt);
    }

    #[test]
    #[ignore = "needs complete MAGE_EDIT_SNAPSHOT, MAGE_EDIT_BASE_SNAPSHOT, and MAGE_EDIT_TURBO_SNAPSHOT"]
    fn complete_edit_snapshots_are_config_identical_and_checkpoint_distinct() {
        let root = |name: &str| {
            std::path::PathBuf::from(
                std::env::var(name).unwrap_or_else(|_| panic!("set {name} to a complete snapshot")),
            )
        };
        let edit = root("MAGE_EDIT_SNAPSHOT");
        let base = root("MAGE_EDIT_BASE_SNAPSHOT");
        let turbo = root("MAGE_EDIT_TURBO_SNAPSHOT");
        for relative in [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/chat_template.json",
            "transformer/config.json",
            "text_encoder/config.json",
            "text_encoder/generation_config.json",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/preprocessor_config.json",
            "text_encoder/tokenizer.json",
            "text_encoder/tokenizer_config.json",
            "text_encoder/video_preprocessor_config.json",
            "text_encoder/vocab.json",
            "vae/config.json",
        ] {
            let expected = std::fs::read(edit.join(relative)).unwrap();
            assert_eq!(
                std::fs::read(base.join(relative)).unwrap(),
                expected,
                "Edit-Base {relative} must be byte-identical to Edit RL"
            );
            assert_eq!(
                std::fs::read(turbo.join(relative)).unwrap(),
                expected,
                "Edit-Turbo {relative} must be byte-identical to Edit RL"
            );
        }

        let check = |root: &Path, variant, revision, hash| {
            verify_checkpoint_identity(
                root,
                variant,
                revision,
                EDIT_IDENTITY_TENSOR,
                EDIT_IDENTITY_BYTES,
                &[3072],
                hash,
            )
        };
        check(
            &edit,
            MageVariant::Edit,
            EDIT_SNAPSHOT_REVISION,
            EDIT_IDENTITY_SHA256,
        )
        .unwrap();
        check(
            &base,
            MageVariant::EditBase,
            EDIT_BASE_SNAPSHOT_REVISION,
            EDIT_BASE_IDENTITY_SHA256,
        )
        .unwrap();
        check(
            &turbo,
            MageVariant::EditTurbo,
            EDIT_TURBO_SNAPSHOT_REVISION,
            EDIT_TURBO_IDENTITY_SHA256,
        )
        .unwrap();

        for wrong in [&edit, &turbo] {
            assert!(
                check(
                    wrong,
                    MageVariant::EditBase,
                    EDIT_BASE_SNAPSHOT_REVISION,
                    EDIT_BASE_IDENTITY_SHA256,
                )
                .is_err(),
                "Edit-Base must reject RL and Turbo transformer weights"
            );
        }
        for wrong in [&base, &turbo] {
            assert!(
                check(
                    wrong,
                    MageVariant::Edit,
                    EDIT_SNAPSHOT_REVISION,
                    EDIT_IDENTITY_SHA256,
                )
                .is_err(),
                "Edit must reject Base and Turbo transformer weights"
            );
        }
        for wrong in [&edit, &base] {
            assert!(
                check(
                    wrong,
                    MageVariant::EditTurbo,
                    EDIT_TURBO_SNAPSHOT_REVISION,
                    EDIT_TURBO_IDENTITY_SHA256,
                )
                .is_err(),
                "Edit-Turbo must reject RL and Base transformer weights"
            );
        }
    }

    #[test]
    fn edit_reference_shapes_are_required_and_preserve_order() {
        let image = |byte| Image {
            width: 1,
            height: 1,
            pixels: vec![byte, byte, byte],
        };
        let request = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: image(1),
                    strength: None,
                },
                Conditioning::MultiReference {
                    images: vec![image(2), image(3)],
                },
            ],
            ..Default::default()
        };
        let refs = edit_references(&request).unwrap();
        assert_eq!(refs.len(), 3);
        assert_eq!(refs[0].as_raw(), &[1, 1, 1]);
        assert_eq!(refs[1].as_raw(), &[2, 2, 2]);
        assert_eq!(refs[2].as_raw(), &[3, 3, 3]);
        assert!(edit_references(&GenerationRequest::default()).is_err());
        let malformed = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0; 3],
                },
                strength: None,
            }],
            ..Default::default()
        };
        assert!(edit_references(&malformed).is_err());
    }

    #[test]
    fn rl_load_enters_the_real_snapshot_loader() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-mage-flow".into()));
        let err = load(MageVariant::Rl, &spec)
            .err()
            .expect("missing snapshot must fail");
        assert!(
            !matches!(err, Error::Unsupported(_)),
            "RL must not regress to the scaffold refusal: {err}"
        );
    }

    #[test]
    fn base_has_a_distinct_registration_and_enters_the_full_snapshot_loader() {
        assert_eq!(descriptor_base().id, "mage_flow_base");
        assert_eq!(
            MageVariant::Base.upstream_repo(),
            "microsoft/Mage-Flow-Base"
        );
        assert_eq!((REGISTRATION_BASE.descriptor)().id, "mage_flow_base");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-mage-flow-base".into()));
        let err = load_base(&spec)
            .err()
            .expect("missing Base snapshot must fail");
        assert!(
            !matches!(err, Error::Unsupported(_)),
            "Base must enter the same complete component-tree loader: {err}"
        );
    }

    #[test]
    fn base_platform_defaults_are_thirty_steps_with_real_cfg() {
        assert_eq!(MageVariant::Base.default_steps(), 30);
        assert_eq!(MageVariant::Base.default_cfg(), 5.0);
        let descriptor = descriptor_for(MageVariant::Base);
        assert!(descriptor.capabilities.supports_guidance);
        assert!(descriptor.capabilities.supports_negative_prompt);
        let request = GenerationRequest {
            prompt: "test".into(),
            negative_prompt: Some("artifact".into()),
            width: 1024,
            height: 1024,
            guidance: Some(5.0),
            ..Default::default()
        };
        validate_generation_request(&descriptor, &request).unwrap();
    }

    #[test]
    fn turbo_has_a_distinct_registration_and_enters_the_full_snapshot_loader() {
        assert_eq!(descriptor_turbo().id, "mage_flow_turbo");
        assert_eq!(
            MageVariant::Turbo.upstream_repo(),
            "microsoft/Mage-Flow-Turbo"
        );
        assert_eq!((REGISTRATION_TURBO.descriptor)().id, "mage_flow_turbo");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-mage-flow-turbo".into()));
        let err = load_turbo(&spec)
            .err()
            .expect("missing Turbo snapshot must fail");
        assert!(
            !matches!(err, Error::Unsupported(_)),
            "Turbo must enter the same complete component-tree loader: {err}"
        );
    }

    #[test]
    fn turbo_platform_defaults_are_four_steps_with_cfg_and_negative_prompt_off() {
        assert_eq!(MageVariant::Turbo.default_steps(), 4);
        assert_eq!(MageVariant::Turbo.default_cfg(), 1.0);
        let descriptor = descriptor_for(MageVariant::Turbo);
        assert!(!descriptor.capabilities.supports_guidance);
        assert!(!descriptor.capabilities.supports_negative_prompt);
        let plain = GenerationRequest {
            prompt: "test".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        validate_generation_request(&descriptor, &plain).unwrap();
        let mut negative = plain.clone();
        negative.negative_prompt = Some("must not be encoded".into());
        assert!(validate_generation_request(&descriptor, &negative).is_err());
        let mut guided = plain;
        guided.guidance = Some(2.0);
        assert!(validate_generation_request(&descriptor, &guided).is_err());
    }

    #[test]
    fn rl_platform_defaults_and_exact_native_sizes_validate() {
        assert_eq!(MageVariant::Rl.default_steps(), 20);
        assert_eq!(MageVariant::Rl.default_cfg(), 5.0);
        let descriptor = descriptor_for(MageVariant::Rl);
        assert_eq!(descriptor.capabilities.max_count, MAX_COUNT);
        for &(width, height) in &[
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (512, 2048),
            (2048, 512),
            (1232, 688),
        ] {
            let req = GenerationRequest {
                prompt: "test".into(),
                width,
                height,
                ..Default::default()
            };
            validate_generation_request(&descriptor, &req).unwrap();
        }
        for &(width, height) in &[(496, 512), (512, 2064), (513, 512)] {
            let req = GenerationRequest {
                prompt: "test".into(),
                width,
                height,
                ..Default::default()
            };
            assert!(
                validate_generation_request(&descriptor, &req).is_err(),
                "{width}x{height} must be rejected"
            );
        }
        let mut batch = GenerationRequest {
            prompt: "test".into(),
            width: 2048,
            height: 2048,
            count: MAX_COUNT,
            ..Default::default()
        };
        validate_generation_request(&descriptor, &batch).unwrap();
        batch.count += 1;
        assert!(validate_generation_request(&descriptor, &batch).is_err());
    }
}
