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
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer"],
        &["vae"],
    )
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
