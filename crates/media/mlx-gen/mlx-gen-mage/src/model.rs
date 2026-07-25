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
//! Each variant has its own future id (rather than one id plus a switch), keeping the variant part
//! of the worker's model cache key. sc-14041 registers only the completed RL id.
//!
//! The RL id is composed into the shipped platform catalog. The other descriptors remain available
//! for their owning stories, but are not registered until their variant-specific paths are complete.
//!
//! [`mlx_gen_catalog::provider_registry`]: https://docs.rs/mlx-gen-catalog

use mlx_gen::{
    Capabilities, ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant, Result, WeightsSource,
};

use crate::config::{FAMILY, MAX_SIZE, MIN_SIZE, SIZE_MULTIPLE};
use crate::{resolve_gs_key, MageFlowPipeline};

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

/// Build a variant's weights-free descriptor.
///
/// Capability fields that later stories own are left at their conservative `Default` (`false` /
/// empty) rather than pre-announced: quant tiers are sc-14046, LoRA/LoKr routing is sc-14057, and
/// the curated sampler/scheduler menus are sc-14041's once the flow-match loop exists. A
/// descriptor is a promise to the worker, so the scaffold promises only what it can point at in
/// the published configs.
pub fn descriptor_for(variant: MageVariant) -> ModelDescriptor {
    ModelDescriptor {
        // The diffusers component tree (`transformer/`, `vae/`, `text_encoder/`, `scheduler/`)
        // arrives as one `WeightsSource::Dir`; no extra caller-provisioned components.
        required_components: &[],
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
            supported_quants: &[] as &[Quant],
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            // The walking skeleton exposes one output per request. Packed multi-image generation
            // remains disabled until its memory behavior is calibrated in sc-14046.
            max_count: 1,
            mac_only: true,
            ..Default::default()
        },
    }
}

/// Construct a Mage-Flow generator from a [`LoadSpec`].
///
pub fn load(variant: MageVariant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if variant != MageVariant::Rl {
        return Err(Error::Unsupported(format!(
            "{}: only the Mage-Flow RL checkpoint is enabled in sc-14041",
            variant.id()
        )));
    }
    if spec.precision != Precision::Bf16 || spec.quantize.is_some() || !spec.adapters.is_empty() {
        return Err(Error::Unsupported(
            "mage_flow: sc-14041 supports the dense bf16 RL checkpoint without adapters".into(),
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
    Ok(Box::new(MageFlow {
        descriptor: descriptor_for(variant),
        pipeline: MageFlowPipeline::load(root)?,
    }))
}

pub struct MageFlow {
    descriptor: ModelDescriptor,
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
        if req.cancel.is_cancelled() {
            return Err(mlx_gen::gen_core::Error::Canceled);
        }
        let steps = req.steps.unwrap_or(MageVariant::Rl.default_steps());
        let cfg = req.guidance.unwrap_or(MageVariant::Rl.default_cfg());
        let seed = req.seed.unwrap_or(0) as i64;
        let key = resolve_gs_key(None)?;
        let pixels = self
            .pipeline
            .generate_trace(
                &req.prompt,
                req.negative_prompt.as_deref().unwrap_or(" "),
                req.height,
                req.width,
                steps as usize,
                cfg,
                seed,
                &key,
                false,
                on_progress,
            )?
            .image_u8;
        mlx_rs::transforms::eval([&pixels]).map_err(Error::from)?;
        let bytes = pixels
            .try_as_slice::<u8>()
            .map_err(|e| Error::Msg(format!("mage_flow: RGB8 output is not host-readable: {e}")))?
            .to_vec();
        Ok(GenerationOutput::Images(vec![Image {
            width: req.width,
            height: req.height,
            pixels: bytes,
        }]))
    }
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
    if req.count != 1 {
        return Err(mlx_gen::gen_core::Error::Msg(
            "mage_flow sc-14041 generates exactly one image".into(),
        ));
    }
    Ok(())
}

/// Every side of a Mage-Flow request must be a multiple of this (the VAE's 16× downsample;
/// `patch_size == 1` adds no further stride). Re-exported at the model layer because SceneWorks
/// pins each advertised resolution bucket to an engine stride constant.
pub const REQUIRED_SIZE_MULTIPLE: u32 = SIZE_MULTIPLE;

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
                pub const $registration = $descriptor_fn => $load_fn
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
    fn non_rl_variants_remain_explicitly_disabled() {
        let spec = LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()));
        let err = load(MageVariant::Base, &spec)
            .err()
            .expect("non-RL variant must remain disabled");
        assert!(matches!(err, Error::Unsupported(_)));
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
    fn rl_platform_defaults_and_size_buckets_validate() {
        assert_eq!(MageVariant::Rl.default_steps(), 20);
        assert_eq!(MageVariant::Rl.default_cfg(), 5.0);
        let descriptor = descriptor_for(MageVariant::Rl);
        for &(width, height) in &[(512, 512), (1024, 1024), (2048, 2048), (512, 2048)] {
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
    }
}
