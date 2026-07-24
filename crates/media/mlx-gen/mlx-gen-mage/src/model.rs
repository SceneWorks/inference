//! The six Mage-Flow variants, their [`ModelDescriptor`]s, and the explicit registration
//! constants — plus the scaffold's deliberately-failing [`load`].
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
//! Registering each variant under its own id (rather than one id plus a switch) keeps the variant
//! part of the worker's model cache key, matching the `z_image`/`z_image_turbo` and
//! `sensenova_u1_8b`/`_fast` precedent.
//!
//! ## Why this is not in the shipped catalog yet
//!
//! [`load`] cannot produce an image until the four P1 ports land, so [`REGISTRATIONS`] is
//! **published but not wired into [`mlx_gen_catalog::provider_registry`]**. Catalog inclusion is a
//! deliberate edit in this workspace (`CLAUDE.md`: *"a provider crate existing in the repo does not
//! mean it ships"*), and advertising six descriptors whose `load` hard-errors would put six broken
//! entries in front of users. The catalog carries the crate as a compiled member and names it in
//! `mlx_gen_catalog::PENDING_REGISTRATION_CRATES`; sc-14041 flips the registration on once the
//! pipeline renders, and sc-14047 owns the manifest/classifier/schema surface that goes with it.
//!
//! [`mlx_gen_catalog::provider_registry`]: https://docs.rs/mlx-gen-catalog

use mlx_gen::{
    Capabilities, ConditioningKind, Error, Generator, LoadSpec, Modality, ModelDescriptor, Quant,
    Result,
};

use crate::config::{FAMILY, MAX_SIZE, MIN_SIZE, SIZE_MULTIPLE};

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
            // The reference packs a list of prompts into a single forward; 8 matches the platform's
            // other image families. Revisit with the fit gate + memory calibration (sc-14046).
            max_count: 8,
            mac_only: true,
            ..Default::default()
        },
    }
}

/// Construct a Mage-Flow generator from a [`LoadSpec`].
///
/// **Scaffold (sc-14037): this always fails.** Returning a typed
/// [`Error::Unsupported`] is deliberate — a half-built pipeline that
/// renders noise would be worse than a loud, greppable refusal. The body is replaced by sc-14041
/// once the text encoder (sc-14038), VAE (sc-14039), DiT (sc-14040) and watermarked noise
/// (sc-14104) exist.
pub fn load(variant: MageVariant, _spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Err(Error::Unsupported(format!(
        "{}: the mlx-gen-mage provider is a scaffold (sc-14037) — the Qwen3-VL text encoder \
         (sc-14038), Mage-VAE (sc-14039), NR-MMDiT (sc-14040) and Gaussian-Shading noise \
         (sc-14104) are not ported yet, so no {} weights can be loaded",
        variant.id(),
        variant.upstream_repo()
    )))
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
        /// composes. See the module docs for why the shipped MLX catalog does not consume this yet.
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
    fn scaffold_load_fails_typed_and_names_the_blocking_stories() {
        let spec = LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()));
        let err = load(MageVariant::Rl, &spec)
            .err()
            .expect("scaffold must not load");
        assert!(
            matches!(err, Error::Unsupported(_)),
            "must bridge to gen_core::Error::Unsupported, not a generic Msg: {err}"
        );
        let text = err.to_string();
        for needle in ["sc-14037", "sc-14038", "sc-14039", "sc-14040", "sc-14104"] {
            assert!(text.contains(needle), "{needle} missing from: {text}");
        }
    }
}
