//! `mlx-gen-perceptual` — the **one seam** between an MLX trainer and the auxiliary perceptual
//! losses of epic 2123 (E8). The MLX twin of `candle-gen-perceptual`.
//!
//! A trainer never assembles losses itself. It describes its latent family in an
//! [`AuxLossContext`] (its x0 decoder and error label) and calls:
//!
//! - [`perceptual_footprint`] in its memory preflight (E7) — the decoder (only when an enabled loss
//!   decodes pixels) plus every enabled loss, from configs, before anything loads;
//! - [`build_perceptual_path`] before latent caching — `None` when no aux loss is enabled (nothing
//!   loads; the step is the plain diffusion step), else a ready [`PerceptualPath`];
//!
//! then drives the path as the kit documents ([`mlx_gen::train::perceptual`]): references per
//! (item, bucket) cache entry, `AuxAlternation` keyed on the real item index, `plan` / `aux_loss` /
//! `combine_step_loss` in the step.
//!
//! ## Adding a loss (S10 identity/face, S11 body, S12 latent losses …)
//! Append one [`AuxArm`] to [`ARMS`]: its name, whether `cfg` enables it, its [`PerceptualInput`]
//! (pixel losses get the family decoder; latent losses run even on a family with no decoder), its
//! [`AuxModelFootprint`] at a training geometry, and its `build` (load the frozen model from the
//! config's model dir). No trainer loop changes: every trainer that calls this builder picks the new
//! loss up; a trainer declares it by setting the technique flag in its `TrainerDescriptor` (the
//! gen-core floor refuses it elsewhere).
//!
//! **Twin-crate rule.** `mlx-gen-perceptual` and `candle-gen-perceptual` are twins: add an arm (and
//! any new [`DecoderSpec`] variant) to BOTH crates in the same PR. `scripts/check-workspace.py`'s
//! cross-backend comparison reads same-named `pub const` items in the two crates, so it flags an
//! `ARMS` slice whose text differs (route crate-specific calls through a same-named private fn, as
//! `depth_footprint` does); it does not compare enums or functions, so a `DecoderSpec` variant
//! or a builder change must be mirrored by hand.

use std::path::Path;

use mlx_gen::gen_core;
use mlx_gen::gen_core::train::TrainingConfig;
use mlx_gen::train::perceptual::{
    perceptual_footprint_bytes, AuxLoss, AuxModelFootprint, PerceptualInput, PerceptualPath,
    X0Decoder,
};
use mlx_gen::train::tae::{TinyDecoder, TinyDecoderSpec};
use mlx_gen::train::taehv::{TaehvConfig, TaehvDecoder};
use mlx_gen::{Error, Result};

/// A decoder a trainer supplies itself (a video tiny decoder run per frame, or a full-VAE
/// fallback for a family with no tiny decoder). Loaded only when an enabled loss decodes pixels.
pub trait CustomDecoder {
    /// Human name for errors (e.g. `"TAEW2_1"`, `"Mage VAE decoder"`).
    fn name(&self) -> &'static str;
    /// Pre-load memory figures of one differentiable decode to an `h × w` frame.
    fn footprint(&self, h: u32, w: u32) -> AuxModelFootprint;
    /// Load the decoder; `dir` is `TrainingConfig::perceptual_decoder_dir`.
    fn load(&self, dir: Option<&Path>) -> Result<Box<dyn X0Decoder>>;
}

/// The trainer's x0 decoder for its latent family.
pub enum DecoderSpec {
    /// No pixel decoder: pixel losses are refused with a typed error; latent losses still run.
    None,
    /// A TAESD-family tiny decoder loaded from `TrainingConfig::perceptual_decoder_dir`.
    Tiny {
        /// Display name for errors (e.g. `"TAEF1"`).
        name: &'static str,
        /// The decoder structure (`TinyDecoderConfig::taef1().into()`, `TinyDecoderSpec::taef2()`,
        /// …).
        config: TinyDecoderSpec,
    },
    /// A TAEHV tiny video decoder (`taew2_1` / `taew2_2` / `taeltx2_3`) loaded from
    /// `TrainingConfig::perceptual_decoder_dir`, decoding each latent frame as a `T = 1` clip.
    Taehv {
        /// Display name for errors (e.g. `"TAEW2.1"`).
        name: &'static str,
        config: TaehvConfig,
    },
    /// A trainer-built decoder.
    Custom(Box<dyn CustomDecoder>),
}

impl DecoderSpec {
    /// The decoder's display name (`None` for [`DecoderSpec::None`]).
    pub fn name(&self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Tiny { name, .. } | Self::Taehv { name, .. } => Some(name),
            Self::Custom(c) => Some(c.name()),
        }
    }

    fn footprint(&self, h: u32, w: u32) -> Option<AuxModelFootprint> {
        match self {
            Self::None => None,
            Self::Tiny { config, .. } => Some(config.footprint(h, w)),
            Self::Taehv { config, .. } => Some(config.footprint(h, w)),
            Self::Custom(c) => Some(c.footprint(h, w)),
        }
    }
}

/// What the builder needs to know about the calling trainer.
pub struct AuxLossContext<'a> {
    /// Error-message prefix (e.g. `"sdxl trainer"`).
    pub label: &'a str,
    /// The family's x0 decoder.
    pub decoder: DecoderSpec,
    /// The family's E-LatentLPIPS weight set (`None`: no E-LatentLPIPS weights for this latent
    /// space).
    pub latent_lpips: Option<gen_core::train::LatentLpipsFamily>,
}

/// The training geometry a footprint is sized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuxGeometry {
    /// Decoded image height / width (the largest bucket).
    pub height: u32,
    pub width: u32,
    /// Frames decoded per aux step (1 for images; the per-step frame subset for video).
    pub frames: u32,
    /// Cached reference entries (items × buckets).
    pub entries: usize,
}

impl AuxGeometry {
    /// A square single-frame image geometry.
    pub fn image(edge: u32, entries: usize) -> Self {
        Self {
            height: edge,
            width: edge,
            frames: 1,
            entries,
        }
    }
}

/// One auxiliary loss arm of the builder.
pub struct AuxArm {
    /// Short name (matches the loss's `PerceptualLoss::name`).
    pub name: &'static str,
    /// Whether `cfg` enables this loss.
    pub enabled: fn(&TrainingConfig) -> bool,
    /// What the loss consumes.
    pub input: PerceptualInput,
    /// Its pre-load memory figures for one `h × w` frame.
    pub footprint: fn(&TrainingConfig, u32, u32) -> AuxModelFootprint,
    /// Load the frozen model and wrap it as a scheduled [`AuxLoss`].
    pub build: fn(&TrainingConfig, &AuxLossContext<'_>) -> Result<AuxLoss>,
}

/// Depth anchoring (sc-2125 / sc-24830): Depth-Anything-V2 on the decoded x0.
fn build_depth(cfg: &TrainingConfig, ctx: &AuxLossContext<'_>) -> Result<AuxLoss> {
    let depth = &cfg.depth_anchoring;
    let dir = depth.model_dir.as_ref().ok_or_else(|| {
        Error::Msg(format!(
            "{}: depth anchoring needs the Depth-Anything-V2 {} checkpoint \
             (depth_anchoring.model_dir)",
            ctx.label,
            depth.model_size.as_str()
        ))
    })?;
    let loss =
        mlx_gen_depth::anchor::DepthAnchorLoss::from_dir(dir, depth.model_size).map_err(|e| {
            Error::Msg(format!(
                "{}: could not load Depth-Anything-V2 {} from {}: {e}",
                ctx.label,
                depth.model_size.as_str(),
                dir.display()
            ))
        })?;
    Ok(AuxLoss {
        schedule: depth.schedule,
        loss: Box::new(loss),
    })
}

/// Depth anchoring's pre-load footprint at one `h × w` frame.
fn depth_footprint(cfg: &TrainingConfig, h: u32, w: u32) -> AuxModelFootprint {
    mlx_gen_depth::anchor::depth_anchor_footprint(cfg.depth_anchoring.model_size, h, w)
}

/// The VAE perceptual anchor (sc-24833): a frozen FLUX.2 VAE encoder's multi-scale features on the
/// decoded x0.
fn build_vae_anchor(cfg: &TrainingConfig, ctx: &AuxLossContext<'_>) -> Result<AuxLoss> {
    let va = &cfg.vae_anchor;
    let dir = va.model_dir.as_ref().ok_or_else(|| {
        Error::Msg(format!(
            "{}: the VAE anchor loss needs the FLUX.2 VAE (vae_anchor.model_dir)",
            ctx.label
        ))
    })?;
    let loss = mlx_gen::train::vae_anchor::VaeAnchorLoss::from_dir(dir).map_err(|e| {
        Error::Msg(format!(
            "{}: could not load the FLUX.2 VAE encoder from {}: {e}",
            ctx.label,
            dir.display()
        ))
    })?;
    Ok(AuxLoss {
        schedule: va.schedule,
        loss: Box::new(loss),
    })
}

/// E-LatentLPIPS (sc-24833) on the x0 latent, with the weights of the trainer's latent family
/// (`ctx.latent_lpips`; `None` ⇒ no published weights match ⇒ a named error — the descriptor flag
/// is false there too, so the floor refuses first).
fn build_latent_lpips(cfg: &TrainingConfig, ctx: &AuxLossContext<'_>) -> Result<AuxLoss> {
    let lp = &cfg.latent_lpips;
    let family = ctx.latent_lpips.ok_or_else(|| {
        Error::Msg(format!(
            "{}: no E-LatentLPIPS weights match this trainer's latent family",
            ctx.label
        ))
    })?;
    let dir = lp.model_dir.as_ref().ok_or_else(|| {
        Error::Msg(format!(
            "{}: the E-LatentLPIPS loss needs its weights (latent_lpips.model_dir)",
            ctx.label
        ))
    })?;
    let loss =
        mlx_gen::train::latent_lpips::LatentLpipsLoss::from_dir(dir, family).map_err(|e| {
            Error::Msg(format!(
                "{}: could not load E-LatentLPIPS ({}) from {}: {e}",
                ctx.label,
                family.as_str(),
                dir.display()
            ))
        })?;
    Ok(AuxLoss {
        schedule: lp.schedule,
        loss: Box::new(loss),
    })
}

/// The VAE anchor's pre-load footprint at one `h × w` frame (the FLUX.2 encoder; the decoder that
/// feeds it is counted by the builder).
fn vae_anchor_footprint(_cfg: &TrainingConfig, h: u32, w: u32) -> AuxModelFootprint {
    mlx_gen::train::vae_anchor::vae_anchor_footprint(h, w)
}

/// E-LatentLPIPS's pre-load footprint at one `h × w` frame: every published family has an 8×
/// VAE, so the latent is `h/8 × w/8`; the footprint fn has no trainer context, so it is sized for
/// the 16-channel families (an upper bound for the 4-channel ones — the trunk is identical past the
/// first conv).
fn latent_lpips_footprint(_cfg: &TrainingConfig, h: u32, w: u32) -> AuxModelFootprint {
    mlx_gen::train::latent_lpips::latent_lpips_footprint(
        gen_core::train::LatentLpipsFamily::Flux,
        h.div_ceil(8),
        w.div_ceil(8),
    )
}

/// Every auxiliary loss, in loss-index order. **Extension point**: later stories append an arm.
pub const ARMS: &[AuxArm] = &[
    AuxArm {
        name: "depth",
        enabled: |cfg| cfg.depth_anchoring.schedule.is_enabled(),
        input: PerceptualInput::DecodedPixels,
        footprint: depth_footprint,
        build: build_depth,
    },
    AuxArm {
        name: "vae_anchor",
        enabled: |cfg| cfg.vae_anchor.schedule.is_enabled(),
        input: PerceptualInput::DecodedPixels,
        footprint: vae_anchor_footprint,
        build: build_vae_anchor,
    },
    AuxArm {
        name: "latent_lpips",
        enabled: |cfg| cfg.latent_lpips.schedule.is_enabled(),
        input: PerceptualInput::Latents,
        footprint: latent_lpips_footprint,
        build: build_latent_lpips,
    },
];

/// Whether any auxiliary loss is enabled in `cfg`.
pub fn any_aux_loss(cfg: &TrainingConfig) -> bool {
    ARMS.iter().any(|a| (a.enabled)(cfg))
}

fn enabled_arms<'a>(arms: &'a [AuxArm], cfg: &TrainingConfig) -> Vec<&'a AuxArm> {
    arms.iter().filter(|a| (a.enabled)(cfg)).collect()
}

/// Every enabled loss of `cfg`, loaded, in arm order. Empty when none is enabled.
pub fn build_aux_losses(cfg: &TrainingConfig, ctx: &AuxLossContext<'_>) -> Result<Vec<AuxLoss>> {
    build_aux_losses_with(ARMS, cfg, ctx)
}

fn build_aux_losses_with(
    arms: &[AuxArm],
    cfg: &TrainingConfig,
    ctx: &AuxLossContext<'_>,
) -> Result<Vec<AuxLoss>> {
    enabled_arms(arms, cfg)
        .into_iter()
        .map(|a| (a.build)(cfg, ctx))
        .collect()
}

/// The trainer's perceptual path for `cfg`: `None` when no aux loss is enabled (nothing loads);
/// else the family decoder (loaded only when an enabled loss decodes pixels — a family with
/// [`DecoderSpec::None`] gets a typed error for a pixel loss) plus every enabled loss. Call before
/// latent caching so a missing checkpoint fails fast.
pub fn build_perceptual_path(
    cfg: &TrainingConfig,
    ctx: &AuxLossContext<'_>,
) -> Result<Option<PerceptualPath>> {
    build_perceptual_path_with(ARMS, cfg, ctx)
}

fn build_perceptual_path_with(
    arms: &[AuxArm],
    cfg: &TrainingConfig,
    ctx: &AuxLossContext<'_>,
) -> Result<Option<PerceptualPath>> {
    let enabled = enabled_arms(arms, cfg);
    if enabled.is_empty() {
        return Ok(None);
    }
    let pixel_arm = enabled
        .iter()
        .find(|a| a.input == PerceptualInput::DecodedPixels);
    let decoder: Option<Box<dyn X0Decoder>> = match (pixel_arm, &ctx.decoder) {
        (None, _) => None,
        (Some(a), DecoderSpec::None) => {
            return Err(Error::Msg(format!(
                "{}: the '{}' loss decodes x0 to pixels, but this trainer's latent family has no \
                 x0 decoder",
                ctx.label, a.name
            )))
        }
        (Some(_), DecoderSpec::Tiny { name, config }) => {
            let dir = cfg.perceptual_decoder_dir.as_ref().ok_or_else(|| {
                Error::Msg(format!(
                    "{}: the perceptual losses need the {name} decoder (perceptual_decoder_dir)",
                    ctx.label
                ))
            })?;
            let dec = TinyDecoder::from_dir(dir, config.clone()).map_err(|e| {
                Error::Msg(format!(
                    "{}: could not load the {name} decoder from {}: {e}",
                    ctx.label,
                    dir.display()
                ))
            })?;
            Some(Box::new(dec))
        }
        (Some(_), DecoderSpec::Taehv { name, config }) => {
            let dir = cfg.perceptual_decoder_dir.as_ref().ok_or_else(|| {
                Error::Msg(format!(
                    "{}: the perceptual losses need the {name} decoder (perceptual_decoder_dir)",
                    ctx.label
                ))
            })?;
            let dec = TaehvDecoder::from_path(dir, config.clone()).map_err(|e| {
                Error::Msg(format!(
                    "{}: could not load the {name} decoder from {}: {e}",
                    ctx.label,
                    dir.display()
                ))
            })?;
            Some(Box::new(dec))
        }
        (Some(_), DecoderSpec::Custom(c)) => {
            Some(c.load(cfg.perceptual_decoder_dir.as_deref()).map_err(|e| {
                Error::Msg(format!(
                    "{}: could not load the {} decoder: {e}",
                    ctx.label,
                    c.name()
                ))
            })?)
        }
    };
    let losses = build_aux_losses_with(arms, cfg, ctx)?;
    Ok(Some(PerceptualPath::new(decoder, losses)?))
}

/// The extra training memory (bytes) the enabled aux losses add at `geom` (epic 2123 E7): the
/// decoder when any enabled loss decodes pixels, plus every enabled loss, with per-frame working
/// sets and references scaled by `geom.frames`. `0` when nothing is enabled.
pub fn perceptual_footprint(cfg: &TrainingConfig, decoder: &DecoderSpec, geom: AuxGeometry) -> u64 {
    perceptual_footprint_with(ARMS, cfg, decoder, geom)
}

fn perceptual_footprint_with(
    arms: &[AuxArm],
    cfg: &TrainingConfig,
    decoder: &DecoderSpec,
    geom: AuxGeometry,
) -> u64 {
    let enabled = enabled_arms(arms, cfg);
    if enabled.is_empty() {
        return 0;
    }
    let frames = geom.frames.max(1) as u64;
    let scale = |f: AuxModelFootprint| AuxModelFootprint {
        param_bytes: f.param_bytes,
        working_set_bytes: f.working_set_bytes * frames,
        reference_bytes_per_image: f.reference_bytes_per_image * frames,
    };
    let dec = if enabled
        .iter()
        .any(|a| a.input == PerceptualInput::DecodedPixels)
    {
        decoder.footprint(geom.height, geom.width).map(scale)
    } else {
        None
    };
    let losses: Vec<AuxModelFootprint> = enabled
        .iter()
        .map(|a| scale((a.footprint)(cfg, geom.height, geom.width)))
        .collect();
    perceptual_footprint_bytes(dec, &losses, geom.entries)
}

/// [`perceptual_footprint`] in GiB (the unit most trainer preflights use).
pub fn perceptual_footprint_gb(
    cfg: &TrainingConfig,
    decoder: &DecoderSpec,
    geom: AuxGeometry,
) -> f64 {
    perceptual_footprint(cfg, decoder, geom) as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Test fixtures for trainers on this seam.
pub mod testing {
    use std::path::Path;

    use mlx_gen::gen_core::train::AuxLossSchedule;
    use mlx_gen::train::perceptual::{AuxLoss, PerceptualPath};
    use mlx_gen::train::tae::{synthetic_tiny_decoder_weights, TinyDecoder, TinyDecoderConfig};
    use mlx_gen::Result;
    use mlx_gen_depth::anchor::{synthetic_weights, tiny_config, DepthAnchorLoss};
    use mlx_gen_depth::DepthAnythingV2;

    /// The tiny TAESD-shaped decoder config tests use (`latent_channels` channels, width 8).
    pub fn tiny_decoder_config(latent_channels: i32) -> TinyDecoderConfig {
        TinyDecoderConfig {
            latent_channels,
            channels: 8,
            blocks: [3, 3, 3, 1],
        }
    }

    /// A ready depth-anchoring [`PerceptualPath`] with a random-init tiny decoder for
    /// `latent_channels` + a random-init tiny DA2, scheduled by `schedule` — for trainer step tests
    /// (what [`super::build_perceptual_path`] returns for a depth job, minus the checkpoint reads).
    pub fn tiny_depth_path(
        latent_channels: i32,
        schedule: AuxLossSchedule,
    ) -> Result<PerceptualPath> {
        let cfg = tiny_decoder_config(latent_channels);
        let dec = TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&cfg, 11)?, cfg)?;
        let da2 = tiny_config();
        let depth = DepthAnchorLoss::new(DepthAnythingV2::from_weights(
            &synthetic_weights(&da2, 12)?,
            da2,
        )?);
        PerceptualPath::new(
            Some(Box::new(dec)),
            vec![AuxLoss {
                schedule,
                loss: Box::new(depth),
            }],
        )
    }

    /// Write a random-init tiny decoder checkpoint (`diffusion_pytorch_model.safetensors`) to `dir`.
    pub fn write_tiny_decoder(dir: &Path, cfg: &TinyDecoderConfig, seed: u64) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| mlx_gen::Error::Msg(e.to_string()))?;
        let w = synthetic_tiny_decoder_weights(cfg, seed)?;
        let pairs: Vec<(String, mlx_rs::Array)> = w
            .keys()
            .map(|k| Ok((k.to_string(), w.require(k)?.clone())))
            .collect::<Result<_>>()?;
        let refs: Vec<(&str, &mlx_rs::Array)> =
            pairs.iter().map(|(k, a)| (k.as_str(), a)).collect();
        mlx_rs::Array::save_safetensors(
            refs,
            None,
            dir.join("diffusion_pytorch_model.safetensors"),
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::train::{AuxLossSchedule, DepthModelSize};
    use mlx_gen::train::perceptual::{reference_as, LossReference, PerceptualLoss};
    use mlx_gen::train::tae::TinyDecoderConfig;
    use mlx_rs::Array;
    use std::any::Any;

    fn on() -> TrainingConfig {
        let mut cfg = TrainingConfig::default();
        cfg.depth_anchoring.schedule = AuxLossSchedule {
            weight: 0.1,
            t_min: 0.0,
            t_max: 1.0,
            every_n: 2,
        };
        cfg
    }

    fn taef1() -> DecoderSpec {
        DecoderSpec::Tiny {
            name: "TAEF1",
            config: TinyDecoderConfig::taef1().into(),
        }
    }

    /// E1: nothing enabled ⇒ no path, nothing loaded, zero footprint. Mutation: return
    /// `Some(empty path)` ⇒ red.
    #[test]
    fn everything_off_builds_nothing() {
        let ctx = AuxLossContext {
            label: "t",
            decoder: taef1(),
            latent_lpips: None,
        };
        let cfg = TrainingConfig::default();
        assert!(!any_aux_loss(&cfg));
        assert!(build_perceptual_path(&cfg, &ctx).unwrap().is_none());
        assert!(build_aux_losses(&cfg, &ctx).unwrap().is_empty());
        assert_eq!(
            perceptual_footprint(&cfg, &taef1(), AuxGeometry::image(1024, 4)),
            0
        );
    }

    /// E7: depth on adds the decoder + DA2 (larger for Large); frames scale the per-frame terms.
    /// Mutations: drop the decoder term ⇒ red; drop the `frames` scaling ⇒ red.
    #[test]
    fn footprint_counts_decoder_losses_and_frames() {
        let mut cfg = on();
        let g = AuxGeometry::image(1024, 4);
        let small = perceptual_footprint(&cfg, &taef1(), g);
        let dec = TinyDecoderConfig::taef1().footprint(1024, 1024);
        let da2 = mlx_gen_depth::anchor::depth_anchor_footprint(DepthModelSize::Small, 1024, 1024);
        assert_eq!(small, perceptual_footprint_bytes(Some(dec), &[da2], 4));
        cfg.depth_anchoring.model_size = DepthModelSize::Large;
        assert!(perceptual_footprint(&cfg, &taef1(), g) > small + 1_000_000_000);
        cfg.depth_anchoring.model_size = DepthModelSize::Small;
        let video = perceptual_footprint(&cfg, &taef1(), AuxGeometry { frames: 3, ..g });
        let ws = dec.working_set_bytes + da2.working_set_bytes;
        assert_eq!(
            video - small,
            2 * ws + 2 * 4 * da2.reference_bytes_per_image
        );
    }

    /// A pixel loss on a family with no decoder is a typed error naming the loss; a missing
    /// decoder dir names the decoder. Mutation: skip the `DecoderSpec::None` arm ⇒ red.
    #[test]
    fn missing_decoders_are_named() {
        let none = AuxLossContext {
            label: "fam trainer",
            decoder: DecoderSpec::None,
            latent_lpips: None,
        };
        let e = build_perceptual_path(&on(), &none)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("depth") && e.contains("no x0 decoder"), "{e}");
        let tiny = AuxLossContext {
            label: "fam trainer",
            decoder: taef1(),
            latent_lpips: None,
        };
        let e = build_perceptual_path(&on(), &tiny)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("TAEF1"), "{e}");
        let tmp = tempfile::tempdir().unwrap();
        let mut c = on();
        c.perceptual_decoder_dir = Some(tmp.path().join("nope"));
        let e = build_perceptual_path(&c, &tiny).err().unwrap().to_string();
        assert!(e.contains("TAEF1"), "{e}");
    }

    /// The real load path: a tiny decoder checkpoint on disk loads, then a missing DA2 checkpoint
    /// is a named error (the decoder loaded first).
    #[test]
    fn decoder_loads_from_disk_before_the_losses() {
        let tmp = tempfile::tempdir().unwrap();
        let dec_dir = tmp.path().join("tae");
        let cfg4 = testing::tiny_decoder_config(4);
        testing::write_tiny_decoder(&dec_dir, &cfg4, 1).unwrap();
        let mut c = on();
        c.perceptual_decoder_dir = Some(dec_dir);
        c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
        let ctx = AuxLossContext {
            label: "t",
            decoder: DecoderSpec::Tiny {
                name: "TINY",
                config: cfg4.into(),
            },
            latent_lpips: None,
        };
        let e = build_perceptual_path(&c, &ctx).err().unwrap().to_string();
        assert!(e.contains("Depth-Anything-V2"), "{e}");
    }

    /// A latent-input arm (the shape of S12's E-LatentLPIPS) runs on a family with no decoder.
    /// Mutation: require a decoder for any enabled arm ⇒ red.
    #[test]
    fn a_latent_arm_runs_without_a_decoder() {
        struct L;
        impl PerceptualLoss for L {
            fn name(&self) -> &'static str {
                "latent"
            }
            fn input(&self) -> PerceptualInput {
                PerceptualInput::Latents
            }
            fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
                let m = clean.mean(None)?;
                m.eval()?;
                Ok(Some(Box::new(m)))
            }
            fn loss(&self, live: &Array, r: &dyn Any) -> Result<Array> {
                let r = reference_as::<Array>("latent", r)?;
                Ok(live.mean(None)?.subtract(r)?.square()?)
            }
        }
        let arms = [AuxArm {
            name: "latent",
            enabled: |_| true,
            input: PerceptualInput::Latents,
            footprint: |_, _, _| AuxModelFootprint {
                param_bytes: 5,
                working_set_bytes: 0,
                reference_bytes_per_image: 0,
            },
            build: |_, _| {
                Ok(AuxLoss {
                    schedule: AuxLossSchedule {
                        weight: 1.0,
                        t_min: 0.0,
                        t_max: 1.0,
                        every_n: 1,
                    },
                    loss: Box::new(L),
                })
            },
        }];
        let ctx = AuxLossContext {
            label: "t",
            decoder: DecoderSpec::None,
            latent_lpips: None,
        };
        let cfg = TrainingConfig::default();
        let mut path = build_perceptual_path_with(&arms, &cfg, &ctx)
            .unwrap()
            .unwrap();
        let z = Array::ones::<f32>(&[1, 4, 2, 2]).unwrap();
        path.ensure_reference(0, &z).unwrap();
        let plan = path.plan(1, 0, 0.5).unwrap();
        assert!(path.aux_loss(&plan, 0, &z).unwrap().is_some());
        assert_eq!(
            perceptual_footprint_with(&arms, &cfg, &taef1(), AuxGeometry::image(64, 1)),
            5
        );
    }

    fn latent_on(model_dir: Option<std::path::PathBuf>) -> TrainingConfig {
        let mut cfg = TrainingConfig::default();
        cfg.latent_lpips.schedule = AuxLossSchedule {
            weight: 0.5,
            ..gen_core::train::LATENT_PERCEPTUAL_SCHEDULE
        };
        cfg.latent_lpips.model_dir = model_dir;
        cfg
    }

    fn vae_anchor_on() -> TrainingConfig {
        let mut cfg = TrainingConfig::default();
        cfg.vae_anchor.schedule = AuxLossSchedule {
            weight: 0.5,
            ..gen_core::train::LATENT_PERCEPTUAL_SCHEDULE
        };
        cfg
    }

    /// sc-24833: the E-LatentLPIPS arm builds on a family with NO x0 decoder (it is a latent
    /// loss), loads the family's checkpoint from `latent_lpips.model_dir`, and trains: zero at the
    /// clean latent, positive off it. A context naming no family is a named error. Mutations: give
    /// the arm `PerceptualInput::DecodedPixels` ⇒ the `DecoderSpec::None` build errors ⇒ red; ignore
    /// `ctx.latent_lpips` ⇒ the no-family build succeeds ⇒ red.
    #[test]
    fn the_latent_lpips_arm_runs_without_a_decoder_on_its_family_weights() {
        let tmp = tempfile::tempdir().unwrap();
        write_lpips_checkpoint(&tmp.path().join("sdxl_latest_vgg16_tuned.safetensors"));
        let cfg = latent_on(Some(tmp.path().to_path_buf()));
        assert!(any_aux_loss(&cfg));
        let ctx = AuxLossContext {
            label: "t",
            decoder: DecoderSpec::None,
            latent_lpips: Some(gen_core::train::LatentLpipsFamily::Sdxl),
        };
        let mut path = build_perceptual_path(&cfg, &ctx).unwrap().unwrap();
        assert_eq!(path.losses()[0].loss.name(), "latent_lpips");
        let clean = mlx_rs::random::normal::<f32>(
            &[1, 4, 8, 8],
            None,
            None,
            Some(&mlx_rs::random::key(2).unwrap()),
        )
        .unwrap();
        path.ensure_reference(0, &clean).unwrap();
        let plan = path.plan(1, 0, 0.25).unwrap();
        assert!(plan.diffusion && plan.aux == vec![0], "{plan:?}");
        assert_eq!(
            mlx_scalar(path.aux_loss(&plan, 0, &clean).unwrap().unwrap().weighted),
            0.0
        );
        let off = mlx_rs::ops::add(&clean, Array::from_f32(0.2)).unwrap();
        assert!(mlx_scalar(path.aux_loss(&plan, 0, &off).unwrap().unwrap().weighted) > 0.0);
        let none = AuxLossContext {
            label: "t",
            decoder: DecoderSpec::None,
            latent_lpips: None,
        };
        let e = build_perceptual_path(&cfg, &none)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("no E-LatentLPIPS weights match"), "{e}");
        let e = build_perceptual_path(&latent_on(None), &ctx)
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("latent_lpips.model_dir"), "{e}");
    }

    /// sc-24833: the VAE anchor is a decoded-x0 loss — a family with no decoder gets a typed error
    /// naming it, and a missing FLUX.2 VAE dir is named (after the decoder loads). E7: enabling it
    /// adds the decoder + the FLUX.2 encoder footprint; E-LatentLPIPS adds its own and no decoder.
    /// Mutations: drop the arm's `footprint` term ⇒ red; mark it `Latents` ⇒ the no-decoder build
    /// succeeds ⇒ red.
    #[test]
    fn the_vae_anchor_arm_is_a_pixel_loss_with_its_own_footprint() {
        let none = AuxLossContext {
            label: "t",
            decoder: DecoderSpec::None,
            latent_lpips: None,
        };
        let e = build_perceptual_path(&vae_anchor_on(), &none)
            .err()
            .unwrap()
            .to_string();
        assert!(
            e.contains("vae_anchor") && e.contains("no x0 decoder"),
            "{e}"
        );
        let tmp = tempfile::tempdir().unwrap();
        let dec_dir = tmp.path().join("tae");
        let cfg4 = testing::tiny_decoder_config(4);
        testing::write_tiny_decoder(&dec_dir, &cfg4, 1).unwrap();
        let mut c = vae_anchor_on();
        c.perceptual_decoder_dir = Some(dec_dir);
        let tiny = AuxLossContext {
            label: "t",
            decoder: DecoderSpec::Tiny {
                name: "TINY",
                config: cfg4.into(),
            },
            latent_lpips: None,
        };
        let e = build_perceptual_path(&c, &tiny).err().unwrap().to_string();
        assert!(e.contains("FLUX.2 VAE"), "{e}");
        c.vae_anchor.model_dir = Some(tmp.path().join("no-vae"));
        let e = build_perceptual_path(&c, &tiny).err().unwrap().to_string();
        assert!(e.contains("FLUX.2 VAE encoder"), "{e}");

        let g = AuxGeometry::image(512, 3);
        let dec = TinyDecoderConfig::taef1().footprint(512, 512);
        let va = mlx_gen::train::vae_anchor::vae_anchor_footprint(512, 512);
        assert_eq!(
            perceptual_footprint(&vae_anchor_on(), &taef1(), g),
            perceptual_footprint_bytes(Some(dec), &[va], 3)
        );
        // sc-24833 review: the VAE-anchor references spill to disk, so the footprint does not
        // scale with the cached-entry count (a 50-image × 3-bucket job admits like a 1-entry one).
        assert_eq!(
            perceptual_footprint(&vae_anchor_on(), &taef1(), AuxGeometry::image(512, 150)),
            perceptual_footprint(&vae_anchor_on(), &taef1(), g)
        );
        let lp = mlx_gen::train::latent_lpips::latent_lpips_footprint(
            gen_core::train::LatentLpipsFamily::Flux,
            64,
            64,
        );
        assert_eq!(
            perceptual_footprint(&latent_on(None), &taef1(), g),
            perceptual_footprint_bytes(None, &[lp], 3)
        );
    }

    fn mlx_scalar(a: Array) -> f32 {
        a.eval().unwrap();
        a.item::<f32>()
    }

    fn write_lpips_checkpoint(path: &std::path::Path) {
        let w = mlx_gen::train::latent_lpips::formula_weights(4).into_tensors();
        let mut named: Vec<(&String, &Array)> = w.iter().collect();
        named.sort_unstable_by_key(|(k, _)| *k);
        Array::save_safetensors(
            named.into_iter().map(|(k, v)| (k.as_str(), v)),
            None::<&std::collections::HashMap<String, String>>,
            path,
        )
        .unwrap();
    }
}
