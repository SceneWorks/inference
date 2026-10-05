//! `candle-gen-perceptual` — the **one seam** between a Candle trainer and the auxiliary perceptual
//! losses of epic 2123 (E8). The Candle twin of `mlx-gen-perceptual`.
//!
//! A trainer never assembles losses itself. It describes its latent family in an
//! [`AuxLossContext`] (its x0 decoder, device, error label) and calls:
//!
//! - [`perceptual_footprint`] in its memory preflight (E7) — the decoder (only when an enabled loss
//!   decodes pixels) plus every enabled loss, from configs, before anything loads;
//! - [`build_perceptual_path`] before latent caching — `None` when no aux loss is enabled (nothing
//!   loads; the step is the plain diffusion step), else a ready
//!   [`PerceptualPath`] holding the family decoder and every enabled loss;
//!
//! then drives the path as the kit documents ([`candle_gen::train::perceptual`]): references per
//! (item, bucket) cache entry, [`AuxAlternation`](candle_gen::train::perceptual::AuxAlternation)
//! keyed on the real item index, `plan` / `aux_loss` / `combine_step_loss` in the step.
//!
//! ## Adding a loss (S10 identity/face, S11 body, S12 latent losses …)
//! Append one [`AuxArm`] to [`ARMS`]: its name, whether `cfg` enables it, its
//! [`PerceptualInput`] (pixel losses get the family decoder; latent losses run even on a family
//! with no decoder), its [`AuxModelFootprint`] at a training geometry, and its `build` (load the
//! frozen model from the config's model dir onto `ctx.device`). No trainer loop changes: every
//! trainer that calls this builder picks the new loss up; a trainer declares it by setting the
//! technique flag in its `TrainerDescriptor` (the gen-core floor refuses it elsewhere).

use std::path::Path;

use candle_gen::candle_core::Device;
use candle_gen::gen_core;
use candle_gen::gen_core::train::TrainingConfig;
use candle_gen::train::perceptual::{
    perceptual_footprint_bytes, AuxLoss, AuxModelFootprint, PerceptualInput, PerceptualPath,
    X0Decoder,
};
use candle_gen::train::tae::{TinyDecoder, TinyDecoderConfig};
use candle_gen::{CandleError, Result};

/// A decoder a trainer supplies itself (a video tiny decoder run per frame, or a full-VAE
/// fallback for a family with no tiny decoder). Loaded only when an enabled loss decodes pixels.
pub trait CustomDecoder: Send + Sync {
    /// Human name for errors (e.g. `"TAEW2_1"`, `"Mage VAE decoder"`).
    fn name(&self) -> &'static str;
    /// Pre-load memory figures of one differentiable decode to an `h × w` frame.
    fn footprint(&self, h: u32, w: u32) -> AuxModelFootprint;
    /// Load the decoder; `dir` is `TrainingConfig::perceptual_decoder_dir`.
    fn load(&self, dir: Option<&Path>, device: &Device) -> Result<Box<dyn X0Decoder>>;
}

/// The trainer's x0 decoder for its latent family.
pub enum DecoderSpec {
    /// No pixel decoder: pixel losses are refused with a typed error; latent losses still run.
    None,
    /// A TAESD-family tiny decoder loaded from `TrainingConfig::perceptual_decoder_dir`.
    Tiny {
        /// Display name for errors (e.g. `"TAEF1"`).
        name: &'static str,
        config: TinyDecoderConfig,
    },
    /// A trainer-built decoder.
    Custom(Box<dyn CustomDecoder>),
}

impl DecoderSpec {
    /// The decoder's display name (`None` for [`DecoderSpec::None`]).
    pub fn name(&self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Tiny { name, .. } => Some(name),
            Self::Custom(c) => Some(c.name()),
        }
    }

    fn footprint(&self, h: u32, w: u32) -> Option<AuxModelFootprint> {
        match self {
            Self::None => None,
            Self::Tiny { config, .. } => Some(config.footprint(h, w)),
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
    /// Device the frozen models load onto.
    pub device: &'a Device,
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
        CandleError::Msg(format!(
            "{}: depth anchoring needs the Depth-Anything-V2 {} checkpoint \
             (depth_anchoring.model_dir)",
            ctx.label,
            depth.model_size.as_str()
        ))
    })?;
    let loss =
        candle_gen_depth::anchor::DepthAnchorLoss::from_dir(dir, depth.model_size, ctx.device)
            .map_err(|e| {
                CandleError::Msg(format!(
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
    candle_gen_depth::anchor::depth_anchor_footprint(cfg.depth_anchoring.model_size, h, w)
}

/// Every auxiliary loss, in loss-index order. **Extension point**: later stories append an arm.
pub const ARMS: &[AuxArm] = &[AuxArm {
    name: "depth",
    enabled: |cfg| cfg.depth_anchoring.schedule.is_enabled(),
    input: PerceptualInput::DecodedPixels,
    footprint: depth_footprint,
    build: build_depth,
}];

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
            return Err(CandleError::Msg(format!(
                "{}: the '{}' loss decodes x0 to pixels, but this trainer's latent family has no \
                 x0 decoder",
                ctx.label, a.name
            )))
        }
        (Some(_), DecoderSpec::Tiny { name, config }) => {
            let dir = cfg.perceptual_decoder_dir.as_ref().ok_or_else(|| {
                CandleError::Msg(format!(
                    "{}: the perceptual losses need the {name} decoder (perceptual_decoder_dir)",
                    ctx.label
                ))
            })?;
            let dec = TinyDecoder::from_dir(dir, config.clone(), ctx.device).map_err(|e| {
                CandleError::Msg(format!(
                    "{}: could not load the {name} decoder from {}: {e}",
                    ctx.label,
                    dir.display()
                ))
            })?;
            Some(Box::new(dec))
        }
        (Some(_), DecoderSpec::Custom(c)) => Some(
            c.load(cfg.perceptual_decoder_dir.as_deref(), ctx.device)
                .map_err(|e| {
                    CandleError::Msg(format!(
                        "{}: could not load the {} decoder: {e}",
                        ctx.label,
                        c.name()
                    ))
                })?,
        ),
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

/// Test fixtures for trainers on this seam: tiny random-init decoder + DA2 checkpoints written to a
/// directory, so `build_perceptual_path` runs its real load path without real weights.
pub mod testing {
    use std::path::Path;

    use candle_gen::candle_core::{safetensors, Device, Tensor};
    use candle_gen::train::tae::{synthetic_tiny_decoder_weights, TinyDecoderConfig};
    use candle_gen::Result;

    /// The tiny TAESD-shaped decoder config tests use (`latent_channels` channels, width 8).
    pub fn tiny_decoder_config(latent_channels: usize) -> TinyDecoderConfig {
        TinyDecoderConfig {
            latent_channels,
            channels: 8,
            blocks: [3, 3, 3, 1],
        }
    }

    /// A ready depth-anchoring [`PerceptualPath`](candle_gen::train::perceptual::PerceptualPath)
    /// with a random-init tiny decoder for `latent_channels` + a random-init tiny DA2, on `device`,
    /// scheduled by `schedule` — for trainer step tests (what [`super::build_perceptual_path`]
    /// returns for a depth job, minus the checkpoint reads).
    pub fn tiny_depth_path(
        latent_channels: usize,
        schedule: candle_gen::gen_core::train::AuxLossSchedule,
        device: &Device,
    ) -> Result<candle_gen::train::perceptual::PerceptualPath> {
        use candle_gen::train::perceptual::{AuxLoss, PerceptualPath};
        use candle_gen::train::tae::TinyDecoder;
        let cfg = tiny_decoder_config(latent_channels);
        let dec =
            TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&cfg, 11, device)?, cfg)?;
        let loss = candle_gen_depth::anchor::tiny_depth_anchor_loss(12, device)?;
        PerceptualPath::new(
            Some(Box::new(dec)),
            vec![AuxLoss {
                schedule,
                loss: Box::new(loss),
            }],
        )
    }

    fn save(map: std::collections::HashMap<String, Tensor>, path: &Path) -> Result<()> {
        safetensors::save(&map, path)?;
        Ok(())
    }

    /// Write a random-init tiny decoder checkpoint (`diffusion_pytorch_model.safetensors`) to `dir`.
    pub fn write_tiny_decoder(dir: &Path, cfg: &TinyDecoderConfig, seed: u64) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| candle_gen::CandleError::Msg(e.to_string()))?;
        let w = synthetic_tiny_decoder_weights(cfg, seed, &Device::Cpu)?;
        let map = w
            .keys()
            .map(|k| Ok((k.clone(), w.require(k)?)))
            .collect::<Result<_>>()?;
        save(map, &dir.join("diffusion_pytorch_model.safetensors"))
    }

    /// Write a random-init tiny Depth-Anything-V2 checkpoint (`model.safetensors`) to `dir`. It
    /// parses but is not a published size, so loading it as a published `DepthModelSize` fails —
    /// for loader-path tests; trainer step tests use [`tiny_depth_path`].
    pub fn write_tiny_depth(dir: &Path, seed: u64) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| candle_gen::CandleError::Msg(e.to_string()))?;
        let cfg = candle_gen_depth::anchor::tiny_config();
        let w = candle_gen_depth::anchor::synthetic_weights(&cfg, seed, &Device::Cpu)?;
        let map = w
            .keys()
            .map(|k| Ok((k.clone(), w.require(k)?)))
            .collect::<Result<_>>()?;
        save(map, &dir.join("model.safetensors"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Tensor;
    use candle_gen::gen_core::train::{AuxLossSchedule, DepthModelSize};
    use candle_gen::train::perceptual::{reference_as, LossReference, PerceptualLoss};
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
            config: TinyDecoderConfig::taef1(),
        }
    }

    /// E1: nothing enabled ⇒ no path, nothing loaded, zero footprint. Mutation: return
    /// `Some(empty path)` ⇒ red.
    #[test]
    fn everything_off_builds_nothing() {
        let dev = Device::Cpu;
        let ctx = AuxLossContext {
            label: "t",
            decoder: taef1(),
            latent_lpips: None,
            device: &dev,
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

    /// E7: depth on adds the decoder + DA2 (larger for Large), frames scale the per-frame terms.
    /// Mutations: drop the decoder term ⇒ the TAEF1-vs-None difference vanishes ⇒ red; drop the
    /// `frames` scaling ⇒ red.
    #[test]
    fn footprint_counts_decoder_losses_and_frames() {
        let mut cfg = on();
        let g = AuxGeometry::image(1024, 4);
        let small = perceptual_footprint(&cfg, &taef1(), g);
        let dec = TinyDecoderConfig::taef1().footprint(1024, 1024);
        let da2 =
            candle_gen_depth::anchor::depth_anchor_footprint(DepthModelSize::Small, 1024, 1024);
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
        let dev = Device::Cpu;
        let none = AuxLossContext {
            label: "fam trainer",
            decoder: DecoderSpec::None,
            latent_lpips: None,
            device: &dev,
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
            device: &dev,
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

    /// The real load path: a tiny decoder checkpoint on disk + a DA2 dir that does not hold a
    /// Small checkpoint ⇒ the DA2 load error names Depth-Anything-V2 (the decoder loaded first).
    #[test]
    fn decoder_loads_from_disk_before_the_losses() {
        let tmp = tempfile::tempdir().unwrap();
        let dec_dir = tmp.path().join("tae");
        let cfg4 = testing::tiny_decoder_config(4);
        testing::write_tiny_decoder(&dec_dir, &cfg4, 1).unwrap();
        let da2_dir = tmp.path().join("da2");
        testing::write_tiny_depth(&da2_dir, 2).unwrap();
        let mut c = on();
        c.perceptual_decoder_dir = Some(dec_dir);
        c.depth_anchoring.model_dir = Some(da2_dir);
        let dev = Device::Cpu;
        let ctx = AuxLossContext {
            label: "t",
            decoder: DecoderSpec::Tiny {
                name: "TINY",
                config: cfg4,
            },
            latent_lpips: None,
            device: &dev,
        };
        let e = build_perceptual_path(&c, &ctx).err().unwrap().to_string();
        assert!(e.contains("Depth-Anything-V2"), "{e}");
    }

    /// A latent-input arm (the shape of S12's E-LatentLPIPS) runs on a family with no decoder:
    /// nothing is decoded, the footprint has no decoder term. Mutation: require a decoder for any
    /// enabled arm ⇒ red.
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
            fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
                Ok(Some(Box::new(clean.mean_all()?)))
            }
            fn loss(&self, live: &Tensor, r: &dyn Any) -> Result<Tensor> {
                let r = reference_as::<Tensor>("latent", r)?;
                Ok((live.mean_all()? - r)?.sqr()?)
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
        let dev = Device::Cpu;
        let ctx = AuxLossContext {
            label: "t",
            decoder: DecoderSpec::None,
            latent_lpips: None,
            device: &dev,
        };
        let cfg = TrainingConfig::default();
        let mut path = build_perceptual_path_with(&arms, &cfg, &ctx)
            .unwrap()
            .unwrap();
        let z = Tensor::ones((1, 4, 2, 2), candle_gen::candle_core::DType::F32, &dev).unwrap();
        path.ensure_reference(0, &z).unwrap();
        let plan = path.plan(1, 0, 0.5).unwrap();
        assert!(path.aux_loss(&plan, 0, &z).unwrap().is_some());
        assert_eq!(
            perceptual_footprint_with(&arms, &cfg, &taef1(), AuxGeometry::image(64, 1)),
            5
        );
    }
}
