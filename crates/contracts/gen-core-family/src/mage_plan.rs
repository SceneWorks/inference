//! Weights-free pre-load execution plan for Mage-Flow generation and instruction editing
//! (`PIPE-13`, `DEC-3`, `DEC-13`).
//!
//! Captures the shared host contract across `mlx-gen-mage` and `candle-gen-mage`:
//! - the six checkpoints (`mage_flow`, `mage_flow_base`, `mage_flow_turbo`, `mage_flow_edit`,
//!   `mage_flow_edit_base`, `mage_flow_edit_turbo`),
//! - ATen-exact static-shift `6.0` rectified-flow schedule (including Turbo `N = 4`),
//! - CFG enablement (`use_neg = cfg != 1.0`) and invalid negative-prompt refusal at `cfg == 1.0`,
//! - mandatory Gaussian-Shading watermarked initial latent noise (never plain `randn`),
//! - Edit `[noisy_target, ref_1..ref_N]` target-first clean reference re-concatenation, and
//! - explicit `PIPE-13` batch semantics (packed native resolution up to `50_000` image tokens on MLX
//!   generation vs sequential single-sample execution on Candle and Edit).

use gen_core::sampling::flow_schedule_with_shift;
use gen_core::{Conditioning, Error, GenerationRequest, ModelDescriptor, Result};
use serde::{Deserialize, Serialize};

/// Published Mage-Flow static flow-match shift.
pub const MAGE_STATIC_SHIFT: f32 = 6.0;
/// Upstream native-resolution image-token budget for one packed MLX DiT invocation (`PIPE-13`).
pub const MAGE_MAX_PACKED_IMAGE_TOKENS: usize = 50_000;
/// Spatial downsample multiple required on both width and height.
pub const MAGE_SIZE_MULTIPLE: u32 = 16;
/// VAE latent channel width.
pub const MAGE_LATENT_CHANNELS: usize = 128;
/// Default Gaussian-Shading numeric watermark key (`2026-07-20`).
pub const MAGE_DEFAULT_WATERMARK_KEY: u64 = 20_260_720;
/// Published default watermark payload identifier.
pub const MAGE_WATERMARK_PAYLOAD: &str = "MageFlow";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MageBackendKind {
    Mlx,
    Candle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MageTaskKind {
    Generation,
    Edit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MagePlanVariant {
    Rl,
    Base,
    Turbo,
    Edit,
    EditBase,
    EditTurbo,
}

impl MagePlanVariant {
    pub fn from_id(id: &str) -> Option<Self> {
        Some(match id {
            "mage_flow" => Self::Rl,
            "mage_flow_base" => Self::Base,
            "mage_flow_turbo" => Self::Turbo,
            "mage_flow_edit" => Self::Edit,
            "mage_flow_edit_base" => Self::EditBase,
            "mage_flow_edit_turbo" => Self::EditTurbo,
            _ => return None,
        })
    }

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

    pub const fn task(self) -> MageTaskKind {
        if self.is_edit() {
            MageTaskKind::Edit
        } else {
            MageTaskKind::Generation
        }
    }

    pub const fn is_edit(self) -> bool {
        matches!(self, Self::Edit | Self::EditBase | Self::EditTurbo)
    }

    pub const fn is_turbo(self) -> bool {
        matches!(self, Self::Turbo | Self::EditTurbo)
    }

    pub const fn default_steps(self) -> u32 {
        match self {
            Self::Rl => 20,
            Self::Base | Self::Edit | Self::EditBase => 30,
            Self::Turbo | Self::EditTurbo => 4,
        }
    }

    pub const fn default_cfg(self) -> f32 {
        if self.is_turbo() {
            1.0
        } else {
            5.0
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MageSeedDerivation {
    WrappingI64Offset,
    WrappingU64Offset,
}

impl MageSeedDerivation {
    pub const fn derive(self, base_seed: u64, index: u32) -> u64 {
        match self {
            Self::WrappingI64Offset => (base_seed as i64).wrapping_add(index as i64) as u64,
            Self::WrappingU64Offset => base_seed.wrapping_add(index as u64),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MageWatermarkPolicy {
    pub mandatory: bool,
    pub forbid_plain_randn: bool,
    pub scheme: String,
    pub default_key_u64: u64,
    pub payload: String,
}

impl MageWatermarkPolicy {
    pub fn for_backend(backend: MageBackendKind) -> Self {
        let scheme = match backend {
            MageBackendKind::Mlx => "gaussian_shading_pcg64_mt19937_ndtri".to_owned(),
            MageBackendKind::Candle => "mage-pcg-watermark-v1".to_owned(),
        };
        Self {
            mandatory: true,
            forbid_plain_randn: true,
            scheme,
            default_key_u64: MAGE_DEFAULT_WATERMARK_KEY,
            payload: MAGE_WATERMARK_PAYLOAD.to_owned(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MagePackedGroupPlan {
    pub sample_range_start: usize,
    pub sample_range_end: usize,
    pub grids: Vec<(u32, u32)>,
    pub image_tokens: usize,
    pub sample_seeds: Vec<u64>,
}

/// Explicit `PIPE-13` batch-semantics declaration differentiating MLX native packed batches from
/// sequential single-sample execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum MageBatchSemantics {
    PackedNativeResolution {
        max_packed_image_tokens: usize,
        packs: Vec<MagePackedGroupPlan>,
        seed_derivation: MageSeedDerivation,
        single_sample_trajectory_only: bool,
        single_decode_phase_boundary: bool,
    },
    SequentialSingleSample {
        count: u32,
        sample_seeds: Vec<u64>,
        seed_derivation: MageSeedDerivation,
        stage_residency_requires_count_one: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MageEditSequencePlan {
    pub reference_count: usize,
    pub sequence_order: String,
    pub target_frame_index: u32,
    pub reference_frame_indices: Vec<u32>,
    pub clean_refs_reconcatenated_every_step: bool,
    pub only_target_tokens_stepped: bool,
}

/// Complete pre-load serializable execution plan built by Mage `validate()`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MageRenderPlan {
    pub provider_id: String,
    pub variant: MagePlanVariant,
    pub backend: MageBackendKind,
    pub task: MageTaskKind,
    pub width: u32,
    pub height: u32,
    pub latent_grid: (u32, u32),
    pub count: u32,
    pub steps: usize,
    pub guidance: f32,
    pub use_neg: bool,
    pub is_turbo: bool,
    pub static_shift: f32,
    pub scheduler_shift: Option<f32>,
    pub sigmas: Vec<f32>,
    pub base_seed: u64,
    pub watermark: MageWatermarkPolicy,
    pub batch_semantics: MageBatchSemantics,
    pub edit_sequence: Option<MageEditSequencePlan>,
    pub euler_precision: String,
}

/// Build the exact Diffusers `FlowMatchEulerDiscreteScheduler` static-shift (`6.0`) sigma ladder
/// matching ATen's endpoint-aware `linspace(1, 1/N, N)`.
pub fn mage_flow_sigmas_pure(steps: usize) -> Result<Vec<f32>> {
    if steps == 0 {
        return Err(Error::Msg(
            "mage_flow: steps must be greater than zero".into(),
        ));
    }
    let n = steps as f32;
    let start = 1.0f32;
    let end = 1.0 / n;
    let step = if steps > 1 {
        (end - start) / (steps - 1) as f32
    } else {
        0.0
    };
    let mut sigmas = (0..steps)
        .map(|i| {
            let s = if i < steps / 2 {
                step.mul_add(i as f32, start)
            } else {
                (-step).mul_add((steps - i - 1) as f32, end)
            };
            MAGE_STATIC_SHIFT * s / (1.0 + (MAGE_STATIC_SHIFT - 1.0) * s)
        })
        .collect::<Vec<_>>();
    sigmas.push(0.0);
    Ok(sigmas)
}

/// Partition `(grid, seed)` requests greedily under [`MAGE_MAX_PACKED_IMAGE_TOKENS`].
pub fn plan_mage_packed_groups(
    samples: &[((u32, u32), u64)],
    max_packed_image_tokens: usize,
) -> Result<Vec<MagePackedGroupPlan>> {
    if samples.is_empty() {
        return Err(Error::Msg(
            "mage_flow: generation batch must not be empty".into(),
        ));
    }
    let mut packs = Vec::new();
    let mut start = 0usize;
    let mut grids = Vec::new();
    let mut seeds = Vec::new();
    let mut tokens = 0usize;
    for (index, &(grid, seed)) in samples.iter().enumerate() {
        let sample_tokens = grid.0 as usize * grid.1 as usize;
        if !grids.is_empty() && tokens + sample_tokens > max_packed_image_tokens {
            packs.push(MagePackedGroupPlan {
                sample_range_start: start,
                sample_range_end: index,
                grids: std::mem::take(&mut grids),
                image_tokens: tokens,
                sample_seeds: std::mem::take(&mut seeds),
            });
            start = index;
            tokens = 0;
        }
        grids.push(grid);
        seeds.push(seed);
        tokens += sample_tokens;
    }
    packs.push(MagePackedGroupPlan {
        sample_range_start: start,
        sample_range_end: samples.len(),
        grids,
        image_tokens: tokens,
        sample_seeds: seeds,
    });
    Ok(packs)
}

/// Validate `req` against `descriptor` and construct the pre-load [`MageRenderPlan`].
pub fn build_mage_render_plan(
    descriptor: &ModelDescriptor,
    backend: MageBackendKind,
    req: &GenerationRequest,
    require_edit_references: bool,
) -> Result<MageRenderPlan> {
    descriptor
        .capabilities
        .validate_request(descriptor.id, req)?;
    let variant = MagePlanVariant::from_id(descriptor.id).ok_or_else(|| {
        Error::Unsupported(format!("unknown Mage-Flow provider {}", descriptor.id))
    })?;
    if req.steps == Some(0) {
        return Err(Error::Msg(format!("{}: steps must be >= 1", descriptor.id)));
    }
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!(
            "{}: prompt must not be empty",
            descriptor.id
        )));
    }
    let steps = req.steps.unwrap_or(variant.default_steps()) as usize;
    let guidance = req.guidance.unwrap_or(variant.default_cfg());
    if guidance == 1.0
        && req
            .negative_prompt
            .as_ref()
            .is_some_and(|text| !text.trim().is_empty())
    {
        return Err(Error::Msg(format!(
            "{}: a negative prompt requires guidance != 1.0",
            descriptor.id
        )));
    }
    if !req.width.is_multiple_of(MAGE_SIZE_MULTIPLE)
        || !req.height.is_multiple_of(MAGE_SIZE_MULTIPLE)
    {
        return Err(Error::Msg(format!(
            "mage_flow dimensions must be divisible by {MAGE_SIZE_MULTIPLE}"
        )));
    }

    let edit_sequence = if variant.is_edit() {
        if req.strength.is_some() {
            return Err(Error::Unsupported(
                "mage edit uses full reference-token conditioning and does not support request \
                 strength"
                    .into(),
            ));
        }
        let mut ref_count = 0usize;
        for conditioning in &req.conditioning {
            match conditioning {
                Conditioning::Reference { image, strength } => {
                    if strength.is_some() {
                        return Err(Error::Unsupported(
                            "mage edit uses full reference-token conditioning and does not \
                             support per-reference strength"
                                .into(),
                        ));
                    }
                    validate_rgb8_image(ref_count, image)?;
                    ref_count += 1;
                }
                Conditioning::MultiReference { images } => {
                    for image in images {
                        validate_rgb8_image(ref_count, image)?;
                        ref_count += 1;
                    }
                }
                _ => {}
            }
        }
        if require_edit_references && ref_count == 0 {
            return Err(Error::Msg(
                "mage edit requires Reference or MultiReference conditioning".into(),
            ));
        }
        Some(MageEditSequencePlan {
            reference_count: ref_count,
            sequence_order: "[noisy_target, ref_1..ref_N]".to_owned(),
            target_frame_index: 0,
            reference_frame_indices: (1..=ref_count as u32).collect(),
            clean_refs_reconcatenated_every_step: true,
            only_target_tokens_stepped: true,
        })
    } else {
        None
    };

    let native = mage_flow_sigmas_pure(steps)?;
    let sigmas = flow_schedule_with_shift(
        None,
        MAGE_STATIC_SHIFT.ln(),
        req.scheduler_shift,
        steps,
        &native,
    )?;

    let latent_grid = (
        req.height / MAGE_SIZE_MULTIPLE,
        req.width / MAGE_SIZE_MULTIPLE,
    );
    let base_seed = req.seed.unwrap_or(0);
    let seed_derivation = match backend {
        MageBackendKind::Mlx => MageSeedDerivation::WrappingI64Offset,
        MageBackendKind::Candle => MageSeedDerivation::WrappingU64Offset,
    };
    let sample_seeds: Vec<u64> = (0..req.count)
        .map(|index| seed_derivation.derive(base_seed, index))
        .collect();

    let batch_semantics = match (backend, variant.task()) {
        (MageBackendKind::Mlx, MageTaskKind::Generation) => {
            let entries: Vec<((u32, u32), u64)> = sample_seeds
                .iter()
                .map(|&seed| (latent_grid, seed))
                .collect();
            let packs = plan_mage_packed_groups(&entries, MAGE_MAX_PACKED_IMAGE_TOKENS)?;
            MageBatchSemantics::PackedNativeResolution {
                max_packed_image_tokens: MAGE_MAX_PACKED_IMAGE_TOKENS,
                packs,
                seed_derivation,
                single_sample_trajectory_only: true,
                single_decode_phase_boundary: true,
            }
        }
        _ => MageBatchSemantics::SequentialSingleSample {
            count: req.count,
            sample_seeds,
            seed_derivation,
            stage_residency_requires_count_one: matches!(backend, MageBackendKind::Candle),
        },
    };

    Ok(MageRenderPlan {
        provider_id: descriptor.id.to_owned(),
        variant,
        backend,
        task: variant.task(),
        width: req.width,
        height: req.height,
        latent_grid,
        count: req.count,
        steps,
        guidance,
        use_neg: guidance != 1.0,
        is_turbo: variant.is_turbo(),
        static_shift: MAGE_STATIC_SHIFT,
        scheduler_shift: req.scheduler_shift,
        sigmas,
        base_seed,
        watermark: MageWatermarkPolicy::for_backend(backend),
        batch_semantics,
        edit_sequence,
        euler_precision: "f32_accumulate_cast_output_dtype_once".to_owned(),
    })
}

fn validate_rgb8_image(index: usize, image: &gen_core::Image) -> Result<()> {
    let expected = image.width as usize * image.height as usize * 3;
    if image.width == 0 || image.height == 0 || image.pixels.len() != expected {
        return Err(Error::Msg(format!(
            "mage edit reference {index} is not valid RGB8"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{Capabilities, ConditioningKind, Image, Modality};

    fn mage_descriptor(id: &'static str) -> ModelDescriptor {
        let is_edit = MagePlanVariant::from_id(id).unwrap().is_edit();
        ModelDescriptor {
            encoder_contract: None,
            denoiser_output_latent_space: Some(&gen_core::MAGE_LATENT_SPACE),
            control_kinds: None,
            required_components: &["text_encoder", "vae"],
            id,
            family: "mage",
            backend: "test",
            modality: Modality::Image,
            capabilities: Capabilities {
                supports_negative_prompt: true,
                supports_guidance: true,
                conditioning: if is_edit {
                    vec![
                        ConditioningKind::Reference,
                        ConditioningKind::MultiReference,
                    ]
                } else {
                    vec![]
                },
                min_size: 512,
                max_size: 2048,
                max_count: 8,
                ..Default::default()
            },
        }
    }

    #[test]
    fn mage_plan_round_trips_json_and_pins_contracts() {
        let gen_req = GenerationRequest {
            prompt: "a lantern in fog".into(),
            width: 1024,
            height: 1024,
            count: 2,
            seed: Some(77),
            ..Default::default()
        };
        let plan = build_mage_render_plan(
            &mage_descriptor("mage_flow_turbo"),
            MageBackendKind::Mlx,
            &gen_req,
            true,
        )
        .unwrap();
        assert_eq!(plan.steps, 4);
        assert_eq!(plan.guidance, 1.0);
        assert!(!plan.use_neg);
        assert!(plan.is_turbo);
        assert_eq!(plan.static_shift, 6.0);
        assert!(plan.watermark.mandatory);
        assert!(plan.watermark.forbid_plain_randn);

        let json = serde_json::to_string(&plan).unwrap();
        let decoded: MageRenderPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, plan);
    }

    #[test]
    fn mage_plan_captures_pipe13_packed_mlx_vs_sequential_candle_count_gt_1() {
        // 4 samples at 2048x2048 -> 128x128 = 16_384 tokens each -> 3 fit in pack 0 (49_152 <= 50_000),
        // 1 spills into pack 1 (16_384) on MLX T2I, while Candle plans sequential single-sample execution.
        let req = GenerationRequest {
            prompt: "packed native check".into(),
            width: 2048,
            height: 2048,
            count: 4,
            seed: Some(100),
            ..Default::default()
        };
        let mlx_plan = build_mage_render_plan(
            &mage_descriptor("mage_flow"),
            MageBackendKind::Mlx,
            &req,
            true,
        )
        .unwrap();
        match mlx_plan.batch_semantics {
            MageBatchSemantics::PackedNativeResolution {
                max_packed_image_tokens,
                packs,
                single_sample_trajectory_only,
                single_decode_phase_boundary,
                ..
            } => {
                assert_eq!(max_packed_image_tokens, 50_000);
                assert!(single_sample_trajectory_only);
                assert!(single_decode_phase_boundary);
                assert_eq!(packs.len(), 2);
                assert_eq!(packs[0].sample_range_start, 0);
                assert_eq!(packs[0].sample_range_end, 3);
                assert_eq!(packs[0].image_tokens, 49_152);
                assert_eq!(packs[0].sample_seeds, vec![100, 101, 102]);
                assert_eq!(packs[1].sample_range_start, 3);
                assert_eq!(packs[1].sample_range_end, 4);
                assert_eq!(packs[1].image_tokens, 16_384);
                assert_eq!(packs[1].sample_seeds, vec![103]);
            }
            other => panic!("expected MLX packed semantics, got {other:?}"),
        }

        let candle_plan = build_mage_render_plan(
            &mage_descriptor("mage_flow"),
            MageBackendKind::Candle,
            &req,
            true,
        )
        .unwrap();
        match candle_plan.batch_semantics {
            MageBatchSemantics::SequentialSingleSample {
                count,
                sample_seeds,
                stage_residency_requires_count_one,
                ..
            } => {
                assert_eq!(count, 4);
                assert_eq!(sample_seeds, vec![100, 101, 102, 103]);
                assert!(stage_residency_requires_count_one);
            }
            other => panic!("expected Candle sequential semantics, got {other:?}"),
        }
    }

    #[test]
    fn mage_plan_refuses_invalid_combinations_and_records_edit_sequence() {
        let base_desc = mage_descriptor("mage_flow_turbo");
        let neg_with_cfg1 = GenerationRequest {
            prompt: "test".into(),
            width: 1024,
            height: 1024,
            negative_prompt: Some("bad".into()),
            ..Default::default()
        };
        assert!(
            build_mage_render_plan(&base_desc, MageBackendKind::Candle, &neg_with_cfg1, true)
                .is_err()
        );

        let off_grid = GenerationRequest {
            prompt: "test".into(),
            width: 1000,
            height: 1024,
            ..Default::default()
        };
        assert!(build_mage_render_plan(&base_desc, MageBackendKind::Mlx, &off_grid, true).is_err());

        let edit_desc = mage_descriptor("mage_flow_edit");
        let missing_refs = GenerationRequest {
            prompt: "edit".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(
            build_mage_render_plan(&edit_desc, MageBackendKind::Candle, &missing_refs, true)
                .is_err()
        );

        let ref_img = Image {
            width: 4,
            height: 4,
            pixels: vec![0; 4 * 4 * 3],
        };
        let with_strength = GenerationRequest {
            prompt: "edit".into(),
            width: 1024,
            height: 1024,
            strength: Some(0.5),
            conditioning: vec![Conditioning::Reference {
                image: ref_img.clone(),
                strength: None,
            }],
            ..Default::default()
        };
        let err = build_mage_render_plan(&edit_desc, MageBackendKind::Mlx, &with_strength, true)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));

        let valid_edit = GenerationRequest {
            prompt: "edit".into(),
            width: 1024,
            height: 1024,
            conditioning: vec![Conditioning::MultiReference {
                images: vec![ref_img.clone(), ref_img],
            }],
            ..Default::default()
        };
        let plan =
            build_mage_render_plan(&edit_desc, MageBackendKind::Mlx, &valid_edit, true).unwrap();
        let edit_seq = plan.edit_sequence.expect("edit plan has sequence spec");
        assert_eq!(edit_seq.reference_count, 2);
        assert_eq!(edit_seq.target_frame_index, 0);
        assert_eq!(edit_seq.reference_frame_indices, vec![1, 2]);
        assert!(edit_seq.clean_refs_reconcatenated_every_step);
        assert!(edit_seq.only_target_tokens_stepped);
        assert!(matches!(
            plan.batch_semantics,
            MageBatchSemantics::SequentialSingleSample { .. }
        ));

        for bad_shift in [0.0, -2.0, f32::NAN] {
            let bad_req = GenerationRequest {
                scheduler_shift: Some(bad_shift),
                ..gen_shift_req()
            };
            assert!(
                build_mage_render_plan(&base_desc, MageBackendKind::Mlx, &bad_req, true).is_err()
            );
        }
        let shifted = build_mage_render_plan(
            &base_desc,
            MageBackendKind::Mlx,
            &GenerationRequest {
                scheduler_shift: Some(3.0),
                ..gen_shift_req()
            },
            true,
        )
        .unwrap();
        let unshifted =
            build_mage_render_plan(&base_desc, MageBackendKind::Mlx, &gen_shift_req(), true)
                .unwrap();
        assert_ne!(shifted.sigmas, unshifted.sigmas);
    }

    fn gen_shift_req() -> GenerationRequest {
        GenerationRequest {
            prompt: "shift check".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        }
    }
}
