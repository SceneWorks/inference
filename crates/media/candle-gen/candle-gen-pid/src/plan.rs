//! Pre-load execution planning for Candle PiD pixel-diffusion decode (`E6 / AC1`).
//!
//! Validates student schedule constants, `from_ldm` early-stop `capture_sigma`, spatial tiling
//! candidates / [`DecodeRoutes`] admission, memory budget bounds ([`crate::budget`]), and S10
//! (`RenderStage::Decode`) plan construction before running pixel-space decode compute.

use candle_gen::gen_core::{
    self, Error as CoreError, GenerationRequest, Modality, ModelDescriptor, RenderPlan,
    Result as CoreResult,
};
use candle_gen::{CandleError, Result};

use crate::budget;
use crate::config::{PidConfig, SampleType, SamplerConfig};
use crate::decode_routes::DecodeRoutes;
use crate::sampler::Sampler;

/// Released distilled 4-step student schedule (`student_t_list`, descending to 0.0).
pub const STUDENT_SCHEDULE: [f32; 5] = [0.999, 0.866, 0.634, 0.342, 0.0];

/// Released flow-matching timestep scale factor (`t * fm_timescale`).
pub const FM_TIMESCALE: f32 = 1000.0;

fn route_pid_decode() -> gen_core::RouteRecord {
    gen_core::RouteRecord {
        route_id: "pid",
        family: "pid",
        backend: "candle",
        modality: Modality::Image,
        mac_only: false,
        structure: gen_core::RouteStructure {
            denoiser: gen_core::ModelBinding::family_driver(gen_core::FamilyDriverSpec::new(
                crate::sampler::pid_driver_support(),
            )),
            ..Default::default()
        },
        limits: gen_core::ProductLimits::default(),
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::DeterministicOffset,
            max_count: 0,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults::default(),
        bindings: gen_core::RouteBindings::default(),
    }
}

fn pid_decode_descriptor() -> ModelDescriptor {
    route_pid_decode().compute_descriptor()
}

/// Spatial-tiling resolution plan for one PiD decode invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PidTilingPlan {
    pub tile_edge_px: i32,
    pub overlap_px: i32,
    pub whole_fits: bool,
    pub selected_by_request: bool,
}

/// Validated pre-decode PiD plan covering schedule, `from_ldm` degrade σ, memory budget, and
/// `DecodeRoutes` / spatial tiling admission (`E6 / AC1`).
#[derive(Clone, Debug, PartialEq)]
pub struct PidDecodePlan {
    pub model_id: String,
    pub sample_type: SampleType,
    pub student_t_list: Vec<f32>,
    pub fm_timescale: f32,
    pub capture_sigma: f32,
    pub scale: i32,
    pub vae_compression: i32,
    pub width: u32,
    pub height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub count: u32,
    pub seed: u64,
    pub safe_gib: f64,
    pub tiling: PidTilingPlan,
}

impl PidDecodePlan {
    /// Validate a [`SamplerConfig`] schedule and `from_ldm` degrade σ.
    pub fn validate_sampler_config(
        model_id: &str,
        sampler: &SamplerConfig,
        capture_sigma: f32,
    ) -> Result<()> {
        if sampler.student_t_list.len() < 2
            || sampler
                .student_t_list
                .iter()
                .any(|t| !t.is_finite() || *t < 0.0)
            || sampler.student_t_list.windows(2).any(|w| w[0] < w[1])
            || sampler.student_t_list.last() != Some(&0.0)
        {
            return Err(CandleError::from(CoreError::Unsupported(format!(
                "{model_id}: PiD sampler schedule must descend monotonically to 0.0"
            ))));
        }
        if !sampler.fm_timescale.is_finite() || sampler.fm_timescale <= 0.0 {
            return Err(CandleError::from(CoreError::Unsupported(format!(
                "{model_id}: PiD fm_timescale must be finite and positive"
            ))));
        }
        if !capture_sigma.is_finite() || capture_sigma < 0.0 {
            return Err(CandleError::from(CoreError::Unsupported(format!(
                "{model_id}: PiD capture_sigma (from_ldm) must be finite and non-negative (got {capture_sigma})"
            ))));
        }
        Ok(())
    }

    /// Build a standalone lightweight plan directly from an already-constructed `Sampler` and
    /// `PidDecoder` parameters (used when callers invoke `PidDecoder::new` without `PidEngine`).
    pub(crate) fn from_decoder_parts(
        sampler: &Sampler,
        capture_sigma: f32,
        scale: i32,
        vae_compression: i32,
        seed: u64,
        tile: Option<(i32, i32)>,
    ) -> Self {
        let tiling = match tile {
            Some((edge, overlap)) => PidTilingPlan {
                tile_edge_px: edge,
                overlap_px: overlap,
                whole_fits: false,
                selected_by_request: false,
            },
            None => PidTilingPlan {
                tile_edge_px: budget::PREFERRED_TILE_EDGE,
                overlap_px: budget::DEFAULT_TILE_OVERLAP,
                whole_fits: true,
                selected_by_request: false,
            },
        };
        Self {
            model_id: "pid".to_owned(),
            sample_type: sampler.sample_type(),
            student_t_list: sampler.t_list().to_vec(),
            fm_timescale: sampler.timescale(),
            capture_sigma,
            scale,
            vae_compression,
            width: 0,
            height: 0,
            output_width: 0,
            output_height: 0,
            count: 1,
            seed,
            safe_gib: f64::INFINITY,
            tiling,
        }
    }

    /// Compile and validate a full pre-decode plan from request-level / field-level parameters,
    /// enforcing budget admission and optional [`DecodeRoutes`] verification before decoding.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_for_fields(
        model_id: &str,
        cfg: &PidConfig,
        sampler_cfg: &SamplerConfig,
        decode_routes: Option<&DecodeRoutes>,
        width: u32,
        height: u32,
        count: u32,
        capture_sigma: f32,
        seed: u64,
        safe_gib: f64,
        selected_tiling: Option<(i32, i32)>,
    ) -> Result<Self> {
        Self::validate_sampler_config(model_id, sampler_cfg, capture_sigma)?;
        if width == 0 || height == 0 || cfg.sr_scale <= 0 || cfg.latent_spatial_down_factor <= 0 {
            return Err(CandleError::from(CoreError::Unsupported(format!(
                "{model_id}: PiD decode requires positive geometry and scale"
            ))));
        }
        if !safe_gib.is_finite() || safe_gib <= 0.0 {
            return Err(CandleError::from(CoreError::Unsupported(format!(
                "{model_id}: PiD safe memory budget must be finite and positive (got {safe_gib})"
            ))));
        }

        let scale_u32 = cfg.sr_scale as u32;
        let output_width = width
            .checked_mul(scale_u32)
            .ok_or_else(|| CandleError::Msg(format!("{model_id}: PiD output width overflow")))?;
        let output_height = height
            .checked_mul(scale_u32)
            .ok_or_else(|| CandleError::Msg(format!("{model_id}: PiD output height overflow")))?;
        let th = i32::try_from(output_height)
            .map_err(|_| CandleError::Msg(format!("{model_id}: PiD output height exceeds i32")))?;
        let tw = i32::try_from(output_width)
            .map_err(|_| CandleError::Msg(format!("{model_id}: PiD output width exceeds i32")))?;

        budget::guard(model_id, count, width, height, cfg.sr_scale, cfg, safe_gib)?;

        Self::validate_selected_tiling(selected_tiling, decode_routes)?;

        let tiling = match selected_tiling {
            Some((edge, overlap)) => {
                budget::validate_tile(model_id, edge, overlap, th, tw)?;
                let selected_peak_gib = budget::tiled_peak_gib(
                    count.max(1) as i32,
                    th,
                    tw,
                    edge,
                    cfg.patch_size,
                    cfg.hidden_size,
                );
                if selected_peak_gib > safe_gib {
                    return Err(CandleError::from(CoreError::Unsupported(format!(
                        "{model_id}: selected PiD tile {edge}px for {output_width}x{output_height} \
                         needs ~{selected_peak_gib:.2} GiB, exceeding the {safe_gib:.2} GiB safe \
                         memory budget"
                    ))));
                }
                PidTilingPlan {
                    tile_edge_px: edge,
                    overlap_px: overlap,
                    whole_fits: false,
                    selected_by_request: true,
                }
            }
            None => {
                let plan = budget::plan_tile_edge(
                    count.max(1) as i32,
                    th,
                    tw,
                    cfg.patch_size,
                    cfg.hidden_size,
                    safe_gib,
                );
                PidTilingPlan {
                    tile_edge_px: plan.edge,
                    overlap_px: plan.overlap,
                    whole_fits: plan.whole_fits,
                    selected_by_request: false,
                }
            }
        };

        Ok(Self {
            model_id: model_id.to_owned(),
            sample_type: sampler_cfg.sample_type,
            student_t_list: sampler_cfg.student_t_list.clone(),
            fm_timescale: sampler_cfg.fm_timescale,
            capture_sigma,
            scale: cfg.sr_scale,
            vae_compression: cfg.latent_spatial_down_factor,
            width,
            height,
            output_width,
            output_height,
            count: count.max(1),
            seed,
            safe_gib,
            tiling,
        })
    }

    /// Validate explicit `(tile_edge, overlap)` against `decode_routes`, falling back to
    /// [`DecodeRoutes::default_pid`] when `decode_routes` is `None`.
    pub(crate) fn validate_selected_tiling(
        selected_tiling: Option<(i32, i32)>,
        decode_routes: Option<&DecodeRoutes>,
    ) -> Result<()> {
        let Some((edge, overlap)) = selected_tiling else {
            return Ok(());
        };
        let default_routes;
        let routes = match decode_routes {
            Some(routes) => routes,
            None => {
                default_routes = DecodeRoutes::default_pid();
                &default_routes
            }
        };
        routes
            .validate(true, u32::try_from(edge).ok(), u32::try_from(overlap).ok())
            .map_err(|msg| CandleError::from(CoreError::Unsupported(msg)))
    }

    /// Resolve and route-validate request-level bounded-decode tiling before weight loads.
    pub(crate) fn resolve_request_tiling(
        req: &GenerationRequest,
        decode_routes: Option<&DecodeRoutes>,
    ) -> Result<Option<(i32, i32)>> {
        let Some(memory) = req.memory else {
            return Ok(None);
        };
        if !memory.tile_vae_decode {
            return Ok(None);
        }
        if let Some(routes) = decode_routes {
            routes
                .validate(true, memory.decode_tile_edge, memory.decode_overlap)
                .map_err(|msg| CandleError::from(CoreError::Unsupported(msg)))?;
        } else if memory.decode_tile_edge.is_some() {
            let overlap = memory
                .decode_overlap
                .or(Some(crate::budget::DEFAULT_TILE_OVERLAP as u32));
            DecodeRoutes::default_pid()
                .validate(true, memory.decode_tile_edge, overlap)
                .map_err(|msg| CandleError::from(CoreError::Unsupported(msg)))?;
        } else {
            return Ok(None);
        }
        Ok(crate::engine::selected_decode_tiling(req))
    }

    /// Compile and validate a [`PidDecodePlan`] from a [`GenerationRequest`], honoring `req.use_pid`,
    /// `selected_decode_tiling(req)`, and [`DecodeRoutes`] admission (defaulting to
    /// [`DecodeRoutes::default_pid`] when `decode_routes` is `None`).
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        model_id: &str,
        cfg: &PidConfig,
        sampler_cfg: &SamplerConfig,
        decode_routes: Option<&DecodeRoutes>,
        req: &GenerationRequest,
        capture_sigma: f32,
        seed: u64,
        safe_gib: f64,
    ) -> Result<Self> {
        if !req.use_pid() {
            return Err(CandleError::from(CoreError::Unsupported(format!(
                "{model_id}: cannot compile PidDecodePlan when req.use_pid is false"
            ))));
        }
        let selected_tiling = Self::resolve_request_tiling(req, decode_routes)?;
        Self::compile_for_fields(
            model_id,
            cfg,
            sampler_cfg,
            decode_routes,
            req.width,
            req.height,
            req.count,
            capture_sigma,
            seed,
            safe_gib,
            selected_tiling,
        )
    }

    /// Effective `(tile_edge, overlap)` applied during decode, or `None` when executing whole-image.
    pub fn applied_tiling(&self) -> Option<(i32, i32)> {
        if self.tiling.whole_fits && !self.tiling.selected_by_request {
            None
        } else {
            Some((self.tiling.tile_edge_px, self.tiling.overlap_px))
        }
    }

    /// Compile an S10 (`RenderStage::Decode`) `RenderPlan` for standalone PiD decode runs.
    pub fn render_plan(
        &self,
        output_width: u32,
        output_height: u32,
        batch: u32,
    ) -> CoreResult<RenderPlan> {
        let sampler_name = match self.sample_type {
            SampleType::Sde => "pid_sde",
            SampleType::Ode => "pid_ode",
        };
        let steps = self.student_t_list.len().saturating_sub(1);
        let descriptor = pid_decode_descriptor();
        RenderPlan::decode_stage(
            &descriptor,
            output_width,
            output_height,
            batch.max(1),
            self.seed,
            sampler_name,
            self.student_t_list.clone(),
            steps,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::GenerationMemory;

    #[test]
    fn pid_decode_plan_compiles_default_schedule_and_from_ldm_capture() {
        let cfg = PidConfig::sr4x();
        let sampler = SamplerConfig::distill_4step();
        assert_eq!(sampler.student_t_list, STUDENT_SCHEDULE);
        assert_eq!(sampler.fm_timescale, FM_TIMESCALE);

        let routes = DecodeRoutes::new("qwenimage", [768, 640, 512], 64).unwrap();
        let req = GenerationRequest {
            width: 512,
            height: 512,
            count: 2,
            features: candle_gen::gen_core::FeatureParams {
                pid: Some(candle_gen::gen_core::PidParams {
                    capture_sigma: Some(0.25),
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        let plan = PidDecodePlan::compile(
            "qwenimage",
            &cfg,
            &sampler,
            Some(&routes),
            &req,
            0.25,
            7,
            96.0,
        )
        .unwrap();
        assert_eq!((plan.output_width, plan.output_height), (2048, 2048));
        assert!(plan.tiling.whole_fits);
        assert_eq!(plan.applied_tiling(), None);
        assert_eq!(plan.capture_sigma, 0.25);

        let s10 = plan.render_plan(2048, 2048, 1).unwrap();
        assert!(s10.is_decode_stage());
        assert_eq!(s10.executed_steps(), 4);
    }

    #[test]
    fn pid_decode_plan_enforces_decode_routes_and_budget_guard() {
        let cfg = PidConfig::sr4x();
        let sampler = SamplerConfig::distill_4step();
        let routes = DecodeRoutes::new("z_image_turbo", [768, 640, 512], 64).unwrap();

        let bad_tile = GenerationRequest {
            width: 512,
            height: 512,
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                ..Default::default()
            }),
            features: candle_gen::gen_core::FeatureParams {
                pid: Some(candle_gen::gen_core::PidParams::enabled()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(PidDecodePlan::compile(
            "z_image_turbo",
            &cfg,
            &sampler,
            Some(&routes),
            &bad_tile,
            0.0,
            1,
            96.0,
        )
        .is_err());

        let good_tile = GenerationRequest {
            width: 512,
            height: 512,
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(2048),
                decode_overlap: Some(256),
                ..Default::default()
            }),
            features: candle_gen::gen_core::FeatureParams {
                pid: Some(candle_gen::gen_core::PidParams::enabled()),
                ..Default::default()
            },
            ..Default::default()
        };
        let tiled = PidDecodePlan::compile(
            "z_image_turbo",
            &cfg,
            &sampler,
            Some(&routes),
            &good_tile,
            0.0,
            1,
            96.0,
        )
        .unwrap();
        assert_eq!(tiled.applied_tiling(), Some((2048, 256)));

        let huge = GenerationRequest {
            width: 2048,
            height: 2048,
            features: candle_gen::gen_core::FeatureParams {
                pid: Some(candle_gen::gen_core::PidParams::enabled()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            PidDecodePlan::compile("qwenimage", &cfg, &sampler, None, &huge, 0.0, 1, 2.0).is_err()
        );

        let unmeasured_tile = GenerationRequest {
            width: 512,
            height: 512,
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(1536),
                decode_overlap: Some(64),
                ..Default::default()
            }),
            features: candle_gen::gen_core::FeatureParams {
                pid: Some(candle_gen::gen_core::PidParams::enabled()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(PidDecodePlan::compile(
            "qwenimage",
            &cfg,
            &sampler,
            None,
            &unmeasured_tile,
            0.0,
            1,
            96.0,
        )
        .is_err());
        assert!(PidDecodePlan::compile_for_fields(
            "qwenimage",
            &cfg,
            &sampler,
            None,
            512,
            512,
            1,
            0.0,
            1,
            96.0,
            Some((1536, 64)),
        )
        .is_err());
        assert!(
            PidDecodePlan::compile("qwenimage", &cfg, &sampler, None, &bad_tile, 0.0, 1, 96.0,)
                .is_err()
        );
        let default_pid_good =
            PidDecodePlan::compile("qwenimage", &cfg, &sampler, None, &good_tile, 0.0, 1, 96.0)
                .unwrap();
        assert_eq!(default_pid_good.applied_tiling(), Some((2048, 256)));
    }

    #[test]
    fn pid_decode_plan_rejects_selected_tile_exceeding_memory_budget() {
        let cfg = PidConfig::sr4x();
        let sampler = SamplerConfig::distill_4step();
        // 1536x1536 with sr4x() -> 6144x6144 output under 3.0 GiB safe budget:
        // budget::guard (1024px min tile) fits (~2.75 GiB), so automatic planning succeeds with
        // 1024px, whereas an explicit (2048, 256) selection (~3.42 GiB) must be rejected.
        let auto_plan = PidDecodePlan::compile_for_fields(
            "qwenimage",
            &cfg,
            &sampler,
            None,
            1536,
            1536,
            1,
            0.0,
            1,
            3.0,
            None,
        )
        .unwrap();
        assert_eq!(auto_plan.applied_tiling(), Some((1024, 256)));

        let err = PidDecodePlan::compile_for_fields(
            "qwenimage",
            &cfg,
            &sampler,
            None,
            1536,
            1536,
            1,
            0.0,
            1,
            3.0,
            Some((2048, 256)),
        )
        .expect_err("explicit 2048px tile exceeding 3.0 GiB budget must be refused");
        assert!(
            err.to_string()
                .contains("exceeding the 3.00 GiB safe memory budget"),
            "unexpected error: {err}"
        );
    }
}
