//! The single executable Supra2 request policy shared by native adapters and WASM validation.
//! Tokenization, checkpoint loading, sampling math and model execution remain in their owners.
use serde::{Deserialize, Serialize};

pub const PROVIDER_ID: &str = "supra2_img";
pub const FAMILY: &str = "supra2";
pub const IMAGE_SIZE: u32 = 256;
pub const DEFAULT_STEPS: u32 = 50;
pub const DEFAULT_GUIDANCE: f32 = 3.0;
pub const DEFAULT_SEED: u64 = 0;
pub const MAX_IMAGES: u32 = 64;
pub const SAMPLER: &str = "euler";
pub const SCHEDULER: &str = "supra2_uniform_flow";

/// Only the fields inspected by Supra2 policy; native adapters project media presence without
/// copying media buffers. Missing serialized fields use the published native model defaults.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Supra2PolicyRequest {
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub seed: Option<u64>,
    pub negative_prompt: Option<String>,
    pub prompt_batch_present: bool,
    pub conditioning_present: bool,
}
impl Default for Supra2PolicyRequest {
    fn default() -> Self {
        Self {
            width: IMAGE_SIZE,
            height: IMAGE_SIZE,
            count: 1,
            steps: None,
            guidance: None,
            seed: None,
            negative_prompt: None,
            prompt_batch_present: false,
            conditioning_present: false,
        }
    }
}

/// Exact payload of the native `Error::Unsupported` variant, without a runtime dependency.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Supra2PolicyError(pub String);
impl std::fmt::Display for Supra2PolicyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for Supra2PolicyError {}

/// A validated request with the published defaults applied.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Supra2Resolved {
    pub steps: u32,
    pub guidance: f32,
    /// `guidance > 1.0` — upstream's `use_cfg`.
    pub use_cfg: bool,
    /// Seed of image 0; image `i` uses the native `image_seed(seed, i)` helper.
    pub seed: u64,
    pub count: u32,
}

impl Supra2Resolved {
    /// Apply defaults and refuse what the published interface does not have. Call after the
    /// descriptor's shared `Capabilities::validate_request` floor.
    pub fn resolve(req: &Supra2PolicyRequest) -> Result<Self, Supra2PolicyError> {
        if req.width != IMAGE_SIZE || req.height != IMAGE_SIZE {
            return Err(Supra2PolicyError(format!(
                "supra2: native resolution is {IMAGE_SIZE}x{IMAGE_SIZE} only (got {}x{})",
                req.width, req.height
            )));
        }
        if req
            .negative_prompt
            .as_deref()
            .is_some_and(|n| !n.is_empty())
        {
            return Err(Supra2PolicyError(
                "supra2: negative prompts are not part of the published interface (CFG uses the checkpoint's unconditional embedding)".into(),
            ));
        }
        if req.prompt_batch_present || req.conditioning_present {
            return Err(Supra2PolicyError(
                "supra2: one text prompt per request; prompt batches and image conditioning are unsupported".into(),
            ));
        }
        let steps = req.steps.unwrap_or(DEFAULT_STEPS);
        if steps == 0 {
            return Err(Supra2PolicyError("supra2: steps must be at least 1".into()));
        }
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        if !guidance.is_finite() {
            return Err(Supra2PolicyError("supra2: guidance must be finite".into()));
        }
        let count = req.count.max(1);
        if count > MAX_IMAGES {
            return Err(Supra2PolicyError(format!(
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
