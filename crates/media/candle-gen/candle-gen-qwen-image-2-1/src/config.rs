//! Frozen upstream configuration: the production component geometry, size presets and defaults,
//! and the `config.json` readers the loader uses so a miniature parity snapshot and the production
//! snapshot go through one code path.
//!
//! The parsers are backend-neutral and live in [`gen_core_family::qwen_image_2_1::config`]
//! (sc-27329), shared with the MLX twin; this module re-exports them so `config::*` paths stay
//! stable, and adds what is bound to this engine: reading the JSON files ([`ConfigFile`]), and the
//! vision-tower config typed in `candle-llm`.

use std::path::Path;

use candle_gen::{CandleError as Error, Result};
pub use gen_core_family::qwen_image_2_1::config::*;
use serde_json::Value;

/// Which checkpoint of the Qwen-Image 2.1 family a generator serves (sc-25901). One architecture,
/// one transformer `config.json`, one set of presets and one licence; the two differ only in how
/// the denoise schedule is obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    /// `Qwen/Qwen-Image-2.1`: the resolution-shifted schedule ([`SchedulerConfig`]), 40 steps.
    Base,
    /// `Qwen/Qwen-Image-2.1-Turbo`: the distilled checkpoint, sampled over the sigma list its own
    /// snapshot saves ([`SavedSigmaSchedule`]).
    Turbo,
}

impl Variant {
    /// The registry id this variant is served under.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Base => crate::MODEL_ID,
            Self::Turbo => crate::TURBO_MODEL_ID,
        }
    }

    /// The variant registered as `id`, if any.
    pub fn from_id(id: &str) -> Option<Self> {
        [Self::Base, Self::Turbo]
            .into_iter()
            .find(|variant| variant.id() == id)
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = candle_gen::gen_core::__model_input_reader!(
std::fs::read(path),
"crates/media/candle-gen/candle-gen-qwen-image-2-1/src/config.rs::read_json::std::fs::read@4341",
ArtifactInput,
path,
|__model_input_path| std::fs::read(__model_input_path)
)
    .map_err(|e| Error::Msg(format!("qwen_image_2_1: read {}: {e}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| Error::Msg(format!("qwen_image_2_1: parse {}: {e}", path.display())))
}

/// `from_json_file` for the shared component configs: read the file through this engine's
/// model-input reader, then parse it with the shared `from_value`.
pub trait ConfigFile: Sized {
    /// Parse an already-read `config.json` value.
    fn parse(value: &Value) -> Result<Self>;

    /// Read and parse a `config.json` file.
    fn from_json_file(path: &Path) -> Result<Self> {
        Self::parse(&read_json(path)?)
    }
}

macro_rules! config_file {
    ($($ty:ty),* $(,)?) => {$(
        impl ConfigFile for $ty {
            fn parse(value: &Value) -> Result<Self> {
                Ok(<$ty>::from_value(value)?)
            }
        }
    )*};
}

config_file!(
    TransformerConfig,
    TextEncoderConfig,
    VaeConfig,
    SchedulerConfig
);

/// The Qwen3-VL **vision** tower geometry plus the host-side processor settings the reference
/// route needs (`text_encoder/config.json` → `vision_config`, and `processor/`'s image-processor
/// block). Parsed here so the tiny parity snapshot — which deliberately ships a miniature tower
/// and a miniature pixel budget — drives exactly the production code path.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    /// The tower itself, in `candle-llm`'s shared Qwen-VL form (one architecture for Qwen3-VL and
    /// Qwen3.6, hence the `Qwen35…` spelling).
    pub tower: candle_llm::models::Qwen35VisionConfig,
    /// Host preprocessing (patch/merge/temporal geometry, pixel budget, mean/std).
    pub processor: candle_llm::image::Qwen35ImageProcessor,
}

impl VisionConfig {
    pub fn from_value(
        text_encoder_config: &Value,
        processor_config: Option<&Value>,
    ) -> Result<Self> {
        let tower = candle_llm::models::Qwen35VisionConfig::from_json(text_encoder_config)
            .map_err(|e| Error::Msg(format!("qwen_image_2_1: text_encoder/config.json: {e}")))?;
        let mut processor = candle_llm::image::Qwen35ImageProcessor {
            patch_size: tower.patch_size as usize,
            temporal_patch_size: tower.temporal_patch_size as usize,
            merge_size: tower.spatial_merge_size as usize,
            ..Default::default()
        };
        // `processor/preprocessor_config.json`, or the `image_processor` block of
        // `processor/processor_config.json`. Absent, the released geometry stands.
        if let Some(block) = processor_config.and_then(|v| v.get("image_processor").or(Some(v))) {
            // `size.shortest_edge` / `size.longest_edge` is how both the released and the
            // miniature processor serialise the budget; the bare `min_pixels` / `max_pixels`
            // spelling is the older `Qwen2VLImageProcessor` form and wins when present.
            let size = block.get("size");
            for (bare, edge, out) in [
                ("min_pixels", "shortest_edge", &mut processor.min_pixels),
                ("max_pixels", "longest_edge", &mut processor.max_pixels),
            ] {
                if let Some(x) = block
                    .get(bare)
                    .and_then(Value::as_u64)
                    .or_else(|| size.and_then(|s| s.get(edge)).and_then(Value::as_u64))
                {
                    *out = x as usize;
                }
            }
            for (key, out) in [
                ("image_mean", &mut processor.mean),
                ("image_std", &mut processor.std),
            ] {
                if let Some(values) = block.get(key).and_then(Value::as_array) {
                    for (slot, value) in out.iter_mut().zip(values) {
                        if let Some(x) = value.as_f64() {
                            *slot = x as f32;
                        }
                    }
                }
            }
        }
        if processor.min_pixels == 0 || processor.max_pixels < processor.min_pixels {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: the image processor's pixel budget is empty \
                 ([{}, {}])",
                processor.min_pixels, processor.max_pixels
            )));
        }
        Ok(Self { tower, processor })
    }

    /// `output_resolution` for this snapshot: upstream's 1024-px default, clamped into the side
    /// lengths the snapshot's own processor accepts without `smart_resize` rebinding the grid, and
    /// snapped down onto the 32-px grid.
    ///
    /// Upstream takes `output_resolution` as a pipeline argument; gen-core's request carries no
    /// field for it, so the default is derived rather than passed. On the released snapshot the
    /// budget is `[256², 4096²]` and this is exactly 1024. The miniature parity snapshot's budget
    /// is three orders of magnitude smaller, and deriving the value is what lets it exercise this
    /// path at all.
    pub fn output_resolution(&self) -> u32 {
        let multiple = SIZE_MULTIPLE as f64;
        let floor = (self.processor.min_pixels as f64).sqrt();
        let ceiling = (self.processor.max_pixels as f64).sqrt();
        let target = f64::from(OUTPUT_RESOLUTION).min(ceiling).max(floor);
        let snapped = (target / multiple).floor() * multiple;
        (snapped.max(multiple) as u32).max(SIZE_MULTIPLE)
    }
}
