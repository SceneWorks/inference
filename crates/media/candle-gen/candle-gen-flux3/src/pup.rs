//! Native Candle adapter for the shared FLUX 3 prompt-upsampling stage.
//!
//! Templates, reference preprocessing, annotations and generated-only sampling belong to
//! `gen_core::flux3::pup`; this module only provisions the optional native Qwen3.5 VLM.

use std::path::Path;

use candle_gen::gen_core::core_llm::{self, ModelRequirements, TextLlm};
use candle_gen::gen_core::flux3::pup::{
    self, run_text_llm, PromptUpsampler, UpsampleMode, UpsampleOptions, UpsampleOutcome,
    UpsamplePackage, UpsamplePlan, UpsamplerInput, UPSAMPLER_COMPONENT,
};
use candle_gen::gen_core::{self, CancelFlag, GenerationRequest, LoadSpec, Progress};
use candle_gen::{CandleError, Result};

/// The optional separately loaded Qwen3.5 prompt-upscaling component.
pub struct Flux3PromptUpsampler {
    package: UpsamplePackage,
    llm: Box<dyn TextLlm>,
}

impl std::fmt::Debug for Flux3PromptUpsampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Flux3PromptUpsampler")
            .field("package", &self.package)
            .field("provider", &self.llm.descriptor().id)
            .finish()
    }
}

/// No cross-request prefix cache or CUDA graphs are needed for this isolated caption stage.
/// The same spec is used by the loader and by its header-only load-memory bound.
fn llm_spec(dir: &Path) -> core_llm::LoadSpec {
    let mut spec = core_llm::LoadSpec::dense(dir.to_string_lossy().into_owned());
    spec.prefix_cache_bytes = Some(0);
    spec.cuda_graphs = Some(false);
    spec
}

impl Flux3PromptUpsampler {
    pub fn load(dir: &Path) -> Result<Self> {
        let package = UpsamplePackage::open(dir)?;
        let requirements = ModelRequirements {
            vision: true,
            video: false,
            audio: false,
            constraints: Vec::new(),
        };
        let llm = candle_llm::load_for_model_with(&llm_spec(dir), &requirements).map_err(|e| {
            CandleError::Unsupported(format!(
                "FLUX 3 {UPSAMPLER_COMPONENT}: cannot load native Qwen3.5 VLM at {}: {e}; \
                 install the pinned prompt-upsampler package or disable prompt enhancement",
                dir.display()
            ))
        })?;
        Ok(Self { package, llm })
    }

    pub fn from_spec(spec: &LoadSpec) -> Result<Option<Self>> {
        UpsamplePackage::component_dir(spec)?
            .map(|dir| Self::load(&dir))
            .transpose()
    }

    pub fn engine(&self) -> &dyn TextLlm {
        self.llm.as_ref()
    }

    /// Fence the actual VLM device/stream, not an image generator's proxy device.
    /// The slot must still remove/drop THIS owner; synchronization is not eviction.
    pub fn synchronize_before_release(&self) -> gen_core::Result<()> {
        self.llm
            .synchronize_before_release()
            .map_err(|error| match error {
                core_llm::Error::Canceled => gen_core::Error::Canceled,
                core_llm::Error::Unsupported(detail) => gen_core::Error::Unsupported(detail),
                other => gen_core::Error::Backend(Box::new(other)),
            })
    }
}

impl PromptUpsampler for Flux3PromptUpsampler {
    fn package(&self) -> &UpsamplePackage {
        &self.package
    }

    fn generate(
        &self,
        plan: &UpsamplePlan,
        options: &UpsampleOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<String> {
        run_text_llm(self.llm.as_ref(), plan, options, cancel, on_progress)
    }
}

/// Source/load bound only, not the full request peak. The native LLM separately admits its
/// request-time KV/activation allocation before generation. No measured memory is claimed here.
pub fn load_memory_bytes(dir: &Path) -> gen_core::Result<u64> {
    UpsamplePackage::open(dir)?;
    let estimate =
        candle_llm::LlamaProvider::load_memory_estimate(&llm_spec(dir), cfg!(feature = "cuda"))
            .map_err(|e| {
                gen_core::Error::Unsupported(format!("FLUX 3 {UPSAMPLER_COMPONENT}: {e}"))
            })?;
    Ok(estimate
        .device_required_bytes
        .unwrap_or(estimate.host_required_bytes))
}

pub fn text_input(req: &GenerationRequest) -> UpsamplerInput {
    UpsamplerInput {
        prompt: req.prompt.clone(),
        references: Vec::new(),
        boxes: None,
        mode: UpsampleMode::infer(0, pup::detect(&req.prompt).is_some()),
        bypass: !req.enhance_prompt,
    }
}

/// Resolve and record the effective caption before prompt-derived seeding or image encoding.
pub fn effective_prompt(
    upsampler: Option<&Flux3PromptUpsampler>,
    req: &GenerationRequest,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<UpsampleOutcome> {
    let outcome = pup::upsample(
        upsampler.map(|u| u as &dyn PromptUpsampler),
        &gen_core::flux3::references::upsampler_input(req)?,
        &UpsampleOptions {
            temperature: req.enhance_temperature,
            max_tokens: req.enhance_max_tokens,
            seed: req.seed,
        },
        &req.cancel,
        on_progress,
    )?;
    req.prompt_enhancement.emit(outcome.report.clone());
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_pup_load_cannot_retain_a_global_graph_or_prefix_parameter_cache() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-flux3/tests/fixtures/tiny-pup-snapshot");
        let spec = llm_spec(&dir);
        assert_eq!(spec.cuda_graphs, Some(false));
        assert_eq!(spec.prefix_cache_bytes, Some(0));
        // Header-only CUDA pricing does NOT open a GPU. This catches the default
        // CUDA graph policy even when the test itself runs in a CPU build.
        let estimate = candle_llm::LlamaProvider::load_memory_estimate(&spec, true).unwrap();
        assert_eq!(estimate.graph_param_cache_bytes, 0);
        assert_eq!(estimate.prefix_cache_bytes, 0);
    }

    use gen_core::{PromptEnhancementOutcome, PromptEnhancementSink};
    use std::sync::{Arc, Mutex};

    #[test]
    fn bypass_keeps_bytes_and_records_the_original_caption() {
        let reports = Arc::new(Mutex::new(Vec::new()));
        let sink = reports.clone();
        let req = GenerationRequest {
            prompt: "  original caption\n".into(),
            prompt_enhancement: PromptEnhancementSink::new(move |r| {
                candle_gen::lock_recover(&sink).push(r)
            }),
            ..Default::default()
        };
        let outcome =
            effective_prompt(None, &req, &mut |_| panic!("bypass must not infer")).unwrap();
        assert_eq!(outcome.caption, req.prompt);
        let recorded = candle_gen::lock_recover(&reports);
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].effective_prompt, req.prompt);
        assert_eq!(recorded[0].outcome, PromptEnhancementOutcome::Absent);
    }

    #[test]
    fn authored_removal_reaches_native_caption_without_upsampler_or_gpu() {
        let req = GenerationRequest {
            prompt: "remove cat".into(),
            conditioning: vec![gen_core::Conditioning::Reference {
                image: gen_core::Image { width: 64, height: 64, pixels: vec![23; 64*64*3] }, strength: None,
            }],
            flux3_layout: Some(gen_core::flux3::layouts::Layout::parse(r#"{"version":1,"boxes":[{"id":"cat","from":"ref_image_0","src_bbox":[0.1,0.2,0.6,0.7],"tgt_bbox":null}]}"#).unwrap()),
            ..Default::default()
        };
        let outcome =
            effective_prompt(None, &req, &mut |_| panic!("bypass must not infer")).unwrap();
        assert_eq!(outcome.caption, "remove cat [{\"id\": \"cat\", \"from\": \"ref_image_0\", \"src_bbox\": [100, 200, 600, 700], \"tgt_bbox\": null, \"desc\": \"\"}]");
        assert_eq!(outcome.report.original_prompt, "remove cat");
        assert!(outcome
            .report
            .locked_annotations
            .unwrap()
            .contains("\"tgt_bbox\": null"));
    }

    #[test]
    fn enabled_without_component_is_an_actionable_typed_error() {
        let req = GenerationRequest {
            enhance_prompt: true,
            ..Default::default()
        };
        let error = effective_prompt(None, &req, &mut |_| {}).unwrap_err();
        assert!(
            matches!(error, CandleError::Unsupported(ref detail) if detail.contains(UPSAMPLER_COMPONENT))
        );
    }
}
