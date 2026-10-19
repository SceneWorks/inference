//! The MLX half of the FLUX 3 **prompt upsampler** (sc-26238): load the separate
//! `black-forest-labs/flux-3-image-klein-pup` package (a Qwen3.5-9B VLM) through the shared
//! `mlx-llm` provider and run the tensor-neutral stage of [`gen_core::flux3::pup`] on it.
//!
//! Everything that decides *what* the model is asked — the mode, the system prompt, the
//! scaffolds, the box grammar, the reference policy, the window budget, the cancel bridge, the
//! progress events and the provenance report — lives in `gen_core`; this file owns only the
//! engine: the package is loaded with `mlx_llm::load_for_model_with` and vision required, so a
//! snapshot without the Qwen3-VL tower (or of another architecture) is refused at load, and
//! [`PromptUpsampler::generate`] is [`run_text_llm`] over that engine. The Candle twin
//! (sc-26235's crate) is the same twenty lines over `candle_llm`.
//!
//! The upsampler is an **optional co-requisite** (E5): the generator loads it only when
//! `LoadSpec::components["prompt_upsampler"]` names the package directory, and prices it
//! separately from the image model and its text encoder (its 18.8 GB of bf16 weights are a
//! second model resident beside the DiT, as upstream's vLLM share of the GPU is).
//!
//! [`run_text_llm`]: gen_core::flux3::pup::run_text_llm

use std::path::{Path, PathBuf};

use gen_core::flux3::pup::{
    self, run_text_llm, PromptUpsampler, UpsampleMode, UpsampleOptions, UpsampleOutcome,
    UpsamplePackage, UpsamplePlan, UpsamplerInput,
};
use mlx_gen::gen_core::core_llm::{self, ModelRequirements, TextLlm};
use mlx_gen::{gen_core, CancelFlag, Error, GenerationRequest, LoadSpec, Progress, Result};

pub use gen_core::flux3::pup::{
    UPSAMPLER_COMPONENT, UPSAMPLER_REPO, UPSAMPLER_REVISION, UPSAMPLER_WEIGHTS_SHA256,
};

/// The loaded upsampler: the package view plus the MLX engine serving it.
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

impl Flux3PromptUpsampler {
    /// Load the package at `dir`: the host-side view first (every required file present,
    /// `config.json` the Qwen3.5 VLM), then the engine with vision required.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with_policy(dir, false)
    }

    /// An exclusive caption phase cannot benefit from a cross-request KV cache.
    /// Keep the historical resident load's defaults unchanged.
    pub fn load_bounded(dir: &Path) -> Result<Self> {
        Self::load_with_policy(dir, true)
    }

    fn load_with_policy(dir: &Path, bounded: bool) -> Result<Self> {
        let package = UpsamplePackage::open(dir)?;
        let mut spec = core_llm::LoadSpec::dense(dir.to_string_lossy().into_owned());
        if bounded {
            spec.prefix_cache_bytes = Some(0);
        }
        let requirements = ModelRequirements {
            vision: true,
            video: false,
            audio: false,
            constraints: Vec::new(),
        };
        let llm = mlx_llm::load_for_model_with(&spec, &requirements).map_err(|e| {
            Error::Msg(format!(
                "flux3 prompt upsampler: load {} ({UPSAMPLER_REPO} @ {UPSAMPLER_REVISION}): {e}",
                dir.display()
            ))
        })?;
        Ok(Self { package, llm })
    }

    /// Load the upsampler a generator spec names, or `None` when the spec carries no
    /// [`UPSAMPLER_COMPONENT`].
    pub fn from_spec(spec: &LoadSpec) -> Result<Option<Self>> {
        match UpsamplePackage::component_dir(spec)? {
            Some(dir) => Self::load(&dir).map(Some),
            None => Ok(None),
        }
    }

    pub fn package_dir(&self) -> PathBuf {
        self.package.dir().to_path_buf()
    }

    /// The engine serving the package (its descriptor names the provider).
    pub fn engine(&self) -> &dyn TextLlm {
        self.llm.as_ref()
    }

    /// Fence the VLM's retained load/execution streams before its actual owner drops.
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

/// The per-request knobs of the stage, read off a [`GenerationRequest`] (`enhance_prompt`,
/// `enhance_temperature`, `enhance_max_tokens`, `seed`).
pub fn options_for(req: &GenerationRequest) -> UpsampleOptions {
    UpsampleOptions::from_generation_request(req)
}

/// The stage input of the distilled text-to-image route (sc-26234's request surface carries no
/// references and no boxes, so the mode is [`UpsampleMode::Text`]); the editing (sc-26236) and
/// layout (sc-26237) stories build their inputs with references / boxes and the matching mode.
pub fn text_input(req: &GenerationRequest) -> UpsamplerInput {
    UpsamplerInput {
        prompt: req.prompt.clone(),
        references: Vec::new(),
        boxes: None,
        mode: UpsampleMode::infer(0, pup::detect(&req.prompt).is_some()),
        bypass: !req.enhance_prompt,
    }
}

/// The generator's pre-generation hook: resolve the request's effective caption and emit its
/// provenance through the request's [`PromptEnhancementSink`]. Identity when `enhance_prompt`
/// is off (the caption is the prompt byte-for-byte, reported as `Absent`); a typed refusal when
/// it is on and no upsampler was loaded.
///
/// [`PromptEnhancementSink`]: gen_core::PromptEnhancementSink
pub fn effective_prompt(
    upsampler: Option<&Flux3PromptUpsampler>,
    req: &GenerationRequest,
    input: &UpsamplerInput,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<UpsampleOutcome> {
    let outcome = pup::upsample(
        upsampler.map(|u| u as &dyn PromptUpsampler),
        input,
        &options_for(req),
        &req.cancel,
        on_progress,
    )?;
    req.prompt_enhancement.emit(outcome.report.clone());
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{PromptEnhancementOutcome, PromptEnhancementReport, PromptEnhancementSink};
    use std::sync::{Arc, Mutex};

    #[test]
    fn options_and_input_read_the_request_surface() {
        let req = GenerationRequest {
            prompt: "a red bicycle".into(),
            enhance_prompt: true,
            enhance_temperature: Some(0.3),
            enhance_max_tokens: Some(64),
            seed: Some(7),
            ..Default::default()
        };
        assert_eq!(
            options_for(&req),
            UpsampleOptions {
                temperature: Some(0.3),
                max_tokens: Some(64),
                seed: Some(7),
            }
        );
        let input = text_input(&req);
        assert_eq!(input.mode, UpsampleMode::Text);
        assert!(!input.bypass && input.references.is_empty() && input.boxes.is_none());
        assert!(text_input(&GenerationRequest::default()).bypass);
    }

    #[test]
    fn bypass_without_an_upsampler_is_the_identity_and_reports_absent() {
        let seen: Arc<Mutex<Vec<PromptEnhancementReport>>> = Arc::default();
        let sink = seen.clone();
        let req = GenerationRequest {
            prompt: "as typed \n".into(),
            prompt_enhancement: PromptEnhancementSink::new(move |r| sink.lock().unwrap().push(r)),
            ..Default::default()
        };
        let outcome = effective_prompt(None, &req, &text_input(&req), &mut |_| {
            panic!("no progress on bypass")
        })
        .unwrap();
        assert_eq!(outcome.caption, "as typed \n");
        let reports = seen.lock().unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].outcome, PromptEnhancementOutcome::Absent);
        assert_eq!(reports[0].effective_prompt, "as typed \n");
        assert_eq!(reports[0].mode, None);
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
        let input = gen_core::flux3::references::upsampler_input(&req).unwrap();
        let outcome =
            effective_prompt(None, &req, &input, &mut |_| panic!("bypass must not infer")).unwrap();
        assert_eq!(outcome.caption, "remove cat [{\"id\": \"cat\", \"from\": \"ref_image_0\", \"src_bbox\": [100, 200, 600, 700], \"tgt_bbox\": null, \"desc\": \"\"}]");
        assert_eq!(outcome.report.original_prompt, "remove cat");
        assert!(outcome
            .report
            .locked_annotations
            .unwrap()
            .contains("\"tgt_bbox\": null"));
    }

    #[test]
    fn near_limit_native_layout_caption_reaches_final_parameters_once() {
        let config = gen_core::flux3::Flux3Config::from_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot"),
        )
        .unwrap();
        let layout=gen_core::flux3::layouts::Layout::from_json(serde_json::json!({"version":1,"boxes":[{"id":"large","bbox":[0.1,0.2,0.7,0.8],"desc":"x".repeat(9000)}]})).unwrap();
        let req = GenerationRequest {
            prompt: "scene".into(),
            flux3_layout: Some(layout.clone()),
            ..Default::default()
        };
        gen_core::flux3::validate_generation_request(
            gen_core::flux3::DISTILLED_MODEL_ID,
            &req,
            &config,
        )
        .unwrap();
        let input = gen_core::flux3::references::upsampler_input(&req).unwrap();
        let bypass = effective_prompt(None, &req, &input, &mut |_| panic!("bypass")).unwrap();
        for caption in [
            bypass.caption.clone(),
            format!("{} {}", "enhanced ".repeat(300), bypass.caption),
        ] {
            let params =
                gen_core::flux3::GenerationParams::resolve_caption(&req, caption.clone(), &config)
                    .unwrap();
            assert_eq!(params.prompt, caption);
            assert_eq!(params.seed, gen_core::flux3::prompt_seed(&caption));
            assert_eq!(req.flux3_layout.as_ref(), Some(&layout));
        }
    }

    #[test]
    fn enhance_without_an_upsampler_is_a_typed_refusal() {
        let req = GenerationRequest {
            prompt: "p".into(),
            enhance_prompt: true,
            ..Default::default()
        };
        let err = effective_prompt(None, &req, &text_input(&req), &mut |_| {}).unwrap_err();
        assert!(
            matches!(&err, Error::Unsupported(m) if m.contains(UPSAMPLER_COMPONENT)),
            "{err}"
        );
    }

    #[test]
    fn a_missing_package_is_a_typed_load_error_naming_the_component() {
        let dir = tempfile::tempdir().unwrap();
        let err = Flux3PromptUpsampler::load(dir.path()).unwrap_err();
        assert!(err.to_string().contains("config.json"), "{err}");
        let package = gen_core::WeightsSource::Dir(dir.path().join("package"));
        let mut spec = LoadSpec::new(package.clone());
        assert!(Flux3PromptUpsampler::from_spec(&spec).unwrap().is_none());
        spec.components.insert(
            UPSAMPLER_COMPONENT.to_string(),
            gen_core::WeightsSource::Dir(dir.path().to_path_buf()),
        );
        assert!(Flux3PromptUpsampler::from_spec(&spec).is_err());
        spec.components.insert(
            UPSAMPLER_COMPONENT.to_string(),
            gen_core::WeightsSource::File(dir.path().join("model.safetensors")),
        );
        let err = Flux3PromptUpsampler::from_spec(&spec).unwrap_err();
        assert!(err.to_string().contains("package directory"), "{err}");
    }
}
