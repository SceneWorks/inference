//! Shared native StarVector `RouteRecord` declarations and [`gen_core::Generator`] driver
//! (`SW-6982`, `PIPE-12`, `REG-6`, `XL-4`).
//!
//! Each platform bundle registers `starvector_1b` and (on MLX/CUDA) `starvector_8b` as real
//! [`gen_core::Modality::Vector`] generator routes through [`gen_core::ProviderRegistryBuilder::register_route`],
//! backed by the existing [`core_llm::TextLlmRegistry`] StarVector providers (`mlx-starvector-1b`,
//! `mlx-starvector-8b`, `candle-starvector-1b`, `candle-starvector-8b`). Existing `TextLlmRegistry`
//! IDs remain untouched for `core-llm` / ChatWorks compatibility (`XL-4`).

use std::time::Duration;

use core_llm::{
    CancelFlag as LlmCancelFlag, Content, ImageRef, LoadSpec as LlmLoadSpec, Message, Quantize,
    Role, Sampling, StarVectorFinishReason, StarVectorProvider, StarVectorRequest,
    StarVectorStreamEvent, TextLlm, TextLlmRegistry, TextLlmRequest,
};
use gen_core::vector::{
    build_vector_output, VectorSanitizeOptions, DEFAULT_PREVIEW_DIMENSION, MAX_SVG_BYTES,
};
use gen_core::{
    Conditioning, ConditioningKind, FamilyExtensionKind, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, Progress, Quant, RouteRecord, WeightsSource,
};

/// Canonical registered generator route ID for StarVector-1B (`image_to_svg`).
pub const STARVECTOR_1B_ID: &str = "starvector_1b";
/// Canonical registered generator route ID for StarVector-8B (`image_to_svg`).
pub const STARVECTOR_8B_ID: &str = "starvector_8b";
/// Shared model family identifier for StarVector vector routes.
pub const STARVECTOR_FAMILY: &str = "starvector";

/// Backing `TextLlmRegistry` provider IDs preserved for `XL-4`.
pub const MLX_STARVECTOR_1B_LLM_ID: &str = "mlx-starvector-1b";
pub const MLX_STARVECTOR_8B_LLM_ID: &str = "mlx-starvector-8b";
pub const CANDLE_STARVECTOR_1B_LLM_ID: &str = "candle-starvector-1b";
pub const CANDLE_STARVECTOR_8B_LLM_ID: &str = "candle-starvector-8b";

const DEFAULT_MAX_WALL_TIME: Duration = Duration::from_secs(600);

struct StarVectorRouteSpec {
    route_id: &'static str,
    backend: &'static str,
    mac_only: bool,
    architecture: &'static str,
    vision_component: &'static str,
    decoder_component: &'static str,
    supported_quants: &'static [Quant],
}

fn build_starvector_route(
    spec: StarVectorRouteSpec,
    bindings: gen_core::RouteBindings,
) -> RouteRecord {
    let StarVectorRouteSpec {
        route_id,
        backend,
        mac_only,
        architecture,
        vision_component,
        decoder_component,
        supported_quants,
    } = spec;
    RouteRecord {
        route_id,
        family: STARVECTOR_FAMILY,
        backend,
        modality: Modality::Vector,
        mac_only,
        structure: gen_core::RouteStructure {
            conditioning: gen_core::ConditioningSurface {
                kinds: vec![ConditioningKind::Reference],
                supports_negative_prompt: false,
                pass0_requirement: gen_core::Pass0ConditioningRequirement::RequiresReference,
                modules: vec![vision_component],
                ..Default::default()
            },
            denoiser: gen_core::ModelBinding::non_diffusion(architecture),
            adapters: gen_core::AdapterSurface::unsupported(),
            sampling: gen_core::SamplingSurface {
                supports_guidance: false,
                ..Default::default()
            },
            residency: gen_core::ResidencySurface {
                lifetime_graph: vec![
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::ConditioningEncode,
                        [vision_component],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        [decoder_component],
                        false,
                    ),
                ],
                supports_kv_cache: true,
                ..Default::default()
            },
            weight_plan: gen_core::WeightPlanDeclaration {
                source_kind: gen_core::WeightSourceKind::SnapshotDirectory,
                supported_quants,
                ..Default::default()
            },
            supported_family_extensions: vec![FamilyExtensionKind::Vector],
            ..Default::default()
        },
        limits: gen_core::ProductLimits {
            min_size: 0,
            max_size: 0,
            ..Default::default()
        },
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::DeterministicSynthesis,
            max_count: 1,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults::default(),
        bindings,
    }
}

/// Static descriptor for MLX `starvector_1b`.
pub fn descriptor_mlx_1b() -> ModelDescriptor {
    route_mlx_1b(unavailable_loader).compute_descriptor()
}

/// Static descriptor for MLX `starvector_8b`.
pub fn descriptor_mlx_8b() -> ModelDescriptor {
    route_mlx_8b(unavailable_loader).compute_descriptor()
}

/// Static descriptor for Candle `starvector_1b`.
pub fn descriptor_candle_1b() -> ModelDescriptor {
    route_candle_1b(unavailable_loader).compute_descriptor()
}

/// Static descriptor for Candle `starvector_8b`.
pub fn descriptor_candle_8b() -> ModelDescriptor {
    route_candle_8b(unavailable_loader).compute_descriptor()
}

/// Build the primary [`RouteRecord`] for MLX `starvector_1b`.
pub fn route_mlx_1b(load: fn(&LoadSpec) -> gen_core::Result<Box<dyn Generator>>) -> RouteRecord {
    build_starvector_route(
        StarVectorRouteSpec {
            route_id: STARVECTOR_1B_ID,
            backend: "mlx",
            mac_only: true,
            architecture: "starvector_1b_clip_gptbigcode",
            vision_component: "starvector_clip_vision",
            decoder_component: "starvector_gptbigcode_decoder",
            supported_quants: &[Quant::Q4, Quant::Q8],
        },
        gen_core::RouteBindings::generator(descriptor_mlx_1b, load),
    )
}

/// Build the primary [`RouteRecord`] for MLX `starvector_8b`.
pub fn route_mlx_8b(load: fn(&LoadSpec) -> gen_core::Result<Box<dyn Generator>>) -> RouteRecord {
    build_starvector_route(
        StarVectorRouteSpec {
            route_id: STARVECTOR_8B_ID,
            backend: "mlx",
            mac_only: true,
            architecture: "starvector_8b_siglip_starcoder2",
            vision_component: "starvector_siglip_vision",
            decoder_component: "starvector_starcoder2_decoder",
            supported_quants: &[Quant::Q4, Quant::Q8],
        },
        gen_core::RouteBindings::generator(descriptor_mlx_8b, load),
    )
}

/// Build the primary [`RouteRecord`] for Candle `starvector_1b`.
pub fn route_candle_1b(load: fn(&LoadSpec) -> gen_core::Result<Box<dyn Generator>>) -> RouteRecord {
    build_starvector_route(
        StarVectorRouteSpec {
            route_id: STARVECTOR_1B_ID,
            backend: "candle",
            mac_only: false,
            architecture: "starvector_1b_clip_gptbigcode",
            vision_component: "starvector_clip_vision",
            decoder_component: "starvector_gptbigcode_decoder",
            supported_quants: &[Quant::Q4, Quant::Q8],
        },
        gen_core::RouteBindings::generator(descriptor_candle_1b, load),
    )
}

/// Build the primary [`RouteRecord`] for Candle `starvector_8b`.
pub fn route_candle_8b(load: fn(&LoadSpec) -> gen_core::Result<Box<dyn Generator>>) -> RouteRecord {
    build_starvector_route(
        StarVectorRouteSpec {
            route_id: STARVECTOR_8B_ID,
            backend: "candle",
            mac_only: false,
            architecture: "starvector_8b_siglip_starcoder2",
            vision_component: "starvector_siglip_vision",
            decoder_component: "starvector_starcoder2_decoder",
            supported_quants: &[Quant::Q4, Quant::Q8],
        },
        gen_core::RouteBindings::generator(descriptor_candle_8b, load),
    )
}

/// Descriptor-only unavailable loader used by hosted capability-export registries (e.g.
/// `runtime-cpu::cuda_descriptor_catalog()`).
pub fn unavailable_loader(_spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Err(gen_core::Error::Unsupported(
        "StarVector descriptor-only route cannot be loaded without its native runtime backend"
            .into(),
    ))
}

/// Load a [`StarVectorGenerator`] backed by `llm_provider_id` in `text_registry`.
pub fn load_from_text_registry(
    descriptor: ModelDescriptor,
    text_registry: &TextLlmRegistry,
    llm_provider_id: &'static str,
    spec: &LoadSpec,
) -> gen_core::Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    gen_core::reject_unknown_components(spec, &[], descriptor.id)?;
    if !spec.adapters.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.identity.is_some()
        || spec.pid.is_some()
        || spec.ip_adapter.is_some()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{} does not accept LoRA, control, identity, PiD, or IP-Adapter overlays",
            descriptor.id
        )));
    }
    let source = match &spec.weights {
        WeightsSource::Dir(path) | WeightsSource::File(path) => {
            path.to_string_lossy().into_owned()
        }
    };
    let quantize = match spec.quantize {
        None => None,
        Some(Quant::Q4) => Some(Quantize::Q4),
        Some(Quant::Q8) => Some(Quantize::Q8),
        Some(Quant::Nvfp4) => Some(Quantize::Nvfp4),
        Some(other) => {
            return Err(gen_core::Error::Unsupported(format!(
                "{} does not support quantization tier {other:?}",
                descriptor.id
            )));
        }
    };
    let mut llm_spec = LlmLoadSpec::dense(source);
    llm_spec.quantize = quantize;
    let provider = text_registry
        .load_textllm(llm_provider_id, &llm_spec)
        .map_err(map_llm_error)?;
    let generator = StarVectorGenerator::new(descriptor, provider)?;
    Ok(Box::new(generator))
}

/// [`gen_core::Generator`] adapter driving a native [`core_llm::StarVectorProvider`] and returning
/// typed [`GenerationOutput::Vector`] (`PIPE-12`, `E4`).
pub struct StarVectorGenerator {
    descriptor: ModelDescriptor,
    provider: Box<dyn TextLlm>,
}

impl StarVectorGenerator {
    /// Wrap a loaded [`core_llm::TextLlm`] that exposes [`TextLlm::as_starvector_provider`].
    pub fn new(descriptor: ModelDescriptor, provider: Box<dyn TextLlm>) -> gen_core::Result<Self> {
        if provider.as_starvector_provider().is_none() {
            return Err(gen_core::Error::Msg(format!(
                "LLM provider `{}` does not implement StarVectorProvider for route `{}`",
                provider.descriptor().id,
                descriptor.id
            )));
        }
        Ok(Self {
            descriptor,
            provider,
        })
    }

    fn starvector(&self) -> &dyn StarVectorProvider {
        self.provider
            .as_starvector_provider()
            .expect("StarVectorGenerator validated StarVectorProvider at construction")
    }

    fn build_svg_request(&self, request: &GenerationRequest) -> gen_core::Result<StarVectorRequest> {
        if !request.prompt.trim().is_empty() {
            return Err(gen_core::Error::Unsupported(format!(
                "{} im2svg is image-conditioned and does not accept free-form text prompts",
                self.descriptor.id
            )));
        }
        let image = match request.conditioning.as_slice() {
            [Conditioning::Reference { image, .. }] => image,
            [] => {
                return Err(gen_core::Error::Msg(format!(
                    "{} requires exactly one reference conditioning image",
                    self.descriptor.id
                )));
            }
            [_] => {
                return Err(gen_core::Error::Unsupported(format!(
                    "{} requires Conditioning::Reference image conditioning",
                    self.descriptor.id
                )));
            }
            _ => {
                return Err(gen_core::Error::Unsupported(format!(
                    "{} accepts exactly one reference conditioning image",
                    self.descriptor.id
                )));
            }
        };
        let image_ref = ImageRef::new(image.width, image.height, image.pixels.clone())
            .map_err(gen_core::Error::Msg)?;

        let sv = self.starvector();
        let text_desc = sv.descriptor();
        let sv_desc = sv.starvector_descriptor();
        let ext = request.extensions.vector.as_ref();

        let max_new_tokens = ext
            .and_then(|v| v.max_new_tokens)
            .unwrap_or(text_desc.capabilities.max_new_tokens);
        if max_new_tokens == 0 {
            return Err(gen_core::Error::Msg(format!(
                "{}: vector.max_new_tokens must be non-zero",
                self.descriptor.id
            )));
        }

        let temperature = ext.and_then(|v| v.temperature).unwrap_or(0.0);
        let top_p = ext.and_then(|v| v.top_p).unwrap_or(1.0);
        let top_k = ext.and_then(|v| v.top_k).unwrap_or(0) as usize;
        let repetition_penalty = ext.and_then(|v| v.repetition_penalty).unwrap_or(1.0);
        let repetition_context = ext
            .and_then(|v| v.repetition_context)
            .map(|c| c as usize)
            .unwrap_or_else(|| {
                if (repetition_penalty - 1.0).abs() > f32::EPSILON {
                    64
                } else {
                    0
                }
            });
        let sampling = Sampling {
            temperature,
            top_p,
            top_k,
            presence_penalty: 0.0,
            presence_penalty_generated_only: false,
            repetition_penalty,
            repetition_context,
        };

        let mut text_request = TextLlmRequest::new(
            vec![Message {
                role: Role::User,
                content: vec![Content::Image(image_ref)],
                thinking: None,
                tool_calls: Vec::new(),
            }],
            max_new_tokens,
        );
        text_request.sampling = sampling;
        text_request.seed = request.seed;
        text_request.cancel = LlmCancelFlag::from_arc(request.cancel.as_arc().clone());

        let default_max_svg_bytes = if sv_desc.max_svg_bytes > 0 {
            sv_desc.max_svg_bytes
        } else {
            MAX_SVG_BYTES
        };
        let max_svg_bytes = ext
            .and_then(|v| v.max_svg_bytes)
            .unwrap_or(default_max_svg_bytes);
        let max_wall_time = ext
            .and_then(|v| v.max_wall_time_ms)
            .map(Duration::from_millis)
            .or(sv_desc.max_wall_time)
            .unwrap_or(DEFAULT_MAX_WALL_TIME);

        Ok(StarVectorRequest::new(
            text_request,
            max_svg_bytes,
            max_wall_time,
        ))
    }
}

impl Generator for StarVectorGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request_vector(self.descriptor.id, request)?;
        let svg_request = self.build_svg_request(request)?;
        self.starvector()
            .validate_svg(&svg_request)
            .map_err(map_llm_error)
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        if request.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        self.descriptor
            .capabilities
            .validate_request_vector(self.descriptor.id, request)?;
        let svg_request = self.build_svg_request(request)?;
        let sv = self.starvector();
        sv.validate_svg(&svg_request).map_err(map_llm_error)?;

        let total_tokens = svg_request.text_request.max_new_tokens;
        let completed = sv
            .generate_svg_with_execution_report(&svg_request, &mut |event| match event {
                StarVectorStreamEvent::Progress { generated_tokens } => {
                    on_progress(Progress::Step {
                        current: generated_tokens.min(total_tokens),
                        total: total_tokens,
                    });
                }
                StarVectorStreamEvent::Source { .. } | StarVectorStreamEvent::Done { .. } => {}
            })
            .map_err(map_llm_error)?;

        if completed.output.finish_reason == StarVectorFinishReason::Cancelled
            || request.cancel.is_cancelled()
        {
            return Err(gen_core::Error::Canceled);
        }

        on_progress(Progress::Decoding);

        let ext = request.extensions.vector.as_ref();
        let preview_max_dimension = ext
            .and_then(|v| v.preview_max_dimension)
            .unwrap_or(DEFAULT_PREVIEW_DIMENSION);
        let render_preview = ext.is_none_or(|v| v.render_preview);
        let allow_non_publishable = ext.is_some_and(|v| v.allow_non_publishable);
        let sanitize_options = VectorSanitizeOptions {
            max_bytes: svg_request.max_svg_bytes,
            render_preview,
            preview_max_dimension,
        };
        let prompt_tokens = Some(sv.starvector_descriptor().projection.image_token_count as usize);
        let vector_output = build_vector_output(
            completed.output.svg.as_deref(),
            completed.output.finish_reason,
            completed.output.generated_tokens,
            prompt_tokens,
            completed.output.generated_bytes,
            sv.descriptor().id.as_str(),
            sanitize_options,
            allow_non_publishable,
        )?;
        Ok(GenerationOutput::Vector(vector_output))
    }
}

fn map_llm_error(err: core_llm::Error) -> gen_core::Error {
    match err {
        core_llm::Error::Canceled => gen_core::Error::Canceled,
        core_llm::Error::Unsupported(msg) => gen_core::Error::Unsupported(msg),
        core_llm::Error::InvalidRequest(msg) => gen_core::Error::Msg(msg),
        other => gen_core::Error::backend(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_llm::{
        starvector::starvector_8b, StarVectorOutput, StreamEvent, TextLlmDescriptor, TextLlmOutput,
        TextLlmRequest,
    };
    use gen_core::{Image, VectorExtension, VectorFinishReason};

    struct MockStarVectorProvider {
        text_desc: TextLlmDescriptor,
        sv_desc: core_llm::StarVectorDescriptor,
        output: StarVectorOutput,
    }

    impl TextLlm for MockStarVectorProvider {
        fn descriptor(&self) -> &TextLlmDescriptor {
            &self.text_desc
        }

        fn as_starvector_provider(&self) -> Option<&dyn StarVectorProvider> {
            Some(self)
        }

        fn validate(&self, request: &TextLlmRequest) -> core_llm::Result<()> {
            core_llm::starvector_conditioning_image("MockStarVector", request).map(|_| ())
        }

        fn generate(
            &self,
            _request: &TextLlmRequest,
            _on_event: &mut dyn FnMut(StreamEvent),
        ) -> core_llm::Result<TextLlmOutput> {
            Err(core_llm::Error::Unsupported(
                "mock uses StarVectorProvider::generate_svg".into(),
            ))
        }
    }

    impl StarVectorProvider for MockStarVectorProvider {
        fn starvector_descriptor(&self) -> &core_llm::StarVectorDescriptor {
            &self.sv_desc
        }

        fn generate_svg(
            &self,
            _request: &StarVectorRequest,
            on_event: &mut dyn FnMut(StarVectorStreamEvent),
        ) -> core_llm::Result<StarVectorOutput> {
            on_event(StarVectorStreamEvent::Progress {
                generated_tokens: self.output.generated_tokens,
            });
            on_event(StarVectorStreamEvent::Done {
                finish_reason: self.output.finish_reason,
                generated_tokens: self.output.generated_tokens,
                generated_bytes: self.output.generated_bytes,
            });
            Ok(self.output.clone())
        }
    }

    fn mock_request() -> GenerationRequest {
        let mut req = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 8,
                    height: 8,
                    pixels: vec![128; 8 * 8 * 3],
                },
                strength: None,
            }],
            ..Default::default()
        };
        req.extensions.vector = Some(VectorExtension {
            max_new_tokens: Some(128),
            render_preview: true,
            preview_max_dimension: Some(64),
            ..Default::default()
        });
        req
    }

    #[test]
    fn starvector_route_records_pass_tier0_conformance() {
        for route in [
            route_mlx_1b(unavailable_loader),
            route_mlx_8b(unavailable_loader),
            route_candle_1b(unavailable_loader),
            route_candle_8b(unavailable_loader),
        ] {
            gen_core::verify_route_conformance(&route)
                .unwrap_or_else(|e| panic!("{} failed conformance: {e}", route.route_id));
            let desc = route.compute_descriptor();
            assert_eq!(desc.modality, Modality::Vector);
            assert_eq!(desc.family, STARVECTOR_FAMILY);
            assert_eq!(
                gen_core::export::ExportedRouteV2::from_route_record(&route, None, false).modes,
                vec!["image_to_svg".to_string()]
            );
        }
    }

    #[test]
    fn starvector_generator_sanitizes_svg_renders_preview_and_populates_terminal_metadata() {
        let raw_svg = r##"<svg width="40" height="40" onload="alert(1)"><script>boom()</script><rect x="4" y="4" width="32" height="32" fill="#0080ff"/></svg>"##;
        let provider = MockStarVectorProvider {
            text_desc: starvector_8b::text_descriptor(MLX_STARVECTOR_8B_LLM_ID, "mlx"),
            sv_desc: starvector_8b::starvector_descriptor(),
            output: StarVectorOutput {
                svg: Some(raw_svg.to_string()),
                generated_tokens: 37,
                generated_bytes: raw_svg.len(),
                finish_reason: StarVectorFinishReason::CompleteRoot,
                kv_cache: None,
            },
        };
        let gen = StarVectorGenerator::new(descriptor_mlx_8b(), Box::new(provider))
            .expect("wrap mock StarVector provider");

        let mut saw_step = false;
        let mut saw_decoding = false;
        let output = gen
            .generate(&mock_request(), &mut |progress| match progress {
                Progress::Step { current, total } => {
                    saw_step = current == 37 && total == 128;
                }
                Progress::Decoding => saw_decoding = true,
                _ => {}
            })
            .expect("StarVector generation succeeds");

        assert!(saw_step, "expected token progress step");
        assert!(saw_decoding, "expected Decoding progress before sanitize/preview");

        let GenerationOutput::Vector(vec_out) = output else {
            panic!("expected GenerationOutput::Vector");
        };
        assert!(!vec_out.svg.contains("<script"));
        assert!(!vec_out.svg.contains("onload"));
        assert!(vec_out.svg.contains("<rect"));
        assert_eq!(vec_out.finish_reason, VectorFinishReason::Stop);
        assert!(!vec_out.truncated);
        assert_eq!(vec_out.generated_tokens, 37);
        assert_eq!(vec_out.prompt_tokens, Some(576));
        let preview = vec_out.preview.expect("derived preview must be present");
        assert_eq!((preview.width, preview.height), (40, 40));
        assert_eq!(
            vec_out.terminal.provider_id.as_deref(),
            Some(MLX_STARVECTOR_8B_LLM_ID)
        );
        assert_eq!(
            vec_out.terminal.provider_finish_reason.as_deref(),
            Some("complete_root")
        );
        assert_eq!(vec_out.terminal.generated_bytes, Some(raw_svg.len()));
        assert_eq!(vec_out.terminal.viewport, Some([40, 40]));
        assert!(vec_out.terminal.svg_sha256.is_some());
    }

    #[test]
    fn starvector_generator_surfaces_token_limit_and_refuses_text_prompts() {
        let provider = MockStarVectorProvider {
            text_desc: starvector_8b::text_descriptor(CANDLE_STARVECTOR_8B_LLM_ID, "candle"),
            sv_desc: starvector_8b::starvector_descriptor(),
            output: StarVectorOutput {
                svg: None,
                generated_tokens: 128,
                generated_bytes: 512,
                finish_reason: StarVectorFinishReason::TokenLimit,
                kv_cache: None,
            },
        };
        let gen = StarVectorGenerator::new(descriptor_candle_8b(), Box::new(provider))
            .expect("wrap mock StarVector provider");

        let mut bad_req = mock_request();
        bad_req.prompt = "not allowed".into();
        assert!(gen.validate(&bad_req).is_err());

        let mut limit_req = mock_request();
        limit_req.extensions.vector.as_mut().unwrap().allow_non_publishable = true;
        let GenerationOutput::Vector(vec_out) = gen
            .generate(&limit_req, &mut |_| {})
            .expect("token limit returns typed VectorOutput when allow_non_publishable=true")
        else {
            panic!("expected GenerationOutput::Vector");
        };
        assert_eq!(vec_out.finish_reason, VectorFinishReason::Length);
        assert!(vec_out.truncated);
        assert!(vec_out.svg.is_empty());
        assert!(vec_out.preview.is_none());
        assert_eq!(
            vec_out.terminal.provider_finish_reason.as_deref(),
            Some("token_limit")
        );
    }
}
