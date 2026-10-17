//! `Flux3Klein` — the distilled FLUX 3 Image Klein text-to-image [`Generator`] on Candle:
//! descriptor, load, request validation and the staged generate (text encoder → DiT +
//! autoencoder), registered as [`MODEL_ID`]. The request/load surface is the MLX twin's, field for
//! field (both read `gen_core::flux3`).

use std::path::Path;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::flux3::{
    prompt_seed, validate_generation_request, Flux3Config, GenerationParams, GenerationResources,
    DISTILLED_MODEL_ID, FAMILY, SAMPLERS, TEXT_ENCODER_COMPONENT,
};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationReport, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, Precision, Progress, SizeFloor,
};
use candle_gen::residency::Residency;
use candle_gen::{CandleError as Error, Result};

use crate::autoencoder::Flux3AutoEncoder;
use crate::dit::Flux3Dit;
use crate::pipeline::{decode_tokens, denoise_one};
use crate::text_encoder::{Flux3TextEncoder, TextConditioning};

/// Registry id (the SceneWorks worker's `payload.model`) — the same id as the MLX twin, bound to
/// the `candle` backend by the descriptor.
pub const MODEL_ID: &str = DISTILLED_MODEL_ID;
/// Output sides must be multiples of the token size.
pub const SIZE_MULTIPLE: u32 = gen_core::flux3::DOWNSAMPLING_RATIO;
/// Smallest side (`pipeline.MIN_SIDE`).
pub const MIN_SIZE: u32 = gen_core::flux3::MIN_SIDE;
/// Largest side (`pipeline.MAX_SIDE`).
pub const MAX_SIZE: u32 = gen_core::flux3::MAX_SIDE;
/// Images per request (rendered one after another, each from `seed + i`).
pub const MAX_COUNT: u32 = 8;

/// Identity + capabilities, constructible without weights.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        // The packed 16× latent space of FLUX 3's own autoencoder (BatchNorm-stat tokens): no
        // shared decoder consumes it, so nothing is advertised and every decoder swap fails closed.
        denoiser_output_latent_space: None,
        control_kinds: None,
        // Both components default to the package's bundled copies; neither is required.
        required_components: &[],
        id: MODEL_ID,
        family: FAMILY,
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // A flow map: no CFG, so no negative prompt and no guidance scale (upstream refuses
            // a guidance on the distilled checkpoint).
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // The one integrator, the flow-map Euler step; the schedule has no name and its
            // alpha is fixed, so no scheduler is advertised.
            samplers: SAMPLERS.to_vec(),
            schedulers: Vec::new(),
            // LoRA loading is sc-26239 (with the base checkpoint).
            supports_lora: false,
            supports_lokr: false,
            // No per-step preview: the packed latent has no fitted RGB projection and the
            // autoencoder decode is the only exact view of a step.
            supports_preview: false,
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            max_count: MAX_COUNT,
            mac_only: false,
            // Quantized variants are sc-26241.
            supported_quants: &[],
            supports_sequential_offload: true,
            // The optional native VLM resolves the effective caption before encoding.
            supports_prompt_enhancement: true,
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SIZE_MULTIPLE,
            },
            ..Default::default()
        },
    }
}

/// The heavy half: the DiT and the autoencoder.
pub struct Heavy {
    pub dit: Flux3Dit,
    pub ae: Flux3AutoEncoder,
}

/// A loaded distilled generator.
pub struct Flux3Klein {
    descriptor: ModelDescriptor,
    config: Flux3Config,
    device: Device,
    residency: Residency<Flux3TextEncoder, Heavy>,
    prompt_upsampler: Option<crate::pup::Flux3PromptUpsampler>,
    /// The spec this generator was loaded from: the tier and residency policy the shared
    /// memory-strategy seam checks a request selection against.
    spec: LoadSpec,
    /// The loaded memory contract, published through [`Generator::memory_strategy_contract`].
    memory_strategy: gen_core::MemoryProviderContract,
}

/// Whether one request runs staged: the request's admitted `GenerationMemory::stage_residency`
/// (written by the memory-strategy request scope) OR a `Sequential` load policy.
pub(crate) fn stage_residency_for(spec: &LoadSpec, req: &GenerationRequest) -> bool {
    req.memory.is_some_and(|memory| memory.stage_residency)
        || spec.offload_policy == gen_core::OffloadPolicy::Sequential
}

/// The DiT compute dtype a load spec selects: the release's bf16 autocast by default, the
/// oracle's FP32 path on an explicit `Precision::Fp32`. The autoencoder is f32 either way.
pub fn compute_dtype(spec: &LoadSpec) -> DType {
    match spec.precision {
        Precision::Bf16 => DType::BF16,
        Precision::Fp32 => DType::F32,
    }
}

/// Load the DiT from the package's `model.safetensors` onto `device` at `compute`.
pub fn load_dit(
    weights: &Path,
    config: &Flux3Config,
    compute: DType,
    device: &Device,
) -> Result<Flux3Dit> {
    Flux3Dit::load(weights, &config.dit, compute, device)
}

/// Load the autoencoder (f32) onto `device`.
pub fn load_autoencoder(path: &Path, device: &Device) -> Result<Flux3AutoEncoder> {
    Flux3AutoEncoder::load(path, device)
}

pub(crate) fn refuse_unsupported_spec(spec: &LoadSpec) -> Result<()> {
    crate::memory_strategy::refuse_deferred_load_shape(MODEL_ID, spec)?;
    let refusals: [(&str, bool); 8] = [
        ("quantize", spec.quantize.is_some()),
        ("adapters", !spec.adapters.is_empty()),
        ("text_encoder", spec.text_encoder.is_some()),
        ("control", spec.control.is_some()),
        ("extra_controls", !spec.extra_controls.is_empty()),
        ("ip_adapter", spec.ip_adapter.is_some()),
        ("pid", spec.pid.is_some()),
        ("identity", spec.identity.is_some()),
    ];
    for (field, set) in refusals {
        if set {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: LoadSpec::{field} is not supported by the distilled FLUX 3 route \
                 (quantized variants are sc-26241, LoRA is sc-26239; the text encoder is the \
                 '{TEXT_ENCODER_COMPONENT}' component)"
            )));
        }
    }
    Ok(())
}

/// Construct the generator from the package: `spec.weights` = the package root (`config.json` +
/// `model.safetensors`), with the bundled `text_encoder/` and `ae.safetensors` unless the
/// `text_encoder` / `autoencoder` components override them. `Resident` (default) loads every
/// component now; `Sequential` loads the encoder, encodes, drops it, then loads the DiT and the
/// autoencoder, per request. A `Resident` generator runs one request the same staged way when
/// its admitted `GenerationMemory::stage_residency` is set. Everything runs on the build's device
/// ([`candle_gen::default_device`]: `cuda:0` on a CUDA build).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    gen_core::reject_unsupported_adapters(
        MODEL_ID,
        &descriptor().capabilities,
        spec.adapters.len(),
    )?;
    Ok(Box::new(load_inner(spec)?))
}

fn load_inner(spec: &LoadSpec) -> Result<Flux3Klein> {
    refuse_unsupported_spec(spec)?;
    let package_dir = match &spec.weights {
        gen_core::WeightsSource::Dir(dir) => dir.clone(),
        gen_core::WeightsSource::File(file) => {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: the weights resource must be the package directory, not {}",
                file.display()
            )))
        }
    };
    let config = Flux3Config::from_dir(&package_dir)?;
    config.validate_supported()?;
    if !config.dit.flow_map {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: {} is a CFG checkpoint; this route serves the distilled flow map (the \
             base checkpoint is sc-26239)",
            config.model
        )));
    }
    let resources = GenerationResources::from_spec(spec, &config, MODEL_ID)?;
    // Priced from the resources' tensor headers BEFORE any weight is read, so a spec the
    // registration refuses is the same refusal through `load` on every offload policy
    // (`Sequential` defers every component to the first render).
    let memory_strategy = crate::memory_strategy::generation_contract(spec, &resources, &config)?;
    let prompt_upsampler = crate::pup::Flux3PromptUpsampler::from_spec(spec)?;
    let device = candle_gen::default_device().map_err(|e| {
        Error::Unsupported(format!(
        "{MODEL_ID}: cannot initialize the selected native Candle device: {e}; use a supported \
         NVIDIA CUDA device and driver, or the native MLX backend on Apple Silicon"
    ))
    })?;
    let compute = compute_dtype(spec);
    let te_dir = resources.text_encoder_dir.clone();
    let te_cfg = config.text_encoder.clone();
    let te_device = device.clone();
    let text_in = config.dit.text_in_channels;
    let dit_path = resources.dit_weights();
    let ae_path = resources.autoencoder_file.clone();
    let heavy_cfg = config.clone();
    let heavy_device = device.clone();
    let residency = Residency::from_policy(
        spec.offload_policy,
        move || Flux3TextEncoder::load(&te_dir, &te_cfg, text_in, &te_device),
        move |_use_pid| {
            let dit = load_dit(&dit_path, &heavy_cfg, compute, &heavy_device)?;
            let ae = load_autoencoder(&ae_path, &heavy_device)?;
            if ae.latent_channels() != heavy_cfg.dit.image_in_channels {
                return Err(Error::Msg(format!(
                    "{MODEL_ID}: the autoencoder at {} packs {} channels per token, the DiT \
                     reads {}",
                    ae_path.display(),
                    ae.latent_channels(),
                    heavy_cfg.dit.image_in_channels
                )));
            }
            Ok(Heavy { dit, ae })
        },
    )?;
    Ok(Flux3Klein {
        descriptor: descriptor(),
        config,
        device,
        residency,
        prompt_upsampler,
        spec: spec.clone(),
        memory_strategy,
    })
}

/// The residency driver also calls this boundary with `None` after an encoding error. Do not
/// condition synchronization on a successful encoding: queued work must finish before text drops.
fn encoder_materialization_boundary<T>(
    synchronize: impl FnOnce() -> Result<()>,
) -> impl FnOnce(Option<&T>) -> Result<()> {
    move |_| synchronize()
}

/// Every decode is a cancellation boundary, including the last one before results can escape.
fn decode_batch<T, Out>(
    tokens: &[T],
    cancel: &gen_core::CancelFlag,
    mut decode: impl FnMut(&T) -> Result<Out>,
) -> Result<Vec<Out>> {
    let mut images = Vec::with_capacity(tokens.len());
    for token in tokens {
        candle_gen::residency::check_cancel(cancel)?;
        images.push(decode(token)?);
    }
    candle_gen::residency::check_cancel(cancel)?;
    Ok(images)
}

fn checked_completion<T>(cancel: &gen_core::CancelFlag, value: T) -> Result<T> {
    candle_gen::residency::check_cancel(cancel)?;
    Ok(value)
}

/// Fixed-flags Candle image adapter to the shared prepared-request driver. This forwards phase
/// ordering/locking/eviction to gen-core; the seam lets CPU witnesses exercise the production call.
#[allow(clippy::too_many_arguments)]
fn prepared_image_request<Text, HeavyPhase, Prepared, Encoded, Out>(
    residency: &Residency<Text, HeavyPhase>,
    stage: bool,
    cancel: &gen_core::CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    prepare: impl FnOnce(&mut dyn FnMut(Progress)) -> Result<Prepared>,
    encode: impl FnOnce(&Text, Prepared) -> Result<Encoded>,
    materialize: impl FnOnce(Option<&Encoded>) -> Result<()>,
    render: impl FnOnce(&HeavyPhase, Encoded, &mut dyn FnMut(Progress)) -> Result<Out>,
) -> Result<Out> {
    residency.run_request_scoped_with_prepare(
        stage,
        false,
        cancel,
        false,
        on_progress,
        prepare,
        encode,
        materialize,
        render,
    )
}

impl Flux3Klein {
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.validate_impl(req)?;
        let images = prepared_image_request(
            &self.residency,
            stage_residency_for(&self.spec, req),
            &req.cancel,
            on_progress,
            |on_progress| -> Result<GenerationParams> {
                // The shared driver evicts a staged warm pair BEFORE this separate VLM runs.
                // Upsampling still precedes BOTH effective-caption seeding and text encoding.
                let outcome =
                    crate::pup::effective_prompt(self.prompt_upsampler.as_ref(), req, on_progress)?;
                let mut effective = req.clone();
                effective.prompt = outcome.caption;
                Ok(GenerationParams::resolve(
                    &effective,
                    prompt_seed(&effective.prompt),
                    &self.config,
                )?)
            },
            |te: &Flux3TextEncoder,
             params: GenerationParams|
             -> Result<(TextConditioning, GenerationParams)> {
                let text = te.encode(&params.prompt)?;
                Ok((text, params))
            },
            encoder_materialization_boundary(|| Ok(self.device.synchronize()?)),
            |heavy: &Heavy, (text, params): (TextConditioning, GenerationParams), on_progress| {
                let total = params.steps * req.count;
                // Every image's Euler loop first, then ONE `Decoding` and the decodes: the
                // progress contract reports the decode phase once per request.
                let mut tokens = Vec::with_capacity(req.count as usize);
                for c in 0..req.count {
                    let done = c * params.steps;
                    let seed = candle_gen::image_seed(params.seed, c);
                    tokens.push(denoise_one(
                        &heavy.dit,
                        &text,
                        &params,
                        seed,
                        &req.cancel,
                        |step| {
                            on_progress(Progress::Step {
                                current: done + step as u32,
                                total,
                            })
                        },
                    )?);
                }
                on_progress(Progress::Decoding);
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let (h, w) = params.latent_shape()?;
                decode_batch(&tokens, &req.cancel, |t| decode_tokens(&heavy.ae, t, h, w))
            },
        )?;
        checked_completion(
            &req.cancel,
            GenerationReport {
                output: Some(GenerationOutput::Images(images)),
                artifacts: None,
                warnings: Vec::new(),
                prefix_kv_cache: None,
            },
        )
    }

    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        validate_generation_request(MODEL_ID, req, &self.config)?;
        refuse_unhonored_memory(req)?;
        Ok(())
    }
}

/// Refuse the memory-ladder controls this route has no mechanism for, so none is accepted and
/// ignored: the autoencoder decode is not tiled (sc-26241), there is no bounded-attention path
/// through the DiT and no block-streaming loader. Staged residency is the one control honoured
/// ([`stage_residency_for`]), and it is the only one the route's request scope ever writes.
fn refuse_unhonored_memory(req: &GenerationRequest) -> Result<()> {
    let Some(memory) = req.memory else {
        return Ok(());
    };
    let refusals = [
        ("tile_vae_decode", memory.tile_vae_decode),
        ("chunk_attention", memory.chunk_attention),
        (
            "stream_transformer_blocks",
            memory.stream_transformer_blocks,
        ),
    ];
    for (control, set) in refusals {
        if set {
            return Err(Error::Unsupported(format!(
                "{MODEL_ID}: GenerationMemory::{control} is not a memory strategy of the Candle \
                 distilled FLUX 3 route (it stages residency only)"
            )));
        }
    }
    Ok(())
}

impl Generator for Flux3Klein {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    /// The registered contract, priced from the loaded resources: the same one
    /// [`crate::memory_strategy::MEMORY_REGISTRATION`] builds for this spec.
    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.spec, &self.memory_strategy, context)
    }

    /// The registered executable seam on the loaded generator: the scope's `configure_request`
    /// writes the admitted `GenerationMemory` onto the request, which [`Self::generate`] honours
    /// through `Residency::run_request_scoped`.
    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        crate::memory_strategy::registered_begin_request(&self.spec, &self.memory_strategy, context)
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        let report = self.generate_impl(req, on_progress)?;
        report
            .output
            .ok_or_else(|| gen_core::Error::Msg("flux3: the render produced no output".into()))
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationReport> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::WeightsSource;

    #[test]
    fn candle_prepare_driver_evicts_before_pup_once_and_observes_prepare_cancellation() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        };
        struct Component {
            kind: &'static str,
            id: usize,
            events: Arc<Mutex<Vec<String>>>,
        }
        impl Drop for Component {
            fn drop(&mut self) {
                candle_gen::lock_recover(&self.events)
                    .push(format!("drop-{}-{}", self.kind, self.id));
            }
        }
        for stage in [true, false] {
            for cancel_prepare in [false, true] {
                let events = Arc::new(Mutex::new(Vec::new()));
                let text_count = Arc::new(AtomicUsize::new(0));
                let heavy_count = Arc::new(AtomicUsize::new(0));
                let te = events.clone();
                let tc = text_count.clone();
                let he = events.clone();
                let hc = heavy_count.clone();
                let advisory = events.clone();
                let residency = Residency::from_policy(
                    gen_core::OffloadPolicy::Resident,
                    move || {
                        let id = tc.fetch_add(1, Ordering::SeqCst);
                        candle_gen::lock_recover(&te).push(format!("load-text-{id}"));
                        Ok(Component {
                            kind: "text",
                            id,
                            events: te.clone(),
                        })
                    },
                    move |_| {
                        let id = hc.fetch_add(1, Ordering::SeqCst);
                        candle_gen::lock_recover(&he).push(format!("load-heavy-{id}"));
                        Ok(Component {
                            kind: "heavy",
                            id,
                            events: he.clone(),
                        })
                    },
                )
                .unwrap()
                .with_staged_advisory(move || {
                    candle_gen::lock_recover(&advisory).push("stage".into())
                });
                candle_gen::lock_recover(&events).clear(); // The old resident pair is already warm.
                let cancel = gen_core::CancelFlag::new();
                let result = prepared_image_request(
                    &residency,
                    stage,
                    &cancel,
                    &mut |_| {},
                    |_| {
                        candle_gen::lock_recover(&events).push("pup".into());
                        if cancel_prepare {
                            cancel.cancel();
                        }
                        Ok(17)
                    },
                    |_: &Component, prepared| {
                        candle_gen::lock_recover(&events).push("encode".into());
                        Ok(prepared)
                    },
                    encoder_materialization_boundary(|| {
                        candle_gen::lock_recover(&events).push("sync".into());
                        Ok(())
                    }),
                    |_: &Component, encoded, _| {
                        candle_gen::lock_recover(&events).push("render".into());
                        Ok(encoded)
                    },
                );
                let seen = candle_gen::lock_recover(&events).clone();
                let pup = seen.iter().position(|s| s == "pup").unwrap();
                assert_eq!(
                    seen.iter().filter(|s| s.as_str() == "stage").count(),
                    usize::from(stage)
                );
                for component in ["text", "heavy"] {
                    let old_drop = format!("drop-{component}-0");
                    assert_eq!(
                        seen.iter().filter(|s| **s == old_drop).count(),
                        usize::from(stage)
                    );
                    if stage {
                        assert!(
                            seen.iter().position(|s| *s == old_drop).unwrap() < pup,
                            "{seen:?}"
                        );
                    }
                }
                if cancel_prepare {
                    assert!(matches!(result, Err(Error::Canceled)));
                    assert!(
                        !seen
                            .iter()
                            .any(|s| s == "encode" || s == "render" || s.starts_with("load-")),
                        "{seen:?}"
                    );
                } else {
                    assert_eq!(result.unwrap(), 17);
                    assert!(pup < seen.iter().position(|s| s == "encode").unwrap());
                    if stage {
                        assert_eq!(text_count.load(Ordering::SeqCst), 2);
                        assert_eq!(heavy_count.load(Ordering::SeqCst), 2);
                        assert_eq!(
                            seen.iter().filter(|s| s.as_str() == "drop-text-1").count(),
                            1
                        );
                        assert_eq!(
                            seen.iter().filter(|s| s.as_str() == "drop-heavy-1").count(),
                            1
                        );
                    } else {
                        assert_eq!(text_count.load(Ordering::SeqCst), 1);
                        assert_eq!(heavy_count.load(Ordering::SeqCst), 1);
                    }
                }
            }
        }
    }

    #[test]
    fn cancellation_during_decode_stops_the_batch_and_withholds_the_last_result() {
        for count in [1, 8] {
            let cancel = gen_core::CancelFlag::new();
            let mut entered = 0;
            let tokens = vec![17; count];
            let result = decode_batch(&tokens, &cancel, |token| {
                entered += 1; // The decoder has begun, not merely the denoise/progress phase.
                cancel.cancel();
                Ok(*token)
            });
            assert!(matches!(result, Err(Error::Canceled)));
            assert_eq!(
                entered, 1,
                "do not decode the remaining images after cancellation"
            );
        }
    }

    #[test]
    fn cancellation_after_decode_completion_still_prevents_publication() {
        let cancel = gen_core::CancelFlag::new();
        let images = decode_batch(&[17, 23], &cancel, |token| Ok(*token)).unwrap();
        assert_eq!(images, [17, 23]);
        cancel.cancel();
        assert!(matches!(
            checked_completion(&cancel, images),
            Err(Error::Canceled)
        ));
    }

    #[test]
    fn encoder_boundary_syncs_before_drop_on_success_and_error_preserving_primary_error() {
        use std::sync::{Arc, Mutex};
        struct Text(Arc<Mutex<Vec<&'static str>>>);
        impl Drop for Text {
            fn drop(&mut self) {
                candle_gen::lock_recover(&self.0).push("text-drop");
            }
        }
        for (encode_fails, sync_fails) in
            [(false, false), (true, false), (true, true), (false, true)]
        {
            let events = Arc::new(Mutex::new(Vec::new()));
            let load_events = events.clone();
            let heavy_events = events.clone();
            let residency = Residency::from_policy(
                gen_core::OffloadPolicy::Sequential,
                move || {
                    candle_gen::lock_recover(&load_events).push("text-load");
                    Ok(Text(load_events.clone()))
                },
                move |_| {
                    candle_gen::lock_recover(&heavy_events).push("heavy-load");
                    Ok(())
                },
            )
            .unwrap();
            let result = residency.run_request_scoped(
                true,
                false,
                &gen_core::CancelFlag::new(),
                false,
                &mut |_| {},
                |_: &Text| {
                    candle_gen::lock_recover(&events).push("encode");
                    if encode_fails {
                        Err(Error::Msg("primary-encoding-error".into()))
                    } else {
                        Ok(17)
                    }
                },
                encoder_materialization_boundary(|| {
                    candle_gen::lock_recover(&events).push("synchronize");
                    if sync_fails {
                        Err(Error::Msg("synchronization-error".into()))
                    } else {
                        Ok(())
                    }
                }),
                |_: &(), encoded, _| {
                    candle_gen::lock_recover(&events).push("render");
                    Ok(encoded)
                },
            );
            let seen = candle_gen::lock_recover(&events);
            assert_eq!(
                &seen[..4],
                &["text-load", "encode", "synchronize", "text-drop"]
            );
            if encode_fails {
                assert!(
                    matches!(result, Err(Error::Msg(ref message)) if message == "primary-encoding-error")
                );
                assert_eq!(seen.len(), 4);
            } else if sync_fails {
                assert!(
                    matches!(result, Err(Error::Msg(ref message)) if message == "synchronization-error")
                );
                assert_eq!(seen.len(), 4);
            } else {
                assert_eq!(result.unwrap(), 17);
                assert_eq!(&seen[4..], &["heavy-load", "render"]);
            }
        }
    }

    #[test]
    fn descriptor_is_a_flow_map_and_honest() {
        let d = descriptor();
        assert_eq!(d.id, "flux3_klein");
        assert_eq!(d.family, "flux3");
        assert_eq!(d.backend, "candle");
        assert!(!d.capabilities.mac_only);
        assert!(d.denoiser_output_latent_space.is_none());
        assert_eq!(d.capabilities.samplers, ["euler"]);
        assert!(d.capabilities.schedulers.is_empty());
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.supports_lora && !d.capabilities.supports_lokr);
        assert!(!d.capabilities.supports_preview);
        assert!(d.required_components.is_empty());
        assert_eq!(
            d.capabilities.size_floor,
            SizeFloor::RangeCheckedOnGrid { multiple: 16 }
        );
        assert_eq!(
            (d.capabilities.min_size, d.capabilities.max_size),
            (64, 4096)
        );
    }

    /// Every unsupported load axis is a typed `Unsupported`, never a stringified message and
    /// never a torch fallback.
    ///
    /// *Mutation that reds this:* dropping any row of `refuse_unsupported_spec`'s table.
    #[test]
    fn unsupported_spec_fields_are_typed_refusals() {
        let sentinel = || LoadSpec::new(WeightsSource::Dir("/flux3-not-on-disk".into()));
        let mut quant = sentinel();
        quant.quantize = Some(gen_core::Quant::Q4);
        let mut adapters = sentinel();
        adapters.adapters.push(gen_core::AdapterSpec {
            path: "/lora-not-on-disk".into(),
            scale: 1.0,
            kind: gen_core::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        });
        let mut pid = sentinel();
        pid.pid = Some(gen_core::PidWeights {
            checkpoint: WeightsSource::File("/pid-not-on-disk.safetensors".into()),
            gemma: WeightsSource::Dir("/gemma-not-on-disk".into()),
        });
        for (field, spec) in [
            ("quantize", quant),
            ("adapters", adapters),
            ("pid", pid),
            (
                "load_shape",
                sentinel().with_load_shape(gen_core::LoadShape::DeferredMaterialization),
            ),
        ] {
            match load(&spec) {
                Err(gen_core::Error::Unsupported(m)) => assert!(m.contains(field), "{m}"),
                other => panic!("{field}: expected a typed refusal, got {:?}", other.err()),
            }
        }
    }

    /// *Mutation that reds this:* dropping either arm of the `||` in `stage_residency_for`.
    #[test]
    fn stage_residency_is_the_request_selection_or_the_sequential_policy() {
        use gen_core::{GenerationMemory, OffloadPolicy};
        let resident = LoadSpec::new(WeightsSource::Dir("/flux3-not-on-disk".into()));
        let sequential = resident
            .clone()
            .with_offload_policy(OffloadPolicy::Sequential);
        let plain = GenerationRequest::default();
        let staged = GenerationRequest {
            memory: Some(GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(!stage_residency_for(&resident, &plain));
        assert!(stage_residency_for(&resident, &staged));
        assert!(stage_residency_for(&sequential, &plain));
    }

    /// *Mutation that reds this:* removing any row of `refuse_unhonored_memory`'s table.
    #[test]
    fn unhonoured_memory_controls_are_typed_refusals() {
        use gen_core::GenerationMemory;
        let with = |memory: GenerationMemory| GenerationRequest {
            memory: Some(memory),
            ..Default::default()
        };
        refuse_unhonored_memory(&GenerationRequest::default()).unwrap();
        refuse_unhonored_memory(&with(GenerationMemory {
            stage_residency: true,
            ..Default::default()
        }))
        .unwrap();
        for (control, memory) in [
            (
                "tile_vae_decode",
                GenerationMemory {
                    tile_vae_decode: true,
                    ..Default::default()
                },
            ),
            (
                "chunk_attention",
                GenerationMemory {
                    chunk_attention: true,
                    attention_chunk_size: Some(1 << 26),
                    ..Default::default()
                },
            ),
            (
                "stream_transformer_blocks",
                GenerationMemory {
                    stream_transformer_blocks: true,
                    ..Default::default()
                },
            ),
        ] {
            match refuse_unhonored_memory(&with(memory)) {
                Err(Error::Unsupported(message)) => {
                    assert!(message.contains(control), "{message}")
                }
                other => panic!("{control}: expected a typed refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_missing_package_is_a_load_error_naming_the_config() {
        let dir = tempfile::tempdir().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        let err = match load(&spec) {
            Err(err) => err,
            Ok(_) => panic!("an empty package must fail"),
        };
        assert!(err.to_string().contains("config.json"), "{err}");
    }
}
