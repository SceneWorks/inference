//! Header-only, per-phase COUNTED pricing of one Z-Image Turbo iOS request (epic 16774): the
//! `mlx_gen::ios_pricing` interface for `z_image_turbo` (text, reference) and
//! `z_image_turbo_control` (control, control + reference).
//!
//! Every price here is the price the production guard of that quantum checks, computed by the SAME
//! functions the guard calls (`crate::quantum`, the modules' `*_bytes` / `quantum_scratch`, the
//! shared `mlx_gen::ios_quantum` kernels) over the request's shapes, and labelled with the exact
//! label the guard passes. The modules those functions read (weight shapes and dtypes: packed or
//! dense, bf16 or f32) are built from the snapshot's safetensors HEADERS as never-evaluated,
//! payload-free shape carriers (`zeros` of each header's shape and dtype): no tensor payload is read
//! and no forward graph is built. Device admission can run it.
//!
//! What runs on iOS is what is priced: the bounded VAE encode at
//! [`crate::vae::decoder::IOS_ENCODE_TILE_PX`] and the query-chunked VAE attention
//! (`VaeAttention::bounded_plan`) engage only under the iOS quantum scope, and a
//! deferred-materialization load streams every text-encoder layer, DiT block and ControlNet main
//! block through its own quantum (each block read inside it) whatever the window scope (an
//! all-covering window still materializes one block per quantum). The request's execution decides
//! staging (`stage_residency`), the decode tile edge and the DiT attention budget, read through the
//! pipeline's own resolvers.
//!
//! Each quantum is priced with what is resident while it runs: the components loaded under the
//! request's regime (staged: the text encoder only through conditioning, the DiT and ControlNet
//! dropped before the decode) plus the carried tensors live across it (the caption features, the
//! latents, the control context, the hints, and the quantum's own live input). A phase's price is its
//! worst quantum (`IosPricing::take`). The caption is priced at the full padded token window (the
//! prompt-dependent valid-token count is bounded by it).
//!
//! Not priced (a typed `Unsupported`): a load-time requantization (a snapshot stored at another tier
//! than requested), LoRA/LoKr adapters (their residuals depend on adapter payload structure), a PiD
//! decode overlay, and modes this provider does not serve.

use std::path::PathBuf;

use mlx_gen::attention::AttentionPlan;
use mlx_gen::gen_core::weightsmeta::{Dtype as HeaderDtype, SafetensorsTensorHeader};
use mlx_gen::gen_core::{GenerationMemory, LoadShape, LoadSpec, MemoryMode, WeightsSource};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosPricingRequest, IosWindowPrice, IosWorkBasis};
use mlx_gen::ios_quantum::{Backend, ConvCall};
use mlx_gen::weights::Weights;
#[cfg(test)]
use mlx_gen::GenerationRequest;
use mlx_gen::{Error, Result};
use mlx_rs::{Array, Dtype};

use crate::control_transformer::ZImageControlTransformer;
use crate::control_transformer_block::ZImageControlBlock;
use crate::quantum as q;
use crate::text_encoder::{EncoderLayer, TextEncoder, ZTextEncoderConfig};
use crate::transformer::{ZImageTransformer, ZImageTransformerConfig};
use crate::transformer_block::ZImageTransformerBlock;
use crate::vae::Vae;

#[path = "ios_pricing/adapter_sources.rs"]
mod adapter_sources;
#[path = "ios_pricing/pid_sources.rs"]
mod pid_sources;
#[path = "ios_pricing/pid_work_basis.rs"]
mod pid_work_basis;

/// The phase every Z-Image quantum label (and the shared labels a Z-Image request emits: the
/// curated sampler's step and an adapter load) runs in. `None` for a label Z-Image never emits.
pub fn ios_quantum_phase(label: &str) -> Option<IosPhase> {
    use IosPhase::*;
    Some(match label {
        q::TE_LOAD
        | q::DIT_LOAD
        | q::VAE_LOAD
        | q::CONTROL_LOAD
        | q::CONTROL_QUANTIZE
        | mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM => Load,
        q::TE_EMBED | q::TE_LAYER | q::CAPTION_FEATURES => Conditioning,
        q::INIT_IMAGE
        | q::VAE_ENCODE_IN
        | q::VAE_ENCODE_RESNET
        | q::VAE_ENCODE_ATTENTION
        | q::VAE_DOWNSAMPLE
        | q::VAE_ENCODE_OUT
        | q::VAE_ENCODE_TILE
        | q::VAE_ENCODE_TILED_STAGE
        | q::CONTROL_CONTEXT => Encode,
        q::LATENTS
        | q::DIT_PREPARE
        | q::DIT_EMBED
        | q::DIT_CAPTION
        | q::DIT_BLOCK
        | q::DIT_CONTEXT_BLOCK
        | q::DIT_UNIFY
        | q::DIT_OUTPUT
        | q::CONTROL_EMBED
        | q::CONTROL_BLOCK
        | mlx_gen::sampler::SAMPLER_STEP_QUANTUM => Denoise,
        q::VAE_DECODE_IN
        | q::VAE_DECODE_RESNET
        | q::VAE_DECODE_ATTENTION
        | q::VAE_UPSAMPLE
        | q::VAE_DECODE_OUT
        | q::VAE_DECODE_TILE
        | q::VAE_DECODE_TILED_STAGE
        | q::DECODED_TO_IMAGE => Decode,
        _ => return None,
    })
}

/// Legacy diagnostic inventory, NOT an accepted additivity domain or capture-family policy.
/// The old `derive`/packed-matrix fixtures do not certify correlated native/process transfer.
/// Runtime and capture grouping preserve distinct actual control, control+init and reference
/// families until a source-certified domain proves sharing. Empty here grants no inheritance.
pub const IOS_NON_ADDITIVE_MODES: &[&str] = &[];

/// Validate the actual Z conditioning-role partition. `aggregate` retains the existing memory
/// geometry meaning: ALL image carriers, not just true `Conditioning::Reference` inputs. Callers
/// must reject unknown conditioning kinds before projecting these three explicit scalar counts.
/// This pure provider rule is shared by live pricing and payload-free captured-token replay.
pub fn validate_pricing_roles(
    provider: &str,
    aggregate: u32,
    true_refs: u32,
    masks: u32,
    controls: u32,
) -> Result<()> {
    let sum = true_refs
        .checked_add(masks)
        .and_then(|sum| sum.checked_add(controls));
    let provider_roles = match provider {
        crate::model::MODEL_ID => controls == 0,
        crate::model_control::MODEL_ID => controls == 1,
        _ => false,
    };
    if sum != Some(aggregate) || true_refs > 1 || masks != 0 || !provider_roles {
        return Err(Error::Unsupported(format!(
            "{provider}: invalid Z-Image pricing conditioning-role partition"
        )));
    }
    Ok(())
}

fn pricing_roles_from_kinds(
    kinds: impl IntoIterator<Item = mlx_gen::ConditioningKind>,
) -> Result<(u32, u32, u32)> {
    let mut counts = [0_u32; 3];
    for kind in kinds {
        let role = match kind {
            mlx_gen::ConditioningKind::Reference => 0,
            mlx_gen::ConditioningKind::Mask => 1,
            mlx_gen::ConditioningKind::Control => 2,
            _ => {
                return Err(Error::Unsupported(
                    "Z-Image pricing: unsupported conditioning kind".into(),
                ))
            }
        };
        counts[role] = counts[role].checked_add(1).ok_or_else(|| {
            Error::Unsupported("Z-Image pricing conditioning count overflow".into())
        })?;
    }
    Ok((counts[0], counts[1], counts[2]))
}

/// Price one iOS request header-only on the device backend (Metal).
pub fn ios_pricing(req: &IosPricingRequest<'_>) -> Result<IosPricing> {
    req.validate_work_context()?;
    let generation = mlx_gen::sampler::flow_sampling::require_generation(req.generation)?;
    let (true_refs, masks, controls) = pricing_roles_from_kinds(
        generation
            .conditioning
            .iter()
            .map(mlx_gen::Conditioning::kind),
    )?;
    validate_pricing_roles(
        req.provider,
        req.geometry.reference_count,
        true_refs,
        masks,
        controls,
    )?;
    let initial_width = crate::sampling::initial_state_width(generation, true_refs == 1)?;
    let route = Route::from_controls(req, generation.count, initial_width, true_refs)?;
    let sampling = crate::sampling::resolve_with_init(generation, route.mode.img2img())?;
    price_request_controls(req, route, sampling, initial_width)
}

/// Payload-free anchor replay. No GenerationRequest or RGB carrier is constructed. The caller
/// supplies explicit typed roles from its validated closed conditioning set, never an aggregate
/// guessed to be a Reference count. No compatibility wrapper invents missing role facts.
pub fn ios_pricing_with_token(
    req: &IosPricingRequest<'_>,
    token: &str,
    count: u32,
    true_reference_count: u32,
    mask_count: u32,
    control_count: u32,
) -> Result<IosPricing> {
    req.validate_work_context()?;
    validate_pricing_roles(
        req.provider,
        req.geometry.reference_count,
        true_reference_count,
        mask_count,
        control_count,
    )?;
    let (recipe,pid_work)=pid_work_basis::decode(token,req.generation.map(pid_work_basis::PidWorkBasis::actual))?;
    let sampling = mlx_gen::sampler::flow_sampling::ResolvedFlowControls::from_recipe(
        &recipe,
        crate::sampling::defaults(),
        mlx_gen::TimestepConvention::OneMinusSigma,
    )?;
    let active_init = true_reference_count == 1
        && crate::pipeline::init_time_step(sampling.steps(), recipe.effective_strength()) > 0;
    let initial_width = if active_init { 4 } else { 2 };
    let route = Route::from_controls(req, count, initial_width, true_reference_count)?;
    price_request_controls_with_pid_work(req, route, sampling, initial_width, pid_work)
}

fn price_request_controls(
    req: &IosPricingRequest<'_>,
    route: Route,
    sampling: mlx_gen::sampler::flow_sampling::ResolvedFlowControls,
    initial_width: u64,
) -> Result<IosPricing> {
    price_request_controls_with_pid_work(req,route,sampling,initial_width,req.generation.map(pid_work_basis::PidWorkBasis::actual))
}

fn price_request_controls_with_pid_work(
    req:&IosPricingRequest<'_>,route:Route,
    sampling:mlx_gen::sampler::flow_sampling::ResolvedFlowControls,initial_width:u64,
    pid_work:Option<pid_work_basis::PidWorkBasis>,
)->Result<IosPricing> {
    if req.spec.pid.is_some() || pid_sources::requested(&sampling)? {
        // Original overlay spec/validated controls reach the fixed PiD source consumer BEFORE
        // Facts' legacy overlay refusal. Adapter-refused cases never open later PiD files.
        return pid_sources::price_before_existing_guard(req,&route,&sampling,initial_width,pid_work);
    }
    if !req.spec.adapters.is_empty() {
        // Reach the real source-case consumer before the legacy Facts adapter guard. Base-only
        // metadata extraction leaves the ORIGINAL requested adapter vector on the visitor path.
        return adapter_sources::price_before_existing_guard(req, &route, &sampling, initial_width);
    }
    let facts = Facts::from_spec(req.provider, req.spec)?;
    if facts
        .quant_conversions
        .iter()
        .any(|component| !component.is_empty())
    {
        return Err(Error::Unsupported(format!("{}: load-time quantization requires complete first-use and streamed source-owner windows", req.provider)));
    }
    let model = Model::new(facts)?;
    let mut envelope = std::collections::BTreeMap::<String, IosWindowPrice>::new();
    let mut absent = std::collections::BTreeMap::new();
    model.visit_quanta(&route, Backend::Metal, &mut |quanta| {
        if quanta.iter().any(|op| op.source_vae.as_ref().is_some_and(|world| !world.conversions.is_empty())) {
            return Err(Error::Unsupported(format!("{}: load-time quantization requires complete first-use and streamed source-owner windows", req.provider)));
        }
        let priced = price_source_quanta(&model.facts, quanta, &route, &sampling, initial_width)?;
        absent.extend(priced.absent_phases);
        // Envelope only AFTER the whole successful history went through constructor, packing,
        // forward metadata and bounded cold/recurrent/serial replay. Keep one WHOLE winning
        // window, never independent resident/scratch/role maxima or an allocated world product.
        for window in priced.windows {
            let key = window.class.clone().expect("source operation class");
            match envelope.get_mut(&key) {
                Some(old) if old.logical_peak_bytes() >= window.logical_peak_bytes() => {}
                Some(old) => *old = window,
                None => { envelope.insert(key, window); }
            }
        }
        Ok(())
    })?;
    let mut pricing = IosPricing::default();
    for window in envelope.into_values() { pricing.take_window(window); }
    pricing.absent_phases = absent;
    Ok(pricing)
}

/// Source-bounded callback presence, never an allocated requested-N schedule. Native Z has exactly
/// N positive nodes plus its clean terminal; a curated schedule can stride/deduplicate that count.
/// For a sliced curated schedule retain both POSSIBLE families, not a fabricated effective N.
fn source_callback_presence(
    route: &Route,
    sampling: &mlx_gen::sampler::flow_sampling::ResolvedFlowControls,
) -> Result<(bool, bool)> {
    let start = if route.mode.img2img() {
        crate::pipeline::init_time_step(sampling.steps(), sampling.recipe().effective_strength())
    } else {
        0
    };
    if sampling.uses_native_schedule() {
        if start > sampling.steps() {
            return Err(Error::Unsupported(
                "Z-Image init index exceeds the native schedule".into(),
            ));
        }
        Ok((start < sampling.steps(), start == sampling.steps()))
    } else {
        Ok((true, start > 0))
    }
}

fn callback_parameter_history(
    first_model_callback: bool,
    completed_outputs: u32,
) -> QuantizationHistory {
    if first_model_callback && completed_outputs == 0 {
        QuantizationHistory::RequestFirst
    } else {
        QuantizationHistory::DenoiseRecurrent
    }
}

fn replay_window(replay: &[Option<IosWindowPrice>], index: usize) -> Result<IosWindowPrice> {
    replay
        .get(index)
        .and_then(Option::as_ref)
        .cloned()
        .ok_or_else(|| Error::Msg("missing SAME-operation Z source replay candidate".into()))
}

fn price_source_quanta(
    facts: &Facts,
    quanta: &[Priced],
    route: &Route,
    sampling: &mlx_gen::sampler::flow_sampling::ResolvedFlowControls,
    initial_width: u64,
) -> Result<IosPricing> {
    let elements = u64::from(route.width / crate::pipeline::SPATIAL_SCALE)
        * u64::from(route.height / crate::pipeline::SPATIAL_SCALE)
        * crate::pipeline::LATENT_CHANNELS as u64;
    // Finite possible epoch families, not a user-N replay or actual capture witness. F32 return
    // width is a logical promotion envelope; it does not certify backing or process growth.
    let callbacks = mlx_gen::sampler::solver_memory::solver_callback_windows(
        sampling.solver(),
        sampling.convention(),
        elements,
        initial_width,
        4,
    )?;
    let host_schedule = sampling.host_schedule_bytes()?;
    let (forward_possible, zero_possible) = source_callback_presence(route, sampling)?;
    let first = facts.replay_quanta(quanta, route, QuantizationHistory::RequestFirst, 0, false)?;
    let endpoints = if route.count > 1 {
        vec![0, route.count - 1]
    } else {
        vec![0]
    };
    let mut pricing = IosPricing::default();
    let mut take = |mut window: IosWindowPrice| -> Result<()> {
        // The actual heavy closure builds its schedule before VAE init/control encodes and drops
        // it before returning latents to decode. Do not move its owners into load or late decode.
        if matches!(window.phase, IosPhase::Encode | IosPhase::Denoise) {
            window.held_bytes = window
                .held_bytes
                .checked_add(host_schedule)
                .ok_or_else(|| Error::Unsupported("Z host schedule owners overflow".into()))?;
        }
        pricing.take_window(window);
        Ok(())
    };
    for (index, operation) in quanta.iter().enumerate() {
        if !matches!(operation.window.phase, IosPhase::Denoise | IosPhase::Decode) {
            take(replay_window(&first, index)?)?;
        }
    }
    for completed in endpoints {
        let entry_history = callback_parameter_history(true, completed);
        let entry = facts.replay_quanta(quanta, route, entry_history, completed, false)?;
        let recurrent = facts.replay_quanta(
            quanta,
            route,
            QuantizationHistory::DenoiseRecurrent,
            completed,
            false,
        )?;
        let zero = if zero_possible {
            Some(facts.replay_quanta(
                quanta,
                route,
                QuantizationHistory::NoModelForward,
                completed,
                false,
            )?)
        } else {
            None
        };
        for (index, operation) in quanta.iter().enumerate() {
            if operation.window.phase != IosPhase::Denoise {
                continue;
            }
            if denoise_callback_body(operation.label) {
                if !forward_possible {
                    continue;
                }
                let step = operation.label == mlx_gen::sampler::SAMPLER_STEP_QUANTUM;
                for callback in &callbacks {
                    // Only output0's FIRST actual model evaluation can consume persistent cold
                    // parents. A corrector/history callback or any later output cannot. Streamed
                    // local blocks remain cold in operation.scratch, independently of this cursor.
                    let history =
                        callback_parameter_history(callback.first_model_callback, completed);
                    let source = if step || history == QuantizationHistory::DenoiseRecurrent {
                        &recurrent
                    } else {
                        &entry
                    };
                    let mut window = replay_window(source, index)?;
                    window.class = Some(format!(
                        "{}/{}/{}",
                        window.class.as_deref().unwrap(),
                        sampling.allocation_class(),
                        callback.family()
                    ));
                    let legacy_current = elements
                        .checked_mul(if step { 8 } else { 4 })
                        .ok_or_else(|| {
                            Error::Unsupported("Z callback state bytes overflow".into())
                        })?;
                    let current = if step || operation.label == q::DIT_EMBED {
                        0
                    } else {
                        callback.state_bytes
                    };
                    window.held_bytes = window
                        .held_bytes
                        .checked_sub(legacy_current)
                        .and_then(|value| value.checked_add(current))
                        .and_then(|value| value.checked_add(callback.callback_outer_bytes))
                        .ok_or_else(|| {
                            Error::Unsupported(
                                "Z callback owners do not match their source operation".into(),
                            )
                        })?;
                    take(window)?;
                }
                if step {
                    let mut final_window = replay_window(&recurrent, index)?;
                    final_window.class = Some(format!(
                        "{}/{}/solve-complete",
                        final_window.class.as_deref().unwrap(),
                        sampling.allocation_class()
                    ));
                    final_window.held_bytes = final_window
                        .held_bytes
                        .checked_sub(elements.saturating_mul(8))
                        .ok_or_else(|| {
                            Error::Unsupported(
                                "Z solve-complete state is not in its source operation".into(),
                            )
                        })?;
                    take(final_window)?;
                }
            } else {
                // Noise/prepare/control embedding run even if the sliced solve has NO callbacks.
                // For later outputs, only a preceding positive solve warmed persistent DiT graphs.
                if forward_possible {
                    take(replay_window(&entry, index)?)?;
                }
                if let Some(zero) = &zero {
                    take(replay_window(zero, index)?)?;
                }
            }
        }
    }
    // Z decode_batch CONSUMES its latent Vec. This is not Anima/SANA's borrowed latent lifetime:
    // remaining latents fall as prior RGB outputs grow, so retain the two coincident endpoints.
    for history in [
        QuantizationHistory::RequestFirst,
        QuantizationHistory::NoModelForward,
    ] {
        if (history == QuantizationHistory::RequestFirst && !forward_possible)
            || (history == QuantizationHistory::NoModelForward && !zero_possible)
        {
            continue;
        }
        for late in [false, true] {
            if late && route.count == 1 {
                continue;
            }
            let replay = facts.replay_quanta(quanta, route, history, 0, late)?;
            for (index, operation) in quanta.iter().enumerate() {
                if operation.window.phase != IosPhase::Decode {
                    continue;
                }
                let mut window = replay_window(&replay, index)?;
                if late {
                    let removed = u64::from(route.count - 1)
                        .checked_mul(elements)
                        .and_then(|value| value.checked_mul(4))
                        .ok_or_else(|| {
                            Error::Unsupported("Z consumed latent owners overflow".into())
                        })?;
                    let rgb = u64::from(route.count - 1)
                        .checked_mul(3)
                        .and_then(|value| value.checked_mul(u64::from(route.width)))
                        .and_then(|value| value.checked_mul(u64::from(route.height)))
                        .ok_or_else(|| Error::Unsupported("Z prior RGB owners overflow".into()))?;
                    window.held_bytes = window
                        .held_bytes
                        .checked_sub(removed)
                        .and_then(|value| value.checked_add(rgb))
                        .ok_or_else(|| {
                            Error::Unsupported(
                                "Z consuming decode owners do not match their source operation"
                                    .into(),
                            )
                        })?;
                }
                take(window)?;
            }
        }
    }
    drop(take);
    if !route.mode.img2img() && !route.mode.control() {
        pricing.phase_not_run(
            IosPhase::Encode,
            "text mode has neither active init nor control encode",
        );
    }
    Ok(pricing)
}

/// Base-provider wrapper. Global dispatch across both Z-Image providers must call the explicit
/// `_for` form; a configured ControlNet is not activated just by Some(spec.control) on base Z.
pub fn configured_overlay_semantic_facts(spec: &LoadSpec) -> Result<serde_json::Value> {
    configured_overlay_semantic_facts_for(crate::model::MODEL_ID, spec)
}

/// Actual provider-selected overlay graph. Uses native target aliases/packing rules and headers;
/// only an indispensable bounded alpha control may read payload bytes. No model/Array/media.
pub fn configured_overlay_semantic_facts_for(
    provider: &str,
    spec: &LoadSpec,
) -> Result<serde_json::Value> {
    use mlx_gen::adapters::facts::{dtype_name, linear_header_facts, native_header_dtype};
    use mlx_gen::gen_core::weightsmeta as wm;
    let (root, control) = if provider == crate::model::MODEL_ID {
        (
            crate::model::resolve_precision_and_root(
                spec,
                crate::model::PRECISION_ERROR,
                crate::model::FILE_ERROR,
            )?,
            None,
        )
    } else if provider == crate::model_control::MODEL_ID {
        let (root, source) = crate::model_control::resolve_control_base_and_control(
            spec,
            provider,
            crate::model_control::PRECISION_ERROR,
        )?;
        (root, Some(source))
    } else {
        return Err(Error::Unsupported(format!(
            "Z-Image overlay facts: unknown provider {provider}"
        )));
    };
    let pid = spec
        .pid
        .as_ref()
        .map(|pid| -> Result<serde_json::Value> {
            let facts =
                mlx_gen_pid::engine::configured_header_facts(pid, crate::model::PID_BACKBONE)?;
            let parameters =
                |headers: &[SafetensorsTensorHeader]| -> Result<Vec<serde_json::Value>> {
                    headers
                        .iter()
                        .map(|h| {
                            Ok(serde_json::json!({"parameter":h.name,
                "shape":h.shape,"dtype":dtype_name(native_header_dtype(h.dtype)?)}))
                        })
                        .collect()
                };
            Ok(
                serde_json::json!({"kind":"pid_flux_student","v1pt5":facts.v1pt5,
            "latent_channels":facts.config.lq_latent_channels,
            "latent_spatial_down_factor":facts.config.latent_spatial_down_factor,
            "sr_scale":facts.config.sr_scale,"caption_dtype":"BF16",
            "student":parameters(&facts.student)?,"gemma":parameters(&facts.gemma)?}),
            )
        })
        .transpose()?;
    let load_quantize = if let Some(quant) = spec.quantize {
        // The real load calls transformer.quantize even on a matching packed component. Each
        // packed leaf is a no-op, but a dense leaf in a mixed snapshot still converts.
        mlx_gen::quant::needs_load_time_quant(root, "transformer", quant.bits(), provider)?;
        Some(quant.bits())
    } else {
        None
    };
    let headers =
        wm::safetensors_path_tensor_headers(root.join("transformer")).map_err(Error::from)?;
    for header in &headers {
        native_header_dtype(header.dtype)?;
    }
    let mut headers = headers
        .into_iter()
        .map(|header| (header.name.clone(), header))
        .collect::<std::collections::BTreeMap<_, _>>();
    crate::loader::remap_transformer_header_keys(&mut headers);
    let cfg = ZImageTransformerConfig::turbo();
    let (adapters, _) = configured_z_adapter_facts(spec, &headers, &cfg, load_quantize)?;
    let quantization_parents = configured_z_quantization_parents(&headers, &cfg, load_quantize)?;
    let lazy_parent_projections = quantization_parents
        .iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();
    let control = if let Some(source) = control {
        let component = Component::read(source)?;
        let headers = component
            .headers
            .into_iter()
            .map(|header| (header.name.clone(), header))
            .collect::<std::collections::BTreeMap<_, _>>();
        let quantized_targets = control_quantization_targets()
            .into_iter()
            .map(|(path, _)| path)
            .collect::<std::collections::BTreeSet<_>>();
        let mut parameters = Vec::new();
        for header in headers.values() {
            let dtype = native_header_dtype(header.dtype)?;
            if header.name.ends_with(".scales") || header.name.ends_with(".biases") {
                continue;
            }
            let block_matrix = header
                .name
                .strip_suffix(".weight")
                .is_some_and(|path| quantized_targets.contains(path));
            if block_matrix {
                let base = header.name.strip_suffix(".weight").unwrap();
                // The actual composed loader always quantizes ControlNet blocks at requested bits;
                // its control_x_embedder stays dense because its input width is not group aligned.
                let facts = linear_header_facts(
                    &headers,
                    base,
                    mlx_gen::quant::DEFAULT_GROUP_SIZE,
                    spec.quantize.map(|q| q.bits()),
                )?;
                let retained_dtype = if facts.is_quantized {
                    headers
                        .get(&format!("{base}.scales"))
                        .map(|h| native_header_dtype(h.dtype))
                        .transpose()?
                        .unwrap_or(Dtype::Bfloat16)
                } else {
                    dtype
                };
                parameters.push(
                    serde_json::json!({"parameter":base,"shape":facts.base_shape,
                    "layout":if facts.is_quantized {"affine_packed_linear"} else {"dense_linear"},
                    "dtype":dtype_name(retained_dtype)}),
                );
            } else {
                parameters.push(
                    serde_json::json!({"parameter":header.name,"shape":header.shape,
                    "layout":"dense_parameter","dtype":dtype_name(dtype)}),
                );
            }
        }
        Some(
            serde_json::json!({"kind":"fun_controlnet_union","parameters":parameters,
            "main_places":crate::control_transformer::CONTROL_LAYERS_PLACES,
            "refiner_places":crate::control_transformer::CONTROL_REFINER_PLACES,
            "context_dtype":"F32","embedder_quantized":false}),
        )
    } else {
        None
    };
    Ok(
        serde_json::json!({"schema":"z_image_configured_overlay_semantics_v1","adapters":adapters,"control":control,"pid":pid,
            "load_quantization":{"bits":if lazy_parent_projections.is_empty() {None} else {load_quantize},"lazy_parent_projections":lazy_parent_projections}}),
    )
}

/// Known logical source parents of the actual DiT load-time pack. The returned bytes are a
/// non-subtractive source-owner upper only; this does not complete the load/first-use profile.
/// Already packed leaves contribute no conversion parents, even in a mixed component.
fn configured_z_quantization_parents(
    headers: &std::collections::BTreeMap<String, SafetensorsTensorHeader>,
    cfg: &ZImageTransformerConfig,
    bits: Option<i32>,
) -> Result<Vec<(String, u64)>> {
    let Some(bits) = bits else {
        return Ok(Vec::new());
    };
    let source = Component {
        files: Vec::new(),
        headers: headers.values().cloned().collect(),
    };
    let targets = crate::transformer::adapter_header_targets(cfg, true)
        .into_iter()
        .map(|(_, header)| (header, false))
        .collect::<Vec<_>>();
    let (_, conversions) = source.load_quantized_with_bias(
        &targets,
        bits,
        crate::quant::dit_quantization_has_bias,
    )?;
    conversions
        .into_iter()
        .map(|(path, old, _, casts)| {
            Ok((
                path,
                old.checked_add(casts).ok_or_else(|| {
                    Error::Unsupported("quantization parent ownership exceeds byte range".into())
                })?,
            ))
        })
        .collect()
}

fn configured_z_adapter_facts(
    spec: &LoadSpec,
    headers: &std::collections::BTreeMap<String, SafetensorsTensorHeader>,
    cfg: &ZImageTransformerConfig,
    load_quantize: Option<i32>,
) -> Result<(
    serde_json::Value,
    Vec<(String, mlx_gen::adapters::facts::AdapterFacts)>,
)> {
    use mlx_gen::adapters::facts::linear_header_facts;
    let kohya = crate::transformer::adapter_header_targets(&cfg, false)
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();
    let mut projections = std::collections::BTreeMap::new();
    for (path, header) in crate::transformer::adapter_header_targets(&cfg, true) {
        projections.insert(
            path,
            linear_header_facts(
                &headers,
                &header,
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
                load_quantize,
            )?,
        );
    }
    mlx_gen::adapters::loader::configured_adapter_facts(
        &spec.adapters,
        &kohya,
        mlx_gen::adapters::DEFAULT_LYCORIS_DTYPE,
        |path| {
            let Some(canonical) = crate::transformer::canonical_adapter_path(&cfg, path) else {
                return Ok(None);
            };
            Ok(projections
                .get(&canonical)
                .cloned()
                .map(|facts| (canonical, facts)))
        },
    )
}

#[cfg(test)]
mod pricing_role_tests {
    use super::*;
    use mlx_gen::ConditioningKind as Kind;

    #[test]
    fn actual_role_projection_preserves_control_only_and_optional_reference() {
        let actual = [
            vec![],
            vec![Kind::Control],
            vec![Kind::Control, Kind::Reference],
            vec![Kind::Reference],
        ]
        .into_iter()
        .map(|kinds| pricing_roles_from_kinds(kinds).unwrap())
        .collect::<Vec<_>>();
        assert_eq!(actual, [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)]);
    }

    #[test]
    fn distinct_or_unknown_image_kinds_are_never_assumed_true_references() {
        let actual = [
            Kind::ReferenceRgba,
            Kind::MultiReference,
            Kind::Depth,
            Kind::VideoClip,
        ]
        .into_iter()
        .map(|kind| pricing_roles_from_kinds([kind]).is_err())
        .collect::<Vec<_>>();
        assert_eq!(actual, [true, true, true, true]);
    }

    #[test]
    fn actual_z_role_bounds_accept_only_the_supported_closed_partitions() {
        let base = crate::model::MODEL_ID;
        let control = crate::model_control::MODEL_ID;
        let actual = [
            (base, 0, 0, 0, 0),
            (base, 1, 1, 0, 0),
            (control, 1, 0, 0, 1),
            (control, 2, 1, 0, 1),
        ]
        .into_iter()
        .map(|(p, a, r, m, c)| validate_pricing_roles(p, a, r, m, c).is_ok())
        .collect::<Vec<_>>();
        assert_eq!(actual, [true, true, true, true]);
    }

    #[test]
    fn aggregate_role_mismatches_masks_and_duplicate_controls_are_rejected() {
        let base = crate::model::MODEL_ID;
        let control = crate::model_control::MODEL_ID;
        let actual = [
            (control, 2, 0, 0, 1),
            (control, 2, 0, 0, 2),
            (control, 2, 0, 1, 1),
            (control, 1, 1, 0, 0),
            (base, 1, 0, 0, 1),
            (base, 2, 2, 0, 0),
            ("unknown-z-provider", 0, 0, 0, 0),
            (control, 0, u32::MAX, 0, 1),
        ]
        .into_iter()
        .map(|(p, a, r, m, c)| validate_pricing_roles(p, a, r, m, c).is_err())
        .collect::<Vec<_>>();
        assert_eq!(actual, [true; 8]);
    }
}

fn denoise_callback_body(label: &str) -> bool {
    matches!(
        label,
        q::DIT_EMBED
            | q::DIT_CAPTION
            | q::DIT_BLOCK
            | q::DIT_CONTEXT_BLOCK
            | q::DIT_UNIFY
            | q::DIT_OUTPUT
            | q::CONTROL_BLOCK
    ) || label == mlx_gen::sampler::SAMPLER_STEP_QUANTUM
}

/// One priced quantum: the guard's label, its counted scratch and what is resident while it runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Priced {
    pub(crate) label: &'static str,
    pub(crate) resident: u64,
    pub(crate) scratch: u64,
    pub(crate) window: IosWindowPrice,
    // SAME producer operation BEFORE conversion partitioning. These are not differences between
    // independent maxima and do not assert allocation disjointness or native backing release.
    source_structural: u64,
    source_held: u64,
    source_point: usize,
    source_vae: Option<std::sync::Arc<VaeOwnerHistory>>,
}

/// The per-phase pricing of `quanta`: each phase its worst quantum.
#[cfg(test)]
pub(crate) fn fold(quanta: &[Priced]) -> IosPricing {
    let mut pricing = IosPricing::default();
    for p in quanta {
        pricing.take_window(p.window.clone());
    }
    pricing
}

/// The request modes this provider serves on iOS (`mlx-gen-ios-catalog::ios_admission::work_mode`:
/// a control request is `Other("control")`, with or without a reference init image).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Text,
    Reference,
    Control,
    ControlReference,
}

impl Mode {
    fn key(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Reference => "reference",
            Self::Control => "control",
            Self::ControlReference => "control-reference",
        }
    }
    pub(crate) fn control(self) -> bool {
        matches!(self, Mode::Control | Mode::ControlReference)
    }
    pub(crate) fn img2img(self) -> bool {
        matches!(self, Mode::Reference | Mode::ControlReference)
    }
}

/// One request's route: mode, geometry, and the execution levers written onto it.
#[derive(Clone, Debug)]
pub(crate) struct Route {
    pub(crate) mode: Mode,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) count: u32,
    pub(crate) execution: GenerationMemory,
}

impl Route {
    fn from_controls(
        req: &IosPricingRequest<'_>,
        count: u32,
        initial_width: u64,
        true_reference_count: u32,
    ) -> Result<Self> {
        if count == 0 {
            return Err(Error::Unsupported(
                "pricing requires a positive serial output count".into(),
            ));
        }
        let control_route = req.provider == crate::model_control::MODEL_ID;
        // Both callers already applied validate_pricing_roles to the explicit typed partition.
        // The mode must agree with TRUE Reference presence, not the aggregate image-carrier count.
        let refs_valid = if control_route {
            true_reference_count <= 1
        } else if matches!(req.mode, MemoryMode::ImageToImage) {
            true_reference_count == 1
        } else {
            true_reference_count == 0
        };
        if !refs_valid {
            return Err(Error::Unsupported(
                "Z-Image typed mode/reference roles disagree".into(),
            ));
        }
        let mode = match (req.mode, control_route) {
            (MemoryMode::TextToImage, false) => Mode::Text,
            (MemoryMode::ImageToImage, false) => Mode::Reference,
            (MemoryMode::Other(m), true) if m == "control" => {
                // A control image contributes to the aggregate but is never an img2img init.
                if true_reference_count == 1 {
                    Mode::ControlReference
                } else {
                    Mode::Control
                }
            }
            (mode, _) => {
                return Err(Error::Unsupported(format!(
                    "{}: iOS pricing does not serve mode {mode:?}",
                    req.provider
                )))
            }
        };
        // Carrier presence is not an actual VAE init: the production start rule treats an absent
        // or zero strength as txt2img. Keep ControlNet context but omit the inactive init encode.
        let mode = if mode.img2img() && initial_width == 2 {
            if mode.control() {
                Mode::Control
            } else {
                Mode::Text
            }
        } else {
            mode
        };
        Ok(Self {
            mode,
            width: req.geometry.width,
            height: req.geometry.height,
            count: count.max(1),
            execution: *req.execution,
        })
    }

    /// The request the pipeline's resolvers read (geometry, count, execution).
    #[cfg(test)]
    pub(crate) fn request(&self) -> GenerationRequest {
        GenerationRequest {
            width: self.width,
            height: self.height,
            count: self.count,
            memory: Some(self.execution),
            ..Default::default()
        }
    }
}

/// Actual ControlNet quantizer targets: the native main/refiner constructor loops, with the
/// before-projection only on block zero of EACH stack. The dense patch embedder is deliberately
/// absent, even if some unrelated checkpoint tensor happens to be two-dimensional.
pub(crate) fn control_quantization_targets() -> Vec<(String, bool)> {
    [
        (
            "control_layers",
            crate::control_transformer::CONTROL_LAYERS_PLACES.len(),
        ),
        (
            "control_noise_refiner",
            crate::control_transformer::CONTROL_REFINER_PLACES.len(),
        ),
    ]
    .into_iter()
    .flat_map(|(stack, count)| {
        (0..count).flat_map(move |index| {
            ZImageControlBlock::quantization_header_targets(&format!("{stack}.{index}"), index == 0)
        })
    })
    .collect()
}

/// One component's safetensors files and their headers.
#[derive(Clone, Debug)]
pub(crate) struct Component {
    pub(crate) files: Vec<PathBuf>,
    pub(crate) headers: Vec<SafetensorsTensorHeader>,
}

impl Component {
    pub(crate) fn read(source: &WeightsSource) -> Result<Self> {
        let files = q::component_files(source)?;
        if files.is_empty() {
            return Err(Error::Msg(format!(
                "iOS pricing: no .safetensors under {source:?}"
            )));
        }
        let mut headers = Vec::new();
        for file in &files {
            headers.extend(
                mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(file)
                    .map_err(|e| Error::Msg(e.to_string()))?,
            );
        }
        Ok(Self { files, headers })
    }

    /// Build only the actual packed tensor METADATA; never execute the quantizer or evaluate a
    /// parent. `(projection, old owners, retained outputs, temporary casts)` records keep the
    /// same source conversion together so later first-use partitions need no independent maxima.
    /// Already packed leaves are exactly the native no-op. TokenEmbedding packs as-is; Linear
    /// casts weight/bias to BF16. These are logical source owners, not physical backing bounds.
    #[cfg(test)]
    fn load_quantized(
        &self,
        targets: &[(String, bool)],
        bits: i32,
    ) -> Result<(Self, Vec<(String, u64, u64, u64)>)> {
        self.load_quantized_with_bias(targets, bits, |_| true)
    }

    /// Convert only parameters the native constructor actually carries. A bias-free Linear ignores
    /// an authored sibling bias; retain that raw header unchanged without inventing a module cast,
    /// packed parent, or forward input. Its dictionary/cache/native owners are separate inventory,
    /// not released by this policy. `targets` is the provider's closed native quantizer schema.
    fn load_quantized_with_bias(
        &self,
        targets: &[(String, bool)],
        bits: i32,
        has_bias: impl Fn(&str) -> bool,
    ) -> Result<(Self, Vec<(String, u64, u64, u64)>)> {
        use mlx_gen::adapters::facts::{linear_header_facts, native_header_dtype, TensorFacts};
        if !matches!(bits, 4 | 8) {
            return Err(Error::Unsupported(format!(
                "unsupported load-time quantization Q{bits}"
            )));
        }
        let mut headers = self
            .headers
            .iter()
            .map(|h| (h.name.clone(), h.clone()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut records = Vec::new();
        let mut visited = std::collections::BTreeSet::new();
        let tensor = |header: &SafetensorsTensorHeader| -> Result<TensorFacts> {
            Ok(TensorFacts {
                shape: header
                    .shape
                    .iter()
                    .map(|&dim| {
                        i32::try_from(dim).map_err(|_| {
                            Error::Unsupported("quantization source exceeds native shape".into())
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                dtype: native_header_dtype(header.dtype)?,
                strides: Vec::new(),
            })
        };
        let make_header = |name: String,
                           dtype: HeaderDtype,
                           shape: Vec<usize>,
                           width: u64|
         -> Result<SafetensorsTensorHeader> {
            let data_bytes = shape.iter().try_fold(width, |bytes, &dim| {
                bytes
                    .checked_mul(dim as u64)
                    .ok_or_else(|| Error::Msg("quantization metadata byte overflow".into()))
            })?;
            Ok(SafetensorsTensorHeader {
                name,
                dtype,
                shape,
                data_bytes,
            })
        };
        // Resolve the SAME constructor view once, before the native quantizer walk. Aliasing is
        // per suffix, not per projection: source.weight may coexist with target.bias/scales. The
        // original dictionary remains distinct from this module view and effective output metadata.
        let origins = crate::loader::transformer_header_origins(&headers);
        let mut module_headers = headers.clone();
        crate::loader::remap_transformer_header_keys(&mut module_headers);
        for (canonical, embedding) in targets {
            let module_base = crate::loader::TRANSFORMER_KEY_ALIASES
                .iter()
                .find_map(|(from, to)| {
                    (canonical.as_str() == *from || canonical.as_str() == *to)
                        .then(|| (*to).to_owned())
                })
                .unwrap_or_else(|| canonical.clone());
            let weight_key = format!("{module_base}.weight");
            let weight_source = origins.get(&weight_key).ok_or_else(|| {
                Error::Msg(format!("missing quantization source {weight_key}"))
            })?;
            let base = weight_source.strip_suffix(".weight").ok_or_else(|| {
                Error::Msg("native quantization weight has no weight suffix".into())
            })?;
            if !visited.insert(weight_source.clone()) {
                continue;
            }
            linear_header_facts(
                &module_headers,
                &module_base,
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
                Some(bits),
            )?;
            if module_headers.contains_key(&format!("{module_base}.scales")) {
                continue;
            }
            let weight = module_headers.get(&weight_key).cloned().ok_or_else(|| {
                Error::Msg(format!("missing quantization module input {weight_key}"))
            })?;
            let weight_facts = tensor(&weight)?;
            let bias_key = format!("{module_base}.bias");
            let bias = if !*embedding && has_bias(canonical) {
                module_headers.get(&bias_key).cloned()
            } else {
                None
            };
            let bias_facts = bias.as_ref().map(tensor).transpose()?;
            let input = weight.shape[1];
            let output = weight.shape[0];
            let group = mlx_gen::quant::DEFAULT_GROUP_SIZE as usize;
            let cols = input
                .checked_mul(bits as usize)
                .and_then(|width| width.checked_div(32))
                .ok_or_else(|| Error::Msg("quantization packed width overflow".into()))?;
            let scale_dtype = if *embedding {
                weight.dtype
            } else {
                HeaderDtype::BF16
            };
            let scale_width = if *embedding {
                match native_header_dtype(weight.dtype)? {
                    Dtype::Float16 | Dtype::Bfloat16 => 2,
                    Dtype::Float32 => 4,
                    _ => {
                        return Err(Error::Unsupported(
                            "native token embedding quantization requires floating weights".into(),
                        ))
                    }
                }
            } else {
                2
            };
            let packed = make_header(
                format!("{base}.weight"),
                HeaderDtype::U32,
                vec![output, cols],
                4,
            )?;
            let scales = make_header(
                format!("{base}.scales"),
                scale_dtype,
                vec![output, input / group],
                scale_width,
            )?;
            let offsets = make_header(
                format!("{base}.biases"),
                scale_dtype,
                vec![output, input / group],
                scale_width,
            )?;
            let alloc = mlx_gen::ios_quantum::allocated_bytes;
            let mut old = alloc(weight_facts.bytes());
            let mut retained = alloc(packed.data_bytes)
                .saturating_add(alloc(scales.data_bytes))
                .saturating_add(alloc(offsets.data_bytes));
            let casts = if *embedding {
                0
            } else {
                q::linear_quantization_cast_bytes(&weight_facts)
            };
            headers.insert(packed.name.clone(), packed);
            headers.insert(scales.name.clone(), scales);
            headers.insert(offsets.name.clone(), offsets);
            if !*embedding {
                if let (Some(source), Some(facts)) = (&bias, &bias_facts) {
                    if facts.dtype != Dtype::Bfloat16 {
                        old = old.saturating_add(alloc(facts.bytes()));
                        let bias_source = origins.get(&bias_key).ok_or_else(|| {
                            Error::Msg(format!("missing quantization bias origin {bias_key}"))
                        })?;
                        let converted = make_header(
                            bias_source.clone(),
                            HeaderDtype::BF16,
                            source.shape.clone(),
                            2,
                        )?;
                        retained = retained.saturating_add(alloc(converted.data_bytes));
                        headers.insert(converted.name.clone(), converted);
                    }
                }
            }
            let expected_parents = old.checked_add(casts).ok_or_else(|| {
                Error::Unsupported("quantization parent ownership exceeds byte range".into())
            })?;
            let actual_parent_formula =
                q::lazy_quantization_parent_bytes(&weight_facts, bias_facts.as_ref(), *embedding);
            if actual_parent_formula != expected_parents {
                return Err(Error::Msg(
                    "inconsistent quantization parent metadata".into(),
                ));
            }
            records.push((canonical.clone(), old, retained, casts));
        }
        Ok((
            Self {
                files: self.files.clone(),
                headers: headers.into_values().collect(),
            },
            records,
        ))
    }

    /// Effective ControlNet forward metadata only. RAW component headers/read roots remain on
    /// Facts::control and are used separately for the load and each new local streamed replay.
    fn control_load_quantized(&self, bits: Option<i32>) -> Result<Self> {
        match bits {
            Some(bits) => Ok(self
                .load_quantized_with_bias(
                    &control_quantization_targets(),
                    bits,
                    crate::quant::dit_quantization_has_bias,
                )?
                .0),
            None => Ok(self.clone()),
        }
    }

    /// The header-only load floor over the tensors `keep` keeps — the guard's own
    /// [`mlx_gen::ios_quantum::load_floor_bytes`].
    pub(crate) fn floor(&self, keep: impl Fn(&str) -> bool) -> Result<u64> {
        mlx_gen::ios_quantum::load_floor_bytes(&self.files, keep)
    }

    /// The bytes the tensors `keep` keeps occupy once loaded: each tensor's stored bytes as the
    /// allocator holds them (page-rounded above one page, [`mlx_gen::ios_quantum::allocated_bytes`]).
    pub(crate) fn resident(&self, keep: impl Fn(&str) -> bool) -> u64 {
        self.headers
            .iter()
            .filter(|h| keep(&h.name))
            .map(|h| mlx_gen::ios_quantum::allocated_bytes(h.data_bytes))
            .sum()
    }

    /// Exact DiT constructor keys and their winning source-key origins. The actual constructor's
    /// access recorder supplies the keys; no rank/name heuristic invents module membership. Raw
    /// extra/overwritten dictionary roots stay on this Component. Deferred layer keys remain in
    /// this map even when the loader does not materialize them: graph reachability is NOT payload
    /// allocation, an exact descriptor heap charge, or a last-owner/release certificate.
    fn dit_constructed_origins(
        &self,
        accessed: impl IntoIterator<Item = String>,
    ) -> Result<std::collections::BTreeMap<String, String>> {
        let headers = self.headers.iter().map(|h| (h.name.clone(), h.clone()))
            .collect::<std::collections::BTreeMap<_, _>>();
        let origins = crate::loader::transformer_header_origins(&headers);
        accessed.into_iter().map(|key| {
            let source = origins.get(&key).cloned().ok_or_else(|| {
                Error::Msg(format!("missing constructed DiT source origin {key}"))
            })?;
            Ok((key, source))
        }).collect()
    }

    /// Source-logical materialized module payload subtotal only. Constructor correspondence keeps
    /// the deferred graph roots; this selector must not promote their header capacity into N loaded
    /// layer payloads or remove raw dictionary/cache/descriptor obligations from another domain.
    fn dit_materialized_module_bytes(
        &self,
        constructed: &std::collections::BTreeMap<String, String>,
        streamed: bool,
    ) -> u64 {
        let sources = constructed.values().map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        self.resident(|name| {
            sources.contains(name) && crate::loader::dit_load_keeps(streamed, name)
        })
    }

    /// [`Self::resident`] over the tensors under `prefix`.
    fn bytes_under(&self, prefix: &str) -> u64 {
        self.resident(|name| name.starts_with(prefix))
    }

    /// A lazy view of every tensor as a payload-free shape carrier (never evaluated: only shapes and
    /// dtypes are read by the prices).
    fn view(&self) -> Result<Weights> {
        let mut map = std::collections::HashMap::new();
        for h in &self.headers {
            let shape: Vec<i32> = h.shape.iter().map(|&d| d as i32).collect();
            map.insert(h.name.clone(), carrier(&shape, header_dtype(h.dtype)?)?);
        }
        Ok(Weights::from_map(map))
    }
}

fn header_dtype(d: HeaderDtype) -> Result<Dtype> {
    Ok(match d {
        HeaderDtype::BOOL => Dtype::Bool,
        HeaderDtype::U8 => Dtype::Uint8,
        HeaderDtype::I8 => Dtype::Int8,
        HeaderDtype::I16 => Dtype::Int16,
        HeaderDtype::U16 => Dtype::Uint16,
        HeaderDtype::F16 => Dtype::Float16,
        HeaderDtype::BF16 => Dtype::Bfloat16,
        HeaderDtype::I32 => Dtype::Int32,
        HeaderDtype::U32 => Dtype::Uint32,
        HeaderDtype::F32 => Dtype::Float32,
        HeaderDtype::I64 => Dtype::Int64,
        HeaderDtype::U64 => Dtype::Uint64,
        other => {
            return Err(Error::Unsupported(format!(
                "iOS pricing: no MLX dtype for a {other:?} tensor"
            )))
        }
    })
}

/// A never-evaluated array of `shape` and `dtype`: a shape carrier for the price functions.
fn carrier(shape: &[i32], dtype: Dtype) -> Result<Array> {
    Ok(mlx_rs::ops::zeros_dtype(shape, dtype)?)
}

/// The architecture and header facts one request's pricing reads.
#[derive(Clone, Debug)]
pub(crate) struct Facts {
    pub(crate) te_cfg: ZTextEncoderConfig,
    pub(crate) dit_cfg: ZImageTransformerConfig,
    pub(crate) te: Component,
    pub(crate) dit: Component,
    pub(crate) vae: Component,
    /// The Fun-Controlnet-Union checkpoint (the control route only).
    pub(crate) control: Option<Component>,
    /// The tokenizer's padded window (the text encoder's sequence length).
    pub(crate) window_tokens: u32,
    /// The caption's valid tokens (at most the window).
    pub(crate) caption_tokens: u32,
    /// The load's replayed quantization (`LoadSpec::quantize`): packs the dense ControlNet blocks.
    pub(crate) quant_bits: Option<i32>,
    /// A deferred-materialization snapshot load: every stack streams through per-block quanta.
    pub(crate) streamable: bool,
    /// Raw source metadata remains separate from effective packed module metadata. Streamed reads
    /// consume the former; forward kernels consume the latter. No payload is duplicated here.
    pub(crate) load_components: Option<[Component; 3]>,
    /// Matched `(projection, old owners, retained outputs, casts)` for TE / DiT / VAE.
    pub(crate) quant_conversions: [Vec<(String, u64, u64, u64)>; 3],
    pub(crate) te_quant_bits: Option<i32>,
}

/// Which ordinary, completed callback precedes this bounded source replay. These describe
/// logical graph reachability only, not native Data's last owner or a process-charge credit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantizationHistory {
    RequestFirst,
    DenoiseRecurrent,
    NoModelForward,
}

/// Mutable traversal state for a single bounded logical candidate. It never contains native
/// arrays, payload, an observed witness or a physical release assertion.
struct QuantizationCursor {
    seen: [std::collections::BTreeSet<String>; 3],
    dit_embed_entry: Option<[std::collections::BTreeSet<String>; 3]>,
    denoise_started: bool,
    decode_started: bool,
    history: QuantizationHistory,
    late_decode: bool,
}

impl QuantizationCursor {
    fn new(history: QuantizationHistory, late_decode: bool) -> Self {
        Self {
            seen: std::array::from_fn(|_| std::collections::BTreeSet::new()),
            dit_embed_entry: None,
            denoise_started: false,
            decode_started: false,
            history,
            late_decode,
        }
    }

    #[cfg(test)]
    fn window(
        &mut self,
        facts: &Facts,
        phase: IosPhase,
        label: &str,
        point: usize,
        structural: u64,
        staged: bool,
    ) -> Result<Option<(u64, u64, bool)>> {
        self.window_with_vae(facts, phase, label, point, structural, staged, None)
    }

    fn window_with_vae(
        &mut self,
        facts: &Facts,
        phase: IosPhase,
        label: &str,
        point: usize,
        structural: u64,
        staged: bool,
        vae: Option<&VaeOwnerHistory>,
    ) -> Result<Option<(u64, u64, bool)>> {
        // Preparation/control embedding run before the sliced solve. A no-transition slice
        // executes neither model callbacks nor sampler updates. Sigma == 0 ALONE is not this
        // condition: actual solver terminal handling calls denoise/model BEFORE its alias branch.
        if self.history == QuantizationHistory::NoModelForward && denoise_callback_body(label) {
            return Ok(None);
        }
        if phase == IosPhase::Denoise && !self.denoise_started {
            if self.history == QuantizationHistory::DenoiseRecurrent {
                // Later serial outputs already consumed persistent DiT graphs BEFORE preparing
                // their new noise/caption fields. Deferred local block graphs are excluded here;
                // the source operation still charges their cold replay scratch on EVERY callback.
                facts.quantization_complete_forward(IosPhase::Denoise, &mut self.seen)?;
            }
            self.denoise_started = true;
        }
        if label == q::DIT_EMBED {
            if let Some(entry) = &self.dit_embed_entry {
                // F32 and BF16 are alternatives for ONE factory operation, not two callbacks.
                self.seen = entry.clone();
            } else {
                self.dit_embed_entry = Some(self.seen.clone());
            }
        }
        if phase == IosPhase::Decode && !self.decode_started {
            if self.late_decode {
                // A decoder pass never consumes the unused VAE encoder or a skipped DiT.
                facts.quantization_complete_forward(IosPhase::Decode, &mut self.seen)?;
                if let Some(vae) = vae {
                    for conversion in &vae.conversions {
                        if facts.quantization_position(2, &conversion.path)?.0 == IosPhase::Decode {
                            self.seen[2].insert(conversion.path.clone());
                        }
                    }
                }
            }
            self.decode_started = true;
        }
        let (mut resident, mut created, mut pending) = facts.quantization_owner_window_for_components(
            phase, label, if label == q::DIT_EMBED { 0 } else { point },
            structural, staged, &mut self.seen, vae.is_none(),
        )?;
        // No VAE module exists before its load guard. Staging never removes the VAE from heavy
        // encode/denoise/decode; it removes it only while conditioning the text encoder.
        let vae_present = match phase {
            IosPhase::Load => !matches!(label, q::TE_LOAD | q::DIT_LOAD | q::CONTROL_LOAD | q::VAE_LOAD),
            IosPhase::Conditioning => !staged,
            _ => true,
        };
        if let Some(vae) = vae.filter(|_| vae_present) {
            let (entry, outputs, cold) = vae.window(facts, phase, label, point, &mut self.seen[2])?;
            resident = resident.checked_add(entry).ok_or_else(|| Error::Msg("VAE entry owner overflow".into()))?;
            created = created.checked_add(outputs).ok_or_else(|| Error::Msg("VAE created owner overflow".into()))?;
            pending |= cold;
        }
        Ok(Some((resident, created, pending)))
    }
}

impl Facts {
    /// Replay only logical owners over SAME-operation records made by Model::quanta. This never
    /// constructs/evaluates a model or replays requested N. The serial affine family has just its
    /// actual first/last output endpoints; load/conditioning/encode cannot own completed outputs.
    fn replay_quanta(
        &self,
        source: &[Priced],
        route: &Route,
        history: QuantizationHistory,
        completed_outputs: u32,
        late_decode: bool,
    ) -> Result<Vec<Option<IosWindowPrice>>> {
        let last = route.count.checked_sub(1).ok_or_else(|| {
            Error::Unsupported("Z source replay requires a positive serial count".into())
        })?;
        if completed_outputs != 0 && completed_outputs != last {
            return Err(Error::Unsupported(
                "Z source replay requires a serial endpoint".into(),
            ));
        }
        let latent = u64::from(route.width / crate::pipeline::SPATIAL_SCALE)
            .checked_mul(u64::from(route.height / crate::pipeline::SPATIAL_SCALE))
            .and_then(|value| value.checked_mul(crate::pipeline::LATENT_CHANNELS as u64))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| Error::Unsupported("Z source replay latent bytes overflow".into()))?;
        let completed = latent
            .checked_mul(u64::from(completed_outputs))
            .ok_or_else(|| {
                Error::Unsupported("Z source replay completed owners overflow".into())
            })?;
        let mut cursor = QuantizationCursor::new(history, late_decode);
        let mut out = Vec::with_capacity(source.len());
        let mut current_vae = None;
        for operation in source {
            let identity = operation.source_vae.as_ref().map(std::sync::Arc::as_ptr);
            if identity != current_vae {
                cursor = QuantizationCursor::new(history, late_decode);
                current_vae = identity;
            }
            let phase = operation.window.phase;
            let Some((structural, created, pending)) = cursor.window_with_vae(
                self,
                phase,
                operation.label,
                operation.source_point,
                operation.source_structural,
                route.execution.stage_residency,
                operation.source_vae.as_deref(),
            )?
            else {
                out.push(None);
                continue;
            };
            let mut window = operation.window.clone();
            window.resident_bytes = structural;
            window.held_bytes = operation
                .source_held
                .checked_add(created)
                .and_then(|value| {
                    value.checked_add(if phase == IosPhase::Denoise {
                        completed
                    } else {
                        0
                    })
                })
                .ok_or_else(|| Error::Unsupported("Z source replay held owners overflow".into()))?;
            let history_key = match history {
                QuantizationHistory::RequestFirst => "request-first",
                QuantizationHistory::DenoiseRecurrent => "denoise-recurrent",
                QuantizationHistory::NoModelForward => "no-model-forward",
            };
            // Endpoint identity is semantic, never the user's requested count. Counts remain
            // checked coefficients above; equal class strings do not certify physical equivalence.
            let serial_endpoint = if !matches!(phase, IosPhase::Denoise | IosPhase::Decode) {
                "request-prefix"
            } else if route.count == 1 {
                "serial-first-final"
            } else if (phase == IosPhase::Decode && late_decode)
                || (phase == IosPhase::Denoise && completed_outputs != 0)
            {
                "serial-later-final"
            } else {
                "serial-first"
            };
            window.class = Some(format!(
                "z-image/{}/{}/{}/source-op-{}/{history_key}/{serial_endpoint}/decode-late-{late_decode}{}",
                route.mode.key(), phase.as_key(), operation.label, operation.source_point,
                if pending { "/source-first-use" } else { "" },
            ));
            // Local streamed weights are reconstructed afresh at THIS operation; their actual
            // RAW read/packing scratch is unchanged even when persistent module graphs are warm.
            window.scratch_bytes = operation.scratch;
            out.push(Some(window));
        }
        Ok(out)
    }

    fn quantization_loaded(&self, component: usize, path: &str) -> bool {
        match component {
            0 => crate::loader::te_load_keeps(self.streamable, &format!("{path}.weight")),
            1 => crate::loader::dit_load_keeps(self.streamable, &format!("{path}.weight")),
            _ => true,
        }
    }

    /// Advance only conversions actually consumed by a completed forward of this component.
    /// In particular, a text route never consumes the VAE encoder, and a zero-callback solve
    /// never consumes the DiT. Unused resident lazy shells and streamed replay roots are NOT
    /// evaluated by this logical cursor and receive no physical release credit.
    fn quantization_complete_forward(
        &self,
        phase: IosPhase,
        seen: &mut [std::collections::BTreeSet<String>; 3],
    ) -> Result<()> {
        for (component, paths) in self.quant_conversions.iter().enumerate() {
            for (path, _, _, _) in paths {
                if self.quantization_loaded(component, path)
                    && self.quantization_position(component, path)?.0 == phase
                {
                    seen[component].insert(path.clone());
                }
            }
        }
        Ok(())
    }

    /// The production quantum that first consumes a converted parameter. Positions are bounded
    /// factory positions, never requested-step replay. Unknown native targets fail closed.
    fn quantization_position(
        &self,
        component: usize,
        path: &str,
    ) -> Result<(IosPhase, &'static str, usize)> {
        let index = |rest: &str| {
            rest.split('.')
                .next()
                .unwrap_or("")
                .parse::<usize>()
                .map_err(|_| Error::Msg(format!("invalid quantization first-use target {path}")))
        };
        match component {
            0 if path == "model.embed_tokens" => Ok((IosPhase::Conditioning, q::TE_EMBED, 0)),
            0 => {
                let rest = path.strip_prefix("model.layers.").ok_or_else(|| {
                    Error::Unsupported(format!("unmapped encoder quantization target {path}"))
                })?;
                Ok((IosPhase::Conditioning, q::TE_LAYER, index(rest)?))
            }
            1 => {
                let (label, point) =
                    if path.starts_with("all_x_embedder.") || path.starts_with("t_embedder.") {
                        (q::DIT_EMBED, 0)
                    } else if path == "cap_embedder.1" {
                        (q::DIT_CAPTION, 0)
                    } else if path.starts_with("all_final_layer.") {
                        (q::DIT_OUTPUT, 0)
                    } else if let Some(rest) = path.strip_prefix("noise_refiner.") {
                        (q::DIT_BLOCK, index(rest)?)
                    } else if let Some(rest) = path.strip_prefix("context_refiner.") {
                        (q::DIT_CONTEXT_BLOCK, index(rest)?)
                    } else if let Some(rest) = path.strip_prefix("layers.") {
                        (q::DIT_BLOCK, self.dit_cfg.n_refiner_layers + index(rest)?)
                    } else {
                        return Err(Error::Unsupported(format!(
                            "unmapped DiT quantization target {path}"
                        )));
                    };
                Ok((IosPhase::Denoise, label, point))
            }
            2 if path.starts_with("encoder.mid_block.attentions.0.") => {
                Ok((IosPhase::Encode, q::VAE_ENCODE_ATTENTION, 0))
            }
            2 if path.starts_with("decoder.mid_block.attentions.0.") => {
                Ok((IosPhase::Decode, q::VAE_DECODE_ATTENTION, 0))
            }
            _ => Err(Error::Unsupported(format!(
                "unmapped component quantization target {path}"
            ))),
        }
    }

    /// Logical first-use partition of ONE source operation. Replace each same-source effective
    /// payload with its still-live original until that operation consumes it; add current casts and
    /// outputs together. This is not a difference of independent phase maxima or a physical credit.
    /// A completed entry describes only ordinary non-traced/non-retained inference after the
    /// production finish's selected-stream boundary. Other handles/backend Data are separate proof
    /// obligations; no allocator/process cancellation is authorized by this bookkeeping.
    #[cfg(test)]
    fn quantization_owner_window(
        &self,
        phase: IosPhase,
        label: &str,
        point: usize,
        structural: u64,
        staged: bool,
        seen: &mut [std::collections::BTreeSet<String>; 3],
    ) -> Result<(u64, u64, bool)> {
        self.quantization_owner_window_for_components(phase, label, point, structural, staged, seen, true)
    }

    fn quantization_owner_window_for_components(
        &self,
        phase: IosPhase,
        label: &str,
        point: usize,
        structural: u64,
        staged: bool,
        seen: &mut [std::collections::BTreeSet<String>; 3],
        legacy_vae: bool,
    ) -> Result<(u64, u64, bool)> {
        let present = match phase {
            IosPhase::Load => match label {
                q::TE_LOAD => [false, false, false],
                q::DIT_LOAD => [!staged, false, false],
                q::CONTROL_LOAD | q::VAE_LOAD => [!staged, true, false],
                _ => [!staged, true, true],
            },
            IosPhase::Conditioning => [true, !staged, !staged],
            IosPhase::Encode | IosPhase::Denoise => [!staged, true, true],
            IosPhase::Decode => [!staged, !staged, true],
        };
        let add = |a: u64, b: u64| {
            a.checked_add(b)
                .ok_or_else(|| Error::Unsupported("quantization owner window byte overflow".into()))
        };
        let mut removed = 0;
        let mut entry = 0;
        let mut created = 0;
        let mut pending = false;
        for component in 0..if legacy_vae { 3 } else { 2 } {
            if !present[component] {
                continue;
            }
            for (path, old, new, casts) in &self.quant_conversions[component] {
                // Deferred resident shells are lazy descriptors, not already materialized payloads.
                // The real local replay block's RAW read/packing is charged by its counted guard.
                if !self.quantization_loaded(component, path) {
                    continue;
                }
                removed = add(removed, *new)?;
                if seen[component].contains(path) {
                    entry = add(entry, *new)?;
                    continue;
                }
                pending = true;
                entry = add(entry, *old)?;
                let first = self.quantization_position(component, path)?;
                if first.0 == phase && first.1 == label && first.2 == point {
                    created = add(created, add(*new, *casts)?)?;
                    seen[component].insert(path.clone());
                }
            }
        }
        let unchanged = structural.checked_sub(removed).ok_or_else(|| {
            Error::Msg(
                "quantization outputs are not contained in this operation's structural owners"
                    .into(),
            )
        })?;
        Ok((add(unchanged, entry)?, created, pending))
    }

    /// The facts of `spec`'s snapshot at the production geometry.
    pub(crate) fn from_spec(provider: &str, spec: &LoadSpec) -> Result<Self> {
        let WeightsSource::Dir(root) = &spec.weights else {
            return Err(Error::Unsupported(format!(
                "{provider}: iOS pricing reads a snapshot directory"
            )));
        };
        if !spec.adapters.is_empty() {
            return Err(Error::Unsupported(format!(
                "{provider}: iOS pricing does not price LoRA/LoKr adapter residuals"
            )));
        }
        if spec.pid.is_some() {
            return Err(Error::Unsupported(format!(
                "{provider}: a PiD decode overlay is not an iOS load"
            )));
        }
        let stored = mlx_gen::quant::packed_quant_bits(root, "transformer")?;
        let requested = spec.quantize.map(|q| q.bits());
        if stored != requested {
            return Err(Error::Unsupported(format!(
                "{provider}: the snapshot is stored at {stored:?} bits but the load requests \
                 {requested:?}; a load-time requantization is not priced"
            )));
        }
        let control = match (provider == crate::model_control::MODEL_ID, &spec.control) {
            (true, Some(control)) => Some(Component::read(control)?),
            (true, None) => {
                return Err(Error::Unsupported(format!(
                    "{provider}: needs the Fun-Controlnet-Union checkpoint"
                )))
            }
            (false, _) => None,
        };
        let te_cfg = ZTextEncoderConfig::z_image();
        let dit_cfg = ZImageTransformerConfig::turbo();
        let selected_encoder = crate::active_encoder_contract().source_for_load(spec, root)?;
        let te_quant_bits =
            selected_encoder.load_time_quant_bits(requested.or(stored), provider)?;
        let mut raw_te = selected_encoder.read_unchanged(Component::read)?;
        // Preserve ALL pinned direct-shard dictionary headers, including authored but unused keys.
        // The sealed language subset is a separate constructor/quantizer input, not a replacement
        // for that raw inventory. Neither receipt broadens the validated source or shard contract.
        raw_te.headers = selected_encoder.tensor_headers()?;
        let selected_te = Component {
            files: raw_te.files.clone(),
            headers: selected_encoder
                .materialized_language_tensor_headers(&crate::active_encoder_contract())?,
        };
        let raw_dit = Component::read(&WeightsSource::Dir(root.join("transformer")))?;
        let raw_vae = Component::read(&WeightsSource::Dir(root.join("vae")))?;
        let te_targets =
            crate::text_encoder::encoder::quantization_header_targets("model", &te_cfg);
        let dit_targets = crate::transformer::adapter_header_targets(&dit_cfg, true)
            .into_iter()
            .map(|(_, header)| (header, false))
            .collect::<Vec<_>>();
        let convert = |source: &Component,
                       targets: &[(String, bool)],
                       bits: Option<i32>,
                       has_bias: fn(&str) -> bool| {
            if let Some(bits) = bits {
                source.load_quantized_with_bias(targets, bits, has_bias)
            } else {
                Ok((source.clone(), Vec::new()))
            }
        };
        let (te, te_conversions) = convert(&selected_te, &te_targets, te_quant_bits, |_| {
            crate::quant::TEXT_LINEAR_HAS_BIAS
        })?;
        let (dit, dit_conversions) = convert(
            &raw_dit,
            &dit_targets,
            requested,
            crate::quant::dit_quantization_has_bias,
        )?;
        // VAE conversion is AFTER each actual successful remap + constructor history. A packed
        // canonical raw leaf can be overwritten by a dense nested source before construction.
        let vae = raw_vae.clone();
        let vae_conversions = Vec::new();
        Ok(Self {
            te_cfg,
            dit_cfg,
            te,
            dit,
            vae,
            control,
            window_tokens: crate::loader::MAX_LENGTH as u32,
            caption_tokens: crate::loader::MAX_LENGTH as u32,
            quant_bits: requested,
            streamable: spec.load_shape == LoadShape::DeferredMaterialization,
            load_components: Some([raw_te, raw_dit, raw_vae]),
            quant_conversions: [te_conversions, dit_conversions, vae_conversions],
            te_quant_bits,
        })
    }
}

/// The transformer the route runs: the base DiT, or the ControlNet composition over it.
enum Dit {
    Base(Box<ZImageTransformer>),
    Control(Box<ZImageControlTransformer>),
}

impl Dit {
    fn base(&self) -> &ZImageTransformer {
        match self {
            Dit::Base(t) => t,
            Dit::Control(c) => &c.base,
        }
    }
}

/// The ControlNet facts: the composition, its streamed main blocks (with their per-block read) and
/// its resident bytes before and after the load's quantization.
struct ControlFacts {
    /// Main control block `i` and what its quantum reads when streamed.
    main: Vec<(ZImageControlBlock, u64)>,
    /// The load floor (what the load reads up front), and what it holds once read.
    floor: u64,
    loaded: u64,
    /// Each block's (control_layers 0..15, then the refiner) stored bytes.
    block_stored: Vec<u64>,
    /// Each block's resident bytes once packed at the load's bits (= stored when not packed).
    block_packed: Vec<u64>,
    /// Each block's load-time quantization price (`ZImageControlBlock::quantize_bytes`).
    block_quantize: Vec<u64>,
    /// Whether each block is resident after the load reads (the refiner, or an unstreamed main block).
    block_loaded: Vec<bool>,
    /// The non-block tensors (the patch embedder).
    embedder: u64,
}

/// One current dictionary handle, not an independently selectable field alternative. Alias and
/// transpose paths are ordered; an overwritten source is read from the CURRENT map, never raw.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VaeSourceValue {
    raw: String,
    path: Vec<crate::loader::VaeRemapStep>,
    header: SafetensorsTensorHeader,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VaeRemapRead {
    encoder: bool,
    step: crate::loader::VaeRemapStep,
    read: VaeSourceValue,
    overwritten: Option<VaeSourceValue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VaeRemapHistory {
    current: std::collections::BTreeMap<String, VaeSourceValue>,
    reads: Vec<VaeRemapRead>,
}

/// Failed native prefixes are delivered separately, and can never become a refusing tuple inside
/// the successful-world consumer. `current` and `reads` are the exact state BEFORE the failing op.
struct VaeRemapRefusal<'a> {
    prefix: &'a VaeRemapHistory,
    step: &'a crate::loader::VaeRemapStep,
}

/// An explicit local DFS frame replaces a native call frame. `undo` belongs to the operation
/// entering this frame; remaining ORIGINAL snapshot operations are shared and restored on pop.
struct VaeRemapFrame {
    encoder: bool,
    next: usize,
    undo: Option<(usize, crate::loader::VaeRemapStep, Option<VaeSourceValue>)>,
}

enum VaeRemapOutcome {
    Complete(VaeRemapHistory),
    Refused { prefix: VaeRemapHistory, step: crate::loader::VaeRemapStep },
}

/// ONE component's lazy insert/undo iterator. Both phase DFS depths live on the heap, not the
/// native thread stack. Only one next complete history/refusal is yielded; none are precomputed.
struct VaeComponentHistories {
    state: VaeRemapHistory,
    remaining: [Vec<crate::loader::VaeRemapStep>; 2],
    frames: Vec<VaeRemapFrame>,
}

impl VaeComponentHistories {
    fn new(current: std::collections::BTreeMap<String, VaeSourceValue>) -> Self {
        let decoder = current.keys().filter_map(|key| crate::loader::vae_remap_step(key, false)).collect();
        Self {
            state: VaeRemapHistory { current, reads: Vec::new() },
            remaining: [decoder, Vec::new()],
            frames: vec![VaeRemapFrame { encoder: false, next: 0, undo: None }],
        }
    }

    fn backtrack(&mut self) {
        let frame = self.frames.pop().expect("active VAE traversal frame");
        if let Some((index, step, overwritten)) = frame.undo {
            self.state.reads.pop();
            match overwritten {
                Some(old) => { self.state.current.insert(step.target.clone(), old); }
                None => { self.state.current.remove(&step.target); }
            }
            self.remaining[usize::from(frame.encoder)].insert(index, step);
        }
    }
}

impl Iterator for VaeComponentHistories {
    type Item = VaeRemapOutcome;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let frame = self.frames.last_mut()?;
            let encoder = frame.encoder;
            let phase = usize::from(encoder);
            if self.remaining[phase].is_empty() {
                if encoder {
                    let complete = self.state.clone();
                    self.backtrack();
                    return Some(VaeRemapOutcome::Complete(complete));
                }
                if frame.next == 0 {
                    frame.next = 1;
                    // EXACT completed-state snapshot: decoder.encoder.* may have inserted keys.
                    let next = self.state.current.keys()
                        .filter_map(|key| crate::loader::vae_remap_step(key, true)).collect();
                    self.remaining[1] = next;
                    self.frames.push(VaeRemapFrame { encoder: true, next: 0, undo: None });
                } else {
                    self.backtrack();
                }
                continue;
            }
            let index = frame.next;
            if index >= self.remaining[phase].len() {
                self.backtrack();
                continue;
            }
            frame.next += 1;
            let step = self.remaining[phase].remove(index);
            let state = &mut self.state;
            let read = state.current.get(&step.source).cloned();
            let Some(mut value) = read.clone().filter(|v| !step.transpose || v.header.shape.len() == 4) else {
                let prefix = state.clone();
                self.remaining[phase].insert(index, step.clone());
                return Some(VaeRemapOutcome::Refused { prefix, step });
            };
            if step.transpose {
                let s = &value.header.shape;
                value.header.shape = vec![s[0], s[2], s[3], s[1]];
            }
            value.header.name = step.target.clone();
            value.path.push(step.clone());
            let overwritten = state.current.insert(step.target.clone(), value);
            state.reads.push(VaeRemapRead {
                encoder, step: step.clone(), read: read.expect("checked require"),
                overwritten: overwritten.clone(),
            });
            self.frames.push(VaeRemapFrame { encoder, next: 0, undo: Some((index, step, overwritten)) });
        }
    }
}

/// Disjoint read/write supports commute: each op reads one key and inserts one key, so two
/// supports with no common key cannot affect each other's require, transpose, insert or failure.
/// The partition includes decoder-generated encoder keys AND their encoder targets. Its possible
/// key domain is used ONLY for connectivity; actual encoder operations are snapshotted anew from
/// EACH completed decoder state below. No initial-key union is executed as an encoder snapshot.
struct VaeSourceInventory {
    raw: Component,
    supports: Vec<std::collections::BTreeSet<String>>,
    seeds: Vec<VaeRemapHistory>,
}

impl VaeSourceInventory {
    fn new(source: &Component) -> Result<Self> {
        let mut domain = source.headers.iter().map(|h| h.name.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let decoder = domain.iter().filter_map(|k| crate::loader::vae_remap_step(k, false))
            .collect::<Vec<_>>();
        domain.extend(decoder.iter().map(|op| op.target.clone()));
        let encoder = domain.iter().filter_map(|k| crate::loader::vae_remap_step(k, true))
            .collect::<Vec<_>>();
        let mut supports = domain.into_iter().map(|k| std::collections::BTreeSet::from([k]))
            .collect::<Vec<_>>();
        for op in decoder.iter().chain(&encoder) {
            let mut joined = std::collections::BTreeSet::from([op.source.clone(), op.target.clone()]);
            let mut i = 0;
            while i < supports.len() {
                if !supports[i].is_disjoint(&joined) {
                    joined.extend(supports.remove(i));
                    // `joined` grew; recheck earlier supports (including generated targets).
                    i = 0;
                } else {
                    i += 1;
                }
            }
            supports.push(joined);
        }
        let mut inventory = Self { raw: source.clone(), supports, seeds: Vec::new() };
        for index in 0..inventory.supports.len() {
            let mut seed = None;
            let mut first_refusal = None;
            inventory.visit_component(index, &mut |world| {
                seed = Some(world.clone());
                Ok(true)
            }, &mut |failure| {
                first_refusal.get_or_insert_with(|| format!(
                    "VAE native remap refused {} after {} reads",
                    failure.step.source, failure.prefix.reads.len(),
                ));
                Ok(())
            })?;
            inventory.seeds.push(seed.ok_or_else(|| {
                Error::Msg(first_refusal.unwrap_or_else(|| "VAE remap has no successful order".into()))
            })?);
        }
        Ok(inventory)
    }

    fn component_histories(&self, index: usize) -> VaeComponentHistories {
        let support = &self.supports[index];
        let current = self.raw.headers.iter().filter(|h| support.contains(&h.name)).map(|h| {
            (h.name.clone(), VaeSourceValue { raw: h.name.clone(), path: Vec::new(), header: h.clone() })
        }).collect();
        VaeComponentHistories::new(current)
    }

    /// DFS remains confined to ONE collision component, but its explicit heap frames make native
    /// call depth independent of input operation count. Preserve complete effects and refusal
    /// prefixes without equivalence merging, a depth/alias cap, or an eager list of histories.
    fn visit_component(
        &self,
        index: usize,
        complete: &mut dyn FnMut(&VaeRemapHistory) -> Result<bool>,
        refused: &mut dyn FnMut(VaeRemapRefusal<'_>) -> Result<()>,
    ) -> Result<bool> {
        for outcome in self.component_histories(index) {
            match outcome {
                VaeRemapOutcome::Complete(world) => {
                    if complete(&world)? { return Ok(true); }
                }
                VaeRemapOutcome::Refused { prefix, step } => {
                    refused(VaeRemapRefusal { prefix: &prefix, step: &step })?;
                }
            }
        }
        Ok(false)
    }

    /// A lazy joint consumer over demanded components only. Other complete independent histories
    /// remain factored, not globally permuted. Seeds are actual successful effects, never one-field
    /// maxima. No Cartesian list of worlds is allocated, and no fields from two worlds are mixed.
    fn visit_selected(
        &self,
        keys: &std::collections::BTreeSet<String>,
        consumer: &mut dyn FnMut(&VaeRemapHistory) -> Result<()>,
    ) -> Result<()> {
        let selected = self.supports.iter().enumerate().filter_map(|(i, support)| {
            (!support.is_disjoint(keys)).then_some(i)
        }).collect::<Vec<_>>();
        // Lazy odometer: one iterator and one chosen COMPLETE effect per demanded component.
        // Independent component count does not nest visit_component/consumer native call frames.
        // Replacing a chosen effect is the joint-frame undo; no global world product is stored.
        let mut worlds = self.seeds.clone();
        let mut frames = (0..selected.len()).map(|_| None).collect::<Vec<Option<VaeComponentHistories>>>();
        let mut depth = 0;
        loop {
            if depth == selected.len() {
                let mut reads = worlds.iter().enumerate().filter(|(i, _)| !selected.contains(i))
                    .flat_map(|(_, world)| world.reads.clone()).collect::<Vec<_>>();
                for &index in &selected { reads.extend(worlds[index].reads.clone()); }
                // SAME canonical interleaving as before: unselected then selected local effects,
                // stable by phase. This is not a captured native HashMap-order claim.
                reads.sort_by_key(|read| read.encoder);
                consumer(&VaeRemapHistory {
                    current: worlds.iter().flat_map(|world| world.current.clone()).collect(),
                    reads,
                })?;
                if depth == 0 { break; }
                depth -= 1;
                continue;
            }
            let index = selected[depth];
            let iterator = frames[depth].get_or_insert_with(|| self.component_histories(index));
            let next = iterator.find_map(|outcome| match outcome {
                VaeRemapOutcome::Complete(world) => Some(world),
                VaeRemapOutcome::Refused { .. } => None,
            });
            if let Some(world) = next {
                worlds[index] = world;
                depth += 1;
                if depth < selected.len() { frames[depth] = None; }
            } else {
                frames[depth] = None;
                if depth == 0 { break; }
                depth -= 1;
            }
        }
        Ok(())
    }
}

impl VaeSourceInventory {
    fn component(world: &VaeRemapHistory) -> Component {
        Component { files: Vec::new(), headers: world.current.values().map(|v| v.header.clone()).collect() }
    }

    /// Discover dependencies with the SAME constructor, including reads preceding a genuine failed
    /// packed selector. Key presence is invariant under these insert-only remaps; value-dependent
    /// failures do not erase the dependencies that could make a different complete history succeed.
    /// Repeat only when the demanded support grows, then consume complete joint tuples. Unrelated
    /// dictionary components remain separate source inventory, not a factorial global product.
    fn visit_constructed(
        &self,
        bits: Option<i32>,
        consumer: &mut dyn FnMut(&Vae, std::sync::Arc<VaeOwnerHistory>) -> Result<()>,
    ) -> Result<()> {
        let mut demanded = std::collections::BTreeSet::new();
        loop {
            let mut accessed = demanded.clone();
            self.visit_selected(&demanded, &mut |world| {
                let view = Self::component(world).view()?;
                let _attempt = crate::loader::build_remapped_vae(&view);
                accessed.extend(view.accessed_entries().into_iter().map(|(key, _)| key));
                Ok(())
            })?;
            if accessed == demanded { break; }
            demanded = accessed;
        }
        let targets = ["", "encoder."].into_iter().flat_map(|prefix| {
            crate::vae::attention::QUANTIZATION_LINEAR_PATHS.into_iter()
                .map(move |leaf| (format!("{prefix}mid_block.attentions.0.{leaf}"), false))
        }).collect::<Vec<_>>();
        let mut accepted = 0usize;
        let mut first_refusal = None;
        self.visit_selected(&demanded, &mut |world| {
            let source = Self::component(world);
            // A new recorder whose arrays are MOVED at the remap/ctor boundary; remapper accesses
            // are lineage above, not constructor membership. These are metadata handles only.
            let view = source.view()?;
            let view = Weights::from_map(view.into_tensors());
            let raw = crate::loader::build_remapped_vae(&view);
            let constructed = view.accessed_entries().into_iter().map(|(key, _)| key).collect();
            let attempt = (|| {
                let raw = raw?;
                let (effective, conversions) = if let Some(bits) = bits {
                    source.load_quantized_with_bias(&targets, bits, |_| true)?
                } else {
                    (source, Vec::new())
                };
                let owners = std::sync::Arc::new(VaeOwnerHistory::from_constructed(world, &constructed, &conversions)?);
                // The SAME closed native selector sees the WHOLE effective tuple, never separate
                // maxima for weight/bias/scales/offset. Conversion follows successful raw discovery.
                let effective_vae = if conversions.is_empty() { raw } else {
                    crate::loader::build_remapped_vae(&effective.view()?)?
                };
                Ok((effective_vae, owners))
            })();
            match attempt {
                Ok((vae, owners)) => {
                    accepted += 1;
                    consumer(&vae, owners)?;
                }
                Err(error) => { if first_refusal.is_none() { first_refusal = Some(error); } }
            }
            Ok(())
        })?;
        if accepted == 0 {
            return Err(first_refusal.unwrap_or_else(|| Error::Msg("VAE has no successful constructor history".into())));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VaeConversionOwners {
    path: String,
    parents: std::collections::BTreeMap<String, u64>,
    outputs: u64,
    casts: u64,
}

/// ONE successful remap/constructor/conversion history. Unconverted fields and pending parents
/// form a union by raw source identity; converted outputs are distinct graph nodes. A transpose
/// retains its ordered view ancestry in `remap`, not an invented new raw backing allocation.
/// These remain logical module owners, NOT Data/cache/foreign-graph or physical release evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VaeOwnerHistory {
    remap: VaeRemapHistory,
    unchanged: std::collections::BTreeMap<String, u64>,
    conversions: Vec<VaeConversionOwners>,
}

impl VaeOwnerHistory {
    fn from_constructed(
        remap: &VaeRemapHistory,
        constructed: &std::collections::BTreeSet<String>,
        conversions: &[(String, u64, u64, u64)],
    ) -> Result<Self> {
        let mut converted = std::collections::BTreeSet::new();
        let mut owners = Vec::new();
        for (base, _, outputs, casts) in conversions {
            let mut parents = std::collections::BTreeMap::new();
            for suffix in ["weight", "bias"] {
                let key = format!("{base}.{suffix}");
                if let Some(value) = remap.current.get(&key) {
                    if suffix == "weight" || value.header.dtype != HeaderDtype::BF16 {
                        converted.insert(key);
                        parents.insert(value.raw.clone(), mlx_gen::ios_quantum::allocated_bytes(value.header.data_bytes));
                    }
                }
            }
            owners.push(VaeConversionOwners {
                path: if base.starts_with("encoder.") { base.clone() } else { format!("decoder.{base}") },
                parents, outputs: *outputs, casts: *casts,
            });
        }
        let mut unchanged = std::collections::BTreeMap::new();
        for key in constructed.difference(&converted) {
            let value = remap.current.get(key).ok_or_else(|| Error::Msg(format!("missing constructed VAE source {key}")))?;
            unchanged.insert(value.raw.clone(), mlx_gen::ios_quantum::allocated_bytes(value.header.data_bytes));
        }
        Ok(Self { remap: remap.clone(), unchanged, conversions: owners })
    }

    fn window(
        &self,
        facts: &Facts,
        phase: IosPhase,
        label: &str,
        point: usize,
        seen: &mut std::collections::BTreeSet<String>,
    ) -> Result<(u64, u64, bool)> {
        let mut entry = self.unchanged.clone();
        let mut outputs = 0u64;
        let mut created = 0u64;
        let mut pending = false;
        let add = |a: u64, b: u64| a.checked_add(b).ok_or_else(|| Error::Msg("VAE source owners overflow".into()));
        for conversion in &self.conversions {
            if seen.contains(&conversion.path) {
                outputs = add(outputs, conversion.outputs)?;
            } else {
                pending = true;
                entry.extend(conversion.parents.clone());
                if facts.quantization_position(2, &conversion.path)? == (phase, label, point) {
                    created = add(created, add(conversion.outputs, conversion.casts)?)?;
                    seen.insert(conversion.path.clone());
                }
            }
        }
        let resident = entry.values().try_fold(outputs, |sum, bytes| add(sum, *bytes))?;
        Ok((resident, created, pending))
    }
}

/// The modules and byte totals one load's pricing reads, built once from its [`Facts`].
pub(crate) struct Model {
    facts: Facts,
    te_embedding: mlx_gen::nn::TokenEmbedding,
    /// Actual initial constructor keys: all configured layers ordinarily, embedding ONLY streamed.
    /// Future local-i metadata below is not a retained layer graph or materialized payload.
    te_constructor_keys: std::collections::BTreeSet<String>,
    /// Forward metadata for every executed layer and its actual local-i raw read/packing price.
    te_layers: Vec<(EncoderLayer, u64)>,
    dit: Dit,
    /// Each unified block and what its quantum reads when streamed.
    dit_blocks: Vec<(ZImageTransformerBlock, u64)>,
    control: Option<ControlFacts>,
    te_floor: u64,
    dit_floor: u64,
    vae_floor: u64,
    /// Logical constructor output graph inventory, separate from raw/remapper dictionary reads.
    vae_sources: VaeSourceInventory,
    /// Constructor-key -> actual source-key correspondence, including ALL retained deferred main
    /// layer shells. This nonnumeric graph inventory is separate from the materialized-module
    /// payload subtotal below and from the full raw dictionary/load floor on Facts. Descriptor heap,
    /// retained quantizer graphs, native caches/backing and physical charge are NOT assigned zero
    /// by excluding unevaluated shells from that payload subtotal; those obligations remain open.
    dit_constructor_origins: std::collections::BTreeMap<String, String>,
}

impl Model {
    pub(crate) fn new(facts: Facts) -> Result<Self> {
        let streamable = facts.streamable;
        let te_cfg = facts.te_cfg;
        let initial_te_view = facts.te.view()?;
        // SAME selector as both production constructors. A streamed encoder has NO layer shells;
        // do not manufacture a full TextEncoder merely to price its embedding quantum.
        let te_embedding = crate::quant::embedding(&initial_te_view, "model.embed_tokens")?;
        if !streamable {
            let _ordinary_constructor =
                TextEncoder::from_weights(&initial_te_view, "model", &te_cfg)?;
        }
        let te_constructor_keys = initial_te_view.accessed_entries().into_iter()
            .map(|(key, _)| key).collect::<std::collections::BTreeSet<_>>();
        let raw_te = facts
            .load_components
            .as_ref()
            .map_or(&facts.te, |raw| &raw[0]);
        let raw_dit = facts
            .load_components
            .as_ref()
            .map_or(&facts.dit, |raw| &raw[1]);
        let mut te_view = facts.te.view()?;
        let mut te_load_view = raw_te.view()?;
        let te_layers = (0..te_cfg.n_layers)
            .map(|i| {
                let prefix = format!("model.layers.{i}");
                let (layer, _) = crate::text_encoder::stream::layer_counted(
                    &mut te_view,
                    &prefix,
                    &te_cfg,
                    None,
                )?;
                let (_, read) = crate::text_encoder::stream::layer_counted(
                    &mut te_load_view,
                    &prefix,
                    &te_cfg,
                    facts.te_quant_bits,
                )?;
                Ok((layer, read))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut dit_view = facts.dit.view()?;
        crate::loader::remap_transformer_keys(&mut dit_view);
        let base = ZImageTransformer::from_weights(&dit_view, "", facts.dit_cfg.clone())?;
        let dit_constructor_origins = facts.dit.dit_constructed_origins(
            dit_view.accessed_entries().into_iter().map(|(key, _)| key),
        )?;
        let mut block_view = facts.dit.view()?;
        let mut block_load_view = raw_dit.view()?;
        let dit_blocks = (0..facts.dit_cfg.n_layers)
            .map(|i| {
                let prefix = format!("layers.{i}");
                let (block, _) = crate::block_stream::base_block_counted(
                    &mut block_view,
                    &prefix,
                    facts.dit_cfg.block_cfg(),
                    None,
                )?;
                let (_, read) = crate::block_stream::base_block_counted(
                    &mut block_load_view,
                    &prefix,
                    facts.dit_cfg.block_cfg(),
                    facts.quant_bits,
                )?;
                Ok((block, read))
            })
            .collect::<Result<Vec<_>>>()?;

        let raw_vae = facts.load_components.as_ref().map_or(&facts.vae, |raw| &raw[2]);
        let vae_sources = VaeSourceInventory::new(raw_vae)?;

        let (dit, control) =
            match &facts.control {
                None => (Dit::Base(Box::new(base)), None),
                Some(component) => {
                    // Resident ControlNet blocks are packed AND materialized in their actual iOS load
                    // quanta; deferred main blocks pack a NEW local graph at every replay. Forward
                    // modules must therefore use EFFECTIVE metadata, while loads/counts retain RAW.
                    let effective = component.control_load_quantized(facts.quant_bits)?;
                    let composed =
                        ZImageControlTransformer::from_weights(base, &effective.view()?, "")?;
                    let mut forward_view = effective.view()?;
                    let mut read_view = component.view()?;
                    let main = (0..crate::control_transformer::CONTROL_LAYERS_PLACES.len())
                        .map(|i| {
                            let prefix = format!("control_layers.{i}");
                            let (block, _) = crate::block_stream::control_block_counted(
                                &mut forward_view,
                                &prefix,
                                facts.dit_cfg.block_cfg(),
                                i == 0,
                                None,
                            )?;
                            let (_, read) = crate::block_stream::control_block_counted(
                                &mut read_view,
                                &prefix,
                                facts.dit_cfg.block_cfg(),
                                i == 0,
                                facts.quant_bits,
                            )?;
                            Ok((block, read))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    // (prefix, initially loaded, has_before_proj), in the actual main/refiner order.
                    let blocks: Vec<(String, bool, bool)> =
                        composed
                            .control_layers
                            .iter()
                            .enumerate()
                            .map(|(i, _)| (format!("control_layers.{i}."), !streamable, i == 0))
                            .chain(composed.control_noise_refiner().iter().enumerate().map(
                                |(i, _)| (format!("control_noise_refiner.{i}."), true, i == 0),
                            ))
                            .collect();
                    let block_stored: Vec<u64> = blocks
                        .iter()
                        .map(|(prefix, _, _)| component.bytes_under(prefix))
                        .collect();
                    // This includes the ACTUAL converted dense bias, not its old F32 source width.
                    let block_packed = blocks
                        .iter()
                        .map(|(prefix, _, _)| effective.bytes_under(prefix))
                        .collect();
                    let raw_quant_view = component.view()?;
                    let block_quantize = blocks
                        .iter()
                        .map(|(prefix, _, before)| {
                            if let Some(bits) = facts.quant_bits {
                                let raw = ZImageControlBlock::from_weights(
                                    &raw_quant_view,
                                    prefix.trim_end_matches('.'),
                                    facts.dit_cfg.block_cfg(),
                                    *before,
                                )?;
                                Ok(raw.quantize_bytes(bits))
                            } else {
                                Ok(0)
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let block_loaded = blocks.iter().map(|(_, loaded, _)| *loaded).collect();
                    let keep = |name: &str| crate::loader::control_load_keeps(streamable, name);
                    let control = ControlFacts {
                        main,
                        floor: component.floor(keep)?,
                        loaded: component.resident(keep),
                        embedder: effective.bytes_under("control_all_x_embedder."),
                        block_stored,
                        block_packed,
                        block_quantize,
                        block_loaded,
                    };
                    (Dit::Control(Box::new(composed)), Some(control))
                }
            };
        Ok(Self {
            te_constructor_keys,
            dit_constructor_origins,
            vae_sources,
            te_floor: facts
                .te
                .floor(|n| crate::loader::te_load_keeps(streamable, n))?,
            dit_floor: facts
                .dit
                .floor(|n| crate::loader::dit_load_keeps(streamable, n))?,
            vae_floor: facts.vae.floor(|_| true)?,
            facts,
            te_embedding,
            te_layers,
            dit,
            dit_blocks,
            control,
        })
    }

    /// The source-logical payload subtotal of modules materialized by this loader, not all file
    /// headers and not unevaluated deferred-shell capacity. The full constructor map still retains
    /// the shell roots; raw extra/overwritten keys remain on Facts, and local streamed-i read/pack
    /// payload is counted independently. This is NOT a native/physical lower bound or release credit.
    pub(crate) fn dit_resident(&self) -> u64 {
        self.facts.dit.dit_materialized_module_bytes(
            &self.dit_constructor_origins,
            self.facts.streamable,
        ) + self.dit.base().rope.resident_bytes()
    }

    #[cfg(test)]
    pub(crate) fn dit_constructor_origins(&self) -> &std::collections::BTreeMap<String, String> {
        &self.dit_constructor_origins
    }

    /// Inspect the representations the pricing factory ACTUALLY uses for forward quanta. Test
    /// metadata only: neither this nor TensorFacts/strides is a native backing/release witness.
    #[cfg(test)]
    pub(crate) fn control_forward_is_packed(&self) -> Option<(bool, bool)> {
        let Dit::Control(control) = &self.dit else {
            return None;
        };
        let packed = |block: &ZImageControlBlock| {
            block
                .projections()
                .iter()
                .all(|linear| linear.is_quantized())
        };
        let main = if self.facts.streamable {
            self.control
                .as_ref()?
                .main
                .iter()
                .all(|(block, _)| packed(block))
        } else {
            control.control_layers.iter().all(packed)
        };
        Some((main, control.control_noise_refiner().iter().all(packed)))
    }

    /// The ControlNet resident once loaded (and packed at the load's bits): the patch embedder plus
    /// every block the load leaves resident (the refiner; the main blocks unless a deferred load
    /// streams them — `ZImageControlTransformer::quantize` packs only resident blocks).
    pub(crate) fn control_resident(&self) -> u64 {
        self.control.as_ref().map_or(0, |c| {
            c.embedder
                + (0..c.block_stored.len())
                    .filter(|&i| c.block_loaded[i])
                    .map(|i| c.block_packed[i])
                    .sum::<u64>()
        })
    }

    /// Construct each source operation once. Serial denoise records start at output zero;
    /// bounded history/serial candidates replay only their scalar ownership records, not models.
    #[cfg(test)]
    pub(crate) fn quanta(&self, route: &Route, backend: Backend) -> Result<Vec<Priced>> {
        let mut out = Vec::new();
        self.visit_quanta(route, backend, &mut |world| { out.extend_from_slice(world); Ok(()) })?;
        Ok(out)
    }

    fn visit_quanta(
        &self,
        route: &Route,
        backend: Backend,
        consumer: &mut dyn FnMut(&[Priced]) -> Result<()>,
    ) -> Result<()> {
        self.vae_sources.visit_constructed(self.facts.quant_bits, &mut |vae, owners| {
            consumer(&self.quanta_for_vae(route, backend, vae, owners)?)
        })
    }

    fn quanta_for_vae(
        &self,
        route: &Route,
        backend: Backend,
        vae: &Vae,
        vae_owners: std::sync::Arc<VaeOwnerHistory>,
    ) -> Result<Vec<Priced>> {
        let history = QuantizationHistory::RequestFirst;
        let late_decode = false;
        let f = &self.facts;
        if route.mode.control() != self.control.is_some() {
            return Err(Error::Unsupported(format!(
                "iOS pricing: mode {:?} does not match the loaded route",
                route.mode
            )));
        }
        let staged = route.execution.stage_residency;
        let streamable = f.streamable;
        // The pipeline's own validation of the rung-4 window against the load shape.
        crate::pipeline::resolve_block_window_for(
            Some(route.execution),
            streamable,
            "z_image_turbo",
        )?;

        let te_res = f.te.resident(|name| self.te_constructor_keys.contains(name));
        let dit_res = self.dit_resident();
        // This baseline EXCLUDES VAE fields by component identity. The cursor adds their complete
        // same-history owner union; no effective payload upper is later subtracted as if exact.
        let ctrl_res = self.control_resident();
        let heavy = dit_res + ctrl_res;
        let te_cfg = &f.te_cfg;
        let s = f.window_tokens as i32;
        let n_valid = f.caption_tokens.min(f.window_tokens) as i32;
        let hidden = te_cfg.hidden_size;
        let mode = route.mode;
        let cap_dtype = if mode.img2img() {
            Dtype::Float32
        } else {
            Dtype::Bfloat16
        };
        let cap_bytes = q::ab(
            mlx_gen::ios_quantum::dtype_bytes(cap_dtype),
            &[n_valid.max(0) as u64, hidden as u64],
        );
        let f32b = |dims: &[i32]| q::ab(4, &dims.iter().map(|&d| q::u(d)).collect::<Vec<_>>());

        let latent_upper = 4
            * u64::from(route.width / crate::pipeline::SPATIAL_SCALE)
            * u64::from(route.height / crate::pipeline::SPATIAL_SCALE)
            * crate::pipeline::LATENT_CHANNELS as u64;
        // Conditioning/encode precede the serial loop; the first output owns no completed finals.
        // Later-output endpoint owners are added only by Facts::replay_quanta in Denoise.
        let completed = 0_u64;
        let init_owner = if mode.img2img() { latent_upper } else { 0 };
        let context_owner = if mode.control() {
            (2 * crate::pipeline::LATENT_CHANNELS as u64 + 1) * latent_upper
                / crate::pipeline::LATENT_CHANNELS as u64
        } else {
            0
        };
        let image_tokens = u64::from(route.width / crate::pipeline::SPATIAL_SCALE)
            * u64::from(route.height / crate::pipeline::SPATIAL_SCALE)
            / (f.dit_cfg.patch_size.max(1) as u64).pow(2);
        let padded_image_tokens = image_tokens.div_ceil(32) * 32;
        let control_embedding = if mode.control() {
            4 * padded_image_tokens * f.dit_cfg.dim as u64
        } else {
            0
        };
        // Correspondence only, not native evidence: the winning envelope window names the SAME
        // complete raw/remap/constructor/conversion history used by its forward and replay.
        use sha2::Digest;
        let vae_owner_domain = format!("z-image/source-logical-owners-v1/vae-history-{:x}",
            sha2::Sha256::digest(format!("{vae_owners:?}").as_bytes()));
        let mut out = Vec::new();
        let mut points = std::collections::BTreeMap::<&str, usize>::new();
        let operation_extent = std::cell::Cell::new((route.height, route.width));
        let mut quant_cursor = QuantizationCursor::new(history, late_decode);
        let mut push = |label: &'static str, resident: u64, scratch: u64| -> Result<()> {
            let phase = ios_quantum_phase(label).expect("source-owned quantum label");
            let structural = match phase {
                IosPhase::Load => match label {
                    q::TE_LOAD => 0,
                    q::DIT_LOAD => {
                        if staged {
                            0
                        } else {
                            te_res
                        }
                    }
                    _ => resident.saturating_sub(if staged { cap_bytes } else { 0 }),
                },
                IosPhase::Conditioning => te_res + if staged { 0 } else { heavy },
                IosPhase::Encode | IosPhase::Denoise => heavy + if staged { 0 } else { te_res },
                IosPhase::Decode => {
                    if staged { 0 } else { te_res + dit_res + ctrl_res }
                }
            };
            // `resident` is the SAME operation's source sum; this split does not subtract two
            // independently maximized prices and supplies no physical lower/cancellation bound.
            let held = match label {
                // These guards precede construction of noise/prepared fields/control embedding.
                // Do not carry the later callback's cached inputs backwards over their creation.
                q::LATENTS => cap_bytes + completed + context_owner,
                q::DIT_PREPARE => cap_bytes + completed + context_owner + init_owner + latent_upper,
                q::CONTROL_EMBED => resident
                    .saturating_sub(structural)
                    .saturating_sub(control_embedding)
                    .saturating_sub(context_owner),
                _ => resident.saturating_sub(structural),
            };
            let point = points.entry(label).or_default();
            let source_structural = structural;
            let source_point = *point;
            let Some((structural, quant_created, quant_pending)) =
                quant_cursor.window_with_vae(f, phase, label, *point, structural, staged, Some(&vae_owners))?
            else {
                return Ok(());
            };
            let basis = IosWorkBasis {
                forward_rows: if phase == IosPhase::Load { 0 } else { 1 },
                token_count: if phase == IosPhase::Conditioning {
                    f.window_tokens as u64
                } else if phase == IosPhase::Denoise {
                    u64::from(route.width / crate::pipeline::SPATIAL_SCALE)
                        * u64::from(route.height / crate::pipeline::SPATIAL_SCALE)
                        / (f.dit_cfg.patch_size.max(1) as u64).pow(2)
                } else {
                    0
                },
                model_width: operation_extent.get().1,
                model_height: operation_extent.get().0,
                processed_plane_bytes: if phase == IosPhase::Encode {
                    12 * u64::from(route.width) * u64::from(route.height)
                } else {
                    0
                },
                output_plane_bytes: 3 * u64::from(route.width) * u64::from(route.height),
                patch_width: if phase == IosPhase::Denoise {
                    f.dit_cfg.patch_size as u32
                } else {
                    0
                },
                patch_height: if phase == IosPhase::Denoise {
                    f.dit_cfg.patch_size as u32
                } else {
                    0
                },
                ..Default::default()
            };
            let mut window = IosWindowPrice {
                phase,
                class: Some(format!(
                    "z-image/{}/{}/{label}/source-op-{}",
                    mode.key(),
                    phase.as_key(),
                    *point
                )),
                label: label.into(),
                resident_bytes: structural,
                held_bytes: held.checked_add(quant_created).ok_or_else(|| {
                    Error::Unsupported("quantization held-owner byte overflow".into())
                })?,
                scratch_bytes: scratch,
                basis: Some(basis),
                owner_domain: vae_owner_domain.clone(),
            };
            if quant_pending {
                window.class = window
                    .class
                    .map(|class| format!("{class}/source-first-use"));
            }
            if phase == IosPhase::Denoise || phase == IosPhase::Decode {
                let history_key = match history {
                    QuantizationHistory::RequestFirst => "request-first",
                    QuantizationHistory::DenoiseRecurrent => "denoise-recurrent",
                    QuantizationHistory::NoModelForward => "no-model-forward",
                };
                window.class = window
                    .class
                    .map(|class| format!("{class}/{history_key}/decode-late-{late_decode}"));
            }
            *point += 1;
            out.push(Priced {
                label,
                resident,
                scratch,
                window,
                source_structural,
                source_held: held,
                source_point,
                source_vae: Some(vae_owners.clone()),
            });
            Ok(())
        };

        // ---- Load: each component floor, with whatever is resident when it loads. ----
        push(q::TE_LOAD, 0, self.te_floor)?;
        let at_heavy = if staged { cap_bytes } else { te_res };
        push(q::DIT_LOAD, at_heavy, self.dit_floor)?;
        let mut loaded = at_heavy + dit_res;
        if let Some(c) = &self.control {
            push(q::CONTROL_LOAD, loaded, c.floor)?;
            loaded += c.loaded;
        }
        push(q::VAE_LOAD, loaded, self.vae_floor)?;
        if let (Some(c), Some(_)) = (&self.control, f.quant_bits) {
            // `quantize_quantum` over each resident block (the main blocks unless streamed, then
            // the refiner): dense while it packs, packed after.
            for i in (0..c.block_stored.len()).filter(|&i| c.block_loaded[i]) {
                push(q::CONTROL_QUANTIZE, loaded, c.block_quantize[i])?;
                loaded = loaded - c.block_stored[i] + c.block_packed[i];
            }
        }

        // ---- Conditioning ----
        let cond_res = te_res + if staged { 0 } else { heavy };
        push(
            q::TE_EMBED,
            cond_res,
            TextEncoder::embedding_bytes(
                &self.te_embedding,
                (te_cfg.hidden_size, te_cfg.head_dim),
                1,
                s,
            ),
        )?;
        // Constructor inventory, executed layer count, and returned [-2] state are separate facts:
        // BOTH ordinary and streamed loops execute/finish every configured layer. Only the first
        // loop entry aliases prev/second_to_last; later entries carry two distinct hidden results.
        for (index, (layer, read)) in self.te_layers.iter().enumerate() {
            let te_carry = crate::text_encoder::encoder::layer_carry_bytes(
                index,
                1,
                s,
                hidden,
                te_cfg.head_dim,
            );
            let scratch = layer
                .quantum_scratch(1, s, 4, backend)
                .saturating_add(if streamable { *read } else { 0 });
            push(q::TE_LAYER, cond_res + te_carry, scratch)?;
        }
        push(
            q::CAPTION_FEATURES,
            cond_res + f32b(&[1, s, hidden]),
            crate::pipeline::caption_features_bytes(n_valid, hidden),
        )?;
        if !mode.img2img() {
            push(
                q::CAPTION_FEATURES,
                cond_res + f32b(&[n_valid, hidden]),
                crate::pipeline::caption_cast_bytes((n_valid as u64) * hidden as u64),
            )?;
        }

        // ---- Encode (inside the denoise phase's components) ----
        let (w, h) = (route.width as i32, route.height as i32);
        let spatial = crate::pipeline::SPATIAL_SCALE as i32;
        let (lh, lw) = (h / spatial, w / spatial);
        let lc = crate::pipeline::LATENT_CHANNELS;
        let latent = f32b(&[lc, 1, lh, lw]);
        let context = f32b(&[2 * lc + 1, 1, lh, lw]);
        let render_res = heavy + if staged { 0 } else { te_res } + cap_bytes;
        // The control image encodes first; a reference init second (the context then resident).
        let mut encodes = Vec::new();
        if mode.control() {
            encodes.push((true, 0));
        }
        if mode.img2img() {
            encodes.push((false, if mode.control() { context } else { 0 }));
        }
        for (is_control, carried) in encodes {
            let base = render_res + carried;
            push(
                q::INIT_IMAGE,
                base,
                crate::pipeline::init_image_bytes(route.width, route.height),
            )?;
            for (label, live, scratch, op_h, op_w) in self.encode_quanta(vae, w, h, backend)? {
                operation_extent.set((op_h as u32, op_w as u32));
                push(label, base + f32b(&[1, 3, h, w]) + live, scratch)?;
                operation_extent.set((route.height, route.width));
            }
            if is_control {
                push(
                    q::CONTROL_CONTEXT,
                    base + latent,
                    crate::pipeline::control_context_bytes(lc, 1, lh, lw),
                )?;
            }
        }

        // ---- Denoise ----
        let _glue = crate::CompileGlueGuard::enable();
        let dit = self.dit.base();
        let cfg = &f.dit_cfg;
        let dim = cfg.dim;
        let budget = crate::pipeline::attention_budget_for(Some(route.execution));
        let plan = AttentionPlan::budgeted(budget);
        let img_tokens = (1 / cfg.f_patch_size) * (lh / cfg.patch_size) * (lw / cfg.patch_size);
        let (_, _, cap_pad) = crate::transformer::caption_pos_keep(n_valid);
        let c = n_valid + cap_pad;
        let (_, _, img_pad) = crate::transformer::image_pos_keep(
            1 / cfg.f_patch_size,
            lh / cfg.patch_size,
            lw / cfg.patch_size,
            c,
        );
        let n = img_tokens + img_pad;
        let l = n + c;
        let rope_dim: i32 = cfg.axes_dims.iter().sum();
        let cap = carrier(&[n_valid, cfg.cap_feat_dim], cap_dtype)?;
        let cap_tokens_dtype = if cap_pad > 0 {
            Dtype::Float32
        } else {
            cap_dtype
        };
        let cap_tokens = carrier(&[c, cfg.cap_feat_dim], cap_tokens_dtype)?;
        let x_freqs = carrier(&[n, rope_dim], Dtype::Float32)?;
        let cap_freqs = carrier(&[c, rope_dim], Dtype::Float32)?;
        let unified_freqs = carrier(&[l, rope_dim], Dtype::Float32)?;
        let stream = |rows: i32| carrier(&[1, rows, dim], Dtype::Float32);
        let (x_stream, cap_stream, unified) = (stream(n)?, stream(c)?, stream(l)?);
        let row = |rows: i32| f32b(&[1, rows, dim]);
        let prep = mlx_gen::ios_quantum::bytes(&cap_tokens)
            + f32b(&[n, rope_dim])
            + f32b(&[c, rope_dim])
            + f32b(&[l, rope_dim]);
        // img2img: the clean init latents, and the bf16 noise `denoise_batch` blended them with —
        // its binding outlives the denoise of that image.
        let noise = q::ab(2, &[lc as u64, 1, lh as u64, lw as u64]);
        let mut denoise_res = render_res
            + prep
            + latent
            + completed
            + if mode.img2img() { latent + noise } else { 0 };
        if mode.control() {
            // The control context and its cached embedding.
            denoise_res += context + row(n);
        }
        let elements = (lc as u64) * (lh as u64) * (lw as u64);
        push(
            q::LATENTS,
            denoise_res,
            q::latents_bytes(elements, mode.img2img()),
        )?;
        push(
            q::DIT_PREPARE,
            denoise_res,
            dit.prepare_bytes((lc, 1, lh, lw), &cap),
        )?;
        if let Dit::Control(control) = &self.dit {
            let cc = carrier(&[2 * lc + 1, 1, lh, lw], Dtype::Float32)?;
            push(
                q::CONTROL_EMBED,
                denoise_res,
                control.control_embed_bytes(&cc, img_pad),
            )?;
        }
        // Step 0 of a text-to-image denoise runs on the bf16 noise; every later step (and an
        // img2img blend) on f32 latents.
        let mut latent_dtypes = vec![Dtype::Float32];
        if !mode.img2img() {
            latent_dtypes.push(Dtype::Bfloat16);
        }
        for dtype in latent_dtypes {
            let x = carrier(&[lc, 1, lh, lw], dtype)?;
            push(
                q::DIT_EMBED,
                denoise_res,
                dit.embed_bytes(&x, img_pad, backend),
            )?;
        }
        let x_emb_live = row(n);
        // The control refiner: its 2 hints and threaded state live through the base refiner.
        let mut refiner_live = 0;
        if let Dit::Control(control) = &self.dit {
            for (k, block) in control.control_noise_refiner().iter().enumerate() {
                let scratch = block.quantum_scratch(&x_stream, &x_stream, &x_freqs, plan, backend);
                push(
                    q::CONTROL_BLOCK,
                    denoise_res + x_emb_live + row(n) + (k as u64) * row(n),
                    scratch,
                )?;
            }
            refiner_live = (control.control_noise_refiner().len() as u64 + 1) * row(n);
        }
        for (i, block) in dit.noise_refiner.iter().enumerate() {
            let hinted = self.control.is_some()
                && crate::control_transformer::hint_index(
                    &crate::control_transformer::CONTROL_REFINER_PLACES,
                    i,
                )
                .is_some();
            let scratch = block
                .quantum_scratch(&x_stream, &x_freqs, plan, backend)
                .saturating_add(if hinted {
                    crate::control_transformer::add_hint_bytes(&x_stream)
                } else {
                    0
                });
            push(
                q::DIT_BLOCK,
                denoise_res + x_emb_live + refiner_live,
                scratch,
            )?;
        }
        let streams_res = denoise_res + x_emb_live + refiner_live;
        push(
            q::DIT_CAPTION,
            streams_res,
            dit.caption_bytes(&cap_tokens, backend),
        )?;
        for block in &dit.context_refiner {
            push(
                q::DIT_CONTEXT_BLOCK,
                streams_res + row(c),
                block.quantum_scratch(&cap_stream, &cap_freqs, plan, backend),
            )?;
        }
        let unify = crate::transformer::unify_bytes(&x_stream, &cap_stream);
        push(q::DIT_UNIFY, streams_res + row(c), unify)?;
        let mut stack_res = streams_res + row(c) + row(l);
        if let Some(cf) = &self.control {
            // The control stream's own unify, then the 15 main blocks (their hints accumulate).
            push(q::DIT_UNIFY, stack_res, unify)?;
            let resident_blocks = match &self.dit {
                Dit::Control(control) => &control.control_layers,
                Dit::Base(_) => unreachable!(),
            };
            for (k, (block, read)) in cf.main.iter().enumerate() {
                let (block, read) = if streamable {
                    (block, *read)
                } else {
                    (&resident_blocks[k], 0)
                };
                let scratch = block
                    .quantum_scratch(&unified, &unified, &unified_freqs, plan, backend)
                    .saturating_add(read);
                push(
                    q::CONTROL_BLOCK,
                    stack_res + row(l) + (k as u64) * row(l),
                    scratch,
                )?;
            }
            stack_res += (cf.main.len() as u64) * row(l);
        }
        for (i, (streamed, read)) in self.dit_blocks.iter().enumerate() {
            let (block, read) = if streamable {
                (streamed, *read)
            } else {
                (&dit.layers[i], 0)
            };
            let hinted = self.control.is_some()
                && crate::control_transformer::hint_index(
                    &crate::control_transformer::CONTROL_LAYERS_PLACES,
                    i,
                )
                .is_some();
            let scratch = block
                .quantum_scratch(&unified, &unified_freqs, plan, backend)
                .saturating_add(read)
                .saturating_add(if hinted {
                    crate::control_transformer::add_hint_bytes(&unified)
                } else {
                    0
                });
            push(q::DIT_BLOCK, stack_res, scratch)?;
        }
        push(
            q::DIT_OUTPUT,
            stack_res,
            dit.output_bytes(&unified, backend),
        )?;
        push(
            mlx_gen::sampler::SAMPLER_STEP_QUANTUM,
            denoise_res + latent,
            mlx_gen::ios_quantum::sampler_step_bytes(elements),
        )?;
        drop(_glue);

        // ---- Decode: staged, only the VAE stays (the DiT and ControlNet shed). ----
        let decode_res = if staged {
                0
            } else {
                te_res + dit_res + ctrl_res
            }
            + latent * u64::from(route.count);
        let tiling = crate::pipeline::decode_tiling_for(
            Some(route.execution),
            route.width,
            route.height,
            staged,
        );
        for (label, live, scratch, op_h, op_w) in
            self.decode_quanta(vae, w, h, tiling.as_ref(), backend)?
        {
            operation_extent.set((op_h as u32, op_w as u32));
            push(label, decode_res + live, scratch)?;
            operation_extent.set((route.height, route.width));
        }
        let pixels = 3 * (h as u64) * (w as u64);
        push(
            q::DECODED_TO_IMAGE,
            decode_res + q::ab(4, &[pixels]),
            q::zimage_pixel_conversion_bytes(pixels, 4),
        )?;
        Ok(out)
    }

    /// The VAE encode of one `w x h` image as the device runs it ([`Vae::encode`]: the bounded,
    /// layer-wise tiled encode above [`crate::vae::decoder::IOS_ENCODE_TILE_PX`], else whole), as
    /// `(label, live input beside the image, scratch)`.
    fn encode_quanta(
        &self,
        vae: &Vae,
        w: i32,
        h: i32,
        b: Backend,
    ) -> Result<Vec<(&'static str, u64, u64, i32, i32)>> {
        let encoder = vae
            .encoder()
            .ok_or_else(|| Error::Msg("iOS pricing: the VAE has no encoder".into()))?;
        let (conv_in, downs, mid, conv_out) = encoder.parts();
        let labels = &q::ENCODE_LABELS;
        let tile_px = crate::vae::decoder::IOS_ENCODE_TILE_PX;
        let tiled = h > tile_px || w > tile_px;
        let image = carrier(&[1, 3, h, w], Dtype::Float32)?;
        let mut out = Vec::new();
        let (mut ch, mut cw) = (h, w);
        let mut c = mlx_gen::ios_quantum::conv_channels(conv_in.weight()).1;
        if tiled {
            let x = [1, h, w, 3];
            stage_and_tiles(
                &mut out,
                labels,
                0,
                &x,
                conv_in.weight(),
                tile_px.max(3),
                false,
                false,
                b,
            );
            let mut scale = 1;
            for down in downs {
                let edge = ((tile_px + scale - 1) / scale).max(3);
                let (resnets, downsampler) = down.parts();
                let next = if downsampler.is_some() {
                    scale * 2
                } else {
                    scale
                };
                // The caller holds the block's input across the whole block
                // (`DownEncoderBlock::forward_tiled(&h)`): live beside every layer after the first.
                let block_in = q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                for (k, r) in resnets.iter().enumerate() {
                    let held = if k > 0 { block_in } else { 0 };
                    let live = held + q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                    let (c1, c2) = r.conv_weights();
                    let x = [1, ch, cw, c];
                    stage_and_tiles(&mut out, labels, live, &x, c1, edge, true, true, b);
                    let cout = mlx_gen::ios_quantum::conv_channels(c1).1;
                    let stage = mlx_gen::ios_quantum::tiled_stage_bytes(&x, c, cout, 4, true);
                    for t in tile_extents(ch, cw, edge) {
                        out.push((
                            labels.tile,
                            live + stage,
                            mlx_gen::ios_quantum::tile_bytes(
                                &[1, t.0, t.1, cout],
                                normalized(cw),
                                4,
                                c2,
                                false,
                                b,
                            ),
                            t.0,
                            t.1,
                        ));
                    }
                    c = mlx_gen::ios_quantum::conv_channels(c2).1;
                }
                if let Some(ds) = downsampler {
                    let core = ((tile_px + next - 1) / next).max(1);
                    let held = if resnets.is_empty() { 0 } else { block_in };
                    let live = held + q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                    let (oh, ow) = ((ch - 2) / 2 + 1, (cw - 2) / 2 + 1);
                    let cout = ds.weight().shape()[0];
                    let stage = mlx_gen::ios_quantum::downsample_stage_bytes(&[1, oh, ow, cout], 4);
                    out.push((labels.stage, live, stage, ch, cw));
                    let crop = [1, 2 * core.min(oh) + 1, 2 * core.min(ow) + 1, c];
                    out.push((
                        labels.tile,
                        live + stage,
                        mlx_gen::ios_quantum::downsample_tile_bytes(&crop, 4, ds.weight(), b),
                        crop[1],
                        crop[2],
                    ));
                    (ch, cw, c) = (oh, ow, cout);
                }
                scale = next;
            }
        } else {
            let call = ConvCall::new(&[1, h, w, 3], 4, conv_in.weight(), 1, 1);
            out.push((
                q::VAE_ENCODE_IN,
                0,
                mlx_gen::ios_quantum::single_conv_bytes(&call, 0, 0, b),
                ch,
                cw,
            ));
            for down in downs {
                let (resnets, downsampler) = down.parts();
                // The caller holds the block's input across the whole block
                // (`DownEncoderBlock::forward(&h)`): live beside every layer after the first.
                let block_in = q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                for (k, r) in resnets.iter().enumerate() {
                    let held = if k > 0 { block_in } else { 0 };
                    let x = carrier(&[1, c, ch, cw], Dtype::Float32)?;
                    out.push((
                        labels.resnet,
                        held + mlx_gen::ios_quantum::bytes(&x),
                        r.quantum_scratch(&x, b),
                        ch,
                        cw,
                    ));
                    c = mlx_gen::ios_quantum::conv_channels(r.conv_weights().1).1;
                }
                if let Some(ds) = downsampler {
                    let held = if resnets.is_empty() { 0 } else { block_in };
                    let call = ConvCall::new(&[1, ch + 1, cw + 1, c], 4, ds.weight(), 2, 0);
                    out.push((
                        q::VAE_DOWNSAMPLE,
                        held + q::ab(4, &[1, c as u64, ch as u64, cw as u64]),
                        mlx_gen::ios_quantum::single_conv_bytes(&call, 1, 0, b),
                        ch,
                        cw,
                    ));
                    (ch, cw) = ((ch - 2) / 2 + 1, (cw - 2) / 2 + 1);
                    c = ds.weight().shape()[0];
                }
            }
        }
        mid_quanta(&mut out, labels, mid, c, ch, cw, b)?;
        let call = ConvCall::new(&[1, ch, cw, c], 4, conv_out.weight(), 1, 1);
        out.push((
            q::VAE_ENCODE_OUT,
            q::ab(4, &[1, c as u64, ch as u64, cw as u64]),
            mlx_gen::ios_quantum::norm_conv_bytes(&call, b)
                .saturating_add(Vae::latent_mean_bytes(&image)),
            ch,
            cw,
        ));
        Ok(out)
    }

    /// The decode of one `w x h` image (`decode_batch`: [`Vae::decode_tiled`] under `tiling`, else
    /// [`Vae::decode`]), as `(label, live input, scratch)`.
    fn decode_quanta(
        &self,
        vae: &Vae,
        w: i32,
        h: i32,
        tiling: Option<&mlx_gen::tiling::TilingConfig>,
        b: Backend,
    ) -> Result<Vec<(&'static str, u64, u64, i32, i32)>> {
        let decoder = vae.decoder();
        let (conv_in, mid, ups, conv_out) = decoder.parts();
        let labels = &q::DECODE_LABELS;
        let spatial = crate::pipeline::SPATIAL_SCALE as i32;
        let (lh, lw) = (h / spatial, w / spatial);
        let lc = crate::pipeline::LATENT_CHANNELS;
        let mut out = Vec::new();
        let call = ConvCall::new(&[1, lh, lw, lc], 4, conv_in.weight(), 1, 1);
        out.push((
            q::VAE_DECODE_IN,
            0,
            mlx_gen::ios_quantum::single_conv_bytes(&call, 0, 0, b)
                .saturating_add(Vae::denorm_bytes((lc * lh * lw) as u64)),
            lh,
            lw,
        ));
        let mut c = mlx_gen::ios_quantum::conv_channels(conv_in.weight()).1;
        mid_quanta(&mut out, labels, mid, c, lh, lw, b)?;
        // The head (the mid block's output) stays live through the tail.
        let head = q::ab(4, &[1, c as u64, lh as u64, lw as u64]);
        let tile =
            tiling.filter(|t| t.needs_tiling(mlx_gen::tiling::VaeTiling::QWEN_IMAGE, 1, lh, lw));
        let (mut ch, mut cw) = (lh, lw);
        match tile.and_then(|t| t.spatial.as_ref()).map(|s| s.tile_px) {
            Some(tile_px) => {
                let edge_at = |remaining: i32| ((tile_px + remaining - 1) / remaining).max(3);
                let mut remaining = ups
                    .iter()
                    .filter(|u| u.upsamples())
                    .fold(1_i32, |s, _| s.saturating_mul(2));
                for (i, up) in ups.iter().enumerate() {
                    let edge = edge_at(remaining);
                    let next = if up.upsamples() {
                        (remaining / 2).max(1)
                    } else {
                        remaining
                    };
                    let (resnets, upsampler) = up.parts();
                    // The caller holds the block's input across the whole block
                    // (`UpDecoderBlock::forward_tiled(&h)`): live beside every layer after the
                    // first (the first block's input is the head, already live).
                    let block_in = held_block_input(i, c, ch, cw);
                    for (k, r) in resnets.iter().enumerate() {
                        let held = if k > 0 { block_in } else { 0 };
                        let live = held + head + q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                        let (c1, c2) = r.conv_weights();
                        let x = [1, ch, cw, c];
                        stage_and_tiles(&mut out, labels, live, &x, c1, edge, true, true, b);
                        let cout = mlx_gen::ios_quantum::conv_channels(c1).1;
                        let stage = mlx_gen::ios_quantum::tiled_stage_bytes(&x, c, cout, 4, true);
                        for t in tile_extents(ch, cw, edge) {
                            out.push((
                                labels.tile,
                                live + stage,
                                mlx_gen::ios_quantum::tile_bytes(
                                    &[1, t.0, t.1, cout],
                                    normalized(cw),
                                    4,
                                    c2,
                                    false,
                                    b,
                                ),
                                t.0,
                                t.1,
                            ));
                        }
                        c = mlx_gen::ios_quantum::conv_channels(c2).1;
                    }
                    if let Some(u) = upsampler {
                        let held = if resnets.is_empty() { 0 } else { block_in };
                        let live = held + head + q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                        let x = [1, 2 * ch, 2 * cw, c];
                        stage_and_tiles(
                            &mut out,
                            labels,
                            live,
                            &x,
                            u.weight(),
                            edge_at(next),
                            false,
                            false,
                            b,
                        );
                        (ch, cw) = (2 * ch, 2 * cw);
                        c = mlx_gen::ios_quantum::conv_channels(u.weight()).1;
                    }
                    remaining = next;
                }
                let live = head + q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                stage_and_tiles(
                    &mut out,
                    labels,
                    live,
                    &[1, ch, cw, c],
                    conv_out.weight(),
                    tile_px.max(3),
                    false,
                    true,
                    b,
                );
            }
            None => {
                for (i, up) in ups.iter().enumerate() {
                    let (resnets, upsampler) = up.parts();
                    // The caller holds the block's input across the whole block
                    // (`UpDecoderBlock::forward(&h)`), as in the tiled tail.
                    let block_in = held_block_input(i, c, ch, cw);
                    for (k, r) in resnets.iter().enumerate() {
                        let held = if k > 0 { block_in } else { 0 };
                        let x = carrier(&[1, c, ch, cw], Dtype::Float32)?;
                        out.push((
                            labels.resnet,
                            held + head + mlx_gen::ios_quantum::bytes(&x),
                            r.quantum_scratch(&x, b),
                            ch,
                            cw,
                        ));
                        c = mlx_gen::ios_quantum::conv_channels(r.conv_weights().1).1;
                    }
                    if let Some(u) = upsampler {
                        let held = if resnets.is_empty() { 0 } else { block_in };
                        let call = ConvCall::new(&[1, 2 * ch, 2 * cw, c], 4, u.weight(), 1, 1);
                        out.push((
                            q::VAE_UPSAMPLE,
                            held + head + q::ab(4, &[1, c as u64, ch as u64, cw as u64]),
                            mlx_gen::ios_quantum::single_conv_bytes(&call, 1, 0, b),
                            ch,
                            cw,
                        ));
                        (ch, cw) = (2 * ch, 2 * cw);
                        c = mlx_gen::ios_quantum::conv_channels(u.weight()).1;
                    }
                }
                let call = ConvCall::new(&[1, ch, cw, c], 4, conv_out.weight(), 1, 1);
                out.push((
                    q::VAE_DECODE_OUT,
                    head + q::ab(4, &[1, c as u64, ch as u64, cw as u64]),
                    mlx_gen::ios_quantum::norm_conv_bytes(&call, b),
                    ch,
                    cw,
                ));
            }
        }
        Ok(out)
    }
}

/// The f32 input of up block `i` (`[1, c, h, w]`) the decoder holds across that block's layers —
/// zero for the first block, whose input is the head (already priced live through the tail).
fn held_block_input(i: usize, c: i32, h: i32, w: i32) -> u64 {
    if i == 0 {
        0
    } else {
        q::ab(4, &[1, c as u64, h as u64, w as u64])
    }
}

/// The mid block's quanta (resnet, attention, resnet) over a `[1, c, h, w]` f32 activation.
fn mid_quanta(
    out: &mut Vec<(&'static str, u64, u64, i32, i32)>,
    labels: &'static q::VaeLabels,
    mid: &crate::vae::mid_block::UNetMidBlock,
    c: i32,
    h: i32,
    w: i32,
    b: Backend,
) -> Result<()> {
    let (r0, _attention, r1) = mid.parts();
    let x = carrier(&[1, c, h, w], Dtype::Float32)?;
    let live = mlx_gen::ios_quantum::bytes(&x);
    out.push((labels.resnet, live, r0.quantum_scratch(&x, b), h, w));
    let n = h * w;
    let rows = crate::vae::VaeAttention::bounded_plan()
        .budget
        .query_block(1, 1, n, n);
    out.push((
        labels.attention,
        live,
        mlx_gen::ios_quantum::vae_attention_bytes(1, n, c, rows, 4, b),
        h,
        w,
    ));
    out.push((labels.resnet, live, r1.quantum_scratch(&x, b), h, w));
    Ok(())
}

/// One layer-wise tiled stage over `x` (NHWC, f32) run by `weight`'s 3x3 convolution (on crops
/// GroupNorm + SiLU preprocessed when `normalized`) — its stage
/// check ([`q::before_tiled_stage`]) and its largest tile check ([`q::before_tile`]) at `edge`.
#[allow(clippy::too_many_arguments)]
fn stage_and_tiles(
    out: &mut Vec<(&'static str, u64, u64, i32, i32)>,
    labels: &'static q::VaeLabels,
    live: u64,
    x: &[i32; 4],
    weight: &Array,
    edge: i32,
    resnet: bool,
    normalized: bool,
    b: Backend,
) {
    let (cin, cout) = mlx_gen::ios_quantum::conv_channels(weight);
    let stage = mlx_gen::ios_quantum::tiled_stage_bytes(x, cin, cout, 4, resnet);
    let source = mlx_gen::ios_quantum::TileSource {
        full_width: x[2],
        normalized,
    };
    out.push((labels.stage, live, stage, x[1], x[2]));
    for t in tile_extents(x[1], x[2], edge) {
        out.push((
            labels.tile,
            live + stage,
            mlx_gen::ios_quantum::tile_bytes(&[1, t.0, t.1, x[3]], source, 4, weight, false, b),
            t.0,
            t.1,
        ));
    }
}

/// All distinct actual border/interior operand extents; never collapse different tile mechanisms
/// to the largest independently priced shape before the anchor consumer sees their windows.
fn tile_extents(h: i32, w: i32, max_tile_edge: i32) -> Vec<(i32, i32)> {
    if h <= max_tile_edge && w <= max_tile_edge {
        return vec![(h, w)];
    }
    let axis = |n: i32| {
        let core = (max_tile_edge - 2).max(1);
        (0..n)
            .step_by(core as usize)
            .map(|start| ((start + core).min(n) + 1).min(n) - (start - 1).max(0))
            .collect::<std::collections::BTreeSet<_>>()
    };
    let mut out = Vec::new();
    for th in axis(h) {
        for tw in axis(w) {
            out.push((th, tw));
        }
    }
    out
}

/// The tile source of a GroupNorm + SiLU preprocessed crop of a `full_width`-wide activation.
fn normalized(full_width: i32) -> mlx_gen::ios_quantum::TileSource {
    mlx_gen::ios_quantum::TileSource {
        full_width,
        normalized: true,
    }
}

/// The largest halo-expanded tile `mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc` cuts from an
/// `h x w` activation at `max_tile_edge`: the whole activation when it fits one tile; otherwise
/// cores of `max_tile_edge - 2` with a one-pixel halo on every interior side, so the largest extent
/// along each axis (every row/column pair is a tile).
#[cfg(test)]
pub(crate) fn tile_extent(h: i32, w: i32, max_tile_edge: i32) -> (i32, i32) {
    if h <= max_tile_edge && w <= max_tile_edge {
        return (h, w);
    }
    let axis = |n: i32| {
        let core = max_tile_edge - 2;
        let (mut start, mut largest) = (0, 0);
        while start < n {
            let end = (start + core).min(n);
            largest = largest.max((end + 1).min(n) - (start - 1).max(0));
            start = end;
        }
        largest
    };
    (axis(h), axis(w))
}

#[cfg(test)]
mod load_quantization_header_tests {
    use super::*;
    fn header(
        name: &str,
        dtype: HeaderDtype,
        shape: &[usize],
        width: u64,
    ) -> SafetensorsTensorHeader {
        SafetensorsTensorHeader {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: shape.iter().fold(width, |bytes, &dim| bytes * dim as u64),
        }
    }
    #[test]
    fn vae_histories_preserve_joint_overwrites_dynamic_encoder_and_failed_prefixes() {
        let a = "decoder.conv_in.weight";
        let c = "decoder.decoder.conv_in.weight";
        let target = "conv_in.conv.weight";
        let source = Component { files: Vec::new(), headers: vec![
            header(a, HeaderDtype::F16, &[2, 3, 5, 7], 2),
            header(c, HeaderDtype::F32, &[11, 13, 17, 19], 4),
            header(target, HeaderDtype::F16, &[23, 29, 31, 37], 2),
        ] };
        let inventory = VaeSourceInventory::new(&source).unwrap();
        let mut results = Vec::new();
        inventory.visit_selected(&std::collections::BTreeSet::from([target.into()]), &mut |world| {
            let value = &world.current[target];
            results.push((value.raw.clone(), value.header.shape.clone(), value.header.dtype,
                value.path.iter().map(|op| op.source.clone()).collect::<Vec<_>>(),
                world.current[a].raw.clone()));
            Ok(())
        }).unwrap();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(results, vec![
            (a.into(), vec![2, 5, 7, 3], HeaderDtype::F16, vec![a.into()], c.into()),
            (c.into(), vec![11, 17, 19, 13], HeaderDtype::F32, vec![c.into(), a.into()], c.into()),
        ], "VAE-HOST-JOINT-OVERWRITE: no initial-target, repeated-op or impossible field world");

        let generated = "decoder.encoder.conv_in.weight";
        let source = Component { files: Vec::new(), headers: vec![
            header(generated, HeaderDtype::F16, &[2, 3, 5, 7], 2),
        ] };
        let inventory = VaeSourceInventory::new(&source).unwrap();
        let mut generated_worlds = 0;
        inventory.visit_selected(&std::collections::BTreeSet::from(["encoder.conv_in.conv.weight".into()]), &mut |world| {
            assert_eq!(world.current.contains_key("encoder.conv_in.conv.weight"), true,
                "VAE-HOST-DYNAMIC-ENCODER-EXISTS");
            let value = &world.current["encoder.conv_in.conv.weight"];
            assert_eq!((&value.raw, &value.header.shape, value.path.len(), world.reads.iter().map(|r| r.encoder).collect::<Vec<_>>()),
                (&generated.to_owned(), &vec![2, 5, 7, 3], 2, vec![false, true]),
                "VAE-HOST-DYNAMIC-ENCODER: decoder-generated key executes in encoder snapshot");
            generated_worlds += 1;
            Ok(())
        }).unwrap();
        assert_eq!(generated_worlds, 1);

        let source = Component { files: Vec::new(), headers: vec![
            header(a, HeaderDtype::F16, &[2, 3], 2),
            header(c, HeaderDtype::F32, &[11, 13, 17, 19], 4),
        ] };
        let inventory = VaeSourceInventory::new(&source).unwrap();
        let component = inventory.supports.iter().position(|s| s.contains(target)).unwrap();
        let (mut successes, mut failures) = (0, 0);
        inventory.visit_component(component, &mut |world| {
            assert_eq!(world.current[target].raw, c, "VAE-HOST-SUCCESS-AFTER-SHADOW");
            successes += 1;
            Ok(false)
        }, &mut |failure| {
            assert_eq!((failure.step.source.as_str(), failure.prefix.reads.len(), failure.prefix.current[a].header.shape.len()),
                (a, 0, 2), "VAE-HOST-NATIVE-FAILURE-PREFIX");
            failures += 1;
            Ok(())
        }).unwrap();
        assert_eq!((successes, failures), (1, 1), "VAE-HOST-NO-BLANKET-REFUSAL");
    }

    fn empty_vae_facts() -> Facts {
        let empty = Component { files: Vec::new(), headers: Vec::new() };
        Facts { te_cfg: ZTextEncoderConfig::z_image(), dit_cfg: ZImageTransformerConfig::turbo(),
            te: empty.clone(), dit: empty.clone(), vae: empty, control: None,
            window_tokens: 1, caption_tokens: 1, quant_bits: Some(4), streamable: false,
            load_components: None, quant_conversions: std::array::from_fn(|_| Vec::new()), te_quant_bits: None }
    }

    #[test]
    fn vae_post_remap_conversion_and_replay_keep_same_source_owners_additive() {
        let base = "mid_block.attentions.0.to_q";
        let a = format!("decoder.{base}.weight");
        let c = format!("decoder.decoder.{base}.weight");
        let bias = format!("{base}.bias");
        let untouched = "conv_norm_out.norm.weight";
        let source = Component { files: Vec::new(), headers: vec![
            header(&a, HeaderDtype::F16, &[4, 64], 2),
            header(&c, HeaderDtype::F32, &[4, 64], 4),
            header(&bias, HeaderDtype::F32, &[4], 4),
            header(untouched, HeaderDtype::BF16, &[4], 2),
        ] };
        let facts = empty_vae_facts();
        let inventory = VaeSourceInventory::new(&source).unwrap();
        let keys = std::collections::BTreeSet::from([format!("{base}.weight"), bias.clone(), untouched.into()]);
        let mut cases = Vec::new();
        inventory.visit_selected(&keys, &mut |world| {
            let component = VaeSourceInventory::component(world);
            let (effective, records) = component.load_quantized_with_bias(&[(base.into(), false)], 4, |_| true)?;
            let owners = VaeOwnerHistory::from_constructed(world, &keys, &records)?;
            let old = world.current[&format!("{base}.weight")].header.data_bytes + 16;
            let mut cursor = QuantizationCursor::new(QuantizationHistory::RequestFirst, false);
            assert_eq!(cursor.window_with_vae(&facts, IosPhase::Decode, q::VAE_DECODE_IN, 0, 0, true, Some(&owners))?,
                Some((old + 8, 0, true)), "VAE-HOST-ADDITIVE-COLD: unrelated owners are not subtracted uppers");
            assert_eq!(cursor.window_with_vae(&facts, IosPhase::Decode, q::VAE_DECODE_ATTENTION, 0, 0, true, Some(&owners))?,
                Some((old + 8, 152 + 512, true)), "VAE-HOST-SAME-HISTORY-CONVERSION");
            assert_eq!(cursor.window_with_vae(&facts, IosPhase::Decode, q::VAE_DECODE_OUT, 0, 0, true, Some(&owners))?,
                Some((152 + 8, 0, false)), "VAE-HOST-RECURRENT-OUTPUTS");
            let mut late = QuantizationCursor::new(QuantizationHistory::NoModelForward, true);
            assert_eq!(late.window_with_vae(&facts, IosPhase::Decode, q::VAE_DECODE_IN, 0, 0, false, Some(&owners))?,
                Some((160, 0, false)), "VAE-HOST-LATE-SERIAL-DECODE");
            assert_eq!(effective.headers.iter().find(|h| h.name == format!("{base}.weight")).unwrap().shape, vec![4, 8],
                "VAE-HOST-ACTUAL-PACKED-SHAPE");
            cases.push((world.current[&format!("{base}.weight")].raw.clone(), old));
            Ok(())
        }).unwrap();
        cases.sort();
        let mut expected = vec![(a, 528), (c, 1040)];
        expected.sort();
        assert_eq!(cases, expected, "VAE-HOST-CONVERSION-AFTER-ORIGIN-DISCOVERY");
    }

    fn complete_vae_history_headers() -> Component {
        fn resnet(headers: &mut Vec<SafetensorsTensorHeader>, prefix: &str) {
            for norm in ["norm1", "norm2"] {
                for suffix in ["weight", "bias"] { headers.push(header(&format!("{prefix}.{norm}.{suffix}"), HeaderDtype::BF16, &[64], 2)); }
            }
            for conv in ["conv1", "conv2"] {
                headers.push(header(&format!("{prefix}.{conv}.weight"), HeaderDtype::BF16, &[64, 64, 3, 3], 2));
                headers.push(header(&format!("{prefix}.{conv}.bias"), HeaderDtype::BF16, &[64], 2));
            }
        }
        let mut headers = Vec::new();
        for side in ["decoder", "encoder"] {
            let encoder = side == "encoder";
            for (conv, input, output) in [("conv_in", if encoder { 3 } else { 16 }, 64), ("conv_out", 64, if encoder { 32 } else { 3 })] {
                headers.push(header(&format!("{side}.{conv}.weight"), HeaderDtype::BF16, &[output, input, 3, 3], 2));
                headers.push(header(&format!("{side}.{conv}.bias"), HeaderDtype::BF16, &[output], 2));
            }
            for suffix in ["weight", "bias"] {
                headers.push(header(&format!("{side}.conv_norm_out.{suffix}"), HeaderDtype::BF16, &[64], 2));
                headers.push(header(&format!("{side}.mid_block.attentions.0.group_norm.{suffix}"), HeaderDtype::BF16, &[64], 2));
            }
            for leaf in crate::vae::attention::QUANTIZATION_LINEAR_PATHS {
                headers.push(header(&format!("{side}.mid_block.attentions.0.{leaf}.weight"), HeaderDtype::BF16, &[64, 64], 2));
                headers.push(header(&format!("{side}.mid_block.attentions.0.{leaf}.bias"), HeaderDtype::BF16, &[64], 2));
            }
            for index in 0..2 { resnet(&mut headers, &format!("{side}.mid_block.resnets.{index}")); }
            for block in 0..4 {
                let direction = if encoder { "down" } else { "up" };
                for layer in 0..if encoder { 2 } else { 3 } {
                    resnet(&mut headers, &format!("{side}.{direction}_blocks.{block}.resnets.{layer}"));
                }
                if block < 3 {
                    headers.push(header(&format!("{side}.{direction}_blocks.{block}.{direction}samplers.0.conv.weight"), HeaderDtype::BF16, &[64, 64, 3, 3], 2));
                    headers.push(header(&format!("{side}.{direction}_blocks.{block}.{direction}samplers.0.conv.bias"), HeaderDtype::BF16, &[64], 2));
                }
            }
        }
        Component { files: Vec::new(), headers }
    }

    #[test]
    fn vae_histories_match_native_ordered_remap_conversion_and_forward() {
        const CHILD: &str = "ZIMAGE_VAE_HISTORY_CPU_CHILD";
        const FQN: &str = "ios_pricing::load_quantization_header_tests::vae_histories_match_native_ordered_remap_conversion_and_forward";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FQN, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1").status().unwrap();
            assert!(status.success(), "VAE history CPU child failed");
            return;
        }
        // BEFORE any fixture/Array: the installed CPU device AND native default CPU stream,
        // checked through real bindings, with Metal made fatal. No inherited default is trusted.
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(crate::test_metal_guard::forbid_metal());
        assert!(crate::test_metal_guard::metal_forbidden());
        assert_eq!(mlx_rs::Device::try_default().unwrap(), mlx_rs::Device::cpu());
        assert_eq!(unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) }, 0);
        assert_eq!(mlx_rs::Stream::new().get_index().unwrap(), mlx_rs::Stream::cpu().get_index().unwrap());
        eprintln!("ZIMAGE_CPU_GUARDS_READY_V1 device=cpu native_stream=cpu fatal_metal_trap=1");
        let normal_end = std::cell::Cell::new(false);
        let completed = std::cell::Cell::new(0u8);
        mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(1 << 34, || -> Result<()> {
            let _scope = mlx_gen::ios_quantum::Scope::try_new()?;
            let a = "decoder.conv_in.weight";
            let c = "decoder.decoder.conv_in.weight";
            for (index, c_first) in [true, false].into_iter().enumerate() {
                let mut view = Weights::from_map([
                    (a.into(), Array::ones::<f32>(&[2, 3, 1, 1])?),
                    (c.into(), Array::ones::<f32>(&[2, 3, 1, 1])?.multiply(Array::from_f32(2.0))?),
                    ("conv_in.conv.weight".into(), Array::ones::<f32>(&[2, 1, 1, 3])?.multiply(Array::from_f32(99.0))?),
                    ("conv_in.conv.bias".into(), Array::zeros::<f32>(&[2])?),
                ].into_iter().collect());
                let order = if c_first { vec![c.into(), a.into()] } else { vec![a.into(), c.into()] };
                crate::loader::remap_vae_decoder_ordered(&mut view, order)?;
                let view = Weights::from_map(view.into_tensors());
                assert_eq!(view.require("conv_in.conv.weight")?.shape(), &[2, 1, 1, 3],
                    "VAE-NATIVE-TWO-ORDER-WEIGHT-SHAPE");
                let conv = crate::vae::conv_layers::ConvLayer::from_weights(&view, "conv_in")?;
                let output = conv.forward(&Array::ones::<f32>(&[1, 3, 2, 2])?)?;
                output.eval()?;
                assert_eq!((view.require("conv_in.conv.weight")?.shape().to_vec(), output.as_slice::<f32>().iter().copied().fold(0.0f32, f32::max)),
                    (vec![2, 1, 1, 3], if c_first { 6.0 } else { 3.0 }),
                    "VAE-NATIVE-TWO-ORDER-CONV: current-source overwrite must reach actual constructor/forward");
                completed.set(completed.get() | (1 << index));
                eprintln!("ZIMAGE_VAE_HISTORY_NATIVE_MECHANISM_V1 index={index}");
            }
            {
                let mut view = Weights::from_map([
                    ("decoder.encoder.conv_in.weight".into(), Array::ones::<f32>(&[2, 3, 1, 1])?),
                ].into_iter().collect());
                crate::loader::remap_vae_decoder_ordered(&mut view, vec!["decoder.encoder.conv_in.weight".into()])?;
                crate::loader::remap_vae_encoder(&mut view)?;
                assert_eq!(view.require("encoder.conv_in.conv.weight")?.shape(), &[2, 1, 1, 3],
                    "VAE-NATIVE-GENERATED-ENCODER: actual second snapshot includes newly inserted key");
                view.require("encoder.conv_in.conv.weight")?.eval()?;
                completed.set(completed.get() | (1 << 2));
                eprintln!("ZIMAGE_VAE_HISTORY_NATIVE_MECHANISM_V1 index=2");
            }
            {
                let mut view = Weights::from_map([
                    (a.into(), Array::ones::<f32>(&[2, 3])?),
                    (c.into(), Array::ones::<f32>(&[2, 3, 1, 1])?),
                ].into_iter().collect());
                assert!(crate::loader::remap_vae_decoder_ordered(&mut view, vec![a.into(), c.into()]).is_err(),
                    "VAE-NATIVE-FAILED-RANK-PREFIX");
                // The SAME native state failed before inserting a target; C may still overwrite A
                // and the other ORIGINAL order succeeds, rather than refusing all native worlds.
                assert!(view.get("conv_in.conv.weight").is_none());
                crate::loader::remap_vae_decoder_ordered(&mut view, vec![c.into(), a.into()])?;
                assert_eq!(view.require("conv_in.conv.weight")?.shape(), &[2, 1, 1, 3],
                    "VAE-NATIVE-FEASIBLE-RANK-ORDER");
                completed.set(completed.get() | (1 << 3));
                eprintln!("ZIMAGE_VAE_HISTORY_NATIVE_MECHANISM_V1 index=3");
            }
            let base = "mid_block.attentions.0.to_q";
            let a = format!("decoder.{base}.weight");
            let c = format!("decoder.decoder.{base}.weight");
            for (index, c_first) in [true, false].into_iter().enumerate() {
                let mut view = Weights::from_map([
                    (a.clone(), Array::ones::<f32>(&[4, 64])?.as_dtype(Dtype::Float16)?),
                    (c.clone(), Array::ones::<f32>(&[4, 64])?.multiply(Array::from_f32(2.0))?),
                    (format!("{base}.bias"), Array::ones::<f32>(&[4])?.multiply(Array::from_f32(3.0))?),
                ].into_iter().collect());
                let order = if c_first { vec![c.clone(), a.clone()] } else { vec![a.clone(), c.clone()] };
                crate::loader::remap_vae_decoder_ordered(&mut view, order)?;
                let view = Weights::from_map(view.into_tensors());
                let mut linear = crate::quant::lin(&view, base, true)?;
                assert_eq!(linear.base_shape(), vec![4, 64]);
                linear.quantize(4, None)?;
                linear.materialize_weights()?;
                let result = linear.forward(&Array::ones::<f32>(&[1, 64])?)?.as_dtype(Dtype::Float32)?;
                result.eval()?;
                assert_eq!(result.as_slice::<f32>(), &[if c_first { 131.0 } else { 67.0 }; 4],
                    "VAE-NATIVE-POST-REMAP-CONVERT-FORWARD: no canonical packed/dense preclassification");
                let (codes, scales, offsets, _, _, bits) = linear.quantized_params().unwrap();
                assert_eq!((codes.shape().to_vec(), scales.shape().to_vec(), offsets.shape().to_vec(), bits, linear.bias().unwrap().dtype()),
                    (vec![4, 8], vec![4, 1], vec![4, 1], 4, Dtype::Bfloat16),
                    "VAE-NATIVE-CLOSED-PACK-TUPLE");
                completed.set(completed.get() | (1 << (4 + index)));
                eprintln!("ZIMAGE_VAE_HISTORY_NATIVE_MECHANISM_V1 index={}", 4 + index);
            }
            // Exercise the ACTUAL pricing consumer, not just a certificate-shaped history API:
            // full native VAE+encoder constructor -> discovered remap histories -> closed metadata
            // conversion -> effective native ctor -> actual selected attention forward -> replay.
            for packed in [false, true] {
                let mut source = complete_vae_history_headers();
                let a = "decoder.mid_block.attentions.0.to_out.0.weight";
                let c = "decoder.decoder.mid_block.attentions.0.to_out.0.weight";
                let weight = source.headers.iter_mut().find(|h| h.name == a).unwrap();
                *weight = if packed { header(a, HeaderDtype::U32, &[64, 8], 4) }
                    else { header(a, HeaderDtype::F16, &[64, 64], 2) };
                source.headers.push(header(c, HeaderDtype::F32, &[64, 64], 4));
                if packed {
                    source.headers.push(header("decoder.mid_block.attentions.0.to_out.0.scales", HeaderDtype::BF16, &[64, 1], 2));
                    source.headers.push(header("decoder.mid_block.attentions.0.to_out.0.biases", HeaderDtype::BF16, &[64, 1], 2));
                }
                let inventory = VaeSourceInventory::new(&source)?;
                let mut accepted = 0;
                let consumer_result = inventory.visit_constructed(Some(4), &mut |vae, owners| {
                    assert_eq!(owners.conversions.len(), if packed { 7 } else { 8 },
                        "VAE-NATIVE-ACTUAL-CONSUMER-CONVERTS-DISCOVERED-FIELDS");
                    let (_, mid, _, _) = vae.decoder().parts();
                    let (_, attention, _) = mid.parts();
                    let result = attention.forward(&Array::ones::<f32>(&[1, 64, 1, 1])?)?;
                    result.eval()?;
                    assert_eq!((result.shape().to_vec(), result.as_slice::<f32>().to_vec()),
                        (vec![1, 64, 1, 1], vec![1.0; 64]),
                        "VAE-NATIVE-ACTUAL-CONSUMER-FORWARD");
                    let facts = empty_vae_facts();
                    let mut cold = QuantizationCursor::new(QuantizationHistory::RequestFirst, false);
                    let before = cold.window_with_vae(&facts, IosPhase::Decode, q::VAE_DECODE_IN, 0, 0, true, Some(&owners))?.unwrap();
                    let at = cold.window_with_vae(&facts, IosPhase::Decode, q::VAE_DECODE_ATTENTION, 0, 0, true, Some(&owners))?.unwrap();
                    let after = cold.window_with_vae(&facts, IosPhase::Decode, q::VAE_DECODE_OUT, 0, 0, true, Some(&owners))?.unwrap();
                    assert_eq!((before.0, before.1, at.0, at.1),
                        (at.0, 0, before.0, (if packed { 3 } else { 4 }) * 2304 + if packed { 0 } else { 8192 }),
                        "VAE-NATIVE-ACTUAL-CONSUMER-COLD-REPLAY");
                    assert!(after.0 > 0 && after.2, "unexecuted encoder conversion parents remain logically pending");
                    accepted += 1;
                    Ok(())
                });
                assert_eq!(consumer_result.is_ok(), true,
                    "VAE-NATIVE-DISCOVERY-FAILED-PACKED-READS: {:?}", consumer_result.as_ref().err());
                consumer_result?;
                assert_eq!(accepted, if packed { 1 } else { 2 },
                    "VAE-NATIVE-FAILED-PACKED-SELECTOR-DOES-NOT-REFUSE-FEASIBLE-WORLD");
                let index = if packed { 7 } else { 6 };
                completed.set(completed.get() | (1 << index));
                eprintln!("ZIMAGE_VAE_HISTORY_NATIVE_MECHANISM_V1 index={index}");
            }
            normal_end.set(true);
            Ok(())
        }).unwrap();
        assert!(normal_end.get(), "actual VAE history body incomplete");
        assert_eq!(completed.get(), 0b1111_1111, "actual VAE history full mechanism mask");
        let sync = unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) };
        assert_eq!(sync, 0, "VAE CPU synchronization after owners/Scope drop failed");
        eprintln!("ZIMAGE_VAE_HISTORY_NATIVE_BODY_COMPLETED_V1 mechanisms=8 fixture_owners_dropped=1 cpu_sync=0");
    }

    #[test]
    fn packed_metadata_preserves_actual_linear_embedding_casts_and_matched_owners() {
        let source = Component {
            files: Vec::new(),
            headers: vec![
                header("proj.weight", HeaderDtype::F16, &[4, 64], 2),
                header("proj.bias", HeaderDtype::F16, &[4], 2),
                header("embed.weight", HeaderDtype::F32, &[4, 64], 4),
                header("norm.weight", HeaderDtype::BF16, &[4], 2),
            ],
        };
        let result = source.load_quantized(&[("proj".into(), false), ("embed".into(), true)], 4);
        assert!(result.is_ok());
        let (effective, conversions) = result.unwrap();
        let find = |name: &str| effective.headers.iter().find(|h| h.name == name).unwrap();
        assert_eq!(find("proj.weight").dtype, HeaderDtype::U32);
        assert_eq!(find("proj.weight").shape, [4, 8]);
        assert_eq!(find("proj.scales").dtype, HeaderDtype::BF16);
        assert_eq!(find("proj.bias").dtype, HeaderDtype::BF16);
        assert_eq!(find("embed.scales").dtype, HeaderDtype::F32);
        assert_eq!(find("norm.weight").dtype, HeaderDtype::BF16);
        assert_eq!(source.headers[0].dtype, HeaderDtype::F16);
        let alloc = mlx_gen::ios_quantum::allocated_bytes;
        assert_eq!(
            conversions[0],
            (
                "proj".into(),
                alloc(512) + alloc(8),
                alloc(128) + alloc(8) + alloc(8) + alloc(8),
                q::ab(2, &[4, 64])
            )
        );
        assert_eq!(
            conversions[1],
            (
                "embed".into(),
                alloc(1024),
                alloc(128) + alloc(16) + alloc(16),
                0
            )
        );
        let (_, repeated) = effective
            .load_quantized(&[("proj".into(), false), ("embed".into(), true)], 4)
            .unwrap();
        assert!(repeated.is_empty());
    }
    #[test]
    fn mixed_leaf_metadata_keeps_existing_pack_and_converts_only_dense_source() {
        let source = Component {
            files: Vec::new(),
            headers: vec![
                header("packed.weight", HeaderDtype::U32, &[4, 8], 4),
                header("packed.scales", HeaderDtype::BF16, &[4, 1], 2),
                header("packed.biases", HeaderDtype::BF16, &[4, 1], 2),
                header("dense.weight", HeaderDtype::BF16, &[4, 64], 2),
            ],
        };
        let result = source.load_quantized(&[("packed".into(), false), ("dense".into(), false)], 4);
        assert!(result.is_ok());
        let (effective, conversions) = result.unwrap();
        assert_eq!(conversions.len(), 1);
        assert_eq!(conversions[0].0, "dense");
        assert_eq!(conversions[0].3, 0);
        assert_eq!(
            effective
                .headers
                .iter()
                .find(|h| h.name == "packed.weight")
                .unwrap()
                .shape,
            [4, 8]
        );
        assert_eq!(
            effective
                .headers
                .iter()
                .find(|h| h.name == "dense.weight")
                .unwrap()
                .shape,
            [4, 8]
        );
    }
    #[test]
    fn first_use_partition_keeps_original_current_creation_and_later_retained_owners_distinct() {
        let cfg = ZImageTransformerConfig::turbo();
        let target = crate::transformer::adapter_header_targets(&cfg, true)
            .into_iter()
            .find_map(|(_, header)| header.starts_with("all_x_embedder.").then_some(header))
            .expect("actual transformer factory exposes its image embedder");
        let source = Component {
            files: Vec::new(),
            headers: vec![
                header(&format!("{target}.weight"), HeaderDtype::F16, &[4, 64], 2),
                header(&format!("{target}.bias"), HeaderDtype::F16, &[4], 2),
            ],
        };
        let (effective, records) = source
            .load_quantized(&[(target.clone(), false)], 4)
            .unwrap();
        let (_, old, new, casts) = records[0].clone();
        let empty = Component {
            files: Vec::new(),
            headers: Vec::new(),
        };
        let facts = Facts {
            te_cfg: ZTextEncoderConfig::z_image(),
            dit_cfg: cfg,
            te: empty.clone(),
            dit: effective.clone(),
            vae: empty,
            control: None,
            window_tokens: 1,
            caption_tokens: 1,
            quant_bits: Some(4),
            streamable: false,
            load_components: None,
            quant_conversions: [Vec::new(), records, Vec::new()],
            te_quant_bits: None,
        };
        let mut seen = std::array::from_fn(|_| std::collections::BTreeSet::new());
        let structural = effective.resident(|_| true);
        assert_eq!(
            facts
                .quantization_owner_window(
                    IosPhase::Denoise,
                    q::DIT_PREPARE,
                    0,
                    structural,
                    true,
                    &mut seen
                )
                .unwrap(),
            (old, 0, true)
        );
        assert_eq!(
            facts
                .quantization_owner_window(
                    IosPhase::Denoise,
                    q::DIT_EMBED,
                    0,
                    structural,
                    true,
                    &mut seen
                )
                .unwrap(),
            (old, new + casts, true)
        );
        assert!(seen[1].contains(&target));
        assert_eq!(
            facts
                .quantization_owner_window(
                    IosPhase::Denoise,
                    q::DIT_BLOCK,
                    0,
                    structural,
                    true,
                    &mut seen
                )
                .unwrap(),
            (new, 0, false)
        );
        // A staged conditioner does not simultaneously own the not-yet-loaded heavy component.
        let mut unseen = std::array::from_fn(|_| std::collections::BTreeSet::new());
        assert_eq!(
            facts
                .quantization_owner_window(
                    IosPhase::Conditioning,
                    q::TE_EMBED,
                    0,
                    0,
                    true,
                    &mut unseen
                )
                .unwrap(),
            (0, 0, false)
        );
        let mut streamed = facts;
        streamed.streamable = true;
        streamed.quant_conversions[1][0].0 = "layers.0.attention.to_q".into();
        assert_eq!(
            streamed
                .quantization_owner_window(
                    IosPhase::Denoise,
                    q::DIT_BLOCK,
                    streamed.dit_cfg.n_refiner_layers,
                    0,
                    true,
                    &mut unseen
                )
                .unwrap(),
            (0, 0, false)
        );
        assert!(
            unseen[1].is_empty(),
            "unused resident lazy shells must not be marked as materialized replay payloads"
        );
    }

    fn raw_control_headers() -> Component {
        let mut headers = Vec::new();
        for (path, _) in control_quantization_targets() {
            headers.push(header(
                &format!("{path}.weight"),
                HeaderDtype::F16,
                &[4, 64],
                2,
            ));
            // Native control projections have biases; base adaLN consumes one when present.
            if path.ends_with("after_proj")
                || path.ends_with("before_proj")
                || path.ends_with("adaLN_modulation.0")
            {
                headers.push(header(&format!("{path}.bias"), HeaderDtype::F32, &[4], 4));
            }
        }
        headers.push(header(
            "control_all_x_embedder.2-1.weight",
            HeaderDtype::BF16,
            &[4, 132],
            2,
        ));
        headers.push(header(
            "control_layers.0.attention_norm1.weight",
            HeaderDtype::BF16,
            &[4],
            2,
        ));
        headers.push(header(
            "control_layers.0.unlisted_matrix.weight",
            HeaderDtype::F32,
            &[4, 64],
            4,
        ));
        Component {
            files: Vec::new(),
            headers,
        }
    }

    #[test]
    fn control_target_schema_uses_native_blocks_and_leaves_patch_embedder_dense() {
        let targets = control_quantization_targets()
            .into_iter()
            .map(|(path, _)| path)
            .collect::<std::collections::BTreeSet<_>>();
        let per_block = crate::transformer_block::adapter_target_paths().len() + 1;
        let count = (crate::control_transformer::CONTROL_LAYERS_PLACES.len()
            + crate::control_transformer::CONTROL_REFINER_PLACES.len())
            * per_block
            + 2;
        assert_eq!(
            (
                targets.len(),
                targets.contains("control_layers.0.before_proj"),
                targets.contains("control_noise_refiner.0.before_proj"),
                targets.contains("control_layers.1.before_proj"),
                targets.contains("control_all_x_embedder.2-1")
            ),
            (count, true, true, false, false),
        );
    }

    #[test]
    fn control_effective_headers_convert_f32_bias_without_replacing_raw_inputs() {
        let raw = raw_control_headers();
        let effective = raw.control_load_quantized(Some(8)).unwrap();
        let find = |source: &Component, name: &str| {
            let h = source
                .headers
                .iter()
                .find(|header| header.name == name)
                .unwrap();
            (h.dtype, h.shape.clone(), h.data_bytes)
        };
        assert_eq!(
            (
                find(&raw, "control_layers.0.after_proj.bias"),
                find(&effective, "control_layers.0.after_proj.bias"),
                find(&effective, "control_layers.0.after_proj.weight"),
                find(&effective, "control_all_x_embedder.2-1.weight"),
                find(&effective, "control_layers.0.attention_norm1.weight"),
                find(&effective, "control_layers.0.unlisted_matrix.weight")
            ),
            (
                (HeaderDtype::F32, vec![4], 16),
                (HeaderDtype::BF16, vec![4], 8),
                (HeaderDtype::U32, vec![4, 16], 256),
                (HeaderDtype::BF16, vec![4, 132], 1056),
                (HeaderDtype::BF16, vec![4], 8),
                (HeaderDtype::F32, vec![4, 64], 1024)
            ),
        );
    }

    #[test]
    fn control_mixed_leaf_pack_is_per_leaf_and_repeated_pack_is_noop() {
        // Independent packed preimage from the existing leaf producer, not from the new control
        // wrapper being tested; a wrapper-wide no-op must not become its own expected oracle.
        let packed = raw_control_headers()
            .load_quantized(&control_quantization_targets(), 4)
            .unwrap()
            .0;
        let mut mixed = packed.clone();
        let base = "control_layers.1.after_proj";
        mixed.headers.retain(|h| {
            ![
                format!("{base}.weight"),
                format!("{base}.scales"),
                format!("{base}.biases"),
            ]
            .contains(&h.name)
        });
        mixed.headers.push(header(
            &format!("{base}.weight"),
            HeaderDtype::BF16,
            &[4, 64],
            2,
        ));
        let repacked = mixed.control_load_quantized(Some(4)).unwrap();
        let repeated = packed.control_load_quantized(Some(4)).unwrap();
        let signature = |source: &Component| {
            source
                .headers
                .iter()
                .map(|h| (h.name.clone(), h.dtype, h.shape.clone(), h.data_bytes))
                .collect::<Vec<_>>()
        };
        let expected = signature(&packed);
        assert_eq!(
            (signature(&repacked), signature(&repeated)),
            (expected.clone(), expected)
        );
    }

    #[test]
    fn constructor_bias_policy_keeps_ignored_source_siblings_out_of_conversion_parents() {
        // These are native constructor targets, not a rank-2 packing heuristic. The extra F32 bias
        // is deliberately still in the raw/effective dictionary: ignoring it as a module input is
        // NOT a claim that the file Array/cache/native backing has been released.
        let cases: [(&str, fn(&str) -> bool, bool); 7] = [
            (
                "model.layers.0.self_attn.q_proj",
                |_| crate::quant::TEXT_LINEAR_HAS_BIAS,
                false,
            ),
            (
                "model.layers.0.mlp.gate_proj",
                |_| crate::quant::TEXT_LINEAR_HAS_BIAS,
                false,
            ),
            (
                "layers.0.attention.to_q",
                crate::quant::dit_quantization_has_bias,
                false,
            ),
            (
                "layers.0.feed_forward.w1",
                crate::quant::dit_quantization_has_bias,
                false,
            ),
            (
                "layers.0.adaLN_modulation.0",
                crate::quant::dit_quantization_has_bias,
                true,
            ),
            (
                "control_layers.0.before_proj",
                crate::quant::dit_quantization_has_bias,
                true,
            ),
            ("decoder.mid_block.attentions.0.to_q", |_| true, true),
        ];
        for bits in [4, 8] {
            for &(path, policy, module_bias) in &cases {
                let raw = Component {
                    files: Vec::new(),
                    headers: vec![
                        header(&format!("{path}.weight"), HeaderDtype::F16, &[4, 64], 2),
                        header(&format!("{path}.bias"), HeaderDtype::F32, &[4], 4),
                    ],
                };
                let (effective, records) = raw
                    .load_quantized_with_bias(&[(path.into(), false)], bits, policy)
                    .unwrap();
                let bias = effective
                    .headers
                    .iter()
                    .find(|h| h.name == format!("{path}.bias"))
                    .unwrap();
                let allocated = mlx_gen::ios_quantum::allocated_bytes;
                let weight = allocated(4 * 64 * 2);
                let old_bias = allocated(4 * 4);
                let converted_bias = allocated(4 * 2);
                let packed = allocated(4 * 64 * bits as u64 / 8) + 2 * allocated(4 * 2);
                assert_eq!(
                    (raw.headers[1].dtype, bias.dtype, bias.data_bytes, records),
                    (
                        HeaderDtype::F32,
                        if module_bias {
                            HeaderDtype::BF16
                        } else {
                            HeaderDtype::F32
                        },
                        if module_bias { 8 } else { 16 },
                        vec![(
                            path.into(),
                            weight + if module_bias { old_bias } else { 0 },
                            packed + if module_bias { converted_bias } else { 0 },
                            weight,
                        )],
                    ),
                    "constructor bias ownership for {path} Q{bits}",
                );
            }
        }
    }

    #[test]
    fn constructor_bias_policy_reaches_all_native_dit_and_control_target_families() {
        let cfg = ZImageTransformerConfig::turbo();
        let targets = crate::transformer::adapter_header_targets(&cfg, true)
            .into_iter()
            .map(|(_, path)| path)
            .chain(
                control_quantization_targets()
                    .into_iter()
                    .map(|(path, _)| path),
            );
        let mut kinds = std::collections::BTreeSet::new();
        for path in targets {
            let (kind, expected) = if path.contains(".attention.") {
                ("attention", false)
            } else if path.contains(".feed_forward.") {
                ("feed-forward", false)
            } else if path.ends_with(".before_proj") || path.ends_with(".after_proj") {
                ("control-projection", true)
            } else if path.contains(".adaLN_modulation.") {
                ("adaLN", true)
            } else {
                ("global", true)
            };
            assert_eq!(
                crate::quant::dit_quantization_has_bias(&path),
                expected,
                "{path}",
            );
            kinds.insert(kind);
        }
        assert_eq!(
            kinds,
            ["adaLN", "attention", "control-projection", "feed-forward", "global"]
                .into_iter()
                .collect(),
        );
    }

    #[test]
    fn dit_alias_conversion_uses_independent_suffix_origins_and_preserves_raw_inventory() {
        let from = "t_embedder.mlp.0";
        let to = "t_embedder.linear1";
        // Native aliases are independent for EVERY suffix. Target weights are deliberately
        // incompatible shadow inputs when a source weight exists; they are not module parents.
        for bits in [4, 8] {
            for (source_weight, source_bias, packed, source_scales) in [
                (true, false, false, false),
                (true, true, false, false),
                (false, true, false, false),
                (true, false, true, false),
                (true, true, true, true),
            ] {
                let mut headers = vec![
                    header(&format!("{to}.weight"), if source_weight { HeaderDtype::F32 } else { HeaderDtype::F16 }, &[if source_weight { 8 } else { 4 }, 64], if source_weight { 4 } else { 2 }),
                    header(&format!("{to}.bias"), if source_bias { HeaderDtype::F16 } else { HeaderDtype::F32 }, &[4], if source_bias { 2 } else { 4 }),
                ];
                if source_weight {
                    headers.push(header(&format!("{from}.weight"), if packed { HeaderDtype::U32 } else { HeaderDtype::F16 }, &[4, if packed { 64 * bits as usize / 32 } else { 64 }], if packed { 4 } else { 2 }));
                }
                if source_bias {
                    headers.push(header(&format!("{from}.bias"), HeaderDtype::F32, &[4], 4));
                }
                if packed {
                    headers.push(header(&format!("{to}.scales"), HeaderDtype::BF16, &[4, 1], 2));
                    headers.push(header(&format!("{to}.biases"), HeaderDtype::BF16, &[4, 1], 2));
                    if source_scales {
                        headers.push(header(&format!("{from}.scales"), HeaderDtype::BF16, &[4, 1], 2));
                    }
                }
                let raw = Component { files: Vec::new(), headers };
                let signature = |c: &Component| c.headers.iter().map(|h| (h.name.clone(), h.dtype, h.shape.clone(), h.data_bytes)).collect::<Vec<_>>();
                let original = signature(&raw);
                let (effective, records) = raw.load_quantized_with_bias(
                    &[(from.into(), false)], bits, crate::quant::dit_quantization_has_bias,
                ).unwrap();
                let mut module = effective.headers.iter().map(|h| (h.name.clone(), h.clone())).collect();
                crate::loader::remap_transformer_header_keys(&mut module);
                let bias_origin = if source_bias { format!("{from}.bias") } else { format!("{to}.bias") };
                let winner = effective.headers.iter().find(|h| h.name == bias_origin).unwrap();
                let expected_bias = if packed { HeaderDtype::F32 } else { HeaderDtype::BF16 };
                let expected_records = if packed { Vec::new() } else {
                    vec![(from.into(), 528, 4 * 64 * bits as u64 / 8 + 24, 512)]
                };
                assert_eq!(
                    (
                        module[&format!("{to}.weight")].shape.clone(),
                        module[&format!("{to}.weight")].dtype,
                        module[&format!("{to}.bias")].dtype,
                        winner.dtype,
                        records,
                        signature(&raw),
                    ),
                    (vec![4, 64 * bits as usize / 32], HeaderDtype::U32, expected_bias, expected_bias, expected_records, original),
                    "per-suffix source={source_weight}/{source_bias}, packed={packed}/{source_scales} Q{bits}",
                );
                if source_bias {
                    let shadow = effective.headers.iter().find(|h| h.name == format!("{to}.bias")).unwrap();
                    assert_eq!(shadow.dtype, HeaderDtype::F16, "shadowed target bias is not the converted source");
                }
            }
        }
    }

    #[test]
    fn dit_constructor_inventory_separates_dictionary_shell_and_loaded_payload_domains() {
        let raw = Component {
            files: Vec::new(),
            headers: vec![
                header("t_embedder.mlp.0.weight", HeaderDtype::F16, &[4, 64], 2),
                header("t_embedder.linear1.weight", HeaderDtype::F32, &[8, 64], 4),
                header("t_embedder.linear1.bias", HeaderDtype::F32, &[4], 4),
                header("layers.0.attention.to_q.weight", HeaderDtype::F16, &[4, 64], 2),
                header("layers.1.attention.to_q.weight", HeaderDtype::F16, &[4, 64], 2),
                header("unconsumed_matrix.weight", HeaderDtype::F32, &[2, 4096], 4),
            ],
        };
        let constructed = raw.dit_constructed_origins([
            "t_embedder.linear1.weight", "t_embedder.linear1.bias",
            "layers.0.attention.to_q.weight", "layers.1.attention.to_q.weight",
        ].into_iter().map(str::to_owned)).unwrap();
        assert_eq!(
            (
                constructed["t_embedder.linear1.weight"].as_str(),
                constructed["t_embedder.linear1.bias"].as_str(),
                constructed.keys().filter(|k| k.starts_with("layers.")).count(),
                raw.dit_materialized_module_bytes(&constructed, false),
                raw.dit_materialized_module_bytes(&constructed, true),
                raw.headers.len(),
            ),
            ("t_embedder.mlp.0.weight", "t_embedder.linear1.bias", 2, 1552, 528, 6),
            "retained shell graph/header inventory is NOT N materialized payloads; raw roots remain separate",
        );
    }

    // HOST-only: actual metadata conversion producer; no Array, model, payload or native work.
    fn history_facts() -> Facts {
        let cfg = ZImageTransformerConfig::turbo();
        let targets = crate::transformer::adapter_header_targets(&cfg, true);
        let embed = targets
            .iter()
            .find(|(_, path)| path.starts_with("all_x_embedder."))
            .unwrap()
            .1
            .clone();
        let output = targets
            .iter()
            .find(|(_, path)| path.starts_with("all_final_layer."))
            .unwrap()
            .1
            .clone();
        let convert = |paths: Vec<String>| {
            let source = Component {
                files: Vec::new(),
                headers: paths
                    .iter()
                    .map(|path| header(&format!("{path}.weight"), HeaderDtype::F16, &[4, 64], 2))
                    .collect(),
            };
            source
                .load_quantized(
                    &paths
                        .into_iter()
                        .map(|path| (path, false))
                        .collect::<Vec<_>>(),
                    4,
                )
                .unwrap()
        };
        let (dit, dit_conversions) = convert(vec![embed, output]);
        let leaf = crate::vae::attention::QUANTIZATION_LINEAR_PATHS[0];
        let (vae, vae_conversions) = convert(vec![
            format!("encoder.mid_block.attentions.0.{leaf}"),
            format!("decoder.mid_block.attentions.0.{leaf}"),
        ]);
        Facts {
            te_cfg: ZTextEncoderConfig::z_image(),
            dit_cfg: cfg,
            te: Component {
                files: Vec::new(),
                headers: Vec::new(),
            },
            dit,
            vae,
            control: None,
            window_tokens: 1,
            caption_tokens: 1,
            quant_bits: Some(4),
            streamable: false,
            load_components: None,
            quant_conversions: [Vec::new(), dit_conversions, vae_conversions],
            te_quant_bits: None,
        }
    }

    #[test]
    fn alternative_embed_inputs_start_from_the_same_cold_source_operation() {
        let facts = history_facts();
        let mut cursor = QuantizationCursor::new(QuantizationHistory::RequestFirst, false);
        let structural = facts.dit.resident(|_| true) + facts.vae.resident(|_| true);
        let cold = facts
            .quant_conversions
            .iter()
            .flatten()
            .map(|(_, old, _, _)| old)
            .sum::<u64>();
        let embed = facts.quant_conversions[1]
            .iter()
            .find(|(path, _, _, _)| path.starts_with("all_x_embedder."))
            .unwrap();
        let expected = Some((cold, embed.2 + embed.3, true));
        let first = cursor
            .window(&facts, IosPhase::Denoise, q::DIT_EMBED, 0, structural, true)
            .unwrap();
        let alternative = cursor
            .window(&facts, IosPhase::Denoise, q::DIT_EMBED, 1, structural, true)
            .unwrap();
        assert_eq!((first, alternative), (expected, expected));
    }

    #[test]
    fn recurrent_dit_callback_keeps_only_its_own_remaining_lazy_parents() {
        let facts = history_facts();
        let mut cursor = QuantizationCursor::new(callback_parameter_history(false, 0), false);
        let mut later_output = QuantizationCursor::new(callback_parameter_history(true, 1), false);
        let dit = facts.dit.resident(|_| true);
        let vae = facts.vae.resident(|_| true);
        let pending_vae = facts.quant_conversions[2]
            .iter()
            .map(|(_, old, _, _)| old)
            .sum::<u64>();
        let actual = cursor
            .window(&facts, IosPhase::Denoise, q::DIT_EMBED, 0, dit + vae, true)
            .unwrap();
        let later = later_output
            .window(&facts, IosPhase::Denoise, q::DIT_EMBED, 0, dit + vae, true)
            .unwrap();
        let expected = Some((dit + pending_vae, 0, true));
        assert_eq!((actual, later), (expected, expected));
    }

    #[test]
    fn zero_forward_first_and_late_decode_preserve_unconsumed_dit_and_encoder() {
        let facts = history_facts();
        let mut first = QuantizationCursor::new(QuantizationHistory::NoModelForward, false);
        let mut late = QuantizationCursor::new(QuantizationHistory::NoModelForward, true);
        let structural = facts.dit.resident(|_| true) + facts.vae.resident(|_| true);
        let cold = facts
            .quant_conversions
            .iter()
            .flatten()
            .map(|(_, old, _, _)| old)
            .sum::<u64>();
        let decoder = facts.quant_conversions[2]
            .iter()
            .find(|(path, _, _, _)| path.starts_with("decoder."))
            .unwrap();
        let absent = first
            .window(
                &facts,
                IosPhase::Denoise,
                q::DIT_EMBED,
                0,
                structural,
                false,
            )
            .unwrap();
        let a = first
            .window(
                &facts,
                IosPhase::Decode,
                q::VAE_DECODE_ATTENTION,
                0,
                structural,
                false,
            )
            .unwrap();
        let b = late
            .window(
                &facts,
                IosPhase::Decode,
                q::VAE_DECODE_ATTENTION,
                0,
                structural,
                false,
            )
            .unwrap();
        assert_eq!(
            (absent, a, b),
            (
                None,
                Some((cold, decoder.2 + decoder.3, true)),
                Some((cold - decoder.1 + decoder.2, 0, true))
            )
        );
    }

    #[test]
    fn streamed_replay_scratch_stays_cold_and_serial_owners_stay_inside_denoise() {
        let mut facts = history_facts();
        facts.streamable = true;
        let target = crate::transformer::adapter_header_targets(&facts.dit_cfg, true)
            .into_iter()
            .find(|(_, path)| path.starts_with("layers.0."))
            .unwrap()
            .1;
        let raw = Component {
            files: Vec::new(),
            headers: vec![header(
                &format!("{target}.weight"),
                HeaderDtype::F16,
                &[4, 64],
                2,
            )],
        };
        let (packed, conversions) = raw.load_quantized(&[(target, false)], 4).unwrap();
        // A source-record replay test, NOT a complete native block-price/backing certificate.
        // This matched conversion's cold read/output/cast contribution must survive every replay.
        let (_, old, new, casts) = conversions[0].clone();
        let scratch = old + new + casts;
        facts.dit.headers.extend(packed.headers);
        facts.quant_conversions[1].extend(conversions);
        let structural = facts
            .dit
            .resident(|name| crate::loader::dit_load_keeps(true, name))
            + facts.vae.resident(|_| true);
        let record = |phase, label: &'static str, point| Priced {
            label,
            resident: structural + 7,
            scratch,
            window: IosWindowPrice {
                phase,
                class: Some("host-source-record-only".into()),
                label: label.into(),
                resident_bytes: structural,
                held_bytes: 7,
                scratch_bytes: scratch,
                basis: None,
                owner_domain: "host-source-record-only".into(),
            },
            source_structural: structural,
            source_held: 7,
            source_point: point,
            source_vae: None,
        };
        let source = vec![
            record(IosPhase::Encode, q::INIT_IMAGE, 0),
            record(
                IosPhase::Denoise,
                q::DIT_BLOCK,
                facts.dit_cfg.n_refiner_layers,
            ),
            record(IosPhase::Decode, q::VAE_DECODE_IN, 0),
        ];
        let route = Route {
            mode: Mode::Reference,
            width: 16,
            height: 16,
            count: 2,
            execution: GenerationMemory::default(),
        };
        let first = facts
            .replay_quanta(
                &source,
                &route,
                QuantizationHistory::DenoiseRecurrent,
                0,
                false,
            )
            .unwrap();
        let last = facts
            .replay_quanta(
                &source,
                &route,
                QuantizationHistory::DenoiseRecurrent,
                1,
                false,
            )
            .unwrap();
        let larger_route = Route {
            count: 3,
            ..route.clone()
        };
        let later = facts
            .replay_quanta(
                &source,
                &larger_route,
                QuantizationHistory::DenoiseRecurrent,
                2,
                false,
            )
            .unwrap();
        let cold_first = facts
            .replay_quanta(
                &source,
                &route,
                callback_parameter_history(true, 0),
                0,
                false,
            )
            .unwrap();
        let single_route = Route {
            count: 1,
            ..route.clone()
        };
        let single = facts
            .replay_quanta(
                &source,
                &single_route,
                callback_parameter_history(true, 0),
                0,
                false,
            )
            .unwrap();
        let decode_final = facts
            .replay_quanta(
                &source,
                &larger_route,
                QuantizationHistory::NoModelForward,
                0,
                true,
            )
            .unwrap();
        let resident = facts
            .dit
            .resident(|name| crate::loader::dit_load_keeps(true, name))
            + facts.quant_conversions[2]
                .iter()
                .map(|(_, old, _, _)| old)
                .sum::<u64>();
        let a = first[1].as_ref().unwrap();
        let b = last[1].as_ref().unwrap();
        assert_eq!(
            (
                a.resident_bytes,
                b.resident_bytes,
                a.scratch_bytes,
                b.scratch_bytes,
                first
                    .iter()
                    .map(|window| window.as_ref().unwrap().held_bytes)
                    .collect::<Vec<_>>(),
                last.iter()
                    .map(|window| window.as_ref().unwrap().held_bytes)
                    .collect::<Vec<_>>(),
                later
                    .iter()
                    .map(|window| window.as_ref().unwrap().held_bytes)
                    .collect::<Vec<_>>(),
                b.class == later[1].as_ref().unwrap().class,
                cold_first[1].as_ref().unwrap().class != b.class,
                single[1]
                    .as_ref()
                    .unwrap()
                    .class
                    .as_deref()
                    .unwrap()
                    .contains("/serial-first-final/"),
                decode_final[2]
                    .as_ref()
                    .unwrap()
                    .class
                    .as_deref()
                    .unwrap()
                    .contains("/serial-later-final/"),
            ),
            (
                resident,
                resident,
                scratch,
                scratch,
                vec![7, 7, 7],
                vec![7, 7 + 4 * 16 * 2 * 2, 7],
                vec![7, 7 + 2 * 4 * 16 * 2 * 2, 7],
                true,
                true,
                true,
                true,
            ),
        );
    }

    #[test]
    fn native_zero_forward_and_curated_possible_families_are_source_bound() {
        let recipe = |scheduler: &str| {
            serde_json::from_str::<mlx_gen::sampler::flow_sampling::FlowSamplingRecipe>(&format!(
            "{{\"version\":1,\"solver\":\"euler\",\"scheduler\":{scheduler},\"steps\":4,\"guidance\":null,\"strength\":1.0,\"reference_strength\":null,\"use_pid\":false,\"pid_capture_sigma\":null}}"
        )).unwrap()
        };
        let native = mlx_gen::sampler::flow_sampling::ResolvedFlowControls::from_recipe(
            &recipe("null"),
            crate::sampling::defaults(),
            mlx_gen::TimestepConvention::OneMinusSigma,
        )
        .unwrap();
        let curated = mlx_gen::sampler::flow_sampling::ResolvedFlowControls::from_recipe(
            &recipe("\"beta\""),
            crate::sampling::defaults(),
            mlx_gen::TimestepConvention::OneMinusSigma,
        )
        .unwrap();
        let mut route = Route {
            mode: Mode::Reference,
            width: 16,
            height: 16,
            count: 1,
            execution: GenerationMemory::default(),
        };
        let zero = source_callback_presence(&route, &native).unwrap();
        let possible = source_callback_presence(&route, &curated).unwrap();
        route.mode = Mode::Text;
        let positive = source_callback_presence(&route, &native).unwrap();
        assert_eq!(
            (zero, possible, positive),
            ((false, true), (true, true), (true, false))
        );
    }

    #[test]
    fn staged_late_decode_drops_dit_stage_without_evaluating_unused_encoder() {
        let facts = history_facts();
        let mut late = QuantizationCursor::new(QuantizationHistory::NoModelForward, true);
        let encoder = facts.quant_conversions[2]
            .iter()
            .find(|(path, _, _, _)| path.starts_with("encoder."))
            .unwrap();
        let decoder = facts.quant_conversions[2]
            .iter()
            .find(|(path, _, _, _)| path.starts_with("decoder."))
            .unwrap();
        let actual = late
            .window(
                &facts,
                IosPhase::Decode,
                q::VAE_DECODE_IN,
                0,
                facts.vae.resident(|_| true),
                true,
            )
            .unwrap();
        assert_eq!(
            (
                actual,
                late.seen[1].is_empty(),
                late.seen[2].contains(&encoder.0)
            ),
            (Some((encoder.1 + decoder.2, 0, true)), true, false)
        );
    }
}
