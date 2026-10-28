// PRIVATE cfg(test) source reuse, included only by Z's fixture unit and catalog unit tests.
// No native arrays/device/counters, public exports, dependency edge, family policy or certificate.
use std::path::Path;
use mlx_gen::gen_core::{Conditioning, ControlKind, GenerationMemory, GenerationRequest,
    Image, LoadSpec, WeightsSource};
use mlx_gen::{Error, Result, Solver};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ManualMode { Text, Reference, Control, ControlReference }
impl ManualMode {
    fn control(self) -> bool { matches!(self, Self::Control | Self::ControlReference) }
    fn reference(self) -> bool { matches!(self, Self::Reference | Self::ControlReference) }
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Position { ResidentEndpoint, OptimizedEndpoint, Intermediate }

pub(crate) struct Inputs<'a> {
    pub provider: &'a str,
    pub mode: ManualMode,
    pub position: Position,
    pub original_spec: &'a LoadSpec,
    pub actual_text_source: &'a Path,
    pub actual_control_source: &'a Path,
    pub width: u32,
    pub height: u32,
    pub count: u32,
    pub execution: GenerationMemory,
    // Owned host-image values only. The actual Snapshot adapter may copy image BYTES to describe
    // the unchanged original borrowed argument; never a native Array/root or an execution change.
    pub hint: Image,
    pub init: Option<Image>,
    pub control_kind: ControlKind,
    pub configurations: [String; 4],
    pub original_factory_source: &'a str,
    pub actual_vocab_size: i32,
    pub te_explicit_window: Option<usize>,
    pub te_full_cover_fallback: Option<usize>,
    pub dit_selector_window: Option<usize>,
    pub actual_te_layers: usize,
    pub actual_base_blocks: usize,
    pub actual_control_blocks: Option<usize>,
}

pub(crate) struct Projection {
    pub provider: String,
    pub described_spec: LoadSpec,
    pub request: GenerationRequest,
    // Additional ACTUAL program inputs absent from the default request DTO. This is a private
    // execution description, not a public candidate or a certificate of public-path equivalence.
    pub program: serde_json::Value,
}

fn path_text(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| Error::Msg("fixture sidecar requires exact UTF-8 path metadata".into()))
}

pub(crate) fn project(input: Inputs<'_>) -> Result<Projection> {
    let WeightsSource::Dir(root) = &input.original_spec.weights else {
        return Err(Error::Msg("original fixture sidecar requires its literal directory source".into()));
    };
    if input.mode.reference() != input.init.is_some() {
        return Err(Error::Msg("manual init image and actual route disagree".into()));
    }
    // Consistency of the caller's PRIVATE position description, not an endpoint selector or a
    // proof of optimality. Endpoint selection itself remains the real provider/catalog helper.
    let form_matches = match input.position {
        Position::ResidentEndpoint => input.original_spec.offload_policy == mlx_gen::gen_core::OffloadPolicy::Resident
            && input.original_spec.load_shape == mlx_gen::gen_core::LoadShape::EagerMaterialization
            && !input.execution.stage_residency && !input.execution.stream_transformer_blocks,
        Position::OptimizedEndpoint => input.original_spec.offload_policy == mlx_gen::gen_core::OffloadPolicy::Sequential
            && input.original_spec.load_shape == mlx_gen::gen_core::LoadShape::DeferredMaterialization
            && input.execution.stage_residency && input.execution.stream_transformer_blocks,
        Position::Intermediate => input.original_spec.offload_policy == mlx_gen::gen_core::OffloadPolicy::Sequential
            && input.original_spec.load_shape == mlx_gen::gen_core::LoadShape::DeferredMaterialization
            && input.execution == GenerationMemory::default(),
    };
    if !form_matches { return Err(Error::Msg("private position does not match actual load/execution form".into())); }
    let tokens = original_token_program(input.original_factory_source, input.actual_vocab_size)?;
    let dit_literal = root.join("transformer");
    let vae_literal = root.join("vae");
    let text_source = path_text(input.actual_text_source)?;
    let control_source = path_text(input.actual_control_source)?;
    let mut described_spec = input.original_spec.clone();
    described_spec.control = input.mode.control().then(||
        WeightsSource::File(input.actual_control_source.to_owned()));
    let mut conditioning = Vec::new();
    let unused_hint = if input.mode.control() {
        conditioning.push(Conditioning::Control {
            image: input.hint, kind: input.control_kind, scale: Some(0.8),
        });
        None
    } else {
        // The original Text/Reference call still passes a hint argument, but never consumes it.
        // Record provenance without inventing a conditioning carrier or copying its pixel payload.
        Some(serde_json::json!({"width": input.hint.width, "height": input.hint.height,
            "consumed_by_original_body": false}))
    };
    if let Some(image) = input.init {
        conditioning.push(Conditioning::Reference { image, strength: None });
    }
    let position = match input.position {
        Position::ResidentEndpoint => "resident_endpoint",
        Position::OptimizedEndpoint => "optimized_endpoint",
        Position::Intermediate => "intermediate_deferred_full_cover_unstaged",
    };
    let request = GenerationRequest {
        width: input.width, height: input.height, count: input.count,
        memory: Some(input.execution), seed: Some(7), steps: Some(3),
        sampler: Some(Solver::Euler.name().into()),
        conditioning,
        // No authored strength is invented to make manual start=1 match public init resolution.
        strength: None,
        ..Default::default()
    };
    let program = serde_json::json!({
        "scope": "private original fixture execution description, NOT public catalog-path replay",
        "provider": input.provider, "mode": format!("{:?}", input.mode), "position": position,
        "actual_sources": {"text_encoder": text_source,
            "transformer": path_text(&dit_literal)?, "vae": path_text(&vae_literal)?,
            "control": input.mode.control().then_some(control_source)},
        "original_spec_control": format!("{:?}", input.original_spec.control),
        "actual_load_form": {"offload": format!("{:?}", input.original_spec.offload_policy),
            "shape": format!("{:?}", input.original_spec.load_shape),
            "precision": format!("{:?}", input.original_spec.precision),
            "heavy_quant": format!("{:?}", input.original_spec.quantize), "TE_load_quant": null},
        "manual_flow": {"factory": "FlowMatchEuler::for_static_shift", "steps": 3_u32,
            "shift": 3.0_f32, "shift_bits": 3.0_f32.to_bits(),
            "start": usize::from(input.mode.reference()), "seed": 7_u64,
            "control_scale": input.mode.control().then_some(0.8_f32),
            "control_scale_bits": input.mode.control().then_some(0.8_f32.to_bits()),
            "hint_already_preprocessed": input.mode.control(), "authored_strength": null},
        "geometry": {"width": input.width, "height": input.height, "count": input.count},
        "execution": format!("{:?}", input.execution),
        "windows": {"resolver_provider": "z_image_turbo",
            "TE_explicit": input.te_explicit_window, "TE_full_cover_fallback": input.te_full_cover_fallback,
            "DIT_selector": input.dit_selector_window, "actual_TE_layers": input.actual_te_layers,
            "actual_base_blocks": input.actual_base_blocks, "actual_control_blocks": input.actual_control_blocks},
        "actual_constructor_configurations": input.configurations, "unused_hint_argument": unused_hint,
        "tokens": tokens
    });
    Ok(Projection { provider: input.provider.into(), described_spec, request, program })
}


// Source-described HOST token values only. This does NOT read an Array/native payload, alter the
// original token factory, or claim the description was a runtime capture.
pub(crate) fn source_i32_values_hash(values: &[i32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for value in values { hash.update(value.to_le_bytes()); }
    format!("{:x}", hash.finalize())
}

pub(crate) fn original_token_program(source: &str, vocab_size: i32) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    fn constant(source: &str, name: &str) -> Result<i32> {
        let prefix = format!("const {name}: i32 = ");
        let start = source.find(prefix.as_str())
            .ok_or_else(|| Error::Msg(format!("missing original token constant {name}")))? + prefix.len();
        let end = source[start..].find(';')
            .ok_or_else(|| Error::Msg("unterminated original token constant".into()))? + start;
        source[start..end].trim().parse::<i32>()
            .map_err(|_| Error::Msg("original token constant is not literal i32 metadata".into()))
    }
    let window = constant(source, "WINDOW")?;
    let valid = constant(source, "VALID")?;
    if window <= 0 || valid < 0 || valid > window || vocab_size <= 0 {
        return Err(Error::Msg("invalid private original token dimensions/vocabulary".into()));
    }
    let begin = source.find("pub(crate) fn run_request(")
        .ok_or_else(|| Error::Msg("missing original request source".into()))?;
    let end = source[begin..].find("\nfn largest_by_label")
        .ok_or_else(|| Error::Msg("missing original request source end".into()))? + begin;
    let original = &source[begin..end];
    for actual_factory in [
        "let ids: Vec<i32> = (0..WINDOW).map(|i| 3 + i * 7).collect();",
        "let mask: Vec<i32> = (0..WINDOW).map(|i| i32::from(i < VALID)).collect();",
        "&Array::from_slice(&ids, &[1, WINDOW])",
        "&Array::from_slice(&mask, &[1, WINDOW])",
    ] {
        if !original.contains(actual_factory) {
            return Err(Error::Msg("original token factory/layout changed; source description must be revalidated".into()));
        }
    }
    let ids = (0..window).map(|i| i.checked_mul(7).and_then(|n| n.checked_add(3))
        .ok_or_else(|| Error::Msg("original token ID arithmetic overflow".into())))
        .collect::<Result<Vec<_>>>()?;
    if ids.iter().any(|&id| id < 0 || id >= vocab_size) {
        return Err(Error::Msg("source-described IDs exceed actual embedding vocabulary".into()));
    }
    let mask = (0..window).map(|i| i32::from(i < valid)).collect::<Vec<_>>();
    let mut factory = Sha256::new();
    factory.update(original.as_bytes());
    factory.update(format!("WINDOW={window};VALID={valid};vocab={vocab_size}").as_bytes());
    Ok(serde_json::json!({
        "provenance": "source-described original HOST values, NOT runtime Array/payload capture",
        "source_factory_scope_sha256": format!("{:x}", factory.finalize()),
        "window": window, "valid_tokens": valid, "caption_valid_prefix": valid,
        "shape": [1, window], "layout": "row-major contiguous",
        "value_hash_encoding": "little-endian i32 source values, not runtime Buffer bytes",
        "input_ids": {"dtype": "i32", "construction": "3 + i * 7 for every i in 0..WINDOW",
            "values_sha256": source_i32_values_hash(&ids), "values": ids},
        "attention_mask": {"dtype": "i32", "construction": "i32::from(i < VALID)",
            "values_sha256": source_i32_values_hash(&mask), "values": mask},
        "padding": {"policy": "right attention-mask zeros after valid prefix",
            "masked_tail_count": window - valid, "masked_ids": "same formula; not replaced with a PAD ID",
            "pad_token_id_applied": null},
        "vocabulary": {"size": vocab_size, "dependency": "actual Snapshot.te_cfg.vocab_size / validated embedding contract",
            "range": "0 <= every source-created ID < vocabulary size"}
    }))
}
