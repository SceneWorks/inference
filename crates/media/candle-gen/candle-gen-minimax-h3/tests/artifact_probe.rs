//! Bounded **actual-checkpoint numeric probe** — candle consumer (sc-26679 terminal repair).
//!
//! # What this proves, and what it refuses to claim
//!
//! This module consumes a manifest produced OUTSIDE the repository by the SceneWorks staging /
//! extractor toolchain. It reads ONLY the declared tensor ranges from an actual frozen artifact and
//! compares the **native production functions** (`quant::lin(..).forward`, `CurveAdaLn`) against a
//! pinned CPU-oracle expectation that is bound to those exact selected payload bytes.
//!
//! It does NOT implement an arithmetic oracle, does not re-run the producer's math, does not load a
//! complete model, and makes no claim about avalanche/audio-video generation. The pinned oracle is
//! produced by the existing pinned producers in `tests/fixtures/*.json` / the frozen external
//! references (`comfy-model-tools` ConvRot, `comfy-kitchen` W4A8, the WanGP pruned-curve sources) or
//! one owned extension thereof; the manifest records that producer identity and the oracle file is
//! sha256-bound.
//!
//! The whole-file digest is NOT recomputed (the actual artifact is tens of gigabytes). Trust is
//! bound through the sha256'd staging-receipt projection, which itself names the campaign's full
//! staging receipt + sha256 (see `artifact_probe_support`). Selected payload bytes are still hashed
//! independently and compared to the oracle's per-tensor digests.
//!
//! Ordinary CI runs only the tiny-fixture parser/preflight tests below; the numeric test is
//! `#[ignore]`d and is selected explicitly:
//!
//! ```text
//! H3_ACTUAL_PROBE_MANIFEST=/outside/manifest.json \
//! H3_ACTUAL_PROBE_RECEIPT_DIR=/outside/receipts \
//! cargo test -p candle-gen-minimax-h3 --test integration -- \
//!   artifact_probe::actual_checkpoint_projections_match_pinned_cpu_oracle --exact --ignored
//! ```
//!
//! On a candle build without `cuda`/`metal` the device is a labeled CPU fallback and is never a
//! device proof; packed ConvRot/W4A8 natives refuse on such a build by construction.

#[path = "support/artifact_probe_support.rs"]
mod support;

use std::collections::{BTreeMap, HashMap};
use std::io::Seek;
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::weightsmeta::{safetensors_reader_tensor_locations, Dtype as MetaDtype};
use candle_gen::Weights;
use candle_gen_minimax_h3::dit::CurveAdaLn;
use candle_gen_minimax_h3::{quant, MiniMaxH3DitConfig};
use gen_core_h3::pruned_adaln::PrunedAdaLn;
use support::TABLE_KEY;
use support::{
    receipt_file_name, run_probe, run_probe_at, Case, DeviceInfo, HeaderIndex, IndexedTensor,
    Manifest, NativeOutput, PResult, ProbeError, SelectedPayload, KIND_CONVROT, KIND_INTERPOLATION,
    KIND_MIXED, KIND_REDUCED, KIND_W4A8, RECEIPT_ENV,
};

/// Canonical packed operand sets, in the native loaders' own dialect.
const W4A8_OPERANDS: &[(&str, &str)] = &[
    ("weight", "I8"),
    ("weight_codebook", "F32"),
    ("weight_s_rel", "F8_E4M3"),
    ("weight_s_channel", "F32"),
    ("comfy_quant", "U8"),
];
const CONVROT_OPERANDS: &[(&str, &str)] = &[
    ("weight", "I8"),
    ("weight_scale", "F32"),
    ("comfy_quant", "U8"),
];

/// Header-only index: the shared, validated safetensors layout reader. Never reads payload.
fn load_index(path: &Path) -> PResult<HeaderIndex> {
    let file = std::fs::File::open(path).map_err(|error| {
        ProbeError::new(
            "artifact_missing",
            format!("cannot open {}: {error}", path.display()),
        )
    })?;
    let file_len = file
        .metadata()
        .map_err(|error| {
            ProbeError::new("artifact_missing", format!("artifact metadata: {error}"))
        })?
        .len();
    // No BufReader: the shared reader reads the 8-byte length and exactly the header bytes, and
    // `stream_position` below proves the header-only boundary was not crossed by prefetching.
    let mut file = file;
    let layout = safetensors_reader_tensor_locations(&mut file, file_len, path)
        .map_err(|error| ProbeError::new("artifact_header_invalid", error.to_string()))?;
    let position = file
        .stream_position()
        .map_err(|error| ProbeError::new("artifact_header_invalid", error.to_string()))?;
    if position != layout.data_start {
        return Err(ProbeError::new(
            "artifact_header_invalid",
            format!(
                "header reader stopped at {position}, payload starts at {}",
                layout.data_start
            ),
        ));
    }
    let mut index = HeaderIndex {
        file_len: layout.file_len,
        header_bytes_read: position,
        tensors: BTreeMap::new(),
    };
    for tensor in layout.tensors {
        index.insert(IndexedTensor {
            name: tensor.header.name,
            dtype: tensor.header.dtype.as_str().to_owned(),
            shape: tensor.header.shape,
            data_bytes: tensor.header.data_bytes,
            file_offset: tensor.file_offset,
        });
    }
    Ok(index)
}

fn require_dtype(index: &HeaderIndex, raw: &str, expected: &[&str], what: &str) -> PResult<()> {
    let tensor = index.require(raw)?;
    if !expected.contains(&tensor.dtype.as_str()) {
        return Err(ProbeError::new(
            "selection_bounds",
            format!(
                "{what} {raw:?} has dtype {}, expected one of {expected:?}",
                tensor.dtype
            ),
        ));
    }
    Ok(())
}

fn resolve(case: &Case, index: &HeaderIndex, manifest: &Manifest) -> PResult<Vec<String>> {
    let raw_of = |canonical: &str| case.select.raw_of(canonical);
    match case.kind.as_str() {
        KIND_W4A8 | KIND_CONVROT => {
            let base = case.select.base.as_deref().expect("parsed base");
            let companions: &[(&str, &str)] = match case.kind.as_str() {
                KIND_W4A8 => W4A8_OPERANDS,
                KIND_CONVROT => CONVROT_OPERANDS,
                _ => unreachable!("outer match restricts kinds"),
            };
            let mut names = Vec::with_capacity(companions.len() + usize::from(case.select.bias));
            for (suffix, dtype) in companions {
                let raw = raw_of(&format!("{base}.{suffix}"));
                require_dtype(index, &raw, &[*dtype], &case.id)?;
                names.push(raw);
            }
            // `scales`/`biases` affine siblings are a contradictory operand set for a native
            // packed base; refuse them here, before any payload read. (The dense-vs-ConvRot
            // contrast lives inside the pinned ConvRot oracle's `mixedLayer` companion leg; there
            // is no standalone dense kind and no standalone dense fidelity claim.)
            for forbidden in ["scales", "biases"] {
                if index
                    .tensors
                    .contains_key(&raw_of(&format!("{base}.{forbidden}")))
                {
                    return Err(ProbeError::new(
                        "selection_bounds",
                        format!(
                            "{}: {base}.{forbidden} contradicts the packed native operand set",
                            case.id
                        ),
                    ));
                }
            }
            if case.select.bias {
                let raw = raw_of(&format!("{base}.bias"));
                require_dtype(index, &raw, &["F16", "BF16", "F32"], &case.id)?;
                names.push(raw);
            }
            Ok(names)
        }
        KIND_MIXED => {
            let primary = case.select.base.as_deref().expect("parsed base");
            let convrot = case
                .select
                .convrot_base
                .as_deref()
                .expect("parsed convrot_base");
            if let Some(expected) = case.select.expect_route.as_deref() {
                if expected != "w4a8" {
                    return Err(ProbeError::new(
                        "selection_bounds",
                        format!(
                            "{}: mixed case expects the primary (W4A8) route; \
                             expect_route={expected}",
                            case.id
                        ),
                    ));
                }
            }
            let mut names = Vec::new();
            for (base, operands) in [(primary, W4A8_OPERANDS), (convrot, CONVROT_OPERANDS)] {
                for (suffix, dtype) in operands {
                    let raw = raw_of(&format!("{base}.{suffix}"));
                    require_dtype(index, &raw, &[*dtype], &case.id)?;
                    names.push(raw);
                }
                for forbidden in ["scales", "biases"] {
                    if index
                        .tensors
                        .contains_key(&raw_of(&format!("{base}.{forbidden}")))
                    {
                        return Err(ProbeError::new(
                            "selection_bounds",
                            format!(
                                "{}: {base}.{forbidden} contradicts the packed native operand set",
                                case.id
                            ),
                        ));
                    }
                }
            }
            for (base, wanted) in [
                (primary, case.select.w4a8_bias),
                (convrot, case.select.convrot_bias),
            ] {
                if wanted {
                    let raw = raw_of(&format!("{base}.bias"));
                    require_dtype(index, &raw, &["F16", "BF16", "F32"], &case.id)?;
                    names.push(raw);
                }
            }
            Ok(names)
        }
        KIND_INTERPOLATION | KIND_REDUCED => {
            if !case.select.topology {
                return Err(ProbeError::new(
                    "selection_bounds",
                    format!(
                        "{}: candle pruned cases require select.topology=true (CurveAdaLn assembles \
                         the complete validated topology)",
                        case.id
                    ),
                ));
            }
            if !manifest.dit_explicit {
                return Err(ProbeError::new(
                    "selection_bounds",
                    format!(
                        "{}: candle pruned cases require manifest.dit bound from the actual source \
                         config, not shipped defaults",
                        case.id
                    ),
                ));
            }
            let topology = pruned_topology(case, index, manifest)?;
            let canonical: Vec<String> = topology
                .operands()
                .iter()
                .map(|operand| operand.name.clone())
                .collect();
            match case.kind.as_str() {
                KIND_INTERPOLATION => {
                    let table = case.select.table.as_deref().expect("parsed table");
                    if table != TABLE_KEY || !canonical.iter().any(|name| name == TABLE_KEY) {
                        return Err(ProbeError::new(
                            "selection_bounds",
                            format!("{}: table must be canonical {TABLE_KEY:?}", case.id),
                        ));
                    }
                    let raw = raw_of(TABLE_KEY);
                    let tensor = index.require(&raw)?;
                    if let Some(grid) = case.select.grid {
                        if tensor.shape.first() != Some(&grid) {
                            return Err(ProbeError::new(
                                "selection_bounds",
                                format!("{}: table grid mismatch {:?}", case.id, tensor.shape),
                            ));
                        }
                    }
                    if let Some(rank) = case.select.rank {
                        if tensor.shape.get(1) != Some(&rank) {
                            return Err(ProbeError::new(
                                "selection_bounds",
                                format!("{}: table rank mismatch {:?}", case.id, tensor.shape),
                            ));
                        }
                    }
                }
                _ => {
                    let prefix = case.select.prefix.as_deref();
                    let canonical_weight = match (&case.select.weight, prefix) {
                        (Some(weight), _) => weight.clone(),
                        (None, Some(prefix)) => format!("{prefix}.weight"),
                        (None, None) => unreachable!("parsed"),
                    };
                    let canonical_bias = match (&case.select.bias_tensor, prefix) {
                        (Some(bias), _) => bias.clone(),
                        (None, Some(prefix)) => format!("{prefix}.bias"),
                        (None, None) => unreachable!("parsed"),
                    };
                    for name in [&canonical_weight, &canonical_bias] {
                        if !canonical.iter().any(|operand| operand == name) {
                            return Err(ProbeError::new(
                                "selection_bounds",
                                format!("{}: {name:?} is not a canonical pruned operand", case.id),
                            ));
                        }
                    }
                    if let Some(rank) = case.select.rank {
                        let tensor = index.require(&raw_of(&canonical_weight))?;
                        if tensor.shape.get(1) != Some(&rank) {
                            return Err(ProbeError::new(
                                "selection_bounds",
                                format!(
                                    "{}: reduced weight rank mismatch {:?}",
                                    case.id, tensor.shape
                                ),
                            ));
                        }
                    }
                    if let Some(out_features) = case.select.out_features {
                        let tensor = index.require(&raw_of(&canonical_weight))?;
                        if tensor.shape.first() != Some(&out_features) {
                            return Err(ProbeError::new(
                                "selection_bounds",
                                format!(
                                    "{}: reduced weight out_features mismatch {:?}",
                                    case.id, tensor.shape
                                ),
                            ));
                        }
                    }
                }
            }
            Ok(canonical.iter().map(|name| raw_of(name)).collect())
        }
        other => Err(ProbeError::new(
            "manifest_malformed",
            format!("{}: unknown kind {other}", case.id),
        )),
    }
}

fn pruned_topology(case: &Case, index: &HeaderIndex, manifest: &Manifest) -> PResult<PrunedAdaLn> {
    let dit = manifest.dit;
    PrunedAdaLn::inspect(
        dit.hidden_size,
        dit.num_layers,
        dit.time_embed_dim,
        |canonical| {
            let raw = case.select.raw_of(canonical);
            index.tensors.get(&raw).and_then(|tensor| {
                MetaDtype::parse(&tensor.dtype).map(|dtype| (dtype, tensor.shape.clone()))
            })
        },
    )
    .map_err(|error| ProbeError::new("pruned_topology_invalid", error))?
    .ok_or_else(|| {
        ProbeError::new(
            "not_pruned_topology",
            format!(
                "{}: artifact does not carry the complete direct pruned AdaLN topology",
                case.id
            ),
        )
    })
}

fn config(dit: &support::DitGeometry) -> MiniMaxH3DitConfig {
    MiniMaxH3DitConfig {
        hidden_size: dit.hidden_size,
        num_layers: dit.num_layers,
        time_embed_dim: dit.time_embed_dim,
        ..Default::default()
    }
}

fn candle_dtype(dtype: &str) -> PResult<DType> {
    Ok(match dtype {
        "U8" => DType::U8,
        "U32" => DType::U32,
        "I16" => DType::I16,
        "I32" => DType::I32,
        "I64" => DType::I64,
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        "F8_E4M3" => DType::F8E4M3,
        // Candle has no signed-byte dtype. The shared native reader preserves an I8 operand's
        // exact two's-complement bytes as U8 (see `Weights::from_files_native_int8`); the native
        // ConvRot/W4A8 loaders consume that representation.
        "I8" => DType::U8,
        other => {
            return Err(ProbeError::new(
                "unsupported_dtype",
                format!("candle cannot materialize stored dtype {other}"),
            ))
        }
    })
}

fn tensor_map(selected: &[SelectedPayload], device: &Device) -> PResult<HashMap<String, Tensor>> {
    let mut map = HashMap::with_capacity(selected.len());
    for payload in selected {
        let dtype = candle_dtype(&payload.dtype)?;
        let tensor = Tensor::from_raw_buffer(&payload.bytes, dtype, &payload.shape, device)
            .map_err(|error| {
                ProbeError::new(
                    "native_error",
                    format!("{}: materialize {}: {error}", payload.name, payload.dtype),
                )
            })?;
        if map.insert(payload.canonical.clone(), tensor).is_some() {
            return Err(ProbeError::new(
                "selection_bounds",
                format!("canonical tensor {} is bound twice", payload.canonical),
            ));
        }
    }
    Ok(map)
}

#[cfg(feature = "cuda")]
fn selected_device() -> PResult<(Device, DeviceInfo)> {
    let device = Device::new_cuda(0)
        .map_err(|error| ProbeError::new("device_unavailable", format!("CUDA device: {error}")))?;
    let info = DeviceInfo {
        kind: "cuda".into(),
        label: format!("{:?}", device.location()),
        index: Some(0),
        cpu_fallback: false,
        visible_devices: std::env::var("CUDA_VISIBLE_DEVICES").ok(),
        note: Some(
            "candle pinned fork exposes ordinal/location but no CUDA UUID; CUDA_VISIBLE_DEVICES \
             is recorded verbatim"
                .into(),
        ),
    };
    Ok((device, info))
}

#[cfg(all(not(feature = "cuda"), feature = "metal"))]
fn selected_device() -> PResult<(Device, DeviceInfo)> {
    let device = Device::new_metal(0)
        .map_err(|error| ProbeError::new("device_unavailable", format!("Metal device: {error}")))?;
    let info = DeviceInfo {
        kind: "metal".into(),
        label: format!("{:?}", device.location()),
        index: Some(0),
        cpu_fallback: false,
        visible_devices: None,
        note: None,
    };
    Ok((device, info))
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn selected_device() -> PResult<(Device, DeviceInfo)> {
    Ok((
        Device::Cpu,
        DeviceInfo {
            kind: "cpu".into(),
            label: "Cpu".into(),
            index: None,
            cpu_fallback: true,
            visible_devices: std::env::var("CUDA_VISIBLE_DEVICES").ok(),
            note: Some(
                "this candle build has no cuda/metal feature; CPU fallback is NOT device proof"
                    .into(),
            ),
        },
    ))
}

fn input_tensor(case: &Case, device: &Device) -> PResult<Tensor> {
    let (shape, values) = case.require_input()?;
    let tensor = Tensor::from_vec(values.to_vec(), shape.to_vec(), device)
        .map_err(|error| ProbeError::new("input_shape", format!("input tensor: {error}")))?;
    match case.input_dtype.as_str() {
        "F32" => Ok(tensor),
        "BF16" => tensor
            .to_dtype(DType::BF16)
            .map_err(|error| ProbeError::new("input_shape", format!("BF16 input: {error}"))),
        other => Err(ProbeError::new(
            "manifest_malformed",
            format!("unsupported input dtype {other}"),
        )),
    }
}

fn finish(
    output: Tensor,
    route: Option<String>,
    detail: Option<String>,
    device: DeviceInfo,
) -> PResult<NativeOutput> {
    let output = output
        .to_dtype(DType::F32)
        .map_err(|error| ProbeError::new("native_error", format!("cast output: {error}")))?;
    let shape = output.dims().to_vec();
    let values = output
        .flatten_all()
        .map_err(|error| ProbeError::new("native_error", format!("flatten output: {error}")))?
        .to_vec1::<f32>()
        .map_err(|error| ProbeError::new("native_error", format!("read output: {error}")))?;
    Ok(NativeOutput {
        shape,
        values,
        route,
        detail,
        device,
    })
}

fn projection(
    case: &Case,
    selected: &[SelectedPayload],
    device: &Device,
    device_info: &DeviceInfo,
) -> PResult<NativeOutput> {
    let base = case.select.base.as_deref().expect("parsed base");
    let weights = Weights::from_map(tensor_map(selected, device)?);
    let tiered = quant::lin(&weights, quant::DIT, base, case.select.bias, DType::F32)
        .map_err(|error| ProbeError::new("native_error", format!("quant::lin({base}): {error}")))?;
    let route = if tiered.linear.is_w4a8() {
        "w4a8"
    } else if tiered.linear.is_convrot_int8() {
        "convrot"
    } else if tiered.linear.is_packed() {
        "affine"
    } else {
        "dense"
    };
    if let Some(expected) = case.select.expect_route.as_deref() {
        if expected != route {
            return Err(ProbeError::new(
                "route_mismatch",
                format!("{base}: expected route {expected}, native selected {route}"),
            ));
        }
    }
    let input = input_tensor(case, device)?;
    let output = tiered
        .linear
        .forward(&input)
        .map_err(|error| ProbeError::new("native_error", format!("{base} forward: {error}")))?;
    let detail = format!(
        "quant::lin compute_dtype=F32 base_shape={:?} base_bytes={}",
        tiered.linear.base_shape(),
        tiered.base_bytes
    );
    finish(
        output,
        Some(route.into()),
        Some(detail),
        device_info.clone(),
    )
}

fn pruned(
    case: &Case,
    manifest: &Manifest,
    selected: &[SelectedPayload],
    device: &Device,
    device_info: &DeviceInfo,
) -> PResult<NativeOutput> {
    let operands = tensor_map(selected, device)?;
    let config = config(&manifest.dit);
    let weights = Weights::from_map(operands.clone());
    let topology = CurveAdaLn::inspect(&weights, &config)
        .map_err(|error| ProbeError::new("pruned_topology_invalid", error.to_string()))?
        .ok_or_else(|| {
            ProbeError::new(
                "not_pruned_topology",
                format!(
                    "{}: artifact is not a direct pruned AdaLN checkpoint",
                    case.id
                ),
            )
        })?;
    let curve = CurveAdaLn::assemble(topology, operands, &config, device).map_err(|error| {
        ProbeError::new("native_error", format!("CurveAdaLn::assemble: {error}"))
    })?;
    if case.kind == KIND_INTERPOLATION {
        let timesteps = case.require_timesteps()?;
        let coordinates = curve
            .coordinates(timesteps, device)
            .map_err(|error| ProbeError::new("native_error", format!("coordinates: {error}")))?;
        return finish(
            coordinates,
            Some("curve-ada-ln".into()),
            Some(format!(
                "CurveAdaLn grid={} rank={} timesteps={}",
                curve.grid(),
                curve.rank(),
                timesteps.len()
            )),
            device_info.clone(),
        );
    }
    let (shape, values) = case.require_input()?;
    let coordinates = Tensor::from_vec(values.to_vec(), shape.to_vec(), device)
        .map_err(|error| ProbeError::new("input_shape", format!("coordinates: {error}")))?;
    let projected = match case
        .select
        .weight
        .clone()
        .or_else(|| {
            case.select
                .prefix
                .as_ref()
                .map(|prefix| format!("{prefix}.weight"))
        })
        .as_deref()
    {
        Some("norm_out.linear.weight") => curve.norm_out_projection(),
        Some(weight) => {
            let layer = weight
                .strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.strip_suffix(".adaln_proj.linear.weight"))
                .and_then(|index| index.parse::<usize>().ok())
                .ok_or_else(|| {
                    ProbeError::new(
                        "selection_bounds",
                        format!("{}: cannot derive block index from {weight:?}", case.id),
                    )
                })?;
            curve.block_projection(layer)
        }
        None => unreachable!("parsed"),
    }
    .map_err(|error| ProbeError::new("native_error", format!("CurveAdaLn projection: {error}")))?;
    let output = projected
        .forward(&coordinates)
        .map_err(|error| ProbeError::new("native_error", format!("projection forward: {error}")))?;
    finish(
        output,
        Some("curve-ada-ln".into()),
        Some(format!(
            "CurveProjection compute_dtype={:?}",
            projected.compute_dtype()
        )),
        device_info.clone(),
    )
}

/// One Weights map carrying a packed W4A8 operand set AND a ConvRot INT8 operand set. Both native
/// loaders run over the SAME map, each must keep its own dialect, and the native outputs are
/// concatenated `[w4a8, convrot]` along the last axis to match the pinned producers' combined
/// expectation.
fn mixed(
    case: &Case,
    selected: &[SelectedPayload],
    device: &Device,
    device_info: &DeviceInfo,
) -> PResult<NativeOutput> {
    let primary_base = case.select.base.as_deref().expect("parsed base");
    let convrot_base = case
        .select
        .convrot_base
        .as_deref()
        .expect("parsed convrot_base");
    let weights = Weights::from_map(tensor_map(selected, device)?);
    let primary = quant::lin(
        &weights,
        quant::DIT,
        primary_base,
        case.select.w4a8_bias,
        DType::F32,
    )
    .map_err(|error| {
        ProbeError::new(
            "native_error",
            format!("quant::lin({primary_base}): {error}"),
        )
    })?;
    if !primary.linear.is_w4a8() {
        return Err(ProbeError::new(
            "route_mismatch",
            format!("{primary_base}: mixed primary leg did not load as packed W4A8"),
        ));
    }
    let secondary = quant::lin(
        &weights,
        quant::DIT,
        convrot_base,
        case.select.convrot_bias,
        DType::F32,
    )
    .map_err(|error| {
        ProbeError::new(
            "native_error",
            format!("quant::lin({convrot_base}): {error}"),
        )
    })?;
    if !secondary.linear.is_convrot_int8() {
        return Err(ProbeError::new(
            "route_mismatch",
            format!("{convrot_base}: mixed companion leg is not ConvRot INT8"),
        ));
    }
    let input = input_tensor(case, device)?;
    let w4 = primary
        .linear
        .forward(&input)
        .map_err(|error| ProbeError::new("native_error", format!("W4A8 forward: {error}")))?
        .to_dtype(DType::F32)
        .map_err(|error| ProbeError::new("native_error", format!("W4A8 cast: {error}")))?;
    let conv = secondary
        .linear
        .forward(&input)
        .map_err(|error| ProbeError::new("native_error", format!("ConvRot forward: {error}")))?
        .to_dtype(DType::F32)
        .map_err(|error| ProbeError::new("native_error", format!("ConvRot cast: {error}")))?;
    let (wd, cd) = (w4.dims(), conv.dims());
    if wd.is_empty() || cd.len() != wd.len() || wd[..wd.len() - 1] != cd[..cd.len() - 1] {
        return Err(ProbeError::new(
            "shape_mismatch",
            format!("mixed legs have incompatible output shapes {wd:?} / {cd:?}"),
        ));
    }
    let axis = wd.len() - 1;
    let output = Tensor::cat(&[&w4, &conv], axis).map_err(|error| {
        ProbeError::new("native_error", format!("mixed concatenation: {error}"))
    })?;
    finish(
        output,
        Some("mixed-w4a8-convrot".into()),
        Some(format!(
            "same Weights map: W4A8 {primary_base} + ConvRot {convrot_base}; \
             concat order [w4a8, convrot] along axis {axis}"
        )),
        device_info.clone(),
    )
}

fn compute(
    case: &Case,
    manifest: &Manifest,
    selected: &[SelectedPayload],
    device: &Device,
    device_info: &DeviceInfo,
) -> PResult<NativeOutput> {
    match case.kind.as_str() {
        KIND_W4A8 | KIND_CONVROT => projection(case, selected, device, device_info),
        KIND_MIXED => mixed(case, selected, device, device_info),
        KIND_INTERPOLATION | KIND_REDUCED => pruned(case, manifest, selected, device, device_info),
        other => Err(ProbeError::new(
            "manifest_malformed",
            format!("unknown kind {other}"),
        )),
    }
}

/// **Manual only.** Bounded actual-checkpoint numeric probe against a pinned CPU oracle.
#[test]
#[ignore = "manual only: real artifact and selected device; requires H3_ACTUAL_PROBE_MANIFEST and H3_ACTUAL_PROBE_RECEIPT_DIR outside both repositories"]
fn actual_checkpoint_projections_match_pinned_cpu_oracle() {
    // Device initialization is LAZY: metadata-only preflight (and every refusal receipt) must not
    // open a GPU, and a machine without one must still get per-case refusal receipts.
    let mut compute_case = |case: &Case, manifest: &Manifest, selected: &[SelectedPayload]| {
        let (device, device_info) = selected_device()?;
        compute(case, manifest, selected, &device, &device_info)
    };
    let receipts = run_probe("candle", load_index, resolve, &mut compute_case)
        .unwrap_or_else(|error| panic!("{error}"));
    println!(
        "H3_ACTUAL_CHECKPOINT_PROBE backend=candle cases={} receipt_root={}",
        receipts.len(),
        std::env::var(RECEIPT_ENV).unwrap_or_default()
    );
}

// =================================================================================================
// Pure parser / preflight tests (tiny safetensors bytes; no model, no accelerator, no execution)
// =================================================================================================

use serde_json::{json, Value};
use support::{f32_le, sha256_hex_bytes, staging_receipt_json, write_tiny_safetensors};

const TINY_DESCRIPTOR: &str = r#"{"format":"convrot_int8","group_size":16}"#;

fn input_values_sha(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256_hex_bytes(&bytes)
}

fn tiny_tensor_shas() -> (String, String, String) {
    (
        sha256_hex_bytes(&[0, 1, 2, 3]),
        sha256_hex_bytes(&f32_le(&[1.0, 2.0])),
        sha256_hex_bytes(TINY_DESCRIPTOR.as_bytes()),
    )
}

struct TinyFixture {
    _dir: tempfile::TempDir,
    artifact: std::path::PathBuf,
    artifact_sha: String,
    artifact_bytes: u64,
    staging: std::path::PathBuf,
    oracle: std::path::PathBuf,
    receipt_dir: std::path::PathBuf,
}

fn tiny_fixture() -> TinyFixture {
    let dir = tempfile::tempdir().expect("tempdir");
    // Outside the workspace: tempfile roots live under the OS temp dir. A tiny ConvRot operand
    // set is enough to exercise selection/binding refusal paths without any accelerator.
    let artifact = dir.path().join("artifact.safetensors");
    write_tiny_safetensors(
        &artifact,
        &[
            ("p.weight", "I8", vec![2, 2], vec![0, 1, 2, 3]),
            ("p.weight_scale", "F32", vec![2], f32_le(&[1.0, 2.0])),
            (
                "p.comfy_quant",
                "U8",
                vec![TINY_DESCRIPTOR.len()],
                TINY_DESCRIPTOR.as_bytes().to_vec(),
            ),
        ],
    );
    let artifact_sha = sha256_hex_bytes(&[0xAB; 32]);
    let artifact_bytes = std::fs::metadata(&artifact).unwrap().len();
    let parent = dir.path().join("stage-receipt.json");
    std::fs::write(
        &parent,
        br#"{"schema":"h3-finetune-stage-receipt","schema_version":1}"#,
    )
    .unwrap();
    let parent_sha = sha256_hex_bytes(&std::fs::read(&parent).unwrap());
    let staging_bytes = staging_receipt_json(
        &artifact,
        &artifact_sha,
        artifact_bytes,
        &parent,
        &parent_sha,
    );
    let staging = dir.path().join("staging-projection.json");
    std::fs::write(&staging, &staging_bytes).unwrap();
    let oracle = dir.path().join("oracle.json");
    let receipt_dir = dir.path().join("receipts");
    TinyFixture {
        _dir: dir,
        artifact,
        artifact_sha,
        artifact_bytes,
        staging,
        oracle,
        receipt_dir,
    }
}

fn tiny_manifest(
    fixture: &TinyFixture,
    oracle_sha: &str,
    oracle_bytes: &[u8],
    cases: Value,
) -> std::path::PathBuf {
    std::fs::write(&fixture.oracle, oracle_bytes).unwrap();
    let staging_sha = sha256_hex_bytes(&std::fs::read(&fixture.staging).unwrap());
    let manifest = json!({
        "schema": "sceneworks.h3.actual_checkpoint_probe",
        "schema_version": 1,
        "backend": "candle",
        "artifact": {
            "path": fixture.artifact,
            "sha256": fixture.artifact_sha,
            "bytes": fixture.artifact_bytes,
        },
        "staging": {
            "receipt_path": fixture.staging,
            "receipt_sha256": staging_sha,
            "verified_artifact_sha256": fixture.artifact_sha,
        },
        "oracle": {
            "path": fixture.oracle,
            "sha256": oracle_sha,
            "producer": "pinned-h3-cpu-oracle",
            "producer_version": 1,
            "artifact_sha256": fixture.artifact_sha,
            "producer_provenance": [{
                "script": support::PINNED_CONVROT_SCRIPT,
                "sha256": support::pinned_producer_sha256(support::PINNED_CONVROT_SCRIPT)
                    .expect("pinned convrot producer"),
            }],
        },
        "limits": {"max_cases": 8, "max_selected_tensors": 8, "max_selected_bytes": 4096},
        "cases": cases,
    });
    let path = fixture._dir.path().join("manifest.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    path
}

fn convrot_case(base: &str, selected: Value, aliases: Value) -> Value {
    json!({
        "id": "convrot.tiny",
        "kind": "convrot_projection",
        "select": {
            "base": base,
            "bias": false,
            "expect_route": "convrot",
            "aliases": aliases,
        },
        "input": {"dtype": "BF16", "shape": [1, 2], "values": [1.0, -2.0]},
        "tolerance": {"max_abs": 0.0},
        "selected_tensors": selected,
    })
}

fn oracle_document(
    artifact_sha: &str,
    weight_sha: &str,
    scale_sha: &str,
    quant_sha: &str,
) -> Vec<u8> {
    serde_json::to_vec_pretty(&json!({
        "schema": "sceneworks.h3.actual_checkpoint_oracle",
        "schema_version": 1,
        "producer": "pinned-h3-cpu-oracle",
        "producer_version": 1,
        "artifact_sha256": artifact_sha,
        "cases": [{
            "id": "convrot.tiny",
            "input": {
                "dtype": "BF16",
                "shape": [1, 2],
                "values_sha256": input_values_sha(&[1.0, -2.0]),
            },
            "tolerance": {"max_abs": 0.0},
            "selected": [
                {"name": "p.weight", "sha256": weight_sha, "dtype": "I8", "shape": [2, 2], "bytes": 4},
                {"name": "p.weight_scale", "sha256": scale_sha, "dtype": "F32", "shape": [2], "bytes": 8},
                {"name": "p.comfy_quant", "sha256": quant_sha, "dtype": "U8", "shape": [TINY_DESCRIPTOR.len()], "bytes": TINY_DESCRIPTOR.len()},
            ],
            "output": {"shape": [1, 2], "values": [0.0, 0.0]},
        }],
    }))
    .unwrap()
}

#[test]
fn preflight_refuses_missing_artifact_before_any_payload_read() {
    let fixture = tiny_fixture();
    std::fs::remove_file(&fixture.artifact).unwrap();
    let oracle_bytes = oracle_document(
        &fixture.artifact_sha,
        &"11".repeat(32),
        &"22".repeat(32),
        &"33".repeat(32),
    );
    let oracle_sha = sha256_hex_bytes(&oracle_bytes);
    let manifest = tiny_manifest(
        &fixture,
        &oracle_sha,
        &oracle_bytes,
        json!([convrot_case(
            "p",
            json!(["p.weight", "p.weight_scale", "p.comfy_quant"]),
            json!({})
        )]),
    );
    let mut executed = false;
    let result = run_probe_at(
        "candle",
        &manifest,
        &fixture.receipt_dir,
        load_index,
        resolve,
        |_, _, _| -> PResult<NativeOutput> {
            executed = true;
            Err(ProbeError::new("unexpected_execution", "must not execute"))
        },
    );
    assert!(result.is_err());
    assert!(
        !executed,
        "preflight refusal must not reach native execution"
    );
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(fixture.receipt_dir.join("candle.manifest-refusal.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["refusal"]["code"], "artifact_missing");
    assert_eq!(receipt["payload_reads"], 0);
}

#[test]
fn preflight_refuses_missing_selection_before_any_payload_read() {
    let fixture = tiny_fixture();
    let oracle_bytes = oracle_document(
        &fixture.artifact_sha,
        &"11".repeat(32),
        &"22".repeat(32),
        &"33".repeat(32),
    );
    let oracle_sha = sha256_hex_bytes(&oracle_bytes);
    let mut case = convrot_case(
        "p",
        json!(["p.weight", "p.weight_scale", "p.comfy_quant"]),
        json!({}),
    );
    case["select"]["base"] = json!("absent");
    case["selected_tensors"] =
        json!(["absent.weight", "absent.weight_scale", "absent.comfy_quant"]);
    let manifest = tiny_manifest(&fixture, &oracle_sha, &oracle_bytes, json!([case]));
    let mut executed = false;
    let result = run_probe_at(
        "candle",
        &manifest,
        &fixture.receipt_dir,
        load_index,
        resolve,
        |_, _, _| -> PResult<NativeOutput> {
            executed = true;
            Err(ProbeError::new("unexpected_execution", "must not execute"))
        },
    );
    assert!(result.is_err(), "{result:?}");
    assert!(!executed);
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(
            fixture
                .receipt_dir
                .join(receipt_file_name("candle", "convrot.tiny")),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["status"], "refused");
    assert_eq!(receipt["refusal"]["code"], "selection_missing");
    assert_eq!(receipt["payload_reads"], 0);
}

#[test]
fn preflight_refuses_tensor_sha_mismatch_before_native_execution() {
    let fixture = tiny_fixture();
    let oracle_bytes = oracle_document(
        &fixture.artifact_sha,
        &"11".repeat(32),
        &"22".repeat(32),
        &"33".repeat(32),
    );
    let oracle_sha = sha256_hex_bytes(&oracle_bytes);
    let manifest = tiny_manifest(
        &fixture,
        &oracle_sha,
        &oracle_bytes,
        json!([convrot_case(
            "p",
            json!(["p.weight", "p.weight_scale", "p.comfy_quant"]),
            json!({})
        )]),
    );
    let mut executed = false;
    let result = run_probe_at(
        "candle",
        &manifest,
        &fixture.receipt_dir,
        load_index,
        resolve,
        |_, _, _| -> PResult<NativeOutput> {
            executed = true;
            Err(ProbeError::new("unexpected_execution", "must not execute"))
        },
    );
    assert!(result.is_err());
    assert!(
        !executed,
        "a wrong oracle tensor digest must refuse before native execution"
    );
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(
            fixture
                .receipt_dir
                .join(receipt_file_name("candle", "convrot.tiny")),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["status"], "refused");
    assert_eq!(receipt["refusal"]["code"], "tensor_sha_mismatch");
    assert_eq!(receipt["payload_reads"], 3);
    assert_eq!(receipt["artifact"]["whole_file_digest_recomputed"], false);
}

#[test]
fn malformed_manifest_refusal_still_writes_a_receipt() {
    let fixture = tiny_fixture();
    let manifest = fixture._dir.path().join("broken.json");
    std::fs::write(&manifest, b"{ this is not json").unwrap();
    let result = run_probe_at(
        "candle",
        &manifest,
        &fixture.receipt_dir,
        load_index,
        resolve,
        |_, _, _| -> PResult<NativeOutput> {
            Err(ProbeError::new("unexpected_execution", "must not execute"))
        },
    );
    assert!(result.is_err());
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(fixture.receipt_dir.join("candle.manifest-refusal.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["refusal"]["code"], "manifest_malformed");
    assert_eq!(receipt["payload_reads"], 0);
}

#[test]
fn staging_projection_must_bind_a_full_file_verification_and_parent_receipt() {
    let fixture = tiny_fixture();
    let oracle_bytes = oracle_document(
        &fixture.artifact_sha,
        &"11".repeat(32),
        &"22".repeat(32),
        &"33".repeat(32),
    );
    let oracle_sha = sha256_hex_bytes(&oracle_bytes);
    let manifest = tiny_manifest(
        &fixture,
        &oracle_sha,
        &oracle_bytes,
        json!([convrot_case(
            "p",
            json!(["p.weight", "p.weight_scale", "p.comfy_quant"]),
            json!({})
        )]),
    );
    // Drop the transport fields that prove the campaign staging step verified the full file.
    let mut staging: Value =
        serde_json::from_slice(&std::fs::read(&fixture.staging).unwrap()).unwrap();
    staging["full_file_digest_verified"] = json!(false);
    staging
        .as_object_mut()
        .unwrap()
        .remove("parent_receipt_sha256");
    let bytes = serde_json::to_vec_pretty(&staging).unwrap();
    std::fs::write(&fixture.staging, &bytes).unwrap();
    let mut manifest_json: Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    manifest_json["staging"]["receipt_sha256"] = json!(sha256_hex_bytes(&bytes));
    std::fs::write(
        &manifest,
        serde_json::to_vec_pretty(&manifest_json).unwrap(),
    )
    .unwrap();
    let result = run_probe_at(
        "candle",
        &manifest,
        &fixture.receipt_dir,
        load_index,
        resolve,
        |_, _, _| -> PResult<NativeOutput> {
            Err(ProbeError::new("unexpected_execution", "must not execute"))
        },
    );
    assert!(result.is_err());
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(fixture.receipt_dir.join("candle.manifest-refusal.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["refusal"]["code"], "staging_binding_mismatch");
    assert_eq!(receipt["payload_reads"], 0);
}

#[test]
fn raw_to_canonical_aliases_are_naming_only_and_still_hash_raw_bytes() {
    let fixture = tiny_fixture();
    // Actual finetuned files may name a tensor differently from the canonical loader key. The
    // alias is naming only: the oracle still binds the RAW tensor id + bytes.
    let oracle_bytes = oracle_document(
        &fixture.artifact_sha,
        &"11".repeat(32),
        &"22".repeat(32),
        &"33".repeat(32),
    );
    let oracle_sha = sha256_hex_bytes(&oracle_bytes);
    let aliases = json!({
        "p.weight": "canonical.weight",
        "p.weight_scale": "canonical.weight_scale",
        "p.comfy_quant": "canonical.comfy_quant",
    });
    let case = convrot_case(
        "canonical",
        json!(["p.weight", "p.weight_scale", "p.comfy_quant"]),
        aliases,
    );
    let manifest = tiny_manifest(&fixture, &oracle_sha, &oracle_bytes, json!([case]));
    let result = run_probe_at(
        "candle",
        &manifest,
        &fixture.receipt_dir,
        load_index,
        resolve,
        |_, _, _| -> PResult<NativeOutput> {
            Err(ProbeError::new("unexpected_execution", "must not execute"))
        },
    );
    assert!(result.is_err());
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(
            fixture
                .receipt_dir
                .join(receipt_file_name("candle", "convrot.tiny")),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["refusal"]["code"], "tensor_sha_mismatch");
    assert_eq!(receipt["selected"][0]["name"], "p.weight");
    assert_eq!(receipt["selected"][0]["canonical"], "canonical.weight");
}

#[test]
fn oversized_manifest_refuses_before_any_payload_read() {
    let fixture = tiny_fixture();
    let manifest = fixture._dir.path().join("huge.json");
    let mut bytes = b"{\"schema\":".to_vec();
    bytes.resize(support::MAX_MANIFEST_BYTES as usize + 1, b' ');
    std::fs::write(&manifest, &bytes).unwrap();
    let result = run_probe_at(
        "candle",
        &manifest,
        &fixture.receipt_dir,
        load_index,
        resolve,
        |_, _, _| -> PResult<NativeOutput> {
            Err(ProbeError::new("unexpected_execution", "must not execute"))
        },
    );
    assert!(result.is_err());
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(fixture.receipt_dir.join("candle.manifest-refusal.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["refusal"]["code"], "metadata_too_large");
    assert_eq!(receipt["payload_reads"], 0);
}

#[test]
fn wrong_numeric_expectation_goes_red() {
    let fixture = tiny_fixture();
    let (weight_sha, scale_sha, quant_sha) = tiny_tensor_shas();
    let oracle_bytes = oracle_document(&fixture.artifact_sha, &weight_sha, &scale_sha, &quant_sha);
    let oracle_sha = sha256_hex_bytes(&oracle_bytes);
    let manifest = tiny_manifest(
        &fixture,
        &oracle_sha,
        &oracle_bytes,
        json!([convrot_case(
            "p",
            json!(["p.weight", "p.weight_scale", "p.comfy_quant"]),
            json!({})
        )]),
    );
    let mut executed = false;
    let result = run_probe_at(
        "candle",
        &manifest,
        &fixture.receipt_dir,
        load_index,
        resolve,
        |_, _, _| -> PResult<NativeOutput> {
            executed = true;
            Ok(NativeOutput {
                shape: vec![1, 2],
                values: vec![1.0, 1.0],
                route: Some("fake".into()),
                detail: None,
                device: DeviceInfo {
                    kind: "fake".into(),
                    label: "fake".into(),
                    index: None,
                    cpu_fallback: false,
                    visible_devices: None,
                    note: None,
                },
            })
        },
    );
    assert!(result.is_err());
    assert!(executed, "the numeric comparison must actually run");
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(
            fixture
                .receipt_dir
                .join(receipt_file_name("candle", "convrot.tiny")),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["status"], "fail");
    assert_eq!(receipt["pass"], false);
    assert_eq!(receipt["comparison"]["code"], "max_abs_exceeded");
    assert_eq!(receipt["comparison"]["max_abs"], 1.0);
    assert_eq!(receipt["comparison"]["tolerance"], 0.0);
}

#[test]
fn unwritable_receipt_path_goes_red_instead_of_faking_success() {
    let fixture = tiny_fixture();
    let (weight_sha, scale_sha, quant_sha) = tiny_tensor_shas();
    let oracle_bytes = oracle_document(&fixture.artifact_sha, &weight_sha, &scale_sha, &quant_sha);
    let oracle_sha = sha256_hex_bytes(&oracle_bytes);
    let manifest = tiny_manifest(
        &fixture,
        &oracle_sha,
        &oracle_bytes,
        json!([convrot_case(
            "p",
            json!(["p.weight", "p.weight_scale", "p.comfy_quant"]),
            json!({})
        )]),
    );
    // A regular file cannot be used as the receipt directory: the selector must go RED instead of
    // silently returning a receipt list with a fake "{}" success.
    let blocked = fixture._dir.path().join("blocked-receipts");
    std::fs::write(&blocked, b"not a directory").unwrap();
    let mut executed = false;
    let result = run_probe_at(
        "candle",
        &manifest,
        &blocked,
        load_index,
        resolve,
        |_, _, _| -> PResult<NativeOutput> {
            executed = true;
            Err(ProbeError::new("unexpected_execution", "must not execute"))
        },
    );
    let message = format!(
        "{:?}",
        result.expect_err("an unwritable receipt path must be RED")
    );
    assert!(
        message.contains("could not be written"),
        "receipt write failure must be surfaced: {message}"
    );
    assert!(!executed, "preflight must not reach execution");
}

#[test]
fn cpu_fallback_never_counts_as_device_proof() {
    let fixture = tiny_fixture();
    let (weight_sha, scale_sha, quant_sha) = tiny_tensor_shas();
    let oracle_bytes = oracle_document(&fixture.artifact_sha, &weight_sha, &scale_sha, &quant_sha);
    let oracle_sha = sha256_hex_bytes(&oracle_bytes);
    let manifest = tiny_manifest(
        &fixture,
        &oracle_sha,
        &oracle_bytes,
        json!([convrot_case(
            "p",
            json!(["p.weight", "p.weight_scale", "p.comfy_quant"]),
            json!({})
        )]),
    );
    let mut executed = false;
    let result = run_probe_at(
        "candle",
        &manifest,
        &fixture.receipt_dir,
        load_index,
        resolve,
        |_, _, _| -> PResult<NativeOutput> {
            executed = true;
            Ok(NativeOutput {
                shape: vec![1, 2],
                values: vec![0.0, 0.0],
                route: Some("fake".into()),
                detail: None,
                device: DeviceInfo {
                    kind: "fake".into(),
                    label: "fake".into(),
                    index: None,
                    cpu_fallback: true,
                    visible_devices: None,
                    note: None,
                },
            })
        },
    );
    assert!(result.is_err());
    assert!(executed, "the numeric comparison must actually run");
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(
            fixture
                .receipt_dir
                .join(receipt_file_name("candle", "convrot.tiny")),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["status"], "fail");
    assert_eq!(receipt["pass"], false);
    assert_eq!(
        receipt["comparison"]["code"],
        "cpu_fallback_not_device_proof"
    );
    assert_eq!(receipt["comparison"]["max_abs"], 0.0);
    assert_eq!(receipt["comparison"]["tolerance"], 0.0);
    assert_eq!(receipt["native"]["device_proof"], false);
    assert_eq!(receipt["native"]["device"]["cpu_fallback"], true);
}
