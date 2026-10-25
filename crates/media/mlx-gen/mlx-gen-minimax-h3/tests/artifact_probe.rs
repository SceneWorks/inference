//! Bounded **actual-checkpoint numeric probe** — MLX consumer (sc-26679 terminal repair).
//!
//! Mirror of the candle `artifact_probe` consumer: it reads ONLY the declared tensor ranges from an
//! actual frozen artifact and compares the **native production functions**
//! (`quant::lin(..).forward`, `PrunedTable`, `PrunedReducedLinear`) against a pinned CPU-oracle
//! expectation bound to those exact selected payload bytes.
//!
//! It implements no arithmetic oracle and re-derives no producer math. The pinned oracle comes from
//! the existing producers (`scripts/h3_pruned_adaln_oracle.py`, `scripts/h3_convrot_int8_oracle.py`,
//! `scripts/ci/generate_minimax_h3_w4a8_fixture.py`) or one owned extension, and the oracle file is
//! sha256-bound in the manifest. No model load, no generation, no avalanche claims.
//!
//! Ordinary CI runs only the tiny-fixture parser/preflight tests below; the numeric test is
//! `#[ignore]`d and selected explicitly:
//!
//! ```text
//! H3_ACTUAL_PROBE_MANIFEST=/outside/manifest.json \
//! H3_ACTUAL_PROBE_RECEIPT_DIR=/outside/receipts \
//! cargo test -p mlx-gen-minimax-h3 --test integration -- \
//!   artifact_probe::actual_checkpoint_projections_match_pinned_cpu_oracle --exact --ignored
//! ```

#[path = "../../../candle-gen/candle-gen-minimax-h3/tests/support/artifact_probe_support.rs"]
mod support;

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::io::Seek;
use std::path::Path;

use mlx_gen::gen_core::weightsmeta::safetensors_reader_tensor_locations;
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::dit::pruned_adaln::{PrunedReducedLinear, PrunedTable};
use mlx_gen_minimax_h3::quant;
use mlx_rs::{Array, Device, DeviceType, Dtype};
use support::{
    receipt_file_name, run_probe, run_probe_at, Case, DeviceInfo, HeaderIndex, IndexedTensor,
    Manifest, NativeOutput, PResult, ProbeError, SelectedPayload, KIND_CONVROT, KIND_INTERPOLATION,
    KIND_MIXED, KIND_REDUCED, KIND_W4A8, RECEIPT_ENV, TABLE_KEY,
};

/// Canonical packed operand sets (STORED dtypes; MLX maps F8 raw bytes to Uint8 arrays).
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

/// Header-only index via the shared, validated safetensors layout reader. Never reads payload.
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

fn resolve(case: &Case, index: &HeaderIndex, _manifest: &Manifest) -> PResult<Vec<String>> {
    let raw_of = |canonical: &str| case.select.raw_of(canonical);
    match case.kind.as_str() {
        KIND_W4A8 | KIND_CONVROT => {
            let base = case.select.base.as_deref().expect("parsed base");
            // MLX keeps the raw F8_E4M3 bytes as Uint8 (the W4A8 loader requires Uint8 s_rel) and
            // consumes the I8 codes at their native signed width.
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
            // The dense-vs-ConvRot contrast lives inside the pinned ConvRot oracle's `mixedLayer`
            // companion leg; there is no standalone dense kind and no standalone dense fidelity
            // claim.
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
        KIND_INTERPOLATION | KIND_REDUCED => match case.kind.as_str() {
            KIND_INTERPOLATION => {
                let table = case.select.table.as_deref().expect("parsed table");
                if table != TABLE_KEY {
                    return Err(ProbeError::new(
                        "selection_bounds",
                        format!("{}: table must be canonical {TABLE_KEY:?}", case.id),
                    ));
                }
                let raw = raw_of(TABLE_KEY);
                let tensor = index.require(&raw)?;
                if tensor.shape.len() != 2 {
                    return Err(ProbeError::new(
                        "selection_bounds",
                        format!("{}: table must be 2-D, got {:?}", case.id, tensor.shape),
                    ));
                }
                require_dtype(index, &raw, &["F32"], &case.id)?;
                if let Some(grid) = case.select.grid {
                    if tensor.shape[0] != grid {
                        return Err(ProbeError::new(
                            "selection_bounds",
                            format!("{}: table grid mismatch {:?}", case.id, tensor.shape),
                        ));
                    }
                }
                if let Some(rank) = case.select.rank {
                    if tensor.shape[1] != rank {
                        return Err(ProbeError::new(
                            "selection_bounds",
                            format!("{}: table rank mismatch {:?}", case.id, tensor.shape),
                        ));
                    }
                }
                Ok(vec![raw])
            }
            _ => {
                let prefix = mlx_prefix(case)?;
                let canonical_weight = format!("{prefix}.weight");
                let canonical_bias = format!("{prefix}.bias");
                if let Some(weight) = case.select.weight.as_deref() {
                    if weight != canonical_weight {
                        return Err(ProbeError::new(
                            "selection_bounds",
                            format!(
                                "{}: MLX PrunedReducedLinear consumes {canonical_weight:?}; \
                                     explicit weight {weight:?} is not that canonical operand",
                                case.id
                            ),
                        ));
                    }
                }
                if let Some(bias) = case.select.bias_tensor.as_deref() {
                    if bias != canonical_bias {
                        return Err(ProbeError::new(
                            "selection_bounds",
                            format!(
                                "{}: MLX PrunedReducedLinear consumes {canonical_bias:?}; \
                                     explicit bias {bias:?} is not that canonical operand",
                                case.id
                            ),
                        ));
                    }
                }
                let raw_weight = raw_of(&canonical_weight);
                let raw_bias = raw_of(&canonical_bias);
                let weight = index.require(&raw_weight)?;
                let bias = index.require(&raw_bias)?;
                require_dtype(index, &raw_weight, &["F16", "BF16", "F32"], &case.id)?;
                require_dtype(index, &raw_bias, &["F16", "BF16", "F32"], &case.id)?;
                if weight.shape.len() != 2 || bias.shape.len() != 1 {
                    return Err(ProbeError::new(
                        "selection_bounds",
                        format!(
                            "{}: reduced weight/bias must be 2-D/1-D, got {:?}/{:?}",
                            case.id, weight.shape, bias.shape
                        ),
                    ));
                }
                if bias.shape[0] != weight.shape[0] {
                    return Err(ProbeError::new(
                        "selection_bounds",
                        format!(
                            "{}: reduced bias {} disagrees with out_features {}",
                            case.id, bias.shape[0], weight.shape[0]
                        ),
                    ));
                }
                if let Some(rank) = case.select.rank {
                    if weight.shape[1] != rank {
                        return Err(ProbeError::new(
                            "selection_bounds",
                            format!(
                                "{}: reduced weight rank mismatch {:?}",
                                case.id, weight.shape
                            ),
                        ));
                    }
                }
                if let Some(out_features) = case.select.out_features {
                    if weight.shape[0] != out_features {
                        return Err(ProbeError::new(
                            "selection_bounds",
                            format!(
                                "{}: reduced weight out_features mismatch {:?}",
                                case.id, weight.shape
                            ),
                        ));
                    }
                }
                Ok(vec![raw_weight, raw_bias])
            }
        },
        other => Err(ProbeError::new(
            "manifest_malformed",
            format!("{}: unknown kind {other}", case.id),
        )),
    }
}

fn mlx_prefix(case: &Case) -> PResult<String> {
    if let Some(prefix) = case.select.prefix.clone() {
        return Ok(prefix);
    }
    case.select
        .weight
        .as_deref()
        .and_then(|weight| weight.strip_suffix(".weight"))
        .map(str::to_owned)
        .ok_or_else(|| {
            ProbeError::new(
                "selection_bounds",
                format!(
                    "{}: reduced case needs select.prefix or select.weight",
                    case.id
                ),
            )
        })
}

fn dtype_width(dtype: &str) -> PResult<usize> {
    Ok(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        other => {
            return Err(ProbeError::new(
                "unsupported_dtype",
                format!("mlx cannot materialize stored dtype {other}"),
            ))
        }
    })
}

fn mlx_dtype(dtype: &str) -> PResult<Dtype> {
    Ok(match dtype {
        "BOOL" => Dtype::Bool,
        "U8" => Dtype::Uint8,
        "I8" => Dtype::Int8,
        "U16" => Dtype::Uint16,
        "I16" => Dtype::Int16,
        "U32" => Dtype::Uint32,
        "I32" => Dtype::Int32,
        "U64" => Dtype::Uint64,
        "I64" => Dtype::Int64,
        "F16" => Dtype::Float16,
        "BF16" => Dtype::Bfloat16,
        "F32" => Dtype::Float32,
        "F64" => Dtype::Float64,
        // MLX's ConvRot/W4A8 loaders require the raw F8 byte payload as Uint8, exactly like the
        // source dialect stores it.
        "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => Dtype::Uint8,
        other => {
            return Err(ProbeError::new(
                "unsupported_dtype",
                format!("mlx cannot materialize stored dtype {other}"),
            ))
        }
    })
}

fn i32_shape(shape: &[usize]) -> PResult<Vec<i32>> {
    shape
        .iter()
        .map(|&dim| {
            i32::try_from(dim).map_err(|_| {
                ProbeError::new("selection_bounds", format!("dimension {dim} exceeds i32"))
            })
        })
        .collect()
}

/// Copy selected raw bytes into an MLX array at the source dtype.
///
/// SAFETY: `shape × dtype-width` is checked against the exact number of bytes read from the
/// artifact's declared range before the pointer is handed over; MLX copies the buffer.
fn array_from_payload(payload: &SelectedPayload) -> PResult<Array> {
    let dtype = mlx_dtype(&payload.dtype)?;
    let shape = i32_shape(&payload.shape)?;
    let elements: usize = payload.shape.iter().product();
    let expected = elements
        .checked_mul(dtype_width(&payload.dtype)?)
        .ok_or_else(|| ProbeError::new("selection_bounds", "tensor byte count overflow"))?;
    if expected != payload.bytes.len() {
        return Err(ProbeError::new(
            "selection_bounds",
            format!(
                "{}: {} bytes read, shape {:?} at {} needs {}",
                payload.name,
                payload.bytes.len(),
                payload.shape,
                payload.dtype,
                expected
            ),
        ));
    }
    Ok(unsafe { Array::from_raw_data(payload.bytes.as_ptr() as *const c_void, &shape, dtype) })
}

fn tensor_map(selected: &[SelectedPayload]) -> PResult<HashMap<String, Array>> {
    let mut map = HashMap::with_capacity(selected.len());
    for payload in selected {
        if map
            .insert(payload.canonical.clone(), array_from_payload(payload)?)
            .is_some()
        {
            return Err(ProbeError::new(
                "selection_bounds",
                format!("canonical tensor {} is bound twice", payload.canonical),
            ));
        }
    }
    Ok(map)
}

fn array_f32(shape: &[usize], values: &[f32]) -> PResult<Array> {
    let shape = i32_shape(shape)?;
    if shape.iter().map(|&d| d as usize).product::<usize>() != values.len() {
        return Err(ProbeError::new(
            "input_shape",
            format!(
                "input shape {shape:?} disagrees with {} values",
                values.len()
            ),
        ));
    }
    Ok(Array::from_slice(values, &shape))
}

fn input_array(case: &Case) -> PResult<Array> {
    let (shape, values) = case.require_input()?;
    let base = array_f32(shape, values)?;
    match case.input_dtype.as_str() {
        "F32" => Ok(base),
        "BF16" => base
            .as_dtype(Dtype::Bfloat16)
            .map_err(|error| ProbeError::new("input_shape", format!("BF16 input: {error}"))),
        other => Err(ProbeError::new(
            "manifest_malformed",
            format!("unsupported input dtype {other}"),
        )),
    }
}

fn output_f32(
    output: &Array,
    route: Option<String>,
    detail: Option<String>,
    device: DeviceInfo,
) -> PResult<NativeOutput> {
    let output = output
        .as_dtype(Dtype::Float32)
        .map_err(|error| ProbeError::new("native_error", format!("cast output: {error}")))?;
    mlx_rs::transforms::eval([&output])
        .map_err(|error| ProbeError::new("native_error", format!("eval output: {error}")))?;
    let shape: Vec<usize> = output.shape().iter().map(|&dim| dim as usize).collect();
    let values = output.as_slice::<f32>().to_vec();
    Ok(NativeOutput {
        shape,
        values,
        route,
        detail,
        device,
    })
}

fn selected_device() -> PResult<(Device, DeviceInfo)> {
    let device = Device::try_default()
        .map_err(|error| ProbeError::new("device_unavailable", format!("MLX device: {error}")))?;
    let device_type = device.get_type().map_err(|error| {
        ProbeError::new("device_unavailable", format!("MLX device type: {error}"))
    })?;
    let index = device.get_index().ok().map(i64::from);
    let cpu_fallback = matches!(device_type, DeviceType::Cpu);
    let info = DeviceInfo {
        kind: match device_type {
            DeviceType::Gpu => "mlx-default-gpu".into(),
            DeviceType::Cpu => "mlx-default-cpu".into(),
        },
        label: format!("mlx device type {device_type:?} index {index:?}"),
        index,
        cpu_fallback,
        visible_devices: None,
        note: Some(
            "MLX exposes device type/index, not a hardware UUID; MLX_USE_GPU controls the default"
                .into(),
        ),
    };
    Ok((device, info))
}

fn projection(
    case: &Case,
    selected: &[SelectedPayload],
    device_info: &DeviceInfo,
) -> PResult<NativeOutput> {
    let base = case.select.base.as_deref().expect("parsed base");
    let weights = Weights::from_map(tensor_map(selected)?);
    let mut linear = quant::lin(&weights, base, case.select.bias)
        .map_err(|error| ProbeError::new("native_error", format!("quant::lin({base}): {error}")))?;
    let route = if linear.w4a8_base().is_some() {
        "w4a8"
    } else if linear.convrot_base().is_some() {
        "convrot"
    } else if linear.quantized_params().is_some() {
        "affine"
    } else if linear.dense_weight().is_some() {
        "dense"
    } else {
        return Err(ProbeError::new(
            "route_mismatch",
            format!("{base}: native loader produced no recognized base"),
        ));
    };
    if let Some(expected) = case.select.expect_route.as_deref() {
        if expected != route {
            return Err(ProbeError::new(
                "route_mismatch",
                format!("{base}: expected route {expected}, native selected {route}"),
            ));
        }
    }
    if route == "dense" {
        // The dense base keeps its stored dtype; widen it once (the pruned path's `_lock_dtype`
        // discipline) so the comparison is against the pinned F32 CPU expectation rather than a
        // narrower activation rounding.
        linear
            .cast_weights(Dtype::Float32)
            .map_err(|error| ProbeError::new("native_error", format!("cast_weights: {error}")))?;
    }
    let input = input_array(case)?;
    let output = linear
        .forward(&input)
        .map_err(|error| ProbeError::new("native_error", format!("{base} forward: {error}")))?;
    let detail = format!(
        "quant::lin route={route} base_shape={:?}",
        linear.base_shape()
    );
    output_f32(
        &output,
        Some(route.into()),
        Some(detail),
        device_info.clone(),
    )
}

fn pruned(
    case: &Case,
    selected: &[SelectedPayload],
    device_info: &DeviceInfo,
) -> PResult<NativeOutput> {
    let weights = Weights::from_map(tensor_map(selected)?);
    if case.kind == KIND_INTERPOLATION {
        let canonical = TABLE_KEY;
        let payload = selected
            .iter()
            .find(|payload| payload.canonical == canonical)
            .ok_or_else(|| {
                ProbeError::new("selection_missing", format!("no selected {canonical}"))
            })?;
        let grid = payload.shape[0];
        let rank = payload.shape[1];
        if let (Some(expected_grid), Some(expected_rank)) = (case.select.grid, case.select.rank) {
            if (grid, rank) != (expected_grid, expected_rank) {
                return Err(ProbeError::new(
                    "selection_bounds",
                    format!(
                        "{}: selected table is [{grid}, {rank}], manifest declares \
                         [{expected_grid}, {expected_rank}]",
                        case.id
                    ),
                ));
            }
        }
        let table = PrunedTable::from_weights(&weights, grid, rank)
            .map_err(|error| ProbeError::new("native_error", format!("PrunedTable: {error}")))?;
        let timesteps = case.require_timesteps()?;
        let coordinates = table
            .coordinates(timesteps)
            .map_err(|error| ProbeError::new("native_error", format!("coordinates: {error}")))?;
        return output_f32(
            &coordinates,
            Some("pruned-table".into()),
            Some(format!(
                "PrunedTable grid={grid} rank={rank} timesteps={}",
                timesteps.len()
            )),
            device_info.clone(),
        );
    }
    let prefix = mlx_prefix(case)?;
    let weight = selected
        .iter()
        .find(|payload| payload.canonical == format!("{prefix}.weight"))
        .ok_or_else(|| {
            ProbeError::new("selection_missing", format!("no selected {prefix}.weight"))
        })?;
    let out_features = weight.shape[0];
    let rank = weight.shape[1];
    let linear = PrunedReducedLinear::from_weights(&weights, &prefix, out_features, rank).map_err(
        |error| {
            ProbeError::new(
                "native_error",
                format!("PrunedReducedLinear({prefix}): {error}"),
            )
        },
    )?;
    let (shape, values) = case.require_input()?;
    let coordinates = array_f32(shape, values)?;
    let output = linear
        .forward(&coordinates)
        .map_err(|error| ProbeError::new("native_error", format!("projection forward: {error}")))?;
    output_f32(
        &output,
        Some("pruned-reduced".into()),
        Some(format!(
            "PrunedReducedLinear out_features={out_features} rank={rank} \
             source_dtype={:?}/{:?}",
            linear.source_weight_dtype(),
            linear.source_bias_dtype()
        )),
        device_info.clone(),
    )
}

/// One Weights map carrying a packed W4A8 operand set AND a ConvRot INT8 operand set; both native
/// loaders must keep their own dialect and the outputs are concatenated `[w4a8, convrot]` along the
/// last axis to match the pinned producers' combined expectation.
fn mixed(
    case: &Case,
    selected: &[SelectedPayload],
    device_info: &DeviceInfo,
) -> PResult<NativeOutput> {
    let primary_base = case.select.base.as_deref().expect("parsed base");
    let convrot_base = case
        .select
        .convrot_base
        .as_deref()
        .expect("parsed convrot_base");
    let weights = Weights::from_map(tensor_map(selected)?);
    let primary = quant::lin(&weights, primary_base, case.select.w4a8_bias).map_err(|error| {
        ProbeError::new(
            "native_error",
            format!("quant::lin({primary_base}): {error}"),
        )
    })?;
    if primary.w4a8_base().is_none() {
        return Err(ProbeError::new(
            "route_mismatch",
            format!("{primary_base}: mixed primary leg did not load as packed W4A8"),
        ));
    }
    let secondary =
        quant::lin(&weights, convrot_base, case.select.convrot_bias).map_err(|error| {
            ProbeError::new(
                "native_error",
                format!("quant::lin({convrot_base}): {error}"),
            )
        })?;
    if secondary.convrot_base().is_none() {
        return Err(ProbeError::new(
            "route_mismatch",
            format!("{convrot_base}: mixed companion leg is not ConvRot INT8"),
        ));
    }
    let input = input_array(case)?;
    let w4 = primary
        .forward(&input)
        .map_err(|error| ProbeError::new("native_error", format!("W4A8 forward: {error}")))?
        .as_dtype(Dtype::Float32)
        .map_err(|error| ProbeError::new("native_error", format!("W4A8 cast: {error}")))?;
    let conv = secondary
        .forward(&input)
        .map_err(|error| ProbeError::new("native_error", format!("ConvRot forward: {error}")))?
        .as_dtype(Dtype::Float32)
        .map_err(|error| ProbeError::new("native_error", format!("ConvRot cast: {error}")))?;
    let (wd, cd) = (w4.shape().to_vec(), conv.shape().to_vec());
    if wd.is_empty() || cd.len() != wd.len() || wd[..wd.len() - 1] != cd[..cd.len() - 1] {
        return Err(ProbeError::new(
            "shape_mismatch",
            format!("mixed legs have incompatible output shapes {wd:?} / {cd:?}"),
        ));
    }
    let axis = wd.len() as i32 - 1;
    let output = mlx_rs::ops::concatenate_axis(&[&w4, &conv], axis).map_err(|error| {
        ProbeError::new("native_error", format!("mixed concatenation: {error}"))
    })?;
    output_f32(
        &output,
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
    _manifest: &Manifest,
    selected: &[SelectedPayload],
    device_info: &DeviceInfo,
) -> PResult<NativeOutput> {
    match case.kind.as_str() {
        KIND_W4A8 | KIND_CONVROT => projection(case, selected, device_info),
        KIND_MIXED => mixed(case, selected, device_info),
        KIND_INTERPOLATION | KIND_REDUCED => pruned(case, selected, device_info),
        other => Err(ProbeError::new(
            "manifest_malformed",
            format!("unknown kind {other}"),
        )),
    }
}

/// **Manual only.** Bounded actual-checkpoint numeric probe against a pinned CPU oracle.
#[test]
#[ignore = "manual only: real artifact and Metal device; requires H3_ACTUAL_PROBE_MANIFEST and H3_ACTUAL_PROBE_RECEIPT_DIR outside both repositories"]
fn actual_checkpoint_projections_match_pinned_cpu_oracle() {
    // Device initialization is LAZY: metadata-only preflight (and every refusal receipt) must not
    // touch MLX, and a machine without the default device must still get refusal receipts.
    let mut compute_case = |case: &Case, manifest: &Manifest, selected: &[SelectedPayload]| {
        let (_device, device_info) = selected_device()?;
        compute(case, manifest, selected, &device_info)
    };
    let receipts = run_probe("mlx", load_index, resolve, &mut compute_case)
        .unwrap_or_else(|error| panic!("{error}"));
    println!(
        "H3_ACTUAL_CHECKPOINT_PROBE backend=mlx cases={} receipt_root={}",
        receipts.len(),
        std::env::var(RECEIPT_ENV).unwrap_or_default()
    );
}

// =================================================================================================
// Pure parser / preflight tests (tiny safetensors bytes; no model, no accelerator, no execution)
// =================================================================================================

use serde_json::{json, Value};
use support::{f32_le, sha256_hex_bytes, staging_receipt_json, write_tiny_safetensors};

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
    let artifact_sha = sha256_hex_bytes(&[0xCD; 32]);
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

fn tiny_manifest(fixture: &TinyFixture, oracle_bytes: &[u8], cases: Value) -> std::path::PathBuf {
    std::fs::write(&fixture.oracle, oracle_bytes).unwrap();
    let staging_sha = sha256_hex_bytes(&std::fs::read(&fixture.staging).unwrap());
    let manifest = json!({
        "schema": "sceneworks.h3.actual_checkpoint_probe",
        "schema_version": 1,
        "backend": "mlx",
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
            "sha256": sha256_hex_bytes(oracle_bytes),
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

fn convrot_case(base: &str) -> Value {
    json!({
        "id": "convrot.tiny",
        "kind": "convrot_projection",
        "select": {
            "base": base,
            "bias": false,
            "expect_route": "convrot",
        },
        "input": {"dtype": "BF16", "shape": [1, 2], "values": [1.0, -2.0]},
        "tolerance": {"max_abs": 0.0},
        "selected_tensors": [
            format!("{base}.weight"),
            format!("{base}.weight_scale"),
            format!("{base}.comfy_quant"),
        ],
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
    let manifest = tiny_manifest(&fixture, &oracle_bytes, json!([convrot_case("p")]));
    let mut executed = false;
    let result = run_probe_at(
        "mlx",
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
    assert!(!executed);
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(fixture.receipt_dir.join("mlx.manifest-refusal.json")).unwrap(),
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
    let manifest = tiny_manifest(&fixture, &oracle_bytes, json!([convrot_case("absent")]));
    let mut executed = false;
    let result = run_probe_at(
        "mlx",
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
    assert!(!executed);
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(
            fixture
                .receipt_dir
                .join(receipt_file_name("mlx", "convrot.tiny")),
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
    let manifest = tiny_manifest(&fixture, &oracle_bytes, json!([convrot_case("p")]));
    let mut executed = false;
    let result = run_probe_at(
        "mlx",
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
    assert!(!executed);
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(
            fixture
                .receipt_dir
                .join(receipt_file_name("mlx", "convrot.tiny")),
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
    std::fs::write(&manifest, b"{ not json").unwrap();
    let result = run_probe_at(
        "mlx",
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
        &std::fs::read(fixture.receipt_dir.join("mlx.manifest-refusal.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["refusal"]["code"], "manifest_malformed");
    assert_eq!(receipt["payload_reads"], 0);
}

#[test]
fn staging_projection_must_bind_a_full_file_verification() {
    let fixture = tiny_fixture();
    let oracle_bytes = oracle_document(
        &fixture.artifact_sha,
        &"11".repeat(32),
        &"22".repeat(32),
        &"33".repeat(32),
    );
    let manifest = tiny_manifest(&fixture, &oracle_bytes, json!([convrot_case("p")]));
    let mut staging: Value =
        serde_json::from_slice(&std::fs::read(&fixture.staging).unwrap()).unwrap();
    staging["full_file_digest_verified"] = json!(false);
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
        "mlx",
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
        &std::fs::read(fixture.receipt_dir.join("mlx.manifest-refusal.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["refusal"]["code"], "staging_binding_mismatch");
    assert_eq!(receipt["payload_reads"], 0);
}

#[test]
fn oversized_manifest_refuses_before_any_payload_read() {
    let fixture = tiny_fixture();
    let manifest = fixture._dir.path().join("huge.json");
    let mut bytes = b"{\"schema\":".to_vec();
    bytes.resize(support::MAX_MANIFEST_BYTES as usize + 1, b' ');
    std::fs::write(&manifest, &bytes).unwrap();
    let result = run_probe_at(
        "mlx",
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
        &std::fs::read(fixture.receipt_dir.join("mlx.manifest-refusal.json")).unwrap(),
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
    let manifest = tiny_manifest(&fixture, &oracle_bytes, json!([convrot_case("p")]));
    let mut executed = false;
    let result = run_probe_at(
        "mlx",
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
                .join(receipt_file_name("mlx", "convrot.tiny")),
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
    let manifest = tiny_manifest(&fixture, &oracle_bytes, json!([convrot_case("p")]));
    // A regular file cannot be used as the receipt directory: the selector must go RED instead of
    // silently returning a receipt list with a fake "{}" success.
    let blocked = fixture._dir.path().join("blocked-receipts");
    std::fs::write(&blocked, b"not a directory").unwrap();
    let mut executed = false;
    let result = run_probe_at(
        "mlx",
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
    let manifest = tiny_manifest(&fixture, &oracle_bytes, json!([convrot_case("p")]));
    let mut executed = false;
    let result = run_probe_at(
        "mlx",
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
                .join(receipt_file_name("mlx", "convrot.tiny")),
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
