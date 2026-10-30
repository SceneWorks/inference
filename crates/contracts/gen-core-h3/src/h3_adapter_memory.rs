//! Header-only allocation accounting for the actual H3 residual installers.
//!
//! This is not a file-size overlay: MLX dense LoKr retains a full BF16 delta, whereas
//! its packed route retains composed Kronecker factors. Candle retains source-dtype
//! LoRA factors and F32 composed LoKr factors, plus their prepared compute copies.
//! All peaks below are source-level upper bounds (views may share storage and MLX
//! may fuse operations), not measured allocator high-water marks.

use std::collections::BTreeMap;

use crate::h3_memory::{NativeBackend, RowDomain};
pub use gen_core::h3_adapter_memory::{
    H3AdapterMemoryProfile, H3AdapterProjectionMemory, H3AdapterTarget,
};
use gen_core::weightsmeta::{self, Dtype, SafetensorsTensorHeader};
use gen_core::{AdapterSpec, Error, Result};

#[derive(Clone, Debug)]
pub struct H3AdapterFileHeaders {
    pub tensors: Vec<SafetensorsTensorHeader>,
    pub metadata: BTreeMap<String, String>,
    /// Serialized source buffer retained by Candle's read_adapter. Used only at
    /// construction, never as installed residency. Synthetic header fixtures may
    /// set zero, in which case the tensor payload sum is used.
    pub source_buffer_bytes: u64,
    pub disabled: bool,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Unsupported(format!("H3 adapter allocation: {}", message.into()))
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("byte count overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("byte count overflow"))
}
fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0, add)
}
fn width(h: &SafetensorsTensorHeader) -> Result<u64> {
    if !h.is_float() {
        return Err(invalid(format!("{}: nonfloating factor", h.name)));
    }
    Ok(h.dtype.size() as u64)
}
fn matrix(h: &SafetensorsTensorHeader) -> Result<(u64, u64)> {
    if h.shape.len() != 2 || h.shape.contains(&0) {
        return Err(invalid(format!("{}: expected nonempty matrix", h.name)));
    }
    Ok((h.shape[0] as u64, h.shape[1] as u64))
}

/// Header and metadata reads only. No tensor payload (including LoRA factors) is read.
pub fn profile_stack(
    backend: NativeBackend,
    specs: &[AdapterSpec],
    targets: &[H3AdapterTarget],
) -> Result<H3AdapterMemoryProfile> {
    let files = specs
        .iter()
        .map(|spec| {
            Ok(H3AdapterFileHeaders {
                tensors: weightsmeta::safetensors_path_tensor_headers(&spec.path)?,
                metadata: weightsmeta::safetensors_file_metadata(&spec.path)?,
                source_buffer_bytes: std::fs::metadata(&spec.path)?.len(),
                disabled: spec.scale == 0.0,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    profile_from_headers(backend, &files, targets)
}

const PREFIXES: [&str; 4] = [
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "model.",
];
fn normalize(path: &str) -> String {
    PREFIXES
        .iter()
        .find_map(|prefix| path.strip_prefix(prefix))
        .unwrap_or(path)
        .to_string()
}
fn raw_container(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("blocks.") {
        format!("transformer_blocks.{rest}")
    } else if let Some(rest) = path.strip_prefix("token_refiner.blocks.") {
        format!("token_refiner.refiner_blocks.{rest}")
    } else {
        path.to_string()
    }
}
fn lora_key(key: &str) -> Option<(String, bool)> {
    for (suffix, down) in [
        (".lora_A.default.weight", true),
        (".lora_B.default.weight", false),
        (".lora_down.default.weight", true),
        (".lora_up.default.weight", false),
        (".lora_A.weight", true),
        (".lora_B.weight", false),
        (".lora_down.weight", true),
        (".lora_up.weight", false),
    ] {
        if let Some(path) = key.strip_suffix(suffix) {
            return Some((normalize(path), down));
        }
    }
    None
}

/// Source key routing mirrors H3's Comfy conversion. Fused QKV rank is bounded by
/// the shared-A route: deciding the block-diagonal optimization needs factor values,
/// so the larger supported route is used without reading any weight tensor. This
/// bounds both shared-A and block-diagonal files, never rejects either representation.
fn lora_routes(path: &str) -> Vec<String> {
    let path = raw_container(path);
    for (raw, runtime) in [
        (".attn.out_proj", ".attn.to_out.0"),
        (".mlp.fc1", ".ff.net.0.proj"),
        (".mlp.fc2", ".ff.net.2"),
    ] {
        if let Some(root) = path.strip_suffix(raw) {
            return vec![format!("{root}{runtime}")];
        }
    }
    if let Some(root) = path.strip_suffix(".attn.qkv_proj") {
        return ["to_q", "to_k", "to_v"]
            .into_iter()
            .map(|name| format!("{root}.attn.{name}"))
            .collect();
    }
    vec![path]
}

/// Source-selected shapes, including mixed dense/affine/native packed targets.
/// Native projections must be supplied from validated descriptor parsing; affine
/// logical input follows H3's fixed group-64 loader, not a requested quant tier.
pub fn targets_from_headers(
    headers: &[SafetensorsTensorHeader],
    native: &[crate::h3_memory::NativeProjection],
    paths: &[String],
    activation_width: u64,
) -> Result<Vec<H3AdapterTarget>> {
    let by_name = headers
        .iter()
        .map(|h| (h.name.as_str(), h))
        .collect::<BTreeMap<_, _>>();
    paths
        .iter()
        .map(|path| {
            if let Some(p) = native.iter().find(|p| &p.base == path) {
                return Ok(H3AdapterTarget {
                    path: path.clone(),
                    input: p.input,
                    output: p.output,
                    packed: true,
                    activation_width: p.activation_width,
                    activation_dtype: if p.activation_width == 4 {
                        Dtype::F32
                    } else {
                        Dtype::BF16
                    },
                    rows: p.rows,
                });
            }
            let h = by_name
                .get(format!("{path}.weight").as_str())
                .ok_or_else(|| invalid(format!("missing selected target {path}")))?;
            let (output, stored_input) = matrix(h)?;
            let scales = by_name.get(format!("{path}.scales").as_str());
            let (input, packed) = if let Some(scales) = scales {
                let (n, groups) = matrix(scales)?;
                let input = mul(groups, 64)?;
                let bits = mul(stored_input, 32)?;
                if h.dtype != Dtype::U32
                    || n != output
                    || bits % input != 0
                    || ![4, 8].contains(&(bits / input))
                {
                    return Err(invalid(format!("invalid affine target {path}")));
                }
                (input, true)
            } else {
                width(h)?;
                (stored_input, false)
            };
            Ok(H3AdapterTarget {
                path: path.clone(),
                input,
                output,
                packed,
                activation_width,
                activation_dtype: if activation_width == 4 {
                    Dtype::F32
                } else {
                    Dtype::BF16
                },
                rows: if path.starts_with("token_refiner.") {
                    RowDomain::Conditioning
                } else {
                    RowDomain::Packed
                },
            })
        })
        .collect()
}

/// Ordered stack construction: one file's source map is live at a time, and only
/// already-installed originals from earlier files overlap it. Caches are forward
/// allocations, not phantom warmed state at construction.
pub fn profile_from_headers(
    backend: NativeBackend,
    files: &[H3AdapterFileHeaders],
    targets: &[H3AdapterTarget],
) -> Result<H3AdapterMemoryProfile> {
    let mut profile = H3AdapterMemoryProfile::default();
    let mut installed = 0;
    for file in files {
        let source = sum(file.tensors.iter().map(|h| h.data_bytes))?;
        // Match the install's classifier ordering even for a file containing LoKr.
        weightsmeta::classify_minimax_h3_trainer_namespace(
            file.tensors.iter().map(|h| h.name.as_str()),
            file.metadata.get("ss_network_module").map(String::as_str),
            file.metadata.get("networkType").map(String::as_str),
        )
        .map_err(invalid)?;
        let before = profile.projections.len();
        let lokr = file.tensors.iter().any(|h| h.name.contains(".lokr_w"))
            || file
                .metadata
                .get("networkType")
                .is_some_and(|s| s == "lokr");
        let temporary = if lokr {
            price_lokr(backend, file, targets, &mut profile.projections)?
        } else {
            price_lora(backend, file, targets, &mut profile.projections)?
        };
        if before == profile.projections.len() {
            return Err(invalid("adapter file has no matching targets"));
        }
        let current = sum(profile.projections[before..]
            .iter()
            .map(|p| p.installed_bytes))?;
        // Candle's serialized buffer drops when read_adapter returns, BEFORE
        // conversion/device installation. Take the maximum of those lifetimes,
        // never charge the serialized copy beside future device payloads.
        let read_peak = match backend {
            NativeBackend::Candle { .. } => {
                sum([installed, source, file.source_buffer_bytes.max(source)])?
            }
            NativeBackend::Mlx { .. } => add(installed, source)?,
        };
        let construct_peak = sum([installed, source, current, temporary])?;
        profile.install_peak_bytes = profile
            .install_peak_bytes
            .max(read_peak)
            .max(construct_peak);
        installed = add(installed, current)?;
    }
    profile.resident_bytes = sum(profile.projections.iter().map(|p| p.resident_bytes))?;
    Ok(profile)
}

fn price_lora(
    backend: NativeBackend,
    file: &H3AdapterFileHeaders,
    targets: &[H3AdapterTarget],
    out: &mut Vec<H3AdapterProjectionMemory>,
) -> Result<u64> {
    let trainer = weightsmeta::classify_minimax_h3_trainer_namespace(
        file.tensors.iter().map(|h| h.name.as_str()),
        file.metadata.get("ss_network_module").map(String::as_str),
        file.metadata.get("networkType").map(String::as_str),
    )
    .map_err(invalid)?;
    if trainer {
        weightsmeta::validate_minimax_h3_trainer_layout(
            file.tensors
                .iter()
                .map(|h| (h.name.clone(), h.shape.clone())),
            file.metadata.get("ss_network_module").map(String::as_str),
            file.metadata
                .get(weightsmeta::MINIMAX_H3_TOKEN_REFINER_METADATA_KEY)
                .map(String::as_str),
        )
        .map_err(invalid)?;
    }
    let mut groups = BTreeMap::<
        String,
        (
            Option<&SafetensorsTensorHeader>,
            Option<&SafetensorsTensorHeader>,
        ),
    >::new();
    for h in &file.tensors {
        let classified = if trainer {
            weightsmeta::parse_minimax_h3_trainer_key(&h.name).and_then(|p| {
                use weightsmeta::MiniMaxH3TrainerRole;
                match p.role {
                    MiniMaxH3TrainerRole::Alpha => None,
                    role => Some((
                        format!("blocks.{}.{}", p.block, p.leaf.dotted()),
                        role == MiniMaxH3TrainerRole::Down,
                    )),
                }
            })
        } else {
            lora_key(&h.name)
        };
        let Some((path, down)) = classified else {
            continue;
        };
        let pair = groups.entry(path).or_default();
        if down {
            pair.0 = Some(h);
        } else {
            pair.1 = Some(h);
        }
    }
    let mut temporary = 0u64;
    let mut construction = 0u64;
    for (path, (down, up)) in groups {
        let (Some(down), Some(up)) = (down, up) else {
            return Err(invalid(format!("orphan LoRA target {path}")));
        };
        let (rank, input) = matrix(down)?;
        let (source_output, up_rank) = matrix(up)?;
        if rank != up_rank {
            return Err(invalid(format!("rank mismatch {path}")));
        }
        let (dw, uw) = (width(down)?, width(up)?);
        let routes = lora_routes(&path);
        let fused = routes.len() == 3;
        if fused && source_output % 3 != 0 {
            return Err(invalid("unequal fused QKV outputs"));
        }
        let output = source_output / routes.len() as u64;
        // Swapped FFN up factors and fused split conversions can independently own
        // their source-shaped copy, before final orientation/fold/device transfer.
        if fused || path.ends_with(".mlp.fc1") {
            temporary = add(temporary, up.data_bytes)?;
        }
        for path in routes {
            let target = targets.iter().find(|t| t.path == path).ok_or_else(|| {
                invalid(format!(
                    "target {path} is absent or not adapter-addressable"
                ))
            })?;
            if target.input != input || target.output != output {
                return Err(invalid(format!("factor/base shape mismatch {path}")));
            }
            let a = mul(rank, input)?;
            let b = mul(rank, output)?;
            let installed = sum([mul(a, dw)?, mul(b, uw)?])?;
            let e = target.activation_width;
            let prepared = match backend {
                NativeBackend::Candle { .. } if !file.disabled => sum([
                    if down.dtype == target.activation_dtype {
                        0
                    } else {
                        mul(a, e)?
                    },
                    if up.dtype == target.activation_dtype {
                        0
                    } else {
                        mul(b, e)?
                    },
                ])?,
                _ => 0,
            };
            // Transposed A and folded/transposed B construction: both may allocate
            // on Candle; MLX transpose is a view and B's multiply owns one payload.
            construction = match backend {
                // Eager target construction ends before the next target starts.
                NativeBackend::Candle { .. } => construction.max(sum([installed, mul(b, uw)?])?),
                // MLX's installed lazy nodes can retain the construction graph.
                NativeBackend::Mlx { .. } => add(construction, mul(b, uw)?)?,
            };
            let promoted = match backend {
                // MLX promotes mixed FP16/BF16 to F32, despite equal widths.
                NativeBackend::Mlx { .. }
                    if [down.dtype, up.dtype, target.activation_dtype].contains(&Dtype::F16)
                        && [down.dtype, up.dtype, target.activation_dtype]
                            .contains(&Dtype::BF16) =>
                {
                    4.max(dw).max(uw)
                }
                NativeBackend::Mlx { .. } => e.max(dw).max(uw),
                NativeBackend::Candle { .. } => e,
            };
            // Low-rank intermediate, residual, scaled residual, narrowing and add.
            // Includes input promotion where the MLX mixed-dtype matmul needs one.
            let forward = if file.disabled {
                0
            } else {
                sum([
                    mul(rank, promoted)?,
                    mul(output, promoted * 2)?,
                    mul(output, e * 2)?,
                    if promoted > e {
                        mul(input, promoted)?
                    } else {
                        0
                    },
                ])?
            };
            let fixed = if file.disabled || matches!(backend, NativeBackend::Candle { .. }) {
                0
            } else {
                // Mixed-dtype matmul may materialize promoted factor operands.
                sum([
                    if dw == promoted { 0 } else { mul(a, promoted)? },
                    if uw == promoted { 0 } else { mul(b, promoted)? },
                ])?
            };
            out.push(H3AdapterProjectionMemory {
                target: target.clone(),
                installed_bytes: installed,
                resident_bytes: add(installed, prepared)?,
                forward_fixed_bytes: fixed,
                forward_bytes_per_row: forward,
            });
        }
    }
    add(temporary, construction)
}

fn lokr_leg(
    parts: &BTreeMap<&str, &SafetensorsTensorHeader>,
    name: &str,
) -> Result<(u64, u64, bool)> {
    if let Some(h) = parts.get(name) {
        width(h)?;
        if h.shape.len() < 2 || h.shape.contains(&0) {
            return Err(invalid(format!("{}: invalid LoKr factor shape", h.name)));
        }
        let rows = h.shape[0] as u64;
        return Ok((rows, h.element_count()? / rows, h.shape.len() == 2));
    }
    let a = parts
        .get(format!("{name}_a").as_str())
        .ok_or_else(|| invalid(format!("missing {name}_a")))?;
    let b = parts
        .get(format!("{name}_b").as_str())
        .ok_or_else(|| invalid(format!("missing {name}_b")))?;
    width(a)?;
    width(b)?;
    let (m, k) = matrix(a)?;
    let (k2, n) = matrix(b)?;
    if k != k2 {
        return Err(invalid("LoKr inner factor mismatch"));
    }
    Ok((m, n, true))
}

fn price_lokr(
    backend: NativeBackend,
    file: &H3AdapterFileHeaders,
    targets: &[H3AdapterTarget],
    out: &mut Vec<H3AdapterProjectionMemory>,
) -> Result<u64> {
    let mut groups = BTreeMap::<String, BTreeMap<&str, &SafetensorsTensorHeader>>::new();
    for h in &file.tensors {
        for factor in [
            "lokr_w1_a",
            "lokr_w1_b",
            "lokr_w1",
            "lokr_w2_a",
            "lokr_w2_b",
            "lokr_w2",
        ] {
            if let Some(path) = h.name.strip_suffix(format!(".{factor}").as_str()) {
                let path = match backend {
                    // MLX apply_lokr uses Dotted resolution: unlike Candle, no
                    // Comfy/PEFT prefix rewrite is applied to native LoKr groups.
                    NativeBackend::Mlx { .. } => path.to_string(),
                    NativeBackend::Candle { .. } => normalize(path),
                };
                groups.entry(path).or_default().insert(factor, h);
                break;
            }
        }
    }
    let mut temporary = 0u64;
    for (path, parts) in groups {
        let (a, c, linear1) = lokr_leg(&parts, "lokr_w1")?;
        let (b, d, linear2) = lokr_leg(&parts, "lokr_w2")?;
        let structured = linear1 && linear2;
        if !structured && matches!(backend, NativeBackend::Candle { .. }) {
            return Err(invalid("Candle H3 installer requires matrix LoKr factors"));
        }
        let (input, output) = (mul(c, d)?, mul(a, b)?);
        let paths = match backend {
            NativeBackend::Mlx { .. } => vec![path.clone()],
            NativeBackend::Candle { .. } => {
                if let Some(target) = targets
                    .iter()
                    .find(|t| format!("lora_unet_{}", t.path.replace('.', "_")) == path)
                {
                    vec![target.path.clone()]
                } else {
                    lora_routes(&path)
                }
            }
        };
        let f1 = mul(a, c)?;
        let f2 = mul(b, d)?;
        let factors = add(f1, f2)?;
        let source_width = parts
            .values()
            .map(|h| width(h))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(4)
            .max(4);
        for target_path in &paths {
            let target = targets
                .iter()
                .find(|t| &t.path == target_path)
                .ok_or_else(|| {
                    invalid(format!(
                        "target {target_path} is absent or not adapter-addressable"
                    ))
                })?;
            let matches = if structured {
                target.input == input && mul(target.output, paths.len() as u64)? == output
            } else {
                mul(target.input, target.output)? == mul(input, output)?
            };
            if !matches {
                return Err(invalid(format!("LoKr/base shape mismatch {target_path}")));
            }
            let e = target.activation_width;
            let dense =
                matches!(backend, NativeBackend::Mlx { .. }) && (!target.packed || !structured);
            let installed = if dense {
                mul(mul(input, output)?, 2)?
            } else {
                mul(
                    factors,
                    if matches!(backend, NativeBackend::Mlx { .. }) {
                        2
                    } else {
                        4
                    },
                )?
            };
            let prepared = match backend {
                NativeBackend::Candle { .. } if !file.disabled => {
                    // Prepared w2.t().contiguous owns a tensor even at F32. At
                    // BF16, w1 and w2 both own casts; w2's pretranspose cast is
                    // scratch, not a second retained cache.
                    add(if e == 4 { 0 } else { mul(f1, e)? }, mul(f2, e)?)?
                }
                _ => 0,
            };
            // Source composition, scale output, dtype conversion and (Candle)
            // host/device copies. Sum live graph operands, not file length.
            let construction = sum([
                mul(factors, source_width)?,
                mul(f2, source_width)?,
                if dense {
                    mul(mul(input, output)?, source_width * 2)?
                } else {
                    installed
                },
            ])?;
            temporary = match backend {
                NativeBackend::Candle { .. } => temporary.max(construction),
                NativeBackend::Mlx { .. } => add(temporary, construction)?,
            };
            let fixed = if file.disabled {
                0
            } else {
                match backend {
                    NativeBackend::Mlx { .. }
                        if dense && target.activation_dtype != Dtype::BF16 =>
                    {
                        mul(mul(input, output)?, e)?
                    }
                    NativeBackend::Mlx { .. }
                        if !dense && target.activation_dtype != Dtype::BF16 =>
                    {
                        mul(factors, e)?
                    }
                    NativeBackend::Candle { .. } if e != 4 => mul(f2, e)?,
                    _ => 0,
                }
            };
            let row_bytes = if file.disabled {
                0
            } else if dense {
                mul(target.output, e * 4)?
            } else {
                // MLX contracts w1 first: [N,a,d]. Candle contracts w2 first:
                // [N,c,b], retains the fused output before slicing, and may copy
                // contiguous input/broadcasted tiny left factor for batched GEMM.
                let intermediate = match backend {
                    NativeBackend::Mlx { .. } => mul(a, d)?,
                    NativeBackend::Candle { .. } => sum([mul(c, b)?, input, mul(a, c)?])?,
                };
                mul(
                    sum([intermediate, mul(output, 2)?, mul(target.output, 2)?])?,
                    e,
                )?
            };
            out.push(H3AdapterProjectionMemory {
                target: target.clone(),
                installed_bytes: installed,
                resident_bytes: add(installed, prepared)?,
                forward_fixed_bytes: fixed,
                forward_bytes_per_row: row_bytes,
            });
        }
    }
    Ok(temporary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::h3_memory::{NativeRows, FFN_INTERMEDIATE_BUDGET};

    const MLX: NativeBackend = NativeBackend::Mlx { tile_columns: 64 };
    const CANDLE: NativeBackend = NativeBackend::Candle {
        tile_rows: 64,
        workspace_bytes: 0,
    };

    fn h(name: &str, dtype: Dtype, shape: &[usize]) -> SafetensorsTensorHeader {
        SafetensorsTensorHeader {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: shape.iter().product::<usize>() as u64 * dtype.size() as u64,
        }
    }
    fn target(path: &str, input: u64, output: u64, packed: bool) -> H3AdapterTarget {
        H3AdapterTarget {
            path: path.into(),
            input,
            output,
            packed,
            activation_width: 2,
            activation_dtype: Dtype::BF16,
            rows: if path.starts_with("token_refiner.") {
                RowDomain::Conditioning
            } else {
                RowDomain::Packed
            },
        }
    }
    fn file(tensors: Vec<SafetensorsTensorHeader>) -> H3AdapterFileHeaders {
        H3AdapterFileHeaders {
            tensors,
            metadata: BTreeMap::new(),
            source_buffer_bytes: 0,
            disabled: false,
        }
    }
    fn lokr(path: &str) -> H3AdapterFileHeaders {
        file(vec![
            h(&format!("{path}.lokr_w1"), Dtype::BF16, &[4, 4]),
            h(&format!("{path}.lokr_w2_a"), Dtype::BF16, &[1792, 1]),
            h(&format!("{path}.lokr_w2_b"), Dtype::BF16, &[1, 1792]),
        ])
    }
    fn rows(packed: u64, conditioning: u64) -> NativeRows {
        NativeRows {
            task: crate::h3_memory::NativeTaskSelection::Base,
            packed,
            conditioning,
            schedule: 0,
            visual_elements: 0,
            global: 0,
        }
    }

    #[test]
    fn tiny_lokr_file_retains_actual_dense_delta_on_mlx() {
        let path = "transformer_blocks.0.attn.to_q";
        let f = lokr(path);
        let source: u64 = f.tensors.iter().map(|h| h.data_bytes).sum();
        let p = profile_from_headers(MLX, &[f], &[target(path, 7168, 7168, false)]).unwrap();
        assert_eq!(p.resident_bytes, 102_760_448);
        assert!(p.resident_bytes > source * 1000);
        assert!(p.install_peak_bytes > p.resident_bytes);
    }

    #[test]
    fn packed_lokr_composes_small_factors_not_saved_lowrank_payload() {
        let path = "transformer_blocks.0.attn.to_q";
        let p =
            profile_from_headers(MLX, &[lokr(path)], &[target(path, 7168, 7168, true)]).unwrap();
        assert_eq!(p.resident_bytes, (4 * 4 + 1792 * 1792) * 2);
        assert!(p.resident_bytes < 102_760_448);
    }

    #[test]
    fn candle_lokr_includes_f32_originals_and_bf16_prepared_cache() {
        let path = "transformer_blocks.0.attn.to_q";
        let p = profile_from_headers(CANDLE, &[lokr(path)], &[target(path, 7168, 7168, false)])
            .unwrap();
        assert_eq!(p.projections[0].installed_bytes, (16 + 1792 * 1792) * 4);
        assert_eq!(p.resident_bytes, (16 + 1792 * 1792) * 6);
        assert_eq!(p.projections[0].forward_fixed_bytes, 1792 * 1792 * 2);
    }

    #[test]
    fn candle_lora_prices_actual_source_dtype_and_prepared_pair() {
        let path = "transformer_blocks.0.attn.to_q";
        for dtype in [Dtype::BF16, Dtype::F32] {
            let f = file(vec![
                h(&format!("{path}.lora_A.default.weight"), dtype, &[3, 8]),
                h(&format!("{path}.lora_B.default.weight"), dtype, &[12, 3]),
            ]);
            let p = profile_from_headers(CANDLE, &[f], &[target(path, 8, 12, true)]).unwrap();
            let count = (8 + 12) * 3;
            let width = if dtype == Dtype::F32 { 6 } else { 2 };
            assert_eq!(p.resident_bytes, count * width);
        }
    }

    #[test]
    fn mixed_target_encodings_are_not_a_global_tier_switch() {
        let q = "transformer_blocks.0.attn.to_q";
        let k = "transformer_blocks.0.attn.to_k";
        let mut f = lokr(q);
        f.tensors.extend(lokr(k).tensors);
        let p = profile_from_headers(
            MLX,
            &[f],
            &[target(q, 7168, 7168, false), target(k, 7168, 7168, true)],
        )
        .unwrap();
        assert_eq!(
            p.projections[0].installed_bytes + p.projections[1].installed_bytes,
            102_760_448 + (16 + 1792 * 1792) * 2
        );
    }

    #[test]
    fn comfy_qkv_and_swapped_ffn_are_priced_without_reading_factor_values() {
        let f = file(vec![
            h(
                "blocks.0.attn.qkv_proj.lora_down.weight",
                Dtype::BF16,
                &[6, 8],
            ),
            h(
                "blocks.0.attn.qkv_proj.lora_up.weight",
                Dtype::BF16,
                &[36, 6],
            ),
            h("blocks.0.mlp.fc1.lora_A.weight", Dtype::F32, &[2, 8]),
            h("blocks.0.mlp.fc1.lora_B.weight", Dtype::F32, &[32, 2]),
        ]);
        let mut targets = ["to_q", "to_k", "to_v"]
            .into_iter()
            .map(|n| target(&format!("transformer_blocks.0.attn.{n}"), 8, 12, true))
            .collect::<Vec<_>>();
        targets.push(target("transformer_blocks.0.ff.net.0.proj", 8, 32, false));
        let p = profile_from_headers(CANDLE, &[f], &targets).unwrap();
        // Shared-A rank-six route bounds block-diagonal rank-two conversion as well.
        assert_eq!(p.resident_bytes, 3 * (8 + 12) * 6 * 2 + (8 + 32) * 2 * 6);
        assert_eq!(p.projections.len(), 4);
    }

    #[test]
    fn candle_fused_lokr_keeps_each_full_factor_cache_and_conditioning_rows() {
        let f = file(vec![
            h(
                "token_refiner.refiner_blocks.0.attn.qkv_proj.lokr_w1",
                Dtype::F32,
                &[3, 2],
            ),
            h(
                "token_refiner.refiner_blocks.0.attn.qkv_proj.lokr_w2",
                Dtype::F32,
                &[12, 4],
            ),
        ]);
        let targets = ["to_q", "to_k", "to_v"]
            .into_iter()
            .map(|n| {
                target(
                    &format!("token_refiner.refiner_blocks.0.attn.{n}"),
                    8,
                    12,
                    true,
                )
            })
            .collect::<Vec<_>>();
        let p = profile_from_headers(CANDLE, &[f], &targets).unwrap();
        assert_eq!(p.resident_bytes, 3 * (3 * 2 + 12 * 4) * 6);
        assert_eq!(p.forward_working_bytes(CANDLE, rows(900, 0)), 0);
        assert!(p.forward_working_bytes(CANDLE, rows(0, 2)) > 0);
    }

    #[test]
    fn ordered_install_peak_is_not_steady_residency_plus_peak() {
        let path = "transformer_blocks.0.attn.to_q";
        let targets = [target(path, 8, 12, true)];
        let small = file(vec![
            h(&format!("{path}.lora_A.weight"), Dtype::BF16, &[1, 8]),
            h(&format!("{path}.lora_B.weight"), Dtype::BF16, &[12, 1]),
        ]);
        let big = file(vec![
            h(&format!("{path}.lora_A.weight"), Dtype::F32, &[100, 8]),
            h(&format!("{path}.lora_B.weight"), Dtype::F32, &[12, 100]),
        ]);
        let a = profile_from_headers(CANDLE, &[small.clone(), big.clone()], &targets).unwrap();
        let b = profile_from_headers(CANDLE, &[big, small], &targets).unwrap();
        assert_eq!(a.resident_bytes, b.resident_bytes);
        assert_ne!(a.install_peak_bytes, b.install_peak_bytes);
    }

    #[test]
    fn excluded_adaln_target_is_not_silently_dropped_on_either_topology() {
        let f = file(vec![
            h(
                "transformer_blocks.0.adaln_proj.linear.lora_A.weight",
                Dtype::BF16,
                &[1, 8],
            ),
            h(
                "transformer_blocks.0.adaln_proj.linear.lora_B.weight",
                Dtype::BF16,
                &[12, 1],
            ),
        ]);
        for backend in [MLX, CANDLE] {
            assert!(profile_from_headers(
                backend,
                std::slice::from_ref(&f),
                &[target("transformer_blocks.0.attn.to_q", 8, 12, true)]
            )
            .is_err());
        }
    }

    #[test]
    fn affine_logical_shape_is_source_encoded_not_q4_assumed() {
        for (columns, expected) in [(16, 4), (32, 8)] {
            let headers = [
                h(
                    "transformer_blocks.0.attn.to_q.weight",
                    Dtype::U32,
                    &[12, columns],
                ),
                h(
                    "transformer_blocks.0.attn.to_q.scales",
                    Dtype::BF16,
                    &[12, 2],
                ),
            ];
            let paths = ["transformer_blocks.0.attn.to_q".to_string()];
            let targets = targets_from_headers(&headers, &[], &paths, 2).unwrap();
            assert_eq!(targets[0].input, 128);
            assert_eq!(columns as u64 * 32 / targets[0].input, expected);
            assert!(targets[0].packed);
        }
    }

    #[test]
    fn candle_ffn_scratch_uses_the_actual_outer_tile_cap() {
        let t = target("transformer_blocks.0.ff.net.0.proj", 8, 32, false);
        let p = H3AdapterProjectionMemory {
            target: t,
            installed_bytes: 0,
            resident_bytes: 0,
            forward_fixed_bytes: 77,
            forward_bytes_per_row: 3,
        };
        let cap = gen_core::attention_budget::AttentionBudget::from_score_elements(
            FFN_INTERMEDIATE_BUDGET,
            false,
        )
        .query_block_rows(32, 1_000_000_000);
        assert_eq!(
            p.forward_working_bytes(CANDLE, rows(1_000_000_000, 0)),
            77 + cap * 3
        );
        assert_eq!(p.forward_working_bytes(CANDLE, rows(0, 0)), 0);
    }

    #[test]
    fn fp16_is_not_a_bf16_prepared_cache_alias() {
        let path = "transformer_blocks.0.attn.to_q";
        let f = file(vec![
            h(&format!("{path}.lora_A.weight"), Dtype::F16, &[2, 8]),
            h(&format!("{path}.lora_B.weight"), Dtype::F16, &[12, 2]),
        ]);
        let p = profile_from_headers(CANDLE, &[f], &[target(path, 8, 12, true)]).unwrap();
        assert_eq!(p.resident_bytes, (8 + 12) * 2 * 4);
    }

    #[test]
    fn exact_trainer_census_uses_the_same_qkv_and_ffn_routes() {
        let mut f = file(Vec::new());
        f.metadata.insert(
            "ss_network_module".into(),
            "networks.lora_minimax_h3".into(),
        );
        f.metadata.insert(
            weightsmeta::MINIMAX_H3_TOKEN_REFINER_METADATA_KEY.into(),
            "False".into(),
        );
        let mut targets = Vec::new();
        for index in 0..50 {
            for (leaf, input, output) in [
                ("attn_qkv_proj", 5376, 21504),
                ("attn_out_proj", 7168, 5376),
                ("mlp_fc1", 5376, 28672),
                ("mlp_fc2", 14336, 5376),
            ] {
                let stem = format!("lora_unet_blocks_{index}_{leaf}");
                f.tensors.push(h(
                    &format!("{stem}.lora_down.weight"),
                    Dtype::BF16,
                    &[3, input],
                ));
                f.tensors.push(h(
                    &format!("{stem}.lora_up.weight"),
                    Dtype::BF16,
                    &[output, 3],
                ));
                f.tensors
                    .push(h(&format!("{stem}.alpha"), Dtype::F32, &[1]));
            }
            for leaf in ["to_q", "to_k", "to_v"] {
                targets.push(target(
                    &format!("transformer_blocks.{index}.attn.{leaf}"),
                    5376,
                    7168,
                    true,
                ));
            }
            for (leaf, input, output) in [
                ("attn.to_out.0", 7168, 5376),
                ("ff.net.0.proj", 5376, 28672),
                ("ff.net.2", 14336, 5376),
            ] {
                targets.push(target(
                    &format!("transformer_blocks.{index}.{leaf}"),
                    input,
                    output,
                    true,
                ));
            }
        }
        let p = profile_from_headers(MLX, &[f], &targets).unwrap();
        assert_eq!(p.projections.len(), 50 * 6);
        assert!(p
            .projections
            .iter()
            .all(|p| p.target.path.starts_with("transformer_blocks.")));
    }

    #[test]
    fn nondeferrable_mlx_factor_shape_still_prices_a_dense_delta_on_packed_base() {
        let path = "transformer_blocks.0.attn.to_q";
        let f = file(vec![
            h(&format!("{path}.lokr_w1"), Dtype::BF16, &[2, 2]),
            h(&format!("{path}.lokr_w2"), Dtype::BF16, &[4, 4, 1, 1]),
        ]);
        let p = profile_from_headers(MLX, &[f], &[target(path, 8, 8, true)]).unwrap();
        assert_eq!(p.resident_bytes, 8 * 8 * 2);
    }

    #[test]
    fn disabled_residual_keeps_installed_payload_without_preparation_or_scratch() {
        let path = "transformer_blocks.0.attn.to_q";
        let mut f = lokr(path);
        f.disabled = true;
        let p = profile_from_headers(CANDLE, &[f], &[target(path, 7168, 7168, true)]).unwrap();
        assert_eq!(p.resident_bytes, p.projections[0].installed_bytes);
        assert!(p.resident_bytes > 0);
        assert_eq!(p.forward_working_bytes(CANDLE, rows(100, 100)), 0);
    }

    #[test]
    fn f32_lokr_preparation_still_owns_the_transposed_right_factor() {
        let path = "transformer_blocks.0.attn.to_q";
        let f = file(vec![
            h(&format!("{path}.lokr_w1"), Dtype::BF16, &[2, 2]),
            h(&format!("{path}.lokr_w2"), Dtype::BF16, &[4, 4]),
        ]);
        let mut t = target(path, 8, 8, true);
        t.activation_width = 4;
        t.activation_dtype = Dtype::F32;
        let p = profile_from_headers(CANDLE, &[f], &[t]).unwrap();
        assert_eq!(p.projections[0].installed_bytes, (4 + 16) * 4);
        assert_eq!(p.resident_bytes, 4 * 4 + 16 * 8);
    }

    #[test]
    fn forward_lifetimes_max_sequential_targets_and_only_mlx_sums_same_target_stack() {
        let projection = |path: &str, fixed, per_row| H3AdapterProjectionMemory {
            target: target(path, 8, 8, true),
            installed_bytes: 0,
            resident_bytes: 0,
            forward_fixed_bytes: fixed,
            forward_bytes_per_row: per_row,
        };
        let mut profile = H3AdapterMemoryProfile {
            resident_bytes: 0,
            install_peak_bytes: 0,
            projections: vec![
                projection("transformer_blocks.0.attn.to_q", 10, 2),
                projection("transformer_blocks.1.attn.to_q", 20, 3),
                projection("token_refiner.refiner_blocks.0.attn.to_q", 100, 1),
            ],
        };
        let r = rows(5, 2);
        // Sequential packed targets cost max(20,35), never 20+35. The
        // conditioning domain has its own rows and a separate lifetime.
        assert_eq!(
            profile.forward_working_bytes_for(MLX, r, RowDomain::Packed),
            35
        );
        assert_eq!(
            profile.forward_working_bytes_for(CANDLE, r, RowDomain::Packed),
            35
        );
        assert_eq!(profile.forward_working_bytes(MLX, r), 102);
        profile
            .projections
            .push(projection("transformer_blocks.1.attn.to_q", 7, 4));
        // Same-target MLX lazy residual nodes can coexist until evaluation;
        // Candle eagerly finishes one residual before starting the next.
        assert_eq!(
            profile.forward_working_bytes_for(MLX, r, RowDomain::Packed),
            62
        );
        assert_eq!(
            profile.forward_working_bytes_for(CANDLE, r, RowDomain::Packed),
            35
        );
    }
}
