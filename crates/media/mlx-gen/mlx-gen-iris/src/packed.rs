//! Pre-quantized Iris artifacts (epic 16774): the packed `q8` / `q4` tiers written ONCE, offline,
//! on the CPU, so a load reads packed tensors directly instead of reading the release's FP32
//! backbone (~12 GB) and the whole Qwen3-VL snapshot (vision tower included) and quantizing them at
//! load — a load floor no 8 GB phone can admit.
//!
//! **The layout** is the load-time quantization's, written to disk byte for byte:
//!
//! * **Backbone export** (`config.yaml` + `model.safetensors`, plus `empty_prompt.safetensors` for
//!   the task exports): every projection [`crate::nn::ProjectionLayout::of`] quantizes is stored as
//!   `{p}.weight` (u32 codes over the zero-padded input), `{p}.scales` and `{p}.biases` (bf16) — the
//!   exact triple [`crate::nn::pack_projection`] builds at load — with an i32 `{p}.in_features`
//!   beside a projection whose input was padded (the release's SwiGLU down projections). Every
//!   other matrix is stored at bf16 (the compute width the packed tiers run at, which the dense
//!   loader casts it to), vectors and the text position table as released.
//! * **Text encoder** (a Qwen3-VL snapshot with ONLY the language tower): every attention / MLP
//!   projection packed at the tower's group size on the stored bf16 grid
//!   ([`crate::text_encoder::text_encoder_quant_spec`]), the embedding and norms bf16, one
//!   `model.safetensors`, and a `text_config.quantization` block in `config.json` — the
//!   pre-quantized layout `mlx_llm` loads as stored.
//!
//! **The tier is read from the layout**: [`artifact_tier`] from the backbone's `.scales` /
//! codes shapes, [`text_encoder_tier`] from the tower's `quantization` block. A load whose
//! requested tier is not the artifact's is refused by the encoder-free loaders and publishes no
//! calibration identity on every route ([`crate::memory_strategy`]).
//!
//! The dense FP32 release and its quantize-at-load path are untouched: a dense artifact loads
//! exactly as before.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Quant, Result};
use mlx_rs::ops::pad;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::nn::{BackboneQuant, ProjectionLayout, QUANT_GROUP_SIZE};

/// Companion of a packed projection whose input was zero-padded to a group multiple: its logical
/// input width (i32 `[1]`).
pub const IN_FEATURES_SUFFIX: &str = ".in_features";

/// The text position table, kept as released by every loader.
const POSITION_TABLE: &str = "y_pos_embedding";

fn quant_of(bits: u64) -> Option<Quant> {
    match bits {
        4 => Some(Quant::Q4),
        8 => Some(Quant::Q8),
        _ => None,
    }
}

/// Bits of a [`Quant`] this family packs.
pub fn quant_bits(quant: Quant) -> Result<i32> {
    match quant {
        Quant::Q4 => Ok(4),
        Quant::Q8 => Ok(8),
        other => Err(Error::Unsupported(format!(
            "iris: the packed tiers are q8 and q4; {other:?} is not one of them"
        ))),
    }
}

/// The tier of the backbone weight file at `path`, from its tensor headers alone: `None` for a
/// dense file, else the one width every `{p}.scales` / `{p}.weight` pair packs at
/// (`bits = codes columns · 32 / (scales columns · group)`). Fails closed on a scales tensor
/// without codes, a ratio that is not an exact 4- or 8-bit pack, or two widths in one file.
pub fn artifact_tier(path: &Path) -> Result<Option<Quant>> {
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)
        .map_err(|e| Error::Msg(format!("iris: reading {}: {e}", path.display())))?;
    let shapes: HashMap<&str, &[usize]> = headers
        .iter()
        .map(|h| (h.name.as_str(), h.shape.as_slice()))
        .collect();
    let mut tier = None;
    for header in &headers {
        let Some(base) = header.name.strip_suffix(".scales") else {
            continue;
        };
        let fail = |detail: &str| {
            Error::Unsupported(format!(
                "iris: {} — packed projection `{base}` {detail}",
                path.display()
            ))
        };
        let codes = shapes
            .get(format!("{base}.weight").as_str())
            .ok_or_else(|| fail("has scales but no codes"))?;
        let (&[out, code_cols], &[scale_rows, scale_cols]) = (*codes, header.shape.as_slice())
        else {
            return Err(fail("is not a 2-D pack"));
        };
        let group = QUANT_GROUP_SIZE as u64;
        let width = (scale_cols as u64) * group;
        let packed = (code_cols as u64) * 32;
        if out != scale_rows || width == 0 || !packed.is_multiple_of(width) {
            return Err(fail("is not an exact affine pack"));
        }
        let quant = quant_of(packed / width).ok_or_else(|| fail("is not a 4- or 8-bit pack"))?;
        match tier {
            None => tier = Some(quant),
            Some(seen) if seen == quant => {}
            Some(_) => {
                return Err(fail(
                    "packs at a different width than an earlier projection",
                ))
            }
        }
    }
    Ok(tier)
}

/// The tier of the text-encoder snapshot at `dir`: its `text_config.quantization` bits (the
/// pre-quantized layout `mlx_llm` loads as stored), `None` for a dense snapshot.
pub fn text_encoder_tier(dir: &Path) -> Result<Option<Quant>> {
    let cfg = mlx_llm::ModelConfig::from_dir(dir).map_err(|e| {
        Error::Msg(format!(
            "iris: reading the text encoder config at {}: {e}",
            dir.display()
        ))
    })?;
    match cfg.quantization {
        None => Ok(None),
        Some(spec) => quant_of(spec.bits as u64).map(Some).ok_or_else(|| {
            Error::Unsupported(format!(
                "iris: the text encoder at {} is packed at {} bits; the tiers are q8 and q4",
                dir.display(),
                spec.bits
            ))
        }),
    }
}

/// Refuse a load whose requested tier is not the packed artifact's: a packed artifact loads only
/// at its own tier (`requested` `None` follows it). A dense artifact is the caller's to decide.
pub fn check_requested(
    what: &str,
    artifact: Option<Quant>,
    requested: Option<Quant>,
) -> Result<()> {
    match (artifact, requested) {
        (Some(stored), Some(asked)) if stored != asked => Err(Error::Unsupported(format!(
            "iris: {what} is packed at {stored:?}; a {asked:?} load would requantize it — load it at \
             its own tier"
        ))),
        _ => Ok(()),
    }
}

/// The encoder-free loaders' tier rule: a packed export loads at its own tier (a request for
/// another is refused), and a dense export has one tier — a `q8`/`q4` request on it is refused
/// rather than requantized at load.
pub fn refuse_dense_requested_as_packed(
    id: &str,
    weights: &Path,
    requested: Option<Quant>,
) -> Result<()> {
    let artifact = artifact_tier(weights)?;
    check_requested(id, artifact, requested)?;
    if artifact.is_none() {
        if let Some(asked) = requested {
            return Err(Error::Unsupported(format!(
                "{id}: LoadSpec::quantize {asked:?} names a packed tier, but {} is the dense \
                 export; load the packed {asked:?} export (`crate::packed`)",
                weights.display()
            )));
        }
    }
    Ok(())
}

/// Whether the backbone key `name` (the export prefix stripped) is cast to the compute dtype by
/// the dense loaders: every tensor of rank two or more but the text position table.
fn cast_at_load(name: &str, rank: usize) -> bool {
    rank >= 2 && name != POSITION_TABLE
}

/// Pack one dense `[out, in]` projection the way [`crate::nn::Loader::linear`] packs it at load:
/// bf16, zero-padded to a group multiple, codes on the stored bf16 grid. Returns the triple and,
/// for a padded input, its logical width.
fn pack_dense(weight: &Array, bits: i32) -> Result<(Array, Array, Array, Option<i32>)> {
    let in_features = *weight.shape().last().unwrap_or(&0);
    let ProjectionLayout::Quantized { padded_in, .. } =
        ProjectionLayout::of(in_features as usize, Some(BackboneQuant { bits }))
    else {
        return Err(Error::Msg("iris convert: not a packed projection".into()));
    };
    let in_pad = padded_in as i32 - in_features;
    let weight = weight.as_dtype(Dtype::Bfloat16)?;
    let weight = if in_pad > 0 {
        pad(&weight, &[(0, 0), (0, in_pad)][..], None, None)?
    } else {
        weight
    };
    let (codes, scales, biases) =
        mlx_llm::primitives::quant::quantize_on_stored_grid(&weight, QUANT_GROUP_SIZE as i32, bits)
            .map_err(|e| Error::Msg(format!("iris convert: {e}")))?;
    Ok((codes, scales, biases, (in_pad > 0).then_some(in_features)))
}

fn save(path: &Path, tensors: &HashMap<String, Array>) -> Result<()> {
    let mut entries: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    entries.sort_unstable_by_key(|(k, _)| *k);
    Array::save_safetensors(entries, None::<&HashMap<String, String>>, path)?;
    Ok(())
}

/// Write the packed tier of the backbone export at `src` (`config.yaml` + `model.safetensors`, and
/// for a task export `empty_prompt.safetensors`) to `dst`. `prefix` is the export's backbone key
/// prefix (`""` for the generation / restoration exports, `"pixel."` for depth); keys outside it
/// (the depth reducer) are copied as released. One tensor is read, converted and evaluated at a
/// time, so the conversion holds the packed output plus one source tensor.
pub fn prequantize_export(src: &Path, dst: &Path, bits: i32, prefix: &str) -> Result<()> {
    if !matches!(bits, 4 | 8) {
        return Err(Error::Unsupported(format!(
            "iris convert: q{bits} is not a packed tier (q8, q4)"
        )));
    }
    let weights_file = src.join(mlx_gen::gen_core::iris::BACKBONE_WEIGHTS_FILE);
    if artifact_tier(&weights_file)?.is_some() {
        return Err(Error::Unsupported(format!(
            "iris convert: {} is already packed",
            weights_file.display()
        )));
    }
    let mut source = Weights::from_file(&weights_file)?;
    let mut keys: Vec<String> = source.keys().map(str::to_owned).collect();
    keys.sort();
    let mut out: HashMap<String, Array> = HashMap::with_capacity(keys.len());
    for key in keys {
        let Some(array) = source.remove(&key) else {
            continue;
        };
        mlx_rs::transforms::eval_pending_loads([&array])?;
        let Some(name) = key.strip_prefix(prefix) else {
            eval([&array])?;
            out.insert(key, array);
            continue;
        };
        let packs = name.ends_with(".weight")
            && array.ndim() == 2
            && matches!(
                ProjectionLayout::of(
                    *array.shape().last().unwrap_or(&0) as usize,
                    Some(BackboneQuant { bits })
                ),
                ProjectionLayout::Quantized { .. }
            );
        if packs {
            let base = key.strip_suffix(".weight").expect("checked");
            let (codes, scales, biases, in_features) = pack_dense(&array, bits)?;
            eval([&codes, &scales, &biases])?;
            out.insert(format!("{base}.weight"), codes);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
            if let Some(width) = in_features {
                out.insert(
                    format!("{base}{IN_FEATURES_SUFFIX}"),
                    Array::from_slice(&[width], &[1]),
                );
            }
        } else if cast_at_load(name, array.ndim()) {
            let cast = array.as_dtype(Dtype::Bfloat16)?;
            eval([&cast])?;
            out.insert(key, cast);
        } else {
            eval([&array])?;
            out.insert(key, array);
        }
    }
    drop(source);
    std::fs::create_dir_all(dst)?;
    save(
        &dst.join(mlx_gen::gen_core::iris::BACKBONE_WEIGHTS_FILE),
        &out,
    )?;
    drop(out);
    for sidecar in ["config.yaml", "empty_prompt.safetensors"] {
        let from = src.join(sidecar);
        if from.is_file() {
            std::fs::copy(&from, dst.join(sidecar))?;
        }
    }
    Ok(())
}

/// Write the packed, text-only tier of the Qwen3-VL snapshot at `src` to `dst`: the language
/// tower's tensors only (the vision tower is dropped), projections packed at the tower's group
/// size on the stored bf16 grid, everything else bf16, in one `model.safetensors`; `config.json`
/// gains `text_config.quantization`; the tokenizer and other non-weight sidecars are copied.
pub fn prequantize_text_encoder(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    let quant = quant_of(bits as u64).ok_or_else(|| {
        Error::Unsupported(format!(
            "iris convert: q{bits} is not a packed tier (q8, q4)"
        ))
    })?;
    if text_encoder_tier(src)?.is_some() {
        return Err(Error::Unsupported(format!(
            "iris convert: the text encoder at {} is already packed",
            src.display()
        )));
    }
    let cfg = mlx_llm::ModelConfig::from_dir(src)
        .map_err(|e| Error::Msg(format!("iris convert: {e}")))?;
    let spec =
        crate::text_encoder::text_encoder_quant_spec(&cfg, Some(quant))?.expect("a requested tier");
    let untied = !cfg.tie_word_embeddings;
    let embedding = format!(
        "{}embed_tokens.weight",
        crate::text_encoder::LANGUAGE_TOWER_PREFIX
    );
    let mut out: HashMap<String, Array> = HashMap::new();
    for shard in crate::text_encoder::weight_shards(src, untied)? {
        let tensors = Array::load_safetensors(&shard)?;
        let mut names: Vec<String> = tensors
            .keys()
            .filter(|n| crate::text_encoder::is_text_tower_tensor(n, untied))
            .cloned()
            .collect();
        names.sort();
        let mut tensors = tensors;
        for name in names {
            let array = tensors.remove(&name).expect("listed");
            mlx_rs::transforms::eval_pending_loads([&array])?;
            let dense = array.as_dtype(Dtype::Bfloat16)?;
            let packs = dense.ndim() == 2
                && name.ends_with(".weight")
                && name != embedding
                && name.starts_with(crate::text_encoder::LANGUAGE_TOWER_PREFIX);
            if packs {
                let base = name.strip_suffix(".weight").expect("checked");
                let (codes, scales, biases) = mlx_llm::primitives::quant::quantize_on_stored_grid(
                    &dense,
                    spec.group_size,
                    spec.bits,
                )
                .map_err(|e| Error::Msg(format!("iris convert: {e}")))?;
                eval([&codes, &scales, &biases])?;
                out.insert(format!("{base}.weight"), codes);
                out.insert(format!("{base}.scales"), scales);
                out.insert(format!("{base}.biases"), biases);
            } else {
                eval([&dense])?;
                out.insert(name, dense);
            }
        }
    }
    std::fs::create_dir_all(dst)?;
    save(&dst.join("model.safetensors"), &out)?;
    drop(out);
    let mut config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(src.join("config.json"))
            .map_err(|e| Error::Msg(format!("iris convert: reading config.json: {e}")))?,
    )
    .map_err(|e| Error::Msg(format!("iris convert: parsing config.json: {e}")))?;
    let block = serde_json::json!({ "group_size": spec.group_size, "bits": spec.bits });
    match config.get_mut("text_config") {
        Some(text) => text["quantization"] = block,
        None => config["quantization"] = block,
    }
    std::fs::write(dst.join("config.json"), config.to_string())?;
    for entry in std::fs::read_dir(src)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let weights = name.ends_with(".safetensors") || name == "model.safetensors.index.json";
        if path.is_file()
            && !weights
            && name != "config.json"
            && !mlx_gen::gen_core::weightsmeta::is_hidden_file(&path)
        {
            std::fs::copy(&path, dst.join(name))?;
        }
    }
    Ok(())
}

/// What one conversion converts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvertKind {
    /// A backbone export (generation / restoration: no key prefix; depth: `pixel.`).
    Export { depth: bool },
    /// The Qwen3-VL text encoder (language tower only).
    TextEncoder,
}

/// Tensor-storage envelope of a CPU conversion, from headers before any payload is read: the
/// packed output (every projection at its packed size, the rest at bf16 or as released) plus the
/// largest source tensor and its bf16 and padded working copies. Not a process RSS bound — the
/// standalone command needs an external RSS guard too.
pub fn cpu_conversion_tensor_bytes(kind: ConvertKind, src: &Path, bits: i32) -> Result<u64> {
    let quant = Some(BackboneQuant { bits });
    let overflow = || Error::Msg("iris convert: byte count overflow".into());
    let (files, untied, group): (Vec<std::path::PathBuf>, bool, i32) = match kind {
        ConvertKind::Export { .. } => (
            vec![src.join(mlx_gen::gen_core::iris::BACKBONE_WEIGHTS_FILE)],
            false,
            QUANT_GROUP_SIZE as i32,
        ),
        ConvertKind::TextEncoder => {
            let cfg = mlx_llm::ModelConfig::from_dir(src)
                .map_err(|e| Error::Msg(format!("iris convert: {e}")))?;
            let quant = quant_of(bits as u64)
                .ok_or_else(|| Error::Unsupported(format!("iris convert: q{bits}")))?;
            let spec = crate::text_encoder::text_encoder_quant_spec(&cfg, Some(quant))?
                .expect("a requested tier");
            let untied = !cfg.tie_word_embeddings;
            (
                crate::text_encoder::weight_shards(src, untied)?,
                untied,
                spec.group_size,
            )
        }
    };
    let mut output = 0u64;
    let mut largest = 0u64;
    for file in &files {
        for h in mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(file)
            .map_err(|e| Error::Msg(e.to_string()))?
        {
            if matches!(kind, ConvertKind::TextEncoder)
                && !crate::text_encoder::is_text_tower_tensor(&h.name, untied)
            {
                continue;
            }
            let elements = h.shape.iter().map(|&d| d as u64).product::<u64>();
            let bytes =
                match h.shape.as_slice() {
                    [out, input] if h.name.ends_with(".weight") => match kind {
                        ConvertKind::TextEncoder => {
                            let padded = *input as u64;
                            (*out as u64)
                                .checked_mul(padded * bits as u64 / 8 + padded / group as u64 * 4)
                                .ok_or_else(overflow)?
                        }
                        ConvertKind::Export { .. } => ProjectionLayout::of(*input, quant)
                            .weight_bytes(*out as u64, *input as u64, 2),
                    },
                    shape if shape.len() >= 2 => elements * 2,
                    _ => h.data_bytes,
                };
            output = output.checked_add(bytes).ok_or_else(overflow)?;
            largest = largest.max(h.data_bytes);
        }
    }
    // The source tensor, its bf16 cast and its padded copy, all at most the source's size.
    output
        .checked_add(largest.checked_mul(3).ok_or_else(overflow)?)
        .ok_or_else(overflow)
}

/// Explicit CPU-only conversion: refuses before reading any payload when the header envelope
/// exceeds `tensor_budget`, then runs with BOTH the default device and stream on the CPU (the
/// previous device is restored, also on error). Serialize with other MLX callers.
pub fn prequantize_cpu(
    kind: ConvertKind,
    src: &Path,
    dst: &Path,
    bits: i32,
    tensor_budget: u64,
) -> Result<()> {
    if !matches!(bits, 4 | 8) {
        return Err(Error::Unsupported(format!(
            "iris convert: q{bits} is not a packed tier (q8, q4)"
        )));
    }
    let required = cpu_conversion_tensor_bytes(kind, src, bits)?;
    if required > tensor_budget {
        return Err(Error::Unsupported(format!(
            "iris convert: refused before load — tensor envelope {required} B exceeds the \
             {tensor_budget} B budget"
        )));
    }
    struct RestoreDevice(mlx_rs::Device);
    impl Drop for RestoreDevice {
        fn drop(&mut self) {
            mlx_rs::Device::set_default(&self.0);
        }
    }
    let _restore = RestoreDevice(mlx_rs::Device::try_default()?);
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || match kind {
        ConvertKind::Export { depth } => {
            prequantize_export(src, dst, bits, if depth { "pixel." } else { "" })
        }
        ConvertKind::TextEncoder => prequantize_text_encoder(src, dst, bits),
    })
}
