//! Header-only upper bounds for the actual MLX checkpoint load paths.
use crate::config::Architecture;
use crate::primitives::projection::QuantSpec;
use core_llm::{Error, LoadSpec, Result};
use serde_json::Value;
use std::{fs::File, io::Read, path::Path};

fn overflow() -> Error {
    Error::Load("MLX load memory estimate overflow".into())
}

/// Price source buffers, conversion outputs and staging without evaluating any MLX array.
///
/// Safetensors: the stored payload once plus every allocation the loader derives from it
/// ([`derived_bytes`]). A verified (architecture × conversion) charges exactly that; anything
/// else keeps the conservative two-copy bound, raised to the derived total where that is larger,
/// so an unverified load is never charged less than its derived allocations. Both add the
/// load's host heap ([`host_bytes`]) (sc-24446).
pub(crate) fn required_bytes(spec: &LoadSpec) -> Result<u64> {
    let path = Path::new(&spec.source);
    if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        let language = gguf_bytes(path, false)?;
        let projector = spec
            .projector_source
            .as_ref()
            .map(|p| gguf_bytes(Path::new(p), true))
            .transpose()?
            .unwrap_or(0);
        return language.checked_add(projector).ok_or_else(overflow);
    }
    let source = core_llm::checkpoint_payload_bytes(path)?;
    let config: Value = serde_json::from_reader(
        File::open(path.join("config.json")).map_err(|e| Error::Load(e.to_string()))?,
    )
    .map_err(|e| Error::Load(e.to_string()))?;
    let quant = spec.quantize.map(crate::provider::quant_spec).transpose()?;
    let headers = safetensors_headers(path)?;
    let derived = source
        .checked_add(derived_bytes(&headers, quant)?)
        .ok_or_else(overflow)?;
    let arrays = if verified_load(&config, &headers, quant.is_some()) {
        derived
    } else {
        derived.max(source.checked_mul(2).ok_or_else(overflow)?)
    };
    arrays.checked_add(host_bytes(path)?).ok_or_else(overflow)
}

/// Host heap per `tokenizer.json` byte: the parsed vocabulary, merges and added-token tables.
/// The sc-24446 probes measured 12–17× (Llama 3.2, Qwen3, Gemma 2); 24× keeps headroom.
const TOKENIZER_HEAP_PER_BYTE: u64 = 24;

/// Host heap a load adds independent of the tokenizer: the chat template, the lazy graph's node
/// allocations, and the Metal pipeline states its load-boundary checksum and first forward
/// compile (measured ≤ 30 MiB beside the tokenizer).
const LOAD_HOST_BYTES: u64 = 64 * 1024 * 1024;

/// Host (non-MLX) memory a safetensors load adds beside its arrays (sc-24446): the tokenizer
/// it parses and a fixed allowance for everything else the load builds on the heap. The Metal
/// device and MLX's kernel library are process-wide one-time costs, not a load's.
fn host_bytes(dir: &Path) -> Result<u64> {
    let tokenizer = match std::fs::metadata(dir.join("tokenizer.json")) {
        Ok(meta) => meta.len(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => return Err(Error::Load(e.to_string())),
    };
    tokenizer
        .checked_mul(TOKENIZER_HEAP_PER_BYTE)
        .and_then(|v| v.checked_add(LOAD_HOST_BYTES))
        .ok_or_else(overflow)
}

/// Whether this (architecture × conversion) has a derived bound backed by a reading of its
/// constructor and a guarded real-weight peak (`docs/reference/qwen38/native-memory-admission.md`
/// has the table). Dispatches exactly as the loader does ([`Architecture::from_config`]).
///
/// Unverified, by name: every MoE checkpoint (expert banks are stacked or split into new arrays —
/// priced by [`derived_bytes`], never measured here); a stored-quantized checkpoint outside the
/// Qwen3.5 family; load-time quantization of Qwen3-VL; Phi-3, Qwen2-MoE, GLM-4, DeepSeek-V2 and
/// plain Gemma 4.
fn verified_load(config: &Value, headers: &[Value], quantize_at_load: bool) -> bool {
    let names = || {
        headers
            .iter()
            .filter_map(Value::as_object)
            .flat_map(|h| h.keys())
    };
    if names().any(|name| expert_index(name).is_some() || name.contains(".mlp.experts.")) {
        return false;
    }
    let stored_quantized = names().any(|name| name.ends_with(".scales"));
    match Architecture::from_config(config) {
        Ok(Architecture::Qwen35) => true,
        Ok(Architecture::Qwen3Vl) => !quantize_at_load,
        Ok(
            Architecture::Qwen3
            | Architecture::Llama
            | Architecture::Gemma2
            | Architecture::Gemma4Unified,
        ) => !stored_quantized,
        _ => false,
    }
}

/// Every `*.safetensors` header in `dir`, read header-only.
pub(crate) fn safetensors_headers(dir: &Path) -> Result<Vec<Value>> {
    let mut headers = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| Error::Load(e.to_string()))? {
        let path = entry.map_err(|e| Error::Load(e.to_string()))?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
            continue;
        }
        let mut file = File::open(&path).map_err(|e| Error::Load(e.to_string()))?;
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| Error::Load(e.to_string()))?;
        let len = u64::from_le_bytes(prefix);
        if len > 64 * 1024 * 1024 {
            return Err(Error::Load(
                "safetensors admission header exceeds 64 MiB".into(),
            ));
        }
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header)
            .map_err(|e| Error::Load(e.to_string()))?;
        headers.push(serde_json::from_slice(&header).map_err(|e| Error::Load(e.to_string()))?);
    }
    Ok(headers)
}

/// The conversion intermediates every `*.safetensors` header in `dir` implies with no load-time
/// quantization ([`derived_bytes`]), read header-only.
fn safetensors_dir_extra(dir: &Path) -> Result<u64> {
    derived_bytes(&safetensors_headers(dir)?, None)
}

/// Resident bytes a companion MTP head adds to a load (epic sc-24432 E7, story sc-24444): its
/// safetensors payload (packed words, affine scales/biases and norm vectors stay resident in the
/// predictor) plus the conversion intermediates the header implies — the BF16 cast of any wider
/// float and the `1 + w` norm results. Header-only; never constructs a tensor. The head's
/// per-request attention cache is priced by request admission on the `mtp` route.
pub(crate) fn companion_head_bytes(dir: &Path) -> Result<u64> {
    core_llm::companion_head_payload_bytes(dir)?
        .checked_add(safetensors_dir_extra(dir)?)
        .ok_or_else(overflow)
}

/// MLX rounds every Metal buffer above one page up to a whole 16 KiB page; each allocation below
/// is charged this rounding once.
const ALLOCATION_ROUNDING: u64 = 16 * 1024;

/// Bytes the MLX safetensors loaders allocate **beyond** the stored payload, from the headers
/// alone (sc-24446). Every term is additive: a derived array either stays resident in the model
/// or, once consumed, returns its buffer to MLX's freed-buffer cache, which is reused only for an
/// allocation of (page-rounded) equal size and released only by `clear_cache` (the first decode
/// step) or allocator pressure — so a consumed transient still occupies memory through the load
/// and the first forward. The payload itself is read into owned Metal buffers (MLX `Load` is a
/// `pread` into `allocator::malloc`, not a mapping) and every accessed tensor is resident after
/// the constructors' `verify_accessed_gpu_view`, before any derived array is evaluated.
fn derived_bytes(headers: &[Value], quant: Option<QuantSpec>) -> Result<u64> {
    headers.iter().try_fold(0u64, |total, header| {
        total
            .checked_add(header_derived_bytes(header, quant)?)
            .ok_or_else(overflow)
    })
}

/// `n·bits/8` packed `uint32` words plus BF16 scales and biases per `group_size` elements: the
/// three outputs of MLX `quantize` on a BF16 input.
fn quantized_bytes(elements: u64, quant: QuantSpec) -> Result<u64> {
    let bits = u64::try_from(quant.bits).map_err(|_| overflow())?;
    let group = u64::try_from(quant.group_size).map_err(|_| overflow())?;
    let packed = elements
        .checked_mul(bits)
        .map(|v| v / 8)
        .ok_or_else(overflow)?;
    let affine = (elements / group).checked_mul(4).ok_or_else(overflow)?;
    packed.checked_add(affine).ok_or_else(overflow)
}

/// The expert index of a per-expert tensor (`…experts.{e}.…`), if this is one.
fn expert_index(name: &str) -> Option<u64> {
    let (_, rest) = name.split_once("experts.")?;
    rest.split('.').next()?.parse().ok()
}

fn header_derived_bytes(header: &Value, quant: Option<QuantSpec>) -> Result<u64> {
    let mut extra = 0u64;
    let mut add = |bytes: u64| -> Result<()> {
        extra = extra
            .checked_add(bytes)
            .and_then(|v| v.checked_add(ALLOCATION_ROUNDING))
            .ok_or_else(overflow)?;
        Ok(())
    };
    for (name, info) in header
        .as_object()
        .ok_or_else(|| Error::Load("safetensors header is not an object".into()))?
    {
        if name == "__metadata__" {
            continue;
        }
        let shape = info["shape"]
            .as_array()
            .ok_or_else(|| Error::Load("safetensors tensor lacks shape".into()))?
            .iter()
            .map(|v| v.as_u64().ok_or_else(overflow))
            .collect::<Result<Vec<u64>>>()?;
        let elements = shape
            .iter()
            .try_fold(1u64, |n, &v| n.checked_mul(v).ok_or_else(overflow))?;
        let dtype = info["dtype"]
            .as_str()
            .ok_or_else(|| Error::Load("safetensors tensor lacks dtype".into()))?;
        let width = match dtype {
            "F64" | "I64" | "U64" => 8,
            "F32" | "I32" | "U32" => 4,
            "F16" | "BF16" | "I16" | "U16" => 2,
            _ => 1,
        };
        let stored = elements.checked_mul(width).ok_or_else(overflow)?;
        let bf16 = elements.checked_mul(2).ok_or_else(overflow)?;
        let float = matches!(dtype, "F16" | "BF16" | "F32" | "F64");
        let vision = name.starts_with("model.visual.") || name.starts_with("vision_tower.");
        // The stored buffer itself (payload) is page-rounded too.
        add(0)?;
        // The load-boundary checksum widens a byte view to u32 when the size is not word-aligned.
        if stored % 4 != 0 {
            add(stored.checked_mul(4).ok_or_else(overflow)?)?;
        }
        // Language tensors are cast to BF16. MLX astype returns the original Array for an
        // unchanged dtype; vision clones retain their original dtype and U32 packed words remain
        // compact.
        if !vision && float && dtype != "BF16" {
            add(bf16)?;
        }
        // Vector norms (`1 + w`), exp(A_log) and sign transforms can retain both source and
        // result. Four bytes per element covers two additional BF16 intermediates.
        if !vision && shape.len() <= 1 {
            add(elements.checked_mul(4).ok_or_else(overflow)?)?;
        }
        // A channels-last patch kernel may require a contiguous transpose before reshape.
        if vision && name.ends_with("patch_embed.proj.weight") {
            add(elements.checked_mul(4).ok_or_else(overflow)?)?;
        }
        // Gathered splits are copies: Qwen3.6's fused expert `gate_up_proj` is split into its
        // gate and up halves (`take_axis`), Gemma 4's `[n, 2, hidden]` position table into its
        // row and column tables.
        if name.ends_with(".mlp.experts.gate_up_proj")
            || (name.ends_with(".pos_embedding") && shape.len() == 3)
        {
            add(bf16)?;
        }
        // Load-time affine quantization: MLX `quantize` writes new packed words, scales and
        // biases (one GPU kernel, no float intermediates) for every projection the decoders
        // quantize. Embeddings and the LM head stay dense; every other floating matrix whose
        // input width divides the group is charged (a dense router or 1-row gate is
        // over-charged, never missed).
        let quantized = match quant {
            Some(q)
                if !vision
                    && float
                    && shape.len() >= 2
                    && !name.contains("embed_tokens")
                    && !name.ends_with("lm_head.weight")
                    && !name.ends_with(".scales")
                    && !name.ends_with(".biases")
                    && u64::try_from(q.group_size)
                        .ok()
                        .filter(|&g| g > 0)
                        .is_some_and(|g| shape.last().is_some_and(|last| last % g == 0)) =>
            {
                Some(quantized_bytes(elements, q)?)
            }
            _ => None,
        };
        if let Some(bytes) = quantized {
            add(bytes)?;
        }
        // Per-expert tensors are stacked into one `[experts, …]` array per projection
        // (`SwitchLinear::stack`, sc-24440): a copy of each expert's loaded form — quantized,
        // BF16, or the stored packed part.
        if expert_index(name).is_some() {
            add(match quantized {
                Some(bytes) => bytes,
                None if float && !name.ends_with(".scales") && !name.ends_with(".biases") => bf16,
                None => stored,
            })?;
        }
    }
    Ok(extra)
}

fn gguf_bytes(path: &Path, projector: bool) -> Result<u64> {
    let file = crate::gguf::GgufFile::open(path).map_err(|e| Error::Load(e.to_string()))?;
    let mut retained = 0u64;
    let mut staging = 0u64;
    for tensor in &file.tensors {
        let elements = tensor
            .shape
            .iter()
            .try_fold(1u64, |n, &v| n.checked_mul(v as u64).ok_or_else(overflow))?;
        let (keep, temporary) = gguf_tensor_bytes(
            elements,
            tensor.shape.last().copied().unwrap_or(0) as u64,
            tensor.ggml_type,
            projector,
        )?;
        retained = retained.checked_add(keep).ok_or_else(overflow)?;
        staging = staging.max(temporary);
    }
    // GGUF is memory mapped. Include every source byte even though pages are reclaimable, all
    // retained outputs, and the largest tensor's simultaneous conversion/reorder host buffers.
    core_llm::checkpoint_payload_bytes(path)?
        .checked_add(retained)
        .and_then(|v| v.checked_add(staging))
        .ok_or_else(overflow)
}

fn gguf_tensor_bytes(n: u64, width: u64, kind: u32, projector: bool) -> Result<(u64, u64)> {
    if !projector && matches!(kind, 142 | 143) {
        // U32 affine words: n/4 bytes. Both F32 and F16 scale+bias arrays may coexist until
        // lazy conversion completes: n/16+n/32. Explicit signs use four bytes per column.
        let signs = width.checked_mul(4).ok_or_else(overflow)?;
        let kept = n
            .checked_mul(11)
            .and_then(|v| v.checked_div(32))
            .and_then(|v| v.checked_add(signs))
            .ok_or_else(overflow)?;
        let temporary = n
            .checked_mul(5)
            .and_then(|v| v.checked_div(16))
            .and_then(|v| v.checked_add(signs))
            .ok_or_else(overflow)?;
        Ok((kept, temporary))
    } else {
        // Dense language: F32 source + BF16 cast + norm result. Projector remains F32.
        // Reordering dense language rows also holds an old and new host Vec<f32>.
        let multiplier = if projector { 4 } else { 8 };
        let bytes = n.checked_mul(multiplier).ok_or_else(overflow)?;
        Ok((bytes, bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// MLX's Metal page (`vm_page_size` on Apple silicon): every buffer above it is rounded up.
    const R: u64 = 16 * 1024;
    /// The fixed host allowance of a load with no `tokenizer.json`.
    const H: u64 = 64 * 1024 * 1024;

    /// A load's host heap: 24 bytes per `tokenizer.json` byte plus the fixed allowance.
    #[test]
    fn a_load_is_charged_its_tokenizer_heap_and_a_fixed_host_allowance() {
        let (dir, payload) = snapshot(
            json!({"model_type": "qwen3"}),
            &[("model.norm.weight", "BF16", &[64])],
        );
        let bare = required(&dir, None);
        assert_eq!(bare, payload + 2 * R + 64 * 4 + H);
        std::fs::write(dir.path().join("tokenizer.json"), vec![b' '; 1000]).unwrap();
        assert_eq!(required(&dir, None), bare + 24 * 1000);
    }

    /// Every stored tensor and every derived array is charged one whole 16 KiB page of rounding.
    #[test]
    fn each_buffer_is_charged_a_metal_page_of_rounding() {
        let one = json!({"m.weight": {"dtype": "BF16", "shape": [64, 64]}});
        assert_eq!(derived_bytes(&[one], None).unwrap(), R);
    }

    /// A one-file snapshot with zeroed payloads for `tensors` (name, dtype, shape) under
    /// `config`; returns the directory and its payload bytes.
    fn snapshot(config: Value, tensors: &[(&str, &str, &[u64])]) -> (tempfile::TempDir, u64) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for (name, dtype, shape) in tensors {
            let width = match *dtype {
                "F32" | "U32" => 4,
                "F64" => 8,
                _ => 2,
            };
            let end = offset + width * shape.iter().product::<u64>();
            header.insert(
                name.to_string(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}),
            );
            offset = end;
        }
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(&header);
        bytes.resize(bytes.len() + offset as usize, 0);
        std::fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();
        (dir, bytes.len() as u64)
    }

    fn required(dir: &tempfile::TempDir, quantize: Option<core_llm::Quantize>) -> u64 {
        let mut spec = LoadSpec::dense(dir.path().display().to_string());
        spec.quantize = quantize;
        required_bytes(&spec).unwrap()
    }

    type Tensors = Vec<(String, &'static str, Vec<u64>)>;

    /// A dense decoder slice: embedding, one projection, one norm, untied head.
    fn dense(prefix: &str) -> Tensors {
        vec![
            (
                format!("{prefix}.embed_tokens.weight"),
                "BF16",
                vec![16, 64],
            ),
            (
                format!("{prefix}.layers.0.mlp.up_proj.weight"),
                "BF16",
                vec![1024, 1024],
            ),
            (format!("{prefix}.norm.weight"), "BF16", vec![64]),
            ("lm_head.weight".into(), "BF16", vec![16, 64]),
        ]
    }

    fn refs(t: &Tensors) -> Vec<(&str, &str, &[u64])> {
        t.iter()
            .map(|(n, d, s)| (n.as_str(), *d, s.as_slice()))
            .collect()
    }

    /// Every verified architecture, unquantized: the payload once plus each buffer's page
    /// rounding and the norm's two BF16 intermediates — no second copy of any BF16 matrix.
    #[test]
    fn verified_unquantized_loads_charge_the_payload_once_plus_derived_arrays() {
        for (config, prefix) in [
            (json!({"model_type": "qwen3_5"}), "model.language_model"),
            (json!({"model_type": "qwen3_5_text"}), "model"),
            (json!({"model_type": "qwen3_vl"}), "model.language_model"),
            (json!({"model_type": "prism_hadamard_qwen35"}), "model"),
            (json!({"model_type": "qwen3"}), "model"),
            (json!({"model_type": "llama"}), "model"),
            (json!({"model_type": "mistral"}), "model"),
            (json!({"model_type": "gemma2"}), "model"),
            (
                json!({"model_type": "gemma4_unified"}),
                "model.language_model",
            ),
        ] {
            let tensors = dense(prefix);
            let (dir, payload) = snapshot(config.clone(), &refs(&tensors));
            assert_eq!(
                required(&dir, None),
                payload + 4 * R + (64 * 4 + R) + H,
                "{config}"
            );
        }
    }

    /// Load-time Q4/Q8 adds the packed words (`n·bits/8`) and BF16 scales + biases
    /// (`4·n/group`) of every quantized projection — never of the embedding or the LM head —
    /// on top of the BF16 payload that stays resident until the quantized arrays are evaluated.
    #[test]
    fn quantize_at_load_charges_packed_words_scales_and_biases_per_bits() {
        let up = 1024 * 1024;
        for (quantize, quantized) in [
            (core_llm::Quantize::Q4, up / 2 + up / 64 * 4),
            (core_llm::Quantize::Q8, up + up / 64 * 4),
        ] {
            for model_type in ["qwen3_5", "qwen3", "llama", "gemma2", "gemma4_unified"] {
                let tensors = dense("model");
                let (dir, payload) = snapshot(json!({ "model_type": model_type }), &refs(&tensors));
                assert_eq!(
                    required(&dir, Some(quantize)),
                    payload + 4 * R + (64 * 4 + R) + (quantized + R) + H,
                    "{model_type} {quantize:?}"
                );
            }
        }
        // A matrix whose input width does not divide the group is never quantized.
        let (dir, payload) = snapshot(
            json!({"model_type": "qwen3"}),
            &[("model.layers.0.mlp.up_proj.weight", "BF16", &[128, 48])],
        );
        assert_eq!(
            required(&dir, Some(core_llm::Quantize::Q4)),
            payload + R + H
        );
        // NVFP4 is refused by name, never priced as another format.
        let mut spec = LoadSpec::dense(dir.path().display().to_string());
        spec.quantize = Some(core_llm::Quantize::Nvfp4);
        assert!(required_bytes(&spec).is_err());
    }

    /// Unverified (architecture × conversion) keeps the two-copy bound: MoE checkpoints,
    /// stored-quantized snapshots outside Qwen3.5, Qwen3-VL load-time quantization, and the
    /// architectures with no measured peak.
    #[test]
    fn unverified_loads_keep_the_two_copy_bound() {
        let cases: [(Value, Tensors, Option<core_llm::Quantize>); 9] = [
            (json!({"model_type": "phi3"}), dense("model"), None),
            (json!({"model_type": "glm4"}), dense("model"), None),
            (json!({"model_type": "deepseek_v2"}), dense("model"), None),
            (json!({"model_type": "gemma4"}), dense("model"), None),
            (json!({"model_type": "other"}), dense("model"), None),
            (
                json!({"model_type": "qwen3_vl"}),
                dense("model.language_model"),
                Some(core_llm::Quantize::Q4),
            ),
            (
                json!({"model_type": "qwen3_5_moe"}),
                vec![(
                    "model.language_model.layers.0.mlp.experts.down_proj".into(),
                    "BF16",
                    vec![2, 1024, 1024],
                )],
                None,
            ),
            (
                json!({"model_type": "qwen2_moe"}),
                vec![(
                    "model.layers.0.mlp.experts.0.down_proj.weight".into(),
                    "BF16",
                    vec![1024, 1024],
                )],
                None,
            ),
            (
                json!({"model_type": "llama"}),
                vec![
                    (
                        "model.layers.0.mlp.up_proj.weight".into(),
                        "U32",
                        vec![1024, 128],
                    ),
                    (
                        "model.layers.0.mlp.up_proj.scales".into(),
                        "BF16",
                        vec![1024, 16],
                    ),
                    (
                        "model.layers.0.mlp.up_proj.biases".into(),
                        "BF16",
                        vec![1024, 16],
                    ),
                ],
                None,
            ),
        ];
        for (config, tensors, quantize) in cases {
            let (dir, payload) = snapshot(config.clone(), &refs(&tensors));
            let derived = payload
                + derived_bytes(
                    &safetensors_headers(dir.path()).unwrap(),
                    quantize.map(|q| crate::provider::quant_spec(q).unwrap()),
                )
                .unwrap();
            let charged = required(&dir, quantize) - H;
            assert!(charged >= 2 * payload, "{config}: {charged} < two copies");
            assert_eq!(charged, derived.max(2 * payload), "{config}");
        }
    }

    /// An unverified load is charged its derived allocations when they exceed two copies: a
    /// fused Qwen3.6 expert bank quantized to Q8 holds its BF16 payload, the gate/up split
    /// copies and the quantized outputs at once.
    #[test]
    fn a_derived_bound_above_two_copies_is_charged_in_full() {
        let (dir, payload) = snapshot(
            json!({"model_type": "qwen3_5_moe"}),
            &[(
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                "BF16",
                &[4, 128, 64],
            )],
        );
        let n = 4 * 128 * 64;
        let derived = payload + R + (2 * n + R) + (n + n / 64 * 4 + R);
        assert!(derived > 2 * payload);
        assert_eq!(required(&dir, Some(core_llm::Quantize::Q8)), derived + H);
    }

    /// Per-expert banks are stacked into one array per projection (`SwitchLinear::stack`): a
    /// copy of each expert's loaded form — BF16 when dense, the quantized triple (on top of the
    /// per-expert triple it was stacked from) at load-time Q4.
    #[test]
    fn per_expert_stacking_is_a_copy_of_the_loaded_form() {
        let tensors: [(&str, &str, &[u64]); 1] = [(
            "model.layers.0.mlp.experts.3.up_proj.weight",
            "BF16",
            &[128, 64],
        )];
        let headers = [json!({
            tensors[0].0: {"dtype": "BF16", "shape": [128, 64], "data_offsets": [0, 16384]}
        })];
        let n = 128 * 64;
        assert_eq!(derived_bytes(&headers, None).unwrap(), R + (2 * n + R));
        let q = n / 2 + n / 64 * 4;
        assert_eq!(
            derived_bytes(&headers, Some(QuantSpec::q4())).unwrap(),
            R + 2 * (q + R)
        );
        assert_eq!(expert_index(tensors[0].0), Some(3));
        assert_eq!(
            expert_index("model.layers.0.mlp.shared_experts.up_proj.weight"),
            None
        );
        assert_eq!(
            expert_index("model.layers.0.mlp.experts.gate_up_proj"),
            None
        );
    }

    /// Casts, vector intermediates, word-unaligned checksum views, split copies and vision
    /// transposes are each charged; same-dtype BF16 matrices are not.
    #[test]
    fn conversions_are_charged_and_shared_weights_are_not() {
        let h = json!({
            "model.language_model.layers.0.weight": {"dtype": "BF16", "shape": [1024, 1024]},
            "model.language_model.norm.weight": {"dtype": "BF16", "shape": [1024]},
            "mtp.weight": {"dtype": "F32", "shape": [16, 16]},
            "model.layers.0.layer_scalar": {"dtype": "BF16", "shape": [1]},
            "model.visual.proj.weight": {"dtype": "F16", "shape": [32, 32]},
            "model.visual.patch_embed.proj.weight": {"dtype": "BF16", "shape": [8, 2, 2]},
            "model.vision_embedder.pos_embedding": {"dtype": "BF16", "shape": [4, 2, 8]},
        });
        let expected = 7 * R
            + (1024 * 4 + R)
            + (16 * 16 * 2 + R)
            + (2 * 4 + R)
            + (4 + R)
            + (32 * 4 + R)
            + (64 * 2 + R);
        assert_eq!(derived_bytes(&[h], None).unwrap(), expected);
        assert!(derived_bytes(&[json!({"x":{"dtype":"F32","shape":[u64::MAX,2]}})], None).is_err());
        assert!(quantized_bytes(u64::MAX, QuantSpec::q8()).is_err());
    }

    /// Guarded real-weight probes (sc-24446), 2026-10-01, Apple M5 Max (Mac17,6), 128 GiB unified
    /// memory, macOS 26: one load through `LlamaProvider::load` plus one-token requests
    /// (`tests/load_admission_probe.rs`, `phys_footprint` sampled in-process every 100 ms, over a
    /// baseline taken after Metal and MLX's kernel library initialized). `measured_peak` is the
    /// load's share of the peak: for an unquantized load, the footprint growth through the end of
    /// the load (nothing is derived later); for load-time quantization — whose arrays are only
    /// evaluated by the first forward — the first request's peak growth minus the request's own
    /// MLX working set (a second identical request on the materialized model), which request
    /// admission prices. Each estimate is `required_bytes` for the same pinned snapshot; the
    /// `pinned_real_weight_estimates_match_the_recorded_table` audit recomputes it from the
    /// headers so the table cannot drift from the code. These are recorded evidence, not a
    /// machine golden: the test asserts only the relation.
    pub(super) const MEASURED: &[Measured] = &[
        Measured {
            config: "Gemma 2 2B-it BF16",
            snapshot: "models--SceneWorks--gemma-2-2b-it/snapshots/684c553b5b41a1c835989d89f62f585e6269a7de",
            quantize: None,
            estimate: 5_723_841_512,
            measured_peak: 5_518_511_596,
        },
        Measured {
            config: "Gemma 2 2B-it load-time Q4",
            snapshot: "models--SceneWorks--gemma-2-2b-it/snapshots/684c553b5b41a1c835989d89f62f585e6269a7de",
            quantize: Some(core_llm::Quantize::Q4),
            estimate: 6_865_478_632,
            measured_peak: 6_586_586_804,
        },
        Measured {
            config: "Llama 3.2 1B Instruct BF16",
            snapshot: "models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/863c846a9ac6fad4e49e1743d52984dff262e953",
            quantize: None,
            estimate: 2_760_013_225,
            measured_peak: 2_622_474_088,
        },
        Measured {
            config: "Llama 3.2 1B Instruct load-time Q4",
            snapshot: "models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/863c846a9ac6fad4e49e1743d52984dff262e953",
            quantize: Some(core_llm::Quantize::Q4),
            estimate: 3_309_204_905,
            measured_peak: 3_023_299_764,
        },
        Measured {
            config: "Llama 3.2 1B Instruct load-time Q8",
            snapshot: "models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/863c846a9ac6fad4e49e1743d52984dff262e953",
            quantize: Some(core_llm::Quantize::Q8),
            estimate: 3_795_744_169,
            measured_peak: 3_685_410_288,
        },
        Measured {
            config: "Qwen3-1.7B BF16",
            snapshot: "models--mlx-community--Qwen3-1.7B-bf16/snapshots/9cd6692855d3e06772228e9a962b2606359b2d24",
            quantize: None,
            estimate: 3_789_864_045,
            measured_peak: 3_573_647_904,
        },
        Measured {
            config: "Qwen3-1.7B load-time Q4",
            snapshot: "models--mlx-community--Qwen3-1.7B-bf16/snapshots/9cd6692855d3e06772228e9a962b2606359b2d24",
            quantize: Some(core_llm::Quantize::Q4),
            estimate: 4_585_798_765,
            measured_peak: 4_418_239_520,
        },
        Measured {
            config: "Qwen3-8B load-time Q4",
            snapshot: "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
            quantize: Some(core_llm::Quantize::Q4),
            estimate: 20_644_038_072,
            measured_peak: 19_874_702_744,
        },
        Measured {
            config: "Qwen3-8B load-time Q8",
            snapshot: "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
            quantize: Some(core_llm::Quantize::Q8),
            estimate: 24_116_921_784,
            measured_peak: 23_905_933_824,
        },
        Measured {
            config: "Qwen3-8B BF16",
            snapshot: "models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
            quantize: None,
            estimate: 16_732_915_128,
            measured_peak: 16_513_117_776,
        },
        Measured {
            config: "Gemma 4 unified (LTX-2.5 enhancer) BF16",
            snapshot: "models--SceneWorks--ltx-2.5-mlx/snapshots/791ef61731ad067bd13ebff8cc0f07532476d9ef/enhancer",
            quantize: None,
            estimate: 24_796_675_216,
            measured_peak: 24_304_141_352,
        },
        Measured {
            config: "Gemma 4 unified (LTX-2.5 enhancer) load-time Q4",
            snapshot: "models--SceneWorks--ltx-2.5-mlx/snapshots/791ef61731ad067bd13ebff8cc0f07532476d9ef/enhancer",
            quantize: Some(core_llm::Quantize::Q4),
            estimate: 30_962_780_304,
            measured_peak: 29_924_992_928,
        },
        Measured {
            config: "Qwen3.8-27B load-time Q4",
            snapshot: "models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            quantize: Some(core_llm::Quantize::Q4),
            estimate: 69_916_536_056,
            measured_peak: 69_476_846_942,
        },
        Measured {
            config: "Qwen3.8-27B load-time Q8",
            snapshot: "models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            quantize: Some(core_llm::Quantize::Q8),
            estimate: 82_304_150_776,
            measured_peak: 80_639_984_014,
        },
    ];

    /// One recorded probe.
    pub(super) struct Measured {
        /// What was loaded: snapshot (HF repo @ revision) and conversion.
        pub(super) config: &'static str,
        /// Snapshot directory relative to `LOAD_PROBE_SNAPSHOT_ROOT` (the local hub directory).
        pub(super) snapshot: &'static str,
        pub(super) quantize: Option<core_llm::Quantize>,
        /// `required_bytes` for this snapshot and conversion.
        pub(super) estimate: u64,
        /// Peak `phys_footprint` growth over the pre-load footprint, sampled every 100 ms
        /// in-process, across load + a one-token request.
        pub(super) measured_peak: u64,
    }

    /// Margin the estimate must keep above every recorded peak, in tenths of a percent (0.5%).
    /// The tightest recorded margin is Qwen3.8-27B at load-time Q4 (0.63%).
    const MARGIN_PERMILLE: u64 = 5;

    #[test]
    fn every_recorded_probe_peak_is_covered_by_its_estimate_with_margin() {
        for m in MEASURED {
            assert!(
                m.estimate >= m.measured_peak + m.measured_peak * MARGIN_PERMILLE / 1000,
                "{}: estimate {} does not cover measured peak {} with {MARGIN_PERMILLE}‰ margin",
                m.config,
                m.estimate,
                m.measured_peak
            );
        }
    }

    /// Header-only: recompute each recorded estimate from the local pinned snapshot.
    #[test]
    #[ignore = "requires the pinned snapshots under LOAD_PROBE_SNAPSHOT_ROOT"]
    fn pinned_real_weight_estimates_match_the_recorded_table() {
        let hub = std::path::PathBuf::from(
            std::env::var("LOAD_PROBE_SNAPSHOT_ROOT").expect("set LOAD_PROBE_SNAPSHOT_ROOT"),
        );
        for m in MEASURED {
            let mut spec = LoadSpec::dense(hub.join(m.snapshot).display().to_string());
            spec.quantize = m.quantize;
            assert_eq!(required_bytes(&spec).unwrap(), m.estimate, "{}", m.config);
        }
    }

    /// Header-only operational audit: never constructs a tensor or starts a model.
    #[test]
    #[ignore = "requires MLX_LOAD_ADMISSION_CASES containing pinned local paths and expected bounds"]
    fn pinned_header_only_load_admission_bounds() {
        let path = std::env::var("MLX_LOAD_ADMISSION_CASES").expect("set MLX_LOAD_ADMISSION_CASES");
        let cases: Value = serde_json::from_reader(File::open(path).unwrap()).unwrap();
        let mut mismatches = Vec::new();
        for case in cases.as_array().unwrap() {
            let mut spec = LoadSpec::dense(case["source"].as_str().unwrap());
            if let Some(projector) = case["projector_source"].as_str() {
                spec = spec.with_projector(projector);
            }
            spec.quantize = match case["quantize"].as_str() {
                None => None,
                Some("q4") => Some(core_llm::Quantize::Q4),
                Some("q8") => Some(core_llm::Quantize::Q8),
                Some(other) => panic!("unknown quantize {other}"),
            };
            let actual = required_bytes(&spec).unwrap();
            println!(
                "{}",
                json!({"source":spec.source,"projector_source":spec.projector_source,
                    "quantize":case["quantize"],"required_bytes":actual})
            );
            if case["expected_bytes"].as_u64() != Some(actual) {
                mismatches.push(spec.source);
            }
        }
        assert!(mismatches.is_empty(), "{mismatches:?}");
    }

    #[test]
    fn gguf_counts_source_conversion_outputs_and_largest_staging() {
        assert_eq!(
            gguf_tensor_bytes(1024, 128, 142, false).unwrap(),
            (864, 832)
        );
        assert_eq!(
            gguf_tensor_bytes(1024, 128, 143, false).unwrap(),
            (864, 832)
        );
        assert_eq!(
            gguf_tensor_bytes(1024, 128, 1, false).unwrap(),
            (8192, 8192)
        );
        assert_eq!(gguf_tensor_bytes(1024, 128, 8, true).unwrap(), (4096, 4096));
        assert!(gguf_tensor_bytes(u64::MAX, 128, 143, false).is_err());
    }
}
