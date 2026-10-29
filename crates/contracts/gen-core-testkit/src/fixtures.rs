//! Sparse weight fixtures and encoder-contract fixtures shared by provider conformance tests.

use super::*;

/// Mark `path` as an NTFS sparse file so a later `set_len` reserves no clusters for the hole.
///
/// Fixture weights are sized like the real checkpoints they stand in for — an encoder fixture is
/// several GB, a full video snapshot hundreds — because the gates under test *stat* the file rather
/// than read it. On APFS and ext4 that costs nothing: extending past the end of a file leaves a
/// hole. NTFS is the exception. It allocates every cluster on `SetEndOfFile` unless the file already
/// carries `FILE_ATTRIBUTE_SPARSE_FILE`, so fixtures that were free on the Mac and Linux lanes wrote
/// their full nominal size on Windows — enough to fill the CUDA box's system drive twice and fail a
/// `candle-worker` run with `StorageFull` (os error 112).
///
/// **Ordering is load-bearing.** The attribute lives on a file that exists, and `File::create` is
/// `CREATE_ALWAYS`, which *clears* it — so flagging before the create silently does nothing. Call
/// this between creating the file and the `set_len` that extends it; a header written on either side
/// of the call is fine, since writes to a sparse file allocate only the range they touch. Extending
/// an already-flagged file through a later `OpenOptions::open` keeps the attribute, so a consumer
/// that reopens a fixture to append a tensor inherits sparseness for free.
///
/// Best effort by design. A fixture that lands dense still holds exactly the right bytes, so a
/// failure here is a disk-space regression, never a test failure — it warns and returns. The
/// workspace forbids `unsafe`, so the flag goes on through `fsutil` rather than a raw
/// `FSCTL_SET_SPARSE`; off Windows the whole thing is a no-op.
pub fn mark_sparse(path: &std::path::Path) {
    #[cfg(windows)]
    {
        let outcome = std::process::Command::new("fsutil")
            .args(["sparse", "setflag"])
            .arg(path)
            .stdin(std::process::Stdio::null())
            .output();
        match outcome {
            Ok(output) if output.status.success() => {}
            Ok(output) => {
                // fsutil reports its own failures on stdout; keep stderr as the fallback.
                let mut detail = String::from_utf8_lossy(&output.stdout).trim().to_owned();
                if detail.is_empty() {
                    detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
                }
                eprintln!(
                    "gen-core-testkit: `fsutil sparse setflag {}` failed ({}: {detail}); this \
                     fixture will allocate its full length",
                    path.display(),
                    output.status,
                );
            }
            Err(error) => eprintln!(
                "gen-core-testkit: could not run `fsutil sparse setflag {}` ({error}); this fixture \
                 will allocate its full length",
                path.display(),
            ),
        }
    }
    #[cfg(not(windows))]
    {
        let _ = path;
    }
}

/// Copy a fixture whose tensor payload is a hole, without materializing that hole.
///
/// [`std::fs::copy`] reads and writes every byte, so copying a multi-GB fixture allocates the whole
/// payload in the destination even when the source is a hole — the copy is how a fixture that costs
/// nothing to *create* still fills a disk. This reproduces the source's header and logical length
/// onto a freshly-flagged destination instead.
///
/// The result is byte-for-byte what `std::fs::copy` would have produced **for a fixture**, whose
/// payload is never written and therefore reads back as zeros. Do not reach for this to copy a real
/// checkpoint: any non-zero payload byte in `source` would be silently dropped.
pub fn copy_sparse_fixture(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::{Read as _, Write as _};

    let mut input = std::fs::File::open(source)?;
    let total = input.metadata()?.len();
    let mut declared = [0_u8; 8];
    input.read_exact(&mut declared)?;
    let header_len = u64::from_le_bytes(declared);
    if header_len.checked_add(8).is_none_or(|end| end > total) {
        return Err(std::io::Error::other(
            "fixture header runs past the end of the file",
        ));
    }
    let mut header = vec![0_u8; usize::try_from(header_len).map_err(std::io::Error::other)?];
    input.read_exact(&mut header)?;

    let mut output = std::fs::File::create(destination)?;
    // Same ordering rule as everywhere else: after the create, before the extend.
    mark_sparse(destination);
    output.write_all(&declared)?;
    output.write_all(&header)?;
    output.set_len(total)?;
    Ok(())
}

/// Write a sparse, validation-complete text-encoder fixture for provider load-gate tests.
///
/// Every required layer projection, norm, architecture signature, and optional packed affine
/// triple is present. The payload is sparse, so production loaders must not execute it; tests use it
/// only to prove fail-fast contract admission before deferred materialization.
pub fn write_encoder_contract_fixture(
    root: &std::path::Path,
    contract: EncoderContract,
) -> std::io::Result<()> {
    write_encoder_contract_fixture_with_quant(root, contract, None)
}

/// Extend the sparse language fixture with the exact `vision_config` and `visual.*` tensor surface.
/// This is for edit/load-gate tests only; production loaders must not materialize the sparse file.
pub fn write_multimodal_encoder_contract_fixture(
    root: &std::path::Path,
    language: EncoderContract,
    vision: VisionEncoderContract,
) -> std::io::Result<()> {
    write_multimodal_encoder_contract_fixture_with_quant(root, language, vision, None)
}

/// Packed counterpart to [`write_multimodal_encoder_contract_fixture`]. The language matrices use
/// the requested Q4/Q8 affine triples while the checkpoint-coupled vision surface remains dense,
/// matching multimodal providers whose runtime quantizes only the substitutable language tower.
pub fn write_multimodal_encoder_contract_fixture_with_quant(
    root: &std::path::Path,
    language: EncoderContract,
    vision: VisionEncoderContract,
    quant_bits: Option<i32>,
) -> std::io::Result<()> {
    use std::io::{Read as _, Seek as _, SeekFrom, Write as _};

    write_encoder_contract_fixture_with_quant(root, language, quant_bits)?;
    vision
        .validate_definition(&language)
        .map_err(|error| std::io::Error::other(error.to_string()))?;

    let config_path = root.join("config.json");
    let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    match vision.architecture {
        gen_core::VisionEncoderArchitecture::Qwen3Vl
        | gen_core::VisionEncoderArchitecture::Qwen2_5Vl => {
            let model_type = match vision.architecture {
                gen_core::VisionEncoderArchitecture::Qwen3Vl => "qwen3_vl",
                gen_core::VisionEncoderArchitecture::Qwen2_5Vl => "qwen2_5_vl",
                gen_core::VisionEncoderArchitecture::PixtralMistral3 => unreachable!(),
            };
            let mut vision_config = serde_json::json!({
                "model_type": model_type,
                "hidden_act": vision.hidden_activation,
                "hidden_size": vision.hidden_size,
                "intermediate_size": vision.intermediate_size,
                "depth": vision.num_hidden_layers,
                "num_heads": vision.num_attention_heads,
                "out_hidden_size": vision.output_width,
                "rope_theta": vision.rope_theta.get(),
                "patch_size": vision.patch_size,
                "temporal_patch_size": vision.temporal_patch_size,
                "spatial_merge_size": vision.spatial_merge_size,
                "in_channels": vision.in_channels,
            });
            let normalization_field = match vision.architecture {
                gen_core::VisionEncoderArchitecture::Qwen3Vl => "layer_norm_eps",
                gen_core::VisionEncoderArchitecture::Qwen2_5Vl => "rms_norm_eps",
                gen_core::VisionEncoderArchitecture::PixtralMistral3 => unreachable!(),
            };
            vision_config[normalization_field] = serde_json::json!(vision.normalization_eps.get());
            if let Some(value) = vision.num_position_embeddings {
                vision_config["num_position_embeddings"] = serde_json::json!(value);
            }
            if !vision.deepstack_visual_indexes.is_empty() {
                vision_config["deepstack_visual_indexes"] =
                    serde_json::json!(vision.deepstack_visual_indexes);
            }
            if let Some(value) = vision.window_size {
                vision_config["window_size"] = serde_json::json!(value);
            }
            if !vision.full_attention_block_indexes.is_empty() {
                vision_config["fullatt_block_indexes"] =
                    serde_json::json!(vision.full_attention_block_indexes);
            }
            config["vision_config"] = vision_config;
        }
        gen_core::VisionEncoderArchitecture::PixtralMistral3 => {
            let text_config = config;
            config = serde_json::json!({
                "architectures": ["Mistral3ForConditionalGeneration"],
                "image_token_index": 10,
                "model_type": "mistral3",
                "multimodal_projector_bias": false,
                "projector_hidden_act": "gelu",
                "spatial_merge_size": vision.spatial_merge_size,
                "text_config": text_config,
                "vision_config": {
                    "attention_dropout": 0.0,
                    "head_dim": vision.hidden_size / vision.num_attention_heads,
                    "hidden_act": vision.hidden_activation,
                    "hidden_size": vision.hidden_size,
                    "intermediate_size": vision.intermediate_size,
                    "model_type": "pixtral",
                    "num_attention_heads": vision.num_attention_heads,
                    "num_channels": vision.in_channels,
                    "num_hidden_layers": vision.num_hidden_layers,
                    "patch_size": vision.patch_size,
                    "rope_theta": vision.rope_theta.get()
                },
                "vision_feature_layer": -1
            });
        }
    }
    std::fs::write(&config_path, serde_json::to_vec(&config)?)?;

    let weights_path = root.join("model.safetensors");
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&weights_path)?;
    // Redundant today, and deliberately kept: the language writer called at the top of this
    // function created and flagged this exact file, and `OPEN_EXISTING` preserves that flag. What
    // the re-assertion buys is that the `set_len` below stays correct on its own terms — it only
    // *grows* the file, so it would allocate the new range against any base that was not flagged.
    mark_sparse(&weights_path);
    let mut len = [0_u8; 8];
    file.read_exact(&mut len)?;
    let old_header_len = u64::from_le_bytes(len) as usize;
    let mut encoded = vec![0_u8; old_header_len];
    file.read_exact(&mut encoded)?;
    let mut header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&encoded)?;
    let mut offset = header
        .values()
        .filter_map(|entry| entry.get("data_offsets")?.as_array()?.get(1)?.as_u64())
        .max()
        .unwrap_or(0);
    for (name, shape) in vision
        .expected_headers()
        .map_err(|error| std::io::Error::other(error.to_string()))?
    {
        let bytes = shape.iter().try_fold(2_u64, |total, &dimension| {
            total.checked_mul(dimension as u64)
        });
        let bytes = bytes.ok_or_else(|| std::io::Error::other("vision fixture size overflow"))?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| std::io::Error::other("vision fixture offset overflow"))?;
        header.insert(
            name,
            serde_json::json!({"dtype": "F16", "shape": shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let encoded = serde_json::to_vec(&header)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&(encoded.len() as u64).to_le_bytes())?;
    file.write_all(&encoded)?;
    file.set_len(8 + encoded.len() as u64 + offset)?;
    Ok(())
}

/// Build the exact tensor-header facts serialized by [`write_encoder_contract_fixture_with_quant`]
/// without creating its production-sized sparse payload.
///
/// Footprint tests should use these facts when their subject is projection or numeric-tier policy,
/// reserving the sparse writer for bounded source-admission and loader-wiring coverage. The writer
/// below consumes this function too, so the in-memory and on-disk fixture surfaces cannot drift.
pub fn encoder_contract_fixture_tensor_headers(
    contract: EncoderContract,
    quant_bits: Option<i32>,
) -> std::io::Result<Vec<gen_core::SafetensorsTensorHeader>> {
    use gen_core::weightsmeta::Dtype;

    if quant_bits.is_some_and(|bits| !matches!(bits, 4 | 8)) {
        return Err(std::io::Error::other(
            "encoder fixture quantization must be Q4 or Q8",
        ));
    }
    if quant_bits.is_some() && contract.packing.is_none() {
        return Err(std::io::Error::other(
            "dense-only encoder contract cannot build packed header facts",
        ));
    }

    let prefix = match contract.architecture {
        "qwen3" | "qwen2_5_vl_text" => "model",
        "qwen3_vl_text" => "language_model",
        "mistral" => "language_model.model",
        architecture => {
            return Err(std::io::Error::other(format!(
                "test fixture has no header signature for {architecture}"
            )))
        }
    };
    fn push_matrix(
        tensors: &mut Vec<(String, Vec<usize>, Dtype)>,
        quant: Option<(i32, usize)>,
        base: String,
        output: usize,
        input: usize,
    ) {
        if let Some((bits, group_size)) = quant {
            let bits = bits as usize;
            tensors.extend([
                (
                    format!("{base}.weight"),
                    vec![output, input * bits / 32],
                    Dtype::U32,
                ),
                (
                    format!("{base}.scales"),
                    vec![output, input / group_size],
                    Dtype::F16,
                ),
                (
                    format!("{base}.biases"),
                    vec![output, input / group_size],
                    Dtype::F16,
                ),
            ]);
        } else {
            tensors.push((format!("{base}.weight"), vec![output, input], Dtype::F16));
        }
    }

    let mut tensors: Vec<(String, Vec<usize>, Dtype)> = Vec::new();
    let packing = quant_bits.zip(contract.packing);
    push_matrix(
        &mut tensors,
        packing
            .filter(|(_, packing)| packing.pack_embedding)
            .map(|(bits, packing)| (bits, packing.group_size)),
        format!("{prefix}.embed_tokens"),
        contract.vocab_size,
        contract.hidden_size,
    );
    let attention_width = contract.num_attention_heads * contract.head_dim;
    let kv_width = contract.num_key_value_heads * contract.head_dim;
    for layer in 0..contract.loaded_hidden_layers {
        let base = format!("{prefix}.layers.{layer}");
        for (suffix, output, input) in [
            ("self_attn.q_proj", attention_width, contract.hidden_size),
            ("self_attn.k_proj", kv_width, contract.hidden_size),
            ("self_attn.v_proj", kv_width, contract.hidden_size),
            ("self_attn.o_proj", contract.hidden_size, attention_width),
            (
                "mlp.gate_proj",
                contract.intermediate_size,
                contract.hidden_size,
            ),
            (
                "mlp.up_proj",
                contract.intermediate_size,
                contract.hidden_size,
            ),
            (
                "mlp.down_proj",
                contract.hidden_size,
                contract.intermediate_size,
            ),
        ] {
            push_matrix(
                &mut tensors,
                packing.map(|(bits, packing)| (bits, packing.group_size)),
                format!("{base}.{suffix}"),
                output,
                input,
            );
        }
        tensors.extend([
            (
                format!("{base}.input_layernorm.weight"),
                vec![contract.hidden_size],
                Dtype::F16,
            ),
            (
                format!("{base}.post_attention_layernorm.weight"),
                vec![contract.hidden_size],
                Dtype::F16,
            ),
        ]);
        match contract.architecture {
            "qwen3" | "qwen3_vl_text" => tensors.extend([
                (
                    format!("{base}.self_attn.q_norm.weight"),
                    vec![contract.head_dim],
                    Dtype::F16,
                ),
                (
                    format!("{base}.self_attn.k_norm.weight"),
                    vec![contract.head_dim],
                    Dtype::F16,
                ),
            ]),
            "qwen2_5_vl_text" => tensors.extend([
                (
                    format!("{base}.self_attn.q_proj.bias"),
                    vec![attention_width],
                    Dtype::F16,
                ),
                (
                    format!("{base}.self_attn.k_proj.bias"),
                    vec![kv_width],
                    Dtype::F16,
                ),
                (
                    format!("{base}.self_attn.v_proj.bias"),
                    vec![kv_width],
                    Dtype::F16,
                ),
            ]),
            "mistral" => {}
            _ => unreachable!("unsupported architecture rejected above"),
        }
    }
    if contract.requires_final_norm {
        tensors.push((
            format!("{prefix}.norm.weight"),
            vec![contract.hidden_size],
            Dtype::F16,
        ));
    }
    if contract.requires_lm_head {
        let parent = match contract.architecture {
            "mistral" => "language_model",
            architecture => {
                return Err(std::io::Error::other(format!(
                    "test fixture has no LM-head signature for {architecture}"
                )))
            }
        };
        push_matrix(
            &mut tensors,
            packing
                .filter(|(_, packing)| packing.pack_lm_head)
                .map(|(bits, packing)| (bits, packing.group_size)),
            format!("{parent}.lm_head"),
            contract.vocab_size,
            contract.hidden_size,
        );
    }

    tensors
        .into_iter()
        .map(|(name, shape, dtype)| {
            let element_bytes = match dtype {
                Dtype::U32 => 4,
                Dtype::F16 => 2,
                _ => unreachable!("fixture generator emits only U32 and F16"),
            };
            let data_bytes = shape
                .iter()
                .try_fold(element_bytes, |total: u64, &dimension| {
                    total.checked_mul(dimension as u64)
                });
            let data_bytes =
                data_bytes.ok_or_else(|| std::io::Error::other("encoder fixture size overflow"))?;
            Ok(gen_core::SafetensorsTensorHeader {
                name,
                dtype,
                shape,
                data_bytes,
            })
        })
        .collect()
}

pub fn write_encoder_contract_fixture_with_quant(
    root: &std::path::Path,
    contract: EncoderContract,
    quant_bits: Option<i32>,
) -> std::io::Result<()> {
    use std::io::Write as _;

    std::fs::create_dir_all(root)?;
    // Most provider tests pass `<snapshot>/text_encoder` here. Keep those fixtures truthful now
    // that production validation binds the retained tokenizer too. A standalone component fixture
    // intentionally gets no tokenizer: it is the weights-only inheritance surface.
    if root.file_name().and_then(std::ffi::OsStr::to_str) == Some("text_encoder") {
        if let Some(snapshot_root) = root.parent() {
            write_encoder_contract_tokenizer_fixture(snapshot_root, contract)?;
        }
    }
    let mut config = serde_json::json!({
        "model_type": contract.architecture,
        "hidden_size": contract.hidden_size,
        "intermediate_size": contract.intermediate_size,
        "num_hidden_layers": contract.num_hidden_layers,
        "num_attention_heads": contract.num_attention_heads,
        "num_key_value_heads": contract.num_key_value_heads,
        "head_dim": contract.head_dim,
        "vocab_size": contract.vocab_size,
        "hidden_act": contract.hidden_activation,
        "attention_dropout": contract.attention_dropout.get(),
        "rms_norm_eps": contract.rms_norm_eps.get(),
        "rope_theta": contract.rope_theta.get(),
        "max_position_embeddings": contract.max_position_embeddings,
    });
    if let gen_core::EncoderConfigBool::Required(value) = contract.attention_bias {
        config["attention_bias"] = serde_json::json!(value);
    }
    if let gen_core::EncoderConfigBool::Required(value) = contract.tie_word_embeddings {
        config["tie_word_embeddings"] = serde_json::json!(value);
    }
    for (field, value) in [
        ("bos_token_id", contract.bos_token_id),
        ("eos_token_id", contract.eos_token_id),
        ("image_token_id", contract.image_token_id),
        ("vision_start_token_id", contract.vision_start_token_id),
        ("vision_end_token_id", contract.vision_end_token_id),
    ] {
        if let Some(value) = value {
            config[field] = serde_json::json!(value);
        }
    }
    for required in contract.tokenizer.required_tokens {
        if let Some(field) = required.config_field {
            config[field] = serde_json::json!(required.id);
        }
    }
    if !contract.mrope_section.is_empty() {
        config["rope_scaling"] = serde_json::json!({
            "mrope_section": contract.mrope_section,
            "rope_type": "default",
        });
    }
    if let Some(interleaved) = contract.mrope_interleaved {
        config["rope_scaling"]["mrope_interleaved"] = serde_json::json!(interleaved);
    }
    if let Some(bits) = quant_bits {
        let packing = contract.packing.ok_or_else(|| {
            std::io::Error::other("dense-only encoder contract cannot write a packed fixture")
        })?;
        config["quantization"] =
            serde_json::json!({"bits": bits, "group_size": packing.group_size});
    }
    std::fs::write(root.join("config.json"), serde_json::to_vec(&config)?)?;

    let tensors = encoder_contract_fixture_tensor_headers(contract, quant_bits)?;

    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    for tensor in tensors {
        let dtype = match tensor.dtype {
            gen_core::weightsmeta::Dtype::U32 => "U32",
            gen_core::weightsmeta::Dtype::F16 => "F16",
            _ => unreachable!("fixture generator emits only U32 and F16"),
        };
        let end = offset
            .checked_add(tensor.data_bytes)
            .ok_or_else(|| std::io::Error::other("encoder fixture offset overflow"))?;
        header.insert(
            tensor.name,
            serde_json::json!({"dtype": dtype, "shape": tensor.shape, "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let encoded = serde_json::to_vec(&header)?;
    let weights_path = root.join("model.safetensors");
    let mut file = std::fs::File::create(&weights_path)?;
    // Between the create and the `set_len`: `File::create` is `CREATE_ALWAYS` and would clear an
    // earlier flag, and the `set_len` is what allocates. See [`mark_sparse`].
    mark_sparse(&weights_path);
    file.write_all(&(encoded.len() as u64).to_le_bytes())?;
    file.write_all(&encoded)?;
    file.set_len(8 + encoded.len() as u64 + offset)?;
    Ok(())
}

/// Write a small, parseable tokenizer artifact satisfying the exact literals declared by an
/// encoder contract. The first candidate path is authoritative, matching provider precedence.
pub fn write_encoder_contract_tokenizer_fixture(
    snapshot_root: &std::path::Path,
    contract: EncoderContract,
) -> std::io::Result<std::path::PathBuf> {
    let candidate = contract
        .tokenizer
        .artifact_candidates
        .first()
        .ok_or_else(|| std::io::Error::other("encoder contract has no tokenizer candidate"))?;
    let path = snapshot_root.join(candidate);
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("tokenizer candidate has no parent"))?;
    std::fs::create_dir_all(parent)?;

    let mut vocab = serde_json::Map::new();
    let used = contract
        .tokenizer
        .required_tokens
        .iter()
        .map(|required| required.id)
        .collect::<std::collections::BTreeSet<_>>();
    let unknown_id = (0_i64..)
        .find(|id| !used.contains(id))
        .ok_or_else(|| std::io::Error::other("cannot allocate tokenizer fixture unknown id"))?;
    vocab.insert("<fixture-unk>".into(), serde_json::json!(unknown_id));
    for required in contract.tokenizer.required_tokens {
        vocab.insert(required.literal.into(), serde_json::json!(required.id));
    }
    let added_tokens = contract
        .tokenizer
        .required_tokens
        .iter()
        .map(|required| {
            serde_json::json!({
                "id": required.id,
                "content": required.literal,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect::<Vec<_>>();
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<fixture-unk>",
        },
    });
    std::fs::write(&path, serde_json::to_vec(&tokenizer)?)?;
    Ok(path)
}
