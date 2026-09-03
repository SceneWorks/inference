//! # gen-core-testkit
//!
//! A **contract conformance suite** for gen-core providers — the image/video/pure-audio
//! [`gen_core::Generator`] (this module), [`gen_core::Trainer`](crate::trainer),
//! [`gen_core::Captioner`](crate::captioner), and the audio-contract family (sc-12853):
//! [`gen_core::Generator`] under [`Modality::Audio`]
//! ([`crate::audio_generator`]), [`gen_core::VoiceEmbedder`](crate::voice_embedder),
//! [`gen_core::AudioTransform`](crate::audio_transform),
//! [`gen_core::Transcriber`](crate::transcriber), and
//! [`gen_core::AudioEmbedder`](crate::audio_embedder). Given any boxed provider — an MLX family from
//! `mlx-gen`, a candle-gen provider, or a `crates/audio` candle provider — it exercises the
//! behavioral guarantees the contract *promises but cannot express in the type system*: typed
//! cancellation, progress monotonicity, seed determinism, and capability honesty. Both backends run
//! it in CI, so a provider that silently ignores `CancelFlag` or reports no progress (the sc-4380
//! class of bug) becomes a CI failure instead of a field report (epic 3720, sc-4481/sc-4895).
//!
//! The testkit has **zero tensor dependencies** — it depends only on `gen-core` and drives the
//! provider purely through the public contract, so it builds and runs on the Linux gen-core lane
//! against an in-crate stub exactly as it does on the macOS lane against a real MLX family.
//!
//! ## Usage
//!
//! ```ignore
//! // macOS lane, real family — generator, trainer, captioner:
//! let registry = mlx_gen_z_image::provider_registry().unwrap();
//! gen_core_testkit::conformance(
//!     || registry.load("z_image_turbo", &spec).unwrap(),
//!     &gen_core_testkit::Profile::cheap(),
//! );
//! gen_core_testkit::trainer_conformance(
//!     || registry.load_trainer("z_image_turbo", &spec).unwrap(),
//!     &gen_core_testkit::TrainerProfile::cheap(items, out_dir),
//! );
//! let registry = mlx_gen_joycaption::provider_registry().unwrap();
//! gen_core_testkit::captioner_conformance(
//!     || registry.load_captioner("joy_caption", &spec).unwrap(),
//!     &gen_core_testkit::CaptionerProfile::cheap(),
//! );
//! ```
//!
//! The individual `check_*` functions are public so a provider's own tests can target one guarantee
//! at a time; the `*_conformance` entry points run them all and panic with the aggregated failures.

pub mod audio_embedder;
pub mod audio_generator;
pub mod audio_transform;
pub mod captioner;
pub mod memory_strategy;
pub mod trainer;
pub mod transcriber;
pub mod voice_embedder;

pub use audio_embedder::{
    audio_embedder_conformance, check_audio_embed_joint, check_audio_embedder_registry,
    AudioEmbedderProfile,
};
pub use audio_generator::{
    audio_conformance, check_audio_cancellation, check_audio_multi_speaker, check_audio_output,
    check_audio_precancellation, check_audio_progress, check_audio_progress_contract,
    check_audio_seed_determinism, check_audio_streaming, check_audio_validate_honesty,
    check_multi_turn, check_video_to_audio, AudioProfile,
};
pub use audio_transform::{
    audio_transform_conformance, check_audio_transform_cardinality,
    check_audio_transform_coherence, check_audio_transform_registry,
    check_audio_transform_validate, AudioTransformProfile,
};
pub use captioner::{
    captioner_conformance, check_captioner_cancellation, check_captioner_progress,
    check_captioner_registry, check_captioner_validate, CaptionerProfile,
};
pub use memory_strategy::{
    assert_memory_contract_asset_facts_conform, assert_memory_contract_facts_conform,
    check_memory_contract_asset_facts, check_memory_contract_facts,
    check_memory_contract_surface_registry, check_memory_strategy_contract,
    check_memory_strategy_registry, memory_contract_surface_registry_conformance,
    memory_strategy_conformance, memory_strategy_registry_conformance,
};
pub use trainer::{
    check_trainer_cancellation, check_trainer_progress, check_trainer_registry,
    check_trainer_validate, trainer_conformance, TrainerProfile,
};
pub use transcriber::{
    check_transcriber_cancellation, check_transcriber_output, check_transcriber_progress,
    check_transcriber_registry, check_transcriber_validate, transcriber_conformance,
    TranscriberProfile,
};
pub use voice_embedder::{
    check_voice_embed, check_voice_embed_rejects_short, check_voice_embedder_registry,
    voice_embedder_conformance, VoiceEmbedderProfile,
};

use gen_core::{
    Capabilities, Conditioning, EncoderContract, Error, GenerationOutput, GenerationRequest,
    Generator, Image, Modality, Progress, StepSupport, VisionEncoderContract,
};

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

/// The lax `Progress::Step` monotonicity contract used by the captioner conformance checks (6942;
/// the text-LLM checks left with sc-7189): at least one step; a constant non-zero `total`; a
/// strictly-increasing `current` in
/// `1..=total`. `id` labels the model, `op` the emitting method (e.g. `"caption()"`) for the
/// no-events error. Token/phase-based decoders use this rather than the generator's exact-step check.
pub(crate) fn check_progress_steps(id: &str, op: &str, steps: &[(u32, u32)]) -> Result<(), String> {
    if steps.is_empty() {
        return Err(format!(
            "progress[{id}]: {op} emitted no Progress::Step events"
        ));
    }
    let total = steps[0].1;
    if total == 0 {
        return Err(format!("progress[{id}]: Progress::Step.total was 0"));
    }
    let mut prev = 0u32;
    for &(current, t) in steps {
        if t != total {
            return Err(format!(
                "progress[{id}]: Step.total changed mid-run ({total} then {t})"
            ));
        }
        if current < 1 || current > total {
            return Err(format!(
                "progress[{id}]: Step.current {current} out of range 1..={total}"
            ));
        }
        if current <= prev {
            return Err(format!(
                "progress[{id}]: Step.current must strictly increase; saw {prev} then {current}"
            ));
        }
        prev = current;
    }
    Ok(())
}

/// Cheap-request parameters for the conformance run. Keep these at the model's *minimum* valid
/// size and a tiny step count — the suite runs `generate` several times, so the macOS-lane cost is
/// `~4 ×` one cheap render.
#[derive(Clone, Debug)]
pub struct Profile {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    /// Denoise steps the request asks for **and** the value the model is expected to resolve to:
    /// [`check_progress`] asserts `Progress::Step.total == steps`. If a model clamps/transforms
    /// `req.steps`, set this to the resolved count, not the requested one.
    pub steps: u32,
    pub seed: u64,
    /// Steps requested for [`check_cancellation`] only — needs headroom (≥ 3) so that a provider
    /// honoring cancellation visibly stops before completion. Generation is cancelled at the first
    /// step boundary, so only ~1 forward actually runs regardless of this value.
    pub cancel_steps: u32,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            prompt: "a fox".to_owned(),
            width: 256,
            height: 256,
            steps: 2,
            seed: 42,
            cancel_steps: 6,
        }
    }
}

impl Profile {
    /// The cheapest generally-valid profile: 256², 2 steps, fixed seed. 256 is a multiple of the
    /// common VAE×patch alignment (16) and ≥ every current family's `min_size`.
    pub fn cheap() -> Self {
        Self::default()
    }
}

/// The in-capability request the positive checks expect the model to accept (and the
/// progress/seed checks render from). Only the fields the profile pins are set; everything else is
/// the contract default (notably `count: 1`).
fn base_request(profile: &Profile) -> GenerationRequest {
    GenerationRequest {
        prompt: profile.prompt.clone(),
        width: profile.width,
        height: profile.height,
        steps: Some(profile.steps),
        seed: Some(profile.seed),
        ..Default::default()
    }
}

/// The raw output pixels, flattened across images/frames — the unit the seed-determinism check
/// compares byte-for-byte. `pub(crate)` so the audio-generator harness ([`crate::audio_generator`])
/// reuses the same byte extraction for its own seed-determinism check over `GenerationOutput::Audio`.
pub(crate) fn output_bytes(out: &GenerationOutput) -> Vec<u8> {
    match out {
        GenerationOutput::Images(imgs) => {
            imgs.iter().flat_map(|i| i.pixels.iter().copied()).collect()
        }
        GenerationOutput::Video { frames, .. } => frames
            .iter()
            .flat_map(|f| f.pixels.iter().copied())
            .collect(),
        // Pure audio (sc-12834): the PCM samples' little-endian bytes are the deterministic unit.
        GenerationOutput::Audio(track) => {
            track.samples.iter().flat_map(|s| s.to_le_bytes()).collect()
        }
    }
}

/// A `width × height` all-zero RGB image, for building conditioning the model should reject.
fn blank_image(profile: &Profile) -> Image {
    Image {
        width: profile.width,
        height: profile.height,
        pixels: vec![0u8; profile.width as usize * profile.height as usize * 3],
    }
}

/// The first easily-constructed [`Conditioning`] whose kind the model does **not** advertise, or
/// `None` if it accepts all of the candidates (then the negative-conditioning sub-check is skipped).
fn undeclared_conditioning(caps: &Capabilities, profile: &Profile) -> Option<Conditioning> {
    [
        Conditioning::Mask {
            image: blank_image(profile),
        },
        Conditioning::Depth {
            image: blank_image(profile),
        },
        Conditioning::Reference {
            image: blank_image(profile),
            strength: None,
        },
    ]
    .into_iter()
    .find(|c| !caps.accepts(c.kind()))
}

/// **Validate honesty.** Everything the descriptor advertises is accepted by `validate()`, and
/// requests that exceed the advertised surface (oversize, overcount, undeclared conditioning) are
/// rejected by `validate()` — *before* any expensive work, not by `generate()` panicking later.
pub fn check_validate_honesty(g: &dyn Generator, profile: &Profile) -> Result<(), String> {
    let desc = g.descriptor();
    let caps = &desc.capabilities;
    let id = desc.id;

    // Positive: the declared cheap request must be accepted.
    g.validate(&base_request(profile)).map_err(|e| {
        format!("validate-honesty[{id}]: the in-capability cheap request ({}x{}, {} steps) was rejected by validate(): {e}", profile.width, profile.height, profile.steps)
    })?;

    // Positive: every advertised sampler must be accepted.
    for &s in &caps.samplers {
        let mut r = base_request(profile);
        r.sampler = Some(s.to_owned());
        if let Err(e) = g.validate(&r) {
            return Err(format!(
                "validate-honesty[{id}]: advertised sampler {s:?} was rejected by validate(): {e}"
            ));
        }
    }

    // Positive + negative: the advertised step surface must be honest — an exact menu (sc-19502) or
    // an inclusive range (sc-19559). Every ADMITTED count must validate, and a count outside must be
    // refused; the pair is the point, because each half alone is satisfiable by a broken provider: a
    // lane that ignores `req.steps` entirely (which `mlx-gen-ltx` did) passes the positive half
    // trivially, and a lane that rejects everything passes the negative half trivially.
    //
    // Skipped entirely for the unconstrained majority, so this adds no requirement to a model that
    // never opted in.
    if !caps.supported_steps.is_unconstrained() {
        // Probe the ENDPOINTS, not the whole interval: SVD's range is 1..=200 and Kolors' is
        // 1..=1100, so enumerating every admitted count would run 1100 validate() calls for no
        // extra signal. An exact menu is small, so it is probed in full.
        let admitted: Vec<u32> = match &caps.supported_steps {
            StepSupport::Unconstrained => Vec::new(),
            StepSupport::Exact(counts) => counts.clone(),
            StepSupport::Range { min, max } => {
                if min == max {
                    vec![*min]
                } else {
                    vec![*min, *max]
                }
            }
        };
        for steps in admitted {
            let mut r = base_request(profile);
            r.steps = Some(steps);
            if let Err(e) = g.validate(&r) {
                return Err(format!(
                    "validate-honesty[{id}]: advertised step count {steps} was rejected by validate(): {e}"
                ));
            }
        }
        // The smallest positive count the model does NOT admit. Searching for a GAP rather than
        // always probing `ceiling + 1` keeps this meaningful for a model with a discontinuous menu
        // (a range check would admit the gap but still refuse `ceiling + 1`). The search bound is
        // `ceiling + 1`, which no constrained surface admits, so it always finds something.
        let ceiling = caps.supported_steps.ceiling().unwrap_or(0) + 1;
        if let Some(off) = (1..=ceiling).find(|s| !caps.supported_steps.admits(*s)) {
            let mut r = base_request(profile);
            r.steps = Some(off);
            if g.validate(&r).is_ok() {
                return Err(format!(
                    "validate-honesty[{id}]: step count {off} is outside the advertised surface \
                     {:?} but was accepted by validate() — an advertised step constraint that \
                     admits other counts silently ignores the caller's `steps`",
                    caps.supported_steps
                ));
            }
        }
    }

    // Negative: a size above max_size must be rejected — but only for providers whose contract
    // includes a size axis. Audio-lane providers (Modality::Audio) legitimately do NOT range-check
    // width/height: those fields are meaningless for audio, so a conformant audio model validates
    // through the size-skipping floor (Capabilities::validate_request_audio) and advertises no size
    // bound (min_size/max_size are the unused 0 — sc-13314). Probing max_size+64 (== 64x64 when
    // max_size is 0) would misfire against that exemption — the provider correctly accepts it
    // (sc-13705). The oversize check stays fully enforced for image/video providers, which always
    // advertise a real max_size; an audio provider's own surface honesty is covered by the
    // purpose-built audio_conformance suite (crate::audio_generator). This is the one entry point an
    // audio family may share with the image suite (moss-sfx, acestep), so the exemption lives here.
    if desc.modality != Modality::Audio {
        if let Some(big) = caps.max_size.checked_add(64) {
            let mut r = base_request(profile);
            r.width = big;
            r.height = big;
            if g.validate(&r).is_ok() {
                return Err(format!(
                    "validate-honesty[{id}]: a {big}x{big} request (above max_size {}) was accepted by validate()",
                    caps.max_size
                ));
            }
        }
    }

    // Negative: a count above max_count must be rejected.
    if let Some(many) = caps.max_count.checked_add(1) {
        let mut r = base_request(profile);
        r.count = many;
        if g.validate(&r).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: count {many} (above max_count {}) was accepted by validate()",
                caps.max_count
            ));
        }
    }

    // Negative: an undeclared conditioning kind must be rejected.
    if let Some(cond) = undeclared_conditioning(caps, profile) {
        let kind = cond.kind();
        let mut r = base_request(profile);
        r.conditioning = vec![cond];
        if g.validate(&r).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: undeclared {kind:?} conditioning was accepted by validate() \
                 (descriptor advertises {:?})",
                caps.conditioning
            ));
        }
    }

    Ok(())
}

/// **Progress.** `Progress::Step{current,total}` is monotone and complete: `current` runs exactly
/// `1..=total`, `total` is constant, and `total` equals the profile's resolved step count.
pub fn check_progress(g: &dyn Generator, profile: &Profile) -> Result<(), String> {
    check_progress_with(g, &base_request(profile), Some(profile.steps))
}

/// **Progress (request-supplied).** The general form of [`check_progress`] for providers whose
/// `generate` needs a request the text-only `base_request` cannot express — image→video (SVD),
/// super-resolution (SeedVR2), and the renderer families (Bernini, scail2), the same shape as
/// [`check_cancellation_with`]. Asserts `Progress::Step{current,total}` is monotone and complete
/// (`current` runs exactly `1..=total`, `total` constant); when `expected_total` is `Some`, `total`
/// must equal it (pass the value the model resolves the request's step count to — for a multi-stage
/// pipeline that folds its stages into one bar, the folded grand total).
pub fn check_progress_with(
    g: &dyn Generator,
    req: &GenerationRequest,
    expected_total: Option<u32>,
) -> Result<(), String> {
    let id = g.descriptor().id;
    let mut steps: Vec<(u32, u32)> = Vec::new();
    g.generate(req, &mut |p| {
        if let Progress::Step { current, total } = p {
            steps.push((current, total));
        }
    })
    .map_err(|e| format!("progress[{id}]: generate() failed on the cheap request: {e}"))?;

    if steps.is_empty() {
        return Err(format!(
            "progress[{id}]: generate() emitted no Progress::Step events"
        ));
    }
    let total = steps[0].1;
    if let Some((c, t)) = steps.iter().find(|(_, t)| *t != total) {
        return Err(format!(
            "progress[{id}]: Step.total changed mid-run ({total} then {t} at current={c})"
        ));
    }
    let observed: Vec<u32> = steps.iter().map(|(c, _)| *c).collect();
    let expected: Vec<u32> = (1..=total).collect();
    if observed != expected {
        return Err(format!(
            "progress[{id}]: Step.current must be exactly 1..={total} (monotone, complete, no repeats); got {observed:?}"
        ));
    }
    if let Some(want) = expected_total {
        if total != want {
            return Err(format!(
                "progress[{id}]: Step.total ({total}) != the expected resolved step count ({want}). \
                 Pass the value the model resolves the request's steps to.",
            ));
        }
    }
    Ok(())
}

/// **Progress contract (the whole-class property, sc-11133 / F-030/F-050/F-136/F-162/F-164).** The
/// structural invariant every `generate` must satisfy, checked over the *ordered* event stream rather
/// than just the `Step` values [`check_progress`] inspects:
///
/// 1. **Monotone, in-bounds `Step`.** `Step.current` is non-decreasing and never exceeds `Step.total`
///    (no >100% overrun — the F-050 multi-eval-sampler class), and `Step.total` is constant.
/// 2. **Reaches `total`.** The final `Step.current` equals `Step.total` — the bar is never frozen
///    below completion (the F-030 PiD-early-stop class where a truncated schedule never reaches its
///    advertised total).
/// 3. **`Decoding` exactly once.** The terminal `Progress::Decoding` phase is emitted once — not zero
///    times (frozen through the longest stage — F-030) and not once-per-output (the restarting-bar
///    class — F-136/F-162).
///
/// This is deliberately *laxer than* [`check_progress`] on the `Step` axis (non-decreasing with
/// repeats allowed, rather than exactly `1..=total`) so it fits multi-stage / folded-bar pipelines,
/// while adding the `Decoding`-cardinality guarantee [`check_progress`] does not express. Providers
/// that fold N outputs into one bar (`total = N × steps`) satisfy it; providers that restart the bar
/// per output, overrun it, or skip `Decoding` fail it. Use [`check_progress_contract_with`] for the
/// image→video / super-resolution / renderer families whose `generate` needs a request the text-only
/// `base_request` cannot express.
pub fn check_progress_contract(g: &dyn Generator, profile: &Profile) -> Result<(), String> {
    check_progress_contract_with(g, &base_request(profile))
}

/// **Progress contract (request-supplied).** The general form of [`check_progress_contract`] for
/// providers whose `generate` needs conditioning the text-only `base_request` cannot supply
/// (image→video, super-resolution, the renderer families) — the same shape as
/// [`check_cancellation_with`]. Asserts the same three invariants: monotone in-bounds `Step`, the bar
/// reaches `total`, and `Progress::Decoding` is emitted exactly once.
pub fn check_progress_contract_with(
    g: &dyn Generator,
    req: &GenerationRequest,
) -> Result<(), String> {
    let id = g.descriptor().id;
    let mut steps: Vec<(u32, u32)> = Vec::new();
    let mut decoding_count = 0u32;
    g.generate(req, &mut |p| match p {
        Progress::Step { current, total } => steps.push((current, total)),
        Progress::Decoding => decoding_count += 1,
        Progress::Loading(_) => {}
    })
    .map_err(|e| format!("progress-contract[{id}]: generate() failed on the cheap request: {e}"))?;

    if steps.is_empty() {
        return Err(format!(
            "progress-contract[{id}]: generate() emitted no Progress::Step events"
        ));
    }
    let total = steps[0].1;
    if total == 0 {
        return Err(format!(
            "progress-contract[{id}]: Progress::Step.total was 0"
        ));
    }
    let mut prev = 0u32;
    for &(current, t) in &steps {
        if t != total {
            return Err(format!(
                "progress-contract[{id}]: Step.total changed mid-run ({total} then {t} at current={current})"
            ));
        }
        if current > total {
            return Err(format!(
                "progress-contract[{id}]: Step.current {current} exceeds total {total} (>100% overrun — \
                 a multi-eval sampler or wrapped counter must clamp/derive current so it never overruns; F-050)"
            ));
        }
        if current < prev {
            return Err(format!(
                "progress-contract[{id}]: Step.current must be monotone non-decreasing; saw {prev} then {current}"
            ));
        }
        prev = current;
    }
    if prev != total {
        return Err(format!(
            "progress-contract[{id}]: Step.current reached {prev} but total is {total} — the bar must reach \
             its total by completion (a truncated/early-stopped schedule must report the effective total, not \
             a stale one; F-030)"
        ));
    }
    if decoding_count != 1 {
        return Err(format!(
            "progress-contract[{id}]: Progress::Decoding was emitted {decoding_count} times, expected exactly 1 \
             (zero = the decode stage is invisible/frozen; >1 = the bar restarts per output — F-030/F-136/F-162)"
        ));
    }
    Ok(())
}

/// **Cancellation.** Tripping `CancelFlag` at the first step boundary makes `generate` return the
/// **typed** `Err(Error::Canceled)` (not a stringified `Msg`) within a bounded number of further
/// steps (≤ 2), and produces no partial output.
pub fn check_cancellation(g: &dyn Generator, profile: &Profile) -> Result<(), String> {
    let mut req = base_request(profile);
    // `cancel_steps` is headroom, not a knob every model has (sc-19502). A model advertising a fixed
    // schedule (`Capabilities::supported_steps`) refuses any other count, so handing it the profile's
    // headroom value would fail `validate` and report a CANCELLATION defect for a step-count
    // rejection. Fall back to the largest advertised count — it is the most headroom the model can
    // legally be given, and for the distilled LTX schedule that is 8, well past the ≥ 3 this needs.
    let caps = &g.descriptor().capabilities;
    req.steps = Some(if caps.supported_steps.admits(profile.cancel_steps) {
        profile.cancel_steps
    } else {
        // The most headroom the model can legally be given. For the distilled LTX schedule that is
        // 8, well past the ≥ 3 this needs; for a ranged model the profile value is admitted, so
        // this arm is only reached by an exact menu.
        caps.supported_steps
            .ceiling()
            .unwrap_or(profile.cancel_steps)
    });
    check_cancellation_with(g, &req)
}

/// **Cancellation (request-supplied).** The general form of [`check_cancellation`] for providers
/// whose `generate` needs conditioning the text-only `base_request` cannot supply — image→video
/// (SVD), super-resolution (SeedVR2), and the renderer families (Bernini, scail2). The caller builds
/// a model-appropriate request (its own conditioning + a step count with headroom, ≥ 3, so a
/// honoring provider visibly stops before completion); this helper trips `req.cancel` at the first
/// emitted `Progress::Step` and asserts `generate` returns the **typed** `Err(Error::Canceled)`
/// within a bounded number of further steps (≤ 2), with no partial output.
pub fn check_cancellation_with(g: &dyn Generator, req: &GenerationRequest) -> Result<(), String> {
    let id = g.descriptor().id;
    let cancel = req.cancel.clone();

    let mut tripped = false;
    let mut steps_after_trip = 0u32;
    let mut last_current = 0u32;
    let result = g.generate(req, &mut |p| {
        if let Progress::Step { current, .. } = p {
            last_current = current;
            if tripped {
                steps_after_trip += 1;
            } else {
                cancel.cancel();
                tripped = true;
            }
        }
    });

    if !tripped {
        return Err(format!(
            "cancellation[{id}]: no Progress::Step was emitted, so cancellation could not be exercised \
             (a provider must report step progress for cooperative cancellation to be observable)"
        ));
    }
    match result {
        Ok(_) => Err(format!(
            "cancellation[{id}]: generate() ran to completion despite CancelFlag set at step 1 \
             (reached step {last_current}); it must return Err(Error::Canceled)"
        )),
        Err(Error::Canceled) if steps_after_trip > 2 => Err(format!(
            "cancellation[{id}]: returned Canceled but emitted {steps_after_trip} further Progress::Step events \
             after the cancel trip (contract allows at most 2)"
        )),
        Err(Error::Canceled) => Ok(()),
        Err(other) => Err(format!(
            "cancellation[{id}]: must return the typed Err(Error::Canceled) on cancel, got {other:?} \
             — a stringified Error::Msg breaks the typed-cancellation contract (epic 3720 D3)"
        )),
    }
}

/// **Pre-generate cancellation (the non-denoise-seam class).** A request whose `CancelFlag` is
/// **already tripped when `generate` is called** must return the typed `Err(Error::Canceled)` and
/// produce no output — the provider must consult the flag *before* running its expensive pre-denoise
/// work (prompt/vision encode, reference VAE encodes, identity tower, sequential component loads),
/// not only inside the denoise loop.
///
/// This complements [`check_cancellation`], which trips the flag *mid-denoise* (at the first emitted
/// `Progress::Step`) and therefore only exercises the denoise loop. The whole class of "cancellation
/// regresses at the encode / VAE-decode / identity / load seams the denoise loop doesn't cover"
/// (the sc-11128 / F-018/F-019/F-029/F-037/F-108/F-142/F-135 family) is what this check mechanically
/// guards: a provider that runs its full encode→denoise→decode before ever looking at the flag fails
/// here even though it might pass the mid-denoise check. Mirrors the captioner's pre-inference
/// cancellation contract ([`check_captioner_cancellation`]).
///
/// Note on lazy backends: because this hands an *already*-cancelled request, the provider's up-front
/// check observes the trip without needing a forced `eval` — the false-green trap (a cancel arriving
/// *during* a lazily-built encode) is a per-provider concern the provider's own tests must cover by
/// forcing materialization at the seam; the contract-level guarantee this enforces is that such a
/// check exists at all.
pub fn check_precancellation(g: &dyn Generator, profile: &Profile) -> Result<(), String> {
    check_precancellation_with(g, &base_request(profile))
}

/// **Pre-generate cancellation (request-supplied).** The general form of [`check_precancellation`]
/// for providers whose `generate` needs conditioning the text-only `base_request` cannot supply
/// (image→video, super-resolution, the renderer families) — the same shape as
/// [`check_cancellation_with`]. Trips `req.cancel` up front, then asserts the typed
/// `Err(Error::Canceled)` with no partial output.
pub fn check_precancellation_with(
    g: &dyn Generator,
    req: &GenerationRequest,
) -> Result<(), String> {
    let id = g.descriptor().id;
    let mut req = req.clone();
    req.cancel = Default::default();
    req.cancel.cancel();
    match g.generate(&req, &mut |_| {}) {
        Ok(_) => Err(format!(
            "pre-cancellation[{id}]: generate() returned Ok despite a CancelFlag already tripped at \
             call time; it must consult req.cancel before its pre-denoise encode/load work and return \
             Err(Error::Canceled)"
        )),
        Err(Error::Canceled) => Ok(()),
        Err(other) => Err(format!(
            "pre-cancellation[{id}]: must return the typed Err(Error::Canceled) for an already-cancelled \
             request, got {other:?} — a provider that only checks cancel inside the denoise loop (or \
             stringifies the error) fails the non-denoise-seam contract (sc-11128)"
        )),
    }
}

/// **Seed determinism (same backend).** Two runs of the identical request+seed produce
/// byte-identical output. Cross-backend equality is *not* a goal (RNG algorithms differ); this is
/// the guarantee that makes the seeded per-step RNG (D6) mandatory.
pub fn check_seed_determinism(g: &dyn Generator, profile: &Profile) -> Result<(), String> {
    let id = g.descriptor().id;
    let req = base_request(profile);
    let a = g
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("seed[{id}]: first generate() failed: {e}"))?;
    let b = g
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("seed[{id}]: second generate() failed: {e}"))?;
    let (ba, bb) = (output_bytes(&a), output_bytes(&b));
    if ba.len() != bb.len() {
        return Err(format!(
            "seed[{id}]: same seed produced different output sizes ({} vs {} bytes)",
            ba.len(),
            bb.len()
        ));
    }
    if let Some(i) = ba.iter().zip(&bb).position(|(x, y)| x != y) {
        return Err(format!(
            "seed[{id}]: same request+seed produced different pixels (first diff at byte {i}: {} vs {}, of {} bytes)",
            ba[i], bb[i], ba.len()
        ));
    }
    // A provider that *ignores* the seed would also pass the identical-twice check above, so verify a
    // DIFFERENT seed actually changes the output (F-085).
    let mut req_alt = base_request(profile);
    req_alt.seed = Some(profile.seed.wrapping_add(0x9E37_79B9));
    let c = g
        .generate(&req_alt, &mut |_| {})
        .map_err(|e| format!("seed[{id}]: alternate-seed generate() failed: {e}"))?;
    let bc = output_bytes(&c);
    if bc.len() == ba.len() && bc.iter().zip(&ba).all(|(x, y)| x == y) {
        return Err(format!(
            "seed[{id}]: a different seed produced byte-identical output ({} bytes) — the provider \
             appears to ignore the seed",
            ba.len()
        ));
    }
    Ok(())
}

/// **CFG-off render (sc-17418).** For a model that advertises `supports_guidance`, `guidance = 1.0`
/// — classifier-free guidance **off**, a single conditioned branch — must be either honestly
/// rejected by `validate` or actually served by `generate`. What it must never be is *accepted and
/// then fatal*, which is what sc-14195 was: candle SDXL passed `guidance = 1.0` through `validate`
/// and then died mid-denoise on `shape mismatch in matmul, lhs: [10, 4096, 64], rhs: [20, 64, 77]`,
/// because the conditioning stayed CFG-batched while the latent did not.
///
/// [`Profile`] never sets `guidance`, so before this check **no family on either backend had ever
/// run a CFG-off render under conformance** — which is precisely why sc-14195 stayed green in CI
/// for the whole time the lane was broken. The CFG on/off fork is hand-written per family, so this
/// is a regression guard against the next family repeating it, not a claim about today's tree.
///
/// Two things are asserted, and the second is the one with teeth:
///
/// 1. **Accepted ⇒ renders.** If `validate` takes it, `generate` must not fail. A model that cannot
///    serve CFG-off is expected to say so in `validate` (the honest-rejection stance the shared
///    validate-honesty check already rewards); an `Err` from `validate` skips the rest.
/// 2. **The negative prompt is inert at exactly 1.0.** Two renders that differ *only* in their
///    negative prompt must be byte-identical. This is not a stylistic claim — at `g = 1.0` the CFG
///    combine `uncond + g·(cond − uncond)` reduces algebraically to `cond`, so the unconditional
///    branch cannot influence the result under *any* correct implementation, whether the engine
///    folds to a single forward or literally runs the combine. An engine that narrowed its
///    conditioning to the **wrong row** would render the negative prompt instead of the prompt, and
///    would fail here — which a "does it still error?" check cannot catch, since narrowing to the
///    wrong row stops erroring just as well as narrowing to the right one.
///
/// Costs at most two `generate` calls, and none at all for a CFG-free model.
pub fn check_cfg_off_render(g: &dyn Generator, profile: &Profile) -> Result<(), String> {
    let d = g.descriptor();
    let id = d.id;
    // CFG-free families (distilled/guidance-embedded: chroma, krea, ltx, seedvr2, the turbo tiers)
    // advertise no guidance axis, so there is no CFG-off contract to hold them to.
    if !d.capabilities.supports_guidance {
        return Ok(());
    }
    let has_negative = d.capabilities.supports_negative_prompt;
    let cfg_off = |negative: Option<&str>| {
        let mut req = base_request(profile);
        req.guidance = Some(1.0);
        req.negative_prompt = negative.map(str::to_owned);
        req
    };

    // Probe with the first negative prompt when the model takes one, so the two-render comparison
    // below reuses this render rather than paying for a third.
    let first = cfg_off(has_negative.then_some(CFG_OFF_NEGATIVE_A));
    // An honest rejection is a legitimate stance; only silent acceptance obliges a working render.
    if g.validate(&first).is_err() {
        return Ok(());
    }
    let out_a = g.generate(&first, &mut |_| {}).map_err(|e| {
        format!(
            "cfg_off[{id}]: validate() accepted guidance = 1.0 but generate() failed: {e} — a model \
             that cannot serve the CFG-off single-branch path must reject it in validate() with an \
             actionable error, not fail mid-denoise (sc-14195). If this is a batch/shape error, the \
             conditioning is still CFG-batched while the latent is not."
        )
    })?;

    if !has_negative {
        return Ok(());
    }
    let second = cfg_off(Some(CFG_OFF_NEGATIVE_B));
    if g.validate(&second).is_err() {
        return Ok(());
    }
    let out_b = g.generate(&second, &mut |_| {}).map_err(|e| {
        format!("cfg_off[{id}]: generate() with a second negative prompt failed: {e}")
    })?;

    let (ba, bb) = (output_bytes(&out_a), output_bytes(&out_b));
    if ba.len() != bb.len() {
        return Err(format!(
            "cfg_off[{id}]: changing only the negative prompt at guidance = 1.0 changed the output \
             SIZE ({} vs {} bytes) — at CFG-off the negative branch must not be consumed at all",
            ba.len(),
            bb.len()
        ));
    }
    if let Some(i) = ba.iter().zip(&bb).position(|(x, y)| x != y) {
        return Err(format!(
            "cfg_off[{id}]: at guidance = 1.0 the negative prompt changed the output (first diff at \
             byte {i}: {} vs {}, of {} bytes). At g = 1.0 the CFG combine reduces to the conditional \
             prediction, so the unconditional/negative branch must be inert. The usual cause is a \
             CFG-off path that narrows its conditioning to the WRONG row — rendering the negative \
             prompt instead of the prompt (sc-14195).",
            ba[i],
            bb[i],
            ba.len()
        ));
    }
    Ok(())
}

/// The two negative prompts [`check_cfg_off_render`] swaps between. Deliberately describing very
/// different images, so an engine that wrongly consumes the negative branch produces a visibly
/// different render rather than two near-identical ones that might collide byte-for-byte.
const CFG_OFF_NEGATIVE_A: &str = "a red circle on a plain white background";
const CFG_OFF_NEGATIVE_B: &str = "a dense city street at night, neon signs, heavy rain";

/// **Registry round-trip.** The provider's descriptor `id` is present in the explicit registry
/// supplied by the caller (a missing catalog entry is the runtime "engine not found" trap).
pub fn check_registry_roundtrip(
    registry: &gen_core::ProviderRegistry,
    g: &dyn Generator,
) -> Result<(), String> {
    let id = g.descriptor().id;
    if registry
        .generators()
        .any(|registration| (registration.descriptor)().id == id)
    {
        Ok(())
    } else {
        Err(format!(
            "registry[{id}]: descriptor id not found in the explicit provider registry (gen-core {})",
            gen_core::VERSION
        ))
    }
}

/// **Descriptor sweep (weights-free).** Run the registry-wide descriptor-level conformance sweep
/// over the explicit catalog supplied by the caller and panic with the aggregated violations — the
/// test-helper idiom of [`conformance`], minus any model load. Because no weights are touched,
/// providers wire this as a default (non-`#[ignore]`d) test; behavioral checks stay weights-gated.
pub fn registry_conformance(registry: &gen_core::ProviderRegistry) {
    let errs = registry.descriptor_conformance_errors();
    if !errs.is_empty() {
        panic!(
            "gen-core descriptor conformance FAILED ({} violations, gen-core {}):\n  - {}",
            errs.len(),
            gen_core::VERSION,
            errs.join("\n  - ")
        );
    }
}

/// **Named-component load gate (sc-13658).** A model that declares
/// [`required_components`](gen_core::ModelDescriptor::required_components) must convert a missing or
/// unrecognized [`LoadSpec::components`](gen_core::LoadSpec::components) entry into a **load-time**
/// error — not a mid-render fetch (the perth class this seam exists to kill) and not a first-`generate`
/// failure. Given the provider's fallible `load` closure, a `base_spec` that stages every required
/// component (so `load` clears the gate), and the model's declared `required` ids, this asserts:
///
/// 1. **Missing required component → load error.** Removing any one required id from
///    `base_spec.components` makes `load` return `Err`, and the error names the missing id (so the
///    failure is provably the component gate — [`gen_core::require_component`] — not an unrelated
///    load error).
/// 2. **Unknown component key → load error.** Adding a component key the model does not declare makes
///    `load` return `Err` (the [`gen_core::reject_unknown_components`] typed-`Unsupported` guard).
///
/// It drives the provider's real `load`, so it exercises the actual validators rather than a mock —
/// a provider that forgets to call `require_component` (silently proceeding to a mid-render fetch) or
/// `reject_unknown_components` (silently ignoring a stray key) fails here. Only meaningful for a model
/// with a non-empty `required_components`; returns `Ok(())` when the gate holds.
pub fn check_component_load_gate(
    load: impl Fn(&gen_core::LoadSpec) -> gen_core::Result<Box<dyn Generator>>,
    base_spec: &gen_core::LoadSpec,
    required: &[&str],
) -> Result<(), String> {
    if required.is_empty() {
        return Err(
            "component-load-gate: `required` is empty — this check only applies to a model that \
             declares required_components"
                .to_string(),
        );
    }

    // 1. Each required component, removed in turn, must make load() fail — and fail *because of* that
    //    component (its id must appear in the error), not for some unrelated reason.
    for &missing in required {
        let mut spec = base_spec.clone();
        spec.components.remove(missing);
        match load(&spec) {
            Ok(_) => {
                return Err(format!(
                    "component-load-gate: load() succeeded with the required component '{missing}' \
                     missing — a declared required component must be a load-time error (call \
                     gen_core::require_component in load(), before any weight read or generate)"
                ));
            }
            Err(e) => {
                let msg = e.to_string();
                if !msg.contains(missing) {
                    return Err(format!(
                        "component-load-gate: load() failed with '{missing}' missing, but the error \
                         did not name the component ({msg:?}) — the failure must be the missing \
                         component gate, not an unrelated load error"
                    ));
                }
            }
        }
    }

    // 2. A component key the model does not declare must be rejected at load.
    let bogus = "__testkit_unknown_component__";
    let mut spec = base_spec.clone();
    spec.components.insert(
        bogus.to_owned(),
        gen_core::WeightsSource::File(std::path::PathBuf::from("/dev/null")),
    );
    match load(&spec) {
        Ok(_) => Err(format!(
            "component-load-gate: load() accepted an unrecognized component key '{bogus}' — an \
             unknown component key must be rejected at load (call \
             gen_core::reject_unknown_components in load())"
        )),
        Err(_) => Ok(()),
    }
}

/// Run the full conformance suite against a freshly-`make`d generator. Panics with every failure
/// aggregated (one bullet per failed guarantee) — the test-helper idiom, like a fat `assert`.
///
/// `make` is `Fn` so callers may hand it an explicit registry loader
/// (`|| registry.load(id, &spec).unwrap()`) or an in-crate stub; it is invoked once. The generator is shared across checks (`generate` is
/// `&self` and stateless across calls), so the whole suite is one model load.
pub fn conformance(make: impl Fn() -> Box<dyn Generator>, profile: &Profile) {
    let g = make();
    let g: &dyn Generator = g.as_ref();

    type Check = fn(&dyn Generator, &Profile) -> Result<(), String>;
    let checks: [Check; 7] = [
        check_validate_honesty,
        check_progress,
        check_progress_contract,
        check_cancellation,
        check_precancellation,
        check_seed_determinism,
        check_cfg_off_render,
    ];

    let failures: Vec<String> = checks
        .into_iter()
        .filter_map(|f| f(g, profile).err())
        .collect();
    if !failures.is_empty() {
        panic!(
            "gen-core conformance FAILED for `{}` ({} backend, gen-core {}):\n  - {}",
            g.descriptor().id,
            g.descriptor().backend,
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests;
