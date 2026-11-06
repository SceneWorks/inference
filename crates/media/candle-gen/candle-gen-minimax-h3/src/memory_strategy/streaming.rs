//! Header-only runtime-format residency and CPU preparation bounds for the CUDA loader.
use super::*;
use candle_gen::gen_core::{weightsmeta, StreamedWeightFacts};
use std::{collections::BTreeMap, path::Path};

fn add(a: u64, b: u64) -> candle_gen::gen_core::Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| CoreError::Unsupported("H3 streaming allocation overflow".into()))
}

pub(super) fn dit_window(
    path: &Path,
    allocation: &gen_core_h3::h3_memory::NativeComponentAllocation,
    bytes: u64,
    pruned: bool,
) -> candle_gen::gen_core::Result<Option<candle_gen::gen_core::h3_memory::window::WindowAllocation>>
{
    if !allocation.resolved
        || !crate::block_stream::component_supported(path)
            .map_err(|e| CoreError::Unsupported(e.to_string()))?
    {
        return Ok(None);
    }
    let cfg = crate::dit::MiniMaxH3DitConfig::from_diffusers_json(&std::fs::read_to_string(
        path.join("config.json"),
    )?)
    .map_err(|e| CoreError::Unsupported(e.to_string()))?;
    let (_, _, decode) = runtime_payloads(path, false)?;
    let facts = candle_gen::gen_core::h3_memory::window::CudaWorkingSet {
        hidden: cfg.hidden_size as u64,
        heads: cfg.num_attention_heads as u64,
        head_dim: cfg.attention_head_dim as u64,
        rotary_dim: cfg.rotary_dim() as u64,
        ffn_inner: cfg.ffn_dim as u64,
        element_bytes: 2,
        projection_decode_bytes: decode,
    };
    let mut window =
        candle_gen::gen_core::h3_memory::window::WindowAllocation::from_resident_tensors(
            &allocation.resident_by_name,
            &allocation.projections,
            bytes.saturating_sub(allocation.resident_bytes),
            pruned,
        )?
        .with_cuda_working_set(facts)?;
    if window.n_blocks() != cfg.num_layers {
        // A partial inventory can still price its existing resident contract. It cannot
        // advertise streaming without the complete contiguous configured block stack.
        return Ok(None);
    }
    if !pruned {
        window = window.with_cuda_adaln_working_sets(cuda_adaln_working_sets(path, &cfg)?)?;
    }
    Ok(Some(window))
}

fn cuda_adaln_working_sets(
    path: &Path,
    cfg: &crate::dit::MiniMaxH3DitConfig,
) -> candle_gen::gen_core::Result<Vec<candle_gen::gen_core::h3_memory::window::CudaAdalnWorkingSet>>
{
    let invalid =
        || CoreError::Unsupported("invalid or missing CUDA AdaLN projection facts".into());
    let headers = weightsmeta::safetensors_path_tensor_headers(path)?;
    let names = headers
        .iter()
        .map(|h| (h.name.as_str(), h))
        .collect::<BTreeMap<_, _>>();
    let output = (cfg.hidden_size as u64)
        .checked_mul(18)
        .ok_or_else(invalid)?;
    let output_dim = usize::try_from(output).map_err(|_| invalid())?;
    let input = cfg.time_embed_dim as u64;
    if output == 0 || input == 0 || cfg.num_layers == 0 {
        return Err(invalid());
    }
    (0..cfg.num_layers)
        .map(|layer| {
            let base = format!("transformer_blocks.{layer}.adaln_proj.linear");
            let weight = names
                .get(format!("{base}.weight").as_str())
                .ok_or_else(invalid)?;
            let bias = names
                .get(format!("{base}.bias").as_str())
                .ok_or_else(invalid)?;
            // Linear broadcasts its bias, and AdaLN subsequently reshapes the output.
            // Preserve scalar/singleton-prefixed and row-dependent accepted layouts, not
            // just canonical [N]. Leading non-singletons would change output volume.
            let bias_columns = bias.shape.last().copied().unwrap_or(1);
            let bias_rows = bias.shape.iter().rev().nth(1).copied().unwrap_or(1) as u64;
            if !bias.is_float()
                || bias_rows == 0
                || bias_columns != 1 && bias_columns != output_dim
                || bias.shape.iter().rev().skip(2).any(|&n| n != 1)
            {
                return Err(invalid());
            }
            let affine = if let Some(scales) = names.get(format!("{base}.scales").as_str()) {
                let biases = names
                    .get(format!("{base}.biases").as_str())
                    .ok_or_else(invalid)?;
                candle_gen::quant::mlx_packed_qtensor_resident_bytes(
                    weight,
                    scales,
                    biases,
                    candle_gen::quant::MLX_GROUP_SIZE,
                )?;
                if scales.shape.as_slice() != [output_dim, (input / 64) as usize]
                    || !input.is_multiple_of(64)
                {
                    return Err(invalid());
                }
                true
            } else {
                if !weight.is_float() || weight.shape.as_slice() != [output_dim, input as usize] {
                    return Err(invalid());
                }
                false
            };
            Ok(
                candle_gen::gen_core::h3_memory::window::CudaAdalnWorkingSet {
                    input,
                    output,
                    affine,
                    bias_rows,
                },
            )
        })
        .collect()
}

/// Runtime payloads collapse affine triples into the exact GGUF container used by CUDA.
/// Returns payload-by-source-name, host constructor peak and the largest root/body decode peak.
/// Full AdaLN decode is priced separately at its current precompute projection, not in denoise.
fn runtime_payloads(
    path: &Path,
    text_encoder: bool,
) -> candle_gen::gen_core::Result<(BTreeMap<String, u64>, u64, u64)> {
    let headers = weightsmeta::safetensors_path_tensor_headers(path)?;
    if headers.is_empty() {
        return Err(CoreError::Unsupported(format!(
            "H3 allocation source is empty: {}",
            path.display()
        )));
    }
    let names = headers
        .iter()
        .map(|h| (h.name.as_str(), h))
        .collect::<BTreeMap<_, _>>();
    let mut payload = BTreeMap::new();
    let mut source = 0;
    let mut derived = 0;
    let mut decode = 0;
    let mut prepare_scratch = 0;
    for h in &headers {
        source = add(source, h.data_bytes)?;
        if h.name.ends_with(".scales") || h.name.ends_with(".biases") {
            continue;
        }
        if let Some(base) = h.name.strip_suffix(".weight") {
            if let Some(scales) = names.get(format!("{base}.scales").as_str()) {
                let biases = names
                    .get(format!("{base}.biases").as_str())
                    .ok_or_else(|| {
                        CoreError::Unsupported("H3 affine bias operand missing".into())
                    })?;
                let packed = candle_gen::quant::mlx_packed_qtensor_resident_bytes(
                    h,
                    scales,
                    biases,
                    candle_gen::quant::MLX_GROUP_SIZE,
                )?;
                // TieredEmbedding retains its host triple; every projection prepares a GGUF base.
                if text_encoder && base.ends_with("embed_tokens") {
                    payload.insert(
                        h.name.clone(),
                        add(add(h.data_bytes, scales.data_bytes)?, biases.data_bytes)?,
                    );
                } else {
                    derived = add(derived, packed)?;
                    payload.insert(h.name.clone(), packed);
                    let n = *h
                        .shape
                        .first()
                        .ok_or_else(|| CoreError::Unsupported("invalid affine weight".into()))?
                        as u64;
                    let groups = *scales
                        .shape
                        .get(1)
                        .ok_or_else(|| CoreError::Unsupported("invalid affine scales".into()))?
                        as u64;
                    let elements = n
                        .checked_mul(groups)
                        .and_then(|v| v.checked_mul(candle_gen::quant::MLX_GROUP_SIZE as u64))
                        .ok_or_else(|| {
                            CoreError::Unsupported("H3 affine decode size overflow".into())
                        })?;
                    // Q8 CPU repacking may transiently decode one full F32 weight. The
                    // source map and completed runtime packs are already counted below.
                    prepare_scratch =
                        prepare_scratch.max(elements.checked_mul(4).ok_or_else(|| {
                            CoreError::Unsupported("H3 affine preparation overflow".into())
                        })?);
                    if !h.name.contains(".adaln_proj.linear.") {
                        // QLinear first owns F32 dequantization plus the BF16 target cast;
                        // the TE's F32 compute aliases its already-F32 dequantized weight.
                        decode = decode.max(
                            elements
                                .checked_mul(if text_encoder { 4 } else { 6 })
                                .ok_or_else(|| {
                                    CoreError::Unsupported("H3 affine decode size overflow".into())
                                })?,
                        );
                    }
                }
                continue;
            }
        }
        let norm =
            text_encoder && (h.name.contains("layernorm") || h.name.contains("_norm.weight"));
        let width = if norm { 4 } else { 2 };
        let bytes = if h.is_float() {
            h.materialized_bytes(width)?
        } else {
            h.data_bytes
        };
        if bytes != h.data_bytes {
            derived = add(derived, bytes)?;
        }
        if h.shape.len() == 2
            && h.is_float()
            && !h.name.contains("embed_tokens")
            && !h.name.contains(".adaln_proj.linear.")
        {
            decode = decode.max(h.materialized_bytes(if text_encoder { 4 } else { 2 })?);
        }
        payload.insert(h.name.clone(), bytes);
    }
    // Repacking can decode one complete current projection while source and earlier runtime
    // packs remain live. It never expands every block or repeats conversion during a window.
    Ok((
        payload,
        add(add(source, derived)?, prepare_scratch)?,
        decode,
    ))
}

#[cfg(all(test, feature = "cuda", feature = "model-inference-proof"))]
pub(crate) fn proof_text_encoder_inventory(
    path: &Path,
) -> candle_gen::gen_core::Result<(StreamedWeightFacts, u64)> {
    let facts = text_encoder_stream(path)?.ok_or_else(|| {
        CoreError::Unsupported("missing source-owned TE inventory for diagnostic".into())
    })?;
    let (_, _, decode) = runtime_payloads(path, true)?;
    Ok((facts, decode))
}

pub(super) fn text_encoder_stream(
    path: &Path,
) -> candle_gen::gen_core::Result<Option<StreamedWeightFacts>> {
    if weightsmeta::safetensors_path_bytes(path) == 0 {
        return Ok(None);
    }
    let cfg = crate::text_encoder::MiniMaxH3TeConfig::from_config_json(&std::fs::read_to_string(
        path.join("config.json"),
    )?)
    .map_err(|e| CoreError::Unsupported(e.to_string()))?;
    let mut blocks = vec![0; cfg.layers_to_run()];
    // The dense/packed token embedding remains HOST-resident. Only selected rows upload.
    // Rotary cos/sin are f32 and the grounded vision trunk is an independent resident phase.
    let mut root = (2u64 * 16384)
        .checked_mul(cfg.head_dim as u64)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| CoreError::Unsupported("H3 rotary allocation overflow".into()))?;
    let prefix = format!("{}.layers.", crate::text_encoder::LM_PREFIX);
    let (payload, _, _) = runtime_payloads(path, true)?;
    for (name, bytes) in payload {
        if let Some(tail) = name.strip_prefix(&prefix) {
            let (index, _) = tail
                .split_once('.')
                .ok_or_else(|| CoreError::Unsupported("invalid H3 text layer".into()))?;
            let index = index
                .parse::<usize>()
                .map_err(|_| CoreError::Unsupported("invalid H3 text layer".into()))?;
            if let Some(block) = blocks.get_mut(index) {
                *block = add(*block, bytes)?;
            }
        } else if name.starts_with(crate::text_encoder::VISION_PREFIX) {
            root = add(root, bytes)?;
        }
    }
    if blocks.is_empty() || blocks.contains(&0) {
        return Err(CoreError::Unsupported(
            "H3 text source omits a consumed layer".into(),
        ));
    }
    Ok(Some(StreamedWeightFacts {
        resident_bytes: root,
        stacks: vec![blocks],
    }))
}

/// Host RAM for source preparation and the retained runtime-format stack. Successive TE/DiT
/// components take a maximum. Veda host predictor/scorer overlap the prepared DiT, not TE load.
pub fn streaming_host_peak_bytes(
    spec: &LoadSpec,
    req: candle_gen::gen_core::veda::workspace::Request<'_>,
) -> candle_gen::gen_core::Result<u64> {
    host_peak_bytes(spec, req, true)
}

/// Both routes prepare host-packed text weights; only streaming retains the host DiT stack
/// throughout denoising. Native resident representations are priced from their actual headers.
pub fn host_peak_bytes(
    spec: &LoadSpec,
    req: candle_gen::gen_core::veda::workspace::Request<'_>,
    streamed: bool,
) -> candle_gen::gen_core::Result<u64> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(p) => p.parent().unwrap_or(p).to_path_buf(),
    };
    let staged = spec
        .components
        .get(BASE_DIT_PARTITION)
        .and_then(|w| match w {
            WeightsSource::Dir(p) => Some(p.as_path()),
            _ => None,
        });
    let te_override = spec
        .components
        .get(crate::tier::TEXT_ENCODER_COMPONENT)
        .and_then(|w| match w {
            WeightsSource::Dir(p) => Some(p.as_path()),
            _ => None,
        });
    let paths = crate::tier::MiniMaxH3TierPaths::resolve(&root, staged, te_override);
    let te = te_override.map_or_else(
        || root.join(crate::tier::TEXT_ENCODER_COMPONENT),
        Path::to_path_buf,
    );
    let (_, te_prepare, _) = runtime_payloads(&te, true)?;
    let reference = req.conditioning.iter().any(|c| {
        matches!(
            c,
            candle_gen::gen_core::Conditioning::Reference { .. }
                | candle_gen::gen_core::Conditioning::MultiReference { .. }
                | candle_gen::gen_core::Conditioning::ReferenceVideo { .. }
                | candle_gen::gen_core::Conditioning::ReferenceAudio { .. }
        )
    });
    let dit = if reference {
        &paths.reference_dit_dir
    } else {
        &paths.dit_dir
    };
    let (payload, prepare, _) = runtime_payloads(dit, false)?;
    let retained = payload.values().try_fold(0u64, |n, &v| add(n, v))?;
    let veda = gen_core_h3::veda::workspace::for_video_request(
        req,
        candle_gen::gen_core::veda::workspace::Backend::Cuda,
    )?
    .map_or(0, |bound| bound.host_peak_bytes());
    let adapters = if spec.adapters.is_empty() {
        0
    } else {
        let allocation = super::dit_allocation(dit, false)?;
        let profile = super::adapters::profile(dit, &spec.adapters, &allocation.projections)?;
        // Includes installed originals/prepared caches and source-file install transients;
        // file length alone misses F32 factors and the first-forward prepared representation.
        profile.install_peak_bytes.max(profile.resident_bytes)
    };
    let media = u64::from(req.width)
        .checked_mul(u64::from(req.height))
        .and_then(|v| v.checked_mul(u64::from(req.frames)))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| CoreError::Unsupported("H3 host media size overflow".into()))?;
    add(
        add(
            te_prepare
                .max(prepare)
                .max(if streamed { add(retained, veda)? } else { veda }),
            adapters,
        )?,
        media,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn headers(path: &Path, entries: &[(String, &str, Vec<usize>)]) {
        use std::io::Write;
        let mut header = serde_json::Map::new();
        let mut offset = 0;
        for (name, dtype, shape) in entries {
            let bytes = shape.iter().product::<usize>() * if *dtype == "U32" { 4 } else { 2 };
            header.insert(name.clone(), serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,offset+bytes]}));
            offset += bytes;
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&json).unwrap();
        file.set_len((8 + json.len() + offset) as u64).unwrap();
    }

    #[test]
    fn cuda_adaln_headers_separate_precompute_from_body_decode_and_refuse_unknown() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("model.safetensors");
        let cfg = crate::dit::MiniMaxH3DitConfig {
            hidden_size: 64,
            time_embed_dim: 64,
            num_layers: 2,
            ..Default::default()
        };
        for bits in [None, Some(4), Some(8)] {
            let mut entries = Vec::new();
            for (base, output) in [
                ("transformer_blocks.0.adaln_proj.linear", 1152),
                ("transformer_blocks.1.adaln_proj.linear", 1152),
                ("transformer_blocks.0.ff.net.0.proj", 128),
            ] {
                if let Some(bits) = bits {
                    entries.push((
                        format!("{base}.weight"),
                        "U32",
                        vec![output, 64 * bits / 32],
                    ));
                    entries.push((format!("{base}.scales"), "BF16", vec![output, 1]));
                    entries.push((format!("{base}.biases"), "BF16", vec![output, 1]));
                } else {
                    entries.push((format!("{base}.weight"), "BF16", vec![output, 64]));
                }
                entries.push((format!("{base}.bias"), "BF16", vec![output]));
            }
            headers(&file, &entries);
            let facts = cuda_adaln_working_sets(root.path(), &cfg).unwrap();
            assert_eq!(
                facts,
                vec![
                    candle_gen::gen_core::h3_memory::window::CudaAdalnWorkingSet {
                        input: 64,
                        output: 1152,
                        affine: bits.is_some(),
                        bias_rows: 1,
                    };
                    2
                ]
            );
            let (payload, prepare, body_decode) = runtime_payloads(root.path(), false).unwrap();
            assert_eq!(
                body_decode,
                128 * 64 * if bits.is_some() { 6 } else { 2 },
                "AdaLN scratch belongs to precompute; body affine dequant owns F32+BF16"
            );
            if bits.is_some() {
                assert!(prepare >= payload.values().sum::<u64>() + 4 * 1152 * 64);
            }
            for (shape, rows) in [
                (vec![], 1),
                (vec![1], 1),
                (vec![1, 1152], 1),
                (vec![1, 1, 1152], 1),
                (vec![3, 1152], 3),
                (vec![3, 1], 3),
            ] {
                let mut broadcast = entries.clone();
                broadcast
                    .iter_mut()
                    .find(|(name, _, _)| name == "transformer_blocks.0.adaln_proj.linear.bias")
                    .unwrap()
                    .2 = shape;
                headers(&file, &broadcast);
                assert_eq!(
                    cuda_adaln_working_sets(root.path(), &cfg).unwrap()[0].bias_rows,
                    rows
                );
            }
            headers(&file, &entries);
            assert!(crate::block_stream::component_supported(root.path()).unwrap());
            let mut unknown = cfg.clone();
            unknown.time_embed_dim = 0;
            assert!(cuda_adaln_working_sets(root.path(), &unknown).is_err());
            unknown = cfg.clone();
            unknown.hidden_size = usize::MAX;
            assert!(cuda_adaln_working_sets(root.path(), &unknown).is_err());
            let mut partial = entries.clone();
            partial.retain(|(name, _, _)| name != "transformer_blocks.1.adaln_proj.linear.bias");
            headers(&file, &partial);
            assert!(
                cuda_adaln_working_sets(root.path(), &cfg).is_err(),
                "missing consumed projection operand is not zero scratch"
            );
            // Specialized native migration remains unavailable instead of being treated as affine.
            entries.push((
                "transformer_blocks.0.adaln_proj.linear.weight_scale".into(),
                "BF16",
                vec![1152, 1],
            ));
            headers(&file, &entries);
            assert!(!crate::block_stream::component_supported(root.path()).unwrap());
        }
    }

    #[test]
    fn header_inventory_prices_cuda_text_windows_and_host_embedding_separately() {
        let root = tempfile::tempdir().unwrap();
        let te = root.path().join("text_encoder");
        let mut names = (0..50)
            .map(|i| format!("model.language_model.layers.{i}.self_attn.q_proj.weight"))
            .collect::<Vec<_>>();
        names.push("model.language_model.embed_tokens.weight".into());
        gen_core_testkit::h3_memory::write_float_operands(&te, names, "BF16");
        std::fs::write(te.join("config.json"), "{}").unwrap();
        let facts = text_encoder_stream(&te).unwrap().unwrap();
        assert_eq!(facts.stacks[0], vec![4; 50]);
        assert_eq!(facts.peak_bytes(None) - facts.peak_bytes(Some(1)), 49 * 4);
        let (payload, prepare, _) = runtime_payloads(&te, true).unwrap();
        assert_eq!(payload["model.language_model.embed_tokens.weight"], 4);
        assert!(prepare >= payload.values().sum::<u64>());
        let dit = root.path().join(BASE_DIT_PARTITION);
        gen_core_testkit::h3_memory::write_native_component(&dit, 2);
        gen_core_testkit::h3_memory::write_float_operands(
            &dit,
            (0..8).map(|i| format!("transformer_blocks.{i}.attn.to_q.weight")),
            "BF16",
        );
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().into()));
        let request = candle_gen::gen_core::GenerationRequest {
            width: 1344,
            height: 768,
            frames: Some(345),
            ..Default::default()
        };
        let resident = host_peak_bytes(&spec, (&request).into(), false).unwrap();
        let streamed = streaming_host_peak_bytes(&spec, (&request).into()).unwrap();
        assert!(resident > 0 && streamed >= resident);
        std::fs::remove_file(te.join("model.safetensors")).unwrap();
        assert!(
            streaming_host_peak_bytes(&spec, (&request).into()).is_err(),
            "missing source cannot become zero RAM"
        );
    }
}
