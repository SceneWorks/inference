//! Small header/descriptor fixtures driving the two production H3 admission contracts. No tensor
//! runtime, model values, network, or downloaded assets. Only descriptors have nonzero payloads.
use std::collections::BTreeMap;
use std::path::Path;

/// Return exact resident operand bytes (excluding shared backend buffers) and the unique S4 groups.
/// F32 heads remain F32, F16/BF16/F32 block companions follow the block compute width.
pub fn write_native_component(dir: &Path, block_width: u64) -> (u64, Vec<u64>) {
    write_native_component_with_outputs(dir, block_width, 32)
}

fn write_native_component_with_outputs(dir: &Path, block_width: u64, n: u64) -> (u64, Vec<u64>) {
    let mut rows: BTreeMap<String, (&str, Vec<u64>, Vec<u8>)> = BTreeMap::new();
    let mut resident = 0;
    for (i, group) in [64, 256, 64, 256].into_iter().enumerate() {
        let base = format!("transformer_blocks.{i}.attn.to_q");
        let payload =
            format!(r#"{{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":{group}}}"#)
                .into_bytes();
        rows.insert(format!("{base}.weight"), ("I8", vec![n, group * 2], vec![]));
        rows.insert(format!("{base}.weight_scale"), ("F32", vec![n, 1], vec![]));
        rows.insert(
            format!("{base}.comfy_quant"),
            ("U8", vec![payload.len() as u64], payload),
        );
        resident += n * group * 2 + 4 * n;
    }
    let base = "transformer_blocks.4.attn.to_q";
    let payload =
        br#"{"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256}"#.to_vec();
    rows.insert(format!("{base}.weight"), ("I8", vec![n, 256], vec![]));
    rows.insert(
        format!("{base}.weight_s_rel"),
        ("F8_E4M3", vec![n, 32], vec![]),
    );
    rows.insert(format!("{base}.weight_s_channel"), ("F32", vec![n], vec![]));
    rows.insert(format!("{base}.weight_codebook"), ("F32", vec![16], vec![]));
    rows.insert(
        format!("{base}.comfy_quant"),
        ("U8", vec![payload.len() as u64], payload),
    );
    resident += n * 512 / 2 + n * 512 / 16 + 4 * n + 64;
    for (i, dtype) in ["F16", "BF16", "F32"].into_iter().enumerate() {
        rows.insert(
            format!("transformer_blocks.{}.ff.net.2.weight", 5 + i),
            (dtype, vec![16, 16], vec![]),
        );
        resident += 16 * 16 * block_width;
    }
    rows.insert("proj_in.weight".into(), ("F32", vec![16, 16], vec![]));
    resident += 16 * 16 * 4;
    write_rows(dir, rows);
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "num_attention_heads":2,"attention_head_dim":128,"hidden_size":256,"num_layers":8,
            "num_refiner_layers":2,"ffn_dim":256,"in_channels":24,"audio_in_channels":32,
            "patch_size":[1,2,2],"text_dim":5120,"freq_dim":256,"time_embed_hidden_dim":256,
            "time_embed_dim":64,"rope_freq_dim":16,"rope_theta":10000.0,"norm_eps":1e-5,
            "qk_norm_eps":1e-5,"final_norm_eps":1e-5
        })
        .to_string(),
    )
    .unwrap();
    (resident, vec![64, 256])
}

fn write_rows(dir: &Path, rows: BTreeMap<String, (&str, Vec<u64>, Vec<u8>)>) {
    let mut headers = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, (dtype, shape, payload)) in rows {
        let width = match dtype {
            "F32" => 4,
            "F16" | "BF16" => 2,
            _ => 1,
        };
        let size = shape.iter().product::<u64>() * width;
        let start = data.len();
        data.resize(start + size as usize, 0);
        if !payload.is_empty() {
            data[start..].copy_from_slice(&payload);
        }
        headers.insert(
            name,
            serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&headers).unwrap();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

/// Tiny tensor inventories for constructor/readset accounting, not executable model fixtures.
pub fn write_float_operands(dir: &Path, names: impl IntoIterator<Item = String>, dtype: &str) {
    let rows = names
        .into_iter()
        .map(|name| {
            let shape = if name.ends_with(".weight_v") {
                vec![2, 2, 3]
            } else if name.ends_with(".weight_g") {
                vec![2, 1, 1]
            } else if name.ends_with(".filter") {
                vec![1, 1, 12]
            } else {
                vec![2]
            };
            (name, (dtype, shape, vec![]))
        })
        .collect();
    write_rows(dir, rows);
}

/// Explicit constructor configs; decoder depth intentionally differs from the stock preset.
pub fn write_vae_configs(root: &Path) {
    std::fs::create_dir_all(root.join("vae")).unwrap();
    std::fs::write(root.join("vae/config.json"), serde_json::json!({
        "latent_channels":24,"in_channels":3,"out_channels":3,"decoder_num_layers":3,
        "decoder_num_attention_heads":32,"decoder_attention_head_dim":64,"decoder_num_register_tokens":4,
        "decoder_ffn_mult":4,"decoder_rope_theta":100.0,"decoder_rope_dim_ratio":0.75,"decoder_norm_eps":1e-5,
        "clip_length":17,"token_drop":3,"latents_mean":vec![0.;24],"latents_std":vec![1.;24],
        "block_out_channels":[128,256,256,512,512,1024],"layers_per_block":2,
        "spatial_downsample_factors":[2,2,2,2,1,1],"temporal_downsample_factors":[1,2,2,1,1,1],
        "norm_num_groups":32,"norm_eps":1e-6
    }).to_string()).unwrap();
    let audio = root.join("FL2VA/audio_vae");
    std::fs::create_dir_all(&audio).unwrap();
    std::fs::write(audio.join("config.json"), serde_json::json!({
        "source_config_path":"config.yaml","source_metadata_path":"metadata.json","source_safetensors_path":"model.safetensors",
        "output_channel":2,"latents_mean":vec![0.;32],"latents_std":vec![1.;32]
    }).to_string()).unwrap();
    std::fs::write(
        audio.join("config.yaml"),
        "model_config:\n  sr: 32000\n  decoder_dim: 1024\n  vae_latent_channels: 32\n",
    )
    .unwrap();
    std::fs::write(
        audio.join("metadata.json"),
        serde_json::json!({"metadata":{"kwargs":{
            "sample_rate":32000,"encoder_rates":[2,4,4,5,5],"decoder_rates":[5,5,2,2,2,2,2],
            "decoder_type":"bigvgan","attn_proj":true
        }}})
        .to_string(),
    )
    .unwrap();
}

/// Dense-only imports must not lose the source-grounded planner simply because they have no
/// integer descriptor. The native loaders preserve F32 heads and cast only the dense block stack.
pub fn assert_dense_contract(
    root: &Path,
    contract_for: impl Fn(&gen_core::LoadSpec) -> gen_core::Result<gen_core::MemoryProviderContract>,
    mlx: bool,
) {
    let dir = root.join("transformer");
    write_native_component(&dir, 2);
    write_float_operands(
        &dir,
        [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
        ]
        .map(str::to_owned),
        "F32",
    );
    let mut spec = gen_core::LoadSpec::new(gen_core::WeightsSource::Dir(root.into()));
    let contract = contract_for(&spec).unwrap();
    assert_eq!(contract.asset_facts.transformer_bytes, 3 * 2 * 2);
    let native = contract
        .phase_facts
        .as_ref()
        .unwrap()
        .native_quant
        .as_ref()
        .unwrap();
    assert!(native.tasks.iter().all(|t| t.projections.is_empty()));
    assert_eq!(
        contract.evicted_component_bytes(),
        0,
        "the supplied source contains no AdaLN allocation to evict"
    );
    if mlx {
        spec.precision = gen_core::Precision::Fp32;
        assert_eq!(
            contract_for(&spec).unwrap().asset_facts.transformer_bytes,
            3 * 2 * 4
        );
    }
}

/// Packed-layout bounds must include flattened stereo rows exactly once. The provider supplies
/// its real cost helper's volume, not a re-derived oracle from the admission implementation.
pub fn assert_stereo_packing_bound(actual_target_rows: u64) {
    let mut req = gen_core::GenerationRequest {
        width: 1344,
        height: 768,
        frames: Some(345),
        prompt: String::new(),
        ..Default::default()
    };
    let base = gen_core::h3_memory::request_rows((&req).into(), 4).unwrap();
    assert!(
        base.packed >= actual_target_rows,
        "target packed bound must cover the native stereo volume"
    );
    let rate = 8000u32;
    let samples_per_channel = rate as usize * 60;
    req.conditioning
        .push(gen_core::Conditioning::ReferenceAudio {
            audio: gen_core::AudioTrack {
                samples: vec![0.; samples_per_channel * 2],
                sample_rate: rate,
                channels: 2,
                stems: vec![],
            },
            strength: None,
        });
    let referenced = gen_core::h3_memory::request_rows((&req).into(), 4).unwrap();
    assert_eq!(
        referenced.packed - base.packed,
        60 * 40 * u64::from(gen_core::h3_memory::AUDIO_OUTPUT_CHANNELS),
        "per-channel duration becomes channel-major rows once, including a long reference clip"
    );
}

/// Both providers' real contract constructors must reflect actual selected native source data.
pub fn assert_native_contract(
    root: &Path,
    staged: &Path,
    contract_for: impl Fn(&gen_core::LoadSpec) -> gen_core::Result<gen_core::MemoryProviderContract>,
    mlx: bool,
) {
    use gen_core::{LoadSpec, Precision, WeightsSource};
    let (resident, groups) = write_native_component(staged, 2);
    // An upstream component is deliberately larger. A redirected task may only use the staged
    // task's sibling, never this upstream reference partition.
    write_native_component_with_outputs(&root.join("transformer_ref"), 4, 64);
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
    spec.components.insert(
        "transformer".into(),
        WeightsSource::Dir(staged.to_path_buf()),
    );
    let contract = contract_for(&spec).unwrap();
    let shared = groups.iter().map(|g| g * g * 4).sum::<u64>() + if mlx { 0 } else { 32 << 20 };
    assert_eq!(contract.asset_facts.transformer_bytes, resident + shared,
        "native source payloads, original scales, descriptor exclusion and unique group/context ownership");
    let facts = contract
        .phase_facts
        .as_ref()
        .unwrap()
        .native_quant
        .as_ref()
        .unwrap();
    assert_eq!(
        facts.tasks.iter().filter(|t| t.resident_bytes > 0).count(),
        1,
        "the staged task's absent sibling is not the root's dense reference partition"
    );
    assert_eq!(
        facts
            .tasks
            .iter()
            .map(|t| t.projections.len())
            .sum::<usize>(),
        5
    );
    assert!(
        facts.tasks.iter().all(|t| t.evicted_bytes == 0),
        "no imaginary dense AdaLN allocation in this source"
    );
    let rows = gen_core::h3_memory::NativeRows {
        packed: 17,
        conditioning: 9,
        schedule: 5,
        visual_elements: 0,
        global: 0,
    };
    assert!(facts.denoise_peak_bytes(rows) > contract.asset_facts.transformer_bytes);
    assert!(
        facts.denoise_peak_bytes(gen_core::h3_memory::NativeRows { packed: 34, ..rows })
            > facts.denoise_peak_bytes(rows)
    );
    if mlx {
        spec.precision = Precision::Fp32;
        let wide = contract_for(&spec).unwrap();
        assert_eq!(
            wide.asset_facts.transformer_bytes - contract.asset_facts.transformer_bytes,
            3 * 16 * 16 * 2,
            "dense blocks widen, native I8/W4A8 operands and original F32 heads do not"
        );
    }
}
