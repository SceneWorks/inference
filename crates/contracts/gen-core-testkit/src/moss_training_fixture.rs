//! Run-private, synthetic full-pipeline MOSS fixture data. No tensor library, downloads or
//! published-checkpoint identity: these files are for explicit backend test-support factories only.
use gen_core::{Error, Result, TrainingItem};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}
type Tensors = BTreeMap<String, HostTensor>;
fn tensor(map: &mut Tensors, name: &str, shape: &[usize]) {
    let n: usize = shape.iter().product();
    let norm = shape.len() == 1
        && (name.ends_with("norm.weight")
            || name.contains("norm1.weight")
            || name.contains("norm2.weight")
            || name.contains("norm_out.weight")
            || name.ends_with("q_norm.weight")
            || name.ends_with("k_norm.weight"));
    let values = (0..n)
        .map(|i| {
            if norm || name == "bn.running_var" {
                1.
            } else if name.ends_with(".bias") || name == "bn.running_mean" {
                0.
            } else {
                0.08 * ((i % 127) as f32 * 0.173 + 0.3).sin()
            }
        })
        .collect();
    assert!(
        map.insert(
            name.into(),
            HostTensor {
                shape: shape.to_vec(),
                values
            }
        )
        .is_none(),
        "duplicate fixture tensor"
    );
}
fn parameter(map: &mut Tensors, name: &str, shape: &[usize], bias: usize) {
    tensor(map, &format!("{name}.weight"), shape);
    tensor(map, &format!("{name}.bias"), &[bias]);
}
fn resnet(map: &mut Tensors, name: &str) {
    parameter(map, &format!("{name}.norm1"), &[32], 32);
    parameter(map, &format!("{name}.conv1"), &[32, 32, 3, 3], 32);
    parameter(map, &format!("{name}.norm2"), &[32], 32);
    parameter(map, &format!("{name}.conv2"), &[32, 32, 3, 3], 32);
}
fn vae() -> Tensors {
    let mut map = Tensors::new();
    parameter(&mut map, "encoder.conv_in", &[32, 3, 3, 3], 32);
    for block in 0..4 {
        for layer in 0..2 {
            resnet(
                &mut map,
                &format!("encoder.down_blocks.{block}.resnets.{layer}"),
            );
        }
        if block < 3 {
            parameter(
                &mut map,
                &format!("encoder.down_blocks.{block}.downsamplers.0.conv"),
                &[32, 32, 3, 3],
                32,
            );
        }
    }
    for side in ["encoder", "decoder"] {
        for layer in 0..2 {
            resnet(&mut map, &format!("{side}.mid_block.resnets.{layer}"));
        }
        let prefix = format!("{side}.mid_block.attentions.0");
        parameter(&mut map, &format!("{prefix}.group_norm"), &[32], 32);
        for projection in ["to_q", "to_k", "to_v", "to_out.0"] {
            parameter(&mut map, &format!("{prefix}.{projection}"), &[32, 32], 32);
        }
    }
    parameter(&mut map, "encoder.conv_norm_out", &[32], 32);
    parameter(&mut map, "encoder.conv_out", &[64, 32, 3, 3], 64);
    parameter(&mut map, "quant_conv", &[64, 64, 1, 1], 64);
    parameter(&mut map, "post_quant_conv", &[32, 32, 1, 1], 32);
    parameter(&mut map, "decoder.conv_in", &[32, 32, 3, 3], 32);
    for block in 0..4 {
        for layer in 0..3 {
            resnet(
                &mut map,
                &format!("decoder.up_blocks.{block}.resnets.{layer}"),
            );
        }
        if block < 3 {
            parameter(
                &mut map,
                &format!("decoder.up_blocks.{block}.upsamplers.0.conv"),
                &[32, 32, 3, 3],
                32,
            );
        }
    }
    parameter(&mut map, "decoder.conv_norm_out", &[32], 32);
    parameter(&mut map, "decoder.conv_out", &[3, 32, 3, 3], 3);
    tensor(&mut map, "bn.running_mean", &[128]);
    tensor(&mut map, "bn.running_var", &[128]);
    map
}
fn text() -> Tensors {
    let mut map = Tensors::new();
    tensor(&mut map, "language_model.embed_tokens.weight", &[32, 32]);
    tensor(&mut map, "language_model.norm.weight", &[32]);
    let prefix = "language_model.layers.0";
    for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
        tensor(&mut map, &format!("{prefix}.{name}"), &[32]);
    }
    for (name, shape) in [
        ("mlp.gate_proj.weight", vec![64, 32]),
        ("mlp.up_proj.weight", vec![64, 32]),
        ("mlp.down_proj.weight", vec![32, 64]),
        ("self_attn.q_proj.weight", vec![64, 32]),
        ("self_attn.k_proj.weight", vec![16, 32]),
        ("self_attn.v_proj.weight", vec![16, 32]),
        ("self_attn.o_proj.weight", vec![32, 32]),
        ("self_attn.q_norm.weight", vec![8]),
        ("self_attn.k_norm.weight", vec![8]),
    ] {
        tensor(&mut map, &format!("{prefix}.{name}"), &shape);
    }
    map
}
fn transformer() -> Result<Tensors> {
    // Reuse the immutable author's SMALL tensor inventory, not its checkpoint/provenance claim.
    // The new synthetic profile expands only the real VAE and TE interface dimensions.
    let author: Value = serde_json::from_str(include_str!(
        "../../gen-core/tests/fixtures/moss_image/tiny_author/fixture.json"
    ))
    .map_err(|e| Error::Msg(e.to_string()))?;
    let mut map = Tensors::new();
    for (name, value) in author["weights"]
        .as_object()
        .ok_or_else(|| Error::Msg("fixture inventory missing".into()))?
    {
        let mut shape = value["shape"]
            .as_array()
            .ok_or_else(|| Error::Msg("fixture shape missing".into()))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|v| v as usize)
                    .ok_or_else(|| Error::Msg("invalid fixture dimension".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        match name.as_str() {
            "video_embedder.weight" => shape = vec![16, 128],
            "text_embedder.proj.weight" => shape = vec![16, 32],
            "final_linear_video.weight" => shape = vec![128, 16],
            _ => (),
        }
        if [
            "video_embedder.weight",
            "text_embedder.proj.weight",
            "final_linear_video.weight",
        ]
        .contains(&name.as_str())
        {
            tensor(&mut map, name, &shape);
        } else {
            let values = value["values"]
                .as_array()
                .ok_or_else(|| Error::Msg("fixture values missing".into()))?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|v| v as f32)
                        .ok_or_else(|| Error::Msg("invalid fixture value".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            map.insert(name.clone(), HostTensor { shape, values });
        }
    }
    Ok(map)
}
fn bf16_bytes(value: f32) -> Result<[u8; 2]> {
    if !value.is_finite() {
        return Err(Error::Msg("nonfinite synthetic BF16 source".into()));
    }
    // IEEE round-to-nearest, ties-to-even (the shared gen-core helper). The ordinary source ABI is
    // BF16 even when the selected trainer later casts the frozen backbone to FP32 compute.
    let rounded = gen_core::numeric::bf16_bits(value);
    if rounded & 0x7f80 == 0x7f80 {
        return Err(Error::Msg("synthetic BF16 source overflow".into()));
    }
    Ok(rounded.to_le_bytes())
}
fn write_tensors(path: &Path, tensors: Tensors, transformer: bool) -> Result<u64> {
    let mut header = BTreeMap::<String, Value>::new();
    let mut bytes = Vec::new();
    header.insert("__metadata__".into(),json!({"fixture_identity":"sceneworks-synthetic-moss-full-pipeline-v1","not_published_weights":"true"}));
    for (name, tensor) in &tensors {
        let n = tensor
            .shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| Error::Msg("fixture shape overflow".into()))?;
        if n != tensor.values.len() || tensor.values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Msg("invalid fixture shape/value population".into()));
        }
        let start = bytes.len();
        let bf16 = transformer && name != "rope.bands";
        for value in &tensor.values {
            if bf16 {
                bytes.extend(bf16_bytes(*value)?);
            } else {
                bytes.extend(value.to_le_bytes());
            }
        }
        header.insert(
            name.clone(),
            json!({"dtype":if bf16 { "BF16" } else { "F32" },"shape":tensor.shape,"data_offsets":[start,bytes.len()]}),
        );
    }
    let mut json = serde_json::to_vec(&header).map_err(|e| Error::Msg(e.to_string()))?;
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut file = Vec::with_capacity(8 + json.len() + bytes.len());
    file.extend((json.len() as u64).to_le_bytes());
    file.extend(json);
    file.extend(bytes);
    fs::create_dir_all(
        path.parent()
            .ok_or_else(|| Error::Msg("missing fixture parent".into()))?,
    )?;
    fs::write(path, &file)?;
    let actual = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    if actual.len() != tensors.len()
        || actual.iter().any(|h| {
            let expected_dtype = if transformer && h.name != "rope.bands" {
                gen_core::weightsmeta::Dtype::BF16
            } else {
                gen_core::weightsmeta::Dtype::F32
            };
            tensors.get(&h.name).is_none_or(|t| h.shape != t.shape) || h.dtype != expected_dtype
        })
    {
        return Err(Error::Msg(
            "written fixture header differs from generated tensor schema".into(),
        ));
    }
    Ok(file.len() as u64)
}

/// Executed by every real fixture constructor, including the hosted header-demand test.
/// Expected bit patterns are literal IEEE cases, independent of the converter under test.
fn validate_source_serialization(root: &Path) -> Result<()> {
    let cases: [(u32, u16); 13] = [
        (0x0000_0000, 0x0000),
        (0x8000_0000, 0x8000),
        (0x3f80_0000, 0x3f80),
        (0x3f80_8000, 0x3f80),
        (0x3f81_8000, 0x3f82),
        (0xbf80_8000, 0xbf80),
        (0x0001_0000, 0x0001),
        (0x0000_0001, 0x0000),
        (0x8000_0001, 0x8000),
        (0x0000_8000, 0x0000),
        (0x0000_8001, 0x0001),
        (0x0080_0000, 0x0080),
        (0x7f7f_0000, 0x7f7f),
    ];
    for (bits, expected) in cases {
        assert_eq!(bf16_bytes(f32::from_bits(bits))?, expected.to_le_bytes());
    }
    for bits in [
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f7f_8000,
    ] {
        assert!(
            bf16_bytes(f32::from_bits(bits)).is_err(),
            "nonfinite/finite overflow must refuse"
        );
    }
    let path = root.join("source-abi-probe.safetensors");
    let tensors = BTreeMap::from([
        (
            "fixture.weight".into(),
            HostTensor {
                shape: vec![cases.len()],
                values: cases
                    .iter()
                    .map(|&(bits, _)| f32::from_bits(bits))
                    .collect(),
            },
        ),
        (
            "rope.bands".into(),
            HostTensor {
                shape: vec![1],
                values: vec![f32::from_bits(0x3f80_8000)],
            },
        ),
    ]);
    write_tensors(&path, tensors, true)?;
    let raw = fs::read(&path)?;
    let length = u64::from_le_bytes(raw[..8].try_into().expect("generated header length")) as usize;
    let header: Value =
        serde_json::from_slice(&raw[8..8 + length]).map_err(|e| Error::Msg(e.to_string()))?;
    assert_eq!(header["fixture.weight"]["dtype"], "BF16");
    assert_eq!(header["fixture.weight"]["shape"], json!([cases.len()]));
    assert_eq!(
        header["fixture.weight"]["data_offsets"],
        json!([0, cases.len() * 2])
    );
    assert_eq!(header["rope.bands"]["dtype"], "F32");
    assert_eq!(header["rope.bands"]["shape"], json!([1]));
    assert_eq!(
        header["rope.bands"]["data_offsets"],
        json!([cases.len() * 2, cases.len() * 2 + 4])
    );
    let expected: Vec<u8> = cases
        .iter()
        .flat_map(|&(_, bits)| bits.to_le_bytes())
        .collect();
    let payload = &raw[8 + length..];
    assert_eq!(&payload[..expected.len()], expected.as_slice());
    assert_eq!(&payload[expected.len()..], &0x3f80_8000_u32.to_le_bytes());
    fs::remove_file(path)?; // Only this constructor-owned probe in its newly created scratch root.
    Ok(())
}

#[derive(Clone, Debug)]
pub struct Fixture {
    pub root: PathBuf,
    pub dataset_dir: PathBuf,
    pub config: gen_core::moss_image::MossImageConfig,
    pub files: BTreeMap<String, String>,
    pub source_identity: String,
    pub tensor_file_bytes: u64,
}
impl Fixture {
    /// The caller supplies a new path inside its own scratch/TempDir. Existing data is never removed.
    pub fn create(root: &Path) -> Result<Self> {
        fs::create_dir(root)?;
        validate_source_serialization(root)?;
        let config = gen_core::moss_image::MossImageConfig {
            dim: 16,
            ffn_dim: 24,
            num_heads: 2,
            num_layers: 3,
            text_dim: 32,
            freq_dim: 8,
            latent_channels: 128,
        };
        let mut tensor_file_bytes = 0;
        for (name, tensors) in [
            ("transformer/model.safetensors", transformer()?),
            ("text_encoder/model.safetensors", text()),
            ("vae/diffusion_pytorch_model.safetensors", vae()),
        ] {
            tensor_file_bytes +=
                write_tensors(&root.join(name), tensors, name.starts_with("transformer/"))?;
        }
        let qwen = json!({"text_config":{"model_type":"qwen3_5_text","hidden_size":32,"num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-6,"rope_theta":10000000.,"partial_rotary_factor":0.5,"max_position_embeddings":4096,"tie_word_embeddings":false,"full_attention_interval":1,"linear_num_value_heads":4,"linear_num_key_heads":2,"linear_key_head_dim":4,"linear_value_head_dim":4,"linear_conv_kernel_dim":4},"vision_config":{"model_type":"qwen3_5"}});
        fs::write(
            root.join("text_encoder/config.json"),
            serde_json::to_vec(&qwen).map_err(|e| Error::Msg(e.to_string()))?,
        )?;
        let words = [
            "[UNK]",
            "<|im_start|>",
            "<|im_end|>",
            "user",
            "assistant",
            "style",
            "portrait",
            "landscape",
            "figure",
            "light",
            "blue",
            "red",
        ];
        let vocab = words
            .iter()
            .enumerate()
            .map(|(i, w)| ((*w).to_owned(), json!(i)))
            .collect::<BTreeMap<_, _>>();
        let tokenizer = json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":vocab,"unk_token":"[UNK]"}});
        fs::create_dir(root.join("tokenizer"))?;
        fs::write(
            root.join("tokenizer/tokenizer.json"),
            serde_json::to_vec(&tokenizer).map_err(|e| Error::Msg(e.to_string()))?,
        )?;
        let dataset_dir = root.join("dataset");
        fs::create_dir(&dataset_dir)?;
        let mask_dir = root.join("masks");
        fs::create_dir(&mask_dir)?;
        for (index, (width, height, caption)) in [
            (80, 40, "style landscape blue"),
            (40, 80, "style portrait red"),
            (48, 48, "figure light"),
        ]
        .into_iter()
        .enumerate()
        {
            let image = image::RgbImage::from_fn(width, height, |x, y| {
                image::Rgb([
                    ((x * 3 + y + index as u32 * 17) % 256) as u8,
                    ((y * 5 + x) % 256) as u8,
                    ((x ^ y) % 256) as u8,
                ])
            });
            image
                .save(dataset_dir.join(format!("image{index}.png")))
                .map_err(|e| Error::Msg(e.to_string()))?;
            image::GrayImage::from_fn(width, height, |x, _| {
                image::Luma([if x < width / 2 { 255 } else { 0 }])
            })
            .save(mask_dir.join(format!("mask{index}.png")))
            .map_err(|e| Error::Msg(e.to_string()))?;
            fs::write(dataset_dir.join(format!("image{index}.txt")), caption)?;
        }
        let evaluation_dir = root.join("evaluation");
        fs::create_dir(&evaluation_dir)?;
        image::RgbImage::from_fn(64, 48, |x, y| {
            image::Rgb([
                ((x * 7 + y * 3) % 256) as u8,
                ((x + y * 11) % 256) as u8,
                ((x * 3 + y * 5) % 256) as u8,
            ])
        })
        .save(evaluation_dir.join("heldout.png"))
        .map_err(|e| Error::Msg(e.to_string()))?;
        image::GrayImage::from_pixel(64, 48, image::Luma([255]))
            .save(evaluation_dir.join("heldout-mask.png"))
            .map_err(|e| Error::Msg(e.to_string()))?;
        let mut files = BTreeMap::new();
        for name in [
            "transformer/model.safetensors",
            "text_encoder/model.safetensors",
            "text_encoder/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "tokenizer/tokenizer.json",
        ] {
            files.insert(
                name.into(),
                format!("{:x}", Sha256::digest(fs::read(root.join(name))?)),
            );
        }
        let source_identity = format!(
            "synthetic-moss-full-pipeline-v1:{:x}",
            Sha256::digest(serde_json::to_vec(&files).map_err(|e| Error::Msg(e.to_string()))?)
        );
        Ok(Self {
            root: root.to_path_buf(),
            dataset_dir,
            config,
            files,
            source_identity,
            tensor_file_bytes,
        })
    }
    pub fn evaluation_manifest(&self) -> Result<crate::moss_training_probe::EvaluationManifest> {
        let mut item = TrainingItem::captioned(
            self.root.join("evaluation/heldout.png"),
            "style figure blue light".into(),
        );
        item.subject_mask_path = Some(self.root.join("evaluation/heldout-mask.png"));
        crate::moss_training_probe::EvaluationManifest::new(
            item,
            32,
            32,
            0x4845_4c44_4f55_5401,
            0.375,
        )
    }
    pub fn items(&self) -> Result<Vec<TrainingItem>> {
        (0..3)
            .map(|i| {
                let mut item = TrainingItem::captioned(
                    self.dataset_dir.join(format!("image{i}.png")),
                    fs::read_to_string(self.dataset_dir.join(format!("image{i}.txt")))?,
                );
                item.subject_mask_path = Some(self.root.join("masks").join(format!("mask{i}.png")));
                Ok(item)
            })
            .collect()
    }
    pub fn verify(&self) -> Result<()> {
        for (name, digest) in &self.files {
            if &format!("{:x}", Sha256::digest(fs::read(self.root.join(name))?)) != digest {
                return Err(Error::Msg("synthetic fixture content changed".into()));
            }
        }
        Ok(())
    }
}

/// The four cases deliberately distribute cache and optimizer controls; this is not a Cartesian
/// campaign, benchmark or real-weight result. Both backends call this with their REAL factories.
pub fn case_config(case: usize) -> Result<gen_core::TrainingConfig> {
    use gen_core::{AspectRatioBucket, NetworkType, TrainingConfig};
    let (cache_latents, cache_text_embeddings, optimizer, network) = match case {
        0 => (true, true, "adamw", NetworkType::Lora),
        1 => (true, false, "adam", NetworkType::Lokr),
        2 => (false, true, "prodigyopt", NetworkType::Lora),
        3 => (false, false, "rose", NetworkType::Lokr),
        _ => return Err(Error::Msg("unknown full-pipeline fixture case".into())),
    };
    let config = TrainingConfig {
        rank: 2,
        alpha: 0.7,
        learning_rate: 0.01,
        optimizer: optimizer.into(),
        network_type: network,
        decompose_factor: 2,
        lora_target_modules: vec!["q".into(), "ffn_up_gate".into(), "ffn_down".into()],
        steps: 6,
        batch_size: 2,
        gradient_accumulation: 2,
        gradient_checkpointing: case.is_multiple_of(2),
        train_dtype: if case < 2 { "bf16" } else { "f32" }.into(),
        resolution: 32,
        seed: 173,
        lr_scheduler: gen_core::LrSchedule::Cosine,
        lr_warmup_steps: 1,
        save_every: 1,
        sample_every: 2,
        sample_prompts: vec!["style landscape blue".into()],
        sample_steps: 2,
        sample_guidance_scale: 1.,
        cache_latents,
        cache_text_embeddings,
        memory_limit_bytes: Some(2 * 1024 * 1024 * 1024),
        subject_mask_loss: if case == 3 {
            Some(gen_core::SubjectMaskLoss {
                background_weight: 0.25,
                subject_weight: 1.,
            })
        } else {
            None
        },
        aspect_ratio_buckets: vec![
            AspectRatioBucket {
                width: 64,
                height: 32,
                repeats: 1,
            },
            AspectRatioBucket {
                width: 32,
                height: 64,
                repeats: 2,
            },
            AspectRatioBucket {
                width: 32,
                height: 32,
                repeats: 1,
            },
        ],
        ..Default::default()
    };
    Ok(config)
}

pub fn assert_recoverable_case(
    data: &Fixture,
    root: &Path,
    case: usize,
    mut trainer: impl FnMut() -> Result<Box<dyn gen_core::Trainer>>,
    mut generator: impl FnMut(Vec<gen_core::AdapterSpec>) -> Result<Box<dyn gen_core::Generator>>,
) -> Result<()> {
    use gen_core::moss_image_training::recovery::{bundle, DataPlan};
    use gen_core::{CancelFlag, NetworkType, TrainingProgress, TrainingRequest};
    let config = case_config(case)?;
    let network = config.network_type;
    let optimizer = config.optimizer.clone();
    let full = root.join("uninterrupted");
    let resumed = root.join("interrupted");
    fs::create_dir_all(&full)?;
    fs::create_dir_all(&resumed)?;
    let request = TrainingRequest {
        items: data.items()?,
        config,
        output_dir: full.clone(),
        file_name: "adapter.safetensors".into(),
        trigger_words: vec!["style".into()],
        cancel: CancelFlag::new(),
    };
    let plan = DataPlan::from_request(&request)?;
    assert!(
        plan.entries.iter().any(|g| g.width != g.height),
        "test must actually execute rectangles"
    );
    let mut ordinary = trainer()?;
    let mut full_steps = Vec::new();
    let mut previews = 0;
    let result = ordinary.train(&request, &mut |event| match event {
        TrainingProgress::Training { step, total, loss } => {
            assert_eq!(total, 6);
            assert!(loss.is_finite());
            full_steps.push(step);
        }
        TrainingProgress::Sample { image, .. } => {
            assert_eq!((image.width, image.height), (32, 32));
            assert_eq!(image.pixels.len(), 32 * 32 * 3);
            previews += 1;
        }
        _ => (),
    })?;
    assert_eq!(full_steps, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(previews, 3);
    assert_eq!(result.steps, 6);
    assert!(result.final_loss.is_finite());
    assert!(result.adapter_path.is_file());
    drop(ordinary);
    let mut interrupted_request = request.clone();
    interrupted_request.output_dir = resumed.clone();
    interrupted_request.cancel = CancelFlag::new();
    let cancel = interrupted_request.cancel.clone();
    let mut same_worker = trainer()?;
    let stopped = same_worker.train(&interrupted_request, &mut |event| {
        if matches!(event, TrainingProgress::Checkpoint { step: 3 }) {
            cancel.cancel();
        }
    });
    assert!(matches!(stopped,Err(Error::Canceled)),"actual active cancellation must return Canceled, not a fake successful output: {stopped:?}");
    let partial = bundle::read(&bundle::latest(&resumed, "adapter")?)?;
    assert_eq!(partial.step, 3);
    assert_eq!(partial.update, 1);
    assert!(partial.accumulated_tokens > 0);
    assert_eq!(partial.sample_cursor, plan.batch(3)?.sample_cursor);
    let mut retry = interrupted_request.clone();
    retry.cancel = CancelFlag::new();
    retry.config.resume = true;
    same_worker.validate(&retry)?; // Positive baseline at one update + a partial window.
    for mutation in 0..5 {
        let mut wrong = retry.clone();
        match mutation {
            0 => wrong.items[0].caption.push_str(" changed"),
            1 => wrong.config.learning_rate *= 2.,
            2 => wrong.config.optimizer = if optimizer == "adam" { "adamw" } else { "adam" }.into(),
            3 => wrong.config.steps += 1,
            _ => wrong.config.aspect_ratio_buckets.reverse(),
        }
        assert!(
            same_worker.validate(&wrong).is_err(),
            "resume mismatch {mutation} accepted"
        );
    }
    let mut missing = retry.clone();
    missing.output_dir = root.join("no-state");
    assert!(same_worker.validate(&missing).is_err());
    let image = &retry.items[0].image_path;
    let original = fs::read(image)?;
    let mut changed = image::open(image)
        .map_err(|e| Error::Msg(e.to_string()))?
        .to_rgb8();
    let pixel = changed.get_pixel_mut(0, 0);
    pixel.0[0] ^= 127;
    changed.save(image).map_err(|e| Error::Msg(e.to_string()))?;
    assert!(
        same_worker.validate(&retry).is_err(),
        "actual image content mismatch accepted"
    );
    fs::write(image, original)?;
    same_worker.validate(&retry)?;
    let mut resumed_steps = Vec::new();
    let recovered = same_worker.train(&retry, &mut |event| {
        if let TrainingProgress::Training { step, total, loss } = event {
            assert_eq!(total, 6);
            assert!(loss.is_finite());
            resumed_steps.push(step);
        }
    })?;
    assert_eq!(resumed_steps, vec![4, 5, 6]);
    assert_eq!(recovered.steps, result.steps);
    assert_eq!(recovered.final_loss.to_bits(),result.final_loss.to_bits(),"same original horizon and partial gradient state must preserve the final objective exactly on the CPU reference path");
    let a = gen_core::weightsmeta::CheckpointMeta::from_file(&result.adapter_path)?;
    let b = gen_core::weightsmeta::CheckpointMeta::from_file(&recovered.adapter_path)?;
    let a_keys = a.keys().collect::<Vec<_>>();
    let b_keys = b.keys().collect::<Vec<_>>();
    assert_eq!(a_keys, b_keys);
    for key in a_keys {
        let left = a.tensor(key).unwrap();
        let right = b.tensor(key).unwrap();
        assert_eq!(left.shape, right.shape);
        assert_eq!(left.dtype, right.dtype);
        assert_eq!(
            left.data, right.data,
            "actual exported factor {key} differs after resume"
        );
    }
    let final_state = bundle::read(&bundle::latest(&resumed, "adapter")?)?;
    assert_eq!(final_state.step, 6);
    assert_eq!(final_state.update, 3);
    assert_eq!(final_state.accumulated_tokens, 0);
    assert!(
        same_worker.validate(&retry).is_err(),
        "completed original horizon must not restart"
    );
    // Caller limits lower physical/oracle admission and never become an override of discovery.
    for cap in [0, 1] {
        let mut low = request.clone();
        low.cancel = CancelFlag::new();
        low.config.memory_limit_bytes = Some(cap);
        assert!(
            same_worker.validate(&low).is_err(),
            "user cap {cap} ignored"
        );
    }
    // A new non-resume request on the SAME trainer still executes after canceled + completed runs.
    let mut again = request.clone();
    again.cancel = CancelFlag::new();
    again.output_dir = root.join("reentry");
    again.config.steps = 1;
    again.config.save_every = 0;
    again.config.sample_every = 0;
    let next = same_worker.train(&again, &mut |_| {})?;
    assert_eq!(next.steps, 1);
    assert!(next.adapter_path.is_file());
    let adapter = gen_core::AdapterSpec::new(
        recovered.adapter_path.clone(),
        1.,
        if network == NetworkType::Lokr {
            gen_core::AdapterKind::Lokr
        } else {
            gen_core::AdapterKind::Lora
        },
    );
    let model = generator(vec![adapter])?;
    let request = gen_core::GenerationRequest {
        prompt: "style landscape blue".into(),
        width: 32,
        height: 32,
        steps: Some(2),
        guidance: Some(1.),
        seed: Some(173),
        moss_image: Some(gen_core::moss_image::MossImageOptions {
            max_sequence_length: 32,
            uncond_skip_layers: Some(Vec::new()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let generated = model.generate(&request, &mut |_| {})?;
    match generated {
        gen_core::GenerationOutput::Images(images) => {
            assert_eq!(images.len(), 1);
            assert_eq!((images[0].width, images[0].height), (32, 32));
            assert_eq!(images[0].pixels.len(), 32 * 32 * 3);
        }
        _ => panic!("real generator returned no image"),
    }
    assert!(
        model.moss_image_execution_report().is_none(),
        "synthetic source must never masquerade as a sealed published-author execution receipt"
    );
    assert!(
        !model.adapter_apply_reports().is_empty(),
        "saved adapter must actually install on the real generator"
    );
    data.verify()?;
    Ok(())
}

/// Header-only pre-execution profile. Tensor demand comes from the ACTUAL backend estimator;
/// source-file writer/read staging is derived from actual files. The separate host allowance is
/// explicit engineering headroom, not a measured peak or universal process-size guarantee.
pub fn header_demand_profile(
    data: &Fixture,
    output: &Path,
    mut estimate: impl FnMut(
        gen_core::moss_image::MossImageVariant,
        &gen_core::TrainingRequest,
    ) -> Result<Value>,
) -> Result<Value> {
    let mut plans = Vec::new();
    for case in 0..4 {
        let req = gen_core::TrainingRequest {
            items: data.items()?,
            config: case_config(case)?,
            output_dir: output.into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: vec!["style".into()],
            cancel: gen_core::CancelFlag::new(),
        };
        let variant = if case < 2 {
            gen_core::moss_image::MossImageVariant::Base
        } else {
            gen_core::moss_image::MossImageVariant::Sft
        };
        let mut plan = estimate(variant, &req)?;
        let peak = plan["tensor_peak_bytes"]
            .as_u64()
            .ok_or_else(|| Error::Msg("actual header tensor peak missing".into()))?;
        let file_staging = data
            .tensor_file_bytes
            .checked_mul(4)
            .ok_or_else(|| Error::Msg("fixture host staging overflow".into()))?;
        let reserve = 512 * 1024 * 1024u64;
        let host_staging = plan["host_staging_peak_bytes"]
            .as_u64()
            .ok_or_else(|| Error::Msg("actual host staging domain missing".into()))?;
        let host = peak
            .checked_add(host_staging)
            .and_then(|v| v.checked_add(file_staging))
            .and_then(|v| v.checked_add(reserve))
            .ok_or_else(|| Error::Msg("fixture host envelope overflow".into()))?;
        if host > 2 * 1024 * 1024 * 1024 {
            return Err(Error::Unsupported(format!("actual fixture case {case} needs {host} host-envelope bytes, above its declared 2-GiB CPU safety ceiling")));
        }
        plan["host_envelope_bytes"] = json!(host);
        plan["source_file_staging_bytes"] = json!(file_staging);
        plan["declared_host_reserve_bytes"] = json!(reserve);
        plan["cache_latents"] = json!(req.config.cache_latents);
        plan["cache_text_embeddings"] = json!(req.config.cache_text_embeddings);
        plan["optimizer"] = json!(req.config.optimizer);
        plans.push(plan);
    }
    let encoder_peak = plans[0]["encoder_only_peak_bytes"]
        .as_u64()
        .ok_or_else(|| Error::Msg("actual encoder-only demand missing".into()))?;
    let token_host = encoder_peak
        .checked_add(
            data.tensor_file_bytes
                .checked_mul(4)
                .ok_or_else(|| Error::Msg("encoder staging overflow".into()))?,
        )
        .and_then(|n| n.checked_add(512 * 1024 * 1024))
        .ok_or_else(|| Error::Msg("encoder host envelope overflow".into()))?;
    if token_host > 2 * 1024 * 1024 * 1024 {
        return Err(Error::Unsupported(
            "4096-token CPU shape case does not fit its declared host envelope".into(),
        ));
    }
    Ok(
        json!({"profile":"synthetic-moss-full-pipeline-cpu-v1","cpu_reference":true,"token_bound_case_host_envelope_bytes":token_host,"source_fixture_identity":data.source_identity,"tensor_file_bytes":data.tensor_file_bytes,"host_reserve_is_engineering_allowance":true,"measured_peak_bytes":null,"cases":plans}),
    )
}

/// Exact same tokenizer and REAL encoder closures, with no padded/fake hidden-state tensors.
pub fn assert_ragged_token_bounds(
    data: &Fixture,
    output: &Path,
    bound: impl Fn(&gen_core::TrainingRequest) -> Result<u64>,
    encode: impl Fn(&gen_core::TrainingRequest) -> Result<Vec<u64>>,
) -> Result<()> {
    let base = gen_core::TrainingRequest {
        items: data.items()?,
        config: case_config(0)?,
        output_dir: output.into(),
        file_name: "unused.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: gen_core::CancelFlag::new(),
    };
    let mut short = base.clone();
    short.items.truncate(1);
    short.items[0].caption = "style".into();
    short.config.sample_every = 0;
    let mut long = short.clone();
    long.items[0].caption = "landscape ".repeat(128);
    let mut mixed = base.clone();
    mixed.items[0].caption = "style".into();
    mixed.items[1].caption = "figure ".repeat(96);
    mixed.config.sample_every = 0;
    let mut preview = short.clone();
    preview.config.sample_every = 1;
    preview.config.sample_prompts = vec!["portrait ".repeat(192)];
    let mut trigger = short.clone();
    trigger.items[0].caption = "figure".into();
    trigger.config.trigger_word = Some("style ".repeat(64));
    let mut truncated = short.clone();
    truncated.items[0].caption = "style ".repeat(5000);
    let mut limits = Vec::new();
    for request in [&short, &long, &mixed, &preview, &trigger, &truncated] {
        let limit = bound(request)?;
        let actual = encode(request)?;
        let expected_prompts =
            gen_core::moss_image_training::recovery::conditioning_prompts(request);
        assert_eq!(
            actual.len(),
            expected_prompts.len(),
            "all caption/preview/negative paths must execute"
        );
        assert_eq!(
            limit,
            actual
                .iter()
                .copied()
                .max()
                .expect("real nonempty encoded rows"),
            "actual pre-load bound must match executed ragged shapes"
        );
        assert!(actual.iter().all(|&n| n > 0 && n <= 4096));
        limits.push(limit);
    }
    assert!(
        limits[1] > limits[0]
            && limits[2] > limits[0]
            && limits[3] > limits[1]
            && limits[4] > limits[0]
    );
    assert_eq!(
        limits[5], 4096,
        "public right-truncation limit remains 4096"
    );
    Ok(())
}

#[cfg(test)]
mod source_abi_tests {
    use super::*;
    #[test]
    fn bf16_source_rounding_is_finite_and_ties_to_even() {
        for (value, bits) in [
            (0.0, 0x0000_u16),
            (-0.0, 0x8000),
            (1.0, 0x3f80),
            (f32::from_bits(0x3f80_8000), 0x3f80),
            (f32::from_bits(0x3f81_8000), 0x3f82),
            (f32::from_bits(0xbf80_8000), 0xbf80),
        ] {
            assert_eq!(bf16_bytes(value).unwrap(), bits.to_le_bytes());
        }
        for value in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            -f32::MAX,
        ] {
            assert!(bf16_bytes(value).is_err());
        }
    }
    #[test]
    fn ordinary_transformer_source_is_bf16_except_rope_and_other_towers_stay_f32() {
        let temp = tempfile::tempdir().unwrap();
        let fixture = Fixture::create(&temp.path().join("fixture")).unwrap();
        for relative in [
            "transformer/model.safetensors",
            "text_encoder/model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ] {
            let headers =
                gen_core::weightsmeta::safetensors_path_tensor_headers(fixture.root.join(relative))
                    .unwrap();
            assert!(!headers.is_empty());
            for header in headers {
                let expected =
                    if relative.starts_with("transformer/") && header.name != "rope.bands" {
                        gen_core::weightsmeta::Dtype::BF16
                    } else {
                        gen_core::weightsmeta::Dtype::F32
                    };
                assert_eq!(header.dtype, expected, "{relative}:{}", header.name);
            }
        }
        fixture.verify().unwrap();
    }
}
