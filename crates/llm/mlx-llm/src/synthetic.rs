//! Seeded synthetic checkpoints for the crate's unit suites (story sc-24434): a builder that fills
//! a [`Weights`] map with small random tensors, and a word-level tokenizer, so a provider can be
//! driven end to end on a shape-valid fixture without real weights.

use std::collections::HashMap;

use mlx_rs::Array;

use core_llm::Tokenizer;

use crate::primitives::sampler::{SplitMix64, TokenRng};
use crate::primitives::Weights;

/// A seeded synthetic checkpoint under construction.
pub(crate) struct Synth {
    rng: SplitMix64,
    map: HashMap<String, Array>,
}

impl Synth {
    /// An empty checkpoint whose random tensors are drawn from `seed`.
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::new(seed),
            map: HashMap::new(),
        }
    }

    /// A uniform `[-0.4, 0.4)` tensor.
    pub(crate) fn randn(&mut self, key: impl Into<String>, shape: &[i32]) -> &mut Self {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (self.rng.next_f32() - 0.5) * 0.8).collect();
        self.map.insert(key.into(), Array::from_slice(&data, shape));
        self
    }

    /// A constant tensor.
    pub(crate) fn fill(&mut self, key: impl Into<String>, shape: &[i32], value: f32) -> &mut Self {
        let n: i32 = shape.iter().product();
        self.map.insert(
            key.into(),
            Array::from_slice(&vec![value; n as usize], shape),
        );
        self
    }

    /// A LayerNorm pair under `prefix`: `weight` ones, `bias` zeros.
    pub(crate) fn layer_norm(&mut self, prefix: &str, width: i32) -> &mut Self {
        self.fill(format!("{prefix}.weight"), &[width], 1.0).fill(
            format!("{prefix}.bias"),
            &[width],
            0.0,
        )
    }

    /// A biased linear `[out, in]` under `prefix`.
    pub(crate) fn linear(&mut self, prefix: &str, out: i32, input: i32) -> &mut Self {
        self.randn(format!("{prefix}.weight"), &[out, input])
            .randn(format!("{prefix}.bias"), &[out])
    }

    /// Scale the tensor at `key` by `factor` (sharpens a fixture's attention so its output
    /// depends on positions).
    pub(crate) fn scale(&mut self, key: &str, factor: f32) -> &mut Self {
        let a = &self.map[key];
        let shape = a.shape().to_vec();
        let data: Vec<f32> = a.as_slice::<f32>().iter().map(|x| x * factor).collect();
        self.map
            .insert(key.to_string(), Array::from_slice(&data, &shape));
        self
    }

    /// Zero row `row` of the 2-D tensor at `key`, so a tied head scores that id exactly `0`.
    pub(crate) fn zero_row(&mut self, key: &str, row: i32) -> &mut Self {
        let a = &self.map[key];
        let shape = a.shape().to_vec();
        let mut data = a.as_slice::<f32>().to_vec();
        let width = shape[1] as usize;
        data[row as usize * width..(row as usize + 1) * width].fill(0.0);
        self.map
            .insert(key.to_string(), Array::from_slice(&data, &shape));
        self
    }

    /// The finished checkpoint.
    pub(crate) fn weights(&mut self) -> Weights {
        Weights::from_map(std::mem::take(&mut self.map))
    }
}

/// A whitespace word-level tokenizer over `t0 … t{vocab-1}` (unknown words are `t0`), plus
/// `extra` words at explicit ids.
pub(crate) fn word_tokenizer(vocab: usize, extra: &[(&str, u32)]) -> Tokenizer {
    Tokenizer::from_json(&word_tokenizer_json(vocab, extra)).unwrap()
}

/// The `tokenizer.json` text of [`word_tokenizer`].
pub(crate) fn word_tokenizer_json(vocab: usize, extra: &[(&str, u32)]) -> String {
    let mut entries: Vec<String> = (0..vocab).map(|i| format!("\"t{i}\": {i}")).collect();
    entries.extend(extra.iter().map(|(word, id)| format!("\"{word}\": {id}")));
    format!(
        r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
            "decoder": null,
            "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
        entries.join(", ")
    )
}

/// A seeded, shape-valid Prism/Bonsai Qwen3.5 hybrid held in memory (story sc-24444): one Gated
/// DeltaNet layer and one full-attention layer, every large projection a packed two-bit Prism
/// matrix with random codes, `bias == -scale` affine parameters and random explicit `±1` signs per
/// input width (block 128), exactly the parts [`PrismMlxPack`](crate::prism::PrismMlxPack)
/// validates. Norms are direct multipliers, as in the published artifacts.
pub(crate) struct PrismFixture {
    pub(crate) config: serde_json::Value,
    pub(crate) weights: Weights,
    pub(crate) pack: crate::prism::PrismMlxPack,
    /// The packed modules (`path`, `embedding`) and their Hadamard contract, for a snapshot writer.
    modules: Vec<(String, bool)>,
    hadamard: core_llm::PrismHadamardMetadata,
}

/// The text geometry of [`prism_qwen35`] (hidden 128, heads 2/1 × 64, MLP 256, vocab 64).
pub(crate) fn prism_qwen35_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "prism_hadamard_qwen35",
        "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 128, "num_hidden_layers": 2,
            "intermediate_size": 256, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 64, "vocab_size": 64, "rms_norm_eps": 1e-6, "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.5, "max_position_embeddings": 512,
            "tie_word_embeddings": false, "full_attention_interval": 2,
            "linear_num_value_heads": 2, "linear_num_key_heads": 1, "linear_key_head_dim": 64,
            "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 0, "mtp_use_dedicated_embeddings": false
        }
    })
}

/// Build [`PrismFixture`] from `seed`.
pub(crate) fn prism_qwen35(seed: u64) -> PrismFixture {
    use std::collections::{BTreeMap, BTreeSet};

    use mlx_rs::Dtype;

    const BLOCK: i32 = 128;
    let mut rng = SplitMix64::new(seed);
    let mut signs_by_width = BTreeMap::new();
    for width in [128usize, 256] {
        let signs: Vec<i8> = (0..width)
            .map(|_| if rng.next_u64() & 1 == 0 { -1 } else { 1 })
            .collect();
        signs_by_width.insert(width, signs);
    }
    // (module path, inverse (embedding), rows, input width, scale magnitude)
    let packed: [(&str, bool, i32, i32, f32); 15] = [
        ("lm_head", false, 64, 128, 0.25),
        ("model.embed_tokens", true, 64, 128, 0.5),
        (
            "model.layers.0.linear_attn.in_proj_qkv",
            false,
            256,
            128,
            0.05,
        ),
        (
            "model.layers.0.linear_attn.in_proj_z",
            false,
            128,
            128,
            0.05,
        ),
        ("model.layers.0.linear_attn.out_proj", false, 128, 128, 0.05),
        ("model.layers.0.mlp.gate_proj", false, 256, 128, 0.05),
        ("model.layers.0.mlp.up_proj", false, 256, 128, 0.05),
        ("model.layers.0.mlp.down_proj", false, 128, 256, 0.05),
        ("model.layers.1.self_attn.q_proj", false, 256, 128, 0.05),
        ("model.layers.1.self_attn.k_proj", false, 64, 128, 0.05),
        ("model.layers.1.self_attn.v_proj", false, 64, 128, 0.05),
        ("model.layers.1.self_attn.o_proj", false, 128, 128, 0.05),
        ("model.layers.1.mlp.gate_proj", false, 256, 128, 0.05),
        ("model.layers.1.mlp.up_proj", false, 256, 128, 0.05),
        ("model.layers.1.mlp.down_proj", false, 128, 256, 0.05),
    ];
    let mut map = HashMap::new();
    let mut forward = BTreeSet::new();
    let mut inverse = BTreeSet::new();
    for (path, embedding, rows, width, magnitude) in packed {
        let words: Vec<u32> = (0..rows * width / 16)
            .map(|_| rng.next_u64() as u32)
            .collect();
        let groups = rows * width / BLOCK;
        let scales: Vec<f32> = (0..groups)
            .map(|_| magnitude * (0.5 + rng.next_f32()))
            .collect();
        let scales = Array::from_slice(&scales, &[rows, width / BLOCK])
            .as_dtype(Dtype::Float16)
            .unwrap();
        let biases = mlx_rs::ops::negative(&scales).unwrap();
        let signs: Vec<f32> = signs_by_width[&(width as usize)]
            .iter()
            .map(|&s| s as f32)
            .collect();
        let base = format!("language_model.{path}");
        map.insert(
            format!("{base}.weight"),
            Array::from_slice(&words, &[rows, width / 16]),
        );
        map.insert(format!("{base}.scales"), scales);
        map.insert(format!("{base}.biases"), biases);
        map.insert(format!("{base}.signs"), Array::from_slice(&signs, &[width]));
        if embedding {
            inverse.insert(format!("{base}.weight"));
        } else {
            forward.insert(format!("{base}.weight"));
        }
    }
    let mut synth = Synth::new(seed ^ 0x5eed);
    let p = "language_model.model";
    // Direct-multiplier norms near one; small GDN decay/gate parameters.
    for (key, width) in [
        (format!("{p}.norm.weight"), 128),
        (format!("{p}.layers.0.input_layernorm.weight"), 128),
        (format!("{p}.layers.0.post_attention_layernorm.weight"), 128),
        (format!("{p}.layers.1.input_layernorm.weight"), 128),
        (format!("{p}.layers.1.post_attention_layernorm.weight"), 128),
        (format!("{p}.layers.1.self_attn.q_norm.weight"), 64),
        (format!("{p}.layers.1.self_attn.k_norm.weight"), 64),
        (format!("{p}.layers.0.linear_attn.norm.weight"), 64),
    ] {
        synth.randn(key.clone(), &[width]);
        let shifted = mlx_rs::ops::add(&synth.map[&key], Array::from_f32(1.0)).unwrap();
        synth.map.insert(key, shifted);
    }
    synth
        .randn(
            format!("{p}.layers.0.linear_attn.in_proj_a.weight"),
            &[2, 128],
        )
        .randn(
            format!("{p}.layers.0.linear_attn.in_proj_b.weight"),
            &[2, 128],
        )
        .randn(
            format!("{p}.layers.0.linear_attn.conv1d.weight"),
            &[256, 1, 4],
        )
        .randn(format!("{p}.layers.0.linear_attn.A_log"), &[2])
        .randn(format!("{p}.layers.0.linear_attn.dt_bias"), &[2]);
    map.extend(std::mem::take(&mut synth.map));
    let hadamard = core_llm::PrismHadamardMetadata {
        block_size: BLOCK as usize,
        signs_by_width,
        forward_weight_names: forward,
        inverse_weight_names: inverse,
        gdn_v_grouped: true,
    };
    let modules: Vec<(String, bool)> = packed
        .iter()
        .map(|(path, embedding, ..)| ((*path).to_owned(), *embedding))
        .collect();
    let pack =
        crate::prism::PrismMlxPack::from_gguf(hadamard.clone(), modules.iter().cloned()).unwrap();
    PrismFixture {
        config: prism_qwen35_config(),
        weights: Weights::from_map(map),
        pack,
        modules,
        hadamard,
    }
}

/// Write [`prism_qwen35`] as a frozen Prism MLX snapshot directory (schema-2 `config.json`,
/// `hadamard.json`, `model.safetensors`, a word tokenizer over `t0 … t63`), so the provider's real
/// load path — config dispatch, [`PrismMlxPack::from_dir`](crate::prism::PrismMlxPack::from_dir),
/// admission — runs on it.
pub(crate) fn write_prism_snapshot(dir: &std::path::Path, seed: u64) {
    let fixture = prism_qwen35(seed);
    let mut config = fixture.config.clone();
    let extra = serde_json::json!({
        "schema_version": 2, "base_model_type": "qwen3_5",
        "tensor_namespace": "mlx-vlm-qwen3_5", "requires_runtime": "runtime/artifact.py",
        "hadamard_config": "hadamard.json", "gdn_activation_layout": "grouped",
        "components": {"text": true, "vision": false, "mtp": false},
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "modules": fixture.modules.iter().map(|(path, embedding)| serde_json::json!({
            "path": path, "block": fixture.hadamard.block_size, "embedding": embedding,
            "dtype": "float16"
        })).collect::<Vec<_>>(),
    });
    for (key, value) in extra.as_object().unwrap() {
        config[key] = value.clone();
    }
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let widths: Vec<usize> = fixture.hadamard.signs_by_width.keys().copied().collect();
    let values: Vec<f64> = fixture
        .hadamard
        .signs_by_width
        .values()
        .flatten()
        .map(|&s| f64::from(s))
        .collect();
    let hadamard = serde_json::json!({
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": fixture.hadamard.block_size,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.sign_mode": "explicit",
        "prism.hadamard.weight_names": fixture.hadamard.forward_weight_names,
        "prism.hadamard.inverse_weight_names": fixture.hadamard.inverse_weight_names,
        "prism.hadamard.sign_widths": widths,
        "prism.hadamard.sign_values": values,
        "prism.hadamard.gdn_v_grouped": true,
    });
    std::fs::write(
        dir.join("hadamard.json"),
        serde_json::to_vec(&hadamard).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), word_tokenizer_json(64, &[])).unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        r#"{"eos_token_id":999}"#,
    )
    .unwrap();
    let map = fixture.weights.into_map();
    let refs: Vec<_> = map.iter().map(|(n, a)| (n.as_str(), a)).collect();
    Array::save_safetensors(refs, None, dir.join("model.safetensors")).unwrap();
}

/// Write a standalone companion MTP head (`model_type` `qwen3_5_mtp`, bare tensor names, MLX
/// affine Q4 group 64 with BF16 scales — the published `EigenLabs/Qwen3.8-27B-MTP-4bit` layout)
/// into `dir`, with the geometry of `text_config` and random weights from `seed`. `q_rows`
/// overrides the stored `q_proj` output rows (to build a head whose tensors disagree with its own
/// config).
pub(crate) fn write_companion_head(
    dir: &std::path::Path,
    text_config: &serde_json::Value,
    seed: u64,
    q_rows: Option<i32>,
) {
    use mlx_rs::Dtype;

    let geometry = |k: &str| text_config[k].as_i64().unwrap() as i32;
    let (h, heads, kv, hd, inter) = (
        geometry("hidden_size"),
        geometry("num_attention_heads"),
        geometry("num_key_value_heads"),
        geometry("head_dim"),
        geometry("intermediate_size"),
    );
    let mut text = text_config.clone();
    text["mtp_num_hidden_layers"] = serde_json::json!(1);
    let config = serde_json::json!({
        "model_type": "qwen3_5_mtp",
        "block_size": 3,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "quantization_config": {"bits": 4, "group_size": 64, "mode": "affine"},
        "text_config": text,
        "tie_word_embeddings": false,
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut synth = Synth::new(seed);
    let mut tensors = Vec::new();
    for (name, rows, cols) in [
        ("fc", h, 2 * h),
        (
            "layers.0.self_attn.q_proj",
            q_rows.unwrap_or(2 * heads * hd),
            h,
        ),
        ("layers.0.self_attn.k_proj", kv * hd, h),
        ("layers.0.self_attn.v_proj", kv * hd, h),
        ("layers.0.self_attn.o_proj", h, heads * hd),
        ("layers.0.mlp.gate_proj", inter, h),
        ("layers.0.mlp.up_proj", inter, h),
        ("layers.0.mlp.down_proj", h, inter),
    ] {
        let key = format!("{name}.weight");
        synth.randn(key.clone(), &[rows, cols]);
        let dense = synth
            .map
            .remove(&key)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let (w, scales, biases) = mlx_rs::ops::quantize(&dense, 64, 4).unwrap();
        tensors.push((key, w));
        tensors.push((
            format!("{name}.scales"),
            scales.as_dtype(Dtype::Bfloat16).unwrap(),
        ));
        tensors.push((
            format!("{name}.biases"),
            biases.as_dtype(Dtype::Bfloat16).unwrap(),
        ));
    }
    for (name, width) in [
        ("pre_fc_norm_embedding", h),
        ("pre_fc_norm_hidden", h),
        ("norm", h),
        ("layers.0.input_layernorm", h),
        ("layers.0.post_attention_layernorm", h),
        ("layers.0.self_attn.q_norm", hd),
        ("layers.0.self_attn.k_norm", hd),
    ] {
        // Zero-centred Qwen3.8 checkpoint convention: the runtime applies `1 + w`.
        let key = format!("{name}.weight");
        synth.randn(key.clone(), &[width]);
        let small = synth
            .map
            .remove(&key)
            .unwrap()
            .multiply(Array::from_f32(0.1))
            .unwrap();
        tensors.push((key, small.as_dtype(Dtype::Bfloat16).unwrap()));
    }
    let refs: Vec<_> = tensors.iter().map(|(n, a)| (n.as_str(), a)).collect();
    Array::save_safetensors(refs, None, dir.join("model.safetensors")).unwrap();
}
