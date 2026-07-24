//! The scaffold's load-bearing gate (sc-14037): every constant in `mlx_gen_mage::config` is
//! checked against a **committed, byte-pinned copy of the real published config** or against the
//! **vendored frozen reference** — never against itself.
//!
//! Fixtures under `tests/fixtures/` are verbatim copies of
//! `microsoft/Mage-Flow @ 9f46d09d`'s `transformer/config.json`, `vae/config.json`,
//! `scheduler/scheduler_config.json` and `text_encoder/config.json` — **all four** component
//! configs, so no component is held to a weaker standard than the others. All six Mage-Flow
//! repositories ship them byte-identically (independently hashed across `Mage-Flow`, `-Base`,
//! `-Turbo`, `-Edit`, `-Edit-Base`, `-Edit-Turbo`), so pinning one copy pins the family.
//!
//! Constants with no home in any published config (the timestep-embedder block, the joint-attention
//! order, the VL long-edge cap, the native-resolution bounds) are pinned against the **vendored
//! frozen reference** via [`vendored`] instead. The claim in the first paragraph is meant literally
//! and was mutation-tested: flipping any covered constant fails at least one test here.

use mlx_gen_mage::config::{self, MageFlowConfig, QwenVlTextConfig};
use sha2::{Digest, Sha256};

const TRANSFORMER_CONFIG: &str = include_str!("fixtures/transformer_config.json");
const VAE_CONFIG: &str = include_str!("fixtures/vae_config.json");
const SCHEDULER_CONFIG: &str = include_str!("fixtures/scheduler_config.json");
const TEXT_ENCODER_CONFIG: &str = include_str!("fixtures/text_encoder_config.json");

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Byte tripwire: the fixtures are the published files, not a paraphrase. If a re-download ever
/// disagrees, every constant transcribed from them is suspect.
#[test]
fn fixtures_are_the_published_config_bytes() {
    assert_eq!(
        sha256(TRANSFORMER_CONFIG.as_bytes()),
        "8493c3b2722738c2a824ac82b1fd9c89fefb4e354fc88363207193db7fe702de",
        "transformer/config.json fixture drifted from the published bytes"
    );
    assert_eq!(
        sha256(VAE_CONFIG.as_bytes()),
        "abd124d603d6c6a03e9d0f2aa6d113b8c4afda0738400bdf2f99240aeaeaff76",
        "vae/config.json fixture drifted from the published bytes"
    );
    assert_eq!(
        sha256(SCHEDULER_CONFIG.as_bytes()),
        "438fd8bcf254740e5d3f3e9800bbd9c571e342ab87885388d1505b7531c69c02",
        "scheduler/scheduler_config.json fixture drifted from the published bytes"
    );
    assert_eq!(
        sha256(TEXT_ENCODER_CONFIG.as_bytes()),
        "edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d",
        "text_encoder/config.json fixture drifted from the published bytes"
    );
}

/// The nine consumed fields parse out of the real file and equal the built-in production config.
#[test]
fn real_transformer_config_parses_to_the_production_constants() {
    let parsed = MageFlowConfig::from_transformer_config_json(TRANSFORMER_CONFIG).unwrap();
    assert_eq!(parsed, MageFlowConfig::mage_flow());
    assert_eq!(parsed.in_channels, 128);
    assert_eq!(parsed.out_channels, 128);
    assert_eq!(parsed.context_in_dim, 2560);
    assert_eq!(parsed.hidden_size, 3072);
    assert_eq!(parsed.num_heads, 24);
    assert_eq!(parsed.depth, 12);
    assert_eq!(parsed.axes_dim, vec![16, 56, 56]);
    assert!(!parsed.checkpoint);
    assert_eq!(parsed.patch_size, 1);
    assert_eq!(parsed.head_dim(), 128);
    assert_eq!(parsed.axes_dim.iter().sum::<i32>(), parsed.head_dim());
}

/// **The discriminating half.** A test that only parses the real file would pass against a reader
/// that ignored the JSON and returned [`MageFlowConfig::mage_flow`]. Feed a config whose every
/// consumed field differs from production and require the parsed values to follow the *file*.
#[test]
fn parser_reads_the_file_rather_than_returning_the_defaults() {
    // Deliberately self-consistent: 4096 / 32 = head_dim 128 = 32 + 48 + 48.
    let mutated = r#"{
        "in_channels": 64,
        "out_channels": 32,
        "context_in_dim": 4096,
        "hidden_size": 4096,
        "num_heads": 32,
        "depth": 24,
        "axes_dim": [32, 48, 48],
        "checkpoint": true,
        "patch_size": 2
    }"#;
    let parsed = MageFlowConfig::from_transformer_config_json(mutated).unwrap();
    assert_ne!(parsed, MageFlowConfig::mage_flow());
    assert_eq!(parsed.in_channels, 64);
    assert_eq!(parsed.out_channels, 32);
    assert_eq!(parsed.context_in_dim, 4096);
    assert_eq!(parsed.hidden_size, 4096);
    assert_eq!(parsed.num_heads, 32);
    assert_eq!(parsed.depth, 24);
    assert_eq!(parsed.axes_dim, vec![32, 48, 48]);
    assert!(parsed.checkpoint);
    assert_eq!(parsed.patch_size, 2);
    assert_eq!(parsed.head_dim(), 128);
}

/// Each of the nine fields is required — a silently-defaulted `depth` or `axes_dim` would build a
/// plausible-looking model with the wrong geometry.
#[test]
fn every_consumed_field_is_required() {
    let mut base: serde_json::Value = serde_json::from_str(TRANSFORMER_CONFIG).unwrap();
    for key in [
        "in_channels",
        "out_channels",
        "context_in_dim",
        "hidden_size",
        "num_heads",
        "depth",
        "axes_dim",
        "checkpoint",
        "patch_size",
    ] {
        let mut without = base.clone();
        without.as_object_mut().unwrap().remove(key);
        let err = MageFlowConfig::from_transformer_config_json(&without.to_string())
            .err()
            .unwrap_or_else(|| panic!("dropping '{key}' must be an error, not a default"))
            .to_string();
        assert!(err.contains(key), "error for missing '{key}' was: {err}");
    }
    // Sanity: the untouched clone still parses, so the loop above is testing removal and not a
    // broken round-trip through `serde_json::Value`.
    base.as_object_mut().unwrap().remove("_class_name");
    MageFlowConfig::from_transformer_config_json(&base.to_string()).unwrap();
}

/// `sum(axes_dim) == head_dim` is the reference's own assertion (`mage_flow.py:70`).
#[test]
fn inconsistent_axes_dim_is_rejected() {
    let bad = r#"{
        "in_channels": 128, "out_channels": 128, "context_in_dim": 2560,
        "hidden_size": 3072, "num_heads": 24, "depth": 12,
        "axes_dim": [16, 56, 55], "checkpoint": false, "patch_size": 1
    }"#;
    let err = MageFlowConfig::from_transformer_config_json(bad)
        .expect_err("axes_dim summing to 127 must be rejected")
        .to_string();
    assert!(err.contains("127"), "{err}");
    assert!(err.contains("128"), "{err}");

    let two_axis = r#"{
        "in_channels": 128, "out_channels": 128, "context_in_dim": 2560,
        "hidden_size": 3072, "num_heads": 24, "depth": 12,
        "axes_dim": [64, 64], "checkpoint": false, "patch_size": 1
    }"#;
    assert!(MageFlowConfig::from_transformer_config_json(two_axis).is_err());
}

/// The published-but-code-hardcoded keys are *verified*, not read. The reference hardcodes them in
/// Python and ignores the JSON, so a checkpoint that disagreed would be silently misinterpreted —
/// this must fail loudly instead.
#[test]
fn drifting_code_hardcoded_keys_are_rejected() {
    // One plausible drift per pinned key. The **architecture selectors** are the consequential
    // half: each of these describes a genuinely different model that the reference — and a naive
    // port — would run as though it had said the opposite, with no shape mismatch to catch it.
    let cases: [(&str, serde_json::Value); 16] = [
        // scalars / shapes
        ("theta", serde_json::json!(5000)),
        ("mlp_ratio", serde_json::json!(2.0)),
        ("static_shift", serde_json::json!(3.0)),
        ("depth_single_blocks", serde_json::json!(2)),
        ("qkv_bias", serde_json::json!(false)),
        ("guidance_embed", serde_json::json!(true)),
        ("txt_max_length", serde_json::json!(4096)),
        // architecture selectors
        ("rope_type", serde_json::json!("3d_rope")),
        ("time_type", serde_json::json!("flux_proj")),
        ("double_block_type", serde_json::json!("single_stream")),
        ("apply_text_rotary_emb", serde_json::json!(true)),
        ("vec_in_dim", serde_json::json!(768)),
        ("vec_type", serde_json::json!("pooled_clip")),
        ("schedule_mode", serde_json::json!("flux")),
        ("use_time_shift", serde_json::json!(true)),
        ("packing", serde_json::json!(false)),
    ];
    assert_eq!(
        cases.len(),
        config::pinned_config_keys().len(),
        "every pinned key needs a drift case"
    );
    for (key, value) in cases {
        let mut v: serde_json::Value = serde_json::from_str(TRANSFORMER_CONFIG).unwrap();
        v.as_object_mut().unwrap().insert(key.to_string(), value);
        let err = MageFlowConfig::from_transformer_config_json(&v.to_string())
            .err()
            .unwrap_or_else(|| panic!("a drifting '{key}' must be rejected"))
            .to_string();
        assert!(err.contains(key), "error for drifting '{key}' was: {err}");
    }

    // A present-but-wrong-TYPED value is a mismatch too, not a skip: an `as_str()` that quietly
    // returns `None` would let `"rope_type": 3` through.
    for (key, value) in [
        ("rope_type", serde_json::json!(3)),
        ("qkv_bias", serde_json::json!("true")),
        ("theta", serde_json::json!("10000")),
        ("vec_type", serde_json::json!("msrope")),
    ] {
        let mut v: serde_json::Value = serde_json::from_str(TRANSFORMER_CONFIG).unwrap();
        v.as_object_mut().unwrap().insert(key.to_string(), value);
        assert!(
            MageFlowConfig::from_transformer_config_json(&v.to_string()).is_err(),
            "a wrong-typed '{key}' must be rejected, not skipped"
        );
    }

    // …and the *published* values for every one of those keys pass, so the check discriminates
    // rather than rejecting everything.
    MageFlowConfig::from_transformer_config_json(TRANSFORMER_CONFIG).unwrap();
}

/// **The coverage guard.** The pinned set must equal the reference's own `_meta` strip-set minus
/// the three entries nothing reads — parsed out of the vendored `pipeline.py`, so the guard cannot
/// silently shrink back to the scalar-only subset it started as.
#[test]
fn pinned_keys_cover_the_references_whole_strip_set() {
    let pipeline = vendored("pipeline.py");
    let body = pipeline
        .split("_meta = {")
        .nth(1)
        .and_then(|s| s.split('}').next())
        .expect("the `_meta` strip-set moved in the vendored pipeline.py");
    let mut meta: Vec<&str> = body
        .split('"')
        .enumerate()
        // Quoted pieces sit at odd indices of a split on `"`.
        .filter_map(|(i, piece)| (i % 2 == 1).then_some(piece))
        .collect();
    meta.sort_unstable();
    meta.dedup();
    assert_eq!(meta.len(), 19, "the strip-set changed size: {meta:?}");

    let mut expected: Vec<&str> = meta
        .iter()
        .copied()
        .filter(|k| !config::INFORMATIONAL_META_KEYS.contains(k))
        .collect();
    expected.sort_unstable();

    let mut pinned = config::pinned_config_keys();
    pinned.sort_unstable();

    assert_eq!(
        pinned,
        expected,
        "pinned_config_keys() must cover the reference's entire `_meta` strip-set except \
         {:?} — an uncovered selector is one the port would silently misinterpret",
        config::INFORMATIONAL_META_KEYS
    );
    // The exclusions are real members of the set, not typos that quietly exclude nothing.
    for key in config::INFORMATIONAL_META_KEYS {
        assert!(meta.contains(key), "{key} is not in the reference `_meta`");
    }
    // And none of the nine CONSUMED fields ended up pinned instead of read.
    for consumed in [
        "in_channels",
        "out_channels",
        "context_in_dim",
        "hidden_size",
        "num_heads",
        "depth",
        "axes_dim",
        "checkpoint",
        "patch_size",
    ] {
        assert!(
            !pinned.contains(&consumed),
            "{consumed} must be READ, not pinned"
        );
        assert!(!meta.contains(&consumed), "{consumed} must not be stripped");
    }
}

/// The constants this crate hardcodes must match what the published file actually says.
#[test]
fn hardcoded_constants_match_the_published_json() {
    let t: serde_json::Value = serde_json::from_str(TRANSFORMER_CONFIG).unwrap();
    assert_eq!(t["theta"].as_f64().unwrap() as f32, config::ROPE_THETA);
    assert_eq!(t["mlp_ratio"].as_f64().unwrap() as f32, config::MLP_RATIO);
    assert_eq!(
        t["static_shift"].as_f64().unwrap() as f32,
        config::STATIC_SHIFT
    );
    assert_eq!(
        t["depth_single_blocks"].as_u64().unwrap() as usize,
        config::DEPTH_SINGLE_BLOCKS
    );
    assert_eq!(t["qkv_bias"].as_bool().unwrap(), config::QKV_BIAS);
    assert_eq!(
        t["guidance_embed"].as_bool().unwrap(),
        config::GUIDANCE_EMBED
    );
    // The architecture selectors — read from the file into the constants, not restated. These
    // identify the DiT as a reparameterised Z-Image S3-DiT and carry the two flags the RoPE and
    // attention ports depend on.
    assert_eq!(t["rope_type"].as_str().unwrap(), config::ROPE_TYPE);
    assert_eq!(t["time_type"].as_str().unwrap(), config::TIME_TYPE);
    assert_eq!(
        t["double_block_type"].as_str().unwrap(),
        config::DOUBLE_BLOCK_TYPE
    );
    assert_eq!(t["schedule_mode"].as_str().unwrap(), config::SCHEDULE_MODE);
    assert_eq!(
        t["apply_text_rotary_emb"].as_bool().unwrap(),
        config::APPLY_TEXT_ROTARY_EMB
    );
    assert_eq!(
        t["use_time_shift"].as_bool().unwrap(),
        config::USE_TIME_SHIFT
    );
    assert_eq!(t["packing"].as_bool().unwrap(), config::PACKING);
    assert_eq!(t["vec_in_dim"].as_i64().unwrap() as i32, config::VEC_IN_DIM);
    assert_eq!(t["vec_type"].as_str(), config::VEC_TYPE);
    assert!(t["vec_type"].is_null());

    let vae: serde_json::Value = serde_json::from_str(VAE_CONFIG).unwrap();
    assert_eq!(
        vae["latent_channels"].as_i64().unwrap() as i32,
        config::LATENT_CHANNELS
    );
    assert_eq!(
        vae["downsample_factor"].as_u64().unwrap() as u32,
        config::VAE_DOWNSAMPLE_FACTOR
    );
    assert_eq!(
        vae["sample_posterior"].as_bool().unwrap(),
        config::VAE_CONFIG_SAMPLE_POSTERIOR
    );
    // The DiT's latent width is the VAE's latent width — the two configs must agree.
    assert_eq!(
        MageFlowConfig::mage_flow().in_channels,
        config::LATENT_CHANNELS
    );
    // …and there is no scale/shift between them.
    assert!(config::LATENT_SCALE_SHIFT.is_none());

    let sched: serde_json::Value = serde_json::from_str(SCHEDULER_CONFIG).unwrap();
    assert_eq!(
        sched["num_train_timesteps"].as_u64().unwrap() as u32,
        config::NUM_TRAIN_TIMESTEPS
    );
    assert_eq!(
        sched["shift"].as_f64().unwrap() as f32,
        config::STATIC_SHIFT
    );
    assert_eq!(
        sched["use_dynamic_shifting"].as_bool().unwrap(),
        config::USE_DYNAMIC_SHIFTING
    );
}

// -------------------------------------------------------------------------------------------
// Constants pinned against the VENDORED FROZEN REFERENCE rather than the weight configs.
// -------------------------------------------------------------------------------------------

/// The vendored `microsoft/Mage @ df7f84d9` copy that is this crate's architectural source of
/// truth. Reading it here turns "the templates were transcribed correctly" from a claim into a
/// check — a hand-typed chat template that is one space or one token off silently shifts
/// `drop_idx` and corrupts every conditioning tensor.
fn vendored(relative: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../_vendor/mage_flow")
        .join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("vendored reference {} is unreadable: {e}", path.display()))
}

/// Rebuild a Python implicit-concatenation string literal: collect the double-quoted pieces in
/// `source`, concatenate them, and interpret the `\n` escapes both languages spell identically.
fn python_literal(source: &str) -> String {
    let mut out = String::new();
    let mut rest = source;
    while let Some(open) = rest.find('"') {
        rest = &rest[open + 1..];
        let close = rest.find('"').expect("unterminated python string piece");
        out.push_str(&rest[..close]);
        rest = &rest[close + 1..];
    }
    out.replace("\\n", "\n")
}

/// Pull `PROMPT_TEMPLATE[key]`'s `(template, start_idx)` out of the vendored `utils.py`.
fn vendored_prompt_template(utils: &str, key: &str) -> (String, usize) {
    let entry = utils
        .split(&format!("\"{key}\": {{"))
        .nth(1)
        .unwrap_or_else(|| panic!("PROMPT_TEMPLATE[{key:?}] not found in the vendored utils.py"));
    let template_src = entry
        .split("\"template\"")
        .nth(1)
        .and_then(|s| s.split("\"start_idx\"").next())
        .expect("no \"template\" before \"start_idx\"");
    let start_idx = entry
        .split("\"start_idx\":")
        .nth(1)
        .and_then(|s| s.split(',').next())
        .map(str::trim)
        .and_then(|s| s.parse::<usize>().ok())
        .expect("start_idx is not an integer");
    (python_literal(template_src), start_idx)
}

#[test]
fn prompt_templates_and_drop_indices_match_the_vendored_reference() {
    let utils = vendored("models/utils.py");
    let (gen_template, gen_drop) = vendored_prompt_template(&utils, "mage-flow");
    let (edit_template, edit_drop) = vendored_prompt_template(&utils, "mage-flow-edit");

    assert_eq!(config::PROMPT_TEMPLATE_GEN, gen_template);
    assert_eq!(config::PROMPT_TEMPLATE_EDIT, edit_template);
    assert_eq!(config::DROP_IDX_GEN, gen_drop);
    assert_eq!(config::DROP_IDX_EDIT, edit_drop);

    // The two entries really are distinct — otherwise both pairs of assertions above could hold
    // against a single mis-spliced block.
    assert_ne!(config::PROMPT_TEMPLATE_GEN, config::PROMPT_TEMPLATE_EDIT);
    assert_ne!(config::DROP_IDX_GEN, config::DROP_IDX_EDIT);
    assert!(config::PROMPT_TEMPLATE_GEN.contains("{}"));
    assert!(config::PROMPT_TEMPLATE_EDIT.contains("{}"));
    // The splice really did reassemble a multi-piece literal, not just the first fragment.
    assert!(config::PROMPT_TEMPLATE_GEN.ends_with("<|im_start|>assistant\n"));
    assert!(config::PROMPT_TEMPLATE_EDIT.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn watermark_constants_match_the_vendored_reference() {
    let latent = vendored("models/modules/mage_latent.py");
    assert!(
        latent.contains(&format!("DEFAULT_GS_KEY = {}\n", config::GS_DEFAULT_KEY)),
        "GS_DEFAULT_KEY {} not found in mage_latent.py",
        config::GS_DEFAULT_KEY
    );
    assert!(
        latent.contains(&format!("DEFAULT_GS_PAYLOAD = \"{}\"", config::GS_PAYLOAD)),
        "GS_PAYLOAD {:?} not found in mage_latent.py",
        config::GS_PAYLOAD
    );
    assert!(
        latent.contains(&format!("_MSG_BITS = {}\n", config::GS_MESSAGE_BITS)),
        "GS_MESSAGE_BITS {} not found in mage_latent.py",
        config::GS_MESSAGE_BITS
    );
}

#[test]
fn rope_and_ffn_corrections_are_pinned_to_the_reference() {
    let layers = vendored("models/modules/mage_layers.py");
    let dit = vendored("models/mage_flow.py");

    // theta is hardcoded in code, NOT read from `transformer/config.json`.
    assert!(
        dit.contains(&format!(
            "MageFlowEmbedRope(theta={}, axes_dim=self.axes_dim, scale_rope=True)",
            config::ROPE_THETA as i64
        )),
        "the hardcoded msrope construction moved; re-verify ROPE_THETA / SCALE_ROPE"
    );
    assert_eq!(
        config::SCALE_ROPE,
        dit.contains("scale_rope=True"),
        "SCALE_ROPE must track the reference's centred height/width coordinates"
    );
    assert!(
        layers.contains(&format!("torch.arange({})", config::ROPE_TABLE_LEN)),
        "the {}-entry msrope table moved",
        config::ROPE_TABLE_LEN
    );

    // The correction that matters most: gelu-approximate, NOT the z-image sibling's SwiGLU.
    assert!(
        layers.contains(&format!("activation_fn=\"{}\"", config::FFN_ACTIVATION)),
        "the DiT FFN activation moved; it must stay {:?} and must NOT become SwiGLU",
        config::FFN_ACTIVATION
    );
    assert!(!layers.contains("SwiGLU"));

    // The pooled text vector is discarded by the DiT.
    assert!(dit.contains("txt_vec = torch.zeros("));
}

// -------------------------------------------------------------------------------------------
// Text-encoder constants — pinned against the published `text_encoder/config.json` values.
// -------------------------------------------------------------------------------------------

/// Every `QwenVlTextConfig` field and every TE scalar read out of the **byte-pinned published
/// file**, not retyped literals. Previously this block was the one component config held to a
/// weaker standard than the other three.
#[test]
fn text_encoder_config_is_the_published_qwen3_vl_lm() {
    let root: serde_json::Value = serde_json::from_str(TEXT_ENCODER_CONFIG).unwrap();
    let text = &root["text_config"];
    let te = QwenVlTextConfig::mage_flow();

    assert_eq!(text["hidden_size"].as_i64().unwrap() as i32, te.hidden_size);
    assert_eq!(
        text["num_hidden_layers"].as_u64().unwrap() as usize,
        te.num_layers
    );
    assert_eq!(
        text["num_attention_heads"].as_i64().unwrap() as i32,
        te.num_attention_heads
    );
    assert_eq!(
        text["num_key_value_heads"].as_i64().unwrap() as i32,
        te.num_key_value_heads
    );
    assert_eq!(text["head_dim"].as_i64().unwrap() as i32, te.head_dim);
    assert_eq!(
        text["intermediate_size"].as_i64().unwrap() as i32,
        te.intermediate_size
    );
    assert_eq!(text["vocab_size"].as_i64().unwrap() as i32, te.vocab_size);
    assert_eq!(
        text["rope_scaling"]["mrope_section"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        te.mrope_section.to_vec()
    );
    assert_eq!(text["attention_bias"].as_bool().unwrap(), te.attention_bias);
    assert_eq!(
        text["tie_word_embeddings"].as_bool().unwrap(),
        te.tie_word_embeddings
    );

    // The three free-standing TE scalars, from the same file.
    assert_eq!(
        text["rms_norm_eps"].as_f64().unwrap() as f32,
        config::TE_RMS_NORM_EPS
    );
    assert_eq!(text["rope_theta"].as_f64().unwrap(), config::TE_ROPE_THETA);
    assert_eq!(text["hidden_act"].as_str().unwrap(), config::TE_HIDDEN_ACT);
    // …and the interleaved-M-RoPE flag the section split is meaningless without.
    assert_eq!(text["rope_scaling"]["mrope_interleaved"], true);

    // head_dim is DECOUPLED from hidden_size / num_heads (32 × 128 = 4096 ≠ 2560) — the single
    // most likely thing for a port to derive instead of read.
    assert_eq!(te.head_dim, 128);
    assert_ne!(te.head_dim, te.hidden_size / te.num_attention_heads);
    // The DiT's text-conditioning width is the LM's hidden size.
    assert_eq!(MageFlowConfig::mage_flow().context_in_dim, te.hidden_size);
    // The vision tower (sc-14048) is deliberately not modelled yet, but the file must still be the
    // Qwen3-VL one those constants came from.
    assert_eq!(root["architectures"][0], "Qwen3VLForConditionalGeneration");
    assert_eq!(root["vision_config"]["out_hidden_size"], 2560);
}

/// [`config::TXT_MAX_LENGTH`] and the `+ drop_idx` term. The budget is published, the term is not,
/// and every parity golden uses a short prompt — so nothing else would catch either mistake.
#[test]
fn prompt_truncation_budget_matches_the_published_config_and_the_reference() {
    let t: serde_json::Value = serde_json::from_str(TRANSFORMER_CONFIG).unwrap();
    assert_eq!(
        t["txt_max_length"].as_u64().unwrap() as usize,
        config::TXT_MAX_LENGTH,
        "TXT_MAX_LENGTH must be the published budget, not the reference dataclass default (4096)"
    );
    // The misleading default really is 4096 and really is overridden — assert both halves, so the
    // doc comment warning about it cannot rot.
    let dit = vendored("models/mage_flow.py");
    assert!(dit.contains("txt_max_length: int = Field(default=4096)"));
    let pipeline = vendored("pipeline.py");
    assert!(pipeline.contains(&format!(
        "txt_max_length=tcfg.get(\"txt_max_length\", {})",
        config::TXT_MAX_LENGTH
    )));

    // The `+ drop_idx` term (`pipeline.py:225`).
    assert!(
        pipeline.contains("max_len = model.txt_enc.tokenizer_max_length + drop_idx"),
        "the `+ drop_idx` truncation term moved; re-verify max_prompt_tokens"
    );
    assert_eq!(
        config::max_prompt_tokens(config::DROP_IDX_GEN),
        config::TXT_MAX_LENGTH + 34
    );
    assert_eq!(
        config::max_prompt_tokens(config::DROP_IDX_EDIT),
        config::TXT_MAX_LENGTH + 64
    );
    assert_eq!(config::max_prompt_tokens(config::DROP_IDX_GEN), 2082);
    assert_eq!(config::max_prompt_tokens(config::DROP_IDX_EDIT), 2112);
    // Both budgets leave exactly TXT_MAX_LENGTH conditioning tokens after the drop.
    assert_ne!(
        config::max_prompt_tokens(config::DROP_IDX_GEN),
        config::max_prompt_tokens(config::DROP_IDX_EDIT)
    );
}

/// The timestep-embedder block, pinned against `mage_layers.py`. sc-14040 consumes these five
/// values verbatim and no published config carries any of them.
#[test]
fn timestep_embedder_constants_are_pinned_to_the_reference() {
    let layers = vendored("models/modules/mage_layers.py");

    // One construction line carries four of the five (`mage_layers.py:93`).
    let ctor = format!(
        "Timesteps(num_channels={}, flip_sin_to_cos={}, downscale_freq_shift={}, scale={})",
        config::FREQUENCY_EMBEDDING_SIZE,
        if config::TIMESTEP_FLIP_SIN_TO_COS {
            "True"
        } else {
            "False"
        },
        config::TIMESTEP_DOWNSCALE_FREQ_SHIFT as i64,
        config::TIMESTEP_SCALE as i64,
    );
    assert!(
        layers.contains(&ctor),
        "timestep projection changed; expected to find `{ctor}`"
    );
    // The projection width must match what `TimestepEmbedding` consumes (`mage_layers.py:94`).
    assert!(layers.contains(&format!(
        "TimestepEmbedding(in_channels={}, time_embed_dim=embedding_dim)",
        config::FREQUENCY_EMBEDDING_SIZE
    )));
    // `max_period` is the signature default of `get_timestep_embedding` (`mage_layers.py:30`) —
    // never overridden at the call site, which is exactly why it is easy to mis-transcribe.
    // NOTE the trailing comma: without it, `10000 -> 1000` still matches as a PREFIX and the check
    // silently passes. (Mutation testing caught exactly that.)
    assert!(
        layers.contains(&format!(
            "max_period: int = {},",
            config::TIMESTEP_MAX_PERIOD as i64
        )),
        "max_period moved; expected {}",
        config::TIMESTEP_MAX_PERIOD as i64
    );
    // The deliberate bf16 downcast of the frequency table (`mage_layers.py:45`).
    assert_eq!(
        config::TIMESTEP_FREQS_BF16,
        layers.contains("emb = torch.exp(exponent).to(timesteps.dtype)"),
        "TIMESTEP_FREQS_BF16 must track the reference's dtype downcast"
    );
}

/// The remaining DiT scalars/flags with no published-config home.
#[test]
fn dit_norm_and_attention_order_are_pinned_to_the_reference() {
    let dit = vendored("models/mage_flow.py");
    let layers = vendored("models/modules/mage_layers.py");

    // NORM_EPS at both DiT sites, and as the block default. Rendered from the constant, so a
    // mutation to 1e-5 stops matching the reference text.
    let eps = format!("{:e}", config::NORM_EPS);
    assert!(
        dit.contains(&format!("RMSNorm(params.context_in_dim, eps={eps})")),
        "txt_norm eps moved; expected eps={eps}"
    );
    assert!(
        dit.contains(&format!("elementwise_affine=False, eps={eps})")),
        "norm_out eps moved; expected eps={eps}"
    );
    assert!(
        layers.contains(&format!("eps: float = {eps},")),
        "the transformer block's eps default moved; expected {eps}"
    );

    // Joint-attention order: [text, image]. The reference expresses it as scatter offsets, not a
    // `cat` — text lands at each sample's start, image is offset by that sample's text length
    // (`mage_layers.py:466-467`). Assert the *code*, not the commented-out `cat` above it.
    assert_eq!(
        config::TEXT_STREAM_FIRST,
        layers.contains("img_dest_indices = joint_cu_lens[img_sample_ids] + txt_lens[img_sample_ids] + img_intra_pos"),
        "TEXT_STREAM_FIRST must track the reference's joint-attention scatter offsets"
    );
    assert!(
        layers.contains("txt_dest_indices = joint_cu_lens[txt_sample_ids] + txt_intra_pos"),
        "the text stream must start at each sample's offset with no image shift"
    );
}

/// The edit-path VL conditioning cap — a keyword-argument default, invisible to every config file.
#[test]
fn vl_cond_long_edge_is_pinned_to_the_reference() {
    let pipeline = vendored("pipeline.py");
    assert!(
        pipeline.contains(&format!("vl_cond_long_edge={},", config::VL_COND_LONG_EDGE)),
        "the VL long-edge cap moved; expected {}",
        config::VL_COND_LONG_EDGE
    );
    assert!(pipeline.contains("_resize_long_edge(p, vl_cond_long_edge)"));
}

/// The native-resolution envelope, pinned against the reference's own README rather than the epic.
#[test]
fn native_resolution_bounds_are_pinned_to_the_reference() {
    let readme = vendored("README.md");
    assert!(
        readme.contains(&format!(
            "generates from **{} to {}** on any aspect ratio",
            config::MIN_SIZE,
            config::MAX_SIZE
        )),
        "the documented native-resolution range moved; expected {}-{}",
        config::MIN_SIZE,
        config::MAX_SIZE
    );
    let pipeline = vendored("pipeline.py");
    assert!(
        pipeline.contains(&format!(
            "def _make_divisible_by_{}(size: int) -> int:",
            config::SIZE_MULTIPLE
        )),
        "the geometry stride moved; expected a multiple of {}",
        config::SIZE_MULTIPLE
    );
}

#[test]
fn geometry_constants_follow_the_vae_stride() {
    // Bound to locals so this stays a runtime check rather than a compile-time tautology.
    let (multiple, min, max) = (config::SIZE_MULTIPLE, config::MIN_SIZE, config::MAX_SIZE);
    assert_eq!(multiple, config::VAE_DOWNSAMPLE_FACTOR);
    assert_eq!(min % multiple, 0);
    assert_eq!(max % multiple, 0);
    assert!(min < max);
    // Every descriptor's advertised bounds are these constants — the worker pins resolution
    // buckets to the engine stride, so a drift here silently widens what the family accepts.
    for variant in mlx_gen_mage::MageVariant::ALL {
        let caps = mlx_gen_mage::descriptor_for(variant).capabilities;
        assert_eq!(
            (caps.min_size, caps.max_size),
            (min, max),
            "{}",
            variant.id()
        );
    }
}
