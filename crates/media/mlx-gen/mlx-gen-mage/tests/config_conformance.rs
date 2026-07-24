//! The scaffold's load-bearing gate (sc-14037): every constant in `mlx_gen_mage::config` is
//! checked against a **committed, byte-pinned copy of the real published config** or against the
//! **vendored frozen reference** — never against itself.
//!
//! Fixtures under `tests/fixtures/` are verbatim copies of
//! `microsoft/Mage-Flow @ 9f46d09d`'s `transformer/config.json`, `vae/config.json` and
//! `scheduler/scheduler_config.json`. All six Mage-Flow repositories ship these three files
//! byte-identically (independently hashed across `Mage-Flow`, `-Base`, `-Turbo`, `-Edit`,
//! `-Edit-Base`, `-Edit-Turbo`), so pinning one copy pins the family.

use mlx_gen_mage::config::{self, MageFlowConfig, QwenVlTextConfig};
use sha2::{Digest, Sha256};

const TRANSFORMER_CONFIG: &str = include_str!("fixtures/transformer_config.json");
const VAE_CONFIG: &str = include_str!("fixtures/vae_config.json");
const SCHEDULER_CONFIG: &str = include_str!("fixtures/scheduler_config.json");

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
    let cases: [(&str, serde_json::Value); 6] = [
        ("theta", serde_json::json!(5000)),
        ("mlp_ratio", serde_json::json!(2.0)),
        ("static_shift", serde_json::json!(3.0)),
        ("depth_single_blocks", serde_json::json!(2)),
        ("qkv_bias", serde_json::json!(false)),
        ("guidance_embed", serde_json::json!(true)),
    ];
    for (key, value) in cases {
        let mut v: serde_json::Value = serde_json::from_str(TRANSFORMER_CONFIG).unwrap();
        v.as_object_mut().unwrap().insert(key.to_string(), value);
        let err = MageFlowConfig::from_transformer_config_json(&v.to_string())
            .err()
            .unwrap_or_else(|| panic!("a drifting '{key}' must be rejected"))
            .to_string();
        assert!(err.contains(key), "error for drifting '{key}' was: {err}");
    }
    // …and the *published* values for those same keys pass, so the check discriminates rather
    // than rejecting everything.
    MageFlowConfig::from_transformer_config_json(TRANSFORMER_CONFIG).unwrap();
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
    // The declarations that identify this DiT as a reparameterised Z-Image S3-DiT, and the two
    // flags the RoPE port depends on.
    assert_eq!(t["rope_type"], "msrope");
    assert_eq!(t["time_type"], "qwen_proj");
    assert_eq!(t["double_block_type"], "double_stream");
    assert_eq!(t["schedule_mode"], "z-image");
    assert_eq!(t["apply_text_rotary_emb"], false);
    assert_eq!(t["use_time_shift"], false);

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
        latent.contains(&format!("DEFAULT_GS_KEY = {}", config::GS_DEFAULT_KEY)),
        "GS_DEFAULT_KEY {} not found in mage_latent.py",
        config::GS_DEFAULT_KEY
    );
    assert!(
        latent.contains(&format!("DEFAULT_GS_PAYLOAD = \"{}\"", config::GS_PAYLOAD)),
        "GS_PAYLOAD {:?} not found in mage_latent.py",
        config::GS_PAYLOAD
    );
    assert!(
        latent.contains(&format!("_MSG_BITS = {}", config::GS_MESSAGE_BITS)),
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

#[test]
fn text_encoder_config_is_the_published_qwen3_vl_lm() {
    let te = QwenVlTextConfig::mage_flow();
    assert_eq!(te.hidden_size, 2560);
    assert_eq!(te.num_layers, 36);
    assert_eq!(te.num_attention_heads, 32);
    assert_eq!(te.num_key_value_heads, 8);
    assert_eq!(te.intermediate_size, 9728);
    assert_eq!(te.vocab_size, 151_936);
    assert_eq!(te.mrope_section, [24, 20, 20]);
    assert!(!te.attention_bias);
    assert!(te.tie_word_embeddings);
    // head_dim is DECOUPLED from hidden_size / num_heads (32 × 128 = 4096 ≠ 2560) — the single
    // most likely thing for a port to derive instead of read.
    assert_eq!(te.head_dim, 128);
    assert_ne!(te.head_dim, te.hidden_size / te.num_attention_heads);
    // The DiT's text-conditioning width is the LM's hidden size.
    assert_eq!(MageFlowConfig::mage_flow().context_in_dim, te.hidden_size);
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
