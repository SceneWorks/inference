//! Frozen target tensor schema and original shard identities from the release index.
//! The optional training/MTP companion is outside the conditional generation model.

pub const TARGET_SHARDS: &[(&str, &str)] = &[
    (
        "model-00001.safetensors",
        "8e7d17dd41efe79746bcff1e51a892ec6bf8092887cba88827cefc77b19b3fd3",
    ),
    (
        "model-00002.safetensors",
        "8fee2d7786386d45639a612ee11642d047eeaa0606b4225123c7aff396b80673",
    ),
    (
        "model-00003.safetensors",
        "85dca81a009c5a814855ad026ada6959d509cd0175b6880cc2c89b6a135a45c7",
    ),
    (
        "model-00004.safetensors",
        "e682478f625e6786bf2d428404d259dcbb97d65765e03798a09cbac9a84e0716",
    ),
    (
        "model-00005.safetensors",
        "4b6b528980d7ebc8b976bd0092552758fd87f1cb5cfd589d573cb169f1f32f05",
    ),
    (
        "model-00006.safetensors",
        "f736a6d35716ffed5f9033d733c21f92f1677902755a15adea44bd3fbe2a50c3",
    ),
    (
        "model-00007.safetensors",
        "89b8928cb35915fbbbf8ab50ffd1dad58c72312e29bb839b3337ee7e562e6aac",
    ),
    (
        "model-00008.safetensors",
        "42bc4726d6f3028f1a8a074f11fe3de82fc119687d69cc07fbc0885cdeedbdaa",
    ),
    (
        "model-00009.safetensors",
        "a1d12fbdc13785f5f9af34d7cb26ccd93b5761429cb4ea7067aead70056ae147",
    ),
    (
        "model-00010.safetensors",
        "ab662a9fe6583c4cea66bbdc033fb243ba71a55fe9b5155c5bdcdd854701cf97",
    ),
    (
        "model-00011.safetensors",
        "3bd59cddbf2dee7a2d33d14bfa0dcfaa1d2356f37be64a2164789393b752d31d",
    ),
];

type TensorSpec = (&'static str, &'static [usize], &'static str);

const DECODER: &[TensorSpec] = &[
    ("input_layernorm.weight", &[4096], "BF16"),
    ("mlp.down_proj.weight", &[4096, 12288], "BF16"),
    ("mlp.gate_proj.weight", &[12288, 4096], "BF16"),
    ("mlp.up_proj.weight", &[12288, 4096], "BF16"),
    ("post_attention_layernorm.weight", &[4096], "BF16"),
];

const FULL: &[TensorSpec] = &[
    ("self_attn.k_norm.weight", &[256], "BF16"),
    ("self_attn.k_proj.weight", &[1024, 4096], "BF16"),
    ("self_attn.o_proj.weight", &[4096, 4096], "BF16"),
    ("self_attn.q_norm.weight", &[256], "BF16"),
    ("self_attn.q_proj.weight", &[8192, 4096], "BF16"),
    ("self_attn.v_proj.weight", &[1024, 4096], "BF16"),
];

const LINEAR: &[TensorSpec] = &[
    ("linear_attn.A_log", &[32], "F32"),
    ("linear_attn.conv1d.weight", &[8192, 1, 4], "BF16"),
    ("linear_attn.dt_bias", &[32], "F32"),
    ("linear_attn.in_proj_a.weight", &[32, 4096], "BF16"),
    ("linear_attn.in_proj_b.weight", &[32, 4096], "BF16"),
    ("linear_attn.in_proj_qkv.weight", &[8192, 4096], "BF16"),
    ("linear_attn.in_proj_z.weight", &[4096, 4096], "BF16"),
    ("linear_attn.norm.weight", &[128], "BF16"),
    ("linear_attn.out_proj.weight", &[4096, 4096], "BF16"),
];

const MHC: &[TensorSpec] = &[
    ("mhc_attn.bias", &[8], "F32"),
    ("mhc_attn.weight", &[8, 8192], "F32"),
    ("mhc_ffn.bias", &[8], "F32"),
    ("mhc_ffn.weight", &[8, 8192], "F32"),
];

const ROOT: &[TensorSpec] = &[
    ("lm_head.weight", &[248320, 4096], "BF16"),
    (
        "model.language_model.embed_tokens.weight",
        &[248320, 4096],
        "BF16",
    ),
    ("model.language_model.loop_gate.logits", &[4096], "F32"),
    ("model.language_model.norm.weight", &[4096], "BF16"),
    ("model.visual.merger.linear_fc1.bias", &[4608], "BF16"),
    (
        "model.visual.merger.linear_fc1.weight",
        &[4608, 4608],
        "BF16",
    ),
    ("model.visual.merger.linear_fc2.bias", &[4096], "BF16"),
    (
        "model.visual.merger.linear_fc2.weight",
        &[4096, 4608],
        "BF16",
    ),
    ("model.visual.merger.norm.bias", &[1152], "BF16"),
    ("model.visual.merger.norm.weight", &[1152], "BF16"),
    ("model.visual.patch_embed.proj.bias", &[1152], "BF16"),
    (
        "model.visual.patch_embed.proj.weight",
        &[1152, 3, 2, 16, 16],
        "BF16",
    ),
    ("model.visual.pos_embed.weight", &[2304, 1152], "BF16"),
];

const VISUAL: &[TensorSpec] = &[
    ("attn.proj.bias", &[1152], "BF16"),
    ("attn.proj.weight", &[1152, 1152], "BF16"),
    ("attn.qkv.bias", &[3456], "BF16"),
    ("attn.qkv.weight", &[3456, 1152], "BF16"),
    ("mlp.linear_fc1.bias", &[4304], "BF16"),
    ("mlp.linear_fc1.weight", &[4304, 1152], "BF16"),
    ("mlp.linear_fc2.bias", &[1152], "BF16"),
    ("mlp.linear_fc2.weight", &[1152, 4304], "BF16"),
    ("norm1.bias", &[1152], "BF16"),
    ("norm1.weight", &[1152], "BF16"),
    ("norm2.bias", &[1152], "BF16"),
    ("norm2.weight", &[1152], "BF16"),
];

/// Enumerate the original BF16 release layout (including its retained FP32 tensors).
pub fn target_tensors() -> impl Iterator<Item = (String, &'static [usize], &'static str)> {
    let mut result = Vec::new();
    let mut group = |prefix: &str, entries: &[TensorSpec]| {
        result.extend(
            entries
                .iter()
                .map(|&(suffix, shape, dtype)| (format!("{prefix}{suffix}"), shape, dtype)),
        );
    };
    group("", ROOT);
    for i in 0..32 {
        let prefix = format!("model.language_model.layers.{i}.");
        group(&prefix, DECODER);
        group(&prefix, if i % 4 == 3 { FULL } else { LINEAR });
        if (4..28).contains(&i) {
            group(&prefix, MHC);
        }
    }
    for i in 0..27 {
        group(&format!("model.visual.blocks.{i}."), VISUAL);
    }
    result.into_iter()
}
