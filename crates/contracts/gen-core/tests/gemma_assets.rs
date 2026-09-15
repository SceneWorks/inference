//! sc-18762 — the self-contained Gemma text-encoder assets + the exactly-one-BOS tokenizer policy.
//!
//! Three layers of coverage:
//!
//! 1. **Synthetic packed files** (always run). A hand-built safetensors exercises every unpack
//!    path, including the failure modes that must be loud: absent metadata, absent tensor, absent
//!    required sidecar, a float-dtype asset, offsets past EOF, and a truncated file. The scar this
//!    guards is upstream's `docs/ltx-2.5-gemma4-missing-tensors.md` — 11 tensors silently landing
//!    at random init because a non-strict load swallowed a wrong key remap.
//! 2. **Tiny real tokenizers** (always run). Two committed HF fast tokenizers that differ only in
//!    whether the `post_processor` emits `<bos>` — the Gemma 4 and Gemma 3 shapes — so both BOS
//!    bugs are covered in CI without a 32 MB vocabulary.
//! 3. **Real-asset parity** (`#[ignore]`, opt-in via env). Token-id goldens produced by the
//!    upstream Python path over the actual shipped tokenizers; see
//!    `tests/fixtures/gen_ltx_gemma_token_parity.py`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use gen_core::gemma_assets::{
    flatten_gemma4_unified_key, is_gemma_asset_key, GemmaAssets, GemmaTeKeyMap, LtxGemmaTokenizer,
    GEMMA_CONFIG_METADATA_KEY, HF_ASSET_TENSOR_PREFIX, TOKENIZER_JSON_TENSOR_KEY,
};
use gen_core::tokenizer::ensure_single_leading_bos;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// Env var naming the real single-file LTX-2.5 text encoder for the `#[ignore]`d parity tests.
const REAL_TE_ENV: &str = "LTX25_TE_SAFETENSORS";
/// Env var naming a real gemma-3-12b-it snapshot directory (the LTX-2.3 `gemma/` sub-dir).
const REAL_GEMMA3_ENV: &str = "LTX23_GEMMA_DIR";

const MINIMAL_CONFIG: &str =
    r#"{"model_type":"gemma4_unified","gemma_version":"gemma4-12b-ltx-v1"}"#;

// =================================================================================================
// Synthetic packed-safetensors builder
// =================================================================================================

/// One tensor to pack: `(dtype string, shape, raw little-endian payload bytes)`.
struct PackedTensor {
    dtype: &'static str,
    shape: Vec<usize>,
    payload: Vec<u8>,
    /// Byte length the header *claims*, when it must differ from what is actually written (used to
    /// forge a range running past EOF). `None` = claim the payload length.
    declared_bytes: Option<usize>,
}

impl PackedTensor {
    fn new(dtype: &'static str, shape: Vec<usize>, payload: Vec<u8>) -> Self {
        Self {
            dtype,
            shape,
            payload,
            declared_bytes: None,
        }
    }

    fn u8(bytes: &[u8]) -> Self {
        Self::new("U8", vec![bytes.len()], bytes.to_vec())
    }

    /// Widen each byte to a `width`-byte little-endian element, as a ComfyUI pack storing an
    /// asset at a non-`uint8` integer dtype does. The high bytes are deliberately non-zero so a
    /// reader that concatenates raw bytes instead of taking the low byte of each element fails.
    fn widened(dtype: &'static str, width: usize, bytes: &[u8]) -> Self {
        let mut payload = Vec::with_capacity(bytes.len() * width);
        for (index, byte) in bytes.iter().enumerate() {
            payload.push(*byte);
            for high in 1..width {
                payload.push(((index + high) % 251) as u8);
            }
        }
        Self::new(dtype, vec![bytes.len()], payload)
    }
}

/// Write a safetensors file: 8-byte LE header length, JSON header, then the contiguous payloads.
fn write_packed(
    path: &Path,
    metadata: &BTreeMap<String, String>,
    tensors: &BTreeMap<String, PackedTensor>,
) {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert(
            "__metadata__".into(),
            serde_json::Value::Object(
                metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                    .collect(),
            ),
        );
    }
    let mut offset = 0usize;
    let mut data = Vec::new();
    for (name, tensor) in tensors {
        let end = offset + tensor.declared_bytes.unwrap_or(tensor.payload.len());
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            }),
        );
        data.extend_from_slice(&tensor.payload);
        offset = end;
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header json");
    let mut file = std::fs::File::create(path).expect("create packed file");
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("write header len");
    file.write_all(&header).expect("write header");
    file.write_all(&data).expect("write data");
}

fn tiny_tokenizer_json(name: &str) -> Vec<u8> {
    std::fs::read(PathBuf::from(FIXTURES).join(name)).expect("tiny tokenizer fixture")
}

fn tokenizer_config(pad: &str) -> String {
    format!(
        r#"{{"bos_token":"<bos>","eos_token":"<eos>","pad_token":{pad},"tokenizer_class":"GemmaTokenizer"}}"#
    )
}

/// A complete, valid packed text encoder: config metadata, `tokenizer_json`, both required
/// sidecars, and one weight tensor so the file is not asset-only.
fn valid_pack(
    tokenizer_fixture: &str,
) -> (BTreeMap<String, String>, BTreeMap<String, PackedTensor>) {
    let metadata = BTreeMap::from([
        (
            GEMMA_CONFIG_METADATA_KEY.to_string(),
            MINIMAL_CONFIG.to_string(),
        ),
        ("format".to_string(), "pt".to_string()),
    ]);
    let tensors = BTreeMap::from([
        (
            TOKENIZER_JSON_TENSOR_KEY.to_string(),
            PackedTensor::u8(&tiny_tokenizer_json(tokenizer_fixture)),
        ),
        (
            format!("{HF_ASSET_TENSOR_PREFIX}tokenizer_config.json"),
            PackedTensor::u8(tokenizer_config("\"<pad>\"").as_bytes()),
        ),
        (
            format!("{HF_ASSET_TENSOR_PREFIX}processor_config.json"),
            PackedTensor::u8(br#"{"processor_class":"Gemma4Processor"}"#),
        ),
        (
            format!("{HF_ASSET_TENSOR_PREFIX}chat_template.jinja"),
            PackedTensor::u8(b"{{ messages }}"),
        ),
        (
            "model.embed_tokens.weight".to_string(),
            PackedTensor::u8(&[1, 2, 3, 4]),
        ),
    ]);
    (metadata, tensors)
}

fn temp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

// =================================================================================================
// Unpacking the packed single-file text encoder
// =================================================================================================

#[test]
fn packed_single_file_round_trips_config_tokenizer_and_sidecars() {
    let dir = temp_dir();
    let path = dir.path().join("te.safetensors");
    let (metadata, tensors) = valid_pack("tiny_gemma4_tokenizer.json");
    write_packed(&path, &metadata, &tensors);

    let assets = GemmaAssets::from_single_file(&path).expect("load packed assets");
    assert_eq!(assets.config_json(), MINIMAL_CONFIG);
    assert_eq!(
        assets.config_model_type().expect("model_type"),
        "gemma4_unified"
    );
    assert_eq!(
        assets.tokenizer_json(),
        tiny_tokenizer_json("tiny_gemma4_tokenizer.json")
    );
    let names: Vec<&str> = assets.sidecar_names().collect();
    assert_eq!(
        names,
        vec![
            "chat_template.jinja",
            "processor_config.json",
            "tokenizer_config.json"
        ]
    );
    assert_eq!(
        assets.sidecar_str("chat_template.jinja").expect("jinja"),
        "{{ messages }}"
    );
}

#[test]
fn packed_assets_accept_any_integer_dtype_not_just_u8() {
    // Upstream `_tensor_to_bytes` casts non-uint8 integer tensors with `astype(np.uint8)`; a Comfy
    // pack has been seen storing `tokenizer_json` as int8. Assuming U8 would either reject the file
    // or read the padding bytes as tokenizer content.
    let tokenizer = tiny_tokenizer_json("tiny_gemma4_tokenizer.json");
    for (dtype, width) in [("I8", 1usize), ("I16", 2), ("U32", 4), ("I64", 8)] {
        let dir = temp_dir();
        let path = dir.path().join("te.safetensors");
        let (metadata, mut tensors) = valid_pack("tiny_gemma4_tokenizer.json");
        tensors.insert(
            TOKENIZER_JSON_TENSOR_KEY.to_string(),
            PackedTensor::widened(dtype, width, &tokenizer),
        );
        tensors.insert(
            format!("{HF_ASSET_TENSOR_PREFIX}processor_config.json"),
            PackedTensor::widened(dtype, width, br#"{"processor_class":"Gemma4Processor"}"#),
        );
        write_packed(&path, &metadata, &tensors);

        let assets = GemmaAssets::from_single_file(&path)
            .unwrap_or_else(|e| panic!("{dtype} packed assets should load: {e}"));
        assert_eq!(
            assets.tokenizer_json(),
            tokenizer,
            "{dtype} tokenizer bytes"
        );
        assert_eq!(
            assets
                .sidecar_str("processor_config.json")
                .expect("sidecar"),
            r#"{"processor_class":"Gemma4Processor"}"#,
            "{dtype} sidecar bytes"
        );
        LtxGemmaTokenizer::from_assets(&assets)
            .unwrap_or_else(|e| panic!("{dtype} tokenizer should build: {e}"));
    }
}

#[test]
fn float_dtype_asset_is_refused_rather_than_truncated() {
    let dir = temp_dir();
    let path = dir.path().join("te.safetensors");
    let (metadata, mut tensors) = valid_pack("tiny_gemma4_tokenizer.json");
    tensors.insert(
        TOKENIZER_JSON_TENSOR_KEY.to_string(),
        PackedTensor::new("F32", vec![4], vec![0u8; 16]),
    );
    write_packed(&path, &metadata, &tensors);

    let error = GemmaAssets::from_single_file(&path).expect_err("float asset must be refused");
    let text = error.to_string();
    assert!(text.contains("tokenizer_json"), "{text}");
    assert!(text.contains("integer dtype"), "{text}");
}

#[test]
fn missing_gemma_config_metadata_fails_loudly() {
    let dir = temp_dir();
    let path = dir.path().join("te.safetensors");
    let (_, tensors) = valid_pack("tiny_gemma4_tokenizer.json");
    write_packed(&path, &BTreeMap::new(), &tensors);

    let text = GemmaAssets::from_single_file(&path)
        .expect_err("missing gemma_config must fail")
        .to_string();
    assert!(text.contains(GEMMA_CONFIG_METADATA_KEY), "{text}");
}

#[test]
fn missing_tokenizer_tensor_fails_loudly() {
    let dir = temp_dir();
    let path = dir.path().join("te.safetensors");
    let (metadata, mut tensors) = valid_pack("tiny_gemma4_tokenizer.json");
    tensors.remove(TOKENIZER_JSON_TENSOR_KEY);
    write_packed(&path, &metadata, &tensors);

    let text = GemmaAssets::from_single_file(&path)
        .expect_err("missing tokenizer_json must fail")
        .to_string();
    assert!(text.contains(TOKENIZER_JSON_TENSOR_KEY), "{text}");
    assert!(text.contains("missing tensor"), "{text}");
}

#[test]
fn missing_required_sidecar_fails_loudly_and_names_every_gap() {
    let dir = temp_dir();
    let path = dir.path().join("te.safetensors");
    let (metadata, mut tensors) = valid_pack("tiny_gemma4_tokenizer.json");
    tensors.remove(&format!("{HF_ASSET_TENSOR_PREFIX}tokenizer_config.json"));
    tensors.remove(&format!("{HF_ASSET_TENSOR_PREFIX}processor_config.json"));
    write_packed(&path, &metadata, &tensors);

    let text = GemmaAssets::from_single_file(&path)
        .expect_err("missing sidecars must fail")
        .to_string();
    assert!(text.contains("tokenizer_config.json"), "{text}");
    assert!(text.contains("processor_config.json"), "{text}");
}

#[test]
fn asset_offsets_past_end_of_file_fail_loudly() {
    let dir = temp_dir();
    let path = dir.path().join("te.safetensors");
    let (metadata, mut tensors) = valid_pack("tiny_gemma4_tokenizer.json");
    // Declare 4 KiB of tokenizer but write nothing for it: the offsets run past EOF. This is the
    // last tensor in offset order, so only its own range is forged.
    tensors.insert(
        TOKENIZER_JSON_TENSOR_KEY.to_string(),
        PackedTensor {
            declared_bytes: Some(4096),
            ..PackedTensor::new("U8", vec![4096], Vec::new())
        },
    );
    write_packed(&path, &metadata, &tensors);

    let text = GemmaAssets::from_single_file(&path)
        .expect_err("out-of-range offsets must fail")
        .to_string();
    assert!(text.contains(TOKENIZER_JSON_TENSOR_KEY), "{text}");
    assert!(text.contains("truncated"), "{text}");
}

#[test]
fn declared_shape_disagreeing_with_payload_length_fails_loudly() {
    let dir = temp_dir();
    let path = dir.path().join("te.safetensors");
    let (metadata, mut tensors) = valid_pack("tiny_gemma4_tokenizer.json");
    tensors.insert(
        format!("{HF_ASSET_TENSOR_PREFIX}processor_config.json"),
        PackedTensor::new("U8", vec![9999], b"{}".to_vec()),
    );
    write_packed(&path, &metadata, &tensors);

    let text = GemmaAssets::from_single_file(&path)
        .expect_err("shape/byte-length disagreement must fail")
        .to_string();
    assert!(text.contains("processor_config.json"), "{text}");
    assert!(text.contains("requires"), "{text}");
}

#[test]
fn truncated_file_short_read_fails_loudly() {
    let dir = temp_dir();
    let path = dir.path().join("te.safetensors");
    let (metadata, tensors) = valid_pack("tiny_gemma4_tokenizer.json");
    write_packed(&path, &metadata, &tensors);

    // Chop the file in half — the header still promises the full tokenizer payload.
    let full = std::fs::metadata(&path).expect("stat").len();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("reopen");
    file.set_len(full / 2).expect("truncate");
    drop(file);

    let text = GemmaAssets::from_single_file(&path)
        .expect_err("a truncated pack must fail")
        .to_string();
    assert!(
        text.contains("truncated") || text.contains("short read"),
        "{text}"
    );
}

#[test]
fn metadata_string_sidecars_fill_gaps_but_never_shadow_tensors() {
    let dir = temp_dir();
    let path = dir.path().join("te.safetensors");
    let (mut metadata, mut tensors) = valid_pack("tiny_gemma4_tokenizer.json");
    // processor_config.json only as a metadata string (the older/Comfy pack shape) …
    tensors.remove(&format!("{HF_ASSET_TENSOR_PREFIX}processor_config.json"));
    metadata.insert(
        "processor_config.json".to_string(),
        r#"{"processor_class":"FromMetadata"}"#.to_string(),
    );
    // … while chat_template.jinja exists BOTH ways; the tensor must win.
    metadata.insert(
        "chat_template.jinja".to_string(),
        "FROM METADATA".to_string(),
    );
    write_packed(&path, &metadata, &tensors);

    let assets = GemmaAssets::from_single_file(&path).expect("load");
    assert_eq!(
        assets
            .sidecar_str("processor_config.json")
            .expect("fallback"),
        r#"{"processor_class":"FromMetadata"}"#
    );
    assert_eq!(
        assets
            .sidecar_str("chat_template.jinja")
            .expect("tensor wins"),
        "{{ messages }}"
    );
}

#[test]
fn directory_root_layout_still_loads() {
    // LTX-2.3 shipped Gemma as a directory; the same asset type must serve both sources so the
    // tokenizer/BOS policy is shared rather than forked per generation.
    let dir = temp_dir();
    let root = dir.path().join("gemma");
    let nested = root.join("_readout_proj");
    std::fs::create_dir_all(&nested).expect("mkdir");
    std::fs::write(root.join("config.json"), MINIMAL_CONFIG).expect("config");
    std::fs::write(
        root.join("tokenizer.json"),
        tiny_tokenizer_json("tiny_gemma3_tokenizer.json"),
    )
    .expect("tokenizer");
    std::fs::write(
        root.join("tokenizer_config.json"),
        tokenizer_config("\"<pad>\""),
    )
    .expect("tokenizer_config");
    std::fs::write(
        nested.join("processor_config.json"),
        r#"{"processor_class":"Gemma3Processor"}"#,
    )
    .expect("processor_config");
    std::fs::write(root.join("model-00001-of-00002.safetensors"), b"not read")
        .expect("weight shard");

    let assets = GemmaAssets::from_root(&root).expect("load root");
    assert_eq!(assets.config_json(), MINIMAL_CONFIG);
    assert!(assets.sidecar("processor_config.json").is_ok());
    // config.json / tokenizer.json are the canonical files, never sidecars.
    let names: Vec<&str> = assets.sidecar_names().collect();
    assert!(!names.contains(&"config.json"), "{names:?}");
    assert!(!names.contains(&"tokenizer.json"), "{names:?}");

    // `load` dispatches on the path shape.
    assert!(GemmaAssets::load(&root).is_ok());
    let text = GemmaAssets::load(root.join("config.json"))
        .expect_err("a non-safetensors file is not a Gemma source")
        .to_string();
    assert!(
        text.contains("neither a directory nor a .safetensors file"),
        "{text}"
    );
}

// =================================================================================================
// Key layout — ComfyUI-flattened ⇄ legacy HF towers
// =================================================================================================

#[test]
fn flatten_maps_every_hf_tower_and_is_idempotent() {
    let cases = [
        (
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ),
        (
            "model.vision_embedder.patch_dense.weight",
            "vision_model.patch_dense.weight",
        ),
        (
            "model.embed_vision.embedding_projection.weight",
            "multi_modal_projector.embedding_projection.weight",
        ),
        (
            "model.embed_audio.embedding_projection.weight",
            "audio_projector.embedding_projection.weight",
        ),
        // Already flattened (the shipped 2.5 layout) — untouched.
        (
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ),
        (
            "vision_model.patch_dense.weight",
            "vision_model.patch_dense.weight",
        ),
        (
            "text_embedding_projection.video_aggregate_embed.weight",
            "text_embedding_projection.video_aggregate_embed.weight",
        ),
    ];
    for (input, expected) in cases {
        let flat = flatten_gemma4_unified_key(input);
        assert_eq!(flat.as_ref(), expected, "{input}");
        assert_eq!(
            flatten_gemma4_unified_key(&flat).as_ref(),
            expected,
            "{input} is not idempotent"
        );
    }
}

#[test]
fn key_map_accepts_both_layouts_and_skips_packed_assets() {
    let hf = [
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.vision_embedder.patch_dense.weight",
        "model.embed_vision.embedding_projection.weight",
        "model.embed_audio.embedding_projection.weight",
        TOKENIZER_JSON_TENSOR_KEY,
        "hf_asset__tokenizer_config.json",
    ];
    let comfy = [
        "model.layers.0.mlp.up_proj.weight",
        "vision_model.patch_dense.weight",
        "multi_modal_projector.embedding_projection.weight",
        "audio_projector.embedding_projection.weight",
        TOKENIZER_JSON_TENSOR_KEY,
        "hf_asset__tokenizer_config.json",
    ];
    let from_hf = GemmaTeKeyMap::from_keys(hf).expect("hf layout");
    let from_comfy = GemmaTeKeyMap::from_keys(comfy).expect("comfy layout");

    let canonical: Vec<&str> = from_hf.canonical_keys().collect();
    assert_eq!(canonical, from_comfy.canonical_keys().collect::<Vec<_>>());
    assert_eq!(
        from_hf.len(),
        4,
        "packed assets must not be treated as weights"
    );
    assert!(!from_hf.contains(TOKENIZER_JSON_TENSOR_KEY));
    assert!(is_gemma_asset_key(TOKENIZER_JSON_TENSOR_KEY));
    assert!(is_gemma_asset_key("hf_asset__anything.json"));
    assert!(!is_gemma_asset_key("model.layers.0.mlp.up_proj.weight"));

    // The source key round-trips, so a loader can bind either spelling through one canonical name.
    assert_eq!(
        from_hf
            .source_for("vision_model.patch_dense.weight")
            .expect("source"),
        "model.vision_embedder.patch_dense.weight"
    );
    assert_eq!(
        from_comfy
            .source_for("vision_model.patch_dense.weight")
            .expect("source"),
        "vision_model.patch_dense.weight"
    );
    assert_eq!(from_hf.count_with_prefix("vision_model."), 1);
    assert_eq!(from_hf.count_with_prefix("audio_projector."), 1);
}

#[test]
fn key_map_rejects_a_layout_collision() {
    let mixed = [
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    ];
    let text = GemmaTeKeyMap::from_keys(mixed)
        .expect_err("a mixed layout must not silently drop one spelling")
        .to_string();
    assert!(text.contains("both map to"), "{text}");
}

#[test]
fn a_missing_tower_is_an_error_not_a_random_init() {
    // The upstream scar: `load_sd(..., strict=False)` left 11 tower tensors at random init when the
    // remap was wrong. Every lookup here is fallible, and `require_all` names the whole gap at once.
    let partial = GemmaTeKeyMap::from_keys(["model.layers.0.mlp.up_proj.weight"]).expect("build");
    let text = partial
        .source_for("vision_model.patch_dense.weight")
        .expect_err("absent key must error")
        .to_string();
    assert!(text.contains("missing tensor"), "{text}");

    let text = partial
        .require_all([
            "model.layers.0.mlp.up_proj.weight",
            "vision_model.patch_dense.weight",
            "multi_modal_projector.embedding_projection.weight",
            "audio_projector.embedding_projection.weight",
        ])
        .expect_err("require_all must fail")
        .to_string();
    assert!(text.contains("missing 3 required tensor(s)"), "{text}");
    assert!(text.contains("vision_model.patch_dense.weight"), "{text}");
    assert!(
        text.contains("audio_projector.embedding_projection.weight"),
        "{text}"
    );

    assert!(GemmaTeKeyMap::default().is_empty());
}

// =================================================================================================
// The exactly-one-BOS policy
// =================================================================================================

#[test]
fn ensure_single_leading_bos_covers_both_upstream_bugs() {
    // Gemma 4: no BOS from the post-processor → prepend one.
    let mut ids = vec![10, 11, 12];
    ensure_single_leading_bos(&mut ids, 2, 8);
    assert_eq!(ids, vec![2, 10, 11, 12]);

    // Gemma 3: BOS already present → do NOT add a second (the duplicate-BOS bug).
    let mut ids = vec![2, 10, 11];
    ensure_single_leading_bos(&mut ids, 2, 8);
    assert_eq!(ids, vec![2, 10, 11]);

    // Empty encode (empty prompt) → BOS only.
    let mut ids: Vec<i32> = Vec::new();
    ensure_single_leading_bos(&mut ids, 2, 8);
    assert_eq!(ids, vec![2]);

    // A BOS that is not leading is content, not the special prefix.
    let mut ids = vec![10, 2, 11];
    ensure_single_leading_bos(&mut ids, 2, 8);
    assert_eq!(ids, vec![2, 10, 2, 11]);

    // Prepending onto a full sequence re-truncates to max_length.
    let mut ids = vec![10, 11, 12, 13];
    ensure_single_leading_bos(&mut ids, 2, 4);
    assert_eq!(ids, vec![2, 10, 11, 12]);

    // Already-full and already-BOS: untouched.
    let mut ids = vec![2, 11, 12, 13];
    ensure_single_leading_bos(&mut ids, 2, 4);
    assert_eq!(ids, vec![2, 11, 12, 13]);
}

const BOS: i32 = 2;
const PAD: i32 = 0;

fn tiny_tokenizer(fixture: &str) -> LtxGemmaTokenizer {
    LtxGemmaTokenizer::from_parts(
        &tiny_tokenizer_json(fixture),
        &tokenizer_config("\"<pad>\""),
        fixture,
    )
    .expect("build tiny tokenizer")
}

#[test]
fn gemma4_shape_gets_exactly_one_bos_and_left_padding() {
    let tokenizer = tiny_tokenizer("tiny_gemma4_tokenizer.json");
    assert_eq!(tokenizer.bos_id(), BOS);
    assert_eq!(tokenizer.pad_id(), PAD);

    let out = tokenizer
        .encode("a red fox in the snow", 8)
        .expect("encode");
    assert_eq!(out.ids, vec![PAD, BOS, 4, 5, 6, 7, 8, 9]);
    assert_eq!(out.mask, vec![0, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(out.ids.iter().filter(|id| **id == BOS).count(), 1);

    // The upstream `text.strip()` — surrounding whitespace must not change the ids.
    let stripped = tokenizer
        .encode("  \n a red fox in the snow \t ", 8)
        .expect("encode");
    assert_eq!(stripped.ids, out.ids);

    // Empty / whitespace-only prompts encode to BOS alone, left-padded (never a zero-length row).
    for prompt in ["", "   \n\t "] {
        let out = tokenizer.encode(prompt, 4).expect("encode");
        assert_eq!(out.ids, vec![PAD, PAD, PAD, BOS], "{prompt:?}");
        assert_eq!(out.mask, vec![0, 0, 0, 1], "{prompt:?}");
    }

    // Truncation keeps the BOS and drops from the tail.
    let out = tokenizer
        .encode("a red fox in the snow", 4)
        .expect("encode");
    assert_eq!(out.ids, vec![BOS, 4, 5, 6]);
    assert_eq!(out.mask, vec![1, 1, 1, 1]);

    // Non-ASCII survives the byte round trip.
    let out = tokenizer.encode("café 日本語", 4).expect("encode");
    assert_eq!(out.ids, vec![PAD, BOS, 10, 11]);

    assert!(
        tokenizer.encode("a", 0).is_err(),
        "max_length 0 must be rejected"
    );
}

#[test]
fn gemma3_shape_is_not_double_bosed() {
    let tokenizer = tiny_tokenizer("tiny_gemma3_tokenizer.json");
    // This fixture's post_processor already prepends <bos>; the policy must be a no-op.
    let out = tokenizer
        .encode("a red fox in the snow", 8)
        .expect("encode");
    assert_eq!(out.ids, vec![PAD, BOS, 4, 5, 6, 7, 8, 9]);
    assert_eq!(out.ids.iter().filter(|id| **id == BOS).count(), 1);

    let out = tokenizer.encode("", 4).expect("encode");
    assert_eq!(out.ids, vec![PAD, PAD, PAD, BOS]);
    assert_eq!(out.ids.iter().filter(|id| **id == BOS).count(), 1);

    // Both generations agree token-for-token once the policy is applied — that is the point of
    // having one policy rather than a per-generation branch.
    let gemma4 = tiny_tokenizer("tiny_gemma4_tokenizer.json");
    for prompt in ["", "a red fox", "café 日本語", "  a red fox in the snow  "] {
        assert_eq!(
            tokenizer.encode(prompt, 8).expect("g3").ids,
            gemma4.encode(prompt, 8).expect("g4").ids,
            "{prompt:?}"
        );
    }
}

#[test]
fn tokenizer_config_without_a_bos_token_is_refused() {
    let text = LtxGemmaTokenizer::from_parts(
        &tiny_tokenizer_json("tiny_gemma4_tokenizer.json"),
        r#"{"eos_token":"<eos>","pad_token":"<pad>"}"#,
        "no-bos",
    )
    .expect_err("a tokenizer with no declared BOS cannot satisfy the encode contract")
    .to_string();
    assert!(text.contains("bos_token"), "{text}");

    // An AddedToken object is the other HF spelling and must resolve identically.
    let object_form = LtxGemmaTokenizer::from_parts(
        &tiny_tokenizer_json("tiny_gemma4_tokenizer.json"),
        r#"{"bos_token":{"content":"<bos>","special":true},"eos_token":"<eos>"}"#,
        "object-form",
    )
    .expect("AddedToken object form");
    assert_eq!(object_form.bos_id(), BOS);
    // No pad_token declared → upstream falls back to EOS.
    assert_eq!(object_form.pad_id(), 1);

    let text = LtxGemmaTokenizer::from_parts(
        &tiny_tokenizer_json("tiny_gemma4_tokenizer.json"),
        r#"{"bos_token":"<not-in-vocab>","eos_token":"<eos>"}"#,
        "bad-bos",
    )
    .expect_err("an unknown BOS string must not silently become id 0")
    .to_string();
    assert!(text.contains("not in the tokenizer vocabulary"), "{text}");
}

// =================================================================================================
// Offline: the packed text encoder needs no other file, and no model cache
// =================================================================================================

/// Serializes the one test that mutates process-global env so a parallel test never observes it.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct EnvGuard(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl EnvGuard {
    fn set(vars: &[(&'static str, &Path)]) -> Self {
        let saved = vars
            .iter()
            .map(|(key, value)| {
                let previous = std::env::var_os(key);
                std::env::set_var(key, value);
                (*key, previous)
            })
            .collect();
        Self(saved)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, previous) in self.0.drain(..) {
            match previous {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

#[test]
fn packed_text_encoder_loads_offline_against_an_empty_cache_and_reads_no_other_file() {
    // The acceptance criterion for sc-18762: loading the 2.5 text encoder requires ZERO files
    // besides the one safetensors — no `google/gemma-3-12b-it` co-requisite snapshot, no cache
    // lookup, no network.
    //
    // The hub-cache env vars themselves cannot be named in workspace Rust: `check-workspace.py`'s
    // `RUST_BANNED_SUBSTRINGS` gate bans them tree-wide (tests included), which already proves no
    // inference source consults a hub cache. What this test adds is that the *default* cache
    // location is empty and stays empty: the hub cache derives from `XDG_CACHE_HOME`/`HOME`, both
    // redirected here at a temp root that is pre-seeded with an empty cache directory.
    let _lock = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let model_dir = temp_dir();
    let cache_dir = temp_dir();
    let empty_cache = cache_dir.path().join("huggingface");
    std::fs::create_dir_all(&empty_cache).expect("seed an empty cache root");
    let path = model_dir
        .path()
        .join("gemma4-12b-with-proj-ltx-2.5-bf16.safetensors");
    let (metadata, tensors) = valid_pack("tiny_gemma4_tokenizer.json");
    write_packed(&path, &metadata, &tensors);

    let _env = EnvGuard::set(&[
        ("XDG_CACHE_HOME", cache_dir.path()),
        ("HOME", cache_dir.path()),
    ]);

    let assets = GemmaAssets::from_single_file(&path).expect("offline load");
    let tokenizer = LtxGemmaTokenizer::from_assets(&assets).expect("offline tokenizer");
    let out = tokenizer
        .encode("a red fox in the snow", 8)
        .expect("offline encode");
    assert_eq!(out.ids, vec![PAD, BOS, 4, 5, 6, 7, 8, 9]);

    // Nothing appeared beside the model file, and the empty cache stayed empty.
    let model_entries: Vec<_> = std::fs::read_dir(model_dir.path())
        .expect("read model dir")
        .map(|e| e.expect("entry").file_name())
        .collect();
    assert_eq!(model_entries.len(), 1, "{model_entries:?}");
    assert_eq!(
        std::fs::read_dir(&empty_cache).expect("read cache").count(),
        0,
        "the load must not populate a model cache"
    );
    assert_eq!(
        std::fs::read_dir(cache_dir.path())
            .expect("read cache root")
            .count(),
        1,
        "the load must not create anything under the cache root"
    );
}

// =================================================================================================
// Real-asset token-id parity (opt-in)
// =================================================================================================

struct ParityGolden {
    max_length: usize,
    bos_token_id: i32,
    pad_token_id: i32,
    cases: Vec<ParityCase>,
}

struct ParityCase {
    name: String,
    prompt: String,
    ids: Vec<i32>,
    mask: Vec<i32>,
    raw_first_id: Option<i32>,
}

fn parity_golden(name: &str) -> ParityGolden {
    let text = std::fs::read_to_string(PathBuf::from(FIXTURES).join(name)).expect("golden fixture");
    let json: serde_json::Value = serde_json::from_str(&text).expect("parse golden");
    let int = |value: &serde_json::Value, key: &str| -> i64 {
        value
            .get(key)
            .and_then(serde_json::Value::as_i64)
            .unwrap_or_else(|| panic!("{name}: missing integer {key}"))
    };
    let ints = |value: &serde_json::Value, key: &str| -> Vec<i32> {
        value
            .get(key)
            .and_then(serde_json::Value::as_array)
            .unwrap_or_else(|| panic!("{name}: missing array {key}"))
            .iter()
            .map(|v| v.as_i64().expect("integer element") as i32)
            .collect()
    };
    let cases = json
        .get("cases")
        .and_then(serde_json::Value::as_object)
        .unwrap_or_else(|| panic!("{name}: missing cases"))
        .iter()
        .map(|(case, value)| ParityCase {
            name: case.clone(),
            prompt: value
                .get("prompt")
                .and_then(serde_json::Value::as_str)
                .expect("prompt")
                .to_string(),
            ids: ints(value, "ids"),
            mask: ints(value, "mask"),
            raw_first_id: value
                .get("raw_first_id")
                .and_then(serde_json::Value::as_i64)
                .map(|v| v as i32),
        })
        .collect();
    ParityGolden {
        max_length: int(&json, "max_length") as usize,
        bos_token_id: int(&json, "bos_token_id") as i32,
        pad_token_id: int(&json, "pad_token_id") as i32,
        cases,
    }
}

/// Assert the Rust policy reproduces the upstream Python ids for every case, and that the
/// exactly-one-BOS invariant holds. `expect_raw_bos` says whether this tokenizer's `post_processor`
/// is supposed to emit the BOS itself — the difference between the two Gemma generations.
fn assert_parity(tokenizer: &LtxGemmaTokenizer, golden: &ParityGolden, expect_raw_bos: bool) {
    assert_eq!(tokenizer.bos_id(), golden.bos_token_id);
    assert_eq!(tokenizer.pad_id(), golden.pad_token_id);
    assert!(!golden.cases.is_empty());
    for case in &golden.cases {
        let name = &case.name;
        let out = tokenizer
            .encode(&case.prompt, golden.max_length)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(out.ids, case.ids, "{name}: ids");
        assert_eq!(out.mask, case.mask, "{name}: mask");

        // The un-policied encode is what the reference proves is broken/whole per generation:
        // Gemma 4's post_processor emits no BOS at all, Gemma 3's always emits exactly one.
        // `literal_bos_prefix` is excluded from the negative direction: its prompt *text* opens
        // with the `<bos>` added-token string, so even Gemma 4's raw encode starts with id 2.
        if expect_raw_bos {
            assert_eq!(
                case.raw_first_id,
                Some(golden.bos_token_id),
                "{name}: raw BOS"
            );
        } else if name != "literal_bos_prefix" {
            assert_ne!(
                case.raw_first_id,
                Some(golden.bos_token_id),
                "{name}: raw BOS"
            );
        }

        let count = out
            .ids
            .iter()
            .filter(|id| **id == golden.bos_token_id)
            .count();
        assert!(count >= 1, "{name}: no BOS at all");
        assert_eq!(
            out.ids[golden.max_length - out.mask.iter().filter(|m| **m == 1).count()],
            golden.bos_token_id,
            "{name}: the first unmasked token must be the BOS"
        );
        // Exactly one BOS — except for the deliberate `literal_bos_prefix` case, where the prompt
        // *text* itself contains the `<bos>` added-token string. There the policy correctly adds
        // none (the reference behaves identically), and the id equality above pins it.
        if name != "literal_bos_prefix" {
            assert_eq!(count, 1, "{name}: expected exactly one BOS, got {count}");
        }
    }
}

/// Opt-in: `LTX25_TE_SAFETENSORS=<…>/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
/// cargo test -p sceneworks-gen-core --test gemma_assets -- --ignored`.
/// Reads only the header plus ~32 MB of asset tensors out of the 26.3 GB file.
#[test]
#[ignore = "needs the real LTX-2.5 text encoder; set LTX25_TE_SAFETENSORS"]
fn real_packed_ltx25_text_encoder_matches_reference_token_ids() {
    let path = std::env::var(REAL_TE_ENV)
        .unwrap_or_else(|_| panic!("set {REAL_TE_ENV} to the packed LTX-2.5 text encoder"));
    let assets = GemmaAssets::from_single_file(&path).expect("unpack real text encoder");

    assert_eq!(
        assets.config_model_type().expect("model_type"),
        "gemma4_unified"
    );
    let names: Vec<&str> = assets.sidecar_names().collect();
    assert_eq!(
        names,
        vec![
            "chat_template.jinja",
            "generation_config.json",
            "processor_config.json",
            "tokenizer_config.json"
        ]
    );
    assert_eq!(
        assets.tokenizer_json().len(),
        32_169_626,
        "packed tokenizer.json bytes"
    );
    assert_eq!(
        assets.sidecar("chat_template.jinja").expect("jinja").len(),
        17_466
    );
    assert_eq!(
        assets.sidecar("tokenizer_config.json").expect("cfg").len(),
        2_749
    );
    assert_eq!(
        assets.sidecar("processor_config.json").expect("cfg").len(),
        1_382
    );
    assert_eq!(
        assets.sidecar("generation_config.json").expect("cfg").len(),
        255
    );

    let tokenizer = LtxGemmaTokenizer::from_assets(&assets).expect("build real tokenizer");
    assert_parity(
        &tokenizer,
        &parity_golden("ltx25_gemma4_token_parity.json"),
        false,
    );
}

/// Opt-in: `LTX23_GEMMA_DIR=<…>/ltx-2.3/gemma cargo test -p sceneworks-gen-core \
/// --test gemma_assets -- --ignored`. Proves the shared policy does not regress Gemma 3.
#[test]
#[ignore = "needs a real gemma-3-12b-it snapshot dir; set LTX23_GEMMA_DIR"]
fn real_gemma3_directory_root_matches_reference_token_ids() {
    let dir = std::env::var(REAL_GEMMA3_ENV)
        .unwrap_or_else(|_| panic!("set {REAL_GEMMA3_ENV} to a gemma-3-12b-it snapshot dir"));
    let assets = GemmaAssets::from_root(&dir).expect("load gemma-3 root");
    let tokenizer = LtxGemmaTokenizer::from_assets(&assets).expect("build gemma-3 tokenizer");
    assert_parity(
        &tokenizer,
        &parity_golden("ltx23_gemma3_token_parity.json"),
        true,
    );
}
