//! sc-18762 — the LTX prompt tokenizers on the MLX side.
//!
//! * [`Ltx25Tokenizer`] builds entirely from the **packed** single-file 2.5 text encoder (no
//!   separate Gemma snapshot, no HF cache) and lifts to `(1, max_length)` MLX arrays.
//! * [`LtxTokenizer`] (2.3, Gemma-3) keeps its ids unchanged while now running the same
//!   exactly-one-BOS guard, so the duplicate-BOS regression cannot reappear.
//!
//! The tiny tokenizers below are byte-identical to
//! `gen-core/tests/fixtures/tiny_gemma{3,4}_tokenizer.json` (regenerate with that directory's
//! `gen_tiny_gemma_tokenizers.py`); they are inlined so this crate does not reach across a crate
//! boundary for a fixture. They differ only in whether the `post_processor` emits `<bos>` — the
//! measured difference between the Gemma 4 and Gemma 3 tokenizers.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use mlx_gen_ltx::{Ltx25Tokenizer, LtxTokenizer};

const TINY_GEMMA4: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<pad>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":1,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"<bos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":3,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,"a":4,"red":5,"fox":6,"in":7,"the":8,"snow":9,"café":10,"日本語":11},"unk_token":"<unk>"}}"#;

const TINY_GEMMA3: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<pad>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":1,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":2,"content":"<bos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":3,"content":"<unk>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"<bos>":{"id":"<bos>","ids":[2],"tokens":["<bos>"]}}},"decoder":null,"model":{"type":"WordLevel","vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,"a":4,"red":5,"fox":6,"in":7,"the":8,"snow":9,"café":10,"日本語":11},"unk_token":"<unk>"}}"#;

const TOKENIZER_CONFIG: &str = r#"{"bos_token":"<bos>","eos_token":"<eos>","pad_token":"<pad>","tokenizer_class":"GemmaTokenizer"}"#;
const PROCESSOR_CONFIG: &str = r#"{"processor_class":"Gemma4Processor"}"#;
const GEMMA_CONFIG: &str = r#"{"model_type":"gemma4_unified","gemma_version":"gemma4-12b-ltx-v1"}"#;

/// Write a packed single-file text encoder in the shipped 2.5 shape: `__metadata__.gemma_config`
/// plus `tokenizer_json` / `hf_asset__*` `U8` tensors, alongside a weight tensor.
fn write_packed_te(path: &Path) {
    let assets: BTreeMap<&str, &[u8]> = BTreeMap::from([
        ("tokenizer_json", TINY_GEMMA4.as_bytes()),
        (
            "hf_asset__tokenizer_config.json",
            TOKENIZER_CONFIG.as_bytes(),
        ),
        (
            "hf_asset__processor_config.json",
            PROCESSOR_CONFIG.as_bytes(),
        ),
        ("model.embed_tokens.weight", &[7u8, 7, 7, 7]),
    ]);
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".into(),
        serde_json::json!({ "gemma_config": GEMMA_CONFIG, "format": "pt" }),
    );
    let mut offset = 0usize;
    let mut data = Vec::new();
    for (name, bytes) in &assets {
        let end = offset + bytes.len();
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": "U8",
                "shape": [bytes.len()],
                "data_offsets": [offset, end],
            }),
        );
        data.extend_from_slice(bytes);
        offset = end;
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header json");
    let mut file = std::fs::File::create(path).expect("create packed te");
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("header len");
    file.write_all(&header).expect("header");
    file.write_all(&data).expect("data");
}

fn ids_of(array: &mlx_rs::Array) -> Vec<i32> {
    array.as_slice::<i32>().to_vec()
}

#[test]
fn packed_te_tokenizer_needs_no_other_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir
        .path()
        .join("gemma4-12b-with-proj-ltx-2.5-bf16.safetensors");
    write_packed_te(&path);

    let tokenizer = Ltx25Tokenizer::from_packed_te_file(&path).expect("load packed tokenizer");
    assert_eq!(tokenizer.bos_id(), 2);
    assert_eq!(tokenizer.pad_id(), 0);

    let (ids, mask) = tokenizer
        .encode("a red fox in the snow", 8)
        .expect("encode");
    assert_eq!(ids.shape(), &[1, 8]);
    assert_eq!(mask.shape(), &[1, 8]);
    // Exactly one leading <bos> (the Gemma 4 post-processor supplies none), left-padded.
    assert_eq!(ids_of(&ids), vec![0, 2, 4, 5, 6, 7, 8, 9]);
    assert_eq!(ids_of(&mask), vec![0, 1, 1, 1, 1, 1, 1, 1]);

    // Upstream strips the prompt before encoding.
    let (stripped, _) = tokenizer
        .encode("  a red fox in the snow \n", 8)
        .expect("encode");
    assert_eq!(ids_of(&stripped), ids_of(&ids));

    // An empty prompt is legal on the 2.5 path: a lone <bos>.
    let (ids, mask) = tokenizer.encode("", 4).expect("encode empty");
    assert_eq!(ids_of(&ids), vec![0, 0, 0, 2]);
    assert_eq!(ids_of(&mask), vec![0, 0, 0, 1]);

    // Truncation keeps the BOS and drops from the tail.
    let (ids, _) = tokenizer
        .encode("a red fox in the snow", 4)
        .expect("encode");
    assert_eq!(ids_of(&ids), vec![2, 4, 5, 6]);

    assert_eq!(tokenizer.decode(&[4, 5, 6]).expect("decode"), "a red fox");
    // The dir still holds exactly the one file the tokenizer came from.
    assert_eq!(std::fs::read_dir(dir.path()).expect("read dir").count(), 1);
}

#[test]
fn packed_te_load_fails_loudly_when_an_asset_is_absent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("te.safetensors");
    // A weights-only file — no packed assets at all.
    let header = serde_json::json!({
        "model.embed_tokens.weight": { "dtype": "U8", "shape": [4], "data_offsets": [0, 4] },
    });
    let header = serde_json::to_vec(&header).expect("json");
    let mut file = std::fs::File::create(&path).expect("create");
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("len");
    file.write_all(&header).expect("header");
    file.write_all(&[0u8; 4]).expect("data");
    drop(file);

    let text = Ltx25Tokenizer::from_packed_te_file(&path)
        .expect_err("a weights-only file is not a self-contained text encoder")
        .to_string();
    assert!(text.contains("gemma_config"), "{text}");
}

#[test]
fn gemma3_path_keeps_exactly_one_bos() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gemma = dir.path().join("gemma");
    std::fs::create_dir_all(&gemma).expect("mkdir");
    std::fs::write(gemma.join("tokenizer.json"), TINY_GEMMA3).expect("write tokenizer");

    let tokenizer = LtxTokenizer::from_dir(&gemma).expect("load 2.3 tokenizer");
    let (ids, mask) = tokenizer
        .encode("a red fox in the snow", 8)
        .expect("encode");
    // The gemma-3 post-processor already prepends <bos>; the guard must not add a second.
    assert_eq!(ids_of(&ids), vec![0, 2, 4, 5, 6, 7, 8, 9]);
    assert_eq!(ids_of(&mask), vec![0, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(ids_of(&ids).iter().filter(|id| **id == 2).count(), 1);

    // Truncation still keeps the leading BOS.
    let (ids, _) = tokenizer
        .encode("a red fox in the snow", 4)
        .expect("encode");
    assert_eq!(ids_of(&ids), vec![2, 4, 5, 6]);

    // The 2.3 contract still refuses an empty prompt (unchanged).
    assert!(tokenizer.encode("", 8).is_err());
}

#[test]
fn gemma3_path_repairs_a_tokenizer_that_emits_no_bos() {
    // The regression direction the guard exists for: if a Gemma snapshot ever ships without the
    // BOS post-processor, the 2.3 path must still produce exactly one leading BOS rather than
    // silently conditioning on a BOS-less sequence.
    let dir = tempfile::tempdir().expect("tempdir");
    let gemma = dir.path().join("gemma");
    std::fs::create_dir_all(&gemma).expect("mkdir");
    std::fs::write(gemma.join("tokenizer.json"), TINY_GEMMA4).expect("write tokenizer");

    let tokenizer = LtxTokenizer::from_dir(&gemma).expect("load");
    let (ids, _) = tokenizer
        .encode("a red fox in the snow", 8)
        .expect("encode");
    assert_eq!(ids_of(&ids), vec![0, 2, 4, 5, 6, 7, 8, 9]);
}

/// Opt-in end-to-end proof against the shipped 26.3 GB text encoder: only the header and the packed
/// asset tensors are read — no weights, no GPU.
///
/// `LTX25_TE_SAFETENSORS=<…>/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
///  cargo test -p mlx-gen-ltx --test integration -- gemma_tokenizer:: --ignored`
#[test]
#[ignore = "needs the real LTX-2.5 text encoder; set LTX25_TE_SAFETENSORS"]
fn real_packed_te_tokenizes_without_a_gemma_snapshot() {
    let path = std::env::var("LTX25_TE_SAFETENSORS")
        .expect("set LTX25_TE_SAFETENSORS to the packed LTX-2.5 text encoder");
    let tokenizer =
        Ltx25Tokenizer::from_packed_te_file(Path::new(&path)).expect("unpack real tokenizer");
    assert_eq!(tokenizer.bos_id(), 2);
    assert_eq!(tokenizer.pad_id(), 0);

    let (ids, mask) = tokenizer
        .encode(
            "A cinematic shot of a red fox running through snow at dawn.",
            64,
        )
        .expect("encode");
    let ids = ids_of(&ids);
    let mask = ids_of(&mask);
    assert_eq!(ids.len(), 64);
    let valid = mask.iter().filter(|m| **m == 1).count();
    assert_eq!(valid, 14, "13 prompt tokens + one prepended BOS");
    assert_eq!(ids[64 - valid], 2, "the first unmasked token is the BOS");
    assert_eq!(ids.iter().filter(|id| **id == 2).count(), 1);
    assert!(ids[..64 - valid].iter().all(|id| *id == 0), "left pad");
}
