//! Real-`qwen.tiktoken` parity of the text tokenizer, prompt prefixes and plan format (sc-22990).
//!
//! The lib tests run the same cases on a committed synthetic rank table; `qwen.tiktoken` itself is
//! never committed (its redistribution is gated by the crate's licence policy), so these tests
//! read it from the pinned snapshot. `#[ignore]`d in ordinary runs; under `--ignored` a missing
//! variable panics rather than passing silently.
//!
//! * `YUE2_HF_HUB` — a Hugging Face hub directory holding the pinned `m-a-p/YuE2-3B` revision.
//! * `YUE2_REF_PYTHON` — the pinned reference interpreter
//!   (`~/.cache/sceneworks-yue2-ref/venv/bin/python`, see `scripts/reference/yue2/README.md`),
//!   for the plan round trip through upstream's own `SymbolicPlan.load`.
//!
//! ```text
//! YUE2_HF_HUB=/path/to/hub YUE2_REF_PYTHON=~/.cache/sceneworks-yue2-ref/venv/bin/python \
//!   cargo test --release -p candle-audio-yue2 --test protocol_real -- --ignored --nocapture
//! ```
//!
//! CPU only; nothing but the 2.5 MB rank table is read (peak RSS well under 1 GB).

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use candle_audio_yue2::inventory::{self, ComponentId};
use candle_audio_yue2::plan::{PlanStep, SymbolicPlan};
use candle_audio_yue2::protocol::{negative_prefix, token_prefixes, GenerationConfig, SongRequest};
use candle_audio_yue2::{SnapshotDirs, Yue2TextTokenizer};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("real-weight test run without {name} (see module docs)")),
    )
}

fn yue2_3b_snapshot() -> PathBuf {
    let repo = ComponentId::QwenTiktoken.component().repo;
    env_path("YUE2_HF_HUB")
        .join(format!("models--{}", repo.id.replace('/', "--")))
        .join("snapshots")
        .join(repo.revision)
}

fn tokenizer() -> Yue2TextTokenizer {
    let repo = ComponentId::QwenTiktoken.component().repo;
    let dirs = SnapshotDirs::new().with(repo.id, yue2_3b_snapshot());
    let start = Instant::now();
    let tok = Yue2TextTokenizer::load(&dirs).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("verified + loaded qwen.tiktoken in {:.2?}", start.elapsed());
    tok
}

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/protocol/tokenizer_qwen.json");
    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn check<E: std::fmt::Display>(label: &str, expected: &Value, got: Result<Vec<u32>, E>) {
    match (expected.get("ok"), got) {
        (Some(ok), Ok(got)) => assert_eq!(got, ids(ok), "{label}"),
        (None, Err(_)) => {}
        (Some(_), Err(e)) => panic!("{label}: upstream accepts, native refuses: {e}"),
        (None, Ok(_)) => panic!(
            "{label}: upstream refuses ({}), native accepts",
            expected["error"]
        ),
    }
}

/// Encode, decode, positive / planner / negative prefixes for English, Chinese, supplied ABC and
/// all three cot modes on the pinned `qwen.tiktoken` equal upstream's.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB (see the module docs)"]
fn qwen_tiktoken_matches_upstream() {
    let tok = tokenizer();
    let fixture = fixture();
    let start = Instant::now();
    let encode = fixture["encode"].as_array().unwrap();
    for case in encode {
        let text = case["text"].as_str().unwrap();
        assert_eq!(
            tok.encode(text).unwrap(),
            ids(&case["ids"]),
            "encode {}",
            case["name"]
        );
    }
    let decode = fixture["decode"].as_array().unwrap();
    for case in decode {
        assert_eq!(
            tok.decode(&ids(&case["ids"])),
            case["text"].as_str().unwrap(),
            "decode {}",
            case["name"]
        );
    }
    let prefixes = fixture["prefixes"].as_array().unwrap();
    for case in prefixes {
        let name = case["name"].as_str().unwrap();
        let request = SongRequest::from_json(&case["request"]).unwrap();
        assert_eq!(request.text(), case["text"].as_str().unwrap(), "{name}");
        check(
            &format!("{name} planner"),
            &case["planner"],
            token_prefixes(&request, &tok, None),
        );
        check(
            &format!("{name} negative_without_abc"),
            &case["negative_without_abc"],
            negative_prefix(&request, &tok, None),
        );
        if let Some(abc) = case.get("abc_ids") {
            let abc = ids(abc);
            check(
                &format!("{name} positive"),
                &case["positive"],
                token_prefixes(&request, &tok, Some(&abc)),
            );
            check(
                &format!("{name} negative"),
                &case["negative"],
                negative_prefix(&request, &tok, Some(&abc)),
            );
        }
    }
    for case in fixture["abc_id_cases"].as_array().unwrap() {
        let request = SongRequest::from_json(&case["request"]).unwrap();
        let abc = ids(&case["abc_ids"]);
        check(
            "abc_id positive",
            &case["positive"],
            token_prefixes(&request, &tok, Some(&abc)),
        );
        check(
            "abc_id negative",
            &case["negative"],
            negative_prefix(&request, &tok, Some(&abc)),
        );
    }
    eprintln!(
        "{} encode, {} decode, {} prefix cases in {:.2?}",
        encode.len(),
        decode.len(),
        prefixes.len(),
        start.elapsed()
    );
}

/// The loader parses only the bytes that were pinned: a `qwen.tiktoken` replaced between
/// verification and the read is refused, not parsed.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB (see the module docs)"]
fn a_tiktoken_swapped_after_verification_is_refused() {
    let repo = ComponentId::QwenTiktoken.component().repo;
    let copy = tempfile::tempdir().unwrap();
    let path = copy.path().join("qwen.tiktoken");
    std::fs::copy(yue2_3b_snapshot().join("qwen.tiktoken"), &path).unwrap();
    let dirs = SnapshotDirs::new().with(repo.id, copy.path());
    let verified =
        candle_audio_yue2::snapshot::resolve_component(ComponentId::QwenTiktoken, &dirs).unwrap();
    Yue2TextTokenizer::from_verified(&verified).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    let err = Yue2TextTokenizer::from_verified(&verified).unwrap_err();
    assert!(
        matches!(
            err,
            candle_audio_yue2::TokenizerError::ChangedAfterVerification { .. }
        ),
        "{err}"
    );
    let reverified =
        candle_audio_yue2::snapshot::resolve_component(ComponentId::QwenTiktoken, &dirs);
    assert!(
        reverified.is_err(),
        "the swapped file no longer verifies either"
    );
}

/// Reports how long one unbroken 32k-character piece (ASCII letters, then CJK) takes to encode
/// on the real table. Timing is printed, never asserted (machine-dependent); the structural bound
/// on merge work is a lib test (`tokenizer::tests::merge_work_is_linear_in_the_piece`).
#[test]
#[ignore = "real weights: set YUE2_HF_HUB (see the module docs)"]
fn long_unbroken_piece_encode_time() {
    let tok = tokenizer();
    let ascii: String = (0..32_768u32)
        .map(|i| char::from(b'a' + (i.wrapping_mul(2_654_435_761) >> 27) as u8 % 26))
        .collect();
    let cjk: String = (0..32_768u32)
        .map(|i| char::from_u32(0x4E00 + (i.wrapping_mul(2_654_435_761) >> 20) % 0x5000).unwrap())
        .collect();
    for (name, text) in [("ascii", &ascii), ("cjk", &cjk)] {
        let start = Instant::now();
        let ids = tok.encode(text).unwrap();
        eprintln!(
            "{name}: 32768 chars ({} bytes) -> {} ids in {:.2?}",
            text.len(),
            ids.len(),
            start.elapsed()
        );
        assert_eq!(tok.decode(&ids), *text);
    }
}

/// The checkpoint's own `yue2_generation_config.json` (pinned bytes) is exactly the protocol
/// defaults.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB (see the module docs)"]
fn pinned_generation_config_is_the_protocol_default() {
    let name = "yue2_generation_config.json";
    let pinned = ComponentId::Lm.component().file(name).unwrap();
    let bytes = std::fs::read(yue2_3b_snapshot().join(name)).unwrap();
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(digest, pinned.sha256);
    let config = GenerationConfig::from_json(&serde_json::from_slice(&bytes).unwrap()).unwrap();
    assert_eq!(config, GenerationConfig::default());
    assert_eq!(inventory::REPOS.len(), 5);
}

/// Plans saved natively load in upstream's own `SymbolicPlan.load` with the exact IDs.
#[test]
#[ignore = "reference env: set YUE2_HF_HUB and YUE2_REF_PYTHON (see the module docs)"]
fn native_saved_plans_load_in_upstream() {
    let python = env_path("YUE2_REF_PYTHON");
    let tok = tokenizer();
    let fixture = fixture();
    let root = tempfile::tempdir().unwrap();
    let mut expected = Map::new();
    for case in fixture["prefixes"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let request = SongRequest::from_json(&case["request"]).unwrap();
        let plan = match SymbolicPlan::prepare(request, &tok).unwrap() {
            PlanStep::Ready(plan) => plan,
            PlanStep::GenerateAbc(planning) => {
                let abc = ids(&case["abc_ids"]);
                planning.finish(&tok, abc, Map::new(), false).unwrap()
            }
        };
        plan.save(&root.path().join(name)).unwrap();
        expected.insert(
            name.into(),
            serde_json::json!({"abc_ids": plan.abc_ids(), "prefix": plan.prefix(), "abc": plan.abc()}),
        );
    }
    let script = r#"
import json, sys
from pathlib import Path
from yue2.pipeline import SymbolicPlan
out = {}
for d in sorted(Path(sys.argv[1]).iterdir()):
    p = SymbolicPlan.load(d)
    out[d.name] = {"abc_ids": p.abc_ids, "prefix": p.prefix, "abc": p.abc}
print(json.dumps(out))
"#;
    let start = Instant::now();
    let output = Command::new(&python)
        .args(["-c", script])
        .arg(root.path())
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "upstream load failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let loaded: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(loaded, Value::Object(expected));
    eprintln!(
        "{} native plans loaded by upstream in {:.2?}",
        loaded.as_object().unwrap().len(),
        start.elapsed()
    );
}
