use super::*;
use crate::events::{decode_sequence, encode_decoded_sequence, parse_tokens_txt};

/// Upstream's pinned `tokenizer_fingerprint` (`config.json` of `m-a-p/SheetSage2`) and vocabulary
/// size, recomputed natively. Mutation that must fail: drop one structure label, or reorder the
/// meter pairs.
#[test]
fn vocabulary_matches_the_checkpoint_fingerprint() {
    let tokenizer = Tokenizer::new(300.0, 100, Some("5ba3325af0344c7f")).unwrap();
    assert_eq!(tokenizer.n_tokens(), 31_678);
    assert_eq!(tokenizer.fingerprint(), "5ba3325af0344c7f");
    assert!(Tokenizer::new(299.0, 100, Some("5ba3325af0344c7f")).is_err());
}

/// The tiny reference model's vocabulary (1 s window) matches the fingerprint upstream computed for
/// it, so the fingerprint function is exercised on a second payload.
#[test]
fn tiny_vocabulary_matches_upstream() {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("../../testdata/tiny/reference.json")).unwrap();
    let fp = reference["tokenizer"]["fingerprint"].as_str().unwrap();
    let tokenizer = Tokenizer::new(1.0, 100, Some(fp)).unwrap();
    assert_eq!(
        u64::from(tokenizer.n_tokens()),
        reference["tokenizer"]["n_tokens"].as_u64().unwrap()
    );
}

/// Oracles whose descriptions come from the head code (or are identical under both revisions: the
/// release only differs in printing a flat key's raw sharp name, which `synth_eb_release` shows and
/// which is therefore not listed here).
const ORACLES: [&str; 7] = [
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/synth_full/tokens.txt"),
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/real_full/tokens.txt"),
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/real_melody/tokens.txt"),
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/synth_eb_head/tokens.txt"),
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/synth_merged/tokens.txt"),
    include_str!(
        "../../../../../scripts/reference/sheetsage2/artifacts/long_multiwindow/tokens.txt"
    ),
    include_str!("../../../../../scripts/reference/sheetsage2/artifacts/silence/tokens.txt"),
];

/// Every token of every committed oracle is described exactly as upstream described it, and every
/// window decodes strictly and re-encodes to the same ids.
#[test]
fn committed_oracles_describe_decode_and_reencode_exactly() {
    let tokenizer = Tokenizer::new(300.0, 100, Some("5ba3325af0344c7f")).unwrap();
    let mut described = 0;
    for text in ORACLES {
        for line in text
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let mut parts = line.split('\t');
            let _index = parts.next().unwrap();
            let token: u32 = parts.next().unwrap().parse().unwrap();
            assert_eq!(tokenizer.describe(token).unwrap(), parts.next().unwrap());
            described += 1;
        }
        for record in parse_tokens_txt(text).unwrap() {
            let decoded = decode_sequence(&tokenizer, &record.tokens, true).unwrap();
            assert_eq!(
                encode_decoded_sequence(&tokenizer, &decoded).unwrap(),
                record.tokens
            );
        }
    }
    assert!(described > 5_000, "{described}");
}

#[test]
fn prompts_normalize_like_upstream() {
    let tokenizer = Tokenizer::new(300.0, 100, None).unwrap();
    assert_eq!(
        tokenizer
            .normalize_prompts(&["<|melody_full|>", "timestamp", "timestamp"])
            .unwrap(),
        vec!["timestamp", "melody_full"]
    );
    assert!(tokenizer
        .normalize_prompts(&["chord_full", "chord_majmin"])
        .is_err());
    assert!(tokenizer.normalize_prompts::<&str>(&[]).is_err());
    assert!(tokenizer.normalize_prompts(&["lyrics"]).is_err());
    assert_eq!(
        tokenizer.prompt_prefix(&FULL_TASK_PROMPTS).unwrap(),
        vec![1, 4, 5, 6, 7, 9, 11, 3]
    );
}
