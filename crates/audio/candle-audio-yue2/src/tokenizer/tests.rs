//! Tokenizer parity with the pinned upstream `YuE2TextTokenizer` on the committed synthetic rank
//! table (the real `qwen.tiktoken` parity runs in `tests/protocol_real.rs`, env-gated).

use super::*;
use crate::test_fixtures::{ids, json, synthetic};

fn case<'a>(fixture: &'a serde_json::Value, section: &str, name: &str) -> &'a serde_json::Value {
    fixture[section]
        .as_array()
        .expect("section")
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("no {section} case {name}"))
}

/// Every encode case — English, Chinese, ABC, NFD/NFC, whitespace, contractions, specials-as-text,
/// Unicode 16/17 edges — produces upstream's exact IDs.
#[test]
fn encode_matches_upstream_on_every_case() {
    let fixture = json("tokenizer_synthetic.json");
    let tok = synthetic();
    let cases = fixture["encode"].as_array().unwrap();
    assert!(cases.len() >= 25, "fixture lost cases");
    for case in cases {
        let text = case["text"].as_str().unwrap();
        assert_eq!(
            tok.encode(text).unwrap(),
            ids(&case["ids"]),
            "encode case {}",
            case["name"]
        );
    }
}

/// Every decode case — round trips, malformed UTF-8 (U+FFFD), special tokens, and IDs past the
/// text vocabulary that upstream drops — produces upstream's exact text.
#[test]
fn decode_matches_upstream_on_every_case() {
    let fixture = json("tokenizer_synthetic.json");
    let tok = synthetic();
    for case in fixture["decode"].as_array().unwrap() {
        assert_eq!(
            tok.decode(&ids(&case["ids"])),
            case["text"].as_str().unwrap(),
            "decode case {}",
            case["name"]
        );
    }
}

/// NFC is Python 3.12's (Unicode 15.0): a Unicode-16 base + an old combining mark stays
/// uncomposed, and a Unicode-16 precomposed letter stays undecomposed. Plain `nfc()` (Unicode 17
/// data) composes the first — this is the defect the age split prevents.
#[test]
fn nfc_is_pinned_to_the_reference_unicode_version() {
    let tok = synthetic();
    let todhri = "\u{105d2}\u{0307}";
    assert_ne!(
        todhri.nfc().collect::<String>(),
        todhri,
        "the crate's data composes it"
    );
    assert_eq!(tok.normalize(todhri), todhri);
    assert_eq!(tok.normalize("\u{105c9}"), "\u{105c9}");
    assert_eq!(tok.normalize("Cafe\u{301}"), "Caf\u{e9}");
    // A Unicode-15 run on each side of a later code point is still normalized.
    assert_eq!(
        tok.normalize("e\u{301}\u{105d2}\u{0307}e\u{301}"),
        "\u{e9}\u{105d2}\u{0307}\u{e9}"
    );
}

/// The pre-tokenizer's letter class is the reference's Unicode 16.0: a Unicode-16 letter is
/// `\p{L}`, a Unicode-17 letter is not. Today's `regex-syntax` tables are Unicode 16 too, so this
/// holds with or without the `\p{Age=16.0}` intersection in `PATTERN`; it is the check that fails
/// if a lockfile bump brings Unicode-17 tables and the intersection has been removed.
#[test]
fn letter_class_is_unicode_16() {
    let fixture = json("tokenizer_synthetic.json");
    let tok = synthetic();
    for name in ["letters_unicode16", "letters_unicode17"] {
        let c = case(&fixture, "encode", name);
        assert_eq!(
            tok.encode(c["text"].as_str().unwrap()).unwrap(),
            ids(&c["ids"])
        );
    }
    let pieces: Vec<&str> = tok
        .pattern
        .find_iter("a\u{10940}b")
        .map(|m| m.unwrap().as_str())
        .collect();
    assert_eq!(pieces, ["a", "\u{10940}b"]);
}

/// Special-token text in the input is ordinary text; it never becomes a special ID.
#[test]
fn special_token_text_encodes_as_ordinary_tokens() {
    let tok = synthetic();
    let ids = tok.encode("<|endoftext|><abc></abc>").unwrap();
    assert!(ids.iter().all(|&id| id < ORDINARY_TOKENS), "{ids:?}");
    assert_eq!(tok.decode(&ids), "<|endoftext|><abc></abc>");
}

#[test]
fn special_token_table_matches_upstream_slots() {
    assert_eq!(special_token(151_643).as_deref(), Some("<|endoftext|>"));
    assert_eq!(special_token(151_650).as_deref(), Some("<sep>"));
    assert_eq!(special_token(151_651).as_deref(), Some("<extra_0>"));
    assert_eq!(special_token(151_846).as_deref(), Some("<extra_195>"));
    assert_eq!(special_token(151_847).as_deref(), Some("<abc>"));
    assert_eq!(special_token(151_848).as_deref(), Some("</abc>"));
    assert_eq!(special_token(151_849).as_deref(), Some("<extra_198>"));
    assert_eq!(special_token(151_850).as_deref(), Some("<extra_199>"));
    assert_eq!(special_token(151_851), None);
    assert_eq!(special_token(151_642), None);
    assert_eq!(TEXT_VOCAB, 151_851);
}

fn table(lines: &[(&[u8], usize)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (token, rank) in lines {
        out.extend(base64_encode(token).bytes());
        out.extend(format!(" {rank}\n").bytes());
    }
    out
}

fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn bytes_table() -> Vec<(Vec<u8>, usize)> {
    (0..=255u8).map(|b| (vec![b], b as usize)).collect()
}

fn as_lines(t: &[(Vec<u8>, usize)]) -> Vec<u8> {
    table(&t.iter().map(|(b, r)| (&b[..], *r)).collect::<Vec<_>>())
}

/// The production parser requires exactly the checkpoint's 151643 ordinary tokens.
#[test]
fn production_parser_refuses_a_table_that_is_not_checkpoint_native() {
    let err = Yue2TextTokenizer::from_tiktoken_bytes(&as_lines(&bytes_table())).unwrap_err();
    assert!(
        matches!(err, TokenizerError::Vocabulary { ref found, .. } if found == "256 tokens"),
        "{err}"
    );
}

#[test]
fn malformed_tables_are_refused() {
    let mut dup_rank = bytes_table();
    dup_rank.push((b"ab".to_vec(), 7));
    assert!(matches!(
        Yue2TextTokenizer::from_partial_ranks_for_tests(&as_lines(&dup_rank)),
        Err(TokenizerError::Malformed { .. })
    ));
    let mut dup_token = bytes_table();
    dup_token.push((b"a".to_vec(), 256));
    assert!(matches!(
        Yue2TextTokenizer::from_partial_ranks_for_tests(&as_lines(&dup_token)),
        Err(TokenizerError::Vocabulary { .. })
    ));
    let mut gap = bytes_table();
    gap.push((b"ab".to_vec(), 300));
    assert!(matches!(
        Yue2TextTokenizer::from_partial_ranks_for_tests(&as_lines(&gap)),
        Err(TokenizerError::Vocabulary { .. })
    ));
    let missing_byte: Vec<_> = bytes_table()
        .into_iter()
        .map(|(b, r)| {
            if r == 0x41 {
                (b"xy".to_vec(), r)
            } else {
                (b, r)
            }
        })
        .collect();
    assert!(matches!(
        Yue2TextTokenizer::from_partial_ranks_for_tests(&as_lines(&missing_byte)),
        Err(TokenizerError::Vocabulary { ref found, .. }) if found.contains("0x41")
    ));
    for bad in [
        &b"QQ== x\n"[..],
        b"QQ==\n",
        b"Q!== 1\n",
        b"QQ== 1 2\n",
        b"QR== 1\n",
    ] {
        assert!(
            matches!(
                Yue2TextTokenizer::from_partial_ranks_for_tests(bad),
                Err(TokenizerError::Malformed { .. })
            ),
            "{:?}",
            String::from_utf8_lossy(bad)
        );
    }
}

#[test]
fn base64_is_strict_and_standard() {
    for (text, bytes) in [
        ("QQ==", &b"A"[..]),
        ("QUI=", b"AB"),
        ("QUJD", b"ABC"),
        ("//79", b"\xff\xfe\xfd"),
    ] {
        assert_eq!(
            base64_decode(text.as_bytes()).as_deref(),
            Some(bytes),
            "{text}"
        );
        assert_eq!(base64_encode(bytes), text);
    }
    for bad in ["", "QQ", "QQ=", "Q===", "QQ==QQ==", "QR==", "QUJ=", "QU-D"] {
        assert_eq!(base64_decode(bad.as_bytes()), None, "{bad}");
    }
}

/// Merging is by lowest rank, not left to right: with `bc` ranked before `ab`, `abc` splits as
/// `a` + `bc`.
#[test]
fn merges_follow_rank_order_not_position() {
    let mut t = bytes_table();
    t.push((b"bc".to_vec(), 256));
    t.push((b"ab".to_vec(), 257));
    let tok = Yue2TextTokenizer::from_partial_ranks_for_tests(&as_lines(&t)).unwrap();
    assert_eq!(tok.encode("abc").unwrap(), vec![b'a' as u32, 256]);
    assert_eq!(tok.encode("ab").unwrap(), vec![257]);
    assert_eq!(tok.encode("abab").unwrap(), vec![257, 257]);
    // A recurring pair merges leftmost first: `aaa` with `aa` ranked is `aa` + `a`.
    let mut t = bytes_table();
    t.push((b"aa".to_vec(), 256));
    let tok = Yue2TextTokenizer::from_partial_ranks_for_tests(&as_lines(&t)).unwrap();
    assert_eq!(tok.encode("aaa").unwrap(), vec![256, b'a' as u32]);
}

/// The straightforward merge (re-rank every adjacent pair, join the lowest, leftmost on ties) —
/// the oracle the linked-list/heap merge must agree with.
fn reference_merge(piece: &[u8], rank_of: impl Fn(&[u8]) -> Option<u32>) -> Vec<usize> {
    let mut bounds: Vec<usize> = (0..=piece.len()).collect();
    loop {
        let mut best: Option<(u32, usize)> = None;
        for i in 0..bounds.len().saturating_sub(2) {
            if let Some(rank) = rank_of(&piece[bounds[i]..bounds[i + 2]]) {
                if best.is_none_or(|(r, _)| rank < r) {
                    best = Some((rank, i));
                }
            }
        }
        match best {
            Some((_, i)) => {
                bounds.remove(i + 1);
            }
            None => return bounds,
        }
    }
}

/// Deterministic pseudo-random text over the synthetic table's corpus alphabet (English, CJK,
/// ABC punctuation), so pieces hit many merges, recurring pairs and byte fallbacks.
fn pseudo_random_text(seed: u32, chars: usize) -> String {
    let alphabet: Vec<char> = "the nightaeiou 夜色我等风吹过那条老街|:/[]ABCDEFG,'"
        .chars()
        .collect();
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..chars)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            alphabet[state as usize % alphabet.len()]
        })
        .collect()
}

/// The heap merge agrees with the reference merge on every piece of 300 pseudo-random texts and
/// on long unbroken runs.
#[test]
fn byte_pair_merge_matches_the_reference_merge() {
    let tok = synthetic();
    let rank_of = |b: &[u8]| tok.encoder.get(b).copied();
    let mut pieces = 0;
    for seed in 0..300 {
        let text = pseudo_random_text(seed, 1 + seed as usize % 97);
        for piece in tok.pattern.find_iter(&text) {
            let piece = piece.unwrap().as_str().as_bytes();
            assert_eq!(
                byte_pair_merge(piece, rank_of),
                reference_merge(piece, rank_of),
                "{:?}",
                String::from_utf8_lossy(piece)
            );
            pieces += 1;
        }
    }
    assert!(pieces > 1000, "{pieces}");
    for run in [
        "nightnightnight",
        "aaaaaaaaaaaaa",
        "夜色夜色夜色夜色",
        "  \n \n  ",
    ] {
        let piece = run.repeat(20);
        assert_eq!(
            byte_pair_merge(piece.as_bytes(), rank_of),
            reference_merge(piece.as_bytes(), rank_of)
        );
    }
}

/// One long unbroken piece costs a linear number of rank lookups (at most 3 per byte), not one
/// per adjacent pair per merge. Counting lookups bounds the work without a timing assertion; the
/// merge loop around them is a heap, `O(log n)` per lookup.
#[test]
fn merge_work_is_linear_in_the_piece() {
    let tok = synthetic();
    for text in [
        "night".repeat(6_554),
        "夜色我等风吹过那条老街".repeat(1_000),
        pseudo_random_text(7, 32_768).replace([' ', '|', ':', '/', '[', ']', ',', '\''], ""),
    ] {
        let piece = text.as_bytes();
        let mut lookups = 0usize;
        let bounds = byte_pair_merge(piece, |b| {
            lookups += 1;
            tok.encoder.get(b).copied()
        });
        let merges = piece.len() + 1 - bounds.len();
        assert!(
            merges * 8 > piece.len(),
            "only {merges} merges in {} bytes",
            piece.len()
        );
        assert!(
            lookups <= 3 * piece.len(),
            "{lookups} lookups for {} bytes",
            piece.len()
        );
        let ids = tok.encode(&text).unwrap();
        assert_eq!(tok.decode(&ids), text);
    }
}

/// Loading goes through verification: a `qwen.tiktoken` that fails the pinned size/hash never
/// reaches the parser.
#[test]
fn load_refuses_an_unverified_tiktoken() {
    let root = tempfile::tempdir().unwrap();
    let repo = ComponentId::QwenTiktoken.component().repo;
    let snap = root.path().join("snap");
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(snap.join(TIKTOKEN_FILE), as_lines(&bytes_table())).unwrap();
    let dirs = SnapshotDirs::new().with(repo.id, &snap);
    assert!(matches!(
        Yue2TextTokenizer::load(&dirs),
        Err(TokenizerError::Asset(_))
    ));
    let err = Yue2TextTokenizer::load(&SnapshotDirs::new()).unwrap_err();
    assert!(
        matches!(err, TokenizerError::Asset(AssetError::CacheMiss { .. })),
        "{err}"
    );
}
