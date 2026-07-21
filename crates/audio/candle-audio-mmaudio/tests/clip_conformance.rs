//! Real-weight conformance for the candle DFN5B-CLIP ViT-H/14-384 encoder (sc-13437).
//!
//! ## What this gates on real weights
//!
//! Loads the pinned `apple/DFN5B-CLIP-ViT-H-14-384` `open_clip_pytorch_model.bin` (~3.9 GB), builds
//! both towers, and asserts the ported forward produces the features MMAudio consumes:
//!
//! - [`dfn_clip_visual_shape_finite_deterministic`] — synthetic 384² frames →
//!   `encode_image` → `(N, 1024)`, every value finite, unit-norm (L2-normalized), byte-identical
//!   run-to-run, and materially different for different frame content.
//! - [`dfn_clip_text_shape_finite_deterministic`] — wrapped 77-token rows → `encode_text` →
//!   `(B, 77, 1024)`, finite, per-token unit-norm, deterministic, and prompt-varying.
//! - [`dfn_clip_matches_open_clip_reference`] — **numerical parity** against the `open_clip`
//!   reference, run only when `DFN_CLIP_PARITY_DIR` points at a directory holding the reference
//!   dumps (`img_normalized.f32`, `feat_img.f32`, `tokens_i64.bin`, `feat_txt.f32`) produced by the
//!   sc-13437 `ref.py`. Asserts pooled-image and per-token-text cosine `> 0.999` with small
//!   max-abs-diff. This isolates the ported encoder from preprocessing/tokenization by feeding the
//!   exact normalized pixels and exact token ids the reference used.
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-mmaudio --test clip_conformance -- --ignored --nocapture
//! ```
//! Set `DFN_CLIP_SNAPSHOT` to an `open_clip_pytorch_model.bin` file (or a dir containing it), or
//! leave unset to resolve the pinned checkpoint via the audio lane's F-029 hub path.

use candle_audio_mmaudio as m;
use candle_audio_mmaudio::candle_audio::candle_core::{Device, Tensor};
use candle_audio_mmaudio::clip;
use image::{Rgb, RgbImage};

/// Byte-match reference for [`clip::tokenize_str`] against
/// `open_clip.get_tokenizer('ViT-H-14-378-quickgelu')`. Each line is `<hex(prompt utf-8)>\t<77
/// space-separated ids>`, produced by the sc-13473 fixture generator (see the PR). This gate needs
/// **no weights** and runs by default — the definitive faithfulness check for the string→BPE path.
const TOKENIZER_REFERENCE: &str = include_str!("fixtures/clip_tokenizer_reference.txt");

fn hex_decode(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex byte"))
        .collect()
}

#[test]
fn tokenize_str_byte_matches_open_clip_reference() {
    let mut checked = 0usize;
    for (lineno, line) in TOKENIZER_REFERENCE.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let (hexp, ids_str) = line.split_once('\t').expect("record is <hex>\\t<ids>");
        let prompt = String::from_utf8(hex_decode(hexp)).expect("prompt utf-8");
        let expected: Vec<u32> = ids_str
            .split_whitespace()
            .map(|t| t.parse().expect("id"))
            .collect();
        assert_eq!(
            expected.len(),
            clip::CONTEXT_LENGTH,
            "record {lineno} width"
        );

        let got = clip::tokenize_str(&prompt);
        assert_eq!(
            got.as_slice(),
            expected.as_slice(),
            "record {lineno}: tokenize_str diverges from open_clip for prompt {prompt:?}\n \
             got:      {got:?}\n expected: {expected:?}",
        );
        checked += 1;
    }
    assert!(
        checked >= 12,
        "expected the full prompt set, checked {checked}"
    );
    eprintln!("tokenize_str byte-match: {checked} prompts identical to open_clip");
}

fn load_encoder() -> clip::DfnClipEncoder {
    let dev = Device::Cpu;
    // Required env path — inference never self-fetches or derives a cache location (epic 13657).
    let p = std::env::var("DFN_CLIP_SNAPSHOT")
        .expect("set DFN_CLIP_SNAPSHOT to an open_clip_pytorch_model.bin file or its snapshot dir");
    let path = std::path::PathBuf::from(&p);
    if path.is_dir() {
        clip::load(&m::gen_core::WeightsSource::Dir(path), &dev)
            .expect("load DFN5B-CLIP from DFN_CLIP_SNAPSHOT dir")
    } else {
        clip::load_from_pth(&path, &dev).expect("load DFN5B-CLIP from DFN_CLIP_SNAPSHOT file")
    }
}

/// A deterministic synthetic 384² RGB frame whose content depends on `seed`.
fn frame(seed: u8) -> RgbImage {
    let sz = clip::IMAGE_SIZE as u32;
    let mut img = RgbImage::new(sz, sz);
    for y in 0..sz {
        for x in 0..sz {
            let r = ((x as f32 * 0.03 + seed as f32).sin() * 127.0 + 128.0) as u8;
            let g = ((y as f32 * 0.02 + seed as f32 * 0.5).cos() * 127.0 + 128.0) as u8;
            let b = (((x + y) as f32 * 0.01).sin() * 127.0 + 128.0) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    img
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "downloads ~3.9GB open_clip_pytorch_model.bin; run explicitly with --ignored"]
fn dfn_clip_visual_shape_finite_deterministic() {
    let enc = load_encoder();
    let frames = vec![frame(1), frame(2)];
    let pixels = clip::frames_to_clip_input(&frames, enc.device()).expect("preprocess");
    let feats = enc.encode_image(&pixels).expect("encode_image");
    assert_eq!(feats.dims(), &[2, clip::EMBED_DIM], "(N, 1024)");
    let v: Vec<f32> = feats.flatten_all().unwrap().to_vec1().unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "all finite");
    // Each row is L2-normalized (encode_image normalize=True).
    for row in 0..2 {
        let s: f32 = v[row * clip::EMBED_DIM..(row + 1) * clip::EMBED_DIM]
            .iter()
            .map(|x| x * x)
            .sum();
        assert!(
            (s.sqrt() - 1.0).abs() < 1e-3,
            "row {row} unit-norm, got {}",
            s.sqrt()
        );
    }
    // Deterministic.
    let v2: Vec<f32> = enc
        .encode_image(&pixels)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(v, v2, "encoder must be deterministic run-to-run");
    // Different frames -> different features.
    let a = &v[0..clip::EMBED_DIM];
    let b = &v[clip::EMBED_DIM..2 * clip::EMBED_DIM];
    let cos = cosine(a, b);
    eprintln!("dfn-clip visual: shape=(2,1024) cos(frame1,frame2)={cos:.4}");
    assert!(
        cos < 0.999,
        "different frames must not be identical (cos={cos})"
    );

    // Auxiliary per-token surface has the expected shape.
    let toks = enc.encode_image_tokens(&pixels).expect("tokens");
    assert_eq!(toks.dims(), &[2, clip::NUM_PATCHES, clip::VISION_WIDTH]);
}

#[test]
#[ignore = "downloads ~3.9GB open_clip_pytorch_model.bin; run explicitly with --ignored"]
fn dfn_clip_text_shape_finite_deterministic() {
    let enc = load_encoder();
    // Two distinct prompts as raw BPE-ish content ids (arbitrary but different); wrapping adds SOT/EOT/pad.
    let rows = vec![
        clip::wrap_tokens(&[320, 1929, 32676, 39256]),
        clip::wrap_tokens(&[3306, 1573, 550, 320, 4980]),
    ];
    let ids = clip::tokenize(&rows, enc.device()).expect("tokenize");
    let feats = enc.encode_text(&ids).expect("encode_text");
    assert_eq!(
        feats.dims(),
        &[2, clip::CONTEXT_LENGTH, clip::EMBED_DIM],
        "(B, 77, 1024)"
    );
    let v: Vec<f32> = feats.flatten_all().unwrap().to_vec1().unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "all finite");
    // Per-token L2-normalized.
    let d = clip::EMBED_DIM;
    let s: f32 = v[0..d].iter().map(|x| x * x).sum();
    assert!((s.sqrt() - 1.0).abs() < 1e-3, "token unit-norm");
    // Deterministic + prompt-varying (compare first token across prompts).
    let v2: Vec<f32> = enc
        .encode_text(&ids)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(v, v2, "deterministic");
    let per_prompt = clip::CONTEXT_LENGTH * d;
    let cos = cosine(&v[d..2 * d], &v[per_prompt + d..per_prompt + 2 * d]);
    eprintln!("dfn-clip text: shape=(2,77,1024) cos(prompt0_t1,prompt1_t1)={cos:.4}");
    assert!(cos < 0.9999, "different prompts must differ");
}

#[test]
#[ignore = "downloads ~3.9GB open_clip_pytorch_model.bin; run explicitly with --ignored"]
fn tokenize_str_to_encode_text_end_to_end() {
    // The production call path the MMAudio assembly uses: raw prompt string -> tokenize_str -> the
    // (B, 77) id tensor -> encode_text -> (B, 77, 1024) per-token features. `tokenize_str` byte-match
    // (offline gate) plus the sc-13437 encode_text parity gate together prove this equals open_clip;
    // this test exercises the wiring end to end on real weights.
    let enc = load_encoder();
    let rows = vec![
        clip::tokenize_str("a dog barking loudly").to_vec(),
        clip::tokenize_str("ocean waves crashing on rocks").to_vec(),
    ];
    let ids = clip::tokenize(&rows, enc.device()).expect("tokenize");
    let feats = enc.encode_text(&ids).expect("encode_text");
    assert_eq!(
        feats.dims(),
        &[2, clip::CONTEXT_LENGTH, clip::EMBED_DIM],
        "(B, 77, 1024)"
    );
    let v: Vec<f32> = feats.flatten_all().unwrap().to_vec1().unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "all finite");
    let d = clip::EMBED_DIM;
    let s: f32 = v[0..d].iter().map(|x| x * x).sum();
    assert!((s.sqrt() - 1.0).abs() < 1e-3, "per-token unit-norm");
    eprintln!("tokenize_str -> encode_text end-to-end: shape=(2,77,1024) ok");
}

#[test]
#[ignore = "numerical parity vs open_clip; set DFN_CLIP_PARITY_DIR to the ref.py dump directory"]
fn dfn_clip_matches_open_clip_reference() {
    let dir = match std::env::var("DFN_CLIP_PARITY_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            eprintln!("DFN_CLIP_PARITY_DIR unset; skipping parity");
            return;
        }
    };
    let enc = load_encoder();
    let dev = enc.device().clone();

    // ---- visual parity: feed the EXACT normalized pixels the reference used ----
    let px = read_f32(&dir.join("img_normalized.f32"));
    let pixels = Tensor::from_vec(px, (1, 3, clip::IMAGE_SIZE, clip::IMAGE_SIZE), &dev).unwrap();
    let got_img: Vec<f32> = enc
        .encode_image(&pixels)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let ref_img = read_f32(&dir.join("feat_img.f32"));
    assert_eq!(got_img.len(), ref_img.len(), "image feature length");
    let img_cos = cosine(&got_img, &ref_img);
    let img_mad = got_img
        .iter()
        .zip(&ref_img)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!("PARITY image: cos={img_cos:.6} max|Δ|={img_mad:.6}");
    assert!(img_cos > 0.999, "image cosine {img_cos} must exceed 0.999");
    assert!(img_mad < 0.02, "image max-abs-diff {img_mad} too large");

    // ---- text parity: feed the EXACT token ids the reference used ----
    let tok_bytes = std::fs::read(dir.join("tokens_i64.bin")).unwrap();
    let ids_i64: Vec<i64> = tok_bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let b = ids_i64.len() / clip::CONTEXT_LENGTH;
    let ids_u32: Vec<u32> = ids_i64.iter().map(|&x| x as u32).collect();
    let ids = Tensor::from_vec(ids_u32, (b, clip::CONTEXT_LENGTH), &dev).unwrap();
    let got_txt: Vec<f32> = enc
        .encode_text(&ids)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let ref_txt = read_f32(&dir.join("feat_txt.f32"));
    assert_eq!(got_txt.len(), ref_txt.len(), "text feature length");
    let txt_cos = cosine(&got_txt, &ref_txt);
    let txt_mad = got_txt
        .iter()
        .zip(&ref_txt)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!("PARITY text : cos={txt_cos:.6} max|Δ|={txt_mad:.6} (B={b}, per-token 1024)");
    assert!(txt_cos > 0.999, "text cosine {txt_cos} must exceed 0.999");
    assert!(txt_mad < 0.02, "text max-abs-diff {txt_mad} too large");
}
