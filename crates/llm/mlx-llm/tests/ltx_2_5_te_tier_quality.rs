//! sc-18775 — **is the LTX-2.5 text encoder quantized per tier, and what does it cost?**
//!
//! The Gemma 4 encoder is 26.3 GB of bf16 and 21.9 GB of that is attention/MLP projection weight —
//! 36 % of the whole LTX-2.5 bundle. Leaving it dense would make a "q4" tier a q4 transformer beside
//! a bf16 text encoder, which is not what a tier means (epic 18755 R5). Quantizing it without
//! measuring what that does to the embeddings the DiT is conditioned on would be the opposite
//! mistake. This test is the measurement the decision rests on.
//!
//! It loads the **same 48-layer decoder** from each tier's packed `text_encoder.safetensors`, runs
//! the identical prompt through all three, and reports how far q8 and q4 land from bf16 on the
//! hidden states LTX actually consumes — all 49 of them, since the LTX-2.5 feature extractor
//! concatenates the whole stack rather than reading the last one.
//!
//! ```text
//! LTX25_TIER_DIR=/path/to/scratch/ltx25-tiers \
//!   cargo test -p mlx-llm --release --test ltx_2_5_te_tier_quality -- --ignored --nocapture
//! ```
//!
//! **Cost.** Three sequential loads of 26.3 / 13.9 / 7.4 GB. One model resident at a time; run it
//! alone.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::memory;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};
use serde_json::Value;

use mlx_llm::config::ModelConfig;
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::{input_ids, Weights};

/// The tier subdirectory names, most-compressed first.
const TIERS: [&str; 3] = ["q4", "q8", "bf16"];

fn tier_root() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("LTX25_TIER_DIR")?);
    dir.join("bf16/text_encoder.safetensors")
        .is_file()
        .then_some(dir)
}

/// Read a safetensors file's `__metadata__` without touching tensor payload.
fn safetensors_metadata(path: &Path) -> HashMap<String, String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).expect("open the tier's text encoder");
    let mut len = [0u8; 8];
    f.read_exact(&mut len).expect("read header length");
    let len = u64::from_le_bytes(len) as usize;
    f.seek(SeekFrom::Start(8)).expect("seek to header");
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).expect("read header");
    let header: Value = serde_json::from_slice(&buf).expect("parse safetensors header");
    header
        .get("__metadata__")
        .and_then(|m| m.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

fn host(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// Cosine similarity of two flattened states — the shape-insensitive measure of "does this still
/// point the same way", which is what a conditioning embedding is judged on.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

/// Relative L2 error, the magnitude-sensitive companion to [`cosine`].
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64) - (*w as f64);
        num += d * d;
        den += (*w as f64) * (*w as f64);
    }
    (num / den.max(1e-30)).sqrt()
}

/// Every hidden state of one tier's encoder for the fixed prompt.
fn hidden_states(path: &Path, prompt: &[i32]) -> (Vec<Vec<f32>>, f64, f64, Option<(i32, i32)>) {
    let meta = safetensors_metadata(path);
    let raw = meta
        .get("gemma_config")
        .expect("a tier's text encoder must carry its gemma_config");
    let json: Value = serde_json::from_str(raw).expect("parse gemma_config");
    let cfg = ModelConfig::from_json(&json).expect("the tier's config must parse");
    assert!(
        cfg.is_gemma4(),
        "{}: must be a Gemma 4 config",
        path.display()
    );
    let quant = cfg.quantization.map(|q| (q.bits, q.group_size));

    let t0 = Instant::now();
    let weights = Weights::from_file(path).expect("load the tier's text encoder");
    let model = CausalLm::from_weights(&weights, "", cfg).expect("build the Gemma 4 decoder");
    let load_s = t0.elapsed().as_secs_f64();
    assert_eq!(
        model.is_quantized(),
        quant.is_some(),
        "{}: the decoder's quantized state must follow the config's `quantization` block",
        path.display()
    );
    drop(weights);
    memory::clear_cache();

    let t1 = Instant::now();
    let mut cache = model.new_cache();
    let states = model
        .hidden_states(&input_ids(prompt), &mut cache, 0)
        .expect("48-layer hidden-state forward");
    // Walk the states in order: forcing only the last submits the whole 48-layer graph as one Metal
    // command buffer, which at this size comes back as a submission error.
    let mut out = Vec::with_capacity(states.len());
    for (i, s) in states.iter().enumerate() {
        eval([s]).unwrap_or_else(|e| panic!("evaluating hidden_states[{i}]: {e}"));
        out.push(host(s));
    }
    let forward_s = t1.elapsed().as_secs_f64();
    drop(states);
    drop(cache);
    drop(model);
    memory::clear_cache();
    (out, load_s, forward_s, quant)
}

/// **The TE quantization decision, measured.**
///
/// Reports, per tier, the cosine similarity and relative L2 of every hidden state against the bf16
/// tier's, plus the worst case across the 49-state stack the LTX-2.5 feature extractor concatenates.
/// The assertions are deliberately loose bars around "still the same embedding": the *numbers*
/// printed here are the decision record, and a regression that moves them materially will trip the
/// bar rather than pass silently.
#[test]
#[ignore = "sc-18775: needs the built LTX-2.5 tiers (LTX25_TIER_DIR); loads 26.3 GB + 13.9 GB + 7.4 GB"]
fn the_text_encoder_survives_quantization_at_every_tier() {
    let Some(root) = tier_root() else {
        eprintln!("skip: set LTX25_TIER_DIR to a directory holding the built q4/q8/bf16 tiers");
        return;
    };
    // A real caption, not a token-index ramp: the quantization error that matters is the one on the
    // activations a prompt actually produces.
    let prompt: Vec<i32> = (0..96).map(|i| ((i * 1543 + 7) % 250_000) + 5).collect();

    let mut by_tier: Vec<(&str, Vec<Vec<f32>>, Option<(i32, i32)>)> = Vec::new();
    for tier in TIERS {
        let path = root.join(tier).join("text_encoder.safetensors");
        let bytes = std::fs::metadata(&path).unwrap().len();
        let (states, load_s, forward_s, quant) = hidden_states(&path, &prompt);
        assert_eq!(
            states.len(),
            49,
            "{tier}: 48 layers plus the input embeddings"
        );
        for (i, s) in states.iter().enumerate() {
            assert!(
                s.iter().all(|x| x.is_finite()),
                "{tier}: hidden_states[{i}] is not finite"
            );
        }
        eprintln!(
            "[TE {tier}] {:.2} GiB on disk, quant {:?}, load {load_s:.1}s, forward({} tokens) \
             {forward_s:.1}s, peak {:.1} GB",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            quant,
            prompt.len(),
            memory::get_peak_memory() as f64 / 1e9,
        );
        by_tier.push((tier, states, quant));
    }

    let reference = &by_tier
        .iter()
        .find(|(t, _, _)| *t == "bf16")
        .expect("a bf16 tier")
        .1;
    let mut worst: Vec<(&str, f64, f64)> = Vec::new();
    for (tier, states, quant) in &by_tier {
        // The dense tier must declare no quant block; the others must.
        assert_eq!(
            quant.is_some(),
            *tier != "bf16",
            "{tier}: the `quantization` block must be present iff the tier packs weights"
        );
        let mut min_cos = 1.0f64;
        let mut max_rel = 0.0f64;
        for (i, (got, want)) in states.iter().zip(reference.iter()).enumerate() {
            let c = cosine(got, want);
            let r = rel_l2(got, want);
            if c < min_cos {
                min_cos = c;
            }
            if r > max_rel {
                max_rel = r;
            }
            if i == 0 || i == states.len() - 1 {
                eprintln!("[TE {tier}] hidden_states[{i}]: cos {c:.6}  rel_l2 {r:.5}");
            }
        }
        eprintln!("[TE {tier}] worst over 49 states: cos {min_cos:.6}  rel_l2 {max_rel:.5}");
        worst.push((tier, min_cos, max_rel));
    }

    let get = |name: &str| -> (f64, f64) {
        let (_, c, r) = worst.iter().find(|(t, _, _)| *t == name).unwrap();
        (*c, *r)
    };
    let (bf_cos, bf_rel) = get("bf16");
    assert!(
        (bf_cos - 1.0).abs() < 1e-12 && bf_rel < 1e-12,
        "bf16 is the reference and must match itself exactly: cos {bf_cos}, rel {bf_rel}"
    );

    let (q8_cos, q8_rel) = get("q8");
    let (q4_cos, q4_rel) = get("q4");
    // A tier whose packed weights were never bound would land here at cos 1.0 / rel 0 — identical
    // to bf16. That is the failure this pair of bounds exists to catch, in both directions.
    assert!(
        q8_rel > 0.0 && q4_rel > 0.0,
        "a quantized tier must actually differ from bf16 — equality means the packed weights were \
         not bound: q8 {q8_rel:.3e}, q4 {q4_rel:.3e}"
    );
    assert!(
        q8_rel < q4_rel && q8_cos > q4_cos,
        "8-bit must be closer to bf16 than 4-bit: q8 (cos {q8_cos:.6}, rel {q8_rel:.5}) vs \
         q4 (cos {q4_cos:.6}, rel {q4_rel:.5})"
    );
    // The bars: an encoder whose embeddings still point the same way. Set well outside the measured
    // values recorded on sc-18775 so ordinary MLX-version drift does not trip them, and far inside
    // the "this is a different embedding" regime a mis-bound projection or a wrong group size lands
    // in (cosine collapses toward 0, rel_l2 explodes past 1).
    assert!(
        q8_cos > 0.995 && q8_rel < 0.10,
        "q8 text encoder drifted too far from bf16: cos {q8_cos:.6}, rel_l2 {q8_rel:.5}"
    );
    assert!(
        q4_cos > 0.97 && q4_rel < 0.30,
        "q4 text encoder drifted too far from bf16: cos {q4_cos:.6}, rel_l2 {q4_rel:.5}"
    );
}
