//! sc-18775 — **is the LTX-2.5 text encoder quantized per tier, and what does it cost?**
//!
//! The Gemma 4 encoder is 26.3 GB of bf16 and 21.9 GB of that is attention/MLP projection weight —
//! 36 % of the whole LTX-2.5 bundle. Leaving it dense would make a "q4" tier a q4 transformer beside
//! a bf16 text encoder, which is not what a tier means (epic 18755 R5). Quantizing it without
//! measuring what that does to the embeddings the DiT is conditioned on would be the opposite
//! mistake. This test is the measurement the decision rests on.
//!
//! It takes the **dense** encoder — the `bf16` tier's `text_encoder.safetensors` — packs its 328
//! attention/MLP projections at 8 and at 4 bits *in this test*, runs the identical prompt through
//! all three, and reports how far each lands from bf16 on the hidden states LTX actually consumes:
//! all 49 of them, since the LTX-2.5 feature extractor concatenates the whole stack and per-token-RMS
//! normalizes it rather than reading the last one. Judging on the final layer alone would call q4
//! excellent (cos 0.999945); across the stack it is not (worst cos 0.889414).
//!
//! **Why it packs its own weights instead of reading each tier's file.** The measurement decided the
//! tiers: `q4` failed, so the shipped `q4` tier carries the *dense* encoder
//! (`mlx_gen_ltx::tiers::TEXT_ENCODER_Q4_QUALITY`). Reading the encoders back out of the tiers would
//! therefore make the evidence for a rejection depend on shipping the thing it rejected — the
//! q4 number would silently become a bf16-vs-bf16 comparison the moment the decision took effect.
//! Deriving both packed encoders here keeps the finding reproducible and independent of the layout
//! it produced. The shipped decision is pinned separately, at the bottom.
//!
//! ```text
//! LTX25_TIER_DIR=/path/to/scratch/ltx25-tiers \
//!   cargo test -p mlx-llm --release --test integration -- ltx_2_5_te_tier_quality:: \
//!     --ignored --nocapture
//! ```
//!
//! **Cost.** One 22.3 GiB dense load plus two in-test quantizations of it; ~25 GB peak. One model
//! resident at a time; run it alone.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::ops::quantize;
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use mlx_rs::{memory, Device, Dtype};
use serde_json::Value;

use mlx_llm::config::ModelConfig;
use mlx_llm::models::CausalLm;
use mlx_llm::primitives::{input_ids, Weights};

/// The widths measured, most-compressed first. `None` is the dense reference.
const WIDTHS: [(&str, Option<i32>); 3] = [("q4", Some(4)), ("q8", Some(8)), ("bf16", None)];

/// The affine group width the tiers are built at (`mlx_gen_ltx::tiers::DEFAULT_GROUP_SIZE`).
const GROUP: i32 = 64;

/// The Gemma 4 projections a tier packs — the same suffix set
/// `mlx_gen_ltx::tiers::GEMMA_QUANT_SUFFIXES` selects, and the exact set
/// `Projection::load_quantized` reads a `.scales` sibling for. `embed_tokens` is deliberately absent:
/// it is a lookup, not a matmul.
const GEMMA_QUANT_SUFFIXES: [&str; 7] = [
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
];

fn is_quantizable(key: &str) -> bool {
    key.strip_suffix(".weight")
        .is_some_and(|base| GEMMA_QUANT_SUFFIXES.iter().any(|s| base.ends_with(s)))
}

fn tier_root() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("LTX25_TIER_DIR")?);
    dir.join("bf16/text_encoder.safetensors")
        .is_file()
        .then_some(dir)
}

/// Runs a block on MLX's **CPU** stream, restoring the previous device on drop.
///
/// Packing 21.9 GB of projections is a bandwidth-bound offline job with no kernel worth
/// dispatching, and putting it on the Metal queue is what trips
/// `kIOGPUCommandBufferCallbackErrorTimeout` — after which every later submission in the process
/// returns `SubmissionsIgnored`, so the failure is not even local to the width that caused it. Both
/// were measured on sc-18775. RAII rather than a one-way switch, because the forwards below want the
/// GPU back.
struct CpuStream {
    previous: Device,
}

impl CpuStream {
    fn enter() -> Self {
        let previous = Device::try_default().expect("a default device");
        Device::set_default(&Device::cpu());
        Self { previous }
    }
}

impl Drop for CpuStream {
    fn drop(&mut self) {
        Device::set_default(&self.previous);
    }
}

/// Pack the dense encoder's projections at `bits`, returning the weight map a tier would have
/// written. `bits == None` returns the dense map unchanged.
fn pack(dense: &Weights, bits: Option<i32>) -> (Weights, usize) {
    let mut map: HashMap<String, Array> = HashMap::new();
    let mut packed = 0usize;
    let _cpu = CpuStream::enter();
    for key in dense.keys().map(str::to_string).collect::<Vec<_>>() {
        let value = dense
            .require(&key)
            .expect("a key the map just listed")
            .clone();
        let Some(bits) = bits.filter(|_| is_quantizable(&key)) else {
            map.insert(key, value);
            continue;
        };
        let shape = value.shape();
        assert_eq!(shape.len(), 2, "{key}: selected but not rank-2");
        assert_eq!(
            shape[1] % GROUP,
            0,
            "{key}: input axis {} % {GROUP}",
            shape[1]
        );
        let base = key
            .strip_suffix(".weight")
            .expect("checked by is_quantizable");
        let (q, scales, biases) =
            quantize(&value, GROUP, bits).unwrap_or_else(|e| panic!("quantize {key}: {e}"));
        eval([&q, &scales, &biases]).unwrap_or_else(|e| panic!("evaluating {key}: {e}"));
        map.insert(format!("{base}.weight"), q);
        map.insert(format!("{base}.scales"), scales);
        map.insert(format!("{base}.biases"), biases);
        packed += 1;
    }
    (Weights::from_map(map), packed)
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

/// Byte budget for one `eval` submission. Same bound the tier converter uses.
const EVAL_BATCH_BYTES: usize = 512 * 1024 * 1024;

/// Force a tier's mmapped tensors resident in **bounded batches**, before anything builds a graph
/// over them.
///
/// `Weights::from_file` is lazy: every tensor is an unevaluated `Load` over the mapped file. Left
/// that way, the first `eval` of the forward drags the whole cold file — 7.7 GB at q4, 24 GB at
/// bf16 — into a single Metal command buffer, and the watchdog kills it:
/// `kIOGPUCommandBufferCallbackErrorTimeout`, measured on this test's first real run (sc-18775).
/// Worse, one killed buffer poisons the process — every later submission returns
/// `kIOGPUCommandBufferCallbackErrorSubmissionsIgnored` — so the failure is not local to the tier
/// that caused it. Walking the tensors in ≤512 MB submissions keeps every buffer well inside the
/// watchdog's budget, which is the same reason `mlx_gen_ltx::tiers` batches its conversion evals.
fn materialize_in_batches(weights: &Weights) {
    let keys: Vec<String> = weights.keys().map(str::to_string).collect();
    let mut batch: Vec<Array> = Vec::new();
    let mut bytes = 0usize;
    for key in &keys {
        let array = weights
            .require(key)
            .expect("a key the map just listed")
            .clone();
        bytes = bytes.saturating_add(array.nbytes());
        batch.push(array);
        if bytes >= EVAL_BATCH_BYTES {
            eval(batch.iter()).unwrap_or_else(|e| panic!("materializing weights: {e}"));
            batch.clear();
            bytes = 0;
        }
    }
    if !batch.is_empty() {
        eval(batch.iter()).unwrap_or_else(|e| panic!("materializing weights: {e}"));
    }
}

/// The dense encoder's `gemma_config`, with a `quantization` block stamped in when `bits` is set.
///
/// Stamped into **`text_config`**, not the wrapper's top level: Gemma 4 `nests_text_config()`, so
/// `ModelConfig::from_json` rebinds to `text_config` before it looks for `quantization` and a block
/// written above it is invisible. That is the same placement `mlx_gen_ltx::tiers` writes, and
/// getting it wrong is what made the tiers' packed encoder unloadable on this test's first run.
fn config_for(path: &Path, bits: Option<i32>) -> ModelConfig {
    let meta = safetensors_metadata(path);
    let raw = meta
        .get("gemma_config")
        .expect("the dense encoder must carry its gemma_config");
    let mut json: Value = serde_json::from_str(raw).expect("parse gemma_config");
    assert!(
        json.get("quantization").is_none(),
        "the dense encoder must not already declare a quantization block"
    );
    if let Some(bits) = bits {
        let nested = json
            .get_mut("text_config")
            .and_then(Value::as_object_mut)
            .expect("a Gemma 4 wrapper nests its decoder fields under `text_config`");
        nested.insert(
            "quantization".to_string(),
            serde_json::json!({ "group_size": GROUP, "bits": bits, "mode": "affine" }),
        );
    }
    let cfg = ModelConfig::from_json(&json).expect("the config must parse");
    assert!(
        cfg.is_gemma4(),
        "{}: must be a Gemma 4 config",
        path.display()
    );
    assert_eq!(
        cfg.quantization.map(|q| (q.bits, q.group_size)),
        bits.map(|b| (b, GROUP)),
        "the stamped block must be the one `from_json` reads"
    );
    cfg
}

/// One width's run: every hidden state for the fixed prompt, plus what it cost and what geometry the
/// decoder actually bound.
struct Measured {
    /// The 49 hidden states, host-side as f32.
    states: Vec<Vec<f32>>,
    pack_s: f64,
    forward_s: f64,
    /// `(bits, group_size)` the built decoder read from its config — `None` for the dense run.
    quant: Option<(i32, i32)>,
}

/// Every hidden state of one width's encoder for the fixed prompt.
fn hidden_states(path: &Path, dense: &Weights, bits: Option<i32>, prompt: &[i32]) -> Measured {
    let cfg = config_for(path, bits);
    let quant = cfg.quantization.map(|q| (q.bits, q.group_size));

    let t0 = Instant::now();
    let (weights, packed) = pack(dense, bits);
    assert_eq!(
        packed > 0,
        bits.is_some(),
        "a packed width must actually pack projections, a dense one must pack none"
    );
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
    Measured {
        states: out,
        pack_s: load_s,
        forward_s,
        quant,
    }
}

/// **The TE quantization decision, measured.**
///
/// Reports, per width, the cosine similarity and relative L2 of every hidden state against the dense
/// encoder's, plus the worst case across the 49-state stack the LTX-2.5 feature extractor
/// concatenates. The *numbers* printed here are the decision record; the assertions pin the finding
/// they produced — **q8 passes the bar and q4 does not** — so that a later change which quietly made
/// q4 acceptable, or q8 unacceptable, is a red test rather than a silent drift away from the tier
/// layout this measurement chose.
#[test]
#[ignore = "sc-18775: needs the built LTX-2.5 tiers (LTX25_TIER_DIR); one 22.3 GiB load, ~25 GB peak"]
fn the_text_encoder_survives_quantization_at_every_tier() {
    let Some(root) = tier_root() else {
        eprintln!("skip: set LTX25_TIER_DIR to a directory holding the built q4/q8/bf16 tiers");
        return;
    };
    // A real caption, not a token-index ramp: the quantization error that matters is the one on the
    // activations a prompt actually produces.
    let prompt: Vec<i32> = (0..96).map(|i| ((i * 1543 + 7) % 250_000) + 5).collect();

    // Every width is packed from the *same* dense encoder, so any difference measured below is the
    // quantization and nothing else.
    let path = root.join("bf16").join("text_encoder.safetensors");
    let bytes = std::fs::metadata(&path).unwrap().len();
    let dense = Weights::from_file(&path).expect("load the dense text encoder");
    materialize_in_batches(&dense);
    eprintln!(
        "[TE source] {:.2} GiB on disk, {} tensors",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        dense.len()
    );

    let mut by_width: Vec<(&str, Measured)> = Vec::new();
    for (width, bits) in WIDTHS {
        let run = hidden_states(&path, &dense, bits, &prompt);
        assert_eq!(
            run.states.len(),
            49,
            "{width}: 48 layers plus the input embeddings"
        );
        for (i, s) in run.states.iter().enumerate() {
            assert!(
                s.iter().all(|x| x.is_finite()),
                "{width}: hidden_states[{i}] is not finite"
            );
        }
        eprintln!(
            "[TE {width}] quant {:?}, pack {:.1}s, forward({} tokens) {:.1}s, peak {:.1} GB",
            run.quant,
            run.pack_s,
            prompt.len(),
            run.forward_s,
            memory::get_peak_memory() as f64 / 1e9,
        );
        by_width.push((width, run));
    }

    let reference = &by_width
        .iter()
        .find(|(w, _)| *w == "bf16")
        .expect("a dense reference")
        .1
        .states;
    let mut worst: Vec<(&str, f64, f64)> = Vec::new();
    for (width, run) in &by_width {
        let states = &run.states;
        assert_eq!(
            run.quant.is_some(),
            *width != "bf16",
            "{width}: the `quantization` block must be present iff the weights are packed"
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
                eprintln!("[TE {width}] hidden_states[{i}]: cos {c:.6}  rel_l2 {r:.5}");
            }
        }
        eprintln!("[TE {width}] worst over 49 states: cos {min_cos:.6}  rel_l2 {max_rel:.5}");
        worst.push((width, min_cos, max_rel));
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
    // Packed weights that were never bound would land here at cos 1.0 / rel 0 — identical to bf16.
    // That is the failure this pair of bounds exists to catch, in both directions.
    assert!(
        q8_rel > 0.0 && q4_rel > 0.0,
        "a packed width must actually differ from bf16 — equality means the packed weights were \
         not bound: q8 {q8_rel:.3e}, q4 {q4_rel:.3e}"
    );
    assert!(
        q8_rel < q4_rel && q8_cos > q4_cos,
        "8-bit must be closer to bf16 than 4-bit: q8 (cos {q8_cos:.6}, rel {q8_rel:.5}) vs \
         q4 (cos {q4_cos:.6}, rel {q4_rel:.5})"
    );
    // The bars: an encoder whose embeddings still point the same way. Set well outside the values
    // measured on sc-18775 so ordinary MLX-version drift does not trip them, and far inside the
    // "this is a different embedding" regime a mis-bound projection or a wrong group size lands in
    // (cosine collapses toward 0, rel_l2 explodes past 1).
    const Q8_BAR: (f64, f64) = (0.995, 0.10);
    const Q4_BAR: (f64, f64) = (0.97, 0.30);
    assert!(
        q8_cos > Q8_BAR.0 && q8_rel < Q8_BAR.1,
        "q8 text encoder drifted too far from bf16: cos {q8_cos:.6}, rel_l2 {q8_rel:.5}"
    );
    // **The finding, pinned.** q4 measured cos 0.889414 / rel_l2 0.53488 on 2026-08-25 — outside the
    // bar in both coordinates — which is why the shipped q4 tier carries the *dense* encoder
    // (`mlx_gen_ltx::tiers::TEXT_ENCODER_Q4_QUALITY`). Asserting the failure rather than deleting the
    // bar is what keeps the tier layout and this measurement tied together: if a future MLX or a
    // different group width brought q4 inside the bar, this goes red and the exemption gets
    // revisited, instead of the tier quietly shipping a dense encoder nothing justifies any more.
    assert!(
        !(q4_cos > Q4_BAR.0 && q4_rel < Q4_BAR.1),
        "q4 now MEETS the quality bar (cos {q4_cos:.6} > {}, rel_l2 {q4_rel:.5} < {}) — the \
         measured basis for shipping a dense encoder in the q4 tier no longer holds; re-run the \
         decision in `mlx_gen_ltx::tiers::text_encoder_dense_reason`",
        Q4_BAR.0,
        Q4_BAR.1
    );
}

/// **The decision, as shipped.** The `q4` tier carries a dense encoder; `q8` carries a packed one.
///
/// Separate from the measurement above on purpose: that test proves *what quantizing costs*, this
/// one proves *what the converter did about it*. Keeping them in one test would let a converter
/// regression hide behind a passing measurement, or vice versa.
#[test]
#[ignore = "sc-18775: needs the built LTX-2.5 tiers (LTX25_TIER_DIR)"]
fn the_shipped_tiers_match_the_measured_text_encoder_decision() {
    let Some(root) = tier_root() else {
        eprintln!("skip: set LTX25_TIER_DIR to a directory holding the built q4/q8/bf16 tiers");
        return;
    };
    for (tier, packed) in [("q4", false), ("q8", true), ("bf16", false)] {
        let path = root.join(tier).join("text_encoder.safetensors");
        let meta = safetensors_metadata(&path);
        let json: Value = serde_json::from_str(&meta["gemma_config"]).expect("gemma_config");
        // Read the block where `ModelConfig::from_json` reads it, not where it is convenient to look.
        let declared = json
            .get("text_config")
            .and_then(|t| t.get("quantization"))
            .is_some();
        assert_eq!(
            declared, packed,
            "{tier}: `text_config.quantization` must be declared iff this tier packs the encoder"
        );
        assert!(
            json.get("quantization").is_none(),
            "{tier}: the block belongs in `text_config`, not the wrapper's top level"
        );
        let cfg = ModelConfig::from_json(&json).expect("the tier's config must parse");
        assert_eq!(
            cfg.quantization.is_some(),
            packed,
            "{tier}: what `mlx_llm` actually reads must agree with what the tier shipped"
        );
        eprintln!("[TE shipped] {tier}: packed={declared}");
    }
}
