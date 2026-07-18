//! sc-12831 (epic 12732) — **UMT5 TE quantization: parity + TE-encode active-peak** on real Mac
//! weights. The companion to `ti2v_5b_offload_footprint.rs` (which asserts the whole-generation floor):
//! this isolates the **text encoder** — the story's actual subject — to (a) characterize the numerics
//! delta of packing the UMT5 projections (Q4/Q8) vs the bf16 baseline and (b) measure the pure
//! TE-encode active peak at **both** q4 and q8 (the TE file is the same ~11 GB bf16 at every DiT tier,
//! so its quantized peak is tier-independent and measurable here without a full q8 5B snapshot).
//!
//! **Parity posture (parity-reframe):** packing the weights is a *numerics change*, NOT bit-exact — the
//! goal is "no quality regression," quantified here as high cosine similarity of the prompt embedding to
//! the bf16 baseline (Q8 ≈ lossless; Q4 aggressive-but-coherent). Activations stay f32 throughout
//! (`quantized_matmul` accumulates fp32), so the unscaled-logit softmax the module doc calls out is
//! unaffected — only the projection weights are lower precision.
//!
//! `#[ignore]` + env-gated, GPU-heavy. Point `WAN_TI2V_5B_MODEL_DIR` at any converted TI2V-5B tier dir
//! (its `t5_encoder.safetensors` + `tokenizer.json` — the bf16 TE is identical across tiers):
//!
//! ```text
//! WAN_TI2V_5B_MODEL_DIR=~/.cache/huggingface/hub/models--SceneWorks--wan2.2-ti2v-5b-mlx/snapshots/<h>/q4 \
//!   cargo test -p mlx-gen-wan --test te_quant_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

use mlx_gen::weights::Weights;
use mlx_gen_wan::{load_tokenizer, Umt5Encoder, WanModelConfig, WanQuant};

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(|s| {
        let s = s.to_string_lossy();
        if let Some(rest) = s.strip_prefix("~/") {
            if let Some(home) = std::env::var_os("HOME") {
                return PathBuf::from(format!("{}/{rest}", home.to_string_lossy()));
            }
        }
        PathBuf::from(s.to_string())
    })
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Cosine similarity + max|Δ| of two equal-shape f32 embedding buffers (same tokenizer + prompt ⇒ same
/// `[seq, dim]`), a scale-robust "same direction / no quality regression" measure.
fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "embedding length mismatch");
    let (mut dot, mut na, mut nb, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
        max_abs = max_abs.max((x - y).abs() as f64);
    }
    (dot / (na.sqrt() * nb.sqrt()), max_abs)
}

/// Prompt battery: English + the load-bearing multilingual (Chinese) default negative prompt — the
/// case most sensitive to embedding drift (mixed-script tokens, the longest real prompt).
fn prompts(cfg: &WanModelConfig) -> Vec<String> {
    vec![
        "a red fox trotting across a snowy meadow at sunrise, cinematic".to_string(),
        "an astronaut riding a horse on the moon, dramatic lighting, 8k".to_string(),
        cfg.sample_neg_prompt.clone(),
    ]
}

/// One tier's TE build + encode, measuring the **isolated TE-encode active peak** (no DiT/VAE — this is
/// exactly the sc-12796 residual stage). `bits = None` ⇒ the dense bf16 baseline.
fn encode_all(
    model_dir: &std::path::Path,
    cfg: &WanModelConfig,
    bits: Option<i32>,
) -> (Vec<Vec<f32>>, usize) {
    let tok = load_tokenizer(model_dir.join("tokenizer.json"), cfg.text_len).expect("tokenizer");
    clear_cache();
    reset_peak_memory();
    let enc = match bits {
        None => {
            let w = Weights::from_file(model_dir.join("t5_encoder.safetensors")).expect("t5");
            Umt5Encoder::from_weights(&w, cfg).expect("dense umt5")
        }
        Some(b) => {
            let mut w = Weights::from_file(model_dir.join("t5_encoder.safetensors")).expect("t5");
            let q = WanQuant {
                bits: b,
                group_size: 64,
            };
            Umt5Encoder::from_weights_quantized(&mut w, cfg, q).expect("quantized umt5")
        }
    };
    let embeds: Vec<Vec<f32>> = prompts(cfg)
        .iter()
        .map(|p| {
            let e = enc.encode(&tok, p).expect("encode");
            mlx_rs::transforms::eval([&e]).unwrap();
            e.as_slice::<f32>().to_vec()
        })
        .collect();
    let peak = get_peak_memory();
    (embeds, peak)
}

#[test]
#[ignore = "needs a converted Wan2.2-TI2V-5B tier (WAN_TI2V_5B_MODEL_DIR); GPU-heavy real-weight parity"]
fn umt5_te_quant_parity_and_active_peak() {
    let model_dir = match env_path("WAN_TI2V_5B_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_TI2V_5B_MODEL_DIR to a converted TI2V-5B tier dir");
            return;
        }
    };
    let cfg = WanModelConfig::wan22_ti2v_5b();

    // Baseline: the bit-exact bf16 dense encoder (the pre-sc-12831 path).
    let (base, base_peak) = encode_all(&model_dir, &cfg, None);
    println!(
        "[bf16 baseline] TE-encode active peak = {:.2} GiB",
        gib(base_peak)
    );

    // Q8 is the **production** TE floor (see `model::effective_te_quant`); Q4 is measured alongside as
    // the **rejected** alternative — its lower cosine is exactly why the TE floors at Q8, not the DiT's
    // Q4. Both are fresh loads (the quantized build consumes `w`). `(bits, min_cosine, production)`:
    // the Q8 floor (0.998) sits ~1.8e-3 below the measured 0.9998 — strict enough to catch any real
    // degradation, loose enough to absorb Metal's ~1e-3 reduced-precision matmul noise (no flakes).
    for (bits, min_cos, production) in [(8i32, 0.9980f64, true), (4i32, 0.9500f64, false)] {
        let (q, peak) = encode_all(&model_dir, &cfg, Some(bits));
        let mut worst_cos = 1.0f64;
        let mut worst_max = 0.0f64;
        for (i, (bp, qp)) in base.iter().zip(&q).enumerate() {
            let (cos, max_abs) = compare(bp, qp);
            println!("[Q{bits} prompt {i}] cosine={cos:.6}  max|Δ|={max_abs:.4e}");
            worst_cos = worst_cos.min(cos);
            worst_max = worst_max.max(max_abs);
        }
        let tag = if production {
            "PRODUCTION"
        } else {
            "rejected (drift too high)"
        };
        println!(
            "[Q{bits} {tag}] TE-encode active peak = {:.2} GiB  (bf16 baseline {:.2})  worst cosine={worst_cos:.6}  worst max|Δ|={worst_max:.4e}",
            gib(peak),
            gib(base_peak)
        );

        // (1) sc-12831 acceptance: the packed TE-encode active peak is far below the ~11.83 GiB bf16 TE
        // (holds for BOTH widths — Q8 7.72, Q4 5.77 — the whole point being that packing retires it).
        assert!(
            gib(peak) < 9.0,
            "Q{bits} TE-encode active peak {:.2} GiB not below 9 GiB — is the TE actually packed?",
            gib(peak)
        );
        assert!(
            (peak as f64) < 0.75 * (base_peak as f64),
            "Q{bits} TE-encode active peak {:.2} GiB is not meaningfully below the bf16 baseline {:.2} GiB",
            gib(peak),
            gib(base_peak)
        );
        // (2) parity-reframe: no quality regression — the embedding keeps ~the same direction. The Q8
        // floor (0.9990) is the "no quality regression" bar; Q4's looser floor (0.9500) just bounds the
        // measured drift that motivates flooring at Q8. Mutation guard: a broken pack (wrong
        // scales/transpose) collapses the cosine well below either floor and this fails.
        assert!(
            worst_cos >= min_cos,
            "Q{bits} embedding cosine {worst_cos:.6} below the {min_cos} floor — quantization degraded \
             the prompt embedding beyond the expected bound",
        );
    }
}
