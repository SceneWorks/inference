//! sc-12914 — first-class real-weight UMT5 Q8 measurement for the optimized Wan2.2 A14B and
//! VACE-Fun A14B paths. This is the 14B analogue of `te_quant_parity.rs`, deliberately loading only
//! each snapshot's shared text encoder: the dual experts are not the subject of this gate and would
//! obscure the TE residency measurement.
//!
//! The test requires both snapshots so a green run is evidence for both acceptance paths:
//!
//! ```text
//! WAN_A14B_MODEL_DIR=/path/to/wan2.2-t2v-a14b/q4 \
//! WANVACE_FUN_DIR=/path/to/assembled-vace-fun-a14b \
//!   cargo test -p mlx-gen-wan --release --test a14b_vace_te_quant_measurement \
//!     -- --ignored --nocapture --test-threads=1
//! ```

use std::path::{Path, PathBuf};

use mlx_gen::{weights::Weights, Quant};
use mlx_gen_wan::{
    encode_text_staged_for_tier, load_tokenizer, Umt5Encoder, WanModelConfig, WanVaceConfig,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

const PROMPT: &str = "a red fox trotting across a snowy meadow at sunrise, cinematic";

fn required_dir(var: &str) -> PathBuf {
    let path = std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {var} to the real converted snapshot directory"));
    assert!(
        path.join("t5_encoder.safetensors").is_file(),
        "{var} has no t5_encoder.safetensors"
    );
    assert!(
        path.join("tokenizer.json").is_file(),
        "{var} has no tokenizer.json"
    );
    path
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / 2_f64.powi(30)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "embedding length mismatch");
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    dot / (aa.sqrt() * bb.sqrt())
}

fn encode(root: &Path, cfg: &WanModelConfig, quantized: bool) -> (Vec<f32>, usize) {
    clear_cache();
    reset_peak_memory();
    if quantized {
        let (embedding, _) = encode_text_staged_for_tier(
            root,
            cfg,
            PROMPT,
            &cfg.sample_neg_prompt,
            true,
            Some(Quant::Q4),
        )
        .expect("production quantized-tier UMT5 stage");
        let peak = get_peak_memory();
        return (embedding.as_slice::<f32>().to_vec(), peak);
    }
    let tokenizer = load_tokenizer(root.join("tokenizer.json"), cfg.text_len).expect("tokenizer");
    let encoder = {
        let weights =
            Weights::from_file(root.join("t5_encoder.safetensors")).expect("UMT5 weights");
        Umt5Encoder::from_weights(&weights, cfg).expect("bf16 UMT5")
    };
    let embedding = encoder.encode(&tokenizer, PROMPT).expect("encode prompt");
    mlx_rs::transforms::eval([&embedding]).expect("materialize embedding");
    (embedding.as_slice::<f32>().to_vec(), get_peak_memory())
}

fn measure(label: &str, root: &Path, cfg: &WanModelConfig) {
    let (bf16, bf16_peak) = encode(root, cfg, false);
    let (q8, q8_peak) = encode(root, cfg, true);
    let cos = cosine(&bf16, &q8);
    eprintln!(
        "[{label}] bf16 {:.2} GiB, Q8 {:.2} GiB, cosine {cos:.6}",
        gib(bf16_peak),
        gib(q8_peak)
    );

    assert!(
        cos >= 0.998,
        "{label}: Q8 embedding cosine {cos:.6} fell below 0.998"
    );
    assert!(
        (q8_peak as f64) < bf16_peak as f64 * 0.75,
        "{label}: Q8 peak {:.2} GiB is not at least 25% below bf16 {:.2} GiB; the TE may have been left dense",
        gib(q8_peak),
        gib(bf16_peak)
    );
    assert!(
        gib(q8_peak) < 9.0,
        "{label}: Q8 peak {:.2} GiB exceeds the discriminating 9 GiB packed-TE ceiling",
        gib(q8_peak)
    );
}

#[test]
#[ignore = "needs real converted A14B + assembled VACE-Fun A14B snapshots and Apple Silicon MLX"]
fn a14b_and_vace_fun_q8_te_are_small_and_near_lossless() {
    let a14b = required_dir("WAN_A14B_MODEL_DIR");
    let a14b_cfg = WanModelConfig::from_model_dir(&a14b).expect("A14B config");
    assert!(
        a14b_cfg.dual_model,
        "WAN_A14B_MODEL_DIR must be an A14B dual-expert snapshot"
    );
    measure("Wan2.2 A14B", &a14b, &a14b_cfg);

    let vace = required_dir("WANVACE_FUN_DIR");
    let vace_cfg = WanVaceConfig::vace_fun_from_model_dir(&vace).expect("VACE-Fun config");
    assert!(
        vace_cfg.base.dual_model,
        "WANVACE_FUN_DIR must be a VACE-Fun A14B snapshot"
    );
    measure("Wan2.2 VACE-Fun A14B", &vace, &vace_cfg.base);
}
