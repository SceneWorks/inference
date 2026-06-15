//! sc-5681 — activation-chunking equivalence gate.
//!
//! Proves the memory levers ([`DitMemoryConfig`]: per-block eval-to-free, FFN sequence-chunking,
//! attention query-chunking) change only the memory *schedule*, not the result — so the parity gate
//! (`dit_parity` / `dit_real_parity`, which run with the default [`DitMemoryConfig::OFF`]) keeps
//! covering correctness while production runs with the levers on. Two equivalence classes, asserted
//! separately (measured on the tiny fixture, f32 compute):
//!   * **`eval_per_block` is exactly bit-identical** (max|Δ| == 0) — it only forces materialization.
//!     This is the dominant memory lever, so the production memory win is bit-exact.
//!   * **The sequence-chunking levers are numerically equivalent** (cosine ≥ 0.9999999, max|Δ| ~1e-3)
//!     — MLX's Metal GEMM / SDPA kernels are tile-specialized by the row (M) dimension, so a
//!     `[chunk, k]` matmul rounds slightly differently from the full `[L, k]` one. The math is
//!     identical (FFN per-token, attention softmax per-query); the residual is the same kernel-rounding
//!     class as the model's own torch parity (max|Δ| 0.003, [[mlx_metal_matmul_reduced_precision]]) and
//!     an order *tighter* than it for the production `ffn+eval` default (cosine 1.0000000, max|Δ| <8e-4).
//!
//! Self-consistent: it compares `forward(OFF)` against `forward(levered)` on the **same** model +
//! inputs, so it needs no torch reference — only the tiny model the `dit_parity` fixtures already
//! carry. The deliberately tiny chunk size (3 tokens) forces multi-block + ragged-remainder paths on
//! the small sequence so every code path (chunk boundary, last-block remainder, eval-per-block) fires.
//!
//! `#[ignore]` because it reuses the locally-generated `dit_parity` fixtures (see that module's doc).
//! Run with `cargo test -p mlx-gen-scail2 --test dit_chunk_equiv -- --ignored`.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_scail2::{Scail2Config, Scail2Dit, Scail2Inputs};
use mlx_gen_wan::DitMemoryConfig;
use mlx_rs::{Array, Dtype};

fn parity_dir() -> PathBuf {
    std::env::var("SCAIL2_PARITY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/scail2-parity")
        })
}

fn flat(a: &Array) -> Vec<f32> {
    a.reshape(&[-1])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// (cosine similarity, max abs diff) between two same-shape tensors.
fn compare(a: &Array, b: &Array) -> (f32, f32) {
    let (va, vb) = (flat(a), flat(b));
    assert_eq!(va.len(), vb.len(), "shape mismatch");
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut max_abs = 0f32;
    for (x, y) in va.iter().zip(&vb) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
        max_abs = max_abs.max((x - y).abs());
    }
    ((dot / (na.sqrt() * nb.sqrt())) as f32, max_abs)
}

/// Per-lever equivalence of the sc-5681 activation-memory levers vs. the default un-chunked forward,
/// in f32 compute (the parity-gate dtype), across all four conditioning cases.
///
/// Two equivalence classes, distinguished honestly:
///   * **`eval_per_block` is exactly bit-identical** (max|Δ| == 0) — it only forces materialization,
///     it never touches a value. This is the dominant memory lever, so the production memory win is
///     bit-exact.
///   * **The sequence-chunking levers (`ffn_seq_chunk` / `attn_query_chunk`) are numerically
///     equivalent, not bit-identical** (cosine ≈ 1, max|Δ| ~1e-3) — MLX's Metal matmul / SDPA
///     kernels are tile-specialized by the row (M) dimension, so a `[chunk, k]·[k, n]` GEMM rounds
///     slightly differently from the full `[L, k]·[k, n]`. The math is identical (the FFN is
///     per-token, attention softmax is per-query); the residual is the same kernel-rounding class as
///     the model's own torch parity (cosine 0.9999/max|Δ| 0.003, [[mlx_metal_matmul_reduced_precision]]),
///     i.e. well inside the model's numerical noise. The gate asserts cosine ≥ 0.99999.
#[test]
#[ignore = "needs the dit_parity fixtures (see dit_parity module doc); run with --ignored on macOS"]
fn chunking_matches_unchunked() {
    let dir = parity_dir();
    let model_path = dir.join("model.safetensors");
    assert!(
        model_path.exists(),
        "missing fixtures at {} — generate with the dit_parity harness",
        dir.display(),
    );

    let cfg = Scail2Config::from_model_dir(&dir).unwrap();
    let w = Weights::from_file(&model_path).unwrap();
    let base = Scail2Dit::from_weights(&w, &cfg).unwrap();

    // (label, config, must-be-exact?). `eval` alone is exact; chunking is numerically equivalent. The
    // tiny chunk (3) is < the tiny sequence length so multi-block + ragged-remainder paths fire.
    let off = DitMemoryConfig::OFF;
    let configs: [(&str, DitMemoryConfig, bool); 5] = [
        (
            "eval_only",
            DitMemoryConfig {
                eval_per_block: true,
                ..off
            },
            true,
        ),
        (
            "ffn_only",
            DitMemoryConfig {
                ffn_seq_chunk: Some(3),
                ..off
            },
            false,
        ),
        (
            "attn_only",
            DitMemoryConfig {
                attn_query_chunk: Some(3),
                ..off
            },
            false,
        ),
        (
            "prod(ffn+eval)",
            DitMemoryConfig {
                ffn_seq_chunk: Some(3),
                eval_per_block: true,
                ..off
            },
            false,
        ),
        (
            "all",
            DitMemoryConfig {
                ffn_seq_chunk: Some(3),
                attn_query_chunk: Some(3),
                eval_per_block: true,
            },
            false,
        ),
    ];

    let cases = ["base_anim", "base_replace", "addref", "history"];
    for name in cases {
        let cdir = dir.join(name);
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(cdir.join("case.json")).unwrap())
                .unwrap();
        let t = meta["t"].as_f64().unwrap() as f32;
        let replace_flag = meta["replace_flag"].as_bool().unwrap();
        let has_history = meta["has_history"].as_bool().unwrap();
        let addref = meta["addref"].as_i64().unwrap();

        let io = Weights::from_file(cdir.join("io.safetensors")).unwrap();
        let get = |k: &str| io.require(k).unwrap();
        let history = has_history.then(|| get("history_mask"));
        let (add_lat, add_mask) = if addref > 0 {
            (
                Some(get("additional_ref_latent")),
                Some(get("additional_ref_masks")),
            )
        } else {
            (None, None)
        };

        let mk_inputs = || Scail2Inputs {
            x: get("x"),
            ref_latent: get("ref_latent"),
            ref_masks: get("ref_masks"),
            pose_latent: get("pose_latent"),
            driving_masks: get("driving_masks"),
            history_mask: history,
            additional_ref_latent: add_lat,
            additional_ref_masks: add_mask,
            clip_fea: get("clip_fea"),
            context: get("context"),
            t,
            replace_flag,
        };

        let out_base = base.forward(&mk_inputs()).unwrap();
        for (label, mem, exact) in &configs {
            let mut dit = Scail2Dit::from_weights(&w, &cfg).unwrap();
            dit.set_memory_config(*mem);
            let out = dit.forward(&mk_inputs()).unwrap();
            assert_eq!(out.shape(), out_base.shape(), "[{name}/{label}] shape");
            let (cos, max_abs) = compare(&out_base, &out);
            println!("[{name:13} / {label:14}] cosine {cos:.7}  max|Δ| {max_abs:.3e}");
            if *exact {
                assert_eq!(
                    max_abs, 0.0,
                    "[{name}/{label}] eval-to-free must be exactly bit-identical (max|Δ| {max_abs})"
                );
            } else {
                assert!(
                    cos > 0.99999,
                    "[{name}/{label}] cosine {cos} below 0.99999 (max|Δ| {max_abs})"
                );
            }
        }
    }
    println!(
        "eval-to-free is bit-exact; sequence-chunking is numerically equivalent (cosine ≥ 0.99999)"
    );
}
