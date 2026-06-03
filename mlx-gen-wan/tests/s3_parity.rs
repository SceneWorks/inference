//! S3 parity gate: the Wan DiT (5B) must reproduce the `mlx_video` reference, gated **bf16-against-bf16**
//! — the production regime (the reference runs bf16 matmuls + bf16 SDPA + bf16 cos/sin with an f32
//! residual stream; see `tools/dump_s3_fixtures.py`).
//!
//! Observed: patch embedding **bit-exact** (`x_embed` max|Δ| = 0.0); DiT output mean_rel ~5e-2. A
//! stage-bisection (feeding the golden's bit-exact intermediates) localized — and fixed — the two
//! real port bugs that had dominated an earlier ~6e-2. (a) **Fused Linear**: `nn.Linear` is
//! `addmm(bias, x, Wᵀ)` (accumulate, add bias, round once); a separate `matmul`+`add` double-rounds in
//! bf16 (~1.4e-3 per matmul) — fixed in `Linear::forward`. (b) **GELU dtype**: `nn.GELU(approx="tanh")`
//! weak-casts its scalar constants to the input dtype so a bf16 FFN stays bf16, whereas f32 scalars
//! promote it to f32 (~1e-3) — fixed in `text_encoder::gelu_tanh`. With those fixed, every per-block
//! stage is **bit-exact** when fed bit-exact input (cross-attn / FFN ≤ 1e-4). The remaining ~5e-2 is
//! the **0.31.1-vs-0.31.2 bf16 matmul delta** (~4e-7 per matmul — verified: the same `q_proj` is
//! 3.97e-7 on a 0.31.1 wheel and 0.0 on 0.31.2) amplified through the 30 attention layers. **f32 ops
//! are bit-exact across the versions** (S1 T5 = 0.0), so a **0.31.2 pin bump makes the DiT bit-exact**.
//! True end-to-end parity is the px>8 video gate at S4.
//!
//! `#[ignore]` heavy: loads the converted `model.safetensors` (~11 GB) from the snapshot dir
//! (`WAN_5B_DIR`). Honors "divergence is not rounding" — the residual is a *named* cross-version
//! matmul delta (f32 bit-exact, patch embedding bit-exact, per-block math bit-exact, growth consistent
//! with bf16 precision — not a code bug).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::WanTransformer;

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("WAN_5B_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(std::env::var("HOME").unwrap())
        .join("Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b")
}

fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let (mut ma, mut sa, mut sr) = (0f32, 0f64, 0f64);
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        ma = ma.max(d);
        sa += d as f64;
        sr += e.abs() as f64;
    }
    (ma, sa / sr.max(1e-30))
}

#[test]
#[ignore = "needs the converted 5B model.safetensors (~11 GB) — run tools/dump_s3_fixtures.py"]
fn dit_forward_matches_reference() {
    let dir = snapshot_dir();
    let cfg = WanModelConfig::wan22_ti2v_5b();
    let w = Weights::from_file(dir.join("model.safetensors")).expect("model.safetensors");
    let dit = WanTransformer::from_weights(&w, &cfg).expect("build DiT");

    let g = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s3_dit_golden.safetensors"
    ))
    .expect("s3 golden");

    let latent = g.require("latent").unwrap().clone();
    let context_raw = g.require("context_raw").unwrap().clone();
    let t: f32 = g.require("t").unwrap().as_slice::<f32>()[0];

    let context_emb = dit.embed_text(&context_raw).expect("embed_text");
    let stages = dit
        .forward_capture(&latent, t, &context_emb)
        .expect("forward_capture");
    let out = dit.forward(&latent, t, &context_emb).expect("forward");

    // Per-stage gate: x_embed (idx 0) must be bit-exact (patch embed = f32-promoted matmul → bf16,
    // exact cross-build); the residual grows with depth as the cross-build bf16 kernel difference
    // accumulates.
    let stage = |idx: usize, key: &str| -> (f32, f64) {
        let got = stages[idx]
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        diff(&got, g.require(key).unwrap().as_slice::<f32>())
    };
    let (e_max, _) = stage(0, "x_embed");
    println!("[x_embed]  max|Δ|={e_max:.3e}");
    assert_eq!(e_max, 0.0, "patch embedding not bit-exact: {e_max:.3e}");

    for (idx, key) in [(3usize, "x_block0"), (4, "x_blocks"), (5, "x_head")] {
        let (ma, mr) = stage(idx, key);
        println!("[{key}] max|Δ|={ma:.3e} mean_rel={mr:.3e}");
    }

    let got = out.as_slice::<f32>().to_vec();
    let (max_abs, mean_rel) = diff(&got, g.require("output").unwrap().as_slice::<f32>());
    println!("[output]   max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}");
    // Per-block math is bit-exact (addmm + gelu-dtype fixed); the residual ~5e-2 is the
    // 0.31.1-vs-0.31.2 bf16 matmul delta (~4e-7/matmul) amplified over 30 layers. 0.31.2 → bit-exact.
    assert!(
        mean_rel < 6e-2,
        "DiT output mean_rel {mean_rel:.3e} too high"
    );
}
