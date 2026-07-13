//! sc-3192 diagnostic: does the MLX bf16 merge retain the (tiny) distill delta the same way the
//! torch reference does? The distill delta is ~0.2% of the weight magnitude — below the bf16 ULP
//! for large weights — so this verifies the port doesn't silently flush it. Compares the port's
//! merged bf16 weight to the torch reference merged weight on real tensors.
//!
//! `#[ignore]` — needs `tools/`-dumped `distill_realweight_merge.safetensors` (gitignored).
//! Run: cargo test -p mlx-gen-sensenova --test distill_merge_realweight -- --ignored --nocapture

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen_sensenova::lora_delta;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/distill_realweight_merge.safetensors"
);

const TARGETS: &[&str] = &[
    "language_model.model.layers.0.self_attn.q_proj_mot_gen",
    "language_model.model.layers.0.mlp_mot_gen.gate_proj",
    "fm_modules.fm_head.0",
];

fn f32v(a: &Array) -> Vec<f32> {
    let n = a.shape().iter().product::<i32>();
    a.as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .reshape(&[n])
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

#[test]
#[ignore = "needs the dumped real-weight merge fixture; run with --ignored"]
fn mlx_merge_retains_delta_like_torch() {
    let fix = std::path::Path::new(FIXTURE);
    if !fix.exists() {
        eprintln!("skipping: {} missing", fix.display());
        return;
    }
    let w = Weights::from_file(FIXTURE).expect("load fixture");

    for tgt in TARGETS {
        let base = w.require(&format!("base.{tgt}")).unwrap().clone();
        let want = w.require(&format!("merged.{tgt}")).unwrap().clone();

        // Port merge: lora_delta + the core bf16 merge seam (exactly what `merge_distill_lora` does).
        let delta = lora_delta(&w, tgt).unwrap().unwrap();
        let mut lin = AdaptableLinear::dense(base.clone(), None);
        lin.merge_dense_delta(&delta).unwrap();
        let (got_w, _) = lin.dense_weight().unwrap();

        let (b, g, want_v) = (f32v(&base), f32v(got_w), f32v(&want));
        // Delta retained by each side, and how close the port's merged weight is to torch's.
        let ref_delta: f32 = want_v.iter().zip(&b).map(|(&m, &x)| (m - x).abs()).sum();
        let port_delta: f32 = g.iter().zip(&b).map(|(&m, &x)| (m - x).abs()).sum();
        let max_abs = g
            .iter()
            .zip(&want_v)
            .fold(0f32, |m, (&a, &c)| m.max((a - c).abs()));
        let peak = want_v.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-9);
        println!(
            "{tgt}\n    port-merged vs torch-merged: peak-rel={:.3e}  |  Σ|delta| port={port_delta:.2} ref={ref_delta:.2} (ratio {:.4})",
            max_abs / peak,
            port_delta / ref_delta
        );
        // The port must retain essentially the same delta as torch (bf16 rounding is shared); a
        // silent flush would show ratio ≪ 1. Weight-level agreement is the bf16-merge floor.
        assert!(
            (port_delta / ref_delta - 1.0).abs() < 0.05,
            "{tgt}: port retains {:.3}× the reference delta — merge is dropping the LoRA",
            port_delta / ref_delta
        );
        assert!(
            max_abs / peak < 5e-3,
            "{tgt}: port-merged vs torch-merged peak-rel {:.3e} too large",
            max_abs / peak
        );
    }
}
