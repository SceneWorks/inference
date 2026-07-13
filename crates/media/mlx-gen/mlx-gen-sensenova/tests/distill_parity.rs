//! sc-3192: the 8-step distill-LoRA merge math matches the reference `load_and_merge_lora_weight`.
//!
//! Synthetic weight-free golden (the repo pattern): tiny base weights + a tiny LoRA, merged by the
//! genuine reference (`sensenova_u1/utils/lora.py`) in `tools/dump_sensenova_distill_golden.py`.
//! Three targets mirror the real distill-LoRA shapes — a gen-path attention projection, a gen-path
//! SwiGLU projection, and an FM-head Linear — and one uses `alpha != rank` (scale 2.0) so a
//! hardcoded scale-1.0 would fail.
//!
//! Two checks per target:
//! 1. **Delta correctness** — `base + lora_delta(target)` matches the reference merged weight within
//!    the f32 MLX-Metal-vs-torch matmul floor (MLX's Metal f32 matmul is not full precision, ~1e-3
//!    relative; the `(up @ down)` reduction is the only non-exact step, everything else is exact).
//! 2. **Merge seam** — the core [`AdaptableLinear::merge_dense_delta`] seam reproduces `W + δ`
//!    bit-for-bit (same backend), i.e. it is a true weight merge, not a forward-time residual.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test distill_parity -- --nocapture`

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen_sensenova::lora_delta;
use mlx_rs::ops::add;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/distill_golden.safetensors"
);

const TARGETS: &[&str] = &[
    "language_model.model.layers.0.self_attn.q_proj_mot_gen",
    "language_model.model.layers.0.mlp_mot_gen.gate_proj",
    "fm_modules.fm_head.0",
];

/// (peak abs diff, peak-relative `max|Δ|/max|b|`).
fn errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    (max_diff, max_diff / peak)
}

#[test]
fn distill_merge_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");

    for tgt in TARGETS {
        let base = w.require(&format!("__base__.{tgt}")).expect("base").clone();
        let want = w
            .require(&format!("__merged__.{tgt}"))
            .expect("merged")
            .clone();

        // The LoRA carries this target → delta is Some.
        let delta = lora_delta(&w, tgt).expect("delta").expect("target present");

        // (1) Delta correctness vs the reference merge (cross-backend f32 matmul floor).
        let merged = add(&base, &delta).unwrap();
        let (abs, rel) = errors(&merged, &want);
        println!("{tgt}\n    base+delta vs ref: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}");
        assert!(
            rel < 5e-3,
            "{tgt}: merged peak-rel {rel:.3e} exceeds the f32 MLX-Metal matmul floor (5e-3)"
        );

        // (2) The merge seam is a true weight add (same-backend, bit-exact `W + δ`).
        let mut lin = AdaptableLinear::dense(base.clone(), None);
        lin.merge_dense_delta(&delta).unwrap();
        let (seam_w, _) = lin.dense_weight().expect("dense after merge");
        let (seam_abs, _) = errors(seam_w, &merged);
        assert_eq!(
            seam_abs, 0.0,
            "{tgt}: merge_dense_delta must reproduce base+delta bit-for-bit, got |Δ|={seam_abs:.3e}"
        );
    }

    // An absent target yields None (the merge skips it, never errors).
    assert!(
        lora_delta(&w, "language_model.model.layers.0.self_attn.q_proj")
            .unwrap()
            .is_none(),
        "understanding-path target is not in the distill LoRA → None"
    );
}
