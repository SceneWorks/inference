//! ArcFace (iresnet100 / glintr100) numerical parity vs the onnx reference (sc-3081).
//!
//! The fidelity gate of the whole face stack: PuLID/InstantID were trained on the antelopev2
//! `glintr100` embeddings, so the MLX port must reproduce them (cosine ≈ 1.0). Goldens are produced
//! by `tools/convert_glintr100.py` (converted weights + deterministic inputs + onnx embeddings) and
//! live under `tools/golden/` (gitignored, local-only) — hence `#[ignore]`.
//!
//! Run:
//!   ~/.dwpose-spike/venv/bin/python tools/convert_glintr100.py   # once, to produce the goldens
//!   cargo test -p mlx-gen-face --release --test arcface_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_face::ArcFace;

fn golden(name: &str) -> Weights {
    let path = format!("{}/../tools/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    Weights::from_file(&path).unwrap_or_else(|e| {
        panic!("missing golden {path}: {e}\nRun tools/convert_glintr100.py first.")
    })
}

#[test]
#[ignore = "needs local goldens from tools/convert_glintr100.py"]
fn arcface_cosine_parity() {
    let w = golden("arcface_iresnet100.safetensors");
    let g = golden("arcface_goldens.safetensors");
    let inputs = g.require("inputs").unwrap(); // [K, 112, 112, 3] f32, normalized
    let want = g.require("embeddings").unwrap(); // [K, 512] f32, raw onnx output

    let net = ArcFace::from_weights(&w).unwrap();
    let got = net.forward(inputs).unwrap();

    let k = want.shape()[0] as usize;
    let dim = want.shape()[1] as usize;
    assert_eq!(
        got.shape(),
        &[k as i32, dim as i32],
        "embedding shape mismatch"
    );

    let got_v = got.try_as_slice::<f32>().unwrap();
    let want_v = want.try_as_slice::<f32>().unwrap();

    let mut min_cos = f32::INFINITY;
    let mut max_abs = 0.0f32;
    for i in 0..k {
        let a = &got_v[i * dim..(i + 1) * dim];
        let b = &want_v[i * dim..(i + 1) * dim];
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for j in 0..dim {
            dot += a[j] as f64 * b[j] as f64;
            na += (a[j] as f64).powi(2);
            nb += (b[j] as f64).powi(2);
            max_abs = max_abs.max((a[j] - b[j]).abs());
        }
        let cos = (dot / (na.sqrt() * nb.sqrt())) as f32;
        println!("face {i}: cosine = {cos:.8}, ||onnx|| = {:.4}", nb.sqrt());
        min_cos = min_cos.min(cos);
    }
    println!("min cosine = {min_cos:.8}, max abs diff = {max_abs:.6}");
    assert!(
        min_cos >= 0.9999,
        "ArcFace embedding cosine {min_cos:.8} < 0.9999 vs glintr100 onnx (max abs {max_abs:.6})"
    );
}
