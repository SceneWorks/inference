//! ArcFace identity cosine between two images (epic 10871 P4.2 identity-preservation scoring).
//! Detects the largest face in each image (SCRFD), aligns + embeds it (glintr100/iresnet100),
//! L2-normalizes, and prints the cosine similarity — the same detect→align→embed stack the
//! InstantID/likeness path uses. Point `FACE_SCRFD`/`FACE_ARCFACE` at the required detector +
//! embedder safetensors (inference never self-fetches or derives a cache location, epic 13657).
//!
//! Run: `cargo run --release --example face_cosine -p mlx-gen-face -- <imgA> <imgB>`

use mlx_gen::weights::Weights;
use mlx_gen_face::FaceAnalysis;

fn load_rgb(path: &str) -> (Vec<u8>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"))
        .to_rgb8();
    let (w, h) = img.dimensions();
    (img.into_raw(), h as usize, w as usize)
}

fn l2(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a_path = args
        .get(1)
        .expect("usage: face_cosine <imgA> <imgB>")
        .clone();
    let b_path = args
        .get(2)
        .expect("usage: face_cosine <imgA> <imgB>")
        .clone();
    let scrfd_path = std::env::var("FACE_SCRFD")
        .expect("set FACE_SCRFD to the scrfd_10g.safetensors path (epic 13657)");
    let arcface_path = std::env::var("FACE_ARCFACE")
        .expect("set FACE_ARCFACE to the arcface_iresnet100.safetensors path (epic 13657)");

    let scrfd_w = Weights::from_file(&scrfd_path).expect("scrfd weights");
    let arcface_w = Weights::from_file(&arcface_path).expect("arcface weights");
    let fa = FaceAnalysis::load(&scrfd_w, &arcface_w).expect("face analysis stack");

    let (a, ah, aw) = load_rgb(&a_path);
    let (b, bh, bw) = load_rgb(&b_path);
    let faces_a = fa.analyze(&a, ah, aw).expect("analyze A");
    let faces_b = fa.analyze(&b, bh, bw).expect("analyze B");
    eprintln!(
        "[cosine] A={a_path}: {} face(s); B={b_path}: {} face(s)",
        faces_a.len(),
        faces_b.len()
    );
    if faces_a.is_empty() || faces_b.is_empty() {
        eprintln!("[cosine] NO FACE detected in one image — cannot score identity");
        println!(
            "cosine NA (missing face: A={} B={})",
            faces_a.len(),
            faces_b.len()
        );
        return;
    }
    let fa0 = &faces_a[0];
    let fb0 = &faces_b[0];
    eprintln!(
        "[cosine] A top face det={:.3} bbox={:?}; B top face det={:.3} bbox={:?}",
        fa0.det_score, fa0.bbox, fb0.det_score, fb0.bbox
    );
    let ea = l2(&fa0.embedding);
    let eb = l2(&fb0.embedding);
    let cos: f32 = ea.iter().zip(&eb).map(|(x, y)| x * y).sum();
    println!("cosine {cos:.4}");
}
