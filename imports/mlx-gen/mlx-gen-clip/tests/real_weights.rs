//! Real-weights test (sc-6535): loads the cached `openai/clip-vit-large-patch14` snapshot and checks
//! the CLIP image embedder produces a sane 768-d vector with the right cosine geometry. `#[ignore]`d
//! (multi-GB weights), per the mlx-gen convention. Run with:
//!
//! ```sh
//! CLIP_VIT_L_SNAPSHOT=/path/to/clip-vit-large-patch14 \
//!   cargo test -p mlx-gen-clip --test real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::gen_core::runtime::{LoadSpec, WeightsSource};
use mlx_gen::media::Image;
use mlx_gen_clip::load;

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("CLIP_VIT_L_SNAPSHOT")
            .expect("set CLIP_VIT_L_SNAPSHOT to the openai/clip-vit-large-patch14 snapshot dir"),
    )
}

/// A uniform-colour image. Center-crop→resize to 224² makes it size-invariant, so two solids of the
/// same colour preprocess byte-identically → identical embedding (a clean determinism check).
fn solid(w: u32, h: u32, rgb: [u8; 3]) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h) {
        pixels.extend_from_slice(&rgb);
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "loads openai/clip-vit-large-patch14 (~1.7GB); set CLIP_VIT_L_SNAPSHOT"]
fn embeds_real_images_with_sane_cosine_geometry() {
    let embedder =
        load(&LoadSpec::new(WeightsSource::Dir(snapshot()))).expect("load clip embedder");
    assert_eq!(embedder.descriptor().embedding_dim, 768);

    let red = embedder.embed(&solid(64, 64, [220, 30, 30])).unwrap();
    let red_big = embedder.embed(&solid(96, 96, [220, 30, 30])).unwrap();
    let blue = embedder.embed(&solid(64, 64, [30, 30, 220])).unwrap();

    // Right dimensionality + a non-degenerate vector.
    assert_eq!(red.len(), 768, "CLIP ViT-L/14 embedding is 768-d");
    assert!(red.iter().any(|&x| x != 0.0), "embedding is not all-zero");

    // Determinism / size-invariance: the same colour at two sizes → identical embedding.
    let self_cos = cosine(&red, &red_big);
    assert!(
        self_cos > 0.999,
        "same colour at two sizes should match (cos={self_cos})"
    );

    // Colour sensitivity: a different colour is measurably less similar than an identical image.
    let cross_cos = cosine(&red, &blue);
    assert!(
        cross_cos < self_cos,
        "red·blue ({cross_cos}) should be < red·red ({self_cos})"
    );
    println!(
        "clip ok: dim={}, red·red={self_cos:.5}, red·blue={cross_cos:.5}",
        red.len()
    );
}
