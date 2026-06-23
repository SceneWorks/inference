//! The `ImageEmbedder` contract: a single global image embedding (CLIP-style) per image, for the
//! Dataset Doctor analysis job (epic 6529 P2, sc-6535).
//!
//! Backend-neutral like every other gen-core contract — host types only (`Vec<f32>`, [`Image`]), no
//! `mlx_rs::Array` / candle `Tensor`. MLX implements it in `mlx-gen-clip`, candle in
//! `candle-gen-clip`. Both feed the same consumer: the worker's `dataset_analysis` job, which embeds
//! every dataset item and derives set-level findings (near-duplicate clustering, diversity, caption
//! alignment, aesthetic) from the vectors.
//!
//! Unlike [`FaceEmbedder`](crate::face::FaceEmbedder) — which *detects* faces and embeds each one —
//! this embeds the whole image into one vector in a single fixed space (e.g. CLIP ViT-L/14, 768-d).
//! Like the face embedding, the returned vector is **raw** (un-normalized); callers L2-normalize for
//! cosine similarity.

use crate::media::Image;
use crate::Result;

/// A whole-image embedding provider (a CLIP-style vision encoder).
pub trait ImageEmbedder: Send + Sync {
    /// Stable identity + advertised shape, constructible without loading weights.
    fn descriptor(&self) -> &ImageEmbedderDescriptor;

    /// Embed one image into its raw (un-normalized) vector of length
    /// [`ImageEmbedderDescriptor::embedding_dim`]. Callers L2-normalize for cosine similarity.
    fn embed(&self, image: &Image) -> Result<Vec<f32>>;

    /// Embed a batch of images. The default maps [`embed`](Self::embed) over the slice; a provider
    /// can override with a single batched forward for throughput (CLIP batches well, and a dataset
    /// is N images at once). Order matches the input.
    fn embed_batch(&self, images: &[Image]) -> Result<Vec<Vec<f32>>> {
        images.iter().map(|image| self.embed(image)).collect()
    }
}

/// An image embedder's stable identity + advertised shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageEmbedderDescriptor {
    /// Stable id (e.g. `"clip_vit_l14"`).
    pub id: &'static str,
    /// Provider family (`"image-embed"`).
    pub family: &'static str,
    /// Tensor backend that registered this embedder (`"mlx"` | `"candle"`); used by the worker's
    /// per-backend capability advertisement.
    pub backend: &'static str,
    /// Dimensionality of the returned embedding (768 for CLIP ViT-L/14).
    pub embedding_dim: usize,
    /// The embedding-space identifier (e.g. `"clip-vit-l14"`). Two vectors are only comparable when
    /// their `space` matches — it guards the dataset-analysis cache + cosine math against silently
    /// mixing vectors from different encoders (a future EVA-CLIP/SigLIP swap).
    pub space: &'static str,
    /// Whether this embedder only runs on macOS (the MLX implementation); candle sets this `false`.
    pub mac_only: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstEmbedder {
        descriptor: ImageEmbedderDescriptor,
        value: Vec<f32>,
    }

    impl ImageEmbedder for ConstEmbedder {
        fn descriptor(&self) -> &ImageEmbedderDescriptor {
            &self.descriptor
        }
        fn embed(&self, _image: &Image) -> Result<Vec<f32>> {
            Ok(self.value.clone())
        }
    }

    fn image() -> Image {
        Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        }
    }

    #[test]
    fn embed_returns_the_raw_vector() {
        let embedder = ConstEmbedder {
            descriptor: ImageEmbedderDescriptor {
                id: "test",
                family: "image-embed",
                backend: "mlx",
                embedding_dim: 3,
                space: "test-space",
                mac_only: true,
            },
            value: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(embedder.descriptor().embedding_dim, 3);
        assert_eq!(embedder.embed(&image()).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn default_embed_batch_maps_over_embed_preserving_order() {
        let embedder = ConstEmbedder {
            descriptor: ImageEmbedderDescriptor {
                id: "test",
                family: "image-embed",
                backend: "mlx",
                embedding_dim: 2,
                space: "test-space",
                mac_only: true,
            },
            value: vec![0.5, 0.5],
        };
        let batch = embedder.embed_batch(&[image(), image()]).unwrap();
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|v| v == &vec![0.5, 0.5]));
    }
}
