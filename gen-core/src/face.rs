//! The `FaceEmbedder` contract: face detection + identity embedding for the identity-preserving
//! image pipelines (InstantID, PuLID-FLUX) and the worker's keypoint-extract surface (epic 5482).
//!
//! Backend-neutral, like every other gen-core contract: a SCRFD-style five-point detector plus an
//! ArcFace-style recognition embedder (the insightface `antelopev2` family), expressed purely in
//! host types — no `mlx_rs::Array`, no candle `Tensor`. MLX implements it in `mlx-gen-face`; candle
//! implements it in `candle-gen-face` (epic 5480, sc-5490). Both feed the same downstream consumers:
//!
//! * **InstantID** (sc-5491) — the raw 512-d embedding drives the IP-Adapter resampler, the five
//!   landmarks render the IdentityNet pose-control image.
//! * **PuLID-FLUX** (sc-5492) — the raw 512-d embedding is the `id_ante_embedding` half of IDFormer.
//! * **kps_extract** (epic 5482) — the five landmarks are the worker's keypoint surface.

use crate::media::Image;
use crate::Result;

/// One detected face. Coordinates are in original-image pixels; `embedding` is the **raw**
/// (un-normalized) ArcFace recognition vector — callers L2-normalize for cosine similarity, and
/// the identity pipelines feed it raw. `embedding` is empty when produced by a detect-only call
/// ([`FaceEmbedder::detect`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DetectedFace {
    /// `[x1, y1, x2, y2]` bounding box in original-image pixels.
    pub bbox: [f32; 4],
    /// The five SCRFD landmarks — left-eye, right-eye, nose, left-mouth, right-mouth — in
    /// original-image pixels. These are the alignment anchors and the IdentityNet pose anchors.
    pub kps: [[f32; 2]; 5],
    /// SCRFD detection confidence (sigmoid'd).
    pub det_score: f32,
    /// Raw 512-d ArcFace embedding (un-normalized). Empty for detect-only results.
    pub embedding: Vec<f32>,
}

/// A face detection + identity-embedding provider (SCRFD detector + ArcFace recognizer).
///
/// The three entry points mirror insightface's `FaceAnalysis`: a cheap detect-only sweep, a full
/// detect-and-embed sweep, and a single-largest-face convenience. All face-ordered results are
/// **largest-first** (descending bounding-box area), the insightface convention the identity
/// pipelines rely on when they take "the" face.
pub trait FaceEmbedder: Send + Sync {
    /// Stable identity + capability metadata, constructible without loading weights.
    fn descriptor(&self) -> &FaceEmbedderDescriptor;

    /// Detect every face, largest-first. Embeddings are **not** computed — each returned
    /// [`DetectedFace::embedding`] is empty. This is the cheap path for callers that need only the
    /// bounding boxes / landmarks (e.g. pose-keypoint extraction).
    fn detect(&self, image: &Image) -> Result<Vec<DetectedFace>>;

    /// Detect every face and compute each one's raw ArcFace embedding, largest-first.
    fn analyze(&self, image: &Image) -> Result<Vec<DetectedFace>>;

    /// Detect and embed only the largest face. Returns [`Error::Msg`](crate::Error::Msg) when the
    /// image contains no detectable face. The default runs [`analyze`](Self::analyze) and takes the
    /// first (largest) result; a provider can override it to embed only the largest detection.
    fn largest_face(&self, image: &Image) -> Result<DetectedFace> {
        self.analyze(image)?
            .into_iter()
            .next()
            .ok_or_else(|| crate::Error::Msg(format!("{}: no face detected", self.descriptor().id)))
    }
}

/// A face embedder's stable identity + advertised shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceEmbedderDescriptor {
    /// Stable id (e.g. `"antelopev2"`).
    pub id: &'static str,
    /// Provider family (`"face"`).
    pub family: &'static str,
    /// Tensor backend that registered this embedder (`"mlx"` | `"candle"`); used by the worker's
    /// per-backend capability advertisement (sc-4906, epic 3720).
    pub backend: &'static str,
    /// Dimensionality of [`DetectedFace::embedding`] (512 for ArcFace `glintr100`).
    pub embedding_dim: usize,
    /// Whether this embedder only runs on macOS (the MLX implementation); the candle implementation
    /// sets this `false`.
    pub mac_only: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal in-memory embedder: every "face" is a unit square whose embedding encodes its
    /// area, so the largest-first ordering and the default `largest_face` are exercised without a
    /// tensor backend.
    struct StubEmbedder {
        descriptor: FaceEmbedderDescriptor,
        faces: Vec<DetectedFace>,
    }

    impl FaceEmbedder for StubEmbedder {
        fn descriptor(&self) -> &FaceEmbedderDescriptor {
            &self.descriptor
        }
        fn detect(&self, _image: &Image) -> Result<Vec<DetectedFace>> {
            Ok(self
                .faces
                .iter()
                .map(|f| DetectedFace {
                    embedding: Vec::new(),
                    ..f.clone()
                })
                .collect())
        }
        fn analyze(&self, _image: &Image) -> Result<Vec<DetectedFace>> {
            Ok(self.faces.clone())
        }
    }

    fn descriptor() -> FaceEmbedderDescriptor {
        FaceEmbedderDescriptor {
            id: "stub",
            family: "face",
            backend: "candle",
            embedding_dim: 512,
            mac_only: false,
        }
    }

    fn face(area: f32) -> DetectedFace {
        let side = area.sqrt();
        DetectedFace {
            bbox: [0.0, 0.0, side, side],
            kps: [[1.0, 2.0]; 5],
            det_score: 0.9,
            embedding: vec![area; 512],
        }
    }

    fn image() -> Image {
        Image {
            width: 8,
            height: 8,
            pixels: vec![0u8; 8 * 8 * 3],
        }
    }

    #[test]
    fn detect_drops_embeddings_but_keeps_geometry() {
        let e = StubEmbedder {
            descriptor: descriptor(),
            faces: vec![face(100.0)],
        };
        let dets = e.detect(&image()).unwrap();
        assert_eq!(dets.len(), 1);
        assert!(dets[0].embedding.is_empty());
        assert_eq!(dets[0].kps, [[1.0, 2.0]; 5]);
    }

    #[test]
    fn largest_face_default_takes_the_first_analyzed() {
        let e = StubEmbedder {
            descriptor: descriptor(),
            // Already largest-first, the insightface order an implementation must return.
            faces: vec![face(400.0), face(100.0)],
        };
        let largest = e.largest_face(&image()).unwrap();
        assert_eq!(largest.embedding.len(), 512);
        assert_eq!(largest.embedding[0], 400.0);
    }

    #[test]
    fn largest_face_errors_when_no_face() {
        let e = StubEmbedder {
            descriptor: descriptor(),
            faces: Vec::new(),
        };
        let err = e.largest_face(&image()).unwrap_err();
        assert!(matches!(err, crate::Error::Msg(_)));
        assert!(err.to_string().contains("no face detected"));
    }

    #[test]
    fn descriptor_advertises_embedding_dim() {
        assert_eq!(descriptor().embedding_dim, 512);
        assert_eq!(descriptor().backend, "candle");
    }
}
