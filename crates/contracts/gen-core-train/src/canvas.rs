//! The aspect-preserving training canvas — one rule every square-bucket trainer uses to size a
//! captioned item at a bucket edge.
//!
//! A bucket edge `edge` names an **area budget** of about `edge²` pixels, not a square: each item
//! keeps its own aspect, is resized (never cropped, never upscaled past its own area) onto a canvas
//! of at most `edge²` pixels, and both canvas sides floor to the family's `stride` (its VAE factor
//! times its patch size, or whatever multiple its latent grid needs to tile). Before this rule the
//! trainers centre-cropped every item to its largest square and resized that to `edge²` — a
//! 1365×2048 portrait lost a third of its frame and was upscaled 1.5×, training soft, washed-out
//! adapters for families whose DiTs take `h ≠ w` and render non-square anyway.
//!
//! Two bounds keep the canvas inside what every family can position-encode:
//!
//! * a source more extreme than [`MAX_TRAINING_ASPECT`] (4:1) is centre-cropped to that aspect
//!   first ([`canvas_crop`]) — a panorama strip would otherwise train a canvas many times longer
//!   than any render;
//! * a family whose positional table caps a side (`max_side`) scales an over-long canvas down at
//!   its aspect until the long side fits (a smaller area, never a further crop).
//!
//! The rule is pure arithmetic over the source dimensions. The source box ([`canvas_crop`], also
//! [`TrainingCanvas::crop`]) depends only on the source's own dimensions — never on the bucket edge
//! or the family — so it is a [`CropRule`](crate::subject_mask::CropRule) a trainer hands its
//! subject masks (which share the image's dimensions) exactly as it cuts the image: the whole frame
//! for any source within the aspect bound.

use crate::subject_mask::CropBox;

/// The most elongated aspect (long side / short side) a training canvas keeps; more extreme sources
/// are centre-cropped to it.
pub const MAX_TRAINING_ASPECT: f64 = 4.0;

/// The canvas `(width, height)` of a `width × height` source at bucket edge `edge`: the source's
/// own aspect at about `edge²` pixels, never upscaled past the source (a source smaller than
/// `edge²` trains at its own area), both sides floored to `stride` (at least one stride). The
/// canvas therefore never exceeds the source's area (once that holds a stride per side) nor
/// `edge²`; the aspect drift the floor introduces is under one stride per side. A degenerate (zero)
/// side falls back to the square `edge` canvas.
///
/// This is the MiniMax-H3-Image trainer's rule (SW-F369), parameterised by the family's stride.
pub fn aspect_canvas((width, height): (u32, u32), edge: u32, stride: u32) -> (u32, u32) {
    let stride_f = f64::from(stride.max(1));
    // The epsilon keeps a side that lands on a stride multiple through a square root from flooring
    // a whole stride down on rounding error.
    let snap = |v: f64| ((v / stride_f + 1e-9).floor() * stride_f).max(stride_f) as u32;
    if width == 0 || height == 0 {
        let e = snap(f64::from(edge));
        return (e, e);
    }
    let (w, h) = (f64::from(width), f64::from(height));
    let area = f64::from(edge) * f64::from(edge);
    if w * h <= area {
        // Never upscale: a source at or below the edge's area trains at its own size.
        return (snap(w), snap(h));
    }
    let scale = (area / (w * h)).sqrt();
    (snap(w * scale), snap(h * scale))
}

/// The training source box of a `width × height` image: the whole frame within
/// [`MAX_TRAINING_ASPECT`], else the centred box at exactly that aspect. Edge- and
/// family-independent, so it doubles as the trainer's subject-mask
/// [`CropRule`](crate::subject_mask::CropRule).
pub fn canvas_crop(width: u32, height: u32) -> CropBox {
    aspect_crop(width, height, MAX_TRAINING_ASPECT)
}

/// The centred source box a `width × height` image is cut to before its canvas resize: the whole
/// frame when its aspect is within `max_aspect`, else the centred box at exactly that aspect (the
/// short side kept whole, the long side floored, the offset floored like
/// [`CropBox::center_square`]).
pub fn aspect_crop(width: u32, height: u32, max_aspect: f64) -> CropBox {
    if width == 0 || height == 0 {
        return CropBox::full(width, height);
    }
    let max_aspect = if max_aspect.is_finite() {
        max_aspect.max(1.0)
    } else {
        MAX_TRAINING_ASPECT
    };
    let (long, short) = (width.max(height), width.min(height));
    let cap = ((f64::from(short) * max_aspect) + 1e-9).floor() as u32;
    if long <= cap {
        return CropBox::full(width, height);
    }
    if width >= height {
        CropBox {
            x: (width - cap) / 2,
            y: 0,
            width: cap,
            height,
        }
    } else {
        CropBox {
            x: 0,
            y: (height - cap) / 2,
            width,
            height: cap,
        }
    }
}

/// One item's training geometry at one bucket edge: the source box it is cut to and the canvas it
/// is resized onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingCanvas {
    /// The source box (the whole frame unless the source is more elongated than the bound).
    pub crop: CropBox,
    /// The canvas width in pixels (a multiple of the family stride).
    pub width: u32,
    /// The canvas height in pixels (a multiple of the family stride).
    pub height: u32,
}

impl TrainingCanvas {
    /// The canvas `(width, height)`.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Canvas pixels.
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// The training geometry of a `width × height` source at bucket edge `edge` for a family whose
/// canvas sides are multiples of `stride` and (optionally) at most `max_side`: cut the source to
/// [`canvas_crop`], size the cut with [`aspect_canvas`], and — when the long side exceeds
/// `max_side` — scale the canvas down at its aspect until it fits (stride-floored, at least one
/// stride). A `max_side` below `edge` is raised to `edge`, so the cap never shrinks the square
/// canvas the family trained at before.
pub fn training_canvas(
    (width, height): (u32, u32),
    edge: u32,
    stride: u32,
    max_side: Option<u32>,
) -> TrainingCanvas {
    let crop = canvas_crop(width, height);
    let (mut w, mut h) = aspect_canvas((crop.width, crop.height), edge, stride);
    if let Some(cap) = max_side {
        let stride_f = f64::from(stride.max(1));
        let cap = cap.max(edge);
        let long = w.max(h);
        if long > cap {
            let k = f64::from(cap) / f64::from(long);
            let snap = |v: u32| {
                ((f64::from(v) * k / stride_f + 1e-9).floor() * stride_f).max(stride_f) as u32
            };
            (w, h) = (snap(w), snap(h));
        }
    }
    TrainingCanvas {
        crop,
        width: w,
        height: h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_canvas_keeps_the_aspect_at_the_edge_area_and_never_upscales() {
        // Larger than edge²: scaled down at its aspect, stride-floored, area within edge².
        let (w, h) = aspect_canvas((3000, 2000), 2048, 32);
        assert_eq!((w % 32, h % 32), (0, 0));
        assert!((f64::from(w) / f64::from(h) - 1.5).abs() < 0.05);
        assert!(u64::from(w) * u64::from(h) <= 2048 * 2048);
        // A square source at edge²: the square canvas; a larger square scales down.
        assert_eq!(aspect_canvas((2048, 2048), 2048, 32), (2048, 2048));
        assert_eq!(aspect_canvas((4096, 4096), 1000, 32), (992, 992));
        // The 1365×2048 portrait at 2048: kept whole at its own area — not centre-cropped to a
        // square and upscaled to 2048².
        assert_eq!(aspect_canvas((1365, 2048), 2048, 32), (1344, 2048));
        assert_eq!(aspect_canvas((1365, 2048), 2048, 16), (1360, 2048));
        // The same portrait at 1024: scaled down, height stays the long side.
        let (w, h) = aspect_canvas((1365, 2048), 1024, 16);
        assert!(h > w && u64::from(w) * u64::from(h) <= 1024 * 1024);
        assert!((f64::from(w) / f64::from(h) - 1365.0 / 2048.0).abs() < 0.03);
        // Landscape keeps width as the long side: the pair is (width, height).
        assert_eq!(aspect_canvas((2048, 1365), 2048, 32), (2048, 1344));
        // A tiny source keeps one stride; a degenerate one falls back to the square edge.
        assert_eq!(aspect_canvas((10, 4000), 16, 32), (32, 320));
        assert_eq!(aspect_canvas((10, 20), 2048, 32), (32, 32));
        assert_eq!(aspect_canvas((0, 20), 1000, 32), (992, 992));
        // Never larger than the source's area (once it holds a stride on each side).
        for (sw, sh) in [
            (1365, 2048),
            (640, 480),
            (5000, 300),
            (1024, 1024),
            (3000, 2000),
        ] {
            for stride in [16, 32, 64] {
                let (w, h) = aspect_canvas((sw, sh), 2048, stride);
                assert!(u64::from(w) * u64::from(h) <= u64::from(sw) * u64::from(sh));
                assert_eq!((w % stride, h % stride), (0, 0));
            }
        }
    }

    #[test]
    fn the_hoisted_rule_matches_the_h3_trainer() {
        use crate::h3_image::TRAINING_CANVAS_STRIDE;
        for dims in [
            (1365, 2048),
            (3000, 2000),
            (640, 480),
            (4096, 4096),
            (10, 4000),
        ] {
            for edge in [512, 1024, 2048] {
                let (w, h) = aspect_canvas(dims, edge, TRAINING_CANVAS_STRIDE);
                assert_eq!(
                    (w % TRAINING_CANVAS_STRIDE, h % TRAINING_CANVAS_STRIDE),
                    (0, 0)
                );
                // Matches the canvas the bounded rule yields for any source within 4:1.
                if dims.0.max(dims.1) <= 4 * dims.0.min(dims.1) {
                    let c = training_canvas(dims, edge, TRAINING_CANVAS_STRIDE, None);
                    assert_eq!(c.size(), (w, h));
                    assert_eq!(c.crop, CropBox::full(dims.0, dims.1));
                }
            }
        }
    }

    #[test]
    fn an_extreme_strip_is_centre_cropped_to_the_aspect_bound() {
        // 5000×300 is 16.7:1 — cropped to a centred 1200×300 (4:1), then sized at 1024².
        let c = training_canvas((5000, 300), 1024, 32, None);
        assert_eq!(
            c.crop,
            CropBox {
                x: 1900,
                y: 0,
                width: 1200,
                height: 300
            }
        );
        // The crop's own area (360 000) is below 1024², so it trains at its own size.
        assert_eq!(c.size(), (1184, 288));
        // A tall strip crops the height, centred.
        let c = training_canvas((400, 4000), 512, 16, None);
        assert_eq!(
            c.crop,
            CropBox {
                x: 0,
                y: 1200,
                width: 400,
                height: 1600
            }
        );
        let (w, h) = c.size();
        assert!(h > w && u64::from(w) * u64::from(h) <= 512 * 512);
        // Exactly 4:1 is kept whole.
        assert_eq!(
            aspect_crop(4000, 1000, MAX_TRAINING_ASPECT),
            CropBox::full(4000, 1000)
        );
    }

    #[test]
    fn a_positional_side_cap_scales_an_over_long_canvas_down() {
        // A 4:1 source at 2048 is 4096×1024; a 3072 cap scales it to 3072×768 (same aspect, same
        // crop — the source box never depends on the cap).
        let c = training_canvas((8000, 2000), 2048, 16, Some(3072));
        assert_eq!(c.crop, CropBox::full(8000, 2000));
        assert_eq!(c.size(), (3072, 768));
        assert_eq!(
            training_canvas((8000, 2000), 2048, 16, None).size(),
            (4096, 1024)
        );
        // Every canvas long side stays within the cap across shapes and edges, the crop is the
        // edge-independent `canvas_crop`, and the area stays within edge².
        for edge in [512u32, 1024, 1536, 2048] {
            for dims in [
                (5000, 300),
                (300, 5000),
                (1365, 2048),
                (4000, 1000),
                (8000, 6000),
            ] {
                let c = training_canvas(dims, edge, 16, Some(3072));
                assert!(c.width.max(c.height) <= 3072, "{dims:?} @ {edge}: {c:?}");
                assert!(c.area() <= u64::from(edge) * u64::from(edge));
                assert_eq!(c.crop, canvas_crop(dims.0, dims.1));
                assert_eq!((c.width % 16, c.height % 16), (0, 0));
            }
        }
        // A cap below the edge never shrinks the square the family trained before.
        assert_eq!(
            training_canvas((4096, 4096), 4096, 16, Some(3072)).size(),
            (4096, 4096)
        );
        // The 1365×2048 portrait (1.5:1) is untouched by a 3072 cap at 2048.
        let c = training_canvas((1365, 2048), 2048, 16, Some(3072));
        assert_eq!(c.crop, CropBox::full(1365, 2048));
        assert_eq!(c.size(), (1360, 2048));
    }

    #[test]
    fn the_worst_case_canvas_area_is_the_square_edge() {
        // The memory preflight prices the square `edge²`: no canvas exceeds it.
        for edge in [256u32, 512, 768, 1024, 2048] {
            for dims in [
                (1, 1),
                (1365, 2048),
                (9999, 10),
                (2048, 2048),
                (12000, 9000),
            ] {
                let c = training_canvas(dims, edge, 16, None);
                assert!(c.area() <= u64::from(edge) * u64::from(edge));
            }
        }
    }
}
