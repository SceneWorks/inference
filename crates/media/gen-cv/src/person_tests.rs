use super::*;
#[test]
fn letterbox_centers_pad_on_the_short_axis() {
    // bus.jpg is 810×1080 (w×h): fit-by-height → 480×640, 80px pad each side in x.
    let lb = Letterbox::compute(810, 1080);
    assert!((lb.ratio - 0.59259).abs() < 1e-4, "ratio {}", lb.ratio);
    assert_eq!(lb.pad_x, 80.0);
    assert_eq!(lb.pad_y, 0.0);
}

#[test]
fn un_letterbox_inverts_the_forward_mapping() {
    let lb = Letterbox::compute(810, 1080);
    // A point at original (300, 540) maps forward then back to itself.
    let fx = 300.0 * lb.ratio + lb.pad_x;
    let fy = 540.0 * lb.ratio + lb.pad_y;
    assert!((lb.un_x(fx) - 300.0).abs() < 1e-3);
    assert!((lb.un_y(fy) - 540.0).abs() < 1e-3);
}

#[test]
fn decode_keeps_person_anchors_above_conf_and_builds_xyxy() {
    // (1,6,2): 4 box channels + 2 class channels, 2 anchors, channel-major.
    let shape = [1_i64, 6, 2];
    // anchor0: strong person; anchor1: below conf.
    let data = vec![
        100.0, 200.0, // cx
        100.0, 200.0, // cy
        40.0, 10.0, // w
        80.0, 10.0, // h
        0.9, 0.1, // class0 = person
        0.1, 0.05, // class1 = other
    ];
    let lb = Letterbox {
        ratio: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };
    let dets = decode(&data, &shape, &lb, 0.25, 640, 640).expect("valid shape decodes");
    assert_eq!(dets.len(), 1, "only the above-conf person anchor survives");
    let d = dets[0];
    assert!((d.x1 - 80.0).abs() < 1e-3);
    assert!((d.y1 - 60.0).abs() < 1e-3);
    assert!((d.x2 - 120.0).abs() < 1e-3);
    assert!((d.y2 - 140.0).abs() < 1e-3);
    assert!((d.score - 0.9).abs() < 1e-6);
}

#[test]
fn decode_clamps_boxes_to_the_frame() {
    let shape = [1_i64, 5, 1];
    // A box hanging off the top-left corner.
    let data = vec![10.0, 10.0, 60.0, 60.0, 0.9];
    let lb = Letterbox {
        ratio: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };
    let dets = decode(&data, &shape, &lb, 0.25, 640, 640).expect("valid shape decodes");
    assert_eq!(dets.len(), 1);
    assert_eq!(dets[0].x1, 0.0);
    assert_eq!(dets[0].y1, 0.0);
}

#[test]
fn decode_rejects_malformed_output_shape_and_length() {
    let lb = Letterbox {
        ratio: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };
    // Rank < 3 (a 2-D output from a mis-pinned ONNX export) → Engine error, not an OOB panic.
    let rank2 = decode(&[0.0; 12], &[1, 6], &lb, 0.25, 640, 640);
    assert!(
        matches!(rank2, Err(Error::Engine(_))),
        "rank-2 output must be rejected, got {rank2:?}"
    );
    // Well-formed rank but the data buffer is shorter than channels×anchors → Engine error.
    // (1,6,2) needs 12 values; hand it 5 so `data[score_ch*anchors + a]` would read OOB.
    let short = decode(&[0.0; 5], &[1, 6, 2], &lb, 0.25, 640, 640);
    assert!(
        matches!(short, Err(Error::Engine(_))),
        "truncated data buffer must be rejected, got {short:?}"
    );
    // channels < 5 means this is not the detector head we pinned. Treating it as no detections
    // would hide a mis-pinned export as "no people found".
    let no_person = decode(&[0.0; 8], &[1, 4, 2], &lb, 0.25, 640, 640);
    assert!(
        matches!(no_person, Err(Error::Engine(_))),
        "channels<5 must be rejected, got {no_person:?}"
    );
}

/// sc-8906 / F-104: the shared letterbox preprocessor places pixels + pads identically
/// whichever layout the platform selects. Runs against the layout compiled on this lane
/// (NHWC on macOS, NCHW off-Mac) via `InputLayout::index`, so the SAME buffer contents are
/// asserted through the SAME indexing the production path uses — pinning that the dedup
/// preserved both the geometry and the (previously divergent) per-layout index math.
#[test]
fn preprocess_letterbox_places_pixels_and_pads_per_layout() {
    // A square image fills the 640² input edge-to-edge (ratio 1.0, no pad), so every
    // in-bounds pixel is written and the buffer carries the source color, not PAD_VALUE.
    let mut img = RgbImage::new(DET as u32, DET as u32);
    for px in img.pixels_mut() {
        *px = image::Rgb([10, 20, 30]);
    }
    for layout in [InputLayout::Nhwc, InputLayout::Nchw] {
        let (buf, lb) = preprocess_letterbox(&img, layout);
        assert_eq!(buf.len(), DET * DET * 3);
        assert!((lb.ratio - 1.0).abs() < 1e-6, "square image → ratio 1.0");
        assert_eq!((lb.pad_x, lb.pad_y), (0.0, 0.0), "no letterbox pad");
        // A center pixel carries the source color at every channel via the layout's own index.
        let (cx, cy) = (DET / 2, DET / 2);
        for (c, &raw) in [10u8, 20, 30].iter().enumerate() {
            let v = buf[layout.index(cx, cy, c)];
            assert!(
                (v - raw as f32 / 255.0).abs() < 1e-3,
                "channel {c} at center should be {raw}/255, got {v}"
            );
        }

        // A portrait image (narrower than tall) fits by height → horizontal pad columns stay
        // at PAD_VALUE/255. Column 0 is entirely inside the pad band.
        let portrait = RgbImage::from_pixel(200, 400, image::Rgb([255, 255, 255]));
        let (pbuf, plb) = preprocess_letterbox(&portrait, layout);
        assert!(plb.pad_x > 0.0, "portrait letterboxes with horizontal pad");
        let pad = PAD_VALUE / 255.0;
        assert!(
            (pbuf[layout.index(0, DET / 2, 0)] - pad).abs() < 1e-6,
            "left pad column stays at PAD_VALUE"
        );
    }
}

#[test]
fn nms_drops_high_overlap_keeps_disjoint() {
    let strong = Detection {
        x1: 0.0,
        y1: 0.0,
        x2: 100.0,
        y2: 100.0,
        score: 0.9,
    };
    let overlap = Detection {
        x1: 5.0,
        y1: 5.0,
        x2: 105.0,
        y2: 105.0,
        score: 0.8,
    };
    let disjoint = Detection {
        x1: 400.0,
        y1: 400.0,
        x2: 500.0,
        y2: 500.0,
        score: 0.7,
    };
    let kept = nms(vec![overlap, disjoint, strong], NMS_IOU);
    assert_eq!(kept.len(), 2, "overlapping duplicate suppressed");
    assert!((kept[0].score - 0.9).abs() < 1e-6, "highest score first");
    assert!(kept.iter().any(|d| (d.score - 0.7).abs() < 1e-6));
}
