use std::path::PathBuf;

use super::*;

fn mask(width: u32, height: u32, f: impl Fn(u32, u32) -> f32) -> SubjectMask {
    let mut values = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            values.push(f(x, y));
        }
    }
    SubjectMask {
        width,
        height,
        values,
    }
}

const ON: SubjectMaskLoss = SubjectMaskLoss {
    background_weight: 0.0,
    subject_weight: 1.0,
};

#[test]
fn center_square_matches_the_trainers_crop() {
    // Same integer arithmetic as mlx-gen's `center_crop_square` / candle-gen's `load_image_tensor`.
    assert_eq!(
        CropBox::center_square(12, 8),
        CropBox {
            x: 2,
            y: 0,
            width: 8,
            height: 8
        }
    );
    assert_eq!(
        CropBox::center_square(7, 10),
        CropBox {
            x: 0,
            y: 1,
            width: 7,
            height: 7
        }
    );
    assert_eq!(CropBox::center_square(5, 5), CropBox::full(5, 5));
}

#[test]
fn area_resample_integer_ratio_is_the_block_mean() {
    // 4x4 → 2x2: each cell is the mean of its 2x2 block.
    let m = mask(4, 4, |x, y| (y * 4 + x) as f32);
    let out = m.area_resample(CropBox::full(4, 4), 2, 2);
    assert_eq!(out, vec![2.5, 4.5, 10.5, 12.5]);
}

#[test]
fn area_resample_fractional_edges_get_fractional_weight() {
    // 3 px → 2 cells: cell 0 covers px0 fully + half of px1; cell 1 the other half + px2.
    let m = mask(3, 1, |x, _| if x == 1 { 1.0 } else { 0.0 });
    let out = m.area_resample(CropBox::full(3, 1), 2, 1);
    assert!((out[0] - 1.0 / 3.0).abs() < 1e-6, "{out:?}");
    assert!((out[1] - 1.0 / 3.0).abs() < 1e-6, "{out:?}");
    // A constant mask stays constant under any ratio (taps sum to 1).
    let ones = mask(7, 5, |_, _| 1.0);
    for v in ones.area_resample(CropBox::full(7, 5), 3, 2) {
        assert!((v - 1.0).abs() < 1e-6);
    }
}

#[test]
fn asymmetric_mask_stays_aligned_after_center_crop() {
    // A 12x8 landscape image whose subject occupies only the crop's right half (x ∈ [6, 10)).
    // The centre crop is x ∈ [2, 10); on a 2x2 grid the subject must land in the RIGHT column only.
    // A resampler that ignored the crop origin (x0 = 2) would put subject mass in the left column.
    let m = mask(12, 8, |x, _| if (6..10).contains(&x) { 1.0 } else { 0.0 });
    let out = m.area_resample(CropBox::center_square(12, 8), 2, 2);
    assert_eq!(out, vec![0.0, 1.0, 0.0, 1.0]);
    // And the uncropped full frame would NOT be aligned the same way (x ∈ [6,10) of [0,12)).
    let full = m.area_resample(CropBox::full(12, 8), 2, 2);
    assert_ne!(full, out);
}

#[test]
fn require_subject_masks_names_the_missing_images_capped() {
    let mut items: Vec<TrainingItem> = (0..13)
        .map(|i| TrainingItem::captioned(PathBuf::from(format!("/d/img{i:02}.png")), "c".into()))
        .collect();
    items[0].subject_mask_path = Some(PathBuf::from("/d/masks/0.png"));
    let err = require_subject_masks("t", &items).unwrap_err().to_string();
    assert!(err.contains("12 of 13 have none"), "{err}");
    assert!(
        err.contains("img01.png") && err.contains("img10.png"),
        "{err}"
    );
    assert!(
        !err.contains("img00.png"),
        "the masked image is not named: {err}"
    );
    assert!(
        !err.contains("img11.png"),
        "names past the cap are counted: {err}"
    );
    assert!(err.ends_with("and 2 more"), "{err}");
    for item in &mut items {
        item.subject_mask_path = Some(PathBuf::from("/d/m.png"));
    }
    assert!(require_subject_masks("t", &items).is_ok());
}

fn write_png(dir: &Path, name: &str, w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> PathBuf {
    let path = dir.join(name);
    image::GrayImage::from_fn(w, h, |x, y| image::Luma([f(x, y)]))
        .save(&path)
        .unwrap();
    path
}

#[test]
fn latent_weights_follow_the_mask_and_refuse_bad_masks() {
    let dir = tempfile::tempdir().unwrap();
    let img = write_png(dir.path(), "img.png", 12, 8, |_, _| 128);
    let right = write_png(dir.path(), "right.png", 12, 8, |x, _| {
        if (6..10).contains(&x) {
            255
        } else {
            0
        }
    });
    let mut item = TrainingItem::captioned(img.clone(), "c".into());

    // Missing.
    let err = subject_mask_latent_weights("t", &item, &ON, CropBox::center_square, 2, 2)
        .unwrap_err()
        .to_string();
    assert!(err.contains("img.png has no subject mask"), "{err}");

    // Weight map: bg 0.25, subject 1.0 → right column 1.0, left 0.25.
    item.subject_mask_path = Some(right);
    let cfg = SubjectMaskLoss {
        background_weight: 0.25,
        subject_weight: 1.0,
    };
    let w = subject_mask_latent_weights("t", &item, &cfg, CropBox::center_square, 2, 2).unwrap();
    assert_eq!(w, vec![0.25, 1.0, 0.25, 1.0]);

    // Size mismatch.
    item.subject_mask_path = Some(write_png(dir.path(), "small.png", 6, 4, |_, _| 255));
    let err = subject_mask_latent_weights("t", &item, &ON, CropBox::center_square, 2, 2)
        .unwrap_err()
        .to_string();
    assert!(err.contains("is 6x4 but the image is 12x8"), "{err}");

    // Empty.
    item.subject_mask_path = Some(write_png(dir.path(), "empty.png", 12, 8, |_, _| 0));
    let err = subject_mask_latent_weights("t", &item, &ON, CropBox::center_square, 2, 2)
        .unwrap_err()
        .to_string();
    assert!(err.contains("is empty"), "{err}");

    // Subject only outside the centre crop (x < 2): refused at bg 0, accepted at bg > 0.
    item.subject_mask_path = Some(write_png(dir.path(), "edge.png", 12, 8, |x, _| {
        if x < 2 {
            255
        } else {
            0
        }
    }));
    let err = subject_mask_latent_weights("t", &item, &ON, CropBox::center_square, 2, 2)
        .unwrap_err()
        .to_string();
    assert!(err.contains("outside the training crop"), "{err}");
    let w = subject_mask_latent_weights("t", &item, &cfg, CropBox::center_square, 2, 2).unwrap();
    assert_eq!(w, vec![0.25; 4]);
}

/// sc-24832: perceptual masks load only when the restricted normal loss is on, map reference keys
/// to items item-major, and crop with the trainer's rule before resampling onto the pixel grid.
/// Mutation: key the item by `entry` instead of `entry / entries_per_item` ⇒ entry 1 reads item 1
/// (left-half mask) instead of item 0 ⇒ red.
#[test]
fn perceptual_masks_follow_the_item_major_cache_and_crop() {
    let dir = tempfile::tempdir().unwrap();
    // A 12×8 image; item 0's mask is the right half, item 1's the left half.
    let img = write_png(dir.path(), "img.png", 12, 8, |_, _| 128);
    let right = write_png(
        dir.path(),
        "right.png",
        12,
        8,
        |x, _| if x >= 6 { 255 } else { 0 },
    );
    let left = write_png(
        dir.path(),
        "left.png",
        12,
        8,
        |x, _| if x < 6 { 255 } else { 0 },
    );
    let item = |m: &PathBuf| {
        let mut it = TrainingItem::captioned(img.clone(), "c".into());
        it.subject_mask_path = Some(m.clone());
        it
    };
    let items = vec![item(&right), item(&left)];
    let mut cfg = TrainingConfig::default();
    assert!(
        PerceptualSubjectMasks::load("t", &items, &cfg, 2, CropBox::center_square)
            .unwrap()
            .is_none()
    );
    cfg.body_losses.normal.weight = 0.1;
    cfg.body_losses.normal_restrict_to_subject = true;
    let m = PerceptualSubjectMasks::load("t", &items, &cfg, 2, CropBox::center_square)
        .unwrap()
        .unwrap();
    // Center-square crop of 12×8 is x ∈ [2, 10): the right half covers crop columns 4..8.
    assert_eq!(m.pixel_mask(0, 2, 1).unwrap(), vec![0.0, 1.0]);
    assert_eq!(
        m.pixel_mask(1, 2, 1).unwrap(),
        vec![0.0, 1.0],
        "entry 1 is item 0's second bucket"
    );
    assert_eq!(m.pixel_mask(2, 2, 1).unwrap(), vec![1.0, 0.0]);
    assert!(m.pixel_mask(4, 2, 1).is_err());
    // A missing mask is refused naming the image.
    let mut bad = items.clone();
    bad[1].subject_mask_path = None;
    let e = PerceptualSubjectMasks::load("t", &bad, &cfg, 2, CropBox::center_square).unwrap_err();
    assert!(e.to_string().contains("img.png"), "{e}");
}
