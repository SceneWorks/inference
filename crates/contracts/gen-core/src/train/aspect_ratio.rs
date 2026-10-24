//! Rectangular training geometry and deterministic homogeneous microbatch scheduling.
//! Legacy square BucketSchedule/cache_index semantics are deliberately separate and unchanged.
use super::{subject_mask::CropBox, BucketSchedule, TrainingConfig, MAX_RESOLUTION_BUCKETS};
use serde::{Deserialize, Serialize};

pub const RECTANGULAR_POLICY: &str = "moss-aspect-assignment-batch-v1";
pub const CROP_POLICY: &str = "largest-ratio-center-floor-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AspectRatioBucket {
    pub width: u32,
    pub height: u32,
    pub repeats: u32,
}

pub fn validate(config: &TrainingConfig) -> crate::Result<()> {
    let rows = &config.aspect_ratio_buckets;
    if rows.is_empty() {
        return Ok(());
    }
    if !config.resolution_buckets.is_empty() {
        return Err(crate::Error::Msg(
            "square and rectangular training buckets cannot be combined".into(),
        ));
    }
    if rows.len() > MAX_RESOLUTION_BUCKETS {
        return Err(crate::Error::Msg(
            "at most eight rectangular bucket rows are supported".into(),
        ));
    }
    let mut seen = std::collections::HashSet::new();
    for row in rows {
        if row.width == 0
            || row.height == 0
            || !row.width.is_multiple_of(32)
            || !row.height.is_multiple_of(32)
            || !(1..=100).contains(&row.repeats)
        {
            return Err(crate::Error::Msg("rectangular dimensions must be positive multiples of 32 and repeats must be 1..100".into()));
        }
        if !seen.insert((row.width, row.height)) {
            return Err(crate::Error::Msg(
                "duplicate rectangular bucket geometry".into(),
            ));
        }
    }
    Ok(())
}

fn product(a: u128, b: u128) -> crate::Result<u128> {
    a.checked_mul(b)
        .ok_or_else(|| crate::Error::Msg("aspect-ratio comparison overflow".into()))
}

/// Minimum symmetric aspect distortion, with stable earliest-row ties. No floating-point sorting.
pub fn assignment(width: u32, height: u32, rows: &[AspectRatioBucket]) -> crate::Result<usize> {
    if width == 0 || height == 0 || rows.is_empty() {
        return Err(crate::Error::Msg(
            "aspect assignment needs nonempty rows and positive source geometry".into(),
        ));
    }
    let mut best = None;
    for (index, row) in rows.iter().enumerate() {
        if row.width == 0 || row.height == 0 {
            return Err(crate::Error::Msg(
                "aspect assignment needs positive bucket geometry".into(),
            ));
        }
        let a = product(u128::from(width), u128::from(row.height))?;
        let b = product(u128::from(height), u128::from(row.width))?;
        let (num, den) = (a.max(b), a.min(b));
        let improve = match best {
            None => true,
            Some((_, n, d)) => product(num, d)? < product(n, den)?,
        };
        if improve {
            best = Some((index, num, den));
        }
    }
    Ok(best.expect("nonempty rows").0)
}

/// Largest source rectangle at the target aspect, flooring size and half-offset consistently.
pub fn center_crop(
    width: u32,
    height: u32,
    target_w: u32,
    target_h: u32,
) -> crate::Result<CropBox> {
    if [width, height, target_w, target_h].contains(&0) {
        return Err(crate::Error::Msg(
            "center crop requires positive geometry".into(),
        ));
    }
    let (crop_w, crop_h) =
        if u64::from(width) * u64::from(target_h) > u64::from(height) * u64::from(target_w) {
            (
                (u64::from(height) * u64::from(target_w) / u64::from(target_h)) as u32,
                height,
            )
        } else {
            (
                width,
                (u64::from(width) * u64::from(target_h) / u64::from(target_w)) as u32,
            )
        };
    if crop_w == 0 || crop_h == 0 {
        return Err(crate::Error::Msg(
            "center crop rounded to zero pixels".into(),
        ));
    }
    Ok(CropBox {
        x: (width - crop_w) / 2,
        y: (height - crop_h) / 2,
        width: crop_w,
        height: crop_h,
    })
}

/// Shared RGB preprocessing. Mask callers use the same center_crop box with the shared mask resampler.
pub fn resize_rgb(source: &image::RgbImage, row: AspectRatioBucket) -> crate::Result<Vec<f32>> {
    let crop = center_crop(source.width(), source.height(), row.width, row.height)?;
    let pixels =
        image::imageops::crop_imm(source, crop.x, crop.y, crop.width, crop.height).to_image();
    crate::imageops::resize_lanczos_u8(
        pixels.as_raw(),
        crop.height as usize,
        crop.width as usize,
        row.height as usize,
        row.width as usize,
    )
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AspectBatch {
    pub bucket: usize,
    pub items: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct AspectSchedule {
    pub assignments: Vec<usize>,
    rows: Vec<AspectRatioBucket>,
    batch_size: usize,
    seed: u64,
}

impl AspectSchedule {
    pub fn new(
        dimensions: &[(u32, u32)],
        config: &TrainingConfig,
        seed: u64,
    ) -> crate::Result<Self> {
        validate(config)?;
        if dimensions.is_empty() || config.batch_size == 0 || config.aspect_ratio_buckets.is_empty()
        {
            return Err(crate::Error::Msg(
                "rectangular schedule requires items, rows and a positive batch size".into(),
            ));
        }
        let assignments = dimensions
            .iter()
            .map(|&(w, h)| assignment(w, h, &config.aspect_ratio_buckets))
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Self {
            assignments,
            rows: config.aspect_ratio_buckets.clone(),
            batch_size: config.batch_size as usize,
            seed,
        })
    }

    /// Per-bucket seeded shuffle, homogeneous chunks retaining tails, then batch-list shuffle.
    pub fn epoch(&self, epoch: u64) -> Vec<AspectBatch> {
        let mut batches = Vec::new();
        for (bucket, row) in self.rows.iter().enumerate() {
            let mut items = Vec::new();
            for (item, &selected) in self.assignments.iter().enumerate() {
                if selected == bucket {
                    for _ in 0..row.repeats {
                        items.push(item);
                    }
                }
            }
            BucketSchedule::shuffle_items(
                &mut items,
                self.seed ^ 0x4153_5045_4354_0001,
                epoch,
                bucket as u64,
            );
            for chunk in items.chunks(self.batch_size) {
                batches.push(AspectBatch {
                    bucket,
                    items: chunk.to_vec(),
                });
            }
        }
        BucketSchedule::shuffle_items(&mut batches, self.seed ^ 0x4241_5443_4845_0001, epoch, 0);
        batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(width: u32, height: u32, repeats: u32) -> AspectRatioBucket {
        AspectRatioBucket {
            width,
            height,
            repeats,
        }
    }
    #[test]
    fn symmetric_assignment_keeps_ties_order_and_orientation() {
        let rows = [
            row(64, 32, 2),
            row(32, 64, 3),
            row(64, 64, 1),
            row(128, 64, 1),
        ];
        assert_eq!(assignment(900, 400, &rows).unwrap(), 0);
        assert_eq!(assignment(400, 900, &rows).unwrap(), 1);
        assert_eq!(assignment(601, 600, &rows).unwrap(), 2);
        assert_eq!(assignment(u32::MAX, u32::MAX - 1, &rows).unwrap(), 2);
        assert!(assignment(0, 10, &rows).is_err());
    }
    #[test]
    fn crop_floors_size_and_offset_for_asymmetric_source() {
        assert_eq!(
            center_crop(101, 60, 64, 32).unwrap(),
            CropBox {
                x: 0,
                y: 5,
                width: 101,
                height: 50
            }
        );
        assert_eq!(
            center_crop(60, 101, 32, 64).unwrap(),
            CropBox {
                x: 5,
                y: 0,
                width: 50,
                height: 101
            }
        );
    }
    #[test]
    fn batches_assign_once_repeat_by_selected_row_and_keep_short_tails() {
        let config = TrainingConfig {
            aspect_ratio_buckets: vec![row(64, 32, 2), row(32, 64, 3)],
            batch_size: 3,
            ..Default::default()
        };
        let schedule = AspectSchedule::new(&[(80, 40), (40, 80), (80, 40)], &config, 173).unwrap();
        assert_eq!(schedule.assignments, vec![0, 1, 0]);
        let batches = schedule.epoch(7);
        assert_eq!(batches, schedule.epoch(7));
        assert!(batches.iter().any(|b| b.items.len() == 1));
        let mut counts = [0; 3];
        for batch in batches {
            for item in batch.items {
                assert_eq!(schedule.assignments[item], batch.bucket);
                counts[item] += 1;
            }
        }
        assert_eq!(counts, [2, 3, 2]);
    }
    #[test]
    fn validation_refuses_duplicates_mixed_and_bad_repeats_without_a_dimension_cap() {
        let mut config = TrainingConfig {
            aspect_ratio_buckets: vec![row(u32::MAX - 31, 32, 100)],
            ..Default::default()
        };
        validate(&config).unwrap();
        config.aspect_ratio_buckets.push(row(u32::MAX - 31, 32, 1));
        assert!(validate(&config).is_err());
        config.aspect_ratio_buckets = vec![row(64, 32, 101)];
        assert!(validate(&config).is_err());
        config.aspect_ratio_buckets = vec![row(64, 32, 1)];
        config.resolution_buckets = vec![super::super::ResolutionBucket {
            resolution: 64,
            repeats: 1,
        }];
        assert!(validate(&config).is_err());
    }
}
