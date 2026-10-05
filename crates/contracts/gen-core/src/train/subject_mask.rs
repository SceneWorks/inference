//! Subject-masked loss weighting (epic 2123, sc-24828) — the backend-neutral half.
//!
//! A subject mask is a single-channel image the same size as its training image (white = subject,
//! black = background, soft edges in between). When [`TrainingConfig::subject_mask_loss`](super::TrainingConfig::subject_mask_loss) is on,
//! every trainer turns each item's mask into a **latent-grid weight map** here, caches it next to
//! the item's latent, and multiplies its per-element loss by that map before reducing.
//!
//! # Geometry — the mask follows the image exactly
//!
//! The trainer crops its image with a [`CropBox`] (today the centred largest square,
//! [`CropBox::center_square`]; a full-frame stretch is [`CropBox::full`]) and resizes the crop onto
//! the training canvas, which the VAE then downsamples onto the latent grid. The mask is cropped
//! with the **same** box and resampled straight onto the latent grid with an exact **area average**
//! (box filter with fractional source-pixel coverage), so every latent cell gets the mean mask value
//! of the source pixels it covers: a cell straddling the subject edge gets a fractional weight, and
//! an asymmetric mask stays aligned with its image after the crop. The crop box is an argument, so a
//! trainer that crops differently (e.g. aspect buckets) passes its own box and caches one map per
//! (image, bucket) — the resampler itself is geometry-agnostic.
//!
//! # Reduction — weighted mean
//!
//! The weighted loss is `mean(w ⊙ ℓ)` over every latent element (the same divisor as the unweighted
//! mean), **not** `Σ(w ⊙ ℓ) / Σw`. With `background_weight = subject_weight = 1` it is exactly the
//! unweighted loss; with `background_weight = 0` a background cell contributes zero loss and zero
//! gradient. Every backend uses this one convention.
//! Because the divisor stays the full element count, the overall gradient magnitude scales by
//! about `background_weight + (subject_weight − background_weight) · coverage` (coverage = the
//! mean mask value): with `background_weight = 0` and a subject covering 20% of the frame the
//! gradient — and so the effective learning rate — is roughly a fifth of the unmasked run's, so
//! raise the learning rate (or the background weight) in proportion if that is not wanted.
//!
//! # Missing and empty masks
//!
//! A run with mask loss on refuses (rather than trains unmasked on) any item without a mask
//! ([`require_subject_masks`]), any mask whose size differs from its image, and any **empty**
//! (all-black) mask: an empty mask means "no subject found", and with a zero background weight it
//! would silently drop the image from the loss. A mask whose subject lies entirely outside the
//! training crop is refused for the same reason when `background_weight == 0`.

use std::path::Path;

use super::{SubjectMaskLoss, TrainingConfig, TrainingItem};

/// At most this many missing images are named in a refusal; the rest are counted.
const MISSING_NAME_CAP: usize = 10;

/// Refuse a mask-loss run whose items do not all carry a
/// [`subject_mask_path`](TrainingItem::subject_mask_path). The message names the images that lack
/// one (the first ten, then "and N more").
pub fn require_subject_masks(label: &str, items: &[TrainingItem]) -> crate::Result<()> {
    let missing: Vec<String> = items
        .iter()
        .filter(|item| item.subject_mask_path.is_none())
        .map(|item| display_name(&item.image_path))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    Err(crate::Error::Msg(format!(
        "{label}: subject-masked loss needs a subject mask for every dataset image; {} of {} have \
         none: {}",
        missing.len(),
        items.len(),
        name_list(&missing)
    )))
}

/// `a, b, c` — or the first ten names plus `and N more`.
pub fn name_list(names: &[String]) -> String {
    let shown = names.len().min(MISSING_NAME_CAP);
    let mut out = names[..shown].join(", ");
    if names.len() > shown {
        out.push_str(&format!(" and {} more", names.len() - shown));
    }
    out
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// A rectangle of the source image, in source pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl CropBox {
    /// The centred largest square of a `width × height` image — exactly the box
    /// `mlx_gen::train::dataset::center_crop_square` and `candle_gen::train::dataset` cut
    /// (`side = min(w, h)`, origin `((w − side) / 2, (h − side) / 2)`, integer division).
    pub fn center_square(width: u32, height: u32) -> Self {
        let side = width.min(height);
        Self {
            x: (width - side) / 2,
            y: (height - side) / 2,
            width: side,
            height: side,
        }
    }

    /// The whole `width × height` image (a trainer that stretches the full frame to its canvas).
    pub fn full(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }
}

/// A decoded subject mask: `values[y * width + x] ∈ [0, 1]` (1 = subject).
#[derive(Clone, Debug, PartialEq)]
pub struct SubjectMask {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}

impl SubjectMask {
    /// Decode a mask image (any format the `image` crate reads; colour is reduced to luma).
    pub fn load(path: &Path) -> crate::Result<Self> {
        let gray = image::open(path)
            .map_err(|e| crate::Error::Msg(format!("decode subject mask {}: {e}", path.display())))?
            .to_luma8();
        let (width, height) = gray.dimensions();
        Ok(Self {
            width,
            height,
            values: gray
                .into_raw()
                .into_iter()
                .map(|v| f32::from(v) / 255.0)
                .collect(),
        })
    }

    /// `true` when every pixel is background (no subject at all).
    pub fn is_empty(&self) -> bool {
        self.values.iter().all(|&v| v == 0.0)
    }

    /// Area-average the `crop` region onto a `grid_w × grid_h` grid (row-major). Each output cell is
    /// the coverage-weighted mean of the source pixels under it, with fractional coverage at cell
    /// borders — the exact box-filter resample, so any crop/grid ratio (integer or not) is aligned.
    pub fn area_resample(&self, crop: CropBox, grid_w: usize, grid_h: usize) -> Vec<f32> {
        assert!(
            crop.width > 0
                && crop.height > 0
                && crop.x + crop.width <= self.width
                && crop.y + crop.height <= self.height,
            "subject mask crop {crop:?} outside {}x{}",
            self.width,
            self.height
        );
        let cols = coverage(crop.x, crop.width, grid_w);
        let rows = coverage(crop.y, crop.height, grid_h);
        let w = self.width as usize;
        // Horizontal pass over the crop's rows, then vertical.
        let mut horiz = vec![0f32; crop.height as usize * grid_w];
        for sy in 0..crop.height as usize {
            let src = &self.values[(crop.y as usize + sy) * w..][..w];
            for (gx, taps) in cols.iter().enumerate() {
                horiz[sy * grid_w + gx] = taps.iter().map(|&(sx, cov)| src[sx] * cov).sum();
            }
        }
        let mut out = vec![0f32; grid_h * grid_w];
        for (gy, taps) in rows.iter().enumerate() {
            for gx in 0..grid_w {
                out[gy * grid_w + gx] = taps
                    .iter()
                    .map(|&(sy, cov)| horiz[(sy - crop.y as usize) * grid_w + gx] * cov)
                    .sum();
            }
        }
        out
    }
}

/// For each of `cells` output cells spanning `[start, start + len)`, the source pixels it covers and
/// each one's normalised coverage (the taps of one cell sum to 1).
fn coverage(start: u32, len: u32, cells: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = f64::from(len) / cells as f64;
    (0..cells)
        .map(|c| {
            let lo = c as f64 * scale;
            let hi = (c + 1) as f64 * scale;
            let mut taps = Vec::new();
            let mut p = lo.floor() as u32;
            while f64::from(p) < hi && p < len {
                let cov = (f64::from(p + 1).min(hi) - f64::from(p).max(lo)) / scale;
                if cov > 0.0 {
                    taps.push(((start + p) as usize, cov as f32));
                }
                p += 1;
            }
            taps
        })
        .collect()
}

/// The **latent-grid loss weight map** of one training item — the one entry point every trainer
/// calls while caching (sc-24828). Reads `item.subject_mask_path`, checks it is the size of
/// `item.image_path`, crops it with `crop_of(image_w, image_h)` (the trainer's own crop rule),
/// area-averages it onto `grid_w × grid_h` (the item's latent spatial grid) and maps each cell's
/// mask value `m` to `cfg.weight(m)`. Row-major `[grid_h, grid_w]`.
///
/// Refuses (with `label` and the image name) a missing mask, a size mismatch, an empty mask, and —
/// when `background_weight == 0` — a mask whose subject lies entirely outside the crop.
pub fn subject_mask_latent_weights(
    label: &str,
    item: &TrainingItem,
    cfg: &SubjectMaskLoss,
    crop_of: impl FnOnce(u32, u32) -> CropBox,
    grid_w: usize,
    grid_h: usize,
) -> crate::Result<Vec<f32>> {
    PreparedSubjectMask::load(label, item, cfg)?.latent_weights(label, crop_of, grid_w, grid_h)
}

/// One item's subject mask, decoded and checked **once** (sc-24828) so a trainer that caches the
/// item at several resolution buckets (sc-2127) resamples it per bucket without re-reading the
/// file: [`load`](Self::load) refuses a missing, mis-sized or empty mask;
/// [`latent_weights`](Self::latent_weights) crops it with the bucket's crop box and area-averages
/// it onto that bucket's latent grid.
#[derive(Clone, Debug)]
pub struct PreparedSubjectMask {
    name: String,
    image_dims: (u32, u32),
    mask: SubjectMask,
    cfg: SubjectMaskLoss,
}

impl PreparedSubjectMask {
    /// Decode and check `item`'s mask against its image (see [`subject_mask_latent_weights`]).
    pub fn load(label: &str, item: &TrainingItem, cfg: &SubjectMaskLoss) -> crate::Result<Self> {
        let name = display_name(&item.image_path);
        let mask_path = item.subject_mask_path.as_deref().ok_or_else(|| {
            crate::Error::Msg(format!(
                "{label}: subject-masked loss is on but image {name} has no subject mask"
            ))
        })?;
        let (iw, ih) = image::image_dimensions(&item.image_path).map_err(|e| {
            crate::Error::Msg(format!(
                "{label}: read image size of {}: {e}",
                item.image_path.display()
            ))
        })?;
        let mask = SubjectMask::load(mask_path)?;
        if (mask.width, mask.height) != (iw, ih) {
            return Err(crate::Error::Msg(format!(
                "{label}: subject mask for image {name} is {}x{} but the image is {iw}x{ih}; \
                 regenerate the dataset's subject masks",
                mask.width, mask.height
            )));
        }
        if mask.is_empty() {
            return Err(crate::Error::Msg(format!(
                "{label}: subject mask for image {name} is empty (no subject was found); fix or \
                 replace the mask, or remove the image"
            )));
        }
        Ok(Self {
            name,
            image_dims: (iw, ih),
            mask,
            cfg: *cfg,
        })
    }

    /// `None` when masked loss is off (`cfg` is `None`; no file is read), else [`load`](Self::load).
    pub fn load_if_enabled(
        label: &str,
        item: &TrainingItem,
        cfg: Option<&SubjectMaskLoss>,
    ) -> crate::Result<Option<Self>> {
        cfg.map(|cfg| Self::load(label, item, cfg)).transpose()
    }

    /// The row-major `[grid_h, grid_w]` loss-weight map for one cached latent: the mask cropped
    /// with `crop_of(image_w, image_h)`, area-averaged onto the latent grid, mapped through the
    /// weights. Refuses (when `background_weight == 0`) a crop the subject lies entirely outside.
    pub fn latent_weights(
        &self,
        label: &str,
        crop_of: impl FnOnce(u32, u32) -> CropBox,
        grid_w: usize,
        grid_h: usize,
    ) -> crate::Result<Vec<f32>> {
        let (iw, ih) = self.image_dims;
        let m = self.mask.area_resample(crop_of(iw, ih), grid_w, grid_h);
        if self.cfg.background_weight == 0.0 && m.iter().all(|&v| v == 0.0) {
            return Err(crate::Error::Msg(format!(
                "{label}: the subject in image {}'s mask lies entirely outside the training crop, \
                 so with background_weight 0 the image would contribute no loss",
                self.name
            )));
        }
        Ok(m.into_iter().map(|v| self.cfg.weight(v)).collect())
    }
}

impl PreparedSubjectMask {
    /// The mask's raw coverage (no loss weights) of the `crop_of(image_w, image_h)` region,
    /// area-averaged onto an `out_w × out_h` pixel grid, row-major — the pixel-space mask a
    /// decoded-x0 perceptual loss averages over (sc-24832).
    pub fn pixel_coverage(
        &self,
        crop_of: impl FnOnce(u32, u32) -> CropBox,
        out_w: usize,
        out_h: usize,
    ) -> Vec<f32> {
        let (iw, ih) = self.image_dims;
        self.mask.area_resample(crop_of(iw, ih), out_w, out_h)
    }
}

/// The job's subject masks for the **decoded-x0 perceptual losses** (sc-24832: the normal loss
/// restricted to the subject), handed to a trainer's perceptual path once
/// (`PerceptualPath::attach_subject_masks` in either kit) so every reference it builds receives the
/// mask of its cache entry's item, cropped with the same box as the image (the trainer's own crop
/// rule, exactly as its subject-masked-loss weights use) and area-averaged onto that entry's decoded
/// pixel grid.
#[derive(Clone, Debug)]
pub struct PerceptualSubjectMasks {
    /// Per item: its mask and the trainer's crop rule for that item's image.
    masks: Vec<(PreparedSubjectMask, fn(u32, u32) -> CropBox)>,
    entries_per_item: usize,
}

impl PerceptualSubjectMasks {
    /// Whether `cfg` turns on a perceptual loss that reads subject masks.
    pub fn needed(cfg: &TrainingConfig) -> bool {
        cfg.body_losses.normal.is_enabled() && cfg.body_losses.normal_restrict_to_subject
    }

    /// `None` when no perceptual loss needs masks (no file is read); else every item's mask,
    /// decoded and checked once (a missing, mis-sized or empty mask is refused naming the image,
    /// as for subject-masked loss). The trainer's cache is item-major with `entries_per_item`
    /// entries (resolution buckets) per item — a reference key `entry` belongs to item
    /// `entry / entries_per_item` — and `crop_of` is the trainer's image crop rule (one for every
    /// item; [`load_with`](Self::load_with) picks it per item).
    pub fn load(
        label: &str,
        items: &[TrainingItem],
        cfg: &TrainingConfig,
        entries_per_item: usize,
        crop_of: fn(u32, u32) -> CropBox,
    ) -> crate::Result<Option<Self>> {
        Self::load_with(label, items, cfg, entries_per_item, |_| crop_of)
    }

    /// [`load`](Self::load) with the crop rule chosen per item (e.g. an edit pair's whole frame vs a
    /// captioned item's centre square).
    pub fn load_with(
        label: &str,
        items: &[TrainingItem],
        cfg: &TrainingConfig,
        entries_per_item: usize,
        crop_for: impl Fn(&TrainingItem) -> fn(u32, u32) -> CropBox,
    ) -> crate::Result<Option<Self>> {
        if !Self::needed(cfg) {
            return Ok(None);
        }
        // The mask's loss weights are unused here (raw coverage only).
        let identity = SubjectMaskLoss {
            background_weight: 1.0,
            subject_weight: 1.0,
        };
        let masks = items
            .iter()
            .map(|item| {
                Ok((
                    PreparedSubjectMask::load(label, item, &identity)?,
                    crop_for(item),
                ))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Some(Self {
            masks,
            entries_per_item: entries_per_item.max(1),
        }))
    }

    /// Build from already-prepared masks (tests and trainers that hold them).
    pub fn from_prepared(
        masks: Vec<PreparedSubjectMask>,
        entries_per_item: usize,
        crop_of: fn(u32, u32) -> CropBox,
    ) -> Self {
        Self {
            masks: masks.into_iter().map(|m| (m, crop_of)).collect(),
            entries_per_item: entries_per_item.max(1),
        }
    }

    /// Reference key `entry`'s mask on an `out_w × out_h` pixel grid (row-major). An entry past
    /// the job's items is an error (a trainer keying references outside its cache).
    pub fn pixel_mask(&self, entry: usize, out_w: usize, out_h: usize) -> crate::Result<Vec<f32>> {
        let item = entry / self.entries_per_item;
        let (mask, crop_of) = self.masks.get(item).ok_or_else(|| {
            crate::Error::Msg(format!(
                "perceptual subject masks: reference {entry} is item {item}, but the job has {} \
                 item masks",
                self.masks.len()
            ))
        })?;
        Ok(mask.pixel_coverage(*crop_of, out_w, out_h))
    }
}

#[cfg(test)]
mod tests;
