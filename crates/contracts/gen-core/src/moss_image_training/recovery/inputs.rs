//! One immutable run's dataset identities, header facts and bounded authenticated decode buffers.
use crate::train::subject_mask::{PreparedSubjectMask, SubjectMask, SubjectMaskLoss};
use crate::{Error, PinnedWeightsFile, Result, TrainingRequest};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

#[derive(Clone, Debug)]
struct FilePin {
    token: PinnedWeightsFile,
    bytes: u64,
    dimensions: (u32, u32),
}
fn codec_allowance() -> Result<u64> {
    image::Limits::default().max_alloc.ok_or_else(|| {
        Error::Unsupported("image library exposes no bounded decoder allocation allowance".into())
    })
}
// PNG/JPEG are the image formats explicitly enabled by the workspace. Header preflight never
// constructs a decoder, which could otherwise inflate ancillary metadata before admission.
fn endian16(bytes: [u8; 2], little: bool) -> u16 {
    if little {
        u16::from_le_bytes(bytes)
    } else {
        u16::from_be_bytes(bytes)
    }
}
fn endian32(bytes: [u8; 4], little: bool) -> u32 {
    if little {
        u32::from_le_bytes(bytes)
    } else {
        u32::from_be_bytes(bytes)
    }
}
fn endian64(bytes: [u8; 8], little: bool) -> u64 {
    if little {
        u64::from_le_bytes(bytes)
    } else {
        u64::from_be_bytes(bytes)
    }
}
fn tiff_dimensions(file: &mut File, little: bool) -> Result<(u32, u32)> {
    file.seek(SeekFrom::Start(2))?;
    let mut magic = [0; 2];
    file.read_exact(&mut magic)?;
    let big = match endian16(magic, little) {
        42 => false,
        43 => true,
        _ => return Err(Error::Unsupported("invalid TIFF header".into())),
    };
    let offset = if big {
        let mut size = [0; 4];
        file.read_exact(&mut size)?;
        if endian16([size[0], size[1]], little) != 8 || size[2..] != [0, 0] {
            return Err(Error::Unsupported(
                "unsupported BigTIFF offset shape".into(),
            ));
        }
        let mut b = [0; 8];
        file.read_exact(&mut b)?;
        endian64(b, little)
    } else {
        let mut b = [0; 4];
        file.read_exact(&mut b)?;
        u64::from(endian32(b, little))
    };
    file.seek(SeekFrom::Start(offset))?;
    let count = if big {
        let mut b = [0; 8];
        file.read_exact(&mut b)?;
        endian64(b, little)
    } else {
        let mut b = [0; 2];
        file.read_exact(&mut b)?;
        u64::from(endian16(b, little))
    };
    let start = file.stream_position()?;
    let entry = if big { 20u64 } else { 12 };
    let end = count
        .checked_mul(entry)
        .and_then(|n| n.checked_add(start))
        .ok_or_else(|| Error::Unsupported("TIFF header table overflow".into()))?;
    if end > file.metadata()?.len() {
        return Err(Error::Unsupported("truncated TIFF header table".into()));
    }
    let mut width = None;
    let mut height = None;
    for index in 0..count {
        file.seek(SeekFrom::Start(start + index * entry))?;
        let mut bytes = [0u8; 20];
        file.read_exact(&mut bytes[..entry as usize])?;
        let tag = endian16([bytes[0], bytes[1]], little);
        if tag != 256 && tag != 257 {
            continue;
        }
        let ty = endian16([bytes[2], bytes[3]], little);
        let (population, inline) = if big {
            (
                endian64(
                    bytes[4..12]
                        .try_into()
                        .map_err(|_| Error::Msg("TIFF count".into()))?,
                    little,
                ),
                12usize,
            )
        } else {
            (
                u64::from(endian32(
                    bytes[4..8]
                        .try_into()
                        .map_err(|_| Error::Msg("TIFF count".into()))?,
                    little,
                )),
                8usize,
            )
        };
        if population == 0 {
            return Err(Error::Unsupported("empty TIFF dimension tag".into()));
        }
        let size = match ty {
            3 => 2,
            4 => 4,
            16 => 8,
            _ => {
                return Err(Error::Unsupported(
                    "unsupported TIFF dimension representation".into(),
                ))
            }
        };
        let mut value = [0u8; 8];
        if population
            .checked_mul(size as u64)
            .ok_or_else(|| Error::Unsupported("TIFF value overflow".into()))?
            > (if big { 8 } else { 4 })
        {
            let offset = if big {
                endian64(
                    bytes[inline..inline + 8]
                        .try_into()
                        .map_err(|_| Error::Msg("TIFF offset".into()))?,
                    little,
                )
            } else {
                u64::from(endian32(
                    bytes[inline..inline + 4]
                        .try_into()
                        .map_err(|_| Error::Msg("TIFF offset".into()))?,
                    little,
                ))
            };
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut value[..size])?;
        } else {
            value[..size].copy_from_slice(&bytes[inline..inline + size]);
        }
        let value = match size {
            2 => u64::from(endian16([value[0], value[1]], little)),
            4 => u64::from(endian32(
                value[..4]
                    .try_into()
                    .map_err(|_| Error::Msg("TIFF dimension".into()))?,
                little,
            )),
            _ => endian64(value, little),
        };
        let value = u32::try_from(value).map_err(|_| {
            Error::Unsupported("TIFF geometry exceeds native image coordinates".into())
        })?;
        if tag == 256 {
            width = Some(value)
        } else {
            height = Some(value)
        }
        if let (Some(width), Some(height)) = (width, height) {
            return Ok((width, height));
        }
    }
    Err(Error::Unsupported("TIFF frame geometry missing".into()))
}
fn other_dimensions(file: &mut File, signature: &[u8]) -> Result<(u32, u32)> {
    if signature.starts_with(b"GIF87a") || signature.starts_with(b"GIF89a") {
        return Ok((
            u32::from(u16::from_le_bytes([signature[6], signature[7]])),
            u32::from(u16::from_le_bytes([signature[8], signature[9]])),
        ));
    }
    if signature.starts_with(b"II") || signature.starts_with(b"MM") {
        return tiff_dimensions(file, signature.starts_with(b"II"));
    }
    if signature.starts_with(b"BM") {
        file.seek(SeekFrom::Start(14))?;
        let mut size = [0; 4];
        file.read_exact(&mut size)?;
        let size = u32::from_le_bytes(size);
        if size == 12 {
            let mut b = [0; 4];
            file.read_exact(&mut b)?;
            return Ok((
                u32::from(u16::from_le_bytes([b[0], b[1]])),
                u32::from(u16::from_le_bytes([b[2], b[3]])),
            ));
        }
        if size >= 40 {
            let mut b = [0; 8];
            file.read_exact(&mut b)?;
            let w = i32::from_le_bytes(
                b[..4]
                    .try_into()
                    .map_err(|_| Error::Msg("BMP width".into()))?,
            );
            let h = i32::from_le_bytes(
                b[4..]
                    .try_into()
                    .map_err(|_| Error::Msg("BMP height".into()))?,
            );
            return Ok((
                u32::try_from(w).map_err(|_| Error::Unsupported("negative BMP width".into()))?,
                h.unsigned_abs(),
            ));
        }
        return Err(Error::Unsupported("unsupported BMP header".into()));
    }
    if signature.starts_with(b"RIFF") && signature.get(8..12) == Some(b"WEBP".as_slice()) {
        file.seek(SeekFrom::Start(12))?;
        loop {
            let mut chunk = [0; 8];
            file.read_exact(&mut chunk)?;
            let size = u32::from_le_bytes(
                chunk[4..]
                    .try_into()
                    .map_err(|_| Error::Msg("WebP chunk".into()))?,
            );
            if &chunk[..4] == b"VP8X" {
                let mut b = [0; 10];
                file.read_exact(&mut b)?;
                return Ok((
                    1 + u32::from_le_bytes([b[4], b[5], b[6], 0]),
                    1 + u32::from_le_bytes([b[7], b[8], b[9], 0]),
                ));
            }
            if &chunk[..4] == b"VP8L" {
                let mut b = [0; 5];
                file.read_exact(&mut b)?;
                if b[0] != 0x2f {
                    return Err(Error::Unsupported("invalid lossless WebP header".into()));
                }
                let bits = u32::from_le_bytes(
                    b[1..]
                        .try_into()
                        .map_err(|_| Error::Msg("WebP dimensions".into()))?,
                );
                return Ok(((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1));
            }
            if &chunk[..4] == b"VP8 " {
                let mut b = [0; 10];
                file.read_exact(&mut b)?;
                if b[3..6] != [0x9d, 0x01, 0x2a] {
                    return Err(Error::Unsupported("invalid lossy WebP frame header".into()));
                }
                return Ok((
                    u32::from(u16::from_le_bytes([b[6], b[7]]) & 0x3fff),
                    u32::from(u16::from_le_bytes([b[8], b[9]]) & 0x3fff),
                ));
            }
            file.seek(SeekFrom::Current(i64::from(size) + i64::from(size % 2)))?;
        }
    }
    Err(Error::Unsupported(
        "unsupported training image/mask header".into(),
    ))
}
fn dimensions(file: &mut File) -> Result<(u32, u32)> {
    let mut signature = [0u8; 24];
    file.read_exact(&mut signature)?;
    file.seek(SeekFrom::Start(0))?;
    if &signature[..8] == b"\x89PNG\r\n\x1a\n" && &signature[12..16] == b"IHDR" {
        return Ok((
            u32::from_be_bytes(
                signature[16..20]
                    .try_into()
                    .map_err(|_| Error::Msg("PNG header".into()))?,
            ),
            u32::from_be_bytes(
                signature[20..24]
                    .try_into()
                    .map_err(|_| Error::Msg("PNG header".into()))?,
            ),
        ));
    }
    if signature[..2] == [0xff, 0xd8] {
        file.seek(SeekFrom::Start(2))?;
        loop {
            let mut byte = [0u8; 1];
            file.read_exact(&mut byte)?;
            if byte[0] != 0xff {
                return Err(Error::Unsupported("malformed JPEG marker".into()));
            }
            while byte[0] == 0xff {
                file.read_exact(&mut byte)?;
            }
            let marker = byte[0];
            if marker == 0xd9 || marker == 0xda {
                return Err(Error::Unsupported("JPEG has no frame geometry".into()));
            }
            if marker == 0xd8 || marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
                continue;
            }
            let mut bytes = [0u8; 2];
            file.read_exact(&mut bytes)?;
            let length = u16::from_be_bytes(bytes);
            if length < 2 {
                return Err(Error::Unsupported("invalid JPEG marker length".into()));
            }
            if (0xc0..=0xcf).contains(&marker) && ![0xc4, 0xc8, 0xcc].contains(&marker) {
                if length < 7 {
                    return Err(Error::Unsupported("invalid JPEG frame".into()));
                }
                let mut frame = [0u8; 5];
                file.read_exact(&mut frame)?;
                return Ok((
                    u32::from(u16::from_be_bytes([frame[3], frame[4]])),
                    u32::from(u16::from_be_bytes([frame[1], frame[2]])),
                ));
            }
            file.seek(SeekFrom::Current(i64::from(length) - 2))?;
        }
    }
    other_dimensions(file, &signature)
}
fn read_payload(reader: impl Read, size: u64, expected: &[u8; 32]) -> Result<Vec<u8>> {
    let limit = size
        .checked_add(1)
        .ok_or_else(|| Error::Unsupported("dataset file size overflow".into()))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(
            usize::try_from(limit).map_err(|_| {
                Error::Unsupported("dataset file exceeds host address space".into())
            })?,
        )
        .map_err(|e| Error::Unsupported(format!("cannot reserve admitted dataset buffer: {e}")))?;
    reader.take(limit).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != size || Sha256::digest(&bytes).as_slice() != expected {
        return Err(Error::Unsupported(
            "actual dataset decode bytes differ from pinned identity".into(),
        ));
    }
    Ok(bytes)
}
impl FilePin {
    fn capture(path: &Path, budget: u64) -> Result<Self> {
        if std::fs::metadata(path)?.len() > budget {
            return Err(Error::Unsupported(
                "encoded dataset input exceeds available host allocation budget".into(),
            ));
        }
        let token = PinnedWeightsFile::pin(path)?;
        let (bytes, dimensions) = token.read_unchanged(|path| {
            let mut file = File::open(path)?;
            let bytes = file.metadata()?.len();
            if bytes > budget {
                return Err(Error::Unsupported(
                    "encoded dataset input grew beyond host budget".into(),
                ));
            }
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 65536];
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                hash.update(&buffer[..count]);
            }
            if hash.finalize().as_slice() != token.content_sha256() {
                return Err(Error::Unsupported(
                    "dataset file changed during header capture".into(),
                ));
            }
            file.seek(SeekFrom::Start(0))?;
            let dimensions = dimensions(&mut file)?;
            if bytes == 0 || dimensions.0 == 0 || dimensions.1 == 0 {
                return Err(Error::Unsupported("dataset image/mask is empty".into()));
            }
            Ok((bytes, dimensions))
        })?;
        Ok(Self {
            token,
            bytes,
            dimensions,
        })
    }
    fn read(&self) -> Result<Vec<u8>> {
        self.token.read_unchanged(|path| {
            let file = File::open(path)?;
            if file.metadata()?.len() != self.bytes {
                return Err(Error::Unsupported(
                    "dataset file size changed before decode".into(),
                ));
            }
            read_payload(file, self.bytes, self.token.content_sha256())
        })
    }
    fn decode(&self, budget: u64) -> Result<image::DynamicImage> {
        let pixels = u64::from(self.dimensions.0)
            .checked_mul(u64::from(self.dimensions.1))
            .ok_or_else(|| Error::Unsupported("decoder geometry overflow".into()))?;
        let allowance = codec_allowance()?;
        let maximum = pixels
            .checked_mul(32)
            .and_then(|n| n.checked_add(allowance))
            .ok_or_else(|| Error::Unsupported("decoder allocation bound overflow".into()))?;
        let total = maximum
            .checked_add(self.bytes)
            .ok_or_else(|| Error::Unsupported("encoded/decoded allocation overflow".into()))?;
        if total > budget {
            return Err(Error::Unsupported(
                "encoded/decoded image or mask exceeds admitted host allocation budget".into(),
            ));
        }
        let bytes = self.read()?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(self.dimensions.0);
        limits.max_image_height = Some(self.dimensions.1);
        limits.max_alloc = Some(maximum);
        let mut reader =
            image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
        reader.limits(limits);
        reader
            .decode()
            .map_err(|e| Error::Msg(format!("bounded dataset decode refused: {e}")))
    }
}
#[derive(Clone, Debug)]
pub(super) struct Inputs {
    images: Vec<FilePin>,
    masks: Vec<Option<FilePin>>,
}
impl Inputs {
    pub(super) fn capture(req: &TrainingRequest, budget: u64) -> Result<Self> {
        let mut images = Vec::new();
        let mut masks = Vec::new();
        for item in &req.items {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let image = FilePin::capture(&item.image_path, budget)?;
            let mask = if req.config.subject_mask_loss.is_some() {
                Some(FilePin::capture(
                    item.subject_mask_path.as_deref().ok_or_else(|| {
                        Error::Unsupported("configured subject mask missing".into())
                    })?,
                    budget,
                )?)
            } else {
                None
            };
            if mask
                .as_ref()
                .is_some_and(|m| m.dimensions != image.dimensions)
            {
                return Err(Error::Unsupported(
                    "dataset mask and image geometry differ".into(),
                ));
            }
            images.push(image);
            masks.push(mask);
        }
        Ok(Self { images, masks })
    }
    pub(super) fn image_digest(&self, item: usize) -> Result<[u8; 32]> {
        self.images
            .get(item)
            .map(|p| *p.token.content_sha256())
            .ok_or_else(|| Error::Msg("image digest index outside snapshot".into()))
    }
    pub(super) fn mask_digest(&self, item: usize) -> Result<Option<[u8; 32]>> {
        self.masks
            .get(item)
            .map(|p| p.as_ref().map(|p| *p.token.content_sha256()))
            .ok_or_else(|| Error::Msg("mask digest index outside snapshot".into()))
    }
    pub(super) fn dimensions(&self) -> Vec<(u32, u32)> {
        self.images.iter().map(|p| p.dimensions).collect()
    }
    pub(super) fn verify(&self) -> Result<()> {
        for token in self.images.iter().chain(self.masks.iter().flatten()) {
            token.token.verify_unchanged()?;
        }
        Ok(())
    }
    pub(super) fn rgb(&self, item: usize, budget: u64) -> Result<image::RgbImage> {
        self.images
            .get(item)
            .ok_or_else(|| Error::Msg("dataset image index is outside pinned snapshot".into()))?
            .decode(budget)
            .map(|image| image.to_rgb8())
    }
    pub(super) fn subject_mask(
        &self,
        req: &TrainingRequest,
        item: usize,
        cfg: Option<&SubjectMaskLoss>,
        budget: u64,
    ) -> Result<Option<PreparedSubjectMask>> {
        let Some(cfg) = cfg else { return Ok(None) };
        let image = self
            .images
            .get(item)
            .ok_or_else(|| Error::Msg("dataset mask item outside snapshot".into()))?;
        let mask = self
            .masks
            .get(item)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::Unsupported("configured pinned subject mask missing".into()))?;
        let gray = mask.decode(budget)?.to_luma8();
        let subject = SubjectMask {
            width: gray.width(),
            height: gray.height(),
            values: gray
                .into_raw()
                .into_iter()
                .map(|v| f32::from(v) / 255.)
                .collect(),
        };
        PreparedSubjectMask::from_decoded(
            "MOSS",
            &req.items[item].image_path,
            image.dimensions,
            subject,
            cfg,
        )
        .map(Some)
    }
    pub(super) fn decode_buffer_bound(&self) -> Result<u64> {
        let allowance = codec_allowance()?;
        let mut maximum = 0;
        for (image, mask) in self.images.iter().zip(&self.masks) {
            let pixels = u64::from(image.dimensions.0)
                .checked_mul(u64::from(image.dimensions.1))
                .ok_or_else(|| Error::Unsupported("dataset pixel population overflow".into()))?;
            let image_buffers = pixels
                .checked_mul(38)
                .and_then(|n| n.checked_add(image.bytes))
                .and_then(|n| n.checked_add(allowance))
                .ok_or_else(|| Error::Unsupported("dataset decode buffer overflow".into()))?;
            let mask_buffers = if let Some(mask) = mask {
                pixels
                    .checked_mul(38)
                    .and_then(|n| n.checked_add(mask.bytes))
                    .and_then(|n| n.checked_add(allowance))
                    .ok_or_else(|| Error::Unsupported("dataset mask buffer overflow".into()))?
            } else {
                0
            };
            maximum =
                maximum.max(image_buffers.checked_add(mask_buffers).ok_or_else(|| {
                    Error::Unsupported("dataset combined buffers overflow".into())
                })?);
        }
        Ok(maximum)
    }
    pub(super) fn fingerprint(&self, req: &TrainingRequest) -> Result<String> {
        if self.images.len() != req.items.len() {
            return Err(Error::Msg("dataset snapshot population differs".into()));
        }
        let mut hash = Sha256::new();
        hash.update(b"moss-pinned-dataset-v2");
        for (i, item) in req.items.iter().enumerate() {
            hash.update((i as u64).to_le_bytes());
            // Prepared dataset paths may change on worker retry; content, captions, order and
            // masks are semantic. The live run still binds every literal path through its pin.
            let caption = item.caption.as_bytes();
            hash.update((caption.len() as u64).to_le_bytes());
            hash.update(caption);
            hash.update(self.images[i].token.content_sha256());
            if let Some(mask) = &self.masks[i] {
                hash.update([1]);
                hash.update(mask.token.content_sha256());
            } else {
                hash.update([0]);
            }
        }
        Ok(format!("{:x}", hash.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(root: &Path, masked: bool) -> TrainingRequest {
        let image = root.join("image.png");
        image::RgbImage::from_pixel(64, 32, image::Rgb([11, 27, 53]))
            .save(&image)
            .unwrap();
        let mut item = crate::TrainingItem::captioned(image, "actual caption".into());
        if masked {
            let mask = root.join("mask.png");
            image::GrayImage::from_pixel(64, 32, image::Luma([255]))
                .save(&mask)
                .unwrap();
            item.subject_mask_path = Some(mask);
        }
        TrainingRequest {
            items: vec![item],
            config: crate::TrainingConfig {
                resolution: 32,
                subject_mask_loss: masked.then_some(crate::SubjectMaskLoss {
                    background_weight: 0.25,
                    subject_weight: 1.,
                }),
                ..Default::default()
            },
            output_dir: root.into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: crate::CancelFlag::new(),
        }
    }
    #[test]
    fn captured_image_and_mask_refuse_actual_tamper_before_decode() {
        let root = tempfile::tempdir().unwrap();
        let req = request(root.path(), true);
        let input = Inputs::capture(&req, 2 * 1024 * 1024 * 1024).unwrap();
        input.rgb(0, 2 * 1024 * 1024 * 1024).unwrap();
        input
            .subject_mask(
                &req,
                0,
                req.config.subject_mask_loss.as_ref(),
                2 * 1024 * 1024 * 1024,
            )
            .unwrap()
            .unwrap();
        std::fs::write(&req.items[0].image_path, b"changed after capture").unwrap();
        assert!(input.rgb(0, 2 * 1024 * 1024 * 1024).is_err());
        std::fs::write(
            req.items[0].subject_mask_path.as_ref().unwrap(),
            b"changed mask",
        )
        .unwrap();
        assert!(input
            .subject_mask(
                &req,
                0,
                req.config.subject_mask_loss.as_ref(),
                2 * 1024 * 1024 * 1024
            )
            .is_err());
    }
    #[test]
    fn same_size_change_growth_and_truncation_during_owned_read_are_refused() {
        struct Mutating {
            file: File,
            path: std::path::PathBuf,
            kind: u8,
            changed: bool,
        }
        impl Read for Mutating {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if !self.changed {
                    let n = self.file.read(&mut buffer[..1])?;
                    self.changed = true;
                    match self.kind {
                        0 => {
                            let mut f = std::fs::OpenOptions::new().write(true).open(&self.path)?;
                            f.seek(SeekFrom::Start(1))?;
                            std::io::Write::write_all(&mut f, b"X")?;
                        }
                        1 => {
                            let mut f =
                                std::fs::OpenOptions::new().append(true).open(&self.path)?;
                            std::io::Write::write_all(&mut f, b"growth")?;
                        }
                        _ => {
                            std::fs::OpenOptions::new()
                                .write(true)
                                .open(&self.path)?
                                .set_len(1)?;
                        }
                    }
                    return Ok(n);
                }
                self.file.read(buffer)
            }
        }
        for kind in 0..3 {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("payload");
            let original = b"sealed-owned-payload";
            std::fs::write(&path, original).unwrap();
            let digest: [u8; 32] = Sha256::digest(original).into();
            let reader = Mutating {
                file: File::open(&path).unwrap(),
                path,
                kind,
                changed: false,
            };
            assert!(read_payload(reader, original.len() as u64, &digest).is_err());
        }
    }
    #[test]
    fn encoded_size_decoded_bomb_and_mask_geometry_refuse_before_allocation() {
        let root = tempfile::tempdir().unwrap();
        let req = request(root.path(), false);
        let encoded = std::fs::metadata(&req.items[0].image_path).unwrap().len();
        assert!(FilePin::capture(&req.items[0].image_path, encoded - 1).is_err());
        let input = Inputs::capture(&req, 2 * 1024 * 1024 * 1024).unwrap();
        assert!(
            matches!(input.rgb(0,1),Err(Error::Unsupported(message)) if message.contains("allocation budget"))
        );
        let mut bytes = std::fs::read(&req.items[0].image_path).unwrap();
        bytes[16..20].copy_from_slice(&u32::MAX.to_be_bytes());
        bytes[20..24].copy_from_slice(&u32::MAX.to_be_bytes());
        std::fs::write(&req.items[0].image_path, &bytes).unwrap();
        let input = Inputs::capture(&req, 2 * 1024 * 1024 * 1024).unwrap();
        assert!(input.decode_buffer_bound().is_err());
        assert!(input.rgb(0, 2 * 1024 * 1024 * 1024).is_err());
        let req = request(root.path(), true);
        image::GrayImage::from_pixel(32, 64, image::Luma([255]))
            .save(req.items[0].subject_mask_path.as_ref().unwrap())
            .unwrap();
        assert!(Inputs::capture(&req, 2 * 1024 * 1024 * 1024).is_err());
    }
    #[test]
    fn every_existing_product_image_format_has_real_matching_header_and_decode() {
        let root = tempfile::tempdir().unwrap();
        for (extension, format) in [
            ("png", image::ImageFormat::Png),
            ("jpg", image::ImageFormat::Jpeg),
            ("webp", image::ImageFormat::WebP),
            ("bmp", image::ImageFormat::Bmp),
            ("gif", image::ImageFormat::Gif),
            ("tiff", image::ImageFormat::Tiff),
        ] {
            let path = root.path().join(format!("image.{extension}"));
            image::RgbImage::from_fn(64, 32, |x, y| image::Rgb([x as u8, y as u8, (x ^ y) as u8]))
                .save_with_format(&path, format)
                .unwrap();
            let pin = FilePin::capture(&path, 2 * 1024 * 1024 * 1024).unwrap();
            assert_eq!(
                pin.dimensions,
                (64, 32),
                "{extension} actual header geometry"
            );
            let decoded = pin.decode(2 * 1024 * 1024 * 1024).unwrap();
            assert_eq!(
                (decoded.width(), decoded.height()),
                pin.dimensions,
                "{extension} actual bounded decoder geometry"
            );
        }
    }
}
