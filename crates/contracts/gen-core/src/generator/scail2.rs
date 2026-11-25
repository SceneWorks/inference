//! Registered SCAIL-2 character pairs and the exact one-character Resident carrier.

use super::*;

/// The trained source-position table has IDs 0 through 5, including the primary character.
pub const SCAIL2_MAX_CHARACTERS: usize = 6;

/// One character and its paired color-coded segmentation mask.
#[derive(Clone, Copy, Debug)]
pub struct Scail2CharacterRef<'a> {
    pub image: &'a Image,
    pub mask: &'a Image,
}

/// Registered animation consumes all ordered character pairs. Replacement consumes one.
#[derive(Debug)]
pub struct Scail2CharacterReferences<'a> {
    pub primary: Scail2CharacterRef<'a>,
    pub additional: Vec<Scail2CharacterRef<'a>>,
}

impl<'a> Scail2CharacterReferences<'a> {
    pub fn reference_count(&self) -> u32 {
        1 + self.additional.len() as u32
    }

    pub fn from_request(id: &str, req: &'a GenerationRequest) -> crate::Result<Self> {
        Self::parse(
            id,
            req.video_mode.as_deref() == Some("replacement"),
            &req.conditioning,
        )
    }

    /// Strict product grammar, shared by pixel receipts and the pre-load physical join.
    /// Validates ordering and aligned frame origins without reading image bytes.
    pub(super) fn catalog_carrier(
        conditioning: &'a [Conditioning],
        replacement: bool,
    ) -> crate::Result<(Self, ControlClipRef<'a>)> {
        let Some((
            Conditioning::ControlClip {
                frames,
                mask,
                masking_strength,
                start_frame,
                mode: replacement_mode,
            },
            characters,
        )) = conditioning.split_last()
        else {
            return Err(Error::Unsupported(
                "SCAIL-2 catalog conditioning must end with one ControlClip".into(),
            ));
        };
        if characters.iter().any(|entry| {
            !matches!(
                entry,
                Conditioning::Reference { strength: None, .. } | Conditioning::Mask { .. }
            )
        }) {
            return Err(Error::Unsupported(
                "SCAIL-2 catalog characters require ordered Reference(strength unset), Mask pairs"
                    .into(),
            ));
        }
        let characters = Scail2CharacterReferences::parse("scail2", replacement, characters)?;
        if frames.len() != mask.len() || frames.is_empty() {
            return Err(Error::Unsupported(format!(
                "SCAIL-2 admission requires equal non-empty driving frame/mask counts, got {} and {}",
                frames.len(),
                mask.len()
            )));
        }
        if *start_frame != 0 {
            return Err(Error::Unsupported(format!(
                "SCAIL-2 admission requires a frame-0 driving clip, got {start_frame}"
            )));
        }
        Ok((
            characters,
            ControlClipRef {
                frames,
                mask,
                masking_strength: *masking_strength,
                start_frame: *start_frame,
                mode: *replacement_mode,
            },
        ))
    }
    fn parse(id: &str, replacement: bool, conditioning: &'a [Conditioning]) -> crate::Result<Self> {
        let mut primary = None;
        let mut additional = Vec::new();
        let mut pending = None;
        for (index, entry) in conditioning.iter().enumerate() {
            match entry {
                Conditioning::Reference { image, .. } => {
                    if pending.replace(image).is_some() {
                        return Err(Error::Unsupported(format!(
                            "{id}: Reference without its following Mask before conditioning[{index}]"
                        )));
                    }
                }
                Conditioning::Mask { image: mask } => {
                    let image = pending.take().ok_or_else(|| {
                        Error::Unsupported(format!(
                            "{id}: conditioning[{index}] is a Mask without a preceding Reference"
                        ))
                    })?;
                    let pair = Scail2CharacterRef { image, mask };
                    if primary.is_none() {
                        primary = Some(pair);
                    } else if replacement || additional.len() >= SCAIL2_MAX_CHARACTERS - 1 {
                        return Err(Error::Unsupported(format!(
                            "{id}: too many reference/mask pairs for this mode; no pairs were truncated"
                        )));
                    } else {
                        additional.push(pair);
                    }
                }
                Conditioning::MultiReference { .. } => {
                    return Err(Error::Unsupported(format!(
                        "{id}: MultiReference carries only images and has no paired color-coded segmentation masks; pass ordered Reference, Mask pairs instead"
                    )));
                }
                _ => {}
            }
        }
        if pending.is_some() {
            return Err(Error::Unsupported(format!(
                "{id}: Reference without its following Mask; pass ordered Reference, Mask pairs"
            )));
        }
        let primary = primary.ok_or_else(|| {
            Error::Msg(format!(
                "{id}: a Reference character image is required with its paired color-coded Mask"
            ))
        })?;
        Ok(Self {
            primary,
            additional,
        })
    }
}
/// Exact SCAIL-2 public animation/replacement carrier: one character image, its paired color mask, and one
/// driving clip with a one-to-one frame/mask sequence. The three physical conditioning entries are
/// one conceptual character reference for memory-evidence geometry.
#[derive(Clone, Copy, Debug)]
pub struct Scail2AnimationConditioningRef<'a> {
    pub character: &'a Image,
    pub character_mask: &'a Image,
    pub driving_frames: &'a [Image],
    pub driving_masks: &'a [Image],
}

impl Scail2AnimationConditioningRef<'_> {
    /// The persisted product catalog axis and byte seal for a SCAIL-2 carrier. Native-shape
    /// axes key the memory estimate; the trailing digest binds every ordered source pixel.
    pub fn catalog_receipt(
        backend: &str,
        mode: &str,
        width: u32,
        height: u32,
        conditioning: &[Conditioning],
    ) -> crate::Result<String> {
        use sha2::Digest;
        const RECEIPT_DOMAIN: &str = "scail2-carrier-v1";
        const SEAL_DOMAIN: &str = "scail2-carrier-request-seal-v1";
        fn seal_images(role: &str, images: &[Image], seal: &mut sha2::Sha256) -> String {
            seal.update((role.len() as u64).to_le_bytes());
            seal.update(role.as_bytes());
            let shapes = images
                .iter()
                .enumerate()
                .map(|(ordinal, image)| {
                    seal.update((ordinal as u32).to_le_bytes());
                    seal.update(image.width.to_le_bytes());
                    seal.update(image.height.to_le_bytes());
                    seal.update(&image.pixels);
                    format!("{ordinal}:{}x{}", image.width, image.height)
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{role}-{}:{shapes}", images.len())
        }

        let task = match mode {
            "animate_character" => "animation",
            "replace_person" => "replacement",
            other => {
                return Err(Error::Unsupported(format!(
                    "SCAIL-2 admission has no carrier for mode {other}"
                )))
            }
        };
        let (characters, driving) =
            Scail2CharacterReferences::catalog_carrier(conditioning, task == "replacement")?;
        let ControlClipRef {
            frames,
            mask,
            masking_strength,
            mode: replacement_mode,
            ..
        } = driving;
        let single = characters.additional.is_empty();
        let (receipt_domain, seal_domain) = if single {
            (RECEIPT_DOMAIN, SEAL_DOMAIN)
        } else {
            ("scail2-carrier-v2", "scail2-carrier-request-seal-v2")
        };
        let mut seal = sha2::Sha256::new();
        seal.update(seal_domain.as_bytes());
        seal.update(task.as_bytes());
        seal.update(masking_strength.to_bits().to_le_bytes());
        let mut character_axes = Vec::new();
        if !single {
            character_axes.push(format!("characters-{}", 1 + characters.additional.len()));
        }
        for (index, pair) in std::iter::once(characters.primary)
            .chain(characters.additional)
            .enumerate()
        {
            for (role, image) in [("reference", pair.image), ("reference-mask", pair.mask)] {
                let role = if single {
                    role.to_owned()
                } else {
                    format!("{role}-{index}")
                };
                character_axes.push(seal_images(&role, std::slice::from_ref(image), &mut seal));
            }
        }
        let characters = character_axes.join(":");
        let frames_axis = seal_images("driving-frames", frames, &mut seal);
        let mask_axis = seal_images("driving-mask", mask, &mut seal);
        Ok(format!(
            "{receipt_domain}:backend-{backend}:task-{task}:composite-{width}x{height}:{characters}:{frames_axis}:{mask_axis}:strength-{:08x}:mode-{replacement_mode:?}:frame-0+{seal_domain}-{:x}",
            masking_strength.to_bits(),
            seal.finalize()
        ))
    }

    pub const fn reference_count(&self) -> u32 {
        1
    }

    pub fn identity_shape(&self, mode: &str) -> crate::Result<String> {
        let control = self
            .driving_frames
            .first()
            .ok_or_else(|| Error::Unsupported("scail2 carrier has no driving frames".to_owned()))?;
        Ok(format!(
            "{mode}:reference:{}x{}:control:{}x{}x{}",
            self.character.width,
            self.character.height,
            control.width,
            control.height,
            self.driving_frames.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scail2_catalog_preserves_single_and_seals_all_plural_character_roles() {
        let image = |width, height, tag| Image {
            width,
            height,
            pixels: vec![tag; (width * height * 3) as usize],
        };
        let character = |tag| {
            [
                Conditioning::Reference {
                    image: image(4, 2, tag),
                    strength: None,
                },
                Conditioning::Mask {
                    image: image(4, 2, 0),
                },
            ]
        };
        let mut input = character(5).to_vec();
        input.push(Conditioning::ControlClip {
            frames: vec![image(2, 3, 2), image(3, 2, 3)],
            mask: vec![image(2, 3, 0), image(3, 2, 1)],
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::default(),
        });
        let receipt = |mode, conditioning: &[Conditioning]| {
            Scail2AnimationConditioningRef::catalog_receipt("mlx", mode, 16, 24, conditioning)
        };
        assert_eq!(receipt("animate_character",&input).unwrap(), concat!("scail2-carrier-v1:backend-mlx:task-animation:composite-16x24:reference-1:0:4x2:reference-mask-1:0:4x2:", "driving-frames-2:0:2x3,1:3x2:driving-mask-2:0:2x3,1:3x2:strength-3f800000:mode-FaceOnly:frame-0+", "scail2-carrier-request-seal-v1-3142cc77a779c1fe0125bca88ca18263c0496c043d45cb4d8631d03c91c614cf"));
        let control = input.pop().unwrap();
        for count in 2..=7 {
            input.extend(character(count));
            input.push(control.clone());
            if count == 7 {
                assert!(receipt("animate_character", &input).is_err());
                break;
            }
            let baseline = receipt("animate_character", &input).unwrap();
            assert!(baseline.contains(&format!("characters-{count}:")));
            assert!(baseline.contains(&format!("reference-mask-{}-1:0:4x2", count - 1)));
            assert!(baseline.starts_with("scail2-carrier-v2:"));
            assert!(receipt("replace_person", &input).is_err());
            let mut crossed = input.clone();
            crossed.swap(0, 2);
            assert_ne!(baseline, receipt("animate_character", &crossed).unwrap());
            crossed.swap(0, 1);
            assert!(receipt("animate_character", &crossed).is_err());
            input.pop();
        }
    }
}
