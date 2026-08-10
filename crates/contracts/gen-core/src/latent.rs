//! Backend-neutral latent-space identity and compatibility contracts.
//!
//! A tensor shape alone is not enough to decide whether a denoiser output may be handed to a
//! decoder. Two spaces can share a channel count while differing in spatial/temporal compression,
//! patch packing, or normalization. These descriptors keep those facts weights-free and make the
//! compatibility decision default-deny when either side is unknown.

/// A latent grid's spatial pixel compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatialCompression {
    pub height: u16,
    pub width: u16,
}

/// Whether the decoder seam receives a native latent grid or a patch-packed grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentPatchLayout {
    Unpacked,
    Packed { patch_height: u8, patch_width: u8 },
}

/// Temporal pixel-to-latent law.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentTemporalLaw {
    /// Still-image space with no temporal compression contract.
    None,
    /// Causal video law `latent_frames = (pixel_frames - 1) / 4 + 1`.
    Causal4x,
}

/// Stable identity for fixed per-channel normalization vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentNormalizationStats {
    /// Human-stable identity for logs and capability artifacts.
    pub identity: &'static str,
    /// Number of entries in each of the mean/std vectors.
    pub channels: u16,
    /// FNV-1a over the exact IEEE-754 bits of both vectors, including vector boundaries.
    pub content_hash: u64,
}

impl LatentNormalizationStats {
    /// Build a stable identity from the exact per-channel vectors.
    pub const fn from_vectors(
        identity: &'static str,
        mean: &'static [f32],
        std: &'static [f32],
    ) -> Self {
        assert!(
            !mean.is_empty(),
            "latent normalization mean must not be empty"
        );
        assert!(
            mean.len() == std.len(),
            "latent normalization vectors must match"
        );
        assert!(mean.len() <= u16::MAX as usize, "too many latent channels");
        Self {
            identity,
            channels: mean.len() as u16,
            content_hash: normalization_vectors_hash(mean, std),
        }
    }
}

/// How the sampler-space latent maps to the VAE's raw latent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentNormalization {
    Identity,
    /// `raw = latent / scale + shift`, stored as exact IEEE-754 bits.
    Affine {
        scale_bits: u32,
        shift_bits: u32,
    },
    /// Fixed per-channel `raw = latent * std + mean` vectors.
    PerChannel(LatentNormalizationStats),
    /// Per-channel values loaded from weights. The layout is known, but without a content hash two
    /// independently loaded instances are deliberately not declared compatible.
    LearnedPerChannel {
        identity: &'static str,
    },
}

impl LatentNormalization {
    pub const fn affine(scale: f32, shift: f32) -> Self {
        Self::Affine {
            scale_bits: scale.to_bits(),
            shift_bits: shift.to_bits(),
        }
    }

    pub fn affine_values(self) -> Option<(f32, f32)> {
        match self {
            Self::Affine {
                scale_bits,
                shift_bits,
            } => Some((f32::from_bits(scale_bits), f32::from_bits(shift_bits))),
            _ => None,
        }
    }

    fn is_compatible_with(self, other: Self) -> bool {
        match (self, other) {
            // The identity names the normalization family, not the actual learned vectors. Without
            // a content hash, accepting two learned instances would turn missing evidence into a
            // compatibility claim.
            (Self::LearnedPerChannel { .. }, _) | (_, Self::LearnedPerChannel { .. }) => false,
            _ => self == other,
        }
    }
}

/// Complete identity of the tensor at a denoiser-to-decoder boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentSpace {
    pub channels: u16,
    pub spatial_compression: SpatialCompression,
    pub patch_layout: LatentPatchLayout,
    pub temporal_law: LatentTemporalLaw,
    pub normalization: LatentNormalization,
}

impl LatentSpace {
    /// Exact, fail-closed compatibility. Learned normalization without a content hash remains
    /// incompatible even when its family identity matches.
    pub fn is_compatible_with(&self, decoder_input: &Self) -> bool {
        self.channels == decoder_input.channels
            && self.spatial_compression == decoder_input.spatial_compression
            && self.patch_layout == decoder_input.patch_layout
            && self.temporal_law == decoder_input.temporal_law
            && self
                .normalization
                .is_compatible_with(decoder_input.normalization)
    }
}

/// Compare an advertised denoiser output to a decoder input. Missing descriptors are incompatible.
pub fn latent_spaces_compatible(
    denoiser_output: Option<&LatentSpace>,
    decoder_input: Option<&LatentSpace>,
) -> bool {
    matches!((denoiser_output, decoder_input), (Some(output), Some(input)) if output.is_compatible_with(input))
}

/// One weights-free alternate-decoder choice exposed through provider introspection.
///
/// The option carries both sides of the contract: the component id a caller must stage in
/// [`crate::LoadSpec::components`] and the exact latent space the decoder consumes. Provider
/// eligibility is deliberately explicit; latent compatibility is necessary but does not claim that
/// a provider has wired the alternate decoder into its render path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecoderOption {
    pub id: &'static str,
    pub label: &'static str,
    pub component_id: &'static str,
    pub input_latent_space: &'static LatentSpace,
    pub eligible_backends: &'static [&'static str],
    pub eligible_provider_ids: &'static [&'static str],
    pub license_component: &'static str,
    pub experimental: bool,
}

/// Stable request value for the Wan 2.1 z16 image-decoder substitution.
pub const WAN_2_1_VAE_DECODER_ID: &str = "wan_2_1_vae";

const WAN_2_1_VAE_ELIGIBLE_PROVIDERS: &[&str] = &[
    "krea_2_turbo",
    "krea_2_raw",
    "krea_2_edit",
    "krea_2_turbo_edit",
    "krea_2_turbo_control",
    "qwen_image",
    "qwen_image_control",
    "qwen_image_edit",
];

/// The complete alternate-decoder registry. Consumers derive menus from this table through
/// [`crate::ModelDescriptor::compatible_decoder_options`]; they must not reproduce provider-id lists.
pub const DECODER_OPTIONS: &[DecoderOption] = &[DecoderOption {
    id: WAN_2_1_VAE_DECODER_ID,
    label: "Wan 2.1 VAE",
    component_id: crate::VAE_COMPONENT,
    input_latent_space: &WAN_Z16_LATENT_SPACE,
    eligible_backends: &["mlx"],
    eligible_provider_ids: WAN_2_1_VAE_ELIGIBLE_PROVIDERS,
    license_component: "wan2_1_t2v_14b_diffusers",
    experimental: true,
}];

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001b3;

const fn hash_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ byte as u64).wrapping_mul(FNV_PRIME)
}

const fn hash_u32(mut hash: u64, value: u32) -> u64 {
    let bytes = value.to_le_bytes();
    let mut i = 0;
    while i < bytes.len() {
        hash = hash_byte(hash, bytes[i]);
        i += 1;
    }
    hash
}

/// Stable content hash over two per-channel vectors. The two lengths are hashed before their
/// respective IEEE-754 payloads so concatenation cannot create an ambiguous identity.
pub const fn normalization_vectors_hash(mean: &[f32], std: &[f32]) -> u64 {
    let mut hash = hash_u32(FNV_OFFSET_BASIS, mean.len() as u32);
    let mut i = 0;
    while i < mean.len() {
        hash = hash_u32(hash, mean[i].to_bits());
        i += 1;
    }
    hash = hash_u32(hash, std.len() as u32);
    i = 0;
    while i < std.len() {
        hash = hash_u32(hash, std[i].to_bits());
        i += 1;
    }
    hash
}

/// Shared Qwen-Image / Krea 2 / Wan-z16 normalization. This is the single numeric definition used
/// by both tensor backends and every provider that reuses this VAE fit.
#[rustfmt::skip]
pub const QWEN_WAN_Z16_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
#[rustfmt::skip]
pub const QWEN_WAN_Z16_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
];

/// Wan2.2 TI2V-5B z48 normalization, shared by the MLX and candle ports.
#[rustfmt::skip]
pub const WAN_Z48_MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.157, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
];
#[rustfmt::skip]
pub const WAN_Z48_STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

pub const QWEN_WAN_Z16_NORMALIZATION: LatentNormalizationStats =
    LatentNormalizationStats::from_vectors(
        "qwen-wan-z16-v1",
        &QWEN_WAN_Z16_MEAN,
        &QWEN_WAN_Z16_STD,
    );
pub const WAN_Z48_NORMALIZATION: LatentNormalizationStats =
    LatentNormalizationStats::from_vectors("wan-z48-v1", &WAN_Z48_MEAN, &WAN_Z48_STD);

pub const QWEN_KREA_Z16_LATENT_SPACE: LatentSpace = LatentSpace {
    channels: 16,
    spatial_compression: SpatialCompression {
        height: 8,
        width: 8,
    },
    patch_layout: LatentPatchLayout::Unpacked,
    temporal_law: LatentTemporalLaw::None,
    normalization: LatentNormalization::PerChannel(QWEN_WAN_Z16_NORMALIZATION),
};

/// Single-frame Wan 2.1 z16 decoder input. A one-frame Wan latent is exactly the Qwen-Image/Krea
/// image latent contract, which is the checked basis for the experimental decoder swap.
pub const WAN_Z16_LATENT_SPACE: LatentSpace = QWEN_KREA_Z16_LATENT_SPACE;

/// Wan z16 denoiser output for a video request. This keeps the causal temporal law explicit without
/// making the single-frame decoder contract falsely incompatible with Qwen-Image/Krea.
pub const WAN_Z16_VIDEO_LATENT_SPACE: LatentSpace = LatentSpace {
    channels: 16,
    spatial_compression: SpatialCompression {
        height: 8,
        width: 8,
    },
    patch_layout: LatentPatchLayout::Unpacked,
    temporal_law: LatentTemporalLaw::Causal4x,
    normalization: LatentNormalization::PerChannel(QWEN_WAN_Z16_NORMALIZATION),
};

pub const WAN_Z48_LATENT_SPACE: LatentSpace = LatentSpace {
    channels: 48,
    spatial_compression: SpatialCompression {
        height: 16,
        width: 16,
    },
    patch_layout: LatentPatchLayout::Unpacked,
    temporal_law: LatentTemporalLaw::Causal4x,
    normalization: LatentNormalization::PerChannel(WAN_Z48_NORMALIZATION),
};

pub const FLUX1_LATENT_SPACE: LatentSpace = LatentSpace {
    channels: 16,
    spatial_compression: SpatialCompression {
        height: 8,
        width: 8,
    },
    patch_layout: LatentPatchLayout::Unpacked,
    temporal_law: LatentTemporalLaw::None,
    normalization: LatentNormalization::affine(0.3611, 0.1159),
};

pub const SD3_LATENT_SPACE: LatentSpace = LatentSpace {
    channels: 16,
    spatial_compression: SpatialCompression {
        height: 8,
        width: 8,
    },
    patch_layout: LatentPatchLayout::Unpacked,
    temporal_law: LatentTemporalLaw::None,
    normalization: LatentNormalization::affine(1.5305, 0.0609),
};

pub const SDXL_LATENT_SPACE: LatentSpace = LatentSpace {
    channels: 4,
    spatial_compression: SpatialCompression {
        height: 8,
        width: 8,
    },
    patch_layout: LatentPatchLayout::Unpacked,
    temporal_law: LatentTemporalLaw::None,
    normalization: LatentNormalization::affine(0.13025, 0.0),
};

pub const FLUX2_PACKED_LATENT_SPACE: LatentSpace = LatentSpace {
    channels: 128,
    spatial_compression: SpatialCompression {
        height: 16,
        width: 16,
    },
    patch_layout: LatentPatchLayout::Packed {
        patch_height: 2,
        patch_width: 2,
    },
    temporal_law: LatentTemporalLaw::None,
    normalization: LatentNormalization::LearnedPerChannel {
        identity: "flux2-batchnorm-v1",
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_z16_vectors_have_one_stable_identity() {
        assert_eq!(QWEN_WAN_Z16_NORMALIZATION.channels, 16);
        assert_eq!(
            QWEN_WAN_Z16_NORMALIZATION.content_hash,
            normalization_vectors_hash(&QWEN_WAN_Z16_MEAN, &QWEN_WAN_Z16_STD)
        );
        assert_eq!(
            QWEN_KREA_Z16_LATENT_SPACE, WAN_Z16_LATENT_SPACE,
            "single-frame Wan-z16, Qwen-Image, and Krea share the exact latent space"
        );
        assert_ne!(
            WAN_Z16_VIDEO_LATENT_SPACE.normalization,
            WAN_Z48_LATENT_SPACE.normalization
        );
        assert!(QWEN_KREA_Z16_LATENT_SPACE.is_compatible_with(&QWEN_KREA_Z16_LATENT_SPACE));
        assert!(QWEN_KREA_Z16_LATENT_SPACE.is_compatible_with(&WAN_Z16_LATENT_SPACE));
        assert!(!QWEN_KREA_Z16_LATENT_SPACE.is_compatible_with(&WAN_Z16_VIDEO_LATENT_SPACE));
        assert!(!WAN_Z16_VIDEO_LATENT_SPACE.is_compatible_with(&WAN_Z48_LATENT_SPACE));
    }

    #[test]
    fn packed_unpacked_and_missing_descriptors_fail_closed() {
        let mut packed = QWEN_KREA_Z16_LATENT_SPACE;
        packed.patch_layout = LatentPatchLayout::Packed {
            patch_height: 2,
            patch_width: 2,
        };
        assert!(!QWEN_KREA_Z16_LATENT_SPACE.is_compatible_with(&packed));
        assert!(!latent_spaces_compatible(
            Some(&QWEN_KREA_Z16_LATENT_SPACE),
            None
        ));
        assert!(!latent_spaces_compatible(
            None,
            Some(&QWEN_KREA_Z16_LATENT_SPACE)
        ));
        assert!(!latent_spaces_compatible(None, None));
    }

    #[test]
    fn learned_normalization_without_content_hash_fails_closed() {
        assert!(!FLUX2_PACKED_LATENT_SPACE.is_compatible_with(&FLUX2_PACKED_LATENT_SPACE));
    }

    #[test]
    fn alternate_decoder_registry_is_explicit_experimental_and_z16_only() {
        assert_eq!(DECODER_OPTIONS.len(), 1);
        let option = DECODER_OPTIONS[0];
        assert_eq!(option.id, WAN_2_1_VAE_DECODER_ID);
        assert_eq!(option.component_id, crate::VAE_COMPONENT);
        assert!(
            option.experimental,
            "alternate lane must remain default-off"
        );
        assert_eq!(option.eligible_backends, &["mlx"]);
        assert!(option.eligible_provider_ids.contains(&"qwen_image"));
        assert!(option.eligible_provider_ids.contains(&"krea_2_turbo"));
        assert!(option
            .eligible_provider_ids
            .contains(&"krea_2_turbo_control"));
        assert!(latent_spaces_compatible(
            Some(&QWEN_KREA_Z16_LATENT_SPACE),
            Some(option.input_latent_space)
        ));
        assert!(!latent_spaces_compatible(
            Some(&WAN_Z48_LATENT_SPACE),
            Some(option.input_latent_space)
        ));
    }
}
