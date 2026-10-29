//! Wan I2V routes, public modes, backends and adapter receipts.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WanI2vRoute {
    Ti2v5b,
    /// Dual-expert A14B **text**-to-video (`wan2_2_t2v_14b`).
    ///
    /// It is a member of this authority for its *physical* identity, which is the I2V-A14B's
    /// exactly: one snapshot layout, one inventory, one repository policy family, one adapter
    /// multiplicity rule. Its *request* surface is the one thing that differs — it carries no
    /// image conditioning at all — so it admits [`WanPublicMode::TextToVideo`] and refuses every
    /// image-conditioned carrier (`accepts_first_last_rate`, and the `image_to_video`
    /// arms of `request_mode` / `contract_for_mode_key` / `validate_context`, all of which
    /// exclude it by name).
    T2v14b,
    I2v14b,
    Vace,
    VaceFun,
}

impl WanI2vRoute {
    /// Every route this authority seals, in a fixed order.
    ///
    /// Sweeps derive their cell count from this instead of freezing a literal, so adding a route
    /// without giving it a repository policy, an inventory and its own calibration cell is a
    /// compile error or a red test rather than a silently unswept row.
    pub const ALL: [Self; 5] = [
        Self::Ti2v5b,
        Self::T2v14b,
        Self::I2v14b,
        Self::Vace,
        Self::VaceFun,
    ];

    pub fn for_provider(provider_id: &str) -> crate::Result<Self> {
        match provider_id {
            "wan2_2_ti2v_5b" => Ok(Self::Ti2v5b),
            "wan2_2_t2v_14b" => Ok(Self::T2v14b),
            "wan2_2_i2v_14b" => Ok(Self::I2v14b),
            "wan_vace" => Ok(Self::Vace),
            "wan2_2_vace_fun_14b" => Ok(Self::VaceFun),
            _ => Err(crate::Error::Unsupported(format!(
                "{provider_id}: not a Wan I2V memory route"
            ))),
        }
    }

    pub fn provider_id(self) -> &'static str {
        match self {
            Self::Ti2v5b => "wan2_2_ti2v_5b",
            Self::T2v14b => "wan2_2_t2v_14b",
            Self::I2v14b => "wan2_2_i2v_14b",
            Self::Vace => "wan_vace",
            Self::VaceFun => "wan2_2_vace_fun_14b",
        }
    }

    /// Whether this route's transformer is a dual-expert MoE pair (`transformer` +
    /// `transformer_2`, or `high_noise_model` + `low_noise_model`).
    pub(super) fn dual_expert(self) -> bool {
        matches!(self, Self::T2v14b | Self::I2v14b | Self::VaceFun)
    }

    pub fn public_geometries(self) -> &'static [(u32, u32)] {
        match self {
            Self::Ti2v5b => &[(832, 480), (1280, 704), (704, 1280)],
            // `builtin.models.jsonc` gives `wan_2_2_t2v_14b` and `wan_2_2_i2v_14b` the same
            // `resolutions` list; they are the same A14B geometry cap (`maxPixels` 921_600).
            Self::T2v14b | Self::I2v14b => &[(832, 480), (480, 832), (1280, 720), (720, 1280)],
            Self::Vace | Self::VaceFun => &[(832, 480), (480, 832), (1280, 720), (720, 1280)],
        }
    }

    /// The public `(fps, frames)` rate menu this route admits.
    ///
    /// `pub` since sc-22736 so a measurement harness can validate a planned video geometry against
    /// the ENGINE's own menu instead of restating the manifest's `fps` / `durations` lists in a
    /// second place — the two would then be free to drift, and the drift would only ever surface as
    /// a refused capture on a host that already holds the weights.
    pub fn accepts_rate(self, fps: u32, frames: u32) -> bool {
        match self {
            Self::Ti2v5b => match fps {
                16 => [65, 81, 97, 113, 129].contains(&frames),
                24 => [97, 121, 145, 169, 193].contains(&frames),
                _ => false,
            },
            // Same `fps: [16]` / `durations: [3, 4, 5]` manifest menu as the I2V-A14B.
            Self::T2v14b | Self::I2v14b => fps == 16 && [45, 61, 77].contains(&frames),
            Self::Vace | Self::VaceFun => fps == 16 && [45, 61, 77].contains(&frames),
        }
    }

    /// The first/last-frame rate menu. Only Ti2V-5B has one.
    ///
    /// `T2v14b` is `false` for a structural reason, not an unimplemented one: a text-to-video
    /// route has no frame to pin at either boundary, so a first/last carrier is not a rate this
    /// route could ever accept.
    pub(super) fn accepts_first_last_rate(self, fps: u32, frames: u32) -> bool {
        self == Self::Ti2v5b
            && match fps {
                16 => [61, 77, 93, 109, 125].contains(&frames),
                24 => [93, 117, 141, 165, 189].contains(&frames),
                _ => false,
            }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WanPublicMode {
    /// The one carrier-free public mode: a prompt and nothing else. Only [`WanI2vRoute::T2v14b`]
    /// admits it, and it is the only mode that route admits.
    TextToVideo,
    ImageToVideo,
    FirstLastFrame,
    ExtendClip,
    VideoBridge,
    ReplacePerson,
}

impl WanPublicMode {
    pub(super) fn key(self) -> &'static str {
        match self {
            Self::TextToVideo => "text_to_video",
            Self::ImageToVideo => "image_to_video",
            Self::FirstLastFrame => "first_last_frame",
            Self::ExtendClip => "extend_clip",
            Self::VideoBridge => "video_bridge",
            Self::ReplacePerson => "replace_person",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WanI2vBackend {
    Mlx,
    Candle,
}

impl WanI2vBackend {
    pub(super) fn key(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::Candle => "candle",
        }
    }

    pub(super) fn realization(self) -> MemoryBackendRealization {
        match self {
            Self::Mlx => MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: false,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
            Self::Candle => MemoryBackendRealization::CandleCuda {
                device_residency: true,
                host_backed_weights: false,
                host_to_device_block_materialization: false,
                block_materialization: crate::MemoryWindowMaterialization::DeviceFormatTransfer,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WanAdapterReceipt {
    pub ordinal: u32,
    pub path: PathBuf,
    pub kind: &'static str,
    pub scale_bits: u32,
    pub pass_scale_bits: Vec<u32>,
    pub expert: &'static str,
    pub digest: String,
    pub source_bytes: u64,
    pub persistent_bytes: u64,
    pub realization: &'static str,
    pub stability: &'static str,
}
