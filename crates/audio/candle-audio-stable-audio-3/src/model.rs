//! Variant-bound Stable Audio 3 generator contracts and lazy provider loads.
//!
//! `small-music` (sc-14543) and `small-sfx` (sc-14544) are architecturally identical 433M
//! post-trained checkpoints: the shipped `model_config.json` files differ only in the conditioner
//! `repo_id`, the training-only ARC discriminator fields, and the demo prompts. Only the learned
//! weights differ. That makes a single unconstrained loader a real mis-wiring hazard, because
//! [`gen_core::ModelRegistration::load`] receives no provider id — nothing in the `LoadSpec` says
//! which registration the caller reached. Every load therefore goes through
//! [`load_variant`], which binds the expected [`Variant`] at the call site and rejects a snapshot
//! whose conditioner `repo_id` or pinned file hashes belong to another checkpoint.
//!
//! `medium` (sc-14545) is a different graph, not a checkpoint swap: a 1.45B `1536x24` differential
//! DiT over the 852M SAME-L autoencoder, with a `16,777,216`-sample maximum instead of the smalls'
//! `5,292,032`. Every geometry the wrapper pins is therefore per [`Variant`] through
//! [`Variant::geometry`], so a small snapshot cannot authenticate as medium on shape alone and
//! vice versa.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_audio::gen_core::{
    self, AudioTrack, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, OffloadPolicy, Precision, Progress, WeightsSource,
};
use sha2::{Digest, Sha256};

use crate::dit::Guidance;
use crate::pipeline::{
    StableAudio3Pipeline, SynthesisParameters, VariantGeometry, CHANNELS, DEFAULT_GUIDANCE,
    DEFAULT_STEPS, SAMPLE_RATE,
};
use crate::sampler::SamplerKind;
use crate::weights::SnapshotLayout;
use crate::{resolve_device, DevicePolicy};

pub const MODEL_ID: &str = "stable_audio_3_small_music";
pub const HUB_REPO: &str = "stabilityai/stable-audio-3-small-music";
pub const HUB_REVISION: &str = "0fef1392cd842149a2b6d445e181c97608faac06";

pub const SFX_MODEL_ID: &str = "stable_audio_3_small_sfx";
pub const SFX_HUB_REPO: &str = "stabilityai/stable-audio-3-small-sfx";
pub const SFX_HUB_REVISION: &str = "ae12755283df9d62ca39a9b050a39a0b607b8c20";

pub const MEDIUM_MODEL_ID: &str = "stable_audio_3_medium";
pub const MEDIUM_HUB_REPO: &str = "stabilityai/stable-audio-3-medium";
pub const MEDIUM_HUB_REVISION: &str = "27b5a21b791b1b033d193a9e1e3ce78493f102f9";

/// The smalls' advertised logical maximum.
///
/// Their `sample_size` is `5,292,032` frames, i.e. `120.00072562...` s, so `120.0` is the largest
/// round second count the adapted geometry can serve exactly.
pub const MAX_DURATION_SECS: f32 = 120.0;

/// Medium's advertised logical maximum, matching the number Stability publishes.
///
/// Medium's `sample_size` is `16,777,216` frames, i.e. `380.43573696...` s. The advertised cap is
/// the published `380` s: it is strictly inside the geometric ceiling, so `floor(380 * 44100)`
/// frames are always available, and it is the number the model card states. The residual
/// `0.4357 s` is not reachable through the descriptor by design — advertising a cap the adapted
/// geometry can only *just* satisfy would make the exact-framing gate depend on `f32` rounding.
pub const MEDIUM_MAX_DURATION_SECS: f32 = 380.0;

/// The exact frame ceiling behind [`MEDIUM_MAX_DURATION_SECS`], as declared by medium's
/// `model_config.json` `sample_size`.
pub const MEDIUM_MAX_SAMPLE_SIZE: usize = 16_777_216;

/// The exact frame ceiling behind [`MAX_DURATION_SECS`].
pub const SMALL_MAX_SAMPLE_SIZE: usize = 5_292_032;

pub const MAX_STEPS: u32 = 500;
pub const GUIDANCE_RANGE: (f32, f32) = (0.0, 25.0);

/// The registered post-trained Stable Audio 3 checkpoints.
///
/// All three share the bundled encoder-only T5Gemma stack, the 44.1 kHz stereo output geometry, the
/// eight-step Pingpong default, and the `rf_denoiser` objective. They differ in DiT size, in
/// autoencoder (SAME-S for the smalls, SAME-L for medium), and in maximum duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// `stable_audio_3_small_music` — 433M text-to-music, SAME-S, 120 s.
    SmallMusic,
    /// `stable_audio_3_small_sfx` — 433M text-to-sound-effects / Foley, SAME-S, 120 s.
    ///
    /// Distinct from the shipped `moss_sfx_v2` provider: SA3 SFX is 44.1 kHz **stereo** with a
    /// 120-second logical maximum, where MOSS-SoundEffect is 48 kHz **mono**. They are different
    /// quality tiers and different output shapes, not interchangeable ids.
    SmallSfx,
    /// `stable_audio_3_medium` — 1.45B differential DiT over SAME-L, 380 s (sc-14545).
    ///
    /// The only released SA3 checkpoint Stability tags for **both** `music` and `sound-effects`;
    /// the two smalls are single-domain specialists. The descriptor contract has no machine-readable
    /// domain field, so that coverage is documentation-only here — see [`descriptor_for`] and
    /// `sc-15041`.
    Medium,
}

/// Every architectural quantity the strict wrapper pins, per registered checkpoint.
///
/// This exists because the wrapper's job is to reject a snapshot that is not the exact checkpoint
/// the caller's provider id names. Before sc-14545 the small geometry was hard-coded in the
/// validator, which meant medium could only ever have been admitted by loosening the check for
/// everything. Making it a per-variant record keeps each id's rejection surface as tight as it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariantShape {
    /// `model_config.json` `sample_size`, the frame ceiling the adapted geometry clamps to.
    pub sample_size: usize,
    /// Root safetensors tensor count.
    pub total_keys: usize,
    /// `model.*` (DiT) tensor count.
    pub dit_keys: usize,
    /// `pretransform.model.{encoder,decoder,bottleneck}.*` tensor count.
    pub autoencoder_keys: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    /// `attn_kwargs.differential` — direct-subtraction attention. Medium only.
    pub differential: bool,
}

/// `stabilityai/stable-audio-3-small-{music,sfx}` — 685 root tensors, SAME-S, ordinary attention.
pub const SMALL_SHAPE: VariantShape = VariantShape {
    sample_size: SMALL_MAX_SAMPLE_SIZE,
    total_keys: 685,
    dit_keys: 438,
    autoencoder_keys: 244,
    embed_dim: 1_024,
    depth: 20,
    num_heads: 16,
    differential: false,
};

/// `stabilityai/stable-audio-3-medium` — 997 root tensors, SAME-L, differential attention.
///
/// The `472` autoencoder tensors are byte-for-byte the standalone `stabilityai/SAME-L` inventory
/// (472 tensors / 53 biases), which is what makes the SAME-L parity oracles applicable to the
/// embedded copy.
pub const MEDIUM_SHAPE: VariantShape = VariantShape {
    sample_size: MEDIUM_MAX_SAMPLE_SIZE,
    total_keys: 997,
    dit_keys: 522,
    autoencoder_keys: 472,
    embed_dim: 1_536,
    depth: 24,
    num_heads: 24,
    differential: true,
};

impl Variant {
    /// Every registered variant, in registration order.
    pub const ALL: [Variant; 3] = [Variant::SmallMusic, Variant::SmallSfx, Variant::Medium];

    pub const fn model_id(self) -> &'static str {
        match self {
            Self::SmallMusic => MODEL_ID,
            Self::SmallSfx => SFX_MODEL_ID,
            Self::Medium => MEDIUM_MODEL_ID,
        }
    }

    pub const fn hub_repo(self) -> &'static str {
        match self {
            Self::SmallMusic => HUB_REPO,
            Self::SmallSfx => SFX_HUB_REPO,
            Self::Medium => MEDIUM_HUB_REPO,
        }
    }

    pub const fn hub_revision(self) -> &'static str {
        match self {
            Self::SmallMusic => HUB_REVISION,
            Self::SmallSfx => SFX_HUB_REVISION,
            Self::Medium => MEDIUM_HUB_REVISION,
        }
    }

    /// The advertised logical maximum duration, in seconds.
    pub const fn max_duration_secs(self) -> f32 {
        match self {
            Self::SmallMusic | Self::SmallSfx => MAX_DURATION_SECS,
            Self::Medium => MEDIUM_MAX_DURATION_SECS,
        }
    }

    /// The duration used when a request carries no `audio.target_duration`.
    ///
    /// Upstream's variable-length schedule wastes no compute on unrequested length, so the shipped
    /// semantics for an unspecified duration is "the checkpoint's full length". That is what the
    /// smalls already do (`120 s` is both their default and their maximum) and medium keeps it.
    ///
    /// # The cost of that on medium, stated plainly
    ///
    /// It means a request that omits `audio.target_duration` renders **380 s** — a 6.3-minute track.
    /// Measured on an M5 Max: ≈ 57–92 s on Metal depending on machine load, and extrapolating this
    /// crate's own CPU-vs-Metal ratio, ≈ 10–16 minutes on CPU. The smalls' unspecified-duration render
    /// is 120 s / ≈ 10 s on Metal, so this is an order of magnitude more expensive for the same
    /// omission.
    ///
    /// It is kept anyway. A shorter default only for medium would make three ids in one family obey
    /// two different rules for the same missing field, which is a worse contract than an expensive
    /// but uniform one — and there is no principled shorter value: any number picked here would be a
    /// product choice this crate has no basis to make, while "the checkpoint's full length" at least
    /// follows from the architecture. Callers that care about cost pass the field; it is optional,
    /// not absent.
    ///
    /// What is genuinely missing is *discoverability*: [`descriptor_for`] advertises
    /// `max_audio_duration_secs` but `Capabilities` has no field for the default, so a caller cannot
    /// see what omitting the field will cost without reading this source. Adding one is an additive
    /// `gen-core` contract change and is tracked with the other additive descriptor gaps as
    /// `sc-15041`.
    pub const fn default_duration_secs(self) -> f32 {
        self.max_duration_secs()
    }

    /// The exact architectural record the strict wrapper authenticates against.
    pub const fn shape(self) -> VariantShape {
        match self {
            Self::SmallMusic | Self::SmallSfx => SMALL_SHAPE,
            Self::Medium => MEDIUM_SHAPE,
        }
    }

    /// The wrapper's validation record: this variant's shape bound to its published repository.
    pub const fn geometry(self) -> VariantGeometry {
        VariantGeometry {
            shape: self.shape(),
            expected_repo: self.hub_repo(),
        }
    }

    const fn pins(self) -> &'static [SnapshotFilePin] {
        match self {
            Self::SmallMusic => MUSIC_SNAPSHOT_FILE_PINS,
            Self::SmallSfx => SFX_SNAPSHOT_FILE_PINS,
            Self::Medium => MEDIUM_SNAPSHOT_FILE_PINS,
        }
    }

    /// Every weight-license row this registration contributes to the catalog: the composite
    /// effective-restriction row, the root checkpoint row, and the bundled T5Gemma component row.
    pub const fn weight_licenses(self) -> &'static [gen_core::WeightLicenseEntry] {
        match self {
            Self::SmallMusic => MUSIC_WEIGHT_LICENSES,
            Self::SmallSfx => SFX_WEIGHT_LICENSES,
            Self::Medium => MEDIUM_WEIGHT_LICENSES,
        }
    }

    pub fn descriptor(self) -> ModelDescriptor {
        descriptor_for(self)
    }
}

struct SnapshotFilePin {
    relative: &'static str,
    bytes: u64,
    sha256: &'static str,
}

const MUSIC_SNAPSHOT_FILE_PINS: &[SnapshotFilePin] = &[
    SnapshotFilePin {
        relative: "model_config.json",
        bytes: 10_341,
        sha256: "100776f25af5aa83f70e0c6b384de6690cb4e5ad01c24f7cfbb6524d18765f06",
    },
    SnapshotFilePin {
        relative: "model.safetensors",
        bytes: 2_270_384_940,
        sha256: "da85866b11b01d0694d990785f6abbd79c8064df1b0e6f8aea52935e0ef84b64",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/config.json",
        bytes: 2_540,
        sha256: "575334409716886ac2952f5a275ed92868deef8a0ea560258d9970a431c6fb3a",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/model.safetensors",
        bytes: 1_183_022_944,
        sha256: "9b05ea5a4f211d023832f706fb2c0e83e4fc721b6da35ab69ceb0b55eb7800d3",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/tokenizer.json",
        bytes: 34_362_429,
        sha256: "7794135caa3ea73918949c902a781cc61dab674a4b59c17d85931c77c1114cbd",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/tokenizer.model",
        bytes: 4_241_003,
        sha256: "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    },
];

/// `stabilityai/stable-audio-3-small-sfx@ae12755283df9d62ca39a9b050a39a0b607b8c20`.
///
/// The bundled T5Gemma config, weights, and both tokenizer files are byte-identical to the
/// small-music snapshot; the root checkpoint and its `model_config.json` are not. The root
/// safetensors header carries no identity metadata, so this SHA-256 pin is the only thing that
/// separates the two 2,270,384,940-byte checkpoints on disk.
const SFX_SNAPSHOT_FILE_PINS: &[SnapshotFilePin] = &[
    SnapshotFilePin {
        relative: "model_config.json",
        bytes: 10_454,
        sha256: "a8aa5d45ae3d6524d3cd4e85e0d6e7d8d401267e7c6f28214bca8aae7b77bdeb",
    },
    SnapshotFilePin {
        relative: "model.safetensors",
        bytes: 2_270_384_940,
        sha256: "ed9cf1b6172f1a8c2921a9560c21109ff3239524563ced9dce6dcdef41e2f515",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/config.json",
        bytes: 2_540,
        sha256: "575334409716886ac2952f5a275ed92868deef8a0ea560258d9970a431c6fb3a",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/model.safetensors",
        bytes: 1_183_022_944,
        sha256: "9b05ea5a4f211d023832f706fb2c0e83e4fc721b6da35ab69ceb0b55eb7800d3",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/tokenizer.json",
        bytes: 34_362_429,
        sha256: "7794135caa3ea73918949c902a781cc61dab674a4b59c17d85931c77c1114cbd",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/tokenizer.model",
        bytes: 4_241_003,
        sha256: "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    },
];

/// `stabilityai/stable-audio-3-medium@27b5a21b791b1b033d193a9e1e3ce78493f102f9`.
///
/// The bundled T5Gemma config, weights, and both tokenizer files are byte-identical to both small
/// snapshots — medium ships the same encoder. The root checkpoint is a different size entirely
/// (9.22 GB against 2.27 GB), so unlike the music/SFX pair the byte length alone already separates
/// medium from either small; the SHA-256 pin still carries the authentication, because a byte
/// length is not an identity.
const MEDIUM_SNAPSHOT_FILE_PINS: &[SnapshotFilePin] = &[
    SnapshotFilePin {
        relative: "model_config.json",
        bytes: 10_360,
        sha256: "4f8846649df59167e1792d134acb6fc2bb7105227c5455300bad6cb107e20c88",
    },
    SnapshotFilePin {
        relative: "model.safetensors",
        bytes: 9_222_116_660,
        sha256: "48d9c65e290e7bcd5194e0633bfc2424a59ee9683f5c2d58762d997b7d8ce0b5",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/config.json",
        bytes: 2_540,
        sha256: "575334409716886ac2952f5a275ed92868deef8a0ea560258d9970a431c6fb3a",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/model.safetensors",
        bytes: 1_183_022_944,
        sha256: "9b05ea5a4f211d023832f706fb2c0e83e4fc721b6da35ab69ceb0b55eb7800d3",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/tokenizer.json",
        bytes: 34_362_429,
        sha256: "7794135caa3ea73918949c902a781cc61dab674a4b59c17d85931c77c1114cbd",
    },
    SnapshotFilePin {
        relative: "t5gemma-b-b-ul2/tokenizer.model",
        bytes: 4_241_003,
        sha256: "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    },
];

pub const ROOT_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Stability-AI-Community",
    name: "Stability AI Community License",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-small-music/blob/0fef1392cd842149a2b6d445e181c97608faac06/LICENSE.md",
    attribution: Some("Stable Audio 3 Small Music © Stability AI"),
    commercial_use: false,
    restriction: Some(
        "Use is governed by the Stability AI Community License, including its revenue threshold and prohibited-use terms.",
    ),
};

pub const GEMMA_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Gemma-Terms",
    name: "Gemma Terms of Use",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-small-music/blob/0fef1392cd842149a2b6d445e181c97608faac06/LICENSE_GEMMA.md",
    attribution: Some("T5Gemma model weights © Google"),
    commercial_use: true,
    restriction: Some("Use is governed by the Gemma Terms of Use and Prohibited Use Policy."),
};

pub const SFX_ROOT_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Stability-AI-Community",
    name: "Stability AI Community License",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-small-sfx/blob/ae12755283df9d62ca39a9b050a39a0b607b8c20/LICENSE.md",
    attribution: Some("Stable Audio 3 Small SFX © Stability AI"),
    commercial_use: false,
    restriction: Some(
        "Use is governed by the Stability AI Community License, including its revenue threshold and prohibited-use terms.",
    ),
};

pub const SFX_GEMMA_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Gemma-Terms",
    name: "Gemma Terms of Use",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-small-sfx/blob/ae12755283df9d62ca39a9b050a39a0b607b8c20/LICENSE_GEMMA.md",
    attribution: Some("T5Gemma model weights © Google"),
    commercial_use: true,
    restriction: Some("Use is governed by the Gemma Terms of Use and Prohibited Use Policy."),
};

pub const MEDIUM_ROOT_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Stability-AI-Community",
    name: "Stability AI Community License",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-medium/blob/27b5a21b791b1b033d193a9e1e3ce78493f102f9/LICENSE.md",
    attribution: Some("Stable Audio 3 Medium © Stability AI"),
    commercial_use: false,
    restriction: Some(
        "Use is governed by the Stability AI Community License, including its revenue threshold and prohibited-use terms.",
    ),
};

pub const MEDIUM_GEMMA_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Gemma-Terms",
    name: "Gemma Terms of Use",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-medium/blob/27b5a21b791b1b033d193a9e1e3ce78493f102f9/LICENSE_GEMMA.md",
    attribution: Some("T5Gemma model weights © Google"),
    commercial_use: true,
    restriction: Some("Use is governed by the Gemma Terms of Use and Prohibited Use Policy."),
};

const MUSIC_WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[
    gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: None,
        license: ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: Some("root"),
        license: ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: MODEL_ID,
        component: Some("t5gemma"),
        license: GEMMA_WEIGHT_LICENSE,
    },
];

const SFX_WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[
    gen_core::WeightLicenseEntry {
        provider_id: SFX_MODEL_ID,
        component: None,
        license: SFX_ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: SFX_MODEL_ID,
        component: Some("root"),
        license: SFX_ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: SFX_MODEL_ID,
        component: Some("t5gemma"),
        license: SFX_GEMMA_WEIGHT_LICENSE,
    },
];

const MEDIUM_WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[
    gen_core::WeightLicenseEntry {
        provider_id: MEDIUM_MODEL_ID,
        component: None,
        license: MEDIUM_ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: MEDIUM_MODEL_ID,
        component: Some("root"),
        license: MEDIUM_ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: MEDIUM_MODEL_ID,
        component: Some("t5gemma"),
        license: MEDIUM_GEMMA_WEIGHT_LICENSE,
    },
];

/// Every Stable Audio 3 weight-license row, in registration order (music, SFX, medium).
///
/// The DiT, SAME pretransform, and learned conditioner all live inside the single
/// `model.safetensors` root artifact and are covered by the `root` row; the bundled T5Gemma stack
/// is a separately licensed component and carries its own row. Three rows per registration, not
/// four: medium's SAME-L is not a separate artifact, it is a namespace inside the same file.
pub const WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[
    MUSIC_WEIGHT_LICENSES[0],
    MUSIC_WEIGHT_LICENSES[1],
    MUSIC_WEIGHT_LICENSES[2],
    SFX_WEIGHT_LICENSES[0],
    SFX_WEIGHT_LICENSES[1],
    SFX_WEIGHT_LICENSES[2],
    MEDIUM_WEIGHT_LICENSES[0],
    MEDIUM_WEIGHT_LICENSES[1],
    MEDIUM_WEIGHT_LICENSES[2],
];

/// Build the descriptor for one registered variant.
///
/// All three post-trained checkpoints share the same mathematically active batch-CFG/APG/rescale
/// path, so they expose the same guidance and negative-prompt surface. They differ only in
/// `max_audio_duration_secs`.
///
/// # Domain metadata is documentation-only
///
/// `stable_audio_3_medium` is the only released SA3 checkpoint Stability tags for **both** `music`
/// and `sound-effects`; `stable_audio_3_small_music` and `stable_audio_3_small_sfx` are
/// single-domain specialists. [`Capabilities`] has no field that can carry that, and `family` is
/// deliberately not overloaded to carry it either. sc-14545 therefore accepts **documentation-only**
/// domain metadata: the ids, this doc comment, and the crate-level module doc are the entire signal
/// a consumer gets, and no typed domain coverage is claimed. Adding typed domain / channel-count /
/// quality-tier fields is an additive contract change tracked as `sc-15041`.
///
/// # The advertised cap is also the default, and the descriptor cannot say so
///
/// `max_audio_duration_secs` is advertised; [`Variant::default_duration_secs`] is not, because
/// `Capabilities` has no field for it — and on this family the two are equal. A request that omits
/// `audio.target_duration` therefore renders the **advertised maximum**: 120 s on either small, and
/// **380 s** on medium (≈ 57–92 s of Metal compute, ≈ 16 minutes on CPU). That is a real cost
/// difference between ids whose descriptors are otherwise identical, and nothing in the descriptor
/// signals it. The reasoning for keeping the default equal to the cap, and for treating the missing
/// capability field as the actual gap, is on [`Variant::default_duration_secs`]; the field itself is
/// tracked with the other additive descriptor gaps as `sc-15041`.
pub fn descriptor_for(variant: Variant) -> ModelDescriptor {
    ModelDescriptor {
        required_components: &[],
        id: variant.model_id(),
        family: "stable_audio_3",
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["pingpong", "euler", "rk4", "dpmpp"],
            schedulers: vec![],
            supported_guidance_methods: vec!["cfg", "apg", "cfg_rescale"],
            min_size: 0,
            max_size: 0,
            max_count: 1,
            mac_only: false,
            audio_sample_rates: vec![SAMPLE_RATE],
            max_audio_duration_secs: Some(variant.max_duration_secs()),
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
        },
    }
}

pub fn descriptor() -> ModelDescriptor {
    descriptor_for(Variant::SmallMusic)
}

pub fn sfx_descriptor() -> ModelDescriptor {
    descriptor_for(Variant::SmallSfx)
}

pub fn medium_descriptor() -> ModelDescriptor {
    descriptor_for(Variant::Medium)
}

fn verify_file_pin(
    model_id: &str,
    repo: &str,
    revision: &str,
    path: &std::path::Path,
    pin: &SnapshotFilePin,
) -> gen_core::Result<()> {
    let metadata = std::fs::metadata(path).map_err(|error| {
        gen_core::Error::Msg(format!(
            "{model_id}: read pinned file {}: {error}",
            path.display()
        ))
    })?;
    if metadata.len() != pin.bytes {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: {} byte length {} does not match {repo}@{revision}",
            pin.relative,
            metadata.len()
        )));
    }
    let file = std::fs::File::open(path).map_err(|error| {
        gen_core::Error::Msg(format!(
            "{model_id}: open pinned file {}: {error}",
            path.display()
        ))
    })?;
    let mut reader = std::io::BufReader::with_capacity(1024 * 1024, file);
    let mut digest = Sha256::new();
    std::io::copy(&mut reader, &mut digest).map_err(|error| {
        gen_core::Error::Msg(format!(
            "{model_id}: hash pinned file {}: {error}",
            path.display()
        ))
    })?;
    let actual = format!("{:x}", digest.finalize());
    if actual != pin.sha256 {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: {} SHA-256 does not match {repo}@{revision}",
            pin.relative
        )));
    }
    Ok(())
}

/// Re-hash every pinned file and reject a snapshot that no longer authenticates.
///
/// `cancel` is polled between pins. This pass costs ~6.9 s over 3.45 GB on an M-series Mac, and on
/// the lazy pipeline path it runs with both the generation and pipeline mutexes held; without the
/// poll a request cancelled during cold start would not observe it until the whole verification
/// and load had finished. Polling between pins rather than inside the hash loop keeps the check
/// off the hot path while bounding the unobserved window to one file.
fn verify_snapshot_identity(
    variant: Variant,
    root: &Path,
    cancel: Option<&gen_core::CancelFlag>,
) -> gen_core::Result<()> {
    for pin in variant.pins() {
        if cancel.is_some_and(|cancel| cancel.is_cancelled()) {
            return Err(gen_core::Error::Canceled);
        }
        verify_file_pin(
            variant.model_id(),
            variant.hub_repo(),
            variant.hub_revision(),
            &root.join(pin.relative),
            pin,
        )?;
    }
    Ok(())
}

pub(crate) fn validate_request(
    variant: Variant,
    descriptor: &ModelDescriptor,
    request: &GenerationRequest,
) -> gen_core::Result<()> {
    let model_id = descriptor.id;
    descriptor
        .capabilities
        .validate_request_audio(model_id, request)?;
    if request.prompt.trim().is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: prompt must not be empty"
        )));
    }
    let audio = request.audio.clone().unwrap_or_default();
    let duration = audio
        .target_duration
        .unwrap_or_else(|| variant.default_duration_secs());
    if duration < 1.0 / SAMPLE_RATE as f32 {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: audio.target_duration must contain at least one 44.1 kHz frame"
        )));
    }
    if audio.bpm.is_some() || audio.musical_key.is_some() || audio.lyrics.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id} exposes text-prompt audio generation only; BPM, key, and lyrics conditioning are unsupported"
        )));
    }
    if let Some(steps) = request.steps {
        if steps > MAX_STEPS {
            return Err(gen_core::Error::Msg(format!(
                "{model_id}: steps {steps} exceeds the {MAX_STEPS}-step model limit"
            )));
        }
    }
    if let Some(guidance) = request.guidance {
        if !(GUIDANCE_RANGE.0..=GUIDANCE_RANGE.1).contains(&guidance) {
            return Err(gen_core::Error::Msg(format!(
                "{model_id}: guidance {guidance} outside {}..={}",
                GUIDANCE_RANGE.0, GUIDANCE_RANGE.1
            )));
        }
    }
    if request.guidance_momentum.unwrap_or(0.0) != 0.0 {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id} does not support APG momentum"
        )));
    }
    let method = request.guidance_method.as_deref();
    if request.guidance_eta.is_some() && method != Some("apg") {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id}: guidance_eta is only supported with guidance_method=apg"
        )));
    }
    if request.guidance_norm_threshold.is_some() && method != Some("apg") {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id}: guidance_norm_threshold is only supported with guidance_method=apg"
        )));
    }
    Ok(())
}

fn synthesis_parameters(variant: Variant, request: &GenerationRequest) -> SynthesisParameters {
    let audio = request.audio.clone().unwrap_or_default();
    let method = request.guidance_method.as_deref();
    let guidance = Guidance {
        cfg_scale: request.guidance.unwrap_or(DEFAULT_GUIDANCE as f32) as f64,
        apg_scale: match method {
            Some("cfg") | Some("cfg_rescale") => 0.0,
            Some("apg") => 1.0 - request.guidance_eta.unwrap_or(0.0) as f64,
            None => 1.0,
            Some(_) => unreachable!("capability validation rejects unknown methods"),
        },
        cfg_norm_threshold: request.guidance_norm_threshold.unwrap_or(0.0) as f64,
        scale_phi: if method == Some("cfg_rescale") {
            1.0
        } else {
            0.0
        },
    };
    SynthesisParameters {
        duration_secs: audio
            .target_duration
            .unwrap_or_else(|| variant.default_duration_secs()),
        steps: request.steps.unwrap_or(DEFAULT_STEPS as u32) as usize,
        sampler: match request.sampler.as_deref() {
            None | Some("pingpong") => SamplerKind::Pingpong,
            Some("euler") => SamplerKind::Euler,
            Some("rk4") => SamplerKind::Rk4,
            Some("dpmpp") => SamplerKind::Dpmpp,
            Some(_) => unreachable!("capability validation rejects unknown samplers"),
        },
        guidance,
        seed: request.seed.unwrap_or_else(gen_core::default_seed),
    }
}

/// One registered post-trained Stable Audio 3 checkpoint, bound to its [`Variant`].
pub struct StableAudio3Generator {
    variant: Variant,
    descriptor: ModelDescriptor,
    root: PathBuf,
    pipeline: Mutex<Option<Arc<StableAudio3Pipeline>>>,
    generation: Mutex<()>,
}

impl StableAudio3Generator {
    pub fn variant(&self) -> Variant {
        self.variant
    }

    fn pipeline(
        &self,
        cancel: &gen_core::CancelFlag,
    ) -> gen_core::Result<Arc<StableAudio3Pipeline>> {
        let mut guard = match self.pipeline.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(pipeline) = guard.as_ref() {
            return Ok(pipeline.clone());
        }
        let layout = SnapshotLayout::from_dir(&self.root)?;
        // Re-authenticate before any tensor is mmapped. `load_variant` already verified this
        // snapshot, but tensors are materialized lazily here, and the two sibling checkpoints have
        // byte-identical file lengths and no identity metadata in the safetensors header. Without
        // this second pass, swapping `model.safetensors` between load and first generate — leaving
        // `model_config.json` in place, so the `repo_id` check inside `from_layout` still passes —
        // would be served without complaint. Measured cost of this second pass on an M-series Mac:
        // +6.9 s, SHA-256 over the 3.45 GB of pinned files at ~500 MB/s, against a load-plus-first-
        // generate of 6.9 s without it. It runs once per generator, and generators are constructed
        // once and serve many requests; the alternative is serving unauthenticated weights.
        //
        // The caller's cancel flag is threaded in so a request cancelled during this cold-start
        // window observes it between pins instead of after the full load.
        verify_snapshot_identity(self.variant, &layout.root, Some(cancel))?;
        let device = resolve_device(DevicePolicy::Default)?;
        let pipeline = Arc::new(StableAudio3Pipeline::from_layout(
            &layout,
            self.variant.geometry(),
            &device,
        )?);
        *guard = Some(pipeline.clone());
        Ok(pipeline)
    }
}

/// Load one registered variant from a caller-provisioned snapshot directory.
///
/// `expected` is supplied by the registration site, never inferred from the snapshot: a snapshot
/// that authenticates as the *other* checkpoint is rejected rather than silently accepted under the
/// wrong provider id.
///
/// Weights are not read here. The pinned identity established by this call is re-verified on the
/// generator's lazy pipeline path immediately before the tensors are mmapped, so a snapshot mutated
/// between load and first generate is rejected rather than served.
pub fn load_variant(expected: Variant, spec: &LoadSpec) -> gen_core::Result<StableAudio3Generator> {
    let model_id = expected.model_id();
    let root = match &spec.weights {
        WeightsSource::Dir(path) => path.clone(),
        WeightsSource::File(path) => {
            return Err(gen_core::Error::Msg(format!(
                "{model_id} requires a snapshot directory, got {}",
                path.display()
            )));
        }
    };
    if spec.quantize.is_some()
        || spec.precision != Precision::Bf16
        || !spec.adapters.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
        || spec.offload_policy != OffloadPolicy::Resident
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id} accepts only its native dense self-contained snapshot"
        )));
    }
    let layout = SnapshotLayout::from_dir(&root)?;
    crate::pipeline::validate_layout(&layout, expected.geometry())?;
    // No request is in flight on the load path, so there is no cancel flag to honour here.
    verify_snapshot_identity(expected, &layout.root, None)?;
    Ok(StableAudio3Generator {
        variant: expected,
        descriptor: descriptor_for(expected),
        root,
        pipeline: Mutex::new(None),
        generation: Mutex::new(()),
    })
}

pub fn load_generator(spec: &LoadSpec) -> gen_core::Result<StableAudio3Generator> {
    load_variant(Variant::SmallMusic, spec)
}

pub fn load_sfx_generator(spec: &LoadSpec) -> gen_core::Result<StableAudio3Generator> {
    load_variant(Variant::SmallSfx, spec)
}

pub fn load_medium_generator(spec: &LoadSpec) -> gen_core::Result<StableAudio3Generator> {
    load_variant(Variant::Medium, spec)
}

pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_variant(Variant::SmallMusic, spec)?))
}

pub fn sfx_load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_variant(Variant::SmallSfx, spec)?))
}

pub fn medium_load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_variant(Variant::Medium, spec)?))
}

impl Generator for StableAudio3Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(self.variant, &self.descriptor, request)
    }

    fn generate(
        &self,
        request: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(request)?;
        if request.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        // Candle's shared Metal graph is not safe to execute concurrently: command-buffer
        // interleaving can change the resulting PCM even when every request owns its RNG stream.
        // Serialize graph execution while keeping all stochastic state request-local.
        let _generation = match self.generation.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let parameters = synthesis_parameters(self.variant, request);
        let pipeline = self.pipeline(&request.cancel)?;
        let cancel = request.cancel.clone();
        let progress = std::cell::RefCell::new(on_progress);
        let mut step_progress = |current: usize, total: usize| {
            (progress.borrow_mut())(Progress::Step {
                current: current as u32,
                total: total as u32,
            });
        };
        let mut decoding = || (progress.borrow_mut())(Progress::Decoding);
        let samples = pipeline.synthesize(
            &request.prompt,
            request.negative_prompt.as_deref(),
            parameters,
            &mut step_progress,
            &mut decoding,
            &|| cancel.is_cancelled(),
        )?;
        Ok(GenerationOutput::Audio(AudioTrack {
            samples,
            sample_rate: SAMPLE_RATE,
            channels: CHANNELS as u16,
            stems: Vec::new(),
        }))
    }
}

candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

candle_audio::register_generators! {
    pub const SFX_REGISTRATION = sfx_descriptor => sfx_load
}

candle_audio::register_generators! {
    pub const MEDIUM_REGISTRATION = medium_descriptor => medium_load
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{
        AdapterKind, AdapterSpec, AudioParams, Conditioning, Image, Quant,
    };

    const VARIANTS: [Variant; 3] = Variant::ALL;

    /// The +6.9 s cold-start re-hash runs with both the generation and pipeline mutexes held, so a
    /// request cancelled during it must observe the cancellation there rather than after the load.
    ///
    /// The uncancelled leg is what makes this discriminating: with the poll removed, the cancelled
    /// leg would reach the filesystem and return the same missing-file error, so a test that only
    /// asserted "cancelled returns Canceled" would pass either way.
    #[test]
    fn cold_start_snapshot_verification_observes_cancellation_before_hashing() {
        let root = Path::new("/nonexistent/sa3-small-snapshot");

        let uncancelled = verify_snapshot_identity(Variant::SmallSfx, root, None)
            .expect_err("a missing snapshot must not verify");
        assert!(
            !matches!(uncancelled, gen_core::Error::Canceled),
            "without a cancel flag the pass must reach the filesystem, got {uncancelled:?}"
        );

        let cancel = gen_core::CancelFlag::new();
        cancel.cancel();
        let cancelled = verify_snapshot_identity(Variant::SmallSfx, root, Some(&cancel))
            .expect_err("a cancelled request must not verify");
        assert!(
            matches!(cancelled, gen_core::Error::Canceled),
            "a cancelled request must observe cancellation before hashing, got {cancelled:?}"
        );
    }

    fn request() -> GenerationRequest {
        GenerationRequest {
            prompt: "orchestral post-rock with bowed strings".into(),
            audio: Some(AudioParams {
                target_duration: Some(30.0),
                sample_rate: Some(SAMPLE_RATE),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_is_honest_and_conformant() {
        for variant in VARIANTS {
            let descriptor = descriptor_for(variant);
            assert_eq!(descriptor.id, variant.model_id());
            assert_eq!(descriptor.family, "stable_audio_3");
            assert_eq!(descriptor.backend, "candle");
            assert!(matches!(descriptor.modality, Modality::Audio));
            assert_eq!(descriptor.capabilities.audio_sample_rates, [SAMPLE_RATE]);
            // sc-14545 made the cap per variant. Asserting it against `variant.max_duration_secs()`
            // alone would be a tautology, so the expected value is spelled out per id: the smalls
            // publish the crate constant, medium publishes its own.
            let expected_cap = match variant {
                Variant::SmallMusic | Variant::SmallSfx => MAX_DURATION_SECS,
                Variant::Medium => MEDIUM_MAX_DURATION_SECS,
            };
            assert_eq!(
                descriptor.capabilities.max_audio_duration_secs,
                Some(expected_cap),
                "{} advertised duration cap",
                variant.model_id()
            );
            // The advertised cap is also the default for an omitted `audio.target_duration`; see
            // `Variant::default_duration_secs` for the cost that implies on medium.
            assert_eq!(variant.default_duration_secs(), expected_cap);
            // sc-14544: both post-trained objectives share the batch-CFG/APG/rescale math, so SFX
            // must NOT be distinguished from music by a false guidance flag.
            assert!(descriptor.capabilities.supports_negative_prompt);
            assert!(descriptor.capabilities.supports_guidance);
            assert_eq!(
                descriptor.capabilities.samplers,
                ["pingpong", "euler", "rk4", "dpmpp"]
            );
        }
        // The cap must be variant-bound rather than the crate-global it replaced.
        assert_ne!(
            descriptor_for(Variant::Medium)
                .capabilities
                .max_audio_duration_secs,
            descriptor_for(Variant::SmallMusic)
                .capabilities
                .max_audio_duration_secs
        );
        assert_eq!(descriptor().id, "stable_audio_3_small_music");
        assert_eq!(sfx_descriptor().id, "stable_audio_3_small_sfx");
        assert_eq!((REGISTRATION.descriptor)().id, MODEL_ID);
        assert_eq!((SFX_REGISTRATION.descriptor)().id, SFX_MODEL_ID);
    }

    #[test]
    fn the_two_variants_are_distinct_checkpoints_with_distinct_pins() {
        assert_ne!(Variant::SmallMusic.model_id(), Variant::SmallSfx.model_id());
        assert_ne!(Variant::SmallMusic.hub_repo(), Variant::SmallSfx.hub_repo());
        assert_ne!(
            Variant::SmallMusic.hub_revision(),
            Variant::SmallSfx.hub_revision()
        );
        let music = Variant::SmallMusic.pins();
        let sfx = Variant::SmallSfx.pins();
        assert_eq!(music.len(), sfx.len());
        for (music, sfx) in music.iter().zip(sfx) {
            assert_eq!(music.relative, sfx.relative);
        }
        // The root checkpoint and its config differ; the bundled T5Gemma stack is byte-identical.
        let by_name = |pins: &'static [SnapshotFilePin], name: &str| {
            pins.iter().find(|pin| pin.relative == name).unwrap()
        };
        assert_ne!(
            by_name(music, "model.safetensors").sha256,
            by_name(sfx, "model.safetensors").sha256
        );
        assert_ne!(
            by_name(music, "model_config.json").sha256,
            by_name(sfx, "model_config.json").sha256
        );
        assert_eq!(
            by_name(music, "t5gemma-b-b-ul2/model.safetensors").sha256,
            by_name(sfx, "t5gemma-b-b-ul2/model.safetensors").sha256
        );
    }

    #[test]
    fn every_variant_contributes_composite_and_component_license_rows() {
        // Three registered variants x (composite, root, t5gemma). sc-14545 added the medium trio.
        assert_eq!(WEIGHT_LICENSES.len(), 9);
        assert_eq!(VARIANTS.len(), 3);
        for variant in VARIANTS {
            let rows = variant.weight_licenses();
            assert_eq!(rows.len(), 3);
            assert!(rows.iter().all(|row| row.provider_id == variant.model_id()));
            assert_eq!(rows[0].component, None);
            assert_eq!(rows[1].component, Some("root"));
            assert_eq!(rows[2].component, Some("t5gemma"));
            assert_eq!(rows[0].license.spdx_id, "LicenseRef-Stability-AI-Community");
            assert!(!rows[0].license.commercial_use);
            assert!(rows[0].license.restriction.is_some());
            assert_eq!(rows[2].license.spdx_id, "LicenseRef-Gemma-Terms");
            assert!(rows[1].license.source_url.contains(variant.hub_revision()));
            assert!(rows[2].license.source_url.contains(variant.hub_revision()));
            assert!(
                WEIGHT_LICENSES
                    .iter()
                    .filter(|row| row.provider_id == variant.model_id())
                    .count()
                    == 3
            );
        }
    }

    #[test]
    fn request_validation_maps_the_complete_public_surface() {
        for variant in VARIANTS {
            let descriptor = descriptor_for(variant);
            assert!(validate_request(variant, &descriptor, &request()).is_ok());
            let mut valid = request();
            valid.negative_prompt = Some("harsh clipping and speech".into());
            valid.guidance = Some(7.5);
            valid.sampler = Some("dpmpp".into());
            valid.guidance_method = Some("apg".into());
            valid.guidance_eta = Some(0.25);
            valid.guidance_norm_threshold = Some(2.0);
            assert!(validate_request(variant, &descriptor, &valid).is_ok());

            let mut invalid = request();
            invalid.audio.as_mut().unwrap().bpm = Some(120.0);
            assert!(matches!(
                validate_request(variant, &descriptor, &invalid),
                Err(gen_core::Error::Unsupported(_))
            ));

            for field in ["musical_key", "lyrics"] {
                let mut invalid = request();
                match field {
                    "musical_key" => {
                        invalid.audio.as_mut().unwrap().musical_key = Some("D minor".into())
                    }
                    "lyrics" => invalid.audio.as_mut().unwrap().lyrics = Some("sing this".into()),
                    _ => unreachable!(),
                }
                assert!(matches!(
                    validate_request(variant, &descriptor, &invalid),
                    Err(gen_core::Error::Unsupported(_))
                ));
            }

            let mut invalid = request();
            invalid.conditioning.push(Conditioning::Reference {
                image: Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0, 0, 0],
                },
                strength: None,
            });
            assert!(matches!(
                validate_request(variant, &descriptor, &invalid),
                Err(gen_core::Error::Unsupported(_))
            ));

            let mut invalid = request();
            invalid.guidance_method = Some("apg".into());
            invalid.guidance_momentum = Some(0.1);
            assert!(matches!(
                validate_request(variant, &descriptor, &invalid),
                Err(gen_core::Error::Unsupported(_))
            ));

            let mut invalid = request();
            invalid.audio.as_mut().unwrap().target_duration = Some(0.0);
            assert!(validate_request(variant, &descriptor, &invalid).is_err());

            let mut invalid = request();
            invalid.prompt = "  ".into();
            assert!(validate_request(variant, &descriptor, &invalid).is_err());

            let mut invalid = request();
            invalid.steps = Some(MAX_STEPS + 1);
            assert!(validate_request(variant, &descriptor, &invalid).is_err());

            let mut invalid = request();
            invalid.guidance = Some(GUIDANCE_RANGE.1 + 0.1);
            assert!(validate_request(variant, &descriptor, &invalid).is_err());
        }
    }

    #[test]
    fn guidance_methods_map_to_frozen_cfg_apg_endpoints() {
        let mut cfg = request();
        cfg.guidance = Some(4.0);
        cfg.guidance_method = Some("cfg".into());
        assert_eq!(
            synthesis_parameters(Variant::SmallMusic, &cfg).guidance,
            Guidance {
                cfg_scale: 4.0,
                apg_scale: 0.0,
                cfg_norm_threshold: 0.0,
                scale_phi: 0.0,
            }
        );

        let mut apg = cfg.clone();
        apg.guidance_method = Some("apg".into());
        apg.guidance_eta = Some(0.25);
        apg.guidance_norm_threshold = Some(3.0);
        assert_eq!(
            synthesis_parameters(Variant::SmallMusic, &apg)
                .guidance
                .apg_scale,
            0.75
        );
        assert_eq!(
            synthesis_parameters(Variant::SmallMusic, &apg)
                .guidance
                .cfg_norm_threshold,
            3.0
        );

        let mut rescale = cfg;
        rescale.guidance_method = Some("cfg_rescale".into());
        assert_eq!(
            synthesis_parameters(Variant::SmallMusic, &rescale)
                .guidance
                .apg_scale,
            0.0
        );
        assert_eq!(
            synthesis_parameters(Variant::SmallMusic, &rescale)
                .guidance
                .scale_phi,
            1.0
        );
    }

    #[test]
    fn load_rejects_every_non_native_shape_before_touching_the_snapshot() {
        let missing = PathBuf::from("does-not-exist");
        for loader in [
            load as fn(&LoadSpec) -> gen_core::Result<Box<dyn Generator>>,
            sfx_load,
        ] {
            assert!(loader(&LoadSpec::new(WeightsSource::File(missing.clone()))).is_err());

            let dense = || LoadSpec::new(WeightsSource::Dir(missing.clone()));
            let mut specs = Vec::new();
            specs.push(dense().with_quant(Quant::Q4));

            let mut precision = dense();
            precision.precision = Precision::Fp32;
            specs.push(precision);

            specs.push(dense().with_adapters(vec![AdapterSpec::new(
                missing.clone(),
                1.0,
                AdapterKind::Lora,
            )]));
            specs.push(dense().with_control(WeightsSource::File(missing.clone())));
            specs.push(dense().with_extra_control(WeightsSource::File(missing.clone())));
            specs.push(dense().with_ip_adapter(WeightsSource::Dir(missing.clone())));
            specs.push(dense().with_pid(
                WeightsSource::File(missing.clone()),
                WeightsSource::Dir(missing.clone()),
            ));
            specs.push(dense().with_component("unsupported", WeightsSource::Dir(missing.clone())));
            specs.push(dense().with_offload_policy(OffloadPolicy::Sequential));

            let mut text = dense();
            text.text_encoder = Some(WeightsSource::Dir(missing.clone()));
            specs.push(text);

            let mut identity = dense();
            identity.identity = Some(Default::default());
            specs.push(identity);

            for spec in specs {
                assert!(matches!(
                    loader(&spec),
                    Err(gen_core::Error::Unsupported(_))
                ));
            }
        }
    }

    #[test]
    fn pinned_file_authentication_rejects_size_and_payload_drift() {
        let root = std::env::temp_dir().join(format!(
            "sa3-provider-pin-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("fixture");
        std::fs::write(&path, b"exact").unwrap();
        let exact = SnapshotFilePin {
            relative: "fixture",
            bytes: 5,
            sha256: "fa79d4746c21cd960a17b92db8976ddef95a7e20b590721f8e0fa7847a05e486",
        };
        verify_file_pin(MODEL_ID, HUB_REPO, HUB_REVISION, &path, &exact).unwrap();

        std::fs::write(&path, b"wrong").unwrap();
        assert!(verify_file_pin(MODEL_ID, HUB_REPO, HUB_REVISION, &path, &exact).is_err());
        std::fs::write(&path, b"short").unwrap();
        let wrong_size = SnapshotFilePin { bytes: 4, ..exact };
        assert!(verify_file_pin(MODEL_ID, HUB_REPO, HUB_REVISION, &path, &wrong_size).is_err());
        let _ = std::fs::remove_dir_all(root);
    }
}
