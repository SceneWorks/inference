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
//!
//! The three `-base` checkpoints (sc-14546) are the pre-trained flow-matching models the
//! post-trained ones were distilled from. They sharpen the same hazard in a new direction:
//!
//! * `small-{music,sfx}-base` differ from their post-trained siblings on `diffusion_objective` and
//!   on `sample_size` (`5,324,800` against `5,292,032`) and on nothing else outside `training.*`;
//! * `medium-base` differs from `medium` on **`diffusion_objective` alone** — same 997/522/472
//!   inventory, same `16,777,216` `sample_size`, same `9,222,116,660`-byte root, same declared
//!   conditioner `repo_id`.
//!
//! So [`crate::pipeline::VariantGeometry`] carries the objective per variant rather than pinning
//! `rf_denoiser` globally, and it separates *provenance* ([`Variant::hub_repo`]) from the *gate
//! value* ([`Variant::conditioner_repo`]): base configs declare their post-trained sibling's
//! repository, so those two strings differ for every base id and comparing the wrong one rejects
//! every base snapshot.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_audio::candle_core::Device;
use candle_audio::gen_core::{
    self, AdapterSpec, AudioEditMode, AudioTrack, Capabilities, Conditioning, ConditioningKind,
    GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor,
    OffloadPolicy, Precision, Progress, WeightsSource,
};
use sha2::{Digest, Sha256};

use crate::config::DiffusionObjective;
use crate::dit::Guidance;
use crate::pipeline::{
    AudioEdit, EditRegionSecs, ForwardedConditioning, ReferenceAudio, StableAudio3Pipeline,
    SynthesisParameters, VariantGeometry, BASE_DEFAULT_GUIDANCE, BASE_DEFAULT_STEPS, CHANNELS,
    DEFAULT_GUIDANCE, DEFAULT_STEPS, SAMPLE_RATE,
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

pub const MUSIC_BASE_MODEL_ID: &str = "stable_audio_3_small_music_base";
pub const MUSIC_BASE_HUB_REPO: &str = "stabilityai/stable-audio-3-small-music-base";
pub const MUSIC_BASE_HUB_REVISION: &str = "eab5ceee5ad9c1ed38800aff30a8e49d1161c539";

pub const SFX_BASE_MODEL_ID: &str = "stable_audio_3_small_sfx_base";
pub const SFX_BASE_HUB_REPO: &str = "stabilityai/stable-audio-3-small-sfx-base";
pub const SFX_BASE_HUB_REVISION: &str = "cc5ddb990e30daa68336ac61c140c37c7033ab7c";

pub const MEDIUM_BASE_MODEL_ID: &str = "stable_audio_3_medium_base";
pub const MEDIUM_BASE_HUB_REPO: &str = "stabilityai/stable-audio-3-medium-base";
pub const MEDIUM_BASE_HUB_REVISION: &str = "b32993f73c3bdc3864043a72d8032606bba737c8";

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

/// The exact frame ceiling behind [`MAX_DURATION_SECS`] on the two post-trained smalls.
pub const SMALL_MAX_SAMPLE_SIZE: usize = 5_292_032;

/// The two small **base** checkpoints' `sample_size`, which is *not* the post-trained smalls'.
///
/// `5,324,800` frames is `120.743764...` s. The advertised cap stays [`MAX_DURATION_SECS`] anyway:
/// `conformance.rs` requires the advertised cap to be tight, i.e. one second past it must not fit,
/// and `121 s = 5,336,100` frames does not fit in `5,324,800`. So the extra `0.74 s` is
/// unreachable through the descriptor by exactly the same rule that keeps medium at `380` s.
///
/// What this constant *is* used for is the identity gate: it is what separates
/// `stable_audio_3_small_music` from `stable_audio_3_small_music_base` on config alone, in both
/// directions, before `diffusion_objective` is even consulted.
pub const SMALL_BASE_MAX_SAMPLE_SIZE: usize = 5_324_800;

pub const MAX_STEPS: u32 = 500;
pub const GUIDANCE_RANGE: (f32, f32) = (0.0, 25.0);

/// The registered Stable Audio 3 checkpoints: three post-trained, three `-base` (sc-14546).
///
/// All six share the bundled encoder-only T5Gemma stack, the 44.1 kHz stereo output geometry, and
/// the same mathematically active batch-CFG / APG / rescale path. They differ in DiT size, in
/// autoencoder (SAME-S for the smalls, SAME-L for medium), in maximum duration, and — across the
/// post-trained/base split — in `diffusion_objective` and therefore in the sampler and operating
/// point resolved for a request that omits them.
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
    /// `stable_audio_3_small_music_base` — the pre-trained flow-matching music checkpoint
    /// (sc-14546).
    ///
    /// Same 433M `1024x20` graph and same SAME-S autoencoder as [`Variant::SmallMusic`], different
    /// learned weights: this one has not been distilled or adversarially post-trained. Its config
    /// declares `rectified_flow` rather than `rf_denoiser` and a `5,324,800`-frame `sample_size`
    /// rather than `5,292,032`.
    SmallMusicBase,
    /// `stable_audio_3_small_sfx_base` — the pre-trained flow-matching SFX / Foley checkpoint
    /// (sc-14546).
    SmallSfxBase,
    /// `stable_audio_3_medium_base` — the pre-trained flow-matching 1.45B checkpoint (sc-14546).
    ///
    /// The sharpest identity case in the family: against [`Variant::Medium`] it agrees on tensor
    /// inventory (997/522/472), on `sample_size` (`16,777,216`), on root safetensors byte length
    /// (`9,222,116,660`) *and* on conditioner `repo_id`. `diffusion_objective` plus the SHA-256 pin
    /// are the entire gate in both directions.
    MediumBase,
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

/// `stabilityai/stable-audio-3-small-{music,sfx}-base` — the smalls' graph at the base
/// `sample_size`.
///
/// Identical to [`SMALL_SHAPE`] in every architectural field; only the frame ceiling differs, and
/// that single field is what rejects a post-trained small snapshot under a base id and vice versa
/// before the objective check is reached.
pub const SMALL_BASE_SHAPE: VariantShape = VariantShape {
    sample_size: SMALL_BASE_MAX_SAMPLE_SIZE,
    total_keys: SMALL_SHAPE.total_keys,
    dit_keys: SMALL_SHAPE.dit_keys,
    autoencoder_keys: SMALL_SHAPE.autoencoder_keys,
    embed_dim: SMALL_SHAPE.embed_dim,
    depth: SMALL_SHAPE.depth,
    num_heads: SMALL_SHAPE.num_heads,
    differential: SMALL_SHAPE.differential,
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
    ///
    /// Post-trained first, then the three `-base` siblings in the same music / SFX / medium order.
    /// New variants append, so no shipped ordered-surface assertion has to be rewritten.
    pub const ALL: [Variant; 6] = [
        Variant::SmallMusic,
        Variant::SmallSfx,
        Variant::Medium,
        Variant::SmallMusicBase,
        Variant::SmallSfxBase,
        Variant::MediumBase,
    ];

    pub const fn model_id(self) -> &'static str {
        match self {
            Self::SmallMusic => MODEL_ID,
            Self::SmallSfx => SFX_MODEL_ID,
            Self::Medium => MEDIUM_MODEL_ID,
            Self::SmallMusicBase => MUSIC_BASE_MODEL_ID,
            Self::SmallSfxBase => SFX_BASE_MODEL_ID,
            Self::MediumBase => MEDIUM_BASE_MODEL_ID,
        }
    }

    /// The repository this checkpoint was published from.
    ///
    /// **Provenance, not a gate value.** A base snapshot's `model_config.json` declares its
    /// post-trained sibling's repository, so comparing this string to the snapshot's conditioner
    /// `repo_id` rejects every base checkpoint. That comparison uses
    /// [`Variant::conditioner_repo`]; this is used for license source URLs, the CI snapshot
    /// manifest, and error messages.
    pub const fn hub_repo(self) -> &'static str {
        match self {
            Self::SmallMusic => HUB_REPO,
            Self::SmallSfx => SFX_HUB_REPO,
            Self::Medium => MEDIUM_HUB_REPO,
            Self::SmallMusicBase => MUSIC_BASE_HUB_REPO,
            Self::SmallSfxBase => SFX_BASE_HUB_REPO,
            Self::MediumBase => MEDIUM_BASE_HUB_REPO,
        }
    }

    /// The conditioner `repo_id` this checkpoint's own `model_config.json` declares.
    ///
    /// Equal to [`Variant::hub_repo`] for the three post-trained ids and to the **post-trained
    /// sibling's** repository for the three `-base` ids, because that is what Stability shipped
    /// inside the base configs. The value is compared as a string and never resolved over the
    /// network.
    pub const fn conditioner_repo(self) -> &'static str {
        match self {
            Self::SmallMusic | Self::SmallMusicBase => HUB_REPO,
            Self::SmallSfx | Self::SmallSfxBase => SFX_HUB_REPO,
            Self::Medium | Self::MediumBase => MEDIUM_HUB_REPO,
        }
    }

    pub const fn hub_revision(self) -> &'static str {
        match self {
            Self::SmallMusic => HUB_REVISION,
            Self::SmallSfx => SFX_HUB_REVISION,
            Self::Medium => MEDIUM_HUB_REVISION,
            Self::SmallMusicBase => MUSIC_BASE_HUB_REVISION,
            Self::SmallSfxBase => SFX_BASE_HUB_REVISION,
            Self::MediumBase => MEDIUM_BASE_HUB_REVISION,
        }
    }

    /// Whether this is one of the three pre-trained `-base` checkpoints.
    pub const fn is_base(self) -> bool {
        matches!(
            self,
            Self::SmallMusicBase | Self::SmallSfxBase | Self::MediumBase
        )
    }

    /// The `diffusion_objective` this checkpoint's config must declare.
    ///
    /// The only universal post-trained/base discriminator. `rectified_flow` and `rf_denoiser` are
    /// handled identically at every branch in [`crate::dit`] and [`crate::sampler`] — same
    /// prediction target, same `denoised = x - sigma * out` interpretation, same guidance math — so
    /// this field changes exactly two things: which snapshot authenticates under which id, and
    /// which sampler [`SamplerKind::recommended`] resolves for a request that omits one.
    pub const fn objective(self) -> DiffusionObjective {
        if self.is_base() {
            DiffusionObjective::RectifiedFlow
        } else {
            DiffusionObjective::RfDenoiser
        }
    }

    /// The sampler a request that omits `sampler` resolves to.
    ///
    /// Derived from [`Variant::objective`] through [`SamplerKind::recommended`], which is the
    /// frozen-upstream rule (`sampler_type = "pingpong" if diffusion_objective == "rf_denoiser"
    /// else "euler"`). Before sc-14546 that function was dead code on the provider path — the
    /// request-to-parameters mapping hard-coded Pingpong — so the two smalls and medium resolved
    /// correctly only by coincidence.
    ///
    /// Infallible here even though `recommended` returns a `Result`: no registered variant declares
    /// the unreachable `v` objective, and that is asserted rather than assumed
    /// (`recommended_sampler_matches_the_frozen_objective_rule`).
    pub fn recommended_sampler(self) -> SamplerKind {
        SamplerKind::recommended(self.objective())
            .expect("no registered Stable Audio 3 variant declares the v-diffusion objective")
    }

    /// The step count used when a request carries no `steps`.
    ///
    /// `8` for the post-trained checkpoints (upstream's API default and their own
    /// `training.demo.demo_steps`), [`BASE_DEFAULT_STEPS`] for the bases. See that constant for why
    /// `50` is a deliberate product choice rather than an upstream default.
    pub const fn default_steps(self) -> usize {
        if self.is_base() {
            BASE_DEFAULT_STEPS
        } else {
            DEFAULT_STEPS
        }
    }

    /// The guidance used when a request carries no `guidance`.
    ///
    /// `1.0` for the post-trained checkpoints, which is the one value at which
    /// [`crate::dit::StableAudio3Dit::forward_guided`] takes its batch-1 branch and a negative
    /// prompt genuinely has no effect. [`BASE_DEFAULT_GUIDANCE`] for the bases, which puts every
    /// default base render on the batch-2 CFG path.
    pub const fn default_guidance(self) -> f64 {
        if self.is_base() {
            BASE_DEFAULT_GUIDANCE
        } else {
            DEFAULT_GUIDANCE
        }
    }

    /// The advertised logical maximum duration, in seconds.
    ///
    /// The two small bases advertise [`MAX_DURATION_SECS`] like the post-trained smalls even though
    /// their `sample_size` is larger — see [`SMALL_BASE_MAX_SAMPLE_SIZE`].
    pub const fn max_duration_secs(self) -> f32 {
        match self {
            Self::SmallMusic | Self::SmallSfx | Self::SmallMusicBase | Self::SmallSfxBase => {
                MAX_DURATION_SECS
            }
            Self::Medium | Self::MediumBase => MEDIUM_MAX_DURATION_SECS,
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
    ///
    /// # sc-14546 makes the medium-base case sharper still
    ///
    /// `stable_audio_3_medium_base` inherits the same 380 s default *and* the base operating point
    /// of 50 Euler steps at guidance 7, which is a batch-2 CFG forward per step. That is 100 DiT
    /// forwards where post-trained medium does 8 — **12.5x** the example-work, on the same 380 s of
    /// audio. Extrapolating medium's measured 92 s Metal render at 380 s / 8 steps, an omitted
    /// `audio.target_duration` on this id is on the order of **20 minutes** of Metal compute, and
    /// correspondingly worse on CPU. The default is still not special-cased, for the reason above:
    /// six ids in one family obeying two rules for the same missing field is a worse contract than
    /// one expensive uniform rule, and no shorter number follows from anything but taste.
    pub const fn default_duration_secs(self) -> f32 {
        self.max_duration_secs()
    }

    /// The exact architectural record the strict wrapper authenticates against.
    pub const fn shape(self) -> VariantShape {
        match self {
            Self::SmallMusic | Self::SmallSfx => SMALL_SHAPE,
            Self::SmallMusicBase | Self::SmallSfxBase => SMALL_BASE_SHAPE,
            Self::Medium | Self::MediumBase => MEDIUM_SHAPE,
        }
    }

    /// The wrapper's validation record.
    ///
    /// Note the two distinct repository-shaped values: [`Variant::hub_repo`] is provenance and
    /// [`Variant::conditioner_repo`] is the string the snapshot's config must declare. They differ
    /// for every `-base` variant, which is why [`VariantGeometry`] carries both.
    pub const fn geometry(self) -> VariantGeometry {
        VariantGeometry {
            shape: self.shape(),
            hub_repo: self.hub_repo(),
            expected_conditioner_repo: self.conditioner_repo(),
            expected_objective: self.objective(),
        }
    }

    const fn pins(self) -> &'static [SnapshotFilePin] {
        match self {
            Self::SmallMusic => MUSIC_SNAPSHOT_FILE_PINS,
            Self::SmallSfx => SFX_SNAPSHOT_FILE_PINS,
            Self::Medium => MEDIUM_SNAPSHOT_FILE_PINS,
            Self::SmallMusicBase => MUSIC_BASE_SNAPSHOT_FILE_PINS,
            Self::SmallSfxBase => SFX_BASE_SNAPSHOT_FILE_PINS,
            Self::MediumBase => MEDIUM_BASE_SNAPSHOT_FILE_PINS,
        }
    }

    /// Every weight-license row this registration contributes to the catalog: the composite
    /// effective-restriction row, the root checkpoint row, and the bundled T5Gemma component row.
    pub const fn weight_licenses(self) -> &'static [gen_core::WeightLicenseEntry] {
        match self {
            Self::SmallMusic => MUSIC_WEIGHT_LICENSES,
            Self::SmallSfx => SFX_WEIGHT_LICENSES,
            Self::Medium => MEDIUM_WEIGHT_LICENSES,
            Self::SmallMusicBase => MUSIC_BASE_WEIGHT_LICENSES,
            Self::SmallSfxBase => SFX_BASE_WEIGHT_LICENSES,
            Self::MediumBase => MEDIUM_BASE_WEIGHT_LICENSES,
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

/// `stabilityai/stable-audio-3-small-music-base@eab5ceee5ad9c1ed38800aff30a8e49d1161c539`.
///
/// The root checkpoint is exactly `2,270,384,940` bytes — the same length as **both** post-trained
/// small roots and as the SFX base root. Four checkpoints of identical byte length and no identity
/// metadata in any safetensors header, so these SHA-256 digests are the only thing separating them
/// on disk. The bundled T5Gemma stack is byte-identical across all six SA3 repositories.
///
/// `svd_bases.pt` (1.27 GB) is deliberately not pinned: [`crate::weights::SnapshotLayout::from_dir`]
/// names every file it opens and never scans the directory, so the artifact is unreachable from the
/// inference path. `release/real-weight-models.toml` also stops downloading it.
const MUSIC_BASE_SNAPSHOT_FILE_PINS: &[SnapshotFilePin] = &[
    SnapshotFilePin {
        relative: "model_config.json",
        bytes: 8_452,
        sha256: "e97f53a0882a99b0307cf7d87d63423f75925b5def5efc180d29740899c62486",
    },
    SnapshotFilePin {
        relative: "model.safetensors",
        bytes: 2_270_384_940,
        sha256: "79691fac2a08a99a229b98963f8377ddc78c16f5c63f7e3a022375c311093ab4",
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

/// `stabilityai/stable-audio-3-small-sfx-base@cc5ddb990e30daa68336ac61c140c37c7033ab7c`.
const SFX_BASE_SNAPSHOT_FILE_PINS: &[SnapshotFilePin] = &[
    SnapshotFilePin {
        relative: "model_config.json",
        bytes: 8_476,
        sha256: "cecfbca43fa8ece5d1d128fba74a39594ea954b2b128048534f72665c6721165",
    },
    SnapshotFilePin {
        relative: "model.safetensors",
        bytes: 2_270_384_940,
        sha256: "0c7cddb22cd7bf5e8169c6731a9eb7690c27ef5b53e8a4f9b4b2aaf8afcdaea2",
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

/// `stabilityai/stable-audio-3-medium-base@b32993f73c3bdc3864043a72d8032606bba737c8`.
///
/// The sharpest pin in the crate. This root is `9,222,116,660` bytes, the *same* length as the
/// post-trained medium root, under a config that agrees with medium's on tensor inventory,
/// `sample_size` and conditioner `repo_id`. Strip `diffusion_objective` from the validator and this
/// SHA-256 is all that stands between `stable_audio_3_medium` and a pre-trained checkpoint served
/// under it — and vice versa.
const MEDIUM_BASE_SNAPSHOT_FILE_PINS: &[SnapshotFilePin] = &[
    SnapshotFilePin {
        relative: "model_config.json",
        bytes: 7_842,
        sha256: "27e2a299f0bda6ff742d3387398f929299575642f2c6a4d4c4f94830928fd0d5",
    },
    SnapshotFilePin {
        relative: "model.safetensors",
        bytes: 9_222_116_660,
        sha256: "c443fcc4d491475064cd0ff3eb92459b1e5f5060e86d96d016f048e528e24195",
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

/// The `-base` repositories are **ungated** on the Hub, unlike the three post-trained ones.
///
/// That changes acquisition, not terms: each base repository ships the same `LICENSE.md` (Stability
/// AI Community License) and `LICENSE_GEMMA.md` (Gemma Terms of Use) as its post-trained sibling, so
/// the rows below carry exactly the same restrictions. "No click-through" is not "no license".
pub const MUSIC_BASE_ROOT_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Stability-AI-Community",
    name: "Stability AI Community License",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-small-music-base/blob/eab5ceee5ad9c1ed38800aff30a8e49d1161c539/LICENSE.md",
    attribution: Some("Stable Audio 3 Small Music Base © Stability AI"),
    commercial_use: false,
    restriction: Some(
        "Use is governed by the Stability AI Community License, including its revenue threshold and prohibited-use terms.",
    ),
};

pub const MUSIC_BASE_GEMMA_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Gemma-Terms",
    name: "Gemma Terms of Use",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-small-music-base/blob/eab5ceee5ad9c1ed38800aff30a8e49d1161c539/LICENSE_GEMMA.md",
    attribution: Some("T5Gemma model weights © Google"),
    commercial_use: true,
    restriction: Some("Use is governed by the Gemma Terms of Use and Prohibited Use Policy."),
};

pub const SFX_BASE_ROOT_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Stability-AI-Community",
    name: "Stability AI Community License",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-small-sfx-base/blob/cc5ddb990e30daa68336ac61c140c37c7033ab7c/LICENSE.md",
    attribution: Some("Stable Audio 3 Small SFX Base © Stability AI"),
    commercial_use: false,
    restriction: Some(
        "Use is governed by the Stability AI Community License, including its revenue threshold and prohibited-use terms.",
    ),
};

pub const SFX_BASE_GEMMA_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Gemma-Terms",
    name: "Gemma Terms of Use",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-small-sfx-base/blob/cc5ddb990e30daa68336ac61c140c37c7033ab7c/LICENSE_GEMMA.md",
    attribution: Some("T5Gemma model weights © Google"),
    commercial_use: true,
    restriction: Some("Use is governed by the Gemma Terms of Use and Prohibited Use Policy."),
};

pub const MEDIUM_BASE_ROOT_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Stability-AI-Community",
    name: "Stability AI Community License",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-medium-base/blob/b32993f73c3bdc3864043a72d8032606bba737c8/LICENSE.md",
    attribution: Some("Stable Audio 3 Medium Base © Stability AI"),
    commercial_use: false,
    restriction: Some(
        "Use is governed by the Stability AI Community License, including its revenue threshold and prohibited-use terms.",
    ),
};

pub const MEDIUM_BASE_GEMMA_WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-Gemma-Terms",
    name: "Gemma Terms of Use",
    source_url: "https://huggingface.co/stabilityai/stable-audio-3-medium-base/blob/b32993f73c3bdc3864043a72d8032606bba737c8/LICENSE_GEMMA.md",
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

const MUSIC_BASE_WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[
    gen_core::WeightLicenseEntry {
        provider_id: MUSIC_BASE_MODEL_ID,
        component: None,
        license: MUSIC_BASE_ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: MUSIC_BASE_MODEL_ID,
        component: Some("root"),
        license: MUSIC_BASE_ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: MUSIC_BASE_MODEL_ID,
        component: Some("t5gemma"),
        license: MUSIC_BASE_GEMMA_WEIGHT_LICENSE,
    },
];

const SFX_BASE_WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[
    gen_core::WeightLicenseEntry {
        provider_id: SFX_BASE_MODEL_ID,
        component: None,
        license: SFX_BASE_ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: SFX_BASE_MODEL_ID,
        component: Some("root"),
        license: SFX_BASE_ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: SFX_BASE_MODEL_ID,
        component: Some("t5gemma"),
        license: SFX_BASE_GEMMA_WEIGHT_LICENSE,
    },
];

const MEDIUM_BASE_WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[
    gen_core::WeightLicenseEntry {
        provider_id: MEDIUM_BASE_MODEL_ID,
        component: None,
        license: MEDIUM_BASE_ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: MEDIUM_BASE_MODEL_ID,
        component: Some("root"),
        license: MEDIUM_BASE_ROOT_WEIGHT_LICENSE,
    },
    gen_core::WeightLicenseEntry {
        provider_id: MEDIUM_BASE_MODEL_ID,
        component: Some("t5gemma"),
        license: MEDIUM_BASE_GEMMA_WEIGHT_LICENSE,
    },
];

/// Every Stable Audio 3 weight-license row, in registration order.
///
/// The DiT, SAME pretransform, and learned conditioner all live inside the single
/// `model.safetensors` root artifact and are covered by the `root` row; the bundled T5Gemma stack
/// is a separately licensed component and carries its own row. Three rows per registration, not
/// four: medium's SAME-L is not a separate artifact, it is a namespace inside the same file.
///
/// Eighteen rows since sc-14546 (six registrations x three). The `-base` repositories are ungated on
/// the Hub, which is an acquisition difference and nothing else — they ship the same Stability
/// Community and Gemma license files, so their rows carry the same restrictions and the same
/// `commercial_use: false` on the root.
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
    MUSIC_BASE_WEIGHT_LICENSES[0],
    MUSIC_BASE_WEIGHT_LICENSES[1],
    MUSIC_BASE_WEIGHT_LICENSES[2],
    SFX_BASE_WEIGHT_LICENSES[0],
    SFX_BASE_WEIGHT_LICENSES[1],
    SFX_BASE_WEIGHT_LICENSES[2],
    MEDIUM_BASE_WEIGHT_LICENSES[0],
    MEDIUM_BASE_WEIGHT_LICENSES[1],
    MEDIUM_BASE_WEIGHT_LICENSES[2],
];

/// Build the descriptor for one registered variant.
///
/// All six checkpoints share the same mathematically active batch-CFG/APG/rescale path, so they
/// expose the same guidance and negative-prompt surface. They differ only in
/// `max_audio_duration_secs`.
///
/// # The `-base` variants advertise no extra capability, and that is the finding
///
/// sc-14546 was written expecting the base ids to set `supports_negative_prompt` and
/// `supports_guidance` to `true` as a *divergence* from the post-trained ids. Both flags have been
/// `true` for all three post-trained ids since sc-14544, pinned below with a comment saying so, and
/// the code agrees: [`crate::dit::StableAudio3Dit::forward_guided`] takes its batch-1 shortcut on
/// `cfg_scale == 1.0` alone, never on the checkpoint. Upstream's "these parameters have no effect on
/// post-trained checkpoints" is a statement about what post-training did to the *weights*, not about
/// a branch. Setting the flags per variant would therefore have meant *regressing* three shipped
/// descriptors to manufacture a difference that does not exist.
///
/// What genuinely differs is the operating point a request resolves to when it omits `sampler`,
/// `steps` or `guidance` — see [`Variant::recommended_sampler`], [`Variant::default_steps`] and
/// [`Variant::default_guidance`]. `Capabilities` has no field for any of the three, so that
/// difference is not advertised either; it is tracked with the other additive descriptor gaps as
/// `sc-15041`.
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
            // Audio→audio restyle (sc-14547) and bounded source editing (sc-14548), on all six
            // checkpoints. Advertising a kind is what makes `Capabilities::validate_request_audio`
            // *stop* rejecting the variant; both are consumed identically by every id, because the
            // seams are the shared SAME encoder, the shared partial-strength schedule, and the
            // shared `[inpaint_mask, inpaint_masked_input]` local conditioner — none of which is
            // variant-specific.
            conditioning: vec![
                ConditioningKind::ReferenceAudio,
                ConditioningKind::AudioEdit,
                // Multi-region editing (sc-14549) is its **own** kind, not a flag on the one
                // above, and advertising it here is the entire opt-in. A family that did not add
                // this line keeps rejecting multi-region requests as typed `Unsupported` through
                // the shared allowlist with no code of its own — which is why no capability flag
                // and no other descriptor in the repository needed to change.
                ConditioningKind::AudioEditRegions,
            ],
            // sc-14550. The full eight-type native adapter family — `lora`, `dora-rows`,
            // `dora-cols`, `bora` and their four `-xs` siblings — plus the legacy `dora` alias,
            // stacked in request order and folded at load time. See [`crate::adapters`].
            //
            // `supports_lokr` stays false and `AdapterKind::Lokr` is refused as typed
            // `Unsupported`: no published Stable Audio 3 adapter is a Kronecker factorization, and
            // accepting one would mean guessing a decomposition this family does not have.
            supports_lora: true,
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
            // sc-14548. Three modes, and `AudioEditMode::Cover` is deliberately **not** among them.
            //
            // Upstream Stable Audio 3 exposes three paths in total — text-to-audio, audio-to-audio
            // (`init_audio`), and inpaint — and every one of the six checkpoints pins its complete
            // conditioner surface in config: `global = [seconds_total]`,
            // `local = [inpaint_mask, inpaint_masked_input]`. There is no style/cover conditioner in
            // any of them, so `Cover` has nothing to map onto here; gen-core's own doc for that mode
            // names ACE-Step, which does ship it. Leaving it off removes no capability: the thing a
            // caller means by "cover" is a whole-clip restyle, and that is
            // `Conditioning::ReferenceAudio` above, advertised on all six ids with a retention
            // `strength` knob this surface does not have.
            //
            // `Inpaint` and `Repaint` are listed separately because gen-core defines them
            // separately, but for this family they are **aliases**: gen-core's distinction
            // (silence-substituted vs context-conditioned) describes ACE-Step's two native tasks,
            // and SA3 has one mechanism. They are byte-identical for the same request and seed —
            // structurally, because `pipeline::AudioEdit` carries no mode field at all.
            audio_edit_modes: vec![
                AudioEditMode::Inpaint,
                AudioEditMode::Repaint,
                AudioEditMode::Extend,
            ],
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            ..Default::default()
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

pub fn music_base_descriptor() -> ModelDescriptor {
    descriptor_for(Variant::SmallMusicBase)
}

pub fn sfx_base_descriptor() -> ModelDescriptor {
    descriptor_for(Variant::SmallSfxBase)
}

pub fn medium_base_descriptor() -> ModelDescriptor {
    descriptor_for(Variant::MediumBase)
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

/// The `Conditioning::ReferenceAudio` strength assumed when the caller omits it (sc-14547).
///
/// `0.1` retention, i.e. an init noise level of `0.9`, which is the operating point upstream's own
/// audio→audio entry point defaults to. Stated in **contract** units: see
/// [`reference_noise_level`] for why the two orientations are not interchangeable.
pub const DEFAULT_REFERENCE_STRENGTH: f32 = 0.1;

/// The accepted inclusive range for an explicit `Conditioning::ReferenceAudio` strength.
///
/// gen-core's request floor enforces finiteness only (it has no per-model range for this field), so
/// without this gate a `strength` of `-3.0` or `40.0` would reach `build_schedule` and surface as a
/// sampler-internal message about a value the caller never typed.
pub const REFERENCE_STRENGTH_RANGE: (f32, f32) = (0.0, 1.0);

/// Convert the contract's **retention** strength into the sampler's **init noise level**.
///
/// # This is the one place the two orientations meet, on purpose
///
/// Two same-named `strength` parameters with opposite meanings collide on this seam:
///
/// * `Conditioning::ReferenceAudio.strength` documents itself as mirroring the per-reference
///   img2img strength, and this workspace's img2img strength is mflux-derived and **retention**
///   oriented — it is the loop *start* index, so a higher value runs fewer steps and preserves
///   **more** of the source. (That is the inverse of diffusers' convention, which is why the
///   collision is easy to miss.)
/// * Stable Audio 3's sampler `strength` is upstream's `init_noise_level`: `1.0` replaces the
///   source with pure noise, `0.0` returns it untouched.
///
/// So the mapping is the complement, and the contract keeps the retention reading:
///
/// | contract `strength` | init noise level | result |
/// |---|---|---|
/// | `0.0` | `1.0` | pure generation; the source has no influence |
/// | `0.1` (default) | `0.9` | a loose restyle |
/// | `1.0` | `0.0` | the prepared source, returned without a single DiT forward |
///
/// A silent inversion here would still run, still emit plausible audio, and do the opposite of what
/// the caller asked, which is why the sign is gated by a mutation test at the contract endpoint
/// (`tests/reference_audio.rs`) rather than only by this table.
///
/// # A caller-visible consequence of the `1.0` row
///
/// At contract `strength = 1.0` the init noise level is `0.0`, which trips the sampler's
/// `skip_model` short circuit: the DiT never runs, so **no `Progress::Step` is ever emitted**. A
/// caller driving a progress bar sees zero steps and then `Progress::Decoding`. That is correct —
/// there is genuinely nothing to step through — but it is user-visible behaviour of a documented
/// endpoint, so it is stated here and in `docs/migration/SC_14547_REFERENCE_AUDIO_RESTYLE.md`
/// rather than left to be discovered.
pub fn reference_noise_level(strength: Option<f32>) -> f32 {
    1.0 - strength.unwrap_or(DEFAULT_REFERENCE_STRENGTH)
}

/// One request's resolved reference-audio conditioning, borrowed from the request (sc-14547).
#[derive(Debug, Clone, Copy)]
pub struct ResolvedReference<'a> {
    pub track: &'a AudioTrack,
    /// The contract-facing **retention** strength actually in force, explicit or defaulted.
    pub strength: f32,
    /// The sampler-facing init noise level, `1.0 - strength`.
    pub noise_level: f32,
}

/// Extract the single `Conditioning::ReferenceAudio` item, if the request carries one.
///
/// Returns the *first* item without complaint about duplicates: [`validate_request_for`] has already
/// rejected a second one, and every generation path validates first.
pub fn resolve_reference_audio(request: &GenerationRequest) -> Option<ResolvedReference<'_>> {
    request
        .conditioning
        .iter()
        .find_map(|conditioning| match conditioning {
            Conditioning::ReferenceAudio { audio, strength } => Some(ResolvedReference {
                track: audio,
                strength: strength.unwrap_or(DEFAULT_REFERENCE_STRENGTH),
                noise_level: reference_noise_level(*strength),
            }),
            _ => None,
        })
}

/// Build the pipeline-facing [`ReferenceAudio`] a request implies — the single site where the
/// contract's retention `strength` becomes the sampler's `init_noise_level` (sc-14547).
///
/// Extracted out of [`StableAudio3Generator::generate`] deliberately. `generate` needs
/// multi-gigabyte weights, so while this construction lived inside it the **field selection** here
/// (`noise_level`, not `strength`) was reachable only from the real-weight lane — and picking the
/// wrong field is the exact shape of the sign inversion this feature's entire risk budget is spent
/// on. It is now gated weight-free by `tests/reference_audio.rs`
/// `the_request_surface_hands_the_pipeline_the_converted_noise_level`.
pub fn reference_audio_for(request: &GenerationRequest) -> Option<ReferenceAudio<'_>> {
    resolve_reference_audio(request).map(|reference| ReferenceAudio {
        samples: &reference.track.samples,
        sample_rate: reference.track.sample_rate,
        channels: reference.track.channels,
        noise_level: reference.noise_level,
    })
}

/// How far a mode's implied timing may sit from the number the caller supplied: one output frame
/// (sc-14548).
///
/// `AudioParams::target_duration` and `TimeRegion` are `f32` seconds, and one 44.1 kHz frame is
/// `2.3e-5` s — well inside `f32`'s own resolution at 380 s, where an ulp is roughly `1.4` frames.
/// So "the extend region starts exactly at the source's end" cannot be an equality on seconds; it is
/// an equality on frames, expressed as this tolerance.
pub const EDIT_TIMING_TOLERANCE_SECS: f32 = 1.0 / SAMPLE_RATE as f32;

/// One request's resolved audio-edit conditioning, borrowed from the request (sc-14548).
///
/// Every field is on the **post-resample** 44.1 kHz timeline. `mode` is carried here and consumed
/// by [`audio_edit_for`]; nothing past that point can see it, which is what makes
/// `Inpaint` ≡ `Repaint` structural rather than merely tested.
#[derive(Debug, Clone)]
pub struct ResolvedEdit<'a> {
    pub track: &'a AudioTrack,
    pub mode: AudioEditMode,
    /// The source clip's duration once resampled to 44.1 kHz.
    pub source_duration_secs: f32,
    /// Every requested span, ends resolved, in the caller's own order (sc-14549). Non-empty for a
    /// well-formed request; the single-region carrier resolves to exactly **one** entry, which is
    /// what keeps both carriers on one path instead of a legacy path plus a parallel copy.
    ///
    /// Deliberately *not* normalized here: the union is computed once, in
    /// [`crate::pipeline::edit_geometry`], so validation errors can name the span the caller
    /// actually wrote rather than a merged one they never typed.
    pub regions: Vec<EditRegionSecs>,
    /// Whether the request used the multi-region carrier
    /// ([`Conditioning::AudioEditRegions`]). Read only by validation — for the mode gate and for
    /// error wording. The *geometry* is identical either way, which is the point.
    pub multi_region: bool,
    /// The output duration this mode implies: the source's own duration for an inpaint/repaint, the
    /// region's end for an extend.
    pub output_duration_secs: f32,
}

/// The source clip's duration once resampled onto the model's own 44.1 kHz timeline.
fn source_duration_secs(audio: &AudioTrack) -> f32 {
    let channels = (audio.channels as usize).max(1);
    let source_frames = audio.samples.len() / channels;
    crate::pipeline::resampled_frame_count(source_frames, audio.sample_rate.max(1)) as f32
        / SAMPLE_RATE as f32
}

/// Extract the request's single audio-edit carrier, if it has one (sc-14548, extended to the
/// multi-region carrier by sc-14549).
///
/// Total by construction — it resolves whatever it is given rather than reporting problems, because
/// [`synthesis_parameters`] needs the output duration and every generation path validates first.
/// `validate_audio_edit` is where a missing region, a misplaced extend boundary, a multi-region
/// `Extend`, or a conflicting `target_duration` is refused; the fallbacks below exist only so this
/// cannot panic on input that validation has already rejected.
///
/// # Both carriers resolve here, into the same shape
///
/// [`Conditioning::AudioEdit`] and [`Conditioning::AudioEditRegions`] differ only in how many spans
/// they carry and in whether an end may be elided. Past this function they are indistinguishable: a
/// single-region edit is the one-element list. Nothing downstream branches on which carrier arrived,
/// so there is no second code path for the multi-region case to drift away from — and, conversely,
/// no way for the multi-region case to quietly reuse only the machinery the single-region case
/// happened to exercise.
///
/// Returns the *first* carrier without complaint about duplicates, for the same reason
/// [`resolve_reference_audio`] does: two edits are already refused by `validate_audio_edit`, which
/// counts **both** kinds together.
pub fn resolve_audio_edit(request: &GenerationRequest) -> Option<ResolvedEdit<'_>> {
    request
        .conditioning
        .iter()
        .find_map(|conditioning| match conditioning {
            Conditioning::AudioEdit {
                audio,
                mode,
                region,
                ..
            } => {
                let source = source_duration_secs(audio);
                let start_secs = region.map_or(0.0, |region| region.start_secs);
                // The one place `end_secs: None` still means "to the end of the clip" — the
                // single-region shorthand, unchanged. The multi-region carrier refuses `None`
                // outright at the gen-core floor, because with an unordered region list "the end of
                // the clip" has no unambiguous owner.
                let end_secs = region.and_then(|region| region.end_secs).unwrap_or(source);
                let output_duration_secs = match mode {
                    AudioEditMode::Extend => end_secs,
                    _ => source,
                };
                Some(ResolvedEdit {
                    track: audio,
                    mode: *mode,
                    source_duration_secs: source,
                    regions: vec![EditRegionSecs {
                        start_secs,
                        end_secs,
                    }],
                    multi_region: false,
                    output_duration_secs,
                })
            }
            Conditioning::AudioEditRegions {
                audio,
                mode,
                regions,
                ..
            } => {
                let source = source_duration_secs(audio);
                Some(ResolvedEdit {
                    track: audio,
                    mode: *mode,
                    source_duration_secs: source,
                    regions: regions
                        .iter()
                        .map(|region| EditRegionSecs {
                            start_secs: region.start_secs,
                            // Every end is `Some` on this carrier (the gen-core floor refuses
                            // `None`); the fallback exists only so this stays total on input
                            // validation has already rejected.
                            end_secs: region.end_secs.unwrap_or(source),
                        })
                        .collect(),
                    multi_region: true,
                    // Multi-region is bounded-interior only — `Extend` is refused in
                    // `validate_audio_edit` — so the output is always exactly the source's length.
                    output_duration_secs: source,
                })
            }
            _ => None,
        })
}

/// Build the pipeline-facing [`AudioEdit`] a request implies — the single site where a gen-core
/// `AudioEditMode` plus an `Option<TimeRegion>` becomes a concrete `[start, end)` (sc-14548).
///
/// Extracted out of [`StableAudio3Generator::generate`] for the same reason
/// [`reference_audio_for`] was, and it is the same lesson: `generate` needs multi-gigabyte weights,
/// so a field selection made inside it is reachable only from a real-weight lane. The mistakes this
/// site can make are all silent — handing the pipeline the caller's `end_secs` where the source's
/// end belongs, swapping the two endpoints, or letting the mode leak past here — and none of them
/// changes a shape. It is gated weight-free by `tests/audio_edit.rs`
/// `the_request_surface_hands_the_pipeline_the_resolved_region`.
///
/// The mode is consumed **here**: `Extend` and `Inpaint`/`Repaint` differ only in which numbers come
/// out, and [`AudioEdit`] has no mode field for anything downstream to branch on.
pub fn audio_edit_for(request: &GenerationRequest) -> Option<AudioEdit<'_>> {
    resolve_audio_edit(request).map(|edit| AudioEdit {
        samples: &edit.track.samples,
        sample_rate: edit.track.sample_rate,
        channels: edit.track.channels,
        regions: edit.regions,
    })
}

/// Validate a request against one variant's own descriptor, without a snapshot.
///
/// `Generator::validate` needs a loaded generator, which needs multi-gigabyte weights; every
/// request-shaping rule this family owns is weight-free, so exposing the pair lets the whole
/// rejection surface be gated in the PR lane instead of only on a real-weight runner.
pub fn validate_request_for(variant: Variant, request: &GenerationRequest) -> gen_core::Result<()> {
    validate_request(variant, &descriptor_for(variant), request)
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
    reject_reference_and_edit_combination(model_id, request)?;
    validate_reference_audio(model_id, request)?;
    validate_audio_edit(variant, model_id, request)?;
    Ok(())
}

/// A source clip may play one role per request, never two (sc-14547 / sc-14548).
///
/// `ReferenceAudio` hands its clip to the sampler as `init_data`; `AudioEdit` hands its clip to the
/// DiT as masked local input and starts from pure noise. A request carrying both is asking for two
/// contradictory treatments of the source, so it is refused rather than silently resolved by
/// whichever branch runs first.
///
/// # This check only became reachable in sc-14548
///
/// sc-14547 shipped it as defence in depth and said so: `audio_edit_modes` was empty and `AudioEdit`
/// was not in `conditioning`, so `Capabilities::validate_request_audio`'s generic allowlist refused
/// any `AudioEdit` item on its own and this never fired. Advertising the kind is exactly what turns
/// it on, which is why it is hoisted here — called **once**, before both per-kind validators, so
/// the caller sees the same message regardless of which conditioning item is written first. Left
/// inside `validate_reference_audio` it would have been reachable only via the reference arm, and a
/// duplicate copy in the edit arm would have been dead code that could disagree.
fn reject_reference_and_edit_combination(
    model_id: &str,
    request: &GenerationRequest,
) -> gen_core::Result<()> {
    let has_reference = request
        .conditioning
        .iter()
        .any(|conditioning| matches!(conditioning, Conditioning::ReferenceAudio { .. }));
    // Both edit carriers count (sc-14549): the rule is about the *role* the source clip plays, and
    // a multi-region edit hands its clip to the DiT exactly as a single-region one does.
    let has_edit = request.conditioning.iter().any(|conditioning| {
        matches!(
            conditioning,
            Conditioning::AudioEdit { .. } | Conditioning::AudioEditRegions { .. }
        )
    });
    if has_reference && has_edit {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id}: reference audio and audio editing cannot be combined in one request"
        )));
    }
    Ok(())
}

/// Gate the bounded source-edit conditioning (sc-14548).
///
/// The generic floor above has already admitted the kind, rejected any mode outside
/// [`descriptor_for`]'s three (which is where `AudioEditMode::Cover` dies), enforced finiteness on
/// every float, and checked `start >= 0` / `end > start` in isolation. Everything that needs to know
/// what a *Stable Audio 3* edit is — arity, the strength refusal, the source clip's shape, where a
/// region may sit relative to the clip, and which output duration each mode implies — is this
/// family's own contract and lives here.
fn validate_audio_edit(
    variant: Variant,
    model_id: &str,
    request: &GenerationRequest,
) -> gen_core::Result<()> {
    // **Both** carriers, counted together (sc-14549). Two legacy edits, two multi-region edits, or
    // one of each are all the same mistake: `GenerationRequest::audio_edit()` and
    // `audio_edit_regions()` are each first-match-only and neither sees the other, so without this
    // the second carrier would be silently ignored rather than refused. Counting the kinds
    // separately would leave the *mixed* case — one of each — passing both counts at one apiece.
    let edits = request
        .conditioning
        .iter()
        .filter(|conditioning| {
            matches!(
                conditioning,
                Conditioning::AudioEdit { .. } | Conditioning::AudioEditRegions { .. }
            )
        })
        .count();
    if edits == 0 {
        return Ok(());
    }
    // Typed `Unsupported`, matching the reference arity refusal: "one edit per request" is a
    // statement about what this family can do. Note this is arity of *carriers*, not of regions —
    // several regions inside one `AudioEditRegions` are exactly what sc-14549 added.
    if edits > 1 {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id}: exactly one audio edit is supported per request, got {edits}; several \
             spans belong in the regions list of a single Conditioning::AudioEditRegions, which \
             regenerates them in one pass"
        )));
    }
    let strength = request
        .conditioning
        .iter()
        .find_map(|conditioning| match conditioning {
            Conditioning::AudioEdit { strength, .. }
            | Conditioning::AudioEditRegions { strength, .. } => Some(strength),
            _ => None,
        })
        .expect("a nonzero edit count finds one");
    // Refused, never ignored. Stable Audio 3's inpaint conditioner is a hard binary mask times the
    // encoded source, concatenated as channels: there is no scalar anywhere on this path a
    // "strength" could modulate, so honouring it would mean inventing a semantic. gen-core treats
    // `AudioEdit.strength` as a first-class float, so accepting and discarding it is the
    // "appears to work, does nothing" failure mode — the shipped idiom is to refuse
    // (`candle-gen-mage/src/lib.rs` does the same for per-reference strength inside its
    // conditioning walk). The one Stable Audio 3 scalar that *does* read like a strength is
    // `init_noise_level`, and that belongs to `Conditioning::ReferenceAudio`, named here so the
    // caller is pointed at the knob that exists rather than told "no".
    if let Some(strength) = strength {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id}: audio edit strength ({strength}) is not supported — this family's inpaint \
             conditioning is a hard binary mask with no blend parameter; use \
             Conditioning::ReferenceAudio, whose strength controls how much of the source is \
             retained across the whole clip"
        )));
    }
    let Some(edit) = resolve_audio_edit(request) else {
        unreachable!("a nonzero edit count resolves");
    };
    let track = edit.track;
    // The gen-core finiteness floor does not walk `AudioTrack::samples`, so the provider must.
    if track.samples.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: audio edit source must contain at least one frame"
        )));
    }
    if let Some(value) = track
        .samples
        .iter()
        .copied()
        .find(|value| !value.is_finite())
    {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: audio edit source contains the non-finite sample {value}"
        )));
    }
    if track.sample_rate == 0 {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: audio edit source must declare a non-zero sample rate"
        )));
    }
    if track.channels == 0 {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: audio edit source must declare at least one channel"
        )));
    }
    if !track.samples.len().is_multiple_of(track.channels as usize) {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: audio edit source has {} samples, not a whole number of {}-channel frames",
            track.samples.len(),
            track.channels
        )));
    }
    // Every advertised mode is a *bounded* region mode, so a region is mandatory on all three.
    // (`Cover`, the one gen-core mode that ignores it, is not advertised — see `descriptor_for`.)
    // On the single-region carrier that means `region: Some(..)`; on the multi-region carrier the
    // gen-core floor has already refused an empty list, and this catches it either way.
    let carries_region =
        match request
            .conditioning
            .iter()
            .find_map(|conditioning| match conditioning {
                Conditioning::AudioEdit { region, .. } => Some(region.is_some()),
                Conditioning::AudioEditRegions { regions, .. } => Some(!regions.is_empty()),
                _ => None,
            }) {
            Some(present) => present,
            None => unreachable!("a nonzero edit count finds one"),
        };
    if !carries_region {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: {:?} requires an audio edit region",
            edit.mode
        )));
    }
    let source = edit.source_duration_secs;
    // `Extend` is a **single-tail** operation and stays on the single-region carrier. Multi-region
    // means "regenerate these interior spans in one pass": there is exactly one tail, so a list of
    // them has no meaning, and the output length an extend implies is the region's end — which is
    // not well-defined when region order is not significant. Refused as a typed capability gap
    // rather than silently reinterpreted as an inpaint.
    if edit.multi_region && edit.mode == AudioEditMode::Extend {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id}: {:?} is a single-tail operation and is not supported on the multi-region \
             carrier; use Conditioning::AudioEdit with one region to continue past the source's \
             end",
            edit.mode
        )));
    }
    for (index, region) in edit.regions.iter().enumerate() {
        // Only the single-region carrier can reach the Extend arm (guarded above), so `regions`
        // holds exactly one entry there and this loop states the tail rule once.
        match edit.mode {
            AudioEditMode::Extend => {
                // The appended tail must start where the prepared source ends: an extend that
                // begins before the source's end is an inpaint the caller has mislabelled, and one
                // that begins after it asks the model to invent a gap it is given no context for.
                if (region.start_secs - source).abs() > EDIT_TIMING_TOLERANCE_SECS {
                    return Err(gen_core::Error::Msg(format!(
                        "{model_id}: an Extend region must start at the source's end ({source}s), \
                         got {}s",
                        region.start_secs
                    )));
                }
                // `end_secs = None` resolved to the source's end, which for an extend is a no-op.
                if region.end_secs <= source {
                    return Err(gen_core::Error::Msg(format!(
                        "{model_id}: an Extend region must end past the source ({source}s), got \
                         {}s — name the new total length as end_secs",
                        region.end_secs
                    )));
                }
            }
            _ => {
                if region.start_secs >= source {
                    return Err(gen_core::Error::Msg(format!(
                        "{model_id}: audio edit region {index} start {}s is at or past the \
                         source's end ({source}s); use AudioEditMode::Extend to continue past it",
                        region.start_secs
                    )));
                }
                if region.end_secs > source + EDIT_TIMING_TOLERANCE_SECS {
                    return Err(gen_core::Error::Msg(format!(
                        "{model_id}: audio edit region {index} end {}s is past the source's end \
                         ({source}s); use AudioEditMode::Extend to continue past it",
                        region.end_secs
                    )));
                }
            }
        }
        // A region narrower than one latent frame masks nothing: `[ceil(start/4096),
        // ceil(end/4096))` is empty, the local conditioner is the unedited source there, and the
        // render silently comes back identical to the input. Refuse instead of returning a no-op
        // that looks like success — checked on **every** requested region, so a caller who names
        // one usable span and one degenerate one is told about the degenerate one rather than
        // having it quietly disappear from the union.
        let (start_sample, end_sample) =
            crate::pipeline::edit_region_samples(region.start_secs, region.end_secs);
        let (start_latent, end_latent) =
            crate::pipeline::edit_region_latents(start_sample, end_sample);
        if end_latent <= start_latent {
            return Err(gen_core::Error::Msg(format!(
                "{model_id}: audio edit region {index} [{}s, {}s) spans no latent frame (positions \
                 {}..{}); it must cover at least one {}-sample frame (~{:.3}s)",
                region.start_secs,
                region.end_secs,
                start_latent,
                end_latent,
                crate::sampler::LATENT_DOWNSAMPLING,
                crate::sampler::LATENT_DOWNSAMPLING as f32 / SAMPLE_RATE as f32
            )));
        }
    }
    // The mode decides the output length, so a `target_duration` that disagrees is a request the
    // provider would have to silently overrule. Refuse it instead — and note the generic floor's
    // duration cap was applied to `target_duration`, which an extend need not carry, so the
    // resolved length is capped here too.
    let audio = request.audio.clone().unwrap_or_default();
    if let Some(target) = audio.target_duration {
        if (target - edit.output_duration_secs).abs() > EDIT_TIMING_TOLERANCE_SECS {
            return Err(gen_core::Error::Msg(format!(
                "{model_id}: audio.target_duration {target}s conflicts with the {:?} output length \
                 {}s implied by the source and region",
                edit.mode, edit.output_duration_secs
            )));
        }
    }
    let cap = variant.max_duration_secs();
    if edit.output_duration_secs > cap {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: the {:?} output length {}s exceeds the supported maximum {cap}s",
            edit.mode, edit.output_duration_secs
        )));
    }
    if edit.output_duration_secs < 1.0 / SAMPLE_RATE as f32 {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: the {:?} output length {}s must contain at least one 44.1 kHz frame",
            edit.mode, edit.output_duration_secs
        )));
    }
    Ok(())
}

/// Gate the audio→audio restyle conditioning (sc-14547).
///
/// The generic floor above admits `ReferenceAudio` (the descriptor now advertises it) and enforces
/// exactly one thing about it: that an explicit `strength` is finite. Everything a source clip can
/// be wrong about — arity, PCM shape, declared rate/channels, and the strength *range* — is this
/// family's own contract and is enforced here, before any of it reaches the resampler or
/// `build_schedule`, where the message would name an internal value the caller never typed.
fn validate_reference_audio(model_id: &str, request: &GenerationRequest) -> gen_core::Result<()> {
    let references = request
        .conditioning
        .iter()
        .filter(|conditioning| matches!(conditioning, Conditioning::ReferenceAudio { .. }))
        .count();
    if references == 0 {
        return Ok(());
    }
    // Typed `Unsupported`, not `Msg`: "one clip only" is a statement about what this family *can
    // do*, the same category as the `AudioEdit`-combination refusal, and this epic frames
    // conditioning refusals as typed. The malformed-clip rejections further down stay `Msg` —
    // those are about the caller's data, not about the model's capability. Both sides of that
    // split are asserted by type in `tests/reference_audio.rs`.
    if references > 1 {
        return Err(gen_core::Error::Unsupported(format!(
            "{model_id}: exactly one reference audio clip is supported, got {references}"
        )));
    }
    // The `ReferenceAudio` + `AudioEdit` combination is refused by
    // `reject_reference_and_edit_combination`, which `validate_request` calls **before** this and
    // before `validate_audio_edit`. sc-14547 had that check here, where it was unreachable (the
    // generic floor refused any `AudioEdit` item on its own because the kind was not advertised);
    // sc-14548 advertises the kind, so it is now live and was hoisted to one call site so both
    // orderings produce the same message.
    let Some(reference) = resolve_reference_audio(request) else {
        unreachable!("a nonzero reference count resolves");
    };
    let track = reference.track;
    if track.samples.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: reference audio must contain at least one frame"
        )));
    }
    if let Some(value) = track
        .samples
        .iter()
        .copied()
        .find(|value| !value.is_finite())
    {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: reference audio contains the non-finite sample {value}"
        )));
    }
    if track.sample_rate == 0 {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: reference audio must declare a non-zero sample rate"
        )));
    }
    if track.channels == 0 {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: reference audio must declare at least one channel"
        )));
    }
    if !track.samples.len().is_multiple_of(track.channels as usize) {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: reference audio has {} samples, not a whole number of {}-channel frames",
            track.samples.len(),
            track.channels
        )));
    }
    let strength = reference.strength;
    if !(REFERENCE_STRENGTH_RANGE.0..=REFERENCE_STRENGTH_RANGE.1).contains(&strength) {
        return Err(gen_core::Error::Msg(format!(
            "{model_id}: reference audio strength {strength} outside {}..={}",
            REFERENCE_STRENGTH_RANGE.0, REFERENCE_STRENGTH_RANGE.1
        )));
    }
    Ok(())
}

/// Resolve one request onto this variant's own runtime synthesis parameters.
///
/// Every "omitted" default is variant-bound (sc-14546): the post-trained ids resolve to upstream's
/// 8-step Pingpong at `cfg_scale = 1.0`, the `-base` ids to [`BASE_DEFAULT_STEPS`] Euler steps at
/// [`BASE_DEFAULT_GUIDANCE`]. Explicit request fields always win, on every id.
///
/// The sampler default now goes through [`Variant::recommended_sampler`] rather than the
/// `None | Some("pingpong") => Pingpong` literal it replaced. That literal happened to be right for
/// the three post-trained ids and is wrong for all three base ids, and it also meant
/// [`SamplerKind::recommended`] — the port of upstream's own rule — was dead code on the provider
/// path.
///
/// `pub` (and hidden) so the weight-free lane can assert the seam that decides this path's
/// geometry: `duration_secs` comes from `audio.target_duration`, or this variant's default, and
/// **never** from the reference clip's extent. Everything downstream — the adapted sample size, the
/// padded length `prepare_reference_pcm` conforms to, and the attention mask — is derived from that
/// one number, so a source-extent leak here would be invisible in `prepare_reference_pcm`'s own
/// output.
#[doc(hidden)]
pub fn synthesis_parameters(variant: Variant, request: &GenerationRequest) -> SynthesisParameters {
    let audio = request.audio.clone().unwrap_or_default();
    let method = request.guidance_method.as_deref();
    let guidance = Guidance {
        cfg_scale: request
            .guidance
            .unwrap_or(variant.default_guidance() as f32) as f64,
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
        // An `AudioEdit` **decides** the output length rather than merely constraining it: an
        // inpaint's output is exactly as long as its source, and an extend's is exactly its
        // region's end. `target_duration` is not consulted on that path — `validate_audio_edit`
        // has already refused any value that disagrees, so this is the same number, resolved from
        // the side that owns it. Without this an extend whose caller omitted `target_duration`
        // would render the variant's 120 s / 380 s default.
        duration_secs: resolve_audio_edit(request).map_or_else(
            || {
                audio
                    .target_duration
                    .unwrap_or_else(|| variant.default_duration_secs())
            },
            |edit| edit.output_duration_secs,
        ),
        steps: request.steps.unwrap_or(variant.default_steps() as u32) as usize,
        sampler: match request.sampler.as_deref() {
            None => variant.recommended_sampler(),
            Some("pingpong") => SamplerKind::Pingpong,
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
    /// The caller's adapter stack, in request order (sc-14550).
    ///
    /// Kept as *specs* rather than as a resolved plan because the plan's tensors must live on the
    /// compute device, and the device is not chosen until the lazy [`Self::pipeline`] path runs.
    /// `load_variant` still resolves the identical stack on the host first, so a malformed or
    /// mismatched adapter fails at **load** rather than at first generate.
    adapters: Vec<AdapterSpec>,
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
        let plan = resolve_adapter_plan(&layout, &self.adapters, &device)?;
        let pipeline = Arc::new(StableAudio3Pipeline::from_layout_with_adapters(
            &layout,
            self.variant.geometry(),
            &device,
            &plan,
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
    // Judged before the snapshot is opened: a `Lokr` request, or one carrying LTX's `pass_scales`
    // or Wan's `moe_expert`, is a statement about the request that reading the checkpoint cannot
    // change. `load_adapter` calls the identical function, so the two paths cannot drift.
    for adapter in &spec.adapters {
        crate::adapters::validate_spec_shape(adapter).map_err(crate::adapters::into_unsupported)?;
    }
    let layout = SnapshotLayout::from_dir(&root)?;
    crate::pipeline::validate_layout(&layout, expected.geometry())?;
    // No request is in flight on the load path, so there is no cancel flag to honour here.
    verify_snapshot_identity(expected, &layout.root, None)?;
    // Resolve the adapter stack against this checkpoint *now*, on the host, and throw the result
    // away. The plan the pipeline actually folds is rebuilt on the compute device, but doing the
    // work here is what makes a malformed, mistyped, pickle-format, or key-mismatched adapter fail
    // at `load_variant` — the moment the caller can still act on it — instead of at the first
    // generate, minutes later, behind a snapshot hash and a cold start.
    resolve_adapter_plan(&layout, &spec.adapters, &Device::Cpu)?;
    Ok(StableAudio3Generator {
        variant: expected,
        descriptor: descriptor_for(expected),
        root,
        adapters: spec.adapters.clone(),
        pipeline: Mutex::new(None),
        generation: Mutex::new(()),
    })
}

/// Resolve a `LoadSpec` adapter stack against one snapshot, or return the empty plan.
///
/// Separated from both call sites so the load-time refusal and the device-time fold read the
/// *same* function rather than two spellings of it — a divergence between them would be a load
/// that accepts an adapter the fold then cannot apply, discovered only under weights.
///
/// The target set comes from the checkpoint's safetensors header alone, so nothing is materialized
/// to decide whether an adapter matches.
fn resolve_adapter_plan(
    layout: &SnapshotLayout,
    specs: &[AdapterSpec],
    device: &Device,
) -> gen_core::Result<crate::adapters::AdapterPlan> {
    if specs.is_empty() {
        return Ok(crate::adapters::AdapterPlan::default());
    }
    let header = crate::weights::safetensors_shapes(&layout.weights_path)
        .map_err(crate::adapters::into_unsupported)?;
    let targets = crate::adapters::adaptable_targets(&header);
    crate::adapters::plan_for(specs, &targets, device).map_err(crate::adapters::into_unsupported)
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

pub fn load_music_base_generator(spec: &LoadSpec) -> gen_core::Result<StableAudio3Generator> {
    load_variant(Variant::SmallMusicBase, spec)
}

pub fn load_sfx_base_generator(spec: &LoadSpec) -> gen_core::Result<StableAudio3Generator> {
    load_variant(Variant::SmallSfxBase, spec)
}

pub fn load_medium_base_generator(spec: &LoadSpec) -> gen_core::Result<StableAudio3Generator> {
    load_variant(Variant::MediumBase, spec)
}

pub fn music_base_load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_variant(Variant::SmallMusicBase, spec)?))
}

pub fn sfx_base_load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_variant(Variant::SmallSfxBase, spec)?))
}

pub fn medium_base_load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_variant(Variant::MediumBase, spec)?))
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
        // Both conditioning paths resolve through a function that is reachable — and gated —
        // without weights, because `generate` is not: `reference_audio_for` carries the one
        // retention → `init_noise_level` conversion (sc-14547), `audio_edit_for` carries the one
        // mode + region → `[start, end)` resolution (sc-14548). The combination is refused in
        // `validate` above, so at most one of these is `Some`.
        // Each `..._for(request)` below is one token wide, and substituting either for `None` is a
        // silent, complete feature deletion that no weight-free lane can see, because this method
        // needs weights. So the resolved values travel bundled with the request facts they were
        // resolved from, and `synthesize_conditioned` re-checks the pair on the far side of the
        // boundary — an earlier revision checked them here and left the call's own argument list
        // unguarded, which is the same shape one seam further along. There is deliberately no
        // `match` selecting a synthesis method: that one entry point takes the whole receipt and
        // decides internally, so there is no arm for an edit to be routed into.
        let conditioning = ForwardedConditioning {
            request_has_reference: request
                .conditioning
                .iter()
                .any(|item| matches!(item, Conditioning::ReferenceAudio { .. })),
            request_has_edit: request.conditioning.iter().any(|item| {
                matches!(
                    item,
                    Conditioning::AudioEdit { .. } | Conditioning::AudioEditRegions { .. }
                )
            }),
            reference: reference_audio_for(request),
            edit: audio_edit_for(request),
        };
        let samples = pipeline.synthesize_conditioned(
            &request.prompt,
            request.negative_prompt.as_deref(),
            parameters,
            conditioning,
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

candle_audio::register_generators! {
    pub const MUSIC_BASE_REGISTRATION = music_base_descriptor => music_base_load
}

candle_audio::register_generators! {
    pub const SFX_BASE_REGISTRATION = sfx_base_descriptor => sfx_base_load
}

candle_audio::register_generators! {
    pub const MEDIUM_BASE_REGISTRATION = medium_base_descriptor => medium_base_load
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{
        AdapterKind, AdapterSpec, AudioParams, Conditioning, Image, Quant,
    };

    const VARIANTS: [Variant; 6] = Variant::ALL;

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
            // would be a tautology, so re-derive the mapping here: the smalls publish the crate
            // constant, medium publishes its own. Stated precisely, because a fix-cycle-2 review
            // caught this being overclaimed: this re-derives the variant -> *named constant*
            // mapping, so it catches `descriptor_for` wiring the wrong constant to an id, but it
            // does **not** pin the constants' values — a global edit to `MEDIUM_MAX_DURATION_SECS`
            // moves both sides together. The numeric literals (120 / 120 / 380) and the
            // `sample_size` reachability they imply are pinned in
            // `tests/conformance.rs::each_variant_advertises_a_cap_its_own_geometry_can_serve`,
            // which is the gate that fails on a value change.
            let expected_cap = match variant {
                Variant::SmallMusic
                | Variant::SmallSfx
                | Variant::SmallMusicBase
                | Variant::SmallSfxBase => MAX_DURATION_SECS,
                Variant::Medium | Variant::MediumBase => MEDIUM_MAX_DURATION_SECS,
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
            //
            // sc-14546 re-reads this as the tripwire it is. The story asked for the `-base` ids to
            // carry these two flags as a *divergence* from the post-trained ids; they have been
            // `true` for every post-trained id since sc-14544, and `dit.rs`'s batch-1 shortcut keys
            // on `cfg_scale == 1.0` and nothing else. Implementing that literally would mean
            // flipping these three assertions to `false` for the post-trained ids — regressing
            // shipped descriptors to invent a difference. If this loop starts failing, believe it.
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
        assert_eq!(medium_descriptor().id, "stable_audio_3_medium");
        assert_eq!(
            music_base_descriptor().id,
            "stable_audio_3_small_music_base"
        );
        assert_eq!(sfx_base_descriptor().id, "stable_audio_3_small_sfx_base");
        assert_eq!(medium_base_descriptor().id, "stable_audio_3_medium_base");
        assert_eq!((REGISTRATION.descriptor)().id, MODEL_ID);
        assert_eq!((SFX_REGISTRATION.descriptor)().id, SFX_MODEL_ID);
        assert_eq!((MEDIUM_REGISTRATION.descriptor)().id, MEDIUM_MODEL_ID);
        assert_eq!(
            (MUSIC_BASE_REGISTRATION.descriptor)().id,
            MUSIC_BASE_MODEL_ID
        );
        assert_eq!((SFX_BASE_REGISTRATION.descriptor)().id, SFX_BASE_MODEL_ID);
        assert_eq!(
            (MEDIUM_BASE_REGISTRATION.descriptor)().id,
            MEDIUM_BASE_MODEL_ID
        );
    }

    /// The identity gate's two repository-shaped fields must not collapse back into one.
    ///
    /// This is the highest-probability regression in sc-14546: `Variant::geometry` used to set
    /// `expected_repo: self.hub_repo()`, and `validate_layout` compared that to the snapshot's
    /// declared conditioner `repo_id`. Every base config declares its *post-trained sibling's*
    /// repository, so a base variant whose gate value is its own `-base` repo rejects every base
    /// snapshot that will ever exist. The inverse mistake — deleting the check — reopens the two
    /// architecturally identical smalls to each other.
    ///
    /// Weight-free and discriminating in both directions: it asserts the fields differ exactly on
    /// the three base variants and agree exactly on the three post-trained ones.
    #[test]
    fn base_variants_declare_their_post_trained_siblings_conditioner_repo() {
        for variant in VARIANTS {
            let geometry = variant.geometry();
            assert_eq!(geometry.hub_repo, variant.hub_repo());
            assert_eq!(
                geometry.expected_conditioner_repo,
                variant.conditioner_repo()
            );
            if variant.is_base() {
                assert_ne!(
                    variant.conditioner_repo(),
                    variant.hub_repo(),
                    "{}: a base snapshot declares its post-trained sibling's repo_id, so the gate \
                     value must NOT be the base repository",
                    variant.model_id()
                );
                assert!(variant.hub_repo().ends_with("-base"));
                assert!(!variant.conditioner_repo().ends_with("-base"));
            } else {
                assert_eq!(
                    variant.conditioner_repo(),
                    variant.hub_repo(),
                    "{}: a post-trained snapshot declares its own repo_id",
                    variant.model_id()
                );
            }
        }
        // Each base variant shares its gate value with exactly its own post-trained sibling.
        for (base, post) in [
            (Variant::SmallMusicBase, Variant::SmallMusic),
            (Variant::SmallSfxBase, Variant::SmallSfx),
            (Variant::MediumBase, Variant::Medium),
        ] {
            assert_eq!(base.conditioner_repo(), post.conditioner_repo());
            assert_ne!(base.hub_repo(), post.hub_repo());
            assert_ne!(base.hub_revision(), post.hub_revision());
            // ...and the objective is what separates them, in both directions.
            assert_ne!(base.objective(), post.objective());
            assert_eq!(
                base.objective(),
                crate::config::DiffusionObjective::RectifiedFlow
            );
            assert_eq!(
                post.objective(),
                crate::config::DiffusionObjective::RfDenoiser
            );
        }
        // The two small bases are separated from each other by the gate value alone, exactly as the
        // two post-trained smalls are.
        assert_ne!(
            Variant::SmallMusicBase.conditioner_repo(),
            Variant::SmallSfxBase.conditioner_repo()
        );
    }

    /// What each pair actually discriminates on, asserted rather than described.
    ///
    /// The medium pair is the sharp one: identical inventory, identical `sample_size`, identical
    /// declared `repo_id`. If `expected_objective` were ever dropped from `VariantGeometry`, this
    /// fails here — weight-free — instead of in a real-weight lane.
    #[test]
    fn each_post_trained_base_pair_has_at_least_one_config_level_discriminator() {
        // Smalls: `sample_size` differs *and* the objective differs.
        for (base, post) in [
            (Variant::SmallMusicBase, Variant::SmallMusic),
            (Variant::SmallSfxBase, Variant::SmallSfx),
        ] {
            assert_eq!(base.shape().sample_size, SMALL_BASE_MAX_SAMPLE_SIZE);
            assert_eq!(post.shape().sample_size, SMALL_MAX_SAMPLE_SIZE);
            assert_ne!(base.shape().sample_size, post.shape().sample_size);
            // Everything else about the graph is the same, which is the point.
            assert_eq!(base.shape().total_keys, post.shape().total_keys);
            assert_eq!(base.shape().embed_dim, post.shape().embed_dim);
            assert_eq!(base.shape().depth, post.shape().depth);
            assert_eq!(base.shape().differential, post.shape().differential);
        }
        // Medium: the shapes are *identical*, so the objective is the entire config-level gate.
        assert_eq!(
            Variant::MediumBase.shape(),
            Variant::Medium.shape(),
            "medium and medium-base agree on every architectural field; if that ever stops being \
             true the doc on `Variant::MediumBase` has to change with it"
        );
        assert_eq!(
            Variant::MediumBase.conditioner_repo(),
            Variant::Medium.conditioner_repo()
        );
        assert_ne!(
            Variant::MediumBase.geometry(),
            Variant::Medium.geometry(),
            "the only thing left is `expected_objective` — without it the two geometries are equal \
             and the SHA-256 pin is the sole defence"
        );
    }

    /// Every omitted request field resolves to this variant's own operating point (sc-14546).
    ///
    /// Discriminating: the post-trained arm fails if the base numbers leak into it, the base arm
    /// fails if the old hard-coded `8 / 1.0 / Pingpong` survives, and the explicit-override arm
    /// fails if the defaults were made unconditional.
    #[test]
    fn omitted_request_fields_resolve_to_the_variants_own_operating_point() {
        for variant in VARIANTS {
            let resolved = synthesis_parameters(variant, &request());
            let (steps, guidance, sampler) = if variant.is_base() {
                (
                    BASE_DEFAULT_STEPS,
                    BASE_DEFAULT_GUIDANCE,
                    SamplerKind::Euler,
                )
            } else {
                (DEFAULT_STEPS, DEFAULT_GUIDANCE, SamplerKind::Pingpong)
            };
            assert_eq!(
                resolved.steps,
                steps,
                "{} default steps",
                variant.model_id()
            );
            assert_eq!(
                resolved.guidance.cfg_scale,
                guidance,
                "{} default guidance",
                variant.model_id()
            );
            assert_eq!(
                resolved.sampler,
                sampler,
                "{} default sampler",
                variant.model_id()
            );

            // Euler must not allocate Pingpong's per-step random tensors. This is the property the
            // omitted-sampler default is worth having, stated as an executed assertion rather than
            // as a claim about the solver.
            let latent_elements = 256 * 16;
            let estimate =
                crate::sampler::resource_estimate(resolved.sampler, steps, 1, latent_elements)
                    .unwrap();
            if variant.is_base() {
                assert_eq!(estimate.full_latent_noise_draws, 0);
                assert_eq!(estimate.seeded_noise_device_elements, 0);
            } else {
                assert_eq!(estimate.full_latent_noise_draws, steps);
                assert_eq!(
                    estimate.seeded_noise_device_elements,
                    latent_elements * steps
                );
            }
            assert_eq!(estimate.model_calls, steps);

            // Explicit fields win on every id, base included.
            let mut explicit = request();
            explicit.steps = Some(3);
            explicit.guidance = Some(2.5);
            explicit.sampler = Some("dpmpp".into());
            let resolved = synthesis_parameters(variant, &explicit);
            assert_eq!(resolved.steps, 3);
            assert_eq!(resolved.guidance.cfg_scale, 2.5);
            assert_eq!(resolved.sampler, SamplerKind::Dpmpp);
            // ...including asking a base id for the post-trained default explicitly.
            let mut pingpong = request();
            pingpong.sampler = Some("pingpong".into());
            assert_eq!(
                synthesis_parameters(variant, &pingpong).sampler,
                SamplerKind::Pingpong
            );
        }
        // The defaults must actually differ across the split, or none of the above discriminates.
        assert_ne!(DEFAULT_STEPS, BASE_DEFAULT_STEPS);
        assert_ne!(DEFAULT_GUIDANCE, BASE_DEFAULT_GUIDANCE);
    }

    /// `SamplerKind::recommended` is the frozen-upstream rule; sc-14546 puts it on the provider path.
    ///
    /// The `expect` inside [`Variant::recommended_sampler`] is only sound while no registered
    /// variant declares the unreachable `v` objective, so that is asserted rather than assumed.
    #[test]
    fn recommended_sampler_matches_the_frozen_objective_rule() {
        for variant in VARIANTS {
            assert_ne!(variant.objective(), crate::config::DiffusionObjective::V);
            assert_eq!(
                variant.recommended_sampler(),
                SamplerKind::recommended(variant.objective()).unwrap()
            );
            assert_eq!(
                variant.recommended_sampler(),
                if variant.is_base() {
                    SamplerKind::Euler
                } else {
                    SamplerKind::Pingpong
                }
            );
        }
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

    /// No two registered checkpoints may share a root digest, and every pin set must be complete.
    ///
    /// sc-14546 makes this worth generalising: four of the six roots are exactly `2,270,384,940`
    /// bytes and the remaining two are exactly `9,222,116,660`, so byte length authenticates
    /// nothing. The T5Gemma stack is byte-identical across all six, which is why it is asserted
    /// *equal* rather than distinct.
    #[test]
    fn every_registered_checkpoint_has_a_distinct_root_digest_and_a_complete_pin_set() {
        let by_name = |pins: &'static [SnapshotFilePin], name: &str| {
            pins.iter()
                .find(|pin| pin.relative == name)
                .unwrap_or_else(|| panic!("missing pin {name}"))
        };
        let gemma = by_name(
            Variant::SmallMusic.pins(),
            "t5gemma-b-b-ul2/model.safetensors",
        )
        .sha256;
        let mut roots = Vec::new();
        let mut configs = Vec::new();
        for variant in VARIANTS {
            let pins = variant.pins();
            assert_eq!(pins.len(), 6, "{}", variant.model_id());
            for relative in [
                "model_config.json",
                "model.safetensors",
                "t5gemma-b-b-ul2/config.json",
                "t5gemma-b-b-ul2/model.safetensors",
                "t5gemma-b-b-ul2/tokenizer.json",
                "t5gemma-b-b-ul2/tokenizer.model",
            ] {
                let pin = by_name(pins, relative);
                assert_eq!(pin.sha256.len(), 64, "{}", variant.model_id());
            }
            assert_eq!(
                by_name(pins, "t5gemma-b-b-ul2/model.safetensors").sha256,
                gemma,
                "{}: the bundled T5Gemma is byte-identical across all six SA3 repositories",
                variant.model_id()
            );
            roots.push(by_name(pins, "model.safetensors").sha256);
            configs.push(by_name(pins, "model_config.json").sha256);
        }
        let unique_roots: std::collections::BTreeSet<_> = roots.iter().collect();
        assert_eq!(
            unique_roots.len(),
            VARIANTS.len(),
            "two registrations share a root SHA-256"
        );
        let unique_configs: std::collections::BTreeSet<_> = configs.iter().collect();
        assert_eq!(unique_configs.len(), VARIANTS.len());
        // The byte lengths deliberately do NOT authenticate: four roots share one length, two share
        // another. Committed so nobody "optimises" the hash away.
        let lengths: std::collections::BTreeSet<u64> = VARIANTS
            .iter()
            .map(|variant| by_name(variant.pins(), "model.safetensors").bytes)
            .collect();
        assert_eq!(lengths.len(), 2, "{lengths:?}");
    }

    #[test]
    fn every_variant_contributes_composite_and_component_license_rows() {
        // Six registered variants x (composite, root, t5gemma). sc-14545 added the medium trio;
        // sc-14546 the three `-base` trios. The base repositories are ungated on the Hub but ship
        // the same Stability Community and Gemma license files, so the rows are not weaker.
        assert_eq!(WEIGHT_LICENSES.len(), 18);
        assert_eq!(VARIANTS.len(), 6);
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
            medium_load,
            music_base_load,
            sfx_base_load,
            medium_base_load,
        ] {
            assert!(loader(&LoadSpec::new(WeightsSource::File(missing.clone()))).is_err());

            let dense = || LoadSpec::new(WeightsSource::Dir(missing.clone()));
            let mut specs = Vec::new();
            specs.push(dense().with_quant(Quant::Q4));

            let mut precision = dense();
            precision.precision = Precision::Fp32;
            specs.push(precision);

            // sc-14550: a `Lora` adapter is no longer refused *as a shape* — the family is
            // supported — but `Lokr` still is, and it is refused **before** the snapshot is opened,
            // which is what this loop tests. The accepted-`Lora` half is gated weight-free in
            // `tests/adapters.rs`; here the point is only that the pre-snapshot guard still names
            // the kinds this provider cannot serve.
            specs.push(dense().with_adapters(vec![AdapterSpec::new(
                missing.clone(),
                1.0,
                AdapterKind::Lokr,
            )]));
            specs.push(dense().with_adapters(vec![AdapterSpec::new(
                missing.clone(),
                1.0,
                AdapterKind::Lora,
            )
            .with_pass_scales(vec![1.0, 0.5])]));
            specs.push(dense().with_adapters(vec![AdapterSpec::new(
                missing.clone(),
                1.0,
                AdapterKind::Lora,
            )
            .with_moe_expert(gen_core::MoeExpert::High)]));
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
        // `ThreadId`, not `Thread::name()`: under libtest the thread name *is* the test path
        // (`model::tests::pinned_file_authentication_rejects_size_and_payload_drift`), and `:` is
        // an illegal character in a Windows path component, so `create_dir_all` returned
        // `InvalidFilename` (os error 123) on the Windows CUDA runner the moment this crate's unit
        // tests were first run there. `weights.rs` already uses the id for the same reason.
        let root = std::env::temp_dir().join(format!(
            "sa3-provider-pin-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
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
