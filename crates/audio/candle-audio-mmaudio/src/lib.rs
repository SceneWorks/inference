//! # candle-audio-mmaudio
//!
//! Shared **MMAudio** video→audio provider crate for the SceneWorks Candle audio lane
//! (epic sc-12833, `docs/architecture/audio-backend-strategy.md`). This slice (**sc-13438**)
//! establishes the crate and ports MMAudio's **Synchformer synchronization encoder** — the
//! frame-aligned visual conditioner — natively onto the workspace's pinned candle revision.
//! **sc-13437** adds the [`clip`] module: MMAudio's semantic conditioner, the **DFN5B-CLIP
//! ViT-H/14-384** open_clip encoder (visual → 1024-d per-frame features; 77-token text tower →
//! per-token last-hidden-state), parity-verified against `open_clip`. Later MMAudio slices add the
//! flow-matching DiT, the VAE/vocoder, and the generator that registers into
//! `candle-audio-catalog`. Nothing is registered here (model-internal encoders), mirroring how
//! `candle-audio-moss-tts-realtime` stayed unregistered until its codec landed.
//!
//! ## What Synchformer is, and what MMAudio actually uses
//!
//! Synchformer (Iashin et al., *"Synchformer: Efficient Synchronization from Sparse Cues"*, arXiv
//! 2310.16043; repo `v-iashin/Synchformer`, MIT) is a segment-level audio-visual synchronization
//! model. **MMAudio uses only its visual branch** — the `vfeat_extractor` — as a frozen feature
//! extractor. MMAudio's `mmaudio/ext/synchformer/synchformer.py` instantiates *only*
//! `self.vfeat_extractor = MotionFormer(extract_features=True, factorize_space_time=True,
//! agg_space_module='TransformerEncoderLayer', agg_time_module='torch.nn.Identity',
//! add_global_repr=False)` and, at load time, **discards every non-`vfeat_extractor.` key** (the
//! audio AST branch `afeat_extractor`, the projections `vproj`/`aproj`, and the AV-sync
//! `transformer`). So the faithful port target is exactly the MotionFormer visual path ending at
//! 768-d features — reconstructed here from the vendored source, not guessed.
//!
//! ## The reconstructed module graph (verified against MMAudio source)
//!
//! MotionFormer is a **ViT-B/16 with divided (factorized) space-time attention**, configured by
//! MMAudio's `divided_224_16x4.yaml`. Dims: `embed_dim=768`, `depth=12`, `heads=12`, `mlp_ratio=4`
//! (hidden 3072), `LayerNorm(eps=1e-6)`, pre-norm. See [`config`] for every constant with its YAML
//! key. Forward, for input segments `(S, C=3, T=16, H=224, W=224)`:
//!
//! 1. **3D patch embed** ([`blocks::PatchEmbed3d`]): a non-overlapping `Conv3d`, kernel = stride =
//!    `(z=2, 16, 16)`, `3→768`. `T=16 → 8` temporal × `14×14` spatial = **1568 patch tokens**.
//!    Because stride == kernel, it is implemented as an exact patchify + linear projection (the
//!    pinned candle revision exposes no `Conv3d`), preserving the Conv3d weight's `(c,z,h,w)`
//!    element order.
//! 2. **CLS + separate positional embeddings** ([`sync`]): prepend one learnable `cls_token`; add
//!    `total = tile(spatial_pos, T) + repeat_interleave(temp_embed, 196)` with the CLS position
//!    prepended (`VIT.POS_EMBED == "separate"`). Token order is temporal-major (`t·196 + h·14 + w`).
//! 3. **12 × [`blocks::DividedSpaceTimeBlock`]**: each does **temporal** attention
//!    (`norm3 → timeattn`, `b (f n) d -> (b n) f d`) then **spatial** attention
//!    (`norm1 → attn`, `b (f n) d -> (b f) n d`) then **MLP** (`norm2 → fc1 → GELU(erf) → fc2`),
//!    each a residual add, in that order. The CLS token attends to — and is attended by — every
//!    space-time token in both passes ([`blocks::DividedAttention`]).
//! 4. **Final LayerNorm**, drop CLS → `(S, 1568, 768)`, restore to `(S, 768, t=8, h=14, w=14)`.
//! 5. **Spatial aggregation** ([`agg::SpatialAggLayer`], MMAudio's `SpatialTransformerEncoderLayer`
//!    — a pre-norm `nn.TransformerEncoderLayer` with a CLS token): per frame, CLS-pool the `14×14`
//!    grid → `(S, t=8, 768)`.
//! 6. **Temporal aggregation is `Identity`** in MMAudio's config, so the 8 temporal tokens per
//!    segment are **kept** (not collapsed to one vector — that Identity swap is exactly how MMAudio
//!    retains temporal resolution for frame-aligned conditioning). Output: **`(S, 8, 768)`**.
//!
//! ## The 24-vs-25 fps question — resolved to **25**
//!
//! The README/paper mention ~24-25 fps loosely. The operational rate is **25**: MMAudio's
//! `mmaudio/model/sequence_config.py` sets `sync_frame_rate = 25`, the Synchformer data reencode
//! targets `vfps=25`, and the arithmetic is decisive — one segment is `NUM_FRAMES/fps = 16/25 =
//! 0.64 s` exactly, whereas `0.64 × 24 = 15.36` is non-integer. 25 fps is the only rate consistent
//! with the 16-frame segment. Recorded in [`config::SYNC_FRAME_RATE`]. Segments overlap 50%
//! (`sync_step_size = 8`); a typical 8 s / 200-frame clip → 24 segments → `(24, 8, 768)`.
//!
//! ## Preprocessing
//!
//! [`preprocess`]: shorter-edge → 224 (CatmullRom, approximating the reference's torchvision
//! bicubic — not a bit-exact match), center-crop 224², scale to `[0,1]`, normalize
//! mean/std = 0.5 → `[-1, 1]` (`DATA.MEAN`/`STD`, **not** ImageNet stats), then window into
//! overlapping 16-frame segments.
//!
//! ## Weights + license
//!
//! `hkchengrex/MMAudio` @ [`model::HUB_REVISION`], file
//! [`model::WEIGHTS_PATH`] (`ext_weights/synchformer_state_dict.pth`, ~907 MB). The `.pth` holds the
//! full Synchformer state dict; only the `vfeat_extractor.*` sub-tree is loaded. License: **MIT**
//! (© 2024 Vladimir Iashin), recorded in [`model::WEIGHT_LICENSE`] with a training-data-provenance
//! restriction note.

pub use candle_audio;
pub use candle_audio::gen_core;
pub use candle_audio::{AudioError, Result};

pub mod agg;
pub mod bigvgan;
pub mod blocks;
pub mod clip;
pub mod config;
pub mod generator;
pub mod mmdit;
pub mod model;
pub mod output;
pub mod preprocess;
pub mod sync;
pub mod vae;

pub use clip::DfnClipEncoder;
pub use mmdit::{Conditions, Config as MmDitConfig, MmAudioDit};
pub use model::{
    load, load_from_pth, resolve_pinned_weights, HUB_REPO, HUB_REVISION, MODEL_ID, WEIGHTS_PATH,
    WEIGHT_LICENSE, WEIGHT_LICENSE_ENTRY,
};
pub use sync::SynchformerVisualEncoder;

pub use bigvgan::BigVganVocoder;
pub use output::{AudioDecoder16k, BIGVGAN_MODEL_ID, VAE_MODEL_ID};
pub use vae::MelVaeDecoder;

// NB: `generator::load` is intentionally NOT re-exported at the crate root — the crate already
// re-exports `model::load` (the Synchformer loader). Reach the shipping generator's loader via
// `candle_audio_mmaudio::generator::load` (the registration constant carries it for the catalog).
pub use generator::{
    resolve_pinned_snapshot, MmAudioGenerator, MmAudioPipeline, MAX_DURATION_SECS,
    MODEL_ID as GENERATOR_ID, REGISTRATION, SAMPLE_RATE as GENERATOR_SAMPLE_RATE,
};

/// Add the shipping MMAudio video→audio generator (`mmaudio_small_16k`) to an explicit audio registry
/// builder (catalog composition, sc-12843).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_generator(generator::REGISTRATION)
}

/// Build the complete explicit MMAudio provider catalog (this crate's own surface).
pub fn provider_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    register_providers(gen_core::ProviderRegistryBuilder::new()).build()
}

/// This crate's **per-component** model-weight-license entries (sc-13332) — one row per ported
/// checkpoint: the Synchformer visual encoder (sc-13438), the DFN5B-CLIP ViT-H/14 encoder (sc-13437),
/// the MM-DiT flow-matching generator `mmaudio_small_16k` (sc-13439), and the 16k output path's
/// mel-VAE + BigVGAN (sc-13440). This is the detailed provenance record (each checkpoint's own SPDX /
/// attribution / restriction). The **catalog** aggregates the single composite
/// [`SHIPPED_WEIGHT_LICENSES`] entry instead (its ship-gate keys one license row per *registered*
/// provider id, and only `mmaudio_small_16k` registers).
pub const WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] = &[
    model::WEIGHT_LICENSE_ENTRY,
    clip::WEIGHT_LICENSE_ENTRY,
    mmdit::WEIGHT_LICENSE_ENTRY,
    output::VAE_WEIGHT_LICENSE_ENTRY,
    output::BIGVGAN_WEIGHT_LICENSE_ENTRY,
];

/// The **catalog-facing** weight-license surface: exactly one composite row keyed by the shipping
/// provider id `mmaudio_small_16k` (sc-12843). `candle-audio-catalog::weight_licenses()` folds this
/// into the model-licenses manifest — one entry per registered provider, as its ship-gate requires.
/// The composite carries the *intersection* (strictest) of the five component licenses
/// ([`WEIGHT_LICENSES`]); see [`generator::WEIGHT_LICENSE`] for the rationale.
pub const SHIPPED_WEIGHT_LICENSES: &[gen_core::WeightLicenseEntry] =
    &[generator::WEIGHT_LICENSE_ENTRY];
