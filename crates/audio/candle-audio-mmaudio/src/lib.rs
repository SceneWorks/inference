//! # candle-audio-mmaudio
//!
//! Shared **MMAudio** video→audio provider crate for the SceneWorks Candle audio lane
//! (epic sc-12833, `docs/architecture/audio-backend-strategy.md`). This slice (**sc-13438**)
//! establishes the crate and ports MMAudio's **Synchformer synchronization encoder** — the
//! frame-aligned visual conditioner — natively onto the workspace's pinned candle revision.
//! **sc-13437** adds the [`clip`] module: MMAudio's semantic conditioner, the **DFN5B-CLIP
//! ViT-H/14-384** open_clip encoder (visual → 1024-d per-frame features; 77-token text tower →
//! per-token last-hidden-state), parity-verified against `open_clip`. Later slices add the
//! flow-matching [`mmdit`] DiT (sc-13439), the [`vae`]/[`bigvgan`] output path (sc-13440), and the
//! shipping [`generator`] that registers **`mmaudio_small_16k`** into `candle-audio-catalog`
//! (sc-12843). **sc-13441** adds the 44.1 kHz quality-ceiling path: the `large_44k_v2` MM-DiT preset
//! ([`mmdit::Config::large_44k_v2`], 1.03B), the 44k mel-VAE ([`vae::Config::vae_44k`], 40-d latent /
//! 128-band mel) + the external **NVIDIA BigVGAN v2** vocoder
//! ([`bigvgan::Config::bigvgan_v2_44khz_128band_512x`]), and the sibling [`generator_44k`] that
//! registers **`mmaudio_large_44k`**.
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
//! (© 2024 Vladimir Iashin), recorded in [`model::COMPONENT_LICENSE`].

pub use candle_audio;
pub use candle_audio::gen_core;
pub use candle_audio::{AudioError, Result};

pub mod agg;
pub mod bigvgan;
pub mod blocks;
pub mod clip;
pub mod config;
pub mod generator;
pub mod generator_44k;
pub mod mmdit;
pub mod model;
pub mod output;
pub mod preprocess;
pub mod sync;
pub mod vae;

pub use clip::DfnClipEncoder;
pub use mmdit::{Conditions, Config as MmDitConfig, MmAudioDit};
pub use model::{
    load, load_from_pth, COMPONENT_LICENSE, HUB_REPO, HUB_REVISION, MODEL_ID, WEIGHTS_PATH,
};
pub use sync::SynchformerVisualEncoder;

pub use bigvgan::BigVganVocoder;
pub use output::{AudioDecoder16k, BIGVGAN_MODEL_ID, VAE_MODEL_ID};
pub use vae::MelVaeDecoder;

// NB: `generator::load` is intentionally NOT re-exported at the crate root — the crate already
// re-exports `model::load` (the Synchformer loader). Reach the shipping generator's loader via
// `candle_audio_mmaudio::generator::load` (the registration constant carries it for the catalog).
pub use generator::{
    MmAudioGenerator, MmAudioPipeline, MAX_DURATION_SECS, MODEL_ID as GENERATOR_ID, REGISTRATION,
    SAMPLE_RATE as GENERATOR_SAMPLE_RATE,
};

pub use generator_44k::{
    MmAudio44kPipeline, MmAudioLarge44kGenerator, MODEL_ID as GENERATOR_ID_44K,
    REGISTRATION as REGISTRATION_44K, SAMPLE_RATE as GENERATOR_SAMPLE_RATE_44K,
};

pub use output::AudioDecoder44k;

/// Add the shipping MMAudio video→audio generators to an explicit audio registry builder (catalog
/// composition): the 16 kHz `mmaudio_small_16k` (sc-12843) then the 44.1 kHz quality-ceiling sibling
/// `mmaudio_large_44k` (sc-13441), in that stable order.
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(generator::REGISTRATION)
        .register_generator(generator_44k::REGISTRATION)
}

/// Build the complete explicit MMAudio provider catalog (this crate's own surface).
pub fn provider_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    register_providers(gen_core::ProviderRegistryBuilder::new()).build()
}

/// This crate's schema-3 component licence rows (sc-16663) — **one row per loaded artifact**, not
/// one per (provider, artifact) pair.
///
/// Eight rows for two providers. The Synchformer visual encoder and the DFN5B-CLIP conditioner are
/// each loaded by *both* `mmaudio_small_16k` and `mmaudio_large_44k`, and are the same upstream
/// artifact in both cases — so each contributes one row that both providers point at. v2 duplicated
/// them, once per provider, giving two places for one checkpoint's terms to disagree.
pub const COMPONENT_LICENSES: &[gen_core::ComponentLicense] = &[
    model::COMPONENT_LICENSE,
    clip::COMPONENT_LICENSE,
    mmdit::COMPONENT_LICENSE,
    output::VAE_COMPONENT_LICENSE,
    output::BIGVGAN_COMPONENT_LICENSE,
    mmdit::COMPONENT_LICENSE_44K,
    output::VAE_COMPONENT_LICENSE_44K,
    output::BIGVGAN_V2_COMPONENT_LICENSE,
];

/// The provider→component mapping for both registered MMAudio generators (sc-16663).
///
/// # The hand-authored composite is gone, and it was under-reporting
///
/// v2 gave each provider a hand-typed `component == None` composite row asserting an "effective"
/// restriction — `commercial_use: false` plus a paragraph naming the strictest of five licences.
/// Schema 3 **derives** the provider view from the components below with
/// [`gen_core::provider_terms`], and the derived union is a strict **superset** of what the composite
/// disclosed: both rows named the Apple ML Research restriction on DFN5B-CLIP, but neither mentioned
/// that the same Apple text (§2) also requires handing a copy of the agreement to any redistributee.
/// A union is a set union, so that duty now surfaces; a prose "intersection of five licences" had
/// nowhere to put it.
///
/// The two composites also differed only in prose. `LicenseRef-MMAudio-small-16k-composite` and
/// `LicenseRef-MMAudio-large-44k-composite` were distinct SPDX-shaped identifiers for term sets that
/// were in fact identical — the 44k path swaps a CC-BY-NC-4.0 BigVGAN for an MIT one, and MIT's
/// attribution duty is already carried by four other components. Deriving the view makes that
/// equality visible instead of implying a difference that does not exist.
pub const PROVIDER_COMPONENTS: &[gen_core::ProviderComponents] = &[
    gen_core::ProviderComponents {
        provider_id: generator::MODEL_ID,
        components: &[
            model::MODEL_ID,
            clip::MODEL_ID,
            mmdit::COMPONENT_KEY,
            output::VAE_COMPONENT_KEY,
            output::BIGVGAN_COMPONENT_KEY,
        ],
    },
    gen_core::ProviderComponents {
        provider_id: generator_44k::MODEL_ID,
        components: &[
            model::MODEL_ID,
            clip::MODEL_ID,
            mmdit::COMPONENT_KEY_44K,
            output::VAE_COMPONENT_KEY_44K,
            output::BIGVGAN_V2_COMPONENT_KEY,
        ],
    },
];
