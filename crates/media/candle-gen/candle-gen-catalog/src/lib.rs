//! Explicit, complete provider catalog for the SceneWorks Candle media platform.
//!
//! Provider crates own their registrations; this top-level crate owns only platform composition and
//! stable ordering. Applications should construct one [`ProviderRegistry`] with [`provider_registry`]
//! and route all media loads through it.

pub use candle_gen as media;
pub use candle_gen::gen_core::{ProviderRegistry, ProviderRegistryBuilder};

pub mod licenses;

/// Complete backend package surface owned by the Candle runtimes.
///
/// Some modules are ordinary registry providers; `depth`, `face`, `instantid`, `pid`, `pulid`, and
/// `sam3` are intentionally bespoke utilities consumed through provider-specific APIs.
pub mod providers {
    pub use candle_gen_anima as anima;
    pub use candle_gen_bernini as bernini;
    pub use candle_gen_boogu as boogu;
    pub use candle_gen_chroma as chroma;
    pub use candle_gen_clip as clip;
    pub use candle_gen_depth as depth;
    pub use candle_gen_face as face;
    pub use candle_gen_flux as flux;
    pub use candle_gen_flux2 as flux2;
    pub use candle_gen_ideogram as ideogram;
    pub use candle_gen_instantid as instantid;
    pub use candle_gen_joycaption as joycaption;
    pub use candle_gen_kolors as kolors;
    pub use candle_gen_krea as krea;
    pub use candle_gen_lens as lens;
    pub use candle_gen_ltx as ltx;
    pub use candle_gen_mage as mage;
    pub use candle_gen_minimax_h3 as minimax_h3;
    pub use candle_gen_mochi as mochi;
    pub use candle_gen_pid as pid;
    pub use candle_gen_pulid as pulid;
    pub use candle_gen_qwen_image as qwen_image;
    pub use candle_gen_sam3 as sam3;
    pub use candle_gen_sana as sana;
    pub use candle_gen_scail2 as scail2;
    pub use candle_gen_sd3 as sd3;
    pub use candle_gen_sdxl as sdxl;
    pub use candle_gen_seedvr2 as seedvr2;
    pub use candle_gen_sensenova as sensenova;
    pub use candle_gen_svd as svd;
    pub use candle_gen_wan as wan;
    pub use candle_gen_z_image as z_image;
}

/// Platform-owned crates consumed through provider-specific APIs rather than the registry
/// `load(id, spec)` path (depth maps, face analysis, segmentation, identity conditioning,
/// the PiD latent decoder). Listed here so their platform membership is as explicit as a
/// registered generator. Note `pulid` is bespoke here, whereas MLX ships it as `pulid_flux`.
pub const BESPOKE_UTILITY_CRATES: &[&str] = &["depth", "face", "instantid", "pid", "pulid", "sam3"];

/// Machine-readable disposition for a provider-owned memory contract that intentionally cannot be
/// represented by an ordinary [`candle_gen::gen_core::MemoryRegistration`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BespokeMemoryRouteWaiver {
    pub provider_id: &'static str,
    pub crate_name: &'static str,
    pub owner: &'static str,
    pub reason: &'static str,
    pub contract_path: &'static str,
    pub verification_path: &'static str,
}

/// Descriptor-less Candle memory routes excluded from ordinary registry reconciliation.
///
/// PuLID is worker-owned and constructs a path-shaped FLUX + identity-stack contract through
/// `PulidFlux::load_with_memory_context`; inventing a generator registration would make the public
/// provider topology false. Consumers must reconcile this explicit waiver instead.
pub const BESPOKE_MEMORY_ROUTE_WAIVERS: &[BespokeMemoryRouteWaiver] =
    &[BespokeMemoryRouteWaiver {
        provider_id: candle_gen_pulid::memory_strategy::PROVIDER_ID,
        crate_name: "pulid",
        owner: "candle-gen-pulid",
        reason: "worker-owned bespoke route with a path-shaped memory contract and no LoadSpec/Generator registration",
        contract_path: "crates/media/candle-gen/candle-gen-pulid/src/memory_strategy.rs",
        verification_path: "crates/media/candle-gen/candle-gen-pulid/src/pulid_flux.rs",
    }];

/// Add every provider shipped by the Candle media platform to an explicit registry builder.
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    // The engine's checkpoint codec table is registered here, exactly once, before any family
    // crate: codecs are engine-level (sc-20634/sc-20385), so no family `register_providers` may
    // register one, and the builder refuses a duplicate row outright.
    let registry = candle_gen::logical_weights::register_checkpoint_codecs(registry);
    // The GGUF **container** row (sc-20649) is the one codec the engine crate cannot own: decoding
    // it needs candle's `gguf_file` reader and the ggml block constants, which live with the
    // implementation in `candle-gen-wan`. It is registered here, once, alongside the engine table
    // for the same reason the engine table is — so exactly one place composes the codec set.
    let registry = registry.register_checkpoint_codec(candle_gen::gen_core::GGUF_CONTAINER_CODEC);
    let registry = candle_gen_anima::register_providers(registry);
    let registry = candle_gen_bernini::register_providers(registry);
    let registry = candle_gen_boogu::register_providers(registry);
    let registry = candle_gen_chroma::register_providers(registry);
    let registry = candle_gen_clip::register_providers(registry);
    let registry = candle_gen_flux::register_providers(registry);
    let registry = candle_gen_flux2::register_providers(registry);
    let registry = candle_gen_ideogram::register_providers(registry);
    let registry = candle_gen_joycaption::register_providers(registry);
    let registry = candle_gen_kolors::register_providers(registry);
    let registry = candle_gen_krea::register_providers(registry);
    let registry = candle_gen_lens::register_providers(registry);
    let registry = candle_gen_ltx::register_providers(registry);
    let registry = candle_gen_mage::register_providers(registry);
    // sc-17156: the generator and the memory-strategy registration land TOGETHER.
    // `ProviderRegistryBuilder::build` rejects a memory-strategy registration whose `provider_id`
    // has no matching generator, so this line was absent while the crate shipped only the contract.
    let registry = candle_gen_minimax_h3::register_providers(registry);
    let registry = candle_gen_mochi::register_providers(registry);
    let registry = candle_gen_qwen_image::register_providers(registry);
    let registry = candle_gen_sana::register_providers(registry);
    let registry = candle_gen_scail2::register_providers(registry);
    let registry = candle_gen_sd3::register_providers(registry);
    let registry = candle_gen_sdxl::register_providers(registry);
    let registry = candle_gen_seedvr2::register_providers(registry);
    let registry = candle_gen_sensenova::register_providers(registry);
    let registry = candle_gen_svd::register_providers(registry);
    let registry = candle_gen_wan::register_providers(registry);
    candle_gen_z_image::register_providers(registry)
}

/// Build the complete explicit Candle media provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<ProviderRegistry> {
    register_providers(ProviderRegistryBuilder::new()).build()
}

/// Build the complete Candle registry contract surface without loading model weights or requiring
/// CUDA. On CUDA builds the production catalog already owns these registrations; other platforms
/// append the same contract-only registrations to the ordinary generator catalog.
pub fn memory_contract_surface_registry() -> candle_gen::gen_core::Result<ProviderRegistry> {
    let registry = register_providers(ProviderRegistryBuilder::new());
    #[cfg(not(feature = "cuda"))]
    let registry = candle_gen_flux::register_memory_contract_surfaces(registry);
    #[cfg(not(feature = "cuda"))]
    let registry = candle_gen_flux2::register_memory_contract_surfaces(registry);
    #[cfg(not(feature = "cuda"))]
    let registry = candle_gen_krea::register_memory_contract_surfaces(registry);
    #[cfg(not(feature = "cuda"))]
    let registry = candle_gen_lens::register_memory_contract_surfaces(registry);
    #[cfg(not(feature = "cuda"))]
    let registry = candle_gen_ltx::register_memory_contract_surfaces(registry);
    #[cfg(not(feature = "cuda"))]
    let registry = candle_gen_mage::register_memory_contract_surfaces(registry);
    #[cfg(not(feature = "cuda"))]
    let registry = candle_gen_minimax_h3::register_memory_contract_surfaces(registry);
    #[cfg(not(feature = "cuda"))]
    let registry = candle_gen_qwen_image::register_memory_contract_surfaces(registry);
    #[cfg(not(feature = "cuda"))]
    let registry = candle_gen_z_image::register_memory_contract_surfaces(registry);
    #[cfg(not(feature = "cuda"))]
    let registry = candle_gen_bernini::register_memory_contract_surfaces(registry);
    #[cfg(not(feature = "cuda"))]
    let registry = candle_gen_scail2::register_memory_contract_surfaces(registry);
    registry.build()
}

/// Resolve the load-bearing VAE geometry for a modelled Candle video generator.
///
/// Each provider owns its id-to-decoder assignment. Mochi uses a different decode architecture and
/// is outside the video-memory-ladder scope.
pub fn vae_tiling(provider_id: &str) -> Option<media::gen_core::tiling::VaeTiling> {
    candle_gen_ltx::vae_tiling(provider_id)
        .or_else(|| candle_gen_wan::vae_tiling(provider_id))
        .or_else(|| candle_gen_bernini::vae_tiling(provider_id))
        .or_else(|| candle_gen_scail2::vae_tiling(provider_id))
        .or_else(|| candle_gen_svd::vae_tiling(provider_id))
}

/// Resolve a provider-owned conservative single-pass VAE decode memory profile.
///
/// This composes the calibrated cost functions used by provider budget planners. The result excludes
/// DiT/text-encoder weights. LTX's calibrated fixed term mixes decoder/base/runtime costs, so it is
/// retained in full and identifies zero substitutable decoder-resident bytes; Wan-family profiles
/// contain activation/accumulator work and likewise identify zero resident bytes. Use
/// [`media::VideoDecodeMemoryProfile::checked_composed_peak`] for checked composition and any declared
/// substitution. With LTX's zero attribution the whole mixed floor is deliberately preserved, even
/// though that may conservatively overlap a contract decoder charge. Unsupported ids, zero
/// dimensions, and arithmetic overflow return `None`. SVD currently exports its exact tiling
/// geometry but not a profile here: its actual peak depends on both the request's decode-chunk size
/// and its live-free-VRAM tile selection. SceneWorks' 8-frame product chunk is below the 14-frame
/// write cap at both shipped SVD geometries, while the provider library's 25-frame default is over
/// it; publishing one budget-independent scalar would conflate those regimes.
pub fn conservative_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<media::VideoDecodeMemoryProfile> {
    candle_gen_ltx::conservative_video_decode_memory_profile(provider_id, width, height, frames)
        .or_else(|| {
            candle_gen_wan::conservative_video_decode_memory_profile(
                provider_id,
                width,
                height,
                frames,
            )
        })
        .or_else(|| {
            candle_gen_bernini::conservative_video_decode_memory_profile(
                provider_id,
                width,
                height,
                frames,
            )
        })
        .or_else(|| {
            candle_gen_scail2::conservative_video_decode_memory_profile(
                provider_id,
                width,
                height,
                frames,
            )
        })
}

// -------------------------------------------------------------------------------------------------
// Model-weight licence surface (sc-16667) — the Candle media half.
//
// DISCLOSURE ONLY. Nothing in this section blocks, gates, degrades or withholds anything, and
// nothing added here ever should. It exists so a consumer can SHOW a user which upstream artifacts a
// render touched and what those texts name; whether a given use is permitted is the consumer's
// evaluation of those facts against its own situation, which this crate knows nothing about.
//
// Three layers (see `gen_core::license`): the reviewed licence FAMILIES, one COMPONENT row per
// upstream checkpoint — shared with the MLX catalog, because a licence is a property of the
// checkpoint and both engines load the same ones — and the per-backend provider→component mapping in
// [`licenses`], whose term union is DERIVED and never hand-authored.
// -------------------------------------------------------------------------------------------------

/// The licence families every component row this catalog reaches resolves against — the reviewed
/// unit, shared with the audio and MLX catalogs.
pub fn license_families() -> &'static [media::gen_core::LicenseFamily] {
    media::gen_core::LICENSE_FAMILIES
}

/// The **shared** media checkpoint table: one row per upstream artifact whose licence has been read.
///
/// Deliberately not per-backend and deliberately not filtered to what this catalog reaches. It is
/// the same slice `mlx-gen-catalog` reads, so the two catalogs cannot drift into two different
/// answers about one upstream artifact; rows no registered Candle id loads are simply unreferenced
/// by [`provider_components`].
pub fn component_licenses() -> &'static [media::gen_core::ComponentLicense] {
    media::gen_core::MEDIA_COMPONENT_LICENSES
}

/// Which components each registered Candle provider id loads — the mapping
/// [`media::gen_core::provider_terms`] derives a provider's effective terms from.
///
/// Nine registered ids have no row because every component they load is a pinned hole in the shared
/// table; `licenses::tests` names which nine and why, deliberately as `#[cfg(test)]` data so no gate
/// can read it. Those ids still ship and still render — a missing disclosure never withholds a
/// provider, and registration is never conditioned on this mapping.
pub fn provider_components() -> &'static [media::gen_core::ProviderComponents] {
    licenses::PROVIDER_COMPONENTS
}

/// The model-licences manifest JSON at `schema_version` 3 for **this** catalog, in the same shape
/// the audio catalog emits and the release tooling ships beside the SPDX SBOM.
///
/// Not a committed artifact on its own: `release/model-weight-licenses.json` carries the audio lane
/// today, and merging the three catalogs' manifests into one file is sc-16664's job. Output is
/// deterministic, so a merge can compare byte-for-byte.
pub fn component_licenses_manifest_json() -> String {
    media::gen_core::component_licenses_manifest_json(
        license_families(),
        component_licenses(),
        provider_components(),
    )
}

/// The **advanced** quant tiers this Candle catalog surfaces beyond the universal group-wise affine
/// `Q4`/`Q8` (which every provider advertises via `Capabilities::supported_quants`).
///
/// The NVFP4 FP4 tensor-core tier ([`media::gen_core::Quant::Nvfp4`], epic 11037, sc-11042 **Option A**
/// — a *distinct* creative-choice tier, never an auto-swap of `q4`) is surfaced **only** when the
/// catalog is compiled with the `cuda` feature, i.e. on the consumer Blackwell (`sm_120`) runtime where
/// the FP4 cores exist and the sc-11039 cuBLASLt FP4 GEMM / [`media::quant::Nvfp4Linear`] serve it
/// natively packed (epic 11037 SC#6). The CPU candle bundle (dequant→bf16 fallback, no FP4 compute win)
/// and the MLX/macOS runtime (no FP4 hardware, a separate `mlx-gen-catalog`) do **not** surface it — a
/// deliberate, pinned platform difference (CONTRIBUTING: pin catalog-surface differences rather than
/// paper over them; see `nvfp4_tier_surface_is_cuda_only`).
///
/// This is the inference-repo registration point per the 2026-07-14 epic replan: an NVFP4 tier reaches
/// the SceneWorks worker only through this catalog, shipped by runtime-tag (sc-12006); the worker-side
/// tier-select that *requests* it is deferred to the post-tag phase.
pub fn nvfp4_quant_tiers() -> &'static [media::gen_core::Quant] {
    #[cfg(feature = "cuda")]
    {
        &[media::gen_core::Quant::Nvfp4]
    }
    #[cfg(not(feature = "cuda"))]
    {
        &[]
    }
}

/// Whether **this compilation** of the catalog surfaces the NVFP4 tier — i.e. whether it resolved with
/// the `cuda` feature. Equivalent to `!nvfp4_quant_tiers().is_empty()`, exposed as a `const` because a
/// *dependent* crate cannot ask that question with `cfg!`.
///
/// `cfg!(feature = "cuda")` only ever reads the features of the crate it is written in, so a bundle
/// such as `runtime-cpu` — which has no `cuda` feature of its own — cannot mirror the rule pinned by
/// `nvfp4_tier_surface_is_cuda_only` by writing the same `cfg`. It has to read the catalog's resolved
/// answer, and this is it. That matters because Cargo **feature unification** makes the answer a
/// property of the *resolved graph*, not of the bundle: any build that pulls in `runtime-cpu` and
/// `runtime-cuda` together enables `candle-gen-catalog/cuda` once, for every consumer, so a CPU bundle
/// can legitimately observe the tier surfaced. That resolution is not a supported lane (CLAUDE.md: CPU,
/// CUDA, and MLX are mutually exclusive platform targets, never additive features), but a bundle-side
/// assertion should stay honest under it rather than fail spuriously.
pub const SURFACES_NVFP4_TIER: bool = cfg!(feature = "cuda");

/// Bidirectional guard for `Capabilities::supports_preview` on the Candle catalog (epic 16948,
/// sc-16951) — the candle counterpart of `mlx-gen-catalog`'s
/// `preview_capability_matches_every_wired_shipped_route_bidirectionally`.
///
/// ## Why the ported MLX guard is not enough on its own
///
/// The MLX guard compares two things that are both hand-maintained: an allowlist of ids and the
/// `supports_preview` flags on the descriptors. That pins the two lists to each other, which is
/// worth having — a descriptor cannot advertise without a deliberate edit here — but it says
/// nothing about whether the wiring exists. Edit both sides and it stays green.
///
/// So this module adds a second, **derived** half. Whether a provider crate actually emits is read
/// out of that crate's own sources, in either of the two shapes candle families come in:
///
/// * a **shared-driver** family passes a preview hook where it could pass `None` to
///   `candle_gen::run_flow_sampler` / `run_curated_sampler` / `run_scm_sampler`;
/// * a **bespoke** family owns its denoise loop and emits by calling the shared preview module —
///   `PreviewHook::emit` / `PreviewHook::emit_step`, or the `emit_preview` / `emit_preview_at`
///   free functions underneath them.
///
/// Both shapes are declared per crate in `PROVIDER_CRATES` and checked against the sources, so a
/// crate cannot silently change shape and have its emission fact read as `no`. Neither fact can be
/// produced by editing an allowlist, so the two halves cannot be made to agree by editing one place.
///
/// The source-level scan follows the sc-16950 route inventory in `candle-gen-krea/src/preview.rs`,
/// widened from one crate and one driver to every registered crate, all three drivers, and the
/// bespoke emission calls.
///
/// ## What the scan reads, and what it deliberately does not
///
/// Only the crate's **shipped** module tree is scanned: the walk starts at `src/lib.rs` and follows
/// `mod` declarations, so an out-of-line `#[cfg(test)] mod NAME;` file (`candle-gen-krea`'s
/// `testfix.rs`, `candle-gen-z-image`'s four `*_validate.rs` GPU-validation modules, twenty-odd
/// others) is never read as shipped code. Reading them would be actively harmful in both
/// directions: a hooked call in an `#[ignore]`d validation file would demand a **false**
/// advertisement, and a `None` call in a test fixture would force a test file into a dark
/// declaration. Every `.rs` file under `src` must land in exactly one of those two buckets — an
/// unreachable file fails the scan rather than being silently skipped.
///
/// Test-only items are stripped from the shipped files first, by the *meaning* of their `cfg`
/// predicate rather than by its spelling: `#[cfg(test)]` and `#[cfg(all(test, …))]` are test-only
/// and go, while `#[cfg(any(test, feature = "testkit"))]` and `#[cfg(not(test))]` genuinely ship in
/// some configuration and stay.
///
/// ## The amendment protocol — read this before adding a family
///
/// This guard lands once, holding Krea alone, and is then **amended by each later family story in
/// that story's own PR**: sc-16952…sc-16960 each wire a family and, in the same PR,
///
/// 1. add its exact route ids to `PREVIEW_ROUTE_IDS`,
/// 2. flip its descriptors' `supports_preview`, and
/// 3. add that family's own **route inventory** — the per-file `hooked` / `direct` counts and the
///    pinned `DarkSite`s — to its `ProviderCrate` row, the way sc-16950 pinned Krea's.
///
/// Do not file a follow-up story for any of the three. The derived half is what makes (1) and (2)
/// self-enforcing: the moment a family's sources emit, its descriptors must advertise and the build
/// fails until they do. Step (3) is what keeps a **wired** family honest afterwards — without an
/// exact per-file count, blanking one route of an already-inventoried file changes nothing any
/// assertion can see.
///
/// ## Three classes, and a route is in exactly one (sc-16961)
///
/// A route that does not advertise previews is **not** thereby a rejected one, and the difference is
/// the difference between "someone should wire this" and "do not spend GPU time on this":
///
/// * `PREVIEW_ROUTE_IDS` — **wired**. Emits, and advertises `supports_preview: true`.
/// * `PREVIEW_INERT_ROUTE_IDS` — **no-go**, carried over from epic 16624 rather than re-measured. A
///   fit is a property of the VAE latent space, not of the backend, so a linear approximation that
///   misses the holdout bar on MLX misses it on candle too. **Do not re-run these fits**; the
///   `NoGo` basis on each row says whether it rides a measurement or a deliberate non-measurement.
/// * `PREVIEW_DEFERRED_ROUTE_IDS` — **viable but unwired**. Empty after sc-17218 wired the last
///   deferred family, but retained as an explicit class so a future measured-but-unwired route cannot
///   be mistaken for a no-go.
///
/// `the_no_go_set_and_the_wired_set_partition_every_shipped_route` makes those three total over the
/// registered surface, so a newly registered route must be classified rather than defaulting into
/// silence, and the no-go set cannot go stale as the catalog grows. The full record —
/// numbers, per-family lineage, and the four VAE-relation shapes this epic observed — is
/// `docs/migration/evidence/sc-16961-preview-no-go-carry-over.md`.
#[cfg(test)]
mod preview_advertising {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::{Path, PathBuf};

    use super::ProviderRegistryBuilder;

    // ---- The declared half: exact route ids ------------------------------------------------------

    /// Every shipped Candle route wired for per-step latent previews, by **exact id**.
    ///
    /// Ids, not families: Krea's Turbo / Raw / Edit are three rows because they are three
    /// descriptors, and a family whose routes are wired one at a time has to show that here rather
    /// than hide it behind a family name. sc-16959's Sana base and Sprint will be two rows for the
    /// same reason — they run different drivers and carry different fits.
    ///
    /// Krea (sc-16950), Qwen-Image (sc-16952) and Anima (sc-16953). The two Qwen-family entries are
    /// the two directions in which ids and render lanes fail to correspond, which is why neither may
    /// ever be inferred from the other:
    ///
    /// * **Qwen-Image contributes one row for three lanes.** The crate registers a single generator
    ///   descriptor (`qwen_image`); its edit and ControlNet/Fun lanes are bespoke providers the worker
    ///   drives by name, carrying a `preview` field on their own request types rather than a
    ///   descriptor, so there is no second id to advertise. The derived half still holds all three to
    ///   account: the route inventory on `candle-gen-qwen-image` below pins one hooked site per lane.
    /// * **Anima contributes three rows for one lane.** `anima_base`, `anima_aesthetic` and
    ///   `anima_turbo` are three registered descriptors over one architecture, differing only in the
    ///   DiT weights file, and they share a single `pipeline::AnimaPipeline::generate` body — so one
    ///   hooked sampler site wires all three at once, and the inventory below pins exactly one.
    ///
    /// The SDXL family (sc-16954) adds a third failure of correspondence, in a direction neither Qwen
    /// entry shows: **one id can cover lanes that are not all sampler sites**. `sdxl` and `kolors` are
    /// one descriptor each, but each registered route has *two* denoise lanes — a curated
    /// `run_curated_sampler` lane and a bespoke loop (SDXL Lightning; the Kolors native leading Euler)
    /// — and each crate additionally ships name-driven providers (SDXL edit / IP-Adapter, Kolors
    /// pose-control / IP-Adapter) that carry a `preview` field rather than a descriptor. The
    /// inventories below therefore mix `hooked` and `direct` counts in the same crate, which no
    /// earlier family needed.
    ///
    /// The FLUX.2 family (sc-16955) is the widest single unlock and adds a fourth failure of
    /// correspondence: **one latent space, three crates, two packing orders**. `flux2_klein_9b` /
    /// `flux2_dev` are one crate's two descriptors over three render lanes (txt2img plus the
    /// name-driven edit and strict-pose control providers, which carry a `preview` field rather than a
    /// descriptor); `lens` / `lens_turbo` reach the same 32-channel fit through `candle-gen-lens`,
    /// which owns no projector at all and re-exports `candle-gen-flux2`'s; and `ideogram_4` /
    /// `ideogram_4_turbo` share that fit while packing the 128 transformer channels in a different
    /// order, so they own the de-normalize/unpatchify half and reuse only the projection.
    ///
    /// Ideogram is also this table's first genuine `Denoise::Bespoke` **wired** crate: it drives no
    /// shared sampler anywhere, so it becomes visible below through a direct emission call rather than
    /// through a hooked call site. That is the case the direct-emission scanner was hardened for.
    ///
    /// `candle-gen-boogu` was deliberately absent from sc-16955 because it loads a plain 244-tensor
    /// **16-channel** `AutoencoderKL`, not FLUX.2's 32-channel space. sc-16956 then proved the positive
    /// relationship: Boogu's f32 tensors round exactly onto FLUX.1's bf16 bits, key for key. sc-17218
    /// wires that reused raw-latent projector across all three registered ids. Base and Edit have one
    /// hooked shared-driver site each; Turbo has a hooked curated lane plus a direct emission in its
    /// default native DMD loop, always observing the running latent entering the outer step.
    ///
    /// The FLUX.1 family (sc-16956) is the first to make the descriptor-less half of the platform
    /// *visible* rather than merely absent. It contributes seven rows — `flux1_schnell` / `flux1_dev`
    /// (one crate, two descriptors, one shared txt2img lane) and `chroma1_hd` / `chroma1_base` /
    /// `chroma1_flash` (one crate, three descriptors, one shared lane over a **byte-identical** VAE, so
    /// `candle-gen-chroma` re-exports `candle-gen-flux`'s projector rather than owning one) — while
    /// wiring **five** lanes. The two rows that do not appear here are the reason
    /// `BESPOKE_PROVIDER_CRATES` exists:
    ///
    /// * `candle-gen-flux`'s name-driven Fun-ControlNet-Union and XLabs IP-Adapter providers carry a
    ///   `preview` field on their own request types rather than a descriptor. They are still counted —
    ///   in this crate's route inventory below, which pins one hooked site per file.
    /// * `candle-gen-pulid` registers **nothing at all** and is not even composed by
    ///   `register_providers`, yet it owns a `run_flow_sampler` site of its own. Before sc-16956 no
    ///   table in this module could see it: `PROVIDER_CRATES` is keyed on a registration function.
    ///
    /// Z-Image (sc-16957) contributes two rows — `z_image_turbo` and `z_image`, one crate, two
    /// descriptors — while wiring **nine** lanes across three files, and it adds two things no earlier
    /// family showed:
    ///
    /// * **Both wiring layers on the same registered pair.** `pipeline.rs` holds four hooked driver
    ///   sites (each descriptor's resident route plus its staged-residency twin, which a
    ///   `stage_residency` request reaches instead), while `control.rs` mixes two hooked sites with two
    ///   direct emissions: the *base* control lanes drive the shared sampler and the *distilled Turbo*
    ///   ones own bespoke Euler loops. The same crate is `Denoise::Shared` and emits directly, which is
    ///   why both counts appear on one file's row below.
    /// * **The `_control` route ids in this crate are memory strategies, not descriptors.**
    ///   `z_image_turbo_control` / `z_image_control` register memory contracts and their weights-free
    ///   behavior seams, but no generator descriptor, so they have no id to advertise here — exactly
    ///   the `candle-gen-flux` control/IP shape, and the reason the two ids above cover nine lanes.
    ///   `edit.rs`'s img2img provider is the same: a name-driven worker stream carrying a `preview`
    ///   field on its own request type.
    ///
    /// Z-Image also settles a question sc-16955 raised and sc-16956 half-answered. Its VAE is
    /// **byte-identical to FLUX.1-dev's** — the same `f5b59a26…40a3` container, whose `vae/config.json`
    /// names `flux-dev` as its origin — so epic 16624 committed two fits over one latent space. Both
    /// are kept: `candle-gen-z-image/src/preview.rs` uses the Z-Image-measured one for MLX parity, and
    /// consolidating them is a cross-engine decision, not a candle one. See
    /// `docs/migration/evidence/sc-16957-z-image-candle-preview.md`.
    ///
    /// SD3.5 (sc-16958) contributes three rows — `sd3_5_large`, `sd3_5_large_turbo` and
    /// `sd3_5_medium` — while wiring **six** lanes through a single call site, and it is the first
    /// entry that is plainly the *simple* shape after five families that were not: one crate, one
    /// `run_flow_sampler` site, no bespoke loop, no name-driven provider, no trainer, no dark site.
    /// The three descriptors differ only in the MMDiT checkpoint and whether CFG is enabled, and each
    /// reaches that one site through both a txt2img and an img2img / `Reference` lane — the img2img
    /// fork blends its VAE-encoded reference into `x_t` and shortens the σ schedule *before* the
    /// driver call rather than opening a second one.
    ///
    /// SD3.5 also settles the 16-channel question the other direction from Z-Image and Boogu. Its VAE
    /// is **not** FLUX.1-dev's: same architecture, same 167,666,902-byte container, same 244 bf16
    /// keys and shapes, and **0 of 244 tensors byte-identical** (`8f53304a…c109dc` against
    /// `f5b59a26…40a3`), with its own `1.5305` / `0.0609` normalization. So this is a genuinely
    /// distinct latent space rather than a third fit over one — sc-17309, which tracks the
    /// Z-Image / FLUX.1 duplication, must not gain an SD3.5 row. Sharing
    /// `z_image::vae::AutoEncoderKL` as a Rust *type* is exactly the reasoning this table refuses to
    /// ground a reuse in. See `docs/migration/evidence/sc-16958-sd3-candle-preview.md`.
    ///
    /// SANA (sc-16959) closes Tier 1 and contributes the two rows this table's "ids, not families"
    /// rule was written for. `sana_1600m` and `sana_sprint_1600m` are one crate, two registered
    /// descriptors, **two user-reachable lanes and two different shared sampler drivers** — the only
    /// candle family in this epic that drives more than one:
    ///
    /// * `sana_1600m` reaches `candle_gen::run_flow_sampler` through `pipeline::denoise_cfg`
    ///   (true-CFG flow-match Euler over a static shift-3.0 schedule, the whole curated epic-7114
    ///   sampler menu advertised, so `heun` / `dpmpp_sde` exercise the multi-eval dedup on that one
    ///   site);
    /// * `sana_sprint_1600m` reaches `candle_gen::run_scm_sampler` through `pipeline::denoise_sprint`
    ///   (CFG-free SCM / TrigFlow consistency, 1–4 steps, only the `"default"` sentinel advertised —
    ///   the SCM loop is not a curated `Solver` at all).
    ///
    /// They also carry **two different fits**, and that is what makes the two rows load-bearing rather
    /// than clerical. The two snapshots ship DC-AE autoencoders at an *identical* 1,249,044,836-byte
    /// container size whose SHA-256s differ — `15a4b09e…d9d87f` (base) against `dfd991d1…4454bb`
    /// (Sprint) — and each is byte-identical to the file the corresponding epic-16624 fit was measured
    /// on. A tensor walk says exactly how they relate, and the answer is a shape none of this epic's
    /// earlier stories showed: they **partially overlap**. 320 of 375 tensors are byte-identical,
    /// including the *entire* 179-tensor encoder, and all 55 that differ are in the `decoder.` subtree
    /// — Sprint's DC-AE 1.1 is a decoder-tail fine-tune of base's DC-AE 1.0. So this is **one latent
    /// space with two decoders**, and since an RGB preview fit maps a latent to *decoded* pixels, one
    /// fit still cannot serve both routes. Shipping one for both is the specific mistake sc-16959 was
    /// written to avoid; `candle-gen-sana/src/preview.rs` pins that three ways, and here
    /// `sana_base_and_sprint_are_two_independent_rows` pins that the crate hooks **both** shared
    /// drivers while `source_level_wiring_and_advertised_capability_agree_for_every_provider_crate`
    /// pins each of the two ids on its own.
    ///
    /// Sprint additionally needs a `1/σ_data` correction the flow cohort does not: `run_scm_sampler`
    /// pre-scales its running latent by `σ_data` and hands the hook the scaled tensor. That is the
    /// candle spelling of `mlx-gen-sana`'s `inverse_sigma_data` argument.
    ///
    /// SenseNova-U1 (sc-16960) closes the epic's **Tier 2** and contributes two rows —
    /// `sensenova_u1_8b` and `sensenova_u1_8b_fast`, one crate, two descriptors, one lane. It is the
    /// only family here that needed a fit of its own, and the reason is not the one the epic
    /// predicted:
    ///
    /// * **The epic's premise was wrong — SenseNova-U1 has no VAE at all.** It is a unified dual-path
    ///   Qwen3 MoT backbone whose flow-matching head predicts `3·(patch·merge)²` values per token,
    ///   which `unpatchify` folds straight back into `[1, 3, H, W]`. The model **denoises in pixel
    ///   space**: the running state of the loop *is* the image in `[-1, 1]`, and the "decode" is the
    ///   affine map `tensor_to_image` applies. So the measured fit is over **three** channels, which
    ///   on its own rules it out of every epic-16624 reuse (4, 16 and 32-channel VAE latents) with no
    ///   hash comparison needed. `tests/fit_preview_rgb.rs` measures it and
    ///   `the_snapshot_ships_no_autoencoder` proves the absence structurally — snapshot layout,
    ///   `config.json`, and the shard headers.
    /// * **It is the second genuine `Denoise::Bespoke` wired crate**, after Ideogram. A `git grep` of
    ///   the three shared drivers across `candle-gen-sensenova/src` returns nothing, and that is
    ///   deliberate rather than incidental: the descriptor advertises an **empty** sampler menu
    ///   because the AR backbone's per-step `KvCache` mutation makes any multi-eval curated solver
    ///   unsound. So its wiring becomes visible below through a direct emission call — one, in
    ///   `t2i.rs` — rather than through a hooked call site.
    /// * **It has a second denoise loop that stays dark on purpose.** `it2i_denoise` is the
    ///   off-registry understanding surface (VQA / Document-Studio interleave), reachable only
    ///   through `interleave_gen`, advertised by no descriptor, and known-corrupted on the edit path.
    ///   It is out of scope for sc-16960 and emits nothing, which is why the inventory below reads
    ///   `direct: 1` rather than `2`.
    ///
    /// `supports_preview` does **not** collapse to a single shipped boolean when this epic completes,
    /// and SenseNova is the permanent reason: it is candle-only, MLX never wired it, so at least one
    /// route stays engine-split for good. See `docs/migration/evidence/sc-16960-sensenova-candle-preview.md`.
    ///
    /// `instantid` is deliberately absent from *this* list and cannot be added: it registers no
    /// descriptor at all (`BESPOKE_UTILITY_CRATES`), so it has no id to advertise. Three shipped tests
    /// hold that in place — the second half of
    /// `temporal_and_super_resolution_routes_stay_outside_preview_advertising` asserts by exact id that
    /// no registered descriptor is ever named `instantid`, `pulid` or `pulid_flux`;
    /// `every_shipped_generator_is_covered_by_the_wiring_table` requires the table below to cover
    /// exactly the registered surface, so a new registration could not slip in uninventoried either;
    /// and `every_bespoke_utility_crate_is_covered_by_the_bespoke_table` requires each of the six
    /// descriptor-less crates to be inventoried. InstantID reaches `candle-gen-sdxl`'s
    /// `denoise_curated` / `denoise_ip_multi_control` and passes `None` at both, exactly as MLX left it.
    const PREVIEW_ROUTE_IDS: &[&str] = &[
        "krea_2_turbo",
        "krea_2_raw",
        "krea_2_edit",
        "krea_2_turbo_edit",
        "qwen_image",
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
        "boogu_image",
        "boogu_image_turbo",
        "boogu_image_edit",
        "sdxl",
        "kolors",
        "flux2_klein_9b",
        "flux2_dev",
        "lens",
        "lens_turbo",
        "ideogram_4",
        "ideogram_4_turbo",
        "flux1_schnell",
        "flux1_dev",
        "chroma1_hd",
        "chroma1_base",
        "chroma1_flash",
        "z_image_turbo",
        "z_image",
        "sd3_5_large",
        "sd3_5_large_turbo",
        "sd3_5_medium",
        "sana_1600m",
        "sana_sprint_1600m",
        "sensenova_u1_8b",
        "sensenova_u1_8b_fast",
    ];

    /// The two SANA rows above, named so `sana_base_and_sprint_are_two_independent_rows` can bind
    /// them to `candle-gen-sana` specifically — the generalised per-id check reads its ids back out
    /// of the registry and so cannot say *which* crate registered one. Every row in this table is
    /// asserted individually, by that generalised check; this pair additionally carries the
    /// two-driver assertion. See that test and `candle-gen-sana/src/preview.rs`.
    const SANA_ROUTE_IDS: [&str; 2] = ["sana_1600m", "sana_sprint_1600m"];

    /// Why one route is preview-inert — which epic-16624 finding it rides, and whether that finding
    /// is a **measurement** or the deliberate absence of one.
    ///
    /// The distinction is the whole point of sc-16961. Epic 16624 measured three latent spaces against
    /// a **holdout R² ≥ 0.88** bar and rejected them; it closed the rest *without* measuring them. A
    /// row that quotes a holdout number for a space nobody measured is a fabricated provenance, and a
    /// row that quotes a **fit** (in-sample) number where a **holdout** (out-of-sample) one belongs is
    /// the specific confusion sc-16954 was bounced for.
    ///
    /// **What this type actually enforces, and what it does not.** A *per-route row* cannot invent a
    /// number: `PREVIEW_INERT_ROUTE_IDS` names a variant, and the numbers live on the variant. That is
    /// the mutation this enum closes. It does **not** by itself stop someone attaching a pair to the
    /// wrong variant — `NoGo::measured` is an ordinary `match` — so the variant→measurement mapping is
    /// pinned separately by `the_recorded_no_go_measurements_stay_labelled_fit_versus_holdout`, which
    /// asserts [`NoGo::measured`] is `Some` for precisely [`NoGo::MEASURED`] and `None` for every other
    /// variant in [`NoGo::ALL`].
    ///
    /// Full record, including the lineage evidence behind each variant:
    /// `docs/migration/evidence/sc-16961-preview-no-go-carry-over.md`.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum NoGo {
        /// LTX's 128-channel space. Measured and rejected.
        Ltx,
        /// Mage's 128-channel spatial space. Measured and rejected.
        Mage,
        /// Mochi's 12-channel space. Measured and rejected.
        Mochi,
        /// The Wan **z16** space — `candle_gen_wan::vae16::WanVae16`, built from
        /// `Vae16Config::wan21()` (`z_dim: 16`, `base_dim: 96`, non-residual, no spatial patchify).
        /// **Never measured**; closed under the temporal program gate. Only `wan2_2_t2v_14b`,
        /// `wan2_2_i2v_14b`, `wan_vace`, and `wan2_2_vace_fun_14b` occupy it inside
        /// `candle-gen-wan`; Bernini and Scail2
        /// build the same VAE from that crate, so they ride this row rather than any measured one.
        ///
        /// **Not** the 5B's space — see [`NoGo::WanZ48`]. `vae16.rs`'s own module docs enumerate three
        /// structural axes on which the two differ, and sc-16637 was explicit that a single fit would
        /// never have covered both.
        WanZ16,
        /// The Wan **z48** space — `candle_gen_wan::vae::WanVae` / `AutoencoderKLWan`, built from
        /// `VaeConfig::ti2v_5b()` (`z_dim: 48`, `base_dim: 256`, `is_residual`, `patch_size: 2` so
        /// `conv_out` emits 12 channels and unpatchifies). `wan2_2_ti2v_5b` is the only registered
        /// candle route in it. **Never measured**, closed under the same temporal program gate.
        ///
        /// This is a separate variant rather than a comment on [`NoGo::WanZ16`] because sc-16637's
        /// closure turned on exactly this distinction: "the registered family spans z16 `WanVae` IDs …
        /// and the distinct z48 `Wan22Vae` `wan2_2_ti2v_5b`; a single fit would never have covered the
        /// full surface." Collapsing the two here would erase the finding in the record whose entire
        /// job is to preserve it.
        WanZ48,
        /// SVD's temporal video space — `AutoencoderKLTemporalDecoder`, four latent channels behind a
        /// **temporal** decoder. **Never measured**; same program gate. Its four channels are not
        /// SDXL's four channels, and a per-frame linear map is not an approximation of a decode that
        /// mixes frames.
        SvdTemporal,
        /// SeedVR2. Excluded on its **shape**, not on a number: a one-step super-resolution upscaler
        /// over a low-resolution input has no multi-step progression to preview. No holdout R² was
        /// ever measured for it and none is quoted.
        Seedvr2SuperResolution,
        /// MiniMax-H3's **joint audio+video** space (sc-17156). Excluded on its shape, not on a
        /// number: the video half decodes through a 36-layer *transformer* whose output projection
        /// performs all 16x spatial and 4x temporal upsampling, so there is no per-frame linear map
        /// from a latent to a preview frame the way there is for a CNN VAE — and the packed sequence
        /// interleaves audio rows that have no picture at all. No holdout R^2 was ever measured for
        /// it and none is quoted.
        MinimaxH3Joint,
    }

    impl NoGo {
        /// Every variant, so a test can quantify over the whole enum rather than over a hand-listed
        /// subset that a new variant silently escapes. `the_recorded_no_go_measurements_stay_labelled_
        /// fit_versus_holdout` iterates this and pins which rows carry numbers.
        const ALL: [NoGo; 8] = [
            NoGo::Ltx,
            NoGo::Mage,
            NoGo::Mochi,
            NoGo::WanZ16,
            NoGo::WanZ48,
            NoGo::SvdTemporal,
            NoGo::Seedvr2SuperResolution,
            NoGo::MinimaxH3Joint,
        ];

        /// Exactly the variants epic 16624 put a number on. Kept beside [`NoGo::measured`] so the
        /// test can assert the two agree — the mapping from variant to measurement is otherwise
        /// unpinned, and attaching a borrowed pair to an unmeasured space is precisely the failure
        /// this record exists to prevent.
        const MEASURED: [NoGo; 3] = [NoGo::Ltx, NoGo::Mage, NoGo::Mochi];

        /// The story or stories that settled this latent space, or `None` for the one row epic 16624
        /// never had to consider because it is excluded structurally.
        ///
        /// Not always a single id. SVD in particular was **not** settled by sc-16637 — that story is
        /// "Tier 3: fit the Wan latent space and wire wan", and neither its description nor either of
        /// its comments mentions SVD. It was routed to Tier 3 by **sc-16633** ("Route svd to Tier 3";
        /// "`mlx-gen-svd` … remains false pending sc-16636's temporal contract"), closed by the
        /// program gate **sc-16636** declared, and adjudicated on the candle side by **sc-16954**.
        fn settled_by(self) -> Option<&'static str> {
            match self {
                NoGo::Ltx => Some("sc-16638"),
                NoGo::Mage => Some("sc-16639"),
                NoGo::Mochi => Some("sc-16640"),
                NoGo::WanZ16 | NoGo::WanZ48 => Some("sc-16637"),
                NoGo::SvdTemporal => Some(
                    "closed under the sc-16636 program gate; routed to Tier 3 by sc-16633; \
                     candle-side adjudication sc-16954",
                ),
                NoGo::Seedvr2SuperResolution => None,
                NoGo::MinimaxH3Joint => None,
            }
        }

        /// `Some((fit, holdout))` for a space epic 16624 actually measured; `None` for one it closed
        /// without measuring. Ordered fit-then-holdout and labelled at every use site.
        ///
        /// The variants that return `Some` are pinned against [`NoGo::MEASURED`] by
        /// `the_recorded_no_go_measurements_stay_labelled_fit_versus_holdout`, so moving a pair onto
        /// an unmeasured space fails the build rather than quietly routing that space down the
        /// measured branch.
        fn measured(self) -> Option<(&'static str, &'static str)> {
            match self {
                NoGo::Ltx => Some(("0.984291", "0.618575")),
                NoGo::Mage => Some(("0.938091", "0.806216")),
                NoGo::Mochi => Some(("0.846932", "0.807202")),
                NoGo::WanZ16
                | NoGo::WanZ48
                | NoGo::SvdTemporal
                | NoGo::Seedvr2SuperResolution
                | NoGo::MinimaxH3Joint => None,
            }
        }

        /// The one-line reason, reproduced verbatim in every assertion message so a failure states
        /// *why* rather than reporting a bare boolean.
        fn reason(self) -> String {
            let basis = match self.measured() {
                Some((fit, holdout)) => format!(
                    "measured and REJECTED against the 0.88 holdout bar: fit R² (in-sample) {fit}, \
                     holdout R² (out-of-sample) {holdout}"
                ),
                None => match self {
                    NoGo::WanZ16 => {
                        "the Wan z16 space (candle_gen_wan::vae16::WanVae16, Vae16Config::wan21) — \
                         NEVER measured, closed under the temporal program gate; Bernini and Scail2 \
                         build the same VAE from candle-gen-wan and ride this row, not a measured \
                         one. This is NOT wan2_2_ti2v_5b's space — the 5B is z48"
                    }
                    NoGo::WanZ48 => {
                        "the Wan z48 space (candle_gen_wan::vae::WanVae / AutoencoderKLWan, \
                         VaeConfig::ti2v_5b: z_dim 48, base_dim 256, is_residual, patch_size 2) — \
                         NEVER measured, closed under the same temporal program gate. Structurally \
                         distinct from the z16 WanVae16 the A14B/VACE routes, Bernini and Scail2 \
                         load, so nothing measured or assumed about either space transfers to the \
                         other"
                    }
                    NoGo::SvdTemporal => {
                        "a temporal video space (AutoencoderKLTemporalDecoder, four channels behind a \
                         temporal decoder) — NEVER measured, same program gate; not SDXL's \
                         four-channel space"
                    }
                    NoGo::MinimaxH3Joint => {
                        "MiniMax-H3's joint audio+video space — NEVER measured, and excluded on its \
                         SHAPE: the video half decodes through a 36-layer transformer whose output \
                         projection performs all 16x spatial and 4x temporal upsampling, so there \
                         is no per-frame linear latent→pixel map to fit, and the packed sequence \
                         interleaves audio rows that carry no picture at all"
                    }
                    NoGo::Seedvr2SuperResolution => {
                        "a one-step super-resolution upscaler over a low-resolution input — no \
                         multi-step progression exists to preview, and NO holdout number was ever \
                         measured for it"
                    }
                    NoGo::Ltx | NoGo::Mage | NoGo::Mochi => {
                        unreachable!("a measured row takes the Some branch above")
                    }
                }
                .to_string(),
            };
            match self.settled_by() {
                Some(story) => format!("{basis} ({story})"),
                None => basis,
            }
        }
    }

    /// The routes that stay preview-inert, carried over from epic 16624 rather than re-measured: an
    /// RGB fit is a property of a VAE latent space, not of a backend, so a linear approximation that
    /// misses the holdout bar on MLX misses it on candle too.
    ///
    /// **Candle must not re-run these fits.** This is a settled negative, not an open question — see
    /// [`NoGo`] for which finding each row rides and
    /// `docs/migration/evidence/sc-16961-preview-no-go-carry-over.md` for the evidence. If a future
    /// method makes one viable it reopens as a **new** story with a **new** measurement; nothing here
    /// is "to be decided".
    ///
    /// Twenty-two ids across nine crates. The crate set is *derived* from these ids by
    /// `no_no_go_family_acquires_a_preview_fit_or_a_fit_producer` rather than restated, so a new
    /// registration in one of those crates joins the scan automatically.
    const PREVIEW_INERT_ROUTE_IDS: &[(&str, NoGo)] = &[
        // `candle-gen-wan` registers routes in TWO latent spaces, and the split is load-bearing —
        // sc-16637 closed on exactly this point. The 5B's provider builds `VaeConfig::ti2v_5b()` →
        // `vae::WanVae` (`candle-gen-wan/src/lib.rs`), z48; the A14B pair and VACE build
        // `Vae16Config::wan21()` → `vae16::WanVae16` (`src/wan14b.rs`, `src/model_vace.rs`), z16.
        // `vae16.rs`'s module docs enumerate three structural axes on which the two differ.
        ("wan2_2_ti2v_5b", NoGo::WanZ48),
        ("wan2_2_t2v_14b", NoGo::WanZ16),
        ("wan2_2_i2v_14b", NoGo::WanZ16),
        ("wan_vace", NoGo::WanZ16),
        ("wan2_2_vace_fun_14b", NoGo::WanZ16),
        ("ltx_2_3_distilled", NoGo::Ltx),
        // The 2.5 DiT preserves LTX's latent geometry, so the settled LTX preview fit/holdout
        // decision carries forward; adding the provider is not authorization to rerun that fit.
        ("ltx_2_5_distilled", NoGo::Ltx),
        ("minimax_h3", NoGo::MinimaxH3Joint),
        ("mochi_1", NoGo::Mochi),
        ("mage_flow", NoGo::Mage),
        ("mage_flow_base", NoGo::Mage),
        ("mage_flow_turbo", NoGo::Mage),
        ("mage_flow_edit", NoGo::Mage),
        ("mage_flow_edit_base", NoGo::Mage),
        ("mage_flow_edit_turbo", NoGo::Mage),
        // Bernini's renderer IS Wan2.2-T2V-A14B finetuned: `candle-gen-bernini` takes
        // `candle-gen-wan` as a path dependency and `Components::load` builds
        // `WanVae16::new_with_encoder(&Vae16Config::wan21(), …)`. Same latent space, same closure.
        ("bernini_renderer", NoGo::WanZ16),
        ("bernini", NoGo::WanZ16),
        // SCAIL-2 is Wan2.1-14B I2V; `Scail2Pipeline` and its `Components` both hold a `WanVae16`.
        ("scail2_14b", NoGo::WanZ16),
        ("svd_xt", NoGo::SvdTemporal),
        ("seedvr2", NoGo::Seedvr2SuperResolution),
        ("seedvr2_3b", NoGo::Seedvr2SuperResolution),
        ("seedvr2_7b", NoGo::Seedvr2SuperResolution),
    ];

    /// The third class: registered routes that neither emit previews **nor** are no-gos.
    ///
    /// The class is empty after sc-17218 wired Boogu, but keeping it explicit prevents a future
    /// viable-but-unwired route from being silently absorbed into the no-go set. The total-partition
    /// assertion below keeps all three classes honest as the registry grows.
    const PREVIEW_DEFERRED_ROUTE_IDS: &[(&str, &str)] = &[];

    // ---- The derived half: what the provider sources actually do ---------------------------------

    /// How a crate reaches a denoise loop. Declared here and checked against the sources by
    /// `the_wiring_table_pins_how_each_crate_denoises`, so a crate cannot change shape — a
    /// refactor from the shared driver to a bespoke loop, or the reverse — and have its emission
    /// fact quietly become "does not emit".
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Denoise {
        /// Drives `candle_gen`'s shared samplers. Preview wiring is a hook argument at a call site.
        Shared,
        /// Owns its denoise loop, so it has **no** shared-driver call site at all: SenseNova-U1
        /// (sc-16960) and Ideogram (sc-16955), plus the video families that stay preview-inert.
        /// Declaring the shape is what lets such a crate be *verified* rather than hard-failed for
        /// having nothing to hook — it becomes visible through a direct emission call instead.
        Bespoke,
    }

    /// One shared-sampler call site that deliberately passes no hook, pinned by driver **and
    /// occurrence index** rather than by file.
    ///
    /// File granularity was the wrong unit: one `"training.rs"` entry silences every site in that
    /// file, now and forever, so blanking a *different* route in an already-dark file stays green.
    /// Every family with a trainer (`sdxl`, `lens`, `ltx`, `wan`, `z-image`) will need an entry
    /// here, which makes that edit routine rather than remarkable — exactly the situation in which
    /// a blanket must not be available.
    struct DarkSite {
        /// The driver function name, as in `SAMPLER_DRIVERS`.
        driver: &'static str,
        /// 0-based occurrence index of this driver's call within the file.
        index: usize,
        /// Why this site emits nothing. Non-empty, checked.
        reason: &'static str,
    }

    /// One file of a wired crate's route inventory: exactly how many of its sampler sites pass a
    /// hook, how many direct emission calls it makes, and which of its sites are deliberately dark.
    struct FileRoutes {
        /// `src`-relative, `/`-separated on every platform.
        file: &'static str,
        /// Shared-driver call sites in this file that pass a preview hook.
        hooked: usize,
        /// Direct `PreviewHook::emit` / `emit_step` / `emit_preview` / `emit_preview_at` calls —
        /// how a bespoke denoise loop's wiring shows up.
        direct: usize,
        /// Sites that pass the literal `None`.
        dark: &'static [DarkSite],
    }

    /// One registered provider crate: where its sources live, how to ask it for its own ids, how it
    /// denoises, and — once it is wired — its exact per-file route inventory.
    struct ProviderCrate {
        /// Directory name under `crates/media/candle-gen`.
        dir: &'static str,
        /// The crate's own registration function. Its ids are read back out of a registry built
        /// from it alone, so this table never restates an id and cannot drift from one.
        register: fn(ProviderRegistryBuilder) -> ProviderRegistryBuilder,
        /// Shared-driver or bespoke — checked against the sources.
        denoise: Denoise,
        /// The route inventory, one row per source file that drives a sampler or emits directly.
        /// **Empty for an unwired crate**, and required to match the scan exactly for a wired one:
        /// `every_wired_crate_pins_its_exact_route_inventory` fails on a count that has moved, so a
        /// newly blanked route in an already-inventoried file cannot hide behind a neighbour.
        routes: &'static [FileRoutes],
    }

    /// Every crate `register_providers` composes that ships a generator. `clip` and `joycaption`
    /// register no generator, so they contribute no `supports_preview` surface and are omitted;
    /// `every_shipped_generator_is_covered_by_the_wiring_table` fails if that ever stops being true,
    /// or if a new provider crate joins the catalog without joining this table.
    const PROVIDER_CRATES: &[ProviderCrate] = &[
        ProviderCrate {
            dir: "candle-gen-anima",
            register: candle_gen_anima::register_providers,
            denoise: Denoise::Shared,
            // sc-16953's inventory: one hooked `run_flow_sampler` site, in the single txt2img render
            // lane all three variants share. No dark site — this crate has no trainer and no second
            // denoise — and no direct emission, because `preview.rs` holds only the 5-D Cosmos →
            // `[1, C, h, w]` layout adaptation in front of the reused QwenVae fit.
            routes: &[FileRoutes {
                file: "pipeline.rs",
                hooked: 1,
                direct: 0,
                dark: &[],
            }],
        },
        ProviderCrate {
            dir: "candle-gen-bernini",
            register: candle_gen_bernini::register_providers,
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-boogu",
            register: candle_gen_boogu::register_providers,
            denoise: Denoise::Shared,
            // sc-17218's resident inventory plus sc-20787's exact staged twin. Base, Turbo's
            // curated/img2img lane, and Edit each drive one hooked flow-sampler site per residency
            // path. Turbo's default route owns one direct four-step DMD loop per residency path and
            // emits before prediction and re-noise. No dark site: every user-reachable denoise lane
            // has a live sink seam.
            routes: &[FileRoutes {
                file: "pipeline.rs",
                hooked: 6,
                direct: 2,
                dark: &[],
            }],
        },
        ProviderCrate {
            dir: "candle-gen-chroma",
            register: candle_gen_chroma::register_providers,
            denoise: Denoise::Shared,
            // sc-16956's inventory: one hooked `run_flow_sampler` site, in the single txt2img render
            // lane all three variants share. No dark site — this crate has no trainer and no second
            // denoise — and no direct emission, because `preview.rs` is a `pub use` shim over
            // `candle-gen-flux`'s projector: Chroma ships a VAE byte-identical to FLUX.1-dev's, so it
            // reuses that crate's committed 16-channel fit instead of owning a second copy.
            routes: &[FileRoutes {
                file: "pipeline.rs",
                hooked: 1,
                direct: 0,
                dark: &[],
            }],
        },
        ProviderCrate {
            dir: "candle-gen-flux",
            register: candle_gen_flux::register_providers,
            denoise: Denoise::Shared,
            // sc-16956's inventory: one hooked `run_flow_sampler` site per shipped render lane — the
            // registered txt2img route both `flux1_*` descriptors share (`pipeline.rs`), the name-driven
            // Fun-ControlNet-Union strict-pose provider (`control_provider.rs`, the lane the worker
            // already hands a live sink to), and the name-driven XLabs IP-Adapter provider
            // (`ip_provider.rs`). The latter two carry a `preview` field on their own request types
            // rather than a descriptor, which is why this crate's two ids cover three lanes.
            //
            // No dark site: this crate has no trainer and no second denoise. All three project AFTER
            // `flux::sampling::unpack` — the same function `decode_latents` calls — which is why
            // `preview.rs` holds no sampler site and no direct emission: it carries only the reused
            // 16-channel fit and the projector it is applied through.
            routes: &[
                FileRoutes {
                    file: "control_provider.rs",
                    hooked: 1,
                    direct: 0,
                    dark: &[],
                },
                FileRoutes {
                    file: "ip_provider.rs",
                    hooked: 1,
                    direct: 0,
                    dark: &[],
                },
                FileRoutes {
                    file: "pipeline.rs",
                    hooked: 1,
                    direct: 0,
                    dark: &[],
                },
            ],
        },
        ProviderCrate {
            dir: "candle-gen-flux2",
            register: candle_gen_flux2::register_providers,
            denoise: Denoise::Shared,
            // sc-16955's inventory: one hooked `run_flow_sampler` site per shipped render lane — the
            // registered txt2img route (`lib.rs`), the name-driven reference-edit provider
            // (`edit_provider.rs`), and the name-driven strict-pose control provider
            // (`control_provider.rs`). No dark site: this crate has no trainer and no second denoise.
            // All three project AFTER `pipeline::unpack_latents_at` and the VAE's own bn de-normalize +
            // 2×2 unpatchify, which is why `pipeline.rs`, `vae.rs` and `preview.rs` hold no sampler site
            // and no direct emission — `preview.rs` carries only the reused 32-channel fit and the
            // projector it is applied through.
            routes: &[
                FileRoutes {
                    file: "control_provider.rs",
                    hooked: 1,
                    direct: 0,
                    dark: &[],
                },
                FileRoutes {
                    file: "edit_provider.rs",
                    hooked: 1,
                    direct: 0,
                    dark: &[],
                },
                FileRoutes {
                    file: "lib.rs",
                    hooked: 1,
                    direct: 0,
                    dark: &[],
                },
            ],
        },
        ProviderCrate {
            dir: "candle-gen-ideogram",
            // A bespoke `pipeline::denoise` flow-match loop, no shared-driver site anywhere —
            // sc-16955 wires it through a direct emission call, not a hook argument.
            register: candle_gen_ideogram::register_providers,
            denoise: Denoise::Bespoke,
            // sc-16955's inventory, and the first time a **wired** crate's whole contribution is a
            // direct emission: one `emit_preview_at` call inside `pipeline::denoise`, the single lane
            // both registered ids (`ideogram_4`, `ideogram_4_turbo`) and both conditioning modes
            // (txt2img and the reference/mask edit) reach. `hooked: 0` is the point rather than a gap —
            // there is no sampler call site in this crate to hook, which is what `Denoise::Bespoke`
            // declares and `the_wiring_table_pins_how_each_crate_denoises` verifies. `preview.rs` holds
            // the `(ph,pw,c)`-order projector but emits nothing itself.
            routes: &[FileRoutes {
                file: "pipeline.rs",
                hooked: 0,
                direct: 1,
                dark: &[],
            }],
        },
        ProviderCrate {
            dir: "candle-gen-kolors",
            register: candle_gen_kolors::register_providers,
            denoise: Denoise::Shared,
            // sc-16954/sc-20790 inventory. Kolors has one hooked driver call — the registered route's
            // curated resident lane in `pipeline.rs` — and resident plus request-staged leading-Euler
            // loops in each of `pipeline.rs`, `control.rs`, and `ip_provider.rs`, all emitting directly.
            // The two providers' CURATED lanes are invisible here by construction: they reach
            // `candle_gen_sdxl::denoise_curated` rather than a driver of their own, so the hook they
            // build is counted in `candle-gen-sdxl`'s `denoise.rs` row. No dark site — this crate has
            // no trainer.
            routes: &[
                FileRoutes {
                    file: "control.rs",
                    hooked: 0,
                    direct: 2,
                    dark: &[],
                },
                FileRoutes {
                    file: "ip_provider.rs",
                    hooked: 0,
                    direct: 2,
                    dark: &[],
                },
                FileRoutes {
                    file: "pipeline.rs",
                    hooked: 1,
                    direct: 2,
                    dark: &[],
                },
            ],
        },
        ProviderCrate {
            dir: "candle-gen-krea",
            register: candle_gen_krea::register_providers,
            denoise: Denoise::Shared,
            // sc-16950's inventory, restated as counts so a blanked route is a diff here too: the
            // pose-control route, the seven `pipeline` render routes (Turbo three-stage / t2i /
            // img2img, Raw t2i / multi-phase / img2img, and the shared Turbo+Raw edit), and the one
            // deliberately dark trainer site.
            routes: &[
                FileRoutes {
                    file: "control_provider.rs",
                    hooked: 1,
                    direct: 0,
                    dark: &[],
                },
                FileRoutes {
                    file: "pipeline.rs",
                    hooked: 7,
                    direct: 0,
                    dark: &[],
                },
                FileRoutes {
                    file: "training.rs",
                    hooked: 0,
                    direct: 0,
                    dark: &[DarkSite {
                        driver: "run_flow_sampler",
                        index: 0,
                        reason: "the trainer's periodic sample render drives the sampler from a \
                                 synthetic request that carries no PreviewSink, so it passes `None` \
                                 on purpose (sc-16950 pins that as a decision in the Krea crate's \
                                 own inventory)",
                    }],
                },
            ],
        },
        ProviderCrate {
            dir: "candle-gen-lens",
            register: candle_gen_lens::register_providers,
            denoise: Denoise::Shared,
            // sc-16955's inventory. Lens has ONE sampler site — `Pipeline::denoise` in `lib.rs`, which
            // forwards its caller's hook — plus the deliberately dark trainer site. The site count is
            // deliberately not the lane count: three callers reach that one site (the resident
            // `render`, the sequential `render_sequential`, and the `denoise_for_parity` seam), and
            // only the first two have a `PreviewSink`. That distinction is invisible at this
            // granularity, so `candle-gen-lens/src/preview.rs` pins the per-caller classification
            // against the crate's own sources. No direct emission: Lens owns no projector at all and
            // re-exports `candle-gen-flux2`'s, so `preview.rs` is a `pub use` shim.
            routes: &[
                FileRoutes {
                    file: "lib.rs",
                    hooked: 1,
                    direct: 0,
                    dark: &[],
                },
                FileRoutes {
                    file: "training.rs",
                    hooked: 0,
                    direct: 0,
                    dark: &[DarkSite {
                        driver: "run_flow_sampler",
                        index: 0,
                        reason: "the trainer's periodic sample render drives the sampler from a \
                                 synthetic request that carries no PreviewSink — its result is \
                                 delivered as a finished TrainingProgress::Sample image, not as a \
                                 live denoise stream — so it passes `None` on purpose (the same \
                                 decision sc-16950 recorded for Krea's trainer and sc-16954 for \
                                 SDXL's)",
                    }],
                },
            ],
        },
        ProviderCrate {
            dir: "candle-gen-ltx",
            register: candle_gen_ltx::register_providers,
            denoise: Denoise::Shared,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-mage",
            register: candle_gen_mage::register_providers,
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-minimax-h3",
            register: candle_gen_minimax_h3::register_providers,
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-mochi",
            register: candle_gen_mochi::register_providers,
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-qwen-image",
            register: candle_gen_qwen_image::register_providers,
            denoise: Denoise::Shared,
            // sc-16952's inventory: one hooked `run_flow_sampler` site per shipped render lane —
            // base txt2img (`lib.rs`), reference edit (`edit.rs`), and 2512-Fun ControlNet
            // (`control_fun.rs`). No dark site: this crate has no trainer and no second denoise.
            // All three project AFTER `pipeline::unpack_latents`, which is why `pipeline.rs` itself
            // holds no sampler site and no direct emission.
            routes: &[
                FileRoutes {
                    file: "control_fun.rs",
                    hooked: 1,
                    direct: 0,
                    dark: &[],
                },
                FileRoutes {
                    file: "edit.rs",
                    hooked: 1,
                    direct: 0,
                    dark: &[],
                },
                FileRoutes {
                    file: "lib.rs",
                    hooked: 1,
                    direct: 0,
                    dark: &[],
                },
            ],
        },
        ProviderCrate {
            dir: "candle-gen-sana",
            register: candle_gen_sana::register_providers,
            denoise: Denoise::Shared,
            // sc-16959's inventory, and the only row in this table whose two hooked sites are two
            // DIFFERENT drivers: `pipeline.rs` holds one `run_flow_sampler` call (`denoise_cfg`, the
            // `sana_1600m` lane) and one `run_scm_sampler` call (`denoise_sprint`, the
            // `sana_sprint_1600m` lane). `hooked: 2` is therefore also the lane count — each
            // registered descriptor has exactly one user-reachable txt2img lane, because both `load`
            // functions refuse quantization, adapters and control / IP-adapter overlays outright, so
            // the crate ships no img2img fork and no name-driven provider.
            //
            // No dark site: this crate has no trainer and no second denoise anywhere. No direct
            // emission either — `preview.rs` carries only the two reused epic-16624 32-channel fits,
            // the layout check, and the `1/σ_data` correction the SCM route's pre-scaled running
            // latent needs; SANA's latent is already the `[1, C, h, w]` contract on both routes, with
            // nothing to unpack and no frame axis to drop.
            routes: &[FileRoutes {
                file: "pipeline.rs",
                hooked: 2,
                direct: 0,
                dark: &[],
            }],
        },
        ProviderCrate {
            dir: "candle-gen-scail2",
            register: candle_gen_scail2::register_providers,
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-sd3",
            register: candle_gen_sd3::register_providers,
            denoise: Denoise::Shared,
            // Two hooked `run_flow_sampler` sites: the warm Resident `render_core` and the
            // request-local Staged denoise phase. Each carries all six route/mode lanes (three
            // descriptors x T2I/I2I) through the same schedule, CFG, seed, and preview contract.
            // No dark site: `load_variant` refuses control / IP-adapter overlays, so this crate ships
            // no descriptor-less render lane, and it has no trainer. No direct emission either —
            // `preview.rs` holds only the reused epic-16624 16-channel fit and a layout check, since
            // SD3.5's running latent is already the `[1, C, h, w]` contract with nothing to unpack.
            routes: &[FileRoutes {
                file: "pipeline.rs",
                hooked: 2,
                direct: 0,
                dark: &[],
            }],
        },
        ProviderCrate {
            dir: "candle-gen-sdxl",
            register: candle_gen_sdxl::register_providers,
            denoise: Denoise::Shared,
            // sc-16954's inventory, and the first to mix both wiring layers in one crate. SDXL ships
            // SIX emitting lanes across four files plus one deliberately dark trainer site:
            //   * `control_provider.rs` — the generic ControlNet route's bespoke Euler loop
            //     (direct).
            //   * `pipeline.rs` — the registered route's two lanes: the curated driver call (hooked)
            //     and the bespoke Lightning Euler loop (direct).
            //   * `denoise.rs` — the shared helpers the name-driven providers, Kolors and InstantID
            //     all reach: `denoise_curated` (hooked, forwarding its caller's hook) and the bespoke
            //     `denoise_ip_multi_control` ancestral loop (direct).
            //   * `edit_provider.rs` — the bespoke img2img/inpaint ancestral loop (direct).
            //   * `training.rs` — the trainer's periodic sample render, dark on purpose.
            // `ip_provider.rs` gets no row: it drives `denoise::denoise_curated` and the ancestral
            // loop rather than a driver of its own, so its wiring is counted where those live.
            routes: &[
                FileRoutes {
                    file: "control_provider.rs",
                    hooked: 0,
                    direct: 1,
                    dark: &[],
                },
                FileRoutes {
                    file: "denoise.rs",
                    hooked: 1,
                    direct: 1,
                    dark: &[],
                },
                FileRoutes {
                    file: "edit_provider.rs",
                    hooked: 0,
                    direct: 1,
                    dark: &[],
                },
                FileRoutes {
                    file: "pipeline.rs",
                    hooked: 1,
                    direct: 1,
                    dark: &[],
                },
                FileRoutes {
                    file: "training.rs",
                    hooked: 0,
                    direct: 0,
                    dark: &[DarkSite {
                        driver: "run_curated_sampler",
                        index: 0,
                        reason: "the trainer's periodic sample render drives the sampler from a \
                                 synthetic request that carries no PreviewSink — its result is \
                                 delivered as a finished TrainingProgress::Sample image, not as a \
                                 live denoise stream — so it passes `None` on purpose (the same \
                                 decision sc-16950 recorded for Krea's trainer)",
                    }],
                },
            ],
        },
        ProviderCrate {
            dir: "candle-gen-seedvr2",
            register: candle_gen_seedvr2::register_providers,
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-sensenova",
            // The Tier 2 family, and the epic's one measured fit. Its bespoke flow-match loop in
            // `t2i.rs` drives no shared sampler at all, so sc-16960 becomes visible here through a
            // direct emission call rather than a driver argument.
            register: candle_gen_sensenova::register_providers,
            denoise: Denoise::Bespoke,
            // Two direct emissions live in `t2i.rs`: the registered T2I loop and the registered
            // reference/MultiReference it2i loop. Both ids reach those routes through
            // `T2iModel::{generate, it2i_generate}`; neither drives a shared sampler, which is
            // what `Denoise::Bespoke` above declares. `preview.rs` gets no row: it carries only the
            // measured three-channel pixel-space fit and the pool to the token grid, so it neither
            // drives a sampler nor emits. The understanding/VQA interleave entry point reuses the
            // same it2i loop rather than adding a third emission site.
            routes: &[FileRoutes {
                file: "t2i.rs",
                hooked: 0,
                direct: 2,
                dark: &[],
            }],
        },
        ProviderCrate {
            dir: "candle-gen-svd",
            register: candle_gen_svd::register_providers,
            denoise: Denoise::Shared,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-wan",
            register: candle_gen_wan::register_providers,
            denoise: Denoise::Shared,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-z-image",
            register: candle_gen_z_image::register_providers,
            denoise: Denoise::Shared,
            // sc-16957's inventory — nine emitting lanes across three files, plus one deliberately
            // dark trainer site, and the first crate to mix both wiring layers on a *registered* pair:
            //   * `control.rs` — the name-driven Fun-ControlNet provider's FOUR lanes. Its base halves
            //     (staged `denoise_base_with`, resident `generate_base`) drive the shared sampler with
            //     a hook; its distilled Turbo halves (staged `denoise_turbo_with`, resident
            //     `generate_turbo`) own bespoke flow-Euler loops and emit directly.
            //   * `edit.rs` — the name-driven img2img / masked-edit provider's bespoke loop, which
            //     emits over the REDUCED `start..steps` tail its strength selects.
            //   * `pipeline.rs` — the two registered descriptors' four hooked driver sites: each of
            //     `z_image_turbo` and `z_image` has a resident route (`render` / `render_base`) and a
            //     staged-residency twin (`denoise_sequential` / `denoise_base_sequential`) that a
            //     `stage_residency` request reaches instead. txt2img and img2img share a site; the
            //     reference only changes the start step.
            //   * `training.rs` — the trainer's periodic sample render, dark on purpose.
            // `preview.rs` gets no row: it carries only the reused 16-channel fit and the frame-axis
            // drop that reaches it, so it neither drives a sampler nor emits.
            routes: &[
                FileRoutes {
                    file: "control.rs",
                    hooked: 2,
                    direct: 2,
                    dark: &[],
                },
                FileRoutes {
                    file: "edit.rs",
                    hooked: 0,
                    direct: 1,
                    dark: &[],
                },
                FileRoutes {
                    file: "pipeline.rs",
                    hooked: 4,
                    direct: 0,
                    dark: &[],
                },
                FileRoutes {
                    file: "training.rs",
                    hooked: 0,
                    direct: 0,
                    dark: &[DarkSite {
                        driver: "run_flow_sampler",
                        index: 0,
                        reason: "the trainer's periodic sample render drives the sampler from a \
                                 synthetic request that carries no PreviewSink — its result is \
                                 delivered as a finished TrainingProgress::Sample image, not as a \
                                 live denoise stream — so it passes `None` on purpose (the same \
                                 decision sc-16950 recorded for Krea's trainer, sc-16954 for SDXL's \
                                 and sc-16955 for Lens's)",
                    }],
                },
            ],
        },
    ];

    /// One **descriptor-less** provider crate — a [`super::BESPOKE_UTILITY_CRATES`] member, consumed
    /// through a provider-specific API rather than through `gen_core`'s registry.
    ///
    /// Why this table exists (sc-16956): [`ProviderCrate`] is keyed on a registration function, so a
    /// crate that registers nothing cannot appear there — and until FLUX.1 landed, none of the six
    /// owned a denoise loop, so their absence cost nothing. `candle-gen-pulid` breaks that: it drives
    /// its own `run_flow_sampler` over the FLUX.1-dev backbone it composes, is not even reached by
    /// `register_providers`, and would therefore have been wired with **no** guard at all — the exact
    /// "one lane left dark on a shipped route" failure this module exists to prevent.
    ///
    /// All six are listed, wired or not, and `every_bespoke_utility_crate_is_covered_by_the_bespoke_table`
    /// pins that against `BESPOKE_UTILITY_CRATES` itself, so a seventh utility crate — or a sampler
    /// site appearing in one of the five that have none — cannot join uninventoried. There is no
    /// advertising half here, by construction: a crate with no descriptor has nothing to advertise, and
    /// `temporal_and_super_resolution_routes_stay_outside_preview_advertising` is what keeps it that way.
    struct BespokeCrate {
        /// Directory name under `crates/media/candle-gen`.
        dir: &'static str,
        /// Shared-driver or bespoke — checked against the sources exactly as [`ProviderCrate`]'s is.
        denoise: Denoise,
        /// The route inventory, one row per source file that drives a sampler or emits directly.
        /// **Empty for a crate with no denoise loop**, which is five of the six.
        routes: &'static [FileRoutes],
    }

    const BESPOKE_PROVIDER_CRATES: &[BespokeCrate] = &[
        // Depth estimation (DWPose/MiDaS-lineage hints). No denoise loop at all.
        BespokeCrate {
            dir: "candle-gen-depth",
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        // SCRFD + ArcFace + BiSeNet. Detection and embedding only.
        BespokeCrate {
            dir: "candle-gen-face",
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        // InstantID composes `candle-gen-sdxl`'s `denoise_curated` / `denoise_ip_multi_control` rather
        // than driving a sampler of its own, so its wiring is counted in that crate's `denoise.rs` row
        // and it has no site here. It passes `None` at both, exactly as MLX left it.
        BespokeCrate {
            dir: "candle-gen-instantid",
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        // The PiD super-resolving decoder: a decode seam, not a denoise.
        BespokeCrate {
            dir: "candle-gen-pid",
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        // sc-16956's inventory, and the only wired member of this table: ONE hooked
        // `run_flow_sampler` site, in `PulidFlux::generate`'s dev flow denoise. `preview.rs` is a
        // `pub use` shim over `candle-gen-flux`'s projector — PuLID composes that crate's own
        // `FluxRefBackbone`, so the latent, the unpack and the VAE are literally the registered
        // `flux1_dev` route's — so it holds neither a site nor a direct emission. No dark site: this is
        // the crate's only denoise.
        BespokeCrate {
            dir: "candle-gen-pulid",
            denoise: Denoise::Shared,
            routes: &[FileRoutes {
                file: "pulid_flux.rs",
                hooked: 1,
                direct: 0,
                dark: &[],
            }],
        },
        // Segment Anything 3. Masks, no denoise.
        BespokeCrate {
            dir: "candle-gen-sam3",
            denoise: Denoise::Bespoke,
            routes: &[],
        },
    ];

    /// Every crate whose sources this module scans, as `(dir, denoise, routes)` — the registered
    /// provider crates and the descriptor-less ones together.
    ///
    /// The three source-derived assertions (denoise shape, no undeclared dark site, exact route
    /// inventory) apply to both tables identically; only the id-keyed assertions are `PROVIDER_CRATES`-
    /// only, because a `BespokeCrate` has no ids.
    fn scanned_crates() -> Vec<(&'static str, Denoise, &'static [FileRoutes])> {
        PROVIDER_CRATES
            .iter()
            .map(|provider| (provider.dir, provider.denoise, provider.routes))
            .chain(
                BESPOKE_PROVIDER_CRATES
                    .iter()
                    .map(|provider| (provider.dir, provider.denoise, provider.routes)),
            )
            .collect()
    }

    /// One shared sampler driver: the call text scanned for, the **0-based position of its
    /// `preview` parameter**, and its total parameter count. Positions are read from the front, so
    /// a family that passes a named `predict` instead of an inline closure parses the same way, and
    /// both numbers are re-derived from `candle-gen/src/sampler.rs` by
    /// `the_sampler_driver_signatures_pin_the_preview_argument_position` — a signature change fails
    /// there rather than silently shifting what "position 7" names here.
    ///
    /// `run_av_curated_sampler` (`candle-gen/src/sampler.rs`) is deliberately **absent**: LTX's
    /// joint video+audio driver has no `preview` parameter at all, and LTX is in the epic-16624
    /// holdout rejection set, so there is nothing to hook. That same test asserts it still has no
    /// such parameter, so extending sc-16949's hook to it cannot quietly leave this table behind.
    struct SamplerDriver {
        /// The driver's function name — also what a `DarkSite` names.
        function: &'static str,
        /// The call text scanned for in provider sources.
        call: &'static str,
        preview_at: usize,
        arity: usize,
    }

    const SAMPLER_DRIVERS: &[SamplerDriver] = &[
        SamplerDriver {
            function: "run_flow_sampler",
            call: "run_flow_sampler(",
            preview_at: 7,
            arity: 9,
        },
        SamplerDriver {
            function: "run_curated_sampler",
            call: "run_curated_sampler(",
            preview_at: 7,
            arity: 9,
        },
        SamplerDriver {
            function: "run_scm_sampler",
            call: "run_scm_sampler(",
            preview_at: 5,
            arity: 7,
        },
    ];

    /// The `candle-gen` driver that must stay out of `SAMPLER_DRIVERS` for as long as it carries no
    /// preview seam.
    const UNHOOKED_SAMPLER_DRIVER: &str = "run_av_curated_sampler";

    /// A bespoke denoise loop emits by calling the shared preview machinery directly rather than by
    /// handing a hook to a driver. Both layers count, because both are load-bearing house style:
    /// `PreviewHook::emit` / `emit_step` is what the drivers themselves call
    /// (`candle-gen/src/sampler.rs`), so it is what a bespoke loop copies; `emit_preview` /
    /// `emit_preview_at` are the free functions underneath. Listing only the free functions was the
    /// original hole — a bespoke loop written the canonical way was invisible, which would have hit
    /// SenseNova (sc-16960) and Ideogram (sc-16955), the two crates the constant exists for.
    const DIRECT_EMISSION_FUNCTION_CALLS: &[&str] = &["emit_preview(", "emit_preview_at("];
    const DIRECT_EMISSION_METHOD_CALLS: &[(&str, usize)] = &[(".emit(", 4), (".emit_step(", 3)];

    /// One shared-sampler call site found in a provider crate's sources.
    struct SamplerSite {
        /// `src`-relative path, `/`-separated on every platform.
        file: String,
        /// The driver's function name.
        driver: &'static str,
        /// 0-based occurrence index of this driver's call within the file.
        index: usize,
        /// Whether its `preview` argument is anything other than the literal `None`.
        hooked: bool,
    }

    /// Direct emission calls found in one file.
    struct DirectEmission {
        file: String,
        count: usize,
    }

    /// What a crate's sources say about preview emission.
    struct CrateWiring {
        sites: Vec<SamplerSite>,
        direct: Vec<DirectEmission>,
        /// `src`-relative files the crate actually compiles into its shipped surface.
        scanned: Vec<String>,
        /// `src`-relative files reachable only through a test-only `mod` declaration, and therefore
        /// deliberately excluded.
        excluded: Vec<String>,
    }

    impl CrateWiring {
        /// Whether this crate emits previews at all. Derived from the sources, never declared.
        fn emits(&self) -> bool {
            self.sites.iter().any(|site| site.hooked) || !self.direct.is_empty()
        }
    }

    /// `crates/media/candle-gen`, resolved from this crate's manifest directory.
    fn candle_gen_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("candle-gen-catalog sits inside crates/media/candle-gen")
            .to_path_buf()
    }

    // ---- Stripping test-only code ----------------------------------------------------------------

    /// Rust source with comments, string / char literals, and **test-only** items removed, plus the
    /// names of the out-of-line `mod NAME;` declarations that were removed with them.
    struct Stripped {
        code: String,
        test_only_mods: Vec<String>,
    }

    /// What a `cfg(…)` predicate says about `test`.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum CfgTest {
        /// No `test` token anywhere: an ordinary platform / feature gate.
        Absent,
        /// The item exists **only** under `cfg(test)` — safe to strip.
        TestOnly,
        /// The item also ships in some non-test configuration, so it stays in the scan.
        AlsoShips,
    }

    /// Classify a `cfg(…)` predicate by **meaning**, not spelling.
    ///
    /// The spelling matters because both alternative forms are live house style in this tree —
    /// `#[cfg(any(test, feature = "testkit"))]` and `#[cfg(all(test, unix))]` in `candle-gen`
    /// itself — and a substring test for the literal `cfg(test` matches neither, so an
    /// `all(test, …)` module used to survive the strip and read as shipped code.
    fn classify_cfg(predicate: &str) -> CfgTest {
        if !tokens(predicate).any(|token| token == "test") {
            return CfgTest::Absent;
        }
        if predicate_is_test_only(predicate) {
            CfgTest::TestOnly
        } else {
            // `any(test, feature = "testkit")` and `not(test)` are real shipped code in some
            // configuration. Stripping them would UNDER-scan, so they stay.
            CfgTest::AlsoShips
        }
    }

    /// Whether a `cfg` predicate can only hold under `--test`: `test` itself, an `all(…)` with a
    /// test-only conjunct, or an `any(…)` whose every alternative is test-only.
    fn predicate_is_test_only(predicate: &str) -> bool {
        let predicate = predicate.trim();
        if predicate == "test" {
            return true;
        }
        if let Some(inner) = strip_call(predicate, "all") {
            return split_top_level(inner)
                .iter()
                .any(|part| predicate_is_test_only(part));
        }
        if let Some(inner) = strip_call(predicate, "any") {
            let parts = split_top_level(inner);
            return !parts.is_empty() && parts.iter().all(|part| predicate_is_test_only(part));
        }
        false
    }

    /// The inside of `name(…)` when `predicate` is exactly that call.
    fn strip_call<'a>(predicate: &'a str, name: &str) -> Option<&'a str> {
        let rest = predicate.strip_prefix(name)?.trim_start();
        let inner = rest.strip_prefix('(')?.strip_suffix(')')?;
        // Reject `all(a)(b)`-shaped nonsense: the stripped suffix must be the matching paren.
        let mut depth = 0i32;
        for ch in inner.chars() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        return None;
                    }
                }
                _ => {}
            }
        }
        (depth == 0).then_some(inner)
    }

    /// Split on commas at paren depth zero.
    fn split_top_level(text: &str) -> Vec<&str> {
        let mut parts = Vec::new();
        let mut depth = 0i32;
        let mut start = 0usize;
        for (offset, ch) in text.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    parts.push(text[start..offset].trim());
                    start = offset + 1;
                }
                _ => {}
            }
        }
        let last = text[start..].trim();
        if !last.is_empty() {
            parts.push(last);
        }
        parts
    }

    /// Identifier-ish tokens of a predicate. String literals are already gone by the time a
    /// predicate reaches here from the post-strip sweep, and are dropped on the way in from an
    /// attribute, so `feature = "testkit"` never contributes a `test` token.
    fn tokens(text: &str) -> impl Iterator<Item = &str> {
        text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|token| !token.is_empty())
    }

    /// Rust source with comments, string / char literals, and test-only items removed.
    ///
    /// Test items have to go before anything else looks at the text: `candle-gen-krea`'s own
    /// preview tests call `run_flow_sampler` from a helper and quote the driver's name in a string
    /// literal and in prose, so a scan of the raw file would find call sites that do not exist in
    /// the shipped route and would mis-parse the ones that do.
    fn code_only(file: &str, source: &str) -> Stripped {
        let chars: Vec<char> = source.chars().collect();
        let mut out = String::with_capacity(source.len());
        let mut test_only_mods: Vec<String> = Vec::new();
        let mut i = 0usize;
        // `Some((bracket depth, has opened its top-level block))` while consuming the item a
        // test-only `cfg` attribute applies to, alongside the text consumed so far — an out-of-line
        // `mod NAME;` removed this way names a file the scan must not read either.
        let mut skipping: Option<(i32, bool)> = None;
        let mut skipped = String::new();

        while i < chars.len() {
            let ch = chars[i];

            if ch == '/' && chars.get(i + 1) == Some(&'/') {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            }
            if ch == '/' && chars.get(i + 1) == Some(&'*') {
                i += 2;
                let mut nesting = 1usize;
                while i < chars.len() && nesting > 0 {
                    if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                        nesting += 1;
                        i += 2;
                    } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                        nesting -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                assert_eq!(nesting, 0, "{file}: unterminated block comment");
                continue;
            }
            if let Some(end) = raw_string_end(file, &chars, i) {
                i = end;
                continue;
            }
            if ch == '"' {
                i += 1;
                let mut escaped = false;
                let mut closed = false;
                while i < chars.len() {
                    let c = chars[i];
                    i += 1;
                    if escaped {
                        escaped = false;
                    } else if c == '\\' {
                        escaped = true;
                    } else if c == '"' {
                        closed = true;
                        break;
                    }
                }
                assert!(closed, "{file}: unterminated string literal");
                continue;
            }
            // A `'` opens a char literal only when it closes; otherwise it is a lifetime.
            if ch == '\'' && (chars.get(i + 1) == Some(&'\\') || chars.get(i + 2) == Some(&'\'')) {
                i += 1;
                if chars.get(i) == Some(&'\\') {
                    i += 1;
                    while i < chars.len() && chars[i] != '\'' {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
                assert_eq!(
                    chars.get(i),
                    Some(&'\''),
                    "{file}: malformed character literal"
                );
                i += 1;
                continue;
            }
            if skipping.is_none() {
                // A file's inner attributes form a PROLOGUE, and a non-`cfg` one may sit ahead of the
                // file-level gate below — `#![allow(dead_code)]` before `#![cfg(test)]`, the shape
                // `candle-gen-flux/src/vae/diffusers.rs` already opens with. Emitting it would end the
                // whitespace-only run the gate keys on, so the gate would miss the `#![cfg(test)]`
                // behind it and the sweep at the bottom would hard-fail on a file that ships nothing.
                // Dropping it costs the scan nothing: an attribute is never a sampler or hook site.
                if out.chars().all(char::is_whitespace) && !matches_at(&chars, i, "#![cfg(") {
                    if let Some(end) = inner_attribute_end(file, &chars, i) {
                        i = end;
                        continue;
                    }
                }
                // A FILE-LEVEL inner `#![cfg(…)]` applies to the whole module, so a test-only one
                // means the file ships nothing at all — `candle-gen-instantid` and `candle-gen-pulid`
                // both open `src/validate.rs` that way (sc-16956 brought them into the scan through
                // BESPOKE_PROVIDER_CRATES). Recognised only while nothing shipped has been emitted
                // yet — comments, whitespace and the non-`cfg` inner attributes skipped just above all
                // preserve that run — which is the only position where it can mean "the whole file":
                // an inner attribute inside an inline `mod` applies to that module alone, and treating
                // it as file-scope would UNDER-scan.
                if let Some((predicate, end)) = inner_cfg_attribute(file, &chars, i) {
                    if out.chars().all(char::is_whitespace) {
                        return match classify_cfg(&predicate) {
                            CfgTest::TestOnly => Stripped {
                                code: String::new(),
                                test_only_mods: Vec::new(),
                            },
                            _ => {
                                let rest: String = chars[end..].iter().collect();
                                code_only(file, &rest)
                            }
                        };
                    }
                }
                if let Some((predicate, end)) = cfg_attribute(file, &chars, i) {
                    if classify_cfg(&predicate) == CfgTest::TestOnly {
                        i = end;
                        skipping = Some((0, false));
                        skipped.clear();
                        continue;
                    }
                    // A cfg that also ships stays. Preserve the whole attribute rather than
                    // letting the ordinary literal stripper erase its feature value: source-based
                    // catalog tests must distinguish `feature = "cuda"` from another feature.
                    out.extend(chars[i..end].iter());
                    i = end;
                    continue;
                }
            }

            if let Some((depth, entered)) = skipping.as_mut() {
                skipped.push(ch);
                let mut finished = false;
                match ch {
                    '(' | '[' | '{' => {
                        *depth += 1;
                        if ch == '{' && *depth == 1 {
                            *entered = true;
                        }
                    }
                    ')' | ']' | '}' => {
                        *depth -= 1;
                        assert!(*depth >= 0, "{file}: unbalanced test-only item");
                        finished = *depth == 0 && *entered;
                    }
                    // A test-only item is not always a block or a statement: the attribute is also
                    // used on a single struct field, struct-literal field, or enum variant, all of
                    // which end at a comma rather than a semicolon or a brace.
                    ';' | ',' if *depth == 0 => finished = true,
                    _ => {}
                }
                if finished {
                    skipping = None;
                    test_only_mods.extend(module_declarations(&skipped));
                    skipped.clear();
                }
                i += 1;
                continue;
            }

            out.push(ch);
            i += 1;
        }

        assert!(skipping.is_none(), "{file}: a test-only item never closed");
        // Belt and braces: the strip above only recognises a `cfg` in attribute position at the top
        // of the char stream. Sweep the survivor for any `cfg(…)` that classifies as test-only and
        // fail loudly rather than scan a test module as if it were shipped code.
        for predicate in surviving_cfg_predicates(file, &out) {
            assert_ne!(
                classify_cfg(&predicate),
                CfgTest::TestOnly,
                "{file}: a test-only cfg({predicate}) survived the strip — teach `code_only` \
                 about it"
            );
        }
        Stripped {
            code: out,
            test_only_mods,
        }
    }

    fn matches_at(chars: &[char], at: usize, needle: &str) -> bool {
        needle
            .chars()
            .enumerate()
            .all(|(offset, c)| chars.get(at + offset) == Some(&c))
    }

    fn is_ident_char(c: Option<&char>) -> bool {
        matches!(c, Some(c) if c.is_alphanumeric() || *c == '_')
    }

    /// The predicate text and the index just past the closing `]` of a `#[cfg(…)]` attribute
    /// starting at `at`, or `None` if one does not start there. String literals inside the
    /// predicate are dropped, so `feature = "testkit"` cannot contribute a `test` token.
    fn cfg_attribute(file: &str, chars: &[char], at: usize) -> Option<(String, usize)> {
        const OPEN: &str = "#[cfg(";
        if !matches_at(chars, at, OPEN) {
            return None;
        }
        let (predicate, past_paren) = balanced_predicate(file, chars, at + OPEN.chars().count());
        let mut i = past_paren;
        while chars.get(i).is_some_and(|c| c.is_whitespace()) {
            i += 1;
        }
        assert_eq!(
            chars.get(i),
            Some(&']'),
            "{file}: #[cfg({predicate})…] does not close with `]` — teach `code_only` about it"
        );
        Some((predicate, i + 1))
    }

    /// The predicate text and the index just past the closing `]` of an **inner** `#![cfg(…)]`
    /// attribute starting at `at`, or `None` if one does not start there.
    ///
    /// The inner form is what a whole-file gate looks like (`#![cfg(test)]` at the top of
    /// `src/validate.rs`); [`cfg_attribute`] only recognises the outer `#[cfg(…)]` that precedes an
    /// item, so before sc-16956 an inner one survived the strip and tripped the sweep below.
    fn inner_cfg_attribute(file: &str, chars: &[char], at: usize) -> Option<(String, usize)> {
        const OPEN: &str = "#![cfg(";
        if !matches_at(chars, at, OPEN) {
            return None;
        }
        let (predicate, past_paren) = balanced_predicate(file, chars, at + OPEN.chars().count());
        let mut i = past_paren;
        while chars.get(i).is_some_and(|c| c.is_whitespace()) {
            i += 1;
        }
        assert_eq!(
            chars.get(i),
            Some(&']'),
            "{file}: #![cfg({predicate})…] does not close with `]` — teach `code_only` about it"
        );
        Some((predicate, i + 1))
    }

    /// The index just past the closing `]` of **any** inner `#![…]` attribute starting at `at`, or
    /// `None` if one does not start there.
    ///
    /// Used to step over the non-`cfg` inner attributes of a file's prologue so they cannot displace
    /// the file-level `#![cfg(…)]` gate. Brackets nest and string literals (plain and raw) are
    /// skipped, so a `]` inside `#![doc = "…]…"]` does not close the attribute early.
    fn inner_attribute_end(file: &str, chars: &[char], at: usize) -> Option<usize> {
        if !matches_at(chars, at, "#![") {
            return None;
        }
        let mut depth = 0usize;
        let mut i = at + 2; // land on the `[`
        while i < chars.len() {
            if let Some(end) = raw_string_end(file, chars, i) {
                i = end;
                continue;
            }
            let ch = chars[i];
            if ch == '"' {
                i += 1;
                let mut escaped = false;
                let mut closed = false;
                while i < chars.len() {
                    let c = chars[i];
                    i += 1;
                    if escaped {
                        escaped = false;
                    } else if c == '\\' {
                        escaped = true;
                    } else if c == '"' {
                        closed = true;
                        break;
                    }
                }
                assert!(
                    closed,
                    "{file}: unterminated string inside an inner attribute"
                );
                continue;
            }
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                _ => {}
            }
            i += 1;
        }
        panic!("{file}: an inner `#![…]` attribute never closes — teach `code_only` about it");
    }

    /// Every `cfg(…)` predicate left in stripped code, wherever it sits.
    fn surviving_cfg_predicates(file: &str, code: &str) -> Vec<String> {
        let chars: Vec<char> = code.chars().collect();
        let mut predicates = Vec::new();
        let mut i = 0usize;
        while i < chars.len() {
            if matches_at(&chars, i, "cfg(") && !is_ident_char(chars.get(i.wrapping_sub(1))) {
                let (predicate, past) = balanced_predicate(file, &chars, i + 4);
                predicates.push(predicate);
                i = past;
                continue;
            }
            i += 1;
        }
        predicates
    }

    /// The text up to the paren matching the one already consumed, plus the index just past it.
    fn balanced_predicate(file: &str, chars: &[char], after_open: usize) -> (String, usize) {
        let mut predicate = String::new();
        let mut depth = 1usize;
        let mut i = after_open;
        while i < chars.len() {
            let ch = chars[i];
            if ch == '"' {
                i += 1;
                let mut escaped = false;
                let mut closed = false;
                while i < chars.len() {
                    let c = chars[i];
                    i += 1;
                    if escaped {
                        escaped = false;
                    } else if c == '\\' {
                        escaped = true;
                    } else if c == '"' {
                        closed = true;
                        break;
                    }
                }
                assert!(closed, "{file}: unterminated string inside a cfg predicate");
                continue;
            }
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return (predicate, i + 1);
                    }
                }
                _ => {}
            }
            predicate.push(ch);
            i += 1;
        }
        panic!("{file}: unterminated cfg predicate")
    }

    /// The index just past a raw string literal starting at `at` (`r"…"`, `r#"…"#`, `br#"…"#`), or
    /// `None` if one does not start there.
    fn raw_string_end(file: &str, chars: &[char], at: usize) -> Option<usize> {
        if at > 0 {
            let previous = chars[at - 1];
            if previous.is_alphanumeric() || previous == '_' {
                return None;
            }
        }
        let mut i = at;
        if chars.get(i) == Some(&'b') {
            i += 1;
        }
        if chars.get(i) != Some(&'r') {
            return None;
        }
        i += 1;
        let hashes = chars[i..].iter().take_while(|c| **c == '#').count();
        i += hashes;
        if chars.get(i) != Some(&'"') {
            return None;
        }
        i += 1;
        while i < chars.len() {
            if chars[i] == '"'
                && chars[i + 1..]
                    .iter()
                    .take(hashes)
                    .filter(|c| **c == '#')
                    .count()
                    == hashes
            {
                return Some(i + 1 + hashes);
            }
            i += 1;
        }
        panic!("{file}: unterminated raw string literal");
    }

    // ---- The shipped module tree -----------------------------------------------------------------

    /// The out-of-line `mod NAME;` declarations at the top level of one source file.
    ///
    /// Only depth-0 declarations are resolved: one nested inside an inline `mod` block resolves
    /// against a different directory, so it is deliberately left alone and the file it names then
    /// shows up as unreachable, which `module_tree` reports rather than skipping.
    fn module_declarations(code: &str) -> Vec<String> {
        let chars: Vec<char> = code.chars().collect();
        let mut names = Vec::new();
        let mut depth = 0i32;
        let mut i = 0usize;
        while i < chars.len() {
            match chars[i] {
                '{' => {
                    depth += 1;
                    i += 1;
                    continue;
                }
                '}' => {
                    depth -= 1;
                    i += 1;
                    continue;
                }
                _ => {}
            }
            let boundary = !is_ident_char(chars.get(i.wrapping_sub(1)));
            if depth == 0
                && boundary
                && matches_at(&chars, i, "mod")
                && !is_ident_char(chars.get(i + 3))
            {
                let mut j = i + 3;
                while chars.get(j).is_some_and(|c| c.is_whitespace()) {
                    j += 1;
                }
                let start = j;
                while is_ident_char(chars.get(j)) {
                    j += 1;
                }
                if j > start {
                    let name: String = chars[start..j].iter().collect();
                    while chars.get(j).is_some_and(|c| c.is_whitespace()) {
                        j += 1;
                    }
                    if chars.get(j) == Some(&';') {
                        names.push(name);
                        i = j + 1;
                        continue;
                    }
                }
            }
            i += 1;
        }
        names
    }

    /// One crate's shipped module tree plus the files only its test modules reach.
    struct ModuleTree {
        /// `src`-relative path → stripped code, for every file the crate compiles when it ships.
        shipped: BTreeMap<String, String>,
        /// `src`-relative paths reachable only through a test-only `mod` declaration.
        test_only: BTreeSet<String>,
    }

    fn relative_to(src: &Path, path: &Path) -> String {
        path.strip_prefix(src)
            .expect("walked from src")
            .to_string_lossy()
            .replace('\\', "/")
    }

    /// Resolve `mod NAME;` declared in `parent` against the declaring file's module directory.
    fn resolve_module(parent: &str, module_dir: &Path, name: &str) -> (PathBuf, PathBuf) {
        let flat = module_dir.join(format!("{name}.rs"));
        if flat.is_file() {
            return (flat, module_dir.join(name));
        }
        let nested = module_dir.join(name).join("mod.rs");
        if nested.is_file() {
            return (nested, module_dir.join(name));
        }
        panic!(
            "{parent}: `mod {name};` resolves to neither {} nor {} — the scan cannot read the \
             module's sources, and an unread file's emission fact would silently be `no`",
            flat.display(),
            nested.display()
        )
    }

    /// Walk one crate's module tree from `src/lib.rs`, following shipped `mod` declarations.
    ///
    /// This is the fix for the out-of-line test-module hole: `#[cfg(test)] mod NAME;` plus
    /// `src/NAME.rs` is pervasive here (20+ files across ten crates) and those files carry no
    /// `cfg` attribute of their own, so walking `src` blindly parses them as shipped code.
    fn module_tree(dir: &str, src: &Path) -> ModuleTree {
        let root = src.join("lib.rs");
        assert!(
            root.is_file(),
            "{dir}: no src/lib.rs — the module walk has no root, so the scan would read nothing"
        );

        let mut shipped: BTreeMap<String, String> = BTreeMap::new();
        let mut test_only: BTreeSet<String> = BTreeSet::new();
        let mut test_roots: Vec<(PathBuf, PathBuf)> = Vec::new();
        let mut pending = vec![(root, src.to_path_buf())];

        while let Some((path, module_dir)) = pending.pop() {
            let relative = relative_to(src, &path);
            if shipped.contains_key(&relative) {
                continue;
            }
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let stripped = code_only(&relative, &source);
            for name in module_declarations(&stripped.code) {
                pending.push(resolve_module(&relative, &module_dir, &name));
            }
            for name in &stripped.test_only_mods {
                test_roots.push(resolve_module(&relative, &module_dir, name));
            }
            shipped.insert(relative, stripped.code);
        }

        // A test-only module's own children are test-only too.
        while let Some((path, module_dir)) = test_roots.pop() {
            let relative = relative_to(src, &path);
            if !test_only.insert(relative.clone()) {
                continue;
            }
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let stripped = code_only(&relative, &source);
            for name in module_declarations(&stripped.code)
                .into_iter()
                .chain(stripped.test_only_mods)
            {
                test_roots.push(resolve_module(&relative, &module_dir, &name));
            }
        }

        for path in rust_sources(src) {
            let relative = relative_to(src, &path);
            assert!(
                shipped.contains_key(&relative) || test_only.contains(&relative),
                "{dir}: src/{relative} is reachable from neither a shipped nor a test-only `mod` \
                 declaration — the scan cannot classify it, and silently skipping a file would \
                 make its emission fact read as `no`"
            );
        }

        ModuleTree { shipped, test_only }
    }

    /// Whether one source file declares a `#[test] fn architecture_facts_*`.
    ///
    /// Deliberately reads the raw file rather than a stripped module tree: the declaration lives
    /// inside `#[cfg(test)]`, which the shipped-code scan removes by design. Requiring the `#[test]`
    /// attribute between the previous item and the name is what keeps a plain helper named
    /// `architecture_facts_for(...)` from passing for a test that runs.
    fn has_architecture_facts_test(path: &Path) -> bool {
        let source = std::fs::read_to_string(path).unwrap_or_default();
        source
            .match_indices("fn architecture_facts_")
            .any(|(index, _)| {
                let window = &source[index.saturating_sub(200)..index];
                window
                    .rfind("#[test]")
                    .is_some_and(|attribute| !window[attribute..].contains(" fn "))
            })
    }

    /// Every `.rs` file under `dir`, in a deterministic order.
    fn rust_sources(dir: &Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
            .map(|entry| entry.expect("readable directory entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                files.extend(rust_sources(&path));
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
        files
    }

    // ---- Reading the call sites ------------------------------------------------------------------

    /// The top-level, comma-separated arguments of one call, given the stripped code after its open
    /// paren. The window is bounded by the call's own **bracket balance** and ends at its closing
    /// paren; nothing about it keys off an argument's name or an argument's position from the end.
    ///
    /// Inline closures are ordinary arguments here, not the boundary. sc-16950's Krea inventory
    /// bounded its window at the first top-level `|` because every Krea site's last argument is the
    /// predict closure — but the trainer sites in several crates also pass their progress callback
    /// as an inline `&mut |_: Progress| {}`, so a first-`|` bound stops six arguments early and
    /// never reaches the preview argument at all. Consuming a closure's parameter list and letting
    /// its body ride the ordinary bracket balance is what makes one scanner work for every crate.
    fn call_arguments(site: &str, rest: &str) -> Vec<String> {
        let normalize = |text: &str| text.split_whitespace().collect::<Vec<_>>().join(" ");
        let chars: Vec<char> = rest.chars().collect();
        let mut args: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut depth = 1usize;
        let mut i = 0usize;

        while i < chars.len() {
            let ch = chars[i];
            i += 1;
            match ch {
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(ch);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    if depth == 0 {
                        // A trailing comma leaves nothing between it and the paren; a `predict`
                        // passed by name leaves the name.
                        let last = normalize(&current);
                        if !last.is_empty() {
                            args.push(last);
                        }
                        return args;
                    }
                    current.push(ch);
                }
                ',' if depth == 1 => {
                    args.push(normalize(&current));
                    current.clear();
                }
                '|' if depth == 1 => {
                    // A closure's parameter list: consume it whole, so the commas and the closing
                    // pipe inside it are never mistaken for the call's own.
                    while i < chars.len() && chars[i] != '|' {
                        i += 1;
                    }
                    assert!(
                        i < chars.len(),
                        "{site} has an unterminated closure parameter list"
                    );
                    i += 1;
                    current.push_str(" <closure> ");
                }
                _ => current.push(ch),
            }
        }
        panic!("{site} is unterminated: no closing paren before end of file")
    }

    /// Every shared-sampler call site in one stripped source file.
    fn sampler_sites(file: &str, code: &str) -> Vec<SamplerSite> {
        let mut sites = Vec::new();
        for driver in SAMPLER_DRIVERS {
            let mut cursor = 0usize;
            let mut index = 0usize;
            while let Some(offset) = code[cursor..].find(driver.call) {
                let args_start = cursor + offset + driver.call.len();
                let site = format!("{file}: {} call #{index}", driver.function);
                let args = call_arguments(&site, &code[args_start..]);
                // Exact arity, re-derived from the driver's own signature by
                // `the_sampler_driver_signatures_pin_the_preview_argument_position`. An inline
                // closure normalises to exactly one argument, so both call shapes give the same
                // count; accepting a second count would only widen the window in which a mis-split
                // lands silently on a neighbouring argument.
                assert_eq!(
                    args.len(),
                    driver.arity,
                    "{site}: expected {} arguments with the preview argument at position {}, \
                     parsed {args:?}",
                    driver.arity,
                    driver.preview_at
                );
                let argument = args[driver.preview_at].as_str();
                // Classified, not searched: a mis-split or a moved argument lands here as a loud
                // failure instead of a silent "no hook" (or a silent "hooked").
                let hooked = match argument {
                    "None" => false,
                    other if other.contains("preview") || other.contains("hook") => true,
                    other => panic!(
                        "{site}: cannot classify preview argument {other:?} — it must be `None` or \
                         name a preview hook"
                    ),
                };
                sites.push(SamplerSite {
                    file: file.to_string(),
                    driver: driver.function,
                    index,
                    hooked,
                });
                cursor = args_start;
                index += 1;
            }
        }
        sites
    }

    /// Count direct preview emissions without treating every unrelated `.emit(...)` method as a
    /// preview. Free preview functions have unique names. Method calls must have both the canonical
    /// preview receiver (`*hook*` / `*preview*`) and the exact `PreviewHook::{emit,emit_step}` arity.
    /// This remains a lexical source guard, but it is tied to the actual preview call context rather
    /// than to a generic Rust method name shared by request-local provenance sinks.
    fn direct_emission_count(file: &str, code: &str) -> usize {
        let mut count: usize = DIRECT_EMISSION_FUNCTION_CALLS
            .iter()
            .map(|call| code.matches(call).count())
            .sum();
        for &(call, arity) in DIRECT_EMISSION_METHOD_CALLS {
            let mut cursor = 0usize;
            while let Some(offset) = code[cursor..].find(call) {
                let method_at = cursor + offset;
                let receiver = method_receiver(&code[..method_at]);
                let args_start = method_at + call.len();
                let site = format!("{file}: {receiver}{call}");
                let args = call_arguments(&site, &code[args_start..]);
                if (receiver.contains("hook") || receiver.contains("preview"))
                    && args.len() == arity
                {
                    count += 1;
                }
                cursor = args_start;
            }
        }
        count
    }

    /// Last identifier before a method-call dot. The source is already stripped of comments and
    /// literals; whitespace between receiver and dot is accepted.
    fn method_receiver(before_dot: &str) -> &str {
        let trimmed = before_dot.trim_end();
        let start = trimmed
            .char_indices()
            .rev()
            .find_map(|(index, ch)| (!ch.is_ascii_alphanumeric() && ch != '_').then_some(index + 1))
            .unwrap_or(0);
        &trimmed[start..]
    }

    /// Read one provider crate's preview wiring out of its **shipped** module tree.
    ///
    /// Keyed on the directory rather than on a `ProviderCrate`, so the descriptor-less
    /// [`BespokeCrate`] rows go through the very same scan.
    fn scan(dir: &str) -> CrateWiring {
        let src = candle_gen_root().join(dir).join("src");
        assert!(
            src.is_dir(),
            "{}: no src directory — the wiring table names a crate that does not exist, so its \
             emission fact would silently read as `no`",
            src.display()
        );
        let tree = module_tree(dir, &src);
        let mut wiring = CrateWiring {
            sites: Vec::new(),
            direct: Vec::new(),
            scanned: Vec::new(),
            excluded: tree.test_only.iter().cloned().collect(),
        };
        for (relative, code) in &tree.shipped {
            wiring.sites.extend(sampler_sites(relative, code));
            let count = direct_emission_count(relative, code);
            if count > 0 {
                wiring.direct.push(DirectEmission {
                    file: relative.clone(),
                    count,
                });
            }
            wiring.scanned.push(relative.clone());
        }
        assert!(
            !wiring.scanned.is_empty(),
            "{}: no Rust sources found — an empty scan reads as `does not emit` and would make \
             every assertion below vacuous",
            src.display()
        );
        wiring
    }

    /// One file's route tally — the unit both the sources and the wiring table are reduced to.
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct FileTally {
        file: String,
        hooked: usize,
        direct: usize,
        /// `(driver, occurrence index)` of every site passing the literal `None`.
        dark: Vec<(String, usize)>,
    }

    /// What the sources resolve to, one row per file that drives a sampler or emits directly.
    fn derived_tallies(wiring: &CrateWiring) -> Vec<FileTally> {
        let mut by_file: BTreeMap<&str, FileTally> = BTreeMap::new();
        for site in &wiring.sites {
            let row = by_file
                .entry(site.file.as_str())
                .or_insert_with(|| FileTally {
                    file: site.file.clone(),
                    hooked: 0,
                    direct: 0,
                    dark: Vec::new(),
                });
            if site.hooked {
                row.hooked += 1;
            } else {
                row.dark.push((site.driver.to_string(), site.index));
            }
        }
        for emission in &wiring.direct {
            let row = by_file
                .entry(emission.file.as_str())
                .or_insert_with(|| FileTally {
                    file: emission.file.clone(),
                    hooked: 0,
                    direct: 0,
                    dark: Vec::new(),
                });
            row.direct += emission.count;
        }
        let mut tallies: Vec<FileTally> = by_file.into_values().collect();
        for tally in &mut tallies {
            tally.dark.sort();
        }
        tallies.sort();
        tallies
    }

    /// What the wiring table declares, in the same shape.
    fn declared_tallies(routes: &[FileRoutes]) -> Vec<FileTally> {
        let mut tallies: Vec<FileTally> = routes
            .iter()
            .map(|routes| FileTally {
                file: routes.file.to_string(),
                hooked: routes.hooked,
                direct: routes.direct,
                dark: {
                    let mut dark: Vec<(String, usize)> = routes
                        .dark
                        .iter()
                        .map(|site| (site.driver.to_string(), site.index))
                        .collect();
                    dark.sort();
                    dark
                },
            })
            .collect();
        tallies.sort();
        tallies
    }

    /// The generator ids one provider crate registers, read back from a registry built from it
    /// alone rather than restated in the table above.
    fn ids_of(provider: &ProviderCrate) -> Vec<String> {
        (provider.register)(ProviderRegistryBuilder::new())
            .build()
            .unwrap_or_else(|error| panic!("{}: {error}", provider.dir))
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect()
    }

    /// Every shipped generator id that advertises preview support.
    fn advertising_ids() -> BTreeSet<String> {
        super::provider_registry()
            .expect("catalog")
            .generators()
            .map(|registration| (registration.descriptor)())
            .filter(|descriptor| descriptor.capabilities.supports_preview)
            .map(|descriptor| descriptor.id.to_string())
            .collect()
    }

    // ---- The guard -------------------------------------------------------------------------------

    /// The ported MLX guard: the declared allowlist and the shipped descriptors must agree exactly,
    /// in **both** directions, on exact ids — a wired route that does not advertise, an advertising
    /// route that is not wired, and an allowlist entry naming an id this platform does not ship are
    /// all failures. Weights-free: descriptors only, no snapshot and no device.
    #[test]
    fn preview_capability_matches_every_wired_shipped_route_bidirectionally() {
        let registry = super::provider_registry().expect("catalog");
        let descriptors: Vec<_> = registry
            .generators()
            .map(|registration| (registration.descriptor)())
            .collect();

        for id in PREVIEW_ROUTE_IDS {
            let descriptor = descriptors
                .iter()
                .find(|descriptor| descriptor.id == *id)
                .unwrap_or_else(|| panic!("preview allowlist contains unshipped provider {id}"));
            assert!(
                descriptor.capabilities.supports_preview,
                "wired preview provider {id} must advertise support"
            );
        }

        let expected: BTreeSet<String> = PREVIEW_ROUTE_IDS
            .iter()
            .map(|id| (*id).to_string())
            .collect();
        assert_eq!(
            expected.len(),
            PREVIEW_ROUTE_IDS.len(),
            "the preview allowlist must not repeat an id"
        );
        assert_eq!(
            advertising_ids(),
            expected,
            "only providers with an actual PreviewSink denoise route may advertise support"
        );
    }

    /// **The two SANA routes drive two different sampler drivers, and both must be hooked**
    /// (sc-16959).
    ///
    /// The per-id half of what this row originally carried has been **generalised** into
    /// `source_level_wiring_and_advertised_capability_agree_for_every_provider_crate`, because the
    /// hole it patched was never SANA's: the bidirectional guard above is a set comparison, and the
    /// source-level guard only required *some* id of a wired crate to advertise, so on any of the ten
    /// multi-id crates one route could be dropped from both sides while its siblings covered for it.
    /// That is now checked for every registered id of every wired crate. See that row for the sd3
    /// mutation that proved it live on merged code.
    ///
    /// What stays here is the part that is genuinely SANA's and that nothing else asserts: this is
    /// the only candle family in the epic that drives **two** shared sampler drivers, so the crate
    /// must hook `run_flow_sampler` (base, true-CFG flow-match over DC-AE 1.0) *and*
    /// `run_scm_sampler` (Sprint, CFG-free SCM over DC-AE 1.1) — one site each, no dark site. A
    /// single hooked site would leave one route emitting nothing while both ids kept advertising,
    /// and that is the mistake this story exists to avoid: the two routes carry different committed
    /// fits, so one can never stand in for the other.
    ///
    /// The two ids are still checked on their own terms below — registered by `candle-gen-sana`
    /// itself rather than by some other crate that happened to claim the id, and exactly two of
    /// them — because the generalised row derives its id list from the registry and so cannot pin
    /// *which* crate a given id belongs to.
    #[test]
    fn sana_base_and_sprint_are_two_independent_rows() {
        let sana = PROVIDER_CRATES
            .iter()
            .find(|provider| provider.dir == "candle-gen-sana")
            .expect("candle-gen-sana must be in the wiring table");
        let registered = ids_of(sana);
        let advertising = advertising_ids();

        for id in SANA_ROUTE_IDS {
            assert!(
                PREVIEW_ROUTE_IDS.contains(&id),
                "{id} must be its OWN row in PREVIEW_ROUTE_IDS — the base flow route and the Sprint \
                 SCM route are different drivers over different latent spaces and neither covers the \
                 other"
            );
            assert!(
                registered.iter().any(|registered| registered == id),
                "{id} must be registered by candle-gen-sana itself, not merely named in a list"
            );
            assert!(
                advertising.contains(id),
                "{id} must advertise supports_preview on its own descriptor"
            );
        }
        assert_eq!(
            registered.len(),
            SANA_ROUTE_IDS.len(),
            "candle-gen-sana registers exactly the two routes this row accounts for: {registered:?}"
        );

        // And the crate really does drive BOTH shared samplers with a hook — one site each. A single
        // hooked site would mean one of the two routes is dark while the other keeps both ids
        // advertising, which is the exact shape the two rows above exist to make visible.
        let wiring = scan(sana.dir);
        let mut hooked: Vec<&str> = wiring
            .sites
            .iter()
            .filter(|site| site.hooked)
            .map(|site| site.driver)
            .collect();
        hooked.sort_unstable();
        assert_eq!(
            hooked,
            vec!["run_flow_sampler", "run_scm_sampler"],
            "candle-gen-sana must hook the flow driver (base) AND the SCM driver (Sprint); got \
             {hooked:?}"
        );
        assert!(
            wiring.sites.iter().all(|site| site.hooked),
            "candle-gen-sana declares no dark site, so every sampler call it makes must be hooked"
        );
    }

    /// The half the MLX guard does not have: the allowlist above is checked against the **sources**,
    /// so it cannot be satisfied by editing lists.
    ///
    /// For every registered provider crate, whether it emits is derived from its own code — a
    /// sampler call site that passes a hook, or a bespoke loop making a direct emission call — and
    /// that fact must agree with whether **every one of** its ids advertises. Both directions fail: a
    /// descriptor flipped ahead of the wiring, and a family wired without flipping its descriptors.
    /// The second is what makes sc-16952…sc-16960 self-enforcing.
    ///
    /// ## Why the wired branch is per id rather than "any id" (sc-16959 review)
    ///
    /// This row originally asserted only that a wired crate had *some* advertising id, and
    /// `preview_capability_matches_every_wired_shipped_route_bidirectionally` is a **set** equality —
    /// so dropping one id from `PREVIEW_ROUTE_IDS` *and* from its descriptor left both green, with
    /// that route's siblings covering for it. That is not hypothetical: on merged code, flipping
    /// `candle-gen-sd3`'s `supports_preview` to `!matches!(variant, Variant::Medium)` and deleting
    /// `"sd3_5_medium"` from the allowlist took this whole suite through with **zero** failures,
    /// while `sd3_5_medium` — which reaches the same hooked `run_flow_sampler` site as its two
    /// siblings — silently stopped advertising a capability it has.
    ///
    /// Eleven crates ship more than one id and were all exposed: krea ×3, anima ×3, chroma ×3,
    /// sd3_5 ×3, flux ×2, flux2 ×2, lens ×2, ideogram ×2, z-image ×2, sana ×2, sensenova ×2. So the
    /// check is now **per registered id**: every id of a crate whose sources emit must be in the
    /// allowlist *and* advertising, on its own. Fourteen wired crates register 29 ids between them,
    /// which is exactly `PREVIEW_ROUTE_IDS.len()` — the two halves meet with nothing left over.
    ///
    /// `sana_base_and_sprint_are_two_independent_rows` is kept alongside this, not subsumed by it:
    /// its load-bearing half is the driver **pair** (`run_flow_sampler` *and* `run_scm_sampler` from
    /// one crate), which is SANA-specific and which nothing else in this module asserts.
    #[test]
    fn source_level_wiring_and_advertised_capability_agree_for_every_provider_crate() {
        let advertising = advertising_ids();
        for provider in PROVIDER_CRATES {
            let wiring = scan(provider.dir);
            let ids = ids_of(provider);
            let advertised: Vec<&String> =
                ids.iter().filter(|id| advertising.contains(*id)).collect();

            if wiring.emits() {
                let hooked: Vec<&str> = wiring
                    .sites
                    .iter()
                    .filter(|site| site.hooked)
                    .map(|site| site.file.as_str())
                    .collect();
                let direct: Vec<&str> = wiring
                    .direct
                    .iter()
                    .map(|emission| emission.file.as_str())
                    .collect();
                assert!(
                    !advertised.is_empty(),
                    "{} emits previews (hooked sites: {hooked:?}, direct emission: {direct:?}) but \
                     none of its routes {ids:?} advertise supports_preview — flip the descriptors \
                     and add the ids to PREVIEW_ROUTE_IDS in the same PR",
                    provider.dir
                );
                // Per id, so a sibling cannot cover for a route that quietly stopped advertising.
                for id in &ids {
                    let siblings: Vec<&String> = ids.iter().filter(|other| *other != id).collect();
                    assert!(
                        PREVIEW_ROUTE_IDS.contains(&id.as_str()),
                        "{} emits previews (hooked sites: {hooked:?}, direct emission: {direct:?}) \
                         but its route {id} is missing from PREVIEW_ROUTE_IDS — every id a wired \
                         crate registers is its OWN row, and its siblings {siblings:?} do not cover \
                         for it. If this route genuinely cannot preview, it does not belong in a \
                         wired crate's registration; say so in a story rather than dropping the row",
                        provider.dir
                    );
                    assert!(
                        advertising.contains(id),
                        "{} emits previews (hooked sites: {hooked:?}, direct emission: {direct:?}) \
                         but its route {id} does not advertise supports_preview on its own \
                         descriptor — the bidirectional row above is a SET comparison, so its \
                         siblings {siblings:?} satisfy it while this route goes dark to every UI \
                         that reads the capability",
                        provider.dir
                    );
                }
            } else {
                assert!(
                    advertised.is_empty(),
                    "{} advertises supports_preview on {advertised:?} but nothing in its shipped \
                     sources emits: no sampler call site passes a preview hook and no bespoke loop \
                     calls the direct preview functions/methods",
                    provider.dir
                );
            }
        }
    }

    /// The declared denoise shape must match the sources, so a crate cannot lose its call sites —
    /// to a refactor, a rename, or a scanner that resolved nothing — and read as "does not emit".
    ///
    /// It is also what lets a bespoke family be **verified** rather than hard-failed: a crate with
    /// no shared-driver site is not broken, it is `Bespoke`, and its wiring shows up as a direct
    /// emission call instead.
    #[test]
    fn the_wiring_table_pins_how_each_crate_denoises() {
        for (dir, denoise, _) in scanned_crates() {
            let wiring = scan(dir);
            let sites = wiring.sites.len();
            match denoise {
                Denoise::Shared => assert!(
                    sites > 0,
                    "{dir} is declared Denoise::Shared but its shipped sources drive no sampler \
                     driver at all — either it moved to a bespoke loop (say so in the table) or \
                     the scan is reading the wrong files"
                ),
                Denoise::Bespoke => assert_eq!(
                    sites, 0,
                    "{dir} is declared Denoise::Bespoke but its shipped sources drive a shared \
                     sampler — say Denoise::Shared so its sites are inventoried"
                ),
            }
        }
    }

    /// The driver table is re-derived from `candle-gen/src/sampler.rs` rather than trusted: the
    /// `preview` parameter must sit at the declared position of a signature with the declared arity.
    /// A signature change fails here — loudly, in one place — instead of silently shifting which
    /// argument `preview_at` names at every call site in the workspace.
    ///
    /// `run_av_curated_sampler` is pinned in the negative for the same reason (the LTX joint
    /// video+audio driver, which the epic-16624 holdout set leaves preview-inert): it is excluded
    /// from the scan **because** it has no `preview` parameter, so if sc-16949's hook is ever
    /// extended to it, this is what notices.
    #[test]
    fn the_sampler_driver_signatures_pin_the_preview_argument_position() {
        let path = candle_gen_root()
            .join("candle-gen")
            .join("src")
            .join("sampler.rs");
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let code = code_only("candle-gen/src/sampler.rs", &source).code;

        for driver in SAMPLER_DRIVERS {
            let parameters = function_parameters(&code, driver.function);
            assert_eq!(
                parameters.len(),
                driver.arity,
                "{} takes {parameters:?}, not {} arguments — SAMPLER_DRIVERS is stale",
                driver.function,
                driver.arity
            );
            assert_eq!(
                parameters[driver.preview_at], "preview",
                "{}'s parameter {} is {:?}, not `preview` — SAMPLER_DRIVERS is stale and every \
                 call-site classification above is reading the wrong argument",
                driver.function, driver.preview_at, parameters[driver.preview_at]
            );
        }

        let unhooked = function_parameters(&code, UNHOOKED_SAMPLER_DRIVER);
        assert!(
            !unhooked.iter().any(|parameter| parameter == "preview"),
            "{UNHOOKED_SAMPLER_DRIVER} has grown a `preview` parameter ({unhooked:?}) — it is \
             absent from SAMPLER_DRIVERS only because it had none, so add it there (with its \
             position) before any family can hook it"
        );
    }

    /// The parameter names of `fn name(…)` in stripped source, in order.
    fn function_parameters(code: &str, name: &str) -> Vec<String> {
        let needle = format!("fn {name}(");
        let at = code
            .find(&needle)
            .unwrap_or_else(|| panic!("{name} is no longer declared in candle-gen/src/sampler.rs"));
        call_arguments(&format!("fn {name}"), &code[at + needle.len()..])
            .into_iter()
            .map(|parameter| {
                let (binding, _) = parameter.split_once(':').unwrap_or_else(|| {
                    panic!("{name}: cannot read a parameter name out of {parameter:?}")
                });
                binding.trim().trim_start_matches("mut ").trim().to_string()
            })
            .collect()
    }

    /// A file-level `#![cfg(test)]` is found even when other inner attributes sit ahead of it.
    ///
    /// The gate keys on "nothing shipped has been emitted yet", so before this row any preceding
    /// inner attribute ended that run and hid the `#![cfg(test)]` behind it — and the post-strip
    /// sweep then hard-failed on a file that ships nothing at all. `#![allow(dead_code)]` ahead of a
    /// module gate is a real shape here: `candle-gen-flux/src/vae/diffusers.rs` opens with one.
    ///
    /// The last two cases keep the skip honest in the other direction — a file-level `cfg` that
    /// *does* ship still yields its code, and an inner attribute is only stepped over in the
    /// prologue, never once shipped code has been seen.
    #[test]
    fn a_file_level_cfg_test_is_found_behind_other_inner_attributes() {
        let stripped = |source: &str| code_only("synthetic.rs", source).code;
        let body = "fn denoise() { candle_gen::run_flow_sampler(preview); }";

        // The regression this row exists for, plus the bare form it must not disturb.
        for prologue in [
            "#![allow(dead_code)]\n",
            "",
            "//! docs\n#![allow(dead_code)]\n#![allow(clippy::too_many_arguments)]\n",
            "#![doc = \"a ] bracket in a string\"]\n",
        ] {
            let code = stripped(&format!("{prologue}#![cfg(test)]\n{body}\n"));
            assert!(
                code.trim().is_empty(),
                "a file gated test-only behind {prologue:?} must strip to nothing, got {code:?}"
            );
        }

        // A shipping file-level cfg still yields its body, prologue or not.
        let code = stripped(&format!(
            "#![allow(dead_code)]\n#![cfg(feature = \"cuda\")]\n{body}\n"
        ));
        assert!(
            code.contains("run_flow_sampler"),
            "a cuda-gated file ships — its sampler site must survive the strip, got {code:?}"
        );

        // Non-vacuity for the "prologue only" half: an inner attribute inside an inline `mod` after
        // shipped code applies to that module alone. Treating it as file-scope would UNDER-scan.
        let code = stripped(&format!("{body}\nmod inner {{ #![allow(dead_code)] }}\n"));
        assert!(
            code.contains("run_flow_sampler"),
            "an inner attribute below shipped code must not blank the file, got {code:?}"
        );
    }

    /// A wired crate must be wired **everywhere**, not just somewhere: once a crate emits at all,
    /// every one of its sampler sites has to pass a hook unless that exact site — file, driver, and
    /// occurrence index — is declared dark with a reason. That is what stops a family from wiring
    /// one route, flipping every descriptor, and leaving the rest of its routes silently blank.
    ///
    /// Declarations are checked in the other two directions too: a declared site that is no longer
    /// dark, and a dark declaration on a crate that is not wired at all, are both stale.
    #[test]
    fn a_wired_crate_leaves_no_undeclared_dark_sampler_site() {
        for (dir, _, routes) in scanned_crates() {
            let wiring = scan(dir);
            let declared: BTreeSet<(&str, &str, usize)> = routes
                .iter()
                .flat_map(|routes| {
                    routes
                        .dark
                        .iter()
                        .map(move |site| (routes.file, site.driver, site.index))
                })
                .collect();

            for routes in routes {
                for site in routes.dark {
                    assert!(
                        !site.reason.trim().is_empty(),
                        "{dir}: {} {} #{} is declared dark with no reason",
                        routes.file,
                        site.driver,
                        site.index
                    );
                }
            }

            if !wiring.emits() {
                assert!(
                    declared.is_empty(),
                    "{dir} declares dark sampler sites {declared:?} but emits no previews at all — a \
                     dark declaration only means something on a wired crate"
                );
                continue;
            }

            let undeclared: Vec<String> = wiring
                .sites
                .iter()
                .filter(|site| {
                    !site.hooked
                        && !declared.contains(&(site.file.as_str(), site.driver, site.index))
                })
                .map(|site| format!("{}: {} #{}", site.file, site.driver, site.index))
                .collect();
            assert!(
                undeclared.is_empty(),
                "{dir} is wired for previews but these sampler sites still pass `None`: \
                 {undeclared:?} — pass a hook, or declare that exact site in its file's `dark` list \
                 with the reason it emits nothing"
            );

            for (file, driver, index) in &declared {
                assert!(
                    wiring.sites.iter().any(|site| {
                        site.file == *file
                            && site.driver == *driver
                            && site.index == *index
                            && !site.hooked
                    }),
                    "{dir}: {file} declares {driver} #{index} dark, but that site no longer passes \
                     `None` — remove the stale declaration"
                );
            }
        }
    }

    /// The route inventory: a wired crate pins **exactly** how many sites each of its files hooks,
    /// how many direct emissions it makes, and which sites are dark — and an unwired crate pins
    /// nothing, because it has nothing to pin.
    ///
    /// This is the assertion that keeps a wired family honest after it lands. Everything else is an
    /// equality between "does this crate emit" and "does it advertise", which stays true when a
    /// family wires six of seven routes. Only an exact count notices the seventh, and only counts
    /// kept per file survive the obvious workaround of declaring the whole file dark. It doubles as
    /// the non-vacuity pin: a scanner that silently resolved nothing reads as "does not emit",
    /// which agrees with every *unwired* crate, so a wired crate's positive numbers are what prove
    /// the scan resolves anything at all.
    ///
    /// **Each Tier 1 story adds its own family's rows here, in its own PR.**
    #[test]
    fn every_wired_crate_pins_its_exact_route_inventory() {
        let mut wired = 0usize;
        let mut hooked_sites = 0usize;
        for (dir, _, routes) in scanned_crates() {
            let wiring = scan(dir);
            if !wiring.emits() {
                assert!(
                    routes.is_empty(),
                    "{dir} pins a route inventory but emits no previews — an inventory only means \
                     something on a wired crate"
                );
                continue;
            }
            wired += 1;
            hooked_sites += wiring.sites.iter().filter(|site| site.hooked).count();
            assert_eq!(
                derived_tallies(&wiring),
                declared_tallies(routes),
                "{dir}: the route inventory in the wiring table disagrees with the crate's sources. \
                 Every file that drives a sampler or emits directly needs a row with exact counts \
                 — blanking one route of an already-inventoried file must be a diff here too."
            );
        }
        assert!(
            wired > 0 && hooked_sites > 0,
            "no crate resolved as wired ({wired} crates, {hooked_sites} hooked sites) — every \
             assertion in this module would then be vacuously satisfied by a scanner that read \
             nothing"
        );
    }

    #[test]
    fn direct_preview_scan_ignores_prompt_report_emit_but_counts_preview_context() {
        let code = r#"
            req.prompt_enhancement.emit(report);
            hook.emit(&counter, &sigmas, sigma, &latents);
            preview.emit_step(&counter, step, &state);
            candle_gen::preview::emit_preview_at(&sink, &counter, step, || frame());
            audit_hook.emit(one, two, three);
        "#;
        assert_eq!(direct_emission_count("synthetic.rs", code), 3);
    }

    /// Out-of-line `#[cfg(test)] mod NAME;` files are excluded from the scan.
    ///
    /// They carry no `cfg` attribute of their own, so a blind walk of `src` parses them as shipped
    /// code — in both harmful directions. A hooked sampler call in `candle-gen-z-image`'s
    /// `#[ignore]`d GPU-validation modules would make the crate read as emitting and **demand** a
    /// false `supports_preview` advertisement on `z_image` / `z_image_turbo`; a `None` call in
    /// `candle-gen-krea`'s `testfix.rs` would force a test fixture into a dark declaration. Six of
    /// the ten affected crates are direct sc-16952 / sc-16954 / sc-16955 / sc-16957 / sc-16958
    /// targets.
    #[test]
    fn out_of_line_cfg_test_modules_are_not_scanned_as_shipped_code() {
        let expectations: &[(&str, &[&str], &[&str])] = &[
            ("candle-gen-krea", &["testfix.rs"], &["pipeline.rs"]),
            (
                "candle-gen-z-image",
                &[
                    "base_img2img_validate.rs",
                    "control_validate.rs",
                    "edit_validate.rs",
                    "turbo_img2img_validate.rs",
                ],
                &["lib.rs"],
            ),
            (
                "candle-gen-qwen-image",
                &[
                    "comfyui_vae_validate.rs",
                    "control_fun_validate.rs",
                    "edit_validate.rs",
                    "vision_validate.rs",
                ],
                &["lib.rs"],
            ),
            (
                "candle-gen-sdxl",
                &["edit_validate.rs", "ip_validate.rs"],
                &["lib.rs"],
            ),
            ("candle-gen-sd3", &["img2img_validate.rs"], &["lib.rs"]),
            ("candle-gen-boogu", &["img2img_validate.rs"], &["lib.rs"]),
            ("candle-gen-flux", &["ip_validate.rs"], &["lib.rs"]),
            (
                "candle-gen-kolors",
                &["control_validate.rs", "ip_validate.rs"],
                &["lib.rs"],
            ),
            ("candle-gen-bernini", &["testfix.rs"], &["lib.rs"]),
        ];

        for (dir, test_only, shipped) in expectations {
            assert!(
                PROVIDER_CRATES.iter().any(|provider| provider.dir == *dir),
                "{dir} is in the wiring table"
            );
            let wiring = scan(dir);
            let src = candle_gen_root().join(dir).join("src");
            for file in *test_only {
                assert!(
                    src.join(file).is_file(),
                    "{dir}/src/{file} no longer exists — this expectation is stale and proves \
                     nothing"
                );
                assert!(
                    wiring.excluded.iter().any(|excluded| excluded == file),
                    "{dir}: src/{file} is an out-of-line test module and must not be scanned as \
                     shipped code, but the scan read it (excluded: {:?})",
                    wiring.excluded
                );
                assert!(
                    !wiring.scanned.iter().any(|scanned| scanned == file),
                    "{dir}: src/{file} was scanned as shipped code"
                );
            }
            for file in *shipped {
                assert!(
                    wiring.scanned.iter().any(|scanned| scanned == file),
                    "{dir}: src/{file} must be scanned as shipped code, but was not (scanned: \
                     {:?})",
                    wiring.scanned
                );
            }
        }
    }

    /// Cross-check against the *other* scanner: `candle-gen-krea` pins the same inventory against
    /// its own sources in sc-16950 (`candle-gen-krea/src/preview.rs`), by different code with a
    /// different bounding rule. Two independent readings of one crate agreeing is what makes this
    /// module's numbers evidence rather than a restatement of its own parse.
    #[test]
    fn the_source_scan_resolves_the_krea_wiring_it_claims_to_verify() {
        assert!(
            PROVIDER_CRATES
                .iter()
                .any(|provider| provider.dir == "candle-gen-krea"),
            "Krea is in the wiring table"
        );
        let wiring = scan("candle-gen-krea");

        let tally = |hooked: bool| {
            let mut by_file: Vec<(String, usize)> = Vec::new();
            for site in wiring.sites.iter().filter(|site| site.hooked == hooked) {
                match by_file.iter_mut().find(|(file, _)| *file == site.file) {
                    Some((_, count)) => *count += 1,
                    None => by_file.push((site.file.clone(), 1)),
                }
            }
            by_file.sort();
            by_file
        };

        assert_eq!(
            tally(true),
            [
                ("control_provider.rs".to_string(), 1),
                ("pipeline.rs".to_string(), 7),
            ],
            "the seven pipeline render routes plus the pose-control route must all pass a hook"
        );
        assert_eq!(
            tally(false),
            [("training.rs".to_string(), 1)],
            "the trainer sample render is the only Krea sampler site with no sink to emit into"
        );
        assert!(
            wiring.direct.is_empty(),
            "Krea emits through the shared driver, not through a bespoke loop"
        );
        assert!(wiring.emits());
    }

    /// Every generator the catalog ships is covered by the wiring table, so a new provider crate
    /// cannot join the platform with its preview advertising unguarded.
    #[test]
    fn every_shipped_generator_is_covered_by_the_wiring_table() {
        let shipped: BTreeSet<String> = super::provider_registry()
            .expect("catalog")
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let mut covered: BTreeSet<String> = BTreeSet::new();
        for provider in PROVIDER_CRATES {
            for id in ids_of(provider) {
                assert!(
                    covered.insert(id.clone()),
                    "{id} is registered by more than one crate in the wiring table"
                );
            }
        }
        assert_eq!(
            covered, shipped,
            "PROVIDER_CRATES must cover exactly the shipped generator surface"
        );
    }

    /// Every descriptor-less provider crate is inventoried, and the two tables are disjoint.
    ///
    /// [`super::BESPOKE_UTILITY_CRATES`] is the shipped list of crates consumed through a
    /// provider-specific API rather than through the registry, and it is what
    /// `runtime-{cpu,cuda}`'s surface tests pin. Deriving this table's membership from it — rather
    /// than restating six names — is what stops a seventh utility crate from arriving with a denoise
    /// loop and no guard, which is precisely how `candle-gen-pulid` would have landed before sc-16956.
    ///
    /// The disjointness half matters too: a crate that appeared in both tables would be scanned twice
    /// and could satisfy `every_wired_crate_pins_its_exact_route_inventory` from whichever row happened
    /// to be right.
    #[test]
    fn every_bespoke_utility_crate_is_covered_by_the_bespoke_table() {
        let expected: BTreeSet<String> = super::BESPOKE_UTILITY_CRATES
            .iter()
            .map(|name| format!("candle-gen-{name}"))
            .collect();
        let declared: BTreeSet<String> = BESPOKE_PROVIDER_CRATES
            .iter()
            .map(|provider| provider.dir.to_string())
            .collect();
        assert_eq!(
            declared.len(),
            BESPOKE_PROVIDER_CRATES.len(),
            "the bespoke table must not repeat a crate"
        );
        assert_eq!(
            declared, expected,
            "BESPOKE_PROVIDER_CRATES must cover exactly the descriptor-less crates \
             BESPOKE_UTILITY_CRATES names — one of them (candle-gen-pulid) drives a sampler, so an \
             uncovered member could be wired with no guard at all"
        );

        let registered: BTreeSet<String> = PROVIDER_CRATES
            .iter()
            .map(|provider| provider.dir.to_string())
            .collect();
        assert!(
            registered.is_disjoint(&declared),
            "a crate may appear in only one wiring table: {:?}",
            registered.intersection(&declared).collect::<Vec<_>>()
        );

        // Non-vacuity: exactly one member of this table is wired today, and it is PuLID. A scan that
        // resolved nothing would read every row as unwired and satisfy every assertion above.
        let wired: Vec<&str> = BESPOKE_PROVIDER_CRATES
            .iter()
            .filter(|provider| scan(provider.dir).emits())
            .map(|provider| provider.dir)
            .collect();
        assert_eq!(
            wired,
            ["candle-gen-pulid"],
            "sc-16956 wired PuLID and nothing else in this table"
        );
    }

    #[test]
    fn every_descriptorless_memory_route_has_one_machine_readable_waiver() {
        let waivers = super::BESPOKE_MEMORY_ROUTE_WAIVERS;

        // Shape, not population. This used to open with `waivers.len() == 1` and then index
        // `waivers[0]`, so a second legitimate waiver would go RED without any coverage having been
        // lost. The load-bearing claim is the derived set below: the waiver table names exactly the
        // descriptor-less crates the source scan finds emitting — and every row is then validated on
        // its own terms, so a second waiver is checked rather than merely counted.
        let waived_crates = waivers
            .iter()
            .map(|waiver| format!("candle-gen-{}", waiver.crate_name))
            .collect::<BTreeSet<_>>();
        let descriptorless_memory_crates = BESPOKE_PROVIDER_CRATES
            .iter()
            .filter(|provider| scan(provider.dir).emits())
            .map(|provider| provider.dir.to_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            waived_crates, descriptorless_memory_crates,
            "the waiver table must name exactly the descriptor-less crates that emit"
        );
        assert_eq!(
            waived_crates.len(),
            waivers.len(),
            "the waiver table must not repeat a crate"
        );
        // Non-vacuity: an empty scan would satisfy an empty-set equality while proving nothing.
        assert!(
            !waivers.is_empty(),
            "at least one descriptor-less memory route is waived today; an empty table means the \
             source scan resolved nothing"
        );

        let registry = super::memory_contract_surface_registry().unwrap();
        // `candle_gen_root()` is `crates/media/candle-gen`; the waiver paths are repo-relative.
        let repo_root = candle_gen_root()
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .expect("crates/media/candle-gen sits three levels below the repo root")
            .to_path_buf();
        for waiver in waivers {
            assert!(
                super::BESPOKE_UTILITY_CRATES.contains(&waiver.crate_name),
                "waived crate {:?} is not a declared descriptor-less utility crate",
                waiver.crate_name
            );
            assert_eq!(
                waiver.owner,
                format!("candle-gen-{}", waiver.crate_name),
                "a waiver's owner must be the crate that owns the route"
            );
            assert!(
                !waiver.provider_id.trim().is_empty() && !waiver.reason.trim().is_empty(),
                "waiver {:?} must name a route and a reason",
                waiver.crate_name
            );
            // The evidence paths are the whole point of a machine-readable waiver: a moved or
            // renamed contract silently turns the row into an unfalsifiable claim.
            for path in [waiver.contract_path, waiver.verification_path] {
                let expected_prefix = format!("crates/media/candle-gen/{}/src/", waiver.owner);
                assert!(
                    path.starts_with(&expected_prefix),
                    "waiver {:?} cites {path:?}, which is outside its owning crate",
                    waiver.crate_name
                );
                assert!(
                    repo_root.join(path).is_file(),
                    "waiver {:?} cites {path:?}, which does not exist",
                    waiver.crate_name
                );
            }
            // The waived route really is absent from both registries — that absence is what the
            // waiver exists to document.
            assert!(
                registry.generators().all(|registration| {
                    let id = (registration.descriptor)().id;
                    id != waiver.provider_id && id != waiver.crate_name
                }),
                "waived route {:?} has a generator registration after all",
                waiver.provider_id
            );
            assert!(
                registry
                    .memory_strategy_registrations()
                    .all(|registration| registration.provider_id != waiver.provider_id),
                "waived route {:?} has a memory-strategy registration after all",
                waiver.provider_id
            );
        }
    }

    /// The crates that own a Candle memory-strategy registration, and how each one reaches the
    /// catalog. Wiring only: like [`ProviderCrate`] this table names no provider id and no total, so
    /// it cannot drift from one — the ids are read back out of a registry built from each crate
    /// alone. Its membership is policed against the sources by
    /// [`every_memory_route_crate_reaches_the_catalog`].
    struct MemoryRouteCrate {
        /// Directory name under `crates/media/candle-gen`.
        dir: &'static str,
        register_providers: fn(ProviderRegistryBuilder) -> ProviderRegistryBuilder,
        /// `Some` for a crate that publishes weights-free contract surfaces on every platform.
        /// `None` marks a route reachable only through a CUDA-gated `register_providers` — the
        /// asymmetry the old `24`/`23` registration pin used to encode, unless it publishes a
        /// resident-only witness directly from `register_providers`.
        register_surfaces: Option<fn(ProviderRegistryBuilder) -> ProviderRegistryBuilder>,
        /// A resident-only route can be registered on every platform without publishing optimized
        /// contract surfaces; this keeps that intentional asymmetry explicit.
        resident_only_on_cpu: bool,
    }

    const MEMORY_ROUTE_CRATES: &[MemoryRouteCrate] = &[
        MemoryRouteCrate {
            dir: "candle-gen-anima",
            register_providers: candle_gen_anima::register_providers,
            register_surfaces: Some(candle_gen_anima::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-bernini",
            register_providers: candle_gen_bernini::register_providers,
            register_surfaces: Some(candle_gen_bernini::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-boogu",
            register_providers: candle_gen_boogu::register_providers,
            register_surfaces: Some(candle_gen_boogu::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-chroma",
            register_providers: candle_gen_chroma::register_providers,
            register_surfaces: Some(candle_gen_chroma::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-ideogram",
            register_providers: candle_gen_ideogram::register_providers,
            register_surfaces: Some(candle_gen_ideogram::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-flux",
            register_providers: candle_gen_flux::register_providers,
            register_surfaces: Some(candle_gen_flux::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-flux2",
            register_providers: candle_gen_flux2::register_providers,
            register_surfaces: Some(candle_gen_flux2::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-kolors",
            register_providers: candle_gen_kolors::register_providers,
            register_surfaces: Some(candle_gen_kolors::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-krea",
            register_providers: candle_gen_krea::register_providers,
            register_surfaces: Some(candle_gen_krea::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-lens",
            register_providers: candle_gen_lens::register_providers,
            register_surfaces: Some(candle_gen_lens::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-ltx",
            register_providers: candle_gen_ltx::register_providers,
            register_surfaces: Some(candle_gen_ltx::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-mage",
            register_providers: candle_gen_mage::register_providers,
            register_surfaces: Some(candle_gen_mage::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-minimax-h3",
            register_providers: candle_gen_minimax_h3::register_providers,
            register_surfaces: Some(candle_gen_minimax_h3::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-qwen-image",
            register_providers: candle_gen_qwen_image::register_providers,
            register_surfaces: Some(candle_gen_qwen_image::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-sana",
            register_providers: candle_gen_sana::register_providers,
            register_surfaces: Some(candle_gen_sana::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-scail2",
            register_providers: candle_gen_scail2::register_providers,
            register_surfaces: Some(candle_gen_scail2::register_memory_contract_surfaces),
            resident_only_on_cpu: true,
        },
        MemoryRouteCrate {
            dir: "candle-gen-sd3",
            register_providers: candle_gen_sd3::register_providers,
            register_surfaces: Some(candle_gen_sd3::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-sdxl",
            register_providers: candle_gen_sdxl::register_providers,
            register_surfaces: Some(candle_gen_sdxl::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-sensenova",
            register_providers: candle_gen_sensenova::register_providers,
            register_surfaces: Some(candle_gen_sensenova::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-svd",
            register_providers: candle_gen_svd::register_providers,
            register_surfaces: Some(candle_gen_svd::register_memory_contract_surfaces),
            resident_only_on_cpu: true,
        },
        MemoryRouteCrate {
            dir: "candle-gen-wan",
            register_providers: candle_gen_wan::register_providers,
            register_surfaces: None,
            resident_only_on_cpu: false,
        },
        MemoryRouteCrate {
            dir: "candle-gen-z-image",
            register_providers: candle_gen_z_image::register_providers,
            register_surfaces: Some(candle_gen_z_image::register_memory_contract_surfaces),
            resident_only_on_cpu: false,
        },
    ];

    /// Anchor the catalog's memory-strategy registration set to the provider crate sources.
    ///
    /// The surface/composed assertions in `every_registered_memory_strategy_rejects_cross_route_
    /// decode_geometry` all read the same registry, so a registration that disappears together with
    /// its fixture shrinks every one of them consistently and stays green. The old `24`/`23` pin was
    /// the only thing anchoring that set to something outside the registry, and this test replaces
    /// it with a source-derived anchor instead of the count.
    ///
    /// What it catches, each one mutation-checked:
    ///
    /// * A crate that owns a memory route in its sources but contributes nothing to the catalog —
    ///   including the catalog forgetting to wire its `register_memory_contract_surfaces`, and a
    ///   provider's own registration function quietly registering nothing.
    /// * The mirror case: a route that is supposed to be CUDA-only turning up on a CPU catalog.
    /// * A registration narrowed to a test-only gate. The scan reads `module_tree`'s *shipped* code,
    ///   which strips `cfg(test)` items, so `cfg(any(feature = "cuda", test))` decaying to
    ///   `cfg(test)` — the regression that would silently empty the CUDA catalog's Wan route — drops
    ///   the crate out of the scan and fails the table comparison on every build, not just on CUDA.
    /// * A crate gaining or losing memory registrations without updating [`MEMORY_ROUTE_CRATES`].
    ///
    /// The middle ground — a gate that still ships but stops naming `feature = "cuda"` — is not
    /// checked here because the compiler already prevents it: the registration constants carry the
    /// same gate, so widening only the call site fails to resolve them.
    ///
    /// What it does not catch: deleting a registration from a provider crate's own sources moves the
    /// scan with it. That is a visible deletion in the owning crate rather than a silent wiring
    /// regression, and the owning crate's tests cover it.
    #[test]
    fn every_memory_route_crate_reaches_the_catalog() {
        const REGISTRATION_CALLS: [&str; 3] = [
            "register_memory_strategy(",
            "register_composed_memory_strategy(",
            "register_resident_only_memory_contract(",
        ];

        // --- What the sources say ------------------------------------------------------------
        let mut owns_memory_route = BTreeSet::new();
        let mut publishes_surfaces = BTreeSet::new();
        let mut derives_architecture_facts = BTreeSet::new();
        let mut tests_architecture_facts = BTreeSet::new();
        let mut crate_dirs: Vec<String> = std::fs::read_dir(candle_gen_root())
            .expect("crates/media/candle-gen is readable")
            .map(|entry| entry.expect("readable directory entry").path())
            .filter(|path| path.join("src/lib.rs").is_file())
            .map(|path| {
                path.file_name()
                    .expect("a crate directory has a name")
                    .to_string_lossy()
                    .into_owned()
            })
            // The catalog composes the others; it registers no memory route of its own.
            .filter(|dir| dir != "candle-gen-catalog")
            .collect();
        crate_dirs.sort();
        assert!(
            crate_dirs.len() > 1,
            "the crate walk found nothing; every scan below would be vacuous"
        );
        for dir in &crate_dirs {
            let src = candle_gen_root().join(dir).join("src");
            let tree = module_tree(dir, &src);
            // Stripped code, so a call named only in a comment cannot enrol a crate.
            let code = tree.shipped.values().cloned().collect::<String>();
            if REGISTRATION_CALLS.iter().any(|call| code.contains(call)) {
                owns_memory_route.insert(dir.clone());
            }
            if code.contains("fn register_memory_contract_surfaces") {
                publishes_surfaces.insert(dir.clone());
            }
            // The derivation itself is shipped code, so the stripped tree is the right source: a
            // crate that only *mentions* `architecture_facts` in prose does not enrol.
            if code.contains("fn architecture_facts") {
                derives_architecture_facts.insert(dir.clone());
            }
            // Its test is not shipped code, so this reads the raw files — and requires the `#[test]`
            // attribute right in front of the name, so a helper called `architecture_facts_for`
            // cannot stand in for a test that runs.
            if rust_sources(&src)
                .iter()
                .any(|path| has_architecture_facts_test(path))
            {
                tests_architecture_facts.insert(dir.clone());
            }
        }
        assert!(
            !owns_memory_route.is_empty(),
            "no crate registers a memory route; the table comparison below would be vacuous"
        );

        // --- ...and every one of them must derive and test its architecture facts -------------
        //
        // AC (sc-22661 / epic SC-22657 E2). The registry-wide surface walk cannot see this: it
        // requires `MemoryArchitectureFacts::default()` on every Candle weights-free surface, which
        // is exactly what a crate with no derivation at all publishes. Scanning the sources instead
        // catches the crate that never wrote one — and, because `catalog_ids == expected_ids` below
        // pins `owns_memory_route` to precisely the crates behind the catalog's
        // `memory_strategy_registrations()`, this is that provider set stated in crate terms.
        assert_eq!(
            owns_memory_route
                .difference(&derives_architecture_facts)
                .collect::<BTreeSet<_>>(),
            BTreeSet::new(),
            "every crate contributing a memory registration must derive its architecture facts \
             (`fn architecture_facts`); these contribute one and derive nothing"
        );
        assert_eq!(
            owns_memory_route
                .difference(&tests_architecture_facts)
                .collect::<BTreeSet<_>>(),
            BTreeSet::new(),
            "every crate contributing a memory registration must own a `#[test] fn \
             architecture_facts_*` over that derivation; these derive facts nothing asserts"
        );

        // --- The wiring table must match them ------------------------------------------------
        let declared = MEMORY_ROUTE_CRATES
            .iter()
            .map(|owner| owner.dir.to_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            declared.len(),
            MEMORY_ROUTE_CRATES.len(),
            "the memory-route table must not repeat a crate"
        );
        assert_eq!(
            declared, owns_memory_route,
            "MEMORY_ROUTE_CRATES must cover exactly the crates whose sources register a memory route"
        );
        assert_eq!(
            MEMORY_ROUTE_CRATES
                .iter()
                .filter(|owner| owner.register_surfaces.is_some())
                .map(|owner| owner.dir.to_owned())
                .collect::<BTreeSet<_>>(),
            publishes_surfaces
                .intersection(&owns_memory_route)
                .cloned()
                .collect::<BTreeSet<_>>(),
            "a row's register_surfaces must be Some exactly when its crate publishes weights-free \
             contract surfaces"
        );

        // --- ...and the catalog must carry exactly what those crates contribute ---------------
        let catalog_ids = super::memory_contract_surface_registry()
            .expect("catalog")
            .memory_strategy_registrations()
            .map(|registration| registration.provider_id)
            .collect::<BTreeSet<_>>();
        let mut expected_ids = BTreeSet::new();
        for owner in MEMORY_ROUTE_CRATES {
            let mut builder = (owner.register_providers)(ProviderRegistryBuilder::new());
            // Mirror `memory_contract_surface_registry`: append weights-free surfaces off CUDA only
            // when `register_providers` does not already carry them. Most Candle image routes add
            // their memory surface only on CUDA; SenseNova publishes the same surface on every
            // platform, so registering it twice would be a duplicate.
            if !cfg!(feature = "cuda") {
                if let Some(register) = owner.register_surfaces {
                    let provider_ids = (owner.register_providers)(ProviderRegistryBuilder::new())
                        .build()
                        .unwrap_or_else(|error| panic!("{}: {error}", owner.dir))
                        .memory_strategy_registrations()
                        .map(|registration| registration.provider_id)
                        .collect::<BTreeSet<_>>();
                    if provider_ids.is_empty() {
                        builder = register(builder);
                    }
                }
            }
            let ids = builder
                .build()
                .unwrap_or_else(|error| panic!("{}: {error}", owner.dir))
                .memory_strategy_registrations()
                .map(|registration| registration.provider_id)
                .collect::<BTreeSet<_>>();
            let reachable = owner.register_surfaces.is_some()
                || owner.resident_only_on_cpu
                || cfg!(feature = "cuda");
            assert_eq!(
                !ids.is_empty(),
                reachable,
                "{} contributes {} memory registrations on this build; expected {}",
                owner.dir,
                if ids.is_empty() { "no" } else { "some" },
                if reachable { "some" } else { "none" }
            );
            expected_ids.extend(ids);
        }
        assert_eq!(
            catalog_ids, expected_ids,
            "the catalog's memory-strategy set must be exactly what the memory-route crates \
             contribute on this build"
        );
    }

    /// Every catalog provider whose registration changes under CUDA must receive that feature from
    /// the composition crate. Otherwise Cargo enables `candle-gen-catalog/cuda` while leaving the
    /// provider on its non-CUDA registration path: neither side of a `cfg(feature = "cuda")` /
    /// `cfg(not(feature = "cuda"))` split owns the registration.
    ///
    /// The provider set is derived from the shipped module trees and the catalog's actual
    /// `register_providers` body. In particular, this does not search comments or test modules,
    /// and it does not use a hand-maintained list that could miss the next CUDA-gated provider.
    #[test]
    fn cuda_feature_forwarding_covers_every_cuda_gated_catalog_registration() {
        fn function_body<'a>(source: &'a str, function: &str) -> &'a str {
            let start = source
                .find(function)
                .unwrap_or_else(|| panic!("missing {function:?}"));
            let rest = &source[start..];
            let open = rest
                .find('{')
                .unwrap_or_else(|| panic!("{function:?} has no body"));
            let body_start = open + 1;
            let mut depth = 1usize;
            for (offset, ch) in rest[body_start..].char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return &rest[body_start..body_start + offset];
                        }
                    }
                    _ => {}
                }
            }
            panic!("{function:?} has an unterminated body");
        }

        /// Whether a `cfg` predicate changes with the `cuda` feature. This deliberately treats
        /// `any`, `all`, and `not` alike: a provider registration whose branch depends on CUDA in
        /// any boolean position needs the catalog to forward that feature rather than selecting an
        /// accidental fallback branch.
        fn predicate_mentions_cuda(predicate: &str) -> bool {
            let predicate = predicate.trim();
            if let Some(rest) = predicate.strip_prefix("feature") {
                return rest
                    .trim_start()
                    .strip_prefix('=')
                    .is_some_and(|value| value.trim().trim_matches('"') == "cuda");
            }
            for combinator in ["all", "any", "not"] {
                if let Some(inner) = strip_call(predicate, combinator) {
                    return split_top_level(inner)
                        .iter()
                        .any(|part| predicate_mentions_cuda(part));
                }
            }
            false
        }

        /// Every `cfg` predicate attached inside the registration body that mentions CUDA.
        /// `module_tree` has already removed comments and test-only modules, so this reads only
        /// compiled source rather than prose which happens to quote a cfg expression.
        fn has_cuda_gated_registration(source: &str) -> bool {
            const OPEN: &str = "#[cfg(";
            let mut rest = source;
            while let Some(offset) = rest.find(OPEN) {
                let predicate = &rest[offset + OPEN.len()..];
                let mut depth = 1usize;
                for (end, ch) in predicate.char_indices() {
                    match ch {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                if predicate_mentions_cuda(&predicate[..end]) {
                                    return true;
                                }
                                rest = &predicate[end + 1..];
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                assert_eq!(depth, 0, "unterminated cfg predicate in registration body");
            }
            false
        }

        // Parser controls: these are all valid CUDA-dependent registration predicates, including
        // Wan's `any(feature = "cuda", test)` form and a negated fallback branch.
        for predicate in [
            "feature = \"cuda\"",
            "any(feature = \"cuda\", test)",
            "all(unix, feature = \"cuda\")",
            "not(feature = \"cuda\")",
        ] {
            assert!(
                predicate_mentions_cuda(predicate),
                "CUDA predicate parser missed {predicate:?}"
            );
        }
        assert!(
            !predicate_mentions_cuda("any(feature = \"metal\", test)"),
            "CUDA predicate parser must not match another feature"
        );

        let catalog_src = candle_gen_root().join("candle-gen-catalog/src");
        let catalog_tree = module_tree("candle-gen-catalog", &catalog_src);
        let catalog_registration = function_body(
            catalog_tree
                .shipped
                .get("lib.rs")
                .expect("catalog lib.rs is shipped"),
            "pub fn register_providers",
        );

        let mut cuda_gated_catalog_providers = BTreeSet::new();
        let mut crate_dirs: Vec<_> = std::fs::read_dir(candle_gen_root())
            .expect("crates/media/candle-gen is readable")
            .map(|entry| entry.expect("readable directory entry").path())
            .filter(|path| path.join("src/lib.rs").is_file())
            .collect();
        crate_dirs.sort();
        for crate_dir in crate_dirs {
            let dir = crate_dir
                .file_name()
                .expect("a crate directory has a name")
                .to_string_lossy()
                .into_owned();
            if dir == "candle-gen-catalog" {
                continue;
            }
            let provider_call = format!("{}::register_providers(", dir.replace('-', "_"));
            if !catalog_registration.contains(&provider_call) {
                continue;
            }
            let provider_tree = module_tree(&dir, &crate_dir.join("src"));
            let provider_registration = function_body(
                provider_tree
                    .shipped
                    .get("lib.rs")
                    .expect("provider lib.rs is shipped"),
                "pub fn register_providers",
            );
            if has_cuda_gated_registration(provider_registration) {
                cuda_gated_catalog_providers.insert(dir);
            }
        }
        assert!(
            !cuda_gated_catalog_providers.is_empty(),
            "the CUDA registration scan found no catalog providers; its forwarding assertion would be vacuous"
        );
        assert!(
            cuda_gated_catalog_providers.contains("candle-gen-wan"),
            "Wan's cfg(any(feature = \"cuda\", test)) registration branch must be covered"
        );

        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(4)
            .expect("candle-gen-catalog sits four levels below the workspace root");
        let output = std::process::Command::new(env!("CARGO"))
            .args([
                "metadata",
                "--no-deps",
                "--offline",
                "--format-version",
                "1",
                "--manifest-path",
            ])
            .arg(workspace_root.join("Cargo.toml"))
            .output()
            .unwrap_or_else(|error| panic!("cargo metadata: {error}"));
        assert!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("cargo metadata emits JSON");
        let features = metadata["packages"]
            .as_array()
            .expect("cargo metadata reports packages")
            .iter()
            .find(|package| package["name"] == "candle-gen-catalog")
            .expect("candle-gen-catalog is a workspace member")["features"]
            .as_object()
            .expect("catalog package reports features");
        let feature_members = |feature: &str| {
            features[feature]
                .as_array()
                .unwrap_or_else(|| panic!("catalog has no {feature:?} feature"))
                .iter()
                .map(|member| member.as_str().expect("feature member is a string"))
                .collect::<BTreeSet<_>>()
        };

        let cuda_forwarded = feature_members("cuda");
        for provider in &cuda_gated_catalog_providers {
            let forwarding = format!("{provider}/cuda");
            assert!(
                cuda_forwarded.contains(forwarding.as_str()),
                "{provider} has a CUDA-gated register_providers branch, so candle-gen-catalog's \
                 cuda feature must forward {forwarding:?}"
            );
        }

        // The MiniMax-H3 seam has both backend paths. CUDA is covered by the class check above;
        // Metal has no analogous gated registration branch, so pin its forwarding explicitly.
        assert!(
            feature_members("metal").contains("candle-gen-minimax-h3/metal"),
            "candle-gen-catalog's metal feature must forward candle-gen-minimax-h3/metal"
        );
    }

    /// The carried-over no-go set stays outside advertising, by exact id, and stays out of the
    /// allowlist. The reason is a settled finding (see [`NoGo`]), not an open question — candle must
    /// not re-run those fits.
    ///
    /// **The failure message names the finding rather than reporting a bare boolean** (sc-16961).
    /// "`wan_vace` must not advertise previews" tells the next author nothing about *why*, and the
    /// most likely next move after reading it is to go and measure — which is the one thing this set
    /// exists to prevent. Each message now carries the settling story and whether the basis is a
    /// measurement (`fit` vs `holdout` R², labelled, never interchanged) or a deliberate
    /// non-measurement.
    ///
    /// The second half mirrors the MLX guard's InstantID assertion: on candle, InstantID **and**
    /// PuLID are bespoke composition APIs rather than registered generators, so neither may acquire
    /// an invented registration on the way to acquiring a preview.
    #[test]
    fn temporal_and_super_resolution_routes_stay_outside_preview_advertising() {
        let registry = super::provider_registry().expect("catalog");
        let descriptors: Vec<_> = registry
            .generators()
            .map(|registration| (registration.descriptor)())
            .collect();

        for (id, basis) in PREVIEW_INERT_ROUTE_IDS {
            let descriptor = descriptors
                .iter()
                .find(|descriptor| descriptor.id == *id)
                .unwrap_or_else(|| panic!("{id} must remain a registered candle generator"));
            assert!(
                !descriptor.capabilities.supports_preview,
                "{id} must not advertise previews — {}. Carried over from epic 16624 rather than \
                 re-measured, because an RGB fit is a property of the VAE latent space and not of the \
                 backend. Do NOT re-run the fit: if a new method makes this viable it reopens as a \
                 NEW story with a NEW measurement. Record: \
                 docs/migration/evidence/sc-16961-preview-no-go-carry-over.md",
                basis.reason()
            );
            assert!(
                !PREVIEW_ROUTE_IDS.contains(id),
                "{id} cannot be both wired and preview-inert"
            );
        }

        for id in ["instantid", "pulid", "pulid_flux"] {
            assert!(
                descriptors.iter().all(|descriptor| descriptor.id != id),
                "{id} is a bespoke candle composition API and must not gain an invented registration"
            );
        }
    }

    /// **Every registered route is in exactly one of three classes** — wired, no-go, or deferred —
    /// and the three cover the shipped surface with nothing left over (sc-16961).
    ///
    /// This is what stops the no-go record going stale. sc-16951 first pinned the inert ids and the
    /// rest of the epic landed on top of it; without a total partition, a route registered in that
    /// window would simply be absent from every list, and "the no-go set is complete" would be a claim
    /// nobody could check. With it, a new registration fails the build until an author *decides* which
    /// class it belongs to — which is the decision this whole epic is about, made once, in writing.
    ///
    /// The third class is load-bearing and is not a synonym for "not wired yet". Boogu does not
    /// advertise previews and is emphatically **not** a rejection: sc-16956 proved its VAE is FLUX.1's,
    /// so a fit that clears the bar already exists for its space (sc-17218 is the wiring). Collapsing
    /// the two would either lose a viable family into the rejected pile or force the no-go set to mean
    /// "everything that does not advertise", which is not a record of anything.
    ///
    /// The counts are asserted, not merely the partition: moving a route between classes is a
    /// decision, and a decision should have to be written down here rather than absorbed silently by a
    /// set that happens to still add up.
    #[test]
    fn the_no_go_set_and_the_wired_set_partition_every_shipped_route() {
        let registered: BTreeSet<String> = super::provider_registry()
            .expect("catalog")
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        let wired: BTreeSet<String> = PREVIEW_ROUTE_IDS.iter().map(|id| id.to_string()).collect();
        let no_go: BTreeSet<String> = PREVIEW_INERT_ROUTE_IDS
            .iter()
            .map(|(id, _)| id.to_string())
            .collect();
        let deferred: BTreeSet<String> = PREVIEW_DEFERRED_ROUTE_IDS
            .iter()
            .map(|(id, _)| id.to_string())
            .collect();

        // A deferred route is only distinguishable from a no-go one by the story that will wire it.
        // Without a named story the third class degenerates into "unclassified, but quietly".
        for (id, story) in PREVIEW_DEFERRED_ROUTE_IDS {
            assert!(
                story.starts_with("sc-") && story.len() > 3,
                "{id} is deferred rather than rejected, so it must name the story that wires it — \
                 got {story:?}"
            );
        }

        for (label, class, declared) in [
            ("wired", &wired, PREVIEW_ROUTE_IDS.len()),
            ("no-go", &no_go, PREVIEW_INERT_ROUTE_IDS.len()),
            ("deferred", &deferred, PREVIEW_DEFERRED_ROUTE_IDS.len()),
        ] {
            assert_eq!(
                class.len(),
                declared,
                "the {label} class must not repeat an id"
            );
        }
        assert!(
            wired.is_disjoint(&no_go)
                && wired.is_disjoint(&deferred)
                && no_go.is_disjoint(&deferred),
            "a route belongs to exactly one class; overlaps: wired/no-go {:?}, wired/deferred {:?}, \
             no-go/deferred {:?}",
            wired.intersection(&no_go).collect::<Vec<_>>(),
            wired.intersection(&deferred).collect::<Vec<_>>(),
            no_go.intersection(&deferred).collect::<Vec<_>>()
        );

        let mut classified: BTreeSet<String> = wired.clone();
        classified.extend(no_go.iter().cloned());
        classified.extend(deferred.iter().cloned());
        assert_eq!(
            classified,
            registered,
            "every registered generator must be classified exactly once. Unclassified (registered but \
             in no list) — decide whether it is wired, a no-go carried over from epic 16624, or \
             deferred-but-viable, and say so: {:?}. Phantom (listed but not registered): {:?}",
            registered.difference(&classified).collect::<Vec<_>>(),
            classified.difference(&registered).collect::<Vec<_>>()
        );

        // Deliberately NOT a pinned total. Two consecutive main syncs (sc-18306+VACE-Fun, then
        // MiniMax-H3+Krea Turbo edit) each tripped a frozen (registered, wired, no_go, deferred)
        // tuple whose only failure mode was "two branches merged" — never a defect. The decision
        // record this test exists for is the *classification*: the partition, disjointness, and
        // classified == registered assertions above already force every new registration to be
        // filed into exactly one class, in writing, in the three pinned id lists. What remains
        // load-bearing here is that the catalog, wired set, and permanent no-go set do not silently
        // empty out. The deferred class is intentionally allowed to reach zero: sc-17218 wired its
        // final member. The class and its provenance validation remain above so the next viable but
        // unwired route must still be recorded explicitly.
        for (label, len) in [
            ("registered", registered.len()),
            ("wired", wired.len()),
            ("no-go", no_go.len()),
        ] {
            assert!(
                len > 0,
                "the {label} preview class must not be empty — an empty class means a whole \
                 decision category was silently dropped, not that the catalog shrank honestly"
            );
        }
    }

    /// **No no-go family may acquire a preview fit, a fit producer, or an emission call** (sc-16961).
    ///
    /// The advertising guard above catches a descriptor that starts claiming previews. It does *not*
    /// catch the expensive half: an author who never reaches a descriptor because they are still on
    /// the CUDA box deriving `RGB_FACTORS` for a space epic 16624 already rejected. That work is the
    /// waste this story exists to prevent, and it is visible in the sources long before it is visible
    /// in a capability.
    ///
    /// The eight crate directories are **derived from the inert ids** — a crate is a no-go crate iff
    /// every id it registers is inert — and then pinned. The derivation is what makes a new
    /// registration inside one of them join this scan automatically; the pin is what makes an id
    /// silently changing crates, or a crate acquiring a route nobody classified, a failure rather than
    /// a quietly shorter scan.
    ///
    /// Deliberately a raw-text scan over the crate's whole tree — `src/`, `tests/`, everything —
    /// rather than the comment-stripped shipped-module scan the wiring assertions use. A fit parked
    /// under `#[cfg(test)]`, or in a `tests/` producer that never ships, is exactly the thing being
    /// forbidden. The cost is that *mentioning* a marker in one of these crates also fails; that is
    /// intended, and the message says where the discussion belongs instead.
    ///
    /// **Positive controls make it non-vacuous.** A marker list that silently stopped matching — a
    /// renamed constant, a moved module — would read as "no no-go crate has a fit" forever and pass.
    /// So the same scan is run against `candle-gen-flux`, which carries a committed fit, and the same
    /// producer-filename scan against `candle-gen-sensenova`, which carries `tests/fit_preview_rgb.rs`,
    /// and both must trip.
    ///
    /// **What it detects, and how firmly.** [`FIT_MARKERS`] is a list of exact substrings, so on its
    /// own it is a *name* heuristic and a determined author could rename around it. Two things narrow
    /// that: [`latent_fit_shapes`] matches the fit's shape `[[f32; 3]; N]` rather than its name, so a
    /// committed table is caught whatever it is called; and anything that actually *emits* has to
    /// reach `PreviewHook` / `PreviewSink` / `project_latents`, which are matched by name and are not
    /// optional. What is left uncovered is a fit stored in some other representation entirely — a
    /// `Vec<Vec<f32>>`, or coefficients read from a data file — which is well past "someone has
    /// started the work this record says not to start" and is not what this scan is for.
    #[test]
    fn no_no_go_family_acquires_a_preview_fit_or_a_fit_producer() {
        let no_go_ids: BTreeSet<&str> = PREVIEW_INERT_ROUTE_IDS.iter().map(|(id, _)| *id).collect();
        let mut crates: Vec<&str> = PROVIDER_CRATES
            .iter()
            .filter(|provider| {
                let ids = ids_of(provider);
                !ids.is_empty() && ids.iter().all(|id| no_go_ids.contains(id.as_str()))
            })
            .map(|provider| provider.dir)
            .collect();
        crates.sort_unstable();
        assert_eq!(
            crates,
            [
                "candle-gen-bernini",
                "candle-gen-ltx",
                "candle-gen-mage",
                "candle-gen-minimax-h3",
                "candle-gen-mochi",
                "candle-gen-scail2",
                "candle-gen-seedvr2",
                "candle-gen-svd",
                "candle-gen-wan",
            ],
            "the 22 no-go ids must resolve to exactly these nine crates — a mismatch means an id \
             moved crates or a crate acquired a route that is not accounted for"
        );

        for dir in &crates {
            let (markers, producers) = fit_evidence(dir);
            assert!(
                markers.is_empty(),
                "{dir} is in the epic-16624 no-go set and must not acquire a preview fit. Found \
                 {markers:?}. The rejection is a property of the latent space, not of the backend, so \
                 there is nothing here to re-derive — see \
                 docs/migration/evidence/sc-16961-preview-no-go-carry-over.md, and if a new method \
                 makes this family viable, open a NEW story with a NEW measurement rather than \
                 landing a fit here. (Discussing a marker also trips this scan; that discussion \
                 belongs in the evidence doc.)"
            );
            assert!(
                producers.is_empty(),
                "{dir} must not gain a preview fit producer or harness: {producers:?}. Deriving \
                 coefficients for a rejected latent space is the CUDA-box time sc-16961 exists to save."
            );

            // And it still emits nothing — the source-level statement of the same fact.
            let wiring = scan(dir);
            assert!(
                !wiring.emits(),
                "{dir} must not emit previews: {:?}",
                wiring
                    .sites
                    .iter()
                    .filter(|site| site.hooked)
                    .map(|site| (&site.file, site.driver, site.index))
                    .collect::<Vec<_>>()
            );
        }

        // Positive controls: the detectors must be capable of firing on a crate that really does
        // carry a fit, and on one that really does carry a producer.
        let (flux_markers, _) = fit_evidence("candle-gen-flux");
        assert!(
            flux_markers
                .iter()
                .any(|found| found.ends_with(": RGB_FACTORS")),
            "the marker scan no longer detects candle-gen-flux's committed fit, so its silence on \
             the no-go crates proves nothing: {flux_markers:?}"
        );
        assert!(
            flux_markers
                .iter()
                .any(|found| found.ends_with(": [[f32; 3]; 16]")),
            "the SHAPE scan no longer detects candle-gen-flux's 16x3 fit table. The name list is \
             evadable by renaming the constant; the shape check is what makes that not enough, so it \
             must be demonstrably alive: {flux_markers:?}"
        );
        let (_, sensenova_producers) = fit_evidence("candle-gen-sensenova");
        assert!(
            sensenova_producers
                .iter()
                .any(|path| path.ends_with("fit_preview_rgb.rs")),
            "the producer scan no longer detects candle-gen-sensenova's fit_preview_rgb.rs: \
             {sensenova_producers:?}"
        );

        // The loosened producer patterns, checked directly rather than via a crate that happens to
        // contain one. The first three are real evasions that slipped past the four exact filenames;
        // the `ordinary` list below must keep NOT matching, or the patterns would sweep up shipped
        // modules that merely mention preview samples.
        for evasion in [
            "tests/fit_ltx_rgb.rs",
            "tests/preview_fit.rs",
            "tests/nested/preview_real_weights_v2.rs",
            "src/preview.rs",
        ] {
            assert!(
                is_fit_producer(evasion),
                "{evasion} must be recognised as a preview-fit producer"
            );
        }
        for ordinary in ["src/model.rs", "src/preview_samples.rs", "tests/decode.rs"] {
            assert!(
                !is_fit_producer(ordinary),
                "{ordinary} is not a fit producer; the loosened patterns must stay confined to tests/"
            );
        }

        // And the shape check discriminates a fit table from an ordinary 3x3 kernel — the exclusion
        // that keeps candle-gen-seedvr2's shipped wavelet blur from reading as a committed fit.
        assert_eq!(
            latent_fit_shapes("const RGB_FACTORS: [[f32; 3]; 128] = ["),
            ["[[f32; 3]; 128]"],
            "a C x 3 fit table must be detected by shape alone"
        );
        assert_eq!(
            latent_fit_shapes("const RGB_FACTORS: [[f32;3];CHANNELS] = ["),
            ["[[f32; 3]; CHANNELS]"],
            "a non-literal row count counts as a hit rather than being parsed away, and the match \
             must not depend on rustfmt's spacing"
        );
        assert!(
            latent_fit_shapes("const KERNEL: [[f32; 3]; 3] = [").is_empty(),
            "a 3x3 kernel is not a fit table; no no-go latent space has three channels"
        );
    }

    /// The recorded epic-16624 numbers stay **labelled** and stay in the right column (sc-16961).
    ///
    /// sc-16954 was bounced for comparing an in-sample statistic against an out-of-sample one, and
    /// this record is where that confusion would do the most damage: a future author reading LTX's
    /// `0.984291` as the holdout number would conclude the space is fine and go and re-derive it. So
    /// each measured row is checked to carry both values, labelled, in fit-then-holdout order, with
    /// the holdout below the 0.88 bar and never above the fit.
    ///
    /// The **relationship between the rows** is pinned too, because it is the argument. At least one
    /// row must have a fit comfortably *above* the bar while its holdout is below it — that is the
    /// counter-intuitive shape (LTX: `0.984291` → `0.618575`) that makes an in-sample number
    /// dangerous to quote. And at least one row must have a fit that is **itself** below the bar
    /// (Mochi, `0.846932`), which says the linear model does not describe that space even in-sample,
    /// so no larger corpus rescues it. Losing either shape would leave the record looking like three
    /// near-misses.
    ///
    /// The unmeasured rows are checked the other way: they must carry **no** number at all, so a later
    /// edit cannot quietly attach a borrowed holdout figure to a space nobody measured.
    ///
    /// **The variant→measurement mapping itself is pinned first**, against [`NoGo::MEASURED`]. Without
    /// that, moving a pair onto an unmeasured variant simply routes it down the *measured* branch of
    /// every check below and passes: the shape assertions only ever iterate the three rows that are
    /// supposed to have numbers, so a fourth acquiring one is invisible to them. Since a variant can
    /// span several routes, that single edit would attach a borrowed holdout number to a whole family
    /// at once — the exact fabricated provenance this record exists to prevent.
    #[test]
    fn the_recorded_no_go_measurements_stay_labelled_fit_versus_holdout() {
        const BAR: f64 = 0.88;
        let parse = |value: &str| value.parse::<f64>().expect("a decimal R²");

        // Which spaces carry numbers at all, pinned exactly. `NoGo::ALL` is exhaustive, so a new
        // variant lands here rather than escaping every assertion in this test.
        let carries_numbers: Vec<NoGo> = NoGo::ALL
            .into_iter()
            .filter(|basis| basis.measured().is_some())
            .collect();
        assert_eq!(
            carries_numbers,
            NoGo::MEASURED.to_vec(),
            "epic 16624 measured exactly {:?} and nothing else. A variant that gained a (fit, \
             holdout) pair here did NOT gain a measurement — attaching one row's numbers to another \
             space is a fabricated provenance, and because a variant spans several routes it would \
             attach them to a whole family at once. If a space really was measured, that is a NEW \
             story with a NEW measurement, not an edit to this match arm.",
            NoGo::MEASURED
        );
        for basis in NoGo::ALL {
            assert_eq!(
                basis.measured().is_some(),
                NoGo::MEASURED.contains(&basis),
                "{basis:?}: NoGo::measured and NoGo::MEASURED disagree about whether this space was \
                 measured; they are the same fact written twice and must not drift"
            );
        }

        for basis in NoGo::ALL {
            let reason = basis.reason();
            match basis.measured() {
                Some((fit, holdout)) => {
                    assert!(
                        parse(fit) >= parse(holdout),
                        "{basis:?}: the fit (in-sample) R² {fit} cannot be below the holdout \
                         (out-of-sample) R² {holdout} — the two are the wrong way round"
                    );
                    assert!(
                        parse(holdout) < BAR,
                        "{basis:?}: the holdout (out-of-sample) R² {holdout} must be below the {BAR} \
                         bar, or this is not a rejection"
                    );
                    assert!(
                        reason.contains("fit R² (in-sample)")
                            && reason.contains("holdout R² (out-of-sample)"),
                        "{basis:?}: both statistics must be labelled where they are quoted, never \
                         left as bare numbers: {reason}"
                    );
                    let (fit_at, holdout_at) = (
                        reason.find(fit).expect("the fit value appears"),
                        reason.find(holdout).expect("the holdout value appears"),
                    );
                    assert!(
                        fit_at < holdout_at,
                        "{basis:?}: fit before holdout, so the two are never read swapped: {reason}"
                    );
                }
                None => {
                    assert!(
                        reason.contains("NEVER measured") || reason.contains("NO holdout number"),
                        "{basis:?}: an unmeasured row must say so plainly: {reason}"
                    );
                    for measured in NoGo::MEASURED {
                        let (fit, holdout) = measured.measured().expect("a measured row");
                        assert!(
                            !reason.contains(fit) && !reason.contains(holdout),
                            "{basis:?} must not borrow {measured:?}'s numbers — its latent space was \
                             never measured: {reason}"
                        );
                    }
                }
            }
        }

        // The two shapes that carry the argument, kept from collapsing into "three near-misses".
        let measured: Vec<(&str, &str)> =
            NoGo::ALL.into_iter().filter_map(NoGo::measured).collect();
        assert_eq!(
            measured.len(),
            3,
            "epic 16624 measured exactly three spaces"
        );
        assert!(
            measured
                .iter()
                .any(|(fit, holdout)| parse(fit) > BAR && parse(holdout) < BAR),
            "at least one row must show a fit ABOVE the bar collapsing to a holdout BELOW it — that \
             is why an in-sample number must never be quoted as if it settled anything: {measured:?}"
        );
        assert!(
            measured.iter().any(|(fit, _)| parse(fit) < BAR),
            "at least one row must show a fit that is itself below the bar — the linear model does \
             not describe that space even in-sample, so no larger corpus rescues it: {measured:?}"
        );
    }

    /// **`candle-gen-wan` registers routes in two different latent spaces, and the record says which
    /// is which** — pinned against the provider sources, not against the previous draft (sc-16961).
    ///
    /// This is the one lineage claim in the whole record that is easy to get backwards, and getting it
    /// backwards is not cosmetic: a future author who hits the no-go assertion on `wan2_2_ti2v_5b`
    /// would be told its space is `WanVae16` and that Bernini and Scail2 share it — all false. It would
    /// also erase the distinction sc-16637 closed on: "the registered family spans z16 `WanVae` IDs …
    /// and the distinct z48 `Wan22Vae` `wan2_2_ti2v_5b`; a single fit would never have covered the full
    /// surface."
    ///
    /// So the id→variant assignment is asserted exactly, **and** each side is grounded in the file that
    /// actually builds the VAE. The disposition is no-go either way; only the recorded lineage differs,
    /// which is precisely why nothing else in the suite would notice it being wrong.
    #[test]
    fn the_wan_routes_are_recorded_in_the_latent_space_their_provider_builds() {
        let by_variant = |wanted: NoGo| -> BTreeSet<&str> {
            PREVIEW_INERT_ROUTE_IDS
                .iter()
                .filter(|(_, basis)| *basis == wanted)
                .map(|(id, _)| *id)
                .collect()
        };

        assert_eq!(
            by_variant(NoGo::WanZ48),
            BTreeSet::from(["wan2_2_ti2v_5b"]),
            "the Wan z48 space holds exactly the 5B route: its provider builds \
             `VaeConfig::ti2v_5b()` -> `vae::WanVae`"
        );
        assert_eq!(
            by_variant(NoGo::WanZ16),
            BTreeSet::from([
                "wan2_2_t2v_14b",
                "wan2_2_i2v_14b",
                "wan_vace",
                "wan2_2_vace_fun_14b",
                "bernini_renderer",
                "bernini",
                "scail2_14b",
            ]),
            "the Wan z16 space holds exactly the routes that build `Vae16Config::wan21()` -> \
             `vae16::WanVae16` — the A14B pair, VACE, both Bernini ids and Scail2. The 5B is NOT one \
             of them"
        );

        // Grounded in the typed provider assignments consumed by the actual decode paths. Unlike
        // the former source-token scan, changing a provider-local VAE alias changes this value.
        let z48 = candle_gen_wan::WAN_Z48_VAE_TILING;
        let z16 = candle_gen_wan::WAN_Z16_VAE_TILING;
        assert_ne!(z48, z16);
        assert_eq!(z48.full_res_channels, 64);
        assert_eq!(z16.full_res_channels, 96);
        for id in by_variant(NoGo::WanZ48) {
            assert_eq!(super::vae_tiling(id), Some(z48), "{id}");
        }
        for id in by_variant(NoGo::WanZ16) {
            assert_eq!(super::vae_tiling(id), Some(z16), "{id}");
        }

        // And the two reasons stay distinguishable to a reader who only ever sees a failure message.
        let (z16_reason, z48_reason) = (NoGo::WanZ16.reason(), NoGo::WanZ48.reason());
        assert!(
            z16_reason.contains("z16")
                && z16_reason.contains("vae16::WanVae16")
                && z16_reason.contains("Vae16Config::wan21")
                && !z16_reason.contains("VaeConfig::ti2v_5b"),
            "the z16 reason must name its own VAE and config and never the 5B's: {z16_reason}"
        );
        assert!(
            z48_reason.contains("z48")
                && z48_reason.contains("vae::WanVae")
                && z48_reason.contains("VaeConfig::ti2v_5b")
                && z48_reason.contains("distinct from the z16"),
            "the z48 reason must name its own VAE and config, and say plainly that it is not the z16 \
             space, since that confusion is the whole reason this variant exists: {z48_reason}"
        );
    }

    /// Every `.rs` file under one candle-gen crate, whether or not it ships.
    ///
    /// Deliberately not `module_tree`: this scan is looking for work that has *started*, and work
    /// starts in `tests/` and under `#[cfg(test)]` before it reaches a shipped module.
    fn all_rust_files(dir: &str) -> Vec<PathBuf> {
        let root = candle_gen_root().join(dir);
        assert!(
            root.is_dir(),
            "{}: no such crate directory — an empty scan reads as `no fit here` and would make the \
             no-go assertions vacuous",
            root.display()
        );
        rust_sources(&root)
    }

    /// Source tokens that mean a crate has acquired a preview fit or the machinery to emit one.
    ///
    /// **Why the bare word `preview` is not usable:** every descriptor that declines previews writes
    /// `supports_preview: false`, families that train ship *preview samples* (sc-8650), and
    /// `candle-gen-mochi` is literally `genmo/mochi-1-preview`. Seven of the eight no-go crates
    /// contain the token today — all but `candle-gen-mage`, which writes no `supports_preview` line at
    /// all because its `Capabilities` literal ends `..Default::default()`. A substring match on it
    /// could never have been written, so the list below names constructs instead.
    ///
    /// These are exact substrings, which makes them a *name* heuristic and therefore evadable by
    /// renaming. [`latent_fit_shapes`] closes the main hole by matching the fit's **shape** rather than
    /// its name; what remains genuinely load-bearing is that anything which actually *emits* must reach
    /// `PreviewHook` / `PreviewSink` / `project_latents`, all matched here.
    const FIT_MARKERS: &[&str] = &[
        "RGB_FACTORS",
        "RGB_BIAS",
        "LATENT_RGB",
        "LATENT_TO_RGB",
        "project_latents",
        "PreviewHook",
        "PreviewCounter",
        "PreviewFrame",
        "PreviewSink",
        "emit_preview",
    ];

    /// File names that only ever exist to derive or validate a preview fit, matched anywhere in the
    /// crate. Loosened patterns under `tests/` are handled by [`is_fit_producer`].
    const FIT_PRODUCER_FILES: &[&str] = &[
        "fit_preview_rgb.rs",
        "preview_real_weights.rs",
        "preview_wiring.rs",
        "preview.rs",
    ];

    /// Whether a crate-relative path is a preview-fit producer or harness.
    ///
    /// Four exact names anywhere in the crate, plus — **under `tests/` only** — any `fit_*.rs` or any
    /// name containing `preview`. The exact list alone was evadable by naming the producer anything
    /// else (`tests/fit_ltx_rgb.rs`, `tests/preview_fit.rs` both slipped through); the loosened
    /// patterns are confined to `tests/` so that a shipped module which legitimately mentions preview
    /// samples in its filename is not swept up.
    fn is_fit_producer(relative: &str) -> bool {
        let name = relative.rsplit('/').next().unwrap_or(relative);
        if FIT_PRODUCER_FILES.contains(&name) {
            return true;
        }
        let Some(stem) = name.strip_suffix(".rs") else {
            return false;
        };
        let under_tests = relative.starts_with("tests/") || relative.contains("/tests/");
        under_tests && (stem.starts_with("fit_") || stem.contains("preview"))
    }

    /// Occurrences of the committed-fit **shape** `[[f32; 3]; N]` in one source, excluding `N == 3`.
    ///
    /// A preview fit is a `C x 3` table of least-squares constants — `RGB_FACTORS: [[f32; 3]; C]` for a
    /// `C`-channel latent space. The shape is what a fit *is*, so this catches a table committed under
    /// any name at all, which the substring list above cannot. Whitespace is squeezed out first so the
    /// match does not depend on rustfmt's spacing.
    ///
    /// `N == 3` is excluded because `[[f32; 3]; 3]` is an ordinary 3x3 kernel and one already ships in
    /// a no-go crate: `candle-gen-seedvr2/src/color.rs`'s wavelet blur `KERNEL`. The exclusion cannot
    /// hide a fit for any no-go space — none of them has three latent channels (LTX 128, Mage 128,
    /// Mochi 12, Wan z16 16, Wan z48 48, SVD 4) and SeedVR2 has no multi-step progression to fit at
    /// all. An `N` that is not a plain literal counts as a hit rather than being parsed away.
    fn latent_fit_shapes(source: &str) -> Vec<String> {
        const SHAPE: &str = "[[f32;3];";
        let squished: String = source.chars().filter(|c| !c.is_whitespace()).collect();
        let mut hits = Vec::new();
        let mut rest = squished.as_str();
        while let Some(at) = rest.find(SHAPE) {
            let tail = &rest[at + SHAPE.len()..];
            match tail.find(']') {
                Some(end) => {
                    let rows = &tail[..end];
                    if rows != "3" {
                        hits.push(format!("[[f32; 3]; {rows}]"));
                    }
                    rest = &tail[end + 1..];
                }
                None => {
                    hits.push("[[f32; 3]; <unterminated>".to_string());
                    break;
                }
            }
        }
        hits
    }

    /// `(markers found, producer files found)` for one crate — the raw evidence both no-go assertions
    /// and their positive controls read.
    fn fit_evidence(dir: &str) -> (Vec<String>, Vec<String>) {
        let root = candle_gen_root().join(dir);
        let mut markers = BTreeSet::new();
        let mut producers = BTreeSet::new();
        for path in all_rust_files(dir) {
            let relative = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            if is_fit_producer(&relative) {
                producers.insert(relative.clone());
            }
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            for marker in FIT_MARKERS {
                if source.contains(marker) {
                    markers.insert(format!("{relative}: {marker}"));
                }
            }
            for shape in latent_fit_shapes(&source) {
                markers.insert(format!("{relative}: {shape}"));
            }
        }
        (
            markers.into_iter().collect(),
            producers.into_iter().collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use candle_gen::gen_core::ConditioningKind;

    #[test]
    fn checkpoint_codecs_register_once_and_every_row_has_an_engine_implementation() {
        use candle_gen::logical_weights::{BASELINE_CODECS, CODEC_IMPLEMENTATION_IDS};

        use candle_gen::gen_core::GGUF_CONTAINER_CODEC;

        let registry = super::provider_registry().unwrap();
        let registered: Vec<_> = registry.checkpoint_codecs().codecs().copied().collect();
        // The engine's safetensors table, then the one GGUF container row the catalog adds because
        // its implementation lives in `candle-gen-wan` rather than in the engine crate. Still
        // exactly one composition point: no family `register_providers` may add a row.
        let expected: Vec<_> = BASELINE_CODECS
            .iter()
            .copied()
            .chain(std::iter::once(GGUF_CONTAINER_CODEC))
            .collect();
        assert_eq!(
            registered, expected,
            "the composed Candle catalog must carry the engine codec table exactly once plus the \
             GGUF container row; no family crate may add or repeat a row"
        );
        let mut registered_ids: Vec<&str> = registered.iter().map(|codec| codec.codec_id).collect();
        registered_ids.sort_unstable();
        let mut implemented: Vec<&str> = CODEC_IMPLEMENTATION_IDS
            .iter()
            .copied()
            .chain(std::iter::once(
                candle_gen_wan::GGUF_CODEC_IMPLEMENTATION_ID,
            ))
            .collect();
        implemented.sort_unstable();
        assert_eq!(
            registered_ids, implemented,
            "every registered codec row needs a decode implementation and vice versa"
        );
        // The GGUF row is reachable by its own container encoding, and — critically — is NOT
        // reachable from any safetensors dtype: `WeightEncoding::from_dtype` can never produce
        // `GgufContainer`, so a safetensors U8/fp8 tensor cannot be routed into the GGUF decoder.
        assert!(registry
            .checkpoint_codecs()
            .for_encoding(candle_gen::gen_core::WeightEncoding::GgufContainer)
            .is_some_and(|codec| codec.codec_id == GGUF_CONTAINER_CODEC.codec_id));
        // The plan-side registry `candle-gen-wan` compiles against carries the same row.
        assert_eq!(
            candle_gen_wan::gguf_codec_registry()
                .for_encoding(candle_gen::gen_core::WeightEncoding::GgufContainer)
                .map(|codec| codec.codec_id),
            Some(GGUF_CONTAINER_CODEC.codec_id)
        );
        assert!(registry
            .checkpoint_codecs()
            .for_encoding(candle_gen::gen_core::WeightEncoding::DenseBf16)
            .is_some());
        assert!(registry
            .checkpoint_codecs()
            .for_encoding(candle_gen::gen_core::WeightEncoding::Fp8E4M3)
            .is_some());
    }

    /// Every `mapping_id` the composed Candle catalog registers resolves to a real
    /// `LogicalKeyMapping` on this backend, **or** is explicitly declared not plan-driven here.
    ///
    /// Both directions matter, and both were broken before sc-20651: five declared mapping ids had
    /// no implementation anywhere (they read like backed routes and were not), and nothing proved
    /// that a mapping the Candle lane really does plan through is declared as such. The mirror of
    /// `mlx-gen-catalog`'s `canonical_mappings_are_backed_by_declared_implementations`.
    #[test]
    fn canonical_mappings_are_backed_by_declared_implementations() {
        use candle_gen::gen_core::{CheckpointBackend, LogicalKeyMapping};

        let registry = super::provider_registry().unwrap();
        // The complete set of `LogicalKeyMapping` implementations reachable from the Candle
        // platform. Adding one without declaring it (or declaring one without adding it) fails
        // below.
        //
        // The Krea mapping carries a file-detected namespace prefix and an optional architecture
        // config; neither affects `mapping_id`, so the id-surface check below uses the
        // no-config, bare-prefix form.
        let krea =
            candle_gen_krea::native_mapping::KreaNativeToDiffusersMapping::without_config("");
        // The FLUX.2 mapping carries the variant architecture config (it declares logical shapes
        // and the fused-transform geometry from it); the id surface is config-independent.
        let flux2_cfg = candle_gen_flux2::config::Flux2Variant::Klein9b.config();
        let flux2 = candle_gen_flux2::Flux2BflToDiffusersMapping::new(&flux2_cfg);
        let implementations: &[&dyn LogicalKeyMapping] =
            &[&candle_gen_wan::WanNativeToDiffusersMapping, &krea, &flux2];

        let mut declared_here = 0usize;
        for adapter in registry.checkpoint_adapters() {
            for mapping in adapter.canonical_mappings {
                let implemented = implementations
                    .iter()
                    .any(|implementation| implementation.mapping_id() == mapping.mapping_id);
                let declared = mapping
                    .plan_driven_backends
                    .contains(&CheckpointBackend::Candle);
                assert_eq!(
                    implemented,
                    declared,
                    "adapter {} dialect {:?}: mapping id {:?} is {} on Candle but {} — a declared \
                     mapping must resolve to a real implementation, and an implementation must be \
                     declared (loader-native dialects declare no backend at all)",
                    adapter.adapter_id,
                    mapping.dialect,
                    mapping.mapping_id,
                    if implemented {
                        "implemented"
                    } else {
                        "unimplemented"
                    },
                    if declared {
                        "declared plan-driven"
                    } else {
                        "declared loader-native"
                    },
                );
                declared_here += usize::from(declared);
            }
        }
        assert_eq!(
            declared_here,
            implementations.len(),
            "every Candle mapping implementation must be claimed by exactly one registered dialect"
        );

        // And the Wan mapping really is the refusing remap the GGUF route plans through: a native
        // key resolves to its diffusers name, and a foreign one refuses rather than passing
        // through unchanged.
        let wan = candle_gen_wan::WanNativeToDiffusersMapping;
        assert_eq!(
            wan.logical_key("blocks.0.self_attn.q.weight").as_deref(),
            Some("blocks.0.attn1.to_q.weight")
        );
        assert_eq!(wan.logical_key("vace_blocks.0.before_proj.weight"), None);

        // ...and the Krea mapping really is the native-mmdit remap the Kitchen NVFP4 import plans
        // through: a native key resolves to its diffusers name, a foreign one refuses.
        assert_eq!(
            krea.logical_key("blocks.0.attn.wq.weight").as_deref(),
            Some("transformer_blocks.0.attn.to_q.weight")
        );
        assert_eq!(krea.logical_key("blocks.0.attn.bogus"), None);
    }

    #[test]
    fn checkpoint_adapter_catalog_uses_shared_portable_authority_and_real_candle_bindings() {
        use candle_gen::gen_core::{
            CheckpointBackend, ImportedModelOperation, ImportedModelRegistration,
            ImportedModelSource, BASE_SNAPSHOT_COMPONENT, FLUX2_CHECKPOINT_ADAPTER,
            KREA_2_CHECKPOINT_ADAPTER, QWEN_IMAGE_CHECKPOINT_ADAPTER, SDXL_CHECKPOINT_ADAPTER,
            WAN_CHECKPOINT_ADAPTER, Z_IMAGE_CHECKPOINT_ADAPTER,
        };

        let registry = super::provider_registry().unwrap();
        let expected = [
            &FLUX2_CHECKPOINT_ADAPTER,
            &KREA_2_CHECKPOINT_ADAPTER,
            &QWEN_IMAGE_CHECKPOINT_ADAPTER,
            &SDXL_CHECKPOINT_ADAPTER,
            // The Wan 2.2 ComfyUI expert pair (sc-20644). Registering it widened this frozen
            // corpus by one row; the corpus is a SHAPE assertion, so it is rewritten here in the
            // same change rather than exempted.
            &WAN_CHECKPOINT_ADAPTER,
            &Z_IMAGE_CHECKPOINT_ADAPTER,
        ];
        let adapters: Vec<_> = registry.checkpoint_adapters().collect();
        assert_eq!(adapters.len(), expected.len());
        for portable in expected {
            let bound = adapters
                .iter()
                .copied()
                .find(|adapter| adapter.adapter_id == portable.adapter_id)
                .unwrap_or_else(|| panic!("missing Candle adapter {}", portable.adapter_id));
            assert!(
                bound.has_same_portable_metadata(portable),
                "{} drifted from portable metadata",
                portable.adapter_id
            );
            assert!(bound
                .backend_bindings
                .iter()
                .all(|binding| binding.backend == CheckpointBackend::Candle));
        }

        let binding_operations = |adapter_id| {
            adapters
                .iter()
                .copied()
                .find(|adapter| adapter.adapter_id == adapter_id)
                .unwrap()
                .backend_bindings
                .iter()
                .map(|binding| binding.operation)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            binding_operations(KREA_2_CHECKPOINT_ADAPTER.adapter_id),
            [
                ImportedModelOperation::Generate,
                ImportedModelOperation::Edit,
                ImportedModelOperation::MultiPhase,
            ],
            "Candle truthfully omits the MLX-only Krea pose route"
        );
        assert_eq!(
            binding_operations(SDXL_CHECKPOINT_ADAPTER.adapter_id),
            [ImportedModelOperation::Generate],
            "Candle truthfully omits the fused SDXL edit route (sc-20651 review). The previous \
             expectation here asserted that Candle 'implements' it; it did not. The binding \
             existed, but it named the txt2img `sdxl` provider, whose descriptor declares no \
             `Reference` conditioning — so the capability floor refused every request the route \
             admitted. The Candle SDXL edit stack (`edit_provider::SdxlEdit`) is a name-driven \
             provider that needs a diffusers snapshot dir and a staged fp16-fix VAE; it has no \
             `LdmComponents`-fed constructor and so cannot serve a fused single-file import. \
             `SDXL_CHECKPOINT_ADAPTER.eligible_backends` is `[Mlx, Candle]`, so the per-operation \
             completeness check does not oblige Candle to bind Edit — exactly as Candle Krea \
             omits the MLX-only pose route above. MLX keeps its Edit binding and honors it."
        );
        for adapter in &adapters {
            if adapter.eligible_backends == [CheckpointBackend::Candle] {
                for operation in adapter.operations {
                    assert!(
                        adapter
                            .backend_bindings
                            .iter()
                            .any(|binding| binding.operation == *operation),
                        "single-backend Candle adapter {} leaves {operation:?} implementation-free",
                        adapter.adapter_id
                    );
                }
            }
        }
        let expected_legacy_projection = [
            ImportedModelRegistration {
                family: "flux2",
                source: ImportedModelSource::ComfyUiTree,
                operation: ImportedModelOperation::Generate,
                provider_id: candle_gen_flux2::config::FLUX2_DEV_ID,
                required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
                inherit_adapters: true,
            },
            // sc-21485: the klein universal BFL transformer single file routes to
            // `flux2_klein_9b` — a separate artifact shape from the dev ComfyUI tree above.
            ImportedModelRegistration {
                family: "flux2",
                source: ImportedModelSource::TransformerFile,
                operation: ImportedModelOperation::Generate,
                provider_id: candle_gen_flux2::config::FLUX2_KLEIN_9B_ID,
                required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
                inherit_adapters: true,
            },
            ImportedModelRegistration {
                family: "krea_2",
                source: ImportedModelSource::TransformerFile,
                operation: ImportedModelOperation::Generate,
                provider_id: candle_gen_krea::KREA_2_TURBO_ID,
                required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
                inherit_adapters: true,
            },
            ImportedModelRegistration {
                family: "krea_2",
                source: ImportedModelSource::TransformerFile,
                operation: ImportedModelOperation::Edit,
                provider_id: candle_gen_krea::KREA_2_TURBO_EDIT_ID,
                required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
                inherit_adapters: true,
            },
            ImportedModelRegistration {
                family: "krea_2",
                source: ImportedModelSource::TransformerFile,
                operation: ImportedModelOperation::MultiPhase,
                provider_id: candle_gen_krea::KREA_2_RAW_ID,
                required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
                inherit_adapters: true,
            },
            ImportedModelRegistration {
                family: "qwen-image",
                source: ImportedModelSource::ComfyUiTree,
                operation: ImportedModelOperation::Generate,
                provider_id: candle_gen_qwen_image::config::MODEL_ID,
                required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
                inherit_adapters: true,
            },
            ImportedModelRegistration {
                family: "sdxl",
                source: ImportedModelSource::FusedCheckpoint,
                operation: ImportedModelOperation::Generate,
                provider_id: candle_gen_sdxl::MODEL_ID,
                required_components: Some(&["tokenizer_clip_l", "tokenizer_clip_bigg"]),
                inherit_adapters: true,
            },
            // No `sdxl` + `FusedCheckpoint` + `Edit` row: Candle does not bind that operation
            // (sc-20651 review). See the `binding_operations` expectation above for why.
            ImportedModelRegistration {
                // Wan's imported route takes NO caller-staged components: the UMT5 encoder, VAE
                // and tokenizer come from a resident snapshot tier the caller resolves, not from
                // `LoadSpec::components`. And `inherit_adapters` is false because
                // `load_from_comfyui_experts` has no adapter seam (sc-20644).
                family: "wan-video",
                source: ImportedModelSource::ComfyUiTree,
                operation: ImportedModelOperation::Generate,
                provider_id: candle_gen_wan::config::MODEL_ID_T2V_14B,
                required_components: None,
                inherit_adapters: false,
            },
            ImportedModelRegistration {
                family: "z-image",
                source: ImportedModelSource::ComfyUiTree,
                operation: ImportedModelOperation::Generate,
                provider_id: candle_gen_z_image::MODEL_ID,
                required_components: Some(&[BASE_SNAPSHOT_COMPONENT]),
                inherit_adapters: true,
            },
        ];
        assert_eq!(
            registry.imported_models().copied().collect::<Vec<_>>(),
            expected_legacy_projection,
            "the legacy catalog surface must be the exact pre-registry compatibility projection"
        );
    }

    #[test]
    fn modelled_video_provider_ids_have_typed_vae_assignments() {
        let registry = super::provider_registry().unwrap();
        let registered: Vec<&str> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id)
            .collect();
        let expected = [
            (
                candle_gen_ltx::config::MODEL_ID,
                candle_gen_ltx::VAE_TILING,
                Some(2_725_804_800),
            ),
            (
                candle_gen_wan::config::MODEL_ID,
                candle_gen_wan::WAN_Z48_VAE_TILING,
                Some(382_730_240),
            ),
            (
                candle_gen_wan::config::MODEL_ID_T2V_14B,
                candle_gen_wan::WAN_Z16_VAE_TILING,
                Some(265_830_400),
            ),
            (
                candle_gen_wan::config::MODEL_ID_I2V_14B,
                candle_gen_wan::WAN_Z16_VAE_TILING,
                Some(265_830_400),
            ),
            (
                candle_gen_wan::config::MODEL_ID_VACE,
                candle_gen_wan::WAN_Z16_VAE_TILING,
                Some(265_830_400),
            ),
            (
                candle_gen_wan::model_vace_fun::MODEL_ID_VACE_FUN,
                candle_gen_wan::model_vace_fun::ProviderVae::VAE_TILING,
                Some(265_830_400),
            ),
            (
                candle_gen_bernini::MODEL_ID,
                candle_gen_bernini::VAE_TILING,
                Some(265_830_400),
            ),
            (
                candle_gen_bernini::bernini::MODEL_ID,
                candle_gen_bernini::VAE_TILING,
                Some(265_830_400),
            ),
            (
                candle_gen_scail2::MODEL_ID,
                candle_gen_scail2::VAE_TILING,
                Some(265_830_400),
            ),
            (
                candle_gen_svd::config::MODEL_ID,
                candle_gen_svd::VAE_TILING,
                None,
            ),
        ];

        for (provider_id, tiling, peak_bytes) in expected {
            assert!(
                registered.contains(&provider_id),
                "unregistered {provider_id}"
            );
            assert_eq!(
                super::vae_tiling(provider_id),
                Some(tiling),
                "{provider_id}"
            );
            assert_eq!(
                super::conservative_video_decode_memory_profile(provider_id, 64, 64, 9)
                    .map(|profile| profile.working_set_bytes()),
                peak_bytes,
                "{provider_id}"
            );
        }
        assert_eq!(super::vae_tiling("ltx_2_3"), None);
        assert_eq!(super::vae_tiling(candle_gen_mochi::MODEL_ID), None);
        assert_eq!(super::vae_tiling("krea_realtime_14b"), None);
        assert_eq!(super::vae_tiling("not_a_provider"), None);
        for provider_id in [
            "ltx_2_3",
            candle_gen_svd::config::MODEL_ID,
            candle_gen_mochi::MODEL_ID,
            "krea_realtime_14b",
            "not_a_provider",
        ] {
            assert_eq!(
                super::conservative_video_decode_memory_profile(provider_id, 64, 64, 9),
                None
            );
        }
        assert_eq!(
            super::conservative_video_decode_memory_profile(
                candle_gen_ltx::config::MODEL_ID,
                64,
                0,
                9,
            ),
            None
        );
        assert_eq!(
            super::conservative_video_decode_memory_profile(
                candle_gen_wan::config::MODEL_ID_T2V_14B,
                u32::MAX,
                u32::MAX,
                u32::MAX,
            ),
            None
        );

        let ltx = super::conservative_video_decode_memory_profile(
            candle_gen_ltx::config::MODEL_ID,
            64,
            64,
            9,
        )
        .unwrap();
        assert_eq!(ltx.resident_decoder_bytes_included(), 0);
        assert_eq!(
            ltx.checked_composed_peak(2_500_000_000, 2_500_000_000),
            Some(5_225_804_800)
        );
        assert_eq!(
            ltx.checked_composed_peak(2_900_000_000, 2_900_000_000),
            Some(5_625_804_800)
        );

        let wan = super::conservative_video_decode_memory_profile(
            candle_gen_wan::config::MODEL_ID_T2V_14B,
            64,
            64,
            9,
        )
        .unwrap();
        assert_eq!(wan.resident_decoder_bytes_included(), 0);
        assert_eq!(
            wan.checked_composed_peak(1_000_000_000, 600_000_000),
            Some(1_265_830_400)
        );
    }

    #[test]
    fn every_registered_generator_advertises_its_exact_latent_space() {
        use candle_gen::gen_core::{
            LatentSpace, FLUX1_LATENT_SPACE, FLUX2_PACKED_LATENT_SPACE, LTX_VIDEO_LATENT_SPACE,
            MAGE_LATENT_SPACE, MOCHI_VIDEO_LATENT_SPACE, QWEN_KREA_Z16_LATENT_SPACE,
            SANA_LATENT_SPACE, SD3_LATENT_SPACE, SDXL_LATENT_SPACE, SEEDVR2_VIDEO_LATENT_SPACE,
            SVD_LATENT_SPACE, WAN_Z16_VIDEO_LATENT_SPACE, WAN_Z48_LATENT_SPACE,
        };

        fn expected(
            descriptor: &candle_gen::gen_core::ModelDescriptor,
        ) -> Option<&'static LatentSpace> {
            match descriptor.family {
                "anima" | "qwen-image" | "krea_2" => Some(&QWEN_KREA_Z16_LATENT_SPACE),
                "wan" if descriptor.id == "wan2_2_ti2v_5b" => Some(&WAN_Z48_LATENT_SPACE),
                "bernini" | "scail2" | "wan" => Some(&WAN_Z16_VIDEO_LATENT_SPACE),
                "flux" | "boogu" | "chroma" | "z-image" => Some(&FLUX1_LATENT_SPACE),
                "stable-diffusion-3" => Some(&SD3_LATENT_SPACE),
                "sdxl" | "kolors" => Some(&SDXL_LATENT_SPACE),
                "flux2" | "ideogram" | "lens" => Some(&FLUX2_PACKED_LATENT_SPACE),
                "ltx" => Some(&LTX_VIDEO_LATENT_SPACE),
                "mage_flow" => Some(&MAGE_LATENT_SPACE),
                "mochi" => Some(&MOCHI_VIDEO_LATENT_SPACE),
                "sana" => Some(&SANA_LATENT_SPACE),
                "seedvr2" => Some(&SEEDVR2_VIDEO_LATENT_SPACE),
                "svd" => Some(&SVD_LATENT_SPACE),
                // SenseNova's flow head emits RGB patches directly; there is no latent decoder seam.
                "sensenova-u1" => None,
                // MiniMax-H3's denoiser emits a 24-channel joint audio+video latent on the
                // 17-frame clip lattice (token-dropped, seam-blended dual decode — see the crate's
                // `chunking` module). No `LatentTemporalLaw` variant expresses that mapping and no
                // external decoder can consume it, so the descriptor deliberately advertises
                // nothing and fails closed against every decoder swap.
                "minimax_h3" => None,
                family => panic!(
                    "{} has unclassified latent lineage for registered family {family}",
                    descriptor.id
                ),
            }
        }

        let registry = super::provider_registry().unwrap();
        for registration in registry.generators() {
            let descriptor = (registration.descriptor)();
            assert_eq!(
                descriptor.denoiser_output_latent_space,
                expected(&descriptor),
                "{} must advertise the latent space its decoder consumes",
                descriptor.id
            );
        }
    }

    /// AC (sc-22661 / epic SC-22657 E1+E2): every registered contract surface in the composed media
    /// catalog publishes an honest byte decomposition and no fabricated architecture axis.
    ///
    /// This is the registry-wide half of the story's acceptance test. The surfaces are built
    /// **weights-free** — the registry names the sentinel snapshot
    /// `/__sceneworks_memory_contract_surface__`, which is not on disk — so every provider here
    /// lands on the walk's Candle arm: `check_memory_contract_asset_facts` is the byte half that
    /// must hold, and `MemoryArchitectureFacts::default()` is the required E2 state. That second
    /// rule catches the converse defect — a provider that published an architecture axis on a
    /// contract built with nothing to read would have hardcoded it from its own provider id.
    ///
    /// It cannot, on its own, catch a provider that derives *nothing* — `default()` is exactly what
    /// that provider publishes here too. `every_materializable_provider_derives_geometry_from_a_snapshot_root`
    /// is the arm that does, and the two run over the same composed registry.
    ///
    /// `runtime-cuda` runs this walk over the CUDA bundle's registry; this one keeps the
    /// composition root itself covered on a lane that needs no accelerator.
    #[test]
    fn every_registered_contract_surface_publishes_honest_facts() {
        let registry = super::memory_contract_surface_registry().unwrap();
        gen_core_testkit::memory_contract_surface_registry_facts_conformance(&registry, None);
        // Non-vacuous: the walk must have had surfaces to reject.
        assert!(
            !registry.memory_contract_surfaces().unwrap().is_empty(),
            "the composed catalog must publish contract surfaces for the facts walk to check"
        );
    }

    /// The providers whose admission accepts a synthetic snapshot root, and which must therefore
    /// derive at least one architecture axis from it (sc-22661).
    ///
    /// This is the non-vacuous half of the E2 story. A provider that returned
    /// `MemoryArchitectureFacts::default()` unconditionally satisfies every weights-free assertion
    /// in `every_registered_contract_surface_publishes_honest_facts` — that walk *requires*
    /// `default()` on the Candle arm — so only rebuilding the contract against a materialized root
    /// can separate "derives nothing" from "has nothing to derive yet". Reverting any one of these
    /// crates' `architecture_facts` to `::default()` turns the assertion below red.
    ///
    /// The catalog's remaining providers are absent for one reason in four shapes, each a
    /// *pre-existing admission rule* rather than anything about facts: the route demands an exact
    /// resolved catalog route (chroma, sd3, sdxl, qwen-image-edit), an exact immutable turnkey
    /// repo/revision (boogu, sana, ideogram, kolors' base tier), a specific tier subdirectory or
    /// component file on disk (scail2's `dit.safetensors`, ltx-2.5's video VAE, anima's
    /// `diffusion_models/`, sensenova's shards, svd's unquantized surface, ltx-2.3's plain split q4
    /// tier), or a whole snapshot layout to walk (krea, bernini, qwen-image). Standing those up is
    /// building each provider's own load fixture — which is exactly what each crate's own
    /// `architecture_facts_*` test does, and which `every_memory_route_crate_reaches_the_catalog`
    /// proves every memory-route crate has.
    const MATERIALIZED_ROOT_PROVIDERS: &[&str] = &[
        "candle_kolors_control",
        "candle_kolors_ipadapter",
        "flux1_dev",
        "flux1_schnell",
        "flux2_dev",
        "flux2_klein_9b",
        "krea_2_turbo_control",
        "lens",
        "lens_turbo",
        "mage_flow",
        "mage_flow_base",
        "mage_flow_edit",
        "mage_flow_edit_base",
        "mage_flow_edit_turbo",
        "mage_flow_turbo",
        "minimax_h3",
        "z_image",
        "z_image_control",
        "z_image_turbo",
        "z_image_turbo_control",
    ];

    /// AC (sc-22661 / epic SC-22657 E2): the registry-level walk is non-vacuous. Every provider
    /// that can be handed a materialized snapshot root must derive geometry from it — the assertion
    /// an unconditional `MemoryArchitectureFacts::default()` fails, and the one the weights-free
    /// walk structurally cannot make.
    #[test]
    fn every_materializable_provider_derives_geometry_from_a_snapshot_root() {
        use std::collections::BTreeSet;

        let tmp = tempfile::tempdir().unwrap();
        let root = synthetic_snapshot_root(tmp.path());
        let expected = MATERIALIZED_ROOT_PROVIDERS
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(
            expected.len(),
            MATERIALIZED_ROOT_PROVIDERS.len(),
            "the materialized-root list must not repeat a provider"
        );
        let registry = super::memory_contract_surface_registry().unwrap();
        let registered = registry
            .memory_strategy_registrations()
            .map(|registration| registration.provider_id)
            .collect::<BTreeSet<_>>();
        assert!(
            expected.is_subset(&registered),
            "the materialized-root list names providers this catalog does not register: {:?}",
            expected.difference(&registered).collect::<Vec<_>>()
        );

        let lookup = |provider_id: &str| expected.contains(provider_id).then(|| root.clone());
        let coverage = gen_core_testkit::memory_contract_surface_registry_facts_conformance(
            &registry,
            Some(&lookup),
        );
        assert_eq!(
            coverage.materialized_providers_checked,
            MATERIALIZED_ROOT_PROVIDERS.len(),
            "every listed provider must have been rebuilt against the synthetic root"
        );
    }

    /// A synthetic snapshot root carrying the component layouts and config keys the catalog's
    /// providers read.
    fn synthetic_snapshot_root(base: &std::path::Path) -> std::path::PathBuf {
        let root = base.join("sc22661-synthetic-snapshot");
        for (component, config) in SYNTHETIC_COMPONENT_CONFIGS {
            let dir = if component.is_empty() {
                root.clone()
            } else {
                root.join(component)
            };
            std::fs::create_dir_all(&dir).expect("synthetic component dir");
            std::fs::write(dir.join("config.json"), config).expect("synthetic component config");
        }
        root
    }

    const SYNTHETIC_COMPONENT_CONFIGS: &[(&str, &str)] = &[
        (
            "",
            r#"{"patch_size": 2, "llm_config": {"hidden_size": 4096, "num_hidden_layers": 32,
                "num_attention_heads": 32, "head_dim": 128}}"#,
        ),
        (
            "transformer",
            r#"{"in_channels": 64, "out_channels": 64, "hidden_size": 3072, "num_heads": 24,
                "num_attention_heads": 24, "attention_head_dim": 128, "depth": 12,
                "num_layers": 19, "num_single_layers": 38, "patch_size": 2, "caption_channels": 2304,
                "num_key_value_heads": 8, "cross_attention_dim": 2048, "context_in_dim": 2560,
                "axes_dim": [16, 56, 56], "checkpoint": false}"#,
        ),
        (
            "unet",
            r#"{"in_channels": 4, "block_out_channels": [320, 640, 1280],
                "attention_head_dim": [5, 10, 20], "layers_per_block": 2, "num_attention_heads": 20}"#,
        ),
        (
            "vae",
            r#"{"latent_channels": 16, "z_channels": 16, "block_out_channels": [128, 256, 512, 512],
                "scale_factor_spatial": 8, "temperal_downsample": [false, true, true],
                "patch_size": [1, 2, 2], "dim_mult": [1, 2, 4, 4]}"#,
        ),
    ];

    #[test]
    fn every_registered_memory_strategy_rejects_cross_route_decode_geometry() {
        use std::collections::{BTreeMap, BTreeSet};

        let registry = super::provider_registry().unwrap();
        let contract_registry = super::memory_contract_surface_registry().unwrap();
        gen_core_testkit::memory_contract_surface_registry_conformance(&contract_registry);
        let surfaces = contract_registry.memory_contract_surfaces().unwrap();

        // Shape, not population. The registration/surface/composed totals used to be pinned as
        // hand-maintained numbers (`24`/`23`, `21 * 12 + 2 * 16 [+ 6]`, `4`), which any legitimate
        // catalog growth re-trips while saying nothing about *which* provider moved.
        //
        // Scope, stated precisely: every claim below reads the same registry, so it constrains that
        // registry's *internal consistency* — a witness set that is orphaned, doubled, ragged, or
        // mismatched against the composed-route seam. It cannot, on its own, notice a registration
        // that vanished together with its fixture, because that shrinks both sides at once. The
        // registration set is anchored to something outside the registry — the provider crate
        // sources — by `every_memory_route_crate_reaches_the_catalog`, which is what replaced the
        // `24`/`23` pin. Read the two together.

        // Coverage is exact in both directions: each memory-strategy registration either publishes
        // optimized surfaces or is a resident-only witness, and nothing else reaches the inventory.
        let strategy_ids = contract_registry
            .memory_strategy_registrations()
            .map(|registration| registration.provider_id)
            .collect::<BTreeSet<_>>();
        let resident_only_ids = contract_registry
            .resident_only_memory_contract_registrations()
            .map(|registration| registration.provider_id)
            .collect::<BTreeSet<_>>();
        assert!(
            resident_only_ids.is_subset(&strategy_ids),
            "resident-only witnesses without a memory-strategy registration: {:?}",
            resident_only_ids
                .difference(&strategy_ids)
                .collect::<Vec<_>>()
        );
        let surfaced_ids = surfaces
            .iter()
            .map(|surface| surface.contract.provider_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            surfaced_ids,
            strategy_ids
                .difference(&resident_only_ids)
                .copied()
                .collect::<BTreeSet<_>>(),
            "optimized surfaces must cover exactly the non-resident-only memory registrations"
        );

        // What the old `21 * 12 + 2 * 16 + 6` arithmetic was really asserting, per provider: a
        // witness set is a full rectangle over the residency/materialization axes for every tier it
        // publishes (12 = 3 tiers x 2 policies x 2 shapes, 16 = 4 tiers x 4, and TI2V-5B's 6 =
        // 3 tiers x 2 policies x its single eager shape). Stated as a rectangle it is derived, so it
        // holds at any catalog size — and on the tier and shape axes it is stronger, because a
        // provider that dropped one (policy, shape) cell only moved the old global sum by
        // coincidence.
        //
        // A rectangle alone cannot see a *whole* axis collapse, though: drop every Sequential
        // surface and the remaining Resident-only set is still a perfect rectangle. The residency
        // axis is the one with a universe that does not vary by provider — every Candle memory route
        // is witnessed under both offload policies — so it is pinned to the `OffloadPolicy` enum
        // rather than to the provider's own set. Tier and shape stay per-provider derived: those
        // genuinely differ (TI2V-5B has no deferred block loader, tiers vary with what a route
        // ships).
        //
        // `MemoryContractSurfaceSelector::id` is the stable `tier:policy:shape` encoding of the three
        // axes, so splitting it recovers them without needing `Ord` on the runtime enums.
        let axes = |id: &'static str| {
            let mut parts = id.split(':');
            let tier = parts.next().expect("selector ids start with a tier");
            let policy = parts.next().expect("selector ids name a residency policy");
            let shape = parts
                .next()
                .expect("selector ids name a materialization shape");
            assert_eq!(parts.next(), None, "unexpected selector id shape: {id:?}");
            (tier, policy, shape)
        };
        let offload_policy_universe = {
            use candle_gen::gen_core::{
                LoadShape, MemoryContractSurfaceSelector, MemoryContractSurfaceTier, OffloadPolicy,
            };
            // Exhaustiveness guard: a new `OffloadPolicy` variant has to be classified here on
            // purpose instead of quietly shrinking the universe every provider is measured against.
            fn _every_policy_is_listed(policy: OffloadPolicy) {
                match policy {
                    OffloadPolicy::Resident | OffloadPolicy::Sequential => {}
                }
            }
            let tokens = [OffloadPolicy::Resident, OffloadPolicy::Sequential]
                .into_iter()
                .map(|offload_policy| {
                    axes(
                        MemoryContractSurfaceSelector {
                            tier: MemoryContractSurfaceTier::Bf16,
                            offload_policy,
                            load_shape: LoadShape::EagerMaterialization,
                        }
                        .id(),
                    )
                    .1
                })
                .collect::<Vec<_>>();
            let universe = tokens.iter().copied().collect::<BTreeSet<_>>();
            assert_eq!(
                universe.len(),
                tokens.len(),
                "each OffloadPolicy must encode a distinct selector token"
            );
            universe
        };

        let mut published = BTreeMap::<&str, BTreeSet<_>>::new();
        for surface in &surfaces {
            published
                .entry(surface.contract.provider_id.as_str())
                .or_default()
                .insert(axes(surface.selector.id()));
        }
        for (provider_id, selectors) in &published {
            let tiers = selectors
                .iter()
                .map(|(tier, ..)| *tier)
                .collect::<BTreeSet<_>>();
            let policies = selectors
                .iter()
                .map(|(_, policy, _)| *policy)
                .collect::<BTreeSet<_>>();
            let shapes = selectors
                .iter()
                .map(|(.., shape)| *shape)
                .collect::<BTreeSet<_>>();
            let mut rectangle = BTreeSet::new();
            for tier in &tiers {
                for policy in &policies {
                    for shape in &shapes {
                        rectangle.insert((*tier, *policy, *shape));
                    }
                }
            }
            assert_eq!(
                *selectors, rectangle,
                "{provider_id} publishes a ragged witness set: every tier it declares must be \
                 witnessed on every residency/materialization combination it supports"
            );
            assert_eq!(
                policies, offload_policy_universe,
                "{provider_id} witnesses only the {policies:?} residency policies; every Candle \
                 memory route must be witnessed across the whole OffloadPolicy universe \
                 {offload_policy_universe:?}. A rectangle survives a whole axis collapsing, so this \
                 is checked against the enum rather than against the provider's own set"
            );
        }

        // Composed routes are exactly the memory registrations with no standalone generator:
        // `register_memory_strategy` rejects an unmatched id, so `register_composed_memory_strategy`
        // is the only seam that admits one. Deriving the set that way asserts the `composed` flag
        // agrees with the registration seam rather than pinning how many composed routes exist.
        let generator_ids = contract_registry
            .generators()
            .map(|registration| (registration.descriptor)().id)
            .collect::<BTreeSet<_>>();
        let observed_composed = surfaces
            .iter()
            .filter(|surface| surface.composed)
            .map(|surface| surface.contract.provider_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            observed_composed,
            surfaced_ids
                .iter()
                .copied()
                .filter(|id| !generator_ids.contains(id))
                .collect::<BTreeSet<_>>(),
            "the composed flag must mark exactly the generator-less memory routes"
        );
        assert!(
            !observed_composed.is_empty(),
            "the Candle catalog publishes composed memory routes; none were observed"
        );

        let spec = candle_gen::gen_core::LoadSpec::new(candle_gen::gen_core::WeightsSource::Dir(
            "/nonexistent".into(),
        ));
        gen_core_testkit::memory_strategy_registry_conformance(&registry, &spec);
    }

    #[test]
    fn krea_raw_and_edit_publish_exact_request_scoped_candle_surfaces() {
        use candle_gen::gen_core::{
            MemoryContractSurfaceTier, MemoryStrategy, MemoryStrategySupport,
        };

        let registry = super::memory_contract_surface_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        for provider_id in ["krea_2_raw", "krea_2_edit", "krea_2_turbo_edit"] {
            let provider_surfaces: Vec<_> = surfaces
                .iter()
                .filter(|surface| surface.contract.provider_id == provider_id)
                .collect();
            assert_eq!(provider_surfaces.len(), 12, "{provider_id}");
            let tiers = provider_surfaces
                .iter()
                .map(|surface| surface.resolved_artifact_tier())
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(
                tiers,
                [
                    MemoryContractSurfaceTier::Bf16,
                    MemoryContractSurfaceTier::Q4,
                    MemoryContractSurfaceTier::Q8,
                ]
                .into_iter()
                .collect(),
                "{provider_id}"
            );
            // sc-22735: these are weights-free declarations, so their identities live in the
            // registry-behavior namespace and carry their own (route, declared tier) key. The
            // production strings — `krea-2-<route>-<tier>-cuda-staged-residency-v1` — are what a
            // measured anchor binds to, and no catalog surface may republish one.
            let mut declared = std::collections::BTreeSet::new();
            for surface in provider_surfaces {
                assert!(!surface.composed, "{provider_id}");
                assert_eq!(
                    surface.contract.asset_facts,
                    candle_gen::gen_core::MemoryAssetFacts::default(),
                    "{provider_id}"
                );
                let fingerprint = &surface.contract.calibration.as_ref().unwrap().fingerprint;
                let tier = match surface.resolved_artifact_tier() {
                    MemoryContractSurfaceTier::Bf16 => "bf16",
                    MemoryContractSurfaceTier::Q4 => "q4",
                    MemoryContractSurfaceTier::Q8 => "q8",
                    other => panic!("{provider_id}: unexpected surface tier {other:?}"),
                };
                assert_eq!(
                    fingerprint,
                    &format!(
                        "krea-2-candle-registry-behavior-v1-{}-{tier}",
                        provider_id.replace('_', "-")
                    ),
                    "{provider_id}:{}",
                    surface.selector.id()
                );
                for cell in ["q4", "q8", "bf16"] {
                    assert_ne!(
                        fingerprint,
                        &format!(
                            "{}-{cell}-cuda-staged-residency-v1",
                            provider_id.replace('_', "-")
                        ),
                        "{provider_id}: a weights-free surface must never republish a production \
                         calibration identity"
                    );
                }
                declared.insert(fingerprint.clone());
                for strategy in MemoryStrategy::ALL {
                    let expected = matches!(
                        strategy,
                        MemoryStrategy::Resident | MemoryStrategy::StagedResidency
                    );
                    assert_eq!(
                        surface.contract.capability(strategy).unwrap().support
                            == MemoryStrategySupport::Implemented,
                        expected,
                        "{provider_id}:{}:{strategy:?}",
                        surface.selector.id()
                    );
                }
            }
            assert_eq!(
                declared.len(),
                3,
                "{provider_id}: one declaration per artifact tier: {declared:?}"
            );
        }
    }

    #[test]
    fn z_image_and_lens_publish_exact_typed_candle_rung_four_surfaces() {
        use candle_gen::gen_core::{
            LoadShape, MemoryContractSurfaceTier, MemoryStrategy, MemoryStrategySupport,
            OffloadPolicy,
        };

        let registry = super::memory_contract_surface_registry().unwrap();
        let surfaces = registry.memory_contract_surfaces().unwrap();
        for (provider_id, expected_count, composed, fingerprint) in [
            (
                "z_image",
                6,
                false,
                "z-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v2",
            ),
            (
                "z_image_turbo",
                6,
                false,
                "z-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v2",
            ),
            (
                "z_image_control",
                6,
                true,
                "z-image-cuda-base-control-host-decode-streamed-device-format-blocks-v2",
            ),
            (
                "z_image_turbo_control",
                6,
                true,
                "z-image-cuda-base-control-host-decode-streamed-device-format-blocks-v2",
            ),
            // sc-22732: the Lens weights-free surfaces publish a per-cell static behavior identity
            // instead of leaking the measured q4 Lens-Turbo production string onto all 24 of them,
            // so these two entries are the route-exact PREFIX of that namespace.
            ("lens", 2, false, "lens-candle-registry-behavior-v1-lens-"),
            (
                "lens_turbo",
                2,
                false,
                "lens-candle-registry-behavior-v1-lens-turbo-",
            ),
        ] {
            let provider_surfaces: Vec<_> = surfaces
                .iter()
                .filter(|surface| surface.contract.provider_id == provider_id)
                .collect();
            assert_eq!(provider_surfaces.len(), 12, "{provider_id}");
            let mut implemented = 0;
            for surface in provider_surfaces {
                let expected = if provider_id.starts_with("lens") {
                    matches!(
                        surface.resolved_artifact_tier(),
                        MemoryContractSurfaceTier::Q4 | MemoryContractSurfaceTier::Q8
                    ) && surface.selector.offload_policy == OffloadPolicy::Sequential
                        && surface.selector.load_shape == LoadShape::DeferredMaterialization
                } else {
                    matches!(
                        surface.resolved_artifact_tier(),
                        MemoryContractSurfaceTier::Bf16
                            | MemoryContractSurfaceTier::Q4
                            | MemoryContractSurfaceTier::Q8
                    ) && surface.selector.load_shape == LoadShape::DeferredMaterialization
                };
                let rung = surface
                    .contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .expect("complete Candle ladder");
                assert_eq!(
                    rung.support,
                    if expected {
                        MemoryStrategySupport::Implemented
                    } else {
                        MemoryStrategySupport::Missing
                    },
                    "{}:{}",
                    provider_id,
                    surface.selector.id()
                );
                implemented += usize::from(expected);
                assert_eq!(surface.composed, composed, "{provider_id}");
                let published = &surface.contract.calibration.as_ref().unwrap().fingerprint;
                if provider_id.starts_with("lens") {
                    assert!(
                        published.starts_with(fingerprint),
                        "{provider_id}:{} published {published}",
                        surface.selector.id()
                    );
                    // `…-v1-lens-` is itself a prefix of `…-v1-lens-turbo-`, so the check above
                    // cannot tell the base route's identity from the turbo route's. Pin the
                    // discrimination explicitly rather than leaving it to prefix arithmetic.
                    assert_eq!(
                        published.starts_with("lens-candle-registry-behavior-v1-lens-turbo-"),
                        provider_id == "lens_turbo",
                        "{provider_id}:{} published {published}",
                        surface.selector.id()
                    );
                    // The weights-free namespace must never be the measured production string.
                    assert_ne!(
                        published.as_str(),
                        "lens-candle-cuda-shared-ladder-device-format-blocks-v1",
                        "{provider_id}:{} leaks the measured production identity",
                        surface.selector.id()
                    );
                } else {
                    assert_eq!(published, fingerprint, "{provider_id}");
                }
                assert_eq!(
                    surface.contract.asset_facts,
                    candle_gen::gen_core::MemoryAssetFacts::default(),
                    "{provider_id}: weights-free surfaces cannot claim real inventory"
                );
                if composed {
                    assert!(
                        surface.spec.control.is_some(),
                        "{provider_id}:{} missing mandatory control source",
                        surface.selector.id()
                    );
                }
            }
            assert_eq!(implemented, expected_count, "{provider_id}");
        }
    }

    #[test]
    fn cfg_capability_matrix_matches_the_registered_candle_render_paths() {
        let registry = super::provider_registry().unwrap();
        let descriptor = |id: &str| {
            registry
                .generators()
                .find_map(|registration| {
                    let descriptor = (registration.descriptor)();
                    (descriptor.id == id).then_some(descriptor)
                })
                .unwrap_or_else(|| panic!("{id} missing from Candle catalog"))
        };

        let klein = descriptor("flux2_klein_9b").capabilities;
        assert!(klein.supports_guidance);
        assert!(klein.supports_negative_prompt);
        assert!(!klein.supports_true_cfg);

        for id in ["sd3_5_large", "sd3_5_medium"] {
            let capabilities = descriptor(id).capabilities;
            assert!(capabilities.supports_guidance, "{id}");
            assert!(capabilities.supports_negative_prompt, "{id}");
            assert!(!capabilities.supports_true_cfg, "{id}");
        }
        let turbo = descriptor("sd3_5_large_turbo").capabilities;
        assert!(!turbo.supports_guidance);
        assert!(!turbo.supports_negative_prompt);
        assert!(!turbo.supports_true_cfg);

        for id in ["sensenova_u1_8b", "sensenova_u1_8b_fast"] {
            let capabilities = descriptor(id).capabilities;
            assert!(capabilities.supports_guidance, "{id}");
            assert!(!capabilities.supports_negative_prompt, "{id}");
            assert!(capabilities.supports_true_cfg, "{id} image CFG");
            assert_eq!(
                capabilities.conditioning,
                vec![
                    ConditioningKind::Reference,
                    ConditioningKind::MultiReference,
                ],
                "{id} registered conditioned-image surface"
            );
        }
    }

    #[test]
    fn complete_catalog_has_stable_conforming_surface() {
        let registry = super::provider_registry().unwrap();
        let generators: Vec<String> = registry
            .generators()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        let trainers: Vec<String> = registry
            .trainers()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        let captioners: Vec<String> = registry
            .captioners()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        let image_embedders: Vec<String> = registry
            .image_embedders()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        let text_embedders: Vec<String> = registry
            .text_embedders()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();

        assert_eq!(registry.transforms().len(), 0);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
        assert!(registry
            .generators()
            .all(|r| (r.descriptor)().backend == "candle"));
        assert!(registry
            .trainers()
            .all(|r| (r.descriptor)().backend == "candle"));
        assert_eq!(
            generators,
            [
                "anima_base",
                "anima_aesthetic",
                "anima_turbo",
                "bernini_renderer",
                "bernini",
                "boogu_image",
                "boogu_image_turbo",
                "boogu_image_edit",
                "chroma1_hd",
                "chroma1_base",
                "chroma1_flash",
                "flux1_schnell",
                "flux1_dev",
                "flux2_klein_9b",
                "flux2_dev",
                "ideogram_4",
                "ideogram_4_turbo",
                "kolors",
                "krea_2_turbo",
                "krea_2_raw",
                "krea_2_edit",
                "krea_2_turbo_edit",
                "lens_turbo",
                "lens",
                "ltx_2_3_distilled",
                "ltx_2_5_distilled",
                "mage_flow",
                "mage_flow_base",
                "mage_flow_turbo",
                "mage_flow_edit",
                "mage_flow_edit_base",
                "mage_flow_edit_turbo",
                "minimax_h3",
                "mochi_1",
                "qwen_image",
                "sana_1600m",
                "sana_sprint_1600m",
                "scail2_14b",
                "sd3_5_large",
                "sd3_5_large_turbo",
                "sd3_5_medium",
                "sdxl",
                "seedvr2",
                "seedvr2_3b",
                "seedvr2_7b",
                "sensenova_u1_8b",
                "sensenova_u1_8b_fast",
                "svd_xt",
                "wan2_2_ti2v_5b",
                "wan2_2_t2v_14b",
                "wan2_2_i2v_14b",
                "wan_vace",
                "wan2_2_vace_fun_14b",
                "z_image_turbo",
                "z_image",
            ]
        );
        assert_eq!(
            trainers,
            [
                "anima_base",
                "kolors",
                "krea_2_raw",
                "krea_2_control",
                "lens",
                "ltx_2_3",
                "ltx_2_5_distilled",
                "mage_flow_base",
                "sd3_5_large",
                "sd3_5_medium",
                "sdxl",
                "wan2_2_t2v_14b",
                "wan2_2_i2v_14b",
                "wan2_2_ti2v_5b",
                "z_image_turbo",
            ]
        );
        assert_eq!(
            captioners,
            ["fancyfeast/llama-joycaption-beta-one-hf-llava"]
        );
        assert_eq!(image_embedders, ["clip_vit_l14"]);
        assert_eq!(text_embedders, ["clip_vit_l14_text"]);

        // sc-16667: the pinned surface and the model-weight licence mapping move together — this is
        // where a surface change and a mapping change meet. Five of the seven trainer ids are also
        // generator ids, which is why 55 generators + 2 trainer-only ids + 1 captioner + 2
        // embedders are 60 distinct ids.
        //
        // Registration is never conditioned on the mapping: 50 < 60 because ten ids load nothing
        // the shared checkpoint table covers, and they ship exactly as before. That gap is a hole in
        // our metadata for CI to report, and `licenses::tests` pins which ten and why — as
        // `#[cfg(test)]` data, so no gate can read it and suppress them.
        let distinct: std::collections::BTreeSet<&String> = generators
            .iter()
            .chain(&trainers)
            .chain(&captioners)
            .chain(&image_embedders)
            .chain(&text_embedders)
            .collect();
        assert_eq!(distinct.len(), 60);
        assert_eq!(super::provider_components().len(), 50);
    }

    /// The manifest emitter runs on **this** catalog's three slices, and its output is
    /// deterministic.
    ///
    /// Deliberately cheap. Byte-stability of the shared emitter was settled in sc-16663 (#406,
    /// #411) and the audio lane gates its bytes against a committed file; there is no committed
    /// Candle media manifest to compare against, because merging the three catalogs into one
    /// release artifact is sc-16664's job. So this pins only what is this crate's to pin: the
    /// function is reachable, emits the schema-3 shape with all three layers present, and returns
    /// the same bytes call to call.
    #[test]
    fn component_licenses_manifest_json_is_well_formed_and_stable() {
        let generated = super::component_licenses_manifest_json();
        assert!(!generated.is_empty());

        let parsed: serde_json::Value =
            serde_json::from_str(&generated).expect("manifest is valid JSON");
        assert_eq!(parsed["schema_version"], 3);
        assert_eq!(parsed["kind"], "model-weight-licenses");
        // A consumer reads one document and finds all three layers in it.
        for section in ["families", "components", "providers"] {
            assert!(
                parsed[section]
                    .as_array()
                    .is_some_and(|rows| !rows.is_empty()),
                "manifest section {section:?} is missing or empty"
            );
        }
        assert_eq!(
            parsed["providers"].as_array().expect("providers").len(),
            super::provider_components().len()
        );

        assert_eq!(
            super::component_licenses_manifest_json(),
            generated,
            "the emitter must be deterministic — a merged release artifact compares bytes"
        );
    }

    #[test]
    fn mage_is_shipped_with_truthful_quant_surface() {
        let registry = super::provider_registry().expect("catalog");
        for id in [
            "mage_flow",
            "mage_flow_base",
            "mage_flow_turbo",
            "mage_flow_edit",
            "mage_flow_edit_base",
            "mage_flow_edit_turbo",
        ] {
            let descriptor = registry
                .generators()
                .map(|registration| (registration.descriptor)())
                .find(|descriptor| descriptor.id == id)
                .unwrap_or_else(|| panic!("{id} missing from Candle catalog"));
            assert_eq!(descriptor.backend, "candle");
            assert!(!descriptor.capabilities.mac_only);
            assert_eq!(
                descriptor.capabilities.supported_quants,
                &[
                    super::media::gen_core::Quant::Q4,
                    super::media::gen_core::Quant::Q8
                ]
            );
        }
    }

    /// Pin the NVFP4 tier's catalog surface (epic 11037, sc-11042 Option A): the FP4 tier is exposed
    /// **only** under the `cuda` feature (consumer Blackwell `sm_120`); the CPU candle bundle surfaces
    /// no advanced tier. The MLX/macOS runtime uses a separate `mlx-gen-catalog` with no such surface —
    /// the third leg of the pinned platform difference.
    #[test]
    fn nvfp4_tier_surface_is_cuda_only() {
        #[cfg(feature = "cuda")]
        assert_eq!(
            super::nvfp4_quant_tiers(),
            &[super::media::gen_core::Quant::Nvfp4],
            "NVFP4 must be surfaced on the cuda catalog"
        );
        #[cfg(not(feature = "cuda"))]
        assert!(
            super::nvfp4_quant_tiers().is_empty(),
            "NVFP4 must NOT be surfaced on a non-cuda (CPU) candle catalog"
        );

        // `SURFACES_NVFP4_TIER` is what dependent bundles read instead of a `cfg!` they cannot write;
        // it must never drift from the tier list itself.
        assert_eq!(
            super::SURFACES_NVFP4_TIER,
            !super::nvfp4_quant_tiers().is_empty(),
            "SURFACES_NVFP4_TIER must agree with nvfp4_quant_tiers()"
        );
    }

    #[test]
    fn krea_cuda_memory_contract_is_not_exposed_by_the_cpu_catalog() {
        let registry = super::provider_registry().expect("catalog");
        let spec = super::media::gen_core::LoadSpec::new(
            super::media::gen_core::WeightsSource::Dir("/nonexistent".into()),
        );
        let contract = registry
            .memory_strategy_contract("krea_2_turbo", &spec)
            .expect("known Krea generator");
        #[cfg(feature = "cuda")]
        assert!(
            contract.is_some(),
            "CUDA catalog must expose the Krea CUDA contract"
        );
        #[cfg(not(feature = "cuda"))]
        assert!(
            contract.is_none(),
            "CPU catalog must leave Krea on its compatibility default"
        );
    }
}
