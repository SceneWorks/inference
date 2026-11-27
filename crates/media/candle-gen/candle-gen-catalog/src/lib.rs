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
    pub use candle_gen_flux3 as flux3;
    pub use candle_gen_ideogram as ideogram;
    pub use candle_gen_instantid as instantid;
    pub use candle_gen_iris as iris;
    pub use candle_gen_joycaption as joycaption;
    pub use candle_gen_kolors as kolors;
    pub use candle_gen_krea as krea;
    pub use candle_gen_lens as lens;
    pub use candle_gen_ltx as ltx;
    pub use candle_gen_mage as mage;
    pub use candle_gen_minimax_h3 as minimax_h3;
    pub use candle_gen_mochi as mochi;
    pub use candle_gen_moss_image as moss_image;
    pub use candle_gen_pid as pid;
    pub use candle_gen_pulid as pulid;
    pub use candle_gen_qwen_image as qwen_image;
    pub use candle_gen_qwen_image_2_1 as qwen_image_2_1;
    pub use candle_gen_sam3 as sam3;
    pub use candle_gen_sana as sana;
    pub use candle_gen_scail2 as scail2;
    pub use candle_gen_sd3 as sd3;
    pub use candle_gen_sdxl as sdxl;
    pub use candle_gen_seedvr2 as seedvr2;
    pub use candle_gen_sensenova as sensenova;
    pub use candle_gen_supra2 as supra2;
    pub use candle_gen_svd as svd;
    pub use candle_gen_wan as wan;
    pub use candle_gen_z_image as z_image;
}

/// Platform-owned crates consumed through provider-specific APIs rather than the registry
/// `load(id, spec)` path (depth maps, face analysis, segmentation, the PiD latent decoder).
/// Listed here so their platform membership is as explicit as a registered generator.
pub const BESPOKE_UTILITY_CRATES: &[&str] = &["depth", "face", "pid", "sam3"];

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
/// Empty after `SW-6982` registered `pulid_flux` and `instantid` as full `RouteRecord` generators.
pub const BESPOKE_MEMORY_ROUTE_WAIVERS: &[BespokeMemoryRouteWaiver] = &[];

fn register_providers_with_flux3_mode(
    registry: ProviderRegistryBuilder,
    cuda_flux3_target: bool,
) -> ProviderRegistryBuilder {
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
    let registry = if cuda_flux3_target {
        candle_gen_flux3::register_cuda_descriptor_providers(registry)
    } else {
        candle_gen_flux3::register_providers(registry)
    };
    let registry = candle_gen_ideogram::register_providers(registry);
    let registry = candle_gen_instantid::register_providers(registry);
    let registry = candle_gen_iris::register_providers(registry);
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
    let registry = candle_gen_moss_image::register_providers(registry);
    let registry = candle_gen_pulid::register_providers(registry);
    let registry = candle_gen_qwen_image::register_providers(registry);
    let registry = candle_gen_qwen_image_2_1::register_providers(registry);
    let registry = candle_gen_sana::register_providers(registry);
    let registry = candle_gen_scail2::register_providers(registry);
    let registry = candle_gen_sd3::register_providers(registry);
    let registry = candle_gen_sdxl::register_providers(registry);
    let registry = candle_gen_seedvr2::register_providers(registry);
    let registry = candle_gen_sensenova::register_providers(registry);
    // sc-24214: Supra2-IMG under the same engine id as its MLX twin (`supra2_img`).
    let registry = candle_gen_supra2::register_providers(registry);
    let registry = candle_gen_svd::register_providers(registry);
    let registry = candle_gen_wan::register_providers(registry);
    candle_gen_z_image::register_providers(registry)
}

/// Add every provider shipped by the Candle media platform to an explicit registry builder.
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    register_providers_with_flux3_mode(registry, false)
}

/// Build the complete explicit Candle media provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<ProviderRegistry> {
    register_providers(ProviderRegistryBuilder::new()).build()
}

/// Register the hardware-free descriptor-only Candle media providers representing the shipping
/// CUDA feature set onto `registry` (`SW-6980`, `SW-6982`, `REG-3`).
pub fn register_cuda_descriptor_providers(
    registry: ProviderRegistryBuilder,
) -> ProviderRegistryBuilder {
    let registry = register_providers_with_flux3_mode(registry, true);
    #[cfg(not(feature = "cuda"))]
    let registry = {
        let registry = candle_gen_flux::register_memory_contract_surfaces(registry);
        let registry = candle_gen_flux2::register_memory_contract_surfaces(registry);
        let registry = candle_gen_krea::register_memory_contract_surfaces(registry);
        let registry = candle_gen_lens::register_memory_contract_surfaces(registry);
        let registry = candle_gen_ltx::register_memory_contract_surfaces(registry);
        let registry = candle_gen_mage::register_memory_contract_surfaces(registry);
        let registry = candle_gen_minimax_h3::register_memory_contract_surfaces(registry);
        let registry = candle_gen_pulid::register_memory_contract_surfaces(registry);
        let registry = candle_gen_qwen_image::register_memory_contract_surfaces(registry);
        let registry = candle_gen_z_image::register_memory_contract_surfaces(registry);
        let registry = candle_gen_bernini::register_memory_contract_surfaces(registry);
        let registry = candle_gen_scail2::register_memory_contract_surfaces(registry);
        candle_gen_wan::register_memory_contract_surfaces(registry)
    };
    registry
}

/// Build the hardware-free descriptor-only Candle media provider catalog representing the
/// shipping CUDA feature set (`SW-6980`, `REG-3`).
///
/// Produces the exact route, descriptor, quantization-tier (including FLUX 3 FP8R / NVFP4),
/// and memory-contract surface of [`provider_registry`] under `--features cuda` without
/// requiring the NVIDIA CUDA toolkit.
pub fn cuda_descriptor_provider_registry() -> candle_gen::gen_core::Result<ProviderRegistry> {
    register_cuda_descriptor_providers(ProviderRegistryBuilder::new()).build()
}

/// Advanced quant tiers surfaced by the shipping CUDA feature set, usable in hardware-free
/// descriptor-only exports without `--features cuda` (`SW-6980`, `REG-3`).
pub fn cuda_descriptor_nvfp4_quant_tiers() -> &'static [media::gen_core::Quant] {
    &[media::gen_core::Quant::Nvfp4]
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
    let registry = candle_gen_pulid::register_memory_contract_surfaces(registry);
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

#[cfg(test)]
mod tests;
