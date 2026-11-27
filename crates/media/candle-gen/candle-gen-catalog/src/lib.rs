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

/// Whether a registered provider's model attention runs through `candle-gen`'s shared SDPA helpers
/// (`sdpa_budgeted_*` / `sdpa_planned_*` / the `training_sdpa_*` inference entry points) — the one
/// dispatch the job-level attention switches act on (SW-6130 flash toggle, SW-6132 lossy opt-ins).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedAttentionRouting {
    /// The provider's transformer / UNet attention goes through the shared helpers: the flash
    /// toggle reaches it, and so does each lossy kernel the provider can take
    /// ([`LossyServe::Served`]) when opted in. Each call is still eligibility-checked; a call a
    /// lossy kernel cannot serve keeps FlashAttention / the budgeted path, counted in
    /// `AttentionPathCounts::{int8_refusals, sol_refusals}`. A kernel the provider's attention can
    /// never take (f32 attention, a head width or attention pattern the kernel does not serve) is
    /// [`LossyServe::Never`] and refused upfront by [`admit_lossy_attention`].
    SharedHelpers {
        /// Whether the provider's attention can take the INT8 QK kernel (SW-6131).
        int8: LossyServe,
        /// Whether the provider's attention can take the Sol-Attn kernel (SW-6137).
        sol: LossyServe,
    },
    /// The provider's model attention is its own code (outside the shared helpers): no job-level
    /// attention switch reaches it, so a product must refuse a lossy-attention opt-in for it with a
    /// typed error rather than silently render the exact path ([`admit_lossy_attention`]).
    Bespoke,
}

/// Whether a [`SharedAttentionRouting::SharedHelpers`] provider's attention can take one lossy
/// kernel at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossyServe {
    /// Its attention calls are shaped for the kernel (bf16 / f16 on CUDA, a head width and an
    /// attention pattern the kernel serves); each call is still eligibility-checked.
    Served,
    /// Its attention can never take the kernel, and why (the typed refusal's text): opting in
    /// would only render the exact path.
    Never(&'static str),
}

impl SharedAttentionRouting {
    /// Shared helpers, both lossy kernels served.
    pub const ALL: Self = Self::SharedHelpers {
        int8: LossyServe::Served,
        sol: LossyServe::Served,
    };

    /// Shared helpers, the INT8 kernel served and Sol-Attn never (`why`).
    pub const fn int8_only(why: &'static str) -> Self {
        Self::SharedHelpers {
            int8: LossyServe::Served,
            sol: LossyServe::Never(why),
        }
    }

    /// Shared helpers (the flash toggle reaches it) but neither lossy kernel can serve (`why`).
    pub const fn exact_only(why: &'static str) -> Self {
        Self::SharedHelpers {
            int8: LossyServe::Never(why),
            sol: LossyServe::Never(why),
        }
    }
}

/// Every provider this catalog registers (generators, transforms, captioners, embedders) plus the
/// worker-owned bespoke routes ([`BESPOKE_MEMORY_ROUTE_WAIVERS`], InstantID), and how its model
/// attention routes. Trainers are not listed: training attention always runs the
/// materialized path (epic E9), whatever the job switches say. `catalog_attention_routing_covers_every_provider`
/// (tests) pins this table to the registry, so a new provider must state its routing here.
pub const SHARED_ATTENTION_ROUTING: &[(&str, SharedAttentionRouting)] = {
    use SharedAttentionRouting as R;
    const ALL: R = R::ALL;
    const BESPOKE: R = R::Bespoke;
    const F32: &str = "runs its attention in f32; the INT8 / Sol-Attn kernels take bf16 / f16 only";
    const HD64: &str = "has 64-wide attention heads; Sol-Attn takes head dim 128 only";
    const HD120: &str = "has 120-wide attention heads; the INT8 kernel takes head dim 64 / 128 \
                         and Sol-Attn 128";
    const HD256: &str = "has 256-wide attention heads; the INT8 kernel takes head dim 64 / 128 \
                         and Sol-Attn 128";
    const BLOCK_CAUSAL: &str = "attends block-causally (no unmasked whole-sequence \
                                self-attention); Sol-Attn takes only that";
    const MASKED_F32: &str = "runs its masked attention in f32; the INT8 / Sol-Attn kernels take \
                              unmasked bf16 / f16 only";
    &[
        ("anima_base", ALL),
        ("anima_aesthetic", ALL),
        ("anima_turbo", ALL),
        // Bernini renders with Wan's `WanTransformer`.
        ("bernini_renderer", ALL),
        ("bernini", ALL),
        ("boogu_image", R::exact_only(HD120)),
        ("boogu_image_turbo", R::exact_only(HD120)),
        ("boogu_image_edit", R::exact_only(HD120)),
        // Chroma's DiT loads and attends at f32.
        ("chroma1_hd", R::exact_only(F32)),
        ("chroma1_base", R::exact_only(F32)),
        ("chroma1_flash", R::exact_only(F32)),
        ("flux1_schnell", ALL),
        ("flux1_dev", ALL),
        // FLUX.2 pins every component's compute dtype to f32.
        ("flux2_klein_9b", R::exact_only(F32)),
        ("flux2_dev", R::exact_only(F32)),
        // FLUX.3's packed attention upcasts its operands to an f32 kernel dtype.
        ("flux3_klein", R::exact_only(F32)),
        ("flux3_klein_base", R::exact_only(F32)),
        ("ideogram_4", R::exact_only(HD256)),
        ("ideogram_4_turbo", R::exact_only(HD256)),
        ("iris_3b", ALL),
        // Kolors' dense UNet is candle's stock f32 attention; its packed UNet is the SDXL UNet at
        // f32.
        ("kolors", R::exact_only(F32)),
        ("krea_2_turbo", ALL),
        ("krea_2_raw", ALL),
        ("krea_2_edit", ALL),
        ("krea_2_turbo_edit", ALL),
        ("krea_2_turbo_control", ALL),
        ("lens_turbo", R::int8_only(HD64)),
        ("lens", R::int8_only(HD64)),
        // LTX's DiT attention upcasts Q/K/V to f32 before the shared helper.
        ("ltx_2_3_distilled", R::exact_only(F32)),
        ("ltx_2_5_distilled", R::exact_only(F32)),
        // Mage's DiT attention goes through `training_sdpa_bhsd` (the shared planned helper).
        ("mage_flow", ALL),
        ("mage_flow_base", ALL),
        ("mage_flow_turbo", ALL),
        ("mage_flow_edit", ALL),
        ("mage_flow_edit_base", ALL),
        ("mage_flow_edit_turbo", ALL),
        ("minimax_h3", ALL),
        ("minimax_h3_image", ALL),
        // Mochi's joint attention always carries the text padding mask.
        ("mochi_1", R::exact_only(MASKED_F32)),
        // MOSS-Image's DiT attention is its own explicit-f32 materialized path.
        ("moss_image_base", BESPOKE),
        ("moss_image_sft", BESPOKE),
        ("qwen_image", ALL),
        ("qwen_image_2_1", R::int8_only(BLOCK_CAUSAL)),
        ("qwen_image_2_1_turbo", R::int8_only(BLOCK_CAUSAL)),
        // Sana: ReLU linear self-attention plus its own masked cross-attention.
        ("sana_1600m", BESPOKE),
        ("sana_sprint_1600m", BESPOKE),
        // SCAIL2's DiT loads at f32.
        ("scail2_14b", R::exact_only(F32)),
        ("sd3_5_large", R::int8_only(HD64)),
        ("sd3_5_large_turbo", R::int8_only(HD64)),
        ("sd3_5_medium", R::int8_only(HD64)),
        ("sdxl", R::int8_only(HD64)),
        ("seedvr2", ALL),
        ("seedvr2_3b", ALL),
        ("seedvr2_7b", ALL),
        // SenseNova's block-causal attention widens to f32 activations.
        ("sensenova_u1_8b", R::exact_only(MASKED_F32)),
        ("sensenova_u1_8b_fast", R::exact_only(MASKED_F32)),
        // Supra-2's DiT has its own materialized `sdpa`.
        ("supra2_img", BESPOKE),
        // SVD's UNet runs f32 (half precision only under the `SVD_FORCE_*` diagnostics).
        ("svd_xt", R::exact_only(F32)),
        ("wan2_2_ti2v_5b", ALL),
        ("wan2_2_t2v_14b", ALL),
        ("wan2_2_i2v_14b", ALL),
        ("wan_vace", ALL),
        ("wan2_2_vace_fun_14b", ALL),
        ("z_image_turbo", ALL),
        ("z_image", ALL),
        ("iris_3b_restore", ALL),
        // The worker-owned bespoke identity routes denoise with FLUX.1 / the SDXL UNet (f16).
        (candle_gen_pulid::memory_strategy::PROVIDER_ID, ALL),
        (
            candle_gen_instantid::memory_strategy::PROVIDER_ID,
            R::int8_only(HD64),
        ),
        // The captioner and the CLIP embedders run their own attention.
        ("fancyfeast/llama-joycaption-beta-one-hf-llava", BESPOKE),
        ("clip_vit_l14", BESPOKE),
        ("clip_vit_l14_text", BESPOKE),
    ]
};

/// How `provider_id`'s model attention routes ([`SharedAttentionRouting`]); `None` for an id this
/// catalog does not register.
pub fn shared_attention_routing(provider_id: &str) -> Option<SharedAttentionRouting> {
    SHARED_ATTENTION_ROUTING
        .iter()
        .find_map(|(id, routing)| (*id == provider_id).then_some(*routing))
}

/// The one upfront check a product runs before setting a job's lossy-attention opt-ins
/// (`candle_gen::fused_attention::set_lossy_attention`) for `provider_id` (SW-6132): `Ok` when
/// nothing is requested, or when the provider's attention routes through the shared helpers, can
/// take every requested kernel ([`LossyServe::Served`]) and this build compiles it; otherwise a
/// typed [`media::gen_core::Error::Unsupported`] naming the provider and the reason — never a
/// silently exact render. Per-call constraints (mask, head dim, dtype, the device's compute
/// capability, cross-attention for Sol-Attn) are checked at each attention call and counted in
/// `AttentionPathCounts::{int8_refusals, sol_refusals}`.
pub fn admit_lossy_attention(
    provider_id: &str,
    opt_ins: &media::fused_attention::LossyAttentionOptIns,
) -> media::gen_core::Result<()> {
    use media::fused_attention::{INT8_ATTN_COMPILED, SOL_ATTN_COMPILED};
    use media::gen_core::Error;
    if !opt_ins.any() {
        return Ok(());
    }
    let requested = match (opt_ins.int8.is_some(), opt_ins.sol.is_some()) {
        (true, true) => "int8_attention + sol_attention",
        (true, false) => "int8_attention",
        _ => "sol_attention",
    };
    opt_ins
        .validate()
        .map_err(|refusal| Error::Unsupported(format!("{provider_id}: {refusal}")))?;
    let (int8, sol) = match shared_attention_routing(provider_id) {
        None => {
            return Err(Error::Unsupported(format!(
                "{requested}: `{provider_id}` is not a Candle provider this runtime registers"
            )))
        }
        Some(SharedAttentionRouting::Bespoke) => {
            return Err(Error::Unsupported(format!(
                "{requested}: `{provider_id}` runs its own attention, outside the shared Candle \
                 attention helpers the option acts on"
            )))
        }
        Some(SharedAttentionRouting::SharedHelpers { int8, sol }) => (int8, sol),
    };
    for (asked, serve, name) in [
        (opt_ins.int8.is_some(), int8, "int8_attention"),
        (opt_ins.sol.is_some(), sol, "sol_attention"),
    ] {
        if let (true, LossyServe::Never(why)) = (asked, serve) {
            return Err(Error::Unsupported(format!("{name}: `{provider_id}` {why}")));
        }
    }
    if opt_ins.int8.is_some() && !INT8_ATTN_COMPILED {
        return Err(Error::Unsupported(format!(
            "int8_attention: this build does not compile the INT8 attention kernel (`{provider_id}`)"
        )));
    }
    if opt_ins.sol.is_some() && !SOL_ATTN_COMPILED {
        return Err(Error::Unsupported(format!(
            "sol_attention: this build does not compile the Sol-Attn kernel (`{provider_id}`)"
        )));
    }
    Ok(())
}

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
    let registry = candle_gen_flux3::register_providers(registry);
    let registry = candle_gen_ideogram::register_providers(registry);
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
mod preview_advertising;

#[cfg(test)]
mod tests;
