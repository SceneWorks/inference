//! Explicit, complete provider catalog for the SceneWorks Candle media platform.
//!
//! Provider crates own their registrations; this top-level crate owns only platform composition and
//! stable ordering. Applications should construct one [`ProviderRegistry`] with [`provider_registry`]
//! and route all media loads through it.

pub use candle_gen as media;
pub use candle_gen::gen_core::{ProviderRegistry, ProviderRegistryBuilder};

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

/// Add every provider shipped by the Candle media platform to an explicit registry builder.
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
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
    const PREVIEW_ROUTE_IDS: &[&str] = &[
        "krea_2_turbo",
        "krea_2_raw",
        "krea_2_edit",
        "qwen_image",
        "anima_base",
        "anima_aesthetic",
        "anima_turbo",
    ];

    /// The routes epic 16624 **measured and rejected**, carried over into candle rather than
    /// re-measured: an RGB fit is a property of a VAE latent space, not of a backend.
    ///
    /// The temporal latent spaces missed the .88 holdout bar — LTX fit .984 / holdout .619, Mage
    /// .938 / .806, Mochi .847 / .807 — and Wan, Bernini, Scail2, SVD and SeedVR2 ride the same
    /// rejection. These are settled measurements, not open questions; sc-16961 records the full
    /// evidence and this list is its executable half. Candle must not re-run those fits.
    const PREVIEW_INERT_ROUTE_IDS: &[&str] = &[
        "wan2_2_ti2v_5b",
        "wan2_2_t2v_14b",
        "wan2_2_i2v_14b",
        "wan_vace",
        "ltx_2_3_distilled",
        "mochi_1",
        "mage_flow",
        "mage_flow_base",
        "mage_flow_turbo",
        "mage_flow_edit",
        "mage_flow_edit_base",
        "mage_flow_edit_turbo",
        "bernini_renderer",
        "bernini",
        "scail2_14b",
        "svd_xt",
        "seedvr2",
        "seedvr2_3b",
        "seedvr2_7b",
    ];

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
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-chroma",
            register: candle_gen_chroma::register_providers,
            denoise: Denoise::Shared,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-flux",
            register: candle_gen_flux::register_providers,
            denoise: Denoise::Shared,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-flux2",
            register: candle_gen_flux2::register_providers,
            denoise: Denoise::Shared,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-ideogram",
            // A bespoke `pipeline::denoise` flow-match loop, no shared-driver site anywhere —
            // sc-16955 wires it through a direct emission call, not a hook argument.
            register: candle_gen_ideogram::register_providers,
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-kolors",
            register: candle_gen_kolors::register_providers,
            denoise: Denoise::Shared,
            routes: &[],
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
            routes: &[],
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
            routes: &[],
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
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-sdxl",
            register: candle_gen_sdxl::register_providers,
            denoise: Denoise::Shared,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-seedvr2",
            register: candle_gen_seedvr2::register_providers,
            denoise: Denoise::Bespoke,
            routes: &[],
        },
        ProviderCrate {
            dir: "candle-gen-sensenova",
            // The Tier 2 family: its own VAE and a bespoke flow-match loop in `t2i.rs`, so sc-16960
            // becomes visible here through a direct emission call rather than a driver argument.
            register: candle_gen_sensenova::register_providers,
            denoise: Denoise::Bespoke,
            routes: &[],
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
            routes: &[],
        },
    ];

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
    const DIRECT_EMISSION_CALLS: &[&str] =
        &["emit_preview(", "emit_preview_at(", ".emit(", ".emit_step("];

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
                if let Some((predicate, end)) = cfg_attribute(file, &chars, i) {
                    if classify_cfg(&predicate) == CfgTest::TestOnly {
                        i = end;
                        skipping = Some((0, false));
                        skipped.clear();
                        continue;
                    }
                    // A cfg that also ships stays: emit its `#` and let the rest parse normally.
                    out.push(ch);
                    i += 1;
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

    /// Read one provider crate's preview wiring out of its **shipped** module tree.
    fn scan(provider: &ProviderCrate) -> CrateWiring {
        let src = candle_gen_root().join(provider.dir).join("src");
        assert!(
            src.is_dir(),
            "{}: no src directory — the wiring table names a crate that does not exist, so its \
             emission fact would silently read as `no`",
            src.display()
        );
        let tree = module_tree(provider.dir, &src);
        let mut wiring = CrateWiring {
            sites: Vec::new(),
            direct: Vec::new(),
            scanned: Vec::new(),
            excluded: tree.test_only.iter().cloned().collect(),
        };
        for (relative, code) in &tree.shipped {
            wiring.sites.extend(sampler_sites(relative, code));
            let count: usize = DIRECT_EMISSION_CALLS
                .iter()
                .map(|call| code.matches(call).count())
                .sum();
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
    fn declared_tallies(provider: &ProviderCrate) -> Vec<FileTally> {
        let mut tallies: Vec<FileTally> = provider
            .routes
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

    /// The half the MLX guard does not have: the allowlist above is checked against the **sources**,
    /// so it cannot be satisfied by editing lists.
    ///
    /// For every registered provider crate, whether it emits is derived from its own code — a
    /// sampler call site that passes a hook, or a bespoke loop making a direct emission call — and
    /// that fact must agree with whether any of its ids advertise. Both directions fail: a
    /// descriptor flipped ahead of the wiring, and a family wired without flipping its descriptors.
    /// The second is what makes sc-16952…sc-16960 self-enforcing.
    #[test]
    fn source_level_wiring_and_advertised_capability_agree_for_every_provider_crate() {
        let advertising = advertising_ids();
        for provider in PROVIDER_CRATES {
            let wiring = scan(provider);
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
            } else {
                assert!(
                    advertised.is_empty(),
                    "{} advertises supports_preview on {advertised:?} but nothing in its shipped \
                     sources emits: no sampler call site passes a preview hook and no bespoke loop \
                     calls {DIRECT_EMISSION_CALLS:?}",
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
        for provider in PROVIDER_CRATES {
            let wiring = scan(provider);
            let sites = wiring.sites.len();
            match provider.denoise {
                Denoise::Shared => assert!(
                    sites > 0,
                    "{} is declared Denoise::Shared but its shipped sources drive no sampler \
                     driver at all — either it moved to a bespoke loop (say so in the table) or \
                     the scan is reading the wrong files",
                    provider.dir
                ),
                Denoise::Bespoke => assert_eq!(
                    sites, 0,
                    "{} is declared Denoise::Bespoke but its shipped sources drive a shared \
                     sampler — say Denoise::Shared so its sites are inventoried",
                    provider.dir
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

    /// A wired crate must be wired **everywhere**, not just somewhere: once a crate emits at all,
    /// every one of its sampler sites has to pass a hook unless that exact site — file, driver, and
    /// occurrence index — is declared dark with a reason. That is what stops a family from wiring
    /// one route, flipping every descriptor, and leaving the rest of its routes silently blank.
    ///
    /// Declarations are checked in the other two directions too: a declared site that is no longer
    /// dark, and a dark declaration on a crate that is not wired at all, are both stale.
    #[test]
    fn a_wired_crate_leaves_no_undeclared_dark_sampler_site() {
        for provider in PROVIDER_CRATES {
            let wiring = scan(provider);
            let declared: BTreeSet<(&str, &str, usize)> = provider
                .routes
                .iter()
                .flat_map(|routes| {
                    routes
                        .dark
                        .iter()
                        .map(move |site| (routes.file, site.driver, site.index))
                })
                .collect();

            for routes in provider.routes {
                for site in routes.dark {
                    assert!(
                        !site.reason.trim().is_empty(),
                        "{}: {} {} #{} is declared dark with no reason",
                        provider.dir,
                        routes.file,
                        site.driver,
                        site.index
                    );
                }
            }

            if !wiring.emits() {
                assert!(
                    declared.is_empty(),
                    "{} declares dark sampler sites {declared:?} but emits no previews at all — a \
                     dark declaration only means something on a wired crate",
                    provider.dir
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
                "{} is wired for previews but these sampler sites still pass `None`: {undeclared:?} \
                 — pass a hook, or declare that exact site in its file's `dark` list with the \
                 reason it emits nothing",
                provider.dir
            );

            for (file, driver, index) in &declared {
                assert!(
                    wiring.sites.iter().any(|site| {
                        site.file == *file
                            && site.driver == *driver
                            && site.index == *index
                            && !site.hooked
                    }),
                    "{}: {file} declares {driver} #{index} dark, but that site no longer passes \
                     `None` — remove the stale declaration",
                    provider.dir
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
        for provider in PROVIDER_CRATES {
            let wiring = scan(provider);
            if !wiring.emits() {
                assert!(
                    provider.routes.is_empty(),
                    "{} pins a route inventory but emits no previews — an inventory only means \
                     something on a wired crate",
                    provider.dir
                );
                continue;
            }
            wired += 1;
            hooked_sites += wiring.sites.iter().filter(|site| site.hooked).count();
            assert_eq!(
                derived_tallies(&wiring),
                declared_tallies(provider),
                "{}: the route inventory in PROVIDER_CRATES disagrees with the crate's sources. \
                 Every file that drives a sampler or emits directly needs a row with exact counts \
                 — blanking one route of an already-inventoried file must be a diff here too.",
                provider.dir
            );
        }
        assert!(
            wired > 0 && hooked_sites > 0,
            "no crate resolved as wired ({wired} crates, {hooked_sites} hooked sites) — every \
             assertion in this module would then be vacuously satisfied by a scanner that read \
             nothing"
        );
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
            let provider = PROVIDER_CRATES
                .iter()
                .find(|provider| provider.dir == *dir)
                .unwrap_or_else(|| panic!("{dir} is in the wiring table"));
            let wiring = scan(provider);
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
        let krea = PROVIDER_CRATES
            .iter()
            .find(|provider| provider.dir == "candle-gen-krea")
            .expect("Krea is in the wiring table");
        let wiring = scan(krea);

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

    /// The carried-over no-go set stays outside advertising, by exact id, and stays out of the
    /// allowlist. The reason is a settled measurement (see `PREVIEW_INERT_ROUTE_IDS`), not an open
    /// question — candle must not re-run those fits.
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

        for id in PREVIEW_INERT_ROUTE_IDS {
            let descriptor = descriptors
                .iter()
                .find(|descriptor| descriptor.id == *id)
                .unwrap_or_else(|| panic!("{id} must remain a registered candle generator"));
            assert!(
                !descriptor.capabilities.supports_preview,
                "{id} is in the epic-16624 holdout rejection set and must not advertise previews"
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
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_registered_memory_strategy_rejects_cross_route_decode_geometry() {
        let registry = super::provider_registry().unwrap();
        let spec = candle_gen::gen_core::LoadSpec::new(candle_gen::gen_core::WeightsSource::Dir(
            "/nonexistent".into(),
        ));
        gen_core_testkit::memory_strategy_registry_conformance(&registry, &spec);
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
            assert!(
                !capabilities.supports_true_cfg,
                "{id} has no Candle reference/image-CFG path"
            );
            assert!(
                capabilities.conditioning.is_empty(),
                "{id} is Candle txt2img only"
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
                "lens_turbo",
                "lens",
                "ltx_2_3_distilled",
                "mage_flow",
                "mage_flow_base",
                "mage_flow_turbo",
                "mage_flow_edit",
                "mage_flow_edit_base",
                "mage_flow_edit_turbo",
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
                "z_image_turbo",
                "z_image",
            ]
        );
        assert_eq!(
            trainers,
            [
                "krea_2_raw",
                "krea_2_control",
                "lens",
                "ltx_2_3",
                "sdxl",
                "wan2_2_t2v_14b",
                "z_image_turbo",
            ]
        );
        assert_eq!(
            captioners,
            ["fancyfeast/llama-joycaption-beta-one-hf-llava"]
        );
        assert_eq!(image_embedders, ["clip_vit_l14"]);
        assert_eq!(text_embedders, ["clip_vit_l14_text"]);
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
