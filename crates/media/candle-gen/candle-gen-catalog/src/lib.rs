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
/// out of that crate's own sources: it emits if some call to a shared sampler driver
/// (`candle_gen::run_flow_sampler` / `run_curated_sampler` / `run_scm_sampler`) passes a preview
/// hook where it could pass `None`, or if a bespoke denoise loop calls `candle_gen::preview`'s
/// `emit_preview` / `emit_preview_at` directly. Neither fact can be produced by editing an
/// allowlist, so the two halves cannot be made to agree by editing one place.
///
/// The source-level scan follows the sc-16950 route inventory in `candle-gen-krea/src/preview.rs`,
/// widened from one crate and one driver to every registered crate and all three drivers.
///
/// ## The amendment protocol — read this before adding a family
///
/// This guard lands once, holding Krea alone, and is then **amended by each later family story in
/// that story's own PR**: sc-16952…sc-16960 each wire a family, add its exact route ids to
/// `PREVIEW_ROUTE_IDS`, and flip its descriptors. Do not file a follow-up story for the amendment.
/// The derived half is what makes that self-enforcing in the direction that matters: the moment a
/// family's sources pass a hook, its descriptors must advertise, and the build fails until they do.
#[cfg(test)]
mod preview_advertising {
    use std::collections::BTreeSet;
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
    /// Krea alone today: sc-16950 is the only family wiring merged.
    const PREVIEW_ROUTE_IDS: &[&str] = &["krea_2_turbo", "krea_2_raw", "krea_2_edit"];

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

    /// One registered provider crate: where its sources live, how to ask it for its own ids, and
    /// which of its files are allowed to drive a sampler with no preview hook.
    struct ProviderCrate {
        /// Directory name under `crates/media/candle-gen`.
        dir: &'static str,
        /// The crate's own registration function. Its ids are read back out of a registry built
        /// from it alone, so this table never restates an id and cannot drift from one.
        register: fn(ProviderRegistryBuilder) -> ProviderRegistryBuilder,
        /// `src`-relative files whose sampler sites deliberately pass no hook, once the crate is
        /// wired at all. Empty for an unwired crate — `a_wired_crate_leaves_no_undeclared_dark_sampler_site`
        /// rejects a declaration on a crate that emits nothing, and a declaration that has gone stale.
        dark_files: &'static [&'static str],
    }

    /// Every crate `register_providers` composes that ships a generator. `clip` and `joycaption`
    /// register no generator, so they contribute no `supports_preview` surface and are omitted;
    /// `every_shipped_generator_is_covered_by_the_wiring_table` fails if that ever stops being true,
    /// or if a new provider crate joins the catalog without joining this table.
    const PROVIDER_CRATES: &[ProviderCrate] = &[
        ProviderCrate {
            dir: "candle-gen-anima",
            register: candle_gen_anima::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-bernini",
            register: candle_gen_bernini::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-boogu",
            register: candle_gen_boogu::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-chroma",
            register: candle_gen_chroma::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-flux",
            register: candle_gen_flux::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-flux2",
            register: candle_gen_flux2::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-ideogram",
            register: candle_gen_ideogram::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-kolors",
            register: candle_gen_kolors::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-krea",
            // The trainer's periodic sample render drives the sampler from a synthetic request that
            // carries no `PreviewSink`, so it passes `None` on purpose (sc-16950 pins that as a
            // decision in the Krea crate's own inventory). Every other Krea site is hooked.
            register: candle_gen_krea::register_providers,
            dark_files: &["training.rs"],
        },
        ProviderCrate {
            dir: "candle-gen-lens",
            register: candle_gen_lens::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-ltx",
            register: candle_gen_ltx::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-mage",
            register: candle_gen_mage::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-mochi",
            register: candle_gen_mochi::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-qwen-image",
            register: candle_gen_qwen_image::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-sana",
            register: candle_gen_sana::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-scail2",
            register: candle_gen_scail2::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-sd3",
            register: candle_gen_sd3::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-sdxl",
            register: candle_gen_sdxl::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-seedvr2",
            register: candle_gen_seedvr2::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-sensenova",
            register: candle_gen_sensenova::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-svd",
            register: candle_gen_svd::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-wan",
            register: candle_gen_wan::register_providers,
            dark_files: &[],
        },
        ProviderCrate {
            dir: "candle-gen-z-image",
            register: candle_gen_z_image::register_providers,
            dark_files: &[],
        },
    ];

    /// The shared sampler drivers and the **0-based position of their `preview` argument**. The
    /// position is read from the front, so a family that passes a named `predict` instead of an
    /// inline closure parses the same way; `sampler_arity_is_pinned_by_classification` explains why
    /// a signature change cannot silently shift it.
    const SAMPLER_DRIVERS: &[(&str, usize)] = &[
        ("run_flow_sampler(", 7),
        ("run_curated_sampler(", 7),
        ("run_scm_sampler(", 5),
    ];

    /// A bespoke denoise loop emits by calling the shared preview module directly rather than
    /// through a driver — SenseNova-U1 and Ideogram own their loops, so this is how their wiring
    /// will become visible to this guard.
    const DIRECT_EMISSION_CALLS: &[&str] = &["emit_preview(", "emit_preview_at("];

    /// One shared-sampler call site found in a provider crate's sources.
    struct SamplerSite {
        /// `src`-relative path, `/`-separated on every platform.
        file: String,
        driver: &'static str,
        index: usize,
        /// Whether its `preview` argument is anything other than the literal `None`.
        hooked: bool,
    }

    /// What a crate's sources say about preview emission.
    struct CrateWiring {
        sites: Vec<SamplerSite>,
        /// `src`-relative files calling `emit_preview` / `emit_preview_at` directly.
        direct: Vec<String>,
        files_scanned: usize,
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

    /// Rust source with comments, string / char literals, and `#[cfg(test)]` items removed.
    ///
    /// Test modules have to go before anything else looks at the text: `candle-gen-krea`'s own
    /// preview tests call `run_flow_sampler` from a helper and quote the driver's name in a string
    /// literal and in prose, so a scan of the raw file would find call sites that do not exist in
    /// the shipped route and would mis-parse the ones that do.
    fn code_only(file: &str, source: &str) -> String {
        let chars: Vec<char> = source.chars().collect();
        let mut out = String::with_capacity(source.len());
        let mut i = 0usize;
        // `Some((bracket depth, has opened its top-level block))` while consuming the item a
        // `#[cfg(test)]` attribute applies to.
        let mut skipping: Option<(i32, bool)> = None;

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
            if skipping.is_none() && matches_at(&chars, i, CFG_TEST) {
                i += CFG_TEST.chars().count();
                skipping = Some((0, false));
                continue;
            }

            if let Some((depth, entered)) = skipping.as_mut() {
                match ch {
                    '(' | '[' | '{' => {
                        *depth += 1;
                        if ch == '{' && *depth == 1 {
                            *entered = true;
                        }
                    }
                    ')' | ']' | '}' => {
                        *depth -= 1;
                        assert!(*depth >= 0, "{file}: unbalanced #[cfg(test)] item");
                        if *depth == 0 && *entered {
                            skipping = None;
                        }
                    }
                    // A `#[cfg(test)]` item is not always a block or a statement: it is also used
                    // on a single struct field, struct-literal field, or enum variant, all of which
                    // end at a comma rather than a semicolon or a brace.
                    ';' | ',' if *depth == 0 => skipping = None,
                    _ => {}
                }
                i += 1;
                continue;
            }

            out.push(ch);
            i += 1;
        }

        assert!(
            skipping.is_none(),
            "{file}: a #[cfg(test)] item never closed"
        );
        // A `#[cfg(test)]` spelled any other way (`#[cfg(all(test, …))]`) survives the strip above.
        // Fail loudly rather than scan a test module as if it were shipped code.
        assert!(
            !out.contains("cfg(test"),
            "{file}: an unrecognised cfg(test) form survived the strip — teach `code_only` about it"
        );
        out
    }

    const CFG_TEST: &str = "#[cfg(test)]";

    fn matches_at(chars: &[char], at: usize, needle: &str) -> bool {
        needle
            .chars()
            .enumerate()
            .all(|(offset, c)| chars.get(at + offset) == Some(&c))
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
        for (driver, preview_at) in SAMPLER_DRIVERS {
            let mut cursor = 0usize;
            let mut index = 0usize;
            while let Some(offset) = code[cursor..].find(driver) {
                let args_start = cursor + offset + driver.len();
                let site = format!("{file}: {driver} call #{index}");
                let args = call_arguments(&site, &code[args_start..]);
                // The preview argument is the last one before an inline closure, or the
                // second-to-last when the closure is passed by name. Anything else means the
                // driver's signature moved and the position below no longer names what it says.
                assert!(
                    args.len() == preview_at + 1 || args.len() == preview_at + 2,
                    "{site}: expected the preview argument at position {preview_at} of \
                     {} or {} arguments, parsed {args:?}",
                    preview_at + 1,
                    preview_at + 2
                );
                let argument = args[*preview_at].as_str();
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
                    driver,
                    index,
                    hooked,
                });
                cursor = args_start;
                index += 1;
            }
        }
        sites
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

    /// Read one provider crate's preview wiring out of its `src` tree.
    fn scan(provider: &ProviderCrate) -> CrateWiring {
        let src = candle_gen_root().join(provider.dir).join("src");
        assert!(
            src.is_dir(),
            "{}: no src directory — the wiring table names a crate that does not exist, so its \
             emission fact would silently read as `no`",
            src.display()
        );
        let mut wiring = CrateWiring {
            sites: Vec::new(),
            direct: Vec::new(),
            files_scanned: 0,
        };
        for path in rust_sources(&src) {
            let relative = path
                .strip_prefix(&src)
                .expect("walked from src")
                .to_string_lossy()
                .replace('\\', "/");
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let code = code_only(&relative, &source);
            wiring.sites.extend(sampler_sites(&relative, &code));
            if DIRECT_EMISSION_CALLS.iter().any(|call| code.contains(call)) {
                wiring.direct.push(relative);
            }
            wiring.files_scanned += 1;
        }
        assert!(
            wiring.files_scanned > 0,
            "{}: no Rust sources found — an empty scan reads as `does not emit` and would make \
             every assertion below vacuous",
            src.display()
        );
        wiring
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
    /// sampler call site that passes a hook, or a bespoke loop calling `emit_preview` — and that
    /// fact must agree with whether any of its ids advertise. Both directions fail: a descriptor
    /// flipped ahead of the wiring, and a family wired without flipping its descriptors. The second
    /// is what makes sc-16952…sc-16960 self-enforcing.
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
                assert!(
                    !advertised.is_empty(),
                    "{} emits previews (hooked sites: {hooked:?}, direct emission: {:?}) but none \
                     of its routes {ids:?} advertise supports_preview — flip the descriptors and \
                     add the ids to PREVIEW_ROUTE_IDS in the same PR",
                    provider.dir,
                    wiring.direct
                );
            } else {
                assert!(
                    advertised.is_empty(),
                    "{} advertises supports_preview on {advertised:?} but nothing in its sources \
                     emits: no sampler call site passes a preview hook and no bespoke loop calls \
                     emit_preview",
                    provider.dir
                );
            }
        }
    }

    /// A wired crate must be wired **everywhere**, not just somewhere: once a crate emits at all,
    /// every one of its sampler sites has to pass a hook unless its file is declared dark with a
    /// reason. That is what stops a family from wiring one route, flipping every descriptor, and
    /// leaving the rest of its routes silently blank.
    ///
    /// Declarations are checked in the other two directions too — a dark file with no dark site
    /// left in it, and a dark declaration on a crate that is not wired at all, are both stale.
    #[test]
    fn a_wired_crate_leaves_no_undeclared_dark_sampler_site() {
        for provider in PROVIDER_CRATES {
            let wiring = scan(provider);
            let declared: BTreeSet<&str> = provider.dark_files.iter().copied().collect();

            if !wiring.emits() {
                assert!(
                    declared.is_empty(),
                    "{} declares dark sampler files {declared:?} but emits no previews at all — a \
                     dark declaration only means something on a wired crate",
                    provider.dir
                );
                continue;
            }

            let undeclared: Vec<String> = wiring
                .sites
                .iter()
                .filter(|site| !site.hooked && !declared.contains(site.file.as_str()))
                .map(|site| format!("{}: {} #{}", site.file, site.driver, site.index))
                .collect();
            assert!(
                undeclared.is_empty(),
                "{} is wired for previews but these sampler sites still pass `None`: {undeclared:?} \
                 — pass a hook, or declare the file in `dark_files` with the reason it emits nothing",
                provider.dir
            );

            for file in &declared {
                assert!(
                    wiring
                        .sites
                        .iter()
                        .any(|site| site.file == *file && !site.hooked),
                    "{}: dark_files names {file}, which no longer drives a sampler without a hook \
                     — remove the stale declaration",
                    provider.dir
                );
            }
        }
    }

    /// Positive evidence that the scan is not vacuous.
    ///
    /// Everything above is an equality between a derived fact and a declared one, and a scanner that
    /// silently resolved nothing would satisfy most of it: no sites found reads as "does not emit",
    /// which agrees with every unwired crate. So pin what the scan actually resolves for the one
    /// wired crate — the eight hooked Krea render sites and the one deliberately dark trainer site,
    /// the same inventory `candle-gen-krea` pins against its own sources in sc-16950. If this
    /// disagrees with that, one of the two scanners is reading the crate wrong.
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
