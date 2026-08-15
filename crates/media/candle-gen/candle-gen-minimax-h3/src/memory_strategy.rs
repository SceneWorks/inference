//! MiniMax-H3's shared-ladder `MemoryProviderContract` on Candle/CUDA (sc-18659).
//!
//! The sibling declaration to `mlx_gen_minimax_h3::memory_strategy`. The contract carries a
//! [`MemoryBackendRealization`], so the two backends are *allowed* to differ — and here they
//! genuinely do, in three ways that are worth stating before the code:
//!
//! 1. **No fitted curve.** Candle has only the floor arm, so the formula is
//!    [`MemoryFormulaKind::AssetBytesPlusHeadroom`] rather than MLX's phase envelope, and
//!    [`MemoryProviderContract::calibration`] is `None`. A provider with no calibration identity
//!    can run its resident path and can never claim a verified optimized fit — which is exactly
//!    the truth here, and is enforced: `standard_memory_strategy_safety_check` refuses any
//!    optimized selection when the identity is absent.
//! 2. **Staged residency is implemented but under-declared (sc-17156 / sc-18660).** MLX declares
//!    [`MemoryStrategy::StagedResidency`] `Implemented`. This lane now *does the same thing* —
//!    `MiniMaxH3::generate_impl` releases each heavy component before mapping the next, and the
//!    provider **forces** `OffloadPolicy::Sequential` so a caller cannot opt out — but the rung
//!    stays declared `Missing` until its executable behavior seam lands, and that seam is itself
//!    gated on a `calibration` identity this backend does not have. An implemented optimized rung
//!    without a seam fails catalog conformance, and under-declaring is the safe direction: the
//!    staging still happens, and nothing is admitted on an unexercisable claim.
//! 3. **No fused streaming SDPA.** The MLX verdict that attention scratch is already streamed —
//!    peak tracking `4·B·H·S·D` with no materialized score tensor — is a property of *MLX's* fused
//!    kernel and **must not be copied here**. Candle materializes scores, so bounding attention
//!    (sc-18661) may buy real memory on this backend even if it buys none on MLX.
//!
//! # Why the asset facts are the full four components anyway
//!
//! `conditioning_bytes` charges the 66.71 GB Qwen3-VL-32B text encoder even though this crate
//! cannot yet execute it. Asset facts are the render's byte floor, not a capability claim: a
//! candle render of this family needs the conditioner, and a contract that declared zero there
//! would publish a floor small enough to admit a request that cannot possibly run. Capability
//! lives in `strategies`, where every optimized rung is honestly `Missing`.
//!
//! # Stage attribution
//!
//! The ~53 GB memory floor measured for this family is the **conditioning** stage — the dense text
//! encoder in isolation — not the DiT and not activation pressure. The DiT's own denoise-resident
//! cost is a separate, genuinely tiered quantity. Those measurements were taken on MLX; sc-17156
//! landed this backend's pipeline but no fitted curve of its own, which is the other reason
//! `calibration` is `None`.

use candle_gen::gen_core::{
    safetensors_path_bytes, LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryFormulaKind, MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryProviderContract,
    MemoryRunContext, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategySupport, MemoryWindowMaterialization, ResidentRequestMemory, WeightsSource,
};

use crate::MODEL_ID;

// --- measured asset facts -------------------------------------------------------------------
//
// Exact `.safetensors` bytes under each component directory of the upstream bf16 snapshot
// (`MiniMaxAI/MiniMax-H3` @ `939557dc`). Identical to the MLX sibling's, deliberately: these are
// facts about the checkpoint, not about a backend, and the two must not drift apart.

/// Qwen3-VL-32B text encoder — 14 shards, the conditioning component. 66.71 GB.
pub const TEXT_ENCODER_BYTES: u64 = 66_714_912_872;

/// One 33 B DiT partition at bf16 — 14 shards. 66.28 GB. A render loads exactly one of
/// `transformer` / `transformer_ref`, so it is charged once.
pub const DIT_BF16_BYTES: u64 = 66_280_504_216;

/// Video VAE — 3 shards; the decoder is a 36-layer transformer. 10.42 GB.
pub const VIDEO_VAE_BYTES: u64 = 10_415_558_888;

/// Audio VAE — 1 shard. 0.61 GB.
pub const AUDIO_VAE_BYTES: u64 = 605_429_340;

/// Exact bytes the AdaLN precompute-and-evict drops, asserted against the loader in
/// [`crate::dit::adaln`]. Declared here so both backends' contracts carry the same figure; sc-18665
/// turns it into a typed resident-component exclusion the ladder can see.
pub const ADALN_EVICTED_BYTES: u64 = 26_020_915_200;

/// The load shape this loader actually has, pinned rather than mirrored from the spec.
///
/// [`LoadShape::DeferredMaterialization`] means transformer blocks are materialized through a block
/// schedule. `MiniMaxH3Dit::load` builds the whole stack, so this provider is
/// [`LoadShape::EagerMaterialization`] whatever a caller asks for. sc-18662 changes it.
pub const LOAD_SHAPE: LoadShape = LoadShape::EagerMaterialization;

/// The DiT component directory a flat snapshot carries.
const DIT_COMPONENT: &str = "transformer";

struct ComponentBytes {
    text_encoder: u64,
    dit: u64,
    video_vae: u64,
    audio_vae: u64,
}

impl ComponentBytes {
    fn resolve(spec: &LoadSpec) -> Self {
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root.clone(),
            WeightsSource::File(path) => path.parent().unwrap_or(path).to_path_buf(),
        };
        let dit = match spec.components.get(DIT_COMPONENT) {
            Some(WeightsSource::Dir(staged)) => staged.clone(),
            _ => root.join(DIT_COMPONENT),
        };
        Self {
            text_encoder: safetensors_path_bytes(root.join("text_encoder")),
            dit: safetensors_path_bytes(dit),
            video_vae: safetensors_path_bytes(root.join("vae")),
            audio_vae: safetensors_path_bytes(root.join("audio_vae")),
        }
    }

    /// The two decoders are one contract field; H3 is the first family with two of them.
    fn decoder(&self) -> u64 {
        self.video_vae.saturating_add(self.audio_vae)
    }

    fn base(&self) -> u64 {
        self.text_encoder
            .saturating_add(self.dit)
            .saturating_add(self.decoder())
    }
}

/// The five capability entries. **Exactly one is `Implemented` — rung 0, `Resident`. Rungs 1-4 are
/// all `Missing`**, which is what the `match` below returns and what the ladder reads.
///
/// Rung 1 is the one that needs stating carefully, because the pipeline and the declaration disagree
/// **on purpose**. sc-17156 made the staging real — `MiniMaxH3::generate_impl` drops each heavy
/// component and synchronizes the device before mapping the next, and `MiniMaxH3` forces
/// [`OffloadPolicy::Sequential`](candle_gen::gen_core::OffloadPolicy) whatever the caller asks for,
/// so a caller cannot opt out. The *declaration* deliberately did not follow: the rung stays
/// `Missing` until its executable weights-free behavior seam lands, and that seam is gated on a
/// `calibration` identity this crate does not have (sc-18660 pinned that chain in a test). The long
/// note on the arm below gives the full reasoning, and the module header says the same thing.
///
/// So "implemented in the pipeline" and "declared `Implemented`" are two different facts here, and
/// only the first is true. Reading this as rung 1 being available to the ladder would be wrong in
/// the direction that matters: nothing is admitted on the strength of it.
///
/// Every entry publishes an empty [`MemoryParameterRanges`], which is correct in both directions:
/// rung 0 owns no numeric parameters, and a `Missing` rung must not publish a domain it cannot
/// honor. Flipping any of rungs 1-4 to `Implemented` without filling its domain is a conformance
/// error, not a silent under-declaration.
fn strategies() -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
                // Rung 1 is **implemented in code** as of sc-17156 — `MiniMaxH3::generate_impl`
                // stages all three phases and the offload policy is forced — but it stays declared
                // `Missing` here, and that is a deliberate under-declaration, not an oversight.
                //
                // `check_memory_strategy_contract` requires every *implemented optimized* rung to
                // carry an executable weights-free **behavior seam** (a `MemoryBehaviorRegistration`
                // with per-route fixtures, the MLX sibling's `MEMORY_BEHAVIOR`). Declaring the rung
                // without it fails catalog conformance — correctly: an optimized declaration with no
                // seam is a capability claim nothing can execute against.
                //
                // Under-declaring is the safe direction. `Missing` means the ladder never *selects*
                // staged residency for this provider, so the staging still happens (it is forced)
                // and nothing is admitted on the strength of an unproven declaration. The reverse —
                // declaring it and having the seam absent — would let admission believe a rung it
                // cannot exercise.
                //
                // Rung 2 (sc-18660) is `Missing` for the very same reason, and it too is **not** a
                // porting gap: the mechanism is present — `MiniMaxH3VideoVae::decode_clip` tiles at
                // the reference 256/64 through `crate::spatial_tiling::BoundedStitch`, the same
                // seam the MLX sibling declares `Implemented` on. What this backend lacks is one
                // link further back than the seam: a weights-free `MemoryBehaviorRegistration`
                // needs a `MemoryBehaviorFixture`, and that needs a `calibration` identity this
                // crate does not have (see `contract.calibration`). So the blocker for **both**
                // rungs is the absent calibration identity — not the seam, and not the mechanism.
                // A seam cannot be written before the identity exists.
                // `an_optimized_rung_here_is_blocked_by_the_missing_calibration_identity_not_the_\
                // seam` drives that chain and fails the moment it is removed.
                //
                // Rungs 3/4: sc-18661 / sc-18662.
                _ => MemoryStrategySupport::Missing,
            },
            parameters: MemoryParameterRanges::default(),
        })
        .collect()
}

fn build_contract(components: &ComponentBytes) -> MemoryProviderContract {
    MemoryProviderContract {
        provider_id: MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            // There is no block-wise host→device materialization path in this crate at all: the
            // DiT loads as a whole stack. sc-18662 builds one.
            host_to_device_block_materialization: false,
            // Answered even though rung 4 is `Missing`, because the field is deliberately not
            // optional. It is accurate for the loader as it exists: this crate has no packed tier
            // and therefore no MLX-affine → GGML repack seam, so a window would be a mapped read
            // plus a host-to-device copy of bytes already in the accelerator's form. **sc-18662
            // must re-verify this** if a packed candle tier ever lands — that is the change that
            // turns a conforming realization into a `HostFormatConversion` one.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: strategies(),
        pid_decode_routes: None,
        load_shape: LOAD_SHAPE,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        // Empty, and paired with the `Missing` rung-1 declaration above for the same reason: the
        // phases ARE implemented (`generate_impl` releases each component and synchronizes the
        // device via `crate::dit::release_device_memory`), but the rung they belong to cannot be
        // declared until its behavior seam exists, and a lifecycle block attached to a `Missing`
        // rung is a claim with no rung to hang on. The seam waits on a `calibration` identity, so
        // the rung and this block flip together when one lands.
        lifecycle: MemoryLifecycleCapabilities::default(),
        // The floor arm, and only the floor arm.
        formula: MemoryFormulaKind::AssetBytesPlusHeadroom,
        // No fitted curve exists for this backend. `None` is the honest state, and it is load
        // bearing: it makes every optimized selection fail closed at admission.
        calibration: None,
        asset_facts: MemoryAssetFacts {
            base_bytes: components.base(),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.decoder(),
            overlay_bytes: 0,
        },
        runtime: Default::default(),
    }
}

/// The production contract: asset facts read off the resolved snapshot.
pub fn contract_for(spec: &LoadSpec) -> candle_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(&ComponentBytes::resolve(spec)))
}

/// The weights-free fixture contract: the identical route declaration with zero asset facts and no
/// filesystem traversal.
pub fn weights_free_contract(
    _spec: &LoadSpec,
) -> candle_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(&ComponentBytes {
        text_encoder: 0,
        dit: 0,
        video_vae: 0,
        audio_vae: 0,
    }))
}

/// The provider's real admission check, callable before any weight file is opened.
///
/// The shared check is sufficient here and a route gate would be a lie: with no optimized rung and
/// no calibration identity, every admission this provider can accept is the resident baseline, and
/// the geometry gates belong to the pipeline sc-17156 landed (`crate::denoise::geometry`), not to
/// this contract.
pub fn safety_check(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    candle_gen::gen_core::default_memory_strategy_safety_check(contract, context)
}

/// The memory-strategy registration for `minimax_h3` on candle.
///
/// **Reachable from `candle-gen-catalog` as of sc-17156.** It was not before, and that was
/// deliberate rather than an oversight: `ProviderRegistryBuilder::build` rejects a memory-strategy
/// registration whose `provider_id` has no matching generator, and this crate shipped no generator.
/// Wiring it through `register_composed_memory_strategy` would have satisfied the builder by
/// declaring a composition root that did not exist — the exact "provider contract with no
/// executable owner" that seam exists to prevent. The generator landed with sc-17156, so the
/// catalog line landed with it.
///
/// That coupling was a guard rather than a hope, and it still holds in the other direction: this
/// constant is registered by [`crate::register_providers`], and
/// `tests::a_generator_landing_here_forces_the_catalog_line` builds *that* inventory, so a
/// generator registered around it — leaving this registration behind — fails there.
pub const MEMORY_REGISTRATION: candle_gen::gen_core::MemoryRegistration =
    candle_gen::gen_core::MemoryRegistration {
        provider_id: MODEL_ID,
        contract: contract_for,
        safety_check,
    };

/// The weights-free contract fixture paired with [`MEMORY_REGISTRATION`].
pub const MEMORY_CONTRACT_FIXTURE: candle_gen::gen_core::MemoryContractFixtureRegistration =
    candle_gen::gen_core::MemoryContractFixtureRegistration {
        provider_id: MODEL_ID,
        contract: weights_free_contract,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        LoadShape, MemoryBudget, MemoryCacheState, MemoryCalibrationIdentity, MemoryGeometry,
        MemoryMode, MemoryNumericTier, MemoryPhase, MemoryRunContext, MemorySelection,
        MemoryStrategyParameters, ProviderRegistryBuilder,
    };
    use std::path::Path;

    /// One named, independently applied mutation of a known-good contract.
    type ContractMutation = (&'static str, Box<dyn Fn(&mut MemoryProviderContract)>);

    fn weightless_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    fn declared() -> MemoryProviderContract {
        weights_free_contract(&weightless_spec()).expect("weights-free contract")
    }

    fn candle_backend() -> MemoryBackendRealization {
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        }
    }

    /// Sparse `.safetensors` shards of exact sizes. `safetensors_path_bytes` stats rather than
    /// parses, so this costs no disk and still exercises the real directory-name wiring.
    fn sparse_snapshot(root: &Path, sizes: &[(&str, u64)]) {
        for (component, bytes) in sizes {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).expect("component dir");
            let file = std::fs::File::create(dir.join("model.safetensors")).expect("shard");
            file.set_len(*bytes).expect("sparse shard");
        }
    }

    fn full_snapshot(root: &Path) {
        sparse_snapshot(
            root,
            &[
                ("text_encoder", TEXT_ENCODER_BYTES),
                ("transformer", DIT_BF16_BYTES),
                ("vae", VIDEO_VAE_BYTES),
                ("audio_vae", AUDIO_VAE_BYTES),
                // Byte-identical to `transformer`; a render loads exactly one, so it must not be
                // charged on top.
                ("transformer_ref", DIT_BF16_BYTES),
            ],
        );
    }

    // --- AC1: declared Resident, not fallen-back Resident ---------------------------------------

    /// **The honest state of this backend is resident-only**, so the strategy table alone cannot
    /// distinguish this declaration from `compatibility_default` — asserting on it would be the
    /// false green AC1 warns about. The distinguisher is the one that actually matters to a
    /// consumer: a fallback contract publishes a **zero** byte floor for a 144 GB family, and this
    /// one publishes the measured components split across the fields the formula reads.
    #[test]
    fn resolved_contract_is_declared_and_not_the_compatibility_default() {
        let root = tempfile::tempdir().expect("tempdir");
        full_snapshot(root.path());
        let contract =
            (MEMORY_REGISTRATION.contract)(&LoadSpec::new(WeightsSource::Dir(root.path().into())))
                .expect("registered contract");

        let fallback = MemoryProviderContract::compatibility_default(MODEL_ID, candle_backend());
        assert_ne!(contract, fallback);
        assert_eq!(
            fallback.asset_facts.base_bytes, 0,
            "the fallback publishes a zero floor — that is what makes it unsafe here"
        );
        assert_eq!(
            contract.asset_facts.base_bytes,
            TEXT_ENCODER_BYTES + DIT_BF16_BYTES + VIDEO_VAE_BYTES + AUDIO_VAE_BYTES
        );
        assert_eq!(contract.asset_facts.conditioning_bytes, TEXT_ENCODER_BYTES);
        assert_eq!(contract.asset_facts.transformer_bytes, DIT_BF16_BYTES);
        assert_eq!(
            contract.asset_facts.decoder_bytes,
            VIDEO_VAE_BYTES + AUDIO_VAE_BYTES,
            "both decoders are charged, and only the decoders"
        );
        assert!(contract.conformance_errors().is_empty());
    }

    /// A misspelled provider id would produce a contract for a family that does not exist. The two
    /// registration constants and the contract must all agree on one id.
    #[test]
    fn every_declaration_agrees_on_one_provider_id() {
        assert_eq!(MODEL_ID, "minimax_h3");
        assert_eq!(declared().provider_id, MODEL_ID);
        assert_eq!(MEMORY_REGISTRATION.provider_id, MODEL_ID);
        assert_eq!(MEMORY_CONTRACT_FIXTURE.provider_id, MODEL_ID);
        assert_eq!(
            (MEMORY_CONTRACT_FIXTURE.contract)(&weightless_spec())
                .expect("fixture")
                .provider_id,
            MEMORY_REGISTRATION.provider_id
        );
    }

    /// This crate's package name, as `cargo metadata` reports it.
    const THIS_CRATE: &str = "candle-gen-minimax-h3";

    /// The catalog crate whose dependency edge on this one is the tripwire.
    const CATALOG_CRATE: &str = "candle-gen-catalog";

    /// Does `candle-gen-catalog` declare a dependency on this crate?
    ///
    /// Asked of `cargo metadata --no-deps --offline`, which is Cargo's own parse of the workspace
    /// manifests. Every workspace member reports its declared dependencies with each dependency's
    /// **package** name already resolved, so renames, quoting style, dotted keys,
    /// `workspace = true` inheritance and the `dev-`/`build-`/`target.'cfg(…)'` sections all arrive
    /// in the same `name` field and fall to one comparison. Lexing the TOML by hand instead is what
    /// let four spellings of the documented rename route go green (sc-18659).
    ///
    /// `--no-deps` scopes the answer to what the catalog *itself* declares: a route that reaches
    /// this crate transitively, through an intermediate crate that depends on it, is not covered.
    fn catalog_depends_on_this_crate() -> bool {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(4)
            .expect("candle-gen-minimax-h3 sits four levels under the workspace root");
        let output = std::process::Command::new(env!("CARGO"))
            .args([
                "metadata",
                "--no-deps",
                "--offline",
                "--format-version",
                "1",
                "--manifest-path",
            ])
            .arg(root.join("Cargo.toml"))
            .output()
            .unwrap_or_else(|error| panic!("cargo metadata: {error}"));
        assert!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("cargo metadata emits JSON");
        let catalog = metadata["packages"]
            .as_array()
            .expect("cargo metadata reports a `packages` array")
            .iter()
            .find(|package| package["name"] == CATALOG_CRATE)
            .unwrap_or_else(|| {
                panic!("{CATALOG_CRATE} is not a workspace member — this check would answer `false` for the wrong reason")
            });
        catalog["dependencies"]
            .as_array()
            .expect("a package reports a `dependencies` array")
            .iter()
            .any(|dependency| dependency["name"] == THIS_CRATE)
    }

    /// Every `.rs` under `candle-gen-catalog/src`, at any depth, concatenated — used only to check
    /// that the registration *call* landed, never to decide reachability. Reachability is
    /// [`catalog_depends_on_this_crate`]; scanning sources for that would reintroduce the bypass.
    ///
    /// The walk recurses because the call may land in `src/<subdir>/*.rs`, which a top-level-only
    /// `read_dir` would miss, making `wired` false on the `Ok` arm even though the call landed.
    fn catalog_sources() -> String {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("candle-gen-minimax-h3 sits inside crates/media/candle-gen")
            .join("candle-gen-catalog/src");
        let mut sources = Vec::new();
        push_rs_sources(&dir, &mut sources);
        assert!(
            !sources.is_empty(),
            "{} holds no .rs file — the source scan would vacuously pass",
            dir.display()
        );
        sources.join("\n")
    }

    /// Appends the text of every `.rs` file at or below `dir`.
    ///
    /// Symlinks are skipped. `Path::is_dir` resolves them, so a link would be followed into whatever
    /// tree it points at, and a broken `*.rs` link would panic in `read_to_string`.
    fn push_rs_sources(dir: &Path, sources: &mut Vec<String>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap_or_else(|error| panic!("{}: {error}", dir.display()));
        for entry in &entries {
            let path = entry.path();
            let file_type = entry
                .file_type()
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                push_rs_sources(&path, sources);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                sources.push(
                    std::fs::read_to_string(&path)
                        .unwrap_or_else(|error| panic!("{}: {error}", path.display())),
                );
            }
        }
    }

    /// **The tripwire on the missing catalog line**, and it reads the crate's *real* registration
    /// inventory — [`crate::register_providers`], the one function a catalog line would call —
    /// rather than a builder assembled inside the test, which could never contain a generator and
    /// so could never detect one arriving.
    ///
    /// sc-17156 added `.register_generator(…)` to `register_providers`, so `build()` now succeeds
    /// and this test takes its `Ok` arm: it demands the catalog line, and that line exists. Before
    /// then, `build()` rejected the memory strategy for want of a generator and the `Err` arm
    /// asserted the catalog must **not** carry the line (it would break
    /// `candle_gen_catalog::provider_registry()`), so wiring the line early failed here too.
    ///
    /// Both arms stay live, which is the safety property the declaration's doc comment claims,
    /// actually guarded: removing the catalog line while the generator is still registered fails
    /// here, and the failure message names the line to add.
    ///
    /// The `Err` arm's needle is the catalog's declared *dependencies*, not its sources — see
    /// [`catalog_depends_on_this_crate`] for what that covers and the one route it does not.
    /// Sources are the wrong needle here: a generator registered straight from the catalog
    /// would never pass through this crate's inventory and would leave the memory contract
    /// unregistered, and it could be written in any catalog module, so a scan of `lib.rs` alone
    /// goes green in exactly the state this test exists to catch.
    ///
    /// A known, accepted wart: a bare `pub use candle_gen_minimax_h3 as …;` re-export — which needs
    /// the dependency but registers nothing — false-reds the `Err` arm. That is the correct
    /// trade-off for a needle with no false *greens*, and the message says what to check.
    #[test]
    fn a_generator_landing_here_forces_the_catalog_line() {
        let reached = catalog_depends_on_this_crate();
        let wired =
            reached && catalog_sources().contains("candle_gen_minimax_h3::register_providers");
        match crate::register_providers(ProviderRegistryBuilder::new()).build() {
            Err(error) => {
                assert!(
                    error.to_string().contains("no matching generator"),
                    "unexpected rejection: {error}"
                );
                assert!(
                    !reached,
                    "candle-gen-catalog's Cargo.toml now depends on candle-gen-minimax-h3, but \
                     this crate's own `register_providers` still registers no generator. Either \
                     the catalog line landed early — `provider_registry()` cannot build a memory \
                     strategy with no matching generator — or a generator was registered straight \
                     from the catalog (in any module) and bypassed this crate's inventory, leaving \
                     MEMORY_REGISTRATION unregistered. Register generators in \
                     `candle_gen_minimax_h3::register_providers` (sc-17156). A dependency added \
                     for a bare re-export with no registration trips this too — drop it until the \
                     generator exists"
                );
            }
            Ok(registry) => {
                let generators: Vec<&str> = registry
                    .generators()
                    .map(|registration| (registration.descriptor)().id)
                    .collect();
                assert!(
                    wired,
                    "this crate now registers generator(s) {generators:?}, so its memory contract \
                     is finally wirable — add `candle-gen-minimax-h3` to candle-gen-catalog's \
                     Cargo.toml (reached: {reached}) and `let registry = \
                     candle_gen_minimax_h3::register_providers(registry);` to its \
                     `register_providers` (and the matching `ProviderCrate` row), or the contract \
                     stays unreachable at runtime (sc-17156)"
                );
            }
        }
    }

    // --- AC2: nothing optimized is declared, and nothing optimized is reachable ------------------

    /// **Rungs 1-4 are all declared `Missing`, and each is independently refused at selection.**
    ///
    /// Rung 1 is the interesting member: sc-17156 made the staging real in `generate_impl` and
    /// forced the offload policy, but the declaration deliberately did not follow (see
    /// `strategies`), so it is refused here exactly like 2-4. **Two** independent things would each
    /// have to change before it became admittable — the declaration itself, and the `calibration`
    /// identity that is `None` — and the assertions below check the declaration and the admission
    /// decision separately so a change to either is visible rather than absorbed by the other.
    #[test]
    fn no_optimized_rung_is_declared_or_selectable() {
        let contract = declared();
        assert!(
            contract.calibration.is_none(),
            "candle has no fitted curve for this family"
        );
        for strategy in MemoryStrategy::ALL {
            // Only rung 0 is declared `Implemented`; 1-4 are `Missing`. `is_optimized()` — which
            // decides which arms below run — is `!matches!(self, Resident)`, so `StagedResidency`
            // IS optimized and the refusal arms DO exercise rung 1. That is the point of the test,
            // not an exception to it: the rung whose mechanism exists is refused all the same.
            let expected = if strategy == MemoryStrategy::Resident {
                MemoryStrategySupport::Implemented
            } else {
                MemoryStrategySupport::Missing
            };
            assert_eq!(
                contract.capability(strategy).expect("entry").support,
                expected,
                "{strategy:?}"
            );
            if strategy.is_optimized() {
                assert!(
                    contract
                        .validate_selection(&MemorySelection {
                            strategy,
                            tier: MemoryNumericTier {
                                precision: candle_gen::gen_core::Precision::Bf16,
                                quant: None,
                                component_precision_floors: &[],
                            },
                            parameters: MemoryStrategyParameters::default(),
                        })
                        .is_err(),
                    "{strategy:?} must not be selectable"
                );
                assert!(
                    matches!(
                        safety_check(&weightless_spec(), &contract, &context(strategy)),
                        MemorySafetyDecision::Reject { .. }
                    ),
                    "{strategy:?} must be refused at admission"
                );
            }
        }
        // The control arm: the one rung this backend does implement is admitted, so the rejections
        // above are not an always-rejecting check.
        assert_eq!(
            safety_check(
                &weightless_spec(),
                &declared(),
                &context(MemoryStrategy::Resident)
            ),
            MemorySafetyDecision::Accept
        );
    }

    fn context(strategy: MemoryStrategy) -> MemoryRunContext {
        MemoryRunContext {
            selection: MemorySelection {
                strategy,
                tier: MemoryNumericTier {
                    precision: candle_gen::gen_core::Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
                parameters: MemoryStrategyParameters::default(),
            },
            calibration_abi: candle_gen::gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: String::new(),
            load_shape: LOAD_SHAPE,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1344,
                height: 768,
                batch: 1,
                frames: 124,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 256 * 1024 * 1024 * 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1024 * 1024 * 1024,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "unit-test".to_owned(),
        }
    }

    // --- AC3: weights-free conformance, and a malformed contract fails ---------------------------

    /// The same check the registry path runs, on the same fixture factory, in the default lane.
    #[test]
    fn fixture_contract_conforms_weights_free() {
        let fixture = (MEMORY_CONTRACT_FIXTURE.contract)(&weightless_spec()).expect("fixture");
        assert_eq!(
            fixture.asset_facts,
            MemoryAssetFacts::default(),
            "the fixture must inject zero asset facts without touching the filesystem"
        );
        gen_core_testkit::memory_strategy_conformance(&fixture);

        // ...and it must not diverge from the production declaration in anything but bytes.
        let root = tempfile::tempdir().expect("tempdir");
        full_snapshot(root.path());
        let production =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        assert_eq!(fixture.strategies, production.strategies);
        assert_eq!(fixture.lifecycle, production.lifecycle);
        assert_eq!(fixture.formula, production.formula);
        assert_eq!(fixture.calibration, production.calibration);
        assert_eq!(fixture.load_shape, production.load_shape);
        assert_eq!(fixture.backend, production.backend);
    }

    /// Each mutation is applied **alone** to a known-good contract, so each guard is proven to
    /// detect its own breakage rather than the set proving itself.
    #[test]
    fn each_contract_mutation_is_independently_detected() {
        assert!(
            gen_core_testkit::check_memory_strategy_contract(&declared()).is_ok(),
            "the shipped contract must conform, or every mutation below is vacuous"
        );

        let mutations: Vec<ContractMutation> = vec![
            (
                "a dropped strategy entry",
                Box::new(|c: &mut MemoryProviderContract| {
                    c.strategies
                        .retain(|entry| entry.strategy != MemoryStrategy::BoundedDecode);
                }),
            ),
            (
                "a duplicated strategy entry",
                Box::new(|c: &mut MemoryProviderContract| {
                    let first = c.strategies[0].clone();
                    c.strategies.push(first);
                }),
            ),
            (
                "a Resident baseline that is not implemented",
                Box::new(|c: &mut MemoryProviderContract| {
                    for entry in &mut c.strategies {
                        if entry.strategy == MemoryStrategy::Resident {
                            entry.support = MemoryStrategySupport::Missing;
                        }
                    }
                }),
            ),
            (
                "an empty StructurallyNotApplicable reason",
                Box::new(|c: &mut MemoryProviderContract| {
                    for entry in &mut c.strategies {
                        if entry.strategy == MemoryStrategy::BoundedAttention {
                            entry.support = MemoryStrategySupport::StructurallyNotApplicable {
                                reason: "   ".to_owned(),
                            };
                        }
                    }
                }),
            ),
            (
                "StagedResidency implemented with no lifecycle phases",
                Box::new(|c: &mut MemoryProviderContract| {
                    for entry in &mut c.strategies {
                        if entry.strategy == MemoryStrategy::StagedResidency {
                            entry.support = MemoryStrategySupport::Implemented;
                        }
                    }
                }),
            ),
            (
                "BoundedDecode implemented with no tile domain",
                Box::new(|c: &mut MemoryProviderContract| {
                    implement_without_range(c, MemoryStrategy::BoundedDecode)
                }),
            ),
            (
                "BoundedAttention implemented with no chunk domain",
                Box::new(|c: &mut MemoryProviderContract| {
                    implement_without_range(c, MemoryStrategy::BoundedAttention)
                }),
            ),
            (
                "BoundedTransformerResidency implemented with no window domain",
                Box::new(|c: &mut MemoryProviderContract| {
                    implement_without_range(c, MemoryStrategy::BoundedTransformerResidency)
                }),
            ),
            (
                "base_bytes that does not equal its components",
                Box::new(|c: &mut MemoryProviderContract| c.asset_facts.base_bytes += 1),
            ),
            (
                "a malformed calibration fingerprint",
                Box::new(|c: &mut MemoryProviderContract| {
                    c.calibration = Some(MemoryCalibrationIdentity::new("No_Version", LOAD_SHAPE));
                }),
            ),
            (
                "a calibration load shape that disagrees with the contract",
                Box::new(|c: &mut MemoryProviderContract| {
                    c.calibration = Some(MemoryCalibrationIdentity::new(
                        "minimax-h3-candle-v1",
                        LoadShape::DeferredMaterialization,
                    ));
                }),
            ),
        ];

        for (name, mutate) in mutations {
            let mut contract = declared();
            mutate(&mut contract);
            assert!(
                gen_core_testkit::check_memory_strategy_contract(&contract).is_err(),
                "conformance must reject {name}"
            );
        }
    }

    /// AC5's failure shape: a lever declared `Implemented` with no [`MemoryParameterRanges`].
    fn implement_without_range(contract: &mut MemoryProviderContract, strategy: MemoryStrategy) {
        for entry in &mut contract.strategies {
            if entry.strategy == strategy {
                entry.support = MemoryStrategySupport::Implemented;
                entry.parameters = MemoryParameterRanges::default();
            }
        }
        match strategy {
            MemoryStrategy::BoundedDecode => contract.lifecycle.decode_tiling = true,
            MemoryStrategy::BoundedAttention => contract.lifecycle.attention_chunking = true,
            MemoryStrategy::BoundedTransformerResidency => {
                contract.lifecycle.transformer_window_materialization = true
            }
            _ => {}
        }
    }

    // --- AC5: parameter ranges are declared exactly where they are owned ------------------------

    #[test]
    fn parameter_ranges_are_owned_by_the_rung_that_consumes_them() {
        let contract = declared();
        assert!(contract.conformance_errors().is_empty());
        for capability in &contract.strategies {
            assert_eq!(
                capability.parameters,
                MemoryParameterRanges::default(),
                "{:?} owns no numeric parameters on this backend",
                capability.strategy
            );
        }
        for strategy in [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let mut mutated = declared();
            implement_without_range(&mut mutated, strategy);
            assert!(
                !mutated.conformance_errors().is_empty(),
                "{strategy:?} implemented with no MemoryParameterRanges must fail"
            );
        }
    }

    // --- AC4: measured asset facts --------------------------------------------------------------

    /// Tolerance for the GB figures recorded on sc-18659, in bytes. The story quotes 66.73 GB for
    /// the text encoder; the measured on-disk total is 66,714,912,872 B = 66.715 GB, so the byte
    /// constants are authoritative and the GB figures are held only to this window.
    const GB_TOLERANCE_BYTES: u64 = 20_000_000;

    fn assert_within(measured: u64, story_gb: f64, what: &str) {
        let story_bytes = (story_gb * 1e9) as u64;
        let delta = measured.abs_diff(story_bytes);
        assert!(
            delta <= GB_TOLERANCE_BYTES,
            "{what}: measured {measured} B ({:.3} GB) is {delta} B from the recorded {story_gb} GB, \
             outside the {GB_TOLERANCE_BYTES} B tolerance",
            measured as f64 / 1e9
        );
    }

    #[test]
    fn measured_component_bytes_match_the_recorded_footprints() {
        assert_within(DIT_BF16_BYTES, 66.28, "33 B DiT partition at bf16");
        assert_within(TEXT_ENCODER_BYTES, 66.73, "Qwen3-VL-32B text encoder");
        assert_within(VIDEO_VAE_BYTES, 10.42, "video VAE");
        assert_within(AUDIO_VAE_BYTES, 0.61, "audio VAE");
        assert_eq!(
            ADALN_EVICTED_BYTES, 26_020_915_200,
            "the exact bytes crate::dit::adaln releases"
        );
    }

    /// The two backends must not drift apart on facts about the same checkpoint.
    #[test]
    fn the_measured_facts_are_the_same_numbers_the_mlx_sibling_declares() {
        // Mirrored deliberately rather than shared through a dependency: `candle-gen-*` crates do
        // not depend on `mlx-gen-*`. These literals are the same ones
        // `mlx_gen_minimax_h3::memory_strategy` declares, and both are asserted against the story's
        // recorded GB figures above, so a drift on either side fails there first.
        assert_eq!(TEXT_ENCODER_BYTES, 66_714_912_872);
        assert_eq!(DIT_BF16_BYTES, 66_280_504_216);
        assert_eq!(VIDEO_VAE_BYTES, 10_415_558_888);
        assert_eq!(AUDIO_VAE_BYTES, 605_429_340);
    }

    #[test]
    fn a_staged_dit_component_is_charged_at_its_own_size() {
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
        let staged = tempfile::tempdir().expect("tempdir");
        const Q4_BYTES: u64 = 18_779_970_678;
        sparse_snapshot(staged.path(), &[("transformer", Q4_BYTES)]);

        let contract = contract_for(
            &LoadSpec::new(WeightsSource::Dir(root.path().into())).with_component(
                DIT_COMPONENT,
                WeightsSource::Dir(staged.path().join(DIT_COMPONENT)),
            ),
        )
        .expect("contract");
        assert_eq!(contract.asset_facts.transformer_bytes, Q4_BYTES);
    }

    // --- the declaration facts that are easy to get wrong ----------------------------------------

    /// The load shape is pinned to the loader, not mirrored from the request.
    #[test]
    fn load_shape_is_pinned_to_the_loader_not_taken_from_the_spec() {
        let spec = weightless_spec();
        assert_eq!(spec.load_shape, LoadShape::DeferredMaterialization);
        assert_eq!(
            contract_for(&spec).expect("contract").load_shape,
            LoadShape::EagerMaterialization
        );
        // ...and the spec IS read: the asset facts come off it.
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[("audio_vae", AUDIO_VAE_BYTES)]);
        let resolved =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        assert_eq!(resolved.asset_facts.decoder_bytes, AUDIO_VAE_BYTES);
    }

    /// **What actually blocks an optimized rung on this backend, executed rather than asserted**
    /// (sc-18660).
    ///
    /// The MLX sibling declares rung 2 `Implemented` because its spatial tiling is reachable from a
    /// request. **The mechanism exists here too** — `MiniMaxH3VideoVae::decode_clip` tiles at the
    /// same reference geometry, through the same `BoundedStitch` — so the natural conclusion is
    /// that this crate under-declares, and the natural fix is to add the weights-free
    /// `MemoryBehaviorRegistration` seam the MLX sibling has.
    ///
    /// **That fix is not available on this tree, and the reason is one link further back than the
    /// seam.** The chain, each link driven below rather than described:
    ///
    /// 1. this backend has no fitted curve of its own — sc-17156 landed the pipeline, not
    ///    measurements — so `calibration` is `None`;
    /// 2. `standard_memory_behavior_context` **requires** a calibration identity, so no
    ///    `MemoryBehaviorFixture` can be constructed at all;
    /// 3. without a fixture there is no `MemoryBehaviorRegistration`;
    /// 4. `check_memory_strategy_contract` rejects a contract implementing any *optimized* rung
    ///    with no behavior seam.
    ///
    /// So the blocker is the absent calibration identity — not the seam, and not the mechanism. The
    /// seam cannot be written before the identity exists. This test **fails the moment a
    /// calibration identity lands**, which is exactly when the flip becomes possible, and it names
    /// the rungs waiting on it.
    #[test]
    fn an_optimized_rung_here_is_blocked_by_the_missing_calibration_identity_not_the_seam() {
        let contract = declared();

        // Link 1 — load-bearing, not incidental.
        assert!(
            contract.calibration.is_none(),
            "a calibration identity landed: rungs 1 and 2 can now be declared here. Add the \
             weights-free MemoryBehaviorRegistration seam (model it on \
             mlx_gen_minimax_h3::memory_strategy::MEMORY_BEHAVIOR) and flip them — the mechanisms \
             already exist in this crate."
        );

        // Link 2 — executed. This is the call every behavior seam must make, and it errors here.
        let refused = candle_gen::gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedDecode,
            MemoryNumericTier {
                precision: candle_gen::gen_core::Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            candle_gen::gen_core::MemoryBehaviorRoute {
                mode: candle_gen::gen_core::MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .expect_err("a behavior context must not be constructible without a calibration identity");
        assert!(
            refused.to_string().contains("calibration identity"),
            "expected the identity refusal, got: {refused}"
        );

        // …while the MECHANISM rung 2 would declare is genuinely present in this crate today, so
        // the gap is a declaration gap and not a porting gap.
        // Read through `Default`, not off the constants: comparing a constant with itself would
        // pass with the mechanism deleted.
        let tiling = crate::spatial_tiling::SpatialTiling::default();
        assert!(
            tiling.enabled,
            "the shipped candle VAE must tile by default"
        );
        assert_eq!((tiling.tile_height, tiling.tile_width), (256, 256));
        assert_eq!((tiling.overlap_height, tiling.overlap_width), (64, 64));
        // …and the stitcher rung 2 would declare is constructible here today.
        assert!(crate::spatial_tiling::BoundedStitch::new(2, 2, &[1], &[1]).is_ok());
    }

    /// The three ways this backend's declaration diverges from the MLX sibling's, pinned so a later
    /// slice cannot copy an MLX verdict across without the test noticing.
    #[test]
    fn the_candle_declaration_differs_from_mlx_where_the_backends_differ() {
        let contract = declared();
        // 1. No fitted curve: the floor arm only, and no calibration identity.
        assert_eq!(contract.formula, MemoryFormulaKind::AssetBytesPlusHeadroom);
        assert!(contract.calibration.is_none());
        // 2. Staged residency is IMPLEMENTED IN CODE as of sc-17156 but still declared `Missing`,
        //    because an implemented optimized rung needs a behavior seam this lane does not have
        //    yet, and that seam waits on a `calibration` identity (see
        //    `an_optimized_rung_here_is_blocked_by_the_missing_calibration_identity_not_the_seam`).
        //    Under-declaring is the safe direction — see `strategies`.
        assert_eq!(contract.lifecycle, MemoryLifecycleCapabilities::default());
        assert!(contract.lifecycle.phases.is_empty());
        assert!(!contract.lifecycle.synchronized_phase_release);
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .expect("entry")
                .support,
            MemoryStrategySupport::Missing
        );
        // 3. No fused streaming SDPA: rung 3 stays open here on its own evidence, never on MLX's.
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .expect("entry")
                .support,
            MemoryStrategySupport::Missing,
            "candle materializes attention scores; the MLX streaming verdict does not carry over"
        );
        assert!(matches!(
            contract.backend,
            MemoryBackendRealization::CandleCuda { .. }
        ));
        assert_eq!(
            contract.backend.backend_id(),
            "candle",
            "the evidence key must not be recorded under the MLX backend"
        );
    }

    /// A phase hook declared without an implementation is the false declaration conformance cannot
    /// catch. This pins the honest state: no phase is declared, because the rung they belong to is
    /// not declared either — the behavior seam that would carry both waits on a `calibration`
    /// identity this backend does not have.
    ///
    /// The release primitive the phases WILL declare is exercised here anyway, so this test also
    /// fails if it is removed while the declaration is pending.
    #[test]
    fn no_lifecycle_phase_is_declared_without_an_implementation() {
        let contract = declared();
        for phase in [
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ] {
            assert!(
                !contract.lifecycle.phases.contains(&phase),
                "{phase:?} must not be declared while StagedResidency is Missing"
            );
        }
        crate::dit::release_device_memory(&candle_gen::candle_core::Device::Cpu)
            .expect("the phase-release primitive must exist and succeed");
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        assert!(!contract.lifecycle.transformer_window_materialization);
    }
}
