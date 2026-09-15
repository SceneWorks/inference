# Full Codebase Review — inference — 2026-08-01

## Executive summary

- **Repository at a glance:** Rust 1.96.0 Cargo workspace, 93 path members (contracts, MLX/Candle LLM + media engines, Candle audio family, platform runtime bundles) plus Python 3 tooling and GitHub Actions CI.
- **Review focus:** the last week of changes (`8b053451..HEAD`, 2026-07-24 → 2026-08-01: 674 files, ~180k insertions) — dominated by the shared memory-optimization ladder (gen-core `memory_strategy` contract, rungs 1–4, request-scoped residency, calibration evidence) and its first seven provider adoptions, plus the Stable Audio 3 port and heavy CI/tooling churn. Whole-file review was applied to the gen-core memory contract, residency, and every provider `memory_strategy`/`memory` module; diff-focused review with full-file context everywhere else. Other issues encountered along the way are recorded regardless of age.
- **Headline:** The gen-core contract layer itself is unusually rigorous — invariants are documented, mutation-tested, and the ladder arithmetic (rung ordering, engagement, block windows, chunk tables, unit conversions) verified clean across two independent passes. The known Candle 26× per-window rung-4 defect is genuinely fixed, not papered over. The risk is concentrated in the **adoption layer**: the safety gate, selection→`GenerationMemory` translation, and request scope are hand-copied per provider, and in only seven adopters that duplication has already produced three real semantic divergences that the conformance testkit cannot detect. Before committing the remaining ~130 rollout stories, hoist the gate and translation into gen-core, extend the testkit with behavioral probes, and add the load-shape axis to the evidence key (an evidence-ABI change that gets more expensive with every story that lands measurements).
- **Counts:** Critical: 0 | High: 10 | Medium: 23 | Low: 45 | Info: 14 (F-182..F-273).

### Readiness verdict for the remaining rollout

The foundation question this review was asked to answer: **the contract is ready; the adoption template is not.** Per-adopter ladder plumbing runs ~200–1000 lines (z-image MLX ~1010, qwen ~500, krea 2×~320, lens ~282, flux2 ~192, mage ~330, candle z-image ~408, candle krea ~250, candle lens ~200), of which roughly 250 lines per story are copy-paste of gate/translation/scope machinery. Divergence rate so far: ~3 semantic defects per 7 adopters (F-182, F-183, F-186), none currently detectable by conformance. Fix F-182/F-183/F-187 (one hoisting story), F-184 (testkit probes), and F-185 (evidence-key ABI) **before** the rollout; with those done the remaining stories become mostly declarative and the defect class dies structurally.

## Critical findings

None.

## High findings

#### [F-182] Fold the calibration handshake into the shared default safety check
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/contracts/gen-core/src/memory_strategy.rs:1403-1433`; `crates/media/candle-gen/candle-gen-krea/src/lib.rs:1364,1486`
- **Finding:** `MemoryRunContext` documents `calibration_abi`/`calibration_fingerprint` as the admission handshake, but `default_memory_strategy_safety_check` never reads either field — it validates only the selection and the budget fit. Eight MLX/Candle providers hand-roll the mismatch rejection; `candle-gen-krea` registers the bare default and has **zero** references to `calibration_fingerprint`/`calibration_abi` anywhere in the crate (verified by grep), so its provider-side gate accepts a fingerprint-mismatched optimized selection end-to-end.
- **Impact:** The module header's own invariant — "fingerprint-mismatched … evidence is never turned into a claimed optimized fit by provider-local policy" — currently rests entirely on the SceneWorks caller for two shipped Candle providers, and the function's name ("default…safety_check") invites the next 130 adopters to assume it is sufficient.
- **Suggested fix:** Move the `context.calibration_abi != identity.abi || context.calibration_fingerprint != identity.fingerprint → Reject` check into `default_memory_strategy_safety_check` (contract and context both carry the identity; no new inputs needed), then delete the eight provider copies. Pair with the F-184 testkit probe so regression is impossible.
- **Confidence:** High (verified: no calibration read in the default; no handshake anywhere in candle-gen-krea)

#### [F-183] Reconverge the provider safety gates — tier validation is missing on the flagship adopter
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/media/mlx-gen/mlx-gen-z-image/src/memory_strategy.rs:887-973`; `crates/media/mlx-gen/mlx-gen-qwen-image/src/memory_strategy.rs:275-282`; `crates/media/mlx-gen/mlx-gen-krea/src/memory_strategy.rs:151-219`, `block_memory_strategy.rs:113-171`; `crates/media/mlx-gen/mlx-gen-lens/src/memory_strategy.rs:94-152`; `crates/media/candle-gen/candle-gen-lens/src/lib.rs:1664-1684`
- **Finding:** Six hand-rolled `safety_check` functions repeat the same ~60-line pipeline (handshake → `validate_selection` → optional route gate → budget), and the check *set* has already diverged: qwen, krea-base, lens, and candle-lens verify `context.selection.tier` against the loaded `precision`/`quant`; z-image and krea-pose do not (`grep 'selection\.tier'` in z-image's module returns nothing; its `registered_safety_check(_spec, …)` discards the spec). The budget-rejection string is copy-pasted verbatim in four crates plus gen-core.
- **Impact:** A caller bug that admits a q8-calibrated selection (with q8-measured peaks) against a q4-loaded z-image generator sails through the provider gate — the exact defense-in-depth failure the gate exists to catch. Multiplied over the rollout, each new story re-decides which checks to include.
- **Suggested fix:** One gen-core helper — e.g. `standard_memory_strategy_safety_check(contract, context, loaded_tier: Option<MemoryNumericTier>, route_gate: Option<&dyn Fn(..)>)` — performing handshake + tier + selection + budget in one audited place; providers pass only their route closure. Retrofit the six existing gates in the same story as F-182.
- **Confidence:** High (verified)

#### [F-184] Extend the memory-contract conformance suite so a behaviorally wrong adopter fails
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/contracts/gen-core-testkit/src/memory_strategy.rs:11-180`; `crates/contracts/gen-core/src/generator.rs:74-80`
- **Finding:** The testkit checks static `conformance_errors` plus exactly one behavioral family — four PiD-route admission probes — and only for contracts that declare `pid_decode_routes` (`let Some(routes) = contract.pid_decode_routes.as_ref() else { return; }` at line 113 ends all behavioral checking). Probe contexts use `total_bytes: u64::MAX, predicted_peak_bytes: 0`, so budget rejection is never exercised. Nothing pins: fingerprint-mismatch rejection, tier-mismatch rejection, translation-vs-`engaged_composition` parity (F-186), double-`finish` scope semantics, cross-route geometry pairs (native edge + PiD overlap is never probed), or that a Generator publishing a contract actually overrides `begin_memory_strategy_request` (the trait default returns `Ok(None)`, so a contract can advertise Implemented rungs whose selection is accepted but never executed).
- **Impact:** Direct answer to "can a future provider adopt the ladder wrong and still compile + pass": yes, in at least five distinct ways — three of which already occurred in the first seven adopters (F-182, F-183, F-186). The conformance suite is the only automated guard the 130-story rollout has.
- **Suggested fix:** Add universal probes to `check_memory_registration`: (a) mutated fingerprint/ABI must Reject; (b) mutated tier must Reject; (c) `predicted_peak_bytes = u64::MAX` vs finite budget must Reject; (d) each Implemented optimized rung must Accept a valid selection (non-vacuity); (e) `configure_request` must write exactly `contract.generation_memory(selection)` (after F-187) and `finish` semantics must match one pinned rule; (f) cross-paired route geometry probes.
- **Confidence:** High (verified early-return and probe budget)

#### [F-185] Add the load-shape axis to the evidence key and handshake before measurements accumulate
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/contracts/gen-core/src/memory_strategy.rs:1859-1872`; `crates/media/candle-gen/candle-gen-z-image/src/memory_strategy.rs:124`; `crates/media/mlx-gen/mlx-gen-krea/src/block_memory_strategy.rs:32-33`; `crates/media/mlx-gen/mlx-gen-lens/src/memory_strategy.rs:16`
- **Finding:** `MemoryEvidenceKey` has no load-shape axis (route/backend/tier/mode/overlay/geometry/strategy/composition/parameters only). Z-image proved shape is load-bearing — its Eager and Deferred baselines measured 9.550 vs 4.847 GiB and it invented the `-deferred`/`-eager` fingerprint-suffix convention; qwen copied it, but krea-base and lens publish a single fingerprint across shapes, and candle z-image hardcodes `load_shape: LoadShape::DeferredMaterialization` (verified at line 124) without reading the spec, so its contract asserts the rung-4 prerequisite regardless of how the generator was actually loaded.
- **Impact:** Evidence keyed under one materialization shape can authorize a fit under the other (~2× error in the dangerous direction) for every adopter that doesn't hand-copy the suffix convention. This is an evidence-ABI change: every story that lands measurements first makes it more expensive.
- **Suggested fix:** Add `load_shape: LoadShape` to `MemoryEvidenceKey` and the calibration identity (or make `MemoryCalibrationIdentity` a struct `{base_fingerprint, load_shape}` so the suffix is typed, not conventional); fix candle z-image to derive `load_shape` from the spec. Do this before the rollout, not at story 60.
- **Confidence:** High (verified)

#### [F-186] Declare krea-base's rung-4 → staged-residency prerequisite so evidence identity matches execution
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/media/mlx-gen/mlx-gen-krea/src/block_memory_strategy.rs:229-241` (scope), contract in the same file (no `additional_prerequisites`); contrast `crates/media/mlx-gen/mlx-gen-qwen-image/src/memory_strategy.rs:234-240`
- **Finding:** Krea-base's scope hardcodes `stage_residency: true` into the rung-4 `GenerationMemory` (verified at lines 232/236), but its contract declares no `(BoundedTransformerResidency → StagedResidency)` prerequisite (verified: zero `additional_prerequisites` in the file), so `engaged_composition(BoundedTransformerResidency)` = `[Resident, BoundedTransformerResidency]` while the run physically stages residency. Qwen declares the edge correctly.
- **Impact:** Krea rung-4 evidence is keyed to a composition that omits a mechanism that runs (and whose calibration table was measured *with* staging). Any exclusion/selector logic reading `engaged_composition` will mis-predict Krea, and the pattern is reproducible by any rollout story because nothing pins translation-vs-engagement parity (F-184e).
- **Suggested fix:** Add the one-line prerequisite edge mirroring qwen; land the F-184 parity probe in the same change so the class dies.
- **Confidence:** High (verified)

#### [F-187] Hoist the selection→GenerationMemory translation into gen-core
- **Category:** redundant
- **Severity:** High
- **Location:** `crates/media/mlx-gen/mlx-gen-z-image/src/memory_strategy.rs:659-701`; `crates/media/mlx-gen/mlx-gen-qwen-image/src/memory_strategy.rs:355-388`; `crates/media/candle-gen/candle-gen-z-image/src/memory_strategy.rs:31-62`; hand-rolled variants at `crates/media/mlx-gen/mlx-gen-krea/src/block_memory_strategy.rs:229-248`, `mlx-gen-lens/src/memory_strategy.rs:210-224`, `mlx-gen-krea/src/memory_strategy.rs:131-149`
- **Finding:** The `engages()`-driven translation is copied ~35 lines essentially verbatim in three crates, while three more adopters hand-roll a `match` over strategies — the exact hazard z-image's own comment warns about ("a hardcoded `..decode` arm is the same hazard as a `>=` comparison wearing different syntax"). Divergence has already happened twice: candle z-image's copy silently drops `transformer_window_component`, and krea-pose's translation never sets `stage_residency` despite declaring rung 1 Implemented. Both `GenerationMemory` and `MemoryProviderContract` are gen-core types; nothing here is backend-specific.
- **Impact:** This function *is* the executable meaning of "engaged composition". Six implementations produced two silent drops in week one; multiplied across the rollout it is the highest-probability source of "evidence says X ran, Y actually ran".
- **Suggested fix:** Add `MemoryProviderContract::generation_memory(&self, &MemorySelection) -> Option<GenerationMemory>` in gen-core (z-image's version, component scope included); delete the six copies; assert parity in the testkit (F-184e).
- **Confidence:** High

#### [F-188] Krea-realtime silently renders aligned-down geometry for off-grid requests
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/media/mlx-gen/mlx-gen-krea-realtime/src/pipeline.rs:117`; `crates/media/mlx-gen/mlx-gen-krea-realtime/src/t2v.rs:793-794,527-528`
- **Finding:** The descriptor advertises `SizeFloor::RangeChecked` (no grid), and latent geometry is derived with flooring integer division (`job.height as usize / SPATIAL_STRIDE`), then rendered at `latent * 8` — so an explicit 644×484 request passes validation and silently renders 640×480 (verified). The only model-local rejection is "smaller than one 8px VAE cell".
- **Impact:** Exactly the silent-geometry-substitution defect this same week eliminated elsewhere: scail2 got `ResolvedDownstreamExplicitGrid` for it (sc-16198/sc-15807, "refusing an explicit 1280×730 instead of silently rendering 1280×704") and Wan rejects off-grid. The newest crate in the family violates the family rule the week it was established.
- **Suggested fix:** Advertise `SizeFloor::RangeCheckedOnGrid` with the model's true grid multiple on the descriptor so the shared floor rejects off-grid before load; add a pin test like scail2's `explicit_off_grid_size_is_refused`.
- **Confidence:** High (verified)

#### [F-189] Packed Candle Krea tiers now hard-require a writable model directory
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/media/candle-gen/candle-gen-krea/src/loader.rs:105-112`; `crates/media/candle-gen/candle-gen/src/quant/sidecar.rs:148-165`
- **Finding:** `Weights::from_dir` unconditionally runs `PackedWeightSidecars::prepare` for every packed (q4/q8) component open — including plain resident loads that never stream — and `prepare` unconditionally does `fs::create_dir_all` of `.candle-device-format-v1` inside the caller-provisioned component directory plus opens a lock file with `create(true).write(true)` (verified at sidecar.rs:149,160). On a read-only snapshot this fails even when a fully valid sidecar cache already exists.
- **Impact:** Epic 13657's contract is that user-supplied models at arbitrary caller-provisioned paths must load; a read-only path (read-only mount, shared immutable cache) is a legal such path, and packed Krea tiers that loaded before this week no longer do. The sidecars also silently roughly double the on-disk footprint beside the user's weights. (Z-image only hits this when rung 4 is actually selected, since it prepares lazily.)
- **Suggested fix:** Fall back to the in-memory repack path (or a caller-configurable cache root) when the component dir is not writable; at minimum skip lock-file creation when all artifacts already validate; surface the write-and-disk-space requirement in the provider contract/docs.
- **Confidence:** High on the mechanism; Medium that read-only snapshots occur in current deployments — but the epic-13657 contract makes this a High regardless.

#### [F-190] Surface the Stability "Powered by Stability AI" attribution obligation in the license manifest
- **Category:** security
- **Severity:** High
- **Location:** `crates/audio/candle-audio-stable-audio-3/src/model.rs:703-826`; `release/model-weight-licenses.json:245-420`
- **Finding:** All 18 SA3 weight-license rows carry `attribution: Some("Stable Audio 3 … © Stability AI")`, but the Stability AI Community License requires products using the materials to prominently display "Powered by Stability AI" — a string that appears nowhere in the repository (verified: `grep -rin "powered by"` over crates/, release/, docs/ returns nothing). The machine-readable manifest is the only signal the product's licenses page consumes.
- **Impact:** SceneWorks builds its attributions surface from this manifest; shipping the six SA3 providers without the mandated mark is a license breach for the whole family, discoverable only by a human re-reading the license.
- **Suggested fix:** Extend the six `*_ROOT_WEIGHT_LICENSE` attribution (or restriction) strings to carry the display requirement, regenerate `release/model-weight-licenses.json`, and add a catalog test asserting the string is present for every `LicenseRef-Stability-AI-Community` row.
- **Confidence:** High

#### [F-191] Classify `docs/migration/sa3-*-reference/**` out of the docs-only CI lane
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `scripts/ci/select_lanes.py:55-65`; e.g. `crates/audio/candle-audio-stable-audio-3/tests/primitive_oracle.rs:86` (non-ignored), `tests/chunked_oracle.rs:88`, `tests/dit_oracle.rs:98-107`
- **Finding:** Every path under `docs/` selects only the `docs` lane (link check only), but this week landed load-bearing Rust test fixtures under `docs/migration/sa3-*-reference/*.safetensors` that non-`#[ignore]`d PR-time tests read (verified: `frozen_upstream_missing_branches_match` at primitive_oracle.rs:86 carries no ignore attribute).
- **Impact:** A PR that only regenerates a reference artifact skips `candle_cpu`/`macos_metal`/`windows_cuda` entirely, so the only executable parity gates against the changed artifact never run; a corrupted or wrongly regenerated oracle lands green and fails on the next unrelated crate-touching PR — a direct hole in the fail-safe lane-selection invariant.
- **Suggested fix:** Add a rule before the `docs/` branch routing `docs/migration/sa3-*-reference/` (or generally any `docs/**` path referenced from `crates/**/tests`) to the compile/test lanes as well; add a `test_select_lanes.py` case.
- **Confidence:** High (verified)

## Medium findings

#### [F-192] Fix effective-budget over-report when committed exceeds total
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/contracts/gen-core/src/memory_strategy.rs:1676-1691`
- **Finding:** `MemoryBudget::effective_bytes` computes `total.saturating_sub(committed)` *before* adding `reclaimable_bytes`, so when `committed_bytes > total_bytes` (overcommit — plausible on unified memory where `total` is a cap) the deficit is silently dropped. Verified: `{total:100, committed:110, reclaimable:50, headroom:10}` should yield 30 and returns 40. The existing saturation test uses `reclaimable:5`, where both formulas collapse to 0, masking the bug.
- **Impact:** `fits()` — the ladder's one authoritative accept/reject — can accept a predicted peak up to `committed − total` bytes larger than what is actually free after reclamation, i.e. an OOM admitted in precisely the constrained regime the ladder exists for.
- **Suggested fix:** Compute in signed/checked arithmetic (`(total − committed + reclaimable − headroom).clamp(0, total − headroom)` over `i128`) and add an overcommit-with-large-reclaimable test.
- **Confidence:** High on the arithmetic (verified); Medium that `committed > total` occurs in production accounting.

#### [F-193] Candle z-image's registered admission check proves less than its loaded route enforces
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/contracts/gen-core/src/registry.rs:126-143` (doc); `crates/media/candle-gen/candle-gen-z-image/src/lib.rs:243-257,603-630`
- **Finding:** `MemoryRegistration::safety_check` is documented as "the same function the loaded Generator delegates to". Candle z-image registers the bare default, but the loaded route's real admission is `memory_strategy::validate_context` inside `begin_memory_strategy_request`, which additionally rejects fingerprint mismatch, `has_phases`, and PiD+optimized — rejections that now happen mid-request after the caller committed budget, and that weights-free registry conformance can never observe. MLX z-image registers the full check.
- **Impact:** A shared selector consulting the registered check admits contexts the route then rejects during generation; the registry-doc invariant is quietly unenforced for the Candle lane.
- **Suggested fix:** Register a route function wrapping `validate_context` (as MLX does), or resolve automatically via the F-182/F-183 hoist.
- **Confidence:** High

#### [F-194] MemoryAssetFacts semantics have forked four ways across adopters
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/media/mlx-gen/mlx-gen-z-image/src/memory_strategy.rs:532-644`; `crates/media/mlx-gen/mlx-gen-qwen-image/src/memory_strategy.rs:157-166`; `crates/media/mlx-gen/mlx-gen-krea/src/memory_strategy.rs:106-129`; `crates/media/candle-gen/candle-gen-krea/src/lib.rs:1337-1343`; `crates/media/mlx-gen/mlx-gen-mage/src/model.rs:112-113`
- **Finding:** Z-image keeps `base_bytes` = base model only and declares the control network as typed, quant-projected `MemoryResidentComponent`s; qwen and krea-pose fold `overlay_bytes` *into* `base_bytes` while also declaring the `OverlayBytes` formula variable; candle-krea zeroes asset facts entirely; mage back-derives bytes from a decimal-GB estimate. `total_resident_bytes()` adds overlay only when typed components exist, so qwen/krea-pose are numerically right today but double-count the day they adopt typed components — which the SC-16065 direction encourages.
- **Impact:** Fields with the same name mean different things per provider; the double-count trap fires exactly on the "upgrade" path.
- **Suggested fix:** Pin the invariant in gen-core (`base_bytes` excludes `overlay_bytes`; conformance error when both `AssetBytes` and `OverlayBytes` are declared ambiguously) and hoist one overlay-sizing helper (z-image's tensor-header projection is the correct one).
- **Confidence:** High for the fork; Medium for the double-count consequence (depends on SceneWorks read sites).

#### [F-195] Qwen control overlay bytes are dense on-disk size, not resident size
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/media/mlx-gen/mlx-gen-qwen-image/src/memory_strategy.rs:157-166,244-255`; `crates/media/mlx-gen/mlx-gen-qwen-image/src/model_control.rs:290`
- **Finding:** `source_bytes` reports the control checkpoint's raw file length (for a `Dir`, a non-recursive sum of all files including JSON), but the load quantizes the dense control branch to the requested Q4/Q8 tier — so qwen declares ~2× (Q8) to ~3.6× (Q4) the true resident bytes, and a control `Dir` with subdirectories would silently under-count instead.
- **Impact:** `asset_facts` is documented "provider-owned, load-exact"; the overstated overlay feeds admission arithmetic and calibration variables, causing spurious fit rejections on small hosts (fail-closed, but wrong facts).
- **Suggested fix:** Reuse z-image's tensor-header packed-size projection; hoist the helper into `mlx_gen` since two families need it.
- **Confidence:** High

#### [F-196] Krea native single-file loads declare all-zero asset facts under the directory loads' fingerprint
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/media/mlx-gen/mlx-gen-krea/src/model.rs:1264-1272`; `crates/media/mlx-gen/mlx-gen-krea/src/block_memory_strategy.rs:75-81`
- **Finding:** `component_footprint` returns `PerComponentBytes::default()` (all zeros) for `WeightsSource::File`, so native ComfyUI-merge loads publish `base_bytes = 0` etc. while declaring the same `krea-2-mlx-request-peak-…` calibration fingerprint as directory loads whose facts are real.
- **Impact:** `AssetBytes` is a declared formula variable; evaluating promoted evidence against zero asset bytes under-predicts the peak for single-file loads, and the fingerprint cannot distinguish the shapes. The safety gate then accepts on an under-prediction — the OOM direction.
- **Suggested fix:** Size the single file into the component facts, or refuse to declare a calibration identity when asset facts are unknown.
- **Confidence:** High

#### [F-197] Request-scope lifecycle machinery is sextuplicated with drifted guarantees
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/media/mlx-gen/mlx-gen-z-image/src/memory_strategy.rs:703-882`; `mlx-gen-qwen-image/src/memory_strategy.rs:390-496`; `mlx-gen-krea/src/memory_strategy.rs:238-341` and `block_memory_strategy.rs:193-305`; `mlx-gen-lens/src/memory_strategy.rs:174-281`; `crates/media/candle-gen/candle-gen-z-image/src/memory_strategy.rs:330`
- **Finding:** Six MLX scopes duplicate the terminal barrier (`eval` + `clear_cache` + Drop guard) and geometry checks with real behavioral drift: z-image/krea-pose guard every hook with `ensure_active` and make double-`finish` an error, while qwen/krea-base/lens have no guard at all (finished scopes silently accept further configuration); geometry comparison is `count > batch` on MLX vs `count != batch` on candle; window validation handles the ragged tail on z-image/candle but demands exact `block_count == WINDOW` with hardcoded magic block counts (60/28/24, plus candle z-image's `const BLOCKS: u32 = 30` duplicating its config's `n_layers`) on the others.
- **Impact:** ~120 lines per story of copyable scope code whose safety property (teardown exactly once, hooks dead after finish) is re-decided per story; the cleanup semantics the contract type promises are only as uniform as the weakest copy.
- **Suggested fix:** Add an `MlxRequestScopeCore` (barrier + cache-evict + finished-flag + Drop + geometry/route checks, parameterized by block count and window table) to `mlx-gen` and the `device.synchronize()` twin to `candle-gen`; derive block counts from config; pin double-finish semantics in the testkit (F-184).
- **Confidence:** High

#### [F-198] flux2's ported legacy gate is a template-poisoning counter-example
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/media/mlx-gen/mlx-gen-flux2/src/memory_strategy.rs:20-166`
- **Finding:** The flux2 adoption violates three of the module's own ownership rules: measured coefficients live in the provider (`BASE_GB = 62.9; PER_TOKEN_GB = 0.001_489`), the provider overwrites the caller's `predicted_peak_bytes` (contradicting "the caller remains the sole owner of live-budget accounting"), it parses a reference count out of the free-text `overlay` field (`overlay.strip_prefix("references=")`) that other adopters use as overlay *identity* for evidence keying, and it marks implementable rungs `StructurallyNotApplicable` where siblings use `Missing` — and SNA has real semantics (it vacuously satisfies prerequisite edges).
- **Impact:** Nothing marks the port non-normative; a rollout author grepping for "how did an adopter do X" has a 1-in-7 chance of copying the pattern the contract forbids, and `overlay` now means request-count on one provider and overlay-id on the rest.
- **Suggested fix:** Move the reference count into a typed `MemoryRunContext` field and re-home the coefficients as provider-attached evidence; or at minimum brand the module "legacy shape — do not template", change SNA to `Missing`, and add a conformance rule that overlay strings never carry structured data.
- **Confidence:** High

#### [F-199] The calibration evidence pipeline records magnitude only — no fingerprint provenance
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `scripts/release/verify_residency_ab.py:1-60`; the `*_residency_real_weights.rs` harnesses; `crates/contracts/gen-core/src/memory_strategy.rs:1898-1912`
- **Finding:** The measurement path is: `#[ignore]`d real-weight tests print `SEQ_AB … mode=… peak_mib=…` lines → `verify_residency_ab.py` regex-extracts one resident and one staged peak and asserts a minimum reduction. Nothing records or checks the calibration fingerprint, ABI, tier, geometry, or engaged composition; `MemoryEvidence` has no serde derives and no writer in this repo; per-tier peak tables are hand-copied into doc comments.
- **Impact:** The struct-level ABI is stable enough to scale, but provenance is not: 130 stories will each mint a fingerprint and a doc-table by hand, and nothing detects "layout changed, fingerprint didn't" — the one failure the fingerprint exists for. Cheap to fix now, effectively impossible to retrofit at story 100.
- **Suggested fix:** Have the harnesses emit one machine-readable line carrying the `MemoryEvidenceKey` fields + fingerprint + ABI + peak, and extend the verifier to refuse logs whose fingerprint doesn't match the crate's exported const.
- **Confidence:** High for what exists; Medium on the SceneWorks-side interplay.

#### [F-200] The audio lane sits entirely outside the ladder while the gen-core doc implies coverage
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/contracts/gen-core/src/memory_strategy.rs:16-18`; `crates/audio/candle-audio-stable-audio-3/src/model.rs:2014-2028`
- **Finding:** No crate under `crates/audio/` references `memory_strategy` or `MemoryProviderContract`; every audio generator inherits the trait default (`memory_strategy_contract() -> None`); SA3 hard-rejects any `OffloadPolicy` other than `Resident`; the audio catalog runs no memory-registry conformance (both media catalogs do). The gen-core doc line "video and audio have all four [rungs]" is a vocabulary claim that reads as a coverage claim.
- **Impact:** `stable_audio_3_medium` is the largest resident audio model (~10.4 GB F32) with no fit prediction, no downtier path, and no offload; a ladder-driven consumer treats every audio render as un-gateable. Also a scope fact for the rollout: the ~130 stories' surface includes an entire lane at zero adoption.
- **Suggested fix:** Soften the doc line to mechanism-applicability, and track audio-lane adoption explicitly (SA3 already has crate-local `sampler::resource_estimate` machinery, currently consumed only by tests); wire the audio catalog into registry conformance when the first contract lands.
- **Confidence:** High

#### [F-201] Krea control contract advertises StagedResidency regardless of the loaded offload policy
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/media/mlx-gen/mlx-gen-krea/src/memory_strategy.rs:46-82,151-211`; contrast `block_memory_strategy.rs:88-99`
- **Finding:** The control contract marks `Resident | StagedResidency | BoundedDecode` Implemented and sets `synchronized_phase_release: true` unconditionally, though staging is realized only by a `Sequential` load; `safety_check` never compares the selection to the loaded `offload_policy`. The sibling base-krea contract conditions both on `OffloadPolicy::Sequential` — the two copies drifted within the week. `KreaControlMemoryScope::configure_request` also sets no `stage_residency` flag for a staged selection.
- **Impact:** A Resident-loaded control model accepts a StagedResidency selection whose promoted evidence assumes the encoder was dropped before the heavy phase; the render keeps it co-resident and the real peak exceeds the accepted prediction — accept-what-you-can't-honor.
- **Suggested fix:** Gate staged support and `synchronized_phase_release` on `spec.offload_policy == Sequential`, matching the sibling.
- **Confidence:** High on the discrepancy; Medium on reachability (a well-behaved caller aligns spec and selection).

#### [F-202] Krea's rung-4 gate admits dense+load-quantize loads that re-quantize every block per window per step
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/media/mlx-gen/mlx-gen-krea/src/block_memory_strategy.rs:35-40`; `crates/media/mlx-gen/mlx-gen-krea/src/block_stream.rs:114-117`; contrast `mlx-gen-lens/src/memory_strategy.rs:20-28`
- **Finding:** `is_streamable_spec` requires Dir + Sequential + DeferredMaterialization + no diff-patch but does not exclude `spec.quantize` over a dense snapshot; in that shape `KreaBlockStream::materialize` runs `block.quantize(bits)` on every materialized window — a bf16→packed repack per block per window per denoise step. Lens's `is_streamable_spec` deliberately excludes `spec.quantize`; krea's does not.
- **Impact:** The exact per-window-repack shape behind the known Candle 26× rung-4 slowdown, reachable on MLX — plus evidence drift: the calibration fingerprint (measured on packed turnkeys) is applied to a load shape with an extra ~0.8 GiB transient and per-window quantize compute it cannot see.
- **Suggested fix:** Add `spec.quantize.is_none()` to `is_streamable_spec` like Lens (packed snapshots still stream fine), or hoist the quantize and fingerprint the dense+quantize shape separately.
- **Confidence:** High on the mechanism; Medium on production reachability (turnkeys ship pre-packed).

#### [F-203] Krea CFG render paths materialize every DiT block twice per step under rung 4
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/media/mlx-gen/mlx-gen-krea/src/pipeline.rs:651-663,707-719,914-926,1065-1078`
- **Finding:** In the full-CFG, multiphase, and edit paths, `cond` and `uncond` each run an independent windowed pass (`forward_prepared_windowed` twice per step with the same window), so each block is re-opened and re-materialized (and, per F-202, potentially re-quantized) twice per denoise step.
- **Impact:** Doubles rung-4's dominant per-step cost on every CFG render; the sc-16352 calibration table only covers the CFG-free `krea_2_turbo` route, so the cost is also unmeasured evidence-wise.
- **Suggested fix:** Batch cond+uncond (batch=2) into a single windowed traversal, or let the windowed driver carry multiple states per window so each block materializes once per step.
- **Confidence:** High

#### [F-204] Krea-realtime has no model-local frame cap — huge clips die by SIGKILL, not typed refusal
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/media/mlx-gen/mlx-gen-krea-realtime/src/pipeline.rs:331-336`; `crates/media/mlx-gen/mlx-gen-krea-realtime/src/generate.rs:515-516`
- **Finding:** The only frame bound is gen-core's 1M-frame sanity cap (explicitly "not a model limit"); `run_ar_loop_conditioned` allocates the whole clip's noise up front (`random::normal` of the full `[c, num_frames, h, w]`) and accumulates every chunk's output. Wan enforces `MAX_WAN_FRAMES`; krea-realtime enforces nothing.
- **Impact:** A large `frames` request passes validation, loads ~28 GB of weights, then jetsams the host mid-run — the "OOM instead of a fast refusal" failure the same week's scail2 work names. The KV cache is bounded; the latent/noise/output buffers are not.
- **Suggested fix:** Add a model-local max-latent-frames check returning a typed error before staging components, mirroring Wan.
- **Confidence:** High

#### [F-205] Krea-realtime builds the full Sq×Sk mask matrix on every forward, then usually discards it
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs:57-101,481-487`
- **Finding:** `block_causal_mask` allocates and fills the full `q_pos.len() × kv_pos.len()` `Vec<f32>` and scans every cell before the `any_masked` early-out — which the code's own docs call "the common single-block AR step".
- **Impact:** At 832×480 with the bounded window, roughly a 66M-element (~260 MB) host allocation plus a 66M-iteration CPU loop per denoise forward, all wasted in the standard path.
- **Suggested fix:** Decide all-allowed analytically first and only build the matrix when a mask is actually needed.
- **Confidence:** High

#### [F-206] Candle Krea Turbo publishes rung 4 as Implemented even when load-time adapters make it unexecutable
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/media/candle-gen/candle-gen-krea/src/lib.rs:1258-1266`; `crates/media/candle-gen/candle-gen-krea/src/pipeline.rs` (`load_dit` adapter rejection)
- **Finding:** `registered_krea_turbo_memory_strategy_contract(_spec)` ignores the `LoadSpec`, so a generator loaded with LoRA/LoKr/diff-patch adapters still publishes `BoundedTransformerResidency: Implemented`; nothing checks adapters until `load_dit(stream_blocks=true)` fails with an untyped `CandleError::Msg` — after the text encoder was loaded, run, and dropped. Candle z-image solves exactly this by degrading rung 4 to `Missing` when `!spec.adapters.is_empty()`.
- **Impact:** Violates the module's "typed rejection rather than a silently different execution" rule: the selector believes rung 4 is available, burns a full TE phase, and the failure classifies as a failed job rather than an unselectable rung.
- **Suggested fix:** Make the Krea registered contract spec-aware like z-image's, or reject in `begin_memory_strategy_request`.
- **Confidence:** High

#### [F-207] Sidecar preparation re-hashes the full component per staged request — non-cancelably on Krea
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/media/candle-gen/candle-gen/src/quant/sidecar.rs:213-298,545-572,746-765`; `crates/media/candle-gen/candle-gen-krea/src/loader.rs:105-112`
- **Finding:** Every `prepare*` computes a SHA-256 digest over all packed source bytes, and for each existing artifact `validate_sidecar` re-hashes the entire sidecar payload — ~2× the component's bytes hashed even on a fully warm cache. Krea's staged routes reload components per request, and Krea uses the non-cancelable `prepare` (no `CancelFlag`), unlike z-image's `prepare_prefix_cancelable`.
- **Impact:** Order of 10+ GiB hashed per staged Krea request (DiT+TE) as uninterruptible CPU seconds added to the constrained-host path the ladder is meant to help.
- **Suggested fix:** Memoize validated sidecars in-process keyed by (dir, file mtimes/lengths); switch Krea to the cancelable variant; reserve the full payload re-hash for a corrupt-recovery path.
- **Confidence:** High

#### [F-208] SA3 resolves the adapter stack — including host Jacobi SVD — twice per generator
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/audio/candle-audio-stable-audio-3/src/model.rs:2039-2044,1982`
- **Finding:** `load_variant` resolves the full adapter plan on the host for fail-fast and throws it away; `pipeline()` resolves the identical plan again. For the `-xs` adapter types each resolution runs `svd::jacobi_svd_top_k`, documented as `O(sweeps · n² · m)` and "not amortized" on a 1.45B checkpoint.
- **Impact:** Minutes of duplicated single-threaded f64 host SVD added to cold start whenever an `-xs` adapter is configured.
- **Suggested fix:** Cache the host-resolved factor tensors (device-independent) at load and only transfer to device in `pipeline()`.
- **Confidence:** High

#### [F-209] SA3 resamples the entire reference clip before trimming to the variant cap
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/audio/candle-audio-stable-audio-3/src/pipeline.rs:464-479`; `crates/audio/candle-audio-stable-audio-3/src/model.rs:1796-1864`
- **Finding:** `validate_reference_audio` bounds shape and finiteness but not length; `prepare_reference_pcm` runs the whole caller buffer through the ~197-tap FIR before trimming to `target_frames` (≤ 380 s), scans finiteness twice, and pads the intermediate buffer across *all* source channels (`target_frames * source_channels`, ~535 MB for 8-channel) before downmixing to two.
- **Impact:** A one-hour 48 kHz stereo reference costs ~3×10¹⁰ f64 MACs plus large transient allocations before the first sampler step, inside the serialized generation mutex.
- **Suggested fix:** Resample only the needed prefix (+ filter margin) or reject over-cap references; downmix before padding; drop the duplicate finiteness scan.
- **Confidence:** High

#### [F-210] The new shared resampler pays per-tap bounds checks and weight accumulation on every frame
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/audio/candle-audio/src/dsp.rs:202-219`
- **Finding:** The Kaiser-sinc resampler (now behind Whisper, CLAP, Chatterbox, MOSS, SA3) runs ~197 f64 MACs *plus* a range check and `included_weight` accumulation per tap for every output frame, although only the first/last `half_taps` frames can overhang and the kernel is already normalized for interior frames.
- **Impact:** All audio call sites pay ~100× the old per-sample cost single-threaded; an interior/edge split would roughly halve the constant with byte-identical output.
- **Suggested fix:** Split the loop into a checked edge region and an unchecked plain-dot-product interior; consider a caller-facing input-length guard.
- **Confidence:** High on the arithmetic; Medium on wall-clock materiality.

#### [F-211] Chatterbox and CLAP resample full uncapped tracks and then keep 6–10 seconds
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/audio/candle-audio-chatterbox/src/model.rs:341-346`; `crates/audio/candle-audio-clap/src/mel.rs:126-128`; contrast `candle-audio-chatterbox/src/s3gen.rs:45-49`
- **Finding:** Both resample the entire input and then truncate; the sibling S3Gen path in the same crate deliberately caps *before* resampling. Pre-existing ordering whose cost the new resampler (F-210) just multiplied ~100×.
- **Impact:** A one-hour CLAP `embed_audio` input allocates ~690 MB of resampled f32 (plus the mono copy) to keep 10 s.
- **Suggested fix:** Apply the prefix/center cap pre-resample with a tap margin, matching s3gen.
- **Confidence:** High

#### [F-212] Inert double-escaped regex silently unenforces the runner-temp workflow policy
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `scripts/tests/test_ci_workflow_policy.py:221-224`
- **Finding:** `assertNotRegex(workflow, r"(?m)^      [A-Z][A-Z0-9_]+: \\$\\{\\{ runner\\.temp \\}\\}")` is double-escaped inside a raw string, so it matches a literal backslash and can never match a real `${{ runner.temp }}` env line — a test that cannot fail (behavior confirmed by executing the pattern).
- **Impact:** The "no job-level env var may be assigned `${{ runner.temp }}`" policy the suite claims to pin is unenforced; a regression passes CI.
- **Suggested fix:** Single-escape the pattern and add a self-check that it matches a synthetic violating line.
- **Confidence:** High

#### [F-213] The Mage edit-variants producer skips the reference-environment pins its sibling enforces
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `scripts/release/provision_mage_edit_variants.py:317-365`; contrast `scripts/release/provision_mage_oracles.py` (`_validate_reference_environment`)
- **Finding:** `provision_mage_oracles.py` refuses to produce goldens unless the interpreter is exactly 3.12.10 and all twelve reference-package pins match, and records the environment in its manifest; `provision_mage_edit_variants.py` runs the same producer for the Base/Turbo goldens with no environment validation and no `referenceEnvironment` manifest field.
- **Impact:** Goldens regenerated under a drifted torch/transformers pass all structural validation while embedding different numerics — the exact failure mode the pins exist to prevent.
- **Suggested fix:** Factor out and call `_validate_reference_environment()`; record it in the manifest.
- **Confidence:** High

#### [F-214] The listening-protocol deviation check ignores rating rows and accepts duplicates
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `scripts/audio/sa3_listening_blind.py:662-713,729-757`
- **Finding:** `preregistration_deviations` reconciles only ABX counts (contrast, null, listeners); rating rows are never reconciled against the pre-registered `screens × slots × panel` count, and `unblind` appends duplicate `(listener, screen, slot)` rating rows without rejection.
- **Impact:** A double-submitted or truncated ratings CSV shifts the MOS means without the mandated non-overridable `DEVIATION` flag — in the module whose stated purpose is that a trimmed panel cannot read as a pre-registered result.
- **Suggested fix:** Reject duplicate rating keys in `unblind`; add observed-rating-count lines to the deviation report.
- **Confidence:** High

## Low findings

#### [F-215] Deduplicate the canonical-composition predicate in evidence validation
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/contracts/gen-core/src/memory_strategy.rs:1924-1935,1972-1979` (also the auxiliary-bytes computation at `:864-892`)
- **Finding:** The "non-empty, strictly ascending engaged composition" predicate is duplicated verbatim in `validation_errors` and `optimized_eligibility`; the auxiliary-bytes computation is likewise duplicated between `predicted_peak_from_base` and `decompose_predicted_peak`.
- **Impact:** A future change to canonical form must be made twice; the string-error path and the verdict path can diverge.
- **Suggested fix:** Extract `has_canonical_composition()` and one auxiliary-bytes helper.
- **Confidence:** High

#### [F-216] Evict-then-fail-to-reload permanently poisons a resident-only Residency
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/contracts/gen-core/src/residency.rs:64-89,231-245,339-356`
- **Finding:** `ensure_warm_locked` evicts the warm pair on a `streamable` mismatch (and `run_request_scoped(stage=true)` evicts unconditionally) before knowing the loaders can rebuild; for a `Residency::resident(..)` value the reload loaders always error, so one staged/streamable request bricks every subsequent request. Latent today — current callers route resident values only through the fixed `run(..)` path — but nothing in the type prevents it.
- **Impact:** A future caller wiring `stage_residency`/`stream_transformer_blocks` through to an in-memory (single-file) generator converts a should-be-rejection into a permanently broken warm generator.
- **Suggested fix:** For resident-only sources, reject the staged/streamable request before evicting (e.g. a `can_rebuild` flag).
- **Confidence:** High on mechanism; Medium on reachability.

#### [F-217] Pick one mutex-poison policy inside Residency
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/contracts/gen-core/src/residency.rs:199-205,216-220`
- **Finding:** `with_resident_parts` recovers from a poisoned lock while `warm()` converts poison into an error — two policies for the same mutex, unexplained.
- **Impact:** After a panic mid-request, introspection reads state every execution path refuses to touch — confusing during incidents.
- **Suggested fix:** Standardize (the error path) and comment why.
- **Confidence:** High

#### [F-218] MemoryCacheSemantics' variant name lags the axes SC-16211 added
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/contracts/gen-core/src/memory_strategy.rs:749-753`
- **Finding:** The sole, field-less variant `StrategyTierParametersGeometryAndOverlay` *is* the normative cache-key statement, but SC-16211 made engaged composition an execution-identity axis and the evidence key also carries mode and backend; a provider keying its warm cache exactly as the name says omits them.
- **Impact:** A warm generator can legally (per the enum) reuse cached state across requests whose engagement compositions differ.
- **Suggested fix:** Extend/rename the variant (as a new variant, keeping compat surfaces explicit) or document why composition/mode cannot vary within one warm generator.
- **Confidence:** Medium

#### [F-219] Fault injection rides the production request surface gated only by convention
- **Category:** security
- **Severity:** Low
- **Location:** `crates/contracts/gen-core/src/generator.rs:495-500`
- **Finding:** `GenerationMemory::calibration_error_phase` is a `#[doc(hidden)]` public field on the production request that makes adopting providers fail deterministically mid-render; nothing in `validate_request` or the shared floor rejects it (deliberate design, sc-15968).
- **Impact:** Any request constructor (including a consumer deserializing user-supplied knobs) can trigger provider-internal fault paths — a typed error, but with no capability gate or audit trail.
- **Suggested fix:** Gate behind `cfg(any(test, feature = "calibration"))` or have the shared floor reject it absent an explicit harness flag.
- **Confidence:** Medium (accepted-risk observation)

#### [F-220] Composed memory routes cannot publish activation-memory anchors
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/contracts/gen-core/src/registry.rs:489-503,634-660`
- **Finding:** `memory_strategy_contract` accepts a composed route id, but `activation_memory_bytes_1024` hard-errors for any id with no generator registration — so a platform-composed route can declare a full memory contract but never the activation anchor the same caller consumes next.
- **Impact:** Composed routes adopting SC-16065-era accounting get a hard error where ordinary routes get the documented `Ok(None)` unmeasured state.
- **Suggested fix:** Accept composed ids (returning `Ok(None)` when unmeasured) or document the asymmetry.
- **Confidence:** Medium

#### [F-221] The ladder request-preamble is copy-pasted across all four Z-Image variants and three Qwen variants
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-z-image/src/model.rs:557-577`, `model_base.rs:226-246`, `model_control.rs:365-385`, `model_base_control.rs:208-228`; three qwen `generate_impl`s
- **Finding:** The ~15-line block (stage/streamable derivation, decode tiling, attention budget, block window, encoder window, three per-phase `calibration_fault` calls) is byte-identical across the four z-image variants (verified by the reviewing agent's grep) and repeated in qwen's three.
- **Impact:** The next rung, or a fix like F-183, must be applied in four/seven places; a missed sibling gives one variant a different memory lifecycle than its contract claims.
- **Suggested fix:** Hoist into a `pipeline::resolve_request_rungs(req, &contract, MODEL_ID)` helper; consider letting `impl_generator!` take the memory hooks.
- **Confidence:** High

#### [F-222] Z-Image base asset facts don't project load-time quantization, unlike its own control component
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-z-image/src/memory_strategy.rs:532-562` vs `:569-644`
- **Finding:** Base `asset_facts` sums on-disk dense bytes — exact for turnkey tiers, ~4× overstated for the supported dense-snapshot + `spec.quantize` load — while the same file projects packed bytes precisely for the control checkpoint at the same tier.
- **Impact:** Overstated `AssetBytes` skews admission (fail-closed) and calibration inputs; the in-file asymmetry invites drift.
- **Suggested fix:** Apply the packed projection to base components when load-time quant is set, or document that facts are only exact for turnkey tiers.
- **Confidence:** Medium

#### [F-223] Qwen fires the Conditioning calibration fault before any conditioning work exists
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-qwen-image/src/model.rs:380-384` (same in `model_edit.rs:414`, `model_control.rs:431`)
- **Finding:** The crate's stated convention is that conformance faults fire only after the selected phase executed, "so the propagated error proves residency cleanup"; Denoise and Decode honor it, but the Conditioning fault fires in `generate_impl` before the text encoder is even loaded.
- **Impact:** The conditioning-phase cleanup-on-error conformance case is vacuous for all three qwen providers.
- **Suggested fix:** Move the fault into the encode closure after `encode_prompt`, matching z-image.
- **Confidence:** High

#### [F-224] Qwen silently drops the selected tiling when a PiD decoder is present
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-qwen-image/src/pipeline.rs:186-190`
- **Finding:** The `(Some(pid), _)` match arm ignores a `Some(cfg)` tiling entirely; unreachable through the selector, but a hand-built `req.memory` with `use_pid` + `tile_vae_decode: true` executes an untiled PiD decode silently.
- **Impact:** The innermost seam degrades instead of rejecting; an upstream routing bug would be recorded as a bounded decode that never ran bounded.
- **Suggested fix:** Make `(Some(_), Some(_))` a typed error.
- **Confidence:** Medium

#### [F-225] Qwen's contract hides its PiD-route refusal from static introspection
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-qwen-image/src/memory_strategy.rs:120-135,179-186`
- **Finding:** The contract publishes only native edges and leaves `pid_decode_routes: None`, yet the gate validates route-aware — so a `use_pid` selection at rung ≥ 2 can never be admitted and nothing in the static declaration says so. Z-image publishes the union plus the explicit split for exactly this visibility (SC-15775).
- **Impact:** Callers discover the refusal only at admission time; conformance keying off `pid_decode_routes` sees a provider with ostensibly no PiD route while its request surface advertises `use_pid`.
- **Suggested fix:** Declare the routes and publish the union, or declare the refusal statically.
- **Confidence:** Medium

#### [F-226] Mage floor_bits enforces a hidden minimum that can diverge from the published floor table
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-mage/src/quant.rs:140-162`
- **Finding:** `floor_bits` resolves through the descriptor-visible `COMPONENT_PRECISION_FLOORS` table, then applies `.max(documented_minimum)` from separate private constants — reintroducing the invisible-substitution possibility the shared function exists to remove.
- **Impact:** If the public table is relaxed, descriptors and evidence identity advertise the relaxed tier while the packer silently keeps 8 bits.
- **Suggested fix:** Drop the `.max()`; assert the table ≥ minimums in a unit test so divergence fails loudly.
- **Confidence:** Medium

#### [F-227] Qwen control arms a block stream its contract declares unusable
- **Category:** dead-code
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-qwen-image/src/model_control.rs:261-263`
- **Finding:** `load_heavy` arms `with_block_stream` whenever the spec is streamable, but the control contract hard-codes `transformer_window_materialization = false`, so the stream can never be driven.
- **Impact:** Dead capability that reads as if the control route streams; a future "fix" could drive it without the control branch being bounded.
- **Suggested fix:** Don't arm the stream on the control route, or comment that it is intentionally inert.
- **Confidence:** High

#### [F-228] Krea control asset facts swallow errors and zero out Dir-based overlays
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-krea/src/memory_strategy.rs:106-129`
- **Finding:** `asset_facts` uses `from_spec_subdirs(...).unwrap_or_default()` (errors become zeros, unlike the sibling contract which propagates), and a `WeightsSource::Dir` control overlay maps to `overlay_bytes = 0` with no signal.
- **Impact:** Malformed snapshots or dir-shaped overlays silently understate facts, feeding the F-196 under-prediction path.
- **Suggested fix:** Propagate errors like `block_memory_strategy`; sum directory bytes for Dir overlays.
- **Confidence:** High

#### [F-229] Self-scanning source test pins exact rustfmt formatting
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-krea/src/model_control.rs:503-524`
- **Finding:** The geometry-preservation test `include_str!`s its own source and asserts exact byte sequences including a 24-space indentation run.
- **Impact:** Any rustfmt churn breaks it without behavior change (this repo has such history), while a renamed regression could still pass — it pins formatting, not the property.
- **Suggested fix:** Whitespace-insensitive matches, or an API-level seam recording the geometry used.
- **Confidence:** High

#### [F-230] mlx-gen-catalog doc comments contradict the code they annotate
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-catalog/src/lib.rs:55-60,379-393`
- **Finding:** The `PENDING_REGISTRATION_CRATES` doc contains a garbled merged sentence, and `mage_rl_is_on_the_shipped_platform_surface` keeps a comment stating Mage "must **not** reach the shipped registry until it can load" while its body asserts Mage IS shipped and the pending list is empty.
- **Impact:** The catalog surface tests are the reviewable source of truth for what ships; inverted docs will mislead the next editor of the composition root.
- **Suggested fix:** Rewrite both comments to the post-sc-14041 reality.
- **Confidence:** High

#### [F-231] Krea's decode-phase VAE load skips the loading-boundary cancel contract
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/media/candle-gen/candle-gen-krea/src/pipeline.rs:209-213`
- **Finding:** Text and DiT phases go through `enter_loading_boundary` (progress callback, then cancel check — unit-tested); the VAE phase does `check_cancel` → bare `on_progress` → `load_vae`, so a cancel raised inside the callback is unobserved until the VAE fully loads.
- **Impact:** A consumer cancelling from the loading callback still pays the full VAE load on the path built for constrained cards.
- **Suggested fix:** Route the VAE load through `enter_loading_boundary` (also gaining the Decode-phase fault-injection point).
- **Confidence:** High

#### [F-232] Krea's scope-less re-validation covers only one of the three rung parameters
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/media/candle-gen/candle-gen-krea/src/lib.rs:300-314`
- **Finding:** SC-15792 added generate-time re-validation for `transformer_window_size` precisely because a hand-built `GenerationMemory` can bypass the scope — but the same arm reads only booleans for decode and attention: a hand-built `decode_tile_edge: Some(256)` or unexpected chunk value silently executes at the provider constants (512/128, 128 Mi) instead of being rejected.
- **Impact:** The "silently different execution than selected" gap the window fix closed remains open for two of three parameters; calibration evidence for such requests describes a run that never happened.
- **Suggested fix:** Reject non-published `decode_tile_edge`/`decode_overlap` values in the same guard, matching the window treatment.
- **Confidence:** High

#### [F-233] The usize→u64 attention-budget sentinel mapping is copy-pasted four times
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/media/candle-gen/candle-gen-krea/src/transformer/mod.rs:19-33`, `transformer/block.rs:12-22`; `crates/media/candle-gen/candle-gen-z-image/src/packed_dit.rs:55-65`; canonical private copy at `crates/media/candle-gen/candle-gen/src/attention.rs:117-124`
- **Finding:** The safety-relevant `usize::MAX → u64::MAX` sentinel mapping (with its 32-bit-widening rationale comment) is re-implemented in three provider files while `candle_gen::attention` already holds the identical private helper.
- **Impact:** A future sentinel fix must land in four places.
- **Suggested fix:** Export one `budget_from_usize` from `candle_gen::attention` and delete the copies.
- **Confidence:** High

#### [F-234] A streamed Z-Image DiT reached through the plain forward defaults to an all-blocks window
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/media/candle-gen/candle-gen-z-image/src/packed_dit.rs:610-626`
- **Finding:** `forward_with_attention_plan` (and via it `forward`) passes `self.cfg.n_layers.max(1)` as the window, so a block-streamed model driven through any path that doesn't thread the admitted window materializes the whole trunk in one window — correct output, resident-scale peak, no error. Latent (only `forward_with_memory` is called on streamed models today).
- **Impact:** A future edit/control route handed a streamed model silently forfeits the entire rung-4 bound — the silent-vacuous-bound shape this week's tests hunt elsewhere.
- **Suggested fix:** Default the plain path to `DEFAULT_TRANSFORMER_WINDOW` for a Streamed trunk, or error when driven without an explicit window.
- **Confidence:** High on the code; impact contingent on a future caller.

#### [F-235] SCAIL-2 judges edge bounds on raw geometry but area on the lattice projection
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/media/candle-gen/candle-gen-scail2/src/pipeline.rs` (`reject_unrenderable_geometry`, `resolve_render_size`)
- **Finding:** sc-16197 made the area cap measure the aligned (rendered) geometry, but the min/max edge check in the same function still measures the raw sentinel-resolved size — a 1290×704 driving clip is rejected in auto-size mode though the rendered 1280×704 is in-envelope. The function also reformats the error text `reject_over_area` already produced.
- **Impact:** Spurious rejections for slightly off-lattice source media (the message offers the explicit-size workaround, so no one is stranded); possibly deliberate policy, but uncommented.
- **Suggested fix:** Apply edge bounds to the aligned geometry for the sentinel path, or comment the asymmetry as policy; reuse the existing error text.
- **Confidence:** Medium

#### [F-236] moss-sfx `window_seconds` ignores its parameter and double-buffers the tail copy
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/audio/candle-audio-moss-sfx/src/pipeline.rs:223-225,302-305`
- **Finding:** `window_seconds(&self, _seconds: f32)` returns a fixed 30 s window regardless of its argument (the revert of the collapsed-solver bug is justified; the signature now lies), and `full[..out_len].to_vec()` copies the always-30 s decode buffer where `truncate` is free.
- **Impact:** A 3 s SFX request decodes 30 s and allocates it twice; the dangling parameter invites re-introducing the collapse bug.
- **Suggested fix:** Drop the parameter; `truncate` in place; note the deliberate peak increase.
- **Confidence:** High

#### [F-237] AceStep recomputes a request-invariant timbre encode per generation
- **Category:** efficiency
- **Severity:** Low
- **Location:** `crates/audio/candle-audio-acestep/src/condition.rs:294-296`
- **Finding:** The silence-latent timbre conditioning (750 frames through a 4-layer hidden-2048 encoder) has constant inputs and weights yet runs on every `encode_context`, materializing 750×750 attention per head per layer.
- **Impact:** Tens of MB of transient allocations and a full encoder pass per request for a constant `[1,1,hidden]` value.
- **Suggested fix:** Cache the pooled vector behind a `OnceLock`.
- **Confidence:** High (invariance); Medium (magnitude)

#### [F-238] AceStep dead `project` parameter and three stale special-token doc sites
- **Category:** dead-code
- **Severity:** Low
- **Location:** `crates/audio/candle-audio-acestep/src/condition.rs:194-201,265`; `crates/audio/candle-audio-acestep/src/pipeline.rs:659`
- **Finding:** Both remaining `Encoder::forward` call sites pass `project: true`; the `false` branch and three comments describe the `special_token` this very diff removed.
- **Impact:** Unreachable branch plus docs contradicting the shipped conditioning path.
- **Suggested fix:** Remove the parameter; update the comments.
- **Confidence:** High

#### [F-239] Range-check SA3 `guidance_eta` at request validation
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/audio/candle-audio-stable-audio-3/src/model.rs:1498-1503,1895`; `crates/audio/candle-audio-stable-audio-3/src/sampler.rs:1078-1080`
- **Finding:** `validate_request` checks that `guidance_eta` accompanies `method=apg` but not its range; `apg_scale = 1.0 - eta` then fails only at generate time — after the cold-start hash and weight load — with a message naming a derived internal value, contrary to the crate's own stated validation philosophy.
- **Impact:** `guidance_eta: -0.5` costs a multi-GB cold start before failing with an error about "APG scale".
- **Suggested fix:** Add the domain check to `validate_request` beside the existing eta/method pairing.
- **Confidence:** High

#### [F-240] Two resample call sites stringify the typed AudioError the same diff introduced
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/audio/candle-audio-chatterbox/src/mel24.rs:170-171`; `crates/audio/candle-audio-moss-tts-realtime/src/codec.rs:1105-1106`
- **Finding:** Two `dsp::resample` sites map the typed error into `Error::Msg(error.to_string())` while sibling sites in the same diff propagate it typed.
- **Impact:** Callers lose validation-vs-device distinction on exactly two of ~8 converted paths.
- **Suggested fix:** Convert to the typed pattern.
- **Confidence:** High

#### [F-241] Chatterbox duplicates the reference-preparation preamble and re-copies mono buffers
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/audio/candle-audio-chatterbox/src/model.rs:445-453,537-550`; `crates/audio/candle-audio-chatterbox-ve/src/model.rs:147,163-166`
- **Finding:** The validate→cancel→find-reference→prepare block (carrying the just-fixed stereo-correctness invariant) is copy-pasted at two sites, and the embedder's defensive `to_mono` is a full-length copy of a buffer already guaranteed mono.
- **Impact:** Editing one preamble and not the other reintroduces the interleaved-stereo bug just fixed; 3–4 avoidable full-clip copies on the clone path.
- **Suggested fix:** Extract one `prepared_reference(request)` helper; make `to_mono` skip when `channels == 1`.
- **Confidence:** High

#### [F-242] The Kaiser fallback path recomputes a Bessel kernel per output frame, silently
- **Category:** efficiency
- **Severity:** Low
- **Location:** `crates/audio/candle-audio/src/dsp.rs:34,165-199`
- **Finding:** When `phase_count × taps` exceeds the 4.19M-element table cap (unusual but reachable rate pairs), the fallback recomputes a full kernel per output frame; the precomputed table is also `phase_count` separate `Vec<f64>`s.
- **Impact:** An oddball but valid sample rate turns seconds of audio into minutes of CPU with no signal.
- **Suggested fix:** Error explicitly above the cap or bound the fallback; flatten the table.
- **Confidence:** Medium

#### [F-243] SA3 cold-start docs quote the smalls' hashing cost for the medium variants
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/audio/candle-audio-stable-audio-3/src/model.rs:1163-1169,1968-1976`
- **Finding:** Both the `verify_snapshot_identity` doc and the `pipeline()` comment quote "~6.9 s over 3.45 GB" — the smalls' pin set; medium pins ~10.44 GB (~21 s per pass at the quoted rate) and the pass deliberately runs twice, so medium's cold start carries ~42 s of hashing the arithmetic hides.
- **Impact:** The documented trade-off understates the cost for the two 1.45B ids — the exact number a future "optimize cold start" decision will consult.
- **Suggested fix:** State both figures where the double-verification trade-off is argued.
- **Confidence:** High

#### [F-244] SA3 dtype-policy docs omit the CPU exception t5gemma applies
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/audio/candle-audio-stable-audio-3/src/pipeline.rs:122-126` vs `crates/audio/candle-audio-stable-audio-3/src/t5gemma.rs:36-46,644-646`
- **Finding:** `pipeline.rs` states "BF16 weights, F32 compute, one BF16 rounding at the raw-embedding boundary", but `t5gemma.rs` applies that rounding only on Metal/CUDA and keeps CPU raw output F32.
- **Impact:** Someone debugging a CPU-vs-Metal embedding diff from the pipeline doc will conclude parity is broken when it is policy.
- **Suggested fix:** Add the per-location caveat to the policy paragraph.
- **Confidence:** Medium

#### [F-245] check-workspace's cfg(test) parser mis-tracks bare `<` comparisons
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `scripts/check-workspace.py` (`_cfg_item_end`)
- **Finding:** `_cfg_item_end` counts every top-level `<` as opening a generic list; a `cfg(test)` item containing a bare comparison leaves `angle_depth` nonzero and the span is dropped — so that test-only item is *not* blanked from the adoption-evidence stream.
- **Impact:** A `DecodeRoutes::new`/`validate` call confined to test code could satisfy the sc-15775 adoption gate — the fail-open direction (the trigger stream is documented fail-closed).
- **Suggested fix:** Only treat `<` as a generic opener after identifier/`::` context, or reset depth at `;`/`{` for semicolon items.
- **Confidence:** Medium

#### [F-246] SA3 snapshot revisions are hardcoded ~20 times across real-weights jobs
- **Category:** redundant
- **Severity:** Low
- **Location:** `.github/workflows/real-weights.yml` (e.g. L551, L583-584, L631, L697-700, plus CUDA twins)
- **Finding:** Each SA3 job re-hardcodes `$RUNNER_TEMP/<model>/<40-hex-revision>`; the same literal appears up to 6 times, and a TOML-only revision bump leaves a directory whose name lies about its contents (the marker wins on the empty-dir path).
- **Impact:** Revision bumps need ~20 coordinated edits; a missed one undermines the dirname-as-revision convention the verify scripts fall back on.
- **Suggested fix:** Derive the path from the manifest (small `snapshot_path.py` helper) or hoist one env value per model.
- **Confidence:** High

#### [F-247] Mage oracle verification hashes every file twice
- **Category:** efficiency
- **Severity:** Low
- **Location:** `scripts/release/provision_mage_oracles.py:473,521-523`
- **Finding:** `verify()` computes `_sha256` per file in `_validate_files`, then `_validate_manifest_file_record` recomputes the same digests.
- **Impact:** Multi-hundred-MB VAE bundles hashed twice on every run and cache verification.
- **Suggested fix:** Compare against the already-computed records.
- **Confidence:** High

#### [F-248] Queued real-weights dispatches can be silently evicted
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `.github/workflows/real-weights.yml:18-20`
- **Finding:** The concurrency group keys on `github.ref` with `cancel-in-progress: false`; GitHub keeps at most one *pending* run per group, so burst-dispatching three profiles evicts the middle one — the drop-the-middle pathology ci.yml's own comment documents and fixed for pushes.
- **Impact:** An operator queuing several profile runs loses one with no failure signal (weekly schedule unaffected).
- **Suggested fix:** Key the group on `${{ github.ref }}-${{ inputs.profile || 'schedule' }}`.
- **Confidence:** Medium

#### [F-249] Move the ~190-line story narrative off the SA3 CI step
- **Category:** readability
- **Severity:** Low
- **Location:** `.github/workflows/ci.yml` ("Test Stable Audio 3 weight-free quality gates" step)
- **Finding:** One 11-flag `run:` command carries ~190 lines of story-by-story history; the load-bearing invariant is already enforced by `test_sa3_ci_target_coverage.py`.
- **Impact:** Genuine policy changes get lost in narrative diffs of a 723-line workflow.
- **Suggested fix:** Keep the 10-line invariant + pointers; move the history into the migration docs it already cites.
- **Confidence:** High

#### [F-250] check-review-findings turns a force-pushed base into a hard CI failure
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `scripts/check-review-findings.py:84-92,171-175`
- **Finding:** `git cat-file -e` with `check=True` on a `github.event.before` SHA that no longer exists locally raises and fails the `changes` job with "invalid base revision".
- **Impact:** A force-push or GC'd before-SHA reddens the job that gates every lane, for a reason unrelated to finding ids.
- **Suggested fix:** On `CalledProcessError`, warn and fall back to no-base validation (same posture as `workflow_dispatch`).
- **Confidence:** Medium

#### [F-251] The SAME-S/SAME-L clean-checkout `.venv` allowance can never match
- **Category:** dead-code
- **Severity:** Low
- **Location:** `scripts/reference/sa3_same_reference.py:280-288`; `scripts/reference/sa3_same_l_reference.py:143-151`
- **Finding:** `git status --porcelain --untracked-files=all` expands untracked directories per-file, so no line ends with `" .venv/"`; the allowance is dead (fails safe, over-strict). `sa3_reference.py:452-459` implements the intent correctly with a prefix check.
- **Impact:** Operators with a non-ignored `.venv` get a misleading wall of "not clean" entries.
- **Suggested fix:** Adopt the prefix-based check (see F-252).
- **Confidence:** High

#### [F-252] Consolidate the six divergent copies of the SA3 reference-script helpers
- **Category:** redundant
- **Severity:** Low
- **Location:** `scripts/reference/sa3_same_reference.py`, `sa3_same_l_reference.py`, `sa3_chunked_autoencoder_reference.py`, `sa3_small_music_provider_reference.py`, `sa3_text_reference.py`, `sa3_primitives_reference.py`
- **Finding:** `sha256_file` (6 copies), `validate_upstream` (6 copies, four different dirty-checkout policies), `portable_values`/`tensor_records` (3 identical copies each) are duplicated per script, though siblings already demonstrate importing from `sa3_reference`.
- **Impact:** The divergence already produced F-251; future fixes will land in some copies and not others.
- **Suggested fix:** Move shared helpers into `sa3_reference.py`; keep intentional policy differences explicit.
- **Confidence:** High

#### [F-253] `environ or os.environ` defeats explicit environment isolation
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `scripts/reference/sa3_reference.py:227-231`
- **Finding:** Truthiness instead of `is None`: an explicitly passed empty mapping falls back to the real process environment — the opposite of the caller's isolation intent.
- **Impact:** A test passing `environ={}` reads the operator's real `SA3_*_SNAPSHOT` vars and can pass/fail for the wrong reason.
- **Suggested fix:** `environ = os.environ if environ is None else environ`.
- **Confidence:** High

#### [F-254] Drop the `--components` flag that can never select a subset
- **Category:** redundant
- **Severity:** Low
- **Location:** `scripts/reference/sa3_reference.py:795-801,1017-1022`
- **Finding:** `generate` hard-fails unless the selection is exactly all eight components, yet the CLI advertises per-component choices; the `unknown` branch is additionally dead because argparse `choices` forbids unknown values.
- **Impact:** Misleading CLI surface; dead validation code.
- **Suggested fix:** Remove the flag.
- **Confidence:** High

#### [F-255] Anchor the two cwd-relative default output paths to the repo root
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `scripts/reference/sa3_small_music_provider_reference.py:427-431`; `scripts/reference/sa3_chunked_autoencoder_reference.py:499-503`
- **Finding:** These two default `--output` to a relative `Path("docs/migration/...")` while every sibling anchors at the repo root; run from any other cwd, verify looks in (and generate writes to) the wrong place.
- **Impact:** Confusing "missing manifest" failures or artifacts outside the committed directory.
- **Suggested fix:** Use the ROOT-anchored default.
- **Confidence:** High

#### [F-256] Validate listening-manifest `file` entries before joining paths
- **Category:** security
- **Severity:** Low
- **Location:** `scripts/audio/sa3_listening_blind.py:455-479,596-616`
- **Finding:** `materialize` copies `source_dir / take["file"]` with `file` taken straight from the operator-authored JSON manifest; nothing rejects `"../../…"` values, so the read side can traverse outside `--source-dir` — including, ironically, copying the unblinding key into the panel directory.
- **Impact:** A crafted/corrupted manifest copies arbitrary readable files into the blinded panel.
- **Suggested fix:** Reject values where `Path(f).name != f` at index/assign time.
- **Confidence:** High

#### [F-257] Give `DecodeRoutes` its error-joining constructor instead of three verbatim wrappers
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-z-image/src/memory_strategy.rs:384-391`; `mlx-gen-qwen-image/src/memory_strategy.rs:38-45`; `mlx-gen-krea/src/memory_strategy.rs:24-31`
- **Finding:** The 8-line `fn decode_routes(provider_id) -> CoreResult<DecodeRoutes>` wrapper is byte-identical in three crates; fingerprint naming meanwhile has no convention (date-stamped, story-stamped, `-vN`, shape-suffixed all mixed).
- **Impact:** `DecodeRoutes` was hoisted so adopters "cannot get it wrong quietly" — the wrapper should ride along; 130 free-form fingerprints will be un-lintable.
- **Suggested fix:** Add a `CoreResult` constructor in `mlx-gen-pid`; document a fingerprint grammar (superseded if F-185's typed identity lands).
- **Confidence:** High

#### [F-258] Evidence-key `mode`/`backend` are stringly-typed shadows of existing enums
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/contracts/gen-core/src/memory_strategy.rs:1859-1872` vs `:1700-1727`
- **Finding:** `MemoryEvidenceKey.mode: String` and `.backend: String` shadow the typed `MemoryMode` and `MemoryBackendRealization::backend_id()`; nothing enforces the casing convention the tests use.
- **Impact:** Two providers spelling one mode differently produce silently disjoint evidence cells — an unfalsifiable "no evidence" state rather than an error.
- **Suggested fix:** Key on `MemoryMode` (already `Eq`) or a canonical `as_key()`.
- **Confidence:** High

#### [F-259] KreaMemoryScope and LensMemoryScope carry a `finished` flag they never check
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `crates/media/mlx-gen/mlx-gen-krea/src/block_memory_strategy.rs:211-292`; `crates/media/mlx-gen/mlx-gen-lens/src/memory_strategy.rs:192-268`
- **Finding:** The control scope guards every method with `ensure_active()`; the two scopes written this week from the same template carry the flag but never consult it in `configure_request`/`enter_phase`/`materialize_transformer_window`, and hard-code layer counts (28/24) instead of deriving them. (The systemic fix is F-197; recorded separately because these two are one-line guards fixable immediately.)
- **Impact:** A finished scope silently accepts further configuration on exactly the two newest copies.
- **Suggested fix:** Add the guard now; derive block counts from config.
- **Confidence:** High

## Informational

#### [F-260] Several run-context and evidence fields have no in-repo readers
- **Category:** dead-code
- **Severity:** Info
- **Location:** `crates/contracts/gen-core/src/memory_strategy.rs:1721,1725-1726,2025-2031`
- **Finding:** `MemoryRunContext::cache_state`, `evidence_revision`, and `has_reference` are only ever constructed in this repo; `MemoryRejection`, `MemoryEvidence` (beyond gen-core), and `is_above_selected_tier` have zero in-repo consumers. Plausibly SceneWorks-side vocabulary (the caller owns evidence and selection), which this repo cannot confirm.
- **Impact:** If SceneWorks also does not read them, they are dead contract weight every provider must populate.
- **Suggested fix:** Cross-check SceneWorks usage; drop or mark "caller-facing, informational".
- **Confidence:** High on the in-repo facts; Low on actual deadness.

#### [F-261] Multi-region audio testkit fallback yields a misleading failure message
- **Category:** readability
- **Severity:** Info
- **Location:** `crates/contracts/gen-core-testkit/src/audio_generator.rs` (`check_multi_region_audio_edit`)
- **Finding:** A provider advertising `AudioEditRegions` with neither `Inpaint` nor `Repaint` gets `Inpaint` forced and then fails with "validate() rejected a valid two-region edit" — the request was not valid for that surface.
- **Impact:** A future provider with an exotic mode surface debugs a phantom bug.
- **Suggested fix:** Fail with a dedicated "no bounded-span edit mode advertised" message.
- **Confidence:** High

#### [F-262] Comment the hardcoded `use_pid=true` in the warm loader
- **Category:** readability
- **Severity:** Info
- **Location:** `crates/contracts/gen-core/src/residency.rs:266`
- **Finding:** `(self.loaders.load_heavy)(true, streamable)` — the `true` is the deliberate "warm residents load the PiD superset once" policy, but the rationale lives only in a backend crate; in the contract crate it reads like a bug. Verified correct.
- **Impact:** A future "fix" threading the request's `use_pid` would silently drop the PiD engine from warm providers.
- **Suggested fix:** One-line comment naming the policy.
- **Confidence:** High

#### [F-263] Sole `unreachable!()` in the new mage code where a typed error would do
- **Category:** readability
- **Severity:** Info
- **Location:** `crates/media/mlx-gen/mlx-gen-mage/src/attention.rs:274-280`
- **Finding:** Guarded and currently correct, but the only `unreachable!` in the four crates' new code, on the render path, in a repo whose convention is typed errors.
- **Suggested fix:** `parts.pop().expect(..)` or fold into the concatenate arm.
- **Confidence:** High

#### [F-264] `fold_block_sequence` has one caller while its twin loop is re-inlined beside it
- **Category:** redundant
- **Severity:** Info
- **Location:** `crates/media/candle-gen/candle-gen-krea/src/transformer/mod.rs:137-148,305-317`
- **Finding:** Documented as serving "the resident trunk paths" (plural) but called only by `forward_edit_with_memory`; the resident arm of `forward_with_memory` re-implements the identical per-block cancel+forward loop inline.
- **Suggested fix:** Route the resident arm through it.
- **Confidence:** High

#### [F-265] Krea-realtime restages all heavy components on every generate
- **Category:** efficiency
- **Severity:** Info
- **Location:** `crates/media/mlx-gen/mlx-gen-krea-realtime/src/t2v.rs:696-717,847-868`
- **Finding:** Every `generate` reloads UMT5, the 14B DiT, and the VAE, reinstalls adapters, and for a dense+quantize spec would re-run full in-memory DiT quantization per request; `stage_components` also opens the transformer weights twice. Family-consistent with scail2 (28 GB models can't sit warm), so an observation, not a defect.
- **Suggested fix:** If the dense+quantize tier ever becomes a real path, cache the built transformer or require the pre-packed turnkeys.
- **Confidence:** Medium

#### [F-266] SA3 was inserted mid-catalog-order rather than appended
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `crates/audio/candle-audio-catalog/src/lib.rs:70-77` and the three bundle surface tests
- **Finding:** The six SA3 ids register between `acestep` and `moss_tts_realtime`, changing the existing ordered surface against the "later stories extend these exact assertions" convention. All four ordered-surface tests were updated consistently; the three runtime bundles otherwise show zero copy-paste drift.
- **Suggested fix:** None required; note the grouping rationale if intentional.
- **Confidence:** High

#### [F-267] SHA-256 pinning forecloses user-supplied SA3 fine-tunes; the unused T5Gemma decoder is byte-required
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `crates/audio/candle-audio-stable-audio-3/src/model.rs:474-701`; `crates/audio/candle-audio-stable-audio-3/src/weights.rs:468-550`
- **Finding:** Every load path re-authenticates against hardcoded pins (deliberate, follows the mage precedent for byte-indistinguishable siblings), and the inventory gate requires the exact 206 never-read decoder tensors (~620 MB on disk). Both deliberate; recorded because the descriptors don't advertise these capability restrictions.
- **Suggested fix:** If fine-tune support is ever wanted, a new unpinned provider id — not a loosened gate.
- **Confidence:** High

#### [F-268] The resampler swap is an unrecorded numerics boundary
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `crates/audio/candle-audio/src/dsp.rs:104-143`
- **Finding:** The Kaiser-sinc resampler intentionally differs numerically from the six linear interpolators it replaced; cached voice embeddings/tone-color vectors computed under the old code are no longer reproducible, and no migration note records it.
- **Suggested fix:** One line in `docs/migration/` naming the swap as a numerics boundary.
- **Confidence:** Medium

#### [F-269] The `real_weights` CI lane is computed but consumed nowhere
- **Category:** dead-code
- **Severity:** Info
- **Location:** `scripts/ci/select_lanes.py:13-23`; `.github/workflows/ci.yml` `changes` outputs
- **Finding:** `select_lanes` computes a `real_weights` lane and the changes step writes it to `GITHUB_OUTPUT`, but no job output declares or reads it; `real-weights.yml` does no path selection. Pre-existing.
- **Suggested fix:** Wire it as an informational annotation or comment it as intentional.
- **Confidence:** High

#### [F-270] pip installs outside the mage venv pin versions but not hashes
- **Category:** security
- **Severity:** Info
- **Location:** `.github/workflows/real-weights.yml` (~15 `pip install "huggingface_hub==1.20.1"` sites); `.github/workflows/ci.yml:120-122`
- **Finding:** Only the mage-reference venv uses `--require-hashes`; the other installs are version-pinned but hash-unpinned on persistent self-hosted runners. Triggers are trusted-only, so hardening rather than a hole.
- **Suggested fix:** A small shared hashed constraints file.
- **Confidence:** Medium

#### [F-271] Hoist the per-tensor `__import__` out of the SA3 serialization loop
- **Category:** efficiency
- **Severity:** Info
- **Location:** `scripts/reference/sa3_reference.py:520-523`
- **Finding:** `_save_tensors` re-resolves `safetensors.torch.save` via `__import__` inside the per-tensor loop.
- **Suggested fix:** Import once.
- **Confidence:** High

#### [F-272] Inputs-contract literals duplicate their own constants
- **Category:** redundant
- **Severity:** Info
- **Location:** `scripts/reference/sa3_dit_reference.py:56-65` vs `:299-308`; `scripts/reference/sa3_sampler_guidance_reference.py:76-83` vs `:199-206`
- **Finding:** Both scripts define `EXPECTED_INPUTS` for `verify`, then rebuild the identical dict inline in `generate`.
- **Suggested fix:** `"inputs": EXPECTED_INPUTS`.
- **Confidence:** High

#### [F-273] Verifier failure messages go to stdout / raw tracebacks
- **Category:** readability
- **Severity:** Info
- **Location:** `scripts/release/verify_mage_candle_oracles.py:343`; `scripts/release/provision_mage_edit_variants.py:317-400`
- **Finding:** One verifier prints FAILED to stdout (siblings use stderr); another lets `RuntimeError` escape as a raw traceback in verify mode. Exit codes are correct in both, so CI is unaffected.
- **Suggested fix:** Route to stderr; catch and return 1.
- **Confidence:** High

## Themes and systemic observations

1. **The contract is strong; the adoption template is the risk.** Two independent passes over `gen-core/src/memory_strategy.rs` found its ladder arithmetic, rung ordering, engagement policy, block-window math, and unit handling clean and mutation-tested. Every High finding except F-188–F-191 is about the *seams*: hand-copied gates, translations, and scopes that have already diverged three ways in seven adopters. The `gen_core::block_window` hoist proves the team knows how to fix this class; do the same for the gate (F-182/F-183), the translation (F-187), and the scope core (F-197) before the ~130-story rollout multiplies the current defect rate.
2. **The conformance testkit is the rollout's only automated guard, and today it is nearly vacuous behaviorally** (F-184). Every hoist above should land with a testkit probe in the same story so the class of defect dies structurally, not statistically.
3. **The sibling-gap pattern is alive** (consistent with this repo's review history): krea-realtime missed the off-grid and frame-cap family rules its siblings enforce (F-188, F-204); the krea control contract drifted from base-krea within one week (F-201); qwen vs z-image differ on tier checks, scope guards, fault timing, and asset-fact semantics. When fixing any one of these, fix its siblings in the same story.
4. **Asset facts and evidence keying need one owner.** Four different overlay-sizing behaviors (F-194–F-196), stringly key fields (F-258), free-form fingerprints (F-257), and the missing load-shape axis (F-185) all say the same thing: the evidence ABI is half-typed. Typing it now is cheap; at story 100 it is not.
5. **The audio half of "memory week" trades memory and CPU up** — universal 197-tap resampler (F-210), full-clip resampling before caps (F-209, F-211), 30 s SFX windows (F-236), doubled adapter SVD (F-208) — while the audio lane sits entirely outside the ladder (F-200). Fine as a sequencing choice, but worth stating in the epic rather than leaving implicit.
6. **CI/tooling discipline is high** (fork-PR isolation verified, cache poisoning mitigated, anti-vacuity self-tests mostly real), with two watch items: gates that cannot fail (F-212, and the F-184 conformance early-return) and the docs-lane classification hole (F-191). A "can this test actually fail?" mutation pass over new gates would have caught both.

## Coverage notes

- **Reviewed:** the full `8b053451..HEAD` week across all changed areas — gen-core memory/residency contracts (whole-file), all seven ladder adoptions on both backends (whole-module), krea-realtime, scail2, lens, mage, wan/sdxl diffs, the SA3 port (whole-crate), candle-audio family diffs, all three runtime bundles + both media catalogs + audio catalog, CI workflows, lane selection, release/reference/listening tooling, and the migration docs. Root config (`Cargo.toml`, `.cargo/config.toml`, `AGENTS.md`, `CONTRIBUTING.md`) reviewed directly.
- **Verified independently:** all ten High findings were re-confirmed against the code by the coordinating reviewer (grep/read), including the budget arithmetic (F-192) and the non-ignored fixture test behind F-191. Gates run on the tree at review time: `check-workspace.py`, `check_docs.py`, `check-review-findings.py`, `cargo fmt --all --check`, the 292 Python tooling tests, and the 487 gen-core contract/testkit tests — all green.
- **Not reviewed:** LLM crates (`core-llm`, `mlx-llm`, `candle-llm` — zero changes in the window); pre-existing code in unchanged providers beyond what the changed code touches; real-weight behavior (no weights were loaded — findings about runtime peaks rest on code reading plus the in-repo calibration tables); the SceneWorks consumer side (several findings — F-194, F-199, F-260 — have consequences contingent on its read sites and are marked accordingly); vendored/generated content (`Cargo.lock`, committed reference `.safetensors` artifacts were treated as opaque).
- **Numbering:** F-182..F-273 allocated per the sc-16542 repository-wide registry; the allocation row is added to `docs/code-review-finding-allocations.tsv` in the same change as this document.



