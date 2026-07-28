# SC-15449: shared image-memory contract migration

## Authority boundary

`gen_core::image_memory` is the tensor-neutral provider contract for image-memory planning.

- Providers own static strategy capability, lifecycle hooks, formula shape, backend realization,
  load-exact asset facts, and `IMAGE_MEMORY_CALIBRATION_ABI` plus a content fingerprint.
- Manifest/generated evidence owns coefficients, calibrated envelopes, provenance, tier/mode/overlay
  coverage, geometry, and the exact tile/chunk/window candidates exercised.
- The SceneWorks worker owns live budget and reclaimable accounting, ordered least-cost selection,
  fallback/rejection, telemetry, and user advice.
- A provider safety gate is defense in depth. It returns only `Accept` or `Reject`; it cannot silently
  replace the selected strategy, parameter values, or numeric tier.

The selected `ImageMemoryNumericTier` is immutable within `ImageMemorySelection`. Memory strategy
selection must never cross BF16/FP32/Q8/Q4/NVFP4 or another numeric regime.

## Compatibility and adoption

Existing providers require no source changes. They inherit:

- `Generator::image_memory_contract() -> None`;
- a resident-only safety default (optimized selections are rejected);
- `Generator::begin_image_memory_request() -> Ok(None)`; and
- no pre-load `ImageMemoryRegistration`.

`ImageMemoryProviderContract::compatibility_default` expresses that state explicitly: Resident is
implemented, all optimized rungs are Missing, and calibration is absent. It preserves existing
resident behavior without claiming an optimized fit.

An adopting provider:

1. returns a five-rung contract from a separate `ImageMemoryRegistration`;
2. exposes the same contract from the loaded generator;
3. implements the request-scoped lifecycle scope;
4. changes its calibration fingerprint whenever layout, quantization floors, or execution structure
   invalidate evidence; and
5. runs `gen_core_testkit::image_memory_conformance`.

`begin_image_memory_request` returns an executable request scope. Its `configure_request` method
translates the shared selection into any provider-native request controls; `finish` is called for
success, cancellation, and error. Dropping Krea's scope without finishing still synchronizes the
device as a last-resort cleanup guard.

The separate registration is intentional: adding a field to every existing `ModelRegistration`
would create provider-wide churn. `ProviderRegistry::image_memory_contract` returns `Ok(None)` for a
known non-adopter and an error for an unknown id or malformed adopted contract.

All named runtime bundles expose the contract at `runtime_{cpu,cuda,macos}::image_memory`.

## Ladder and backend realizations

The stable order is Resident, Staged residency, Bounded decode, Bounded attention, then Bounded
transformer residency. Rungs are cumulative unless evidence verifies a cheaper equivalent. Each
implemented parameterized rung owns a non-empty production domain: decode owns tile edge and overlap,
attention owns chunk size, and transformer residency owns block-window size. A selection supplies all
parameters required by its rung and cheaper cumulative rungs; missing, out-of-domain, or irrelevant
parameters are rejected.

Candle/CUDA realization describes device residency, host-backed weights, and optional host-to-device
block materialization. MLX/Metal realization instead describes bounded wired residency,
lazy/mmap-backed materialization, explicit evaluation/synchronization, and cache eviction. MLX
providers are not required to pretend unified memory performs CUDA-style transfers.

Staged residency requires Conditioning, Denoise, and Decode hooks with synchronized phase release.
The remaining constrained rungs require their corresponding decode-tiling, attention-chunking, or
transformer-window hook.

Request scopes must finish exactly once for success, cancellation, or error. Cache identity includes
strategy, tier, parameters, geometry, and overlay. Warm runs revalidate the budget and reapply
request-scoped state rather than inheriting the prior request's selection. Cancellation and error
paths synchronize and release active phases/windows.

## Evidence eligibility

The five conformance states and six evidence dimensions are represented directly. An optimized rung
is eligible only when:

- conformance is Verified;
- all six dimensions are Satisfied;
- the evidence ABI and fingerprint exactly match the current provider contract; and
- the worker has already matched the fully qualified evidence key (resolved route, backend, installed
  tier, mode, overlay, geometry, strategy, and exact parameters).

Unknown, stale, fingerprint-mismatched, and out-of-envelope records remain unverified and cannot
select an optimized fit. Exact budget boundaries fit (`predicted_peak <= effective_budget`).
Rejections may include a smaller verified geometry only when evidence actually measured it.

Numerical parity is `Exact` where operation ordering permits it. Otherwise evidence names a
deterministic tolerance metric and limit or a versioned golden fixture; “looks similar” is not a
contract. Verified optimized evidence also requires an observed peak and an explicit passing parity
result. Empty metrics/fixtures, NaN/infinite/negative tolerances, failed parity, and unexecuted parity
all prevent optimized eligibility.

## Reconciliation of existing estimators

### Krea Candle/CUDA

Krea phase curves map to `ImageMemoryFormulaKind::PhaseEnvelope`. Existing measured coefficients and
boundaries remain manifest evidence; they do not move into the provider crate. The worker evaluates
those unchanged curves through the shared selector. `krea_2_turbo` is the first adopter: the CUDA
Candle catalog returns its five-rung provider contract with calibration fingerprint
`krea-turbo-cuda-phase-curves-v1`. The provider contributes lifecycle capability, Candle/CUDA
realization, asset facts, and that fingerprint. Its local gate may reject a shared choice but must not
pick a different rung or tier. Raw and Edit remain compatibility-default (`Ok(None)`) until their
separate mode/envelope evidence is reconciled. The CPU Candle catalog also returns `Ok(None)` for
Turbo; a CUDA realization is never exposed by a named CPU bundle.

Krea's request scope maps the cumulative rungs onto the existing, measured
`GenerationMemory::{tile_vae_decode, chunk_attention, stream_transformer_blocks}` controls, using
the production 512/128 decode tile, 128 Mi-element attention budget, and one-block transformer
window. It accepts only sequentially loaded ordinary Turbo text-to-image. Reference/img2img, Edit,
PiD, and multi-phase requests remain outside that evidence envelope and are rejected before render.

### Generic MLX

The on-disk/load-exact weight sum plus constant headroom maps to
`ImageMemoryFormulaKind::AssetBytesPlusHeadroom`. Until current-environment evidence and a matching
provider fingerprint exist, it remains Implemented/unverified and cannot authorize an optimized fit.
The worker, not each provider, makes the final decision.

### Mage-Flow MLX

Mage's request-aware estimator maps to `PhaseEnvelope` or `Affine` over the declared geometry and
strategy variables. Its calibrated coefficients move to generated evidence while provider-specific
structure and asset facts remain in the provider. The local estimator stays only as a rejecting
safety gate until parity tests show the shared prediction and local prediction agree throughout the
production envelope. No safety gate is removed before those parity tests pass.

This path also demonstrates that a non-Krea provider adopts the contract without copying Krea's
selector.
