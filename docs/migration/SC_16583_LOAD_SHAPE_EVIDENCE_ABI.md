# SC-16583: load-shape evidence ABI 2

`LoadShape` is now a required typed axis of the memory-calibration handshake and evidence key.
`MEMORY_CALIBRATION_ABI` is **2**. There is no fallback from ABI 2 to ABI 1 and no inference of
shape from calibration-fingerprint suffixes.

## Contract changes

- `MemoryCalibrationIdentity::new(base_fingerprint, load_shape)` requires the implementation's
  actual `LoadShape`. The base fingerprint describes provider content only.
- `MemoryRunContext::load_shape` must exactly equal the contract calibration identity.
- `MemoryEvidenceKey::load_shape` must exactly equal both `MemoryProviderContract::load_shape` and
  its calibration identity. Eager evidence cannot authorize Deferred and Deferred evidence cannot
  authorize Eager, even when ABI and base fingerprint match.
- Contract conformance rejects an identity whose shape differs from its contract.
- Previously packaged ABI-1 evidence and manifest rows are stale and must be regenerated. Matching
  by an old `-eager` or `-deferred` fingerprint suffix is intentionally unsupported.

## SceneWorks consumer migration

SceneWorks must serialize a required `loadShape` key/target and carry it through exact matching:

1. Bump the memory calibration bundle to `schemaVersion: 4`, the harness to
   `sceneworks-memory-v5`, and each record's calibration ABI to `2`.
2. Add typed load shape to `CalibrationBinding` and `EvidenceQuery`; include it in the exact record
   filter. Missing or mismatched shape is stale/unverified, never a compatible fallback.
3. Populate `load_shape` in every `MemoryEvidenceKey` builder and populate
   `load_shape` in every `MemoryRunContext` builder.
4. Update the MLX and Candle adapter protocols to carry `loadShape` explicitly. Adapters must not
   recover it by parsing a calibration fingerprint.
5. Regenerate all packaged evidence, manifest calibration rows, fixtures, and golden protocol
   payloads under schema 4 / harness v5.

Concrete consumers requiring edits in the SceneWorks repository include:

- `crates/sceneworks-core/src/memory_calibration.rs`
- `crates/sceneworks-worker/src/mlx_fit_gate.rs`
- `crates/sceneworks-worker/src/candle_memory_strategy.rs`
- `crates/sceneworks-worker/src/krea_control_fit.rs`
- `crates/sceneworks-worker/src/vram_gate.rs`
- `crates/sceneworks-worker/src/memory_strategy.rs`
- `crates/sceneworks-worker/src/image_jobs/flux2.rs`
- `crates/sceneworks-worker/src/image_jobs/base.rs`
- `crates/sceneworks-memory-adapter/src/bin/mlx.rs`
- `crates/sceneworks-memory-adapter/src/bin/candle.rs`
- `crates/sceneworks-worker/tests/fixtures/mlx-memory-calibration.json`

This inference change deliberately does not edit the sibling SceneWorks repository; that repository
must consume the breaking source contract and regenerate evidence before selecting optimized fits.
