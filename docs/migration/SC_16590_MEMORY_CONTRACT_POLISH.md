# SC-16590 memory-contract polish

SC-16590 tightens gen-core's memory identity and lifecycle contracts without changing the persisted
calibration schema. `MEMORY_CALIBRATION_ABI` remains **2**: load shape is still the latest numeric
calibration boundary, while this change replaces unconstrained Rust strings with typed values whose
protocol spellings are unchanged.

## Consumer migrations

- `MemoryEvidenceKey.backend` is now `MemoryBackend::{Mlx,Candle}`. Convert to or from persisted
  `"mlx"` / `"candle"` only at the evidence-protocol boundary through `MemoryBackend::as_key()`.
- `MemoryEvidenceKey.mode` is now `MemoryMode`. Existing `text_to_image`, `image_to_image`, and
  `edit` rows map to the named variants; provider-specific modes use `MemoryMode::Other` and
  `MemoryMode::as_key()` at the protocol boundary.
- `Residency::with_resident_parts` now returns `Result<Option<T>, R::Error>`. Propagate the outer
  error before interpreting `None`; a poisoned residency mutex is no longer presented as readable
  state or as an ordinary cold cache.
- The request-level cache declaration is now
  `StrategyTierParametersModeGeometryOverlayAndEngagedComposition`. A cache spanning loaded
  generators must additionally key its outer entry by resolved provider, backend realization, and
  load shape.
- Calibration executables that set `GenerationMemory::calibration_error_phase` must instead call
  `GenerationMemory::authorize_calibration_fault(phase)` (or set the authorization pair exactly).
  Production requests leave both hidden fields at their defaults. The shared request floor rejects
  incomplete pairs before provider execution.

SceneWorks' MLX and Candle fit gates construct `MemoryEvidenceKey` values and its two memory-adapter
executables inject calibration failures. Those call sites must be migrated together when SceneWorks
advances its inference pin; otherwise the typed key change fails at compile time and an unpaired
fault request is rejected at validation rather than at its requested physical phase.

## Runtime compatibility

- Ordinary requests retain default-inert calibration controls.
- Known platform-composed memory routes now report `Ok(None)` for an unmeasured 1024-square
  activation anchor; genuinely unknown route ids still error.
- Resident-only component sources reject staged, streamable, exclusive-staged, and direct-eviction
  requests before dropping their sole warm pair.
- Effective budget accounting now preserves committed-over-total deficits before adding reclaimable
  bytes, preventing over-admission under unified-memory pressure.
