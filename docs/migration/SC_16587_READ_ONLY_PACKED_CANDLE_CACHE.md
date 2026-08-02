# Read-only packed Candle cache placement (sc-16587)

Packed MLX q4/q8 weights need a one-time conversion into Candle's GGML device format. The converted
bytes remain file-backed so staged/windowed providers do not retain a second model-sized copy in
anonymous host memory.

`PackedWeightSidecars` now applies this placement policy:

1. A complete, content-valid `.candle-device-format-v1` cache beside the component is reused
   read-only. This path neither creates nor acquires `.prepare.lock`.
2. If the adjacent cache is incomplete, a complete content-valid external cache is reused read-only,
   also without creating or acquiring a lock.
3. If neither location is complete and the component directory is writable, conversion keeps using
   the model-adjacent cache. Missing/corrupt recovery and atomic publication are serialized by the
   exclusive preparation lock.
4. If the model-adjacent cache cannot create files, conversion uses a hashed component namespace in
   the writable external cache. Embedders can pass the root through
   `Weights::from_dir_with_external_cache_root` /
   `PackedWeightSidecars::prepare_with_external_cache_root`; otherwise set
   `SCENEWORKS_CANDLE_DEVICE_CACHE_DIR`. Without an override, the platform per-user cache is used
   (`LOCALAPPDATA/SceneWorks`, `~/Library/Caches/SceneWorks`, or
   `$XDG_CACHE_HOME/sceneworks` / `~/.cache/sceneworks`).

The external path is a fallback, not a packed-feature downgrade: Krea still consumes mapped
device-format sidecars and retains its normal resident or staged behavior. A cold read-only snapshot
therefore needs write permission only in the selected external cache.

## Disk and lifecycle contract

The cache hashes source tensor bytes, shapes, dtype, packed bit width, group size, and format version.
Changing a source selects a different immutable artifact. Payload hashes are verified before reuse;
invalid artifacts are rebuilt under the lock when the selected location is writable.

Operators must budget external or model-adjacent cache space for approximately one additional
device-format copy of the packed projections. Old content-addressed artifacts are not deleted
automatically. Clearing either cache is safe while no inference process is using it: the next packed
load recreates the missing entries. A complete warm cache requires read access only.
