# Vendored `candle-core` — CUDA-graph parameter cache (sc-24441)

This directory is `candle-core` `0.10.2` from

    https://github.com/huggingface/candle @ 1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d

(the revision the workspace pins `candle-core` / `candle-nn` / `candle-transformers` /
`candle-flash-attn` to) with **two upstream commits cherry-picked, unmodified**:

| Upstream commit | Title | Files |
| --- | --- | --- |
| `cabbc30191b352814d8bb94e821e82faa883c7b5` | Add htod cache for cuda graphs (#3564) | `src/cuda_backend/device.rs` |
| `f53ed3bfbd2e3291c70e4db81eaf0faeccbb8e83` | Parameter cache for CUDA kernel launch (#3598) | `src/cuda_backend/{device,mod}.rs` |

It is wired into the build by the workspace `[patch]` (root `Cargo.toml`), next to the vendored
`candle-kernels`, and is **excluded** from the workspace members so no workspace-wide `fmt` /
`clippy` / `test` rewrites or lints upstream code.

## Why

The decode CUDA-graph runner (`candle-llm` `decode/graph.rs`, epic sc-24432 story sc-24441)
records a decode step with stream capture. At `1e6aa85e`, every candle CUDA op whose kernel takes
a layout — `index_select`, `where_cond`, comparisons, reductions, gathers / scatters, and any
unary / binary / copy on a non-contiguous operand — uploads its `[dims, strides]` from a
temporary host `Vec` (`SlicePtrOrNull::params_from_layout` / `CudaDevice::clone_htod`). A capture
records that as a memcpy node reading a host address freed as soon as the op returns, so a replay
reads garbage (sc-24134 finding 6: 851 such uploads in one Qwen3.8-27B step; the runner's census
refuses them as `host_upload_in_capture`).

`f53ed3bf` routes those parameter vectors through a per-`(device, params)` cache of device buffers
**while a `CudaDevice::enable_cuda_graph_htod_cache` guard is alive**: the eager warm-up fills it,
the capture reuses it, and any host<->device copy the cache does not cover is an error during
capture instead of a silently broken graph. `cabbc301` introduces that guard (and a content-keyed
cache for small `clone_htod` / `memcpy_htod` uploads made under it); `f53ed3bf` does not compile
without it, and it is not an ancestor of `1e6aa85e` (the pin sits on a different upstream branch
line), so it is taken as the prerequisite.

**Not taken:** `3ba428d0` (Improve CUDA stream consistency, #3565), which sits between the two
upstream and moves `Device::new_cuda` onto the per-thread stream — an eager behaviour change this
vendor must not carry. A full candle bump was rejected for the same reason: it brings ~110 commits
including "Optimize CUDA kernels #3600", which moves media numerics and stales their parity
evidence.

## Changes vs upstream `1e6aa85e`

1. `src/cuda_backend/device.rs`, `src/cuda_backend/mod.rs` — exactly the two cherry-picks above
   (`git cherry-pick cabbc301 f53ed3bf` on an upstream `1e6aa85e` checkout reproduces these files
   byte for byte).
2. `Cargo.toml` — upstream inherits every field and dependency from its workspace
   (`*.workspace = true`), which does not exist here. Each inherited entry is spelled out with the
   value upstream's root manifest at `1e6aa85e` declares; the three sibling crates upstream
   reaches by path (`candle-kernels`, `candle-metal-kernels`, `candle-ug`) are the same crates at
   the same git revision the pin resolved before, so `Cargo.lock` changes by one line (the
   `source` of `candle-core`). Features, targets, dev-dependencies, benches and examples are
   upstream's.
3. `VENDORED.md` — this file.

Everything else (`src/**`, `tests/**`, `benches/**`, `examples/**`, `README.md`, `LICENSE`) is
byte-for-byte upstream. Diff against an upstream checkout to confirm these are the sole deltas.

## Eager behaviour is unchanged outside the guard

Every behavioural line the cherry-picks add is behind `cuda_graph_htod_cache_enabled()` — a
thread-local guard depth that only `enable_cuda_graph_htod_cache` raises:

* `SlicePtrOrNull::params_from_vec` with no guard returns `SlicePtrOrNull::Ptr(dev.clone_htod(..))`,
  the exact call every replaced site made before; `params_from_layout` still returns `Null` for a
  contiguous layout.
* `clone_htod` / `memcpy_htod` with no guard take the unchanged `self.stream.*` call.
* `clone_dtoh` / `memcpy_dtoh` / `to_cpu_storage` call `check_capture_copy`, which returns `Ok(())`
  immediately with no guard.
* The only unconditional differences are type-level: `T: 'static` on `clone_htod` /
  `memcpy_htod` (every candle dtype and every workspace caller satisfies it — the CUDA compile
  lane builds them all) and the `SlicePtrOrNull::Cached` variant, which is constructed only under
  the guard.

In this workspace the guard is held only by the graph runner, and only around its warm-up and
capture steps (`candle-llm` `decode/graph.rs`). The runtime evidence is the CUDA lane: the
existing candle-llm / candle-gen / candle-audio CUDA suites run unchanged against this copy, and
`candle-llm`'s `cuda_graphs` suite checks the layout-bearing ops bit for bit with and without the
guard.

## Consumers

**A `[patch]` only takes effect in the top-level workspace.** Any consumer that builds these
crates by git with `--features cuda` (SceneWorks, ChatWorks) resolves upstream `candle-core` unless
its own root manifest patches it to this directory — and upstream `1e6aa85e` has no
`enable_cuda_graph_htod_cache`, so such a build does not compile. Add, next to the existing
`candle-kernels` entry:

```toml
[patch."https://github.com/huggingface/candle"]
candle-core = { git = "https://github.com/SceneWorks/inference", rev = "<the pinned inference rev>" }
```

## MAINTENANCE — re-vendor on every candle pin bump

`scripts/bump_pins.py` refuses a candle bump while this copy exists. To bump:

1. Re-copy `candle-core/` from the new revision over this directory (keep this file).
2. Re-apply `cabbc301` and `f53ed3bf` unless the new revision contains them (upstream main after
   2026-06-24 does, and then this vendor and its `[patch]` can be dropped).
3. Re-derive `Cargo.toml` from the new upstream workspace manifest as described above.
4. Move every pin (`Cargo.toml`, `scripts/check-workspace.py` `PINNED_WORKSPACE_DEPENDENCIES` and
   `VENDORED_PACKAGES`), re-vendor `candle-kernels` (its own `VENDORED.md`), regenerate the lock,
   and run the CUDA lane (`gh workflow run ci.yml -f lanes=windows-cuda`).
