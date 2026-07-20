# Vendored `candle-kernels` — multi-arch fatbin fork (sc-7544)

This is a **local fork** of `candle-kernels` `0.10.2`, copied verbatim from the pinned candle revision

    https://github.com/huggingface/candle @ 1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d

(the same rev the workspace pins `candle-core` / `candle-nn` / `candle-transformers` to). The copy
was originally taken at rev `65ecb58c11d2244a7e60c71bdcdb19b15b0a4343`; neither the 2026-07-02 pin
bump to `c1e6756a89` (upstream UAF fix #3493) nor the sc-12111 bump to `1e6aa85e` (upstream #3531
depthwise-conv launches) required a re-copy — `candle-kernels/` is byte-identical across those revs
apart from `src/compatibility.cuh`, where the copy is *ahead* of the pin (see delta 3 below). It is
wired into the build via a `[patch]` in the workspace `Cargo.toml`:

```toml
[patch."https://github.com/huggingface/candle"]
candle-kernels = { path = "vendor/candle-kernels" }
```

**A `[patch]` only takes effect in the TOP-LEVEL workspace.** This one covers builds of *this*
repository; any consumer that pins these crates by git (SceneWorks) resolves candle-kernels from
upstream candle unless its own root manifest carries an equivalent patch. That is exactly how the
sc-7544 fix silently dropped out of the SceneWorks desktop packaging at the candle-gen → inference
cutover (sc-13510) — SceneWorks now patches candle-kernels to this vendored copy (same repo, same
pinned rev as its other inference pins) and guards it with
`crates/sceneworks-worker/tests/candle_kernels_patch_guard.rs` plus a packaging-time `cuobjdump`
check. If this vendor dir ever moves or is dropped, update that consumer patch in the same change.

## Changes vs upstream

There are **three** changes vs upstream:

1. `build.rs` adds three `-gencode` flags to the **statically-linked quant/moe kernel** build
   (`build_lib()` → `libmoe.a`), turning its single-arch SASS object into a true **multi-arch fatbin**
   (the sc-7544 Blackwell fix, detailed below).
2. `src/cast.cu` adds the **`int32_t` source casts** (`cast_i32_f32` et al., search for `sc-9601`).
   Upstream omits I32 source casts, but candle-core's `to_dtype` still *dispatches* `cast_i32_f32` for an
   int32 tensor (`cuda_backend`), so without the symbol an on-device `i32 → f32` cast fails with "named
   symbol not found". The INT8-ConvRot int8 IGEMM accumulates in int32; this cast lets its per-row
   dequant fold stay on-device instead of a per-forward int32→host round-trip. `PTX`-embedded (goes
   through `build_ptx()`), so no `libmoe.a` change. Upstreamable (arguably an upstream gap).
3. `src/compatibility.cuh` keeps upstream fix **#3558** (`__hmax_nan`/`__hmin_nan` device fallbacks
   guarded by `__CUDA_ARCH__ < 800`, from the original `65ecb58c` copy) which the `1e6aa85e` pin —
   on a different upstream branch line — *predates* (it still has the older
   `__CUDACC_VER < 12.2 && __CUDA_ARCH__ < 750` guard). Not a local invention, just a newer upstream
   state retained on purpose: for every arch this build targets (sm_80+ SASS, compute_80+ PTX) both
   guards compile to the same thing, and downgrading would drop the Turing fix for nothing.

Everything else — every other `.cu`/`.cuh` source, `lib.rs`, `ffi.rs`, `Cargo.toml` — is byte-for-byte
upstream, so candle-core links an otherwise-identical Rust/symbol surface (a fatter `libmoe.a` + the new
`cast_i32_*` PTX symbols). Diff against the upstream rev to confirm these are the sole deltas.

### Why

`candle-kernels` compiles its GGUF `QMatMul` kernels (`mmq_gguf/*`, `moe/*`, `mmvq_gguf`) with
`nvcc -c` (a SASS **object**, no PTX) — unlike the dense kernels, which go through `build_ptx()` and
embed forward-JIT-able `compute_80` PTX. cudaforge emits one `-gencode` from `CUDA_COMPUTE_CAP`; at
the `=80` packaging baseline that is `code=sm_80` (an Ampere-only cubin). On a Blackwell **sm_120**
GPU there is no compatible cubin and no PTX to JIT, so the quant matmul **silently returns zeros**
(dense models work, quantized models render black/NaN). See the story sc-7544 and the
`candle-cuda-quant-needs-native-sm120` project memory. The fatbin embeds native `sm_80` + `sm_90` +
`sm_120` SASS plus `compute_120` PTX, so one binary runs natively Ampere → Ada → Hopper → Blackwell
and JITs forward to newer archs.

The regression is guarded by `candle-gen/tests/cuda_quant_smoke.rs` (runs in `scripts/check-cuda.ps1`).

## MAINTENANCE — re-vendor on every candle pin bump

The `[patch]` forces **these** kernel sources onto whatever `candle-core` rev the workspace pins. They
match only as long as this copy is from the **same** candle rev. **When the candle pin bumps**
(`candle-core`/`candle-nn`/`candle-transformers` rev in the workspace `Cargo.toml`):

1. Re-copy `candle-kernels/` from the new rev's checkout over `vendor/candle-kernels/`.
2. Re-apply the `build.rs` `-gencode` block above (search for `sc-7544`).
3. Re-apply the `src/cast.cu` `int32_t` cast block (search for `sc-9601`) unless upstream has added it.
4. Keep the `src/compatibility.cuh` `__CUDA_ARCH__ < 800` fallback guard (upstream #3558) unless the
   new rev already includes it.
5. Re-run the CUDA gate (`pwsh scripts/check-cuda.ps1`) — `cuda_quant_smoke` must pass on Blackwell.
6. SceneWorks patches candle-kernels to this dir at its pinned inference rev; once the bump lands
   there (`node scripts/bump-inference.mjs`), its `candle_kernels_patch_guard` test re-verifies the
   lockstep automatically.

If a bump ever lands without re-vendoring, candle-core may get **stale kernels** (subtle breakage or
link errors). If candle/cudaforge ever gains native multi-target fatbin support, drop this vendor and
the `[patch]` and configure the cap list directly.
