# Vendored `candle-flash-attn` — multi-arch fatbin fork (sc-26691)

This is a local fork of `candle-flash-attn` `0.10.2`, copied from the pinned candle revision

    https://github.com/huggingface/candle @ 1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d

It is wired in as a plain **path** workspace dependency (`candle-flash-attn = { path = ... }` in the
root `Cargo.toml`), not through a `[patch]`. A path dependency inside this repository resolves the
same way for git-pinned consumers (SceneWorks), so — unlike `vendor/candle-kernels` — no consumer
needs a matching `[patch]` line for the fork to take effect.

## Changes vs upstream

1. `build.rs` adds three `-gencode` flags (`sm_90`, `sm_120` SASS and `compute_120` PTX) on top of
   cudaforge's single `CUDA_COMPUTE_CAP`-derived arch. Upstream builds one SASS cubin with no PTX, so
   the SceneWorks cap=80 packaging builds would carry an Ampere-only kernel that cannot launch on the
   Blackwell (sm_120) cards MiniMax-H3 renders on. Same ladder and rationale as `vendor/candle-kernels`.
2. `Cargo.toml` names `candle-core` by the pinned git revision instead of upstream's sibling path
   (`../candle-core`, which does not exist here); the root `[patch]` redirects it to
   `vendor/candle-core`. Upstream's dev-dependencies (a sibling-path `candle-nn`) and its `tests/`
   directory are dropped for the same reason.

`LICENSE-MIT` / `LICENSE-APACHE` are upstream's repository-root files, copied unmodified.

Everything else (`src/`, `kernels/`, `README.md`) is byte-for-byte upstream.

## Runtime guard

`candle-gen/build.rs` parses this `build.rs`'s `.arg("-gencode=...")` calls and bakes the ladder into
candle-gen; `candle_gen::fused_attention` dispatches to the kernel only when the device's compute
capability is >= 8.0 and covered by that ladder (SASS within one major, PTX forward). If you change the
flags here, the guard follows automatically; if you move this directory, update `candle-gen/build.rs`.

## Re-vendoring

Copy `candle-flash-attn/` from the new pinned revision, re-apply the two deltas above, and update the
revision recorded here.
