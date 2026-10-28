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

3. `src/lib.rs`: the dense `FlashAttn` path converts every stride and extent it hands the kernel's
   `u32` parameter ABI with a checked `u32::try_from` (an error) instead of an `as u32` truncation
   (an out-of-bounds read). `candle_gen::fused_attention::flash_eligibility` already refuses such
   calls; this is the backstop for any other caller. The var-len path is unchanged (unused here).
4. `build.rs`: reruns on any change under `kernels/` and passes `.watch(["kernels"])` to cudaforge,
   so every header's content is part of each object's rebuild key. Upstream only listed some of
   the headers, so a persistent `CANDLE_FLASH_ATTN_BUILD_DIR` cache could serve objects compiled
   against a stale header. (nvcc flags / the `-gencode` ladder are in cudaforge's args hash and the
   arch in its per-object key already; an nvcc *toolkit* upgrade is not — clear the cache dir.)
5. `Cargo.toml` declares `license = "(MIT OR Apache-2.0) AND BSD-3-Clause"` (upstream: `MIT OR
   Apache-2.0`), which is what the crate actually distributes — see *Licensing* below.

## Licensing

The crate combines three works, and a binary that links it carries all three:

| Part | Licence | Copyright | File here |
| --- | --- | --- | --- |
| Rust glue (`src/`, `build.rs`) | MIT OR Apache-2.0 | the candle authors (Hugging Face) | `LICENSE-MIT`, `LICENSE-APACHE` |
| CUDA kernels (`kernels/`) | BSD-3-Clause | Tri Dao / the flash-attention contributors (`AUTHORS-FLASH-ATTENTION`) | `LICENSE-FLASH-ATTENTION` |
| CUTLASS (headers fetched by cudaforge at `CUTLASS_COMMIT`, instantiated into `libflashattention.a`) | BSD-3-Clause | NVIDIA Corporation & affiliates | `LICENSE-CUTLASS.txt` |

- `LICENSE-MIT` / `LICENSE-APACHE` are candle's repository-root files, copied unmodified.
- `LICENSE-FLASH-ATTENTION` and `AUTHORS-FLASH-ATTENTION` are `LICENSE` / `AUTHORS` from
  https://github.com/Dao-AILab/flash-attention, copied unmodified from tag `v2.7.2`
  (`b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7`). candle pulled these FA2 kernels in through
  huggingface/candle#2688-#2690 (2024-12-31, "pull in latest FA2"), i.e. the v2.7.x line, and has
  since modified them (head-dim 512, #3417). flash-attention's `LICENSE` has not changed since
  `64f42cd0576344ad21c178abe2cd58aac4c92e4e` (2022-09-09), so the text is byte-identical at every
  revision candle could have vendored (verified against that commit).
- `LICENSE-CUTLASS.txt` is `LICENSE.txt` from https://github.com/NVIDIA/cutlass at exactly
  `CUTLASS_COMMIT` = `7d49e6c7e2f8896c47f586706e67e1fb215529dc`, copied unmodified.

Both BSD-3-Clause licences require binary redistributions to reproduce the copyright notice,
conditions and disclaimer in their documentation: any product that ships a binary built with
`candle-gen/flash-attn` must carry `LICENSE-FLASH-ATTENTION` and `LICENSE-CUTLASS.txt` in its
third-party notices. BSD-3-Clause is on the `deny.toml` allow-list.

Everything else (`src/` apart from delta 3, `kernels/`, `README.md`) is byte-for-byte upstream.

## Runtime guard

`candle-gen/build.rs` parses this `build.rs`'s `.arg("-gencode=...")` calls and bakes the ladder into
candle-gen; `candle_gen::fused_attention` dispatches to the kernel only when the device's compute
capability is >= 8.0 and covered by that ladder (SASS within one major, PTX forward). If you change the
flags here, the guard follows automatically; if you move this directory, update `candle-gen/build.rs`.

## Re-vendoring

Copy `candle-flash-attn/` from the new pinned revision, re-apply the deltas above, and update the
revision recorded here. Re-copy `LICENSE-CUTLASS.txt` if `CUTLASS_COMMIT` moves, and re-check the
flash-attention `LICENSE` / `AUTHORS` if the kernels are re-synced from a newer FA2.
