# Vendored `candle-metal-kernels` — rank > 4 strided-reduce indexing fix

This directory is `candle-metal-kernels` `0.10.2` from

    https://github.com/huggingface/candle @ 1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d

(the revision the workspace pins `candle-core` / `candle-nn` / `candle-transformers` /
`candle-flash-attn` to, and the one `vendor/candle-core` is taken from) with **one upstream fix
applied, unmodified**:

| Upstream commit | Title | Files |
| --- | --- | --- |
| `2395577a43cc704e486edf5389cb14ba97eba0f8` | Fix Metal rank-5+ strided reduce reading out of bounds (#3873) | `src/metal_src/reduce.metal` |

It is wired into the build by the workspace `[patch]` (root `Cargo.toml`), next to the vendored
`candle-core` and `candle-kernels`, and is **excluded** from the workspace members so no
workspace-wide `fmt` / `clippy` / `test` rewrites or lints upstream code.

## The defect

candle-core's Metal `reduce_op` moves the reduced dims to the end of `[dims, strides]` and, unless
the input is contiguous **and** the reduced dims are its trailing dims, launches the `*_strided`
reduce kernels (`sum`, and so `mean`; `min`, `max`, `argmin`, `argmax`; every dtype). Those decode
a flat element index with `get_strided_index_t`: ranks 1–4 go through the fixed-size
`strided_indexer<D>`, which peels `dims[D-1]` (innermost) first and leaves `idx * strides[0]` for
the outermost; every larger rank goes through `get_strided_idx_fallback<4>`, which is meant to peel
the `num_dims - 4` innermost dims and hand the remainder to `strided_indexer<4>` over `dims[0..4]`.
At `1e6aa85e` its loop computed

    for (d = D; d < num_dims; d++) dim_idx = num_dims - 1 - d;   // D = 4

which for rank 5 peels `dims[0]` — the outermost — as if it were the innermost, then decodes the
leftover over `dims[0..4]` again: `dims[4]` is never decoded and the final `idx * strides[0]` runs
past the end of the source buffer. The result is wrong for every rank > 4 strided reduction and,
being an out-of-bounds read, depends on whatever the allocator placed next to the buffer (sc-25909
saw a NaN-poisoned pool turn Qwen-Image 2.1 reference latents non-finite). Ranks ≤ 4 and
contiguous trailing reductions (the `fast_*` kernels) were never affected; the CPU and CUDA
backends have their own indexers and are unaffected.

## Changes vs upstream `1e6aa85e`

1. `src/metal_src/reduce.metal` `get_strided_idx_fallback` — exactly upstream `2395577a`'s one
   line, `dim_idx = num_dims - 1 - (d - D)`: the loop now peels `dims[num_dims-1]` down to `dims[D]`
   (innermost first, as `strided_indexer` and `utils.metal`'s `get_strided_index` do) and leaves
   `dims[0..D]` to the fixed-size indexer. `2395577a` is not an ancestor of `1e6aa85e` (the pin sits
   on a different upstream branch line), so it is applied by hand; its other half, an
   `expand_grad` case in `candle-core/tests/grad_tests.rs`, is not taken — `vendor/candle-core` stays
   as its own `VENDORED.md` describes, and the CI-buildable pins below cover the same path.
2. `LICENSE-MIT`, `LICENSE-APACHE` — upstream's repository-root licence texts at `1e6aa85e`, copied
   unmodified. The crate is `MIT OR Apache-2.0` and upstream ships these only at the repository root,
   so a crate-directory copy would otherwise redistribute the source without its licence.
3. `VENDORED.md` — this file.

Everything else (`src/**`, `examples/**`, `Cargo.toml`, `README.md`) is byte-for-byte upstream;
upstream's `Cargo.toml` for this crate inherits nothing from its workspace, so it needed no
re-derivation. Diff against an upstream checkout to confirm these are the sole deltas.

## Pins

* `candle-gen` `tests/metal_strided_reduce.rs` — every reduce op at ranks 5 and 6 over every dim
  (dim 0 and the trailing dim included) and three two-dim sums, on contiguous, transposed, narrowed
  and broadcast inputs, Metal against the CPU. Red on the unfixed kernel (292 of 312 cases).
* `candle-gen-qwen-image-2-1` `tests/metal_reference_latents.rs` — the production victim: reference
  latents encoded on Metal after the reference text conditioning match the CPU's.

Both are `#[cfg(feature = "metal")]` and are the only Candle tests CI runs on a Metal device: the
`macos-metal` job's *Test Candle Metal reduce pins* step (`.github/workflows/ci.yml`; selected for
any change under `crates/media/candle-gen`, which includes this directory). Locally:
`cargo test -p candle-gen -p candle-gen-qwen-image-2-1 --features metal --test integration --
metal_strided_reduce:: metal_reference_latents::`.

## Consumers

**A `[patch]` only takes effect in the top-level workspace.** A consumer that builds these crates
by git with the Metal backend (SceneWorks' `runtime-macos` `audio-metal` lane) resolves upstream
`candle-metal-kernels` — the unfixed kernel — unless its own root manifest patches it, next to its
existing `candle-core` / `candle-kernels` entries:

```toml
[patch."https://github.com/huggingface/candle"]
candle-metal-kernels = { git = "https://github.com/SceneWorks/inference", rev = "<the pinned inference rev>" }
```

## MAINTENANCE — re-vendor on every candle pin bump

`scripts/bump_pins.py` refuses a candle bump while this copy exists. To bump:

1. Re-copy `candle-metal-kernels/` from the new revision over this directory (keep this file), and
   the repository-root `LICENSE-MIT` / `LICENSE-APACHE` from the same revision.
2. Re-apply `2395577a` unless the new revision contains it (upstream `main` after 2026-08-13 does) —
   once it does, drop this vendor, its `[patch]` entry, its workspace `exclude`, and its
   `scripts/check-workspace.py` `VENDORED_PACKAGES` / `scripts/bump_pins.py` `vendored` entries.
3. Move every pin and re-vendor `candle-core` and `candle-kernels` as their own `VENDORED.md`
   describe, then run the Metal pins above on a Mac.
