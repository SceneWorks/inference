# Mage-Flow test fixtures

Two kinds live here: the published **component configs** (below), and one **numeric parity
fixture** produced by running the vendored reference itself.

## `mage_flow_small.safetensors` — NR-MMDiT parity (sc-14040)

A complete 2-block `MageFlow` NR-MMDiT at dim 24 with `torch.manual_seed(0)` random weights, in
**f32**, plus its inputs, its output, and its first block's inputs/outputs — all captured from the
vendored reference (`_vendor/mage_flow/`). Regenerate with

```sh
<ref-venv>/bin/python crates/media/mlx-gen/tools/dump_mage_flow_small.py
```

Consumed by `tests/mage_flow_small.rs`, which runs in the **default** `cargo test` — no licensed
weights, no gitignored goldens.

It exists because the real-weights goldens cannot do this job. Those are bf16, and the published
checkpoint's block-0 modulation gates reach ~1e8, so twelve bf16 blocks amplify rounding to a
**2e-2 mean-relative floor** (the port's own f32-vs-bf16 spread is 2.8e-2). Real porting mistakes
are smaller than that floor at the output — substituting the `mlx-gen-z-image` sibling's SwiGLU
gate for `gelu-approximate` moves `dit_out` by only ~1.7e-2 — so the real-weights gate cannot
discriminate them at any tolerance. In f32 at tiny dims the floor is 2.4e-3 and the same mutation
lands 30× outside it.

Two packings are captured, because they exercise different code: `gen` (fused-CFG generation — two
attention segments, one `img_shapes` entry each) and `edit` (`[target, ref×3]` in **one** attention
segment carrying **four** `img_shapes` entries, `pipeline.py:517-519`). The second is the only
configuration in which the msrope **frame axis** changes the attention scores instead of cancelling
out, so it is where the frame index is gated at the output level.

Random weights, no licensed data: MIT reference code, nothing derived from the published
checkpoints.

## Component configs

Verbatim, unmodified copies of **all four** component configs published by
**`microsoft/Mage-Flow`** (revision `9f46d09dce8a6211a5aaf157cc99754ac402a2fc`), committed so
`config_conformance.rs` can check every constant in `mlx_gen_mage::config` against the real file
instead of against itself. All four are here deliberately: a component whose config is *not*
committed ends up pinned only against literals retyped in the test, which is not a check at all.

| file | upstream path | SHA-256 |
| --- | --- | --- |
| `transformer_config.json` | `transformer/config.json` | `8493c3b2722738c2a824ac82b1fd9c89fefb4e354fc88363207193db7fe702de` |
| `vae_config.json` | `vae/config.json` | `abd124d603d6c6a03e9d0f2aa6d113b8c4afda0738400bdf2f99240aeaeaff76` |
| `scheduler_config.json` | `scheduler/scheduler_config.json` | `438fd8bcf254740e5d3f3e9800bbd9c571e342ab87885388d1505b7531c69c02` |
| `text_encoder_config.json` | `text_encoder/config.json` | `edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d` |

All **six** Mage-Flow repositories — `Mage-Flow`, `-Base`, `-Turbo`, `-Edit`, `-Edit-Base`,
`-Edit-Turbo` — ship these four files byte-identically (verified by hashing every local snapshot),
so one copy pins the family. Only the transformer *weights* and the model cards' default
`steps`/`cfg` differ between variants.

Constants with no home in any of these files — the timestep-embedder block, the joint-attention
order, the VL long-edge cap, the native-resolution bounds — are pinned against the vendored frozen
reference (`_vendor/mage_flow/`) instead. Nothing in `config.rs` is checked against itself.

**Licence:** the Mage-Flow model repositories and `github.com/microsoft/Mage` are MIT, which permits
redistribution. These are metadata files, not weights; the frozen reference implementation is
vendored separately under `crates/media/mlx-gen/_vendor/mage_flow/` with its own `LICENSE`,
per-file checksums and re-vendor policy in `../../../_vendor/VENDORED.md`. The weight-licence
manifest row (`release/model-weight-licenses.json`) lands with the catalog surface in sc-14047.

**Re-fetching:** copy the files out of a fresh snapshot and re-run
`cargo test -p mlx-gen-mage --test config_conformance`. A SHA-256 mismatch means the published
config changed — treat every constant transcribed from it as suspect and re-read
`_vendor/MAGE_FLOW_GAPS.md` before touching `src/config.rs`.
