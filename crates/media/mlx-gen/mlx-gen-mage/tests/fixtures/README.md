# Mage-Flow fixtures

Two kinds live here: the **published component configs** (immediately below) and one small
**arithmetic golden** (last section).

## Component config fixtures

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

## `te_micro_golden.safetensors` — the text encoder's arithmetic oracle (sc-14038)

A **32 KB** dump of a real `transformers.Qwen3VLTextModel` — the same class the vendored reference
patches — instantiated at toy dimensions (hidden 16, 3 layers, 4 q / 2 kv heads, `head_dim` 8,
FFN 12, vocab 24) with seeded random weights, in f32, eager attention. It carries the LM state dict
under `lm.*` plus `io.input_ids` and `io.last_hidden_state`.

Unlike the boundary goldens in `crates/media/mlx-gen/tools/golden/` — real 4.1B weights, multi-GB,
gitignored, `#[ignore]`d consumers — this one is **committed and runs in the default `cargo test`
lane**, because it needs no model weights. That is the point: every other weights-free test in
`text_encoder_forward.rs` pins *topology* (shapes, layer counts, packing isolation, causality), and
none of them would catch a regression in GQA `repeat_kv` grouping, QK-RMSNorm placement, the SwiGLU
gate/up order or `o_proj`. This fixture gates the **arithmetic** of the whole composition against
upstream, and `the_micro_oracle_rejects_composition_swaps` proves it discriminates by swapping
same-shaped tensor pairs (measured 20–26× the tolerance).

The toy dimensions deliberately preserve every production *structure*: a real GQA group size,
`head_dim` decoupled from `hidden / heads`, all three interleaved-M-RoPE sections populated, and
`intermediate != hidden`. Norm scales carry a wide (0.5) spread so a `q_norm`/`k_norm` swap is a
real signal rather than a near-no-op.

**Regenerating:** `python3 crates/media/mlx-gen/tools/dump_mage_te_micro_golden.py` (needs torch +
transformers, no weights). Deterministic — re-running reproduces the file byte-for-byte. Regenerate
only when the upstream `Qwen3VLTextModel` arithmetic legitimately changes, and say so on the story:
silently re-blessing this file would convert a real regression into a passing test.
