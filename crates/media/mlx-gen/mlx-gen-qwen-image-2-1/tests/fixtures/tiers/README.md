# Committed packed-tier fixtures (sc-24112)

The packed `model.safetensors` of the `q8` and `q4` tiers of the miniature parity snapshot
(`../tiny-snapshot`), produced by `mlx_gen_qwen_image_2_1::convert::prequantize_turnkey`.

**Only `model.safetensors` is committed, and only for `transformer/` and `text_encoder/`.**

* `config.json` is **not** committed. A packed component's config is the source component's config
  plus a `{"quantization": {...}}` marker, so committing it would couple these fixtures to every
  unrelated edit of `../tiny-snapshot/*/config.json` (sc-24110 edits exactly that: token ids and
  `deepstack_visual_indexes`). Both backends' tests compose it at run time from the source config
  plus the marker — `tiers::composed_marker_config`.
* The rest of a tier (`vae/`, `processor/`, `scheduler/`, `model_index.json`) is copied through
  **dense and unchanged** by the converter, so the tests compose it from `../tiny-snapshot`.

These exist so the **Candle** backend can be held to the *same artefact* the MLX converter writes,
rather than to a second hand-built packed fixture that could drift. The cross-crate path convention
is the one `candle-gen-qwen-image-2-1/tests/common/mod.rs` already uses for the parity fixtures.

## What pins them, and what deliberately does not

* `tiers::the_committed_tier_fixtures_have_the_converters_shape` — same key set, same shapes and
  dtypes as a fresh conversion, and a correctly composed marker.
* `tiers::the_committed_packed_weights_dequantize_onto_the_dense_ones` — the committed codes
  dequantize (on the **CPU stream**) onto the dense weights they came from, inside the analytic
  half-level bound of affine group-64 quantization. This is what pins the *values*, and it is
  device-independent.
* There is deliberately **no byte-for-byte golden** against a fresh conversion. That conversion runs
  `mlx_rs::ops::quantize` on whatever device the runner has, so such a golden would be
  machine-dependent across the two macOS runners. Byte reproducibility is pinned instead as a
  same-run property, by `tiers::conversion_is_byte_reproducible_across_runs`.

## Regenerating

Regenerate only when `../tiny-snapshot`'s **weights** change (a config-only edit does not affect
these files). Do not add `config.json` back.

```sh
FIX=crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures
for TIER in q8 q4; do
  QWEN21_SRC=$FIX/tiny-snapshot QWEN21_DST=/tmp/q21tiers QWEN21_TIER=$TIER \
    cargo run --release --example qwen_image_2_1_prequant -p mlx-gen-qwen-image-2-1
  for C in transformer text_encoder; do
    cp /tmp/q21tiers/$TIER/$C/model.safetensors $FIX/tiers/$TIER/$C/model.safetensors
  done
done
```

## Why the tiers are partly dense here

The miniature geometry is 32/64 wide and a shippable tier declares exactly one
`quantization.group_size` (64), so every `Linear` narrower than that stays dense. On this fixture
that leaves three packed DiT leaves and the SwiGLU `down_proj` of both Qwen3 layers — enough that
both backends' packed paths are genuinely exercised
(`tiers::the_fixture_tiers_pack_both_the_dit_and_the_tower` pins it). The released geometry has no
such width, so in production the same converter packs everything. See
`mlx_gen_qwen_image_2_1::quant`.

## A tier is one width

`q4/text_encoder/model.safetensors` is a **Q4** artefact, distinct from and smaller than the `q8`
one. An earlier revision of this story held the tower at Q8 on the q4 tier; that was withdrawn —
a tier is a whole-pipeline contract, so selecting q4 runs q4 everywhere.

## Provenance of the real tiers

The production tiers are produced by the same converter from the frozen
`Qwen/Qwen-Image-2.1` snapshot and re-hosted at **`SceneWorks/qwen-image-2-1-mlx`**, one subdirectory
per tier (`q8/`, `q4/`), each carrying the `CHANGES.md` change record and the `SHA256SUMS` manifest
the converter itself writes beside the weights (`convert::prequantize_turnkey`, sc-24114). That
is the repository the SceneWorks manifest half pins. Until that upload happens the tiers are
`supported_quants` a caller cannot yet download; the terminal story owns closing that gap.
