# Committed packed-tier fixtures (sc-24112)

The two **packed components** of the `q8` and `q4` tiers of the miniature parity snapshot
(`../tiny-snapshot`), produced by `mlx_gen_qwen_image_2_1::convert::prequantize_turnkey`. Only
`transformer/` and `text_encoder/` are committed: the rest of a tier (`vae/`, `processor/`,
`scheduler/`, `model_index.json`) is copied through **dense and unchanged**, so both backends' tests
compose a full tier at run time from these two dirs plus the dense remainder of `../tiny-snapshot`.

These exist so the **Candle** backend can be held to the *same artefact* the MLX converter writes,
rather than to a second hand-built packed fixture that could drift. The cross-crate path convention
is the one `candle-gen-qwen-image-2-1/tests/common/mod.rs` already uses for the parity fixtures.

They are also the reproducibility pin: `tiers::the_committed_tier_fixtures_match_a_fresh_conversion`
re-runs the converter and asserts the result is **byte-identical** to what is committed here. That
test is the MLX (macOS/Metal) lane's, so a reproducibility break shows up as a red there rather than
as a silently different published artefact.

## Regenerating

```sh
FIX=crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures
for TIER in q8 q4; do
  QWEN21_SRC=$FIX/tiny-snapshot QWEN21_DST=/tmp/q21tiers QWEN21_TIER=$TIER \
    cargo run --release --example qwen_image_2_1_prequant -p mlx-gen-qwen-image-2-1
  rm -rf $FIX/tiers/$TIER/transformer $FIX/tiers/$TIER/text_encoder
  cp -R /tmp/q21tiers/$TIER/transformer /tmp/q21tiers/$TIER/text_encoder $FIX/tiers/$TIER/
done
```

Regenerate only when `../tiny-snapshot` itself changes. If the byte-identity test reds without a
snapshot change, that is a change in `mlx_rs::ops::quantize` or in the safetensors writer — a real
finding about artefact reproducibility, not a fixture to refresh.

## Why the tiers are partly dense here

The miniature geometry is 32/64 wide and a shippable tier declares exactly one
`quantization.group_size` (64), so every `Linear` narrower than that stays dense. The released
geometry has no such width, so in production the same converter packs everything. See
`mlx_gen_qwen_image_2_1::quant`.
