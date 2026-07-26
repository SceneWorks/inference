# Stable Audio 3 shared primitive parity (`sc-14536`)

The primitive oracle in `sa3-primitives-reference/` is deliberately separate from the locked
eight-model `sc-14534` reference manifest. It was generated from upstream commit
`124e8a799f57a1f665495ecb72e547d0a62867f1` with the exact pinned Python environment and actual
weights from the four representative snapshots recorded in its manifest. Generation and
verification cross-check the story ID, upstream commit, `uv.lock` digest, exact Python/package
versions, and every source repository/revision/file digest against the `sc-14534` P0 manifest and
snapshot lock. Mutation tests prove each provenance field is enforced.

The fixture covers the shipped capability matrix:

- small DiT ordinary self/cross attention, RMSNorm in fp32, AdaLN with exact
  `sigmoid(1 - gate)`, local conditioning, GLU-FF mult 4, independently routed self/cross key
  padding and additive masks, and 64 memory tokens;
- medium DiT RMSNorm differential attention using direct `ordinary - differential` subtraction;
- SAME-S DyT, fp32 half-split RoPE, differential attention, and SiLU GLU-FF mult 3;
- SAME-L `sin(pi*x)` GLU-FF from decoder block 5. With depth 12 and
  `sinusoidal_blocks = 8`, the strict `<` threshold selects seven indices, 5 through 11. The
  public band-mask builder also locks the future SAME-L `[1, 1]` sliding-window shape;
- upstream custom LayerNorm and RMSNorm with `fix_scale` and non-fp32 `force_fp32`, attention's
  distinct `torch.nn.LayerNorm` `weight`/`bias` state layout, LayerScale, and differential
  cross-attention with both mask kinds;
- legacy weight-normalized convolution materialization, 256-sample patched stereo transform, and
  SoftNorm train/eval noise scales using a saved noise tensor. The oracle executes the frozen
  upstream `WNConv1d.forward` and `SoftNormBottleneck.decode` methods directly; only the random
  noise source is patched to make decode deterministic.

The Rust real-weight test reports cosine similarity and fp32 maximum absolute error for each
primitive. The acceptance floor is cosine `0.9999`; the max-absolute bounds are recorded alongside
the empirical results rather than substituted for cosine. Mutations cover the corrected direct
subtraction, DyT alpha, eval noise, padded-tail, SAME-L threshold, gate, independent assembled
self/cross mask wiring, zero-initialized branch outputs, LayerScale's zero-init override, and
weight-layout branches.

On the pinned CPU run, every reported cosine was `1.000000000`. The full small block (which
accumulates ordinary self-attention, cross-attention, two norms, AdaLN gates, local conditioning,
and a mult-4 FF) had max-absolute error `0.001617432`, so its locked bound is `0.002`. The medium
differential attention, SAME-S block, SAME-L sinusoidal FF, and weight-normalized convolution were
all below `0.000088`; memory, patching, and SoftNorm paths were bit-identical. These are fp32
matmul-order bounds, not relaxed cosine substitutes.

Regenerate:

```text
<upstream>/.venv/bin/python scripts/reference/sa3_primitives_reference.py \
  --upstream <upstream> --small <small-model.safetensors> \
  --medium <medium-model.safetensors> --same-s <SAME-S-model.safetensors> \
  --same-l <SAME-L-model.safetensors> \
  --output docs/migration/sa3-primitives-reference
```

Verify without accessing model weights:

```text
python3 scripts/reference/sa3_primitives_reference.py --verify \
  --output docs/migration/sa3-primitives-reference
```
