# sc-16630 FLUX.2-family latent preview evidence

## Projection fit

`mlx-gen-flux2/tests/fit_preview_rgb.rs` is the runnable real-weight producer. It fits an ordinary
least-squares RGB projection from the de-normalized, unpatchified raw `[1, 32, H/8, W/8]` VAE latent
to an 8×8-average-pooled native decode. The transformer-space `[1, H/16·W/16, 128]` tokens are not
used as projection inputs.

The retained run used FLUX.2 Klein 9B, 256², eight flow-Euler steps, eight diverse fit prompts and
four disjoint prompt/seed holdouts:

- fit R² RGB `(0.77472, 0.77362, 0.74224)`, overall `0.76409`;
- holdout R² RGB `(0.64247, 0.71049, 0.72381)`, overall `0.69504`;
- exact fixed-seed equality between the legacy sampler and explicit inert latent-hook path for both
  final raw-latent f32 values and final decoded RGB8 bytes.

Corpus-specific regression floors are encoded in the producer: `0.70` per fit channel, `0.58` per
holdout channel, and `0.65` holdout overall. Reproduce with:

```sh
MLX_GEN_FLUX2_SNAPSHOT=/path/to/flux2-klein-9b/bf16 \
FLUX2_PREVIEW_ARTIFACT_DIR=/path/to/retained-output \
cargo test --locked --release -p mlx-gen-flux2 --test fit_preview_rgb -- --ignored --nocapture
```

The retained decoded/projected comparison set was reviewed directly. Across every fit and holdout,
the projection preserved dominant palette, large foreground/background masses, and subject placement.
The disjoint holdouts retained library warm-interior/cool-window zoning, flower placement, dog/beach/sky
bands, and the red-car/night foreground mass. Fine detail is absent and the output is visibly blocky at
32², as expected for a decorative denoise-progress preview rather than a substitute VAE decode.

## Learned-basis transfer

The transfer claim is tensor-value based:

- FLUX.2 bf16 VAE SHA-256:
  `ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04`.
- Lens base q4 and Lens turbo bf16 VAE SHA-256:
  `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5`.
  The Lens files are f32 rather than bf16; all 251 tensors / 84,046,371 values round exactly,
  round-to-nearest-even, to the FLUX.2 bf16 values. Lens base and turbo files are byte-identical.
- Ideogram 4 q4 VAE SHA-256:
  `bb9ba30dec375f7fef52a4e47cda26e9354082710849d531df69eca724ce3bc9`.
  All 250 learned tensors are byte-identical to FLUX.2. The sole file-level tensor difference is the
  unused `bn.num_batches_tracked` scalar, stored as i32 by Ideogram and i64 by FLUX.2.

Lens therefore has an exact learned-basis transfer despite different container precision. Ideogram
has an exact learned-weight transfer despite different metadata/tracking-scalar serialization. Its
DiT's patch-major `(ph,pw,c)` packing remains distinct and is explicitly unpatchified before projection.

## Runtime evidence limits

The public `flux2_klein_9b` registry route was also run with an active sink at 256², eight Euler
steps. It emitted exactly eight 32² frames numbered 1 through 8; a fixed-seed active render and inert
render had byte-identical final RGB8 output. Direct review of the retained strip found a clear
noise-to-image progression, with mountain/sky/lake bands and the fox's foreground mass emerging by
the later frames. Reproduce with:

```sh
MLX_GEN_FLUX2_SNAPSHOT=/path/to/flux2-klein-9b/bf16 \
FLUX2_PREVIEW_ARTIFACT_DIR=/path/to/retained-output \
cargo test --locked --release -p mlx-gen-flux2 --test preview_real_weights \
  -- --ignored --nocapture
```

Lens-Turbo received a separate controlled real-weight transfer run. The test encodes once, drops the
text encoder, warms the dense `LensHeavy` generator with an inert 256² four-step render, then compares
fixed-seed active and inert renders on that same warmed generator. The active/inert final RGB8 bytes
were exactly equal and the active run emitted exactly four 32² frames numbered 1 through 4. The final
image was coherent and on-prompt. The four pre-step Turbo frames remained noise-like, however: this
four-step strip does **not** demonstrate useful compositional progression before the final solver
advancement, and no stronger Lens-Turbo preview-quality claim is made. Reproduce with:

```sh
LENS_PREVIEW_SNAPSHOT=/path/to/lens-turbo/bf16 \
LENS_PREVIEW_ARTIFACT_DIR=/path/to/retained-output \
cargo test --locked --release -p mlx-gen-lens --test preview_real_weights \
  -- --ignored --nocapture
```

Only the Ideogram q4 VAE was materialized for provenance verification; the text encoder, conditional
DiT, unconditional DiT, and turbo LoRA were not downloaded. Therefore no claim of a full Ideogram
provider runtime render is made here. Ideogram's exact learned-weight transfer and provider-owned
patch-major unpatch path are covered without substituting that evidence for a full render.
