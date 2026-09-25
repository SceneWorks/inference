# Krea Realtime Q4 smoke repair (SC-24196)

## Reproduced failures and scope

The rc.2 job at inference `57ae17d01f335f1d2f3bbf328b84424a56b66ceb`
[ran four of six tests successfully](https://github.com/SceneWorks/inference/actions/runs/36094774959/job/108043079680).
The strength-zero VideoClip request omitted the now-required `video_to_video` mode.
The generated-latent decode sweep failed its negative control: old-window mean
error was 4.28/255, below its assumed 5/255 floor, while the product plan was
0.46/255. Both existing product-quality conditions were satisfied.

The earlier [24 August failure](https://github.com/SceneWorks/inference/actions/runs/32685027164/job/97308449384)
was a Metal timeout during v2v generation, **not** a failed pixel-identity assertion.
Its sweep passed: old-window error 10.44/255, product error 0.10/255.
These findings do not implicate epic 24128.

## Controlled decoder comparison

On 25 September, a temporary diagnostic on the rc.2 source generated one latent
tensor and decoded that **same tensor** with fixed VAE weights using both decoder
partitions. The earlier partition is the `decode_tiled` body at `6f3a84ef4`:
denormalize once, then run `conv2` and the entire decoder per tile. The current
partition runs `conv2` and the middle blocks densely, then tiles the upsample tail.
The temporary legacy implementation was removed after measurement.

Environment: Apple M5 Max, 128 GiB, Rust 1.96.0, release build; MLX binding
`d5a7fc018d713a37091e1cd102873eab355a00c6`, explicit matching 26.2/Release metallib.
Snapshot `SceneWorks/krea-realtime-14b-mlx` at
`e68e9a3d98187fdf6936838ffcf6df5aa48d6626`, Q4; 832x480, 33 output frames;
the existing fox prompt and seed 7. Q4 DiT SHA-256:
`fe93dd4be3f261cc05f27100a5e299c1c4ccc84269902f97d3ecb05c929569ed`;
VAE SHA-256: `42159a8b571dbeb3ea40327b88a6161a5342c0511202af7c031360629757163d`.

| Decode of the same latents | Mean absolute error versus single-pass (/255) |
| --- | ---: |
| Whole decoder per tile, old 8/4 window | 16.943128 |
| Current decoder partition, old 8/4 window | 10.439532 |
| Current product plan, spatial 256/64, no temporal tiling | 0.104532 |

This proves the decoder partition affects the control. It does **not** establish
why rc.2 generated different content: the standalone run reproduced the August
statistics despite using current source. The sampler, prompt and seed were
unchanged between the supplied passing revision and rc.2, but the MLX revision
also changed. A complete attribution of historical content differences remains
unproven.

The first local six-case diagnostic encountered the reported UMT5 Metal timeout
after the sibling's 90 GiB single-pass experiment; it was interrupted. The sweep
then passed alone, including the paired measurement, in 510.08 seconds. The final
harness clears MLX's process-wide unused cache before snapshot loading so one
case's decode scratch is not carried into the next case. This is resource
isolation, not a proven cure for every historical Metal timeout.

## Final regression contract

- Share the strength-zero request builder between the weights-free contract test
  and the real-weight test. Removing its explicit video mode must be rejected.
- Keep the fixed-source sibling's `old_err > 5` known-corruption gate unchanged.
- The generated-content sweep retains **both** `d_product < 4` and
  `d_product < d_old * 0.35`, plus its spatial-error, seam and memory gates.
  Replacing the product plan with the old plan still fails the relative gate;
  identical/flat decodes with both errors zero also fail it. Remove only the
  duplicate absolute corruption floor on generated content.
- Remove the 8/8 row: latent overlap is clamped to `tile - 1`, so it runs the same
  latent 2/1 plan as 8/4. There is no additional coverage in that row.
- Retain the workflow's exact six-passing-test gate and every identity/coherence
  threshold. No production decoder, sampler, or validation rule changes.

The 17 non-ignored `generate_smoke` tests pass locally. Final six-case real-weight
acceptance and CI results are recorded on SC-24196 and its pull request; the
standalone diagnostic is not a substitute for those gates.
