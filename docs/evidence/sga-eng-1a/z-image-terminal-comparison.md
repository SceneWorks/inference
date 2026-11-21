# Candle Z-Image terminal comparison

The two native Candle CUDA cases use the published Diffusers model and the
request from the corresponding real-weight conformance test. Each compares the
same variant from the merged ENG-4a engine `e86b8d5fc493c16b2d9d852e0c9ecbdefee58294`
and the exact dispatched source on one leased physical Blackwell GPU (SM 12.0).

| Case | Official immutable snapshot | Native request |
| --- | --- | --- |
| Turbo | `Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6` | `z_image_turbo`, 256×256, four Euler steps with static shift 3, `a fox` |
| Base | `Tongyi-MAI/Z-Image@04cc4abb7c5069926f75c9bfde9ef43d49423021` | `z_image`, 512×512, twelve Euler steps with static shift 6, CFG 4.0, `a photo of a rusty robot holding a lit candle, dramatic cinematic lighting`, negative `blurry, low quality, watermark` |

Both requests use seed 42, count one, the resident BF16 compute route and the
native VAE. The provider casts the official Turbo FP32 transformer shards to
BF16. The Base transformer is published in BF16. The profile authenticates
the pinned input files before and after the pair and rejects extra
safetensors shards in the native component directories.

Both exact source builds compile the provider plan without loading a model
before the controller claims the GPU. A completed generation must match the
plan's exact sigma bits, complete every outer step, and report BF16 state
from the same CUDA device. The replay verifier checks shape, spatial variation, mean in
`(8, 248)`, standard deviation above 8, and identical RGB8 SHA-256 across
both source arms. This is an AP13(a) zero-difference proof for these requests.

Each variant has two unscored renders, one per source arm, followed by six
AB/BA pairs with one fresh process per arm and a warm OS file cache. Measure
the synchronized sampler interval per outer step and the end-to-end interval
from provider load through materialized RGB8. Compare six matched timing ratios
and their 96.875% min/max interval with the 1.02 ceiling. Compare the maximum
observed high-water in each matched memory domain with the same ceiling. The
private CUDA pool is 30 GiB. The guard budgets 34 GiB child GPU memory and
48 GiB child host memory with 4 GiB GPU and 8 GiB host reserves. Each child has
300 seconds. A ceiling overlap is inconclusive at this fixed budget.

This comparison covers the two plain text-to-image routes. The separate
`z-image-q4-dit` Tier 1 case covers packed resident and sidecar graphs with
LoRA. Combined FP8 imports, control, edit and training retain their
independent evidence requirements.
