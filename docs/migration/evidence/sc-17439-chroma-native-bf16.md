# sc-17439: Chroma native-BF16 T5 and VAE CUDA evidence

Date: 2026-08-04

## Verdict

The Candle Chroma T5-XXL and FLUX.1 VAE can remain at their published BF16 width. All three q4
Chroma entries completed the same three-image CUDA comparison against the former F32 loader. Every
image cleared the pre-merge parity floor of PSNR >= 35 dB, global RGB SSIM >= 0.98, and RGB cosine
>= 0.98. The measured settled device residency fell by 11.3-11.9 GB per loaded entry.

The Chroma DiT remains F32. The T5 sequence embedding is promoted by the existing
`context_embedder` input cast, and the decoded BF16 VAE output is promoted to F32 before RGB
conversion.

## Environment and immutable inputs

- Host: Windows, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB VRAM.
- Driver / toolkit: 596.36 / CUDA 12.9; release build with Visual Studio 2022 Build Tools.
- F32 baseline: inference `origin/main` at `67bcbeb4d0d35817b24df4a7e9f84695daa8c23f`.
- BF16 candidate: the commit containing this record.
- `chroma1_base`: `SceneWorks/chroma1-base-mlx@e7330dda29d00ffdeeb719b28e92ee74cff0884c/q4`.
- `chroma1_hd`: `SceneWorks/chroma1-hd-mlx@9d99afe1ebca67032476756bc70d4a7152bc1bd5/q4`.
- `chroma1_flash`: `SceneWorks/chroma1-flash-mlx@6a9cb6178709559461506bf247f708d0d1008d00/q4`.

Both binaries were built independently from separate worktrees and target directories:

```text
cargo build --locked --release -p candle-gen-chroma \
  --example chroma-txt2img --features cuda
```

The comparison used GPU 0 while otherwise idle. `--measure-vram` uses the shared device-level
`VramProbe`: its phase covers the cold load plus render, then samples settled residency after the
call while the generator still holds every component. Reported quantities subtract each run's
recorded idle baseline; every baseline was 0.0-0.5 GB and passed the harness's `< 1.0 GB` trust gate.

## Identical render inputs

- Prompt: `a photo of a rusty robot holding a lit candle, dramatic cinematic lighting, highly detailed`
- Negative prompt for Base / HD: `blurry, low quality, malformed`
- Seeds: 17439, 17440, 17441 (`--seed 17439 --count 3`)
- Geometry: 512x512
- Steps: 8
- Variant-native true CFG: Base / HD 4.0; Flash 1.0 (single-forward)
- Sampler / scheduler: the provider defaults, unchanged between binaries

Command shape, run once with the F32 binary and once with the BF16 binary for each immutable
snapshot:

```text
chroma-txt2img.exe --snapshot <snapshot>/q4 --model <base|hd|flash> \
  --prompt "a photo of a rusty robot holding a lit candle, dramatic cinematic lighting, highly detailed" \
  [--negative "blurry, low quality, malformed"] \
  --seed 17439 --steps 8 --width 512 --height 512 --count 3 --repeat 1 \
  --measure-vram --out <variant>-<f32|bf16>.png
```

## Per-image comparison

Metrics are computed over the complete 512x512 RGB images, never a crop or spot sample. PSNR uses
the full-image RGB RMSE. Global SSIM is the mean of the three channel SSIM values with the standard
`C1=(0.01*255)^2` and `C2=(0.03*255)^2` constants. Cosine is over all flattened RGB values.

| entry | seed | MAE | RMSE | PSNR dB | global SSIM | RGB cosine | max channel delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 17439 | 0.468816 | 0.959792 | 48.4873 | 0.99784523 | 0.99920283 | 43 |
| Base | 17440 | 0.382731 | 0.665831 | 51.6635 | 0.99795738 | 0.99948583 | 22 |
| Base | 17441 | 0.530589 | 1.341691 | 45.5778 | 0.99673863 | 0.99806113 | 80 |
| HD | 17439 | 1.011929 | 4.024395 | 36.0368 | 0.98521089 | 0.98966859 | 196 |
| HD | 17440 | 0.200812 | 0.776212 | 50.3312 | 0.99498611 | 0.99628732 | 69 |
| HD | 17441 | 0.559998 | 1.262276 | 46.1077 | 0.99485978 | 0.99705208 | 82 |
| Flash | 17439 | 0.699033 | 1.400107 | 45.2076 | 0.99818224 | 0.99966845 | 51 |
| Flash | 17440 | 0.556273 | 1.152147 | 46.9006 | 0.99920015 | 0.99985920 | 141 |
| Flash | 17441 | 0.555429 | 1.022528 | 47.9373 | 0.99948662 | 0.99995704 | 41 |

The lowest-scoring pair, HD seed 17439, was also inspected visually side by side. Composition,
subject placement, candle lighting, pose, silhouette, and fine surface texture remain aligned; the
largest local drift is confined to small eye/detail pixels.

## Device residency and timing

`VramProbe` reports decimal GB, matching the repository's memory measurement convention.

| entry | F32 settled | BF16 settled | measured reduction | F32 3-image time | BF16 3-image time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 31.2 GB | 19.3 GB | 11.9 GB | 84.5 s | 80.0 s |
| HD | 30.6 GB | 19.3 GB | 11.3 GB | 89.3 s | 77.8 s |
| Flash | 30.6 GB | 19.3 GB | 11.3 GB | 59.8 s | 63.4 s |

At this geometry the cold peak equaled the post-call settled value in every run. The measured
11.3-11.9 GB reduction is retained as the outcome; the ledger must not substitute a theoretical 2x
weight-width estimate for the observed device delta.
