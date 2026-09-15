# sc-17218 — Candle Boogu per-step latent previews

Validated 2026-08-20 on the inference story branch from `f17c82544558666f8d843b4e344e3efa0fd08ed7`.
This is the final implementation and CUDA acceptance record for epic 16948.

## Decision

All three registered Candle Boogu routes are preview-capable:

| Route | Default denoise lane | Preview seam | Observation point |
|---|---|---|---|
| `boogu_image` | shared flow sampler, true CFG | hooked | running native latent entering each outer step |
| `boogu_image_turbo` | provider-owned DMD student loop | direct `emit_step` | initial noise, then the previous clean estimate after re-noise |
| `boogu_image_edit` | shared flow sampler, true CFG with reference context | hooked | target running native latent entering each outer step |

Turbo also takes the hooked shared-driver lane for img2img or when a curated sampler/scheduler is
selected. The default native DMD loop previews **before** prediction. It therefore does not show the
transient clean estimate that exists between prediction and re-noise; it shows the state the next DiT
evaluation actually consumes, matching the shared sampler contract. This avoids a route-dependent
observation point and guarantees one frame per outer step.

CFG stays inside each route's prediction closure. The integrated tensor remains batch-1 target latent,
so the preview never sees an unconditional half or Edit's reference context.

## Fit reuse and tensor identity

Boogu denoises an unpacked `[1, 16, h, w]` latent. It reuses
`candle_gen_flux::preview::project_raw_latents`; it does not copy the FLUX.1 coefficients and does not
use FLUX.1's packed-token projector.

The deployed SceneWorks snapshot is `SceneWorks/boogu-image-mlx` revision
`a459e614d408bfdf57089c32cc3da706f5a017de`. Every advertised Q8/Q4 route tier resolves to this VAE
container:

| Tier | VAE SHA-256 |
|---|---|
| `base`, `base-q4` | `8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94` |
| `turbo`, `turbo-q4` | `8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94` |
| `edit`, `edit-q4` | `8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94` |

The upstream `Boogu/Boogu-Image-0.1-Turbo` snapshot
`7c475e94ddb10529daa9142942d297675dde1acc` has the same VAE container hash. FLUX.1-dev revision
`3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` uses a bf16 container with SHA-256
`f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`.

The existing tensor validator was run against the deployed Q4 VAE and that FLUX.1-dev donor:

```text
boogu vs flux1: 244 tensors, 83819683 values, bf16-round-identical
```

The configs also agree on a 16-channel plain `AutoencoderKL`, scaling `0.3611`, and shift `0.1159`.
Container hashes differ because Boogu stores f32 while the donor stores bf16; tensor-value identity at
the donor dtype is the relevant claim.

## Wiring and weights-free validation

`candle-gen-boogu/src/pipeline.rs` contains three hooked shared-driver calls (Base, Turbo curated/img2img,
Edit) and one direct native-Turbo emission. The catalog pins that exact `hooked: 3`, `direct: 1`, no-dark
site inventory. All three descriptors advertise `supports_preview: true`, all three exact ids are in
`PREVIEW_ROUTE_IDS`, and the deferred class is now empty.

Focused results:

```text
cargo test -p candle-gen-boogu --lib
71 passed; 0 failed; 2 ignored

cargo test -p candle-gen-catalog preview_advertising
20 passed; 0 failed
```

The Boogu tests drive a real OneMinusSigma flow schedule with Heun, first proving model evaluations
exceed outer steps, then proving frames remain exactly one per outer step. They also compare the final
latent from no hook, an inert sink, and an active sink; all values are identical.

## One-time CUDA acceptance

This was intentionally a **one-time acceptance run**, not a recurring gate. The harness is
`#[ignore]`d and no workflow invokes it. It must not be added to CI; future reruns require an explicit
request.

Host:

- NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB
- driver 596.36; CUDA toolkit 12.9
- Rust/Cargo 1.96.0
- Q4 Base, Turbo, and Edit packages from the SceneWorks revision above
- 512×512, seed 17,218

Each strip appends the finished decoded image as its final tile. Full metrics are committed beside each
strip in `docs/migration/evidence/sc-17218/`.

The reused FLUX.1 fit's in-sample `R² = 0.98224` sets a correlation ceiling of
`sqrt(0.98224) = 0.9911`; a floor cannot honestly exceed it. The schedule determines how much of that
ceiling the last **pre-step** preview can reach because the terminal advancement is not previewed. Each
floor is therefore a separate fraction of the common fit-derived ceiling, supported by that lane's
retained measurement and rounded down with about 0.03 absolute headroom:

- Base default: `0.960 / 0.9911 = 96.9%` of ceiling; measured `+0.9890`.
- Base Heun: `0.950 / 0.9911 = 95.9%`; measured `+0.9824`.
- Turbo native DMD: `0.945 / 0.9911 = 95.3%`; measured `+0.9772`.
- Edit default: `0.940 / 0.9911 = 94.8%`; measured `+0.9706`.

These are deliberately four constants, not one family threshold: Base's shifted Euler schedule, Heun's
multi-eval advancement, Turbo's re-noised DMD state, and Edit's reference-conditioned flow path leave
different terminal shares even though they reuse one fit.

| Lane | Frames / outer steps | Progress evaluations | First → last correlation | First → last mean distance | Floor |
|---|---:|---:|---:|---:|---:|
| Base default, 8 steps | 8 / 8 | 8 | `+0.0335 → +0.9890` | `84.92 → 13.58` | `0.96` |
| Base Heun, 4 steps | 4 / 4 | `> 4` | `+0.0081 → +0.9824` | `84.11 → 29.96` | `0.95` |
| Turbo native DMD, 4 steps | 4 / 4 | 4 | `-0.0080 → +0.9772` | `73.40 → 23.22` | `0.945` |
| Edit default, 8 steps | 8 / 8 | 8 | `-0.0645 → +0.9706` | `95.66 → 12.88` | `0.94` |

For every lane, coarse correlation rose at every frame, distance to the final image fell at every
frame, and every consecutive preview differed. Base Heun emitted exactly four numbered frames even
though its progress callback proved more than four model evaluations. Turbo was rendered twice on the
same warmed generator and seed, first with an inert sink and then with an active sink; final pixel bytes
were identical.

Boogu uses `OneMinusSigma` flow sampling, whose `input_scale` is exactly `1.0` across the schedule, so
the preview hook intentionally has no sigma correction. As the cheap clipping check, the first tile of
each retained strip has a rail-clipped RGB-channel fraction of `0.000570` (values equal to 0 or 255),
well below the `0.05` readability bound. A weights-free unit-normal first-state test independently pins
the same convention and bound.

Retained strips:

- [Base default](sc-17218/boogu-base-default-strip.png)
- [Base Heun](sc-17218/boogu-base-heun-strip.png)
- [Turbo native](sc-17218/boogu-turbo-native-strip.png)
- [Edit default](sc-17218/boogu-edit-default-strip.png)

## Result

Boogu's three registered routes satisfy epic 16948's per-step preview contract without a refit, without
changing denoise results, and without creating a recurring CUDA/CI cost. There are no remaining
deferred Candle preview routes in the catalog.
