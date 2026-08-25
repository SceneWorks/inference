# SC-17181 — MLX SDXL-family preview sigma audit

SC-17181 fixes the one latent-convention mismatch in the 38 registered MLX preview routes. The
shared preview emitter now has an additive sigma-aware path; existing zero-argument projectors still
run through the original API and are unchanged.

## Affected discrete SDXL-family routes

| route | running latent at preview time | correction |
| --- | --- | --- |
| SDXL curated samplers and CFG++ | raw k-diffusion VE `x0 + noise * sigma` | project `x / sqrt(sigma^2 + 1)` through the sigma-aware emitter |
| Kolors curated samplers | the same raw VE convention, through the re-exported SDXL driver | the same sigma-aware projector |
| Kolors native Euler, including img2img/control/IP compositions | raw diffusers Euler latent; `scale_model_input` computes the fitted domain | preview the already-computed model input |
| SDXL Lightning/LCM/TCD acceleration | sampler-specific input scaling | preview the already-computed model input |
| SDXL ancestral Euler | its running latent is already the fit domain and `scale_model_input` is identity | the same model-input seam is byte-identical |

The curated projector rejects a missing sigma instead of emitting an unscaled decorative frame.
The weights-free regression uses the same large-sigma fixture as the candle finding: the raw frame
clips more than half its RGB values to 0/255, while the corrected frame clips less than 0.10.

## Other registered preview families

Every route in `mlx-gen-catalog::PREVIEW_PROVIDER_IDS` was traced from its preview call to its
sampler convention:

| latent/route cohort | registered families | audit result |
| --- | --- | --- |
| Qwen/Krea 16-channel | Qwen-Image, Krea 2, Anima | flow-match running latents are the fitted domain; no sigma correction |
| FLUX.1 16-channel | FLUX.1, Chroma, PuLID-FLUX | flow-match input scale is identity; no correction |
| FLUX.2 32-channel | FLUX.2, Ideogram 4, Lens | flow-match input scale is identity; layout unpacking only |
| Z-Image 16-channel | Z-Image base/turbo and control variants | flow-match input scale is identity; layout unpacking only |
| SD3 16-channel | SD3.5 large/turbo/medium | flow-match running latents are the fitted domain; no correction |
| SANA 32-channel | SANA Base | flow-match running latent is the fitted domain |
| SANA 32-channel | SANA Sprint | already removes the separate SCM `sigma_data` prior scale explicitly; schedule-sigma scaling would be wrong |

The temporal/no-go families from epic 16624 remain unadvertised and outside this change: Wan, LTX,
Mage, Mochi, SVD, and SeedVR2. InstantID remains a struct-only composition rather than a registered
preview route; its SDXL denoise helpers preserve their existing inert default unless reached through
an advertised composition such as Kolors.

## Mac real-weight evidence

On 2026-08-20, the ignored release test
`registered_curated_route_keeps_the_first_frame_off_the_rails` ran on an Apple M5 Max against the
resolved SceneWorks RealVisXL MLX bf16 snapshot. The registered curated Euler route emitted all eight
requested previews, and the first frame's exact 0/255 rail fraction was `0.059896`, below the story's
`0.10` acceptance threshold.

The retained evidence bundle contains the eight numbered frames, a preview strip, and the final
image. As a mutation check, replacing the correction factor with `1.0` made
`ve_correction_removes_early_frame_saturation` fail its `< 0.10` assertion; restoring the factor made
the same test pass.
