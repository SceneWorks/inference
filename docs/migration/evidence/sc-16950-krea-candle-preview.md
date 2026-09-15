# sc-16950 candle Krea 2 latent-preview evidence

Epic 16948 wires `PreviewSink` into the candle engines. Krea is Tier 1: it reuses the QwenVae RGB fit
epic 16624 committed on the MLX side and adds **no** new fit. This file records what makes that reuse
legitimate and what the real-weight run actually showed.

## The fit is reused, not refitted

`crates/media/candle-gen/candle-gen-qwen-image/src/preview.rs` carries `RGB_FACTORS` / `RGB_BIAS`
transcribed verbatim from `crates/media/mlx-gen/mlx-gen-qwen-image/src/preview.rs`. There is
deliberately no candle producer: a second least-squares solve of the same latent space would be a
second source of truth for one set of numbers. The MLX producer
(`mlx-gen-qwen-image/tests/fit_preview_rgb.rs`) remains the only way these constants are re-derived.

## Learned-basis transfer — grounded in tensor bytes

The claim being checked is not "both crates name a type `QwenVae`" but "candle Krea loads the same VAE
weights the fit was measured against".

| snapshot | revision | `vae/diffusion_pytorch_model.safetensors` SHA-256 |
| --- | --- | --- |
| `krea/Krea-2-Turbo` | `1161245028ef398cd0a951101b2bbf486464f841` | `ab1b61103959913d6c7e628cf793dbb2ca4726a40a3b3ae206c52b8e75bf6f08` |
| `krea/Krea-2-Raw` | `4ad9f4b627a647fad78b3dfeebb09f2654aeb494` | `ab1b61103959913d6c7e628cf793dbb2ca4726a40a3b3ae206c52b8e75bf6f08` |
| `SceneWorks/qwen-image-mlx` (`q4/` and `q8/` share one file) | `8080a4171f1c8b7fca6c30491eafbe6ffab754bf` | `0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344` |

- Krea Turbo and Krea Raw publish the **same file**, so every candle Krea route — turbo, raw, edit,
  control, multi-phase, both img2img siblings — loads one set of VAE bytes. `crate::vae::load_vae`
  reads `<snapshot root>/vae`, and the edit LoRA (`SceneWorks/krea-edit`), the identity-edit LoRA and
  the pose-control overlay (`SceneWorks/krea2-pose-controlnet-beta`) are all single-file overlays on
  that same snapshot, contributing no VAE of their own.
- Against the Qwen-Image snapshot the MLX fit was measured on, the transfer is **exact**: 194 of 194
  tensors, 126,892,531 values, identical. The two files differ only in container width — the published
  Krea `vae/` is an f32 container in which *every* value has zero low-16 mantissa bits (i.e. is exactly
  bf16-representable), and the MLX snapshot stores those same values as bf16. This is the sc-16630
  Lens/Ideogram situation with the stronger outcome: not "rounds exactly", but equal.
- `latents_mean` / `latents_std` — the per-channel de-normalization that *defines* the normalized
  16-channel space the fit lives in — are identical in both `vae/config.json` files. Krea's config also
  carries `_class_name = "AutoencoderKLQwenImage"` and `_name_or_path = "Qwen/Qwen-Image"`.

Reproduce (the row prints all three hashes and the tensor comparison):

```sh
KREA_TURBO_DIR=/path/to/Krea-2-Turbo \
KREA_RAW_DIR=/path/to/Krea-2-Raw \
QWEN_IMAGE_VAE_FILE=/path/to/qwen-image-mlx/q8/vae/diffusion_pytorch_model.safetensors \
  cargo test --release -p candle-gen-krea --test preview_real_weights \
    krea_vae_is_the_qwen_fit_vae -- --ignored --nocapture
```

The hashes are pinned as constants in that test, so a snapshot swap fails here rather than silently
applying a fit that belongs to a different latent space.

## Wiring

Krea keeps its denoise state in the spatial QwenVae layout `[1, 16, H/8, W/8]` at every step of every
route, so wiring is the sc-16949 projector hook and nothing else — no route restructures its loop and
no route unpacks anything.

| route | site |
| --- | --- |
| Turbo t2i (three-stage residency) | `pipeline::render_three_stage` |
| Turbo t2i (resident / sequential) | `pipeline::render_from_context` |
| Turbo img2img | `pipeline::render_img2img_from_context` |
| Raw t2i (true CFG) | `pipeline::render_base_from_contexts` |
| Raw multi-phase | `pipeline::render_multiphase` |
| Raw img2img (true CFG) | `pipeline::render_base_img2img_from_contexts` |
| Edit — Turbo **and** Raw | `pipeline::render_edit_from_context` |
| Pose control | `control_provider::Krea2ControlHeavy::render` |

`preview::tests::every_krea_render_route_passes_a_preview_hook` walks the source of both files and
fails if any `run_flow_sampler` site is missing its hook, so a route added later cannot be silently
left dark. The trainer's periodic sample render (`training::render_sample`) is the one deliberate
exception — it renders from a synthetic request that carries no sink — and is pinned as such.

Two properties are structural rather than defensive, and are pinned by
`preview::tests::sampler_hook_only_ever_sees_the_single_target_latent`:

- **CFG never reaches the preview.** Every true-CFG Krea route runs both forwards inside the predict
  closure and returns one combined velocity, so there is no fused `[2, …]` batch in the sampler at all.
- **Edit references never reach the preview.** `forward_edit_with_memory` concatenates the encoded
  reference latents into the DiT sequence internally; the running latent stays the target alone.

Multi-phase is the one route that needs more than the default hook: it resolves ONE global σ schedule
and drives a sampler call per phase over a contiguous slice, so it supplies a global counter through
`candle_gen::preview::PreviewHook::over_schedule` and its frames run `1..=total` across phase
boundaries instead of restarting at each one — matching `mlx-gen-krea`.

`Capabilities.supports_preview` stays `false` here by design; sc-16951 flips it alongside the
`candle-gen-catalog` bidirectional guard.

## Real-weight run (CUDA)

`turbo_preview_frames_evolve_toward_the_final_image` renders the registered `krea_2_turbo` engine
twice on one warmed generator at the same seed — once with an inert sink, once with a live one — and
asserts:

- the active render's output RGB8 bytes equal the inert render's, byte for byte;
- an N-step render emits exactly N frames numbered `1..=N`, each carrying `total == N`;
- every frame is latent-resolution (`size/8` square);
- consecutive frames differ (mean |Δ| > 0.5), so the strip is not N copies of one image;
- distance to the finished render falls at **every** step and ends under 60 % of the first frame's;
- resemblance — Pearson correlation against a 16² thumbnail of the finished render — rises at every
  step, starts below 0.35 (the first frame is pre-denoise noise and must *not* already look like the
  result) and ends above 0.85.

Correlation rather than absolute distance is the resemblance metric on purpose. The hook emits
*before* each solver advancement, so the last frame is always one step short of the render; and the
projection is a global linear approximation of the decode (R² 0.9586), so even a fully converged
latent keeps an offset and gain error against the true pixels. Both effects inflate an absolute
distance without saying anything about whether the preview looks like the image.

```sh
KREA_TURBO_DIR=/path/to/Krea-2-Turbo \
KREA_PREVIEW_ARTIFACT_DIR=/path/to/retained-output \
  cargo test --release --features cuda -p candle-gen-krea --test preview_real_weights \
    turbo_preview_frames_evolve_toward_the_final_image -- --ignored --nocapture
```

Retained run — `krea/Krea-2-Turbo` @ `1161245028ef398cd0a951101b2bbf486464f841`, 1024², 8-step Turbo,
seed 0, prompt "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour",
`RTX PRO 6000 Blackwell`, CUDA 12.9 / MSVC 14.44, `CUDA_COMPUTE_CAP=120`:

- exactly 8 frames, numbered 1..=8, `total = 8`, each 128×128 (1024 / 8);
- active-sink and inert-sink output RGB8 bytes **exactly equal** at the same seed;
- distance to the finished render fell monotonically, 79.49 → 26.35 mean |Δ| (0.33× the first frame);
- resemblance rose monotonically at every step:

  | frame | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | correlation with final | +0.144 | +0.422 | +0.567 | +0.709 | +0.831 | +0.917 | +0.965 | +0.987 |
  | coarse mean \|Δ\| | 54.33 | 51.10 | 48.85 | 45.84 | 41.67 | 35.82 | 27.90 | 16.98 |

The retained strip is committed at
[`sc-16950/turbo-1024-8step-strip.png`](sc-16950/turbo-1024-8step-strip.png) — the eight 128² frames
side by side, left to right. It was reviewed directly. Frame 1 is uniform colour noise with no
structure. The fox's
mass and the snow/tree-line banding are discernible by frame 4–5, the subject is unambiguous by frame
6, and frames 7–8 read as a lower-resolution, noisier version of the finished render with the same
pose, framing and palette. This is a clear noise-to-image progression, not eight views of noise and not
eight copies of one image.

The provenance row was run on the same box against the cached snapshots and reported the exact
194-tensor / 126,892,531-value transfer quoted above.

## Limits of this evidence

- The Raw, edit, control, img2img and multi-phase routes share this exact projector and latent layout,
  and their wiring is pinned at the source level, but the retained real-weight strip is Turbo t2i. The
  numbering, dedup and swallow behaviour of the other routes is covered weights-free.
- The multi-eval one-frame-per-step property is proven weights-free against `heun` and `dpmpp_sde`
  driving the real `run_flow_sampler`, with an assertion that those solvers really did evaluate more
  than once per step (otherwise the row would pass vacuously against an Euler fallback).
- A preview is decorative: the projection is a global linear map at 1/8 resolution, so fine detail is
  absent by construction. It is a progress indicator, not a substitute VAE decode.
