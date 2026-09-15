# sc-16953 candle Anima latent-preview evidence

Epic 16948 wires `PreviewSink` into the candle engines. Anima is Tier 1 and the third QwenVae family:
it reuses the RGB fit epic 16624 committed on the MLX side and adds **no** new fit. This file records
what makes that reuse legitimate and what the real-weight run actually showed.

## The fit is reused, not refitted

`crates/media/candle-gen/candle-gen-anima/src/preview.rs` defines **no constants**. It calls
`candle_gen_qwen_image::preview::project_spatial_latents`, the seam sc-16952 documented, and
re-exports that crate's `PREVIEW_LATENT_CHANNELS` rather than restating the number. There is
deliberately no candle producer of the coefficients: the MLX producer
(`mlx-gen-qwen-image/tests/fit_preview_rgb.rs`) remains the only way they are re-derived.

`preview::tests::the_fit_is_the_shared_qwenvae_one` projects the same latent through both the Anima
seam and the Qwen-Image seam and requires the pixels to be equal, so a copy of the constants could not
be introduced here without failing.

## Learned-basis transfer — grounded in tensor bytes

The claim being checked is not "both crates name a type `QwenVae`" but "candle Anima loads the same
VAE weights the fit was measured against". Anima publishes those weights as a **single file in the
original Qwen naming**, so it is genuinely a different file from the fit donor and a hash equality
would not have settled anything. `crate::vae::convert_vae_key` — the production rename `QwenVae` reads
Anima's checkpoint through — is what makes the two comparable.

| snapshot | revision | file | SHA-256 | bytes |
| --- | --- | --- | --- | --- |
| `circlestone-labs/Anima` | `53eec3898af698b2cf2a11379021fc9c5465d228` | `split_files/vae/qwen_image_vae.safetensors` | `a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f` | 253,806,246 |
| `SceneWorks/qwen-image-mlx` (`q4/` and `q8/` share one file) | `8080a4171f1c8b7fca6c30491eafbe6ffab754bf` | `q8/vae/diffusion_pytorch_model.safetensors` | `0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344` | 253,806,966 |

Measured on the CUDA box against both cached snapshots:

- Under `convert_vae_key` the Anima key set maps **bijectively** onto the fit donor's — no collision,
  no orphan in either direction — which is also why `QwenVae` can read Anima's file at all.
- **194 of 194 tensors, 126,892,531 values, bit-identical.** Both files are bf16 containers, so this
  is a strictly stronger result than sc-16950's Krea comparison, which had an f32-vs-bf16 container
  difference to argue past, and it matches sc-16952's byte-identity outcome in substance.
- The 720-byte file-size difference is the safetensors **header alone**: the longer diffusers key
  names, plus the payload re-ordering that follows from sorting a different name set. The two payload
  regions therefore hash differently while every individual tensor is equal — which is exactly the
  situation a file hash cannot see and a tensor comparison can.
- One Anima variant is not "the" variant here: all three DiT checkpoints
  (`anima-base-v1.0`, `anima-aesthetic-v1.0`, `anima-turbo-v1.0`) live under one `split_files/` root
  with one `vae/` shard, so this single file is the VAE every Anima route loads.

`latents_mean` / `latents_std` — the per-channel de-normalization that *defines* the normalized
16-channel space the fit lives in — need no cross-file comparison here and get none. Anima publishes
no `vae/config.json`; candle's `QwenVae` carries those values as Rust constants and this crate reuses
that very type, so the de-normalization is definitionally the same code rather than two files that
happen to agree. `anima_vae_bytes_are_the_pinned_snapshot` asserts the config's *absence*, so whoever
adds one later has to decide which is authoritative.

Reproduce (both rows fail rather than skip when their inputs are unset):

```sh
ANIMA_PREVIEW_DIR=E:\huggingface\hub\models--circlestone-labs--Anima\snapshots\53eec38…\split_files \
ANIMA_QWEN_FIT_VAE=E:\huggingface\hub\models--SceneWorks--qwen-image-mlx\snapshots\8080a41…\q8\vae\diffusion_pytorch_model.safetensors \
  cargo test --release -p candle-gen-anima --test preview_real_weights \
    anima_vae -- --ignored --nocapture
```

## The latent shape at the emission point — verified, not assumed

Anima denoises in a **5-D Cosmos** latent, not the 4-D spatial one Krea uses and not the packed token
space Qwen-Image uses. `pipeline::create_noise` samples `[1, 16, 1, H/8, W/8]` and
`transformer::CosmosDiT::forward` unpatchifies back to that same rank, so every `run_flow_sampler`
running latent stays 5-D from the first σ to the last.

That is rank 5, so handing it straight to `candle_gen::preview::project_latents` fails the
`[1, C, h, w]` contract outright. `preview::project_single_frame_latents` drops the length-1 temporal
axis first — **the same squeeze the decode tail already applies** before `QwenVae::decode` — and only
then applies the fit. Because the geometry travels entirely inside the latent, the projector takes no
`width`/`height`: hook geometry and latent geometry are not merely bound to one source, there is only
one source to be bound to. The emitted frames measuring exactly `H/8 × W/8` on the real renders below
is the runtime confirmation that the squeeze ran; an un-squeezed projection could not have produced a
frame at all.

The layout adaptation lives in `candle-gen-anima` rather than in `candle-gen-qwen-image` (where the
MLX twin put it) because on candle the 5-D Cosmos layout is Anima's alone — candle Qwen-Image denoises
packed. `project_spatial_latents` is the shared reuse seam for the fitted coefficients; each family
owns its own way of reaching it.

## Wiring

| route id | render lane | CFG |
| --- | --- | --- |
| `anima_base` | `pipeline::AnimaPipeline::generate` | true CFG, 2 DiT forwards/eval |
| `anima_aesthetic` | `pipeline::AnimaPipeline::generate` | true CFG, 2 DiT forwards/eval |
| `anima_turbo` | `pipeline::AnimaPipeline::generate` | merged CFG-free student, 1 forward |

**Three ids, one render lane.** All three variants are the same architecture and differ only in the
DiT weights file, so a single hooked `run_flow_sampler` site wires the whole family. This is the
mirror image of sc-16952, where one id (`qwen_image`) covered three lanes — the two cases together are
why the catalog keeps ids and route inventories as separate counts and infers neither from the other.

Opting in is the sc-16949 projector hook and nothing else: the driver owns frame numbering, multi-eval
dedup and the swallow-on-failure contract, and the denoise loop is unchanged. The `PreviewSink`
travels as a `GenOptions` field so the hook is built next to the sampler call it feeds.

Two properties are structural rather than defensive:

- **CFG never reaches the preview.** `anima_base` / `anima_aesthetic` run the conditional and
  unconditional DiT forwards *inside* the predict closure and return one combined velocity
  (`v_uncond + guidance·(v_cond − v_uncond)`). No fused `[2, …]` batch exists anywhere in the sampler,
  so there is no unconditional half for a preview to project. Pinned by
  `preview::tests::cfg_never_exposes_the_unconditional_half_to_the_preview`, which drives the real
  sampler with a closure shaped like the render lane's.
- **There are no reference or control tokens to leak.** Anima is txt2img only — `load_variant` rejects
  `spec.control`, `spec.extra_controls` and `spec.ip_adapter` — and the text conditioning reaches the
  DiT as `encoder_hidden_states`, never as part of the latent. There is no edit route to constrain.

### Guards, and what happens when they are mutated

Two independent readings of the same sources, both amended in this PR:

- `preview::tests::the_render_lane_passes_a_preview_hook` (crate-local) parses `pipeline.rs`,
  requires exactly one sampler site, and asserts the preview argument **positionally**.
  `no_other_shipped_module_drives_a_sampler` plus `the_negative_pin_covers_every_shipped_module` make
  that "exactly one in the crate" rather than "exactly one in the file I happened to look at".
- `candle-gen-catalog`'s `preview_advertising` module: `anima_base` / `anima_aesthetic` /
  `anima_turbo` added to `PREVIEW_ROUTE_IDS`, and the `candle-gen-anima` row given its exact route
  inventory (`pipeline.rs`: 1 hooked, 0 direct, 0 dark).

Mutation-checked in both directions rather than assumed:

| mutation | result |
| --- | --- |
| `Some(&preview)` → `None` at the sampler site | crate-local `the_render_lane_passes_a_preview_hook` **fails**; catalog `source_level_wiring_and_advertised_capability_agree_for_every_provider_crate` **fails** ("advertises … but nothing in its shipped sources emits"); catalog `every_wired_crate_pins_its_exact_route_inventory` **fails** |
| hook restored, `supports_preview` back to `false` | catalog `preview_capability_matches_every_wired_shipped_route_bidirectionally` **fails** ("wired preview provider anima_base must advertise support"); `source_level_wiring_…` **fails** with the opposite message |

## Real-weight run (CUDA)

Each variant is rendered twice on one warmed generator at the same seed — once with an inert sink,
once with a live one — and the strip is held to one shared analysis, so no variant can be closed with
a weaker measurement than another.

```sh
ANIMA_PREVIEW_DIR=…\split_files ANIMA_PREVIEW_ARTIFACT_DIR=E:\out\sc-16953 \
  cargo test --release --features cuda -p candle-gen-anima --test preview_real_weights \
    -- --ignored --nocapture
```

Retained run — `circlestone-labs/Anima` @ `53eec3898af698b2cf2a11379021fc9c5465d228`, 1024², seed
16953, prompt *"Anime illustration of a silver-haired traveler beneath cherry blossoms at sunset,
detailed, cinematic lighting."*, `RTX PRO 6000 Blackwell`, CUDA 12.9 / MSVC 14.44,
`CUDA_COMPUTE_CAP=120`. All five rows passed in 53.8 s.

For every variant:

- exactly N frames, numbered `1..=N`, each carrying `total == N`;
- every frame 128×128 (1024 / 8) RGB8 — VAE-latent resolution, which is also the proof the temporal
  squeeze ran;
- active-sink and inert-sink output RGB8 bytes **exactly equal** at the same seed;
- consecutive frames differ (mean |Δ| > 0.5), so the strip is not N copies of one image;
- distance to the finished render falls at **every** step and ends under 60 % of the first frame's;
- resemblance (Pearson correlation against a 16² thumbnail of the finished render) rises at every step
  and ends above 0.85.

| variant | steps | mean \|Δ\| to final, first → last | ratio | correlation, first → last |
| --- | --- | --- | --- | --- |
| `anima_base` (CFG 4.5) | 12 | 88.13 → 28.14 | 0.32× | +0.087 → +0.967 |
| `anima_aesthetic` (CFG 4.5) | 12 | 90.23 → 28.81 | 0.32× | +0.085 → +0.976 |
| `anima_turbo` (CFG-free) | 10 | 90.44 → 28.59 | 0.32× | +0.081 → +0.969 |

Per-frame correlation with the finished render:

| frame | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `anima_base` | +0.087 | +0.166 | +0.345 | +0.471 | +0.612 | +0.727 | +0.816 | +0.877 | +0.918 | +0.943 | +0.957 | +0.967 |
| `anima_aesthetic` | +0.085 | +0.166 | +0.312 | +0.426 | +0.532 | +0.651 | +0.759 | +0.843 | +0.904 | +0.943 | +0.966 | +0.976 |
| `anima_turbo` | +0.081 | +0.145 | +0.277 | +0.423 | +0.582 | +0.723 | +0.832 | +0.908 | +0.952 | +0.969 | — | — |

Retained strips, reviewed directly (frames left to right):

- [`sc-16953/base-1024-12step-strip.png`](sc-16953/base-1024-12step-strip.png)
- [`sc-16953/aesthetic-1024-12step-strip.png`](sc-16953/aesthetic-1024-12step-strip.png)
- [`sc-16953/turbo-1024-10step-strip.png`](sc-16953/turbo-1024-10step-strip.png)
- [`sc-16953/finals-contact-sheet.png`](sc-16953/finals-contact-sheet.png) — the three finished
  renders (base, aesthetic, turbo), each downscaled to 384², confirming three genuinely different
  images from three DiT checkpoints rather than one render measured three times.

Read directly, the base strip is a clean noise-to-image progression: frames 1–4 are uniform colour
noise with no structure; the figure's silhouette and the horizontal ground/blossom banding separate by
frames 6–7; the subject is unambiguous by frame 9; and frames 11–12 read as a lower-resolution,
noisier version of the finished render with the same pose, framing and palette — silver hair, dark
jacket, the warm orange sunset in the lower left, blossoms above. Turbo shows the same progression
compressed into ten steps. This is neither N views of noise nor N copies of one image.

### On the first-frame ceiling

sc-16950's `r_first < 0.35` is deliberately **not** ported. Correlation is taken over flattened RGB
triplets, so it carries channel-mean structure as well as spatial structure, and the fit's intercept
(0.406, 0.386, 0.287) is itself R > G > B — which every warm-lit render also is. Genuine pre-denoise
noise therefore starts at a non-zero, *scene-dependent* floor. This story uses sc-16952's shape: a
loose `r_first < 0.75` ceiling plus a required rise of `r_last − r_first > 0.30`. The rise is what
cannot be faked — a strip that opened on the finished image has nowhere to rise to — and it is layered
with the strictly monotone rise, the falling mean |Δ|, and the per-frame movement floor. All three
Anima variants happen to open near +0.08, comfortably under either bound; the looser ceiling is about
not failing an honest lane for the colour of its prompt, not about this run needing the room.

## Limits of this evidence

- Anima has exactly one render lane, and all three shipped ids were rendered on it, so there is no
  route in this crate whose runtime behaviour rests on a source scan alone.
- The multi-eval one-frame-per-step property is proven weights-free against `heun` and `dpmpp_sde`
  driving the real `run_flow_sampler` over the crate's real `anima_sigmas` schedule, with an assertion
  that those solvers really did evaluate more than once per step — otherwise the row would pass
  vacuously against an Euler fallback. The shipped default solver `er_sde` has its own row.
- LoRA/LoKr and the Q4/Q8 packed tiers ride the same render lane and the same hook; they were not
  rendered here. Nothing in the preview path reads adapter or quant state.
- A preview is decorative: the projection is a global linear map at 1/8 resolution, so fine detail is
  absent by construction. It is a progress indicator, not a substitute VAE decode. A projection
  failure loses one frame and can never fail a render.
