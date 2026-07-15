# candle-gen

Rust-native generative image (and, later, video) model inference on
[candle](https://github.com/huggingface/candle) — the **Windows/CUDA sibling** of
[`mlx-gen`](../mlx-gen/README.md) (Apple MLX). Both crates implement the **same** backend-neutral
[`gen_core`](../../contracts/gen-core) contract (SceneWorks epic 3720), so a consumer selects one
named runtime bundle and calls the identical `Generator` / registry API regardless of which tensor
backend is underneath.

> **Status: a maturing multi-family engine on the Candle/CUDA lane.** candle-gen now hosts **20+
> cataloged model families** across image and video, plus a captioner, text/image embedders, and
> host-side conditioning utilities — every provider publishes an ordinary registration value that
> `candle-gen-catalog` composes explicitly, keeps the deterministic CPU-seeded-noise contract
> (launch-portable per seed), and rides the `CandleError → gen_core::Error` bridge. The families below
> are GPU-validated on an RTX PRO 6000 (Blackwell **sm_120**). The core `candle-gen` crate supplies the
> shared device/dtype seam, weight loaders, seeded noise, the unified sampler/scheduler framework (epic
> 7114), the LoRA/LoKr training + inference-merge harness, and the Q4/Q8 + MLX-packed quantization seam.
>
> **Image generators**
>
> | family | registered engine id(s) | architecture / notes |
> |--------|-------------------------|----------------------|
> | SDXL | `sdxl` | dual-CLIP UNet (`sdxl` + RealVisXL); img2img, LoRA train+merge, quant |
> | Z-Image | `z_image`, `z_image_turbo` | Qwen3 TE → DiT flow-match; base CFG + distilled Turbo; LoRA trainer |
> | FLUX.1 | `flux1_schnell`, `flux1_dev` | CLIP-L + T5-XXL → FLUX DiT flow-match |
> | FLUX.2 | `flux2_klein_9b`, `flux2_dev`, `flux2_dev_edit`, `flux2_dev_control` | from-scratch MMDiT; txt2img + Edit + Fun-Controlnet-Union |
> | Qwen-Image | `qwen_image` | 60-layer dual-stream MMDiT; txt2img + Edit + Fun-Controlnet-Union (VACE) + Lightning + LoRA/LoKr + MLX-packed quant |
> | Anima | `anima_base`, `anima_aesthetic`, `anima_turbo` | Cosmos-Predict2 DiT anime T2I (ER-SDE-3 sampler) |
> | Chroma | `chroma1_hd`, `chroma1_base`, `chroma1_flash` | FLUX-schnell-derived MMDiT |
> | Kolors | `kolors` | SDXL UNet + ChatGLM3-6B encoder |
> | SenseNova-U1 | `sensenova_u1_8b` | NEO-Unify dual-path Qwen3 MoT + flow head; + distilled fast tier |
> | Ideogram 4 | `ideogram_4` | + Turbo + edit |
> | Boogu-Image | `boogu_image`, `boogu_image_turbo`, `boogu_image_edit` | Lumina2/OmniGen2 MMDiT; Base/Turbo/Edit |
> | Krea 2 | `krea_2_turbo`, `krea_2_raw`, `krea_2_edit`, `krea_2_control`, `krea_2_turbo_edit` | 12B DiT; Turbo/Raw/Edit/Control + CFG-free Turbo-edit; LoRA/LoKr trainer |
> | SD3.5 | `sd3_5_large`, `sd3_5_large_turbo`, `sd3_5_medium` | MMDiT + triple text-encoder aggregator + 16-ch VAE |
> | Lens | `lens`, `lens_turbo` | gpt-oss-20B MoE encoder + dual-stream MMDiT + FLUX.2 VAE |
> | Bernini | `bernini`, `bernini_renderer` | ByteDance Wan2.2-T2V-A14B dual-expert renderer (planner + renderer) |
>
> Identity-preserving stacks compose the above rather than registering standalone engines:
> **InstantID** (`candle-gen-instantid`) layers the IP-Adapter/ControlNet + kps/openpose control render +
> `gen_core::FaceEmbedder` onto candle SDXL, and **PuLID-FLUX** (`candle-gen-pulid`) injects an
> EVA02-CLIP + IDFormer perceiver into FLUX.1-dev.
>
> **Video generators**
>
> | family | registered engine id(s) | architecture / notes |
> |--------|-------------------------|----------------------|
> | Wan2.2 | `wan2_2_ti2v_5b`, `wan2_2_t2v_14b`, `wan2_2_i2v_14b`, `wan_vace` | UMT5-XXL + WanTransformer3D (from-scratch conv3d); TI2V/I2V, dual-expert 14B, VACE, LoRA, quant, tiling |
> | LTX-2.3 | `ltx_2_3_distilled` | distilled 22B; Gemma-3-12B TE + connector + AVTransformer3D + CausalVideoAutoencoder |
> | SVD | `svd_xt` | Stable Video Diffusion img2vid-xt |
> | SCAIL-2 | `scail2_14b` | Wan2.1-14B I2V character animation / cross-identity replacement |
> | SeedVR2 | `seedvr2` | one-step DiT + 3D causal video VAE super-resolution upscaler (image **and** video) |
>
> **Captioner, embedders & conditioning utilities**
>
> - **JoyCaption** (`candle-gen-joycaption`) — the first `Captioner`: a from-scratch LLaVA
>   (SigLIP-so400m tower + gelu-MLP projector + Llama-3.1-8B decoder) turning an image into a caption.
> - **CLIP ViT-L/14** (`candle-gen-clip`) — `clip_vit_l14` image embedder + `clip_vit_l14_text` text
>   embedder providers (Dataset Doctor).
> - **Face** (`candle-gen-face`) — SCRFD detector + ArcFace embedder implementing `gen_core::FaceEmbedder`
>   for the InstantID / PuLID ports.
> - **SAM3** (`candle-gen-sam3`) — a Segment-Anything-3 concept segmenter (off-Mac person-track PCS), a
>   plain utility rather than a registered generator.
> - **Depth Anything V2** (`candle-gen-depth`) — monocular depth estimator used as a host-side ControlNet
>   depth preprocessor.
> - **PiD** (`candle-gen-pid`) — NVIDIA Pixel-Diffusion latent→pixel super-resolving decoder seam.
>
> **candle pinned to git main (post-0.10.2)** — REQUIRED for Blackwell sm_120. The crates.io 0.10.2
> release throws `CUDA_ERROR_INVALID_PTX` at the first candle-kernels kernel whenever
> candle-transformers is linked (plain candle-core works). The git rev clears it and is
> source-compatible. See `[workspace.dependencies]`.

## Layout

```
candle-gen/                 # workspace root
  candle-gen/               # core crate: gen_core + candle re-exports; device/dtype seam, weight
                            #   loaders, seeded noise, unified sampler/scheduler, LoRA/LoKr train +
                            #   inference-merge harness, Q4/Q8 + MLX-packed quant; CandleError bridge
  # --- image generators ---
  candle-gen-sdxl/          # SDXL / RealVisXL (dual-CLIP UNet) — txt2img, img2img, LoRA, quant
  candle-gen-z-image/       # Z-Image / Z-Image-Turbo (Qwen3 TE → DiT flow-match) + LoRA trainer
  candle-gen-flux/          # FLUX.1 [schnell]+[dev] (CLIP-L + T5-XXL → FLUX DiT)
  candle-gen-flux2/         # FLUX.2 klein-9B + dev + dev-edit + dev-control (from-scratch MMDiT + Qwen3)
  candle-gen-qwen-image/    # Qwen-Image: txt2img + Edit + Fun-Controlnet-Union + Lightning + LoRA + packed quant
  candle-gen-anima/         # Anima (Cosmos-Predict2 DiT) anime T2I — base/aesthetic/turbo
  candle-gen-chroma/        # Chroma (chroma1_hd/base/flash) FLUX-schnell-derived MMDiT
  candle-gen-kolors/        # Kolors (SDXL UNet + ChatGLM3-6B encoder)
  candle-gen-sensenova/     # SenseNova-U1 (NEO-Unify Qwen3 MoT + flow head) + distilled fast tier
  candle-gen-ideogram/      # Ideogram 4 (+ Turbo, edit)
  candle-gen-boogu/         # Boogu-Image-0.1 (Lumina2/OmniGen2 MMDiT) — base/turbo/edit
  candle-gen-krea/          # Krea 2 — turbo/raw/edit/control/turbo-edit + LoRA/LoKr trainer
  candle-gen-sd3/           # Stable Diffusion 3.5 Large/Large-Turbo/Medium (MMDiT + triple TE + 16-ch VAE)
  candle-gen-lens/          # Lens / Lens-Turbo (gpt-oss-20B MoE encoder + dual-stream MMDiT + FLUX.2 VAE)
  candle-gen-bernini/       # Bernini (ByteDance Wan2.2-T2V-A14B dual-expert renderer)
  candle-gen-instantid/     # InstantID identity-preserving SDXL (IP-Adapter/ControlNet + FaceEmbedder)
  candle-gen-pulid/         # PuLID-FLUX identity injection (EVA02-CLIP + IDFormer into FLUX.1-dev)
  # --- video generators ---
  candle-gen-wan/           # Wan2.2 TI2V-5B + T2V/I2V-14B + VACE (from-scratch conv3d); LoRA, quant, tiling
  candle-gen-ltx/           # LTX-2.3 (distilled 22B) txt2video (Gemma-3-12B TE + connector + AVTransformer3D)
  candle-gen-svd/           # Stable Video Diffusion (img2vid-xt)
  candle-gen-scail2/        # SCAIL-2 (Wan2.1-14B I2V character animation / identity replacement)
  candle-gen-seedvr2/       # SeedVR2 one-step DiT super-resolution upscaler (image + video)
  # --- captioner / embedders / conditioning utilities ---
  candle-gen-joycaption/    # JoyCaption (LLaVA: SigLIP + Llama-3.1-8B) image→caption Captioner
  candle-gen-clip/          # CLIP ViT-L/14 image + text embedder providers
  candle-gen-face/          # SCRFD detector + ArcFace embedder (gen_core FaceEmbedder)
  candle-gen-sam3/          # SAM3 concept segmenter (person-track PCS utility)
  candle-gen-depth/         # Depth Anything V2 monocular depth (ControlNet preprocessor)
  candle-gen-pid/           # NVIDIA PiD pixel-diffusion latent→pixel decoder seam
  vendor/
    candle-kernels/         # local fork: multi-arch fatbin for the CUDA quant path (sc-7544; see Packaging)
  scripts/
    check-gen-core-skew.sh  # version-skew gate: fails if >1 sceneworks-gen-core resolves
    check-cuda.ps1          # local cuda gate: vcvars + cargo build/test --features cuda (run pre-push)
    package-cuda.ps1        # bundle a CUDA build + redist DLLs into dist/ (sc-3676; see Packaging)
  .github/workflows/ci.yml  # macOS/Linux fmt+clippy+check+test + skew self-test; manual Windows/CUDA lane
```

A provider crate publishes one or more named registration constants and a `register_providers`
builder function. `candle-gen-catalog` intentionally enumerates every family in the supported
Candle surface, making platform inclusion visible in review and exact-surface tests. Stable engine
ids are unchanged (e.g. `candle-gen-sdxl` exposes `"sdxl"`, which the SceneWorks worker maps both
`sdxl` and `realvisxl` onto), and every descriptor reports `backend: "candle"`.

## Backends / features

The default build is **CPU** (`candle-core`'s default) and works on macOS with no extra features.

| feature      | backend                | platform        | in `default`? |
|--------------|------------------------|-----------------|---------------|
| *(none)*     | CPU                    | all (Mac dev)   | yes           |
| `metal`      | Apple Metal GPU        | macOS           | no            |
| `cuda`       | NVIDIA CUDA            | Windows/Linux   | no            |

`cuda` needs the CUDA toolkit and **does not build on Mac**; all CUDA-only code is gated behind
`#[cfg(feature = "cuda")]`. A `flash-attn` feature used to exist as a no-op alias (`= ["cuda"]`,
forwarded crate-to-crate) that wired no fused kernels; it was removed in sc-9032. When the
`candle-flash-attn` slice is scheduled, reintroduce it behind real gated code, not a bare alias.

## The NVFP4 lane (Blackwell `sm_120` only) — epic 11037

**NVFP4** = 4-bit float **E2M1** elements + an **FP8 (E4M3)** micro-scale per **16-element block** + a
second-level **FP32 per-tensor** scale ⇒ **~4.5 effective bits/weight**. It is a *compute* lane, not
just a storage format: the cuBLASLt FP4 GEMM (`CUDA_R_4F_E2M1` operands + `VEC16_UE4M3` block scales)
runs on the Blackwell FP4 tensor cores. Everything lives in `candle-gen/src/quant/`
(`nvfp4.rs` packer/container, `cublaslt.rs` GEMM + on-device activation quantizer, `nvfp4_linear.rs`
the `Nvfp4Linear` layer + policy, `nvfp4_outlier.rs` the sparsity metric).

### Hardware scope: `sm_120` only — `sm_100` is explicitly out of scope

| target | status |
|---|---|
| **consumer Blackwell `sm_120`** (RTX PRO 6000 / RTX 50-series) | **the only supported NVFP4 target.** Validated on the 2× RTX PRO 6000 rig (CUDA 12.9, MSVC 14.44). Plain `sm_120` — `sm_120a` is *not* required. |
| datacenter Blackwell `sm_100` (B100/B200) | **explicitly out of scope for this epic.** Not built, not gated on, not validated. A separate effort. |
| pre-Blackwell CUDA, CPU, Metal | no FP4 hardware → transparent fallback (below). |

**The gate is capability-probed, not assumed** (`CublasLt::meets_nvfp4_floor`). Below the floor — or on
CPU, a non-`cuda` build, or an FP4-ineligible shape (padded-K not a multiple of `NVFP4_K_ALIGN`, or
N % 16 ≠ 0) — `Nvfp4Linear` **transparently serves a dequant→bf16 dense matmul instead and never
panics**. An NVFP4 model therefore loads and runs everywhere; it just doesn't light the FP4 cores off
Blackwell. That is SC#4, and it is asserted both at layer level
(`quant::nvfp4_linear::tests::cpu_selects_dequant_fallback_and_forwards`) and at model level
(`candle-gen-sana`'s `transformer::tests::nvfp4_plan_falls_back_cleanly_off_blackwell`).

### W4A4 vs W4A16 — two regimes, and which one you actually want today

The FP4 tensor-core MMA needs **both** operands in E2M1. So:

* **W4A4** (FP4 weight × FP4 activation) — the only regime that lights the FP4 cores. The FP4 GEMM
  core measures **1.35–1.98×** over bf16 on real Sana shapes (sc-11044; SC#1's GEMM-level target).
* **W4A16** (FP4 weight × bf16 activation) — the weight is dequantized to bf16 **once at load and then
  held resident**, and run dense. **No FP4 compute win — and no VRAM win either**: a W4A16 layer costs
  the full dense bf16 footprint (see [Native packed serving](#native-packed-serving-sc6)). What it buys
  is *numerical stability* on the outlier class. It is the outlier-class override and the capability
  fallback.

**Current perf reality (read before quoting a multiple).** The W4A4 activation quantizer is **unfused**
— a chain of candle ops rather than one kernel — and that overhead currently **swamps the FP4 GEMM
win** end-to-end (sc-11044 measured ~25.8 ms/fwd for W4A4 vs ~0.05 ms for W4A16 at layer level). So:

> **W4A16 is the throughput default; W4A4 ships opt-in** (correctness + the ~4.5-bit footprint are
> real today). Making W4A4 a net end-to-end win is gated on **[sc-12078](https://app.shortcut.com/trefry/story/12078)**
> — a fused CUDA activation-quantize kernel. Until that lands, the ~2× is a **GEMM-core** number, not
> an end-to-end one.

### SC#1 — what Sana can and cannot tell you (read before quoting an end-to-end ratio)

sc-11045 benched the real SANA-1.6B trunk end-to-end. **Do not quote its vs-dense ratios as SC#1's
number of record.** They are real measurements, but ~93% of their denominator is an unrelated
**candle-core defect**, so they say almost nothing about NVFP4:

| per denoise step, SANA-1.6B @ 1024px, sm_120 | time | share |
|---|---:|---:|
| `conv_depth` — the Mix-FFN 3×3 **depthwise** conv (`groups = 2·hidden = 11200`), ×20 blocks | **19.65 s** | **93.4%** |
| all other convs | ~1.32 s | 6.3% |
| **all 163 NVFP4-eligible linears** | **~0.08 s** | **0.4%** |
| total | 21.05 s | |

The cause is `candle-core/src/conv.rs:331-338`: a grouped conv is decomposed into **one kernel launch
per group**, then a `cat` of **11200** tensors — 982 ms/call, GPU utilization 0–1% (host-launch-bound,
not compute-bound). Per block: linears **1.99 ms (0.20%)** vs convs **984.8 ms (99.80%)**, of which the
depthwise alone is **982.5 ms**. **The NVFP4 lane touches 0.4% of the step, so SC#1's end-to-end ceiling
on Sana as it runs today is ~1.002× even if the FP4 lane were infinitely fast.** A vs-dense ratio
measured against that denominator is an artifact of the conv defect, not a property of NVFP4.

**The one signal that does survive** the conv noise, because it is a *marginal* cost measured against
an otherwise-identical leg: **W4A4 adds ~9.1 s/step of unfused activation-quantizer overhead**
(≈28 ms/forward/CFG-branch — consistent with sc-11044's ~25.8 ms/fwd layer-level number). That is real
sc-12078 evidence, and it is the honest sc-11045 throughput finding.

> **Sana cannot settle SC#1 or SC#2**, on three independent counts: the conv-dominated FFN above; **no
> bf16 path** (the trunk is f32-only, so the epic's "vs the bf16 compute path" baseline does not exist
> here); and **no Q4 tier** wired for Sana (so SC#2's honest baseline — the int4 tier NVFP4 would
> *replace* — cannot be measured either). SC#1/SC#2 are retargeted to a **Flux-family DiT**
> (linear FFN + `QLinear` Q4 already wired) under
> **[sc-12110](https://app.shortcut.com/trefry/story/12110)**, which is the vehicle of record for both.
> What sc-11045 *does* settle is SC#3 (stability), SC#4 (the Blackwell gate), SC#6 (packed serving, see
> below), and the spike's residual activation-outlier partition gate.

### The mixed-precision policy (why not blanket W4A4)

NVFP4 W4A4 reproduces the sc-7702 collapse: an activation **outlier** sharing a 16-block crushes its
co-located channels to E2M1 zero. Damage scales with outlier **sparsity** (spike sc-11038: benign 0.991
cosine → ~2 sparse outliers 0.984 → 8 → 0.966 → dense → 0.000). So the default is **not** blanket W4A4:

* **benign compute-bulk** — **self-attention** (`attn1`) + FF projections → **W4A4**;
* **outlier class** → **W4A16** (bf16 activation): the text→DiT `caption_projection`, **the whole
  cross-attention block** (`attn2` / `cross_att*` — Q and `to_out` as well as K/V), the trunk's
  **final** `proj_out` head, and the **first two & last** DiT blocks.

`ActPrecision::for_outlier_layer` carries this as a substring policy over dotted layer keys;
`ActPrecision::partition_layers` is its explicit, testable form.

**Two things the substring form gets wrong if you are careless, both fixed in sc-11045's review:**

* **Spelling.** `contains("cross_attn")` does **not** match `cross_attention` (position 10 is `e`, not
  `n`). The policy matches `cross_att`/`crossatt` — covering `cross_attn`, `cross_attention`,
  `crossattn`, `crossattention` — *and* retains the pre-sc-11045 `cross` + K/V clause verbatim, so the
  widening is a provable superset rather than an accidental narrowing.
* **Scope.** `contains("proj_out")` over-fires. The 438× measurement is SANA's single **top-level**
  head, but the bare substring also matches per-block layers that are the spike's explicitly *benign*
  W4A4 class — `candle-gen-ltx` remaps `ff.net.2` → `ff.proj_out` (the FF output of all 48 blocks), and
  `candle-gen-flux`/`candle-gen-chroma` name `single_transformer_blocks.{i}.proj_out` (the fused
  attn+MLP output `[5·hidden → hidden]`, the largest GEMM in each of 38 single blocks). Firing there
  would pull the compute bulk out of the FP4 lane. The final head is therefore **stated by the
  provider** (`LayerRole::final_proj()`, the same seam as `is_edge_block`), with a conservative
  name anchor as the fallback — never a bare substring.

Both defects are **latent, not live**: `candle-gen-sana` is the policy's only consumer today, and
neither defect changes its partition (Sana spells cross-attention `attn2`, and its only `proj_out` *is*
the trunk head — it measures 68 W4A4 / 95 W4A16 either way). They are traps armed for whichever crate
wires up NVFP4 next — which, per [sc-12110](https://app.shortcut.com/trefry/story/12110), is a
Flux-family DiT, i.e. exactly the naming that trips them. That is why they are fixed here, before the
retarget, rather than after.

> **The outlier class was WIDENED in sc-11045 by real measurement — do not narrow it back to the
> spike's prose.** The spike named the class "cross-attn K/V", and the original policy took that
> literally, leaving cross-attn **Q** and **`to_out`** on W4A4. Capturing per-layer activation-outlier
> sparsity across a **real Sana-1.6B denoise** (`candle-gen-sana`'s `ActProbe`) refuted that reading: of
> the 109 projections the old policy sent to W4A4, **27 measured `OutlierClass::Dense`** — 17 ×
> `attn2.to_out.0`, 6 × `attn2.to_q` (per-block crush ratios up to **5124×**), `proj_out` (438×), and
> block **1**'s self-attention (176×). The entire cross-attention block reads caption-derived context,
> so it carries the caption's massive activations regardless of which projection you name. Guarding
> `attn2` wholesale restores the spike's *actual* intent (W4A4 == self-attn + FF). This is the empirical
> gate synthetic activations could not close.
>
> **The widening is strictly the safe direction — relative to the pre-sc-11045 rule, and proven, not
> asserted.** Every clause that rule guarded is still guarded verbatim, so the policy can only ever move
> a layer **W4A4 → W4A16** (and W4A16 is already the throughput default). That property is a test:
> `differential_widening_is_strictly_safe` re-implements the old rule and asserts the superset over a
> cross-provider name corpus. It is what caught the `cross_attn`/`cross_attention` narrowing described
> above — which, before the fix, made this very claim false in the collapse-prone direction.
>
> **The widening is not free — it is paid in VRAM.** On SANA it moved the mixed policy from **109 W4A4 /
> 54 W4A16** to **68 W4A4 / 95 W4A16**: 41 more projections now hold a dense bf16 weight, which is most
> of the gap between the 0.28× packed ratio and the mixed policy's measured **0.70×** (SC#6 table
> above). Re-measured on a live denoise, the partition now holds — **68 W4A4: 47 Benign, 21 Sparse, 0
> Dense** — and 35 of the 95 W4A16 layers do measure Dense, i.e. the override is earning its keep.

### Native packed serving (SC#6)

The point of the format is the footprint, so the serving path **must not** full-dequant to bf16 in
VRAM. `Nvfp4Linear` **in the W4A4 regime** stages the packed weight to the device **once** and never
allocates a dense bf16 copy; resident device bytes == the packed NVFP4 footprint (**0.281×** the bf16
size, ~4.5 eff bits/wt), proven by contention-immune tensor byte-accounting at layer level (sc-11041)
and at whole-model level (sc-11045, `SanaTransformer::nvfp4_report`).

**Scope that claim to W4A4, because resident VRAM is regime-dependent.** W4A16 is realized by
dequantizing the weight to bf16 at construction and holding it resident, so *only* the packed W4A4 path
delivers the footprint. Measured on the real SANA-1.6B trunk on `sm_120` (163 projections, **1550.8 MiB**
dense bf16 — `nvfp4_sana_dit_real_model_vram_footprint`):

| regime | FP4-lit | resident FP4 | resident bf16 | resident total | ratio |
|---|---:|---:|---:|---:|---:|
| **blanket W4A4** — bench only, *not* shipping | 163/163 | 437.6 MiB | 0.0 MiB | **437.6 MiB** | **0.28×** |
| **mixed** — the shipping policy | 68/163 | 183.6 MiB | 900.0 MiB | **1083.6 MiB** | **0.70×** |
| **blanket W4A16** — the throughput default above | 0/163 | 0.0 MiB | 1550.8 MiB | **1550.8 MiB** | **1.00×** |

Read that honestly:

* **SC#6 is satisfied — and its claim is scoped to the packed W4A4 serving path**, which is a *bench*
  regime, not what ships. That is a real proof of the packed-forward path (0.2822×, 4.51 effective
  bits/weight), not a claim about the shipping product's VRAM.
* **The shipping mixed policy costs 1083.6 MiB — a ~30% saving vs dense bf16, not ~72%.** 95 of its 163
  projections (58%) are outlier-class and hold a full dense bf16 weight.
* **Blanket W4A16 — the regime the throughput guidance above tells you to use — saves nothing at all in
  VRAM** (1.00×). There, NVFP4 buys stability and on-disk/load-time storage only.

`Nvfp4Report::footprint_ratio()` is regime-aware and reports exactly the table above;
`packed_footprint_ratio()` is the format's ~0.28× and is **not** a residency claim. (Before sc-11045's
review these were the same number: the ratio divided the *host* packed container by bf16, so a W4A16 leg
reporting 163/163 dequant→bf16 still printed "0.2822".)

### The tier is a distinct `Quant::Nvfp4` — a deliberate creative choice

NVFP4 is surfaced as its **own** `gen_core::Quant::Nvfp4` tier (sc-11042), *not* as a silent Blackwell
backend for the existing `Quant::Q4`. Its numerics differ from the int4 tier, and per the epic's SC#5 a
quant tier is a **creative** decision: NVFP4 must never silently replace an existing tier's numerics
without an explicit, user-visible choice. Selecting it is an act of intent, and it is only honored on
`sm_120`.

### Driving a real model: the Sana NVFP4 seam

`candle-gen-sana` carries the reference wiring (`nvfp4_dit.rs`): `SanaTransformer::from_weights_planned`
serves the trunk's q/k/v/out, `caption_projection` and `proj_out` projections through `Nvfp4Linear`
under a `DitPlan` (mixed policy, or blanket W4A4/W4A16 for a controlled bench), with an optional
`ActProbe` capturing per-layer activation sparsity across denoise steps.

**Note what is not covered.** SANA's Mix-FFN is a `GLUMBConv` built from *convolutions*, not linears, so
a real slice of the trunk's FLOPs sits outside the **current seam** — an honest ceiling on any
end-to-end multiple. "Outside the seam" is not "outside the lane by construction", though: **2 of the 3
GLUMBConv convs are 1×1 — i.e. GEMMs in disguise** (`conv_inverted` and `conv_point`, measured 2.37
ms/block vs the linears' 1.99 ms/block) and could be routed through `Nvfp4Linear` with a reshape,
roughly **doubling the lane's reachable coverage**. Only `conv_depth` (3×3 **depthwise**) genuinely
cannot be a GEMM. Extending the seam to the 1×1s is not wired today.

End-to-end validation lives in `candle-gen-sana/tests/nvfp4_sana_dit_gpu.rs` (cuda-gated, `#[ignore]`d,
real-weight).

## Packaging (Windows / CUDA) — sc-3676

The goal is **one distributable CUDA worker that runs on every NVIDIA GPU we support, not just the
build box's Blackwell** — the "central fat binary, like torch" model.

### How portability works: PTX-JIT for dense kernels, a multi-arch fatbin for quantized kernels

candle-kernels has **two** compile paths, and they need different portability treatments (verified
against the vendored sources):

- **Dense kernels** build via cudaforge `.build_ptx()` → `nvcc --ptx`, emitting one **`compute_80`
  PTX** (virtual ISA) per kernel. The driver JIT-compiles that PTX to the runtime GPU's native SASS
  at first load, so it runs on **every NVIDIA arch ≥ sm_80** — Ampere (sm_80/86) → Ada (sm_89) →
  Hopper (sm_90) → Blackwell (sm_120) — from a single embedded PTX. (Tradeoff: it does not use
  sm_90a/sm_120a arch-accelerated tensor features, and the first run is slower while the driver JITs;
  the result caches per-GPU under `%APPDATA%\NVIDIA\ComputeCache`.)
- **Quantized + MoE kernels** (`mmq_gguf/*`, `moe/*`, `mmvq_gguf` — the GGUF `QMatMul`) build via
  cudaforge `.build_lib()` → `nvcc -c`: a **static `libmoe.a` of SASS, _not_ PTX**. cudaforge emits
  one `-gencode` from `CUDA_COMPUTE_CAP` (`GpuArch::parse` runs `parse::<usize>()` on the whole
  string, so a `;`-list does **not** parse — there is no multi-cap support). At the `=80` baseline the
  archive held only an **sm_80 cubin**; SASS is not forward-compatible across major arches and there
  is no PTX to JIT, so on **Blackwell sm_120 every quant matmul silently returned zeros** — dense
  models rendered but quantized models came out black (**sc-7544**; the dense PTX path masked it).

**The fix (sc-7544): a multi-arch fatbin for the quant path.** cudaforge can't emit a cap list and the
candle pin is upstream (not a fork), so candle-kernels is **locally forked** in `vendor/candle-kernels`
(identical to the pinned rev except three `-gencode` lines in `build.rs`) and patched in via the
workspace `[patch]`. nvcc accumulates `-gencode` flags, so `libmoe.a` becomes a real fatbin embedding
**sm_80 + sm_90 + sm_120 SASS + `compute_120` PTX** — one binary that runs natively Ampere → Ada →
Hopper → Blackwell and JITs forward to newer archs. Keep `CUDA_COMPUTE_CAP=80` in the recipes (it
seeds the sm_80 baseline for both paths). Verified on RTX PRO 6000 (sm_120): `cuobjdump --list-elf`
shows sm_80/sm_90/sm_120 cubin per kernel, and `candle-gen`'s `cuda_quant_smoke` test has the Q4/Q8
`QMatMul` matching the CPU reference (cos ≈ 1.0, vs cos ≈ 0 / all-zeros before). That smoke runs in
the CUDA gate so the regression can't return silently. **Re-vendor on every candle pin bump** — see
`vendor/candle-kernels/VENDORED.md`.

### Build

Build-time needs the **CUDA 12.9 toolkit (nvcc)** + **VS 2022 v143 (MSVC 14.4x) Build Tools**; the
build is driven through `vcvars64.bat`. From a `cmd` shell that has sourced vcvars:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_COMPUTE_CAP=80
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
cargo build --release -p candle-gen-sdxl --example sdxl-txt2img --features cuda
```

The scripted, reproducible form of this — sources vcvars, sets the env, runs `cargo build/test
--workspace --features cuda` — is `scripts/check-cuda.ps1`. **Run it before pushing CUDA-touching
changes**: the CPU/Metal CI lanes are blind to `#[cfg(feature = "cuda")]` code, so this is the real
cuda gate.

```powershell
pwsh scripts/check-cuda.ps1            # build + test
pwsh scripts/check-cuda.ps1 -SkipTests # build-only smoke check
```

The `windows-cuda` lane in `.github/workflows/ci.yml` runs the same recipe but is **manual-only**
(`workflow_dispatch`) — it needs no standing runner. To run it in CI you must first register a
self-hosted `[self-hosted, windows, cuda]` runner, then dispatch the workflow by hand. (GitHub's
hosted GPU larger-runners are Tesla T4 / sm_75, below our sm_80 baseline, so they can't run it.)

### Bundle the runtime DLLs

The target machine needs the CUDA **runtime** libraries but should **not** require a CUDA Toolkit
install. `scripts/package-cuda.ps1` copies the binary plus the redistributable DLLs (which cudarc
dynamic-links, resolved from the exe's own directory) into `dist/`:

```powershell
pwsh scripts/package-cuda.ps1 -BinaryPath target\release\examples\sdxl-txt2img.exe
```

Bundled redist DLLs (CUDA 12.9; the script globs the version suffixes):

| DLL                          | provides            |
|------------------------------|---------------------|
| `cudart64_12.dll`            | CUDA runtime        |
| `cublas64_12.dll`            | cuBLAS              |
| `cublasLt64_12.dll`          | cuBLAS-Lt           |
| `curand64_10.dll`            | cuRAND              |
| `nvrtc64_120_0.dll`          | NVRTC               |
| `nvrtc-builtins64_129.dll`   | NVRTC builtins      |

The script also writes a `RUNTIME.txt` manifest into the bundle. Verified: with the bundle's DLLs
present and the **CUDA toolkit removed from `PATH`**, the binary runs end-to-end (DLLs resolve from
the exe's directory).

### Minimum driver

The **NVIDIA driver is not bundled** (it is not redistributable) and is what JIT-compiles the PTX +
provides `libcuda`. For the bundled **CUDA 12.9** runtime the floor is:

- **Windows: driver ≥ 576.02** (CUDA 12.9 GA).
- GPU compute capability **≥ 8.0** (Ampere / RTX 30-series or newer).

Older drivers should be updated from nvidia.com; the CUDA runtime DLLs in the bundle do **not** lift
the driver requirement.

## gen-core pinning (read before bumping)

`sceneworks-gen-core` is a path dependency in the canonical workspace and must resolve exactly once
in every product graph. Two releases silently create two contract type identities and two copies of
host policy; explicit registries prevent hidden discovery, but duplicated contracts remain an
unsupported and drift-prone graph. Run the gate:

```bash
bash scripts/check-gen-core-skew.sh            # checks candle-gen's build graph
bash scripts/check-gen-core-skew.sh --self-test  # proves the gate fires on canned skew
```

When bumping the gen-core pin, bump it in lockstep with the worker's `mlx-gen` + `sceneworks-gen-core`
pins.

## Develop

```bash
cargo fmt --all
cargo check --workspace                 # CPU (Mac default)
cargo check --workspace --features metal  # Metal backend builds
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                  # registry-resolution + bridge tests
```

The candle version this workspace settled on is recorded in `[workspace.dependencies]`
(`candle-core` / `candle-nn`).
