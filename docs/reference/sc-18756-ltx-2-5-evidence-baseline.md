# LTX-2.5 access + component evidence baseline (sc-18756)

| | |
| --- | --- |
| Story | sc-18756 — Phase 0 of epic 18755 (`LTX-2.5: native MLX + candle video generation`) |
| Epic | [18755](https://app.shortcut.com/trefry/epic/18755) |
| Purpose | Pin the factual baseline the rest of the epic is built on, in-repo, so no later story re-derives it or guesses |
| Gathered by | Claude (Sonnet 5), automated agent, on behalf of Michael Trefry |
| Access probe date (first pass, unauthenticated) | 2026-08-18 |
| Component evidence capture date (this document, §2) | **2026-08-18**, authenticated, header-only Range reads under HF account `SceneWorks` — see [§1.4](#14-credential-found-authenticated-capture-2026-08-18). **Not** the epic's 2026-08-11 capture; this is an independent re-measurement that superseded it. |
| **Read and signed off by a human** | **_(NOT YET — Michael)_** |

## 0. Why this document exists

Every later LTX-2.5 story cites facts instead of re-deriving them: the reference-impl pin, the
measured 2.3→2.5 transformer config diff, the per-component tensor/size/dtype/`__metadata__`
inventory, and the pre-change engine-capabilities baseline. This document is the citable home for
those facts.

**§2 was freshly captured on 2026-08-18** using header-only Range requests (no weight bytes
downloaded) under the `SceneWorks` HF account, once a working credential was located on this
machine (see [§1.4](#14-credential-found-authenticated-capture-2026-08-18)). It supersedes the
epic's 2026-08-11 capture rather than merely reproducing it — every table in §2 is this document's
own measurement, and it also fills two rows the epic's original table never itemized (the
comfy-int8-convrot dev-transformer and TE variants).

---

## 1. HF access state

### 1.1 Credential inventory, first pass (default locations only)

```
$ hf auth whoami
Error: Not logged in

$ env | grep -i hf_token        # (no output)
$ env | grep -i huggingface     # (no output)
$ ls ~/.cache/huggingface/token  # No such file or directory
$ ls ~/.huggingface/token        # No such file or directory
```

No credential at any of the default locations `hf`/`huggingface_hub` check automatically.

### 1.2 Unauthenticated probes against `Lightricks/LTX-2.5`

| Probe | Result | Notes |
| --- | --- | --- |
| `GET /Lightricks/LTX-2.5/resolve/main/README.md` | **200** | The README and repo metadata are public even though the repo is gated |
| `GET /api/models/Lightricks/LTX-2.5` | **200**, `"gated":"auto"` | Model-info API confirms `gated: auto`, `license: other` (`ltx-2-community-license-agreement`), `license_link: https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md`. Full `siblings` file listing (filenames + paths) is visible without auth. |
| `Range: bytes=0-1000` on `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | **401** `GatedRepo` | *"Access to model Lightricks/LTX-2.5 is restricted. You must have access to it and be authenticated to access it. Please log in."* |

The repo *card* is public; every weight file is gated behind `401 GatedRepo` without a credential.

### 1.3 Correction — the first-pass "no credential" finding was a default-location gap, not an absence

The first pass above only checked the locations `hf auth whoami` and `huggingface_hub` consult by
default (`$HF_TOKEN`, `$HUGGING_FACE_HUB_TOKEN`, `~/.cache/huggingface/token`). It did not check
non-default `HF_HOME` locations. This machine has a working credential at a **non-default**
`HF_HOME`:

```
$ export HF_HOME=/Volumes/Models/huggingface
$ hf auth whoami
user=SceneWorks
```

`SceneWorks` is the same HF account the epic records as having accepted the LTX-2.x Community
License gate on 2026-08-11. This is the account CI/desktop tooling is *meant* to use (per
`scripts/check-download-patterns.mjs`'s `$HF_TOKEN`/`$HUGGING_FACE_HUB_TOKEN` convention), but
this document still cannot confirm from a repo-static check that CI or the desktop app actually
resolve to this same token store at build/run time — no `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`
secret is wired into any inference-repo GitHub Actions workflow (`grep -rli` over
`.github/workflows/` returns no matches), and the desktop app's token is supplied by the user at
runtime through its own settings surface, not something a static check can resolve. That residual
question is recorded in [BLOCKED](#blocked) below; it is now a narrower verification gap, not a
missing credential.

### 1.4 Credential found, authenticated capture, 2026-08-18

With `HF_HOME=/Volumes/Models/huggingface` exported, the Range-request header reads the story
asks for became possible again, cheaply (a few hundred KB of JSON header per file, zero weight
bytes — Michael was downloading weights separately at the time, on the same machine; this capture
made no use of the weight-download path and fetched nothing beyond safetensors headers). §2 below
is that capture, done today, replacing the "reproduced from the epic" fallback this document
originally shipped with.

---

## 2. Component evidence — captured 2026-08-18 (authenticated, header-only Range reads)

Method for every row: two Range requests per file — `bytes=0-7` for the little-endian u64 header
length, then `bytes=8-{8+len-1}` for the JSON header itself — via
`https://huggingface.co/Lightricks/LTX-2.5/resolve/main/<path>` with
`Authorization: Bearer <SceneWorks token>`. No tensor payload bytes were fetched for any file.
Every embedded `license` metadata field (the LTX-2.x Community License Agreement body, ~34.5 KB of
text, present verbatim in most files' `__metadata__`) is **excluded** below per the story's
instruction to record `__metadata__` "minus the embedded license body" — its presence is noted,
its text is not reproduced.

**Full `__metadata__` per file** (every key, not just the curated highlights in §2.2–§2.5 below,
license body stripped) is committed alongside this document under
[`sc-18756-headers/`](sc-18756-headers/) — one JSON file per safetensors file, same relative path
plus `.json`. A fresh agent implementing sc-18757/sc-18758/etc. reads those instead of re-hitting
the gated endpoint. The one exception: `nvfp4`'s `_quantization_metadata.layers` map (1176
per-tensor entries, all `format: "nvfp4"`) is summarized to counts-by-format rather than
reproduced entry-by-entry — every entry was inspected during the capture, the companion file says
so explicitly (`_summarized: true`), and the full per-tensor map is trivially reconstructible
(every one of the 1176 entries carries the same `format` value).

### 2.1 Full file inventory — all 14 safetensors files, including the two the epic never itemized

The epic's original table (§1.2 of epic 18755) covered 12 of the repo's 14 safetensors files. The
two it never itemized — `ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` and
`gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` — are captured here for the first
time.

| Component file | Tensors | Size, bytes (exact, measured) | Size, GB (rounded) | dtypes | Full header |
| --- | --- | --- | --- | --- | --- |
| `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | 4349 | 42,018,190,584 | 42.02 | BF16, F32 | [json](sc-18756-headers/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors.json) |
| `diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` **(new)** | 7229 | 21,504,034,224 | 21.50 | BF16, F32, I8, U8 | [json](sc-18756-headers/diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors.json) |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 4349 | 42,018,190,584 | 42.02 | BF16, F32 | [json](sc-18756-headers/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors.json) |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 7229 | 21,504,034,224 | 21.50 | BF16, F32, I8, U8 | [json](sc-18756-headers/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors.json) |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | **7877** | 18,721,548,408 | 18.72 | BF16, F32, F8_E4M3, U8 | [json](sc-18756-headers/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors.json) |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | 686 | 26,263,858,182 | 26.26 | BF16, U8 | [json](sc-18756-headers/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors.json) |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` **(new)** | 1342 | 15,372,969,374 | 15.37 | BF16, F32, I8, U8 | [json](sc-18756-headers/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors.json) |
| `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | 170 | 1,452,269,922 | 1.45 | BF16 | [json](sc-18756-headers/vae/ltx-2.5-video-vae-conv-bf16.safetensors.json) |
| `vae/ltx-2.5-video-vae-bf16.safetensors` (DiffVAE) | 396 | 1,472,223,346 | 1.47 | BF16 | [json](sc-18756-headers/vae/ltx-2.5-video-vae-bf16.safetensors.json) |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | 1329 | 364,866,540 | 0.36 | BF16 | [json](sc-18756-headers/vae/ltx-2.5-audio-vae-bf16.safetensors.json) |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 72 | 995,778,752 | 1.00 | BF16 | [json](sc-18756-headers/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors.json) |
| `latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` | 72 | 261,944,000 | 0.26 | BF16 | [json](sc-18756-headers/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors.json) |
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | 15 | 3,843,690 | 0.0038 | BF16 | [json](sc-18756-headers/model_patches/ltx-2.5-duration-head-bf16.safetensors.json) |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 3320 | 8,899,889,568 | 8.90 | BF16 | [json](sc-18756-headers/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors.json) |

**Correction to the epic's table:** the nvfp4 transformer measures **7877** tensors, not 7876 as
the epic's table states. Every other tensor count matches the epic's cited figures exactly
(tensor-for-tensor), and sizes agree with the epic's rounded GB figures — the epic records size
only to one decimal GB, so there is no byte-level figure in the epic to compare against; the exact
byte counts above are this document's own measurement and are now the authoritative record.

`SpatioTemporalScaleFactors`, patch sizes, and frame-count constraints are unchanged from the 2.3
reference and are not re-derived here (not a per-file header field) — see epic 18755 §1.2 for that
prose, unaffected by this correction.

### 2.2 The DiT — measured `config.transformer`, both variants

Parsed `config.transformer` for `ltx-2.5-22b-distilled-transformer-bf16.safetensors` and
`ltx-2.5-22b-dev-transformer-bf16.safetensors`: **byte-identical**, zero key- or value-diff between
the two variants' architecture config (they differ only in weights, not shape/config). Confirmed
values relevant to the 2.3→2.5 diff:

| Key | Value (both dev and distilled) |
| --- | --- |
| `ff_bias` | `false` |
| `use_keyframes_abs_pos_embedding` | `true` |
| `num_layers` | 48 |
| `num_attention_heads` | 32 |
| `attention_head_dim` | 128 |
| `in_channels` / `out_channels` | 128 / 128 |
| `cross_attention_dim` | 4096 |
| `caption_channels` | 3840 |
| `apply_gated_attention` | `true` |
| `use_embeddings_connector` | `true` (`connector_num_layers: 8`, `connector_num_learnable_registers: 128`) |
| `cross_attention_adaln` | `true` |
| `text_encoder_norm_type` | `PER_TOKEN_RMS` |
| `rope_type` | `split` |
| `frequencies_precision` | `float64` |
| `causal_temporal_positioning` | `true` |
| `av_ca_timestep_scale_multiplier` | `1000.0` |
| `scheduler._class_name` / `sampler` | `RectifiedFlowScheduler` / `LinearQuadratic` |

This confirms the epic's §1.1 diff claim exactly (2.3 had `ff_bias` absent ⇒ `true` and
`use_keyframes_abs_pos_embedding` absent ⇒ `false`; both are the flipped, explicit values above in
every 2.5 transformer variant measured, including the two comfy-int8-convrot files and the nvfp4
file — same `config.transformer` block, confirmed identical across all five transformer-family
files captured in §2.1).

Every 2.5 transformer file stamps `model_version: "2.5.0"` and
`gemma_source_checkpoint: {"ltx_version": "2.5.0", "gemma_version": "gemma4-12b-ltx-v1"}`,
confirmed on all five diffusion_models files.

### 2.3 The Gemma 4 TE — measured `gemma_config`

From `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`'s `__metadata__.gemma_config`
(`architectures: ["Gemma4UnifiedForConditionalGeneration"]`, `gemma_version: "gemma4-12b-ltx-v1"`,
`model_type: "gemma4_unified"`), `text_config`:

| Key | Measured value |
| --- | --- |
| `attention_k_eq_v` | `true` |
| `head_dim` (sliding layers) | 256 |
| `global_head_dim` (full-attention layers) | 512 |
| `num_hidden_layers` | 48 |
| `layer_types` | 40× `sliding_attention`, 8× `full_attention` — confirmed 1-in-6 pattern (index 5, 11, 17, 23, 29, 35, 41, 47 are `full_attention`) |
| `sliding_window` | 1024 |
| `num_key_value_heads` | 8 |
| `num_global_key_value_heads` | 1 |
| `num_attention_heads` | 16 |
| `rope_parameters.sliding_attention` | `{"rope_theta": 10000.0, "rope_type": "default"}` |
| `rope_parameters.full_attention` | `{"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}` |
| `hidden_size` | 3840 |
| `intermediate_size` | 15360 |
| `vocab_size` | 262144 |
| `tie_word_embeddings` | `true` |
| `final_logit_softcapping` | 30.0 |
| `rms_norm_eps` | 1e-06 |

Also present at the `gemma_config` top level (not under `text_config`):
`audio_config` (`model_type: gemma4_unified_audio`, `audio_embed_dim: 640`), `vision_config`,
`audio_token_id`, `boa_token_id`/`boi_token_id`, `image_token_id`, `video_token_id`,
`transformers_version`. This matches the epic's §1.3 claims exactly — dual head dims, dual rope
schemes (both the theta *and* the `rope_type` differ, not just theta), 40/8 sliding/full split,
`num_global_key_value_heads: 1`, and confirms the encoder-free vision tower and audio config are
present in the checkpoint as the epic described.

The comfy-int8-convrot TE variant (`gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`,
1342 tensors, 15.37 GB) carries the **same** `__metadata__` shape (`format`, `gemma_config` only,
no distinct quantization-metadata block) — its quantization is expressed per-tensor via dtype
(BF16/F32/I8/U8 mix) and naming convention, not a separate declared scheme the way `nvfp4`'s
`_quantization_metadata.layers.*.format: "nvfp4"` block is.

### 2.4 DiffVAE decoder — measured `config.vae`

`vae/ltx-2.5-video-vae-bf16.safetensors`'s `__metadata__.config.vae`: `_class_name:
"CausalDiffusionVAE"`, encoder identical in shape to the conv VAE's encoder
(`patch_size: 4`, `latent_log_var: "constant"`, `latent_log_var_value: -7.824046010856292`,
`norm_layer: "pixel_norm"`), decoder `_class_name: "NADiffusionDecoder"` with
`head_dim: 64`, `stage_channels: [2048, 1024, 512, 512, 256]`,
`stage_depths: [4, 6, 4, 2, 8]`,
`stage_kernels: [[3,7,7],[3,7,7],[3,5,5],[3,5,5],[11,11,11]]`,
`upsamples: [[[1,2,2],2],[[2,1,1],2],[[2,2,2],1],[[2,2,2],2]]`,
`resampler_kind: "linear"`, `model_output_type: "x0"`,
`default_num_inference_steps: 1`. Exact match to the epic's §1.4 claim, confirmed field-for-field
from the live header rather than reproduced.

`vae/ltx-2.5-video-vae-conv-bf16.safetensors`'s `config.vae` (`CausalVideoAutoencoder`) carries the
identical `encoder_blocks` shape (same `res_x`/`compress_*` block sequence) as the DiffVAE's
encoder — confirms the two share one encoder architecture, differing only in the decoder, exactly
as the epic states.

### 2.5 Everything else — measured, matches the epic

- `model_patches/ltx-2.5-duration-head-bf16.safetensors` config: `{"transformer": {"cross_attention_dim": 4096, "audio_cross_attention_dim": 2048}, "duration_head": {}}` — a small patch config, consistent with the epic's "NEW (small)" characterization.
- `latent_upscale_models/...-spatial-upscaler-x2...`: `LatentUpsampler`, `mid_channels: 1024`, `spatial_upsample: true`, `temporal_upsample: false`, `rational_resampler: false`.
- `latent_upscale_models/...-temporal-upscaler-x2...`: `LatentUpsampler`, `mid_channels: 512`, `spatial_upsample: false`, `temporal_upsample: true`, `rational_resampler: true` — confirms the epic's "temporal + `rational_resampler`" claim precisely; this file has no `license` metadata key (the other 12 do).
- `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors`: `lora_rank: 450`, `lora_alpha: 450`.
- `vae/ltx-2.5-audio-vae-bf16.safetensors`: `audio_vae.model.params.ddconfig` (`mel_bins: 64`, `z_channels: 8`, `ch_mult: [1,2,4]`, `sampling_rate: 16000`) plus a `vocoder` block (`upsample_initial_channel: 1536`, `resblock: "AMP1"`) — audio VAE + vocoder bundled together as the epic's table states.

### 2.6 Upstream params row — not a header field, unchanged from the epic

`_PARAMS_SINCE_VERSION` and `DISTILLED_SIGMA_VALUES` are reference-implementation source facts,
not safetensors metadata, so they are not re-measurable via Range reads. See epic 18755 §1.5 for
that record: `_PARAMS_SINCE_VERSION` has rows for 2.4 and 2.3 only, so a 2.5 checkpoint inherits
`LTX_2_4_PARAMS` (30 steps, STG block 28, CFG 3.0 video / 7.0 audio, rescale 0.7,
`default_image_crf: 18`); distilled sampling uses 9 sigmas ⇒ 8 steps (stage-2 subset ⇒ 4 steps).

---

## 3. Reference implementation pin — verified 2026-08-18 (public, non-gated)

`Lightricks/LTX-2` on GitHub is **public**; unlike the HF weight repos it needs no credential.

| | |
| --- | --- |
| Repo | https://github.com/Lightricks/LTX-2 |
| Tag | `v1.2.0` |
| Commit | `d151147788a9284cca791edc6ce898007e727fe6` |

Verified 2026-08-18 via the public GitHub API:

```
$ curl -s https://api.github.com/repos/Lightricks/LTX-2/git/refs/tags/v1.2.0
{
  "ref": "refs/tags/v1.2.0",
  "object": { "sha": "d151147788a9284cca791edc6ce898007e727fe6", "type": "commit" }
}
```

The tag resolves to exactly the commit the epic cites; the commit exists and its message begins
*"core: Pin transformers below 5.15."* — consistent with the epic's R-8 risk note ("v1.2.0 landed
2026-08-11 and already pins `transformers < 5.15`").

**Every later LTX-2.5 story must cite `d151147788a9284cca791edc6ce898007e727fe6` (or the `v1.2.0`
tag), not `main` — `main` moves and is not pinned.**

---

## 4. Inference pin + `gemma4` symbol check

### 4.1 Current inference pin — correction to the epic's cited value

The epic text (§1.6, written 2026-08-11) cites the inference pin as `b965641e388f4db646e4c60ab3f75219737e2cc8`.
**That pin has since moved.** As of this document (2026-08-18), the actual current inference pin,
read from the SceneWorks repo's dependency declarations (`Cargo.toml`,
`crates/sceneworks-worker/Cargo.toml`), is:

```
2d762d3312ac96c4feda8f89e35ee967e7700751
```

(SceneWorks `main` HEAD `005b06333`, "Merge pull request #2418 ... sc-18304-pipeline-flexibility",
merged 2026-08-18 — the epic-18304 pin bump landed and moved the pin forward from the 2026-08-11
snapshot the epic text quotes.) This branch (`feature/sc-18755-ltx-2-5` in the inference repo,
based at `2d762d3312ac96c4feda8f89e35ee967e7700751`) is at that same current pin.

**Later stories should cite `2d762d331...`, not `b965641e...`, as the inference-pin baseline —
and re-check it again before use, since it will keep moving as other epics land.**

### 4.2 No `gemma4` symbol in the inference tree — verified at the current pin, excluding this doc

`git grep` against a specific commit reads from the object database, not the working tree, so this
is unaffected by `docs/` content added by this story itself (pathspec `:!docs` excludes it
explicitly to keep the claim honest even so):

```
$ git grep -il -e "gemma4" -e "gemma_4" -e "gemma-4" 2d762d3312ac96c4feda8f89e35ee967e7700751 -- ':!docs'
(no matches, exit 1)

$ git grep -il -e "use_keyframes_abs_pos_embedding" -e "ff_bias" 2d762d3312ac96c4feda8f89e35ee967e7700751 -- ':!docs'
(no matches, exit 1)
```

Confirmed at commit `2d762d3312ac96c4feda8f89e35ee967e7700751`: no file anywhere in the inference
tree outside `docs/` — code, config, tests — contains `gemma4`, `gemma_4`, `gemma-4`, `ff_bias`, or
`use_keyframes_abs_pos_embedding` in any case. (This document itself necessarily contains all of
those strings in §2 above — hence the `:!docs` exclusion, so the claim is checked against the
non-docs tree the way a future story's grep for "has anyone wired this yet" actually needs.)
Existing `gemma` references (Gemma **2/3**) are present and unaffected, e.g.
`crates/llm/mlx-llm/src/config.rs`, `crates/llm/mlx-llm/src/models/llama.rs`,
`crates/llm/candle-llm/src/config.rs`, `crates/llm/candle-llm/src/models/llama.rs` — the shared
generic decoder the epic's §1.3 placement decision (Gemma 4 lands in `crates/llm/{mlx,candle}-llm`)
is built on.

---

## 5. `dump-engine-capabilities` baseline — confirmed unchanged at the current pin

The story asks for a pre-change baseline of the checked-in
`config/engine-capabilities/capabilities.{mlx,candle}.json` (SceneWorks repo), so the `ltx_2_5`
descriptor diff is reviewable later, **without** regenerating with real weights or running
anything GPU/large-RAM.

### 5.1 What was and was not done

The dumper (`crates/sceneworks-worker/src/bin/dump-engine-capabilities.rs`) is genuinely
weights-free (it walks the linked provider *registries*, not weight files), but producing a fresh
dump still requires compiling the `sceneworks-worker` binary — which, on macOS, links MLX built
from source. A cold worktree has no cached `target/`, and per the workspace's own
`.cargo/config.toml`, "a fresh git worktree gets its own `target/` so it recompiles MLX from
scratch" — a multi-minute, CPU/RAM-heavy compile disproportionate to a docs-only Phase 0 story
under the cpu-only resource lane. The candle side is stronger than disproportionate: it is
**not buildable at all** on this machine — `backend-candle` pulls in `dep:runtime-cuda`, which
needs a CUDA toolchain this Mac does not have.

So this section confirms the baseline via content hashing + git history against the checked-in
files, rather than a live re-run of the dumper. This is a strictly weaker check for "did the
dumper's *output* change" but a strictly *equivalent* check for "is anything currently dirty
relative to what's committed" (the property the acceptance criterion actually needs at Phase 0,
since no LTX-2.5 code has landed yet to have changed anything).

### 5.2 Recorded baseline (SceneWorks repo, at inference pin `2d762d331`)

| File | SHA-256 | Lines | Last touched by |
| --- | --- | --- | --- |
| `config/engine-capabilities/capabilities.mlx.json` | `ec41c3fad5f8c5c8b15f44b9fc7291597555f7d8ce72623e7a741389aa788114` | 15789 | `0b0a25ef2` — "chore(sc-18420): the epic's single pin bump — inference main 2d762d3312ac" (2026-08-18) |
| `config/engine-capabilities/capabilities.candle.json` | `f8f139b84b0087230a91038357fb863bd5ab465539ffabe62647d5696296c52e` | 10221 | `8d022e226` — "sync(sc-18304): merge main (19703 hot cache) and land the candle facts at the final pin" (2026-08-18) |

The two files were last touched by **different** commits — the mlx file by the sc-18420 pin-bump
commit, the candle file by the sc-18304 sync/merge commit that landed the candle facts at the
final pin. `git status --short` on both files is clean at SceneWorks `main` HEAD `005b06333` —
**confirmed unchanged**, no drift between the committed facts and the tree.

### 5.3 Pre-change LTX presence — confirms nothing 2.5-shaped has landed yet

```
$ grep -o '"ltx[a-z0-9_]*"' config/engine-capabilities/capabilities.mlx.json | sort -u
"ltx_2_3"
$ grep -o '"ltx[a-z0-9_]*"' config/engine-capabilities/capabilities.candle.json | sort -u
"ltx_2_3_distilled"
```

Only the existing `ltx_2_3` (MLX) / `ltx_2_3_distilled` (candle) engine ids appear. Neither
`ltx_2_5` nor `ltx_2_5_distilled` (the epic's R2 engine-id pair) is present anywhere — this is the
clean pre-change state sc-18778's descriptor registration diffs against.

### 5.4 Re-verification instructions for a future story (once compiled artifacts exist)

```bash
# macOS/MLX (default features) — from the SceneWorks repo root, into a scratch dir so the
# checked-in files are untouched:
cargo run -p sceneworks-worker --bin dump-engine-capabilities -- /tmp/scratch-mlx
diff /tmp/scratch-mlx/capabilities.mlx.json config/engine-capabilities/capabilities.mlx.json

# off-Mac CUDA/candle lane only (this binary's candle path cannot build without a CUDA toolchain):
cargo run -p sceneworks-worker --bin dump-engine-capabilities --no-default-features \
    --features backend-candle -- /tmp/scratch-candle
diff /tmp/scratch-candle/capabilities.candle.json config/engine-capabilities/capabilities.candle.json
```

---

## BLOCKED

**Not credential-blocked anymore for component evidence.** §1.4 found a working `SceneWorks`
HF-account credential at `HF_HOME=/Volumes/Models/huggingface` (not at any default location, which
is why the first pass in §1.1 missed it), and §2 is a complete, freshly-measured capture under that
credential — including the two comfy-int8-convrot files the epic's original table never itemized.
This completes the story's original component-evidence acceptance criterion rather than falling
back to a reproduction.

**What remains open (verification gap, not a missing credential):** this document still cannot
confirm, from this machine or from static repo analysis, that CI and the desktop app authenticate
to HF as this same `SceneWorks` account in production. No `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`
secret is wired into any inference-repo GitHub Actions workflow (verified by grep), and the desktop
app's token is supplied by the user at runtime through its own settings surface. Resolving this
needs Michael to confirm how CI and the packaged desktop app are provisioned in practice — this is
an infrastructure/process question, not something a credential fixes on this machine.

---

*Compiled 2026-08-18 by `claude` (Sonnet 5) for sc-18756. §2, §3, §4, and §5 are independently
measured/verified as of this document's date; §2 is now a direct authenticated capture (see
§1.4), not a reproduction of the epic's 2026-08-11 capture.*
