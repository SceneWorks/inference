# LTX-2.5 access + component evidence baseline (sc-18756)

| | |
| --- | --- |
| Story | sc-18756 — Phase 0 of epic 18755 (`LTX-2.5: native MLX + candle video generation`) |
| Epic | [18755](https://app.shortcut.com/trefry/epic/18755) |
| Purpose | Pin the factual baseline the rest of the epic is built on, in-repo, so no later story re-derives it or guesses |
| Gathered by | Claude (Sonnet 5), automated agent, on behalf of Michael Trefry |
| Access probe date (this document) | **2026-08-18** |
| Component evidence provenance | **Captured 2026-08-11** by an authenticated session under HF account **`SceneWorks`**, after the LTX-2.x Community License gate was accepted on that account. Reproduced here from epic 18755 §1 (which is itself the record of that capture) — **not independently re-measured on 2026-08-18**. See [§2](#2-component-evidence-2026-08-11-capture-reproduced-not-re-measured). |
| **Read and signed off by a human** | **_(NOT YET — Michael)_** |

## 0. Why this document exists

Every later LTX-2.5 story cites facts instead of re-deriving them: the reference-impl pin, the
measured 2.3→2.5 transformer config diff, the per-component tensor/size/dtype/`__metadata__`
inventory, and the pre-change engine-capabilities baseline. This document is the citable home for
those facts, plus a dated record of what HF access actually looks like from this machine today.

**Nothing in §2 was re-measured for this document.** `Lightricks/LTX-2.5` is a gated repository
and no HF credential is configured on this machine (see §1). §2 is a faithful transcription of the
measured data already recorded in the epic description, captured 2026-08-11 by a different,
authenticated session. It is reproduced here, in-repo, so a fresh agent does not need to re-hit the
gated endpoints (and cannot, without a credential) to get at it — not presented as freshly measured.

---

## 1. HF access state — measured 2026-08-18, unauthenticated

### 1.1 Credential inventory on this machine

```
$ hf auth whoami
Error: Not logged in

$ env | grep -i hf_token        # (no output)
$ env | grep -i huggingface     # (no output)
$ ls ~/.cache/huggingface/token  # No such file or directory
$ ls ~/.huggingface/token        # No such file or directory
```

No HF token is configured on this machine — no CLI login, no environment variable, no cached
token file.

### 1.2 Unauthenticated probes against `Lightricks/LTX-2.5`

Three probes, all run 2026-08-18, all unauthenticated:

| Probe | Result | Notes |
| --- | --- | --- |
| `GET /Lightricks/LTX-2.5/resolve/main/README.md` | **200** | The README and repo metadata are public even though the repo is gated |
| `GET /api/models/Lightricks/LTX-2.5` | **200**, `"gated":"auto"` | Model-info API confirms `gated: auto`, `license: other` (`ltx-2-community-license-agreement`), `license_link: https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md`. Full `siblings` file listing (filenames + paths) is visible without auth. |
| `Range: bytes=0-1000` on `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (the safetensors-header read the epic's evidence table is built from) | **401** | `x-error-code: GatedRepo`, body: *"Access to model Lightricks/LTX-2.5 is restricted. You must have access to it and be authenticated to access it. Please log in."* |

**Conclusion — the actual, current access state:** the repo's *card* (README, model-info JSON,
file listing) is reachable without a token, but every **weight file** is gated behind
`401 GatedRepo`. This confirms the story's prediction exactly: the Range-request header reads
that produced §2's data cannot be reproduced from this machine right now. The gate is per-account
and, per the epic, was accepted on the `SceneWorks` HF account on 2026-08-11 — that acceptance
does not travel to this machine without a credential.

### 1.3 Gate status for the account CI and the desktop app actually use

This machine has no HF credential, so this document **cannot** verify which HF account (if any)
CI or the desktop app authenticate as, or whether that account currently has the LTX-2.x gate
accepted. What is verifiable statically, from the repo:

- `grep -rli "HF_TOKEN\|HUGGING_FACE" .github/workflows/` in this repo returns **no matches** —
  no `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` secret is wired into any inference-repo GitHub Actions
  workflow.
- `scripts/check-download-patterns.mjs` (the SceneWorks-repo download-reachability auditor) reads
  its token from `process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN` — i.e. the tooling
  supports a token but does not embed or default one.
- The desktop app's HF token (if any) is user-supplied at runtime via its own settings surface,
  not something inspectable from a static repo check.

**This is exactly the credential gap in [BLOCKED](#blocked-credential--michael-only) below** —
resolving it requires Michael's own account state, not something derivable from this machine.

---

## 2. Component evidence (2026-08-11 capture, reproduced — not re-measured)

Everything in this section is transcribed verbatim from epic 18755 §1 (activity authored
2026-08-11 by an authenticated `claude` session under the `SceneWorks` HF account, after the gate
was accepted). **Re-verify against a fresh authenticated read once credentials are restored on
this machine** — see [BLOCKED](#blocked-credential--michael-only).

### 2.1 The DiT is nearly unchanged from 2.3

Diffing `__metadata__.config.transformer` of `ltx-2.3-22b-distilled-1.1.safetensors` against
`ltx-2.5-22b-distilled-transformer-bf16.safetensors` yields **exactly two changed keys**:

| Key | LTX-2.3 | LTX-2.5 |
| --- | --- | --- |
| `ff_bias` | absent (⇒ `true`) | `false` |
| `use_keyframes_abs_pos_embedding` | absent (⇒ `false`) | `true` |

Everything else is identical: `num_layers: 48`, `num_attention_heads: 32`,
`attention_head_dim: 128`, `in/out_channels: 128`, `cross_attention_dim: 4096`,
`caption_channels: 3840`, `apply_gated_attention: true`, `use_embeddings_connector: true`
(8 connector layers, 128 learnable registers), `cross_attention_adaln: true`,
`text_encoder_norm_type: PER_TOKEN_RMS`, `rope_type: split`, `frequencies_precision: float64`,
`causal_temporal_positioning: true`, `av_ca_timestep_scale_multiplier: 1000.0`, scheduler
`RectifiedFlowScheduler`/`LinearQuadratic`.

**Verified pre-change (2026-08-18, this document):** neither `ff_bias` nor
`use_keyframes_abs_pos_embedding` appears anywhere in the inference tree today
(`grep -ril "use_keyframes_abs_pos_embedding\|ff_bias"` — no matches), confirming these two flags
are genuinely new wiring, not already-present dead config.

### 2.2 Split checkpoints replace the 2.3 all-in-one

2.3 shipped **one** file (5947 tensors) whose metadata carried `transformer` + `vae` +
`audio_vae` + `vocoder` together. 2.5's transformer file carries **only** `transformer` +
`scheduler`; the other sections are `null`. Each component ships its own config:

| Component file | Tensors | Size | Config `_class_name` | Status |
| --- | --- | --- | --- | --- |
| `diffusion_models/…-distilled-transformer-bf16` | 4349 | 42.0 GB | `AVTransformer3DModel` | reuse + 2 flags |
| `diffusion_models/…-dev-transformer-bf16` | 4349 | 42.0 GB | same config incl. `ff_bias:false` | reuse |
| `diffusion_models/…-distilled-…-comfy-int8-convrot` | 7229 | 21.5 GB | I8+U8+BF16+F32 | evaluate |
| `diffusion_models/…-distilled-…-nvfp4` | 7876 | 18.7 GB | F8_E4M3+U8 | Blackwell-only |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16` | 686 | 26.3 GB | `Gemma4UnifiedForConditionalGeneration` | **NEW** |
| `vae/ltx-2.5-video-vae-conv-bf16` | 170 | 1.45 GB | `CausalVideoAutoencoder` | **drop-in reuse** |
| `vae/ltx-2.5-video-vae-bf16` (DiffVAE) | 396 | 1.47 GB | `CausalDiffusionVAE` / `NADiffusionDecoder` | **NEW** |
| `vae/ltx-2.5-audio-vae-bf16` | 1329 | 0.36 GB | audio_vae + vocoder | **drop-in reuse** |
| `latent_upscale_models/…-spatial-upscaler-x2` | 72 | 1.00 GB | `LatentUpsampler` mid 1024 | drop-in reuse |
| `latent_upscale_models/…-temporal-upscaler-x2` | 72 | 0.26 GB | `LatentUpsampler` temporal + `rational_resampler` | **NEW path** |
| `model_patches/ltx-2.5-duration-head-bf16` | 15 | ~4 MB | `DurationHead` | **NEW (small)** |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16` | 3320 | 8.90 GB | `lora_rank:450`, `lora_alpha:450` | reuse |

Every 2.5 file stamps `model_version: 2.5.0`; every transformer also stamps
`gemma_source_checkpoint: {"ltx_version":"2.5.0","gemma_version":"gemma4-12b-ltx-v1"}`. Upstream
**hard-errors** when the TE's declared `gemma_version` disagrees
(`encoder_configurator._check_gemma_version`), so the loader must plumb and honour that assertion
rather than trusting file layout.

`SpatioTemporalScaleFactors` stays **32×32×8** (`diffusion_video_decoder.py:115` pins
`.default()`), and the conv VAE's `encoder_blocks`/`patch_size: 4` are unchanged from 2.3 —
latent geometry is unchanged. Frame count stays `n % 8 == 1`; W/H must be ÷32 upstream, but
SceneWorks keeps **÷64** because stage 1 runs at `//2//32` (same reason as `ltx_2_3`).

**Independently confirmed, non-gated (2026-08-18, this document):** `Lightricks/LTX-2.5`'s public
`siblings` file listing (via the unauthenticated `/api/models/Lightricks/LTX-2.5` call in §1.2)
matches this component list byte-for-byte on filenames — every path in the table above appears in
the live listing, plus `ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` and
`gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` (comfy int8-convrot variants of the
dev transformer and the TE respectively, not itemized in the epic's table). File **contents**
(tensor counts, sizes, dtypes, `__metadata__`) remain gated and are not independently re-verified
here.

### 2.3 The Gemma 4 TE is the real new model

`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` is **fully self-contained** — no separate
`google/gemma-4-12B` download, unlike 2.3's separate 26.4 GB `gemma/` co-requisite. Measured
contents: 664 `model.layers.N.*` tensors (48 layers), `model.embed_tokens`, `model.norm`, the
`text_embedding_projection.{video,audio}_aggregate_embed` heads (the "with-proj" part), the
encoder-free vision tower (`vision_model.patch_dense`/`patch_ln1`/`patch_ln2`/`pos_embedding`/
`pos_norm`), `multi_modal_projector`, `audio_projector`, and packed HF assets as uint8 tensors:
`tokenizer_json`, `hf_asset__chat_template.jinja`, `hf_asset__tokenizer_config.json`,
`hf_asset__processor_config.json`, `hf_asset__generation_config.json`. Config arrives in
`__metadata__.gemma_config`.

Gemma 4 (`gemma4_unified`, Google, released 2026-06-03) differs from the shipped Gemma 3 port in
ways that are **not** parameter tweaks:

- `attention_k_eq_v: true` — K and V share a projection.
- **Two head dims**: `head_dim: 256` for sliding layers, `global_head_dim: 512` for full-attention
  layers.
- **Two rope schemes**: sliding layers `rope_type: default`, θ=10 000; full-attention layers
  `rope_type: proportional`, θ=1 000 000, `partial_rotary_factor: 0.25`.
- 48 layers as **40 sliding / 8 full**, `sliding_window: 1024`, full layer every 6th.
- `num_key_value_heads: 8` but `num_global_key_value_heads: 1`.
- `hidden_size: 3840`, `intermediate_size: 15360`, `vocab_size: 262144`,
  `tie_word_embeddings: true`, `final_logit_softcapping: 30.0`, `rms_norm_eps: 1e-6`.

Encode is text-only (`self.model.model(input_ids, attention_mask, output_hidden_states=True)` —
no pixel values), so the vision tower is required only for the optional i2v prompt enhancer.

### 2.4 DiffVAE decoder

`CausalDiffusionVAE` keeps the **same conv `Encoder`** as the conv VAE (`patch_size: 4`,
`latent_log_var: constant`, value `-7.824046010856292`, `pixel_norm`) and replaces only the
decoder with `NADiffusionDecoder`: neighborhood attention, `head_dim: 64`,
`stage_channels: [2048,1024,512,512,256]`, `stage_depths: [4,6,4,2,8]`, 3D `stage_kernels`
`[[3,7,7],[3,7,7],[3,5,5],[3,5,5],[11,11,11]]`,
`upsamples: [[[1,2,2],2],[[2,1,1],2],[[2,2,2],1],[[2,2,2],2]]`, `resampler_kind: linear`,
`model_output_type: x0`, **`default_num_inference_steps: 1`**. Upstream backs it with
NATTEN/CUTLASS-FNA plus `chunked_eager`/`chunked_compile`/`combined_compile`/Blackwell-DSL modes
whose memory coefficients differ by **>4×** (11 / 7 / 5 / 2.5 × `stage5_channels`). The conv VAE
remains a complete fallback.

### 2.5 Upstream params row

`_PARAMS_SINCE_VERSION` contains rows for **2.4 and 2.3 only**. A 2.5 checkpoint inherits
`LTX_2_4_PARAMS`: 30 steps, STG block 28, CFG 3.0 video / 7.0 audio, rescale 0.7,
`default_image_crf: 18` (2.3 used 33). Distilled sampling: `DISTILLED_SIGMA_VALUES` = 9 sigmas ⇒
**8 steps**, stage-2 subset ⇒ 4 steps. Trainer validation defaults moved to **960×544×89 @ 24 fps,
30 steps, STG block 28**.

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
HEAD `2d762d3312ac96c4feda8f89e35ee967e7700751`) is at that same current pin.

**Later stories should cite `2d762d331...`, not `b965641e...`, as the inference-pin baseline —
and re-check it again before use, since it will keep moving as other epics land.**

### 4.2 No `gemma4` symbol in the inference tree — verified at the current pin

```
$ grep -ril "gemma4\|gemma_4\|gemma-4" .      # (run from the inference repo root)
(no matches)
```

Confirmed at commit `2d762d3312ac96c4feda8f89e35ee967e7700751` (this branch's current HEAD): no
file anywhere in the inference tree — code, config, docs, tests — contains `gemma4`, `gemma_4`,
or `gemma-4` in any case. Existing `gemma` references (Gemma **2/3**) are present and unaffected,
e.g. `crates/llm/mlx-llm/src/config.rs`, `crates/llm/mlx-llm/src/models/llama.rs`,
`crates/llm/candle-llm/src/config.rs`, `crates/llm/candle-llm/src/models/llama.rs` — the shared
generic decoder the epic's §1.3 placement decision (Gemma 4 lands in `crates/llm/{mlx,candle}-llm`)
is built on. This corroborates the epic's `mlx-llm/Cargo.toml` claim that "mlx-gen will consume
mlx-llm" is already the documented direction, not aspirational.

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
| `config/engine-capabilities/capabilities.mlx.json` | `ec41c3fad5f8c5c8b15f44b9fc7291597555f7d8ce72623e7a741389aa788114` | 15789 | `8d022e226` — "sync(sc-18304): merge main (19703 hot cache) and land the candle facts at the final pin" (2026-08-18) |
| `config/engine-capabilities/capabilities.candle.json` | `f8f139b84b0087230a91038357fb863bd5ab465539ffabe62647d5696296c52e` | 10221 | same commit |

`git status --short` on both files is clean at SceneWorks `main` HEAD `005b06333` — **confirmed
unchanged**, no drift between the committed facts and the tree.

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

## BLOCKED (credential — Michael only)

**No HF token is configured on this machine**, and `Lightricks/LTX-2.5` /
`Lightricks/LTX-2.5-Diffusers` are gated (`gated: auto`). §1.2 confirms the actual, current
access state: the repo card is public but every weight file 401s (`GatedRepo`) unauthenticated.

**To unblock:** run `hf auth login` on this machine with the `SceneWorks` HF account — the same
account that accepted the LTX-2.x Community License gate on 2026-08-11 (per the epic). Once that
credential is present:

1. Re-run the Range-request header reads behind §2 and confirm they still match (the account's
   access is revocable and per-account, per the epic's own caveat).
2. Verify §1.3's open question — which HF account CI and the desktop app actually authenticate
   as, and whether that account (as opposed to a dev token) currently has the gate accepted. This
   document could not determine that from static analysis: no `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`
   secret is wired into any inference-repo GitHub Actions workflow today, and the desktop app's
   token is user-supplied at runtime.

Nothing else in this story is blocked on this credential — §3 (reference-impl pin), §4 (inference
pin + `gemma4` grep), and §5 (capabilities baseline) are all independently verified above.

---

*Compiled 2026-08-18 by `claude` (Sonnet 5) for sc-18756. §2 is a reproduction of epic 18755 §1's
2026-08-11 authenticated capture; §1, §3, §4, and §5 are independently measured/verified as of
this document's date, from this (unauthenticated, cpu-only) machine.*
