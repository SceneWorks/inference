# Media checkpoint census — what the media lane actually loads (sc-16665)

| | |
| --- | --- |
| Story | sc-16665 — shared checkpoint licence table, one tensor-neutral source both media catalogs read |
| Epic | 16660 |
| Feeds | sc-16666 (MLX provider→component mapping), sc-16667 (Candle provider→component mapping) |
| Gathered by | Claude (Opus 5), automated agent, on behalf of Michael Trefry |
| Date | **2026-08-02** |
| Source of truth | this repository at `origin/main` `5083b0ec`. **Nothing here was read from the network.** |
| Companion | [Licence family evidence pack (sc-16662)](sc-16662-licence-family-evidence.md) — the sixteen families this census is measured against |

## What this document is

A **factual census of components**, not a licence table. It answers one question: for every provider
id the two media catalogs actually register, which upstream checkpoints does that provider load —
the DiT/UNet/transformer *and every auxiliary* (text encoders, VAEs, tokenizers, vision towers,
ControlNets, adapters, face-analysis models, latent decoders).

It deliberately does **not** assign licence families and does **not** state what any checkpoint is
licensed under. Where the code names an upstream, that name is transcribed with a file:line. Where
the code does **not** say where a component came from, the row reads **UNDETERMINED** — that is a
finding, not a gap to be filled with a plausible guess. Assigning a family to an UNDETERMINED
component would be inventing the fact the table exists to record.

## Disclosure only

Nothing in this surface blocks, gates, degrades or withholds anything, and nothing built from it
ever should. It exists so a consumer can *see* which upstream artifacts a render touched.

## Why the table has to be component-keyed

Three facts from this census, each of which a provider-id-keyed table cannot express:

1. **Auxiliaries outnumber DiTs.** A typical image provider loads four to seven distinct upstream
   artifacts. `boogu_image` is the epic's example (Boogu DiT + Qwen3-VL-8B + FLUX.1 VAE) and it is
   on the low side.
2. **Components are shared across dozens of providers.** One `openai/clip-vit-large-patch14`
   snapshot serves the two embedder ids, SDXL, SD3.5 and the FLUX IP-adapter path. The PiD latent
   decoder is an optional overlay on **fourteen** provider crates. Keyed by provider id, the same
   artifact would be transcribed dozens of times and drift dozens of ways.
3. **Some components belong to no provider id at all.** Six crates per backend register nothing and
   still load real checkpoints — see [the overlay section](#non-registry-overlay-components-epic-r6).
   Today those checkpoints are invisible to any provider-keyed view.

## Method, and its one real limit

epic-13657 removed self-fetch, so no provider crate carries a `HUB_REPO`/`HUB_REVISION` constant;
every component arrives as a caller-provisioned local path (`WeightsSource::Dir`/`File`). Upstream
identity therefore survives only in **prose** — crate-level and module rustdoc, loader doc comments,
NOTICE files, the directory and file names a loader expects, conformance fixtures, and
`release/real-weight-models.toml`. This census reads all of those.

The limit that follows is structural and worth stating plainly: **a redistributed component inside
another party's snapshot usually has no recorded origin.** When Chroma ships its own `text_encoder/`,
the code says "T5-XXL" and stops; it does not say which T5-XXL. Roughly a third of the component
rows below are UNDETERMINED for exactly this reason, and no amount of further grepping changes that
— the information is not in the repository. Settling them means reading the upstream model cards,
which is network work this story did not do.

# READ FIRST — what this census found that the epic did not already know

## 1. UNCOVERED — 47 of the 90 distinct upstream artifacts have no landed family anchored to them

The sixteen families in `crates/contracts/gen-core/src/license/families.rs` were chosen for the
**generator bases**. They cover those well: FLUX.1 [dev], Krea 2, LTX-2, Anima, SD3.5, SVD-XT, SDXL,
Gemma, PiD, insightface, ChatGLM3 and Llama-3.1 are all anchored to a real family. What they do not
cover is the long tail — **ControlNets, IP-adapters, identity encoders, distill LoRAs, community
re-hosts, and six whole model vendors**.

Counted from the [index](#distinct-upstream-checkpoint-index): **47 UNCOVERED, 15 generic?, 28
anchored** out of 90 rows. "UNCOVERED" here means precisely: *no landed family is anchored to this
licensor or model, and the repository's own declaration is not recorded anywhere in this repository,
so whether an existing family fits cannot be settled from the code.* Many of the 47 will turn out to
be Apache-2.0 or MIT once someone reads the cards. The ones most likely to need a **new** family are
the bespoke vendor licences:

| licensor | artifacts | why a new family is likely |
| --- | --- | --- |
| Black Forest Labs, FLUX.2 | `FLUX.2-klein-9B`, `FLUX.2-klein-9b-kv`, `FLUX.2-dev` | BFL publishes a separate licence per model generation; `flux-1-dev-non-commercial` is transcribed from the FLUX.1 [dev] text and must not be stretched over FLUX.2 |
| Ideogram | `ideogram-ai/ideogram-4-fp8` | stated **gated**; a click-through release with no licence recorded in-repo |
| Kwai-Kolors | `Kolors-diffusers` (repo level), `Kolors-ControlNet-Pose`, `Kolors-IP-Adapter-Plus` | sc-16662 **U6** left the Kolors repo-level document unresolved; the two adapter repos were never considered |
| Microsoft Research | `microsoft/Lens`, `microsoft/Lens-Turbo` | the crate records that **Microsoft pulled the original repository**, so the licence document may not be retrievable at all |
| SenseTime | `SenseNova-U1-8B-MoT` + its LoRA repo | no family, no in-repo declaration |
| zai-org | `SCAIL-2` | no family, no in-repo declaration |
| Meta | `facebook/sam2.1-*`, `facebook/sam3` | SAM releases carry their own licences; nothing in the table is close |
| Boogu | `Boogu-Image-0.1-Base` / `-Turbo` / Edit | the *only* upstream pointer is a tools README; `candle-gen-boogu` asserts "Apache-2.0, ungated" with nothing behind it |
| ByteDance | `Bernini-Diffusers` | no family, no declaration; supplies six distinct prefixes |
| lodestones | `Chroma1-HD` / `-Base` / `-Flash` | no family, no declaration |
| Wan-AI / alibaba-pai | five Wan and four Fun-Controlnet repos | no family; only one of the nine carries any in-repo licence prose |

## 2. UNDETERMINED — 24 loaded components whose upstream the code never states

These are components the providers demonstrably load and whose origin **is not recorded anywhere in
the repository**. They are not gaps in this census; they are gaps in the repository. Each one is a
component row that cannot be written without reading an upstream card.

Two distinct shapes:

**(a) Redistributed inside another party's snapshot.** The code names the architecture and stops —
because that is all the loader needs. 16 rows: FLUX.1's CLIP-L / T5-XXL / VAE; FLUX.2's Qwen3 and
Mistral3 towers, the Pixtral vision tower and multimodal projector; Chroma's T5-XXL; Boogu's
Qwen3-VL-8B and FLUX.1 VAE; Anima's Qwen3-0.6B and Qwen-Image VAE; Krea's Qwen3-VL-4B; Mage's
Qwen3-VL-4B; Ideogram's Qwen3-VL-8B; Lens's gpt-oss-20b; Z-Image's Qwen3; SD3.5's T5-XXL;
Kolors's ChatGLM3-6B; qwen-image's Qwen2.5-VL; SCAIL-2's UMT5 / open-CLIP ViT-H / Wan2.1 VAE;
bernini's and krea-realtime's "stock Wan2.2" UMT5 / VAE / tokenizer.

**(b) Named by package, never by repository.** 8 rows: antelopev2 ArcFace `glintr100`; antelopev2
SCRFD-10g; facexlib `parsing_bisenet`; the Boogu Edit checkpoint; ByteDance Hyper-FLUX 8-step LoRA;
ostris TurboTime LoRA; the lightx2v Wan step-distill diff-patch; the `sat-scail2` DPO LoRA. Three of
these are produced by in-house converter scripts that walk an onnx graph, so the artifact on disk has
no upstream file it corresponds to byte-for-byte.

**One component has three different stated origins.** Gemma-2-2B-IT — PiD's caption encoder and
SANA's text encoder — is described as `SceneWorks/gemma-2-2b-it` (MLX prose),
`Efficient-Large-Model/gemma-2-2b-it` (PiD tests) and "bundled in the SANA diffusers repository" (the
Candle SANA NOTICE). The canonical `google/gemma-2-2b-it` id is never written. **And the two backends
name different upstreams for the LTX text encoder**: `mlx-community/gemma-3-12b-it-bf16` (MLX) vs
`google/gemma-3-12b-it` (Candle).

## 3. U4 — settled: `nvidia-open-model` has **no shipped checkpoint**

The family should not carry a component row, because there is nothing to point at it.

- Repo-wide, "Cosmos" resolves to exactly three buckets: an **architecture-config transcription** in
  `mlx-gen-anima/src/config.rs:5` and `candle-gen-anima/src/config.rs:6` (`Cosmos-2.0-Diffusion-2B-Text2Image`);
  Rust type names (`CosmosDiT`, `CosmosRope`, `cosmos_image_rope`); and test fixtures containing the
  key strings. `candle-gen-anima/src/transformer.rs:1` is explicit: the port is from diffusers
  `transformer_cosmos.py`, and "weight keys are the **original Cosmos** names".
- The weights actually loaded are `circlestone-labs/Anima`'s single-file DiT, which *bundles* the
  Cosmos-topology transformer together with the `AnimaTextConditioner`
  (`candle-gen-anima/src/loader.rs:4`). Anima is governed by the CircleStone licence, family
  `circlestone-labs-non-commercial`.
- There are **zero** hits repo-wide for `nvidia/Cosmos`, a Cosmos-Predict2 repository id, or any
  Cosmos snapshot / `WeightsSource` path.
- The only NVIDIA checkpoint that ships is `nvidia/PiD`, which is **NSCLv1**, family
  `nvidia-nsclv1` — a different text, correctly kept as a separate family by sc-16662.

**Recommendation for the sc-16662 sign-off:** close U4 as *no shipped checkpoint*.
`nvidia-open-model` is dead surface unless a Cosmos checkpoint is later shipped. Whether to keep an
unreferenced family in the table is a policy call, not a fact — but the fact is now settled.

## 4. U5 is already fixed — do not carry it forward

sc-16662's U5 (the `candle-gen-sana/NOTICE` claiming an NVIDIA licence behind a 404 URL) was
**corrected on `main` before this census ran**: commit `eef5166a` *"fix(candle-gen-sana): SANA
weights are Apache-2.0, not the NVIDIA licence [sc-16906] (#402)"*, plus `79dc20d3` which removed a
preceding false claim that the example downloads weights at runtime. The file now states Apache-2.0
for both SANA checkpoints, cites the repository's own `LICENSE`, and attributes the Gemma text
encoder to the Gemma Terms. It agrees with what the crate loads. **U5 should be closed.**

# The registered surface — verified against the epic's numbers

**Catalog inclusion is what "shipped" means** (epic R9). The authority is what
`mlx_gen_catalog::provider_registry()` and `candle_gen_catalog::provider_registry()` actually
register, pinned by each catalog's `complete_catalog_has_stable_surface` test
(`crates/media/mlx-gen/mlx-gen-catalog/src/lib.rs:378`,
`crates/media/candle-gen/candle-gen-catalog/src/lib.rs:219`). Every count below was taken from those
ordered lists at `origin/main` `5083b0ec`.

| | epic 16660 says | this census found | verdict |
| --- | --- | --- | --- |
| MLX generators | 65 | **65** | confirmed |
| Candle generators | 51 | **51** | confirmed |
| shared generator ids | 50 | **50** | confirmed |
| distinct generator ids (union) | 66 | **66** | confirmed |
| captioner | 1 | **1** — `fancyfeast/llama-joycaption-beta-one-hf-llava`, both backends | confirmed |
| embedders | 2 | **2** — `clip_vit_l14` (image) + `clip_vit_l14_text` (text), both backends | confirmed |
| trainers | *not stated* | **MLX 15, Candle 7** — 16 distinct ids, **two of which have no generator row** | **the epic omits an entire registration kind** |

The 15 MLX-only generator ids: `flux1_dev_control`, `flux2_dev_control`, `flux2_dev_edit`,
`flux2_klein_9b_edit`, `flux2_klein_9b_kv_edit`, `krea_2_turbo_control`, `krea_2_turbo_edit`,
`krea_realtime_14b`, `ltx_2_3`, `pulid_flux`, `qwen_image_control`, `qwen_image_edit`,
`wan2_2_vace_fun_14b`, `z_image_control`, `z_image_turbo_control`.
The single Candle-only generator id: `ltx_2_3_distilled`.

## Trainers are a real gap in the epic's scope

MLX trainers: `anima_base`, `anima_aesthetic`, `anima_turbo`, `kolors`, `krea_2_raw`, `lens`,
`ltx_2_3`, `mage_flow_base`, `sd3_5_large`, `sd3_5_medium`, `sdxl`, `wan2_2_t2v_14b`,
`wan2_2_i2v_14b`, `wan2_2_ti2v_5b`, `z_image_turbo`.
Candle trainers: `krea_2_raw`, `krea_2_control`, `lens`, `ltx_2_3`, `sdxl`, `wan2_2_t2v_14b`,
`z_image_turbo`.

Two Candle trainer ids are **not** Candle generator ids and therefore appear nowhere in the epic's
66: **`krea_2_control`** (trains a ControlNet branch whose overlay is applied at `krea_2_turbo`
inference — `candle-gen-krea/src/control_trainer.rs:42`) and **`ltx_2_3`** (the generator id there
is `ltx_2_3_distilled`). A trainer loads checkpoints exactly as a generator does, so if the derived
provider view is meant to answer "what did this run touch", these two ids need rows too.

## Registered ≠ implemented, in both directions

Several routes exist as working code on one backend but are deliberately kept off that backend's
catalog surface. A component-keyed table is unaffected by this — the components are the same — but
sc-16666 and sc-16667 must key their *provider* rows off the registered lists, not off what the
crates contain:

| route | MLX | Candle |
| --- | --- | --- |
| FLUX.1 ControlNet | registered `flux1_dev_control` | bespoke `Flux1DevControl`, unregistered |
| FLUX.1 / SDXL IP-Adapter | wired into the registered generator | bespoke `IpAdapterFlux` / `ip_provider`, unregistered |
| FLUX.2 edit + control | registered ×3 | bespoke `Flux2Edit` / `Flux2Control`, unregistered |
| Krea 2 control + turbo-edit | registered ×2 | `control_provider`, unregistered (but a `krea_2_control` **trainer** is registered) |
| Qwen-Image control + edit | registered ×2 | bespoke `QwenFunControl` / `QwenEdit`, unregistered |
| Z-Image control ×2 | registered ×2 | bespoke `control.rs`, unregistered |
| Kolors ControlNet + IP-Adapter | wired into the registered generator | two bespoke providers, unregistered |
| PuLID-FLUX | registered `pulid_flux` | **the whole crate is unregistered** — Candle lists `pulid` in `BESPOKE_UTILITY_CRATES` |
| InstantID | unregistered on **both**; the MLX catalog test asserts `instantid` must never gain an invented registration | unregistered |

The MLX catalog's `PENDING_REGISTRATION_CRATES` is empty, so nothing is waiting in the wings there.

# Non-registry overlay components (epic R6)

Both catalogs name their unregistered crates in a `BESPOKE_UTILITY_CRATES` const, so the set is not
guesswork:

- MLX (`mlx-gen-catalog/src/lib.rs:53`): `depth`, `face`, `instantid`, `pid`, `sam2`, `sam3`
- Candle (`candle-gen-catalog/src/lib.rs:52`): `depth`, `face`, `instantid`, `pid`, `pulid`, `sam3`

Six per backend, seven distinct crates. Every one of them loads real checkpoints and **none of them
appears in any provider-keyed view**, because none of them has a provider id. That is the sharpest
argument for a component-keyed table: today these artifacts are invisible.

| crate | registers | checkpoints it loads | which registered providers depend on it |
| --- | --- | --- | --- |
| `*-gen-pid` | nothing | **`nvidia/PiD`** — eight distinct student files, one per latent space × resolution tier (`qwenimage` 2k→4k; `flux` 2k and 2k→4k; `sd3` 2k and 2k→4k; `sdxl` 2k→4k; `flux2` 2k and 2k→4k), plus the **Gemma-2-2B-IT** caption encoder and its tokenizer | **14 crates** on each backend: boogu, chroma, flux, flux2, ideogram, instantid, kolors, krea, lens, qwen-image, sana, sdxl, z-image (+ Candle pulid). Optional per generation, via `LoadSpec::pid` |
| `*-gen-face` | nothing | **antelopev2 ArcFace `glintr100`** (`arcface_iresnet100.safetensors`), **antelopev2 SCRFD-10g** (`scrfd_10g.safetensors`), and **facexlib `parsing_bisenet`** (`bisenet_parsing.safetensors`). All three are produced by in-house converter scripts; **no upstream repository id exists anywhere in the repo** | `pulid_flux` (MLX, registered) and the instantid overlay, on both backends. Candle additionally exposes it as `gen_core::FaceEmbedder` with descriptor id `antelopev2` |
| `*-gen-instantid` | nothing — the MLX catalog test asserts it must never gain a registration | `InstantX/InstantID` (IdentityNet ControlNet + IP-Adapter), an SDXL base (prose says "stock SDXL (RealVisXL)"; the concrete `SG161222/RealVisXL_V5.0` id appears only in the manifest and a Candle validator), optional `xinsir/controlnet-openpose-sdxl-1.0`, plus the whole face stack. Candle additionally requires `madebyollin/sdxl-vae-fp16-fix` | **none** — worker-invoked directly |
| `*-gen-depth` | nothing | `depth-anything/Depth-Anything-V2-Small-hf` (one `model.safetensors`; Base/Large are supported but Small is the shipped default) | **none** — a preprocessor consumed through its own API |
| `mlx-gen-sam2` (MLX only) | nothing | `facebook/sam2.1-hiera-large` (default) and `facebook/sam2.1-hiera-base-plus` (speed variant), converted in-house and mirrored at `SceneWorks/sam2-mlx` | **none in this repository** — a repo-wide grep for `mlx_gen_sam2::` outside the crate returns zero hits; its consumer is the external SceneWorks worker |
| `*-gen-sam3` | nothing | `facebook/sam3` — a single checkpoint carrying the PE ViT backbone, a CLIP-H text tower + projection, the CLIP BPE tokenizer, the DETR detector, the mask head, the geometry/exemplar encoder and a SAM2.1-style tracker. Candle loads it **raw**, with no conversion | **none in this repository** — same as sam2 |
| `candle-gen-pulid` (Candle only) | nothing | FLUX.1-dev base, `guozinan/PuLID`, `QuanSun/EVA-CLIP` EVA02-CLIP-L-14-336, the face stack, and — unlike MLX — the PiD decoder | **none** on Candle. The MLX twin **is** registered as `pulid_flux` |

## The epic's R6 claim, corrected

The epic states that insightface antelopev2 (ArcFace glintr100 + SCRFD) is a hard dependency of
`mlx-gen-face`, on which PuLID-FLUX and InstantID depend. **That is correct**, and this census adds
three things to it:

1. **The face stack is three checkpoints, not two.** The third, `bisenet_parsing`, is a port of
   **facexlib**'s `parsing_bisenet.pth` — a different upstream from insightface entirely. It is used
   on the PuLID path to whiten non-face regions before the EVA-CLIP crop
   (`mlx-gen-face/src/bisenet.rs:1`). `insightface-research-only` does **not** reach it, so it is
   uncovered.
2. **PuLID loads a fourth uncovered artifact**, the EVA02-CLIP-L-14-336 visual tower, whose upstream
   (`QuanSun/EVA-CLIP` → `EVA02_CLIP_L_336_psz14_s6B.pt`) is named **only inside a converter
   script**, `crates/media/mlx-gen/tools/convert_eva_clip.py:5`, and in neither provider crate.
3. **PiD, not face, is the widest overlay.** It reaches fourteen crates per backend and carries a
   non-commercial NVIDIA licence plus a Gemma text encoder. `mlx-gen-pid/src/lib.rs:23` states
   *"PiD weights are NVIDIA NSCLv1 (non-commercial). The NC restriction flows to PiD-decoded
   output"* — and **not one of the fourteen consuming crates repeats it**. That is exactly the
   invisibility the component table is meant to remove: today, whether a render touched PiD is
   knowable only from the `LoadSpec`, and nothing derives a disclosure from it.

# Distinct upstream checkpoint index

One row per **named upstream repository whose weights are loaded**. Where one repository supplies
several separately-trained weight files (Anima's three DiTs, Wan's expert pairs, PiD's per-latent
students), the row says so — the licence is a property of the repository, so that is the natural
granularity for a component row, but sc-16666/16667 may split further if a consumer needs per-file
attribution.

`family?` records **coverage only** — whether one of the sixteen landed families is *anchored* to
this licensor or model (`anchored`), whether a generic family would plausibly serve but nothing
in-repo confirms it (`generic?`), or whether neither is true (`UNCOVERED`). **No family is assigned
here and no licence is asserted.** The `declared` column of a real component row must come from
reading the upstream card, which this story did not do.

## Generator / captioner / embedder bases

| # | upstream repository | used by | notes | family? |
| --- | --- | --- | --- | --- |
| 1 | `circlestone-labs/Anima` | `anima_base`, `anima_aesthetic`, `anima_turbo` | three separately-trained DiT files in one repo | anchored (`circlestone-labs-non-commercial`) |
| 2 | `circlestone-labs/Anima-Official-LoRAs` | anima (optional) | | anchored |
| 3 | `ByteDance/Bernini-Diffusers` | `bernini`, `bernini_renderer` | one combined index supplying 6 prefixes (2 DiT experts, planner, connector, vit_decoder, mask tokens) | **UNCOVERED** |
| 4 | `Boogu/Boogu-Image-0.1-Base` | `boogu_image` | stated only in a tools README | **UNCOVERED** |
| 5 | `Boogu/Boogu-Image-0.1-Turbo` | `boogu_image_turbo` | same | **UNCOVERED** |
| 6 | Boogu Edit checkpoint | `boogu_image_edit` | **repo UNDETERMINED** | **UNCOVERED** |
| 7 | `lodestones/Chroma1-HD` | `chroma1_hd` | | **UNCOVERED** |
| 8 | `lodestones/Chroma1-Base` | `chroma1_base` | | **UNCOVERED** |
| 9 | `lodestones/Chroma1-Flash` | `chroma1_flash` | | **UNCOVERED** |
| 10 | `black-forest-labs/FLUX.1-schnell` | `flux1_schnell` | Candle rustdoc says Apache-2.0; unconfirmed | generic? (`apache-2-0`) |
| 11 | `black-forest-labs/FLUX.1-dev` | `flux1_dev`, `flux1_dev_control`, `pulid_flux`, mochi/chroma VAE lineage | pinned rev `3de623fc…`, `license = "FLUX-1-dev Non-Commercial License"` | anchored (`flux-1-dev-non-commercial`) |
| 12 | `black-forest-labs/FLUX.2-klein-9B` | `flux2_klein_9b`, `flux2_klein_9b_edit` | | **UNCOVERED** |
| 13 | `black-forest-labs/FLUX.2-klein-9b-kv` | `flux2_klein_9b_kv_edit` (MLX only) | separately distilled | **UNCOVERED** |
| 14 | `black-forest-labs/FLUX.2-dev` | `flux2_dev`, `flux2_dev_edit`, `flux2_dev_control` | | **UNCOVERED** |
| 15 | `ideogram-ai/ideogram-4-fp8` | `ideogram_4`, `ideogram_4_turbo` | stated **gated**; supplies conditional DiT, unconditional DiT, TE, VAE, turbo LoRA | **UNCOVERED** |
| 16 | `fancyfeast/llama-joycaption-beta-one-hf-llava` | the captioner id | pinned rev `ebf414ea…`, `license = "Llama 3.1 Community License"` | anchored (`llama-3-1-community`) |
| 17 | `Kwai-Kolors/Kolors-diffusers` | `kolors` | UNet + VAE + the ChatGLM3-6B TE (row 85) | **UNCOVERED** at repo level (sc-16662 U6) |
| 18 | `krea/Krea-2-Turbo` | `krea_2_turbo`, `krea_2_turbo_edit`, `krea_2_turbo_control` | | anchored (`krea-2-community`) |
| 19 | `krea/Krea-2-Raw` | `krea_2_raw`, `krea_2_edit` | | anchored |
| 20 | `krea/krea-realtime-video` | `krea_realtime_14b` | re-hosted `SceneWorks/krea-realtime-14b-mlx` rev `e68e9a3d…`, `license = "Apache-2.0"` | generic? (`apache-2-0`) |
| 21 | `microsoft/Lens` | `lens` | the crate records that Microsoft **pulled this repository**; the training base is now the `SceneWorks/Lens` rehost | **UNCOVERED** |
| 22 | `microsoft/Lens-Turbo` | `lens_turbo` | | **UNCOVERED** |
| 23 | `Lightricks/LTX-2.3` | `ltx_2_3`, `ltx_2_3_distilled` | one bundled file = DiT + video VAE + audio VAE + vocoder + connector; plus 3 separate upsampler files | anchored (`ltx-2-community`) |
| 24 | `microsoft/Mage-Flow` | `mage_flow` | pinned, `license = "MIT"` | anchored (`mit`) |
| 25 | `microsoft/Mage-Flow-Base` | `mage_flow_base` (+ MLX trainer) | **not pinned** | anchored (`mit`) by sibling |
| 26 | `microsoft/Mage-Flow-Turbo` | `mage_flow_turbo` | **not pinned** | anchored by sibling |
| 27 | `microsoft/Mage-Flow-Edit` | `mage_flow_edit` | pinned, MIT | anchored |
| 28 | `microsoft/Mage-Flow-Edit-Base` | `mage_flow_edit_base` | pinned, MIT | anchored |
| 29 | `microsoft/Mage-Flow-Edit-Turbo` | `mage_flow_edit_turbo` | pinned, MIT | anchored |
| 30 | `genmo/mochi-1-preview` | `mochi_1` | pinned rev `14be5fce…`, `license = "Apache-2.0"` | generic? (`apache-2-0`) |
| 31 | `Qwen/Qwen-Image` | `qwen_image` (+ the VAE anima and krea reuse) | pinned rev `75e0b4be…`, `license = "Apache-2.0"` | generic? (`apache-2-0`) |
| 32 | `Qwen/Qwen-Image-2512` | `qwen_image_control` | not pinned | generic? |
| 33 | `Qwen/Qwen-Image-Edit` (`-2511`) | `qwen_image_edit` | not pinned | generic? |
| 34 | `Efficient-Large-Model/Sana_1600M_1024px_diffusers` | `sana_1600m` | `candle-gen-sana/NOTICE` states Apache-2.0 (verified 2026-08-02) | generic? (`apache-2-0`) |
| 35 | `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` | `sana_sprint_1600m` | same NOTICE | generic? |
| 36 | `zai-org/SCAIL-2` | `scail2_14b` | loaded as the converted `SceneWorks/scail2-mlx` | **UNCOVERED** |
| 37 | `stabilityai/stable-diffusion-3.5-large` | `sd3_5_large` | | anchored (`stability-ai-community`) |
| 38 | `stabilityai/stable-diffusion-3.5-large-turbo` | `sd3_5_large_turbo` | | anchored |
| 39 | `stabilityai/stable-diffusion-3.5-medium` | `sd3_5_medium` | different architecture (MMDiT-X) | anchored |
| 40 | `stabilityai/stable-diffusion-xl-base-1.0` | `sdxl` | pinned rev `46216598…` | anchored (`creativeml-openrail-pp-m`) |
| 41 | `SG161222/RealVisXL_V5.0` | `sdxl` (alternate base), instantid | pinned rev `ac93e0dd…` | generic? (an SDXL derivative) |
| 42 | `numz/SeedVR2_comfyUI` | `seedvr2`, `seedvr2_3b`, `seedvr2_7b` | a **community re-host**; two DiT files + one shared VAE. No ByteDance repo id appears anywhere | **UNCOVERED** |
| 43 | `sensenova/SenseNova-U1-8B-MoT` | `sensenova_u1_8b`, `_fast` | unified model: no VAE, no separate TE | **UNCOVERED** |
| 44 | `sensenova/SenseNova-U1-8B-MoT-LoRAs` | `sensenova_u1_8b_fast` | 8-step distill LoRA | **UNCOVERED** |
| 45 | `stabilityai/stable-video-diffusion-img2vid-xt` | `svd_xt` | UNet + temporal-decoder VAE + ViT-H image encoder | anchored (`stability-ai-community`) |
| 46 | `Wan-AI/Wan2.2-TI2V-5B` (+ `-Diffusers`) | `wan2_2_ti2v_5b` | includes the z48 VAE | **UNCOVERED** |
| 47 | `Wan-AI/Wan2.2-T2V-A14B` (+ `-Diffusers`) | `wan2_2_t2v_14b`, and the stock UMT5/z16-VAE/tokenizer that bernini, scail2, krea-realtime and wan_vace reuse | **two expert checkpoints** | **UNCOVERED** |
| 48 | Wan2.2 I2V-A14B | `wan2_2_i2v_14b` | two expert checkpoints, in_dim 36 | **UNCOVERED** |
| 49 | `Wan-AI/Wan2.1-VACE-1.3B-diffusers` / `-14B-diffusers` | `wan_vace` | transformer only; shares the base trio | **UNCOVERED** |
| 50 | `alibaba-pai/Wan2.2-VACE-Fun-A14B` | `wan2_2_vace_fun_14b` (MLX only) | two VACE expert checkpoints | **UNCOVERED** |
| 51 | `Tongyi-MAI/Z-Image-Turbo` | `z_image_turbo`, `z_image_turbo_control` | pinned rev `f332072a…`, `license = "Apache-2.0"` | generic? (`apache-2-0`) |
| 52 | `Tongyi-MAI/Z-Image` | `z_image`, `z_image_control` | **not pinned** | generic? |

## ControlNets, adapters and identity encoders (additional checkpoints)

| # | upstream repository | used by | family? |
| --- | --- | --- | --- |
| 53 | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` | `flux1_dev_control` | **UNCOVERED** |
| 54 | `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` | `flux2_dev_control` | **UNCOVERED** |
| 55 | `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union` | `qwen_image_control` | crate rustdoc says "Apache-2.0, ungated" → generic? |
| 56 | `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` | `z_image_turbo_control` | **UNCOVERED** |
| 57 | `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` | `z_image_control` | **UNCOVERED** |
| 58 | `Kwai-Kolors/Kolors-ControlNet-Pose` | `kolors` (optional) | **UNCOVERED** |
| 59 | `Kwai-Kolors/Kolors-IP-Adapter-Plus` (+ its bundled CLIP-ViT-L/14-336 image tower) | `kolors` (optional) | **UNCOVERED** |
| 60 | `XLabs-AI/flux-ip-adapter` | `flux1_*` (optional) | **UNCOVERED** |
| 61 | `h94/IP-Adapter` | `sdxl` (optional); pinned rev `018e4027…` | **UNCOVERED** |
| 62 | `InstantX/InstantID` (IdentityNet ControlNet + IP-Adapter) | the instantid overlay crate | **UNCOVERED** |
| 63 | `xinsir/controlnet-openpose-sdxl-1.0` | instantid pose mode | **UNCOVERED** |
| 64 | `guozinan/PuLID` (`pulid_flux_v0.9.1.safetensors`, rev `492b1451…`) | `pulid_flux` | **UNCOVERED** |
| 65 | `lightx2v/Qwen-Image-Lightning` | `qwen_image` (optional) | **UNCOVERED** |
| 66 | `lightx2v/Qwen-Image-Edit-2511-Lightning` | `qwen_image_edit` (optional) | **UNCOVERED** |
| 67 | ByteDance Hyper-FLUX 8-step LoRA | `flux1_dev` (optional) — **repo UNDETERMINED** | **UNCOVERED** |
| 68 | ostris TurboTime LoRA (`turbo_lora.safetensors`) | `ideogram_4_turbo` — **repo UNDETERMINED**, bundled in-snapshot | **UNCOVERED** |
| 69 | lightx2v Wan step-distill diff-patch | `scail2_14b` (optional) — **repo UNDETERMINED** | **UNCOVERED** |
| 70 | `sat-scail2` Bias-Aware DPO LoRA | `scail2_14b` (optional) — **repo UNDETERMINED** | **UNCOVERED** |
| 71 | `SceneWorks/krea2-pose-controlnet-beta` | `krea_2_turbo_control` | trained in-house on Krea 2 → anchored (`krea-2-community`) by derivation |
| 72 | `madebyollin/sdxl-vae-fp16-fix` | Candle `sdxl` (**required** component); pinned rev `207b116d…` | **UNCOVERED** |

## Standalone auxiliary encoders, decoders and utilities

| # | upstream artifact | used by | family? |
| --- | --- | --- | --- |
| 73 | `openai/clip-vit-large-patch14` | `clip_vit_l14` + `clip_vit_l14_text` (both backends), SDXL tokenizer, FLUX IP image tower; pinned rev `32bd6428…`, `license = "MIT"` | anchored (`mit`) |
| 74 | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` | Candle SDXL bigG tokenizer; pinned rev `743c27bd…` | generic? |
| 75 | `google/umt5-xxl` (+ its tokenizer) | wan ×5, bernini, scail2, krea-realtime | generic? |
| 76 | `nvidia/PiD` | the optional latent-decoder overlay on **14 crates**; **8 distinct student files** (per latent space × resolution tier) | anchored (`nvidia-nsclv1`) |
| 77 | Gemma-2-2B-IT | the PiD caption encoder **and** the SANA text encoder. Three origins are described in-repo (`SceneWorks/gemma-2-2b-it`, `Efficient-Large-Model/gemma-2-2b-it`, "bundled in the SANA repo"); the canonical Google repo id is never written | anchored (`gemma-terms`) |
| 78 | Gemma-3-12B-IT | the LTX-2.3 text encoder. MLX names `mlx-community/gemma-3-12b-it-bf16`, Candle names `google/gemma-3-12b-it` — **the two backends name different upstreams for the same component** | anchored (`gemma-terms`) |
| 79 | `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit` | the MLX-only LTX uncensored prompt enhancer | anchored (`gemma-terms`) by derivation |
| 80 | `depth-anything/Depth-Anything-V2-Small-hf` | the depth overlay crate; the crate's own rustdoc states `apache-2.0, ungated` | generic? (`apache-2-0`) |
| 81 | antelopev2 ArcFace `glintr100` (`arcface_iresnet100.safetensors`) | the face overlay crate → pulid, instantid. **Repo UNDETERMINED** — converted locally from the onnx graph | anchored (`insightface-research-only`) |
| 82 | antelopev2 SCRFD-10g (`scrfd_10g.safetensors`) | same | anchored (`insightface-research-only`) |
| 83 | facexlib `parsing_bisenet.pth` (`bisenet_parsing.safetensors`) | same — the PuLID whitening path. **Repo UNDETERMINED**. **Not insightface**, so `insightface-research-only` does not reach it | **UNCOVERED** |
| 84 | `QuanSun/EVA-CLIP` → `EVA02_CLIP_L_336_psz14_s6B.pt` | `pulid_flux`. Named **only in the converter script**, not in either crate | **UNCOVERED** |
| 85 | ChatGLM3-6B text encoder | `kolors`. Ships inside `Kwai-Kolors/Kolors-diffusers`; **repo UNDETERMINED** — no `THUDM/chatglm3-6b` string exists in the repo | anchored (`chatglm3-model-license`) |
| 86 | `SceneWorks/kolors-chatglm3-tokenizer` | `kolors` — a **derived** fast tokenizer built in-house from ChatGLM3's `spiece.model` | anchored by derivation |
| 87 | `facebook/sam2.1-hiera-large` | the sam2 overlay crate (MLX only) | **UNCOVERED** |
| 88 | `facebook/sam2.1-hiera-base-plus` | sam2 speed variant | **UNCOVERED** |
| 89 | `facebook/sam3` | the sam3 overlay crate (both backends) — one checkpoint carrying vision encoder, CLIP text tower, DETR detector, mask head, geometry encoder and tracker | **UNCOVERED** |
| 90 | `QuantStack/Wan2.2-TI2V-5B-GGUF` | Candle `wan2_2_ti2v_5b` GGUF path — a re-host | **UNCOVERED** |

# Provider id -> components, by family

Families in alphabetical order. `depth`, `face`, `instantid`, `pid`, `sam2`, `sam3` and Candle
`pulid` are covered in the overlay section above and are not repeated here.

## anima — `mlx-gen-anima` / `candle-gen-anima`

Registered: MLX **3 generators + 3 trainers** (`anima_base`, `anima_aesthetic`, `anima_turbo`);
Candle **3 generators, no trainers**.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Anima DiT — base (`anima-base-v1.0.safetensors`) | `circlestone-labs/Anima` | `mlx-gen-anima/src/lib.rs:9`, `src/config.rs:149` |
| Anima DiT — aesthetic (`anima-aesthetic-v1.0.safetensors`) | `circlestone-labs/Anima` (separately fine-tuned weights) | `src/config.rs:150` |
| Anima DiT — turbo (`anima-turbo-v1.0.safetensors`) | `circlestone-labs/Anima` (merged CFG-free few-step student) | `src/lib.rs:7`, `src/config.rs:151` |
| `AnimaTextConditioner` | **bundled inside the DiT file** under `net.llm_adapter.*` — not a separate artifact | `src/lib.rs:14`, `src/loader.rs:5` |
| Qwen3-0.6B base text encoder | architecture named; the file loaded is `text_encoders/qwen_3_06b_base.safetensors` **inside the Anima snapshot** — origin repo **UNDETERMINED** | `src/lib.rs:18`, `src/loader.rs:32` |
| Qwen-Image `AutoencoderKLQwenImage` VAE | architecture named; file is `vae/qwen_image_vae.safetensors` **inside the Anima snapshot** — origin repo **UNDETERMINED** | `src/lib.rs:20`, `src/loader.rs:33` |
| Qwen2 BPE tokenizer (`assets/qwen_tokenizer.json`) | vendored asset — repo **UNDETERMINED** | `src/tokenizer.rs:23` |
| T5 SentencePiece tokenizer | "google-t5, vocab 32128", shared with chroma — repo **UNDETERMINED** | `src/tokenizer.rs:25` |
| Anima official LoRAs (optional, caller-supplied) | `circlestone-labs/Anima-Official-LoRAs` | `mlx-gen-anima/tests/lora_injection.rs:2` |

All three provider ids load the same encoder/VAE/tokenizer set and differ only in the DiT file.
MLX reuses z-image's Qwen3 decoder block; Candle transcribes it locally. No component-set
difference between backends.

## bernini — `mlx-gen-bernini` / `candle-gen-bernini`

Registered: both backends, **2 generators** — `bernini_renderer`, `bernini`.

`bernini_renderer` (Wan2.2-T2V-A14B architecture verbatim):

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Bernini dual-expert DiT — high noise | `ByteDance/Bernini-Diffusers`, prefix `diff_dec` | `mlx-gen-bernini/src/convert.rs:3` |
| Bernini dual-expert DiT — low noise | `ByteDance/Bernini-Diffusers`, prefix `diff_dec_low` | `mlx-gen-bernini/src/convert.rs:12` |
| UMT5-XXL text encoder | "the **stock Wan2.2** UMT5/VAE/tokenizer" — copied from a base Wan2.2 snapshot, repo **UNDETERMINED** | `src/pipeline.rs:2`, `mlx-gen-wan/src/convert.rs:864` |
| Wan z16 VAE (`AutoencoderKLWan`) | stock Wan2.2, repo **UNDETERMINED** | same |
| UMT5 tokenizer | stock Wan2.2, repo **UNDETERMINED** | same |

`bernini` adds, all from the same `ByteDance/Bernini-Diffusers` combined index:

| component | upstream | evidence |
| --- | --- | --- |
| Qwen2.5-VL-7B planner backbone + vision tower | `ByteDance/Bernini-Diffusers`, prefix `mllm` | `src/convert.rs:11`, `src/bernini.rs:185` |
| `MLPConnector` | same package, prefix `connector` | `src/convert.rs:13` |
| `DiffLoss_FM` / `SimpleMLPAdaLN` clip-diff head | same package, prefix `vit_decoder` | `src/convert.rs:12` |
| MAR mask token | same package, prefix `mask_tokens` | `src/convert.rs:14` |
| Qwen2.5-VL tokenizer | copied verbatim from the package's `mllm/` | `src/convert.rs:23` |

The Bernini package also ships a redundant UMT5 copy (prefix `t5_text_encoder`) which the converter
deliberately skips in favour of the standalone stock-Wan directory (`src/convert.rs:11`, `:56`).
MLX uses a flat file layout, Candle diffusers subdirs — same upstream, different on-disk shape.

## boogu — `mlx-gen-boogu` / `candle-gen-boogu`

Registered: both backends, **3 generators** — `boogu_image`, `boogu_image_turbo`, `boogu_image_edit`.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Boogu `BooguImageTransformer2DModel` DiT — Base | `Boogu/Boogu-Image-0.1-Base`, stated **only in a dev-tools README**, not in `src/` | `mlx-gen-boogu/tools/README.md:12` |
| Boogu DiT — Turbo | `Boogu/Boogu-Image-0.1-Turbo`, same README | `mlx-gen-boogu/tools/README.md:12` |
| Boogu DiT — Edit | **UNDETERMINED** — no upstream repo stated anywhere; loaded from a third snapshot root (`BOOGU_EDIT_DIR`) | `mlx-gen-boogu/tests/pipeline_e2e.rs:31` |
| Qwen3-VL-8B-Instruct condition encoder (text tower) | architecture named; loaded from `mllm/` inside the Boogu snapshot — origin repo **UNDETERMINED** | `mlx-gen-boogu/src/lib.rs:9`, `src/loader.rs:33` |
| Qwen3-VL vision tower (edit path) | same `mllm/` dir — **UNDETERMINED** | `src/loader.rs:44` |
| Qwen3-VL tokenizer | same `mllm/` dir — **UNDETERMINED** | `src/tokenizer.rs:25` |
| FLUX.1 16-channel `AutoencoderKL` | architecture named FLUX.1; file is `vae/` inside the Boogu snapshot — origin repo **UNDETERMINED** | `src/lib.rs:10`, `src/loader.rs:84` |
| PiD latent decoder (optional overlay) | see the overlay section | `src/model.rs:42` |

The epic's worked example (`boogu_image` = Boogu DiT + Qwen3-VL-8B + FLUX.1 VAE, three licences) is
**confirmed as the architecture**, but only the DiT's upstream repo is recorded anywhere in the repo,
and only in a tools README. The Qwen3-VL and FLUX.1-VAE components are redistributed inside the
Boogu snapshot and the code never says which upstream they were taken from.

## chroma — `mlx-gen-chroma` / `candle-gen-chroma`

Registered: both backends, **3 generators** — `chroma1_hd`, `chroma1_base`, `chroma1_flash`.
This is the **best-documented family**: the upstream repo is a per-variant constant in the config.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Chroma DiT — HD | `lodestones/Chroma1-HD` | `mlx-gen-chroma/src/config.rs:52` |
| Chroma DiT — Base | `lodestones/Chroma1-Base` | `src/config.rs:53` |
| Chroma DiT — Flash (few-step distilled) | `lodestones/Chroma1-Flash` | `src/config.rs:54` |
| T5-XXL text encoder | loaded from the Chroma repo's own `text_encoder/`; base model named `google/t5-v1.1-xxl`, origin repo **UNDETERMINED** | `src/loader.rs:43` |
| T5-XXL tokenizer | **vendored asset**, built from Chroma's own `spiece.model` | `src/loader.rs:9` |
| FLUX.1 16-ch `AutoencoderKL` | loaded from the Chroma repo's own `vae/`; MLX literally calls `mlx_gen_flux::load_vae` | `src/loader.rs:49` |
| PiD latent decoder (optional overlay) | reuses the `flux` PiD student | `src/model.rs:219` |

## clip — `mlx-gen-clip` / `candle-gen-clip`

Registered: both backends, **1 image embedder (`clip_vit_l14`) + 1 text embedder
(`clip_vit_l14_text`), no generators**. These are the epic's "2 embedders".

| component | upstream stated in code | evidence |
| --- | --- | --- |
| CLIP ViT-L/14 vision tower + `visual_projection` | `openai/clip-vit-large-patch14` | `mlx-gen-clip/src/lib.rs:4` |
| CLIP ViT-L/14 text tower + `text_projection` + tokenizer | same repo, same snapshot dir | `src/lib.rs:143` |

Both ids load **one** snapshot of **one** repo — one checkpoint, two heads. `release/real-weight-models.toml`
pins `openai/clip-vit-large-patch14` but scoped to the **SDXL tokenizer only**
(`expected_files = ["tokenizer.json"]`, `SDXL_TOKENIZER_CLIP_L_DIR`), while the embedder crates read
`CLIP_VIT_L_SNAPSHOT` and need `model.safetensors`.

## flux (FLUX.1) — `mlx-gen-flux` / `candle-gen-flux`

Registered: MLX **3 generators** (`flux1_schnell`, `flux1_dev`, `flux1_dev_control`);
Candle **2 generators** (`flux1_schnell`, `flux1_dev`). Candle keeps ControlNet and IP-Adapter as
**bespoke, unregistered** worker-invoked providers (`candle-gen-flux/src/control_provider.rs`,
`src/ip_provider.rs`).

| component | upstream stated in code | evidence |
| --- | --- | --- |
| FLUX.1-schnell MMDiT | `black-forest-labs/FLUX.1-schnell` | `mlx-gen-flux/src/config.rs:44` |
| FLUX.1-dev MMDiT | `black-forest-labs/FLUX.1-dev` (pinned rev `3de623fc…`, `release/real-weight-models.toml:64`) | `mlx-gen-flux/src/config.rs:45` |
| CLIP-L text encoder (`text_encoder/`) | Candle names the layout `openai/clip-vit-large-patch14`; MLX states no repo. Weights come from the FLUX snapshot — origin repo **UNDETERMINED** | `candle-gen-flux/src/flux1_load.rs:72`, `mlx-gen-flux/src/loader.rs:49` |
| T5-XXL text encoder (`text_encoder_2/`) | Candle names the layout `google/t5-v1_1-xxl`; weights from the FLUX snapshot — origin repo **UNDETERMINED** | `candle-gen-flux/src/packed_te.rs:266` |
| T5 tokenizer (`tokenizer_2/tokenizer.json`) | from the FLUX snapshot — **UNDETERMINED** | `mlx-gen-flux/src/loader.rs:153` |
| CLIP tokenizer | **vendored crate asset** `assets/clip_tokenizer.json`, deliberately not read from the snapshot | `mlx-gen-flux/src/loader.rs:18` |
| FLUX.1 16-ch `AutoencoderKL` | from the FLUX snapshot's `vae/` (MLX) or root `ae.safetensors` (Candle) — **UNDETERMINED** as a separate artifact; it is part of the FLUX repo | `mlx-gen-flux/src/loader.rs:113` |
| IP-Adapter (optional) | `XLabs-AI/flux-ip-adapter` | `mlx-gen-flux/src/loader.rs:92` |
| IP image tower (optional) | `openai/clip-vit-large-patch14` (vision tower) | `mlx-gen-flux/src/loader.rs:93` |
| Hyper-FLUX 8-step LoRA (optional, dev) | "ByteDance Hyper-FLUX" — repo **UNDETERMINED** | `mlx-gen-flux/src/config.rs:23` |
| `flux1_dev_control`: **additional** ControlNet | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` | `mlx-gen-flux/src/loader.rs:66` |
| PiD latent decoder (optional overlay) | see the overlay section, `flux` student | `mlx-gen-flux/src/model.rs:204` |

`flux1_dev_control` loads the dev weights **plus** a genuinely separate ControlNet checkpoint — not a
sampler variation. MLX also documents that its published `SceneWorks/flux1-dev-mlx` snapshot is
byte-identical to `black-forest-labs/FLUX.1-dev` rev `3de623fc…` (`src/preview.rs:23`).

## flux2 (FLUX.2) — `mlx-gen-flux2` / `candle-gen-flux2`

Registered: MLX **6 generators**; Candle **2** (`flux2_klein_9b`, `flux2_dev`), with edit and control
kept bespoke and unregistered.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| FLUX.2 klein-9B MMDiT | `black-forest-labs/FLUX.2-klein-9B` — serves `flux2_klein_9b` **and** `flux2_klein_9b_edit` (same checkpoint, different reference-image tokenization) | `mlx-gen-flux2/src/config.rs:97` |
| FLUX.2 klein-9b-kv MMDiT | `black-forest-labs/FLUX.2-klein-9b-kv` — **a separately distilled checkpoint**, MLX-only | `mlx-gen-flux2/src/config.rs:102` |
| FLUX.2-dev MMDiT | `black-forest-labs/FLUX.2-dev` — serves `flux2_dev` and `flux2_dev_edit` | `mlx-gen-flux2/src/config.rs:104` |
| Qwen3 dense text encoder (klein `text_encoder/`) | inside the FLUX.2-klein snapshot — **UNDETERMINED** | `mlx-gen-flux2/src/loader.rs:73` |
| Mistral3 language tower (dev `text_encoder/`) | inside the FLUX.2-dev snapshot — **UNDETERMINED** | `mlx-gen-flux2/src/loader.rs:101` |
| Pixtral vision tower + Mistral3 multimodal projector (dev only, MLX only) | inside the dev `text_encoder/` — **UNDETERMINED**. Feeds caption upsampling, not edit conditioning | `mlx-gen-flux2/src/loader.rs:237`, `src/caption_upsample.rs:1` |
| 32-channel `AutoencoderKL-Flux2` VAE | inside the FLUX.2 snapshot; explicitly **not** the FLUX.1 16-ch VAE | `mlx-gen-flux2/src/lib.rs:5`, `src/loader.rs:149` |
| tokenizers (Qwen2 template / Mistral-Pixtral template) | inside the respective snapshot — **UNDETERMINED** | `src/loader.rs:56`, `:83` |
| `flux2_dev_control`: **additional** control branch | `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` | `mlx-gen-flux2/src/model_control.rs:2` |
| PiD (optional overlay) | `flux2` student | `mlx-gen-flux2/Cargo.toml:15` |

FLUX.2 shares **no** crate-local code and no VAE with FLUX.1 (`mlx-gen-flux2/src/lib.rs:3`). The edit
variants add no checkpoint — reference images are VAE-encoded and concatenated as tokens.

## ideogram — `mlx-gen-ideogram` / `candle-gen-ideogram`

Registered: both backends, **2 generators** — `ideogram_4`, `ideogram_4_turbo`.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Conditional DiT | `ideogram-ai/ideogram-4-fp8` (stated **gated**), converted offline | `mlx-gen-ideogram/src/config.rs:20`, `src/lib.rs:7` |
| **Unconditional DiT** — a second, separately trained DiT for the asymmetric-CFG negative branch | same repo | `mlx-gen-ideogram/src/loader.rs:61` |
| Qwen3-VL-8B-Instruct text encoder | architecture named; repo **UNDETERMINED** (vision tensors present but unused) | `mlx-gen-ideogram/src/lib.rs:8`, `src/loader.rs:9` |
| Qwen3-VL tokenizer | from the Ideogram snapshot — **UNDETERMINED** | `src/loader.rs:81` |
| `AutoencoderKLFlux2`-architecture VAE | weights from the Ideogram snapshot; loaded through flux2's `Flux2Vae` | `src/loader.rs:68` |
| `ideogram_4_turbo`: bundled TurboTime LoRA (`turbo_lora.safetensors`) | "ostris TurboTime" — repo **UNDETERMINED**; only the filename, bundled inside the turbo snapshot | `src/config.rs:22`, `src/model.rs:190` |
| PiD (optional overlay) | `flux2` student | `Cargo.toml:16` |

`ideogram_4_turbo` is the **same base weights** plus the bundled LoRA and no unconditional DiT
(`src/config.rs:15`).

## joycaption — `mlx-gen-joycaption` / `candle-gen-joycaption`

Registered: both backends, **1 captioner** — id `fancyfeast/llama-joycaption-beta-one-hf-llava`
(the id *is* the upstream repo id). This is the epic's "+1 captioner".

| component | upstream stated in code | evidence |
| --- | --- | --- |
| SigLIP2 / SigLIP-so400m vision tower | inside `fancyfeast/llama-joycaption-beta-one-hf-llava` | `crates/llm/mlx-llm/src/joycaption.rs:4`, `candle-gen-joycaption/src/lib.rs:6` |
| Two-layer GELU MLP LLaVA projector | same snapshot | `crates/llm/mlx-llm/src/joycaption.rs:5` |
| Llama-3.1 8B decoder | same snapshot | `crates/llm/mlx-llm/src/joycaption.rs:8` |
| tokenizer | same snapshot | `mlx-gen-joycaption/src/model.rs:46` |

Pinned in `release/real-weight-models.toml:27` at revision `ebf414ea…` with
`license = "Llama 3.1 Community License"`. This is the only media component whose licence is already
recorded in a repository-owned data file.

## kolors — `mlx-gen-kolors` / `candle-gen-kolors`

Registered: MLX **1 generator + 1 trainer** (`kolors`); Candle **1 generator, no trainer**, with
ControlNet and IP-Adapter bespoke and unregistered.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Kolors SDXL-family U-Net (`unet/`) | `Kwai-Kolors/Kolors-diffusers` | `mlx-gen-kolors/src/registry.rs:183` |
| Kolors VAE (`vae/`) | same snapshot | `src/model.rs:259` |
| **ChatGLM3-6B text encoder** (`text_encoder/`) | architecture named; ships inside the Kolors-diffusers snapshot; no `THUDM/chatglm3-6b` string anywhere — repo **UNDETERMINED** | `mlx-gen-kolors/src/lib.rs:3`, `src/chatglm3.rs:1` |
| ChatGLM3 **fast** tokenizer | **derived offline, not upstream** — built by `tools/build_kolors_tokenizer.py` and republished as `SceneWorks/kolors-chatglm3-tokenizer` | `src/tokenizer.rs:5`, `src/convert.rs:22` |
| ControlNet-Pose (optional) — **additional** checkpoint | `Kwai-Kolors/Kolors-ControlNet-Pose` | `src/registry.rs:186` |
| IP-Adapter-Plus (optional) — **additional** checkpoint | `Kwai-Kolors/Kolors-IP-Adapter-Plus` | `src/ip_adapter.rs:2` |
| IP image tower | CLIP-ViT-L/14-336; staged inside the IP-Adapter snapshot — repo **UNDETERMINED** | `src/ip_adapter.rs:5` |
| PiD (optional overlay) | `sdxl` student | `Cargo.toml:15` |

This settles the component half of sc-16662's **U6**: the ChatGLM3-6B **text encoder** is a real,
separately-licensable component that ships inside a repository whose own card declares `apache-2.0`.
The census records both facts and assigns neither.

## krea — `mlx-gen-krea` / `candle-gen-krea`

Registered: MLX **5 generators** (`krea_2_turbo`, `krea_2_raw`, `krea_2_edit`, `krea_2_turbo_edit`,
`krea_2_turbo_control`) **+ 1 trainer** (`krea_2_raw`); Candle **3 generators**
(`krea_2_turbo`, `krea_2_raw`, `krea_2_edit`) **+ 2 trainers** (`krea_2_raw`, `krea_2_control`).

`krea_2_control` is a **Candle trainer id with no generator row** — it trains a ControlNet branch
whose overlay is applied at `krea_2_turbo` inference
(`candle-gen-krea/src/control_trainer.rs:42`). Candle's matching *inference* provider exists but is
deliberately not registered (`candle-gen-krea/src/control_provider.rs:15`).

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Krea 2 Turbo DiT (`Krea2Transformer2DModel`, TDM-distilled) | `krea/Krea-2-Turbo` | `mlx-gen-krea/src/config.rs:234` |
| Krea 2 Raw DiT (undistilled full-CFG base) | `krea/Krea-2-Raw` | `mlx-gen-krea/src/training.rs:2` |
| Qwen3-VL-4B-Instruct text encoder | ships inside the Krea snapshot's `text_encoder/` — origin repo **UNDETERMINED** | `mlx-gen-krea/src/lib.rs:20` |
| Qwen3-VL vision tower (edit path) | same `text_encoder/` dir — **UNDETERMINED** | `candle-gen-krea/src/vision.rs:5` |
| `AutoencoderKLQwenImage` VAE | **`Qwen/Qwen-Image`** — the Krea `vae/config.json` declares `_name_or_path = "Qwen/Qwen-Image"` and the weights are stated byte-identical | `mlx-gen-krea/src/vae.rs:4` |
| tokenizer | from the Krea snapshot's `tokenizer/` — **UNDETERMINED** | `src/pipeline.rs:149` |
| pose ControlNet overlay (`krea_2_turbo_control`) | **trained in-house**; the real-weight harness names `SceneWorks/krea2-pose-controlnet-beta` | `mlx-gen-krea/tests/control_branch_tier_real_weights.rs:33` |
| PiD (optional overlay) | `flux` latent tag | `Cargo.toml:20` |

`krea_2_edit` runs on the **Raw** checkpoint, `krea_2_turbo_edit` the same edit surface on the
**Turbo** checkpoint (`mlx-gen-krea/src/lib.rs:34`). Only the DiT weights differ across all
variants.

## krea-realtime — `mlx-gen-krea-realtime` (MLX only; no Candle sibling)

Registered: **1 generator** — `krea_realtime_14b`.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Krea Realtime DiT (transformer-only checkpoint) | `krea/krea-realtime-video`, rehosted as `SceneWorks/krea-realtime-14b-mlx`; stated to be **Wan 2.1 T2V 14B weight-for-weight** | `src/lib.rs:5`, `src/convert.rs:124`; pinned at `release/real-weight-models.toml:879` rev `e68e9a3d…`, `license = "Apache-2.0"` |
| UMT5-XXL text encoder | "the stock Wan components" — repo **UNDETERMINED** | `src/t2v.rs:606` |
| Wan z16 VAE | stock Wan — **UNDETERMINED** | `src/lib.rs:7` |
| UMT5 tokenizer | stock Wan — **UNDETERMINED** | `src/t2v.rs:622` |

Two test-only LoRAs are pinned in `real-weight-models.toml`: `shauray/Origami_WanLora` and
`Kijai/WanVideo_comfy`. They are fixtures, not shipped components.

## lens — `mlx-gen-lens` / `candle-gen-lens`

Registered: both backends, **2 generators** (`lens_turbo`, `lens`) **+ 1 trainer** (`lens`).

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Lens 48-layer dual-stream MMDiT (base) | `microsoft/Lens` | `mlx-gen-lens/src/schedule.rs:32` |
| Lens-Turbo MMDiT (distilled, 4 steps) | `microsoft/Lens-Turbo` | `mlx-gen-lens/src/schedule.rs:27` |
| gpt-oss-20b MoE text encoder (encoder-only) | architecture named; weights ship inside the Lens snapshot's `text_encoder/`; no `openai/gpt-oss-20b` reference anywhere — repo **UNDETERMINED** | `mlx-gen-lens/src/config.rs:3`, `src/pipeline.rs:75` |
| `AutoencoderKLFlux2` VAE | Lens's own `vae/` weights loaded into the shared flux2 VAE module | `candle-gen-lens/src/vae.rs:14` |
| o200k_harmony tokenizer | from the Lens snapshot — **UNDETERMINED** | `src/text.rs:1` |
| PromptReasoner (optional, off by default) | a **second gpt-oss copy** (+`lm_head`) in the same snapshot | `src/reasoner.rs:1` |
| PiD (optional overlay) | `flux2` student | `candle-gen-lens/src/lib.rs:56` |

The crate also records that **Microsoft pulled the original `microsoft/Lens` repository** and the
training base is now the `SceneWorks/Lens` rehost (`mlx-gen-lens/src/training.rs:993`) — a
`source_url` that no longer resolves upstream.

## ltx — `mlx-gen-ltx` / `candle-gen-ltx`

Registered: MLX **1 generator + 1 trainer**, both id `ltx_2_3`; Candle **1 generator**
`ltx_2_3_distilled` **+ 1 trainer** `ltx_2_3`. `ltx_2_3_distilled` is the **only Candle-only
generator id in the whole media lane**, and Candle's `ltx_2_3` is a trainer-only id.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| AV DiT `AVTransformer3DModel` (22B, distilled) | `Lightricks/LTX-2.3` — single bundled file | `candle-gen-ltx/src/lib.rs:129`, `mlx-gen-ltx/src/convert.rs:3` |
| Causal video VAE (encoder + decoder) | same bundled `Lightricks/LTX-2.3` file | `mlx-gen-ltx/src/convert.rs:4` |
| Audio VAE | same bundled file | `mlx-gen-ltx/src/convert.rs:5` |
| HiFi-GAN vocoder | same bundled file | `mlx-gen-ltx/src/convert.rs:6` |
| Connector + text-embedding projection | same bundled file | `mlx-gen-ltx/src/convert.rs:6` |
| Spatial x2 / x1.5 and temporal x2 latent upsamplers (MLX render path only) | `Lightricks/LTX-2.3`, named by filename | `mlx-gen-ltx/src/convert.rs:70` |
| **Gemma-3-12B-IT text encoder — a separate snapshot** | MLX names `mlx-community/gemma-3-12b-it-bf16`; Candle names `google/gemma-3-12b-it`. **The two backends name different upstreams for the same component.** | `mlx-gen-ltx/src/model.rs:18`, `candle-gen-ltx/src/lib.rs:589` |
| Uncensored prompt enhancer (MLX only, optional) | `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit` | `mlx-gen-ltx/src/model.rs:351` |

The deleted `LTX_GEMMA_DIR` env side channel pointed at the Gemma snapshot. The **component is still
loaded** — it now arrives through the required `LoadSpec::text_encoder` slot, with a negative test
asserting the env var cannot resurrect it (`candle-gen-ltx/src/lib.rs:788`).

## mage — `mlx-gen-mage` / `candle-gen-mage`

Registered: both backends, **6 generators**; MLX adds **1 trainer** (`mage_flow_base`), Candle none.
Six upstream repositories, one architecture — the most fully-stated family in the lane
(`mlx-gen-mage/src/model.rs:12`):

| provider id | DiT checkpoint |
| --- | --- |
| `mage_flow` | `microsoft/Mage-Flow` |
| `mage_flow_base` | `microsoft/Mage-Flow-Base` |
| `mage_flow_turbo` | `microsoft/Mage-Flow-Turbo` |
| `mage_flow_edit` | `microsoft/Mage-Flow-Edit` |
| `mage_flow_edit_base` | `microsoft/Mage-Flow-Edit-Base` |
| `mage_flow_edit_turbo` | `microsoft/Mage-Flow-Edit-Turbo` |

Shared by all six:

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Qwen3-VL-4B text encoder | ships in every `microsoft/Mage-Flow*` repo's `text_encoder/`, **bit-identical across all six** — origin repo **UNDETERMINED** | `mlx-gen-mage/src/text_encoder/load.rs:13` |
| Qwen2 tokenizer | same `text_encoder/` dir — **UNDETERMINED** | `src/text_encoder/load.rs:9` |
| Qwen3-VL vision tower (edit path) | same dir — **UNDETERMINED** | `src/text_encoder/load.rs:16` |
| Mage-VAE one-step 128-ch codec | the same six repos' `vae/` | `src/vae/mod.rs:1` |

Four of the six are pinned with revisions and `license = "MIT"` in `release/real-weight-models.toml`
(`Mage-Flow`, `-Edit`, `-Edit-Base`, `-Edit-Turbo`); **`Mage-Flow-Base` and `Mage-Flow-Turbo` are
registered providers with no pin**, and `Mage-Flow-Base` is the MLX trainer target.

`crates/media/mlx-gen/_vendor/mage_flow/` is **source code, not weights** — a verbatim MIT-licensed
copy of `github.com/microsoft/Mage` at `df7f84d9…`, used as a dev-only parity oracle, recorded in
`mlx-gen/NOTICE` and `_vendor/VENDORED.md`. No checkpoint is vendored anywhere in the repository.

## mochi — `mlx-gen-mochi` / `candle-gen-mochi`

Registered: both backends, **1 generator** — `mochi_1`.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| AsymmDiT MMDiT (10B) | `genmo/mochi-1-preview` (pinned rev `14be5fce…`, `license = "Apache-2.0"`, `release/real-weight-models.toml:448`) | `mlx-gen-mochi/src/lib.rs:3` |
| T5-XXL text encoder | architecture identified as `google/t5-v1.1-xxl`; weights load from the Mochi snapshot's `text_encoder/` — origin repo **UNDETERMINED** | `mlx-gen-mochi/src/config.rs:10` |
| T5-XXL fast tokenizer | **vendored crate asset** `assets/t5_tokenizer.json`, byte-identical to Chroma's | `src/tokenizer.rs:1` |
| AsymmVAE (`AutoencoderKLMochi`), decode-only | Mochi snapshot's `vae/` | `src/vae.rs:1` |

No adapters, ControlNets, refiners or safety models.

## pulid — `mlx-gen-pulid` (registered) / `candle-gen-pulid` (**bespoke, unregistered**)

Registered: MLX **1 generator** (`pulid_flux`). Candle registers **nothing** — `PulidFlux` is a plain
struct driven directly by the worker (`candle-gen-pulid/src/lib.rs:17`), which is why Candle lists
`pulid` in `BESPOKE_UTILITY_CRATES`. So `pulid_flux` is an MLX-only shipped id whose Candle twin
exists as code but is not on the catalog surface.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| FLUX.1-dev backbone (DiT + VAE + CLIP-L + T5) | `FluxVariant::Dev`; the repo id `black-forest-labs/FLUX.1-dev` is written in the **Candle** crate (`candle-gen-pulid/src/pulid_flux.rs:87`) but **not** in the MLX one | `mlx-gen-pulid/src/pulid_flux.rs:445` |
| PuLID identity encoder — IDFormer + 20 `PerceiverAttentionCA` (`pulid_flux_v0.9.1.safetensors`) | `guozinan/PuLID` — stated in Candle production code and in MLX tests only (`mlx-gen-catalog/tests/preview_real_weights.rs:25` pins rev `492b1451…`) | `candle-gen-pulid/src/pulid_flux.rs:93` |
| EVA02-CLIP-L-14-336 visual tower | `QuanSun/EVA-CLIP` → `EVA02_CLIP_L_336_psz14_s6B.pt` — **stated only in the converter script** `mlx-gen/tools/convert_eva_clip.py:5`, not in either crate | `mlx-gen-pulid/src/pulid_flux.rs:403` |
| SCRFD detector, ArcFace iresnet100, BiSeNet parser | via the face overlay crate — see below | `mlx-gen-pulid/Cargo.toml:14` |
| PiD latent decoder | **Candle only** — `candle-gen-pulid/Cargo.toml:27`; MLX PuLID has no PiD edge | |

## qwen-image — `mlx-gen-qwen-image` / `candle-gen-qwen-image`

Registered: MLX **3 generators** (`qwen_image`, `qwen_image_control`, `qwen_image_edit`); Candle
**1** (`qwen_image`), with control and edit kept bespoke and unregistered.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Qwen-Image 60-layer MMDiT | `Qwen/Qwen-Image` (pinned rev `75e0b4be…`, `license = "Apache-2.0"`, `release/real-weight-models.toml:45`) | `mlx-gen-qwen-image/src/loader.rs:1` |
| Qwen2.5-VL text encoder | inside the Qwen-Image snapshot's `text_encoder/` — origin repo **UNDETERMINED** | `src/loader.rs:61` |
| causal-Conv3d `AutoencoderKLQwenImage` VAE | same snapshot's `vae/` | `src/loader.rs:148` |
| Qwen2 BPE tokenizer | the upstream repo ships only `vocab.json` + `merges.txt`; the fast `tokenizer.json` is **materialized in-house** by `tools/build_qwen_tokenizer.py`, republished as `SceneWorks/qwen-image-tokenizer` | `src/loader.rs:37`, `candle-gen-qwen-image/src/config.rs:73` |
| Lightning distill LoRA (optional) | `lightx2v/Qwen-Image-Lightning` | `src/sampler.rs:19` |
| `qwen_image_control`: **different base** `Qwen/Qwen-Image-2512` **plus an additional** control branch `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union` | both stated | `src/model_control.rs:158`, `:1` |
| `qwen_image_edit`: **different base** `Qwen/Qwen-Image-Edit` (validated reference `-2511`) **plus** the Qwen2.5-VL vision transformer and the Qwen2-VL image processor | stated | `src/model_edit.rs:4`, `src/loader.rs:68` |
| PiD (optional overlay) | `qwenimage` student | `src/pipeline.rs:272` |

The two MLX variant ids are the clearest case in the lane of a provider id that loads a **different
base checkpoint**, not just a different sampler — a distinction a provider-keyed table can express
but a family-keyed one cannot.

## sana — `mlx-gen-sana` / `candle-gen-sana`

Registered: both backends, **2 generators** — `sana_1600m`, `sana_sprint_1600m`.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| SANA Linear-DiT trunk | `Efficient-Large-Model/Sana_1600M_1024px_diffusers` | `mlx-gen-sana/src/transformer.rs:4` |
| SANA-Sprint Linear-DiT trunk (a distinct distilled checkpoint) | `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` | `mlx-gen-sana/src/config.rs:161` |
| DC-AE f32c32 autoencoder (Base) and its Sprint counterpart | port target `mit-han-lab/dc-ae-f32c32-sana-1.0`; the **tensors load from each SANA repo's own `vae/`**, and MLX records that Base and Sprint ship *different* DC-AE revisions (1.0 vs 1.1) | `mlx-gen-sana/src/lib.rs:9`, `src/preview.rs:66` |
| Gemma-2-2B-IT caption encoder | model named; three different origins are described in-repo — `SceneWorks/gemma-2-2b-it` (MLX prose), `Efficient-Large-Model/gemma-2-2b-it` (PiD tests), and "bundled in the SANA diffusers repository" (the NOTICE). Canonical repo **UNDETERMINED** | `mlx-gen-sana/src/model.rs:19`, `candle-gen-sana/NOTICE` |
| Gemma tokenizer | co-located with the Gemma weights | `src/text_encoder.rs:69` |

**The sc-16662 U5 defect is already fixed.** `candle-gen-sana/NOTICE` no longer claims an NVIDIA
licence: commit `eef5166a` ("SANA weights are Apache-2.0, not the NVIDIA licence [sc-16906]",
2026-08-02) replaced the claim and the 404 URL with an Apache-2.0 statement for both checkpoints, and
`79dc20d3` removed a preceding false "downloads model weights at runtime" claim. The file now agrees
with what the crate loads. U5 should be closed rather than carried forward.

## scail2 — `mlx-gen-scail2` / `candle-gen-scail2`

Registered: both backends, **1 generator** — `scail2_14b`. A character-animation / motion-transfer
model on a Wan2.1-14B-I2V backbone.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| SCAIL-2 DiT | `zai-org/SCAIL-2` is stated as the config source of truth; the loaded artifact is the converted `SceneWorks/scail2-mlx` snapshot | `mlx-gen-scail2/src/config.rs:11`, `src/pipeline.rs:120` |
| Wan2.1 z16 VAE | "stock Wan2.1 VAE" — repo **UNDETERMINED** | `src/lib.rs:18` |
| UMT5-XXL text encoder | named; repo **UNDETERMINED** | `src/pipeline.rs:121` |
| open-CLIP XLM-RoBERTa ViT-H/14 visual tower | architecture named; repo **UNDETERMINED** | `src/clip.rs:1` |
| tokenizer | from the snapshot — **UNDETERMINED** | `src/generate.rs:421` |
| lightx2v step-distill diff-patch (optional) | vendor named; repo **UNDETERMINED**. Targets vanilla Wan2.1-I2V-14B, so some tensors are deliberately skipped | `src/lora.rs:6` |
| Bias-Aware DPO refinement LoRA `sat-scail2` (optional) | named; repo **UNDETERMINED** | `candle-gen-scail2/src/adapters.rs:9` |

## sd3 — `mlx-gen-sd3` / `candle-gen-sd3`

Registered: both backends **3 generators** (`sd3_5_large`, `sd3_5_large_turbo`, `sd3_5_medium`);
MLX adds **2 trainers** (`sd3_5_large`, `sd3_5_medium`), Candle none.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| SD3.5 Large MMDiT | `stabilityai/stable-diffusion-3.5-large` | `mlx-gen-sd3/src/config.rs:154` |
| SD3.5 Large-Turbo MMDiT (ADD-distilled, same architecture) | `stabilityai/stable-diffusion-3.5-large-turbo` | `src/config.rs:155` |
| SD3.5 Medium MMDiT-X (a **different** architecture) | `stabilityai/stable-diffusion-3.5-medium` | `src/config.rs:156` |
| CLIP-L text encoder (`text_encoder/`) | in-snapshot; the canonical repo `openai/clip-vit-large-patch14` is named only in the Candle tokenizer-parity harness | `mlx-gen-sd3/src/text.rs:6`, `candle-gen-sd3/src/clip_tokenizer.rs:4` |
| OpenCLIP-bigG text encoder (`text_encoder_2/`) | in-snapshot; canonical repo `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` named only in the same harness | `candle-gen-sd3/src/clip_tokenizer.rs:5` |
| T5-XXL text encoder (`text_encoder_3/`) | in-snapshot; "identical to FLUX's T5" — standalone repo **UNDETERMINED** | `mlx-gen-sd3/src/text.rs:11` |
| two CLIP BPE tokenizers + one T5 tokenizer | in-snapshot | `src/loader.rs:180`, `:186` |
| 16-ch `AutoencoderKL` | in-snapshot `vae/`; module reused from z-image, whose VAE the code says derives from FLUX's | `mlx-gen-sd3/src/vae.rs:6` |

## sdxl — `mlx-gen-sdxl` / `candle-gen-sdxl`

Registered: both backends, **1 generator + 1 trainer**, id `sdxl`. The same id serves the
`SG161222/RealVisXL_V5.0` base and the Lightning tiers.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| SDXL `UNet2DConditionModel` | `stabilityai/stable-diffusion-xl-base-1.0` (pinned rev `46216598…`, `release/real-weight-models.toml:702`) | `mlx-gen-sdxl/src/loader.rs:1` |
| alternate base | `SG161222/RealVisXL_V5.0` (pinned rev `ac93e0dd…`) | `release/real-weight-models.toml:762` |
| CLIP-L text encoder, OpenCLIP-bigG text encoder | in-snapshot on MLX | `src/loader.rs:9`, `:10` |
| CLIP-L tokenizer (**Candle: caller-staged component `tokenizer_clip_l`**) | `openai/clip-vit-large-patch14` (pinned rev `32bd6428…`) | `candle-gen-sdxl/src/pipeline.rs:170` |
| CLIP-bigG tokenizer (**Candle: `tokenizer_clip_bigg`**) | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` (pinned rev `743c27bd…`) | same |
| VAE | **backends differ**: MLX loads the in-snapshot `vae/` at f32; **Candle requires `madebyollin/sdxl-vae-fp16-fix`** as a staged component because the base VAE NaNs in f16 | `mlx-gen-sdxl/src/model.rs:295`, `candle-gen-sdxl/src/pipeline.rs:164` |
| ControlNet branches (optional) | "a diffusers `ControlNetModel` checkpoint" — repo **UNDETERMINED**; caller-supplied, which is why the catalog test lists `sdxl` in `NO_BRANCH_POLICY` | `mlx-gen-sdxl/src/loader.rs:177` |
| IP-Adapter (optional) — CLIP ViT-H/14 image encoder + Resampler + decoupled K/V | `h94/IP-Adapter` (pinned rev `018e4027…`) | `mlx-gen-sdxl/src/loader.rs:196` |
| PiD (optional overlay) | `sdxl` student | `src/model.rs:121` |

The VAE divergence is a genuine **component-set difference between backends for one provider id** —
exactly the shape sc-16666 and sc-16667 must be free to express independently.

## seedvr2 — `mlx-gen-seedvr2` / `candle-gen-seedvr2`

Registered: both backends, **3 generators** — `seedvr2`, `seedvr2_3b`, `seedvr2_7b`.
`seedvr2` (bare) resolves to the 3B checkpoint (`mlx-gen-seedvr2/src/registry.rs:38`).

| component | upstream stated in code | evidence |
| --- | --- | --- |
| SeedVR2 3B DiT (`seedvr2_ema_3b_fp16.safetensors`) | `numz/SeedVR2_comfyUI` — a **community re-host**; no ByteDance repo id appears anywhere | `mlx-gen-seedvr2/src/registry.rs:11` |
| SeedVR2 7B DiT (`seedvr2_ema_7b_fp16.safetensors`) — a distinct checkpoint | same re-host | `src/registry.rs:34` |
| 3D causal video VAE (`ema_vae_fp16.safetensors`), shared 3B↔7B | same re-host | `src/pipeline.rs:117` |
| precomputed negative-prompt embedding `(1, 58, 5120)` | **compiled into the binary** via `include_bytes!("../data/neg_embed.safetensors")`. Derivation source **UNDETERMINED** | `src/pipeline.rs:75` |

No text encoder, no tokenizer, no separate upscaler network.

## sensenova — `mlx-gen-sensenova` / `candle-gen-sensenova`

Registered: both backends, **2 generators** — `sensenova_u1_8b`, `sensenova_u1_8b_fast`.
A *unified* multimodal model: one network does understanding and generation, so it has **no VAE and
no separate text encoder**.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Dual-path Qwen3 "Mixture-of-Transformers" backbone (42 layers) | `sensenova/SenseNova-U1-8B-MoT` | `mlx-gen-sensenova/src/lib.rs:6`, `src/model.rs:162` |
| vision encoder | **not a separate checkpoint** — a Conv patch-embedder inside the same weights | `src/lib.rs:24` |
| tokenizer | the snapshot ships only `vocab.json` + `merges.txt`; the fast `tokenizer.json` is materialized in-house or borrowed from a sibling quant tier | `src/text.rs:4` |
| `sensenova_u1_8b_fast`: 8-step distill LoRA | `sensenova/SenseNova-U1-8B-MoT-LoRAs` → `SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors` | `src/distill.rs:39` |

The two ids share **the same base weights**; `_fast` merges the LoRA before quantization.

## svd — `mlx-gen-svd` / `candle-gen-svd`

Registered: both backends, **1 generator** — `svd_xt`. Image-conditioned, no prompt encoder.

| component | upstream stated in code | evidence |
| --- | --- | --- |
| `UNetSpatioTemporalConditionModel` | `stabilityai/stable-video-diffusion-img2vid-xt` | `mlx-gen-svd/src/lib.rs:4` |
| `AutoencoderKLTemporalDecoder` | same snapshot's `vae/` | `src/vae.rs:1` |
| CLIP ViT-H/14 image encoder (`CLIPVisionModelWithProjection`, 1280→1024) | same snapshot's `image_encoder/` | `src/image_encoder.rs:1` |

## wan — `mlx-gen-wan` / `candle-gen-wan`

Registered: MLX **5 generators + 3 trainers**; Candle **4 generators + 1 trainer** (no
`wan2_2_vace_fun_14b`).

| provider id | primary checkpoint(s) stated in code | shared auxiliaries |
| --- | --- | --- |
| `wan2_2_ti2v_5b` | `Wan-AI/Wan2.2-TI2V-5B` (MLX native) / `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (Candle) | UMT5-XXL + tokenizer, **z48** Wan2.2 VAE |
| `wan2_2_t2v_14b` | `Wan-AI/Wan2.2-T2V-A14B(-Diffusers)` — **two expert checkpoints** (high-noise `transformer/`, low-noise `transformer_2/`) | UMT5-XXL + tokenizer, **z16** Wan2.1 VAE |
| `wan2_2_i2v_14b` | Wan2.2 I2V-A14B (in_dim 36) — **two expert checkpoints**; a distinct checkpoint from T2V | same; **no CLIP image encoder** — the image enters by channel concat |
| `wan_vace` | `Wan-AI/Wan2.1-VACE-1.3B-diffusers` / `-14B-diffusers` (config-driven, not hardcoded) | shares the base Wan UMT5 + z16 VAE + tokenizer; only the transformer is VACE-specific |
| `wan2_2_vace_fun_14b` (MLX only) | `alibaba-pai/Wan2.2-VACE-Fun-A14B` — **two VACE expert checkpoints** | same shared trio |

UMT5-XXL is stated as `google/umt5-xxl` (`candle-gen-wan/src/text_encoder.rs:1`) and the tokenizer as
`google/umt5-xxl/tokenizer.json` (`mlx-gen-wan/src/convert.rs:257`) — one of the few auxiliary
components in the lane whose upstream repo is actually written down. `QuantStack/Wan2.2-TI2V-5B-GGUF`
is a Candle-side GGUF re-host of the 5B DiT.

## z-image — `mlx-gen-z-image` / `candle-gen-z-image`

Registered: MLX **4 generators + 1 trainer** (`z_image`, `z_image_control`, `z_image_turbo`,
`z_image_turbo_control`; trainer `z_image_turbo`); Candle **2 generators + 1 trainer** — the control
providers exist as code but are bespoke and unregistered (`candle-gen-z-image/src/lib.rs:52`).

| component | upstream stated in code | evidence |
| --- | --- | --- |
| Z-Image-Turbo DiT (guidance-distilled) | `Tongyi-MAI/Z-Image-Turbo` (pinned rev `f332072a…`, `license = "Apache-2.0"`, `release/real-weight-models.toml:86`) | `mlx-gen-z-image/src/model.rs:4` |
| Z-Image base DiT (undistilled, identical architecture) | `Tongyi-MAI/Z-Image` — **not pinned** in the manifest | `src/model_base.rs:22` |
| Qwen3-style text encoder (hidden 2560, 36 layers) | in-snapshot `text_encoder/` — standalone Qwen3 repo **UNDETERMINED** | `src/text_encoder/mod.rs:1` |
| Qwen tokenizer | in-snapshot | `src/loader.rs:27` |
| 16-ch `AutoencoderKL` | in-snapshot `vae/`; the code states **"Z-Image ships Flux1-dev's 16-ch VAE"**, which is why the PiD `zimage` alias resolves to the `flux` student | `src/model.rs:58`, `mlx-gen-pid/src/registry.rs:7` |
| `z_image_turbo_control`: **additional** control checkpoint | `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` | `src/model_control.rs:2` |
| `z_image_control`: **additional** control checkpoint | `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` | `src/model_base_control.rs:3` |
| PiD (optional overlay) | `flux` student via the `zimage`/`zimage-turbo` alias | `src/model.rs:58` |

# Doc / NOTICE claims that look inconsistent with what the code loads

sc-16662 found one (`candle-gen-sana/NOTICE`); it is now fixed. These are the ones this census
found. **None is a licence finding** — each is a statement in the repository that does not match the
repository's own code, which is the class of defect the table exists to prevent.

| # | claim | where | why it looks wrong |
| --- | --- | --- | --- |
| 1 | "Apache-2.0, ungated" asserted for the Boogu weights | `candle-gen-boogu/src/lib.rs:23` | Nothing in-repo substantiates it. The only upstream pointer for Boogu is a tools README that states no licence, there is no manifest pin, and the MLX sibling makes no licence claim at all. A licence asserted in one backend's rustdoc and nowhere else is exactly the shape of the SANA defect |
| 2 | Boogu Turbo is "the same Base weights-arch" | `candle-gen-boogu/src/lib.rs:10` | MLX describes Base/Turbo/Edit as three **snapshots** (`mlx-gen-boogu/src/model.rs:9`), the tests point at three separate roots, and the tools README names two different upstream repositories. If Turbo is a distinct checkpoint it needs its own component row; the two backends currently disagree |
| 3 | `microsoft/Lens` named as the load source in five places | `mlx-gen-lens/src/registry.rs:268`, `candle-gen-lens/src/lib.rs:1549`, `schedule.rs:27,32`, `config.rs:3` | The same crate records that **Microsoft pulled the original repository** (`mlx-gen-lens/src/training.rs:993`). A `source_url` field requires a document that can be re-read; these point at a repository the repo itself says is gone |
| 4 | The two backends name **different upstreams for the same LTX text encoder** | `mlx-gen-ltx/src/model.rs:18` (`mlx-community/gemma-3-12b-it-bf16`) vs `candle-gen-ltx/src/lib.rs:589` (`google/gemma-3-12b-it`) | One is a community re-quantization of the other. Both cannot be the `source_url` of one component row; sc-16666 and sc-16667 will have to either agree or carry two rows |
| 5 | Gemma-2-2B-IT is given **three** origins | `mlx-gen-sana/src/model.rs:19`, `mlx-gen-pid/tests/caption_real.rs:23`, `candle-gen-sana/NOTICE` | `SceneWorks/gemma-2-2b-it`, `Efficient-Large-Model/gemma-2-2b-it`, and "bundled in the SANA diffusers repository". The canonical Google id is never written anywhere |
| 6 | SeedVR2 is called "the **ByteDance** … upscaler" | `mlx-gen-seedvr2/src/lib.rs:3`, `candle-gen-seedvr2/src/lib.rs:4` | The only checkpoint source named anywhere in the loader is the community re-host `numz/SeedVR2_comfyUI` (`registry.rs:11`). No ByteDance repository id appears in either crate |
| 7 | SeedVR2's bundled embedding is documented under the wrong filename | `mlx-gen-seedvr2/src/lib.rs:14` and `src/pipeline.rs:9` say `pos_emb.safetensors` | The file compiled into the binary and loaded is `data/neg_embed.safetensors` (`src/pipeline.rs:75`). No `pos_emb.safetensors` exists in the crate |
| 8 | `mlx-gen/NOTICE` closes with "Model weights are not distributed with mlx-gen" | `crates/media/mlx-gen/NOTICE` | Mostly true, but **both** seedvr2 crates `include_bytes!` a `(1, 58, 5120)` SeedVR2-derived negative-prompt embedding into every shipped binary (`mlx-gen-seedvr2/src/pipeline.rs:75`, `candle-gen-seedvr2/src/pipeline.rs:60`). It is a derived tensor, not a weight file, but it *is* redistributed and it is in no NOTICE |
| 9 | `_vendor/VENDORED.md` asserts the vendored Mage tree is "byte-for-byte upstream — no local patches" and that a documented `diff -r` "is empty" | `crates/media/mlx-gen/_vendor/VENDORED.md` | `requirements-oracles.in` and `requirements-oracles.txt` are present in the tree, were added locally by `a0dfa0b7`, and appear in neither the exclusion list nor the SHA-256 manifest. Code only — no weights are vendored anywhere in the repository |
| 10 | `mlx-gen/NOTICE` names no checkpoint upstream at all | `crates/media/mlx-gen/NOTICE` | Its scope is *source-code* lineage (MLX, mlx-rs, mflux, diffusers, Cephes) plus two dev-only fixtures. That is a correct scope for a code NOTICE — recorded here only so nobody mistakes it for a weight-attribution record. **The media lane has no weight-attribution record today**; `release/model-weight-licenses.json` currently carries audio rows only |
| 11 | PiD's non-commercial flow-through is stated in one crate and no consumer | `mlx-gen-pid/src/lib.rs:23` | Fourteen crates per backend wire the PiD decode seam; none repeats it. Note also that the statement *"the NC restriction flows to PiD-decoded output"* is a SceneWorks engineering reading, not a quote — sc-16662's **U3** deliberately declined to land `NonCommercialOutputs` on any family for exactly this reason. The two statements should not be allowed to drift apart |

# What sc-16666 and sc-16667 actually face

## The shape of the work

`ComponentLicense` (`crates/contracts/gen-core/src/license.rs:440`) needs seven fields per row:
`component`, `source_url`, `gated`, `declared`, `family`, `attribution`, `retrieved`. This census
supplies the **component identity** and, for 80 of the 90 index rows, a usable `source_url`. It supplies **none** of
`declared`, `family`, `gated` or `attribution` — those require reading upstream cards, which is
network work neither this story nor the family story did for components.

The audio lane is the precedent: 33 component rows and 18 provider rows in
`release/model-weight-licenses.json`.

## Sizing

| | audio (landed) | media (this census) |
| --- | --- | --- |
| component rows | 33 | **~90**, or ~110 if per-weight-file granularity is chosen (Anima ×3, PiD ×8, Wan expert pairs ×4, LTX upsamplers ×3) |
| provider registrations | 18 | **MLX 83** (65 generators + 15 trainers + 1 captioner + 2 embedders) and **Candle 61** (51 + 7 + 1 + 2) |
| distinct provider ids per backend | — | **MLX 68** (all 15 MLX trainer ids are also generator ids) and **Candle 56** (`krea_2_control` and `ltx_2_3` are trainer-only) |
| distinct provider ids across both | — | **70** — 66 generator ids + `krea_2_control` + the captioner + the 2 embedders (`ltx_2_3` is already an MLX generator id) |

**The transcription itself is small; the research is not.** Once a component row exists, mapping a
provider id to its component list is mechanical — most ids reuse one of about a dozen component
bundles, and the two backends load the same components for 50 shared generator ids. Realistically:

- **sc-16665** (this table): ~90 component keys, of which **47 need an upstream card read** before a
  `declared`/`family` can be written and **24 need the upstream *identified* first**. That research —
  not the Rust — is the critical path.
- **sc-16666 / sc-16667**: each is a provider→components mapping over an already-written component
  table. Per backend that is ~60–80 short lists. The genuine per-backend work is the handful of real
  divergences, and they are all listed above: SDXL's VAE (MLX in-snapshot vs Candle's required
  `madebyollin/sdxl-vae-fp16-fix`), FLUX.2's Pixtral tower and `klein-9b-kv` checkpoint (MLX only),
  LTX's Gemma naming, `wan2_2_vace_fun_14b` and the two `z_image*_control` checkpoints (MLX only),
  `pulid_flux` (MLX-registered, Candle-bespoke), and `krea_2_control` / `ltx_2_3` as Candle
  trainer-only ids.

## Three decisions the slices need taken before they start

1. **Granularity.** One row per upstream repository, or one per weight file? The licence is a
   property of the repository; the artifact a render touched is the file. The index below uses
   repository granularity and flags every multi-file repository.
2. **Redistributed components.** When Chroma ships its own `text_encoder/`, is that a T5-XXL row or a
   Chroma row? `ComponentLicense::source_url`'s own doc comment already answers this for the
   analogous audio case — the bundled `Qwen3-Embedding-0.6B` inside ACE-Step points at
   `Qwen/Qwen3-Embedding-0.6B`, *"where `apache-2.0` is the value actually published"*. Applying that
   rule to media requires identifying 16 upstreams the code does not name.
3. **Trainers.** In or out? They load the same checkpoints and the epic's scope statement does not
   mention them.
