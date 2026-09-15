# Checkpoint licence evidence — the media lane's 90 upstream artifacts (sc-16665)

| | |
| --- | --- |
| Story | sc-16665 — shared checkpoint licence table. **This is the research half**; a later pass writes the Rust |
| Epic | 16660 |
| Builds on | [Media checkpoint census (sc-16665)](sc-16665-media-checkpoint-census.md) — the 90 artifacts, read from the code |
| Measured against | [Licence family evidence pack (sc-16662)](sc-16662-licence-family-evidence.md) — the 16 landed families |
| Gathered by | Claude (Opus 5), automated agent, on behalf of Michael Trefry |
| Retrieval date for **every** quote and every `gated` value below | **2026-08-02** |
| Method | Hugging Face model-info API (`https://huggingface.co/api/models/<repo>`) for the declared identifier, `gated` and the resolved `sha`; then the canonical licence document itself — `raw/main/<file>` when ungated, `hf download <repo> <file>` under HF account **`SceneWorks`** when gated |
| Gated reads | `black-forest-labs/FLUX.2-dev`, `FLUX.2-klein-9B`, `FLUX.2-klein-9b-kv`, `ideogram-ai/ideogram-4-fp8`, `facebook/sam3`, `black-forest-labs/FLUX.1-dev` — all retrieved 2026-08-02 with the authenticated CLI |
| **Read and signed off by a human** | **_(NOT YET — Michael)_** |

## What this document is

**Facts with verbatim quotes.** For each checkpoint: the identifier the upstream declares *verbatim*,
the URL of the document that identifier was transcribed from, the retrieval date, whether the
repository is gated, and either the landed family it maps to (with the quote that justifies it) or
the statement that it needs a **new** family (with the operative quote for each term that family
would carry).

It contains **no legal conclusions**. Where a text is ambiguous the ambiguous passage is quoted and
marked **AMBIGUOUS**. Where no canonical text was reachable the row says **NOT FOUND**. Where the
upstream could not be identified the row says **UNDETERMINED** with what was tried. Nothing was
guessed.

Every quote is reproduced under fair use for the purpose of review, limited to the operative clause
relied on, and attributed to the URL it came from. **No licence is reproduced in full.**

## Disclosure only — the constraint that governs every entry

Nothing built from this table blocks, gates, degrades or withholds anything. A term records what a
text **names**, never what a user may or may not do. **Where a licence is silent, the surface stays
silent** — in particular, `NonCommercialOutputs` is proposed for **no** checkpoint and **no** new
family here, exactly as sc-16662 decided (U3) and as `families.rs` test-enforces. Two of the three
new families proposed below address Outputs *in the opposite direction* (they permit commercial use
of Outputs), and those quotes are recorded so the absence is evidenced rather than merely unstated.

The typed vocabulary is fixed by the landed contract and terms are only ever drawn from it:
`AttributionRequired`, `NoticeFileRequired`, `NonCommercialWeights`, `NonCommercialOutputs`,
`RevenueCeiling{amount_usd, boundary}`, `RegistrationRequired{contact: Option}`,
`AcceptableUsePolicy{url: Option}`, `DeployerObligation{text}`, `DownstreamLicenseCopy{family}`,
`DownstreamRestrictions{family}`, and `GatedAccess` — which lives on the **component**
(`ComponentLicense::gated`), not the family (sc-16898, U7).

## Row granularity

**Repository-level**, per the epic's decision, with the five exceptions listed in
[Granularity exceptions](#granularity-exceptions--where-a-repository-level-row-is-genuinely-wrong).
For a component redistributed inside another party's snapshot, `source_url` points at the
**upstream** card, not the redistributor's — the redistributor's licence tag declares the *bundle*.
[X4](#x4-a-redistributors-licence-tag-is-demonstrably-not-the-components-licence--comfy-orglens)
is the finding that shows why: a redistributor's stated attribution for a component is, in one
observed case, verifiably wrong against the primary source.

---

# UNRESOLVED — read this first

Twelve items. Each is **AMBIGUOUS** (the text does not settle it), **NOT FOUND** (no canonical text
was reachable) or **UNDETERMINED** (the upstream could not be identified). None was resolved by
guessing.

| # | item | class | who it blocks |
| --- | --- | --- | --- |
| [X1](#x1-every-microsoftmage-flow-repository-is-gone-from-the-hub-not-found) | all six `microsoft/Mage-Flow*` repos are **gone from the Hub** — and they are pinned by revision in `release/real-weight-models.toml` | NOT FOUND | 6 census rows the census recorded as `anchored (mit)` |
| [X2](#x2-microsoftlens-and-lens-turbo-are-gone-too--mit-survives-only-second-hand) | `microsoft/Lens` / `Lens-Turbo` likewise gone; **MIT survives only second-hand** | NOT FOUND | 2 rows |
| [X3](#x3-scenewoorkslens-ships-three-licences-in-one-mit-declared-repository) | `SceneWorks/Lens` declares `mit` while its own README says the `vae/` is FLUX.2-dev's, under BFL's non-commercial licence | AMBIGUOUS | a granularity exception |
| [X4](#x4-a-redistributors-licence-tag-is-demonstrably-not-the-components-licence--comfy-orglens) | `Comfy-Org/Lens` attributes FLUX.2-dev as **`apache-2.0`**; BFL's own gated `LICENSE.md` is a non-commercial licence | AMBIGUOUS (one side verifiably wrong) | the `source_url` convention |
| [X5](#x5-boogu-declares-apache-20-over-a-flux1-vae-it-does-not-identify) | Boogu declares Apache-2.0 repo-wide but its README says the VAE is "the open-source **FLUX.1 VAE**" — dev or schnell is never said | AMBIGUOUS | 3 Boogu rows + a granularity exception |
| [X6](#x6-kolors-u6-carried-forward-unresolved-and-now-with-both-texts-read) | Kolors: card `apache-2.0`, `MODEL_LICENSE` (Chinese, 2024/7/6) committed beside the weights with a commercial-registration clause. **sc-16662 U6, still open** | AMBIGUOUS | 1 row + whether a `kolors-model-license` family is needed |
| [X7](#x7-kwai-kolorskolors-controlnet-pose-declares-no-licence-at-all) | `Kwai-Kolors/Kolors-ControlNet-Pose` declares **no licence of any kind** | NOT FOUND | 1 row |
| [X8](#x8-openaiclip-vit-large-patch14-declares-no-licence--and-the-repo-pins-it-as-mit) | `openai/clip-vit-large-patch14` declares **no licence**; `release/real-weight-models.toml` pins it as `MIT` | NOT FOUND | the most widely shared component in the catalog |
| [X9](#x9-a-third-bfl-licence-version-is-in-circulation-v20-inside-an-alibaba-pai-controlnet) | three distinct BFL texts are in play — v1.1.1 (landed), **v2.0** (inside alibaba-pai's ControlNet), **v2.1** (BFL's FLUX.2 repos) | AMBIGUOUS | 1 row; whether one or two new BFL families |
| [X10](#x10-sceneworkskrea2-pose-controlnet-beta-declares-experimental-research-only-with-no-text) | `SceneWorks/krea2-pose-controlnet-beta` declares `experimental-research-only` and publishes **no text**. The census called it `anchored (krea-2-community) by derivation`; the repo says otherwise | NOT FOUND | 1 row; it is SceneWorks' own repo |
| [X11](#x11-theclusteramoral-gemma-3-12b-v2-mlx-4bit-declares-apache-20-not-gemma) | `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit` — a Gemma-3 derivative — declares **`apache-2.0`**, not `gemma` | AMBIGUOUS | 1 row |
| [X12](#x12-eight-components-remain-undetermined) | **8 components** whose upstream repository still cannot be named | UNDETERMINED | 8 component rows |

## X1. Every `microsoft/Mage-Flow*` repository is **gone from the Hub** (NOT FOUND)

The census records six Mage-Flow rows as `anchored (mit)`, four of them **pinned by revision** in
`release/real-weight-models.toml`. All six now return `RepositoryNotFoundError` (HTTP 404) under an
**authenticated** read on 2026-08-02 — authenticated, so this is a genuine absence and not a gate
refusal:

| repo id | pinned revision in `release/real-weight-models.toml` | 2026-08-02 |
| --- | --- | --- |
| `microsoft/Mage-Flow` | `faca09c18c1c19458e7fbc3f7bce6f7a7d4d01a9` | **404** |
| `microsoft/Mage-Flow-Base` | *(not pinned)* | **404** |
| `microsoft/Mage-Flow-Turbo` | *(not pinned)* | **404** |
| `microsoft/Mage-Flow-Edit` | `b01d524f86498b7dabcc4b3572c6d264d786a16e` | **404** |
| `microsoft/Mage-Flow-Edit-Base` | `8654a7bc0283ab2946385230b5b2eb944e0b76ea` | **404** |
| `microsoft/Mage-Flow-Edit-Turbo` | `14427bd7627d3a25436497a5939e1096f6a0d523` | **404** |

Measured with `huggingface_hub.HfApi().model_info()` under HF account `SceneWorks`.

**The only reachable declaration is a third party's.** `Comfy-Org/Mage-Flow` declares
`license: mit` (HF model-info API, 2026-08-02, `gated: False`). That is a redistributor's tag, not
Microsoft's. `mage-flow-community/Mage-Flow` likewise declares `license: mit`.

**Consequence.** The six Mage-Flow rows can carry family `mit` only on **second-hand** evidence.
The in-repo comment `MIT (microsoft/Mage).` at `release/real-weight-models.toml:783` (repeated at
`:852`) records what a SceneWorks engineer read at pin time and is the closest thing to a
contemporaneous first-party record. **Michael: decide whether a second-hand declaration is acceptable as `declared`, or whether
these six rows carry `declared: "MIT"` with a `source_url` that no longer resolves.** Either way the
`source_url` must not silently point at a 404.

## X2. `microsoft/Lens` and `Lens-Turbo` are gone too — MIT survives only second-hand

The crate's own prose already recorded that Microsoft pulled the repository; this confirms it and
adds that the same is true of the Turbo variant. Both return `RepositoryNotFoundError` under
authentication, 2026-08-02.

Two redistributors record what Microsoft declared. `Comfy-Org/Lens` `README.md`
(`sha` `198d6ddf4d9fac0d8b0548dc9be4310452f5c146`, retrieved 2026-08-02) lists its sources verbatim:

> "Original model repository:
> - https://huggingface.co/microsoft/Lens (mit)"

— `https://huggingface.co/Comfy-Org/Lens/raw/main/README.md`. And `SceneWorks/Lens` (the rehost the
crate now trains from) declares `license: mit` in its own card metadata and states:

> "Self-contained diffusers-layout snapshot of Microsoft's **Lens** text-to-image model, re-assembled
> for in-house (SceneWorks) use after Microsoft removed the original `microsoft/Lens` repository from
> the Hub."

— `https://huggingface.co/SceneWorks/Lens/raw/main/README.md`, retrieved 2026-08-02, `sha`
`5c5521d4417a3cae55816929ece69319d1e7712a`.

`Comfy-Org/Lens` itself declares `license: other` with **no `license_name`**, so its own card does
not name a licence for the files it ships; the `(mit)` above is prose about the deleted upstream.
**NOT FOUND at first hand.** Same decision as X1.

## X3. `SceneWorks/Lens` ships three licences in one `mit`-declared repository

The repo's card declares `license: mit`. Its README then tabulates its own components:

> "| `text_encoder/`  | gpt-oss-20b (MXFP4), used encoder-only — from [`openai/gpt-oss-20b`] | Apache-2.0 |
> | `vae/` | FLUX.2 VAE (`AutoencoderKLFlux2`) — from [`black-forest-labs/FLUX.2-dev`] | FLUX.2-dev license |"

— `https://huggingface.co/SceneWorks/Lens/raw/main/README.md`, retrieved 2026-08-02.

So an `mit`-declared repository states, in its own words, that one of its directories is governed by
BFL's FLUX.2 licence — which is **non-commercial** (see the new family below). A repository-level
`mit` row would mis-declare that file. **AMBIGUOUS as to what `declared` should be for the repo;
unambiguous that per-file rows are required here.** See
[Granularity exceptions](#granularity-exceptions--where-a-repository-level-row-is-genuinely-wrong).

## X4. A redistributor's licence tag is demonstrably **not** the component's licence — `Comfy-Org/Lens`

This is the single clearest justification for the epic's decision #2, and it is worth recording as a
fact rather than a principle.

`Comfy-Org/Lens` `README.md` attributes its bundled VAE like this:

> "- https://huggingface.co/black-forest-labs/FLUX.2-dev (apache-2.0)"

— `https://huggingface.co/Comfy-Org/Lens/raw/main/README.md`, retrieved 2026-08-02.

BFL's own gated `LICENSE.md` for that repository, read under authentication the same day at `sha`
`26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`, opens:

> "FLUX Non-Commercial License v2.1"

and its card metadata declares `license: other`, `license_name: flux-non-commercial-license`.
`apache-2.0` appears nowhere in it.

**A redistributor stated a licence for a component and was wrong.** Nothing here is a criticism of
Comfy-Org — the point is structural: had the table taken `declared` from the snapshot the code
actually loads, it would have transcribed "Apache-2.0" for a non-commercial artifact. **Every
redistributed-component row must source from the upstream card.**

## X5. Boogu declares Apache-2.0 over a "FLUX.1 VAE" it does not identify

All three Boogu repositories declare `license: apache-2.0` (`gated: False`, 2026-08-02) and the
README repeats it:

> "**Boogu-Image-0.1** is a competitive **Apache-2.0 open-source unified image generation and editing
> model family**"

— `https://huggingface.co/Boogu/Boogu-Image-0.1-Base/raw/main/README.md`, retrieved 2026-08-02.

The same README also says, in its limitations section:

> "Because we use the open-source **FLUX.1 VAE**, reconstruction loss is relatively large…"

— same source. **It never says whether that is FLUX.1 [dev] or FLUX.1-schnell.** The two carry
different licences: `black-forest-labs/FLUX.1-schnell` declares `license: apache-2.0`;
`black-forest-labs/FLUX.1-dev` declares `flux-1-dev-non-commercial-license` and is non-commercial.
"open-source" in the sentence above is suggestive of schnell and is **not** a licence identifier.

**AMBIGUOUS — recorded, not resolved.** Grepped the repository for `schnell` in every Boogu crate
and its READMEs: no occurrence. This is the highest-value single unknown in the whole set, because
`boogu_image` is the epic's worked example and a wrong reading here is the difference between an
Apache row and a non-commercial one.

## X6. Kolors (U6) — carried forward unresolved, and now with **both** texts read

sc-16662 left U6 open. Both documents have now been retrieved; the conflict is real.

**Document A — the card.** `Kwai-Kolors/Kolors-diffusers` HF model-info API, 2026-08-02:
`license: apache-2.0`, `gated: False`, no `license_name`, no `license_link`.

**Document B — `MODEL_LICENSE`, committed beside the weights**, which
`mlx-gen-kolors/src/convert.rs` copies verbatim into every converted snapshot. It is in Chinese and
opens `模型许可协议` / `模型发布日期：2024/7/6` ("Model License Agreement" / "model release date:
2024/7/6"). Its §2(c) is the operative divergence from Apache-2.0:

> "附加商业条款：若您希望将模型及模型衍生品用作商业用途，则您必须向许可人申请许可，许可人可自行决定向您授予许可。"

— `https://huggingface.co/Kwai-Kolors/Kolors-diffusers/raw/main/MODEL_LICENSE`, retrieved
2026-08-02. (Working rendering, **not** a translation of record: *Additional commercial terms: if
you wish to use the model and model derivatives for commercial purposes, you must apply to the
licensor for a licence, which the licensor may grant at its sole discretion.*)

Two further operative clauses, same source:

> §2(b)(i) "您必须向所有该模型作品或使用该作品的产品或服务的任何第三方接收者提供模型作品的来源和本协议的副本；"

> §3(a) "您对本模型作品的使用必须遵守适用法律法规（包括贸易合规法律法规），并遵守《服务协议》(https://kolors.kuaishou.com/agreement)。您必须将本第 3(a) 和 3(b) 条中提及的使用限制作为可执行条款纳入任何规范本模型作品使用和/或分发的协议"

**One corroborating fact.** SceneWorks' own derived tokenizer repo,
`SceneWorks/kolors-chatglm3-tokenizer`, declares `license: other`, `license_name: kolors`,
`license_link: https://huggingface.co/Kwai-Kolors/Kolors-diffusers/blob/main/MODEL_LICENSE`
(HF model-info API, 2026-08-02) — i.e. an existing SceneWorks publication already treats
`MODEL_LICENSE` as governing. That is evidence of house practice, **not** of what Kwai-Kolors
intends.

**AMBIGUOUS. Michael decides.** If `MODEL_LICENSE` governs, a **new family** is needed
(`kolors-model-license`); its candidate terms and quotes are set out
[below](#candidate-new-family-4-kolors-model-license--conditional-on-x6). If the card governs, the
row is `apache-2-0` and the committed `MODEL_LICENSE` is noise. **Do not resolve this by picking the
less restrictive one.**

## X7. `Kwai-Kolors/Kolors-ControlNet-Pose` declares **no licence at all**

HF model-info API, 2026-08-02: `gated: False`, `cardData.license` **absent**, no `license_name`, no
`license_link`, and **no licence-shaped file** in the repository's sibling listing (checked against
`licen|notice|policy|terms|MODEL_LICENSE`). `https://huggingface.co/Kwai-Kolors/Kolors-ControlNet-Pose/raw/main/README.md`
returns **HTTP 404** — the repo has no model card.

**NOT FOUND.** The sibling `Kwai-Kolors/Kolors-IP-Adapter-Plus` *does* declare `apache-2.0`, so this
is not a Kwai-Kolors-wide policy; it is one repository with nothing stated. The honest entry is a
blank.

## X8. `openai/clip-vit-large-patch14` declares **no licence** — and the repo pins it as MIT

HF model-info API, 2026-08-02: `gated: False`, `cardData.license` **absent**, and no licence file in
the sibling listing. The model card is long and discusses intended use at length but names no
licence:

> "The model is intended as a research output for research communities."

— `https://huggingface.co/openai/clip-vit-large-patch14/raw/main/README.md`, retrieved 2026-08-02.
That is a statement of intended use, **not** a licence grant, and it must not be transcribed as one.

`release/real-weight-models.toml:723` records `license = "MIT"` for this exact pinned revision
(`32bd64288804d66eefd0ccbe215aa642df71cc41`), and the census anchors it to family `mit`. The MIT
identifier is presumably taken from OpenAI's `openai/CLIP` **source** repository on GitHub — a
different artifact from the weights on the Hub. (`release/real-weight-models.toml:422` shows the
house already knows this distinction exists: it corrected `openai/whisper-base` to Apache-2.0
"distinct from the MIT license on OpenAI's Whisper *source* repository".)

**NOT FOUND at the weights' own card.** This matters more than any other row here: the census counts
this one snapshot as serving both embedder ids, the SDXL tokenizer, and the FLUX IP-adapter image
tower. **Michael: decide whether `declared` is a blank, or `"MIT"` sourced from the GitHub repo with
that provenance stated on the row.** The same NOT FOUND applies to
`openai/clip-vit-large-patch14-336`, which `Kwai-Kolors/Kolors-IP-Adapter-Plus` identifies as its
image tower ("We employ the Openai-CLIP-336 model as the image encoder" —
`https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/raw/main/README.md`, 2026-08-02).

## X9. A **third** BFL licence version is in circulation — v2.0, inside an alibaba-pai ControlNet

Three distinct Black Forest Labs texts are loaded by this repository's providers, all read
2026-08-02:

| text | first line, verbatim | where it was read | bytes |
| --- | --- | --- | --- |
| **v1.1.1** — the landed family | `FLUX.1 [dev] Non-Commercial License v1.1.1` | `black-forest-labs/FLUX.1-dev` `LICENSE.md`, authenticated, `sha` `3de623fc…` | 18,621 |
| **v2.0** | `FLUX [dev] Non-Commercial License v2.0` | `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` `LICENSE.txt`, ungated | 18,764 |
| **v2.1** | `FLUX Non-Commercial License v2.1` | `black-forest-labs/FLUX.2-dev` `LICENSE.md`, authenticated, `sha` `26afe3a7…` | 18,158 |

`black-forest-labs/FLUX.2-klein-9B`'s `LICENSE.md` is **byte-identical** to FLUX.2-dev's (both
SHA-256 `468D9F4332C0C895…`, verified by hash on the two cached blobs, 2026-08-02).
`FLUX.2-klein-9b-kv`'s `LICENSE` is a different blob (`E98F298DAE1BCC91…`) whose first line is
nonetheless the same `FLUX Non-Commercial License v2.1`.

The v2.0 text's own Models definition explicitly enumerates FLUX.2:

> "'Models' includes the models denoted as FLUX.x [dev] … including but not limited to FLUX.1 [dev],
> … FLUX.1 Krea [dev], and FLUX.2 [dev]"

— `https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union/raw/main/LICENSE.txt`,
retrieved 2026-08-02.

**Compounding it**, that repository's card `license_link` is
`https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.txt` — and BFL's file is
`LICENSE.**md**`, not `.txt`. The link is dead.

**AMBIGUOUS.** BFL has published at least two texts covering FLUX.2 and a third party ships the
older one beside its weights. **Michael: decide whether v2.0 and v2.1 are one family or two.** The
default that costs nothing is to give BFL's own repos the v2.1 family and record the alibaba-pai row
as declaring v2.0 with its own `source_url` — which is what the tables below do, marked.

## X10. `SceneWorks/krea2-pose-controlnet-beta` declares `experimental-research-only` with no text

HF model-info API, 2026-08-02: `license: other`, `license_name: **experimental-research-only**`,
`gated: False`, `base_model: krea/Krea-2-Turbo`, and **no licence file** in the sibling listing. The
card's body is a warning, not a grant:

> "⚠️ **PURELY EXPERIMENTAL — DO NOT USE IN PRODUCTION.** These are **feasibility-spike** checkpoints
> from a GO/NO-GO experiment, **not** a finished model."

— `https://huggingface.co/SceneWorks/krea2-pose-controlnet-beta/raw/main/README.md`, retrieved
2026-08-02.

The census recorded this row as "trained in-house on Krea 2 → anchored (`krea-2-community`) by
derivation". **The repository itself declares a different identifier**, and no text stands behind
it. **NOT FOUND** — and it is SceneWorks' own repository, so it is the one row Michael can fix
rather than merely decide.

## X11. `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit` declares `apache-2.0`, not `gemma`

HF model-info API, 2026-08-02: `license: **apache-2.0**`, `gated: False`. The census records this
row (the MLX-only LTX uncensored prompt enhancer) as `anchored (gemma-terms) by derivation`. The
repository says otherwise, and no text in it was found that reconciles the two.

**AMBIGUOUS.** Both readings are available and neither is settled by the evidence: transcribe what
the repository declares (`apache-2.0`), or record that a Gemma-3 derivative carries the Gemma Terms
regardless of what a re-publisher tags it. **This is the same question class as X4**, in the
direction where SceneWorks would be the one over-reading a third party's tag. Do not resolve it by
preferring the stricter one either — record the declaration and flag it.

## X12. Eight components remain UNDETERMINED

Full detail in [Job 2](#job-2--the-24-undetermined-components). The eight the network could not
settle:

| component | what was tried |
| --- | --- |
| FLUX.2-klein's Qwen3 dense text encoder | FLUX.2 card, README and `LICENSE.md` name no third-party model; §9 mentions "Third Party Materials" generically |
| FLUX.2-dev's Mistral3 language tower | same; no Mistral repo id anywhere in the FLUX.2 card or the repo's Rust |
| FLUX.2-dev's Pixtral vision tower | same |
| FLUX.2-dev's multimodal projector | same; may be BFL-authored, which would make the FLUX.2 row correct — unestablished |
| Anima's Qwen3-0.6B base text encoder | `circlestone-labs/Anima` card names no upstream; the file is `text_encoders/qwen_3_06b_base.safetensors` inside the Anima snapshot |
| Krea's / Mage's / Ideogram's / Z-Image's Qwen3-VL towers | each vendor's card names the architecture at most; no repo id |
| the lightx2v Wan step-distill diff-patch consumed by `scail2_14b` | narrowed to a family (see Job 2) but **not** to one repository |
| the `sat-scail2` Bias-Aware DPO LoRA | searched HF for `sat-scail2`, `scail2`, `SCAIL` (40+ results reviewed) — **no repository by that name exists**. "sat" most likely names a *key layout* (SwissArmyTransformer), not a publisher |

---

# NEW families needed — three, plus one conditional

## New family 1 — `flux-non-commercial-v2-1`

**Covers:** `black-forest-labs/FLUX.2-dev`, `FLUX.2-klein-9B`, `FLUX.2-klein-9b-kv` — and, per X3,
the FLUX.2 VAE redistributed inside `SceneWorks/Lens` / `SceneWorks/lens-mlx`.

| | |
| --- | --- |
| name | FLUX Non-Commercial License v2.1 |
| proposed id | `flux-non-commercial-v2-1` |
| proposed SPDX | `LicenseRef-FLUX-Non-Commercial-v2.1` |
| `text_url` | `https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.md` (gated) |
| read at | `sha` `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`, 2026-08-02, HF account `SceneWorks` |
| declared identifier | `flux-non-commercial-license` (card `license_name`; `license: other`) |

**Why it cannot reuse `flux-1-dev-non-commercial`:** that family's `name` is
`FLUX.1 [dev] Non-Commercial License v1.1.1` and its `text_url` is the FLUX.1 text on GitHub. This
is a different document with a different title, a different Models definition, and — decisively —
**two obligations the v1.1.1 text does not impose in the same shape** (the Attribution Notice and
the AI-disclosure duty). Stretching v1.1.1 over FLUX.2 is exactly what the census warned against.

### Terms, each with its operative quote

All quotes from `black-forest-labs/FLUX.2-dev` `LICENSE.md`, retrieved 2026-08-02 under HF account
`SceneWorks`.

**`NonCommercialWeights`** — §2(b):

> "You may only access, use, Distribute, or create Derivatives of the FLUX Model or Derivatives for
> Non-Commercial Purposes."

**No `NonCommercialOutputs`** — §2(d) addresses Outputs in the opposite direction, so its absence is
evidenced, not merely unquoted:

> "You may use Output for any purpose (including for commercial purposes), except as expressly
> prohibited herein."

**`DownstreamLicenseCopy{"flux-non-commercial-v2-1"}`** — §3(a):

> "you must make available a copy of this License to third-party recipients of the FLUX Mode and/or
> Derivatives you Distribute, and specify that any rights to use the FLUX Model and/or Derivatives
> shall be directly granted by Company to said third-party recipients pursuant to this License"

*(The `Mode` typo is in the source; quoted as found.)*

**`NoticeFileRequired`** and **`AttributionRequired`** — §3(b), one clause, and unlike Gemma's it
names itself an *Attribution* Notice, which is precisely the textual hook sc-16662's **U11** offers
as the distinguisher:

> "you must prominently display the following notice alongside the Distribution of the FLUX Model or
> Derivative (such as via a \"Notice\" text file distributed as part of such FLUX Model or
> Derivative) (the \"Attribution Notice\")"

> **U11 linkage.** If Michael settles U11 as **option A** (the "attribution notice" wording governs),
> this family takes **both** terms and is consistent with `nvidia-open-model`. If **option C**
> (a notice clause yields only `NoticeFileRequired`), it takes one. The row is written both ways here
> so the transcription pass follows U11 rather than deciding it.

**`DeployerObligation`** ×2 — §2(e), two distinct duties in one sentence. Content filtering:

> "implement and maintain content filtering measures (\"Content Filters\") for your use of the FLUX
> Model or Derivatives to prevent the creation, display, transmission, generation, or dissemination
> of unlawful or infringing content"

and AI-generation disclosure — **new relative to the landed v1.1.1 family**:

> "ensure Output includes disclosure (or other indication) that the Output was generated or modified
> using artificial intelligence technologies to the extent required under applicable law."

**`AcceptableUsePolicy{Some("https://bfl.ai/legal/usage-policy")}`** — and this is a **change from
sc-16662's U2 finding**. The v2.1 licence text enumerates prohibited uses inline in §4 and names no
policy; but the card's gate prompt does, verbatim:

> "By clicking \"Agree\", you agree to the \[FLUX Non-Commercial License Agreement\]\(…\) and
> acknowledge the \[Acceptable Use Policy\]\(https://bfl.ai/legal/usage-policy\)."

(Square brackets and parentheses escaped above so the quote is not read as Markdown links; the
prompt's own link target for the licence is elided in the source quote and is **not** reconstructed
here.)

— `black-forest-labs/FLUX.2-dev` card metadata `extra_gated_prompt`, HF model-info API, 2026-08-02.
**That URL was fetched and resolves: HTTP 200.** This is the same evidence shape sc-16662 accepted
for Krea (U2), so the precedent is already set.

> **Reportable to sc-16662's U2.** FLUX.1 [dev]'s gate prompt *still* cites the non-existent
> `POLICY.md` — re-verified 2026-08-02, the card's `extra_gated_prompt` is unchanged and names
> `https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/POLICY.md`. So BFL now publishes a
> live usage policy for FLUX.2 while FLUX.1's citation remains dead. **The live URL must not be
> back-ported onto the FLUX.1 family** — that would be inventing an address for a text that does not
> name it. FLUX.1's `AcceptableUsePolicy{None}` stands.

## New family 2 — `ideogram-4-non-commercial`

**Covers:** `ideogram-ai/ideogram-4-fp8` (**gated**) and `ostris/ideogram_4_turbotime_lora`.

| | |
| --- | --- |
| name | Ideogram Non-Commercial Model Agreement (Last Updated: June 3, 2026) |
| proposed id | `ideogram-4-non-commercial` |
| proposed SPDX | `LicenseRef-Ideogram-4-Non-Commercial` |
| `text_url` | `https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md` (gated) |
| read at | `sha` `ee79a7237b519f1402ceacf952f30c8a31ec5073`, 2026-08-02, HF account `SceneWorks` |
| declared identifier | `ideogram-4-non-commercial` (card `license_name`; `license: other`) |

All quotes from `ideogram-ai/ideogram-4-fp8` `LICENSE.md`, retrieved 2026-08-02 under HF account
`SceneWorks`.

**`NonCommercialWeights`** — §2:

> "We hereby permit you to use, reproduce, Distribute, copy, create derivative works of (including
> Model Derivatives), and make modifications to the Model for Non-Commercial Purposes subject to the
> terms of this Agreement"

**No `NonCommercialOutputs`** — §7 addresses Outputs, and the only Output restriction is
anti-competitive-training, not commerce:

> "We claim no rights in outputs you generate using the Model. You are responsible for outputs and
> their subsequent uses. You may not use any Output to develop, train, fine-tune or distill a model
> or other product or services that is competitive with the Model"

> **AMBIGUOUS, flagged not resolved.** §1(d) folds a *use of Outputs* into the definition of
> Non-Commercial Purposes: "any use … that involves generating Output to include in, or to advertise
> or promote, revenue-generating products or services, in each case, is not a Non-Commercial
> Purpose." Whether that reaches Outputs as a licence term or merely scopes the weights grant is a
> legal read. Under the sc-16662 U3 rule — silence is silence, and this is not silence but a
> *definition* — the honest transcription is to **not** land `NonCommercialOutputs` and to record
> this passage. That is what is proposed.

**`DownstreamRestrictions{"ideogram-4-non-commercial"}`** — §3(i), the heavier "no less restrictive"
shape:

> "all permitted use of the reproduced and re-Distributed Model or Model Derivatives must be on terms
> that are no less restrictive than those set forth in this Agreement for the Model"

**`DownstreamLicenseCopy{"ideogram-4-non-commercial"}`** — §3(ii):

> "you provide all third party recipients of the Model or Model Derivative a copy of this Agreement"

**`NoticeFileRequired`** and **`AttributionRequired`** — §3(iii), again self-described as an
attribution notice (same U11 hook as FLUX.2 above):

> "you retain in all copies of the Model or Model Derivatives that you Distribute the following
> attribution notice within a \"Notice\" text file that accompanies such copy: \"Ideogram 4 is
> provided under and subject to the Ideogram Non-Commercial Model Agreement available at
> https://github.com/ideogram-oss/ideogram-4/model_licenses/LICENSE-IDEOGRAM-4-NON-COMMERCIAL. All
> rights reserved. Copyright © Ideogram, Inc.\""

The prescribed notice is quoted **complete** here — an earlier pass of this note recorded it elided
at "…Model Agreement…", and the component rows had landed that prefix as if it were the whole
notice. Re-read 2026-08-02 at the same `sha` `ee79a7237b519f1402ceacf952f30c8a31ec5073`; both rows
now carry the full string from one shared constant, so a drift job comparing the landed attribution
against §3(iii) matches.

**`AcceptableUsePolicy{Some("https://ideogram.ai/legal/usage-policy")}`** — §4, and here the URL is
**in the licence text itself**, not merely in a gate prompt:

> "adhere to the Acceptable Use Policy available at https://ideogram.ai/legal/usage-policy, which is
> hereby incorporated by reference into this Agreement"

Fetched 2026-08-02: **HTTP 200**.

**`DeployerObligation`** — §4:

> "You are responsible for implementing appropriate safety measures, including content filters and
> human oversight, suitable for your use case and to prevent the creation, display, generation or
> reproduction of unlawful or infringing content"

## New family 3 — `meta-sam-license`

**Covers:** `facebook/sam3` (**gated: manual**).

| | |
| --- | --- |
| name | SAM License (Last Updated: November 19, 2025) |
| proposed id | `meta-sam-license` |
| proposed SPDX | `LicenseRef-Meta-SAM` |
| `text_url` | `https://huggingface.co/facebook/sam3/blob/main/LICENSE` (gated) |
| read at | `sha` `3c879f39826c281e95690f02c7821c4de09afae7`, 2026-08-02, HF account `SceneWorks` |
| declared identifier | `other` — the card sets `license: other` with **no `license_name`**; the text's own title is `SAM License` |

**This licence is NOT non-commercial.** Its grant is unrestricted as to purpose — which is the
opposite of what "a bespoke Meta licence" might be assumed to mean, and worth stating plainly. §1(a):

> "You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license
> under Meta's intellectual property or other rights owned by Meta embodied in the SAM Materials to
> use, reproduce, distribute, copy, create derivative works of, and make modifications to the SAM
> Materials."

**`DownstreamRestrictions{"meta-sam-license"}`** and **`DownstreamLicenseCopy{"meta-sam-license"}`**
— §1(b)(i), both shapes in one sentence:

> "If you distribute or make the SAM Materials, or any derivative works thereof, available to a third
> party, you may only do so under the terms of this Agreement and you shall provide a copy of this
> Agreement with any such SAM Materials."

**`DeployerObligation`** — §1(b)(ii), quoted in full (the sentence carries no elision):

> "If you submit for publication the results of research you perform on, using, or otherwise in
> connection with SAM Materials, you must acknowledge the use of SAM Materials in your publication."

**No `AttributionRequired`** — and this is **U8 applied, not a new decision**. The duty above is
real, but it binds only on submitting research for publication, while `AttributionRequired` reads as
an unconditional duty on every use; landing it would make every SAM 3 render's derived union name an
obligation this text does not impose on it. sc-16662's open item **U8** already settled that shape
for `llama-3-1-community`: a 700M-MAU threshold is not a `RevenueCeiling` because the typed term
would be "a false transcription", so the condition is disclosed verbatim as a `DeployerObligation`
instead. The same reasoning selects the same shape here. The duty is disclosed with its condition
intact rather than dropped, and `facebook/sam3`'s component row therefore carries
`attribution: None`.

**No `NoticeFileRequired`** — the text names no notice file; `Notice` does not occur.
**No `AcceptableUsePolicy`** — §1(b)(iii)–(v) enumerate restrictions inline (trade controls, ITAR,
reverse engineering) and reference no external policy document.
**No `RevenueCeiling`, no `RegistrationRequired`** — neither appears.

All quotes from `facebook/sam3` `LICENSE`, retrieved 2026-08-02 under HF account `SceneWorks`.

> **Note for the component rows:** `facebook/sam2.1-hiera-large` and `facebook/sam2.1-hiera-base-plus`
> declare `license: apache-2.0` (`gated: False`) and do **not** take this family. Meta licenses SAM
> 2.1 and SAM 3 differently. The census's "SAM releases carry their own licences; nothing in the
> table is close" is half right: SAM 2.1 is plain Apache-2.0, SAM 3 is not.

## Candidate new family 4 — `kolors-model-license` — **conditional on X6**

Only needed if Michael settles X6 toward `MODEL_LICENSE`. Terms it would carry, with the quotes
already given in [X6](#x6-kolors-u6-carried-forward-unresolved-and-now-with-both-texts-read):

| term | clause |
| --- | --- |
| `RegistrationRequired{contact: None}` | §2(c) — apply to the licensor; **no address is given in the text**, so `None`, the same shape as LTX-2 |
| `DownstreamLicenseCopy{"kolors-model-license"}` | §2(b)(i) |
| `DownstreamRestrictions{"kolors-model-license"}` | §3(a) — "作为可执行条款纳入" (incorporate as enforceable provisions) |
| `AcceptableUsePolicy{Some("https://kolors.kuaishou.com/agreement")}` | §3(a), the URL is in the text |

**Do not land this family unless X6 is settled.** `text_url` would be
`https://huggingface.co/Kwai-Kolors/Kolors-diffusers/raw/main/MODEL_LICENSE`; `name` would need to
carry the Chinese title `模型许可协议` or an agreed English rendering — **that choice is Michael's,
because a rendering is not a quote.**

---

# Job 1 — the 47 UNCOVERED checkpoints

**45 of 47 now have a licence read.** 40 map to a landed family; 5 need one of the new families
above; 2 are NOT FOUND (X7, X10). Two more are AMBIGUOUS as to *which* family (X6 Kolors, X9
alibaba-pai) but their texts were read.

`gated` values are the HF `gated` field from the model-info API, 2026-08-02. `declared` is the
identifier the upstream states verbatim — from `cardData.license_name` where present, else
`cardData.license`, else the licence document's own title.

## Maps to landed family `apache-2-0` (28 rows)

All read from the HF model-info API on **2026-08-02**; each row's `source_url` is that repository's
model card, `https://huggingface.co/<repo>`. All `gated: False` unless noted.

| # | repository | declared | notes |
| --- | --- | --- | --- |
| 3 | `ByteDance/Bernini-Diffusers` | `apache-2.0` | README: "## 📄 License / Apache License 2.0." Sibling `ByteDance/Bernini-R` ships an Apache `LICENSE` |
| 4 | `Boogu/Boogu-Image-0.1-Base` | `apache-2.0` | **see X5** — the VAE inside is not identified |
| 5 | `Boogu/Boogu-Image-0.1-Turbo` | `apache-2.0` | see X5 |
| 6 | `Boogu/Boogu-Image-0.1-Edit` | `apache-2.0` | **census UNDETERMINED now RESOLVED** — the repo exists; see Job 2 |
| 7 | `lodestones/Chroma1-HD` | `apache-2.0` | README: "based on **FLUX.1-schnell**. It is fully **Apache 2.0 licensed**" |
| 8 | `lodestones/Chroma1-Base` | `apache-2.0` | |
| 9 | `lodestones/Chroma1-Flash` | `apache-2.0` | |
| 42 | `numz/SeedVR2_comfyUI` | `apache-2.0` | `base_model:` lists `ByteDance-Seed/SeedVR2-7B` and `-3B`, **both `apache-2.0`** — resolves the census's "no ByteDance repo id appears anywhere" |
| 43 | `sensenova/SenseNova-U1-8B-MoT` | `apache-2.0` | ships a `LICENSE` whose first lines are `Apache License / Version 2.0, January 2004` — verified, not assumed |
| 44 | `sensenova/SenseNova-U1-8B-MoT-LoRAs` | `apache-2.0` | |
| 46 | `Wan-AI/Wan2.2-TI2V-5B` (+ `-Diffusers`) | `apache-2.0` | |
| 47 | `Wan-AI/Wan2.2-T2V-A14B` (+ `-Diffusers`) | `apache-2.0` | see the Wan output quote below |
| 48 | `Wan-AI/Wan2.2-I2V-A14B` (+ `-Diffusers`) | `apache-2.0` | |
| 49 | `Wan-AI/Wan2.1-VACE-1.3B-diffusers` and `Wan-AI/Wan2.1-VACE-14B-diffusers` | `apache-2.0` | Two rows, two repositories, **both ids spelled in full** — each card was read on its own, not inferred from its sibling: 1.3B at `sha` `ec4d2cb0…`, 14B at `sha` `db79b90c…`, both `2026-08-02`, both front matter `license: apache-2.0`. The shorthand this row previously used (`-14B-diffusers`) left `WAN2_1_VACE_14B_DIFFUSERS`'s `source_url` as the only one of the 71 whose repository id the evidence never spelled out |
| 50 | `alibaba-pai/Wan2.2-VACE-Fun-A14B` | `apache-2.0` | |
| 55 | `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union` | `apache-2.0` | confirms the crate rustdoc |
| 56 | `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` | `apache-2.0` | |
| 57 | `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` | `apache-2.0` | |
| 59 | `Kwai-Kolors/Kolors-IP-Adapter-Plus` | `apache-2.0` | its bundled image tower is CLIP-336 — **X8** |
| 61 | `h94/IP-Adapter` | `apache-2.0` | agrees with the pin at `release/real-weight-models.toml:686` |
| 62 | `InstantX/InstantID` | `apache-2.0` | |
| 63 | `xinsir/controlnet-openpose-sdxl-1.0` | `apache-2.0` | |
| 64 | `guozinan/PuLID` | `apache-2.0` | |
| 65 | `lightx2v/Qwen-Image-Lightning` | `apache-2.0` | |
| 66 | `lightx2v/Qwen-Image-Edit-2511-Lightning` | `apache-2.0` | |
| 87 | `facebook/sam2.1-hiera-large` | `apache-2.0` | **not** the SAM 3 licence |
| 88 | `facebook/sam2.1-hiera-base-plus` | `apache-2.0` | |
| 90 | `QuantStack/Wan2.2-TI2V-5B-GGUF` | `apache-2.0` | a re-host; README: "direct conversion of [Wan-AI/Wan2.2-TI2V-5B] … all original licensing terms and usage restrictions remain in effect" |

**Wan's own output statement**, worth recording because it is the mirror of the FLUX/Ideogram
Outputs clauses and confirms no `NonCommercialOutputs` anywhere near these rows:

> "The models in this repository are licensed under the Apache 2.0 License. We claim no rights over
> the your generated contents"

— `https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B/raw/main/README.md`, retrieved 2026-08-02 (typo in
source).

## Maps to landed family `mit` (6 rows)

| # | repository / artifact | declared | source_url | gated | note |
| --- | --- | --- | --- | --- | --- |
| 36 | `zai-org/SCAIL-2` | `mit` | `https://huggingface.co/zai-org/SCAIL-2` | False | the converted `SceneWorks/scail2-mlx` also declares `mit` |
| 72 | `madebyollin/sdxl-vae-fp16-fix` | `mit` | `https://huggingface.co/madebyollin/sdxl-vae-fp16-fix` | False | **required** Candle SDXL component; agrees with the pin at `release/real-weight-models.toml:750` |
| 84 | `QuanSun/EVA-CLIP` | `mit` | `https://huggingface.co/QuanSun/EVA-CLIP` | False | the file is `EVA02_CLIP_L_336_psz14_s6B.pt`, named in `crates/media/mlx-gen/tools/convert_eva_clip.py:6` |
| 83 | facexlib `parsing_bisenet` | `MIT License` | `https://raw.githubusercontent.com/xinntao/facexlib/master/LICENSE` | n/a | **census UNDETERMINED now RESOLVED.** First lines: `MIT License` / `Copyright (c) 2020 Xintao Wang` |
| 21 | `microsoft/Lens` | `mit` — **second-hand only** | *(none — repo 404)* | n/a | **X2** |
| 22 | `microsoft/Lens-Turbo` | `mit` — second-hand only | *(none — repo 404)* | n/a | **X2** |

`laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` (#74) and `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` also
declare `mit`, resolving the census's `generic?`.

## Maps to landed family `flux-1-dev-non-commercial` (3 rows)

These are the **stricter-than-assumed** rows: three adapters that a reader would expect to be
permissive and which declare BFL's non-commercial licence.

| # | repository | declared | source_url | gated |
| --- | --- | --- | --- | --- |
| 53 | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` | `flux-1-dev-non-commercial-license` | card `license_link` → `https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md` | False |
| 60 | `XLabs-AI/flux-ip-adapter` | `flux-1-dev-non-commercial-license` | same `license_link` (the card's copy is truncated to `…/LICENSE.`) | False |
| 67 | ByteDance Hyper-FLUX 8-step LoRA → **`ByteDance/Hyper-SD`** | `FLUX.1 [dev] Non-Commercial License` | `https://huggingface.co/ByteDance/Hyper-SD/raw/main/LICENSE.md` | False |

Row 67 was **UNDETERMINED in the census** and is now resolved. `ByteDance/Hyper-SD` declares no card
licence but commits a `LICENSE.md` that opens:

> "> For Flux.1-DEV-related models, please agree with the following license.
>
> **FLUX.1 \\[dev\\] Non-Commercial License**"

— `https://huggingface.co/ByteDance/Hyper-SD/raw/main/LICENSE.md`, retrieved 2026-08-02. The repo's
file listing contains `Hyper-FLUX.1-dev-8steps-lora.safetensors` (and a 16-step sibling), which is
the artifact `mlx-gen-flux/src/config.rs:23` describes. **Per-file granularity is arguably needed
here** — `Hyper-SD`'s SDXL LoRAs are covered by a different part of the same file — but the FLUX
LoRA this repo loads is unambiguously under the quoted clause.

## Needs new family `flux-non-commercial-v2-1` (3 rows)

| # | repository | declared | source_url | gated | sha read |
| --- | --- | --- | --- | --- | --- |
| 14 | `black-forest-labs/FLUX.2-dev` | `flux-non-commercial-license` | `https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.md` | **auto** | `26afe3a7…` |
| 12 | `black-forest-labs/FLUX.2-klein-9B` | `flux-non-commercial-license` | `https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md` | **auto** | `92196c8e…` (text byte-identical to FLUX.2-dev's) |
| 13 | `black-forest-labs/FLUX.2-klein-9b-kv` | `flux-non-commercial-license` | `https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv/blob/main/LICENSE` | **auto** | `a6dfb36e…` (different blob, same title) |

## Needs new family `ideogram-4-non-commercial` (2 rows)

| # | repository | declared | source_url | gated | sha read |
| --- | --- | --- | --- | --- | --- |
| 15 | `ideogram-ai/ideogram-4-fp8` | `ideogram-4-non-commercial` | `https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md` | **auto** | `ee79a723…` |
| 68 | ostris TurboTime LoRA → **`ostris/ideogram_4_turbotime_lora`** | `ideogram-4-non-commercial` | card `license_link` → the same LICENSE.md | False | `63f528b4…` |

Row 68 was **UNDETERMINED in the census** and is now resolved. The repository's card front-matter,
verbatim:

> "license: other
> license_name: ideogram-4-non-commercial
> license_link: https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md
> base_model: ideogram-ai/ideogram-4-fp8"

— `https://huggingface.co/ostris/ideogram_4_turbotime_lora/raw/main/README.md`, retrieved
2026-08-02. **The LoRA is not independently licensed** — it declares the model vendor's
non-commercial agreement. `ostris/ideogram_4_unconditional_lora` (which supplies the unconditional
DiT path) declares the same.

## Needs new family `meta-sam-license` (1 row)

| # | repository | declared | source_url | gated | sha read |
| --- | --- | --- | --- | --- | --- |
| 89 | `facebook/sam3` | `other` (card); document title `SAM License` | `https://huggingface.co/facebook/sam3/blob/main/LICENSE` | **manual** | `3c879f39…` |

## AMBIGUOUS or NOT FOUND (4 rows)

| # | repository | state | see |
| --- | --- | --- | --- |
| 17 | `Kwai-Kolors/Kolors-diffusers` | declared `apache-2.0` (card) **vs** `MODEL_LICENSE` (committed, Chinese, commercial-registration clause) | **X6** |
| 58 | `Kwai-Kolors/Kolors-ControlNet-Pose` | **NOT FOUND** — no licence, no card | **X7** |
| 54 | `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` | declared `flux-dev-non-commercial-license`; ships **v2.0**, card `license_link` is dead | **X9** |
| 71 | `SceneWorks/krea2-pose-controlnet-beta` | **NOT FOUND** — declares `experimental-research-only`, no text | **X10** |

## Also corrected against the census

| # | artifact | census said | evidence says |
| --- | --- | --- | --- |
| 76 | `nvidia/PiD` | `anchored (nvidia-nsclv1)` | **confirmed, but the card declares nothing.** `cardData.license` is absent; the declaration lives in the README body: "This model is released under the [NSCLv1](https://huggingface.co/nvidia/PixelDiT-1300M-1024px/blob/main/LICENSE) License. The work and any derivative works may only be used for non-commercial (research or evaluation) purposes." — `https://huggingface.co/nvidia/PiD/raw/main/README.md`, 2026-08-02. `gated: False`. **`source_url` must be the README, not the card metadata.** |
| 2 | `circlestone-labs/Anima-Official-LoRAs` | `anchored` | **confirmed by text.** The repo declares no card licence but commits a `LICENSE.md` opening `CircleStone Labs Non-Commercial License v1.0` — `https://huggingface.co/circlestone-labs/Anima-Official-LoRAs/raw/main/LICENSE.md`, 2026-08-02 |
| 79 | `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit` | `anchored (gemma-terms) by derivation` | declares **`apache-2.0`** — **X11** |
| 41 | `SG161222/RealVisXL_V5.0` | `generic?` | declares `openrail++` (card, `gated: False`) → family `creativeml-openrail-pp-m`. **Ships no licence text of its own**; the family's canonical text remains SDXL's `LICENSE.md`. Agrees with `release/real-weight-models.toml:765` |
| 75 | `google/umt5-xxl` | `generic?` | declares `apache-2.0` → family `apache-2-0` |
| 34/35 | `Efficient-Large-Model/Sana_*` | `generic?` | `apache-2.0` confirmed on the card; corroborates sc-16662 **U5**, already closed |
| 30/31/32/33/51/52/20 | mochi-1, Qwen-Image ×3, Z-Image ×2, krea-realtime | `generic?` | all declare `apache-2.0`. `Qwen/Qwen-Image` additionally ships an Apache-2.0 `LICENSE` (verified first lines) |

---

# Job 2 — the 24 UNDETERMINED components

**16 of 24 identified.** Eight remain UNDETERMINED and are listed in [X12](#x12-eight-components-remain-undetermined).

## Shape (a) — redistributed inside another party's snapshot (16 rows)

Per decision #2, `source_url` is the **upstream** card. Where the upstream could not be established,
the row stays UNDETERMINED and the redistributor's tag is recorded as *context only* — never as the
component's `declared`.

| component | upstream | declared | source_url | status |
| --- | --- | --- | --- | --- |
| FLUX.1's CLIP-L (`text_encoder/`) | `openai/clip-vit-large-patch14` — named for the layout at `candle-gen-flux/src/flux1_load.rs:72` | **none** | card declares nothing | **IDENTIFIED, licence NOT FOUND — X8** |
| FLUX.1's T5-XXL (`text_encoder_2/`) | `google/t5-v1_1-xxl` — named for the layout at `candle-gen-flux/src/packed_te.rs:266` | `apache-2.0` | `https://huggingface.co/google/t5-v1_1-xxl` | **IDENTIFIED** (architecture repo; byte-provenance of the redistributed file not verified) |
| FLUX.1's VAE | **not third-party** — BFL's own `ae.safetensors` inside the FLUX.1 repo | per the FLUX.1 repo row | `black-forest-labs/FLUX.1-{dev,schnell}` | **RESOLVED — no separate row needed** |
| FLUX.2-klein's Qwen3 tower | — | — | — | **UNDETERMINED** |
| FLUX.2-dev's Mistral3 tower | — | — | — | **UNDETERMINED** |
| FLUX.2-dev's Pixtral vision tower | — | — | — | **UNDETERMINED** |
| FLUX.2-dev's multimodal projector | — | — | — | **UNDETERMINED** |
| Chroma's T5-XXL | `google/t5-v1_1-xxl`; and the Chroma repo declares itself Apache-2.0 over what it ships | `apache-2.0` | `https://huggingface.co/google/t5-v1_1-xxl` | **IDENTIFIED** |
| Boogu's Qwen3-VL-8B | candidate `Qwen/Qwen3-VL-8B-Instruct` (`apache-2.0`) — **not established**; the file is `mllm/` inside the Boogu snapshot | — | — | **UNDETERMINED** (repo-level Boogu Apache-2.0 covers it as distributed) |
| Boogu's FLUX.1 VAE | **AMBIGUOUS between `FLUX.1-dev` (non-commercial) and `FLUX.1-schnell` (Apache-2.0)** | — | — | **X5 — the highest-value unknown** |
| Anima's Qwen3-0.6B base | candidate `Qwen/Qwen3-0.6B-Base` (`apache-2.0`) — not established | — | — | **UNDETERMINED** |
| Anima's Qwen-Image VAE | candidate `Qwen/Qwen-Image` (`apache-2.0`, pinned in-repo at `release/real-weight-models.toml:48`) — not established | — | — | **UNDETERMINED** |
| Krea's Qwen3-VL-4B | — | — | — | **UNDETERMINED** |
| Mage's Qwen3-VL-4B | — | — | — | **UNDETERMINED** (and the Mage upstream itself is now 404 — X1) |
| Ideogram's Qwen3-VL-8B | — | — | — | **UNDETERMINED** |
| **Lens's gpt-oss-20b** | **`openai/gpt-oss-20b`** | `apache-2.0` | `https://huggingface.co/openai/gpt-oss-20b` | **RESOLVED** — see below |
| Z-Image's Qwen3 | — | — | — | **UNDETERMINED** |
| SD3.5's T5-XXL | `google/t5-v1_1-xxl` architecture; as distributed it sits inside the SD3.5 snapshot governed by `stability-ai-community` | `apache-2.0` (upstream) | `https://huggingface.co/google/t5-v1_1-xxl` | **IDENTIFIED** |
| **Kolors's ChatGLM3-6B** | **`THUDM/chatglm3-6b`** — and its live mirror **`zai-org/chatglm3-6b`**; both exist, both ship `MODEL_LICENSE`, neither declares a card `license` | `MODEL_LICENSE` (document title) | `https://raw.githubusercontent.com/THUDM/ChatGLM3/main/MODEL_LICENSE` (family 14's `text_url`) | **RESOLVED** — the census's "no `THUDM/chatglm3-6b` string exists anywhere in the repo" is a repo gap, not an upstream one |
| qwen-image's Qwen2.5-VL | candidate `Qwen/Qwen2.5-VL-7B-Instruct` (`apache-2.0`) — not established; `Qwen/Qwen-Image` ships an Apache-2.0 `LICENSE` covering what it distributes | — | — | **UNDETERMINED** (bundle-level Apache-2.0 established) |
| SCAIL-2's UMT5 / open-CLIP ViT-H / Wan2.1 VAE | candidates `google/umt5-xxl` (`apache-2.0`), `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (`mit`), Wan-AI Wan2.1 (`apache-2.0`) — none established; `zai-org/SCAIL-2` declares `mit` over what it ships | — | — | **UNDETERMINED** (bundle-level MIT established) |
| bernini's "stock Wan2.2" trio | **`Wan-AI/Wan2.2-T2V-A14B`** — the converter deliberately reads the standalone stock-Wan directory rather than Bernini's bundled copy (`mlx-gen-bernini/src/convert.rs:11`, `:56`) | `apache-2.0` | `https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B` | **IDENTIFIED from the code + upstream card** |
| krea-realtime's "stock Wan" trio | **Wan-AI Wan2.1-T2V-14B**, not Wan2.2 — `release/real-weight-models.toml:874` states the rehost is "Wan-2.1-T2V-14B weight-for-weight" and `:877` that each tier ships "the stock Wan `t5_encoder`/`vae`/`tokenizer`" | `apache-2.0` | `https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers` | **IDENTIFIED — and it corrects the census**, which grouped this under "stock Wan2.2" |

### The Lens resolution, in full

`SceneWorks/Lens` `README.md` is the only place in reach that itemises Lens's components, and it does
so as a table (retrieved 2026-08-02, `sha` `5c5521d4417a3cae55816929ece69319d1e7712a`):

> "| `text_encoder/`  | gpt-oss-20b (MXFP4), used encoder-only — from [`openai/gpt-oss-20b`] | Apache-2.0 |"

`Comfy-Org/Lens` independently lists `https://huggingface.co/openai/gpt-oss-20b (apache-2.0)` among
its sources, and `openai/gpt-oss-20b`'s own card declares `license: apache-2.0` (`gated: False`,
2026-08-02) — so the upstream card corroborates both redistributors. **`source_url` is
`https://huggingface.co/openai/gpt-oss-20b`.**

> **One extra fact for the row:** `openai/gpt-oss-20b` commits a `USAGE_POLICY` file alongside its
> Apache `LICENSE`. Its operative sentence is a compliance statement, not an incorporated policy
> document: "By using OpenAI gpt-oss-20b, you agree to comply with all applicable law."
> (`https://huggingface.co/openai/gpt-oss-20b/raw/main/USAGE_POLICY`, 2026-08-02). It names no
> restrictions and no external address. **`AcceptableUsePolicy` should NOT be derived from it** —
> "comply with applicable law" is not a use policy in the sense the variant carries.

## Shape (b) — named by package, never by repository (8 rows)

| component | outcome | evidence |
| --- | --- | --- |
| antelopev2 ArcFace `glintr100` | **repo UNDETERMINED, family settled.** No HF repository ships the antelopev2 pack under a stable id; the artifact on disk is produced by an in-house converter walking the onnx graph, so it corresponds to no upstream file byte-for-byte | family `insightface-research-only`, `text_url` `https://raw.githubusercontent.com/deepinsight/insightface/master/README.md` (as landed). Note `https://raw.githubusercontent.com/deepinsight/insightface/master/LICENSE` returns **404** — insightface publishes its restriction in the README, not a LICENSE file, which is why sc-16662 sourced it there |
| antelopev2 SCRFD-10g | same | same |
| facexlib `parsing_bisenet` | **RESOLVED** | `https://raw.githubusercontent.com/xinntao/facexlib/master/LICENSE` → `MIT License` / `Copyright (c) 2020 Xintao Wang`, retrieved 2026-08-02 → family `mit`. **Confirms the census: this is not insightface and `insightface-research-only` does not reach it** |
| the Boogu Edit checkpoint | **RESOLVED** | `Boogu/Boogu-Image-0.1-Edit` exists, `license: apache-2.0`, `gated: False` (2026-08-02). The example at `candle-gen-boogu/examples/boogu-edit.rs:58` defaults to `D:/models/Boogu-Image-0.1-Edit`, matching the repo name exactly |
| Hyper-FLUX 8-step LoRA | **RESOLVED** | `ByteDance/Hyper-SD` → `Hyper-FLUX.1-dev-8steps-lora.safetensors`, under the FLUX.1 [dev] Non-Commercial License per the repo's own `LICENSE.md`. See Job 1 row 67 |
| ostris TurboTime LoRA | **RESOLVED** | `ostris/ideogram_4_turbotime_lora`, declaring `ideogram-4-non-commercial`. See Job 1 row 68 |
| the lightx2v Wan step-distill diff-patch | **family settled, repo UNDETERMINED.** `candle-gen-scail2/src/adapters.rs` states the file "targets vanilla Wan2.1-I2V (`patch_embedding` in_dim **36**)", which narrows it to lightx2v's Wan2.1-I2V step-distill releases: `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v` and `…-720P-…`. Both declare `apache-2.0`; the 480P repo also ships `LICENSE.txt`. **Which of the two is loaded is not stated anywhere in the repository** | HF model-info API + `author=lightx2v` listing, 2026-08-02 |
| the `sat-scail2` Bias-Aware DPO LoRA | **UNDETERMINED — NOT FOUND.** Searched the HF model index for `sat-scail2` (0 results), `scail2` and `SCAIL` (40+ results reviewed; none named `sat-scail2` or published by a `sat*` author). The in-repo reference is a bare parenthetical at `candle-gen-scail2/src/adapters.rs:9` — "the **Bias-Aware DPO** refinement LoRA (`sat-scail2`)". `sat` most plausibly names a *key layout* (SwissArmyTransformer), not a publisher — **but that is a hypothesis, not a finding, and it is recorded as such** | |

---

# Gemma-2-2B-IT and the LTX text encoder — settled

## Gemma-2-2B-IT: **the Gemma Terms govern, on all three stated origins**

All four repositories, HF model-info API, 2026-08-02:

| repository | `license` | `gated` | licence files |
| --- | --- | --- | --- |
| `google/gemma-2-2b-it` — **the canonical Google id, never written in this repo** | `gemma` | **manual** | *(none committed; the card's tag is the declaration)* |
| `Efficient-Large-Model/gemma-2-2b-it` (PiD tests) | `gemma` | False | *(none)* |
| `SceneWorks/gemma-2-2b-it` (MLX prose) | `gemma` | False | `LICENSE`, `NOTICE`, `PROHIBITED_USE_POLICY.md` |
| "bundled in the SANA repo" (Candle SANA NOTICE) | — | — | the SANA `NOTICE`, corrected on `main` by `eef5166a`, attributes it to the Gemma Terms |

**All three stated origins declare the same identifier, `gemma`.** There is no conflict to resolve:
every one is a redistribution of Google's Gemma-2-2B-IT and every one carries family `gemma-terms`,
whose `text_url` is `https://ai.google.dev/gemma/terms`. `SceneWorks/gemma-2-2b-it`'s committed
`LICENSE` says so in its own first lines:

> "Gemma is provided under and subject to the Gemma Terms of Use found at
> https://ai.google.dev/gemma/terms"

— `https://huggingface.co/SceneWorks/gemma-2-2b-it/raw/main/LICENSE`, retrieved 2026-08-02.

**Which governs, for the table:** the **Gemma Terms of Use** — one family, one text, regardless of
which snapshot a consumer provisions. The only thing the choice of repository changes is the
component's `gated` value, and that is a per-snapshot distribution fact, not a licence fact (U7).

**Recommendation for the Rust pass:** one component row, `declared: "gemma"`, `source_url:
"https://ai.google.dev/gemma/terms"`, family `gemma-terms`. Set `gated: false` — the artifact this
codebase actually loads is one of the two ungated re-hosts, and a `true` sourced from Google's
`manual` gate would describe a repository no provider reads. **Record on the row that the canonical
licensor id (`google/gemma-2-2b-it`, `gated: manual`) is never written in the codebase**, since
that, not the licence, is the fact worth surfacing.

## The LTX text encoder: **the two backends do not disagree about the licence**

| repository | `license` | `gated` |
| --- | --- | --- |
| `google/gemma-3-12b-it` (Candle) | `gemma` | **manual** |
| `mlx-community/gemma-3-12b-it-bf16` (MLX) | `gemma` | False |

`mlx-community/gemma-3-12b-it-bf16` is a bf16 re-host of Google's checkpoint and declares the **same
identifier**. **Both resolve to family `gemma-terms` and the same canonical text.** The backends name
different *distributions*, not different licences, and the apparent divergence the census flagged is
a `gated` difference only.

**Which governs:** the Gemma Terms of Use, `https://ai.google.dev/gemma/terms`, for both. One
component row serves both backends — which is exactly the outcome the component-keyed design exists
to produce. Same `gated: false` recommendation and same rationale as above.

**Do not extend this to `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit`** — that repository declares
`apache-2.0`, not `gemma`, and is held open as **X11**.

---

# Granularity exceptions — where a repository-level row is genuinely wrong

Five. Everything else in the census is correctly keyed at the repository.

| # | repository | why per-file is required |
| --- | --- | --- |
| 1 | **`SceneWorks/Lens`, `SceneWorks/lens-mlx`** | Declares `mit`. Its **own README** assigns three different licences across its directories — `transformer/` MIT, `text_encoder/` + `tokenizer/` Apache-2.0, and `vae/` "FLUX.2-dev license", which is non-commercial. A repo-level `mit` row would state, in a disclosure surface, that a non-commercial artifact is MIT. **X3** |
| 2 | **`nvidia/PiD`** | The repo is NSCLv1 (non-commercial). Its `checkpoints/` directory also ships `ae.safetensors` (FLUX.1 VAE), `flux2_ae.safetensors`, `sdxl_vae.safetensors`, `sd3_vae/diffusion_pytorch_model.safetensors` and `QwenImage_VAE_2d.pth` — **five third parties' encoders redistributed inside one vendor's non-commercial repository** (file listing from the HF API, 2026-08-02). Which files a run touches depends on the `LoadSpec`'s latent space, so a single row cannot describe it |
| 3 | **`Boogu/Boogu-Image-0.1-*`** | Declares `apache-2.0` repo-wide while its README says the `vae/` is "the open-source **FLUX.1 VAE**". If that is FLUX.1 [dev]'s, one directory is not Apache-2.0. **X5** — and note this is a *conditional* exception: it becomes unnecessary the moment X5 resolves to schnell |
| 4 | **`Kwai-Kolors/Kolors-diffusers`** | Up to three documents over one repository: the card's `apache-2.0`, the committed `MODEL_LICENSE`, and the `text_encoder/` which is ChatGLM3-6B under **its** `MODEL_LICENSE` (family `chatglm3-model-license`). The ChatGLM3 split is required regardless of how **X6** resolves |
| 5 | **`ByteDance/Hyper-SD`** | One `LICENSE.md` scopes different clauses to different weight families — "For Flux.1-DEV-related models, please agree with the following license" introduces the FLUX text, and the repo also ships SDXL and SD1.5 LoRAs. Only the FLUX file is loaded here, so a *scoped* row is sufficient in practice; flagged because a future SDXL-LoRA row from the same repo would need its own |

**Not an exception, deliberately:** `ideogram-ai/ideogram-4-fp8` bundles the ostris TurboTime LoRA
inside its snapshot, but both artifacts declare the *same* identifier, so no per-file split is
required — only the `source_url` differs (ostris's card for the LoRA, Ideogram's `LICENSE.md` for
the base). Recorded so the transcription pass does not split it unnecessarily.

---

# Transcription notes for the Rust pass

1. **`gated` is a component field, not a family field** (sc-16898/U7). The values above are the HF
   `gated` field measured 2026-08-02: `auto` and `manual` both mean gated; `False` means not.
   Gated in this set: the three FLUX.2 repos (`auto`), `ideogram-ai/ideogram-4-fp8` (`auto`),
   `facebook/sam3` (`manual`), `black-forest-labs/FLUX.1-dev` (`auto`), `krea/Krea-2-Turbo` /
   `Krea-2-Raw` (`auto`), `stabilityai/stable-diffusion-3.5-large` (`auto`),
   `google/gemma-2-2b-it` and `google/gemma-3-12b-it` (`manual`). **Everything else measured for
   this note is ungated.**
2. **No row proposes `NonCommercialOutputs`.** Three of the texts read here address Outputs and all
   three permit commercial use of them (FLUX v2.1 §2(d), Ideogram §7, Wan's README). The
   `families.rs` test that fails on re-adding the term stays satisfied.
3. **Three `source_url`s must not be a repository card**: `nvidia/PiD` (the declaration is in the
   README body), `ByteDance/Hyper-SD` (in `LICENSE.md`, no card tag), and
   `circlestone-labs/Anima-Official-LoRAs` (in `LICENSE.md`, no card tag).
4. **Two `source_url`s do not resolve at all** — the six Mage-Flow repos and the two Lens repos
   (X1, X2). A dead `source_url` in a drift-checking table is worse than an explicit null; give
   those rows a null `source_url` and carry the second-hand provenance in prose.
5. **`declared` is verbatim.** Where the card supplies `license_name`, that string is `declared`
   (`flux-non-commercial-license`, `ideogram-4-non-commercial`, `experimental-research-only`,
   `flux-1-dev-non-commercial-license`, `flux-dev-non-commercial-license`, `kolors`). Where it
   supplies only `license`, that is `declared` (`apache-2.0`, `mit`, `gemma`, `openrail++`,
   `other`). Where neither exists, the document's own title is `declared` (`SAM License`,
   `MIT License`, `模型许可协议`) and the row says where it came from.

---

# Known holes — the rows sc-16665 deliberately did **not** write

**Added by the transcription pass (the Rust half of sc-16665).** The table that landed in
`crates/contracts/gen-core/src/license/components.rs` carries **71** `ComponentLicense` rows. The
census counted 90 distinct upstream artifacts and identified further redistributed components inside
them, so the table is deliberately incomplete. Every gap below is a checkpoint the media lane
demonstrably loads and for which this document could not produce a licence.

**Nothing was invented to close a gap** — not a row, not a family, not a `declared` string, not a
`source_url`. Omission is the designed outcome: sc-16669's ship gate reports a missing row and
**fails CI**, and per the epic's product constraint missing licence data never withholds a provider.
A hole is therefore loud and tracked; a fabricated row would be silent and wrong.

Each key below is pinned by the `known_holes_stay_absent` test in `license::components`, so adding
any of these later is a **deliberate diff**: the author has to delete the entry, which is the moment
to check that a primary source now exists.

## UNDETERMINED — the upstream repository could not be named (22 keys)

| key | what it is | see |
| --- | --- | --- |
| `flux2_klein_qwen3_text_encoder` | FLUX.2-klein's Qwen3 dense text encoder | [X12](#x12-eight-components-remain-undetermined) |
| `flux2_dev_mistral3_tower` | FLUX.2-dev's Mistral3 language tower | X12 |
| `flux2_dev_pixtral_vision_tower` | FLUX.2-dev's Pixtral vision tower | X12 |
| `flux2_dev_multimodal_projector` | FLUX.2-dev's multimodal projector (may be BFL-authored — unestablished) | X12 |
| `anima_qwen3_0_6b_text_encoder` | Anima's Qwen3-0.6B base text encoder | X12 |
| `anima_qwen_image_vae` | Anima's bundled Qwen-Image VAE | Job 2 |
| `boogu_qwen3_vl_8b` | Boogu's Qwen3-VL-8B condition encoder | Job 2 |
| `krea_qwen3_vl_4b` | Krea 2's Qwen3-VL-4B tower | X12 |
| `mage_qwen3_vl_4b` | Mage's Qwen3-VL-4B tower (and the Mage upstream itself 404s) | X12 |
| `ideogram_qwen3_vl_8b` | Ideogram 4's Qwen3-VL-8B tower | X12 |
| `z_image_qwen3_text_encoder` | Z-Image's Qwen3 tower | X12 |
| `qwen_image_qwen2_5_vl` | qwen-image's Qwen2.5-VL tower | Job 2 |
| `scail2_umt5`, `scail2_open_clip_vit_h`, `scail2_wan2_1_vae` | SCAIL-2's three bundled auxiliaries | Job 2 |
| `lightx2v_wan_step_distill_diff_patch` | narrowed to two lightx2v repositories; which one is loaded is stated nowhere | X12 |
| `sat_scail2_dpo_lora` | **no repository by that name exists** under any search; "sat" as a key layout is a hypothesis, not a finding | X12 |
| `pid_flux1_vae`, `pid_flux2_vae`, `pid_sdxl_vae`, `pid_sd3_vae`, `pid_qwen_image_vae` | the five third-party encoders `nvidia/PiD` redistributes inside its `checkpoints/` | [exception 2](#granularity-exceptions--where-a-repository-level-row-is-genuinely-wrong) |

The four vendor Qwen3-VL towers are one class of problem, not four: each vendor's card names the
architecture at most, never a repository id.

## NOT FOUND — the upstream is known and declares nothing re-readable (13 keys)

| key | why | see |
| --- | --- | --- |
| `kolors_controlnet_pose` | declares no licence of any kind and has no model card | [X7](#x7-kwai-kolorskolors-controlnet-pose-declares-no-licence-at-all) |
| `clip_vit_large_patch14` | the card declares nothing; the MIT the repo pins comes from OpenAI's *source* repository. **The most widely shared component in the catalog** | [X8](#x8-openaiclip-vit-large-patch14-declares-no-licence--and-the-repo-pins-it-as-mit) |
| `clip_vit_large_patch14_336` | same, as the Kolors IP-Adapter's image tower | X8 |
| `mage_flow`, `mage_flow_base`, `mage_flow_turbo`, `mage_flow_edit`, `mage_flow_edit_base`, `mage_flow_edit_turbo` | all six `microsoft/Mage-Flow*` repositories return 404 under an authenticated read; MIT survives only second-hand | [X1](#x1-every-microsoftmage-flow-repository-is-gone-from-the-hub-not-found) |
| `microsoft_lens`, `microsoft_lens_turbo`, `lens_transformer` | `microsoft/Lens*` likewise 404; `SceneWorks/Lens`'s `transformer/` inherits the same problem | [X2](#x2-microsoftlens-and-lens-turbo-are-gone-too--mit-survives-only-second-hand) |
| `krea2_pose_controlnet_beta` | declares `experimental-research-only` with no text behind it — **and it is SceneWorks' own repository**, so it is fixable rather than merely decidable | [X10](#x10-sceneworkskrea2-pose-controlnet-beta-declares-experimental-research-only-with-no-text) |

A dead `source_url` in a drift-checking table is worse than no row, and `ComponentLicense::source_url`
cannot be null — so these could not be written even as placeholders.

## AMBIGUOUS — two primary sources disagree, or the family is genuinely open (5 keys)

| key | why | see |
| --- | --- | --- |
| `kolors_diffusers` | card `apache-2.0` vs the committed `MODEL_LICENSE` | [X6](#x6-kolors-u6-carried-forward-unresolved-and-now-with-both-texts-read) / sc-16662 **U6** |
| `sceneworks_kolors_chatglm3_tokenizer` | declares `license_name: kolors`, so its family is whatever U6 settles | X6 |
| `flux2_dev_fun_controlnet_union` | ships BFL **v2.0** while BFL's own repos ship v2.1; one family or two is open | [X9](#x9-a-third-bfl-licence-version-is-in-circulation-v20-inside-an-alibaba-pai-controlnet) |
| `boogu_flux1_vae` | FLUX.1 \[dev\] (non-commercial) or FLUX.1-schnell (Apache-2.0) — never stated. **The epic's own worked example** | [X5](#x5-boogu-declares-apache-20-over-a-flux1-vae-it-does-not-identify) |
| `amoral_gemma_3_12b_v2_mlx_4bit` | a Gemma-3 derivative declaring `apache-2.0` | [X11](#x11-theclusteramoral-gemma-3-12b-v2-mlx-4bit-declares-apache-20-not-gemma) |

## NO DECLARED STRING — no string to transcribe as `declared` (4 keys)

Two shapes, one class: the upstream publishes prose with no identifier behind it, or no declared
string was recorded by a read. Neither is AMBIGUOUS — nothing disagrees, there is simply no string.
The distinction is what tells a reader which holes need research and which need a one-line decision.

**Prose with no identifier —** `antelopev2_arcface_glintr100` and `antelopev2_scrfd_10g`. The
*family* is settled (`insightface-research-only`, whose `text_url` is already landed), and these
carry the strictest terms in the catalog — but insightface publishes **no licence document for the
models**, only README prose. `ComponentLicense::declared` is "the licence identifier as declared
upstream, verbatim", and there is no identifier to transcribe; a prose fragment is a quote, not an
identifier. One line from Michael closes this.

**None recorded by a read —** `stable_diffusion_3_5_large_turbo` and `stable_diffusion_3_5_medium`.
Both are governed by the Stability AI Community License and sc-16662 recorded a declared string for
`stable-diffusion-3.5-large` only (`stabilityai-ai-community`). Assuming the siblings declare the
same string is plausible and unverified, so no row was written. **Reclassified from AMBIGUOUS**: the
stated reason was always "no declared string recorded", which is this class — nothing about these
two is in dispute. This is the cheapest hole on the list to close: one card read each.

## SECOND-HAND ONLY — the only declaration is this repository's own record (2 keys)

| key | why |
| --- | --- |
| `llama_joycaption_beta_one_hf_llava` | `release/real-weight-models.toml` records `license = "Llama 3.1 Community License"` for the pinned revision; the upstream card was never read. Same question class as X1 and X8 |
| `depth_anything_v2_small_hf` | the only statement is a provider crate's own rustdoc ("apache-2.0, ungated"). That is precisely the shape of the SANA defect the census catalogued (`candle-gen-sana/NOTICE`, corrected on `main` by `eef5166a`), so it was not transcribed as an upstream declaration |

## Judgement calls the transcription pass made, and flagged

Recorded here because they are places a reviewer should look first, not because they are unresolved
in the code:

1. **`source_url` follows the contract, not this note's tables.** `ComponentLicense::source_url` is
   documented as *the document `declared` was transcribed from*. Where `declared` is a card's
   `license_name` (the three FLUX.2 rows, the two Ideogram rows, the three FLUX.1-dev adapters), the
   row therefore points at the **card**, not at the `LICENSE.md` this note cites — the licence text
   lives once, on the family's `text_url`. Where `declared` is a document's own title (`SAM License`,
   `MIT License`, `The ChatGLM3-6B License`, `CircleStone Labs Non-Commercial License v1.0`) or lives
   in a README body (`nvidia/PiD`), the row points at that document.
2. **The two Gemma rows name Google's card and record `gated: true`**, following the epic's
   redistributed-component decision (the upstream card, not the redistributor's). This note's own
   recommendation was `gated: false` on the grounds that no provider reads Google's gated repository.
   Both readings are defensible; the row comments say so, and **Michael decides.** Nothing about the
   licence changes either way — one family, one text, per this note's settled section.
3. **U11 was applied, not answered.** `flux-non-commercial-v2-1` and `ideogram-4-non-commercial` each
   take `NoticeFileRequired` **and** `AttributionRequired` from a single clause, because both clauses
   name themselves an *attribution* notice — the textual hook sc-16662's U11 proposes, and the same
   call the landed `nvidia-open-model` already makes. If U11 settles the other way, all four families
   move together, which is the point of holding it open.
4. **`attribution` is `Some` exactly where the family requires it**, and a test enforces both
   directions. An attribution on a row whose licence asks for none would read as an obligation the
   text never stated.
5. **U8 was applied, not re-decided.** `meta-sam-license` carries §1(b)(ii)'s acknowledgement duty as
   a quoted `DeployerObligation` and **no** `AttributionRequired`, because the duty binds only on
   submitting research for publication while the typed term reads as unconditional. That is the rule
   sc-16662's U8 already set for `llama-3-1-community`'s 700M-MAU threshold — a typed term that
   overstates the text is a false transcription, so the condition is disclosed verbatim instead. Two
   families, one rule; no new decision was taken here. `facebook/sam3` therefore carries
   `attribution: None`, and its render's derived union no longer names an attribution obligation the
   licence does not impose on rendering.
