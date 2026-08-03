//! The **shared checkpoint table** for the media lane (sc-16665) — one [`ComponentLicense`] row per
//! upstream checkpoint, read by both media catalogs.
//!
//! # PROVISIONAL — gathered by an agent, not yet signed off by a human
//!
//! Every value here was transcribed from a primary source retrieved on **2026-08-02** by an
//! automated agent, and each one is backed by a quote, a source URL and a retrieval date in
//! `docs/licensing/sc-16665-checkpoint-licence-evidence.md`. That note is the sign-off document, and
//! it also carries the [known holes](#known-holes--rows-that-were-deliberately-not-written). **No
//! human has yet checked the transcriptions.**
//!
//! # Disclosure only
//!
//! Nothing here blocks, gates, degrades or withholds anything, and nothing added here ever should. A
//! row records what an upstream **names**, never what a user may or may not do. In particular
//! [`ComponentLicense::gated`] records that an upstream put a checkpoint behind an acceptance gate —
//! it does not gate anything in this system, and no code may branch on it.
//!
//! # Why one table, shared
//!
//! A licence is a property of the checkpoint, and the MLX and Candle engines load the **same**
//! checkpoints. Only the provider→component mapping legitimately differs per backend, and that lives
//! in each media catalog ([`ProviderComponents`](super::ProviderComponents); sc-16666 for MLX,
//! sc-16667 for Candle). Keeping the rows here — tensor-neutral, beside the families they resolve
//! against and beside [`license_table_conformance_errors`](super::license_table_conformance_errors)
//! — means the two catalogs cannot drift into two different answers about one upstream artifact, and
//! a checkpoint that no provider id loads at all (the PiD decoder, the SAM stacks) still gets a row.
//!
//! # Granularity: one row per **upstream repository**
//!
//! A licence attaches to a repository, so that is the row. Two consequences the census forced:
//!
//! * **A component redistributed inside another party's snapshot is keyed on its own upstream
//!   repository, not on the redistributor's.** [`ComponentLicense::source_url`] says why in general;
//!   sc-16665 proved it concretely: `Comfy-Org/Lens` attributes its bundled FLUX.2-dev VAE as
//!   `apache-2.0`, while BFL's own gated `LICENSE.md` for that checkpoint is the FLUX Non-Commercial
//!   License v2.1. A redistributor is verifiably wrong about a component it bundles. So the Lens
//!   text encoder is the [`gpt_oss_20b`](GPT_OSS_20B) row and its VAE is the
//!   [`flux2_dev`](FLUX2_DEV) row.
//! * **A repository re-published in a second layout by the same publisher is one row.** `Wan-AI`'s
//!   `-Diffusers` siblings carry the same weights under the same declaration; the row names the base
//!   repository and its comment names the sibling. Two genuinely different models in one census line
//!   (Wan2.1-VACE 1.3B and 14B) are two rows.
//!
//! Five checkpoints genuinely need finer rows than their repository, and each is handled where it
//! occurs: `nvidia/PiD` (see [`nvidia_pid_students`](NVIDIA_PID_STUDENTS)), `ByteDance/Hyper-SD`
//! (see [`hyper_sd_flux1_dev_8steps_lora`](HYPER_SD_FLUX1_DEV_8STEPS_LORA)),
//! `Kwai-Kolors/Kolors-diffusers` (see [`chatglm3_6b`](CHATGLM3_6B)), `SceneWorks/Lens` (see
//! [`gpt_oss_20b`](GPT_OSS_20B)) and `Boogu/Boogu-Image-0.1-*` (see
//! [`boogu_image_0_1_base`](BOOGU_IMAGE_0_1_BASE)).
//!
//! # `source_url` picks the document that carries `declared`
//!
//! Two shapes, and which applies is decided by where the declaration lives rather than by taste —
//! the same rule [`ComponentLicense::source_url`] states:
//!
//! * the **repository card**, when `declared` is the card's own `license_name` or `license` tag
//!   (`apache-2.0`, `flux-non-commercial-license`, `openrail++`, `gemma`, …) — the overwhelming
//!   majority of rows;
//! * a **licence document beside the weights**, when the card declares nothing and the artifact's
//!   only declaration is that document's own title (`SAM License`, `MIT License`, `CircleStone Labs
//!   Non-Commercial License v1.0`, `The ChatGLM3-6B License`), or when the declaration lives in the
//!   README body rather than the metadata (`nvidia/PiD`).
//!
//! # `attribution` is set exactly when the family requires it
//!
//! `Some(…)` on every row whose resolved family imposes
//! [`AttributionRequired`](super::LicenseTerm::AttributionRequired), `None`
//! on every row whose family does not. The conformance gate enforces the first half; the test module
//! here enforces the second, so an attribution never silently implies an obligation the text does
//! not state. Where a licence prescribes the notice's wording, the prescribed wording is used
//! verbatim and **in full** — one text does, `ideogram-4-non-commercial` §3(iii), and both its rows
//! share one `IDEOGRAM_4_PRESCRIBED_NOTICE` constant so neither can drift into a fragment of it.
//! Otherwise the string follows the house shape the
//! audio lane established, naming the artifact, the publisher as the repository publishes it, and
//! the licence.
//!
//! # Known holes — rows that were deliberately **not** written
//!
//! The evidence is incomplete, and a fabricated row is worse than a missing one: sc-16669's ship
//! gate reports a missing row and **fails CI**, and per the epic's product constraint it never
//! withholds a provider. So a hole is loud and tracked. `known_holes_stay_absent` in this module's
//! test block pins every one of them by key, so a later row for any of these is a deliberate diff
//! rather than a silent addition. The reason classes:
//!
//! | class | what it means | examples |
//! | --- | --- | --- |
//! | UNDETERMINED | the upstream repository could not be named | FLUX.2's Mistral3 / Pixtral / Qwen3 towers, Anima's Qwen3-0.6B, the four vendor Qwen3-VL towers, PiD's five bundled VAEs |
//! | NOT FOUND | the upstream is known and declares nothing re-readable | `Kwai-Kolors/Kolors-ControlNet-Pose`, the six `microsoft/Mage-Flow*` repos (404), `microsoft/Lens*` (404), `openai/clip-vit-large-patch14` |
//! | AMBIGUOUS | two primary sources disagree, or the family is genuinely open | `Kwai-Kolors/Kolors-diffusers` (U6), `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` (BFL v2.0 vs v2.1), Boogu's FLUX.1 VAE (dev vs schnell) |
//! | NO DECLARED STRING | there is no string to transcribe as `declared` — the upstream publishes prose with no identifier behind it, or no declared string was recorded by a read | the two antelopev2 rows; the SD3.5 `-large-turbo` and `-medium` siblings |
//! | SECOND-HAND ONLY | the only declaration is this repository's own record | `fancyfeast/llama-joycaption-beta-one-hf-llava`, `depth-anything/Depth-Anything-V2-Small-hf` |
//!
//! Six questions are on Michael's sign-off list and are **not** answered here: the Boogu FLUX.1-VAE
//! dev-vs-schnell question, whether second-hand MIT is acceptable for the vanished `microsoft/*`
//! repositories and what `source_url` a dead repository gets (sc-16984), whether
//! `openai/clip-vit-large-patch14`'s `declared` is blank or `"MIT"` with GitHub provenance stated,
//! Kolors U6, whether BFL v2.0 and v2.1 are one family or two, and U11.

use super::ComponentLicense;

/// `circlestone-labs/Anima` — three separately-trained DiT files in one repository, all under one
/// declaration.
pub const ANIMA: ComponentLicense = ComponentLicense {
    component: "anima",
    source_url: "https://huggingface.co/circlestone-labs/Anima",
    gated: false,
    declared: "circlestone-labs-non-commercial-license",
    family: "circlestone-labs-non-commercial",
    attribution: Some(
        "Anima © CircleStone Labs — licensed under the CircleStone Labs Non-Commercial License v1.0",
    ),
    retrieved: "2026-08-02",
};

/// `circlestone-labs/Anima-Official-LoRAs`. The card carries **no** `license` field; the declaration
/// is the title of the `LICENSE.md` committed beside the weights, which is why `source_url` is that
/// file rather than the card.
pub const ANIMA_OFFICIAL_LORAS: ComponentLicense = ComponentLicense {
    component: "anima_official_loras",
    source_url: "https://huggingface.co/circlestone-labs/Anima-Official-LoRAs/raw/main/LICENSE.md",
    gated: false,
    declared: "CircleStone Labs Non-Commercial License v1.0",
    family: "circlestone-labs-non-commercial",
    attribution: Some(
        "Anima Official LoRAs © CircleStone Labs — licensed under the CircleStone Labs \
         Non-Commercial License v1.0",
    ),
    retrieved: "2026-08-02",
};

/// `ByteDance/Bernini-Diffusers` — one combined index supplying six prefixes (two DiT experts, the
/// planner backbone, the connector, the ViT decoder head and the MAR mask tokens). One repository,
/// one declaration, so one row.
pub const BERNINI_DIFFUSERS: ComponentLicense = ComponentLicense {
    component: "bernini_diffusers",
    source_url: "https://huggingface.co/ByteDance/Bernini-Diffusers",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Bernini © ByteDance — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Boogu/Boogu-Image-0.1-Base` — the Boogu-authored DiT.
///
/// **Scoped deliberately.** The repository declares `apache-2.0` repo-wide, and its own README says
/// the bundled `vae/` is "the open-source **FLUX.1 VAE**" without saying whether that is FLUX.1
/// \[dev\] (non-commercial) or FLUX.1-schnell (Apache-2.0). This row covers the Boogu DiT; the
/// bundled VAE and the bundled Qwen3-VL-8B condition encoder are separate components and are
/// **omitted** — see the module's known-holes note. This is the epic's worked example, and the
/// dev-vs-schnell question is the single highest-value unknown in the set.
pub const BOOGU_IMAGE_0_1_BASE: ComponentLicense = ComponentLicense {
    component: "boogu_image_0_1_base",
    source_url: "https://huggingface.co/Boogu/Boogu-Image-0.1-Base",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Boogu-Image-0.1-Base © Boogu — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Boogu/Boogu-Image-0.1-Edit`. Same scoping as [`BOOGU_IMAGE_0_1_BASE`]. The census recorded this
/// checkpoint's repository as UNDETERMINED; the evidence pass established that the repository exists
/// and matches the root `candle-gen-boogu/examples/boogu-edit.rs` defaults to.
pub const BOOGU_IMAGE_0_1_EDIT: ComponentLicense = ComponentLicense {
    component: "boogu_image_0_1_edit",
    source_url: "https://huggingface.co/Boogu/Boogu-Image-0.1-Edit",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Boogu-Image-0.1-Edit © Boogu — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Boogu/Boogu-Image-0.1-Turbo`. Same scoping as [`BOOGU_IMAGE_0_1_BASE`].
pub const BOOGU_IMAGE_0_1_TURBO: ComponentLicense = ComponentLicense {
    component: "boogu_image_0_1_turbo",
    source_url: "https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Boogu-Image-0.1-Turbo © Boogu — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// ChatGLM3-6B — the text encoder redistributed inside `Kwai-Kolors/Kolors-diffusers`.
///
/// **A granularity exception, and the one that stands regardless of how anything else resolves.**
/// Kolors ships up to three documents over one repository (its card's `apache-2.0`, its committed
/// `MODEL_LICENSE`, and this encoder's own licence), and whichever way sc-16662's **U6** is settled,
/// the text encoder is governed by ChatGLM3's `MODEL_LICENSE` and not by the Kolors repository. The
/// Kolors repository row itself is an AMBIGUOUS hole; this split is not.
///
/// `THUDM/chatglm3-6b` and its live mirror `zai-org/chatglm3-6b` both exist and neither declares a
/// card `license`, so `declared` is the governing document's own title and `source_url` is that
/// document.
pub const CHATGLM3_6B: ComponentLicense = ComponentLicense {
    component: "chatglm3_6b",
    source_url: "https://raw.githubusercontent.com/THUDM/ChatGLM3/main/MODEL_LICENSE",
    gated: false,
    declared: "The ChatGLM3-6B License",
    family: "chatglm3-model-license",
    attribution: Some("ChatGLM3-6B © THUDM — licensed under The ChatGLM3-6B License"),
    retrieved: "2026-08-02",
};

/// `lodestones/Chroma1-Base`.
pub const CHROMA1_BASE: ComponentLicense = ComponentLicense {
    component: "chroma1_base",
    source_url: "https://huggingface.co/lodestones/Chroma1-Base",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Chroma1-Base © lodestones — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `lodestones/Chroma1-Flash` — the few-step distilled variant.
pub const CHROMA1_FLASH: ComponentLicense = ComponentLicense {
    component: "chroma1_flash",
    source_url: "https://huggingface.co/lodestones/Chroma1-Flash",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Chroma1-Flash © lodestones — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `lodestones/Chroma1-HD`. The card states the model is "based on **FLUX.1-schnell**. It is fully
/// **Apache 2.0 licensed**" — the Apache declaration covers the DiT and the `vae/` and
/// `text_encoder/` the repository ships. The T5-XXL inside it is keyed separately as
/// [`T5_V1_1_XXL`], per the redistributed-component rule.
pub const CHROMA1_HD: ComponentLicense = ComponentLicense {
    component: "chroma1_hd",
    source_url: "https://huggingface.co/lodestones/Chroma1-HD",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Chroma1-HD © lodestones — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` — the bigG tokenizer / text tower **SDXL and SD3.5**
/// both condition on.
///
/// SD3.5's `text_encoder_2/` is a redistributed copy of this tower, keyed here per the
/// redistributed-component rule; `candle-gen-sd3/src/clip_tokenizer.rs` is where that name is
/// recorded. Naming only the SDXL use under-described a row every registered SD3.5 id reaches on
/// both backends.
pub const CLIP_VIT_BIGG_14_LAION2B: ComponentLicense = ComponentLicense {
    component: "clip_vit_bigg_14_laion2b",
    source_url: "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    gated: false,
    declared: "mit",
    family: "mit",
    attribution: Some("CLIP ViT-bigG/14 (laion2B-39B-b160k) © LAION — licensed under MIT"),
    retrieved: "2026-08-02",
};

/// `xinsir/controlnet-openpose-sdxl-1.0` — the optional pose ControlNet on the InstantID path.
pub const CONTROLNET_OPENPOSE_SDXL_1_0: ComponentLicense = ComponentLicense {
    component: "controlnet_openpose_sdxl_1_0",
    source_url: "https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("ControlNet OpenPose SDXL 1.0 © xinsir — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `QuanSun/EVA-CLIP` → `EVA02_CLIP_L_336_psz14_s6B.pt`, the PuLID visual tower. The upstream is
/// named nowhere in either provider crate — only in `crates/media/mlx-gen/tools/convert_eva_clip.py`.
pub const EVA_CLIP: ComponentLicense = ComponentLicense {
    component: "eva_clip",
    source_url: "https://huggingface.co/QuanSun/EVA-CLIP",
    gated: false,
    declared: "mit",
    family: "mit",
    attribution: Some("EVA-CLIP © QuanSun — licensed under MIT"),
    retrieved: "2026-08-02",
};

/// facexlib's `parsing_bisenet`, shipped as `bisenet_parsing.safetensors` — the face-parsing model
/// the PuLID path uses to whiten non-face regions.
///
/// **Not insightface**, which is why `insightface-research-only` does not reach it: facexlib is a
/// different upstream with its own MIT grant. There is no Hugging Face repository; the declaration
/// is the first line of the project's `LICENSE`, which is also where `source_url` points.
pub const FACEXLIB_PARSING_BISENET: ComponentLicense = ComponentLicense {
    component: "facexlib_parsing_bisenet",
    source_url: "https://raw.githubusercontent.com/xinntao/facexlib/master/LICENSE",
    gated: false,
    declared: "MIT License",
    family: "mit",
    attribution: Some("facexlib © 2020 Xintao Wang — licensed under MIT"),
    retrieved: "2026-08-02",
};

/// `black-forest-labs/FLUX.1-dev`. The repository's own `ae.safetensors` VAE is BFL's, not a
/// third party's, so it needs no separate row; the bundled CLIP-L and T5-XXL do — see
/// [`T5_V1_1_XXL`] and the module's known-holes note for the CLIP-L.
pub const FLUX1_DEV: ComponentLicense = ComponentLicense {
    component: "flux1_dev",
    source_url: "https://huggingface.co/black-forest-labs/FLUX.1-dev",
    gated: true,
    declared: "flux-1-dev-non-commercial-license",
    family: "flux-1-dev-non-commercial",
    attribution: None,
    retrieved: "2026-08-02",
};

/// `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` — one of three adapters a reader would expect
/// to be permissive and which instead declare BFL's non-commercial licence. The card's
/// `license_name` carries the declaration, so `source_url` is the card rather than the
/// `license_link` it points at (that is the family's `text_url`).
pub const FLUX1_DEV_CONTROLNET_UNION_PRO_2_0: ComponentLicense = ComponentLicense {
    component: "flux1_dev_controlnet_union_pro_2_0",
    source_url: "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
    gated: false,
    declared: "flux-1-dev-non-commercial-license",
    family: "flux-1-dev-non-commercial",
    attribution: None,
    retrieved: "2026-08-02",
};

/// `black-forest-labs/FLUX.1-schnell` — Apache-2.0 and ungated, unlike its `-dev` sibling.
pub const FLUX1_SCHNELL: ComponentLicense = ComponentLicense {
    component: "flux1_schnell",
    source_url: "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("FLUX.1 [schnell] © Black Forest Labs — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `black-forest-labs/FLUX.2-dev`. Also the row a `SceneWorks/Lens` render resolves through for its
/// `vae/`, which that repository's own README attributes to this checkpoint.
///
/// The bundled Mistral3 language tower, Pixtral vision tower and multimodal projector are separate
/// components whose upstreams could not be named; they are omitted — see the module's known-holes
/// note.
pub const FLUX2_DEV: ComponentLicense = ComponentLicense {
    component: "flux2_dev",
    source_url: "https://huggingface.co/black-forest-labs/FLUX.2-dev",
    gated: true,
    declared: "flux-non-commercial-license",
    family: "flux-non-commercial-v2-1",
    attribution: Some(
        "FLUX.2 [dev] © Black Forest Labs — licensed under the FLUX Non-Commercial License v2.1",
    ),
    retrieved: "2026-08-02",
};

/// `black-forest-labs/FLUX.2-klein-9B`. Its `LICENSE.md` is byte-identical to [`FLUX2_DEV`]'s
/// (both SHA-256 `468D9F4332C0C895…`).
pub const FLUX2_KLEIN_9B: ComponentLicense = ComponentLicense {
    component: "flux2_klein_9b",
    source_url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B",
    gated: true,
    declared: "flux-non-commercial-license",
    family: "flux-non-commercial-v2-1",
    attribution: Some(
        "FLUX.2 klein 9B © Black Forest Labs — licensed under the FLUX Non-Commercial License v2.1",
    ),
    retrieved: "2026-08-02",
};

/// `black-forest-labs/FLUX.2-klein-9b-kv` — a separately distilled checkpoint. Its licence file is a
/// different blob from [`FLUX2_DEV`]'s, with the same title.
pub const FLUX2_KLEIN_9B_KV: ComponentLicense = ComponentLicense {
    component: "flux2_klein_9b_kv",
    source_url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv",
    gated: true,
    declared: "flux-non-commercial-license",
    family: "flux-non-commercial-v2-1",
    attribution: Some(
        "FLUX.2 klein 9B (KV) © Black Forest Labs — licensed under the FLUX Non-Commercial \
         License v2.1",
    ),
    retrieved: "2026-08-02",
};

/// Gemma-2-2B-IT — the PiD caption encoder and the SANA text encoder.
///
/// Three redistributions are named in this codebase (`SceneWorks/gemma-2-2b-it`,
/// `Efficient-Large-Model/gemma-2-2b-it`, and a copy bundled in the SANA repository) and the
/// canonical Google id is written nowhere. All three declare the **same** identifier, so there is no
/// licence conflict to resolve: one family, one text, whichever snapshot a consumer provisions.
///
/// `source_url` and `gated` name Google's repository, per the redistributed-component rule — that is
/// the upstream, and `gated` describes the licensor's own distribution. The two re-hosts this
/// codebase actually reads are **ungated**; whether the row should describe the licensor or the
/// provisioned snapshot is on the sign-off list.
pub const GEMMA_2_2B_IT: ComponentLicense = ComponentLicense {
    component: "gemma_2_2b_it",
    source_url: "https://huggingface.co/google/gemma-2-2b-it",
    gated: true,
    declared: "gemma",
    family: "gemma-terms",
    attribution: None,
    retrieved: "2026-08-02",
};

/// Gemma-3-12B-IT — the LTX-2.3 text encoder.
///
/// The census flagged that the two backends name different upstreams (Candle
/// `google/gemma-3-12b-it`, MLX the bf16 re-host `mlx-community/gemma-3-12b-it-bf16`). They are not
/// two licences: the re-host declares the same `gemma` identifier and resolves to the same text.
/// One row serves both backends, keyed on the licensor, with the same `gated` caveat as
/// [`GEMMA_2_2B_IT`].
///
/// Do **not** extend this row to `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit`, the MLX-only
/// uncensored prompt enhancer: that repository declares `apache-2.0`, not `gemma`, and is an
/// AMBIGUOUS hole.
pub const GEMMA_3_12B_IT: ComponentLicense = ComponentLicense {
    component: "gemma_3_12b_it",
    source_url: "https://huggingface.co/google/gemma-3-12b-it",
    gated: true,
    declared: "gemma",
    family: "gemma-terms",
    attribution: None,
    retrieved: "2026-08-02",
};

/// `openai/gpt-oss-20b` — the Lens text encoder, used encoder-only.
///
/// **A granularity exception in action.** `SceneWorks/Lens` declares `mit` while its own README
/// assigns three different licences across three directories; this row is the `text_encoder/` half
/// of that split, and the `vae/` half is [`FLUX2_DEV`]. The `transformer/` half has no row: its MIT
/// survives only second-hand from the deleted `microsoft/Lens`, so there is no `source_url` a drift
/// job could re-read.
///
/// The repository also commits a `USAGE_POLICY` file, whose operative sentence is
/// "By using OpenAI gpt-oss-20b, you agree to comply with all applicable law." That names no
/// restrictions and no address, so no [`AcceptableUsePolicy`](super::LicenseTerm::AcceptableUsePolicy)
/// is derived from it — a compliance statement is not a use policy in the sense the variant carries.
pub const GPT_OSS_20B: ComponentLicense = ComponentLicense {
    component: "gpt_oss_20b",
    source_url: "https://huggingface.co/openai/gpt-oss-20b",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("gpt-oss-20b © OpenAI — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// The ByteDance Hyper-FLUX 8-step LoRA, `Hyper-FLUX.1-dev-8steps-lora.safetensors`.
///
/// **A granularity exception.** `ByteDance/Hyper-SD` declares no card licence and commits one
/// `LICENSE.md` that scopes different clauses to different weight families — it opens
/// "For Flux.1-DEV-related models, please agree with the following license", then the FLUX.1 \[dev\]
/// text. The same repository also ships SDXL and SD1.5 LoRAs under a different part of that file, so
/// the row is keyed on the FLUX artifact rather than on the repository. A future SDXL-LoRA row from
/// this repository needs its own key and its own reading.
pub const HYPER_SD_FLUX1_DEV_8STEPS_LORA: ComponentLicense = ComponentLicense {
    component: "hyper_sd_flux1_dev_8steps_lora",
    source_url: "https://huggingface.co/ByteDance/Hyper-SD/raw/main/LICENSE.md",
    gated: false,
    declared: "FLUX.1 [dev] Non-Commercial License",
    family: "flux-1-dev-non-commercial",
    attribution: None,
    retrieved: "2026-08-02",
};

/// The attribution notice `ideogram-4-non-commercial` §3(iii) prescribes — **complete**, not a
/// leading fragment.
///
/// Both Ideogram rows carry this one string because the licence prescribes the notice's wording
/// rather than leaving it to the distributor: §3(iii) requires "the following attribution notice
/// within a \"Notice\" text file that accompanies such copy", and what follows, between the source's
/// own curly quotes, is exactly this text. Read from `ideogram-ai/ideogram-4-fp8` `LICENSE.md` at
/// `sha` `ee79a7237b519f1402ceacf952f30c8a31ec5073` on 2026-08-02 under an authenticated session.
///
/// Shared rather than written twice so the two rows cannot drift from each other, and so a job
/// comparing the landed string against the upstream clause has one place to compare.
const IDEOGRAM_4_PRESCRIBED_NOTICE: &str = "Ideogram 4 is provided under and subject to the \
    Ideogram Non-Commercial Model Agreement available at \
    https://github.com/ideogram-oss/ideogram-4/model_licenses/LICENSE-IDEOGRAM-4-NON-COMMERCIAL. \
    All rights reserved. Copyright © Ideogram, Inc.";

/// `ideogram-ai/ideogram-4-fp8` — the conditional DiT, the unconditional DiT, the text encoder and
/// the VAE.
///
/// The snapshot also bundles the ostris TurboTime LoRA, and that is **not** a granularity exception:
/// both artifacts declare the *same* identifier, so only [`source_url`](ComponentLicense::source_url)
/// differs and [`IDEOGRAM_4_TURBOTIME_LORA`] carries it. The bundled Qwen3-VL-8B tower is a separate
/// component whose upstream could not be named; it is omitted.
pub const IDEOGRAM_4_FP8: ComponentLicense = ComponentLicense {
    component: "ideogram_4_fp8",
    source_url: "https://huggingface.co/ideogram-ai/ideogram-4-fp8",
    gated: true,
    declared: "ideogram-4-non-commercial",
    family: "ideogram-4-non-commercial",
    attribution: Some(IDEOGRAM_4_PRESCRIBED_NOTICE),
    retrieved: "2026-08-02",
};

/// `ostris/ideogram_4_turbotime_lora`. The LoRA is **not** independently licensed: its card declares
/// the model vendor's agreement and links Ideogram's `LICENSE.md`.
pub const IDEOGRAM_4_TURBOTIME_LORA: ComponentLicense = ComponentLicense {
    component: "ideogram_4_turbotime_lora",
    source_url: "https://huggingface.co/ostris/ideogram_4_turbotime_lora",
    gated: false,
    declared: "ideogram-4-non-commercial",
    family: "ideogram-4-non-commercial",
    attribution: Some(IDEOGRAM_4_PRESCRIBED_NOTICE),
    retrieved: "2026-08-02",
};

/// `InstantX/InstantID` — the IdentityNet ControlNet and IP-Adapter of the identity overlay.
pub const INSTANTID: ComponentLicense = ComponentLicense {
    component: "instantid",
    source_url: "https://huggingface.co/InstantX/InstantID",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("InstantID © InstantX — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `h94/IP-Adapter` — the optional SDXL IP-Adapter.
pub const IP_ADAPTER: ComponentLicense = ComponentLicense {
    component: "ip_adapter",
    source_url: "https://huggingface.co/h94/IP-Adapter",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("IP-Adapter © h94 — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Kwai-Kolors/Kolors-IP-Adapter-Plus`. Unlike its `Kolors-ControlNet-Pose` sibling, which declares
/// nothing at all, this repository declares `apache-2.0` — so the absence next door is one
/// repository's omission and not a Kwai-Kolors-wide policy. Its bundled CLIP-ViT-L/14-336 image
/// tower is a separate component and is omitted (that upstream declares no licence — see the
/// known-holes note).
pub const KOLORS_IP_ADAPTER_PLUS: ComponentLicense = ComponentLicense {
    component: "kolors_ip_adapter_plus",
    source_url: "https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Kolors IP-Adapter-Plus © Kwai-Kolors — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `krea/Krea-2-Raw`.
pub const KREA_2_RAW: ComponentLicense = ComponentLicense {
    component: "krea_2_raw",
    source_url: "https://huggingface.co/krea/Krea-2-Raw",
    gated: true,
    declared: "krea-2-community-license",
    family: "krea-2-community",
    attribution: Some(
        "Krea 2 Raw © Krea — licensed under the Krea 2 Community License Agreement v.1",
    ),
    retrieved: "2026-08-02",
};

/// `krea/Krea-2-Turbo`. Its bundled Qwen3-VL-4B tower is a separate component whose upstream could
/// not be named; it is omitted.
pub const KREA_2_TURBO: ComponentLicense = ComponentLicense {
    component: "krea_2_turbo",
    source_url: "https://huggingface.co/krea/Krea-2-Turbo",
    gated: true,
    declared: "krea-2-community-license",
    family: "krea-2-community",
    attribution: Some(
        "Krea 2 Turbo © Krea — licensed under the Krea 2 Community License Agreement v.1",
    ),
    retrieved: "2026-08-02",
};

/// `krea/krea-realtime-video`, loaded through the `SceneWorks/krea-realtime-14b-mlx` re-host. Its
/// "stock Wan" trio is **Wan 2.1**, not Wan 2.2 — see [`WAN2_1_T2V_14B_DIFFUSERS`], which corrects
/// the census.
pub const KREA_REALTIME_VIDEO: ComponentLicense = ComponentLicense {
    component: "krea_realtime_video",
    source_url: "https://huggingface.co/krea/krea-realtime-video",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("krea-realtime-video © Krea — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Lightricks/LTX-2.3` — one bundled file carrying the DiT, the video VAE, the audio VAE, the
/// vocoder and the connector, plus three separate upsampler files. The crate targets LTX-**2.3**,
/// which declares the LTX-**2** community licence. Its Gemma-3-12B-IT text encoder is
/// [`GEMMA_3_12B_IT`].
pub const LTX_2_3: ComponentLicense = ComponentLicense {
    component: "ltx_2_3",
    source_url: "https://huggingface.co/Lightricks/LTX-2.3",
    gated: false,
    declared: "ltx-2-community-license-agreement",
    family: "ltx-2-community",
    attribution: Some("LTX-2.3 © Lightricks — licensed under the LTX-2 Community License"),
    retrieved: "2026-08-02",
};

/// `genmo/mochi-1-preview`.
pub const MOCHI_1_PREVIEW: ComponentLicense = ComponentLicense {
    component: "mochi_1_preview",
    source_url: "https://huggingface.co/genmo/mochi-1-preview",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Mochi 1 (preview) © Genmo — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `nvidia/PiD` — the eight student latent decoders, one per latent space × resolution tier.
///
/// **A granularity exception, and the widest overlay in the catalog**: the PiD decode seam is wired
/// into fourteen provider crates per backend and none of them repeats the terms. Today whether a
/// render touched PiD is knowable only from the `LoadSpec`; this row is what makes it disclosable.
///
/// The key is scoped to NVIDIA's own students **deliberately**. The repository's `checkpoints/`
/// directory also redistributes five third parties' encoders (`ae.safetensors`,
/// `flux2_ae.safetensors`, `sdxl_vae.safetensors`, `sd3_vae/`, `QwenImage_VAE_2d.pth`), which file a
/// run touches depends on the `LoadSpec`'s latent space, and none of those five upstreams could be
/// named. They are separate components and are omitted rather than swept under this row — see the
/// module's known-holes note.
///
/// The card carries **no** `license` metadata field at all; the declaration is in the README body,
/// which is why `source_url` is the README.
pub const NVIDIA_PID_STUDENTS: ComponentLicense = ComponentLicense {
    component: "nvidia_pid_students",
    source_url: "https://huggingface.co/nvidia/PiD/raw/main/README.md",
    gated: false,
    declared: "NSCLv1",
    family: "nvidia-nsclv1",
    attribution: Some("PiD © NVIDIA Corporation — licensed under the NVIDIA License (NSCLv1)"),
    retrieved: "2026-08-02",
};

/// `guozinan/PuLID` → `pulid_flux_v0.9.1.safetensors`.
pub const PULID: ComponentLicense = ComponentLicense {
    component: "pulid",
    source_url: "https://huggingface.co/guozinan/PuLID",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("PuLID © guozinan — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Qwen/Qwen-Image` — the DiT, plus the VAE the Anima and Krea paths reuse. The repository ships an
/// Apache-2.0 `LICENSE` in addition to its card tag. Its Qwen2.5-VL tower is a separate component
/// whose upstream could not be established; it is omitted, and the repository's own Apache-2.0
/// declaration covers what it distributes.
pub const QWEN_IMAGE: ComponentLicense = ComponentLicense {
    component: "qwen_image",
    source_url: "https://huggingface.co/Qwen/Qwen-Image",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Qwen-Image © Qwen — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Qwen/Qwen-Image-2512`.
pub const QWEN_IMAGE_2512: ComponentLicense = ComponentLicense {
    component: "qwen_image_2512",
    source_url: "https://huggingface.co/Qwen/Qwen-Image-2512",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Qwen-Image-2512 © Qwen — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union`.
pub const QWEN_IMAGE_2512_FUN_CONTROLNET_UNION: ComponentLicense = ComponentLicense {
    component: "qwen_image_2512_fun_controlnet_union",
    source_url: "https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some(
        "Qwen-Image-2512-Fun-Controlnet-Union © alibaba-pai — licensed under Apache-2.0",
    ),
    retrieved: "2026-08-02",
};

/// `Qwen/Qwen-Image-Edit`.
pub const QWEN_IMAGE_EDIT: ComponentLicense = ComponentLicense {
    component: "qwen_image_edit",
    source_url: "https://huggingface.co/Qwen/Qwen-Image-Edit",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Qwen-Image-Edit © Qwen — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `lightx2v/Qwen-Image-Edit-2511-Lightning` — the optional few-step distill LoRA.
pub const QWEN_IMAGE_EDIT_2511_LIGHTNING: ComponentLicense = ComponentLicense {
    component: "qwen_image_edit_2511_lightning",
    source_url: "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Qwen-Image-Edit-2511-Lightning © lightx2v — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `lightx2v/Qwen-Image-Lightning` — the optional few-step distill LoRA.
pub const QWEN_IMAGE_LIGHTNING: ComponentLicense = ComponentLicense {
    component: "qwen_image_lightning",
    source_url: "https://huggingface.co/lightx2v/Qwen-Image-Lightning",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Qwen-Image-Lightning © lightx2v — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `SG161222/RealVisXL_V5.0` — the alternate SDXL base, and the base the InstantID overlay names.
/// The repository ships **no** licence text of its own; the family's canonical text remains SDXL's
/// `LICENSE.md`.
pub const REALVISXL_V5_0: ComponentLicense = ComponentLicense {
    component: "realvisxl_v5_0",
    source_url: "https://huggingface.co/SG161222/RealVisXL_V5.0",
    gated: false,
    declared: "openrail++",
    family: "creativeml-openrail-pp-m",
    attribution: Some(
        "RealVisXL V5.0 © SG161222 — licensed under the CreativeML Open RAIL++-M License",
    ),
    retrieved: "2026-08-02",
};

/// `facebook/sam2.1-hiera-base-plus` — the SAM 2.1 speed variant. Apache-2.0, **not** the SAM 3
/// licence.
pub const SAM2_1_HIERA_BASE_PLUS: ComponentLicense = ComponentLicense {
    component: "sam2_1_hiera_base_plus",
    source_url: "https://huggingface.co/facebook/sam2.1-hiera-base-plus",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("SAM 2.1 Hiera Base+ © Meta Platforms — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `facebook/sam2.1-hiera-large` — the SAM 2.1 default. Meta licenses SAM 2.1 and SAM 3
/// **differently**: this is plain Apache-2.0 and ungated, while [`SAM3`] is a bespoke Meta text
/// behind a manual gate.
pub const SAM2_1_HIERA_LARGE: ComponentLicense = ComponentLicense {
    component: "sam2_1_hiera_large",
    source_url: "https://huggingface.co/facebook/sam2.1-hiera-large",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("SAM 2.1 Hiera Large © Meta Platforms — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `facebook/sam3` — one checkpoint carrying the PE ViT backbone, a CLIP-H text tower and
/// projection, the CLIP BPE tokenizer, the DETR detector, the mask head, the geometry/exemplar
/// encoder and a SAM2.1-style tracker.
///
/// The card sets `license: other` with **no** `license_name`, so `declared` is the licence
/// document's own title and `source_url` is that document. Gated for **manual** approval, which
/// [`ComponentLicense::gated`] records as a plain `true` — whether a gate is click-through or
/// human-reviewed is not a distinction this field carries.
///
/// `attribution` is `None`: the SAM License's only acknowledgement duty (§1(b)(ii)) binds on
/// submitting research for publication, not on rendering, so
/// [`META_SAM_LICENSE`](super::families::META_SAM_LICENSE) carries it as a quoted
/// [`DeployerObligation`](super::LicenseTerm::DeployerObligation) rather than
/// [`AttributionRequired`](super::LicenseTerm::AttributionRequired). A string here would name an
/// obligation the text does not impose on a render.
pub const SAM3: ComponentLicense = ComponentLicense {
    component: "sam3",
    source_url: "https://huggingface.co/facebook/sam3/blob/main/LICENSE",
    gated: true,
    declared: "SAM License",
    family: "meta-sam-license",
    attribution: None,
    retrieved: "2026-08-02",
};

/// `Efficient-Large-Model/Sana_1600M_1024px_diffusers`. Its Gemma-2-2B-IT text encoder is
/// [`GEMMA_2_2B_IT`]; `candle-gen-sana/NOTICE` was corrected on `main` by `eef5166a` to say so.
pub const SANA_1600M_1024PX_DIFFUSERS: ComponentLicense = ComponentLicense {
    component: "sana_1600m_1024px_diffusers",
    source_url: "https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("SANA 1600M 1024px © Efficient-Large-Model — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`.
pub const SANA_SPRINT_1_6B_1024PX_DIFFUSERS: ComponentLicense = ComponentLicense {
    component: "sana_sprint_1_6b_1024px_diffusers",
    source_url: "https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some(
        "SANA-Sprint 1.6B 1024px © Efficient-Large-Model — licensed under Apache-2.0",
    ),
    retrieved: "2026-08-02",
};

/// `zai-org/SCAIL-2`, loaded through the converted `SceneWorks/scail2-mlx` re-host, which declares
/// the same identifier. The bundled UMT5, open-CLIP ViT-H and Wan2.1 VAE are separate components
/// whose upstreams could not be established; they are omitted, and the repository's own MIT
/// declaration covers what it distributes.
pub const SCAIL_2: ComponentLicense = ComponentLicense {
    component: "scail_2",
    source_url: "https://huggingface.co/zai-org/SCAIL-2",
    gated: false,
    declared: "mit",
    family: "mit",
    attribution: Some("SCAIL-2 © zai-org — licensed under MIT"),
    retrieved: "2026-08-02",
};

/// `madebyollin/sdxl-vae-fp16-fix` — a **required** component of the Candle SDXL path, and one of
/// the few genuine per-backend component differences (MLX reads the VAE inside the SDXL snapshot).
pub const SDXL_VAE_FP16_FIX: ComponentLicense = ComponentLicense {
    component: "sdxl_vae_fp16_fix",
    source_url: "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix",
    gated: false,
    declared: "mit",
    family: "mit",
    attribution: Some("sdxl-vae-fp16-fix © madebyollin — licensed under MIT"),
    retrieved: "2026-08-02",
};

/// `numz/SeedVR2_comfyUI` — a community re-host supplying two DiT files and one shared VAE. Its card
/// lists `ByteDance-Seed/SeedVR2-7B` and `-3B` as base models, both Apache-2.0, which settles the
/// census's finding that no ByteDance repository id appears anywhere in either crate.
pub const SEEDVR2_COMFYUI: ComponentLicense = ComponentLicense {
    component: "seedvr2_comfyui",
    source_url: "https://huggingface.co/numz/SeedVR2_comfyUI",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("SeedVR2 for ComfyUI © numz — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `sensenova/SenseNova-U1-8B-MoT` — a unified model with no VAE and no separate text encoder. The
/// repository ships a `LICENSE` whose first lines are the Apache-2.0 header, verified rather than
/// assumed.
pub const SENSENOVA_U1_8B_MOT: ComponentLicense = ComponentLicense {
    component: "sensenova_u1_8b_mot",
    source_url: "https://huggingface.co/sensenova/SenseNova-U1-8B-MoT",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("SenseNova-U1-8B-MoT © SenseNova — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `sensenova/SenseNova-U1-8B-MoT-LoRAs` — the 8-step distill LoRA.
pub const SENSENOVA_U1_8B_MOT_LORAS: ComponentLicense = ComponentLicense {
    component: "sensenova_u1_8b_mot_loras",
    source_url: "https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-LoRAs",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("SenseNova-U1-8B-MoT LoRAs © SenseNova — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `stabilityai/stable-diffusion-3.5-large`. The declared string is transcribed as published —
/// `stabilityai-ai-community`, which is Stability's own tag for the text this table calls
/// `stability-ai-community`; the doubled "ai" is the upstream's, not a typo here.
///
/// The SD3.5 `-large-turbo` and `-medium` siblings have **no row**: both are governed by the same
/// text and neither's declared string was recorded by a primary-source read, so writing one would be
/// an assumption. See the module's known-holes note.
pub const STABLE_DIFFUSION_3_5_LARGE: ComponentLicense = ComponentLicense {
    component: "stable_diffusion_3_5_large",
    source_url: "https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
    gated: true,
    declared: "stabilityai-ai-community",
    family: "stability-ai-community",
    attribution: Some("Stable Diffusion 3.5 Large © Stability AI — Powered by Stability AI"),
    retrieved: "2026-08-02",
};

/// `stabilityai/stable-diffusion-xl-base-1.0` — also the repository whose `LICENSE.md` is the
/// canonical text for the whole `creativeml-openrail-pp-m` family.
pub const STABLE_DIFFUSION_XL_BASE_1_0: ComponentLicense = ComponentLicense {
    component: "stable_diffusion_xl_base_1_0",
    source_url: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
    gated: false,
    declared: "openrail++",
    family: "creativeml-openrail-pp-m",
    attribution: Some(
        "Stable Diffusion XL Base 1.0 © Stability AI — licensed under the CreativeML Open \
         RAIL++-M License",
    ),
    retrieved: "2026-08-02",
};

/// `stabilityai/stable-video-diffusion-img2vid-xt` — the UNet, the temporal-decoder VAE and the
/// ViT-H image encoder.
///
/// This row and [`STABLE_DIFFUSION_3_5_LARGE`] are the pair that forced `gated` off the family and
/// onto the component (sc-16898): **one licence text, two distribution settings**. SVD-XT is
/// ungated; SD3.5 Large is not.
pub const STABLE_VIDEO_DIFFUSION_IMG2VID_XT: ComponentLicense = ComponentLicense {
    component: "stable_video_diffusion_img2vid_xt",
    source_url: "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt",
    gated: false,
    declared: "stable-video-diffusion-community",
    family: "stability-ai-community",
    attribution: Some("Stable Video Diffusion img2vid-XT © Stability AI — Powered by Stability AI"),
    retrieved: "2026-08-02",
};

/// `google/t5-v1_1-xxl` — the T5-XXL text encoder redistributed inside the FLUX.1, Chroma and SD3.5
/// snapshots. One upstream, one row, three consumers; that is exactly the sharing the
/// component-keyed design exists to express.
///
/// The architecture repository is what the code names in each case
/// (`candle-gen-flux/src/packed_te.rs`, `candle-gen-chroma/src/loader.rs`); byte-provenance of any
/// redistributed copy against this repository was not verified, and is not something this table can
/// assert.
pub const T5_V1_1_XXL: ComponentLicense = ComponentLicense {
    component: "t5_v1_1_xxl",
    source_url: "https://huggingface.co/google/t5-v1_1-xxl",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("T5 v1.1 XXL © Google — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `google/umt5-xxl` and its tokenizer — the text encoder five Wan routes, Bernini and
/// krea-realtime share.
///
/// **Not SCAIL-2**, though an earlier revision of this line listed it. Every route above reads a
/// *standalone stock-Wan* `text_encoder/`: Bernini's converter copies it "verbatim from a base
/// Wan2.2-T2V-A14B diffusers snapshot" (`candle-gen-bernini/src/convert.rs`), and
/// `Wan-AI/Wan2.1-VACE-14B-diffusers` ships the transformer alone, so VACE must be given one.
/// SCAIL-2's converted snapshot ships its **own bundled** copy whose upstream no source names, which
/// is why [`SCAIL_2`] omits it and `scail2_umt5` is a pinned hole. Sharing `candle-gen-wan`'s
/// `Umt5Encoder` module is an architecture fact, not a provenance one.
pub const UMT5_XXL: ComponentLicense = ComponentLicense {
    component: "umt5_xxl",
    source_url: "https://huggingface.co/google/umt5-xxl",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("UMT5-XXL © Google — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Wan-AI/Wan2.1-T2V-14B-Diffusers` — the "stock Wan" trio krea-realtime reads.
///
/// **This corrects the census**, which grouped krea-realtime's trio under "stock Wan2.2":
/// `release/real-weight-models.toml` states the re-host is "Wan-2.1-T2V-14B weight-for-weight" and
/// that each tier ships the stock Wan `t5_encoder`/`vae`/`tokenizer`.
pub const WAN2_1_T2V_14B_DIFFUSERS: ComponentLicense = ComponentLicense {
    component: "wan2_1_t2v_14b_diffusers",
    source_url: "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Wan2.1-T2V-14B © Wan-AI — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Wan-AI/Wan2.1-VACE-14B-diffusers` — transformer only; the base trio is shared.
///
/// Read on its own, not inferred from its 1.3B sibling: card `sha`
/// `db79b90c60bbb45ceec9e41b9d5a4df934538ac4`, front matter `license: apache-2.0`. The evidence
/// note's row 49 spells both repository ids in full for the same reason — a `retrieved` date has to
/// stand for a read that actually happened against the id in `source_url`.
pub const WAN2_1_VACE_14B_DIFFUSERS: ComponentLicense = ComponentLicense {
    component: "wan2_1_vace_14b_diffusers",
    source_url: "https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Wan2.1-VACE-14B © Wan-AI — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Wan-AI/Wan2.1-VACE-1.3B-diffusers`. Two rows rather than one for the census's single VACE line:
/// 1.3B and 14B are different models in different repositories, which is the granularity the licence
/// attaches at — and each was read at its own repository, card `sha`
/// `ec4d2cb062b548996b179d493fdd05340de702a1` here.
pub const WAN2_1_VACE_1_3B_DIFFUSERS: ComponentLicense = ComponentLicense {
    component: "wan2_1_vace_1_3b_diffusers",
    source_url: "https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Wan2.1-VACE-1.3B © Wan-AI — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Wan-AI/Wan2.2-I2V-A14B` (and its `-Diffusers` sibling) — two expert checkpoints, `in_dim` 36.
pub const WAN2_2_I2V_A14B: ComponentLicense = ComponentLicense {
    component: "wan2_2_i2v_a14b",
    source_url: "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Wan2.2-I2V-A14B © Wan-AI — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Wan-AI/Wan2.2-T2V-A14B` (and its `-Diffusers` sibling) — two expert checkpoints, and the
/// standalone stock-Wan2.2 directory the Bernini converter deliberately reads in preference to
/// Bernini's own bundled UMT5 copy.
///
/// The repository states its own position on outputs: "The models in this repository are licensed
/// under the Apache 2.0 License. We claim no rights over the your generated contents" (typo the
/// source's). Recorded because it is the mirror of the FLUX and Ideogram outputs clauses and
/// confirms that no `NonCommercialOutputs` belongs anywhere near these rows.
pub const WAN2_2_T2V_A14B: ComponentLicense = ComponentLicense {
    component: "wan2_2_t2v_a14b",
    source_url: "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Wan2.2-T2V-A14B © Wan-AI — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Wan-AI/Wan2.2-TI2V-5B` (and its `-Diffusers` sibling) — includes the z48 VAE.
pub const WAN2_2_TI2V_5B: ComponentLicense = ComponentLicense {
    component: "wan2_2_ti2v_5b",
    source_url: "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Wan2.2-TI2V-5B © Wan-AI — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `QuantStack/Wan2.2-TI2V-5B-GGUF` — the Candle GGUF path's re-host. Its own card is the
/// declaration: "direct conversion of \[Wan-AI/Wan2.2-TI2V-5B\] … all original licensing terms and
/// usage restrictions remain in effect". A separate row from [`WAN2_2_TI2V_5B`] because it is a
/// separate repository a consumer provisions and a drift job re-reads.
pub const WAN2_2_TI2V_5B_GGUF: ComponentLicense = ComponentLicense {
    component: "wan2_2_ti2v_5b_gguf",
    source_url: "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Wan2.2-TI2V-5B-GGUF © QuantStack — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `alibaba-pai/Wan2.2-VACE-Fun-A14B` — two VACE expert checkpoints, MLX only today.
pub const WAN2_2_VACE_FUN_A14B: ComponentLicense = ComponentLicense {
    component: "wan2_2_vace_fun_a14b",
    source_url: "https://huggingface.co/alibaba-pai/Wan2.2-VACE-Fun-A14B",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Wan2.2-VACE-Fun-A14B © alibaba-pai — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `XLabs-AI/flux-ip-adapter` — the second of the three adapters that declare BFL's non-commercial
/// licence where a reader would expect a permissive one. Its card's copy of the `license_link` is
/// truncated; the `license_name` is not, and that is what `declared` transcribes.
pub const XLABS_FLUX_IP_ADAPTER: ComponentLicense = ComponentLicense {
    component: "xlabs_flux_ip_adapter",
    source_url: "https://huggingface.co/XLabs-AI/flux-ip-adapter",
    gated: false,
    declared: "flux-1-dev-non-commercial-license",
    family: "flux-1-dev-non-commercial",
    attribution: None,
    retrieved: "2026-08-02",
};

/// `Tongyi-MAI/Z-Image`. Its Qwen3-VL tower is a separate component whose upstream could not be
/// named; it is omitted.
pub const Z_IMAGE: ComponentLicense = ComponentLicense {
    component: "z_image",
    source_url: "https://huggingface.co/Tongyi-MAI/Z-Image",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Z-Image © Tongyi-MAI — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1`.
pub const Z_IMAGE_FUN_CONTROLNET_UNION_2_1: ComponentLicense = ComponentLicense {
    component: "z_image_fun_controlnet_union_2_1",
    source_url: "https://huggingface.co/alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Z-Image-Fun-Controlnet-Union-2.1 © alibaba-pai — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `Tongyi-MAI/Z-Image-Turbo`.
pub const Z_IMAGE_TURBO: ComponentLicense = ComponentLicense {
    component: "z_image_turbo",
    source_url: "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some("Z-Image-Turbo © Tongyi-MAI — licensed under Apache-2.0"),
    retrieved: "2026-08-02",
};

/// `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`.
pub const Z_IMAGE_TURBO_FUN_CONTROLNET_UNION_2_1: ComponentLicense = ComponentLicense {
    component: "z_image_turbo_fun_controlnet_union_2_1",
    source_url: "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
    gated: false,
    declared: "apache-2.0",
    family: "apache-2-0",
    attribution: Some(
        "Z-Image-Turbo-Fun-Controlnet-Union-2.1 © alibaba-pai — licensed under Apache-2.0",
    ),
    retrieved: "2026-08-02",
};

/// Every media checkpoint whose licence has been read, ordered by
/// [`ComponentLicense::component`].
///
/// **Shared by both media catalogs**, because a licence is a property of the checkpoint and the two
/// engines load the same checkpoints. The per-backend part of the surface is the
/// provider→component mapping, which lives in each catalog (sc-16666, sc-16667).
///
/// Membership means "a primary source for this checkpoint was read on the date its row records". It
/// does **not** mean every checkpoint the media lane loads is here: the census counted 90 distinct
/// upstream artifacts and the evidence pass could not settle all of them, so this table is
/// deliberately incomplete and the gaps are enumerated in the module's known-holes note rather than
/// filled with plausible values.
pub const MEDIA_COMPONENT_LICENSES: &[ComponentLicense] = &[
    ANIMA,
    ANIMA_OFFICIAL_LORAS,
    BERNINI_DIFFUSERS,
    BOOGU_IMAGE_0_1_BASE,
    BOOGU_IMAGE_0_1_EDIT,
    BOOGU_IMAGE_0_1_TURBO,
    CHATGLM3_6B,
    CHROMA1_BASE,
    CHROMA1_FLASH,
    CHROMA1_HD,
    CLIP_VIT_BIGG_14_LAION2B,
    CONTROLNET_OPENPOSE_SDXL_1_0,
    EVA_CLIP,
    FACEXLIB_PARSING_BISENET,
    FLUX1_DEV,
    FLUX1_DEV_CONTROLNET_UNION_PRO_2_0,
    FLUX1_SCHNELL,
    FLUX2_DEV,
    FLUX2_KLEIN_9B,
    FLUX2_KLEIN_9B_KV,
    GEMMA_2_2B_IT,
    GEMMA_3_12B_IT,
    GPT_OSS_20B,
    HYPER_SD_FLUX1_DEV_8STEPS_LORA,
    IDEOGRAM_4_FP8,
    IDEOGRAM_4_TURBOTIME_LORA,
    INSTANTID,
    IP_ADAPTER,
    KOLORS_IP_ADAPTER_PLUS,
    KREA_2_RAW,
    KREA_2_TURBO,
    KREA_REALTIME_VIDEO,
    LTX_2_3,
    MOCHI_1_PREVIEW,
    NVIDIA_PID_STUDENTS,
    PULID,
    QWEN_IMAGE,
    QWEN_IMAGE_2512,
    QWEN_IMAGE_2512_FUN_CONTROLNET_UNION,
    QWEN_IMAGE_EDIT,
    QWEN_IMAGE_EDIT_2511_LIGHTNING,
    QWEN_IMAGE_LIGHTNING,
    REALVISXL_V5_0,
    SAM2_1_HIERA_BASE_PLUS,
    SAM2_1_HIERA_LARGE,
    SAM3,
    SANA_1600M_1024PX_DIFFUSERS,
    SANA_SPRINT_1_6B_1024PX_DIFFUSERS,
    SCAIL_2,
    SDXL_VAE_FP16_FIX,
    SEEDVR2_COMFYUI,
    SENSENOVA_U1_8B_MOT,
    SENSENOVA_U1_8B_MOT_LORAS,
    STABLE_DIFFUSION_3_5_LARGE,
    STABLE_DIFFUSION_XL_BASE_1_0,
    STABLE_VIDEO_DIFFUSION_IMG2VID_XT,
    T5_V1_1_XXL,
    UMT5_XXL,
    WAN2_1_T2V_14B_DIFFUSERS,
    WAN2_1_VACE_14B_DIFFUSERS,
    WAN2_1_VACE_1_3B_DIFFUSERS,
    WAN2_2_I2V_A14B,
    WAN2_2_T2V_A14B,
    WAN2_2_TI2V_5B,
    WAN2_2_TI2V_5B_GGUF,
    WAN2_2_VACE_FUN_A14B,
    XLABS_FLUX_IP_ADAPTER,
    Z_IMAGE,
    Z_IMAGE_FUN_CONTROLNET_UNION_2_1,
    Z_IMAGE_TURBO,
    Z_IMAGE_TURBO_FUN_CONTROLNET_UNION_2_1,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::license::families::LICENSE_FAMILIES;
    use crate::license::{
        license_table_conformance_errors, resolve_component, resolve_family, LicenseTerm,
    };

    /// Every const is reachable through the public slice and through the one sanctioned lookup path.
    /// A row declared but left out of [`MEDIA_COMPONENT_LICENSES`] would be invisible to every
    /// consumer while still looking landed in source.
    #[test]
    fn every_row_is_in_the_slice_and_resolves_by_key() {
        assert_eq!(MEDIA_COMPONENT_LICENSES.len(), 71);
        for row in MEDIA_COMPONENT_LICENSES {
            assert_eq!(
                resolve_component(MEDIA_COMPONENT_LICENSES, row.component),
                Some(row),
                "{:?} must resolve to itself",
                row.component
            );
        }
        // The slice is sorted by key, so a reader can find a row and a future diff stays local.
        let mut keys: Vec<&str> = MEDIA_COMPONENT_LICENSES
            .iter()
            .map(|row| row.component)
            .collect();
        let declared_order = keys.clone();
        keys.sort_unstable();
        assert_eq!(
            keys, declared_order,
            "MEDIA_COMPONENT_LICENSES must be sorted by component key"
        );
    }

    /// Every row resolves the family it names. A component pointing at a family that does not exist
    /// contributes **nothing** to a derived term union, so it would under-report silently rather
    /// than fail — which is why this is asserted directly as well as through the table gate.
    #[test]
    fn every_row_resolves_its_family() {
        for row in MEDIA_COMPONENT_LICENSES {
            assert!(
                resolve_family(LICENSE_FAMILIES, row.family).is_some(),
                "component {:?} names unknown family {:?}",
                row.component,
                row.family
            );
        }
    }

    /// The ship-gate over families plus components. Provider rows are per-backend and belong to the
    /// two media catalogs, so the table is checked with that section empty — which is exactly how a
    /// catalog calls it while its mapping is being written.
    #[test]
    fn media_component_table_is_conformant() {
        assert_eq!(
            license_table_conformance_errors(LICENSE_FAMILIES, MEDIA_COMPONENT_LICENSES, &[]),
            Vec::<String>::new()
        );
    }

    /// `attribution` is `Some` exactly when the resolved family imposes
    /// [`LicenseTerm::AttributionRequired`].
    ///
    /// The conformance gate enforces one direction (a requiring family implies a non-blank string).
    /// This enforces the other, so a helpful-looking attribution never appears on a row whose text
    /// does not ask for one and imply an obligation the licence never stated. Five families in this
    /// table require none: `flux-1-dev-non-commercial`, `gemma-terms`, `meta-sam-license` (whose
    /// acknowledgement duty is conditional on publishing research and is carried as a quoted
    /// [`LicenseTerm::DeployerObligation`] instead), and — reachable only through rows this table
    /// does not yet carry — `insightface-research-only` and `apple-mlr`.
    #[test]
    fn attribution_is_present_exactly_where_the_family_requires_it() {
        let mut without: Vec<&str> = Vec::new();
        for row in MEDIA_COMPONENT_LICENSES {
            let family = resolve_family(LICENSE_FAMILIES, row.family).expect("family resolves");
            if family.requires_attribution() {
                assert!(
                    row.attribution.is_some(),
                    "component {:?} resolves to {:?}, which requires attribution",
                    row.component,
                    family.id
                );
            } else {
                assert!(
                    row.attribution.is_none(),
                    "component {:?} records an attribution its family {:?} does not require; an \
                     unrequired attribution reads as an obligation the text never stated",
                    row.component,
                    family.id
                );
                without.push(row.component);
            }
        }
        assert_eq!(
            without,
            vec![
                "flux1_dev",
                "flux1_dev_controlnet_union_pro_2_0",
                "gemma_2_2b_it",
                "gemma_3_12b_it",
                "hyper_sd_flux1_dev_8steps_lora",
                "sam3",
                "xlabs_flux_ip_adapter",
            ]
        );
    }

    /// `gated` is a per-checkpoint distribution fact, and exactly eleven rows carry it.
    ///
    /// Pinned as a set rather than a count because the value is measured, not derived: an upstream
    /// that opens or closes a gate must move this list, and a row that flips without an
    /// accompanying `retrieved` bump is a transcription error rather than an upstream change.
    #[test]
    fn gated_rows_are_the_measured_set() {
        let gated: Vec<&str> = MEDIA_COMPONENT_LICENSES
            .iter()
            .filter(|row| row.gated)
            .map(|row| row.component)
            .collect();
        assert_eq!(
            gated,
            vec![
                "flux1_dev",
                "flux2_dev",
                "flux2_klein_9b",
                "flux2_klein_9b_kv",
                "gemma_2_2b_it",
                "gemma_3_12b_it",
                "ideogram_4_fp8",
                "krea_2_raw",
                "krea_2_turbo",
                "sam3",
                "stable_diffusion_3_5_large",
            ]
        );
    }

    /// One licence text, two distribution settings — the case that forced
    /// [`LicenseTerm::GatedAccess`] off the family and onto the component (sc-16898). Both rows
    /// resolve to `stability-ai-community` and they disagree about `gated`, which a family-level
    /// flag could not have expressed without splitting one reviewed text in two.
    #[test]
    fn one_family_covers_a_gated_and_an_ungated_media_checkpoint() {
        let svd = resolve_component(
            MEDIA_COMPONENT_LICENSES,
            "stable_video_diffusion_img2vid_xt",
        )
        .expect("SVD-XT row");
        let sd35 = resolve_component(MEDIA_COMPONENT_LICENSES, "stable_diffusion_3_5_large")
            .expect("SD3.5 Large row");
        assert_eq!(svd.family, sd35.family);
        assert!(!svd.gated);
        assert!(sd35.gated);
        // And they declare different strings for that one text, which is why `declared` is
        // transcribed rather than normalized.
        assert_ne!(svd.declared, sd35.declared);
    }

    /// The five checkpoints where a repository-level row is genuinely wrong, and the shape each one
    /// landed as. A later edit that quietly re-collapses any of them to its repository fails here.
    #[test]
    fn the_five_granularity_exceptions_produce_the_rows_the_evidence_requires() {
        // 1. SceneWorks/Lens declares `mit` while its own README assigns three licences across
        //    three directories. The text encoder and the VAE are keyed at their own upstreams; the
        //    transformer has no row (its MIT is second-hand from the deleted `microsoft/Lens`).
        let lens_te = resolve_component(MEDIA_COMPONENT_LICENSES, "gpt_oss_20b").expect("Lens TE");
        assert_eq!(
            lens_te.source_url,
            "https://huggingface.co/openai/gpt-oss-20b"
        );
        assert_eq!(lens_te.family, "apache-2-0");
        let lens_vae = resolve_component(MEDIA_COMPONENT_LICENSES, "flux2_dev").expect("Lens VAE");
        assert_eq!(lens_vae.family, "flux-non-commercial-v2-1");
        assert!(resolve_component(MEDIA_COMPONENT_LICENSES, "sceneworks_lens").is_none());
        assert!(resolve_component(MEDIA_COMPONENT_LICENSES, "lens_transformer").is_none());

        // 2. nvidia/PiD is an NSCLv1 repository redistributing five third parties' VAEs. The row is
        //    scoped to NVIDIA's own student decoders and names the README, not the card, because the
        //    card carries no `license` field at all.
        let pid =
            resolve_component(MEDIA_COMPONENT_LICENSES, "nvidia_pid_students").expect("PiD row");
        assert_eq!(pid.family, "nvidia-nsclv1");
        assert_eq!(
            pid.source_url,
            "https://huggingface.co/nvidia/PiD/raw/main/README.md"
        );
        assert!(
            resolve_component(MEDIA_COMPONENT_LICENSES, "nvidia_pid").is_none(),
            "an unscoped PiD row would sweep five third parties' encoders under NVIDIA's licence"
        );

        // 3. Boogu declares apache-2.0 repo-wide over a "FLUX.1 VAE" it never identifies. The three
        //    DiT rows land; the VAE does not.
        for key in [
            "boogu_image_0_1_base",
            "boogu_image_0_1_turbo",
            "boogu_image_0_1_edit",
        ] {
            let row = resolve_component(MEDIA_COMPONENT_LICENSES, key).expect("Boogu row");
            assert_eq!(row.family, "apache-2-0");
        }
        assert!(resolve_component(MEDIA_COMPONENT_LICENSES, "boogu_flux1_vae").is_none());

        // 4. Kolors: the ChatGLM3 text-encoder split is required regardless of how U6 resolves, so
        //    it lands while the repository-level row does not.
        let chatglm = resolve_component(MEDIA_COMPONENT_LICENSES, "chatglm3_6b").expect("ChatGLM3");
        assert_eq!(chatglm.family, "chatglm3-model-license");
        assert!(resolve_component(MEDIA_COMPONENT_LICENSES, "kolors_diffusers").is_none());

        // 5. ByteDance/Hyper-SD scopes different clauses of one LICENSE.md to different weight
        //    families. Only the FLUX-scoped artifact is loaded here, so only it is keyed.
        let hyper = resolve_component(MEDIA_COMPONENT_LICENSES, "hyper_sd_flux1_dev_8steps_lora")
            .expect("Hyper-FLUX LoRA");
        assert_eq!(hyper.family, "flux-1-dev-non-commercial");
        assert_eq!(
            hyper.source_url,
            "https://huggingface.co/ByteDance/Hyper-SD/raw/main/LICENSE.md"
        );
        assert!(resolve_component(MEDIA_COMPONENT_LICENSES, "hyper_sd").is_none());

        // The evidence explicitly records one NON-exception: ideogram-4-fp8 bundles the ostris
        // TurboTime LoRA, and both declare the SAME identifier, so no per-file split is required —
        // only `source_url` differs.
        let base = resolve_component(MEDIA_COMPONENT_LICENSES, "ideogram_4_fp8").expect("Ideogram");
        let lora = resolve_component(MEDIA_COMPONENT_LICENSES, "ideogram_4_turbotime_lora")
            .expect("TurboTime LoRA");
        assert_eq!(base.declared, lora.declared);
        assert_eq!(base.family, lora.family);
        assert_ne!(base.source_url, lora.source_url);
    }

    /// The tripwire for "no row without evidence".
    ///
    /// Every key below is a component the media lane demonstrably loads and for which the evidence
    /// pass could **not** produce a licence: the upstream could not be named (UNDETERMINED), the
    /// upstream declares nothing re-readable (NOT FOUND), two primary sources disagree (AMBIGUOUS),
    /// there is no string to transcribe as `declared` (NO DECLARED STRING — either prose with no
    /// identifier behind it, or none recorded by a read), or the only declaration is this
    /// repository's own record (SECOND-HAND ONLY). Each is enumerated in
    /// `docs/licensing/sc-16665-checkpoint-licence-evidence.md`.
    ///
    /// Omitting them is the designed outcome, not an oversight: sc-16669's ship gate reports a
    /// missing row and fails CI, and per the epic's product constraint missing licence data never
    /// withholds a provider — so a hole is loud, while a fabricated row is silent and wrong. This
    /// test exists so that adding any of these later is a **deliberate diff**: the author has to
    /// delete the key here, which is the moment to check that a primary source now exists.
    #[test]
    fn known_holes_stay_absent() {
        const KNOWN_HOLES: &[(&str, &str)] = &[
            // --- UNDETERMINED: the upstream repository could not be named -------------------
            ("flux2_klein_qwen3_text_encoder", "UNDETERMINED"),
            ("flux2_dev_mistral3_tower", "UNDETERMINED"),
            ("flux2_dev_pixtral_vision_tower", "UNDETERMINED"),
            ("flux2_dev_multimodal_projector", "UNDETERMINED"),
            ("anima_qwen3_0_6b_text_encoder", "UNDETERMINED"),
            ("anima_qwen_image_vae", "UNDETERMINED"),
            ("boogu_qwen3_vl_8b", "UNDETERMINED"),
            ("krea_qwen3_vl_4b", "UNDETERMINED"),
            ("mage_qwen3_vl_4b", "UNDETERMINED"),
            ("ideogram_qwen3_vl_8b", "UNDETERMINED"),
            ("z_image_qwen3_text_encoder", "UNDETERMINED"),
            ("qwen_image_qwen2_5_vl", "UNDETERMINED"),
            // These three are SCAIL-2's own **bundled** auxiliaries, and they stay holes. Reusing
            // `candle-gen-wan`'s `Umt5Encoder` / `WanVae16` modules is not evidence of provenance:
            // Bernini's converter states it copies the stock Wan2.2 `text_encoder/` and `vae/`
            // "verbatim from a base Wan2.2-T2V-A14B diffusers snapshot"
            // (`candle-gen-bernini/src/convert.rs`) and `Wan-AI/Wan2.1-VACE-14B-diffusers` ships the
            // transformer alone, so both of those are established reads of a named upstream. The
            // converted `SceneWorks/scail2-mlx` snapshot ships its own copies and no source names
            // where they came from — see [`SCAIL_2`]. sc-16667 re-checked this and left the keys.
            ("scail2_umt5", "UNDETERMINED"),
            ("scail2_open_clip_vit_h", "UNDETERMINED"),
            ("scail2_wan2_1_vae", "UNDETERMINED"),
            ("lightx2v_wan_step_distill_diff_patch", "UNDETERMINED"),
            // Recorded by the evidence pass as a hypothesis, not a finding: no Hugging Face
            // repository named `sat-scail2` exists under any search.
            ("sat_scail2_dpo_lora", "UNDETERMINED / NOT FOUND"),
            // PiD's five redistributed third-party encoders — see NVIDIA_PID_STUDENTS.
            ("pid_flux1_vae", "UNDETERMINED"),
            ("pid_flux2_vae", "UNDETERMINED"),
            ("pid_sdxl_vae", "UNDETERMINED"),
            ("pid_sd3_vae", "UNDETERMINED"),
            ("pid_qwen_image_vae", "UNDETERMINED"),
            // --- NOT FOUND: the upstream is known and declares nothing re-readable ----------
            ("kolors_controlnet_pose", "NOT FOUND — no licence, no card"),
            (
                "clip_vit_large_patch14",
                "NOT FOUND — card declares nothing",
            ),
            (
                "clip_vit_large_patch14_336",
                "NOT FOUND — card declares nothing",
            ),
            ("mage_flow", "NOT FOUND — repository 404s"),
            ("mage_flow_base", "NOT FOUND — repository 404s"),
            ("mage_flow_turbo", "NOT FOUND — repository 404s"),
            ("mage_flow_edit", "NOT FOUND — repository 404s"),
            ("mage_flow_edit_base", "NOT FOUND — repository 404s"),
            ("mage_flow_edit_turbo", "NOT FOUND — repository 404s"),
            ("microsoft_lens", "NOT FOUND — repository 404s"),
            ("microsoft_lens_turbo", "NOT FOUND — repository 404s"),
            ("lens_transformer", "NOT FOUND — see microsoft_lens"),
            (
                "krea2_pose_controlnet_beta",
                "NOT FOUND — `experimental-research-only` with no text behind it",
            ),
            // --- AMBIGUOUS: two primary sources disagree, or the family is open -------------
            ("kolors_diffusers", "AMBIGUOUS — sc-16662 U6"),
            (
                "sceneworks_kolors_chatglm3_tokenizer",
                "AMBIGUOUS — declares `kolors`; family conditional on U6",
            ),
            (
                "flux2_dev_fun_controlnet_union",
                "AMBIGUOUS — ships BFL v2.0; one family or two is open",
            ),
            ("boogu_flux1_vae", "AMBIGUOUS — FLUX.1 dev or schnell"),
            (
                "amoral_gemma_3_12b_v2_mlx_4bit",
                "AMBIGUOUS — a Gemma-3 derivative declaring `apache-2.0`",
            ),
            // --- NO DECLARED STRING: nothing to transcribe as `declared` --------------------
            // Two shapes reach this class. The insightface rows have a restriction stated in
            // README prose with no identifier behind it; the two SD3.5 siblings simply had no
            // declared string recorded by a primary-source read. Neither is AMBIGUOUS: nothing
            // disagrees, there is just no string. The distinction matters to a reader triaging
            // these — an AMBIGUOUS hole needs a decision, one of these needs a read.
            (
                "stable_diffusion_3_5_large_turbo",
                "NO DECLARED STRING — none recorded by a primary-source read",
            ),
            (
                "stable_diffusion_3_5_medium",
                "NO DECLARED STRING — none recorded by a primary-source read",
            ),
            (
                "antelopev2_arcface_glintr100",
                "NO DECLARED STRING — insightface publishes README prose, not an identifier",
            ),
            (
                "antelopev2_scrfd_10g",
                "NO DECLARED STRING — insightface publishes README prose, not an identifier",
            ),
            // --- SECOND-HAND ONLY: the only declaration is this repository's own record -----
            (
                "llama_joycaption_beta_one_hf_llava",
                "SECOND-HAND ONLY — declared only in release/real-weight-models.toml",
            ),
            (
                "depth_anything_v2_small_hf",
                "SECOND-HAND ONLY — declared only in a provider crate's rustdoc",
            ),
        ];

        for (key, reason) in KNOWN_HOLES {
            assert!(
                resolve_component(MEDIA_COMPONENT_LICENSES, key).is_none(),
                "component {key:?} was added without removing it from the known-holes list \
                 ({reason}); if a primary source now exists, delete the entry here and record the \
                 quote in docs/licensing/sc-16665-checkpoint-licence-evidence.md",
            );
        }
        assert_eq!(KNOWN_HOLES.len(), 46);
    }

    /// No row asserts an outputs restriction, transitively.
    ///
    /// `families.rs` pins that no *family* carries [`LicenseTerm::NonCommercialOutputs`]; this pins
    /// that no *component* reaches one, so the guarantee holds at the layer a consumer actually
    /// joins over. Several rows here resolve to families that restrict non-commercial use of the
    /// weights and say nothing about outputs — inferring the stronger term from that silence is the
    /// reading sc-16662's U3 declined to make.
    #[test]
    fn no_component_reaches_an_outputs_restriction() {
        for row in MEDIA_COMPONENT_LICENSES {
            let family = resolve_family(LICENSE_FAMILIES, row.family).expect("family resolves");
            assert!(
                !family.imposes(LicenseTerm::NonCommercialOutputs),
                "component {:?} reaches non_commercial_outputs through {:?}",
                row.component,
                family.id
            );
        }
    }

    /// Every `retrieved` is the single date both evidence packs record, and every `source_url` is an
    /// absolute URL a drift job can fetch.
    ///
    /// The second half is the property that makes the field useful at all: the evidence pass found
    /// eight upstream repositories that now 404, and a row pointing at one of them would look like
    /// provenance while being unre-readable. Those repositories have no rows, which is why this
    /// assertion can be unconditional.
    #[test]
    fn provenance_fields_are_re_readable() {
        for row in MEDIA_COMPONENT_LICENSES {
            assert_eq!(
                row.retrieved, "2026-08-02",
                "component {:?} carries a retrieval date neither evidence pack records",
                row.component
            );
            assert!(
                row.source_url.starts_with("https://"),
                "component {:?} has a source_url a drift job cannot fetch: {:?}",
                row.component,
                row.source_url
            );
        }
    }
}
