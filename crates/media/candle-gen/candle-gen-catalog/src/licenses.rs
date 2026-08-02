//! Which shared checkpoint rows each **registered Candle** provider id loads (sc-16667).
//!
//! # Disclosure only
//!
//! Everything here exists so a consumer can **show** a user which upstream artifacts a render
//! touched and what their licence texts name. Nothing in this module blocks, gates, degrades or
//! withholds anything, and nothing added here ever should — a mapping describes, it does not
//! decide. The nine registered ids that reach no shared row at all are a list of **gaps in the
//! data**, not a suppression list. That list lives in `tests` as
//! `IDS_WITHOUT_A_RESOLVABLE_COMPONENT` and is deliberately `#[cfg(test)]` rather than a public
//! const, so no gate can quietly use it to suppress the ids it names. Every one of them is
//! registered by [`provider_registry`](super::provider_registry), ships, and renders exactly as it
//! did before this module existed.
//!
//! # What this module is, and is not
//!
//! The [`ComponentLicense`](candle_gen::gen_core::ComponentLicense) rows themselves are
//! **backend-independent** and live once, in
//! [`MEDIA_COMPONENT_LICENSES`](candle_gen::gen_core::MEDIA_COMPONENT_LICENSES) — a licence is a
//! property of the checkpoint, and the MLX and Candle engines load the same checkpoints. Only the
//! provider-to-component mapping legitimately differs per backend, because the two catalogs register
//! different id sets. This module is the Candle half; `mlx-gen-catalog` carries the MLX half
//! (sc-16666). **No component row is duplicated here**, and none may be: a row added on one side
//! only would let the two catalogs answer differently about one upstream artifact.
//!
//! # How each list was derived
//!
//! From `docs/licensing/sc-16665-media-checkpoint-census.md`, which enumerates from the code — file
//! and line — every component each registered provider loads, cross-checked against what the
//! **Candle** crates themselves name. Where the two backends diverge, the Candle reading wins here;
//! the divergences are pinned by `per_backend_divergences_are_explicit` below rather than papered
//! over, per `CONTRIBUTING.md`.
//!
//! # Reusing a module is not provenance
//!
//! Three ids — `bernini`, `wan_vace`, `scail2_14b` — drive the **same two** `candle-gen-wan`
//! modules (`text_encoder::Umt5Encoder` at `TextEncoderConfig::umt5_xxl()`, `vae16::WanVae16` at
//! `Vae16Config::wan21()`) over a `text_encoder/` and a `vae/` under their own snapshot root. Only
//! the first two list `umt5_xxl` and `wan2_2_t2v_a14b`. That looks arbitrary and is not, so the
//! reason is written here rather than left to be re-litigated:
//!
//! * `bernini` — `candle-gen-bernini/src/convert.rs` states the snapshot it builds copies the stock
//!   Wan2.2 `text_encoder/`, `vae/` and `tokenizer/` **verbatim from a base Wan2.2-T2V-A14B
//!   diffusers snapshot**. The upstream is named by the code that puts the files there.
//! * `wan_vace` — `Wan-AI/Wan2.1-VACE-14B-diffusers` ships the transformer alone, so the encoder and
//!   VAE a VACE run reads can only have been staged from a stock Wan snapshot.
//! * `scail2_14b` — the loaded artifact is the converted `SceneWorks/scail2-mlx` re-host, which
//!   ships **its own bundled** `text_encoder/`, `vae/` and `clip/`. No source states where those
//!   copies came from; `candle-gen-scail2` names the architectures ("UMT5", "z16 Wan VAE") and
//!   stops, and the sc-16665 evidence pass recorded `google/umt5-xxl` and Wan2.1 as *unestablished
//!   candidates*. `scail2_umt5` and `scail2_wan2_1_vae` are therefore pinned holes, and the shared
//!   [`SCAIL_2`](candle_gen::gen_core::license::components::SCAIL_2) row says so in as many words.
//!
//! The rule the three cases share: a component is keyed at an upstream when **something states that
//! upstream** — the crate, the converter, or the repository's contents. Sharing a Rust module is a
//! fact about architecture. Keying SCAIL-2's bundled copies at the Wan rows on that basis is exactly
//! the "plausible-looking substitute row" this table exists to refuse.
//!
//! # Scope: the components a run **always** loads
//!
//! [`ProviderComponents`] carries a flat list with no
//! optionality, so a list has to mean one thing. It means: the artifacts every run of that id
//! loads. Two classes are therefore deliberately absent, and each is absent for a reason a reader
//! can check:
//!
//! * **Caller-selected add-ons** — the SDXL and FLUX IP-adapters, the Qwen-Image and
//!   Qwen-Image-Edit Lightning LoRAs, the Hyper-FLUX 8-step LoRA, the Anima official LoRAs, the
//!   Kolors ControlNet and IP-Adapter. A run may load none of them, so listing them would report
//!   obligations a given render never touched.
//! * **The non-registry overlays** — the PiD latent decoder, the face stack, depth, InstantID,
//!   SAM 3 and the bespoke Candle PuLID path: all six of
//!   [`BESPOKE_UTILITY_CRATES`](super::BESPOKE_UTILITY_CRATES). They register **no provider id at
//!   all** on this backend, so a provider-keyed table cannot
//!   express them; sc-16668 owns that surface. This is the widest gap in the module and it is
//!   deliberate: `nvidia/PiD` is an optional decode seam on fourteen Candle crates and no provider
//!   id distinguishes a run that used it from one that did not.
//!
//! A checkpoint that is **merged before the run** is *not* in either class and **is** listed:
//! `sensenova_u1_8b_fast` merges its distill LoRA and `ideogram_4_turbo` its bundled TurboTime LoRA
//! on every run, so both rows appear.
//!
//! # Incomplete lists are the honest state
//!
//! 46 checkpoints the media lane demonstrably loads have **no row** in the shared table, because
//! the sc-16665 evidence pass could not name the upstream, found nothing re-readable, or found two
//! primary sources disagreeing; they are pinned by `known_holes_stay_absent` in `gen-core`'s
//! component module. A provider that loads one of them simply has a shorter list. That is a
//! CI-visible hole, reported by sc-16669's gate — never a reason to withhold an id, and never
//! something to paper over by pointing at a plausible-looking substitute row.
//!
//! Nine registered ids reach **zero** resolvable components and therefore have no row here at all.
//! They register, load and render unchanged; what is missing is our metadata, and it is pinned by
//! `every_registered_id_is_mapped_or_a_named_hole`:
//!
//! | id(s) | why |
//! | --- | --- |
//! | the six `mage_flow*` generators | every `microsoft/Mage-Flow*` repository 404s and the Qwen3-VL-4B tower bundled in each `text_encoder/` could not be named, so neither the DiT, the VAE nor the encoder has a row |
//! | `fancyfeast/llama-joycaption-beta-one-hf-llava` | one snapshot carries the whole captioner, and its only declaration is second-hand (`release/real-weight-models.toml`) |
//! | `clip_vit_l14`, `clip_vit_l14_text` | both read one `openai/clip-vit-large-patch14` snapshot, whose card declares nothing |

use candle_gen::gen_core::ProviderComponents;

/// Which shared component rows each registered Candle provider id loads, ordered by
/// [`ProviderComponents::provider_id`].
///
/// Sorted so a reader can find an id and a future diff stays local; the emitted manifest sorts
/// independently, so this ordering carries no meaning beyond readability.
///
/// One row per **distinct** id: five Candle trainer ids (`krea_2_raw`, `lens`, `sdxl`,
/// `wan2_2_t2v_14b`, `z_image_turbo`) are also generator ids and load the same checkpoints, so they
/// appear once — a duplicate `provider_id` is rejected by the conformance gate. `krea_2_control`
/// and `ltx_2_3` are trainer-**only** ids with no Candle generator, which is why they appear here
/// and in no generator list.
pub const PROVIDER_COMPONENTS: &[ProviderComponents] = &[
    // --- anima ------------------------------------------------------------------------------
    // Three separately-trained DiT files in one repository. The bundled Qwen3-0.6B text encoder
    // and Qwen-Image VAE are pinned holes, so all three lists are the DiT repository alone.
    ProviderComponents {
        provider_id: "anima_aesthetic",
        components: &["anima"],
    },
    ProviderComponents {
        provider_id: "anima_base",
        components: &["anima"],
    },
    ProviderComponents {
        provider_id: "anima_turbo",
        components: &["anima"],
    },
    // --- bernini ----------------------------------------------------------------------------
    // `bernini` adds the planner, connector, ViT-decoder head and mask tokens, but all six
    // prefixes come from the one `ByteDance/Bernini-Diffusers` index, so the two ids carry the
    // same list. The converter deliberately reads the standalone stock-Wan UMT5/VAE/tokenizer in
    // preference to Bernini's own bundled UMT5 copy (`candle-gen-bernini/src/convert.rs`).
    ProviderComponents {
        provider_id: "bernini",
        components: &["bernini_diffusers", "umt5_xxl", "wan2_2_t2v_a14b"],
    },
    ProviderComponents {
        provider_id: "bernini_renderer",
        components: &["bernini_diffusers", "umt5_xxl", "wan2_2_t2v_a14b"],
    },
    // --- boogu ------------------------------------------------------------------------------
    // Three snapshots, one architecture. The bundled Qwen3-VL-8B condition encoder and the
    // "FLUX.1 VAE" the README never resolves to dev or schnell are both pinned holes, so only the
    // DiT repositories are listed.
    ProviderComponents {
        provider_id: "boogu_image",
        components: &["boogu_image_0_1_base"],
    },
    ProviderComponents {
        provider_id: "boogu_image_edit",
        components: &["boogu_image_0_1_edit"],
    },
    ProviderComponents {
        provider_id: "boogu_image_turbo",
        components: &["boogu_image_0_1_turbo"],
    },
    // --- chroma -----------------------------------------------------------------------------
    // Each repository's own `vae/` rides its row; the T5-XXL inside its `text_encoder/` is keyed
    // at the upstream the crate names, per the redistributed-component rule.
    ProviderComponents {
        provider_id: "chroma1_base",
        components: &["chroma1_base", "t5_v1_1_xxl"],
    },
    ProviderComponents {
        provider_id: "chroma1_flash",
        components: &["chroma1_flash", "t5_v1_1_xxl"],
    },
    ProviderComponents {
        provider_id: "chroma1_hd",
        components: &["chroma1_hd", "t5_v1_1_xxl"],
    },
    // --- flux (FLUX.1) ----------------------------------------------------------------------
    // Candle registers two ids; ControlNet and IP-Adapter exist as bespoke, unregistered providers
    // (`control_provider.rs`, `ip_provider.rs`) and are therefore not on this surface. The
    // `ae.safetensors` VAE is BFL's own and rides the base row; the CLIP-L tower is a pinned hole.
    ProviderComponents {
        provider_id: "flux1_dev",
        components: &["flux1_dev", "t5_v1_1_xxl"],
    },
    ProviderComponents {
        provider_id: "flux1_schnell",
        components: &["flux1_schnell", "t5_v1_1_xxl"],
    },
    // --- flux2 (FLUX.2) ---------------------------------------------------------------------
    // Candle registers two of MLX's six; edit and control are bespoke and unregistered. The
    // 32-channel VAE ships inside each FLUX.2 snapshot and rides its row; the Qwen3 and Mistral3
    // towers are pinned holes.
    ProviderComponents {
        provider_id: "flux2_dev",
        components: &["flux2_dev"],
    },
    ProviderComponents {
        provider_id: "flux2_klein_9b",
        components: &["flux2_klein_9b"],
    },
    // --- ideogram ---------------------------------------------------------------------------
    // Both DiTs, the VAE and the text encoder ship in one gated snapshot. `ideogram_4_turbo`
    // merges the bundled TurboTime LoRA on **every** run, so that row is listed rather than
    // treated as an optional add-on. The Qwen3-VL-8B tower is a pinned hole.
    ProviderComponents {
        provider_id: "ideogram_4",
        components: &["ideogram_4_fp8"],
    },
    ProviderComponents {
        provider_id: "ideogram_4_turbo",
        components: &["ideogram_4_fp8", "ideogram_4_turbotime_lora"],
    },
    // --- kolors -----------------------------------------------------------------------------
    // The `Kwai-Kolors/Kolors-diffusers` repository row is an AMBIGUOUS hole (sc-16662 U6), so the
    // UNet, VAE and derived tokenizer contribute nothing. The ChatGLM3-6B text encoder is governed
    // by its own licence whichever way U6 lands, which is why it is the one component here.
    ProviderComponents {
        provider_id: "kolors",
        components: &["chatglm3_6b"],
    },
    // --- krea -------------------------------------------------------------------------------
    // The VAE is `Qwen/Qwen-Image` by the Krea `vae/config.json`'s own `_name_or_path`.
    // `krea_2_edit` runs the edit surface on the **Raw** checkpoint. `krea_2_control` is a
    // trainer-only id whose produced overlay applies at `krea_2_turbo` inference, so it loads the
    // Turbo snapshot. The Qwen3-VL-4B tower is a pinned hole.
    ProviderComponents {
        provider_id: "krea_2_control",
        components: &["krea_2_turbo", "qwen_image"],
    },
    ProviderComponents {
        provider_id: "krea_2_edit",
        components: &["krea_2_raw", "qwen_image"],
    },
    ProviderComponents {
        provider_id: "krea_2_raw",
        components: &["krea_2_raw", "qwen_image"],
    },
    ProviderComponents {
        provider_id: "krea_2_turbo",
        components: &["krea_2_turbo", "qwen_image"],
    },
    // --- lens -------------------------------------------------------------------------------
    // `SceneWorks/Lens` assigns three licences across three directories: the `text_encoder/` is
    // `openai/gpt-oss-20b` and the `vae/` is the FLUX.2-dev codec that repository's own README
    // attributes. The `transformer/` half has no row — both `microsoft/Lens` and
    // `microsoft/Lens-Turbo` 404, so its MIT survives only second-hand. Both ids share the encoder
    // and the VAE.
    ProviderComponents {
        provider_id: "lens",
        components: &["gpt_oss_20b", "flux2_dev"],
    },
    ProviderComponents {
        provider_id: "lens_turbo",
        components: &["gpt_oss_20b", "flux2_dev"],
    },
    // --- ltx --------------------------------------------------------------------------------
    // One bundled `Lightricks/LTX-2.3` file carries the AV DiT, both VAEs, the vocoder and the
    // connector; the Gemma-3-12B-IT text encoder is a separate required snapshot. On Candle the
    // **generator** id is `ltx_2_3_distilled` and `ltx_2_3` is trainer-only — the mirror image of
    // MLX, where one `ltx_2_3` id serves both kinds.
    ProviderComponents {
        provider_id: "ltx_2_3",
        components: &["ltx_2_3", "gemma_3_12b_it"],
    },
    ProviderComponents {
        provider_id: "ltx_2_3_distilled",
        components: &["ltx_2_3", "gemma_3_12b_it"],
    },
    // --- mochi ------------------------------------------------------------------------------
    // The AsymmVAE ships inside the Mochi snapshot and rides its row; the T5-XXL encoder is the
    // same `google/t5-v1_1-xxl` FLUX and Chroma condition on, which is what the crate names
    // (`candle-gen-mochi/src/config.rs`).
    ProviderComponents {
        provider_id: "mochi_1",
        components: &["mochi_1_preview", "t5_v1_1_xxl"],
    },
    // --- qwen-image -------------------------------------------------------------------------
    // Candle registers the base id only; control and edit are bespoke and unregistered. The VAE
    // rides the repository row, the Qwen2.5-VL tower is a pinned hole, and the fast tokenizer is
    // built in-house.
    ProviderComponents {
        provider_id: "qwen_image",
        components: &["qwen_image"],
    },
    // --- sana -------------------------------------------------------------------------------
    // Each SANA repository ships its own DC-AE autoencoder, so the VAE rides the DiT row; the
    // Gemma-2-2B-IT caption encoder is a separate snapshot.
    ProviderComponents {
        provider_id: "sana_1600m",
        components: &["sana_1600m_1024px_diffusers", "gemma_2_2b_it"],
    },
    ProviderComponents {
        provider_id: "sana_sprint_1600m",
        components: &["sana_sprint_1_6b_1024px_diffusers", "gemma_2_2b_it"],
    },
    // --- scail2 -----------------------------------------------------------------------------
    // The DiT's own repository is the only settled component: the stock Wan2.1 VAE, the UMT5 tower
    // and the open-CLIP ViT-H visual tower this route loads are all pinned holes, because the code
    // names the architecture and stops. **This is not the same case as `bernini` / `wan_vace`
    // above** even though all three drive `candle-gen-wan`'s `Umt5Encoder` + `WanVae16` from a
    // `text_encoder/` and `vae/` under their own snapshot root — see the module doc's
    // "Reusing a module is not provenance" section for why the three diverge.
    ProviderComponents {
        provider_id: "scail2_14b",
        components: &["scail_2"],
    },
    // --- sd3 (SD3.5) ------------------------------------------------------------------------
    // Three text encoders — CLIP-L (a pinned hole), OpenCLIP-bigG and T5-XXL — each keyed at the
    // upstream `candle-gen-sd3/src/clip_tokenizer.rs` names for it. The `-large-turbo` and
    // `-medium` DiTs have no row (no declared string was recorded for either), so those two ids
    // carry their encoders alone.
    ProviderComponents {
        provider_id: "sd3_5_large",
        components: &[
            "stable_diffusion_3_5_large",
            "clip_vit_bigg_14_laion2b",
            "t5_v1_1_xxl",
        ],
    },
    ProviderComponents {
        provider_id: "sd3_5_large_turbo",
        components: &["clip_vit_bigg_14_laion2b", "t5_v1_1_xxl"],
    },
    ProviderComponents {
        provider_id: "sd3_5_medium",
        components: &["clip_vit_bigg_14_laion2b", "t5_v1_1_xxl"],
    },
    // --- sdxl -------------------------------------------------------------------------------
    // One id, two interchangeable bases (`stabilityai/stable-diffusion-xl-base-1.0` and
    // `SG161222/RealVisXL_V5.0`), both listed because either may be provisioned.
    // `madebyollin/sdxl-vae-fp16-fix` is a **required** caller-staged component here and has no
    // counterpart on the MLX side, which reads the in-snapshot VAE at f32 — the sharpest
    // per-backend component difference in the lane. The bigG tokenizer/tower is keyed upstream;
    // CLIP-L is a pinned hole.
    ProviderComponents {
        provider_id: "sdxl",
        components: &[
            "stable_diffusion_xl_base_1_0",
            "realvisxl_v5_0",
            "sdxl_vae_fp16_fix",
            "clip_vit_bigg_14_laion2b",
        ],
    },
    // --- seedvr2 ----------------------------------------------------------------------------
    // Two DiT files and one shared VAE from one community re-host; the bare `seedvr2` id resolves
    // to the 3B checkpoint. No text encoder, no tokenizer.
    ProviderComponents {
        provider_id: "seedvr2",
        components: &["seedvr2_comfyui"],
    },
    ProviderComponents {
        provider_id: "seedvr2_3b",
        components: &["seedvr2_comfyui"],
    },
    ProviderComponents {
        provider_id: "seedvr2_7b",
        components: &["seedvr2_comfyui"],
    },
    // --- sensenova --------------------------------------------------------------------------
    // A unified model: no VAE and no separate text encoder. `_fast` merges the 8-step distill LoRA
    // before quantization on every run, so it carries both rows.
    ProviderComponents {
        provider_id: "sensenova_u1_8b",
        components: &["sensenova_u1_8b_mot"],
    },
    ProviderComponents {
        provider_id: "sensenova_u1_8b_fast",
        components: &["sensenova_u1_8b_mot", "sensenova_u1_8b_mot_loras"],
    },
    // --- svd --------------------------------------------------------------------------------
    // UNet, temporal-decoder VAE and the ViT-H image encoder all ship in one ungated snapshot.
    ProviderComponents {
        provider_id: "svd_xt",
        components: &["stable_video_diffusion_img2vid_xt"],
    },
    // --- wan --------------------------------------------------------------------------------
    // Candle registers four generators and one trainer; `wan2_2_vace_fun_14b` is MLX-only. The
    // GGUF re-host is a Candle-only provisioning of the 5B DiT and is listed because a run may
    // load it in place of the safetensors tree. `wan_vace` is 14B-only on Candle (no 1.3B preset
    // exists in `candle-gen-wan/src/config.rs`) and its repository ships the transformer alone, so
    // it also rides the stock Wan UMT5 and the z16 VAE `candle-gen-wan/src/vae16.rs` ports from
    // `Wan-AI/Wan2.2-T2V-A14B-Diffusers`. The two A14B ids ship that VAE in their own snapshots.
    ProviderComponents {
        provider_id: "wan2_2_i2v_14b",
        components: &["wan2_2_i2v_a14b", "umt5_xxl"],
    },
    ProviderComponents {
        provider_id: "wan2_2_t2v_14b",
        components: &["wan2_2_t2v_a14b", "umt5_xxl"],
    },
    ProviderComponents {
        provider_id: "wan2_2_ti2v_5b",
        components: &["wan2_2_ti2v_5b", "wan2_2_ti2v_5b_gguf", "umt5_xxl"],
    },
    ProviderComponents {
        provider_id: "wan_vace",
        components: &["wan2_1_vace_14b_diffusers", "umt5_xxl", "wan2_2_t2v_a14b"],
    },
    // --- z-image ----------------------------------------------------------------------------
    // Two DiT repositories, identical architecture; the 16-channel VAE and the tokenizer ship in
    // each snapshot. The Qwen3-style text encoder is a pinned hole. The two `*_control` routes
    // exist as code but are bespoke and unregistered on this backend.
    ProviderComponents {
        provider_id: "z_image",
        components: &["z_image"],
    },
    ProviderComponents {
        provider_id: "z_image_turbo",
        components: &["z_image_turbo"],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        license_table_conformance_errors, provider_terms, resolve_component, LicenseTerm,
        LICENSE_FAMILIES, MEDIA_COMPONENT_LICENSES,
    };
    use std::collections::BTreeSet;

    /// The reason string shared by the six `microsoft/Mage-Flow*` ids, so the six cannot drift into
    /// six slightly different explanations of one finding.
    const MAGE_FLOW_REASON: &str = "every component is a NOT FOUND / UNDETERMINED hole — all six \
                                    `microsoft/Mage-Flow*` repositories 404 and the Qwen3-VL-4B \
                                    tower bundled in each `text_encoder/` could not be named";

    /// The registered Candle provider ids whose components are **all** pinned holes in the shared
    /// table, with the reason — so they have no [`PROVIDER_COMPONENTS`] row and a consumer reading
    /// this surface gets an empty licence disclosure for them.
    ///
    /// **This is an annotation, never a filter.** Every id below is registered, ships, and renders
    /// exactly as it did before this module existed; what is absent is our metadata, which is a
    /// hole for CI to report and never a reason to withhold a provider. It is deliberately
    /// `#[cfg(test)]` rather than a public const, so no gate can quietly use it to suppress the ids
    /// it names — the epic's product constraint rules out a `PENDING_*`-shaped allowlist keyed on
    /// missing licence data, and a `pub` list in this repo is a compatibility boundary that cannot
    /// be withdrawn once something starts reading it. `mlx-gen-catalog` settled the same question
    /// the same way (sc-16666). The nine ids and their reasons are also in this module's rustdoc,
    /// so nothing a reader needs is behind `cfg(test)`.
    ///
    /// A row cannot simply be written empty for these, because
    /// [`license_table_conformance_errors`] rejects a provider mapping to no components — an empty
    /// list is indistinguishable from a forgotten one, so the surface spells the absence out here
    /// instead.
    const IDS_WITHOUT_A_RESOLVABLE_COMPONENT: &[(&str, &str)] = &[
        ("mage_flow", MAGE_FLOW_REASON),
        ("mage_flow_base", MAGE_FLOW_REASON),
        ("mage_flow_turbo", MAGE_FLOW_REASON),
        ("mage_flow_edit", MAGE_FLOW_REASON),
        ("mage_flow_edit_base", MAGE_FLOW_REASON),
        ("mage_flow_edit_turbo", MAGE_FLOW_REASON),
        // The captioner id *is* its upstream repository id, and the only declaration anywhere is
        // this repository's own `release/real-weight-models.toml`.
        (
            "fancyfeast/llama-joycaption-beta-one-hf-llava",
            "SECOND-HAND ONLY — `llama_joycaption_beta_one_hf_llava` is a pinned hole",
        ),
        // Both embedder ids load one `openai/clip-vit-large-patch14` snapshot — one checkpoint, two
        // heads, two ids — and that card declares nothing.
        (
            "clip_vit_l14",
            "NOT FOUND — `clip_vit_large_patch14` is a pinned hole",
        ),
        (
            "clip_vit_l14_text",
            "NOT FOUND — `clip_vit_large_patch14` is a pinned hole",
        ),
    ];

    /// Every registered id across every provider kind — the set this mapping is measured against.
    /// Candle registers generators, trainers, one captioner and two embedders; `transforms` is
    /// empty here and is still walked so a future registration cannot slip past unmapped.
    fn registered_ids() -> BTreeSet<String> {
        let registry = super::super::provider_registry().expect("catalog builds");
        let mut ids: BTreeSet<String> = BTreeSet::new();
        ids.extend(
            registry
                .generators()
                .map(|r| (r.descriptor)().id.to_string()),
        );
        ids.extend(registry.trainers().map(|r| (r.descriptor)().id.to_string()));
        ids.extend(
            registry
                .captioners()
                .map(|r| (r.descriptor)().id.to_string()),
        );
        ids.extend(
            registry
                .image_embedders()
                .map(|r| (r.descriptor)().id.to_string()),
        );
        ids.extend(
            registry
                .text_embedders()
                .map(|r| (r.descriptor)().id.to_string()),
        );
        ids.extend(
            registry
                .transforms()
                .map(|r| (r.descriptor)().id.to_string()),
        );
        ids
    }

    /// The whole table — the shared families, the shared component rows and **this** mapping — is
    /// well-formed. This is the ship-gate `gen-core`'s own test could only run with the provider
    /// section empty, because the provider section is per-backend and lives here.
    #[test]
    fn candle_licence_table_is_conformant() {
        assert_eq!(
            license_table_conformance_errors(
                LICENSE_FAMILIES,
                MEDIA_COMPONENT_LICENSES,
                PROVIDER_COMPONENTS
            ),
            Vec::<String>::new()
        );
    }

    /// The mapping is sorted by id and every key it names resolves. Sorting keeps a future diff
    /// local; the resolution check is asserted directly as well as through the gate because an
    /// unresolvable key contributes **nothing** to a derived union — it under-reports rather than
    /// failing.
    #[test]
    fn mapping_is_sorted_and_every_key_resolves() {
        assert_eq!(PROVIDER_COMPONENTS.len(), 47);
        let ids: Vec<&str> = PROVIDER_COMPONENTS.iter().map(|p| p.provider_id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "PROVIDER_COMPONENTS must be sorted by id");

        for provider in PROVIDER_COMPONENTS {
            assert!(
                !provider.components.is_empty(),
                "provider {:?} maps to no components; spell that absence out in \
                 IDS_WITHOUT_A_RESOLVABLE_COMPONENT instead",
                provider.provider_id
            );
            for key in provider.components {
                assert!(
                    resolve_component(MEDIA_COMPONENT_LICENSES, key).is_some(),
                    "provider {:?} names component {key:?}, which is not in the shared table",
                    provider.provider_id
                );
            }
        }
    }

    /// Every registered Candle id is either mapped to at least one component or named — with a
    /// reason — in [`IDS_WITHOUT_A_RESOLVABLE_COMPONENT`], and nothing is both. A mapping for an id
    /// this catalog does not register is equally an error: it would publish a licence disclosure
    /// for a render that cannot happen on this backend.
    #[test]
    fn every_registered_id_is_mapped_or_a_named_hole() {
        let registered = registered_ids();
        assert_eq!(
            registered.len(),
            56,
            "56 distinct Candle provider ids: 51 generators + 7 trainers (5 of them also generator \
             ids) + 1 captioner + 2 embedders"
        );

        let mapped: BTreeSet<&str> = PROVIDER_COMPONENTS.iter().map(|p| p.provider_id).collect();
        let holes: BTreeSet<&str> = IDS_WITHOUT_A_RESOLVABLE_COMPONENT
            .iter()
            .map(|(id, _)| *id)
            .collect();
        assert_eq!(holes.len(), IDS_WITHOUT_A_RESOLVABLE_COMPONENT.len());
        assert_eq!(holes.len(), 9);
        assert_eq!(mapped.len() + holes.len(), registered.len());

        for id in &registered {
            let is_mapped = mapped.contains(id.as_str());
            let is_hole = holes.contains(id.as_str());
            assert!(
                is_mapped ^ is_hole,
                "registered provider {id:?} is {}; every id must be exactly one of mapped or a \
                 named hole",
                if is_mapped {
                    "both mapped and listed as a hole"
                } else {
                    "neither mapped nor listed as a hole"
                }
            );
        }
        for id in mapped.iter().chain(holes.iter()) {
            assert!(
                registered.contains(*id),
                "{id:?} is on the licence surface but this catalog registers no such provider"
            );
        }
    }

    /// Which mapped ids have an **incomplete** list, and which pinned hole each is missing.
    ///
    /// Pinned so a later silent fill-in is a deliberate diff: filling a hole in `gen-core`'s shared
    /// table means deleting its key from `known_holes_stay_absent` **and** adding it to the
    /// affected ids here, which is the moment to check that the new row really is the artifact this
    /// route loads. Same shape, same purpose, one layer up.
    ///
    /// An entry here is not a defect in this module — it is the shared table's gap, surfaced at the
    /// layer a consumer reads. Nothing about it withholds the id.
    #[test]
    fn ids_with_incomplete_lists_are_pinned() {
        // `(provider id, the pinned hole keys this id loads and therefore cannot disclose)`.
        const INCOMPLETE: &[(&str, &[&str])] = &[
            (
                "anima_aesthetic",
                &["anima_qwen3_0_6b_text_encoder", "anima_qwen_image_vae"],
            ),
            (
                "anima_base",
                &["anima_qwen3_0_6b_text_encoder", "anima_qwen_image_vae"],
            ),
            (
                "anima_turbo",
                &["anima_qwen3_0_6b_text_encoder", "anima_qwen_image_vae"],
            ),
            ("boogu_image", &["boogu_qwen3_vl_8b", "boogu_flux1_vae"]),
            (
                "boogu_image_edit",
                &["boogu_qwen3_vl_8b", "boogu_flux1_vae"],
            ),
            (
                "boogu_image_turbo",
                &["boogu_qwen3_vl_8b", "boogu_flux1_vae"],
            ),
            ("flux1_dev", &["clip_vit_large_patch14"]),
            ("flux1_schnell", &["clip_vit_large_patch14"]),
            ("flux2_dev", &["flux2_dev_mistral3_tower"]),
            ("flux2_klein_9b", &["flux2_klein_qwen3_text_encoder"]),
            ("ideogram_4", &["ideogram_qwen3_vl_8b"]),
            ("ideogram_4_turbo", &["ideogram_qwen3_vl_8b"]),
            (
                "kolors",
                &["kolors_diffusers", "sceneworks_kolors_chatglm3_tokenizer"],
            ),
            ("krea_2_control", &["krea_qwen3_vl_4b"]),
            ("krea_2_edit", &["krea_qwen3_vl_4b"]),
            ("krea_2_raw", &["krea_qwen3_vl_4b"]),
            ("krea_2_turbo", &["krea_qwen3_vl_4b"]),
            // Convention on the two Lens ids: each names **its own** deleted upstream plus the
            // shared `lens_transformer` key, because both load an in-snapshot `transformer/`
            // (`candle-gen-lens/src/lib.rs` `component_vb("transformer", …)` on either route) whose
            // MIT survives only second-hand. Listing the shared key on one id and not the other
            // would make the two disclosures differ for identical evidence.
            ("lens", &["microsoft_lens", "lens_transformer"]),
            ("lens_turbo", &["microsoft_lens_turbo", "lens_transformer"]),
            ("qwen_image", &["qwen_image_qwen2_5_vl"]),
            // All three bundled auxiliaries stay holes; the module doc's "Reusing a module is not
            // provenance" section is why this id is short where `bernini` and `wan_vace` are not.
            (
                "scail2_14b",
                &["scail2_umt5", "scail2_open_clip_vit_h", "scail2_wan2_1_vae"],
            ),
            ("sd3_5_large", &["clip_vit_large_patch14"]),
            (
                "sd3_5_large_turbo",
                &["stable_diffusion_3_5_large_turbo", "clip_vit_large_patch14"],
            ),
            (
                "sd3_5_medium",
                &["stable_diffusion_3_5_medium", "clip_vit_large_patch14"],
            ),
            ("sdxl", &["clip_vit_large_patch14"]),
            ("z_image", &["z_image_qwen3_text_encoder"]),
            ("z_image_turbo", &["z_image_qwen3_text_encoder"]),
        ];

        assert_eq!(INCOMPLETE.len(), 27);
        let mapped: BTreeSet<&str> = PROVIDER_COMPONENTS.iter().map(|p| p.provider_id).collect();
        for (id, missing) in INCOMPLETE {
            assert!(
                mapped.contains(id),
                "{id:?} is listed as incomplete but has no mapping row"
            );
            assert!(!missing.is_empty());
            for key in *missing {
                assert!(
                    resolve_component(MEDIA_COMPONENT_LICENSES, key).is_none(),
                    "{id:?} names {key:?} as a missing component, but the shared table now carries \
                     a row for it — add it to this id's `components` and delete it here"
                );
            }
        }

        // The complement is asserted too, so an id cannot quietly move between the two sets: a
        // mapping that gained a component without its hole being filled would otherwise look fine.
        let incomplete: BTreeSet<&str> = INCOMPLETE.iter().map(|(id, _)| *id).collect();
        let complete: Vec<&str> = PROVIDER_COMPONENTS
            .iter()
            .map(|p| p.provider_id)
            .filter(|id| !incomplete.contains(id))
            .collect();
        assert_eq!(
            complete,
            vec![
                "bernini",
                "bernini_renderer",
                "chroma1_base",
                "chroma1_flash",
                "chroma1_hd",
                "ltx_2_3",
                "ltx_2_3_distilled",
                "mochi_1",
                "sana_1600m",
                "sana_sprint_1600m",
                "seedvr2",
                "seedvr2_3b",
                "seedvr2_7b",
                "sensenova_u1_8b",
                "sensenova_u1_8b_fast",
                "svd_xt",
                "wan2_2_i2v_14b",
                "wan2_2_t2v_14b",
                "wan2_2_ti2v_5b",
                "wan_vace",
            ]
        );
    }

    /// The real per-backend differences, pinned rather than papered over (`CONTRIBUTING.md`).
    ///
    /// Component **rows** are shared and must not differ; only which ids exist and what they load
    /// may. These are the Candle side of the divergences the census enumerated.
    #[test]
    fn per_backend_divergences_are_explicit() {
        let components = |id: &str| {
            PROVIDER_COMPONENTS
                .iter()
                .find(|p| p.provider_id == id)
                .unwrap_or_else(|| panic!("{id} is not mapped"))
                .components
        };

        // 1. SDXL's VAE is a REQUIRED separate component on Candle, because the base VAE NaNs in
        //    f16. MLX reads the in-snapshot `vae/` at f32 and lists no such component.
        assert!(components("sdxl").contains(&"sdxl_vae_fp16_fix"));

        // 2. The GGUF re-host of the 5B Wan DiT is a Candle-only provisioning route.
        assert!(components("wan2_2_ti2v_5b").contains(&"wan2_2_ti2v_5b_gguf"));

        // 3. LTX is inverted between the backends: `ltx_2_3_distilled` is the Candle **generator**
        //    and `ltx_2_3` is trainer-only. Both load the same checkpoints, which is why the two
        //    lists are equal — the divergence is in the ids, not the components.
        assert_eq!(components("ltx_2_3"), components("ltx_2_3_distilled"));

        // 4. `krea_2_control` is a Candle trainer id with no Candle generator; its overlay applies
        //    at `krea_2_turbo` inference, so it loads the Turbo snapshot.
        assert_eq!(components("krea_2_control"), components("krea_2_turbo"));

        // Rows in the shared table that no registered Candle id reaches. They are not missing rows
        // — they are checkpoints this backend's registered surface does not load, which is exactly
        // what one shared table plus a per-backend mapping exists to express.
        //
        // The pin is **total**: the set below is asserted equal to the whole unreferenced remainder,
        // not merely contained in it. A partial list would let a newly-added shared row sit
        // unreached and unexplained.
        let referenced: BTreeSet<&str> = PROVIDER_COMPONENTS
            .iter()
            .flat_map(|p| p.components.iter().copied())
            .collect();
        let pinned_elsewhere: BTreeSet<&str> = BTreeSet::from([
            // MLX-registered generator ids with no Candle sibling.
            "flux2_klein_9b_kv",
            "wan2_2_vace_fun_a14b",
            "z_image_fun_controlnet_union_2_1",
            "z_image_turbo_fun_controlnet_union_2_1",
            "qwen_image_2512",
            "qwen_image_2512_fun_controlnet_union",
            "qwen_image_edit",
            "flux1_dev_controlnet_union_pro_2_0",
            "krea_realtime_video",
            "wan2_1_t2v_14b_diffusers",
            "wan2_1_vace_1_3b_diffusers",
            // The bespoke Candle PuLID path and the overlay crates (sc-16668), which register
            // no provider id on this backend. Candle has no `sam2` crate at all, so both SAM 2.1
            // rows are unreached here as well.
            "pulid",
            "eva_clip",
            "facexlib_parsing_bisenet",
            "nvidia_pid_students",
            "sam3",
            "sam2_1_hiera_base_plus",
            "sam2_1_hiera_large",
            "instantid",
            "controlnet_openpose_sdxl_1_0",
            // Caller-selected add-ons, deliberately out of scope for a list with no optionality.
            "anima_official_loras",
            "hyper_sd_flux1_dev_8steps_lora",
            "ip_adapter",
            "kolors_ip_adapter_plus",
            "qwen_image_lightning",
            "qwen_image_edit_2511_lightning",
            "xlabs_flux_ip_adapter",
        ]);
        for elsewhere in &pinned_elsewhere {
            assert!(
                resolve_component(MEDIA_COMPONENT_LICENSES, elsewhere).is_some(),
                "{elsewhere:?} must still be a shared row"
            );
            assert!(
                !referenced.contains(elsewhere),
                "{elsewhere:?} is now reached by a registered Candle id; if that is correct, the \
                 mapping is right and this pin is stale"
            );
        }
        let unreferenced: BTreeSet<&str> = MEDIA_COMPONENT_LICENSES
            .iter()
            .map(|row| row.component)
            .filter(|component| !referenced.contains(component))
            .collect();
        assert_eq!(
            unreferenced, pinned_elsewhere,
            "every shared row no registered Candle id reaches must be pinned above with the reason \
             it is unreached"
        );
    }

    /// A provider's terms are **derived**, and the derivation is what makes a non-obvious component
    /// visible. Three checks a hand-authored composite would have got wrong.
    #[test]
    fn terms_are_derived_from_the_components_not_from_the_headline_model() {
        let terms = |id: &str| {
            let provider = PROVIDER_COMPONENTS
                .iter()
                .find(|p| p.provider_id == id)
                .unwrap_or_else(|| panic!("{id} is not mapped"));
            provider_terms(provider, MEDIA_COMPONENT_LICENSES, LICENSE_FAMILIES)
        };

        // `lens` looks like a Microsoft model and its headline transformer has no row at all; the
        // non-commercial term it names comes from the FLUX.2-dev VAE its `vae/` resolves to.
        let lens = terms("lens");
        assert!(lens.contains(&LicenseTerm::NonCommercialWeights));
        assert!(lens.contains(&LicenseTerm::GatedAccess));

        // `ltx_2_3_distilled` reaches a gated checkpoint even though `Lightricks/LTX-2.3` is not
        // gated: the Gemma text encoder it requires is, and gating rides the component row.
        let ltx = terms("ltx_2_3_distilled");
        assert!(ltx.contains(&LicenseTerm::GatedAccess));
        assert!(
            !resolve_component(MEDIA_COMPONENT_LICENSES, "ltx_2_3")
                .expect("LTX row")
                .gated
        );

        // A plain Chroma render names neither — Apache rows and nothing else.
        let chroma = terms("chroma1_hd");
        assert!(!chroma.contains(&LicenseTerm::NonCommercialWeights));
        assert!(!chroma.contains(&LicenseTerm::GatedAccess));
    }

    /// No registered Candle provider reaches an outputs restriction — the silence sc-16662's U3
    /// took deliberately, pinned at the layer a consumer actually joins over.
    ///
    /// `families.rs` pins that no *family* carries it and `components.rs` that no *component*
    /// reaches one; this pins the third and last layer. Several of these providers load weights
    /// whose licence restricts non-commercial **use of the weights** and says nothing about
    /// renders. Inferring the stronger term from that silence is the reading the epic declined to
    /// make, and if it were made anywhere it would be made here, in the union.
    #[test]
    fn no_registered_candle_provider_reaches_an_outputs_restriction() {
        for provider in PROVIDER_COMPONENTS {
            let terms = provider_terms(provider, MEDIA_COMPONENT_LICENSES, LICENSE_FAMILIES);
            assert!(
                !terms.contains(&LicenseTerm::NonCommercialOutputs),
                "provider {:?} reaches non_commercial_outputs",
                provider.provider_id
            );
        }
    }
}
