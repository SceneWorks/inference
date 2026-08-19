//! The **MLX** provider→component mapping (sc-16666) — which upstream checkpoints each id this
//! catalog registers loads.
//!
//! # Disclosure only
//!
//! Nothing here blocks, gates, degrades or withholds anything, and nothing added here ever should. A
//! row records which artifacts an id **loads**, so a consumer can show a user what the upstream
//! texts name. No code may branch on any of it, and an id whose component list is incomplete — or
//! absent entirely — still registers, still loads and still renders exactly as before. This module
//! adds no [`ProviderRegistry`](crate::ProviderRegistry) edge of any kind.
//!
//! # What lives here, and what does not
//!
//! A licence is a property of the *checkpoint*, so the rows themselves are the shared, tensor-neutral
//! table in [`gen_core::MEDIA_COMPONENT_LICENSES`] (sc-16665), read by **both** media catalogs. The
//! only part that legitimately differs per backend is which ids load which components, and that is
//! this module. The Candle twin is sc-16667.
//!
//! A provider's effective terms are **derived** from its components by [`gen_core::provider_terms`]
//! and never hand-authored here — a typed composite would be a second place to be wrong and could
//! drift from its own components.
//!
//! # The mapping rule: everything an id *can* load
//!
//! [`ProviderComponents`] carries no optionality flag, and the
//! audio lane settled the reading: ACE-Step's row lists its three **cover-only** sft modules beside
//! its always-loaded primary. So a row here is *every* artifact a load of that id can pull in —
//! required bases plus the optional ControlNets, IP-adapters, distill LoRAs and the PiD latent
//! decoder. The derived union answers "which upstream texts can a render through this id involve",
//! which is the question a licences page asks. It is deliberately **not** "which files did this
//! particular render touch": that needs the `LoadSpec`, which no static table has.
//!
//! The PiD seam is the case that makes this worth stating. `nvidia/PiD` is an optional decode overlay
//! on **thirteen** MLX provider crates, `mlx-gen-pid/src/lib.rs:23` records what NVIDIA's text names,
//! and not one consuming crate repeats it. Listing it on every consumer is what makes it visible.
//!
//! # Trainers share their generator's row
//!
//! All fifteen registered MLX trainer ids are also generator ids, and
//! [`ProviderComponents::provider_id`] is unique across the table — duplicates are rejected by
//! [`gen_core::license_table_conformance_errors`] because
//! they are not byte-stable under the manifest's stable sort. One row therefore serves both
//! registrations, which is also correct on the facts: a `sdxl` fine-tune reads the same checkpoints
//! the `sdxl` render does. 65 generators + 1 captioner + 2 embedders = **68 distinct registered ids**.
//!
//! # Incomplete rows, and nine ids with no row at all
//!
//! The shared table is deliberately incomplete: sc-16665 wrote 71 rows and pinned **46** checkpoints
//! as deliberately absent (UNDETERMINED upstream, NOT FOUND, AMBIGUOUS, NO DECLARED STRING,
//! SECOND-HAND ONLY) rather than guessing. A provider that loads one of those has a short list. That
//! is the honest state and it is never papered over by pointing at a plausible-looking neighbour row
//! — `scail2_umt5` is a pinned hole and is **not** resolved to the `umt5_xxl` row, because the
//! evidence pass could not establish that SCAIL-2's bundled copy is Google's.
//!
//! `mapped_ids_are_complete_or_pinned_incomplete` pins which ids are short and **which key is
//! missing**, and asserts each of those keys is still absent from the shared table — so the day
//! sc-16665 gains a row, this test fails and forces the list to be extended. 39 of the 59 mapped ids
//! are short today.
//!
//! Nine registered ids load **nothing** the shared table covers, so they have no row at all. They
//! register, load and render unchanged; what is missing is our metadata, and it is pinned by
//! `every_registered_id_maps_or_is_a_pinned_hole`:
//!
//! | id(s) | why |
//! | --- | --- |
//! | the six `mage_flow*` generators | every `microsoft/Mage-Flow*` repository 404s and the bundled Qwen3-VL-4B tower's upstream is unnamed, so neither the DiT, the VAE nor the encoder has a row |
//! | `fancyfeast/llama-joycaption-beta-one-hf-llava` | one snapshot carries the whole captioner, and its only declaration is second-hand (`release/real-weight-models.toml`) |
//! | `clip_vit_l14`, `clip_vit_l14_text` | both read one `openai/clip-vit-large-patch14` snapshot, whose card declares nothing |
//!
//! # The strictest term in the catalog reaches nothing today
//!
//! Epic 16660's R6 names insightface antelopev2 as the catalog's strictest text and `pulid_flux` as
//! the id that carries it. `pulid_flux` **is** MLX-registered and does depend on the unregistered
//! `mlx-gen-face` stack, so its row lists that stack's components — but two of the three
//! (`antelopev2_arcface_glintr100`, `antelopev2_scrfd_10g`) are pinned holes, so the
//! `insightface-research-only` family currently resolves from no component and appears in no union.
//! `pulid_flux_lists_the_face_stack_it_loads` pins both halves of that, so the gap is visible rather
//! than inferred from an absence.

use mlx_gen::gen_core::{self, ComponentLicense, LicenseFamily, ProviderComponents};

/// Which components each id this catalog registers loads — the per-backend half of the licence
/// surface, in catalog registration order.
///
/// 60 rows over 69 registered ids: the fifteen trainer ids reuse their generator's row, and nine ids
/// load nothing the shared table covers (see the module note). Every key resolves into
/// [`gen_core::MEDIA_COMPONENT_LICENSES`].
pub const MLX_MEDIA_PROVIDER_COMPONENTS: &[ProviderComponents] = &[
    // --- anima -------------------------------------------------------------------------------
    // All three ids differ only in which DiT file inside `circlestone-labs/Anima` they read. The
    // bundled Qwen3-0.6B encoder and Qwen-Image VAE are pinned holes.
    ProviderComponents {
        provider_id: "anima_base",
        components: &["anima", "anima_official_loras"],
    },
    ProviderComponents {
        provider_id: "anima_aesthetic",
        components: &["anima", "anima_official_loras"],
    },
    ProviderComponents {
        provider_id: "anima_turbo",
        components: &["anima", "anima_official_loras"],
    },
    // --- bernini -----------------------------------------------------------------------------
    // `bernini` adds the planner, connector, ViT-decoder head and mask tokens — all prefixes of the
    // one `ByteDance/Bernini-Diffusers` index, so both ids carry the same components. The converter
    // deliberately reads the standalone stock-Wan2.2 directory in preference to Bernini's own
    // bundled UMT5 copy (`mlx-gen-bernini/src/convert.rs:11`).
    ProviderComponents {
        provider_id: "bernini_renderer",
        components: &["bernini_diffusers", "wan2_2_t2v_a14b", "umt5_xxl"],
    },
    ProviderComponents {
        provider_id: "bernini",
        components: &["bernini_diffusers", "wan2_2_t2v_a14b", "umt5_xxl"],
    },
    // --- boogu -------------------------------------------------------------------------------
    // Three separate snapshots, three separate DiT rows. The bundled Qwen3-VL-8B condition encoder
    // and the bundled "FLUX.1 VAE" are pinned holes — the latter because Boogu's own README does not
    // say whether that is FLUX.1 [dev] or FLUX.1-schnell, two different texts.
    ProviderComponents {
        provider_id: "boogu_image",
        components: &["boogu_image_0_1_base", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "boogu_image_turbo",
        components: &["boogu_image_0_1_turbo", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "boogu_image_edit",
        components: &["boogu_image_0_1_edit", "nvidia_pid_students"],
    },
    // --- chroma ------------------------------------------------------------------------------
    // Chroma's Apache declaration covers the `vae/` and `text_encoder/` its own repository ships;
    // the T5-XXL inside is keyed on its own upstream per the redistributed-component rule.
    ProviderComponents {
        provider_id: "chroma1_hd",
        components: &["chroma1_hd", "t5_v1_1_xxl", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "chroma1_base",
        components: &["chroma1_base", "t5_v1_1_xxl", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "chroma1_flash",
        components: &["chroma1_flash", "t5_v1_1_xxl", "nvidia_pid_students"],
    },
    // --- flux (FLUX.1) -----------------------------------------------------------------------
    // The Hyper-FLUX 8-step LoRA is dev-only and schnell never advertises it
    // (`mlx-gen-flux/src/config.rs:24`); the XLabs IP-adapter rides `LoadSpec::ip_adapter` on both.
    // The bundled CLIP-L text encoder is a pinned hole on every FLUX.1 route.
    ProviderComponents {
        provider_id: "flux1_schnell",
        components: &[
            "flux1_schnell",
            "t5_v1_1_xxl",
            "xlabs_flux_ip_adapter",
            "nvidia_pid_students",
        ],
    },
    ProviderComponents {
        provider_id: "flux1_dev",
        components: &[
            "flux1_dev",
            "t5_v1_1_xxl",
            "xlabs_flux_ip_adapter",
            "hyper_sd_flux1_dev_8steps_lora",
            "nvidia_pid_students",
        ],
    },
    // A genuinely separate ControlNet checkpoint, not a sampler variation.
    ProviderComponents {
        provider_id: "flux1_dev_control",
        components: &[
            "flux1_dev",
            "t5_v1_1_xxl",
            "xlabs_flux_ip_adapter",
            "hyper_sd_flux1_dev_8steps_lora",
            "flux1_dev_controlnet_union_pro_2_0",
            "nvidia_pid_students",
        ],
    },
    // --- flux2 (FLUX.2) ----------------------------------------------------------------------
    // The edit variants add no checkpoint — reference images are VAE-encoded and concatenated as
    // tokens (`mlx-gen-flux2/src/config.rs:99`). `-klein-9b-kv` is a separately distilled snapshot,
    // MLX-only. Every FLUX.2 route's bundled language tower is a pinned hole, and the dev route's
    // Pixtral vision tower and multimodal projector (MLX-only, caption upsampling) are two more.
    ProviderComponents {
        provider_id: "flux2_klein_9b",
        components: &["flux2_klein_9b", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "flux2_klein_9b_edit",
        components: &["flux2_klein_9b", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "flux2_klein_9b_kv_edit",
        components: &["flux2_klein_9b_kv", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "flux2_dev",
        components: &["flux2_dev", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "flux2_dev_edit",
        components: &["flux2_dev", "nvidia_pid_students"],
    },
    // The `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` branch this id adds is an AMBIGUOUS hole:
    // it ships BFL's v2.0 text where the base ships v2.1, and whether that is one family or two is
    // an open question on sc-16665's sign-off list.
    ProviderComponents {
        provider_id: "flux2_dev_control",
        components: &["flux2_dev", "nvidia_pid_students"],
    },
    // --- ideogram ----------------------------------------------------------------------------
    // `ideogram_4_turbo` is the same base weights plus the bundled TurboTime LoRA and no
    // unconditional DiT (`mlx-gen-ideogram/src/config.rs:15`).
    ProviderComponents {
        provider_id: "ideogram_4",
        components: &["ideogram_4_fp8", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "ideogram_4_turbo",
        components: &[
            "ideogram_4_fp8",
            "ideogram_4_turbotime_lora",
            "nvidia_pid_students",
        ],
    },
    // --- kolors ------------------------------------------------------------------------------
    // The shortest row relative to what the id loads: the `Kwai-Kolors/Kolors-diffusers` repository
    // that supplies the U-Net and the VAE is itself an AMBIGUOUS hole (sc-16662 U6), as are the
    // derived fast tokenizer, the pose ControlNet and the IP-adapter's bundled CLIP image tower.
    // The ChatGLM3-6B encoder inside that snapshot is keyed on its own governing document.
    ProviderComponents {
        provider_id: "kolors",
        components: &[
            "chatglm3_6b",
            "kolors_ip_adapter_plus",
            "nvidia_pid_students",
        ],
    },
    // --- krea --------------------------------------------------------------------------------
    // Only the DiT differs across the five ids: `krea_2_edit` runs on Raw, `krea_2_turbo_edit` the
    // same edit surface on Turbo (`mlx-gen-krea/src/lib.rs:34`). The native VAE is
    // `Qwen/Qwen-Image`'s, as Krea's own `vae/config.json` declares. Every id can instead load the
    // Wan 2.1 VAE through the registered alternate-decoder seam, so its component is part of each
    // composite row just like the optional PiD component. The Qwen3-VL-4B tower is a pinned hole.
    ProviderComponents {
        provider_id: "krea_2_turbo",
        components: &[
            "krea_2_turbo",
            "qwen_image",
            "nvidia_pid_students",
            "wan2_1_t2v_14b_diffusers",
        ],
    },
    ProviderComponents {
        provider_id: "krea_2_raw",
        components: &[
            "krea_2_raw",
            "qwen_image",
            "nvidia_pid_students",
            "wan2_1_t2v_14b_diffusers",
        ],
    },
    ProviderComponents {
        provider_id: "krea_2_edit",
        components: &[
            "krea_2_raw",
            "qwen_image",
            "nvidia_pid_students",
            "wan2_1_t2v_14b_diffusers",
        ],
    },
    ProviderComponents {
        provider_id: "krea_2_turbo_edit",
        components: &[
            "krea_2_turbo",
            "qwen_image",
            "nvidia_pid_students",
            "wan2_1_t2v_14b_diffusers",
        ],
    },
    // The in-house pose branch (`SceneWorks/krea2-pose-controlnet-beta`) is a NOT FOUND hole — it
    // declares `experimental-research-only` with no text behind it.
    ProviderComponents {
        provider_id: "krea_2_turbo_control",
        components: &[
            "krea_2_turbo",
            "qwen_image",
            "nvidia_pid_students",
            "wan2_1_t2v_14b_diffusers",
        ],
    },
    // --- krea-realtime -----------------------------------------------------------------------
    // Its "stock Wan" trio is Wan **2.1**, not 2.2 — see the `WAN2_1_T2V_14B_DIFFUSERS` row, which
    // corrects the census on this point.
    ProviderComponents {
        provider_id: "krea_realtime_14b",
        components: &[
            "krea_realtime_video",
            "wan2_1_t2v_14b_diffusers",
            "umt5_xxl",
        ],
    },
    // --- lens --------------------------------------------------------------------------------
    // Both ids resolve to the same two components because both DiTs are holes: Microsoft pulled the
    // `microsoft/Lens` repository (`mlx-gen-lens/src/training.rs:993`) and the rehost's MIT survives
    // only second-hand. What does resolve is the `SceneWorks/Lens` split its own README states — the
    // `text_encoder/` half is gpt-oss-20b, the `vae/` half is FLUX.2-dev's.
    ProviderComponents {
        provider_id: "lens_turbo",
        components: &["gpt_oss_20b", "flux2_dev", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "lens",
        components: &["gpt_oss_20b", "flux2_dev", "nvidia_pid_students"],
    },
    // --- ltx ---------------------------------------------------------------------------------
    // One bundled `Lightricks/LTX-2.3` file carries the DiT, both VAEs, the vocoder and the
    // connector; the three upsamplers come from the same repository. The Gemma-3-12B-IT text encoder
    // is a separate snapshot, and MLX's bf16 re-host resolves to the same licensor row as Candle's.
    // The MLX-only uncensored prompt enhancer is an AMBIGUOUS hole.
    ProviderComponents {
        provider_id: "ltx_2_3",
        components: &["ltx_2_3", "gemma_3_12b_it"],
    },
    // --- minimax-h3 --------------------------------------------------------------------------
    // One repository, two licences: the DiT / VAEs / tokenizer are the territorially-exclusive
    // MiniMax H3 Community License, while the bundled `text_encoder/` is byte-identical upstream
    // Qwen3-VL-32B-Instruct under Apache-2.0. Both are rowed, because both are loaded.
    ProviderComponents {
        provider_id: "minimax_h3",
        components: &["minimax_h3", "qwen3_vl_32b_instruct"],
    },
    // --- mochi -------------------------------------------------------------------------------
    // No adapters, ControlNets, refiners or safety models.
    ProviderComponents {
        provider_id: "mochi_1",
        components: &["mochi_1_preview", "t5_v1_1_xxl"],
    },
    // --- pulid -------------------------------------------------------------------------------
    // MLX-only: the Candle twin is a plain struct the worker drives and registers nothing. The row
    // reaches through the unregistered `mlx-gen-face` crate to the artifacts that crate loads, which
    // is the whole point of a component-keyed table — but two of the three face checkpoints are
    // pinned holes, so `insightface-research-only` resolves from nothing today. MLX PuLID has no PiD
    // edge (`mlx-gen-pulid/Cargo.toml`); the Candle twin does.
    ProviderComponents {
        provider_id: "pulid_flux",
        components: &[
            "flux1_dev",
            "t5_v1_1_xxl",
            "pulid",
            "eva_clip",
            "facexlib_parsing_bisenet",
        ],
    },
    // --- qwen-image --------------------------------------------------------------------------
    // The clearest case in the lane of ids that load a **different base checkpoint**, not just a
    // different sampler: control is `Qwen/Qwen-Image-2512`, edit is `Qwen/Qwen-Image-Edit`. Each
    // route's bundled Qwen2.5-VL tower is a pinned hole.
    //
    // All three routes carry a Lightning distill LoRA, but they do NOT all carry the same one — the
    // split follows the transformer config, not the base repository name. T2I and **control** both
    // build the tower through `loader::load_transformer` (`QwenTransformerConfig::qwen_image()`),
    // so the 2512 control base is the same module tree the T2I distill LoRA maps onto; Edit alone
    // takes `load_transformer_edit` (`zero_cond_t: true`), which is why it names the separate
    // `Qwen-Image-Edit-2511-Lightning`. Control reaches the recipe end-to-end: its descriptor
    // advertises `qwen_samplers()` (the `lightning` profile) and `supports_lora`, and
    // `model_control.rs` applies `spec.adapters` onto that base transformer via the same
    // `apply_qwen_adapters` T2I uses, then drops the negative branch under `params.is_lightning`
    // (CFG-distilled → one forward/step). No 2512-specific Lightning repository is named anywhere in
    // `mlx-gen-qwen-image`, and `sampler.rs` states the distill LoRA is supplied via `spec.adapters`
    // with `lightx2v/Qwen-Image-Lightning` as the named instance — so that is the artifact the
    // control id can load, and the row carries it. The distiller trained it against the 2509 base,
    // which is a parity question, not a licence one: the edge exists because the id can load the
    // artifact.
    ProviderComponents {
        provider_id: "qwen_image",
        components: &[
            "qwen_image",
            "qwen_image_lightning",
            "nvidia_pid_students",
            "wan2_1_t2v_14b_diffusers",
        ],
    },
    ProviderComponents {
        provider_id: "qwen_image_control",
        components: &[
            "qwen_image_2512",
            "qwen_image_2512_fun_controlnet_union",
            "qwen_image_lightning",
            "nvidia_pid_students",
            "wan2_1_t2v_14b_diffusers",
        ],
    },
    ProviderComponents {
        provider_id: "qwen_image_edit",
        components: &[
            "qwen_image_edit",
            "qwen_image_edit_2511_lightning",
            "nvidia_pid_students",
            "wan2_1_t2v_14b_diffusers",
        ],
    },
    // --- sana --------------------------------------------------------------------------------
    // The DC-AE tensors load from each SANA repository's own `vae/`, so the repository row covers
    // them; the Gemma-2-2B-IT caption encoder is keyed on Google's repository.
    ProviderComponents {
        provider_id: "sana_1600m",
        components: &[
            "sana_1600m_1024px_diffusers",
            "gemma_2_2b_it",
            "nvidia_pid_students",
        ],
    },
    ProviderComponents {
        provider_id: "sana_sprint_1600m",
        components: &[
            "sana_sprint_1_6b_1024px_diffusers",
            "gemma_2_2b_it",
            "nvidia_pid_students",
        ],
    },
    // --- scail2 ------------------------------------------------------------------------------
    // Five of this id's artifacts are holes: the three bundled auxiliaries (UMT5, open-CLIP ViT-H,
    // Wan2.1 VAE) and both optional LoRAs. The bundled UMT5 is deliberately **not** resolved to
    // `umt5_xxl` — the evidence pass recorded `google/umt5-xxl` as an unestablished candidate, and
    // pointing at it would invent the fact the table exists to record.
    ProviderComponents {
        provider_id: "scail2_14b",
        components: &["scail_2"],
    },
    // --- sd3 ---------------------------------------------------------------------------------
    // Three text encoders, all read in-snapshot and all keyed on their own upstreams. The `-turbo`
    // and `-medium` siblings have no DiT row (NO DECLARED STRING), so those two ids carry only their
    // shared encoders. sd3 has no PiD edge on MLX even though `nvidia/PiD` ships an sd3 student.
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
    // --- sdxl --------------------------------------------------------------------------------
    // One id, two interchangeable bases. **No `sdxl_vae_fp16_fix`**: MLX reads the VAE inside the
    // SDXL snapshot at f32, while Candle requires the staged fp16-fix VAE — the genuine per-backend
    // component difference sc-16665 called out. The caller-supplied ControlNet has no fixed upstream
    // (which is why the catalog lists `sdxl` in `NO_BRANCH_POLICY`), so no row can name it.
    ProviderComponents {
        provider_id: "sdxl",
        components: &[
            "stable_diffusion_xl_base_1_0",
            "realvisxl_v5_0",
            "clip_vit_bigg_14_laion2b",
            "ip_adapter",
            "nvidia_pid_students",
        ],
    },
    // --- seedvr2 -----------------------------------------------------------------------------
    // Two DiT files and one shared VAE, all from one community re-host; `seedvr2` (bare) resolves to
    // the 3B checkpoint. No text encoder, no tokenizer, no separate upscaler network.
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
    // --- sensenova ---------------------------------------------------------------------------
    // A unified model: no VAE and no separate text encoder. `_fast` merges the distill LoRA before
    // quantization, so it is the same base plus one artifact.
    ProviderComponents {
        provider_id: "sensenova_u1_8b",
        components: &["sensenova_u1_8b_mot"],
    },
    ProviderComponents {
        provider_id: "sensenova_u1_8b_fast",
        components: &["sensenova_u1_8b_mot", "sensenova_u1_8b_mot_loras"],
    },
    // --- svd ---------------------------------------------------------------------------------
    // UNet, temporal-decoder VAE and ViT-H image encoder, all in one ungated snapshot.
    ProviderComponents {
        provider_id: "svd_xt",
        components: &["stable_video_diffusion_img2vid_xt"],
    },
    // --- wan ---------------------------------------------------------------------------------
    // Each Wan repository ships its own VAE and tokenizer, so the repository row covers them; only
    // UMT5-XXL is keyed separately, being one of the few auxiliaries whose upstream the code writes
    // down. The GGUF re-host is a Candle-only path and is deliberately absent here. Both VACE routes
    // share the base Wan2.2 14B trio (`mlx-gen-wan/src/convert.rs:489`, `:561`).
    ProviderComponents {
        provider_id: "wan2_2_ti2v_5b",
        components: &["wan2_2_ti2v_5b", "umt5_xxl"],
    },
    ProviderComponents {
        provider_id: "wan2_2_t2v_14b",
        components: &["wan2_2_t2v_a14b", "umt5_xxl"],
    },
    ProviderComponents {
        provider_id: "wan2_2_i2v_14b",
        components: &["wan2_2_i2v_a14b", "umt5_xxl"],
    },
    // Config-driven across the 1.3B and 14B VACE repositories, so both rows are listed.
    ProviderComponents {
        provider_id: "wan_vace",
        components: &[
            "wan2_1_vace_1_3b_diffusers",
            "wan2_1_vace_14b_diffusers",
            "wan2_2_t2v_a14b",
            "umt5_xxl",
        ],
    },
    ProviderComponents {
        provider_id: "wan2_2_vace_fun_14b",
        components: &["wan2_2_vace_fun_a14b", "wan2_2_t2v_a14b", "umt5_xxl"],
    },
    // --- z-image -----------------------------------------------------------------------------
    // Both control ids are MLX-only and each adds a genuinely separate control checkpoint. The VAE
    // ships in-snapshot (the code notes it is FLUX.1-dev's 16-channel design, which is why the PiD
    // `zimage` alias resolves to the `flux` student), and the Qwen3 text encoder is a pinned hole.
    ProviderComponents {
        provider_id: "z_image_turbo",
        components: &["z_image_turbo", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "z_image",
        components: &["z_image", "nvidia_pid_students"],
    },
    ProviderComponents {
        provider_id: "z_image_control",
        components: &[
            "z_image",
            "z_image_fun_controlnet_union_2_1",
            "nvidia_pid_students",
        ],
    },
    ProviderComponents {
        provider_id: "z_image_turbo_control",
        components: &[
            "z_image_turbo",
            "z_image_turbo_fun_controlnet_union_2_1",
            "nvidia_pid_students",
        ],
    },
];

/// The licence families every component row resolves against — the reviewed unit, shared with the
/// Candle media catalog and the audio catalog. Re-exposed here so a consumer reading the MLX surface
/// has the whole table in one place.
pub fn license_families() -> &'static [LicenseFamily] {
    gen_core::LICENSE_FAMILIES
}

/// Every media checkpoint whose licence has been read — the shared, backend-independent table
/// (sc-16665). Re-exposed for the same reason as [`license_families`]; the MLX catalog owns no rows
/// of its own, because a licence is a property of the checkpoint and both engines load the same ones.
pub fn component_licenses() -> &'static [ComponentLicense] {
    gen_core::MEDIA_COMPONENT_LICENSES
}

/// Which components each registered MLX id loads — see [`MLX_MEDIA_PROVIDER_COMPONENTS`].
pub fn provider_components() -> &'static [ProviderComponents] {
    MLX_MEDIA_PROVIDER_COMPONENTS
}

/// This catalog's model-licences manifest JSON at `schema_version` 3, in the shape the release
/// tooling merges (`scripts/release/build_release.py`, `MODEL_LICENSES_GLOB`).
///
/// **Nothing under `release/` carries these bytes yet.** Committing an MLX manifest is a separate
/// slice's call, and it would be a file only a macOS host could regenerate — this crate does not
/// build on Linux or Windows. The function exists so that slice has one place to read from, and so a
/// consumer can obtain the MLX surface in the same shape as the audio one.
pub fn component_licenses_manifest_json() -> String {
    gen_core::component_licenses_manifest_json(
        license_families(),
        component_licenses(),
        provider_components(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{
        license_table_conformance_errors, provider_terms, resolve_component, LicenseTerm,
    };
    use std::collections::BTreeSet;

    /// Every id this catalog registers, across every provider kind — the set the mapping is
    /// measured against. Trainer ids that are also generator ids collapse, which is the point:
    /// `ProviderComponents` is keyed by id and one id gets one row.
    fn registered_ids() -> BTreeSet<String> {
        let registry = crate::provider_registry().unwrap();
        let mut ids = BTreeSet::new();
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
        ids
    }

    fn mapped_ids() -> BTreeSet<String> {
        MLX_MEDIA_PROVIDER_COMPONENTS
            .iter()
            .map(|p| p.provider_id.to_string())
            .collect()
    }

    /// Nine registered ids load nothing the shared table covers, so they carry no row.
    ///
    /// **This is an annotation, never a filter.** Every id below registers, loads and renders
    /// exactly as it did before this module existed; what is absent is our metadata, which is a hole
    /// for CI to report and never a reason to withhold a provider. It is deliberately `#[cfg(test)]`
    /// rather than a public const, so no gate can quietly use it to suppress the ids it names.
    const IDS_WITHOUT_A_RESOLVABLE_COMPONENT: &[(&str, &str)] = &[
        // Every `microsoft/Mage-Flow*` repository 404s (six NOT FOUND holes) and the Qwen3-VL-4B
        // tower bit-identical across all six is UNDETERMINED, so the DiT, the Mage-VAE and the
        // encoder are all unrowed. Four of the six are pinned with `license = "MIT"` in
        // `release/real-weight-models.toml`, which is second-hand and was not written as a row.
        ("mage_flow", "NOT FOUND — microsoft/Mage-Flow 404s"),
        (
            "mage_flow_base",
            "NOT FOUND — microsoft/Mage-Flow-Base 404s",
        ),
        (
            "mage_flow_turbo",
            "NOT FOUND — microsoft/Mage-Flow-Turbo 404s",
        ),
        (
            "mage_flow_edit",
            "NOT FOUND — microsoft/Mage-Flow-Edit 404s",
        ),
        (
            "mage_flow_edit_base",
            "NOT FOUND — microsoft/Mage-Flow-Edit-Base 404s",
        ),
        (
            "mage_flow_edit_turbo",
            "NOT FOUND — microsoft/Mage-Flow-Edit-Turbo 404s",
        ),
        // The vision tower, projector, decoder and tokenizer are one snapshot, and its only
        // declaration in this repository is `release/real-weight-models.toml`'s own record.
        (
            "fancyfeast/llama-joycaption-beta-one-hf-llava",
            "SECOND-HAND ONLY — llama_joycaption_beta_one_hf_llava",
        ),
        // One `openai/clip-vit-large-patch14` snapshot, two heads, no card declaration.
        (
            "clip_vit_l14",
            "NOT FOUND — clip_vit_large_patch14 card declares nothing",
        ),
        (
            "clip_vit_l14_text",
            "NOT FOUND — clip_vit_large_patch14 card declares nothing",
        ),
    ];

    /// The mapped ids whose component list is **short**, and which key is missing from each.
    ///
    /// A live gate, not documentation: every key named here is asserted to be *still absent* from
    /// the shared table, so the day sc-16665 writes one of these rows this test fails and forces the
    /// mapping to be extended. Keys are the ones `known_holes_stay_absent` pins in
    /// `gen_core::license::components`.
    const INCOMPLETE_MAPPED_IDS: &[(&str, &[&str])] = &[
        (
            "anima_base",
            &["anima_qwen3_0_6b_text_encoder", "anima_qwen_image_vae"],
        ),
        (
            "anima_aesthetic",
            &["anima_qwen3_0_6b_text_encoder", "anima_qwen_image_vae"],
        ),
        (
            "anima_turbo",
            &["anima_qwen3_0_6b_text_encoder", "anima_qwen_image_vae"],
        ),
        ("boogu_image", &["boogu_qwen3_vl_8b", "boogu_flux1_vae"]),
        (
            "boogu_image_turbo",
            &["boogu_qwen3_vl_8b", "boogu_flux1_vae"],
        ),
        (
            "boogu_image_edit",
            &["boogu_qwen3_vl_8b", "boogu_flux1_vae"],
        ),
        ("flux1_schnell", &["clip_vit_large_patch14"]),
        ("flux1_dev", &["clip_vit_large_patch14"]),
        ("flux1_dev_control", &["clip_vit_large_patch14"]),
        ("flux2_klein_9b", &["flux2_klein_qwen3_text_encoder"]),
        ("flux2_klein_9b_edit", &["flux2_klein_qwen3_text_encoder"]),
        (
            "flux2_klein_9b_kv_edit",
            &["flux2_klein_qwen3_text_encoder"],
        ),
        (
            "flux2_dev",
            &[
                "flux2_dev_mistral3_tower",
                "flux2_dev_pixtral_vision_tower",
                "flux2_dev_multimodal_projector",
            ],
        ),
        (
            "flux2_dev_edit",
            &[
                "flux2_dev_mistral3_tower",
                "flux2_dev_pixtral_vision_tower",
                "flux2_dev_multimodal_projector",
            ],
        ),
        (
            "flux2_dev_control",
            &[
                "flux2_dev_mistral3_tower",
                "flux2_dev_pixtral_vision_tower",
                "flux2_dev_multimodal_projector",
                "flux2_dev_fun_controlnet_union",
            ],
        ),
        ("ideogram_4", &["ideogram_qwen3_vl_8b"]),
        ("ideogram_4_turbo", &["ideogram_qwen3_vl_8b"]),
        (
            "kolors",
            &[
                "kolors_diffusers",
                "sceneworks_kolors_chatglm3_tokenizer",
                "kolors_controlnet_pose",
                "clip_vit_large_patch14_336",
            ],
        ),
        ("krea_2_turbo", &["krea_qwen3_vl_4b"]),
        ("krea_2_raw", &["krea_qwen3_vl_4b"]),
        ("krea_2_edit", &["krea_qwen3_vl_4b"]),
        ("krea_2_turbo_edit", &["krea_qwen3_vl_4b"]),
        (
            "krea_2_turbo_control",
            &["krea_qwen3_vl_4b", "krea2_pose_controlnet_beta"],
        ),
        ("lens_turbo", &["microsoft_lens_turbo", "lens_transformer"]),
        ("lens", &["microsoft_lens", "lens_transformer"]),
        ("ltx_2_3", &["amoral_gemma_3_12b_v2_mlx_4bit"]),
        (
            "pulid_flux",
            &[
                "antelopev2_arcface_glintr100",
                "antelopev2_scrfd_10g",
                "clip_vit_large_patch14",
            ],
        ),
        ("qwen_image", &["qwen_image_qwen2_5_vl"]),
        ("qwen_image_control", &["qwen_image_qwen2_5_vl"]),
        ("qwen_image_edit", &["qwen_image_qwen2_5_vl"]),
        (
            "scail2_14b",
            &[
                "scail2_umt5",
                "scail2_open_clip_vit_h",
                "scail2_wan2_1_vae",
                "lightx2v_wan_step_distill_diff_patch",
                "sat_scail2_dpo_lora",
            ],
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
        ("z_image_turbo", &["z_image_qwen3_text_encoder"]),
        ("z_image", &["z_image_qwen3_text_encoder"]),
        ("z_image_control", &["z_image_qwen3_text_encoder"]),
        ("z_image_turbo_control", &["z_image_qwen3_text_encoder"]),
    ];

    /// The whole licence table is well-formed with the MLX mapping in it: unique ids, every
    /// `family` and every component key resolves, no id lists a component twice, no id maps to
    /// nothing.
    #[test]
    fn mlx_licence_table_is_conformant() {
        assert_eq!(
            license_table_conformance_errors(
                license_families(),
                component_licenses(),
                provider_components(),
            ),
            Vec::<String>::new()
        );
    }

    /// The decoder registry's `license_component` is not ornamental metadata: every compatible
    /// provider row must carry it, and the emitted manifest must therefore expose the complete
    /// base-plus-Wan component union. The gen-core mutation test proves dropping the donor is red.
    #[test]
    fn alternate_decoder_composites_are_in_the_derived_manifest() {
        assert_eq!(
            gen_core::decoder_license_conformance_errors(
                "mlx",
                gen_core::DECODER_OPTIONS,
                component_licenses(),
                provider_components(),
            ),
            Vec::<String>::new()
        );

        let option = gen_core::DECODER_OPTIONS
            .iter()
            .find(|option| option.id == gen_core::WAN_2_1_VAE_DECODER_ID)
            .expect("Wan decoder is registered");
        let value: serde_json::Value =
            serde_json::from_str(&component_licenses_manifest_json()).unwrap();
        for provider_id in option.eligible_provider_ids {
            let provider = value["providers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|provider| provider["provider_id"] == *provider_id)
                .unwrap_or_else(|| panic!("{provider_id} has a manifest row"));
            assert!(
                provider["components"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|component| component == option.license_component),
                "{provider_id} must expose the selected decoder's component union"
            );
            assert!(provider["terms"].is_array(), "terms remain derived");
        }
    }

    /// Every registered id either carries a mapping row with at least one resolvable component, or
    /// is one of the nine pinned as having nothing in the shared table — and nothing is mapped that
    /// is not registered, so the table cannot rot into naming a provider this catalog dropped.
    #[test]
    fn every_registered_id_maps_or_is_a_pinned_hole() {
        let registered = registered_ids();
        let mapped = mapped_ids();
        let pinned: BTreeSet<String> = IDS_WITHOUT_A_RESOLVABLE_COMPONENT
            .iter()
            .map(|(id, _)| id.to_string())
            .collect();

        assert_eq!(registered.len(), 69, "registered ids: {registered:?}");
        assert_eq!(mapped.len(), 60);
        assert_eq!(pinned.len(), IDS_WITHOUT_A_RESOLVABLE_COMPONENT.len());
        assert_eq!(pinned.len(), 9);

        let unaccounted: Vec<&String> = registered
            .iter()
            .filter(|id| !mapped.contains(*id) && !pinned.contains(*id))
            .collect();
        assert!(
            unaccounted.is_empty(),
            "these registered ids have neither a component mapping nor a pinned reason: \
             {unaccounted:?} — map them, or record why the shared table covers nothing they load"
        );

        let unregistered: Vec<&String> = mapped.difference(&registered).collect();
        assert!(
            unregistered.is_empty(),
            "these ids are mapped but not registered: {unregistered:?}"
        );
        let stale: Vec<&String> = pinned.difference(&registered).collect();
        assert!(
            stale.is_empty(),
            "these pinned ids are no longer registered: {stale:?}"
        );
        assert!(
            pinned.is_disjoint(&mapped),
            "an id cannot be both mapped and pinned as unmapped"
        );

        for provider in MLX_MEDIA_PROVIDER_COMPONENTS {
            assert!(
                !provider.components.is_empty(),
                "{} maps to no components",
                provider.provider_id
            );
            for key in provider.components {
                assert!(
                    resolve_component(component_licenses(), key).is_some(),
                    "{} references unknown component {key:?}",
                    provider.provider_id
                );
            }
        }
    }

    /// Which mapped ids are short, and why — and the keys they are short of are **still holes**.
    ///
    /// The classification is total: every mapped id is either listed here or is complete. A new id
    /// therefore cannot be added without deciding which it is, and a hole cannot be filled upstream
    /// without this failing and forcing the list to be extended.
    #[test]
    fn mapped_ids_are_complete_or_pinned_incomplete() {
        let mapped = mapped_ids();
        let incomplete: BTreeSet<String> = INCOMPLETE_MAPPED_IDS
            .iter()
            .map(|(id, _)| id.to_string())
            .collect();

        assert_eq!(incomplete.len(), INCOMPLETE_MAPPED_IDS.len());
        assert_eq!(incomplete.len(), 39);
        let unknown: Vec<&String> = incomplete.difference(&mapped).collect();
        assert!(
            unknown.is_empty(),
            "these ids are pinned incomplete but are not mapped: {unknown:?}"
        );
        assert_eq!(
            mapped.difference(&incomplete).count(),
            21,
            "the complete ids are the mapped ids minus the pinned-incomplete ones"
        );

        for (id, missing) in INCOMPLETE_MAPPED_IDS {
            assert!(!missing.is_empty(), "{id} is pinned incomplete with no key");
            for key in *missing {
                assert!(
                    resolve_component(component_licenses(), key).is_none(),
                    "component {key:?} now has a row, so {id:?} is no longer short of it — add it \
                     to the mapping and drop it from INCOMPLETE_MAPPED_IDS"
                );
            }
        }
    }

    /// One row per id, and the fifteen trainer registrations reuse their generator's row rather than
    /// duplicating it — `provider_id` is unique across the table, so they have to.
    #[test]
    fn trainer_ids_reuse_their_generator_row() {
        let registry = crate::provider_registry().unwrap();
        let generators: BTreeSet<String> = registry
            .generators()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        let trainers: Vec<String> = registry
            .trainers()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();

        assert_eq!(trainers.len(), 15);
        for id in &trainers {
            assert!(
                generators.contains(id),
                "trainer {id} is not also a generator id, so it needs its own mapping row"
            );
        }

        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for provider in MLX_MEDIA_PROVIDER_COMPONENTS {
            assert!(
                seen.insert(provider.provider_id),
                "duplicate mapping row for {}",
                provider.provider_id
            );
        }
    }

    /// `pulid_flux` reaches through the unregistered `mlx-gen-face` crate to the artifacts that
    /// crate loads — and two of the three are pinned holes, so epic 16660's R6 term reaches no
    /// union today. Both halves are pinned: a component-keyed table is what makes the dependency
    /// expressible at all, and the gap is a data hole rather than a modelling one.
    #[test]
    fn pulid_flux_lists_the_face_stack_it_loads() {
        let pulid = MLX_MEDIA_PROVIDER_COMPONENTS
            .iter()
            .find(|p| p.provider_id == "pulid_flux")
            .expect("pulid_flux is MLX-registered");

        for key in [
            "flux1_dev",
            "t5_v1_1_xxl",
            "pulid",
            "eva_clip",
            "facexlib_parsing_bisenet",
        ] {
            assert!(
                pulid.components.contains(&key),
                "pulid_flux must list {key}"
            );
        }
        // MLX PuLID has no PiD edge; the Candle twin does. A per-backend fact, not a shared one.
        assert!(!pulid.components.contains(&"nvidia_pid_students"));

        // The insightface family is landed and no component resolves to it, so no union carries it.
        for key in ["antelopev2_arcface_glintr100", "antelopev2_scrfd_10g"] {
            assert!(
                resolve_component(component_licenses(), key).is_none(),
                "{key} gained a row — add it to pulid_flux and drop it from INCOMPLETE_MAPPED_IDS"
            );
        }
        assert!(
            component_licenses()
                .iter()
                .all(|row| row.family != "insightface-research-only"),
            "a component now resolves to insightface-research-only; pulid_flux's union must pick it up"
        );
    }

    /// The MLX-only shapes sc-16665 enumerated, pinned so a later edit cannot quietly converge the
    /// two backends where the code genuinely differs.
    #[test]
    fn per_backend_divergences_are_expressed() {
        let components = |id: &str| {
            MLX_MEDIA_PROVIDER_COMPONENTS
                .iter()
                .find(|p| p.provider_id == id)
                .unwrap_or_else(|| panic!("{id} is mapped"))
                .components
        };

        // SDXL: MLX reads the in-snapshot VAE; the fp16-fix VAE is a required Candle component.
        assert!(!components("sdxl").contains(&"sdxl_vae_fp16_fix"));
        assert!(components("sdxl").contains(&"stable_diffusion_xl_base_1_0"));
        // FLUX.2's separately distilled kv checkpoint is MLX-only.
        assert!(components("flux2_klein_9b_kv_edit").contains(&"flux2_klein_9b_kv"));
        assert!(!components("flux2_klein_9b").contains(&"flux2_klein_9b_kv"));
        // Pin the VACE-Fun expert checkpoint map; both Z-Image control checkpoints are MLX-only.
        assert!(components("wan2_2_vace_fun_14b").contains(&"wan2_2_vace_fun_a14b"));
        assert!(components("z_image_control").contains(&"z_image_fun_controlnet_union_2_1"));
        assert!(
            components("z_image_turbo_control").contains(&"z_image_turbo_fun_controlnet_union_2_1")
        );
        // The GGUF re-host is the Candle 5B path only.
        assert!(!components("wan2_2_ti2v_5b").contains(&"wan2_2_ti2v_5b_gguf"));
        // LTX's Gemma naming split resolves to one licensor row on both backends.
        assert!(components("ltx_2_3").contains(&"gemma_3_12b_it"));
    }

    /// The derived union is what a consumer joins over, and it is derived — a term reaches a
    /// provider only because one of its components brought it. Two directions, both load-bearing.
    #[test]
    fn terms_are_derived_from_components() {
        let terms = |id: &str| {
            let provider = MLX_MEDIA_PROVIDER_COMPONENTS
                .iter()
                .find(|p| p.provider_id == id)
                .unwrap_or_else(|| panic!("{id} is mapped"));
            provider_terms(provider, component_licenses(), license_families())
        };

        // FLUX.1-dev names a weights restriction; Mochi's Apache-2.0 pair names none.
        assert!(terms("flux1_dev").contains(&LicenseTerm::NonCommercialWeights));
        assert!(!terms("mochi_1").contains(&LicenseTerm::NonCommercialWeights));
        // A gated component raises GatedAccess into the union even though no family declares it.
        assert!(terms("flux1_dev").contains(&LicenseTerm::GatedAccess));
        assert!(!terms("mochi_1").contains(&LicenseTerm::GatedAccess));

        // The PiD overlay is the **only** reason `sana_1600m` and `z_image` name a non-commercial
        // text at all: every other component on either row carries no non-commercial term
        // (`sana_1600m`'s Gemma-2-2B-IT caption encoder resolves to `gemma-terms`, not
        // `apache-2-0` — but that family states no `NonCommercialWeights`), `mlx-gen-pid` states
        // NVIDIA's terms in one crate, and not one of the thirteen consumers repeats it. Deriving
        // the union from the components is what surfaces that — which is epic 16660's R6, and the
        // reason the strictest term in a row is routinely not its headline model. `seedvr2` has no
        // PiD edge and stays clean, so the assertion is about the overlay and not about the table.
        assert!(terms("sana_1600m").contains(&LicenseTerm::NonCommercialWeights));
        assert!(terms("z_image").contains(&LicenseTerm::NonCommercialWeights));
        assert!(!terms("seedvr2").contains(&LicenseTerm::NonCommercialWeights));
    }

    /// Where a licence is silent on outputs, this surface stays silent — at the provider layer a
    /// consumer actually joins over, not only at the family and component layers sc-16662 and
    /// sc-16665 pinned it. Several ids here reach families that name a non-commercial restriction on
    /// the **weights** and say nothing about renders; inferring the stronger term from that silence
    /// is the reading sc-16662's U3 declined to make.
    #[test]
    fn no_registered_mlx_id_reaches_an_outputs_restriction() {
        for provider in MLX_MEDIA_PROVIDER_COMPONENTS {
            let terms = provider_terms(provider, component_licenses(), license_families());
            assert!(
                !terms.contains(&LicenseTerm::NonCommercialOutputs),
                "{} reaches NonCommercialOutputs; no landed family states one",
                provider.provider_id
            );
        }
    }

    /// The manifest shape is deterministic and carries no legal conclusion — the property the whole
    /// v3 schema exists for.
    #[test]
    fn manifest_round_trips_the_mlx_surface() {
        let json = component_licenses_manifest_json();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(value["schema_version"], 3);
        assert_eq!(value["kind"], "model-weight-licenses");
        assert_eq!(value["providers"].as_array().unwrap().len(), 60);
        assert!(!json.contains("commercial_use"));

        let sdxl = value["providers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["provider_id"] == "sdxl")
            .expect("sdxl is in the manifest");
        assert!(!sdxl["components"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c.as_str() == Some("sdxl_vae_fp16_fix")));
    }
}
