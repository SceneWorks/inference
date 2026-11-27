//! sc-10995 (capstone): the **full Bernini** Generator (provider id `"bernini"`) — the registered
//! pipeline that strings the whole planner → renderer stack together, the candle sibling of
//! `mlx-gen-bernini/src/bernini.rs` (sc-5145). Mirrors `BerniniPipeline.__call__`:
//!
//! ```text
//!   preprocess (ViT tower on the sources + gen-target grid; VAE on the sources)
//!     → 3 planner streams (cond / uncond / imgcond)            [build_stream]
//!     → MAR semantic-planning loop                             [crate::mar::sample_vit_embed]
//!     → 4 renderer prompt streams + UMT5 concat_with_zero_init [crate::assembly]
//!     → ViT-conditioned dual-expert APG denoise                [denoise_bernini_wvitcfg]
//!     → z16 VAE decode → image (1 frame) / video
//! ```
//!
//! The planner (`BerniniPlanner`) is Qwen2.5-VL-7B (penultimate extractor) + [`MlpConnector`] +
//! [`DiffLossFm`] clip-diff head + the MAR mask token; the renderer is the existing dual-expert
//! [`WanTransformer`] MoE + UMT5 + z16 VAE. This assembles the `*_wapg` / `v2v_apg` ViT-conditioned
//! guidance modes ([`crate::forward::vit_one_step`]) that sc-11004 flagged as pending — the planner's
//! ViT-guidance feeding [`denoise_bernini_wvitcfg`].
//!
//! **Weights.** The renderer tier (`transformer/` `transformer_2/` `text_encoder/` `vae/` `tokenizer/`)
//! is produced by [`crate::convert::build_bernini_candle_tier`]; the planner components (`mllm/`
//! `connector/` `vit_decoder/` `mask_tokens.safetensors` + `qwen2_5_vl_config.json`) are the additional
//! candle turnkey layout this loader reads (see `BerniniPlanner::load`). The full 168 GB
//! `ByteDance/Bernini-Diffusers` weights are NOT downloaded here; real-weight semantic GPU-val is sc-11003.
//!
//! **Validation without real weights.** The registered generator loads lazily, so registration, the
//! descriptor, and `validate` are all exercised weightlessly. The genuinely new capstone compute (the
//! ViT-conditioned renderer denoise, [`vit_one_step`] / [`denoise_bernini_wvitcfg`]) is CPU-golden and
//! shape-tested on tiny synthetic experts. The planner seams (backbone, vision, connector, clip_diff,
//! MAR, assembly, template, process) each carry their own merged CPU parity goldens; this module wires
//! them.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use image::RgbImage;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{
    self, CancelFlag, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};
use gen_core_tokenizer::TextTokenizer;

use candle_gen_wan::config::{TextEncoderConfig, TransformerConfig};
use candle_gen_wan::pipeline::{create_noise, frames_to_images};
use candle_gen_wan::rope::assign_source_ids;
use candle_gen_wan::scheduler::Sampler;
use candle_gen_wan::text_encoder::Umt5Encoder;
use candle_gen_wan::transformer::WanTransformer;

use crate::assembly::{concat_with_zero_init, format_mllm_inputs_embeds};
use crate::clip_diff::DiffLossFm;
use crate::components::RendererComponents;
use crate::config::BerniniKnobs;
use crate::connector::MlpConnector;
use crate::convert::{
    PLANNER_CONNECTOR_DIR, PLANNER_MASK_TOKENS_FILE, PLANNER_MASK_TOKENS_KEY,
    PLANNER_MLLM_BACKBONE_PP, PLANNER_MLLM_DIR, PLANNER_MLLM_VISION_PP, PLANNER_QWEN_CONFIG_FILE,
    PLANNER_SIDECAR_FILE, PLANNER_TOKENIZER_FILE, PLANNER_VIT_DECODER_DIR, PLANNER_VIT_DECODER_PP,
};
use crate::forward::{
    num_vit_momentum_buffers, vit_one_step, PackedForward, VitGuidanceParams, VitMode, VitStreams,
};
use crate::guidance::MomentumBuffer;
use crate::latent_dims;
use crate::mar::{
    mar_schedule, post_process_input_embeds, sample_vit_embed_with_progress, StreamState, VitCfg,
};
use crate::pipeline::{
    run_bernini_candle_stage2_driver, BerniniExpert, BerniniStage2Forward, BerniniStage2Omegas,
};
use crate::preprocess::{encode_image, encode_videoclip};
use crate::process::{
    build_attention_mask_4d, generate_unified_inputs, mrope_position_ids, MRopeConfig,
};
use crate::qwen2_5_vl::{Qwen25VlText, QwenVlTextConfig};
use crate::template::BerniniTemplate;
use crate::vision::{VisionConfig, VisionTower};
use crate::vit_preprocess::{
    normalized_frame, pack_patches, preprocess_image, smart_resize, smart_video_nframes, FACTOR,
    IMAGE_MEAN, IMAGE_STD, MERGE_SIZE, PATCH_SIZE, TEMPORAL_PATCH_SIZE,
};
use gen_core_family::bernini_plan::{
    build_bernini_render_plan, BerniniBackendKind, BerniniEntrypoint, BerniniFullVitMode,
    BerniniOutputContract, BerniniRenderPlan,
};

/// SceneWorks engine id — matches `mlx-gen-bernini`'s full pipeline so a consumer resolves the same
/// engine across backends.
pub const MODEL_ID: &str = "bernini";

/// The A14B DiT emits 16-channel latents (z16 VAE).
const Z_DIM: usize = 16;
/// The planner + renderer contexts run bf16 on the DiT side; UMT5 + z16 VAE run f32.
const PLANNER_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;

/// Full-pipeline CLI defaults (`bernini/cli.py` for the `BerniniPipeline` path). A request's `guidance`
/// overrides `omega_txt`; the rest are fixed until the worker surfaces them.
struct FullDefaults;
impl FullDefaults {
    const OMEGA_TGT: f32 = 0.5;
    const PLANNING_STEP: usize = 25;
    const VIT_TXT_CFG: f32 = 1.2;
    const VIT_IMG_CFG: f32 = 1.0;
    const VIT_DENOISING_STEP: usize = 5;
    const NORM_THRESHOLD: f32 = 50.0;
    /// APG momentum for the `v2v_apg` ViT stream (reference default 0.0 — inert). Persisted across the
    /// denoise loop so a future nonzero default actually carries (F-161).
    const MOMENTUM: f32 = 0.0;
    /// Source-media ViT pixel budget (`preprocess_inputs` `vit_min/max_pixels`).
    const VIT_MIN_PIXELS: i64 = 3136;
    const VIT_MAX_PIXELS: i64 = 50176;
    const FPS: u32 = 16;
}

/// Planner knobs read from the `bernini_planner.json` sidecar (else the package-config defaults).
struct PlannerKnobs {
    max_sequence_length: usize,
    num_mask_token: i32,
    clip_diff_depth: usize,
    clip_diff_in_channels: usize,
    clip_diff_shift: f32,
}

impl PlannerKnobs {
    /// A **missing** `bernini_planner.json` sidecar is a legitimate snapshot shape (→ package-config
    /// defaults); a **present but malformed** sidecar surfaces an error naming the file rather than
    /// silently downgrading the planner knobs (F-145).
    fn from_dir(root: &Path) -> CResult<Self> {
        let v = crate::config::read_optional_json(
            &root.join(PLANNER_SIDECAR_FILE),
            "bernini planner sidecar",
        )?
        .unwrap_or(serde_json::Value::Null);
        let i = |k: &str, d: i64| v.get(k).and_then(serde_json::Value::as_i64).unwrap_or(d);
        let cd = v
            .get("clip_diff_cfg")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let cdi = |k: &str, d: i64| cd.get(k).and_then(serde_json::Value::as_i64).unwrap_or(d);
        let cdf = |k: &str, d: f64| cd.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
        Ok(Self {
            max_sequence_length: i("max_sequence_length", 512).max(1) as usize,
            num_mask_token: i("num_mask_token", 4096) as i32,
            // `vit_decoder` (`SimpleMLPAdaLN`) depth is fixed at 16 in the released checkpoint; the
            // sidecar carries z_channels / shift verbatim.
            clip_diff_depth: 16,
            clip_diff_in_channels: cdi("z_channels", 3584) as usize,
            clip_diff_shift: cdf("shift", 2.0) as f32,
        })
    }
}

/// Read the planner's MRoPE / token-id config from `qwen2_5_vl_config.json`. A **missing** config is a
/// legitimate snapshot shape (→ defaults); a **present but malformed** config surfaces an error naming
/// the file rather than silently swapping the MRoPE sections / token ids (F-145).
fn read_mrope_config(path: &Path) -> CResult<MRopeConfig> {
    let d = MRopeConfig::default();
    let v = crate::config::read_optional_json(path, "bernini qwen2_5_vl_config")?
        .unwrap_or(serde_json::Value::Null);
    let vc = v.get("vision_config").unwrap_or(&v);
    let i = |o: &serde_json::Value, k: &str, dv: i64| {
        o.get(k).and_then(serde_json::Value::as_i64).unwrap_or(dv)
    };
    let f = |o: &serde_json::Value, k: &str, dv: f64| {
        o.get(k).and_then(serde_json::Value::as_f64).unwrap_or(dv)
    };
    Ok(MRopeConfig {
        spatial_merge_size: i(vc, "spatial_merge_size", d.spatial_merge_size),
        tokens_per_second: f(vc, "tokens_per_second", d.tokens_per_second),
        image_token_id: i(&v, "image_token_id", d.image_token_id),
        video_token_id: i(&v, "video_token_id", d.video_token_id),
        vision_start_token_id: i(&v, "vision_start_token_id", d.vision_start_token_id),
    })
}

/// The loaded Bernini semantic planner: Qwen2.5-VL backbone + vision tower + connector + clip-diff head
/// + MAR mask token + the host-side templating / MRoPE config.
struct BerniniPlanner {
    backbone: Qwen25VlText,
    vision: VisionTower,
    connector: MlpConnector,
    clip_diff: DiffLossFm,
    /// A single MAR mask token `[1, 1, H]` (`self.mask_tokens[:, :1]`, broadcast over the target).
    mask_token: Tensor,
    mrope: MRopeConfig,
    template: BerniniTemplate,
    knobs: PlannerKnobs,
}

impl BerniniPlanner {
    /// Load the planner components from the candle turnkey layout under `root`:
    ///   - `mllm/` — the Qwen2.5-VL backbone (`model.*`) + vision tower (`visual.*`) + `tokenizer.json`.
    ///   - `connector/` — the `proj_gen.*` / `pred_vit.*` MLP connector.
    ///   - `vit_decoder/` — the clip-diff flow head (`net.*`).
    ///   - `mask_tokens.safetensors` — the MAR mask tokens (`mask_tokens`).
    ///   - `qwen2_5_vl_config.json` — the backbone / vision / MRoPE config.
    ///
    /// Loaded dense bf16 (planner quantization is a follow-up; the renderer experts carry the dominant
    /// footprint and quantize separately via the packed-detect tier).
    fn load(root: &Path, device: &Device) -> CResult<Self> {
        let cfg_path = root.join(PLANNER_QWEN_CONFIG_FILE);
        let qcfg = QwenVlTextConfig::from_config_json(&cfg_path)?;
        let vcfg = VisionConfig::from_config_json(&cfg_path)?;

        let mllm_vb =
            candle_gen::component_vb(root, PLANNER_MLLM_DIR, PLANNER_DTYPE, device, MODEL_ID)?;
        let backbone = Qwen25VlText::new(qcfg, mllm_vb.pp(PLANNER_MLLM_BACKBONE_PP))?;
        let vision = VisionTower::new(vcfg, mllm_vb.pp(PLANNER_MLLM_VISION_PP))?;

        let conn_vb =
            candle_gen::component_vb(root, PLANNER_CONNECTOR_DIR, PLANNER_DTYPE, device, MODEL_ID)?;
        let connector = MlpConnector::new(conn_vb)?;

        let knobs = PlannerKnobs::from_dir(root)?;
        let vd_vb = candle_gen::component_vb(
            root,
            PLANNER_VIT_DECODER_DIR,
            PLANNER_DTYPE,
            device,
            MODEL_ID,
        )?;
        let clip_diff = DiffLossFm::new(
            vd_vb.pp(PLANNER_VIT_DECODER_PP),
            knobs.clip_diff_depth,
            knobs.clip_diff_in_channels,
            knobs.clip_diff_shift,
        )?;

        // `self.mask_tokens[:, :1]` — a single mask token, broadcast over the n_query target slots.
        let mask_map = candle_gen::candle_core::safetensors::load(
            root.join(PLANNER_MASK_TOKENS_FILE),
            device,
        )?;
        let mask_token = mask_map
            .get(PLANNER_MASK_TOKENS_KEY)
            .ok_or_else(|| {
                CandleError::Msg(format!(
                    "bernini: {PLANNER_MASK_TOKENS_FILE} missing `{PLANNER_MASK_TOKENS_KEY}`"
                ))
            })?
            .narrow(1, 0, 1)?
            .to_dtype(PLANNER_DTYPE)?;

        let template = BerniniTemplate::from_tokenizer_file(
            root.join(PLANNER_MLLM_DIR).join(PLANNER_TOKENIZER_FILE),
        )?;
        Ok(Self {
            backbone,
            vision,
            connector,
            clip_diff,
            mask_token,
            mrope: read_mrope_config(&cfg_path)?,
            template,
            knobs,
        })
    }
}

/// One preprocessed source visual: its ViT features (planner conditioning) + grid + VAE latent (renderer
/// conditioning).
struct SourceVisual {
    /// `[merged, H]` planner ViT features.
    vit_feat: Tensor,
    /// `[t, h, w]` ViT grid (drives the token count + MRoPE).
    vit_grid: [i32; 3],
    /// `[1, 16, T, H8, W8]` normalized VAE latent (the renderer source-conditioning latent).
    vae_latent: Tensor,
    /// Original source `(height, width)` (for the conversation's `image` message).
    hw: (i64, i64),
}

/// Convert a public RGB8 [`Image`] to an `image::RgbImage`.
fn to_rgb(img: &Image) -> CResult<RgbImage> {
    RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .ok_or_else(|| CandleError::Msg("bernini: malformed RGB8 conditioning image".into()))
}

/// ViT-encode one image → `[merged, H]` + grid.
fn vit_encode_image(
    planner: &BerniniPlanner,
    rgb: &RgbImage,
    dev: &Device,
) -> CResult<(Tensor, [i32; 3])> {
    let (pixels, grid) = preprocess_image(
        rgb,
        FullDefaults::VIT_MIN_PIXELS,
        FullDefaults::VIT_MAX_PIXELS,
        IMAGE_MEAN,
        IMAGE_STD,
        dev,
    )?;
    let g = [grid[0] as usize, grid[1] as usize, grid[2] as usize];
    let feat = planner.vision.forward(&pixels, &[g])?;
    Ok((feat, grid))
}

/// ViT-encode (already ViT-sampled) video frames → `[merged, H]` + grid. All frames are `smart_resize`d
/// to a common size, normalized, stacked `[F, 3, H, W]`, then `pack_patches` (temporal 2).
fn vit_encode_video(
    planner: &BerniniPlanner,
    frames: &[RgbImage],
    dev: &Device,
) -> CResult<(Tensor, [i32; 3])> {
    let (h0, w0) = (frames[0].height() as i64, frames[0].width() as i64);
    let (rh, rw) = smart_resize(
        h0,
        w0,
        FACTOR,
        FullDefaults::VIT_MIN_PIXELS,
        FullDefaults::VIT_MAX_PIXELS,
    );
    let mut chw_t = Vec::with_capacity(frames.len());
    for f in frames {
        let resized = image::imageops::resize(
            f,
            rw as u32,
            rh as u32,
            image::imageops::FilterType::CatmullRom,
        );
        chw_t.push(normalized_frame(
            resized.as_raw(),
            rh,
            rw,
            IMAGE_MEAN,
            IMAGE_STD,
            dev,
        )?);
    }
    let refs: Vec<&Tensor> = chw_t.iter().collect();
    let frames_t = Tensor::cat(&refs, 0)?; // [F, 3, H, W]
    let (pixels, grid) = pack_patches(&frames_t, PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)?;
    let g = [grid[0] as usize, grid[1] as usize, grid[2] as usize];
    let feat = planner.vision.forward(&pixels, &[g])?;
    Ok((feat, grid))
}

/// The gen-target ViT grid `[t, h, w]` (sizes `n_query`, the MAR token count). Image target (`frames ==
/// 1`) ⇒ `t = 1`; video target samples `vit_fps` (= fps/8) frames from the `num_frames` clip, `t =
/// vit_frames / temporal`. The spatial grid is `smart_resize` of the output H/W under the ViT budget.
fn gen_target_grid(height: i64, width: i64, frames: usize, fps: u32) -> [i32; 3] {
    let (rh, rw) = smart_resize(
        height,
        width,
        FACTOR,
        FullDefaults::VIT_MIN_PIXELS,
        FullDefaults::VIT_MAX_PIXELS,
    );
    let gh = (rh / PATCH_SIZE) as i32;
    let gw = (rw / PATCH_SIZE) as i32;
    let t = if frames <= 1 {
        1
    } else {
        let vit_fps = (fps / 8).max(1) as f64;
        let vit_frames = smart_video_nframes(
            frames as i64,
            fps as f64,
            vit_fps,
            Some(TEMPORAL_PATCH_SIZE),
            None,
            Some(frames as i64),
            false,
        )
        .len() as i64;
        (vit_frames / TEMPORAL_PATCH_SIZE).max(1) as i32
    };
    [t, gh, gw]
}

/// Merged-token count of a ViT grid (`t·h·w / merge²`).
fn grid_tokens(grid: [i32; 3]) -> i64 {
    let m2 = (MERGE_SIZE * MERGE_SIZE) as i32;
    (grid[0] * grid[1] * grid[2] / m2) as i64
}

/// Build one planner stream's [`StreamState`] (cond / uncond / imgcond). `images`/`videos` are the
/// **present** input source visuals (empty for uncond/imgcond); `prompt` is the stream's text (raw for
/// cond/imgcond, negative for uncond). `gen_grid` is the gen-target ViT grid; `gen_is_video` selects the
/// gen slot kind. The gen-target ViT features are zeros (masked by `post_process_input_embeds`).
#[allow(clippy::too_many_arguments)]
fn build_stream(
    planner: &BerniniPlanner,
    task: &str,
    prompt: &str,
    images: &[SourceVisual],
    videos: &[SourceVisual],
    gen_grid: [i32; 3],
    gen_is_video: bool,
    out_h: i64,
    out_w: i64,
    dev: &Device,
) -> CResult<(StreamState, i32)> {
    let image_hw: Vec<(i64, i64)> = images.iter().map(|s| s.hw).collect();
    let output_t = if gen_is_video { 2 } else { 1 };
    let conv = generate_unified_inputs(prompt, &image_hw, videos.len(), output_t, out_h, out_w);

    let mut image_grids: Vec<[i32; 3]> = images.iter().map(|s| s.vit_grid).collect();
    let mut video_grids: Vec<[i32; 3]> = videos.iter().map(|s| s.vit_grid).collect();
    if gen_is_video {
        video_grids.push(gen_grid);
    } else {
        image_grids.push(gen_grid);
    }
    let image_token_nums: Vec<i64> = image_grids.iter().map(|&g| grid_tokens(g)).collect();
    let video_token_nums: Vec<i64> = video_grids.iter().map(|&g| grid_tokens(g)).collect();

    let tout =
        planner
            .template
            .encode_messages(&conv, &image_token_nums, &video_token_nums, task)?;
    let l = tout.input_ids.len();

    // visual_embeds: conversation order = [video feats, image feats, gen-target zeros].
    let h_vit = planner.knobs.clip_diff_in_channels;
    let gen_tokens = grid_tokens(gen_grid) as usize;
    let mut feats: Vec<Tensor> = Vec::new();
    for v in videos {
        feats.push(v.vit_feat.clone());
    }
    for im in images {
        feats.push(im.vit_feat.clone());
    }
    feats.push(Tensor::zeros((gen_tokens, h_vit), PLANNER_DTYPE, dev)?);
    let feat_refs: Vec<&Tensor> = feats.iter().collect();
    let visual_embeds = Tensor::cat(&feat_refs, 0)?.to_dtype(PLANNER_DTYPE)?;

    let to_i64 = |g: &[[i32; 3]]| -> Vec<[i64; 3]> {
        g.iter()
            .map(|&[a, b, c]| [a as i64, b as i64, c as i64])
            .collect()
    };
    // `mrope_position_ids` / `build_attention_mask_4d` build on CPU (pure host computation). Move them
    // onto the backbone's device so the planner forward (`mrope_cos_sin`, `scores.broadcast_add(mask)`)
    // stays device-consistent on CUDA (sc-11148 / F-079).
    let bb_dev = planner.backbone.device();
    let pos = mrope_position_ids(
        &tout.input_ids,
        &to_i64(&image_grids),
        &to_i64(&video_grids),
        &planner.mrope,
    )?
    .to_device(bb_dev)?;
    let mask =
        build_attention_mask_4d(&tout.token_type, &tout.token_segment_ids)?.to_device(bb_dev)?;

    let vin: Vec<bool> = tout.token_type.iter().map(|&t| t == 2).collect();
    let vout: Vec<bool> = tout.token_type.iter().map(|&t| t == 3).collect();

    let embeds = format_mllm_inputs_embeds(
        &planner.backbone,
        &tout.input_ids,
        Some(&visual_embeds),
        &vin,
        &vout,
    )?;
    let embeds = post_process_input_embeds(&embeds, &vout, &planner.mask_token)?;
    // Keep the additive mask in the backbone's activation dtype (its attention adds it to bf16 scores).
    let mask = mask.to_dtype(embeds.dtype())?;

    let gen_idx: Vec<u32> = (0..l).filter(|&i| vout[i]).map(|i| i as u32).collect();
    let n_query = gen_idx.len() as i32;
    Ok((
        StreamState {
            input_embeds: embeds,
            position_ids: pos,
            mask,
            gen_idx,
        },
        n_query,
    ))
}

/// Resolve the full-Bernini [`VitMode`] from the request's `video_mode` (a guidance-mode name preferred)
/// plus the conditioning + output kind. Defaults: video source ⇒ `v2v_apg`; video+image or image-refs→
/// video ⇒ `rv2v_wapg`; otherwise (t2i/i2i/t2v) ⇒ `vae_txt_vit_wapg`.
#[cfg(test)]
fn resolve_vit_mode(
    video_mode: Option<&str>,
    has_video: bool,
    has_image: bool,
    out_video: bool,
) -> VitMode {
    if let Some(s) = video_mode {
        if let Some(m) = VitMode::from_name(s) {
            return m;
        }
        if let Some(m) = task_to_vit_mode(s) {
            return m;
        }
    }
    match (has_video, has_image, out_video) {
        (true, _, _) => {
            if has_image {
                VitMode::Rv2vWapg
            } else {
                VitMode::V2vApg
            }
        }
        (false, true, true) => VitMode::Rv2vWapg, // r2v: image refs → video
        _ => VitMode::VaeTxtVitWapg,              // t2i / i2i / t2v
    }
}

#[cfg(test)]
fn canonical_task(
    video_mode: Option<&str>,
    has_video: bool,
    has_image: bool,
    out_video: bool,
) -> &'static str {
    if let Some(s) = video_mode {
        match s {
            "t2i" => return "t2i",
            "t2v" => return "t2v",
            "i2i" => return "i2i",
            "v2v" => return "v2v",
            "r2v" => return "r2v",
            "rv2v" => return "rv2v",
            _ => {}
        }
    }
    match (has_video, has_image, out_video) {
        (true, true, _) => "rv2v", // video + reference → video editing with reference
        (true, false, _) => "v2v", // video → video editing
        (false, true, true) => "r2v", // image refs → video (subject-to-video)
        (false, true, false) => "i2i", // image → image editing
        (false, false, true) => "t2v", // text → video
        (false, false, false) => "t2i", // text → image
    }
}

#[cfg(test)]
fn task_to_vit_mode(task: &str) -> Option<VitMode> {
    Some(match task {
        "t2i" | "t2v" | "i2i" => VitMode::VaeTxtVitWapg,
        "v2v" | "mv2v" | "ads2v" => VitMode::V2vApg,
        "r2v" | "rv2v" => VitMode::Rv2vWapg,
        _ => return None,
    })
}

fn vit_mode_from_plan(mode: BerniniFullVitMode) -> VitMode {
    match mode {
        BerniniFullVitMode::VaeTxtVit => VitMode::VaeTxtVit,
        BerniniFullVitMode::VaeTxtVitWapg => VitMode::VaeTxtVitWapg,
        BerniniFullVitMode::Rv2vWapg => VitMode::Rv2vWapg,
        BerniniFullVitMode::R2vWapg => VitMode::R2vWapg,
        BerniniFullVitMode::V2vApg => VitMode::V2vApg,
    }
}

/// Primary typed route declaration for the full Bernini pipeline (`SW-6979`, `REG-1`).
pub(crate) fn route_full() -> gen_core::RouteRecord {
    #[cfg_attr(not(any(feature = "cuda", test)), allow(unused_mut))]
    let mut bindings = gen_core::RouteBindings::generator(descriptor, load);
    #[cfg(any(feature = "cuda", test))]
    {
        bindings = bindings
            .with_memory_registration(crate::memory_strategy::FULL_MEMORY_REGISTRATION)
            .with_memory_fixture_registration(crate::memory_strategy::FULL_MEMORY_FIXTURE)
            .with_memory_behavior_registration(crate::memory_strategy::FULL_MEMORY_BEHAVIOR);
    }

    gen_core::RouteRecord {
        route_id: MODEL_ID,
        family: "bernini",
        backend: "candle",
        modality: Modality::Video,
        mac_only: false,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![
                    gen_core::TextEncoderDeclaration::new("qwen2_5_vl_7b", None),
                    gen_core::TextEncoderDeclaration::new("umt5_xxl", Some(512)),
                ],
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                kinds: vec![
                    ConditioningKind::Reference,
                    ConditioningKind::MultiReference,
                    ConditioningKind::VideoClip,
                ],
                control_kinds: None,
                pass0_requirement: gen_core::Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: true,
                modules: vec!["bernini_mar_planner"],
            },
            sampling: gen_core::SamplingSurface {
                samplers: vec!["uni_pc", "unipc"],
                schedulers: vec![],
                guidance_methods: vec![],
                supports_guidance: true,
                supports_true_cfg: false,
                supports_cfg_interval: false,
                requires_sigma_shift: false,
                supports_h3_image_sampler: false,
            },
            denoiser: gen_core::ModelBinding::family_driver(
                gen_core::FamilyDriverSpec::new(crate::pipeline::stage2_driver_support(
                    "candle_bernini_full",
                ))
                .with_expert_switching(),
            ),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("wan_z16_vae"),
                output_latent_space: Some(&candle_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
                pixel_stride: Some(16),
                token_width: None,
                supports_diffusion_decoder: false,
            },
            upsamplers: gen_core::UpsamplerSurface::default(),
            adapters: gen_core::AdapterSurface::lora_and_lokr(
                gen_core::AdapterInstallPolicy::ForwardResidual,
            ),
            residency: gen_core::ResidencySurface {
                lifetime_graph: vec![
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::ConditioningEncode,
                        ["qwen2_5_vl_7b", "bernini_mar_planner"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::PromptEncode,
                        ["umt5_xxl"],
                        true,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        ["wan_a14b_high_expert", "wan_a14b_low_expert"],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        ["wan_z16_vae"],
                        false,
                    ),
                ],
                ..Default::default()
            },
            weight_plan: gen_core::WeightPlanDeclaration {
                source_kind: gen_core::WeightSourceKind::SnapshotDirectory,
                supported_quants: &[Quant::Q4, Quant::Q8],
                component_precision_floors: &[],
                required_components: &[],
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: false,
            },
            output: gen_core::OutputSurface {
                preview: gen_core::PreviewSupport::Unsupported,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            ..Default::default()
        },
        limits: gen_core::ProductLimits {
            min_size: 16,
            max_size: 1280,
            size_floor: gen_core::SizeFloor::RangeChecked,
            supported_steps: gen_core::StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::DeterministicOffset,
            max_count: 1,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults {
            steps: Some(crate::config::Defaults::STEPS as u32),
            guidance: Some(crate::config::Defaults::OMEGA_TXT),
            sampler: Some("uni_pc"),
            schedule: None,
            shift: Some(BerniniKnobs::default().shift),
            strength: None,
            dimensions: Some((832, 480)),
            ..Default::default()
        },
        bindings,
    }
}

/// Stable identity + advertised capabilities for the full Bernini pipeline, computed from
/// [`route_full`].
pub fn descriptor() -> ModelDescriptor {
    route_full().compute_descriptor()
}

/// The loaded full Bernini pipeline: the snapshot dir + the resolved renderer knobs, with the heavy
/// renderer components loaded lazily on the first `generate` and cached (F-097). The planner is
/// intentionally **not** cached — it is loaded, used, and dropped per request (peak VRAM).
pub struct Bernini {
    descriptor: ModelDescriptor,
    knobs: BerniniKnobs,
    root: PathBuf,
    device: Device,
    adapters: Vec<candle_gen::gen_core::AdapterSpec>,
    components: Mutex<Option<Arc<RendererComponents>>>,
    memory_strategy: Option<gen_core::MemoryProviderContract>,
    memory_tier: Option<gen_core::MemoryNumericTier>,
    /// Why this load has no sealed memory receipt. Present exactly when `memory_strategy` is
    /// `None`; it is the refusal the memory seams return instead of admitting an unsealed artifact.
    memory_refusal: String,
    loaded_spec: LoadSpec,
}

impl Bernini {
    /// Build and validate the pre-load [`BerniniRenderPlan`] before loading weights (`DEC-3`).
    pub fn render_plan(&self, req: &GenerationRequest) -> gen_core::Result<BerniniRenderPlan> {
        build_bernini_render_plan(
            &self.descriptor,
            BerniniEntrypoint::Full,
            BerniniBackendKind::Candle,
            self.knobs.to_plan_knobs(),
            req,
        )
    }

    /// The cached shared renderer components — loaded once, then cloned (`Arc`) on every subsequent
    /// generate. The load lock serializes racing first-callers so two concurrent generates don't
    /// both mmap + upload ~50 GB (F-097).
    fn components(&self) -> CResult<Arc<RendererComponents>> {
        candle_gen::cached(&self.components, || {
            if let (Some(contract), Some(tier)) = (&self.memory_strategy, self.memory_tier) {
                crate::memory_strategy::validate_loaded_contract(
                    MODEL_ID,
                    &self.loaded_spec,
                    contract,
                    tier,
                )
                .map_err(|error| CandleError::Msg(error.to_string()))?;
            }
            let components =
                RendererComponents::load(&self.root, &self.device, MODEL_ID, &self.adapters)?;
            if let (Some(contract), Some(tier)) = (&self.memory_strategy, self.memory_tier) {
                crate::memory_strategy::validate_loaded_contract(
                    MODEL_ID,
                    &self.loaded_spec,
                    contract,
                    tier,
                )
                .map_err(|error| CandleError::Msg(error.to_string()))?;
            }
            Ok(Arc::new(components))
        })
    }
}

/// Load the full Bernini pipeline from a combined snapshot dir (the [`crate::convert`] renderer tier +
/// the planner components). Lazy: the heavy weights load on the first `generate`, so registration +
/// descriptor + `validate` resolve for a missing dir.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "bernini: expected a model directory (converted full-Bernini snapshot: transformer/ \
                 transformer_2/ text_encoder/ vae/ tokenizer/ mllm/ connector/ vit_decoder/), not a \
                 single .safetensors file"
                    .into(),
            ))
        }
    };
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "bernini does not support control / VACE / IP-adapter overlays".into(),
        ));
    }
    let knobs = BerniniKnobs::from_dir(&root)?;
    let device = candle_gen::default_device()?;
    // A crossed, corrupted, or unparseable snapshot yields no contract — but it must never yield a
    // silent `Accept` at the memory seams, so carry the refusal with the load.
    #[cfg(any(feature = "cuda", test))]
    let (memory_strategy, memory_tier, memory_refusal) =
        match crate::memory_strategy::contract_for_loaded(spec, MODEL_ID) {
            Ok((contract, tier)) => (Some(contract), Some(tier), String::new()),
            Err(reason) => (None, None, reason),
        };
    #[cfg(not(any(feature = "cuda", test)))]
    let (memory_strategy, memory_tier, memory_refusal) = (
        None,
        None,
        format!("{MODEL_ID}: this build has no Candle/CUDA memory-strategy support"),
    );
    Ok(Box::new(Bernini {
        descriptor: descriptor(),
        knobs,
        root,
        device,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
        memory_strategy,
        memory_tier,
        memory_refusal,
        loaded_spec: spec.clone(),
    }))
}

impl Generator for Bernini {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.render_plan(req)?;
        // Reconciled memory lanes: the existing public still surface and SC-20765's exact V2V
        // surface share the provider but retain separate, fail-closed request identities.
        if req.frames == Some(1) || req.memory.is_some() {
            crate::memory_strategy::validate_public_request(req)?;
        }
        Ok(())
    }

    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        crate::components::capture_renderer_execution(self, req, on_progress, self.adapters.len())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        Ok(self.generate_impl(req, on_progress)?)
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let (Some(contract), Some(tier)) = (&self.memory_strategy, self.memory_tier) else {
            return gen_core::MemorySafetyDecision::Reject {
                reason: self.memory_refusal.clone(),
            };
        };
        crate::memory_strategy::safety_check(contract, tier, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let (Some(contract), Some(tier)) = (&self.memory_strategy, self.memory_tier) else {
            return Err(gen_core::Error::Unsupported(self.memory_refusal.clone()));
        };
        crate::memory_strategy::begin_request(contract, tier, self.device.clone(), context)
    }
}

impl Bernini {
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<GenerationOutput> {
        let dev = &self.device;
        let render_plan = self
            .render_plan(req)
            .map_err(|e| CandleError::Msg(e.to_string()))?;

        // The heavy renderer components (UMT5 + both experts + z16 VAE + tokenizer) are loaded once and
        // cached across generates (F-097); only the planner is reloaded per request (below).
        let comps = self.components()?;

        // --- Geometry + knobs ---
        let frames = render_plan.frames;
        let out_video = frames > 1;
        let width = render_plan.width;
        let height = render_plan.height;
        let steps = render_plan.steps;
        let seed = render_plan.seed.unwrap_or_else(gen_core::default_seed);
        let neg = req.negative_prompt.clone().unwrap_or_default();

        let mode = vit_mode_from_plan(
            render_plan
                .vit_mode
                .expect("Full Bernini plan always resolves a VitMode"),
        );
        // The planner's system prompt keys off a *task* name; map a guidance-mode / absent `video_mode`
        // to a canonical task so it doesn't fall through to the generic system prompt (F-161).
        let task = render_plan.canonical_task.as_str();
        let planner_steps = FullDefaults::PLANNING_STEP as u32;
        let folded_total = render_plan.folded_progress_total as u32;

        // The z16 VAE (with encoder) is resident throughout (cached component, F-097): it VAE-encodes
        // the source media into z16 conditioning latents up front and decodes the final latent at the
        // end.
        let vae = &comps.vae;

        // --- Stage 1: planner (loaded → 3 streams + MAR loop → freed) ---
        // Source latents (VAE) are computed here and carried past the planner drop; the ViT features are
        // consumed inside the planner block.
        let (
            max_seq,
            src_videos,
            src_images,
            s_wtxt_wvit,
            s_wtxt_wovit,
            s_wotxt_wvit,
            s_wotxt_wovit,
        ) = {
            let mut planner = BerniniPlanner::load(&self.root, dev)?;

            // Preprocess the conditioning: videos first, then images (conversation / source_id order).
            let mut videos: Vec<SourceVisual> = Vec::new();
            let mut images: Vec<SourceVisual> = Vec::new();
            for c in &req.conditioning {
                match c {
                    Conditioning::VideoClip { frames: clip, .. } => {
                        let rgb: Vec<RgbImage> = clip.iter().map(to_rgb).collect::<CResult<_>>()?;
                        let vit_frames = sample_vit_frames(&rgb);
                        let (vit_feat, vit_grid) = vit_encode_video(&planner, &vit_frames, dev)?;
                        let vae_latent = encode_videoclip(vae, clip, width, height, dev)?;
                        let hw = (rgb[0].height() as i64, rgb[0].width() as i64);
                        videos.push(SourceVisual {
                            vit_feat,
                            vit_grid,
                            vae_latent,
                            hw,
                        });
                    }
                    Conditioning::Reference { image, .. } => {
                        let rgb = to_rgb(image)?;
                        let (vit_feat, vit_grid) = vit_encode_image(&planner, &rgb, dev)?;
                        let vae_latent = encode_image(vae, image, width, height, dev)?;
                        images.push(SourceVisual {
                            vit_feat,
                            vit_grid,
                            vae_latent,
                            hw: (image.height as i64, image.width as i64),
                        });
                    }
                    Conditioning::MultiReference { images: imgs } => {
                        for image in imgs {
                            let rgb = to_rgb(image)?;
                            let (vit_feat, vit_grid) = vit_encode_image(&planner, &rgb, dev)?;
                            let vae_latent = encode_image(vae, image, width, height, dev)?;
                            images.push(SourceVisual {
                                vit_feat,
                                vit_grid,
                                vae_latent,
                                hw: (image.height as i64, image.width as i64),
                            });
                        }
                    }
                    _ => {}
                }
            }

            let gen_grid = gen_target_grid(
                height as i64,
                width as i64,
                frames,
                req.fps.unwrap_or(FullDefaults::FPS),
            );

            // Three streams: cond (full), imgcond (text, no visuals), uncond (neg text, no visuals).
            let (cond, n_query) = build_stream(
                &planner,
                task,
                &req.prompt,
                &images,
                &videos,
                gen_grid,
                out_video,
                height as i64,
                width as i64,
                dev,
            )?;
            let (imgcond, _) = build_stream(
                &planner,
                task,
                &req.prompt,
                &[],
                &[],
                gen_grid,
                out_video,
                height as i64,
                width as i64,
                dev,
            )?;
            let (uncond, _) = build_stream(
                &planner,
                task,
                &neg,
                &[],
                &[],
                gen_grid,
                out_video,
                height as i64,
                width as i64,
                dev,
            )?;

            if n_query > planner.knobs.num_mask_token {
                return Err(CandleError::Msg(format!(
                    "bernini: gen-target needs {n_query} ViT tokens but the planner has only {} mask \
                     tokens — lower the resolution/frames",
                    planner.knobs.num_mask_token
                )));
            }

            let vit_cfg = VitCfg {
                planning_step: FullDefaults::PLANNING_STEP,
                vit_denoising_step: FullDefaults::VIT_DENOISING_STEP,
                vit_txt_cfg: FullDefaults::VIT_TXT_CFG,
                vit_img_cfg: FullDefaults::VIT_IMG_CFG,
            };
            let order = seeded_permutation(n_query, seed);
            let step_noise = seeded_step_noise(
                n_query,
                vit_cfg.planning_step,
                &order,
                planner.knobs.clip_diff_in_channels,
                seed,
                dev,
            )?;

            // MAR planning loop (disjoint field borrows: backbone/connector shared, clip_diff mutable).
            let mut on_planner_step = |step_1_based: usize| {
                on_progress(Progress::Step {
                    current: step_1_based as u32,
                    total: folded_total,
                });
            };
            let streams = sample_vit_embed_with_progress(
                &planner.backbone,
                &planner.connector,
                &mut planner.clip_diff,
                &cond,
                &uncond,
                &imgcond,
                &vit_cfg,
                &order,
                &step_noise,
                &req.cancel,
                &mut on_planner_step,
                &planner.mask_token,
            )?;

            // Cast the planner streams to f32 for the UMT5 concat (UMT5 runs f32). The renderer's
            // `embed_text` consumes the f32 context (as the renderer path does).
            let f32c = |a: &Tensor| a.to_dtype(ENC_DTYPE);
            let out = (
                planner.knobs.max_sequence_length.max(1),
                videos
                    .iter()
                    .map(|s| s.vae_latent.clone())
                    .collect::<Vec<_>>(),
                images
                    .iter()
                    .map(|s| s.vae_latent.clone())
                    .collect::<Vec<_>>(),
                f32c(&streams.wtxt_wvit)?,
                f32c(&streams.wtxt_wovit)?,
                f32c(&streams.wotxt_wvit)?,
                f32c(&streams.wotxt_wovit)?,
            );
            out
            // Only planner-local state is dropped here. The renderer components in `comps`, loaded
            // through `self.components()` above, remain cached and resident for the whole request
            // (F-097).
        };

        // --- Stage 2: UMT5 text encode + concat_with_zero_init for the 4 renderer streams ---
        // UMT5 encoder + tokenizer are cached components (F-097), not reloaded per generate.
        let (t5_pos, t5_neg) = {
            let (pos, neg) = candle_gen::gen_core::with_prompt_encoding(
                on_progress,
                || -> std::result::Result<_, candle_gen::CandleError> {
                    let pos = umt5_encode(&comps.te, &comps.tok, &req.prompt, dev)?;
                    let neg = umt5_encode(&comps.te, &comps.tok, &neg, dev)?;
                    Ok((pos, neg))
                },
            )?;
            (pos, neg)
        };
        let pe_wtxt_wvit = concat_with_zero_init(&t5_pos, &s_wtxt_wvit, max_seq)?;
        let pe_wtxt_wovit = concat_with_zero_init(&t5_pos, &s_wtxt_wovit, max_seq)?;
        let pe_wotxt_wvit = concat_with_zero_init(&t5_neg, &s_wotxt_wvit, max_seq)?;
        let pe_wotxt_wovit = concat_with_zero_init(&t5_neg, &s_wotxt_wovit, max_seq)?;

        // --- Stage 3: load both experts, ViT-conditioned APG denoise ---
        let (t_lat, h_lat, w_lat) = latent_dims(frames as u32, width, height);
        let init_noise = create_noise(seed, Z_DIM, t_lat, h_lat, w_lat, dev)?;

        // Source ids (videos first, then images — mirrors the packing order).
        let (nv, ni) = (src_videos.len(), src_images.len());
        // ADS2V uses the trained source/reference/first-image triplet; extra images interpolate
        // across ids 1..=3, matching its admission receipt.
        let trained_sources = render_plan.trained_sources;
        let sids = assign_source_ids(nv + ni, trained_sources, self.knobs.interpolate_src_id);
        let video_srcs: Vec<(Tensor, f64)> = src_videos
            .iter()
            .enumerate()
            .map(|(k, v)| (v.clone(), sids[k]))
            .collect();
        let image_srcs: Vec<(Tensor, f64)> = src_images
            .iter()
            .enumerate()
            .map(|(j, im)| (im.clone(), sids[nv + j]))
            .collect();

        let base_g = VitGuidanceParams {
            omega_txt: render_plan.omega_txt,
            omega_img: render_plan.omega_img,
            omega_vid: render_plan.omega_vid,
            omega_tgt: render_plan.omega_tgt.unwrap_or(FullDefaults::OMEGA_TGT),
            eta: render_plan.eta,
            norm_threshold: FullDefaults::NORM_THRESHOLD,
        };

        let dit_cfg = TransformerConfig::t2v_14b();
        let latents = {
            // Both 14B experts are cached components (F-097), not reloaded per generate.
            let streams4 = [
                &pe_wtxt_wvit,
                &pe_wtxt_wovit,
                &pe_wotxt_wvit,
                &pe_wotxt_wovit,
            ];
            let high = BVitExpert::build(&comps.high, streams4)?;
            let low = BVitExpert::build(&comps.low, streams4)?;
            let pf = PackedForward::new(dit_cfg, trained_sources, self.knobs.interpolate_src_id);
            let boundary_ts = render_plan.boundary_timestep;
            let shift = render_plan.shift;
            let sampler = Sampler::parse(Some(&render_plan.sampler));
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: planner_steps + i as u32 + 1,
                    total: folded_total,
                })
            };
            denoise_bernini_wvitcfg(
                &pf,
                mode,
                &low,
                &high,
                boundary_ts,
                sampler,
                steps,
                shift,
                &init_noise,
                &image_srcs,
                &video_srcs,
                &base_g,
                render_plan.omega_scale,
                &req.cancel,
                &mut on_step,
            )?
        };

        // --- Stage 4: z16 VAE decode → image / video ---
        on_progress(Progress::Decoding);
        let decode_cap = crate::memory_strategy::selected_decode_cap(req)?;
        let decoded =
            vae.decode_budgeted_with_cancel_and_tile_cap(&latents, &req.cancel, decode_cap)?;
        let images_out = frames_to_images(&decoded)?;

        match &render_plan.output_contract {
            BerniniOutputContract::Images { .. } => {
                let first = images_out.into_iter().next().ok_or_else(|| {
                    CandleError::Msg("bernini: VAE decode produced no frames".into())
                })?;
                Ok(GenerationOutput::Images(vec![first]))
            }
            BerniniOutputContract::Video { fps, .. } => Ok(GenerationOutput::Video {
                frames: images_out,
                fps: *fps,
                audio: None,
            }),
        }
    }
}

/// One expert (high or low) with its 4 prompt-embed streams already `embed_text`-projected into the
/// expert's context space (the full-Bernini ViT-conditioned path). Each stream is
/// `concat_with_zero_init(UMT5(prompt), planner ViT-context)` in renderer `text_dim` space, so it goes
/// through the same `embed_text` as the renderer's text context.
pub struct BVitExpert<'a> {
    transformer: &'a WanTransformer,
    wtxt_wvit: Tensor,
    wtxt_wovit: Tensor,
    wotxt_wvit: Tensor,
    wotxt_wovit: Tensor,
}

impl<'a> BVitExpert<'a> {
    /// `streams` = `[wtxt_wvit, wtxt_wovit, wotxt_wvit, wotxt_wovit]` prompt-embed contexts (each
    /// `[1, S, text_dim]`).
    pub fn build(dit: &'a WanTransformer, streams: [&Tensor; 4]) -> CResult<Self> {
        Ok(Self {
            transformer: dit,
            wtxt_wvit: dit.embed_text(streams[0])?,
            wtxt_wovit: dit.embed_text(streams[1])?,
            wotxt_wvit: dit.embed_text(streams[2])?,
            wotxt_wovit: dit.embed_text(streams[3])?,
        })
    }

    fn streams(&self) -> VitStreams<'_> {
        VitStreams {
            wtxt_wvit: &self.wtxt_wvit,
            wtxt_wovit: &self.wtxt_wovit,
            wotxt_wvit: &self.wotxt_wvit,
            wotxt_wovit: &self.wotxt_wovit,
        }
    }
}

/// The full-Bernini ViT-conditioned denoise loop (`sample_bernini_wvitcfg`) — the renderer-side compute
/// that consumes the planner's 4 prompt streams. The boundary-switched, [`vit_one_step`]-guided analog of
/// the renderer's plain denoise: each step picks the expert by `switch_dit_boundary`, multiplies **all
/// four** omegas (incl. `omega_tgt`) by `omega_scale` once on the first low-noise step, and applies the
/// flow step. Runs in spatial latent space `[1, 16, T, H8, W8]`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_bernini_wvitcfg(
    pf: &PackedForward,
    mode: VitMode,
    low: &BVitExpert,
    high: &BVitExpert,
    boundary_ts: f64,
    sampler: Sampler,
    steps: usize,
    shift: f64,
    init_noise: &Tensor,
    images: &[(Tensor, f64)],
    videos: &[(Tensor, f64)],
    base_g: &VitGuidanceParams,
    omega_scale: f32,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> CResult<Tensor> {
    candle_gen::execution::policy(
        "bernini_vit_guidance",
        serde_json::json!({
            "mode": format!("{mode:?}"), "boundary_bits": boundary_ts.to_bits(),
            "omega_scale_bits": omega_scale.to_bits(),
            "omega_txt_bits": base_g.omega_txt.to_bits(), "omega_img_bits": base_g.omega_img.to_bits(),
            "omega_vid_bits": base_g.omega_vid.to_bits(), "omega_tgt_bits": base_g.omega_tgt.to_bits(),
            "eta_bits": base_g.eta.to_bits(), "norm_threshold_bits": base_g.norm_threshold.to_bits()
        }),
    );

    struct VitStage2Forward<'a> {
        pf: &'a PackedForward,
        mode: VitMode,
        low: &'a BVitExpert<'a>,
        high: &'a BVitExpert<'a>,
        images: &'a [(Tensor, f64)],
        videos: &'a [(Tensor, f64)],
        base_g: &'a VitGuidanceParams,
    }

    impl BerniniStage2Forward for VitStage2Forward<'_> {
        fn predict_step_velocity(
            &self,
            expert: BerniniExpert,
            latent: &Tensor,
            _step: usize,
            timestep: f64,
            sigma: f32,
            omegas: BerniniStage2Omegas,
            _switched_to_low: bool,
            mbufs: &mut [MomentumBuffer],
        ) -> CResult<Tensor> {
            let expert_ref = match expert {
                BerniniExpert::High => self.high,
                BerniniExpert::Low => self.low,
            };
            let g = omegas.to_vit(self.base_g);
            vit_one_step(
                self.pf,
                expert_ref.transformer,
                self.mode,
                latent,
                self.images,
                self.videos,
                timestep,
                sigma,
                &expert_ref.streams(),
                &g,
                mbufs,
            )
        }
    }

    let mut progress_adapter = |progress: Progress| {
        if let Progress::Step { current, .. } = progress {
            on_step(current as usize - 1);
        }
    };
    run_bernini_candle_stage2_driver(
        "bernini_candle_vit_stage2_driver",
        sampler,
        boundary_ts,
        steps,
        shift,
        init_noise,
        BerniniStage2Omegas::from_vit(base_g),
        omega_scale,
        num_vit_momentum_buffers(mode),
        FullDefaults::MOMENTUM,
        0,
        cancel,
        &mut progress_adapter,
        VitStage2Forward {
            pf,
            mode,
            low,
            high,
            images,
            videos,
            base_g,
        },
    )
}

/// UMT5-encode `prompt` → **unpadded** `[1, S, 4096]` f32 (the `concat_with_zero_init` prepend expects
/// the raw prompt embeds, then pads/truncates the T5+planner concat to `max_sequence_length`). The
/// empty-prompt guard emits one pad token so a 0-length sequence never reaches the embedding gather.
fn umt5_encode(
    enc: &Umt5Encoder,
    tok: &TextTokenizer,
    prompt: &str,
    dev: &Device,
) -> CResult<Tensor> {
    let te_cfg = TextEncoderConfig::umt5_xxl();
    let out = tok
        .tokenize(prompt)
        .map_err(|e| CandleError::Msg(format!("bernini: tokenize: {e}")))?;
    let mut ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
    if ids.is_empty() {
        ids.push(te_cfg.pad_token_id as u32);
    }
    let len = ids.len();
    let input_ids = Tensor::from_vec(ids, (1, len), dev)?;
    Ok(enc.encode(&input_ids)?.to_dtype(ENC_DTYPE)?)
}

/// Reject empty / non-`1+4k` conditioning video clips before the pipeline dereferences `frames[0]`
/// (mirrors the renderer's `encode_videoclip` guard). Free fn so it is unit-testable without weights.
#[cfg(test)]
fn validate_conditioning_video_clips(req: &GenerationRequest) -> CResult<()> {
    for c in &req.conditioning {
        if let Conditioning::VideoClip { frames, .. } = c {
            if frames.is_empty() {
                return Err(CandleError::Msg(
                    "bernini: empty conditioning video clip".into(),
                ));
            }
            if frames.len() % 4 != 1 {
                return Err(CandleError::Msg(format!(
                    "bernini: conditioning video-clip frame count must be 1 + 4·k (got {})",
                    frames.len()
                )));
            }
        }
    }
    Ok(())
}

/// Refuse the [`Conditioning::VideoClip`] knobs Bernini does not implement (sc-20264).
///
/// Bernini consumes a `VideoClip` as a **source clip to condition on**: `generate_impl` VAE-encodes
/// `frames` and hands the latents to the MAR sampler as conditioning tokens. Neither of the
/// variant's two numeric fields has a mechanism here:
///
/// * `strength` is the `1 − strength` denoise mask of the LTX in-context append path. Bernini's
///   conditioning enters through the guidance mode (`omega_vid` / `omega_img` / `omega_txt`) and the
///   ViT/VAE conditioning stack, not through a per-clip denoise mask there is anything to weight.
/// * `frame_idx` is the output latent frame the clip is appended at. Bernini conditions on the clip
///   as a whole and renders its own `req.frames`-length timeline from scratch.
///
/// Until sc-20264 both were read off the request and thrown away. It fires **only on a non-default
/// value**: the contract defaults (`strength = 1.0`, `frame_idx = 0`) pass through unchanged.
///
/// Checked over **every** clip, not the first: Bernini's multi-video modes carry several.
///
/// Shared by BOTH providers this crate registers — `bernini` (here) and `bernini_renderer`
/// (`pipeline.rs`). Byte-for-byte the MLX sibling's `reject_unimplemented_video_clip_knobs`. Typed
/// [`gen_core::Error::Unsupported`], not `Msg`: the worker classifies `Unsupported` as a user-facing
/// invalid-payload refusal and `Msg` as an opaque internal engine failure.
#[cfg(test)]
pub(crate) fn reject_unimplemented_video_clip_knobs(
    id: &str,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    for clip in req.video_clips() {
        if clip.strength != 1.0 {
            return Err(gen_core::Error::Unsupported(format!(
                "{id} does not implement VideoClip strength (got {}); remove it or leave it at the \
                 default 1.0 — Bernini conditions through its guidance mode and has no per-clip \
                 denoise mask to weight",
                clip.strength
            )));
        }
        if clip.frame_idx != 0 {
            return Err(gen_core::Error::Unsupported(format!(
                "{id} does not implement VideoClip frame_idx (got {}); remove it or leave it at \
                 the default 0 — Bernini conditions on the clip as a whole and renders its own \
                 timeline, so there is no position to splice it at",
                clip.frame_idx
            )));
        }
    }
    Ok(())
}

/// Sub-sample a decoded clip to the ViT frame set (`smart_video_nframes`, assuming `target_fps`).
fn sample_vit_frames(frames: &[RgbImage]) -> Vec<RgbImage> {
    let fps = FullDefaults::FPS as f64;
    let vit_fps = (FullDefaults::FPS / 8).max(1) as f64;
    let idx = smart_video_nframes(
        frames.len() as i64,
        fps,
        vit_fps,
        Some(TEMPORAL_PATCH_SIZE),
        None,
        Some(frames.len() as i64),
        false,
    );
    idx.iter()
        .map(|&i| frames[(i as usize).min(frames.len() - 1)].clone())
        .collect()
}

// --- MAR reveal order + per-step FM noise (reference torch/numpy MT19937 draws, sc-11671). ---
// The reference `sample_vit_embed` draws the reveal permutation via numpy `np.random.shuffle` and the
// per-step FM base noise via `torch.randn`; both are reimplemented bit-for-bit in `crate::rng` so the
// candle MAR trajectory matches torch/numpy exactly (golden `mar_mt19937_golden`). A u64 request seed is
// reduced to its low 32 bits (numpy's legacy scalar seeding rejects seeds ≥ 2³²; torch's MT19937 seeds
// from `seed & 0xffffffff` regardless).

/// The reveal permutation of `[0, n)` — numpy `RandomState(seed).shuffle(arange(n))` (bit-exact).
fn seeded_permutation(n: i32, seed: u64) -> Vec<i32> {
    candle_gen::execution::rng_scheme("bernini_numpy_mt19937_shuffle_v1");
    candle_gen::execution::stream("bernini_numpy_mt19937_shuffle_v1", seed as u32 as u64);
    candle_gen::execution::policy(
        "bernini_planner_rng",
        serde_json::json!({"scheme": "numpy-mt19937-shuffle-v1", "query_tokens": n}),
    );
    crate::rng::numpy_shuffle(n.max(0) as usize, seed as u32)
}

/// Per-step base FM noise for the MAR loop — one `[revealed, in]` tensor per planning step (the
/// reference's `torch.randn(n_revealed, in)`, tiled ×3 inside `DiffLossFm::sample`). Skipped steps
/// (`revealed.sum() == 0`) draw no torch noise and get a placeholder `[max(1, np), in]` zero tensor
/// that `sample_vit_embed` never reads (it `continue`s before the `step_noise[step]` lookup).
fn seeded_step_noise(
    n_query: i32,
    planning_step: usize,
    order: &[i32],
    in_channels: usize,
    seed: u64,
    dev: &Device,
) -> CResult<Vec<Tensor>> {
    let schedule = mar_schedule(n_query, planning_step, order);
    candle_gen::execution::rng_scheme("bernini_torch_mt19937_normal_v1");
    candle_gen::execution::stream("bernini_torch_mt19937_normal_v1", seed as u32 as u64);
    candle_gen::execution::policy(
        "bernini_planner_noise",
        serde_json::json!({
            "scheme": "torch-mt19937-normal-v1", "schedule": schedule, "channels": in_channels
        }),
    );
    let per_step = crate::rng::torch_step_noise(&schedule, in_channels, seed as u32);
    let mut out = Vec::with_capacity(planning_step);
    for (revealed, data) in schedule.iter().zip(per_step) {
        if data.is_empty() {
            let np = revealed.len().max(1);
            out.push(Tensor::zeros((np, in_channels), DType::F32, dev)?);
        } else {
            out.push(Tensor::from_vec(data, (revealed.len(), in_channels), dev)?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfix::{tiny_cfg, tiny_dit};
    use candle_gen_wan::config::NUM_TRAIN_TIMESTEPS;

    /// The full `bernini` engine registers + resolves via the gen_core registry (lazy load succeeds for
    /// a missing dir; the descriptor identity is what we pin).
    #[test]
    fn registers_and_resolves() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .expect("bernini is registered");
        assert_eq!(g.descriptor().id, "bernini");
        assert_eq!(g.descriptor().family, "bernini");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
        assert!(!g.descriptor().capabilities.mac_only);
        assert!(!g.descriptor().capabilities.supports_sequential_offload);
        assert!(
            !g.descriptor()
                .capabilities
                .unconditionally_engages_staged_residency
        );
        assert_eq!(
            g.descriptor().capabilities.staged_residency_availability(),
            candle_gen::gen_core::StagedResidencyAvailability::Absent,
        );
    }

    /// sc-11061 layout contract (loader side): the exact on-disk paths [`BerniniPlanner::load`] reads
    /// are built from the shared `crate::convert::PLANNER_*` consts, so the converter's emitted layout
    /// and this reader can never drift. Pins those consts to the concrete relative paths under a snapshot
    /// root — the mirror of `planner_layout_consts_are_stable` in `convert.rs`.
    #[test]
    fn planner_layout_consts_match_loader() {
        let root = Path::new("/snap");
        // Root-level files.
        assert_eq!(
            root.join(PLANNER_QWEN_CONFIG_FILE),
            Path::new("/snap/qwen2_5_vl_config.json")
        );
        assert_eq!(
            root.join(PLANNER_MASK_TOKENS_FILE),
            Path::new("/snap/mask_tokens.safetensors")
        );
        assert_eq!(
            root.join(PLANNER_SIDECAR_FILE),
            Path::new("/snap/bernini_planner.json")
        );
        // Component dirs (mmap'd by `component_vb`) + the tokenizer inside `mllm/`.
        assert_eq!(root.join(PLANNER_MLLM_DIR), Path::new("/snap/mllm"));
        assert_eq!(
            root.join(PLANNER_CONNECTOR_DIR),
            Path::new("/snap/connector")
        );
        assert_eq!(
            root.join(PLANNER_VIT_DECODER_DIR),
            Path::new("/snap/vit_decoder")
        );
        assert_eq!(
            root.join(PLANNER_MLLM_DIR).join(PLANNER_TOKENIZER_FILE),
            Path::new("/snap/mllm/tokenizer.json")
        );
        // Intra-component `.pp(..)` namespaces the loader descends into.
        assert_eq!(PLANNER_MLLM_BACKBONE_PP, "model");
        assert_eq!(PLANNER_MLLM_VISION_PP, "visual");
        assert_eq!(PLANNER_VIT_DECODER_PP, "net");
        assert_eq!(PLANNER_MASK_TOKENS_KEY, "mask_tokens");
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert_eq!(d.id, "bernini");
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(d.capabilities.accepts(ConditioningKind::VideoClip));
        assert!(d.capabilities.supported_quants.contains(&Quant::Q4));
        assert!(d.capabilities.supported_quants.contains(&Quant::Q8));
    }

    #[test]
    fn load_rejects_single_file_and_overlays() {
        let f = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        assert!(load(&f).is_err());
        // A directory source is lazy-loaded, so it resolves past the marker.
        let d = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert!(load(&d).is_ok());
    }

    /// Guidance-mode resolution: an explicit guidance-mode name wins, then a task-type name, then the
    /// conditioning/output defaults.
    #[test]
    fn vit_mode_resolution() {
        assert_eq!(
            resolve_vit_mode(Some("vae_txt_vit_wapg"), false, false, false),
            VitMode::VaeTxtVitWapg
        );
        assert_eq!(
            resolve_vit_mode(Some("rv2v_wapg"), true, true, true),
            VitMode::Rv2vWapg
        );
        // task-name fallback
        assert_eq!(
            resolve_vit_mode(Some("t2i"), false, false, false),
            VitMode::VaeTxtVitWapg
        );
        assert_eq!(
            resolve_vit_mode(Some("v2v"), true, false, true),
            VitMode::V2vApg
        );
        assert_eq!(
            resolve_vit_mode(Some("r2v"), false, true, true),
            VitMode::Rv2vWapg
        );
        // conditioning/output-driven defaults
        assert_eq!(
            resolve_vit_mode(None, false, false, false),
            VitMode::VaeTxtVitWapg
        ); // t2i
        assert_eq!(resolve_vit_mode(None, true, false, true), VitMode::V2vApg); // v2v
        assert_eq!(resolve_vit_mode(None, true, true, true), VitMode::Rv2vWapg); // rv2v
        assert_eq!(resolve_vit_mode(None, false, true, true), VitMode::Rv2vWapg); // r2v
        assert_eq!(
            resolve_vit_mode(None, false, true, false),
            VitMode::VaeTxtVitWapg
        ); // i2i
    }

    #[test]
    fn grid_token_count() {
        assert_eq!(grid_tokens([1, 12, 20]), 60);
        assert_eq!(grid_tokens([5, 12, 20]), 300);
    }

    /// F-161: a guidance-mode name (or absent `video_mode`) maps to a real task name (never the generic
    /// system-prompt fallback); an explicit task name is honored verbatim.
    #[test]
    fn canonical_task_maps_guidance_mode_and_conditioning() {
        // Guidance-mode names are NOT task names → derive from conditioning/output.
        assert_eq!(canonical_task(Some("rv2v_wapg"), true, true, true), "rv2v");
        assert_eq!(
            canonical_task(Some("vae_txt_vit_wapg"), false, false, false),
            "t2i"
        );
        assert_eq!(canonical_task(Some("v2v_apg"), true, false, true), "v2v");
        // Explicit task names are honored verbatim.
        assert_eq!(canonical_task(Some("i2i"), false, true, false), "i2i");
        assert_eq!(canonical_task(Some("t2v"), false, false, true), "t2v");
        // Absent → conditioning/output-derived task.
        assert_eq!(canonical_task(None, false, false, false), "t2i");
        assert_eq!(canonical_task(None, false, false, true), "t2v");
        assert_eq!(canonical_task(None, false, true, false), "i2i");
        assert_eq!(canonical_task(None, false, true, true), "r2v");
        assert_eq!(canonical_task(None, true, false, true), "v2v");
        assert_eq!(canonical_task(None, true, true, true), "rv2v");
    }

    /// F-095 + F-096: the full-pipeline `validate` enforces the shared geometry guard (zero steps /
    /// off-grid size / over-area / bad frame count) and rejects a conditioning-mode name with no source.
    #[test]
    fn validate_geometry_and_mode_conditioning() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let ok = GenerationRequest {
            prompt: "a cat walking across a garden".into(),
            width: 256,
            height: 256,
            frames: Some(17),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            // off-grid size (328 not a multiple of the 14B size multiple) — the F-095 opaque-shape case.
            GenerationRequest {
                prompt: "x".into(),
                width: 328,
                height: 256,
                sampler: Some("uni_pc".into()),
                ..Default::default()
            },
            // explicit zero steps (was silently promoted to 1).
            GenerationRequest {
                prompt: "x".into(),
                width: 256,
                height: 256,
                steps: Some(0),
                sampler: Some("uni_pc".into()),
                ..Default::default()
            },
            // over the max-area envelope.
            GenerationRequest {
                prompt: "x".into(),
                width: 1280,
                height: 1024,
                frames: Some(17),
                sampler: Some("uni_pc".into()),
                ..Default::default()
            },
            // frames not ≡ 1 (mod 4).
            GenerationRequest {
                prompt: "x".into(),
                width: 256,
                height: 256,
                frames: Some(16),
                sampler: Some("uni_pc".into()),
                ..Default::default()
            },
            // F-096: a conditioning-mode name with no source would silently render text-only.
            GenerationRequest {
                prompt: "x".into(),
                width: 256,
                height: 256,
                video_mode: Some("v2v".into()),
                sampler: Some("uni_pc".into()),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
    }

    /// Image targets are single-frame (`t = 1`); video targets sample `vit_fps` frames so `t > 1`; the
    /// spatial grid is the `smart_resize` of the output H/W.
    #[test]
    fn gen_target_grid_image_vs_video() {
        let img = gen_target_grid(480, 832, 1, 16);
        assert_eq!(img[0], 1, "image target is single-frame");
        assert_eq!([img[1], img[2]], [12, 20], "480x832 → 12x20 grid");
        let vid = gen_target_grid(480, 832, 81, 16);
        assert!(vid[0] > 1, "video target spans multiple temporal patches");
        assert_eq!([vid[1], vid[2]], [12, 20], "same spatial grid");
    }

    #[test]
    fn seeded_permutation_is_a_permutation() {
        let n = 60;
        let a = seeded_permutation(n, 42);
        let b = seeded_permutation(n, 42);
        assert_eq!(a, b, "deterministic for a fixed seed");
        let c = seeded_permutation(n, 7);
        assert_ne!(a, c, "different seeds → different order");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "covers [0, n) once");
    }

    #[test]
    fn step_noise_shapes_match_schedule() {
        let n = 60;
        let steps = 25;
        let order = seeded_permutation(n, 42);
        let noise = seeded_step_noise(n, steps, &order, 3584, 42, &Device::Cpu).unwrap();
        assert_eq!(noise.len(), steps);
        let schedule = mar_schedule(n, steps, &order);
        for (s, arr) in noise.iter().enumerate() {
            let np = schedule[s].len().max(1);
            assert_eq!(arr.dims(), &[np, 3584], "step {s} noise shape");
        }
    }

    #[test]
    fn rejects_empty_and_miscounted_conditioning_video_clips() {
        let clip = |n: usize| Conditioning::VideoClip {
            frames: vec![
                Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0u8; 2 * 2 * 3],
                };
                n
            ],
            frame_idx: 0,
            strength: 1.0,
        };
        let req = |conds: Vec<Conditioning>| GenerationRequest {
            conditioning: conds,
            ..Default::default()
        };
        assert!(validate_conditioning_video_clips(&req(vec![clip(0)])).is_err());
        assert!(validate_conditioning_video_clips(&req(vec![clip(3)])).is_err());
        assert!(validate_conditioning_video_clips(&req(vec![clip(1)])).is_ok());
        assert!(validate_conditioning_video_clips(&req(vec![clip(5)])).is_ok());
        assert!(validate_conditioning_video_clips(&req(vec![])).is_ok());
        assert!(validate_conditioning_video_clips(&req(vec![clip(5), clip(0)])).is_err());
    }

    /// sc-20264 — `VideoClip.strength` and `frame_idx` were silently dropped (every construction
    /// site binds `VideoClip { frames, .. }`). Each is now a typed `Unsupported` naming the field
    /// and the model, on BOTH ids this crate registers.
    ///
    /// Non-default values throughout: at the contract defaults the refusal must not fire, which the
    /// companion test pins.
    #[test]
    fn non_default_video_clip_knobs_are_refused_by_name() {
        let clip = |frame_idx: i32, strength: f32| Conditioning::VideoClip {
            frames: vec![Image {
                width: 2,
                height: 2,
                pixels: vec![0u8; 2 * 2 * 3],
            }],
            frame_idx,
            strength,
        };
        let req = |c: Conditioning| GenerationRequest {
            conditioning: vec![c],
            ..Default::default()
        };
        for id in [MODEL_ID, crate::pipeline::MODEL_ID] {
            for (conditioning, field) in [(clip(0, 0.6), "strength"), (clip(4, 1.0), "frame_idx")] {
                let err = reject_unimplemented_video_clip_knobs(id, &req(conditioning))
                    .expect_err("a non-default knob must be refused");
                assert!(
                    matches!(err, gen_core::Error::Unsupported(_)),
                    "{id}/{field}: typed Unsupported, got {err:?}"
                );
                let msg = err.to_string();
                assert!(msg.contains(field), "{id}/{field}: names the field: {msg}");
                assert!(msg.contains(id), "{id}/{field}: names the model: {msg}");
            }
        }

        // Every clip is inspected, not just the first — the multi-video modes carry several.
        let two_clips = GenerationRequest {
            conditioning: vec![clip(0, 1.0), clip(7, 1.0)],
            ..Default::default()
        };
        let err = reject_unimplemented_video_clip_knobs(MODEL_ID, &two_clips)
            .expect_err("clip two's frame_idx must be refused");
        assert!(err.to_string().contains("frame_idx"), "got: {err}");
    }

    /// sc-20264 (adversarial-review follow-up) — the refusal is pinned **through
    /// `Generator::validate`**, not by calling the helper.
    ///
    /// A helper-only test leaves the wiring unbound: deleting the
    /// `reject_unimplemented_video_clip_knobs(id, req)?` line from either provider's `validate`
    /// keeps every helper assertion green while the knobs go back to being silently dropped. Both
    /// candle bernini ids load lazily from a nonexistent dir (`registers_and_resolves`), so the real
    /// registered `Generator` is drivable here without weights — the same way the scail2 / seedvr2 /
    /// krea / VACE refusals are pinned.
    #[test]
    fn generator_validate_refuses_non_default_video_clip_knobs_on_both_ids() {
        let registry = crate::provider_registry().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        // 5 frames: `1 + 4·k`, so the clip clears the shape guard that runs first and the refusal
        // is what the request actually trips.
        let clip = |frame_idx: i32, strength: f32| Conditioning::VideoClip {
            frames: (0..5)
                .map(|_| Image {
                    width: 16,
                    height: 16,
                    pixels: vec![0u8; 16 * 16 * 3],
                })
                .collect(),
            frame_idx,
            strength,
        };
        let req = |c: Conditioning| GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 256,
            height: 256,
            frames: Some(5),
            conditioning: vec![c],
            ..Default::default()
        };
        for id in [MODEL_ID, crate::pipeline::MODEL_ID] {
            let g = registry.load(id, &spec).expect("lazy load");
            for (conditioning, field) in [(clip(0, 0.6), "strength"), (clip(4, 1.0), "frame_idx")] {
                let err = g
                    .validate(&req(conditioning))
                    .expect_err("a non-default knob must be refused at validate");
                assert!(
                    matches!(err, gen_core::Error::Unsupported(_)),
                    "{id}/{field}: typed Unsupported, got {err:?}"
                );
                let msg = err.to_string();
                assert!(msg.contains(field), "{id}/{field}: names the field: {msg}");
                assert!(msg.contains(id), "{id}/{field}: names the model: {msg}");
            }
            // The contract defaults must NOT be refused by this check — whatever else validate
            // decides about the request, it must not be these fields.
            if let Err(e) = g.validate(&req(clip(0, 1.0))) {
                let msg = e.to_string();
                assert!(
                    !msg.contains("does not implement VideoClip"),
                    "{id}: default knobs must pass the refusal: {msg}"
                );
            }
        }
    }

    /// sc-20264 — the refusal fires only on a value a caller actually set. The contract defaults
    /// (`strength = 1.0`, `frame_idx = 0`) — every request SceneWorks builds today — pass through,
    /// as does a request carrying no clip at all.
    #[test]
    fn default_video_clip_knobs_still_pass_unchanged() {
        let default_clip = Conditioning::VideoClip {
            frames: vec![Image {
                width: 2,
                height: 2,
                pixels: vec![0u8; 2 * 2 * 3],
            }],
            frame_idx: 0,
            strength: 1.0,
        };
        for id in [MODEL_ID, crate::pipeline::MODEL_ID] {
            assert!(reject_unimplemented_video_clip_knobs(
                id,
                &GenerationRequest {
                    conditioning: vec![default_clip.clone()],
                    ..Default::default()
                }
            )
            .is_ok());
            assert!(
                reject_unimplemented_video_clip_knobs(id, &GenerationRequest::default()).is_ok()
            );
        }
    }

    /// The full-Bernini ViT-conditioned denoise loop runs end-to-end over a tiny dual-expert (crossing
    /// the boundary so the `omega_scale` switch fires) from the planner's 4 synthetic streams and
    /// preserves the spatial latent shape. Pins the capstone loop plumbing (scheduler / expert switch /
    /// per-step `vit_one_step` / flow step) without real weights.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn denoise_wvitcfg_runs_and_keeps_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let dit = tiny_dit(&cfg, &dev);
        let pf = PackedForward::new(cfg, 5.0, true);
        let noisy = Tensor::randn(0f32, 1f32, (1, 16, 3, 4, 4), &dev).unwrap();
        // 4 distinct prompt streams `[1, S, text_dim]`.
        let mk = |s: f32| {
            Tensor::randn(0f32, 1f32, (1, 5, 16), &dev)
                .unwrap()
                .affine(s as f64, 0.0)
                .unwrap()
        };
        let (s0, s1, s2, s3) = (mk(1.0), mk(0.7), mk(0.4), mk(0.2));
        let streams = [&s0, &s1, &s2, &s3];
        let low = BVitExpert::build(&dit, streams).unwrap();
        let high = BVitExpert::build(&dit, streams).unwrap();
        let g = VitGuidanceParams {
            omega_txt: 4.0,
            omega_img: 4.5,
            omega_vid: 1.25,
            omega_tgt: 0.5,
            eta: 1.0,
            norm_threshold: 50.0,
        };
        let mut on_step = |_i: usize| {};
        let out = denoise_bernini_wvitcfg(
            &pf,
            VitMode::VaeTxtVitWapg,
            &low,
            &high,
            0.875 * NUM_TRAIN_TIMESTEPS as f64, // boundary 875 → crossed within 4 steps
            Sampler::UniPC,
            4,
            5.0,
            &noisy,
            &[],
            &[],
            &g,
            0.8,
            &CancelFlag::default(),
            &mut on_step,
        )
        .expect("denoise");
        assert_eq!(
            out.dims(),
            noisy.dims(),
            "loop preserves spatial latent shape"
        );

        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED bernini::tests::denoise_wvitcfg_runs_and_keeps_shape"
        );
    }
}

#[cfg(test)]
mod execution_identity_tests {
    use super::*;

    #[test]
    fn planner_report_records_effective_low_u32_seeds_and_mixed_rngs() {
        let attempt = candle_gen::execution::Attempt::begin();
        let seed = (1u64 << 40) + 17;
        let order = seeded_permutation(4, seed);
        let _noise = seeded_step_noise(4, 2, &order, 2, seed, &Device::Cpu).unwrap();
        candle_gen::execution::runtime(&Device::Cpu, DType::F32);
        candle_gen::execution::schedule("fixture", &[1.0, 0.0], "flow");
        let output = GenerationOutput::Video {
            frames: vec![Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            }],
            fps: 16,
            audio: None,
        };
        let identity = attempt.finish_image(&descriptor(), &output);
        let candle_gen::gen_core::generator::ExecutionRandomness::Draws { draws, scheme } =
            &identity.resolved().unwrap().randomness
        else {
            panic!("draws")
        };
        assert_eq!(draws.len(), 2);
        assert_eq!(draws[0].seed, Some(17));
        assert_eq!(draws[1].seed, Some(17));
        assert_ne!(draws[0].purpose, draws[1].purpose);
        assert!(scheme.contains("numpy_mt19937"));
        assert!(scheme.contains("torch_mt19937"));
    }
}

#[cfg(test)]
mod render_plan_tests {
    use super::*;
    use gen_core_family::bernini_plan::{
        build_bernini_render_plan, BerniniCanonicalBranch, BerniniPlanKnobs, BerniniStageSpec,
    };

    #[test]
    fn bernini_candle_full_pre_load_plan_pins_mar_stages_branches_mt19937_seeds_and_outputs() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .expect("lazy load bernini");

        let dummy_img = || Image {
            width: 64,
            height: 64,
            pixels: vec![0u8; 64 * 64 * 3],
        };
        let dummy_clip = || Conditioning::VideoClip {
            frames: vec![dummy_img(); 5],
            frame_idx: 0,
            strength: 1.0,
        };

        let img_req = GenerationRequest {
            prompt: "a portrait in warm sunlight".into(),
            width: 512,
            height: 512,
            frames: Some(1),
            video_mode: Some("t2i".into()),
            steps: Some(12),
            sampler: Some("uni_pc".into()),
            seed: Some(77),
            ..Default::default()
        };
        assert!(g.validate(&img_req).is_ok());

        // `build_bernini_render_plan` also resolves omitted `video_mode` with `frames == 1` to `T2i`
        // while `Generator::validate` retains `memory_strategy`'s explicit still-route requirement.
        let omitted_mode_img_req = GenerationRequest {
            video_mode: None,
            ..img_req.clone()
        };
        assert!(g.validate(&omitted_mode_img_req).is_err());

        let img_plan = build_bernini_render_plan(
            g.descriptor(),
            BerniniEntrypoint::Full,
            BerniniBackendKind::Candle,
            BerniniPlanKnobs::default(),
            &img_req,
        )
        .expect("t2i full plan");
        let round_trip: BerniniRenderPlan =
            serde_json::from_str(&serde_json::to_string(&img_plan).unwrap()).unwrap();
        assert_eq!(round_trip, img_plan);
        assert_eq!(img_plan.branch, BerniniCanonicalBranch::T2i);
        assert_eq!(img_plan.canonical_task, "t2i");
        assert_eq!(img_plan.vit_mode, Some(BerniniFullVitMode::VaeTxtVitWapg));
        assert_eq!(img_plan.folded_progress_total, 25 + 12);
        assert_eq!(
            img_plan.output_contract,
            BerniniOutputContract::Images { expected_count: 1 }
        );
        assert_eq!(
            img_plan.seed_scheme.permutation_scheme,
            "bernini_numpy_mt19937_shuffle_v1"
        );
        assert_eq!(
            img_plan.seed_scheme.planner_step_noise_scheme,
            "bernini_torch_mt19937_normal_v1"
        );
        assert_eq!(
            img_plan.stages,
            vec![
                BerniniStageSpec::MarPlannerLoop {
                    planning_steps: 25,
                    vit_denoising_steps: 5,
                    vit_txt_cfg: 1.2,
                    vit_img_cfg: 1.0,
                    input_streams: ["cond".into(), "uncond".into(), "imgcond".into()],
                    handoff_streams: [
                        "wtxt_wvit".into(),
                        "wtxt_wovit".into(),
                        "wotxt_wvit".into(),
                        "wotxt_wovit".into(),
                    ],
                },
                BerniniStageSpec::DropPlanner {
                    flush_device_cache: false,
                },
                BerniniStageSpec::DualWan22A14bExpertsHighToLow {
                    boundary_timestep: 875.0,
                    switch_dit_boundary: 0.875,
                    num_train_timesteps: 1000,
                    omega_scale_once_on_low: 0.8,
                    steps: 12,
                    shift: 5.0,
                },
                BerniniStageSpec::VaeDecode {
                    output: BerniniOutputContract::Images { expected_count: 1 },
                },
            ]
        );

        let vid_req = GenerationRequest {
            prompt: "a sailboat gliding over emerald waves".into(),
            width: 512,
            height: 512,
            frames: Some(17),
            steps: Some(10),
            fps: Some(24),
            sampler: Some("uni_pc".into()),
            ..Default::default()
        };
        let vid_plan = build_bernini_render_plan(
            g.descriptor(),
            BerniniEntrypoint::Full,
            BerniniBackendKind::Candle,
            BerniniPlanKnobs::default(),
            &vid_req,
        )
        .expect("t2v full plan");
        assert_eq!(vid_plan.branch, BerniniCanonicalBranch::T2v);
        assert_eq!(
            vid_plan.output_contract,
            BerniniOutputContract::Video {
                fps: 24,
                latent_frames: 5,
            }
        );

        let ads2v_req = GenerationRequest {
            prompt: "drive product reveal".into(),
            width: 256,
            height: 256,
            frames: Some(5),
            video_mode: Some("ads2v".into()),
            conditioning: vec![
                dummy_clip(),
                Conditioning::Reference {
                    image: dummy_img(),
                    strength: Some(1.0),
                },
            ],
            ..Default::default()
        };
        let ads2v_plan = build_bernini_render_plan(
            g.descriptor(),
            BerniniEntrypoint::Full,
            BerniniBackendKind::Candle,
            BerniniPlanKnobs::default(),
            &ads2v_req,
        )
        .expect("ads2v plan");
        assert_eq!(ads2v_plan.branch, BerniniCanonicalBranch::Ads2v);
        assert_eq!(ads2v_plan.trained_sources, 3.0);
        assert_eq!(ads2v_plan.vit_mode, Some(BerniniFullVitMode::V2vApg));

        // Invalid combos refused pre-load by validate():
        for bad in [
            GenerationRequest {
                prompt: "".into(),
                width: 256,
                height: 256,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "missing cond".into(),
                width: 256,
                height: 256,
                video_mode: Some("rv2v_wapg".into()),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "bad grid".into(),
                width: 256,
                height: 256,
                frames: Some(6),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "zero steps".into(),
                width: 256,
                height: 256,
                steps: Some(0),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "expected pre-load refusal");
        }
    }

    #[test]
    fn bernini_candle_mt19937_seed_schemes_and_step_noise_remain_bit_exact() {
        let low_seed = 42u64;
        let high_seed = (1u64 << 40) + 42;
        assert_eq!(
            seeded_permutation(12, low_seed),
            crate::rng::numpy_shuffle(12, low_seed as u32)
        );
        assert_eq!(
            seeded_permutation(12, high_seed),
            seeded_permutation(12, low_seed),
            "Candle MT19937 reduces u64 seed to low 32 bits"
        );

        let order = seeded_permutation(6, low_seed);
        let planning_step = 25usize;
        let channels = 8usize;
        let schedule = mar_schedule(6, planning_step, &order);
        let expected = crate::rng::torch_step_noise(&schedule, channels, low_seed as u32);
        let got = seeded_step_noise(6, planning_step, &order, channels, low_seed, &Device::Cpu)
            .expect("step noise");
        assert_eq!(got.len(), planning_step);
        for (s, (revealed, tensor)) in schedule.iter().zip(got.iter()).enumerate() {
            let flat = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            if expected[s].is_empty() {
                assert_eq!(flat.len(), revealed.len().max(1) * channels);
                assert!(flat.iter().all(|&x| x == 0.0));
            } else {
                assert_eq!(flat, expected[s], "step {s} MT19937 noise parity");
            }
        }
    }
}
