//! Krea Realtime 14B **text-to-video pipeline** (sc-8439, S6): the end-to-end orchestration that
//! turns a prompt into an assembled RGB clip —
//! `prompt → UMT5 encode → context → AR few-step latents ([`generate_latents`]) → z16 Wan VAE decode
//! → [`GenerationOutput::Video`]`.
//!
//! Every heavy component is **reused from [`mlx_gen_wan`]** (Krea Realtime is Wan 2.1 T2V 14B
//! weight-for-weight): the UMT5-XXL text encoder ([`Umt5Encoder`] + [`load_tokenizer`]) and the z16
//! [`WanVae`] (encode/decode + `decode_to_frames`/`frames_to_images`). The only Krea-specific piece in
//! the chain is the AR causal denoise ([`crate::generate::generate_latents`] over
//! [`crate::CausalKreaTransformer`]).
//!
//! ## Batch vs. streaming
//! S6 delivers the **batch** form: the whole latent sequence is generated, then decoded in one VAE
//! pass ([`decode_latents_to_video`], single-pass; a gen-core [`TilingConfig`] bounds a large decode).
//! The realtime **streaming** per-chunk feat-cached decode (emit-as-you-generate) is deferred to the
//! streaming epic.
//!
//! ## Mac memory bound (sc-8438 S5 follow-up)
//! The shipped checkpoint is global (`local_attn_size = -1`, ~27 GB of KV on Mac). The pipeline path
//! selects the bounded streaming window ([`mac_ar_config`], `local_attn_size =
//! kv_cache_num_frames + num_frames_per_block` frames) so long clips stay memory-feasible — the crate
//! default stays faithful to the checkpoint (global). The `frame_seq_length` / `seq_length` are
//! **derived from the requested resolution** (`resolve_request_config`); the canonical `1560` /
//! `32760` are only the reference resolution's values.
//!
//! The **first-frame VAE re-anchor** (`release_server.py::get_clean_context_frames`, re-encoding the
//! first *decoded* output frame as a persistent clean-context anchor) is a **streaming-only** coherence
//! mechanism and does **not** apply to this batch whole-clip path — see [`generate_t2v_from_components`]
//! for the reasoning. It is deferred to the streaming epic, not silently dropped.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, GenerationOutput, Progress, Result};
use mlx_gen_wan::{decode_to_frames, frames_to_images, load_tokenizer, Umt5Encoder, WanVae};
use mlx_rs::Array;

use crate::causal::CausalKreaTransformer;
use crate::config::KreaRealtimeConfig;
use crate::generate::{generate_latents, ArGenParams};
use crate::load::load_krea_realtime_transformer;

/// z16 Wan VAE temporal compression (a latent frame decodes to `TEMPORAL_STRIDE` output frames).
const TEMPORAL_STRIDE: usize = 4;
/// z16 Wan VAE spatial stride (latent → pixel; 8× per side). Mirrors `WanModelConfig::vae_stride.1/.2`.
const SPATIAL_STRIDE: usize = 8;
/// VAE-decode temporal-window budget (output pixel·frames per `decode_tiled` window) — mirrors the
/// scail2 z16 decode bound so a large batch decode stays memory-bounded (fewer frames/window as the
/// resolution grows). Only consulted when the clip is large enough to tile.
const DECODE_TILE_BUDGET_PXFRAMES: i64 = 3_500_000;

/// A fully-specified Krea Realtime text-to-video job (the engine-internal form
/// [`crate::pipeline`] maps a `GenerationRequest` onto). Dimensions are **pixel** dimensions; the
/// pipeline derives the latent geometry.
pub struct KreaRealtimeJob<'a> {
    pub prompt: &'a str,
    /// Output width in pixels (multiple of `patch·vae_stride` = 16).
    pub width: u32,
    /// Output height in pixels (multiple of `patch·vae_stride` = 16).
    pub height: u32,
    /// Requested **output** frame count (≥ 1). The latent frame count is
    /// `(num_frames − 1)/4 + 1` (the z16 VAE's 4× temporal compression).
    pub num_frames: u32,
    /// Output/playback cadence, carried onto the assembled clip.
    pub fps: u32,
    pub seed: u64,
    /// Few-step denoise-count override (`None` = the config's `denoising_step_list`).
    pub steps: Option<usize>,
}

/// The **Mac memory-feasible** AR config: bound the KV read/store window to the streaming frame count
/// ([`KreaArConfig::streaming_local_attn_frames`](crate::KreaArConfig::streaming_local_attn_frames) =
/// `kv_cache_num_frames + num_frames_per_block`) instead of the shipped global `local_attn_size = -1`
/// (the checkpoint's ~27 GB-of-KV global window). The crate/default config stays faithful to the
/// checkpoint (global); this Mac bound is selected **only** in the pipeline path (sc-8438 S5 follow-up).
pub fn mac_ar_config(base: &KreaRealtimeConfig) -> KreaRealtimeConfig {
    let mut cfg = base.clone();
    cfg.ar.local_attn_size = cfg.ar.streaming_local_attn_frames() as i64;
    cfg
}

/// Latent frame count for `num_frames` **output** frames at the z16 VAE's 4× temporal compression
/// (`(frames − 1)/4 + 1`, the reference latent convention; decode returns `4·T_lat` frames).
fn latent_frame_count(num_frames: u32) -> Result<usize> {
    let f = num_frames as usize;
    let fm1 = f
        .checked_sub(1)
        .ok_or_else(|| Error::Msg("krea t2v: num_frames must be >= 1".into()))?;
    Ok(fm1 / TEMPORAL_STRIDE + 1)
}

/// Build the per-request AR config: the [`mac_ar_config`] streaming-window bound **plus** the
/// resolution-derived `frame_seq_length` / `seq_length`. The canonical `1560` / `32760` are only the
/// *reference resolution's* per-frame token count and total; a different requested resolution yields a
/// different per-frame token count, which the S3 causal forward bakes into its cache windowing + RoPE
/// frame offset (so it must reflect the actual latent geometry, not the baked constant).
fn resolve_request_config(
    base: &KreaRealtimeConfig,
    latent_h: usize,
    latent_w: usize,
    num_latent_frames: usize,
) -> Result<KreaRealtimeConfig> {
    let mut cfg = mac_ar_config(base);
    let (ph, pw) = (cfg.wan.patch_size.1, cfg.wan.patch_size.2);
    if ph == 0 || pw == 0 || !latent_h.is_multiple_of(ph) || !latent_w.is_multiple_of(pw) {
        return Err(Error::Msg(format!(
            "krea t2v: latent {latent_h}x{latent_w} is not divisible by the patch size {ph}x{pw}"
        )));
    }
    let frame_seq_length = (latent_h / ph) * (latent_w / pw);
    if frame_seq_length == 0 {
        return Err(Error::Msg(
            "krea t2v: resolution yields zero tokens per frame".into(),
        ));
    }
    cfg.ar.frame_seq_length = frame_seq_length;
    cfg.ar.seq_length = num_latent_frames * frame_seq_length;
    Ok(cfg)
}

/// A memory-bounded temporal-only VAE-decode tiling for a latent of `[z, T_lat, lat_h, lat_w]` at
/// output size `(out_h, out_w)`, or `None` when the clip is small enough to decode in one pass. Mirrors
/// the scail2 z16 budget: window size shrinks as the resolution grows so the decode peak stays bounded,
/// and each window is capped by the z16 **write** safety bound ([`VaeTiling::WAN::writable_frame_cap`]).
fn decode_tiling(out_h: usize, out_w: usize, out_frames: i32) -> Option<TilingConfig> {
    let px_per_frame = (out_h as i64) * (out_w as i64);
    let budget_frames = DECODE_TILE_BUDGET_PXFRAMES / px_per_frame.max(1);
    let write_cap = VaeTiling::WAN.writable_frame_cap(out_h as i32, out_w as i32);
    let win = budget_frames.min(write_cap) as i32;
    // Snap to the z16 ×4 temporal stride; ≥ 8 output frames (2 latent frames of temporal-conv context).
    let tile_frames = (win / 4 * 4).clamp(8, out_frames.max(8));
    if tile_frames >= out_frames {
        return None; // one window covers the clip → single-pass decode.
    }
    let overlap = (tile_frames / 4).max(1);
    Some(TilingConfig::temporal_only(tile_frames, overlap))
}

/// Decode the AR latent sequence `[z16, T_lat, lat_h, lat_w]` (f32) through the reused z16 Wan
/// [`WanVae`] → an assembled RGB clip ([`GenerationOutput::Video`]). Single-pass for the batch form; a
/// `tiling` config bounds a large decode via gen-core [`mlx_gen::tiling`]. The z16 VAE upsamples `T_lat
/// → 4·T_lat` temporally and `×8` spatially, so the clip is `4·T_lat` frames of `8·lat_h × 8·lat_w`.
pub fn decode_latents_to_video(
    vae: &WanVae,
    latents: &Array,
    fps: u32,
    tiling: Option<&TilingConfig>,
    cancel: &CancelFlag,
) -> Result<GenerationOutput> {
    // `decode_to_frames` reshapes `[C,F,H,W]` → `[1,C,F,H,W]`, decodes (single-pass or tiled), and
    // returns `[F_out, H_out, W_out, 3]` uint8; `frames_to_images` splits it into one `Image`/frame.
    let frames_u8 = decode_to_frames(vae, latents, tiling, Some(cancel))?;
    let frames = frames_to_images(&frames_u8)?;
    Ok(GenerationOutput::Video {
        frames,
        fps,
        audio: None,
    })
}

/// Component-level text-to-video: given the already-built causal transformer + z16 VAE + UMT5
/// `context` `[text_len, text_dim]` (f32), run the AR few-step denoise
/// ([`generate_latents`]) → VAE decode → assembled clip. This is the **weight-free e2e seam** the S6
/// verification drives on a tiny random-weight config (tiny UMT5 context + tiny DiT + tiny VAE): it
/// exercises `context → latents → VAE decode → video` without the 28 GB checkpoint.
///
/// **First-frame VAE re-anchor (sc-8438 S5 follow-up, evaluated in S6).** The reference
/// `release_server.py::get_clean_context_frames` re-encodes the first *decoded* output frame through
/// the VAE and pins it as a persistent clean-context anchor in the rolling KV cache. That is a
/// **streaming-only** mechanism and does not apply here: (1) the batch path decodes exactly **once**,
/// at the very end — there is no incrementally-decoded first frame to re-encode mid-generation; (2)
/// within the batch generation the always-attended `sink_size` prefix + the S5 clean-context KV
/// recompute already provide the persistent clean anchor; (3) the re-anchor's purpose — re-injecting
/// decoded-pixel-clean context during *unbounded* streaming — has no analogue when the full latent
/// sequence is produced in one pass. It is therefore **deferred to the streaming epic** (not dropped);
/// wiring it into the batch path would add a VAE round-trip with no coherence effect on a single-decode
/// clip. Confidence: moderate — the reference sampling path is unavailable here, but the architectural
/// reasoning (single terminal decode, sink + recompute anchor) is sound.
#[allow(clippy::too_many_arguments)]
pub fn generate_t2v_from_components(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    vae: &WanVae,
    context: &Array,
    params: &ArGenParams,
    tiling: Option<&TilingConfig>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    // AR few-step denoise over the causal transformer (owns a fresh KV cache).
    let latents = generate_latents(transformer, cfg, context, params)?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    on_progress(Progress::Decoding);
    decode_latents_to_video(vae, &latents, params.fps, tiling, cancel)
}

/// Load the reused stock-Wan UMT5-XXL text encoder from the snapshot `root` and encode `prompt` →
/// context `[text_len, text_dim]` (f32). Krea Realtime ships transformer-only, so the tokenizer +
/// `t5_encoder.safetensors` are the stock Wan components (provisioned as caller-local paths). CFG is
/// **off** (Krea Realtime is CFG-off), so only the positive prompt is encoded.
fn encode_prompt(root: &Path, cfg: &KreaRealtimeConfig, prompt: &str) -> Result<Array> {
    let tokenizer = load_tokenizer(root.join("tokenizer.json"), cfg.wan.text_len)?;
    let w = Weights::from_file(root.join("t5_encoder.safetensors"))?;
    let enc = Umt5Encoder::from_weights(&w, &cfg.wan)?;
    let context = enc.encode(&tokenizer, prompt)?;
    mlx_rs::transforms::eval([&context])?;
    Ok(context)
}

/// Load the Krea Realtime transformer weight map from the snapshot `root`: a single-file
/// `dit.safetensors` (converted MLX layout) or a sharded `transformer/` directory. The
/// [`load_krea_realtime_transformer`] path handles either on-disk key layout (via
/// [`crate::convert::sanitize_krea_realtime_transformer`]).
fn load_transformer(root: &Path, cfg: &KreaRealtimeConfig) -> Result<CausalKreaTransformer> {
    let dit_file = root.join("dit.safetensors");
    let transformer_dir = root.join("transformer");
    let weights = if dit_file.exists() {
        Weights::from_file(dit_file)?
    } else if transformer_dir.is_dir() {
        Weights::from_dir(transformer_dir)?
    } else {
        return Err(Error::Msg(format!(
            "krea t2v: no transformer weights in {} (expected dit.safetensors or a transformer/ dir)",
            root.display()
        )));
    };
    let raw: HashMap<String, Array> = weights
        .keys()
        .map(|k| (k.to_string(), weights.get(k).expect("listed key").clone()))
        .collect();
    let dit = load_krea_realtime_transformer(raw, cfg)?;
    Ok(CausalKreaTransformer::new(dit, cfg))
}

/// Run the full Krea Realtime text-to-video generation for `job`, loading each reused component from
/// the snapshot `root` (`dit.safetensors` + `t5_encoder.safetensors` + `vae.safetensors` +
/// `tokenizer.json` — stock Wan for the TE / VAE / tokenizer, Krea's own DiT). This is the **real
/// weight** product path; the tiny-config e2e drives [`generate_t2v_from_components`] instead.
///
/// The pipeline sizes the AR generation to the Mac streaming window ([`mac_ar_config`]) and the
/// requested resolution (`resolve_request_config`), encodes the prompt (UMT5), runs the AR denoise,
/// and VAE-decodes to a clip.
pub fn generate_t2v(
    root: &Path,
    base_cfg: &KreaRealtimeConfig,
    job: &KreaRealtimeJob,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    if !root.exists() {
        return Err(Error::Msg(format!(
            "krea t2v: snapshot dir does not exist: {}",
            root.display()
        )));
    }
    if job.width == 0 || job.height == 0 {
        return Err(Error::Msg("krea t2v: width/height must be > 0".into()));
    }
    let latent_h = job.height as usize / SPATIAL_STRIDE;
    let latent_w = job.width as usize / SPATIAL_STRIDE;
    if latent_h == 0 || latent_w == 0 {
        return Err(Error::Msg(format!(
            "krea t2v: {}x{} is smaller than one {SPATIAL_STRIDE}px VAE cell",
            job.width, job.height
        )));
    }
    let num_latent_frames = latent_frame_count(job.num_frames)?;
    let cfg = resolve_request_config(base_cfg, latent_h, latent_w, num_latent_frames)?;

    // Stage 1: UMT5 text encode (loaded → used → freed inside `encode_prompt`).
    on_progress(Progress::Loading(mlx_gen::LoadPhase::TextEncoder));
    let context = encode_prompt(root, &cfg, job.prompt)?;

    // Stage 2: the Krea DiT (reused Wan 2.1 14B), wrapped causal with the Mac + resolution AR config.
    on_progress(Progress::Loading(mlx_gen::LoadPhase::Renderer));
    let transformer = load_transformer(root, &cfg)?;

    // Stage 3: the reused z16 Wan VAE.
    let vae = {
        let w = Weights::from_file(root.join("vae.safetensors"))?;
        WanVae::from_weights(&w)?
    };

    let params = ArGenParams {
        seed: job.seed,
        steps: job.steps,
        num_latent_frames,
        latent_height: latent_h,
        latent_width: latent_w,
        fps: job.fps,
    };
    let out_frames = (num_latent_frames * TEMPORAL_STRIDE) as i32;
    let tiling = decode_tiling(
        latent_h * SPATIAL_STRIDE,
        latent_w * SPATIAL_STRIDE,
        out_frames,
    );
    generate_t2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        &params,
        tiling.as_ref(),
        cancel,
        on_progress,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_frame_count_uses_causal_convention() {
        // (frames − 1)/4 + 1: 1→1, 4→1, 5→2, 8→2, 9→3, 21 latent for 81 frames.
        assert_eq!(latent_frame_count(1).unwrap(), 1);
        assert_eq!(latent_frame_count(4).unwrap(), 1);
        assert_eq!(latent_frame_count(5).unwrap(), 2);
        assert_eq!(latent_frame_count(8).unwrap(), 2);
        assert_eq!(latent_frame_count(9).unwrap(), 3);
        assert_eq!(latent_frame_count(81).unwrap(), 21);
        assert!(latent_frame_count(0).is_err());
    }

    /// sc-8438 S5 follow-up: the Mac AR config bounds the KV window to the streaming frame count
    /// (`kv_cache_num_frames + num_frames_per_block`), NOT the shipped global `-1`. The default config
    /// stays faithful to the checkpoint (global); only the pipeline path picks the bound.
    #[test]
    fn mac_ar_config_bounds_the_kv_window_to_streaming_frames() {
        let base = KreaRealtimeConfig::krea_realtime_14b();
        // The shipped checkpoint is global.
        assert_eq!(base.ar.local_attn_size, -1);
        assert_eq!(base.ar.max_attention_size(), base.ar.seq_length);

        let mac = mac_ar_config(&base);
        let streaming = base.ar.streaming_local_attn_frames(); // 3 + 3 = 6 frames
        assert_eq!(
            mac.ar.local_attn_size, streaming as i64,
            "Mac path bounds local_attn_size to the streaming frame count"
        );
        // The read window is now the bounded 6·frame_seq_length tokens, an order of magnitude under the
        // global 32760.
        assert_eq!(
            mac.ar.max_attention_size(),
            streaming * mac.ar.frame_seq_length
        );
        assert!(mac.ar.max_attention_size() < base.ar.seq_length);
        // The crate default is untouched — the bound lives in the pipeline, not the config default.
        assert_eq!(KreaRealtimeConfig::default().ar.local_attn_size, -1);
    }

    /// The per-request config derives `frame_seq_length` from the actual latent geometry (not the baked
    /// canonical 1560) and keeps the Mac streaming window bound.
    #[test]
    fn resolve_request_config_derives_frame_seq_length_and_keeps_mac_bound() {
        let base = KreaRealtimeConfig::krea_realtime_14b();
        // 32×32 latent under patch 2×2 → (16·16) = 256 tokens/frame.
        let cfg = resolve_request_config(&base, 32, 32, 4).unwrap();
        assert_eq!(cfg.ar.frame_seq_length, 256);
        assert_eq!(cfg.ar.seq_length, 4 * 256);
        // Mac streaming bound retained; the read window rides the derived frame_seq_length.
        assert_eq!(
            cfg.ar.local_attn_size,
            base.ar.streaming_local_attn_frames() as i64
        );
        assert_eq!(
            cfg.ar.max_attention_size(),
            base.ar.streaming_local_attn_frames() * 256
        );
        // A latent not divisible by the patch size is rejected.
        assert!(resolve_request_config(&base, 33, 32, 4).is_err());
    }

    #[test]
    fn decode_tiling_is_single_pass_for_small_clips_and_tiles_large_ones() {
        // A short clip fits one window ⇒ None (single pass).
        assert!(decode_tiling(256, 256, 8).is_none());
        // A long, small-frame clip tiles temporally (window < out_frames).
        let big = decode_tiling(256, 256, 400);
        assert!(big.is_some(), "a 400-frame clip must tile");
    }
}
