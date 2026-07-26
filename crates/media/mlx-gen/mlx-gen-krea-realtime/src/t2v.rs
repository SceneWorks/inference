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
//! The reference's **first-frame VAE re-anchor** (`release_server.py::get_clean_context_frames`,
//! re-encoding the first *decoded* output frame as a persistent clean-context anchor) re-encodes decoded
//! pixels *mid-generation*, so that specific **mechanism** is streaming-coupled and correctly out of this
//! single-terminal-decode batch path. It does **not** follow that a long batch clip is anchored: the Mac
//! path runs the bounded ~6-frame window ([`mac_ar_config`]) for *every* clip and the shipped 14B config
//! sets `sink_size = 0`, so a long clip slides its window with **no** persistent anchor — a real
//! long-range coherence risk **tracked as sc-15127 (S18)**, to be measured/addressed on the gated
//! real-weight run. See [`generate_t2v_from_components`] for the full reasoning.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterSpec, CancelFlag, Error, GenerationOutput, Image, Progress, Result};
use mlx_gen_wan::{
    decode_to_frames, frames_to_images, load_tokenizer, preprocess_i2v_image, Umt5Encoder, WanVae,
};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{random, Array};

use crate::causal::CausalKreaTransformer;
use crate::config::{KreaRealtimeConfig, MODEL_ID};
use crate::generate::{generate_i2v_latents, generate_latents, generate_v2v_latents, ArGenParams};
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
/// → 4·T_lat` temporally and `×8` spatially, so the raw decode is `4·T_lat` frames of `8·lat_h ×
/// 8·lat_w`.
///
/// `out_frames` is the **requested output frame count**. Because the latent count is derived from the
/// requested output count via the causal convention (`T_lat = (frames − 1)/4 + 1`) while the decode is
/// the non-causal `4·T_lat`, a request whose output count is not ≡ 1 (mod 4) over-delivers up to 3 extra
/// **leading** frames (e.g. 81 requested → 21 latent → 84 decoded). When `out_frames` is `Some(n)` and
/// the decode over-delivers, the leading excess is trimmed so the returned clip is exactly `n` frames —
/// mirroring the sibling z16 Wan path's `images.drain(0..trim)` (`mlx-gen-wan::model`). `None` (or a
/// request ≥ the decoded count) returns the full `4·T_lat` decode untrimmed.
pub fn decode_latents_to_video(
    vae: &WanVae,
    latents: &Array,
    fps: u32,
    out_frames: Option<usize>,
    tiling: Option<&TilingConfig>,
    cancel: &CancelFlag,
) -> Result<GenerationOutput> {
    // `decode_to_frames` reshapes `[C,F,H,W]` → `[1,C,F,H,W]`, decodes (single-pass or tiled), and
    // returns `[F_out, H_out, W_out, 3]` uint8; `frames_to_images` splits it into one `Image`/frame.
    let frames_u8 = decode_to_frames(vae, latents, tiling, Some(cancel))?;
    let mut frames = frames_to_images(&frames_u8)?;
    // Trim the leading over-delivery so a batch product returns exactly the requested count (never
    // invents frames when the request is ≥ what was decoded).
    if let Some(requested) = out_frames {
        if requested < frames.len() {
            let excess = frames.len() - requested;
            frames.drain(0..excess);
        }
    }
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
/// **First-frame VAE re-anchor (sc-8438 S5 follow-up, evaluated in S6; long-clip coherence tracked as
/// sc-15127 / S18).** The reference `release_server.py::get_clean_context_frames` re-encodes the first
/// *decoded* output frame through the VAE and pins it as a persistent clean-context anchor in the
/// rolling KV cache. That specific **mechanism** is genuinely streaming-coupled and is correctly out of
/// this batch path: it re-encodes decoded pixels *mid-generation*, but the batch form decodes exactly
/// **once**, at the very end — there is no incrementally-decoded first frame to re-encode while the
/// latents are still being produced. Wiring the pixel re-encode into the batch path would add a VAE
/// round-trip with nothing to anchor against on a single terminal decode.
///
/// **This does *not* mean a long batch clip is anchored.** The Mac path runs the bounded ~6-frame
/// streaming window ([`mac_ar_config`]) for *every* clip, and the shipped 14B config sets
/// `sink_size = 0` (`config.rs`, asserted there), so the always-attended sink prefix is **empty** and
/// anchors nothing. A long batch clip therefore slides that window with **no persistent clean anchor** —
/// a real long-range coherence risk. The reference deliberately *pairs* the bounded window with the
/// first-frame re-anchor to hold such an anchor; the batch path currently has neither the re-anchor nor a
/// non-zero sink. That gap is **tracked as sc-15127 (S18)**: it must be measured — and a batch-compatible
/// anchor chosen (a non-zero `sink_size` latent-space pin is the likely fix, needing no incremental
/// decode) — on the gated real-weight run (S6(b) / S13). It is not addressable on the tiny random-weight
/// fixtures here. Confidence: high that `sink_size = 0` leaves the sliding window unanchored; the
/// coherence *impact* on a long clip is unmeasured until real weights.
#[allow(clippy::too_many_arguments)]
pub fn generate_t2v_from_components(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    vae: &WanVae,
    context: &Array,
    params: &ArGenParams,
    out_frames: Option<usize>,
    tiling: Option<&TilingConfig>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    // AR few-step denoise over the causal transformer (owns a fresh KV cache). The `cancel` +
    // `on_progress` are threaded INTO the loop (sc-8441 S8): it polls the flag per AR step and emits a
    // `Progress::Step` per denoise step, so a mid-clip cancel bails within ~one step (not just here at
    // the stage boundary).
    let latents = generate_latents(transformer, cfg, context, params, cancel, on_progress)?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    on_progress(Progress::Decoding);
    // `out_frames` trims the z16 decode's leading over-delivery back to the requested output count.
    decode_latents_to_video(vae, &latents, params.fps, out_frames, tiling, cancel)
}

/// VAE-encode a reference **still** → clean i2v context latent `[in_dim, 1, latent_h, latent_w]` (f32)
/// via the reused z16 Wan VAE [`WanVae::encode`] (`.mode()`, the Gaussian mean — the reference's image
/// conditioning path, mirroring `mlx-gen-bernini`'s `vae_encode_image`). The image is cover-fit +
/// center-cropped to `(width, height)` and normalized to `[-1, 1]` by the reused
/// [`preprocess_i2v_image`], then encoded and the batch axis dropped. The pixel size `(width, height)`
/// must be `latent · 8`, so the encoded latent matches the AR latent geometry.
fn encode_reference_image(vae: &WanVae, image: &Image, width: u32, height: u32) -> Result<Array> {
    let chw = preprocess_i2v_image(image, width, height)?; // [3, H, W] in [-1, 1]
    let video = chw.expand_dims(1)?.expand_dims(0)?; // [1, 3, 1, H, W]
    let z = vae.encode(&video)?; // [1, z, 1, h8, w8]
    let s = z.shape();
    Ok(z.reshape(&[s[1], s[2], s[3], s[4]])?) // drop the batch axis → [z, 1, h8, w8]
}

/// VAE-encode a **source clip** → clean v2v source latent `[in_dim, T_lat, latent_h, latent_w]` (f32)
/// via the reused z16 Wan VAE [`WanVae::encode_sample`] (`.sample()` — the reference's **video** source
/// path, mirroring `mlx-gen-bernini`'s `vae_encode_video`; `eps` is drawn from `key` so the encode is
/// deterministic given the seed). Each frame is cover-fit + center-cropped to `(width, height)`,
/// stacked on the temporal axis (`T = 1 + 4·k`), encoded, and the batch axis dropped. `T_lat =
/// (T − 1)/4 + 1`.
fn encode_source_clip(
    vae: &WanVae,
    cfg: &KreaRealtimeConfig,
    frames: &[Image],
    width: u32,
    height: u32,
    key: &Array,
) -> Result<Array> {
    if frames.is_empty() {
        return Err(Error::Msg("krea v2v: source clip has no frames".into()));
    }
    let mut chw_t = Vec::with_capacity(frames.len());
    for f in frames {
        chw_t.push(preprocess_i2v_image(f, width, height)?.expand_dims(1)?); // [3, 1, H, W]
    }
    let refs: Vec<&Array> = chw_t.iter().collect();
    let video = concatenate_axis(&refs, 1)?.expand_dims(0)?; // [1, 3, T, H, W]
    let s = video.shape();
    let (t, h, w) = (s[2], s[3], s[4]);
    let t_lat = (t - 1) / 4 + 1; // z16 temporal stride 4
    let z_dim = cfg.wan.vae_z_dim as i32;
    let eps = random::normal::<f32>(&[1, z_dim, t_lat, h / 8, w / 8], None, None, Some(key))?;
    let z = vae.encode_sample(&video, &eps)?; // [1, z, T_lat, h8, w8]
    let s = z.shape();
    Ok(z.reshape(&[s[1], s[2], s[3], s[4]])?) // drop the batch axis → [z, T_lat, h8, w8]
}

/// Component-level **image-to-video** (sc-8440 S7): given the built causal transformer + z16 VAE + UMT5
/// `context`, VAE-encode the reference still, warm the KV cache from it, generate the continuation
/// ([`generate_i2v_latents`]), and VAE-decode → assembled clip. The **weight-free e2e seam** the S7
/// verification drives on a tiny random-weight config (mirrors [`generate_t2v_from_components`]). The
/// returned clip is the reference frame(s) followed by the generated continuation (`F_ref +
/// num_latent_frames` latent frames → `4·(F_ref + num_latent_frames)` output frames), trimmed by
/// `out_frames`. Mirrors the reference `causal_inference.py` (`initial_latent`, image i2v).
#[allow(clippy::too_many_arguments)]
pub fn generate_i2v_from_components(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    vae: &WanVae,
    context: &Array,
    reference_image: &Image,
    params: &ArGenParams,
    out_frames: Option<usize>,
    tiling: Option<&TilingConfig>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let width = (params.latent_width * SPATIAL_STRIDE) as u32;
    let height = (params.latent_height * SPATIAL_STRIDE) as u32;
    // Stage 1: VAE-encode the reference still → clean context latent.
    let reference_latents = encode_reference_image(vae, reference_image, width, height)?;
    mlx_rs::transforms::eval([&reference_latents])?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    // Stage 2: warm the cache from the reference + AR-generate the continuation conditioned on it. The
    // `cancel` + `on_progress` thread INTO the loop (sc-8441 S8) — per-step cancel poll + per-step
    // sampling progress across the generated chunks.
    let latents = generate_i2v_latents(
        transformer,
        cfg,
        context,
        params,
        &reference_latents,
        cancel,
        on_progress,
    )?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    on_progress(Progress::Decoding);
    decode_latents_to_video(vae, &latents, params.fps, out_frames, tiling, cancel)
}

/// Component-level **video-to-video** (sc-8440 S7): VAE-encode the source clip (`.sample()`), generate
/// `params.num_latent_frames` frames conditioned on it at the given denoise `strength`
/// ([`generate_v2v_latents`]), and VAE-decode → assembled clip. The **weight-free e2e seam** the S7
/// verification drives on a tiny random-weight config. The source clip must have exactly
/// `4·(num_latent_frames − 1) + 1` frames (so its encode yields `num_latent_frames` latent frames).
/// A lower `strength` preserves more of the source; `strength = 1` fully regenerates. Mirrors the
/// reference `v2v.py` + `release_server.py`.
#[allow(clippy::too_many_arguments)]
pub fn generate_v2v_from_components(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    vae: &WanVae,
    context: &Array,
    source_frames: &[Image],
    strength: f32,
    params: &ArGenParams,
    out_frames: Option<usize>,
    tiling: Option<&TilingConfig>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let width = (params.latent_width * SPATIAL_STRIDE) as u32;
    let height = (params.latent_height * SPATIAL_STRIDE) as u32;
    // Stage 1: VAE-encode the source clip → clean source latent (deterministic eps from the seed).
    let key = random::key(params.seed)?;
    let source_latents = encode_source_clip(vae, cfg, source_frames, width, height, &key)?;
    mlx_rs::transforms::eval([&source_latents])?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    // Stage 2: strength-controlled AR generation from the renoised source. The `cancel` +
    // `on_progress` thread INTO the loop (sc-8441 S8) — per-step cancel poll + per-step progress.
    let latents = generate_v2v_latents(
        transformer,
        cfg,
        context,
        params,
        &source_latents,
        strength,
        cancel,
        on_progress,
    )?;
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    on_progress(Progress::Decoding);
    decode_latents_to_video(vae, &latents, params.fps, out_frames, tiling, cancel)
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
///
/// Any inference LoRA(s) in `adapters` (sc-15015, S14) are installed onto the built DiT as forward-time
/// residuals over the dense bf16 base via the family-agnostic strict installer — the `mlx-gen-scail2`
/// dense-Wan template. Krea Realtime is Wan-2.1-14B T2V weight-for-weight, so a diffusers / PEFT /
/// kohya / LoKr file resolves against the DiT's module names (the FFN `ffn.0`/`ffn.2` reference keys
/// normalized to the converted `ffn.fc1`/`fc2` by [`CausalKreaTransformer`]'s adaptable host); the
/// installer **errors — never silently drops** — on a format/prefix mismatch or an unmatched target.
/// The quant/packed-tier (Q4/Q8) LoRA install is the separate quant-tier story (S19).
fn load_transformer(
    root: &Path,
    cfg: &KreaRealtimeConfig,
    adapters: &[AdapterSpec],
) -> Result<CausalKreaTransformer> {
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
    let mut transformer = CausalKreaTransformer::new(dit, cfg);
    if !adapters.is_empty() {
        let report =
            mlx_gen::adapters::loader::apply_adapters_strict(&mut transformer, adapters, MODEL_ID)?;
        // Surface the applied count (unmatched targets already errored above, never silently dropped).
        eprintln!(
            "{MODEL_ID}: installed {} LoRA target residual(s) from {} adapter file(s)",
            report.applied,
            adapters.len()
        );
    }
    Ok(transformer)
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
    adapters: &[AdapterSpec],
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

    // Stage the reused components (UMT5 prompt encode + Krea DiT + z16 Wan VAE); any inference LoRA(s)
    // are installed onto the DiT inside `stage_components` (sc-15015, S14).
    let (context, transformer, vae) =
        stage_components(root, &cfg, job.prompt, adapters, on_progress)?;

    let params = ArGenParams {
        seed: job.seed,
        steps: job.steps,
        num_latent_frames,
        latent_height: latent_h,
        latent_width: latent_w,
        fps: job.fps,
    };
    let decoded_frames = (num_latent_frames * TEMPORAL_STRIDE) as i32;
    let tiling = decode_tiling(
        latent_h * SPATIAL_STRIDE,
        latent_w * SPATIAL_STRIDE,
        decoded_frames,
    );
    // Trim the z16 decode back to the exact requested output count (`(f−1)/4+1` latent → `4·T_lat`
    // decoded over-delivers up to 3 leading frames when `num_frames` is not ≡ 1 (mod 4)).
    generate_t2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        &params,
        Some(job.num_frames as usize),
        tiling.as_ref(),
        cancel,
        on_progress,
    )
}

/// Shared real-weight component staging for the t2v/i2v/v2v pipeline paths: UMT5 prompt encode
/// (loaded → used → freed) + the Krea DiT (reused Wan 2.1 14B) + the reused z16 Wan VAE, each from the
/// snapshot `root`.
fn stage_components(
    root: &Path,
    cfg: &KreaRealtimeConfig,
    prompt: &str,
    adapters: &[AdapterSpec],
    on_progress: &mut dyn FnMut(Progress),
) -> Result<(Array, CausalKreaTransformer, WanVae)> {
    on_progress(Progress::Loading(mlx_gen::LoadPhase::TextEncoder));
    let context = encode_prompt(root, cfg, prompt)?;
    on_progress(Progress::Loading(mlx_gen::LoadPhase::Renderer));
    let transformer = load_transformer(root, cfg, adapters)?;
    let w = Weights::from_file(root.join("vae.safetensors"))?;
    let vae = WanVae::from_weights(&w)?;
    Ok((context, transformer, vae))
}

/// Validate the snapshot `root` + pixel size and derive the latent geometry `(latent_h, latent_w)`
/// shared by the i2v/v2v pipeline paths (mirrors `generate_t2v`'s guards).
fn resolve_latent_size(root: &Path, job: &KreaRealtimeJob, what: &str) -> Result<(usize, usize)> {
    if !root.exists() {
        return Err(Error::Msg(format!(
            "krea {what}: snapshot dir does not exist: {}",
            root.display()
        )));
    }
    if job.width == 0 || job.height == 0 {
        return Err(Error::Msg(format!("krea {what}: width/height must be > 0")));
    }
    let latent_h = job.height as usize / SPATIAL_STRIDE;
    let latent_w = job.width as usize / SPATIAL_STRIDE;
    if latent_h == 0 || latent_w == 0 {
        return Err(Error::Msg(format!(
            "krea {what}: {}x{} is smaller than one {SPATIAL_STRIDE}px VAE cell",
            job.width, job.height
        )));
    }
    Ok((latent_h, latent_w))
}

/// Run the full Krea Realtime **image-to-video** generation for `job` + a `reference_image`, loading
/// each reused component from the snapshot `root` (same layout as [`generate_t2v`]). The reference still
/// is VAE-encoded into the first latent frame and warms the AR KV cache; the pipeline generates the
/// remaining `num_latent_frames − 1` frames so the assembled clip is `job.num_frames`. The tiny-config
/// e2e drives [`generate_i2v_from_components`] instead. Mirrors the reference `causal_inference.py`
/// (`initial_latent`, `num_input_frames == 1`).
///
/// **Anchor count (S13 coherence lever).** This warms **one** clean-context frame (the still).
/// `release_server.py::setup_start_frame` instead repeats the still to `kv_cache_num_frames` (= 3)
/// latent frames so the rolling window is fully seeded and the generation blocks stay frame-block
/// aligned (`current_start_frame = kv_cache_num_frames`, a multiple of `num_frame_per_block`). The
/// [`generate_i2v_latents`] engine seam already accepts a multi-frame reference, so the anchor count is
/// a coherence choice to measure/tune on the gated real-weight run (S13), not an engine limitation.
/// Real-weight watchable-clip coherence is gated to S13.
pub fn generate_i2v(
    root: &Path,
    base_cfg: &KreaRealtimeConfig,
    job: &KreaRealtimeJob,
    reference_image: &Image,
    adapters: &[AdapterSpec],
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    let (latent_h, latent_w) = resolve_latent_size(root, job, "i2v")?;
    let total_latent = latent_frame_count(job.num_frames)?;
    // The reference still is latent frame 0 (one clean context frame); generate the rest.
    const F_REF: usize = 1;
    let num_generate = total_latent
        .checked_sub(F_REF)
        .filter(|&n| n >= 1)
        .ok_or_else(|| {
            Error::Msg(
                "krea i2v: num_frames too small — need at least 2 latent frames (a reference \
                 frame + a generated frame)"
                    .into(),
            )
        })?;
    let cfg = resolve_request_config(base_cfg, latent_h, latent_w, total_latent)?;
    let (context, transformer, vae) =
        stage_components(root, &cfg, job.prompt, adapters, on_progress)?;
    let params = ArGenParams {
        seed: job.seed,
        steps: job.steps,
        num_latent_frames: num_generate,
        latent_height: latent_h,
        latent_width: latent_w,
        fps: job.fps,
    };
    let decoded_frames = ((F_REF + num_generate) * TEMPORAL_STRIDE) as i32;
    let tiling = decode_tiling(
        latent_h * SPATIAL_STRIDE,
        latent_w * SPATIAL_STRIDE,
        decoded_frames,
    );
    generate_i2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        reference_image,
        &params,
        Some(job.num_frames as usize),
        tiling.as_ref(),
        cancel,
        on_progress,
    )
}

/// Run the full Krea Realtime **video-to-video** generation for `job` + a `source_frames` clip at a
/// denoise `strength`, loading each reused component from the snapshot `root` (same layout as
/// [`generate_t2v`]). The source clip is VAE-encoded and drives the strength-controlled AR init; the
/// number of generated latent frames is derived from the source length (`(frames − 1)/4 + 1`). The
/// tiny-config e2e drives [`generate_v2v_from_components`] instead. Mirrors the reference `v2v.py` +
/// `release_server.py`. Real-weight watchable-clip coherence is gated to S13.
#[allow(clippy::too_many_arguments)]
pub fn generate_v2v(
    root: &Path,
    base_cfg: &KreaRealtimeConfig,
    job: &KreaRealtimeJob,
    source_frames: &[Image],
    strength: f32,
    adapters: &[AdapterSpec],
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    let (latent_h, latent_w) = resolve_latent_size(root, job, "v2v")?;
    if source_frames.is_empty() {
        return Err(Error::Msg("krea v2v: source clip has no frames".into()));
    }
    // The generated latent-frame count is derived from the source clip length.
    let num_latent = latent_frame_count(source_frames.len() as u32)?;
    let cfg = resolve_request_config(base_cfg, latent_h, latent_w, num_latent)?;
    let (context, transformer, vae) =
        stage_components(root, &cfg, job.prompt, adapters, on_progress)?;
    let params = ArGenParams {
        seed: job.seed,
        steps: job.steps,
        num_latent_frames: num_latent,
        latent_height: latent_h,
        latent_width: latent_w,
        fps: job.fps,
    };
    let decoded_frames = (num_latent * TEMPORAL_STRIDE) as i32;
    let tiling = decode_tiling(
        latent_h * SPATIAL_STRIDE,
        latent_w * SPATIAL_STRIDE,
        decoded_frames,
    );
    generate_v2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        source_frames,
        strength,
        &params,
        Some(job.num_frames as usize),
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
