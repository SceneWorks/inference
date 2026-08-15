//! Krea Realtime 14B **text-to-video pipeline** (sc-8439, S6): the end-to-end orchestration that
//! turns a prompt into an assembled RGB clip —
//! `prompt → UMT5 encode → context → AR few-step latents ([`generate_latents`]) → z16 Wan VAE decode
//! → [`GenerationOutput::Video`]`.
//!
//! Every heavy component is **reused from [`mlx_gen_wan`]** (Krea Realtime is Wan 2.1 T2V 14B
//! weight-for-weight): the UMT5-XXL text encoder ([`Umt5Encoder`] + [`load_tokenizer`]) and the z16
//! [`ProviderVae`] (encode/decode + `decode_to_frames`/`frames_to_images`). The only Krea-specific piece in
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
//! single-terminal-decode batch path. A long batch clip is therefore genuinely unanchored: the Mac path
//! runs the bounded ~6-frame window ([`mac_ar_config`]) for *every* clip and the shipped 14B config sets
//! `sink_size = 0`.
//!
//! **sc-15127 (S18) measured that on real weights (q4, 640×384 and 832×480, 45 latent frames = 13
//! window rolls, three seeds per configuration) and found one thing and one open question.** A long
//! clip *does* drift, well past the measurement's budget. The headline mode is a
//! **colour-cast/tone/structure** drift — the blue–yellow opponent axis (`opp-B-Y`) wins every row-A
//! cell at 832×480 — alongside which a saturation rise is separately observable. **Whether the bounded
//! window causes it is unresolved in both buckets**: the enlarged within-regime A/D/F dose ladder spans
//! 13/10/5 eviction rolls, but its matched-seed drift slope remains inside the predeclared 2·SEM
//! heuristic in both buckets (+0.571 ±1.897/255 per roll at 640×384 and −0.278 ±1.678 at 832×480).
//! Across the full eight-roll span, effects below practical floors of **19.75/255** and **15.65/255**
//! respectively remain unresolved. So **no sink anchor is wired**: the window comparison and the sink
//! comparison are *both* unresolved at three seeds, and permanently-resident KV is not bought on an
//! unresolved comparison. The `sink_size` knob stays plumbed for a checkpoint that ships one, and the
//! drift itself is tracked as **sc-15571**. See [`generate_t2v_from_components`] for the table, the
//! controls, and the explicit limits of the claim.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::tiling::TilingConfig;
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterApplyReport, AdapterSpec, CancelFlag, Error, GenerationOutput, Image, Progress, Quant,
    Result,
};
use mlx_gen_wan::config::WanQuant;
use mlx_gen_wan::model::effective_te_quant;
use mlx_gen_wan::pipeline::auto_tiling_budgeted_z16_quality_overlap;
use mlx_gen_wan::{
    decode_to_frames, frames_to_images, load_tokenizer, preprocess_i2v_image, Umt5Encoder,
};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{random, Array};

use crate::causal::CausalKreaTransformer;
use crate::config::{KreaRealtimeConfig, MODEL_ID};
use crate::generate::{generate_i2v_latents, generate_latents, generate_v2v_latents, ArGenParams};
use crate::load::{
    load_krea_realtime_transformer_with_quant, probe_packed_quant, resolve_load_time_quant,
};
use crate::{ProviderVae, VAE_TILING};

/// z16 Wan VAE temporal compression (a latent frame decodes to `TEMPORAL_STRIDE` output frames).
const TEMPORAL_STRIDE: usize = VAE_TILING.temporal_scale as usize;
/// Model-local ceiling for the full latent clip allocated by the AR loop. The z16 temporal mapping
/// means this accepts at most 1,028 source/output frames (`ceil(frames / 4) == 257`).
pub(crate) const MAX_LATENT_FRAMES: usize = 257;
/// z16 Wan VAE spatial stride (latent → pixel; 8× per side). Mirrors `WanModelConfig::vae_stride.1/.2`.
const SPATIAL_STRIDE: usize = VAE_TILING.spatial_scale as usize;

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
///
/// **`sink_size` is deliberately left at the checkpoint's `0` (sc-15127, S18)** — because the gated
/// real-weight sweep did not produce evidence for one, **not** because it showed the bounded window to
/// be clean. It showed the opposite: long clips drift. What it also showed is that the drift does not
/// track the number of window rolls, so a first-chunk sink — whose entire rationale is surviving
/// eviction — is not what the evidence points at, and it is permanently-resident KV (measured +0.83 GiB
/// for one latent frame, +2.20 GiB for three, at 640×384). Full table, controls and limits on
/// [`generate_t2v_from_components`]. The 27 GB figure is not rhetorical — the global window at
/// 45 latent frames × 832×480 was measured to get SIGKILLed on a 128 GiB host.
pub fn mac_ar_config(base: &KreaRealtimeConfig) -> KreaRealtimeConfig {
    let mut cfg = base.clone();
    cfg.ar.local_attn_size = cfg.ar.streaming_local_attn_frames() as i64;
    // sink_size intentionally untouched — see the doc comment (sc-15127).
    cfg
}

/// Latent frame count for `num_frames` **output** frames at the z16 VAE's 4× temporal compression
/// (`(frames − 1)/4 + 1`, the reference latent convention; decode returns `4·T_lat` frames).
pub(crate) fn latent_frame_count(num_frames: usize) -> Result<usize> {
    let fm1 = num_frames
        .checked_sub(1)
        .ok_or_else(|| Error::Msg("krea realtime: frame count must be >= 1".into()))?;
    Ok(fm1 / TEMPORAL_STRIDE + 1)
}

/// Resolve and enforce the model-local full-clip allocation bound before any component staging.
/// `what` identifies whether the effective generation length came from requested output frames or a
/// V2V source clip while keeping the capability refusal typed.
pub(crate) fn bounded_latent_frame_count(what: &str, num_frames: usize) -> Result<usize> {
    let latent = latent_frame_count(num_frames)?;
    if latent > MAX_LATENT_FRAMES {
        return Err(Error::Unsupported(format!(
            "{MODEL_ID}: {num_frames} {what} frames resolve to {latent} latent frames, exceeding the \
             model maximum of {MAX_LATENT_FRAMES} latent frames (1,028 source/output frames)"
        )));
    }
    Ok(latent)
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
    // A KV cache tier the backbone cannot express is refused HERE (sc-17807), before any component
    // is staged — the alternative is an opaque MLX exception on the first chunk of a clip whose
    // weights are already resident.
    cfg.validate_kv_cache_quant()?;
    Ok(cfg)
}

/// The **memory-budgeted** VAE-decode tiling for a latent of `[z, T_lat, lat_h, lat_w]` at output size
/// `(out_h, out_w)` — `Ok(None)` when a single pass already fits the budget, `Ok(Some(cfg))` for the
/// largest tile that does, and a catchable `Err` when even the smallest tile does not (returned
/// *before* the decode rather than as an OOM kill).
///
/// `#[doc(hidden)] pub` — reachable so a validation harness in `tests/` can decode through the **same**
/// windowing the product path uses (sc-8446, S13; a single-pass control is not comparable to a tiled
/// product decode, and hard-coding the window in a test would silently drift from this policy), but not
/// part of the crate's advertised API. A dedicated `test-support` feature for one function is heavier
/// than the problem.
///
/// ## sc-15325 — this used to compute its own window, and that window CORRUPTED the decode
///
/// The previous policy sized a temporal-only window from a local `DECODE_TILE_BUDGET_PXFRAMES = 3.5e6`
/// output px·frames, bypassing the shared [`budgeted_plan`](mlx_gen::tiling::budgeted_plan) selector
/// that the sibling Wan decodes use. At every shipped bucket ≥ ~233k px/frame that budget yielded an
/// **8-output-frame** window — *two* latent frames — and `TilePlan::plan`'s ÷`temporal_scale` then left
/// an overlap of 1 latent frame (the clamp maximum at a tile of 2). Measured on real Krea latents at
/// 832×480 / 36 output frames, decoding the **same** latent every way against a single-pass reference
/// (`tests/generate_smoke.rs::decode_tiling_sweep_against_single_pass`):
///
/// | tile / overlap (output) | latent tile / overlap | mean abs err /255 | clipping mean / worst | MLX active peak |
/// |---|---|---|---|---|
/// | single-pass | — | 0 (reference) | 0.08% / 0.25% | 85.1 GiB |
/// | 8 / 2 — the original shipped value | 2 / **0** | **18.5** | **9.7% / 26.6%** | 19.8 GiB |
/// | 8 / 4 — the sc-8446 overlap floor | 2 / **1** | 17.1 | 5.2% / 14.7% | 19.8 GiB |
/// | 16 / 4 | 4 / 1 | 7.5 | 1.8% / 12.9% | 38.5 GiB |
/// | 16 / 8 | 4 / 2 | 6.4 | 1.4% / 7.0% | 38.5 GiB |
/// | **32 / 8** | **8 / 2** | **2.5** | **0.08% / 0.25%** | 75.8 GiB |
/// | 32 / 16 | 8 / 4 | 2.0 | 0.14% / 1.1% | 75.8 GiB |
///
/// Not a cosmetic seam: at the original setting one frame in eight blew **26% of its pixels** to
/// near-white with violet/green chroma separation, against 0.25% single-pass. Latent **tile size** is
/// the dominant term (−67…−70% per doubling at matched overlap ratio); overlap is a real but secondary
/// one, and free in peak.
///
/// ## The fix: one policy, in one place, and pay for it spatially
///
/// This function now delegates to [`auto_tiling_budgeted_z16_quality_overlap`] — the same selector,
/// cost model and
/// candidate grid the Wan z16 T2V/I2V/VACE decodes use, which `mlx-gen-scail2` now shares too. Two
/// things follow:
///
///  * **the routing itself is the fix.** The starved window came from *this crate computing its own*,
///    and it now cannot. gen-core's
///    [`MIN_TEMPORAL_TILE_LATENT_FRAMES`](mlx_gen::tiling::MIN_TEMPORAL_TILE_LATENT_FRAMES) floor is a
///    second line — and note it is a **no-op on the z16 grid today**: that grid already bottoms out at
///    a latent-8 tile, and floor-on vs floor-off selects an identical tile in every cell of a
///    6-bucket × 4-frame-count × 5-budget sweep. It exists so a future budget tweak or a new candidate
///    entry cannot re-derive the starved window. Do not undo this delegation on the belief that the
///    floor is what protects the picture — it is prospective insurance; the delegation is the fix;
///  * memory pressure is relieved on the **spatial** axis instead, which is what makes the fix
///    affordable. Measured on the **same real latents** at latent tile 8 / overlap 4, varying only the
///    spatial tile (`tests/generate_smoke.rs::decode_tiling_sweep_against_single_pass`):
///
/// | spatial tile | mean abs err vs single-pass | per-**column** mean abs err | clipping mean / worst | MLX active peak |
/// |---|---|---|---|---|
/// | none (full frame) | 1.954 /255 | 1.629 … 2.440 | 0.14% / 1.05% | 75.77 GiB |
/// | 448 px | 2.021 | 1.696 … 2.505 | 0.14% / 1.04% | 38.57 GiB |
/// | **320 px** | **2.054** | **1.719 … 2.531** | **0.13% / 1.04%** | **20.08 GiB** |
/// | 256 px | 2.075 | 1.735 … 2.579 | 0.13% / 1.04% | 12.99 GiB |
/// | 192 px | 2.121 | 1.769 … 2.630 | 0.13% / 1.02% | 7.49 GiB |
///
/// A 10.1× memory reduction costs 0.17/255. The two axes are simply not comparable — halving the
/// latent *temporal* tile costs 67-70 % more error, shrinking the spatial tile 4.3× costs 8 % —
/// because at ×8 spatial scale even a 192 px tile is 24 latent px wide with an 8-latent-px overlap.
///
/// The per-column column is what rules out a **seam**; a whole-frame mean averages a tile boundary
/// away. It shifts uniformly across the sweep — floor 1.629 → 1.769 as ceiling 2.440 → 2.630 — with no
/// spike at the tile stride, and the 192 px row's worst column is 1.24× its own clip mean. These rows
/// were previously measured on a band-limited synthetic source, which structurally could not have
/// shown either a seam or a starved spatial receptive field; the conclusion held, but the evidence is
/// now real latents.
///
/// **The operating point at 832×480**, with the budget pinned to what the OLD policy actually cost
/// (20.3 GiB): spatial 320/64 + temporal 32/16, i.e. **latent tile 8 / overlap 4** and 40 latent px
/// spatial tiles — **2.05/255 against single-pass at a 20.1 GiB peak**, with clipping (0.13 %/1.04 %)
/// at the single-pass floor. The old window was 17.1/255 with 5.2 % mean / 14.7 % worst-frame clipping
/// at 19.8 GiB (and 24.4/255 with 30.8 % worst-frame clipping at a longer bucket). Single-pass quality
/// for the same memory; the decode's *floor* also drops from ~20 GiB to ~7.5 GiB (the 192 px row), so
/// this lowers the memory bar rather than raising it. At this short a clip the selector in fact
/// returns a **spatial-only** plan (256 px, no temporal tiling at all — 0.31/255), which is the best
/// possible answer for this defect.
///
/// Because the budget is free-aware (`free × 0.85`, `free = MLX limit − resident`, pinnable with
/// `WAN_VAE_BUDGET_GIB`), a large-memory host still gets the fastest plan that fits and a small one
/// tiles further spatially rather than degrading the picture.
///
/// ### Exposure that this closes
///
/// `mlx-gen-scail2` computed the identical window from the same budget and collapsed **harder** —
/// it never received the sc-8446 overlap floor, so its latent overlap was **0**. Measured on its own
/// VAE weights at 832×480: 24.09/255 with 24.1 % mean / **67.6 % worst-frame** clipping, against
/// 0.01 %/0.25 % single-pass. It now calls the same selector (2.21/255, clipping back at the
/// single-pass floor).
///
/// **LTX was measured too and did not reproduce the defect**: at a latent tile of 3 it is 1.73/255
/// with 0.00 % clipping. Take that verdict at its actual width — it is **empirical, at one bucket that
/// never tiles in production, on a source whose amplitudes cannot reach the clip threshold, and the
/// mechanism is unexplained** (the causal-tiling story does not hold: `causal_temporal` is also true
/// for Wan z48, and z16 at matched effective context is still 3.7× worse). See
/// `mlx-gen-ltx/tests/vae_decode_tiling_parity.rs`. The gen-core floor still removes LTX's `(24, 8)`
/// and `(48, 16)` candidates, but as a cheap quality win (6.6× less error for 17 % more peak), not as
/// a defect fix. Wan z16/z48 already bottomed out at latent tile 8 / overlap 2 and are unchanged.
#[doc(hidden)]
pub fn decode_tiling(out_h: usize, out_w: usize, out_frames: i32) -> Result<Option<TilingConfig>> {
    // The z16 VAE this pipeline decodes through is `mlx_gen_wan`'s, so its tiling policy is
    // `mlx_gen_wan`'s too: one selector, one cost model, one set of candidates, shared with the Wan
    // T2V/I2V/VACE decodes and with `mlx-gen-scail2`. The budget is free-aware
    // (`free × 0.85`, `free = MLX limit − resident`) and pinnable via `WAN_VAE_BUDGET_GIB`.
    auto_tiling_budgeted_z16_quality_overlap(out_h as i32, out_w as i32, out_frames)
}

/// Decode the AR latent sequence `[z16, T_lat, lat_h, lat_w]` (f32) through the reused z16 Wan
/// [`ProviderVae`] → an assembled RGB clip ([`GenerationOutput::Video`]). Single-pass for the batch form; a
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
    vae: &ProviderVae,
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
/// **The long batch clip is unanchored, and sc-15127 (S18) measured what that costs.** The Mac path
/// runs the bounded ~6-frame streaming window ([`mac_ar_config`]) for *every* clip and the shipped 14B
/// config sets `sink_size = 0` (`config.rs`, asserted there), so the always-attended sink prefix is
/// empty and a long clip slides its window with no persistent clean anchor.
///
/// The gated real-weight sweep (`tests/generate_smoke.rs::long_clip_coherence_under_the_bounded_window`,
/// q4, 45 latent frames = 180 output frames = **13 window rolls**, **three seeds per configuration**)
/// scores each clip on a descriptor spanning tone, colour and spatial structure, gated on **both** the
/// one-way OLS trend and the plateau excursion, against an **absolute** 8/255 budget. (The budget is
/// absolute because there is no valid within-regime floor to subtract: a bounded window over a long
/// clip evicts *by definition*, so no zero-roll run exists at the shipped window and the shipped clip
/// length. Its two sides are pinned by synthetic gates instead — motion and jitter score 2.81/255, the
/// weakest injected failure shape 11.37.) Drift is the mean over seeds, ± twice the standard error:
///
/// ```text
/// 640x384                        rolls   drift/255   peak GiB  clip%
/// A shipped   (window  6, sink 0)   13   27.51 +-4.72   14.73   2.30
/// B anchored  (window  6, sink 1)   13   19.25 +-9.82   15.56   2.80
/// C anchored  (window  6, sink 3)   13   13.21 +-7.04   16.93   3.15
/// D wider     (window 15, sink 0)   10   30.57 +-3.20   21.64   5.21
/// F wider     (window 30, sink 0)    5   23.67 +-9.82   34.00  10.03
/// E global    (no eviction)          0   34.06 (n=1)    41.90  11.61   <- out of regime, NOT probative
///
/// 832x480 (the crate default, a shipping bucket)
/// A shipped   (window  6, sink 0)   13   39.23 +-7.95   17.62   2.28
/// B anchored  (window  6, sink 1)   13   30.66 +-4.26   18.94   5.45
/// C anchored  (window  6, sink 3)   13   23.43 +-4.90   21.56   3.97
/// D wider     (window 15, sink 0)   10   36.42 +-14.4   29.41   4.67
/// F wider     (window 30, sink 0)    5   40.90 +-17.3   49.14   8.05
/// ```
///
/// Row E is absent at 832×480 by necessity: the global window at 45 latent frames is 70,200 tokens,
/// and at the **800 KiB per DiT token** the bf16 KV actually costs
/// ([`KreaRealtimeConfig::kv_bytes_per_token`](crate::KreaRealtimeConfig::kv_bytes_per_token)) that is
/// **≈ 53.6 GiB of KV** before activations — exactly the problem [`mac_ar_config`] exists to dodge, so
/// it is a finding rather than a harness bug. Even at 640×384 it peaks at 41.90 GiB, enough swap
/// pressure to fill this host's boot volume, which is why it is `n = 1`.
///
/// Two numbers here were corrected in sc-17807 / sc-17324. The KV was quoted at ≈ 38 GiB, from a
/// per-token cost 1.5× too low; and row E was said to **SIGKILL** a 128 GiB host, which is too strong —
/// CI run 30787887176 ran it to completion at 832×480 with a 63.32 GiB MLX peak. The accurate
/// statement is the one `long_clip_coherence_under_the_bounded_window` now carries: row E fits no
/// available host reproducibly (it cleared 128 GiB by ~0.3 GiB once and fits neither bucket on the
/// ~101 GiB `rw-krea` runner), so it is a row with no home rather than an impossible one.
///
/// **Both buckets say the same thing**, which matters because an earlier single-seed version of this
/// measurement had them disagreeing — 832×480 appeared to show a clean sink dose-response that 640×384
/// contradicted. With three seeds and a metric that gates the excursion as well as the trend, the
/// apparent flip is gone: in both buckets the shipped row is far past the budget, and in both the
/// three-dose A/D/F slope is inside the noise.
///
/// **One finding, and one open question.**
///
/// 1. **A long clip does drift**, far past the budget — 27.51/255 against 8. The earlier revision of
///    this doc, which said the answer was "nothing", was wrong. The defect is tracked as **sc-15571**.
///
///    *Corroboration, stated precisely.* `report_artifacts`' mean frame saturation runs 0.21–0.23 →
///    0.31–0.43 over row A (×1.5 to ×1.9) while the 24-frame zero-eviction row Z barely moves
///    (0.225 → 0.237, ×1.05). That is **not** a statistic with "no construction in common" with the
///    drift metric — an earlier wording claimed that and it was false. It is the *same pixel quantity*
///    (per-pixel `max−min` over RGB) under a **different normalisation** (`(max−min)/max` vs `max−min`)
///    and a genuinely **independent aggregation**: head frame vs tail frame, no baseline, no OLS, no
///    z-gate. Independent aggregation is worth something; identical construction it is not.
///
///    And it is a *different channel* from the one that scored row A. The winning descriptor component
///    is now recorded per cell (`S18Cell::component`, gated): at 832×480 all three row-A cells are won
///    by the **blue–yellow opponent axis** (`opp-B-Y`); at 640×384 they are `spatial-sd`, `luma-mean`
///    and `opp-B-Y`. `saturation` wins **no** row-A cell in either bucket. So the earlier
///    "saturation" framing of the mode overstated it: the headline drift is a colour-cast/tone/
///    structure mode, alongside which a saturation rise is separately observable.
/// 2. **Whether the bounded KV window causes it is NOT resolved by this sweep**, in either direction.
///    The enlarged within-regime A/D/F ladder uses windows 6/15/30 at the same 45 latent frames,
///    spanning **13/10/5 eviction rolls**. For each matched seed, OLS fits drift against roll count;
///    across seeds the mean slope is **+0.571 ±1.897/255 per roll** at 640×384 and
///    **−0.278 ±1.678/255 per roll** at 832×480 (mean ± the predeclared 2·SEM heuristic). Both slopes
///    are inside their heuristic, so neither a positive linear dose response nor its absence is
///    established.
///
///    **What the enlarged design can exclude:** across its full eight-roll span, its practical
///    2·SEM magnitude floor is **19.75/255** at 640×384 (72% of shipped row A's drift) and
///    **15.65/255** at 832×480 (40%). Anything smaller remains below the practical floor.
///
///    Row E (the checkpoint's *global* window, zero evictions, 34.06) is **not** evidence here and is
///    no longer cited as such: different attention mask, out of regime, `n = 1`, no variance estimate.
///
/// **A `sink_size` anchor is therefore still NOT wired.** The sink rows do read lower (B 19.25,
/// C 13.21), but the comparison is not resolvable at this sample size: C is 3.30/255 from the repair
/// threshold inside a combined 2·SEM of 11.76 — resolving it would take roughly fifty seeds per
/// configuration, not three. Buying permanently-resident KV (+0.83 GiB for one latent frame, +2.20 GiB
/// for three, measured) on an unresolved comparison is not warranted. The knob remains fully plumbed
/// ([`crate::KreaArConfig::sink_size`] → `sink_tokens()` → `CausalKvCache`, and readable from
/// `config.json`), so a future checkpoint that ships a non-zero sink is honoured without code changes.
/// The reference's *pixel* re-anchor stays out for the structural reason above.
///
/// **Limits of the claim — read these before citing the table.** It covers one prompt, one quantisation
/// tier (q4), three seeds, 45 latent frames, and only the drift modes the descriptor can see (global
/// tone, global colour including the opponent axes, and a 5×5 block-luma spread). It says nothing about
/// semantic or identity drift, texture degradation, or motion quality beyond a freeze check. No row
/// froze (tail motion 3.1–18.8/255 per frame across all 34 recorded cells).
///
/// The seed-to-seed scatter is the same order as the configuration-to-configuration differences, so
/// **the only ranking this sweep supports is row A against the budget.** Every ordering *between* rows
/// in the table — across A/D/F, A vs the sinks, and anything involving row E — is unresolved rather
/// than ranked.
///
/// The budget itself is bracketed by **synthetic** controls only (motion/jitter 2.81, weakest failure
/// shape 11.37). There is no measured same-content floor: row Z, the within-regime zero-eviction row,
/// runs at 29.63/100f over a 12-output-frame post segment against row A's 15.56/100f over 156, so its
/// short-segment slope is *higher* and extrapolating it over-predicts row A's own measured drift by
/// ~1.9×. Row Z cannot be lengthened either — the shipped 6-latent-frame window evicts as soon as a
/// clip passes 6 latent frames. "Past the budget" therefore means past an absolute number pinned by
/// synthetic stimuli, not past a measured baseline of the same content.
///
/// Those figures are 640×384; sc-17324 later measured row Z at the shipping 832×480 bucket for the
/// first time and it replicates — Z 38.29/100f against row A's 18.82, A/Z = 0.49 — so the absent
/// floor is a property of the comparison, not of the smaller bucket.
#[allow(clippy::too_many_arguments)]
pub fn generate_t2v_from_components(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    vae: &ProviderVae,
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
/// via the reused z16 Wan VAE [`ProviderVae::encode`] (`.mode()`, the Gaussian mean — the reference's image
/// conditioning path, mirroring `mlx-gen-bernini`'s `vae_encode_image`). The image is cover-fit +
/// center-cropped to `(width, height)` and normalized to `[-1, 1]` by the reused
/// [`preprocess_i2v_image`], then encoded and the batch axis dropped. The pixel size `(width, height)`
/// must be `latent · 8`, so the encoded latent matches the AR latent geometry.
fn encode_reference_image(
    vae: &ProviderVae,
    image: &Image,
    width: u32,
    height: u32,
) -> Result<Array> {
    let chw = preprocess_i2v_image(image, width, height)?; // [3, H, W] in [-1, 1]
    let video = chw.expand_dims(1)?.expand_dims(0)?; // [1, 3, 1, H, W]
    let z = vae.encode(&video)?; // [1, z, 1, h8, w8]
    let s = z.shape();
    Ok(z.reshape(&[s[1], s[2], s[3], s[4]])?) // drop the batch axis → [z, 1, h8, w8]
}

/// VAE-encode a **source clip** → clean v2v source latent `[in_dim, T_lat, latent_h, latent_w]` (f32)
/// via the reused z16 Wan VAE [`ProviderVae::encode_sample`] (`.sample()` — the reference's **video** source
/// path, mirroring `mlx-gen-bernini`'s `vae_encode_video`; `eps` is drawn from `key` so the encode is
/// deterministic given the seed). Each frame is cover-fit + center-cropped to `(width, height)`,
/// stacked on the temporal axis (`T = 1 + 4·k`), encoded, and the batch axis dropped. `T_lat =
/// (T − 1)/4 + 1`.
fn encode_source_clip(
    vae: &ProviderVae,
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
    vae: &ProviderVae,
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
    vae: &ProviderVae,
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
///
/// `te_quant` is the UMT5 tier for this run (`None` = the dense bf16 encoder). On a quantized DiT tier
/// it is the shared Wan **Q8 floor** ([`mlx_gen_wan::model::effective_te_quant`], sc-12831): Q8 is
/// near-lossless for this drift-sensitive encoder (measured prompt-embedding cosine 0.9998 vs bf16, vs
/// 0.976 at Q4) while roughly halving the encode-phase active peak — so a Q4 *DiT* creative choice is
/// honored without dragging the text encoder into a visible drift (sc-15203).
fn encode_prompt(
    root: &Path,
    cfg: &KreaRealtimeConfig,
    prompt: &str,
    te_quant: Option<WanQuant>,
) -> Result<Array> {
    let tokenizer = load_tokenizer(root.join("tokenizer.json"), cfg.wan.text_len)?;
    let mut w = Weights::from_file(root.join("t5_encoder.safetensors"))?;
    let enc = match te_quant {
        Some(q) => Umt5Encoder::from_weights_quantized(&mut w, &cfg.wan, q)?,
        None => Umt5Encoder::from_weights(&w, &cfg.wan)?,
    };
    let context = enc.encode(&tokenizer, prompt)?;
    mlx_rs::transforms::eval([&context])?;
    Ok(context)
}

/// Open the snapshot's transformer weights: a single-file `dit.safetensors` (the converted MLX layout)
/// or a sharded `transformer/` directory. MLX safetensors loads are **lazy** — this materializes
/// nothing, so the handle can be opened, probed for its packed tier, and dropped for free.
fn open_transformer_weights(root: &Path) -> Result<Weights> {
    let dit_file = root.join(crate::convert::DIT_FILE);
    let transformer_dir = root.join("transformer");
    if dit_file.exists() {
        Weights::from_file(dit_file)
    } else if transformer_dir.is_dir() {
        Weights::from_dir(transformer_dir)
    } else {
        Err(Error::Msg(format!(
            "krea t2v: no transformer weights in {} (expected dit.safetensors or a transformer/ dir)",
            root.display()
        )))
    }
}

/// Load the Krea Realtime transformer weight map from the snapshot `root`: a single-file
/// `dit.safetensors` (converted MLX layout) or a sharded `transformer/` directory. The
/// [`load_krea_realtime_transformer`] path handles either on-disk key layout (via
/// [`crate::convert::sanitize_krea_realtime_transformer`]).
///
/// ## Quant tiers (sc-15203, S19)
///
/// A **pre-quantized (packed Q4/Q8)** snapshot is detected from the weights themselves
/// ([`load::resolve_snapshot_quant`](crate::load::resolve_snapshot_quant)) and built packed directly by
/// the reused Wan loader — no dequant, no load-time re-quantize. A **dense bf16** snapshot honors a
/// caller's `quant` ([`LoadSpec::quantize`](mlx_gen::LoadSpec::quantize)) by packing the DiT in memory
/// after the build, the same `AdaptableLinear::quantize` path the sibling Wan / SCAIL-2 providers use. A
/// request that conflicts with a packed snapshot's own tier is a hard error
/// ([`load::resolve_load_time_quant`](crate::load::resolve_load_time_quant)) rather than a silent
/// downgrade.
///
/// Any inference LoRA(s) in `adapters` (sc-15015, S14) are installed onto the built DiT as **forward-time
/// residuals** via the family-agnostic strict installer. They go on **after** any quantization, so the
/// residual is a dense add over the quantized matmul and the base is never dequantized — which is what
/// makes the adapter path tier-agnostic (the additive-on-packed property epic 10043 / sc-10578
/// established for the Wan family, and the order `mlx-gen-scail2` uses). Krea Realtime is Wan-2.1-14B
/// T2V weight-for-weight, so a diffusers / PEFT / kohya / LoKr file resolves against the DiT's module
/// names (the FFN `ffn.0`/`ffn.2` reference keys normalized to the converted `ffn.fc1`/`fc2` by
/// [`CausalKreaTransformer`]'s adaptable host); the installer **errors — never silently drops** — on a
/// format/prefix mismatch or an unmatched target.
///
/// ## Diff-patch deltas (sc-15326)
///
/// The installer is the **`_with_diff_patch`** variant, so a ComfyUI/lightx2v step-distill or lightning
/// file's `‹module›.diff` / `.diff_b` deltas are folded as well as its low-rank factors. Measured on
/// `lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64` (1459 tensors): 406 low-rank pairs **plus** 447
/// `.diff_b` and 200 `.diff` — 647 keys the plain installer dropped without a word, leaving a render
/// that is changed but not by the whole LoRA.
///
/// All 647 now land, **on every tier**. That is the whole reason this is the `_with_diff_patch` call
/// rather than a warning: the naive switch would have folded only 7 of them at the default Q4 (a
/// weight `.diff` cannot fold into a packed Linear) against 407 at bf16, so the same LoRA would render
/// differently per tier — unacceptable where the quant tier is a creative choice. Both of the channels
/// this file actually uses are **tier-independent in coverage** instead: a `.diff_b` folds into the
/// Linear's **bias**, which a `QuantizedLinear` keeps dense, and the `.diff` deltas all target
/// **norms**, which no tier packs (routed by [`CausalKreaTransformer`]'s `diff_patch_param_mut`).
/// "In coverage" is the exact claim — every key lands on every tier; the bias fold still happens in
/// whatever dtype the base carries, which for this bf16-native DiT is bf16 either way. Anything that
/// still cannot land is returned on `ApplyReport::diff_patch_unapplied` for the provider to put in
/// front of the user, not left in a log line.
fn load_transformer(
    root: &Path,
    cfg: &KreaRealtimeConfig,
    adapters: &[AdapterSpec],
    quant: Option<Quant>,
) -> Result<(CausalKreaTransformer, Vec<AdapterApplyReport>)> {
    let weights = open_transformer_weights(root)?;
    let raw: HashMap<String, Array> = weights
        .keys()
        .map(|k| (k.to_string(), weights.get(k).expect("listed key").clone()))
        .collect();
    let (mut dit, packed) = load_krea_realtime_transformer_with_quant(raw, cfg)?;
    // Reconcile the requested tier against the one actually on disk, then quantize a dense base if the
    // caller asked for it (a no-op over an already-packed base — hence the reconciliation, which turns
    // a width mismatch into a loud error instead of silently serving the stored tier).
    if let Some(q) = resolve_load_time_quant(MODEL_ID, packed, quant)? {
        dit.quantize(q.bits(), None)?;
    }
    let mut transformer = CausalKreaTransformer::new(dit, cfg);
    let adapter_reports = apply_adapters_reported(&mut transformer, adapters)?;
    Ok((transformer, adapter_reports))
}

/// Install the ordered Krea adapter batch and preserve each engine-owned per-file outcome for the
/// provider contract. Public for weight-free integration tests; product callers use
/// [`crate::KreaRealtime`].
#[doc(hidden)]
pub fn apply_adapters_reported(
    transformer: &mut CausalKreaTransformer,
    adapters: &[AdapterSpec],
) -> Result<Vec<AdapterApplyReport>> {
    let reports = mlx_gen::adapters::loader::apply_adapters_strict_with_diff_patch_reported(
        transformer,
        adapters,
        MODEL_ID,
    )?;
    let mut adapter_reports = Vec::with_capacity(adapters.len());
    for (adapter, report) in adapters.iter().zip(reports) {
        eprintln!(
            "{MODEL_ID}: installed {} LoRA target(s) from adapter {}; {} diff-patch delta(s) \
             unapplied",
            report.applied,
            adapter.path.display(),
            report.diff_patch_unapplied.len()
        );
        adapter_reports.push(AdapterApplyReport {
            adapter_path: adapter.path.clone(),
            applied: report.applied,
            skipped: report.diff_patch_unapplied,
        });
    }
    Ok(adapter_reports)
}

/// Run the full Krea Realtime text-to-video generation for `job`, loading each reused component from
/// the snapshot `root` (`dit.safetensors` + `t5_encoder.safetensors` + `vae.safetensors` +
/// `tokenizer.json` — stock Wan for the TE / VAE / tokenizer, Krea's own DiT). This is the **real
/// weight** product path; the tiny-config e2e drives [`generate_t2v_from_components`] instead.
///
/// The pipeline sizes the AR generation to the Mac streaming window ([`mac_ar_config`]) and the
/// requested resolution (`resolve_request_config`), encodes the prompt (UMT5), runs the AR denoise,
/// and VAE-decodes to a clip.
#[allow(clippy::too_many_arguments)]
pub fn generate_t2v(
    root: &Path,
    base_cfg: &KreaRealtimeConfig,
    job: &KreaRealtimeJob,
    adapters: &[AdapterSpec],
    quant: Option<Quant>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    let num_latent = bounded_latent_frame_count("requested output", job.num_frames as usize)?;
    generate_t2v_reported(
        root,
        base_cfg,
        job,
        num_latent,
        adapters,
        quant,
        cancel,
        on_progress,
    )
    .map(|(output, _)| output)
}

/// Provider entrypoint that preserves the public generation output while also returning the actual
/// per-adapter install reports produced during component staging.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_t2v_reported(
    root: &Path,
    base_cfg: &KreaRealtimeConfig,
    job: &KreaRealtimeJob,
    num_latent_frames: usize,
    adapters: &[AdapterSpec],
    quant: Option<Quant>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
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
    let cfg = resolve_request_config(base_cfg, latent_h, latent_w, num_latent_frames)?;

    // Stage the reused components (UMT5 prompt encode + Krea DiT + z16 Wan VAE); any inference LoRA(s)
    // are installed onto the DiT inside `stage_components` (sc-15015, S14).
    let (context, transformer, vae, adapter_reports) =
        stage_components(root, &cfg, job.prompt, adapters, quant, on_progress)?;

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
    )?;
    // Trim the z16 decode back to the exact requested output count (`(f−1)/4+1` latent → `4·T_lat`
    // decoded over-delivers up to 3 leading frames when `num_frames` is not ≡ 1 (mod 4)).
    let output = generate_t2v_from_components(
        &transformer,
        &cfg,
        &vae,
        &context,
        &params,
        Some(job.num_frames as usize),
        tiling.as_ref(),
        cancel,
        on_progress,
    )?;
    Ok((output, adapter_reports))
}

/// Shared real-weight component staging for the t2v/i2v/v2v pipeline paths: UMT5 prompt encode
/// (loaded → used → freed) + the Krea DiT (reused Wan 2.1 14B) + the reused z16 Wan VAE, each from the
/// snapshot `root`.
///
/// The DiT's on-disk tier is probed **before** the text encoder is staged (sc-15203): the UMT5 Q8 floor
/// applies whenever the DiT tier is quantized, but the encoder is loaded first, so the tier has to be
/// known up front. The probe reads only safetensors shape metadata (MLX loads are lazy), so opening and
/// dropping the DiT handle here costs nothing and materializes nothing.
fn stage_components(
    root: &Path,
    cfg: &KreaRealtimeConfig,
    prompt: &str,
    adapters: &[AdapterSpec],
    quant: Option<Quant>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<(
    Array,
    CausalKreaTransformer,
    ProviderVae,
    Vec<AdapterApplyReport>,
)> {
    let te_quant = resolve_te_quant(root, cfg, quant)?;
    on_progress(Progress::Loading(mlx_gen::LoadPhase::TextEncoder));
    let context = encode_prompt(root, cfg, prompt, te_quant)?;
    on_progress(Progress::Loading(mlx_gen::LoadPhase::Renderer));
    let (transformer, adapter_reports) = load_transformer(root, cfg, adapters, quant)?;
    let w = Weights::from_file(root.join("vae.safetensors"))?;
    let vae = ProviderVae::from_weights(&w)?;
    Ok((context, transformer, vae, adapter_reports))
}

/// The UMT5 tier for this run: the shared Wan Q8 floor whenever the DiT is quantized on **either**
/// axis — a pre-quantized snapshot (read back from the packed weights, so a snapshot with no
/// `config.json` is not mistaken for dense) or a load-time `Q4`/`Q8` request over a dense one — and
/// `None` (dense bf16 encoder) on the bf16 tier. Delegates the floor policy itself to
/// [`mlx_gen_wan::model::effective_te_quant`] so the Q8 rationale lives in exactly one place.
fn resolve_te_quant(
    root: &Path,
    cfg: &KreaRealtimeConfig,
    quant: Option<Quant>,
) -> Result<Option<WanQuant>> {
    let packed = {
        let w = open_transformer_weights(root)?;
        probe_packed_quant(&w, &cfg.wan)?
    };
    let mut probe = cfg.wan.clone();
    probe.quantization = packed;
    Ok(effective_te_quant(&probe, quant))
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
#[allow(clippy::too_many_arguments)]
pub fn generate_i2v(
    root: &Path,
    base_cfg: &KreaRealtimeConfig,
    job: &KreaRealtimeJob,
    reference_image: &Image,
    adapters: &[AdapterSpec],
    quant: Option<Quant>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    let total_latent = bounded_latent_frame_count("requested output", job.num_frames as usize)?;
    generate_i2v_reported(
        root,
        base_cfg,
        job,
        total_latent,
        reference_image,
        adapters,
        quant,
        cancel,
        on_progress,
    )
    .map(|(output, _)| output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_i2v_reported(
    root: &Path,
    base_cfg: &KreaRealtimeConfig,
    job: &KreaRealtimeJob,
    total_latent: usize,
    reference_image: &Image,
    adapters: &[AdapterSpec],
    quant: Option<Quant>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
    let (latent_h, latent_w) = resolve_latent_size(root, job, "i2v")?;
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
    let (context, transformer, vae, adapter_reports) =
        stage_components(root, &cfg, job.prompt, adapters, quant, on_progress)?;
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
    )?;
    let output = generate_i2v_from_components(
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
    )?;
    Ok((output, adapter_reports))
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
    quant: Option<Quant>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    let num_latent = bounded_latent_frame_count("V2V source", source_frames.len())?;
    generate_v2v_reported(
        root,
        base_cfg,
        job,
        source_frames,
        num_latent,
        strength,
        adapters,
        quant,
        cancel,
        on_progress,
    )
    .map(|(output, _)| output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_v2v_reported(
    root: &Path,
    base_cfg: &KreaRealtimeConfig,
    job: &KreaRealtimeJob,
    source_frames: &[Image],
    num_latent: usize,
    strength: f32,
    adapters: &[AdapterSpec],
    quant: Option<Quant>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<(GenerationOutput, Vec<AdapterApplyReport>)> {
    if source_frames.is_empty() {
        return Err(Error::Msg("krea v2v: source clip has no frames".into()));
    }
    let (latent_h, latent_w) = resolve_latent_size(root, job, "v2v")?;
    let cfg = resolve_request_config(base_cfg, latent_h, latent_w, num_latent)?;
    let (context, transformer, vae, adapter_reports) =
        stage_components(root, &cfg, job.prompt, adapters, quant, on_progress)?;
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
    )?;
    let output = generate_v2v_from_components(
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
    )?;
    Ok((output, adapter_reports))
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

    #[test]
    fn direct_generation_routes_enforce_the_cap_before_snapshot_resolution() {
        let cfg = KreaRealtimeConfig::default();
        let cancel = CancelFlag::default();
        let mut progress_calls = 0;
        let oversized_output = KreaRealtimeJob {
            prompt: "",
            width: 512,
            height: 512,
            num_frames: 1_029,
            fps: 16,
            seed: 0,
            steps: None,
        };
        let missing = Path::new("/nonexistent-krea-realtime-snapshot");
        let t2v_error = generate_t2v(
            missing,
            &cfg,
            &oversized_output,
            &[],
            None,
            &cancel,
            &mut |_| progress_calls += 1,
        )
        .expect_err("direct T2V must reject before inspecting the snapshot");
        assert!(matches!(t2v_error, Error::Unsupported(_)));

        let reference = Image {
            width: 1,
            height: 1,
            pixels: vec![0; 3],
        };
        let i2v_error = generate_i2v(
            missing,
            &cfg,
            &oversized_output,
            &reference,
            &[],
            None,
            &cancel,
            &mut |_| progress_calls += 1,
        )
        .expect_err("direct I2V must reject before inspecting the snapshot");
        assert!(matches!(i2v_error, Error::Unsupported(_)));

        let bounded_output = KreaRealtimeJob {
            num_frames: 81,
            ..oversized_output
        };
        let source_frames = vec![reference; 1_029];
        let v2v_error = generate_v2v(
            missing,
            &cfg,
            &bounded_output,
            &source_frames,
            0.5,
            &[],
            None,
            &cancel,
            &mut |_| progress_calls += 1,
        )
        .expect_err("direct V2V must cap its source-derived generation length before the snapshot");
        assert!(matches!(v2v_error, Error::Unsupported(_)));
        assert_eq!(
            progress_calls, 0,
            "no direct route may begin component staging"
        );
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

    /// **sc-15127 (S18): the Mac bound ships with NO sink anchor, and that is a measured decision.**
    ///
    /// The gated real-weight sweep found that long clips *do* drift — but **both** comparisons that
    /// would justify buying a sink came back unresolved at three seeds. The A-vs-D contrast (shipped
    /// window vs a 2.5× wider one) is inside the combined between-seed scatter in both buckets, so the
    /// window is neither implicated nor exonerated; and the A-vs-sink contrast is likewise inside that
    /// scatter. A first-chunk sink is permanently-resident KV (+0.83 GiB for one latent frame, +2.20
    /// for three, measured), and permanently-resident KV must not be bought on an unresolved
    /// comparison. This pins the outcome: `mac_ar_config` must not start manufacturing a sink the
    /// measurement did not justify, and - the other half - the `sink_size` knob must stay a real,
    /// honoured knob so a checkpoint that ships one is respected without a code change.
    #[test]
    fn mac_ar_config_ships_no_sink_anchor_but_honours_one_from_the_checkpoint() {
        let base = KreaRealtimeConfig::krea_realtime_14b();
        assert_eq!(
            base.ar.sink_size, 0,
            "the shipped 14B checkpoint has no sink"
        );
        let mac = mac_ar_config(&base);
        assert_eq!(
            mac.ar.sink_size, 0,
            "the Mac bound must not invent a sink anchor — sc-15127 left both the window and the sink \
             comparisons unresolved at three seeds, and a sink is permanently-resident KV"
        );
        assert_eq!(
            mac.ar.sink_tokens(),
            0,
            "an empty sink prefix must cost zero tokens"
        );
        // ...but the knob is not dead: a checkpoint (or `config.json`) that declares a sink is carried
        // through the Mac bound untouched and turns into real always-attended tokens.
        let mut anchored = base.clone();
        anchored.ar.sink_size = 2;
        let mac_anchored = mac_ar_config(&anchored);
        assert_eq!(
            mac_anchored.ar.sink_size, 2,
            "a checkpoint-declared sink must survive the Mac bound"
        );
        assert_eq!(
            mac_anchored.ar.sink_tokens(),
            2 * base.ar.frame_seq_length,
            "sink frames × frame_seq_length (the reference's sink_tokens)"
        );
        // And the bound itself is unchanged by the sink — the two knobs are independent.
        assert_eq!(mac_anchored.ar.local_attn_size, mac.ar.local_attn_size);
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

        // sc-17807 — the shipped request path is bf16 KV (the knob defaults off), a snapshot that
        // declares a valid tier carries it through, and one that declares an impossible tier is
        // refused HERE rather than on the first chunk of an already-staged clip.
        assert_eq!(cfg.ar.kv_cache_quant, None);
        let mut q8 = base.clone();
        q8.ar.kv_cache_quant = Some(crate::KvCacheQuant::Q8);
        assert_eq!(
            resolve_request_config(&q8, 32, 32, 4)
                .unwrap()
                .ar
                .kv_cache_quant,
            Some(crate::KvCacheQuant::Q8)
        );
        let mut bad = base;
        bad.ar.kv_cache_quant = Some(crate::KvCacheQuant {
            bits: 7,
            group_size: 64,
        });
        let err =
            resolve_request_config(&bad, 32, 32, 4).expect_err("7-bit is not an MLX affine width");
        assert!(
            format!("{err}").contains("affine quantization width"),
            "{err}"
        );
    }

    // ── The TE tier seam (sc-15203, S19) ────────────────────────────────────────────────────────

    /// A tiny geometry whose `dim`/`ffn_dim` are both multiples of the MLX group size, so a packed
    /// probe Linear is well-formed under it.
    fn tiny_te_cfg() -> KreaRealtimeConfig {
        let mut c = KreaRealtimeConfig::krea_realtime_14b();
        c.wan.dim = 64;
        c.wan.ffn_dim = 128;
        c
    }

    /// Write a snapshot `root` containing a transformer at `bits` (`None` = dense bf16) under
    /// `file_name` relative to the root — `dit.safetensors`, or `transformer/shard.safetensors` for the
    /// sharded layout. Only the probe Linear is needed: [`probe_packed_quant`] reads shape metadata for
    /// `blocks.0.self_attn.q` plus the presence of any `.scales`, and materializes nothing.
    fn write_probe_snapshot(dir: &Path, file_name: &str, bits: Option<i32>) -> std::path::PathBuf {
        const GROUP: i32 = 64;
        let root = dir.to_path_buf();
        let path = root.join(file_name);
        std::fs::create_dir_all(path.parent().expect("file has a parent")).unwrap();
        let dim = tiny_te_cfg().wan.dim as i32;
        let mut entries: Vec<(String, Array)> = Vec::new();
        match bits {
            Some(b) => {
                entries.push((
                    "blocks.0.self_attn.q.weight".into(),
                    Array::zeros::<u32>(&[dim, dim * b / 32]).unwrap(),
                ));
                entries.push((
                    "blocks.0.self_attn.q.scales".into(),
                    Array::zeros::<f32>(&[dim, dim / GROUP]).unwrap(),
                ));
                entries.push((
                    "blocks.0.self_attn.q.biases".into(),
                    Array::zeros::<f32>(&[dim, dim / GROUP]).unwrap(),
                ));
            }
            None => entries.push((
                "blocks.0.self_attn.q.weight".into(),
                Array::zeros::<f32>(&[dim, dim]).unwrap(),
            )),
        }
        entries.push((
            "blocks.0.self_attn.q.bias".into(),
            Array::zeros::<f32>(&[dim]).unwrap(),
        ));
        let refs: Vec<(&str, &Array)> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        root
    }

    fn scratch(tmp: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
        let d = tmp.path().join(format!(
            "krea_te_quant_{name}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The UMT5 tier is resolved from the DiT tier **actually on disk** (probed before the TE is
    /// staged, since the encoder loads first) and floored at **Q8** — never tier-matched to the DiT.
    ///
    /// Every case here discriminates: the Q4 snapshot must yield Q8 (matching the DiT's 4 would be
    /// wrong, and `None` would drag an ~11 GB dense UMT5 through the encode phase of a Q4 run); the
    /// dense + `Quant::Q4` case must also yield Q8, not `None` and not Q4; and the dense/no-request
    /// case must be `None`, so "always Q8" fails too.
    #[test]
    fn te_quant_is_probed_off_the_snapshot_and_floors_at_q8() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tiny_te_cfg();
        let q8 = Some(WanQuant {
            bits: 8,
            group_size: mlx_gen::quant::DEFAULT_GROUP_SIZE,
        });

        // A pre-quantized snapshot: the tier comes from the WEIGHTS (this cfg declares no
        // `quantization`, so anything reading the manifest instead would answer `None`).
        for dit_bits in [4, 8] {
            let root =
                write_probe_snapshot(&scratch(&tmp, "packed"), "dit.safetensors", Some(dit_bits));
            assert_eq!(
                resolve_te_quant(&root, &cfg, None).unwrap(),
                q8,
                "a packed Q{dit_bits} DiT must floor the UMT5 at Q8"
            );
            // An explicit request never lowers the floor either.
            assert_eq!(resolve_te_quant(&root, &cfg, Some(Quant::Q4)).unwrap(), q8);
            std::fs::remove_dir_all(&root).ok();
        }

        // A dense bf16 snapshot with no request: the encoder stays dense.
        let dense = write_probe_snapshot(&scratch(&tmp, "dense"), "dit.safetensors", None);
        assert_eq!(resolve_te_quant(&dense, &cfg, None).unwrap(), None);
        // …but a load-time Q4 request over that same dense snapshot still floors the TE at Q8.
        assert_eq!(resolve_te_quant(&dense, &cfg, Some(Quant::Q4)).unwrap(), q8);
        assert_eq!(resolve_te_quant(&dense, &cfg, Some(Quant::Q8)).unwrap(), q8);
        // Nvfp4 is candle-only and never an MLX affine tier ⇒ the TE stays dense.
        assert_eq!(
            resolve_te_quant(&dense, &cfg, Some(Quant::Nvfp4)).unwrap(),
            None
        );

        // The sharded `transformer/` layout is probed identically to the single-file one.
        let sharded = write_probe_snapshot(
            &scratch(&tmp, "sharded"),
            "transformer/shard-00001.safetensors",
            Some(4),
        );
        assert_eq!(resolve_te_quant(&sharded, &cfg, None).unwrap(), q8);

        // A root with neither layout errors loudly (rather than silently answering "dense bf16" and
        // letting the whole run proceed on a snapshot that has no transformer at all).
        let empty = scratch(&tmp, "empty");
        let err = resolve_te_quant(&empty, &cfg, None)
            .expect_err("a root with no transformer weights must fail");
        assert!(
            err.to_string().contains("no transformer weights"),
            "got: {err}"
        );

        for d in [dense, sharded, empty] {
            std::fs::remove_dir_all(&d).ok();
        }
    }

    /// A snapshot whose `config.json` geometry disagrees with its own packed tensors is a hard error at
    /// the TE-probe seam too — the probe runs before anything is loaded, so a corrupt/mismatched
    /// snapshot fails here rather than after the ~11 GB encoder has been staged.
    #[test]
    fn te_quant_probe_surfaces_a_snapshot_config_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let root = write_probe_snapshot(&scratch(&tmp, "mismatch"), "dit.safetensors", Some(4));
        // The same packed file read under a config that declares a different tier.
        let mut cfg = tiny_te_cfg();
        cfg.wan.quantization = Some(WanQuant {
            bits: 8,
            group_size: 64,
        });
        let err = resolve_te_quant(&root, &cfg, None)
            .expect_err("a manifest that disagrees with the packed weights must fail");
        assert!(err.to_string().contains("declares Q8"), "got: {err}");
        std::fs::remove_dir_all(&root).ok();
    }

    /// sc-15325 **regression guard** — the defect, expressed as a property of the policy rather than
    /// as today's numbers.
    ///
    /// The shipped bug was a temporal window of 8 output frames = **2 latent frames**, which starves
    /// the z16 decoder's temporal convolutions and corrupts the content of every tile (18.5/255 mean
    /// abs err against single-pass, 26.6% of a worst frame blown to white). This asserts the invariant
    /// that makes that unreachable: **whenever a temporal tile is emitted at a shipped bucket, it must
    /// span at least `MIN_TEMPORAL_TILE_LATENT_FRAMES` latent frames, with at least
    /// `MIN_TEMPORAL_TILE_LATENT_OVERLAP` latent frames of blend** — at any budget, including a budget
    /// far too small for a full-frame decode (which is exactly the regime the old policy failed in).
    ///
    /// It is pinned to a small injected budget rather than the live free-memory probe so it gates the
    /// *policy*, not this host: on the old `DECODE_TILE_BUDGET_PXFRAMES` window this fails at every one
    /// of these buckets (latent tile 2, latent overlap ≤ 1).
    #[test]
    fn decode_tiling_never_starves_the_temporal_receptive_field() {
        use mlx_gen::tiling::{MIN_TEMPORAL_TILE_LATENT_FRAMES, MIN_TEMPORAL_TILE_LATENT_OVERLAP};

        // Every bucket the old policy collapsed at (≥ ~233k px/frame), plus the one it did not.
        const BUCKETS: [(usize, usize); 6] = [
            (384, 640),
            (512, 512),
            (512, 768),
            (480, 832),
            (720, 1280),
            (384, 512),
        ];
        // A deliberately tight budget: small enough that a full-frame single pass cannot fit, so the
        // selector is forced to tile. The old policy answered this regime with a 2-latent-frame tile.
        std::env::set_var("WAN_VAE_BUDGET_GIB", "12");
        for (h, w) in BUCKETS {
            let cfg = decode_tiling(h, w, 81)
                .unwrap_or_else(|e| panic!("{w}x{h} must remain decodable within 12 GiB: {e}"))
                .unwrap_or_else(|| panic!("{w}x{h}/81f cannot fit a 12 GiB single pass"));
            let Some(t) = cfg.temporal else {
                continue; // a spatial-only plan keeps the whole temporal sequence — full context.
            };
            let lat_tile = t.tile_frames / TEMPORAL_STRIDE as i32;
            let lat_over = (t.overlap_frames / TEMPORAL_STRIDE as i32).min(lat_tile - 1);
            assert!(
                lat_tile >= MIN_TEMPORAL_TILE_LATENT_FRAMES,
                "{w}x{h}: temporal tile {} output frames = {lat_tile} LATENT frames, under the \
                 {MIN_TEMPORAL_TILE_LATENT_FRAMES}-latent-frame receptive-field floor (sc-15325)",
                t.tile_frames
            );
            assert!(
                lat_over >= MIN_TEMPORAL_TILE_LATENT_OVERLAP,
                "{w}x{h}: temporal overlap {} output frames = {lat_over} LATENT frames after the \
                 tile−1 clamp, under the {MIN_TEMPORAL_TILE_LATENT_OVERLAP}-frame blend floor",
                t.overlap_frames
            );
        }
        std::env::remove_var("WAN_VAE_BUDGET_GIB");
    }

    /// The other half of the contract: a clip small enough for one pass is still decoded in one pass
    /// (the floor must not force tiling that was not needed), and a long clip still tiles.
    #[test]
    fn decode_tiling_is_single_pass_for_small_clips_and_tiles_large_ones() {
        std::env::set_var("WAN_VAE_BUDGET_GIB", "12");
        assert!(
            decode_tiling(256, 256, 8).unwrap().is_none(),
            "a 256x256/8f clip fits one pass well inside 12 GiB"
        );
        assert!(
            decode_tiling(256, 256, 400).unwrap().is_some(),
            "a 400-frame clip must tile"
        );
        std::env::remove_var("WAN_VAE_BUDGET_GIB");
    }

    /// sc-15445: Krea deliberately keeps the quality-oriented half-tile overlap even though Wan's
    /// own product selector restored the faster candidate overlap. This exact row is mutation-red if
    /// `decode_tiling` is accidentally routed back through Wan's product wrapper.
    #[test]
    fn decode_tiling_retains_kreas_half_tile_overlap() {
        std::env::set_var("WAN_VAE_BUDGET_GIB", "10");
        let cfg = decode_tiling(384, 640, 84)
            .unwrap()
            .expect("the measured Krea row must tile at 10 GiB");
        std::env::remove_var("WAN_VAE_BUDGET_GIB");
        assert_eq!(
            cfg.temporal.map(|t| (t.tile_frames, t.overlap_frames)),
            Some((32, 16)),
        );
    }
}
