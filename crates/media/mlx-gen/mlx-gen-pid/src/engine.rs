//! [`PidEngine`] — the load-once, decode-many entry point a PiD-eligible provider holds (epic 7840,
//! sc-7845). It owns the heavy weights (the `PixDiT` student checkpoint + the Gemma-2 caption encoder)
//! and the per-latent-space [`PidConfig`], and mints a per-generation [`PidDecoder`] bound to that
//! generation's caption + degrade σ + seed via [`PidEngine::decoder`].
//!
//! A PiD decoder is tied to a *latent space*, not a model, so the engine is parameterized by a
//! backbone tag (`"qwenimage"`, `"flux"`, …) resolved against the [`crate::registry`]. The released
//! students all share the `sr4x` `PixDiT` topology; only the LQ latent-channel count differs per
//! space. This is the shared home the Phase-2 wiring stories (qwen/krea sc-7845, flux sc-7846,
//! flux2 sc-7847, sdxl sc-7848) construct PiD through.

use std::path::{Path, PathBuf};

use mlx_rs::Dtype;

use mlx_gen::weights::Weights;
use mlx_gen::{
    flow_capture_plan, CancelFlag, Error, GenerationRequest, PidWeights, Result, WeightsSource,
};

use crate::caption::CaptionEncoder;
use crate::config::{PidConfig, SamplerConfig};
use crate::decoder::PidDecoder;
use crate::gemma2::{Gemma2, Gemma2Config};
use crate::lq::PidNet;
use crate::registry::lookup;
use crate::sampler::Sampler;

/// Filename of the merged Gemma-2-2b-it checkpoint inside the gemma snapshot dir; falls back to
/// loading every `*.safetensors` shard in the dir when absent.
const GEMMA_MERGED_FILE: &str = "gemma-2-2b-it.safetensors";

/// A loaded PiD decoder engine for one latent space — built once, reused across generations.
pub struct PidEngine {
    /// The converted student checkpoint, retained so [`Self::decoder`] can rebuild a [`PidNet`] per
    /// generation (cheap vs the ~100 s decode — `Array` handles are refcounted).
    weights: Weights,
    /// Per-latent-space backbone config (`sr4x` topology + the space's LQ latent-channel count).
    cfg: PidConfig,
    /// The released 4-step SDE distill sampler config.
    sampler_cfg: SamplerConfig,
    /// The Gemma-2-2b caption encoder (loaded once; the projection runs per caption).
    caption: CaptionEncoder,
    /// Key prefix for [`PidNet::from_weights`] — `""` for the converted checkpoint (the EMA export
    /// pre-strips the `net.` nesting).
    ckpt_prefix: &'static str,
}

impl PidEngine {
    /// Build from explicit paths: the converted PiD checkpoint (a single `.safetensors`), the
    /// `gemma-2-2b-it` snapshot dir (weights + `tokenizer.json`), and the backbone latent-space tag
    /// (e.g. `"qwenimage"`). Errors on an unknown/out-of-scope backbone tag.
    pub fn load(checkpoint: &Path, gemma_dir: &Path, backbone: &str) -> Result<Self> {
        let spec = lookup(backbone).ok_or_else(|| {
            Error::Msg(format!(
                "pid: unknown/out-of-scope backbone {backbone:?} (no PiD latent-space mapping)"
            ))
        })?;
        let weights = Weights::from_file(checkpoint)?;

        // The released students share the sr4x PixDiT topology; only the LQ latent-channel count and
        // the latent grid's spatial compression differ per latent space: 16-ch / 8× for qwen/flux/sd3,
        // 4-ch / 8× for sdxl, and **128-ch / 16×** for flux2 (the packed BN latent — see the registry
        // `FLUX2` note, sc-7847). Both fields drive the LQ adapter geometry + `PidDecoder` output size.
        //
        // PiD v1.5 (sc-12142) ships a different LQ topology (wider trunk, per-token scalar gate,
        // replicate padding, PiT injection, 2048 RoPE ref) under the SAME per-space checkpoint slot, so
        // the worker may hand us either a v1.0 or v1.5 file (and fall back v1.5→v1.0 when v1.5 isn't
        // downloaded — sc-12145). Pick the config by sniffing the WEIGHTS, not the filename.
        let mut cfg = if detect_v1pt5(&weights)? {
            PidConfig::sr4x_v1pt5()
        } else {
            PidConfig::sr4x()
        };
        cfg.lq_latent_channels = spec.latent_channels;
        cfg.latent_spatial_down_factor = spec.latent_spatial_down_factor;

        // Gemma: prefer the merged single-file checkpoint, else load the snapshot dir's shards.
        let merged = gemma_dir.join(GEMMA_MERGED_FILE);
        let gw = if merged.is_file() {
            Weights::from_file(&merged)?
        } else {
            Weights::from_dir(gemma_dir)?
        };
        let gemma = Gemma2::from_weights(&gw, "model.", &Gemma2Config::gemma_2_2b())?;
        let caption = CaptionEncoder::new(gemma, gemma_dir.join("tokenizer.json"))?;

        Ok(Self {
            weights,
            cfg,
            sampler_cfg: SamplerConfig::distill_4step(),
            caption,
            ckpt_prefix: "",
        })
    }

    /// Build from a [`PidWeights`] load-spec component (the gen-core seam) for the given backbone tag.
    /// `checkpoint` must be a [`WeightsSource::File`] (the converted `.safetensors`); `gemma` must be a
    /// [`WeightsSource::Dir`] (the snapshot dir).
    pub fn from_spec(pid: &PidWeights, backbone: &str) -> Result<Self> {
        let checkpoint = file_path(&pid.checkpoint, "pid checkpoint")?;
        let gemma_dir = dir_path(&pid.gemma, "pid gemma encoder")?;
        Self::load(&checkpoint, &gemma_dir, backbone)
    }

    /// Spatial SR factor baked into the student (4× for every released backbone).
    pub fn scale(&self) -> i32 {
        self.cfg.sr_scale
    }

    /// VAE spatial compression (latent grid → pixel grid; 8 for the catalog VAEs).
    pub fn vae_compression(&self) -> i32 {
        self.cfg.latent_spatial_down_factor
    }

    /// The backbone config (`patch_size`/`hidden_size`/…) — used by the F-013 decode memory-budget
    /// guard ([`crate::budget::guard`]) at the resolve seam.
    pub fn config(&self) -> &PidConfig {
        &self.cfg
    }

    /// Mint a per-generation [`PidDecoder`] bound to one caption. `sigma` is the LQ degrade level
    /// (0 for a clean-latent decode of a fully-denoised latent); `seed` drives the sampler's noise +
    /// per-step ε. Rebuilds the [`PidNet`] from the retained weights (cheap relative to decode) and
    /// encodes the caption to bf16 embeddings (the released inference dtype).
    pub fn decoder(&self, caption: &str, sigma: f32, seed: u64) -> Result<PidDecoder> {
        let net = PidNet::from_weights(&self.weights, self.ckpt_prefix, &self.cfg)?;
        let caption_embs = self.caption.encode(caption)?.as_dtype(Dtype::Bfloat16)?;
        Ok(PidDecoder::new(
            net,
            Sampler::new(&self.sampler_cfg),
            caption_embs,
            sigma,
            self.cfg.sr_scale,
            self.cfg.latent_spatial_down_factor,
            seed,
        ))
    }
}

/// Resolve the decode seam for one generation (epic 7840) — the shared entry point every PiD-eligible
/// provider calls (Qwen/Krea sc-7845; FLUX.1/Boogu/Chroma/Z-Image sc-7846; flux2/sdxl to follow). It
/// lives here in `mlx-gen-pid` rather than in a provider crate because the providers don't share a
/// dependency edge (Z-Image depends on neither Qwen-Image nor FLUX), but they all depend on this one.
///
/// When `req.use_pid` is set, mint a per-generation [`PidDecoder`] bound to the prompt — a **clean σ=0
/// decode of the fully-denoised latent**, seeded from `base_seed`; the caller passes it (as a
/// `&dyn LatentDecoder`) to its decode call site in place of the native VAE. Errors (rather than
/// silently falling back) if PiD was requested but the model was not loaded with `LoadSpec::pid`. When
/// the flag is unset, returns `None` and the caller uses the native VAE — the byte-exact default path.
///
/// `model_id` only labels the error. The returned decoder owns its caption embeddings + a freshly built
/// `PidNet`, so it lives as long as the borrow passed to the decode site; all `count` images in a
/// request share this one decoder (same prompt → same caption).
///
/// This is the **clean σ=0** entry: it always decodes the fully-denoised latent. The `from_ldm`
/// early-stop x_t-capture (σ>0, decoding a partially-denoised latent — sc-7993) is wired only for the
/// flow-match qwenimage space today via [`resolve_pid_decoder_at_sigma`]; any other latent space that
/// still routes through this function rejects a [`pid_capture_sigma`](GenerationRequest::pid_capture_sigma)
/// request rather than silently dropping it (the σ-frame map for a variance-preserving SDXL student and
/// the flux/flux2 siblings are follow-ons).
pub fn resolve_pid_decoder(
    pid: Option<&PidEngine>,
    req: &GenerationRequest,
    base_seed: u64,
    model_id: &str,
) -> Result<Option<PidDecoder>> {
    if req.use_pid && req.pid_capture_sigma.is_some() {
        return Err(Error::Msg(format!(
            "{model_id}: pid_capture_sigma (from_ldm early-stop) is not wired for this latent space \
             yet — sc-7993 wired the flow-match qwenimage space (Qwen-Image / Krea); the flux / flux2 \
             and the variance-preserving SDXL siblings are follow-ons"
        )));
    }
    resolve_pid_decoder_at_sigma(pid, req, base_seed, model_id, 0.0)
}

/// `from_ldm`-aware variant of [`resolve_pid_decoder`] (sc-7993): mint the per-generation [`PidDecoder`]
/// at an explicit degrade `capture_sigma` (the **achieved** σ of a partially-denoised `x_k`, in the
/// flow-match frame). `0.0` reproduces the clean-latent decode. The caller is responsible for actually
/// truncating its denoise schedule to the matching step (see [`mlx_gen::flow_capture_plan`]),
/// so the latent it later hands to `PidDecoder::decode` really sits at this σ — this function only
/// binds σ into the decoder. Same `use_pid`/loaded-engine contract as [`resolve_pid_decoder`].
pub fn resolve_pid_decoder_at_sigma(
    pid: Option<&PidEngine>,
    req: &GenerationRequest,
    base_seed: u64,
    model_id: &str,
    capture_sigma: f32,
) -> Result<Option<PidDecoder>> {
    if !req.use_pid {
        return Ok(None);
    }
    let engine = pid.ok_or_else(|| {
        Error::Msg(format!(
            "{model_id}: use_pid was requested but no PiD decoder is loaded (load with LoadSpec::pid)"
        ))
    })?;
    Ok(Some(mint_planned_decoder(
        engine,
        model_id,
        &req.prompt,
        req.width,
        req.height,
        capture_sigma,
        base_seed,
        req.cancel.clone(),
    )?))
}

/// Mint a per-generation [`PidDecoder`] with the F-013/sc-10087 decode policy — budget `guard` →
/// `plan_tile_edge` → `with_tiling` — already applied. This is the **single home** for that policy
/// (F-149): registry providers reach it via [`resolve_pid_decoder_at_sigma`], and the struct-API
/// InstantID (which mints `engine.decoder(...)` directly, composing no registered `Generator`) calls it
/// too, so the budget guard + watchdog tiling travel to every consumer instead of being copy-pasted or
/// silently missing.
///
/// PiD super-resolves in pixel space by `engine.scale()`, so a `max_size`-legal `width × height` decodes
/// at `(width·scale) × (height·scale)` — a 1536² request → 6144², which a single whole-image forward
/// can't hold: on Metal it trips the IOGPU watchdog, on CUDA it exhausts VRAM. We **tile** the pixel-space
/// forward rather than refuse (sc-10087): size the tile against this machine's `safe_budget_gib()` (the
/// shared wan/seedvr2 budget) and the Metal watchdog-safe forward edge, and refuse only when even a
/// minimum tile plus the resident output-resolution buffers won't fit.
///
/// The guard/plan price a **single** decode (`B=1`): the returned decoder is shared across a request's
/// `count` loop, but each `decode` holds one output-resolution buffer set, so the concurrent peak never
/// scales with `count` (F-150). `cancel` is bound into the decoder so the ~100 s 4-step decode honors a
/// per-step cancel (F-006) — the `LatentDecoder::decode` trait signature carries no flag.
#[allow(clippy::too_many_arguments)]
pub fn mint_planned_decoder(
    engine: &PidEngine,
    model_id: &str,
    prompt: &str,
    width: u32,
    height: u32,
    capture_sigma: f32,
    seed: u64,
    cancel: CancelFlag,
) -> Result<PidDecoder> {
    let safe_gib = mlx_gen::memory::safe_budget_gib();
    let scale = engine.scale();
    let cfg = engine.config();
    crate::budget::guard(model_id, width, height, scale, cfg, safe_gib)?;
    let (th, tw) = (
        (height * scale as u32) as i32,
        (width * scale as u32) as i32,
    );
    let plan = crate::budget::plan_tile_edge(1, th, tw, cfg.patch_size, cfg.hidden_size, safe_gib);
    let mut decoder = engine
        .decoder(prompt, capture_sigma, seed)?
        .with_cancel(cancel);
    if !plan.whole_fits {
        decoder = decoder.with_tiling(plan.edge, plan.overlap);
    }
    Ok(decoder)
}

/// Resolve the `from_ldm` early-stop for one **flow-match** generation (sc-7993): fold `req.use_pid` +
/// [`req.pid_capture_sigma`](GenerationRequest::pid_capture_sigma) together with the schedule into the
/// two values a wired site needs — the decoder's degrade σ and how many schedule entries to denoise.
///
/// Returns `(capture_sigma, keep)`: pass `capture_sigma` to [`resolve_pid_decoder_at_sigma`] and run the
/// denoise over `&sigmas[..keep]` (the latent then sits at exactly `capture_sigma`, so the two agree).
/// The clean path yields `(0.0, sigmas.len())` — the full schedule, σ=0 — whenever PiD is off, no
/// capture is requested, or the requested ceiling would stop the denoise at/before the img2img
/// `start_step` (no benefit). `start_step` is `0` for txt2img / edit / control.
pub fn flow_capture_for_request(
    req: &GenerationRequest,
    sigmas: &[f32],
    start_step: usize,
) -> (f32, usize) {
    let plan = req
        .use_pid
        .then(|| flow_capture_plan(sigmas, req.pid_capture_sigma))
        .flatten();
    match plan {
        Some(c) if c.keep > start_step + 1 => (c.sigma, c.keep),
        _ => (0.0, sigmas.len()),
    }
}

/// Sniff whether a loaded PiD checkpoint is a **v1.5** student (sc-12141/sc-12142) vs a base `sr4x`
/// v1.0 student, so [`PidEngine::load`] can pick the right [`PidConfig`] from the same per-space slot.
///
/// Two independent signals must agree: the first LQ gate's `content_proj` output width (**1** = v1.5's
/// per-token scalar gate; `hidden_size` = v1.0's per-token-per-dim gate) and the presence of the
/// top-level **`pit_lq_gate`** (v1.5-only). The converted EMA export pre-strips the `net.` nesting, so
/// keys are bare. Errors if the gate is missing (not a PiD student) or the signals disagree (a
/// malformed / version-mixed checkpoint) rather than guessing.
fn detect_v1pt5(w: &Weights) -> Result<bool> {
    let gate_rows = w
        .require("lq_proj.gate_modules.0.content_proj.weight")?
        .shape()[0];
    let scalar_gate = gate_rows == 1;
    let has_pit_gate = w.get("pit_lq_gate.content_proj.weight").is_some();
    if scalar_gate != has_pit_gate {
        return Err(Error::Msg(format!(
            "pid: inconsistent v1.5 checkpoint signals — scalar gate (content_proj rows={gate_rows}) = \
             {scalar_gate}, but pit_lq_gate present = {has_pit_gate}; the checkpoint is malformed or \
             mixes versions"
        )));
    }
    Ok(scalar_gate)
}

/// Extract the single-file path from a [`WeightsSource`], rejecting a directory.
fn file_path(src: &WeightsSource, what: &str) -> Result<PathBuf> {
    match src {
        WeightsSource::File(p) => Ok(p.clone()),
        WeightsSource::Dir(_) => Err(Error::Msg(format!(
            "{what}: expected the converted .safetensors file, got a directory"
        ))),
    }
}

/// Extract the directory path from a [`WeightsSource`], rejecting a single file.
fn dir_path(src: &WeightsSource, what: &str) -> Result<PathBuf> {
    match src {
        WeightsSource::Dir(p) => Ok(p.clone()),
        WeightsSource::File(_) => Err(Error::Msg(format!(
            "{what}: expected a snapshot directory, got a single file"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `PidEngine` is not `Debug` (it owns `Weights`/`CaptionEncoder`), so match rather than
    // `.expect_err()` (which would require `Debug` on the `Ok` payload).
    fn err_string<T>(r: Result<T>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn unknown_backbone_errors() {
        let err = err_string(PidEngine::load(
            Path::new("/nonexistent/ckpt.safetensors"),
            Path::new("/nonexistent/gemma"),
            "dinov2", // out-of-scope (vision-encoder latent, not a VAE latent)
        ));
        assert!(err.contains("out-of-scope backbone"), "got: {err}");
    }

    #[test]
    fn from_spec_rejects_swapped_sources() {
        // checkpoint must be a File, gemma must be a Dir — a swap is rejected before any load.
        let swapped = PidWeights {
            checkpoint: WeightsSource::Dir("/nonexistent/ckpt".into()),
            gemma: WeightsSource::Dir("/nonexistent/gemma".into()),
        };
        let err = err_string(PidEngine::from_spec(&swapped, "qwenimage"));
        assert!(err.contains("converted .safetensors file"), "got: {err}");
    }

    #[test]
    fn resolve_pid_decoder_off_is_none() {
        // use_pid unset → None (the native VAE path), even with no engine loaded.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(resolve_pid_decoder(None, &req, 0, "some_model")
            .unwrap()
            .is_none());
    }

    #[test]
    fn resolve_pid_decoder_requested_without_engine_errors() {
        // use_pid set but no PiD loaded → a clear error, not a silent VAE fallback. `PidDecoder` is
        // not `Debug`, so match rather than `.expect_err()`.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            use_pid: true,
            ..Default::default()
        };
        let err = err_string(resolve_pid_decoder(None, &req, 0, "some_model"));
        assert!(err.contains("no PiD decoder is loaded"), "got: {err}");
    }

    #[test]
    fn resolve_pid_decoder_rejects_capture_sigma_for_unwired_space() {
        // A latent space still on the clean-σ=0 resolve must not silently drop a from_ldm request
        // (sc-7993): pid_capture_sigma + use_pid → a clear "not wired for this latent space" error,
        // surfaced before any load. The flow-match qwenimage sites use resolve_pid_decoder_at_sigma.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            use_pid: true,
            pid_capture_sigma: Some(0.2),
            ..Default::default()
        };
        let err = err_string(resolve_pid_decoder(None, &req, 0, "flux"));
        assert!(
            err.contains("not wired for this latent space"),
            "got: {err}"
        );
    }

    #[test]
    #[ignore = "needs the converted PiD v1.5 flux safetensors (PID_V1PT5_CKPT)"]
    fn detect_v1pt5_true_on_real_v1pt5_checkpoint() {
        // sc-12142: the sniff picks v1.5 for a real v1.5 student (scalar gate + pit_lq_gate present).
        let path = std::env::var("PID_V1PT5_CKPT")
            .expect("set PID_V1PT5_CKPT to the converted v1.5 flux safetensors");
        let w = Weights::from_file(&path).unwrap();
        assert!(
            detect_v1pt5(&w).unwrap(),
            "v1.5 checkpoint should sniff as v1.5"
        );
    }

    #[test]
    fn resolve_pid_decoder_ignores_capture_sigma_when_pid_off() {
        // pid_capture_sigma is only consulted under use_pid — off → None (native VAE), no error.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            use_pid: false,
            pid_capture_sigma: Some(0.2),
            ..Default::default()
        };
        assert!(resolve_pid_decoder(None, &req, 0, "flux")
            .unwrap()
            .is_none());
    }
}
