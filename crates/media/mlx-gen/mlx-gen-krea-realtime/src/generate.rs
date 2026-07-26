//! Krea Realtime 14B **autoregressive chunk driver** (sc-8437 S4; clean-context recompute sc-8438 S5).
//!
//! The AR loop that turns the S3 causal forward + persistent KV cache
//! ([`CausalKreaTransformer`]) into a latent video sequence, mirroring
//! the reference `causal_inference.py:177-245`. For each of `ceil(num_frames / num_frames_per_block)`
//! chunks (a chunk = `num_frames_per_block` latent frames) it runs the Self-Forcing few-step denoise
//! ([`FewStepSchedule`]) — one batch-1 forward per step, **CFG off** — threading the persistent cache
//! and advancing the global token offset so chunk *k* attends chunk *k−1*'s committed context.
//!
//! **Cache commit — exactly one per chunk.** With [`KreaArConfig::do_kv_recomp`](crate::KreaArConfig::do_kv_recomp)
//! **off** (the S4 baseline / A-B) the final near-clean denoise step runs
//! [`CausalKreaTransformer::forward_chunk`] (commits), every earlier step
//! [`CausalKreaTransformer::forward_chunk_readonly`] (reads, appends nothing). With it **on** (the
//! shipped default, **S5**) *every* denoise step is read-only and one extra forward reruns on the
//! chunk's clean `x0` at [`KreaArConfig::context_noise`](crate::KreaArConfig::context_noise) and commits
//! *that* clean-context K/V — the reference's `causal_inference.py:227-236` "rerun with timestep zero to
//! update KV cache using clean context". The output is the **latent sequence**
//! `[out_dim, num_frames, H, W]` (f32) — UMT5 text encoding and VAE decode / clip assembly (incl. the
//! first-frame VAE re-anchor) are wired at the pipeline level in **S6**; i2v/v2v conditioning is **S7**;
//! long-clip coherence with real weights is S13.
//! `context` (the UMT5 text embedding) is taken as an input parameter; the DiT-side text embedding +
//! cross-attention K/V are built here once per prompt.

use mlx_gen::{Error, Result};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{random, Array};

use crate::causal::{CausalKreaTransformer, CausalKvCache};
use crate::config::KreaRealtimeConfig;
use crate::scheduler::{euler_x0, renoise_step, FewStepSchedule};

/// A text-to-video autoregressive generation request (t2v only — latents start from seeded noise).
#[derive(Clone, Debug)]
pub struct ArGenParams {
    /// Deterministic seed: fixes the per-clip init noise and the per-step renoise noise.
    pub seed: u64,
    /// Denoising-steps override — `None` uses the config's `denoising_step_list` (5 forwards for the
    /// shipped 14B), `Some(n)` rebuilds an `n`-forward schedule. See [`FewStepSchedule::new`].
    pub steps: Option<usize>,
    /// Number of **latent** frames to generate (the caller derives this from the requested duration ×
    /// [`fps`](Self::fps) via the VAE's temporal compression, which S6 owns).
    pub num_latent_frames: usize,
    /// Latent spatial height. Must satisfy `(latent_height / patch_h) · (latent_width / patch_w) =
    /// frame_seq_length` (the model's canonical per-frame token count) — see [`generate_latents`].
    pub latent_height: usize,
    /// Latent spatial width. See [`latent_height`](Self::latent_height).
    pub latent_width: usize,
    /// Output frames-per-second — carried onto the assembled clip at the pipeline level (S6); it does
    /// not affect the latent sequence produced here.
    pub fps: u32,
}

/// Per-chunk latent-frame counts for `num_frames` at `frames_per_block`: full blocks then a
/// (possibly partial) trailing block. Length `ceil(num_frames / frames_per_block)`.
fn chunk_frame_counts(num_frames: usize, frames_per_block: usize) -> Vec<usize> {
    if frames_per_block == 0 || num_frames == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut remaining = num_frames;
    while remaining > 0 {
        let take = remaining.min(frames_per_block);
        out.push(take);
        remaining -= take;
    }
    out
}

/// Run the full t2v AR generation, owning a fresh KV cache. See [`generate_latents_into`].
///
/// `context` is the UMT5 text embedding `[text_len, text_dim]` (f32) — this builds the DiT text
/// embedding + per-prompt cross-attention K/V once, then drives the chunk loop. Returns the latent
/// sequence `[out_dim, num_latent_frames, latent_height, latent_width]` (f32).
///
/// Errors if the latent geometry does not yield exactly `frame_seq_length` tokens per frame (the S3
/// causal forward bakes `frame_seq_length` into its cache windowing and RoPE frame offset, so the
/// caller must size latents to the model's canonical per-frame token count).
pub fn generate_latents(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    context: &Array,
    params: &ArGenParams,
) -> Result<Array> {
    let mut cache = transformer.new_cache();
    generate_latents_into(transformer, cfg, context, params, &mut cache)
}

/// [`generate_latents`] against a caller-owned KV cache — the seam S5 (clean-context recompute) and S6
/// (the `Generator`) drive, and the seam tests inspect to assert the cache grows by exactly one chunk
/// per chunk (KV threading). The `cache` must be empty on entry; it holds the full clip's committed
/// self-attention K/V on return (`stored_tokens() == num_latent_frames · frame_seq_length`).
pub fn generate_latents_into(
    transformer: &CausalKreaTransformer,
    cfg: &KreaRealtimeConfig,
    context: &Array,
    params: &ArGenParams,
    cache: &mut CausalKvCache,
) -> Result<Array> {
    let (ph, pw) = (cfg.wan.patch_size.1, cfg.wan.patch_size.2);
    let frame_seq_length = cfg.ar.frame_seq_length;
    if ph == 0
        || pw == 0
        || !params.latent_height.is_multiple_of(ph)
        || !params.latent_width.is_multiple_of(pw)
    {
        return Err(Error::Msg(format!(
            "krea AR: latent {}x{} is not divisible by the patch size {ph}x{pw}",
            params.latent_height, params.latent_width
        )));
    }
    let per_frame_tokens = (params.latent_height / ph) * (params.latent_width / pw);
    if per_frame_tokens != frame_seq_length {
        return Err(Error::Msg(format!(
            "krea AR: latent {}x{} yields {per_frame_tokens} tokens/frame but the model's \
             frame_seq_length is {frame_seq_length}; size the latents to the canonical per-frame \
             token count",
            params.latent_height, params.latent_width
        )));
    }
    if !cache.is_empty() {
        return Err(Error::Msg(
            "krea AR: generate_latents_into requires an empty KV cache".into(),
        ));
    }

    // The Self-Forcing few-step schedule (shift + denoising_step_list from the config; caller override).
    let schedule = FewStepSchedule::new(
        cfg.ar.timestep_shift as f64,
        &cfg.ar.denoising_step_list,
        params.steps,
    )?;

    // DiT text embedding + per-prompt cross-attention K/V (position-independent; built once).
    let ctx = transformer.inner().embed_text(context)?;
    let cross_kv = transformer.prepare_cross_kv(&ctx)?;

    // The per-chunk forward: `commit = true` appends this forward's self-attention K/V to the
    // persistent cache; `commit = false` reads the committed history and appends nothing. The AR loop
    // decides which forward commits (the S4 final denoise step, or the S5 clean-context recompute).
    // Exactly one forward per call (CFG off — no uncond branch).
    let denoise = |chunk: &Array, t: f32, start: usize, commit: bool| -> Result<Array> {
        if commit {
            transformer.forward_chunk(chunk, t, &cross_kv, start, cache)
        } else {
            transformer.forward_chunk_readonly(chunk, t, &cross_kv, start, cache)
        }
    };

    run_ar_loop(
        &schedule,
        cfg.wan.in_dim,
        frame_seq_length,
        cfg.ar.num_frames_per_block,
        params,
        cfg.ar.do_kv_recomp,
        cfg.ar.context_noise,
        denoise,
    )
}

/// The AR chunk-loop core, generic over the per-chunk forward so it is unit-testable without weights.
///
/// `denoise(chunk, t, start_token, commit) -> velocity` runs one batch-1 forward: it reads the
/// committed history at `start_token` and returns the model velocity `[out_dim, F_chunk, H, W]`;
/// `commit = true` appends that forward's self-attention K/V to the persistent cache. The loop owns the
/// schedule, the deterministic RNG (init noise + per-step renoise), the Euler `x0` estimate, the
/// renoise, and — when `do_kv_recomp` — the S5 **clean-context KV recompute**: after a chunk's denoise
/// steps (all read-only in that mode), it reruns one forward on the chunk's clean `x0` at
/// `context_noise` and commits *that* K/V (mirroring `causal_inference.py:227-236`). Exactly one commit
/// per chunk in both modes: the final denoise step when `do_kv_recomp` is off (the S4 baseline), or the
/// recompute forward when it is on. Returns the frame-axis concatenation of every chunk's final `x0`,
/// `[out_dim, num_frames, H, W]` (f32).
#[allow(clippy::too_many_arguments)]
fn run_ar_loop(
    schedule: &FewStepSchedule,
    channels: usize,
    frame_seq_length: usize,
    frames_per_block: usize,
    params: &ArGenParams,
    do_kv_recomp: bool,
    context_noise: f32,
    mut denoise: impl FnMut(&Array, f32, usize, bool) -> Result<Array>,
) -> Result<Array> {
    let step_ts = schedule.step_timesteps();
    if step_ts.is_empty() {
        return Err(Error::Msg("krea AR: empty denoising schedule".into()));
    }
    let n_steps = step_ts.len();
    let c = channels as i32;
    let (h, w) = (params.latent_height as i32, params.latent_width as i32);
    let num_frames = params.num_latent_frames;

    // Init the whole clip's noise once from the seed (then slice per chunk, matching the reference's
    // `noise[:, block]`), and carry a split PRNG key for the per-step renoise draws — all seed-derived.
    let mut key = random::key(params.seed)?;
    let (noise_key, next) = random::split(&key, 2)?;
    key = next;
    let full_noise =
        random::normal::<f32>(&[c, num_frames as i32, h, w], None, None, Some(&noise_key))?;

    let mut outputs: Vec<Array> = Vec::new();
    let mut frame_cursor = 0usize;
    let mut start_token = 0usize;
    for chunk_frames in chunk_frame_counts(num_frames, frames_per_block) {
        // This chunk's init noise: full_noise[:, frame_cursor : frame_cursor + chunk_frames].
        let idx: Vec<i32> = (frame_cursor as i32..(frame_cursor + chunk_frames) as i32).collect();
        let mut cur = full_noise.take_axis(Array::from_slice(&idx, &[idx.len() as i32]), 1)?;

        // Commit the chunk's K/V at the final denoise step ONLY when the recompute is off (the S4
        // baseline). With recompute on, every denoise step is read-only and the single commit is the
        // clean-context recompute below — exactly one commit per chunk either way.
        let mut chunk_x0: Option<Array> = None;
        for (i, &t) in step_ts.iter().enumerate() {
            let is_final = i + 1 == n_steps;
            let commit = is_final && !do_kv_recomp;
            let velocity = denoise(&cur, t as f32, start_token, commit)?;
            let x0_i = euler_x0(&cur, &velocity, schedule.sigma_at_timestep(t))?;
            if is_final {
                mlx_rs::transforms::eval([&x0_i])?;
                chunk_x0 = Some(x0_i);
            } else {
                // Renoise x0 to the next step's noise level with fresh seed-derived Gaussian noise.
                let sigma_next = schedule.sigma_at_timestep(step_ts[i + 1]);
                let (eps_key, next) = random::split(&key, 2)?;
                key = next;
                let eps = random::normal::<f32>(x0_i.shape(), None, None, Some(&eps_key))?;
                cur = renoise_step(&x0_i, &eps, sigma_next)?;
                mlx_rs::transforms::eval([&cur])?;
            }
        }
        let x0 = chunk_x0.ok_or_else(|| Error::Msg("krea AR: empty denoising schedule".into()))?;

        // Step 3.3: clean-context KV recompute — rerun one forward on the chunk's clean x0 at
        // context_noise and commit *that* K/V (the read-only denoise steps left the cache untouched, so
        // this is the chunk's one and only commit). Mirrors `causal_inference.py:227-236`.
        if do_kv_recomp {
            let _ = denoise(&x0, context_noise, start_token, true)?;
        }

        outputs.push(x0);
        frame_cursor += chunk_frames;
        start_token += chunk_frames * frame_seq_length;
    }

    let refs: Vec<&Array> = outputs.iter().collect();
    Ok(concatenate_axis(&refs, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn params(seed: u64, frames: usize) -> ArGenParams {
        ArGenParams {
            seed,
            steps: None,
            num_latent_frames: frames,
            latent_height: 4,
            latent_width: 4,
            fps: 16,
        }
    }

    #[test]
    fn chunk_frame_counts_ceils_and_partials() {
        assert_eq!(chunk_frame_counts(6, 3), vec![3, 3]); // exact
        assert_eq!(chunk_frame_counts(7, 3), vec![3, 3, 1]); // ceil, partial tail
        assert_eq!(chunk_frame_counts(5, 2), vec![2, 2, 1]);
        assert_eq!(chunk_frame_counts(1, 3), vec![1]);
        assert_eq!(chunk_frame_counts(0, 3), Vec::<usize>::new());
        assert_eq!(chunk_frame_counts(3, 0), Vec::<usize>::new());
        // Count == ceil(frames / fpb).
        assert_eq!(chunk_frame_counts(7, 3).len(), 7usize.div_ceil(3));
    }

    /// The loop structure with a fake denoiser (no weights), **recompute off** (the S4 baseline):
    /// forward count = chunks · steps, `commit` fires exactly once per chunk (the last step),
    /// `start_token` advances by `chunk_frames · frame_seq_length` and equals a simulated cache position
    /// on every call (KV threading at the loop level — chunk *k* is driven against the committed history,
    /// never a reset), and the output has the correct `[C, num_frames, H, W]` shape.
    #[test]
    fn ar_loop_structure_and_kv_threading() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let fsl = 4; // 4×4 latent, patch 2×2 → 2·2 tokens/frame
        let fpb = 3;
        let p = params(1234, 7); // 3 chunks: [3, 3, 1]

        // Simulated cache position: advances only when a chunk commits, like the real cache.
        let sim_stored = RefCell::new(0usize);
        let calls = RefCell::new(Vec::<(f32, usize, bool)>::new());

        // Recompute OFF ⇒ the final denoise step commits (S4 behaviour).
        let out = run_ar_loop(
            &schedule,
            16,
            fsl,
            fpb,
            &p,
            false,
            0.0,
            |chunk, t, start, commit| {
                // Every step of a chunk sees the committed history (== the simulated cache position).
                assert_eq!(
                    start,
                    *sim_stored.borrow(),
                    "chunk forward must run against the committed history"
                );
                calls.borrow_mut().push((t, start, commit));
                if commit {
                    *sim_stored.borrow_mut() += chunk.shape()[1] as usize * fsl;
                }
                // Fake velocity: zeros ⇒ x0 = cur (the loop's RNG still drives the output).
                Ok(Array::zeros::<f32>(chunk.shape()).unwrap())
            },
        )
        .unwrap();

        // 3 chunks × 5 steps = 15 forwards; commit true exactly 3 times (the final step of each chunk).
        let calls = calls.into_inner();
        assert_eq!(calls.len(), 15);
        assert_eq!(calls.iter().filter(|(_, _, f)| *f).count(), 3);
        // The commit fires on the terminal timestep 0 (the S4 final near-clean step).
        assert!(calls
            .iter()
            .filter(|(_, _, f)| *f)
            .all(|(t, _, _)| *t == 0.0));
        // start_token sequence: chunk 0 @ 0 (×5), chunk 1 @ 3·fsl (×5), chunk 2 @ 6·fsl (×5).
        assert!(calls[0..5].iter().all(|(_, s, _)| *s == 0));
        assert!(calls[5..10].iter().all(|(_, s, _)| *s == 3 * fsl));
        assert!(calls[10..15].iter().all(|(_, s, _)| *s == 6 * fsl));
        // The committed history ends at the full clip's token count.
        assert_eq!(*sim_stored.borrow(), 7 * fsl);

        // Output shape [C, num_frames, H, W].
        assert_eq!(out.shape(), &[16, 7, 4, 4]);
    }

    /// S5 clean-context recompute wiring + toggle (weight-free). With `do_kv_recomp = true` every
    /// denoise step is read-only and a single extra forward commits per chunk **at `context_noise`, on
    /// the chunk's clean `x0`** — not the noisy final denoise input. A fake non-zero velocity makes the
    /// clean `x0` differ from the final step's input so the commit input is discriminated; a single
    /// chunk makes the loop output equal that `x0`.
    #[test]
    fn ar_loop_recompute_commits_clean_context_when_on() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let n_steps = schedule.num_steps();
        let fsl = 4;
        let fpb = 2;
        let p = params(1234, 2); // exactly ONE chunk ⇒ loop output == that chunk's clean x0
        let ctx_noise = 7.0f32; // sentinel: the recompute must run at exactly this timestep

        // (t, start, commit, input_sum). A non-zero constant velocity ⇒ x0 = cur − σ·1 ≠ cur, so the
        // recompute's clean-x0 input is clearly discriminable from the noisy final denoise input.
        let calls = RefCell::new(Vec::<(f32, usize, bool, f32)>::new());
        let denoise = |chunk: &Array, t: f32, start: usize, commit: bool| -> Result<Array> {
            let s = mlx_rs::ops::sum(chunk, None).unwrap().item::<f32>();
            calls.borrow_mut().push((t, start, commit, s));
            Ok(Array::ones::<f32>(chunk.shape())?)
        };
        let out = run_ar_loop(&schedule, 16, fsl, fpb, &p, true, ctx_noise, denoise).unwrap();
        let calls = calls.into_inner();

        // n_steps read-only denoise forwards + exactly one commit (the recompute).
        assert_eq!(
            calls.len(),
            n_steps + 1,
            "denoise steps (read-only) + one recompute"
        );
        assert_eq!(
            calls.iter().filter(|(_, _, c, _)| *c).count(),
            1,
            "exactly one commit per chunk (the recompute), not the final denoise step"
        );
        // The denoise steps are all read-only; the single commit is the recompute at context_noise.
        assert!(calls[..n_steps].iter().all(|(_, _, c, _)| !*c));
        let (rt, _rs, rc, r_sum) = *calls.last().unwrap();
        assert!(rc, "the last forward is the commit");
        assert_eq!(rt, ctx_noise, "the recompute runs at context_noise");
        // The recompute ran on the clean x0 (== the single-chunk output), NOT the final denoise input.
        let out_sum = mlx_rs::ops::sum(&out, None).unwrap().item::<f32>();
        let final_denoise_input_sum = calls[n_steps - 1].3;
        assert!(
            (r_sum - out_sum).abs() < 1e-3,
            "recompute input must be the clean x0 (loop output): got {r_sum}, want {out_sum}"
        );
        assert!(
            (r_sum - final_denoise_input_sum).abs() > 1e-4,
            "recompute input must differ from the noisy final denoise input ({final_denoise_input_sum})"
        );
    }

    /// Recompute toggle: `do_kv_recomp = false` reproduces the S4 commit pattern (no extra forward; the
    /// final denoise step is the single commit) — proving the two modes are a clean A/B.
    #[test]
    fn ar_loop_recompute_off_matches_s4_commit_pattern() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let n_steps = schedule.num_steps();
        let p = params(1234, 2); // one chunk
        let calls = RefCell::new(Vec::<(f32, bool)>::new());
        let denoise = |chunk: &Array, t: f32, _s: usize, commit: bool| -> Result<Array> {
            calls.borrow_mut().push((t, commit));
            Ok(Array::zeros::<f32>(chunk.shape()).unwrap())
        };
        run_ar_loop(&schedule, 16, 4, 2, &p, false, 0.0, denoise).unwrap();
        let calls = calls.into_inner();
        assert_eq!(calls.len(), n_steps, "no recompute forward when off");
        assert_eq!(
            calls.iter().filter(|(_, c)| *c).count(),
            1,
            "one commit: the final step"
        );
        assert!(calls.last().unwrap().1 && calls.last().unwrap().0 == 0.0);
    }

    /// Determinism: identical seed ⇒ identical latents; a different seed ⇒ different latents.
    #[test]
    fn ar_loop_is_seed_deterministic() {
        let schedule = FewStepSchedule::new(5.0, &[1000, 937, 833, 625, 0], None).unwrap();
        let zeros = |chunk: &Array, _t: f32, _s: usize, _f: bool| -> Result<Array> {
            Ok(Array::zeros::<f32>(chunk.shape()).unwrap())
        };

        let a = run_ar_loop(&schedule, 16, 4, 3, &params(7, 6), false, 0.0, zeros).unwrap();
        let b = run_ar_loop(&schedule, 16, 4, 3, &params(7, 6), false, 0.0, zeros).unwrap();
        let c = run_ar_loop(&schedule, 16, 4, 3, &params(8, 6), false, 0.0, zeros).unwrap();

        let diff_same = mlx_rs::ops::max(
            mlx_rs::ops::abs(mlx_rs::ops::subtract(&a, &b).unwrap()).unwrap(),
            None,
        )
        .unwrap()
        .item::<f32>();
        assert_eq!(diff_same, 0.0, "same seed must be bit-identical");

        let diff_seed = mlx_rs::ops::max(
            mlx_rs::ops::abs(mlx_rs::ops::subtract(&a, &c).unwrap()).unwrap(),
            None,
        )
        .unwrap()
        .item::<f32>();
        assert!(diff_seed > 0.0, "a different seed must change the latents");
    }
}
