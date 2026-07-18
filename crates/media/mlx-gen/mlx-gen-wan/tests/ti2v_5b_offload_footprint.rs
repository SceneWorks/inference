//! sc-12796 (epic 12732) — **real-Mac footprint + parity** for the Wan2.2 **TI2V-5B** (dense)
//! sequential component offload.
//!
//! Drives the public product path (`provider_registry().load(id, spec)` → `Generator::generate`) twice
//! over the SAME seeded request — once [`OffloadPolicy::Resident`] (the byte-identical pre-offload path)
//! and once [`OffloadPolicy::Sequential`] (the staged `clear_cache` flush) — and asserts the story's two
//! acceptance criteria on a Mac.
//!
//! **What differs from the A14B expert-swap harness ([`expert_swap_footprint`]):** the dense 5B has NO
//! experts to swap, and it **already stages** TE → DiT → z48 VAE (each `Weights`/encoder dropped by
//! scope, `eval`'d before the next loads). So its **active** high-water mark (`get_peak_memory`, which
//! tracks live-array bytes — NOT the buffer cache) is already bounded by the largest single stage under
//! BOTH policies — Sequential does **not** lower it. What Sequential changes is the **buffer cache**:
//! `clear_cache` returns each dead component's freed buffers to the OS immediately, instead of leaving
//! the ~11 GB f32 UMT5 TE (and, for TI2V, the VAE encoder) warm in MLX's cache — RSS / wired-memory
//! pressure — through denoise + decode. So the discriminating metric here is the **footprint**
//! (`get_active_memory() + get_cache_memory()`) sampled during the denoise steps, not the active peak.
//!
//! 1. **Denoise footprint drops by ~the dead TE.** The `Sequential` peak footprint (active + cache
//!    sampled across the denoise/decode callbacks) is meaningfully below the `Resident` one — the flushed
//!    TE (and VAE-encoder) are off-GPU during denoise instead of cache-resident.
//! 2. **Output parity preserved.** The two runs produce **bit-identical** frames (`clear_cache` changes
//!    residency/lifetime, never numerics). And the **active** peak is ≈ equal (staging already bounds it).
//!
//! Both the **T2V** (pure noise, no VAE encode) and **TI2V** (a `Reference` image → the z48 VAE-encoder
//! staged, then flushed) paths are covered.
//!
//! `#[ignore]` + env-gated (needs the converted TI2V-5B snapshot), GPU-heavy. Point
//! `WAN_TI2V_5B_MODEL_DIR` at a converted snapshot **tier** dir (q4 / q8 / bf16 — each has
//! `model.safetensors` + `t5_encoder.safetensors` + `vae.safetensors` + `config.json` +
//! `tokenizer.json`). The TE is f32-compute (~11 GB weights) regardless of the DiT tier, so the flush
//! win is ~the TE at every tier.
//!
//! ```text
//! WAN_TI2V_5B_MODEL_DIR=~/.cache/huggingface/hub/models--SceneWorks--wan2.2-ti2v-5b-mlx/snapshots/<hash>/q4 \
//!   cargo test -p mlx-gen-wan --test ti2v_5b_offload_footprint -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory,
};

use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Progress,
    WeightsSource,
};
use mlx_gen_wan::MODEL_ID;

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(|s| {
        let s = s.to_string_lossy();
        if let Some(rest) = s.strip_prefix("~/") {
            if let Some(home) = std::env::var_os("HOME") {
                return PathBuf::from(format!("{}/{rest}", home.to_string_lossy()));
            }
        }
        PathBuf::from(s.to_string())
    })
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// A small non-flat RGB gradient — the TI2V `Reference` (first-frame) conditioning image.
fn gradient_image(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 255 / w.max(1)) as u8);
            pixels.push((y * 255 / h.max(1)) as u8);
            pixels.push(128);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// One render's memory profile: the **active** peak (live-array high-water, `get_peak_memory`) and the
/// peak **footprint** (`active + cache` — the RSS proxy — sampled across the denoise/decode callbacks).
struct Profile {
    frames: Vec<Image>,
    peak_active: usize,
    peak_footprint: usize,
}

/// Run one seeded 5B generation under `policy` and return its [`Profile`]. `clear_cache` +
/// `reset_peak_memory` before the render so the measurements reflect only this generation. The
/// `on_progress` closure samples `active + cache` on every denoise/decode event — the window where the
/// dead TE (cache-resident under `Resident`, flushed under `Sequential`) is or isn't off-GPU.
fn render_and_measure(
    model_dir: &std::path::Path,
    policy: OffloadPolicy,
    conditioning: Vec<Conditioning>,
) -> Profile {
    let gen = mlx_gen_wan::provider_registry()
        .unwrap()
        .load(
            MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(model_dir.to_path_buf())).with_offload_policy(policy),
        )
        .unwrap_or_else(|e| panic!("load {MODEL_ID}: {e}"));

    // Small-but-real geometry so all four renders finish quickly. 480 = 15·32 is the on-grid MIN_SIZE
    // floor; 5 = 1 + 4·1 frames is one VAE temporal chunk. `unipc` is the native solver.
    let req = GenerationRequest {
        prompt: "a red fox trotting across a snowy meadow at sunrise, cinematic".into(),
        width: 480,
        height: 480,
        frames: Some(5),
        steps: Some(6),
        seed: Some(42),
        sampler: Some("unipc".into()),
        conditioning,
        ..Default::default()
    };
    gen.validate(&req).expect("validate");

    clear_cache();
    reset_peak_memory();
    let mut peak_footprint = 0usize;
    let mut on_progress = |p: Progress| {
        if matches!(p, Progress::Step { .. } | Progress::Decoding) {
            let f = get_active_memory() + get_cache_memory();
            if f > peak_footprint {
                peak_footprint = f;
            }
        }
    };
    let out = gen.generate(&req, &mut on_progress).expect("generate");
    let peak_active = get_peak_memory();

    let frames = match out {
        GenerationOutput::Video { frames, .. } => frames,
        other => panic!("expected Video, got {other:?}"),
    };
    Profile {
        frames,
        peak_active,
        peak_footprint,
    }
}

/// Assert the story's acceptance: (1) the `Sequential` denoise footprint (active + cache) drops
/// meaningfully below `Resident` (the dead TE/VAE flushed off-GPU), and (2) the frames are bit-identical
/// AND the active peak is ≈ equal (staging already bounds it; only the cache/RSS moves).
fn assert_offload_wins(label: &str, res: &Profile, seq: &Profile) {
    println!(
        "[{label}] Resident   active peak = {:.2} GiB",
        gib(res.peak_active)
    );
    println!(
        "[{label}] Sequential active peak = {:.2} GiB",
        gib(seq.peak_active)
    );
    println!(
        "[{label}] Resident   footprint (active+cache) peak = {:.2} GiB",
        gib(res.peak_footprint)
    );
    println!(
        "[{label}] Sequential footprint (active+cache) peak = {:.2} GiB",
        gib(seq.peak_footprint)
    );
    println!(
        "[{label}] footprint drop = {:.2} GiB ({:.0}% of the Resident footprint)",
        gib(res.peak_footprint.saturating_sub(seq.peak_footprint)),
        100.0 * (res.peak_footprint.saturating_sub(seq.peak_footprint)) as f64
            / res.peak_footprint as f64
    );

    // (2a) Parity — the flush only changes residency/lifetime, so frames must be bit-identical.
    assert_eq!(
        res.frames.len(),
        seq.frames.len(),
        "{label}: frame count differs across residency policies"
    );
    for (i, (r, s)) in res.frames.iter().zip(&seq.frames).enumerate() {
        assert_eq!(
            (r.width, r.height),
            (s.width, s.height),
            "{label}: frame {i} dims"
        );
        assert_eq!(
            r.pixels, s.pixels,
            "{label}: frame {i} differs between Resident and Sequential — the offload must be \
             numerics-preserving (same weights per timestep, only residency changes)"
        );
    }

    // (2b) Active peak ≈ equal — the dense 5B already stages, so `get_peak_memory` (live arrays) is
    // bounded by the largest single stage under both policies; `clear_cache` touches the cache, not the
    // active set. A small tolerance for allocator noise; Sequential must not be materially WORSE.
    assert!(
        seq.peak_active as f64 <= res.peak_active as f64 * 1.05,
        "{label}: Sequential active peak {:.2} GiB unexpectedly ABOVE Resident {:.2} GiB — the flush \
         should never raise the active high-water mark",
        gib(seq.peak_active),
        gib(res.peak_active)
    );

    // (1) Footprint — Sequential flushes the ~11 GB f32 TE (and, for TI2V, the VAE encoder) out of the
    // cache before/through denoise, so its active+cache peak must be meaningfully below Resident's.
    // Mutation guard: drop the `clear_cache` calls in `generate_impl` and the two footprints converge —
    // this assertion then fails. The floor (≥ 10% below) is conservative; the real drop is ~the TE.
    assert!(
        seq.peak_footprint < res.peak_footprint,
        "{label}: Sequential footprint {:.2} GiB must be below Resident {:.2} GiB (TE/VAE flushed)",
        gib(seq.peak_footprint),
        gib(res.peak_footprint)
    );
    assert!(
        (seq.peak_footprint as f64) <= 0.90 * (res.peak_footprint as f64),
        "{label}: Sequential footprint {:.2} GiB is only marginally below Resident {:.2} GiB — \
         expected ~the dead f32 TE (~11 GB) less; is the TE really `clear_cache`-flushed off-GPU?",
        gib(seq.peak_footprint),
        gib(res.peak_footprint)
    );
}

#[test]
#[ignore = "needs a converted Wan2.2-TI2V-5B snapshot tier (WAN_TI2V_5B_MODEL_DIR); GPU-heavy"]
fn t2v_5b_offload_drops_the_footprint_and_preserves_output() {
    let model_dir = match env_path("WAN_TI2V_5B_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_TI2V_5B_MODEL_DIR to a converted TI2V-5B snapshot tier dir");
            return;
        }
    };
    // Pure-noise T2V (no conditioning) — Stage 1b loads no VAE encoder, so the win is the dead TE alone.
    let res = render_and_measure(&model_dir, OffloadPolicy::Resident, vec![]);
    let seq = render_and_measure(&model_dir, OffloadPolicy::Sequential, vec![]);
    assert_offload_wins("t2v", &res, &seq);
}

#[test]
#[ignore = "needs a converted Wan2.2-TI2V-5B snapshot tier (WAN_TI2V_5B_MODEL_DIR); GPU-heavy"]
fn ti2v_5b_offload_drops_the_footprint_and_preserves_output() {
    let model_dir = match env_path("WAN_TI2V_5B_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_TI2V_5B_MODEL_DIR to a converted TI2V-5B snapshot tier dir");
            return;
        }
    };
    // TI2V mask-blend: a `Reference` image → the z48 VAE **encoder** is staged in Stage 1b, then flushed
    // under Sequential before the DiT loads (the second `clear_cache` site, on top of the TE flush).
    let reference = || {
        vec![Conditioning::Reference {
            image: gradient_image(128, 128),
            strength: None,
        }]
    };
    let res = render_and_measure(&model_dir, OffloadPolicy::Resident, reference());
    let seq = render_and_measure(&model_dir, OffloadPolicy::Sequential, reference());
    assert_offload_wins("ti2v", &res, &seq);
}
