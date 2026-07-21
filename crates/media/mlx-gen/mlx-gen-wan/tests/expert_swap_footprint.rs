//! sc-12736 (epic 12732) — **real-Mac footprint + parity** for the Wan2.2-A14B MoE expert swap.
//!
//! Drives the public product path (`provider_registry().load(id, spec)` → `Generator::generate`) twice
//! over the SAME seeded request — once [`OffloadPolicy::Resident`] (both ~8-9 GB experts co-resident,
//! the pre-swap path) and once [`OffloadPolicy::Sequential`] (the expert swap: only the ACTIVE expert
//! resident, TE/VAE freed off-GPU) — and asserts the two Pillar-1 acceptance criteria on a Mac:
//!
//! 1. **Footprint drops by ~one expert.** The `Sequential` MLX peak (`get_peak_memory`) is meaningfully
//!    below the `Resident` peak — the denoise stage holds one expert instead of two.
//! 2. **Output parity preserved.** The two runs produce **bit-identical** frames (same weights per
//!    timestep; only residency/lifetime changes).
//!
//! Both the **T2V** (0.875 boundary, no conditioning) and **I2V** (0.900 boundary, `y` channel-concat
//! from a Reference image) swap paths are covered — they share one `generate_impl` / `denoise_expert`,
//! so the two arms confirm the shared swap machinery on both boundaries.
//!
//! `#[ignore]` + env-gated (needs the converted A14B snapshot), GPU-heavy, two real renders per arm.
//! Point `WAN_A14B_MODEL_DIR` (T2V) / `WAN_I2V_MODEL_DIR` (I2V) at a converted snapshot **tier** dir
//! (bf16 / q8 / q4 — each has `low_noise_model.safetensors` + `high_noise_model.safetensors` +
//! `t5_encoder.safetensors` + `vae.safetensors` + `config.json` + `tokenizer.json`). The q4/q8 tiers
//! run fastest; the win scales with the tier's per-expert size (bf16 ≈ 27 GB, q8 ≈ 14 GB, q4 ≈ 8 GB).
//!
//! ```text
//! WAN_A14B_MODEL_DIR=/path/to/models--SceneWorks--wan2.2-t2v-a14b-mlx/snapshots/<hash>/q8 \
//! WAN_I2V_MODEL_DIR=/path/to/models--SceneWorks--wan2.2-i2v-a14b-mlx/snapshots/<hash>/q4 \
//!   cargo test -p mlx-gen-wan --test expert_swap_footprint -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Progress,
    WeightsSource,
};
use mlx_gen_wan::{MODEL_ID_I2V_14B, MODEL_ID_T2V_14B};

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

/// A small non-flat RGB gradient, the I2V Reference (first-frame) conditioning image.
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

/// Run one seeded A14B generation under `policy` and return `(frames, peak_bytes)`. `clear_cache` +
/// `reset_peak_memory` before the render so the measured peak reflects only this generation's residency.
/// `conditioning` is empty for T2V and a single `Reference` for I2V.
fn render_and_measure(
    model_dir: &std::path::Path,
    model_id: &str,
    policy: OffloadPolicy,
    sampler: &str,
    conditioning: Vec<Conditioning>,
) -> (Vec<Image>, usize) {
    let gen = mlx_gen_wan::provider_registry()
        .unwrap()
        .load(
            model_id,
            &LoadSpec::new(WeightsSource::Dir(model_dir.to_path_buf())).with_offload_policy(policy),
        )
        .unwrap_or_else(|e| panic!("load {model_id}: {e}"));

    // Small-but-real geometry so both runs finish quickly; `unipc` is the native (swap-eligible) solver.
    let req = GenerationRequest {
        prompt: "a red fox trotting across a snowy meadow at sunrise, cinematic".into(),
        width: 256,
        height: 256,
        frames: Some(5),
        steps: Some(6),
        seed: Some(42),
        sampler: Some(sampler.into()),
        conditioning,
        ..Default::default()
    };
    gen.validate(&req).expect("validate");

    clear_cache();
    reset_peak_memory();
    let mut noop = |_p: Progress| {};
    let out = gen.generate(&req, &mut noop).expect("generate");
    let peak = get_peak_memory();

    let frames = match out {
        GenerationOutput::Video { frames, .. } => frames,
        other => panic!("expected Video, got {other:?}"),
    };
    (frames, peak)
}

/// Assert the two Pillar-1 acceptance criteria: (1) the `Sequential` peak drops meaningfully below the
/// `Resident` peak (one expert vs two), and (2) the frames are bit-identical (numerics-preserving).
fn assert_swap_wins(label: &str, res: &(Vec<Image>, usize), seq: &(Vec<Image>, usize)) {
    let (res_frames, res_peak) = res;
    let (seq_frames, seq_peak) = seq;
    println!("[{label}] Resident   MLX peak = {:.2} GiB", gib(*res_peak));
    println!("[{label}] Sequential MLX peak = {:.2} GiB", gib(*seq_peak));
    println!(
        "[{label}] drop = {:.2} GiB ({:.0}% of the Resident peak)",
        gib(res_peak.saturating_sub(*seq_peak)),
        100.0 * (res_peak.saturating_sub(*seq_peak)) as f64 / *res_peak as f64
    );

    // (2) Parity — the swap only changes residency/lifetime, so the frames must be bit-identical.
    assert_eq!(
        res_frames.len(),
        seq_frames.len(),
        "{label}: frame count differs across residency policies"
    );
    for (i, (r, s)) in res_frames.iter().zip(seq_frames).enumerate() {
        assert_eq!(
            (r.width, r.height),
            (s.width, s.height),
            "{label}: frame {i} dims"
        );
        assert_eq!(
            r.pixels, s.pixels,
            "{label}: frame {i} differs between Resident and Sequential — the expert swap must be \
             numerics-preserving (same weights per timestep, only residency changes)"
        );
    }

    // (1) Footprint — Sequential holds one expert where Resident holds two, so its peak must be
    // meaningfully lower. A conservative floor (≥ 15% below); the real drop is ~one expert (larger the
    // fatter the tier — for q4 the ~10.6 GB TE-encode stage partially caps the Sequential peak).
    assert!(
        seq_peak < res_peak,
        "{label}: Sequential peak {:.2} GiB must be below Resident {:.2} GiB (one expert vs two)",
        gib(*seq_peak),
        gib(*res_peak)
    );
    assert!(
        (*seq_peak as f64) <= 0.85 * (*res_peak as f64),
        "{label}: Sequential peak {:.2} GiB is only marginally below Resident {:.2} GiB — expected \
         ~one expert less; is the inactive expert really evicted before the next loads?",
        gib(*seq_peak),
        gib(*res_peak)
    );
}

#[test]
#[ignore = "needs a converted Wan2.2-T2V-A14B snapshot tier (WAN_A14B_MODEL_DIR); GPU-heavy"]
fn t2v_expert_swap_drops_the_peak_and_preserves_output() {
    let model_dir = match env_path("WAN_A14B_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_A14B_MODEL_DIR to a converted T2V-A14B snapshot tier dir");
            return;
        }
    };
    // Resident first (both experts co-resident), then Sequential (the boundary-0.875 expert swap).
    let res = render_and_measure(
        &model_dir,
        MODEL_ID_T2V_14B,
        OffloadPolicy::Resident,
        "unipc",
        vec![],
    );
    let seq = render_and_measure(
        &model_dir,
        MODEL_ID_T2V_14B,
        OffloadPolicy::Sequential,
        "unipc",
        vec![],
    );
    assert_swap_wins("t2v", &res, &seq);
}

#[test]
#[ignore = "needs a converted Wan2.2-T2V-A14B snapshot tier (WAN_A14B_MODEL_DIR); GPU-heavy"]
fn t2v_curated_heun_swap_drops_the_peak_and_preserves_output() {
    let model_dir = match env_path("WAN_A14B_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_A14B_MODEL_DIR to a converted T2V-A14B snapshot tier dir");
            return;
        }
    };
    // Heun evaluates twice per non-terminal step. A boundary-straddling step therefore proves that
    // the swap is routed per evaluation, not naively once per outer step.
    let res = render_and_measure(
        &model_dir,
        MODEL_ID_T2V_14B,
        OffloadPolicy::Resident,
        "heun",
        vec![],
    );
    let seq = render_and_measure(
        &model_dir,
        MODEL_ID_T2V_14B,
        OffloadPolicy::Sequential,
        "heun",
        vec![],
    );
    assert_swap_wins("t2v-curated-heun", &res, &seq);
}

#[test]
#[ignore = "needs a converted Wan2.2-I2V-A14B snapshot tier (WAN_I2V_MODEL_DIR); GPU-heavy"]
fn i2v_expert_swap_drops_the_peak_and_preserves_output() {
    let model_dir = match env_path("WAN_I2V_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_I2V_MODEL_DIR to a converted I2V-A14B snapshot tier dir");
            return;
        }
    };
    // I2V requires a Reference conditioning image (the first frame → the `y` channel-concat), and it
    // exercises the 0.900 boundary + the VAE-encoder staging under `Sequential`.
    let reference = || {
        vec![Conditioning::Reference {
            image: gradient_image(128, 128),
            strength: None,
        }]
    };
    let res = render_and_measure(
        &model_dir,
        MODEL_ID_I2V_14B,
        OffloadPolicy::Resident,
        "unipc",
        reference(),
    );
    let seq = render_and_measure(
        &model_dir,
        MODEL_ID_I2V_14B,
        OffloadPolicy::Sequential,
        "unipc",
        reference(),
    );
    assert_swap_wins("i2v", &res, &seq);
}
