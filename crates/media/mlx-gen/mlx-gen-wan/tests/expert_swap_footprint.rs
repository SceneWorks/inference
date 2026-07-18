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
//! `#[ignore]` + env-gated (needs the converted A14B snapshot), GPU-heavy, one real render per policy.
//! Point `WAN_A14B_MODEL_DIR` at a converted snapshot **tier** dir (bf16 / q8 / q4 — each has
//! `low_noise_model.safetensors` + `high_noise_model.safetensors` + `t5_encoder.safetensors` +
//! `vae.safetensors` + `config.json` + `tokenizer.json`). The q4/q8 tiers run fastest; the win scales
//! with the tier's per-expert size (bf16 ≈ 27 GB, q8 ≈ 14 GB, q4 ≈ 8 GB).
//!
//! ```text
//! WAN_A14B_MODEL_DIR=~/.cache/huggingface/hub/models--SceneWorks--wan2.2-t2v-a14b-mlx/snapshots/<hash>/q8 \
//!   cargo test -p mlx-gen-wan --test expert_swap_footprint -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Progress, WeightsSource,
};
use mlx_gen_wan::MODEL_ID_T2V_14B;

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

/// Run one seeded T2V-A14B generation under `policy` and return `(frames, peak_bytes)`. `reset` +
/// `clear_cache` before each run so the measured peak reflects only this generation's residency.
fn render_and_measure(model_dir: &std::path::Path, policy: OffloadPolicy) -> (Vec<Image>, usize) {
    let gen = mlx_gen_wan::provider_registry()
        .unwrap()
        .load(
            MODEL_ID_T2V_14B,
            &LoadSpec::new(WeightsSource::Dir(model_dir.to_path_buf())).with_offload_policy(policy),
        )
        .expect("load wan2_2_t2v_14b");

    // Small-but-real geometry so both runs finish quickly; `unipc` is the native (swap-eligible) solver.
    let req = GenerationRequest {
        prompt: "a red fox trotting across a snowy meadow at sunrise, cinematic".into(),
        width: 256,
        height: 256,
        frames: Some(5),
        steps: Some(6),
        seed: Some(42),
        sampler: Some("unipc".into()),
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

#[test]
#[ignore = "needs a converted Wan2.2-T2V-A14B snapshot tier (WAN_A14B_MODEL_DIR); GPU-heavy"]
fn expert_swap_drops_the_peak_and_preserves_output() {
    let model_dir = match env_path("WAN_A14B_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_A14B_MODEL_DIR to a converted A14B snapshot tier dir");
            return;
        }
    };

    // Resident first (both experts co-resident), then Sequential (the expert swap).
    let (res_frames, res_peak) = render_and_measure(&model_dir, OffloadPolicy::Resident);
    let (seq_frames, seq_peak) = render_and_measure(&model_dir, OffloadPolicy::Sequential);

    println!("[footprint] Resident   MLX peak = {:.2} GiB", gib(res_peak));
    println!("[footprint] Sequential MLX peak = {:.2} GiB", gib(seq_peak));
    println!(
        "[footprint] drop = {:.2} GiB ({:.0}% of the Resident peak)",
        gib(res_peak.saturating_sub(seq_peak)),
        100.0 * (res_peak.saturating_sub(seq_peak)) as f64 / res_peak as f64
    );

    // (2) Parity: the swap only changes residency/lifetime — the frames must be bit-identical.
    assert_eq!(
        res_frames.len(),
        seq_frames.len(),
        "frame count differs across residency policies"
    );
    for (i, (r, s)) in res_frames.iter().zip(&seq_frames).enumerate() {
        assert_eq!((r.width, r.height), (s.width, s.height), "frame {i} dims");
        assert_eq!(
            r.pixels, s.pixels,
            "frame {i} differs between Resident and Sequential — the expert swap must be \
             numerics-preserving (same weights per timestep, only residency changes)"
        );
    }

    // (1) Footprint: Sequential holds one expert where Resident holds two, so its peak must be
    // meaningfully lower. A conservative floor (≥ 15% below) — the real drop is ~one expert.
    assert!(
        seq_peak < res_peak,
        "Sequential peak {:.2} GiB must be below the Resident peak {:.2} GiB (one expert vs two)",
        gib(seq_peak),
        gib(res_peak)
    );
    assert!(
        (seq_peak as f64) <= 0.85 * (res_peak as f64),
        "Sequential peak {:.2} GiB is only marginally below Resident {:.2} GiB — expected ~one \
         expert less; is the inactive expert really evicted before the next loads?",
        gib(seq_peak),
        gib(res_peak)
    );
}
