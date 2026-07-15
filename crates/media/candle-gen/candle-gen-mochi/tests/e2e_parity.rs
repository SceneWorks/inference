//! **Real-weight CUDA** end-to-end text-to-video parity for Mochi 1 (A5, sc-11989) — the candle twin
//! of `mlx-gen-mochi`'s ignored `e2e_parity` gate. Gated on `feature = "cuda"` + `#[ignore]`d. Runs the
//! [`candle_gen_mochi::denoise`] CFG loop from the reference's own step-0 init latents and checks the
//! post-2-step latent against `mochi_e2e_golden.final_latents [1, 12, 2, 8, 8]`.
//!
//! **Teacher-forced init (RNG-independent):** the reference's `torch.Generator` RNG is not portable, so
//! the init noise is recovered from `dit_golden.hidden_states [2, 12, 2, 8, 8]` — the step-0
//! `latent_model_input = cat([latents]*2)`, and flow-match `init_noise_sigma = 1` means it is the
//! un-scaled seeded latent. Both CFG branches are the same latent at step 0 (asserted below), so
//! `hidden_states[0:1]` **is** the init noise. The text conditioning is teacher-forced from the same
//! golden so the gate isolates the DiT + scheduler + CFG loop (T5 is gated by `te_parity`).
//!
//! **Tolerance:** the STEPS=2 fixture schedule has a huge final step (`sigmas = [0, 0.025, 1.0]` ⇒
//! `dt₁ = 0.975`), which makes the trajectory chaos-amplifying — the port's f32-vs-bf16 per-forward
//! residual (the `dit_parity` floor) is amplified into the observed final divergence. The final gate is
//! bounded to the schedule's own precision-sensitivity band (measured here), plus the tight per-forward
//! guard; a real structural bug escapes the band or breaks the guard.
//!
//! Windows run:
//!   `MOCHI_SNAPSHOT=/path/to/mochi-1-preview cargo test -p candle-gen-mochi --features cuda --test e2e_parity -- --ignored --nocapture`
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::CancelFlag;
use candle_gen::Weights;
use candle_gen_mochi::{
    denoise, load_transformer_var_builder, MochiDitConfig, MochiScheduler, MochiTransformer3DModel,
    DIT_DTYPE,
};

const DIT_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/tools/golden/mochi_dit_golden.safetensors"
);
const E2E_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../mlx-gen/tools/golden/mochi_e2e_golden.safetensors"
);

fn snapshot_dir() -> PathBuf {
    std::env::var("MOCHI_SNAPSHOT")
        .expect("set MOCHI_SNAPSHOT to the mochi-1-preview snapshot dir")
        .into()
}

fn max_abs(t: &Tensor) -> f32 {
    t.abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}
fn mean_abs(t: &Tensor) -> f32 {
    t.abs()
        .unwrap()
        .mean_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}
fn peak_rel(got: &Tensor, want: &Tensor) -> f32 {
    let got = got.to_dtype(DType::F32).unwrap();
    let want = want.to_dtype(DType::F32).unwrap();
    max_abs(&(&got - &want).unwrap()) / max_abs(&want).max(1e-12)
}
fn mean_rel(got: &Tensor, want: &Tensor) -> f32 {
    let got = got.to_dtype(DType::F32).unwrap();
    let want = want.to_dtype(DType::F32).unwrap();
    mean_abs(&(&got - &want).unwrap()) / mean_abs(&want).max(1e-12)
}

#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (bf16 DiT shards) + tools/golden/mochi_{dit,e2e}_golden.safetensors (CUDA)"]
fn e2e_final_latents_match_golden() {
    let device = candle_gen::default_device().unwrap();
    let root = snapshot_dir();
    let dit_g = Weights::from_file(Path::new(DIT_GOLDEN), &device, DType::F32).expect("dit golden");
    let e2e_g = Weights::from_file(Path::new(E2E_GOLDEN), &device, DType::F32).expect("e2e golden");

    // Fixture point rides on the e2e golden meta: geometry = [H, W, FRAMES, STEPS, MAXSEQ], guidance.
    let geom = e2e_g
        .require("geometry")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let geom: Vec<f32> = geom.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let steps = geom[3] as usize;
    let guidance = e2e_g
        .require("guidance")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()[0];
    eprintln!("e2e fixture: geometry={geom:?} steps={steps} guidance={guidance}");

    // --- VERIFY the init-latent teacher-forcing assumption (STOP if it fails; do not fabricate). ---
    let hidden = dit_g.require("hidden_states").unwrap(); // [2, 12, 2, 8, 8]
    assert_eq!(hidden.dims(), &[2, 12, 2, 8, 8], "dit hidden_states shape");
    let h0 = hidden.narrow(0, 0, 1).unwrap();
    let h1 = hidden.narrow(0, 1, 1).unwrap();
    let branch_delta = max_abs(&(&h0 - &h1).unwrap());
    eprintln!("init-latent assumption: max|branch0 - branch1| = {branch_delta:.3e} (must be 0)");
    assert!(
        branch_delta == 0.0,
        "init-latent teacher-forcing INVALID: the two CFG branches differ by {branch_delta:.3e} — \
         halves[0:1] is not the seeded init noise. STOP: adding an explicit init_latents tensor is an \
         oracle change needing sign-off."
    );
    let init = h0.contiguous().unwrap(); // [1, 12, 2, 8, 8] — the seeded init noise.

    // Teacher-force the text conditioning from the same step-0 capture (T5 is gated by te_parity).
    let enc = dit_g.require("encoder_hidden_states").unwrap(); // [2, 256, 4096]
    let enc_mask = dit_g.require("encoder_attention_mask").unwrap(); // [2, 256]

    // Cross-check the scheduler timestep against the reference's step-0 capture.
    let mut sched = MochiScheduler::new();
    sched.set_timesteps(steps, 1.0);
    let ts0 = sched.timesteps()[0];
    let g_ts0 = dit_g
        .require("timestep")
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()[0];
    eprintln!("scheduler timesteps[0]={ts0:.6}  golden timestep[0]={g_ts0:.6}");
    assert!(
        (ts0 - g_ts0).abs() < 1e-2,
        "scheduler step-0 timestep {ts0} disagrees with {g_ts0}"
    );

    // Load the real DiT (f32-activation — the dit_parity regime).
    let dit = MochiTransformer3DModel::new(
        load_transformer_var_builder(&root, DIT_DTYPE, &device).expect("load DiT weights"),
        &MochiDitConfig::default(),
        &device,
    )
    .expect("build DiT");

    // --- TIGHT regression guard: the per-forward (the reducible quantity) is at the dit_parity floor. ---
    let np0 = dit
        .forward(
            &hidden,
            &enc,
            &dit_g.require("timestep").unwrap(),
            &enc_mask,
        )
        .unwrap();
    let np0_pr = peak_rel(&np0, &dit_g.require("noise_pred").unwrap());
    eprintln!("step-0 per-forward noise_pred peak_rel = {np0_pr:.3e} (dit_parity floor)");
    assert!(
        np0_pr < 1.0e-1,
        "per-forward regressed: step-0 noise_pred peak_rel {np0_pr:.3e} exceeds the dit_parity floor"
    );

    // --- Run the actual pipeline denoise loop (the thing under test). ---
    let mut steps_seen: Vec<u32> = Vec::new();
    let latents = denoise(
        &dit,
        &init,
        &enc,
        &enc_mask,
        steps,
        guidance,
        1.0, // Mochi resolution shift = 1
        &CancelFlag::default(),
        &mut |p| {
            if let candle_gen::gen_core::Progress::Step { current, total } = p {
                assert_eq!(total, steps as u32, "progress total");
                steps_seen.push(current);
            }
        },
    )
    .expect("denoise");
    assert_eq!(
        steps_seen,
        (1..=steps as u32).collect::<Vec<_>>(),
        "progress must be monotone 1..=steps"
    );

    let want = e2e_g.require("final_latents").unwrap(); // [1, 12, 2, 8, 8]
    assert_eq!(latents.dims(), want.dims(), "final_latents shape");
    let pr = peak_rel(&latents, &want);
    let mr = mean_rel(&latents, &want);
    eprintln!("E2E final_latents peak_rel = {pr:.3e}  mean_rel = {mr:.3e}");

    // --- Measure the schedule's precision-sensitivity band (self-justifying loose tolerance). ---
    let n = init.elem_count();
    let noise: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 12.9898).sin() * 43_758.547).fract() * 2.0 - 1.0)
        .collect();
    let noise = Tensor::from_vec(noise, init.dims(), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let pert = (&init + noise.affine((3e-3 * mean_abs(&init)) as f64, 0.0).unwrap()).unwrap();
    let latents_pert = denoise(
        &dit,
        &pert,
        &enc,
        &enc_mask,
        steps,
        guidance,
        1.0,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .expect("denoise (perturbed)");
    let chaos_band = mean_rel(&latents_pert, &latents);
    eprintln!(
        "chaos_band: a 3e-3 init perturbation moves the final by mean_rel={chaos_band:.3e} \
         (condition number ~{:.0}×)",
        chaos_band / 3e-3
    );

    // Gate on the residual AND require it within the schedule's own precision-sensitivity band.
    assert!(
        mr < 3.0e-1 && pr < 3.0e-1,
        "E2E final residual peak_rel {pr:.3e} / mean_rel {mr:.3e} exceeds the chaos-amplified budget"
    );
    assert!(
        mr <= 2.0 * chaos_band.max(2e-2),
        "E2E final mean_rel {mr:.3e} exceeds 2× the schedule's precision-sensitivity band \
         {chaos_band:.3e} — beyond chaos, pointing to a structural regression"
    );
}
