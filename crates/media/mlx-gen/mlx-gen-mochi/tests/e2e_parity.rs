//! A4 end-to-end text-to-video parity for Mochi 1 (sc-11988) vs the A1 e2e golden.
//!
//! An **`#[ignore]`d** real-weight test that runs the [`mlx_gen_mochi::pipeline::denoise`] CFG loop
//! from the reference's own step-0 init latents and checks the post-2-step latent against
//! `mochi_e2e_golden.final_latents [1, 12, 2, 8, 8]`.
//!
//! **Why teacher-force the init latents (RNG-independent, no new golden):** MLX's RNG is not portable
//! to the reference's `torch.Generator(1984)`, so a fresh full-pipeline run cannot reproduce the exact
//! seeded init noise the golden was denoised from. But that init noise is *already captured*: the
//! `dit_golden.hidden_states [2, 12, 2, 8, 8]` is the step-0 `latent_model_input = cat([latents]*2)`,
//! and flow-match `init_noise_sigma = 1` means it is the un-scaled seeded latent. Both CFG branches are
//! the same latent at step 0, so `hidden_states[0:1]` **is** the init noise (asserted below). The text
//! conditioning is teacher-forced from the same golden (`encoder_hidden_states`/`_attention_mask`) so
//! the gate isolates the DiT + scheduler + CFG loop (T5 reproduction is gated by `te_parity`).
//!
//! **Tolerance — why the final-latent residual is ~2.5e-1, not the naive ~1e-1 (this is NOT a bug):**
//! the STEPS=2 fixture schedule has a **huge final step** (`sigmas = [0, 0.025, 1.0]` ⇒ `dt₁ = 0.975`),
//! which makes the trajectory **chaos-amplifying**: the `chaos_band_*` measurement below perturbs the
//! init by a bf16-magnitude (~3e-3) relative delta and observes a ~2e-1 final divergence (condition
//! number ≈ 65×). The port runs f32 while the golden is a bf16 2-step trajectory; the *irreducible*
//! f32-vs-bf16 per-forward residual is the `dit_parity` floor (~5.85e-2, re-asserted here as the tight
//! regression guard), and the coarse schedule amplifies it into the observed final divergence. So the
//! final-latent gate is set to the **actual measured residual** (not loosened blindly) and additionally
//! bounded to lie **within the schedule's own precision-sensitivity band** — a real structural bug
//! would push the divergence outside that band or break the tight per-forward guard. (A *tight* e2e
//! would need a bit-exact bf16 per-forward — the LTX sc-2842 pattern — which requires the A3
//! transformer to run bf16 activations; out of A4 scope, tracked as a follow-on if desired.)
//!
//! Run: `MOCHI_SNAPSHOT=/path/to/mochi-1-preview cargo test -p mlx-gen-mochi --test e2e_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, add, max, mean, multiply, split, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_mochi::pipeline::{decode_to_frames, denoise, to_uint8_frames};
use mlx_gen_mochi::{
    load_transformer_weights, load_vae_decoder, MochiDitConfig, MochiScheduler,
    MochiTransformer3DModel,
};

const DIT_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/mochi_dit_golden.safetensors"
);
const E2E_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/mochi_e2e_golden.safetensors"
);

fn snapshot_dir() -> std::path::PathBuf {
    std::env::var("MOCHI_SNAPSHOT")
        .expect("set MOCHI_SNAPSHOT to the mochi-1-preview snapshot dir")
        .into()
}

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}

/// `max|got − want| / max|want|` — peak relative error (the repo convention).
fn peak_rel(got: &Array, want: &Array) -> f32 {
    max_abs(&subtract(&f32(got), &f32(want)).unwrap()) / max_abs(&f32(want)).max(1e-12)
}

/// `mean|got − want| / mean|want|` — mean relative error.
fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = mean(abs(subtract(&f32(got), &f32(want)).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let den = mean(abs(&f32(want)).unwrap(), None).unwrap().item::<f32>();
    num / den.max(1e-12)
}

fn mean_abs(a: &Array) -> f32 {
    mean(abs(&f32(a)).unwrap(), None).unwrap().item::<f32>()
}

/// Fraction of uint8 pixels differing by > 8.
fn px_gt8(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(&f32(got), &f32(want)).unwrap()).unwrap();
    let over = mlx_rs::ops::gt(&diff, Array::from_int(8)).unwrap();
    mlx_rs::ops::sum(&f32(&over), None).unwrap().item::<f32>() / (got.size() as f32)
}

#[test]
#[ignore = "needs $MOCHI_SNAPSHOT (bf16 DiT + VAE shards) + tools/golden/mochi_{dit,e2e}_golden.safetensors"]
fn e2e_final_latents_match_golden() {
    let root = snapshot_dir();
    let dit_g = Weights::from_file(DIT_GOLDEN).expect("dit golden");
    let e2e_g = Weights::from_file(E2E_GOLDEN).expect("e2e golden");

    // Fixture point rides on the e2e golden meta: geometry = [H, W, FRAMES, STEPS, MAXSEQ], guidance.
    let geom = e2e_g
        .require("geometry")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let geom: Vec<i32> = geom.as_slice::<i32>().to_vec();
    let steps = geom[3] as usize;
    let guidance = e2e_g
        .require("guidance")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let guidance = guidance.as_slice::<f32>()[0];
    eprintln!("e2e fixture: geometry={geom:?} steps={steps} guidance={guidance}");

    // --- VERIFY the init-latent teacher-forcing assumption (STOP if it fails; do not fabricate). ---
    // dit_golden.hidden_states [2, 12, 2, 8, 8] = step-0 latent_model_input = cat([latents]*2). Both
    // CFG halves must be identical (they are the same seeded init noise at step 0, init_noise_sigma=1).
    let hidden = dit_g.require("hidden_states").unwrap();
    assert_eq!(hidden.shape(), &[2, 12, 2, 8, 8], "dit hidden_states shape");
    let halves = split(hidden, 2, 0).unwrap();
    let branch_delta = max_abs(&subtract(&f32(&halves[0]), &f32(&halves[1])).unwrap());
    eprintln!("init-latent assumption: max|branch0 - branch1| = {branch_delta:.3e} (must be 0)");
    assert!(
        branch_delta == 0.0,
        "init-latent teacher-forcing INVALID: the two CFG branches of dit_golden.hidden_states \
         differ by {branch_delta:.3e} at step 0 — halves[0:1] is not the seeded init noise. STOP: \
         adding an explicit `init_latents` tensor to the e2e golden is an oracle change needing sign-off."
    );
    let init = halves[0].clone(); // [1, 12, 2, 8, 8] — the seeded init noise.

    // Teacher-force the text conditioning from the same step-0 capture (T5 reproduction is gated by
    // te_parity; here we isolate the DiT + scheduler + CFG loop).
    let enc = dit_g.require("encoder_hidden_states").unwrap(); // [2, 256, 4096]
    let enc_mask = dit_g.require("encoder_attention_mask").unwrap(); // [2, 256]

    // Cross-check the scheduler timestep against the reference's step-0 capture (parity sanity).
    let mut sched = MochiScheduler::new();
    sched.set_timesteps(steps, 1.0);
    let ts: Vec<f32> = sched.timesteps().to_vec();
    let g_ts0 = dit_g
        .require("timestep")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let g_ts0 = g_ts0.as_slice::<f32>()[0];
    eprintln!("scheduler timesteps={ts:?}  golden timestep[0]={g_ts0:.6}");
    assert!(
        (ts[0] - g_ts0).abs() < 1e-2,
        "scheduler step-0 timestep {} disagrees with the golden {g_ts0}",
        ts[0]
    );

    // Load the real DiT (f32 compute — the dit_parity regime).
    let dit = MochiTransformer3DModel::from_weights(
        &load_transformer_weights(&root).expect("load DiT weights"),
        &MochiDitConfig::default(),
        Dtype::Float32,
    )
    .expect("build DiT");

    // --- TIGHT regression guard: the per-forward (the reducible quantity) is at the dit_parity floor.
    // This is what actually protects against a real structural bug (the chaos-amplified final gate
    // below cannot). Runs the transformer through the pipeline's inputs at step 0.
    let np0 = f32(&dit.forward(hidden, enc, dit_g.require("timestep").unwrap(), enc_mask).unwrap());
    let np0_pr = peak_rel(&np0, dit_g.require("noise_pred").unwrap());
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
        enc,
        enc_mask,
        steps,
        guidance,
        1.0, // Mochi resolution shift = 1 (no shift)
        &CancelFlag::default(),
        &mut |p| {
            if let mlx_gen::Progress::Step { current, total } = p {
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
    assert_eq!(latents.shape(), want.shape(), "final_latents shape");
    let pr = peak_rel(&latents, want);
    let mr = mean_rel(&latents, want);
    eprintln!("E2E final_latents peak_rel = {pr:.3e}  mean_rel = {mr:.3e}");

    // --- Measure the schedule's precision-sensitivity band, so the loose final tolerance is
    // self-justifying (not a magic constant): perturb the init by a bf16-magnitude (~3e-3) relative
    // delta and rerun the SAME f32 loop; the induced final divergence is the chaos floor for this
    // fixture. The port's f32-vs-bf16 golden divergence must sit within a small multiple of it.
    let n = init.size() as i32;
    let noise: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 12.9898).sin() * 43758.5453).fract() * 2.0 - 1.0)
        .collect();
    let noise = Array::from_slice(&noise, init.shape());
    let pert = add(
        &init,
        &multiply(&noise, Array::from_f32(3e-3 * mean_abs(&init))).unwrap(),
    )
    .unwrap();
    let latents_pert = denoise(
        &dit, &pert, enc, enc_mask, steps, guidance, 1.0, &CancelFlag::default(), &mut |_| {},
    )
    .expect("denoise (perturbed)");
    let chaos_band = mean_rel(&latents_pert, &latents);
    eprintln!(
        "chaos_band: a 3e-3 init perturbation moves the final by mean_rel={chaos_band:.3e} \
         (condition number ~{:.0}×)",
        chaos_band / 3e-3
    );

    // Gate on the ACTUAL measured residual (documented above), AND require it to lie within the
    // schedule's own precision-sensitivity band (a real bug escapes the band).
    assert!(
        mr < 3.0e-1 && pr < 3.0e-1,
        "E2E final residual peak_rel {pr:.3e} / mean_rel {mr:.3e} exceeds the chaos-amplified budget"
    );
    assert!(
        mr <= 2.0 * chaos_band.max(2e-2),
        "E2E final mean_rel {mr:.3e} exceeds 2× the schedule's precision-sensitivity band \
         {chaos_band:.3e} — that is beyond chaos and points to a structural regression"
    );

    // --- Optional: exercise the full decode path; report the decoded-pixel delta (informational).
    // The 64×64 / 2-latent-frame fixture is a parity fixture (not a real 480p gen) and the latent
    // residual is chaos-driven, so the pixel delta is reported, gated only against total garbage.
    let vae = load_vae_decoder(&root).expect("load VAE");
    let got_frames = decode_to_frames(&vae, &latents, &CancelFlag::default()).expect("decode");
    let want_frames = to_uint8_frames(e2e_g.require("video").unwrap()).expect("golden video -> u8");
    assert_eq!(got_frames.shape(), want_frames.shape(), "frame shape");
    let px = px_gt8(&got_frames, &want_frames);
    eprintln!("E2E decoded frames px>8 = {:.2}%  (informational; chaos-driven)", px * 100.0);
    assert!(px < 0.9, "decoded frames are near-total garbage (px>8 = {:.1}%)", px * 100.0);
}
