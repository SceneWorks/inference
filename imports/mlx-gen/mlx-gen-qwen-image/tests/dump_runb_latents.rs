//! Run B (`from_ldm`) latent dump for the PiD sc-7843 self-consistent validation.
//!
//! Generates the nightmarket prompt through the real Qwen-Image pipeline (1024², 50 steps, true-CFG
//! 4.0, seed 0 — the sc-7931 Run B params) and captures the in-progress `x_t` after steps 44/46/48
//! plus the final clean `x0`, exactly like the reference `from_ldm.py` `--save_xt_steps 44 46 48`.
//! For each capture it saves `tools/golden/pid/runb_<label>.safetensors` holding the **unpacked**
//! latent, its flow-match σ, and the Qwen-VAE 1024² decode (the baseline). The PiD side
//! (`mlx-gen-pid` `from_ldm_decode`) then super-resolves each latent to 4096² and the two are read as
//! the "fair pair" (Qwen-VAE 1024² vs PiD 4096² of the *same* latent).
//!
//! Two-process by design: this holds only the ~55 GB Qwen-Image transformer+TE+VAE; the PiD decode
//! (net + gemma) runs separately so the two large weight sets never coexist. `#[ignore]`d.
//!
//! ```sh
//! cargo test -p mlx-gen-qwen-image --release --test dump_runb_latents -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use mlx_gen::CancelFlag;
use mlx_gen_qwen_image::pipeline::encode_prompt;
use mlx_gen_qwen_image::{
    create_noise, denoise_with_progress, loader, qwen_scheduler, unpack_latents,
};
use mlx_rs::{Array, Dtype};

const PROMPT: &str = "a highly detailed photograph of a bustling night market food stall, glowing paper lanterns, a neon sign reading OPEN, steam rising from a wok, fresh vegetables, reflections on wet cobblestones, shallow depth of field, 35mm";
const W: u32 = 1024;
const H: u32 = 1024;
const STEPS: usize = 50;
const GUID: f32 = 4.0;
const SEED: u64 = 0;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

#[test]
#[ignore = "needs the Qwen/Qwen-Image snapshot"]
fn dump_runb_latents() {
    let root = snapshot();
    let tf = loader::load_transformer(&root).unwrap();
    let te = loader::load_text_encoder(&root).unwrap();
    let vae = loader::load_vae(&root).unwrap();
    let tok = loader::load_tokenizer(&root).unwrap();
    let pos = encode_prompt(&tok, &te, PROMPT, "qwen_image").unwrap();
    let neg = encode_prompt(&tok, &te, " ", "qwen_image").unwrap(); // Run B negative = single space

    let sigmas = qwen_scheduler(STEPS, W, H).sigmas; // length STEPS+1, trailing 0.0
    let cancel = CancelFlag::default();
    let out_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden/pid");

    // Segmented flow-match Euler: capturing x_t after step K == running the schedule slice
    // sigmas[cur..=K] from the current latent (deterministic ODE — no per-step noise), so the four
    // segments total exactly STEPS steps. The captured latent is at σ = sigmas[K].
    let mut latents = create_noise(SEED, W, H).unwrap(); // packed [1, seq, 64], f32
    let mut cur = 0usize;
    for k in [44usize, 46, 48, STEPS] {
        latents = denoise_with_progress(
            &tf,
            None,
            &sigmas[cur..=k],
            SEED,
            latents,
            &pos,
            Some(&neg),
            GUID,
            W,
            H,
            0,
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        cur = k;

        let label = if k == STEPS {
            "x0".to_string()
        } else {
            format!("{k:02}xt")
        };
        let sigma = sigmas[k];
        let unpacked = unpack_latents(&latents, W, H)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap(); // [1, 16, 128, 128]
        let vae_decoded = vae
            .decode(&unpacked)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let sig = Array::from_slice(&[sigma], &[1]);
        let path = format!("{out_dir}/runb_{label}.safetensors");
        Array::save_safetensors(
            vec![
                ("latent", &unpacked),
                ("sigma", &sig),
                ("vae_decoded", &vae_decoded),
            ],
            None,
            Path::new(&path),
        )
        .unwrap();
        eprintln!(
            "[{label}] sigma={sigma:.4} latent {:?} vae {:?} -> {path}",
            unpacked.shape(),
            vae_decoded.shape()
        );
    }
}
