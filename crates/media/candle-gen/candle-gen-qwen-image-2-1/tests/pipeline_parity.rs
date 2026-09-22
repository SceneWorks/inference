//! End-to-end parity vs the frozen `QwenImage21Pipeline` on the committed miniature snapshot
//! (`crates/media/mlx-gen/tools/dump_qwen21_pipeline.py`): from the upstream initial packed latents,
//! the per-step latents of the whole denoise loop and the decoded RGBA image — without guidance
//! (`t2i`), with true CFG (`cfg`), and for a non-square two-slot target (`wide`).
//!
//! Fixture and snapshot are the SAME ones the MLX twin reads.
//!
//! Tolerance: **1e-3 × peak** on the per-step latents and the `[0, 1]` RGBA image. The MLX twin
//! budgets 1e-2 for Metal's reduced-precision f32 matmul; candle CPU f32 against torch CPU f32 is
//! the same arithmetic on the same hardware, so this lane holds a tighter bar. The single-step
//! claims below keep any drift from compounding across steps. Every claim prints its measured
//! numbers.

use candle_gen::gen_core::{CancelFlag, Progress};
use candle_gen_qwen_image_2_1::{
    denoise, load_scheduler_config, load_text_encoder, load_tokenizer, load_transformer, load_vae,
    scheduler, system_prompt_drop_count, unpack_latents, DenoiseInputs,
};

use crate::common::{assert_close, device, tiny_snapshot, Fixture};

const LATENT_TOL: f32 = 1e-3;
const IMAGE_TOL: f32 = 1e-3;

#[test]
fn denoise_trajectory_and_rgba_image_match_upstream() {
    let w = Fixture::open("qwen21_pipeline.safetensors");
    let root = tiny_snapshot();
    let dev = device();
    let tokenizer = load_tokenizer(&root).unwrap();
    let encoder = load_text_encoder(&root, &dev).unwrap();
    let transformer = load_transformer(&root, &dev).unwrap();
    let vae = load_vae(&root, &dev).unwrap();
    let sched = load_scheduler_config(&root).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let prompt = w.meta("prompt").to_owned();
    let negative = w.meta("negative_prompt").to_owned();
    let pos = encoder.encode_prompt(&tokenizer, &prompt, drop).unwrap();
    let neg = encoder.encode_prompt(&tokenizer, &negative, drop).unwrap();

    for case in ["t2i", "cfg", "wide"] {
        let width = w.meta_usize(&format!("{case}/width")) as u32;
        let height = w.meta_usize(&format!("{case}/height")) as u32;
        let steps = w.meta_usize(&format!("{case}/steps"));
        let true_cfg = w
            .meta_opt(&format!("{case}/true_cfg_scale"))
            .map(|v| v.parse::<f32>().expect("a float"))
            .unwrap_or(1.0);
        let sigmas = scheduler::sigmas_for_image(&sched, steps, width, height).unwrap();
        let init = w.tensor(&format!("{case}/latents_init"));

        let mut seen_steps = Vec::new();
        let cancel = CancelFlag::new();
        let latents = denoise(
            DenoiseInputs {
                transformer: &transformer,
                sigmas: &sigmas,
                latents: init.clone(),
                prompt_embeds: &pos,
                negative_embeds: (true_cfg > 1.0).then_some(&neg),
                true_cfg_scale: true_cfg,
                width,
                height,
                sampler: None,
                seed: 0,
                cancel: &cancel,
            },
            &mut |p| {
                if let Progress::Step { current, total } = p {
                    seen_steps.push((current, total));
                }
            },
        )
        .unwrap();
        assert_eq!(
            seen_steps,
            (1..=steps as u32)
                .map(|c| (c, steps as u32))
                .collect::<Vec<_>>(),
            "{case}: progress"
        );
        assert_close(
            &format!("{case}/latents_after_step_{}", steps - 1),
            &latents,
            &w.tensor(&format!("{case}/latents_after_step_{}", steps - 1)),
            LATENT_TOL,
        );

        // Single-step claims: from upstream's own post-step latents, one Euler step must land on
        // upstream's next latents — so a divergence names the step instead of compounding.
        for i in 0..steps {
            let from = if i == 0 {
                init.clone()
            } else {
                w.tensor(&format!("{case}/latents_after_step_{}", i - 1))
            };
            let one = denoise(
                DenoiseInputs {
                    transformer: &transformer,
                    sigmas: &sigmas[i..i + 2],
                    latents: from,
                    prompt_embeds: &pos,
                    negative_embeds: (true_cfg > 1.0).then_some(&neg),
                    true_cfg_scale: true_cfg,
                    width,
                    height,
                    sampler: None,
                    seed: 0,
                    cancel: &cancel,
                },
                &mut |_| {},
            )
            .unwrap();
            assert_close(
                &format!("{case}/single_step_{i}"),
                &one,
                &w.tensor(&format!("{case}/latents_after_step_{i}")),
                LATENT_TOL,
            );
        }

        // RGBA decode of the final latents vs the upstream `[0, 1]` image.
        let unpacked = unpack_latents(&latents, width, height).unwrap();
        let rgba = vae
            .decode_rgba(&vae.denormalize(&unpacked).unwrap())
            .unwrap();
        let rgba01 = ((rgba * 0.5).unwrap() + 0.5).unwrap();
        assert_close(
            &format!("{case}/image_rgba"),
            &rgba01.squeeze(0).unwrap(),
            &w.tensor(&format!("{case}/image_rgba")),
            IMAGE_TOL,
        );
    }
}
