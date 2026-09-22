//! End-to-end parity vs the frozen `QwenImage21Pipeline` on the committed miniature snapshot
//! (`tools/dump_qwen21_pipeline.py`): from the upstream initial packed latents, the per-step
//! latents of the whole denoise loop and the decoded RGBA image — without guidance (`t2i`), with
//! true CFG (`cfg`), and for a non-square two-slot target (`wide`).
//!
//! Tolerance: **1e-2 × peak** on the per-step latents and the `[0, 1]` RGBA image — the
//! repository's bound for f32 Metal matmul chains vs f32 CPU torch (measured on this fixture:
//! ≤ 5e-3 × peak after three Euler steps through the two-block DiT, ≤ 3e-3 on the image). The
//! single-step claims below keep any drift from compounding across steps.

use mlx_gen::gen_core::Progress;
use mlx_gen::CancelFlag;
use mlx_gen_qwen_image_2_1::{
    denoise, load_scheduler_config, load_text_encoder, load_tokenizer, load_transformer, load_vae,
    scheduler, system_prompt_drop_count, unpack_latents, DenoiseInputs,
};
use mlx_rs::ops::{add, multiply};
use mlx_rs::Array;

use crate::common::{assert_close, fixture, meta_f32, meta_str, meta_usize, tiny_snapshot};

const LATENT_TOL: f32 = 1e-2;
const IMAGE_TOL: f32 = 1e-2;

#[test]
fn denoise_trajectory_and_rgba_image_match_upstream() {
    let w = fixture("qwen21_pipeline.safetensors");
    let root = tiny_snapshot();
    let tokenizer = load_tokenizer(&root).unwrap();
    let encoder = load_text_encoder(&root).unwrap();
    let transformer = load_transformer(&root).unwrap();
    let vae = load_vae(&root).unwrap();
    let sched = load_scheduler_config(&root).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let prompt = meta_str(&w, "prompt");
    let negative = meta_str(&w, "negative_prompt");
    let pos = encoder.encode_prompt(&tokenizer, prompt, drop).unwrap();
    let neg = encoder.encode_prompt(&tokenizer, negative, drop).unwrap();

    for case in ["t2i", "cfg", "wide"] {
        let width = meta_usize(&w, &format!("{case}/width")) as u32;
        let height = meta_usize(&w, &format!("{case}/height")) as u32;
        let steps = meta_usize(&w, &format!("{case}/steps"));
        let true_cfg = w
            .metadata(&format!("{case}/true_cfg_scale"))
            .map(|_| meta_f32(&w, &format!("{case}/true_cfg_scale")))
            .unwrap_or(1.0);
        let sigmas = scheduler::sigmas_for_image(&sched, steps, width, height).unwrap();
        let init = w.require(&format!("{case}/latents_init")).unwrap().clone();

        // Every step's latent: run the loop one step at a time from the upstream post-step
        // latents so a divergence localises to a single step rather than compounding.
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
            w.require(&format!("{case}/latents_after_step_{}", steps - 1))
                .unwrap(),
            LATENT_TOL,
        );
        // Single-step claims: from upstream's own post-step latents, one Euler step must land on
        // upstream's next latents — so a divergence names the step instead of compounding.
        for i in 0..steps {
            let from = if i == 0 {
                init.clone()
            } else {
                w.require(&format!("{case}/latents_after_step_{}", i - 1))
                    .unwrap()
                    .clone()
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
                w.require(&format!("{case}/latents_after_step_{i}"))
                    .unwrap(),
                LATENT_TOL,
            );
        }

        // RGBA decode of the final latents vs the upstream `[0, 1]` image.
        let unpacked = unpack_latents(&latents, width, height).unwrap();
        let rgba = vae
            .decode_rgba(&vae.denormalize(&unpacked).unwrap())
            .unwrap();
        let half = Array::from_f32(0.5);
        let rgba01 = add(multiply(&rgba, &half).unwrap(), &half).unwrap();
        let want = w.require(&format!("{case}/image_rgba")).unwrap();
        assert_close(
            &format!("{case}/image_rgba"),
            &rgba01.squeeze_axes(&[0]).unwrap(),
            want,
            IMAGE_TOL,
        );
    }
}
