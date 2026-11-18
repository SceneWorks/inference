//! Perturbed-Attention Guidance (PAG) synthetic-DiT proof for MiniMax-H3-Image (SW-F371).
//!
//! Builds complete synthetic 2-block DiTs across dense, Q8, and Q4 tiers with LoRA/LoKr, AdaLN
//! eviction, and streamed rung-4 execution, so it is gated from ordinary CI, registered in
//! `scripts/ci/model-inference-proofs.json`, and run only by the explicit model-proof workflow.

use super::*;
use crate::image_pipeline::{denoise_image_with_pag, initial_image_rows, ImageLayout};
use candle_gen::gen_core::h3_image::{minimax_h3_image_pag_capability, H3ImageSampler};
use candle_gen::gen_core::{AdapterKind, AdapterSpec, CancelFlag, PagConfig};
use gen_core_h3::h3_schedule::SigmaSchedule;

fn config() -> MiniMaxH3DitConfig {
    MiniMaxH3DitConfig {
        num_attention_heads: 4,
        attention_head_dim: 24,
        hidden_size: 64,
        num_layers: 2,
        num_refiner_layers: 2,
        ffn_dim: 32,
        text_dim: 40,
        freq_dim: 16,
        time_embed_hidden_dim: 64,
        time_embed_dim: 48,
        rope_freq_dim: 2,
        ..Default::default()
    }
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn image_dit_pag_perturbs_only_generated_suffix_across_tiers_adapters_adaln_and_stream() {
    let dir = tempfile::tempdir().unwrap();
    let write_adapter = |name: &str, tensors: Vec<(String, Tensor)>, meta: &[(&str, &str)]| {
        let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
            .iter()
            .map(|(k, t)| {
                let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                (
                    k.clone(),
                    t.dims().to_vec(),
                    v.into_iter().flat_map(f32::to_le_bytes).collect(),
                )
            })
            .collect();
        let views: Vec<(String, safetensors::tensor::TensorView)> = bytes
            .iter()
            .map(|(k, shape, raw)| {
                (
                    k.clone(),
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        shape.clone(),
                        raw,
                    )
                    .unwrap(),
                )
            })
            .collect();
        let meta: std::collections::HashMap<String, String> = meta
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        let path = dir.path().join(name);
        safetensors::serialize_to_file(views, &Some(meta), &path).unwrap();
        path
    };

    let lora_path = write_adapter(
        "pag_lora.safetensors",
        vec![
            (
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight".into(),
                Tensor::full(0.01f32, (2, 64), &Device::Cpu).unwrap(),
            ),
            (
                "transformer.transformer_blocks.0.attn.to_q.lora_B.weight".into(),
                Tensor::full(0.02f32, (96, 2), &Device::Cpu).unwrap(),
            ),
        ],
        &[("alpha", "2")],
    );
    let lokr_path = write_adapter(
        "pag_lokr.safetensors",
        vec![
            (
                "transformer.transformer_blocks.0.attn.to_q.lokr_w1".into(),
                Tensor::full(0.01f32, (12, 8), &Device::Cpu).unwrap(),
            ),
            (
                "transformer.transformer_blocks.0.attn.to_q.lokr_w2".into(),
                Tensor::full(0.02f32, (8, 8), &Device::Cpu).unwrap(),
            ),
        ],
        &[("networkType", "lokr")],
    );
    let adapters = [
        AdapterSpec::new(lora_path, 0.5, AdapterKind::Lora),
        AdapterSpec::new(lokr_path, 0.25, AdapterKind::Lokr),
    ];

    let mut img_cfg = config();
    img_cfg.audio_in_channels = 0;
    img_cfg.variant = crate::dit::DitVariant::Image { num_pdd_heads: 6 };
    let schedule = SigmaSchedule::h3_image_distilled().unwrap();
    let cap = minimax_h3_image_pag_capability();
    let active_pag = cap
        .resolve(Some(&PagConfig {
            scale: 1.2,
            layers: Some(vec![0]),
            ..Default::default()
        }))
        .unwrap();
    let zero_pag = cap
        .resolve(Some(&PagConfig {
            scale: 0.0,
            layers: Some(vec![0]),
            ..Default::default()
        }))
        .unwrap();

    let device = Device::Cpu;
    let image_weights = |bits: Option<usize>, dev: &Device| {
        let mut map = candle_gen::candle_core::safetensors::load(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/dit_block.safetensors"
            ),
            dev,
        )
        .unwrap();
        if let Some(bits) = bits {
            let lanes = 32 / bits;
            let mask = (1u32 << bits) - 1;
            let codes: Vec<u32> = (0..96 * 64 / lanes)
                .map(|i| (0..lanes).fold(0, |w, j| w | (((i + j) as u32 & mask) << (j * bits))))
                .collect();
            map.insert(
                "transformer_blocks.0.attn.to_q.weight".into(),
                Tensor::from_vec(codes, (96, 64 / lanes), dev).unwrap(),
            );
            map.insert(
                "transformer_blocks.0.attn.to_q.scales".into(),
                Tensor::full(0.002f32, (96, 1), dev).unwrap(),
            );
            map.insert(
                "transformer_blocks.0.attn.to_q.biases".into(),
                Tensor::full(-0.01f32, (96, 1), dev).unwrap(),
            );
        }
        for head in 0..6 {
            let vals: Vec<f32> = (0..96 * 64)
                .map(|i| ((i as f32 * 0.13 + head as f32 * 0.7).sin()) * 0.02)
                .collect();
            let bias: Vec<f32> = (0..96)
                .map(|i| ((i + head) as f32 * 0.05).cos() * 0.005)
                .collect();
            map.insert(
                format!("pdd_proj_out.{head}.weight"),
                Tensor::from_vec(vals, (96, 64), dev).unwrap(),
            );
            map.insert(
                format!("pdd_proj_out.{head}.bias"),
                Tensor::from_vec(bias, 96, dev).unwrap(),
            );
        }
        Weights::from_map(map)
    };

    let build_prepared = |bits: Option<usize>, streamed: bool, has_refs: bool| {
        let source_dev = if streamed {
            Device::Cpu
        } else {
            device.clone()
        };
        let w = image_weights(bits, &source_dev);
        let mut dit = MiniMaxH3Dit::from_weights(&w, &img_cfg, &source_dev, DType::F32).unwrap();
        crate::adapters::apply_minimax_h3_image_adapters(&mut dit, &adapters).unwrap();
        if streamed {
            dit.prepare_streamed(&device, 1).unwrap();
        }
        let prepared = PreparedImageDit::prepare(dit, schedule.timesteps(), has_refs).unwrap();
        if !streamed {
            assert!(prepared.released_bytes > 0);
        }
        prepared
    };

    // 1. Exact block-0 prefix preservation on 16-reference edit (`[text(3) | refs(16) | gen(4)]`):
    // text + all 16 `<Picture N>` reference rows stay identical while generated suffix is perturbed.
    let edit_layout = ImageLayout::edit(&[1, 0, 1], &[(2, 2); 16], 4, 4, [1, 2, 2]).unwrap();
    let context_edit = Tensor::from_vec(
        (0..3 * 40)
            .map(|i| (i as f32 * 0.19).sin() * 0.4)
            .collect::<Vec<_>>(),
        (1, 3, 40),
        &device,
    )
    .unwrap();
    let edit_rows = Tensor::from_vec(
        (0..20 * 96)
            .map(|i| (i as f32 * 0.31).cos() * 0.5)
            .collect::<Vec<_>>(),
        (1, 20, 96),
        &device,
    )
    .unwrap();
    let prepared_edit = build_prepared(Some(4), false, true);
    let mut model_edit =
        ImageDit::from_prepared(prepared_edit, &edit_layout, &context_edit).unwrap();
    let mut trace_cond = ImageDitTrace {
        layers: vec![0],
        ..Default::default()
    };
    let mut trace_pert = ImageDitTrace {
        layers: vec![0],
        ..Default::default()
    };
    let _ =
        __model_proof_output!(model_edit.forward_traced(0, &edit_rows, &mut trace_cond)).unwrap();
    let _ = model_edit
        .forward_inner(
            0,
            &edit_rows,
            Some(&mut trace_pert),
            Some((&[0], edit_layout.num_conditioning_rows())),
        )
        .unwrap();
    let h_cond = &trace_cond.hidden[0].1;
    let h_pert = &trace_pert.hidden[0].1;
    let prefix = edit_layout.num_conditioning_rows();
    let gen_len = edit_layout.num_image_rows();
    let prefix_diff = (h_cond.narrow(1, 0, prefix).unwrap() - h_pert.narrow(1, 0, prefix).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let gen_diff = (h_cond.narrow(1, prefix, gen_len).unwrap()
        - h_pert.narrow(1, prefix, gen_len).unwrap())
    .unwrap()
    .abs()
    .unwrap()
    .max_all()
    .unwrap()
    .to_scalar::<f32>()
    .unwrap();
    assert!(
        prefix_diff < 1e-6,
        "prefix (text + 16 refs) moved: {prefix_diff}"
    );
    assert!(
        gen_diff > 1e-4,
        "generated suffix was not perturbed: {gen_diff}"
    );

    // 2. Full 6-step denoise across dense / Q8 / Q4, LoRA + LoKr, AdaLN eviction, and rung-4
    // block streaming on both T2I and 16-reference edit.
    let t2i_layout = ImageLayout::text_to_image(3, 4, 4, [1, 2, 2]).unwrap();
    let x0_t2i = initial_image_rows(4, 4, [1, 2, 2], 17, &device).unwrap();
    for bits in [None, Some(8), Some(4)] {
        let run_t2i = |streamed: bool, pag: Option<&candle_gen::gen_core::PagResolved>| {
            let request = crate::block_stream::tests::request();
            let _guard = streamed.then(|| {
                crate::block_stream::RequestGuard::enter(
                    &request,
                    candle_gen::gen_core::LoadShape::DeferredMaterialization,
                    &device,
                )
                .unwrap()
            });
            let prepared = build_prepared(bits, streamed, false);
            let mut model = ImageDit::from_prepared(prepared, &t2i_layout, &context_edit).unwrap();
            let out = denoise_image_with_pag(
                &mut model,
                &schedule,
                &H3ImageSampler::Lcm {
                    s_noise: 1.1,
                    s_noise_end: 0.9,
                    noise_clip_std: 0.0,
                },
                17,
                &x0_t2i,
                0,
                &device,
                &CancelFlag::default(),
                pag,
                &mut |_| {},
            )
            .unwrap();
            (
                model.forwards(),
                out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            )
        };
        let (f_none, out_none) = run_t2i(false, None);
        let (f_zero, out_zero) = run_t2i(false, Some(&zero_pag));
        assert_eq!(f_none, 6);
        assert_eq!(f_zero, 6);
        assert_eq!(out_none, out_zero);

        let (f_res, out_res) = run_t2i(false, Some(&active_pag));
        let (f_str, out_str) = run_t2i(true, Some(&active_pag));
        assert_eq!(f_res, 10);
        assert_eq!(f_str, 10);
        assert_ne!(out_res, out_none);
        let max_stream_diff = out_res
            .iter()
            .zip(&out_str)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_stream_diff < 1e-5,
            "bits={bits:?} resident vs streamed PAG diff {max_stream_diff}"
        );
    }

    // Also run 6-step 16-reference edit with active PAG on streamed Q8.
    {
        let request = crate::block_stream::tests::request();
        let _guard = crate::block_stream::RequestGuard::enter(
            &request,
            candle_gen::gen_core::LoadShape::DeferredMaterialization,
            &device,
        )
        .unwrap();
        let prepared = build_prepared(Some(8), true, true);
        let mut model = ImageDit::from_prepared(prepared, &edit_layout, &context_edit).unwrap();
        let out = denoise_image_with_pag(
            &mut model,
            &schedule,
            &H3ImageSampler::Euler,
            23,
            &edit_rows,
            16,
            &device,
            &CancelFlag::default(),
            Some(&active_pag),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(model.forwards(), 10);
        assert_eq!(out.dims(), &[1, 4, 96]);
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED dit::model::pag_tests::image_dit_pag_perturbs_only_generated_suffix_across_tiers_adapters_adaln_and_stream");
}
