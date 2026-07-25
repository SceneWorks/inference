//! Fixed-image/instruction parity against the frozen Torch Mage-Flow-Edit pipeline.

mod common;

use common::{error, require_golden};
use image::RgbImage;
use mlx_gen_boogu::vision::preprocess::preprocess_image;
use mlx_gen_mage::model::{EDIT_BASE_SNAPSHOT_REVISION, EDIT_TURBO_SNAPSHOT_REVISION};
use mlx_gen_mage::{GsKey, MageFlowPipeline};
use mlx_rs::Dtype;

const GOLDEN: &str = "mage_flow_edit_golden.safetensors";
const TE_GOLDEN: &str = "mage_flow_te_golden.safetensors";
const INSTRUCTION: &str = "Replace the background with a field of sunflowers";

#[test]
#[ignore = "needs MAGE_EDIT_SNAPSHOT, Metal, and mage_flow_edit_golden.safetensors"]
fn fixed_instruction_edit_matches_the_torch_reference() {
    let root = std::env::var("MAGE_EDIT_SNAPSHOT")
        .expect("set MAGE_EDIT_SNAPSHOT to microsoft/Mage-Flow-Edit");
    let golden = require_golden(GOLDEN);
    let geometry = golden.require("geometry").unwrap().as_slice::<i32>();
    let (height, width, steps) = (geometry[0] as u32, geometry[1] as u32, geometry[3] as usize);
    let seed = golden.require("seed").unwrap().as_slice::<i64>()[0];
    let cfg = golden.require("cfg").unwrap().as_slice::<f32>()[0];
    let edit_revision = golden.metadata("edit_revision");
    let is_edit_base = edit_revision == Some(EDIT_BASE_SNAPSHOT_REVISION);
    if is_edit_base {
        assert_eq!((steps, cfg), (30, 5.0));
    }
    if edit_revision == Some(EDIT_TURBO_SNAPSHOT_REVISION) {
        assert_eq!((steps, cfg), (4, 1.0));
        assert!(!mlx_gen_mage::pipeline::uses_cfg(cfg));
    }
    let key = GsKey::from_u64(golden.require("gs_key").unwrap().as_slice::<i64>()[0] as u64);
    let reference = golden
        .require("ref_u8")
        .unwrap()
        .as_dtype(Dtype::Uint8)
        .unwrap();
    let reference = RgbImage::from_raw(width, height, reference.as_slice::<u8>().to_vec())
        .expect("golden reference is RGB8");
    let target_tokens = golden.require("target_tokens").unwrap().as_slice::<i32>()[0];
    let golden_step0 = golden.require("seq_step0").unwrap();
    let golden_segments = golden_step0
        .split_axis(&[target_tokens, target_tokens * 2], 1)
        .unwrap();
    let reference_tokens = golden_segments[1].clone();

    let pipeline = MageFlowPipeline::load_edit(root, None).unwrap();
    let trace = pipeline
        .edit_trace_from_reference_tokens(
            INSTRUCTION,
            " ",
            &[reference],
            &reference_tokens,
            height,
            width,
            steps,
            cfg,
            seed,
            &key,
            false,
            &mut |_| {},
        )
        .unwrap();
    mlx_rs::transforms::eval([
        &trace.trajectories[0],
        &trace.trajectories[1],
        &trace.final_tokens,
        &trace.image_u8,
    ])
    .unwrap();

    let got_target = trace.trajectories[0]
        .split_axis(&[target_tokens], 1)
        .unwrap()
        .swap_remove(0);
    let want_target = golden_step0
        .split_axis(&[target_tokens], 1)
        .unwrap()
        .swap_remove(0);
    assert_eq!(
        error(&got_target, &want_target).0,
        0.0,
        "the target-first watermarked noise must be exact"
    );
    let (seq0_max, _, seq0_mean) = error(&trace.trajectories[0], golden_step0);
    println!("seq_step0: max_abs={seq0_max:.6} mean_rel={seq0_mean:.6}");
    assert_eq!(
        seq0_max, 0.0,
        "replayed target and reference tokens must be exact"
    );

    for (label, got, want, mean_gate) in [
        (
            "seq_step1",
            &trace.trajectories[1],
            golden.require("seq_step1").unwrap(),
            0.01,
        ),
        (
            "final_tokens",
            &trace.final_tokens,
            golden.require("final_tokens").unwrap(),
            // Edit-Base accumulates bf16 Euler error for 30 transformer forwards. Key the measured
            // long-schedule envelope to its immutable snapshot revision so the original Edit-RL
            // regression gate remains unchanged.
            if is_edit_base { 0.30 } else { 0.20 },
        ),
        (
            "image_u8",
            &trace.image_u8,
            golden.require("image_u8").unwrap(),
            if is_edit_base { 0.10 } else { 0.08 },
        ),
    ] {
        let (max_abs, _, mean_rel) = error(got, want);
        println!("{label}: max_abs={max_abs:.6} mean_rel={mean_rel:.6}");
        assert!(
            mean_rel <= mean_gate,
            "{label}: mean_rel={mean_rel} exceeds {mean_gate}"
        );
    }
}

#[test]
#[ignore = "needs MAGE_EDIT_SNAPSHOT, Metal, and mage_flow_edit_golden.safetensors"]
fn seeded_mlx_posterior_sampling_is_deterministic_and_stochastic() {
    let root = std::env::var("MAGE_EDIT_SNAPSHOT")
        .expect("set MAGE_EDIT_SNAPSHOT to microsoft/Mage-Flow-Edit");
    let golden = require_golden(GOLDEN);
    let geometry = golden.require("geometry").unwrap().as_slice::<i32>();
    let (height, width) = (geometry[0] as u32, geometry[1] as u32);
    let seed = golden.require("seed").unwrap().as_slice::<i64>()[0] as u64;
    let source = golden
        .require("ref_u8")
        .unwrap()
        .as_dtype(Dtype::Uint8)
        .unwrap();
    let reference = RgbImage::from_raw(width, height, source.as_slice::<u8>().to_vec()).unwrap();
    let pipeline = MageFlowPipeline::load_edit(root, None).unwrap();

    let first = pipeline
        .sample_reference_tokens(std::slice::from_ref(&reference), height, width, seed)
        .unwrap();
    let repeated = pipeline
        .sample_reference_tokens(std::slice::from_ref(&reference), height, width, seed)
        .unwrap();
    let moved = pipeline
        .sample_reference_tokens(&[reference], height, width, seed + 1)
        .unwrap();
    mlx_rs::transforms::eval([&first, &repeated, &moved]).unwrap();

    assert_eq!(
        error(&first, &repeated).0,
        0.0,
        "the same seed must reproduce the sampled posterior exactly"
    );
    assert!(
        error(&first, &moved).0 > 0.01,
        "a new seed must move the posterior sample instead of returning the mean"
    );
}

#[test]
#[ignore = "needs MAGE_EDIT_SNAPSHOT, Metal, and mage_flow_te_golden.safetensors"]
fn multimodal_instruction_conditioning_matches_torch() {
    let root = std::env::var("MAGE_EDIT_SNAPSHOT")
        .expect("set MAGE_EDIT_SNAPSHOT to microsoft/Mage-Flow-Edit");
    let golden = require_golden(TE_GOLDEN);
    let source = golden
        .require("edit_vl_ref_u8")
        .unwrap()
        .as_dtype(Dtype::Uint8)
        .unwrap();
    let shape = source.shape();
    let image = RgbImage::from_raw(
        shape[1] as u32,
        shape[0] as u32,
        source.as_slice::<u8>().to_vec(),
    )
    .unwrap();
    let (pixels, grid) = preprocess_image(&image).unwrap();
    let (pixel_max, _, pixel_mean) = error(&pixels, golden.require("edit_pixel_values").unwrap());
    println!("edit_pixel_values: max_abs={pixel_max:.6} mean_rel={pixel_mean:.6}");
    assert_eq!(
        grid,
        [1, 24, 12],
        "the shared Qwen3-VL preprocessor grid must match Torch"
    );
    assert!(
        pixel_max <= 0.02 && pixel_mean <= 0.01,
        "edit pixel preprocessing must agree before testing vision/LM"
    );
    let source = golden
        .require("edit_vl_source_u8")
        .unwrap()
        .as_dtype(Dtype::Uint8)
        .unwrap();
    let source_image = RgbImage::from_raw(1024, 2048, source.as_slice::<u8>().to_vec()).unwrap();
    let encoder = mlx_gen_mage::text_encoder::load_multimodal(root).unwrap();
    let (vision, deepstack, vision_grid) = encoder.vision_features(&source_image).unwrap();
    assert_eq!(vision_grid, grid);
    let ids = encoder
        .edit_input_ids(INSTRUCTION, &[vision.shape()[0] as usize])
        .unwrap();
    let torch_ids = golden
        .require("edit_input_ids")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    assert_eq!(ids, torch_ids, "expanded edit token IDs must match Torch");
    let (t, h, w) = encoder.edit_mrope_axes(&ids, &[grid]).unwrap();
    let torch_positions = golden
        .require("edit_position_ids")
        .unwrap()
        .as_slice::<i64>();
    let seq = ids.len();
    assert_eq!(
        t,
        torch_positions[..seq]
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>(),
        "temporal M-RoPE axis"
    );
    assert_eq!(
        h,
        torch_positions[seq..2 * seq]
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>(),
        "height M-RoPE axis"
    );
    assert_eq!(
        w,
        torch_positions[2 * seq..]
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>(),
        "width M-RoPE axis"
    );
    for (label, got, want) in [
        (
            "edit_vision_embeds",
            &vision,
            golden.require("edit_vision_embeds").unwrap(),
        ),
        (
            "edit_deepstack_0",
            &deepstack[0],
            golden.require("edit_deepstack_0").unwrap(),
        ),
        (
            "edit_deepstack_1",
            &deepstack[1],
            golden.require("edit_deepstack_1").unwrap(),
        ),
        (
            "edit_deepstack_2",
            &deepstack[2],
            golden.require("edit_deepstack_2").unwrap(),
        ),
    ] {
        let (max_abs, _, mean_rel) = error(got, want);
        println!("{label}: max_abs={max_abs:.6} mean_rel={mean_rel:.6}");
        assert!(
            mean_rel <= 0.05,
            "{label}: original-source vision boundary mean_rel={mean_rel} exceeds 0.05"
        );
    }
    let torch_vision = golden.require("edit_vision_embeds").unwrap().clone();
    let torch_deepstack = vec![
        golden.require("edit_deepstack_0").unwrap().clone(),
        golden.require("edit_deepstack_1").unwrap().clone(),
        golden.require("edit_deepstack_2").unwrap().clone(),
    ];
    let isolated = encoder
        .encode_edit_with_features(INSTRUCTION, &[torch_vision], &[torch_deepstack], &[grid])
        .unwrap();
    let (isolated_max, _, isolated_mean) =
        error(&isolated.txt, golden.require("edit_txt").unwrap());
    println!("edit_txt_exact_vision: max_abs={isolated_max:.6} mean_rel={isolated_mean:.6}");
    assert!(
        isolated_max <= 4.0 && isolated_mean <= 0.04,
        "the LM must match Torch from identical vision/deepstack inputs"
    );
    let reordered = encoder
        .encode_edit_with_features(
            INSTRUCTION,
            &[golden.require("edit_vision_embeds").unwrap().clone()],
            &[vec![
                golden.require("edit_deepstack_2").unwrap().clone(),
                golden.require("edit_deepstack_1").unwrap().clone(),
                golden.require("edit_deepstack_0").unwrap().clone(),
            ]],
            &[grid],
        )
        .unwrap();
    assert!(
        error(&reordered.txt, golden.require("edit_txt").unwrap()).2 > 0.04,
        "reordering blocks 5/11/17 must fail the identical-input LM gate"
    );
    let kept_ids = &ids[64..];
    let visual_indices = kept_ids
        .iter()
        .enumerate()
        .filter_map(|(i, &id)| (id == 151_655).then_some(i as i32))
        .collect::<Vec<_>>();
    let text_indices = kept_ids
        .iter()
        .enumerate()
        .filter_map(|(i, &id)| (id != 151_655).then_some(i as i32))
        .collect::<Vec<_>>();
    for (label, indices) in [("visual", visual_indices), ("text", text_indices)] {
        let index = mlx_rs::Array::from_slice(&indices, &[indices.len() as i32]);
        let got = isolated.txt.take_axis(index.clone(), 0).unwrap();
        let want = golden
            .require("edit_txt")
            .unwrap()
            .take_axis(index, 0)
            .unwrap();
        let (max_abs, _, mean_rel) = error(&got, &want);
        println!("edit_txt_exact_vision_{label}: max_abs={max_abs:.6} mean_rel={mean_rel:.6}");
    }
    let early = encoder
        .edit_early_lm_trace(
            INSTRUCTION,
            &[golden.require("edit_vision_embeds").unwrap().clone()],
            &[vec![
                golden.require("edit_deepstack_0").unwrap().clone(),
                golden.require("edit_deepstack_1").unwrap().clone(),
                golden.require("edit_deepstack_2").unwrap().clone(),
            ]],
            &[grid],
        )
        .unwrap();
    for (index, got) in early.iter().enumerate() {
        let key = format!("edit_lm_layer_{index}_pre_inject");
        let (max_abs, _, mean_rel) = error(got, golden.require(&key).unwrap());
        println!("edit_lm_layer_{index}: max_abs={max_abs:.6} mean_rel={mean_rel:.6}");
        assert!(
            mean_rel <= 0.01,
            "LM layer {index} boundary exceeds the strict identical-input gate"
        );
    }
    let conditioning = encoder.encode_edit(INSTRUCTION, &[source_image]).unwrap();
    mlx_rs::transforms::eval([&conditioning.txt]).unwrap();
    assert_eq!(conditioning.seq_lens, vec![93]);
    let (max_abs, _, mean_rel) = error(&conditioning.txt, golden.require("edit_txt").unwrap());
    println!("edit_txt: max_abs={max_abs:.6} mean_rel={mean_rel:.6}");
    assert!(
        max_abs <= 16.0 && mean_rel <= 0.10,
        "combined vision+LM conditioning exceeds measured cross-backend spread"
    );
}
