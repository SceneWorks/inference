//! Real-weight Mac acceptance gate for SC-12794. Run with:
//! `WANVACE_FUN_DIR=<assembled snapshot> cargo test -p mlx-gen-wan --test
//! wanvace_fun_expert_swap_footprint -- --ignored --nocapture`.

use std::path::PathBuf;

use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Progress,
    Quant, ReplacementMode, WeightsSource,
};
use mlx_gen_wan::MODEL_ID_VACE_FUN;
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

fn frame(value: u8) -> Image {
    Image {
        width: 256,
        height: 256,
        pixels: vec![value; 256 * 256 * 3],
    }
}

fn render(root: &std::path::Path, policy: OffloadPolicy) -> (Vec<Image>, usize) {
    let generator = mlx_gen_wan::provider_registry()
        .unwrap()
        .load(
            MODEL_ID_VACE_FUN,
            &LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
                .with_quant(Quant::Q4)
                .with_offload_policy(policy),
        )
        .expect("load VACE-Fun");
    let request = GenerationRequest {
        prompt: "a person walking through a city street".into(),
        width: 256,
        height: 256,
        frames: Some(1),
        steps: Some(6),
        seed: Some(42),
        sampler: Some("unipc".into()),
        conditioning: vec![Conditioning::ControlClip {
            frames: vec![frame(128)],
            mask: vec![frame(255)],
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::FaceOnly,
        }],
        ..Default::default()
    };
    clear_cache();
    reset_peak_memory();
    let output = generator
        .generate(&request, &mut |_progress: Progress| {})
        .expect("generate VACE-Fun");
    let peak = get_peak_memory();
    let GenerationOutput::Video { frames, .. } = output else {
        panic!("expected video output")
    };
    (frames, peak)
}

#[test]
#[ignore = "needs an assembled VACE-Fun snapshot and Apple Silicon MLX"]
fn sequential_drops_one_expert_and_preserves_output() {
    let Some(root) = std::env::var_os("WANVACE_FUN_DIR").map(PathBuf::from) else {
        eprintln!("skip: set WANVACE_FUN_DIR to an assembled VACE-Fun snapshot");
        return;
    };
    let resident = render(&root, OffloadPolicy::Resident);
    let sequential = render(&root, OffloadPolicy::Sequential);

    assert_eq!(resident.0, sequential.0, "residency changed output bytes");
    eprintln!(
        "VACE-Fun peak: Resident {:.2} GiB, Sequential {:.2} GiB",
        resident.1 as f64 / 2_f64.powi(30),
        sequential.1 as f64 / 2_f64.powi(30)
    );
    assert!(
        sequential.1 < resident.1,
        "Sequential did not lower peak memory"
    );
    assert!(
        sequential.1 as f64 <= resident.1 as f64 * 0.85,
        "Sequential peak did not drop by at least 15%"
    );
}
