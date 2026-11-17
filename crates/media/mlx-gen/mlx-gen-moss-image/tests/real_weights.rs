//! Bounded real-weight MOSS-Image Base LoRA training proof on MLX (SW-7172): the MLX twin of the
//! Candle `real_weights::moss_bounded_acceleration_acceptance` ordinary arm.
//!
//! A 2:3 portrait of three horizontal bands (blue / cream / green), at least the run's edge² area,
//! trains a few LoRA steps on its own aspect-preserving canvas (never a centre-cropped square,
//! never upscaled). The one in-training preview renders at item 0's canvas, so it must be a
//! portrait; the saved adapter then reloads through the production generator and renders one
//! portrait sample. Both images are written under the model-proof executor's owned output
//! directory (`MODEL_PROOF_OUTPUT_DIR`, declared row `output_artifacts`) for inspection.
//!
//! `#[ignore]`d — needs the pinned `OpenMOSS-Team/MOSS-Image-Base` snapshot (`MOSS_BASE_SNAPSHOT`)
//! and the SFT snapshot that carries the shared text encoder / tokenizer / VAE
//! (`MOSS_SHARED_SNAPSHOT`), plus the Metal GPU:
//!
//!   cargo test -p mlx-gen-moss-image --features model-inference-proof --test integration \
//!     real_weights::moss_bounded_training_trains_a_portrait -- --ignored --exact --nocapture

use std::path::PathBuf;

use mlx_gen::gen_core::moss_image::{MossImageVariant, MOSS_SHARED_SNAPSHOT_COMPONENT};
use mlx_gen::gen_core::ProviderRegistryBuilder;
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::{
    CancelFlag, GenerationOutput, GenerationRequest, Image, LoadSpec, NetworkType, TrainingConfig,
    TrainingItem, TrainingProgress, TrainingRequest, WeightsSource,
};

fn required(name: &str) -> PathBuf {
    let path =
        PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("supply explicit {name}")));
    assert!(
        path.is_dir(),
        "{name}={} is not a directory",
        path.display()
    );
    path
}

/// The Base trainer/generator spec: the Base transformer plus the SFT shared components.
fn spec(base: &PathBuf, shared: &PathBuf) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(base.clone()));
    spec.components.insert(
        MOSS_SHARED_SNAPSHOT_COMPONENT.into(),
        WeightsSource::Dir(shared.clone()),
    );
    spec
}

fn save(image: &Image, name: &str) {
    if let Some(dir) = std::env::var_os("MODEL_PROOF_OUTPUT_DIR") {
        image::save_buffer(
            PathBuf::from(dir).join(name),
            &image.pixels,
            image.width,
            image.height,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }
}

fn assert_picture(label: &str, image: &Image) {
    assert_eq!(
        image.pixels.len(),
        (image.width * image.height * 3) as usize,
        "{label}: a packed RGB8 buffer"
    );
    let (lo, hi) = image
        .pixels
        .iter()
        .fold((255u8, 0u8), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    assert!(hi > lo, "{label}: a flat constant image");
    assert!(
        image.height > image.width,
        "{label}: renders at the portrait canvas, got {}x{}",
        image.width,
        image.height
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn moss_bounded_training_trains_a_portrait() {
    let base = required("MOSS_BASE_SNAPSHOT");
    let shared = required("MOSS_SHARED_SNAPSHOT");
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();

    // A 256×384 (2:3) portrait of three horizontal bands, at least the run's 256² edge area.
    let resolution = 256u32;
    let (w, h) = (resolution, resolution * 3 / 2);
    let bands = [[80u8, 140, 210], [235, 225, 200], [40, 160, 70]];
    let input = tmp.join("input.png");
    image::RgbImage::from_fn(w, h, |_, y| {
        image::Rgb(bands[(y as usize * bands.len()) / h as usize])
    })
    .save(&input)
    .unwrap();
    const PROMPT: &str = "a blue, cream and green banded tile";

    let steps = 4;
    let registry = mlx_gen_moss_image::register_providers(ProviderRegistryBuilder::new())
        .build()
        .unwrap();
    let mut trainer = registry
        .load_trainer(MossImageVariant::Base.provider_id(), &spec(&base, &shared))
        .expect("the MOSS Base snapshot loads as a trainer");
    let req = TrainingRequest {
        items: vec![TrainingItem::captioned(input, PROMPT.into())],
        config: TrainingConfig {
            rank: 2,
            alpha: 2.,
            learning_rate: 1e-3,
            steps,
            resolution,
            batch_size: 1,
            gradient_accumulation: 1,
            lr_warmup_steps: 0,
            save_every: 0,
            seed: 7,
            network_type: NetworkType::Lora,
            sample_every: steps,
            sample_steps: 8,
            sample_guidance_scale: 4.0,
            sample_prompts: vec![PROMPT.into()],
            ..Default::default()
        },
        output_dir: tmp.join("out"),
        file_name: "adapter.safetensors".into(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };
    let mut losses = Vec::new();
    let mut previews: Vec<Image> = Vec::new();
    let output = trainer
        .train(&req, &mut |p| match p {
            TrainingProgress::Training { step, loss, .. } => {
                eprintln!("[moss-bounded] step {step} loss {loss:.5}");
                losses.push(loss);
            }
            TrainingProgress::Sample { image, .. } => previews.push(image),
            _ => {}
        })
        .expect("MOSS bounded training");
    drop(trainer);
    mlx_rs::memory::clear_cache();

    for (i, preview) in previews.iter().enumerate() {
        save(preview, &format!("preview_{i}.png"));
    }
    assert_eq!(output.steps, steps, "every step ran");
    assert!(
        output.final_loss.is_finite() && losses.iter().all(|l| l.is_finite()),
        "finite losses: {losses:?}"
    );
    assert!(output.adapter_path.is_file(), "adapter written");
    assert_eq!(previews.len(), 1, "one preview at the step-{steps} cadence");
    assert_picture("preview", &previews[0]);

    // Reload the saved adapter through the production generator and render one portrait sample.
    let generator = registry
        .load(
            MossImageVariant::Base.provider_id(),
            &spec(&base, &shared).with_adapters(vec![AdapterSpec::new(
                output.adapter_path.clone(),
                1.,
                AdapterKind::Lora,
            )]),
        )
        .expect("the trained adapter reloads on the MOSS Base generator");
    let rendered = match generator
        .generate(
            &GenerationRequest {
                prompt: PROMPT.into(),
                width: w,
                height: h,
                steps: Some(8),
                seed: Some(7),
                ..Default::default()
            },
            &mut |_| {},
        )
        .expect("render with the reloaded adapter")
    {
        GenerationOutput::Images(mut images) => images.remove(0),
        other => panic!("images expected, got {other:?}"),
    };
    save(&rendered, "adapter_sample.png");
    assert_eq!((rendered.width, rendered.height), (w, h));
    assert_picture("adapter sample", &rendered);
    crate::model_inference_device::record_model_inference_device();
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED real_weights::moss_bounded_training_trains_a_portrait"
    );
}
