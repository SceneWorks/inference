//! Single-case terminal adapter acceptance driver. Compilation/preparation never admits GPU use.
//! Run once per selected Base/SFT × LoRA/LoKr cell only after the coordinator admits device time
//! and supplies an external RSS/time guard. No automatic campaign or guessed memory budget.
use gen_core_moss_image::moss_image_training;
use mlx_gen::{
    gen_core::moss_image::MossImageVariant,
    runtime::{AdapterKind, AdapterSpec},
    GenerationRequest, Generator, LoadSpec, NetworkType, TrainingConfig, TrainingRequest,
    WeightsSource,
};
use std::{error::Error, path::PathBuf};
fn argument(args: &[String], key: &str) -> Result<String, Box<dyn Error>> {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
        .ok_or_else(|| format!("missing {key}").into())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let output = PathBuf::from(argument(&args, "--output")?);
    std::fs::create_dir_all(&output)?;
    let dataset = output.join("synthetic-moss-v1");
    std::fs::create_dir_all(&dataset)?;
    let captions = [
        "sksmoss red circle centered on a pale background",
        "sksmoss blue square centered on a pale background",
        "sksmoss green triangle centered on a pale background",
    ];
    let mut items = Vec::new();
    for (index, caption) in captions.iter().enumerate() {
        let mut image = image::RgbImage::from_pixel(256, 256, image::Rgb([240, 240, 224]));
        for y in 0i32..256 {
            for x in 0i32..256 {
                let inside = match index {
                    0 => (x - 128).pow(2) + (y - 128).pow(2) < 70 * 70,
                    1 => (58..198).contains(&x) && (58..198).contains(&y),
                    _ => y > 45 && y < 210 && (x - 128).abs() < (y - 45) / 2,
                };
                if inside {
                    image.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgb(match index {
                            0 => [220, 30, 30],
                            1 => [30, 60, 220],
                            _ => [30, 180, 60],
                        }),
                    );
                }
            }
        }
        let path = dataset.join(format!("shape-{index}.png"));
        image.save(&path)?;
        std::fs::write(dataset.join(format!("shape-{index}.txt")), caption)?;
        items.push(mlx_gen::gen_core::TrainingItem::captioned(
            path,
            caption.to_string(),
        ));
    }
    if !args.iter().any(|a| a == "--execute") {
        println!(
            "{}",
            serde_json::json!({"prepared_dataset":dataset,"items":items.len(),"executed":false,"policy":"coordinator admission and external RSS/time guard required"})
        );
        return Ok(());
    }
    if std::env::var("MOSS_TRAINING_ADMITTED").as_deref() != Ok("1") {
        return Err(
            "device execution requires coordinator admission (MOSS_TRAINING_ADMITTED=1)".into(),
        );
    }
    let variant = match argument(&args, "--variant")?.as_str() {
        "base" => MossImageVariant::Base,
        "sft" => MossImageVariant::Sft,
        _ => return Err("variant must be base or sft".into()),
    };
    let (network, kind) = match argument(&args, "--network")?.as_str() {
        "lora" => (NetworkType::Lora, AdapterKind::Lora),
        "lokr" => (NetworkType::Lokr, AdapterKind::Lokr),
        _ => return Err("network must be lora or lokr".into()),
    };
    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(argument(
        &args,
        "--snapshot",
    )?)))
    .with_component(
        "moss-shared-snapshot",
        WeightsSource::Dir(PathBuf::from(argument(&args, "--shared")?)),
    );
    let steps: u32 = argument(&args, "--steps")?.parse()?;
    let request = TrainingRequest {
        items,
        config: TrainingConfig {
            steps,
            resolution: 256,
            rank: 4,
            alpha: 6.,
            network_type: network,
            learning_rate: 0.001,
            save_every: 0,
            seed: 1789,
            timestep_type: moss_image_training::SAMPLER.into(),
            trigger_word: Some("sksmoss".into()),
            ..Default::default()
        },
        output_dir: output.clone(),
        file_name: format!("{}-{network:?}.safetensors", variant.provider_id()),
        trigger_words: vec!["sksmoss".into()],
        cancel: Default::default(),
    };
    let mut trainer = match variant {
        MossImageVariant::Base => mlx_gen_moss_image::training::load_base(&spec)?,
        MossImageVariant::Sft => mlx_gen_moss_image::training::load_sft(&spec)?,
    };
    trainer.validate(&request)?;
    let trained = trainer.train(&request, &mut |event| eprintln!("{event:?}"))?;
    drop(trainer);
    let generation = GenerationRequest {
        prompt: captions[0].into(),
        width: 256,
        height: 256,
        seed: Some(1789),
        steps: Some(8),
        guidance: Some(1.),
        ..Default::default()
    };
    let before = {
        let model = mlx_gen_moss_image::MossImage::load(variant, &spec)?;
        match model.generate(&generation, &mut |_| {})? {
            mlx_gen::GenerationOutput::Images(images) => images,
            _ => return Err("expected image output".into()),
        }
    };
    let mut adapted = spec.clone();
    adapted
        .adapters
        .push(AdapterSpec::new(trained.adapter_path.clone(), 1., kind));
    let model = mlx_gen_moss_image::MossImage::load(variant, &adapted)?;
    let after = match model.generate(&generation, &mut |_| {})? {
        mlx_gen::GenerationOutput::Images(images) => images,
        _ => return Err("expected image output".into()),
    };
    if before.len() != 1 || after.len() != 1 || before[0].pixels == after[0].pixels {
        return Err("trained adapter did not change genuine generation pixels".into());
    }
    let receipt = serde_json::json!({"provider":variant.provider_id(),"revision":variant.revision(),"network":format!("{network:?}"),
        "steps":trained.steps,"loss":trained.final_loss,"adapter":trained.adapter_path,
        "posterior":moss_image_training::POSTERIOR,"sampler":moss_image_training::SAMPLER,
        "generation_changed":true,"adapter_reports":format!("{:?}",model.adapter_apply_reports())});
    std::fs::write(
        output.join(format!(
            "{}-{network:?}-receipt.json",
            variant.provider_id()
        )),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}
