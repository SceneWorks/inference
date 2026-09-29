//! SC-20686 Metal-lane product entrypoint for the FLUX.2 Klein reference-edit routes (MLX):
//! `flux2_klein_9b_edit` (reference K/V recomputed every denoise evaluation) and
//! `flux2_klein_9b_kv_edit` (persistent reference-K/V cache extracted on step 0).
//!
//! Loads the route through the MLX provider loader the SceneWorks worker uses, runs one real edit
//! with the given reference(s), and writes the image to `--out` (a `.png` path). With
//! `--sc20686-campaign` it arms the Metal-lane observer (`mlx_gen::sc20686`); without it the
//! observer stays inert and this is an ordinary edit.
//!
//! ```text
//! cargo build --release -p mlx-gen-flux2 --example sc20686_flux2_edit
//! target/release/examples/sc20686_flux2_edit --variant flux2_klein_9b_kv_edit --snapshot <root> \
//!   --reference ref.png [--reference2 ref2.png] --width 512 --height 512 --prompt "…" \
//!   --guidance 1 --steps 4 --out /abs/media.png \
//!   [--sc20686-campaign --sc20686-events /abs/events.jsonl --sc20686-source-ref <40-hex> \
//!    --sc20686-residency sequential [--sc20686-cancel]]
//! ```
//!
//! `--single-only` is accepted for argument parity with the Candle `flux2-edit` coordinate
//! definitions; this entrypoint always performs exactly one generation.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, Progress,
    WeightsSource,
};
use mlx_gen::media::Image;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Frozen SceneWorks-equivalent residency for both FLUX.2 edit routes.
const PRODUCT_RESIDENCY: &str = "sequential";

fn arg(args: &[String], key: &str) -> Option<String> {
    let mut values = args
        .iter()
        .enumerate()
        .filter(|(_, value)| *value == key)
        .filter_map(|(index, _)| args.get(index + 1).cloned());
    let first = values.next();
    if values.next().is_some() {
        return None;
    }
    first
}

fn required(args: &[String], key: &str) -> Result<String> {
    arg(args, key).ok_or_else(|| format!("missing or repeated {key}").into())
}

fn parsed<T: std::str::FromStr>(args: &[String], key: &str) -> Result<T> {
    required(args, key)?
        .parse()
        .map_err(|_| format!("{key} is malformed").into())
}

/// A coordinate argument with a single-mode default. Absent → `default`; repeated → error (an
/// ambiguous campaign input is refused, never resolved by picking one).
fn parsed_or<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> Result<T> {
    match args.iter().filter(|value| *value == key).count() {
        0 => Ok(default),
        _ => parsed(args, key),
    }
}

fn load_image(path: &Path) -> Result<Image> {
    let rgb = image::open(path)?.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let route = required(&args, "--variant")?;
    let campaign = args.iter().any(|value| value == "--sc20686-campaign");
    let cancel_arm = args.iter().any(|value| value == "--sc20686-cancel");
    if campaign {
        let residency = required(&args, "--sc20686-residency")?;
        if residency != PRODUCT_RESIDENCY {
            return Err(format!("{route} campaign residency must be {PRODUCT_RESIDENCY}").into());
        }
    } else if cancel_arm {
        return Err("--sc20686-cancel requires --sc20686-campaign".into());
    }
    let _campaign_request = if campaign {
        let events = arg(&args, "--sc20686-events")
            .filter(|path| path != "-")
            .ok_or("SC-20686 campaign requires a dedicated --sc20686-events <file>")?;
        let request = mlx_gen::sc20686::request_output(
            events,
            required(&args, "--sc20686-source-ref")?,
            PRODUCT_RESIDENCY,
        )?;
        Some(if cancel_arm {
            request.arm_cancellation()
        } else {
            request.arm()
        })
    } else {
        None
    };

    let mut references = vec![load_image(Path::new(&required(&args, "--reference")?))?];
    if let Some(second) = arg(&args, "--reference2") {
        references.push(load_image(Path::new(&second))?);
    }
    let conditioning = if references.len() == 1 {
        vec![Conditioning::Reference {
            image: references.remove(0),
            strength: None,
        }]
    } else {
        vec![Conditioning::MultiReference { images: references }]
    };
    let request = GenerationRequest {
        prompt: parsed_or(&args, "--prompt", "SC-20686 edit one reference".to_owned())?,
        width: parsed_or(&args, "--width", 512)?,
        height: parsed_or(&args, "--height", 512)?,
        count: 1,
        seed: Some(parsed_or(&args, "--seed", 42)?),
        steps: Some(parsed_or(&args, "--steps", 4)?),
        guidance: Some(parsed_or(&args, "--guidance", 1.0)?),
        conditioning,
        ..Default::default()
    };

    let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(required(
        &args,
        "--snapshot",
    )?)))
    .with_offload_policy(OffloadPolicy::Sequential);
    let generator = match route.as_str() {
        "flux2_klein_9b_edit" => mlx_gen_flux2::load_klein_9b_edit(&spec)?,
        "flux2_klein_9b_kv_edit" => mlx_gen_flux2::load_klein_9b_kv_edit(&spec)?,
        other => return Err(format!("unsupported SC-20686 FLUX.2 route: {other}").into()),
    };
    let out = PathBuf::from(required(&args, "--out")?);
    let mut on_progress = |progress: Progress| match progress {
        Progress::Step { current, total } => eprintln!("[sc20686-flux2] step {current}/{total}"),
        Progress::Decoding => eprintln!("[sc20686-flux2] decoding"),
        Progress::Loading(phase) => eprintln!("[sc20686-flux2] loading {phase:?}"),
    };
    let output = match generator.generate(&request, &mut on_progress) {
        Ok(output) => output,
        Err(_) if cancel_arm && mlx_gen::sc20686::campaign_cancelled() => {
            eprintln!("[sc20686-flux2] expected SC-20686 campaign cancellation");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    if cancel_arm {
        return Err(
            "the SC-20686 cancellation arm completed without a product cancellation".into(),
        );
    }
    let GenerationOutput::Images(images) = output else {
        return Err("expected an image output".into());
    };
    let image = images.first().ok_or("the edit produced no image")?;
    image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
        .ok_or("invalid RGB image dimensions")?
        .save(&out)?;
    eprintln!("[sc20686-flux2] wrote {}", out.display());
    Ok(())
}
