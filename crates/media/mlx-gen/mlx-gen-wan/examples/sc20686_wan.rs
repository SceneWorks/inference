//! SC-20686 Metal-lane product entrypoint for the five registered Wan routes (MLX).
//!
//! Loads the requested route through the same MLX provider loader the SceneWorks worker uses,
//! builds the request from the frozen campaign coordinate, runs one real `generate`, and writes the
//! decoded frames as PNGs below `--out`. With `--sc20686-campaign` it arms the Metal-lane observer
//! (`mlx_gen::sc20686`) on this thread; without it the observer stays inert and this is an ordinary
//! render.
//!
//! ```text
//! cargo build --release -p mlx-gen-wan --example sc20686_wan
//! target/release/examples/sc20686_wan --variant wan2_2_t2v_14b --snapshot <tier-root> \
//!   --width 512 --height 512 --frames 17 --prompt "…" --guidance 5 --steps 4 --out /abs/media \
//!   [--sc20686-campaign --sc20686-events /abs/events.jsonl --sc20686-source-ref <40-hex> \
//!    --sc20686-residency sequential [--sc20686-cancel]]
//! ```
//!
//! Route inputs: `--image <png>` (I2V-14B); `--control-dir <dir> --mask-dir <dir>
//! [--reference <png>]` (VACE and VACE-Fun). The campaign adapter owns every `--sc20686-*` flag.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Progress,
    ReplacementMode, WeightsSource,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const ROUTES: [(&str, &str); 5] = [
    ("wan2_2_ti2v_5b", "sequential"),
    ("wan2_2_t2v_14b", "sequential"),
    ("wan2_2_i2v_14b", "sequential"),
    ("wan_vace", "resident"),
    ("wan2_2_vace_fun_14b", "sequential"),
];

fn arg(args: &[String], key: &str) -> Option<String> {
    let mut values = args
        .iter()
        .enumerate()
        .filter(|(_, value)| *value == key)
        .filter_map(|(index, _)| args.get(index + 1).cloned());
    let first = values.next();
    // A repeated flag is ambiguous campaign input; refuse it rather than pick one.
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

fn load_png_dir(dir: &Path) -> Result<Vec<Image>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("png"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no .png frames in {}", dir.display()).into());
    }
    paths.iter().map(|path| load_image(path)).collect()
}

/// The frozen product residency for `route`, applied to `LoadSpec::offload_policy` (the MLX Wan
/// providers' memory strategy: `Sequential` = staged component/expert residency).
fn product_policy(route: &str, residency: &str) -> Result<OffloadPolicy> {
    let expected = ROUTES
        .iter()
        .find(|(id, _)| *id == route)
        .map(|(_, residency)| *residency)
        .ok_or_else(|| format!("unsupported SC-20686 Wan route: {route}"))?;
    if residency != expected {
        return Err(format!("{route} campaign residency must be {expected}").into());
    }
    Ok(match residency {
        "sequential" => OffloadPolicy::Sequential,
        _ => OffloadPolicy::Resident,
    })
}

fn load(route: &str, spec: &LoadSpec) -> Result<Box<dyn mlx_gen::Generator>> {
    Ok(match route {
        "wan2_2_ti2v_5b" => mlx_gen_wan::model::load(spec)?,
        "wan2_2_t2v_14b" => mlx_gen_wan::model::load_t2v_14b(spec)?,
        "wan2_2_i2v_14b" => mlx_gen_wan::model::load_i2v_14b(spec)?,
        "wan_vace" => mlx_gen_wan::model_vace::load(spec)?,
        "wan2_2_vace_fun_14b" => mlx_gen_wan::model_vace::load_vace_fun(spec)?,
        other => return Err(format!("unsupported SC-20686 Wan route: {other}").into()),
    })
}

fn conditioning(route: &str, args: &[String]) -> Result<Vec<Conditioning>> {
    match route {
        "wan2_2_i2v_14b" => Ok(vec![Conditioning::Reference {
            image: load_image(Path::new(&required(args, "--image")?))?,
            strength: None,
        }]),
        "wan_vace" | "wan2_2_vace_fun_14b" => {
            let frames = load_png_dir(Path::new(&required(args, "--control-dir")?))?;
            let mask = load_png_dir(Path::new(&required(args, "--mask-dir")?))?;
            let mut conditioning = vec![Conditioning::ControlClip {
                frames,
                mask,
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            }];
            if let Some(reference) = arg(args, "--reference") {
                conditioning.push(Conditioning::Reference {
                    image: load_image(Path::new(&reference))?,
                    strength: None,
                });
            }
            Ok(conditioning)
        }
        _ => Ok(Vec::new()),
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let route = required(&args, "--variant")?;
    if let Some(declared) = arg(&args, "--sc20686-route") {
        if declared != route {
            return Err("--sc20686-route differs from --variant".into());
        }
    }
    let campaign = args.iter().any(|value| value == "--sc20686-campaign");
    let cancel_arm = args.iter().any(|value| value == "--sc20686-cancel");
    // Campaign mode must name the frozen residency explicitly; an ordinary render uses it.
    let residency = if campaign {
        required(&args, "--sc20686-residency")?
    } else {
        ROUTES
            .iter()
            .find(|(id, _)| *id == route)
            .map(|(_, residency)| (*residency).to_owned())
            .ok_or_else(|| format!("unsupported SC-20686 Wan route: {route}"))?
    };
    let policy = product_policy(&route, &residency)?;
    let _campaign_request = if campaign {
        let events = arg(&args, "--sc20686-events")
            .filter(|path| path != "-")
            .ok_or("SC-20686 campaign requires a dedicated --sc20686-events <file>")?;
        let request = mlx_gen::sc20686::request_output(
            events,
            required(&args, "--sc20686-source-ref")?,
            residency.clone(),
        )?;
        Some(if cancel_arm {
            request.arm_cancellation()
        } else {
            request.arm()
        })
    } else {
        if cancel_arm {
            return Err("--sc20686-cancel requires --sc20686-campaign".into());
        }
        None
    };

    let snapshot = PathBuf::from(required(&args, "--snapshot")?);
    let out = PathBuf::from(required(&args, "--out")?);
    let request = GenerationRequest {
        prompt: parsed_or(
            &args,
            "--prompt",
            "SC-20686 representative still-motion study".to_owned(),
        )?,
        negative_prompt: arg(&args, "--negative"),
        width: parsed_or(&args, "--width", 512)?,
        height: parsed_or(&args, "--height", 512)?,
        frames: Some(parsed_or(&args, "--frames", 17)?),
        count: 1,
        seed: Some(parsed_or(&args, "--seed", 42)?),
        steps: Some(parsed_or(&args, "--steps", 4)?),
        guidance: Some(parsed_or(&args, "--guidance", 5.0)?),
        conditioning: conditioning(&route, &args)?,
        ..Default::default()
    };

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot)).with_offload_policy(policy);
    let generator = load(&route, &spec)?;
    let mut on_progress = |progress: Progress| match progress {
        Progress::Step { current, total } => eprintln!("[sc20686-wan] step {current}/{total}"),
        Progress::Decoding => eprintln!("[sc20686-wan] decoding"),
        Progress::Loading(phase) => eprintln!("[sc20686-wan] loading {phase:?}"),
    };
    let output = match generator.generate(&request, &mut on_progress) {
        Ok(output) => output,
        Err(_) if cancel_arm && mlx_gen::sc20686::campaign_cancelled() => {
            eprintln!("[sc20686-wan] expected SC-20686 campaign cancellation");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    if cancel_arm {
        return Err(
            "the SC-20686 cancellation arm completed without a product cancellation".into(),
        );
    }
    let GenerationOutput::Video { frames, .. } = output else {
        return Err("expected a video output".into());
    };
    if frames.is_empty() {
        return Err("the render produced no frames".into());
    }
    std::fs::create_dir_all(&out)?;
    for (index, frame) in frames.iter().enumerate() {
        let buffer = image::RgbImage::from_raw(frame.width, frame.height, frame.pixels.clone())
            .ok_or("invalid RGB frame dimensions")?;
        buffer.save(out.join(format!("frame_{index:04}.png")))?;
    }
    eprintln!(
        "[sc20686-wan] wrote {} frames to {}",
        frames.len(),
        out.display()
    );
    Ok(())
}
