//! SC-20686 Metal-lane product entrypoint for the five registered Wan routes (MLX).
//!
//! Loads the requested route through the same MLX provider loader the SceneWorks worker uses, with
//! the worker's `LoadSpec` for that route ([`mlx_gen_wan::product_load::product_load_spec`]: the
//! forced Q4 on VACE-Fun, the dense Wan2.1-VACE-1.3B `wan_vace`, the packed tier roots), builds the
//! request from the frozen campaign coordinate, runs one real `generate`, and writes the
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
    ReplacementMode,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const ROUTES: [(&str, &str); 5] = [
    ("wan2_2_ti2v_5b", "sequential"),
    ("wan2_2_t2v_14b", "sequential"),
    ("wan2_2_i2v_14b", "sequential"),
    ("wan_vace", "resident"),
    ("wan2_2_vace_fun_14b", "sequential"),
];

/// Strict command line: every flag is known, appears at most once, and value flags carry a value.
/// Unknown or repeated flags are refused (an ambiguous campaign input is never resolved by picking
/// one), and campaign mode requires every coordinate argument explicitly.
struct Args {
    values: std::collections::BTreeMap<String, String>,
    switches: std::collections::BTreeSet<String>,
}

const VALUE_FLAGS: &[&str] = &[
    "--variant",
    "--sc20686-route",
    "--sc20686-events",
    "--sc20686-source-ref",
    "--sc20686-residency",
    "--snapshot",
    "--out",
    "--prompt",
    "--negative",
    "--width",
    "--height",
    "--frames",
    "--seed",
    "--steps",
    "--guidance",
    "--image",
    "--control-dir",
    "--mask-dir",
    "--reference",
];
const SWITCHES: &[&str] = &[
    "--sc20686-campaign",
    "--sc20686-cancel",
    "--sc20686-schedule-control",
];

impl Args {
    fn parse(raw: &[String]) -> Result<Self> {
        let mut values = std::collections::BTreeMap::new();
        let mut switches = std::collections::BTreeSet::new();
        let mut items = raw.iter().skip(1);
        while let Some(flag) = items.next() {
            let fresh = if SWITCHES.contains(&flag.as_str()) {
                switches.insert(flag.clone())
            } else if VALUE_FLAGS.contains(&flag.as_str()) {
                let value = items
                    .next()
                    .ok_or_else(|| format!("{flag} requires a value"))?;
                values.insert(flag.clone(), value.clone()).is_none()
            } else {
                return Err(format!("unsupported argument: {flag}").into());
            };
            if !fresh {
                return Err(format!("repeated argument: {flag}").into());
            }
        }
        Ok(Self { values, switches })
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    fn has(&self, key: &str) -> bool {
        self.switches.contains(key)
    }

    fn required(&self, key: &str) -> Result<String> {
        self.get(key)
            .map(str::to_owned)
            .ok_or_else(|| format!("missing {key}").into())
    }

    /// A coordinate argument: required in campaign mode, `default` for an ordinary render.
    fn coordinate<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T> {
        match self.get(key) {
            Some(value) => value
                .parse()
                .map_err(|_| format!("{key} is malformed").into()),
            None if self.has("--sc20686-campaign") => {
                Err(format!("SC-20686 campaign requires an explicit {key}").into())
            }
            None => Ok(default),
        }
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

/// The route's `LoadSpec`: the worker's own load decisions (`product_load_spec`) at the frozen
/// campaign residency. Never assembled here, so the campaign measures the product's memory shape.
fn route_load_spec(route: &str, residency: &str, snapshot: &Path) -> Result<LoadSpec> {
    let policy = product_policy(route, residency)?;
    Ok(mlx_gen_wan::product_load::product_load_spec(
        route, snapshot, policy,
    )?)
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

fn conditioning(route: &str, args: &Args) -> Result<Vec<Conditioning>> {
    match route {
        "wan2_2_i2v_14b" => Ok(vec![Conditioning::Reference {
            image: load_image(Path::new(&args.required("--image")?))?,
            strength: None,
        }]),
        "wan_vace" | "wan2_2_vace_fun_14b" => {
            let frames = load_png_dir(Path::new(&args.required("--control-dir")?))?;
            let mask = load_png_dir(Path::new(&args.required("--mask-dir")?))?;
            let mut conditioning = vec![Conditioning::ControlClip {
                frames,
                mask,
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            }];
            if let Some(reference) = args.get("--reference") {
                conditioning.push(Conditioning::Reference {
                    image: load_image(Path::new(reference))?,
                    strength: None,
                });
            }
            Ok(conditioning)
        }
        _ => Ok(Vec::new()),
    }
}

fn main() -> Result<()> {
    let args = Args::parse(&std::env::args().collect::<Vec<_>>())?;
    let route = args.required("--variant")?;
    if let Some(declared) = args.get("--sc20686-route") {
        if declared != route {
            return Err("--sc20686-route differs from --variant".into());
        }
    }
    let campaign = args.has("--sc20686-campaign");
    let cancel_arm = args.has("--sc20686-cancel");
    let control_arm = args.has("--sc20686-schedule-control");
    if !campaign && (cancel_arm || control_arm) {
        return Err(
            "--sc20686-cancel/--sc20686-schedule-control require --sc20686-campaign".into(),
        );
    }
    if cancel_arm && control_arm {
        return Err("the cancellation and schedule-control arms are exclusive".into());
    }
    // Campaign mode must name the frozen residency explicitly; an ordinary render uses it.
    let residency = if campaign {
        args.required("--sc20686-residency")?
    } else {
        ROUTES
            .iter()
            .find(|(id, _)| *id == route)
            .map(|(_, residency)| (*residency).to_owned())
            .ok_or_else(|| format!("unsupported SC-20686 Wan route: {route}"))?
    };
    // Refuse a non-product residency before arming the observer.
    product_policy(&route, &residency)?;
    let _campaign_request = if campaign {
        let events = args
            .get("--sc20686-events")
            .filter(|path| *path != "-")
            .ok_or("SC-20686 campaign requires a dedicated --sc20686-events <file>")?;
        let request = mlx_gen::sc20686::request_output(
            events,
            args.required("--sc20686-source-ref")?,
            residency.clone(),
        )?;
        Some(if cancel_arm {
            request.arm_cancellation()
        } else if control_arm {
            request.arm_schedule_control()
        } else {
            request.arm()
        })
    } else {
        None
    };

    let snapshot = PathBuf::from(args.required("--snapshot")?);
    let out = PathBuf::from(args.required("--out")?);
    let request = GenerationRequest {
        prompt: args.coordinate(
            "--prompt",
            "SC-20686 representative still-motion study".to_owned(),
        )?,
        negative_prompt: args.get("--negative").map(str::to_owned),
        width: args.coordinate("--width", 512)?,
        height: args.coordinate("--height", 512)?,
        frames: Some(args.coordinate("--frames", 17)?),
        count: 1,
        seed: Some(args.coordinate("--seed", 42)?),
        steps: Some(args.coordinate("--steps", 4)?),
        guidance: Some(args.coordinate("--guidance", 5.0)?),
        conditioning: conditioning(&route, &args)?,
        ..Default::default()
    };

    let spec = route_load_spec(&route, &residency, &snapshot)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{Precision, Quant, WeightsSource};

    /// The SceneWorks Mac worker's default-request load quantization per route
    /// (`video_jobs/wan.rs::resolve_wan_tier_dir_and_quant`, `video_jobs/vace.rs`).
    const WORKER_QUANT: [(&str, Option<Quant>); 5] = [
        ("wan2_2_ti2v_5b", None),
        ("wan2_2_t2v_14b", None),
        ("wan2_2_i2v_14b", None),
        ("wan_vace", None),
        ("wan2_2_vace_fun_14b", Some(Quant::Q4)),
    ];

    #[test]
    fn every_route_loads_with_the_product_settings() {
        // The Mac product's `wan_vace` transformer is Wan2.1-VACE-1.3B (12 x 128 heads, 30 layers).
        let vace = tempfile::tempdir().expect("temp root");
        std::fs::create_dir_all(vace.path().join("transformer")).expect("transformer dir");
        std::fs::write(
            vace.path().join("transformer/config.json"),
            r#"{"num_attention_heads": 12, "attention_head_dim": 128, "num_layers": 30}"#,
        )
        .expect("config");
        assert_eq!(ROUTES.len(), WORKER_QUANT.len());
        for (route, residency) in ROUTES {
            let snapshot = if route == "wan_vace" {
                vace.path().to_path_buf()
            } else {
                PathBuf::from("/snapshots/rev/q4")
            };
            let spec = route_load_spec(route, residency, &snapshot).expect(route);
            let quant = WORKER_QUANT
                .iter()
                .find(|(id, _)| *id == route)
                .map(|(_, quant)| *quant)
                .expect(route);
            assert_eq!(spec.quantize, quant, "{route}");
            assert_eq!(spec.precision, Precision::Bf16, "{route}");
            let policy = match residency {
                "sequential" => OffloadPolicy::Sequential,
                _ => OffloadPolicy::Resident,
            };
            assert_eq!(spec.offload_policy, policy, "{route}");
            assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if *dir == snapshot));
            assert!(spec.adapters.is_empty(), "{route}");
            assert!(spec.text_encoder.is_none(), "{route}");
            assert!(spec.components.is_empty(), "{route}");
            assert!(spec.resolved_route.is_none(), "{route}");
        }
    }
}
