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
//! [--reference <png>]` (VACE and VACE-Fun); `--lightning on|off` on the A14B routes (the product
//! default is on; campaign mode must state it), with `--lora-high <file> --lora-low <file>` naming
//! the product's per-architecture Lightning pair when on (which forces the 4-step, guidance-1
//! recipe). The campaign adapter owns every `--sc20686-*` flag.

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Progress,
    ReplacementMode,
};
use mlx_gen_wan::product_load;

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
    "--lightning",
    "--lora-high",
    "--lora-low",
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

/// The route's Lightning pair: `--lightning on|off` on the routes that bake Lightning (an ordinary
/// render may omit it and gets the product default, on), refused on every other route. On requires
/// `--lora-high`/`--lora-low`; off forbids them.
fn lightning_pair(route: &str, args: &Args) -> Result<Option<(PathBuf, PathBuf)>> {
    let bakes = product_load::lightning_default(route);
    let loras = args.get("--lora-high").is_some() || args.get("--lora-low").is_some();
    if !bakes {
        if args.get("--lightning").is_some() || loras {
            return Err(format!("{route} has no Lightning toggle").into());
        }
        return Ok(None);
    }
    let on = match args.get("--lightning") {
        Some("on") => true,
        Some("off") => false,
        Some(other) => return Err(format!("--lightning must be on or off, not {other}").into()),
        None if args.has("--sc20686-campaign") => {
            return Err("SC-20686 campaign requires an explicit --lightning on|off".into())
        }
        None => true,
    };
    if !on {
        if loras {
            return Err("--lora-high/--lora-low require --lightning on".into());
        }
        return Ok(None);
    }
    Ok(Some((
        PathBuf::from(args.required("--lora-high")?),
        PathBuf::from(args.required("--lora-low")?),
    )))
}

/// The route's `LoadSpec`: the product's own load decisions (`product_load_spec`) at the frozen
/// campaign residency. Never assembled here, so the campaign measures the product's memory shape.
/// A Lightning pair must be exactly the files the product resolves in that Lightning snapshot.
fn route_load_spec(
    route: &str,
    residency: &str,
    snapshot: &Path,
    lightning: Option<&(PathBuf, PathBuf)>,
) -> Result<LoadSpec> {
    let policy = product_policy(route, residency)?;
    let lightning_snapshot = lightning
        .map(|(high, _)| {
            high.parent()
                .and_then(Path::parent)
                .ok_or("--lora-high is not inside a Lightning snapshot")
        })
        .transpose()?;
    let spec = product_load::product_load_spec(
        route,
        snapshot,
        policy,
        lightning.is_some(),
        lightning_snapshot,
    )?;
    if let Some((high, low)) = lightning {
        let given = [high.canonicalize()?, low.canonicalize()?];
        let product = spec
            .adapters
            .iter()
            .map(|adapter| adapter.path.canonicalize())
            .collect::<std::io::Result<Vec<_>>>()?;
        if product != given {
            return Err(
                format!("{route}: the LoRA pair is not the product's Lightning pair").into(),
            );
        }
    }
    Ok(spec)
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
    // Refuse a non-product residency or Lightning toggle before arming the observer.
    product_policy(&route, &residency)?;
    let lightning = lightning_pair(&route, &args)?;
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

    if lightning.is_some()
        && (request.steps != Some(product_load::LIGHTNING_STEPS)
            || request.guidance != Some(product_load::LIGHTNING_GUIDANCE))
    {
        return Err(format!(
            "the product's Lightning recipe is {} steps at guidance {}",
            product_load::LIGHTNING_STEPS,
            product_load::LIGHTNING_GUIDANCE
        )
        .into());
    }
    let spec = route_load_spec(&route, &residency, &snapshot, lightning.as_ref())?;
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
    use mlx_gen::gen_core::{MoeExpert, Precision, Quant, WeightsSource};

    /// The product's default-request load quantization per route (`product_load::load_quant`, which
    /// the SceneWorks worker calls): none on the packed q4 tiers and `wan_vace`, Q4 on VACE-Fun.
    const PRODUCT_QUANT: [(&str, Option<Quant>); 5] = [
        ("wan2_2_ti2v_5b", None),
        ("wan2_2_t2v_14b", None),
        ("wan2_2_i2v_14b", None),
        ("wan_vace", None),
        ("wan2_2_vace_fun_14b", Some(Quant::Q4)),
    ];

    fn write(path: &Path, contents: &str) {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("dir");
        std::fs::write(path, contents).expect("write");
    }

    fn args(raw: &[&str]) -> Args {
        let raw: Vec<String> = std::iter::once("sc20686_wan")
            .chain(raw.iter().copied())
            .map(str::to_owned)
            .collect();
        Args::parse(&raw).expect("arguments")
    }

    #[test]
    fn every_route_loads_with_the_product_settings() {
        let root = tempfile::tempdir().expect("temp root");
        let q4 = root.path().join("rev/q4");
        write(
            &q4.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        );
        // The Mac product's `wan_vace` transformer is Wan2.1-VACE-1.3B (12 x 128 heads, 30 layers).
        let vace = root.path().join("wan_vace");
        write(
            &vace.join("transformer/config.json"),
            r#"{"num_attention_heads": 12, "attention_head_dim": 128, "num_layers": 30}"#,
        );
        let loras = root.path().join("lightning");
        for route in ["wan2_2_t2v_14b", "wan2_2_i2v_14b"] {
            let subdir = loras.join(product_load::lightning_subdir(route).expect("A14B"));
            write(&subdir.join("high_noise_model.safetensors"), "high");
            write(&subdir.join("low_noise_model.safetensors"), "low");
        }
        assert_eq!(ROUTES.len(), PRODUCT_QUANT.len());
        for (route, residency) in ROUTES {
            let snapshot = match route {
                "wan_vace" => vace.clone(),
                "wan2_2_vace_fun_14b" => root.path().join("vace_fun"),
                _ => q4.clone(),
            };
            let quant = PRODUCT_QUANT
                .iter()
                .find(|(id, _)| *id == route)
                .map(|(_, quant)| *quant)
                .expect(route);
            let policy = match residency {
                "sequential" => OffloadPolicy::Sequential,
                _ => OffloadPolicy::Resident,
            };
            let pair = product_load::lightning_subdir(route).map(|subdir| {
                (
                    loras.join(subdir).join("high_noise_model.safetensors"),
                    loras.join(subdir).join("low_noise_model.safetensors"),
                )
            });
            // Lightning off everywhere, then the product default (on) where the route bakes it.
            for lightning in [None, pair.as_ref()] {
                let spec = route_load_spec(route, residency, &snapshot, lightning).expect(route);
                assert_eq!(spec.quantize, quant, "{route}");
                assert_eq!(spec.precision, Precision::Bf16, "{route}");
                assert_eq!(spec.offload_policy, policy, "{route}");
                assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if *dir == snapshot));
                assert!(spec.text_encoder.is_none(), "{route}");
                assert!(spec.components.is_empty(), "{route}");
                assert!(spec.resolved_route.is_none(), "{route}");
                let adapters: Vec<_> = spec
                    .adapters
                    .iter()
                    .map(|a| (a.path.clone(), a.scale, a.moe_expert))
                    .collect();
                let expected = lightning
                    .map(|(high, low)| {
                        vec![
                            (high.clone(), 1.0, Some(MoeExpert::High)),
                            (low.clone(), 1.0, Some(MoeExpert::Low)),
                        ]
                    })
                    .unwrap_or_default();
                assert_eq!(adapters, expected, "{route}");
            }
        }
        // A pair from the other architecture is not the product's Lightning pair.
        let (i2v_high, i2v_low) = (
            loras
                .join(product_load::lightning_subdir("wan2_2_i2v_14b").unwrap())
                .join("high_noise_model.safetensors"),
            loras
                .join(product_load::lightning_subdir("wan2_2_i2v_14b").unwrap())
                .join("low_noise_model.safetensors"),
        );
        assert!(route_load_spec(
            "wan2_2_t2v_14b",
            "sequential",
            &q4,
            Some(&(i2v_high, i2v_low))
        )
        .is_err());
    }

    #[test]
    fn the_lightning_toggle_follows_the_product() {
        // Product default: on for the A14B routes; campaign mode must state it.
        assert!(lightning_pair("wan2_2_t2v_14b", &args(&[])).is_err());
        assert!(lightning_pair(
            "wan2_2_t2v_14b",
            &args(&["--sc20686-campaign", "--lightning", "off"])
        )
        .unwrap()
        .is_none());
        // Even with the pair present, campaign mode never falls back to the default.
        assert!(lightning_pair(
            "wan2_2_t2v_14b",
            &args(&[
                "--sc20686-campaign",
                "--lora-high",
                "/h",
                "--lora-low",
                "/l"
            ])
        )
        .is_err());
        let on = lightning_pair(
            "wan2_2_i2v_14b",
            &args(&["--lightning", "on", "--lora-high", "/h", "--lora-low", "/l"]),
        )
        .unwrap();
        assert_eq!(on, Some((PathBuf::from("/h"), PathBuf::from("/l"))));
        // Routes without Lightning refuse the toggle.
        assert!(lightning_pair("wan_vace", &args(&[])).unwrap().is_none());
        assert!(lightning_pair("wan_vace", &args(&["--lightning", "off"])).is_err());
    }
}
