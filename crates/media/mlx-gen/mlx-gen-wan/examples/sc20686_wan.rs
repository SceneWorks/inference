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
//! default is on; campaign mode must state it). When on, `--lightning-hub <hf hub dir>` is the
//! Hugging Face hub the product resolves the `lightx2v/Wan2.2-Lightning` snapshot from, and
//! `--lora-high <file> --lora-low <file>` must be the product's per-architecture pair in it (which
//! forces the 4-step, guidance-1 recipe). The campaign adapter owns every `--sc20686-*` flag.

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
    "--sc20686-budget-bytes",
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
    "--lightning-hub",
    "--lora-high",
    "--lora-low",
];
const SWITCHES: &[&str] = &[
    "--sc20686-campaign",
    "--sc20686-cancel",
    "--sc20686-schedule-control",
    "--sc20686-estimate",
];

/// Schema of the one JSON line `--sc20686-estimate` prints (read by the campaign adapter).
const ESTIMATE_SCHEMA: &str = "sc20686-product-admission-estimate-v1";

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
            None if self.strict() => {
                Err(format!("SC-20686 campaign requires an explicit {key}").into())
            }
            None => Ok(default),
        }
    }

    /// Campaign and estimate modes state every coordinate argument explicitly: the estimate prices
    /// exactly the request the campaign run will make.
    fn strict(&self) -> bool {
        self.has("--sc20686-campaign") || self.has("--sc20686-estimate")
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

/// The A14B Lightning inputs: the Hugging Face hub holding `lightx2v/Wan2.2-Lightning` (the
/// snapshot is resolved from it exactly as the product resolves it) and the sealed LoRA pair the
/// campaign hashed, which may arrive symlink-resolved to its blobs.
#[derive(Debug, PartialEq)]
struct Lightning {
    hub: PathBuf,
    high: PathBuf,
    low: PathBuf,
}

/// The route's Lightning inputs: `--lightning on|off` on the routes that bake Lightning (an ordinary
/// render may omit it and gets the product default, on), refused on every other route. On requires
/// `--lightning-hub`, `--lora-high` and `--lora-low`; off forbids them.
fn lightning_pair(route: &str, args: &Args) -> Result<Option<Lightning>> {
    let bakes = product_load::lightning_default(route);
    let inputs = ["--lightning-hub", "--lora-high", "--lora-low"]
        .iter()
        .any(|flag| args.get(flag).is_some());
    if !bakes {
        if args.get("--lightning").is_some() || inputs {
            return Err(format!("{route} has no Lightning toggle").into());
        }
        return Ok(None);
    }
    let on = match args.get("--lightning") {
        Some("on") => true,
        Some("off") => false,
        Some(other) => return Err(format!("--lightning must be on or off, not {other}").into()),
        None if args.strict() => {
            return Err("SC-20686 campaign requires an explicit --lightning on|off".into())
        }
        None => true,
    };
    if !on {
        if inputs {
            return Err("--lightning-hub/--lora-high/--lora-low require --lightning on".into());
        }
        return Ok(None);
    }
    Ok(Some(Lightning {
        hub: PathBuf::from(args.required("--lightning-hub")?),
        high: PathBuf::from(args.required("--lora-high")?),
        low: PathBuf::from(args.required("--lora-low")?),
    }))
}

/// The route's `LoadSpec`: the product's own load decisions (`product_load_spec`) at the frozen
/// campaign residency. Never assembled here, so the campaign measures the product's memory shape.
/// The Lightning snapshot is resolved from the hub by the product resolver (never derived from a
/// LoRA path, which the adapter hands over symlink-resolved to its blob), and the sealed pair must
/// be byte-for-byte the files the product loads from it.
fn route_load_spec(
    route: &str,
    residency: &str,
    snapshot: &Path,
    lightning: Option<&Lightning>,
) -> Result<LoadSpec> {
    let policy = product_policy(route, residency)?;
    let lightning_snapshot = lightning
        .map(|lightning| product_load::lightning_snapshot_in_hub(&lightning.hub))
        .transpose()?;
    let spec = product_load::product_load_spec(
        route,
        snapshot,
        policy,
        lightning.is_some(),
        lightning_snapshot.as_deref(),
    )?;
    if let Some(lightning) = lightning {
        let given = [
            lightning.high.canonicalize()?,
            lightning.low.canonicalize()?,
        ];
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
    let estimate = args.has("--sc20686-estimate");
    if estimate && campaign {
        return Err(
            "--sc20686-estimate prices a run; it is exclusive with --sc20686-campaign".into(),
        );
    }
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
    // Campaign mode must name the frozen residency explicitly; an ordinary render (and the
    // estimate, which prices the campaign's product residency) uses it.
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
    if estimate {
        // The product admission estimate of exactly this run: no weights load, MLX is untouched.
        // The admission budget (`available - reserve` at spawn time): the decode is priced at the
        // product planner's decision with it, and the adapter pins the run to the same decision.
        let budget = args
            .get("--sc20686-budget-bytes")
            .map(|raw| {
                raw.parse::<u64>()
                    .map_err(|_| "--sc20686-budget-bytes is malformed")
            })
            .transpose()?;
        let priced = mlx_gen_wan::admission_estimate::product_admission_estimate(
            &route, &spec, &request, budget,
        )?;
        println!("{}", estimate_line(&route, &priced));
        return Ok(());
    }
    let out = PathBuf::from(args.required("--out")?);
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

/// The `--sc20686-estimate` output: the estimate, its source and every priced component.
fn estimate_line(
    route: &str,
    priced: &mlx_gen_wan::admission_estimate::WanAdmissionEstimate,
) -> serde_json::Value {
    serde_json::json!({
        "schema": ESTIMATE_SCHEMA,
        "route": route,
        "source": "product-admission-profile",
        "estimateBytes": priced.peak_bytes(),
        "decodeMode": priced.decode.map_or("single-pass-conservative", |decode| decode.mode.as_str()),
        "decodeSafeBudgetGib": priced.decode.map(|decode| decode.safe_budget_gib),
        "phases": priced.phases().iter().map(|(phase, bytes)| (phase.to_string(), serde_json::json!(bytes))).collect::<serde_json::Map<_, _>>(),
        "components": {
            "textEncoderBytes": priced.text_encoder_bytes,
            "vaeBytes": priced.vae_bytes,
            "ditResidentBytes": priced.dit_resident_bytes,
            "denoiseActivationBytes": priced.denoise_activation_bytes,
            "encodeWorkingSetBytes": priced.encode_working_set_bytes,
            "decodeWorkingSetBytes": priced.decode_working_set_bytes,
            "ditLoadTransientBytes": priced.dit_load_transient_bytes,
        },
    })
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
        // A synthetic Hugging Face hub laid out as the cache stores Lightning: snapshot files are
        // symlinks into `blobs/`, and `refs/main` names the snapshot.
        let hub = root.path().join("hub");
        let repo = hub.join("models--lightx2v--Wan2.2-Lightning");
        let lightning_snapshot = repo
            .join("snapshots")
            .join(product_load::LIGHTNING_REVISION);
        write(&repo.join("refs/main"), product_load::LIGHTNING_REVISION);
        for route in ["wan2_2_t2v_14b", "wan2_2_i2v_14b"] {
            let subdir = product_load::lightning_subdir(route).expect("A14B");
            for name in [
                "high_noise_model.safetensors",
                "low_noise_model.safetensors",
            ] {
                let blob = repo.join("blobs").join(format!("{subdir}-{name}"));
                write(&blob, name);
                let link = lightning_snapshot.join(subdir).join(name);
                std::fs::create_dir_all(link.parent().unwrap()).unwrap();
                std::os::unix::fs::symlink(&blob, &link).unwrap();
            }
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
            // The adapter seals the pair by path and hands it over symlink-resolved (blob paths).
            let pair = product_load::lightning_subdir(route).map(|subdir| Lightning {
                hub: hub.clone(),
                high: lightning_snapshot
                    .join(subdir)
                    .join("high_noise_model.safetensors")
                    .canonicalize()
                    .unwrap(),
                low: lightning_snapshot
                    .join(subdir)
                    .join("low_noise_model.safetensors")
                    .canonicalize()
                    .unwrap(),
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
                // The product loads the pair from the resolved snapshot, never the repo root.
                let expected = product_load::lightning_subdir(route)
                    .filter(|_| lightning.is_some())
                    .map(|subdir| {
                        vec![
                            (
                                lightning_snapshot
                                    .join(subdir)
                                    .join("high_noise_model.safetensors"),
                                1.0,
                                Some(MoeExpert::High),
                            ),
                            (
                                lightning_snapshot
                                    .join(subdir)
                                    .join("low_noise_model.safetensors"),
                                1.0,
                                Some(MoeExpert::Low),
                            ),
                        ]
                    })
                    .unwrap_or_default();
                assert_eq!(adapters, expected, "{route}");
            }
        }
        // A pair from the other architecture is not the product's Lightning pair.
        let i2v =
            lightning_snapshot.join(product_load::lightning_subdir("wan2_2_i2v_14b").unwrap());
        let other_architecture = Lightning {
            hub: hub.clone(),
            high: i2v.join("high_noise_model.safetensors"),
            low: i2v.join("low_noise_model.safetensors"),
        };
        assert!(route_load_spec(
            "wan2_2_t2v_14b",
            "sequential",
            &q4,
            Some(&other_architecture)
        )
        .is_err());
    }

    /// `--sc20686-estimate` prices exactly the campaign's request: every coordinate argument and the
    /// Lightning toggle must be stated, never defaulted.
    #[test]
    fn the_estimate_mode_is_as_strict_as_the_campaign() {
        assert_eq!(args(&[]).coordinate("--width", 512_u32).unwrap(), 512);
        assert!(args(&["--sc20686-estimate"])
            .coordinate("--width", 512_u32)
            .is_err());
        assert_eq!(
            args(&["--sc20686-estimate", "--width", "768"])
                .coordinate("--width", 512_u32)
                .unwrap(),
            768
        );
        assert!(lightning_pair("wan2_2_t2v_14b", &args(&["--sc20686-estimate"])).is_err());
        assert!(lightning_pair(
            "wan2_2_t2v_14b",
            &args(&["--sc20686-estimate", "--lightning", "off"])
        )
        .unwrap()
        .is_none());
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
        // Even with the inputs present, campaign mode never falls back to the default.
        let inputs = [
            "--lightning-hub",
            "/hub",
            "--lora-high",
            "/h",
            "--lora-low",
            "/l",
        ];
        let campaign: Vec<&str> = std::iter::once("--sc20686-campaign")
            .chain(inputs)
            .collect();
        assert!(lightning_pair("wan2_2_t2v_14b", &args(&campaign)).is_err());
        let on: Vec<&str> = ["--lightning", "on"].into_iter().chain(inputs).collect();
        assert_eq!(
            lightning_pair("wan2_2_i2v_14b", &args(&on)).unwrap(),
            Some(Lightning {
                hub: PathBuf::from("/hub"),
                high: PathBuf::from("/h"),
                low: PathBuf::from("/l"),
            })
        );
        // On without the hub is refused: the snapshot is never guessed from the LoRA paths.
        assert!(lightning_pair(
            "wan2_2_i2v_14b",
            &args(&["--lightning", "on", "--lora-high", "/h", "--lora-low", "/l"])
        )
        .is_err());
        // Routes without Lightning refuse the toggle.
        assert!(lightning_pair("wan_vace", &args(&[])).unwrap().is_none());
        assert!(lightning_pair("wan_vace", &args(&["--lightning", "off"])).is_err());
    }
}
