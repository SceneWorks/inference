//! SC-20686 Metal-lane product entrypoint for the FLUX.2 Klein reference-edit routes (MLX):
//! `flux2_klein_9b_edit` (reference K/V recomputed every denoise evaluation) and
//! `flux2_klein_9b_kv_edit` (persistent reference-K/V cache extracted on step 0).
//!
//! Loads the route through the MLX provider loader the SceneWorks worker uses, with the worker's
//! `LoadSpec` for it ([`mlx_gen_flux2::product_load::product_load_spec`]), runs one real edit
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
//! Arguments are strict: unknown (including the Candle harness's `--single-only`) or repeated flags
//! are refused, and campaign mode requires every coordinate argument (`--seed` included).

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, Progress,
};
use mlx_gen::media::Image;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Frozen SceneWorks-equivalent residency for both FLUX.2 edit routes.
const PRODUCT_RESIDENCY: &str = "sequential";

/// Strict command line: every flag is known, appears at most once, and value flags carry a value.
/// Unknown or repeated flags are refused (an ambiguous campaign input is never resolved by picking
/// one), and campaign mode requires every coordinate argument explicitly.
struct Args {
    values: std::collections::BTreeMap<String, String>,
    switches: std::collections::BTreeSet<String>,
}

const VALUE_FLAGS: &[&str] = &[
    "--variant",
    "--sc20686-events",
    "--sc20686-source-ref",
    "--sc20686-residency",
    "--sc20686-budget-bytes",
    "--snapshot",
    "--out",
    "--prompt",
    "--width",
    "--height",
    "--seed",
    "--steps",
    "--guidance",
    "--reference",
    "--reference2",
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
            None if self.has("--sc20686-campaign") || self.has("--sc20686-estimate") => {
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

/// The route's `LoadSpec`: the worker's own load decisions (`product_load_spec`) at the frozen
/// campaign residency. Never assembled here, so the campaign measures the product's memory shape.
fn route_load_spec(route: &str, snapshot: &Path) -> Result<LoadSpec> {
    Ok(mlx_gen_flux2::product_load::product_load_spec(
        route,
        snapshot,
        OffloadPolicy::Sequential,
    )?)
}

fn main() -> Result<()> {
    let args = Args::parse(&std::env::args().collect::<Vec<_>>())?;
    let route = args.required("--variant")?;
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
    if campaign {
        let residency = args.required("--sc20686-residency")?;
        if residency != PRODUCT_RESIDENCY {
            return Err(format!("{route} campaign residency must be {PRODUCT_RESIDENCY}").into());
        }
    }
    let _campaign_request = if campaign {
        let events = args
            .get("--sc20686-events")
            .filter(|path| *path != "-")
            .ok_or("SC-20686 campaign requires a dedicated --sc20686-events <file>")?;
        let request = mlx_gen::sc20686::request_output(
            events,
            args.required("--sc20686-source-ref")?,
            PRODUCT_RESIDENCY,
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

    let mut references = vec![load_image(Path::new(&args.required("--reference")?))?];
    if let Some(second) = args.get("--reference2") {
        references.push(load_image(Path::new(second))?);
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
        prompt: args.coordinate("--prompt", "SC-20686 edit one reference".to_owned())?,
        width: args.coordinate("--width", 512)?,
        height: args.coordinate("--height", 512)?,
        count: 1,
        seed: Some(args.coordinate("--seed", 42)?),
        steps: Some(args.coordinate("--steps", 4)?),
        guidance: Some(args.coordinate("--guidance", 1.0)?),
        conditioning,
        ..Default::default()
    };

    let spec = route_load_spec(&route, Path::new(&args.required("--snapshot")?))?;
    if estimate {
        // The product admission estimate of exactly this run: no weights load, MLX is untouched.
        let priced =
            mlx_gen_flux2::admission_estimate::product_admission_estimate(&route, &spec, &request)?;
        println!(
            "{}",
            serde_json::json!({
                "schema": ESTIMATE_SCHEMA,
                "route": route,
                "source": "product-admission-profile",
                "estimateBytes": priced.peak_bytes(),
                // An image VAE decode: no budget-planned tiling (the budget, if given, is unused).
                "decodeMode": "single-pass",
                "decodeSafeBudgetGib": null,
                "phases": priced.phases().iter().map(|(phase, bytes)| (phase.to_string(), serde_json::json!(bytes))).collect::<serde_json::Map<_, _>>(),
                "components": {
                    "textEncoderBytes": priced.text_encoder_bytes,
                    "transformerBytes": priced.transformer_bytes,
                    "vaeBytes": priced.vae_bytes,
                    "activationBytes": priced.activation_bytes,
                    "referenceKvBytes": priced.reference_kv_bytes,
                },
            })
        );
        return Ok(());
    }
    let generator = match route.as_str() {
        "flux2_klein_9b_edit" => mlx_gen_flux2::load_klein_9b_edit(&spec)?,
        "flux2_klein_9b_kv_edit" => mlx_gen_flux2::load_klein_9b_kv_edit(&spec)?,
        other => return Err(format!("unsupported SC-20686 FLUX.2 route: {other}").into()),
    };
    let out = PathBuf::from(args.required("--out")?);
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{Precision, WeightsSource};

    /// The product's default Klein edit load (`product_load`, which the SceneWorks worker calls): the
    /// packed q4 tier root, no load-time quant, and the catalog model as `resolved_route`.
    #[test]
    fn both_routes_load_with_the_product_settings() {
        let root = tempfile::tempdir().expect("temp root");
        let snapshot = root.path();
        std::fs::create_dir_all(snapshot.join("transformer")).expect("transformer");
        std::fs::write(
            snapshot.join("transformer/config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .expect("config");
        for (route, catalog) in [
            ("flux2_klein_9b_edit", "flux2_klein_9b"),
            ("flux2_klein_9b_kv_edit", "flux2_klein_9b_kv"),
        ] {
            let spec = route_load_spec(route, snapshot).expect(route);
            assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if dir == snapshot));
            assert_eq!(spec.resolved_route.as_deref(), Some(catalog), "{route}");
            assert_eq!(spec.quantize, None, "{route}");
            assert_eq!(spec.precision, Precision::Bf16, "{route}");
            assert_eq!(spec.offload_policy, OffloadPolicy::Sequential, "{route}");
            assert!(spec.adapters.is_empty(), "{route}");
            assert!(spec.pid.is_none(), "{route}");
            assert!(spec.components.is_empty(), "{route}");
        }
    }
}
