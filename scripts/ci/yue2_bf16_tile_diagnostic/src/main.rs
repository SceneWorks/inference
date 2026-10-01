//! Same-input numerical diagnostic for the frozen M3 YuE2 standard VAE.
//! This reports the original 1/64 bound; it does not change a production gate.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use candle_audio::candle_core::{self, DType, Device, Tensor};
use candle_audio_yue2::inventory::{ComponentId, YUE2_VAE_REPO};
use candle_audio_yue2::latent::{AcousticLatents, LatentSource};
use candle_audio_yue2::snapshot::resolve_component;
use candle_audio_yue2::vae::{VaeParts, Yue2Vae};
use candle_audio_yue2::SnapshotDirs;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const ENGINE_SHA: &str = "4127a675fc8575555e029e01b7f6867488880a8f";
const ORIGINAL_BOUND: f32 = 1.0 / 64.0;
const RATIO: usize = 1920;
const SEAM_RADIUS: usize = 128;

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn samples_sha256(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn command_output(args: &[&str], dir: &Path) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git").args(args).current_dir(dir).output()?;
    if !output.status.success() {
        return Err(format!(
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn verified_engine() -> Result<PathBuf, Box<dyn Error>> {
    let dir = required_env("YUE2_ENGINE_ROOT")?.canonicalize()?;
    let head = command_output(&["rev-parse", "HEAD"], &dir)?;
    if head != ENGINE_SHA || !command_output(&["status", "--porcelain"], &dir)?.is_empty() {
        return Err(format!("engine must be clean at {ENGINE_SHA}; observed {head}").into());
    }
    Ok(dir)
}

fn required_env(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(
        std::env::var_os(name).ok_or(format!("{name} is required"))?,
    ))
}

fn arguments() -> Result<(PathBuf, PathBuf), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let (mut reference, mut output) = (None, None);
    while let Some(key) = args.next() {
        let value = PathBuf::from(args.next().ok_or("option requires a path")?);
        match key.to_str() {
            Some("--reference-dir") if reference.is_none() => reference = Some(value),
            Some("--output-dir") if output.is_none() => output = Some(value),
            _ => return Err(format!("unknown or duplicate option {key:?}").into()),
        }
    }
    let reference = reference.ok_or("--reference-dir required")?;
    let output = output.ok_or("--output-dir required")?;
    if !reference.is_absolute() || !output.is_absolute() || output.exists() {
        return Err("reference and fresh output directories must be absolute".into());
    }
    Ok((reference, output))
}

fn write_samples(dir: &Path, name: &str, values: &[f32]) -> Result<Value, Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let file = format!("{name}.f32le");
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.join(&file))?;
    out.write_all(&bytes)?;
    Ok(
        json!({"file":file,"sha256":sha256(&bytes),"bytes":bytes.len(),"layout":"interleaved_stereo_f32le"}),
    )
}

fn interleaved(raw: &Tensor, clamp: bool) -> Result<Vec<f32>, Box<dyn Error>> {
    let raw = raw.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let raw = if clamp { raw.clamp(-1f32, 1f32)? } else { raw };
    Ok(raw
        .squeeze(0)?
        .t()?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

fn decoded(
    vae: &Yue2Vae,
    z: &Tensor,
    tiled: bool,
    core: usize,
    halo: usize,
) -> Result<Tensor, Box<dyn Error>> {
    Ok(if tiled {
        vae.decode_tiled(z, core, halo, &|| false, &mut |_, _| {})?
    } else {
        vae.decode_full(z)?
    })
}

fn capture(
    vae: &Yue2Vae,
    latent: &AcousticLatents,
    tiled: bool,
    core: usize,
    halo: usize,
    out: &Path,
    stem: &str,
) -> Result<Value, Box<dyn Error>> {
    let z = latent
        .to_decoder_input(vae.device())?
        .to_dtype(vae.dtype())?;
    let raw = decoded(vae, &z, tiled, core, halo)?;
    let expected_shape = [1, 2, vae.natural_output_length(latent.frames())?];
    if raw.dtype() != vae.dtype() || raw.dims() != expected_shape {
        return Err(format!(
            "raw VAE output {:?}/{:?}, expected {:?}/{expected_shape:?}",
            raw.dtype(),
            raw.dims(),
            vae.dtype()
        )
        .into());
    }
    let raw_values = interleaved(&raw, false)?;
    let clamped_values = interleaved(&raw, true)?;
    if raw_values
        .iter()
        .chain(&clamped_values)
        .any(|v| !v.is_finite())
    {
        return Err("non-finite waveform".into());
    }
    let raw_artifact = write_samples(out, &format!("{stem}-raw"), &raw_values)?;
    let clamped_artifact = write_samples(out, &format!("{stem}-clamped"), &clamped_values)?;
    Ok(json!({
        "rawSha256": samples_sha256(&raw_values),
        "clampedSha256": samples_sha256(&clamped_values),
        "rawDtype": format!("{:?}", raw.dtype()),
        "rawArtifact": raw_artifact,
        "clampedArtifact": clamped_artifact,
        "raw": raw_values,
        "clamped": clamped_values,
    }))
}

fn nearest_seam_distance(frame: usize, seams: &[usize]) -> Option<usize> {
    seams.iter().map(|s| frame.abs_diff(*s)).min()
}

fn residual(a: &[f32], b: &[f32], seams: &[usize]) -> Value {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 2, 0);
    let mut peak = (0f32, 0usize);
    let mut signal = 0f64;
    let mut noise = 0f64;
    let mut seam_peaks = vec![0f32; seams.len()];
    let mut seam_counts = vec![0usize; seams.len()];
    let mut interior_peak = 0f32;
    let mut interior_count = 0usize;
    let mut total_above = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let err = (x - y).abs();
        if err > peak.0 {
            peak = (err, i);
        }
        signal += (y as f64).powi(2);
        noise += ((x - y) as f64).powi(2);
        if err > ORIGINAL_BOUND {
            total_above += 1;
        }
        let frame = i / 2;
        let mut in_seam = false;
        for (j, seam) in seams.iter().enumerate() {
            if frame >= seam.saturating_sub(SEAM_RADIUS) && frame < seam + SEAM_RADIUS {
                in_seam = true;
                seam_peaks[j] = seam_peaks[j].max(err);
                if err > ORIGINAL_BOUND {
                    seam_counts[j] += 1;
                }
            }
        }
        if !in_seam {
            interior_peak = interior_peak.max(err);
            if err > ORIGINAL_BOUND {
                interior_count += 1;
            }
        }
    }
    let seam_stats: Vec<_> = seams
        .iter()
        .enumerate()
        .map(|(i, seam)| {
            json!({
                "sampleFrame": seam,
                "maxAbs": seam_peaks[i],
                "countAboveOriginalBound": seam_counts[i],
            })
        })
        .collect();
    json!({
        "maxAbs": peak.0,
        "rms": (noise / a.len() as f64).sqrt(),
        "snrDb": 10.0 * (signal / noise.max(1e-30)).log10(),
        "argmax": {"sampleFrame": peak.1 / 2, "channel": peak.1 % 2,
            "distanceToNearestSeamFrames": nearest_seam_distance(peak.1 / 2, seams)},
        "originalBound": ORIGINAL_BOUND,
        "passesOriginalBound": peak.0 <= ORIGINAL_BOUND,
        "countAboveOriginalBound": total_above,
        "seams": seam_stats,
        "interior": {"maxAbs": interior_peak, "countAboveOriginalBound": interior_count},
    })
}

fn values(capture: &Value, key: &str) -> Vec<f32> {
    capture[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn describe_capture(capture: &Value) -> Value {
    json!({"rawSha256": capture["rawSha256"], "clampedSha256": capture["clampedSha256"],
        "rawDtype":capture["rawDtype"],
        "rawArtifact":capture["rawArtifact"], "clampedArtifact":capture["clampedArtifact"]})
}

fn pair(a: &Value, b: &Value, seams: &[usize]) -> Value {
    json!({
        "raw": residual(&values(a, "raw"), &values(b, "raw"), seams),
        "clamped": residual(&values(a, "clamped"), &values(b, "clamped"), seams),
    })
}

fn run() -> Result<(), Box<dyn Error>> {
    let engine = verified_engine()?;
    if std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref() != Some("0") {
        return Err("CUDA_VISIBLE_DEVICES must be exactly 0".into());
    }
    let (fixture_root, out) = arguments()?;
    let hub_root = required_env("YUE2_HF_HUB")?;
    let output_parent = out
        .parent()
        .ok_or("output directory has no parent")?
        .canonicalize()?;
    let harness = PathBuf::from(env!("CARGO_MANIFEST_DIR")).canonicalize()?;
    if output_parent.starts_with(&engine) || output_parent.starts_with(&harness) {
        return Err("diagnostic output must be outside engine and harness checkouts".into());
    }
    let meta_path =
        engine.join("crates/audio/candle-audio-yue2/tests/fixtures/vae_real_reference.json");
    let meta: Value = serde_json::from_slice(&fs::read(&meta_path)?)?;
    let fixture_bytes = fs::read(
        fixture_root.join(
            meta["reference_file"]
                .as_str()
                .ok_or("missing reference_file")?,
        ),
    )?;
    let reference_hash = sha256(&fixture_bytes);
    if reference_hash
        != meta["reference_sha256"]
            .as_str()
            .ok_or("missing reference_sha256")?
    {
        return Err("reference fixture SHA-256 mismatch".into());
    }
    let reference = candle_core::safetensors::load_buffer(&fixture_bytes, &Device::Cpu)?;
    let source = AcousticLatents::from_tensor(
        reference.get("long_latent").ok_or("missing long_latent")?,
        LatentSource::Synthesis {
            stage_identity: "precision_reference:long_latent".into(),
        },
    )?;
    let frames = meta["long_frames"].as_u64().ok_or("missing long_frames")? as usize;
    let core = meta["long_core_frames"]
        .as_u64()
        .ok_or("missing long_core_frames")? as usize;
    let halo = 16;
    if frames != 75 || core != 16 || source.frames() != frames {
        return Err("diagnostic requires pinned 75-frame/core-16 reference".into());
    }
    let repo_dir = hub_root
        .join(format!("models--{}", YUE2_VAE_REPO.id.replace('/', "--")))
        .join("snapshots")
        .join(YUE2_VAE_REPO.revision);
    let snapshots = SnapshotDirs::new().with(YUE2_VAE_REPO.id, repo_dir);
    let verified = resolve_component(ComponentId::VaeStandard, &snapshots)?;
    fs::create_dir(&out)?;
    let device = Device::new_cuda(0)?;
    let seams: Vec<usize> = (core..frames).step_by(core).map(|f| f * RATIO).collect();
    let pinned_reference = json!({
        "raw": interleaved(reference.get("standard.long_full_raw").ok_or("missing standard.long_full_raw")?, false)?,
        "clamped": interleaved(reference.get("standard.long_full_raw").unwrap(), true)?,
    });
    let mut runs = BTreeMap::new();
    for (label, dtype, repeats) in [("bf16", DType::BF16, 2), ("f32", DType::F32, 1)] {
        let vae = Yue2Vae::load_with_dtype(&verified, VaeParts::Full, &device, dtype)?;
        if vae.dtype() != dtype {
            return Err(format!("VAE reports {:?}, requested {dtype:?}", vae.dtype()).into());
        }
        if vae.required_halo(core) > halo {
            return Err("pinned halo no longer sufficient".into());
        }
        if vae.identity().weights_sha256 != meta["decoders"]["standard"]["weights_sha256"] {
            return Err("decoder identity differs from pinned fixture".into());
        }
        let mut full = Vec::new();
        let mut tiled = Vec::new();
        for repeat in 0..repeats {
            full.push(capture(
                &vae,
                &source,
                false,
                core,
                halo,
                &out,
                &format!("{label}-full-{}", repeat + 1),
            )?);
            tiled.push(capture(
                &vae,
                &source,
                true,
                core,
                halo,
                &out,
                &format!("{label}-tiled-{}", repeat + 1),
            )?);
        }
        let mut evidence = json!({
            "requestedPolicy": if dtype == DType::BF16 {"bf16"} else {"fp32"},
            "residentDtype": format!("{dtype:?}"),
            "decoderIdentity": vae.identity().to_json(),
            "requiredHalo": vae.required_halo(core),
            "full": full.iter().map(describe_capture).collect::<Vec<_>>(),
            "tiled": tiled.iter().map(describe_capture).collect::<Vec<_>>(),
            "tiledVsFull": pair(&tiled[0], &full[0], &seams),
            "fullVsPinnedFp32": pair(&full[0], &pinned_reference, &seams),
            "tiledVsPinnedFp32": pair(&tiled[0], &pinned_reference, &seams),
        });
        if repeats == 2 {
            evidence["fullRepeat"] = pair(&full[0], &full[1], &seams);
            evidence["tiledRepeat"] = pair(&tiled[0], &tiled[1], &seams);
        }
        runs.insert(label, evidence);
    }
    let report = json!({
        "schemaVersion": 1,
        "purpose": "diagnostic_only_no_gate_change",
        "engineSha": ENGINE_SHA,
        "referenceSha256": reference_hash,
        "referenceMetadataSha256": sha256(&fs::read(meta_path)?),
        "latentIdentity": source.identity().to_json(),
        "backend": "cuda",
        "deviceOrdinal": 0,
        "frames": frames,
        "coreFrames": core,
        "haloFrames": halo,
        "seamRadiusSampleFrames": SEAM_RADIUS,
        "seamsSampleFrames": seams,
        "originalBound": ORIGINAL_BOUND,
        "runs": runs,
    });
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("report.json"))?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("yue2 BF16 tile diagnostic: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seam_window_matches_failed_proofs_half_open_bounds() {
        let mut left = vec![0.0; 1024];
        let right = vec![0.0; 1024];
        left[(256 - SEAM_RADIUS) * 2] = 0.03125;
        left[(256 + SEAM_RADIUS) * 2] = 0.0625;
        let result = residual(&left, &right, &[256]);
        assert_eq!(result["seams"][0]["maxAbs"], 0.03125);
        assert_eq!(result["seams"][0]["countAboveOriginalBound"], 1);
        assert_eq!(result["interior"]["maxAbs"], 0.0625);
        assert_eq!(result["interior"]["countAboveOriginalBound"], 1);
    }

    #[test]
    fn seam_and_interior_are_distinguished() {
        let mut a = vec![0f32; 800];
        let b = vec![0f32; 800];
        a[400] = 0.03125; // frame 200: inside seam window
        a[2] = 0.02; // frame 1: interior
        let result = residual(&a, &b, &[200]);
        assert_eq!(result["maxAbs"], 0.03125);
        assert_eq!(result["argmax"]["sampleFrame"], 200);
        assert_eq!(result["countAboveOriginalBound"], 2);
        assert_eq!(result["seams"][0]["countAboveOriginalBound"], 1);
        assert_eq!(result["interior"]["countAboveOriginalBound"], 1);
        assert_eq!(result["passesOriginalBound"], false);
    }

    #[test]
    fn equality_and_threshold_are_recorded_without_regrading() {
        let a = [0.0, ORIGINAL_BOUND, 0.0, 0.0];
        let b = [0.0; 4];
        let result = residual(&a, &b, &[]);
        assert_eq!(result["maxAbs"], ORIGINAL_BOUND);
        assert_eq!(result["countAboveOriginalBound"], 0);
        assert_eq!(result["passesOriginalBound"], true);
        assert_eq!(result["argmax"]["distanceToNearestSeamFrames"], Value::Null);
    }
}
