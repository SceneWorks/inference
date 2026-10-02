//! Same-input waveform or first-Conv7 numerical diagnostic for the frozen M3 YuE2 standard VAE.
//! This reports the original 1/64 bound; it does not change a production gate.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(feature = "cuda")]
use candle_audio::candle_core::cuda::cudarc::cublas::{self, sys};
use candle_audio::candle_core::{self, DType, Device, Tensor};
use candle_audio::neural_codec::fold_weight_norm;
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
const FIRST_CONV_KERNEL: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Diagnostic {
    Waveform,
    FirstConv,
    FirstConvMath,
}

impl Diagnostic {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "waveform" => Ok(Self::Waveform),
            "first_conv" => Ok(Self::FirstConv),
            "first_conv_math" => Ok(Self::FirstConvMath),
            _ => Err(format!("unsupported diagnostic {value:?}").into()),
        }
    }
}

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

fn arguments() -> Result<(Diagnostic, PathBuf, PathBuf), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let (mut diagnostic, mut reference, mut output) = (None, None, None);
    while let Some(key) = args.next() {
        let value = PathBuf::from(args.next().ok_or("option requires a path")?);
        match key.to_str() {
            Some("--diagnostic") if diagnostic.is_none() => {
                diagnostic = Some(Diagnostic::parse(
                    value.to_str().ok_or("diagnostic is not UTF-8")?,
                )?)
            }
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
    Ok((
        diagnostic.unwrap_or(Diagnostic::Waveform),
        reference,
        output,
    ))
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

struct FirstTensor {
    record: Value,
    values: Vec<f32>,
    channels: usize,
    length: usize,
}

fn first_tensor(
    tensor: &Tensor,
    dtype: DType,
    out: &Path,
    name: &str,
) -> Result<FirstTensor, Box<dyn Error>> {
    let [1, channels, length] = tensor.dims() else {
        return Err(format!("{name} has unexpected shape {:?}", tensor.dims()).into());
    };
    if tensor.dtype() != dtype || *channels == 0 || *length == 0 {
        return Err(format!(
            "{name} has unexpected dtype/shape {:?}/{:?}",
            tensor.dtype(),
            tensor.dims()
        )
        .into());
    }
    // BF16 to F32 is lossless; F32LE retains every original BF16 value for independent analysis.
    let values = tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} contains non-finite values").into());
    }
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in &values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let file = format!("{name}.f32le");
    let mut handle = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join(&file))?;
    handle.write_all(&bytes)?;
    Ok(FirstTensor {
        record: json!({"file":file,"sha256":sha256(&bytes),"bytes":bytes.len(),
            "layout":"bct_f32le","shape":[1,channels,length],"dtype":format!("{dtype:?}")}),
        values,
        channels: *channels,
        length: *length,
    })
}

fn first_conv_capture(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    dtype: DType,
    out: &Path,
    stem: &str,
) -> Result<[FirstTensor; 3], Box<dyn Error>> {
    if input.dtype() != dtype || weight.dtype() != dtype || bias.dtype() != dtype {
        return Err("first Conv7 operand dtype mismatch".into());
    }
    let input_record = first_tensor(input, dtype, out, &format!("{stem}-input"))?;
    // Exactly M3 Conv::forward: k7/p3/s1/d1/groups1, then BF16/F32 bias addition.
    let pre = input.conv1d(weight, 3, 1, 1, 1)?;
    let pre_record = first_tensor(&pre, dtype, out, &format!("{stem}-pre-bias"))?;
    let post = pre.broadcast_add(&bias.reshape((1, (), 1))?)?;
    let post_record = first_tensor(&post, dtype, out, &format!("{stem}-post-bias"))?;
    Ok([input_record, pre_record, post_record])
}

fn compare_first_core(
    full: &FirstTensor,
    tile: &FirstTensor,
    start: usize,
    left: usize,
    core_len: usize,
) -> Value {
    assert_eq!(full.channels, tile.channels);
    assert!(start + core_len <= full.length);
    assert!(start >= left && start - left + core_len <= tile.length);
    let mut maximum = 0f32;
    let mut different = 0usize;
    let mut first = Value::Null;
    for channel in 0..full.channels {
        for offset in 0..core_len {
            let full_value = full.values[channel * full.length + start + offset];
            let tile_value = tile.values[channel * tile.length + start - left + offset];
            let error = (full_value - tile_value).abs();
            maximum = maximum.max(error);
            if full_value.to_bits() != tile_value.to_bits() {
                different += 1;
                if first.is_null() {
                    first = json!({"channel":channel,"globalLatentFrame":start+offset,
                        "fullValue":full_value,"tileValue":tile_value,"absError":error});
                }
            }
        }
    }
    json!({"maxAbs":maximum,"differentValues":different,"firstDifferent":first,
        "comparedValues":full.channels*core_len})
}

fn first_conv_weights(
    verified: &candle_audio_yue2::snapshot::VerifiedComponent,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, Value), Box<dyn Error>> {
    let path = verified
        .weights_path()
        .ok_or("verified standard VAE weights absent")?;
    // The source is integrity-checked immediately above; use the same mapped loader as M3.
    let source = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }?;
    let prefix = "decoder.layers.0";
    let v = source.load(&format!("{prefix}.weight_v"), device)?;
    let g = source.load(&format!("{prefix}.weight_g"), device)?;
    let b = source.load(&format!("{prefix}.bias"), device)?;
    let [channels, 64, FIRST_CONV_KERNEL] = v.dims() else {
        return Err(format!("first Conv7 weight_v shape {:?}", v.dims()).into());
    };
    if v.dtype() != DType::F32
        || g.dtype() != DType::F32
        || b.dtype() != DType::F32
        || g.dims() != [*channels, 1, 1]
        || b.dims() != [*channels]
    {
        return Err("first Conv7 source tensor dtype/shape mismatch".into());
    }
    let weight = fold_weight_norm(&v, &g)?.contiguous()?.to_dtype(dtype)?;
    let bias = b.to_dtype(dtype)?;
    let tensor_hash = |tensor: &Tensor| -> Result<String, Box<dyn Error>> {
        let values = tensor
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(samples_sha256(&values))
    };
    let identity = json!({"sourceDtype":"F32","foldDtype":"F32",
        "residentDtype":format!("{dtype:?}"),"weightShape":weight.dims(),
        "biasShape":bias.dims(),"weightF32LeSha256":tensor_hash(&weight)?,
        "biasF32LeSha256":tensor_hash(&bias)?});
    Ok((weight, bias, identity))
}

#[allow(clippy::too_many_arguments)]
fn run_first_conv(
    engine: &Path,
    out: &Path,
    source: &AcousticLatents,
    verified: &candle_audio_yue2::snapshot::VerifiedComponent,
    device: &Device,
    meta: &Value,
    reference_hash: &str,
    frames: usize,
    core: usize,
    halo: usize,
) -> Result<(), Box<dyn Error>> {
    let standard = &meta["decoders"]["standard"];
    let component = verified.component();
    if component.repo.id != standard["repo"]
        || component.repo.revision != standard["revision"]
        || component.weights().map(|file| file.sha256) != standard["weights_sha256"].as_str()
        || component.file("config.json").map(|file| file.sha256)
            != standard["config_sha256"].as_str()
    {
        return Err("first Conv7 component differs from pinned standard VAE reference".into());
    }
    let mut runs = BTreeMap::new();
    for (label, dtype) in [("bf16", DType::BF16), ("f32", DType::F32)] {
        let (weight, bias, resident) = first_conv_weights(verified, device, dtype)?;
        let z = source.to_decoder_input(device)?.to_dtype(dtype)?;
        let full = first_conv_capture(&z, &weight, &bias, dtype, out, &format!("{label}-full"))?;
        let mut windows = Vec::new();
        for start in (0..frames).step_by(core) {
            let end = frames.min(start + core);
            let left = start.saturating_sub(halo);
            let right = frames.min(end + halo);
            let tile_input = z.narrow(2, left, right - left)?;
            let tile = first_conv_capture(
                &tile_input,
                &weight,
                &bias,
                dtype,
                out,
                &format!("{label}-tile-{}", start / core),
            )?;
            let input = compare_first_core(&full[0], &tile[0], start, left, end - start);
            if input["differentValues"] != 0 {
                return Err(format!("{label} aligned input core differs at start {start}").into());
            }
            windows.push(json!({"start":start,"end":end,"left":left,"right":right,
                "coreLength":end-start,"captures":{
                    "input":tile[0].record,"preBias":tile[1].record,"postBias":tile[2].record},
                "alignedCore":{
                    "input":input,
                    "preBias":compare_first_core(&full[1],&tile[1],start,left,end-start),
                    "postBias":compare_first_core(&full[2],&tile[2],start,left,end-start)}}));
        }
        runs.insert(
            label,
            json!({"resident":resident,"full":{
            "input":full[0].record,"preBias":full[1].record,"postBias":full[2].record},
            "windows":windows}),
        );
    }
    let metadata =
        engine.join("crates/audio/candle-audio-yue2/tests/fixtures/vae_real_reference.json");
    let report = json!({"schemaVersion":2,"selector":"first_conv",
        "purpose":"diagnostic_only_no_gate_change","engineSha":ENGINE_SHA,
        "referenceSha256":reference_hash,"referenceMetadataSha256":sha256(&fs::read(metadata)?),
        "latentIdentity":source.identity().to_json(),"backend":"cuda","deviceOrdinal":0,
        "decoderIdentity":{"repo":YUE2_VAE_REPO.id,"revision":YUE2_VAE_REPO.revision,
            "weights_sha256":meta["decoders"]["standard"]["weights_sha256"],
            "config_sha256":meta["decoders"]["standard"]["config_sha256"]},
        "frames":frames,"coreFrames":core,"haloFrames":halo,
        "operator":{"name":"decoder.layers.0.Conv1d","kernel":FIRST_CONV_KERNEL,
            "padding":3,"stride":1,"dilation":1,"groups":1},
        "originalWaveformObservation":{"runId":"36884387320","clampedMaxAbs":0.03125,
            "originalBound":ORIGINAL_BOUND,"interpretation":"prior_failed_waveform_proof_not_a_first_conv_gate"},
        "runs":runs});
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("report.json"))?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

#[cfg(feature = "cuda")]
struct FirstArm {
    record: Value,
    full: [FirstTensor; 3],
    windows: Vec<[FirstTensor; 3]>,
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn capture_first_arm(
    z: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    dtype: DType,
    out: &Path,
    label: &str,
    resident: Value,
    frames: usize,
    core: usize,
    halo: usize,
) -> Result<FirstArm, Box<dyn Error>> {
    let full = first_conv_capture(z, weight, bias, dtype, out, &format!("{label}-full"))?;
    let mut records = Vec::new();
    let mut windows = Vec::new();
    for start in (0..frames).step_by(core) {
        let end = frames.min(start + core);
        let left = start.saturating_sub(halo);
        let right = frames.min(end + halo);
        let tile = first_conv_capture(
            &z.narrow(2, left, right - left)?,
            weight,
            bias,
            dtype,
            out,
            &format!("{label}-tile-{}", start / core),
        )?;
        records.push(json!({"start":start,"end":end,"left":left,"right":right,
            "coreLength":end-start,"captures":{
                "input":tile[0].record.clone(),"preBias":tile[1].record.clone(),"postBias":tile[2].record.clone()},
            "alignedCore":{
                "input":compare_first_core(&full[0],&tile[0],start,left,end-start),
                "preBias":compare_first_core(&full[1],&tile[1],start,left,end-start),
                "postBias":compare_first_core(&full[2],&tile[2],start,left,end-start)}}));
        windows.push(tile);
    }
    let record = json!({"resident":resident,"full":{
        "input":full[0].record.clone(),"preBias":full[1].record.clone(),"postBias":full[2].record.clone()},
        "windows":records});
    Ok(FirstArm {
        record,
        full,
        windows,
    })
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MathGate {
    InputCoreMismatch,
    NoPositivePreBiasResidual,
    Applicable,
}

#[cfg(feature = "cuda")]
impl MathGate {
    fn as_str(self) -> &'static str {
        match self {
            Self::InputCoreMismatch => "input_core_mismatch",
            Self::NoPositivePreBiasResidual => "no_positive_pre_bias_residual",
            Self::Applicable => "positive_pre_bias_residual",
        }
    }
}

#[cfg(any(feature = "cuda", test))]
fn math_gate(windows: &[Value]) -> MathGate {
    if windows
        .iter()
        .any(|window| window["alignedCore"]["input"]["differentValues"] != 0)
    {
        return MathGate::InputCoreMismatch;
    }
    if windows.iter().any(|window| {
        window["alignedCore"]["preBias"]["differentValues"]
            .as_u64()
            .is_some_and(|count| count > 0)
            && window["alignedCore"]["preBias"]["maxAbs"]
                .as_f64()
                .is_some_and(|maximum| maximum.is_finite() && maximum > 0.0)
    }) {
        MathGate::Applicable
    } else {
        MathGate::NoPositivePreBiasResidual
    }
}

#[cfg(any(feature = "cuda", test))]
fn compare_first_arrays(a: &FirstTensor, b: &FirstTensor, origin: usize) -> Value {
    assert_eq!((a.channels, a.length), (b.channels, b.length));
    let mut maximum = 0f32;
    let mut different = 0usize;
    let mut first = Value::Null;
    for channel in 0..a.channels {
        for frame in 0..a.length {
            let a_value = a.values[channel * a.length + frame];
            let b_value = b.values[channel * b.length + frame];
            let error = (a_value - b_value).abs();
            maximum = maximum.max(error);
            if a_value.to_bits() != b_value.to_bits() {
                different += 1;
                if first.is_null() {
                    first = json!({"channel":channel,"globalLatentFrame":origin+frame,
                        "aValue":a_value,"bValue":b_value,"absError":error});
                }
            }
        }
    }
    json!({"maxAbs":maximum,"differentValues":different,"firstDifferent":first,
        "comparedValues":a.channels*a.length})
}

#[cfg(feature = "cuda")]
fn compare_first_arms(default: &FirstArm, flagged: &FirstArm) -> Value {
    assert_eq!(default.windows.len(), flagged.windows.len());
    let comparison = |a: &[FirstTensor; 3], b: &[FirstTensor; 3], origin| {
        json!({"input":compare_first_arrays(&a[0],&b[0],origin),
            "preBias":compare_first_arrays(&a[1],&b[1],origin),
            "postBias":compare_first_arrays(&a[2],&b[2],origin)})
    };
    json!({"full":comparison(&default.full,&flagged.full,0),
        "windows":default.windows.iter().zip(&flagged.windows).enumerate()
            .map(|(index,(a,b))| {
                let left = default.record["windows"][index]["left"].as_u64().expect("pinned window left");
                comparison(a,b,left as usize)
            })
            .collect::<Vec<_>>()})
}

#[cfg(feature = "cuda")]
struct MathModeGuard {
    device: Device,
    blas: std::sync::Arc<cublas::CudaBlas>,
    events: std::fs::File,
    active: bool,
}

#[cfg(feature = "cuda")]
impl MathModeGuard {
    fn new(device: &Device, out: &Path) -> Result<Self, Box<dyn Error>> {
        let mut guard = Self {
            device: device.clone(),
            blas: device.as_cuda_device()?.cublas_handle(),
            events: OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out.join("math-mode-events.jsonl"))?,
            active: false,
        };
        if guard.get("before_default_arm")? != 0 {
            return Err("cuBLAS handle did not start in CUBLAS_DEFAULT_MATH".into());
        }
        Ok(guard)
    }

    fn event(
        &mut self,
        action: &str,
        status: sys::cublasStatus_t,
        raw: Option<u32>,
    ) -> Result<(), Box<dyn Error>> {
        writeln!(
            self.events,
            "{}",
            json!({"action":action,"status":format!("{status:?}"),"rawMode":raw})
        )?;
        self.events.flush()?;
        Ok(())
    }

    fn get(&mut self, action: &str) -> Result<u32, Box<dyn Error>> {
        // cublasMath_t is repr(u32), but combinations need not be declared enum variants.
        let mut raw = u32::MAX;
        let status =
            unsafe { sys::cublasGetMathMode(*self.blas.handle(), (&mut raw as *mut u32).cast()) };
        self.event(
            action,
            status,
            (status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS).then_some(raw),
        )?;
        status.result()?;
        Ok(raw)
    }

    fn set(&mut self, action: &str, mode: sys::cublasMath_t) -> Result<(), Box<dyn Error>> {
        let status = unsafe { sys::cublasSetMathMode(*self.blas.handle(), mode) };
        self.event(action, status, Some(mode as u32))?;
        status.result()?;
        Ok(())
    }

    fn enable(&mut self) -> Result<(), Box<dyn Error>> {
        self.device.synchronize()?;
        if self.get("before_flagged_arm")? != 0 {
            return Err("cuBLAS math mode changed before the controlled arm".into());
        }
        // Arm restoration before attempting the setter: even a failing setter may alter state.
        self.active = true;
        self.set(
            "set_disallow",
            sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
        )?;
        if self.get("read_disallow")? != 16 {
            return Err("cuBLAS disallow-reduction mode readback differed from 16".into());
        }
        Ok(())
    }

    fn restore(&mut self) -> Result<(), Box<dyn Error>> {
        if !self.active {
            return Ok(());
        }
        self.device.synchronize()?;
        self.set("restore_default", sys::cublasMath_t::CUBLAS_DEFAULT_MATH)?;
        if self.get("read_restored")? != 0 {
            return Err("cuBLAS default math mode restoration readback failed".into());
        }
        self.active = false;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for MathModeGuard {
    fn drop(&mut self) {
        if self.active && self.restore().is_err() {
            eprintln!("cuBLAS math-mode restoration failed; aborting owned diagnostic child");
            std::process::abort();
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_first_conv_math(
    engine: &Path,
    out: &Path,
    source: &AcousticLatents,
    verified: &candle_audio_yue2::snapshot::VerifiedComponent,
    device: &Device,
    meta: &Value,
    reference_hash: &str,
    frames: usize,
    core: usize,
    halo: usize,
) -> Result<(), Box<dyn Error>> {
    let standard = &meta["decoders"]["standard"];
    let component = verified.component();
    if component.repo.id != standard["repo"]
        || component.repo.revision != standard["revision"]
        || component.weights().map(|file| file.sha256) != standard["weights_sha256"].as_str()
        || component.file("config.json").map(|file| file.sha256)
            != standard["config_sha256"].as_str()
    {
        return Err("first Conv7 component differs from pinned standard VAE reference".into());
    }
    let mut mode = MathModeGuard::new(device, out)?;
    let (bf16_weight, bf16_bias, bf16_resident) =
        first_conv_weights(verified, device, DType::BF16)?;
    let z_bf16 = source.to_decoder_input(device)?.to_dtype(DType::BF16)?;
    let default = capture_first_arm(
        &z_bf16,
        &bf16_weight,
        &bf16_bias,
        DType::BF16,
        out,
        "bf16",
        bf16_resident,
        frames,
        core,
        halo,
    )?;
    let (f32_weight, f32_bias, f32_resident) = first_conv_weights(verified, device, DType::F32)?;
    let z_f32 = source.to_decoder_input(device)?.to_dtype(DType::F32)?;
    let f32_arm = capture_first_arm(
        &z_f32,
        &f32_weight,
        &f32_bias,
        DType::F32,
        out,
        "f32",
        f32_resident,
        frames,
        core,
        halo,
    )?;
    let gate = math_gate(
        default.record["windows"]
            .as_array()
            .ok_or("windows absent")?,
    );
    let (status, flagged, cross_arm) = if gate == MathGate::Applicable {
        mode.enable()?;
        let result = capture_first_arm(
            &z_bf16,
            &bf16_weight,
            &bf16_bias,
            DType::BF16,
            out,
            "bf16-disallow",
            default.record["resident"].clone(),
            frames,
            core,
            halo,
        );
        mode.restore()?;
        let flagged = result?;
        let cross = compare_first_arms(&default, &flagged);
        ("collected", Some(flagged.record), Some(cross))
    } else {
        ("not_applicable", None, None)
    };
    mode.events.flush()?;
    let mode_events = fs::read(out.join("math-mode-events.jsonl"))?;
    let metadata =
        engine.join("crates/audio/candle-audio-yue2/tests/fixtures/vae_real_reference.json");
    let report = json!({"schemaVersion":3,"selector":"first_conv_math",
        "purpose":"controlled_diagnostic_only_no_gate_change","engineSha":ENGINE_SHA,
        "referenceSha256":reference_hash,"referenceMetadataSha256":sha256(&fs::read(metadata)?),
        "latentIdentity":source.identity().to_json(),"backend":"cuda","deviceOrdinal":0,
        "decoderIdentity":{"repo":YUE2_VAE_REPO.id,"revision":YUE2_VAE_REPO.revision,
            "weights_sha256":standard["weights_sha256"],"config_sha256":standard["config_sha256"]},
        "frames":frames,"coreFrames":core,"haloFrames":halo,
        "operator":{"name":"decoder.layers.0.Conv1d","kernel":FIRST_CONV_KERNEL,
            "padding":3,"stride":1,"dilation":1,"groups":1},
        "originalWaveformObservation":{"runId":"36884387320","clampedMaxAbs":0.03125,
            "originalBound":ORIGINAL_BOUND,"interpretation":"prior_failed_waveform_proof_not_a_first_conv_gate"},
        "runs":{"bf16":default.record,"f32":f32_arm.record},
        "controlled":{"status":status,"reason":gate.as_str(),"flagged":flagged,
            "crossArm":cross_arm,"modeEventsFile":"math-mode-events.jsonl",
            "modeEventsSha256":sha256(&mode_events)}});
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("report.json"))?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn run_first_conv_math(
    _engine: &Path,
    _out: &Path,
    _source: &AcousticLatents,
    _verified: &candle_audio_yue2::snapshot::VerifiedComponent,
    _device: &Device,
    _meta: &Value,
    _reference_hash: &str,
    _frames: usize,
    _core: usize,
    _halo: usize,
) -> Result<(), Box<dyn Error>> {
    Err("first_conv_math requires the pinned CUDA build".into())
}

fn run() -> Result<(), Box<dyn Error>> {
    let engine = verified_engine()?;
    if std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref() != Some("0") {
        return Err("CUDA_VISIBLE_DEVICES must be exactly 0".into());
    }
    let (diagnostic, fixture_root, out) = arguments()?;
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
    if diagnostic == Diagnostic::FirstConv {
        return run_first_conv(
            &engine,
            &out,
            &source,
            &verified,
            &device,
            &meta,
            &reference_hash,
            frames,
            core,
            halo,
        );
    }
    if diagnostic == Diagnostic::FirstConvMath {
        return run_first_conv_math(
            &engine,
            &out,
            &source,
            &verified,
            &device,
            &meta,
            &reference_hash,
            frames,
            core,
            halo,
        );
    }
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
    fn selector_is_typed_and_waveform_stays_default() {
        assert_eq!(Diagnostic::parse("waveform").unwrap(), Diagnostic::Waveform);
        assert_eq!(
            Diagnostic::parse("first_conv").unwrap(),
            Diagnostic::FirstConv
        );
        assert_eq!(
            Diagnostic::parse("first_conv_math").unwrap(),
            Diagnostic::FirstConvMath
        );
        assert!(Diagnostic::parse("full_vae").is_err());
    }

    #[test]
    fn math_gate_needs_identical_inputs_and_positive_pre_bias_error() {
        let mut windows = vec![json!({"alignedCore":{
            "input":{"differentValues":0},
            "preBias":{"differentValues":1,"maxAbs":0.0}}})];
        assert_eq!(math_gate(&windows), MathGate::NoPositivePreBiasResidual);
        windows[0]["alignedCore"]["preBias"]["maxAbs"] = json!(0.03125);
        assert_eq!(math_gate(&windows), MathGate::Applicable);
        windows[0]["alignedCore"]["input"]["differentValues"] = json!(1);
        assert_eq!(math_gate(&windows), MathGate::InputCoreMismatch);
    }

    #[test]
    fn math_cross_arm_comparison_locates_changed_value() {
        let a = FirstTensor {
            record: Value::Null,
            values: vec![0., 0., 0., 0.],
            channels: 1,
            length: 4,
        };
        let mut b = FirstTensor {
            record: Value::Null,
            values: a.values.clone(),
            channels: 1,
            length: 4,
        };
        b.values[2] = 0.03125;
        let compared = compare_first_arrays(&a, &b, 16);
        assert_eq!(compared["differentValues"], 1);
        assert_eq!(compared["firstDifferent"]["globalLatentFrame"], 18);
        assert_eq!(compared["maxAbs"], 0.03125);
    }

    #[test]
    fn aligned_core_comparison_identifies_first_channel_and_global_frame() {
        let full = FirstTensor {
            record: Value::Null,
            values: vec![
                0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13., 14., 15., 16., 17.,
            ],
            channels: 2,
            length: 8,
        };
        // Window starts at global frame 2. Core covers global frames 3 and 4.
        let mut tile = FirstTensor {
            record: Value::Null,
            values: vec![2., 3., 4., 5., 6., 12., 13., 14., 15., 16.],
            channels: 2,
            length: 5,
        };
        let same = compare_first_core(&full, &tile, 3, 2, 2);
        assert_eq!(same["differentValues"], 0);
        tile.values[tile.length + 2] += 0.03125;
        let changed = compare_first_core(&full, &tile, 3, 2, 2);
        assert_eq!(changed["differentValues"], 1);
        assert_eq!(changed["firstDifferent"]["channel"], 1);
        assert_eq!(changed["firstDifferent"]["globalLatentFrame"], 4);
        assert_eq!(changed["maxAbs"], 0.03125);
    }

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
